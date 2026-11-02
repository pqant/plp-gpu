// plp_gpu.cu — CUDA build of the 3D-PLP decoder (gate 3.3, Spark GB10 sm_121).
//
// Same semantics as plp_gpu.cpp; the ONLY difference is the feasibility
// backend: CudaBackend parallelizes check_feasible_one on the GPU (one thread
// per (candidate, orientation) pair). Everything else (decoder, FSR, JSON,
// PyRng, CLI) is the shared single source plp_common.hpp. The CLI block
// below mirrors plp_gpu.cpp — keep in sync if it changes.
//
// Parity gate: with --single-pass --eps 0 --no-theta the JSON output must be
// bit-identical to the CPython CPU reference (parity/ref_cpu/*.json),
// time_s excluded (machine-dependent).
//
// Build:  make cuda   (see gpu/Makefile)
//         nvcc -O3 -std=c++17 -arch=sm_121 --fmad=false
//              -Xcompiler -ffp-contract=off
// NEVER add -use_fast_math or any fast-math flag: bit parity requires exact
// IEEE-754 semantics (device FMA off, host contraction off).

#include "plp_common.hpp"

#include <cuda_runtime.h>

namespace {

// Device mirrors of plp::Pt (std::array<int,3>) and plp::FastBox — identical
// standard-layout memory layout, so host vectors memcpy directly.
struct DPoint { int x, y, z; };
struct DBox   { int x, y, z, x2, y2, z2; };
static_assert(sizeof(DPoint) == 12, "layout must match std::array<int,3>");
static_assert(sizeof(DBox) == 24, "layout must match FastBox");

__device__ inline long long dminll(long long a, long long b) { return a < b ? a : b; }
__device__ inline long long dmaxll(long long a, long long b) { return a < b ? b : a; }

// Line-by-line __device__ copy of plp::check_feasible_one (plp_common.hpp):
// C2 bounds -> C3 support (integer areas, ONE correctly-rounded double
// division) -> C1 overlap. Same int/long-long casts as the host version, so
// results are bit-identical.
__device__ bool check_feasible_one_dev(int x, int y, int z, const DPoint& o,
                                       const DBox* fast, int n_fast,
                                       int L, int W, int H) {
    int w = o.x, d = o.y, h = o.z;
    if (x + w > L || y + d > W || z + h > H) return false;          // C2
    if (z > 0) {
        long long s = 0;
        for (int i = 0; i < n_fast; i++) {
            const DBox& b = fast[i];
            if (b.z2 == z) {
                long long ox = dminll((long long)x + w, (long long)b.x2)
                               - (long long)dmaxll(x, b.x);
                if (ox > 0) {
                    long long oy = dminll((long long)y + d, (long long)b.y2)
                                   - (long long)dmaxll(y, b.y);
                    if (oy > 0) s += ox * oy;
                }
            }
        }
        // Python: s / (w * d) < TAU  (int/int -> one correctly-rounded double)
        if ((double)s / (double)(w * d) < plp::TAU) return false;   // C3
    }
    for (int i = 0; i < n_fast; i++) {
        const DBox& b = fast[i];
        if (x < b.x2 && b.x < x + w && y < b.y2 && b.y < y + d
            && z < b.z2 && b.z < z + h) return false;               // C1
    }
    return true;
}

// Grid-stride: one thread per (candidate, orientation) pair, candidate-major
// x orientation-minor — the same order as the `feas` list in decode_pass, so
// bits[t] lands at index t = ci * n_ors + oi exactly like the host backend.
// Pairs are independent (each reads only the fast-box array), so any
// thread scheduling yields identical bits.
__global__ void feas_kernel(const DPoint* __restrict__ cand, int n_cand,
                            const DPoint* __restrict__ ors, int n_ors,
                            const DBox* __restrict__ fast, int n_fast,
                            int L, int W, int H,
                            uint8_t* __restrict__ bits) {
    const int total = n_cand * n_ors;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += gridDim.x * blockDim.x) {
        int ci = t / n_ors;
        int oi = t - ci * n_ors;
        bits[t] = check_feasible_one_dev(cand[ci].x, cand[ci].y, cand[ci].z,
                                         ors[oi], fast, n_fast, L, W, H)
                      ? 1 : 0;
    }
}

// ===========================================================================
// Wave 2a — population batching (Opsiyon A: independent streams).
//
// K independent rows, row k seeded with PyRng(seed + k), decode the same
// instance in lockstep. Row k's trajectory is bit-identical to a standalone
// run with that seed (verified per row against the CPU oracle; K=1 with seed
// 0 is the frozen reference). Per step (item) all K rows are at the same
// item -> same ors table and L/W/H; only cand/fast differ per row. The whole
// step then costs ONE H2D (all rows packed in one staging buffer) + ONE
// kernel + ONE sync + ONE D2H, instead of K round trips.
//
// The frozen plp_common.hpp is used as-is (PyRng, orientations,
// item_sequence, sort_dedup_pt, stability_score); decode_pass_batch/run_batch
// below are line-by-line mirrors of decode_pass/run restructured for K rows.
// ===========================================================================

constexpr int K_MAX = 16;

// Per-row layout of the batched staging buffer, passed to the kernel by value.
struct BatchRowArgs {
    int K;
    int n_ors, L, W, H;
    int cand_off[K_MAX];    // DPoint offset of row k's cand in the staging
    int fast_off[K_MAX];    // DBox offset of row k's fast in the fast region
    int n_cand[K_MAX];
    int n_fast[K_MAX];
    int pair_start[K_MAX];  // cumulative (row, cand, ors)-pair offset
};

// K-row generalization of feas_kernel: one thread per (row, candidate,
// orientation) triple, row-major; writes bits at the global pair index so the
// host can split per row. Row lookup is a linear scan (K <= 16, negligible
// next to the fast-box loop).
__global__ void feas_batch_kernel(const DPoint* __restrict__ stage,
                                  const DBox* __restrict__ fast_base,
                                  const DPoint* __restrict__ ors,
                                  uint8_t* __restrict__ bits,
                                  BatchRowArgs a) {
    const int n_ors = a.n_ors;
    const int total = a.pair_start[a.K - 1] + a.n_cand[a.K - 1] * n_ors;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += gridDim.x * blockDim.x) {
        int k = 0;
        for (int r = 1; r < a.K; r++)
            if (t >= a.pair_start[r]) k = r;
        const int local = t - a.pair_start[k];
        const int ci = local / n_ors;
        const int oi = local - ci * n_ors;
        const DPoint& c = stage[a.cand_off[k] + ci];
        bits[t] = check_feasible_one_dev(c.x, c.y, c.z, ors[oi],
                                         fast_base + a.fast_off[k],
                                         a.n_fast[k], a.L, a.W, a.H)
                      ? 1 : 0;
    }
}

// Batched feasibility backend: K rows in lockstep (same ors/L/W/H per step).
// Staging layout: [cand_0..cand_{K-1} | 24B-pad | fast_0..fast_{K-1}].
struct CudaBatchBackend {
    unsigned char* d_stage = nullptr;
    int* d_ors = nullptr;
    uint8_t* d_bits = nullptr;
    size_t cap_stage = 0, cap_ors = 0, cap_bits = 0;
    // Host mirror of the last ors table uploaded to d_ors (content cache).
    std::vector<DPoint> ors_cache_host;
    // Host staging mirror: [cand rows | 24B-pad | fast rows] for one H2D.
    std::vector<unsigned char> staging_host;

    static void check(const char* what, cudaError_t e) {
        if (e == cudaSuccess) return;
        std::fprintf(stderr, "cuda %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }

    template <typename T>
    static void ensure(T** ptr, size_t* cap, size_t need, const char* what) {
        if (need <= *cap) return;
        size_t nc = *cap ? *cap * 2 : 4096;
        if (nc < need) nc = need;
        if (*ptr) { cudaFree(*ptr); *ptr = nullptr; *cap = 0; }
        check(what, cudaMalloc(reinterpret_cast<void**>(ptr), nc * sizeof(T)));
        *cap = nc;
    }

    std::vector<std::vector<uint8_t>> check_all_batch(
            const std::vector<const std::vector<plp::Pt>*>& cands,
            const std::vector<const std::vector<plp::FastBox>*>& fasts,
            const std::vector<plp::Pt>& ors, int L, int W, int H) {
        const int K = (int)cands.size();
        std::vector<int> cand_off(K), fast_off(K), n_cand(K), n_fast(K);
        std::vector<int> pair_start(K, 0);
        size_t total_cand = 0, total_fast = 0, pairs = 0;
        for (int k = 0; k < K; k++) {
            n_cand[k] = (int)cands[k]->size();
            n_fast[k] = (int)fasts[k]->size();
            cand_off[k] = (int)total_cand;
            fast_off[k] = (int)total_fast;
            pair_start[k] = (int)pairs;
            total_cand += (size_t)n_cand[k];
            total_fast += (size_t)n_fast[k];
            pairs += (size_t)n_cand[k] * ors.size();
        }
        std::vector<std::vector<uint8_t>> out(K);
        for (int k = 0; k < K; k++)
            out[k].assign((size_t)n_cand[k] * ors.size(), 0);
        if (pairs == 0) return out;
        ensure(&d_bits, &cap_bits, pairs, "bits malloc");
        // ors is shared across all rows in a step; content-cached exactly as
        // in CudaBackend (same 8 distinct tables per pass for BR3_050).
        bool ors_hit = ors_cache_host.size() == ors.size() &&
                       std::memcmp(ors.data(), ors_cache_host.data(),
                                   ors.size() * sizeof(DPoint)) == 0;
        if (!ors_hit) {
            ensure(&d_ors, &cap_ors, ors.size(), "ors malloc");
            check("h2d ors", cudaMemcpy(d_ors, ors.data(),
                                       ors.size() * sizeof(DPoint),
                                       cudaMemcpyHostToDevice));
            const DPoint* src = reinterpret_cast<const DPoint*>(ors.data());
            ors_cache_host.assign(src, src + ors.size());
        }
        const size_t cand_bytes = total_cand * sizeof(DPoint);
        const size_t pad = (24 - cand_bytes % 24) % 24;
        const size_t stage_bytes = cand_bytes + pad + total_fast * sizeof(DBox);
        ensure(&d_stage, &cap_stage, stage_bytes, "stage malloc");
        staging_host.resize(stage_bytes);
        unsigned char* base = staging_host.data();
        for (int k = 0; k < K; k++) {
            if (n_cand[k])
                std::memcpy(base + (size_t)cand_off[k] * sizeof(DPoint),
                            cands[k]->data(), (size_t)n_cand[k] * sizeof(DPoint));
            if (n_fast[k])
                std::memcpy(base + cand_bytes + pad + (size_t)fast_off[k] * sizeof(DBox),
                            fasts[k]->data(), (size_t)n_fast[k] * sizeof(DBox));
        }
        check("h2d stage", cudaMemcpy(d_stage, staging_host.data(), stage_bytes,
                                      cudaMemcpyHostToDevice));
        const DPoint* cand_dev = reinterpret_cast<const DPoint*>(d_stage);
        const DBox* fast_dev =
            reinterpret_cast<const DBox*>(d_stage + cand_bytes + pad);
        BatchRowArgs a{};
        a.K = K;
        a.n_ors = (int)ors.size();
        a.L = L; a.W = W; a.H = H;
        for (int k = 0; k < K; k++) {
            a.cand_off[k] = cand_off[k];
            a.fast_off[k] = fast_off[k];
            a.n_cand[k] = n_cand[k];
            a.n_fast[k] = n_fast[k];
            a.pair_start[k] = pair_start[k];
        }
        const int block = 256;
        const int grid = (int)std::min<size_t>((pairs + block - 1) / block, 2048);
        feas_batch_kernel<<<grid, block>>>(cand_dev, fast_dev,
                                           reinterpret_cast<const DPoint*>(d_ors),
                                           d_bits, a);
        check("kernel", cudaGetLastError());
        check("sync", cudaDeviceSynchronize());
        std::vector<uint8_t> bits_host(pairs);
        check("d2h bits", cudaMemcpy(bits_host.data(), d_bits, pairs,
                                     cudaMemcpyDeviceToHost));
        for (int k = 0; k < K; k++)
            out[k].assign(bits_host.begin() + pair_start[k],
                          bits_host.begin() + pair_start[k] +
                                  (size_t)n_cand[k] * ors.size());
        return out;
    }

    ~CudaBatchBackend() {
        cudaFree(d_stage);
        cudaFree(d_ors);
        cudaFree(d_bits);
    }
};

// One batched decode row: per-row PyRng + the per-pass state that decode_pass
// keeps as fresh locals (placed/fast/cand reset every pass in run_batch).
struct BatchRow {
    explicit BatchRow(uint64_t s) : rng(s), cand({{0, 0, 0}}) {}
    plp::PyRng rng;
    std::vector<plp::PlacedBox> placed;
    std::vector<plp::FastBox> fast;
    std::vector<plp::Pt> cand;
};

// Line-by-line mirror of plp::decode_pass for K rows in lockstep: same
// item_sequence, same orients, same per-row decision (feas/pick/cand update);
// only the per-step check_all calls of all K rows collapse into one batched
// backend call. Any per-row behavior that differs from decode_pass breaks the
// per-row bit-exact verification.
void decode_pass_batch(const plp::Instance& inst, const std::string& order,
                       bool first_fit, double eps,
                       std::vector<BatchRow>& rows, CudaBatchBackend& be) {
    int L = inst.L, W = inst.W, H = inst.H;
    std::vector<std::vector<plp::Pt>> orients;
    orients.reserve(inst.items.size());
    for (const auto& it : inst.items) orients.push_back(plp::orientations(it));
    std::vector<const std::vector<plp::Pt>*> cptr(rows.size());
    std::vector<const std::vector<plp::FastBox>*> fptr(rows.size());
    for (size_t k = 0; k < rows.size(); k++) {
        cptr[k] = &rows[k].cand;
        fptr[k] = &rows[k].fast;
    }
    for (auto [ti, copy] : plp::item_sequence(inst, order)) {
        (void)copy;
        const auto& ors = orients[ti];
        if (ors.empty()) continue;
        std::vector<std::vector<uint8_t>> bits_all =
            be.check_all_batch(cptr, fptr, ors, L, W, H);
        for (size_t k = 0; k < rows.size(); k++) {
            BatchRow& row = rows[k];
            const auto& bits = bits_all[k];
            std::vector<std::array<int, 4>> feas;   // (x, y, z, oi) in candidate-major order
            for (size_t ci = 0; ci < row.cand.size(); ci++) {
                int x = row.cand[ci][0], y = row.cand[ci][1], z = row.cand[ci][2];
                for (size_t oi = 0; oi < ors.size(); oi++)
                    if (bits[ci * ors.size() + oi])
                        feas.push_back({x, y, z, (int)oi});
            }
            if (feas.empty()) continue;             // skip this copy; pass continues
            std::array<int, 4> pick;
            if (!(first_fit || eps == 0.0)) {
                if (row.rng.random() >= 1.0 - eps)
                    pick = feas[(size_t)row.rng.randrange((int)feas.size())];
                else
                    pick = feas[0];
            } else {
                pick = feas[0];
            }
            int w = ors[pick[3]][0], d = ors[pick[3]][1], h = ors[pick[3]][2];
            row.placed.push_back({pick[0], pick[1], pick[2], w, d, h, (int)ti});
            row.fast.push_back({pick[0], pick[1], pick[2], pick[0] + w, pick[1] + d, pick[2] + h});
            // extreme-point update: swallow points strictly inside the new box,
            // add its 8 corners (they can never be strictly inside an older box)
            std::vector<plp::Pt> nc;
            nc.reserve(row.cand.size() + 8);
            for (const auto& p : row.cand)
                if (!(pick[0] < p[0] && p[0] < pick[0] + w
                      && pick[1] < p[1] && p[1] < pick[1] + d
                      && pick[2] < p[2] && p[2] < pick[2] + h))
                    nc.push_back(p);
            for (int cx : {pick[0], pick[0] + w})
                for (int cy : {pick[1], pick[1] + d})
                    for (int cz : {pick[2], pick[2] + h})
                    nc.push_back({cx, cy, cz});
            plp::sort_dedup_pt(nc);
            row.cand = std::move(nc);
        }
    }
}

// Mirror of plp::run for K independent rows: row k runs with PyRng(seed + k)
// (Opsiyon A); per-row packings, per-row D4' acceptance/best selection. All
// rows stay in lockstep (same pass count).
std::vector<plp::RunResult> run_batch(const plp::Instance& inst, int seed,
                                      int K, double time_limit_s, double eps,
                                      double* theta, const std::string& order,
                                      bool first_fit, bool single_pass,
                                      CudaBatchBackend& be) {
    std::vector<BatchRow> rows;
    rows.reserve(K);
    for (int k = 0; k < K; k++) rows.emplace_back((uint64_t)seed + (uint64_t)k);
    auto t0 = std::chrono::steady_clock::now();
    struct Pack {
        long long volume;
        int n;
        double score;
        std::vector<plp::PlacedBox> placed;
    };
    std::vector<std::vector<Pack>> packings(K);
    int k = 0;
    for (;;) {
        for (auto& row : rows) {
            row.placed.clear();
            row.fast.clear();
            row.cand = {{0, 0, 0}};
        }
        decode_pass_batch(inst, order, first_fit, eps, rows, be);
        k++;
        for (int r = 0; r < K; r++) {
            long long vol = 0;
            for (const auto& b : rows[r].placed) vol += 1LL * b.w * b.d * b.h;
            packings[r].push_back(Pack{vol, (int)rows[r].placed.size(),
                                       plp::stability_score(rows[r].placed),
                                       std::move(rows[r].placed)});
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (single_pass || el >= time_limit_s) break;
    }
    std::vector<plp::RunResult> results(K);
    for (int r = 0; r < K; r++) {
        const auto& pk = packings[r];
        std::vector<const Pack*> accepted;
        for (const auto& p : pk)
            if (theta == nullptr || p.score >= *theta) accepted.push_back(&p);
        // D4': highest volume among accepted; if none accepted, overall best
        // (fallback). Ties keep the earliest packing (strict > only).
        const Pack* best;
        size_t pool_end;
        if (accepted.empty()) {
            best = &pk.front();
            pool_end = pk.size();
        } else {
            best = accepted.front();
            pool_end = accepted.size();
        }
        for (size_t i = 1; i < pool_end; i++) {
            const Pack* p = accepted.empty() ? &pk[i] : accepted[i];
            if (p->volume > best->volume || (p->volume == best->volume && p->n > best->n))
                best = p;
        }
        plp::RunResult& res = results[r];
        for (const auto& b : best->placed)
            res.placements.push_back({b.item_type, b.x, b.y, b.z, b.w, b.d, b.h});
        res.n_boxes = best->n;
        res.volume = best->volume;
        res.score = best->score;
        res.fallback = accepted.empty();
        res.n_passes = k;
        res.seed = seed + r;
        res.time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    return results;
}

struct CudaBackend : plp::FeasibilityBackend {
    unsigned char* d_stage = nullptr;   // [cand | 24B-pad | fast], one H2D
    int* d_ors = nullptr;
    uint8_t* d_bits = nullptr;
    size_t cap_stage = 0, cap_ors = 0, cap_bits = 0;
    // Host mirror of the last ors table uploaded to d_ors (content cache).
    std::vector<DPoint> ors_cache_host;
    // Host staging mirror: [cand | 24B-pad | fast] for a single H2D per call.
    std::vector<unsigned char> staging_host;

    // Report + abort ONLY on a non-success CUDA return code.
    static void check(const char* what, cudaError_t e) {
        if (e == cudaSuccess) return;
        std::fprintf(stderr, "cuda %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }

    // Grow a device buffer geometrically as the decode progresses.
    // (T**: int** does not implicitly convert to void** in C++.)
    template <typename T>
    static void ensure(T** ptr, size_t* cap, size_t need, const char* what) {
        if (need <= *cap) return;
        size_t nc = *cap ? *cap * 2 : 4096;
        if (nc < need) nc = need;
        if (*ptr) { cudaFree(*ptr); *ptr = nullptr; *cap = 0; }
        check(what, cudaMalloc(reinterpret_cast<void**>(ptr), nc * sizeof(T)));
        *cap = nc;
    }

    std::vector<uint8_t> check_all(const std::vector<plp::Pt>& cand,
                                   const std::vector<plp::Pt>& ors,
                                   const std::vector<plp::FastBox>& fast,
                                   int L, int W, int H) override {
        size_t n_cand = cand.size(), n_ors = ors.size(), n_fast = fast.size();
        size_t n = n_cand * n_ors;
        std::vector<uint8_t> bits(n, 0);
        if (n == 0) return bits;
        ensure(&d_bits, &cap_bits, n, "bits malloc");
        // ors (per-item-type orientations) is constant across all calls that
        // carry the same table: keep a host mirror and re-upload only when
        // the content actually changes. ensure() may reallocate d_ors, but
        // only in the upload branch, which refreshes the content right after
        // — the skip path never sees a stale buffer.
        bool ors_hit = ors_cache_host.size() == n_ors &&
                       std::memcmp(ors.data(), ors_cache_host.data(),
                                   n_ors * sizeof(DPoint)) == 0;
        if (!ors_hit) {
            ensure(&d_ors, &cap_ors, n_ors, "ors malloc");
            check("h2d ors", cudaMemcpy(d_ors, ors.data(), n_ors * sizeof(DPoint),
                                       cudaMemcpyHostToDevice));
            const DPoint* src = reinterpret_cast<const DPoint*>(ors.data());
            ors_cache_host.assign(src, src + n_ors);
        }
        // cand + fast packed into ONE staging buffer -> one H2D per call.
        // fast starts on a 24B boundary (DBox alignment); pad bytes are never
        // read by the kernel. Values/offsets are identical to the former
        // separate d_cand/d_fast copies, so the kernel sees the same data.
        const size_t cand_bytes = n_cand * sizeof(DPoint);
        const size_t pad = (24 - cand_bytes % 24) % 24;
        const size_t stage_bytes = cand_bytes + pad + n_fast * sizeof(DBox);
        ensure(&d_stage, &cap_stage, stage_bytes, "stage malloc");
        staging_host.resize(stage_bytes);
        std::memcpy(staging_host.data(), cand.data(), cand_bytes);
        if (n_fast)
            std::memcpy(staging_host.data() + cand_bytes + pad, fast.data(),
                        n_fast * sizeof(DBox));
        check("h2d stage", cudaMemcpy(d_stage, staging_host.data(), stage_bytes,
                                      cudaMemcpyHostToDevice));
        const DPoint* cand_dev = reinterpret_cast<const DPoint*>(d_stage);
        const DBox* fast_dev =
            reinterpret_cast<const DBox*>(d_stage + cand_bytes + pad);
        const int block = 256;
        const int grid = (int)std::min<size_t>((n + block - 1) / block, 2048);
        feas_kernel<<<grid, block>>>(cand_dev, (int)n_cand,
                                     reinterpret_cast<const DPoint*>(d_ors),
                                     (int)n_ors,
                                     fast_dev, (int)n_fast, L, W, H, d_bits);
        check("kernel", cudaGetLastError());
        check("sync", cudaDeviceSynchronize());
        check("d2h bits", cudaMemcpy(bits.data(), d_bits, n, cudaMemcpyDeviceToHost));
        return bits;
    }

    ~CudaBackend() override {
        cudaFree(d_stage);
        cudaFree(d_ors);
        cudaFree(d_bits);
    }
};

// ===========================================================================
// Wave 2b — device-driven decode (latency axis, spec §6 Wave 2b; Turn 6).
//
// The 136-step host decode loop moves into ONE kernel: K rows run as K
// blocks (1 block x 1024 threads per row), each block walking the item
// sequence serially with its own PERSISTENT device state:
//   * MT19937 (mt_[624] + mti) — line-by-line device mirror of plp::PyRng
//     (uint32-exact twist/temper, same random() double expression, same
//     data-dependent randrange rejection loop); lives in global memory and
//     survives across passes -> RNG continuation with no host round trips;
//   * the candidate list as packed (z,y,x) uint64 keys — the sorted+unique
//     invariant is preserved by 2-list merge + adjacent dedup (the final
//     list is unique+sorted, so the result equals sort_dedup_pt regardless
//     of merge implementation);
//   * fast boxes + placed boxes (device-resident, reset each pass like the
//     fresh locals of decode_pass).
// The host per pass does: ONE kernel launch + ONE sync + ONE D2H (K rows'
// placed); H2D per pass = 0 (instance tables uploaded once in init). FSR/
// stability (plp::stability_score) and D4' best-selection stay on the host
// in frozen functions; run_device mirrors run_batch line by line (row k =
// seed k, lockstep passes, same time budget).
//
// Staged bit-exactness (spec §6 risk order): (1) --eps 0 (no RNG, pick =
// feas[0]); (2) V0 (eps 0.3) — exercises the device MT19937 end to end.
// --rngdump N cross-checks the device MT19937 word for word against the
// frozen host plp::PyRng before full decode parity.
// ===========================================================================

constexpr int DEV_BLOCK = 1024;
constexpr int DEV_CAP_CAND = 4096;           // cand cap; overflow -> err flag
constexpr uint64_t DEV_MASK21 = 0x1FFFFFULL; // 21 bits per coordinate
// Turn 7a: dynamic shared layout (bytes): s_bits 24576 + s_cand 32768 +
// s_cmerge 32832 = 90176 B. sm_121 (Spark GB10, measured 2026-10-06): optin
// 101376 B per block -> fits with 11.2 KB margin; set via
// cudaFuncSetAttribute in CudaDeviceBackend::init.
constexpr int DEV_SHARED_BYTES = 90176;

// PlacedBox layout mirror (7 ints) for the device side.
struct DPlaced { int x, y, z, w, d, h, item_type; };
static_assert(sizeof(DPlaced) == 7 * sizeof(int), "layout must match PlacedBox");

// Packed (z,y,x) key: 21 bits/field, numeric order == (z,y,x) lexicographic
// (the sort_dedup_pt order). Host verifies L,W,H < 2^21 before use.
static __host__ __device__ inline uint64_t pack_pt(int x, int y, int z) {
    return ((uint64_t)(unsigned)z << 42) | ((uint64_t)(unsigned)y << 21)
           | (uint64_t)(unsigned)x;
}

// Per-row persistent device state (pure data; row k = d_rows + k).
struct DevRow {
    uint32_t mt[624];
    int mti;
    int n_cand;
    uint64_t cand[DEV_CAP_CAND];             // packed (z,y,x), sorted+unique
    uint64_t cand_merge[DEV_CAP_CAND + 8];   // scratch: survivors / merge
};

// Result slot, D2H'd with the whole K-row slab once per pass:
// {n_placed, err, score, placed[0..n_items)} — the flexible tail is
// allocated beyond the static head (byte stride = 2*sizeof(int) +
// sizeof(double) + n_items*28). Turn 9: score = device stability_score
// (stability_kernel writes it; offset 2*sizeof(int), 8-aligned).
struct DevSlot {
    int n_placed;
    int err;
    double score;
    DPlaced placed[1];
};

// ---------------------------------------------------------------------------
// Device MT19937 — line-by-line mirror of plp::PyRng (plp_common.hpp:162).
// Integer-exact; the only double op is random(), copied 1:1 (the build has
// --fmad=false, so no contraction).
// ---------------------------------------------------------------------------
__device__ inline uint32_t mt_genrand32(uint32_t* mt, int* mti) {
    if (*mti >= 624) {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (mt[i] & 0x80000000u) | (mt[(i + 1) % 624] & 0x7fffffffu);
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1u) mt[i] ^= 0x9908b0dfu;
        }
        *mti = 0;
    }
    uint32_t y = mt[(*mti)++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

__device__ inline double mt_random(uint32_t* mt, int* mti) {
    uint32_t a = mt_genrand32(mt, mti) >> 5;
    uint32_t b = mt_genrand32(mt, mti) >> 6;
    return (a * 67108864.0 + (double)b) * (1.0 / 9007199254740992.0);
}

// randrange(n), n > 0: k = bit_length(n), r = getrandbits(k), reject while
// r >= n — the rejection draws are data-dependent and must be mirrored
// exactly (same word count as the host PyRng for the same trajectory).
__device__ inline int mt_randrange(uint32_t* mt, int* mti, int n) {
    int k = 32 - __clz((unsigned)n);
    int r = (int)(mt_genrand32(mt, mti) >> (32 - k));
    while (r >= n) r = (int)(mt_genrand32(mt, mti) >> (32 - k));
    return r;
}

// PyRng ctor mirror: init_genrand(19650218) + init_by_array(key from seed).
__device__ inline void mt_init(uint32_t* mt, int* mti, uint64_t seed) {
    uint64_t n = seed;
    int bits = 0;
    for (uint64_t t = n; t; t >>= 1) bits++;
    int keyused = (bits == 0) ? 1 : ((bits - 1) / 32 + 1);  // <= 2 (uint64)
    uint32_t key[2];
    for (int j = 0; j < keyused; j++) key[j] = (uint32_t)(n >> (32 * j));
    mt[0] = 19650218u;
    for (int i = 1; i < 624; i++)
        mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint32_t)i;
    int i = 1, j = 0;
    int k = (624 > keyused) ? 624 : keyused;
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u))
                + key[j] + (uint32_t)j;
        i++; j++;
        if (i >= 624) { mt[0] = mt[623]; i = 1; }
        if (j >= keyused) j = 0;
    }
    for (k = 623; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u))
                - (uint32_t)i;
        i++;
        if (i >= 624) { mt[0] = mt[623]; i = 1; }
    }
    mt[0] = 0x80000000u;
    *mti = 624;
}

// One-time per-run seeding: row k = PyRng(seed + k) (Opsiyon A semantics).
__global__ void dev_rng_init_kernel(DevRow* __restrict__ drows, uint64_t seed) {
    if (threadIdx.x != 0) return;
    DevRow* row = drows + blockIdx.x;
    mt_init(row->mt, &row->mti, seed + (uint64_t)blockIdx.x);
}

// --rngdump probe: 16 random() + 16 getrandbits(13) per row (state NOT
// written back; standalone corner check against the frozen host PyRng).
__global__ void dev_rngdump_kernel(const DevRow* __restrict__ drows,
                                   double* __restrict__ d_rand,
                                   uint32_t* __restrict__ d_gb) {
    if (threadIdx.x != 0) return;
    uint32_t mt[624];
    int mti;
    for (int i = 0; i < 624; i++) mt[i] = drows[blockIdx.x].mt[i];
    mti = drows[blockIdx.x].mti;
    for (int i = 0; i < 16; i++) d_rand[blockIdx.x * 16 + i] = mt_random(mt, &mti);
    for (int i = 0; i < 16; i++)
        d_gb[blockIdx.x * 16 + i] = mt_genrand32(mt, &mti) >> (32 - 13);
}

// ---------------------------------------------------------------------------
// The device-driven decode: grid = K blocks (1 per row), 1024 threads.
// Turn 7a: the serial phase is block-parallel and the cand lists live in
// DYNAMIC shared (DEV_SHARED_BYTES; sm_121 optin 101376 B — measured):
//   * s_cand/s_cmerge: shared working copies of row->cand (loaded at kernel
//     start, written back at the end — cross-pass persistence keeps living
//     in the global row state, exactly as before);
//   * filter (cnt + feas[0]): per-candidate thread count + warp reductions
//     (feas[0] = min ci then min oi — the candidate-major frozen order);
//   * swallow: parallel mark + work-efficient block scan (4 x 1024 chunks:
//     warp shfl inclusive scan -> warp 0 exclusive scan of 32 warp totals)
//     + compact -> s_cmerge;
//   * 2-list merge + dedup: the 8 corners are lower_bound-searched in
//     parallel (lanes 0-7) into the compacted survivor list; kept_j =
//     !eq_j; every element's final position = its rank in the sorted-unique
//     union (#{values smaller than it}) -> parallel writes (survivors are
//     read from s_cmerge, so the s_cand rewrite has no in-place race).
//     Result == sort_dedup_pt (unique+sorted is canonical).
// Turn 7b: the pick second pass (idx > 0) is a PARALLEL select of the
// idx-th (0-based) set bit of s_bits — pair order == the host feas list
// order. The pair order is j-major: group j = the CONTIGUOUS block of
// pairs [1024j, 1024j+1024), then warp, then lane — so the select mirrors
// that hierarchy exactly: L1 per-group block totals (warp butterfly +
// thread-0 combine, the filter's 4-chunk pattern per group) + thread-0
// scan -> group G and its base; L2 per-warp totals inside G + warp-0
// exclusive scan -> warp W; an inclusive lane scan in the target warp ->
// the one lane publishing (px,py,pz,oi) to shared. (A per-warp total over
// ALL groups is NOT a contiguous interval of the set-bit order — a warp
// owns pairs {w*32+lane+1024j}, i.e. one 32-pair slice per group, so the
// groups interleave; that ordering assumption was the v1 bug, caught by
// the parity gate when n_pairs > 1024.)
// ALL RNG draws stay on thread 0 (same condition, order and double
// expression as the frozen host logic; s_selpick/s_idx published). Every
// __syncthreads sits on a block-uniform branch (s_cnt/s_selpick/s_upd/
// s_err are shared and read only after the previous sync) and every shfl
// is full-mask + unconditional (guide B.22: inactive target lane ->
// undefined), so no thread can skip a sync -> no deadlock.
// ---------------------------------------------------------------------------
__global__ void decode_device_kernel(DevRow* __restrict__ drows,
                                     DBox* __restrict__ dfast,
                                     char* __restrict__ dslots, int n_items,
                                     size_t slot_stride,
                                     const int* __restrict__ step_ti,
                                     const int* __restrict__ ors_off,
                                     const int* __restrict__ ors_n,
                                     const DPoint* __restrict__ ors_tab,
                                     int n_steps, int L, int W, int H,
                                     double eps, int first_fit) {
    extern __shared__ unsigned char dyn[];
    uint8_t*  s_bits   = dyn;                          // [0)       24576
    uint64_t* s_cand   = reinterpret_cast<uint64_t*>(dyn + 24576);
    uint64_t* s_cmerge = reinterpret_cast<uint64_t*>(dyn + 57344);

    __shared__ uint32_t s_mt[624];
    __shared__ int s_mti;
    __shared__ int s_nfast;
    __shared__ int s_err;
    __shared__ int s_ncand;
    __shared__ int s_wcnt[32], s_wmin[32];   // filter partials
    __shared__ int s_wsum[32], s_wbase[32], s_chunktot[4];  // scan scratch
    __shared__ int s_cnt, s_fmin;
    __shared__ int s_px, s_py, s_pz, s_poi, s_ox, s_oy, s_oz;  // pick publish
    __shared__ int s_selpick, s_idx, s_selwarp, s_gbase;  // Turn 7b select
    __shared__ int s_upd;
    __shared__ uint64_t s_corner[8];
    __shared__ int s_kept[8], s_cout[8];
    __shared__ int s_nsurv;

    const int tid = threadIdx.x;
    const int w = tid >> 5;
    const int lane = tid & 31;
    DevRow* row = drows + blockIdx.x;
    DBox* fast = dfast + (size_t)blockIdx.x * n_items;
    // Turn 9: slot head gained `double score` (offset 2*sizeof(int)).
    // Turn 12: stride passed from host (slot_bytes(), 8-byte padded) — the
    // `double score` at slot offset 8 requires each slot BASE to be
    // 8-aligned; the unpadded stride 16+28*n_items is 4 mod 8 for odd
    // n_items (odd instances, row>0: "misaligned address", BR15 n=141).
    char* slot = dslots + (size_t)blockIdx.x * slot_stride;
    int* slot_nplaced = reinterpret_cast<int*>(slot);
    int* slot_err = slot_nplaced + 1;
    DPlaced* slot_placed =
        reinterpret_cast<DPlaced*>(slot + 2 * sizeof(int) + sizeof(double));

    if (tid < 624) s_mt[tid] = row->mt[tid];
    if (tid == 0) {
        s_mti = row->mti;
        s_nfast = 0;
        s_err = 0;
        // per-pass reset (decode_pass fresh locals / run_batch row clear)
        row->n_cand = 1;
        row->cand[0] = pack_pt(0, 0, 0);
        *slot_nplaced = 0;
        *slot_err = 0;
    }
    __syncthreads();
    // cand -> shared working copy (global row state stays the persistent
    // store; written back at kernel end).
    for (int i = 0; i < row->n_cand; i += DEV_BLOCK)
        s_cand[i] = row->cand[i];
    if (tid == 0) s_ncand = row->n_cand;
    __syncthreads();

    int n_placed = 0;   // thread-0 counter, published to the slot at the end
    int n_fast = 0;     // thread-0 counter, published via s_nfast per step

    for (int s = 0; s < n_steps; s++) {
        const int ti = step_ti[s];
        const int n_ors = ors_n[ti];
        if (n_ors == 0 || s_err != 0) {       // uniform: frozen
            __syncthreads();                   // "if (ors.empty()) continue"
            continue;
        }
        const DPoint* ors = ors_tab + ors_off[ti];
        const int n_cand = s_ncand;
        const int n_pairs = n_cand * n_ors;
        // pair loop: one thread per (candidate, orientation), candidate-major
        // — the same order as decode_pass's `feas` list.
        for (int t = tid; t < n_pairs; t += DEV_BLOCK) {
            const int ci = t / n_ors;
            const int oi = t - ci * n_ors;
            const uint64_t key = s_cand[ci];
            s_bits[t] = (uint8_t)(check_feasible_one_dev(
                (int)(key & DEV_MASK21),
                (int)((key >> 21) & DEV_MASK21),
                (int)(key >> 42),
                ors[oi], fast, s_nfast, L, W, H) ? 1 : 0);
        }
        __syncthreads();
        // parallel filter, candidate-major (== frozen feas order): cnt =
        // |feas|, feas[0] = min ci then min oi; the pick needs exactly ONE
        // entry (feas[0] or feas[idx]), so no full feas list is materialized.
        {
            // Per-chunk fresh partials (stale cross-chunk values would
            // double-count after the butterfly); thread 0 combines each
            // chunk's 32 warp partials BEFORE the next chunk overwrites them.
            int tot_cnt = 0, tot_min = 0x7fffffff;   // thread-0 running sum
            for (int c = 0; c < 4; c++) {
                const int i = c * DEV_BLOCK + tid;
                int ci_cnt = 0, ci_foi = -1;
                if (i < n_cand)
                    for (int oi = 0; oi < n_ors; oi++)
                        if (s_bits[i * n_ors + oi]) {
                            if (ci_foi < 0) ci_foi = oi;
                            ci_cnt++;
                        }
                int tcnt = ci_cnt;
                int tmin = (ci_cnt > 0) ? ((i << 3) | ci_foi) : 0x7fffffff;
                for (int o = 16; o; o >>= 1) {   // classic butterfly: lane 0
                    tcnt += __shfl_down_sync(0xffffffffu, tcnt, o);   // = total
                    const int u = __shfl_down_sync(0xffffffffu, tmin, o);
                    if (u < tmin) tmin = u;
                }
                if (lane == 0) {
                    s_wcnt[w] = tcnt;
                    s_wmin[w] = tmin;
                }
                __syncthreads();
                if (tid == 0) {
                    for (int x = 0; x < 32; x++) {
                        tot_cnt += s_wcnt[x];
                        if (s_wmin[x] < tot_min) tot_min = s_wmin[x];
                    }
                }
                // barrier: thread 0 must finish reading this chunk's
                // s_wcnt/s_wmin BEFORE any warp overwrites them for chunk c+1
                // (write@793 next iteration); without it the read races the
                // next chunk's write -> nondeterministic s_cnt/s_fmin.
                __syncthreads();
            }
            if (tid == 0) {
                s_cnt = tot_cnt;
                s_fmin = tot_min;
            }
            __syncthreads();
        }
        if (s_cnt == 0) continue;        // frozen: feas.empty() -> skip copy
        if (tid == 0) {
            // pick (frozen logic): feas[0], or feas[idx] with
            // idx = randrange(cnt) — RNG draws thread-0 only, same order and
            // conditions as the host decode_pass. Turn 7b: when idx > 0 the
            // idx-th entry is located by the block-parallel select below
            // (pair order == host feas order); the tentative feas[0]
            // coordinates are published so the no-select paths keep working.
            const uint64_t fkey = s_cand[s_fmin >> 3];
            s_px = (int)(fkey & DEV_MASK21);
            s_py = (int)((fkey >> 21) & DEV_MASK21);
            s_pz = (int)(fkey >> 42);
            s_poi = s_fmin & 7;
            s_selpick = 0;
            if (!(first_fit || eps == 0.0)) {
                if (mt_random(s_mt, &s_mti) >= 1.0 - eps) {
                    const int idx = mt_randrange(s_mt, &s_mti, s_cnt);
                    if (idx > 0) {
                        s_idx = idx;
                        s_selpick = 1;
                    }
                }
            }
        }
        __syncthreads();           // publish s_selpick/s_idx/s_px..s_poi
        if (s_selpick) {
            // Turn 7b: select the s_idx-th (0-based) set bit of s_bits —
            // exactly the host's feas[idx] (pair index p = ci*n_ors + oi).
            // Hierarchy == pair order: group j = the contiguous block of
            // pairs [1024j, 1024j+1024), then warp, then lane. Scratch
            // (the filter/merge phases' slots, free here — separated by
            // syncs): s_wsum[0..nJ) group totals, s_wcnt warp partials,
            // s_wbase warp bases, s_selwarp = (G<<5)|W, s_gbase.
            const int nJ = (n_pairs + 1023) >> 10;
            // L1: s_wsum[j] = popcount of group j, in group order — the
            // filter's per-chunk pattern per group (warp butterfly ->
            // thread-0 combine; the barrier keeps the next group's s_wcnt
            // write off this group's read).
            for (int j = 0; j < nJ; j++) {
                const int p = tid + 1024 * j;
                int c = (p < n_pairs) ? (int)s_bits[p] : 0;
                for (int o = 16; o; o >>= 1)
                    c += __shfl_down_sync(0xffffffffu, c, o);
                if (lane == 0) s_wcnt[w] = c;
                __syncthreads();
                if (tid == 0) {
                    int gt = 0;
                    for (int x = 0; x < 32; x++) gt += s_wcnt[x];
                    s_wsum[j] = gt;
                }
                __syncthreads();   // s_wsum[j] visible; s_wcnt free for j+1
            }
            if (tid == 0) {
                int base = 0;
                s_gbase = 0;
                s_selwarp = 0;
                for (int j = 0; j < nJ; j++) {
                    if (s_idx >= base && s_idx < base + s_wsum[j]) {
                        s_gbase = base;
                        s_selwarp = j << 5;
                        break;
                    }
                    base += s_wsum[j];
                }
            }
            __syncthreads();           // s_gbase / s_selwarp(G) visible
            const int G = s_selwarp >> 5;
            {
                const int p = G * 1024 + tid;
                int c = (p < n_pairs) ? (int)s_bits[p] : 0;
                for (int o = 16; o; o >>= 1)
                    c += __shfl_down_sync(0xffffffffu, c, o);
                if (lane == 0) s_wcnt[w] = c;
            }
            __syncthreads();
            if (w == 0) {
                const int x = s_wcnt[lane];
                int inc = x;
                for (int o = 1; o < 32; o <<= 1) {
                    const int u = __shfl_up_sync(0xffffffffu, inc, o);
                    if (lane >= o) inc += u;
                }
                s_wbase[lane] = inc - x;
                const int base = inc - x;
                if (s_idx - s_gbase >= base
                    && s_idx - s_gbase < base + x)
                    // (G<<5) == (s_selwarp & ~0x1f): s_selwarp is j<<5 from L1
                    // (low 5 bits 0), and G = s_selwarp>>5 was read at line 896,
                    // i.e. BEFORE the __syncthreads() at 904. Writing from
                    // registers avoids the in-block LDS/STS of s_selwarp that
                    // racecheck flagged (all warp-0 lanes load it via the
                    // s_idx 64-bit pair while one lane stores it).
                    s_selwarp = (G << 5) | (int)lane;
            }
            __syncthreads();           // s_selwarp=(G<<5)|W / s_wbase visible
            const int W = s_selwarp & 0x1f;
            if (w == W) {
                const int p = G * 1024 + tid;
                const int b = (p < n_pairs) ? (int)s_bits[p] : 0;
                const int oin = s_idx - s_gbase - s_wbase[w];
                int pre = b;
                for (int d = 1; d < 32; d <<= 1) {
                    const int u = __shfl_up_sync(0xffffffffu, pre, d);
                    if (lane >= d) pre += u;
                }
                if (b && pre - b == oin) {   // the s_idx-th bit
                    const int ci = p / n_ors;
                    const int oi = p - ci * n_ors;
                    const uint64_t key = s_cand[ci];
                    s_px = (int)(key & DEV_MASK21);
                    s_py = (int)((key >> 21) & DEV_MASK21);
                    s_pz = (int)(key >> 42);
                    s_poi = oi;
                }
            }
            __syncthreads();           // final s_px..s_poi visible
        }
        if (tid == 0) {
            const int px = s_px, py = s_py, pz = s_pz;
            const DPoint& o = ors[s_poi];
            slot_placed[n_placed] = DPlaced{px, py, pz, o.x, o.y, o.z, ti};
            n_placed++;
            fast[n_fast] = DBox{px, py, pz, px + o.x, py + o.y, pz + o.z};
            n_fast++;
            s_nfast = n_fast;
            // publish the pick for the block-parallel update phase
            s_ox = o.x; s_oy = o.y; s_oz = o.z;
            // extreme-point update (frozen block mirror), block-parallel:
            // survivors = cand \ strictly-inside (frozen predicate) + the 8
            // corners (packed + insertion-sorted on thread 0)
            s_upd = (s_ncand + 8 <= DEV_CAP_CAND) ? 1 : 0;
            if (s_upd) {
                int nc = 0;
                for (int a = 0; a < 2; a++)
                    for (int b = 0; b < 2; b++)
                        for (int c = 0; c < 2; c++)
                            s_corner[nc++] = pack_pt(px + a * o.x,
                                                     py + b * o.y,
                                                     pz + c * o.z);
                for (int i = 1; i < 8; i++) {
                    const uint64_t v = s_corner[i];
                    int j = i - 1;
                    while (j >= 0 && s_corner[j] > v) {
                        s_corner[j + 1] = s_corner[j];
                        j--;
                    }
                    s_corner[j + 1] = v;
                }
            } else {
                *slot_err = 2;   // cand capacity overflow -> host abort
                s_err = 2;
            }
        }
        __syncthreads();           // publish s_ox/s_corner/s_upd/s_nfast
                                   // (s_px..s_poi already published above)
        if (s_upd) {
            // 1) mark + scan + compact (4 chunks of 1024, work-efficient):
            //    compacted survivors keep the original sorted order
            for (int c = 0; c < 4; c++) {
                const int i = c * DEV_BLOCK + tid;
                uint64_t key = 0;
                int surv = 0;
                if (i < n_cand) {
                    key = s_cand[i];
                    const int x = (int)(key & DEV_MASK21);
                    const int y = (int)((key >> 21) & DEV_MASK21);
                    const int z = (int)(key >> 42);
                    surv = !(s_px < x && x < s_px + s_ox
                             && s_py < y && y < s_py + s_oy
                             && s_pz < z && z < s_pz + s_oz);
                }
                int v = surv;
                for (int o = 1; o < 32; o <<= 1) {
                    const int u = __shfl_up_sync(0xffffffffu, v, o);
                    if (lane >= o) v += u;
                }
                if (lane == 31) s_wsum[w] = v;     // lane 31 = warp total
                __syncthreads();
                if (w == 0) {
                    int x = s_wsum[lane];
                    int inc = x;
                    for (int o = 1; o < 32; o <<= 1) {
                        const int u = __shfl_up_sync(0xffffffffu, inc, o);
                        if (lane >= o) inc += u;
                    }
                    s_wbase[lane] = inc - x;   // warp-exclusive base
                    if (lane == 31) s_chunktot[c] = inc;
                }
                __syncthreads();
                int base = 0;
                for (int c2 = 0; c2 < c; c2++) base += s_chunktot[c2];
                if (surv) s_cmerge[base + s_wbase[w] + v - 1] = key;
                __syncthreads();
            }
            if (tid == 0)
                s_nsurv = s_chunktot[0] + s_chunktot[1] + s_chunktot[2]
                          + s_chunktot[3];
            __syncthreads();
            // 2) 8 corners (lanes 0-7): parallel lower_bound into the
            //    compacted survivors + kept flag (eq -> dedup) + rank positions
            if (w == 0 && lane < 8) {
                int lo = 0, hi = s_nsurv;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (s_cmerge[mid] < s_corner[lane]) lo = mid + 1;
                    else hi = mid;
                }
                const int pos = lo;
                const int eq = (pos < s_nsurv)
                               && (s_cmerge[pos] == s_corner[lane]);
                int kb = eq ? 0 : 1;
                // inclusive prefix of the 8 kept flags; shfl_up scan needs
                // power-of-two offsets (1,2,4) — a sequential d++ over-counts
                // kb and pushes s_cout onto a survivor's slot (write race).
                for (int d = 1; d < 8; d <<= 1) {
                    const int u = __shfl_up_sync(0xffu, kb, d);
                    if (lane >= d) kb += u;
                }
                s_kept[lane] = eq ? 0 : 1;
                s_cout[lane] = pos + (kb - (eq ? 0 : 1));
            }
            __syncthreads();
            // 3) parallel writes: final position of every element = its rank
            //    in the sorted-unique union (== #{values smaller than it});
            //    survivors are read from s_cmerge -> no in-place race.
            //    Result == sort_dedup_pt (unique+sorted is canonical).
            for (int j2 = tid; j2 < s_nsurv; j2 += DEV_BLOCK) {
                const uint64_t key = s_cmerge[j2];
                int out = j2;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    if (s_kept[j] && s_corner[j] < key) out++;
                s_cand[out] = key;
            }
            if (w == 0 && lane < 8 && s_kept[lane])
                s_cand[s_cout[lane]] = s_corner[lane];
            if (tid == 0) {
                int nnew = s_nsurv;
                #pragma unroll
                for (int j = 0; j < 8; j++) nnew += s_kept[j];
                s_ncand = nnew;
            }
        }
        __syncthreads();   // publish s_ncand/s_cand, s_nfast to next step
    }

    __syncthreads();
    // Persistent row state: cand back to global + RNG (only thread 0 drew).
    for (int i = 0; i < s_ncand; i += DEV_BLOCK)
        row->cand[i] = s_cand[i];
    if (tid < 624) row->mt[tid] = s_mt[tid];
    if (tid == 0) {
        row->mti = s_mti;
        row->n_cand = s_ncand;
        *slot_nplaced = n_placed;
    }
}

// ---------------------------------------------------------------------------
// Turn 9: device FSR — plp::stability_score off the host (K rows x O(n^2)
// CPU) into a post-decode kernel. Line-by-line mirror of plp::compute_fsr
// + plp::stability_score (plp_common.hpp:354-397); the build is
// --fmad=false, so the double expressions cannot be contracted:
//   m[i] = RHO * (double)(1LL*w*d*h) / 1.0e6;  load = m;
//   outer order = host stable_sort by -z == (z desc, index asc) — the
//   counting sort below; equal-z k's cannot rest on each other (h >= 1)
//   so they never touch one another's load, but two k's may both load
//   onto the same j below -> the serial (z desc, index asc) order is
//   what the host runs and what is reproduced here;
//   for k: areas = {(j, ox*oy): j != k, placed[j].z+placed[j].h == bk.z,
//   ox > 0, oy > 0}  (ll casts 1:1 with the host); total = sum (integer,
//   order-independent); if (total > 0): load[j] += load[k] * (double)a /
//   (double)total;
//   fsr[i] = (MU/A_LAT_G) * load[i] / m[i];  score = #{fsr >= 1.0} / n;
//   empty slot -> 0.0 (host: placed.empty() -> 0.0).
// Structure: outer loop SERIAL (every thread sees k = s_ord[kk],
// warp-uniform), inner j-scan PARALLEL (n <= 136 < 1024: one box per
// thread, distinct writes), thread-0 sums are the only serial chains
// (n integer adds over pipelined L1 loads). NO SHFL / butterfly
// (Turn 8: per-element collectives at small n cost more than they save).
// Static shared, cap 256 (BR3_050: n <= 136, H = 220); overflow -> err 3.
// ---------------------------------------------------------------------------
constexpr int DEV_FSR_CAP = 256;

__global__ void stability_kernel(char* dslots, int n_items, int H,
                                 size_t slot_stride, double mu) {
    // Turn 12: stride from host (slot_bytes(), 8-byte padded) — see
    // decode_device_kernel (odd n_items -> unpadded stride 4 mod 8 ->
    // misaligned `double score` at offset 8 for row>0).
    DevSlot* slot =
        reinterpret_cast<DevSlot*>(dslots + (size_t)blockIdx.x * slot_stride);
    const int tid = threadIdx.x;
    const int n = slot->n_placed;

    __shared__ double s_m[DEV_FSR_CAP];
    __shared__ double s_load[DEV_FSR_CAP];
    __shared__ long long s_area[DEV_FSR_CAP];
    __shared__ int s_ord[DEV_FSR_CAP];
    __shared__ int s_z[DEV_FSR_CAP];
    __shared__ int s_cnt[DEV_FSR_CAP];      // z counts, then per-z cursor
    __shared__ int s_off[DEV_FSR_CAP + 1];  // s_off[z] = #{boxes: z' >= z}
    __shared__ int s_flag[DEV_FSR_CAP];
    __shared__ long long s_total;

    if (n > DEV_FSR_CAP || H > DEV_FSR_CAP) {
        if (tid == 0) slot->err = 3;
        return;
    }
    if (n == 0) {                          // host: placed.empty() -> 0.0
        if (tid == 0) slot->score = 0.0;
        return;
    }

    const DPlaced* placed = slot->placed;
    DPlaced pj;
    const bool have = tid < n;
    if (have) {
        pj = placed[tid];
        double m = plp::RHO * (double)(1LL * pj.w * pj.d * pj.h) / 1.0e6;
        s_m[tid] = m;
        s_load[tid] = m;                   // host: load = m
        s_z[tid] = pj.z;
    }
    if (tid < H) s_cnt[tid] = 0;
    __syncthreads();

    // Stable sort by (z desc, index asc) == host std::stable_sort(-z):
    // s_off[z] = #{boxes: z' >= z}, so the z group (descending order)
    // occupies [s_off[z+1], s_off[z]) in index-ascending order.
    // Thread 0, ~2n+2H shared ops (n, H <= 256).
    if (tid == 0) {
        for (int i = 0; i < n; i++) s_cnt[s_z[i]]++;
        s_off[H] = 0;
        for (int z = H - 1; z >= 0; z--) s_off[z] = s_off[z + 1] + s_cnt[z];
        for (int z = 0; z < H; z++) s_cnt[z] = 0;  // reuse as per-z cursor
        for (int i = 0; i < n; i++) {
            int z = s_z[i];
            s_ord[s_off[z + 1] + s_cnt[z]++] = i;
        }
    }
    __syncthreads();

    // Serial outer loop (host `for (int k : idx)`), parallel j-scan.
    for (int kk = 0; kk < n; kk++) {
        const int k = s_ord[kk];
        const DPlaced& bk = placed[k];
        long long a = 0;
        if (have && tid != k && pj.z + pj.h == bk.z) {
            long long ox = dminll((long long)bk.x + bk.w,
                                  (long long)pj.x + pj.w)
                           - (long long)dmaxll(bk.x, pj.x);
            if (ox > 0) {
                long long oy = dminll((long long)bk.y + bk.d,
                                      (long long)pj.y + pj.d)
                               - (long long)dmaxll(bk.y, pj.y);
                if (oy > 0) a = ox * oy;
            }
        }
        if (have) s_area[tid] = a;         // tid == k (or none) -> 0
        __syncthreads();
        if (tid == 0) {
            long long total = 0;
            for (int j = 0; j < n; j++) total += s_area[j];
            s_total = total;
        }
        __syncthreads();
        if (s_total > 0 && have && a > 0)
            s_load[tid] += s_load[k] * (double)s_area[tid] / (double)s_total;
    }

    // fsr + count (host: fsr[i] = (MU/A_LAT_G) * load[i] / m[i]; f >= 1.0).
    // mu: runtime kernel param (binding-regime campaign); mirrors plp::MU.
    if (have) {
        double fsr = (mu / plp::A_LAT_G) * s_load[tid] / s_m[tid];
        s_flag[tid] = (fsr >= 1.0) ? 1 : 0;
    }
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int i = 0; i < n; i++) c += s_flag[i];
        slot->score = (double)c / (double)n;   // host: (double)c / size
    }
}

// Host-side manager for the 2b buffers: one-time instance tables + K row
// states; per pass = 1 launch + 1 sync + 1 D2H (0 H2D).
struct CudaDeviceBackend {
    DevRow* d_rows = nullptr;
    DBox* d_fast = nullptr;
    char* d_slots = nullptr;
    int* d_step_ti = nullptr;
    int* d_ors_off = nullptr;
    int* d_ors_n = nullptr;
    DPoint* d_ors = nullptr;
    int K = 0, n_items = 0, n_types = 0, n_ors_total = 0;
    int L = 0, W = 0, H = 0;
    double eps = 0.0;
    int first_fit = 0;

    static void check(const char* what, cudaError_t e) {
        if (e == cudaSuccess) return;
        std::fprintf(stderr, "cuda %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }

    size_t slot_bytes() const {
        // Turn 9: + sizeof(double) for the device stability score in the
        // slot head (fetch reads it at offset 2*sizeof(int)).
        // Turn 12: 8-byte pad — the stride must stay 8-aligned: the
        // `double score` at offset 8 of EVERY slot is misaligned for
        // row>0 when 16+28*n_items is 4 mod 8 (odd n_items; DPlaced=28B).
        // Single source of truth: both kernels receive it as slot_stride.
        const size_t s = 2 * sizeof(int) + sizeof(double)
                        + (size_t)n_items * sizeof(DPlaced);
        return (s + 7) & ~(size_t)7;
    }

    void init(const plp::Instance& inst, int seed, int k, double eps_,
              bool ff, const std::string& order) {
        K = k;
        L = inst.L; W = inst.W; H = inst.H;
        eps = eps_;
        first_fit = ff ? 1 : 0;
        n_types = (int)inst.items.size();
        if (L > 0x1FFFFF || W > 0x1FFFFF || H > 0x1FFFFF) {
            std::fprintf(stderr,
                         "--device: container dims exceed the 21-bit pack "
                         "(L=%d W=%d H=%d)\n", L, W, H);
            std::exit(2);
        }
        // Turn 7a: dynamic shared (s_bits + s_cand + s_cmerge) needs opt-in
        // on sm_121 (default 48 KB static limit); 90176 B vs optin 101376 B
        // (measured on Spark GB10) -> fits; a hard error here if it ever does
        // not (e.g. different arch) instead of a silent launch failure.
        check("dev shared optin", cudaFuncSetAttribute(decode_device_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)DEV_SHARED_BYTES));
        const auto seq = plp::item_sequence(inst, order);
        n_items = (int)seq.size();
        std::vector<std::vector<plp::Pt>> orients;
        orients.reserve(n_types);
        for (const auto& it : inst.items)
            orients.push_back(plp::orientations(it));
        std::vector<int> step_ti(n_items), ors_off(n_types), ors_n(n_types);
        size_t tot = 0;
        for (int t = 0; t < n_types; t++) {
            ors_off[t] = (int)tot;
            ors_n[t] = (int)orients[t].size();
            tot += orients[t].size();
        }
        n_ors_total = (int)tot;
        std::vector<DPoint> ors_flat(n_ors_total);
        for (int t = 0; t < n_types; t++)
            for (size_t i = 0; i < orients[t].size(); i++)
                ors_flat[ors_off[t] + i] = {orients[t][i][0],
                                            orients[t][i][1],
                                            orients[t][i][2]};
        for (int s = 0; s < n_items; s++) step_ti[s] = seq[s].first;
        check("dev rows malloc", cudaMalloc(&d_rows, (size_t)K * sizeof(DevRow)));
        check("dev fast malloc",
              cudaMalloc(&d_fast, (size_t)K * n_items * sizeof(DBox)));
        check("dev slots malloc", cudaMalloc(&d_slots, (size_t)K * slot_bytes()));
        check("dev step malloc", cudaMalloc(&d_step_ti, n_items * sizeof(int)));
        check("dev orsoff malloc", cudaMalloc(&d_ors_off, n_types * sizeof(int)));
        check("dev orsn malloc", cudaMalloc(&d_ors_n, n_types * sizeof(int)));
        check("dev ors malloc", cudaMalloc(&d_ors, n_ors_total * sizeof(DPoint)));
        // Host mirror of the per-pass start state (mt is overwritten by
        // dev_rng_init_kernel before the first decode kernel runs).
        std::vector<DevRow> host_rows(K);
        for (int r = 0; r < K; r++) {
            host_rows[r].n_cand = 1;
            host_rows[r].cand[0] = pack_pt(0, 0, 0);
        }
        check("dev h2d rows",
              cudaMemcpy(d_rows, host_rows.data(), (size_t)K * sizeof(DevRow),
                         cudaMemcpyHostToDevice));
        check("dev memset fast",
              cudaMemset(d_fast, 0, (size_t)K * n_items * sizeof(DBox)));
        check("dev memset slots", cudaMemset(d_slots, 0, (size_t)K * slot_bytes()));
        check("dev h2d step", cudaMemcpy(d_step_ti, step_ti.data(),
                                         n_items * sizeof(int),
                                         cudaMemcpyHostToDevice));
        check("dev h2d orsoff", cudaMemcpy(d_ors_off, ors_off.data(),
                                           n_types * sizeof(int),
                                           cudaMemcpyHostToDevice));
        check("dev h2d orsn", cudaMemcpy(d_ors_n, ors_n.data(),
                                         n_types * sizeof(int),
                                         cudaMemcpyHostToDevice));
        check("dev h2d ors", cudaMemcpy(d_ors, ors_flat.data(),
                                        n_ors_total * sizeof(DPoint),
                                        cudaMemcpyHostToDevice));
        dev_rng_init_kernel<<<K, 32>>>(d_rows, (uint64_t)seed);
        check("dev rng init", cudaGetLastError());
        check("dev rng init sync", cudaDeviceSynchronize());
    }

    // One pass: 2 launches (decode + Turn 9 stability) + 1 sync (0 H2D —
    // all row state is device-resident; the single sync covers both
    // kernels, default-stream order).
    void run_pass() {
        const size_t slot_stride = slot_bytes();
        decode_device_kernel<<<K, DEV_BLOCK, DEV_SHARED_BYTES>>>(
            d_rows, d_fast, d_slots, n_items, slot_stride, d_step_ti,
            d_ors_off, d_ors_n,
            d_ors, n_items, L, W, H, eps, first_fit);
        check("dev kernel", cudaGetLastError());
        stability_kernel<<<K, DEV_BLOCK>>>(d_slots, n_items, H, slot_stride,
                                           plp::MU);
        check("dev stability", cudaGetLastError());
        check("dev sync", cudaDeviceSynchronize());
    }

    // One D2H per pass: the K-row slot slab; fills per-row placed + err +
    // the device stability score (Turn 9, offset 2*sizeof(int)).
    void fetch(std::vector<std::vector<plp::PlacedBox>>& out,
               std::vector<double>& scores, int& err_any) {
        std::vector<unsigned char> host((size_t)K * slot_bytes());
        check("dev d2h slots",
              cudaMemcpy(host.data(), d_slots, host.size(),
                         cudaMemcpyDeviceToHost));
        out.assign(K, {});
        scores.assign(K, 0.0);
        err_any = 0;
        for (int k = 0; k < K; k++) {
            const unsigned char* p = host.data() + (size_t)k * slot_bytes();
            int n = 0, e = 0;
            double sc = 0.0;
            std::memcpy(&n, p, sizeof(int));
            std::memcpy(&e, p + sizeof(int), sizeof(int));
            std::memcpy(&sc, p + 2 * sizeof(int), sizeof(double));
            scores[k] = sc;
            if (e != 0) err_any = e;
            auto& v = out[k];
            v.resize(n);
            const DPlaced* sp = reinterpret_cast<const DPlaced*>(
                p + 2 * sizeof(int) + sizeof(double));
            for (int i = 0; i < n; i++)
                v[i] = {sp[i].x, sp[i].y, sp[i].z, sp[i].w, sp[i].d,
                        sp[i].h, sp[i].item_type};
        }
    }

    ~CudaDeviceBackend() {
        cudaFree(d_rows);
        cudaFree(d_fast);
        cudaFree(d_slots);
        cudaFree(d_step_ti);
        cudaFree(d_ors_off);
        cudaFree(d_ors_n);
        cudaFree(d_ors);
    }
};

// Mirror of run_batch for K device-driven rows: row k = PyRng(seed + k)
// (state on device), per-row packings, per-row D4' acceptance/best selection
// (frozen semantics, host). Rows stay in lockstep (same pass count).
std::vector<plp::RunResult> run_device(const plp::Instance& inst, int seed,
                                       int K, double time_limit_s, double eps,
                                       double* theta, const std::string& order,
                                       bool first_fit, bool single_pass,
                                       CudaDeviceBackend& be) {
    be.init(inst, seed, K, eps, first_fit, order);
    auto t0 = std::chrono::steady_clock::now();
    struct Pack {
        long long volume;
        int n;
        double score;
        std::vector<plp::PlacedBox> placed;
    };
    std::vector<std::vector<Pack>> packings(K);
    std::vector<std::vector<plp::PlacedBox>> pass_out;
    std::vector<double> scores;   // Turn 9: device stability (slab)
    int k = 0;
    for (;;) {
        be.run_pass();
        k++;
        int err = 0;
        be.fetch(pass_out, scores, err);
        if (err != 0) {
            std::fprintf(stderr,
                         "device decode error flag %d (cand capacity / state "
                         "overflow?)\n", err);
            std::exit(1);
        }
        for (int r = 0; r < K; r++) {
            long long vol = 0;
            for (const auto& b : pass_out[r]) vol += 1LL * b.w * b.d * b.h;
            // Turn 9: host plp::stability_score (O(n^2) CPU) -> slab score
            // (bit-exact device mirror; parity gate compares the JSON
            // field as an exact double).
            packings[r].push_back(Pack{vol, (int)pass_out[r].size(),
                                       scores[r],
                                       std::move(pass_out[r])});
        }
        double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        if (single_pass || el >= time_limit_s) break;
    }
    std::vector<plp::RunResult> results(K);
    for (int r = 0; r < K; r++) {
        const auto& pk = packings[r];
        std::vector<const Pack*> accepted;
        for (const auto& p : pk)
            if (theta == nullptr || p.score >= *theta) accepted.push_back(&p);
        // D4': highest volume among accepted; if none accepted, overall best
        // (fallback). Ties keep the earliest packing (strict > only).
        const Pack* best;
        size_t pool_end;
        if (accepted.empty()) {
            best = &pk.front();
            pool_end = pk.size();
        } else {
            best = accepted.front();
            pool_end = accepted.size();
        }
        for (size_t i = 1; i < pool_end; i++) {
            const Pack* p = accepted.empty() ? &pk[i] : accepted[i];
            if (p->volume > best->volume || (p->volume == best->volume && p->n > best->n))
                best = p;
        }
        plp::RunResult& res = results[r];
        for (const auto& b : best->placed)
            res.placements.push_back({b.item_type, b.x, b.y, b.z, b.w, b.d, b.h});
        res.n_boxes = best->n;
        res.volume = best->volume;
        res.score = best->score;
        res.fallback = accepted.empty();
        res.n_passes = k;
        res.seed = seed + r;
        res.time_s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    }
    return results;
}

// --rngdump N: device MT19937 corner check. Row k = seed k (as in runs):
// 16 random() + 16 getrandbits(13) on device vs the frozen host plp::PyRng,
// word for word (same probe as selftest()). No instance needed.
void rngdump(int N) {
    if (N < 1) {
        std::fprintf(stderr, "--rngdump N must be >= 1\n");
        std::exit(2);
    }
    DevRow* d_rows = nullptr;
    double* d_rand = nullptr;
    uint32_t* d_gb = nullptr;
    CudaDeviceBackend::check("rngdump rows",
                             cudaMalloc(&d_rows, (size_t)N * sizeof(DevRow)));
    CudaDeviceBackend::check("rngdump memset",
                             cudaMemset(d_rows, 0, (size_t)N * sizeof(DevRow)));
    CudaDeviceBackend::check("rngdump rand",
                             cudaMalloc(&d_rand, (size_t)N * 16 * sizeof(double)));
    CudaDeviceBackend::check("rngdump gb",
                             cudaMalloc(&d_gb, (size_t)N * 16 * sizeof(uint32_t)));
    dev_rng_init_kernel<<<N, 32>>>(d_rows, 0);
    dev_rngdump_kernel<<<N, 32>>>(d_rows, d_rand, d_gb);
    CudaDeviceBackend::check("rngdump launch", cudaGetLastError());
    CudaDeviceBackend::check("rngdump sync", cudaDeviceSynchronize());
    std::vector<double> hr((size_t)N * 16);
    std::vector<uint32_t> hg((size_t)N * 16);
    CudaDeviceBackend::check("rngdump d2h rand",
                             cudaMemcpy(hr.data(), d_rand, hr.size() * sizeof(double),
                                        cudaMemcpyDeviceToHost));
    CudaDeviceBackend::check("rngdump d2h gb",
                             cudaMemcpy(hg.data(), d_gb, hg.size() * sizeof(uint32_t),
                                        cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int k = 0; k < N; k++) {
        plp::PyRng rng((uint64_t)k);
        bool row_ok = true;
        for (int i = 0; i < 16; i++) {
            const double h = rng.random();
            if (hr[(size_t)k * 16 + i] != h) {
                row_ok = false;
                std::fprintf(stderr, "row %d random[%d]: host %s dev %s\n", k, i,
                             plp::py_repr(h).c_str(),
                             plp::py_repr(hr[(size_t)k * 16 + i]).c_str());
            }
        }
        for (int i = 0; i < 16; i++) {
            const uint32_t h = rng.getrandbits(13);
            if (hg[(size_t)k * 16 + i] != h) {
                row_ok = false;
                std::fprintf(stderr, "row %d getrandbits[%d]: host %u dev %u\n",
                             k, i, h, hg[(size_t)k * 16 + i]);
            }
        }
        std::printf("SEED %u: %s\n", (unsigned)k, row_ok ? "PASS" : "FAIL");
        ok = ok && row_ok;
    }
    cudaFree(d_rows);
    cudaFree(d_rand);
    cudaFree(d_gb);
    std::printf("RNGDUMP: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) std::exit(1);
}

// --selftest: host-only RNG + float-format cross-check blocks, identical to
// plp_gpu.cpp (kept in sync). Makes the CUDA binary self-verifiable on Spark
// without an instance file; the reference is gpu/make_selftest_ref.py.
void print_hexfloat(double x) {
    uint64_t bits;
    static_assert(sizeof(double) == sizeof(uint64_t));
    std::memcpy(&bits, &x, sizeof bits);
    std::printf("%016llx\n", (unsigned long long)bits);
}

void selftest() {
    for (uint32_t s : {0u, 1u, 7u, 123456789u}) {
        plp::PyRng rng(s);
        std::printf("SEED %u\n", s);
        for (int i = 0; i < 16; i++) print_hexfloat(rng.random());
        for (int i = 0; i < 16; i++) std::printf("%u\n", rng.getrandbits(13));
    }
    const double vals[] = {
        0.0, 1.0, 0.5, 0.1, 2.0 / 3.0, 0.5785123966942148,
        0.6132075471698113, 0.6595744680851063, 1e-05, 1e-04,
        1e15, 1e16, 0.001, 123456789.5, 1.0 / 121.0, 1.0 / 106.0,
    };
    for (double v : vals) std::printf("REPR %s\n", plp::py_repr(v).c_str());
}

void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s <instance> [--seed N] [--time S] [--eps F] "
                 "[--theta F | --no-theta]\n"
                 "                 [--order volume|input] [--first-fit] "
                 "[--single-pass] [--batch K] [--device K] [--mu F] [--json]\n"
                 "       %s --selftest | --rngdump N\n",
                 prog, prog);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace plp;
    std::string instance_path, order = "volume";
    int seed = 0;
    double time_s = 300.0, eps = EPS_QUALITY, theta = THETA;
    bool no_theta = false, first_fit = false, single_pass = false, as_json = false;
    int batch = 0;   // 2a: K independent rows (row k = PyRng(seed + k)); 0 = off
    int device = 0;  // 2b: K device-driven rows (state on device); 0 = off
    if (const char* e = std::getenv("PLP_MU")) MU = std::strtod(e, nullptr);

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--selftest") { selftest(); return 0; }
        else if (a == "--rngdump") {
            rngdump(std::atoi(next(a.c_str()).c_str()));
            return 0;
        }
        else if (a == "--seed") seed = std::atoi(next(a.c_str()).c_str());
        else if (a == "--time") time_s = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--eps") eps = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--mu") MU = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--theta") theta = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--no-theta") no_theta = true;
        else if (a == "--order") { order = next(a.c_str()); }
        else if (a == "--first-fit") first_fit = true;
        else if (a == "--single-pass") single_pass = true;
        else if (a == "--batch") batch = std::atoi(next(a.c_str()).c_str());
        else if (a == "--device") device = std::atoi(next(a.c_str()).c_str());
        else if (a == "--json") as_json = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (!a.empty() && a[0] != '-') {
            if (!instance_path.empty()) { usage(argv[0]); return 2; }
            instance_path = a;
        } else {
            std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (instance_path.empty()) { usage(argv[0]); return 2; }

    Instance inst;
    try {
        inst = load_instance(instance_path);
    } catch (const InstanceFormatError& e) {
        std::fprintf(stderr, "instance error: %s\n", e.what());
        return 1;
    }
    if (order != "volume" && order != "input") {
        std::fprintf(stderr, "unknown order %s\n", order.c_str());
        return 2;
    }
    if (batch > 0) {
        if (!as_json) {
            std::fprintf(stderr, "--batch requires --json\n");
            return 2;
        }
        if (batch < 1 || batch > K_MAX) {
            std::fprintf(stderr, "--batch K must be in [1, %d]\n", K_MAX);
            return 2;
        }
        CudaBatchBackend be;
        auto results = run_batch(inst, seed, batch, time_s, eps,
                                 no_theta ? nullptr : &theta, order,
                                 first_fit, single_pass, be);
        std::cout << "{\"K\": " << batch << ", \"n_passes\": "
                  << results[0].n_passes << ", \"time_s\": "
                  << py_repr(results[0].time_s) << ", \"rows\": [";
        for (size_t r = 0; r < results.size(); r++) {
            if (r) std::cout << ", ";
            std::cout << to_json(results[r]);
        }
        std::cout << "]}\n";
        return 0;
    }
    if (device > 0) {
        if (!as_json) {
            std::fprintf(stderr, "--device requires --json\n");
            return 2;
        }
        if (batch > 0) {
            std::fprintf(stderr, "--device and --batch are mutually exclusive\n");
            return 2;
        }
        if (device < 1 || device > K_MAX) {
            std::fprintf(stderr, "--device K must be in [1, %d]\n", K_MAX);
            return 2;
        }
        CudaDeviceBackend be;
        auto results = run_device(inst, seed, device, time_s, eps,
                                  no_theta ? nullptr : &theta, order,
                                  first_fit, single_pass, be);
        std::cout << "{\"K\": " << device << ", \"n_passes\": "
                  << results[0].n_passes << ", \"time_s\": "
                  << py_repr(results[0].time_s) << ", \"rows\": [";
        for (size_t r = 0; r < results.size(); r++) {
            if (r) std::cout << ", ";
            std::cout << to_json(results[r]);
        }
        std::cout << "]}\n";
        return 0;
    }
    CudaBackend be;
    RunResult r = run(inst, seed, time_s, eps, no_theta ? nullptr : &theta,
                      order, first_fit, single_pass, be);
    if (as_json) {
        std::cout << to_json(r) << "\n";
    } else {
        std::printf("n_boxes=%d volume_cm3=%lld stability_score=%.4f fallback=%s "
                    "n_passes=%d time_s=%.3f\n",
                    r.n_boxes, (long long)r.volume, r.score,
                    r.fallback ? "true" : "false", r.n_passes, r.time_s);
    }
    return 0;
}
