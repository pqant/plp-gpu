// plp_block_cuda.cu — CUDA build of the block-building insertion engine
// (TRANSITION_BRIEF cp-148: block proof -> CUDA transition phase).
//
// Same semantics as plp_block.cpp (the frozen CPU reference, cp-147/148);
// ONLY the execution substrate changes. The whole block_pass loop lives in
// ONE kernel per pass (plp_gpu.cu Wave-2b pattern, memory plp-turn7a-ncu-
// findings): K rows run as K blocks (1 block x 1024 threads per row, row k =
// PyRng(seed + k)), each walking (plan -> placement) sweeps serially with
// device-resident state:
//   * MT19937 (mt_[624] + mti) — line-by-line device mirror of plp::PyRng,
//     all draws on thread 0 in the exact CPU order (shuffle draws, plan
//     shape picks, placement eps-greedy picks); persists across passes;
//   * candidate list as packed (z,y,x) uint64 keys — sorted+unique invariant
//     preserved by the Turn-7a parallel mark+scan+compact + corner merge
//     (== sort_dedup_pt, canonical unique+sorted result);
//   * feasibility scan one thread per candidate (n_ors == 1 at block
//     granularity — pair order == candidate order, so the Turn-7b select
//     mirrors the host feas[idx] list exactly);
//   * plan phase on thread 0 (RNG-sequential): types from the host-preset
//     (volume desc, ti) order filtered by rem>0 && orientations non-empty,
//     Fisher-Yates eps-shuffle, per-type suffix binary search over the
//     cnt-desc by_type table (suffix == shapes that fit, CPU comment rule)
//     + eps-greedy pick among the top-`topk`; plan ordering (bvol desc, ti,
//     oi, k, a, b — strict total order) by parallel bitonic sort of packed
//     u64 keys (fixed comparison network + total order == unique result ==
//     std::sort outcome);
//   * D4' online best (best_acc/best_all, strict > keeps earliest) stays on
//     the HOST exactly as run_block; FSR/stability is the Turn-9 device
//     stability_kernel (line-by-line mirror of plp::compute_fsr);
//   * blocks decompose into item boxes on thread 0 in CPU blocks-order
//     ((i,j,t) inner loops), so FSR index order == CPU == bit-parity.
// Per pass: 1 decode launch + 1 stability launch + 1 sync + 1 D2H (0 H2D —
// all instance/catalog tables uploaded once in init).
//
// Parity gate (gate 3.3 protocol): --single-pass makes a pass wall-time
// independent -> CPU (plp_block) and GPU JSONs must be byte-identical
// modulo the time_s field; 19 instances x 3 seeds, campaign config
// (--amax 4 --bmax 4 --kmax 6 --eps 0.5 --theta 0.9 --topk 64) AND the
// cp-147 default config.
//
// --pick: only 'first' (the winning campaign config, TRANSITION_BRIEF
// "Kalici parametre ... PICK=first"). 'support'/'contact' were tuning-round
// (cp-148) rules and stay CPU-only — the CLI rejects them explicitly.
//
// Build: make cuda (blockproof/Makefile; same parity-critical float flags
// as gpu/: nvcc --fmad=false, host part -ffp-contract=off). NEVER add
// -use_fast_math: bit parity requires exact IEEE-754 semantics.
//
// Capacity rules (host-verified in init, hard error otherwise; runtime err
// flags 2/4/5 catch dynamic overflows -> host aborts):
//   L,W,H < 2^21 (point packing); L*W*H < 2^25 (bvol sort key);
//   n_types <= 256 (ti sort field); orientations <= 6 <= 8 (oi field);
//   1 <= amax,bmax,kmax <= 7 (3-bit sort fields);
//   sum(qty) <= 512 (block/item/plan caps, FSR caps);
//   H <= 512 (FSR z-count array); catalogue <= 16384 (catidx sort field).

#include "../gpu/plp_common.hpp"

#include <cuda_runtime.h>

namespace {

// ---------------------------------------------------------------------------
// Device data mirrors (identical standard-layout to the host structs used by
// plp_block.cpp: BlockShape fields, PlacedBlock, PlacedBox).
// ---------------------------------------------------------------------------

struct DShape {            // catalogue entry == plp_block.cpp BlockShape
    int ti, oi, a, b, k;
    int cnt;               // a*b*k <= qty <= 512 (int)
    int bw, bd, bh;        // a*w, b*d, k*h
};
struct DBlk {              // placed block == plp_block.cpp PlacedBlock
    int x, y, z, w, d, h;  // whole-block box (C1/C2/C3 unit; w=bw etc.)
    int ti;
    int a, b, k;
};
struct DPlaced { int x, y, z, w, d, h, item_type; };   // item box (decomposed)

static_assert(sizeof(DShape) == 9 * sizeof(int), "pure-data layout");
static_assert(sizeof(DBlk) == 10 * sizeof(int), "pure-data layout");
static_assert(sizeof(DPlaced) == 7 * sizeof(int), "layout must match PlacedBox");

__device__ inline long long dminll(long long a, long long b) { return a < b ? a : b; }
__device__ inline long long dmaxll(long long a, long long b) { return a < b ? b : a; }

constexpr int DEV_BLOCK = 1024;
constexpr int DEV_CAP_CAND = 5120;           // cand cap; overflow -> err flag
constexpr uint64_t DEV_MASK21 = 0x1FFFFFULL; // 21 bits per coordinate
constexpr int NT_MAX = 256;                  // n_types cap
constexpr int BLK_CAP = 512;                 // blocks == item cap per row (sum qty)
constexpr int PLAN_CAP = 512;                // plan cap (bitonic size; sum qty)
constexpr int K_MAX = 16;                    // batched rows (campaign 12 seeds)
constexpr int DEV_FSR_CAP = 512;             // stability_kernel n and H caps
constexpr int NCHUNK_MAX = (DEV_CAP_CAND + DEV_BLOCK - 1) / DEV_BLOCK;  // 5

// Dynamic shared layout (bytes): s_bits 5120 | s_cand 40960 | s_cmerge 41024
// (offsets 8-aligned for uint64). sm_121 (Spark GB10) optin 101376 B per
// block (measured cp-14x); static below adds ~11 KB -> fits with margin.
constexpr int DEV_SHARED_BYTES =
    DEV_CAP_CAND /*bits u8*/ + DEV_CAP_CAND * 8 /*cand*/ +
    (DEV_CAP_CAND + 8) * 8 /*cmerge*/;

// Packed (z,y,x) key: numeric order == (z,y,x) lexicographic (sort_dedup_pt).
static __host__ __device__ inline uint64_t pack_pt(int x, int y, int z) {
    return ((uint64_t)(unsigned)z << 42) | ((uint64_t)(unsigned)y << 21)
           | (uint64_t)(unsigned)x;
}

// Per-row persistent device state (pure data; row k = d_rows + k), mirrors
// V0 DevRow (plp_gpu.cu) — cand survives passes; rem/blocks/slots reset per
// pass (block_pass fresh locals in plp_block.cpp).
struct BlockRow {
    uint32_t mt[624];
    int mti;
    int n_cand;
    uint64_t cand[DEV_CAP_CAND];             // packed (z,y,x), sorted+unique
    uint64_t cand_merge[DEV_CAP_CAND + 8];   // scratch: survivors / merge
};

// Result slot (D2H'd with the K-row slab once per pass):
// {n_placed, err, score, placed[0..n)} — item boxes, CPU decompose order.
struct DevSlot {
    int n_placed;
    int err;
    double score;
    DPlaced placed[1];
};

// ---------------------------------------------------------------------------
// Device MT19937 — line-by-line mirror of plp::PyRng (plp_common.hpp:162),
// copied from plp_gpu.cu (frozen there; integer-exact, --fmad=false).
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

__device__ inline int mt_randrange(uint32_t* mt, int* mti, int n) {
    int k = 32 - __clz((unsigned)n);
    int r = (int)(mt_genrand32(mt, mti) >> (32 - k));
    while (r >= n) r = (int)(mt_genrand32(mt, mti) >> (32 - k));
    return r;
}

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

__global__ void dev_rng_init_kernel(BlockRow* __restrict__ drows, uint64_t seed) {
    if (threadIdx.x != 0) return;
    BlockRow* row = drows + blockIdx.x;
    mt_init(row->mt, &row->mti, seed + (uint64_t)blockIdx.x);
}

__global__ void dev_rngdump_kernel(const BlockRow* __restrict__ drows,
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
// Block-box feasibility — line-by-line mirror of plp::check_feasible_one
// (plp_common.hpp:295) reading the PlacedBlock array (CPU pushes a FastBox
// per placed block; b.z2 == b.z + b.h). One thread per candidate.
// ---------------------------------------------------------------------------
__device__ bool check_block_dev(int x, int y, int z, int w, int d, int h,
                                const DBlk* fast, int n_fast,
                                int L, int W, int H) {
    if (x + w > L || y + d > W || z + h > H) return false;               // C2
    if (z > 0) {
        long long s = 0;
        for (int i = 0; i < n_fast; i++) {
            const DBlk& b = fast[i];
            if (b.z + b.h == z) {
                long long ox = dminll((long long)x + w, (long long)b.x + b.w)
                               - (long long)dmaxll(x, b.x);
                if (ox > 0) {
                    long long oy = dminll((long long)y + d, (long long)b.y + b.d)
                                   - (long long)dmaxll(y, b.y);
                    if (oy > 0) s += ox * oy;
                }
            }
        }
        // Python: s / (w * d) < TAU  (int/int -> one correctly-rounded double)
        if ((double)s / (double)(w * d) < plp::TAU) return false;        // C3
    }
    for (int i = 0; i < n_fast; i++) {
        const DBlk& b = fast[i];
        if (x < b.x + b.w && b.x < x + w && y < b.y + b.d && b.y < y + d
            && z < b.z + b.h && b.z < z + h) return false;               // C1
    }
    return true;
}

// ---------------------------------------------------------------------------
// The device-driven block pass (one kernel launch per pass). Grid = K rows.
// Uniformity discipline (V0 Turn 7a/7b, guide B.22): every __syncthreads sits
// on a block-uniform branch (all conditions are shared values read after the
// previous sync; s_err gates work, never syncs), every shfl is full-mask +
// unconditional. RNG draws are thread-0 only, in exact CPU order.
// ---------------------------------------------------------------------------
__global__ void block_pass_kernel(BlockRow* __restrict__ drows,
                                  DBlk* __restrict__ dblks,
                                  char* __restrict__ dslots,
                                  size_t slot_stride,
                                  const DShape* __restrict__ cat,
                                  const unsigned long long* __restrict__ catkey,
                                  const int* __restrict__ bt_off,
                                  const int* __restrict__ bt,
                                  const int* __restrict__ qty,
                                  const int* __restrict__ orsok,
                                  const int* __restrict__ tord,
                                  int nt, int L, int W, int H,
                                  double eps, int topk) {
    extern __shared__ unsigned char dyn[];
    uint8_t*  s_bits   = dyn;                                    // [0) 5120
    uint64_t* s_cand   = reinterpret_cast<uint64_t*>(dyn + DEV_CAP_CAND);
    uint64_t* s_cmerge = reinterpret_cast<uint64_t*>(dyn + DEV_CAP_CAND * 9);

    __shared__ uint32_t s_mt[624];
    __shared__ int s_mti, s_err, s_ncand, s_nblk, s_nplan, s_ntypes;
    __shared__ int s_rem[NT_MAX], s_types[NT_MAX];
    __shared__ unsigned long long s_pkey[PLAN_CAP];
    __shared__ int s_wcnt[32], s_wmin[32], s_wsum[32], s_wbase[32],
        s_chunktot[NCHUNK_MAX];
    __shared__ int s_cnt, s_fmin;
    __shared__ int s_px, s_py, s_pz, s_selpick, s_idx, s_selwarp, s_gbase;
    __shared__ int s_cti, s_ccnt, s_bw, s_bd, s_bh;              // cur shape
    __shared__ int s_upd, s_any;
    __shared__ uint64_t s_corner[8];
    __shared__ int s_kept[8], s_cout[8], s_nsurv;

    const int tid = threadIdx.x;
    const int w = tid >> 5;
    const int lane = tid & 31;
    BlockRow* row = drows + blockIdx.x;
    DBlk* rblks = dblks + (size_t)blockIdx.x * BLK_CAP;
    // slot base 8-aligned (slot_bytes() pads) -> `double score` at offset 8
    // is aligned for EVERY row (V0 Turn-12 misaligned-address lesson).
    char* slot = dslots + (size_t)blockIdx.x * slot_stride;
    int* slot_nplaced = reinterpret_cast<int*>(slot);
    int* slot_err = slot_nplaced + 1;
    DPlaced* slot_placed =
        reinterpret_cast<DPlaced*>(slot + 2 * sizeof(int) + sizeof(double));

    if (tid < 624) s_mt[tid] = row->mt[tid];
    if (tid == 0) {
        s_mti = row->mti;
        s_err = 0;
        s_nblk = 0;
        s_ncand = 1;
        s_cand[0] = pack_pt(0, 0, 0);
        *slot_nplaced = 0;
        *slot_err = 0;
        // per-pass reset (block_pass fresh locals: rem/cand/blocks/fast)
        row->n_cand = 1;
        row->cand[0] = pack_pt(0, 0, 0);
    }
    if (tid < nt) s_rem[tid] = qty[tid];
    __syncthreads();

    int nblk_reg = 0;   // thread-0 counter (blocks); published to s_nblk

    for (;;) {
        // ---- plan: types filter (thread 0 over the host preset order) ----
        if (tid == 0 && s_err == 0) {
            int tn = 0;
            for (int q = 0; q < nt; q++) {
                const int ti = tord[q];
                if (s_rem[ti] > 0 && orsok[ti]) s_types[tn++] = ti;
            }
            s_ntypes = tn;
        }
        __syncthreads();
        if (s_err != 0) break;
        if (s_ntypes == 0) break;                 // frozen: types.empty()

        // ---- plan: eps-shuffle + per-type decomposition (thread 0) ----
        if (tid == 0) {
            if (eps > 0.0 && mt_random(s_mt, &s_mti) >= 1.0 - eps) {
                for (int i = s_ntypes - 1; i > 0; i--) {
                    const int j = mt_randrange(s_mt, &s_mti, i + 1);
                    const int tmp = s_types[i]; s_types[i] = s_types[j]; s_types[j] = tmp;
                }
            }
            int m = 0;
            for (int q = 0; q < s_ntypes && s_err == 0; q++) {
                const int ti = s_types[q];
                int r = s_rem[ti];
                const int b0 = bt_off[ti], b1 = bt_off[ti + 1];
                const int sz = b1 - b0;
                while (r > 0) {
                    // shp is cnt-DESC: fitting shapes form a suffix — first
                    // index with cnt <= r (binary search; CPU walks back).
                    int lo = 0, hi = sz;
                    while (lo < hi) {
                        const int mid = (lo + hi) >> 1;
                        if (cat[bt[b0 + mid]].cnt <= r) hi = mid; else lo = mid + 1;
                    }
                    const int start = lo;
                    if (start == sz) break;       // unreachable while singles exist
                    int tk = sz - start;
                    if (tk > topk) tk = topk;
                    const int pick =
                        (eps > 0.0 && mt_random(s_mt, &s_mti) >= 1.0 - eps)
                            ? mt_randrange(s_mt, &s_mti, tk) : 0;
                    const int idx = bt[b0 + start + pick];
                    if (m >= PLAN_CAP) { s_err = 5; *slot_err = 5; break; }
                    s_pkey[m++] = catkey[idx];    // low bits carry catidx
                    r -= cat[idx].cnt;
                }
            }
            s_nplan = m;
            // pad the bitonic field with +inf keys (sort to the tail)
            for (int i = m; i < PLAN_CAP; i++) s_pkey[i] = ~0ull;
        }
        __syncthreads();
        if (s_err != 0) break;
        if (s_nplan == 0) break;                  // frozen: plan.empty()

        // ---- parallel bitonic sort of the plan keys (fixed network +
        // strict total order (bvol desc, ti, oi, k, a, b, catidx) == unique
        // result == std::sort outcome on the CPU) ----
        // Direction bit (tid & size2): left half ascending, right half
        // descending — without it the XOR network mis-sorts certain patterns
        // (cp-150 parity gate: always-ascending failed 294/300 random plans).
        for (int size2 = 2; size2 <= PLAN_CAP; size2 <<= 1) {
            for (int stride = size2 >> 1; stride > 0; stride >>= 1) {
                if (tid < PLAN_CAP) {
                    const int j = tid ^ stride;
                    if (tid < j) {
                        const unsigned long long a = s_pkey[tid], b = s_pkey[j];
                        const bool asc = (tid & size2) == 0;
                        if (asc ? (b < a) : (a < b)) { s_pkey[tid] = b; s_pkey[j] = a; }
                    }
                }
                __syncthreads();
            }
        }
        if (tid == 0) s_any = 0;
        __syncthreads();

        // ---- placement: plan order (sorted), candidate-major feasibility ----
        for (int pi = 0; pi < s_nplan; pi++) {
            const int idx = (int)(s_pkey[pi] & 0x3FFFu);
            if (tid == 0 && s_err == 0) {         // publish cur shape
                const DShape s = cat[idx];
                s_cti = s.ti; s_ccnt = s.cnt; s_bw = s.bw; s_bd = s.bd; s_bh = s.bh;
            }
            __syncthreads();
            if (s_err != 0) continue;
            if (s_rem[s_cti] < s_ccnt) continue;  // frozen: stale plan entry

            // feasibility scan — one thread per candidate (n_ors == 1, pair
            // order == candidate order == the frozen feas list order).
            const int n_cand = s_ncand;
            for (int t = tid; t < n_cand; t += DEV_BLOCK) {
                const uint64_t key = s_cand[t];
                s_bits[t] = (uint8_t)((s_err == 0 && check_block_dev(
                    (int)(key & DEV_MASK21),
                    (int)((key >> 21) & DEV_MASK21),
                    (int)(key >> 42),
                    s_bw, s_bd, s_bh, rblks, s_nblk, L, W, H)) ? 1 : 0);
            }
            __syncthreads();

            // parallel filter (count + first): cnt = |feas|, fmin = min set
            // index; adaptive chunk count (cand <= DEV_CAP_CAND).
            const int nchunk = (n_cand + DEV_BLOCK - 1) / DEV_BLOCK;
            int tot_cnt = 0, tot_min = 0x7fffffff;   // thread-0 running
            for (int c = 0; c < nchunk; c++) {
                const int i = c * DEV_BLOCK + tid;
                const int b = (i < n_cand) ? (int)s_bits[i] : 0;
                int tcnt = b;
                int tmin = b ? i : 0x7fffffff;
                for (int o = 16; o; o >>= 1) {
                    tcnt += __shfl_down_sync(0xffffffffu, tcnt, o);
                    const int u = __shfl_down_sync(0xffffffffu, tmin, o);
                    if (u < tmin) tmin = u;
                }
                if (lane == 0) { s_wcnt[w] = tcnt; s_wmin[w] = tmin; }
                __syncthreads();                   // read-after-write on s_w*
                if (tid == 0) {
                    for (int x = 0; x < 32; x++) {
                        tot_cnt += s_wcnt[x];
                        if (s_wmin[x] < tot_min) tot_min = s_wmin[x];
                    }
                }
                __syncthreads();                   // next chunk overwrites s_w*
            }
            if (tid == 0) { s_cnt = tot_cnt; s_fmin = tot_min; }
            __syncthreads();
            if (s_cnt == 0) continue;              // frozen: feas.empty()

            // pick (frozen): feas[0], or feas[idx] with idx = randrange(cnt)
            if (tid == 0) {
                const uint64_t fkey = s_cand[s_fmin];
                s_px = (int)(fkey & DEV_MASK21);
                s_py = (int)((fkey >> 21) & DEV_MASK21);
                s_pz = (int)(fkey >> 42);
                s_selpick = 0;
                if (s_err == 0 && eps > 0.0 && mt_random(s_mt, &s_mti) >= 1.0 - eps) {
                    const int idxr = mt_randrange(s_mt, &s_mti, s_cnt);
                    if (idxr > 0) { s_idx = idxr; s_selpick = 1; }
                }
            }
            __syncthreads();
            if (s_selpick) {
                // Turn-7b select of the s_idx-th set bit of s_bits[0..n_cand)
                // == host feas[idx]. Hierarchy: group j = contiguous pairs
                // [1024j, 1024j+1024), then warp, then lane.
                const int nJ = (n_cand + DEV_BLOCK - 1) / DEV_BLOCK;
                for (int j = 0; j < nJ; j++) {
                    const int p = tid + DEV_BLOCK * j;
                    int c2 = (p < n_cand) ? (int)s_bits[p] : 0;
                    for (int o = 16; o; o >>= 1)
                        c2 += __shfl_down_sync(0xffffffffu, c2, o);
                    if (lane == 0) s_wcnt[w] = c2;
                    __syncthreads();
                    if (tid == 0) {
                        int gt = 0;
                        for (int x = 0; x < 32; x++) gt += s_wcnt[x];
                        s_wsum[j] = gt;
                    }
                    __syncthreads();
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
                __syncthreads();
                const int G = s_selwarp >> 5;
                {
                    const int p = G * DEV_BLOCK + tid;
                    int c2 = (p < n_cand) ? (int)s_bits[p] : 0;
                    for (int o = 16; o; o >>= 1)
                        c2 += __shfl_down_sync(0xffffffffu, c2, o);
                    if (lane == 0) s_wcnt[w] = c2;
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
                    if (s_idx - s_gbase >= base && s_idx - s_gbase < base + x)
                        s_selwarp = (G << 5) | (int)lane;   // register G (V0 race note)
                }
                __syncthreads();
                const int Wt = s_selwarp & 0x1f;
                if (w == Wt) {
                    const int p = G * DEV_BLOCK + tid;
                    const int b = (p < n_cand) ? (int)s_bits[p] : 0;
                    const int oin = s_idx - s_gbase - s_wbase[w];
                    int pre = b;
                    for (int d2 = 1; d2 < 32; d2 <<= 1) {
                        const int u = __shfl_up_sync(0xffffffffu, pre, d2);
                        if (lane >= d2) pre += u;
                    }
                    if (b && pre - b == oin) {     // the s_idx-th set bit
                        const uint64_t key = s_cand[p];
                        s_px = (int)(key & DEV_MASK21);
                        s_py = (int)((key >> 21) & DEV_MASK21);
                        s_pz = (int)(key >> 42);
                    }
                }
                __syncthreads();                   // final s_px..s_pz visible
            }

            // place (thread 0): block record, rem, corners; parallel cand
            // update (frozen Turn-7a: swallow strictly-inside + add 8 corners).
            if (tid == 0) {
                if (nblk_reg >= BLK_CAP) {
                    s_err = 4; *slot_err = 4; s_upd = 0;
                } else if (s_err != 0) {
                    s_upd = 0;
                } else {
                    const int px = s_px, py = s_py, pz = s_pz;
                    rblks[nblk_reg] = DBlk{px, py, pz, s_bw, s_bd, s_bh,
                                           s_cti, cat[idx].a, cat[idx].b, cat[idx].k};
                    nblk_reg++;
                    s_nblk = nblk_reg;
                    s_rem[s_cti] -= s_ccnt;
                    s_any = 1;
                    s_upd = (s_ncand + 8 <= DEV_CAP_CAND) ? 1 : 0;
                    if (!s_upd) { s_err = 2; *slot_err = 2; }
                    int nc = 0;
                    for (int a2 = 0; a2 < 2; a2++)
                        for (int b2 = 0; b2 < 2; b2++)
                            for (int c2 = 0; c2 < 2; c2++)
                                s_corner[nc++] = pack_pt(px + a2 * s_bw,
                                                         py + b2 * s_bd,
                                                         pz + c2 * s_bh);
                    for (int i = 1; i < 8; i++) {   // insertion-sort the 8 keys
                        const uint64_t v = s_corner[i];
                        int j = i - 1;
                        while (j >= 0 && s_corner[j] > v) { s_corner[j + 1] = s_corner[j]; j--; }
                        s_corner[j + 1] = v;
                    }
                }
            }
            __syncthreads();                       // publish pick/corners/s_upd
            if (s_upd) {
                // 1) mark + scan + compact (adaptive chunks, work-efficient)
                for (int c = 0; c < nchunk; c++) {
                    const int i = c * DEV_BLOCK + tid;
                    uint64_t key = 0;
                    int surv = 0;
                    if (i < n_cand) {
                        key = s_cand[i];
                        const int x = (int)(key & DEV_MASK21);
                        const int y = (int)((key >> 21) & DEV_MASK21);
                        const int z = (int)(key >> 42);
                        surv = !(s_px < x && x < s_px + s_bw
                                 && s_py < y && y < s_py + s_bd
                                 && s_pz < z && z < s_pz + s_bh);
                    }
                    int v = surv;
                    for (int o = 1; o < 32; o <<= 1) {
                        const int u = __shfl_up_sync(0xffffffffu, v, o);
                        if (lane >= o) v += u;
                    }
                    if (lane == 31) s_wsum[w] = v;
                    __syncthreads();
                    if (w == 0) {
                        int x = s_wsum[lane];
                        int inc = x;
                        for (int o = 1; o < 32; o <<= 1) {
                            const int u = __shfl_up_sync(0xffffffffu, inc, o);
                            if (lane >= o) inc += u;
                        }
                        s_wbase[lane] = inc - x;
                        if (lane == 31) s_chunktot[c] = inc;
                    }
                    __syncthreads();
                    int base = 0;
                    for (int c2 = 0; c2 < c; c2++) base += s_chunktot[c2];
                    if (surv) s_cmerge[base + s_wbase[w] + v - 1] = key;
                    __syncthreads();
                }
                if (tid == 0) {
                    int t = 0;
                    for (int c = 0; c < nchunk; c++) t += s_chunktot[c];
                    s_nsurv = t;
                }
                __syncthreads();
                // 2) corners (lanes 0-7): lower_bound + dedup + ranks
                if (w == 0 && lane < 8) {
                    int lo = 0, hi = s_nsurv;
                    while (lo < hi) {
                        const int mid = (lo + hi) >> 1;
                        if (s_cmerge[mid] < s_corner[lane]) lo = mid + 1;
                        else hi = mid;
                    }
                    const int pos = lo;
                    const int eq = (pos < s_nsurv) && (s_cmerge[pos] == s_corner[lane]);
                    int kb = eq ? 0 : 1;
                    for (int d2 = 1; d2 < 8; d2 <<= 1) {
                        const int u = __shfl_up_sync(0xffu, kb, d2);
                        if (lane >= d2) kb += u;
                    }
                    s_kept[lane] = eq ? 0 : 1;
                    s_cout[lane] = pos + (kb - (eq ? 0 : 1));
                }
                __syncthreads();
                // 3) rank writes (result == sort_dedup_pt, canonical)
                for (int j2 = tid; j2 < s_nsurv; j2 += DEV_BLOCK) {
                    const uint64_t key = s_cmerge[j2];
                    int out2 = j2;
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        if (s_kept[j] && s_corner[j] < key) out2++;
                    s_cand[out2] = key;
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
            __syncthreads();                       // publish s_ncand/s_nblk
        }
        if (!s_any) break;          // frozen: no sweep progress -> leftovers
    }

    // ---- decompose blocks into item boxes (CPU blocks-order; FSR parity) ----
    if (tid == 0 && s_err == 0) {
        int ni = 0;
        for (int b = 0; b < nblk_reg; b++) {
            const DBlk& d = rblks[b];
            const int iw = d.w / d.a, id = d.d / d.b, ih = d.h / d.k;
            for (int i = 0; i < d.a; i++)
                for (int j = 0; j < d.b; j++)
                    for (int t = 0; t < d.k; t++)
                        slot_placed[ni++] = DPlaced{d.x + i * iw, d.y + j * id,
                                                     d.z + t * ih,
                                                     iw, id, ih, d.ti};
        }
        *slot_nplaced = ni;
    }
    __syncthreads();
    // persistent row state: cand + RNG back to global
    for (int i = 0; i < s_ncand; i += DEV_BLOCK)
        row->cand[i] = s_cand[i];
    if (tid < 624) row->mt[tid] = s_mt[tid];
    if (tid == 0) {
        row->mti = s_mti;
        row->n_cand = s_ncand;
    }
}

// ---------------------------------------------------------------------------
// Turn-9 device FSR (plp_gpu.cu, verbatim; DEV_FSR_CAP 512 for qty up to 391
// on BR0_050). Line-by-line mirror of plp::compute_fsr + stability_score;
// item boxes (decomposed) so fill%/score mean exactly the campaign values.
// ---------------------------------------------------------------------------
__global__ void stability_kernel(char* dslots, int H, size_t slot_stride) {
    DevSlot* slot =
        reinterpret_cast<DevSlot*>(dslots + (size_t)blockIdx.x * slot_stride);
    const int tid = threadIdx.x;
    const int n = slot->n_placed;

    __shared__ double s_m[DEV_FSR_CAP];
    __shared__ double s_load[DEV_FSR_CAP];
    __shared__ long long s_area[DEV_FSR_CAP];
    __shared__ int s_ord[DEV_FSR_CAP];
    __shared__ int s_z[DEV_FSR_CAP];
    __shared__ int s_cnt[DEV_FSR_CAP];
    __shared__ int s_off[DEV_FSR_CAP + 1];
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

    if (tid == 0) {                        // stable (z desc, index asc) order
        for (int i = 0; i < n; i++) s_cnt[s_z[i]]++;
        s_off[H] = 0;
        for (int z = H - 1; z >= 0; z--) s_off[z] = s_off[z + 1] + s_cnt[z];
        for (int z = 0; z < H; z++) s_cnt[z] = 0;
        for (int i = 0; i < n; i++) {
            int z = s_z[i];
            s_ord[s_off[z + 1] + s_cnt[z]++] = i;
        }
    }
    __syncthreads();

    for (int kk = 0; kk < n; kk++) {       // serial outer, parallel j-scan
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
        if (have) s_area[tid] = a;
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

    if (have) {
        double fsr = (plp::MU / plp::A_LAT_G) * s_load[tid] / s_m[tid];
        s_flag[tid] = (fsr >= 1.0) ? 1 : 0;
    }
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int i = 0; i < n; i++) c += s_flag[i];
        slot->score = (double)c / (double)n;
    }
}

// ---------------------------------------------------------------------------
// Host-side mirrors of the CPU plan tables (plp_block.cpp) — same loops, so
// the catalogue/ordering is bit-equal by construction.
// ---------------------------------------------------------------------------

struct BlockShape {
    int ti, oi;
    int a, b, k;
    long long cnt;
    int bw, bd, bh;
    long long bvol() const { return 1LL * bw * bd * bh; }
};

inline bool shape_precedence(const BlockShape& x, const BlockShape& y) {
    if (x.cnt != y.cnt) return x.cnt > y.cnt;
    if (x.bh != y.bh) return x.bh > y.bh;
    if (x.a != y.a) return x.a > y.a;
    if (x.b != y.b) return x.b > y.b;
    return x.oi < y.oi;
}

struct CudaBlockBackend {
    BlockRow* d_rows = nullptr;
    DBlk* d_blks = nullptr;
    char* d_slots = nullptr;
    DShape* d_cat = nullptr;
    unsigned long long* d_key = nullptr;
    int* d_bt_off = nullptr;
    int* d_bt = nullptr;
    int* d_qty = nullptr;
    int* d_orsok = nullptr;
    int* d_tord = nullptr;
    int K = 0, nt = 0, L = 0, W = 0, H = 0;
    double eps = 0.0;
    int topk = 8;

    static void check(const char* what, cudaError_t e) {
        if (e == cudaSuccess) return;
        std::fprintf(stderr, "cuda %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }

    size_t slot_bytes() const {
        const size_t s = 2 * sizeof(int) + sizeof(double)
                        + (size_t)BLK_CAP * sizeof(DPlaced);
        return (s + 7) & ~(size_t)7;
    }

    [[noreturn]] static void die(const char* msg) {
        std::fprintf(stderr, "plp_block_cuda: %s\n", msg);
        std::exit(2);
    }

    void init(const plp::Instance& inst, int seed, int k, double eps_,
              int topk_, int amax, int bmax, int kmax) {
        K = k;
        L = inst.L; W = inst.W; H = inst.H;
        eps = eps_;
        topk = topk_;
        nt = (int)inst.items.size();

        // ---- static capacity rules (host, hard errors) ----
        if (L <= 0 || W <= 0 || H <= 0) die("bad container dims");
        if (L > 0x1FFFFF || W > 0x1FFFFF || H > 0x1FFFFF)
            die("container dims exceed the 21-bit pack");
        if (1LL * L * W * H >= (1LL << 25))
            die("container volume exceeds the 25-bit bvol key (L*W*H >= 2^25)");
        if (nt < 1 || nt > NT_MAX) die("n_types > 256 (sort key field)");
        if (H > DEV_FSR_CAP) die("H exceeds FSR cap");
        long long qty_total = 0;
        for (const auto& it : inst.items) qty_total += it.qty;
        if (qty_total > BLK_CAP) die("sum(qty) > 512 (block/item/plan caps)");
        if (amax < 1 || amax > 7 || bmax < 1 || bmax > 7 || kmax < 1 || kmax > 7)
            die("amax/bmax/kmax must be in [1,7] on the device build");
        if (topk < 1) die("topk must be >= 1");

        std::vector<std::vector<plp::Pt>> orients;
        orients.reserve(nt);
        for (const auto& it : inst.items) orients.push_back(plp::orientations(it));
        std::vector<int> qty(nt), orsok(nt), tord(nt);
        for (int t = 0; t < nt; t++) {
            qty[t] = inst.items[t].qty;
            orsok[t] = orients[t].empty() ? 0 : 1;
            tord[t] = t;
        }
        std::sort(tord.begin(), tord.end(), [&](int x, int y) {
            long long vx = inst.items[x].volume(), vy = inst.items[y].volume();
            if (vx != vy) return vx > vy;
            return x < y;
        });

        // ---- catalogue (identical loops to plp_block.cpp run_block) ----
        std::vector<BlockShape> cat;
        for (size_t ti = 0; ti < inst.items.size(); ti++) {
            for (size_t oi = 0; oi < orients[ti].size(); oi++) {
                const int w = orients[ti][oi][0], d = orients[ti][oi][1],
                            h = orients[ti][oi][2];
                for (int a = 1; a <= amax; a++) {
                    if (a * w > inst.L) break;
                    for (int b = 1; b <= bmax; b++) {
                        if (b * d > inst.W) break;
                        for (int kk = 1; kk <= kmax; kk++) {
                            if (kk * h > inst.H) break;
                            long long cnt = 1LL * a * b * kk;
                            if (cnt > inst.items[ti].qty) break;
                            cat.push_back({(int)ti, (int)oi, a, b, kk, cnt,
                                           a * w, b * d, kk * h});
                        }
                    }
                }
            }
        }
        if (cat.size() > 16384) die("catalogue exceeds 16384 (catidx key field)");
        std::vector<std::vector<int>> by_type(nt);
        for (size_t i = 0; i < cat.size(); i++) by_type[cat[i].ti].push_back((int)i);
        for (auto& v : by_type)
            std::sort(v.begin(), v.end(),
                      [&](int x, int y) { return shape_precedence(cat[x], cat[y]); });

        // ---- device catalogue + sort keys (bvol desc, ti, oi, k desc,
        // a desc, b desc, catidx — strict total order, bitonic-stable) ----
        std::vector<DShape> hcat(cat.size());
        std::vector<unsigned long long> hkey(cat.size());
        for (size_t i = 0; i < cat.size(); i++) {
            const BlockShape& s = cat[i];
            hcat[i] = DShape{s.ti, s.oi, s.a, s.b, s.k, (int)s.cnt,
                             s.bw, s.bd, s.bh};
            const long long bv = s.bvol();
            if (bv >= (1LL << 25)) die("block volume exceeds the 25-bit key");
            hkey[i] = (((unsigned long long)((1LL << 25) - 1 - bv)) << 34)
                    | ((unsigned long long)s.ti << 26)
                    | ((unsigned long long)s.oi << 23)
                    | ((unsigned long long)(7 - s.k) << 20)
                    | ((unsigned long long)(7 - s.a) << 17)
                    | ((unsigned long long)(7 - s.b) << 14)
                    | (unsigned long long)i;
        }
        std::vector<int> bt_off(nt + 1, 0), bt;
        for (int t = 0; t < nt; t++) {
            bt_off[t] = (int)bt.size();
            for (int idx : by_type[t]) bt.push_back(idx);
        }
        bt_off[nt] = (int)bt.size();

        check("shared optin", cudaFuncSetAttribute(block_pass_kernel,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              (int)DEV_SHARED_BYTES));
        check("rows malloc", cudaMalloc(&d_rows, (size_t)K * sizeof(BlockRow)));
        check("blks malloc",
              cudaMalloc(&d_blks, (size_t)K * BLK_CAP * sizeof(DBlk)));
        check("slots malloc", cudaMalloc(&d_slots, (size_t)K * slot_bytes()));
        check("cat malloc", cudaMalloc(&d_cat, cat.size() * sizeof(DShape)));
        check("key malloc", cudaMalloc(&d_key, cat.size() * sizeof(unsigned long long)));
        check("btoff malloc", cudaMalloc(&d_bt_off, (nt + 1) * sizeof(int)));
        check("bt malloc", cudaMalloc(&d_bt, std::max<size_t>(1, bt.size()) * sizeof(int)));
        check("qty malloc", cudaMalloc(&d_qty, nt * sizeof(int)));
        check("orsok malloc", cudaMalloc(&d_orsok, nt * sizeof(int)));
        check("tord malloc", cudaMalloc(&d_tord, nt * sizeof(int)));
        check("h2d cat", cudaMemcpy(d_cat, hcat.data(), cat.size() * sizeof(DShape),
                                    cudaMemcpyHostToDevice));
        check("h2d key", cudaMemcpy(d_key, hkey.data(),
                                    cat.size() * sizeof(unsigned long long),
                                    cudaMemcpyHostToDevice));
        check("h2d btoff", cudaMemcpy(d_bt_off, bt_off.data(),
                                      (nt + 1) * sizeof(int), cudaMemcpyHostToDevice));
        if (!bt.empty())
            check("h2d bt", cudaMemcpy(d_bt, bt.data(), bt.size() * sizeof(int),
                                       cudaMemcpyHostToDevice));
        check("h2d qty", cudaMemcpy(d_qty, qty.data(), nt * sizeof(int),
                                    cudaMemcpyHostToDevice));
        check("h2d orsok", cudaMemcpy(d_orsok, orsok.data(), nt * sizeof(int),
                                      cudaMemcpyHostToDevice));
        check("h2d tord", cudaMemcpy(d_tord, tord.data(), nt * sizeof(int),
                                     cudaMemcpyHostToDevice));
        std::vector<BlockRow> host_rows(K);
        for (int r = 0; r < K; r++) { host_rows[r].n_cand = 1; host_rows[r].cand[0] = pack_pt(0, 0, 0); }
        check("h2d rows", cudaMemcpy(d_rows, host_rows.data(),
                                     (size_t)K * sizeof(BlockRow), cudaMemcpyHostToDevice));
        check("memset slots", cudaMemset(d_slots, 0, (size_t)K * slot_bytes()));
        dev_rng_init_kernel<<<K, 32>>>(d_rows, (uint64_t)seed);
        check("rng init", cudaGetLastError());
        check("rng init sync", cudaDeviceSynchronize());
    }

    // One pass: 2 launches (decode + stability) + 1 sync + 1 D2H in fetch.
    void run_pass() {
        const size_t slot_stride = slot_bytes();
        block_pass_kernel<<<K, DEV_BLOCK, DEV_SHARED_BYTES>>>(
            d_rows, d_blks, d_slots, slot_stride, d_cat, d_key, d_bt_off, d_bt,
            d_qty, d_orsok, d_tord, nt, L, W, H, eps, topk);
        check("block kernel", cudaGetLastError());
        stability_kernel<<<K, DEV_BLOCK>>>(d_slots, H, slot_stride);
        check("stability", cudaGetLastError());
        check("sync", cudaDeviceSynchronize());
    }

    void fetch(std::vector<std::vector<plp::PlacedBox>>& out,
               std::vector<double>& scores, int& err_any) {
        std::vector<unsigned char> host((size_t)K * slot_bytes());
        check("d2h slots", cudaMemcpy(host.data(), d_slots, host.size(),
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

    ~CudaBlockBackend() {
        cudaFree(d_rows); cudaFree(d_blks); cudaFree(d_slots);
        cudaFree(d_cat); cudaFree(d_key); cudaFree(d_bt_off); cudaFree(d_bt);
        cudaFree(d_qty); cudaFree(d_orsok); cudaFree(d_tord);
    }
};

// Mirror of plp_block.cpp run_block for K device-driven rows: row k =
// PyRng(seed + k); ONLINE best (best_acc/best_all, strict > keeps earliest,
// theta accept, fallback to overall best) on the host, exactly as the frozen
// CPU engine (the block engine keeps running maxima — no packings vector).
std::vector<plp::RunResult> run_block_device(const plp::Instance& inst, int seed,
                                             int K, double time_limit_s,
                                             double eps, double* theta,
                                             bool single_pass, int topk,
                                             int amax, int bmax, int kmax,
                                             CudaBlockBackend& be) {
    be.init(inst, seed, K, eps, topk, amax, bmax, kmax);
    auto t0 = std::chrono::steady_clock::now();
    struct Best {
        long long volume = -1;
        int n = 0;
        double score = 0.0;
        std::vector<plp::PlacedBox> placed;
        bool better(long long v, int boxes) const {
            return v > volume || (v == volume && boxes > n);
        }
    };
    std::vector<Best> best_acc(K), best_all(K);
    std::vector<bool> has_acc(K, false);
    std::vector<std::vector<plp::PlacedBox>> pass_out;
    std::vector<double> scores;
    int k = 0;
    for (;;) {
        be.run_pass();
        k++;
        int err = 0;
        be.fetch(pass_out, scores, err);
        if (err != 0) {
            std::fprintf(stderr,
                         "device block-pass error flag %d (capacity overflow "
                         "— cand/blocks/plan?)\n", err);
            std::exit(1);
        }
        for (int r = 0; r < K; r++) {
            long long vol = 0;
            for (const auto& b : pass_out[r]) vol += 1LL * b.w * b.d * b.h;
            const double sc = scores[r];
            const bool upd_all = best_all[r].better(vol, (int)pass_out[r].size());
            const bool upd_acc = (theta == nullptr || sc >= *theta) &&
                                 (!has_acc[r] || best_acc[r].better(vol, (int)pass_out[r].size()));
            if (upd_acc) {
                best_acc[r] = Best{vol, (int)pass_out[r].size(), sc, pass_out[r]};
                has_acc[r] = true;
            }
            if (upd_all)
                best_all[r] = Best{vol, (int)pass_out[r].size(), sc, std::move(pass_out[r])};
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (single_pass || el >= time_limit_s) break;
    }
    std::vector<plp::RunResult> results(K);
    for (int r = 0; r < K; r++) {
        const Best& best = has_acc[r] ? best_acc[r] : best_all[r];
        plp::RunResult& res = results[r];
        for (const auto& b : best.placed)
            res.placements.push_back({b.item_type, b.x, b.y, b.z, b.w, b.d, b.h});
        res.n_boxes = best.n;
        res.volume = best.volume;
        res.score = best.score;
        res.fallback = !has_acc[r];
        res.n_passes = k;
        res.seed = seed + r;
        res.time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    return results;
}

// --rngdump N (V0 probe, device MT19937 vs frozen host PyRng, word-for-word).
void rngdump(int N) {
    if (N < 1) { std::fprintf(stderr, "--rngdump N must be >= 1\n"); std::exit(2); }
    BlockRow* d_rows = nullptr;
    double* d_rand = nullptr;
    uint32_t* d_gb = nullptr;
    CudaBlockBackend::check("rngdump rows", cudaMalloc(&d_rows, (size_t)N * sizeof(BlockRow)));
    CudaBlockBackend::check("rngdump memset", cudaMemset(d_rows, 0, (size_t)N * sizeof(BlockRow)));
    CudaBlockBackend::check("rngdump rand", cudaMalloc(&d_rand, (size_t)N * 16 * sizeof(double)));
    CudaBlockBackend::check("rngdump gb", cudaMalloc(&d_gb, (size_t)N * 16 * sizeof(uint32_t)));
    dev_rng_init_kernel<<<N, 32>>>(d_rows, 0);
    dev_rngdump_kernel<<<N, 32>>>(d_rows, d_rand, d_gb);
    CudaBlockBackend::check("rngdump launch", cudaGetLastError());
    CudaBlockBackend::check("rngdump sync", cudaDeviceSynchronize());
    std::vector<double> hr((size_t)N * 16);
    std::vector<uint32_t> hg((size_t)N * 16);
    CudaBlockBackend::check("rngdump d2h rand", cudaMemcpy(hr.data(), d_rand, hr.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CudaBlockBackend::check("rngdump d2h gb", cudaMemcpy(hg.data(), d_gb, hg.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
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
                std::fprintf(stderr, "row %d getrandbits[%d]: host %u dev %u\n", k, i, h, hg[(size_t)k * 16 + i]);
            }
        }
        std::printf("SEED %u: %s\n", (unsigned)k, row_ok ? "PASS" : "FAIL");
        ok = ok && row_ok;
    }
    cudaFree(d_rows); cudaFree(d_rand); cudaFree(d_gb);
    std::printf("RNGDUMP: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) std::exit(1);
}

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
                 "                 [--amax A] [--bmax B] [--kmax K] [--topk N]\n"
                 "                 [--pick first] [--single-pass] "
                 "[--device K] [--json]\n"
                 "       %s --selftest | --rngdump N\n"
                 "note: device build implements --pick first only (campaign "
                 "config; support/contact are CPU-only tuning rules).\n",
                 prog, prog);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace plp;
    std::string instance_path;
    int seed = 0, amax = 3, bmax = 3, kmax = 4;
    double time_s = 300.0, eps = EPS_QUALITY, theta = THETA;
    bool no_theta = false, single_pass = false, as_json = false;
    int topk = 8;
    int device = 1;   // K device rows (row k = PyRng(seed + k))

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
        else if (a == "--rngdump") { rngdump(std::atoi(next(a.c_str()).c_str())); return 0; }
        else if (a == "--seed") seed = std::atoi(next(a.c_str()).c_str());
        else if (a == "--time") time_s = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--eps") eps = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--theta") theta = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--no-theta") no_theta = true;
        else if (a == "--amax") amax = std::atoi(next(a.c_str()).c_str());
        else if (a == "--bmax") bmax = std::atoi(next(a.c_str()).c_str());
        else if (a == "--kmax") kmax = std::atoi(next(a.c_str()).c_str());
        else if (a == "--single-pass") single_pass = true;
        else if (a == "--topk") topk = std::max(1, std::atoi(next(a.c_str()).c_str()));
        else if (a == "--device") device = std::atoi(next(a.c_str()).c_str());
        else if (a == "--pick") {
            std::string v = next(a.c_str());
            if (v != "first") {
                std::fprintf(stderr,
                             "device build supports --pick first only "
                             "(got: %s)\n", v.c_str());
                return 2;
            }
        }
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
    amax = std::max(1, amax); bmax = std::max(1, bmax); kmax = std::max(1, kmax);
    if (device < 1 || device > K_MAX) {
        std::fprintf(stderr, "--device K must be in [1, %d]\n", K_MAX);
        return 2;
    }

    Instance inst;
    try {
        inst = load_instance(instance_path);
    } catch (const InstanceFormatError& e) {
        std::fprintf(stderr, "instance error: %s\n", e.what());
        return 1;
    }
    CudaBlockBackend be;
    auto results = run_block_device(inst, seed, device, time_s, eps,
                                    no_theta ? nullptr : &theta,
                                    single_pass, topk, amax, bmax, kmax, be);
    if (as_json) {
        if (device == 1) {
            std::cout << to_json(results[0]) << "\n";
            double fill = 100.0 * (double)results[0].volume
                        / (double)(1LL * inst.L * inst.W * inst.H);
            std::fprintf(stderr, "fill_pct=%.4f\n", fill);
        } else {
            std::cout << "{\"K\": " << device << ", \"n_passes\": "
                      << results[0].n_passes << ", \"time_s\": "
                      << py_repr(results[0].time_s) << ", \"rows\": [";
            for (size_t r = 0; r < results.size(); r++) {
                if (r) std::cout << ", ";
                std::cout << to_json(results[r]);
            }
            std::cout << "]}\n";
        }
    } else {
        for (int r = 0; r < device; r++) {
            const RunResult& rr = results[r];
            double fill = 100.0 * (double)rr.volume
                        / (double)(1LL * inst.L * inst.W * inst.H);
            std::printf("seed=%d n_boxes=%d volume_cm3=%lld fill_pct=%.4f "
                        "stability_score=%.4f fallback=%s n_passes=%d time_s=%.3f\n",
                        rr.seed, rr.n_boxes, (long long)rr.volume, fill, rr.score,
                        rr.fallback ? "true" : "false", rr.n_passes, rr.time_s);
        }
    }
    return 0;
}
