// plp_common.hpp — 3D-PLP decoder core (gate 3.3).
//
// Single source of truth shared by the host build (plp_gpu.cpp) and the CUDA
// build (plp_gpu.cu). Bit-compatible with the Python CPU baseline
// (../plp/solver.py, gate 3.2): with eps = 0 and one pass, placements,
// stability score (bit-equal double), and every deterministic JSON field must
// equal `python -m plp.solver --single-pass --eps 0 --no-theta --json`.
//
// Bit-compatibility rules honored here:
//  * floating-point expressions mirror the Python source 1:1 (same operand
//    order, IEEE-754 double, no FMA contraction — builds use -ffp-contract=off)
//  * integer arithmetic stays in integers (long long) until the exact spot
//    where Python converts to float
//  * the MT19937 stream mirrors CPython random.Random(int) bit for bit
//    (verified with --selftest against CPython)
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plp {

// Approved parameters (spec D3/D4'/D6/D7/D8) — identical to solver.py.
// MU is runtime-mutable (--mu / PLP_MU) for the binding-regime campaign only;
// default 0.2 keeps every locked Gate 3.4 code path bit-identical.
inline double MU = 0.2;
constexpr double A_LAT_G = 0.3;   // a_lat = 0.3 g; g cancels in FSR
constexpr double TAU = 0.5;
constexpr double THETA = 0.9;
constexpr double RHO = 50.0;      // kg/m3
constexpr double EPS_QUALITY = 0.3;

// ---------------------------------------------------------------------------
// Instance + parser (thpack "original" format, mirrors ../plp/parser.py)
// ---------------------------------------------------------------------------

struct Item {
    int l, w, h;
    bool lf, wf, hf;
    int qty;
    long long volume() const { return 1LL * l * w * h; }
};

struct Instance {
    int L, W, H;
    std::vector<Item> items;
};

class InstanceFormatError : public std::runtime_error {
public:
    explicit InstanceFormatError(const std::string& m) : std::runtime_error(m) {}
};

std::vector<Instance> parse(const std::string& text) {
    // whitespace-tokenize line by line (like parser.py: non-empty lines only)
    std::vector<std::vector<std::string>> toks;
    {
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            // strip trailing \r
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.find_first_not_of(" \t") == std::string::npos) continue;
            std::istringstream ls(line);
            std::vector<std::string> t;
            std::string w;
            while (ls >> w) t.push_back(w);
            if (!t.empty()) toks.push_back(std::move(t));
        }
    }
    if (toks.empty()) throw InstanceFormatError("empty file");
    auto to_int = [](const std::string& s, const char* what) -> long long {
        char* end = nullptr;
        long long v = std::strtoll(s.c_str(), &end, 10);
        if (end == s.c_str() || *end != '\0')
            throw InstanceFormatError(std::string("not an integer: ") + s + " (" + what + ")");
        return v;
    };
    long long P = to_int(toks[0][0], "instance count");
    if (P <= 0 || toks.size() < 2)
        throw InstanceFormatError("instance count must be >= 1");
    std::vector<Instance> out;
    size_t pos = 1;
    for (long long p = 0; p < P; p++) {
        if (pos >= toks.size() || toks[pos].size() > 2)
            throw InstanceFormatError("malformed problem line");
        pos++;
        if (pos >= toks.size() || toks[pos].size() != 3)
            throw InstanceFormatError("container line must have 3 dimensions");
        long long L = to_int(toks[pos][0], "container dimension");
        long long W = to_int(toks[pos][1], "container dimension");
        long long H = to_int(toks[pos][2], "container dimension");
        if (L <= 0 || W <= 0 || H <= 0)
            throw InstanceFormatError("container dimensions must be positive");
        pos++;
        if (pos >= toks.size() || toks[pos].size() != 1)
            throw InstanceFormatError("type-count line must be a single integer");
        long long n = to_int(toks[pos][0], "type count");
        pos++;
        std::vector<Item> items;
        for (long long k = 1; k <= n; k++) {
            if (pos >= toks.size())
                throw InstanceFormatError("instance truncated");
            const auto& t = toks[pos];
            pos++;
            if (t.size() != 8)
                throw InstanceFormatError("item line must have 8 tokens");
            if (to_int(t[0], "item index") != k)
                throw InstanceFormatError("item index must be sequential");
            long long dl = to_int(t[1], "item dimension");
            long long dw = to_int(t[3], "item dimension");
            long long dh = to_int(t[5], "item dimension");
            if (dl <= 0 || dw <= 0 || dh <= 0)
                throw InstanceFormatError("item dimensions must be positive");
            if ((t[2] != "0" && t[2] != "1") || (t[4] != "0" && t[4] != "1") || (t[6] != "0" && t[6] != "1"))
                throw InstanceFormatError("orientation flags must be 0/1");
            long long qty = to_int(t[7], "item qty");
            if (qty < 0) throw InstanceFormatError("item qty must be >= 0");
            items.push_back({(int)dl, (int)dw, (int)dh, t[2] == "1", t[4] == "1",
                             t[6] == "1", (int)qty});
        }
        out.push_back({(int)L, (int)W, (int)H, std::move(items)});
    }
    if (pos != toks.size())
        throw InstanceFormatError("trailing data after last instance");
    return out;
}

Instance load_instance(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw InstanceFormatError("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    auto v = parse(ss.str());
    if (v.size() != 1) throw InstanceFormatError("expected exactly 1 instance, got " + std::to_string(v.size()));
    return v[0];
}

// ---------------------------------------------------------------------------
// PyRng — bit-compatible with CPython random.Random(int seed)
//
//   seed:     init_by_64bit (CPython int-seed path: "all bits are used")
//   random(): a = next() >> 5 (27 bits), b = next() >> 19 (15 bits),
//             (a * 2^26 + b) / 2^53                      (MersenneTwister_random)
//   randrange(n): _randbelow_with_getrandbits — k = n.bit_length(),
//             r = next() >> (32-k), reject while r >= n
// Verified against CPython with --selftest (rng block).
// ---------------------------------------------------------------------------

class PyRng {
public:
    // CPython random.Random(int) [v3.11.15 _randommodule.c]: n = |seed|,
    // key = little-endian 32-bit chunks of n, init_by_array(key) — pre-seed
    // 19650218U, matsumoto's two passes, mt[0] = 0x80000000U. Verified word
    // for word against random.Random(s).getstate() for 8 seeds.
    explicit PyRng(uint64_t seed) {
        uint64_t n = seed;                 // project seeds are non-negative
        size_t bits = 0;
        for (uint64_t t = n; t; t >>= 1) bits++;
        size_t keyused = (bits == 0) ? 1 : ((bits - 1) / 32 + 1);
        std::vector<uint32_t> key(keyused);
        for (size_t j = 0; j < keyused; j++) key[j] = (uint32_t)(n >> (32 * j));
        init_by_array(key);
    }

    // CPython random_random (genrand_res53): a = g() >> 5 (27 bits),
    // b = g() >> 6 (26 bits), (a*2^26 + b) * 2^-53.
    double random() {
        uint32_t a = genrand32() >> 5;
        uint32_t b = genrand32() >> 6;
        return (a * 67108864.0 + (double)b) * (1.0 / 9007199254740992.0);
    }

    // CPython Random.randrange(n) for 0 < n (int stop form)
    int randrange(int n) {
        int k = 32 - __builtin_clz((unsigned)n);   // bit_length(n), n > 0
        int r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    // Self-test hooks
    uint32_t getrandbits(int k) { return genrand32() >> (32 - k); }  // 1 <= k <= 32
    uint32_t next() { return genrand32(); }

private:
    uint32_t mt_[624];
    int mti_;

    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (int i = 1; i < 624; i++)
            mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + (uint32_t)i;
        mti_ = 624;
    }

    // CPython init_by_array (matsumoto reference, verbatim semantics).
    void init_by_array(const std::vector<uint32_t>& key) {
        size_t keyused = key.size();
        init_genrand(19650218u);
        size_t i = 1, j = 0;
        size_t k = (624 > keyused) ? 624 : keyused;
        for (; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u))
                     + key[j] + (uint32_t)j;
            i++; j++;
            if (i >= 624) { mt_[0] = mt_[623]; i = 1; }
            if (j >= keyused) j = 0;
        }
        for (k = 623; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u))
                     - (uint32_t)i;
            i++;
            if (i >= 624) { mt_[0] = mt_[623]; i = 1; }
        }
        mt_[0] = 0x80000000u;   // MSB set: non-zero initial array
        mti_ = 624;             // CPython: index stays at N after init_by_array
    }

    uint32_t genrand32() {
        if (mti_ >= 624) {
            for (int i = 0; i < 624; i++) {
                uint32_t y = (mt_[i] & 0x80000000u) | (mt_[(i + 1) % 624] & 0x7fffffffu);
                mt_[i] = mt_[(i + 397) % 624] ^ (y >> 1);
                if (y & 1u) mt_[i] ^= 0x9908b0dfu;
            }
            mti_ = 0;
        }
        uint32_t y = mt_[mti_++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }
};

// ---------------------------------------------------------------------------
// Core decoder semantics (mirror of ../plp/solver.py)
// ---------------------------------------------------------------------------

struct PlacedBox {
    int x, y, z, w, d, h;
    int item_type;
};

struct FastBox {
    int x, y, z, x2, y2, z2;   // z2 == z + h (top level)
};

using Pt = std::array<int, 3>;   // candidate point (x, y, z)

// D1: parser order, dedup (a cube yields one). Mirrors allowed_orientations +
// the dedup in solver.orientations.
inline std::vector<Pt> orientations(const Item& it) {
    std::vector<Pt> out;
    auto add = [&](int w, int d, int h) {
        for (const auto& p : out)
            if (p[0] == w && p[1] == d && p[2] == h) return;
        out.push_back({w, d, h});
    };
    if (it.lf) { add(it.w, it.h, it.l); add(it.h, it.w, it.l); }
    if (it.wf) { add(it.l, it.h, it.w); add(it.h, it.l, it.w); }
    if (it.hf) { add(it.l, it.w, it.h); add(it.w, it.l, it.h); }
    return out;
}

// candidate list invariant: sorted by (z, y, x), unique. Mirrors
// `sorted(set, key=(z,y,x))`.
inline void sort_dedup_pt(std::vector<Pt>& v) {
    std::sort(v.begin(), v.end(), [](const Pt& a, const Pt& b) {
        if (a[2] != b[2]) return a[2] < b[2];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[0] < b[0];
    });
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

// C1+C2+C3 feasibility of one (candidate, orientation) pair against the
// current placement. Mirrors the per-candidate loop in decode_pass exactly
// (C2 early-exit, then C3, then C1). Order-independent across pairs, so the
// CUDA build parallelizes it one thread per pair.
inline bool check_feasible_one(int x, int y, int z, const Pt& o,
                               const std::vector<FastBox>& fast,
                               int L, int W, int H) {
    int w = o[0], d = o[1], h = o[2];
    if (x + w > L || y + d > W || z + h > H) return false;          // C2
    if (z > 0) {
        long long s = 0;
        for (const auto& b : fast) {
            if (b.z2 == z) {
                long long ox = std::min((long long)x + w, (long long)b.x2) - std::max(x, b.x);
                if (ox > 0) {
                    long long oy = std::min((long long)y + d, (long long)b.y2) - std::max(y, b.y);
                    if (oy > 0) s += ox * oy;
                }
            }
        }
        // Python: s / (w * d) < TAU  (int/int -> one correctly-rounded double)
        if ((double)s / (double)(w * d) < TAU) return false;        // C3
    }
    for (const auto& b : fast) {
        if (x < b.x2 && b.x < x + w && y < b.y2 && b.y < y + d
            && z < b.z2 && b.z < z + h) return false;               // C1
    }
    return true;
}

// Feasibility backend: fills bits[ci * ors.size() + oi]. Host and CUDA builds
// provide different implementations; results must be bit-identical (that IS
// the gate 3.3 parity check, end-to-end).
struct FeasibilityBackend {
    virtual ~FeasibilityBackend() = default;
    virtual std::vector<uint8_t> check_all(const std::vector<Pt>& cand,
                                           const std::vector<Pt>& ors,
                                           const std::vector<FastBox>& fast,
                                           int L, int W, int H) = 0;
};

// D2: (type, copy) sequence. 'volume': (-volume, type, copy); 'input': file order.
inline std::vector<std::pair<int, int>> item_sequence(const Instance& inst, const std::string& order) {
    std::vector<std::pair<int, int>> seq;
    for (size_t ti = 0; ti < inst.items.size(); ti++)
        for (int c = 0; c < inst.items[ti].qty; c++)
            seq.push_back({(int)ti, c});
    if (order == "volume") {
        std::sort(seq.begin(), seq.end(), [&](const auto& a, const auto& b) {
            long long va = inst.items[a.first].volume(), vb = inst.items[b.first].volume();
            if (va != vb) return va > vb;
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
    } else if (order != "input") {
        throw std::invalid_argument("unknown order " + order);
    }
    return seq;
}

// D4'/D5/D6: per-box FSR on a completed packing. Mirrors compute_fsr 1:1,
// including float op order: ((MU/A_LAT_G) * load[i]) / m[i],
// load[j] += (load[k] * a) / total.
inline std::vector<double> compute_fsr(const std::vector<PlacedBox>& placed) {
    int n = (int)placed.size();
    std::vector<double> m(n), load(n);
    for (int i = 0; i < n; i++)
        m[i] = RHO * (double)(1LL * placed[i].w * placed[i].d * placed[i].h) / 1.0e6;
    load = m;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    // Python: sorted(range(n), key=lambda i: -placed[i].z)  (stable)
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return placed[a].z > placed[b].z;
    });
    for (int k : idx) {
        const auto& bk = placed[k];
        std::vector<std::pair<int, long long>> areas;
        for (int j = 0; j < n; j++) {
            if (j == k || placed[j].z + placed[j].h != bk.z) continue;
            long long ox = std::min((long long)bk.x + bk.w, (long long)placed[j].x + placed[j].w)
                           - std::max(bk.x, placed[j].x);
            if (ox <= 0) continue;
            long long oy = std::min((long long)bk.y + bk.d, (long long)placed[j].y + placed[j].d)
                           - std::max(bk.y, placed[j].y);
            if (oy > 0) areas.push_back({j, ox * oy});
        }
        long long total = 0;
        for (auto& a : areas) total += a.second;
        if (total > 0)
            for (auto& [j, a] : areas)
                load[j] += load[k] * (double)a / (double)total;
    }
    std::vector<double> fsr(n);
    for (int i = 0; i < n; i++)
        fsr[i] = (MU / A_LAT_G) * load[i] / m[i];
    return fsr;
}

// Fraction of boxes with FSR >= 1 (empty packing -> 0.0), mirrors stability_score.
inline double stability_score(const std::vector<PlacedBox>& placed) {
    if (placed.empty()) return 0.0;
    auto fsr = compute_fsr(placed);
    int c = 0;
    for (double f : fsr) if (f >= 1.0) c++;
    return (double)c / (double)placed.size();
}

// D1/D3: one decode pass. eps = 0 (or first_fit) -> lexicographically first
// feasible candidate (deterministic parity regime); eps > 0 -> eps-greedy with
// the shared PyRng (state persists across passes, as in solver.run).
inline std::vector<PlacedBox> decode_pass(const Instance& inst, const std::string& order,
                                          bool first_fit, double eps, PyRng& rng,
                                          FeasibilityBackend& be) {
    int L = inst.L, W = inst.W, H = inst.H;
    std::vector<std::vector<Pt>> orients;
    orients.reserve(inst.items.size());
    for (const auto& it : inst.items) orients.push_back(orientations(it));
    std::vector<PlacedBox> placed;
    std::vector<FastBox> fast;
    std::vector<Pt> cand = {{0, 0, 0}};
    for (auto [ti, copy] : item_sequence(inst, order)) {
        (void)copy;
        const auto& ors = orients[ti];
        if (ors.empty()) continue;
        std::vector<uint8_t> bits = be.check_all(cand, ors, fast, L, W, H);
        std::vector<std::array<int, 4>> feas;   // (x, y, z, oi) in candidate-major order
        for (size_t ci = 0; ci < cand.size(); ci++) {
            int x = cand[ci][0], y = cand[ci][1], z = cand[ci][2];
            for (size_t oi = 0; oi < ors.size(); oi++)
                if (bits[ci * ors.size() + oi])
                    feas.push_back({x, y, z, (int)oi});
        }
        if (feas.empty()) continue;             // skip this copy; pass continues
        std::array<int, 4> pick;
        if (!(first_fit || eps == 0.0)) {
            if (rng.random() >= 1.0 - eps)
                pick = feas[(size_t)rng.randrange((int)feas.size())];
            else
                pick = feas[0];
        } else {
            pick = feas[0];
        }
        int w = ors[pick[3]][0], d = ors[pick[3]][1], h = ors[pick[3]][2];
        placed.push_back({pick[0], pick[1], pick[2], w, d, h, ti});
        fast.push_back({pick[0], pick[1], pick[2], pick[0] + w, pick[1] + d, pick[2] + h});
        // extreme-point update: swallow points strictly inside the new box,
        // add its 8 corners (they can never be strictly inside an older box)
        std::vector<Pt> nc;
        nc.reserve(cand.size() + 8);
        for (const auto& p : cand)
            if (!(pick[0] < p[0] && p[0] < pick[0] + w
                  && pick[1] < p[1] && p[1] < pick[1] + d
                  && pick[2] < p[2] && p[2] < pick[2] + h))
                nc.push_back(p);
        for (int cx : {pick[0], pick[0] + w})
            for (int cy : {pick[1], pick[1] + d})
                for (int cz : {pick[2], pick[2] + h})
                    nc.push_back({cx, cy, cz});
        sort_dedup_pt(nc);
        cand = std::move(nc);
    }
    return placed;
}

struct RunResult {
    std::vector<std::array<int, 7>> placements;   // (item_type, x, y, z, w, d, h)
    int n_boxes = 0;
    long long volume = 0;
    double score = 0.0;
    bool fallback = false;
    int n_passes = 0;
    double time_s = 0.0;
    int seed = 0;
};

// D4' acceptance + best-packing selection; mirrors run()/select_best.
// theta == nullptr (V2) accepts every packing.
inline RunResult run(const Instance& inst, int seed, double time_limit_s, double eps,
                     double* theta, const std::string& order, bool first_fit,
                     bool single_pass, FeasibilityBackend& be) {
    PyRng rng((uint64_t)seed);
    auto t0 = std::chrono::steady_clock::now();
    struct Pack {
        long long volume;
        int n;
        double score;
        std::vector<PlacedBox> placed;
    };
    std::vector<Pack> packings;
    int k = 0;
    for (;;) {
        auto placed = decode_pass(inst, order, first_fit, eps, rng, be);
        k++;
        long long vol = 0;
        for (const auto& b : placed) vol += 1LL * b.w * b.d * b.h;
        packings.push_back(Pack{vol, (int)placed.size(), stability_score(placed), std::move(placed)});
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (single_pass || el >= time_limit_s) break;
    }
    std::vector<const Pack*> accepted;
    for (const auto& p : packings)
        if (theta == nullptr || p.score >= *theta) accepted.push_back(&p);
    // D4': highest volume among accepted; if none accepted, overall best
    // (fallback). Ties keep the earliest packing (strict > only).
    const Pack* best;
    size_t pool_end;
    if (accepted.empty()) {
        best = &packings.front();
        pool_end = packings.size();
    } else {
        best = accepted.front();
        pool_end = accepted.size();
    }
    for (size_t i = 1; i < pool_end; i++) {
        const Pack* p = accepted.empty() ? &packings[i] : accepted[i];
        if (p->volume > best->volume || (p->volume == best->volume && p->n > best->n))
            best = p;
    }
    RunResult r;
    for (const auto& b : best->placed)
        r.placements.push_back({b.item_type, b.x, b.y, b.z, b.w, b.d, b.h});
    r.n_boxes = best->n;
    r.volume = best->volume;
    r.score = best->score;
    r.fallback = accepted.empty();
    r.n_passes = k;
    r.seed = seed;
    r.time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

// ---------------------------------------------------------------------------
// JSON output — layout mirrors Python json.dumps (same key order, ", " / ": "
// separators); floats formatted like Python float repr (shortest, same rules).
// ---------------------------------------------------------------------------

// Format a double exactly like CPython's repr(): shortest digit string that
// round-trips to the same bits (CPython's own algorithm), then fixed form iff
// -4 < e <= 16 in 0.dmm * 10^e form, else 'd[.dd]e±XX' with 2-digit exponents.
// (float::to_chars is unavailable on Apple libc++, so the trim loop uses
// snprintf %.16e — 17 significant digits always round-trip a double.)
inline std::string py_repr(double x) {
    if (x == 0.0) return (std::signbit(x) ? "-0.0" : "0.0");
    if (std::isinf(x) || std::isnan(x)) {   // not expected; mirror repr anyway
        if (std::isnan(x)) return "nan";
        return (x < 0 ? "-inf" : "inf");
    }
    bool neg = x < 0;
    double ax = neg ? -x : x;
    char tmp[64];
    // trim to the shortest precision that round-trips (mirrors CPython;
    // 17 significant digits always round-trip, so search down from 16)
    int dec = 16;
    for (int t = 16; t >= 0; t--) {
        std::snprintf(tmp, sizeof tmp, "%.*e", t, ax);
        if (std::strtod(tmp, nullptr) == ax) dec = t;
        else break;
    }
    std::snprintf(tmp, sizeof tmp, "%.*e", dec, ax);
    // tmp = "d[.ddd]e±XX" -> significant digits + exponent
    const char* ep = std::strchr(tmp, 'e');
    int e10 = std::atoi(ep + 1);
    std::string digits;
    for (const char* c = tmp; c < ep; c++)
        if (*c != '.') digits.push_back(*c);
    int E = e10 + 1;                        // ax = 0.digits * 10^E
    // 0.digits * 10^E:
    //   E >= k:  digits + (E-k) zeros + ".0"          (1.0, 1000000.0)
    //   1<=E<k:  digits[0:E] + "." + digits[E:]       (12.34)
    //   E <= 0:  "0." + (-E) zeros + digits           (0.1, 0.001, 0.0001)
    std::string out;
    if (E >= -3 && E <= 16) {
        int k = (int)digits.size();
        if (E >= k) {
            out = digits;
            out.append((size_t)(E - k), '0');
            out += ".0";
        } else if (E <= 0) {
            out = "0.";
            out.append((size_t)(-E), '0');
            out += digits;
        } else {
            out = digits.substr(0, (size_t)E) + "." + digits.substr((size_t)E);
        }
    } else {
        int ex = E - 1;
        out = std::string(1, digits[0]);
        if (digits.size() > 1) out += "." + digits.substr(1);
        out += "e";
        out += (ex < 0 ? "-" : "+");
        int ae = std::abs(ex);
        if (ae < 10) out += "0";
        out += std::to_string(ae);
    }
    if (neg) out = "-" + out;
    return out;
}

inline std::string to_json(const RunResult& r) {
    std::ostringstream o;
    o << "{\"n_boxes\": " << r.n_boxes
      << ", \"volume_cm3\": " << r.volume
      << ", \"stability_score\": " << py_repr(r.score)
      << ", \"fallback\": " << (r.fallback ? "true" : "false")
      << ", \"n_passes\": " << r.n_passes
      << ", \"time_s\": " << py_repr(r.time_s)
      << ", \"seed\": " << r.seed
      << ", \"placements\": [";
    for (size_t i = 0; i < r.placements.size(); i++) {
        const auto& p = r.placements[i];
        if (i) o << ", ";
        o << "[" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3]
          << ", " << p[4] << ", " << p[5] << ", " << p[6] << "]";
    }
    o << "]}";
    return o.str();
}

}  // namespace plp
