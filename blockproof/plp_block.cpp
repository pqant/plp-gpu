// plp_block.cpp — block-building insertion prototype (block proof; cp-147).
//
// Spike per blockproof/BRIEF.md: does Eley-style type-group block insertion
// break the V0 ~84% fill ceiling on BR1–3 (bar: >=88% 12-seed mean fill)?
// The search unit is a BLOCK: a x b base grid of copies of ONE item type in
// ONE orientation, stacked k high. Everything else stays the shared protocol
// from ../gpu/plp_common.hpp — parser, orientation flags (lf/wf/hf),
// C1/C2/C3 feasibility incl. the >=TAU support rule, CPython-compatible
// MT19937 (PyRng), FSR scoring (stability_score on decomposed item boxes),
// theta acceptance + fallback + best-volume selection, JSON schema. Only the
// search unit changes relative to V0; placements are decomposed into item
// boxes before scoring, so fill% and score mean exactly what they mean in
// the campaign JSONs.
//
// Build: g++ -O3 -std=c++17 -ffp-contract=off plp_block.cpp -o plp_block
// Run:   ./plp_block <instance> --seed S [--time 300] [--eps 0.3]
//                    [--theta 0.9 | --no-theta] [--amax 3] [--bmax 3]
//                    [--kmax 4] [--single-pass] [--json]

#include "../gpu/plp_common.hpp"

namespace {

using namespace plp;

// One block type: cnt copies of item ti in orientation oi as a bw x bd x bh
// rectangular stack.
struct BlockShape {
    int ti, oi;
    int a, b, k;
    long long cnt;
    int bw, bd, bh;
    long long bvol() const { return 1LL * bw * bd * bh; }
};

// Plan-phase precedence within one type: most copies first, then tallest
// stack, then widest base (deterministic; ties by orientation index).
inline bool shape_precedence(const BlockShape& x, const BlockShape& y) {
    if (x.cnt != y.cnt) return x.cnt > y.cnt;
    if (x.bh != y.bh) return x.bh > y.bh;
    if (x.a != y.a) return x.a > y.a;
    if (x.b != y.b) return x.b > y.b;
    return x.oi < y.oi;
}

struct PlacedBlock {
    int x, y, z, w, d, h;   // whole-block box (C1/C2/C3 unit)
    int ti;
    int a, b, k;
};

// Tuning round 1 (cp-148): how to pick among the feasible extreme points.
//   first   — feas[0] (lowest then leftmost); byte-identical to the cp-147 campaign.
//   support — max bottom support area (pallet floor counts as full support);
//             ties broken by side contact area, then by list order.
//   contact — max total contact area (bottom + side faces); ties by bottom,
//             then list order. Strict-> keeps the earliest point on full ties.
enum PickRule { PICK_FIRST = 0, PICK_SUPPORT = 1, PICK_CONTACT = 2 };

// Bottom support + side contact area of block shape s at point p vs the
// current boxes. Bottom mirrors the C3 sum in check_feasible_one (b.z2 == z
// overlap); side counts flush face-to-face contacts only (x or y aligned).
// Wall/flush contacts against the pallet sides are NOT counted.
inline void contact_stats(const Pt& p, const BlockShape& s,
                          const std::vector<FastBox>& fast,
                          long long& bottom, long long& side) {
    bottom = 0; side = 0;
    const long long x = p[0], y = p[1], z = p[2];
    const long long bw = s.bw, bd = s.bd, bh = s.bh;
    if (z == 0) bottom = bw * bd;   // full support from the pallet floor
    for (const auto& b : fast) {
        const long long ox = std::min(x + bw, (long long)b.x2) - std::max(x, (long long)b.x);
        const long long oy = std::min(y + bd, (long long)b.y2) - std::max(y, (long long)b.y);
        const long long oz = std::min(z + bh, (long long)b.z2) - std::max(z, (long long)b.z);
        if (b.z2 == z && ox > 0 && oy > 0) bottom += ox * oy;
        if (oy > 0 && oz > 0 && (b.x2 == x || b.x == x + bw)) side += oy * oz;
        if (ox > 0 && oz > 0 && (b.y2 == y || b.y == y + bd)) side += ox * oz;
    }
}

struct BlockPass {
    std::vector<PlacedBox> items;   // decomposed — V0 scoring/volume unit
    long long volume = 0;
    int n = 0;
};

// One block-building pass: repeated (plan -> placement) sweeps.
//   plan: every type with remaining copies is decomposed into blocks —
//         types in volume-desc order (eps: full shuffle), shapes per type by
//         shape_precedence (eps: random pick among the top-8 feasible ones).
//   placement: blocks in block-volume-desc order (the D2 volume analogue),
//         first feasible extreme point in (z, y, x) order; eps: uniform
//         random among the feasible points — the decode_pass pick rule at
//         block granularity.
BlockPass block_pass(const Instance& inst,
                     const std::vector<std::vector<Pt>>& orients,
                     const std::vector<BlockShape>& cat,
                     const std::vector<std::vector<int>>& by_type,
                     double eps, PickRule rule, int topk, PyRng& rng) {
    const int L = inst.L, W = inst.W, H = inst.H;
    std::vector<int> rem;
    rem.reserve(inst.items.size());
    for (const auto& it : inst.items) rem.push_back(it.qty);

    std::vector<PlacedBlock> blocks;
    std::vector<FastBox> fast;
    std::vector<Pt> cand = {{0, 0, 0}};

    auto place = [&](const BlockShape& s, const Pt& p) {
        blocks.push_back({p[0], p[1], p[2], s.bw, s.bd, s.bh, s.ti, s.a, s.b, s.k});
        fast.push_back({p[0], p[1], p[2], p[0] + s.bw, p[1] + s.bd, p[2] + s.bh});
        // extreme-point update — mirrors decode_pass (swallow interior, add 8 corners)
        std::vector<Pt> nc;
        nc.reserve(cand.size() + 8);
        for (const auto& q : cand)
            if (!(p[0] < q[0] && q[0] < p[0] + s.bw
                  && p[1] < q[1] && q[1] < p[1] + s.bd
                  && p[2] < q[2] && q[2] < p[2] + s.bh))
                nc.push_back(q);
        for (int cx : {p[0], p[0] + s.bw})
            for (int cy : {p[1], p[1] + s.bd})
                for (int cz : {p[2], p[2] + s.bh})
                    nc.push_back({cx, cy, cz});
        sort_dedup_pt(nc);
        cand = std::move(nc);
        rem[s.ti] -= (int)s.cnt;
    };

    for (;;) {
        // ---- plan: block decomposition of every type with copies left ----
        std::vector<int> types;
        for (size_t ti = 0; ti < rem.size(); ti++)
            if (rem[ti] > 0 && !orients[ti].empty()) types.push_back((int)ti);
        if (types.empty()) break;
        std::sort(types.begin(), types.end(), [&](int x, int y) {
            long long vx = inst.items[x].volume(), vy = inst.items[y].volume();
            if (vx != vy) return vx > vy;
            return x < y;
        });
        if (eps > 0.0 && rng.random() >= 1.0 - eps)   // eps-diversity: shuffle type order
            for (size_t i = types.size() - 1; i > 0; i--)
                std::swap(types[i], types[(size_t)rng.randrange((int)i + 1)]);

        std::vector<int> plan;   // indices into cat
        for (int ti : types) {
            int r = rem[ti];
            const auto& shp = by_type[ti];   // pre-sorted by shape_precedence
            while (r > 0) {
                // shp is cnt-DESC: the fitting shapes form a suffix — find its
                // start (first index with cnt <= r), then eps-greedy among the
                // largest cnt that still fits.
                size_t start = shp.size();
                while (start > 0 && cat[shp[start - 1]].cnt <= r) start--;
                if (start == shp.size()) break;   // unreachable while singles exist
                size_t tk = std::min<size_t>((size_t)topk, shp.size() - start);
                size_t pick = (eps > 0.0 && rng.random() >= 1.0 - eps)
                                  ? (size_t)rng.randrange((int)tk) : 0;
                plan.push_back(shp[start + pick]);
                r -= (int)cat[shp[start + pick]].cnt;
            }
        }
        if (plan.empty()) break;

        // placement order: block volume desc (D2 volume-order analogue)
        std::sort(plan.begin(), plan.end(), [&](int x, int y) {
            const BlockShape& s = cat[x];
            const BlockShape& t = cat[y];
            if (s.bvol() != t.bvol()) return s.bvol() > t.bvol();
            if (s.ti != t.ti) return s.ti < t.ti;
            if (s.oi != t.oi) return s.oi < t.oi;
            if (s.k != t.k) return s.k > t.k;
            if (s.a != t.a) return s.a > t.a;
            return s.b > t.b;
        });

        // ---- placement: extreme-point insertion at block granularity ----
        bool any = false;
        for (int idx : plan) {
            const BlockShape& s = cat[idx];
            if (rem[s.ti] < (int)s.cnt) continue;   // planned before, stale
            std::vector<Pt> feas;
            for (const auto& p : cand)
                if (check_feasible_one(p[0], p[1], p[2], {s.bw, s.bd, s.bh}, fast, L, W, H))
                    feas.push_back(p);
            if (feas.empty()) continue;             // this block stays out (for now)
            Pt pick;
            if (eps > 0.0 && rng.random() >= 1.0 - eps) {
                pick = feas[(size_t)rng.randrange((int)feas.size())];
            } else if (rule == PICK_FIRST) {
                pick = feas[0];
            } else {
                long long k0 = -1, k1 = -1;         // primary/secondary score
                size_t bi = 0;
                for (size_t i = 0; i < feas.size(); i++) {
                    long long sup = 0, side = 0;
                    contact_stats(feas[i], s, fast, sup, side);
                    const long long p0 = (rule == PICK_SUPPORT) ? sup : sup + side;
                    const long long p1 = (rule == PICK_SUPPORT) ? side : sup;
                    if (p0 > k0 || (p0 == k0 && p1 > k1)) { k0 = p0; k1 = p1; bi = i; }
                }
                pick = feas[bi];
            }
            place(s, pick);
            any = true;
        }
        if (!any) break;   // no sweep progress -> leftover copies are unplaceable
    }

    // ---- decompose blocks into item boxes (identical geometry: bw = a*w) ----
    BlockPass out;
    for (const auto& bl : blocks) {
        const int iw = bl.w / bl.a, id = bl.d / bl.b, ih = bl.h / bl.k;
        for (int i = 0; i < bl.a; i++)
            for (int j = 0; j < bl.b; j++)
                for (int t = 0; t < bl.k; t++) {
                    out.items.push_back({bl.x + i * iw, bl.y + j * id, bl.z + t * ih,
                                         iw, id, ih, bl.ti});
                    out.volume += 1LL * iw * id * ih;
                }
        out.n += bl.a * bl.b * bl.k;
    }
    return out;
}

RunResult run_block(const Instance& inst, int seed, double time_limit_s, double eps,
                    double* theta, int amax, int bmax, int kmax, bool single_pass,
                    PickRule rule, int topk) {
    std::vector<std::vector<Pt>> orients;
    orients.reserve(inst.items.size());
    for (const auto& it : inst.items) orients.push_back(orientations(it));

    // ---- block catalogue (bounded by amax/bmax/kmax and pallet fit) ----
    std::vector<BlockShape> cat;
    for (size_t ti = 0; ti < inst.items.size(); ti++) {
        for (size_t oi = 0; oi < orients[ti].size(); oi++) {
            const int w = orients[ti][oi][0], d = orients[ti][oi][1], h = orients[ti][oi][2];
            for (int a = 1; a <= amax; a++) {
                if (a * w > inst.L) break;
                for (int b = 1; b <= bmax; b++) {
                    if (b * d > inst.W) break;
                    for (int k = 1; k <= kmax; k++) {
                        if (k * h > inst.H) break;
                        long long cnt = 1LL * a * b * k;
                        if (cnt > inst.items[ti].qty) break;   // k only grows cnt
                        cat.push_back({(int)ti, (int)oi, a, b, k, cnt, a * w, b * d, k * h});
                    }
                }
            }
        }
    }
    std::vector<std::vector<int>> by_type(inst.items.size());
    for (size_t i = 0; i < cat.size(); i++) by_type[cat[i].ti].push_back((int)i);
    for (auto& v : by_type)
        std::sort(v.begin(), v.end(), [&](int x, int y) { return shape_precedence(cat[x], cat[y]); });

    // ---- pass loop + acceptance (identical semantics to run(): theta accept,
    //      fallback to overall best, ties -> more boxes, strict > keeps the
    //      earliest packing). Kept ONLINE (running maxima, no packings vector):
    //      the block engine does ~100k passes/s, so storing every packing
    //      would exhaust RAM at the 300 s budget. ----
    PyRng rng((uint64_t)seed);
    auto t0 = std::chrono::steady_clock::now();
    struct Best {
        long long volume = -1;
        int n = 0;
        double score = 0.0;
        std::vector<PlacedBox> placed;
        bool better(long long v, int boxes) const {
            return v > volume || (v == volume && boxes > n);
        }
    };
    Best best_acc, best_all;
    bool has_acc = false;
    int k = 0;
    for (;;) {
        auto bp = block_pass(inst, orients, cat, by_type, eps, rule, topk, rng);
        k++;
        double sc = stability_score(bp.items);
        bool upd_all = best_all.better(bp.volume, bp.n);
        bool upd_acc = (theta == nullptr || sc >= *theta) &&
                       (!has_acc || best_acc.better(bp.volume, bp.n));
        if (upd_acc) {
            best_acc = Best{bp.volume, bp.n, sc, bp.items};   // copy (small)
            has_acc = true;
        }
        if (upd_all) {
            best_all = Best{bp.volume, bp.n, sc, std::move(bp.items)};   // then move
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (single_pass || el >= time_limit_s) break;
    }
    const Best& best = has_acc ? best_acc : best_all;
    RunResult r;
    for (const auto& b : best.placed)
        r.placements.push_back({b.item_type, b.x, b.y, b.z, b.w, b.d, b.h});
    r.n_boxes = best.n;
    r.volume = best.volume;
    r.score = best.score;
    r.fallback = !has_acc;
    r.n_passes = k;
    r.seed = seed;
    r.time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s <instance> --seed N [--time S] [--eps F] "
                 "[--theta F | --no-theta]\n"
                 "                 [--amax A] [--bmax B] [--kmax K] [--topk N]\n"
                 "                 [--pick first|support|contact]\n"
                 "                 [--single-pass] [--json]\n",
                 prog);
}

}  // namespace

int main(int argc, char** argv) {
    std::string instance_path;
    int seed = 0, amax = 3, bmax = 3, kmax = 4;
    double time_s = 300.0, eps = EPS_QUALITY, theta = THETA;
    bool no_theta = false, single_pass = false, as_json = false;
    PickRule rule = PICK_FIRST;
    int topk = 8;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--seed") seed = std::atoi(next(a.c_str()).c_str());
        else if (a == "--time") time_s = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--eps") eps = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--theta") theta = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--no-theta") no_theta = true;
        else if (a == "--amax") amax = std::atoi(next(a.c_str()).c_str());
        else if (a == "--bmax") bmax = std::atoi(next(a.c_str()).c_str());
        else if (a == "--kmax") kmax = std::atoi(next(a.c_str()).c_str());
        else if (a == "--single-pass") single_pass = true;
        else if (a == "--topk") topk = std::max(1, std::atoi(next(a.c_str()).c_str()));
        else if (a == "--pick") {
            std::string v = next(a.c_str());
            if (v == "first") rule = PICK_FIRST;
            else if (v == "support") rule = PICK_SUPPORT;
            else if (v == "contact") rule = PICK_CONTACT;
            else { std::fprintf(stderr, "bad --pick: %s (first|support|contact)\n", v.c_str()); return 2; }
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

    Instance inst;
    try {
        inst = load_instance(instance_path);
    } catch (const InstanceFormatError& e) {
        std::fprintf(stderr, "instance error: %s\n", e.what());
        return 1;
    }
    RunResult r = run_block(inst, seed, time_s, eps, no_theta ? nullptr : &theta,
                            amax, bmax, kmax, single_pass, rule, topk);
    double fill = 100.0 * (double)r.volume / (double)(1LL * inst.L * inst.W * inst.H);
    if (as_json) {
        std::cout << to_json(r) << "\n";
        std::fprintf(stderr, "fill_pct=%.4f\n", fill);
    } else {
        std::printf("n_boxes=%d volume_cm3=%lld fill_pct=%.4f stability_score=%.4f "
                    "fallback=%s n_passes=%d time_s=%.3f\n",
                    r.n_boxes, (long long)r.volume, fill, r.score,
                    r.fallback ? "true" : "false", r.n_passes, r.time_s);
    }
    return 0;
}
