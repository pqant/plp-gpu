// plp_gpu.cpp — host (g++) build of the 3D-PLP decoder (gate 3.3 stage A).
//
// Validates the C++ port of ../plp/solver.py bit-for-bit BEFORE the CUDA
// backend exists: parity target is the Python CPU baseline
// (`python -m plp.solver --single-pass --eps 0 --no-theta --json`).
// Build: g++ -O3 -std=c++17 -ffp-contract=off plp_gpu.cpp -o plp_gpu
//
// Flags mirror the Python CLI:
//   plp_gpu <instance> [--seed N] [--time S] [--eps F]
//                  [--theta F | --no-theta] [--order volume|input]
//                  [--first-fit] [--single-pass] [--json]
//   plp_gpu --selftest        RNG + float-format cross-check blocks

#include "plp_common.hpp"

namespace {

struct HostBackend : plp::FeasibilityBackend {
    std::vector<uint8_t> check_all(const std::vector<plp::Pt>& cand,
                                   const std::vector<plp::Pt>& ors,
                                   const std::vector<plp::FastBox>& fast,
                                   int L, int W, int H) override {
        std::vector<uint8_t> bits(cand.size() * ors.size(), 0);
        for (size_t ci = 0; ci < cand.size(); ci++) {
            int x = cand[ci][0], y = cand[ci][1], z = cand[ci][2];
            for (size_t oi = 0; oi < ors.size(); oi++)
                bits[ci * ors.size() + oi] =
                    plp::check_feasible_one(x, y, z, ors[oi], fast, L, W, H) ? 1 : 0;
        }
        return bits;
    }
};

// Print the exact IEEE-754 bit pattern of a double as 16 hex digits, for the
// --selftest cross-check against CPython (see gpu/make_selftest_ref.py).
void print_hexfloat(double x) {
    uint64_t bits;
    static_assert(sizeof(double) == sizeof(uint64_t));
    std::memcpy(&bits, &x, sizeof bits);
    std::printf("%016llx\n", (unsigned long long)bits);
}

void selftest() {
    // Block 1: RNG. For several seeds: 16 random() draws (bit hex) and
    // 16 getrandbits(13) (decimal). Cross-check with CPython:
    //   r = random.Random(s)
    //   for _ in range(16): print(format(struct.unpack('<Q', struct.pack('<d', r.random()))[0], '016x'))
    //   for _ in range(16): print(r.getrandbits(13))
    for (uint32_t s : {0u, 1u, 7u, 123456789u}) {
        plp::PyRng rng(s);
        std::printf("SEED %u\n", s);
        for (int i = 0; i < 16; i++) print_hexfloat(rng.random());
        for (int i = 0; i < 16; i++) std::printf("%u\n", rng.getrandbits(13));
    }
    // Block 2: float formatting vs CPython repr(). Cross-check:
    //   for v in (0.0, 1.0, 0.5, 0.1, 2.0/3.0, 0.5785123966942148,
    //             0.6132075471698113, 0.6595744680851063, 1e-05, 1e-04,
    //             1e15, 1e16, 0.001, 123456789.5, 1.0/121.0, 1.0/106.0):
    //       print(repr(v))
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
                 "                 [--order volume|input] [--first-fit] [--mu F] "
                 "[--single-pass] [--json]\n"
                 "       %s --selftest\n",
                 prog, prog);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace plp;
    std::string instance_path, order = "volume";
    int seed = 0;
    double time_s = 300.0, eps = EPS_QUALITY, theta = THETA;
    bool no_theta = false, first_fit = false, single_pass = false, as_json = false;
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
        else if (a == "--seed") seed = std::atoi(next(a.c_str()).c_str());
        else if (a == "--time") time_s = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--eps") eps = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--mu") MU = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--theta") theta = std::strtod(next(a.c_str()).c_str(), nullptr);
        else if (a == "--no-theta") no_theta = true;
        else if (a == "--order") { order = next(a.c_str()); }
        else if (a == "--first-fit") first_fit = true;
        else if (a == "--single-pass") single_pass = true;
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
    HostBackend be;
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
