# plp-gpu — GPU-Accelerated 3D Pallet Loading Solver

CUDA-accelerated heuristic decoder for the 3D pallet loading problem
(3D-PLP) with a friction-aware stability score (Friction Stability
Ratio, FSR). Companion code for the paper:

> E. Erat, "A GPU-Accelerated Heuristic for 3D Pallet Loading with
> Friction-Aware Stability Constraints," Sakarya University, 2026.

## Repository layout

```
plp/            Python CPU baseline (pure stdlib): parser, geometry, solver
gpu/            C++17 / CUDA solver: shared core, host main, CUDA main
tests/          Python unit tests (pytest)
data/           benchmark instances BR1-BR15 + one synthetic instance + manifest
parity/         bit-parity reference JSONs (CPython / g++ / nvcc) + compare script
```

## The decoder

- **One shared source for both C++ builds** — `gpu/plp_common.hpp`
  contains the decoder, FSR, RNG, and JSON output. `gpu/plp_gpu.cpp`
  (C++17 host) and `gpu/plp_gpu.cu` (CUDA) differ only in the
  feasibility backend: `check_feasible_one` runs as one GPU thread per
  (candidate position, orientation) pair.
- **Bit-exact parity with the Python CPU reference** (`plp/solver.py`):
  floating-point expressions mirror the Python source 1:1 (IEEE-754
  double, same operand order, FMA contraction disabled at build time)
  and the MT19937 stream mirrors CPython `random.Random(seed)` bit for
  bit.
- **Friction Stability Ratio (FSR)** — post-hoc, packing-level
  stability-acceptance score (μ = 0.2, a_lat = 0.3 g, θ = 0.9),
  orientation-aware through the cumulative stack load that increases
  the normal force and hence the available friction.
- **JSON output** — placements, per-item FSR, packed volume, wall time.

## CPU baseline (Python)

Pure standard library — no third-party dependencies (Python ≥ 3.9).

```bash
python3 -m plp.solver data/instances/BR1_050.txt --json
```

CLI (identical flags to the C++ binary):

```
python3 -m plp.solver <instance> [--seed N] [--time S] [--eps F]
                         [--theta F | --no-theta]
                         [--order volume|input] [--first-fit]
                         [--single-pass] [--json]
```

Parity configuration against the C++/CUDA builds:
`--single-pass --eps 0 --no-theta --json`.

## GPU / host solver (C++ / CUDA)

Requirements: C++17 compiler (g++ / clang) for the host build; CUDA
toolkit + `nvcc` for the GPU build (developed with CUDA 13.0 on NVIDIA
GB10, `sm_121`; other architectures: `make cuda ARCH=sm_XX`).

```bash
cd gpu
make host   # g++  -O3 -std=c++17 -ffp-contract=off            -> plp_gpu
make cuda   # nvcc -O3 -std=c++17 -arch=sm_121 --fmad=false ... -> plp_gpu_cuda
```

**Do not** add `-use_fast_math` or any fast-math flag: bit parity
against the CPU reference requires exact IEEE-754 semantics (device FMA
off, host contraction off).

```
./plp_gpu <instance> [--seed N] [--time S] [--eps F]
                     [--theta F | --no-theta]
                     [--order volume|input] [--first-fit]
                     [--single-pass] [--json]
./plp_gpu --selftest
```

## Instance format

thpack "original" format (units: cm):

```
line 0:     P                        # number of instances
per instance:
    p [seed]                         # problem id (seed optional)
    L W H                            # container dimensions
    n                                # number of box types
    i l lf w wf h hf qty             # per type: i = 1..n
```

`lf/wf/hf` (0/1): the box may stand on that axis, i.e. that dimension
may be vertical in a placement (max 6 orientations per item). The
standard BR1–BR15 benchmark instances (Bischoff & Ratcliff 1995;
Davies & Bischoff 1999) use a 587×233×220 cm 20-ft ISO container;
provenance (source repo + commit, selection rule, units, per-instance
stats) is recorded in `data/manifest.json`.

## Parity verification

- `gpu/make_selftest_ref.py` regenerates the `--selftest` reference
  from CPython; `./plp_gpu --selftest` cross-checks the RNG stream and
  float formatting against it.
- `parity/ref_cpu/` (CPython), `parity/ref_host/` (g++),
  `parity/ref_gpu/` (nvcc) are the checked-in reference outputs.
- `python3 parity/compare_parity.py DIR_A DIR_B [NAME ...]` diffs any
  two output directories field by field (`time_s` excluded,
  machine-dependent). Exit 0 iff every instance matches bit for bit.

## Tests

```bash
python3 -m pytest tests/ -v
```

Covers the parser (18 cases), the geometry primitives, and the solver
including a single-pass smoke run over all 15 BR instances.

## Measured results (paper benchmark)

| Metric | Value |
|---|---|
| Packed volume vs. deterministic first-fit baseline | +21.9% |
| Speedup vs. CPU baseline (controlled) | 21.7× |
| Speedup under co-resident load | 9.3× |
| GPU↔CPU packed-volume parity | −0.64% |
| Determinism (12 replays of the same config) | 12/12 bit-identical |

FSR is applied as a packing-level acceptance criterion (θ = 0.9 on the
post-hoc score). Under the baseline parameters (μ = 0.2, a_lat = 0.3 g)
the per-item threshold FSR_i ≥ 1 is structurally infeasible (maximum
observed 0.707); the paper's ablation reports this as a null stability
effect and keeps the criterion for parameter regimes where it engages.

## Citing

If you use this code in your research, please cite the paper above.

## License

MIT — see [LICENSE](LICENSE).
