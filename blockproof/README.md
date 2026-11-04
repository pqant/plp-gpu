# blockproof — block-building placement engine (CUDA port)

This directory contains the block-building variant of the placement heuristic:
items are first grouped into rectangular blocks (a x b x k copies of one
orientation), blocks are placed at extreme points, and each packing is scored
by the friction-aware stability model shared with the decoder (`gpu/plp_common.hpp`).

- `plp_block.cpp` — CPU reference (bit-parity source of truth).
- `plp_block_cuda.cu` — CUDA port; one kernel launch runs the whole pass loop
  for K independent seeds (lockstep rows, row k = PyRng(seed + k)).
- Bit parity: `run_cuda_parity.sh` compares CPU vs GPU (`--device 1`)
  single-pass JSON outputs field-by-field on all instances x seeds x 2 configs
  (wall-dependent `time_s` excluded). Last gate: 114/114 PASS on DGX Spark
  (GB10, CUDA 13.0.88, sm_121).

## Build

```
make            # CPU reference  -> ./plp_block
make cuda       # GPU build      -> ./plp_block_cuda   (needs nvcc >= 13, sm_121)
```

Float flags are parity-critical: `--fmad=false` and `-ffp-contract=off`,
never add `-use_fast_math`.

## Run

```
./plp_block_cuda <instance.txt> --device 12 --seed 0 --time 300 \
    --amax 4 --bmax 4 --kmax 6 --eps 0.5 --theta 0.9 --topk 128 --json
```

Campaign harnesses: `run_probe.sh` (winner-k decision, pre-registered rule),
`run_cuda_campaign.sh` (19 instances x K=12 x 300 s, resumable),
`summary_cuda.py` (mean/best fill per instance).

Known limitation (measured): the per-row pass rate is far below the CPU
(single-threaded serial planning/pick inside the kernel); the GPU value here
is seed-parallelism — 12 seeds complete inside one 300 s wall window.
