#!/usr/bin/env python3
"""cp-150 GPU campaign summary — reads results_cuda/BR{n}_050_k{K}.json (K=12 rows schema).

fill% = volume_cm3 / PALLET x 100 (same metric as summary.py). References: the CPU
12-seed confirmation (results_k64/_k128) and V0 (tab:main). Honest note: GPU per-row
pass-rate is ~1/125 of CPU (barrier-bound serial thread-0 core); its value comes from
seed-parallelism (12 seeds in one wall window), not from per-pass speed.
"""
import glob
import json
import os
import sys

PALLET = 30089620  # 587 x 233 x 220
V0_MEAN = {1: 84.02, 2: 84.22, 3: 81.45}
CPU_K128_MEAN = {1: 92.20, 2: 92.38, 3: 89.54}  # results_k128 12-seed confirmation
INSTANCES = tuple(range(19))


def main() -> int:
    k = os.environ.get("K", "128")
    root = os.environ.get("RESULTS", "results_cuda")
    n_done = 0
    for b in INSTANCES:
        files = glob.glob(f"{root}/BR{b}_050_k{k}.json")
        if not files:
            print(f"BR{b}: missing")
            continue
        d = json.load(open(files[0]))
        rows = d["rows"]
        fills = [r["volume_cm3"] / PALLET * 100 for r in rows]
        fb = sum(1 for r in rows if r["fallback"])
        passes = rows[0]["n_passes"]
        mean, best = sum(fills) / len(fills), max(fills)
        n_done += 1
        ref = (f" | CPUk128={CPU_K128_MEAN[b]:.2f} V0={V0_MEAN[b]:.2f}"
               if b in V0_MEAN else "")
        print(f"BR{b}: n={len(rows)} | mean={mean:.2f}% best={best:.2f}% | "
              f"fb={fb}/{len(rows)} | passes/row={passes}{ref}")
    print(f"\n{root}: {n_done}/19 instances")
    bar3 = all(
        (lambda r: sum(x["volume_cm3"] for x in r["rows"]) / len(r["rows"]) / PALLET * 100 >= 88.0)(
            json.load(open(f"{root}/BR{b}_050_k{k}.json")))
        for b in (1, 2, 3) if glob.glob(f"{root}/BR{b}_050_k{k}.json"))
    print(f"BAR (mean>=88 BR1-3, GPU 300s): {'PASS' if bar3 and n_done == 19 else 'check above'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
