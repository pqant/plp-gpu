#!/usr/bin/env bash
# cp-150 GPU probe: winner-k decision by the PRE-REGISTERED rule (TRANSITION_BRIEF):
# higher overall 12-row mean wins across BR1-3; tie goes to the smaller k.
# BR1-3 x K=12 lockstep x 300 s x {k64, k128} -> probe/gpu_k{K}_{inst}_s0.json
set -u
cd "$(dirname "$0")"
mkdir -p probe logs
for k in 64 128; do
  for inst in BR1_050 BR2_050 BR3_050; do
    out="probe/gpu_k${k}_${inst}_s0.json"
    if [ -s "$out" ]; then echo "SKIP $out"; continue; fi
    ./plp_block_cuda ../data/instances/${inst}.txt --device 12 --seed 0 --time 300 \
      --amax 4 --bmax 4 --kmax 6 --eps 0.5 --theta 0.9 --topk "$k" --json \
      > "$out.tmp" 2>> logs/probe.log \
      && mv "$out.tmp" "$out" && echo "DONE k$k $inst" >> logs/probe.log \
      || echo "FAIL k$k $inst" >> logs/probe.log
  done
done
echo PROBE_COMPLETE >> logs/probe.log
