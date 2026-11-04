#!/usr/bin/env bash
# cp-150 bit-parity gate (TRANSITION_BRIEF protocol, gate-3.3 style):
# CPU plp_block vs GPU plp_block_cuda --device 1, --single-pass --json.
# 19 instances x seeds 0..2 x 2 configs (campaign + binary-defaults).
# time_s (wall-dependent) is stripped via sed; n_passes==1 is asserted.
# Run on Spark from the blockproof/ dir.
set -u
cd "$(dirname "$0")"
INST_DIR=../data/instances
OUT=parity_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
CFG_campaign="--amax 4 --bmax 4 --kmax 6 --eps 0.5 --theta 0.9 --topk 64"
CFG_default=""
FAIL=0; TOT=0
for cfg_name in campaign default; do
  eval "CFG=\$CFG_$cfg_name"
  for inst_file in $INST_DIR/BR*_050.txt; do
    inst=$(basename "$inst_file" .txt)
    for seed in 0 1 2; do
      TOT=$((TOT+1))
      tag="${cfg_name}_${inst}_s${seed}"
      ./plp_block "$inst_file" --seed $seed --single-pass --json $CFG > "$OUT/${tag}_cpu.json" 2>/dev/null
      rc_cpu=$?
      ./plp_block_cuda "$inst_file" --device 1 --seed $seed --single-pass --json $CFG > "$OUT/${tag}_gpu.json" 2>/dev/null
      rc_gpu=$?
      if [ $rc_cpu -ne 0 ] || [ $rc_gpu -ne 0 ]; then
        echo "FAIL rc $tag cpu=$rc_cpu gpu=$rc_gpu"
        FAIL=$((FAIL+1)); continue
      fi
      ok=1
      for side in cpu gpu; do
        p="$OUT/${tag}_${side}.json"
        grep -q '"n_passes": 1,' "$p" || { echo "FAIL npass $tag $side"; ok=0; }
        sed -i.bak -E 's/"time_s": [^,]*, //' "$p" && rm -f "$p.bak"
      done
      if [ $ok -eq 1 ] && diff -q "$OUT/${tag}_cpu.json" "$OUT/${tag}_gpu.json" > /dev/null; then
        echo "PASS $tag"
      else
        echo "FAIL diff $tag"
        FAIL=$((FAIL+1))
      fi
    done
  done
done
echo "TOTAL=$TOT FAIL=$FAIL OUTDIR=$OUT"
if [ $FAIL -eq 0 ]; then echo "PARITY_GATE: PASS"; else echo "PARITY_GATE: FAIL"; exit 1; fi
