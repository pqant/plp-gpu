#!/usr/bin/env bash
# cp-150 GPU campaign: 19 instances x 12 seeds (K=12 lockstep, row k = PyRng(seed+k))
# x 300 s x winner-k (pre-registered probe winner). Resumable — skips existing
# non-empty JSON, same convention as run_one.sh.
# Usage: TOPK=128 ./run_cuda_campaign.sh
set -u
cd "$(dirname "$0")"
mkdir -p results_cuda logs
for inst_file in ../data/instances/BR*_050.txt; do
  inst=$(basename "$inst_file" .txt)
  out="results_cuda/${inst}_k${TOPK:-128}.json"
  if [ -s "$out" ]; then continue; fi
  ./plp_block_cuda "$inst_file" --device 12 --seed 0 --time "${TIME:-300}" \
    --amax "${AMAX:-4}" --bmax "${BMAX:-4}" --kmax "${KMAX:-6}" \
    --eps "${EPS:-0.5}" --theta "${THETA:-0.9}" --topk "${TOPK:-128}" \
    --json > "$out.tmp" 2>> logs/campaign.log
  if [ $? -eq 0 ] && grep -q '"rows"' "$out.tmp"; then
    mv "$out.tmp" "$out"
    echo "DONE $inst k${TOPK:-128}" >> logs/campaign.log
  else
    rm -f "$out.tmp"
    echo "FAIL $inst k${TOPK:-128}" >> logs/campaign.log
  fi
done
echo "CAMPAIGN_COMPLETE" >> logs/campaign.log
