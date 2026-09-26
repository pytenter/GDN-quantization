#!/usr/bin/env bash
set -euo pipefail

NEW=/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1/experiments/LING_RECURRENT_DENSE_L6_L7_V1
OLD=/data01/user2/worktrees/ling-sglang-dense-aime26-last20-256k-singleseed-v1/experiments/LING_SGLANG_DENSE_AIME26_LAST20_256K_SINGLESEED_V1
Q13="$OLD/outputs/L3_INT8_R128_DENSE_HADAMARD_STEP0/aime26_13_seed1.json"
CLIENT=2697903
SERVER=2697901
PARENT=2697899
OLD_POSTPROCESS=2699079
LOG="$NEW/logs/l3_early_stop_finalizer.log"

exec >> "$LOG" 2>&1
echo "WAIT_L3_IN_FLIGHT $(date --iso-8601=seconds)"
while [[ ! -f "$Q13" ]]; do
  if ! kill -0 "$CLIENT" 2>/dev/null; then
    echo "in-flight L3 client exited before the retained q13 record appeared" >&2
    exit 1
  fi
  sleep 30
done
echo "L3_Q13_COMPLETE $(date --iso-8601=seconds)"

for _ in $(seq 1 20); do
  kill -0 "$CLIENT" 2>/dev/null || break
  sleep 5
done
if kill -0 "$CLIENT" 2>/dev/null; then
  kill -TERM "$CLIENT"
fi

for _ in $(seq 1 60); do
  kill -0 "$PARENT" 2>/dev/null || break
  sleep 5
done
if kill -0 "$PARENT" 2>/dev/null; then
  kill -TERM "$PARENT"
fi
if kill -0 "$SERVER" 2>/dev/null; then
  kill -TERM "$SERVER"
fi
if kill -0 "$OLD_POSTPROCESS" 2>/dev/null; then
  kill -TERM "$OLD_POSTPROCESS"
fi

python "$NEW/scripts/finalize_l3_early_stop.py" \
  --l3-output-dir "$OLD/outputs/L3_INT8_R128_DENSE_HADAMARD_STEP0" \
  --output "$NEW/analysis/l3_early_stop.json"

for condition in L4_INT8_R128_DENSE_STATE_OLD L5_INT8_R128_DENSE_FUNCTIONAL_OLD; do
  count=0
  if [[ -d "$OLD/outputs/$condition" ]]; then
    count="$(find "$OLD/outputs/$condition" -maxdepth 1 -type f -name 'aime26_*_seed1.json' | wc -l)"
  fi
  if [[ "$count" -ne 0 ]]; then
    echo "unexpected $condition formal outputs: $count" >&2
    exit 1
  fi
done
echo "L3_EARLY_STOP_FINALIZED $(date --iso-8601=seconds)"
