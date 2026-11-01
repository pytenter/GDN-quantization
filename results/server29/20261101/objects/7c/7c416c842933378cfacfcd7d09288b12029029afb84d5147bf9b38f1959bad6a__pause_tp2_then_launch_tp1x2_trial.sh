#!/usr/bin/env bash
set -euo pipefail

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
RECORDS="$REPO/artifacts/ling_256k_length_sensitivity_v1/records/baseline81920/fp_state"
LOG="$REPO/artifacts/ling_256k_length_sensitivity_v1/tp1x2_trial/transition.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "WATCH_START=$(date --iso-8601=seconds)"

while [[ $(find "$RECORDS" -type f -name '*.json' 2>/dev/null | wc -l) -lt 9 ]]; do
  sleep 2
done
echo "BOUNDARY_RECORD_COUNT=9 timestamp=$(date --iso-8601=seconds)"

CLIENT_PID=$(pgrep -f 'run_ling_256k_length_sensitivity.py.*--stage baseline81920' | head -n 1 || true)
if [[ -n "$CLIENT_PID" ]]; then
  kill -INT "$CLIENT_PID" 2>/dev/null || true
fi
for _ in $(seq 1 120); do
  tmux has-session -t ling256_formal 2>/dev/null || break
  sleep 2
done
tmux has-session -t ling256_formal 2>/dev/null && { echo 'BLOCK: TP2 formal did not stop'; exit 1; }

for _ in $(seq 1 120); do
  busy=0
  for index in 2 3; do
    uuid=$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader,nounits)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -Fxq "$uuid" && busy=1 || true
  done
  [[ "$busy" -eq 0 ]] && break
  sleep 2
done
[[ "${busy:-1}" -eq 0 ]] || { echo 'BLOCK: GPU 2 or 3 remained busy'; exit 1; }

tmux new-session -d -s ling256_tp1x2_trial -n smoke "bash -lc '$EXP/run_tp1x2_trial.sh'"
echo "TP1X2_TRIAL_LAUNCHED=$(date --iso-8601=seconds)"

