#!/usr/bin/env bash
set -euo pipefail

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
OUT="$REPO/artifacts/ling_256k_length_sensitivity_v1/tp1x2_trial"
PY=/data/zypan/runtime_compat/ling256_exact/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data/zypan/models/Ling-3.0-tiny
DATASET="$REPO/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
mkdir -p "$OUT"

setsid "$EXP/launch_ling_256k_single_server.sh" 2 31002 >"$OUT/gpu2_server.log" 2>&1 &
PID2=$!
setsid "$EXP/launch_ling_256k_single_server.sh" 3 31003 >"$OUT/gpu3_server.log" 2>&1 &
PID3=$!

cleanup() {
  kill -TERM -- "-$PID2" 2>/dev/null || true
  kill -TERM -- "-$PID3" 2>/dev/null || true
  wait "$PID2" 2>/dev/null || true
  wait "$PID3" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo 'timestamp,index,uuid,memory_used_mib,utilization_gpu_percent' >"$OUT/gpu_memory.csv"
(
  while kill -0 "$PID2" 2>/dev/null || kill -0 "$PID3" 2>/dev/null; do
    stamp=$(date --iso-8601=seconds)
    nvidia-smi -i 2,3 --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | sed "s/^/$stamp,/" >>"$OUT/gpu_memory.csv"
    sleep 1
  done
) &
MONITOR_PID=$!

for _ in $(seq 1 240); do
  kill -0 "$PID2" 2>/dev/null || { echo 'GPU2 server exited'; exit 1; }
  kill -0 "$PID3" 2>/dev/null || { echo 'GPU3 server exited'; exit 1; }
  h2=0; h3=0
  curl -fsS --max-time 2 http://127.0.0.1:31002/health >/dev/null 2>&1 && h2=1 || true
  curl -fsS --max-time 2 http://127.0.0.1:31003/health >/dev/null 2>&1 && h3=1 || true
  [[ "$h2" -eq 1 && "$h3" -eq 1 ]] && break
  sleep 5
done
[[ "${h2:-0}" -eq 1 && "${h3:-0}" -eq 1 ]] || { echo 'readiness timeout'; exit 1; }

export PYTHONPATH="/data/zypan/src/sglang-v0.5.19-ling256/python:$EXP"
"$PY" "$EXP/run_tp1x2_smoke.py" --model "$MODEL" --dataset "$DATASET" \
  --output "$OUT/tp1x2_smoke_summary.json" | tee "$OUT/tp1x2_smoke.stdout.log"

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
awk -F, 'NR>1 {gsub(/ /,"",$2); gsub(/ /,"",$4); if (($4+0)>m[$2]) m[$2]=$4+0} END {for (i in m) print "GPU" i "_PEAK_MIB=" m[i]}' \
  "$OUT/gpu_memory.csv" | sort | tee "$OUT/gpu_memory_peak.txt"

