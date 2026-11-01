#!/usr/bin/env bash
set -u

BASE=/data/zypan
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
SCRIPT_ABS=$BASE/experiments/qwen35_gdn_quant/run_end2end_bit_axis_screening.py
SCRIPT_NAME=run_end2end_bit_axis_screening.py
SHARDS=/data/zypan/results/gdn_end2end_bit_axis_screening_v1_shards
LOG=$SHARDS/two_gpu_int8_column_scheduler.log
GPUS=(0 1)

mkdir -p "$SHARDS"
cd "$BASE" || exit 1
echo "[$(date)] int8-column two-gpu scheduler started" >> "$LOG"

active_main_count() {
  pgrep -f "$SCRIPT_NAME --stage main" | wc -l
}

gpu_is_free() {
  local gpu=$1
  local pids
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
  [[ -z "$pids" ]]
}

first_free_gpu() {
  local gpu
  for gpu in "${GPUS[@]}"; do
    if gpu_is_free "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

expected_for_shard() {
  case "$1" in
    0|1) echo 8 ;;
    2|3) echo 7 ;;
    *) echo 0 ;;
  esac
}

count_column() {
  local shard=$1
  "$PY" - "$SHARDS/shard_00${shard}.jsonl" <<'PY'
import json, sys
n = 0
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            if line.strip() and json.loads(line).get("quantizer") == "INT8-column":
                n += 1
except FileNotFoundError:
    pass
print(n)
PY
}

job_done() {
  local shard=$1
  [[ "$(count_column "$shard")" -ge "$(expected_for_shard "$shard")" ]]
}

job_active() {
  local shard=$1
  pgrep -f "$SCRIPT_NAME --stage main --configs int8_column --shard-id $shard --num-shards 4" >/dev/null
}

launch_job() {
  local shard=$1
  local gpu=$2
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT_ABS" \
    --stage main --configs int8_column --shard-id "$shard" --num-shards 4 \
    --benchmarks MATH-500 --resume > "$SHARDS/int8_column_00${shard}.log" 2>&1 &
  echo $! > "$SHARDS/int8_column_00${shard}.pid"
  echo "[$(date)] launched int8_column shard=$shard gpu=$gpu pid=$!" >> "$LOG"
}

while true; do
  all_done=1
  for shard in 0 1 2 3; do
    if job_done "$shard"; then
      continue
    fi
    all_done=0
    if job_active "$shard"; then
      continue
    fi
    if [[ "$(active_main_count)" -ge 2 ]]; then
      break
    fi
    gpu=$(first_free_gpu || true)
    if [[ -z "${gpu:-}" ]]; then
      break
    fi
    launch_job "$shard" "$gpu"
    sleep 15
  done

  {
    echo "[$(date)] heartbeat active=$(active_main_count)"
    wc -l "$SHARDS"/shard_*.jsonl 2>&1 || true
  } >> "$LOG"

  if [[ "$all_done" -eq 1 ]] && [[ "$(active_main_count)" -eq 0 ]]; then
    echo "[$(date)] int8-column complete; running partial analyze" >> "$LOG"
    "$PY" "$SCRIPT_ABS" --stage analyze >> "$LOG" 2>&1
    echo "[$(date)] partial analyze finished rc=$?" >> "$LOG"
    exit 0
  fi
  sleep 60
done
