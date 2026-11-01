#!/usr/bin/env bash
set -u

BASE=/data/zypan
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
SCRIPT=experiments/qwen35_gdn_quant/run_end2end_bit_axis_screening.py
SCRIPT_ABS=$BASE/$SCRIPT
SCRIPT_NAME=run_end2end_bit_axis_screening.py
SHARDS=/data/zypan/results/gdn_end2end_bit_axis_screening_v1_shards
LOG=$SHARDS/two_gpu_scheduler.log
GPUS=(0 1)

mkdir -p "$SHARDS"
cd "$BASE" || exit 1
echo "[$(date)] two-gpu scheduler started" >> "$LOG"

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

fp_expected() {
  case "$1" in
    0|1) echo 8 ;;
    2|3) echo 7 ;;
    *) echo 0 ;;
  esac
}

count_shard() {
  local shard=$1
  local mode=$2
  "$PY" - "$SHARDS/shard_00${shard}.jsonl" "$mode" <<'PY'
import json, sys
path, mode = sys.argv[1], sys.argv[2]
n = 0
try:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = r.get("quantizer")
            if mode == "fp" and q == "FP_STATE":
                n += 1
            if mode == "quant" and q != "FP_STATE":
                n += 1
except FileNotFoundError:
    pass
print(n)
PY
}

job_done() {
  local kind=$1
  local shard=$2
  local n
  if [[ "$kind" == "fp" ]]; then
    n=$(count_shard "$shard" fp)
    [[ "$n" -ge "$(fp_expected "$shard")" ]]
  else
    n=$(count_shard "$shard" quant)
    [[ "$n" -ge 30 ]]
  fi
}

job_active() {
  local kind=$1
  local shard=$2
  local cfg
  if [[ "$kind" == "fp" ]]; then
    cfg=fp_state
  else
    cfg=int8_row,int8_column,int4_row,int4_column
  fi
  pgrep -f "$SCRIPT_NAME --stage main --configs $cfg --shard-id $shard --num-shards 4" >/dev/null
}

launch_job() {
  local kind=$1
  local shard=$2
  local gpu=$3
  local cfg
  if [[ "$kind" == "fp" ]]; then
    cfg=fp_state
    out="$SHARDS/fp_fill_00${shard}.log"
  else
    cfg=int8_row,int8_column,int4_row,int4_column
    out="$SHARDS/quant_00${shard}.log"
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT_ABS" \
    --stage main --configs "$cfg" --shard-id "$shard" --num-shards 4 \
    --benchmarks MATH-500 --resume > "$out" 2>&1 &
  echo $! > "$SHARDS/${kind}_00${shard}.pid"
  echo "[$(date)] launched $kind shard=$shard gpu=$gpu pid=$!" >> "$LOG"
}

JOBS=(
  "fp 0"
  "fp 2"
  "quant 0"
  "quant 1"
  "quant 2"
  "quant 3"
)

while true; do
  all_done=1
  for entry in "${JOBS[@]}"; do
    set -- $entry
    kind=$1
    shard=$2
    if job_done "$kind" "$shard"; then
      continue
    fi
    all_done=0
    if job_active "$kind" "$shard"; then
      continue
    fi
    if [[ "$(active_main_count)" -ge 2 ]]; then
      break
    fi
    gpu=$(first_free_gpu || true)
    if [[ -z "${gpu:-}" ]]; then
      break
    fi
    launch_job "$kind" "$shard" "$gpu"
    sleep 15
  done

  {
    echo "[$(date)] heartbeat active=$(active_main_count)"
    wc -l "$SHARDS"/shard_*.jsonl 2>&1 || true
  } >> "$LOG"

  if [[ "$all_done" -eq 1 ]] && [[ "$(active_main_count)" -eq 0 ]]; then
    echo "[$(date)] all queued jobs complete; running analyze" >> "$LOG"
    "$PY" "$SCRIPT_ABS" --stage analyze >> "$LOG" 2>&1
    echo "[$(date)] analyze finished rc=$?" >> "$LOG"
    exit 0
  fi
  sleep 60
done
