#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${GDN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
DATA_ROOT=${GDN_DATA_ROOT:-$(cd "$REPO/../.." && pwd)}
if [[ -n ${QWEN35_MODEL_PATH:-} ]]; then
  MODEL=$QWEN35_MODEL_PATH
elif [[ -d $DATA_ROOT/modelscope_models/Qwen3.5-9B ]]; then
  MODEL=$DATA_ROOT/modelscope_models/Qwen3.5-9B
else
  MODEL=$DATA_ROOT/zypan_ling_dist/models/Qwen3.5-9B
fi
if [[ -n ${QWEN_PYTHON_BIN:-} ]]; then
  PYTHON_BIN=$QWEN_PYTHON_BIN
elif [[ -x $DATA_ROOT/.conda/envs/sd310/bin/python3.10 ]]; then
  PYTHON_BIN=$DATA_ROOT/.conda/envs/sd310/bin/python3.10
elif [[ -x /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 ]]; then
  PYTHON_BIN=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
else
  PYTHON_BIN=$(command -v python3)
fi
RUNNER=$REPO/experiments/propagation/run_qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1.py
RUN_DIR=$REPO/runs/qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1
LOG_DIR=${QWEN_MAG_EQ_LOG_DIR:-$DATA_ROOT/runs/qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1/logs}

mkdir -p "$LOG_DIR"

wait_for_pid_file() {
  local pid_file=$1
  local pid
  pid=$(tr -d '[:space:]' < "$pid_file")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

run_worker() {
  local gpu=$1
  local shard=$2
  local stage=$3
  env CUDA_VISIBLE_DEVICES="$gpu" GDN_DATA_ROOT="$DATA_ROOT" QWEN35_MODEL_PATH="$MODEL" \
    "$PYTHON_BIN" "$RUNNER" --stage "$stage" --num-shards 2 --shard-id "$shard" \
    --resume --keep-going > "$LOG_DIR/${stage}_shard${shard}.log" 2>&1
}

echo "[$(date)] waiting for pilot workers"
wait_for_pid_file "$LOG_DIR/pilot_shard0.pid" &
wait0=$!
wait_for_pid_file "$LOG_DIR/pilot_shard1.pid" &
wait1=$!
wait "$wait0" "$wait1"

echo "[$(date)] merging pilot"
env GDN_DATA_ROOT="$DATA_ROOT" QWEN35_MODEL_PATH="$MODEL" \
  "$PYTHON_BIN" "$RUNNER" --stage merge-pilot --num-shards 2 \
  > "$LOG_DIR/merge_pilot.log" 2>&1

pilot_decision=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["PILOT"])' "$RUN_DIR/aggregate_pilot.json")
echo "[$(date)] pilot decision: $pilot_decision"

if ! env GDN_DATA_ROOT="$DATA_ROOT" QWEN35_MODEL_PATH="$MODEL" \
  "$PYTHON_BIN" "$RUNNER" --stage check-formal-prerequisite \
  > "$LOG_DIR/check_formal_prerequisite.log" 2>&1; then
  echo "[$(date)] stop rule applied before formal"
  touch "$LOG_DIR/offline_stopped_after_pilot.done"
  exit 1
fi

echo "[$(date)] launching formal workers on GPUs 0 and 1"
run_worker 0 0 formal &
formal0=$!
echo "$formal0" > "$LOG_DIR/formal_shard0.pid"
run_worker 1 1 formal &
formal1=$!
echo "$formal1" > "$LOG_DIR/formal_shard1.pid"
if wait "$formal0"; then rc0=0; else rc0=$?; fi
if wait "$formal1"; then rc1=0; else rc1=$?; fi
echo "[$(date)] formal workers exited: shard0=$rc0 shard1=$rc1"
if (( rc0 != 0 || rc1 != 0 )); then
  echo "[$(date)] formal worker failure; merge is blocked"
  exit 1
fi

env GDN_DATA_ROOT="$DATA_ROOT" QWEN35_MODEL_PATH="$MODEL" \
  "$PYTHON_BIN" "$RUNNER" --stage merge-formal --num-shards 2 \
  > "$LOG_DIR/merge_formal.log" 2>&1

"$PYTHON_BIN" -c '
import json, sys
summary = json.load(open(sys.argv[1]))
if summary.get("formal_status") != "COMPLETE" or summary.get("FRESH_OUTPUT_VERIFICATION") != "PASS":
    raise SystemExit("formal aggregate did not pass completion checks")
' "$RUN_DIR/aggregate_summary.json"

touch "$LOG_DIR/offline_formal_merge.done"
echo "[$(date)] offline task chain finished"
