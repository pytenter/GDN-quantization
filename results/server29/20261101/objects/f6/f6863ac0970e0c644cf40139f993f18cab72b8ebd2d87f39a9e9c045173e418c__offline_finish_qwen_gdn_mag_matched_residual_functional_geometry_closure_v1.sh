#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${GDN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
DATA_ROOT=${GDN_DATA_ROOT:-$(cd "$REPO/.." && pwd)}
MODEL=${QWEN35_MODEL_PATH:-$DATA_ROOT/modelscope_models/Qwen3.5-9B}
PYTHON_BIN=${QWEN_PYTHON_BIN:-/data/ydai/miniconda3/envs/bitdecode/bin/python3.10}
RUNNER=$REPO/experiments/propagation/run_qwen_gdn_mag_matched_residual_functional_geometry_closure_v1.py
RUN_DIR=$REPO/runs/qwen_gdn_mag_matched_residual_functional_geometry_closure_v1
LOG_DIR=${QWEN_GEOMETRY_LOG_DIR:-$DATA_ROOT/runs/qwen_gdn_mag_matched_residual_functional_geometry_closure_v1/logs}

mkdir -p "$LOG_DIR"

run_worker() {
  local gpu=$1
  local stage=$2
  local shard_id=$3
  local num_shards=$4
  env CUDA_VISIBLE_DEVICES="$gpu" GDN_REPO_ROOT="$REPO" GDN_DATA_ROOT="$DATA_ROOT" \
    QWEN35_MODEL_PATH="$MODEL" "$PYTHON_BIN" "$RUNNER" --stage "$stage" \
    --shard-id "$shard_id" --num-shards "$num_shards" --resume --keep-going \
    > "$LOG_DIR/${stage}_shard${shard_id}.log" 2>&1
}

merge_stage() {
  local stage=$1
  local num_shards=$2
  env GDN_REPO_ROOT="$REPO" GDN_DATA_ROOT="$DATA_ROOT" QWEN35_MODEL_PATH="$MODEL" \
    "$PYTHON_BIN" "$RUNNER" --stage "merge-$stage" --num-shards "$num_shards" \
    > "$LOG_DIR/merge_${stage}.log" 2>&1
}

run_pair() {
  local stage=$1
  run_worker 0 "$stage" 0 2 &
  local pid0=$!
  echo "$pid0" > "$LOG_DIR/${stage}_shard0.pid"
  run_worker 1 "$stage" 1 2 &
  local pid1=$!
  echo "$pid1" > "$LOG_DIR/${stage}_shard1.pid"
  local rc0=0
  local rc1=0
  if wait "$pid0"; then rc0=0; else rc0=$?; fi
  if wait "$pid1"; then rc1=0; else rc1=$?; fi
  echo "[$(date)] $stage workers exited: shard0=$rc0 shard1=$rc1"
  if (( rc0 != 0 || rc1 != 0 )); then
    echo "[$(date)] $stage worker failure; merge blocked"
    return 1
  fi
}

echo "[$(date)] stage0 on GPU 0"
run_worker 0 stage0 0 1
merge_stage stage0 1

echo "[$(date)] smoke on GPU 0"
run_worker 0 smoke 0 1
merge_stage smoke 1

echo "[$(date)] pilot on GPUs 0 and 1"
run_pair pilot
merge_stage pilot 2

if env GDN_REPO_ROOT="$REPO" GDN_DATA_ROOT="$DATA_ROOT" QWEN35_MODEL_PATH="$MODEL" \
  "$PYTHON_BIN" "$RUNNER" --stage check-formal-prerequisite \
  > "$LOG_DIR/check_formal_prerequisite.log" 2>&1; then
  cat "$LOG_DIR/check_formal_prerequisite.log"
else
  echo "[$(date)] pilot did not support formalization; printing pilot summary and stopping"
  cat "$LOG_DIR/check_formal_prerequisite.log"
  cat "$LOG_DIR/merge_pilot.log"
  touch "$LOG_DIR/offline_stopped_after_pilot.done"
  exit 0
fi

echo "[$(date)] formal on GPUs 0 and 1"
run_pair formal
merge_stage formal 2
cat "$LOG_DIR/merge_formal.log"

"$PYTHON_BIN" - "$RUN_DIR/aggregate_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
if summary.get("formal_status") != "COMPLETE":
    raise SystemExit("formal status is not COMPLETE")
if summary.get("FRESH_OUTPUT_VERIFICATION") != "PASS":
    raise SystemExit("fresh output verification failed")
if summary.get("CURRENT_TASK_COMPLETED_AND_STOPPED") != "YES":
    raise SystemExit("task stop indicator failed")
PY

touch "$LOG_DIR/offline_formal_merge.done"
echo "[$(date)] task complete; no next experiment launched"
