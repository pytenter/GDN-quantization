#!/usr/bin/env bash
set -euo pipefail

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
OUT="$REPO/artifacts/ling_256k_length_sensitivity_hadamard_coordinated_v1"
CONFIG="$OUT/LING_256K_VALUE_HADAMARD_COORDINATED_CONFIG.json"
PY=/data/zypan/runtime_compat/ling256_exact/.conda/envs/ling-sglang-aime26/bin/python
RUNNER="$EXP/run_ling_256k_length_sensitivity.py"
COORDINATOR="$EXP/coordinate_ling256_value_h_workers.py"
MODEL=/data/zypan/models/Ling-3.0-tiny
DATASET="$REPO/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
MIGRATED="$REPO/artifacts/ling_256k_length_sensitivity_gpu4_helper_migrated_v1/migration_source/4090_completed_records/formal256k/int8_r128_value_h"
GPU4_OLD="$REPO/artifacts/ling_256k_length_sensitivity_gpu4_helper_migrated_v1/records/formal256k/int8_r128_value_h"
TP1X2_OLD="$REPO/artifacts/ling_256k_length_sensitivity_tp1x2_v1/records/formal256k/int8_r128_value_h"
LOG="$OUT/coordination/join_after_int8.log"

mkdir -p "$OUT/coordination" "$OUT/formal/logs"
exec >>"$LOG" 2>&1
echo "WATCHER_START=$(date --iso-8601=seconds)"

while pgrep -af 'run_ling_256k_length_sensitivity.py' \
  | grep -q -- '--method int8_r128 --stage formal256k'; do
  echo "INT8_STILL_RUNNING=$(date --iso-8601=seconds)"
  sleep 60
done
echo "INT8_CLIENTS_COMPLETE=$(date --iso-8601=seconds)"

# The legacy parent was guarded by temporarily removing execute permission from
# its launcher.  Give it time to cleanly stop the INT8 servers and fail closed
# before restoring the launcher for the coordinated Value-Hadamard workers.
for _ in $(seq 1 120); do
  if ! tmux has-session -t ling256_tp1x2_formal 2>/dev/null; then
    break
  fi
  sleep 5
done
if tmux has-session -t ling256_tp1x2_formal 2>/dev/null; then
  echo "ERROR=legacy_pipeline_did_not_exit_after_guard"
  exit 1
fi

launcher_mode=$(cat "$OUT/coordination/original_launcher_mode.txt")
chmod "$launcher_mode" "$EXP/launch_ling_256k_single_server.sh"
echo "LAUNCHER_MODE_RESTORED=$launcher_mode"

start_server() {
  local gpu="$1" port="$2"
  local session="ling256_h_queue_server_gpu${gpu}"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"
  tmux new-session -d -s "$session" \
    "exec '$EXP/launch_ling_256k_single_server.sh' '$gpu' '$port' int8_r128_value_h >> '$OUT/formal/logs/server_gpu${gpu}.log' 2>&1"
}

wait_ready() {
  local port="$1"
  for _ in $(seq 1 240); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_worker() {
  local gpu="$1" port="$2"
  local worker="gpu${gpu}_queue"
  local server_session="ling256_h_queue_server_gpu${gpu}"
  local worker_session="ling256_h_queue_worker_gpu${gpu}"
  tmux new-session -d -s "$worker_session" \
    "bash -lc '\
      rc=0; \
      \"$PY\" \"$COORDINATOR\" \
        --worker-id \"$worker\" --physical-gpu \"$gpu\" \
        --base-url \"http://127.0.0.1:$port\" \
        --python \"$PY\" --runner \"$RUNNER\" \
        --model \"$MODEL\" --dataset \"$DATASET\" \
        --output-dir \"$OUT\" --config-json \"$CONFIG\" \
        --source-record-dir \"$MIGRATED\" \
        --source-record-dir \"$GPU4_OLD\" \
        --source-record-dir \"$TP1X2_OLD\" \
        >> \"$OUT/formal/logs/worker_gpu${gpu}.log\" 2>&1 || rc=\$?; \
      tmux kill-session -t \"$server_session\" 2>/dev/null || true; \
      exit \$rc'"
}

start_server 2 31002
start_server 3 31003
wait_ready 31002
wait_ready 31003
echo "HADAMARD_SERVERS_READY=$(date --iso-8601=seconds)"
start_worker 2 31002
start_worker 3 31003
echo "HADAMARD_QUEUE_WORKERS_STARTED=$(date --iso-8601=seconds)"
