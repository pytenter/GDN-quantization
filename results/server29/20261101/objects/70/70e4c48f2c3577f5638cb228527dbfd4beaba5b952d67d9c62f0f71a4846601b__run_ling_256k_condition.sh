#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 METHOD PORT STAGE [STAGE ...]" >&2
  exit 2
fi

METHOD="$1"
PORT="$2"
shift 2
case "$METHOD" in fp_state|int8_r128|int8_r128_value_h) ;; *) exit 2;; esac
for stage in "$@"; do
  case "$stage" in baseline81920|formal256k) ;; *) exit 2;; esac
done

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
OUT="$REPO/artifacts/ling_256k_length_sensitivity_v1"
PY=/data/zypan/runtime_compat/ling256_exact/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data/zypan/models/Ling-3.0-tiny
DATASET="$REPO/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
CONFIG="$EXP/LING_256K_EXPERIMENT_CONFIG.json"
LOG_DIR="$OUT/formal/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH="/data/zypan/src/sglang-v0.5.19-ling256/python:$EXP"

printf '\n=== server start %s method=%s port=%s ===\n' "$(date --iso-8601=seconds)" "$METHOD" "$PORT" \
  >> "$LOG_DIR/${METHOD}_server.log"
setsid "$EXP/launch_ling_256k_server.sh" "$METHOD" "$PORT" \
  >> "$LOG_DIR/${METHOD}_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited before readiness" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "server readiness timeout" >&2
  exit 1
fi

for stage in "$@"; do
  "$PY" "$EXP/run_ling_256k_length_sensitivity.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --dataset "$DATASET" --output-dir "$OUT" \
    --config-json "$CONFIG" --method "$METHOD" --stage "$stage" --resume \
    2>&1 | tee -a "$LOG_DIR/${METHOD}_${stage}_client.log"
done
