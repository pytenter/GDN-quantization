#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 WORKER_ID PHYSICAL_GPU PORT" >&2
  exit 2
fi

WORKER_ID="$1"
PHYSICAL_GPU="$2"
PORT="$3"
REPO="/data01/user2/worktrees/aime26-sglang-rotation-v1"
PYTHON="/data01/user2/.conda/envs/ling-sglang-aime26/bin/python"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
DATASET="$REPO/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
OUT="$REPO/artifacts/aime26_v2/official_sampling_81920/ling/formal"
LAUNCH_ROOT="$REPO/artifacts/aime26_v2/official_sampling_81920/ling/launch"
ACTIVE_SERVER_PID=""

cleanup_server() {
  if [[ -n "$ACTIVE_SERVER_PID" ]] && kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$ACTIVE_SERVER_PID" 2>/dev/null || kill -TERM "$ACTIVE_SERVER_PID" 2>/dev/null || true
    wait "$ACTIVE_SERVER_PID" 2>/dev/null || true
  fi
  ACTIVE_SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

cd "$REPO"
mkdir -p "$OUT" "$LAUNCH_ROOT"
echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$PYTHON"
echo "worker_id=$WORKER_ID physical_gpu=$PHYSICAL_GPU port=$PORT assigned_units=90"
echo "config=thinking:on temperature:1.0 top_p:0.95 top_k:20 max_new_tokens:81920 seeds:1,2"
echo "kda_rotation_semantics=CORRECTED_PREFILL_ENDPOINT_V2 redundant_prefill_endpoint_rotation=NO"

for METHOD in fp_state int8_r128 int8_r128_value_h; do
  TAG="formal_w${WORKER_ID}_${METHOD}_v2"
  SERVER_LOG="$LAUNCH_ROOT/${TAG}_server.log"
  AUDIT_JSONL="$REPO/artifacts/aime26_v2/official_sampling_81920/ling/parity/${TAG}_audit.jsonl"
  echo "START_SERVER method=$METHOD tag=$TAG timestamp=$(date --iso-8601=seconds)"
  setsid experiments/aime26/launch_ling_sglang_server.sh "$METHOD" "$PHYSICAL_GPU" "$PORT" "$TAG" \
    >"$SERVER_LOG" 2>&1 &
  ACTIVE_SERVER_PID=$!
  READY=0
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
      echo "server exited before health check method=$METHOD" >&2
      tail -n 100 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "$READY" -ne 1 ]]; then echo "server health timeout method=$METHOD" >&2; exit 1; fi
  echo "SERVER_HEALTHY method=$METHOD pid=$ACTIVE_SERVER_PID timestamp=$(date --iso-8601=seconds)"
  "$PYTHON" -u experiments/aime26/run_ling_sglang_aime26.py \
    --base-url "http://127.0.0.1:$PORT" --dataset "$DATASET" --output-dir "$OUT" \
    --method "$METHOD" --stage formal --gpu "$PHYSICAL_GPU" --audit-jsonl "$AUDIT_JSONL" \
    --max-new-tokens 81920 --worker-id "$WORKER_ID" --num-workers 2 --resume
  cleanup_server
  echo "DONE method=$METHOD timestamp=$(date --iso-8601=seconds)"
done
