#!/usr/bin/env bash
set -euo pipefail

REPO="/data01/user2/worktrees/aime26-sglang-rotation-v1"
PYTHON="/data01/user2/.conda/envs/ling-sglang-aime26/bin/python"
DATASET="$REPO/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
OLD_SMOKE="$REPO/artifacts/aime26_v1/formal/ling_sglang/max_new_tokens_65536/smoke"
RUN_ROOT="$REPO/artifacts/aime26_v2/official_sampling_81920/ling"
NEW_SMOKE="$RUN_ROOT/smoke"
MAX_NEW_TOKENS=81920
WORKER_PIDS=()

cleanup_workers() {
  local pid
  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup_workers EXIT INT TERM
mkdir -p "$RUN_ROOT/logs" "$NEW_SMOKE" "$RUN_ROOT/parity"
cp "$REPO/artifacts/aime26_v1/formal/ling_sglang/parity/gate_d_hadamard.json" "$RUN_ROOT/parity/gate_d_hadamard.json"

run_method() (
  local mode="$1"
  local tag="$2"
  local gpu="$3"
  local port="$4"
  local server_log="$RUN_ROOT/logs/${tag}_server.log"
  local client_log="$RUN_ROOT/logs/${tag}_client.log"
  local audit_jsonl="$REPO/artifacts/aime26_v2/official_sampling_81920/ling/parity/${tag}_audit.jsonl"
  local server_pid=""

  cleanup_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_server EXIT INT TERM

  if [[ -s "$NEW_SMOKE/${mode}.jsonl" ]]; then
    echo "Refusing to overwrite or append to existing 81920 smoke artifact: $NEW_SMOKE/${mode}.jsonl" >&2
    exit 1
  fi

  echo "START mode=$mode gpu=$gpu port=$port max_new_tokens=$MAX_NEW_TOKENS"
  setsid "$REPO/experiments/aime26/launch_ling_sglang_server.sh" \
    "$mode" "$gpu" "$port" "$tag" >"$server_log" 2>&1 &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before health check: mode=$mode" >&2
      tail -n 100 "$server_log" >&2 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "Server health timeout: mode=$mode" >&2
    exit 1
  fi

  "$PYTHON" -u "$REPO/experiments/aime26/run_ling_sglang_aime26.py" \
    --base-url "http://127.0.0.1:$port" \
    --dataset "$DATASET" \
    --output-dir "$RUN_ROOT" \
    --method "$mode" \
    --stage smoke \
    --gpu "$gpu" \
    --audit-jsonl "$audit_jsonl" \
    --max-new-tokens "$MAX_NEW_TOKENS" >"$client_log" 2>&1

  echo "DONE mode=$mode"
)

run_method fp_state smoke81920_fp 0 30000 &
FP_PID=$!
WORKER_PIDS+=("$FP_PID")
run_method int8_r128 smoke81920_r128 1 30001 &
R128_PID=$!
WORKER_PIDS+=("$R128_PID")

wait "$FP_PID"
run_method int8_r128_value_h smoke81920_r128_h_v2 0 30000 &
H_PID=$!
WORKER_PIDS+=("$H_PID")

wait "$R128_PID"
wait "$H_PID"
WORKER_PIDS=()

"$PYTHON" "$REPO/experiments/aime26/audit_ling_smoke_81920.py" \
  --old-dir "$OLD_SMOKE" \
  --new-dir "$NEW_SMOKE" \
  --output "$RUN_ROOT/integrity_report.json" >"$RUN_ROOT/integrity_report.stdout.json"

REPLAY_TAG="replay81920_fp"
REPLAY_SERVER_LOG="$RUN_ROOT/logs/${REPLAY_TAG}_server.log"
setsid "$REPO/experiments/aime26/launch_ling_sglang_server.sh" fp_state 0 30000 "$REPLAY_TAG" >"$REPLAY_SERVER_LOG" 2>&1 &
REPLAY_SERVER_PID=$!
REPLAY_READY=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:30000/health" >/dev/null 2>&1; then REPLAY_READY=1; break; fi
  if ! kill -0 "$REPLAY_SERVER_PID" 2>/dev/null; then tail -n 100 "$REPLAY_SERVER_LOG" >&2 || true; exit 1; fi
  sleep 2
done
if [[ "$REPLAY_READY" -ne 1 ]]; then echo "Replay server health timeout" >&2; exit 1; fi
"$PYTHON" -u "$REPO/experiments/aime26/ling_deterministic_replay.py" \
  --base-url "http://127.0.0.1:30000" --dataset "$DATASET" \
  --output "$RUN_ROOT/launch/deterministic_replay.json" \
  >"$RUN_ROOT/logs/deterministic_replay.stdout.log" 2>&1
kill -TERM -- "-$REPLAY_SERVER_PID" 2>/dev/null || kill -TERM "$REPLAY_SERVER_PID" 2>/dev/null || true
wait "$REPLAY_SERVER_PID" 2>/dev/null || true
"$PYTHON" "$REPO/experiments/aime26/audit_ling_formal_preflight.py" \
  >"$RUN_ROOT/launch/preflight_report.stdout.json"

echo "LING_SMOKE_81920_AND_PREFLIGHT_COMPLETE"
