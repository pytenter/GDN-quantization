#!/usr/bin/env bash
set -euo pipefail

EXP="${EXP:?EXP is required}"
PY="${PY:?PY is required}"
MODEL="${MODEL:?MODEL is required}"
CONDITION="${CONDITION:?CONDITION is required}"
PORT="${PORT:-31400}"
LOGDIR="${LOGDIR:?LOGDIR is required}"

AMENDMENT="$EXP/manifests/protocol_amendment_q29_helper_v2.json"
test -f "$AMENDMENT"
"$PY" - "$AMENDMENT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)
assert d["status"] == "FROZEN_BEFORE_Q29_GENERATION"
assert d["assignment_override"]["problem_id"] == "aime26_29"
assert d["assignment_override"]["to"] == {"host_class": "4090", "physical_gpu": 0}
assert sys.argv[1]
PY

case "$CONDITION" in
  L6) ROTATION="$EXP/rotations/L6_final_rotation.pt" ;;
  L7) ROTATION="$EXP/rotations/L7_final_rotation.pt" ;;
  *) echo "Unsupported condition: $CONDITION" >&2; exit 2 ;;
esac

mkdir -p "$LOGDIR"
MODE="int8_r128_unified_final_r"
PATCH="$EXP/scripts/sglang_kda_unified_final_r_patch.py"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$EXP/scripts/launch_parity_server.sh" 0 "$PORT" "$MODE" "$LOGDIR" "$PATCH" "$ROTATION" \
  >"$LOGDIR/server_launcher.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited before becoming healthy" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

"$PY" "$EXP/scripts/run_formal_recurrent.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --experiment-root "$EXP" \
  --condition "$CONDITION" \
  --problem-ids aime26_29 \
  --physical-gpu 0 \
  --instance-id "${CONDITION}_gpu0_4090_helper_q29" \
  --rotation-file "$ROTATION" \
  --resume
