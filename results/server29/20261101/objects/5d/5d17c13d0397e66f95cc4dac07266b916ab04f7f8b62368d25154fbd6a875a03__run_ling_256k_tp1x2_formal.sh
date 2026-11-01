#!/usr/bin/env bash
set -euo pipefail

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
SOURCE_OUT="$REPO/artifacts/ling_256k_length_sensitivity_v1"
OUT="$REPO/artifacts/ling_256k_length_sensitivity_tp1x2_v1"
PY=/data/zypan/runtime_compat/ling256_exact/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data/zypan/models/Ling-3.0-tiny
DATASET="$REPO/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
CONFIG="$EXP/LING_256K_TP1X2_EXPERIMENT_CONFIG.json"
LOG_DIR="$OUT/formal/logs"

grep -qx 'STAGE_0_STATUS=PASS' "$SOURCE_OUT/stage0/STAGE0_PASS"
"$PY" - "$SOURCE_OUT/tp1x2_trial/tp1x2_smoke_summary.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
assert row["status"] == "PASS", row
assert row["tp_size_per_instance"] == 1, row
assert row["parallel_instances"] == 2, row
assert row["physical_gpus"] == [2, 3], row
assert row["cross_gpu_exact_generated_ids"] is True, row
print("TP1X2_TRIAL_GATE=PASS")
PY

mkdir -p "$LOG_DIR"
printf 'PIPELINE_START=%s\n' "$(date --iso-8601=seconds)" >>"$OUT/pipeline.log"

ODD=()
EVEN=()
for n in $(seq -w 1 30); do
  value=$((10#$n))
  if (( value % 2 )); then
    ODD+=("aime26_$n")
  else
    EVEN+=("aime26_$n")
  fi
done

SERVER_PID2=
SERVER_PID3=
cleanup_servers() {
  for pid in "$SERVER_PID2" "$SERVER_PID3"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "$SERVER_PID2" "$SERVER_PID3"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
  SERVER_PID2=
  SERVER_PID3=
}
trap cleanup_servers EXIT INT TERM

start_servers() {
  local method="$1"
  printf '\n=== server start %s method=%s gpu=2 port=31002 ===\n' "$(date --iso-8601=seconds)" "$method" >>"$LOG_DIR/${method}_gpu2_server.log"
  printf '\n=== server start %s method=%s gpu=3 port=31003 ===\n' "$(date --iso-8601=seconds)" "$method" >>"$LOG_DIR/${method}_gpu3_server.log"
  setsid "$EXP/launch_ling_256k_single_server.sh" 2 31002 "$method" >>"$LOG_DIR/${method}_gpu2_server.log" 2>&1 &
  SERVER_PID2=$!
  setsid "$EXP/launch_ling_256k_single_server.sh" 3 31003 "$method" >>"$LOG_DIR/${method}_gpu3_server.log" 2>&1 &
  SERVER_PID3=$!
  local ready2=0 ready3=0
  for _ in $(seq 1 240); do
    kill -0 "$SERVER_PID2" 2>/dev/null || { echo "GPU2 server exited before readiness" >&2; return 1; }
    kill -0 "$SERVER_PID3" 2>/dev/null || { echo "GPU3 server exited before readiness" >&2; return 1; }
    curl -fsS --max-time 2 http://127.0.0.1:31002/health >/dev/null 2>&1 && ready2=1 || true
    curl -fsS --max-time 2 http://127.0.0.1:31003/health >/dev/null 2>&1 && ready3=1 || true
    [[ "$ready2" -eq 1 && "$ready3" -eq 1 ]] && break
    sleep 5
  done
  [[ "$ready2" -eq 1 && "$ready3" -eq 1 ]] || { echo "server readiness timeout" >&2; return 1; }
  printf 'SERVERS_READY=%s method=%s\n' "$(date --iso-8601=seconds)" "$method" >>"$OUT/pipeline.log"
}

run_stage() {
  local method="$1" stage="$2"
  local log2="$LOG_DIR/${method}_${stage}_gpu2_client.log"
  local log3="$LOG_DIR/${method}_${stage}_gpu3_client.log"
  "$PY" "$EXP/run_ling_256k_length_sensitivity.py" \
    --base-url http://127.0.0.1:31002 --model "$MODEL" --dataset "$DATASET" \
    --output-dir "$OUT" --config-json "$CONFIG" --method "$method" --stage "$stage" \
    --physical-gpu 2 --parallel-instance-id gpu2_odd --problem-ids "${ODD[@]}" --resume \
    >>"$log2" 2>&1 &
  local client2=$!
  "$PY" "$EXP/run_ling_256k_length_sensitivity.py" \
    --base-url http://127.0.0.1:31003 --model "$MODEL" --dataset "$DATASET" \
    --output-dir "$OUT" --config-json "$CONFIG" --method "$method" --stage "$stage" \
    --physical-gpu 3 --parallel-instance-id gpu3_even --problem-ids "${EVEN[@]}" --resume \
    >>"$log3" 2>&1 &
  local client3=$!
  local rc=0
  wait "$client2" || rc=$?
  wait "$client3" || rc=$?
  [[ "$rc" -eq 0 ]] || return "$rc"
  printf 'STAGE_COMPLETE=%s method=%s stage=%s\n' "$(date --iso-8601=seconds)" "$method" "$stage" >>"$OUT/pipeline.log"
}

start_servers fp_state
if [[ "${SKIP_MATCHED_BASELINE:-0}" == "1" ]]; then
  printf 'BASELINE_SKIPPED=%s reason=user_directed_256k_only\n' "$(date --iso-8601=seconds)" >>"$OUT/pipeline.log"
else
  run_stage fp_state baseline81920
fi
run_stage fp_state formal256k
cleanup_servers

start_servers int8_r128
run_stage int8_r128 formal256k
cleanup_servers

start_servers int8_r128_value_h
run_stage int8_r128_value_h formal256k
cleanup_servers

printf 'PIPELINE_COMPLETE=%s\n' "$(date --iso-8601=seconds)" >>"$OUT/pipeline.log"
