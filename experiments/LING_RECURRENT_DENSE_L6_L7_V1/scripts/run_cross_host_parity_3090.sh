#!/usr/bin/env bash
set -euo pipefail

EXP=/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1
PY=/data/zypan/envs/ling-sglang-aime26/bin/python
MODEL=/data/zypan/models/Ling-3.0-tiny
PATCH="$EXP/scripts/sglang_kda_unified_final_r_patch.py"
H_FINAL="$EXP/rotations/H_final_rotation.pt"
REFERENCE_JSON="$EXP/cross_host_reference/forced_reference.json"
ROOT="$EXP/logs/cross_host_h_parity"
OWN_PIDS=()

cleanup() {
  for pid in "${OWN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT INT TERM

run_one() {
  local kind="$1" mode="$2" dir="$ROOT/candidate_$kind" server client status=0
  [[ ! -e "$dir/client_result.json" ]] || { echo "refusing to overwrite $dir" >&2; return 2; }
  mkdir -p "$dir"
  "$EXP/scripts/launch_formal_server_3090.sh" 4 31504 "$mode" "$dir" "$PATCH" "$H_FINAL" > "$dir/server.log" 2>&1 & server=$!
  OWN_PIDS=("$server")
  "$PY" "$EXP/scripts/run_runtime_gate_client.py" --base-url http://127.0.0.1:31504 \
    --model "$MODEL" --mode "$mode" --run-dir "$dir" --reference-json "$REFERENCE_JSON" \
    --wait-seconds 1200 > "$dir/client.log" 2>&1 & client=$!
  wait "$client" || status=$?
  kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true
  OWN_PIDS=()
  [[ "$status" -eq 0 ]] || { echo "cross-host parity client failed: $kind status=$status" >&2; return 1; }
}

[[ -f "$ROOT/reference_fp/client_result.json" && -f "$ROOT/reference_int8/client_result.json" ]] || {
  echo "missing copied RTX4090 parity references" >&2; exit 2;
}
run_one fp fp_state_unified_final_r
run_one int8 int8_r128_unified_final_r
"$PY" "$EXP/scripts/analyze_cross_host_parity.py" --root "$ROOT" --h-final "$H_FINAL" \
  --output "$EXP/analysis/cross_host_4090_3090_parity.json"
