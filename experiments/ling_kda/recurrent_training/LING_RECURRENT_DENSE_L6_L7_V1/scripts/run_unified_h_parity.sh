#!/usr/bin/env bash
set -euo pipefail

EXP=/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1/experiments/LING_RECURRENT_DENSE_L6_L7_V1
OLD_PATCH=/data01/user2/worktrees/ling-sglang-dense-aime26-last20-256k-singleseed-v1/experiments/aime26/sglang_kda_runtime_patch.py
NEW_PATCH="$EXP/scripts/sglang_kda_unified_final_r_patch.py"
H_FINAL="$EXP/rotations/H_final_rotation.pt"
PY=/data01/user2/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data01/user2/models/Ling-3.0-tiny
REFERENCE=/data01/user2/worktrees/ling-sglang-dense-aime26-last20-256k-singleseed-v1/experiments/LING_SGLANG_DENSE_AIME26_LAST20_256K_SINGLESEED_V1/logs/runtime/forced_reference.json
ROOT="$EXP/logs/unified_h_parity"
OWN_PIDS=()

cleanup() {
  for pid in "${OWN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT INT TERM

run_pair() {
  local kind="$1" old_mode="$2" new_mode="$3"
  local old_dir="$ROOT/old_l2_$kind" new_dir="$ROOT/unified_h_$kind"
  local server client status=0
  [[ ! -e "$old_dir/client_result.json" && ! -e "$new_dir/client_result.json" ]] || {
    echo "refusing to overwrite parity result for $kind" >&2; return 2;
  }
  mkdir -p "$old_dir" "$new_dir"
  "$EXP/scripts/launch_parity_server.sh" 1 31301 "$old_mode" "$old_dir" "$OLD_PATCH" "" > "$old_dir/server.log" 2>&1 & server=$!
  OWN_PIDS=("$server")
  "$PY" "$EXP/scripts/run_runtime_gate_client.py" --base-url http://127.0.0.1:31301 --model "$MODEL" --mode "$old_mode" --run-dir "$old_dir" --reference-json "$REFERENCE" --wait-seconds 1200 > "$old_dir/client.log" 2>&1 & client=$!
  wait "$client" || status=$?
  kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true
  OWN_PIDS=()
  [[ "$status" -eq 0 ]] || { echo "old L2 parity client failed: $status" >&2; return 1; }
  status=0
  "$EXP/scripts/launch_parity_server.sh" 1 31301 "$new_mode" "$new_dir" "$NEW_PATCH" "$H_FINAL" > "$new_dir/server.log" 2>&1 & server=$!
  OWN_PIDS=("$server")
  "$PY" "$EXP/scripts/run_runtime_gate_client.py" --base-url http://127.0.0.1:31301 --model "$MODEL" --mode "$new_mode" --run-dir "$new_dir" --reference-json "$REFERENCE" --wait-seconds 1200 > "$new_dir/client.log" 2>&1 & client=$!
  wait "$client" || status=$?
  kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true
  OWN_PIDS=()
  [[ "$status" -eq 0 ]] || { echo "unified-H parity client failed: $status" >&2; return 1; }
  echo "PAIR_PASS $kind"
}

[[ -f "$REFERENCE" ]] || { echo "missing frozen forced reference" >&2; exit 2; }
run_pair fp fp_state_value_h fp_state_unified_final_r
run_pair int8 int8_r128_value_h int8_r128_unified_final_r
"$PY" "$EXP/scripts/analyze_unified_h_parity.py" --root "$ROOT" --h-final "$H_FINAL" --output "$EXP/analysis/unified_h_parity.json"
