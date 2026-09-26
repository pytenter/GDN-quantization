#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then echo "usage: $0 CONDITION ROTATION" >&2; exit 2; fi
CONDITION="$1"
ROTATION="$2"
[[ "$CONDITION" == L6 || "$CONDITION" == L7 ]] || { echo "invalid condition" >&2; exit 2; }

EXP=/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1/experiments/LING_RECURRENT_DENSE_L6_L7_V1
PY=/data01/user2/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data01/user2/models/Ling-3.0-tiny
PATCH="$EXP/scripts/sglang_kda_unified_final_r_patch.py"
GPU0_IDS=(aime26_11 aime26_15 aime26_19 aime26_23 aime26_27)
GPU1_IDS=(aime26_12 aime26_16 aime26_20 aime26_24 aime26_28)
OWN_PIDS=()

cleanup() {
  for pid in "${OWN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT INT TERM

"$PY" - "$EXP" "$ROTATION" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, rotation = Path(sys.argv[1]), Path(sys.argv[2])
condition = rotation.name.split('_', 1)[0]
if condition not in ('L6', 'L7'): raise RuntimeError('cannot derive L6/L7 condition from rotation')
value = json.loads((root / f'analysis/{condition}_4090_3090_int8_parity.json').read_text())
key = f'{condition}_4090_3090_INT8_BITWISE_PARITY'
if value.get(key) != 'PASS': raise RuntimeError(f'{key} is not PASS')
assignment = json.loads((root / 'manifests/hardware_assignment.json').read_text())
if assignment.get('status') != 'FROZEN_BEFORE_FORMAL_GENERATION': raise RuntimeError('hardware assignment is not frozen')
if not rotation.is_file(): raise RuntimeError(f'missing rotation: {rotation}')
print('FORMAL_4090_PREFLIGHT_PASS', hashlib.sha256(rotation.read_bytes()).hexdigest())
PY

d0="$EXP/logs/formal_${CONDITION}_gpu0"; d1="$EXP/logs/formal_${CONDITION}_gpu1"
mkdir -p "$d0" "$d1" "$EXP/outputs/$CONDITION"
"$EXP/scripts/launch_parity_server.sh" 0 31400 int8_r128_unified_final_r "$d0" "$PATCH" "$ROTATION" > "$d0/server.log" 2>&1 & s0=$!
"$EXP/scripts/launch_parity_server.sh" 1 31401 int8_r128_unified_final_r "$d1" "$PATCH" "$ROTATION" > "$d1/server.log" 2>&1 & s1=$!
OWN_PIDS=("$s0" "$s1")
"$PY" "$EXP/scripts/run_formal_recurrent.py" --base-url http://127.0.0.1:31400 --model "$MODEL" \
  --experiment-root "$EXP" --condition "$CONDITION" --problem-ids "${GPU0_IDS[@]}" \
  --physical-gpu 0 --instance-id "${CONDITION}_gpu0_4090" --rotation-file "$ROTATION" --resume > "$d0/client.log" 2>&1 & c0=$!
"$PY" "$EXP/scripts/run_formal_recurrent.py" --base-url http://127.0.0.1:31401 --model "$MODEL" \
  --experiment-root "$EXP" --condition "$CONDITION" --problem-ids "${GPU1_IDS[@]}" \
  --physical-gpu 1 --instance-id "${CONDITION}_gpu1_4090" --rotation-file "$ROTATION" --resume > "$d1/client.log" 2>&1 & c1=$!
status0=0; status1=0
wait "$c0" || status0=$?; wait "$c1" || status1=$?
kill "$s0" "$s1" 2>/dev/null || true; wait "$s0" 2>/dev/null || true; wait "$s1" 2>/dev/null || true
OWN_PIDS=()
[[ "$status0" -eq 0 && "$status1" -eq 0 ]] || { echo "4090 formal failure: gpu0=$status0 gpu1=$status1" >&2; exit 1; }
echo "FORMAL_4090_CONDITION_COMPLETE $CONDITION"
