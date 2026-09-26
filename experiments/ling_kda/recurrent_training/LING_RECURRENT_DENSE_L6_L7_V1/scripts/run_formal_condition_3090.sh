#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then echo "usage: $0 CONDITION ROTATION" >&2; exit 2; fi
CONDITION="$1"
ROTATION="$2"
[[ "$CONDITION" == L6 || "$CONDITION" == L7 ]] || { echo "invalid condition" >&2; exit 2; }

EXP=/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1
PY=/data/zypan/envs/ling-sglang-aime26/bin/python
MODEL=/data/zypan/models/Ling-3.0-tiny
PATCH="$EXP/scripts/sglang_kda_unified_final_r_patch.py"
GPU3_IDS=(aime26_13 aime26_17 aime26_21 aime26_25 aime26_29)
GPU4_IDS=(aime26_14 aime26_18 aime26_22 aime26_26 aime26_30)
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
gate = json.loads((root / f'analysis/{condition}_4090_3090_int8_parity.json').read_text())
key = f'{condition}_4090_3090_INT8_BITWISE_PARITY'
if gate.get(key) != 'PASS': raise RuntimeError(f'{key} is not PASS')
assignment = json.loads((root / 'manifests/hardware_assignment.json').read_text())
if assignment.get('status') != 'FROZEN_BEFORE_FORMAL_GENERATION': raise RuntimeError('hardware assignment is not frozen')
if not rotation.is_file(): raise RuntimeError(f'missing rotation: {rotation}')
print('FORMAL_3090_PREFLIGHT_PASS', hashlib.sha256(rotation.read_bytes()).hexdigest())
PY

d3="$EXP/logs/formal_${CONDITION}_gpu3"; d4="$EXP/logs/formal_${CONDITION}_gpu4"
mkdir -p "$d3" "$d4" "$EXP/outputs/$CONDITION"
"$EXP/scripts/launch_formal_server_3090.sh" 3 31503 int8_r128_unified_final_r "$d3" "$PATCH" "$ROTATION" > "$d3/server.log" 2>&1 & s3=$!
"$EXP/scripts/launch_formal_server_3090.sh" 4 31504 int8_r128_unified_final_r "$d4" "$PATCH" "$ROTATION" > "$d4/server.log" 2>&1 & s4=$!
OWN_PIDS=("$s3" "$s4")
"$PY" "$EXP/scripts/run_formal_recurrent.py" --base-url http://127.0.0.1:31503 --model "$MODEL" \
  --experiment-root "$EXP" --condition "$CONDITION" --problem-ids "${GPU3_IDS[@]}" \
  --physical-gpu 3 --instance-id "${CONDITION}_gpu3_3090" --rotation-file "$ROTATION" --resume > "$d3/client.log" 2>&1 & c3=$!
"$PY" "$EXP/scripts/run_formal_recurrent.py" --base-url http://127.0.0.1:31504 --model "$MODEL" \
  --experiment-root "$EXP" --condition "$CONDITION" --problem-ids "${GPU4_IDS[@]}" \
  --physical-gpu 4 --instance-id "${CONDITION}_gpu4_3090" --rotation-file "$ROTATION" --resume > "$d4/client.log" 2>&1 & c4=$!
status3=0; status4=0
wait "$c3" || status3=$?; wait "$c4" || status4=$?
kill "$s3" "$s4" 2>/dev/null || true; wait "$s3" 2>/dev/null || true; wait "$s4" 2>/dev/null || true
OWN_PIDS=()
[[ "$status3" -eq 0 && "$status4" -eq 0 ]] || { echo "3090 formal failure: gpu3=$status3 gpu4=$status4" >&2; exit 1; }
echo "FORMAL_3090_CONDITION_COMPLETE $CONDITION"
