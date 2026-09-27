#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then echo "usage: $0 H|L6|L7" >&2; exit 2; fi
CONDITION="$1"
[[ "$CONDITION" == H || "$CONDITION" == L6 || "$CONDITION" == L7 ]] || { echo "invalid condition" >&2; exit 2; }

BASE=/data/zypan/experiments
EXP="$BASE/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1"
SOURCE="$BASE/LING_RECURRENT_DENSE_L6_L7_V1"
PY=/data/zypan/envs/ling-sglang-aime26/bin/python
MODEL=/data/zypan/models/Ling-3.0-tiny
PATCH="$SOURCE/scripts/sglang_kda_unified_final_r_patch.py"
ROTATION="$SOURCE/rotations/${CONDITION}_final_rotation.pt"
OWN_PIDS=()

cleanup() {
  for pid in "${OWN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT INT TERM

"$PY" - "$EXP" "$CONDITION" "$ROTATION" <<'PY'
import hashlib,json,pathlib,sys
exp=pathlib.Path(sys.argv[1]); condition=sys.argv[2]; rotation=pathlib.Path(sys.argv[3])
stage=json.loads((exp/'provenance/stage0_audit.json').read_text())
if stage['gates']['PROVENANCE_GATE']!='PASS': raise RuntimeError('provenance gate')
if stage['parity'][condition]['status']!='PASS': raise RuntimeError('parity gate')
key='H_rotation' if condition=='H' else f'{condition}_rotation'
actual=hashlib.sha256(rotation.read_bytes()).hexdigest()
if actual!=stage['artifacts'][key]['expected']: raise RuntimeError('rotation identity gate')
if hashlib.sha256((exp/'scripts/aime26_scorer_v4.py').read_bytes()).hexdigest()!='fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b': raise RuntimeError('scorer gate')
print('CONFIRMATORY_3090_PREFLIGHT_PASS',condition,actual)
PY

d3="$EXP/logs/${CONDITION}_3090_gpu3"; d4="$EXP/logs/${CONDITION}_3090_gpu4"
mkdir -p "$d3" "$d4" "$EXP/outputs/$CONDITION"
"$SOURCE/scripts/launch_formal_server_3090.sh" 3 31603 int8_r128_unified_final_r "$d3" "$PATCH" "$ROTATION" > "$d3/server.log" 2>&1 & s3=$!
"$SOURCE/scripts/launch_formal_server_3090.sh" 4 31604 int8_r128_unified_final_r "$d4" "$PATCH" "$ROTATION" > "$d4/server.log" 2>&1 & s4=$!
OWN_PIDS=("$s3" "$s4")
"$PY" "$EXP/scripts/run_confirmatory_sample.py" --base-url http://127.0.0.1:31603 --model "$MODEL" --experiment-root "$EXP" --source-root "$SOURCE" --condition "$CONDITION" --shard 3090_gpu3 --hardware-class RTX3090 --physical-gpu 3 --instance-id "${CONDITION}_3090_gpu3" --rotation-file "$ROTATION" --resume > "$d3/client.log" 2>&1 & c3=$!
"$PY" "$EXP/scripts/run_confirmatory_sample.py" --base-url http://127.0.0.1:31604 --model "$MODEL" --experiment-root "$EXP" --source-root "$SOURCE" --condition "$CONDITION" --shard 3090_gpu4 --hardware-class RTX3090 --physical-gpu 4 --instance-id "${CONDITION}_3090_gpu4" --rotation-file "$ROTATION" --resume > "$d4/client.log" 2>&1 & c4=$!
status3=0; status4=0
wait "$c3" || status3=$?; wait "$c4" || status4=$?
kill "$s3" "$s4" 2>/dev/null || true; wait "$s3" 2>/dev/null || true; wait "$s4" 2>/dev/null || true
OWN_PIDS=()
[[ "$status3" -eq 0 && "$status4" -eq 0 ]] || { echo "3090 condition failure: gpu3=$status3 gpu4=$status4" >&2; exit 1; }
echo "CONFIRMATORY_3090_CONDITION_COMPLETE $CONDITION"
