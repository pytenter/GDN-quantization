#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then echo "usage: $0 H|L6|L7" >&2; exit 2; fi
CONDITION="$1"
[[ "$CONDITION" == H || "$CONDITION" == L6 || "$CONDITION" == L7 ]] || { echo "invalid condition" >&2; exit 2; }

REPO=/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1
EXP="$REPO/experiments/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1"
SOURCE="$REPO/experiments/LING_RECURRENT_DENSE_L6_L7_V1"
PY=/data01/user2/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data01/user2/models/Ling-3.0-tiny
PATCH="$SOURCE/scripts/sglang_kda_unified_final_r_patch.py"
ROTATION="$SOURCE/rotations/${CONDITION}_final_rotation.pt"
OWN_PIDS=()

cleanup() {
  for pid in "${OWN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT INT TERM

"$PY" - "$EXP" "$SOURCE" "$CONDITION" "$ROTATION" <<'PY'
import hashlib,json,pathlib,sys
exp=pathlib.Path(sys.argv[1]); source=pathlib.Path(sys.argv[2]); condition=sys.argv[3]; rotation=pathlib.Path(sys.argv[4])
stage=json.loads((exp/'provenance/stage0_audit.json').read_text())
if stage['gates']['PROVENANCE_GATE']!='PASS': raise RuntimeError('provenance gate')
if stage['parity'][condition]['status']!='PASS': raise RuntimeError('parity gate')
key='H_rotation' if condition=='H' else f'{condition}_rotation'
actual=hashlib.sha256(rotation.read_bytes()).hexdigest()
if actual!=stage['artifacts'][key]['expected']: raise RuntimeError('rotation identity gate')
if hashlib.sha256((exp/'scripts/aime26_scorer_v4.py').read_bytes()).hexdigest()!='fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b': raise RuntimeError('scorer gate')
print('CONFIRMATORY_4090_PREFLIGHT_PASS',condition,actual)
PY

d0="$EXP/logs/${CONDITION}_4090_gpu0"; d1="$EXP/logs/${CONDITION}_4090_gpu1"
mkdir -p "$d0" "$d1" "$EXP/outputs/$CONDITION"
"$SOURCE/scripts/launch_parity_server.sh" 0 31600 int8_r128_unified_final_r "$d0" "$PATCH" "$ROTATION" > "$d0/server.log" 2>&1 & s0=$!
"$SOURCE/scripts/launch_parity_server.sh" 1 31601 int8_r128_unified_final_r "$d1" "$PATCH" "$ROTATION" > "$d1/server.log" 2>&1 & s1=$!
OWN_PIDS=("$s0" "$s1")
"$PY" "$EXP/scripts/run_confirmatory_sample.py" --base-url http://127.0.0.1:31600 --model "$MODEL" --experiment-root "$EXP" --source-root "$SOURCE" --condition "$CONDITION" --shard 4090_gpu0 --hardware-class RTX4090 --physical-gpu 0 --instance-id "${CONDITION}_4090_gpu0" --rotation-file "$ROTATION" --resume > "$d0/client.log" 2>&1 & c0=$!
"$PY" "$EXP/scripts/run_confirmatory_sample.py" --base-url http://127.0.0.1:31601 --model "$MODEL" --experiment-root "$EXP" --source-root "$SOURCE" --condition "$CONDITION" --shard 4090_gpu1 --hardware-class RTX4090 --physical-gpu 1 --instance-id "${CONDITION}_4090_gpu1" --rotation-file "$ROTATION" --resume > "$d1/client.log" 2>&1 & c1=$!
status0=0; status1=0
wait "$c0" || status0=$?; wait "$c1" || status1=$?
kill "$s0" "$s1" 2>/dev/null || true; wait "$s0" 2>/dev/null || true; wait "$s1" 2>/dev/null || true
OWN_PIDS=()
[[ "$status0" -eq 0 && "$status1" -eq 0 ]] || { echo "4090 condition failure: gpu0=$status0 gpu1=$status1" >&2; exit 1; }
echo "CONFIRMATORY_4090_CONDITION_COMPLETE $CONDITION"
