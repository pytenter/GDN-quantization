#!/usr/bin/env bash
set -euo pipefail

EXP=/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1/experiments/LING_RECURRENT_DENSE_L6_L7_V1
PY=/data01/user2/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data01/user2/models/Ling-3.0-tiny
ROTATION="$EXP/rotations/L7_final_rotation.pt"
AMENDMENT="$EXP/manifests/protocol_amendment_l7_idle_4090_helper_v3.json"
LOGDIR="$EXP/logs/formal_L7_gpu1_helper_v3"
mkdir -p "$LOGDIR"

"$PY" - "$EXP" "$AMENDMENT" "$ROTATION" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, amendment_path, rotation = map(Path, sys.argv[1:])
amendment = json.loads(amendment_path.read_text())
if amendment.get('status') != 'FROZEN_BEFORE_REASSIGNED_GENERATION':
    raise RuntimeError('helper amendment is not frozen')
expected = {'aime26_25', 'aime26_30'}
actual = {row['problem_id'] for row in amendment.get('assignment_overrides', [])}
if actual != expected:
    raise RuntimeError(f'unexpected helper assignment: {actual}')
for qid in expected:
    path = root / 'outputs/L7' / f'{qid}_seed1.json'
    if path.exists():
        raise RuntimeError(f'canonical output already exists before helper launch: {path}')
parity = json.loads((root / 'analysis/L7_4090_3090_int8_parity.json').read_text())
if parity.get('L7_4090_3090_INT8_BITWISE_PARITY') != 'PASS':
    raise RuntimeError('L7 cross-host parity is not PASS')
if hashlib.sha256(rotation.read_bytes()).hexdigest() != '57eaebb47ff031c2b171cbb94563f814811d4f4632098af8339a27d01347060a2':
    raise RuntimeError('L7 rotation hash mismatch')
print('L7_IDLE_4090_GPU1_HELPER_V3_PREFLIGHT_PASS')
PY

curl -fsS http://127.0.0.1:31401/health >/dev/null
"$PY" "$EXP/scripts/run_formal_recurrent.py" \
  --base-url http://127.0.0.1:31401 --model "$MODEL" \
  --experiment-root "$EXP" --condition L7 --problem-ids aime26_25 aime26_30 \
  --physical-gpu 1 --instance-id L7_gpu1_4090_helper_v3 \
  --rotation-file "$ROTATION" --resume > "$LOGDIR/client.log" 2>&1
echo L7_IDLE_4090_GPU1_HELPER_V3_COMPLETE
