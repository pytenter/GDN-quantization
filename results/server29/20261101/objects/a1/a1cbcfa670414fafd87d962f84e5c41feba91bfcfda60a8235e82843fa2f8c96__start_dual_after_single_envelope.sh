#!/usr/bin/env bash
set -euo pipefail

cd /data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2/backward_repeatability_closure/trajectory_equivalence
python=/data/ydai/miniconda3/envs/bitdecode/bin/python

while pgrep -f '[t]rajectory_functional_analysis.py --phase single' >/dev/null; do
  if compgen -G 'analysis/*_probe.error.json' >/dev/null; then
    echo 'probe error found; refusing dual launch' >&2
    exit 1
  fi
  sleep 30
done

"$python" - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('.')
for name, hash_name in (
    ('frozen_single_gpu_trajectory_envelope.json', 'frozen_single_gpu_trajectory_envelope_hash.json'),
    ('frozen_single_gpu_functional_envelope.json', 'frozen_single_gpu_functional_envelope_hash.json'),
):
    path = root / 'analysis' / name
    meta = json.loads((root / 'analysis' / hash_name).read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise SystemExit(f'frozen envelope hash mismatch: {name}')
summary = json.loads((root / 'analysis/single_gpu_10run_summary.json').read_text())
if summary['SINGLE_GPU_FORWARD_GATE'] != 'PASS' or len(summary['runs']) != 10:
    raise SystemExit('single trajectory gate incomplete')
for index in range(1, 11):
    if not (root / 'analysis' / f'single_{index:02d}_probe.json').is_file():
        raise SystemExit(f'missing single probe {index}')
if list((root / 'analysis').glob('*.error.json')):
    raise SystemExit('error file found; refusing dual launch')
print('SINGLE_ENVELOPES_HASH_VERIFIED', flush=True)
PY

echo DUAL_TRAJECTORIES_START
bash run_dual_trajectories.sh > logs/dual_batch.log 2>&1
echo DUAL_TRAJECTORIES_5_COMPLETE
