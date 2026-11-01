#!/usr/bin/env bash
set -euo pipefail

cd /data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2/backward_repeatability_closure/trajectory_equivalence
python=/data/ydai/miniconda3/envs/bitdecode/bin/python

while pgrep -f '[t]rajectory_functional_analysis.py --phase dual' >/dev/null; do
  sleep 30
done

if [[ ! -f analysis/update8_functional_probe.json ]]; then
  echo 'functional analysis ended without required artifact; refusing finalizer' >&2
  exit 1
fi
if compgen -G 'analysis/*.error.json' >/dev/null; then
  echo 'error artifact present; refusing finalizer' >&2
  exit 1
fi

"$python" trajectory_finalize.py
sha256sum -c hashes/artifact_sha256.txt > logs/hash_verification.log
echo FINAL_REPORT_AND_HASH_MANIFEST_VERIFIED
