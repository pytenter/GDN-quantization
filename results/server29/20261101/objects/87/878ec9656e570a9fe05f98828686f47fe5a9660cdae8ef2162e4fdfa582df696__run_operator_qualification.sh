#!/usr/bin/env bash
set -euo pipefail
cd /data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
for i in 1 2 3 4 5; do
  "$PY" scripts/test_deterministic_cumsum.py --mode strict --run "$i"
done
"$PY" scripts/qualify_operator.py --phase determinism
"$PY" scripts/test_deterministic_cumsum.py --mode reference --run 1
"$PY" scripts/qualify_operator.py --phase freeze
"$PY" scripts/qualify_operator.py --phase parity
