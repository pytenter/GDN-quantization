#!/usr/bin/env bash
set -euo pipefail
cd /data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
"$PY" scripts/run_full_probe_parity.py --phase plan
"$PY" scripts/run_full_probe_parity.py --phase reference --index 1
"$PY" scripts/run_full_probe_parity.py --phase strict --index 1
"$PY" scripts/run_full_probe_parity.py --phase parity
for i in 2 3 4 5; do
  "$PY" scripts/run_full_probe_parity.py --phase strict --index "$i"
done
"$PY" scripts/run_full_probe_parity.py --phase determinism
"$PY" scripts/benchmark_patch.py --phase reference
"$PY" scripts/benchmark_patch.py --phase strict
"$PY" scripts/benchmark_patch.py --phase analyze
