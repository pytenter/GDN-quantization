#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1
TASK_PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd /data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1
test ! -e "$TASK_ROOT/logs/phase_a_repaired_001.exitcode"
trap 'code=$?; printf "%s\n" "$code" > "$TASK_ROOT/logs/phase_a_repaired_001.exitcode"' EXIT
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
"$TASK_PY" -u "$TASK_ROOT/scripts/propagation_phase_a_repaired_001.py" --mode execute
# No downstream scientific unit can be launched by this wrapper.
