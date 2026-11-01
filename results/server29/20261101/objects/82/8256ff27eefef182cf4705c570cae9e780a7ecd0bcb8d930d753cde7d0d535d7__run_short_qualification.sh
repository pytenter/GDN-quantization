#!/bin/bash
set -euo pipefail
REPO=/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1
ROOT=$REPO/experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1
AMEND=$ROOT/amendments/STORAGE_ACCOUNTING_CORRECTION_V1
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data/zypan/transformers-qwen35/src CUBLAS_WORKSPACE_CONFIG=:4096:8
trap 'rc=$?; printf "%s\n" "$rc" > "$AMEND/logs/short_qualification.exitcode"' EXIT
for phase in factorization schema fp invariance decomposition isolation; do
    "$PY" "$AMEND/scripts/isolated_replay_engineering.py" --phase "$phase" > "$AMEND/logs/qualification_${phase}.log" 2>&1
done
# G11/G12 and Discovery remain forbidden until the exact completed gates above
# are inspected and the remaining qualification is frozen and run successfully.
