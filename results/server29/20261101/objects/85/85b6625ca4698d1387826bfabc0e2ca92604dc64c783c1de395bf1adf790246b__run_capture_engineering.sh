#!/bin/bash
set -euo pipefail
REPO=/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1
ROOT=$REPO/experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1
AMEND=$ROOT/amendments/STORAGE_ACCOUNTING_CORRECTION_V1
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data/zypan/transformers-qwen35/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$AMEND/logs"
trap 'rc=$?; printf "%s\n" "$rc" > "$AMEND/logs/capture_engineering.exitcode"' EXIT
"$PY" "$AMEND/scripts/capture_original_drive.py" --run 01 > "$AMEND/logs/capture_01.log" 2>&1
"$PY" "$AMEND/scripts/capture_original_drive.py" --run 02 > "$AMEND/logs/capture_02.log" 2>&1
CUDA_VISIBLE_DEVICES= "$PY" "$AMEND/scripts/compare_capture_reproducibility.py" > "$AMEND/logs/capture_comparison.log" 2>&1
# This stage exits after G4. The main agent must inspect its exact gate before
# implementing/launching dependent engineering stages. Never launch Discovery.
