#!/bin/bash
set -euo pipefail
REPO=/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1
ROOT=$REPO/experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1
AMEND=$ROOT/amendments/STORAGE_ACCOUNTING_CORRECTION_V1
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data/zypan/transformers-qwen35/src CUBLAS_WORKSPACE_CONFIG=:4096:8
trap 'rc=$?; printf "%s\n" "$rc" > "$AMEND/logs/long_qualification.exitcode"' EXIT
for run in 01 02 03; do
    "$PY" "$AMEND/scripts/persistent_anchor_replay.py" --mode engineering --length 256 --run "$run" > "$AMEND/logs/repeat256_${run}.log" 2>&1
done
CUDA_VISIBLE_DEVICES= "$PY" "$AMEND/scripts/qualify_long_and_streaming.py" --phase determinism256 > "$AMEND/logs/determinism256.log" 2>&1
"$PY" "$AMEND/scripts/capture_extended_drive.py" --mode engineering2048 > "$AMEND/logs/capture2048.log" 2>&1
for run in 01 02; do
    "$PY" "$AMEND/scripts/persistent_anchor_replay.py" --mode engineering --length 2048 --run "$run" > "$AMEND/logs/repeat2048_${run}.log" 2>&1
done
CUDA_VISIBLE_DEVICES= "$PY" "$AMEND/scripts/qualify_long_and_streaming.py" --phase determinism2048 > "$AMEND/logs/determinism2048.log" 2>&1
CUDA_VISIBLE_DEVICES= "$PY" "$AMEND/scripts/qualify_long_and_streaming.py" --phase streaming > "$AMEND/logs/streaming.log" 2>&1
# Scientific discovery still needs its actual full premeasurement commit.
