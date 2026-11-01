#!/bin/sh
set -eu
ROOT='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_repair_20261025'
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
[ "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)" = "$ROOT" ] || exit 71
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE CUBLAS_WORKSPACE_CONFIG OMP_NUM_THREADS MKL_NUM_THREADS CUDA_DEVICE_ORDER
exec "$PY" scripts/offline_parent.py --root "$ROOT" --execute
