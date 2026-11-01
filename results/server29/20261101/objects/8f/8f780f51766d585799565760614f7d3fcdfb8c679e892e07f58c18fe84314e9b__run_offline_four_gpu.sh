#!/bin/bash
set -eu
ROOT='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_continue_20261026'
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE CUBLAS_WORKSPACE_CONFIG OMP_NUM_THREADS MKL_NUM_THREADS CUDA_DEVICE_ORDER
"$PY" "$ROOT/scripts/record_four_gpu_wait.py" --root "$ROOT" --mode start --bash-pid "$$"
"$PY" "$ROOT/scripts/offline_four_gpu_parent.py" --root "$ROOT" --execute &
PACE_CHILD_PID=$!
"$PY" "$ROOT/scripts/record_four_gpu_wait.py" --root "$ROOT" --mode child --bash-pid "$$" --child-pid "$PACE_CHILD_PID"
set +e
wait "$PACE_CHILD_PID"
PACE_WAIT_RC=$?
set -e
"$PY" "$ROOT/scripts/record_four_gpu_wait.py" --root "$ROOT" --mode exit --bash-pid "$$" --returncode "$PACE_WAIT_RC"
exit "$PACE_WAIT_RC"
