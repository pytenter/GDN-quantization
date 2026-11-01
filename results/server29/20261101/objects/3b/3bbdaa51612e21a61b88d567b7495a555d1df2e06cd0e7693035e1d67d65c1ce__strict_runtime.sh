#!/bin/sh
# Frozen NEW_STRICT configuration/identity entry; never auto-dispatches a GPU.
set -eu
QWEN_TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
QWEN_CPU_PYTHON=${QWEN_CPU_PYTHON:-python3}
case "${1:---help}" in
 --help)
  echo 'Usage: strict_runtime.sh --check | --print-env'
  echo '--check verifies frozen CPU identity only; --print-env emits the Strict process environment.'
  echo 'Authorized bounded GPU qualification uses scripts/run_cublas_controller.py and a separate bound authorization.'
  ;;
 --check)
  exec "$QWEN_CPU_PYTHON" -B "$QWEN_TASK_ROOT/scripts/strict_identity_check.py" --root "$QWEN_TASK_ROOT"
  ;;
 --print-env)
  "$QWEN_CPU_PYTHON" -B "$QWEN_TASK_ROOT/scripts/strict_identity_check.py" --root "$QWEN_TASK_ROOT" >/dev/null
  cat <<'ENV'
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
ENV
  ;;
 *) echo 'Unsupported argument; no arbitrary or unqualified Fast process is launched.' >&2;exit 2;;
esac
