#!/usr/bin/env bash
set -euo pipefail

cd /data/zypan/experiments/qwen35_gdn_quant

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

LOG="/data/zypan/results/gdn_int8_same_norm_distributed_composition_functional_alignment_v1_offline.log"
PID="/data/zypan/results/gdn_int8_same_norm_distributed_composition_functional_alignment_v1_offline.pid"

nohup /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 \
  run_int8_same_norm_distributed_composition_functional_alignment.py \
  --stage pilot > "${LOG}" 2>&1 &

echo $! > "${PID}"
echo "started pid=$(cat "${PID}") gpu=${CUDA_VISIBLE_DEVICES} log=${LOG}"
