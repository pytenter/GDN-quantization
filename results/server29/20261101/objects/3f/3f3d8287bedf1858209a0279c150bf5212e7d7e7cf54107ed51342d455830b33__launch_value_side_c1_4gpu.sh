#!/usr/bin/env bash
set -euo pipefail
cd /data/zypan/experiments/qwen35_gdn_quant
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_value_side_functional_direction_formal_and_transfer.py --stage c1-worker --worker-id gpu0 --task-indices 0,4,8,12,16,20,24,28,32 > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_gpu0.log 2>&1 &
echo $! > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_gpu0.pid
CUDA_VISIBLE_DEVICES=1 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_value_side_functional_direction_formal_and_transfer.py --stage c1-worker --worker-id gpu1 --task-indices 1,5,9,13,17,21,25,29,33 > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_gpu1.log 2>&1 &
echo $! > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_gpu1.pid
CUDA_VISIBLE_DEVICES=2 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_value_side_functional_direction_formal_and_transfer.py --stage c1-worker --worker-id gpu2 --task-indices 2,6,10,14,18,22,26,30,34 > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_gpu2.log 2>&1 &
echo $! > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_gpu2.pid
CUDA_VISIBLE_DEVICES=3 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_value_side_functional_direction_formal_and_transfer.py --stage c1-worker --worker-id gpu3 --task-indices 3,7,11,15,19,23,27,31,35 > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_gpu3.log 2>&1 &
echo $! > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_gpu3.pid
wait
/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_value_side_functional_direction_formal_and_transfer.py --stage c1-merge > /data/zypan/runs/gdn_int8_value_side_functional_direction_formal_and_transfer_v1/stageC1_merge.log 2>&1
