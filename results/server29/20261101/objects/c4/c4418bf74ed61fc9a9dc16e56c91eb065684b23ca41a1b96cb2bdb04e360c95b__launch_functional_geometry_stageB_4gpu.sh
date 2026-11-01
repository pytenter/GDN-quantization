#!/usr/bin/env bash
set -euo pipefail
cd /data/zypan/experiments/qwen35_gdn_quant
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_functional_geometry_audit.py --stage stageB-worker --worker-id gpu0 --task-indices 0,4,8,12,16,20,24 > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_gpu0.log 2>&1 &
echo $! > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_gpu0.pid
CUDA_VISIBLE_DEVICES=1 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_functional_geometry_audit.py --stage stageB-worker --worker-id gpu1 --task-indices 1,5,9,13,17,21,25 > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_gpu1.log 2>&1 &
echo $! > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_gpu1.pid
CUDA_VISIBLE_DEVICES=2 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_functional_geometry_audit.py --stage stageB-worker --worker-id gpu2 --task-indices 2,6,10,14,18,22,26 > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_gpu2.log 2>&1 &
echo $! > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_gpu2.pid
CUDA_VISIBLE_DEVICES=3 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_functional_geometry_audit.py --stage stageB-worker --worker-id gpu3 --task-indices 3,7,11,15,19,23 > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_gpu3.log 2>&1 &
echo $! > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_gpu3.pid
wait
/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_functional_geometry_audit.py --stage stageB-merge > /data/zypan/runs/gdn_int8_same_norm_functional_geometry_audit_v1/stageB_merge.log 2>&1
