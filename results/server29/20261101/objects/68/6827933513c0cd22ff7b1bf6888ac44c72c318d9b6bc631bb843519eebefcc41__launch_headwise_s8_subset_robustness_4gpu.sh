#!/usr/bin/env bash
set -euo pipefail
cd /data/zypan/experiments/qwen35_gdn_quant
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_headwise_s8_subset_robustness_audit.py --stage worker --worker-id gpu0 --unit-indices 0,4,8,12,16 > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_gpu0.log 2>&1 &
echo $! > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_gpu0.pid
CUDA_VISIBLE_DEVICES=1 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_headwise_s8_subset_robustness_audit.py --stage worker --worker-id gpu1 --unit-indices 1,5,9,13,17 > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_gpu1.log 2>&1 &
echo $! > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_gpu1.pid
CUDA_VISIBLE_DEVICES=2 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_headwise_s8_subset_robustness_audit.py --stage worker --worker-id gpu2 --unit-indices 2,6,10,14 > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_gpu2.log 2>&1 &
echo $! > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_gpu2.pid
CUDA_VISIBLE_DEVICES=3 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_headwise_s8_subset_robustness_audit.py --stage worker --worker-id gpu3 --unit-indices 3,7,11,15 > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_gpu3.log 2>&1 &
echo $! > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_gpu3.pid
wait
/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_headwise_s8_subset_robustness_audit.py --stage merge > /data/zypan/results/gdn_int8_headwise_s8_subset_robustness_audit_v1_merge.log 2>&1
