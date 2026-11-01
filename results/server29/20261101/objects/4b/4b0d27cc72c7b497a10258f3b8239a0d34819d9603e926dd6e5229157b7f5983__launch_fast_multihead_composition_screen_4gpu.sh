#!/usr/bin/env bash
set -euo pipefail
cd /data/zypan/experiments/qwen35_gdn_quant
export PYTHONUNBUFFERED=1
LOGDIR=/data/zypan/results

CUDA_VISIBLE_DEVICES=0 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_fast_multihead_composition_screen.py --stage worker --worker-id gpu0 --unit-indices 0,4,8 > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_gpu0.log 2>&1 &
echo $! > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_gpu0.pid
CUDA_VISIBLE_DEVICES=1 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_fast_multihead_composition_screen.py --stage worker --worker-id gpu1 --unit-indices 1,5 > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_gpu1.log 2>&1 &
echo $! > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_gpu1.pid
CUDA_VISIBLE_DEVICES=2 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_fast_multihead_composition_screen.py --stage worker --worker-id gpu2 --unit-indices 2,6 > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_gpu2.log 2>&1 &
echo $! > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_gpu2.pid
CUDA_VISIBLE_DEVICES=3 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_fast_multihead_composition_screen.py --stage worker --worker-id gpu3 --unit-indices 3,7 > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_gpu3.log 2>&1 &
echo $! > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_gpu3.pid
wait
/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 run_int8_same_norm_fast_multihead_composition_screen.py --stage merge > $LOGDIR/gdn_int8_same_norm_fast_multihead_composition_screen_v1_merge.log 2>&1
