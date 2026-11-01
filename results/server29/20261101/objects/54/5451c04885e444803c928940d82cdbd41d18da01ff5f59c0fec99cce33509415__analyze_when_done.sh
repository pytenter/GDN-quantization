#!/usr/bin/env bash
LOG=/data/zypan/results/gdn_end2end_bit_axis_screening_v1_shards/analyze_when_done.log
while pgrep -f 'run_end2end_bit_axis_screening.py --stage main' >/dev/null; do
  date >> "$LOG"
  wc -l /data/zypan/results/gdn_end2end_bit_axis_screening_v1_shards/shard_*.jsonl >> "$LOG" 2>&1 || true
  sleep 300
done
cd /data/zypan
/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 \
  experiments/qwen35_gdn_quant/run_end2end_bit_axis_screening.py \
  --stage analyze >> "$LOG" 2>&1
