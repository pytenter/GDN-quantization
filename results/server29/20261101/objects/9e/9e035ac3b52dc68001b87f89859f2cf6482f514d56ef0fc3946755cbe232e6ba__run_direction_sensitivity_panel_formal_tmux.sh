#!/usr/bin/env bash
set -euo pipefail

cd /data/zypan/experiments/qwen35_gdn_quant
LOG=/data/zypan/logs/gdn_int8_residual_direction_sensitivity_panel_v1.log
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
SCRIPT=run_int8_residual_direction_sensitivity_panel.py

mkdir -p /data/zypan/logs
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] TASK=GDN_INT8_RESIDUAL_DIRECTION_SENSITIVITY_PANEL_V1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] formal start: 4 GPUs, num_shards=4, future_horizon=128, random_directions=16"
} >> "$LOG"

for sid in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="$sid"
    "$PY" "$SCRIPT" --stage formal --shard-id "$sid" --num-shards 4 --future-horizon 128 --resume
    echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] worker done: shard=$sid gpu=$sid" >> "$LOG"
  ) >> "$LOG" 2>&1 &
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] worker started: shard=$sid gpu=$sid pid=$!" >> "$LOG"
done

wait
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] all formal workers done; merging/analyzing" >> "$LOG"
"$PY" "$SCRIPT" --stage analyze --analyze-stage formal --num-shards 4 >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] formal analyze complete" >> "$LOG"
