#!/usr/bin/env bash
set -u

cd /data/zypan/experiments/qwen35_gdn_quant
LOG=/data/zypan/logs/gdn_int8_effective_update_small_prompt_validation_v1.log
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
SCRIPT=run_int8_effective_update_small_prompt_validation.py

mkdir -p /data/zypan/logs
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] TASK=GDN_INT8_EFFECTIVE_UPDATE_METRIC_SMALL_PROMPT_VALIDATION_V1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] formal start: dual GPU only, num_shards=2, max_steps=512"
} >> "$LOG"

CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" --stage formal --shard-id 0 --num-shards 2 --max-steps 512 --resume >> "$LOG" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$SCRIPT" --stage formal --shard-id 1 --num-shards 2 --max-steps 512 --resume >> "$LOG" 2>&1 &
pid1=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] workers started: shard0=$pid0 gpu0, shard1=$pid1 gpu1" >> "$LOG"

wait "$pid0"
rc0=$?
wait "$pid1"
rc1=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] workers finished: shard0_rc=$rc0 shard1_rc=$rc1" >> "$LOG"

if [ "$rc0" -eq 0 ] && [ "$rc1" -eq 0 ]; then
  "$PY" "$SCRIPT" --stage analyze --analyze-stage formal --num-shards 2 >> "$LOG" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] formal analyze complete" >> "$LOG"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] formal analyze skipped because a worker failed" >> "$LOG"
  exit 1
fi
