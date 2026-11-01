#!/usr/bin/env bash
set -u

cd /data/zypan/experiments/qwen35_gdn_quant
LOG=/data/zypan/logs/gdn_int8_residual_strength_causal_intervention_v1.log
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
SCRIPT=run_int8_residual_strength_causal_intervention.py

mkdir -p /data/zypan/logs
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] TASK=GDN_INT8_RESIDUAL_STRENGTH_CAUSAL_INTERVENTION_V1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] formal start: 4 GPUs, num_shards=4, max_steps=512"
} >> "$LOG"

for sid in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$sid" "$PY" "$SCRIPT" --stage formal --shard-id "$sid" --num-shards 4 --max-steps 512 --resume >> "$LOG" 2>&1 &
  pids[$sid]=$!
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] worker started: shard=$sid gpu=$sid pid=${pids[$sid]}" >> "$LOG"
done

rc=0
for sid in 0 1 2 3; do
  wait "${pids[$sid]}"
  one=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] worker finished: shard=$sid rc=$one" >> "$LOG"
  if [ "$one" -ne 0 ]; then
    rc=1
  fi
done

if [ "$rc" -eq 0 ]; then
  "$PY" "$SCRIPT" --stage analyze --analyze-stage formal --num-shards 4 >> "$LOG" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] formal analyze complete" >> "$LOG"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] formal analyze skipped because one or more workers failed" >> "$LOG"
  exit "$rc"
fi
