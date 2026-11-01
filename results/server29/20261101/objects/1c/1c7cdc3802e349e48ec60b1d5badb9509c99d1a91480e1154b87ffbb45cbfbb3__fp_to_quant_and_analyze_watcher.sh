#!/usr/bin/env bash
set -u

BASE=/data/zypan
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
SCRIPT=experiments/qwen35_gdn_quant/run_end2end_bit_axis_screening.py
SHARDS=/data/zypan/results/gdn_end2end_bit_axis_screening_v1_shards
LOG=$SHARDS/fp_to_quant_and_analyze_watcher.log

mkdir -p "$SHARDS"
echo "[$(date)] watcher started" >> "$LOG"

while pgrep -f "$SCRIPT --stage main --configs fp_state" >/dev/null; do
  echo "[$(date)] waiting for fp_state fill" >> "$LOG"
  wc -l "$SHARDS"/shard_*.jsonl >> "$LOG" 2>&1 || true
  sleep 60
done

echo "[$(date)] fp_state fill finished; launching quant shards 0 and 2 if absent" >> "$LOG"
cd "$BASE" || exit 1
for i in 0 2; do
  if ! pgrep -f "$SCRIPT --stage main --configs int8_row,int8_column,int4_row,int4_column --shard-id $i --num-shards 4" >/dev/null; then
    nohup env CUDA_VISIBLE_DEVICES=$i "$PY" "$SCRIPT" \
      --stage main --configs int8_row,int8_column,int4_row,int4_column \
      --shard-id "$i" --num-shards 4 --benchmarks MATH-500 --resume \
      > "$SHARDS/quant_00${i}.log" 2>&1 &
    echo $! > "$SHARDS/quant_00${i}.pid"
    echo "[$(date)] launched quant shard $i pid $(cat "$SHARDS/quant_00${i}.pid")" >> "$LOG"
  else
    echo "[$(date)] quant shard $i already active" >> "$LOG"
  fi
done

while pgrep -f "$SCRIPT --stage main" >/dev/null; do
  echo "[$(date)] waiting for all main jobs" >> "$LOG"
  wc -l "$SHARDS"/shard_*.jsonl >> "$LOG" 2>&1 || true
  sleep 300
done

echo "[$(date)] all main jobs finished; running analyze" >> "$LOG"
cd "$BASE" || exit 1
"$PY" "$SCRIPT" --stage analyze >> "$LOG" 2>&1
echo "[$(date)] analyze finished rc=$?" >> "$LOG"
