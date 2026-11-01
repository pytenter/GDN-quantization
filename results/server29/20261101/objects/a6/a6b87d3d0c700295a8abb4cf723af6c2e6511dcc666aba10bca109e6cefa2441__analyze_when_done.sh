#!/usr/bin/env bash
set -u
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
SCRIPT=/data/zypan/experiments/qwen35_gdn_quant/run_int8_orientation_state_change_mechanism.py
LOGDIR=/data/zypan/results/gdn_int8_orientation_state_change_mechanism_v1_formal_logs
LOG=$LOGDIR/analyze_when_done.log
while pgrep -f 'run_int8_orientation_state_change_mechanism.py --stage formal' >/dev/null; do
  {
    echo "[$(date)] waiting formal"
    ps -eo pid,etime,pcpu,args | grep 'run_int8_orientation_state_change_mechanism.py --stage formal' | grep -v grep || true
    wc -l /data/zypan/results/gdn_int8_orientation_state_change_mechanism_v1_formal_records_shard_*.jsonl 2>&1 || true
  } >> "$LOG"
  sleep 300
done
cd /data/zypan || exit 1
"$PY" "$SCRIPT" --stage analyze --num-shards 4 >> "$LOG" 2>&1
echo "[$(date)] analyze rc=$?" >> "$LOG"
