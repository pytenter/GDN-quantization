#!/usr/bin/env bash
set -u
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
BASE=/data/zypan
SHARDS=/data/zypan/results/gdn_end2end_bit_axis_screening_v1_shards
LOG=$SHARDS/p0_int8_row_analyze_when_done.log
while pgrep -f 'run_end2end_bit_axis_screening.py --stage main --configs int8_row' >/dev/null; do
  {
    echo "[$(date)] waiting for P0 int8_row"
    wc -l "$SHARDS"/shard_*.jsonl 2>&1 || true
    "$PY" - <<'PY'
import json, glob, collections
c=collections.Counter()
for p in glob.glob('/data/zypan/results/gdn_end2end_bit_axis_screening_v1_shards/shard_*.jsonl'):
    for line in open(p, encoding='utf-8'):
        if line.strip(): c[json.loads(line).get('quantizer')]+=1
print(dict(c))
PY
  } >> "$LOG"
  sleep 300
done
cd "$BASE" || exit 1
"$PY" experiments/qwen35_gdn_quant/analyze_int8_row_e2e_completion_v1.py >> "$LOG" 2>&1
"$PY" experiments/qwen35_gdn_quant/run_end2end_bit_axis_screening.py --stage analyze >> "$LOG" 2>&1
