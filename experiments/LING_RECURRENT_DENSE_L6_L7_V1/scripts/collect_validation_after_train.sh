#!/usr/bin/env bash
set -euo pipefail

EXP=/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1
PY=/data/zypan/envs/ling-kda/bin/python3
REPO="$EXP/runtime_repo"
CORPUS="$EXP/source_traces/CALIBRATION_RAW_TEXTS.jsonl"

while [[ ! -f "$EXP/traces/COLLECTION_TRAIN_SHARD0_COMPLETE.json" || ! -f "$EXP/traces/COLLECTION_TRAIN_SHARD1_COMPLETE.json" ]]; do
  p0="$(cat "$EXP/logs/collect_train_shard0_gpu3.pid")"
  p1="$(cat "$EXP/logs/collect_train_shard1_gpu4.pid")"
  if ! kill -0 "$p0" 2>/dev/null || ! kill -0 "$p1" 2>/dev/null; then
    echo "train collection exited without both PASS sentinels" >&2
    exit 1
  fi
  sleep 30
done

env CUDA_VISIBLE_DEVICES=3 LING_MODEL_PATH=/data/zypan/models/Ling-3.0-tiny \
  "$PY" "$EXP/scripts/recurrent_dense_train.py" --phase collect --legacy-repo "$REPO" \
  --corpus "$CORPUS" --old-trace-dir "$EXP/source_traces" --split validation \
  --sequence-length 512 --shard-index 0 --num-shards 2 --output-dir "$EXP/traces" \
  > "$EXP/logs/collect_validation_shard0_gpu3.log" 2>&1 & p0=$!
echo "$p0" > "$EXP/logs/collect_validation_shard0_gpu3.pid"

env CUDA_VISIBLE_DEVICES=4 LING_MODEL_PATH=/data/zypan/models/Ling-3.0-tiny \
  "$PY" "$EXP/scripts/recurrent_dense_train.py" --phase collect --legacy-repo "$REPO" \
  --corpus "$CORPUS" --old-trace-dir "$EXP/source_traces" --split validation \
  --sequence-length 512 --shard-index 1 --num-shards 2 --output-dir "$EXP/traces" \
  > "$EXP/logs/collect_validation_shard1_gpu4.log" 2>&1 & p1=$!
echo "$p1" > "$EXP/logs/collect_validation_shard1_gpu4.pid"

s0=0; s1=0
wait "$p0" || s0=$?
wait "$p1" || s1=$?
[[ "$s0" -eq 0 && "$s1" -eq 0 ]] || { echo "validation collection failed: gpu3=$s0 gpu4=$s1" >&2; exit 1; }
[[ -f "$EXP/traces/COLLECTION_VALIDATION_SHARD0_COMPLETE.json" && -f "$EXP/traces/COLLECTION_VALIDATION_SHARD1_COMPLETE.json" ]]
echo "RECURRENT_TRACE_COLLECTION_COMPLETE"
