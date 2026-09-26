#!/usr/bin/env bash
set -euo pipefail

EXP=/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1
LOG="$EXP/logs/cross_host_parity_watcher.log"
exec >> "$LOG" 2>&1
echo "WAIT_VALIDATION_COLLECTION $(date --iso-8601=seconds)"
while [[ ! -f "$EXP/traces/COLLECTION_VALIDATION_SHARD0_COMPLETE.json" || ! -f "$EXP/traces/COLLECTION_VALIDATION_SHARD1_COMPLETE.json" ]]; do
  watcher="$(cat "$EXP/logs/collect_validation_watcher.pid")"
  if ! kill -0 "$watcher" 2>/dev/null; then
    echo "validation collector stopped before PASS sentinels" >&2
    exit 1
  fi
  sleep 30
done
echo "START_CROSS_HOST_PARITY $(date --iso-8601=seconds)"
"$EXP/scripts/run_cross_host_parity_3090.sh"
echo "CROSS_HOST_PARITY_COMPLETE $(date --iso-8601=seconds)"
