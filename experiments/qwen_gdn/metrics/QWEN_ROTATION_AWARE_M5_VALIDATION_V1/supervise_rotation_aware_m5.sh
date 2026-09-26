#!/usr/bin/env bash
set -u

ROOT=/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen
OUT="$ROOT/experiments/QWEN_ROTATION_AWARE_M5_VALIDATION_V1"
STATUS="$OUT/source_audit/supervisor_status.json"

cd "$ROOT" || exit 1

while true; do
  complete=0
  for shard in 0 1 2 3; do
    file=$(printf '%s/component_replay/shard_%02d/status.json' "$OUT" "$shard")
    if [ -f "$file" ] && grep -q '"status": "COMPLETE"' "$file"; then
      complete=$((complete + 1))
    fi
  done
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"status":"WAITING_FOR_COMPONENT_REPLAY","complete_shards":%d,"updated_at":"%s"}\n' "$complete" "$now" > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
  if [ "$complete" -eq 4 ]; then
    break
  fi
  live=$(tmux ls 2>/dev/null | grep -c '^m5formal' || true)
  if [ "$live" -eq 0 ]; then
    printf '{"status":"FAILED_COMPONENT_REPLAY_STOPPED","complete_shards":%d,"updated_at":"%s"}\n' "$complete" "$now" > "$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"
    exit 2
  fi
  sleep 60
done

printf '{"status":"ANALYZING","complete_shards":4,"updated_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"

if /data/ydai/miniconda3/envs/bitdecode/bin/python "$OUT/analyze_rotation_aware_m5.py" > "$OUT/source_audit/analysis.log" 2>&1; then
  printf '{"status":"COMPLETE","complete_shards":4,"updated_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
else
  code=$?
  printf '{"status":"ANALYSIS_FAILED","complete_shards":4,"exit_code":%d,"updated_at":"%s"}\n' "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
  exit "$code"
fi
