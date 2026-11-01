#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1
PYTHON=/data/ydai/miniconda3/envs/bitdecode/bin/python
SCRIPT="$ROOT/scripts/postprocess_canonical60.py"

echo "[$(date -u +%FT%TZ)] OFFLINE_WATCHER_START"
while true; do
  ready=1
  for worker in 0 1 2 3; do
    if [[ ! -f "$ROOT/logs/worker${worker}.exitcode" ]]; then
      ready=0
    fi
  done
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  sleep 60
done

for worker in 0 1 2 3; do
  rc=$(tr -d '[:space:]' < "$ROOT/logs/worker${worker}.exitcode")
  if [[ "$rc" != "0" ]]; then
    echo "[$(date -u +%FT%TZ)] BLOCKED worker=$worker exitcode=$rc"
    exit 20
  fi
done

echo "[$(date -u +%FT%TZ)] ALL_WORKERS_EXITED_ZERO"
"$PYTHON" "$SCRIPT" --final
echo "[$(date -u +%FT%TZ)] OFFLINE_POSTPROCESS_COMPLETE"
