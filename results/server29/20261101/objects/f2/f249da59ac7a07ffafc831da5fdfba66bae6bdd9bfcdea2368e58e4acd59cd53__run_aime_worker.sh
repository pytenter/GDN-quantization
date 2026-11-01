#!/usr/bin/env bash
set -euo pipefail

WORKER_ID="${1:?worker id 0..3 required}"
case "$WORKER_ID" in 0|1|2|3) ;; *) echo "invalid worker id" >&2; exit 2;; esac

ROOT=/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1
CLOSURE=/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1
PYTHON=/data/ydai/miniconda3/envs/bitdecode/bin/python3.10
RUNNER="$ROOT/scripts/run_qwen_c5_c6_formal.py"
CONFIG_SHA=557ac6670fe6e88606be226ccf0dea289ebd34dd4c2be54e7e54b9b6ee9e0836
MIN_FREE=$((15 * 1024 * 1024 * 1024))

export CUDA_VISIBLE_DEVICES="$WORKER_ID"
export QWEN_C5_C6_OUT_ROOT="$ROOT/outputs"
export PYTHONDONTWRITEBYTECODE=1

run_condition() {
  local method="$1"
  local condition="${method^^}"
  local free
  free=$(df -B1 /data | awk 'NR==2 {print $4}')
  if [[ "$free" -lt "$MIN_FREE" ]]; then
    echo "DISK_STOP: free=$free threshold=$MIN_FREE" >&2
    exit 90
  fi
  echo "[$(date -u +%FT%TZ)] START $condition worker=$WORKER_ID free=$free"
  "$PYTHON" "$RUNNER" \
    --stage formal \
    --method "$method" \
    --worker-id "$WORKER_ID" \
    --num-workers 4 \
    --resume \
    --rotation-file "$CLOSURE/deployment/$condition/frozen_rotation_fp32.pt" \
    --deployment-manifest "$CLOSURE/deployment/$condition/deployment_manifest.json" \
    --formal-config-sha256 "$CONFIG_SHA"
  "$PYTHON" "$RUNNER" \
    --stage verify \
    --method "$method" \
    --worker-id "$WORKER_ID" \
    --num-workers 4
  echo "[$(date -u +%FT%TZ)] COMPLETE $condition worker=$WORKER_ID"
}

run_condition c5
run_condition c6
echo "[$(date -u +%FT%TZ)] WORKER_COMPLETE worker=$WORKER_ID"
