#!/usr/bin/env bash
set -euo pipefail

worker="$1"
gpu="$2"
root="/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_GDN_HADAMARD_VS_LEARNED_ROTATION_MECHANISM_CLOSURE_V1"
py="/data/ydai/miniconda3/envs/bitdecode/bin/python"
runner="$root/scripts/mechanism_capture.py"
out="$root/raw/formal"

check_space() {
  free=$(df -B1 --output=avail /data | tail -1 | tr -d ' ')
  if [ "$free" -lt 16106127360 ]; then
    echo "DISK_STOP: free=$free below 15 GiB" >&2
    exit 88
  fi
}

run_stage() {
  mode="$1"
  setname="$2"
  check_space
  log="$root/logs/worker${worker}_${mode}_${setname}.log"
  echo "START worker=$worker gpu=$gpu mode=$mode set=$setname utc=$(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES="$gpu" "$py" "$runner" \
    --mode "$mode" --condition-set "$setname" \
    --shard-index "$worker" --shard-count 4 \
    --chunk-size 32 --output-dir "$out" 2>&1 | tee "$log"
  echo "COMPLETE worker=$worker mode=$mode set=$setname utc=$(date -u +%FT%TZ)"
}

code=0
{
  run_stage controlled all
  run_stage recurrent baseline
  run_stage recurrent learned
} || code=$?
echo "$code" > "$root/logs/worker${worker}.exitcode"
exit "$code"
