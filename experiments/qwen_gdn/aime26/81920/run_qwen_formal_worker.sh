#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WORKER_ID PHYSICAL_GPU" >&2
  exit 2
fi

WORKER_ID="$1"
PHYSICAL_GPU="$2"
REPO="/data/zypan/worktrees/aime26-sglang-rotation-v1"
PYTHON="/data/ydai/miniconda3/envs/bitdecode/bin/python3.10"
LAUNCH_ROOT="$REPO/artifacts/aime26_v2/official_sampling_81920/qwen/launch"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PYTHONUNBUFFERED=1
cd "$REPO"

mkdir -p "$LAUNCH_ROOT"
echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$PYTHON"
echo "worker_id=$WORKER_ID physical_gpu=$PHYSICAL_GPU assigned_units=45"
echo "config=thinking:on do_sample:true temperature:1.0 top_p:0.95 top_k:20 min_p:0.0 presence_penalty:1.5 repetition_penalty:1.0 max_new_tokens:81920 seeds:1,2"

for METHOD in fp_state int8_c128 int8_c128_key_h; do
  echo "START method=$METHOD timestamp=$(date --iso-8601=seconds)"
  "$PYTHON" -u experiments/aime26/run_qwen_aime26_formal.py \
    --stage formal --method "$METHOD" --worker-id "$WORKER_ID" --num-workers 4 --resume
  echo "DONE method=$METHOD timestamp=$(date --iso-8601=seconds)"
done
