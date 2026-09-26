#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen}"
OUT="$ROOT/results/rotation/qwen_persistent_headroom_confirmation_v1/streaming"
CORPUS="$ROOT/results/rotation/qwen_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl"
STATE="$ROOT/results/rotation/qwen_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt"
FUNCTIONAL="$ROOT/results/rotation/qwen_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt"
SCRIPT="$ROOT/experiments/qwen_gdn/rotation/persistent_headroom_confirmation_v1/streaming_evaluate_qwen.py"
PYTHON="${PYTHON:-/data/ydai/miniconda3/envs/bitdecode/bin/python3.10}"
mkdir -p "$OUT/logs"

for GPU in 0 1 2 3; do
  SESSION="qwen_headroom_v1_g${GPU}"
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "$SESSION already exists"; continue; }
  tmux new-session -d -s "$SESSION" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$GPU '$PYTHON' -u '$SCRIPT' --legacy-root /data/zypan --corpus '$CORPUS' --state-checkpoint '$STATE' --functional-checkpoint '$FUNCTIONAL' --output-dir '$OUT' --shard-index $GPU --shard-count 4 > '$OUT/logs/shard_${GPU}.log' 2>&1"
done
tmux ls | grep qwen_headroom_v1 || true
