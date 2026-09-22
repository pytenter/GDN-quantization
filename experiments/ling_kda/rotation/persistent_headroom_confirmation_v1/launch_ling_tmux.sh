#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling}"
OUT="$ROOT/results/rotation/ling_persistent_headroom_confirmation_v1/streaming"
CORPUS="$ROOT/results/rotation/ling_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl"
STATE="$ROOT/results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt"
FUNCTIONAL="$ROOT/results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt"
TRACES="$ROOT/results/rotation/ling_dense_orthogonal_oracle_v1/traces"
SCRIPT="$ROOT/experiments/ling_kda/rotation/persistent_headroom_confirmation_v1/streaming_evaluate_ling.py"
PYTHON="${PYTHON:-/data01/user2/.conda/envs/ling-kda/bin/python}"
mkdir -p "$OUT/logs"

for GPU in 0 1; do
  SESSION="ling_headroom_v1_g${GPU}"
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "$SESSION already exists"; continue; }
  tmux new-session -d -s "$SESSION" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$GPU '$PYTHON' -u '$SCRIPT' --legacy-repo /data01/user2/repos/GDN-quantization --max-memory-gib 22 --trace-dir '$TRACES' --corpus '$CORPUS' --state-checkpoint '$STATE' --functional-checkpoint '$FUNCTIONAL' --output-dir '$OUT' --shard-index $GPU --shard-count 2 > '$OUT/logs/shard_${GPU}.log' 2>&1"
done
tmux ls | grep ling_headroom_v1 || true
