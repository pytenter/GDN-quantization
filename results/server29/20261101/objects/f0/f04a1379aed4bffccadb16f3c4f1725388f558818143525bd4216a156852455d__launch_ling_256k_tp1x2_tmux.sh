#!/usr/bin/env bash
set -euo pipefail

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
SESSION=ling256_tp1x2_formal

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -n formal \
  "env SKIP_MATCHED_BASELINE=1 bash -lc '$EXP/run_ling_256k_tp1x2_formal.sh'"
echo "launched tmux session: $SESSION"
