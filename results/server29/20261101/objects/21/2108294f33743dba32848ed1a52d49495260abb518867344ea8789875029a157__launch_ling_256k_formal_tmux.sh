#!/usr/bin/env bash
set -euo pipefail

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
SESSION=ling256_formal

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

# The resume driver itself enforces the Stage 0 PASS gate.  Re-running this
# launcher is safe because each successful sample is atomically committed and
# validated before being skipped.
tmux new-session -d -s "$SESSION" -n formal \
  "bash -lc '$EXP/resume_ling_256k_formal.sh'"
echo "launched tmux session: $SESSION"

