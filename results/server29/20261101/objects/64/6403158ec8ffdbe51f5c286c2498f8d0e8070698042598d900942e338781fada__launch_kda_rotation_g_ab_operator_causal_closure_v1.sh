#!/usr/bin/env bash
set -euo pipefail

REPO=/data01/user2/repos/GDN-quantization
PY=/data01/user2/.conda/envs/ling-kda/bin/python
RUNNER="$REPO/experiments/rotation/run_kda_rotation_g_ab_operator_causal_closure_v1.py"
PHASE="${1:-round1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="kda_g_ab_${PHASE}_v1"
LOGDIR="$REPO/results/kda_rotation_g_ab_operator_causal_closure_v1_${PHASE}_tmux_logs_${STAMP}"
mkdir -p "$LOGDIR"

for SHARD in 0 1; do
  GPU="$SHARD"
  OUT="$REPO/results/kda_rotation_g_ab_operator_causal_closure_v1_${PHASE}_shard${SHARD}_gpu${GPU}"
  CMD="cd '$REPO' && CUDA_VISIBLE_DEVICES=$GPU '$PY' '$RUNNER' --scope formal --phase '$PHASE' --unit-shard-index '$SHARD' --unit-shard-count 2 --output-dir '$OUT' --overwrite --keep-going 2>&1 | tee '$LOGDIR/shard${SHARD}_gpu${GPU}.log'"
  if [[ "$SHARD" == 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "shard0" "$CMD"
  else
    tmux new-window -t "$SESSION" -n "shard1" "$CMD"
  fi
done

printf '%s\n' "$LOGDIR" > "$REPO/results/kda_rotation_g_ab_operator_causal_closure_v1_${PHASE}_latest_tmux_logdir.txt"
echo "SESSION=$SESSION"
echo "LOGDIR=$LOGDIR"
