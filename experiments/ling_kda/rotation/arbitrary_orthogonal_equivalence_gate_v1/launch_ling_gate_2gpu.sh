#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data01/user2/worktrees/arbitrary-orthogonal-equivalence-gate-v1-ling}
PY=${PY:-/data01/user2/.conda/envs/ling-sglang-aime26/bin/python}
RUNNER="$REPO/experiments/ling_kda/rotation/arbitrary_orthogonal_equivalence_gate_v1/run_ling_arbitrary_orthogonal_gate.py"
OUT="$REPO/results/rotation/ling_arbitrary_orthogonal_gate_v1"
mkdir -p "$OUT/logs"

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 "$PY" "$RUNNER" --conditions identity hadamard r0 --output-dir "$OUT/gpu0" >"$OUT/logs/gpu0.log" 2>&1 & pids+=("$!")
CUDA_VISIBLE_DEVICES=1 "$PY" "$RUNNER" --conditions r1 r2 --output-dir "$OUT/gpu1" >"$OUT/logs/gpu1.log" 2>&1 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
