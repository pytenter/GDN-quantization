#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/zypan/worktrees/arbitrary-orthogonal-equivalence-gate-v1-qwen}
PY=${PY:-/data/ydai/miniconda3/envs/bitdecode/bin/python3.10}
RUNNER="$REPO/experiments/qwen_gdn/rotation/arbitrary_orthogonal_equivalence_gate_v1/run_qwen_arbitrary_orthogonal_gate.py"
OUT="$REPO/results/rotation/qwen_arbitrary_orthogonal_gate_v1"
mkdir -p "$OUT/logs"

declare -a pids=()
CUDA_VISIBLE_DEVICES=0 "$PY" "$RUNNER" --conditions identity hadamard --output-dir "$OUT/gpu0" >"$OUT/logs/gpu0.log" 2>&1 & pids+=("$!")
CUDA_VISIBLE_DEVICES=1 "$PY" "$RUNNER" --conditions r0 --output-dir "$OUT/gpu1" >"$OUT/logs/gpu1.log" 2>&1 & pids+=("$!")
CUDA_VISIBLE_DEVICES=2 "$PY" "$RUNNER" --conditions r1 --output-dir "$OUT/gpu2" >"$OUT/logs/gpu2.log" 2>&1 & pids+=("$!")
CUDA_VISIBLE_DEVICES=3 "$PY" "$RUNNER" --conditions r2 --output-dir "$OUT/gpu3" >"$OUT/logs/gpu3.log" 2>&1 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
