#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data01/user2/worktrees/orthogonal-numerical-closure-v1-ling}
PY=${PY:-/data01/user2/.conda/envs/ling-kda/bin/python}
RUNNER="$REPO/experiments/ling_kda/rotation/orthogonal_numerical_closure_v1/run_ling_orthogonal_numerical_closure.py"
OUT="$REPO/results/rotation/ling_orthogonal_numerical_closure_v1"
mkdir -p "$OUT/logs"
CUDA_VISIBLE_DEVICES=0,1 "$PY" "$RUNNER" --output-dir "$OUT" >"$OUT/logs/run.log" 2>&1
