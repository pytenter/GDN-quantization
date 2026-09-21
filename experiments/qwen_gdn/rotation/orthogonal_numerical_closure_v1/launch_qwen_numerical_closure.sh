#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/zypan/worktrees/orthogonal-numerical-closure-v1-qwen}
PY=${PY:-/data/ydai/miniconda3/envs/bitdecode/bin/python3.10}
RUNNER="$REPO/experiments/qwen_gdn/rotation/orthogonal_numerical_closure_v1/run_qwen_orthogonal_numerical_closure.py"
OUT="$REPO/results/rotation/qwen_orthogonal_numerical_closure_v1"
mkdir -p "$OUT/logs"
CUDA_VISIBLE_DEVICES=0 "$PY" "$RUNNER" --output-dir "$OUT" >"$OUT/logs/run.log" 2>&1
