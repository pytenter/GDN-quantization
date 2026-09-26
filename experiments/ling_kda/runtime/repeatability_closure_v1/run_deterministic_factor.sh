#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data01/user2/.conda/envs/ling-kda/bin/python}"
GPU="${GPU:-1}"
ROOT="${ROOT:-/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling}"
OUT="${OUT:-results/runtime/ling_repeatability_closure_v1/factor_deterministic_algorithms}"

cd "$ROOT"
mkdir -p "$OUT"
for repeat in 0 1 2; do
  echo "deterministic-algorithms factor repeat=$repeat"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u \
    experiments/ling_kda/runtime/repeatability_closure_v1/run_ling_repeatability.py \
    --mode single --condition Hadamard --doc-index 0 --repeat-id "$repeat" \
    --seed-mode fixed --deterministic-algorithms \
    --legacy-repo /data01/user2/repos/GDN-quantization --max-memory-gib 22 \
    --trace-dir results/rotation/ling_dense_orthogonal_oracle_v1/traces \
    --corpus results/rotation/ling_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl \
    --state-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt \
    --functional-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt \
    --output-dir "$OUT"
done
