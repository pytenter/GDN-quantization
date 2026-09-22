#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling}"
PYTHON_BIN="${PYTHON_BIN:-/data01/user2/.conda/envs/ling-kda/bin/python}"
GPU="${GPU:-1}"
OUT="${OUT:-results/runtime/ling_fresh_process_first_divergence_closure_v2/closure_matrix}"
cd "$ROOT"
mkdir -p "$OUT/fresh_process_runs"
common=(
  experiments/ling_kda/runtime/repeatability_closure_v1/run_ling_repeatability.py
  --mode single
  --legacy-repo /data01/user2/repos/GDN-quantization
  --max-memory-gib 22
  --trace-dir results/rotation/ling_dense_orthogonal_oracle_v1/traces
  --corpus results/rotation/ling_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl
  --state-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt
  --functional-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt
  --output-dir "$OUT"
  --seed-mode fixed
)
for doc in 0 7 15; do
  for condition in Hadamard Dense_State; do
    for repeat in 0 1 2 3 4; do
      echo "closure doc=$doc condition=$condition repeat=$repeat"
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u "${common[@]}" \
        --doc-index "$doc" --condition "$condition" --repeat-id "$repeat"
    done
  done
done
"$PYTHON_BIN" -u experiments/ling_kda/runtime/fresh_process_first_divergence_closure_v2/analyze_closure_matrix.py \
  --input-dir "$OUT/fresh_process_runs" \
  --output "$OUT/closure_matrix_summary.json"
