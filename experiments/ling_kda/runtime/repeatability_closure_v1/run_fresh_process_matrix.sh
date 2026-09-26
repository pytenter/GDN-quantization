#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data01/user2/.conda/envs/ling-kda/bin/python}"
GPU="${GPU:-1}"
ROOT="${ROOT:-/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling}"
OUT="${OUT:-results/runtime/ling_repeatability_closure_v1}"

cd "$ROOT"
mkdir -p "$OUT/fresh_process_runs" logs/ling_repeatability_closure_v1

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
  --doc-index 0
)

for seed_mode in untouched fixed; do
  for condition in Hadamard Dense_State; do
    for repeat in 0 1 2 3 4; do
      echo "fresh-process seed_mode=$seed_mode condition=$condition repeat=$repeat"
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u "${common[@]}" \
        --condition "$condition" --repeat-id "$repeat" --seed-mode "$seed_mode"
    done
  done
done
