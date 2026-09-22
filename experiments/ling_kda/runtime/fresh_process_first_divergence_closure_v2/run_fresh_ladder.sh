#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling}"
PYTHON_BIN="${PYTHON_BIN:-/data01/user2/.conda/envs/ling-kda/bin/python}"
GPU="${GPU:-1}"
OUT="${OUT:-results/runtime/ling_fresh_process_first_divergence_closure_v2}"
SCRIPT="experiments/ling_kda/runtime/fresh_process_first_divergence_closure_v2/capture_layer0_ladder.py"
cd "$ROOT"
mkdir -p "$OUT/ladder" logs/ling_fresh_process_first_divergence_closure_v2

common=(
  "$SCRIPT"
  --legacy-repo /data01/user2/repos/GDN-quantization
  --max-memory-gib 22
  --trace-dir results/rotation/ling_dense_orthogonal_oracle_v1/traces
  --corpus results/rotation/ling_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl
  --state-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt
  --functional-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt
  --output-dir "$OUT/ladder"
  --target-layer "${TARGET_LAYER:-0}"
)

for run in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u "${common[@]}" --run-id "$run" --same-process-repeats "${SAME_PROCESS_REPEATS:-5}"
done

"$PYTHON_BIN" -u experiments/ling_kda/runtime/fresh_process_first_divergence_closure_v2/analyze_ladder.py \
  --input-dir "$OUT/ladder" \
  --output "$OUT/first_divergence_ladder.json" \
  --keep-fixed-input "$OUT/temporary_fixed_operator_input.pt"
