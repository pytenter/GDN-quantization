#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling}"
PYTHON_BIN="${PYTHON_BIN:-/data01/user2/.conda/envs/ling-kda/bin/python}"
GPU="${GPU:-1}"
OUT="${OUT:-results/runtime/ling_fresh_process_first_divergence_closure_v2}"
LABEL="${LABEL:-baseline}"
DRIVER="${DRIVER:-$OUT/frozen_chunk_kda_drivers.pt}"
cd "$ROOT"
mkdir -p "$OUT/isolated/$LABEL"
for run in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u \
    experiments/ling_kda/runtime/fresh_process_first_divergence_closure_v2/isolate_chunk_kda.py \
    --driver-bundle "$DRIVER" --output "$OUT/isolated/$LABEL/run_${run}.json" --run-id "$run" --repeats 5
done
"$PYTHON_BIN" -u experiments/ling_kda/runtime/fresh_process_first_divergence_closure_v2/analyze_isolated_runs.py \
  --input-dir "$OUT/isolated/$LABEL" --output "$OUT/isolated_${LABEL}_summary.json" --label "$LABEL"
