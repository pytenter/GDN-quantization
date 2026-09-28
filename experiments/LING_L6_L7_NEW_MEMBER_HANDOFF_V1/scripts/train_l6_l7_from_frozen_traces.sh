#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=""; TRACE_ROOT=""; RUNTIME_ROOT=""; OUTPUT_ROOT=""; GPU=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --trace-root) TRACE_ROOT="$2"; shift 2 ;;
    --runtime-root) RUNTIME_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
for name in MODEL_ROOT TRACE_ROOT RUNTIME_ROOT OUTPUT_ROOT GPU; do
  [[ -n "${!name}" ]] || { echo "--${name,,} is required" >&2; exit 2; }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINER="$REPO_ROOT/experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/recurrent_dense_train.py"
mkdir -p "$OUTPUT_ROOT/checkpoints/l6" "$OUTPUT_ROOT/checkpoints/l7" "$OUTPUT_ROOT/rotations" "$OUTPUT_ROOT/manifests"

run_trainer() {
  env CUDA_VISIBLE_DEVICES="$GPU" LING_MODEL_PATH="$MODEL_ROOT" \
    "$PYTHON_BIN" "$TRAINER" "$@"
}

# Frozen canonical mathematics: independent Hadamard initialization, H128,
# Adam, lr=0.003, weight_decay=0, global grad clip=1, seed=0, 100 updates.
run_trainer --phase train --legacy-repo "$RUNTIME_ROOT" --trace-dir "$TRACE_ROOT" \
  --objective L6_RECURRENT_DENSE_STATE --gradient-horizon 128 --steps 100 \
  --validation-interval 10 --early-stop-validations 10 \
  --train-sequence-limit 64 --validation-sequence-limit 4 --seed 0 --lr 0.003 \
  --output-dir "$OUTPUT_ROOT/checkpoints/l6"
run_trainer --phase materialize --legacy-repo "$RUNTIME_ROOT" \
  --checkpoint "$OUTPUT_ROOT/checkpoints/l6/best.pt" \
  --condition L6_RECURRENT_DENSE_STATE \
  --output-file "$OUTPUT_ROOT/rotations/L6_final_rotation.pt" \
  --manifest-file "$OUTPUT_ROOT/manifests/L6_final_rotation_materialization.json"

run_trainer --phase train --legacy-repo "$RUNTIME_ROOT" --trace-dir "$TRACE_ROOT" \
  --objective L7_RECURRENT_DENSE_FUNCTIONAL --gradient-horizon 128 --steps 100 \
  --validation-interval 10 --early-stop-validations 10 \
  --train-sequence-limit 64 --validation-sequence-limit 4 --seed 0 --lr 0.003 \
  --output-dir "$OUTPUT_ROOT/checkpoints/l7"
run_trainer --phase materialize --legacy-repo "$RUNTIME_ROOT" \
  --checkpoint "$OUTPUT_ROOT/checkpoints/l7/best.pt" \
  --condition L7_RECURRENT_DENSE_FUNCTIONAL \
  --output-file "$OUTPUT_ROOT/rotations/L7_final_rotation.pt" \
  --manifest-file "$OUTPUT_ROOT/manifests/L7_final_rotation_materialization.json"
