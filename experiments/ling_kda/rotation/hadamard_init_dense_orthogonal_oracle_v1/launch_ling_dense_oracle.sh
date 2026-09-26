#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RUNNER="$ROOT/experiments/ling_kda/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_ling_dense_oracle.py"
EVALUATOR="$ROOT/experiments/ling_kda/rotation/hadamard_init_dense_orthogonal_oracle_v1/evaluate_ling_dense_oracle.py"
STRESS="$ROOT/experiments/ling_kda/rotation/hadamard_init_dense_orthogonal_oracle_v1/stress_ling_dense_oracle.py"
THETA0_GATE="$ROOT/experiments/ling_kda/rotation/hadamard_init_dense_orthogonal_oracle_v1/theta0_logits_gate_ling.py"
PYTHON="${PYTHON:-/data01/user2/.conda/envs/ling-kda/bin/python}"
RESULTS="${RESULTS:-$ROOT/results/rotation/ling_dense_orthogonal_oracle_v1}"
CORPUS="${CORPUS:-$RESULTS/calibration/CALIBRATION_RAW_TEXTS.jsonl}"
TRACES="$RESULTS/traces"
mkdir -p "$RESULTS/logs"

wait_for_traces() {
  for marker in COLLECTION_TRAIN_COMPLETE.json COLLECTION_VALIDATION_COMPLETE.json COLLECTION_HELDOUT_COMPLETE.json; do
    while [[ ! -f "$TRACES/$marker" ]]; do sleep 30; done
  done
}

select_lr() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
rows = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
print(min(rows, key=lambda row: row["best_validation_primary"])["lr"])
PY
}

case "${1:-}" in
  pipeline)
    wait_for_traces
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" --phase gate --trace-dir "$TRACES" \
      --output-dir "$RESULTS/gates" >"$RESULTS/logs/gate.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$THETA0_GATE" --corpus "$CORPUS" \
      --output-dir "$RESULTS/gates" >"$RESULTS/logs/theta0_logits_gate.log" 2>&1

    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" --phase train --trace-dir "$TRACES" \
      --output-dir "$RESULTS/lr_probe/state_1e3" --objective DENSE_STATE --lr 0.001 \
      --steps 100 --validation-interval 50 --train-sequence-limit 16 \
      >"$RESULTS/logs/lr_state_1e3.log" 2>&1 & p0=$!
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$RUNNER" --phase train --trace-dir "$TRACES" \
      --output-dir "$RESULTS/lr_probe/state_3e3" --objective DENSE_STATE --lr 0.003 \
      --steps 100 --validation-interval 50 --train-sequence-limit 16 \
      >"$RESULTS/logs/lr_state_3e3.log" 2>&1 & p1=$!
    wait "$p0" "$p1"

    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" --phase train --trace-dir "$TRACES" \
      --output-dir "$RESULTS/lr_probe/functional_1e3" --objective DENSE_FUNCTIONAL --lr 0.001 \
      --steps 100 --validation-interval 50 --train-sequence-limit 16 \
      >"$RESULTS/logs/lr_functional_1e3.log" 2>&1 & p0=$!
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$RUNNER" --phase train --trace-dir "$TRACES" \
      --output-dir "$RESULTS/lr_probe/functional_3e3" --objective DENSE_FUNCTIONAL --lr 0.003 \
      --steps 100 --validation-interval 50 --train-sequence-limit 16 \
      >"$RESULTS/logs/lr_functional_3e3.log" 2>&1 & p1=$!
    wait "$p0" "$p1"

    state_lr="$(select_lr "$RESULTS/lr_probe/state_1e3/training_summary.json" "$RESULTS/lr_probe/state_3e3/training_summary.json")"
    functional_lr="$(select_lr "$RESULTS/lr_probe/functional_1e3/training_summary.json" "$RESULTS/lr_probe/functional_3e3/training_summary.json")"
    printf '{"DENSE_STATE": %s, "DENSE_FUNCTIONAL": %s}\n' "$state_lr" "$functional_lr" > "$RESULTS/lr_probe/selected_lrs.json"

    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" --phase train --trace-dir "$TRACES" \
      --output-dir "$RESULTS/training/dense_state" --objective DENSE_STATE --lr "$state_lr" \
      --steps 1000 --validation-interval 50 --seed 0 \
      >"$RESULTS/logs/train_state.log" 2>&1 & p0=$!
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$RUNNER" --phase train --trace-dir "$TRACES" \
      --output-dir "$RESULTS/training/dense_functional" --objective DENSE_FUNCTIONAL --lr "$functional_lr" \
      --steps 1000 --validation-interval 50 --seed 0 \
      >"$RESULTS/logs/train_functional.log" 2>&1 & p1=$!
    wait "$p0" "$p1"

    state_ckpt="$RESULTS/training/dense_state/best_dense_state_seed0.pt"
    functional_ckpt="$RESULTS/training/dense_functional/best_dense_functional_seed0.pt"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$EVALUATOR" --trace-dir "$TRACES" --corpus "$CORPUS" \
      --state-checkpoint "$state_ckpt" --functional-checkpoint "$functional_ckpt" \
      --output-dir "$RESULTS/evaluation" >"$RESULTS/logs/evaluation.log" 2>&1
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$STRESS" --corpus "$CORPUS" \
      --state-checkpoint "$state_ckpt" --functional-checkpoint "$functional_ckpt" \
      --persistent-result "$RESULTS/evaluation/persistent_future_kl.json" \
      --output-dir "$RESULTS/evaluation" >"$RESULTS/logs/stress_512.log" 2>&1
    ;;
  *) echo "usage: $0 pipeline" >&2; exit 2 ;;
esac
