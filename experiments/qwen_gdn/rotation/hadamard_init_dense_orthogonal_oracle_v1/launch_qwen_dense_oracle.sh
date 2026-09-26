#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RUNNER="$ROOT/experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py"
RESULTS="${RESULTS:-$ROOT/results/rotation/qwen_dense_orthogonal_oracle_v1}"
CORPUS="${CORPUS:?set CORPUS to CALIBRATION_RAW_TEXTS.jsonl}"
TRACES="$RESULTS/traces"
mkdir -p "$TRACES" "$RESULTS/logs"

case "${1:-}" in
  collect)
    for split in train validation heldout; do
      CUDA_VISIBLE_DEVICES=0 python "$RUNNER" --phase collect --corpus "$CORPUS" --split "$split" --output-dir "$TRACES" 2>&1 | tee "$RESULTS/logs/collect_${split}.log"
    done
    ;;
  gate)
    CUDA_VISIBLE_DEVICES=0 python "$RUNNER" --phase gate --trace-dir "$TRACES" --output-dir "$RESULTS/gates" 2>&1 | tee "$RESULTS/logs/gate.log"
    ;;
  lr-probe)
    for spec in "0 DENSE_STATE 0.001" "1 DENSE_STATE 0.003" "2 DENSE_FUNCTIONAL 0.001" "3 DENSE_FUNCTIONAL 0.003"; do
      set -- $spec; gpu="$1"; objective="$2"; lr="$3"
      CUDA_VISIBLE_DEVICES="$gpu" python "$RUNNER" --phase train --trace-dir "$TRACES" --output-dir "$RESULTS/lr_probe/${objective}_lr${lr}" --objective "$objective" --lr "$lr" --steps 100 --validation-interval 50 --train-sequence-limit 16 >"$RESULTS/logs/lr_${objective}_${lr}.log" 2>&1 &
    done
    wait
    ;;
  *) echo "usage: $0 {collect|gate|lr-probe}" >&2; exit 2 ;;
esac
