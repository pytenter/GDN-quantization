#!/usr/bin/env bash
set -euo pipefail

EXP=/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1
PY=/data/zypan/envs/ling-kda/bin/python3
REPO="$EXP/runtime_repo"
MODEL=/data/zypan/models/Ling-3.0-tiny
TRAINER="$EXP/scripts/recurrent_dense_train.py"
LOG="$EXP/logs/training_pipeline.log"

exec > >(tee -a "$LOG") 2>&1
echo "PIPELINE_START $(date --iso-8601=seconds)"

while [[ ! -f "$EXP/traces/COLLECTION_VALIDATION_SHARD0_COMPLETE.json" || ! -f "$EXP/traces/COLLECTION_VALIDATION_SHARD1_COMPLETE.json" ]]; do
  watcher="$(cat "$EXP/logs/collect_validation_watcher.pid")"
  if ! kill -0 "$watcher" 2>/dev/null; then
    echo "validation watcher exited without both PASS sentinels" >&2
    exit 1
  fi
  sleep 30
done

"$PY" "$EXP/scripts/finalize_trace_manifests.py" \
  --trace-dir "$EXP/traces" \
  --corpus "$EXP/source_traces/CALIBRATION_RAW_TEXTS.jsonl" \
  --eval-samples "$EXP/source_traces/eval_samples.jsonl" \
  --output-dir "$EXP/manifests"
cp "$EXP/manifests/training_data_leakage.json" "$EXP/analysis/training_data_leakage.json"

env CUDA_VISIBLE_DEVICES=3 LING_MODEL_PATH="$MODEL" \
  "$PY" "$TRAINER" --phase smoke --legacy-repo "$REPO" \
  --trace-dir "$EXP/traces" \
  --output-file "$EXP/analysis/gradient_feasibility.json"

"$PY" "$EXP/scripts/finalize_training_metadata.py" verify-smoke \
  --smoke "$EXP/analysis/gradient_feasibility.json" \
  --provenance "$EXP/analysis/recurrent_state_provenance.json"

env CUDA_VISIBLE_DEVICES=3 LING_MODEL_PATH="$MODEL" \
  "$PY" "$TRAINER" --phase horizon --legacy-repo "$REPO" \
  --trace-dir "$EXP/traces" --max-horizon-probe-seconds 900 \
  --output-file "$EXP/analysis/horizon_feasibility.json"

horizon="$("$PY" "$EXP/scripts/finalize_training_metadata.py" freeze-horizon \
  --horizon "$EXP/analysis/horizon_feasibility.json" \
  --config "$EXP/configs/recurrent_training_config.json")"
echo "FROZEN_GRADIENT_HORIZON=$horizon"

env CUDA_VISIBLE_DEVICES=3 LING_MODEL_PATH="$MODEL" \
  "$PY" "$TRAINER" --phase train --legacy-repo "$REPO" \
  --trace-dir "$EXP/traces" --objective L6_RECURRENT_DENSE_STATE \
  --gradient-horizon "$horizon" --steps 100 --validation-interval 10 \
  --train-sequence-limit 64 --validation-sequence-limit 4 --seed 0 --lr 0.003 \
  --output-dir "$EXP/checkpoints/l6"

env CUDA_VISIBLE_DEVICES=3 LING_MODEL_PATH="$MODEL" \
  "$PY" "$TRAINER" --phase materialize --legacy-repo "$REPO" \
  --checkpoint "$EXP/checkpoints/l6/best.pt" \
  --condition L6_RECURRENT_DENSE_STATE \
  --output-file "$EXP/rotations/L6_final_rotation.pt" \
  --manifest-file "$EXP/analysis/L6_final_rotation_materialization.json"

env CUDA_VISIBLE_DEVICES=3 LING_MODEL_PATH="$MODEL" \
  "$PY" "$TRAINER" --phase train --legacy-repo "$REPO" \
  --trace-dir "$EXP/traces" --objective L7_RECURRENT_DENSE_FUNCTIONAL \
  --gradient-horizon "$horizon" --steps 100 --validation-interval 10 \
  --train-sequence-limit 64 --validation-sequence-limit 4 --seed 0 --lr 0.003 \
  --output-dir "$EXP/checkpoints/l7"

env CUDA_VISIBLE_DEVICES=3 LING_MODEL_PATH="$MODEL" \
  "$PY" "$TRAINER" --phase materialize --legacy-repo "$REPO" \
  --checkpoint "$EXP/checkpoints/l7/best.pt" \
  --condition L7_RECURRENT_DENSE_FUNCTIONAL \
  --output-file "$EXP/rotations/L7_final_rotation.pt" \
  --manifest-file "$EXP/analysis/L7_final_rotation_materialization.json"

cp "$EXP/checkpoints/l6/training_curve.json" "$EXP/analysis/training_curves_l6.json"
cp "$EXP/checkpoints/l7/training_curve.json" "$EXP/analysis/training_curves_l7.json"

env CUDA_VISIBLE_DEVICES=3 LING_MODEL_PATH="$MODEL" \
  "$PY" "$EXP/scripts/sanity_compare.py" \
  --trainer "$TRAINER" --legacy-repo "$REPO" --trace-dir "$EXP/traces" \
  --h "$EXP/rotations/H_final_rotation.pt" \
  --l4 "$EXP/rotations/L4_old_final_rotation.pt" \
  --l5 "$EXP/rotations/L5_old_final_rotation.pt" \
  --l6 "$EXP/rotations/L6_final_rotation.pt" \
  --l7 "$EXP/rotations/L7_final_rotation.pt" \
  --sequences 4 --output "$EXP/analysis/sanity_metrics.json"

"$PY" "$EXP/scripts/finalize_training_metadata.py" aggregate \
  --l6-summary "$EXP/checkpoints/l6/training_summary.json" \
  --l7-summary "$EXP/checkpoints/l7/training_summary.json" \
  --l6-materialization "$EXP/analysis/L6_final_rotation_materialization.json" \
  --l7-materialization "$EXP/analysis/L7_final_rotation_materialization.json" \
  --h-materialization "$EXP/analysis/H_final_rotation_materialization.json" \
  --sanity "$EXP/analysis/sanity_metrics.json" \
  --orthogonality "$EXP/analysis/orthogonality.json" \
  --final-materialization "$EXP/analysis/final_rotation_materialization.json" \
  --complete "$EXP/analysis/TRAINING_PIPELINE_COMPLETE.json"

echo "PIPELINE_COMPLETE $(date --iso-8601=seconds)"
