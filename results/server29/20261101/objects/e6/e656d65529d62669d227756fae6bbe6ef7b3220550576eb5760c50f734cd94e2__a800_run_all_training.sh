#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1
PY=/root/miniconda3/envs/qwen-a800/bin/python
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/logs/cleanroom_training_pipeline.exitcode"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while test ! -e "$ROOT/logs/transfer.exitcode"; do sleep 20; done
test "$(< "$ROOT/logs/transfer.exitcode")" = 0
while test ! -e "$ROOT/logs/environment_install_https.exitcode"; do sleep 20; done
test "$(< "$ROOT/logs/environment_install_https.exitcode")" = 0
"$PY" "$ROOT/scripts/a800_qualify.py" --phase environment > "$ROOT/logs/environment_verify.log" 2>&1
for run in 1 2 3; do
    "$PY" "$ROOT/scripts/a800_qualify.py" --phase canary --run "$run" > "$ROOT/logs/canary_${run}.log" 2>&1
done
for run in 1 2 3; do
    "$PY" "$ROOT/scripts/a800_qualify.py" --phase full --run "$run" > "$ROOT/logs/full_model_${run}.log" 2>&1
done
"$PY" "$ROOT/scripts/a800_qualify.py" --phase close > "$ROOT/logs/qualification_close.log" 2>&1
for phase in prepare teacher update0; do
    "$PY" "$ROOT/scripts/a800_matched.py" --phase "$phase" > "$ROOT/logs/matched_${phase}.log" 2>&1
done
for arm in C5R C6R C7; do
    "$PY" "$ROOT/scripts/a800_matched.py" --phase readiness --arm "$arm" > "$ROOT/logs/readiness_${arm}.log" 2>&1
done
for arm in C5R C6R C7; do
    "$PY" "$ROOT/scripts/a800_matched.py" --phase train --arm "$arm" > "$ROOT/logs/training_${arm}.log" 2>&1
done
printf 'ALL_THREE_TRAININGS_COMPLETE_AWAITING_REQUIRED_REPRODUCTION_DEPLOYMENT_ENDPOINT\n'
