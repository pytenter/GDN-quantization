#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1
PY=/root/miniconda3/envs/qwen-a800/bin/python
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/logs/posttraining_pipeline.exitcode"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while test ! -e "$ROOT/logs/cleanroom_training_pipeline_qualified.exitcode"; do sleep 60; done
test "$(< "$ROOT/logs/cleanroom_training_pipeline_qualified.exitcode")" = 0
"$PY" "$ROOT/scripts/a800_posttraining.py" > "$ROOT/logs/posttraining.log" 2>&1
