#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=/root/miniconda3/envs/qwen-a800/bin/python
export PYTHONUNBUFFERED=1
test ! -e "$ROOT/logs/posttrain_closure_pipeline.exitcode"
test ! -e "$ROOT/provenance/preclosure_training_artifact_inventory.json"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/logs/posttrain_closure_pipeline.exitcode"' EXIT
cd "$ROOT/scripts"
nice -n 19 ionice -c 3 "$PY" posttrain_closure_audit.py
"$PY" posttrain_closure_runner.py
nice -n 19 ionice -c 3 "$PY" posttrain_closure_finalize.py
printf 'STOP_BEFORE_3090_DEPLOYMENT_AND_AIME\n'
