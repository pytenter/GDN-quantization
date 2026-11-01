#!/usr/bin/env bash
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
"$PY" -u -B "$R/scripts/gpu_controller.py" --root "$R" --execute >"$R/controller.log" 2>&1
controller_rc=$?
"$PY" -u -B "$R/scripts/finalize_offline.py" "$R" >"$R/finisher.log" 2>&1
finisher_rc=$?
exit "$finisher_rc"
