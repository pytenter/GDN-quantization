#!/bin/bash
set -u
R="$1"
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
"$PY" -u -B "$R/scripts/run_remainder.py" --root "$R" --execute > "$R/controller.log" 2>&1
rc=$?
"$PY" -u -B "$R/scripts/finalize_cpu.py" "$R" > "$R/finisher.log" 2>&1
exit $rc
