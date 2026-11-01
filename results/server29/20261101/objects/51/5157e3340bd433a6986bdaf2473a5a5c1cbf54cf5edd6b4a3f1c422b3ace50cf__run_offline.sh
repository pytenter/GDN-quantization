#!/bin/sh
set -u
R=$1
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
"$PY" -u -B "$R/scripts/gpu_controller.py" --root "$R" --execute > "$R/controller.log" 2>&1
# A true STOP also receives a truthful sealed delivery; never retries any GPU.
"$PY" -u -B "$R/scripts/finalize_offline.py" "$R" > "$R/finisher.log" 2>&1
