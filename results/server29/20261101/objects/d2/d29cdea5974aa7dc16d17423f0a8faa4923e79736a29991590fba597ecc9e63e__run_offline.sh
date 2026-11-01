#!/bin/bash
# One authorized controller execution followed by one CPU terminal finisher.
set -u
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
PYTHON_BIN=/data/ydai/miniconda3/envs/bitdecode/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -u -B "$ROOT/scripts/gpu_controller.py" --root "$ROOT" --execute
controller_rc=$?
if [ ! -f "$ROOT/SESSION_RESULT.json" ]; then
  exit "$controller_rc"
fi
CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -u -B "$ROOT/scripts/finalize_offline.py" "$ROOT"
finisher_rc=$?
if [ "$finisher_rc" -ne 0 ]; then exit "$finisher_rc"; fi
exit "$controller_rc"
