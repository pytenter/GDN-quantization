#!/bin/bash
# One controller, then at most one CPU finisher; preserve actual owned waits.
set -u
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
PYTHON_BIN=/data/ydai/miniconda3/envs/bitdecode/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -u -B "$ROOT/scripts/gpu_controller.py" --root "$ROOT" --execute &
controller_pid=$!
wait "$controller_pid"
controller_rc=$?
if [ ! -f "$ROOT/SESSION_RESULT.json" ]; then
  CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -u -B "$ROOT/scripts/record_offline_chain_exit.py" --root "$ROOT" --controller-pid "$controller_pid" --controller-status "$controller_rc"
  observer_rc=$?
  if [ "$observer_rc" -ne 0 ]; then exit "$observer_rc"; fi
  exit "$controller_rc"
fi
CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -u -B "$ROOT/scripts/finalize_offline.py" "$ROOT" &
finisher_pid=$!
wait "$finisher_pid"
finisher_rc=$?
CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -u -B "$ROOT/scripts/record_offline_chain_exit.py" --root "$ROOT" --controller-pid "$controller_pid" --controller-status "$controller_rc" --finisher-pid "$finisher_pid" --finisher-status "$finisher_rc"
observer_rc=$?
if [ "$observer_rc" -ne 0 ]; then exit "$observer_rc"; fi
if [ "$finisher_rc" -ne 0 ]; then exit "$finisher_rc"; fi
exit "$controller_rc"
