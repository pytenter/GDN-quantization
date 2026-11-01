#!/bin/bash
set -u
set -o pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$TASK_ROOT/../.." && pwd)"
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$REPO" || exit 80
"$PY" "$TASK_ROOT/scripts/canonical60_controller.py"
rc=$?
if [ "$rc" -eq 0 ]; then
  "$PY" "$TASK_ROOT/scripts/canonical60_finalize.py" --success
  rc=$?
else
  "$PY" "$TASK_ROOT/scripts/canonical60_finalize.py" --failure
  finalize_rc=$?
  if [ "$finalize_rc" -ne 0 ]; then rc=$finalize_rc; fi
fi
printf '%s\n' "$rc" > "$TASK_ROOT/logs/offline.exitcode"
exit "$rc"
