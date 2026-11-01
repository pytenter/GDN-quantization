#!/bin/bash
# One owning Bash launch. No set-e: always record the actual integer wait.
HMMT_TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HMMT_PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
"$HMMT_PY" -B - "$HMMT_TASK_ROOT" "$1" "$2" <<'PY'
import os,sys
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'));import execution_common as C
C.atomic(r/'runtime/BASH_STARTED.json',dict(owner=C.identity(os.getppid()),source_sha256=sys.argv[2],permit_sha256=sys.argv[3]))
PY
if [ "$?" -ne 0 ]; then exit 1; fi
"$HMMT_PY" -B "$HMMT_TASK_ROOT/scripts/experiment_parent.py" --source-sha "$1" --permit-sha "$2" >> "$HMMT_TASK_ROOT/runtime/PARENT_STDOUT.log" 2>&1 &
HMMT_PARENT_PID=$!
wait "$HMMT_PARENT_PID"
HMMT_PARENT_WAIT=$?
"$HMMT_PY" -B - "$HMMT_TASK_ROOT" "$1" "$2" "$HMMT_PARENT_PID" "$HMMT_PARENT_WAIT" <<'PY'
import os,sys
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'));import execution_common as C
C.atomic(r/'runtime/BASH_WAIT.json',dict(owning_Bash_wait_integer=int(sys.argv[5]),owning_Bash=C.identity(os.getppid()),parent_pid=int(sys.argv[4]),source_sha256=sys.argv[2],permit_sha256=sys.argv[3]))
PY
if [ "$?" -ne 0 ]; then exit 1; fi
exit "$HMMT_PARENT_WAIT"
