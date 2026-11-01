#!/usr/bin/env bash
# One Bash-owned parent, one controller, no automatic retry or restart.
set +e
root="$1"
source_sha="$2"
permit_sha="$3"
python_bin=/data/ydai/miniconda3/envs/bitdecode/bin/python
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
export PYTHONOPTIMIZE=0
cd "$root" || exit 91
"$python_bin" -B scripts/experiment_parent.py --source-sha "$source_sha" --permit-sha "$permit_sha" > runtime/PARENT_STDOUT.log 2>&1 &
child=$!
"$python_bin" -B - "$root" "$child" "$source_sha" "$permit_sha" <<'PY'
import sys,os,time
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'));import execution_common as C
C.atomic(r/'runtime/PARENT_DISPATCH.json',dict(bash=C.identity(os.getppid()),child=C.identity(int(sys.argv[2])),source_sha256=sys.argv[3],permit_sha256=sys.argv[4],start_monotonic=time.monotonic()))
PY
dispatch_rc=$?
# Even if CPU dispatch recording fails, retain the actual owning wait; never
# detach a live parent or attempt a second parent launch.
wait "$child"
code=$?
"$python_bin" -B - "$root" "$code" "$dispatch_rc" "$source_sha" "$permit_sha" <<'PY'
import sys,os,time,json
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'));import execution_common as C
d=json.loads((r/'runtime/PARENT_DISPATCH.json').read_text()) if (r/'runtime/PARENT_DISPATCH.json').exists() else None
C.atomic(r/'runtime/BASH_WAIT.json',dict(status='ACTUAL_OWNING_BASH_WAIT',bash=C.identity(os.getppid()),parent_dispatch=d,bash_wait_integer=int(sys.argv[2]),dispatch_recording_exit=int(sys.argv[3]),source_sha256=sys.argv[4],permit_sha256=sys.argv[5],observed_monotonic=time.monotonic(),not_inferred=True))
PY
exit "$code"
