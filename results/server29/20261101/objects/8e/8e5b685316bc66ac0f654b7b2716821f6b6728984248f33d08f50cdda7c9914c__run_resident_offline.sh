#!/usr/bin/env bash
# One controller and one chained CPU finalizer. No retry, restart or loop.
set -euo pipefail
ROOT="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON=/data/ydai/miniconda3/envs/bitdecode/bin/python
mkdir -p "$ROOT/logs" "$ROOT/operations"
"$PYTHON" - "$ROOT" "$$" <<'PY'
import json,os,sys,time
from pathlib import Path
r=Path(sys.argv[1]).resolve()
with (r/'OFFLINE_STARTED.json').open('x') as f:
 json.dump(dict(shell_pid=int(sys.argv[2]),root=str(r),monotonic=time.monotonic(),controller_once=True,CPU_finalizer_once=True),f,sort_keys=True)
 f.write('\n');f.flush();os.fsync(f.fileno())
PY
set +e
"$PYTHON" -u -B "$ROOT/scripts/resident_controller.py" --root "$ROOT" --execute >"$ROOT/logs/controller.log" 2>&1
CONTROLLER_RC=$?
set -e
if [[ ! -f "$ROOT/SESSION_RESULT.json" ]]; then
  "$PYTHON" - "$ROOT" "$CONTROLLER_RC" <<'PY'
import json,os,sys,time
from pathlib import Path
r=Path(sys.argv[1])
with (r/'operations/OFFLINE_CONTROLLER_EXIT_WITHOUT_SESSION.json').open('x') as f:
 json.dump(dict(status='REAL_CONTROLLER_FAILURE_NO_SESSION_NOT_FABRICATED',controller_returncode=int(sys.argv[2]),monotonic=time.monotonic(),finisher_not_started=True),f,sort_keys=True)
 f.write('\n');f.flush();os.fsync(f.fileno())
PY
  exit 1
fi
"$PYTHON" - "$ROOT" "$CONTROLLER_RC" <<'PY'
import json,os,sys,time
from pathlib import Path
r=Path(sys.argv[1])
with (r/'FINALIZER_CHAIN_STARTED.json').open('x') as f:
 json.dump(dict(controller_returncode=int(sys.argv[2]),monotonic=time.monotonic(),controller_process_really_returned=True,no_retry=True),f,sort_keys=True)
 f.write('\n');f.flush();os.fsync(f.fileno())
PY
set +e
CUDA_VISIBLE_DEVICES='' "$PYTHON" -u -B "$ROOT/scripts/resident_finalize_offline.py" --root "$ROOT" >"$ROOT/logs/finalizer.log" 2>&1
FINALIZER_RC=$?
set -e
exit "$FINALIZER_RC"
