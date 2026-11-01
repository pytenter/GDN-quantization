#!/usr/bin/env bash
set -u
HERE=$(cd -- "$(dirname -- "$0")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
PY=/data/ydai/miniconda3/envs/bitdecode/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
"$PY" -u "$HERE/scripts/persistent_v2_controller.py"
worker_rc=$?
"$PY" -u "$HERE/scripts/persistent_v2_finalize.py"
closure_rc=$?
if [ "$closure_rc" -eq 0 ]; then
  git add -f "$HERE"/analysis "$HERE"/provenance "$HERE"/reports "$HERE"/manifests "$HERE"/logs/*_*.log "$HERE"/logs/*.exitcode
  git commit -m "Close prospective persistent-worker V2 readiness; stop without AIME"
  commit_rc=$?
  if [ "$commit_rc" -ne 0 ]; then exit "$commit_rc"; fi
  "$PY" -c 'import json,subprocess,sys;from pathlib import Path;p=Path(sys.argv[1]);x=dict(closure_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),git_status=subprocess.check_output(["git","status","--porcelain"],text=True),push_performed=False);p.write_text(json.dumps(x,indent=2)+"\n")' "$HERE/logs/v2_closure_commit.json"
fi
printf '%s\n' "$closure_rc" > "$HERE/logs/offline.exitcode"
exit "$closure_rc"
