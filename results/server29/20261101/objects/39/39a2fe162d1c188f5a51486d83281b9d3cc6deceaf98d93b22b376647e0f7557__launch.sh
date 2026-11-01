#!/bin/bash
set -u
root=/data/zypan/analyses/pace48_lqunroll_dp32_throughput_v1
cd "$root" || exit 91
source_sha="$1"
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /data/ydai/miniconda3/envs/bitdecode/bin/python -B scripts/dp32_parent.py --source-sha "$source_sha" &
parent_pid=$!
wait "$parent_pid"
parent_wait=$?
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /data/ydai/miniconda3/envs/bitdecode/bin/python -B - "$root" "$source_sha" "$parent_wait" "$parent_pid" <<'PY'
import json,os,sys
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'))
import execution_common as C
C.atomic(r/'runtime/BASH_WAIT.json',dict(source_sha256=sys.argv[2],actual_Bash_wait_integer=int(sys.argv[3]),parent_pid=int(sys.argv[4]),bash=C.identity(os.getppid())))
PY
exit "$parent_wait"
