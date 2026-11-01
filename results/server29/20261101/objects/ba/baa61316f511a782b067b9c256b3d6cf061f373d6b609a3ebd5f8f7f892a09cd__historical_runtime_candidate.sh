#!/bin/sh
# CPU identity interface only. GPU work requires a separate future authorization.
set -eu
if [ "$#" -eq 0 ]; then set -- --help; fi
if [ "$#" -ne 1 ]; then echo 'Only --help, --check or --print-env is permitted.' >&2; exit 2; fi
case "$1" in --help|--check|--print-env) ;; *) echo 'GPU execution is not available through this candidate entry.' >&2; exit 2 ;; esac
task_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x /data/ydai/miniconda3/envs/bitdecode/bin/python ]; then
  task_python=/data/ydai/miniconda3/envs/bitdecode/bin/python
else
  task_python=python3
fi
exec "$task_python" -B "$task_dir/scripts/candidate_check.py" "$1"
