#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1
PY=/root/miniconda3/envs/qwen-a800/bin/python
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/logs/environment_finish.exitcode"' EXIT
python "$ROOT/scripts/a800_prefetch_wheels.py" --extra-only
"$PY" -m pip install --no-index --find-links /root/qwen_a800_wheelhouse --find-links /root/autodl-tmp/wheelhouse -r "$ROOT/provenance/resolved_install_requirements_extras.txt"
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/provenance/pip_freeze.txt"
"$PY" -m pip inspect > "$ROOT/provenance/pip_inspect.json"
printf 'ENVIRONMENT_INSTALL_COMPLETE\n'
