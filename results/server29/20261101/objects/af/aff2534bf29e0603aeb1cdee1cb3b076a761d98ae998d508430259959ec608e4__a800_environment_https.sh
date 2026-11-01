#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1
PY=/root/miniconda3/envs/qwen-a800/bin/python
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/logs/environment_install_https.exitcode"' EXIT
test "$("$PY" -c 'import platform; print(platform.python_version())')" = 3.10.18
printf '%s  %s\n' 92af92c569de5da937dd1afb45ecfdd598ec1254cf2e49e3d698cb24d71aae14 /root/autodl-tmp/wheelhouse/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl | sha256sum --check
"$PY" -m pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple -r "$ROOT/scripts/a800_requirements.txt"
"$PY" -m pip install --no-cache-dir --no-deps /root/autodl-tmp/wheelhouse/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl
"$PY" -m pip install --no-deps -e /root/autodl-tmp/canonical_source/transformers-qwen35
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/provenance/pip_freeze.txt"
"$PY" -m pip inspect > "$ROOT/provenance/pip_inspect.json"
printf 'ENVIRONMENT_INSTALL_COMPLETE\n'
