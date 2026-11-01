#!/bin/bash
set -euo pipefail
r=/data/zypan/analyses/qwen_gdn_hadamard_dual_m_aime26_formal_60_v1/repair_sampler_scope_v1
mkdir -p "$r/logs"
exec /data/ydai/miniconda3/envs/bitdecode/bin/python -B "$r/source/controller.py" --root "$r" >> "$r/logs/controller.log" 2>&1
