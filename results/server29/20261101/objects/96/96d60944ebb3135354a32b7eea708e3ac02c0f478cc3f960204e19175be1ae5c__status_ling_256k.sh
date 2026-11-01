#!/usr/bin/env bash
set -euo pipefail
REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
OUT=$REPO/artifacts/ling_256k_length_sensitivity_v1
echo '=== GPU 2,3 ==='
nvidia-smi -i 2,3 --query-gpu=index,uuid,memory.used,utilization.gpu,pstate --format=csv,noheader
echo '=== tmux ==='
tmux ls 2>/dev/null || true
echo '=== health ==='
curl -fsS http://127.0.0.1:31000/health 2>/dev/null || echo NOT_READY
echo
echo '=== records ==='
find "$OUT/records" -type f -name '*.json' -printf '%P\n' 2>/dev/null | sort || true
echo '=== record summary ==='
python3 - "$OUT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
for path in sorted(glob.glob(root + "/records/**/*.json", recursive=True)):
    row = json.load(open(path, encoding="utf-8"))
    print(
        os.path.relpath(path, root),
        f"tokens={row.get('generated_tokens')}",
        f"finish={row.get('finish_reason')}",
        f"v4={row.get('v4_result', {}).get('status')}",
        f"answer={row.get('v4_extracted_answer')}",
        f"correct={row.get('v4_correct')}",
        f"ok={row.get('successful_sample')}",
    )
PY
echo '=== server tail ==='
tail -n 30 "$OUT/stage0/fp_state_256k_server.log" 2>/dev/null || true
