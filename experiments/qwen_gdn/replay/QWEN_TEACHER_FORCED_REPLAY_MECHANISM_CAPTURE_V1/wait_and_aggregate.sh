#!/usr/bin/env bash
set -euo pipefail

cd /data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen
out=experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1

while true; do
  state=$(/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 - <<'PY'
import json
from pathlib import Path

root = Path("experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1/formal")
states = []
for i in range(4):
    path = root / f"shard_{i:02d}" / "status.json"
    states.append(json.loads(path.read_text()).get("status") if path.exists() else "MISSING")
print("FAILED" if "FAILED" in states else "COMPLETE" if states == ["COMPLETE"] * 4 else "RUNNING")
PY
)
  printf '%s capture_status=%s\n' "$(date -Is)" "$state"
  if [[ "$state" == COMPLETE ]]; then
    break
  fi
  if [[ "$state" == FAILED ]]; then
    exit 2
  fi
  sleep 60
done

/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 -u "$out/aggregate.py"
