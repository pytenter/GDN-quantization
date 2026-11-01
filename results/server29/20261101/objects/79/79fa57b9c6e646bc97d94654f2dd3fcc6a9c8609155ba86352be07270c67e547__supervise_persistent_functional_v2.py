#!/usr/bin/env python3
"""Wait for all frozen replay shards, then run the prepared analysis once."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
STATUS = OUT / "source_audit/supervisor_status.json"
ANALYSIS_LOG = OUT / "source_audit/analysis_v2.log"
PYTHON = "/data/ydai/miniconda3/envs/bitdecode/bin/python"


def update(status, **fields):
    record = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **fields}
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, STATUS)


def main():
    while True:
        complete = 0
        progress = []
        for shard in range(4):
            path = OUT / f"replay/shard_{shard:02d}/status.json"
            state = json.loads(path.read_text()) if path.exists() else {"status": "STARTING", "completed": 0, "assigned": 15}
            progress.append({"shard": shard, "status": state["status"], "completed": state.get("completed", 0), "assigned": state.get("assigned", 15)})
            if state["status"] == "COMPLETE":
                complete += 1
            elif state["status"] not in ("RUNNING", "STARTING"):
                update("SHARD_FAILED", progress=progress)
                return 2
            elif subprocess.run(["tmux", "has-session", "-t", f"persistent_formal{shard}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
                update("SHARD_SESSION_MISSING", progress=progress)
                return 2
        if complete == 4:
            break
        update("WAITING_FOR_REPLAY", progress=progress)
        time.sleep(60)
    if ANALYSIS_LOG.exists() or (OUT / "results.json").exists():
        update("ANALYSIS_ALREADY_PRESENT", progress=progress)
        return 2
    update("ANALYSIS_RUNNING", progress=progress)
    with ANALYSIS_LOG.open("x", encoding="utf-8") as stream:
        completed = subprocess.run([PYTHON, str(OUT / "analyze_persistent_functional_v2.py")], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    update("COMPLETE" if completed.returncode == 0 else "ANALYSIS_FAILED", returncode=completed.returncode, progress=progress)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
