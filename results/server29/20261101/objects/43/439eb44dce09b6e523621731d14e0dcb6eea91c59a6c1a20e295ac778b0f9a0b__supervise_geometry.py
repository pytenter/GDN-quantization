#!/usr/bin/env python3
"""Wait for four frozen geometry shards, then run the frozen analyzer once."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1")
STATUS = OUT / "analysis/supervisor_status.json"
LOG = OUT / "analysis/analysis.log"


def save(status, progress, **more):
    value = {"status": status, "progress": progress,
             "disk_free_gib": shutil.disk_usage(OUT).free / 1024 ** 3,
             "updated_at": datetime.now(timezone.utc).isoformat(), **more}
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATUS)


def main():
    while True:
        progress = []
        for index in range(4):
            path = OUT / "analysis" / f"shard_{index:02d}" / "status.json"
            item = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"shard": index, "status": "STARTING", "assigned": 15, "completed": 0}
            progress.append({key: item[key] for key in ("shard", "status", "assigned", "completed")})
            if item["status"] != "COMPLETE":
                alive = subprocess.run(["tmux", "has-session", "-t", f"geometry_formal{index}"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                if not alive:
                    save("REPLAY_FAILED", progress, failed_shard=index)
                    return 1
        if all(item["status"] == "COMPLETE" and item["assigned"] == item["completed"] == 15 for item in progress):
            break
        save("WAITING_FOR_REPLAY", progress)
        time.sleep(60)
    save("ANALYZING", progress)
    with LOG.open("w", encoding="utf-8") as stream:
        outcome = subprocess.run([sys.executable, str(OUT / "analyze_geometry.py")],
                                 cwd=OUT, stdout=stream, stderr=subprocess.STDOUT)
    if outcome.returncode or not (OUT / "run_manifest.json").exists():
        save("ANALYSIS_FAILED", progress, returncode=outcome.returncode)
        return 1
    save("COMPLETE", progress, returncode=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
