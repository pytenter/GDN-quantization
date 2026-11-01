#!/usr/bin/env python3
"""Wait for four frozen replay shards, then run the prewritten analysis once."""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1")
PYTHON = "/data/ydai/miniconda3/envs/bitdecode/bin/python"
ANALYZER = OUT / "analyze_stability_v2.py"
STATUS = OUT / "source_audit/supervisor_status.json"
LOG = OUT / "source_audit/analysis_v2.log"


def save(value):
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, STATUS)


def session_alive(name):
    return subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    if (OUT / "results.json").exists():
        raise RuntimeError("formal result already exists; refusing repeat analysis")
    while True:
        progress = []
        for index in range(4):
            path = OUT / f"trajectory_metrics/shard_{index:02d}/status.json"
            row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status":"STARTING","completed":0,"assigned":15}
            progress.append({"shard":index,"status":row["status"],"completed":row["completed"],"assigned":row["assigned"]})
            if row["status"] != "COMPLETE" and not session_alive(f"stability_formal{index}"):
                save({"status":"REPLAY_FAILED","progress":progress,"reason":f"shard {index} session ended before COMPLETE"})
                raise RuntimeError(f"shard {index} stopped before COMPLETE")
        free = shutil.disk_usage(OUT).free / 1024**3
        if all(row["status"] == "COMPLETE" and row["completed"] == row["assigned"] == 15 for row in progress):
            save({"status":"ANALYSIS_RUNNING","progress":progress,"disk_free_gib":free})
            with LOG.open("x", encoding="utf-8") as log:
                proc = subprocess.run([PYTHON,str(ANALYZER)], cwd=OUT, stdout=log, stderr=subprocess.STDOUT)
            save({"status":"COMPLETE" if proc.returncode == 0 else "ANALYSIS_FAILED","returncode":proc.returncode,"progress":progress,"disk_free_gib":shutil.disk_usage(OUT).free/1024**3})
            if proc.returncode:
                raise RuntimeError(f"analysis failed with return code {proc.returncode}")
            return
        save({"status":"WAITING_FOR_REPLAY","progress":progress,"disk_free_gib":free})
        time.sleep(60)


if __name__ == "__main__":
    main()
