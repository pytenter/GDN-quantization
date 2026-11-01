#!/usr/bin/env python3
"""Precisely snapshot and terminate only the superseded V1 process tree."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


V1_NAME = "QWEN_GDN_HADAMARD_VS_LEARNED_ROTATION_MECHANISM_CLOSURE_V1"
OLD_REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
V1 = OLD_REPO / "experiments/qwen_gdn" / V1_NAME
V2 = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/qwen_gdn/QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2")
OUT = V2 / "provenance/V1_FINAL_PRE_TERMINATION_SNAPSHOT.json"
WORKER_MARK = str(V1 / "scripts/mechanism_capture.py")
WRAPPER_MARK = str(V1 / "scripts/run_mechanism_capture.sh")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def proc(pid: int):
    base = Path(f"/proc/{pid}")
    try:
        stat = (base / "stat").read_text().split()
        cmd = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        cwd = os.readlink(base / "cwd")
        cgroup = (base / "cgroup").read_text().splitlines()
        return {
            "pid": pid, "ppid": int(stat[3]), "pgid": int(stat[4]), "sid": int(stat[5]),
            "stat": stat[2], "starttime_ticks": int(stat[21]), "cmdline": cmd,
            "cwd": cwd, "cgroup": cgroup,
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def all_procs():
    rows = {}
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            row = proc(int(item.name))
            if row:
                rows[row["pid"]] = row
    return rows


def same(row) -> bool:
    now = proc(row["pid"])
    return bool(now and now["starttime_ticks"] == row["starttime_ticks"] and now["cmdline"] == row["cmdline"])


def run(args):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout


def read_statuses():
    controlled = {}
    recurrent = {}
    for shard in range(4):
        for mode, target in (("controlled_all", controlled), ("recurrent_baseline", recurrent)):
            path = V1 / f"raw/formal/{mode}_shard{shard:02d}/status.json"
            if path.is_file():
                try:
                    target[str(shard)] = json.loads(path.read_text())
                except Exception as exc:
                    target[str(shard)] = {"status": "UNREADABLE", "error": repr(exc)}
            else:
                target[str(shard)] = {"status": "NOT_STARTED", "completed": 0, "total": 15}
    return controlled, recurrent


def artifact_summary():
    files = [path for path in V1.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "logs_count": sum(1 for path in files if "logs" in path.relative_to(V1).parts),
        "raw_count": sum(1 for path in files if "raw" in path.relative_to(V1).parts),
    }


def signal_exact(rows, sig, action_log):
    for row in rows:
        if not same(row):
            action_log.append({"utc": utc(), "pid": row["pid"], "signal": sig.name, "result": "NOT_SENT_IDENTITY_CHANGED_OR_EXITED"})
            continue
        os.kill(row["pid"], sig)
        action_log.append({"utc": utc(), "pid": row["pid"], "signal": sig.name, "result": "SENT_TO_EXACT_PID"})


def wait_exact(rows, seconds):
    deadline = time.monotonic() + seconds
    alive = [row for row in rows if same(row)]
    while alive and time.monotonic() < deadline:
        time.sleep(0.5)
        alive = [row for row in rows if same(row)]
    return alive


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    processes = all_procs()
    workers = [row for row in processes.values() if WORKER_MARK in row["cmdline"]]
    wrappers = [row for row in processes.values() if row["cmdline"].startswith("bash " + WRAPPER_MARK)]
    tmux_launcher = [row for row in processes.values() if "tmux new-session" in row["cmdline"] and WRAPPER_MARK in row["cmdline"]]
    if len(workers) != 4 or len(wrappers) != 4:
        raise SystemExit(f"AMBIGUOUS PROCESS SET: workers={len(workers)} wrappers={len(wrappers)}")
    if any("--mode recurrent --condition-set baseline" not in row["cmdline"] for row in workers):
        raise SystemExit("AMBIGUOUS/DOWNSTREAM WORKER PRESENT; no signal sent")
    if {row["ppid"] for row in workers} != {row["pid"] for row in wrappers}:
        raise SystemExit("worker/wrapper parent set mismatch; no signal sent")

    descendants = set(row["pid"] for row in wrappers)
    changed = True
    while changed:
        before = len(descendants)
        descendants.update(row["pid"] for row in processes.values() if row["ppid"] in descendants)
        changed = len(descendants) != before
    known_main = {row["pid"] for row in workers + wrappers}
    other_children = [processes[pid] for pid in sorted(descendants) if pid not in known_main]

    controlled, recurrent = read_statuses()
    controlled_completed = sum(int(row.get("completed", 0) or 0) for row in controlled.values())
    recurrent_completed = sum(int(row.get("completed", 0) or 0) for row in recurrent.values())
    snapshot = {
        "task": "QWEN_V1_FINAL_TERMINATION_AND_V2_STAGE_A_V1",
        "snapshot_utc": utc(),
        "V1_TERMINATION_REASON": "SCIENTIFICALLY_SUPERSEDED_BY_FORMAL_EQUIVALENT_V2",
        "V1_SCIENTIFIC_STATUS": "INCOMPLETE_DESCRIPTIVE_DIAGNOSTIC_ONLY",
        "V1_CONTROLLED_COMPLETED": controlled_completed,
        "V1_CONTROLLED_COMPLETENESS_GATE": "PASS" if controlled_completed == 60 else "FAIL",
        "V1_RECURRENT_BASELINE_COMPLETED": recurrent_completed,
        "V1_RECURRENT_COMPLETENESS_GATE": "PASS" if recurrent_completed == 60 else "PARTIAL",
        "V1_CURRENT_SAMPLE_BY_WORKER": {key: row.get("last") for key, row in recurrent.items()},
        "V1_EXISTING_ARTIFACT_COUNT": artifact_summary(),
        "controlled_statuses": controlled,
        "recurrent_baseline_statuses": recurrent,
        "V1_PYTHON_WORKER_PIDS": sorted(row["pid"] for row in workers),
        "V1_WRAPPER_PIDS": sorted(row["pid"] for row in wrappers),
        "V1_OTHER_CHILD_PIDS": sorted(row["pid"] for row in other_children),
        "tmux_launcher_processes": tmux_launcher,
        "V1_PROCESS_SNAPSHOT_BEFORE_STOP": sorted(workers + wrappers + other_children + tmux_launcher, key=lambda x: x["pid"]),
        "gpu_before": run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]).splitlines(),
        "termination_actions": [],
        "V1_DATA_ALLOWED_USES": ["engineering diagnostics", "descriptive sanity checks", "partial trajectory observations"],
        "V1_DATA_FORBIDDEN_USES": ["primary mechanism evidence", "formal C5/C6 replay evidence", "cross-rotation causal evidence", "new-loss selection gate"],
    }
    OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Hold only the already-verified wrapper PIDs during worker termination to
    # close the short transition race. This is not relied on as persistent
    # containment; wrappers are killed immediately after workers exit.
    signal_exact(wrappers, signal.SIGSTOP, snapshot["termination_actions"])
    signal_exact(workers, signal.SIGTERM, snapshot["termination_actions"])
    survivors = wait_exact(workers, 15)
    snapshot["TERM_NOT_SUFFICIENT"] = [row["pid"] for row in survivors]
    signal_exact(survivors, signal.SIGKILL, snapshot["termination_actions"])
    survivors = wait_exact(survivors, 5)
    if survivors:
        snapshot["termination_error"] = f"worker survivors after KILL: {[row['pid'] for row in survivors]}"
        OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(snapshot["termination_error"])

    # Any tee/helper descendant can now be terminated by its exact identity.
    signal_exact(other_children, signal.SIGTERM, snapshot["termination_actions"])
    remaining_children = wait_exact(other_children, 3)
    signal_exact(remaining_children, signal.SIGKILL, snapshot["termination_actions"])
    wait_exact(remaining_children, 2)

    # A stopped wrapper cannot process TERM, so use exact-PID KILL without CONT.
    signal_exact(wrappers, signal.SIGKILL, snapshot["termination_actions"])
    wrapper_survivors = wait_exact(wrappers, 5)
    if wrapper_survivors:
        snapshot["termination_error"] = f"wrapper survivors after KILL: {[row['pid'] for row in wrapper_survivors]}"
        OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(snapshot["termination_error"])

    sessions_before = run(["tmux", "list-sessions", "-F", "#S"]).splitlines()
    killed_sessions = []
    for name in ("qwen_mech_0", "qwen_mech_1", "qwen_mech_2", "qwen_mech_3"):
        if name in sessions_before:
            subprocess.run(["tmux", "kill-session", "-t", name], check=False)
            killed_sessions.append(name)

    final = all_procs()
    # A tmux server may retain its original creation argv even after these
    # sessions disappear; it is not a V1 scientific process and must not be
    # killed. Only actual worker/wrapper/helper command lines count here.
    remaining_v1 = [row for row in final.values() if (
        WORKER_MARK in row["cmdline"]
        or row["cmdline"].startswith("bash " + WRAPPER_MARK)
        or (row["pid"] in {child["pid"] for child in other_children} and same(row))
    )]
    gpu_after = run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]).splitlines()
    old_worker_pids = {row["pid"] for row in workers}
    snapshot.update({
        "termination_completed_utc": utc(),
        "tmux_sessions_removed": killed_sessions,
        "V1_PROCESSES_REMAINING": remaining_v1,
        "V1_PYTHON_WORKERS_REMAINING": sum(WORKER_MARK in row["cmdline"] for row in remaining_v1),
        "V1_WRAPPERS_REMAINING": sum(WRAPPER_MARK in row["cmdline"] and row["cmdline"].startswith("bash ") for row in remaining_v1),
        "V1_DOWNSTREAM_PROCESS_REMAINING": sum("--condition-set baseline" not in row["cmdline"] and "mechanism_capture.py" in row["cmdline"] for row in remaining_v1),
        "UNEXPECTED_V1_DOWNSTREAM_LAUNCH": any("mechanism_capture.py" in row["cmdline"] and "--condition-set baseline" not in row["cmdline"] for row in remaining_v1),
        "NO_NEW_V1_DOWNSTREAM_STAGE_STARTED": not any("mechanism_capture.py" in row["cmdline"] and "--condition-set baseline" not in row["cmdline"] for row in remaining_v1),
        "gpu_after": gpu_after,
        "V1_GPU_RELEASE_GATE": "PASS" if not any(line.split(",", 1)[0].strip().isdigit() and int(line.split(",", 1)[0]) in old_worker_pids for line in gpu_after) else "FAIL",
        "V1_ALL_PROCESSES_STOPPED": not remaining_v1,
        "artifacts_preserved": True,
    })
    OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: snapshot[key] for key in (
        "V1_CONTROLLED_COMPLETED", "V1_RECURRENT_BASELINE_COMPLETED", "V1_PYTHON_WORKERS_REMAINING",
        "V1_WRAPPERS_REMAINING", "V1_DOWNSTREAM_PROCESS_REMAINING", "V1_GPU_RELEASE_GATE",
        "V1_ALL_PROCESSES_STOPPED", "NO_NEW_V1_DOWNSTREAM_STAGE_STARTED", "TERM_NOT_SUFFICIENT")}, indent=2))
    if remaining_v1 or snapshot["V1_GPU_RELEASE_GATE"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
