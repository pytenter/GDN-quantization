"""Read-only heartbeat guard plus exclusive own audit receipt; no GPU compute."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
from objective import digest
from run_matched_l7 import publish
from audit_phase_a_ledger_snapshot import audit


def command(args):
    return subprocess.check_output(args, text=True)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "PHASE_A_CONFIG.json").read_text())
    ledger = audit(root, time.time())
    assert not list((root / "runs").glob("*.STOP.json")), "RETAINED_STOP_FAIL_CLOSED"
    assert all(x["status"] in ["ACTIVE_OR_FINALIZING", "COMPLETE_MECHANICAL_PASS"]
               for x in ledger["gpu_runs"]), "UNRESOLVED_LAUNCHED_STATE"
    baseline = json.loads((root / "HISTORICAL_IMMUTABILITY_BASELINE.json").read_text())
    checked = 0
    for row in baseline["files"]:
        for kind in ("canonical", "sidecar"):
            entry = row[kind]
            assert Path(entry["path"]).stat().st_size == entry["bytes"]
            assert digest(entry["path"]) == entry["sha256"], "HISTORICAL_BYTE_IDENTITY_FAIL"
            checked += 1
    for path, sha in baseline["attempt04_attempt05_permanent_stop"].items():
        assert digest(path) == sha, "PERMANENT_HISTORICAL_STOP_CHANGED"
    model = json.loads((root / "FRESH_MODEL_WEIGHT_AUDIT.json").read_text())
    for entry in model["files"]:
        st = Path(entry["path"]).stat()
        assert st.st_size == entry["bytes"] and st.st_mtime_ns == entry["mtime_ns"]
    assert not Path("/var/run/reboot-required").exists(), "MAINTENANCE_GATE_FAIL"
    assert shutil.disk_usage(root).free >= 1024**3, "SHORT_STAGE_STORAGE_FAIL"
    gpus = command(["nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"])
    gpu_rows = {int(line.split(",")[0]): [x.strip() for x in line.split(",")]
                for line in gpus.splitlines()}
    applications = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                            "--format=csv,noheader"])
    active = [x for x in ledger["gpu_runs"] if x["status"] == "ACTIVE_OR_FINALIZING"]
    live_own = [x for x in ledger["gpu_runs"] if x.get("process_alive")]
    main_pids = {int(x["pid"]) for x in live_own}
    active_uuids = {x["gpu_uuid"] for x in active}
    holders = {}
    for gpu in [3, 6]:
        assert gpu_rows[gpu][1] == cfg["gpu_uuids"][str(gpu)], "GPU_UUID_CHANGED"
        assert gpu_rows[gpu][2] == cfg["runtime"]["driver"], "FROZEN_DRIVER_CHANGED"
        result = subprocess.run(["fuser", f"/dev/nvidia{gpu}"], capture_output=True, text=True)
        assert result.returncode in [0, 1], "DEVICE_HOLDERS_NOT_PROVABLE"
        holders[gpu] = [int(x) for x in result.stdout.split()]
    parent_rows = command(["ps", "-eo", "pid=,ppid="])
    parents = {int(line.split()[0]): int(line.split()[1]) for line in parent_rows.splitlines()}
    def owned(pid):
        seen = set()
        while pid and pid not in seen:
            if pid in main_pids:
                return True
            seen.add(pid)
            pid = parents.get(pid, 0)
        return False
    for gpu, pids in holders.items():
        unknown = [p for p in pids if not owned(p) and Path(f"/proc/{p}").exists()]
        if cfg["gpu_uuids"][str(gpu)] in active_uuids:
            assert not unknown, ("FOREIGN_DEVICE_HOLDER_DURING_MEASUREMENT", gpu, unknown)
    for line in applications.splitlines():
        uuid, pid, process = [x.strip() for x in line.split(",", 2)]
        if uuid in active_uuids:
            assert int(pid) in main_pids, "FOREIGN_COMPUTE_APP_ON_AUTHORIZED_GPU"
        if int(pid) in main_pids:
            expected = next(x["gpu_uuid"] for x in live_own if x["pid"] == int(pid))
            assert uuid == expected, "OWN_PROCESS_GPU_MAP_CHANGED"
    for row in live_own:
        cmdline = Path(f"/proc/{row['pid']}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert str(root / "run_matched_l7.py") in cmdline and row["document"] in cmdline
    for log in (root / "runs").glob("formal_*.log"):
        assert "Traceback (most recent call last):" not in log.read_text(errors="replace"), "FORMAL_RUNTIME_TRACEBACK"
    receipt = dict(status="READ_ONLY_HARDWARE_HISTORY_AND_LEDGER_AUDIT_PASS", epoch=time.time(),
        ledger=ledger, device_holders=holders, authorized_main_pids=sorted(main_pids),
        historical_files_rehashed=checked, historical_permanent_stops_rehashed=2,
        frozen_model_metadata_fresh=True, active_device_holders_children_of_own_jobs_only=True,
        storage_free_bytes=shutil.disk_usage(root).free, no_reboot_required=True,
        own_audit_code_sha256=digest(__file__), no_gpu_work_no_signals_no_outcome_access=True)
    path = root / f"HEARTBEAT_GUARD_SNAPSHOT_{time.time_ns()}.json"
    publish(path, receipt)
    publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(json.dumps(dict(status=receipt["status"], path=str(path), sha256=digest(path),
        historical_files_rehashed=checked, active_pids=sorted(main_pids),
        total_gpu_hours_upper_bound=ledger["total_gpu_hours_upper_bound"])))


if __name__ == "__main__":
    main()
