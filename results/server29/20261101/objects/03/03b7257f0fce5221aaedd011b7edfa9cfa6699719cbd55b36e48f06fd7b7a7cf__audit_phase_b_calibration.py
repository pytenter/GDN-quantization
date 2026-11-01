"""Read-only hardware/history/actual aggregate ledger, append-only own receipts."""
import json
import os
import subprocess
import time
from pathlib import Path
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common import RUNS, verify, audit_b
from audit_phase_a_hardware_and_history import main as historical_and_a_guard


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    try:
        cfg = verify()
        historical_and_a_guard()
        ledger = audit_b()
        assert all(x["status"] in ("ACTIVE_OR_FINALIZING", "COMPLETE_MECHANICAL_PASS")
                   for x in ledger["phase_b_calibration_runs"]), "UNRESOLVED_B_CAL_STATE"
        live = [x for x in ledger["phase_b_calibration_runs"] if x.get("process_alive")]
        main_pids = {x["pid"] for x in live}
        active_uuids = {x["gpu_uuid"] for x in ledger["phase_b_calibration_runs"] if x["status"] == "ACTIVE_OR_FINALIZING"}
        parents = {int(line.split()[0]): int(line.split()[1]) for line in
                   subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()}
        def owned(pid):
            seen = set()
            while pid and pid not in seen:
                if pid in main_pids:
                    return True
                seen.add(pid); pid = parents.get(pid, 0)
            return False
        holders = {}
        for gpu in (3, 6):
            result = subprocess.run(["fuser", f"/dev/nvidia{gpu}"], capture_output=True, text=True)
            assert result.returncode in (0, 1), "DEVICE_HOLDERS_NOT_PROVABLE"
            holders[gpu] = [int(x) for x in result.stdout.split()]
            if cfg["gpu_uuids"][str(gpu)] in active_uuids:
                assert all(owned(pid) for pid in holders[gpu] if Path(f"/proc/{pid}").exists()), "FOREIGN_DEVICE_HOLDER_DURING_MEASUREMENT"
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
        for line in apps.splitlines():
            uuid, pid, _ = [x.strip() for x in line.split(",", 2)]
            if uuid in active_uuids:
                assert int(pid) in main_pids, "FOREIGN_COMPUTE_APP"
            if int(pid) in main_pids:
                assert next(x["gpu_uuid"] for x in live if x["pid"] == int(pid)) == uuid
        for row in live:
            cmdline = Path(f"/proc/{row['pid']}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            assert str(ROOT / "run_phase_b_calibration.py") in cmdline and row["document"] in cmdline
            assert subprocess.run(["tmux", "has-session", "-t", row["session"]], capture_output=True).returncode == 0
        for log in RUNS.glob("*.log"):
            assert "Traceback (most recent call last):" not in log.read_text(errors="replace"), "B_CAL_RUNTIME_TRACEBACK"
        path = ROOT / f"PHASE_B_CALIBRATION_GUARD_{time.time_ns()}.json"
        publish(path, dict(status="B_CALIBRATION_HARDWARE_HISTORY_TOTAL_LEDGER_PASS", ledger=ledger,
            device_holders=holders, active_pids=sorted(main_pids), no_gpu_work_or_signal=True,
            code_sha256=digest(__file__)))
        publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
        print(json.dumps(dict(status="B_CALIBRATION_HARDWARE_HISTORY_TOTAL_LEDGER_PASS", path=str(path),
            sha256=digest(path), total_gpu_hours_upper_bound=ledger["total_gpu_hours_upper_bound"], active_pids=sorted(main_pids))))
    except Exception as exc:
        latch = ROOT / "PHASE_B_FAIL_CLOSED_STOP.json"
        try:
            publish(latch, dict(status="FAIL_CLOSED_NO_RETRY", error=repr(exc), epoch=time.time(),
                audit_code_sha256=digest(__file__), no_signals_used=True))
        except FileExistsError:
            pass
        raise


if __name__ == "__main__":
    main()
