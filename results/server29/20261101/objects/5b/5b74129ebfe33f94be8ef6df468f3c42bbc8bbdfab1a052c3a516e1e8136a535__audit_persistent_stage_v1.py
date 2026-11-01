"""CPU-only monitoring with append-only whole-task ledger and fail-closed latch."""
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from objective import digest
from run_matched_l7 import ROOT, publish
from persistent_stage_common_v1 import RUNS, ledger, verify, latch


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    try:
        cfg, selection, freeze = verify()
        account = ledger()
        assert all(x["status"] in ("COMPLETE_MECHANICAL_PASS", "ACTIVE_OR_FINALIZING") for x in account["persistent_runs"])
        physical = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu",
                                           "--format=csv,noheader,nounits"], text=True)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
        hardware = {}
        for gpu in (3, 6):
            row = next([x.strip() for x in line.split(",")] for line in physical.splitlines() if int(line.split(",")[0]) == gpu)
            assert row[1] == cfg["gpu_uuids"][str(gpu)] and row[2] == cfg["runtime"]["driver"]
            active = [x for x in account["persistent_runs"] if x["gpu"] == gpu and x["status"] == "ACTIVE_OR_FINALIZING"]
            assert len(active) <= 1
            holders = subprocess.run(["fuser", f"/dev/nvidia{gpu}"], capture_output=True, text=True)
            assert holders.returncode in (0, 1), "DEVICE_HOLDERS_UNKNOWN"
            assigned = [x for x in apps.splitlines() if row[1] in x]
            if active:
                own = active[0]; pid = own.get("pid")
                assert subprocess.run(["tmux", "has-session", "-t", own["session"]], capture_output=True).returncode == 0
                if pid is not None:
                    assert Path(f"/proc/{pid}").exists()
                    assert all(int(x.split(",")[1]) == pid for x in assigned), "EXTERNAL_GPU_CO_OCCUPANCY"
                    assert all(int(x) == pid for x in holders.stdout.split()), "EXTERNAL_DEVICE_HOLDER"
                text = (RUNS / (own["tag"] + ".log")).read_text() if (RUNS / (own["tag"] + ".log")).exists() else ""
                assert not any(x in text for x in ("Traceback (most recent call last)", "CUDA out of memory", "Xid", "NONFINITE")), "RUNTIME_LOG_ANOMALY"
            hardware[str(gpu)] = dict(physical_row=row, compute_apps=assigned,
                device_holders=holders.stdout.strip(), active_own_runs=active, no_process_signals=True)
        stamp = time.time_ns(); path = ROOT / f"PERSISTENT_GUARD_{stamp}.json"
        value = dict(status="PASS", scope="mechanical_guard_not_scientific_confirmation", epoch=time.time(),
                     entire_task_ledger=account, hardware=hardware, no_external_kl_read=True, no_process_signals=True)
        publish(path, value); publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
        print(json.dumps(dict(status="PASS", total_gpu_hours_upper_bound=account["total_gpu_hours_upper_bound"],
            persistent_runs=[{k:x[k] for k in ("tag", "status", "gpu_hours_upper_bound")} for x in account["persistent_runs"]],
            guard_path=str(path), sha256=digest(path))))
    except Exception as exc:
        path = ROOT / f"PERSISTENT_GUARD_EXCEPTION_{time.time_ns()}.json"
        publish(path, dict(status="FAIL_CLOSED_NO_RETRY", error=repr(exc), traceback=traceback.format_exc(),
                           epoch=time.time(), no_process_signals=True))
        latch(repr(exc), "monitor", evidence=str(path))
        raise


if __name__ == "__main__":
    main()
