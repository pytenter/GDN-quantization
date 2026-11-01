"""One human-authorized H-only prefix measurement; no scientific scoring or retries."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True

TASK = "QWEN_GDN_H_DUALM_AIME26_THROUGHPUT_20261017_V1"
PYTHON = "/data/ydai/miniconda3/envs/bitdecode/bin/python"
CONTROL_SHA = "e7edb91fee5b3f32f2f8721ce1546e17be820a7976676e4f1a02045c8ae1a21f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    auth = read(args.authorization)
    assert auth["status"] == "PASS" and auth["task_id"] == TASK and auth["allowed_mode"] == "throughput"
    assert sha(root / "PLAN.json") == auth["plan_sha256"]
    assert sha(root / "SOURCE_MANIFEST.json") == auth["SOURCE_MANIFEST_sha256"]
    plan = read(root / "PLAN.json")
    assert plan["output_cap"] == 2048 and plan["warmup_tokens"] == 128
    assert [u["gpu"] for u in plan["units"]] == [0, 1, 2, 3]
    for row in read(root / "SOURCE_MANIFEST.json")["execution_sources"]:
        p = root / row["path"]
        assert p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], str(p)
    old = Path(plan["old_root"])
    source = old / "source/task_coordinator.py"
    assert sha(source) == CONTROL_SHA
    spec = importlib.util.spec_from_file_location("qualified_control_helpers", source)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, control.signal_stop)
    old_ledger = read(old / "GPU_RESOURCE_LEDGER.json")
    assert old_ledger["status"] == "FAIL_CLOSED_STOP" and old_ledger["all_GPUs_released"]
    assert (old / "USER_REQUESTED_STOP_AND_THROUGHPUT_SCOPE_20261017.json").exists()
    assert "Coordinator stop signal 15" in read(old / "FIRST_FAILURE.json")["traceback"]
    assert not (root / "CONTROL_STARTED.json").exists() and not (root / "FIRST_FAILURE.json").exists()
    requested = set(plan["gpu_uuid_map"].values())
    assert not any(a["UUID"] in requested for a in control.nvml()), "No preemption"
    physical = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    for line in physical.splitlines():
        index, uuid, name = [x.strip() for x in line.split(",")]
        assert plan["gpu_uuid_map"][index] == uuid and "3090" in name
    control.save(root / "CONTROL_STARTED.json", dict(task_id=TASK, pid=os.getpid(), identity=control.identity(os.getpid()),
                 server_unix=time.time(), authorization_sha256=sha(args.authorization)), new=True)
    ledger = dict(task_id=TASK, status="RUNNING_H_ONLY_THROUGHPUT", dispatches=[], all_GPUs_released=False,
                  prior_cancelled_run_GPU_hours_including_parent_failure=old_ledger["cumulative_GPU_hours"],
                  accounting="Remote monotonic dispatch through real process exit and NVML release; no reset of prior cost.")
    jobs = []
    failure = None
    try:
        for gpu in range(4):
            if control.STOP_SIGNAL is not None:
                raise RuntimeError("Operator stop signal " + str(control.STOP_SIGNAL))
            logfile = root / ("gpu" + str(gpu) + ".log")
            handle = logfile.open("xb")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            started = time.monotonic()
            proc = subprocess.Popen([PYTHON, str(root / "source/benchmark_worker.py"), "--root", str(root), "--gpu", str(gpu),
                                     "--authorization", str(args.authorization)], env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(gpu=gpu, gpu_uuid=plan["gpu_uuid_map"][str(gpu)], pid=proc.pid, identity=None,
                          dispatch_unix=time.time(), dispatch_monotonic=started, occupied_seconds=0., returncode=None,
                          process_exited=False, NVML_released=False)
            job = dict(proc=proc, record=record, logfile=logfile, handle=handle, last_size=0, last_progress=started)
            jobs.append(job)
            until = time.monotonic() + 2
            while record["identity"] is None and proc.poll() is None and time.monotonic() < until:
                record["identity"] = control.identity(proc.pid)
                if record["identity"] is None:
                    time.sleep(.01)
            assert record["identity"] and str(root / "source/benchmark_worker.py") in record["identity"]["cmdline"]
        while True:
            if control.STOP_SIGNAL is not None:
                raise RuntimeError("Operator stop signal " + str(control.STOP_SIGNAL))
            now = time.monotonic()
            apps = control.nvml()
            own = {j["proc"].pid for j in jobs}
            assert not [a for a in apps if a["UUID"] in requested and a["pid"] not in own], "Foreign GPU process: stop own only"
            for job in jobs:
                rec = job["record"]
                proc = job["proc"]
                code = proc.poll()
                size = job["logfile"].stat().st_size
                if size != job["last_size"]:
                    job.update(last_size=size, last_progress=now)
                rec.update(process_exited=code is not None, returncode=code, NVML_released=not any(a["pid"] == proc.pid for a in apps))
                if "release_monotonic" not in rec:
                    rec["occupied_seconds"] = now - rec["dispatch_monotonic"]
                    if rec["process_exited"] and rec["NVML_released"]:
                        rec["release_monotonic"] = now
                if code is not None:
                    assert code == 0, "Benchmark worker failed; no retry"
                    result = read(root / "outputs" / ("worker" + str(rec["gpu"])) / "RESULT.json")
                    assert result["status"] == "PASS" and result["method"] == "H_DUALM"
                else:
                    assert now - job["last_progress"] < 1800, "No progress within watchdog"
            ledger.update(dispatches=[j["record"] for j in jobs], updated_unix=time.time())
            ledger["this_benchmark_GPU_hours"] = sum(r["occupied_seconds"] for r in ledger["dispatches"]) / 3600
            ledger["combined_lineage_GPU_hours"] = ledger["this_benchmark_GPU_hours"] + ledger["prior_cancelled_run_GPU_hours_including_parent_failure"]
            control.save(root / "GPU_RESOURCE_LEDGER.json", ledger)
            if all("release_monotonic" in j["record"] for j in jobs):
                break
            time.sleep(2)
    except BaseException:
        failure = dict(status="FAIL_CLOSED_STOP", task_id=TASK, server_unix=time.time(), traceback=traceback.format_exc())
        for job in jobs:
            current = control.identity(job["proc"].pid)
            expected = job["record"]["identity"]
            if job["proc"].poll() is None and current and expected and current["start_tick"] == expected["start_tick"] and str(root / "source/benchmark_worker.py") in current["cmdline"]:
                os.killpg(job["proc"].pid, signal.SIGTERM)
        until = time.monotonic() + 15
        while any(j["proc"].poll() is None for j in jobs) and time.monotonic() < until:
            time.sleep(.5)
        for job in jobs:
            current = control.identity(job["proc"].pid)
            expected = job["record"]["identity"]
            if job["proc"].poll() is None and current and expected and current["start_tick"] == expected["start_tick"] and str(root / "source/benchmark_worker.py") in current["cmdline"]:
                os.killpg(job["proc"].pid, signal.SIGKILL)
        control.save(root / "FIRST_FAILURE.json", failure, new=True)
    finally:
        deadline = time.monotonic() + 60
        while True:
            apps, error = control.nvml_safe()
            own = {j["proc"].pid for j in jobs}
            if not error and all(j["proc"].poll() is not None for j in jobs) and not any(a["pid"] in own for a in apps):
                break
            if time.monotonic() > deadline:
                break
            time.sleep(1)
        now = time.monotonic()
        for job in jobs:
            rec = job["record"]
            rec.update(process_exited=job["proc"].poll() is not None, returncode=job["proc"].poll(),
                       NVML_released=not error and not any(a["pid"] == job["proc"].pid for a in apps))
            if "release_monotonic" not in rec:
                rec["occupied_seconds"] = now - rec["dispatch_monotonic"]
                if rec["process_exited"] and rec["NVML_released"]:
                    rec["release_monotonic"] = now
            job["handle"].close()
        released = all(j["record"]["process_exited"] and j["record"]["NVML_released"] for j in jobs)
        ledger.update(status="FAIL_CLOSED_STOP" if failure or not released else "COMPLETE_H_ONLY_THROUGHPUT",
                      dispatches=[j["record"] for j in jobs], all_GPUs_released=released,
                      NVML_status="UNKNOWN" if error else "VERIFIED", NVML_error=error, updated_unix=time.time())
        ledger["this_benchmark_GPU_hours"] = sum(r["occupied_seconds"] for r in ledger["dispatches"]) / 3600
        ledger["combined_lineage_GPU_hours"] = ledger["this_benchmark_GPU_hours"] + ledger["prior_cancelled_run_GPU_hours_including_parent_failure"]
        control.save(root / "GPU_RESOURCE_LEDGER.json", ledger)
        control.save(root / "COORDINATOR_RESULT.json", dict(status="PASS" if not failure and released else "FAIL_CLOSED_STOP",
                     all_GPUs_released=released, ledger_sha256=sha(root / "GPU_RESOURCE_LEDGER.json")), new=True)
    if failure or not released:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
