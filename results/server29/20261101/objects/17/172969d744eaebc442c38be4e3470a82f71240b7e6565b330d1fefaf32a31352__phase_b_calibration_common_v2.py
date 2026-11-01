"""CPU-only identity, cost, and exclusive-device gates for Phase B calibration.

This module never authorizes a long run, signals a process, or reads future labels.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from objective import digest
from run_matched_l7 import ROOT, publish
from audit_phase_a_ledger_snapshot import audit as audit_a

RUNS = ROOT / "runs_b_calibration"
CONFIG = ROOT / "PHASE_B_CALIBRATION_CONFIG.json"
FREEZE = ROOT / "PHASE_B_CALIBRATION_CODE_FREEZE.json"


def read(path):
    return json.loads(Path(path).read_text())


def checked_json(path):
    value = read(path)
    assert read(Path(path).with_suffix(".sha256.json"))["sha256"] == digest(path), "SIDECAR_MISMATCH"
    return value


def verify():
    freeze = checked_json(FREEZE)
    for name, sha in freeze["files"].items():
        assert digest(ROOT / name) == sha, ("B_CAL_CODE_CHANGED", name)
    for path, sha in freeze["dependencies"].items():
        assert digest(path) == sha, ("B_CAL_DEPENDENCY_CHANGED", path)
    cfg = read(CONFIG)
    a = read(ROOT / "PHASE_A_CONFIG.json")
    for key in ("conditions", "windows", "horizons", "history", "layers", "auxiliary_layer",
                "runtime", "seed", "loss_eps_clamp", "state_weight", "normalization",
                "loss_tensor_conditioning", "pair_tie_break", "route_rule", "gpu_uuids"):
        assert cfg[key] == a[key], ("UNCHANGED_SCIENCE_REQUIRED", key)
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json")
    assert cfg["documents"] == [x["document_id"] for x in selection["selected_documents"]]
    assert cfg["document_gpu_map"] == {x["document_id"]: x["gpu"] for x in selection["selected_documents"]}
    assert not list((ROOT / "runs").glob("*.STOP.json")), "PHASE_A_STOP_NO_RETRY"
    assert not list(RUNS.glob("*.STOP.json")), "B_CAL_STOP_NO_RETRY"
    assert not (ROOT / "PHASE_B_FAIL_CLOSED_STOP.json").exists(), "B_FAIL_CLOSED_STOP_NO_RETRY"
    weights = read(ROOT / "FRESH_MODEL_WEIGHT_AUDIT.json")
    assert weights["status"] == "PASS" and weights["files_rehashed"] == 32
    for item in weights["files"]:
        st = Path(item["path"]).stat()
        assert (st.st_size, st.st_mtime_ns) == (item["bytes"], item["mtime_ns"]), "MODEL_FRESHNESS_FAIL"
    assert not Path("/var/run/reboot-required").exists(), "MAINTENANCE_GATE_FAIL"
    assert shutil.disk_usage(ROOT).free >= 1024**3, "SHORT_STAGE_STORAGE_FAIL"
    return cfg


def audit_b(now=None):
    now = time.time() if now is None else now
    cfg = read(CONFIG)
    a = audit_a(ROOT, now)
    assert all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in a["gpu_runs"]), "PHASE_A_NOT_COMPLETE"
    rows = []
    for receipt_path in sorted(RUNS.glob("*.LAUNCH_AUTHORIZATION.json")):
        receipt = read(receipt_path)
        tag = receipt_path.name.removesuffix(".LAUNCH_AUTHORIZATION.json")
        gpu = cfg["document_gpu_map"][receipt["document"]]
        assert receipt["gpu_uuid"] == cfg["gpu_uuids"][str(gpu)]
        assert receipt["runner_sha256"] == digest(ROOT / "run_phase_b_calibration_v2.py")
        assert receipt["freeze_sha256"] == digest(FREEZE)
        row = dict(tag=tag, document=receipt["document"], mode=receipt["mode"],
                   gpu_uuid=receipt["gpu_uuid"], launch_epoch=receipt["epoch"],
                   receipt_sha256=digest(receipt_path), session=receipt["session"])
        started, canonical, stop = (RUNS / (tag + suffix) for suffix in
                                    (".STARTED.json", ".json", ".STOP.json"))
        if started.exists():
            value = read(started)
            assert value["gpu_uuid"] == row["gpu_uuid"]
            assert value["runner_sha256"] == receipt["runner_sha256"]
            row.update(pid=value["pid"], process_alive=Path(f"/proc/{value['pid']}").exists(),
                       started_sha256=digest(started))
        if stop.exists():
            value = read(stop)
            end, status = value["stopped_epoch"], "FAIL_AND_STOP_NO_RETRY"
            row["stop_sha256"] = digest(stop)
        elif canonical.exists() and canonical.with_suffix(".sha256.json").exists():
            value = checked_json(canonical)
            assert value["valid"] and value["gpu_uuid"] == row["gpu_uuid"]
            assert value["runner_sha256"] == receipt["runner_sha256"]
            assert value["config_sha256"] == digest(CONFIG)
            assert value["fp_reference"]["replay_parity"]["passed"]
            assert all(x["mechanical_gates"]["passed"] for x in value["results"])
            conditions = [x["condition"] for x in value["results"]]
            assert conditions == (cfg["conditions"] if row["mode"] == "formal" else ["IDENTITY"] * 2)
            end, status = value["completed_epoch"], "COMPLETE_MECHANICAL_PASS"
            row.update(canonical_sha256=digest(canonical),
                       sidecar_sha256=digest(canonical.with_suffix(".sha256.json")),
                       fp_reference_content_sha256=value["fp_reference"]["content_sha256"],
                       runner_accounted_gpu_hours=value["accounted_gpu_hours"])
        else:
            end = now
            status = "ACTIVE_OR_FINALIZING" if row.get("process_alive") else "LAUNCHED_STATE_REQUIRES_INSPECTION"
        row.update(status=status, gpu_hours_upper_bound=max(0, end - receipt["epoch"]) / 3600)
        rows.append(row)
    total = a["total_gpu_hours_upper_bound"] + sum(x["gpu_hours_upper_bound"] for x in rows)
    assert total <= 16, "TOTAL_BUDGET_EXCEEDED"
    return dict(schema="phase_a_plus_b_calibration_ledger_v1", epoch=now, phase_a=a,
                phase_b_calibration_runs=rows, total_gpu_hours_upper_bound=total,
                max_gpu_hours=16, loading_startup_fp_smoke_included=True,
                external_experiment_waiting_cpu_counted=False, future_labels_used=False)


def idle_device(gpu, cfg):
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    row = next([x.strip() for x in line.split(",")] for line in lines if int(line.split(",")[0]) == gpu)
    assert row[1] == cfg["gpu_uuids"][str(gpu)], "UUID_MISMATCH"
    assert row[2] == cfg["runtime"]["driver"], "DRIVER_CHANGED"
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                    "--format=csv,noheader"], text=True)
    holders = subprocess.run(["fuser", f"/dev/nvidia{gpu}"], capture_output=True, text=True)
    assert holders.returncode in (0, 1), "DEVICE_EXCLUSIVITY_UNKNOWN"
    idle = row[1] not in apps and int(row[3]) <= 128 and int(row[4]) == 0 and holders.returncode == 1
    return idle, dict(gpu_uuid=row[1], query_row=row, device_holders=holders.stdout.strip(),
                      fuser_returncode=holders.returncode, no_signal_or_process_change=True)
