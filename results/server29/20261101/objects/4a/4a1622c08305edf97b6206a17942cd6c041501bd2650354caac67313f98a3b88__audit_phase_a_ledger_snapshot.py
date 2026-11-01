"""CPU-only, append-only accounting of this experiment's fixed Phase A jobs.

Does not launch, signal, retry, score, or replace any frozen evidence/ledger.
Snapshots include loading/startup time conservatively from launch authorization.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
from objective import digest
from run_matched_l7 import publish


def audit(root, now):
    cfg = json.loads((root / "PHASE_A_CONFIG.json").read_text())
    for filename in ("PREMEASUREMENT_CODE_FREEZE.json", "PREMEASUREMENT_ASSET_SUPPLEMENT.json"):
        freeze = json.loads((root / filename).read_text())
        for name, sha in freeze["files"].items():
            assert digest(root / name) == sha, ("FROZEN_FILE_CHANGED", name)
        for name, sha in freeze.get("dependencies", {}).items():
            assert digest(name) == sha, ("FROZEN_DEPENDENCY_CHANGED", name)
    rows, completed, active = [], 0.0, 0.0
    for receipt_path in sorted((root / "runs").glob("*.LAUNCH_AUTHORIZATION.json")):
        receipt = json.loads(receipt_path.read_text())
        doc, mode = receipt["document"], receipt["mode"]
        gpu = cfg["document_gpu_map"][doc]
        assert receipt["gpu_uuid"] == cfg["gpu_uuids"][str(gpu)]
        assert receipt["runner_sha256"] == digest(root / "run_matched_l7.py")
        tag = receipt_path.name.removesuffix(".LAUNCH_AUTHORIZATION.json")
        canonical = root / "runs" / (tag + ".json")
        stop = root / "runs" / (tag + ".STOP.json")
        started = root / "runs" / (tag + ".STARTED.json")
        row = {"tag": tag, "document": doc, "mode": mode,
               "gpu_uuid": receipt["gpu_uuid"], "launch_epoch": receipt["epoch"],
               "launch_receipt_sha256": digest(receipt_path)}
        if started.exists():
            start = json.loads(started.read_text())
            assert start["gpu_uuid"] == receipt["gpu_uuid"]
            row["started_receipt_sha256"] = digest(started)
            row["pid"] = start["pid"]
            row["process_alive"] = Path(f"/proc/{start['pid']}").exists()
        if stop.exists():
            result = json.loads(stop.read_text())
            end, status = result["stopped_epoch"], "FAIL_AND_STOP_NO_RETRY"
            row["stop_sha256"] = digest(stop)
        elif canonical.exists() and canonical.with_suffix(".sha256.json").exists():
            result = json.loads(canonical.read_text())
            sidecar = json.loads(canonical.with_suffix(".sha256.json").read_text())
            assert sidecar["sha256"] == digest(canonical), "CANONICAL_SIDECAR_MISMATCH"
            assert result["valid"] and result["gpu_uuid"] == receipt["gpu_uuid"]
            assert result["runner_sha256"] == receipt["runner_sha256"]
            assert result["fp_reference"]["replay_parity"]["passed"]
            assert all(x["mechanical_gates"]["passed"] for x in result["results"])
            end, status = result["completed_epoch"], "COMPLETE_MECHANICAL_PASS"
            row["canonical_sha256"] = sidecar["sha256"]
            row["sidecar_sha256"] = digest(canonical.with_suffix(".sha256.json"))
            row["fp_reference_content_sha256"] = result["fp_reference"]["content_sha256"]
            row["runner_accounted_gpu_hours"] = result["accounted_gpu_hours"]
        else:
            end = now
            status = "ACTIVE_OR_FINALIZING" if row.get("process_alive") else "LAUNCHED_STATE_REQUIRES_INSPECTION"
        hours = max(0.0, end - receipt["epoch"]) / 3600
        row.update(status=status, gpu_hours_upper_bound=hours)
        if status in ("COMPLETE_MECHANICAL_PASS", "FAIL_AND_STOP_NO_RETRY"):
            completed += hours
        else:
            active += hours
        rows.append(row)
    assert completed + active <= cfg["max_gpu_hours"], "TOTAL_BUDGET_EXCEEDED"
    return {"schema": "append_only_phase_a_gpu_ledger_v1", "snapshot_epoch": now,
            "completed_gpu_hours": completed, "active_gpu_hours": active,
            "total_gpu_hours_upper_bound": completed + active, "gpu_runs": rows,
            "max_gpu_hours": cfg["max_gpu_hours"], "cpu_audits_gpu_hours": 0,
            "unrelated_external_experiment_hours_counted": False,
            "startup_loading_and_fp_reference_included": True,
            "scientific_outcome_metrics_used": False,
            "initial_ledger_sha256": digest(root / "GPU_COST_LEDGER.json"),
            "snapshot_code_sha256": digest(__file__)}


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "AUDIT_MUST_BE_CPU_ONLY"
    root = Path(__file__).resolve().parent
    snapshot = audit(root, time.time())
    output = root / f"GPU_COST_LEDGER_SNAPSHOT_{time.time_ns()}.json"
    publish(output, snapshot)
    publish(output.with_suffix(".sha256.json"), {"path": str(output), "sha256": digest(output)})
    print(json.dumps({"snapshot": str(output), "sha256": digest(output),
                      "completed_gpu_hours": snapshot["completed_gpu_hours"],
                      "active_gpu_hours": snapshot["active_gpu_hours"],
                      "run_status": [x["status"] for x in snapshot["gpu_runs"]]}))
