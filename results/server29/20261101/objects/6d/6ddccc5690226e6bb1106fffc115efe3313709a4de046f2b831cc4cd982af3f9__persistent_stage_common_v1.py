"""Append-only long-engine identity/device/cost gates, no GPU imports."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common_v2 import verify as verify_cal, audit_b, read, checked_json, idle_device

RUNS = ROOT / "runs_b_persistent"
FREEZE = ROOT / "PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json"
LATCH = ROOT / "PERSISTENT_FAIL_CLOSED_STOP.json"
MODES = {"qualification": dict(endpoint=512, primary=256, secondary=0, repeats=2, wall=900),
         "micro": dict(endpoint=8448, primary=8192, secondary=2048, repeats=1, wall=3600),
         "formal": dict(endpoint=32768, primary=8192, secondary=2048, repeats=1)}


def fingerprint(value):
    # Timings are deliberately excluded, scientific/numerical evidence is not.
    keys = ("endpoint", "fp_reference_full_logit_sha256", "fp_replay_parity_tokens",
            "hybrid_cache_inventories", "results", "full_model_teacher_forced",
            "every_token_qdq_including_after_anchors", "same_gpu_live_fp_reference")
    return hashlib.sha256(json.dumps({k: value[k] for k in keys}, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def verify():
    cfg = verify_cal()
    assert not LATCH.exists(), "PERSISTENT_STOP_NO_RETRY"
    assert not list(RUNS.glob("*.STOP.json")), "PERSISTENT_RUN_STOP_NO_RETRY"
    freeze = checked_json(FREEZE)
    for name, sha in freeze["files"].items():
        assert digest(ROOT / name) == sha, ("PERSISTENT_CODE_CHANGED", name)
    for path, sha in freeze["dependencies"].items():
        assert digest(path) == sha, ("PERSISTENT_ASSET_CHANGED", path)
    source = checked_json(ROOT / "PERSISTENT_SOURCE_CPU_AUDIT_V3.json")
    assert source["status"] == "CPU_SOURCE_AND_ARITHMETIC_AUDIT_PASS_REAL_GPU_PENDING"
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json")
    assert selection["status"] == "BOTH_DOCUMENTS_DISCRIMINATIVE_PAIRS_FROZEN"
    assert selection["route"] == "A" and len(selection["documents"]) == 2
    assert selection["primary"].endswith("[8192,32768)")
    assert {x["document"]: x["gpu"] for x in selection["documents"]} == cfg["document_gpu_map"]
    assert all(x["pair"]["minimum_rank_gap"] >= 3 for x in selection["documents"])
    assert os.uname().release == cfg["runtime"]["kernel"], "KERNEL_CHANGED"
    assert shutil.disk_usage(ROOT).free >= 2 * 1024**3, "PERSISTENT_STORAGE_FAIL"
    return cfg, selection, freeze


def complete(mode, doc, gpu):
    path = RUNS / f"{mode}_{doc}_gpu{gpu}.json"
    value = checked_json(path)
    assert value["valid"] and value["mode"] == mode and value["document"] == doc
    assert value["runner_sha256"] == digest(ROOT / "run_persistent_stage_v1.py")
    assert value["freeze_sha256"] == digest(FREEZE)
    assert value["pair_selection_sha256"] == digest(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json")
    assert value["gpu_uuid"] == read(ROOT / "PHASE_B_CALIBRATION_CONFIG.json")["gpu_uuids"][str(gpu)]
    assert value["endpoint"] == MODES[mode]["endpoint"]
    assert all(x["persistent_integrity"]["passed"] and x["all_finite"] for x in value["stream"]["results"])
    if mode == "qualification":
        assert value["repeat_bitwise_pass"] and value["stream"]["fp_replay_parity_tokens"] == 512
    return value, path


def ledger(now=None):
    now = time.time() if now is None else now
    prior = audit_b(now)
    assert all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in prior["phase_b_calibration_runs"])
    rows = []
    cfg = read(ROOT / "PHASE_B_CALIBRATION_CONFIG.json")
    for p in sorted(RUNS.glob("*.LAUNCH_AUTHORIZATION.json")):
        rec = read(p); tag = p.name.removesuffix(".LAUNCH_AUTHORIZATION.json")
        assert rec["gpu_uuid"] == cfg["gpu_uuids"][str(rec["gpu"])]
        assert rec["runner_sha256"] == digest(ROOT / "run_persistent_stage_v1.py")
        assert rec["freeze_sha256"] == digest(FREEZE)
        row = dict(tag=tag, document=rec["document"], gpu=rec["gpu"], mode=rec["mode"],
                   gpu_uuid=rec["gpu_uuid"], receipt_sha256=digest(p), launch_epoch=rec["epoch"],
                   session=rec["session"])
        started, stop, canonical = (RUNS / (tag + s) for s in (".STARTED.json", ".STOP.json", ".json"))
        if started.exists():
            st = read(started)
            assert st["runner_sha256"] == rec["runner_sha256"] and st["gpu_uuid"] == row["gpu_uuid"]
            row.update(pid=st["pid"], process_alive=Path(f"/proc/{st['pid']}").exists(), started_sha256=digest(started))
        if stop.exists():
            end, status = read(stop)["stopped_epoch"], "FAIL_AND_STOP_NO_RETRY"
            row["stop_sha256"] = digest(stop)
        elif canonical.exists() and canonical.with_suffix(".sha256.json").exists():
            value, _ = complete(rec["mode"], rec["document"], rec["gpu"])
            end, status = value["completed_epoch"], "COMPLETE_MECHANICAL_PASS"
            row.update(canonical_sha256=digest(canonical), sidecar_sha256=digest(canonical.with_suffix(".sha256.json")),
                       runner_accounted_gpu_hours=value["accounted_gpu_hours"])
        else:
            end = now
            startup = not started.exists() and now - rec["epoch"] < 60 and subprocess.run(
                ["tmux", "has-session", "-t", rec["session"]], capture_output=True).returncode == 0
            status = "ACTIVE_OR_FINALIZING" if row.get("process_alive") or startup else "AUTHORIZED_STATE_REQUIRES_INSPECTION"
        row.update(status=status, gpu_hours_upper_bound=max(0, end - rec["epoch"]) / 3600,
                   remaining_wall_reservation_hours=max(0, rec["epoch"] + rec["max_wall_seconds"] - end) / 3600
                   if status in ("ACTIVE_OR_FINALIZING", "AUTHORIZED_STATE_REQUIRES_INSPECTION") else 0)
        rows.append(row)
    total = prior["total_gpu_hours_upper_bound"] + sum(x["gpu_hours_upper_bound"] for x in rows)
    assert total <= 16, "ENTIRE_TASK_GPU_BUDGET_EXCEEDED"
    return dict(schema="entire_task_a_b_calibration_and_persistent_ledger_v1", epoch=now,
                prior_a_and_b_calibration=prior, persistent_runs=rows, total_gpu_hours_upper_bound=total,
                max_gpu_hours=16, all_loading_fp_smoke_micro_startup_included=True,
                external_waiting_and_cpu_not_counted=True, numerical_outcome_used_for_budget=False)


def project_document(micro):
    """Frozen conservative worst linear-per-token attention growth envelope."""
    end, full = 8448, 32768
    assert micro["endpoint"] == end
    stream = micro["stream"]
    rates = [b["elapsed_seconds"] / b["tokens"] for b in stream["blocks"] if b["end"] > 2048]
    assert rates and all(math.isfinite(x) and x > 0 for x in rates)
    # Actual prefix includes loading/JIT/probes; remaining worst-rate grows by
    # full/micro, then the complete prediction receives a separate 1.5 margin.
    seconds = 1.5 * (micro["total_wall_seconds"] + (full - end) * max(rates) * full / end)
    inventories = stream["hybrid_cache_inventories"][str(end - 1)]
    cache_now = sum(x["cache_bytes"] for x in inventories.values())
    cache_full = cache_now * full / end
    predicted_peak = micro["gpu_memory"]["peak_allocated_bytes"] + cache_full - cache_now
    memory_pass = predicted_peak <= .85 * micro["gpu_memory"]["total_bytes"]
    wall_cap = math.ceil(seconds + 30)
    return dict(max_late_block_seconds_per_token=max(rates), growth_multiplier=full / end,
                wall_cap_seconds=wall_cap, conservative_seconds=wall_cap,
                gpu_hours=wall_cap / 3600, predicted_peak_allocated_bytes=predicted_peak,
                memory_safety_fraction=.85, memory_pass=memory_pass,
                all_three_native_full_model_streams_included=True, fp_plus_two_rotations=True)


def latch(reason, stage, **extra):
    try:
        publish(LATCH, dict(status="FAIL_CLOSED_NO_RETRY", reason=reason, stage=stage,
                            epoch=time.time(), no_signals_used=True, **extra))
    except FileExistsError:
        pass
