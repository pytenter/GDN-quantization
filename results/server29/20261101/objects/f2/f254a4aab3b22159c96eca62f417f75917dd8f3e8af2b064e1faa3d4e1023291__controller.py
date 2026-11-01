#!/usr/bin/env python3
"""Detached fail-closed controller for the prospective P1 pilot."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


TASK = "LING_KDA_PERSISTENT_SURROGATE_PROSPECTIVE_VALIDATION_V1"
ROOT = Path("/data/zypan/experiment_runs") / TASK
PY = "/data/zypan/envs/ling-kda/bin/python"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value):
    if path.exists():
        raise FileExistsError(f"NO_OVERWRITE:{path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.link(tmp, path); tmp.unlink()


def event(stage, **kwargs):
    value = {"time_epoch": time.time(), "stage": stage, **kwargs}
    with (ROOT / "logs" / "controller_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    print(json.dumps(value, sort_keys=True), flush=True)


def env_for(gpu):
    value = os.environ.copy()
    value.update({"CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                  "LING_GATED_RMSNORM_FIXED_CONFIG": "BT16_W8", "PYTHONUNBUFFERED": "1"})
    return value


def launch(document, gpu, conditions, role, tag):
    cmd = [PY, str(ROOT / "scripts" / "run_persistent_pair.py"), "--document", document,
           "--gpu", str(gpu), "--conditions", ",".join(conditions), "--role", role, "--tag", tag]
    log = (ROOT / "logs" / f"{tag}.log").open("x")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env_for(gpu))
    return proc, log


def wait_runs(items):
    failures = []
    for tag, proc, log in items:
        code = proc.wait(); log.close()
        event("RUN_EXIT", tag=tag, exitcode=code)
        if code:
            failures.append((tag, code))
    if failures:
        raise RuntimeError(f"RUN_FAILURES:{failures}")


def load_receipt(tag):
    path = ROOT / "runs" / tag / "COMPLETE.json"
    value = json.loads(path.read_text())
    if value.get("gate") != "PASS":
        raise RuntimeError(f"RECEIPT_NOT_PASS:{tag}")
    return value


def calibration_cost():
    selection = json.loads((ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json").read_text())
    total = 0.0
    for doc in selection["selected_documents"]:
        path = ROOT / "runs" / f"calibration_{doc['document_id']}_gpu{doc['physical_gpu']}.json"
        total += json.loads(path.read_text())["accounted_gpu_hours"]
    return total


def actual_winner(receipt, pair):
    by = {x["condition"]: x["primary"]["mean"] for x in receipt["results"]}
    a, b = pair["conditions"]
    return a if by[a] < by[b] else b if by[b] < by[a] else "TIE"


def main():
    event("WAIT_CALIBRATION")
    calibration_exit = ROOT / "logs" / "calibration_pair_exitcodes.json"
    while not calibration_exit.exists():
        time.sleep(30)
    codes = json.loads(calibration_exit.read_text())
    if any(int(v) != 0 for v in codes.values()):
        raise RuntimeError(f"CALIBRATION_FAILED:{codes}")
    event("CALIBRATION_COMPLETE", exitcodes=codes)

    subprocess.run([PY, str(ROOT / "scripts" / "select_pairs.py")], check=True,
                   stdout=(ROOT / "logs" / "select_pairs.log").open("x"), stderr=subprocess.STDOUT)
    pair_path = ROOT / "PROSPECTIVE_ROTATION_SELECTION.json"
    pairs = json.loads(pair_path.read_text())
    if not pairs["primary_pairs_complete"]:
        raise RuntimeError("PRIMARY_PAIRS_INCOMPLETE")
    freeze = {
        "task": TASK, "freeze_epoch": time.time(), "before_any_long_horizon_outcome": True,
        "rotation_selection": {"path": str(pair_path), "sha256": sha256(pair_path)},
        "config_sha256": sha256(ROOT / "frozen_config.json"),
        "runner_sha256": sha256(ROOT / "scripts" / "run_persistent_pair.py"),
        "analyzer_sha256": sha256(ROOT / "scripts" / "analyze_prospective.py"),
        "primary_window": [8192, 32768], "secondary_window": [2048, 32768],
        "primary_metric": "mean full-vocabulary KL(FP||persistent-QDQ)",
        "stage2_go_rule": "all stage1 valid AND P1 correct on both documents AND both P1/P3 pairs frozen AND no duplicate scientific key AND projected total <=20 GPUh",
        "no_post_outcome_pair_change": True,
    }
    atomic_json(ROOT / "freeze" / "PREMEASUREMENT_FREEZE.json", freeze)
    event("PAIR_SCHEDULE_FROZEN", pair_sha256=freeze["rotation_selection"]["sha256"])

    smoke_tag = "correctness_smoke_pg19-test-51410_gpu6"
    p, log = launch("pg19-test-51410", 6, ["IDENTITY", "H", "L6", "L7"], "correctness_smoke", smoke_tag)
    wait_runs([(smoke_tag, p, log)])
    smoke = load_receipt(smoke_tag)
    if not all(x["persistent_integrity"]["passed"] and x["all_finite"] for x in smoke["results"]):
        raise RuntimeError("CORRECTNESS_SMOKE_INVALID")
    calibration_gpu_hours = calibration_cost()
    spent_pre = calibration_gpu_hours + smoke["gpu_hours"]
    # Smoke has FP+4 branches; formal has FP+2 branches.  Scale measured wall
    # time by trajectories and tokens, then retain 25% safety headroom.
    projected_stage1 = smoke["gpu_hours"] * (32768 / 512) * (3 / 5) * 2 * 1.25
    projected_total_stage1 = spent_pre + projected_stage1
    budget_gate = {"calibration_gpu_hours": calibration_gpu_hours, "smoke_gpu_hours": smoke["gpu_hours"],
                   "projected_stage1_additional_gpu_hours_with_25pct_margin": projected_stage1,
                   "projected_total_through_stage1": projected_total_stage1, "stage1_cap": 12.0,
                   "gate": "PASS" if projected_total_stage1 <= 12.0 else "FAIL"}
    atomic_json(ROOT / "freeze" / "PREMEASUREMENT_BUDGET_GATE.json", budget_gate)
    if budget_gate["gate"] != "PASS":
        raise RuntimeError(f"STAGE1_BUDGET_PROJECTION_FAIL:{budget_gate}")
    event("CORRECTNESS_AND_BUDGET_PASS", **budget_gate)

    config = json.loads((ROOT / "frozen_config.json").read_text())
    stage1_jobs = []
    for doc, frozen in sorted(pairs["documents"].items()):
        gpu = config["document_gpu_map"][doc]
        tag = f"stage1_{doc}_gpu{gpu}"
        proc, log = launch(doc, gpu, frozen["primary_pair"]["conditions"], "stage1", tag)
        stage1_jobs.append((tag, proc, log))
    event("STAGE1_STARTED", units=4)
    wait_runs(stage1_jobs)
    stage1_receipts = {}
    for doc, frozen in pairs["documents"].items():
        gpu = config["document_gpu_map"][doc]
        receipt = load_receipt(f"stage1_{doc}_gpu{gpu}")
        if not all(x["persistent_integrity"]["passed"] and x["all_finite"] for x in receipt["results"]):
            raise RuntimeError(f"STAGE1_NUMERICAL_OR_SEMANTIC_FAIL:{doc}")
        stage1_receipts[doc] = receipt
    stage1_total = spent_pre + sum(x["gpu_hours"] for x in stage1_receipts.values())
    if stage1_total > 12.0:
        raise RuntimeError(f"STAGE1_ACTUAL_BUDGET_EXCEEDED:{stage1_total}")
    p1_both = all(actual_winner(stage1_receipts[doc], frozen["primary_pair"]) == frozen["primary_pair"]["left_predicted_winner"]
                  for doc, frozen in pairs["documents"].items())
    event("STAGE1_COMPLETE", gpu_hours_through_stage1=stage1_total, p1_correct_both_documents=p1_both)

    secondary_available = all(x.get("secondary_status") == "FROZEN" for x in pairs["documents"].values())
    stage2_missing = {}
    for doc, frozen in pairs["documents"].items():
        if secondary_available:
            have = set(frozen["primary_pair"]["conditions"])
            stage2_missing[doc] = [x for x in frozen["secondary_pair"]["conditions"] if x not in have]
    projected_stage2 = 0.0
    if p1_both and secondary_available:
        for doc, missing in stage2_missing.items():
            if missing:
                projected_stage2 += stage1_receipts[doc]["gpu_hours"] * ((1 + len(missing)) / 3) * 1.20
    stage2_go = p1_both and secondary_available and stage1_total + projected_stage2 <= 20.0
    stage2_gate = {"p1_correct_both_documents": p1_both, "secondary_pairs_frozen": secondary_available,
                   "missing_new_scientific_conditions": stage2_missing,
                   "projected_stage2_gpu_hours_with_20pct_margin": projected_stage2,
                   "projected_total_gpu_hours": stage1_total + projected_stage2,
                   "gate": "PASS" if stage2_go else "NOT_RUN"}
    atomic_json(ROOT / "analysis" / "STAGE2_ADVANCEMENT_GATE.json", stage2_gate)
    if stage2_go:
        jobs = []
        for doc, missing in stage2_missing.items():
            if not missing:
                continue
            gpu = config["document_gpu_map"][doc]
            tag = f"stage2_{doc}_gpu{gpu}"
            proc, log = launch(doc, gpu, missing, "stage2", tag)
            jobs.append((tag, proc, log))
        event("STAGE2_STARTED", new_scientific_units=sum(len(x) for x in stage2_missing.values()))
        wait_runs(jobs)
    else:
        event("STAGE2_NOT_RUN", **stage2_gate)

    subprocess.run([PY, str(ROOT / "scripts" / "analyze_prospective.py")], check=True,
                   stdout=(ROOT / "logs" / "analyze.log").open("x"), stderr=subprocess.STDOUT)
    event("EXPERIMENT_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        path = ROOT / "CONTROLLER_STOP.json"
        if not path.exists():
            atomic_json(path, {"task": TASK, "gate": "FAIL_AND_STOP", "error": repr(exc),
                               "traceback": traceback.format_exc(), "stopped_epoch": time.time()})
        event("FAIL_AND_STOP", error=repr(exc))
        raise
