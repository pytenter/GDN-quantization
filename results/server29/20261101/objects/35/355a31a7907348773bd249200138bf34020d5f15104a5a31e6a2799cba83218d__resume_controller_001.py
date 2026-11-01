#!/usr/bin/env python3
"""Append-only resume after pre-token deterministic-runtime plumbing failure."""
from __future__ import annotations

import json
import time
import traceback

import controller as base


ROOT = base.ROOT
PY = base.PY
TASK = base.TASK


def main():
    failure = ROOT / "runs" / "correctness_smoke_pg19-test-51410_gpu6.STOP.json"
    controller_failure = ROOT / "CONTROLLER_STOP.json"
    amendment = ROOT / "engineering" / "DETERMINISTIC_RUNTIME_ENABLEMENT_AMENDMENT_001.json"
    if not (failure.exists() and controller_failure.exists() and amendment.exists()):
        raise RuntimeError("REQUIRED_HISTORICAL_FAILURE_OR_AMENDMENT_MISSING")
    old = json.loads(failure.read_text())
    if old.get("error") != "RuntimeError('DETERMINISTIC_ALGORITHMS_DISABLED')":
        raise RuntimeError("UNEXPECTED_PREVIOUS_FAILURE")
    pairs = json.loads((ROOT / "PROSPECTIVE_ROTATION_SELECTION.json").read_text())
    config = json.loads((ROOT / "frozen_config.json").read_text())
    base.event("ENGINEERING_RESUME_001_STARTED", amendment_sha256=base.sha256(amendment))

    smoke_tag = "correctness_smoke_pg19-test-51410_gpu6_resume001"
    p, log = base.launch("pg19-test-51410", 6, ["IDENTITY", "H", "L6", "L7"], "correctness_smoke", smoke_tag)
    base.wait_runs([(smoke_tag, p, log)])
    smoke = base.load_receipt(smoke_tag)
    if not all(x["persistent_integrity"]["passed"] and x["all_finite"] for x in smoke["results"]):
        raise RuntimeError("CORRECTNESS_SMOKE_INVALID")
    calibration_gpu_hours = base.calibration_cost()
    spent_pre = calibration_gpu_hours + smoke["gpu_hours"]
    projected_stage1 = smoke["gpu_hours"] * (32768 / 512) * (3 / 5) * 2 * 1.25
    projected_total_stage1 = spent_pre + projected_stage1
    budget_gate = {"calibration_gpu_hours": calibration_gpu_hours, "smoke_gpu_hours": smoke["gpu_hours"],
                   "failed_pre_token_smoke_gpu_hours": 0.0,
                   "projected_stage1_additional_gpu_hours_with_25pct_margin": projected_stage1,
                   "projected_total_through_stage1": projected_total_stage1, "stage1_cap": 12.0,
                   "gate": "PASS" if projected_total_stage1 <= 12.0 else "FAIL"}
    base.atomic_json(ROOT / "freeze" / "PREMEASUREMENT_BUDGET_GATE.json", budget_gate)
    if budget_gate["gate"] != "PASS":
        raise RuntimeError(f"STAGE1_BUDGET_PROJECTION_FAIL:{budget_gate}")
    base.event("CORRECTNESS_AND_BUDGET_PASS", **budget_gate)

    stage1_jobs = []
    for doc, frozen in sorted(pairs["documents"].items()):
        gpu = config["document_gpu_map"][doc]
        tag = f"stage1_{doc}_gpu{gpu}"
        proc, log = base.launch(doc, gpu, frozen["primary_pair"]["conditions"], "stage1", tag)
        stage1_jobs.append((tag, proc, log))
    base.event("STAGE1_STARTED", units=4)
    base.wait_runs(stage1_jobs)
    stage1_receipts = {}
    for doc, frozen in pairs["documents"].items():
        gpu = config["document_gpu_map"][doc]
        receipt = base.load_receipt(f"stage1_{doc}_gpu{gpu}")
        if not all(x["persistent_integrity"]["passed"] and x["all_finite"] for x in receipt["results"]):
            raise RuntimeError(f"STAGE1_NUMERICAL_OR_SEMANTIC_FAIL:{doc}")
        stage1_receipts[doc] = receipt
    stage1_total = spent_pre + sum(x["gpu_hours"] for x in stage1_receipts.values())
    if stage1_total > 12.0:
        raise RuntimeError(f"STAGE1_ACTUAL_BUDGET_EXCEEDED:{stage1_total}")
    p1_both = all(base.actual_winner(stage1_receipts[doc], frozen["primary_pair"]) == frozen["primary_pair"]["left_predicted_winner"]
                  for doc, frozen in pairs["documents"].items())
    base.event("STAGE1_COMPLETE", gpu_hours_through_stage1=stage1_total, p1_correct_both_documents=p1_both)

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
    base.atomic_json(ROOT / "analysis" / "STAGE2_ADVANCEMENT_GATE.json", stage2_gate)
    if stage2_go:
        jobs = []
        for doc, missing in stage2_missing.items():
            if missing:
                gpu = config["document_gpu_map"][doc]
                tag = f"stage2_{doc}_gpu{gpu}"
                proc, log = base.launch(doc, gpu, missing, "stage2", tag)
                jobs.append((tag, proc, log))
        base.event("STAGE2_STARTED", new_scientific_units=sum(len(x) for x in stage2_missing.values()))
        base.wait_runs(jobs)
    else:
        base.event("STAGE2_NOT_RUN", **stage2_gate)

    import subprocess
    with (ROOT / "logs" / "analyze.log").open("x") as log:
        subprocess.run([PY, str(ROOT / "scripts" / "analyze_prospective.py")], check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    base.event("EXPERIMENT_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        path = ROOT / "CONTROLLER_STOP_RESUME001.json"
        if not path.exists():
            base.atomic_json(path, {"task": TASK, "gate": "FAIL_AND_STOP", "error": repr(exc),
                                    "traceback": traceback.format_exc(), "stopped_epoch": time.time()})
        base.event("FAIL_AND_STOP_RESUME001", error=repr(exc))
        raise
