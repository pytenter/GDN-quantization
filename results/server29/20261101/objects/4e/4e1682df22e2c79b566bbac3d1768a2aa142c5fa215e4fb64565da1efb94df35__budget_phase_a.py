"""Outcome-blind Phase A cost gate, using both real matched smoke runtimes."""
import json
from run_matched_l7 import ROOT, publish
from objective import digest

cfg = json.loads((ROOT/"PHASE_A_CONFIG.json").read_text())
freezed = json.loads((ROOT/"PREMEASUREMENT_CODE_FREEZE.json").read_text())
assert freezed["budget_smoke_projection_margin"] == 1.5
hashes, caps, rows = {}, {}, []
completed = 0
for doc in cfg["documents"]:
    gpu = cfg["document_gpu_map"][doc]
    path = ROOT/"runs"/f"smoke_{doc}_gpu{gpu}.json"
    smoke = json.loads(path.read_text())
    sidecar = json.loads(path.with_suffix(".sha256.json").read_text())
    assert sidecar["sha256"] == digest(path), "SMOKE_CANONICAL_SIDECAR_MISMATCH"
    assert smoke["valid"] and smoke["gpu_uuid"] == cfg["gpu_uuids"][str(gpu)]
    assert smoke["mode"] == "smoke" and len(smoke["results"]) == 2
    assert smoke["results"][0]["windows"] == smoke["results"][1]["windows"]
    assert all(x["mechanical_gates"]["passed"] for x in smoke["results"])
    assert smoke["fp_reference"]["replay_parity"]["passed"]
    assert len(smoke["fp_reference"]["helper_real_forward_checks"]) == 36
    assert smoke["runner_sha256"] == digest(ROOT/"run_matched_l7.py")
    hashes[doc] = digest(path)
    completed += smoke["accounted_gpu_hours"]
    slow = max(x["elapsed_seconds"] for x in smoke["results"])
    load = smoke["model_loading_seconds"]
    fp = smoke["fp_reference"]["elapsed_seconds"]
    overhead = max(0,smoke["total_wall_seconds"]-load-fp-
                   sum(x["elapsed_seconds"] for x in smoke["results"]))
    # All same-run measured components included; 1152/384=3 and all16 roster.
    cap = 1.5*(load+3*fp+3*16*slow+overhead)+10
    caps[doc] = cap
    rows.append({"document":doc,"smoke_gpu_hours":smoke["accounted_gpu_hours"],
                 "formal_seconds_conservative":cap,"margin":1.5})
projection = completed + sum(caps.values())/3600
status = "PASS" if projection <= cfg["phase_a_gpu_cap"] else "ENGINEERING_OR_BUDGET_STOP"
publish(ROOT/"PHASE_A_BUDGET_GATE.json", {"status":status,
    "completed_gpu_hours":completed,"conservative_projected_gpu_hours":projection,
    "per_document_wall_cap_seconds":caps,"smoke_sha256":hashes,"cost_rows":rows,
    "scientific_outcome_metrics_used":False,"hard_total_budget_gpu_hours":16,
    "phase_a_target_cap_gpu_hours":2,"no_retry_or_partial_roster_fallback":True})
print(status,"projected_Phase_A_GPU_hours",projection)
