"""Outcome-blind short-stage budget gate AFTER both new-document GPU smokes."""
import os
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common import RUNS, checked_json, verify, audit_b


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cfg = verify()
    ledger = audit_b()
    assert all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in ledger["phase_b_calibration_runs"])
    assert len(ledger["phase_b_calibration_runs"]) == 2, "TWO_SMOKES_ONLY_REQUIRED"
    old = checked_json(ROOT / "PHASE_B_SHORT_STAGE_RESERVATION.json")
    hashes, caps, costs = {}, {}, []
    for doc in cfg["documents"]:
        gpu = cfg["document_gpu_map"][doc]
        path = RUNS / f"smoke_{doc}_gpu{gpu}.json"
        smoke = checked_json(path)
        assert smoke["valid"] and smoke["fp_reference"]["replay_parity"]["passed"]
        assert smoke["gpu_uuid"] == cfg["gpu_uuids"][str(gpu)] and smoke["runner_sha256"] == digest(ROOT / "run_phase_b_calibration.py")
        assert len(smoke["results"]) == 2 and smoke["results"][0]["windows"] == smoke["results"][1]["windows"]
        assert len(smoke["fp_reference"]["helper_real_forward_checks"]) == 36
        assert all(x["mechanical_gates"]["passed"] for x in smoke["results"])
        slow = max(x["elapsed_seconds"] for x in smoke["results"])
        loading, fp = smoke["model_loading_seconds"], smoke["fp_reference"]["elapsed_seconds"]
        overhead = max(0, smoke["total_wall_seconds"] - loading - fp - sum(x["elapsed_seconds"] for x in smoke["results"]))
        # Same 384->1152 and two real windows as Phase A; take max with real full16 prior timing.
        cap = max(old["phase_a_derived_formal_caps"][doc], 1.5 * (loading + 3 * fp + 48 * slow + overhead) + 20)
        caps[doc], hashes[doc] = cap, digest(path)
        costs.append(dict(document=doc, formal_seconds_conservative=cap, margin=1.5))
    projected = ledger["total_gpu_hours_upper_bound"] + sum(caps.values()) / 3600
    status = "PASS" if projected <= 16 else "ENGINEERING_OR_BUDGET_STOP"
    path = ROOT / "PHASE_B_CALIBRATION_BUDGET_GATE.json"
    publish(path, dict(status=status, completed_gpu_hours=ledger["total_gpu_hours_upper_bound"],
        conservative_projected_total_gpu_hours=projected, per_document_wall_cap_seconds=caps,
        smoke_sha256=hashes, cost_rows=costs, outcome_metrics_used=False,
        calibration_only_not_32k_authorization=True, no_partial_roster_or_retry=True))
    publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(status, "projected_total_gpu_hours", projected)


if __name__ == "__main__":
    main()
