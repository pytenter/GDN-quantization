"""Numerically blind cost gate: both real microbenchmarks, entire four-unit roster."""
import json
import time
from objective import digest
from run_matched_l7 import ROOT, publish
from persistent_stage_common_v1 import verify, complete, ledger, project_document, latch


def main():
    cfg, selection, _ = verify()
    account = ledger()
    assert all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in account["persistent_runs"])
    predictions, sources = {}, {}
    for entry in selection["documents"]:
        complete("qualification", entry["document"], entry["gpu"])
        value, path = complete("micro", entry["document"], entry["gpu"])
        predictions[entry["document"]] = project_document(value)
        sources[entry["document"]] = digest(path)
    total = account["total_gpu_hours_upper_bound"] + sum(x["gpu_hours"] for x in predictions.values())
    passed = total <= 16 and all(x["memory_pass"] for x in predictions.values())
    value = dict(status="PASS" if passed else "ENGINEERING_OR_BUDGET_STOP",
        epoch=time.time(), completed_entire_task_ledger=account, document_projections=predictions,
        conservative_entire_task_gpu_hours=total, max_gpu_hours=16,
        formal_scientific_units=4, independent_documents=2, fresh_same_gpu_fp_references=2,
        loading_smoke_micro_fp_all_prior_cost_included=True, micro_sha256=sources,
        pair_selection_sha256=digest(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json"),
        projection_code_sha256=digest(ROOT / "persistent_stage_common_v1.py"),
        scientific_kl_or_winner_outcomes_used_for_budget=False,
        no_partial_roster_no_replacement_documents=True)
    path = ROOT / "PERSISTENT_COMPLETE_COST_GATE_V1.json"
    publish(path, value); publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    if not passed:
        latch("COMPLETE_FOUR_UNITS_PLUS_FP_COST_OR_NATIVE_CACHE_CAPACITY_GATE_FAIL", "cost",
              cost_gate_sha256=digest(path), scientific_fail=False)
    print(json.dumps(dict(status=value["status"], conservative_entire_task_gpu_hours=total, sha256=digest(path))))


if __name__ == "__main__":
    main()
