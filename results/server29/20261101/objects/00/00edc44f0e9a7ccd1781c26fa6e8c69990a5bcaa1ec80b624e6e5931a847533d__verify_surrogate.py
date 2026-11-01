"""Independent CPU validation of delivered scores, rankings and immutable hashes."""
import csv
import json
import math
import numpy as np
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, sha256, atomic_json

files = ["SURROGATE_PROTOCOL.md", "TRAIN_TIME_INFORMATION_AUDIT.md", "SHORT_HORIZON_PROXY_SCORES.csv",
         "NATURAL_ROTATION_RANKING_REPORT.md", "COST_AND_TRAINABILITY_AUDIT.md", "NEXT_EXPERIMENT_DECISION.json",
         "REPRODUCIBILITY_REPORT.md"]
assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size for p in files)
config = json.loads((ROOT / "frozen_config.json").read_text())
schedule = json.loads((ROOT / "PANEL_SCHEDULE_FREEZE.json").read_text())
freeze = json.loads((ROOT / "PROXY_SCORE_FREEZE.json").read_text())
assert sha256(ROOT / "SHORT_HORIZON_PROXY_SCORES.csv") == freeze["score_csv_sha256"]
for path, digest in freeze["run_files"].items():
    assert sha256(Path(path)) == digest
for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
    digest, name = line.split("  ", 1)
    assert sha256(ROOT / name) == digest, ("ARTIFACT_HASH_MISMATCH", name)
rows = list(csv.DictReader((ROOT / "SHORT_HORIZON_PROXY_SCORES.csv").open()))
assert len(rows) == 2 * len(schedule["conditions"]) * 3
assert len({(x["document"],x["condition"],x["start"]) for x in rows}) == len(rows)
metric_keys = [k for k in rows[0] if k.startswith(("P", "B", "layer20_"))]
for doc in config["documents"]:
    for condition in schedule["conditions"]:
        selected = [x for x in rows if x["document"]==doc and x["condition"]==condition]
        windows = [x for x in selected if x["row_type"]=="WINDOW"]
        aggregate = next(x for x in selected if x["row_type"]=="AGGREGATE")
        assert {int(x["start"]) for x in windows} == {256,1024}
        assert aggregate["gpu_uuid"] == config["gpu_uuids"][str(config["document_gpu_map"][doc])]
        for key in metric_keys:
            assert float(aggregate[key]) == sum(float(x[key]) for x in windows)/2
            assert all(math.isfinite(float(x[key])) for x in selected)
# Independently calculate each rho with NumPy from the saved midrank CSV.
ranks = list(csv.DictReader((ROOT / "ALL_ROTATION_RANKS.csv").open()))
stats = list(csv.DictReader((ROOT / "PANEL_STATISTICS.csv").open()))
for row in stats:
    panel = [x for x in ranks if x["document"]==row["document"] and x["anchor_role"]==row["anchor_role"] and x["metric"]==row["metric"]]
    assert len(panel)==len(schedule["conditions"])
    x = np.array([float(p["proxy_rank"]) for p in panel], dtype=np.float64)
    y = np.array([float(p["target_rank"]) for p in panel], dtype=np.float64)
    rho = float(np.corrcoef(x,y)[0,1])
    assert abs(rho-float(row["spearman_rho"])) < 1e-12
    product = (x[:,None]-x[None,:])*(y[:,None]-y[None,:])
    triangular = product[np.triu_indices(len(x),1)]
    eligible = triangular[triangular != 0]
    concordance = float(np.mean(eligible > 0))
    assert abs(concordance-float(row["concordance"])) < 1e-12
for pair in csv.DictReader((ROOT / "ALL_RANK_REVERSALS.csv").open()):
    assert (float(pair["proxy_a"])-float(pair["proxy_b"]))*(float(pair["FutureKL_a"])-float(pair["FutureKL_b"])) < 0
analysis = json.loads((ROOT / "ANALYSIS_RESULTS.json").read_text())
decision = json.loads((ROOT / "NEXT_EXPERIMENT_DECISION.json").read_text())
assert analysis["status"] == decision["status"]
if not decision["best_candidate"]:
    assert not any(analysis["stable_positive_ranking"][m] and analysis["incremental_tests"][m]["passed"]
                   for m in analysis["stable_positive_ranking"])
ledger = json.loads((ROOT / "GPU_COST_LEDGER.json").read_text())
assert ledger["within_budget"] and ledger["total_gpu_hours_conservative"] <= 4
pre = json.loads((ROOT / "PREMEASUREMENT_FREEZE.json").read_text())
for path,digest in pre["files"].items():
    assert sha256(Path(path)) == digest
analyzer = json.loads((ROOT / "ANALYZER_FREEZE.json").read_text())
assert sha256(ROOT / "finalize_surrogate.py") == analyzer["source_sha256"]
identities = json.loads((ROOT / "REPRODUCIBILITY_IDENTITIES.json").read_text())
assert sha256(Path("/data/zypan/experiment_runs/LING_OBSERVABILITY_MECHANISM_TRIAGE_V1/OBSERVABILITY_ANALYSIS_DATASET.csv")) == identities["target_csv_sha256"]
for run in identities["formal_run_identities"]:
    assert run["fp_reference"]["archived_long_trace_used"] is False
atomic_json(ROOT / "FINAL_VERIFICATION.json", {"status": "PASS", "required_deliverables": len(files),
            "score_rows": len(rows), "independent_numpy_panel_statistics_checked": len(stats),
            "same_frozen_protocol": True, "budget_within_cap": True, "no_old_evidence_modified": True,
            "verifier_sha256": sha256(Path(__file__)), "deliverable_sha256": {p:sha256(ROOT/p) for p in files}})
print("FINAL_INDEPENDENT_VERIFICATION_PASS", decision["status"])
