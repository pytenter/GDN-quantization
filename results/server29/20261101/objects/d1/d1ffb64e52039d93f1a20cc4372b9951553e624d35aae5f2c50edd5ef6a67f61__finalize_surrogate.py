"""Freeze actual scores before target join, then produce the seven deliverables."""
import csv
import itertools
import json
import math
import statistics
import time
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, sha256, atomic_json, canonical_hash, finite_tree

TRIAGE = Path("/data/zypan/experiment_runs/LING_OBSERVABILITY_MECHANISM_TRIAGE_V1")
CAUSAL = Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1")
METRICS = ["P1", "P2_W128", "P3_H32", "P3_H128", "B0", "B1_H32", "B1_H128", "B2_H32", "B2_H128"]
CANDIDATES = METRICS[:4]
BASELINES = METRICS[4:]


def ranks(values):
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for i in ordered[start:end]:
            result[i] = (start + end - 1) / 2 + 1
        start = end
    return result


def correlation(a, b):
    a = [x - statistics.mean(a) for x in a]
    b = [x - statistics.mean(b) for x in b]
    denom = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / denom if denom else None


def concordance(a, b):
    same = eligible = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        product = (a[i] - a[j]) * (b[i] - b[j])
        if product:
            eligible += 1
            same += product > 0
    return same / eligible if eligible else None


def partial_rank(a, b, control):
    def residual(x):
        x, z = ranks(x), ranks(control)
        xm, zm = statistics.mean(x), statistics.mean(z)
        var = sum((v-zm)**2 for v in z)
        slope = sum((v-xm)*(w-zm) for v,w in zip(x,z)) / var if var else 0
        return [v-xm-slope*(w-zm) for v,w in zip(x,z)]
    return correlation(residual(a), residual(b))


def write_text(path, value):
    with path.open("x", encoding="utf-8") as f:
        f.write(value)


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tests():
    assert ranks([2, 1, 2, 4]) == [2.5, 1, 2.5, 4]
    assert correlation(ranks([1,2,3]), ranks([3,2,1])) == -1
    assert concordance([1,2,3], [1,2,3]) == 1
    assert concordance([1,2,3], [3,2,1]) == 0
    assert concordance([1,1,2], [1,1,3]) == 1
    assert correlation([1,1], [2,3]) is None


def planned_primary_pilot_hours(runs):
    steps = [x["elapsed_seconds"] / x["tokens_processed"] for r in runs for x in r["results"]]
    # Two teachers plus eight condition trajectories, one anchor per document.
    # Retain a 25% planning margin; extrapolation from short prefixes is not qualification.
    return 10 * 32768 * statistics.mean(steps) / 3600 * 1.25 + 0.25


def main():
    tests()
    config = json.loads((ROOT / "frozen_config.json").read_text())
    schedule = json.loads((ROOT / "PANEL_SCHEDULE_FREEZE.json").read_text())
    ledger = json.loads((ROOT / "GPU_COST_LEDGER.json").read_text())
    assert ledger["within_budget"]
    runs, raw_rows, scores = [], [], {}
    for doc in config["documents"]:
        gpu = config["document_gpu_map"][doc]
        path = ROOT / "runs" / f"formal_{doc}_gpu{gpu}.json"
        run = json.loads(path.read_text())
        copy = dict(run)
        digest = copy.pop("content_sha256_without_self")
        assert canonical_hash(copy) == digest
        assert run["valid"] and run["starts"] == config["windows"]
        assert run["conditions"] == schedule["conditions"]
        assert run["gpu_uuid"] == config["gpu_uuids"][str(gpu)]
        assert run["fp_reference"]["replay_parity"]["passed"]
        assert run["runner_sha256"] == sha256(ROOT / "run_short_horizon_proxy_v2.py")
        run["run_path"] = str(path)
        run["run_file_sha256"] = sha256(path)
        runs.append(run)
        scores[doc] = {}
        for result in run["results"]:
            assert result["mechanical_gates"]["passed"] and finite_tree(result)
            cond = result["condition"]
            assert cond not in scores[doc]
            scores[doc][cond] = result["aggregate"]
            identity = {"document": doc, "gpu_uuid": run["gpu_uuid"], "token": run["token_file"],
                        "runtime": run["host"], "source": run["fp_reference"]}
            base = {"document": doc, "condition": cond, "gpu_uuid": run["gpu_uuid"], "data_validity": "PASS",
                    "input_sha256": run["token_file"]["sha256"],
                    "rotation_sha256": run["rotation_artifacts"][cond]["sha256"],
                    "reference_identity_sha256": canonical_hash(identity),
                    "run_file_sha256": run["run_file_sha256"],
                    "condition_elapsed_seconds": result["elapsed_seconds"],
                    "condition_gpu_hours": result["gpu_hours"], "shared_fp_seconds": run["fp_reference"]["elapsed_seconds"]}
            for window in result["windows"]:
                raw_rows.append(dict(base, row_type="WINDOW", **window))
            raw_rows.append(dict(base, row_type="AGGREGATE", start="MEAN_256_1024", **result["aggregate"]))
    score_path = ROOT / "SHORT_HORIZON_PROXY_SCORES.csv"
    write_csv(score_path, raw_rows)
    # This receipt is published BEFORE opening the target CSV.
    atomic_json(ROOT / "PROXY_SCORE_FREEZE.json", {"freeze_epoch": time.time(), "score_csv_sha256": sha256(score_path),
        "run_files": {r["run_path"]: r["run_file_sha256"] for r in runs},
        "natural_rotations_per_document": len(schedule["conditions"]), "windows_per_rotation": 2,
        "target_csv_not_yet_opened_by_analyzer": True, "discovery_not_prospective": True})
    data_path = TRIAGE / "OBSERVABILITY_ANALYSIS_DATASET.csv"
    labels = list(csv.DictReader(data_path.open()))
    panels, panel_rows, pair_rows, rank_rows = [], [], [], []
    for doc in config["documents"]:
        conditions = schedule["conditions"]
        for role in ["Primary", "Secondary"]:
            source_rows = [x for x in labels if x["document"] == doc and x["anchor_role"] == role]
            by = {x["condition"]: x for x in source_rows}
            assert len(source_rows) == 16 and len(by) == 16
            source_rows = [by[c] for c in conditions]
            for item in source_rows:
                for key, digest in [("canonical_path", "canonical_sha256"), ("sidecar_path", "sidecar_sha256")]:
                    assert sha256(Path(item[key])) == item[digest]
            target = [float(x["FutureKL"]) for x in source_rows]
            jprop = [float(x["JPROP"]) for x in source_rows]
            panel = {"document": doc, "anchor_role": role, "anchor": int(source_rows[0]["anchor"]),
                     "n": len(conditions), "statistics": {}}
            values = {m: [scores[doc][c][m] for c in conditions] for m in METRICS}
            values.update({"B3_JPROP": jprop, "B3_POR": [float(x["POR"]) for x in source_rows],
                           "B3_HISTORICAL_J_PROP": [float(x["HISTORICAL_J_PROP"]) for x in source_rows]})
            for metric, x in values.items():
                stats = {"spearman_rho": correlation(ranks(x), ranks(target)),
                         "concordance": concordance(x, target),
                         "residual_rank_after_JPROP": partial_rank(x, target, jprop)}
                panel["statistics"][metric] = stats
                panel_rows.append({"document": doc, "anchor_role": role, "metric": metric, **stats})
                for i, c in enumerate(conditions):
                    rank_rows.append({"document": doc, "anchor_role": role, "metric": metric, "condition": c,
                                      "proxy_value": x[i], "proxy_rank": ranks(x)[i], "FutureKL": target[i],
                                      "target_rank": ranks(target)[i]})
                for i, j in itertools.combinations(range(len(conditions)), 2):
                    if (x[i]-x[j])*(target[i]-target[j]) < 0:
                        pair_rows.append({"document": doc, "anchor_role": role, "kind": "RANK_REVERSAL",
                                          "metric": metric, "condition_a": conditions[i], "condition_b": conditions[j],
                                          "proxy_a": x[i], "proxy_b": x[j], "FutureKL_a": target[i], "FutureKL_b": target[j]})
            panels.append(panel)
    matched = []
    for doc in config["documents"]:
        for a, b in itertools.combinations(schedule["conditions"], 2):
            sa, sb = scores[doc][a], scores[doc][b]
            gap = abs(sa["B0"]-sb["B0"]) / ((sa["B0"]+sb["B0"])/2)
            if gap <= 0.10:
                for panel in [p for p in panels if p["document"] == doc]:
                    role = panel["anchor_role"]
                    la = next(x for x in labels if x["document"]==doc and x["condition"]==a and x["anchor_role"]==role)
                    lb = next(x for x in labels if x["document"]==doc and x["condition"]==b and x["anchor_role"]==role)
                    for m in ["P2_W128", "P3_H32", "P3_H128"]:
                        target_product = (sa[m]-sb[m])*(float(la["FutureKL"])-float(lb["FutureKL"]))
                        matched.append({"document": doc, "anchor_role": role, "condition_a": a, "condition_b": b,
                           "relative_B0_gap": gap, "metric": m, "proxy_ratio_a_over_b": sa[m]/sb[m],
                           "FutureKL_ratio_a_over_b": float(la["FutureKL"])/float(lb["FutureKL"]),
                           "agrees": target_product > 0, "tie": target_product == 0})
    write_csv(ROOT / "PANEL_STATISTICS.csv", panel_rows)
    write_csv(ROOT / "ALL_ROTATION_RANKS.csv", rank_rows)
    write_csv(ROOT / "ALL_RANK_REVERSALS.csv", pair_rows)
    write_csv(ROOT / "NATURAL_MAGNITUDE_MATCHED_CONTRASTS.csv", matched)
    stability, incremental = {}, {}
    for m in CANDIDATES:
        stability[m] = all(p["statistics"][m]["spearman_rho"] > 0 for p in panels)
        beats = {b: sum(p["statistics"][m]["spearman_rho"] > p["statistics"][b]["spearman_rho"] and
                       p["statistics"][m]["concordance"] > p["statistics"][b]["concordance"] for p in panels)
                 for b in BASELINES}
        residual = sum(p["statistics"][m]["residual_rank_after_JPROP"] is not None and
                       p["statistics"][m]["residual_rank_after_JPROP"] > 0 for p in panels)
        incremental[m] = {"beats_each_baseline_panels": beats, "positive_residual_panels": residual,
                          "passed": all(n >= 3 for n in beats.values()) and residual >= 3}
    cheaper = {m: sum(p["statistics"]["P2_W128"]["spearman_rho"] >= p["statistics"][m]["spearman_rho"] and
                      p["statistics"]["P2_W128"]["concordance"] >= p["statistics"][m]["concordance"] for p in panels)
               for m in ["P3_H32", "P3_H128"]}
    eligible = [m for m in CANDIDATES if stability[m] and incremental[m]["passed"]]
    if eligible:
        if "P2_W128" in eligible and max(cheaper.values()) >= 3:
            status, best = "CHEAPER_PROXY_PREFERRED", "P2_W128"
        else:
            status, best = "SHORT_HORIZON_READOUT_SURROGATE_SUPPORTED_FOR_VALIDATION", eligible[0]
    else:
        status, best = "SHORT_HORIZON_READOUT_SIGNAL_NOT_STABLE", None
    # Secondary causal reuse is future-query-conditioned, not a train-feasible P2 measurement.
    causal_rows = []
    for folder in sorted((CAUSAL / "runs").glob("*__formal_v1")):
        path = folder / "full/DIRECTION_AUDIT.json"
        if path.exists():
            audit = json.loads(path.read_text())
            causal_rows.append({"run": folder.name, "path": str(path), "sha256": sha256(path),
                                "visible_low_future_query_sensitivity_ratio": audit["visible_to_low_sensitivity_ratio"],
                                "classification": "RETROSPECTIVE_FUTURE_QUERY_CONDITIONED_NOT_P1_P2_P3"})
    atomic_json(ROOT / "CAUSAL_AUXILIARY_REUSE.json", {"rows": causal_rows,
                "P1_P2_P3_early_window_available": False,
                "reason": "retained checkpoints/branch scalar sensitivities do not include each signed branch error tensor and early-window queries",
                "no_repeat_8k": True, "not_evidence_of_natural_rotation_prediction": True})
    result = {"panels": panels, "stable_positive_ranking": stability, "incremental_tests": incremental,
              "P2_matches_or_exceeds_P3_panels": cheaper, "matched_contrast_rows": len(matched),
              "natural_directional_discrimination": "INSUFFICIENT" if not eligible else "DISCOVERY_ONLY",
              "status": status, "best_candidate": best, "cross_runtime": "EXPLORATORY_ASSOCIATION_ONLY"}
    atomic_json(ROOT / "ANALYSIS_RESULTS.json", result)
    decision = {"experiment": config["experiment"], "status": status, "best_candidate": best,
                "independent_validation_worthwhile": bool(eligible), "trainability_readiness": "NOT_ESTABLISHED_STATIC_AUDIT_ONLY",
                "no_training_authorized_or_started": True, "gpu_hours": ledger["total_gpu_hours_conservative"],
                "sample_size": {"documents": 2, "rotations_per_document": len(schedule["conditions"]),
                                "anchor_targets_per_document": 2, "independent_documents": 2},
                "next_step": "independent_same_runtime_validation_then_gradient_readiness" if eligible else
                             "One mechanism question: does readout risk measured later in the trajectory preserve the early-window rotation ordering? No retuning or new rollout authorized.",
                "prospective_plan_only": {"new_documents": 2, "natural_conditions_per_document": 4,
                      "selection": "frozen calibration-only proxy versus B0 ranking discordance, no long-horizon outcomes; lexicographic ties",
                      "anchor": "one Primary8192->32768 target per document; no Secondary continuation in this cost",
                      "gpu_hours_planning_estimate": planned_primary_pilot_hours(runs),
                      "cost_formula": "10 * 32768 * measured_mean_step_seconds / 3600 * 1.25 + 0.25 load allowance",
                      "estimate_status": "short-prefix extrapolation, not full-context cost qualification or execution authorization",
                      "same_runtime_same_uuid_fp_pairing": True, "run_now": False},
                "limitations": ["two correlated discovery documents", "shared-anchor inputs", "cross-driver exploratory association",
                                "historical targets are pulse/no-further-QDQ continuation, not persistent-QDQ through 32K",
                                "B2 is full-logit KL family comparator, not exact historical L7 local output MSE objective"]}
    atomic_json(ROOT / "NEXT_EXPERIMENT_DECISION.json", decision)
    table = ["| Metric | 37702 Primary | 37702 Secondary | 5396 Primary | 5396 Secondary |", "|---|---:|---:|---:|---:|"]
    for m in METRICS + ["B3_JPROP", "B3_POR", "B3_HISTORICAL_J_PROP"]:
        table.append("| " + m + " | " + " | ".join(f'{p["statistics"][m]["spearman_rho"]:.6f} / {p["statistics"][m]["concordance"]:.6f}' for p in panels) + " |")
    report = f"""# Natural rotation ranking report

Final scientific decision: `{status}`. Best validated candidate: `{best}`.

Actual coverage: two documents, {len(schedule['conditions'])} natural rotations each, two early calibration windows each, and four historical document-anchor targets. These are **two independent documents**, not 64 independent replications. No p-values, in-sample R-squared claims or post-hoc window/layer selection are used.

The table reports Spearman rho / pairwise concordance. Higher proxy means greater damage; pairwise ties are excluded. Residual association is Pearson correlation of the rank residuals after an intercept plus rank(JPROP), i.e. partial Spearman, reported in PANEL_STATISTICS.csv.

""" + "\n".join(table) + "\n\nFrozen stability/incremental rules and their complete results:\n\n```json\n" + json.dumps({"stability": stability, "incremental": incremental, "cheaper": cheaper}, indent=2) + "\n```\n"
    report += f"""
All per-condition rankings are in ALL_ROTATION_RANKS.csv; every discordant pair is in ALL_RANK_REVERSALS.csv. No counterexample was dropped. All B0-relative-gap <=10% pairs are enumerated in NATURAL_MAGNITUDE_MATCHED_CONTRASTS.csv ({len(matched)} metric-panel rows). Natural directional discrimination is `{result['natural_directional_discrimination']}`; there was no frozen numerical threshold for a 'significant' P2/P3 contrast, so no such threshold was invented after label inspection.

The subgroup H/L6/L7 can be read directly from the complete rank CSV. Layer20 values are retained in the score CSV only as auxiliary diagnostics, never substituted for all-layer rankings.

Historical JPROP/POR use realized future states/queries and are retrospective benchmarks, not deployable short-window objectives. The historical target is full-vocabulary FP||pulse KL AUC after a natural persistent-QDQ prefix to anchor, with no further QDQ in the continuation. The source error can influence future recurrent propagation and logits, but this target does not measure continued INT8 quantizer feedback through 32K.

Proxies use driver 580.178.04 and labels 580.173.02: **EXPLORATORY_ASSOCIATION_ONLY**, not cross-runtime trajectory equivalence, independent validation, or an Attempt05 merge. Shared calibration scores are reused for both anchors; the anchors are robustness checks, not new independent samples.

Artificial-direction auxiliary data preserve the known future-query sensitivity contrast, but cannot supply the early P1/P2/P3 tensors cheaply. CAUSAL_AUXILIARY_REUSE.json marks this boundary; no 8K continuation was repeated. Extreme VISIBLE/LOW separation is not substituted for natural-rotation evidence.

At most one next mechanism question is proposed in NEXT_EXPERIMENT_DECISION.json. No objective training, optimizer update, new rotation or prospective 32K validation was run.
"""
    write_text(ROOT / "NATURAL_ROTATION_RANKING_REPORT.md", report)
    condition_results = [x for r in runs for x in r["results"]]
    mean_step = statistics.mean(x["elapsed_seconds"]/x["tokens_processed"] for x in condition_results)
    p2_cost = statistics.mean(x["metric_overhead_seconds"]["P2_seconds"]/2 for x in condition_results)
    readout_step = statistics.mean(x["metric_overhead_seconds"]["readout_and_state_seconds"]/256 for x in condition_results)
    kl_step = statistics.mean(x["metric_overhead_seconds"]["functional_kl_seconds"]/256 for x in condition_results)
    costs = {"P1_local_readout_seconds_upper_bound": readout_step, "P2_local_contraction_seconds": p2_cost,
             "P3_H32_rollout_seconds_estimated_from_actual_average_step": 32*mean_step,
             "P3_H128_rollout_seconds_estimated_from_actual_average_step": 128*mean_step,
             "B2_KL_reduction_seconds_per_token": kl_step,
             "all_panel_measurements_share_forward_cost": True,
             "backward_memory_measured": False,
             "persistent_state_lower_bound_H128_FP32_GiB": 128*18*16*128*128*4/(1024**3)}
    atomic_json(ROOT / "COST_SUMMARY.json", costs)
    benefit = []
    for m in CANDIDATES:
        local_seconds = readout_step if m=="P1" else p2_cost if m=="P2_W128" else (32 if m=="P3_H32" else 128)*mean_step
        delta = statistics.mean(p["statistics"][m]["spearman_rho"]-p["statistics"]["B0"]["spearman_rho"] for p in panels)
        benefit.append({"metric": m, "mean_rho_increment_vs_B0": delta, "estimated_local_seconds_per_window": local_seconds,
                        "increment_per_marginal_gpu_hour": delta/(local_seconds/3600),
                        "caution": "descriptive ratio, excludes shared prefix/teacher acquisition; not independent predictive replication"})
    write_csv(ROOT / "PREDICTIVE_BENEFIT_PER_GPU_HOUR.csv", benefit)
    cost_report = f"""# Cost and trainability audit

Conservative total: **{ledger['total_gpu_hours_conservative']:.6f} GPU-hours / 4.0**. Includes four retained engineering-failed probes, repeated micro, model loading, current-runtime FP teacher, actual short prefixes, real quantized forward, metric reductions and shutdown. GPU-hours sum both cards, not elapsed wall time. GPU_COST_LEDGER.json is the budget authority. Maximum measured forward VRAM: {max(x['peak_vram_bytes'] for x in condition_results)/(1024**3):.3f} GiB.

Model loads: {[round(r['model_loading_seconds'],3) for r in runs]} seconds; FP references: {[round(r['fp_reference']['elapsed_seconds'],3) for r in runs]} seconds. Mean actual condition step: {mean_step:.6f} seconds. Each formal rotation computes one 1152-token prefix and both 128-token windows; H32 is a subset of H128, so these integrated measurements do not separately benchmark a dedicated H32-only trainer.

Marginal cost estimates from measured reductions/forward throughput:

```json
{json.dumps(costs, indent=2)}
```

P1 timing includes state-MSE and auxiliary reductions and is an upper bound for P1 alone. P2 timing excludes common FP history acquisition/covariance formation. Dedicated train-time cost requires those shared teacher/prefix costs. PREDICTIVE_BENEFIT_PER_GPU_HOUR.csv reports both the signed rho increment over B0 and the descriptive marginal-cost ratio; a large ratio for a tiny negative/unstable effect is not evidence of utility. Every candidate shares the actual panel acquisition cost; costs must not be summed repeatedly across CSV window/aggregate rows.

## Static trainability findings (no backward or optimizer run)

Original Cayley code constructs A(theta) skew-symmetric and R=(I-A)(I+A)^-1 with torch.linalg.solve; 8128 parameters/layer, shared across heads. Original exact-forward STE is value+(dequant-value).detach(). It permits surrogate gradients through the real forward QDQ, but is a biased identity backward and does not differentiate code/scale discontinuities faithfully.

P1/P2 mathematics can connect physical E through rotation, QDQ and inverse rotation to Cayley parameters with an explicit STE. However the reused historical K3 helper **detaches E and q**, and this evaluator runs inference/no-grad: it is **not trainable as written**. A future implementation must preserve teacher detachment but remove the error-path detach, verify STE forward identity and finite nonzero rotation gradients. No readiness PASS is claimed here.

P3 additionally needs H32/H128 truncated recurrent BPTT, with numerical student state persisting across detached prefixes. FP teacher/query/covariance can remain detached. The state-tensor-only H128 storage lower bound is {costs['persistent_state_lower_bound_H128_FP32_GiB']:.3f} GiB; this excludes kernel intermediates, query/value tensors, model activations, logits and autograd workspaces. Forward peak VRAM does not establish backward feasibility; checkpointing or shorter H32 may be necessary.

Historical training has a BF16 quant boundary/dequant state and local gated o_norm/o_proj relative-MSE functional objective (+0.1 state loss). Our inherited deployed Phase2 uses FP32 physical recurrent-cache dequant, and B2 is full-logit KL. Therefore B2 is a short functional-family baseline, **not an exact reproduction of the historical L7 training loss**, nor proof that a new loss improves L7 optimization. These distinctions must be resolved in any later gradient-readiness probe.

Prediction and trainability remain separate. No optimizer.step, backward GPU experiment, new learned checkpoint or L9 was performed. A same-runtime prospective pilot would use two held-out documents, four calibration-selected natural rotations per document, one Primary8192-to-32768 target per document and same-UUID FP pairing. The measured-throughput extrapolation is {planned_primary_pilot_hours(runs):.3f} GPU-hours: ten 32K-equivalent passes (two teachers + eight conditions), multiplied by 1.25, plus 0.25 hours of load allowance. This is not a full-context benchmark or authorization; a second anchor adds continuation cost, and attention/cache cost can change with context length. This round does not run it.
"""
    write_text(ROOT / "COST_AND_TRAINABILITY_AUDIT.md", cost_report)
    audit = json.loads((ROOT / "SOURCE_AND_ASSET_AUDIT.json").read_text())
    repro = {"frozen_config_sha256": sha256(ROOT / "frozen_config.json"),
             "protocol_sha256": sha256(ROOT / "SURROGATE_PROTOCOL.md"), "analyzer_sha256": sha256(Path(__file__)),
             "score_freeze_sha256": sha256(ROOT / "PROXY_SCORE_FREEZE.json"),
             "target_csv_sha256": sha256(data_path), "formal_run_identities": [{k:r[k] for k in
                ["document", "gpu_uuid", "host", "token_file", "rotation_artifacts", "run_file_sha256", "fp_reference"]} for r in runs],
             "source_audit": audit, "statistics_cpu_tests": "PASS", "no_old_canonical_modified": True}
    atomic_json(ROOT / "REPRODUCIBILITY_IDENTITIES.json", repro)
    write_text(ROOT / "REPRODUCIBILITY_REPORT.md", "# Reproducibility report\n\nComplete source/model/rotation/token/runtime identities and file hashes are recorded in REPRODUCIBILITY_IDENTITIES.json and SHA256SUMS.txt. The premeasurement freeze precedes new proxy measurement, and PROXY_SCORE_FREEZE.json precedes this analyzer's first target CSV open. Historical outcomes were already known: these receipts do not convert discovery into prospective confirmation.\n\nAll 80 inherited canonical and sidecar hashes were rechecked, with no writes to history. Two current-runtime early FP teachers were independently generated on the assigned UUIDs, without the 256 archived long-trace chunks. Exact FP-replay parity and every next-kernel post-QDQ consumption checks passed. Current physical mappings are independent of historical document-manifest GPU ordinal fields.\n\nModel metadata are freshly hashed; weight identity is inherited from the retained direction-experiment model-weight SHA manifest, not newly claimed to have rehashed all weight shards. SOURCE_AND_ASSET_AUDIT.json records existing source commits plus exact imported file hashes; dirty historical worktrees were not reset or edited.\n\nReproduce CPU math with test_proxy_math_v2.py; each measurement command is in launch_panels.py and PANEL_SCHEDULE_FREEZE.json. Reproduction must use a new non-conflicting root and count its GPU time separately. Statistical calculations use standard-library midranks, partial Spearman rank residuals and tie-excluded concordance, with built-in tests in finalize_surrogate.py.\n")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in str(p)
                   and p.suffix not in [".log"] and p.name not in ["SHA256SUMS.txt", "FINAL_STATUS.json"])
    write_text(ROOT / "SHA256SUMS.txt", "".join(f"{sha256(p)}  {p.relative_to(ROOT)}\n" for p in files))
    atomic_json(ROOT / "FINAL_STATUS.json", {"execution_status": "COMPLETE_WITH_DISCOVERY_LIMITATIONS", "scientific_decision": status,
                "completed_epoch": time.time(), "gpu_hours": ledger["total_gpu_hours_conservative"],
                "deliverables": 7, "score_rows": len(raw_rows), "formal_rotation_measurements": len(condition_results),
                "hash_manifest_sha256": sha256(ROOT / "SHA256SUMS.txt"), "old_attempts_unchanged": True})
    print(json.dumps({"event": "FINAL_COMPLETE", "scientific_decision": status, "gpu_hours": ledger["total_gpu_hours_conservative"]}), flush=True)


if __name__ == "__main__":
    main()
