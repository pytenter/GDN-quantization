#!/usr/bin/env python3
"""Validate, budget, and finalize the evidence-guided diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd


ROOT = Path("experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def ids(x):
    return ", ".join(x) if x else "none"


def main():
    paired = json.loads((ROOT / "analysis/paired_summary_extended.json").read_text(encoding="utf-8"))
    smoke = json.loads((ROOT / "analysis/smoke_run_summary.json").read_text(encoding="utf-8"))
    main_run = json.loads((ROOT / "analysis/main_run_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(ROOT / "sparse_metrics.parquet")
    groups = pd.read_csv(ROOT / "analysis/sparse_group_summary.csv")
    rank = pd.read_csv(ROOT / "analysis/sparse_layer_ranking.csv")

    # Verify and project the read-only formal H64 training archive recovered
    # after the first diagnosis was frozen.  No checkpoint load is required.
    provenance_root = ROOT / "training_provenance"
    original_manifest = provenance_root / "hashes/artifact_sha256.txt"
    manifest_expected = {}
    for line in original_manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        manifest_expected[rel] = digest
    provenance_checks = []
    for path in sorted(p for p in provenance_root.rglob("*") if p.is_file()):
        rel = path.relative_to(provenance_root).as_posix()
        if rel == "hashes/artifact_sha256.txt":
            expected = "b81de548f780e9aa28e7849b56b322a8d7fecb4a247c12278eb54961e0776cd8"
        elif rel == "hashes/inventory_provenance.json":
            expected = "880012efce65ec725c9013d98d4d1b8a5c92bd02820f6dcdd6a6abb80a366c91"
        else:
            expected = manifest_expected.get(rel)
        actual = sha256(path)
        provenance_checks.append({"path": rel, "bytes": path.stat().st_size, "expected_sha256": expected, "actual_sha256": actual, "pass": expected == actual})

    c5_train = json.loads((provenance_root / "analysis/C5_training_summary.json").read_text(encoding="utf-8"))
    c6_train = json.loads((provenance_root / "analysis/C6_training_summary.json").read_text(encoding="utf-8"))
    c5_val = json.loads((provenance_root / "analysis/C5_validation_summary.json").read_text(encoding="utf-8"))
    c6_val = json.loads((provenance_root / "analysis/C6_validation_summary.json").read_text(encoding="utf-8"))
    env = json.loads((provenance_root / "configs/environment.json").read_text(encoding="utf-8"))
    cross_rows = []
    for condition, summary in (("C5", c5_val), ("C6", c6_val)):
        for candidate in summary["candidates"]:
            v = candidate["validation"]
            cross_rows.append({
                "condition": condition,
                "update": candidate["update"],
                "target_exposures": candidate["target_exposures"],
                "selected": candidate["update"] == summary["selected_update"],
                "state_relative_mse": v["state"],
                "functional_relative_mse": v["functional"],
                "combined": v["combined"],
                "primary": candidate["primary"],
                "checkpoint_path": candidate["checkpoint_path"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
            })
    cross_path = ROOT / "analysis/h64_validation_cross_objective.csv"
    cross_tmp = cross_path.with_name(cross_path.name + ".tmp")
    pd.DataFrame(cross_rows).to_csv(cross_tmp, index=False)
    os.replace(cross_tmp, cross_path)

    selected_c5 = next(x for x in cross_rows if x["condition"] == "C5" and x["selected"])
    selected_c6 = next(x for x in cross_rows if x["condition"] == "C6" and x["selected"])
    c5_best_functional = min((x for x in cross_rows if x["condition"] == "C5"), key=lambda x: x["functional_relative_mse"])
    c6_best_state = min((x for x in cross_rows if x["condition"] == "C6"), key=lambda x: x["state_relative_mse"])
    provenance_audit = {
        "status": "PASS" if all(x["pass"] for x in provenance_checks) else "FAIL",
        "access_mode": "read_only_recovery_from_user_supplied_original_training_server",
        "original_experiment_path": "/root/autodl-tmp/GDN-formal-c5-c6-h64-v1/experiments/qwen_gdn/recurrent_training/QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1",
        "source_git_branch": env["git_branch"],
        "source_git_head_before_preregistration": env["git_head_before_preregistration"],
        "copied_file_checks": provenance_checks,
        "copied_file_count": len(provenance_checks),
        "candidate_rows": len(cross_rows),
        "candidate_matrix_complete": len(cross_rows) == 40 and c5_val["candidate_count"] == 20 and c6_val["candidate_count"] == 20,
        "formal_runs": {
            "C5": {"status": c5_train["status"], "updates": c5_train["total_updates"], "exposures": c5_train["total_target_exposures"], "retry_count": c5_train["retry_count"], "nan_inf": c5_train["NaN_Inf"], "memory_creep": c5_train["memory_creep"]},
            "C6": {"status": c6_train["status"], "updates": c6_train["total_updates"], "exposures": c6_train["total_target_exposures"], "retry_count": c6_train["retry_count"], "nan_inf": c6_train["NaN_Inf"], "memory_creep": c6_train["memory_creep"]},
        },
        "selected_cross_objective": {"C5": selected_c5, "C6": selected_c6},
        "selection_tradeoffs": {"C5_best_functional": c5_best_functional, "C6_best_state": c6_best_state},
        "limits": ["no explicit update-0 held-out validation row", "validation fidelity is not downstream correctness", "checkpoints were not copied or loaded"],
    }
    atomic_json(ROOT / "analysis/h64_training_provenance_audit.json", provenance_audit)

    errors = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*error.json"))
    expected_samples = {
        "aime26_08_seed1", "aime26_02_seed1", "aime26_14_seed1", "aime26_11_seed1",
        "aime26_13_seed1", "aime26_01_seed1", "aime26_10_seed1", "aime26_25_seed1",
    }
    expected_positions = {0, 7, 31, 127, 255, 511, 1023, 2047}
    expected_conditions = {"H", "C5", "C6"}
    validation = {
        "main_run_pass": main_run["status"] == "PASS",
        "no_error_json": not errors,
        "sample_identity": set(metrics.sample_id.unique()) == expected_samples,
        "condition_identity": set(metrics.condition.unique()) == expected_conditions,
        "position_identity": set(int(x) for x in metrics.token_index_consumed.unique()) == expected_positions,
        "raw_metric_rows": len(metrics) == 14016,
        "unit_summary_rows": len(pd.read_csv(ROOT / "analysis/sparse_unit_summary.csv")) == 192,
        "scope_layer_counts": True,
        "finite_relevant_values": True,
        "quantizer_and_patch_audit": True,
        "h64_training_provenance_identity": provenance_audit["status"] == "PASS",
        "h64_candidate_matrix_complete": provenance_audit["candidate_matrix_complete"],
        "canonical_runner_sha256": main_run["canonical_runner_sha256"],
        "errors": errors,
    }
    for (_s, _c, _p), g in metrics.groupby(["sample_id", "condition", "token_index_consumed"]):
        counts = g.groupby("scope").size().to_dict()
        if counts != {
            "linear_attn_out_proj_output_fp_relative": 24,
            "model_output": 1,
            "state_fp_relative_post_qdq_physical": 24,
            "state_local_qdq_native_basis": 24,
        }:
            validation["scope_layer_counts"] = False
    for scope, cols in {
        "model_output": ["kl_fp_to_condition", "delta_nll_fixed_fp_next_token", "margin_degradation"],
        "state_fp_relative_post_qdq_physical": ["absolute_l2", "relative_l2", "reference_l2", "max_abs_error", "mean_abs_error"],
        "state_local_qdq_native_basis": ["absolute_l2", "relative_l2", "reference_l2", "max_abs_error", "mean_abs_error"],
        "linear_attn_out_proj_output_fp_relative": ["absolute_l2", "relative_l2", "reference_l2", "max_abs_error", "mean_abs_error"],
    }.items():
        g = metrics[metrics.scope == scope]
        for col in cols:
            if not g[col].map(lambda x: math.isfinite(float(x))).all():
                validation["finite_relevant_values"] = False
    if len(main_run["units"]) != 32:
        validation["quantizer_and_patch_audit"] = False
    for unit in main_run["units"]:
        if unit["steps"] != 2048 or set(unit["capture_positions"]) != expected_positions:
            validation["quantizer_and_patch_audit"] = False
        if unit["condition"] == "FP":
            if unit["quantizer_calls"] != 0:
                validation["quantizer_and_patch_audit"] = False
        else:
            if unit["quantizer_calls"] != 2048 or len(unit["quantizer_layers"]) != 24:
                validation["quantizer_and_patch_audit"] = False
            patch = unit["patch_audit"]
            if patch["phase_counts"] != {"chunk_prefill": 24, "recurrent_decode": 49152}:
                validation["quantizer_and_patch_audit"] = False
    validation["all_required_gates"] = all(v for k, v in validation.items() if isinstance(v, bool))
    if not validation["all_required_gates"]:
        atomic_json(ROOT / "analysis/final_validation.json", validation)
        raise RuntimeError(f"final validation failed: {validation}")

    gpu_hours = float(smoke["gpu_hours"]) + float(main_run["gpu_hours"])
    budget = {
        "task": "QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1",
        "caps_gpu_hours": {"overall": 16.0, "smoke_compatibility": 2.0, "sparse_measurement": 8.0, "single_intervention": 6.0},
        "entries": [
            {"stage": "smoke_compatibility", "gpu_count": 1, "wall_seconds": smoke["process_wall_seconds"], "gpu_hours": smoke["gpu_hours"], "status": "PASS"},
            {"stage": "sparse_measurement", "gpu_count": 1, "wall_seconds": main_run["process_wall_seconds"], "gpu_hours": main_run["gpu_hours"], "status": "PASS"},
            {"stage": "targeted_intervention", "gpu_count": 0, "wall_seconds": 0.0, "gpu_hours": 0.0, "status": "NOT_RUN_EVIDENCE_INSUFFICIENT_FOR_SINGLE_CAUSAL_TARGET"},
        ],
        "gpu_hours_used": gpu_hours,
        "gpu_hours_reserved": 0.0,
        "within_all_caps": gpu_hours <= 16.0 and smoke["gpu_hours"] <= 2.0 and main_run["gpu_hours"] <= 8.0,
        "new_artifact_size_cap_bytes": 1073741824,
        "full_density_full60": "CANCELLED_BY_USER",
        "new_free_generation": "NOT_STARTED",
        "new_training": "NOT_STARTED",
    }
    atomic_json(ROOT / "budget_ledger.json", budget)
    atomic_json(ROOT / "analysis/final_validation.json", validation)

    c5h = paired["comparisons"]["C5_vs_H"]
    c6h = paired["comparisons"]["C6_vs_H"]
    c5fp = paired["comparisons"]["C5_vs_FP"]
    c6fp = paired["comparisons"]["C6_vs_FP"]
    c6c5 = paired["comparisons"]["C6_vs_C5"]

    def group(comp, cat):
        return groups[(groups.comparison == comp) & (groups.endpoint_category == cat)].iloc[0]

    c5r, c5s = group("C5_vs_H", "reference_only"), group("C5_vs_H", "candidate_only")
    c6r, c6s = group("C6_vs_H", "reference_only"), group("C6_vs_H", "candidate_only")
    prefix = "latest_position_median_delta_"
    sparse_table = [
        ("C5 regressions", int(c5r.sample_count), c5r),
        ("C5 rescues", int(c5s.sample_count), c5s),
        ("C6 regressions", int(c6r.sample_count), c6r),
        ("C6 rescues", int(c6s.sample_count), c6s),
    ]
    sparse_lines = []
    for label, n, r in sparse_table:
        sparse_lines.append(
            f"| {label} | {n} | {r[prefix+'state_drift_mean_relative_l2']:+.6f} | "
            f"{r[prefix+'local_qdq_mean_relative_l2']:+.6f} | {r[prefix+'function_mean_relative_l2']:+.6f} | "
            f"{r[prefix+'kl_fp_to_condition']:+.6g} | {r[prefix+'delta_nll_fixed_fp_next_token']:+.6g} |"
        )

    c5_reg_top = rank[(rank.comparison == "C5_vs_H") & (rank.endpoint_category == "reference_only")].sort_values("rank_descending_mean_delta").head(5)
    c5_res_top = rank[(rank.comparison == "C5_vs_H") & (rank.endpoint_category == "candidate_only")].sort_values("rank_descending_mean_delta").head(5)
    c6_reg_top = rank[(rank.comparison == "C6_vs_H") & (rank.endpoint_category == "reference_only")].sort_values("rank_descending_mean_delta").head(5)
    c6_res_top = rank[(rank.comparison == "C6_vs_H") & (rank.endpoint_category == "candidate_only")].sort_values("rank_descending_mean_delta").head(5)
    top = lambda x: ", ".join(str(int(v)) for v in x.layer)

    report = f"""# QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1 — Final report

## Executive conclusion

C5 and C6 do not fail because they never help: against Hadamard, C5 rescues 5 samples and C6 rescues 4.  They fail to win overall because those gains are outweighed by 13 and 9 regressions, respectively.  The FP gap is structurally different and larger: C5 loses 19 FP successes and C6 loses 16, while neither method rescues a single FP-wrong sample.  All eight FP-wrong samples are also wrong under H/C5/C6.

The new sparse persistent replay does **not** support one simple universal mechanism such as “learned rotations always have more state drift than Hadamard.”  By recurrent update 2047, C5 regression units have *lower* median FP-relative state drift than H, whereas C6 regression units have higher drift.  Local QDQ differences are near zero and do not separate regressions from rescues.  Function-space and logit effects are sample-specific; one C5 regression (`aime26_08_seed1`) has a large KL/NLL deterioration, but this does not reproduce across the other C5 regressions.  Evidence through 2048 updates therefore supports an **objective/selection mismatch and endpoint churn** more strongly than a single bad layer or uniformly worse quantization.

The most defensible next move is to make checkpoint selection inference-aligned on an independent non-AIME development set: preserve persistent writeback, compare against Hadamard pairwise, and select on downstream multi-step function/logit fidelity plus an explicit regression constraint.  A layer rollback intervention is not justified by this panel, so no intervention was run.

## 1. Formal endpoint evidence

| Condition | Frozen V4 correct |
|---|---:|
| FP | 52/60 |
| Native INT8-C128 | 18/60 |
| Hadamard | 41/60 |
| C5 | 33/60 |
| C6 | 36/60 |

| Comparison | Both correct | Reference only | Candidate only | Both wrong | Net | Question-cluster 95% CI of risk difference | Exact paired p | Holm p in the predeclared five-comparison family |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| C5 vs H | {c5h['both_correct']} | {c5h['reference_only']} | {c5h['candidate_only']} | {c5h['both_wrong']} | {c5h['net_candidate_minus_reference']} | {c5h['cluster_bootstrap_ci95']} | {c5h['exact_mcnemar_two_sided_p']:.6g} | {c5h['holm_p_across_primary_endpoint_family']:.6g} |
| C6 vs H | {c6h['both_correct']} | {c6h['reference_only']} | {c6h['candidate_only']} | {c6h['both_wrong']} | {c6h['net_candidate_minus_reference']} | {c6h['cluster_bootstrap_ci95']} | {c6h['exact_mcnemar_two_sided_p']:.6g} | {c6h['holm_p_across_primary_endpoint_family']:.6g} |
| C5 vs FP | {c5fp['both_correct']} | {c5fp['reference_only']} | {c5fp['candidate_only']} | {c5fp['both_wrong']} | {c5fp['net_candidate_minus_reference']} | {c5fp['cluster_bootstrap_ci95']} | {c5fp['exact_mcnemar_two_sided_p']:.6g} | {c5fp['holm_p_across_primary_endpoint_family']:.6g} |
| C6 vs FP | {c6fp['both_correct']} | {c6fp['reference_only']} | {c6fp['candidate_only']} | {c6fp['both_wrong']} | {c6fp['net_candidate_minus_reference']} | {c6fp['cluster_bootstrap_ci95']} | {c6fp['exact_mcnemar_two_sided_p']:.6g} | {c6fp['holm_p_across_primary_endpoint_family']:.6g} |
| C6 vs C5 (auxiliary) | {c6c5['both_correct']} | {c6c5['reference_only']} | {c6c5['candidate_only']} | {c6c5['both_wrong']} | {c6c5['net_candidate_minus_reference']} | {c6c5['cluster_bootstrap_ci95']} | {c6c5['exact_mcnemar_two_sided_p']:.6g} | not in family |

The earlier canonical report and its original three-comparison Holm result are preserved unchanged.  The table above is a new, explicitly defined family of five primary endpoint comparisons; it is not substituted back into the historical report.

### Exact learned-vs-H sample identities

- C5 regressions (H correct, C5 wrong): {ids(c5h['sample_ids']['reference_only'])}.
- C5 rescues (C5 correct, H wrong): {ids(c5h['sample_ids']['candidate_only'])}.
- C6 regressions (H correct, C6 wrong): {ids(c6h['sample_ids']['reference_only'])}.
- C6 rescues (C6 correct, H wrong): {ids(c6h['sample_ids']['candidate_only'])}.

The full paired table, including all six requested comparisons and every identity, is in `paired_outcomes.csv` and `analysis/paired_summary_extended.json`.

## 2. Behavioral association in existing free-generation outputs

This is descriptive, not causal.  Relative to H, the median C5 regression output is 821 tokens longer, with +0.0258 repeated-8gram fraction, +2 oscillations, +1 retraction, and +0.122 post-commit fraction.  C5 rescues move in the opposite direction (−5368 tokens, −0.0682 repetition, −3 oscillations, −3 retractions, −0.204 post-commit fraction).  C6 shows the same broad asymmetry: regressions are +10525 tokens and +0.163 post-commit fraction, while rescues are −6174 tokens and −0.090 post-commit fraction.  These fields do not alter Frozen V4 scores.

## 3. What the training audit establishes

C5 optimizes recurrent state reconstruction.  C6 optimizes full `linear_attn.out_proj` relative MSE plus `0.1 × state`.  Both use actual persistent student state and canonical QDQ/writeback; H64 detaches the graph but does not reset numerical state.  Both start independently from Hadamard, and C6 is not warm-started from C5.  Deployment reads frozen FP32 Delta matrices directly.

The original H64 training archive was subsequently recovered read-only from the user-supplied training server and hash-preserved under `training_provenance/`.  Both runs completed 125 updates / 1000 exposures without retry, NaN/Inf, model-sentinel change, or memory creep.  The shared 16-document / 128-capture validation panel recorded both state and functional metrics for all 20 candidates per condition.

| Selected checkpoint | State relative-MSE ↓ | Local out-proj relative-MSE ↓ |
|---|---:|---:|
| C5, update 63 / exposure 504 | **0.007916185** | 0.049509294 |
| C6, update 125 / exposure 1000 | 0.008251136 | **0.049016486** |

Selected C5 is 4.06% better than selected C6 on state error; selected C6 is 1.00% better than selected C5 on functional error.  This verifies objective-aligned specialization, but not a broad separation.  It also reveals a concrete selection tradeoff: C5's update-113 candidate reaches functional error 0.046988553 (5.09% below selected C5) at state error 0.008100550 (2.33% above selected C5), so the C5 state-primary rule discarded the functionally best saved candidate.  This strengthens the checkpoint-selection diagnosis without showing that update 113 would improve AIME.

The archive has no explicit update-0 held-out validation row, so neither selected checkpoint can be claimed to improve a shared metric over the exact Hadamard initializer from these records alone.  See `training_objective_audit.md`, `analysis/h64_validation_cross_objective.csv`, and `analysis/h64_training_provenance_audit.json`.

## 4. New sparse persistent-replay diagnostic

Protocol: eight endpoint-stratified seed1 samples, existing FP token paths only, independently persistent FP/H/C5/C6 caches, canonical QDQ after every decode output, eight fixed capture positions through update 2047.  No free generation occurred.  Each metric row is a repeated measurement, not an independent sample.

At the latest capture, the table reports learned minus H medians.  Negative is lower than H.  Rescue strata contain one selected sample each, so their medians are single observations.

| Endpoint stratum | n | State drift Δ | Local QDQ Δ | Layer-output Δ | Full-vocab KL Δ | Fixed-next-token ΔNLL Δ |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(sparse_lines)}

Key interpretation:

1. **Local QDQ is not the discriminator.** All latest-position group medians differ from H by only about `1e-4` or less.
2. **Cumulative state drift is not a universal explanation.** It is lower for C5 regressions but higher for C6 regressions; both selected rescue examples have lower drift.
3. **Downstream effects are heterogeneous.** C5 regression `aime26_08_seed1` has KL Δ `+0.2487` and fixed-token ΔNLL `+2.1701` versus H at update 2047, but the other C5 regression examples do not reproduce that direction.  C6 regression medians show KL Δ `+0.0560` while ΔNLL is mixed.
4. **No stable bad-layer target emerged.** The largest positive state-drift layers for C5 regressions are {top(c5_reg_top)}, but the C5 rescue's list also starts {top(c5_res_top)}.  C6 similarly places {top(c6_reg_top)} in regressions and {top(c6_res_top)} in its rescue.  This overlap is evidence against choosing a 1–2-layer rollback from this panel.
5. **Late history remains untested.** Nothing here establishes behavior after 2048 updates or explains all 81920-token trajectories.

## 5. Hypothesis verdicts

- **H1: uniformly larger learned recurrent drift causes regressions — weakened.** C6 regressions are directionally compatible; C5 regressions contradict a uniform version.
- **H2: downstream decision/trajectory mismatch matters beyond local QDQ — partially supported.** Existing behavior and sample-specific logit divergences support it, while the sparse panel does not yield one universal scalar separator.
- **H3: training/selection objective mismatch is a live design risk — strengthened, not proven causal.** The recovered shared validation table shows expected specialization but only a narrow selected C5/C6 separation, and C5's state-primary rule bypasses a saved checkpoint with 5.09% better functional fidelity.  Endpoint churn is still not captured by either condition's primary metric.

## 6. Why targeted intervention was not run

Neither a unique bad layer nor a reproducible single causal direction was identified.  D1 would select layers from AIME-internal outcomes with no independent confirmation; D2 would add a short branch even though the primary diagnostic already maintained full persistent state from the prompt through 2048 steps.  Spending the remaining budget would add researcher degrees of freedom without resolving the identified evidence gap.

`intervention_results.parquet` is intentionally absent; status is `NOT_RUN`, not missing work.

## 7. Evidence boundaries

- Formal endpoint facts come from Frozen V4 and remain authoritative.
- Behavior analysis is observational.
- Sparse replay is FP-path teacher forcing, not any condition's original free-generation path.
- Layer metrics contain upstream differences and are not contribution percentages.
- Lower state error or KL is not equated with correctness.
- Historical FP/Native/H replay is hypothesis support only and is not presented as C5/C6 evidence.

## 8. Final status

```text
EXISTING_EVIDENCE_ANALYSIS = COMPLETE
SPARSE_DIAGNOSTIC = COMPLETE
TARGETED_INTERVENTION = NOT_RUN
NEXT_OPTIMIZATION_DIRECTION = INDEPENDENT_NON_AIME_PERSISTENT_DOWNSTREAM_CHECKPOINT_SELECTION_WITH_PAIRED_HADAMARD_REGRESSION_CONTROL
GPU_HOURS_USED = {gpu_hours:.6f}
FULL_DENSITY_FULL60 = CANCELLED_BY_USER
NEW_FREE_GENERATION = NOT_STARTED
NEW_TRAINING = NOT_STARTED
```

All gates, hashes, budgets, and reproduction commands are recorded in the accompanying machine-readable artifacts and `REPRODUCE.md`.
"""
    atomic_text(ROOT / "FINAL_REPORT.md", report)

    next_plan = """# Next optimization plan

## Primary direction

Change one major factor: **checkpoint selection**, not architecture or BPTT length.

On a fixed, independent non-AIME development corpus, replay identical teacher-forced token paths with fully persistent cache and canonical C128 writeback.  For every candidate checkpoint, compare against Hadamard pairwise at predeclared early and late positions.  Select using a frozen downstream score built from full `linear_attn.out_proj` fidelity and fixed-next-token ΔNLL/margin, with an explicit cap on the number/magnitude of Hadamard regressions.  Do not select on C5-state and C6-functional primaries separately.

This recommendation follows from three observations: learned endpoint behavior has substantial rescue/regression churn; local QDQ differences do not explain that churn; and the recovered formal H64 validation matrix shows objective-aligned specialization but also a concrete C5 state-selection/function-fidelity tradeoff.  Neither retained primary metric is a direct downstream trajectory or regression-control objective.

## Minimal future experiment (not executed here)

1. Freeze an independently sourced non-AIME development set before training or checkpoint inspection.  If no suitable licensed set is already present, record that as a blocker rather than silently downloading or synthesizing one.
2. Preserve Qwen3.5-9B, INT8-C128, H128 initialization, persistent recurrent writeback, seed/determinism, and the frozen deployment path.
3. Keep Hadamard, current C5, and current C6 as controls.
4. Change only the checkpoint-selection rule described above.  Do not simultaneously change loss, parameterization, horizon, or learned layer mask.
5. Report paired rescue/regression identities as well as mean fidelity metrics.  A net score without regression accounting is insufficient.
6. After selection is frozen on non-AIME development data, AIME26 may be used only as a previously exposed diagnostic benchmark, not an untouched test set.

## Falsification rule

Reject this direction if the frozen downstream selection score does not rank Hadamard/current C5/current C6 in a way that predicts paired regressions on held-out non-AIME documents, or if selecting by it fails to reduce regression count without eliminating rescues.  Do not rescue the hypothesis by changing samples, capture positions, or score weights after seeing results.

## Backup direction

Only after validating the selection signal, test a single conservative Hadamard-anchor/trust-region strength chosen entirely on the independent development set.  Do not infer a layer mask from the present AIME panel: the same high-delta layers appeared in both regressions and rescues.

No new training or generation is authorized by this plan.
"""
    atomic_text(ROOT / "NEXT_OPTIMIZATION_PLAN.md", next_plan)

    reproduce = """# Reproduction

Run from `/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2` with `/data/ydai/miniconda3/envs/bitdecode/bin/python`.

```bash
# CPU-only synthesis of already frozen endpoint artifacts
/data/ydai/miniconda3/envs/bitdecode/bin/python experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/analyze_existing.py

# One-GPU compatibility smoke; re-running is not needed for the completed experiment
CUDA_VISIBLE_DEVICES=0 /data/ydai/miniconda3/envs/bitdecode/bin/python experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/sparse_probe.py --mode smoke

# One-GPU frozen sparse replay; do not run concurrently with the smoke
CUDA_VISIBLE_DEVICES=0 /data/ydai/miniconda3/envs/bitdecode/bin/python experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/sparse_probe.py --mode main

# CPU-only sparse analysis and final validation/reporting
/data/ydai/miniconda3/envs/bitdecode/bin/python experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/analyze_sparse.py
/data/ydai/miniconda3/envs/bitdecode/bin/python experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/finalize_diagnosis.py
```

The completed run must not be re-executed merely to reproduce tables: verify `SHA256SUMS` and read the frozen parquet/CSV/JSON artifacts.  Use `git rev-parse HEAD` to identify the delivery commit containing this report.

The recovered H64 training provenance is a read-only snapshot under `training_provenance/`.  Verify copied-file identity against `analysis/h64_training_provenance_audit.json`; `analysis/h64_validation_cross_objective.csv` is a deterministic projection of the two preserved validation summaries.  No checkpoint or GPU execution is required to reproduce that audit.
"""
    atomic_text(ROOT / "REPRODUCE.md", reproduce)

    # Size is checked after final reports but before SHA256SUMS itself.
    total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    if total > budget["new_artifact_size_cap_bytes"]:
        raise RuntimeError(f"artifact cap exceeded: {total}")
    validation["artifact_bytes_before_sha_manifest"] = total
    validation["artifact_cap_pass"] = True
    atomic_json(ROOT / "analysis/final_validation.json", validation)

    lines = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    atomic_text(ROOT / "SHA256SUMS", "\n".join(lines) + "\n")
    print(json.dumps({"status": "COMPLETE", "gpu_hours_used": gpu_hours, "artifact_bytes": total, "hashed_files": len(lines)}, sort_keys=True))


if __name__ == "__main__":
    main()
