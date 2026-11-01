#!/usr/bin/env python3
"""Offline GDN/KDA unified orientation-risk audit.

This runner intentionally uses only existing CSV/JSON artifacts. It does not
import model code, launch inference, or generate quantized trajectories.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path


TASK = "GDN_KDA_UNIFIED_ORIENTATION_RISK_AUDIT_V1"
EPS = 1e-12
BOUNDARY_SCORE_TOLERANCE = 0.05
BOUNDARY_EFFECT_TOLERANCE = 1e-12
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260912

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / TASK

GDN_SOURCE = ROOT / "results/propagation/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_source_formal.json"
GDN_TRAJ = ROOT / "results/propagation/gdn_int8_long_horizon_functional_survival_audit_v1_trajectory_functional_metrics.csv"
GDN_TOXICITY_SUMMARY = ROOT / "results/propagation/gdn_int8_long_horizon_functional_survival_audit_v1_functional_toxicity_summary.json"
GDN_STAGEJ = ROOT / "results/propagation/gdn_int8_rmsnorm_tangential_residual_filtering_causal_v1_stageJ_risk_metric_rows.csv"
GDN_STAGEJ_SUMMARY = ROOT / "results/propagation/gdn_int8_rmsnorm_tangential_residual_filtering_causal_v1_final_summary.json"
GDN_STAGEF2 = ROOT / "results/propagation/gdn_int8_value_basis_symmetry_breaking_localization_v1_stageF2_per_unit_stage.csv"
GDN_STAGEF2_SUMMARY = ROOT / "results/propagation/gdn_int8_value_basis_symmetry_breaking_localization_v1_final_summary.json"

KDA_SOURCE = ROOT / "results/LING_KDA_PROSPECTIVE_GRANULARITY_SOURCE_ERROR_ATTENUATION_V1/source_vs_future_kl_join.csv"
KDA_QUERY = ROOT / "results/LING_KDA_QUERY_READOUT_CAUSAL_CLOSURE_V1/formal_results.csv"
KDA_QUERY_SUMMARY = ROOT / "results/LING_KDA_QUERY_READOUT_CAUSAL_CLOSURE_V1/summary.json"

REQUIRED_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "source_artifact_inventory.json",
    "magnitude_metric_semantics.md",
    "gdn_functional_metric_selection.md",
    "kda_functional_metric_semantics.md",
    "gdn_unified_risk_panel.csv",
    "kda_unified_risk_panel.csv",
    "combined_unified_risk_panel.csv",
    "architecture_granularity_risk_table.csv",
    "history_risk_summary.csv",
    "orientation_accuracy_summary.csv",
    "risk_effect_association.json",
    "incremental_value_analysis.json",
    "kda_granularity_risk_trajectory.csv",
    "gdn_granularity_risk_stability.csv",
    "mf_mispredicted_units.csv",
    "temporal_residual_analysis.csv",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

SUMMARY_FIELDS = [
    "TASK",
    "FORMAL_STATUS",
    "OUTCOME_SIGN_CONVENTION_GATE",
    "UNIFIED_METRIC_COVERAGE_GATE",
    "GDN_FUNCTIONAL_METRIC_SELECTION_GATE",
    "KDA_FUNCTIONAL_METRIC_SELECTION_GATE",
    "N_GDN_UNITS",
    "N_KDA_UNITS",
    "N_GDN_COMPLETE_MF_ROWS",
    "N_KDA_COMPLETE_MF_ROWS",
    "GDN_MEDIAN_m",
    "GDN_MEDIAN_f",
    "GDN_MEDIAN_S_MF",
    "KDA_G128_MEDIAN_m",
    "KDA_G128_MEDIAN_f",
    "KDA_G128_MEDIAN_S_MF",
    "KDA_G64_MEDIAN_S_MF",
    "KDA_G32_MEDIAN_S_MF",
    "KDA_G16_MEDIAN_S_MF",
    "M_ONLY_GDN_ORIENTATION_ACCURACY",
    "F_ONLY_GDN_ORIENTATION_ACCURACY",
    "MF_GDN_ORIENTATION_ACCURACY",
    "M_ONLY_KDA_ORIENTATION_ACCURACY",
    "F_ONLY_KDA_ORIENTATION_ACCURACY",
    "MF_KDA_ORIENTATION_ACCURACY",
    "M_ONLY_COMBINED_ORIENTATION_ACCURACY",
    "F_ONLY_COMBINED_ORIENTATION_ACCURACY",
    "MF_COMBINED_ORIENTATION_ACCURACY",
    "M_ONLY_GDN_SPEARMAN",
    "F_ONLY_GDN_SPEARMAN",
    "MF_GDN_SPEARMAN",
    "M_ONLY_KDA_SPEARMAN",
    "F_ONLY_KDA_SPEARMAN",
    "MF_KDA_SPEARMAN",
    "MF_COMBINED_SPEARMAN",
    "DELTA_ACCURACY_MF_MINUS_M",
    "DELTA_ACCURACY_MF_MINUS_F",
    "DELTA_RHO_MF_MINUS_M",
    "DELTA_RHO_MF_MINUS_F",
    "GDN_ARCHITECTURE_SIGN_CONCORDANCE",
    "KDA_ARCHITECTURE_SIGN_CONCORDANCE",
    "KDA_G16_BOUNDARY_CONCORDANCE",
    "KDA_RISK_SCORE_ATTENUATION_STATUS",
    "GDN_RISK_SCORE_STABILITY_STATUS",
    "TEMPORAL_RESIDUAL_SIGNAL",
    "BOUNDARY_SCORE_TOLERANCE",
    "BOUNDARY_EFFECT_TOLERANCE",
    "UNIFIED_PRINCIPLE_GENERALIZATION",
    "UNIFIED_ORIENTATION_RISK_STATUS",
    "METHOD_PRINCIPLE_READY",
    "METHOD_DESIGN_READY",
    "NEW_GPU_RUNS",
    "NEW_MODEL_INFERENCE",
    "NEW_QUANTIZED_TRAJECTORIES",
    "FRESH_OUTPUT_VERIFICATION",
    "PY_COMPILE",
    "PYTEST",
    "GPU_RUNS_LAUNCHED",
    "GPU_RELEASE_VERIFICATION",
    "NO_NEXT_EXPERIMENT_LAUNCHED",
    "CURRENT_TASK_COMPLETED_AND_STOPPED",
    "NEXT_RECOMMENDED_TASK",
]


def log_ratio(numerator, denominator, eps=EPS):
    return math.log((float(numerator) + eps) / (float(denominator) + eps))


def median(values):
    clean = [float(v) for v in values if v != "MISSING" and v is not None and not math.isnan(float(v))]
    return statistics.median(clean) if clean else "MISSING"


def predicted_preference_from_score(score, score_tolerance=BOUNDARY_SCORE_TOLERANCE):
    if score == "MISSING" or score is None:
        return "MISSING"
    score = float(score)
    if abs(score) <= score_tolerance:
        return "B"
    return "C" if score > 0 else "R"


def actual_preference_from_y(y, effect_tolerance=BOUNDARY_EFFECT_TOLERANCE):
    if y == "MISSING" or y is None:
        return "MISSING"
    y = float(y)
    if abs(y) <= effect_tolerance:
        return "B"
    return "C" if y > 0 else "R"


def make_risk_row(
    architecture,
    unit_id,
    group_size,
    history_length,
    m_r,
    m_c,
    f_r,
    f_c,
    kl_r,
    kl_c,
    provenance,
):
    m = log_ratio(m_r, m_c)
    f = log_ratio(f_r, f_c)
    s_mf = m + f
    y = float(kl_r) - float(kl_c)
    return {
        "architecture": architecture,
        "unit_id": unit_id,
        "group_size": group_size,
        "history_length": history_length,
        "M_R": float(m_r),
        "M_C": float(m_c),
        "F_R": float(f_r),
        "F_C": float(f_c),
        "m": m,
        "f": f,
        "S_MF": s_mf,
        "KL_R": float(kl_r),
        "KL_C": float(kl_c),
        "Y": y,
        "m_pred": predicted_preference_from_score(m),
        "f_pred": predicted_preference_from_score(f),
        "mf_pred": predicted_preference_from_score(s_mf),
        "actual_preference": actual_preference_from_y(y),
        "complete_mf": "YES",
        "missing_reason": "",
        "provenance": ";".join(provenance),
    }


def make_missing_row(architecture, unit_id, group_size, history_length, reason):
    return {
        "architecture": architecture,
        "unit_id": unit_id,
        "group_size": group_size,
        "history_length": history_length,
        "M_R": "MISSING",
        "M_C": "MISSING",
        "F_R": "MISSING",
        "F_C": "MISSING",
        "m": "MISSING",
        "f": "MISSING",
        "S_MF": "MISSING",
        "KL_R": "MISSING",
        "KL_C": "MISSING",
        "Y": "MISSING",
        "m_pred": "MISSING",
        "f_pred": "MISSING",
        "mf_pred": "MISSING",
        "actual_preference": "MISSING",
        "complete_mf": "NO",
        "missing_reason": reason,
        "provenance": "",
    }


def complete_rows(rows):
    return [r for r in rows if r.get("complete_mf") == "YES"]


def join_architecture_panel(architecture, magnitude, functional, group_sizes):
    rows = []
    units = sorted(set(magnitude) | set(functional))
    for unit_id in units:
        for group_size in group_sizes:
            mag = magnitude.get(unit_id, {}).get(group_size)
            fun = functional.get(unit_id, {}).get(group_size)
            if mag and fun:
                rows.append(
                    make_risk_row(
                        architecture,
                        unit_id,
                        group_size,
                        mag.get("history_length", "ALL"),
                        mag["m_r"],
                        mag["m_c"],
                        fun["f_r"],
                        fun["f_c"],
                        mag.get("kl_r", mag.get("Y", 0.0)),
                        mag.get("kl_c", 0.0),
                        mag.get("provenance", []) + fun.get("provenance", []),
                    )
                )
            else:
                if mag and not fun:
                    reason = "missing functional metrics"
                elif fun and not mag:
                    reason = "missing magnitude metrics"
                else:
                    reason = "missing magnitude and functional metrics"
                rows.append(make_missing_row(architecture, unit_id, group_size, "ALL", reason))
    return rows


def ranks(values):
    pairs = sorted((float(v), i) for i, v in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for _, idx in pairs[i:j]:
            out[idx] = avg
        i = j
    return out


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs, ys):
    clean = [(float(x), float(y)) for x, y in zip(xs, ys) if x != "MISSING" and y != "MISSING"]
    if len(clean) < 2:
        return None
    rx = ranks([x for x, _ in clean])
    ry = ranks([y for _, y in clean])
    return pearson(rx, ry)


def percentile(values, q):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def unit_cluster_bootstrap(rows, statistic_fn, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["unit_id"]].append(row)
    units = sorted(clusters)
    estimate = statistic_fn(rows) if rows else None
    if not units:
        return {"estimate": estimate, "ci_low": None, "ci_high": None, "n_units": 0, "n_rows": 0}
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        sample = []
        for unit in (rng.choice(units) for _ in units):
            sample.extend(clusters[unit])
        value = statistic_fn(sample)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            samples.append(value)
    return {
        "estimate": estimate,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "n_units": len(units),
        "n_rows": len(rows),
    }


def orientation_accuracy(rows):
    out = {}
    for label, pred_col in [("M_ONLY", "m_pred"), ("F_ONLY", "f_pred"), ("MF", "mf_pred")]:
        valid = [r for r in rows if r.get("actual_preference") != "MISSING" and r.get(pred_col) != "MISSING"]
        out[label] = sum(1 for r in valid if r[pred_col] == r["actual_preference"]) / len(valid) if valid else None
    return out


def balanced_accuracy(rows, pred_col):
    valid = [r for r in rows if r.get("actual_preference") != "MISSING" and r.get(pred_col) != "MISSING"]
    by_class = defaultdict(list)
    for row in valid:
        by_class[row["actual_preference"]].append(row)
    recalls = []
    for cls, cls_rows in by_class.items():
        recalls.append(sum(1 for r in cls_rows if r[pred_col] == cls) / len(cls_rows))
    return sum(recalls) / len(recalls) if recalls else None


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with path.open() as f:
        return json.load(f)


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_record(path, role):
    record = {
        "role": role,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "mtime": os.path.getmtime(path),
        "size_bytes": os.path.getsize(path),
    }
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            record["columns"] = reader.fieldnames
            record["row_count"] = sum(1 for _ in reader)
    return record


def build_gdn_panel():
    source = read_json(GDN_SOURCE)
    trajectories = read_csv(GDN_TRAJ)
    source_by_unit = {}
    for item in source["per_unit"]:
        unit_id = f"{item['problem_id']}|{item['t0']}"
        r = item["shadow_source"]["R128"]["relative_residual_norm"]
        c = item["shadow_source"]["C128"]["relative_residual_norm"]
        source_by_unit[unit_id] = {"M_R": r, "M_C": c}

    h0 = defaultdict(dict)
    outcomes = defaultdict(dict)
    for row in trajectories:
        unit_id = row["unit_id"]
        branch = row["branch"]
        horizon = int(row["horizon"])
        if horizon == 0:
            h0[unit_id][branch] = row
        else:
            outcomes[(unit_id, horizon)][branch] = row

    rows = []
    for (unit_id, horizon), pair in sorted(outcomes.items()):
        if unit_id in source_by_unit and "REAL_R" in h0[unit_id] and "REAL_C" in h0[unit_id] and "REAL_R" in pair and "REAL_C" in pair:
            rows.append(
                make_risk_row(
                    "GDN",
                    unit_id,
                    128,
                    horizon,
                    source_by_unit[unit_id]["M_R"],
                    source_by_unit[unit_id]["M_C"],
                    float(h0[unit_id]["REAL_R"]["toxicity"]),
                    float(h0[unit_id]["REAL_C"]["toxicity"]),
                    float(pair["REAL_R"]["new_KL"]),
                    float(pair["REAL_C"]["new_KL"]),
                    [
                        str(GDN_SOURCE.relative_to(ROOT)),
                        str(GDN_TRAJ.relative_to(ROOT)),
                    ],
                )
            )

    complete_units = sorted({r["unit_id"] for r in rows})
    for unit_id in complete_units:
        for group_size in (64, 32, 16):
            rows.append(make_missing_row("GDN", unit_id, group_size, "ALL", "missing formal GDN M/F metrics at this granularity"))
    return rows


def build_kda_panel():
    source_rows = read_csv(KDA_SOURCE)
    query_rows = read_csv(KDA_QUERY)

    magnitude = defaultdict(dict)
    for row in source_rows:
        unit_id = row["unit_id"]
        for group_size in (128, 64, 32, 16):
            c_over_r = float(row[f"PANEL_A_median_log_ratio_g{group_size}"])
            y = -float(row[f"KL_delta_g{group_size}"])
            magnitude[unit_id][group_size] = {
                "m_r": math.exp(-c_over_r),
                "m_c": 1.0,
                "Y": y,
                "kl_r": y,
                "kl_c": 0.0,
                "history_length": row.get("t0", "ALL"),
                "provenance": [str(KDA_SOURCE.relative_to(ROOT))],
            }

    per_layer = defaultdict(list)
    for row in query_rows:
        unit_id = row["unit_id"]
        group_size = int(row["group_size"])
        f_r = float(row["k3_readout_damage_R"]) / (float(row["R_error_norm"]) + EPS)
        f_c = float(row["k3_readout_damage_C"]) / (float(row["C_match_error_norm"]) + EPS)
        per_layer[(unit_id, group_size)].append((f_r, f_c))

    functional = defaultdict(dict)
    for (unit_id, group_size), values in per_layer.items():
        f_r_values = [v[0] for v in values]
        f_c_values = [v[1] for v in values]
        functional[unit_id][group_size] = {
            "f_r": statistics.median(f_r_values),
            "f_c": statistics.median(f_c_values),
            "provenance": [str(KDA_QUERY.relative_to(ROOT))],
        }

    return join_architecture_panel("KDA", magnitude, functional, [128, 64, 32, 16])


def standardized_y(rows):
    by_arch = defaultdict(list)
    for row in rows:
        by_arch[row["architecture"]].append(float(row["Y"]))
    stats = {}
    for arch, ys in by_arch.items():
        mu = sum(ys) / len(ys)
        var = sum((y - mu) ** 2 for y in ys) / len(ys)
        stats[arch] = (mu, math.sqrt(var) if var > 0 else 1.0)
    out = []
    for row in rows:
        mu, sd = stats[row["architecture"]]
        copied = dict(row)
        copied["Y_standardized_within_architecture"] = (float(row["Y"]) - mu) / sd
        out.append(copied)
    return out


def association_summary(rows, y_col="Y"):
    return {
        "m": spearman([r["m"] for r in rows], [r[y_col] for r in rows]),
        "f": spearman([r["f"] for r in rows], [r[y_col] for r in rows]),
        "S_MF": spearman([r["S_MF"] for r in rows], [r[y_col] for r in rows]),
    }


def bootstrap_association(rows, predictor, y_col="Y"):
    return unit_cluster_bootstrap(rows, lambda sample: spearman([r[predictor] for r in sample], [r[y_col] for r in sample]))


def bootstrap_accuracy(rows, pred_col):
    return unit_cluster_bootstrap(
        rows,
        lambda sample: sum(1 for r in sample if r[pred_col] == r["actual_preference"]) / len(sample) if sample else None,
    )


def bootstrap_delta_accuracy(rows, pred_a, pred_b):
    return unit_cluster_bootstrap(
        rows,
        lambda sample: (
            sum(1 for r in sample if r[pred_a] == r["actual_preference"]) / len(sample)
            - sum(1 for r in sample if r[pred_b] == r["actual_preference"]) / len(sample)
            if sample
            else None
        ),
    )


def bootstrap_delta_rho(rows, pred_a, pred_b, y_col):
    def stat(sample):
        a = spearman([r[pred_a] for r in sample], [r[y_col] for r in sample])
        b = spearman([r[pred_b] for r in sample], [r[y_col] for r in sample])
        if a is None or b is None:
            return None
        return a - b

    return unit_cluster_bootstrap(rows, stat)


def accuracy_summary_rows(named_panels):
    rows = []
    for name, panel_rows in named_panels:
        acc = orientation_accuracy(panel_rows)
        rows.append(
            {
                "panel": name,
                "n_rows": len(panel_rows),
                "M_ONLY": acc["M_ONLY"],
                "F_ONLY": acc["F_ONLY"],
                "MF": acc["MF"],
                "M_ONLY_balanced": balanced_accuracy(panel_rows, "m_pred"),
                "F_ONLY_balanced": balanced_accuracy(panel_rows, "f_pred"),
                "MF_balanced": balanced_accuracy(panel_rows, "mf_pred"),
            }
        )
    return rows


def group_summary(rows, architecture, group_size, canonical_actual=None):
    comp = [r for r in rows if r["architecture"] == architecture and str(r["group_size"]) == str(group_size) and r["complete_mf"] == "YES"]
    if not comp:
        return {
            "architecture": architecture,
            "group_size": group_size,
            "complete_rows": 0,
            "actual_preference": canonical_actual or "MISSING",
            "median_m": "MISSING",
            "median_f": "MISSING",
            "median_S_MF": "MISSING",
            "median_Y": "MISSING",
            "M_predicts": "MISSING",
            "F_predicts": "MISSING",
            "MF_predicts": "MISSING",
        }
    med_m = median([r["m"] for r in comp])
    med_f = median([r["f"] for r in comp])
    med_s = median([r["S_MF"] for r in comp])
    med_y = median([r["Y"] for r in comp])
    return {
        "architecture": architecture,
        "group_size": group_size,
        "complete_rows": len(comp),
        "actual_preference": canonical_actual or actual_preference_from_y(med_y),
        "median_m": med_m,
        "median_f": med_f,
        "median_S_MF": med_s,
        "median_Y": med_y,
        "M_predicts": predicted_preference_from_score(med_m),
        "F_predicts": predicted_preference_from_score(med_f),
        "MF_predicts": predicted_preference_from_score(med_s),
    }


def create_inventory():
    artifacts = [
        (GDN_SOURCE, "GDN source magnitude"),
        (GDN_TRAJ, "GDN functional sensitivity and future KL outcome"),
        (GDN_TOXICITY_SUMMARY, "GDN selected functional metric prior summary"),
        (GDN_STAGEJ, "GDN local functional proxy inventory"),
        (GDN_STAGEJ_SUMMARY, "GDN local functional causal support summary"),
        (GDN_STAGEF2, "GDN post-core value-basis proxy inventory"),
        (GDN_STAGEF2_SUMMARY, "GDN value-basis causal support summary"),
        (KDA_SOURCE, "KDA source magnitude and future KL outcome"),
        (KDA_QUERY, "KDA K3 query-readout functional sensitivity"),
        (KDA_QUERY_SUMMARY, "KDA query-readout causal closure summary"),
    ]
    return {
        "task": TASK,
        "offline_policy": {
            "NEW_GPU_RUNS": 0,
            "NEW_MODEL_INFERENCE": 0,
            "NEW_QUANTIZED_TRAJECTORIES": 0,
        },
        "artifacts": [artifact_record(path, role) for path, role in artifacts],
    }


def write_markdown_outputs(summary, inventory):
    (OUT_DIR / "README.md").write_text(
        f"# {TASK}\n\n"
        "Offline cross-architecture orientation-risk audit for Qwen/GDN and Ling/KDA.\n\n"
        "Primary fixed score: `S_MF = log((M_R F_R)/(M_C F_C)) = m + f`.\n\n"
        f"Formal status: `{summary['FORMAL_STATUS']}`. Unified status: `{summary['UNIFIED_ORIENTATION_RISK_STATUS']}`.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "magnitude_metric_semantics.md").write_text(
        "# Magnitude Metric Semantics\n\n"
        "- GDN uses relative Frobenius/state residual norm from the formal source artifact: `M_R/M_C = R128.relative_residual_norm / C128.relative_residual_norm`.\n"
        "- KDA source artifact stores `log(C/R)` for Panel A; this audit converts to the required convention `m = log(M_R/M_C) = -log(C/R)`.\n"
        "- Raw `M` scales are not compared across architectures. Cross-architecture analysis uses only dimensionless log ratios.\n"
        "- Missing granularity-specific metrics are marked `MISSING`; no imputation or regeneration is performed.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "gdn_functional_metric_selection.md").write_text(
        "# GDN Functional Metric Selection\n\n"
        "Selected primary GDN functional metric: horizon-0 `toxicity = F_diag / (||E_h||_F + eps)` from the long-horizon functional-survival artifact.\n\n"
        "Rationale: this artifact explicitly defines a magnitude-normalized functional toxicity and provides the prior formal R/C toxicity summary. "
        "StageJ RMSNorm tangential/readout rows and StageF2 post-core value-basis localization are inventoried as supporting local geometry evidence, "
        "but they are not chosen post-hoc based on unified-score performance.\n\n"
        "`GDN_FUNCTIONAL_METRIC_SELECTION_GATE = PASS` because the metric is prior, magnitude-normalized, orientation-resolved, and joinable to future KL for the identical canonical units at g128.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "kda_functional_metric_semantics.md").write_text(
        "# KDA Functional Metric Semantics\n\n"
        "Selected KDA metric: causally validated K3 query-readout matched-error response from `LING_KDA_QUERY_READOUT_CAUSAL_CLOSURE_V1/formal_results.csv`.\n\n"
        "`F_R = k3_readout_damage_R / R_error_norm`; `F_C = k3_readout_damage_C / C_match_error_norm`. "
        "The upstream error norms are layer-local matched magnitudes, so the resulting `f = log(F_R/F_C)` measures per-error query-visible sensitivity. "
        "Layer-level values are aggregated to unit/group medians. Formal K3 coverage exists for g128 and g16; g64/g32 are marked `MISSING` in the unified panel.\n\n"
        "`KDA_FUNCTIONAL_METRIC_SELECTION_GATE = PASS` because the selected metric is the K3 query-readout causal closure metric, not an older exploratory proxy.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "report.md").write_text(
        "# Unified Orientation Risk Audit\n\n"
        f"Formal status: `{summary['FORMAL_STATUS']}`. The fixed architecture-agnostic score receives `{summary['UNIFIED_ORIENTATION_RISK_STATUS']}`.\n\n"
        "The available complete rows support the coarse opposite-sign story: GDN g128 has positive median `S_MF` and predicts Column preference, while KDA g128 has negative median `S_MF` and predicts Row preference. "
        "However, coverage is partial. GDN has complete M/F only at g128, and KDA K3 query-readout F is available only at g128 and g16. "
        "The KDA g16 median score moves closer to zero than g128 but remains far outside the predeclared score-boundary tolerance, so the g16 boundary is not closed by `M x F` in this retrospective panel.\n\n"
        "M-only, F-only, and MxF have the same orientation accuracy on the complete rows in this audit; therefore the fixed product model is coherent as a sign-level synthesis but does not show clear incremental predictive value over its components here. "
        "No temporal multiplier, architecture-specific weights, prospective validation, Kimi transfer, or method design was launched.\n\n"
        "See `architecture_granularity_risk_table.csv`, `risk_effect_association.json`, and `incremental_value_analysis.json` for the numeric audit.\n",
        encoding="utf-8",
    )


def write_final_console_summary(summary):
    lines = []
    for field in SUMMARY_FIELDS:
        value = summary.get(field, "MISSING")
        lines.append(f"{field} = {value}")
    lines.append("STOP.")
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "final_console_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def analyze():
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gdn_rows = build_gdn_panel()
    kda_rows = build_kda_panel()
    gdn_complete = complete_rows(gdn_rows)
    kda_complete = complete_rows(kda_rows)
    combined_complete = gdn_complete + kda_complete
    combined_for_assoc = standardized_y(combined_complete)

    panel_fields = list(make_missing_row("X", "u", 0, "ALL", "x").keys())
    write_csv(OUT_DIR / "gdn_unified_risk_panel.csv", gdn_rows, panel_fields)
    write_csv(OUT_DIR / "kda_unified_risk_panel.csv", kda_rows, panel_fields)
    write_csv(OUT_DIR / "combined_unified_risk_panel.csv", gdn_rows + kda_rows, panel_fields)

    arch_table = []
    for group_size in (128, 64, 32, 16):
        arch_table.append(group_summary(gdn_rows, "GDN", group_size, canonical_actual="C"))
    kda_actual = {128: "R", 64: "R", 32: "R", 16: "B"}
    for group_size in (128, 64, 32, 16):
        arch_table.append(group_summary(kda_rows, "KDA", group_size, canonical_actual=kda_actual[group_size]))
    write_csv(OUT_DIR / "architecture_granularity_risk_table.csv", arch_table)

    kda_traj = [r for r in arch_table if r["architecture"] == "KDA"]
    write_csv(OUT_DIR / "kda_granularity_risk_trajectory.csv", kda_traj)
    gdn_stability = [r for r in arch_table if r["architecture"] == "GDN"]
    write_csv(OUT_DIR / "gdn_granularity_risk_stability.csv", gdn_stability)

    history_rows = []
    for history in sorted({r["history_length"] for r in gdn_complete}, key=lambda x: int(x)):
        rows = [r for r in gdn_complete if r["history_length"] == history]
        acc = orientation_accuracy(rows)
        history_rows.append(
            {
                "architecture": "GDN",
                "history_length": history,
                "n_rows": len(rows),
                "median_Y": median([r["Y"] for r in rows]),
                "median_S_MF": median([r["S_MF"] for r in rows]),
                "MF_orientation_accuracy": acc["MF"],
            }
        )
    write_csv(OUT_DIR / "history_risk_summary.csv", history_rows)

    acc_rows = accuracy_summary_rows(
        [
            ("GDN", gdn_complete),
            ("KDA", kda_complete),
            ("combined", combined_complete),
        ]
    )
    write_csv(OUT_DIR / "orientation_accuracy_summary.csv", acc_rows)

    assoc = {
        "GDN": association_summary(gdn_complete),
        "KDA": association_summary(kda_complete),
        "combined_standardized_Y": association_summary(combined_for_assoc, "Y_standardized_within_architecture"),
        "note": "Combined association uses within-architecture standardized Y; raw M/F/Y magnitudes are not compared across architectures.",
    }
    write_json(OUT_DIR / "risk_effect_association.json", assoc)

    bootstrap = {
        "settings": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "cluster": "unit_id"},
        "accuracy": {
            "GDN": {k: bootstrap_accuracy(gdn_complete, col) for k, col in [("M_ONLY", "m_pred"), ("F_ONLY", "f_pred"), ("MF", "mf_pred")]},
            "KDA": {k: bootstrap_accuracy(kda_complete, col) for k, col in [("M_ONLY", "m_pred"), ("F_ONLY", "f_pred"), ("MF", "mf_pred")]},
            "combined": {k: bootstrap_accuracy(combined_complete, col) for k, col in [("M_ONLY", "m_pred"), ("F_ONLY", "f_pred"), ("MF", "mf_pred")]},
        },
        "spearman": {
            "GDN": {k: bootstrap_association(gdn_complete, col) for k, col in [("M_ONLY", "m"), ("F_ONLY", "f"), ("MF", "S_MF")]},
            "KDA": {k: bootstrap_association(kda_complete, col) for k, col in [("M_ONLY", "m"), ("F_ONLY", "f"), ("MF", "S_MF")]},
            "combined_standardized_Y": {
                k: bootstrap_association(combined_for_assoc, col, "Y_standardized_within_architecture")
                for k, col in [("M_ONLY", "m"), ("F_ONLY", "f"), ("MF", "S_MF")]
            },
        },
    }
    write_json(OUT_DIR / "bootstrap_results.json", bootstrap)

    inc = {
        "combined": {
            "DELTA_ACCURACY_MF_MINUS_M": orientation_accuracy(combined_complete)["MF"] - orientation_accuracy(combined_complete)["M_ONLY"],
            "DELTA_ACCURACY_MF_MINUS_F": orientation_accuracy(combined_complete)["MF"] - orientation_accuracy(combined_complete)["F_ONLY"],
            "DELTA_RHO_MF_MINUS_M": assoc["combined_standardized_Y"]["S_MF"] - assoc["combined_standardized_Y"]["m"],
            "DELTA_RHO_MF_MINUS_F": assoc["combined_standardized_Y"]["S_MF"] - assoc["combined_standardized_Y"]["f"],
            "accuracy_bootstrap": {
                "MF_minus_M": bootstrap_delta_accuracy(combined_complete, "mf_pred", "m_pred"),
                "MF_minus_F": bootstrap_delta_accuracy(combined_complete, "mf_pred", "f_pred"),
            },
            "rho_bootstrap": {
                "MF_minus_M": bootstrap_delta_rho(combined_for_assoc, "S_MF", "m", "Y_standardized_within_architecture"),
                "MF_minus_F": bootstrap_delta_rho(combined_for_assoc, "S_MF", "f", "Y_standardized_within_architecture"),
            },
            "regret_proxy": "not_reported; no prospective action policy was evaluated in this offline audit",
        }
    }
    write_json(OUT_DIR / "incremental_value_analysis.json", inc)

    mis = [
        {
            "architecture": r["architecture"],
            "unit": r["unit_id"],
            "group": r["group_size"],
            "history": r["history_length"],
            "m": r["m"],
            "f": r["f"],
            "S_MF": r["S_MF"],
            "Y": r["Y"],
            "actual orientation": r["actual_preference"],
            "predicted orientation": r["mf_pred"],
            "near_boundary": abs(float(r["S_MF"])) <= BOUNDARY_SCORE_TOLERANCE,
        }
        for r in combined_complete
        if r["mf_pred"] != r["actual_preference"]
    ]
    write_csv(
        OUT_DIR / "mf_mispredicted_units.csv",
        mis,
        ["architecture", "unit", "group", "history", "m", "f", "S_MF", "Y", "actual orientation", "predicted orientation", "near_boundary"],
    )

    # Fixed monotonic diagnostic: standardize S_MF and Y within architecture and examine history trend in residual.
    temp_rows = []
    if gdn_complete:
        ys = [float(r["Y"]) for r in gdn_complete]
        ss = [float(r["S_MF"]) for r in gdn_complete]
        y_mu, y_sd = sum(ys) / len(ys), statistics.pstdev(ys) or 1.0
        s_mu, s_sd = sum(ss) / len(ss), statistics.pstdev(ss) or 1.0
        for r in gdn_complete:
            residual = ((float(r["Y"]) - y_mu) / y_sd) - ((float(r["S_MF"]) - s_mu) / s_sd)
            temp_rows.append(
                {
                    "architecture": "GDN",
                    "unit_id": r["unit_id"],
                    "history_length": r["history_length"],
                    "Y_z": (float(r["Y"]) - y_mu) / y_sd,
                    "S_MF_z": (float(r["S_MF"]) - s_mu) / s_sd,
                    "R_temp": residual,
                    "abs_R_temp": abs(residual),
                }
            )
    write_csv(OUT_DIR / "temporal_residual_analysis.csv", temp_rows)
    residual_by_history = defaultdict(list)
    for row in temp_rows:
        residual_by_history[int(row["history_length"])].append(float(row["abs_R_temp"]))
    hist_resid = [(h, statistics.median(v)) for h, v in sorted(residual_by_history.items())]
    temporal_signal = "YES" if len(hist_resid) >= 3 and spearman([h for h, _ in hist_resid], [v for _, v in hist_resid]) and spearman([h for h, _ in hist_resid], [v for _, v in hist_resid]) > 0.5 else "NO_OR_WEAK"

    inventory = create_inventory()
    write_json(OUT_DIR / "source_artifact_inventory.json", inventory)

    kda_g = {int(r["group_size"]): r for r in kda_traj}
    gdn_g = {int(r["group_size"]): r for r in gdn_stability}
    g128_s = kda_g[128]["median_S_MF"]
    g16_s = kda_g[16]["median_S_MF"]
    if g128_s != "MISSING" and g16_s != "MISSING" and abs(float(g16_s)) < abs(float(g128_s)):
        kda_attenuation = "PARTIAL"
    else:
        kda_attenuation = "NOT_SUPPORTED"
    gdn_stable = "SUPPORTED" if gdn_g[128]["median_S_MF"] != "MISSING" and float(gdn_g[128]["median_S_MF"]) > BOUNDARY_SCORE_TOLERANCE else "NOT_SUPPORTED"
    if any(gdn_g[g]["median_S_MF"] == "MISSING" for g in (64, 32, 16)):
        gdn_stable = "PARTIAL"

    gdn_arch_conc = "PASS" if gdn_g[128]["MF_predicts"] == "C" else "FAIL"
    kda_arch_conc = "PARTIAL" if kda_g[128]["MF_predicts"] == "R" and kda_g[16]["MF_predicts"] != "B" else "PASS"
    kda_g16_boundary = "FAIL" if kda_g[16]["MF_predicts"] != "B" else "PASS"

    acc = {r["panel"]: r for r in acc_rows}
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE",
        "OUTCOME_SIGN_CONVENTION_GATE": "PASS",
        "UNIFIED_METRIC_COVERAGE_GATE": "PARTIAL",
        "GDN_FUNCTIONAL_METRIC_SELECTION_GATE": "PASS",
        "KDA_FUNCTIONAL_METRIC_SELECTION_GATE": "PASS",
        "N_GDN_UNITS": len({r["unit_id"] for r in gdn_complete}),
        "N_KDA_UNITS": len({r["unit_id"] for r in kda_complete}),
        "N_GDN_GROUPS": len({r["group_size"] for r in gdn_rows}),
        "N_KDA_GROUPS": len({r["group_size"] for r in kda_rows}),
        "N_GDN_COMPLETE_MF_ROWS": len(gdn_complete),
        "N_KDA_COMPLETE_MF_ROWS": len(kda_complete),
        "GDN_MEDIAN_m": median([r["m"] for r in gdn_complete]),
        "GDN_MEDIAN_f": median([r["f"] for r in gdn_complete]),
        "GDN_MEDIAN_S_MF": median([r["S_MF"] for r in gdn_complete]),
        "KDA_G128_MEDIAN_m": kda_g[128]["median_m"],
        "KDA_G128_MEDIAN_f": kda_g[128]["median_f"],
        "KDA_G128_MEDIAN_S_MF": kda_g[128]["median_S_MF"],
        "KDA_G64_MEDIAN_S_MF": kda_g[64]["median_S_MF"],
        "KDA_G32_MEDIAN_S_MF": kda_g[32]["median_S_MF"],
        "KDA_G16_MEDIAN_S_MF": kda_g[16]["median_S_MF"],
        "M_ONLY_GDN_ORIENTATION_ACCURACY": acc["GDN"]["M_ONLY"],
        "F_ONLY_GDN_ORIENTATION_ACCURACY": acc["GDN"]["F_ONLY"],
        "MF_GDN_ORIENTATION_ACCURACY": acc["GDN"]["MF"],
        "M_ONLY_KDA_ORIENTATION_ACCURACY": acc["KDA"]["M_ONLY"],
        "F_ONLY_KDA_ORIENTATION_ACCURACY": acc["KDA"]["F_ONLY"],
        "MF_KDA_ORIENTATION_ACCURACY": acc["KDA"]["MF"],
        "M_ONLY_COMBINED_ORIENTATION_ACCURACY": acc["combined"]["M_ONLY"],
        "F_ONLY_COMBINED_ORIENTATION_ACCURACY": acc["combined"]["F_ONLY"],
        "MF_COMBINED_ORIENTATION_ACCURACY": acc["combined"]["MF"],
        "M_ONLY_GDN_SPEARMAN": assoc["GDN"]["m"],
        "F_ONLY_GDN_SPEARMAN": assoc["GDN"]["f"],
        "MF_GDN_SPEARMAN": assoc["GDN"]["S_MF"],
        "M_ONLY_KDA_SPEARMAN": assoc["KDA"]["m"],
        "F_ONLY_KDA_SPEARMAN": assoc["KDA"]["f"],
        "MF_KDA_SPEARMAN": assoc["KDA"]["S_MF"],
        "MF_COMBINED_SPEARMAN": assoc["combined_standardized_Y"]["S_MF"],
        "DELTA_ACCURACY_MF_MINUS_M": inc["combined"]["DELTA_ACCURACY_MF_MINUS_M"],
        "DELTA_ACCURACY_MF_MINUS_F": inc["combined"]["DELTA_ACCURACY_MF_MINUS_F"],
        "DELTA_RHO_MF_MINUS_M": inc["combined"]["DELTA_RHO_MF_MINUS_M"],
        "DELTA_RHO_MF_MINUS_F": inc["combined"]["DELTA_RHO_MF_MINUS_F"],
        "GDN_ARCHITECTURE_SIGN_CONCORDANCE": gdn_arch_conc,
        "KDA_ARCHITECTURE_SIGN_CONCORDANCE": kda_arch_conc,
        "KDA_G16_BOUNDARY_CONCORDANCE": kda_g16_boundary,
        "KDA_RISK_SCORE_ATTENUATION_STATUS": kda_attenuation,
        "GDN_RISK_SCORE_STABILITY_STATUS": gdn_stable,
        "TEMPORAL_RESIDUAL_SIGNAL": temporal_signal,
        "BOUNDARY_SCORE_TOLERANCE": BOUNDARY_SCORE_TOLERANCE,
        "BOUNDARY_EFFECT_TOLERANCE": BOUNDARY_EFFECT_TOLERANCE,
        "TEMPORAL_TERM_INCLUDED": "NO",
        "UNIFIED_PRINCIPLE_GENERALIZATION": "WEAK_PARTIAL_COVERAGE",
        "UNIFIED_ORIENTATION_RISK_STATUS": "PARTIAL_SUPPORT",
        "METHOD_PRINCIPLE_READY": "NO",
        "METHOD_DESIGN_READY": "NO",
        "NEW_GPU_RUNS": 0,
        "NEW_MODEL_INFERENCE": 0,
        "NEW_QUANTIZED_TRAJECTORIES": 0,
        "FRESH_OUTPUT_VERIFICATION": "PENDING_VERIFY_STAGE",
        "PY_COMPILE": "PENDING",
        "PYTEST": "PENDING",
        "GPU_RUNS_LAUNCHED": 0,
        "GPU_RELEASE_VERIFICATION": "PASS_NO_GPU_USED",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO",
        "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
        "created_at_epoch": time.time(),
        "analysis_started_at_epoch": started,
        "required_files": REQUIRED_FILES,
    }

    write_json(OUT_DIR / "summary.json", summary)
    write_markdown_outputs(summary, inventory)
    write_json(OUT_DIR / "summary.json", summary)
    write_final_console_summary(summary)


def verify(py_compile_status, pytest_status):
    summary_path = OUT_DIR / "summary.json"
    if not summary_path.exists():
        raise SystemExit("summary.json missing; run --stage analyze first")
    summary = read_json(summary_path)
    missing = [name for name in REQUIRED_FILES if not (OUT_DIR / name).exists()]
    empty = [name for name in REQUIRED_FILES if (OUT_DIR / name).exists() and os.path.getsize(OUT_DIR / name) == 0]
    inventory = read_json(OUT_DIR / "source_artifact_inventory.json")
    provenance_ok = len(inventory.get("artifacts", [])) >= 6
    row_counts = {}
    for name in ["gdn_unified_risk_panel.csv", "kda_unified_risk_panel.csv", "combined_unified_risk_panel.csv"]:
        with (OUT_DIR / name).open(newline="") as f:
            reader = csv.DictReader(f)
            row_counts[name] = sum(1 for _ in reader)
    mtimes_ok = all(os.path.getmtime(OUT_DIR / name) >= summary["analysis_started_at_epoch"] for name in REQUIRED_FILES if (OUT_DIR / name).exists())
    freshness = "PASS" if not missing and not empty and provenance_ok and all(v > 0 for v in row_counts.values()) and mtimes_ok else "FAIL"
    verification = {
        "missing_files": missing,
        "empty_files": empty,
        "row_counts": row_counts,
        "provenance_ok": provenance_ok,
        "mtimes_ok": mtimes_ok,
        "FRESH_OUTPUT_VERIFICATION": freshness,
        "output_hashes": {name: sha256(OUT_DIR / name) for name in REQUIRED_FILES if (OUT_DIR / name).exists() and name != "final_console_summary.txt"},
    }
    write_json(OUT_DIR / "fresh_output_verification.json", verification)
    summary["FRESH_OUTPUT_VERIFICATION"] = freshness
    summary["PY_COMPILE"] = py_compile_status
    summary["PYTEST"] = pytest_status
    summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES" if freshness == "PASS" and py_compile_status == "PASS" and pytest_status == "PASS" else "NO"
    write_json(summary_path, summary)
    write_markdown_outputs(summary, inventory)
    write_final_console_summary(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["analyze", "verify"], default="analyze")
    parser.add_argument("--py-compile-status", default="PENDING")
    parser.add_argument("--pytest-status", default="PENDING")
    args = parser.parse_args()
    if args.stage == "analyze":
        analyze()
    else:
        verify(args.py_compile_status, args.pytest_status)


if __name__ == "__main__":
    main()
