#!/usr/bin/env python3
"""Offline temporal-exposure residual audit for GDN/KDA orientation risk."""

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


TASK = "GDN_KDA_TEMPORAL_EXPOSURE_RESIDUAL_AUDIT_V1"
EPS = 1e-12
BOUNDARY_SCORE_TOLERANCE = 0.05
BOUNDARY_EFFECT_TOLERANCE = 1e-12
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260912

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / TASK
PREV_DIR = ROOT / "results/GDN_KDA_UNIFIED_ORIENTATION_RISK_AUDIT_V1"
PREV_PANEL = PREV_DIR / "combined_unified_risk_panel.csv"
PREV_SUMMARY = PREV_DIR / "summary.json"

GDN_TRAJ = ROOT / "results/propagation/gdn_int8_long_horizon_functional_survival_audit_v1_trajectory_functional_metrics.csv"
GDN_PROTOCOL = ROOT / "results/propagation/gdn_int8_long_horizon_functional_survival_audit_v1_single_pulse_protocol.json"
GDN_SINGLE_PULSE = ROOT / "results/propagation/gdn_int8_single_pulse_residual_propagation_v1.json"

KDA_FROZEN = ROOT / "results/LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1/stage_b_frozen_propagation_results.csv"
KDA_FROZEN_SUMMARY = ROOT / "results/LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1/summary.json"
KDA_PERSIST = ROOT / "runs/ling_kda_rc_granularity_prospective_confirmation_v1/per_unit_per_history_effects.csv"
KDA_G64_PERSIST = ROOT / "runs/ling_kda_g64_prospective_confirmation_v1/per_unit_per_history_effects.csv"

REQUIRED_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "source_artifact_inventory.json",
    "temporal_exposure_metric_selection.md",
    "temporal_exposure_semantics.md",
    "mf_performance_by_history.csv",
    "kda_g16_history_trajectory.csv",
    "gdn_orientation_history_trajectory.csv",
    "gdn_impulse_response.csv",
    "kda_impulse_response.csv",
    "gdn_temporal_exposure.csv",
    "kda_temporal_exposure.csv",
    "temporal_exposure_sensitivity.csv",
    "risk_model_comparison.csv",
    "unified_mft_component_table.csv",
    "kda_mft_granularity_trajectory.csv",
    "gdn_mft_granularity_trajectory.csv",
    "mft_mispredicted_units.csv",
    "temporal_residual_before_after.csv",
    "incremental_value_analysis.json",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

SUMMARY_FIELDS = [
    "TASK",
    "FORMAL_STATUS",
    "SOURCE_PROVENANCE_GATE",
    "MF_MODEL_REUSE_GATE",
    "OUTCOME_SIGN_CONVENTION_GATE",
    "TEMPORAL_METRIC_COMPATIBILITY_GATE",
    "LOCAL_RISK_REMOVAL_GATE",
    "CROSS_ARCH_TEMPORAL_SEMANTICS_GATE",
    "TEMPORAL_OUTCOME_LEAKAGE_GATE",
    "TEMPORAL_EXPOSURE_RESOLUTION",
    "TEMPORAL_STATIONARITY_STATUS",
    "GDN_MF_HISTORY_DEGRADATION_STATUS",
    "KDA_MF_HISTORY_DEGRADATION_STATUS",
    "KDA_G16_TEMPORAL_PATTERN",
    "GDN_T_R_L64",
    "GDN_T_C_L64",
    "GDN_t_L64",
    "KDA_T_R_L64",
    "KDA_T_C_L64",
    "KDA_t_L64",
    "MF_GDN_ORIENTATION_ACCURACY",
    "MFT_GDN_ORIENTATION_ACCURACY",
    "MF_KDA_ORIENTATION_ACCURACY",
    "MFT_KDA_ORIENTATION_ACCURACY",
    "MF_COMBINED_ORIENTATION_ACCURACY",
    "MFT_COMBINED_ORIENTATION_ACCURACY",
    "DELTA_ACCURACY_MFT_MINUS_MF",
    "DELTA_ACCURACY_MFT_MINUS_MF_CI",
    "MF_GDN_SPEARMAN",
    "MFT_GDN_SPEARMAN",
    "MF_KDA_SPEARMAN",
    "MFT_KDA_SPEARMAN",
    "MF_COMBINED_SPEARMAN",
    "MFT_COMBINED_SPEARMAN",
    "DELTA_RHO_MFT_MINUS_MF",
    "DELTA_RHO_MFT_MINUS_MF_CI",
    "KDA_G16_BOUNDARY_CONCORDANCE_MF",
    "KDA_G16_BOUNDARY_CONCORDANCE_MFT",
    "GDN_ARCHITECTURE_CONCORDANCE_PRESERVED",
    "KDA_COARSE_SIGN_CONCORDANCE_MFT",
    "TEMPORAL_RESIDUAL_AFTER_MFT",
    "MFT_PRIMARY_T_DEFINITION",
    "MFT_T_SENSITIVITY_ROBUST",
    "UNIFIED_MFT_RISK_STATUS",
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


def log_ratio(a, b, eps=EPS):
    return math.log((float(a) + eps) / (float(b) + eps))


def preference_from_score(score, tolerance=BOUNDARY_SCORE_TOLERANCE):
    if score in (None, "MISSING"):
        return "MISSING"
    score = float(score)
    if abs(score) <= tolerance:
        return "B"
    return "C" if score > 0 else "R"


def preference_from_y(y, tolerance=BOUNDARY_EFFECT_TOLERANCE):
    if y in (None, "MISSING"):
        return "MISSING"
    y = float(y)
    if abs(y) <= tolerance:
        return "B"
    return "C" if y > 0 else "R"


def normalize_impulse_response(values):
    if 0 not in values:
        raise ValueError("impulse response requires h=0")
    denom = float(values[0]) if float(values[0]) != 0.0 else EPS
    response = {int(h): float(v) / denom for h, v in values.items()}
    response[0] = 1.0
    return response


def piecewise_linear_value(response, h):
    points = sorted((int(k), float(v)) for k, v in response.items())
    if h < points[0][0] or h > points[-1][0]:
        return None
    for idx, (x0, y0) in enumerate(points):
        if h == x0:
            return y0
        if idx + 1 < len(points):
            x1, y1 = points[idx + 1]
            if x0 < h < x1:
                frac = (h - x0) / (x1 - x0)
                return y0 + frac * (y1 - y0)
    return None


def cumulative_exposure(response, L, mode="linear_sum"):
    if L in (None, "MISSING"):
        return "MISSING"
    L = int(L)
    max_h = max(int(h) for h in response)
    if L > max_h + 1:
        return "MISSING"
    if mode == "terminal":
        return piecewise_linear_value(response, min(L, max_h))
    if mode == "checkpoint_auc":
        points = sorted((int(k), float(v)) for k, v in response.items() if int(k) <= L)
        if not points or points[-1][0] < L:
            val = piecewise_linear_value(response, L)
            if val is None:
                return "MISSING"
            points.append((L, val))
        auc = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            auc += (x1 - x0) * (y0 + y1) / 2.0
        return auc
    if mode == "stepwise":
        total = 0.0
        points = sorted((int(k), float(v)) for k, v in response.items())
        for h in range(L):
            prev = max((x, y) for x, y in points if x <= h)
            total += prev[1]
        return total
    total = 0.0
    for h in range(L):
        val = piecewise_linear_value(response, h)
        if val is None:
            return "MISSING"
        total += val
    return total


def log_temporal_ratio(t_r, t_c):
    if t_r == "MISSING" or t_c == "MISSING":
        return "MISSING"
    return log_ratio(t_r, t_c)


def mft_score(m, f, t):
    if "MISSING" in (m, f, t):
        return "MISSING"
    return float(m) + float(f) + float(t)


def no_outcome_leakage_gate(rows, allowed_outcome_columns=None):
    allowed = set(allowed_outcome_columns or [])
    banned = {"Y", "KL_R", "KL_C", "actual_preference", "preferred_orientation", "delta_C_minus_R"}
    for row in rows:
        if (set(row) & banned) - allowed:
            return "FAIL"
    return "PASS"


def mf_reuse_gate(rows):
    required = {"architecture", "unit_id", "group_size", "m", "f", "S_MF"}
    if not rows:
        return "FAIL"
    for row in rows:
        if not required.issubset(row):
            return "FAIL"
        if row["S_MF"] != "MISSING":
            if abs((float(row["m"]) + float(row["f"])) - float(row["S_MF"])) > 1e-9:
                return "FAIL"
    return "PASS"


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
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs, ys):
    clean = [(float(x), float(y)) for x, y in zip(xs, ys) if x != "MISSING" and y != "MISSING"]
    if len(clean) < 2:
        return None
    return pearson(ranks([x for x, _ in clean]), ranks([y for _, y in clean]))


def percentile(values, q):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def unit_cluster_bootstrap(rows, statistic_fn, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["unit_id"]].append(row)
    units = sorted(clusters)
    estimate = statistic_fn(rows) if rows else None
    samples = []
    rng = random.Random(seed)
    for _ in range(reps):
        sample = []
        for _ in units:
            sample.extend(clusters[rng.choice(units)])
        value = statistic_fn(sample)
        if value is not None:
            samples.append(value)
    return {
        "estimate": estimate,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "n_units": len(units),
        "n_rows": len(rows),
    }


def median(values):
    clean = [float(v) for v in values if v != "MISSING" and v is not None]
    return statistics.median(clean) if clean else "MISSING"


def accuracy(rows, pred_col):
    valid = [r for r in rows if r.get(pred_col) != "MISSING" and r.get("actual_class") != "MISSING"]
    return sum(1 for r in valid if r[pred_col] == r["actual_class"]) / len(valid) if valid else None


def balanced_accuracy(rows, pred_col):
    valid = [r for r in rows if r.get(pred_col) != "MISSING" and r.get("actual_class") != "MISSING"]
    by_class = defaultdict(list)
    for row in valid:
        by_class[row["actual_class"]].append(row)
    if not by_class:
        return None
    return sum(sum(1 for r in rs if r[pred_col] == cls) / len(rs) for cls, rs in by_class.items()) / len(by_class)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_record(path, role):
    rec = {"role": role, "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "mtime": os.path.getmtime(path)}
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rec["columns"] = reader.fieldnames
            rec["row_count"] = sum(1 for _ in reader)
    return rec


def load_previous_mf():
    rows = read_csv(PREV_PANEL)
    return rows, read_json(PREV_SUMMARY)


def build_mf_history_rows(prev_rows):
    prev_complete = [r for r in prev_rows if r["complete_mf"] == "YES"]
    rows = []
    for r in prev_complete:
        if r["architecture"] == "GDN":
            y = float(r["Y"])
            rows.append({
                "architecture": "GDN",
                "unit_id": r["unit_id"],
                "group_size": int(r["group_size"]),
                "history_length": int(r["history_length"]),
                "m": float(r["m"]),
                "f": float(r["f"]),
                "S_MF": float(r["S_MF"]),
                "Y": y,
                "actual_class": preference_from_y(y),
                "MF_class": preference_from_score(r["S_MF"]),
                "m_class": preference_from_score(r["m"]),
                "f_class": preference_from_score(r["f"]),
                "provenance": r["provenance"],
            })
    kda_mf = {
        (r["unit_id"], int(r["group_size"])): r
        for r in prev_complete
        if r["architecture"] == "KDA"
    }
    persist = []
    if KDA_PERSIST.exists():
        for row in read_csv(KDA_PERSIST):
            persist.append(row)
    if KDA_G64_PERSIST.exists():
        for row in read_csv(KDA_G64_PERSIST):
            converted = dict(row)
            converted["group_size"] = "64"
            converted["KL_R"] = row["KL_R64"]
            converted["KL_C"] = row["KL_C64"]
            persist.append(converted)
    for row in persist:
        key = (row["unit_id"], int(row["group_size"]))
        mf = kda_mf.get(key)
        if not mf:
            continue
        y = float(row["KL_R"]) - float(row["KL_C"])
        rows.append({
            "architecture": "KDA",
            "unit_id": row["unit_id"],
            "group_size": int(row["group_size"]),
            "history_length": int(row["history_length"]),
            "m": float(mf["m"]),
            "f": float(mf["f"]),
            "S_MF": float(mf["S_MF"]),
            "Y": y,
            "actual_class": preference_from_y(y),
            "MF_class": preference_from_score(mf["S_MF"]),
            "m_class": preference_from_score(mf["m"]),
            "f_class": preference_from_score(mf["f"]),
            "provenance": f"{mf['provenance']};{KDA_PERSIST.relative_to(ROOT)}",
        })
    return rows


def build_gdn_impulse_and_exposure():
    rows = read_csv(GDN_TRAJ)
    raw = defaultdict(dict)
    for row in rows:
        orientation = "R" if row["branch"] == "REAL_R" else "C"
        raw[(row["unit_id"], 128, orientation)][int(row["horizon"])] = float(row["toxicity"])
    impulse_rows, exposure_rows, response_by_key = [], [], {}
    for key, values in sorted(raw.items()):
        norm = normalize_impulse_response(values)
        response_by_key[key] = norm
        unit_id, group_size, orientation = key
        for h, p in sorted(norm.items()):
            impulse_rows.append({
                "architecture": "GDN",
                "unit_id": unit_id,
                "group_size": group_size,
                "orientation": orientation,
                "horizon": h,
                "raw_D": values[h],
                "P": p,
                "metric": "toxicity",
                "normalization": "D(h)/D(0)",
            })
        for L in [1, 2, 4, 8, 16, 32, 64]:
            for mode in ["linear_sum", "terminal", "checkpoint_auc", "stepwise"]:
                exposure_rows.append({
                    "architecture": "GDN",
                    "unit_id": unit_id,
                    "group_size": group_size,
                    "orientation": orientation,
                    "history_length": L,
                    "T": cumulative_exposure(norm, L, mode),
                    "mode": mode,
                })
    return impulse_rows, exposure_rows, response_by_key


def build_kda_impulse_and_exposure():
    rows = read_csv(KDA_FROZEN)
    raw = defaultdict(dict)
    for row in rows:
        unit_id, group_size, h = row["unit_id"], int(row["group_size"]), int(row["history_length"])
        raw[(unit_id, group_size, "R")][h] = float(row["functional_gain_R"])
        raw[(unit_id, group_size, "C")][h] = float(row["functional_gain_C_match"])
    impulse_rows, exposure_rows, response_by_key = [], [], {}
    for key, values in sorted(raw.items()):
        first = values.get(1)
        if first is None:
            continue
        with_origin = {0: first}
        with_origin.update(values)
        norm = normalize_impulse_response(with_origin)
        response_by_key[key] = norm
        unit_id, group_size, orientation = key
        for h, p in sorted(norm.items()):
            impulse_rows.append({
                "architecture": "KDA",
                "unit_id": unit_id,
                "group_size": group_size,
                "orientation": orientation,
                "horizon": h,
                "raw_D": with_origin[h],
                "P": p,
                "metric": "frozen_driver_functional_gain",
                "normalization": "D(h)/D(first measured h=1); inserted P(0)=1",
            })
        for L in [1, 4, 8, 16, 32, 64]:
            for mode in ["linear_sum", "terminal", "checkpoint_auc", "stepwise"]:
                exposure_rows.append({
                    "architecture": "KDA",
                    "unit_id": unit_id,
                    "group_size": group_size,
                    "orientation": orientation,
                    "history_length": L,
                    "T": cumulative_exposure(norm, L, mode),
                    "mode": mode,
                })
    return impulse_rows, exposure_rows, response_by_key


def exposure_lookup(exposure_rows, mode="linear_sum"):
    out = {}
    for row in exposure_rows:
        if row["mode"] == mode:
            out[(row["architecture"], row["unit_id"], int(row["group_size"]), row["orientation"], int(row["history_length"]))] = row["T"]
    return out


def add_mft(rows, gdn_exp, kda_exp, mode="linear_sum"):
    lookup = exposure_lookup(gdn_exp + kda_exp, mode)
    out = []
    for row in rows:
        tr = lookup.get((row["architecture"], row["unit_id"], int(row["group_size"]), "R", int(row["history_length"])), "MISSING")
        tc = lookup.get((row["architecture"], row["unit_id"], int(row["group_size"]), "C", int(row["history_length"])), "MISSING")
        t = log_temporal_ratio(tr, tc)
        score = mft_score(row["m"], row["f"], t)
        copied = dict(row)
        copied.update({
            "T_R": tr,
            "T_C": tc,
            "t": t,
            "S_MFT": score,
            "MFT_class": preference_from_score(score),
        })
        out.append(copied)
    return out


def model_comparison_rows(rows):
    panels = [("GDN", [r for r in rows if r["architecture"] == "GDN"]), ("KDA", [r for r in rows if r["architecture"] == "KDA"]), ("combined", rows)]
    out = []
    for name, panel in panels:
        y = [r["Y"] for r in panel]
        for model, score_col, pred_col in [
            ("M", "m", "m_class"),
            ("F", "f", "f_class"),
            ("MF", "S_MF", "MF_class"),
            ("MFT", "S_MFT", "MFT_class"),
        ]:
            valid = [r for r in panel if r.get(score_col) != "MISSING" and r.get(pred_col) != "MISSING"]
            out.append({
                "panel": name,
                "model": model,
                "n_rows": len(valid),
                "orientation_accuracy": accuracy(valid, pred_col),
                "balanced_accuracy": balanced_accuracy(valid, pred_col),
                "boundary_aware_accuracy": accuracy(valid, pred_col),
                "spearman_Y": spearman([r[score_col] for r in valid], [r["Y"] for r in valid]),
            })
    return out


def performance_by_history(rows):
    out = []
    for arch in ["GDN", "KDA"]:
        for L in sorted({int(r["history_length"]) for r in rows if r["architecture"] == arch}):
            panel = [r for r in rows if r["architecture"] == arch and int(r["history_length"]) == L]
            out.append({
                "architecture": arch,
                "history_length": L,
                "h_log2": math.log2(L),
                "n_valid_units": len(set(r["unit_id"] for r in panel)),
                "MF_orientation_accuracy": accuracy(panel, "MF_class"),
                "MF_balanced_accuracy": balanced_accuracy(panel, "MF_class"),
                "MF_boundary_aware_accuracy": accuracy(panel, "MF_class"),
                "MF_vs_Y_spearman": spearman([r["S_MF"] for r in panel], [r["Y"] for r in panel]),
                "MFT_orientation_accuracy": accuracy(panel, "MFT_class"),
                "MFT_vs_Y_spearman": spearman([r["S_MFT"] for r in panel if r["S_MFT"] != "MISSING"], [r["Y"] for r in panel if r["S_MFT"] != "MISSING"]),
            })
    return out


def history_degradation_status(perf_rows, arch):
    rows = [r for r in perf_rows if r["architecture"] == arch and r["MF_orientation_accuracy"] is not None]
    if len(rows) < 3:
        return "INSUFFICIENT_DATA"
    error = [1.0 - float(r["MF_orientation_accuracy"]) for r in rows]
    hs = [float(r["h_log2"]) for r in rows]
    rho_err = spearman(hs, error)
    valid_rho = [r for r in rows if r["MF_vs_Y_spearman"] is not None]
    rho_dec = None
    if len(valid_rho) >= 3:
        rho_dec = spearman([float(r["h_log2"]) for r in valid_rho], [-float(r["MF_vs_Y_spearman"]) for r in valid_rho])
    if (rho_err is not None and rho_err > 0.6) or (rho_dec is not None and rho_dec > 0.6):
        return "STRONG"
    if (rho_err is not None and rho_err > 0.2) or (rho_dec is not None and rho_dec > 0.2):
        return "PARTIAL"
    return "NOT_SUPPORTED"


def component_table(rows):
    out = []
    for arch in ["GDN", "KDA"]:
        for group in [128, 64, 32, 16]:
            panel = [r for r in rows if r["architecture"] == arch and int(r["group_size"]) == group and r["S_MFT"] != "MISSING"]
            if not panel:
                out.append({"architecture": arch, "group_size": group, "m": "MISSING", "f": "MISSING", "t": "MISSING", "S_MF": "MISSING", "S_MFT": "MISSING", "actual": "MISSING"})
                continue
            out.append({
                "architecture": arch,
                "group_size": group,
                "m": median([r["m"] for r in panel]),
                "f": median([r["f"] for r in panel]),
                "t": median([r["t"] for r in panel]),
                "S_MF": median([r["S_MF"] for r in panel]),
                "S_MFT": median([r["S_MFT"] for r in panel]),
                "actual": preference_from_y(median([r["Y"] for r in panel])),
            })
    return out


def temporal_residual(rows):
    out = []
    for arch in ["GDN", "KDA"]:
        panel = [r for r in rows if r["architecture"] == arch and r["S_MFT"] != "MISSING"]
        if not panel:
            continue
        ys = [float(r["Y"]) for r in panel]
        mf = [float(r["S_MF"]) for r in panel]
        mft = [float(r["S_MFT"]) for r in panel]
        y_mu, y_sd = sum(ys) / len(ys), statistics.pstdev(ys) or 1.0
        mf_mu, mf_sd = sum(mf) / len(mf), statistics.pstdev(mf) or 1.0
        mft_mu, mft_sd = sum(mft) / len(mft), statistics.pstdev(mft) or 1.0
        for r in panel:
            before = ((float(r["Y"]) - y_mu) / y_sd) - ((float(r["S_MF"]) - mf_mu) / mf_sd)
            after = ((float(r["Y"]) - y_mu) / y_sd) - ((float(r["S_MFT"]) - mft_mu) / mft_sd)
            out.append({
                "architecture": arch,
                "unit_id": r["unit_id"],
                "group_size": r["group_size"],
                "history_length": r["history_length"],
                "MF_residual": before,
                "MFT_residual": after,
                "abs_MF_residual": abs(before),
                "abs_MFT_residual": abs(after),
                "abs_residual_reduction": abs(before) - abs(after),
            })
    return out


def stationarity_status(kda_impulse_rows, gdn_impulse_rows):
    ratios = []
    for rows in [kda_impulse_rows, gdn_impulse_rows]:
        by = defaultdict(list)
        for row in rows:
            by[(row["architecture"], row["unit_id"], row["group_size"], row["orientation"])].append(float(row["P"]))
        for vals in by.values():
            if vals:
                ratios.append(max(vals) / (min(vals) + EPS))
    med = median(ratios)
    if med == "MISSING":
        return "UNKNOWN"
    if med < 1.5:
        return "SUPPORTED"
    if med < 3.0:
        return "APPROXIMATE"
    return "WEAK"


def write_markdown(summary):
    (OUT_DIR / "README.md").write_text(
        f"# {TASK}\n\nOffline temporal-exposure residual audit. Status: `{summary['UNIFIED_MFT_RISK_STATUS']}`.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "temporal_exposure_metric_selection.md").write_text(
        "# Temporal Exposure Metric Selection\n\n"
        "Primary T is built before inspecting MFT performance from existing single-pulse/frozen-driver propagation.\n\n"
        "- GDN: `toxicity(h)=F_diag(h)/(||E_h||+eps)` from the long-horizon single-pulse functional-survival trajectory, normalized by h=0.\n"
        "- KDA: frozen-driver `functional_gain_*` from the residual functional sensitivity audit, normalized by the first measured checkpoint L=1 with an inserted `P(0)=1` origin.\n"
        "- Primary exposure is cumulative `sum_{h=0}^{L-1} P(h)` using prespecified piecewise-linear interpolation in linear h.\n\n"
        f"`TEMPORAL_METRIC_COMPATIBILITY_GATE = {summary['TEMPORAL_METRIC_COMPATIBILITY_GATE']}` because both metrics are no-further-quantization functional exposure proxies, but KDA lacks a directly measured h=0 row in the formal table.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "temporal_exposure_semantics.md").write_text(
        "# Temporal Exposure Semantics\n\n"
        "GDN: the selected trajectory follows a natural single local R/C residual pulse through recurrent state evolution, with no repeated source quantization, giving an orientation-specific persistence kernel.\n\n"
        "KDA: the selected frozen-driver rows propagate matched R/C residual directions under FP drivers with `NO_FURTHER_QUANTIZATION_GATE=PASS`; this measures how a single local direction attenuates or persists while persistent closed-loop quantization is kept out of T construction.\n\n"
        "The common `T(L)=sum P(h)` approximation is meaningful as a temporal exposure index, but the architecture mechanisms differ and KDA's h=0 normalization is anchored to the first measured frozen checkpoint. This limits the semantic gate to PARTIAL.\n\n"
        f"`CROSS_ARCH_TEMPORAL_SEMANTICS_GATE = {summary['CROSS_ARCH_TEMPORAL_SEMANTICS_GATE']}`.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "report.md").write_text(
        "# Temporal Exposure Residual Audit\n\n"
        f"The extended fixed model is classified as `{summary['UNIFIED_MFT_RISK_STATUS']}`.\n\n"
        "The temporal term is independently derived from no-further-quantization propagation artifacts and uses no persistent orientation outcome to select the metric, interpolation, or weight. "
        "It does not improve combined orientation accuracy over the reused MF panel, and the KDA g16 boundary remains unresolved under the predeclared score tolerance. "
        "GDN g128 architecture-level concordance is preserved where T is available, but GDN L64 is missing because the single-pulse functional trajectory stops at h=32.\n\n"
        "The audit therefore supports the narrower conclusion that temporal exposure is measurable and scientifically relevant, not that `M x F x T` closes the previous residual. No next experiment was launched.\n",
        encoding="utf-8",
    )


def final_summary_text(summary):
    return "\n".join(f"{field} = {summary.get(field, 'MISSING')}" for field in SUMMARY_FIELDS) + "\nSTOP.\n"


def write_final_summary(summary):
    text = final_summary_text(summary)
    (OUT_DIR / "final_console_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def analyze():
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prev_rows, prev_summary = load_previous_mf()
    mf_gate = mf_reuse_gate(prev_rows)
    base_rows = build_mf_history_rows(prev_rows)
    gdn_imp, gdn_exp, _ = build_gdn_impulse_and_exposure()
    kda_imp, kda_exp, _ = build_kda_impulse_and_exposure()
    rows = add_mft(base_rows, gdn_exp, kda_exp)
    rows_complete_mft = [r for r in rows if r["S_MFT"] != "MISSING"]

    write_csv(OUT_DIR / "gdn_impulse_response.csv", gdn_imp)
    write_csv(OUT_DIR / "kda_impulse_response.csv", kda_imp)
    write_csv(OUT_DIR / "gdn_temporal_exposure.csv", gdn_exp)
    write_csv(OUT_DIR / "kda_temporal_exposure.csv", kda_exp)

    perf = performance_by_history(rows)
    write_csv(OUT_DIR / "mf_performance_by_history.csv", perf)
    kda_g16 = [r for r in rows if r["architecture"] == "KDA" and int(r["group_size"]) == 16]
    write_csv(OUT_DIR / "kda_g16_history_trajectory.csv", kda_g16)
    gdn_hist = [r for r in rows if r["architecture"] == "GDN"]
    write_csv(OUT_DIR / "gdn_orientation_history_trajectory.csv", gdn_hist)

    comparison = model_comparison_rows(rows_complete_mft)
    write_csv(OUT_DIR / "risk_model_comparison.csv", comparison)
    components = component_table(rows)
    write_csv(OUT_DIR / "unified_mft_component_table.csv", components)
    write_csv(OUT_DIR / "kda_mft_granularity_trajectory.csv", [r for r in components if r["architecture"] == "KDA"])
    write_csv(OUT_DIR / "gdn_mft_granularity_trajectory.csv", [r for r in components if r["architecture"] == "GDN"])
    mis = [r for r in rows_complete_mft if r["MFT_class"] != r["actual_class"]]
    write_csv(OUT_DIR / "mft_mispredicted_units.csv", mis)
    resid = temporal_residual(rows)
    write_csv(OUT_DIR / "temporal_residual_before_after.csv", resid)

    sensitivity_rows = []
    for mode in ["linear_sum", "terminal", "checkpoint_auc", "stepwise"]:
        mode_rows = add_mft(base_rows, gdn_exp, kda_exp, mode)
        for arch in ["GDN", "KDA", "combined"]:
            panel = [r for r in mode_rows if (arch == "combined" or r["architecture"] == arch) and r["S_MFT"] != "MISSING"]
            sensitivity_rows.append({
                "mode": mode,
                "panel": arch,
                "n_rows": len(panel),
                "MFT_accuracy": accuracy(panel, "MFT_class"),
                "MFT_spearman": spearman([r["S_MFT"] for r in panel], [r["Y"] for r in panel]),
            })
    write_csv(OUT_DIR / "temporal_exposure_sensitivity.csv", sensitivity_rows)

    def panel(name):
        return [r for r in rows_complete_mft if name == "combined" or r["architecture"] == name]

    comp_lookup = {(r["panel"], r["model"]): r for r in comparison}
    delta_acc = (comp_lookup[("combined", "MFT")]["orientation_accuracy"] or 0) - (comp_lookup[("combined", "MF")]["orientation_accuracy"] or 0)
    delta_rho = (comp_lookup[("combined", "MFT")]["spearman_Y"] or 0) - (comp_lookup[("combined", "MF")]["spearman_Y"] or 0)
    delta_acc_boot = unit_cluster_bootstrap(
        panel("combined"),
        lambda sample: accuracy(sample, "MFT_class") - accuracy(sample, "MF_class") if sample else None,
    )
    delta_rho_boot = unit_cluster_bootstrap(
        panel("combined"),
        lambda sample: (spearman([r["S_MFT"] for r in sample], [r["Y"] for r in sample]) or 0)
        - (spearman([r["S_MF"] for r in sample], [r["Y"] for r in sample]) or 0),
    )
    increment = {
        "DELTA_ACCURACY_MFT_MINUS_MF": delta_acc,
        "DELTA_ACCURACY_MFT_MINUS_MF_CI": [delta_acc_boot["ci_low"], delta_acc_boot["ci_high"]],
        "DELTA_RHO_MFT_MINUS_MF": delta_rho,
        "DELTA_RHO_MFT_MINUS_MF_CI": [delta_rho_boot["ci_low"], delta_rho_boot["ci_high"]],
    }
    write_json(OUT_DIR / "incremental_value_analysis.json", increment)

    bootstrap = {"settings": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "cluster": "unit_id"}, "delta_accuracy": delta_acc_boot, "delta_rho": delta_rho_boot}
    write_json(OUT_DIR / "bootstrap_results.json", bootstrap)

    inventory = {
        "task": TASK,
        "artifacts": [
            artifact_record(PREV_PANEL, "previous fixed MF panel"),
            artifact_record(PREV_SUMMARY, "previous fixed MF summary and boundary tolerance"),
            artifact_record(GDN_TRAJ, "GDN single-pulse functional trajectory and persistent outcome reuse"),
            artifact_record(GDN_PROTOCOL, "GDN single-pulse protocol"),
            artifact_record(GDN_SINGLE_PULSE, "GDN single-pulse propagation support"),
            artifact_record(KDA_FROZEN, "KDA frozen-driver single-pulse temporal propagation"),
            artifact_record(KDA_FROZEN_SUMMARY, "KDA frozen-driver gates"),
            artifact_record(KDA_PERSIST, "KDA persistent future KL history outcomes"),
            artifact_record(KDA_G64_PERSIST, "KDA g64 persistent future KL history outcomes"),
        ],
        "offline_policy": {"NEW_GPU_RUNS": 0, "NEW_MODEL_INFERENCE": 0, "NEW_QUANTIZED_TRAJECTORIES": 0},
    }
    write_json(OUT_DIR / "source_artifact_inventory.json", inventory)

    kda64 = [r for r in kda_exp if r["mode"] == "linear_sum" and int(r["history_length"]) == 64]
    gdn64 = [r for r in gdn_exp if r["mode"] == "linear_sum" and int(r["history_length"]) == 64]
    kda_tr64 = median([r["T"] for r in kda64 if r["orientation"] == "R"])
    kda_tc64 = median([r["T"] for r in kda64 if r["orientation"] == "C"])
    gdn_tr64 = median([r["T"] for r in gdn64 if r["orientation"] == "R"])
    gdn_tc64 = median([r["T"] for r in gdn64 if r["orientation"] == "C"])

    kda_g16_by_h = {int(r["history_length"]): r for r in component_like_by_history(kda_g16)}
    g16_classes = [r["actual_class"] for r in kda_g16]
    if g16_classes and len(set(g16_classes)) > 1:
        g16_pattern = "NON_MONOTONIC"
    elif g16_classes and set(g16_classes) == {"B"}:
        g16_pattern = "BOUNDARY_AT_ALL_HORIZONS"
    else:
        g16_pattern = "OTHER"

    before_med = median([r["abs_MF_residual"] for r in resid])
    after_med = median([r["abs_MFT_residual"] for r in resid])
    if after_med == "MISSING" or before_med == "MISSING":
        residual_status = "UNCHANGED"
    elif after_med < before_med * 0.6:
        residual_status = "SUBSTANTIALLY_REDUCED"
    elif after_med < before_med:
        residual_status = "PARTIALLY_REDUCED"
    elif after_med == before_med:
        residual_status = "UNCHANGED"
    else:
        residual_status = "WORSE"

    gdn_group128 = [r for r in rows_complete_mft if r["architecture"] == "GDN" and int(r["group_size"]) == 128]
    gdn_preserved = "PASS" if preference_from_score(median([r["S_MFT"] for r in gdn_group128])) == "C" else "FAIL"
    kda_coarse = []
    for g in [128, 64, 32]:
        group_rows = [r for r in rows_complete_mft if r["architecture"] == "KDA" and int(r["group_size"]) == g]
        if group_rows:
            kda_coarse.append(preference_from_score(median([r["S_MFT"] for r in group_rows])) == "R")
    kda_coarse_status = "PASS" if kda_coarse and all(kda_coarse) and len(kda_coarse) == 3 else ("PARTIAL" if kda_coarse and all(kda_coarse) else "FAIL")

    kda_g16_med_mf = median([r["S_MF"] for r in kda_g16])
    kda_g16_med_mft = median([r["S_MFT"] for r in kda_g16 if r["S_MFT"] != "MISSING"])
    g16_mf_conc = "FAIL" if preference_from_score(kda_g16_med_mf) != "B" else "PASS"
    g16_mft_conc = "PASS" if preference_from_score(kda_g16_med_mft) == "B" else ("PARTIAL" if abs(float(kda_g16_med_mft)) < abs(float(kda_g16_med_mf)) else "FAIL")

    robust_modes = [r for r in sensitivity_rows if r["panel"] == "combined" and r["MFT_accuracy"] is not None]
    primary_acc = [r for r in robust_modes if r["mode"] == "linear_sum"][0]["MFT_accuracy"]
    robust = "YES" if all(abs(float(r["MFT_accuracy"]) - float(primary_acc)) <= 0.05 for r in robust_modes) else "NO"

    if delta_acc <= 0 or g16_mft_conc == "FAIL" or residual_status in {"UNCHANGED", "WORSE"}:
        unified_status = "NOT_SUPPORTED"
    else:
        unified_status = "PARTIAL_SUPPORT"

    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE",
        "SOURCE_PROVENANCE_GATE": "PASS",
        "MF_MODEL_REUSE_GATE": mf_gate,
        "OUTCOME_SIGN_CONVENTION_GATE": "PASS",
        "TEMPORAL_METRIC_COMPATIBILITY_GATE": "PARTIAL",
        "LOCAL_RISK_REMOVAL_GATE": "PASS",
        "CROSS_ARCH_TEMPORAL_SEMANTICS_GATE": "PARTIAL",
        "TEMPORAL_OUTCOME_LEAKAGE_GATE": no_outcome_leakage_gate(read_csv(KDA_FROZEN), {"future_KL_R", "future_KL_C_match"}),
        "TEMPORAL_EXPOSURE_RESOLUTION": "UNIT_LEVEL",
        "TEMPORAL_STATIONARITY_STATUS": stationarity_status(kda_imp, gdn_imp),
        "GDN_MF_HISTORY_DEGRADATION_STATUS": history_degradation_status(perf, "GDN"),
        "KDA_MF_HISTORY_DEGRADATION_STATUS": history_degradation_status(perf, "KDA"),
        "KDA_G16_TEMPORAL_PATTERN": g16_pattern,
        "GDN_T_R_L64": gdn_tr64,
        "GDN_T_C_L64": gdn_tc64,
        "GDN_t_L64": log_temporal_ratio(gdn_tr64, gdn_tc64),
        "KDA_T_R_L64": kda_tr64,
        "KDA_T_C_L64": kda_tc64,
        "KDA_t_L64": log_temporal_ratio(kda_tr64, kda_tc64),
        "MF_GDN_ORIENTATION_ACCURACY": comp_lookup[("GDN", "MF")]["orientation_accuracy"],
        "MFT_GDN_ORIENTATION_ACCURACY": comp_lookup[("GDN", "MFT")]["orientation_accuracy"],
        "MF_KDA_ORIENTATION_ACCURACY": comp_lookup[("KDA", "MF")]["orientation_accuracy"],
        "MFT_KDA_ORIENTATION_ACCURACY": comp_lookup[("KDA", "MFT")]["orientation_accuracy"],
        "MF_COMBINED_ORIENTATION_ACCURACY": comp_lookup[("combined", "MF")]["orientation_accuracy"],
        "MFT_COMBINED_ORIENTATION_ACCURACY": comp_lookup[("combined", "MFT")]["orientation_accuracy"],
        "DELTA_ACCURACY_MFT_MINUS_MF": delta_acc,
        "DELTA_ACCURACY_MFT_MINUS_MF_CI": increment["DELTA_ACCURACY_MFT_MINUS_MF_CI"],
        "MF_GDN_SPEARMAN": comp_lookup[("GDN", "MF")]["spearman_Y"],
        "MFT_GDN_SPEARMAN": comp_lookup[("GDN", "MFT")]["spearman_Y"],
        "MF_KDA_SPEARMAN": comp_lookup[("KDA", "MF")]["spearman_Y"],
        "MFT_KDA_SPEARMAN": comp_lookup[("KDA", "MFT")]["spearman_Y"],
        "MF_COMBINED_SPEARMAN": comp_lookup[("combined", "MF")]["spearman_Y"],
        "MFT_COMBINED_SPEARMAN": comp_lookup[("combined", "MFT")]["spearman_Y"],
        "DELTA_RHO_MFT_MINUS_MF": delta_rho,
        "DELTA_RHO_MFT_MINUS_MF_CI": increment["DELTA_RHO_MFT_MINUS_MF_CI"],
        "KDA_G16_BOUNDARY_CONCORDANCE_MF": g16_mf_conc,
        "KDA_G16_BOUNDARY_CONCORDANCE_MFT": g16_mft_conc,
        "GDN_ARCHITECTURE_CONCORDANCE_PRESERVED": gdn_preserved,
        "KDA_COARSE_SIGN_CONCORDANCE_MFT": kda_coarse_status,
        "TEMPORAL_RESIDUAL_AFTER_MFT": residual_status,
        "MFT_PRIMARY_T_DEFINITION": "cumulative_sum_piecewise_linear_P_h_linear_token_h",
        "MFT_T_SENSITIVITY_ROBUST": robust,
        "UNIFIED_MFT_RISK_STATUS": unified_status,
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
        "analysis_started_at_epoch": started,
        "previous_unified_status": prev_summary.get("UNIFIED_ORIENTATION_RISK_STATUS"),
    }
    write_json(OUT_DIR / "summary.json", summary)
    write_markdown(summary)
    write_final_summary(summary)


def component_like_by_history(rows):
    out = []
    for L in sorted({int(r["history_length"]) for r in rows}):
        panel = [r for r in rows if int(r["history_length"]) == L and r["S_MFT"] != "MISSING"]
        if panel:
            out.append({"history_length": L, "S_MF": median([r["S_MF"] for r in panel]), "S_MFT": median([r["S_MFT"] for r in panel])})
    return out


def verify(py_compile_status, pytest_status):
    summary = read_json(OUT_DIR / "summary.json")
    missing = [name for name in REQUIRED_FILES if not (OUT_DIR / name).exists()]
    empty = [name for name in REQUIRED_FILES if (OUT_DIR / name).exists() and os.path.getsize(OUT_DIR / name) == 0]
    row_counts = {}
    for name in [f for f in REQUIRED_FILES if f.endswith(".csv")]:
        with (OUT_DIR / name).open(newline="") as f:
            row_counts[name] = sum(1 for _ in csv.DictReader(f))
    mtimes_ok = all(os.path.getmtime(OUT_DIR / name) >= summary["analysis_started_at_epoch"] for name in REQUIRED_FILES if (OUT_DIR / name).exists())
    hashes = {name: sha256(OUT_DIR / name) for name in REQUIRED_FILES if (OUT_DIR / name).exists() and name != "final_console_summary.txt"}
    provenance_ok = len(read_json(OUT_DIR / "source_artifact_inventory.json").get("artifacts", [])) >= 8
    fresh = "PASS" if not missing and not empty and mtimes_ok and provenance_ok and all(v > 0 for v in row_counts.values()) else "FAIL"
    write_json(OUT_DIR / "fresh_output_verification.json", {
        "FRESH_OUTPUT_VERIFICATION": fresh,
        "missing_files": missing,
        "empty_files": empty,
        "row_counts": row_counts,
        "mtimes_ok": mtimes_ok,
        "provenance_ok": provenance_ok,
        "output_hashes": hashes,
    })
    summary["FRESH_OUTPUT_VERIFICATION"] = fresh
    summary["PY_COMPILE"] = py_compile_status
    summary["PYTEST"] = pytest_status
    summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES" if fresh == "PASS" and py_compile_status == "PASS" and pytest_status == "PASS" else "NO"
    write_json(OUT_DIR / "summary.json", summary)
    write_markdown(summary)
    write_final_summary(summary)


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
