#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


TASK = "LING_KDA_GRANULARITY_ORIENTATION_ATTENUATION_AUDIT_V1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
OUT_DIR = REPO / "results" / TASK
RC_RUN_DIR = REPO / "runs" / "ling_kda_rc_granularity_prospective_confirmation_v1"
G64_RUN_DIR = REPO / "runs" / "ling_kda_g64_prospective_confirmation_v1"

GROUP_SIZES = [128, 64, 32, 16]
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
N_COMMON_UNITS = 36
N_EXPECTED_PAIRED_EFFECT_ROWS = N_COMMON_UNITS * len(GROUP_SIZES) * len(HISTORY_LENGTHS)
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260912
NEAR_TIE = 1e-12
MONOTONIC_TOL = 1e-12
RATIO_EPS = 1e-15

REQUIRED_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "source_artifact_inventory.json",
    "common36_manifest.csv",
    "common36_manifest.json",
    "common36_per_unit_per_history_effects.csv",
    "common36_unit_level_group_effects.csv",
    "common36_group_summary.csv",
    "paired_granularity_contrasts.csv",
    "history_conditioned_attenuation.csv",
    "attenuation_ratios.json",
    "attenuation_robustness.csv",
    "structural_heterogeneity.csv",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

REQUIRED_CONSOLE_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "COMMON_COHORT_GATE",
    "N_COMMON_UNITS",
    "N_EXPECTED_PAIRED_EFFECT_ROWS",
    "N_VALID_PAIRED_EFFECT_ROWS",
    "COMMON36_G128_EFFECT",
    "COMMON36_G128_BOOTSTRAP_CI",
    "COMMON36_G64_EFFECT",
    "COMMON36_G64_BOOTSTRAP_CI",
    "COMMON36_G32_EFFECT",
    "COMMON36_G32_BOOTSTRAP_CI",
    "COMMON36_G16_EFFECT",
    "COMMON36_G16_BOOTSTRAP_CI",
    "D128_TO_D64",
    "D128_TO_D64_CI",
    "D64_TO_D32",
    "D64_TO_D32_CI",
    "D32_TO_D16",
    "D32_TO_D16_CI",
    "D128_TO_D16",
    "D128_TO_D16_CI",
    "N_DELTA128_GT_DELTA16",
    "N_STRICT_MONOTONIC_UNITS",
    "N_NONINCREASING_WITH_TOLERANCE",
    "MEDIAN_LOG2_GROUP_SLOPE",
    "LOG2_GROUP_SLOPE_CI",
    "A64",
    "A32",
    "A16",
    "HISTORY_ATTENUATION_CLASSIFICATION",
    "STRUCTURAL_HETEROGENEITY_CLASSIFICATION",
    "LEAVE_ONE_UNIT_OUT_SIGN_STABLE",
    "LEAVE_ONE_LAYER_OUT_SIGN_STABLE",
    "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE",
    "ORIENTATION_ATTENUATION_STATUS",
    "FINAL_CLASSIFICATION",
    "FRESH_OUTPUT_VERIFICATION",
    "PY_COMPILE",
    "PYTEST",
    "BASH_SYNTAX_CHECK",
    "GPU_RUNS_LAUNCHED",
    "GPU_RELEASE_VERIFICATION",
    "METHOD_DESIGN_READY",
    "NO_NEXT_EXPERIMENT_LAUNCHED",
    "CURRENT_TASK_COMPLETED_AND_STOPPED",
    "NEXT_RECOMMENDED_TASK",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x):
    return float(x) if finite(x) else None


def mean(xs):
    vals = [float(x) for x in xs if finite(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite(x)]
    return statistics.median(vals) if vals else None


def trimmed_mean(xs, proportion=0.10):
    vals = sorted(float(x) for x in xs if finite(x))
    if not vals:
        return None
    trim = int(len(vals) * proportion)
    trimmed = vals[trim:len(vals) - trim] if trim and len(vals) > 2 * trim else vals
    return sum(trimmed) / len(trimmed)


def bootstrap_ci(vals, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, statistic=median):
    vals = [float(x) for x in vals if finite(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(int(reps)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        out.append(float(statistic(sample)))
    out.sort()
    return [out[int(0.025 * (len(out) - 1))], out[int(0.975 * (len(out) - 1))]]


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_config(row):
    cfg = row.get("config")
    if cfg:
        return str(cfg).upper()
    if row.get("orientation") and row.get("group_size"):
        return f"{str(row['orientation']).upper()}{int(float(row['group_size']))}"
    return "UNKNOWN"


def normalized_unit(unit):
    row = dict(unit)
    row["unit_id"] = str(row["unit_id"])
    row["prompt_id"] = str(row.get("prompt_id", row.get("problem_id", row["unit_id"].split("|")[0])))
    row["sample_id"] = str(row.get("sample_id", row["prompt_id"]))
    row["layer"] = str(row.get("layer", "ALL_KDA"))
    row["head"] = str(row.get("head", "ALL"))
    row["t0"] = int(float(row.get("t0", row["unit_id"].split("|")[-1])))
    row["common_unit_key"] = "|".join([
        row["unit_id"],
        row["sample_id"],
        row["prompt_id"],
        str(row.get("token_sequence_sha256", "")),
        row["layer"],
        row["head"],
        str(row["t0"]),
    ])
    return row


def load_manifest_units():
    manifest = json.loads((RC_RUN_DIR / "primary_36_unit_manifest.json").read_text(encoding="utf-8"))
    return [normalized_unit(u) for u in manifest["units"]]


def load_raw_rows():
    rows = []
    rows.extend(iter_jsonl(RC_RUN_DIR / "results_primary.jsonl"))
    rows.extend(iter_jsonl(G64_RUN_DIR / "results.jsonl"))
    return rows


def normalize_result_row(row):
    cfg = row_config(row)
    group = int(float(row.get("group_size", cfg[1:])))
    return {
        "unit_id": str(row["unit_id"]),
        "prompt_id": str(row.get("prompt_id", row.get("problem_id", str(row["unit_id"]).split("|")[0]))),
        "sample_id": str(row.get("sample_id", row.get("prompt_id", row.get("problem_id", str(row["unit_id"]).split("|")[0])))),
        "layer": str(row.get("layer", "ALL_KDA")),
        "head": str(row.get("head", "ALL")),
        "t0": int(float(row.get("t0", str(row["unit_id"]).split("|")[-1]))),
        "group_size": group,
        "history_length": int(float(row["history_length"])),
        "config": cfg,
        "future_kl_auc": as_float(row.get("future_kl_auc")),
    }


def build_common_effect_tables(units, rows):
    units = [normalized_unit(u) for u in units]
    unit_ids = [u["unit_id"] for u in units]
    unit_meta = {u["unit_id"]: u for u in units}
    normalized = []
    for row in rows:
        nr = normalize_result_row(row)
        if nr["unit_id"] in unit_meta and nr["group_size"] in GROUP_SIZES and nr["history_length"] in HISTORY_LENGTHS and nr["config"] in {f"R{nr['group_size']}", f"C{nr['group_size']}"}:
            normalized.append(nr)
    by = {(r["unit_id"], r["group_size"], r["history_length"], r["config"]): r for r in normalized}
    per_history = []
    missing = []
    for unit_id in unit_ids:
        meta = unit_meta[unit_id]
        for group in GROUP_SIZES:
            for history in HISTORY_LENGTHS:
                r = by.get((unit_id, group, history, f"R{group}"))
                c = by.get((unit_id, group, history, f"C{group}"))
                if not r or not c or r["future_kl_auc"] is None or c["future_kl_auc"] is None:
                    missing.append({"unit_id": unit_id, "group_size": group, "history_length": history})
                    continue
                delta = c["future_kl_auc"] - r["future_kl_auc"]
                per_history.append({
                    "unit_id": unit_id,
                    "prompt_id": meta["prompt_id"],
                    "sample_id": meta["sample_id"],
                    "layer": meta["layer"],
                    "head": meta["head"],
                    "t0": meta["t0"],
                    "group_size": group,
                    "history_length": history,
                    "KL_R": r["future_kl_auc"],
                    "KL_C": c["future_kl_auc"],
                    "delta_C_minus_R": delta,
                    "preferred_orientation": "R" if delta > NEAR_TIE else "C" if delta < -NEAR_TIE else "TIE",
                })
    by_unit_group = defaultdict(list)
    for row in per_history:
        by_unit_group[(row["unit_id"], int(row["group_size"]))].append(row["delta_C_minus_R"])
    unit_group = []
    for unit_id in unit_ids:
        meta = unit_meta[unit_id]
        out = {
            "unit_id": unit_id,
            "prompt_id": meta["prompt_id"],
            "sample_id": meta["sample_id"],
            "layer": meta["layer"],
            "head": meta["head"],
            "t0": meta["t0"],
        }
        complete = True
        for group in GROUP_SIZES:
            vals = by_unit_group.get((unit_id, group), [])
            if len(vals) != len(HISTORY_LENGTHS):
                complete = False
            out[f"delta_g{group}"] = mean(vals) if len(vals) == len(HISTORY_LENGTHS) else None
        if complete:
            unit_group.append(out)
    return {
        "COMMON_COHORT_GATE": "PASS" if len(units) == N_COMMON_UNITS and not missing and len(per_history) == N_EXPECTED_PAIRED_EFFECT_ROWS and len(unit_group) == N_COMMON_UNITS else "FAIL",
        "units": units,
        "per_history": per_history,
        "unit_group": unit_group,
        "missing_cells": missing,
    }


def compute_group_summary(unit_rows, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    out = {}
    rows = []
    for group in GROUP_SIZES:
        key = f"delta_g{group}"
        vals = [r[key] for r in unit_rows]
        ci = bootstrap_ci(vals, bootstrap_reps, bootstrap_seed + group, median)
        row = {
            "group_size": group,
            "median_effect": median(vals),
            "mean_effect": mean(vals),
            "trimmed_mean_effect": trimmed_mean(vals),
            "R_preferred_unit_count": sum(1 for x in vals if x > NEAR_TIE),
            "C_preferred_unit_count": sum(1 for x in vals if x < -NEAR_TIE),
            "tie_count": sum(1 for x in vals if abs(x) <= NEAR_TIE),
            "bootstrap_ci_low": ci[0] if ci else None,
            "bootstrap_ci_high": ci[1] if ci else None,
            "bootstrap_ci": ci,
            "N_UNITS": len(vals),
        }
        out[str(group)] = row
        rows.append(row)
    return out, rows


def contrast_values(unit_rows, coarse, fine):
    return [float(r[f"delta_g{coarse}"]) - float(r[f"delta_g{fine}"]) for r in unit_rows]


def compute_paired_contrasts(unit_rows, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    defs = [
        ("D128_TO_D64", 128, 64),
        ("D64_TO_D32", 64, 32),
        ("D32_TO_D16", 32, 16),
        ("D128_TO_D16", 128, 16),
    ]
    out = {}
    for idx, (name, coarse, fine) in enumerate(defs):
        vals = contrast_values(unit_rows, coarse, fine)
        ci = bootstrap_ci(vals, bootstrap_reps, bootstrap_seed + 100 + idx, median)
        out[name] = {
            "contrast": name,
            "coarse_group": coarse,
            "fine_group": fine,
            "median_contrast": median(vals),
            "mean_contrast": mean(vals),
            "trimmed_mean_contrast": trimmed_mean(vals),
            "bootstrap_ci": ci,
            "bootstrap_ci_low": ci[0] if ci else None,
            "bootstrap_ci_high": ci[1] if ci else None,
            "positive_unit_count": sum(1 for x in vals if x > NEAR_TIE),
            "negative_unit_count": sum(1 for x in vals if x < -NEAR_TIE),
            "tie_count": sum(1 for x in vals if abs(x) <= NEAR_TIE),
            "N_UNITS": len(vals),
        }
    return out


def compute_monotonicity(unit_rows, tol=MONOTONIC_TOL):
    strict = 0
    noninc = 0
    g128_gt_g16 = 0
    for row in unit_rows:
        vals = [float(row[f"delta_g{g}"]) for g in GROUP_SIZES]
        if vals[0] > vals[-1]:
            g128_gt_g16 += 1
        if vals[0] > vals[1] > vals[2] > vals[3]:
            strict += 1
        if vals[0] + tol >= vals[1] and vals[1] + tol >= vals[2] and vals[2] + tol >= vals[3]:
            noninc += 1
    return {
        "N_DELTA128_GT_DELTA16": g128_gt_g16,
        "N_STRICT_MONOTONIC_UNITS": strict,
        "N_NONINCREASING_WITH_TOLERANCE": noninc,
        "MONOTONIC_TOLERANCE": tol,
    }


def ols_slope(xs, ys):
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def compute_log2_slopes(unit_rows, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    xs = [math.log2(g) for g in GROUP_SIZES]
    slopes = []
    for row in unit_rows:
        ys = [float(row[f"delta_g{g}"]) for g in GROUP_SIZES]
        slopes.append(ols_slope(xs, ys))
    ci = bootstrap_ci(slopes, bootstrap_reps, bootstrap_seed + 300, median)
    return {
        "median_slope": median(slopes),
        "mean_slope": mean(slopes),
        "bootstrap_ci": ci,
        "bootstrap_ci_low": ci[0] if ci else None,
        "bootstrap_ci_high": ci[1] if ci else None,
        "positive_slope_unit_count": sum(1 for x in slopes if x > NEAR_TIE),
        "negative_slope_unit_count": sum(1 for x in slopes if x < -NEAR_TIE),
        "tie_slope_unit_count": sum(1 for x in slopes if abs(x) <= NEAR_TIE),
        "unit_slopes": slopes,
    }


def compute_attenuation_ratios(group_summary):
    base = as_float(group_summary["128"]["median_effect"])
    out = {"A128": 1.0, "Reduction128": 0.0, "denominator": base}
    for group in [64, 32, 16]:
        effect = as_float(group_summary[str(group)]["median_effect"])
        ratio = effect / base if base is not None and base > RATIO_EPS else None
        out[f"A{group}"] = ratio
        out[f"Reduction{group}"] = 1 - ratio if ratio is not None else None
    return out


def compute_history_conditioned(per_history, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    by = {(r["unit_id"], int(r["group_size"]), int(r["history_length"])): r for r in per_history}
    unit_ids = sorted({r["unit_id"] for r in per_history})
    rows = []
    positive = 0
    for history in HISTORY_LENGTHS:
        vals = []
        for unit_id in unit_ids:
            c = by[(unit_id, 128, history)]["delta_C_minus_R"] - by[(unit_id, 16, history)]["delta_C_minus_R"]
            vals.append(c)
        ci = bootstrap_ci(vals, bootstrap_reps, bootstrap_seed + 400 + history, median)
        if median(vals) and median(vals) > 0:
            positive += 1
        rows.append({
            "history_length": history,
            "D128_TO_D16_L": median(vals),
            "mean_contrast": mean(vals),
            "trimmed_mean_contrast": trimmed_mean(vals),
            "bootstrap_ci_low": ci[0] if ci else None,
            "bootstrap_ci_high": ci[1] if ci else None,
            "positive_unit_count": sum(1 for x in vals if x > NEAR_TIE),
            "negative_unit_count": sum(1 for x in vals if x < -NEAR_TIE),
            "sign": "POSITIVE" if median(vals) and median(vals) > 0 else "NEGATIVE" if median(vals) and median(vals) < 0 else "TIE",
        })
    early = sum(1 for r in rows[:3] if r["sign"] == "POSITIVE")
    late = sum(1 for r in rows[3:] if r["sign"] == "POSITIVE")
    if positive >= 5:
        cls = "BROAD_ACROSS_HISTORY"
    elif late >= 2 and early <= 1:
        cls = "LONG_HISTORY_DOMINANT"
    elif early >= 2 and late <= 1:
        cls = "SHORT_HISTORY_DOMINANT"
    else:
        cls = "MIXED"
    return rows, cls


def subgroup_rows(unit_rows, key):
    groups = defaultdict(list)
    for row in unit_rows:
        groups[str(row.get(key, ""))].append(float(row["delta_g128"]) - float(row["delta_g16"]))
    return groups


def compute_structural_heterogeneity(unit_rows):
    rows = []
    classifications = []
    baseline = median([float(r["delta_g128"]) - float(r["delta_g16"]) for r in unit_rows])
    for key, label in [("layer", "LAYER_CONDITIONED"), ("head", "HEAD_CONDITIONED"), ("prompt_id", "SAMPLE_CONDITIONED"), ("t0", "T0_CONDITIONED")]:
        groups = subgroup_rows(unit_rows, key)
        signs = []
        for group, vals in sorted(groups.items()):
            eff = median(vals)
            signs.append(1 if eff and eff > 0 else -1 if eff and eff < 0 else 0)
            rows.append({
                "axis": key,
                "classification_label": label,
                "group": group,
                "n_units": len(vals),
                "median_D128_TO_D16": eff,
                "mean_D128_TO_D16": mean(vals),
                "positive_unit_count": sum(1 for x in vals if x > NEAR_TIE),
                "negative_unit_count": sum(1 for x in vals if x < -NEAR_TIE),
            })
        if len(groups) > 1 and any(s <= 0 for s in signs) and baseline and baseline > 0:
            classifications.append(label)
    if not classifications:
        cls = "GLOBAL"
    elif len(classifications) == 1:
        cls = classifications[0]
    else:
        cls = "MIXED"
    return rows, cls


def leave_one_summary(unit_rows, axis):
    baseline_vals = [float(r["delta_g128"]) - float(r["delta_g16"]) for r in unit_rows]
    baseline = median(baseline_vals)
    rows = []
    groups = sorted({str(r.get(axis, "")) for r in unit_rows})
    if len(groups) <= 1:
        rows.append({
            "analysis": f"leave_one_{axis}_out",
            "omitted": groups[0] if groups else "",
            "n_units": len(unit_rows),
            "effect": baseline,
            "sign_reversed": False,
            "status": "NOT_APPLICABLE_SINGLE_SUBGROUP",
        })
        return rows
    for group in groups:
        kept = [r for r in unit_rows if str(r.get(axis, "")) != group]
        if not kept:
            continue
        vals = [float(r["delta_g128"]) - float(r["delta_g16"]) for r in kept]
        rows.append({
            "analysis": f"leave_one_{axis}_out",
            "omitted": group,
            "n_units": len(vals),
            "effect": median(vals),
            "sign_reversed": bool(baseline and baseline > 0 and median(vals) <= 0),
            "status": "EVALUATED",
        })
    return rows


def compute_robustness(unit_rows):
    rows = []
    for axis in ["unit_id", "layer", "prompt_id"]:
        rows.extend(leave_one_summary(unit_rows, axis))
    summary = {}
    for label, analysis in [("UNIT", "leave_one_unit_id_out"), ("LAYER", "leave_one_layer_out"), ("PROMPT", "leave_one_prompt_id_out")]:
        effects = [r["effect"] for r in rows if r["analysis"] == analysis]
        reversals = [r for r in rows if r["analysis"] == analysis and r["sign_reversed"]]
        summary[f"LEAVE_ONE_{label}_OUT_SIGN_STABLE"] = "PASS" if effects and not reversals else "FAIL"
        summary[f"LEAVE_ONE_{label}_OUT_MIN_EFFECT"] = min(effects) if effects else None
        summary[f"LEAVE_ONE_{label}_OUT_MAX_EFFECT"] = max(effects) if effects else None
    summary["NO_SINGLE_SUBGROUP_DRIVES_RESULT"] = "PASS" if summary["LEAVE_ONE_LAYER_OUT_SIGN_STABLE"] == "PASS" and summary["LEAVE_ONE_PROMPT_OUT_SIGN_STABLE"] == "PASS" else "FAIL"
    return rows, summary


def classify_attenuation(group_summary, contrasts, robustness, history_classification, structural_classification):
    primary = contrasts["D128_TO_D16"]
    ci = primary["bootstrap_ci"]
    adjacent_positive = sum(1 for key in ["D128_TO_D64", "D64_TO_D32", "D32_TO_D16"] if as_float(contrasts[key]["median_contrast"]) is not None and contrasts[key]["median_contrast"] > 0)
    robust_ok = (
        robustness.get("LEAVE_ONE_UNIT_OUT_SIGN_STABLE") == "PASS"
        and robustness.get("LEAVE_ONE_LAYER_OUT_SIGN_STABLE") == "PASS"
        and robustness.get("LEAVE_ONE_PROMPT_OUT_SIGN_STABLE") == "PASS"
        and robustness.get("NO_SINGLE_SUBGROUP_DRIVES_RESULT") == "PASS"
    )
    direction_ok = group_summary["128"]["median_effect"] > group_summary["16"]["median_effect"] and primary["median_contrast"] > 0
    if direction_ok and ci and ci[0] > 0 and robust_ok and adjacent_positive >= 2:
        status = "STRONG_SUPPORT"
        final = "KDA_ORIENTATION_ASYMMETRY_ATTENUATES_WITH_FINER_GRANULARITY"
        nxt = "LING_KDA_PROSPECTIVE_GRANULARITY_SOURCE_ERROR_ATTENUATION_V1"
    elif direction_ok and (not ci or ci[0] <= 0 or history_classification != "BROAD_ACROSS_HISTORY" or not robust_ok):
        status = "PARTIAL_SUPPORT"
        final = "KDA_ORIENTATION_ASYMMETRY_ATTENUATION_PARTIALLY_SUPPORTED"
        nxt = "WAIT_FOR_HUMAN_REVIEW"
    elif abs(primary["median_contrast"] or 0.0) <= NEAR_TIE:
        status = "NOT_SUPPORTED"
        final = "KDA_ORIENTATION_ATTENUATION_NOT_SUPPORTED"
        nxt = "WAIT_FOR_HUMAN_REVIEW"
    elif adjacent_positive < 2:
        status = "NON_MONOTONIC"
        final = "KDA_ORIENTATION_EFFECT_NON_MONOTONIC_ACROSS_GRANULARITY"
        nxt = "WAIT_FOR_HUMAN_REVIEW"
    else:
        status = "PARTIAL_SUPPORT"
        final = "KDA_ORIENTATION_ASYMMETRY_ATTENUATION_PARTIALLY_SUPPORTED"
        nxt = "WAIT_FOR_HUMAN_REVIEW"
    return {
        "ORIENTATION_ATTENUATION_STATUS": status,
        "FINAL_CLASSIFICATION": final,
        "NEXT_RECOMMENDED_TASK": nxt,
    }


def inventory_sources():
    paths = [
        RC_RUN_DIR / "primary_36_unit_manifest.json",
        RC_RUN_DIR / "results_primary.jsonl",
        RC_RUN_DIR / "summary.json",
        G64_RUN_DIR / "results.jsonl",
        G64_RUN_DIR / "summary.json",
    ]
    return {
        "TASK": TASK,
        "created_at": now(),
        "NEW_GPU_RUNS": 0,
        "NEW_QUANTIZED_RUNS": 0,
        "sources": [
            {
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else None,
                "sha256": file_sha256(path) if path.exists() else None,
                "mtime": path.stat().st_mtime if path.exists() else None,
            }
            for path in paths
        ],
    }


def fresh_verification(start_time, summary):
    missing = [name for name in REQUIRED_FILES if not (OUT_DIR / name).exists()]
    hashes = {name: file_sha256(OUT_DIR / name) for name in REQUIRED_FILES if (OUT_DIR / name).exists()}
    mtimes = {name: (OUT_DIR / name).stat().st_mtime for name in REQUIRED_FILES if (OUT_DIR / name).exists()}
    stale = [name for name, mtime in mtimes.items() if mtime < start_time]
    ok = (
        not missing
        and not stale
        and summary.get("COMMON_COHORT_GATE") == "PASS"
        and summary.get("N_COMMON_UNITS") == N_COMMON_UNITS
        and summary.get("N_VALID_PAIRED_EFFECT_ROWS") == N_EXPECTED_PAIRED_EFFECT_ROWS
    )
    return {
        "FRESH_OUTPUT_VERIFICATION": "PASS" if ok else "FAIL",
        "missing_files": missing,
        "stale_files": stale,
        "file_hashes": hashes,
        "file_mtimes": mtimes,
    }


def render_console_summary(summary):
    lines = []
    for key in REQUIRED_CONSOLE_KEYS:
        val = summary.get(key)
        if isinstance(val, (dict, list)):
            val = json.dumps(val, sort_keys=True, ensure_ascii=False)
        lines.append(f"{key} = {'' if val is None else val}")
    lines.append("STOP.")
    return "\n".join(lines) + "\n"


def write_report(summary):
    lines = [
        f"# {TASK}",
        "",
        "## Result",
        f"- FORMAL_STATUS: {summary['FORMAL_STATUS']}",
        f"- ORIENTATION_ATTENUATION_STATUS: {summary['ORIENTATION_ATTENUATION_STATUS']}",
        f"- FINAL_CLASSIFICATION: {summary['FINAL_CLASSIFICATION']}",
        f"- D128_TO_D16: {summary['D128_TO_D16']}",
        f"- D128_TO_D16_CI: {summary['D128_TO_D16_CI']}",
        "",
        "## Common-36 Effects",
        f"- g128: {summary['COMMON36_G128_EFFECT']} CI={summary['COMMON36_G128_BOOTSTRAP_CI']}",
        f"- g64: {summary['COMMON36_G64_EFFECT']} CI={summary['COMMON36_G64_BOOTSTRAP_CI']}",
        f"- g32: {summary['COMMON36_G32_EFFECT']} CI={summary['COMMON36_G32_BOOTSTRAP_CI']}",
        f"- g16: {summary['COMMON36_G16_EFFECT']} CI={summary['COMMON36_G16_BOOTSTRAP_CI']}",
        "",
        "## Discipline",
        "This audit is offline only. It uses existing prospective outputs, excludes historical V1 as formal evidence, and launches no GPU/model/KDA runs.",
    ]
    write_text(OUT_DIR / "report.md", "\n".join(lines) + "\n")


def run_audit(args):
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = load_manifest_units()
    rows = load_raw_rows()
    tables = build_common_effect_tables(units, rows)
    write_csv(OUT_DIR / "common36_manifest.csv", tables["units"])
    save_json(OUT_DIR / "common36_manifest.json", {"TASK": TASK, "N_COMMON_UNITS": len(tables["units"]), "units": tables["units"]})
    save_json(OUT_DIR / "source_artifact_inventory.json", inventory_sources())
    if tables["COMMON_COHORT_GATE"] != "PASS":
        summary = {
            "TASK": TASK,
            "FORMAL_STATUS": "INCOMPLETE_MISSING_COMMON_COHORT_DATA",
            "COMMON_COHORT_GATE": tables["COMMON_COHORT_GATE"],
            "N_COMMON_UNITS": len(units),
            "N_VALID_PAIRED_EFFECT_ROWS": len(tables["per_history"]),
            "GPU_RUNS_LAUNCHED": 0,
            "GPU_RELEASE_VERIFICATION": "PASS_NO_GPU_USED",
            "missing_cells": tables["missing_cells"],
        }
        save_json(OUT_DIR / "summary.json", summary)
        return summary

    write_csv(OUT_DIR / "common36_per_unit_per_history_effects.csv", tables["per_history"])
    write_csv(OUT_DIR / "common36_unit_level_group_effects.csv", tables["unit_group"])
    group_summary, group_rows = compute_group_summary(tables["unit_group"], args.bootstrap_reps, BOOTSTRAP_SEED)
    write_csv(OUT_DIR / "common36_group_summary.csv", group_rows)
    contrasts = compute_paired_contrasts(tables["unit_group"], args.bootstrap_reps, BOOTSTRAP_SEED)
    write_csv(OUT_DIR / "paired_granularity_contrasts.csv", contrasts.values())
    monotonicity = compute_monotonicity(tables["unit_group"])
    slopes = compute_log2_slopes(tables["unit_group"], args.bootstrap_reps, BOOTSTRAP_SEED)
    ratios = compute_attenuation_ratios(group_summary)
    save_json(OUT_DIR / "attenuation_ratios.json", ratios)
    history_rows, history_cls = compute_history_conditioned(tables["per_history"], args.bootstrap_reps, BOOTSTRAP_SEED)
    write_csv(OUT_DIR / "history_conditioned_attenuation.csv", history_rows)
    heterogeneity_rows, structural_cls = compute_structural_heterogeneity(tables["unit_group"])
    write_csv(OUT_DIR / "structural_heterogeneity.csv", heterogeneity_rows)
    robustness_rows, robustness = compute_robustness(tables["unit_group"])
    write_csv(OUT_DIR / "attenuation_robustness.csv", robustness_rows)
    classification = classify_attenuation(group_summary, contrasts, robustness, history_cls, structural_cls)
    bootstrap = {
        "BOOTSTRAP_REPS": args.bootstrap_reps,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit",
        "group_summary": group_summary,
        "paired_contrasts": contrasts,
        "log2_group_slope": slopes,
        "history_conditioned": history_rows,
    }
    save_json(OUT_DIR / "bootstrap_results.json", bootstrap)

    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE",
        "COMMON_COHORT_GATE": tables["COMMON_COHORT_GATE"],
        "N_COMMON_UNITS": len(tables["unit_group"]),
        "N_EXPECTED_PAIRED_EFFECT_ROWS": N_EXPECTED_PAIRED_EFFECT_ROWS,
        "N_VALID_PAIRED_EFFECT_ROWS": len(tables["per_history"]),
        "COMMON36_G128_EFFECT": group_summary["128"]["median_effect"],
        "COMMON36_G128_BOOTSTRAP_CI": group_summary["128"]["bootstrap_ci"],
        "COMMON36_G64_EFFECT": group_summary["64"]["median_effect"],
        "COMMON36_G64_BOOTSTRAP_CI": group_summary["64"]["bootstrap_ci"],
        "COMMON36_G32_EFFECT": group_summary["32"]["median_effect"],
        "COMMON36_G32_BOOTSTRAP_CI": group_summary["32"]["bootstrap_ci"],
        "COMMON36_G16_EFFECT": group_summary["16"]["median_effect"],
        "COMMON36_G16_BOOTSTRAP_CI": group_summary["16"]["bootstrap_ci"],
        "D128_TO_D64": contrasts["D128_TO_D64"]["median_contrast"],
        "D128_TO_D64_CI": contrasts["D128_TO_D64"]["bootstrap_ci"],
        "D64_TO_D32": contrasts["D64_TO_D32"]["median_contrast"],
        "D64_TO_D32_CI": contrasts["D64_TO_D32"]["bootstrap_ci"],
        "D32_TO_D16": contrasts["D32_TO_D16"]["median_contrast"],
        "D32_TO_D16_CI": contrasts["D32_TO_D16"]["bootstrap_ci"],
        "D128_TO_D16": contrasts["D128_TO_D16"]["median_contrast"],
        "D128_TO_D16_CI": contrasts["D128_TO_D16"]["bootstrap_ci"],
        **monotonicity,
        "MEDIAN_LOG2_GROUP_SLOPE": slopes["median_slope"],
        "LOG2_GROUP_SLOPE_CI": slopes["bootstrap_ci"],
        "A64": ratios["A64"],
        "A32": ratios["A32"],
        "A16": ratios["A16"],
        "HISTORY_ATTENUATION_CLASSIFICATION": history_cls,
        "STRUCTURAL_HETEROGENEITY_CLASSIFICATION": structural_cls,
        **robustness,
        **classification,
        "FRESH_OUTPUT_VERIFICATION": "NOT_RUN",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": args.bash_syntax_status,
        "GPU_RUNS_LAUNCHED": 0,
        "GPU_RELEASE_VERIFICATION": "PASS_NO_GPU_USED",
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO",
        "created_at": now(),
        "output_dir": str(OUT_DIR),
    }
    write_report(summary)
    write_text(OUT_DIR / "README.md", "\n".join([
        f"# {TASK}",
        "",
        f"- FORMAL_STATUS: {summary['FORMAL_STATUS']}",
        f"- ORIENTATION_ATTENUATION_STATUS: {summary['ORIENTATION_ATTENUATION_STATUS']}",
        f"- output_dir: {OUT_DIR}",
        "",
        "Offline paired common-36 audit. No GPU/model/KDA execution is performed.",
    ]) + "\n")
    save_json(OUT_DIR / "summary.json", summary)
    write_text(OUT_DIR / "final_console_summary.txt", render_console_summary(summary))
    verification = fresh_verification(start_time, summary)
    summary.update(verification)
    summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES" if verification["FRESH_OUTPUT_VERIFICATION"] == "PASS" else "NO"
    save_json(OUT_DIR / "summary.json", summary)
    write_text(OUT_DIR / "final_console_summary.txt", render_console_summary(summary))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--py-compile-status", default="NOT_RUN")
    parser.add_argument("--pytest-status", default="NOT_RUN")
    parser.add_argument("--bash-syntax-status", default="NOT_APPLICABLE")
    parser.add_argument("--console-summary", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out = run_audit(args)
    if args.console_summary:
        print(render_console_summary(out), end="")
    else:
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
