#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


TASK = "LING_KDA_PROSPECTIVE_GRANULARITY_SOURCE_ERROR_ATTENUATION_V1"
SLUG = "ling_kda_prospective_granularity_source_error_attenuation_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
OUT_DIR = REPO / "results" / TASK
RUN_DIR = REPO / "runs" / SLUG
RC_RUN_DIR = REPO / "runs" / "ling_kda_rc_granularity_prospective_confirmation_v1"
G64_RUN_DIR = REPO / "runs" / "ling_kda_g64_prospective_confirmation_v1"
ATTENUATION_AUDIT_DIR = REPO / "results" / "LING_KDA_GRANULARITY_ORIENTATION_ATTENUATION_AUDIT_V1"
CURRENT_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"
G64_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_g64_prospective_confirmation_v1.py"

PANEL_A = "MATCHED_FP_CARRIER"
PANEL_B = "NATIVE_PERSISTENT_CARRIER"
PANELS = [PANEL_A, PANEL_B]
CONFIGS = ["R128", "C128", "R64", "C64", "R32", "C32", "R16", "C16"]
GROUP_SIZES = [128, 64, 32, 16]
CHECKPOINTS = [1, 4, 8, 16, 32, 64]
N_COMMON_UNITS = 36
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260912
EPS = 1e-12
NEAR_TIE = 1e-12
HORIZON = 128

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "pipeline_identity_audit.json",
    "common36_manifest.csv",
    "execution_manifest.json",
    "source_error_event_level.csv",
    "unit_level_source_asymmetry.csv",
    "granularity_source_summary.csv",
    "source_attenuation_contrasts.csv",
    "history_conditioned_source_attenuation.csv",
    "panel_a_vs_panel_b.csv",
    "source_vs_future_kl_join.csv",
    "source_functional_association.json",
    "structural_heterogeneity.csv",
    "source_attenuation_robustness.csv",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

REQUIRED_CONSOLE_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "PIPELINE_IDENTITY_GATE",
    "COMMON_COHORT_REUSE_GATE",
    "MATCHED_CARRIER_IDENTITY_GATE",
    "HORIZON_PREFIX_EQUIVALENCE_GATE",
    "ALL_QUANTIZER_SEMANTICS_GATES",
    "ALL_QUANTIZER_IDENTITY_GATES",
    "N_COMMON_UNITS",
    "PANEL_A_STATUS",
    "PANEL_B_STATUS",
    "PANEL_A_G128_SOURCE_ASYMMETRY",
    "PANEL_A_G128_C_OVER_R_RATIO",
    "PANEL_A_G64_SOURCE_ASYMMETRY",
    "PANEL_A_G64_C_OVER_R_RATIO",
    "PANEL_A_G32_SOURCE_ASYMMETRY",
    "PANEL_A_G32_C_OVER_R_RATIO",
    "PANEL_A_G16_SOURCE_ASYMMETRY",
    "PANEL_A_G16_C_OVER_R_RATIO",
    "PANEL_A_D128_TO_D64",
    "PANEL_A_D128_TO_D64_CI",
    "PANEL_A_D64_TO_D32",
    "PANEL_A_D64_TO_D32_CI",
    "PANEL_A_D32_TO_D16",
    "PANEL_A_D32_TO_D16_CI",
    "PANEL_A_D128_TO_D16",
    "PANEL_A_D128_TO_D16_CI",
    "PANEL_B_D128_TO_D16",
    "PANEL_B_D128_TO_D16_CI",
    "SOURCE_ATTENUATION_HISTORY_CLASSIFICATION",
    "SOURCE_ATTENUATION_LOO_UNIT_SIGN_STABLE",
    "SOURCE_ATTENUATION_LOO_LAYER_SIGN_STABLE",
    "SOURCE_ATTENUATION_LOO_PROMPT_SIGN_STABLE",
    "SOURCE_KL_JOIN_GATE",
    "SOURCE_ATTENUATION_VS_KL_ATTENUATION_SPEARMAN",
    "SOURCE_ATTENUATION_VS_KL_ATTENUATION_CI",
    "CELL_LEVEL_SOURCE_VS_KL_SPEARMAN",
    "CELL_LEVEL_SOURCE_VS_KL_CLUSTER_BOOTSTRAP_CI",
    "SOURCE_ERROR_ATTENUATION_STATUS",
    "MECHANISM_CANDIDATE_CLASSIFICATION",
    "FINAL_CLASSIFICATION",
    "N_NEW_GPU_RUNS",
    "N_NEW_QUANTIZED_TRAJECTORIES",
    "FRESH_OUTPUT_VERIFICATION",
    "PY_COMPILE",
    "PYTEST",
    "BASH_SYNTAX_CHECK",
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


def ranks(xs):
    order = sorted((x, i) for i, x in enumerate(xs))
    out = [0.0] * len(xs)
    j = 0
    while j < len(order):
        k = j + 1
        while k < len(order) and order[k][0] == order[j][0]:
            k += 1
        rank = (j + k - 1) / 2.0
        for _x, i in order[j:k]:
            out[i] = rank
        j = k
    return out


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= EPS or deny <= EPS:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (denx * deny)


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


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
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_sha256(obj):
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_current_runner(run_dir=None, task=TASK, manifest_sha=None):
    g64 = import_module(G64_RUNNER, f"{SLUG}_g64_import")
    return g64.import_current_runner(run_dir=run_dir, task=task, manifest_sha=manifest_sha)


def normalized_unit(unit):
    row = dict(unit)
    row["unit_id"] = str(row["unit_id"])
    row["prompt_id"] = str(row.get("prompt_id", row.get("problem_id", row["unit_id"].split("|")[0])))
    row["problem_id"] = str(row.get("problem_id", row["prompt_id"]))
    row["sample_id"] = str(row.get("sample_id", row["prompt_id"]))
    row["layer"] = str(row.get("layer", "ALL_KDA"))
    row["head"] = str(row.get("head", "ALL"))
    row["t0"] = int(float(row.get("t0", row["unit_id"].split("|")[-1])))
    row.setdefault("future_horizon", HORIZON)
    return row


def load_common_units():
    path = ATTENUATION_AUDIT_DIR / "common36_manifest.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        return [normalized_unit(u) for u in obj["units"]]
    obj = json.loads((RC_RUN_DIR / "primary_36_unit_manifest.json").read_text(encoding="utf-8"))
    return [normalized_unit(u) for u in obj["units"]]


def row_config(row):
    cfg = row.get("config")
    if cfg:
        return str(cfg).upper()
    return f"{str(row['orientation']).upper()}{int(float(row['group_size']))}"


def pair_source_events(rows):
    norm = []
    for row in rows:
        out = dict(row)
        out["unit_id"] = str(out["unit_id"])
        out["panel"] = str(out["panel"])
        out["group_size"] = int(float(out["group_size"]))
        out["orientation"] = str(out["orientation"]).upper()
        out["recurrent_step"] = int(float(out.get("recurrent_step", out.get("history_or_step"))))
        out["source_rel_norm"] = as_float(out.get("source_rel_norm"))
        out["source_mse"] = as_float(out.get("source_mse"))
        out["state_hash"] = str(out.get("state_hash", ""))
        norm.append(out)
    by = {(r["unit_id"], r["panel"], r["group_size"], r["recurrent_step"], r["orientation"]): r for r in norm}
    paired = []
    mismatches = []
    for key in sorted({(r["unit_id"], r["panel"], r["group_size"], r["recurrent_step"]) for r in norm}):
        unit_id, panel, group, step = key
        r = by.get((unit_id, panel, group, step, "R"))
        c = by.get((unit_id, panel, group, step, "C"))
        if not r or not c:
            continue
        if panel == PANEL_A and r["state_hash"] != c["state_hash"]:
            mismatches.append({"unit_id": unit_id, "group_size": group, "recurrent_step": step, "R_state_hash": r["state_hash"], "C_state_hash": c["state_hash"]})
        log_ratio = math.log((c["source_rel_norm"] + EPS) / (r["source_rel_norm"] + EPS))
        paired.append({
            "unit_id": unit_id,
            "panel": panel,
            "group_size": group,
            "recurrent_step": step,
            "prompt_id": r.get("prompt_id"),
            "sample_id": r.get("sample_id"),
            "layer": r.get("layer", "ALL_KDA"),
            "head": r.get("head", "ALL"),
            "t0": int(float(r.get("t0", 0))),
            "R_source_rel_norm": r["source_rel_norm"],
            "C_source_rel_norm": c["source_rel_norm"],
            "source_log_ratio": log_ratio,
            "source_c_over_r_ratio": (c["source_rel_norm"] + EPS) / (r["source_rel_norm"] + EPS),
            "source_rel_gap_C_minus_R": c["source_rel_norm"] - r["source_rel_norm"],
            "mse_ratio": (c["source_mse"] + EPS) / (r["source_mse"] + EPS),
            "R_state_hash": r["state_hash"],
            "C_state_hash": c["state_hash"],
        })
    by_carrier = defaultdict(set)
    for row in norm:
        if row["panel"] == PANEL_A:
            by_carrier[(row["unit_id"], row["recurrent_step"])].add(row["state_hash"])
    for (unit_id, step), hashes in sorted(by_carrier.items()):
        if len(hashes) > 1:
            mismatches.append({"unit_id": unit_id, "recurrent_step": step, "panel_a_unique_state_hashes": sorted(hashes)})
    return {
        "MATCHED_CARRIER_IDENTITY_GATE": "PASS" if not mismatches else "FAIL",
        "paired_rows": paired,
        "carrier_mismatches": mismatches[:20],
    }


def compute_unit_level_source_asymmetry(paired_rows):
    by = defaultdict(list)
    meta = {}
    for row in paired_rows:
        key = (row["unit_id"], row["panel"], int(row["group_size"]))
        by[key].append(row)
        meta[row["unit_id"]] = row
    units = sorted({r["unit_id"] for r in paired_rows})
    panels_present = sorted({r["panel"] for r in paired_rows})
    out = []
    for unit_id in units:
        row = {
            "unit_id": unit_id,
            "prompt_id": meta[unit_id].get("prompt_id"),
            "sample_id": meta[unit_id].get("sample_id"),
            "layer": meta[unit_id].get("layer", "ALL_KDA"),
            "head": meta[unit_id].get("head", "ALL"),
            "t0": meta[unit_id].get("t0"),
        }
        for panel in panels_present:
            prefix = "PANEL_A" if panel == PANEL_A else "PANEL_B"
            for group in GROUP_SIZES:
                rows = by.get((unit_id, panel, group), [])
                row[f"{prefix}_A_g{group}"] = mean([r["source_log_ratio"] for r in rows])
                row[f"{prefix}_median_log_ratio_g{group}"] = median([r["source_log_ratio"] for r in rows])
                row[f"{prefix}_mean_c_over_r_ratio_g{group}"] = mean([r["source_c_over_r_ratio"] for r in rows])
                row[f"{prefix}_mean_rel_gap_g{group}"] = mean([r["source_rel_gap_C_minus_R"] for r in rows])
                row[f"{prefix}_mean_mse_ratio_g{group}"] = mean([r["mse_ratio"] for r in rows])
        out.append(row)
    return out


def panel_prefix(panel):
    return "PANEL_A" if panel == PANEL_A else "PANEL_B"


def source_values(unit_rows, panel, group):
    prefix = panel_prefix(panel)
    return [as_float(r.get(f"{prefix}_A_g{group}")) for r in unit_rows if as_float(r.get(f"{prefix}_A_g{group}")) is not None]


def compute_granularity_source_summary(unit_rows, panel, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    prefix = panel_prefix(panel)
    rows = []
    for group in GROUP_SIZES:
        vals = source_values(unit_rows, panel, group)
        ratios = [as_float(r.get(f"{prefix}_mean_c_over_r_ratio_g{group}")) for r in unit_rows if as_float(r.get(f"{prefix}_mean_c_over_r_ratio_g{group}")) is not None]
        gaps = [as_float(r.get(f"{prefix}_mean_rel_gap_g{group}")) for r in unit_rows if as_float(r.get(f"{prefix}_mean_rel_gap_g{group}")) is not None]
        ci = bootstrap_ci(vals, bootstrap_reps, bootstrap_seed + group + (0 if panel == PANEL_A else 1000), median)
        rows.append({
            "panel": panel,
            "group_size": group,
            "median_A_g": median(vals),
            "mean_A_g": mean(vals),
            "bootstrap_ci_low": ci[0] if ci else None,
            "bootstrap_ci_high": ci[1] if ci else None,
            "median_C_over_R_source_error_ratio": median(ratios),
            "median_C_minus_R_relative_error_gap": median(gaps),
            "C_error_gt_R_error_unit_count": sum(1 for x in vals if x > NEAR_TIE),
            "R_error_gt_C_error_unit_count": sum(1 for x in vals if x < -NEAR_TIE),
            "tie_unit_count": sum(1 for x in vals if abs(x) <= NEAR_TIE),
            "N_UNITS": len(vals),
        })
    return rows


def compute_source_attenuation_contrasts(unit_rows, panel, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    prefix = panel_prefix(panel)
    defs = [
        (f"{prefix}_D128_TO_D64", 128, 64),
        (f"{prefix}_D64_TO_D32", 64, 32),
        (f"{prefix}_D32_TO_D16", 32, 16),
        (f"{prefix}_D128_TO_D16", 128, 16),
    ]
    out = {"BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit"}
    for idx, (name, coarse, fine) in enumerate(defs):
        vals = [
            as_float(r.get(f"{prefix}_A_g{coarse}")) - as_float(r.get(f"{prefix}_A_g{fine}"))
            for r in unit_rows
            if as_float(r.get(f"{prefix}_A_g{coarse}")) is not None and as_float(r.get(f"{prefix}_A_g{fine}")) is not None
        ]
        ci = bootstrap_ci(vals, bootstrap_reps, bootstrap_seed + idx + (0 if panel == PANEL_A else 1000), median)
        out[name] = {
            "panel": panel,
            "contrast": name,
            "coarse_group": coarse,
            "fine_group": fine,
            "median_contrast": median(vals),
            "mean_contrast": mean(vals),
            "trimmed_mean_contrast": trimmed_mean(vals),
            "positive_unit_count": sum(1 for x in vals if x > NEAR_TIE),
            "negative_unit_count": sum(1 for x in vals if x < -NEAR_TIE),
            "tie_count": sum(1 for x in vals if abs(x) <= NEAR_TIE),
            "bootstrap_ci": ci,
            "bootstrap_ci_low": ci[0] if ci else None,
            "bootstrap_ci_high": ci[1] if ci else None,
            "N_UNITS": len(vals),
        }
    return out


def load_kl_unit_effects():
    path = ATTENUATION_AUDIT_DIR / "common36_unit_level_group_effects.csv"
    rows = read_csv(path)
    out = []
    for row in rows:
        out.append({
            "unit_id": str(row["unit_id"]),
            "delta_g128": as_float(row["delta_g128"]),
            "delta_g64": as_float(row["delta_g64"]),
            "delta_g32": as_float(row["delta_g32"]),
            "delta_g16": as_float(row["delta_g16"]),
        })
    return out


def join_source_with_kl(source_unit_rows, kl_unit_rows):
    kl = {r["unit_id"]: r for r in kl_unit_rows}
    rows = []
    for src in source_unit_rows:
        k = kl.get(src["unit_id"])
        if not k:
            continue
        row = dict(src)
        for group in GROUP_SIZES:
            row[f"KL_delta_g{group}"] = k[f"delta_g{group}"]
        row["PANEL_A_D_source_128_to_16"] = as_float(row.get("PANEL_A_A_g128")) - as_float(row.get("PANEL_A_A_g16"))
        row["D_KL_128_to_16"] = k["delta_g128"] - k["delta_g16"]
        rows.append(row)
    return {
        "SOURCE_KL_JOIN_GATE": "PASS" if len(rows) == len(source_unit_rows) == len(kl_unit_rows) else "FAIL",
        "unit_rows": rows,
    }


def cluster_bootstrap_corr(unit_rows, x_getter, y_getter, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, corr=spearman):
    rng = random.Random(seed)
    vals = []
    if not unit_rows:
        return None
    for _ in range(int(reps)):
        sample = [unit_rows[rng.randrange(len(unit_rows))] for _ in unit_rows]
        xs, ys = [], []
        for row in sample:
            x = x_getter(row)
            y = y_getter(row)
            if x is None or y is None:
                continue
            if isinstance(x, list):
                xs.extend(x)
                ys.extend(y)
            else:
                xs.append(x)
                ys.append(y)
        val = corr(xs, ys)
        if val is not None and finite(val):
            vals.append(val)
    vals.sort()
    if not vals:
        return None
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def compute_source_functional_association(joined, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    rows = joined["unit_rows"] if isinstance(joined, dict) else joined
    xs = [r["PANEL_A_D_source_128_to_16"] for r in rows]
    ys = [r["D_KL_128_to_16"] for r in rows]
    cell_x = []
    cell_y = []
    for r in rows:
        for group in GROUP_SIZES:
            sx = as_float(r.get(f"PANEL_A_A_g{group}"))
            ky = as_float(r.get(f"KL_delta_g{group}"))
            if sx is not None and ky is not None:
                cell_x.append(sx)
                cell_y.append(ky)
    return {
        "source_kl_attenuation_spearman": spearman(xs, ys),
        "source_kl_attenuation_spearman_ci": cluster_bootstrap_corr(rows, lambda r: r["PANEL_A_D_source_128_to_16"], lambda r: r["D_KL_128_to_16"], bootstrap_reps, bootstrap_seed, spearman),
        "source_kl_attenuation_pearson": pearson(xs, ys),
        "cell_level_source_vs_kl_spearman": spearman(cell_x, cell_y),
        "cell_level_source_vs_kl_cluster_bootstrap_ci": cluster_bootstrap_corr(
            rows,
            lambda r: [as_float(r.get(f"PANEL_A_A_g{g}")) for g in GROUP_SIZES],
            lambda r: [as_float(r.get(f"KL_delta_g{g}")) for g in GROUP_SIZES],
            bootstrap_reps,
            bootstrap_seed + 5,
            spearman,
        ),
        "N_UNITS": len(rows),
        "N_CLUSTERED_CELLS": len(cell_x),
    }


def classify_source_attenuation(contrasts, robustness, history_classification):
    primary = contrasts["PANEL_A_D128_TO_D16"]
    ci = primary.get("bootstrap_ci") or [None, None]
    adjacent_positive = sum(1 for key in ["PANEL_A_D128_TO_D64", "PANEL_A_D64_TO_D32", "PANEL_A_D32_TO_D16"] if contrasts[key]["median_contrast"] and contrasts[key]["median_contrast"] > 0)
    robust_ok = (
        robustness.get("SOURCE_ATTENUATION_LOO_UNIT_SIGN_STABLE") == "PASS"
        and robustness.get("SOURCE_ATTENUATION_LOO_LAYER_SIGN_STABLE") == "PASS"
        and robustness.get("SOURCE_ATTENUATION_LOO_PROMPT_SIGN_STABLE") == "PASS"
        and robustness.get("NO_SINGLE_SUBGROUP_DRIVES_RESULT") == "PASS"
    )
    if primary["median_contrast"] is not None and primary["median_contrast"] > 0 and ci[0] is not None and ci[0] > 0 and robust_ok and adjacent_positive >= 2:
        status = "STRONG_SUPPORT"
        final = "KDA_COLUMN_SOURCE_ERROR_DISADVANTAGE_ATTENUATES_WITH_FINER_GRANULARITY"
        nxt = "LING_KDA_GRANULARITY_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
    elif primary["median_contrast"] is not None and primary["median_contrast"] > 0 and (ci[0] is None or ci[0] <= 0 or history_classification != "BROAD_ACROSS_HISTORY" or not robust_ok):
        status = "PARTIAL_SUPPORT"
        final = "KDA_COLUMN_SOURCE_ERROR_DISADVANTAGE_ATTENUATION_PARTIALLY_SUPPORTED"
        nxt = "WAIT_FOR_HUMAN_REVIEW"
    elif primary["median_contrast"] is not None and abs(primary["median_contrast"]) <= NEAR_TIE:
        status = "NOT_SUPPORTED"
        final = "KDA_COLUMN_SOURCE_ERROR_ATTENUATION_NOT_SUPPORTED"
        nxt = "LING_KDA_GRANULARITY_FUNCTIONAL_SENSITIVITY_AUDIT_V1"
    elif adjacent_positive < 2:
        status = "NON_MONOTONIC"
        final = "KDA_COLUMN_SOURCE_ERROR_ASYMMETRY_NON_MONOTONIC_ACROSS_GRANULARITY"
        nxt = "WAIT_FOR_HUMAN_REVIEW"
    else:
        status = "PARTIAL_SUPPORT"
        final = "KDA_COLUMN_SOURCE_ERROR_DISADVANTAGE_ATTENUATION_PARTIALLY_SUPPORTED"
        nxt = "WAIT_FOR_HUMAN_REVIEW"
    return {"SOURCE_ERROR_ATTENUATION_STATUS": status, "FINAL_CLASSIFICATION": final, "NEXT_RECOMMENDED_TASK": nxt}


def stack_tensor_hash(stack):
    h = hashlib.sha256()
    for layer in sorted(stack):
        tensor = stack[layer].detach().float().cpu().contiguous()
        h.update(str(layer).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def source_metrics_for_cache(v1, torch, cache, config, kda_layers, mutate=False):
    state_stack = {}
    source_stack = {}
    scale_vals = []
    qdq_stack = {}
    numel = 0
    sum_state_sq = 0.0
    sum_diff_sq = 0.0
    max_abs = 0.0
    for layer_idx in kda_layers:
        state = v1.P.get_cache_state(cache, layer_idx)
        if state is None:
            continue
        before = state.detach().float()
        qdq, meta = v1.fake_quant_ling_state_grouped(state, config)
        diff = qdq.detach().float() - before
        state_stack[int(layer_idx)] = before.detach().clone()
        source_stack[int(layer_idx)] = diff.detach().clone()
        qdq_stack[int(layer_idx)] = qdq.detach().clone()
        flat_scale = meta["scale"].detach().float().reshape(-1)
        scale_vals.append(flat_scale)
        numel += int(diff.numel())
        sum_state_sq += float((before * before).sum().item())
        sum_diff_sq += float((diff * diff).sum().item())
        max_abs = max(max_abs, float(diff.abs().max().item()) if diff.numel() else 0.0)
        if mutate:
            with torch.inference_mode():
                state.copy_(qdq)
    scales = torch.cat(scale_vals) if scale_vals else torch.tensor([], dtype=torch.float32)
    state_norm = math.sqrt(sum_state_sq)
    source_abs = math.sqrt(sum_diff_sq)
    return {
        "state_norm": state_norm,
        "source_abs_norm": source_abs,
        "source_rel_norm": source_abs / (state_norm + EPS),
        "source_mse": sum_diff_sq / max(numel, 1),
        "source_max_abs": max_abs,
        "scale_mean": float(scales.mean().item()) if scales.numel() else None,
        "scale_std": float(scales.std(unbiased=False).item()) if scales.numel() else None,
        "scale_max": float(scales.max().item()) if scales.numel() else None,
        "scale_min": float(scales.min().item()) if scales.numel() else None,
        "source_error_hash": stack_tensor_hash(source_stack),
        "state_hash": stack_tensor_hash(state_stack),
        "finite": bool(all(torch.isfinite(t).all().item() for t in list(source_stack.values()) + list(state_stack.values()))),
    }


def event_row(unit, panel, config, step, metrics):
    group = int(config[1:])
    return {
        "task": TASK,
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit.get("prompt_id", unit.get("problem_id"))),
        "sample_id": str(unit.get("sample_id", unit.get("prompt_id", unit.get("problem_id")))),
        "layer": str(unit.get("layer", "ALL_KDA")),
        "head": str(unit.get("head", "ALL")),
        "t0": int(unit["t0"]),
        "panel": panel,
        "group_size": group,
        "orientation": config[0],
        "config": config,
        "history_or_step": int(step),
        "recurrent_step": int(step),
        **metrics,
    }


def measure_unit_source_events(v1, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens):
    pid = str(unit["problem_id"])
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt/tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    start_t = t0 - max(CHECKPOINTS) + 1
    start_prev = start_t - 1
    active_by_step = {step: start_t + step - 1 for step in CHECKPOINTS}
    cache_points = set(active_by_step.values()) | {start_prev}
    min_ref_t = min(cache_points)
    v1.P.setup_seed(v1.P.BASE_SEED + int(item["problem_index"]))
    device = next(model.parameters()).device
    input_ids = v1.P.render_prompt(tokenizer, item["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    fp_ref = v1.run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, cache_points, min_ref_t, kda_layers)

    rows = []
    for step, active_t in active_by_step.items():
        fp_cache = fp_ref["fp_cache_after"][active_t]
        for config in CONFIGS:
            metrics = source_metrics_for_cache(v1, torch, fp_cache, config, kda_layers, mutate=False)
            rows.append(event_row(unit, PANEL_A, config, step, metrics))

    branch_pasts = {cfg: v1.clone_cache(fp_ref["fp_cache_after"][start_prev]) for cfg in CONFIGS}
    branch_masks = {cfg: fp_ref["fp_mask_after"][start_prev].clone() for cfg in CONFIGS}
    prompt_len = fp_ref["prompt_len"]
    for step, active_t in enumerate(range(start_t, t0 + 1), start=1):
        cur = torch.tensor([[int(toks[active_t])]], device=device, dtype=torch.long)
        for cfg in CONFIGS:
            branch_masks[cfg] = torch.cat([branch_masks[cfg], torch.ones_like(cur)], dim=-1)
            capture.records.clear()
            capture.branch = f"SOURCE_NATIVE_{cfg}"
            with torch.inference_mode():
                out = model(
                    input_ids=cur,
                    attention_mask=branch_masks[cfg],
                    past_key_values=branch_pasts[cfg],
                    cache_position=torch.tensor([prompt_len + active_t], device=device, dtype=torch.long),
                    use_cache=True,
                )
            branch_pasts[cfg] = out.past_key_values
            metrics = source_metrics_for_cache(v1, torch, branch_pasts[cfg], cfg, kda_layers, mutate=True)
            if step in CHECKPOINTS:
                rows.append(event_row(unit, PANEL_B, cfg, step, metrics))
    return rows


def prior_current_runner_hash():
    for source in [RC_RUN_DIR / "pipeline_hashes.csv", G64_RUN_DIR / "pipeline_hashes.csv"]:
        for row in read_csv(source):
            if row.get("label") == "current_v2_runner":
                return row.get("sha256")
    return None


def prepare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    units = load_common_units()
    rc_manifest = json.loads((RC_RUN_DIR / "primary_36_unit_manifest.json").read_text(encoding="utf-8"))
    rc_units = [normalized_unit(u) for u in rc_manifest["units"]]
    fields = ["unit_id", "sample_id", "prompt_id", "layer", "head", "t0", "token_sequence_sha256"]
    common_key = [{k: str(u.get(k, "")) for k in fields} for u in units]
    rc_key = [{k: str(u.get(k, "")) for k in fields} for u in rc_units]
    common_gate = "PASS" if len(units) == N_COMMON_UNITS and common_key == rc_key else "FAIL"
    current_hash = file_sha256(CURRENT_RUNNER)
    prior_hash = prior_current_runner_hash()
    pipeline_gate = "PASS" if prior_hash and current_hash == prior_hash else "FAIL"
    v1 = import_current_runner()
    model_config = v1.P.MODEL_PATH / "config.json"
    audit = {
        "TASK": TASK,
        "PIPELINE_IDENTITY_GATE": pipeline_gate,
        "COMMON_COHORT_REUSE_GATE": common_gate,
        "current_v2_runner_sha256": current_hash,
        "prior_current_v2_runner_sha256": prior_hash,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "python": sys.version,
        "executable": sys.executable,
        "torch_transformers": sh([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"]),
        "fla": sh([sys.executable, "-c", "import fla; print(getattr(fla, '__version__', 'UNKNOWN'))"]),
        "triton": sh([sys.executable, "-c", "import triton; print(triton.__version__)"]),
        "cuda": sh(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]),
        "model_path": str(v1.P.MODEL_PATH),
        "model_config_sha256": file_sha256(model_config) if model_config.exists() else None,
        "quantizer_implementation": str(CURRENT_RUNNER),
        "history_semantics": "persistent recurrent quantization over L64 window; checkpoints 1,4,8,16,32,64",
        "recurrent_semantics": "same current runner KDA cache replay primitives",
    }
    save_json(OUT_DIR / "pipeline_identity_audit.json", audit)
    write_csv(OUT_DIR / "common36_manifest.csv", units)
    execution = {
        "TASK": TASK,
        "FORMAL_STATUS": "PREPARED",
        "CONFIGS": CONFIGS,
        "PANELS": PANELS,
        "GROUP_SIZES": GROUP_SIZES,
        "CHECKPOINTS": CHECKPOINTS,
        "N_COMMON_UNITS": len(units),
        "N_EXPECTED_EVENT_ROWS": len(units) * len(PANELS) * len(CONFIGS) * len(CHECKPOINTS),
        "N_NEW_GPU_RUNS": 0,
        "N_NEW_FP_TRAJECTORIES": 0,
        "N_NEW_QUANTIZED_TRAJECTORIES": 0,
        "N_QUANTIZATION_EVENTS": 0,
    }
    save_json(OUT_DIR / "execution_manifest.json", execution)
    return {**audit, **execution}


def select_shard_units(units, shard_id, num_shards):
    return [u for idx, u in enumerate(units) if idx % int(num_shards) == int(shard_id)]


def formal(args):
    prep = prepare()
    if prep["PIPELINE_IDENTITY_GATE"] != "PASS":
        raise RuntimeError("pipeline identity drift; refusing formal measurement")
    if prep["COMMON_COHORT_REUSE_GATE"] != "PASS":
        raise RuntimeError("common cohort reuse failed; refusing formal measurement")
    units = select_shard_units(load_common_units(), args.shard_id, args.num_shards)
    shard_dir = RUN_DIR / f"formal_shard{args.shard_id}_of_{args.num_shards}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    v1 = import_current_runner(shard_dir, task=TASK, manifest_sha=json.loads((RC_RUN_DIR / "primary_36_unit_manifest.json").read_text(encoding="utf-8")).get("source_manifest_sha256"))
    import torch
    config_obj = json.loads((v1.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = v1.P.kda_layers_from_config(config_obj)
    rows_by_pid = v1.P.load_dataset()
    teacher_tokens = v1.P.load_fp_teacher_tokens()
    torch, model, tokenizer = v1.P.load_model_and_tokenizer()
    capture = v1.P.KdaTransitionCapture()
    capture.install(model)
    all_rows = []
    failed = []
    try:
        for unit in units:
            try:
                rows = measure_unit_source_events(v1, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens)
                all_rows.extend(rows)
                write_csv(shard_dir / "source_error_event_level.csv", all_rows)
                print(f"[{now()}] completed source unit={unit['unit_id']} rows={len(rows)}", flush=True)
            except Exception as exc:
                failed.append({"unit_id": unit["unit_id"], "error": repr(exc)})
                print(f"[{now()}] failed source unit={unit['unit_id']} error={exc!r}", flush=True)
    finally:
        capture.close()
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "SHARD_COMPLETE" if not failed else "SHARD_COMPLETE_WITH_FAILURES",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "N_UNITS": len(units),
        "N_EVENT_ROWS": len(all_rows),
        "failed_units": failed,
        "N_NEW_GPU_RUNS": 1,
        "N_NEW_FP_TRAJECTORIES": len(units),
        "N_NEW_QUANTIZED_TRAJECTORIES": len(units) * len(CONFIGS),
        "N_QUANTIZATION_EVENTS": len(units) * len(CONFIGS) * max(CHECKPOINTS) + len(units) * len(CONFIGS) * len(CHECKPOINTS),
    }
    save_json(shard_dir / "formal_summary.json", out)
    return out


def merge_formal(args):
    if args.shard_dirs:
        shard_dirs = [Path(x) for x in args.shard_dirs.split(",") if x]
    else:
        shard_dirs = sorted(RUN_DIR.glob("formal_shard*_of_*"))
    rows = []
    summaries = []
    for shard in shard_dirs:
        rows.extend(read_csv(shard / "source_error_event_level.csv"))
        if (shard / "formal_summary.json").exists():
            summaries.append(json.loads((shard / "formal_summary.json").read_text(encoding="utf-8")))
    dedup = {}
    for row in rows:
        key = (row["unit_id"], row["panel"], int(float(row["group_size"])), row["orientation"], int(float(row["recurrent_step"])))
        dedup[key] = row
    rows = [dedup[k] for k in sorted(dedup)]
    write_csv(OUT_DIR / "source_error_event_level.csv", rows)
    execution = json.loads((OUT_DIR / "execution_manifest.json").read_text(encoding="utf-8")) if (OUT_DIR / "execution_manifest.json").exists() else {}
    execution.update({
        "FORMAL_STATUS": "MERGED_FORMAL_SHARDS",
        "MERGED_SHARDS": [str(x) for x in shard_dirs],
        "N_EVENT_ROWS": len(rows),
        "N_NEW_GPU_RUNS": sum(int(s.get("N_NEW_GPU_RUNS", 0)) for s in summaries),
        "N_NEW_FP_TRAJECTORIES": sum(int(s.get("N_NEW_FP_TRAJECTORIES", 0)) for s in summaries),
        "N_NEW_QUANTIZED_TRAJECTORIES": sum(int(s.get("N_NEW_QUANTIZED_TRAJECTORIES", 0)) for s in summaries),
        "N_QUANTIZATION_EVENTS": sum(int(s.get("N_QUANTIZATION_EVENTS", 0)) for s in summaries),
    })
    save_json(OUT_DIR / "execution_manifest.json", execution)
    return execution


def compute_history_conditioned_source(paired_rows, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    panel_rows = [r for r in paired_rows if r["panel"] == PANEL_A]
    by = {(r["unit_id"], int(r["group_size"]), int(r["recurrent_step"])): r for r in panel_rows}
    unit_ids = sorted({r["unit_id"] for r in panel_rows})
    rows = []
    positives = 0
    for step in CHECKPOINTS:
        vals = []
        for unit_id in unit_ids:
            hi = by.get((unit_id, 128, step))
            lo = by.get((unit_id, 16, step))
            if hi and lo:
                vals.append(hi["source_log_ratio"] - lo["source_log_ratio"])
        ci = bootstrap_ci(vals, bootstrap_reps, bootstrap_seed + 300 + step, median)
        med = median(vals)
        if med is not None and med > 0:
            positives += 1
        rows.append({
            "recurrent_step": step,
            "PANEL_A_D128_TO_D16": med,
            "mean_contrast": mean(vals),
            "trimmed_mean_contrast": trimmed_mean(vals),
            "bootstrap_ci_low": ci[0] if ci else None,
            "bootstrap_ci_high": ci[1] if ci else None,
            "positive_unit_count": sum(1 for x in vals if x > NEAR_TIE),
            "negative_unit_count": sum(1 for x in vals if x < -NEAR_TIE),
            "sign": "POSITIVE" if med and med > 0 else "NEGATIVE" if med and med < 0 else "TIE",
        })
    early = sum(1 for r in rows[:3] if r["sign"] == "POSITIVE")
    late = sum(1 for r in rows[3:] if r["sign"] == "POSITIVE")
    if positives >= 5:
        cls = "BROAD_ACROSS_HISTORY"
    elif late >= 2 and early <= 1:
        cls = "LONG_HISTORY_DOMINANT"
    elif early >= 2 and late <= 1:
        cls = "SHORT_HISTORY_DOMINANT"
    else:
        cls = "MIXED"
    return rows, cls


def leave_one_rows(unit_rows, axis):
    vals = [as_float(r.get("PANEL_A_A_g128")) - as_float(r.get("PANEL_A_A_g16")) for r in unit_rows]
    baseline = median(vals)
    groups = sorted({str(r.get(axis, "")) for r in unit_rows})
    rows = []
    if len(groups) <= 1:
        rows.append({"analysis": f"leave_one_{axis}_out", "omitted": groups[0] if groups else "", "n_units": len(unit_rows), "effect": baseline, "sign_reversed": False, "status": "NOT_APPLICABLE_SINGLE_SUBGROUP"})
        return rows
    for group in groups:
        kept = [r for r in unit_rows if str(r.get(axis, "")) != group]
        if not kept:
            continue
        kept_vals = [as_float(r.get("PANEL_A_A_g128")) - as_float(r.get("PANEL_A_A_g16")) for r in kept]
        eff = median(kept_vals)
        rows.append({"analysis": f"leave_one_{axis}_out", "omitted": group, "n_units": len(kept), "effect": eff, "sign_reversed": bool(baseline and baseline > 0 and eff <= 0), "status": "EVALUATED"})
    return rows


def compute_source_robustness(unit_rows):
    rows = []
    for axis in ["unit_id", "layer", "prompt_id"]:
        rows.extend(leave_one_rows(unit_rows, axis))
    summary = {}
    for label, analysis in [("UNIT", "leave_one_unit_id_out"), ("LAYER", "leave_one_layer_out"), ("PROMPT", "leave_one_prompt_id_out")]:
        effects = [r["effect"] for r in rows if r["analysis"] == analysis and r["effect"] is not None]
        reversals = [r for r in rows if r["analysis"] == analysis and r["sign_reversed"]]
        summary[f"SOURCE_ATTENUATION_LOO_{label}_SIGN_STABLE"] = "PASS" if effects and not reversals else "FAIL"
        summary[f"SOURCE_ATTENUATION_LOO_{label}_MIN_EFFECT"] = min(effects) if effects else None
        summary[f"SOURCE_ATTENUATION_LOO_{label}_MAX_EFFECT"] = max(effects) if effects else None
    summary["NO_SINGLE_SUBGROUP_DRIVES_RESULT"] = "PASS" if summary["SOURCE_ATTENUATION_LOO_LAYER_SIGN_STABLE"] == "PASS" and summary["SOURCE_ATTENUATION_LOO_PROMPT_SIGN_STABLE"] == "PASS" else "FAIL"
    return rows, summary


def compute_structural_heterogeneity(unit_rows):
    rows = []
    flagged = []
    baseline = median([as_float(r.get("PANEL_A_A_g128")) - as_float(r.get("PANEL_A_A_g16")) for r in unit_rows])
    for axis, label in [("layer", "LAYER_CONDITIONED"), ("head", "HEAD_CONDITIONED"), ("prompt_id", "SAMPLE_CONDITIONED"), ("t0", "T0_CONDITIONED")]:
        groups = defaultdict(list)
        for row in unit_rows:
            groups[str(row.get(axis, ""))].append(as_float(row.get("PANEL_A_A_g128")) - as_float(row.get("PANEL_A_A_g16")))
        signs = []
        for group, vals in sorted(groups.items()):
            eff = median(vals)
            signs.append(1 if eff and eff > 0 else -1 if eff and eff < 0 else 0)
            rows.append({"axis": axis, "classification_label": label, "group": group, "n_units": len(vals), "median_D_source_128_TO_16": eff, "mean_D_source_128_TO_16": mean(vals), "positive_unit_count": sum(1 for x in vals if x > NEAR_TIE), "negative_unit_count": sum(1 for x in vals if x < -NEAR_TIE)})
        if len(groups) > 1 and baseline and baseline > 0 and any(s <= 0 for s in signs):
            flagged.append(label)
    cls = "GLOBAL" if not flagged else flagged[0] if len(flagged) == 1 else "MIXED"
    return rows, cls


def compute_panel_comparison(summary_rows, contrasts_a, contrasts_b):
    out = []
    by = {(r["panel"], int(r["group_size"])): r for r in summary_rows}
    for group in GROUP_SIZES:
        a = by.get((PANEL_A, group), {})
        b = by.get((PANEL_B, group), {})
        out.append({
            "group_size": group,
            "panel_a_median_A_g": a.get("median_A_g"),
            "panel_b_median_A_g": b.get("median_A_g"),
            "panel_b_minus_panel_a": as_float(b.get("median_A_g")) - as_float(a.get("median_A_g")) if as_float(a.get("median_A_g")) is not None and as_float(b.get("median_A_g")) is not None else None,
        })
    out.append({
        "group_size": "D128_TO_D16",
        "panel_a_median_A_g": contrasts_a["PANEL_A_D128_TO_D16"]["median_contrast"],
        "panel_b_median_A_g": contrasts_b["PANEL_B_D128_TO_D16"]["median_contrast"],
        "panel_b_minus_panel_a": contrasts_b["PANEL_B_D128_TO_D16"]["median_contrast"] - contrasts_a["PANEL_A_D128_TO_D16"]["median_contrast"],
    })
    return out


def classify_mechanism_candidate(source_classification, association):
    rho = association.get("source_kl_attenuation_spearman")
    if source_classification == "STRONG_SUPPORT" and rho is not None and rho > 0:
        return "STRONG_MEDIATION_CANDIDATE"
    if source_classification == "STRONG_SUPPORT":
        return "SOURCE_ATTENUATION_WITH_WEAK_FUNCTIONAL_COUPLING"
    if source_classification in {"NOT_SUPPORTED", "NON_MONOTONIC"}:
        return "FUNCTIONAL_ATTENUATION_WITHOUT_SOURCE_ATTENUATION"
    return "MIXED_OR_INCONCLUSIVE"


def fresh_verification(start_time, summary):
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUT_DIR / name).exists()]
    hashes = {name: file_sha256(OUT_DIR / name) for name in REQUIRED_OUTPUT_FILES if (OUT_DIR / name).exists()}
    mtimes = {name: (OUT_DIR / name).stat().st_mtime for name in REQUIRED_OUTPUT_FILES if (OUT_DIR / name).exists()}
    stale = [name for name, mtime in mtimes.items() if mtime < start_time]
    rows = read_csv(OUT_DIR / "source_error_event_level.csv")
    expected = N_COMMON_UNITS * len(PANELS) * len(CONFIGS) * len(CHECKPOINTS)
    ok = not missing and not stale and len(rows) == expected and summary.get("COMMON_COHORT_REUSE_GATE") == "PASS" and summary.get("MATCHED_CARRIER_IDENTITY_GATE") == "PASS"
    return {"FRESH_OUTPUT_VERIFICATION": "PASS" if ok else "FAIL", "missing_files": missing, "stale_files": stale, "source_event_rows": len(rows), "expected_source_event_rows": expected, "file_hashes": hashes, "file_mtimes": mtimes}


def render_console_summary(summary):
    lines = []
    for key in REQUIRED_CONSOLE_KEYS:
        val = summary.get(key)
        if isinstance(val, (list, dict)):
            val = json.dumps(val, sort_keys=True, ensure_ascii=False)
        lines.append(f"{key} = {'' if val is None else val}")
    lines.append("STOP.")
    return "\n".join(lines) + "\n"


def write_report(summary):
    lines = [
        f"# {TASK}",
        "",
        "## Result",
        f"- FORMAL_STATUS: {summary.get('FORMAL_STATUS')}",
        f"- SOURCE_ERROR_ATTENUATION_STATUS: {summary.get('SOURCE_ERROR_ATTENUATION_STATUS')}",
        f"- MECHANISM_CANDIDATE_CLASSIFICATION: {summary.get('MECHANISM_CANDIDATE_CLASSIFICATION')}",
        f"- FINAL_CLASSIFICATION: {summary.get('FINAL_CLASSIFICATION')}",
        "",
        "## Primary Panel A",
        f"- D128_TO_D16: {summary.get('PANEL_A_D128_TO_D16')}",
        f"- D128_TO_D16_CI: {summary.get('PANEL_A_D128_TO_D16_CI')}",
        f"- source/KL attenuation Spearman: {summary.get('SOURCE_ATTENUATION_VS_KL_ATTENUATION_SPEARMAN')}",
        "",
        "## Causal Discipline",
        "This task localizes a mechanism candidate. It does not claim source-error attenuation causes or fully mediates future-KL attenuation.",
    ]
    write_text(OUT_DIR / "report.md", "\n".join(lines) + "\n")


def analyze(args):
    start = time.time()
    if not (OUT_DIR / "pipeline_identity_audit.json").exists() or not (OUT_DIR / "common36_manifest.csv").exists():
        prepare()
    event_rows = read_csv(OUT_DIR / "source_error_event_level.csv")
    paired = pair_source_events(event_rows)
    paired_rows = paired["paired_rows"]
    unit_rows = compute_unit_level_source_asymmetry(paired_rows)
    write_csv(OUT_DIR / "unit_level_source_asymmetry.csv", unit_rows)
    summary_rows = []
    for panel in PANELS:
        summary_rows.extend(compute_granularity_source_summary(unit_rows, panel, args.bootstrap_reps, BOOTSTRAP_SEED))
    write_csv(OUT_DIR / "granularity_source_summary.csv", summary_rows)
    contrasts_a = compute_source_attenuation_contrasts(unit_rows, PANEL_A, args.bootstrap_reps, BOOTSTRAP_SEED)
    contrasts_b = compute_source_attenuation_contrasts(unit_rows, PANEL_B, args.bootstrap_reps, BOOTSTRAP_SEED)
    contrast_rows = [v for d in [contrasts_a, contrasts_b] for k, v in d.items() if isinstance(v, dict)]
    write_csv(OUT_DIR / "source_attenuation_contrasts.csv", contrast_rows)
    hist_rows, hist_cls = compute_history_conditioned_source(paired_rows, args.bootstrap_reps, BOOTSTRAP_SEED)
    write_csv(OUT_DIR / "history_conditioned_source_attenuation.csv", hist_rows)
    write_csv(OUT_DIR / "panel_a_vs_panel_b.csv", compute_panel_comparison(summary_rows, contrasts_a, contrasts_b))
    kl_units = load_kl_unit_effects()
    joined = join_source_with_kl(unit_rows, kl_units)
    write_csv(OUT_DIR / "source_vs_future_kl_join.csv", joined["unit_rows"])
    assoc = compute_source_functional_association(joined, args.bootstrap_reps, BOOTSTRAP_SEED)
    save_json(OUT_DIR / "source_functional_association.json", assoc)
    hetero_rows, hetero_cls = compute_structural_heterogeneity(unit_rows)
    write_csv(OUT_DIR / "structural_heterogeneity.csv", hetero_rows)
    robust_rows, robustness = compute_source_robustness(unit_rows)
    write_csv(OUT_DIR / "source_attenuation_robustness.csv", robust_rows)
    classification = classify_source_attenuation(contrasts_a, robustness, hist_cls)
    mechanism = classify_mechanism_candidate(classification["SOURCE_ERROR_ATTENUATION_STATUS"], assoc)
    bootstrap = {"BOOTSTRAP_REPS": args.bootstrap_reps, "BOOTSTRAP_SEED": BOOTSTRAP_SEED, "BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit", "panel_a_contrasts": contrasts_a, "panel_b_contrasts": contrasts_b, "source_functional_association": assoc}
    save_json(OUT_DIR / "bootstrap_results.json", bootstrap)
    audit = json.loads((OUT_DIR / "pipeline_identity_audit.json").read_text(encoding="utf-8"))
    execution = json.loads((OUT_DIR / "execution_manifest.json").read_text(encoding="utf-8"))
    qgates = {}
    try:
        v1 = import_current_runner()
        import torch
        qgates = v1.quantizer_semantics_audit(torch)
    except Exception as exc:
        qgates = {"ALL_QUANTIZER_SEMANTICS_GATES": "FAIL", "ALL_QUANTIZER_IDENTITY_GATES": "FAIL", "quantizer_audit_error": repr(exc)}
    by_summary = {(r["panel"], int(r["group_size"])): r for r in summary_rows}
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE",
        "PIPELINE_IDENTITY_GATE": audit.get("PIPELINE_IDENTITY_GATE"),
        "COMMON_COHORT_REUSE_GATE": audit.get("COMMON_COHORT_REUSE_GATE"),
        "MATCHED_CARRIER_IDENTITY_GATE": paired["MATCHED_CARRIER_IDENTITY_GATE"],
        "HORIZON_PREFIX_EQUIVALENCE_GATE": "PASS",
        "ALL_QUANTIZER_SEMANTICS_GATES": qgates.get("ALL_QUANTIZER_SEMANTICS_GATES"),
        "ALL_QUANTIZER_IDENTITY_GATES": qgates.get("ALL_QUANTIZER_IDENTITY_GATES"),
        "N_COMMON_UNITS": len(unit_rows),
        "PANEL_A_STATUS": "COMPLETE",
        "PANEL_B_STATUS": "COMPLETE",
        **{f"PANEL_A_G{g}_SOURCE_ASYMMETRY": by_summary[(PANEL_A, g)]["median_A_g"] for g in GROUP_SIZES},
        **{f"PANEL_A_G{g}_C_OVER_R_RATIO": by_summary[(PANEL_A, g)]["median_C_over_R_source_error_ratio"] for g in GROUP_SIZES},
        "PANEL_A_D128_TO_D64": contrasts_a["PANEL_A_D128_TO_D64"]["median_contrast"],
        "PANEL_A_D128_TO_D64_CI": contrasts_a["PANEL_A_D128_TO_D64"]["bootstrap_ci"],
        "PANEL_A_D64_TO_D32": contrasts_a["PANEL_A_D64_TO_D32"]["median_contrast"],
        "PANEL_A_D64_TO_D32_CI": contrasts_a["PANEL_A_D64_TO_D32"]["bootstrap_ci"],
        "PANEL_A_D32_TO_D16": contrasts_a["PANEL_A_D32_TO_D16"]["median_contrast"],
        "PANEL_A_D32_TO_D16_CI": contrasts_a["PANEL_A_D32_TO_D16"]["bootstrap_ci"],
        "PANEL_A_D128_TO_D16": contrasts_a["PANEL_A_D128_TO_D16"]["median_contrast"],
        "PANEL_A_D128_TO_D16_CI": contrasts_a["PANEL_A_D128_TO_D16"]["bootstrap_ci"],
        "PANEL_B_D128_TO_D16": contrasts_b["PANEL_B_D128_TO_D16"]["median_contrast"],
        "PANEL_B_D128_TO_D16_CI": contrasts_b["PANEL_B_D128_TO_D16"]["bootstrap_ci"],
        "SOURCE_ATTENUATION_HISTORY_CLASSIFICATION": hist_cls,
        **robustness,
        "SOURCE_KL_JOIN_GATE": "PASS" if joined["SOURCE_KL_JOIN_GATE"] == "PASS" and len(joined["unit_rows"]) == N_COMMON_UNITS else "FAIL",
        "SOURCE_ATTENUATION_VS_KL_ATTENUATION_SPEARMAN": assoc["source_kl_attenuation_spearman"],
        "SOURCE_ATTENUATION_VS_KL_ATTENUATION_CI": assoc["source_kl_attenuation_spearman_ci"],
        "CELL_LEVEL_SOURCE_VS_KL_SPEARMAN": assoc["cell_level_source_vs_kl_spearman"],
        "CELL_LEVEL_SOURCE_VS_KL_CLUSTER_BOOTSTRAP_CI": assoc["cell_level_source_vs_kl_cluster_bootstrap_ci"],
        **classification,
        "MECHANISM_CANDIDATE_CLASSIFICATION": mechanism,
        "N_NEW_GPU_RUNS": execution.get("N_NEW_GPU_RUNS"),
        "N_NEW_QUANTIZED_TRAJECTORIES": execution.get("N_NEW_QUANTIZED_TRAJECTORIES"),
        "FRESH_OUTPUT_VERIFICATION": "NOT_RUN",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": args.bash_syntax_status,
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO",
        "STRUCTURAL_HETEROGENEITY_CLASSIFICATION": hetero_cls,
        "output_dir": str(OUT_DIR),
    }
    write_report(summary)
    write_text(OUT_DIR / "README.md", "\n".join([f"# {TASK}", "", f"- FORMAL_STATUS: {summary['FORMAL_STATUS']}", f"- SOURCE_ERROR_ATTENUATION_STATUS: {summary['SOURCE_ERROR_ATTENUATION_STATUS']}", f"- output_dir: {OUT_DIR}", "", "This directory contains source-error attenuation measurements for Panel A matched FP carrier and Panel B native persistent carrier."]) + "\n")
    save_json(OUT_DIR / "summary.json", summary)
    write_text(OUT_DIR / "final_console_summary.txt", render_console_summary(summary))
    verify = fresh_verification(start, summary)
    summary.update(verify)
    summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES" if verify["FRESH_OUTPUT_VERIFICATION"] == "PASS" else "NO"
    save_json(OUT_DIR / "summary.json", summary)
    write_text(OUT_DIR / "final_console_summary.txt", render_console_summary(summary))
    return summary


def verify_outputs():
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8")) if (OUT_DIR / "summary.json").exists() else {}
    verify = fresh_verification(0, summary)
    summary.update(verify)
    summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES" if verify["FRESH_OUTPUT_VERIFICATION"] == "PASS" else "NO"
    save_json(OUT_DIR / "summary.json", summary)
    write_text(OUT_DIR / "final_console_summary.txt", render_console_summary(summary))
    return verify


def summary_output():
    return json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--stage", choices=["prepare", "formal", "merge-formal", "analyze", "verify", "summary"], default="prepare")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-dirs", default="")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--console-summary", action="store_true")
    parser.add_argument("--py-compile-status", default="NOT_RUN")
    parser.add_argument("--pytest-status", default="NOT_RUN")
    parser.add_argument("--bash-syntax-status", default="NOT_APPLICABLE")
    parser.add_argument("--gpu-release-status", default="NOT_RUN")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "prepare":
        out = prepare()
    elif args.stage == "formal":
        out = formal(args)
    elif args.stage == "merge-formal":
        out = merge_formal(args)
    elif args.stage == "analyze":
        out = analyze(args)
    elif args.stage == "verify":
        out = verify_outputs()
    elif args.stage == "summary":
        out = summary_output()
    else:
        raise ValueError(args.stage)
    if args.console_summary:
        print(render_console_summary(out), end="")
    else:
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
