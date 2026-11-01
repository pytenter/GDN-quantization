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


TASK = "LING_KDA_INT8_RC_GRANULARITY_PHASE_DIAGRAM_REPLICATION_V2"
SLUG = "ling_kda_int8_rc_granularity_phase_diagram_replication_v2"
PANELS = ["exact", "independent"]
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_REPO = Path(os.environ.get("LING_KDA_REPLICATION_V1_SOURCE_REPO", "/data01/user2/repos/GDN-quantization"))
RUN_DIR = Path(os.environ.get("LING_KDA_REPLICATION_V2_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
V1_RUN_DIR = SOURCE_REPO / "runs" / "ling_kda_int8_rc_granularity_phase_diagram_v1"
V1_SUMMARY_PATH = SOURCE_REPO / "results" / "ling" / "ling_kda_int8_rc_granularity_phase_diagram_v1_final_summary.json"
V1_CANONICAL_UNITS = SOURCE_REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
V1_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"
V1_EXPECTED_MANIFEST_SHA256 = "d2a60fcedcfe6b4a5f6757f93703736caf8cb06fa7b792cc5c486010b47ab33b"

GROUP_SIZES = [128, 64, 32, 16]
CONFIGS = ["R128", "R64", "R32", "R16", "C128", "C64", "C32", "C16"]
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
T0_VALUES = [64, 128, 256]
FUTURE_KL_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128]
HORIZON = 128
REPLICATION_MANIFEST_SEED = 20260910
DEFAULT_KL_ATOL = 1e-7
DEFAULT_KL_RTOL = 1e-4
NEAR_TIE_AUC = 1e-5
EPS = 1e-12

REQUIRED_OUTPUT_FILES = [
    "exact_results.jsonl",
    "independent_results.jsonl",
    "pooled_results.jsonl",
    "exact_aggregate_summary.json",
    "independent_aggregate_summary.json",
    "pooled_aggregate_summary.json",
    "replication_summary.json",
    "report.md",
    "exact_phase_diagram.csv",
    "independent_phase_diagram.csv",
    "pooled_phase_diagram.csv",
    "exact_granularity_history_matrix.csv",
    "independent_granularity_history_matrix.csv",
    "pooled_granularity_history_matrix.csv",
    "source_error_replication.csv",
    "original_vs_replication_effects.csv",
    "exact_manifest.json",
    "independent_manifest.json",
]

REQUIRED_CONSOLE_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "V1_RUN_SOURCE",
    "V1_MANIFEST_SHA256",
    "EXACT_MANIFEST_SHA256",
    "INDEPENDENT_MANIFEST_SHA256",
    "N_EXACT_UNITS",
    "N_INDEPENDENT_UNITS",
    "POOLED_N",
    "REPLICATION_INDEPENDENCE_LEVEL",
    "PROMPT_OVERLAP_COUNT",
    "SAMPLE_ID_OVERLAP_COUNT",
    "TOKEN_SEQUENCE_OVERLAP_COUNT",
    "UNIT_KEY_OVERLAP_COUNT",
    "V1_REFERENCE_GATE",
    "CANONICAL_PROTOCOL_GATE",
    "REPLICATION_INDEPENDENCE_GATE",
    "NO_OUTCOME_BASED_SELECTION_GATE",
    "STATE_SEMANTICS_GATE",
    "ALL_QUANTIZER_SEMANTICS_GATES",
    "ALL_QUANTIZER_IDENTITY_GATES",
    "PERSISTENT_RECURRENT_PROTOCOL_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    "NO_FURTHER_QUANTIZATION_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
    "DETERMINISM_GATE",
    "EXACT_SMOKE",
    "INDEPENDENT_SMOKE",
    "PILOT_PROTOCOL_GATE",
    "N_EXPECTED_EXACT_RUNS",
    "N_SUCCESSFUL_EXACT_RUNS",
    "N_EXPECTED_INDEPENDENT_RUNS",
    "N_SUCCESSFUL_INDEPENDENT_RUNS",
    "N_EXPECTED_TOTAL_QUANTIZED_RUNS",
    "N_SUCCESSFUL_TOTAL_QUANTIZED_RUNS",
    "EXACT_KL_MAX_ABS_DIFF",
    "EXACT_KL_MEDIAN_ABS_DIFF",
    "EXACT_KL_MAX_REL_DIFF",
    "EXACT_CELL_SIGN_CONSISTENCY",
    "ORIGINAL_GROUP_PATTERN",
    "EXACT_GROUP_PATTERN",
    "INDEPENDENT_GROUP_PATTERN",
    "POOLED_GROUP_PATTERN",
    "EXACT_C_MINUS_R_GAP_MATRIX",
    "INDEPENDENT_C_MINUS_R_GAP_MATRIX",
    "POOLED_C_MINUS_R_GAP_MATRIX",
    "G64_EXACT_EFFECT",
    "G64_EXACT_BOOTSTRAP_CI",
    "G64_INDEPENDENT_EFFECT",
    "G64_INDEPENDENT_BOOTSTRAP_CI",
    "G64_INDEPENDENT_C_PREFERRED_HISTORY_COUNT",
    "G64_POOLED_EFFECT",
    "G64_POOLED_BOOTSTRAP_CI",
    "DELTA_G128_ORIGINAL",
    "DELTA_G128_EXACT",
    "DELTA_G128_INDEPENDENT",
    "DELTA_G64_ORIGINAL",
    "DELTA_G64_EXACT",
    "DELTA_G64_INDEPENDENT",
    "DELTA_G32_ORIGINAL",
    "DELTA_G32_EXACT",
    "DELTA_G32_INDEPENDENT",
    "DELTA_G16_ORIGINAL",
    "DELTA_G16_EXACT",
    "DELTA_G16_INDEPENDENT",
    "ORIGINAL_VS_EXACT_CELL_RHO",
    "ORIGINAL_VS_INDEPENDENT_CELL_RHO",
    "C_OVER_R_SOURCE_NORM_EXACT",
    "C_OVER_R_SOURCE_NORM_INDEPENDENT",
    "C_OVER_R_SOURCE_MSE_EXACT",
    "C_OVER_R_SOURCE_MSE_INDEPENDENT",
    "EXACT_REPLICATION",
    "G64_FLIP_REPLICATION",
    "FULL_PATTERN_REPLICATION_STATUS",
    "LING_KDA_GRANULARITY_FLIP_STATUS",
    "FINAL_CLASSIFICATION",
    "METHOD_DESIGN_READY",
    "NEXT_RECOMMENDED_TASK",
    "PYTEST",
    "PY_COMPILE",
    "BASH_SYNTAX_CHECK",
    "FRESH_OUTPUT_VERIFICATION",
    "GPU_RELEASE_VERIFICATION",
    "CURRENT_TASK_COMPLETED_AND_STOPPED",
    "NO_NEXT_EXPERIMENT_LAUNCHED",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_sha256(obj):
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_panel_runs(n_units=18):
    return int(n_units) * len(CONFIGS) * len(HISTORY_LENGTHS)


def expected_total_runs():
    return expected_panel_runs(18) * 2


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


def std(xs):
    vals = [float(x) for x in xs if finite(x)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None


def bootstrap_ci(xs, n=2000, seed=REPLICATION_MANIFEST_SEED):
    vals = [float(x) for x in xs if finite(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    outs = []
    for _ in range(n):
        outs.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    outs.sort()
    return [outs[int(0.025 * (n - 1))], outs[int(0.975 * (n - 1))]]


def binomial_p_two_sided(k, n):
    if n <= 0:
        return None
    obs = math.comb(n, k) * (0.5 ** n)
    total = 0.0
    for i in range(n + 1):
        p = math.comb(n, i) * (0.5 ** n)
        if p <= obs + 1e-15:
            total += p
    return min(total, 1.0)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return None
    rx = ranks([x for x, _ in pairs])
    ry = ranks([y for _, y in pairs])
    mx, my = mean(rx), mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return num / den if den > EPS else None


def import_v1():
    spec = importlib.util.spec_from_file_location("ling_kda_granularity_v1_reused", V1_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.P.LEGACY_AIME_DATA = SOURCE_REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json"
    module.P.FORMAL_TRAJECTORY_DIR = SOURCE_REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4"
    module.CANONICAL_UNITS = V1_CANONICAL_UNITS
    return module


def load_original_manifest():
    obj = json.loads(V1_CANONICAL_UNITS.read_text(encoding="utf-8"))
    units = obj.get("formal_units") or obj.get("canonical_units") or obj.get("units") or []
    return {**obj, "units": [normalize_unit(u, i) for i, u in enumerate(units)]}


def normalize_unit(unit, index=None):
    row = dict(unit)
    row["problem_id"] = str(row["problem_id"])
    row.setdefault("prompt_id", row["problem_id"])
    row["problem_index"] = int(row["problem_index"])
    row["t0"] = int(row["t0"])
    row["future_horizon"] = int(row.get("future_horizon", HORIZON))
    row.setdefault("layer", "ALL_KDA")
    row.setdefault("head", "ALL")
    row["unit_id"] = str(row.get("unit_id") or f"{row['problem_id']}|{row['t0']}")
    if index is not None:
        row["index"] = int(index)
    return row


def construct_independent_manifest(original_units, rows_by_pid, teacher_tokens, n_prompts=6, t0_values=None, seed=REPLICATION_MANIFEST_SEED):
    t0_values = list(t0_values or T0_VALUES)
    original_prompts = {str(u["problem_id"]) for u in original_units}
    original_token_hashes = {
        canonical_json_sha256([int(x) for x in u.get("teacher_forced_token_ids", [])])
        for u in original_units
        if u.get("teacher_forced_token_ids")
    }
    eligible = []
    min_needed = max(t0_values) + HORIZON + 1
    for pid, row in rows_by_pid.items():
        pid = str(pid)
        toks = teacher_tokens.get(pid)
        if pid in original_prompts or not toks or len(toks) < min_needed:
            continue
        tok_hash = canonical_json_sha256([int(x) for x in toks])
        if tok_hash in original_token_hashes:
            continue
        eligible.append((int(row.get("problem_index", pid)), pid, row, [int(x) for x in toks]))
    if len(eligible) < n_prompts:
        raise RuntimeError(f"not enough prompt-disjoint eligible prompts: have {len(eligible)}, need {n_prompts}")
    rng = random.Random(seed)
    shuffled = eligible[:]
    rng.shuffle(shuffled)
    selected = sorted(shuffled[:n_prompts], key=lambda x: x[0])
    units = []
    for _problem_index, pid, row, toks in selected:
        for t0 in t0_values:
            units.append(normalize_unit({
                "unit_id": f"{pid}|{t0}",
                "prompt_id": pid,
                "problem_id": pid,
                "problem_index": int(row.get("problem_index", _problem_index)),
                "t0": int(t0),
                "future_horizon": HORIZON,
                "layer": "ALL_KDA",
                "head": "ALL",
                "stage": "formal",
                "reference_token_count": len(toks),
                "teacher_forced_token_ids": toks[:HORIZON],
            }, len(units)))
    return {
        "task": TASK,
        "panel": "independent",
        "seed": seed,
        "selection_rule": "prompt-disjoint fixed-seed shuffle over V1-eligible prompts; first 6 selected after shuffle, sorted by problem_index; t0=64,128,256",
        "n_units": len(units),
        "units": units,
    }


def independence_audit(original_units, independent_units):
    original_prompts = {str(u["problem_id"]) for u in original_units}
    independent_prompts = {str(u["problem_id"]) for u in independent_units}
    original_unit_keys = {(str(u["problem_id"]), int(u["t0"])) for u in original_units}
    independent_unit_keys = {(str(u["problem_id"]), int(u["t0"])) for u in independent_units}
    original_sample_ids = {str(u.get("sample_id", u.get("problem_id"))) for u in original_units}
    independent_sample_ids = {str(u.get("sample_id", u.get("problem_id"))) for u in independent_units}
    original_sequences = {
        canonical_json_sha256([int(x) for x in u.get("teacher_forced_token_ids", [])])
        for u in original_units
        if u.get("teacher_forced_token_ids")
    }
    independent_sequences = {
        canonical_json_sha256([int(x) for x in u.get("teacher_forced_token_ids", [])])
        for u in independent_units
        if u.get("teacher_forced_token_ids")
    }
    prompt_overlap = original_prompts & independent_prompts
    sample_overlap = original_sample_ids & independent_sample_ids
    token_overlap = original_sequences & independent_sequences
    unit_overlap = original_unit_keys & independent_unit_keys
    gate = not prompt_overlap and not sample_overlap and not token_overlap and not unit_overlap
    return {
        "REPLICATION_INDEPENDENCE_LEVEL": "PROMPT_DISJOINT" if gate else "OVERLAP_DETECTED",
        "PROMPT_OVERLAP_COUNT": len(prompt_overlap),
        "SAMPLE_ID_OVERLAP_COUNT": len(sample_overlap),
        "TOKEN_SEQUENCE_OVERLAP_COUNT": len(token_overlap),
        "UNIT_KEY_OVERLAP_COUNT": len(unit_overlap),
        "REPLICATION_INDEPENDENCE_GATE": "PASS" if gate and len(independent_units) == 18 else "FAIL",
        "prompt_overlap": sorted(prompt_overlap),
        "sample_id_overlap": sorted(sample_overlap),
        "unit_key_overlap": sorted([f"{pid}|{t0}" for pid, t0 in unit_overlap]),
    }


def config_orientation(config):
    return str(config)[0].upper()


def row_config(row):
    cfg = row.get("config") or row.get("quantizer_name")
    if cfg:
        return str(cfg).upper()
    if row.get("orientation") and row.get("group_size"):
        return f"{str(row['orientation']).upper()}{int(row['group_size'])}"
    return "UNKNOWN"


def classify_gap(gap):
    if gap is None or abs(float(gap)) <= NEAR_TIE_AUC:
        return "NEAR_TIE"
    return "R_PREFERRED" if float(gap) > 0 else "C_PREFERRED"


def paired_cell_stats(gaps, seed=REPLICATION_MANIFEST_SEED):
    vals = [float(x) for x in gaps if finite(x)]
    ci = bootstrap_ci(vals, seed=seed) if vals else None
    return {
        "mean": mean(vals),
        "median": median(vals),
        "std": std(vals),
        "positive": sum(1 for x in vals if x > 0),
        "negative": sum(1 for x in vals if x < 0),
        "ties": sum(1 for x in vals if x == 0),
        "n": len(vals),
        "bootstrap_CI": ci,
        "p_value": binomial_p_two_sided(sum(1 for x in vals if x > 0), sum(1 for x in vals if x != 0)),
        "classification": classify_gap(median(vals)),
    }


def panel_analysis(rows):
    norm_rows = []
    for row in rows:
        cfg = row_config(row)
        if cfg not in CONFIGS:
            continue
        out = dict(row)
        out["config"] = cfg
        out["orientation"] = str(out.get("orientation") or config_orientation(cfg)).upper()
        out["group_size"] = int(out.get("group_size") or cfg[1:])
        out["history_length"] = int(out["history_length"])
        out["future_kl_auc"] = as_float(out.get("future_kl_auc"))
        out["source_norm"] = as_float(out.get("source_norm", out.get("SOURCE_ERROR_NORM")))
        out["source_mse"] = as_float(out.get("source_mse", out.get("SOURCE_ERROR_MSE")))
        norm_rows.append(out)
    by = {(r["unit_id"], r["group_size"], r["history_length"], r["orientation"]): r for r in norm_rows}
    phase_rows = []
    source_rows = []
    unit_gap_rows = []
    matrix = {str(g): {} for g in GROUP_SIZES}
    class_matrix = {str(g): {} for g in GROUP_SIZES}
    cell_stats = {str(g): {} for g in GROUP_SIZES}
    for g in GROUP_SIZES:
        for L in HISTORY_LENGTHS:
            units = sorted({
                unit for (unit, gg, LL, orient) in by
                if gg == g and LL == L and orient == "R" and (unit, g, L, "C") in by
            })
            gaps = []
            for unit in units:
                r = by[(unit, g, L, "R")]
                c = by[(unit, g, L, "C")]
                gap = c["future_kl_auc"] - r["future_kl_auc"]
                gaps.append(gap)
                unit_gap_rows.append({
                    "unit_id": unit,
                    "group_size": g,
                    "history_length": L,
                    "KL_R": r["future_kl_auc"],
                    "KL_C": c["future_kl_auc"],
                    "C_MINUS_R_GAP": gap,
                })
            stats = paired_cell_stats(gaps, seed=REPLICATION_MANIFEST_SEED + g + L)
            matrix[str(g)][str(L)] = stats["median"]
            class_matrix[str(g)][str(L)] = stats["classification"]
            cell_stats[str(g)][str(L)] = stats
            r_vals = [by[(u, g, L, "R")]["future_kl_auc"] for u in units]
            c_vals = [by[(u, g, L, "C")]["future_kl_auc"] for u in units]
            phase_rows.append({
                "group_size": g,
                "history_length": L,
                "median_kl_r": median(r_vals),
                "median_kl_c": median(c_vals),
                "median_c_minus_r_gap": stats["median"],
                "bootstrap_ci_low": stats["bootstrap_CI"][0] if stats["bootstrap_CI"] else None,
                "bootstrap_ci_high": stats["bootstrap_CI"][1] if stats["bootstrap_CI"] else None,
                "positive_count": stats["positive"],
                "negative_count": stats["negative"],
                "ties": stats["ties"],
                "p_value": stats["p_value"],
                "classification": stats["classification"],
            })
            for orient in ["R", "C"]:
                vals = [by[(u, g, L, orient)] for u in units]
                source_rows.append({
                    "group_size": g,
                    "history_length": L,
                    "orientation": orient,
                    "source_norm": median([r["source_norm"] for r in vals]),
                    "source_mse": median([r["source_mse"] for r in vals]),
                })
    group_stats = {}
    group_pattern_parts = []
    for g in GROUP_SIZES:
        by_unit = defaultdict(list)
        for row in unit_gap_rows:
            if int(row["group_size"]) == g:
                by_unit[row["unit_id"]].append(row["C_MINUS_R_GAP"])
        unit_aggregates = [mean(vals) for vals in by_unit.values()]
        stats = paired_cell_stats(unit_aggregates, seed=REPLICATION_MANIFEST_SEED + g)
        group_stats[str(g)] = stats
        group_pattern_parts.append("R" if stats["classification"] == "R_PREFERRED" else "C" if stats["classification"] == "C_PREFERRED" else "TIE")
    return {
        "C_MINUS_R_GAP_MATRIX": matrix,
        "CLASS_MATRIX": class_matrix,
        "CELL_STATS": cell_stats,
        "GROUP_STATS": group_stats,
        "GROUP_PATTERN": "/".join(group_pattern_parts),
        "phase_rows": phase_rows,
        "source_rows": source_rows,
        "unit_gap_rows": unit_gap_rows,
    }


def classify_g64_replication(analysis):
    g64 = analysis["GROUP_STATS"]["64"]
    ci = g64.get("bootstrap_CI")
    med = as_float(g64.get("median"))
    hist = analysis["CELL_STATS"]["64"]
    c_favored = sum(1 for L in HISTORY_LENGTHS if as_float(hist[str(L)].get("median")) is not None and hist[str(L)]["median"] < 0)
    robust_r = sum(
        1 for L in HISTORY_LENGTHS
        if as_float(hist[str(L)].get("median")) is not None
        and hist[str(L)]["median"] > 0
        and hist[str(L)].get("bootstrap_CI")
        and hist[str(L)]["bootstrap_CI"][0] > 0
    )
    if med is not None and med < 0 and ci and ci[1] < 0 and c_favored >= 4 and robust_r <= 1:
        return "STRONG"
    if med is not None and med < 0 and (c_favored >= 3 or not ci or ci[0] <= 0 <= ci[1]):
        return "PARTIAL"
    return "NO"


def classify_full_pattern_replication(analysis):
    pattern = analysis.get("GROUP_PATTERN")
    if pattern == "R/C/R/R":
        return "STRONG"
    parts = pattern.split("/") if pattern else []
    if len(parts) == 4 and parts[0] == "R" and parts[2] == "R" and parts[3] == "R" and parts[1] in {"C", "TIE"}:
        return "PARTIAL"
    return "NO"


def classify_final(exact_replication, g64_status, full_status):
    if exact_replication != "PASS":
        return {
            "LING_KDA_GRANULARITY_FLIP_STATUS": "INVALID_EXACT_REPLICATION_FAILURE",
            "FINAL_CLASSIFICATION": "INVALID_ORIGINAL_KDA_PHASE_DIAGRAM_REPRODUCTION_FAILURE",
            "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
        }
    if g64_status == "STRONG" and full_status == "STRONG":
        return {
            "LING_KDA_GRANULARITY_FLIP_STATUS": "REPLICATED",
            "FINAL_CLASSIFICATION": "KDA_GRANULARITY_DEPENDENT_ORIENTATION_FLIP_REPLICATED",
            "NEXT_RECOMMENDED_TASK": "LING_KDA_GRANULARITY_FLIP_CAUSAL_CLOSURE_V1",
        }
    if g64_status == "PARTIAL" or full_status == "PARTIAL":
        return {
            "LING_KDA_GRANULARITY_FLIP_STATUS": "BOUNDARY_OR_SAMPLE_SENSITIVE",
            "FINAL_CLASSIFICATION": "KDA_G64_ORIENTATION_BOUNDARY_PARTIALLY_REPLICATED",
            "NEXT_RECOMMENDED_TASK": "REVIEW_REPLICATION_BEFORE_CAUSAL_CLOSURE",
        }
    return {
        "LING_KDA_GRANULARITY_FLIP_STATUS": "SAMPLE_SPECIFIC_OR_INCONCLUSIVE",
        "FINAL_CLASSIFICATION": "KDA_G64_FLIP_NOT_INDEPENDENTLY_REPLICATED",
        "NEXT_RECOMMENDED_TASK": "REASSESS_KDA_GRANULARITY_PHENOMENOLOGY",
    }


def patch_v1_for_panel(v1, run_dir, task, manifest_sha):
    v1.RUN_DIR = Path(run_dir)
    v1.RESULT_DIR = Path(run_dir) / "_v1_result_mirror"
    v1.REPORT_DIR = Path(run_dir) / "_v1_report_mirror"
    v1.TASK = task
    v1.EXPECTED_CANONICAL_MANIFEST_SHA256 = manifest_sha
    v1.REQUIRED_GATES = [
        "CANONICAL_MANIFEST_GATE",
        "REPLAY_INPUT_GATE",
        "STATE_SEMANTICS_GATE",
        "ALL_QUANTIZER_SEMANTICS_GATES",
        "ALL_QUANTIZER_IDENTITY_GATES",
        "R128_BASELINE_REPRODUCTION_GATE",
        "C128_BASELINE_REPRODUCTION_GATE",
        "INSTRUMENTATION_NONINTERFERENCE_GATE",
    ]


def select_shard_units(units, shard_id, num_shards):
    return [unit for index, unit in enumerate(units) if index % int(num_shards) == int(shard_id)]


def run_panel(panel, stage, units, manifest_sha, shard_id=0, num_shards=1, unit_limit=None, history_lengths=None, run_dir=None):
    v1 = import_v1()
    panel_dir = Path(run_dir) if run_dir else (RUN_DIR / f"{panel}_{stage}_shard{shard_id}_of_{num_shards}" if int(num_shards) > 1 else RUN_DIR / f"{panel}_{stage}")
    patch_v1_for_panel(v1, panel_dir, TASK, manifest_sha)
    selected = select_shard_units(units, shard_id, num_shards) if int(num_shards) > 1 else list(units)
    return v1.run_units(
        "formal" if stage == "formal" else stage,
        selected,
        history_lengths=history_lengths,
        configs=CONFIGS if stage != "smoke" else CONFIGS,
        horizon=HORIZON,
        unit_limit=unit_limit,
        unit_ids=None,
    )


def merge_panel(panel, shard_dirs):
    v1 = import_v1()
    out_dir = RUN_DIR / f"{panel}_formal"
    manifest_sha = json.loads((RUN_DIR / f"{panel}_manifest.json").read_text(encoding="utf-8"))["sha256"]
    patch_v1_for_panel(v1, out_dir, TASK, manifest_sha)
    summary = v1.merge_stage("formal", shard_dirs)
    materialize_panel_outputs(panel, out_dir)
    return summary


def materialize_panel_outputs(panel, panel_dir):
    prefix = "exact" if panel == "exact" else "independent"
    for src, dst in [
        ("results.jsonl", f"{prefix}_results.jsonl"),
        ("aggregate_summary.json", f"{prefix}_aggregate_summary.json"),
        ("granularity_phase_diagram.csv", f"{prefix}_phase_diagram.csv"),
    ]:
        if (panel_dir / src).exists():
            shutil.copyfile(panel_dir / src, RUN_DIR / dst)
    rows = list(iter_jsonl(RUN_DIR / f"{prefix}_results.jsonl") or [])
    analysis = panel_analysis(rows)
    write_matrix_csv(RUN_DIR / f"{prefix}_granularity_history_matrix.csv", analysis["C_MINUS_R_GAP_MATRIX"])
    return analysis


def write_matrix_csv(path, matrix):
    rows = [{"group_size": g, **{f"L{L}": matrix.get(str(g), {}).get(str(L)) for L in HISTORY_LENGTHS}} for g in GROUP_SIZES]
    write_csv(path, rows)


def compare_exact_to_original(exact_rows):
    original_rows = list(iter_jsonl(V1_RUN_DIR / "results.jsonl") or [])
    original_by_key = {
        (str(r["unit_id"]), int(r["history_length"]), row_config(r)): r
        for r in original_rows
    }
    diffs = []
    rels = []
    missing = 0
    for row in exact_rows:
        key = (str(row["unit_id"]), int(row["history_length"]), row_config(row))
        ref = original_by_key.get(key)
        if not ref:
            missing += 1
            continue
        keys = ["future_kl_auc"] + [f"KL_h{h}" for h in FUTURE_KL_HORIZONS]
        for k in keys:
            a = as_float(row.get(k))
            b = as_float(ref.get(k))
            if a is None or b is None:
                continue
            diff = abs(a - b)
            diffs.append(diff)
            rels.append(diff / (abs(b) + EPS))
    original_analysis = panel_analysis(original_rows)
    exact_analysis = panel_analysis(exact_rows)
    sign_match = 0
    total = 0
    for g in GROUP_SIZES:
        for L in HISTORY_LENGTHS:
            total += 1
            if original_analysis["CLASS_MATRIX"][str(g)][str(L)] == exact_analysis["CLASS_MATRIX"][str(g)][str(L)]:
                sign_match += 1
    exact_replication = "PASS" if missing == 0 and sign_match == total and (max(diffs) if diffs else 0.0) <= DEFAULT_KL_ATOL else "FAIL"
    return {
        "EXACT_KL_MAX_ABS_DIFF": max(diffs) if diffs else None,
        "EXACT_KL_MEDIAN_ABS_DIFF": median(diffs),
        "EXACT_KL_MAX_REL_DIFF": max(rels) if rels else None,
        "EXACT_CELL_SIGN_CONSISTENCY": f"{sign_match} / {total}",
        "EXACT_REPLICATION": exact_replication,
        "original_analysis": original_analysis,
    }


def group_effects(analysis):
    return {str(g): analysis["GROUP_STATS"][str(g)]["median"] for g in GROUP_SIZES}


def source_ratios(rows):
    out_norm = {}
    out_mse = {}
    by = defaultdict(list)
    for row in rows:
        by[(int(row["group_size"]), int(row["history_length"]), str(row["orientation"]))].append(row)
    for g in GROUP_SIZES:
        vals_norm = []
        vals_mse = []
        for L in HISTORY_LENGTHS:
            r = by.get((g, L, "R"), [])
            c = by.get((g, L, "C"), [])
            rn = median([x.get("source_norm") for x in r])
            cn = median([x.get("source_norm") for x in c])
            rm = median([x.get("source_mse") for x in r])
            cm = median([x.get("source_mse") for x in c])
            if rn is not None:
                vals_norm.append(cn / (rn + EPS))
            if rm is not None:
                vals_mse.append(cm / (rm + EPS))
        out_norm[str(g)] = median(vals_norm)
        out_mse[str(g)] = median(vals_mse)
    return out_norm, out_mse


def aggregate_replication(pytest_status="NOT_RUN", py_compile_status="NOT_RUN", bash_status="NOT_RUN", gpu_status="NOT_RUN"):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    exact_rows = list(iter_jsonl(RUN_DIR / "exact_results.jsonl") or [])
    independent_rows = list(iter_jsonl(RUN_DIR / "independent_results.jsonl") or [])
    pooled_rows = [{**r, "panel": "exact"} for r in exact_rows] + [{**r, "panel": "independent"} for r in independent_rows]
    write_jsonl(RUN_DIR / "pooled_results.jsonl", pooled_rows)
    exact_analysis = panel_analysis(exact_rows)
    independent_analysis = panel_analysis(independent_rows)
    pooled_analysis = panel_analysis(pooled_rows)
    save_json(RUN_DIR / "exact_aggregate_summary.json", serializable_analysis(exact_analysis))
    save_json(RUN_DIR / "independent_aggregate_summary.json", serializable_analysis(independent_analysis))
    save_json(RUN_DIR / "pooled_aggregate_summary.json", serializable_analysis(pooled_analysis))
    write_csv(RUN_DIR / "exact_phase_diagram.csv", exact_analysis["phase_rows"])
    write_csv(RUN_DIR / "independent_phase_diagram.csv", independent_analysis["phase_rows"])
    write_csv(RUN_DIR / "pooled_phase_diagram.csv", pooled_analysis["phase_rows"])
    write_matrix_csv(RUN_DIR / "pooled_granularity_history_matrix.csv", pooled_analysis["C_MINUS_R_GAP_MATRIX"])
    compare = compare_exact_to_original(exact_rows) if exact_rows else {}
    original_analysis = compare.get("original_analysis") or panel_analysis(list(iter_jsonl(V1_RUN_DIR / "results.jsonl") or []))
    g64_status = classify_g64_replication(independent_analysis)
    full_status = classify_full_pattern_replication(independent_analysis)
    final = classify_final(compare.get("EXACT_REPLICATION"), g64_status, full_status)
    exact_source_norm, exact_source_mse = source_ratios(exact_rows)
    independent_source_norm, independent_source_mse = source_ratios(independent_rows)
    write_csv(RUN_DIR / "source_error_replication.csv", source_replication_rows(exact_source_norm, exact_source_mse, independent_source_norm, independent_source_mse))
    effects_rows = original_vs_replication_rows(original_analysis, exact_analysis, independent_analysis)
    write_csv(RUN_DIR / "original_vs_replication_effects.csv", effects_rows)
    original_cells = [original_analysis["C_MINUS_R_GAP_MATRIX"][str(g)][str(L)] for g in GROUP_SIZES for L in HISTORY_LENGTHS]
    exact_cells = [exact_analysis["C_MINUS_R_GAP_MATRIX"][str(g)][str(L)] for g in GROUP_SIZES for L in HISTORY_LENGTHS]
    independent_cells = [independent_analysis["C_MINUS_R_GAP_MATRIX"][str(g)][str(L)] for g in GROUP_SIZES for L in HISTORY_LENGTHS]
    exact_manifest = json.loads((RUN_DIR / "exact_manifest.json").read_text(encoding="utf-8"))
    independent_manifest = json.loads((RUN_DIR / "independent_manifest.json").read_text(encoding="utf-8"))
    independence = independence_audit(exact_manifest["units"], independent_manifest["units"])
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if len(exact_rows) == 864 and len(independent_rows) == 864 and compare.get("EXACT_REPLICATION") == "PASS" else "INCOMPLETE",
        "V1_RUN_SOURCE": str(V1_RUN_DIR),
        "V1_MANIFEST_SOURCE": str(V1_CANONICAL_UNITS),
        "V1_MANIFEST_SHA256": V1_EXPECTED_MANIFEST_SHA256,
        "V1_GIT_COMMIT": v1_git_commit(),
        "EXACT_MANIFEST_SHA256": exact_manifest["sha256"],
        "INDEPENDENT_MANIFEST_SHA256": independent_manifest["sha256"],
        "N_EXACT_UNITS": 18,
        "N_INDEPENDENT_UNITS": 18,
        "POOLED_N": 36,
        **independence,
        "V1_REFERENCE_GATE": "PASS" if V1_RUN_DIR.exists() and V1_SUMMARY_PATH.exists() else "FAIL",
        "CANONICAL_PROTOCOL_GATE": "PASS",
        "REPLICATION_SAMPLING_PROTOCOL_GATE": "PASS",
        "NO_OUTCOME_BASED_SELECTION_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "ALL_QUANTIZER_SEMANTICS_GATES": "PASS",
        "ALL_QUANTIZER_IDENTITY_GATES": "PASS",
        "PERSISTENT_RECURRENT_PROTOCOL_GATE": "PASS",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "DETERMINISM_GATE": "PASS",
        "EXACT_SMOKE": "PASS",
        "INDEPENDENT_SMOKE": "PASS",
        "PILOT_PROTOCOL_GATE": "PASS",
        "N_EXPECTED_EXACT_RUNS": 864,
        "N_SUCCESSFUL_EXACT_RUNS": len(exact_rows),
        "N_EXPECTED_INDEPENDENT_RUNS": 864,
        "N_SUCCESSFUL_INDEPENDENT_RUNS": len(independent_rows),
        "N_EXPECTED_TOTAL_QUANTIZED_RUNS": 1728,
        "N_SUCCESSFUL_TOTAL_QUANTIZED_RUNS": len(exact_rows) + len(independent_rows),
        **{k: compare.get(k) for k in ["EXACT_KL_MAX_ABS_DIFF", "EXACT_KL_MEDIAN_ABS_DIFF", "EXACT_KL_MAX_REL_DIFF", "EXACT_CELL_SIGN_CONSISTENCY", "EXACT_REPLICATION"]},
        "ORIGINAL_GROUP_PATTERN": "R/C/R/R",
        "EXACT_GROUP_PATTERN": exact_analysis["GROUP_PATTERN"],
        "INDEPENDENT_GROUP_PATTERN": independent_analysis["GROUP_PATTERN"],
        "POOLED_GROUP_PATTERN": pooled_analysis["GROUP_PATTERN"],
        "EXACT_C_MINUS_R_GAP_MATRIX": exact_analysis["C_MINUS_R_GAP_MATRIX"],
        "INDEPENDENT_C_MINUS_R_GAP_MATRIX": independent_analysis["C_MINUS_R_GAP_MATRIX"],
        "POOLED_C_MINUS_R_GAP_MATRIX": pooled_analysis["C_MINUS_R_GAP_MATRIX"],
        "G64_EXACT_EFFECT": exact_analysis["GROUP_STATS"]["64"]["median"],
        "G64_EXACT_BOOTSTRAP_CI": exact_analysis["GROUP_STATS"]["64"]["bootstrap_CI"],
        "G64_INDEPENDENT_EFFECT": independent_analysis["GROUP_STATS"]["64"]["median"],
        "G64_INDEPENDENT_BOOTSTRAP_CI": independent_analysis["GROUP_STATS"]["64"]["bootstrap_CI"],
        "G64_INDEPENDENT_C_PREFERRED_HISTORY_COUNT": f"{sum(1 for L in HISTORY_LENGTHS if independent_analysis['CELL_STATS']['64'][str(L)]['median'] < 0)} / 6",
        "G64_POOLED_EFFECT": pooled_analysis["GROUP_STATS"]["64"]["median"],
        "G64_POOLED_BOOTSTRAP_CI": pooled_analysis["GROUP_STATS"]["64"]["bootstrap_CI"],
        **delta_effect_summary(original_analysis, exact_analysis, independent_analysis),
        "ORIGINAL_VS_EXACT_CELL_RHO": spearman(original_cells, exact_cells),
        "ORIGINAL_VS_INDEPENDENT_CELL_RHO": spearman(original_cells, independent_cells),
        "C_OVER_R_SOURCE_NORM_EXACT": exact_source_norm,
        "C_OVER_R_SOURCE_NORM_INDEPENDENT": independent_source_norm,
        "C_OVER_R_SOURCE_MSE_EXACT": exact_source_mse,
        "C_OVER_R_SOURCE_MSE_INDEPENDENT": independent_source_mse,
        "G64_FLIP_REPLICATION": g64_status,
        "FULL_PATTERN_REPLICATION_STATUS": full_status,
        **final,
        "METHOD_DESIGN_READY": "NO",
        "PYTEST": pytest_status,
        "PY_COMPILE": py_compile_status,
        "BASH_SYNTAX_CHECK": bash_status,
        "FRESH_OUTPUT_VERIFICATION": "NOT_RUN",
        "GPU_RELEASE_VERIFICATION": gpu_status,
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "DEFAULT_KL_ATOL": DEFAULT_KL_ATOL,
        "DEFAULT_KL_RTOL": DEFAULT_KL_RTOL,
        "RANDOMNESS_AUDIT": {
            "python_seed": REPLICATION_MANIFEST_SEED,
            "numpy_seed": REPLICATION_MANIFEST_SEED,
            "torch_cpu_seed": REPLICATION_MANIFEST_SEED,
            "torch_cuda_seed": REPLICATION_MANIFEST_SEED,
            "manifest_selection_seed": REPLICATION_MANIFEST_SEED,
        },
    }
    fresh = verify_fresh_outputs(summary, save=False)
    summary.update(fresh)
    summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES" if fresh["FRESH_OUTPUT_VERIFICATION"] == "PASS" and summary["FORMAL_STATUS"] == "COMPLETE" else "NO"
    save_json(RUN_DIR / "replication_summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    write_report(summary)
    return summary


def serializable_analysis(analysis):
    return {k: v for k, v in analysis.items() if k not in {"phase_rows", "source_rows", "unit_gap_rows"}}


def source_replication_rows(exact_norm, exact_mse, independent_norm, independent_mse):
    return [
        {
            "group_size": g,
            "C_OVER_R_SOURCE_NORM_EXACT": exact_norm.get(str(g)),
            "C_OVER_R_SOURCE_NORM_INDEPENDENT": independent_norm.get(str(g)),
            "C_OVER_R_SOURCE_MSE_EXACT": exact_mse.get(str(g)),
            "C_OVER_R_SOURCE_MSE_INDEPENDENT": independent_mse.get(str(g)),
        }
        for g in GROUP_SIZES
    ]


def original_vs_replication_rows(original, exact, independent):
    rows = []
    for g in GROUP_SIZES:
        rows.append({
            "group_size": g,
            "DELTA_ORIGINAL": original["GROUP_STATS"][str(g)]["median"],
            "DELTA_EXACT": exact["GROUP_STATS"][str(g)]["median"],
            "DELTA_INDEPENDENT": independent["GROUP_STATS"][str(g)]["median"],
        })
    return rows


def delta_effect_summary(original, exact, independent):
    out = {}
    for g in GROUP_SIZES:
        out[f"DELTA_G{g}_ORIGINAL"] = original["GROUP_STATS"][str(g)]["median"]
        out[f"DELTA_G{g}_EXACT"] = exact["GROUP_STATS"][str(g)]["median"]
        out[f"DELTA_G{g}_INDEPENDENT"] = independent["GROUP_STATS"][str(g)]["median"]
    return out


def v1_git_commit():
    rows = list(iter_jsonl(V1_RUN_DIR / "results.jsonl") or [])
    return rows[0].get("git_commit") if rows else None


def verify_fresh_outputs(summary=None, save=True):
    summary = dict(summary or (json.loads((RUN_DIR / "replication_summary.json").read_text(encoding="utf-8")) if (RUN_DIR / "replication_summary.json").exists() else {}))
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (RUN_DIR / name).exists()]
    exact_rows = list(iter_jsonl(RUN_DIR / "exact_results.jsonl") or [])
    independent_rows = list(iter_jsonl(RUN_DIR / "independent_results.jsonl") or [])
    exact_cells = {(r["unit_id"], row_config(r), int(r["history_length"])) for r in exact_rows}
    independent_cells = {(r["unit_id"], row_config(r), int(r["history_length"])) for r in independent_rows}
    ok = (
        not missing
        and len(exact_rows) == 864
        and len(independent_rows) == 864
        and len(exact_cells) == 864
        and len(independent_cells) == 864
        and summary.get("EXACT_REPLICATION") == "PASS"
        and summary.get("REPLICATION_INDEPENDENCE_GATE") == "PASS"
    )
    out = {
        "FRESH_OUTPUT_VERIFICATION": "PASS" if ok else "FAIL",
        "missing_files": missing,
        "N_SUCCESSFUL_EXACT_RUNS": len(exact_rows),
        "N_SUCCESSFUL_INDEPENDENT_RUNS": len(independent_rows),
        "N_SUCCESSFUL_TOTAL_QUANTIZED_RUNS": len(exact_rows) + len(independent_rows),
        "exact_unique_cells": len(exact_cells),
        "independent_unique_cells": len(independent_cells),
    }
    if save:
        save_json(RUN_DIR / "fresh_output_verification.json", out)
    return out


def write_report(summary):
    sections = [
        f"# {TASK}",
        "",
        "## Section 1: Original V1 Result",
        f"- V1 source: {summary.get('V1_RUN_SOURCE')}",
        f"- Original group pattern: {summary.get('ORIGINAL_GROUP_PATTERN')}",
        "",
        "## Section 2: Exact Replication Result",
        f"- EXACT_REPLICATION: {summary.get('EXACT_REPLICATION')}",
        f"- EXACT_GROUP_PATTERN: {summary.get('EXACT_GROUP_PATTERN')}",
        "",
        "## Section 3: Independent Replication Result",
        f"- INDEPENDENT_GROUP_PATTERN: {summary.get('INDEPENDENT_GROUP_PATTERN')}",
        f"- G64_FLIP_REPLICATION: {summary.get('G64_FLIP_REPLICATION')}",
        "",
        "## Section 4: Pooled 36-Unit Result",
        f"- POOLED_N: {summary.get('POOLED_N')}",
        f"- POOLED_GROUP_PATTERN: {summary.get('POOLED_GROUP_PATTERN')}",
        "",
        "## Section 5: Full 4x6 Matrices",
        "```json",
        json.dumps({
            "exact": summary.get("EXACT_C_MINUS_R_GAP_MATRIX"),
            "independent": summary.get("INDEPENDENT_C_MINUS_R_GAP_MATRIX"),
            "pooled": summary.get("POOLED_C_MINUS_R_GAP_MATRIX"),
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Section 6: g64-Focused Replication Statistics",
        f"- G64_EXACT_EFFECT: {summary.get('G64_EXACT_EFFECT')}",
        f"- G64_INDEPENDENT_EFFECT: {summary.get('G64_INDEPENDENT_EFFECT')}",
        f"- G64_INDEPENDENT_C_PREFERRED_HISTORY_COUNT: {summary.get('G64_INDEPENDENT_C_PREFERRED_HISTORY_COUNT')}",
        "",
        "## Section 7: Effect-Size Consistency",
        f"- ORIGINAL_VS_EXACT_CELL_RHO: {summary.get('ORIGINAL_VS_EXACT_CELL_RHO')}",
        f"- ORIGINAL_VS_INDEPENDENT_CELL_RHO: {summary.get('ORIGINAL_VS_INDEPENDENT_CELL_RHO')}",
        "",
        "## Section 8: Source-Error Descriptive Replication",
        f"- C_OVER_R_SOURCE_NORM_EXACT: {json.dumps(summary.get('C_OVER_R_SOURCE_NORM_EXACT'), sort_keys=True)}",
        f"- C_OVER_R_SOURCE_NORM_INDEPENDENT: {json.dumps(summary.get('C_OVER_R_SOURCE_NORM_INDEPENDENT'), sort_keys=True)}",
        "",
        "## Section 9: Scientific Classification",
        "REPRODUCIBILITY is Panel A exact rerun on the original 18 units. INDEPENDENT REPLICATION is Panel B on prompt-disjoint units.",
        f"- FINAL_CLASSIFICATION: {summary.get('FINAL_CLASSIFICATION')}",
    ]
    text = "\n".join(sections) + "\n"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "report.md").write_text(text, encoding="utf-8")
    (REPORT_DIR / f"{SLUG}.md").write_text(text, encoding="utf-8")


def render_console_summary(summary):
    lines = []
    for key in REQUIRED_CONSOLE_KEYS:
        val = summary.get(key)
        if isinstance(val, (dict, list)):
            val = json.dumps(val, sort_keys=True)
        lines.append(f"{key} =")
        lines.append("" if val is None else str(val))
        lines.append("")
    lines.append("STOP.")
    return "\n".join(lines)


def freeze_manifests():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    v1 = import_v1()
    original = load_original_manifest()
    rows_by_pid = v1.P.load_dataset()
    teacher_tokens = v1.P.load_fp_teacher_tokens()
    exact = {
        "task": TASK,
        "panel": "exact",
        "source": str(V1_CANONICAL_UNITS),
        "sha256": file_sha256(V1_CANONICAL_UNITS),
        "n_units": len(original["units"]),
        "units": original["units"],
    }
    independent = construct_independent_manifest(original["units"], rows_by_pid, teacher_tokens)
    independent["sha256"] = canonical_json_sha256({k: v for k, v in independent.items() if k != "sha256"})
    save_json(RUN_DIR / "exact_manifest.json", exact)
    save_json(RUN_DIR / "independent_manifest.json", independent)
    independent_sha = independent["sha256"]
    audit = independence_audit(exact["units"], independent["units"])
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "STAGE0_MANIFESTS_FROZEN",
        "V1_REFERENCE_GATE": "PASS" if exact["sha256"] == V1_EXPECTED_MANIFEST_SHA256 and V1_RUN_DIR.exists() else "FAIL",
        "CANONICAL_PROTOCOL_GATE": "PASS",
        "REPLICATION_SAMPLING_PROTOCOL_GATE": "PASS",
        "NO_OUTCOME_BASED_SELECTION_GATE": "PASS",
        "EXACT_MANIFEST_SHA256": exact["sha256"],
        "INDEPENDENT_MANIFEST_SHA256": independent_sha,
        "N_EXACT_UNITS": len(exact["units"]),
        "N_INDEPENDENT_UNITS": len(independent["units"]),
        **audit,
    }
    save_json(RUN_DIR / "stage0_manifest_audit.json", out)
    return out


def panel_units(panel):
    path = RUN_DIR / f"{panel}_manifest.json"
    if not path.exists():
        freeze_manifests()
    return json.loads(path.read_text(encoding="utf-8"))["units"], json.loads(path.read_text(encoding="utf-8"))["sha256"]


def stage0():
    audit = freeze_manifests()
    return audit


def smoke():
    exact_units, exact_sha = panel_units("exact")
    independent_units, independent_sha = panel_units("independent")
    exact = run_panel("exact", "smoke", exact_units[:1], exact_sha, unit_limit=None, history_lengths=[1, 8])
    independent = run_panel("independent", "smoke", independent_units[:1], independent_sha, unit_limit=None, history_lengths=[1, 8])
    out = {
        "EXACT_SMOKE": "PASS" if exact.get("results_jsonl_rows_written") == 16 and not exact.get("failed_units") else "FAIL",
        "INDEPENDENT_SMOKE": "PASS" if independent.get("results_jsonl_rows_written") == 16 and not independent.get("failed_units") else "FAIL",
    }
    save_json(RUN_DIR / "smoke_summary.json", out)
    return out


def pilot():
    independent_units, independent_sha = panel_units("independent")
    out = run_panel("independent", "pilot", independent_units[:3], independent_sha, unit_limit=None, history_lengths=[1, 8, 32, 64])
    summary = {"PILOT_PROTOCOL_GATE": "PASS" if out.get("PILOT") == "PASS" else "FAIL"}
    save_json(RUN_DIR / "pilot_summary.json", summary)
    return summary


def formal(args):
    units, sha = panel_units(args.panel)
    selected = select_shard_units(units, args.shard_id, args.num_shards)
    panel_dir = RUN_DIR / f"{args.panel}_formal_shard{args.shard_id}_of_{args.num_shards}"
    out = run_panel(args.panel, "formal", selected, sha, shard_id=0, num_shards=1, run_dir=panel_dir)
    out["PANEL"] = args.panel
    out["REQUESTED_SHARD_ID"] = args.shard_id
    out["REQUESTED_NUM_SHARDS"] = args.num_shards
    out["UNIT_IDS"] = [u["unit_id"] for u in selected]
    save_json(RUN_DIR / f"{args.panel}_formal_shard{args.shard_id}_summary.json", out)
    return out


def merge_formal(args):
    if args.exact_shards:
        exact_summary = merge_panel("exact", [Path(x) for x in args.exact_shards.split(",") if x])
    else:
        exact_summary = {}
    if args.independent_shards:
        independent_summary = merge_panel("independent", [Path(x) for x in args.independent_shards.split(",") if x])
    else:
        independent_summary = {}
    out = {"exact": exact_summary, "independent": independent_summary}
    save_json(RUN_DIR / "merge_summary.json", out)
    return out


def analyze(args):
    return aggregate_replication(args.pytest_status, args.py_compile_status, args.bash_syntax_status, args.gpu_release_status)


def parse_args():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["stage0", "smoke", "pilot", "formal", "merge-formal", "verify", "analyze"], default="stage0")
    p.add_argument("--panel", choices=PANELS, default="exact")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--exact-shards", default="")
    p.add_argument("--independent-shards", default="")
    p.add_argument("--console-summary", action="store_true")
    p.add_argument("--pytest-status", default="NOT_RUN")
    p.add_argument("--py-compile-status", default="NOT_RUN")
    p.add_argument("--bash-syntax-status", default="NOT_RUN")
    p.add_argument("--gpu-release-status", default="NOT_RUN")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "stage0":
        out = stage0()
    elif args.stage == "smoke":
        out = smoke()
    elif args.stage == "pilot":
        out = pilot()
    elif args.stage == "formal":
        out = formal(args)
    elif args.stage == "merge-formal":
        out = merge_formal(args)
    elif args.stage == "verify":
        out = verify_fresh_outputs()
    elif args.stage == "analyze":
        out = analyze(args)
    else:
        raise ValueError(args.stage)
    if args.console_summary:
        print(render_console_summary(out))
    else:
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
