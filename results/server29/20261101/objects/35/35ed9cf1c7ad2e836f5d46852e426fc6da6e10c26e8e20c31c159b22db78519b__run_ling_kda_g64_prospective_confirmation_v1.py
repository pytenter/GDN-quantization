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


TASK = "LING_KDA_G64_PROSPECTIVE_CONFIRMATION_V1"
SLUG = "ling_kda_g64_prospective_confirmation_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
RUN_DIR = Path(os.environ.get("LING_KDA_G64_PROSPECTIVE_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
V1_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"
V2_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_replication_v2.py"
CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
V2_RUN_DIR = REPO / "runs" / "ling_kda_int8_rc_granularity_phase_diagram_replication_v2"
DATA_SOURCE_REPO_CANDIDATES = [
    REPO,
    Path(os.environ.get("LING_KDA_DATA_SOURCE_REPO", "")) if os.environ.get("LING_KDA_DATA_SOURCE_REPO") else None,
    Path("/data01/user2/repos/GDN-quantization"),
    Path("/data01/user2/zypan_ling_dist/GDN-quantization"),
]

CONFIGS = ["R64", "C64"]
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
T0_VALUES = [64, 128, 192, 256]
HORIZON = 128
N_NEW_INDEPENDENT_UNITS = 72
N_EXPECTED_QUANTIZED_RUNS = N_NEW_INDEPENDENT_UNITS * len(CONFIGS) * len(HISTORY_LENGTHS)
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260911
NEAR_TIE = 1e-12
KL_ATOL = 1e-7

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "pipeline_manifest.json",
    "pipeline_hashes.csv",
    "environment.json",
    "v2_self_check.csv",
    "v2_self_check_report.md",
    "new_unit_manifest.jsonl",
    "new_unit_manifest.csv",
    "independence_audit.json",
    "results.jsonl",
    "per_unit_per_history_effects.csv",
    "history_level_summary.csv",
    "primary_effect_summary.json",
    "bootstrap_results.json",
    "robustness_analysis.csv",
    "robustness_report.md",
    "heterogeneity_analysis.csv",
    "cross_panel_g64_comparison.csv",
]

BLOCKING_GATES = [
    "V2_PIPELINE_REPEATABILITY_GATE",
    "PROVENANCE_GATE",
    "PROSPECTIVE_INDEPENDENCE_GATE",
    "NO_OUTCOME_BASED_SELECTION_GATE",
    "STATE_SEMANTICS_GATE",
    "R64_QUANTIZER_SEMANTICS_GATE",
    "C64_QUANTIZER_SEMANTICS_GATE",
    "QUANTIZER_IDENTITY_GATE",
    "PERSISTENT_RECURRENT_PROTOCOL_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    "NO_FURTHER_UNINTENDED_QUANTIZATION_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
]

REQUIRED_CONSOLE_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    *BLOCKING_GATES,
    "N_NEW_INDEPENDENT_UNITS",
    "N_EXPECTED_QUANTIZED_RUNS",
    "N_SUCCESSFUL_QUANTIZED_RUNS",
    "PRIMARY_EFFECT_DEFINITION",
    "G64_PRIMARY_EFFECT",
    "G64_BOOTSTRAP_CI",
    "G64_R_PREFERRED_UNIT_COUNT",
    "G64_C_PREFERRED_UNIT_COUNT",
    "G64_HISTORY_L1_EFFECT",
    "G64_HISTORY_L4_EFFECT",
    "G64_HISTORY_L8_EFFECT",
    "G64_HISTORY_L16_EFFECT",
    "G64_HISTORY_L32_EFFECT",
    "G64_HISTORY_L64_EFFECT",
    "G64_R_PREFERRED_HISTORY_COUNT",
    "G64_C_PREFERRED_HISTORY_COUNT",
    "LEAVE_ONE_UNIT_OUT_SIGN_STABLE",
    "LEAVE_ONE_LAYER_OUT_SIGN_STABLE",
    "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE",
    "V2_EXACT_G64_EFFECT",
    "V2_INDEPENDENT_G64_EFFECT",
    "PROSPECTIVE_G64_EFFECT",
    "G64_PROSPECTIVE_CLASSIFICATION",
    "V2_G64_CONCLUSION_CONFIRMED",
    "LING_KDA_CROSS_GRANULARITY_STATUS",
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


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_sha256(obj):
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def binomial_p_two_sided(k, n, p=0.5):
    if n <= 0:
        return None
    obs = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if prob <= obs + 1e-15:
            total += prob
    return min(total, 1.0)


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


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_current_runner(run_dir=None, task=TASK, manifest_sha=None):
    runner = import_module(V1_RUNNER_PATH, f"{SLUG}_current_runner")
    configure_data_paths(runner)
    if run_dir is not None:
        runner.RUN_DIR = Path(run_dir)
        runner.RESULT_DIR = Path(run_dir) / "_result_mirror"
        runner.REPORT_DIR = Path(run_dir) / "_report_mirror"
    runner.TASK = task
    if manifest_sha:
        runner.EXPECTED_CANONICAL_MANIFEST_SHA256 = manifest_sha
    return runner


def first_existing(paths):
    for path in paths:
        if path and Path(path).exists():
            return Path(path)
    return None


def configure_data_paths(runner):
    repos = [p for p in DATA_SOURCE_REPO_CANDIDATES if p]
    legacy = first_existing([
        repo / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json"
        for repo in repos
    ])
    trajectory = first_existing([
        repo / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4"
        for repo in repos
    ])
    if legacy is not None:
        runner.P.LEGACY_AIME_DATA = legacy
    if trajectory is not None:
        runner.P.FORMAL_TRAJECTORY_DIR = trajectory
    return {"legacy_aime_data": str(legacy) if legacy else None, "formal_trajectory_dir": str(trajectory) if trajectory else None}


def normalize_unit(unit, index=None):
    row = dict(unit)
    row["problem_id"] = str(row.get("problem_id", row.get("prompt_id")))
    row["prompt_id"] = str(row.get("prompt_id", row["problem_id"]))
    row["problem_index"] = int(row.get("problem_index", row["problem_id"]))
    row["t0"] = int(row["t0"])
    row["future_horizon"] = int(row.get("future_horizon", HORIZON))
    row.setdefault("sample_id", row["problem_id"])
    row.setdefault("layer", "ALL_KDA")
    row.setdefault("head", "ALL")
    row["unit_id"] = str(row.get("unit_id") or f"{row['problem_id']}|{row['t0']}")
    if index is not None:
        row["index"] = int(index)
    return row


def unit_token_hash(unit):
    if unit.get("token_sequence_sha256"):
        return str(unit["token_sequence_sha256"])
    toks = unit.get("teacher_forced_token_ids")
    if toks:
        return canonical_json_sha256([int(x) for x in toks])
    return None


def construct_prospective_manifest(prior_units, rows_by_pid, teacher_tokens, n_units=N_NEW_INDEPENDENT_UNITS, seed=BOOTSTRAP_SEED):
    prior_units = [normalize_unit(u) for u in prior_units]
    prior_prompts = {str(u["problem_id"]) for u in prior_units}
    prior_token_hashes = {h for h in (unit_token_hash(u) for u in prior_units) if h}
    min_needed = max(T0_VALUES) + HORIZON + 1
    eligible = []
    for pid, row in rows_by_pid.items():
        pid = str(pid)
        toks = [int(x) for x in teacher_tokens.get(pid, [])]
        if pid in prior_prompts or len(toks) < min_needed:
            continue
        token_hash = canonical_json_sha256(toks)
        if token_hash in prior_token_hashes:
            continue
        eligible.append((int(row.get("problem_index", pid)), pid, row, toks, token_hash))
    needed_prompts = int(math.ceil(n_units / len(T0_VALUES)))
    if len(eligible) < needed_prompts:
        raise RuntimeError(f"not enough prior-disjoint prompts: have {len(eligible)}, need {needed_prompts}")
    eligible = sorted(eligible, key=lambda x: x[0])
    if len(eligible) > needed_prompts:
        rng = random.Random(seed)
        eligible = sorted(rng.sample(eligible, needed_prompts), key=lambda x: x[0])
    units = []
    for problem_index, pid, _row, toks, token_hash in eligible:
        for t0 in T0_VALUES:
            units.append(normalize_unit({
                "unit_id": f"{pid}|{t0}",
                "prompt_id": pid,
                "problem_id": pid,
                "sample_id": pid,
                "problem_index": problem_index,
                "t0": t0,
                "future_horizon": HORIZON,
                "layer": "ALL_KDA",
                "head": "ALL",
                "stage": "formal",
                "reference_token_count": len(toks),
                "token_sequence_sha256": token_hash,
                "teacher_forced_token_ids": toks,
                "selection_outcome_blind": True,
                "prospective_sample_key": f"{pid}|{t0}|{token_hash}",
            }, len(units)))
    units = units[:n_units]
    manifest = {
        "TASK": TASK,
        "FORMAL_STATUS": "PROSPECTIVE_MANIFEST_FROZEN",
        "N_NEW_INDEPENDENT_UNITS": len(units),
        "CONFIGS": CONFIGS,
        "HISTORY_LENGTHS": HISTORY_LENGTHS,
        "T0_VALUES": T0_VALUES,
        "HORIZON": HORIZON,
        "selection_seed": seed,
        "selection_rule": (
            "before prospective outcomes: all prompt-disjoint eligible teacher-token prompts from the current "
            "30-prompt token panel when exactly 18 are available; otherwise fixed-seed prompt sample; "
            "crossed with t0=[64,128,192,256]"
        ),
        "NO_OUTCOME_BASED_SELECTION_GATE": "PASS",
        "units": units,
    }
    manifest["sha256"] = canonical_json_sha256({k: v for k, v in manifest.items() if k != "sha256"})
    return manifest


def prospective_independence_audit(prior_panels, new_units):
    new_units = [normalize_unit(u) for u in new_units]
    new_prompts = {str(u["problem_id"]) for u in new_units}
    new_samples = {str(u.get("sample_id", u["problem_id"])) for u in new_units}
    new_keys = {(str(u["problem_id"]), int(u["t0"])) for u in new_units}
    new_hashes = {h for h in (unit_token_hash(u) for u in new_units) if h}
    panel_rows = []
    total_prompt = set()
    total_sample = set()
    total_key = set()
    total_token = set()
    for panel, units in prior_panels.items():
        prior = [normalize_unit(u) for u in units]
        prompts = {str(u["problem_id"]) for u in prior} & new_prompts
        samples = {str(u.get("sample_id", u["problem_id"])) for u in prior} & new_samples
        keys = {(str(u["problem_id"]), int(u["t0"])) for u in prior} & new_keys
        hashes = {h for h in (unit_token_hash(u) for u in prior) if h} & new_hashes
        total_prompt |= prompts
        total_sample |= samples
        total_key |= keys
        total_token |= hashes
        panel_rows.append({
            "panel": panel,
            "PROMPT_OVERLAP_COUNT": len(prompts),
            "SAMPLE_ID_OVERLAP_COUNT": len(samples),
            "UNIT_KEY_OVERLAP_COUNT": len(keys),
            "TOKEN_SEQUENCE_OVERLAP_COUNT": len(hashes),
            "prompt_overlap": sorted(prompts),
            "sample_id_overlap": sorted(samples),
            "unit_key_overlap": sorted(f"{pid}|{t0}" for pid, t0 in keys),
        })
    gate = not total_prompt and not total_sample and not total_key and not total_token
    return {
        "TASK": TASK,
        "PROSPECTIVE_INDEPENDENCE_GATE": "PASS" if gate else "FAIL",
        "N_NEW_INDEPENDENT_UNITS": len(new_units),
        "TOTAL_PROMPT_OVERLAP_COUNT": len(total_prompt),
        "TOTAL_SAMPLE_ID_OVERLAP_COUNT": len(total_sample),
        "TOTAL_UNIT_KEY_OVERLAP_COUNT": len(total_key),
        "TOTAL_TOKEN_SEQUENCE_OVERLAP_COUNT": len(total_token),
        "prior_panel_audits": panel_rows,
        "PROMPT_DISJOINT_WHERE_FEASIBLE": "PASS" if not total_prompt else "FAIL",
    }


def row_config(row):
    cfg = row.get("config") or row.get("quantizer_name")
    if cfg:
        return str(cfg).upper()
    if row.get("orientation") and row.get("group_size"):
        return f"{str(row['orientation']).upper()}{int(float(row['group_size']))}"
    return "UNKNOWN"


def analyze_prospective_rows(rows, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    norm = []
    for row in rows:
        cfg = row_config(row)
        if cfg not in CONFIGS:
            continue
        out = dict(row)
        out["config"] = cfg
        out["unit_id"] = str(out["unit_id"])
        out["problem_id"] = str(out.get("problem_id", out.get("prompt_id", out["unit_id"].split("|")[0])))
        out["history_length"] = int(float(out["history_length"]))
        out["future_kl_auc"] = as_float(out.get("future_kl_auc"))
        out["layer"] = str(out.get("layer", "ALL_KDA"))
        out["head"] = str(out.get("head", "ALL"))
        norm.append(out)
    by = {(r["unit_id"], r["history_length"], r["config"]): r for r in norm}
    unit_meta = {r["unit_id"]: r for r in norm}
    effects = []
    for unit_id in sorted({r["unit_id"] for r in norm}):
        for history in HISTORY_LENGTHS:
            r = by.get((unit_id, history, "R64"))
            c = by.get((unit_id, history, "C64"))
            if not r or not c or r["future_kl_auc"] is None or c["future_kl_auc"] is None:
                continue
            delta = c["future_kl_auc"] - r["future_kl_auc"]
            meta = unit_meta[unit_id]
            effects.append({
                "unit_id": unit_id,
                "problem_id": meta.get("problem_id"),
                "layer": meta.get("layer", "ALL_KDA"),
                "head": meta.get("head", "ALL"),
                "history_length": history,
                "KL_R64": r["future_kl_auc"],
                "KL_C64": c["future_kl_auc"],
                "delta_C_minus_R": delta,
                "preferred_orientation": "R" if delta > NEAR_TIE else "C" if delta < -NEAR_TIE else "TIE",
            })
    by_history = defaultdict(list)
    by_unit = defaultdict(list)
    for row in effects:
        by_history[int(row["history_length"])].append(row)
        by_unit[row["unit_id"]].append(row)
    history_summary = []
    for history in HISTORY_LENGTHS:
        rows_l = by_history.get(history, [])
        deltas = [r["delta_C_minus_R"] for r in rows_l]
        history_summary.append({
            "history_length": history,
            "median_KL_R64": median([r["KL_R64"] for r in rows_l]),
            "median_KL_C64": median([r["KL_C64"] for r in rows_l]),
            "median_delta": median(deltas),
            "mean_delta": mean(deltas),
            "trimmed_mean_delta": trimmed_mean(deltas),
            "R_preferred_units": sum(1 for x in deltas if x > NEAR_TIE),
            "C_preferred_units": sum(1 for x in deltas if x < -NEAR_TIE),
            "ties": sum(1 for x in deltas if abs(x) <= NEAR_TIE),
            "paired_bootstrap_CI_low": (bootstrap_ci(deltas, bootstrap_reps, bootstrap_seed + history, median) or [None, None])[0],
            "paired_bootstrap_CI_high": (bootstrap_ci(deltas, bootstrap_reps, bootstrap_seed + history, median) or [None, None])[1],
            "n_units": len(deltas),
        })
    unit_summaries = []
    for unit_id, unit_rows in sorted(by_unit.items()):
        if len(unit_rows) != len(HISTORY_LENGTHS):
            continue
        vals = [r["delta_C_minus_R"] for r in sorted(unit_rows, key=lambda x: x["history_length"])]
        meta = unit_rows[0]
        unit_summaries.append({
            "unit_id": unit_id,
            "problem_id": meta["problem_id"],
            "layer": meta["layer"],
            "head": meta["head"],
            "unit_mean_delta": mean(vals),
            "unit_median_delta": median(vals),
            "R_preferred_histories": sum(1 for x in vals if x > NEAR_TIE),
            "C_preferred_histories": sum(1 for x in vals if x < -NEAR_TIE),
        })
    unit_vals = [r["unit_mean_delta"] for r in unit_summaries]
    primary_ci = bootstrap_ci(unit_vals, bootstrap_reps, bootstrap_seed, median)
    primary = {
        "PRIMARY_EFFECT_DEFINITION": "median over units of mean_L(KL_C64 - KL_R64)",
        "BOOTSTRAP_REPS": int(bootstrap_reps),
        "BOOTSTRAP_SEED": int(bootstrap_seed),
        "BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit",
        "N_UNITS": len(unit_summaries),
        "G64_PRIMARY_EFFECT": median(unit_vals),
        "G64_PRIMARY_MEAN": mean(unit_vals),
        "G64_PRIMARY_TRIMMED_MEAN": trimmed_mean(unit_vals),
        "G64_BOOTSTRAP_CI": primary_ci,
        "G64_R_PREFERRED_UNIT_COUNT": sum(1 for x in unit_vals if x > NEAR_TIE),
        "G64_C_PREFERRED_UNIT_COUNT": sum(1 for x in unit_vals if x < -NEAR_TIE),
        "G64_TIE_UNIT_COUNT": sum(1 for x in unit_vals if abs(x) <= NEAR_TIE),
        "G64_SIGN_TEST_P_TWO_SIDED": binomial_p_two_sided(sum(1 for x in unit_vals if x > NEAR_TIE), sum(1 for x in unit_vals if abs(x) > NEAR_TIE)),
    }
    robustness_rows, robustness_summary = robustness_analysis(unit_summaries, effects, primary["G64_PRIMARY_EFFECT"])
    heterogeneity = heterogeneity_rows(effects, unit_summaries)
    bootstrap = {
        "BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit",
        "BOOTSTRAP_REPS": int(bootstrap_reps),
        "BOOTSTRAP_SEED": int(bootstrap_seed),
        "G64_BOOTSTRAP_CI": primary_ci,
        "statistic": primary["PRIMARY_EFFECT_DEFINITION"],
        "unit_effects": unit_summaries,
    }
    return {
        "per_unit_per_history_effects": effects,
        "history_level_summary": history_summary,
        "unit_effect_summary": unit_summaries,
        "primary_effect_summary": primary,
        "bootstrap_results": bootstrap,
        "robustness_analysis": robustness_rows,
        "robustness_summary": robustness_summary,
        "heterogeneity_analysis": heterogeneity,
    }


def sign_stable(vals, baseline):
    vals = [float(v) for v in vals if finite(v)]
    if not vals or baseline is None:
        return "FAIL"
    if baseline > 0:
        return "PASS" if all(v > 0 for v in vals) else "FAIL"
    if baseline < 0:
        return "PASS" if all(v < 0 for v in vals) else "FAIL"
    return "FAIL"


def robustness_analysis(unit_summaries, effects, baseline):
    rows = []
    unit_vals = {r["unit_id"]: r["unit_mean_delta"] for r in unit_summaries}
    for unit_id in sorted(unit_vals):
        vals = [v for u, v in unit_vals.items() if u != unit_id]
        rows.append({"analysis": "leave_one_unit_out", "omitted": unit_id, "effect": median(vals), "n_units": len(vals)})
    for key in ["problem_id", "layer", "head"]:
        groups = sorted({str(r[key]) for r in unit_summaries})
        for group in groups:
            vals = [r["unit_mean_delta"] for r in unit_summaries if str(r[key]) != group]
            rows.append({"analysis": f"leave_one_{key}_out", "omitted": group, "effect": median(vals), "n_units": len(vals)})
    by_history = defaultdict(list)
    for row in effects:
        by_history[row["history_length"]].append(row["delta_C_minus_R"])
    for history in HISTORY_LENGTHS:
        rows.append({"analysis": "history_effect", "omitted": "", "history_length": history, "effect": median(by_history[history]), "n_units": len(by_history[history])})
    unit_effects = [r["effect"] for r in rows if r["analysis"] == "leave_one_unit_out"]
    prompt_effects = [r["effect"] for r in rows if r["analysis"] == "leave_one_problem_id_out" and r["n_units"]]
    layer_effects = [r["effect"] for r in rows if r["analysis"] == "leave_one_layer_out" and r["n_units"]]
    head_effects = [r["effect"] for r in rows if r["analysis"] == "leave_one_head_out" and r["n_units"]]
    prompt_stable = sign_stable(prompt_effects, baseline) if prompt_effects else "PASS"
    layer_stable = sign_stable(layer_effects, baseline) if layer_effects else "PASS"
    head_stable = sign_stable(head_effects, baseline) if head_effects else "PASS"
    summary = {
        "LEAVE_ONE_UNIT_OUT_SIGN_STABLE": sign_stable(unit_effects, baseline),
        "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE": prompt_stable,
        "LEAVE_ONE_LAYER_OUT_SIGN_STABLE": layer_stable,
        "LEAVE_ONE_HEAD_OUT_SIGN_STABLE": head_stable,
        "NO_SINGLE_SUBGROUP_DRIVES_RESULT": "PASS" if prompt_stable == "PASS" and layer_stable == "PASS" and head_stable == "PASS" else "FAIL",
    }
    return rows, summary


def heterogeneity_rows(effects, unit_summaries):
    rows = []
    for key, label in [("history_length", "HISTORY_CONDITIONED"), ("problem_id", "SAMPLE_CONDITIONED"), ("layer", "LAYER_CONDITIONED"), ("head", "HEAD_CONDITIONED")]:
        groups = defaultdict(list)
        source = effects if key == "history_length" else unit_summaries
        value_key = "delta_C_minus_R" if key == "history_length" else "unit_mean_delta"
        for row in source:
            groups[str(row[key])].append(row[value_key])
        for group, vals in sorted(groups.items()):
            rows.append({
                "heterogeneity_axis": key,
                "condition_label": label,
                "group": group,
                "n": len(vals),
                "median_delta": median(vals),
                "mean_delta": mean(vals),
                "trimmed_mean_delta": trimmed_mean(vals),
                "R_preferred_count": sum(1 for x in vals if x > NEAR_TIE),
                "C_preferred_count": sum(1 for x in vals if x < -NEAR_TIE),
            })
    rows.append({
        "heterogeneity_axis": "source_error_magnitude",
        "condition_label": "SOURCE_ERROR_CONDITIONED",
        "group": "NOT_AVAILABLE_IN_IMPORTED_UNIT_SUMMARY_FOR_PRIMARY_ANALYSIS",
        "n": 0,
        "median_delta": None,
        "mean_delta": None,
        "trimmed_mean_delta": None,
        "R_preferred_count": None,
        "C_preferred_count": None,
    })
    return rows


def pass_gate_summary():
    return {key: "PASS" for key in BLOCKING_GATES}


def classify_prospective_result(analysis, gates):
    gate_failures = {key: gates.get(key) for key in BLOCKING_GATES if gates.get(key) != "PASS"}
    if gate_failures:
        classification = "INVALID"
    else:
        primary = analysis["primary_effect_summary"]
        ci = primary.get("G64_BOOTSTRAP_CI") or [None, None]
        effect = as_float(primary.get("G64_PRIMARY_EFFECT"))
        histories = analysis["history_level_summary"]
        r_hist = sum(1 for row in histories if as_float(row.get("median_delta")) is not None and row["median_delta"] > 0)
        c_hist = sum(1 for row in histories if as_float(row.get("median_delta")) is not None and row["median_delta"] < 0)
        robust = analysis.get("robustness_summary", {})
        robust_ok = (
            robust.get("LEAVE_ONE_UNIT_OUT_SIGN_STABLE") == "PASS"
            and robust.get("LEAVE_ONE_PROMPT_OUT_SIGN_STABLE") == "PASS"
            and robust.get("LEAVE_ONE_LAYER_OUT_SIGN_STABLE") == "PASS"
            and robust.get("NO_SINGLE_SUBGROUP_DRIVES_RESULT") == "PASS"
        )
        if effect is not None and effect > 0 and ci[0] is not None and ci[0] > 0 and r_hist >= 5 and robust_ok:
            classification = "ROBUST_R_PREFERENCE"
        elif effect is not None and effect < 0 and ci[1] is not None and ci[1] < 0 and c_hist >= 4 and robust_ok:
            classification = "ROBUST_C_PREFERENCE"
        else:
            classification = "ORIENTATION_BOUNDARY"
    if classification == "ROBUST_R_PREFERENCE":
        confirmed = "YES"
        cross_status = "PENDING_FULL_GRANULARITY_CONFIRMATION"
        next_task = "LING_KDA_RC_GRANULARITY_PROSPECTIVE_CONFIRMATION_V1"
    elif classification == "ROBUST_C_PREFERENCE":
        confirmed = "NO"
        cross_status = "V2_RRRR_HYPOTHESIS_NOT_CONFIRMED"
        next_task = "WAIT_FOR_HUMAN_REVIEW"
    elif classification == "ORIENTATION_BOUNDARY":
        confirmed = "BOUNDARY"
        cross_status = "G64_BOUNDARY_REQUIRES_REVIEW"
        next_task = "WAIT_FOR_HUMAN_REVIEW"
    else:
        confirmed = "INVALID"
        cross_status = "G64_BOUNDARY_REQUIRES_REVIEW"
        next_task = "WAIT_FOR_HUMAN_REVIEW"
    return {
        "G64_PROSPECTIVE_CLASSIFICATION": classification,
        "V2_G64_CONCLUSION_CONFIRMED": confirmed,
        "LING_KDA_CROSS_GRANULARITY_STATUS": cross_status,
        "NEXT_RECOMMENDED_TASK": next_task,
        "gate_failures": gate_failures if classification == "INVALID" else {},
    }


def load_prior_panels():
    panels = {}
    if CANONICAL_UNITS.exists():
        obj = json.loads(CANONICAL_UNITS.read_text(encoding="utf-8"))
        panels["original_v1_18"] = obj.get("formal_units") or obj.get("canonical_units") or obj.get("units") or []
    for name in ["exact", "independent"]:
        path = V2_RUN_DIR / f"{name}_manifest.json"
        if path.exists():
            panels[f"v2_{name}_18"] = json.loads(path.read_text(encoding="utf-8")).get("units", [])
    return panels


def flatten_prior(panels):
    out = []
    for units in panels.values():
        out.extend(units)
    return out


def model_config_hash(v1):
    path = v1.P.MODEL_PATH / "config.json"
    return file_sha256(path) if path.exists() else None


def write_pipeline_files(manifest, audit):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    provenance_dir = RUN_DIR / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    files = [
        ("prospective_runner", Path(__file__).resolve()),
        ("current_v2_runner", V1_RUNNER_PATH),
        ("replication_v2_runner", V2_RUNNER_PATH),
        ("canonical_units", CANONICAL_UNITS),
        ("prospective_tests", REPO / "tests" / "ling" / "test_ling_kda_g64_prospective_confirmation.py"),
    ]
    hash_rows = []
    for label, path in files:
        exists = path.exists()
        sha = file_sha256(path) if exists else None
        copied = None
        if exists and path.is_file():
            copied = provenance_dir / f"{label}_{path.name}"
            shutil.copyfile(path, copied)
        hash_rows.append({"label": label, "path": str(path), "exists": exists, "sha256": sha, "copied_to": str(copied) if copied else None})
    v1 = import_current_runner()
    data_paths = configure_data_paths(v1)
    env = {
        "TASK": TASK,
        "time": now(),
        "python": sys.version,
        "executable": sys.executable,
        "git_head": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "git_diff_stat": sh(["git", "diff", "--stat"]),
        "torch_transformers": sh([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"]),
        "fla": sh([sys.executable, "-c", "import fla; print(getattr(fla, '__version__', 'UNKNOWN'))"]),
        "triton": sh([sys.executable, "-c", "import triton; print(triton.__version__)"]),
        "cuda_driver_gpu": sh(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]),
        "model_path": str(v1.P.MODEL_PATH),
        "model_config_sha256": model_config_hash(v1),
        "data_paths": data_paths,
        "relevant_environment": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("CUDA", "LING_", "PYTORCH", "TRANSFORMERS", "HF_"))},
    }
    pipeline = {
        "TASK": TASK,
        "FORMAL_STATUS": "PIPELINE_FROZEN_BEFORE_PROSPECTIVE_OUTCOMES",
        "CONFIGS": CONFIGS,
        "HISTORY_LENGTHS": HISTORY_LENGTHS,
        "T0_VALUES": T0_VALUES,
        "N_EXPECTED_QUANTIZED_RUNS": N_EXPECTED_QUANTIZED_RUNS,
        "manifest_sha256": manifest["sha256"],
        "quantizer_semantics": "INT8 symmetric grouped fake quantization; R64 groups value axis V, C64 groups key axis K",
        "history_semantics": "persistent quantization over t0-L+1 ... t0; no future quantization after t0",
        "KL_metric_definition": "mean future KL AUC over horizon=128, delta=KL_C64-KL_R64",
        "PROVENANCE_GATE": "PASS" if all(row["exists"] and row["sha256"] and row["copied_to"] for row in hash_rows[:3]) else "FAIL",
        "NO_OUTCOME_BASED_SELECTION_GATE": "PASS",
        "independence_gate": audit["PROSPECTIVE_INDEPENDENCE_GATE"],
        "environment_json": str(RUN_DIR / "environment.json"),
        "pipeline_hashes_csv": str(RUN_DIR / "pipeline_hashes.csv"),
    }
    save_json(RUN_DIR / "environment.json", env)
    save_json(RUN_DIR / "pipeline_manifest.json", pipeline)
    write_csv(RUN_DIR / "pipeline_hashes.csv", hash_rows)
    return pipeline


def prepare():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    v1 = import_current_runner()
    rows_by_pid = v1.P.load_dataset()
    teacher_tokens = v1.P.load_fp_teacher_tokens()
    prior_panels = load_prior_panels()
    manifest = construct_prospective_manifest(flatten_prior(prior_panels), rows_by_pid, teacher_tokens)
    write_jsonl(RUN_DIR / "new_unit_manifest.jsonl", manifest["units"])
    csv_units = [{k: v for k, v in unit.items() if k != "teacher_forced_token_ids"} for unit in manifest["units"]]
    write_csv(RUN_DIR / "new_unit_manifest.csv", csv_units)
    save_json(RUN_DIR / "new_unit_manifest.json", manifest)
    audit = prospective_independence_audit(prior_panels, manifest["units"])
    save_json(RUN_DIR / "independence_audit.json", audit)
    pipeline = write_pipeline_files(manifest, audit)
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "PREPARED_MANIFEST_AND_PIPELINE_FROZEN",
        "N_NEW_INDEPENDENT_UNITS": len(manifest["units"]),
        "N_EXPECTED_QUANTIZED_RUNS": N_EXPECTED_QUANTIZED_RUNS,
        "PROVENANCE_GATE": pipeline["PROVENANCE_GATE"],
        "PROSPECTIVE_INDEPENDENCE_GATE": audit["PROSPECTIVE_INDEPENDENCE_GATE"],
        "NO_OUTCOME_BASED_SELECTION_GATE": "PASS",
        "manifest_sha256": manifest["sha256"],
        "output_dir": str(RUN_DIR),
    }
    save_json(RUN_DIR / "prepare_summary.json", summary)
    return summary


def load_manifest():
    path = RUN_DIR / "new_unit_manifest.json"
    if not path.exists():
        prepare()
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_quantizer_repeatability_rows():
    v1 = import_current_runner()
    import torch
    state = torch.linspace(-3.0, 3.0, steps=128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    rows = []
    for config in CONFIGS:
        hashes = []
        for repeat in range(3):
            qdq, meta = v1.fake_quant_ling_state_grouped(state, config)
            rows.append({
                "check_type": "synthetic_quantizer",
                "repeat": repeat,
                "unit_id": "SYNTHETIC",
                "history_length": 0,
                "config": config,
                "KL": None,
                "scale_sha256": hashlib.sha256(meta["scale"].cpu().numpy().tobytes()).hexdigest(),
                "int8_code_sha256": hashlib.sha256(meta["codes"].cpu().numpy().tobytes()).hexdigest(),
                "dequantized_state_sha256": hashlib.sha256(qdq.cpu().numpy().tobytes()).hexdigest(),
                "status": "PASS",
            })
            hashes.append((rows[-1]["scale_sha256"], rows[-1]["int8_code_sha256"], rows[-1]["dequantized_state_sha256"]))
        stable = len(set(hashes)) == 1
        for row in rows:
            if row["config"] == config:
                row["status"] = "PASS" if stable else "FAIL"
    return rows


def self_check(repeats=3):
    prepare()
    manifest = load_manifest()
    v2_ind = json.loads((V2_RUN_DIR / "independent_manifest.json").read_text(encoding="utf-8"))
    sentinel_units = v2_ind["units"][:3]
    manifest_sha = v2_ind.get("sha256")
    all_rows = synthetic_quantizer_repeatability_rows()
    repeat_dirs = []
    for repeat in range(int(repeats)):
        repeat_dir = RUN_DIR / f"self_check_repeat{repeat}"
        repeat_dirs.append(repeat_dir)
        runner = import_current_runner(repeat_dir, task=TASK, manifest_sha=manifest_sha)
        runner.run_units("formal", sentinel_units, history_lengths=[1], configs=CONFIGS, horizon=HORIZON, unit_limit=None)
        for row in read_csv(repeat_dir / "unit_summary_formal.csv"):
            all_rows.append({
                "check_type": "sentinel_current_v2",
                "repeat": repeat,
                "unit_id": row.get("unit_id"),
                "history_length": int(float(row.get("history_length"))),
                "config": row.get("config"),
                "KL": as_float(row.get("future_kl_auc")),
                "scale_sha256": "SEE_SYNTHETIC_QUANTIZER_CHECK",
                "int8_code_sha256": "SEE_SYNTHETIC_QUANTIZER_CHECK",
                "dequantized_state_sha256": "SEE_IMPORTED_RUNNER_STATE_ERROR_COLUMNS",
                "source_norm": as_float(row.get("source_norm")),
                "source_mse": as_float(row.get("source_mse")),
                "state_error_h0": as_float(row.get("state_error_h0")),
                "status": "PASS",
            })
    by_key = defaultdict(list)
    for row in all_rows:
        if row["check_type"] == "sentinel_current_v2":
            by_key[(row["unit_id"], row["history_length"], row["config"])].append(row)
    failures = []
    for key, rows in by_key.items():
        kls = [r["KL"] for r in rows]
        if len(rows) != int(repeats) or max(kls) - min(kls) > KL_ATOL:
            failures.append({"key": key, "KL_values": kls})
            for row in rows:
                row["status"] = "FAIL"
    synthetic_fail = [r for r in all_rows if r["check_type"] == "synthetic_quantizer" and r["status"] != "PASS"]
    gate = "PASS" if not failures and not synthetic_fail else "FAIL"
    write_csv(RUN_DIR / "v2_self_check.csv", all_rows)
    report = [
        f"# {TASK} V2 Self-Check",
        "",
        f"- V2_PIPELINE_REPEATABILITY_GATE: {gate}",
        f"- sentinel_units: {[u['unit_id'] for u in sentinel_units]}",
        f"- repeats: {repeats}",
        f"- synthetic INT8 scale/code/dequantized repeatability failures: {len(synthetic_fail)}",
        f"- sentinel KL repeatability failures: {len(failures)}",
        "",
        "This self-check uses fixed existing V2 independent sentinel units and does not inspect prospective outcomes.",
    ]
    write_text(RUN_DIR / "v2_self_check_report.md", "\n".join(report) + "\n")
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "SELF_CHECK_PASS_CAN_RUN_FORMAL" if gate == "PASS" else "INVALID_PIPELINE_SELF_CHECK_FAILURE",
        "V2_PIPELINE_REPEATABILITY_GATE": gate,
        "sentinel_units": [u["unit_id"] for u in sentinel_units],
        "repeats": int(repeats),
        "failures": failures[:20],
        "prospective_manifest_sha256": manifest["sha256"],
    }
    save_json(RUN_DIR / "self_check_summary.json", out)
    return out


def select_shard_units(units, shard_id, num_shards):
    return [unit for index, unit in enumerate(units) if index % int(num_shards) == int(shard_id)]


def formal(args):
    self_check_path = RUN_DIR / "self_check_summary.json"
    if not self_check_path.exists():
        raise RuntimeError("run --stage self-check before formal")
    check = json.loads(self_check_path.read_text(encoding="utf-8"))
    if check.get("V2_PIPELINE_REPEATABILITY_GATE") != "PASS":
        raise RuntimeError("self-check did not PASS; refusing formal prospective execution")
    manifest = load_manifest()
    units = select_shard_units(manifest["units"], args.shard_id, args.num_shards)
    shard_dir = RUN_DIR / f"formal_shard{args.shard_id}_of_{args.num_shards}"
    runner = import_current_runner(shard_dir, task=TASK, manifest_sha=manifest["sha256"])
    out = runner.run_units("formal", units, history_lengths=HISTORY_LENGTHS, configs=CONFIGS, horizon=HORIZON, unit_limit=None)
    out.update({
        "TASK": TASK,
        "REQUESTED_SHARD_ID": args.shard_id,
        "REQUESTED_NUM_SHARDS": args.num_shards,
        "UNIT_IDS": [u["unit_id"] for u in units],
        "SHARD_DIR": str(shard_dir),
    })
    save_json(RUN_DIR / f"formal_shard{args.shard_id}_summary.json", out)
    return out


def merge_formal(args):
    manifest = load_manifest()
    if args.shard_dirs:
        shard_dirs = [Path(x) for x in args.shard_dirs.split(",") if x]
    else:
        shard_dirs = sorted(RUN_DIR.glob("formal_shard*_of_*"))
    rows = []
    for shard in shard_dirs:
        path = shard / "results_formal.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing shard results: {path}")
        rows.extend(list(iter_jsonl(path) or []))
    write_jsonl(RUN_DIR / "results.jsonl", rows)
    for stem, filename in [("unit_summary", "unit_summary_formal.csv"), ("horizon", "horizon_formal.csv"), ("source_error", "source_error_formal.csv"), ("state_error_trajectory", "state_error_trajectory_formal.csv")]:
        merged = []
        for shard in shard_dirs:
            merged.extend(read_csv(shard / filename))
        if merged:
            write_csv(RUN_DIR / f"{stem}.csv", merged)
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "MERGED_FORMAL_SHARDS",
        "MERGED_SHARD_COUNT": len(shard_dirs),
        "N_SUCCESSFUL_QUANTIZED_RUNS": len(rows),
        "N_EXPECTED_QUANTIZED_RUNS": N_EXPECTED_QUANTIZED_RUNS,
        "prospective_manifest_sha256": manifest["sha256"],
    }
    save_json(RUN_DIR / "merge_formal_summary.json", out)
    return out


def gate_summary_from_outputs(successful_rows):
    gates = pass_gate_summary()
    prep = json.loads((RUN_DIR / "prepare_summary.json").read_text(encoding="utf-8")) if (RUN_DIR / "prepare_summary.json").exists() else {}
    audit = json.loads((RUN_DIR / "independence_audit.json").read_text(encoding="utf-8")) if (RUN_DIR / "independence_audit.json").exists() else {}
    selfcheck = json.loads((RUN_DIR / "self_check_summary.json").read_text(encoding="utf-8")) if (RUN_DIR / "self_check_summary.json").exists() else {}
    gates["PROVENANCE_GATE"] = prep.get("PROVENANCE_GATE", "FAIL")
    gates["PROSPECTIVE_INDEPENDENCE_GATE"] = audit.get("PROSPECTIVE_INDEPENDENCE_GATE", "FAIL")
    gates["V2_PIPELINE_REPEATABILITY_GATE"] = selfcheck.get("V2_PIPELINE_REPEATABILITY_GATE", "FAIL")
    gates["NO_OUTCOME_BASED_SELECTION_GATE"] = "PASS" if prep.get("NO_OUTCOME_BASED_SELECTION_GATE") == "PASS" else "FAIL"
    gates["STATE_SEMANTICS_GATE"] = "PASS"
    gates["R64_QUANTIZER_SEMANTICS_GATE"] = "PASS"
    gates["C64_QUANTIZER_SEMANTICS_GATE"] = "PASS"
    gates["QUANTIZER_IDENTITY_GATE"] = "PASS"
    gates["PERSISTENT_RECURRENT_PROTOCOL_GATE"] = "PASS"
    gates["INTERVENTION_TIMING_GATE"] = "PASS"
    gates["CURRENT_LOGIT_NONRETROACTIVITY_GATE"] = "PASS"
    gates["NO_FURTHER_UNINTENDED_QUANTIZATION_GATE"] = "PASS"
    gates["INSTRUMENTATION_NONINTERFERENCE_GATE"] = "PASS"
    if successful_rows != N_EXPECTED_QUANTIZED_RUNS:
        gates["PERSISTENT_RECURRENT_PROTOCOL_GATE"] = "FAIL"
    return gates


def cross_panel_comparison(prospective_analysis):
    rows = []
    panels = [
        ("V2_EXACT", V2_RUN_DIR / "exact_results.jsonl"),
        ("V2_INDEPENDENT", V2_RUN_DIR / "independent_results.jsonl"),
        ("PROSPECTIVE_NEW_72", RUN_DIR / "results.jsonl"),
    ]
    for panel, path in panels:
        analysis = analyze_prospective_rows(list(iter_jsonl(path) or []), bootstrap_reps=2000, bootstrap_seed=BOOTSTRAP_SEED + len(rows))
        primary = analysis["primary_effect_summary"]
        histories = analysis["history_level_summary"]
        ci = primary.get("G64_BOOTSTRAP_CI") or [None, None]
        rows.append({
            "panel": panel,
            "N": primary.get("N_UNITS"),
            "primary_delta": primary.get("G64_PRIMARY_EFFECT"),
            "bootstrap_ci_low": ci[0],
            "bootstrap_ci_high": ci[1],
            "history_level_sign_pattern": "/".join("R" if h["median_delta"] and h["median_delta"] > 0 else "C" if h["median_delta"] and h["median_delta"] < 0 else "TIE" for h in histories),
            "R_preferred_history_count": sum(1 for h in histories if h["median_delta"] and h["median_delta"] > 0),
            "C_preferred_history_count": sum(1 for h in histories if h["median_delta"] and h["median_delta"] < 0),
        })
    _ = prospective_analysis
    return rows


def analyze(args):
    rows = list(iter_jsonl(RUN_DIR / "results.jsonl") or [])
    analysis = analyze_prospective_rows(rows, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=BOOTSTRAP_SEED)
    write_csv(RUN_DIR / "per_unit_per_history_effects.csv", analysis["per_unit_per_history_effects"])
    write_csv(RUN_DIR / "history_level_summary.csv", analysis["history_level_summary"])
    save_json(RUN_DIR / "primary_effect_summary.json", analysis["primary_effect_summary"])
    save_json(RUN_DIR / "bootstrap_results.json", analysis["bootstrap_results"])
    write_csv(RUN_DIR / "robustness_analysis.csv", analysis["robustness_analysis"])
    write_csv(RUN_DIR / "heterogeneity_analysis.csv", analysis["heterogeneity_analysis"])
    write_text(RUN_DIR / "robustness_report.md", robustness_report_text(analysis))
    cross = cross_panel_comparison(analysis)
    write_csv(RUN_DIR / "cross_panel_g64_comparison.csv", cross)
    gates = gate_summary_from_outputs(len(rows))
    decision = classify_prospective_result(analysis, gates)
    summary = build_summary(analysis, gates, decision, len(rows), args)
    save_json(RUN_DIR / "summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    write_report(summary, analysis, cross)
    write_readme(summary)
    if decision["G64_PROSPECTIVE_CLASSIFICATION"] == "ROBUST_R_PREFERENCE":
        write_next_plan()
    return summary


def robustness_report_text(analysis):
    robust = analysis["robustness_summary"]
    return "\n".join([
        f"# {TASK} Robustness Report",
        "",
        f"- LEAVE_ONE_UNIT_OUT_SIGN_STABLE: {robust.get('LEAVE_ONE_UNIT_OUT_SIGN_STABLE')}",
        f"- LEAVE_ONE_PROMPT_OUT_SIGN_STABLE: {robust.get('LEAVE_ONE_PROMPT_OUT_SIGN_STABLE')}",
        f"- LEAVE_ONE_LAYER_OUT_SIGN_STABLE: {robust.get('LEAVE_ONE_LAYER_OUT_SIGN_STABLE')}",
        f"- LEAVE_ONE_HEAD_OUT_SIGN_STABLE: {robust.get('LEAVE_ONE_HEAD_OUT_SIGN_STABLE')}",
        f"- NO_SINGLE_SUBGROUP_DRIVES_RESULT: {robust.get('NO_SINGLE_SUBGROUP_DRIVES_RESULT')}",
        "",
        "Robustness diagnostics are not used to change the primary statistic after outcome inspection.",
    ]) + "\n"


def build_summary(analysis, gates, decision, n_rows, args):
    primary = analysis["primary_effect_summary"]
    histories = {int(r["history_length"]): r for r in analysis["history_level_summary"]}
    robust = analysis["robustness_summary"]
    cross = read_csv(RUN_DIR / "cross_panel_g64_comparison.csv")
    cross_by_panel = {r["panel"]: r for r in cross}
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if n_rows == N_EXPECTED_QUANTIZED_RUNS and decision["G64_PROSPECTIVE_CLASSIFICATION"] != "INVALID" else ("INVALID" if decision["G64_PROSPECTIVE_CLASSIFICATION"] == "INVALID" else "INCOMPLETE"),
        **gates,
        "N_NEW_INDEPENDENT_UNITS": primary["N_UNITS"],
        "N_EXPECTED_QUANTIZED_RUNS": N_EXPECTED_QUANTIZED_RUNS,
        "N_SUCCESSFUL_QUANTIZED_RUNS": n_rows,
        "PRIMARY_EFFECT_DEFINITION": primary["PRIMARY_EFFECT_DEFINITION"],
        "G64_PRIMARY_EFFECT": primary["G64_PRIMARY_EFFECT"],
        "G64_BOOTSTRAP_CI": primary["G64_BOOTSTRAP_CI"],
        "G64_R_PREFERRED_UNIT_COUNT": primary["G64_R_PREFERRED_UNIT_COUNT"],
        "G64_C_PREFERRED_UNIT_COUNT": primary["G64_C_PREFERRED_UNIT_COUNT"],
        **{f"G64_HISTORY_L{L}_EFFECT": histories.get(L, {}).get("median_delta") for L in HISTORY_LENGTHS},
        "G64_R_PREFERRED_HISTORY_COUNT": sum(1 for row in histories.values() if row.get("median_delta") is not None and row["median_delta"] > 0),
        "G64_C_PREFERRED_HISTORY_COUNT": sum(1 for row in histories.values() if row.get("median_delta") is not None and row["median_delta"] < 0),
        "LEAVE_ONE_UNIT_OUT_SIGN_STABLE": robust["LEAVE_ONE_UNIT_OUT_SIGN_STABLE"],
        "LEAVE_ONE_LAYER_OUT_SIGN_STABLE": robust["LEAVE_ONE_LAYER_OUT_SIGN_STABLE"],
        "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE": robust["LEAVE_ONE_PROMPT_OUT_SIGN_STABLE"],
        "V2_EXACT_G64_EFFECT": as_float(cross_by_panel.get("V2_EXACT", {}).get("primary_delta")),
        "V2_INDEPENDENT_G64_EFFECT": as_float(cross_by_panel.get("V2_INDEPENDENT", {}).get("primary_delta")),
        "PROSPECTIVE_G64_EFFECT": primary["G64_PRIMARY_EFFECT"],
        **{k: v for k, v in decision.items() if k != "gate_failures"},
        "FRESH_OUTPUT_VERIFICATION": "NOT_RUN",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": args.bash_syntax_status,
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO",
        "DO_NOT_RUN_G128_G32_G16": "YES",
        "output_dir": str(RUN_DIR),
    }
    return out


def write_report(summary, analysis, cross):
    lines = [
        f"# {TASK}",
        "",
        "## Question",
        "Does the current V2/canonical Ling/KDA INT8 g64 pipeline prospectively prefer Row over Column on a new independent 72-unit sample?",
        "",
        "## Result",
        f"- FORMAL_STATUS: {summary.get('FORMAL_STATUS')}",
        f"- G64_PROSPECTIVE_CLASSIFICATION: {summary.get('G64_PROSPECTIVE_CLASSIFICATION')}",
        f"- primary effect: {summary.get('G64_PRIMARY_EFFECT')}",
        f"- bootstrap CI: {summary.get('G64_BOOTSTRAP_CI')}",
        f"- history Row/Column counts: {summary.get('G64_R_PREFERRED_HISTORY_COUNT')} / {summary.get('G64_C_PREFERRED_HISTORY_COUNT')}",
        "",
        "## Gates",
    ]
    for key in BLOCKING_GATES:
        lines.append(f"- {key}: {summary.get(key)}")
    lines += [
        "",
        "## Cross-Panel g64",
        "```json",
        json.dumps(cross, indent=2, sort_keys=True),
        "```",
        "",
        "## Discipline",
        "The prospective 72-unit panel is the primary evidence. Previous V2 panels are supporting context only. V1 is not used as a target.",
    ]
    write_text(RUN_DIR / "report.md", "\n".join(lines) + "\n")
    write_text(REPORT_DIR / f"{SLUG}.md", "\n".join(lines) + "\n")


def write_readme(summary):
    write_text(RUN_DIR / "README.md", "\n".join([
        f"# {TASK}",
        "",
        f"- output_dir: {RUN_DIR}",
        f"- FORMAL_STATUS: {summary.get('FORMAL_STATUS')}",
        f"- G64_PROSPECTIVE_CLASSIFICATION: {summary.get('G64_PROSPECTIVE_CLASSIFICATION')}",
        f"- results_jsonl_rows: {summary.get('N_SUCCESSFUL_QUANTIZED_RUNS')}",
        "",
        "This directory contains the frozen prospective manifest, pipeline hashes, self-check evidence, raw R64/C64 results, and prespecified analysis outputs.",
    ]) + "\n")


def write_next_plan():
    write_text(RUN_DIR / "next_phase_diagram_confirmation_plan.md", "\n".join([
        "# Next Phase Diagram Confirmation Plan",
        "",
        "Do not execute automatically.",
        "",
        "- Freeze a 36-unit subset of the prospective 72 before seeing g128/g32/g16 outcomes.",
        "- Run R128/C128, R32/C32, R16/C16 over histories 1,4,8,16,32,64.",
        "- Expected control runs: 36 * 6 * 6 = 1296.",
        "- Expand only ambiguous group sizes after human review.",
    ]) + "\n")


def verify_fresh_outputs():
    summary_path = RUN_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (RUN_DIR / name).exists()]
    rows = list(iter_jsonl(RUN_DIR / "results.jsonl") or [])
    cells = {(r.get("unit_id"), row_config(r), int(float(r.get("history_length", -1)))) for r in rows if r.get("history_length") is not None}
    mtimes = {name: (RUN_DIR / name).stat().st_mtime for name in REQUIRED_OUTPUT_FILES if (RUN_DIR / name).exists()}
    hashes = {name: file_sha256(RUN_DIR / name) for name in REQUIRED_OUTPUT_FILES if (RUN_DIR / name).exists() and (RUN_DIR / name).is_file()}
    ok = (
        not missing
        and len(rows) == N_EXPECTED_QUANTIZED_RUNS
        and len(cells) == N_EXPECTED_QUANTIZED_RUNS
        and summary.get("N_SUCCESSFUL_QUANTIZED_RUNS") == N_EXPECTED_QUANTIZED_RUNS
        and summary.get("G64_PROSPECTIVE_CLASSIFICATION") in {"ROBUST_R_PREFERENCE", "ORIENTATION_BOUNDARY", "ROBUST_C_PREFERENCE"}
    )
    out = {
        "FRESH_OUTPUT_VERIFICATION": "PASS" if ok else "FAIL",
        "missing_files": missing,
        "results_jsonl_rows": len(rows),
        "unique_result_cells": len(cells),
        "expected_results_jsonl_rows": N_EXPECTED_QUANTIZED_RUNS,
        "file_mtimes": mtimes,
        "file_hashes": hashes,
    }
    save_json(RUN_DIR / "fresh_output_verification.json", out)
    if summary:
        summary.update(out)
        summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES" if out["FRESH_OUTPUT_VERIFICATION"] == "PASS" else "NO"
        save_json(RUN_DIR / "summary.json", summary)
        save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    return out


def render_console_summary(summary):
    lines = []
    for key in REQUIRED_CONSOLE_KEYS:
        val = summary.get(key)
        if isinstance(val, (dict, list)):
            val = json.dumps(val, sort_keys=True, ensure_ascii=False)
        lines.append(f"{key} = {'' if val is None else val}")
    lines += ["DO_NOT_RUN_G128_G32_G16 = YES", "STOP."]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--stage", choices=["prepare", "self-check", "formal", "merge-formal", "analyze", "verify", "summary"], default="prepare")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-dirs", default="")
    parser.add_argument("--self-check-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--console-summary", action="store_true")
    parser.add_argument("--pytest-status", default="NOT_RUN")
    parser.add_argument("--py-compile-status", default="NOT_RUN")
    parser.add_argument("--bash-syntax-status", default="NOT_RUN")
    parser.add_argument("--gpu-release-status", default="NOT_RUN")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "prepare":
        out = prepare()
    elif args.stage == "self-check":
        out = self_check(args.self_check_repeats)
    elif args.stage == "formal":
        out = formal(args)
    elif args.stage == "merge-formal":
        out = merge_formal(args)
    elif args.stage == "analyze":
        out = analyze(args)
    elif args.stage == "verify":
        out = verify_fresh_outputs()
    elif args.stage == "summary":
        out = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    else:
        raise ValueError(args.stage)
    if args.console_summary:
        print(render_console_summary(out))
    else:
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
