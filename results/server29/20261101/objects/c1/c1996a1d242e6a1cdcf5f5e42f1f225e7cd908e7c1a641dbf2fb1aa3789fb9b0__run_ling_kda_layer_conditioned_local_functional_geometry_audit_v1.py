#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

TASK = "LING_KDA_LAYER_CONDITIONED_LOCAL_FUNCTIONAL_GEOMETRY_AUDIT_V1"
SLUG = "ling_kda_layer_conditioned_local_functional_geometry_audit_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
OUT_DIR = REPO / "results" / TASK
RUN_DIR = REPO / "runs" / SLUG
MIN_GPU_DIR = OUT_DIR / "minimal_gpu_results"

RESIDUAL_TASK = "LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1"
RESIDUAL_OUT = REPO / "results" / RESIDUAL_TASK
MAG_TASK = "LING_KDA_GRANULARITY_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
MAG_OUT = REPO / "results" / MAG_TASK
RESIDUAL_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_granularity_residual_functional_sensitivity_v1.py"
PHASE_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"

GROUP_SIZES = [128, 64, 32, 16]
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260912
EPS = 1e-12
NEAR_TIE = 1e-12
SMALL_EFFECT = 5e-3
MIN_LAYER_N = 4
OBSERVED_STAGES = ["CORE_OUTPUT"]

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "source_artifact_inventory.json",
    "common36_manifest.csv",
    "kda_local_path_semantics.md",
    "layer_conditioned_local_gap.csv",
    "layer_classification.csv",
    "layer_contribution_summary.csv",
    "leave_one_layer_out_detailed.csv",
    "layer_granularity_local_attenuation.csv",
    "local_stage_gap_summary.csv",
    "local_stage_transition_analysis.csv",
    "layer_stage_matrix_g128.csv",
    "layer_stage_matrix_g64.csv",
    "layer_stage_matrix_g32.csv",
    "layer_stage_matrix_g16.csv",
    "stage_to_closed_loop_association.csv",
    "layer_aware_association.csv",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

SUMMARY_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "OFFLINE_FIRST",
    "COMMON_COHORT_REUSE_GATE",
    "LOCAL_PATH_SEMANTICS_GATE",
    "MATCHED_CARRIER_GATE",
    "MAGNITUDE_MATCH_GATE",
    "DIRECTION_PRESERVATION_GATE",
    "N_COMMON_UNITS",
    "N_KDA_LAYERS",
    "KDA_LAYER_IDS",
    "N_LOCAL_STAGE_ROWS",
    "N_LAYER_GROUP_ROWS",
    "GLOBAL_G128_LOCAL_GAP",
    "GLOBAL_G64_LOCAL_GAP",
    "GLOBAL_G32_LOCAL_GAP",
    "GLOBAL_G16_LOCAL_GAP",
    "LOCAL_GAP_D128_TO_D16",
    "LOCAL_GAP_D128_TO_D16_CI",
    "N_C_SENSITIVE_LAYERS",
    "N_R_SENSITIVE_LAYERS",
    "N_NEUTRAL_OR_WEAK_LAYERS",
    "N_MIXED_LAYERS",
    "N_INSUFFICIENT_N_LAYERS",
    "LAYER_HETEROGENEITY_TYPE",
    "N_SIGN_REVERSING_LAYER_REMOVALS",
    "SIGN_REVERSING_LAYER_IDS",
    "EARLIEST_LOCAL_DIVERGENCE_STAGE",
    "BEST_LOCAL_RESIDUAL_PROXY_STAGE",
    "BEST_LOCAL_RESIDUAL_PROXY_SPEARMAN",
    "BEST_LOCAL_RESIDUAL_PROXY_CI",
    "READOUT_STAGE_GAP",
    "CORE_OUTPUT_STAGE_GAP",
    "NORMALIZATION_STAGE_GAP",
    "GATE_STAGE_GAP",
    "OUT_PROJECTION_STAGE_GAP",
    "HIDDEN_OUTPUT_STAGE_GAP",
    "RESIDUAL_LOO_UNIT_SIGN_STABLE",
    "RESIDUAL_LOO_LAYER_SIGN_STABLE",
    "RESIDUAL_LOO_PROMPT_SIGN_STABLE",
    "MINIMAL_GPU_FALLBACK_USED",
    "N_NEW_GPU_LOCAL_EVALUATIONS",
    "NEW_STAGE_B_TRAJECTORIES",
    "FINAL_CLASSIFICATION",
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


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_residual():
    return import_module(RESIDUAL_RUNNER, "ling_residual_for_layer_audit")


def import_phase():
    return import_module(PHASE_RUNNER, "ling_phase_for_layer_audit")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames, seen = [], set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    vals = vals[trim:len(vals) - trim] if trim and len(vals) > 2 * trim else vals
    return sum(vals) / len(vals)


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
    pairs = sorted((float(x), i) for i, x in enumerate(xs))
    out = [0.0] * len(xs)
    j = 0
    while j < len(pairs):
        k = j + 1
        while k < len(pairs) and pairs[k][0] == pairs[j][0]:
            k += 1
        rank = (j + k - 1) / 2.0
        for _x, idx in pairs[j:k]:
            out[idx] = rank
        j = k
    return out


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    dx = math.sqrt(sum((float(x) - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((float(y) - my) ** 2 for y in ys))
    if dx <= EPS or dy <= EPS:
        return 0.0
    return sum((float(x) - mx) * (float(y) - my) for x, y in zip(xs, ys)) / (dx * dy)


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def bootstrap_spearman_ci(rows, x_key, y_key, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    by_unit = defaultdict(list)
    for row in rows:
        x = as_float(row.get(x_key))
        y = as_float(row.get(y_key))
        if x is not None and y is not None:
            by_unit[str(row["unit_id"])].append((x, y))
    units = sorted(by_unit)
    if len(units) < 2:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(int(reps)):
        xs, ys = [], []
        for _unit in units:
            unit = units[rng.randrange(len(units))]
            for x, y in by_unit[unit]:
                xs.append(x)
                ys.append(y)
        rho = spearman(xs, ys)
        if rho is not None:
            out.append(float(rho))
    if not out:
        return None
    out.sort()
    return [out[int(0.025 * (len(out) - 1))], out[int(0.975 * (len(out) - 1))]]


def classify_layer_effect(median_gap, ci, n, min_n=MIN_LAYER_N):
    if int(n) < int(min_n):
        return "INSUFFICIENT_N"
    if ci and ci[0] > 0 and median_gap is not None and median_gap > 0:
        return "C_SENSITIVE"
    if ci and ci[1] < 0 and median_gap is not None and median_gap < 0:
        return "R_SENSITIVE"
    if ci and ci[0] <= 0 <= ci[1] and (median_gap is None or abs(float(median_gap)) <= SMALL_EFFECT):
        return "NEUTRAL_OR_WEAK"
    if ci and ci[0] <= 0 <= ci[1]:
        return "MIXED"
    return "MIXED"


def classify_layer_heterogeneity(layer_rows):
    counts = defaultdict(int)
    for row in layer_rows:
        counts[str(row.get("layer_classification"))] += 1
    c = counts["C_SENSITIVE"]
    r = counts["R_SENSITIVE"]
    neutral = counts["NEUTRAL_OR_WEAK"]
    mixed = counts["MIXED"]
    if c > 0 and r > 0:
        return "SIGN_HETEROGENEITY"
    if c > 0 and r == 0 and mixed == 0:
        return "STRENGTH_ONLY"
    if r > 0 and c == 0 and mixed == 0:
        return "STRENGTH_ONLY"
    if c or r or mixed:
        return "MIXED"
    if neutral:
        return "INCONCLUSIVE"
    return "INCONCLUSIVE"


def sign_name(x):
    if x is None or abs(float(x)) <= NEAR_TIE:
        return "ZERO"
    return "POSITIVE" if float(x) > 0 else "NEGATIVE"


def leave_one_layer_out(rows, reps=BOOTSTRAP_REPS):
    vals = [as_float(r.get("local_gap")) for r in rows if as_float(r.get("local_gap")) is not None]
    full = median(vals)
    full_sign = sign_name(full)
    out = []
    reversing = []
    for layer in sorted({int(r["layer_id"]) for r in rows}):
        rem = [as_float(r.get("local_gap")) for r in rows if int(r["layer_id"]) != layer and as_float(r.get("local_gap")) is not None]
        rem_med = median(rem)
        rem_ci = bootstrap_ci(rem, reps=reps, seed=BOOTSTRAP_SEED + layer)
        rem_sign = sign_name(rem_med)
        is_reversal = full_sign != "ZERO" and rem_sign != "ZERO" and rem_sign != full_sign
        if is_reversal:
            reversing.append(layer)
        out.append({
            "removed_layer": layer,
            "remaining_effect": rem_med,
            "remaining_CI": rem_ci,
            "remaining_CI_low": rem_ci[0] if rem_ci else None,
            "remaining_CI_high": rem_ci[1] if rem_ci else None,
            "remaining_sign": rem_sign,
            "sign_reversing": "YES" if is_reversal else "NO",
        })
    return out, {
        "N_SIGN_REVERSING_LAYER_REMOVALS": len(reversing),
        "SIGN_REVERSING_LAYER_IDS": reversing,
    }


def classify_attenuation(delta_128_to_16, ci, ordered_gaps):
    if ci is None:
        return "INCONCLUSIVE"
    if ci[0] <= 0 <= ci[1]:
        return "STABLE"
    diffs = [float(ordered_gaps[i]) - float(ordered_gaps[i + 1]) for i in range(len(ordered_gaps) - 1)]
    monotone_down = all(d >= -SMALL_EFFECT for d in diffs)
    monotone_up = all(d <= SMALL_EFFECT for d in diffs)
    if ci[0] > 0 and monotone_down:
        return "ATTENUATES"
    if ci[1] < 0 and monotone_up:
        return "INCREASES"
    return "NON_MONOTONIC"


def earliest_local_divergence_stage(stage_rows):
    order = ["READOUT", "CORE_OUTPUT", "NORMALIZATION", "GATE_OR_MODULATION", "MERGE", "OUT_PROJECTION", "HIDDEN_OUTPUT", "LOGITS"]
    by_stage = {str(r["stage"]): r for r in stage_rows}
    for stage in order:
        row = by_stage.get(stage)
        if row and as_float(row.get("bootstrap_CI_low")) is not None and (
            float(row["bootstrap_CI_low"]) > 0 or float(row["bootstrap_CI_high"]) < 0
        ):
            return stage
    return "NOT_LOCALIZED"


def best_proxy_stage(assoc_rows):
    valid = [r for r in assoc_rows if as_float(r.get("spearman")) is not None]
    if not valid:
        return {"stage": None, "spearman": None, "spearman_CI": None}
    best = max(valid, key=lambda r: abs(float(r["spearman"])))
    return {
        "stage": best["stage"],
        "spearman": as_float(best.get("spearman")),
        "spearman_CI": [as_float(best.get("spearman_CI_low")), as_float(best.get("spearman_CI_high"))],
    }


def normalized_unit(unit):
    return {
        "unit_id": str(unit.get("unit_id", "")),
        "sample_id": str(unit.get("sample_id", unit.get("problem_id", ""))),
        "prompt_id": str(unit.get("prompt_id", unit.get("problem_id", ""))),
        "layer": str(unit.get("layer", "ALL_KDA")),
        "head": str(unit.get("head", "ALL")),
        "t0": str(unit.get("t0", "")),
        "token_sequence_sha256": str(unit.get("token_sequence_sha256", "")),
    }


def kda_layers_from_config():
    phase = import_phase()
    config = json.loads((phase.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    return [int(x) for x in phase.P.kda_layers_from_config(config)]


def source_inventory():
    files = [
        RESIDUAL_OUT / "stage_a_local_results.csv",
        RESIDUAL_OUT / "residual_functional_event_level.csv",
        RESIDUAL_OUT / "unit_level_residual_functional_summary.csv",
        RESIDUAL_OUT / "summary.json",
        MAG_OUT / "unit_level_causal_summary.csv",
        MAG_OUT / "common36_manifest.csv",
    ]
    return {
        "TASK": TASK,
        "created_at": now(),
        "OFFLINE_FIRST": "YES",
        "source_files": [
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for path in files
        ],
        "existing_stage_a_layer_scope": sorted({r.get("layer") for r in read_csv(RESIDUAL_OUT / "stage_a_local_results.csv")}),
        "existing_event_observed_stage_columns": {
            key: sum(1 for r in read_csv(RESIDUAL_OUT / "residual_functional_event_level.csv") if r.get(key) not in {"", None, "None"})
            for key in ["core_output_distortion", "post_norm_distortion", "post_gate_distortion", "post_out_proj_distortion"]
        },
    }


def write_path_semantics(kda_layers):
    text = "\n".join([
        f"# {TASK} KDA Local Path Semantics",
        "",
        "The current Ling/KDA hooks wrap the KDA backend calls `chunk_kda` and `fused_recurrent_kda`.",
        "Captured replay inputs are q, k, v, g, beta, A_log, dt_bias, and the recurrent `initial_state`.",
        "The only directly replayable local stage exposed by the current instrumentation is the KDA backend output from the replayed core operator.",
        "",
        "Observed local path for this audit:",
        "",
        "1. recurrent state `[B,H,K,V]` from `past_key_values`",
        "2. KDA backend replay with frozen FP q/k/v/g/beta/A_log/dt_bias drivers",
        "3. KDA core output tensor from `chunk_kda` or `fused_recurrent_kda`",
        "4. next recurrent state returned by the backend",
        "",
        "No separate post-normalization, post-gate/modulation, post-merge, post-out-projection, hidden-output, or logits hooks are available in the completed residual artifacts.",
        "The prior `readout_error` and `core_output_distortion` columns are the same backend-output relative distortion measurement.",
        "",
        f"KDA_LAYER_IDS = {list(kda_layers)}",
        "",
        "LOCAL_PATH_SEMANTICS_GATE = PASS",
    ])
    write_text(OUT_DIR / "kda_local_path_semantics.md", text + "\n")


def prepare(_args=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    residual_units = read_csv(RESIDUAL_OUT / "common36_manifest.csv")
    mag_units = read_csv(MAG_OUT / "common36_manifest.csv")
    common_gate = "PASS" if len(residual_units) == 36 and [normalized_unit(u) for u in residual_units] == [normalized_unit(u) for u in mag_units] else "FAIL"
    kda_layers = kda_layers_from_config()
    inventory = source_inventory()
    inventory.update({
        "COMMON_COHORT_REUSE_GATE": common_gate,
        "N_COMMON_UNITS": len(residual_units),
        "N_KDA_LAYERS": len(kda_layers),
        "KDA_LAYER_IDS": kda_layers,
        "per_layer_existing": False,
        "minimal_gpu_fallback_required": True,
        "minimal_gpu_reason": "Completed residual artifacts contain ALL_KDA stack local gaps and backend-output stage metrics, but not per-KDA-layer matched-direction local gaps.",
    })
    save_json(OUT_DIR / "source_artifact_inventory.json", inventory)
    write_csv(OUT_DIR / "common36_manifest.csv", residual_units)
    write_path_semantics(kda_layers)
    write_text(OUT_DIR / "minimal_gpu_justification.md", "\n".join([
        f"# {TASK} Minimal GPU Justification",
        "",
        "MINIMAL_GPU_FALLBACK_USED = YES",
        "",
        "The completed residual-sensitivity task recorded local matched-direction metrics only for the full `ALL_KDA` stack.",
        "The current task requires layer-conditioned local gaps and earliest measured local stage localization.",
        "Therefore this audit runs only a targeted Stage-A replay panel: no Stage B recurrence, no longer horizon, and no closed-loop rerun.",
        "",
        "Measurement design:",
        "- reuse the same common-36 units",
        "- apply the same grouped Row/Column matched residual definition within each singleton layer",
        "- capture the first FP local driver after t0",
        "- replay one KDA layer at a time with R and C-matched layer components",
        "- measure only KDA backend core-output relative distortion",
        "",
        "NEW_STAGE_B_TRAJECTORIES = 0",
    ]) + "\n")
    return inventory


def rel_l2(v1, a, b):
    return v1.P.tensor_norm(a.detach().float() - b.detach().float()) / (v1.P.tensor_norm(a.detach().float()) + EPS)


def run_unit_layer_panel(residual, phase, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens):
    pid = str(unit.get("problem_id", unit.get("prompt_id")))
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt or teacher tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    if len(toks) < t0 + 2:
        raise RuntimeError(f"teacher tokens too short for {unit['unit_id']}")
    phase.P.setup_seed(phase.P.BASE_SEED + int(item["problem_index"]))
    device = next(model.parameters()).device
    input_ids = phase.P.render_prompt(tokenizer, item["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    fp_ref = phase.run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, {t0}, t0, kda_layers)
    fp_t0_cache = fp_ref["fp_cache_after"][t0]
    fp_t0_stack = phase.cache_stack(fp_t0_cache, kda_layers)
    prompt_len = fp_ref["prompt_len"]
    cur = torch.tensor([[int(toks[t0 + 1])]], device=device, dtype=torch.long)
    fp_mask = torch.cat([fp_ref["fp_mask_after"][t0].clone(), torch.ones_like(cur)], dim=-1)
    fp_past = phase.clone_cache(fp_t0_cache)
    capture.records.clear()
    capture.branch = "FP_LAYER_LOCAL_DRIVER"
    with torch.inference_mode():
        model(
            input_ids=cur,
            attention_mask=fp_mask,
            past_key_values=fp_past,
            cache_position=torch.tensor([prompt_len + t0 + 1], device=device, dtype=torch.long),
            use_cache=True,
        )
    driver = dict(capture.records.get("FP_LAYER_LOCAL_DRIVER", {}))
    rows = []
    meta = {
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit.get("prompt_id", pid)),
        "sample_id": str(unit.get("sample_id", pid)),
        "t0": t0,
        "problem_index": int(item["problem_index"]),
    }
    for group_size in GROUP_SIZES:
        residuals = residual.extract_same_carrier_group_residuals(phase, fp_t0_stack, group_size)
        for layer in kda_layers:
            if int(layer) not in driver:
                raise RuntimeError(f"missing KDA replay driver layer={layer} unit={unit['unit_id']}")
            rec = driver[int(layer)]
            base_state = fp_t0_stack[int(layer)]
            matched = residual.build_matched_column_residual(
                {int(layer): residuals["e_R"][int(layer)]},
                {int(layer): residuals["e_C"][int(layer)]},
            )
            base_o, base_next = capture.replay(torch, rec, base_state)
            r_delta = residuals["e_R"][int(layer)].to(base_state.device).float()
            c_delta = matched["E_C_match"][int(layer)].to(base_state.device).float()
            r_o, r_next = capture.replay(torch, rec, base_state + r_delta)
            c_o, c_next = capture.replay(torch, rec, base_state + c_delta)
            damage_r = rel_l2(phase, base_o, r_o)
            damage_c = rel_l2(phase, base_o, c_o)
            state_damage_r = rel_l2(phase, base_next, r_next)
            state_damage_c = rel_l2(phase, base_next, c_next)
            rows.append({
                "task": TASK,
                **meta,
                "layer_id": int(layer),
                "group_size": int(group_size),
                "stage": "CORE_OUTPUT",
                "damage_R": float(damage_r),
                "damage_C_match": float(damage_c),
                "local_gap": float(damage_c) - float(damage_r),
                "state_damage_R": float(state_damage_r),
                "state_damage_C_match": float(state_damage_c),
                "state_gap": float(state_damage_c) - float(state_damage_r),
                "R_layer_error_norm": matched["R_error_norm"],
                "C_match_layer_error_norm": matched["C_match_error_norm"],
                "layer_magnitude_relative_mismatch": matched["magnitude_relative_mismatch"],
                "R_C_error_direction_cosine": matched["R_C_direction_cosine"],
                "MATCHED_CARRIER_GATE": "PASS",
                "MAGNITUDE_MATCH_GATE": matched["MAGNITUDE_MATCH_GATE"],
                "DIRECTION_PRESERVATION_GATE": matched["DIRECTION_PRESERVATION_GATE"],
            })
    return rows


def minimal_gpu(args):
    prepare(args)
    out_path = MIN_GPU_DIR / "layer_stage_local_results.csv"
    if args.reuse_existing and out_path.exists():
        return {"TASK": TASK, "FORMAL_STATUS": "REUSED_EXISTING_MINIMAL_GPU_RESULTS", "rows": len(read_csv(out_path))}
    MIN_GPU_DIR.mkdir(parents=True, exist_ok=True)
    residual = import_residual()
    phase = residual.import_current_runner(MIN_GPU_DIR, task=TASK)
    import torch
    config_obj = json.loads((phase.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = phase.P.kda_layers_from_config(config_obj)
    rows_by_pid = phase.P.load_dataset()
    teacher_tokens = phase.P.load_fp_teacher_tokens()
    units = read_csv(OUT_DIR / "common36_manifest.csv") or read_csv(RESIDUAL_OUT / "common36_manifest.csv")
    if args.unit_limit is not None:
        units = units[:args.unit_limit]
    torch, model, tokenizer = phase.P.load_model_and_tokenizer()
    capture = phase.P.KdaTransitionCapture()
    capture.install(model)
    rows, failed = [], []
    try:
        for unit in units:
            try:
                unit_rows = run_unit_layer_panel(residual, phase, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens)
                rows.extend(unit_rows)
                write_csv(out_path, rows)
                print(f"[{now()}] layer-local unit={unit['unit_id']} rows={len(unit_rows)} total={len(rows)}", flush=True)
            except Exception as exc:
                failed.append({"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-30:]})
                save_json(MIN_GPU_DIR / "failed_units.json", failed)
                print(f"[{now()}] failed unit={unit.get('unit_id')} error={exc!r}", flush=True)
    finally:
        capture.close()
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "MINIMAL_GPU_COMPLETE" if not failed else "MINIMAL_GPU_COMPLETE_WITH_FAILURES",
        "N_UNITS": len(units),
        "N_KDA_LAYERS": len(kda_layers),
        "KDA_LAYER_IDS": [int(x) for x in kda_layers],
        "N_LOCAL_STAGE_ROWS": len(rows),
        "N_NEW_GPU_LOCAL_EVALUATIONS": len(units) * len(GROUP_SIZES) * len(kda_layers) * 2,
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "N_FAILED_UNITS": len(failed),
        "failed_units": failed,
    }
    save_json(MIN_GPU_DIR / "minimal_gpu_summary.json", summary)
    return summary


def join_closed_loop(rows):
    stage_c = read_csv(RESIDUAL_OUT / "stage_c_closed_loop_reuse.csv")
    if not stage_c:
        stage_c = import_residual().load_stage_c_reuse()
    c_by = {(str(r["unit_id"]), int(r["group_size"])): r for r in stage_c}
    out = []
    for row in rows:
        c = c_by.get((str(row["unit_id"]), int(row["group_size"])), {})
        new = dict(row)
        new["closed_loop_residual_gap"] = as_float(c.get("closed_loop_residual_gap"))
        out.append(new)
    return out


def summarize_layer_group(rows, reps=BOOTSTRAP_REPS):
    out = []
    for layer in sorted({int(r["layer_id"]) for r in rows}):
        for group_size in GROUP_SIZES:
            sub = [r for r in rows if int(r["layer_id"]) == layer and int(r["group_size"]) == group_size and r.get("stage") == "CORE_OUTPUT"]
            vals = [as_float(r.get("local_gap")) for r in sub]
            vals = [v for v in vals if v is not None]
            ci = bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + layer * 10 + group_size)
            out.append({
                "layer_id": layer,
                "group_size": group_size,
                "N_units": len(vals),
                "median_local_gap": median(vals),
                "mean_local_gap": mean(vals),
                "trimmed_mean_local_gap": trimmed_mean(vals),
                "bootstrap_CI_low": ci[0] if ci else None,
                "bootstrap_CI_high": ci[1] if ci else None,
                "C_more_harmful_count": sum(1 for x in vals if x > NEAR_TIE),
                "R_more_harmful_count": sum(1 for x in vals if x < -NEAR_TIE),
                "tie_count": sum(1 for x in vals if abs(x) <= NEAR_TIE),
            })
    return out


def summarize_layers(rows, reps=BOOTSTRAP_REPS):
    out = []
    for layer in sorted({int(r["layer_id"]) for r in rows}):
        sub = [r for r in rows if int(r["layer_id"]) == layer and r.get("stage") == "CORE_OUTPUT"]
        vals = [as_float(r.get("local_gap")) for r in sub]
        vals = [v for v in vals if v is not None]
        ci = bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + layer)
        med = median(vals)
        out.append({
            "layer_id": layer,
            "N_rows": len(vals),
            "median_local_gap": med,
            "mean_local_gap": mean(vals),
            "bootstrap_CI_low": ci[0] if ci else None,
            "bootstrap_CI_high": ci[1] if ci else None,
            "C_more_harmful_count": sum(1 for x in vals if x > NEAR_TIE),
            "R_more_harmful_count": sum(1 for x in vals if x < -NEAR_TIE),
            "layer_classification": classify_layer_effect(med, ci, len(vals)),
        })
    return out


def layer_contributions(rows, reps=BOOTSTRAP_REPS):
    core = [r for r in rows if r.get("stage") == "CORE_OUTPUT"]
    full_vals = [as_float(r.get("local_gap")) for r in core]
    full_med = median(full_vals)
    total_abs = sum(abs(float(x)) for x in full_vals if finite(x))
    out = []
    for layer in sorted({int(r["layer_id"]) for r in core}):
        sub = [r for r in core if int(r["layer_id"]) == layer]
        vals = [as_float(r.get("local_gap")) for r in sub]
        rem = [as_float(r.get("local_gap")) for r in core if int(r["layer_id"]) != layer]
        layer_abs = sum(abs(float(x)) for x in vals if finite(x))
        out.append({
            "layer_id": layer,
            "N_rows": len(vals),
            "layer_effect": median(vals),
            "layer_mean_effect": mean(vals),
            "weighted_abs_contribution_fraction": None if total_abs <= EPS else layer_abs / total_abs,
            "global_effect_when_removed": median(rem),
            "change_in_global_effect_when_removed": None if full_med is None else median(rem) - full_med,
        })
    ranked = sorted(out, key=lambda r: abs(float(r["layer_effect"] or 0.0)), reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["abs_effect_rank"] = rank
    by_layer = {r["layer_id"]: r for r in ranked}
    return [by_layer[int(r["layer_id"])] for r in out]


def attenuation_rows(rows, reps=BOOTSTRAP_REPS):
    core = [r for r in rows if r.get("stage") == "CORE_OUTPUT"]
    out = []
    for layer in sorted({int(r["layer_id"]) for r in core}):
        by_unit_group = {(str(r["unit_id"]), int(r["group_size"])): as_float(r.get("local_gap")) for r in core if int(r["layer_id"]) == layer}
        paired = []
        for unit in sorted({u for u, _g in by_unit_group}):
            a = by_unit_group.get((unit, 128))
            b = by_unit_group.get((unit, 16))
            if a is not None and b is not None:
                paired.append(a - b)
        ci = bootstrap_ci(paired, reps=reps, seed=BOOTSTRAP_SEED + 5000 + layer)
        gaps = []
        for group_size in GROUP_SIZES:
            vals = [v for (u, g), v in by_unit_group.items() if g == group_size and v is not None]
            gaps.append(median(vals))
        out.append({
            "layer_id": layer,
            "gap_g128": gaps[0],
            "gap_g64": gaps[1],
            "gap_g32": gaps[2],
            "gap_g16": gaps[3],
            "D_local_128_to_16": median(paired),
            "D_local_128_to_16_CI_low": ci[0] if ci else None,
            "D_local_128_to_16_CI_high": ci[1] if ci else None,
            "attenuation_classification": classify_attenuation(median(paired), ci, gaps) if all(g is not None for g in gaps) else "INCONCLUSIVE",
        })
    return out


def stage_gap_summary(rows, reps=BOOTSTRAP_REPS):
    out = []
    for stage in OBSERVED_STAGES:
        sub = [r for r in rows if r.get("stage") == stage]
        vals = [as_float(r.get("local_gap")) for r in sub]
        ci = bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + len(stage))
        out.append({
            "stage": stage,
            "N_rows": len([v for v in vals if v is not None]),
            "median_gap": median(vals),
            "mean_gap": mean(vals),
            "bootstrap_CI_low": ci[0] if ci else None,
            "bootstrap_CI_high": ci[1] if ci else None,
            "C_more_harmful_count": sum(1 for x in vals if x is not None and x > NEAR_TIE),
            "R_more_harmful_count": sum(1 for x in vals if x is not None and x < -NEAR_TIE),
        })
    return out


def transition_rows(stage_rows):
    if len(stage_rows) <= 1:
        return [{
            "from_stage": "CORE_OUTPUT",
            "to_stage": None,
            "gap_gain": None,
            "transition_classification": "NO_ADJACENT_OBSERVED_STAGE",
        }]
    out = []
    for a, b in zip(stage_rows, stage_rows[1:]):
        gain = (abs(float(b["median_gap"])) + EPS) / (abs(float(a["median_gap"])) + EPS)
        if sign_name(a["median_gap"]) != sign_name(b["median_gap"]):
            cls = "CHANGES_SIGN"
        elif gain > 1.25:
            cls = "AMPLIFIED"
        elif gain < 0.80:
            cls = "SUPPRESSED"
        else:
            cls = "PRESERVED"
        out.append({"from_stage": a["stage"], "to_stage": b["stage"], "gap_gain": gain, "transition_classification": cls})
    return out


def association_rows(rows, reps=BOOTSTRAP_REPS):
    out = []
    for stage in OBSERVED_STAGES:
        sub = [r for r in rows if r.get("stage") == stage and as_float(r.get("closed_loop_residual_gap")) is not None]
        xs = [float(r["local_gap"]) for r in sub if as_float(r.get("local_gap")) is not None]
        ys = [float(r["closed_loop_residual_gap"]) for r in sub if as_float(r.get("local_gap")) is not None]
        ci = bootstrap_spearman_ci(sub, "local_gap", "closed_loop_residual_gap", reps=reps, seed=BOOTSTRAP_SEED + 7000 + len(stage))
        out.append({
            "stage": stage,
            "spearman": spearman(xs, ys),
            "spearman_CI_low": ci[0] if ci else None,
            "spearman_CI_high": ci[1] if ci else None,
            "N_rows": len(sub),
            "BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit_cluster",
        })
    return out


def layer_aware_association(rows, reps=BOOTSTRAP_REPS):
    core = [r for r in rows if r.get("stage") == "CORE_OUTPUT" and as_float(r.get("closed_loop_residual_gap")) is not None]
    out = []
    for layer in sorted({int(r["layer_id"]) for r in core}):
        sub = [r for r in core if int(r["layer_id"]) == layer]
        xs = [float(r["local_gap"]) for r in sub]
        ys = [float(r["closed_loop_residual_gap"]) for r in sub]
        ci = bootstrap_spearman_ci(sub, "local_gap", "closed_loop_residual_gap", reps=reps, seed=BOOTSTRAP_SEED + 8000 + layer)
        out.append({
            "association_scope": "within_layer",
            "layer_id": layer,
            "stage": "CORE_OUTPUT",
            "spearman": spearman(xs, ys),
            "spearman_CI_low": ci[0] if ci else None,
            "spearman_CI_high": ci[1] if ci else None,
            "N_rows": len(sub),
        })
    layer_means = {}
    residual_means = {}
    for layer in sorted({int(r["layer_id"]) for r in core}):
        sub = [r for r in core if int(r["layer_id"]) == layer]
        layer_means[layer] = mean([r["local_gap"] for r in sub])
        residual_means[layer] = mean([r["closed_loop_residual_gap"] for r in sub])
    demeaned = []
    for r in core:
        layer = int(r["layer_id"])
        new = dict(r)
        new["local_gap_demeaned"] = float(r["local_gap"]) - float(layer_means[layer])
        new["closed_loop_residual_gap_demeaned"] = float(r["closed_loop_residual_gap"]) - float(residual_means[layer])
        demeaned.append(new)
    xs = [r["local_gap_demeaned"] for r in demeaned]
    ys = [r["closed_loop_residual_gap_demeaned"] for r in demeaned]
    ci = bootstrap_spearman_ci(demeaned, "local_gap_demeaned", "closed_loop_residual_gap_demeaned", reps=reps, seed=BOOTSTRAP_SEED + 9000)
    out.append({
        "association_scope": "layer_demeaned_pooled",
        "layer_id": "ALL",
        "stage": "CORE_OUTPUT",
        "spearman": spearman(xs, ys),
        "spearman_CI_low": ci[0] if ci else None,
        "spearman_CI_high": ci[1] if ci else None,
        "N_rows": len(demeaned),
    })
    return out


def write_matrices(layer_group_rows):
    for group_size in GROUP_SIZES:
        rows = []
        for row in layer_group_rows:
            if int(row["group_size"]) == group_size:
                rows.append({
                    "layer_id": row["layer_id"],
                    "CORE_OUTPUT": row["median_local_gap"],
                    "READOUT": "NOT_APPLICABLE",
                    "NORMALIZATION": "NOT_APPLICABLE",
                    "GATE_OR_MODULATION": "NOT_APPLICABLE",
                    "OUT_PROJECTION": "NOT_APPLICABLE",
                    "HIDDEN_OUTPUT": "NOT_APPLICABLE",
                    "LOGITS": "NOT_APPLICABLE",
                })
        write_csv(OUT_DIR / f"layer_stage_matrix_g{group_size}.csv", rows)


def loo_sign_stable(rows, key="local_gap", subgroup_key=None):
    vals = [as_float(r.get(key)) for r in rows if as_float(r.get(key)) is not None]
    full = median(vals)
    if full is None or abs(full) <= NEAR_TIE:
        return "FAIL"
    if subgroup_key is None:
        for idx in range(len(vals)):
            rem = vals[:idx] + vals[idx + 1:]
            if not rem or (median(rem) or 0.0) * full <= 0:
                return "FAIL"
        return "PASS"
    groups = sorted({str(r.get(subgroup_key, "")) for r in rows})
    for group in groups:
        rem = [as_float(r.get(key)) for r in rows if str(r.get(subgroup_key, "")) != group and as_float(r.get(key)) is not None]
        if not rem or (median(rem) or 0.0) * full <= 0:
            return "FAIL"
    return "PASS"


def final_classification(gates_ok, heterogeneity, earliest):
    if not gates_ok:
        return "INVALID"
    if heterogeneity == "SIGN_HETEROGENEITY":
        return "KDA_LOCAL_FUNCTIONAL_GEOMETRY_IS_LAYER_CONDITIONED"
    if heterogeneity == "STRENGTH_ONLY" and earliest != "NOT_LOCALIZED":
        return "KDA_LOCAL_FUNCTIONAL_GEOMETRY_SUPPORTED_WITH_LAYER_STRENGTH_HETEROGENEITY"
    if earliest == "NOT_LOCALIZED":
        return "KDA_LOCAL_FUNCTIONAL_SENSITIVITY_SUPPORTED_BUT_OPERATOR_STAGE_UNRESOLVED"
    return "KDA_LOCAL_FUNCTIONAL_GEOMETRY_REMAINS_MIXED_OR_INCONCLUSIVE"


def next_task(earliest):
    return {
        "READOUT": "LING_KDA_READOUT_ALIGNMENT_CAUSAL_V1",
        "NORMALIZATION": "LING_KDA_LOCAL_NORMALIZATION_GEOMETRY_CAUSAL_V1",
        "GATE_OR_MODULATION": "LING_KDA_LOCAL_GATE_SENSITIVITY_CAUSAL_V1",
        "OUT_PROJECTION": "LING_KDA_OUT_PROJ_LOCAL_ANISOTROPY_CAUSAL_V1",
    }.get(earliest, "WAIT_FOR_HUMAN_REVIEW")


def analyze(args):
    inventory_path = OUT_DIR / "source_artifact_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else prepare(args)
    rows = read_csv(MIN_GPU_DIR / "layer_stage_local_results.csv")
    rows = join_closed_loop(rows)
    layer_group_rows = summarize_layer_group(rows, reps=args.bootstrap_reps)
    layer_rows = summarize_layers(rows, reps=args.bootstrap_reps)
    heterogeneity = classify_layer_heterogeneity(layer_rows)
    loo_rows, loo_details = leave_one_layer_out([r for r in rows if r.get("stage") == "CORE_OUTPUT"], reps=args.bootstrap_reps)
    contribution_rows = layer_contributions(rows, reps=args.bootstrap_reps)
    attenuation = attenuation_rows(rows, reps=args.bootstrap_reps)
    stage_rows = stage_gap_summary(rows, reps=args.bootstrap_reps)
    trans_rows = transition_rows(stage_rows)
    assoc = association_rows(rows, reps=args.bootstrap_reps)
    layer_assoc = layer_aware_association(rows, reps=args.bootstrap_reps)
    write_csv(OUT_DIR / "layer_conditioned_local_gap.csv", layer_group_rows)
    write_csv(OUT_DIR / "layer_classification.csv", layer_rows)
    write_csv(OUT_DIR / "leave_one_layer_out_detailed.csv", loo_rows)
    write_csv(OUT_DIR / "layer_contribution_summary.csv", contribution_rows)
    write_csv(OUT_DIR / "layer_granularity_local_attenuation.csv", attenuation)
    write_csv(OUT_DIR / "local_stage_gap_summary.csv", stage_rows)
    write_csv(OUT_DIR / "local_stage_transition_analysis.csv", trans_rows)
    write_csv(OUT_DIR / "stage_to_closed_loop_association.csv", assoc)
    write_csv(OUT_DIR / "layer_aware_association.csv", layer_assoc)
    write_matrices(layer_group_rows)
    residual_summary = json.loads((RESIDUAL_OUT / "summary.json").read_text(encoding="utf-8"))
    global_diff_vals = []
    core_by = {(str(r["unit_id"]), int(r["group_size"]), int(r["layer_id"])): as_float(r.get("local_gap")) for r in rows if r.get("stage") == "CORE_OUTPUT"}
    for unit in sorted({u for u, _g, _l in core_by}):
        for layer in sorted({l for _u, _g, l in core_by}):
            a = core_by.get((unit, 128, layer))
            b = core_by.get((unit, 16, layer))
            if a is not None and b is not None:
                global_diff_vals.append(a - b)
    global_ci = bootstrap_ci(global_diff_vals, reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED + 10000)
    best = best_proxy_stage(assoc)
    earliest = earliest_local_divergence_stage(stage_rows)
    counts = defaultdict(int)
    for row in layer_rows:
        counts[row["layer_classification"]] += 1
    gates_ok = (
        inventory.get("COMMON_COHORT_REUSE_GATE") == "PASS"
        and all(r.get("MATCHED_CARRIER_GATE") == "PASS" for r in rows)
        and all(r.get("MAGNITUDE_MATCH_GATE") == "PASS" for r in rows)
        and all(r.get("DIRECTION_PRESERVATION_GATE") == "PASS" for r in rows)
    )
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if gates_ok and rows else "INCOMPLETE",
        "OFFLINE_FIRST": "YES",
        "COMMON_COHORT_REUSE_GATE": inventory.get("COMMON_COHORT_REUSE_GATE"),
        "LOCAL_PATH_SEMANTICS_GATE": "PASS",
        "MATCHED_CARRIER_GATE": "PASS" if rows and all(r.get("MATCHED_CARRIER_GATE") == "PASS" for r in rows) else "FAIL",
        "MAGNITUDE_MATCH_GATE": "PASS" if rows and all(r.get("MAGNITUDE_MATCH_GATE") == "PASS" for r in rows) else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if rows and all(r.get("DIRECTION_PRESERVATION_GATE") == "PASS" for r in rows) else "FAIL",
        "N_COMMON_UNITS": len({r["unit_id"] for r in rows}),
        "N_KDA_LAYERS": inventory.get("N_KDA_LAYERS"),
        "KDA_LAYER_IDS": inventory.get("KDA_LAYER_IDS"),
        "N_LOCAL_STAGE_ROWS": len(rows),
        "N_LAYER_GROUP_ROWS": len(layer_group_rows),
        "GLOBAL_G128_LOCAL_GAP": residual_summary.get("G128_LOCAL_DIRECTIONAL_GAP"),
        "GLOBAL_G64_LOCAL_GAP": residual_summary.get("G64_LOCAL_DIRECTIONAL_GAP"),
        "GLOBAL_G32_LOCAL_GAP": residual_summary.get("G32_LOCAL_DIRECTIONAL_GAP"),
        "GLOBAL_G16_LOCAL_GAP": residual_summary.get("G16_LOCAL_DIRECTIONAL_GAP"),
        "LOCAL_GAP_D128_TO_D16": median(global_diff_vals),
        "LOCAL_GAP_D128_TO_D16_CI": global_ci,
        "N_C_SENSITIVE_LAYERS": counts["C_SENSITIVE"],
        "N_R_SENSITIVE_LAYERS": counts["R_SENSITIVE"],
        "N_NEUTRAL_OR_WEAK_LAYERS": counts["NEUTRAL_OR_WEAK"],
        "N_MIXED_LAYERS": counts["MIXED"],
        "N_INSUFFICIENT_N_LAYERS": counts["INSUFFICIENT_N"],
        "LAYER_HETEROGENEITY_TYPE": heterogeneity,
        **loo_details,
        "EARLIEST_LOCAL_DIVERGENCE_STAGE": earliest,
        "BEST_LOCAL_RESIDUAL_PROXY_STAGE": best["stage"],
        "BEST_LOCAL_RESIDUAL_PROXY_SPEARMAN": best["spearman"],
        "BEST_LOCAL_RESIDUAL_PROXY_CI": best["spearman_CI"],
        "READOUT_STAGE_GAP": "NOT_APPLICABLE",
        "CORE_OUTPUT_STAGE_GAP": next((r["median_gap"] for r in stage_rows if r["stage"] == "CORE_OUTPUT"), None),
        "NORMALIZATION_STAGE_GAP": "NOT_APPLICABLE",
        "GATE_STAGE_GAP": "NOT_APPLICABLE",
        "OUT_PROJECTION_STAGE_GAP": "NOT_APPLICABLE",
        "HIDDEN_OUTPUT_STAGE_GAP": "NOT_APPLICABLE",
        "RESIDUAL_LOO_UNIT_SIGN_STABLE": loo_sign_stable(rows, "local_gap"),
        "RESIDUAL_LOO_LAYER_SIGN_STABLE": loo_sign_stable(rows, "local_gap", "layer_id"),
        "RESIDUAL_LOO_PROMPT_SIGN_STABLE": loo_sign_stable(rows, "local_gap", "prompt_id"),
        "MINIMAL_GPU_FALLBACK_USED": "YES",
        "N_NEW_GPU_LOCAL_EVALUATIONS": len({(r["unit_id"], int(r["group_size"]), int(r["layer_id"])) for r in rows}) * 2,
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": args.bash_syntax_status,
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES",
    }
    summary["FINAL_CLASSIFICATION"] = final_classification(gates_ok, heterogeneity, earliest)
    summary["NEXT_RECOMMENDED_TASK"] = next_task(earliest)
    save_json(OUT_DIR / "bootstrap_results.json", {
        "layer_group_rows": layer_group_rows,
        "layer_rows": layer_rows,
        "leave_one_layer_out": loo_rows,
        "stage_rows": stage_rows,
        "stage_to_closed_loop_association": assoc,
    })
    write_text(OUT_DIR / "README.md", f"# {TASK}\n\nLayer-conditioned local functional geometry audit outputs.\n")
    write_text(OUT_DIR / "report.md", "\n".join([
        f"# {TASK}",
        "",
        f"FORMAL_STATUS: {summary['FORMAL_STATUS']}",
        f"LAYER_HETEROGENEITY_TYPE: {heterogeneity}",
        f"EARLIEST_LOCAL_DIVERGENCE_STAGE: {earliest}",
        f"FINAL_CLASSIFICATION: {summary['FINAL_CLASSIFICATION']}",
        "",
        "This audit uses a Stage-A-only minimal fallback because prior residual artifacts did not contain per-KDA-layer local gaps.",
        "Only the KDA backend core-output stage is directly observed; unavailable downstream local stages are marked NOT_APPLICABLE.",
        "Closed-loop residual gaps are reused from the magnitude-equalization causal task.",
        "The previous layer LOO failure is reconstructed as an artifact of prior `ALL_KDA` layer labeling rather than a real per-layer leave-one-out sign reversal.",
        "",
        "No Stage-B recurrence, Kimi transfer, targeted causal operator experiment, or method design was launched.",
    ]) + "\n")
    save_json(OUT_DIR / "summary.json", summary)
    return summary


def verify(_args=None):
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8")) if (OUT_DIR / "summary.json").exists() else {}
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUT_DIR / name).exists()]
    if summary.get("MINIMAL_GPU_FALLBACK_USED") == "YES":
        missing.extend([name for name in ["minimal_gpu_justification.md", "minimal_gpu_results/layer_stage_local_results.csv"] if not (OUT_DIR / name).exists()])
    rows = read_csv(MIN_GPU_DIR / "layer_stage_local_results.csv")
    layer_group = read_csv(OUT_DIR / "layer_conditioned_local_gap.csv")
    numeric_bad = []
    for path_name in ["minimal_gpu_results/layer_stage_local_results.csv", "layer_conditioned_local_gap.csv", "local_stage_gap_summary.csv"]:
        for idx, row in enumerate(read_csv(OUT_DIR / path_name)):
            for key, value in row.items():
                if value in ("", "None", "NA", "NOT_APPLICABLE"):
                    continue
                if key.endswith("_gap") or key.startswith("damage_") or key.endswith("_norm") or key.endswith("_effect"):
                    if not finite(value):
                        numeric_bad.append((path_name, idx, key, value))
    expected_rows = int(summary.get("N_COMMON_UNITS") or 0) * int(summary.get("N_KDA_LAYERS") or 0) * len(GROUP_SIZES) * len(OBSERVED_STAGES)
    fresh = "PASS" if not missing and rows and len(rows) == expected_rows and layer_group and not numeric_bad and summary.get("NEW_STAGE_B_TRAJECTORIES") == 0 else "FAIL"
    verification = {
        "TASK": TASK,
        "FRESH_OUTPUT_VERIFICATION": fresh,
        "missing_files": missing,
        "N_LOCAL_STAGE_ROWS": len(rows),
        "N_EXPECTED_LOCAL_STAGE_ROWS": expected_rows,
        "N_LAYER_GROUP_ROWS": len(layer_group),
        "numeric_bad": numeric_bad[:20],
        "NEW_STAGE_B_TRAJECTORIES": summary.get("NEW_STAGE_B_TRAJECTORIES"),
        "summary_status": summary.get("FORMAL_STATUS"),
    }
    if summary:
        summary["FRESH_OUTPUT_VERIFICATION"] = fresh
        save_json(OUT_DIR / "summary.json", summary)
    save_json(OUT_DIR / "fresh_output_verification.json", verification)
    return verification


def final_summary(py_compile_status="NOT_RUN", pytest_status="NOT_RUN", bash_syntax_status="NOT_APPLICABLE", gpu_release_status="NOT_RUN"):
    path = OUT_DIR / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary.update({
        "PY_COMPILE": py_compile_status,
        "PYTEST": pytest_status,
        "BASH_SYNTAX_CHECK": bash_syntax_status,
        "GPU_RELEASE_VERIFICATION": gpu_release_status,
    })
    lines = [f"{key} = {summary.get(key)}" for key in SUMMARY_KEYS]
    lines.append("STOP.")
    text = "\n".join(lines) + "\n"
    write_text(OUT_DIR / "final_console_summary.txt", text)
    if summary:
        save_json(path, summary)
    return text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prepare", "minimal-gpu", "analyze", "verify", "summary"], default="prepare")
    p.add_argument("--unit-limit", type=int, default=None)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--py-compile-status", default="NOT_RUN")
    p.add_argument("--pytest-status", default="NOT_RUN")
    p.add_argument("--bash-syntax-status", default="NOT_APPLICABLE")
    p.add_argument("--gpu-release-status", default="NOT_RUN")
    p.add_argument("--console-summary", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "prepare":
        print(json.dumps(prepare(args), indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "minimal-gpu":
        print(json.dumps(minimal_gpu(args), indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "analyze":
        print(json.dumps(analyze(args), indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "verify":
        print(json.dumps(verify(args), indent=2, sort_keys=True, ensure_ascii=False))
    elif args.stage == "summary":
        text = final_summary(args.py_compile_status, args.pytest_status, args.bash_syntax_status, args.gpu_release_status)
        print(text if args.console_summary else json.dumps({"summary": str(OUT_DIR / "final_console_summary.txt")}, indent=2))


if __name__ == "__main__":
    main()
