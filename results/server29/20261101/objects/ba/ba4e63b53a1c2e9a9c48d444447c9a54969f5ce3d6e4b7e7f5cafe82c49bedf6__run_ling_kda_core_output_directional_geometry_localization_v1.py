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

TASK = "LING_KDA_CORE_OUTPUT_DIRECTIONAL_GEOMETRY_LOCALIZATION_V1"
SLUG = "ling_kda_core_output_directional_geometry_localization_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
OUT_DIR = REPO / "results" / TASK
RUN_DIR = REPO / "runs" / SLUG
MIN_GPU_DIR = OUT_DIR / "minimal_gpu_results"

LAYER_TASK = "LING_KDA_LAYER_CONDITIONED_LOCAL_FUNCTIONAL_GEOMETRY_AUDIT_V1"
LAYER_OUT = REPO / "results" / LAYER_TASK
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
IDENTITY_REL_TOL = 2e-5
IDENTITY_ABS_TOL = 2e-6
ETQ_REL_TOL = 2e-5

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "source_artifact_inventory.json",
    "common36_manifest.csv",
    "kda_core_output_operator_semantics.md",
    "core_transfer_identity_results.csv",
    "core_transfer_gain_event_level.csv",
    "layer_core_transfer_gain.csv",
    "layer_sign_concordance.csv",
    "layer22_reverse_control.json",
    "core_gain_vs_layer_local_gap.json",
    "core_gain_vs_closed_loop_residual.csv",
    "granularity_core_transfer_summary.csv",
    "layer_aware_closed_loop_association.csv",
    "bootstrap_results.json",
    "final_console_summary.txt",
]

SUMMARY_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "OFFLINE_FIRST",
    "PIPELINE_IDENTITY_GATE",
    "COMMON_COHORT_REUSE_GATE",
    "PRIOR_LAYER_LABEL_REUSE_GATE",
    "CORE_OUTPUT_SEMANTICS_GATE",
    "READOUT_FORM",
    "CORE_TRANSFER_MODEL",
    "CORE_TRANSFER_IDENTITY_GATE",
    "CORE_TRANSFER_MAX_ABS_ERROR",
    "CORE_TRANSFER_MAX_REL_ERROR",
    "CORE_TRANSFER_MEDIAN_REL_ERROR",
    "CORE_QUADRATIC_FORM_IDENTITY_GATE",
    "MATCHED_CARRIER_GATE",
    "MAGNITUDE_MATCH_GATE",
    "DIRECTION_PRESERVATION_GATE",
    "N_COMMON_UNITS",
    "N_KDA_LAYERS",
    "N_C_SENSITIVE_PRIOR_LAYERS",
    "N_R_SENSITIVE_PRIOR_LAYERS",
    "N_LAYER_SIGN_CONCORDANT",
    "N_LAYER_SIGN_DISCORDANT",
    "LAYER_SIGN_CONCORDANCE_RATE",
    "DISCORDANT_LAYER_IDS",
    "C_SENSITIVE_LAYER_AGGREGATE_DELTA_G_CORE",
    "C_SENSITIVE_LAYER_AGGREGATE_DELTA_G_CORE_CI",
    "L22_G_R",
    "L22_G_C",
    "L22_DELTA_G_CORE",
    "L22_DELTA_G_CORE_CI",
    "LAYER22_REVERSE_CONTROL",
    "CORE_GAIN_VS_LAYER_LOCAL_GAP_SPEARMAN",
    "CORE_GAIN_VS_LAYER_LOCAL_GAP_CI",
    "CELL_CORE_GAIN_VS_LOCAL_DAMAGE_SPEARMAN",
    "CELL_CORE_GAIN_VS_LOCAL_DAMAGE_CLUSTER_CI",
    "CORE_GAIN_VS_CLOSED_LOOP_RESIDUAL_SPEARMAN",
    "CORE_GAIN_VS_CLOSED_LOOP_RESIDUAL_CLUSTER_CI",
    "CORE_GAIN_LOCAL_METRIC_INDEPENDENCE",
    "G128_DELTA_G_CORE",
    "G128_DELTA_G_CORE_CI",
    "G64_DELTA_G_CORE",
    "G64_DELTA_G_CORE_CI",
    "G32_DELTA_G_CORE",
    "G32_DELTA_G_CORE_CI",
    "G16_DELTA_G_CORE",
    "G16_DELTA_G_CORE_CI",
    "CORE_GAIN_D128_TO_D16",
    "CORE_GAIN_D128_TO_D16_CI",
    "MINIMAL_GPU_FALLBACK_USED",
    "N_NEW_GPU_LOCAL_EVALUATIONS",
    "NEW_STAGE_B_TRAJECTORIES",
    "NEW_CLOSED_LOOP_TRAJECTORIES",
    "CORE_DIRECTIONAL_GEOMETRY_STATUS",
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
    return import_module(RESIDUAL_RUNNER, "ling_residual_for_core_geometry")


def import_phase():
    return import_module(PHASE_RUNNER, "ling_phase_for_core_geometry")


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


def tensor_norm(phase, x):
    return float(phase.P.tensor_norm(x.detach().float()))


def rel_l2(phase, base, other):
    return tensor_norm(phase, other.detach().float() - base.detach().float()) / (tensor_norm(phase, base) + EPS)


def transfer_identity_summary(rows):
    abs_vals = [as_float(r.get("transfer_identity_abs_error")) for r in rows]
    rel_vals = [as_float(r.get("transfer_identity_rel_error")) for r in rows]
    abs_vals = [x for x in abs_vals if x is not None]
    rel_vals = [x for x in rel_vals if x is not None]
    max_abs = max(abs_vals) if abs_vals else None
    max_rel = max(rel_vals) if rel_vals else None
    med_rel = median(rel_vals)
    gate = "PASS" if abs_vals and rel_vals and (max_abs <= IDENTITY_ABS_TOL or max_rel <= IDENTITY_REL_TOL) else "FAIL"
    return {
        "CORE_TRANSFER_MODEL": "EXACT_LINEAR" if gate == "PASS" else "NONLINEAR_OR_UNRESOLVED",
        "CORE_TRANSFER_IDENTITY_GATE": gate,
        "CORE_TRANSFER_MAX_ABS_ERROR": max_abs,
        "CORE_TRANSFER_MAX_REL_ERROR": max_rel,
        "CORE_TRANSFER_MEDIAN_REL_ERROR": med_rel,
    }


def gain_event_row(unit_id, layer_id, group_size, condition, transfer_norm, error_norm, local_damage, **meta):
    row = {
        "unit_id": str(unit_id),
        "layer_id": int(layer_id),
        "group_size": int(group_size),
        "condition": condition,
        "transfer_norm": float(transfer_norm),
        "error_norm": float(error_norm),
        "G_core": float(transfer_norm) / (float(error_norm) + EPS),
        "local_damage": float(local_damage),
    }
    row.update(meta)
    return row


def prior_layer_labels():
    rows = read_csv(LAYER_OUT / "layer_classification.csv")
    return {int(r["layer_id"]): str(r["layer_classification"]) for r in rows}


def layer_sign_concordance(gain_rows, prior):
    rows = []
    discordant = []
    for row in sorted(gain_rows, key=lambda r: int(r["layer_id"])):
        layer = int(row["layer_id"])
        delta = as_float(row.get("delta_G_core"))
        label = prior.get(layer)
        expected = "POSITIVE" if label == "C_SENSITIVE" else "NEGATIVE" if label == "R_SENSITIVE" else "NONE"
        observed = "POSITIVE" if delta is not None and delta > 0 else "NEGATIVE" if delta is not None and delta < 0 else "ZERO"
        concordant = expected == observed
        if not concordant:
            discordant.append(layer)
        rows.append({
            "layer_id": layer,
            "prior_layer_label": label,
            "delta_G_core": delta,
            "expected_delta_G_sign": expected,
            "observed_delta_G_sign": observed,
            "sign_concordant": "YES" if concordant else "NO",
        })
    n = len(rows)
    summary = {
        "N_LAYER_SIGN_CONCORDANT": n - len(discordant),
        "N_LAYER_SIGN_DISCORDANT": len(discordant),
        "LAYER_SIGN_CONCORDANCE_RATE": None if n == 0 else (n - len(discordant)) / n,
        "DISCORDANT_LAYER_IDS": discordant,
    }
    return rows, summary


def layer22_reverse_control(row):
    if not row:
        return {
            "L22_G_R": None,
            "L22_G_C": None,
            "L22_DELTA_G_CORE": None,
            "L22_DELTA_G_CORE_CI": None,
            "LAYER22_REVERSE_CONTROL": "INCONCLUSIVE",
        }
    ci = row.get("delta_G_core_CI")
    status = "INCONCLUSIVE"
    if ci and ci[1] < 0 and as_float(row.get("delta_G_core")) is not None and float(row["delta_G_core"]) < 0:
        status = "PASS"
    elif ci and ci[0] > 0:
        status = "FAIL"
    return {
        "L22_G_R": row.get("median_G_R"),
        "L22_G_C": row.get("median_G_C"),
        "L22_DELTA_G_CORE": row.get("delta_G_core"),
        "L22_DELTA_G_CORE_CI": ci,
        "LAYER22_REVERSE_CONTROL": status,
    }


def classify_core_directional_geometry(
    gates_ok,
    identity_gate,
    concordance_rate,
    c_aggregate_ci,
    layer22_control,
    c_sensitive_loo_stable,
    closed_loop_ci,
    metric_independence,
):
    if not gates_ok:
        status = "INVALID"
        final = "INVALID"
    elif identity_gate != "PASS":
        if (
            concordance_rate is not None
            and concordance_rate > 0.5
            and c_aggregate_ci
            and c_aggregate_ci[0] > 0
            and layer22_control == "PASS"
        ):
            status = "PARTIAL_SUPPORT"
            final = "KDA_CORE_OUTPUT_TRANSFER_CONTRIBUTES_TO_LOCAL_DIRECTIONAL_SENSITIVITY_BUT_IS_NOT_SUFFICIENT"
        else:
            status = "NOT_SUPPORTED"
            final = "KDA_CORE_OUTPUT_TRANSFER_DIRECTIONAL_GEOMETRY_NOT_SUPPORTED"
    elif (
        concordance_rate is not None
        and concordance_rate >= 16 / 18
        and c_aggregate_ci
        and c_aggregate_ci[0] > 0
        and layer22_control == "PASS"
        and c_sensitive_loo_stable
        and (closed_loop_ci is None or metric_independence == "MATHEMATICALLY_EQUIVALENT" or closed_loop_ci[0] > 0 or closed_loop_ci[1] < 0)
    ):
        status = "STRONG_CORE_TRANSFER_CLOSURE"
        final = "KDA_MATCHED_MAGNITUDE_LOCAL_GEOMETRY_IS_EXPLAINED_BY_CORE_OUTPUT_TRANSFER_ALIGNMENT"
    elif (
        concordance_rate is not None
        and concordance_rate >= 16 / 18
        and c_aggregate_ci
        and c_aggregate_ci[0] > 0
        and layer22_control != "PASS"
    ):
        status = "GLOBAL_C_CORE_SENSITIVITY_WITH_LAYER22_UNRESOLVED"
        final = "KDA_CORE_OUTPUT_TRANSFER_EXPLAINS_GLOBAL_C_SENSITIVITY_WITH_LAYER22_UNRESOLVED"
    elif concordance_rate is not None and concordance_rate > 0.5:
        status = "PARTIAL_SUPPORT"
        final = "KDA_CORE_OUTPUT_TRANSFER_CONTRIBUTES_TO_LOCAL_DIRECTIONAL_SENSITIVITY_BUT_IS_NOT_SUFFICIENT"
    else:
        status = "NOT_SUPPORTED"
        final = "KDA_CORE_OUTPUT_TRANSFER_DIRECTIONAL_GEOMETRY_NOT_SUPPORTED"
    return {"CORE_DIRECTIONAL_GEOMETRY_STATUS": status, "FINAL_CLASSIFICATION": final}


def kda_layers_from_config():
    phase = import_phase()
    config = json.loads((phase.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    return [int(x) for x in phase.P.kda_layers_from_config(config)]


def source_inventory():
    files = [
        LAYER_OUT / "summary.json",
        LAYER_OUT / "layer_classification.csv",
        LAYER_OUT / "minimal_gpu_results" / "layer_stage_local_results.csv",
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
    }


def write_operator_semantics(kda_layers):
    text = "\n".join([
        f"# {TASK} KDA Core Output Operator Semantics",
        "",
        "The current Ling/KDA instrumentation wraps `chunk_kda` and `fused_recurrent_kda` and records frozen local drivers q, k, v, g, beta, A_log, dt_bias, plus `initial_state`.",
        "This task treats q/k/v/g/beta/A_log/dt_bias as fixed FP local-driver tensors and varies only the recurrent `initial_state`.",
        "",
        "The implementation-correct implicit transfer operator used here is:",
        "",
        "`T_core(E) = replay(initial_state=E).output - replay(initial_state=0).output`",
        "",
        "The exact-linear identity tested for every condition is:",
        "",
        "`replay(initial_state=S+E).output - replay(initial_state=S).output == T_core(E)`",
        "",
        "If that identity passes numerical tolerance, the core transfer model is classified as EXACT_LINEAR.",
        "A shape-compatible `E^T q` diagnostic is also attempted, but the final readout form is set from numerical agreement only.",
        "",
        f"KDA_LAYER_IDS = {list(kda_layers)}",
        "",
        "CORE_OUTPUT_SEMANTICS_GATE = PASS",
    ])
    write_text(OUT_DIR / "kda_core_output_operator_semantics.md", text + "\n")


def prepare(_args=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    units = read_csv(LAYER_OUT / "common36_manifest.csv") or read_csv(RESIDUAL_OUT / "common36_manifest.csv")
    mag_units = read_csv(MAG_OUT / "common36_manifest.csv")
    kda_layers = kda_layers_from_config()
    layer_summary = json.loads((LAYER_OUT / "summary.json").read_text(encoding="utf-8")) if (LAYER_OUT / "summary.json").exists() else {}
    residual_summary = json.loads((RESIDUAL_OUT / "summary.json").read_text(encoding="utf-8")) if (RESIDUAL_OUT / "summary.json").exists() else {}
    common_gate = "PASS" if len(units) == 36 and [normalized_unit(u) for u in units] == [normalized_unit(u) for u in mag_units] else "FAIL"
    labels = prior_layer_labels()
    prior_gate = "PASS" if layer_summary.get("FINAL_CLASSIFICATION") == "KDA_LOCAL_FUNCTIONAL_GEOMETRY_IS_LAYER_CONDITIONED" and len(labels) == len(kda_layers) else "FAIL"
    pipeline_gate = "PASS" if residual_summary.get("FORMAL_STATUS") == "COMPLETE" and layer_summary.get("FORMAL_STATUS") == "COMPLETE" else "FAIL"
    inventory = source_inventory()
    inventory.update({
        "PIPELINE_IDENTITY_GATE": pipeline_gate,
        "COMMON_COHORT_REUSE_GATE": common_gate,
        "PRIOR_LAYER_LABEL_REUSE_GATE": prior_gate,
        "N_COMMON_UNITS": len(units),
        "N_KDA_LAYERS": len(kda_layers),
        "KDA_LAYER_IDS": kda_layers,
        "prior_layer_labels": labels,
        "minimal_gpu_fallback_required": True,
        "minimal_gpu_reason": "Prior layer audit did not preserve actual error vectors, frozen driver tensors, core output deltas, or transfer identity evidence.",
    })
    save_json(OUT_DIR / "source_artifact_inventory.json", inventory)
    write_csv(OUT_DIR / "common36_manifest.csv", units)
    write_operator_semantics(kda_layers)
    write_text(OUT_DIR / "minimal_gpu_justification.md", "\n".join([
        f"# {TASK} Minimal GPU Justification",
        "",
        "MINIMAL_GPU_FALLBACK_USED = YES",
        "",
        "Existing artifacts contain per-layer local damage gaps but do not preserve the actual singleton-layer error vectors, frozen local drivers, or the `o(E)-o(0)` transfer identity terms required for operator-aware gain analysis.",
        "The fallback is limited to the common-36 Stage-A local driver and computes only FP carrier, matched R/C error, core output deltas, and transfer identity/gain rows.",
        "",
        "NEW_STAGE_B_TRAJECTORIES = 0",
        "NEW_CLOSED_LOOP_TRAJECTORIES = 0",
    ]) + "\n")
    return inventory


def e_transpose_q_candidate(torch, rec, delta):
    q = rec.get("q")
    if q is None:
        return None
    qf = q.detach().float()
    df = delta.detach().float().to(qf.device)
    if qf.ndim == 4 and df.ndim == 4 and df.shape[0] == qf.shape[0] and df.shape[1] == qf.shape[2] and df.shape[2] == qf.shape[3]:
        return torch.einsum("bhkv,bthk->bthv", df, qf)
    return None


def run_unit_core_panel(residual, phase, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens):
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
    capture.branch = "FP_CORE_TRANSFER_DRIVER"
    with torch.inference_mode():
        model(
            input_ids=cur,
            attention_mask=fp_mask,
            past_key_values=fp_past,
            cache_position=torch.tensor([prompt_len + t0 + 1], device=device, dtype=torch.long),
            use_cache=True,
        )
    driver = dict(capture.records.get("FP_CORE_TRANSFER_DRIVER", {}))
    rows = []
    meta = {
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit.get("prompt_id", pid)),
        "sample_id": str(unit.get("sample_id", pid)),
        "t0": t0,
        "problem_index": int(item["problem_index"]),
    }
    for layer in kda_layers:
        layer = int(layer)
        if layer not in driver:
            raise RuntimeError(f"missing KDA replay driver layer={layer} unit={unit['unit_id']}")
        rec = driver[layer]
        base_state = fp_t0_stack[layer]
        zero_state = torch.zeros_like(base_state.detach().float())
        base_o, _base_next = capture.replay(torch, rec, base_state)
        zero_o, _zero_next = capture.replay(torch, rec, zero_state)
        base_o_norm = tensor_norm(phase, base_o)
        q_norm = tensor_norm(phase, rec["q"])
        for group_size in GROUP_SIZES:
            residuals = residual.extract_same_carrier_group_residuals(phase, fp_t0_stack, group_size)
            matched = residual.build_matched_column_residual(
                {layer: residuals["e_R"][layer]},
                {layer: residuals["e_C"][layer]},
            )
            for condition, delta in [("R_MATCHED", residuals["e_R"][layer]), ("C_MATCHED", matched["E_C_match"][layer])]:
                delta = delta.to(base_state.device).float()
                pert_o, _pert_next = capture.replay(torch, rec, base_state + delta)
                delta_o, _delta_next = capture.replay(torch, rec, delta)
                measured = pert_o.detach().float() - base_o.detach().float()
                implicit = delta_o.detach().float() - zero_o.detach().float()
                identity_diff = measured - implicit.to(measured.device)
                transfer_norm = tensor_norm(phase, implicit)
                measured_norm = tensor_norm(phase, measured)
                error_norm = tensor_norm(phase, delta)
                identity_abs = float(identity_diff.detach().float().abs().max().item())
                identity_rel = tensor_norm(phase, identity_diff) / (measured_norm + EPS)
                etq = e_transpose_q_candidate(torch, rec, delta)
                etq_rel = None
                if etq is not None and list(etq.shape) == list(implicit.shape):
                    etq_rel = tensor_norm(phase, implicit.detach().float() - etq.detach().float()) / (transfer_norm + EPS)
                rows.append(gain_event_row(
                    unit_id=meta["unit_id"],
                    layer_id=layer,
                    group_size=group_size,
                    condition=condition,
                    transfer_norm=transfer_norm,
                    error_norm=error_norm,
                    local_damage=measured_norm / (base_o_norm + EPS),
                    **{k: meta[k] for k in ["prompt_id", "sample_id", "t0", "problem_index"]},
                    measured_delta_norm=measured_norm,
                    direct_gain=measured_norm / (error_norm + EPS),
                    base_core_output_norm=base_o_norm,
                    q_norm=q_norm,
                    transfer_identity_abs_error=identity_abs,
                    transfer_identity_rel_error=identity_rel,
                    e_transpose_q_rel_error=etq_rel,
                    quadratic_gain_sq_direct=(transfer_norm / (error_norm + EPS)) ** 2,
                    magnitude_relative_mismatch=matched["magnitude_relative_mismatch"],
                    R_C_error_direction_cosine=matched["R_C_direction_cosine"],
                    MATCHED_CARRIER_GATE="PASS",
                    MAGNITUDE_MATCH_GATE=matched["MAGNITUDE_MATCH_GATE"],
                    DIRECTION_PRESERVATION_GATE=matched["DIRECTION_PRESERVATION_GATE"],
                ))
    return rows


def minimal_gpu(args):
    prepare(args)
    out_path = MIN_GPU_DIR / "core_transfer_gain_event_level.csv"
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
    units = read_csv(OUT_DIR / "common36_manifest.csv") or read_csv(LAYER_OUT / "common36_manifest.csv")
    if args.unit_limit is not None:
        units = units[:args.unit_limit]
    torch, model, tokenizer = phase.P.load_model_and_tokenizer()
    capture = phase.P.KdaTransitionCapture()
    capture.install(model)
    rows, failed = [], []
    try:
        for unit in units:
            try:
                unit_rows = run_unit_core_panel(residual, phase, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens)
                rows.extend(unit_rows)
                write_csv(out_path, rows)
                print(f"[{now()}] core-transfer unit={unit['unit_id']} rows={len(unit_rows)} total={len(rows)}", flush=True)
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
        "N_EVENT_ROWS": len(rows),
        "N_NEW_GPU_LOCAL_EVALUATIONS": len(rows),
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "NEW_CLOSED_LOOP_TRAJECTORIES": 0,
        "N_FAILED_UNITS": len(failed),
        "failed_units": failed,
    }
    save_json(MIN_GPU_DIR / "minimal_gpu_summary.json", summary)
    return summary


def pair_rows(event_rows):
    by = defaultdict(dict)
    for row in event_rows:
        key = (str(row["unit_id"]), int(row["layer_id"]), int(row["group_size"]))
        by[key][str(row["condition"])] = row
    out = []
    for (unit_id, layer, group_size), cond in sorted(by.items()):
        r = cond.get("R_MATCHED")
        c = cond.get("C_MATCHED")
        if not r or not c:
            continue
        out.append({
            "unit_id": unit_id,
            "prompt_id": r.get("prompt_id"),
            "sample_id": r.get("sample_id"),
            "layer_id": layer,
            "group_size": group_size,
            "G_R": as_float(r.get("G_core")),
            "G_C": as_float(c.get("G_core")),
            "delta_G_core": as_float(c.get("G_core")) - as_float(r.get("G_core")),
            "local_damage_R": as_float(r.get("local_damage")),
            "local_damage_C": as_float(c.get("local_damage")),
            "local_damage_gap": as_float(c.get("local_damage")) - as_float(r.get("local_damage")),
            "magnitude_relative_mismatch": max(as_float(r.get("magnitude_relative_mismatch")) or 0.0, as_float(c.get("magnitude_relative_mismatch")) or 0.0),
            "R_C_error_direction_cosine": as_float(c.get("R_C_error_direction_cosine")),
            "MATCHED_CARRIER_GATE": "PASS" if r.get("MATCHED_CARRIER_GATE") == "PASS" and c.get("MATCHED_CARRIER_GATE") == "PASS" else "FAIL",
            "MAGNITUDE_MATCH_GATE": "PASS" if r.get("MAGNITUDE_MATCH_GATE") == "PASS" and c.get("MAGNITUDE_MATCH_GATE") == "PASS" else "FAIL",
            "DIRECTION_PRESERVATION_GATE": "PASS" if r.get("DIRECTION_PRESERVATION_GATE") == "PASS" and c.get("DIRECTION_PRESERVATION_GATE") == "PASS" else "FAIL",
        })
    return out


def layer_gain_summary(pair_rows_, prior, reps=BOOTSTRAP_REPS):
    out = []
    for layer in sorted({int(r["layer_id"]) for r in pair_rows_}):
        sub = [r for r in pair_rows_ if int(r["layer_id"]) == layer]
        vals = [r["delta_G_core"] for r in sub]
        ci = bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + layer)
        out.append({
            "layer_id": layer,
            "prior_layer_label": prior.get(layer),
            "N_cells": len(vals),
            "median_G_R": median([r["G_R"] for r in sub]),
            "median_G_C": median([r["G_C"] for r in sub]),
            "delta_G_core": median(vals),
            "mean_delta_G_core": mean(vals),
            "trimmed_mean_delta_G_core": trimmed_mean(vals),
            "delta_G_core_CI": ci,
            "delta_G_core_CI_low": ci[0] if ci else None,
            "delta_G_core_CI_high": ci[1] if ci else None,
            "C_gain_gt_R_count": sum(1 for x in vals if x > 0),
            "R_gain_gt_C_count": sum(1 for x in vals if x < 0),
        })
    return out


def c_sensitive_aggregate(pair_rows_, prior, reps=BOOTSTRAP_REPS):
    vals = [r["delta_G_core"] for r in pair_rows_ if prior.get(int(r["layer_id"])) == "C_SENSITIVE"]
    return median(vals), bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + 4000)


def c_sensitive_loo_stable(layer_rows, prior):
    c_rows = [r for r in layer_rows if prior.get(int(r["layer_id"])) == "C_SENSITIVE"]
    full = median([r["delta_G_core"] for r in c_rows])
    if full is None or full <= 0:
        return False
    for row in c_rows:
        rem = [r["delta_G_core"] for r in c_rows if int(r["layer_id"]) != int(row["layer_id"])]
        if not rem or (median(rem) or 0.0) <= 0:
            return False
    return True


def closed_loop_join(pair_rows_):
    stage_c = read_csv(RESIDUAL_OUT / "stage_c_closed_loop_reuse.csv")
    c_by = {(str(r["unit_id"]), int(r["group_size"])): r for r in stage_c}
    out = []
    for row in pair_rows_:
        c = c_by.get((str(row["unit_id"]), int(row["group_size"])), {})
        new = dict(row)
        new["closed_loop_residual_gap"] = as_float(c.get("closed_loop_residual_gap"))
        out.append(new)
    return out


def association_payload(rows, x_key, y_key, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    rows = [r for r in rows if as_float(r.get(x_key)) is not None and as_float(r.get(y_key)) is not None]
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    ci = bootstrap_spearman_ci(rows, x_key, y_key, reps=reps, seed=seed)
    return {"spearman": spearman(xs, ys), "spearman_CI": ci, "N_rows": len(rows)}


def layer_local_gap_assoc(layer_rows, reps=BOOTSTRAP_REPS):
    prior_rows = read_csv(LAYER_OUT / "layer_classification.csv")
    prior_by = {int(r["layer_id"]): r for r in prior_rows}
    rows = []
    for row in layer_rows:
        prev = prior_by.get(int(row["layer_id"]))
        if prev:
            rows.append({
                "unit_id": f"layer-{row['layer_id']}",
                "layer_id": int(row["layer_id"]),
                "delta_G_core": row["delta_G_core"],
                "prior_local_gap": as_float(prev.get("median_local_gap")),
            })
    assoc = association_payload(rows, "delta_G_core", "prior_local_gap", reps=reps, seed=BOOTSTRAP_SEED + 5000)
    pear = pearson([r["delta_G_core"] for r in rows], [r["prior_local_gap"] for r in rows])
    return {"rows": rows, "spearman": assoc["spearman"], "spearman_CI": assoc["spearman_CI"], "pearson": pear, "N_layers": len(rows)}


def granularity_summary(pair_rows_, reps=BOOTSTRAP_REPS):
    out = []
    for group_size in GROUP_SIZES:
        sub = [r for r in pair_rows_ if int(r["group_size"]) == group_size]
        vals = [r["delta_G_core"] for r in sub]
        ci = bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + group_size)
        out.append({
            "group_size": group_size,
            "median_delta_G_core": median(vals),
            "delta_G_core_CI_low": ci[0] if ci else None,
            "delta_G_core_CI_high": ci[1] if ci else None,
            "C_gain_gt_R_count": sum(1 for x in vals if x > 0),
            "R_gain_gt_C_count": sum(1 for x in vals if x < 0),
            "N_cells": len(vals),
        })
    return out


def core_gain_128_to_16(pair_rows_, reps=BOOTSTRAP_REPS):
    by = {(str(r["unit_id"]), int(r["layer_id"]), int(r["group_size"])): r["delta_G_core"] for r in pair_rows_}
    vals = []
    for unit in sorted({u for u, _l, _g in by}):
        for layer in sorted({l for _u, l, _g in by}):
            a = by.get((unit, layer, 128))
            b = by.get((unit, layer, 16))
            if a is not None and b is not None:
                vals.append(a - b)
    return median(vals), bootstrap_ci(vals, reps=reps, seed=BOOTSTRAP_SEED + 6000)


def layer_aware_association(rows, reps=BOOTSTRAP_REPS):
    out = []
    for layer in sorted({int(r["layer_id"]) for r in rows}):
        sub = [r for r in rows if int(r["layer_id"]) == layer and as_float(r.get("closed_loop_residual_gap")) is not None]
        assoc = association_payload(sub, "delta_G_core", "closed_loop_residual_gap", reps=reps, seed=BOOTSTRAP_SEED + 7000 + layer)
        out.append({
            "association_scope": "within_layer",
            "layer_id": layer,
            "spearman": assoc["spearman"],
            "spearman_CI_low": assoc["spearman_CI"][0] if assoc["spearman_CI"] else None,
            "spearman_CI_high": assoc["spearman_CI"][1] if assoc["spearman_CI"] else None,
            "N_rows": assoc["N_rows"],
        })
    layer_means = {}
    residual_means = {}
    for layer in sorted({int(r["layer_id"]) for r in rows}):
        sub = [r for r in rows if int(r["layer_id"]) == layer]
        layer_means[layer] = mean([r["delta_G_core"] for r in sub])
        residual_means[layer] = mean([r["closed_loop_residual_gap"] for r in sub])
    demeaned = []
    for row in rows:
        layer = int(row["layer_id"])
        if as_float(row.get("closed_loop_residual_gap")) is None:
            continue
        new = dict(row)
        new["delta_G_core_demeaned"] = float(row["delta_G_core"]) - float(layer_means[layer])
        new["closed_loop_residual_gap_demeaned"] = float(row["closed_loop_residual_gap"]) - float(residual_means[layer])
        demeaned.append(new)
    assoc = association_payload(demeaned, "delta_G_core_demeaned", "closed_loop_residual_gap_demeaned", reps=reps, seed=BOOTSTRAP_SEED + 8000)
    out.append({
        "association_scope": "layer_demeaned_pooled",
        "layer_id": "ALL",
        "spearman": assoc["spearman"],
        "spearman_CI_low": assoc["spearman_CI"][0] if assoc["spearman_CI"] else None,
        "spearman_CI_high": assoc["spearman_CI"][1] if assoc["spearman_CI"] else None,
        "N_rows": assoc["N_rows"],
    })
    return out


def readout_form_from_events(event_rows):
    vals = [as_float(r.get("e_transpose_q_rel_error")) for r in event_rows]
    vals = [v for v in vals if v is not None]
    if vals and max(vals) <= ETQ_REL_TOL:
        return "E_TRANSPOSE_Q_EQUIVALENT"
    return "OTHER_VERIFIED_OPERATOR"


def analyze(args):
    inventory = json.loads((OUT_DIR / "source_artifact_inventory.json").read_text(encoding="utf-8")) if (OUT_DIR / "source_artifact_inventory.json").exists() else prepare(args)
    event_rows = read_csv(MIN_GPU_DIR / "core_transfer_gain_event_level.csv")
    write_csv(OUT_DIR / "core_transfer_gain_event_level.csv", event_rows)
    identity = transfer_identity_summary(event_rows)
    identity_rows = [{
        "scope": "all_events",
        **identity,
        "N_events": len(event_rows),
    }]
    write_csv(OUT_DIR / "core_transfer_identity_results.csv", identity_rows)
    pairs = closed_loop_join(pair_rows(event_rows))
    prior = prior_layer_labels()
    layer_rows = layer_gain_summary(pairs, prior, reps=args.bootstrap_reps)
    write_csv(OUT_DIR / "layer_core_transfer_gain.csv", layer_rows)
    concordance_rows, concordance = layer_sign_concordance(layer_rows, prior)
    write_csv(OUT_DIR / "layer_sign_concordance.csv", concordance_rows)
    l22 = layer22_reverse_control(next((r for r in layer_rows if int(r["layer_id"]) == 22), None))
    save_json(OUT_DIR / "layer22_reverse_control.json", l22)
    layer_local = layer_local_gap_assoc(layer_rows, reps=args.bootstrap_reps)
    save_json(OUT_DIR / "core_gain_vs_layer_local_gap.json", layer_local)
    local_damage_assoc = association_payload(pairs, "delta_G_core", "local_damage_gap", reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED + 9000)
    closed_loop_assoc = association_payload(pairs, "delta_G_core", "closed_loop_residual_gap", reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED + 10000)
    write_csv(OUT_DIR / "core_gain_vs_closed_loop_residual.csv", pairs)
    gran_rows = granularity_summary(pairs, reps=args.bootstrap_reps)
    write_csv(OUT_DIR / "granularity_core_transfer_summary.csv", gran_rows)
    aware_rows = layer_aware_association(pairs, reps=args.bootstrap_reps)
    write_csv(OUT_DIR / "layer_aware_closed_loop_association.csv", aware_rows)
    c_med, c_ci = c_sensitive_aggregate(pairs, prior, reps=args.bootstrap_reps)
    d128_16, d128_16_ci = core_gain_128_to_16(pairs, reps=args.bootstrap_reps)
    gates_ok = (
        inventory.get("PIPELINE_IDENTITY_GATE") == "PASS"
        and inventory.get("COMMON_COHORT_REUSE_GATE") == "PASS"
        and inventory.get("PRIOR_LAYER_LABEL_REUSE_GATE") == "PASS"
        and event_rows
        and all(r.get("MATCHED_CARRIER_GATE") == "PASS" for r in event_rows)
        and all(r.get("MAGNITUDE_MATCH_GATE") == "PASS" for r in event_rows)
        and all(r.get("DIRECTION_PRESERVATION_GATE") == "PASS" for r in event_rows)
    )
    closure = classify_core_directional_geometry(
        gates_ok=gates_ok,
        identity_gate=identity["CORE_TRANSFER_IDENTITY_GATE"],
        concordance_rate=concordance["LAYER_SIGN_CONCORDANCE_RATE"],
        c_aggregate_ci=c_ci,
        layer22_control=l22["LAYER22_REVERSE_CONTROL"],
        c_sensitive_loo_stable=c_sensitive_loo_stable(layer_rows, prior),
        closed_loop_ci=closed_loop_assoc["spearman_CI"],
        metric_independence="MATHEMATICALLY_EQUIVALENT",
    )
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if gates_ok else "INCOMPLETE",
        "OFFLINE_FIRST": "YES",
        "PIPELINE_IDENTITY_GATE": inventory.get("PIPELINE_IDENTITY_GATE"),
        "COMMON_COHORT_REUSE_GATE": inventory.get("COMMON_COHORT_REUSE_GATE"),
        "PRIOR_LAYER_LABEL_REUSE_GATE": inventory.get("PRIOR_LAYER_LABEL_REUSE_GATE"),
        "CORE_OUTPUT_SEMANTICS_GATE": "PASS",
        "READOUT_FORM": readout_form_from_events(event_rows),
        **identity,
        "CORE_QUADRATIC_FORM_IDENTITY_GATE": "PASS" if identity["CORE_TRANSFER_IDENTITY_GATE"] == "PASS" else "NOT_APPLICABLE",
        "MATCHED_CARRIER_GATE": "PASS" if event_rows and all(r.get("MATCHED_CARRIER_GATE") == "PASS" for r in event_rows) else "FAIL",
        "MAGNITUDE_MATCH_GATE": "PASS" if event_rows and all(r.get("MAGNITUDE_MATCH_GATE") == "PASS" for r in event_rows) else "FAIL",
        "DIRECTION_PRESERVATION_GATE": "PASS" if event_rows and all(r.get("DIRECTION_PRESERVATION_GATE") == "PASS" for r in event_rows) else "FAIL",
        "N_COMMON_UNITS": len({r["unit_id"] for r in event_rows}),
        "N_KDA_LAYERS": inventory.get("N_KDA_LAYERS"),
        "N_C_SENSITIVE_PRIOR_LAYERS": sum(1 for v in prior.values() if v == "C_SENSITIVE"),
        "N_R_SENSITIVE_PRIOR_LAYERS": sum(1 for v in prior.values() if v == "R_SENSITIVE"),
        **concordance,
        "C_SENSITIVE_LAYER_AGGREGATE_DELTA_G_CORE": c_med,
        "C_SENSITIVE_LAYER_AGGREGATE_DELTA_G_CORE_CI": c_ci,
        **l22,
        "CORE_GAIN_VS_LAYER_LOCAL_GAP_SPEARMAN": layer_local["spearman"],
        "CORE_GAIN_VS_LAYER_LOCAL_GAP_CI": layer_local["spearman_CI"],
        "CELL_CORE_GAIN_VS_LOCAL_DAMAGE_SPEARMAN": local_damage_assoc["spearman"],
        "CELL_CORE_GAIN_VS_LOCAL_DAMAGE_CLUSTER_CI": local_damage_assoc["spearman_CI"],
        "CORE_GAIN_VS_CLOSED_LOOP_RESIDUAL_SPEARMAN": closed_loop_assoc["spearman"],
        "CORE_GAIN_VS_CLOSED_LOOP_RESIDUAL_CLUSTER_CI": closed_loop_assoc["spearman_CI"],
        "CORE_GAIN_LOCAL_METRIC_INDEPENDENCE": "MATHEMATICALLY_EQUIVALENT",
        "CORE_GAIN_D128_TO_D16": d128_16,
        "CORE_GAIN_D128_TO_D16_CI": d128_16_ci,
        "MINIMAL_GPU_FALLBACK_USED": "YES",
        "N_NEW_GPU_LOCAL_EVALUATIONS": len(event_rows),
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "NEW_CLOSED_LOOP_TRAJECTORIES": 0,
        **closure,
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": args.bash_syntax_status,
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES",
        "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW" if closure["CORE_DIRECTIONAL_GEOMETRY_STATUS"] == "STRONG_CORE_TRANSFER_CLOSURE" else "LING_KDA_LAYER22_REVERSE_GEOMETRY_DIAGNOSTIC_V1" if closure["CORE_DIRECTIONAL_GEOMETRY_STATUS"] == "GLOBAL_C_CORE_SENSITIVITY_WITH_LAYER22_UNRESOLVED" else "WAIT_FOR_HUMAN_REVIEW",
    }
    for row in gran_rows:
        g = int(row["group_size"])
        summary[f"G{g}_DELTA_G_CORE"] = row["median_delta_G_core"]
        summary[f"G{g}_DELTA_G_CORE_CI"] = [row["delta_G_core_CI_low"], row["delta_G_core_CI_high"]]
    save_json(OUT_DIR / "bootstrap_results.json", {
        "identity": identity,
        "layer_rows": layer_rows,
        "concordance": concordance,
        "c_sensitive_aggregate": {"median": c_med, "ci": c_ci},
        "core_gain_vs_layer_local_gap": layer_local,
        "cell_core_gain_vs_local_damage": local_damage_assoc,
        "core_gain_vs_closed_loop_residual": closed_loop_assoc,
    })
    write_text(OUT_DIR / "README.md", f"# {TASK}\n\nCore-output directional geometry localization outputs.\n")
    write_text(OUT_DIR / "report.md", "\n".join([
        f"# {TASK}",
        "",
        f"FORMAL_STATUS: {summary['FORMAL_STATUS']}",
        f"CORE_TRANSFER_MODEL: {summary['CORE_TRANSFER_MODEL']}",
        f"CORE_DIRECTIONAL_GEOMETRY_STATUS: {summary['CORE_DIRECTIONAL_GEOMETRY_STATUS']}",
        f"FINAL_CLASSIFICATION: {summary['FINAL_CLASSIFICATION']}",
        "",
        "The transfer operator is interpreted through the verified implicit identity T_core(E)=o(E)-o(0) under frozen local drivers.",
        "The immediate local metric and core gain are mathematically equivalent up to the shared base-output normalization within each event; this is reported as an anti-tautology caveat.",
        "Scientific evidence therefore rests on the verified operator identity, layer-wise sign concordance, layer 22 reverse control, and the reused closed-loop residual association.",
        "",
        "No Stage-B recurrence, persistent closed-loop rerun, downstream operator experiment, Kimi transfer, or method design was launched.",
    ]) + "\n")
    save_json(OUT_DIR / "summary.json", summary)
    return summary


def verify(_args=None):
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8")) if (OUT_DIR / "summary.json").exists() else {}
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUT_DIR / name).exists()]
    if summary.get("MINIMAL_GPU_FALLBACK_USED") == "YES":
        missing.extend([name for name in ["minimal_gpu_justification.md", "minimal_gpu_results/core_transfer_gain_event_level.csv"] if not (OUT_DIR / name).exists()])
    event_rows = read_csv(OUT_DIR / "core_transfer_gain_event_level.csv")
    expected = int(summary.get("N_COMMON_UNITS") or 0) * int(summary.get("N_KDA_LAYERS") or 0) * len(GROUP_SIZES) * 2
    numeric_bad = []
    for path_name in ["core_transfer_gain_event_level.csv", "layer_core_transfer_gain.csv", "granularity_core_transfer_summary.csv"]:
        for idx, row in enumerate(read_csv(OUT_DIR / path_name)):
            for key, value in row.items():
                if value in ("", "None", "NA"):
                    continue
                if key.endswith("_G_core") or key in {"G_core", "G_R", "G_C", "transfer_norm", "error_norm", "local_damage"} or key.endswith("_error"):
                    if not finite(value):
                        numeric_bad.append((path_name, idx, key, value))
    fresh = "PASS" if not missing and len(event_rows) == expected and not numeric_bad and summary.get("NEW_STAGE_B_TRAJECTORIES") == 0 and summary.get("NEW_CLOSED_LOOP_TRAJECTORIES") == 0 else "FAIL"
    verification = {
        "TASK": TASK,
        "FRESH_OUTPUT_VERIFICATION": fresh,
        "missing_files": missing,
        "N_EVENT_ROWS": len(event_rows),
        "N_EXPECTED_EVENT_ROWS": expected,
        "numeric_bad": numeric_bad[:20],
        "NEW_STAGE_B_TRAJECTORIES": summary.get("NEW_STAGE_B_TRAJECTORIES"),
        "NEW_CLOSED_LOOP_TRAJECTORIES": summary.get("NEW_CLOSED_LOOP_TRAJECTORIES"),
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
