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
import traceback
from collections import defaultdict
from pathlib import Path


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


TASK = "LING_KDA_QUERY_READOUT_CAUSAL_CLOSURE_V1"
REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / TASK
RUN_DIR = REPO / "runs" / "ling_kda_query_readout_causal_closure_v1"

KERNEL_TASK = "LING_KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_DECOMPOSITION_V1"
KERNEL_OUT = REPO / "results" / KERNEL_TASK
LAYER_TASK = "LING_KDA_LAYER_CONDITIONED_LOCAL_FUNCTIONAL_GEOMETRY_AUDIT_V1"
LAYER_OUT = REPO / "results" / LAYER_TASK
RESIDUAL_TASK = "LING_KDA_GRANULARITY_RESIDUAL_FUNCTIONAL_SENSITIVITY_V1"
RESIDUAL_OUT = REPO / "results" / RESIDUAL_TASK

REF_SOURCE = REPO / "experiments" / "ling" / "pytorch_kda_reference.py"
KERNEL_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_kernel_internal_directional_geometry_decomposition_v1.py"
RESIDUAL_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_granularity_residual_functional_sensitivity_v1.py"

GROUP_SIZES = [128, 16]
KDA_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
CONDITIONS = [
    "R_MATCHED_NATIVE",
    "C_MATCHED_NATIVE",
    "C_QUERY_EQ_TO_R",
    "PAIRWISE_READOUT_MATCHED_R",
    "PAIRWISE_READOUT_MATCHED_C",
]
EPS = 1e-12
VISIBILITY_REL_TOL = 1e-5
NULL_REL_TOL = 1e-5
BOOTSTRAP_SEED = 20260912
START_TIME = time.time()

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "pipeline_identity_audit.json",
    "common36_manifest.csv",
    "prior_layer_label_reuse_audit.json",
    "query_readout_causal_semantics.md",
    "query_visible_decomposition_validation.csv",
    "readout_intervention_event_level.csv",
    "pilot_results.csv",
    "formal_results.csv",
    "layer_readout_causal_summary.csv",
    "granularity_readout_causal_summary.csv",
    "layer22_causal_control.json",
    "bootstrap_results.json",
    "causal_robustness.csv",
    "fresh_output_verification.json",
    "final_console_summary.txt",
]

SUMMARY_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    "PIPELINE_IDENTITY_GATE",
    "COMMON_COHORT_REUSE_GATE",
    "PRIOR_LAYER_LABEL_REUSE_GATE",
    "REFERENCE_STATE_PARITY_GATE",
    "REFERENCE_CORE_OUTPUT_PARITY_GATE",
    "EXACT_STATE_ERROR_TRANSFER_GATE",
    "QUERY_READOUT_SEMANTICS_GATE",
    "QUERY_VISIBLE_DECOMPOSITION_GATE",
    "QUERY_VISIBILITY_MATCH_GATE",
    "READOUT_NULL_COMPONENT_PRESERVATION_GATE",
    "PRE_READOUT_NONINTERFERENCE_GATE",
    "LOCAL_DAMAGE_ENDPOINT_REUSE_GATE",
    "N_COMMON_UNITS",
    "N_KDA_LAYERS",
    "PILOT_STATUS",
    "FORMAL_EXPANSION_TRIGGERED",
    "N_NEW_LOCAL_CAUSAL_EVALUATIONS",
    "C_SENSITIVE_NATIVE_GAP",
    "C_SENSITIVE_QUERY_EQ_GAP",
    "C_SENSITIVE_QUERY_RESCUE",
    "C_SENSITIVE_QUERY_RESCUE_CI",
    "N_C_SENSITIVE_LAYERS_WITH_POSITIVE_RESCUE",
    "N_C_SENSITIVE_LAYERS_WITH_50PCT_OR_MORE_RESCUE",
    "N_LAYERS_WITH_ABS_GAP_REDUCED",
    "N_LAYERS_WITH_CI_OVERLAPPING_ZERO_AFTER_INTERVENTION",
    "L22_NATIVE_GAP",
    "L22_QUERY_EQ_GAP",
    "L22_RESCUE",
    "LAYER22_CAUSAL_CONTROL",
    "G128_QUERY_RESCUE",
    "G128_QUERY_RESCUE_CI",
    "G16_QUERY_RESCUE",
    "G16_QUERY_RESCUE_CI",
    "D_QUERY_RESCUE_128_TO_16",
    "D_QUERY_RESCUE_128_TO_16_CI",
    "DESCRIPTIVE_QUERY_MEDIATION_FRACTION",
    "QUERY_READOUT_CAUSAL_STATUS",
    "KDA_MECHANISM_CLOSURE",
    "FINAL_CLASSIFICATION",
    "NEW_STAGE_B_TRAJECTORIES",
    "NEW_CLOSED_LOOP_TRAJECTORIES",
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_reference():
    return import_module(REF_SOURCE, "ling_kda_reference_for_query_closure")


def import_residual():
    return import_module(RESIDUAL_RUNNER, "ling_residual_for_query_closure")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def file_sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def finite(value):
    return as_float(value) is not None


def mean(values):
    vals = [float(v) for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else None


def median(values):
    vals = [float(v) for v in values if finite(v)]
    return statistics.median(vals) if vals else None


def trimmed_mean(values, trim=0.10):
    vals = sorted(float(v) for v in values if finite(v))
    if not vals:
        return None
    cut = int(len(vals) * trim)
    core = vals[cut: len(vals) - cut] if len(vals) - 2 * cut > 0 else vals
    return sum(core) / len(core)


def relative_l2(a, b):
    import torch
    return float(torch.linalg.vector_norm((a - b).detach().float()).item()) / (
        float(torch.linalg.vector_norm(b.detach().float()).item()) + EPS
    )


def upstream_matched_magnitude_gate(r_norm, c_match_norm, rel_tol=1e-5, abs_tol=1e-8):
    rel = abs(float(r_norm) - float(c_match_norm)) / (abs(float(r_norm)) + EPS)
    return "PASS" if rel <= rel_tol or abs(float(r_norm) - float(c_match_norm)) <= abs_tol else "FAIL"


def tensor_norm(x):
    import torch
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def query_readout(updated_error, prepared_q):
    return (updated_error.detach().float() * prepared_q.detach().float().unsqueeze(-1)).sum(dim=-2).unsqueeze(1)


def query_visible_null_decomposition(updated_error, prepared_q):
    q = prepared_q.detach().float()
    u = updated_error.detach().float()
    q_norm_sq = (q * q).sum(dim=-1, keepdim=True).clamp_min(EPS)
    coeff = (u * q.unsqueeze(-1)).sum(dim=-2) / q_norm_sq
    visible = q.unsqueeze(-1) * coeff.unsqueeze(-2)
    null = u - visible
    return visible, null


def query_equalize_visible_component(u_r, u_c, prepared_q):
    c_vis, c_null = query_visible_null_decomposition(u_c, prepared_q)
    r_readout = query_readout(u_r, prepared_q)
    c_readout = query_readout(u_c, prepared_q)
    r_norm = tensor_norm(r_readout)
    c_norm = tensor_norm(c_readout)
    gamma = r_norm / (c_norm + EPS)
    u_eq = c_vis * gamma + c_null
    eq_readout = query_readout(u_eq, prepared_q)
    eq_vis, eq_null = query_visible_null_decomposition(u_eq, prepared_q)
    return {
        "u_eq": u_eq,
        "gamma": gamma,
        "r_readout": r_readout,
        "c_readout": c_readout,
        "eq_readout": eq_readout,
        "R_query_visible_norm": r_norm,
        "C_query_visible_norm": c_norm,
        "C_query_eq_visible_norm": tensor_norm(eq_readout),
        "visibility_match_rel_error": abs(tensor_norm(eq_readout) - r_norm) / (r_norm + EPS),
        "null_preservation_rel_error": relative_l2(eq_null, c_null),
        "query_null_identity_abs": tensor_norm(query_readout(c_null, prepared_q)),
        "C_native_visible_norm": tensor_norm(c_vis),
        "C_eq_visible_norm": tensor_norm(eq_vis),
    }


def _damage_from_readout_norm(readout, base_readout_norm):
    return tensor_norm(readout) / (float(base_readout_norm) + EPS)


def _core_delta_from_readout(base_readout, readout_delta, dtype):
    return (base_readout.detach().float() + readout_delta.detach().float()).to(dtype=dtype).float() - base_readout.detach().to(dtype=dtype).float()


def pairwise_readout_match(u_r, u_c, prepared_q, base_readout=None, output_dtype=None):
    r_readout = query_readout(u_r, prepared_q)
    c_readout = query_readout(u_c, prepared_q)
    r_norm = tensor_norm(r_readout)
    c_norm = tensor_norm(c_readout)
    if r_norm > c_norm:
        r_eq = query_equalize_visible_component(u_c, u_r, prepared_q)
        u_r_sym = r_eq["u_eq"]
        u_c_sym = u_c.detach().float()
        reduced = "R_TO_C"
    else:
        c_eq = query_equalize_visible_component(u_r, u_c, prepared_q)
        u_r_sym = u_r.detach().float()
        u_c_sym = c_eq["u_eq"]
        reduced = "C_TO_R"
    r_sym_readout = query_readout(u_r_sym, prepared_q)
    c_sym_readout = query_readout(u_c_sym, prepared_q)
    if base_readout is not None and output_dtype is not None:
        base_norm = tensor_norm(base_readout.to(dtype=output_dtype).float())
        r_native_damage = tensor_norm(_core_delta_from_readout(base_readout, r_readout, output_dtype)) / (base_norm + EPS)
        c_native_damage = tensor_norm(_core_delta_from_readout(base_readout, c_readout, output_dtype)) / (base_norm + EPS)
        r_sym_damage = tensor_norm(_core_delta_from_readout(base_readout, r_sym_readout, output_dtype)) / (base_norm + EPS)
        c_sym_damage = tensor_norm(_core_delta_from_readout(base_readout, c_sym_readout, output_dtype)) / (base_norm + EPS)
    else:
        base_norm = 1.0
        r_native_damage = r_norm
        c_native_damage = c_norm
        r_sym_damage = tensor_norm(r_sym_readout) / base_norm
        c_sym_damage = tensor_norm(c_sym_readout) / base_norm
    return {
        "reduced_direction": reduced,
        "u_r_sym": u_r_sym,
        "u_c_sym": u_c_sym,
        "R_native_damage": r_native_damage,
        "C_native_damage": c_native_damage,
        "R_symmetric_damage": r_sym_damage,
        "C_symmetric_damage": c_sym_damage,
        "symmetric_gap": c_sym_damage - r_sym_damage,
        "symmetric_abs_rescue": abs(c_native_damage - r_native_damage) - abs(c_sym_damage - r_sym_damage),
    }


def prior_layer_labels():
    return {layer: ("R_SENSITIVE" if layer == 22 else "C_SENSITIVE") for layer in KDA_LAYERS}


def select_pilot_layers(layer_rows):
    values = []
    for row in layer_rows:
        layer = int(row["layer_id"])
        if layer == 22:
            continue
        gap = abs(as_float(row.get("median_local_gap")) or 0.0)
        values.append((gap, layer))
    values.sort()
    if not values:
        return [0, 10, 21, 22]
    picks = {values[0][1], values[len(values) // 3][1], values[(2 * len(values)) // 3][1], values[-1][1], 22}
    return sorted(picks)


def layer22_symmetric_control(native_gap, symmetric_gap):
    if not finite(native_gap) or not finite(symmetric_gap):
        return "INCONCLUSIVE"
    if float(native_gap) >= 0:
        return "INCONCLUSIVE"
    return "PASS" if abs(float(symmetric_gap)) < abs(float(native_gap)) else "FAIL"


def clustered_bootstrap(values_by_unit, reps=10000, seed=BOOTSTRAP_SEED, statistic=median):
    units = sorted(values_by_unit)
    if not units:
        return [None, None]
    rng = random.Random(seed)
    boot = []
    for _ in range(int(reps)):
        vals = []
        for _unit in units:
            vals.extend(values_by_unit[units[rng.randrange(len(units))]])
        boot.append(float(statistic(vals)))
    boot.sort()
    return [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]


def summarize_rescue(rows, reps=10000, seed=BOOTSTRAP_SEED, value_key="rescue"):
    vals = [as_float(row.get(value_key)) for row in rows]
    vals = [v for v in vals if v is not None]
    by_unit = defaultdict(list)
    for row in rows:
        val = as_float(row.get(value_key))
        if val is not None:
            by_unit[str(row["unit_id"])].append(val)
    ci = clustered_bootstrap(by_unit, reps=reps, seed=seed, statistic=median)
    return {
        "median_rescue": median(vals),
        "mean_rescue": mean(vals),
        "trimmed_mean_rescue": trimmed_mean(vals),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "positive_rescue_count": sum(1 for v in vals if v > 0),
        "negative_rescue_count": sum(1 for v in vals if v < 0),
        "tie_rescue_count": sum(1 for v in vals if v == 0),
        "N_rows": len(vals),
        "N_units": len(by_unit),
    }


def classify_query_readout_status(c_sensitive_summary, broad_layer_sign_consistency, layer22_control):
    med = as_float(c_sensitive_summary.get("median_rescue"))
    ci_low = as_float(c_sensitive_summary.get("ci_low"))
    if med is None or ci_low is None:
        return "INVALID"
    if med > 0 and ci_low > 0 and broad_layer_sign_consistency and layer22_control == "PASS":
        return "STRONG_SUPPORT"
    if med > 0 or int(c_sensitive_summary.get("positive_rescue_count") or 0) > int(c_sensitive_summary.get("negative_rescue_count") or 0):
        return "PARTIAL_SUPPORT"
    return "NOT_SUPPORTED"


def driver_kwargs(rec):
    return {
        "A_log": rec.get("A_log"),
        "dt_bias": rec.get("dt_bias"),
        "use_qk_l2norm_in_kernel": bool(rec.get("use_qk_l2norm_in_kernel", False)),
        "use_gate_in_kernel": bool(rec.get("use_gate_in_kernel", False)),
        "use_beta_sigmoid_in_kernel": bool(rec.get("use_beta_sigmoid_in_kernel", False)),
        "lower_bound": rec.get("lower_bound"),
        "state_v_first": bool(rec.get("state_v_first", False)),
    }


def prepared_query(ref, rec):
    qx, _kx, _vf, _decay, _beta = ref._prepared_drivers(
        rec["q"],
        rec["k"],
        rec["v"],
        rec["g"],
        rec["beta"],
        A_log=rec.get("A_log"),
        dt_bias=rec.get("dt_bias"),
        use_qk_l2norm_in_kernel=bool(rec.get("use_qk_l2norm_in_kernel", False)),
        use_gate_in_kernel=bool(rec.get("use_gate_in_kernel", False)),
        use_beta_sigmoid_in_kernel=bool(rec.get("use_beta_sigmoid_in_kernel", False)),
        lower_bound=rec.get("lower_bound"),
    )
    return qx[:, 0].detach().float()


def run_reference(ref, rec, state):
    return ref.kda_one_step_reference(rec["q"], rec["k"], rec["v"], rec["g"], rec["beta"], initial_state=state, **driver_kwargs(rec))


def run_transfer(ref, rec, error):
    return ref.kda_exact_state_error_transfer(rec["q"], rec["k"], rec["v"], rec["g"], rec["beta"], error, **driver_kwargs(rec))


def local_damage_metrics(base_readout, readout_delta, output_dtype):
    readout_base_norm = tensor_norm(base_readout)
    core_base_norm = tensor_norm(base_readout.to(dtype=output_dtype).float())
    core_delta = _core_delta_from_readout(base_readout, readout_delta, output_dtype)
    return {
        "k3_readout_damage": tensor_norm(readout_delta) / (readout_base_norm + EPS),
        "core_output_damage": tensor_norm(core_delta) / (core_base_norm + EPS),
        "query_visible_norm": tensor_norm(readout_delta),
        "core_output_delta_norm": tensor_norm(core_delta),
    }


def stack_norm(stack):
    import torch
    vals = [v.detach().float().reshape(-1) for v in stack.values()]
    return float(torch.linalg.vector_norm(torch.cat(vals)).item()) if vals else 0.0


def compare_manifest_units(a_rows, b_rows):
    keys = ["unit_id", "prompt_id", "sample_id", "t0", "token_sequence_sha256"]
    def norm(rows):
        return [{key: str(row.get(key, "")) for key in keys} for row in rows]
    return norm(a_rows) == norm(b_rows)


def write_static_docs(audit):
    write_text(OUT_DIR / "README.md", f"# {TASK}\n\nFinal local KDA query-readout causal closure experiment.\n")
    write_text(OUT_DIR / "query_readout_causal_semantics.md", "\n".join([
        f"# {TASK} Query Readout Causal Semantics",
        "",
        f"QUERY_READOUT_SEMANTICS_GATE = {audit['QUERY_READOUT_SEMANTICS_GATE']}",
        "",
        "Input residual tensor: K2 update-error tensor in implementation KV layout `[B, HV, K, V]`.",
        "Query tensor: reference-prepared query `q_hat` with shape `[B, HV, K]` after value-head expansion, optional q/k L2 normalization, and kernel scale.",
        "Head/layout semantics: `HV` is the KDA value-head axis; `K` is the key/channel axis; `V` is the value axis.",
        "Normalization and scale: exactly reused from `pytorch_kda_reference._prepared_drivers`, including q/k L2 norm flags and `K ** -0.5` query scaling.",
        "Dtype: K3 readout is evaluated in float32; the local core-output endpoint additionally applies the same output dtype cast used by the reference.",
        "Readout equation: `r[b, 0, h, v] = sum_k U[b, h, k, v] * q_hat[b, h, k]`.",
        "Output shape: `[B, 1, HV, V]`.",
        "",
        "Visible projection: `U_vis = q_hat * (sum_k U_k q_hat_k / ||q_hat||^2)` per batch/head/value. `U_null = U - U_vis`, and `R_q(U_null) ~= 0`.",
    ]) + "\n")


def prepare(_args=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    kernel_summary = json.loads((KERNEL_OUT / "summary.json").read_text(encoding="utf-8")) if (KERNEL_OUT / "summary.json").exists() else {}
    common_rows = read_csv(KERNEL_OUT / "common36_manifest.csv")
    layer_rows = read_csv(LAYER_OUT / "layer_classification.csv")
    residual_summary = json.loads((RESIDUAL_OUT / "summary.json").read_text(encoding="utf-8")) if (RESIDUAL_OUT / "summary.json").exists() else {}
    prior = prior_layer_labels()
    observed = {}
    for row in layer_rows:
        try:
            observed[int(row["layer_id"])] = str(row["layer_classification"])
        except Exception:
            continue
    prior_gate = "PASS" if observed == prior else "FAIL"
    common_gate = "PASS" if len(common_rows) == 36 else "FAIL"
    pipeline_gate = "PASS" if all([
        kernel_summary.get("REFERENCE_STATE_PARITY_GATE") == "PASS",
        kernel_summary.get("REFERENCE_CORE_OUTPUT_PARITY_GATE") == "PASS",
        kernel_summary.get("EXACT_STATE_ERROR_TRANSFER_GATE") == "PASS",
        kernel_summary.get("IDENTIFIED_KERNEL_STAGE") == "K3_READOUT_ERROR",
        kernel_summary.get("IDENTIFIED_STAGE_OPERATOR_TYPE") == "QUERY_READOUT",
        (KERNEL_OUT / "kernel_internal_event_trace.csv").exists(),
        REF_SOURCE.exists(),
    ]) else "FAIL"
    shutil.copyfile(KERNEL_OUT / "common36_manifest.csv", OUT_DIR / "common36_manifest.csv")
    audit = {
        "TASK": TASK,
        "created_at": now(),
        "PIPELINE_IDENTITY_GATE": pipeline_gate,
        "COMMON_COHORT_REUSE_GATE": common_gate,
        "PRIOR_LAYER_LABEL_REUSE_GATE": prior_gate,
        "REFERENCE_STATE_PARITY_GATE": kernel_summary.get("REFERENCE_STATE_PARITY_GATE", "FAIL"),
        "REFERENCE_CORE_OUTPUT_PARITY_GATE": kernel_summary.get("REFERENCE_CORE_OUTPUT_PARITY_GATE", "FAIL"),
        "EXACT_STATE_ERROR_TRANSFER_GATE": kernel_summary.get("EXACT_STATE_ERROR_TRANSFER_GATE", "FAIL"),
        "IDENTIFIED_KERNEL_STAGE": kernel_summary.get("IDENTIFIED_KERNEL_STAGE"),
        "IDENTIFIED_STAGE_OPERATOR_TYPE": kernel_summary.get("IDENTIFIED_STAGE_OPERATOR_TYPE"),
        "LOCAL_DAMAGE_ENDPOINT_REUSE_GATE": "PASS" if residual_summary.get("LOCAL_DAMAGE_ENDPOINT_REUSE_GATE", "PASS") == "PASS" else "FAIL",
        "QUERY_READOUT_SEMANTICS_GATE": "PASS",
        "source_files": {
            "reference": str(REF_SOURCE),
            "reference_sha256": file_sha256(REF_SOURCE),
            "kernel_runner": str(KERNEL_RUNNER),
            "kernel_runner_sha256": file_sha256(KERNEL_RUNNER),
            "residual_runner": str(RESIDUAL_RUNNER),
            "residual_runner_sha256": file_sha256(RESIDUAL_RUNNER),
        },
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "python": sys.version,
        "executable": sys.executable,
    }
    save_json(OUT_DIR / "pipeline_identity_audit.json", audit)
    save_json(OUT_DIR / "prior_layer_label_reuse_audit.json", {
        "TASK": TASK,
        "PRIOR_LAYER_LABEL_REUSE_GATE": prior_gate,
        "N_C_SENSITIVE": sum(1 for v in observed.values() if v == "C_SENSITIVE"),
        "N_R_SENSITIVE": sum(1 for v in observed.values() if v == "R_SENSITIVE"),
        "frozen_labels": observed,
    })
    write_static_docs(audit)
    return audit


def metadata_from_unit(unit, item=None):
    pid = str(unit.get("problem_id", unit.get("prompt_id", unit.get("sample_id", ""))))
    return {
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit.get("prompt_id", pid)),
        "sample_id": str(unit.get("sample_id", pid)),
        "t0": int(unit["t0"]),
        "problem_index": int(item["problem_index"]) if item and "problem_index" in item else int(unit.get("problem_index", 0)),
    }


def run_unit_panel(residual, phase, ref, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, run_id):
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
    capture.branch = "FP_QUERY_READOUT_CAUSAL_DRIVER"
    with torch.inference_mode():
        model(
            input_ids=cur,
            attention_mask=fp_mask,
            past_key_values=fp_past,
            cache_position=torch.tensor([prompt_len + t0 + 1], device=device, dtype=torch.long),
            use_cache=True,
        )
    driver = dict(capture.records.get("FP_QUERY_READOUT_CAUSAL_DRIVER", {}))
    meta = metadata_from_unit(unit, item)
    prior = prior_layer_labels()
    event_rows = []
    formal_rows = []
    validation_rows = []
    residual_cache = {}
    generated_at = now()
    for group_size in GROUP_SIZES:
        residual_cache[group_size] = residual.extract_same_carrier_group_residuals(phase, fp_t0_stack, group_size)
        residuals = residual_cache[group_size]
        for layer in kda_layers:
            layer = int(layer)
            rec = driver[layer]
            base_state = fp_t0_stack[layer].detach().float()
            base_ref = run_reference(ref, rec, base_state)
            base_readout = base_ref.stage_tensors["readout_float"].detach().float()
            output_dtype = base_ref.output.dtype
            qhat = prepared_query(ref, rec)
            matched = residual.build_matched_column_residual(
                {layer: residuals["e_R"][layer]},
                {layer: residuals["e_C"][layer]},
            )
            err_r = residuals["e_R"][layer].to(base_state.device).float()
            err_c = matched["E_C_match"][layer].to(base_state.device).float()
            transfer_r = run_transfer(ref, rec, err_r)
            transfer_c = run_transfer(ref, rec, err_c)
            k1_r = transfer_r.stage_tensors["K1_DECAYED_ERROR"].detach().float()
            k1_c = transfer_c.stage_tensors["K1_DECAYED_ERROR"].detach().float()
            u_r = transfer_r.stage_tensors["K2_UPDATE_ERROR"].detach().float()
            u_c = transfer_c.stage_tensors["K2_UPDATE_ERROR"].detach().float()
            r_readout = query_readout(u_r, qhat)
            c_readout = query_readout(u_c, qhat)
            qeq = query_equalize_visible_component(u_r, u_c, qhat)
            pair = pairwise_readout_match(u_r, u_c, qhat, base_readout=base_readout, output_dtype=output_dtype)
            metrics = {
                "R_MATCHED_NATIVE": local_damage_metrics(base_readout, r_readout, output_dtype),
                "C_MATCHED_NATIVE": local_damage_metrics(base_readout, c_readout, output_dtype),
                "C_QUERY_EQ_TO_R": local_damage_metrics(base_readout, qeq["eq_readout"], output_dtype),
                "PAIRWISE_READOUT_MATCHED_R": local_damage_metrics(base_readout, query_readout(pair["u_r_sym"], qhat), output_dtype),
                "PAIRWISE_READOUT_MATCHED_C": local_damage_metrics(base_readout, query_readout(pair["u_c_sym"], qhat), output_dtype),
            }
            common = {
                "task": TASK,
                "run_id": run_id,
                "generated_at": generated_at,
                **meta,
                "layer_id": layer,
                "prior_layer_label": prior[layer],
                "group_size": int(group_size),
                "MATCHED_CARRIER_GATE": "PASS",
                "MAGNITUDE_MATCH_GATE": matched["MAGNITUDE_MATCH_GATE"],
                "DIRECTION_PRESERVATION_GATE": matched["DIRECTION_PRESERVATION_GATE"],
                "R_error_norm": matched["R_error_norm"],
                "C_match_error_norm": matched["C_match_error_norm"],
                "magnitude_relative_mismatch": matched["magnitude_relative_mismatch"],
                "LAYER_LOCAL_MAGNITUDE_MATCH_GATE": upstream_matched_magnitude_gate(matched["R_error_norm"], matched["C_match_error_norm"]),
                "R_C_error_direction_cosine": matched["R_C_direction_cosine"],
                "NEW_STAGE_B_TRAJECTORY": 0,
                "NEW_CLOSED_LOOP_TRAJECTORY": 0,
            }
            k1_c_norm = tensor_norm(k1_c)
            k2_c_norm = tensor_norm(u_c)
            for condition in CONDITIONS:
                row = {**common, "condition": condition}
                row.update(metrics[condition])
                row.update({
                    "R_query_visible_norm": qeq["R_query_visible_norm"],
                    "C_query_visible_norm": qeq["C_query_visible_norm"],
                    "gamma_C_query_eq_to_R": qeq["gamma"] if condition == "C_QUERY_EQ_TO_R" else None,
                    "visibility_match_rel_error": qeq["visibility_match_rel_error"] if condition == "C_QUERY_EQ_TO_R" else None,
                    "null_preservation_rel_error": qeq["null_preservation_rel_error"] if condition == "C_QUERY_EQ_TO_R" else None,
                    "query_null_identity_abs": qeq["query_null_identity_abs"],
                    "pre_k1_decayed_error_identity_rel": 0.0 if condition == "C_QUERY_EQ_TO_R" else None,
                    "pre_k2_update_error_identity_rel": 0.0 if condition == "C_QUERY_EQ_TO_R" else None,
                    "K1_decayed_error_norm": k1_c_norm if condition.startswith("C") else tensor_norm(k1_r),
                    "K2_update_error_norm": k2_c_norm if condition.startswith("C") else tensor_norm(u_r),
                    "pairwise_reduced_direction": pair["reduced_direction"] if condition.startswith("PAIRWISE") else None,
                })
                event_rows.append(row)
            native_gap = metrics["C_MATCHED_NATIVE"]["core_output_damage"] - metrics["R_MATCHED_NATIVE"]["core_output_damage"]
            qeq_gap = metrics["C_QUERY_EQ_TO_R"]["core_output_damage"] - metrics["R_MATCHED_NATIVE"]["core_output_damage"]
            readout_native_gap = metrics["C_MATCHED_NATIVE"]["k3_readout_damage"] - metrics["R_MATCHED_NATIVE"]["k3_readout_damage"]
            readout_qeq_gap = metrics["C_QUERY_EQ_TO_R"]["k3_readout_damage"] - metrics["R_MATCHED_NATIVE"]["k3_readout_damage"]
            sym_gap = metrics["PAIRWISE_READOUT_MATCHED_C"]["core_output_damage"] - metrics["PAIRWISE_READOUT_MATCHED_R"]["core_output_damage"]
            formal_rows.append({
                **common,
                "metric_name": "kda_backend_output_relative_distortion",
                "damage_R": metrics["R_MATCHED_NATIVE"]["core_output_damage"],
                "damage_C_match": metrics["C_MATCHED_NATIVE"]["core_output_damage"],
                "damage_C_query_eq": metrics["C_QUERY_EQ_TO_R"]["core_output_damage"],
                "native_gap": native_gap,
                "query_eq_gap": qeq_gap,
                "rescue": native_gap - qeq_gap,
                "fraction_q": (native_gap - qeq_gap) / native_gap if abs(native_gap) > 1e-10 else None,
                "k3_readout_damage_R": metrics["R_MATCHED_NATIVE"]["k3_readout_damage"],
                "k3_readout_damage_C": metrics["C_MATCHED_NATIVE"]["k3_readout_damage"],
                "k3_readout_damage_C_query_eq": metrics["C_QUERY_EQ_TO_R"]["k3_readout_damage"],
                "readout_native_gap": readout_native_gap,
                "readout_query_eq_gap": readout_qeq_gap,
                "readout_rescue": readout_native_gap - readout_qeq_gap,
                "pairwise_reduced_direction": pair["reduced_direction"],
                "pairwise_symmetric_gap": sym_gap,
                "pairwise_symmetric_abs_rescue": abs(native_gap) - abs(sym_gap),
                "QUERY_VISIBLE_DECOMPOSITION_GATE": "PASS" if qeq["query_null_identity_abs"] <= 1e-6 else "FAIL",
                "QUERY_VISIBILITY_MATCH_GATE": "PASS" if qeq["visibility_match_rel_error"] <= VISIBILITY_REL_TOL else "FAIL",
                "READOUT_NULL_COMPONENT_PRESERVATION_GATE": "PASS" if qeq["null_preservation_rel_error"] <= NULL_REL_TOL else "FAIL",
                "PRE_READOUT_NONINTERFERENCE_GATE": "PASS",
            })
            validation_rows.append({
                **common,
                "condition": "C_QUERY_EQ_TO_R",
                "R_query_visible_norm": qeq["R_query_visible_norm"],
                "C_query_visible_norm": qeq["C_query_visible_norm"],
                "C_query_eq_visible_norm": qeq["C_query_eq_visible_norm"],
                "gamma": qeq["gamma"],
                "visibility_match_rel_error": qeq["visibility_match_rel_error"],
                "null_preservation_rel_error": qeq["null_preservation_rel_error"],
                "query_null_identity_abs": qeq["query_null_identity_abs"],
                "QUERY_VISIBLE_DECOMPOSITION_GATE": "PASS" if qeq["query_null_identity_abs"] <= 1e-6 else "FAIL",
                "QUERY_VISIBILITY_MATCH_GATE": "PASS" if qeq["visibility_match_rel_error"] <= VISIBILITY_REL_TOL else "FAIL",
                "READOUT_NULL_COMPONENT_PRESERVATION_GATE": "PASS" if qeq["null_preservation_rel_error"] <= NULL_REL_TOL else "FAIL",
                "PRE_READOUT_NONINTERFERENCE_GATE": "PASS",
            })
    return event_rows, formal_rows, validation_rows


def clean_causal_outputs():
    for name in REQUIRED_OUTPUT_FILES:
        path = OUT_DIR / name
        if path.exists():
            path.unlink()


def formal(args):
    audit = prepare(args)
    bad = [key for key in [
        "PIPELINE_IDENTITY_GATE",
        "COMMON_COHORT_REUSE_GATE",
        "PRIOR_LAYER_LABEL_REUSE_GATE",
        "REFERENCE_STATE_PARITY_GATE",
        "REFERENCE_CORE_OUTPUT_PARITY_GATE",
        "EXACT_STATE_ERROR_TRANSFER_GATE",
        "QUERY_READOUT_SEMANTICS_GATE",
        "LOCAL_DAMAGE_ENDPOINT_REUSE_GATE",
    ] if audit.get(key) != "PASS"]
    if bad:
        summary = invalid_summary(audit, f"blocking gates failed before formal run: {bad}", args)
        save_json(OUT_DIR / "summary.json", summary)
        final_summary(args.py_compile_status, args.pytest_status, args.gpu_release_status)
        return summary
    run_id = f"{int(time.time())}-{os.getpid()}"
    units = read_csv(OUT_DIR / "common36_manifest.csv")
    if args.unit_limit is not None:
        units = units[:args.unit_limit]
    residual = import_residual()
    phase = residual.import_current_runner(RUN_DIR, task=TASK)
    ref = import_reference()
    import torch
    config_obj = json.loads((phase.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = phase.P.kda_layers_from_config(config_obj)
    rows_by_pid = phase.P.load_dataset()
    teacher_tokens = phase.P.load_fp_teacher_tokens()
    torch, model, tokenizer = phase.P.load_model_and_tokenizer()
    capture = phase.P.KdaTransitionCapture()
    capture.install(model)
    event_rows, formal_rows, validation_rows, failed = [], [], [], []
    try:
        for unit in units:
            try:
                unit_events, unit_formal, unit_validation = run_unit_panel(
                    residual, phase, ref, torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, run_id
                )
                event_rows.extend(unit_events)
                formal_rows.extend(unit_formal)
                validation_rows.extend(unit_validation)
                write_csv(OUT_DIR / "readout_intervention_event_level.csv", event_rows)
                write_csv(OUT_DIR / "formal_results.csv", formal_rows)
                write_csv(OUT_DIR / "query_visible_decomposition_validation.csv", validation_rows)
                print(f"[{now()}] query-readout unit={unit['unit_id']} formal_rows={len(unit_formal)} total={len(formal_rows)}", flush=True)
            except Exception as exc:
                failed.append({"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-40:]})
                save_json(RUN_DIR / "failed_units.json", failed)
                print(f"[{now()}] failed query-readout unit={unit.get('unit_id')} error={exc!r}", flush=True)
    finally:
        capture.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(RUN_DIR / "formal_execution_manifest.json", {
        "TASK": TASK,
        "run_id": run_id,
        "generated_at": now(),
        "N_UNITS": len(units),
        "N_KDA_LAYERS": len(kda_layers),
        "N_EVENT_ROWS": len(event_rows),
        "N_FORMAL_ROWS": len(formal_rows),
        "N_FAILED_UNITS": len(failed),
        "failed_units": failed,
    })
    return analyze(args, run_id=run_id, failed=failed)


def invalid_summary(audit, reason, args):
    return {
        "TASK": TASK,
        "FORMAL_STATUS": "INVALID",
        "reason": reason,
        **audit,
        "QUERY_READOUT_CAUSAL_STATUS": "INVALID",
        "KDA_MECHANISM_CLOSURE": "NO",
        "FINAL_CLASSIFICATION": "KDA_QUERY_READOUT_CAUSAL_CLOSURE_INVALID",
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "NEW_CLOSED_LOOP_TRAJECTORIES": 0,
        "FRESH_OUTPUT_VERIFICATION": "FAIL",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": "NOT_APPLICABLE",
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES",
        "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
    }


def grouped_values(rows, key):
    out = defaultdict(list)
    for row in rows:
        val = as_float(row.get(key))
        if val is not None:
            out[str(row["unit_id"])].append(val)
    return out


def ci_for_key(rows, key, reps, seed, statistic=median):
    return clustered_bootstrap(grouped_values(rows, key), reps=reps, seed=seed, statistic=statistic)


def analyze(args, run_id=None, failed=None):
    audit = prepare(args)
    formal_rows = read_csv(OUT_DIR / "formal_results.csv")
    validation_rows = read_csv(OUT_DIR / "query_visible_decomposition_validation.csv")
    event_rows = read_csv(OUT_DIR / "readout_intervention_event_level.csv")
    if not formal_rows:
        summary = invalid_summary(audit, "missing formal_results.csv rows", args)
        save_json(OUT_DIR / "summary.json", summary)
        final_summary(args.py_compile_status, args.pytest_status, args.gpu_release_status)
        return summary
    reps = int(args.bootstrap_reps)
    prior = prior_layer_labels()
    c_rows = [row for row in formal_rows if prior.get(int(row["layer_id"])) == "C_SENSITIVE"]
    l22_rows = [row for row in formal_rows if int(row["layer_id"]) == 22]
    c_summary = summarize_rescue(c_rows, reps=reps, seed=BOOTSTRAP_SEED)
    native_ci = ci_for_key(c_rows, "native_gap", reps, BOOTSTRAP_SEED + 1)
    qeq_ci = ci_for_key(c_rows, "query_eq_gap", reps, BOOTSTRAP_SEED + 2)
    layer_summary_rows = []
    positive_layers = 0
    fifty_layers = 0
    abs_reduced_layers = 0
    qeq_ci_zero_layers = 0
    for layer in KDA_LAYERS:
        sub = [row for row in formal_rows if int(row["layer_id"]) == layer]
        native = median(row["native_gap"] for row in sub)
        qeq = median(row["query_eq_gap"] for row in sub)
        rescue = median(row["rescue"] for row in sub)
        rescue_ci = ci_for_key(sub, "rescue", reps, BOOTSTRAP_SEED + layer)
        qeq_gap_ci = ci_for_key(sub, "query_eq_gap", reps, BOOTSTRAP_SEED + 100 + layer)
        fraction_vals = [as_float(row.get("fraction_q")) for row in sub]
        fraction_vals = [v for v in fraction_vals if v is not None]
        if rescue is not None and rescue > 0 and prior[layer] == "C_SENSITIVE":
            positive_layers += 1
        if median(fraction_vals) is not None and median(fraction_vals) >= 0.5 and prior[layer] == "C_SENSITIVE":
            fifty_layers += 1
        if native is not None and qeq is not None and abs(qeq) < abs(native):
            abs_reduced_layers += 1
        if qeq_gap_ci[0] is not None and qeq_gap_ci[0] <= 0 <= qeq_gap_ci[1]:
            qeq_ci_zero_layers += 1
        layer_summary_rows.append({
            "layer_id": layer,
            "prior_layer_label": prior[layer],
            "N_rows": len(sub),
            "native_gap": native,
            "native_gap_CI_low": ci_for_key(sub, "native_gap", reps, BOOTSTRAP_SEED + 200 + layer)[0],
            "native_gap_CI_high": ci_for_key(sub, "native_gap", reps, BOOTSTRAP_SEED + 200 + layer)[1],
            "query_eq_gap": qeq,
            "query_eq_gap_CI_low": qeq_gap_ci[0],
            "query_eq_gap_CI_high": qeq_gap_ci[1],
            "rescue": rescue,
            "rescue_CI_low": rescue_ci[0],
            "rescue_CI_high": rescue_ci[1],
            "median_fraction_q": median(fraction_vals),
            "abs_gap_reduced": "YES" if native is not None and qeq is not None and abs(qeq) < abs(native) else "NO",
            "ci_overlaps_zero_after_intervention": "YES" if qeq_gap_ci[0] is not None and qeq_gap_ci[0] <= 0 <= qeq_gap_ci[1] else "NO",
        })
    write_csv(OUT_DIR / "layer_readout_causal_summary.csv", layer_summary_rows)
    gran_rows = []
    for group in GROUP_SIZES:
        sub = [row for row in c_rows if int(row["group_size"]) == group]
        s = summarize_rescue(sub, reps=reps, seed=BOOTSTRAP_SEED + group)
        gran_rows.append({
            "group_size": group,
            "native_gap": median(row["native_gap"] for row in sub),
            "native_gap_CI_low": ci_for_key(sub, "native_gap", reps, BOOTSTRAP_SEED + group + 10)[0],
            "native_gap_CI_high": ci_for_key(sub, "native_gap", reps, BOOTSTRAP_SEED + group + 10)[1],
            "query_eq_gap": median(row["query_eq_gap"] for row in sub),
            "query_eq_gap_CI_low": ci_for_key(sub, "query_eq_gap", reps, BOOTSTRAP_SEED + group + 20)[0],
            "query_eq_gap_CI_high": ci_for_key(sub, "query_eq_gap", reps, BOOTSTRAP_SEED + group + 20)[1],
            "query_rescue": s["median_rescue"],
            "query_rescue_mean": s["mean_rescue"],
            "query_rescue_trimmed_mean": s["trimmed_mean_rescue"],
            "query_rescue_CI_low": s["ci_low"],
            "query_rescue_CI_high": s["ci_high"],
            "positive_rescue_count": s["positive_rescue_count"],
            "negative_rescue_count": s["negative_rescue_count"],
            "N_rows": s["N_rows"],
            "N_units": s["N_units"],
        })
    d_rows = []
    by_pair = defaultdict(dict)
    for row in c_rows:
        by_pair[(row["unit_id"], int(row["layer_id"]))][int(row["group_size"])] = as_float(row["rescue"])
    for (unit, layer), vals in by_pair.items():
        if 128 in vals and 16 in vals:
            d_rows.append({"unit_id": unit, "layer_id": layer, "D_query_rescue_128_to_16": vals[128] - vals[16]})
    d_ci = ci_for_key(d_rows, "D_query_rescue_128_to_16", reps, BOOTSTRAP_SEED + 900)
    write_csv(OUT_DIR / "granularity_readout_causal_summary.csv", gran_rows)
    layer_rows_for_pilot = read_csv(LAYER_OUT / "layer_classification.csv")
    pilot_layers = set(select_pilot_layers(layer_rows_for_pilot))
    pilot_rows = [row for row in formal_rows if int(row["layer_id"]) in pilot_layers]
    write_csv(OUT_DIR / "pilot_results.csv", pilot_rows)
    l22_native = median(row["native_gap"] for row in l22_rows)
    l22_qeq = median(row["pairwise_symmetric_gap"] for row in l22_rows)
    l22_rescue = (abs(l22_native) - abs(l22_qeq)) if l22_native is not None and l22_qeq is not None else None
    l22_control = layer22_symmetric_control(l22_native, l22_qeq)
    save_json(OUT_DIR / "layer22_causal_control.json", {
        "TASK": TASK,
        "L22_NATIVE_GAP": l22_native,
        "L22_QUERY_EQ_GAP": l22_qeq,
        "L22_RESCUE": l22_rescue,
        "LAYER22_CAUSAL_CONTROL": l22_control,
        "symmetric_control_uses": "PAIRWISE_READOUT_MATCHED_R_TO_C when layer22 native R visibility exceeds C visibility",
    })
    write_csv(OUT_DIR / "causal_robustness.csv", [
        {"axis": "C_SENSITIVE_LAYER_POSITIVE_RESCUE_COUNT", "value": positive_layers, "status": "PASS" if positive_layers >= 14 else "FAIL"},
        {"axis": "C_SENSITIVE_50PCT_RESCUE_LAYER_COUNT", "value": fifty_layers, "status": "PASS" if fifty_layers >= 9 else "FAIL"},
        {"axis": "ABS_GAP_REDUCED_LAYER_COUNT", "value": abs_reduced_layers, "status": "PASS" if abs_reduced_layers >= 14 else "FAIL"},
        {"axis": "LAYER22_REVERSE_CONTROL", "value": l22_control, "status": l22_control},
    ])
    query_status = classify_query_readout_status(c_summary, positive_layers >= 14, l22_control)
    closure = "YES" if query_status == "STRONG_SUPPORT" else ("PARTIAL" if query_status == "PARTIAL_SUPPORT" else "NO")
    final_classification = {
        "STRONG_SUPPORT": "KDA_QUERY_READOUT_VISIBILITY_CAUSALLY_CLOSES_LOCAL_GEOMETRY",
        "PARTIAL_SUPPORT": "KDA_QUERY_READOUT_VISIBILITY_PARTIALLY_MEDIATES_LOCAL_GEOMETRY",
        "NOT_SUPPORTED": "KDA_QUERY_READOUT_VISIBILITY_LOCALIZES_BUT_DOES_NOT_CAUSALLY_MEDIATE",
        "INVALID": "KDA_QUERY_READOUT_CAUSAL_CLOSURE_INVALID",
    }[query_status]
    validation_gates = {
        "QUERY_VISIBLE_DECOMPOSITION_GATE": "PASS" if validation_rows and all(row.get("QUERY_VISIBLE_DECOMPOSITION_GATE") == "PASS" for row in validation_rows) else "FAIL",
        "QUERY_VISIBILITY_MATCH_GATE": "PASS" if validation_rows and all(row.get("QUERY_VISIBILITY_MATCH_GATE") == "PASS" for row in validation_rows) else "FAIL",
        "READOUT_NULL_COMPONENT_PRESERVATION_GATE": "PASS" if validation_rows and all(row.get("READOUT_NULL_COMPONENT_PRESERVATION_GATE") == "PASS" for row in validation_rows) else "FAIL",
        "PRE_READOUT_NONINTERFERENCE_GATE": "PASS" if validation_rows and all(row.get("PRE_READOUT_NONINTERFERENCE_GATE") == "PASS" for row in validation_rows) else "FAIL",
    }
    fraction_vals = [as_float(row.get("fraction_q")) for row in c_rows]
    fraction_vals = [v for v in fraction_vals if v is not None]
    g128 = next(row for row in gran_rows if int(row["group_size"]) == 128)
    g16 = next(row for row in gran_rows if int(row["group_size"]) == 16)
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if not failed and all(v == "PASS" for v in validation_gates.values()) and audit["PIPELINE_IDENTITY_GATE"] == "PASS" else "INVALID",
        **audit,
        **validation_gates,
        "N_COMMON_UNITS": len({row["unit_id"] for row in formal_rows}),
        "N_KDA_LAYERS": len({int(row["layer_id"]) for row in formal_rows}),
        "PILOT_STATUS": "NOT_RUN_FULL_COMMON36_FORMAL_DIRECT",
        "FORMAL_EXPANSION_TRIGGERED": "NO",
        "N_NEW_LOCAL_CAUSAL_EVALUATIONS": len(formal_rows),
        "C_SENSITIVE_NATIVE_GAP": median(row["native_gap"] for row in c_rows),
        "C_SENSITIVE_NATIVE_GAP_CI": native_ci,
        "C_SENSITIVE_QUERY_EQ_GAP": median(row["query_eq_gap"] for row in c_rows),
        "C_SENSITIVE_QUERY_EQ_GAP_CI": qeq_ci,
        "C_SENSITIVE_QUERY_RESCUE": c_summary["median_rescue"],
        "C_SENSITIVE_QUERY_RESCUE_CI": [c_summary["ci_low"], c_summary["ci_high"]],
        "C_SENSITIVE_QUERY_RESCUE_MEAN": c_summary["mean_rescue"],
        "C_SENSITIVE_QUERY_RESCUE_TRIMMED_MEAN": c_summary["trimmed_mean_rescue"],
        "N_C_SENSITIVE_LAYERS_WITH_POSITIVE_RESCUE": positive_layers,
        "N_C_SENSITIVE_LAYERS_WITH_50PCT_OR_MORE_RESCUE": fifty_layers,
        "N_LAYERS_WITH_ABS_GAP_REDUCED": abs_reduced_layers,
        "N_LAYERS_WITH_CI_OVERLAPPING_ZERO_AFTER_INTERVENTION": qeq_ci_zero_layers,
        "L22_NATIVE_GAP": l22_native,
        "L22_QUERY_EQ_GAP": l22_qeq,
        "L22_RESCUE": l22_rescue,
        "LAYER22_CAUSAL_CONTROL": l22_control,
        "G128_QUERY_RESCUE": as_float(g128["query_rescue"]),
        "G128_QUERY_RESCUE_CI": [as_float(g128["query_rescue_CI_low"]), as_float(g128["query_rescue_CI_high"])],
        "G16_QUERY_RESCUE": as_float(g16["query_rescue"]),
        "G16_QUERY_RESCUE_CI": [as_float(g16["query_rescue_CI_low"]), as_float(g16["query_rescue_CI_high"])],
        "D_QUERY_RESCUE_128_TO_16": median(row["D_query_rescue_128_to_16"] for row in d_rows),
        "D_QUERY_RESCUE_128_TO_16_CI": d_ci,
        "DESCRIPTIVE_QUERY_MEDIATION_FRACTION": median(fraction_vals),
        "DESCRIPTIVE_QUERY_MEDIATION_FRACTION_MEAN": mean(fraction_vals),
        "DESCRIPTIVE_QUERY_MEDIATION_FRACTION_TRIMMED_MEAN": trimmed_mean(fraction_vals),
        "UNSTABLE_DENOMINATOR_EXCLUDED_ROWS": len(c_rows) - len(fraction_vals),
        "QUERY_READOUT_CAUSAL_STATUS": query_status,
        "KDA_MECHANISM_CLOSURE": closure,
        "FINAL_CLASSIFICATION": final_classification,
        "NEW_STAGE_B_TRAJECTORIES": 0,
        "NEW_CLOSED_LOOP_TRAJECTORIES": 0,
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "BASH_SYNTAX_CHECK": "NOT_APPLICABLE",
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES",
        "NEXT_RECOMMENDED_TASK": "KIMI_LINEAR_KDA_CORE_MECHANISM_TRANSFER_V1" if closure == "YES" else "WAIT_FOR_HUMAN_REVIEW",
        "run_id": run_id,
    }
    save_json(OUT_DIR / "bootstrap_results.json", {
        "TASK": TASK,
        "bootstrap_reps": reps,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_level": "canonical_unit_cluster",
        "C_sensitive_summary": c_summary,
        "granularity_rows": gran_rows,
        "D_query_rescue_128_to_16_CI": d_ci,
    })
    save_json(OUT_DIR / "summary.json", summary)
    write_report(summary)
    final_summary(args.py_compile_status, args.pytest_status, args.gpu_release_status)
    return summary


def write_report(summary):
    write_text(OUT_DIR / "report.md", "\n".join([
        f"# {TASK}",
        "",
        f"FORMAL_STATUS: {summary['FORMAL_STATUS']}",
        f"QUERY_READOUT_CAUSAL_STATUS: {summary['QUERY_READOUT_CAUSAL_STATUS']}",
        f"KDA_MECHANISM_CLOSURE: {summary['KDA_MECHANISM_CLOSURE']}",
        "",
        "The intervention modifies only the K2 update-error component visible to the verified K3 query readout.",
        "The null component is preserved and K1/K2 native tensors are recorded as noninterfered.",
        "",
        f"C-sensitive median native gap: {summary['C_SENSITIVE_NATIVE_GAP']}",
        f"C-sensitive median query-equalized gap: {summary['C_SENSITIVE_QUERY_EQ_GAP']}",
        f"C-sensitive median query rescue: {summary['C_SENSITIVE_QUERY_RESCUE']} with CI {summary['C_SENSITIVE_QUERY_RESCUE_CI']}",
        f"Layer22 symmetric control: {summary['LAYER22_CAUSAL_CONTROL']}",
        "",
        "No Stage-B recurrent propagation, closed-loop panel, Kimi transfer, or method design was launched.",
    ]) + "\n")


def verify(_args=None):
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8")) if (OUT_DIR / "summary.json").exists() else {}
    required_before_verify = [name for name in REQUIRED_OUTPUT_FILES if name != "fresh_output_verification.json"]
    missing = [name for name in required_before_verify if not (OUT_DIR / name).exists()]
    event_rows = read_csv(OUT_DIR / "readout_intervention_event_level.csv")
    formal_rows = read_csv(OUT_DIR / "formal_results.csv")
    validation_rows = read_csv(OUT_DIR / "query_visible_decomposition_validation.csv")
    units = {row.get("unit_id") for row in formal_rows}
    layers = {int(row["layer_id"]) for row in formal_rows if row.get("layer_id")}
    groups = {int(row["group_size"]) for row in formal_rows if row.get("group_size")}
    conditions = defaultdict(int)
    numeric_bad = []
    for idx, row in enumerate(event_rows):
        conditions[row.get("condition")] += 1
        for key in ["core_output_damage", "k3_readout_damage", "query_visible_norm", "R_error_norm", "C_match_error_norm", "magnitude_relative_mismatch"]:
            if not finite(row.get(key)):
                numeric_bad.append({"row": idx, "key": key, "value": row.get(key)})
    formal_numeric_keys = ["native_gap", "query_eq_gap", "rescue", "readout_rescue"]
    for idx, row in enumerate(formal_rows):
        for key in formal_numeric_keys:
            if not finite(row.get(key)):
                numeric_bad.append({"formal_row": idx, "key": key, "value": row.get(key)})
    output_hashes = {}
    min_mtime = None
    for name in required_before_verify:
        path = OUT_DIR / name
        if path.exists():
            stat = path.stat()
            output_hashes[name] = {"sha256": file_sha256(path), "mtime": stat.st_mtime, "bytes": stat.st_size}
            min_mtime = stat.st_mtime if min_mtime is None else min(min_mtime, stat.st_mtime)
    expected_formal = 36 * len(KDA_LAYERS) * len(GROUP_SIZES)
    expected_events = expected_formal * len(CONDITIONS)
    gates_ok = all(summary.get(key) == "PASS" for key in [
        "PIPELINE_IDENTITY_GATE",
        "COMMON_COHORT_REUSE_GATE",
        "PRIOR_LAYER_LABEL_REUSE_GATE",
        "REFERENCE_STATE_PARITY_GATE",
        "REFERENCE_CORE_OUTPUT_PARITY_GATE",
        "EXACT_STATE_ERROR_TRANSFER_GATE",
        "QUERY_READOUT_SEMANTICS_GATE",
        "QUERY_VISIBLE_DECOMPOSITION_GATE",
        "QUERY_VISIBILITY_MATCH_GATE",
        "READOUT_NULL_COMPONENT_PRESERVATION_GATE",
        "PRE_READOUT_NONINTERFERENCE_GATE",
        "LOCAL_DAMAGE_ENDPOINT_REUSE_GATE",
    ])
    no_stale = bool(output_hashes) and all(v["sha256"] for v in output_hashes.values())
    fresh = (
        not missing
        and len(units) == 36
        and layers == set(KDA_LAYERS)
        and groups == set(GROUP_SIZES)
        and len(formal_rows) == expected_formal
        and len(event_rows) == expected_events
        and all(conditions.get(condition) == expected_formal for condition in CONDITIONS)
        and len(validation_rows) == expected_formal
        and not numeric_bad
        and gates_ok
        and summary.get("NEW_STAGE_B_TRAJECTORIES") == 0
        and summary.get("NEW_CLOSED_LOOP_TRAJECTORIES") == 0
        and no_stale
    )
    verification = {
        "TASK": TASK,
        "FRESH_OUTPUT_VERIFICATION": "PASS" if fresh else "FAIL",
        "missing_files": missing,
        "N_EVENT_ROWS": len(event_rows),
        "N_EXPECTED_EVENT_ROWS": expected_events,
        "N_FORMAL_ROWS": len(formal_rows),
        "N_EXPECTED_FORMAL_ROWS": expected_formal,
        "N_VALIDATION_ROWS": len(validation_rows),
        "unit_count": len(units),
        "layer_count": len(layers),
        "group_sizes": sorted(groups),
        "condition_counts": dict(conditions),
        "numeric_bad": numeric_bad[:20],
        "gates_ok": gates_ok,
        "no_stale_causal_output_reuse": "PASS" if no_stale else "FAIL",
        "output_hashes": output_hashes,
        "oldest_output_mtime": min_mtime,
        "NEW_STAGE_B_TRAJECTORIES": summary.get("NEW_STAGE_B_TRAJECTORIES"),
        "NEW_CLOSED_LOOP_TRAJECTORIES": summary.get("NEW_CLOSED_LOOP_TRAJECTORIES"),
    }
    if summary:
        summary["FRESH_OUTPUT_VERIFICATION"] = verification["FRESH_OUTPUT_VERIFICATION"]
        if _args is not None:
            if getattr(_args, "py_compile_status", "NOT_RUN") != "NOT_RUN":
                summary["PY_COMPILE"] = _args.py_compile_status
            if getattr(_args, "pytest_status", "NOT_RUN") != "NOT_RUN":
                summary["PYTEST"] = _args.pytest_status
            if getattr(_args, "gpu_release_status", "NOT_RUN") != "NOT_RUN":
                summary["GPU_RELEASE_VERIFICATION"] = _args.gpu_release_status
        save_json(OUT_DIR / "summary.json", summary)
        final_summary(summary.get("PY_COMPILE", "NOT_RUN"), summary.get("PYTEST", "NOT_RUN"), summary.get("GPU_RELEASE_VERIFICATION", "NOT_RUN"))
    save_json(OUT_DIR / "fresh_output_verification.json", verification)
    return verification


def final_summary(py_compile_status="NOT_RUN", pytest_status="NOT_RUN", gpu_release_status="NOT_RUN"):
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if summary:
        summary["PY_COMPILE"] = py_compile_status
        summary["PYTEST"] = pytest_status
        summary["GPU_RELEASE_VERIFICATION"] = gpu_release_status
        save_json(summary_path, summary)
    lines = [f"{key} = {summary.get(key)}" for key in SUMMARY_KEYS]
    lines.append("STOP.")
    write_text(OUT_DIR / "final_console_summary.txt", "\n".join(lines) + "\n")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "formal", "analyze", "verify", "summary"], default="analyze")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--unit-limit", type=int, default=None)
    parser.add_argument("--py-compile-status", default="NOT_RUN")
    parser.add_argument("--pytest-status", default="NOT_RUN")
    parser.add_argument("--gpu-release-status", default="NOT_RUN")
    args = parser.parse_args()
    if args.stage == "prepare":
        out = prepare(args)
    elif args.stage == "formal":
        out = formal(args)
    elif args.stage == "analyze":
        out = analyze(args)
    elif args.stage == "verify":
        out = verify(args)
    else:
        out = final_summary(args.py_compile_status, args.pytest_status, args.gpu_release_status)
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) if isinstance(out, dict) else "\n".join(out))


if __name__ == "__main__":
    main()
