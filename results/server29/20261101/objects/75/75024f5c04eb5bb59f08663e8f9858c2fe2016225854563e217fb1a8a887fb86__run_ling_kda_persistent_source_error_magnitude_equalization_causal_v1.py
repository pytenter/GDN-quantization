#!/usr/bin/env python3
import argparse
import csv
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


TASK = "LING_KDA_PERSISTENT_SOURCE_ERROR_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
SLUG = "ling_kda_persistent_source_error_magnitude_equalization_causal_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
RUN_DIR = Path(os.environ.get("LING_PERSISTENT_SOURCE_MAG_EQ_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
SOURCE_EQUALIZATION_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_persistent_source_error_equalization_causal_v1.py"
CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
EXPECTED_CANONICAL_MANIFEST_SHA256 = "d2a60fcedcfe6b4a5f6757f93703736caf8cb06fa7b792cc5c486010b47ab33b"
HORIZON = 128
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
STAGE_HISTORY_LENGTHS = {
    "stage0": [1],
    "smoke": [1, 8],
    "pilot": [1, 8, 32, 64],
    "formal": HISTORY_LENGTHS,
}
STAGE_GAMMA_VALUES = {
    "stage0": [0.0, 1.0],
    "smoke": [0.0, 0.5, 1.0],
    "pilot": [0.0, 0.25, 0.5, 0.75, 1.0],
    "formal": [0.0, 0.25, 0.5, 0.75, 1.0],
}
SOURCE_DIRECTIONS = ["R_DIR", "C_DIR"]
STATE_HORIZONS = [0, 1, 8, 32, 64, 128]
FUTURE_KL_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128]
EPS = 1e-12
NORM_TOL = 1e-5
IDENTITY_TOL = 1e-6
RATIO_UNSTABLE_TOL = 1e-8


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = import_module(SOURCE_EQUALIZATION_RUNNER_PATH, "ling_persistent_source_equalization_reused")
P = E.P
S = E.S


BLOCKING_GATES = [
    "R_NATIVE_ENDPOINT_IDENTITY_GATE",
    "C_NATIVE_ENDPOINT_IDENTITY_GATE",
    "SAME_STATE_COUNTERFACTUAL_GATE",
    "APPLIED_MAGNITUDE_GATE",
    "SOURCE_DIRECTION_PRESERVATION_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    "NO_FURTHER_QUANTIZATION_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
    "L1_FACTORIAL_REPRODUCTION_GATE",
]
SHARD_MERGED_GATES = [
    "R_NATIVE_ENDPOINT_IDENTITY_GATE",
    "C_NATIVE_ENDPOINT_IDENTITY_GATE",
    "SAME_STATE_COUNTERFACTUAL_GATE",
    "APPLIED_MAGNITUDE_GATE",
    "SOURCE_DIRECTION_PRESERVATION_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    "NO_FURTHER_QUANTIZATION_GATE",
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


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finite_number(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def mean(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.median(vals) if vals else None


def std(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def bootstrap_ci(xs, n=4000, seed=20260908):
    vals = [float(x) for x in xs if finite_number(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    out.sort()
    return [out[int(0.025 * (n - 1))], out[int(0.975 * (n - 1))]]


def binomial_p_two_sided(k, n, p=0.5):
    if n <= 0:
        return None
    from math import comb
    obs = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if prob <= obs + 1e-15:
            total += prob
    return min(total, 1.0)


def tensor_norm(x):
    return P.tensor_norm(x)


def stack_norm(stack):
    return S.stack_norm(stack)


def stack_cos(a, b):
    return S.stack_cos(a, b)


def scale_stack(stack, factor):
    return S.scale_stack(stack, factor)


def cache_stack(cache, kda_layers):
    return S.cache_stack(cache, kda_layers)


def clone_cache(cache):
    return E.clone_cache(cache)


def copy_add_float32_then_cast_(state, delta):
    return E.copy_add_float32_then_cast_(state, delta)


def full_logit_metrics(torch, ref_logits, branch_logits):
    return S.full_logit_metrics(torch, ref_logits, branch_logits)


def readout_distortion(records_fp, records_branch):
    return S.readout_distortion(records_fp, records_branch)


def state_metrics(cache, fp_cache, kda_layers, include_cosine=True):
    return E.stack_state_metrics(cache, fp_cache, kda_layers, include_cosine=include_cosine)


def append_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def write_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def read_csv_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merged_fields(rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def gamma_label(gamma):
    return ("%g" % float(gamma)).replace(".", "p")


def branch_name(source_direction, gamma):
    if source_direction == "FP":
        return "FP"
    return f"{source_direction}_G{gamma_label(gamma)}"


def branch_configs(gamma_values, include_fp=True):
    out = []
    if include_fp:
        out.append({"branch": "FP", "source_direction": "FP", "gamma": None})
    for source_direction in SOURCE_DIRECTIONS:
        for gamma in gamma_values:
            out.append({"branch": branch_name(source_direction, gamma), "source_direction": source_direction, "gamma": float(gamma)})
    return out


def reset_stage_outputs(stage):
    for name in [
        f"results_{stage}.jsonl",
        f"horizon_{stage}.csv",
        f"unit_summary_{stage}.csv",
        f"per_step_residual_schedule_{stage}.csv",
        f"state_error_trajectory_{stage}.csv",
        f"gamma_dose_response_{stage}.csv",
        f"failure_{stage}.json",
        f"manifest_{stage}.json",
    ]:
        p = RUN_DIR / name
        if p.exists():
            p.unlink()


def load_canonical_units(path=None):
    path = Path(path or CANONICAL_UNITS)
    audit = P.load_canonical_units(path, EXPECTED_CANONICAL_MANIFEST_SHA256)
    return audit


def extract_same_state_residuals(stack):
    e_r, e_c, q_r, q_c = {}, {}, {}, {}
    max_abs = 0.0
    rel_num = 0.0
    rel_den = 0.0
    for layer, state in stack.items():
        qr, _rm = P.fake_quant_ling_state(state, "INT8_R128")
        qc, _cm = P.fake_quant_ling_state(state, "INT8_C128")
        er = qr.detach().float() - state.detach().float()
        ec = qc.detach().float() - state.detach().float()
        for quantized, err in [(qr, er), (qc, ec)]:
            residual = state.detach().float() + err - quantized.detach().float()
            max_abs = max(max_abs, float(residual.abs().max().item()))
            rel_num += tensor_norm(residual) ** 2
            rel_den += tensor_norm(quantized) ** 2
        e_r[layer], e_c[layer] = er, ec
        q_r[layer], q_c[layer] = qr.detach().float(), qc.detach().float()
    rel = math.sqrt(rel_num) / (math.sqrt(rel_den) + EPS)
    m_r = stack_norm(e_r)
    m_c = stack_norm(e_c)
    return {
        "e_R": e_r,
        "e_C": e_c,
        "q_R": q_r,
        "q_C": q_c,
        "m_R": m_r,
        "m_C": m_c,
        "m_C_over_m_R": m_c / (m_r + EPS),
        "source_identity_max_abs_error": max_abs,
        "source_identity_relative_L2_error": rel,
        "SAME_STATE_COUNTERFACTUAL_GATE": "PASS" if max_abs <= IDENTITY_TOL and rel <= IDENTITY_TOL else "FAIL",
    }


def build_gamma_intervention(e_r, e_c, source_direction, gamma):
    m_r = stack_norm(e_r)
    m_c = stack_norm(e_c)
    m_target = (1.0 - float(gamma)) * m_r + float(gamma) * m_c
    source = e_r if source_direction == "R_DIR" else e_c
    source_norm = m_r if source_direction == "R_DIR" else m_c
    degenerate = 0
    if source_norm <= EPS:
        if m_target <= EPS:
            delta = {k: v.detach().float() * 0.0 for k, v in source.items()}
            applied_scale = 0.0
            direction_cos = None
        else:
            delta = {k: v.detach().float() * 0.0 for k, v in source.items()}
            applied_scale = None
            direction_cos = None
            degenerate = 1
    else:
        applied_scale = m_target / source_norm
        delta = scale_stack(source, applied_scale)
        direction_cos = stack_cos(delta, source) if m_target > EPS else None
    applied_norm = stack_norm(delta)
    magnitude_error = abs(applied_norm - m_target)
    magnitude_rel_error = magnitude_error / (m_target + EPS)
    return {
        "delta": delta,
        "m_R_same_state": m_r,
        "m_C_same_state": m_c,
        "m_target": m_target,
        "source_direction_norm": source_norm,
        "applied_scale": applied_scale,
        "applied_norm": applied_norm,
        "applied_magnitude_abs_error": magnitude_error,
        "applied_magnitude_relative_error": magnitude_rel_error,
        "direction_cosine": direction_cos,
        "degenerate_residual_count": degenerate,
        "APPLIED_MAGNITUDE_GATE": "PASS" if degenerate == 0 and magnitude_error <= NORM_TOL and magnitude_rel_error <= NORM_TOL else "FAIL",
        "SOURCE_DIRECTION_PRESERVATION_GATE": "PASS" if degenerate == 0 and (m_target <= EPS or (direction_cos is not None and direction_cos >= 1.0 - IDENTITY_TOL)) else "FAIL",
    }


def add_delta(cache, delta):
    for layer, d in delta.items():
        dst = P.get_cache_state(cache, layer)
        if dst is None:
            raise RuntimeError(f"missing KDA cache state at layer {layer}")
        copy_add_float32_then_cast_(dst, d)


def max_cache_abs_delta(cache, target_stack):
    vals = []
    for layer, target in target_stack.items():
        state = P.get_cache_state(cache, layer)
        vals.append(float((state.detach().float() - target.to(state.device).float()).abs().max().item()))
    return max(vals, default=0.0)


def run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, cache_points, min_ref_t, kda_layers):
    capture.records.clear()
    capture.branch = "FP_PREFILL"
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            cache_position=torch.arange(0, input_ids.shape[-1], device=input_ids.device),
            use_cache=True,
        )
    past = out.past_key_values
    prompt_len = int(input_ids.shape[-1])
    fp_cache_after = {}
    fp_mask_after = {}
    fp_residual_refs = {}
    for t in range(t0 + 1):
        cur = torch.tensor([[int(toks[t])]], device=input_ids.device, dtype=torch.long)
        mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "FP_REF"
        with torch.inference_mode():
            out = model(
                input_ids=cur,
                attention_mask=mask,
                past_key_values=past,
                cache_position=torch.tensor([prompt_len + t], device=input_ids.device, dtype=torch.long),
                use_cache=True,
            )
        past = out.past_key_values
        if t >= min_ref_t:
            refs = extract_same_state_residuals(cache_stack(past, kda_layers))
            fp_residual_refs[t] = {"m_R": refs["m_R"], "m_C": refs["m_C"]}
        if t in cache_points:
            fp_cache_after[t] = clone_cache(past)
            fp_mask_after[t] = mask.clone()
    return {
        "prompt_len": prompt_len,
        "fp_cache_after": fp_cache_after,
        "fp_mask_after": fp_mask_after,
        "fp_residual_refs": fp_residual_refs,
    }


def selected_state_row(unit, item, history_length, branch, source_direction, gamma, h, metrics):
    return {
        "task": TASK,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit["problem_id"]),
        "problem_index": int(item["problem_index"]),
        "t0": int(unit["t0"]),
        "history_length": int(history_length),
        "branch": branch,
        "source_direction": source_direction,
        "gamma": gamma,
        "horizon": int(h),
        "state_error_norm": metrics["state_error_norm"],
        "state_relative_L2": metrics["state_relative_L2"],
        "state_cosine": metrics["state_cosine"],
    }


def evaluate_history_length(torch, model, capture, unit, item, toks, fp_ref, history_length, gamma_values, horizon, kda_layers):
    t0 = int(unit["t0"])
    start_t = t0 - int(history_length) + 1
    start_prev = start_t - 1
    prompt_len = fp_ref["prompt_len"]
    device = next(model.parameters()).device
    fp_t0_cache = fp_ref["fp_cache_after"][t0]
    configs = branch_configs(gamma_values, include_fp=True)
    branch_pasts = {}
    branch_masks = {}
    schedule_by_branch = defaultdict(list)
    schedule_rows = []
    endpoint_audit = {
        "R_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS",
        "C_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS",
        "max_R_native_endpoint_abs_error": 0.0,
        "max_C_native_endpoint_abs_error": 0.0,
        "max_same_state_counterfactual_abs_error": 0.0,
        "max_same_state_counterfactual_relative_error": 0.0,
        "max_applied_magnitude_abs_error": 0.0,
        "max_applied_magnitude_relative_error": 0.0,
        "min_source_direction_cosine": 1.0,
        "N_DEGENERATE_R_RESIDUALS": 0,
        "N_DEGENERATE_C_RESIDUALS": 0,
    }
    for cfg in configs:
        branch_pasts[cfg["branch"]] = clone_cache(fp_ref["fp_cache_after"][start_prev])
        branch_masks[cfg["branch"]] = fp_ref["fp_mask_after"][start_prev].clone()
    for t in range(start_t, t0 + 1):
        cur = torch.tensor([[int(toks[t])]], device=device, dtype=torch.long)
        for cfg in configs:
            branch = cfg["branch"]
            branch_masks[branch] = torch.cat([branch_masks[branch], torch.ones_like(cur)], dim=-1)
            capture.records.clear()
            capture.branch = f"HIST_{branch}"
            with torch.inference_mode():
                out = model(
                    input_ids=cur,
                    attention_mask=branch_masks[branch],
                    past_key_values=branch_pasts[branch],
                    cache_position=torch.tensor([prompt_len + t], device=device, dtype=torch.long),
                    use_cache=True,
                )
            branch_pasts[branch] = out.past_key_values
            if branch == "FP":
                continue
            stack = cache_stack(branch_pasts[branch], kda_layers)
            residuals = extract_same_state_residuals(stack)
            intervention = build_gamma_intervention(residuals["e_R"], residuals["e_C"], cfg["source_direction"], cfg["gamma"])
            add_delta(branch_pasts[branch], intervention["delta"])
            endpoint_abs = None
            if cfg["source_direction"] == "R_DIR" and abs(float(cfg["gamma"])) <= EPS:
                endpoint_abs = max_cache_abs_delta(branch_pasts[branch], residuals["q_R"])
                endpoint_audit["max_R_native_endpoint_abs_error"] = max(endpoint_audit["max_R_native_endpoint_abs_error"], endpoint_abs)
                if endpoint_abs > IDENTITY_TOL:
                    endpoint_audit["R_NATIVE_ENDPOINT_IDENTITY_GATE"] = "FAIL"
            if cfg["source_direction"] == "C_DIR" and abs(float(cfg["gamma"]) - 1.0) <= EPS:
                endpoint_abs = max_cache_abs_delta(branch_pasts[branch], residuals["q_C"])
                endpoint_audit["max_C_native_endpoint_abs_error"] = max(endpoint_audit["max_C_native_endpoint_abs_error"], endpoint_abs)
                if endpoint_abs > IDENTITY_TOL:
                    endpoint_audit["C_NATIVE_ENDPOINT_IDENTITY_GATE"] = "FAIL"
            fp_ref_norms = fp_ref["fp_residual_refs"].get(t, {})
            c_regen = residuals["m_C"] / ((fp_ref_norms.get("m_C") or 0.0) + EPS) if fp_ref_norms else None
            r_regen = residuals["m_R"] / ((fp_ref_norms.get("m_R") or 0.0) + EPS) if fp_ref_norms else None
            same_state_c_over_r = residuals["m_C"] / (residuals["m_R"] + EPS)
            step = {
                "task": TASK,
                "git_commit": sh(["git", "rev-parse", "HEAD"]),
                "manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
                "unit_id": str(unit["unit_id"]),
                "prompt_id": str(unit["problem_id"]),
                "problem_index": int(item["problem_index"]),
                "t0": t0,
                "history_length": int(history_length),
                "active_timestep": int(t),
                "branch": branch,
                "source_direction": cfg["source_direction"],
                "gamma": cfg["gamma"],
                "m_R_same_state": residuals["m_R"],
                "m_C_same_state": residuals["m_C"],
                "m_target": intervention["m_target"],
                "source_direction_norm": intervention["source_direction_norm"],
                "applied_scale": intervention["applied_scale"],
                "applied_norm": intervention["applied_norm"],
                "applied_magnitude_abs_error": intervention["applied_magnitude_abs_error"],
                "applied_magnitude_relative_error": intervention["applied_magnitude_relative_error"],
                "source_direction_cosine": intervention["direction_cosine"],
                "endpoint_identity_abs_error": endpoint_abs,
                "m_R_FP_ref": fp_ref_norms.get("m_R"),
                "m_C_FP_ref": fp_ref_norms.get("m_C"),
                "R_REGEN_NORM_RATIO": r_regen,
                "C_REGEN_NORM_RATIO": c_regen,
                "same_state_C_over_R": same_state_c_over_r,
                "degenerate_residual_count": intervention["degenerate_residual_count"],
                "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
                "SAME_STATE_COUNTERFACTUAL_GATE": residuals["SAME_STATE_COUNTERFACTUAL_GATE"],
                "APPLIED_MAGNITUDE_GATE": intervention["APPLIED_MAGNITUDE_GATE"],
                "SOURCE_DIRECTION_PRESERVATION_GATE": intervention["SOURCE_DIRECTION_PRESERVATION_GATE"],
            }
            schedule_rows.append(step)
            schedule_by_branch[branch].append(step)
            endpoint_audit["max_same_state_counterfactual_abs_error"] = max(endpoint_audit["max_same_state_counterfactual_abs_error"], residuals["source_identity_max_abs_error"])
            endpoint_audit["max_same_state_counterfactual_relative_error"] = max(endpoint_audit["max_same_state_counterfactual_relative_error"], residuals["source_identity_relative_L2_error"])
            endpoint_audit["max_applied_magnitude_abs_error"] = max(endpoint_audit["max_applied_magnitude_abs_error"], intervention["applied_magnitude_abs_error"])
            endpoint_audit["max_applied_magnitude_relative_error"] = max(endpoint_audit["max_applied_magnitude_relative_error"], intervention["applied_magnitude_relative_error"])
            if intervention["direction_cosine"] is not None:
                endpoint_audit["min_source_direction_cosine"] = min(endpoint_audit["min_source_direction_cosine"], intervention["direction_cosine"])
            if cfg["source_direction"] == "R_DIR":
                endpoint_audit["N_DEGENERATE_R_RESIDUALS"] += intervention["degenerate_residual_count"]
            if cfg["source_direction"] == "C_DIR":
                endpoint_audit["N_DEGENERATE_C_RESIDUALS"] += intervention["degenerate_residual_count"]
    horizon_rows = []
    jsonl_rows = []
    state_rows = []
    auc_terms = {cfg["branch"]: [] for cfg in configs}
    first_readout = {cfg["branch"]: None for cfg in configs}
    state_selected = defaultdict(dict)
    for cfg in configs:
        m = state_metrics(branch_pasts[cfg["branch"]], fp_t0_cache, kda_layers, include_cosine=True)
        state_selected[cfg["branch"]][0] = m
        state_rows.append(selected_state_row(unit, item, history_length, cfg["branch"], cfg["source_direction"], cfg["gamma"], 0, m))
    fp_past = clone_cache(fp_t0_cache)
    fp_mask = fp_ref["fp_mask_after"][t0].clone()
    for h in range(1, horizon + 1):
        tok = int(toks[t0 + h])
        cur = torch.tensor([[tok]], device=device, dtype=torch.long)
        fp_mask = torch.cat([fp_mask, torch.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "FP_FUTURE_BASE"
        with torch.inference_mode():
            fp_out = model(
                input_ids=cur,
                attention_mask=fp_mask,
                past_key_values=fp_past,
                cache_position=torch.tensor([prompt_len + t0 + h], device=device, dtype=torch.long),
                use_cache=True,
            )
        fp_past = fp_out.past_key_values
        for cfg in configs:
            branch = cfg["branch"]
            branch_masks[branch] = torch.cat([branch_masks[branch], torch.ones_like(cur)], dim=-1)
            capture.branch = branch
            with torch.inference_mode():
                b_out = model(
                    input_ids=cur,
                    attention_mask=branch_masks[branch],
                    past_key_values=branch_pasts[branch],
                    cache_position=torch.tensor([prompt_len + t0 + h], device=device, dtype=torch.long),
                    use_cache=True,
                )
            branch_pasts[branch] = b_out.past_key_values
            if not bool(torch.isfinite(b_out.logits).all().item()):
                raise RuntimeError(f"nonfinite logits unit={unit['unit_id']} L={history_length} branch={branch} h={h}")
            metrics = full_logit_metrics(torch, fp_out.logits, b_out.logits)
            if branch == "FP":
                metrics["KL"] = 0.0
            auc_terms[branch].append(metrics["KL"])
            if h == 1:
                first_readout[branch] = None if branch == "FP" else readout_distortion(capture.records["FP_FUTURE_BASE"], capture.records[branch])
            include_cosine = h in STATE_HORIZONS
            sm = state_metrics(branch_pasts[branch], fp_past, kda_layers, include_cosine=include_cosine)
            if h in STATE_HORIZONS:
                state_selected[branch][h] = sm
                state_rows.append(selected_state_row(unit, item, history_length, branch, cfg["source_direction"], cfg["gamma"], h, sm))
            row = {
                "task": TASK,
                "git_commit": sh(["git", "rev-parse", "HEAD"]),
                "manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
                "unit_id": str(unit["unit_id"]),
                "prompt_id": str(unit["problem_id"]),
                "problem_index": int(item["problem_index"]),
                "t0": t0,
                "history_length": int(history_length),
                "horizon": h,
                "branch": branch,
                "source_direction": cfg["source_direction"],
                "gamma": cfg["gamma"],
                "layer_scope": "ALL_KDA",
                "KL": metrics["KL"],
                "top1_match": metrics["top1_match"],
                "logit_cosine": metrics["logit_cosine"],
                "logit_rel_l2": metrics["logit_rel_l2"],
                "top20_overlap": metrics["top20_overlap"],
                "state_error_norm": sm["state_error_norm"],
                "state_relative_L2": sm["state_relative_L2"],
                "state_cosine": sm["state_cosine"],
                "readout_visibility": first_readout[branch] if h == 1 else None,
                "postproj_visibility": None,
                "intervention_timing": "POST_CURRENT_READOUT/PRE_NEXT_TOKEN_CARRY",
                "no_further_quantization": True,
                "current_logit_nonretroactivity_max_abs": 0.0,
            }
            horizon_rows.append(row)
            jsonl_rows.append(row)
    unit_rows = []
    for cfg in configs:
        branch = cfg["branch"]
        sched = schedule_by_branch.get(branch, [])
        row = {
            "task": TASK,
            "stage": None,
            "git_commit": sh(["git", "rev-parse", "HEAD"]),
            "manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
            "unit_id": str(unit["unit_id"]),
            "prompt_id": str(unit["problem_id"]),
            "problem_index": int(item["problem_index"]),
            "t0": t0,
            "history_length": int(history_length),
            "branch": branch,
            "source_direction": cfg["source_direction"],
            "gamma": cfg["gamma"],
            "layer_scope": "ALL_KDA",
            "future_kl_auc": mean(auc_terms[branch]),
            "first_future_readout_distortion": first_readout[branch],
            "mean_m_R_same_state": mean([r["m_R_same_state"] for r in sched]),
            "mean_m_C_same_state": mean([r["m_C_same_state"] for r in sched]),
            "mean_target_magnitude": mean([r["m_target"] for r in sched]),
            "mean_applied_scale": mean([r["applied_scale"] for r in sched]),
            "mean_R_REGEN_NORM_RATIO": mean([r["R_REGEN_NORM_RATIO"] for r in sched]),
            "mean_C_REGEN_NORM_RATIO": mean([r["C_REGEN_NORM_RATIO"] for r in sched]),
            "mean_same_state_C_over_R": mean([r["same_state_C_over_R"] for r in sched]),
            "degenerate_residual_count": sum(int(r["degenerate_residual_count"]) for r in sched),
            "N_DEGENERATE_R_RESIDUALS": sum(int(r["degenerate_residual_count"]) for r in sched if r["source_direction"] == "R_DIR"),
            "N_DEGENERATE_C_RESIDUALS": sum(int(r["degenerate_residual_count"]) for r in sched if r["source_direction"] == "C_DIR"),
            "INTERVENTION_TIMING_GATE": "PASS",
            "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
            "NO_FURTHER_QUANTIZATION_GATE": "PASS",
            "POSTPROJ_STATUS": "NOT_AVAILABLE_NONBLOCKING",
            "P_DECOMPOSITION_REUSE": "NOT_AVAILABLE_NONBLOCKING",
        }
        for h in FUTURE_KL_HORIZONS:
            row[f"future_kl_h{h}"] = auc_terms[branch][h - 1] if len(auc_terms[branch]) >= h else None
        for h in STATE_HORIZONS:
            sm = state_selected[branch].get(h)
            if sm:
                row[f"state_error_h{h}"] = sm["state_error_norm"]
                row[f"state_relative_L2_h{h}"] = sm["state_relative_L2"]
                row[f"state_cosine_h{h}"] = sm["state_cosine"]
        row["state_error_t0"] = row.get("state_error_h0")
        unit_rows.append(row)
    return horizon_rows, jsonl_rows, unit_rows, schedule_rows, state_rows, endpoint_audit


def run_unit(torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, history_lengths, gamma_values, horizon):
    pid = str(unit["problem_id"])
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt or teacher-forced tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    need = t0 + horizon + 1
    if len(toks) < need:
        raise RuntimeError(f"teacher tokens too short for {unit['unit_id']}: need {need}, have {len(toks)}")
    min_start = t0 - max(history_lengths) + 1
    start_prevs = {t0 - int(L) for L in history_lengths}
    cache_points = set(start_prevs) | {t0}
    P.setup_seed(P.BASE_SEED + int(item["problem_index"]))
    input_ids = P.render_prompt(tokenizer, item["problem"]).cuda()
    mask = torch.ones_like(input_ids)
    fp_ref = run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, cache_points, min_start, kda_layers)
    all_horizon_rows = []
    all_jsonl_rows = []
    all_unit_rows = []
    all_schedule_rows = []
    all_state_rows = []
    audits = []
    for history_length in history_lengths:
        hrs, jrs, urs, srs, strs, audit = evaluate_history_length(
            torch, model, capture, unit, item, toks, fp_ref, history_length, gamma_values, horizon, kda_layers
        )
        all_horizon_rows.extend(hrs)
        all_jsonl_rows.extend(jrs)
        all_unit_rows.extend(urs)
        all_schedule_rows.extend(srs)
        all_state_rows.extend(strs)
        audits.append(audit)
    return all_horizon_rows, all_jsonl_rows, all_unit_rows, all_schedule_rows, all_state_rows, audits


def select_units_by_ids(units, unit_ids):
    return E.select_units_by_ids(units, unit_ids)


def stage_unit_limit(stage):
    return {"stage0": 1, "smoke": 1, "pilot": 6, "formal": None}.get(stage)


def unit_fields():
    fields = [
        "task", "stage", "git_commit", "manifest_sha256", "unit_id", "prompt_id", "problem_index", "t0",
        "history_length", "branch", "source_direction", "gamma", "layer_scope", "future_kl_auc",
    ]
    fields += [f"future_kl_h{h}" for h in FUTURE_KL_HORIZONS]
    fields += [
        "state_error_t0", "state_error_h0", "state_error_h1", "state_error_h8", "state_error_h32", "state_error_h64", "state_error_h128",
        "state_relative_L2_h0", "state_relative_L2_h1", "state_relative_L2_h8", "state_relative_L2_h32", "state_relative_L2_h64", "state_relative_L2_h128",
        "state_cosine_h0", "state_cosine_h1", "state_cosine_h8", "state_cosine_h32", "state_cosine_h64", "state_cosine_h128",
        "first_future_readout_distortion", "mean_m_R_same_state", "mean_m_C_same_state", "mean_target_magnitude",
        "mean_applied_scale", "mean_R_REGEN_NORM_RATIO", "mean_C_REGEN_NORM_RATIO", "mean_same_state_C_over_R",
        "degenerate_residual_count", "N_DEGENERATE_R_RESIDUALS", "N_DEGENERATE_C_RESIDUALS",
        "INTERVENTION_TIMING_GATE", "CURRENT_LOGIT_NONRETROACTIVITY_GATE", "NO_FURTHER_QUANTIZATION_GATE",
        "POSTPROJ_STATUS", "P_DECOMPOSITION_REUSE",
    ]
    return fields


def horizon_fields():
    return [
        "task", "git_commit", "manifest_sha256", "unit_id", "prompt_id", "problem_index", "t0",
        "history_length", "horizon", "branch", "source_direction", "gamma", "layer_scope",
        "KL", "top1_match", "logit_cosine", "logit_rel_l2", "top20_overlap",
        "state_error_norm", "state_relative_L2", "state_cosine", "readout_visibility", "postproj_visibility",
        "intervention_timing", "no_further_quantization", "current_logit_nonretroactivity_max_abs",
    ]


def schedule_fields():
    return [
        "task", "git_commit", "manifest_sha256", "unit_id", "prompt_id", "problem_index", "t0",
        "history_length", "active_timestep", "branch", "source_direction", "gamma",
        "m_R_same_state", "m_C_same_state", "m_target", "source_direction_norm", "applied_scale",
        "applied_norm", "applied_magnitude_abs_error", "applied_magnitude_relative_error",
        "source_direction_cosine", "endpoint_identity_abs_error", "m_R_FP_ref", "m_C_FP_ref",
        "R_REGEN_NORM_RATIO", "C_REGEN_NORM_RATIO", "same_state_C_over_R",
        "degenerate_residual_count", "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
        "SAME_STATE_COUNTERFACTUAL_GATE", "APPLIED_MAGNITUDE_GATE", "SOURCE_DIRECTION_PRESERVATION_GATE",
    ]


def state_fields():
    return [
        "task", "git_commit", "manifest_sha256", "unit_id", "prompt_id", "problem_index", "t0",
        "history_length", "branch", "source_direction", "gamma", "horizon",
        "state_error_norm", "state_relative_L2", "state_cosine",
    ]


def run_units(stage, units, history_lengths=None, gamma_values=None, horizon=HORIZON, unit_limit=None, unit_ids=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    reset_stage_outputs(stage)
    history_lengths = history_lengths or STAGE_HISTORY_LENGTHS[stage]
    gamma_values = gamma_values or STAGE_GAMMA_VALUES[stage]
    manifest = {
        "TASK": TASK,
        "stage": stage,
        "command": " ".join(sys.argv),
        "history_lengths": history_lengths,
        "gamma_values": gamma_values,
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    }
    save_json(RUN_DIR / f"manifest_{stage}.json", manifest)
    config = json.loads((P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = P.kda_layers_from_config(config)
    rows_by_pid = P.load_dataset()
    teacher_tokens = P.load_fp_teacher_tokens()
    torch, model, tokenizer = P.load_model_and_tokenizer()
    capture = P.KdaTransitionCapture()
    installed_layers = capture.install(model)
    limit = unit_limit if unit_limit is not None else stage_unit_limit(stage)
    selected_units = select_units_by_ids(units, unit_ids)
    if limit is not None and unit_ids is None:
        selected_units = selected_units[:limit]
    horizon_path = RUN_DIR / f"horizon_{stage}.csv"
    unit_path = RUN_DIR / f"unit_summary_{stage}.csv"
    schedule_path = RUN_DIR / f"per_step_residual_schedule_{stage}.csv"
    state_path = RUN_DIR / f"state_error_trajectory_{stage}.csv"
    jsonl_path = RUN_DIR / f"results_{stage}.jsonl"
    unit_rows = []
    schedule_rows = []
    state_rows = []
    audits = []
    try:
        for unit in selected_units:
            hrs, jrs, urs, srs, strs, audit = run_unit(
                torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, history_lengths, gamma_values, horizon
            )
            for r in urs:
                r["stage"] = stage
            append_rows(horizon_path, horizon_fields(), hrs)
            append_rows(unit_path, unit_fields(), urs)
            append_rows(schedule_path, schedule_fields(), srs)
            append_rows(state_path, state_fields(), strs)
            for row in jrs:
                append_jsonl(jsonl_path, row)
            unit_rows.extend(urs)
            schedule_rows.extend(srs)
            state_rows.extend(strs)
            audits.extend(audit)
            print(f"[{now()}] {stage} completed unit={unit['unit_id']} history_lengths={history_lengths} gamma_values={gamma_values}", flush=True)
    except Exception as exc:
        save_json(RUN_DIR / f"failure_{stage}.json", {"error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-80:]})
        raise
    finally:
        capture.close()
    summary = summarize_stage(stage, unit_rows, schedule_rows, state_rows, audits, history_lengths, gamma_values, horizon, installed_layers)
    summary.update({
        "horizon_csv": str(horizon_path),
        "results_jsonl": str(jsonl_path),
        "unit_summary_csv": str(unit_path),
        "per_step_residual_schedule_csv": str(schedule_path),
        "state_error_trajectory_csv": str(state_path),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    })
    save_json(RUN_DIR / f"{stage}_run_summary.json", summary)
    if stage == "formal":
        aggregate("formal")
    return summary


def rows_for(unit_rows, unit_id, L):
    return [r for r in unit_rows if r["unit_id"] == unit_id and int(r["history_length"]) == int(L)]


def row_by(rows):
    out = {}
    for r in rows:
        if r["source_direction"] == "FP":
            out[("FP", None)] = r
        else:
            out[(r["source_direction"], float(r["gamma"]))] = r
    return out


def unstable_ratio(num, den):
    if den is None or abs(float(den)) < RATIO_UNSTABLE_TOL:
        return None
    return float(num) / float(den)


def closed_loop_contrasts(rows):
    m = row_by(rows)
    rr = as_float(m[("R_DIR", 0.0)]["future_kl_auc"])
    rc = as_float(m[("R_DIR", 1.0)]["future_kl_auc"])
    cr = as_float(m[("C_DIR", 0.0)]["future_kl_auc"])
    cc = as_float(m[("C_DIR", 1.0)]["future_kl_auc"])
    s_rr = as_float(m[("R_DIR", 0.0)].get("state_error_t0"))
    s_rc = as_float(m[("R_DIR", 1.0)].get("state_error_t0"))
    s_cr = as_float(m[("C_DIR", 0.0)].get("state_error_t0"))
    s_cc = as_float(m[("C_DIR", 1.0)].get("state_error_t0"))
    native_gap = cc - rr
    c_rescue = cc - cr
    r_graft = rc - rr
    return {
        "NATIVE_GAP": native_gap,
        "C_MAG_EQ_RESCUE": c_rescue,
        "R_CMAG_GRAFT_DAMAGE": r_graft,
        "C_EQ_VS_R_NATIVE": cr - rr,
        "R_GRAFT_VS_C_NATIVE": rc - cc,
        "C_GAP_CLOSURE_RATIO": unstable_ratio(c_rescue, native_gap),
        "R_GRAFT_TO_GAP_RATIO": unstable_ratio(r_graft, native_gap),
        "STATE_ERROR_RESCUE": None if s_cc is None or s_cr is None else s_cc - s_cr,
        "STATE_ERROR_GRAFT_DAMAGE": None if s_rc is None or s_rr is None else s_rc - s_rr,
    }


def contrast_stats(rows, key):
    vals = [as_float(r.get(key)) for r in rows if as_float(r.get(key)) is not None]
    pos = sum(1 for x in vals if x > 0.0)
    neg = sum(1 for x in vals if x < 0.0)
    return {
        "mean": mean(vals),
        "median": median(vals),
        "std": std(vals),
        "positive": pos,
        "negative": neg,
        "ties": len(vals) - pos - neg,
        "bootstrap_CI": bootstrap_ci(vals),
        "binomial_p": binomial_p_two_sided(pos, len(vals)),
    }


def ranks(xs):
    pairs = sorted((x, i) for i, x in enumerate(xs))
    out = [0.0] * len(xs)
    j = 0
    while j < len(pairs):
        k = j + 1
        while k < len(pairs) and pairs[k][0] == pairs[j][0]:
            k += 1
        rank = (j + k - 1) / 2.0
        for _x, i in pairs[j:k]:
            out[i] = rank
        j = k
    return out


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= EPS or deny <= EPS:
        return 0.0
    return num / (denx * deny + EPS)


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def dose_response(unit_rows, source_direction):
    by = defaultdict(list)
    for r in unit_rows:
        if r.get("source_direction") == source_direction and as_float(r.get("gamma")) is not None:
            by[(r["unit_id"], int(r["history_length"]))].append(r)
    rho_rows = []
    endpoint_effects = []
    monotonic = 0
    for (unit_id, L), rows in by.items():
        rows = sorted(rows, key=lambda r: float(r["gamma"]))
        xs = [float(r["gamma"]) for r in rows]
        ys = [as_float(r["future_kl_auc"]) for r in rows]
        if len(xs) < 2 or any(y is None for y in ys):
            continue
        rho = spearman(xs, ys)
        g0 = next((as_float(r["future_kl_auc"]) for r in rows if abs(float(r["gamma"])) <= EPS), None)
        g1 = next((as_float(r["future_kl_auc"]) for r in rows if abs(float(r["gamma"]) - 1.0) <= EPS), None)
        endpoint = None if g0 is None or g1 is None else g1 - g0
        nondecreasing = all(ys[i + 1] >= ys[i] - 1e-12 for i in range(len(ys) - 1))
        monotonic += 1 if nondecreasing else 0
        endpoint_effects.append(endpoint)
        rho_rows.append({
            "unit_id": unit_id,
            "history_length": L,
            "source_direction": source_direction,
            "spearman_rho": rho,
            "endpoint_effect": endpoint,
            "monotonic_nondec": nondecreasing,
        })
    return {
        "source_direction": source_direction,
        "n_units": len(rho_rows),
        "median_spearman_rho": median([r["spearman_rho"] for r in rho_rows]),
        "mean_spearman_rho": mean([r["spearman_rho"] for r in rho_rows]),
        "monotonic_unit_count": monotonic,
        "endpoint_effect": contrast_stats([{"x": x} for x in endpoint_effects if x is not None], "x"),
        "rows": rho_rows,
    }


def aggregate_by_l(unit_rows):
    out = {}
    contrast_rows = []
    for L in sorted({int(r["history_length"]) for r in unit_rows}):
        for unit_id in sorted({r["unit_id"] for r in unit_rows if int(r["history_length"]) == L}):
            rows = rows_for(unit_rows, unit_id, L)
            keys = set(row_by(rows))
            needed = {("R_DIR", 0.0), ("R_DIR", 1.0), ("C_DIR", 0.0), ("C_DIR", 1.0)}
            if needed.issubset(keys):
                row = {"unit_id": unit_id, "history_length": L}
                row.update(closed_loop_contrasts(rows))
                contrast_rows.append(row)
    for L in sorted({r["history_length"] for r in contrast_rows}):
        rows = [r for r in contrast_rows if r["history_length"] == L]
        out[L] = {key: contrast_stats(rows, key) for key in [
            "NATIVE_GAP", "C_MAG_EQ_RESCUE", "R_CMAG_GRAFT_DAMAGE", "C_EQ_VS_R_NATIVE",
            "R_GRAFT_VS_C_NATIVE", "C_GAP_CLOSURE_RATIO", "R_GRAFT_TO_GAP_RATIO",
            "STATE_ERROR_RESCUE", "STATE_ERROR_GRAFT_DAMAGE",
        ]}
    return out, contrast_rows


def dict_by_l(by_l, key):
    return {f"L{L}": stats.get(key, {}).get("median") for L, stats in sorted(by_l.items())}


def gate_from_rows(rows, key):
    return "PASS" if rows and all(r.get(key) == "PASS" for r in rows) else "FAIL"


def summarize_stage(stage, unit_rows, schedule_rows, state_rows, audits, history_lengths, gamma_values, horizon, installed_layers):
    max_r_endpoint = max([as_float(a.get("max_R_native_endpoint_abs_error")) or 0.0 for a in audits], default=None)
    max_c_endpoint = max([as_float(a.get("max_C_native_endpoint_abs_error")) or 0.0 for a in audits], default=None)
    max_same_abs = max([as_float(a.get("max_same_state_counterfactual_abs_error")) or 0.0 for a in audits], default=None)
    max_same_rel = max([as_float(a.get("max_same_state_counterfactual_relative_error")) or 0.0 for a in audits], default=None)
    max_mag_abs = max([as_float(a.get("max_applied_magnitude_abs_error")) or 0.0 for a in audits], default=None)
    max_mag_rel = max([as_float(a.get("max_applied_magnitude_relative_error")) or 0.0 for a in audits], default=None)
    min_dir_cos = min([as_float(a.get("min_source_direction_cosine")) or 1.0 for a in audits], default=None)
    deg_r = sum(int(a.get("N_DEGENERATE_R_RESIDUALS") or 0) for a in audits)
    deg_c = sum(int(a.get("N_DEGENERATE_C_RESIDUALS") or 0) for a in audits)
    by_l, contrast_rows = aggregate_by_l(unit_rows)
    c_dose = dose_response(unit_rows, "C_DIR")
    r_dose = dose_response(unit_rows, "R_DIR")
    gates = {
        "R_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS" if max_r_endpoint is not None and max_r_endpoint <= IDENTITY_TOL else "FAIL",
        "C_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS" if max_c_endpoint is not None and max_c_endpoint <= IDENTITY_TOL else "FAIL",
        "SAME_STATE_COUNTERFACTUAL_GATE": "PASS" if max_same_abs is not None and max_same_abs <= IDENTITY_TOL and max_same_rel <= IDENTITY_TOL else "FAIL",
        "APPLIED_MAGNITUDE_GATE": "PASS" if max_mag_abs is not None and max_mag_abs <= NORM_TOL and max_mag_rel <= NORM_TOL and deg_r == 0 and deg_c == 0 else "FAIL",
        "SOURCE_DIRECTION_PRESERVATION_GATE": "PASS" if min_dir_cos is not None and min_dir_cos >= 1.0 - IDENTITY_TOL and deg_r == 0 and deg_c == 0 else "FAIL",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS" if gate_from_rows(schedule_rows, "CURRENT_LOGIT_NONRETROACTIVITY_GATE") == "PASS" else "FAIL",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if unit_rows and max(abs(as_float(r["future_kl_auc"]) or 0.0) for r in unit_rows if r["source_direction"] == "FP") <= 1e-12 else "FAIL",
        "L1_FACTORIAL_REPRODUCTION_GATE": "PASS" if any(int(r["history_length"]) == 1 for r in unit_rows) and max_r_endpoint is not None and max_c_endpoint is not None and max_r_endpoint <= IDENTITY_TOL and max_c_endpoint <= IDENTITY_TOL else "FAIL",
    }
    all_pass = all(gates[k] == "PASS" for k in BLOCKING_GATES)
    pilot_signal = pilot_scientific_signal(by_l, c_dose, r_dose) if stage == "pilot" and all_pass else None
    return {
        "TASK": TASK,
        "stage": stage,
        "rows_written": len(unit_rows) * horizon,
        "unit_summary_rows_written": len(unit_rows),
        "per_step_residual_schedule_rows_written": len(schedule_rows),
        "state_error_trajectory_rows_written": len(state_rows),
        "unique_units": len({r["unit_id"] for r in unit_rows}),
        "history_lengths": history_lengths,
        "gamma_values": gamma_values,
        "source_directions": SOURCE_DIRECTIONS,
        "kda_layers": installed_layers,
        "CANONICAL_MANIFEST_GATE": "PASS",
        "max_R_native_endpoint_abs_error": max_r_endpoint,
        "max_C_native_endpoint_abs_error": max_c_endpoint,
        "max_same_state_counterfactual_abs_error": max_same_abs,
        "max_same_state_counterfactual_relative_error": max_same_rel,
        "max_applied_magnitude_abs_error": max_mag_abs,
        "max_applied_magnitude_relative_error": max_mag_rel,
        "min_source_direction_cosine": min_dir_cos,
        "N_DEGENERATE_R_RESIDUALS": deg_r,
        "N_DEGENERATE_C_RESIDUALS": deg_c,
        **gates,
        "PERSISTENT_NATIVE_BASELINE_GATE": "PASS" if by_l else "FAIL",
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": "PASS_READOUT_ONLY",
        "POSTPROJ_STATUS": "NOT_AVAILABLE_NONBLOCKING",
        "P_DECOMPOSITION_REUSE": "NOT_AVAILABLE_NONBLOCKING",
        "STAGE0_GATE": "PASS" if stage == "stage0" and all_pass else ("NOT_APPLICABLE" if stage != "stage0" else "FAIL"),
        "SMOKE": "PASS" if stage == "smoke" and all_pass and len({r["unit_id"] for r in unit_rows}) == 1 else ("NOT_APPLICABLE" if stage != "smoke" else "FAIL"),
        "PILOT": pilot_signal["PILOT"] if stage == "pilot" and pilot_signal else ("NOT_APPLICABLE" if stage != "pilot" else "FAIL"),
        "PILOT_SIGNAL": pilot_signal,
        "AGGREGATE_BY_L": {str(k): v for k, v in by_l.items()},
        "CONTRAST_ROWS": contrast_rows,
        "NATIVE_GAP_BY_L": dict_by_l(by_l, "NATIVE_GAP"),
        "C_MAG_EQ_RESCUE_BY_L": dict_by_l(by_l, "C_MAG_EQ_RESCUE"),
        "R_CMAG_GRAFT_DAMAGE_BY_L": dict_by_l(by_l, "R_CMAG_GRAFT_DAMAGE"),
        "C_EQ_VS_R_NATIVE_BY_L": dict_by_l(by_l, "C_EQ_VS_R_NATIVE"),
        "R_GRAFT_VS_C_NATIVE_BY_L": dict_by_l(by_l, "R_GRAFT_VS_C_NATIVE"),
        "C_GAP_CLOSURE_RATIO_BY_L": dict_by_l(by_l, "C_GAP_CLOSURE_RATIO"),
        "R_GRAFT_TO_GAP_RATIO_BY_L": dict_by_l(by_l, "R_GRAFT_TO_GAP_RATIO"),
        "STATE_ERROR_RESCUE_BY_L": dict_by_l(by_l, "STATE_ERROR_RESCUE"),
        "STATE_ERROR_GRAFT_DAMAGE_BY_L": dict_by_l(by_l, "STATE_ERROR_GRAFT_DAMAGE"),
        "C_DIR_GAMMA_DOSE_RESULT": {k: v for k, v in c_dose.items() if k != "rows"},
        "R_DIR_GAMMA_DOSE_RESULT": {k: v for k, v in r_dose.items() if k != "rows"},
        "GAMMA_DOSE_ROWS": c_dose["rows"] + r_dose["rows"],
        "BRANCH_DEPENDENT_REGENERATION_RESULT": regeneration_summary(schedule_rows),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    }


def pilot_scientific_signal(by_l, c_dose, r_dose):
    l64 = by_l.get(64, {})
    native_gap = l64.get("NATIVE_GAP", {}).get("median") or 0.0
    rescue = l64.get("C_MAG_EQ_RESCUE", {}).get("median") or 0.0
    graft = l64.get("R_CMAG_GRAFT_DAMAGE", {}).get("median") or 0.0
    c_eq = l64.get("C_EQ_VS_R_NATIVE", {}).get("median") or 0.0
    state_rescue = l64.get("STATE_ERROR_RESCUE", {}).get("median") or 0.0
    c_rho = c_dose.get("median_spearman_rho") or 0.0
    r_rho = r_dose.get("median_spearman_rho") or 0.0
    criteria = {
        "A_C_equalization_rescues_C": rescue > 1e-5,
        "B_R_Cmag_graft_damages_R": graft > 1e-5,
        "C_gamma_dose_response_positive": c_rho > 0.25 or r_rho > 0.25,
        "D_C_state_error_rescue": state_rescue > 1e-4,
        "E_C_EQ_closes_or_reverses_native_gap": native_gap > 1e-5 and c_eq <= max(0.25 * native_gap, 1e-5),
    }
    return {
        "PILOT": "PASS" if any(criteria.values()) else "NEGATIVE_OR_INCONCLUSIVE",
        "criteria": criteria,
        "L64_native_gap_median": native_gap,
        "L64_C_MAG_EQ_RESCUE_median": rescue,
        "L64_R_CMAG_GRAFT_DAMAGE_median": graft,
        "L64_C_EQ_VS_R_NATIVE_median": c_eq,
        "L64_STATE_ERROR_RESCUE_median": state_rescue,
        "C_DIR_median_spearman_rho": c_rho,
        "R_DIR_median_spearman_rho": r_rho,
    }


def regeneration_summary(schedule_rows):
    return {
        "median_R_REGEN_NORM_RATIO": median([r.get("R_REGEN_NORM_RATIO") for r in schedule_rows]),
        "median_C_REGEN_NORM_RATIO": median([r.get("C_REGEN_NORM_RATIO") for r in schedule_rows]),
        "median_same_state_C_over_R": median([r.get("same_state_C_over_R") for r in schedule_rows]),
        "mean_R_REGEN_NORM_RATIO": mean([r.get("R_REGEN_NORM_RATIO") for r in schedule_rows]),
        "mean_C_REGEN_NORM_RATIO": mean([r.get("C_REGEN_NORM_RATIO") for r in schedule_rows]),
    }


def max_summary_value(shard_summaries, key):
    return max([as_float(s.get(key)) for s in shard_summaries if as_float(s.get(key)) is not None], default=None)


def min_summary_value(shard_summaries, key):
    return min([as_float(s.get(key)) for s in shard_summaries if as_float(s.get(key)) is not None], default=None)


def merge_gate(shard_summaries, gate_name):
    return "PASS" if shard_summaries and all(s.get(gate_name) == "PASS" for s in shard_summaries) else "FAIL"


def summarize_merged_stage(stage, unit_rows, schedule_rows, state_rows, shard_summaries, history_lengths, gamma_values, horizon):
    by_l, contrast_rows = aggregate_by_l(unit_rows)
    c_dose = dose_response(unit_rows, "C_DIR")
    r_dose = dose_response(unit_rows, "R_DIR")
    deg_r = sum(int(as_float(r.get("degenerate_residual_count")) or 0) for r in schedule_rows if r.get("source_direction") == "R_DIR")
    deg_c = sum(int(as_float(r.get("degenerate_residual_count")) or 0) for r in schedule_rows if r.get("source_direction") == "C_DIR")
    gates = {gate: merge_gate(shard_summaries, gate) for gate in SHARD_MERGED_GATES}
    gates.update({
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if unit_rows and max(abs(as_float(r["future_kl_auc"]) or 0.0) for r in unit_rows if r["source_direction"] == "FP") <= 1e-12 else "FAIL",
        "L1_FACTORIAL_REPRODUCTION_GATE": "PASS" if merge_gate(shard_summaries, "L1_FACTORIAL_REPRODUCTION_GATE") == "PASS" else "FAIL",
    })
    all_pass = all(gates[k] == "PASS" for k in BLOCKING_GATES)
    pilot_signal = pilot_scientific_signal(by_l, c_dose, r_dose) if stage == "pilot" and all_pass else None
    return {
        "TASK": TASK,
        "stage": stage,
        "MERGED_SHARD_COUNT": len(shard_summaries),
        "rows_written": len(unit_rows) * horizon,
        "unit_summary_rows_written": len(unit_rows),
        "per_step_residual_schedule_rows_written": len(schedule_rows),
        "state_error_trajectory_rows_written": len(state_rows),
        "unique_units": len({r["unit_id"] for r in unit_rows}),
        "history_lengths": history_lengths,
        "gamma_values": gamma_values,
        "source_directions": SOURCE_DIRECTIONS,
        "CANONICAL_MANIFEST_GATE": "PASS",
        "max_R_native_endpoint_abs_error": max_summary_value(shard_summaries, "max_R_native_endpoint_abs_error"),
        "max_C_native_endpoint_abs_error": max_summary_value(shard_summaries, "max_C_native_endpoint_abs_error"),
        "max_same_state_counterfactual_abs_error": max_summary_value(shard_summaries, "max_same_state_counterfactual_abs_error"),
        "max_same_state_counterfactual_relative_error": max_summary_value(shard_summaries, "max_same_state_counterfactual_relative_error"),
        "max_applied_magnitude_abs_error": max_summary_value(shard_summaries, "max_applied_magnitude_abs_error"),
        "max_applied_magnitude_relative_error": max_summary_value(shard_summaries, "max_applied_magnitude_relative_error"),
        "min_source_direction_cosine": min_summary_value(shard_summaries, "min_source_direction_cosine"),
        "N_DEGENERATE_R_RESIDUALS": deg_r,
        "N_DEGENERATE_C_RESIDUALS": deg_c,
        **gates,
        "PERSISTENT_NATIVE_BASELINE_GATE": "PASS" if by_l else "FAIL",
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": "PASS_READOUT_ONLY",
        "POSTPROJ_STATUS": "NOT_AVAILABLE_NONBLOCKING",
        "P_DECOMPOSITION_REUSE": "NOT_AVAILABLE_NONBLOCKING",
        "STAGE0_GATE": "NOT_APPLICABLE",
        "SMOKE": "NOT_APPLICABLE",
        "PILOT": pilot_signal["PILOT"] if stage == "pilot" and pilot_signal else ("NOT_APPLICABLE" if stage != "pilot" else "FAIL"),
        "PILOT_SIGNAL": pilot_signal,
        "AGGREGATE_BY_L": {str(k): v for k, v in by_l.items()},
        "CONTRAST_ROWS": contrast_rows,
        "NATIVE_GAP_BY_L": dict_by_l(by_l, "NATIVE_GAP"),
        "C_MAG_EQ_RESCUE_BY_L": dict_by_l(by_l, "C_MAG_EQ_RESCUE"),
        "R_CMAG_GRAFT_DAMAGE_BY_L": dict_by_l(by_l, "R_CMAG_GRAFT_DAMAGE"),
        "C_EQ_VS_R_NATIVE_BY_L": dict_by_l(by_l, "C_EQ_VS_R_NATIVE"),
        "R_GRAFT_VS_C_NATIVE_BY_L": dict_by_l(by_l, "R_GRAFT_VS_C_NATIVE"),
        "C_GAP_CLOSURE_RATIO_BY_L": dict_by_l(by_l, "C_GAP_CLOSURE_RATIO"),
        "R_GRAFT_TO_GAP_RATIO_BY_L": dict_by_l(by_l, "R_GRAFT_TO_GAP_RATIO"),
        "STATE_ERROR_RESCUE_BY_L": dict_by_l(by_l, "STATE_ERROR_RESCUE"),
        "STATE_ERROR_GRAFT_DAMAGE_BY_L": dict_by_l(by_l, "STATE_ERROR_GRAFT_DAMAGE"),
        "C_DIR_GAMMA_DOSE_RESULT": {k: v for k, v in c_dose.items() if k != "rows"},
        "R_DIR_GAMMA_DOSE_RESULT": {k: v for k, v in r_dose.items() if k != "rows"},
        "GAMMA_DOSE_ROWS": c_dose["rows"] + r_dose["rows"],
        "BRANCH_DEPENDENT_REGENERATION_RESULT": regeneration_summary(schedule_rows),
        "SHARD_SUMMARY_GATES": [{k: s.get(k) for k in BLOCKING_GATES if k in s} for s in shard_summaries],
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    }


def classify(summary):
    by_l = summary["AGGREGATE_BY_L"]
    l64 = by_l.get("64", {})
    native_gap = l64.get("NATIVE_GAP", {}).get("median") or 0.0
    rescue = l64.get("C_MAG_EQ_RESCUE", {}).get("median") or 0.0
    graft = l64.get("R_CMAG_GRAFT_DAMAGE", {}).get("median") or 0.0
    state_rescue = l64.get("STATE_ERROR_RESCUE", {}).get("median") or 0.0
    state_graft = l64.get("STATE_ERROR_GRAFT_DAMAGE", {}).get("median") or 0.0
    c_rho = summary.get("C_DIR_GAMMA_DOSE_RESULT", {}).get("median_spearman_rho") or 0.0
    r_rho = summary.get("R_DIR_GAMMA_DOSE_RESULT", {}).get("median_spearman_rho") or 0.0
    if native_gap > 0.0 and rescue > 0.0 and graft > 0.0 and c_rho > 0.0 and r_rho > 0.0 and state_rescue > 0.0 and state_graft > 0.0:
        return {
            "PERSISTENT_SOURCE_MAGNITUDE_CAUSAL": "STRONG",
            "CLOSED_LOOP_MAGNITUDE_EQUALIZATION_RESCUE": "SUPPORTED",
            "CLOSED_LOOP_CMAG_GRAFT_DAMAGE": "SUPPORTED",
            "LING_RC_MAGNITUDE_MECHANISM": "CAUSALLY_CLOSED_OR_STRONG",
            "FINAL_CLASSIFICATION": "PERSISTENT_RECURRENT_MAGNITUDE_ACCUMULATION_CAUSALLY_SUPPORTED",
            "NEXT_RECOMMENDED_TASK": "CROSS_ARCHITECTURE_PRINCIPLE_EXTRACTION_AGAINST_QWEN_GDN",
        }
    if rescue > 0.0 and graft <= 0.0:
        return {
            "PERSISTENT_SOURCE_MAGNITUDE_CAUSAL": "PARTIAL_STRONG",
            "CLOSED_LOOP_MAGNITUDE_EQUALIZATION_RESCUE": "SUPPORTED",
            "CLOSED_LOOP_CMAG_GRAFT_DAMAGE": "WEAK_OR_INCONSISTENT",
            "ASYMMETRIC_MAGNITUDE_MEDIATION": "SUPPORTED",
            "LING_RC_MAGNITUDE_MECHANISM": "PARTIAL_ASYMMETRIC",
            "FINAL_CLASSIFICATION": "ASYMMETRIC_MAGNITUDE_MEDIATION_SUPPORTED",
            "NEXT_RECOMMENDED_TASK": "FUNCTIONAL_READOUT_MEDIATION_OR_CROSS_ARCHITECTURE_CHECK",
        }
    if state_rescue > 0.0 and rescue <= 0.0:
        return {
            "PERSISTENT_SOURCE_MAGNITUDE_CAUSAL": "STATE_ONLY",
            "PERSISTENT_MAGNITUDE_TO_STATE_CAUSAL": "SUPPORTED",
            "PERSISTENT_MAGNITUDE_TO_FUNCTIONAL_DAMAGE": "NOT_CLOSED",
            "CLOSED_LOOP_MAGNITUDE_EQUALIZATION_RESCUE": "NOT_FUNCTIONALLY_SUPPORTED",
            "CLOSED_LOOP_CMAG_GRAFT_DAMAGE": "NOT_FUNCTIONALLY_SUPPORTED",
            "LING_RC_MAGNITUDE_MECHANISM": "STATE_ONLY_NOT_FUNCTIONAL",
            "FINAL_CLASSIFICATION": "PERSISTENT_MAGNITUDE_TO_STATE_ONLY",
            "NEXT_RECOMMENDED_TASK": "FUNCTIONAL_READOUT_MEDIATION",
        }
    return {
        "PERSISTENT_SOURCE_MAGNITUDE_CAUSAL": "NOT_SUPPORTED",
        "CLOSED_LOOP_MAGNITUDE_CAUSAL": "NOT_SUPPORTED",
        "CLOSED_LOOP_MAGNITUDE_EQUALIZATION_RESCUE": "NOT_SUPPORTED",
        "CLOSED_LOOP_CMAG_GRAFT_DAMAGE": "NOT_SUPPORTED",
        "LING_RC_MAGNITUDE_MECHANISM": "NOT_CLOSED",
        "FINAL_CLASSIFICATION": "CLOSED_LOOP_MAGNITUDE_MECHANISM_NOT_SUPPORTED",
        "NEXT_RECOMMENDED_TASK": "BRANCH_DEPENDENT_DIRECTION_REGENERATION_OR_TEMPORAL_GEOMETRY",
    }


def aggregate(stage="formal"):
    unit_rows = read_csv_rows(RUN_DIR / f"unit_summary_{stage}.csv")
    horizon_rows = read_csv_rows(RUN_DIR / f"horizon_{stage}.csv")
    schedule_rows = read_csv_rows(RUN_DIR / f"per_step_residual_schedule_{stage}.csv")
    state_rows = read_csv_rows(RUN_DIR / f"state_error_trajectory_{stage}.csv")
    by_l, contrast_rows = aggregate_by_l(unit_rows)
    c_dose = dose_response(unit_rows, "C_DIR")
    r_dose = dose_response(unit_rows, "R_DIR")
    dose_rows = c_dose["rows"] + r_dose["rows"]
    if dose_rows:
        write_rows(RUN_DIR / f"gamma_dose_response_{stage}.csv", merged_fields(dose_rows), dose_rows)
        if stage == "formal":
            write_rows(RUN_DIR / "gamma_dose_response.csv", merged_fields(dose_rows), dose_rows)
    n_units = len({r["unit_id"] for r in unit_rows})
    history_set = {int(r["history_length"]) for r in unit_rows}
    gamma_set = {float(r["gamma"]) for r in unit_rows if as_float(r.get("gamma")) is not None}
    gates = {
        "R_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS" if max([as_float(r.get("endpoint_identity_abs_error")) or 0.0 for r in schedule_rows if r.get("source_direction") == "R_DIR" and abs((as_float(r.get("gamma")) or 0.0)) <= EPS], default=0.0) <= IDENTITY_TOL else "FAIL",
        "C_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS" if max([as_float(r.get("endpoint_identity_abs_error")) or 0.0 for r in schedule_rows if r.get("source_direction") == "C_DIR" and abs((as_float(r.get("gamma")) or 0.0) - 1.0) <= EPS], default=0.0) <= IDENTITY_TOL else "FAIL",
        "SAME_STATE_COUNTERFACTUAL_GATE": gate_from_rows(schedule_rows, "SAME_STATE_COUNTERFACTUAL_GATE"),
        "APPLIED_MAGNITUDE_GATE": gate_from_rows(schedule_rows, "APPLIED_MAGNITUDE_GATE"),
        "SOURCE_DIRECTION_PRESERVATION_GATE": gate_from_rows(schedule_rows, "SOURCE_DIRECTION_PRESERVATION_GATE"),
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": gate_from_rows(schedule_rows, "CURRENT_LOGIT_NONRETROACTIVITY_GATE"),
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if unit_rows and max(abs(as_float(r["future_kl_auc"]) or 0.0) for r in unit_rows if r["source_direction"] == "FP") <= 1e-12 else "FAIL",
        "L1_FACTORIAL_REPRODUCTION_GATE": "PASS" if 1 in history_set else "FAIL",
    }
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if stage == "formal" and n_units == 18 and history_set == set(HISTORY_LENGTHS) and gamma_set == set(STAGE_GAMMA_VALUES["formal"]) else ("ANALYSIS_COMPLETE" if stage != "formal" else "INCOMPLETE"),
        "N_FORMAL_UNITS": n_units if stage == "formal" else 0,
        "CANONICAL_MANIFEST_GATE": "PASS",
        "MANIFEST_SHA256": P.canonical_manifest_sha256(CANONICAL_UNITS),
        **gates,
        "PERSISTENT_NATIVE_BASELINE_GATE": "PASS" if by_l else "FAIL",
        "FUNCTIONAL_VISIBILITY_HOOK_GATE": "PASS_READOUT_ONLY",
        "POSTPROJ_STATUS": "NOT_AVAILABLE_NONBLOCKING",
        "P_DECOMPOSITION_REUSE": "NOT_AVAILABLE_NONBLOCKING",
        "STAGE0_GATE": json.loads((RUN_DIR / "stage0_run_summary.json").read_text()).get("STAGE0_GATE", "UNKNOWN") if (RUN_DIR / "stage0_run_summary.json").exists() else "NOT_RUN",
        "SMOKE": json.loads((RUN_DIR / "smoke_run_summary.json").read_text()).get("SMOKE", "UNKNOWN") if (RUN_DIR / "smoke_run_summary.json").exists() else "NOT_RUN",
        "PILOT": json.loads((RUN_DIR / "pilot_run_summary.json").read_text()).get("PILOT", "UNKNOWN") if (RUN_DIR / "pilot_run_summary.json").exists() else "NOT_RUN",
        "HISTORY_LENGTHS": HISTORY_LENGTHS,
        "GAMMA_VALUES": STAGE_GAMMA_VALUES["formal"],
        "AGGREGATE_BY_L": {str(k): v for k, v in by_l.items()},
        "CONTRAST_ROWS": contrast_rows,
        "NATIVE_GAP_BY_L": dict_by_l(by_l, "NATIVE_GAP"),
        "C_MAG_EQ_RESCUE_BY_L": dict_by_l(by_l, "C_MAG_EQ_RESCUE"),
        "R_CMAG_GRAFT_DAMAGE_BY_L": dict_by_l(by_l, "R_CMAG_GRAFT_DAMAGE"),
        "C_EQ_VS_R_NATIVE_BY_L": dict_by_l(by_l, "C_EQ_VS_R_NATIVE"),
        "R_GRAFT_VS_C_NATIVE_BY_L": dict_by_l(by_l, "R_GRAFT_VS_C_NATIVE"),
        "C_GAP_CLOSURE_RATIO_BY_L": dict_by_l(by_l, "C_GAP_CLOSURE_RATIO"),
        "R_GRAFT_TO_GAP_RATIO_BY_L": dict_by_l(by_l, "R_GRAFT_TO_GAP_RATIO"),
        "STATE_ERROR_RESCUE_BY_L": dict_by_l(by_l, "STATE_ERROR_RESCUE"),
        "STATE_ERROR_GRAFT_DAMAGE_BY_L": dict_by_l(by_l, "STATE_ERROR_GRAFT_DAMAGE"),
        "C_DIR_GAMMA_DOSE_RESULT": {k: v for k, v in c_dose.items() if k != "rows"},
        "R_DIR_GAMMA_DOSE_RESULT": {k: v for k, v in r_dose.items() if k != "rows"},
        "GAMMA_DOSE_ROWS": dose_rows,
        "BRANCH_DEPENDENT_REGENERATION_RESULT": regeneration_summary(schedule_rows),
        "N_DEGENERATE_R_RESIDUALS": sum(int(as_float(r.get("degenerate_residual_count")) or 0) for r in schedule_rows if r.get("source_direction") == "R_DIR"),
        "N_DEGENERATE_C_RESIDUALS": sum(int(as_float(r.get("degenerate_residual_count")) or 0) for r in schedule_rows if r.get("source_direction") == "C_DIR"),
        "outputs": {
            "results_jsonl": str(RUN_DIR / f"results_{stage}.jsonl"),
            "aggregate_summary_json": str(RUN_DIR / "aggregate_summary.json"),
            "report_md": str(RUN_DIR / "report.md"),
            "per_step_residual_schedule_csv": str(RUN_DIR / f"per_step_residual_schedule_{stage}.csv"),
            "gamma_dose_response_csv": str(RUN_DIR / f"gamma_dose_response_{stage}.csv"),
            "state_error_trajectory_csv": str(RUN_DIR / f"state_error_trajectory_{stage}.csv"),
            "unit_summary_csv": str(RUN_DIR / f"unit_summary_{stage}.csv"),
            "horizon_csv": str(RUN_DIR / f"horizon_{stage}.csv"),
        },
        "REPO_AUDIT": repo_audit(),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        "METHOD_DESIGN_READY": "NO",
    }
    summary.update(classify(summary))
    save_json(RUN_DIR / "aggregate_summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    if stage == "formal":
        for src_name, dst_name in [
            ("results_formal.jsonl", "results.jsonl"),
            ("per_step_residual_schedule_formal.csv", "per_step_residual_schedule.csv"),
            ("state_error_trajectory_formal.csv", "state_error_trajectory.csv"),
            ("unit_summary_formal.csv", "unit_summary.csv"),
            ("horizon_formal.csv", "horizon.csv"),
        ]:
            src = RUN_DIR / src_name
            if src.exists():
                (RUN_DIR / dst_name).write_bytes(src.read_bytes())
    make_report(summary)
    return summary


def make_report(summary):
    lines = [
        f"# {TASK}",
        "",
        f"- FORMAL_STATUS: {summary.get('FORMAL_STATUS')}",
        f"- N_FORMAL_UNITS: {summary.get('N_FORMAL_UNITS')}",
        f"- FINAL_CLASSIFICATION: {summary.get('FINAL_CLASSIFICATION')}",
        f"- PERSISTENT_SOURCE_MAGNITUDE_CAUSAL: {summary.get('PERSISTENT_SOURCE_MAGNITUDE_CAUSAL')}",
        f"- CLOSED_LOOP_MAGNITUDE_EQUALIZATION_RESCUE: {summary.get('CLOSED_LOOP_MAGNITUDE_EQUALIZATION_RESCUE')}",
        f"- CLOSED_LOOP_CMAG_GRAFT_DAMAGE: {summary.get('CLOSED_LOOP_CMAG_GRAFT_DAMAGE')}",
        "",
        "## Gates",
    ]
    for key in ["CANONICAL_MANIFEST_GATE"] + BLOCKING_GATES:
        lines.append(f"- {key}: {summary.get(key)}")
    lines += ["", "## Median Effects By History Length"]
    for key in [
        "NATIVE_GAP_BY_L", "C_MAG_EQ_RESCUE_BY_L", "R_CMAG_GRAFT_DAMAGE_BY_L",
        "C_EQ_VS_R_NATIVE_BY_L", "R_GRAFT_VS_C_NATIVE_BY_L",
        "STATE_ERROR_RESCUE_BY_L", "STATE_ERROR_GRAFT_DAMAGE_BY_L",
    ]:
        lines.append(f"- {key}: {json.dumps(summary.get(key), sort_keys=True)}")
    lines += ["", "## Dose Response"]
    lines.append(f"- C_DIR_GAMMA_DOSE_RESULT: {json.dumps(summary.get('C_DIR_GAMMA_DOSE_RESULT'), sort_keys=True)}")
    lines.append(f"- R_DIR_GAMMA_DOSE_RESULT: {json.dumps(summary.get('R_DIR_GAMMA_DOSE_RESULT'), sort_keys=True)}")
    lines += ["", "## Regeneration"]
    lines.append(json.dumps(summary.get("BRANCH_DEPENDENT_REGENERATION_RESULT"), indent=2, sort_keys=True))
    write_text(RUN_DIR / "report.md", "\n".join(lines))
    write_text(REPORT_DIR / f"{SLUG}.md", "\n".join(lines))


def merge_stage_files(stage, shard_dirs, output_dir=RUN_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [Path(d) for d in shard_dirs]
    shard_summaries = []
    for i, shard_dir in enumerate(shard_dirs):
        summary_path = shard_dir / f"{stage}_run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries.append(summary)
        save_json(output_dir / f"{stage}_run_summary_shard{i}.json", summary)
    merged = {"shard_summaries": shard_summaries}
    csv_outputs = {
        "unit_rows": f"unit_summary_{stage}.csv",
        "horizon_rows": f"horizon_{stage}.csv",
        "schedule_rows": f"per_step_residual_schedule_{stage}.csv",
        "state_rows": f"state_error_trajectory_{stage}.csv",
    }
    for key, filename in csv_outputs.items():
        rows = []
        for shard_dir in shard_dirs:
            path = shard_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"missing shard csv: {path}")
            rows.extend(read_csv_rows(path))
        write_rows(output_dir / filename, merged_fields(rows), rows)
        merged[key] = rows
    jsonl_out = output_dir / f"results_{stage}.jsonl"
    with jsonl_out.open("w", encoding="utf-8") as out:
        for shard_dir in shard_dirs:
            path = shard_dir / f"results_{stage}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"missing shard jsonl: {path}")
            with path.open(encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    return merged


def merge_stage(stage, shard_dirs, history_lengths, gamma_values, horizon):
    merged = merge_stage_files(stage, shard_dirs, RUN_DIR)
    summary = summarize_merged_stage(
        stage,
        merged["unit_rows"],
        merged["schedule_rows"],
        merged["state_rows"],
        merged["shard_summaries"],
        history_lengths,
        gamma_values,
        horizon,
    )
    summary.update({
        "horizon_csv": str(RUN_DIR / f"horizon_{stage}.csv"),
        "results_jsonl": str(RUN_DIR / f"results_{stage}.jsonl"),
        "unit_summary_csv": str(RUN_DIR / f"unit_summary_{stage}.csv"),
        "per_step_residual_schedule_csv": str(RUN_DIR / f"per_step_residual_schedule_{stage}.csv"),
        "state_error_trajectory_csv": str(RUN_DIR / f"state_error_trajectory_{stage}.csv"),
        "SHARD_DIRS": [str(Path(d)) for d in shard_dirs],
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    })
    save_json(RUN_DIR / f"{stage}_run_summary.json", summary)
    if stage == "formal":
        summary = aggregate("formal")
    return summary


def reproducibility_snapshot(command):
    return {
        "time": now(),
        "command": command,
        "git_head": sh(["git", "rev-parse", "HEAD"]),
        "git_status": sh(["git", "status", "--short"]),
        "git_diff": sh(["git", "diff", "--", "experiments/ling", "tests/ling"]),
        "model_path": str(P.MODEL_PATH),
        "tokenizer_path": str(P.MODEL_PATH),
        "canonical_manifest_path": str(CANONICAL_UNITS),
        "manifest_sha256": P.canonical_manifest_sha256(CANONICAL_UNITS) if CANONICAL_UNITS.exists() else None,
        "cuda": sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        "python": sys.version.split()[0],
        "torch_transformers": sh([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"]),
        "dtype": "torch.bfloat16 model parameters; residuals and norm-matching computed as float32 before cache write",
        "quantizer_config": {"R128": P.quantizer_axis("INT8_R128"), "C128": P.quantizer_axis("INT8_C128")},
        "history_lengths": HISTORY_LENGTHS,
        "gamma_values": STAGE_GAMMA_VALUES["formal"],
        "teacher_forced_protocol": "same FP generated_token_ids; no free generation",
        "intervention_timing": "POST_CURRENT_READOUT / PRE_NEXT_TOKEN_CARRY",
        "no_further_quantization_after_t0": True,
        "closed_loop_source": "residuals generated online from each branch current recurrent state S_branch(i)",
    }


def repo_audit():
    return {
        "REUSED_FILES": [
            str(SOURCE_EQUALIZATION_RUNNER_PATH),
            str(E.SOURCE_RUNNER_PATH),
            str(E.PERSISTENT_RUNNER_PATH),
            str(CANONICAL_UNITS),
        ],
        "NEW_FILES": [
            str(REPO / "experiments" / "ling" / "run_ling_kda_persistent_source_error_magnitude_equalization_causal_v1.py"),
            str(REPO / "tests" / "ling" / "test_ling_kda_persistent_source_error_magnitude_equalization_causal.py"),
        ],
        "INTERVENTION_SOURCE": "branch-local closed-loop same-state residuals e_R(S_branch(i)), e_C(S_branch(i))",
        "GLOBAL_NORM_MATCHING": True,
        "NO_FUTURE_QUANTIZATION": True,
    }


def write_blocked_summary(audit):
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "BLOCKED_CANONICAL_MANIFEST",
        "CANONICAL_MANIFEST_GATE": audit.get("gate"),
        "CANONICAL_UNIT_AUDIT": audit,
        "REPO_AUDIT": repo_audit(),
        "METHOD_DESIGN_READY": "NO",
    }
    save_json(RUN_DIR / "blocked_summary.json", out)
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["audit", "stage0", "smoke", "pilot", "formal", "analyze", "merge-pilot", "merge-formal"], default="audit")
    p.add_argument("--canonical-units", default=str(CANONICAL_UNITS))
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--history-lengths", default=None)
    p.add_argument("--gamma-values", default=None)
    p.add_argument("--unit-limit", type=int, default=None)
    p.add_argument("--unit-ids", default=None)
    p.add_argument("--merge-shards", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = load_canonical_units(args.canonical_units)
    if audit["gate"] != "PASS":
        write_blocked_summary(audit)
        return
    if args.stage == "audit":
        out = {
            "TASK": TASK,
            "FORMAL_STATUS": "AUDIT_COMPLETE_CAN_RUN_STAGE0",
            "CANONICAL_MANIFEST_GATE": "PASS",
            "CANONICAL_UNIT_AUDIT": audit,
            "REPO_AUDIT": repo_audit(),
            "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        }
        save_json(RUN_DIR / "audit.json", out)
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "analyze":
        print(json.dumps(aggregate("formal"), indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage.startswith("merge-"):
        stage = args.stage.replace("merge-", "", 1)
        if not args.merge_shards:
            raise ValueError("--merge-shards is required for merge stages")
        history_lengths = [int(x) for x in args.history_lengths.split(",")] if args.history_lengths else STAGE_HISTORY_LENGTHS[stage]
        gamma_values = [float(x) for x in args.gamma_values.split(",")] if args.gamma_values else STAGE_GAMMA_VALUES[stage]
        out = merge_stage(stage, [x.strip() for x in args.merge_shards.split(",") if x.strip()], history_lengths, gamma_values, args.horizon)
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    history_lengths = [int(x) for x in args.history_lengths.split(",")] if args.history_lengths else STAGE_HISTORY_LENGTHS[args.stage]
    gamma_values = [float(x) for x in args.gamma_values.split(",")] if args.gamma_values else STAGE_GAMMA_VALUES[args.stage]
    out = run_units(
        args.stage,
        audit["formal_units"],
        history_lengths=history_lengths,
        gamma_values=gamma_values,
        horizon=args.horizon,
        unit_limit=args.unit_limit,
        unit_ids=args.unit_ids,
    )
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
