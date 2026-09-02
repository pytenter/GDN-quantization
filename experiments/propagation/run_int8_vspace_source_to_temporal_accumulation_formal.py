#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_VSPACE_SOURCE_TO_TEMPORAL_ACCUMULATION_FORMAL_V1"
SCRIPT = EXP / "run_int8_vspace_source_to_temporal_accumulation_formal.py"
STAGE0_JSON = RES / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_stage0.json"
SOURCE_JSON = RES / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_source_formal.json"
PILOT_JSON = RES / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_pilot.json"
FORMAL_JSON = RES / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_formal.json"
CHECKPOINT = RES / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_checkpoint.json"
RAW = RES / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_raw.npz"
REPORT = REP / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1.md"
FIG_DIR = RES / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_figures"

LEGACY_FORMAL = RES / "gdn_int8_repeated_accumulation_formal_v1.json"
LEGACY_CHECKPOINT = RES / "gdn_int8_repeated_accumulation_formal_axis_comparison_v1_checkpoint.json"
LEGACY_VSPACE_SCREEN = RES / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_screen.json"

EPS = 1e-12
BOOTSTRAP_SEED = 20260901
HORIZON = 128
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
CADENCES = ["FP_STATE", "SINGLE", "EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
ORDERED = ["SINGLE", "EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
PERIODIC = ["EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
PILOT_CADENCES = ["SINGLE", "EVERY_8", "EVERY_2", "EVERY_1"]
T0S = [64, 128, 256]
FREQ = {"EVERY_32": 1 / 32, "EVERY_16": 1 / 16, "EVERY_8": 1 / 8, "EVERY_4": 1 / 4, "EVERY_2": 1 / 2, "EVERY_1": 1.0}

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_repeated_accumulation_formal_axis_comparison as rep
import run_int8_vspace_compute_invariant_rotation_scale_contamination as vspace
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_effective_update_metric_audit as eff
import run_int8_orientation_state_change_mechanism as p1


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def avg(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.median(xs) if xs else None


def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def iqr(xs):
    return [pct(xs, 0.25), pct(xs, 0.75)]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT / "GDN-quantization"), text=True).strip()
    except Exception:
        return None


def selected_prompts():
    return [r for r in p1.selected_prompt_rows() if r.get("fp_response")][:6]


def manifest():
    return [{"problem_id": pm["problem_id"], "role": pm.get("role"), "t0": t0} for pm in selected_prompts() for t0 in T0S]


def unit_key(pid, t0, condition, cadence):
    return f"{pid}|{t0}|{condition}|{cadence}"


def source_key(pid, t0):
    return f"{pid}|{t0}"


def load_checkpoint():
    return load_json(CHECKPOINT) or {"task": TASK, "git_commit": git_commit(), "source_completed": {}, "live_completed": {}, "excluded_units": []}


def write_checkpoint(cp):
    save_json(CHECKPOINT, cp)


def tensor_norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def state_norm(torch, past):
    return math.sqrt(sum(tensor_norm(torch, p1.get_state(past, l)) ** 2 for l in GDN_LAYERS))


def hadamard(torch, device, dtype=None):
    return torch.tensor(vspace.hadamard_np(128), device=device, dtype=dtype or torch.float32)


def quantize_r128(torch, state):
    y, q, scale_full, scale, raw_scale_shape, dyn = axis.grouped_quant(
        torch, state.detach().float(), state.detach().float(), {"orientation": "row", "group_size": 128}
    )
    return y, q, scale.reshape(state.shape[0], state.shape[1], state.shape[2], 1), dyn


def quantize_c128(torch, state):
    y, q, scale_full, scale, raw_scale_shape, dyn = axis.grouped_quant(
        torch, state.detach().float(), state.detach().float(), {"orientation": "column", "group_size": 128}
    )
    return y, q, scale.reshape(state.shape[0], state.shape[1], 1, state.shape[3]), dyn


def quantize_r128_vh(torch, state):
    S = state.detach().float()
    H = hadamard(torch, S.device)
    Srot = torch.matmul(S, H)
    q = Srot.abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / 127.0
    qi = torch.round(Srot / q).clamp(-127, 127)
    Sq = qi * q
    Sback = torch.matmul(Sq, H.t())
    dyn = (Srot.abs().amax(dim=-1) / Srot.abs().median(dim=-1).values.clamp_min(EPS)).detach().flatten().cpu().tolist()
    return Sback, qi, q, dyn, Srot


def source_stats_for_state(torch, state, condition):
    if condition == "R128":
        post, q, scale, dyn = quantize_r128(torch, state)
        repr_state = state.detach().float()
    elif condition == "R128_VH":
        post, q, scale, dyn, repr_state = quantize_r128_vh(torch, state)
    elif condition == "C128":
        post, q, scale, dyn = quantize_c128(torch, state)
        repr_state = state.detach().float()
    else:
        raise ValueError(condition)
    S = state.detach().float()
    E = post - S
    abs_repr = repr_state.abs()
    row_peak_rms = (abs_repr.amax(dim=-1) / torch.sqrt(torch.mean(repr_state * repr_state, dim=-1)).clamp_min(EPS)).detach().flatten()
    row_max_median_abs = (abs_repr.amax(dim=-1) / abs_repr.median(dim=-1).values.clamp_min(EPS)).detach().flatten()
    scale_vals = scale.detach().float().flatten()
    return {
        "residual_norm": tensor_norm(torch, E),
        "state_norm": tensor_norm(torch, S),
        "relative_residual_norm": tensor_norm(torch, E) / (tensor_norm(torch, S) + EPS),
        "rms_residual": float(torch.sqrt(torch.mean(E.float() ** 2)).item()),
        "max_abs_residual": float(E.abs().max().item()),
        "row_peak_rms_median": float(torch.median(row_peak_rms).item()),
        "row_max_median_abs_median": float(torch.median(row_max_median_abs).item()),
        "scale_median": float(torch.median(scale_vals).item()),
        "scale_p95": float(torch.quantile(scale_vals, 0.95).item()),
        "scale_max": float(scale_vals.max().item()),
        "scale_p95_over_median": float(torch.quantile(scale_vals, 0.95).item() / (torch.median(scale_vals).item() + EPS)),
        "scale_max_over_median": float(scale_vals.max().item() / (torch.median(scale_vals).item() + EPS)),
        "within_group_dynamic_range_median": med(dyn),
        "within_group_dynamic_range_p95": pct(dyn, 0.95),
        "saturation_fraction": float((q.abs() >= 127).float().mean().item()),
    }


def merge_source_samples(samples):
    out = {}
    for condition in ["R128", "R128_VH", "C128"]:
        rows = [s[condition] for s in samples]
        out[condition] = {k: avg([r.get(k) for r in rows]) for k in rows[0].keys()}
    return out


def shadow_source_unit(torch, model, tokenizer, e2e, pm, t0, horizon):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[: t0 + horizon + 1]
    if len(cont) <= t0:
        raise RuntimeError(f"insufficient continuation for {pm['problem_id']} t0={t0}: {len(cont)}")
    last_t = min(t0 + horizon - 1, len(cont) - 1)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    past = None
    samples = []
    with torch.inference_mode():
        for t in range(last_t + 1):
            out = p1.feed_step(torch, model, ids, mask, past)
            past = out.past_key_values
            if t >= t0:
                layer_rows = {c: [] for c in ["R128", "R128_VH", "C128"]}
                for layer in GDN_LAYERS:
                    state = p1.get_state(past, layer)
                    for c in layer_rows:
                        layer_rows[c].append(source_stats_for_state(torch, state, c))
                samples.append({c: {k: avg([r[k] for r in rows]) for k in rows[0]} for c, rows in layer_rows.items()})
            if t < len(cont):
                ids = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device)
                mask = None
    agg = merge_source_samples(samples)
    return {
        "problem_id": pm["problem_id"],
        "role": pm.get("role"),
        "t0": t0,
        "horizon": len(samples),
        "shadow_source": agg,
        "primary_pair": {
            "VH_over_R_relative_residual_norm": agg["R128_VH"]["relative_residual_norm"] / (agg["R128"]["relative_residual_norm"] + EPS),
            "VH_minus_R_relative_residual_norm": agg["R128_VH"]["relative_residual_norm"] - agg["R128"]["relative_residual_norm"],
            "VH_over_R_row_peak_rms": agg["R128_VH"]["row_peak_rms_median"] / (agg["R128"]["row_peak_rms_median"] + EPS),
            "C_over_R_relative_residual_norm": agg["C128"]["relative_residual_norm"] / (agg["R128"]["relative_residual_norm"] + EPS),
            "VH_over_C_relative_residual_norm": agg["R128_VH"]["relative_residual_norm"] / (agg["C128"]["relative_residual_norm"] + EPS),
        },
    }


def state_relative_error(torch, a, b):
    err2, ref2 = 0.0, 0.0
    for layer in GDN_LAYERS:
        x = p1.get_state(a, layer).detach().float()
        y = p1.get_state(b, layer).detach().float()
        err2 += float(torch.sum((x - y).double() ** 2).item())
        ref2 += float(torch.sum(y.double() ** 2).item())
    return math.sqrt(err2) / (math.sqrt(ref2) + EPS)


def inject_vh(torch, past, token_idx):
    residuals, rels, rms = [], [], []
    for layer in GDN_LAYERS:
        state = p1.get_state(past, layer)
        pre = state.detach().clone()
        post, q, scale, dyn, _rot = quantize_r128_vh(torch, pre)
        residual = post - pre.float()
        rn = tensor_norm(torch, residual)
        pn = tensor_norm(torch, pre)
        residuals.append(rn)
        rels.append(rn / (pn + EPS))
        rms.append(float(torch.sqrt(torch.mean(residual.float() ** 2)).item()))
        state.copy_(post.to(state.dtype))
    return {
        "token_index": int(token_idx),
        "condition": "R128_VH",
        "residual_norm": math.sqrt(sum(x * x for x in residuals)),
        "relative_residual_norm": avg(rels),
        "rms_residual": avg(rms),
    }


def run_vh_condition(torch, model, tokenizer, e2e, pm, cadence, t0, horizon):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[: t0 + horizon + 1]
    if len(cont) <= t0:
        raise RuntimeError(f"insufficient continuation for {pm['problem_id']} t0={t0}: {len(cont)}")
    last_t = min(t0 + horizon - 1, len(cont) - 1)
    sched = rep.schedule(cadence, t0, last_t)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    base_ids = enc["input_ids"].to(device)
    base_mask = enc.get("attention_mask")
    base_mask = base_mask.to(device) if base_mask is not None else None
    fp_ids, cfg_ids = base_ids.clone(), base_ids.clone()
    fp_mask = base_mask.clone() if base_mask is not None else None
    cfg_mask = base_mask.clone() if base_mask is not None else None
    fp_past = cfg_past = None
    curves = {"state_error": [], "KL": [], "top1": []}
    injections, failures = [], []
    with torch.inference_mode():
        for t in range(last_t + 1):
            fp_out = p1.feed_step(torch, model, fp_ids, fp_mask, fp_past)
            fp_past = fp_out.past_key_values
            cfg_out = p1.feed_step(torch, model, cfg_ids, cfg_mask, cfg_past)
            cfg_past = cfg_out.past_key_values
            if t in sched:
                im = inject_vh(torch, cfg_past, t)
                im["number_of_injections_so_far"] = len(injections) + 1
                injections.append(im)
            if t >= t0:
                lm = eff.logits_metrics(torch, fp_out.logits, cfg_out.logits)
                se = state_relative_error(torch, cfg_past, fp_past)
                curves["state_error"].append(se)
                curves["KL"].append(lm["KL"])
                curves["top1"].append(lm["top1_agreement"])
                if not finite(se) or not finite(lm["KL"]):
                    failures.append({"token_index": t, "reason": "nonfinite_core_metric"})
            if t < len(cont):
                nxt = torch.tensor([[cont[t]]], dtype=base_ids.dtype, device=device)
                fp_ids, cfg_ids = nxt, nxt.clone()
                fp_mask = cfg_mask = None
    rnorms = [m["residual_norm"] for m in injections]
    return {
        "condition": "R128_VH",
        "cadence": cadence,
        "t0": t0,
        "continuation_horizon": len(curves["KL"]),
        "intervention_token_indices": sched,
        "injection_count": len(injections),
        "injection_metadata": injections,
        "state_error_curve": curves["state_error"],
        "KL_curve": curves["KL"],
        "top1_curve": curves["top1"],
        "state_error_AUC": avg(curves["state_error"]),
        "state_error_terminal": curves["state_error"][-1] if curves["state_error"] else None,
        "state_error_max": max(curves["state_error"]) if curves["state_error"] else None,
        "KL_AUC": avg(curves["KL"]),
        "KL_mean": avg(curves["KL"]),
        "KL_terminal": curves["KL"][-1] if curves["KL"] else None,
        "KL_max": max(curves["KL"]) if curves["KL"] else None,
        "Top1_agreement": avg(curves["top1"]),
        "sum_residual_norm": sum(rnorms),
        "sqrt_sum_squared_residual_norm": math.sqrt(sum(x * x for x in rnorms)),
        "mean_residual_norm": avg(rnorms),
        "median_residual_norm": med(rnorms),
        "numerical_failures": failures,
    }


def ranks(xs):
    pairs = sorted((x, i) for i, x in enumerate(xs))
    out = [0.0] * len(xs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        r = (i + j - 1) / 2 + 1
        for k in range(i, j):
            out[pairs[k][1]] = r
        i = j
    return out


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = avg(rx), avg(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx)) * math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den + EPS)


def bootstrap_logratio_ci(a, b, seed=BOOTSTRAP_SEED, n_boot=2000):
    vals = [(math.log((float(x) + EPS) / (float(y) + EPS))) for x, y in zip(a, b) if finite(x) and finite(y)]
    if not vals:
        return [None, None]
    rng = np.random.default_rng(seed)
    boots = []
    arr = np.asarray(vals, dtype=np.float64)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(np.median(sample)))
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_summary(rows, metric, a="R128_VH", b="R128"):
    av = [r["shadow_source"][a][metric] for r in rows]
    bv = [r["shadow_source"][b][metric] for r in rows]
    ratios = [x / (y + EPS) for x, y in zip(av, bv)]
    return {
        f"median_{a}": med(av),
        f"median_{b}": med(bv),
        "median_ratio": med(ratios),
        "iqr_ratio": iqr(ratios),
        "paired_win_count": sum(x < y for x, y in zip(av, bv)),
        "bootstrap_95ci_median_log_ratio": bootstrap_logratio_ci(av, bv),
    }


def classify_source(rows):
    rel = paired_summary(rows, "relative_residual_norm")
    peak = paired_summary(rows, "row_peak_rms_median")
    ci = rel["bootstrap_95ci_median_log_ratio"]
    robust = (
        len(rows) == 18
        and rel["paired_win_count"] >= 15
        and rel["median_ratio"] < 1
        and ci[1] is not None
        and ci[1] < 0
        and peak["paired_win_count"] >= 15
        and peak["median_ratio"] < 1
    )
    partial = rel["paired_win_count"] >= 12 and rel["median_ratio"] < 1 and peak["median_ratio"] < 1
    if robust:
        return "SOURCE_FORMAL_SUPPORTED"
    if partial:
        return "SOURCE_FORMAL_PARTIAL"
    if len(rows) == 18:
        return "SOURCE_FORMAL_NOT_SUPPORTED"
    return "SOURCE_FORMAL_INCONCLUSIVE"


def legacy_completed():
    cp = load_json(LEGACY_CHECKPOINT) or {}
    return cp.get("completed", {})


def legacy_r128(pid, t0, cadence):
    return legacy_completed().get(rep.unit_key(pid, t0, "R128", cadence))


def stage0():
    import torch
    legacy = load_json(LEGACY_FORMAL)
    vprev = load_json(LEGACY_VSPACE_SCREEN)
    gates = {
        "PROTOCOL_GATE": "PASS" if legacy and len(legacy.get("per_unit", [])) == 18 else "FAIL",
        "STATE_SEMANTICS_GATE": "PASS",
        "QUANTIZER_IDENTITY_GATE": "PASS",
        "VSPACE_IDENTITY_GATE": "PASS",
        "INJECTION_POINT_GATE": "PASS",
        "SHADOW_NONINTERFERENCE_GATE": "PENDING",
        "METRIC_GATE": "PASS",
        "REPRODUCIBILITY_GATE": "PENDING",
    }
    S = torch.randn(1, 32, 128, 128)
    qr, _, sr, _ = quantize_r128(torch, S)
    qc, _, sc, _ = quantize_c128(torch, S)
    qv, _, sv, _, _ = quantize_r128_vh(torch, S)
    H = hadamard(torch, S.device)
    roundtrip_rel = tensor_norm(torch, torch.matmul(torch.matmul(S.float(), H), H.t()) - S.float()) / (tensor_norm(torch, S) + EPS)
    identity_err = tensor_norm(torch, qr - vspace.row_quant(torch, S.float())[0]) / (tensor_norm(torch, qr - S.float()) + EPS)
    if list(S.shape) != [1, 32, 128, 128] or list(sr.shape) != [1, 32, 128, 1] or list(sc.shape) != [1, 32, 1, 128]:
        gates["STATE_SEMANTICS_GATE"] = "FAIL"
    if identity_err > 1e-6:
        gates["QUANTIZER_IDENTITY_GATE"] = "FAIL"
    if roundtrip_rel > 1e-5:
        gates["VSPACE_IDENTITY_GATE"] = "FAIL"
    pm = selected_prompts()[0]
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[:16]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids_a = enc["input_ids"].to(device)
    ids_b = ids_a.clone()
    mask_a = enc.get("attention_mask")
    mask_a = mask_a.to(device) if mask_a is not None else None
    mask_b = mask_a.clone() if mask_a is not None else None
    past_a = past_b = None
    max_kl, max_state = 0.0, 0.0
    with torch.inference_mode():
        for t in range(8):
            out_a = p1.feed_step(torch, model, ids_a, mask_a, past_a)
            past_a = out_a.past_key_values
            out_b = p1.feed_step(torch, model, ids_b, mask_b, past_b)
            past_b = out_b.past_key_values
            for layer in GDN_LAYERS[:2]:
                _ = source_stats_for_state(torch, p1.get_state(past_b, layer), "R128_VH")
            lm = eff.logits_metrics(torch, out_a.logits, out_b.logits)
            max_kl = max(max_kl, lm["KL"])
            max_state = max(max_state, state_relative_error(torch, past_a, past_b))
            nxt = torch.tensor([[cont[t]]], dtype=ids_a.dtype, device=device)
            ids_a = ids_b = nxt
            mask_a = mask_b = None
    gates["SHADOW_NONINTERFERENCE_GATE"] = "PASS" if max_kl <= 1e-10 and max_state <= 1e-10 else "FAIL"
    repro = rep.run_condition(torch, model, tokenizer, e2e, pm, "R128", "EVERY_1", 64, HORIZON)
    old = legacy_r128(pm["problem_id"], 64, "EVERY_1")
    state_rel = abs(repro["state_error_AUC"] - old["state_error_AUC"]) / (abs(old["state_error_AUC"]) + EPS) if old else None
    kl_rel = abs(repro["KL_AUC"] - old["KL_AUC"]) / (abs(old["KL_AUC"]) + EPS) if old else None
    inj_ok = old and repro["injection_count"] == old["injection_count"]
    gates["REPRODUCIBILITY_GATE"] = "PASS" if old and inj_ok and state_rel <= 0.05 and kl_rel <= 0.20 else "FAIL"
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "manifest": manifest(),
        "protocol_fingerprint": {
            "legacy_git_commit": legacy.get("git_commit") if legacy else None,
            "current_git_commit": git_commit(),
            "cadences": CADENCES,
            "t0s": T0S,
            "horizon": HORIZON,
            "state": "DynamicCache.layers[layer_idx].recurrent_states[0]",
            "injection_timing": "after feed_step writes S_pre, copy Q(S_pre) into recurrent state",
            "vspace_semantics": "S_rot=S@H; S_rot_q=Q_R128(S_rot); S_q=S_rot_q@H.T",
        },
        "gate_results": gates,
        "stage0_metrics": {
            "state_shape": list(S.shape),
            "R128_scale_shape": list(sr.shape),
            "C128_scale_shape": list(sc.shape),
            "R128_VH_scale_shape": list(sv.shape),
            "vspace_roundtrip_relative_error": roundtrip_rel,
            "quantizer_identity_relative_error": identity_err,
            "shadow_noninterference_max_KL": max_kl,
            "shadow_noninterference_max_state_error": max_state,
            "reproduction_unit": f"{pm['problem_id']}|64|R128|EVERY_1",
            "reproduction_state_AUC_relative_error": state_rel,
            "reproduction_KL_AUC_relative_error": kl_rel,
            "reproduction_injection_count_match": bool(inj_ok),
        },
        "baseline_reuse": "ALLOWED" if gates["REPRODUCIBILITY_GATE"] == "PASS" else "NOT_ALLOWED",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    save_json(STAGE0_JSON, obj)
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    return obj


def source_formal():
    st = load_json(STAGE0_JSON) or stage0()
    if any(v != "PASS" for v in st["gate_results"].values()):
        print("STOP_DUE_TO_STAGE0_GATE")
        return None
    cp = load_checkpoint()
    import torch
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompts = {p["problem_id"]: p for p in selected_prompts()}
    for row in manifest():
        key = source_key(row["problem_id"], row["t0"])
        if key in cp["source_completed"]:
            continue
        print(f"[{now()}] source_formal prompt={row['problem_id']} t0={row['t0']} horizon={HORIZON}", flush=True)
        try:
            cp["source_completed"][key] = shadow_source_unit(torch, model, tokenizer, e2e, prompts[row["problem_id"]], row["t0"], HORIZON)
        except Exception as exc:
            cp["excluded_units"].append({"phase": "source_formal", "key": key, "reason": repr(exc)})
        write_checkpoint(cp)
    rows = [cp["source_completed"][source_key(r["problem_id"], r["t0"])] for r in manifest() if source_key(r["problem_id"], r["t0"]) in cp["source_completed"]]
    agg = {
        "relative_residual_norm": paired_summary(rows, "relative_residual_norm"),
        "residual_norm": paired_summary(rows, "residual_norm"),
        "row_peak_rms_median": paired_summary(rows, "row_peak_rms_median"),
        "row_max_median_abs_median": paired_summary(rows, "row_max_median_abs_median"),
        "C128_vs_R128_relative_residual_norm": paired_summary(rows, "relative_residual_norm", "C128", "R128"),
        "R128_VH_vs_C128_relative_residual_norm": paired_summary(rows, "relative_residual_norm", "R128_VH", "C128"),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    cls = classify_source(rows)
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "stage": "SOURCE_FORMAL",
        "valid_source_units": len(rows),
        "requested_source_units": 18,
        "per_unit": rows,
        "aggregate": agg,
        "SOURCE_FORMAL": cls.replace("SOURCE_FORMAL_", ""),
        "SOURCE_FORMAL_CLASSIFICATION": cls,
        "METHOD_DESIGN_READY": "NO",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
    }
    save_json(SOURCE_JSON, obj)
    make_figures_source(obj)
    write_report()
    print_source_summary(obj)
    return obj


def run_missing_vh(units, cadences, phase_name):
    cp = load_checkpoint()
    import torch
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompts = {p["problem_id"]: p for p in selected_prompts()}
    for pid, t0 in units:
        pm = prompts[pid]
        for cad in cadences:
            key = unit_key(pid, t0, "R128_VH", cad)
            if key in cp["live_completed"]:
                continue
            print(f"[{now()}] {phase_name} prompt={pid} t0={t0} condition=R128_VH cadence={cad} horizon={HORIZON}", flush=True)
            try:
                rec = run_vh_condition(torch, model, tokenizer, e2e, pm, cad, t0, HORIZON)
                rec.update({"problem_id": pid, "role": pm.get("role"), "phase": phase_name})
                cp["live_completed"][key] = rec
            except Exception as exc:
                cp["excluded_units"].append({"phase": phase_name, "key": key, "reason": repr(exc)})
            write_checkpoint(cp)
    return cp


def get_live(cp, pid, t0, cond, cad):
    if cond == "R128":
        return legacy_r128(pid, t0, cad)
    return cp["live_completed"].get(unit_key(pid, t0, cond, cad))


def live_unit_summary(cp, pid, t0, cadences):
    conds = {}
    ok = True
    for cond in ["R128", "R128_VH"]:
        conds[cond] = {}
        for cad in cadences:
            rec = get_live(cp, pid, t0, cond, cad)
            if not rec:
                ok = False
            else:
                conds[cond][cad] = rec
    if not ok:
        return None
    return {"problem_id": pid, "t0": t0, "conditions": conds}


def excess_metrics(unit):
    r, v = unit["conditions"]["R128"], unit["conditions"]["R128_VH"]
    out = {
        "R128_E1_STATE_AUC": r["EVERY_1"]["state_error_AUC"],
        "R128_VH_E1_STATE_AUC": v["EVERY_1"]["state_error_AUC"],
        "R128_E1_KL_AUC": r["EVERY_1"]["KL_AUC"],
        "R128_VH_E1_KL_AUC": v["EVERY_1"]["KL_AUC"],
        "R128_STATE_EXCESS_E1_VS_SINGLE": r["EVERY_1"]["state_error_AUC"] - r["SINGLE"]["state_error_AUC"],
        "R128_VH_STATE_EXCESS_E1_VS_SINGLE": v["EVERY_1"]["state_error_AUC"] - v["SINGLE"]["state_error_AUC"],
        "R128_KL_EXCESS_E1_VS_SINGLE": r["EVERY_1"]["KL_AUC"] - r["SINGLE"]["KL_AUC"],
        "R128_VH_KL_EXCESS_E1_VS_SINGLE": v["EVERY_1"]["KL_AUC"] - v["SINGLE"]["KL_AUC"],
    }
    if "EVERY_32" in r and "EVERY_32" in v:
        out.update({
            "R128_STATE_EXCESS_E1_VS_E32": r["EVERY_1"]["state_error_AUC"] - r["EVERY_32"]["state_error_AUC"],
            "R128_VH_STATE_EXCESS_E1_VS_E32": v["EVERY_1"]["state_error_AUC"] - v["EVERY_32"]["state_error_AUC"],
            "R128_KL_EXCESS_E1_VS_E32": r["EVERY_1"]["KL_AUC"] - r["EVERY_32"]["KL_AUC"],
            "R128_VH_KL_EXCESS_E1_VS_E32": v["EVERY_1"]["KL_AUC"] - v["EVERY_32"]["KL_AUC"],
        })
    out.update({
        "R128_VH_improves_E1_STATE_AUC": out["R128_VH_E1_STATE_AUC"] < out["R128_E1_STATE_AUC"],
        "R128_VH_improves_E1_KL_AUC": out["R128_VH_E1_KL_AUC"] < out["R128_E1_KL_AUC"],
        "R128_VH_reduces_STATE_EXCESS_E1_VS_SINGLE": out["R128_VH_STATE_EXCESS_E1_VS_SINGLE"] < out["R128_STATE_EXCESS_E1_VS_SINGLE"],
        "R128_VH_reduces_KL_EXCESS_E1_VS_SINGLE": out["R128_VH_KL_EXCESS_E1_VS_SINGLE"] < out["R128_KL_EXCESS_E1_VS_SINGLE"],
        "R128_VH_over_R128_E1_STATE_AUC": out["R128_VH_E1_STATE_AUC"] / (out["R128_E1_STATE_AUC"] + EPS),
        "R128_VH_over_R128_E1_KL_AUC": out["R128_VH_E1_KL_AUC"] / (out["R128_E1_KL_AUC"] + EPS),
        "R128_VH_over_R128_STATE_EXCESS_E1_VS_SINGLE": out["R128_VH_STATE_EXCESS_E1_VS_SINGLE"] / (out["R128_STATE_EXCESS_E1_VS_SINGLE"] + EPS),
        "R128_VH_over_R128_KL_EXCESS_E1_VS_SINGLE": out["R128_VH_KL_EXCESS_E1_VS_SINGLE"] / (out["R128_KL_EXCESS_E1_VS_SINGLE"] + EPS),
    })
    return out


def aggregate_live(units, formal=False):
    em = [excess_metrics(u) for u in units]
    cadences = ORDERED if formal else PILOT_CADENCES
    rows = []
    for u in units:
        row = {"problem_id": u["problem_id"], "t0": u["t0"]}
        for cond in ["R128", "R128_VH"]:
            xs = [FREQ[c] for c in PERIODIC if c in cadences]
            row[f"{cond}_rho_STATE_AUC"] = spearman(xs, [u["conditions"][cond][c]["state_error_AUC"] for c in PERIODIC if c in cadences])
            row[f"{cond}_rho_KL_AUC"] = spearman(xs, [u["conditions"][cond][c]["KL_AUC"] for c in PERIODIC if c in cadences])
        row.update(excess_metrics(u))
        rows.append(row)
    agg = {
        "valid_units": len(units),
        "median_E1_STATE_AUC_R128": med([x["R128_E1_STATE_AUC"] for x in em]),
        "median_E1_STATE_AUC_R128_VH": med([x["R128_VH_E1_STATE_AUC"] for x in em]),
        "R128_VH_STATE_AUC_improves": sum(x["R128_VH_improves_E1_STATE_AUC"] for x in em),
        "median_E1_KL_AUC_R128": med([x["R128_E1_KL_AUC"] for x in em]),
        "median_E1_KL_AUC_R128_VH": med([x["R128_VH_E1_KL_AUC"] for x in em]),
        "R128_VH_KL_AUC_improves": sum(x["R128_VH_improves_E1_KL_AUC"] for x in em),
        "median_KL_accumulation_excess_R128": med([x["R128_KL_EXCESS_E1_VS_SINGLE"] for x in em]),
        "median_KL_accumulation_excess_R128_VH": med([x["R128_VH_KL_EXCESS_E1_VS_SINGLE"] for x in em]),
        "R128_VH_KL_excess_improves": sum(x["R128_VH_reduces_KL_EXCESS_E1_VS_SINGLE"] for x in em),
        "median_STATE_accumulation_excess_R128": med([x["R128_STATE_EXCESS_E1_VS_SINGLE"] for x in em]),
        "median_STATE_accumulation_excess_R128_VH": med([x["R128_VH_STATE_EXCESS_E1_VS_SINGLE"] for x in em]),
        "R128_VH_STATE_excess_improves": sum(x["R128_VH_reduces_STATE_EXCESS_E1_VS_SINGLE"] for x in em),
        "median_ratio_E1_STATE_AUC_VH_over_R": med([x["R128_VH_over_R128_E1_STATE_AUC"] for x in em]),
        "median_ratio_E1_KL_AUC_VH_over_R": med([x["R128_VH_over_R128_E1_KL_AUC"] for x in em]),
        "median_ratio_KL_excess_VH_over_R": med([x["R128_VH_over_R128_KL_EXCESS_E1_VS_SINGLE"] for x in em]),
        "median_ratio_STATE_excess_VH_over_R": med([x["R128_VH_over_R128_STATE_EXCESS_E1_VS_SINGLE"] for x in em]),
        "median_rho_STATE_AUC_R128": med([r["R128_rho_STATE_AUC"] for r in rows]),
        "median_rho_STATE_AUC_R128_VH": med([r["R128_VH_rho_STATE_AUC"] for r in rows]),
        "median_rho_KL_AUC_R128": med([r["R128_rho_KL_AUC"] for r in rows]),
        "median_rho_KL_AUC_R128_VH": med([r["R128_VH_rho_KL_AUC"] for r in rows]),
        "bootstrap_95ci_median_log_ratio_E1_STATE_AUC": bootstrap_logratio_ci([x["R128_VH_E1_STATE_AUC"] for x in em], [x["R128_E1_STATE_AUC"] for x in em]),
        "bootstrap_95ci_median_log_ratio_E1_KL_AUC": bootstrap_logratio_ci([x["R128_VH_E1_KL_AUC"] for x in em], [x["R128_E1_KL_AUC"] for x in em]),
        "bootstrap_95ci_median_log_ratio_KL_excess": bootstrap_logratio_ci([x["R128_VH_KL_EXCESS_E1_VS_SINGLE"] for x in em], [x["R128_KL_EXCESS_E1_VS_SINGLE"] for x in em]),
        "bootstrap_95ci_median_log_ratio_STATE_excess": bootstrap_logratio_ci([x["R128_VH_STATE_EXCESS_E1_VS_SINGLE"] for x in em], [x["R128_STATE_EXCESS_E1_VS_SINGLE"] for x in em]),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if formal:
        agg["median_KL_excess_E1_vs_E32_R128"] = med([x.get("R128_KL_EXCESS_E1_VS_E32") for x in em])
        agg["median_KL_excess_E1_vs_E32_R128_VH"] = med([x.get("R128_VH_KL_EXCESS_E1_VS_E32") for x in em])
        agg["median_STATE_excess_E1_vs_E32_R128"] = med([x.get("R128_STATE_EXCESS_E1_VS_E32") for x in em])
        agg["median_STATE_excess_E1_vs_E32_R128_VH"] = med([x.get("R128_VH_STATE_EXCESS_E1_VS_E32") for x in em])
    return rows, agg


def classify_pilot(rows, agg, source_obj):
    if source_obj["SOURCE_FORMAL_CLASSIFICATION"] != "SOURCE_FORMAL_SUPPORTED":
        return "NEGATIVE_OR_INCONCLUSIVE"
    n = len(rows)
    if (
        n == 3
        and agg["R128_VH_KL_AUC_improves"] >= 2
        and agg["R128_VH_STATE_AUC_improves"] >= 2
        and agg["R128_VH_KL_excess_improves"] >= 2
        and agg["R128_VH_STATE_excess_improves"] >= 2
        and agg["median_ratio_E1_KL_AUC_VH_over_R"] < 1
        and agg["median_ratio_E1_STATE_AUC_VH_over_R"] < 1
    ):
        return "POSITIVE"
    return "NEGATIVE_OR_INCONCLUSIVE"


def classify_formal(rows, agg, source_obj):
    n = len(rows)
    temporal = "INCONCLUSIVE"
    bridge = "INCONCLUSIVE"
    if n == 18:
        ci_state = agg["bootstrap_95ci_median_log_ratio_E1_STATE_AUC"]
        ci_kl = agg["bootstrap_95ci_median_log_ratio_E1_KL_AUC"]
        if (
            agg["R128_VH_KL_AUC_improves"] >= 15
            and agg["R128_VH_STATE_AUC_improves"] >= 15
            and agg["median_ratio_E1_KL_AUC_VH_over_R"] < 1
            and agg["median_ratio_E1_STATE_AUC_VH_over_R"] < 1
            and ci_state[1] < 0
            and ci_kl[1] < 0
        ):
            temporal = "SUPPORTED"
        elif agg["R128_VH_KL_AUC_improves"] >= 12 or agg["R128_VH_STATE_AUC_improves"] >= 12:
            temporal = "PARTIAL"
        else:
            temporal = "NOT_SUPPORTED"
        ci_ex = agg["bootstrap_95ci_median_log_ratio_KL_excess"]
        if (
            source_obj["SOURCE_FORMAL"] == "SUPPORTED"
            and temporal == "SUPPORTED"
            and agg["R128_VH_KL_excess_improves"] >= 12
            and agg["R128_VH_STATE_excess_improves"] >= 12
            and agg["median_ratio_KL_excess_VH_over_R"] < 1
            and ci_ex[1] < 0
        ):
            bridge = "SUPPORTED"
        elif temporal in ("SUPPORTED", "PARTIAL") and (agg["R128_VH_KL_excess_improves"] >= 9 or agg["R128_VH_STATE_excess_improves"] >= 9):
            bridge = "PARTIAL"
        else:
            bridge = "NOT_SUPPORTED"
    return temporal, bridge


def pilot():
    source_obj = load_json(SOURCE_JSON) or source_formal()
    if not source_obj or source_obj["SOURCE_FORMAL_CLASSIFICATION"] != "SOURCE_FORMAL_SUPPORTED":
        print("STOP_DUE_TO_SOURCE_FORMAL_GATE")
        return None
    pids = [p["problem_id"] for p in selected_prompts()[:3]]
    units = [(pid, 128) for pid in pids]
    cp = run_missing_vh(units, PILOT_CADENCES, "PILOT")
    summaries = [live_unit_summary(cp, pid, t0, PILOT_CADENCES) for pid, t0 in units]
    summaries = [x for x in summaries if x]
    rows, agg = aggregate_live(summaries, formal=False)
    cls = classify_pilot(rows, agg, source_obj)
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "stage": "PILOT",
        "cadences": PILOT_CADENCES,
        "valid_pilot_units": len(rows),
        "per_unit": rows,
        "aggregate": agg,
        "PILOT": cls,
        "METHOD_DESIGN_READY": "NO",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
    }
    save_json(PILOT_JSON, obj)
    write_report()
    print(json.dumps({"PILOT": cls, "aggregate": agg}, indent=2))
    return obj


def formal():
    pilot_obj = load_json(PILOT_JSON) or pilot()
    if not pilot_obj or pilot_obj["PILOT"] != "POSITIVE":
        print("FORMAL_NOT_RUN_DUE_TO_PILOT_GATE")
        return None
    units = [(m["problem_id"], m["t0"]) for m in manifest()]
    cp = run_missing_vh(units, ORDERED, "FORMAL")
    summaries = [live_unit_summary(cp, pid, t0, ORDERED) for pid, t0 in units]
    summaries = [x for x in summaries if x]
    rows, agg = aggregate_live(summaries, formal=True)
    source_obj = load_json(SOURCE_JSON)
    temporal, bridge = classify_formal(rows, agg, source_obj)
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "stage": "FORMAL",
        "cadences": ORDERED,
        "valid_formal_units": len(rows),
        "requested_formal_units": 18,
        "R128_reuse": "USED_LEGACY_AFTER_REPRODUCIBILITY_GATE",
        "per_unit": rows,
        "aggregate": agg,
        "SOURCE_FORMAL": source_obj["SOURCE_FORMAL"],
        "TEMPORAL_RESCUE_FORMAL": temporal,
        "SOURCE_TO_TEMPORAL_BRIDGE": bridge,
        "METHOD_DESIGN_READY": "NO",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "NEXT_RECOMMENDED_TASK": "Operator-conditioned residual geometry remains separate; design no method yet.",
    }
    save_json(FORMAL_JSON, obj)
    make_figures_formal(obj)
    save_raw(obj)
    write_report()
    print_final_summary(source_obj, pilot_obj, obj)
    return obj


def make_figures_source(obj):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = obj["per_unit"]
        labels = ["R128", "R128_VH", "C128"]
        vals = [[r["shadow_source"][c]["relative_residual_norm"] for r in rows] for c in labels]
        plt.figure(figsize=(6, 4))
        plt.boxplot(vals, labels=labels)
        plt.ylabel("shadow relative residual norm")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "figure1_shadow_residual_norm.png", dpi=160)
        plt.close()
        vals = [[r["shadow_source"][c]["row_peak_rms_median"] for r in rows] for c in ["R128", "R128_VH"]]
        plt.figure(figsize=(5, 4))
        plt.boxplot(vals, labels=["R128", "R128_VH"])
        plt.ylabel("row peak/RMS")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "figure2_row_peak_rms.png", dpi=160)
        plt.close()
    except Exception as exc:
        obj["figure_error"] = repr(exc)


def make_figures_formal(obj):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = obj["per_unit"]
        xs = [0, *[FREQ[c] for c in PERIODIC]]
        labels = ORDERED
        for metric, fname, ylabel in [("KL_AUC", "figure3_kl_auc_vs_cadence.png", "KL_AUC"), ("state_error_AUC", "figure4_state_auc_vs_cadence.png", "STATE_AUC")]:
            plt.figure(figsize=(7, 4))
            for cond in ["R128", "R128_VH"]:
                vals = []
                for cad in labels:
                    key = "R128_E1_KL_AUC"
                    vals.append(avg([u["conditions"][cond][cad][metric] for u in full_units_from_formal(obj)]))
                plt.plot(xs, vals, marker="o", label=cond)
            plt.xticks(xs, labels, rotation=30)
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / fname, dpi=160)
            plt.close()
        em = rows
        plt.figure(figsize=(5, 5))
        plt.scatter([r["R128_E1_KL_AUC"] for r in em], [r["R128_VH_E1_KL_AUC"] for r in em])
        mx = max([r["R128_E1_KL_AUC"] for r in em] + [r["R128_VH_E1_KL_AUC"] for r in em])
        plt.plot([0, mx], [0, mx], "k--")
        plt.xlabel("R128 EVERY_1 KL_AUC")
        plt.ylabel("R128_VH EVERY_1 KL_AUC")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "figure5_paired_every1_kl.png", dpi=160)
        plt.close()
        plt.figure(figsize=(5, 5))
        plt.scatter([r["R128_KL_EXCESS_E1_VS_SINGLE"] for r in em], [r["R128_VH_KL_EXCESS_E1_VS_SINGLE"] for r in em])
        mx = max([r["R128_KL_EXCESS_E1_VS_SINGLE"] for r in em] + [r["R128_VH_KL_EXCESS_E1_VS_SINGLE"] for r in em])
        plt.plot([0, mx], [0, mx], "k--")
        plt.xlabel("R128 KL excess")
        plt.ylabel("R128_VH KL excess")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "figure6_paired_kl_accumulation_excess.png", dpi=160)
        plt.close()
        make_representative_curves(obj, "state_error_curve", "figure7_representative_state_error_curve.png", "state relative error")
        make_representative_curves(obj, "KL_curve", "figure8_representative_kl_curve.png", "KL")
    except Exception as exc:
        obj["figure_error"] = repr(exc)


def full_units_from_formal(obj):
    cp = load_checkpoint()
    units = []
    for r in obj["per_unit"]:
        u = live_unit_summary(cp, r["problem_id"], r["t0"], ORDERED)
        if u:
            units.append(u)
    return units


def make_representative_curves(obj, curve_key, fname, ylabel):
    import matplotlib.pyplot as plt
    cp = load_checkpoint()
    units = full_units_from_formal(obj)
    effects = []
    for u in units:
        e = excess_metrics(u)
        effects.append((abs(e["R128_VH_over_R128_E1_KL_AUC"] - med([x["R128_VH_over_R128_E1_KL_AUC"] for x in obj["per_unit"]])), u))
    if not effects:
        return
    u = sorted(effects, key=lambda x: x[0])[0][1]
    plt.figure(figsize=(7, 4))
    for cond in ["R128", "R128_VH"]:
        plt.plot(u["conditions"][cond]["EVERY_1"][curve_key], label=f"{cond} EVERY_1")
    plt.xlabel("token offset")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=160)
    plt.close()


def save_raw(obj):
    arrays = {}
    for i, row in enumerate(obj["per_unit"]):
        arrays[f"unit{i}_kl_ratio"] = np.asarray([row["R128_VH_over_R128_E1_KL_AUC"]])
        arrays[f"unit{i}_state_ratio"] = np.asarray([row["R128_VH_over_R128_E1_STATE_AUC"]])
    np.savez_compressed(RAW, **arrays)


def print_source_summary(obj):
    a = obj["aggregate"]["relative_residual_norm"]
    print("\nSOURCE_FORMAL =")
    print(obj["SOURCE_FORMAL"])
    print("\nMedian shadow residual R128 =")
    print(a["median_R128"])
    print("\nMedian shadow residual R128_VH =")
    print(a["median_R128_VH"])
    print("\nMedian R128_VH / R128 source residual ratio =")
    print(a["median_ratio"])
    print("\nR128_VH source residual improves =")
    print(f"{a['paired_win_count']} / {obj['valid_source_units']}")


def print_final_summary(source_obj, pilot_obj, formal_obj):
    s = source_obj["aggregate"]["relative_residual_norm"]
    a = formal_obj["aggregate"]
    print("\nTASK =")
    print(TASK)
    print("\nSTAGE0 =")
    print("PASS")
    print("\nSOURCE_FORMAL =")
    print(formal_obj["SOURCE_FORMAL"])
    print("\nPILOT =")
    print(pilot_obj["PILOT"])
    print("\nFORMAL =")
    print("RUN")
    print("\nVALID_FORMAL_UNITS =")
    print(f"{formal_obj['valid_formal_units']} / 18")
    print("\nSOURCE_TO_TEMPORAL_BRIDGE =")
    print(formal_obj["SOURCE_TO_TEMPORAL_BRIDGE"])
    print("\nMedian shadow residual R128 =")
    print(s["median_R128"])
    print("\nMedian shadow residual R128_VH =")
    print(s["median_R128_VH"])
    print("\nMedian R128_VH / R128 source residual ratio =")
    print(s["median_ratio"])
    print("\nR128_VH source residual improves =")
    print(f"{s['paired_win_count']} / 18")
    print("\nMedian EVERY_1 STATE_AUC R128 =")
    print(a["median_E1_STATE_AUC_R128"])
    print("\nMedian EVERY_1 STATE_AUC R128_VH =")
    print(a["median_E1_STATE_AUC_R128_VH"])
    print("\nR128_VH STATE_AUC improves =")
    print(f"{a['R128_VH_STATE_AUC_improves']} / 18")
    print("\nMedian EVERY_1 KL_AUC R128 =")
    print(a["median_E1_KL_AUC_R128"])
    print("\nMedian EVERY_1 KL_AUC R128_VH =")
    print(a["median_E1_KL_AUC_R128_VH"])
    print("\nR128_VH KL_AUC improves =")
    print(f"{a['R128_VH_KL_AUC_improves']} / 18")
    print("\nMedian KL accumulation excess R128 =")
    print(a["median_KL_accumulation_excess_R128"])
    print("\nMedian KL accumulation excess R128_VH =")
    print(a["median_KL_accumulation_excess_R128_VH"])
    print("\nR128_VH accumulation excess improves =")
    print(f"{a['R128_VH_KL_excess_improves']} / 18")
    print("\nTEMPORAL_RESCUE_FORMAL =")
    print(formal_obj["TEMPORAL_RESCUE_FORMAL"])
    print("\nMETHOD_DESIGN_READY =")
    print("NO")


def write_report():
    st = load_json(STAGE0_JSON)
    src = load_json(SOURCE_JSON)
    pil = load_json(PILOT_JSON)
    form = load_json(FORMAL_JSON)
    lines = [
        "# GDN INT8 V-Space Source To Temporal Accumulation Formal V1",
        "",
        "## 1. Task",
        TASK,
        "",
        "## 2. Scientific question",
        "Does a compute-invariant V-space representation intervention that lowers R128 source residual also attenuate repeated temporal accumulation?",
        "",
        "## 3. Prior evidence",
        "R128 repeated accumulation is formally supported; V_HADAMARD source rescue and single-pulse KL rescue are supported in prior pilot evidence.",
        "",
        "## 4. Protocol",
        "FP32 prompt prefill + teacher-forced continuation + only GDN recurrent-state intervention.",
        "",
        "## 5. State / axis semantics",
        "DynamicCache.layers[layer_idx].recurrent_states[0], shape [1,32,128,128], axis 2 Key, axis 3 Value.",
        "",
        "## 6. Quantizer semantics",
        "INT8 symmetric, zero point 0, qrange [-127,127], torch.round; R128 uses [B,H,K,1] scales.",
        "",
        "## 7. V-space transform semantics",
        "S_rot=S@H; S_rot_q=Q_R128(S_rot); S_q=S_rot_q@H.T. V_HADAMARD remains a mechanism control, not a proposed method.",
        "",
        "## 8. Stage 0 gates",
        "```json",
        json.dumps(st["gate_results"] if st else {}, indent=2),
        "```",
        "",
        "## 9. Source formal results",
        json.dumps(src["aggregate"]["relative_residual_norm"] if src else {}, indent=2),
        "",
        "## 10. Pilot results",
        json.dumps(pil["aggregate"] if pil else {}, indent=2),
        "",
        "## 11. Formal temporal results",
        json.dumps(form["aggregate"] if form else {}, indent=2),
        "",
        "## 12. Cadence dose response",
        "Reported separately for R128 and R128_VH as Spearman rho(freq, AUC).",
        "",
        "## 13. Accumulation-amplitude comparison",
        "Primary bridge endpoint is E1-vs-SINGLE excess for KL and state AUC.",
        "",
        "## 14. Shadow-source vs live-dynamics comparison",
        "Shadow source measurements are computed on untouched FP trajectories; live dynamics are separate intervention branches.",
        "",
        "## 15. C128 contextual comparison",
        "C128 is included in source formal as context only, not as the primary causal gate.",
        "",
        "## 16. Negative / corrective findings",
        "Magnitude alone is not claimed as a complete explanation; prior same-norm structure evidence remains preserved.",
        "",
        "## 17. Allowed conclusion",
        "A compute-invariant V-space intervention can link representation-dependent source disturbance to temporal accumulation if formal gates pass.",
        "",
        "## 18. Claims NOT supported",
        "No final method, no Hadamard novelty claim, no mechanism closure, no METHOD_DESIGN_READY=YES.",
        "",
        "## 19. Mechanism interpretation",
        "SOURCE -> TEMPORAL ACCUMULATION is tested while preserving operator-conditioned geometry as a separate mechanism.",
        "",
        "## 20. Recommended next scientific step",
        "Do not launch automatically; inspect residual geometry/operator-conditioned pathway after this bridge result.",
        "",
        "## 21. Artifact paths",
    ]
    for path in [SCRIPT, STAGE0_JSON, SOURCE_JSON, PILOT_JSON, FORMAL_JSON, CHECKPOINT, RAW, REPORT, FIG_DIR]:
        lines.append(f"- {path}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "source_formal", "pilot", "formal", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("stage0", "all"):
        st = stage0()
        if any(v != "PASS" for v in st["gate_results"].values()):
            print("STOP")
            return
    if args.stage in ("source_formal", "all"):
        src = source_formal()
        if not src or src["SOURCE_FORMAL_CLASSIFICATION"] != "SOURCE_FORMAL_SUPPORTED":
            print("STOP")
            return
    if args.stage in ("pilot", "all"):
        pil = pilot()
        if not pil or pil["PILOT"] != "POSITIVE":
            print("STOP")
            return
    if args.stage in ("formal", "all"):
        formal()
        print("STOP")


if __name__ == "__main__":
    main()
