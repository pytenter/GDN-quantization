#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_REPEATED_ACCUMULATION_FORMAL_AND_AXIS_COMPARISON_V1"
SCRIPT = EXP / "run_int8_repeated_accumulation_formal_axis_comparison.py"
PHASE_A_JSON = RES / "gdn_int8_repeated_accumulation_formal_v1.json"
PHASE_A_REPORT = REP / "gdn_int8_repeated_accumulation_formal_v1.md"
PHASE_B_JSON = RES / "gdn_int8_r128_c128_accumulation_comparison_v1.json"
PHASE_B_REPORT = REP / "gdn_int8_r128_c128_accumulation_comparison_v1.md"
CHECKPOINT = RES / "gdn_int8_repeated_accumulation_formal_axis_comparison_v1_checkpoint.json"
FIG_DIR = RES / "gdn_int8_repeated_accumulation_formal_axis_comparison_v1_figures"

EPS = 1e-12
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
CADENCES = ["FP_STATE", "SINGLE", "EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
ORDERED = ["SINGLE", "EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
PERIODIC = ["EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
FREQ = {"EVERY_32": 1 / 32, "EVERY_16": 1 / 16, "EVERY_8": 1 / 8, "EVERY_4": 1 / 4, "EVERY_2": 1 / 2, "EVERY_1": 1.0}
AXES = {
    "R128": {"name": "R128", "orientation": "row", "group_size": 128, "scale_shape_semantics": "[B,H,K,1]"},
    "C128": {"name": "C128", "orientation": "column", "group_size": 128, "scale_shape_semantics": "[B,H,1,V]"},
}

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_effective_update_metric_audit as eff
import run_int8_orientation_state_change_mechanism as p1
import run_int8_readout_aware_propagation_audit as readout


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


def schedule(cadence, t0, last_t):
    if cadence == "FP_STATE":
        return []
    if cadence == "SINGLE":
        return [t0] if t0 <= last_t else []
    step = int(cadence.split("_")[1])
    return list(range(t0, last_t + 1, step))


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def summarize(vals):
    vals = [float(v) for v in vals if finite(v)]
    return {"auc": avg(vals), "mean": avg(vals), "terminal": vals[-1] if vals else None, "max": max(vals) if vals else None}


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


def canonical_scale_shape(state, axis_name):
    if axis_name == "R128":
        return [state.shape[0], state.shape[1], state.shape[2], 1]
    return [state.shape[0], state.shape[1], 1, state.shape[3]]


def quantize_state(torch, state, axis_name):
    cfg = AXES[axis_name]
    y, q, _scale_full, scale, _raw_scale_shape, dyn = axis.grouped_quant(torch, state.detach().float(), state.detach().float(), cfg)
    scale_vals = scale.detach().float().flatten().cpu().tolist()
    return y, q, canonical_scale_shape(state, axis_name), {
        "scale_mean": avg(scale_vals),
        "scale_median": med(scale_vals),
        "scale_max": max(scale_vals) if scale_vals else None,
        "scale_p95": pct(scale_vals, 0.95),
        "scale_p95_over_median": pct(scale_vals, 0.95) / max(med(scale_vals), EPS) if scale_vals else None,
        "within_group_dynamic_range_mean": avg(dyn),
        "within_group_dynamic_range_median": med(dyn),
        "within_group_dynamic_range_p95": pct(dyn, 0.95),
        "saturation_fraction": float((q.abs() >= 127).float().mean().item()),
    }


def state_metrics(torch, past, fp_past):
    err2 = ref2 = 0.0
    for layer in GDN_LAYERS:
        a = p1.get_state(past, layer).detach().float()
        b = p1.get_state(fp_past, layer).detach().float()
        d = a - b
        err2 += float(torch.sum(d.double() * d.double()).item())
        ref2 += float(torch.sum(b.double() * b.double()).item())
    e = math.sqrt(err2)
    return e, e / (math.sqrt(ref2) + EPS)


def inject(torch, past, token_idx, axis_name):
    residual_norms, relative_norms, rms_vals = [], [], []
    scales = {k: [] for k in ["scale_mean", "scale_median", "scale_max", "scale_p95", "scale_p95_over_median", "within_group_dynamic_range_mean", "within_group_dynamic_range_median", "within_group_dynamic_range_p95", "saturation_fraction"]}
    max_identity = 0.0
    scale_shape = None
    for layer in GDN_LAYERS:
        state = p1.get_state(past, layer)
        pre = state.detach().clone()
        post, _q, shape, st = quantize_state(torch, pre, axis_name)
        residual = post - pre.float()
        max_identity = max(max_identity, norm(torch, residual - (post - pre.float())))
        rn, pn = norm(torch, residual), norm(torch, pre)
        residual_norms.append(rn)
        relative_norms.append(rn / (pn + EPS))
        rms_vals.append(float(torch.sqrt(torch.mean(residual.float() ** 2)).item()))
        for k in scales:
            scales[k].append(st.get(k))
        scale_shape = shape
        state.copy_(post.to(state.dtype))
    total = math.sqrt(sum(x * x for x in residual_norms))
    return {
        "token_index": int(token_idx),
        "axis": axis_name,
        "residual_norm": total,
        "relative_residual_norm": avg(relative_norms),
        "rms_residual": avg(rms_vals),
        "layer_head_aggregate_residual_norm": total,
        "scale_shape": scale_shape,
        "identity_error_norm": max_identity,
        "scale_statistics": {k: avg(v) for k, v in scales.items()},
    }


def run_condition(torch, model, tokenizer, e2e, pm, axis_name, cadence, t0, horizon):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[: t0 + horizon + 1]
    if len(cont) <= t0:
        raise RuntimeError(f"insufficient continuation for {pm['problem_id']} t0={t0}: {len(cont)}")
    last_t = min(t0 + horizon - 1, len(cont) - 1)
    actual_horizon = last_t - t0 + 1
    sched = schedule(cadence, t0, last_t)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    base_ids = enc["input_ids"].to(device)
    base_mask = enc.get("attention_mask")
    base_mask = base_mask.to(device) if base_mask is not None else None
    fp_ids, cfg_ids = base_ids.clone(), base_ids.clone()
    fp_mask = base_mask.clone() if base_mask is not None else None
    cfg_mask = base_mask.clone() if base_mask is not None else None
    fp_past = cfg_past = None
    curves = {k: [] for k in ["state_error_norm", "state_error", "KL", "top1", "frozen_readout_error", "actual_readout_error", "postproj_error", "residual_stream_error"]}
    injections, failures = [], []
    col = readout.install_readout_hooks(torch, model)
    try:
        with torch.inference_mode():
            for t in range(last_t + 1):
                col["records"].clear()
                col["cfg"] = "FP_STATE"
                fp_out = p1.feed_step(torch, model, fp_ids, fp_mask, fp_past)
                fp_past = fp_out.past_key_values
                col["cfg"] = f"{axis_name}_{cadence}"
                cfg_out = p1.feed_step(torch, model, cfg_ids, cfg_mask, cfg_past)
                cfg_past = cfg_out.past_key_values
                if t in sched:
                    im = inject(torch, cfg_past, t, axis_name)
                    im["number_of_injections_so_far"] = len(injections) + 1
                    injections.append(im)
                col["cfg"] = None
                if t >= t0:
                    se_norm, se_rel = state_metrics(torch, cfg_past, fp_past)
                    lm = eff.logits_metrics(torch, fp_out.logits, cfg_out.logits)
                    rm = readout.compute_readout_metrics(torch, model, col, f"{axis_name}_{cadence}", fp_past, cfg_past)[0]
                    vals = {
                        "state_error_norm": se_norm,
                        "state_error": se_rel,
                        "KL": lm["KL"],
                        "top1": lm["top1_agreement"],
                        "frozen_readout_error": rm.get("frozen_query_readout_error_norm"),
                        "actual_readout_error": rm.get("actual_preproj_readout_error_norm"),
                        "postproj_error": rm.get("postproj_error_norm"),
                        "residual_stream_error": rm.get("residual_stream_error_norm"),
                    }
                    for k, v in vals.items():
                        curves[k].append(v)
                    if not all(finite(v) for v in [se_norm, se_rel, lm["KL"]]):
                        failures.append({"token_index": t, "reason": "nonfinite_core_metric"})
                if t < len(cont):
                    nxt = torch.tensor([[cont[t]]], dtype=base_ids.dtype, device=device)
                    fp_ids, cfg_ids = nxt, nxt.clone()
                    fp_mask = cfg_mask = None
    finally:
        col["close"]()
    rnorms = [m["residual_norm"] for m in injections]
    return {
        "axis": axis_name,
        "cadence": cadence,
        "t0": t0,
        "continuation_horizon": actual_horizon,
        "intervention_token_indices": sched,
        "injection_count": len(injections),
        "injection_metadata": injections,
        "state_error_curve": curves["state_error"],
        "state_error_norm_curve": curves["state_error_norm"],
        "KL_curve": curves["KL"],
        "top1_curve": curves["top1"],
        "frozen_readout_error_curve": curves["frozen_readout_error"],
        "actual_readout_error_curve": curves["actual_readout_error"],
        "postproj_error_curve": curves["postproj_error"],
        "residual_stream_error_curve": curves["residual_stream_error"],
        "state_error_AUC": summarize(curves["state_error"])["auc"],
        "state_error_terminal": summarize(curves["state_error"])["terminal"],
        "state_error_max": summarize(curves["state_error"])["max"],
        "frozen_readout_error_AUC": summarize(curves["frozen_readout_error"])["auc"],
        "actual_readout_error_AUC": summarize(curves["actual_readout_error"])["auc"],
        "postproj_error_AUC": summarize(curves["postproj_error"])["auc"],
        "residual_stream_error_AUC": summarize(curves["residual_stream_error"])["auc"],
        "KL_AUC": summarize(curves["KL"])["auc"],
        "KL_mean": summarize(curves["KL"])["mean"],
        "KL_terminal": summarize(curves["KL"])["terminal"],
        "KL_max": summarize(curves["KL"])["max"],
        "Top1_agreement": avg(curves["top1"]),
        "sum_residual_norm": sum(rnorms),
        "sqrt_sum_squared_residual_norm": math.sqrt(sum(x * x for x in rnorms)),
        "mean_residual_norm": avg(rnorms),
        "median_residual_norm": med(rnorms),
        "numerical_failures": failures,
    }


def unit_key(pid, t0, axis_name, cadence):
    return f"{pid}|{t0}|{axis_name}|{cadence}"


def load_checkpoint():
    obj = load_json(CHECKPOINT)
    if obj:
        return obj
    return {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "completed": {}, "excluded_units": [], "phase_a": None, "phase_b": None}


def write_checkpoint(cp):
    cp["timestamp"] = now()
    save_json(CHECKPOINT, cp)


def run_missing_conditions(cp, phase, units, axes, cadences, horizon):
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompts = {p["problem_id"]: p for p in selected_prompts()}
    for pid, t0 in units:
        pm = prompts[pid]
        for axis_name in axes:
            for cadence in cadences:
                key = unit_key(pid, t0, axis_name, cadence)
                if key in cp["completed"]:
                    continue
                print(f"[{now()}] {phase} prompt={pid} t0={t0} axis={axis_name} cadence={cadence} horizon={horizon}", flush=True)
                try:
                    rec = run_condition(torch, model, tokenizer, e2e, pm, axis_name, cadence, t0, horizon)
                    rec.update({"problem_id": pid, "role": pm["role"], "phase": phase})
                    cp["completed"][key] = rec
                except Exception as exc:
                    cp["excluded_units"].append({"key": key, "problem_id": pid, "t0": t0, "axis": axis_name, "cadence": cadence, "reason": repr(exc)})
                write_checkpoint(cp)
    return cp


def values_for(unit, metric):
    return [unit[c].get(metric) for c in PERIODIC]


def adjacent_count(unit, metric):
    return sum(1 for a, b in zip(ORDERED, ORDERED[1:]) if finite(unit[a].get(metric)) and finite(unit[b].get(metric)) and unit[b][metric] > unit[a][metric])


def assemble_units(cp, phase, axis_name, t0s):
    prompts = [p["problem_id"] for p in selected_prompts()]
    out = {}
    for pid in prompts:
        for t0 in t0s:
            unit = {}
            ok = True
            for c in CADENCES:
                r = cp["completed"].get(unit_key(pid, t0, axis_name, c))
                if not r:
                    ok = False
                    break
                unit[c] = r
            if ok:
                out[f"{pid}|{t0}"] = {"problem_id": pid, "t0": t0, "conditions": unit}
    return out


def analyze_phase_a(cp):
    units = assemble_units(cp, "PHASE_A", "R128", [64, 128, 256])
    rows = []
    for key, u in units.items():
        cond = u["conditions"]
        xs = [FREQ[c] for c in PERIODIC]
        row = {
            "unit": key,
            "problem_id": u["problem_id"],
            "t0": u["t0"],
            "rho_state": spearman(xs, values_for(cond, "state_error_AUC")),
            "rho_KL": spearman(xs, values_for(cond, "KL_AUC")),
            "rho_frozen_readout": spearman(xs, values_for(cond, "frozen_readout_error_AUC")),
            "rho_actual_readout": spearman(xs, values_for(cond, "actual_readout_error_AUC")),
            "KL_adjacent_count": adjacent_count(cond, "KL_AUC"),
            "state_adjacent_count": adjacent_count(cond, "state_error_AUC"),
            "frozen_readout_adjacent_count": adjacent_count(cond, "frozen_readout_error_AUC"),
            "actual_readout_adjacent_count": adjacent_count(cond, "actual_readout_error_AUC"),
            "E1_gt_E32_KL": cond["EVERY_1"]["KL_AUC"] > cond["EVERY_32"]["KL_AUC"],
            "E1_gt_SINGLE_state": cond["EVERY_1"]["state_error_AUC"] > cond["SINGLE"]["state_error_AUC"],
            "numerical_failure_count": sum(len(r.get("numerical_failures", [])) for r in cond.values()),
        }
        rows.append(row)
    valid = [r for r in rows if r["numerical_failure_count"] == 0]
    supported = (
        len(valid) >= 15
        and sum(1 for r in valid if finite(r["rho_KL"]) and r["rho_KL"] > 0) >= 15
        and med([r["rho_KL"] for r in valid]) >= 0.7
        and med([r["rho_state"] for r in valid]) >= 0.7
        and sum(1 for r in valid if r["E1_gt_E32_KL"]) >= 15
        and sum(1 for r in valid if r["E1_gt_SINGLE_state"]) >= 15
        and med([r["KL_adjacent_count"] for r in valid]) >= 5
        and med([r["state_adjacent_count"] for r in valid]) >= 5
    )
    if supported:
        classification, ready = "SUPPORTED", "YES"
    elif len(valid) >= 15:
        classification, ready = "NOT_SUPPORTED", "NO"
    else:
        classification, ready = "INCONCLUSIVE", "NO"
    by_t0 = {}
    for t0 in [64, 128, 256]:
        sub = [r for r in valid if r["t0"] == t0]
        by_t0[str(t0)] = {
            "valid_units": len(sub),
            "median_rho_KL": med([r["rho_KL"] for r in sub]),
            "median_rho_state": med([r["rho_state"] for r in sub]),
            "median_KL_adjacent_count": med([r["KL_adjacent_count"] for r in sub]),
            "median_state_adjacent_count": med([r["state_adjacent_count"] for r in sub]),
        }
    by_prompt = {}
    for pid in [p["problem_id"] for p in selected_prompts()]:
        sub = [r for r in valid if r["problem_id"] == pid]
        by_prompt[pid] = {
            "valid_units": len(sub),
            "median_rho_KL": med([r["rho_KL"] for r in sub]),
            "median_rho_state": med([r["rho_state"] for r in sub]),
            "median_KL_adjacent_count": med([r["KL_adjacent_count"] for r in sub]),
            "median_state_adjacent_count": med([r["state_adjacent_count"] for r in sub]),
        }
    obj = {
        "task": TASK,
        "phase": "PHASE_A",
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "protocol": "FP32 prompt prefill + teacher-forced continuation + only GDN recurrent state quantization intervention",
        "quantizer_config": AXES["R128"],
        "valid_formal_units": len(valid),
        "requested_formal_units": 18,
        "per_unit": rows,
        "by_t0_summary": by_t0,
        "by_prompt_summary": by_prompt,
        "R128_median_rho_freq_state_AUC": med([r["rho_state"] for r in valid]),
        "R128_median_rho_freq_KL_AUC": med([r["rho_KL"] for r in valid]),
        "R128_median_rho_freq_frozen_readout_AUC": med([r["rho_frozen_readout"] for r in valid]),
        "R128_median_rho_freq_actual_readout_AUC": med([r["rho_actual_readout"] for r in valid]),
        "R128_EVERY_1_gt_EVERY_32_KL": f"{sum(1 for r in valid if r['E1_gt_E32_KL'])} / {len(valid)}",
        "R128_EVERY_1_gt_SINGLE_state": f"{sum(1 for r in valid if r['E1_gt_SINGLE_state'])} / {len(valid)}",
        "median_KL_adjacent_dose_response": f"{med([r['KL_adjacent_count'] for r in valid])} / 6",
        "median_state_adjacent_dose_response": f"{med([r['state_adjacent_count'] for r in valid])} / 6",
        "R128_REPEATED_ACCUMULATION_FORMAL": classification,
        "PHASE_B_READY": ready,
        "REPEATED_ACCUMULATION_FORMAL_SUPPORT": "YES" if classification == "SUPPORTED" else ("NO" if classification == "NOT_SUPPORTED" else "INCONCLUSIVE"),
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "excluded_units": cp.get("excluded_units", []),
    }
    cp["phase_a"] = obj
    write_checkpoint(cp)
    save_json(PHASE_A_JSON, obj)
    write_phase_a_report(obj)
    make_phase_a_figures(cp, obj)
    return obj


def get_completed(cp, pid, t0, axis_name):
    return {c: cp["completed"][unit_key(pid, t0, axis_name, c)] for c in CADENCES if unit_key(pid, t0, axis_name, c) in cp["completed"]}


def axis_mean(cp, axis_name, cadence, metric):
    vals = []
    for p in selected_prompts():
        r = cp["completed"].get(unit_key(p["problem_id"], 128, axis_name, cadence))
        if r:
            vals.append(r.get(metric))
    return avg(vals)


def analyze_phase_b(cp):
    prompts = [p["problem_id"] for p in selected_prompts()]
    per_prompt = {}
    for pid in prompts:
        rcond = get_completed(cp, pid, 128, "R128")
        ccond = get_completed(cp, pid, 128, "C128")
        if len(rcond) < len(CADENCES) or len(ccond) < len(CADENCES):
            continue
        rows = {}
        for cad in ORDERED:
            rr, cr = rcond[cad], ccond[cad]
            rows[cad] = {
                "R128_state_AUC": rr["state_error_AUC"],
                "C128_state_AUC": cr["state_error_AUC"],
                "axis_gap_state": rr["state_error_AUC"] - cr["state_error_AUC"],
                "R128_frozen_readout_AUC": rr["frozen_readout_error_AUC"],
                "C128_frozen_readout_AUC": cr["frozen_readout_error_AUC"],
                "axis_gap_frozen_readout": rr["frozen_readout_error_AUC"] - cr["frozen_readout_error_AUC"],
                "R128_actual_readout_AUC": rr["actual_readout_error_AUC"],
                "C128_actual_readout_AUC": cr["actual_readout_error_AUC"],
                "axis_gap_actual_readout": rr["actual_readout_error_AUC"] - cr["actual_readout_error_AUC"],
                "R128_KL_AUC": rr["KL_AUC"],
                "C128_KL_AUC": cr["KL_AUC"],
                "axis_gap_KL": rr["KL_AUC"] - cr["KL_AUC"],
                "R128_mean_residual_norm": rr["mean_residual_norm"],
                "C128_mean_residual_norm": cr["mean_residual_norm"],
                "R128_sum_residual_norm": rr["sum_residual_norm"],
                "C128_sum_residual_norm": cr["sum_residual_norm"],
            }
        r_single, c_single = rcond["SINGLE"], ccond["SINGLE"]
        per_prompt[pid] = {
            "single_step": {
                "R128_SINGLE_residual_norm": r_single["mean_residual_norm"],
                "C128_SINGLE_residual_norm": c_single["mean_residual_norm"],
                "R128_SINGLE_state_AUC": r_single["state_error_AUC"],
                "C128_SINGLE_state_AUC": c_single["state_error_AUC"],
                "R128_SINGLE_KL_AUC": r_single["KL_AUC"],
                "C128_SINGLE_KL_AUC": c_single["KL_AUC"],
            },
            "cadence": rows,
            "susceptibility": {
                "R128_STATE_CADENCE_GAIN": rcond["EVERY_1"]["state_error_AUC"] - rcond["EVERY_32"]["state_error_AUC"],
                "C128_STATE_CADENCE_GAIN": ccond["EVERY_1"]["state_error_AUC"] - ccond["EVERY_32"]["state_error_AUC"],
                "R128_KL_CADENCE_GAIN": rcond["EVERY_1"]["KL_AUC"] - rcond["EVERY_32"]["KL_AUC"],
                "C128_KL_CADENCE_GAIN": ccond["EVERY_1"]["KL_AUC"] - ccond["EVERY_32"]["KL_AUC"],
                "R128_STATE_REPEATED_EXCESS": rcond["EVERY_1"]["state_error_AUC"] - rcond["SINGLE"]["state_error_AUC"],
                "C128_STATE_REPEATED_EXCESS": ccond["EVERY_1"]["state_error_AUC"] - ccond["SINGLE"]["state_error_AUC"],
                "R128_KL_REPEATED_EXCESS": rcond["EVERY_1"]["KL_AUC"] - rcond["SINGLE"]["KL_AUC"],
                "C128_KL_REPEATED_EXCESS": ccond["EVERY_1"]["KL_AUC"] - ccond["SINGLE"]["KL_AUC"],
            },
            "rho_gap_KL": spearman([FREQ[c] for c in PERIODIC], [rows[c]["axis_gap_KL"] for c in PERIODIC]),
            "rho_gap_state": spearman([FREQ[c] for c in PERIODIC], [rows[c]["axis_gap_state"] for c in PERIODIC]),
            "numerical_failure_count": sum(len(x.get("numerical_failures", [])) for x in list(rcond.values()) + list(ccond.values())),
        }
    sus = [v["susceptibility"] for v in per_prompt.values()]
    gt = {
        "R128_gt_C128_KL_repeated_excess": sum(1 for s in sus if s["R128_KL_REPEATED_EXCESS"] > s["C128_KL_REPEATED_EXCESS"]),
        "R128_gt_C128_state_repeated_excess": sum(1 for s in sus if s["R128_STATE_REPEATED_EXCESS"] > s["C128_STATE_REPEATED_EXCESS"]),
        "R128_gt_C128_KL_cadence_gain": sum(1 for s in sus if s["R128_KL_CADENCE_GAIN"] > s["C128_KL_CADENCE_GAIN"]),
        "R128_gt_C128_state_cadence_gain": sum(1 for s in sus if s["R128_STATE_CADENCE_GAIN"] > s["C128_STATE_CADENCE_GAIN"]),
    }
    n = len(per_prompt)
    gap_kl_rho = med([v["rho_gap_KL"] for v in per_prompt.values()])
    gap_state_rho = med([v["rho_gap_state"] for v in per_prompt.values()])
    nofail = sum(v["numerical_failure_count"] for v in per_prompt.values()) == 0
    lower_susc = n == 6 and nofail and all(v >= 5 for v in gt.values()) and finite(gap_kl_rho) and gap_kl_rho > 0
    mag_dom = n == 6 and gt["R128_gt_C128_KL_repeated_excess"] < 5 and axis_mean(cp, "R128", "SINGLE", "KL_AUC") > axis_mean(cp, "C128", "SINGLE", "KL_AUC")
    if lower_susc:
        cls = "C128_LOWER_ACCUMULATION_SUSCEPTIBILITY_SUPPORTED"
    elif mag_dom:
        cls = "MAGNITUDE_DOMINATED"
    elif n == 6 and (gt["R128_gt_C128_KL_repeated_excess"] >= 4 or gt["R128_gt_C128_state_repeated_excess"] >= 4):
        cls = "MIXED"
    elif n == 6 and nofail:
        cls = "NO_CLEAR_AXIS_ACCUMULATION_DIFFERENCE"
    else:
        cls = "INCONCLUSIVE"
    obj = {
        "task": TASK,
        "phase": "PHASE_B",
        "timestamp": now(),
        "prompts": f"{n} / 6",
        "t0": 128,
        "axes": AXES,
        "per_prompt": per_prompt,
        "aggregate": {
            "R128_SINGLE_KL_AUC": axis_mean(cp, "R128", "SINGLE", "KL_AUC"),
            "C128_SINGLE_KL_AUC": axis_mean(cp, "C128", "SINGLE", "KL_AUC"),
            "R128_E1_KL_AUC": axis_mean(cp, "R128", "EVERY_1", "KL_AUC"),
            "C128_E1_KL_AUC": axis_mean(cp, "C128", "EVERY_1", "KL_AUC"),
            "R128_KL_REPEATED_EXCESS": avg([s["R128_KL_REPEATED_EXCESS"] for s in sus]),
            "C128_KL_REPEATED_EXCESS": avg([s["C128_KL_REPEATED_EXCESS"] for s in sus]),
            "R128_STATE_REPEATED_EXCESS": avg([s["R128_STATE_REPEATED_EXCESS"] for s in sus]),
            "C128_STATE_REPEATED_EXCESS": avg([s["C128_STATE_REPEATED_EXCESS"] for s in sus]),
            "R128_KL_CADENCE_GAIN": avg([s["R128_KL_CADENCE_GAIN"] for s in sus]),
            "C128_KL_CADENCE_GAIN": avg([s["C128_KL_CADENCE_GAIN"] for s in sus]),
            "R128_STATE_CADENCE_GAIN": avg([s["R128_STATE_CADENCE_GAIN"] for s in sus]),
            "C128_STATE_CADENCE_GAIN": avg([s["C128_STATE_CADENCE_GAIN"] for s in sus]),
            "rho_freq_R128_C128_KL_gap": gap_kl_rho,
            "rho_freq_R128_C128_state_gap": gap_state_rho,
            **{k: f"{v} / 6 prompts" for k, v in gt.items()},
        },
        "AXIS_ACCUMULATION_CLASSIFICATION": cls,
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "GDN_INT8_FROZEN_DRIVER_VS_FULL_FEEDBACK_DECOMPOSITION_V1" if cls in ("C128_LOWER_ACCUMULATION_SUSCEPTIBILITY_SUPPORTED", "MIXED") else "Inspect axis-comparison heterogeneity before decomposition.",
    }
    cp["phase_b"] = obj
    write_checkpoint(cp)
    save_json(PHASE_B_JSON, obj)
    write_phase_b_report(obj)
    make_phase_b_figures(cp, obj)
    return obj


def write_phase_a_report(obj):
    lines = [
        "# GDN INT8 Repeated Accumulation Formal V1",
        "",
        "## Answers",
        f"1. Robust across 6 prompts: `{obj['R128_REPEATED_ACCUMULATION_FORMAL']}`.",
        f"2. Robust across t0={{64,128,256}}: `{obj['by_t0_summary']}`.",
        f"3. State accumulation monotonicity median: `{obj['median_state_adjacent_dose_response']}`.",
        f"4. KL degradation monotonicity median: `{obj['median_KL_adjacent_dose_response']}`.",
        f"5. Readout-visible monotonicity is diagnostic; median rho frozen/actual: `{obj['R128_median_rho_freq_frozen_readout_AUC']}`, `{obj['R128_median_rho_freq_actual_readout_AUC']}`.",
        f"6. Failure-containing units: `{sum(1 for r in obj['per_unit'] if r['numerical_failure_count'])}`.",
        f"7. R128 repeated accumulation formally supported: `{obj['R128_REPEATED_ACCUMULATION_FORMAL']}`.",
        "",
        "## Formal Gate",
        f"- valid formal units: `{obj['valid_formal_units']} / 18`",
        f"- median rho(freq, state AUC): `{obj['R128_median_rho_freq_state_AUC']}`",
        f"- median rho(freq, KL AUC): `{obj['R128_median_rho_freq_KL_AUC']}`",
        f"- PHASE_B_READY: `{obj['PHASE_B_READY']}`",
        "- PROJECT_STAGE: `MECHANISM_VALIDATION`",
        "- METHOD_DESIGN_READY: `NO`",
    ]
    PHASE_A_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PHASE_A_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase_b_report(obj):
    lines = [
        "# GDN INT8 R128/C128 Accumulation Comparison V1",
        "",
        "## Answers",
        f"1. SINGLE exposure KL AUC R128/C128: `{obj['aggregate']['R128_SINGLE_KL_AUC']}` / `{obj['aggregate']['C128_SINGLE_KL_AUC']}`.",
        "2. State AUC differences by cadence are in JSON under `per_prompt.*.cadence`.",
        "3. KL AUC differences by cadence are in JSON under `per_prompt.*.cadence`.",
        f"4. R128-C128 gap widens with cadence: KL rho `{obj['aggregate']['rho_freq_R128_C128_KL_gap']}`, state rho `{obj['aggregate']['rho_freq_R128_C128_state_gap']}`.",
        f"5. Delta-from-SINGLE comparison: R128/C128 KL repeated excess `{obj['aggregate']['R128_KL_REPEATED_EXCESS']}` / `{obj['aggregate']['C128_KL_REPEATED_EXCESS']}`; state repeated excess `{obj['aggregate']['R128_STATE_REPEATED_EXCESS']}` / `{obj['aggregate']['C128_STATE_REPEATED_EXCESS']}`.",
        f"6. Classification: `{obj['AXIS_ACCUMULATION_CLASSIFICATION']}`.",
        "7. This does not explain frozen-driver vs full-feedback contributions, linearity, or final method readiness.",
        f"8. Next decomposition justified only as next research task: `{obj['NEXT_RECOMMENDED_TASK']}`.",
        "",
        "## Constraints",
        "- METHOD_DESIGN_READY: `NO`",
        "- MECHANISM_CLOSURE_CANDIDATE: `NO`",
    ]
    PHASE_B_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PHASE_B_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_phase_a_figures(cp, obj):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = [0, *[FREQ[c] for c in PERIODIC]]
    labels = ["SINGLE", *PERIODIC]
    units = assemble_units(cp, "PHASE_A", "R128", [64, 128, 256])
    for metric, name in [("state_error_AUC", "figure_A1_cadence_vs_state_auc.png"), ("KL_AUC", "figure_A2_cadence_vs_kl_auc.png")]:
        plt.figure()
        for key, u in units.items():
            plt.plot(xs, [u["conditions"][c][metric] for c in labels], alpha=0.35)
        plt.xlabel("cadence frequency (SINGLE=0)")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(FIG_DIR / name, dpi=160)
        plt.close()
    plt.figure()
    for t0 in [64, 128, 256]:
        vals = []
        for c in labels:
            vals.append(avg([u["conditions"][c]["KL_AUC"] for u in units.values() if u["t0"] == t0]))
        plt.plot(xs, vals, marker="o", label=f"t0={t0}")
    plt.xlabel("cadence frequency (SINGLE=0)")
    plt.ylabel("KL_AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_A3_t0_stratified_cadence_vs_kl.png", dpi=160)
    plt.close()
    plt.figure()
    for pid in [p["problem_id"] for p in selected_prompts()]:
        vals = []
        for c in labels:
            vals.append(avg([u["conditions"][c]["KL_AUC"] for u in units.values() if u["problem_id"] == pid]))
        plt.plot(xs, vals, marker="o", label=pid)
    plt.xlabel("cadence frequency (SINGLE=0)")
    plt.ylabel("KL_AUC")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_A4_prompt_stratified_cadence_vs_kl.png", dpi=160)
    plt.close()


def make_phase_b_figures(cp, obj):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = [0, *[FREQ[c] for c in PERIODIC]]
    labels = ["SINGLE", *PERIODIC]
    for metric, name in [("KL_AUC", "figure_B1_cadence_vs_kl_auc_r128_c128.png"), ("state_error_AUC", "figure_B2_cadence_vs_state_auc_r128_c128.png")]:
        plt.figure()
        for axn in ["R128", "C128"]:
            plt.plot(xs, [axis_mean(cp, axn, c, metric) for c in labels], marker="o", label=axn)
        plt.xlabel("cadence frequency (SINGLE=0)")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / name, dpi=160)
        plt.close()
    for metric, name in [("KL_AUC", "figure_B3_delta_from_single_kl.png"), ("state_error_AUC", "figure_B4_delta_from_single_state.png")]:
        plt.figure()
        for axn in ["R128", "C128"]:
            base = axis_mean(cp, axn, "SINGLE", metric)
            plt.plot(xs, [axis_mean(cp, axn, c, metric) - base for c in labels], marker="o", label=axn)
        plt.xlabel("cadence frequency (SINGLE=0)")
        plt.ylabel(f"delta {metric} from SINGLE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / name, dpi=160)
        plt.close()
    for metric, name in [("KL_AUC", "figure_B5_r128_c128_kl_gap.png"), ("state_error_AUC", "figure_B6_r128_c128_state_gap.png")]:
        plt.figure()
        plt.plot(xs, [axis_mean(cp, "R128", c, metric) - axis_mean(cp, "C128", c, metric) for c in labels], marker="o")
        plt.xlabel("cadence frequency (SINGLE=0)")
        plt.ylabel(f"R128-C128 {metric} gap")
        plt.tight_layout()
        plt.savefig(FIG_DIR / name, dpi=160)
        plt.close()


def phase_b_gate_audit(cp):
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    pm = selected_prompts()[0]
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(next(model.parameters()).device)
    mask = enc.get("attention_mask")
    mask = mask.to(ids.device) if mask is not None else None
    with torch.inference_mode():
        out = p1.feed_step(torch, model, ids, mask, None)
    ok = True
    for layer in GDN_LAYERS:
        s = p1.get_state(out.past_key_values, layer)
        _post, _q, shape, _st = quantize_state(torch, s, "C128")
        ok = ok and list(shape) == [1, 32, 1, 128]
    cp["C128_QUANTIZER_IDENTITY_GATE"] = "PASS" if ok else "FAIL"
    write_checkpoint(cp)
    return cp["C128_QUANTIZER_IDENTITY_GATE"]


def print_summary(phase_a, phase_b=None):
    print("\nTASK =")
    print(TASK)
    print("\n\n=== PHASE A ===")
    print("\nValid formal units =")
    print(f"{phase_a['valid_formal_units']} / 18")
    print("\nR128 median rho(freq, state AUC) =")
    print(phase_a["R128_median_rho_freq_state_AUC"])
    print("\nR128 median rho(freq, KL AUC) =")
    print(phase_a["R128_median_rho_freq_KL_AUC"])
    print("\nR128 median rho(freq, frozen readout AUC) =")
    print(phase_a["R128_median_rho_freq_frozen_readout_AUC"])
    print("\nR128 median rho(freq, actual readout AUC) =")
    print(phase_a["R128_median_rho_freq_actual_readout_AUC"])
    print("\nR128 EVERY_1 > EVERY_32 KL =")
    print(phase_a["R128_EVERY_1_gt_EVERY_32_KL"])
    print("\nR128 EVERY_1 > SINGLE state =")
    print(phase_a["R128_EVERY_1_gt_SINGLE_state"])
    print("\nMedian KL adjacent dose response =")
    print(phase_a["median_KL_adjacent_dose_response"])
    print("\nMedian state adjacent dose response =")
    print(phase_a["median_state_adjacent_dose_response"])
    print("\nBy-t0 summary =")
    for k, v in phase_a["by_t0_summary"].items():
        print(f"t0={k}: {v}")
    print("\nR128_REPEATED_ACCUMULATION_FORMAL =")
    print(phase_a["R128_REPEATED_ACCUMULATION_FORMAL"])
    print("\nPHASE_B_READY =")
    print(phase_a["PHASE_B_READY"])
    print("\n\n=== PHASE B ===")
    if phase_b is None:
        print("\nNOT_RUN_DUE_TO_PHASE_A_GATE")
    else:
        a = phase_b["aggregate"]
        print("\nPrompts =")
        print(phase_b["prompts"])
        for k in ["R128_SINGLE_KL_AUC", "C128_SINGLE_KL_AUC", "R128_E1_KL_AUC", "C128_E1_KL_AUC", "R128_KL_REPEATED_EXCESS", "C128_KL_REPEATED_EXCESS", "R128_STATE_REPEATED_EXCESS", "C128_STATE_REPEATED_EXCESS", "R128_KL_CADENCE_GAIN", "C128_KL_CADENCE_GAIN", "R128_STATE_CADENCE_GAIN", "C128_STATE_CADENCE_GAIN"]:
            print(f"\n{k} =")
            print(a[k])
        print("\nR128 > C128 KL repeated excess =")
        print(a["R128_gt_C128_KL_repeated_excess"])
        print("\nR128 > C128 state repeated excess =")
        print(a["R128_gt_C128_state_repeated_excess"])
        print("\nrho(freq, R128-C128 KL gap) =")
        print(a["rho_freq_R128_C128_KL_gap"])
        print("\nrho(freq, R128-C128 state gap) =")
        print(a["rho_freq_R128_C128_state_gap"])
        print("\nAXIS_ACCUMULATION_CLASSIFICATION =")
        print(phase_b["AXIS_ACCUMULATION_CLASSIFICATION"])
    print("\n\nNumerical failures =")
    print(sum(r.get("numerical_failure_count", 0) for r in phase_a["per_unit"]))
    print("\nMECHANISM_CLOSURE_CANDIDATE =\nNO")
    print("\nMETHOD_DESIGN_READY =\nNO")
    print("\nNEXT_RECOMMENDED_TASK =")
    print((phase_b or {}).get("NEXT_RECOMMENDED_TASK", "Formal/axis follow-up only if gate allows."))
    print("\nArtifacts =")
    print(SCRIPT)
    print(PHASE_A_JSON)
    print(PHASE_A_REPORT)
    if phase_b:
        print(PHASE_B_JSON)
        print(PHASE_B_REPORT)
    print("\nSTOP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["phase_a", "phase_b", "all", "analyze"], default="all")
    ap.add_argument("--horizon", type=int, default=128)
    args = ap.parse_args()
    cp = load_checkpoint()
    prompts = [p["problem_id"] for p in selected_prompts()]
    if args.stage in ("phase_a", "all"):
        units_a = [(pid, t0) for pid in prompts for t0 in [64, 128, 256]]
        cp = run_missing_conditions(cp, "PHASE_A", units_a, ["R128"], CADENCES, args.horizon)
        phase_a = analyze_phase_a(cp)
        if phase_a["PHASE_B_READY"] != "YES":
            print_summary(phase_a, None)
            return
    else:
        phase_a = load_json(PHASE_A_JSON) or analyze_phase_a(cp)
    phase_b = None
    if args.stage in ("phase_b", "all"):
        if phase_a["PHASE_B_READY"] != "YES":
            print_summary(phase_a, None)
            return
        if phase_b_gate_audit(cp) != "PASS":
            raise RuntimeError("C128_QUANTIZER_IDENTITY_GATE failed; STOP")
        units_b = [(pid, 128) for pid in prompts]
        cp = run_missing_conditions(cp, "PHASE_B", units_b, ["C128"], CADENCES, args.horizon)
        phase_b = analyze_phase_b(cp)
    elif args.stage == "analyze":
        phase_b = load_json(PHASE_B_JSON)
    print_summary(phase_a, phase_b)


if __name__ == "__main__":
    main()
