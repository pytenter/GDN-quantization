#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_REPEATED_RESIDUAL_ACCUMULATION_CADENCE_CAUSAL_EXPERIMENT_V1"
SCRIPT = EXP / "run_int8_repeated_residual_accumulation_cadence.py"
STAGE0 = RES / "gdn_int8_repeated_residual_accumulation_cadence_v1_stage0.json"
SMOKE = RES / "gdn_int8_repeated_residual_accumulation_cadence_v1_smoke.json"
PILOT = RES / "gdn_int8_repeated_residual_accumulation_cadence_v1_pilot.json"
CHECKPOINT = RES / "gdn_int8_repeated_residual_accumulation_cadence_v1_checkpoint.json"
REPORT = REP / "gdn_int8_repeated_residual_accumulation_cadence_v1.md"
FIG_DIR = RES / "gdn_int8_repeated_residual_accumulation_cadence_v1_figures"

EPS = 1e-12
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
CADENCES = ["FP_STATE", "SINGLE", "EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
ORDERED_CADENCES = ["SINGLE", "EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
PERIODIC_CADENCES = ["EVERY_32", "EVERY_16", "EVERY_8", "EVERY_4", "EVERY_2", "EVERY_1"]
FREQ = {"EVERY_32": 1 / 32, "EVERY_16": 1 / 16, "EVERY_8": 1 / 8, "EVERY_4": 1 / 4, "EVERY_2": 1 / 2, "EVERY_1": 1.0}
R128_CFG = {"name": "R128", "orientation": "row", "group_size": 128}

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
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


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


def selected_prompts(n=3):
    rows = [r for r in p1.selected_prompt_rows() if r.get("fp_response")]
    return rows[:n]


def schedule_indices(cadence, t0, last_t):
    if cadence == "FP_STATE":
        return []
    if cadence == "SINGLE":
        return [t0] if t0 <= last_t else []
    step = int(cadence.split("_")[1])
    return list(range(t0, last_t + 1, step))


def quantize_r128(torch, state):
    y, q, _scale_full, scale, scale_shape, dyn = axis.grouped_quant(torch, state.detach().float(), state.detach().float(), R128_CFG)
    canonical_scale_shape = [state.shape[0], state.shape[1], state.shape[2], 1]
    scale_vals = scale.detach().float().flatten().cpu().tolist()
    return y, q, scale, canonical_scale_shape, dyn, {
        "scale_mean": avg(scale_vals),
        "scale_median": med(scale_vals),
        "scale_max": max(scale_vals) if scale_vals else None,
        "scale_p95": pct(scale_vals, 0.95),
        "scale_p95_over_median": (pct(scale_vals, 0.95) / max(med(scale_vals), EPS)) if scale_vals else None,
        "within_group_dynamic_range_mean": avg(dyn),
        "within_group_dynamic_range_median": med(dyn),
        "within_group_dynamic_range_p95": pct(dyn, 0.95),
        "saturation_fraction": float((q.abs() >= 127).float().mean().item()),
    }


def state_totals(torch, past, ref_past):
    err2 = ref2 = 0.0
    for layer in GDN_LAYERS:
        a = p1.get_state(past, layer).detach().float()
        b = p1.get_state(ref_past, layer).detach().float()
        d = a - b
        err2 += float(torch.sum(d.double() * d.double()).item())
        ref2 += float(torch.sum(b.double() * b.double()).item())
    err = math.sqrt(err2)
    return err, err / (math.sqrt(ref2) + EPS)


def summarize_curve(vals):
    vals = [float(v) for v in vals if finite(v)]
    return {
        "auc": avg(vals),
        "terminal": vals[-1] if vals else None,
        "max": max(vals) if vals else None,
        "mean": avg(vals),
    }


def rankdata(xs):
    pairs = sorted((x, i) for i, x in enumerate(xs))
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        r = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[pairs[k][1]] = r
        i = j
    return ranks


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    rx = rankdata([p[0] for p in pairs])
    ry = rankdata([p[1] for p in pairs])
    mx, my = avg(rx), avg(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx)) * math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den + EPS)


def inject_r128(torch, past, token_idx):
    meta = []
    total_r2 = total_pre2 = rms_num = count = 0.0
    max_identity = 0.0
    for layer in GDN_LAYERS:
        state = p1.get_state(past, layer)
        pre = state.detach().clone()
        post, _q, _scale, scale_shape, _dyn, scale_stats = quantize_r128(torch, pre)
        residual = post - pre.float()
        ident = norm(torch, residual - (post - pre.float()))
        max_identity = max(max_identity, ident)
        rnorm = norm(torch, residual)
        pnorm = norm(torch, pre)
        total_r2 += rnorm * rnorm
        total_pre2 += pnorm * pnorm
        rms_num += float(torch.mean(residual.float() ** 2).item())
        count += 1
        state.copy_(post.to(state.dtype))
        meta.append({
            "token_index": int(token_idx),
            "layer_idx": int(layer),
            "residual_norm": rnorm,
            "relative_residual_norm": rnorm / (pnorm + EPS),
            "rms_residual": float(torch.sqrt(torch.mean(residual.float() ** 2)).item()),
            "scale_shape": list(scale_shape),
            **scale_stats,
        })
    total_r = math.sqrt(total_r2)
    return {
        "token_index": int(token_idx),
        "residual_norm": total_r,
        "relative_residual_norm": total_r / (math.sqrt(total_pre2) + EPS),
        "rms_residual": math.sqrt(rms_num / max(count, 1)),
        "layer_head_aggregate_residual_norm": total_r,
        "identity_error_norm": max_identity,
        "layer_metadata": meta,
    }


def readout_metrics(torch, model, col, cfg, fp_past, cfg_past):
    rm, _layer_rows = readout.compute_readout_metrics(torch, model, col, cfg, fp_past, cfg_past)
    return rm


def run_condition(torch, model, tokenizer, e2e, pm, cadence, t0, horizon, collect_readout=True):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[: t0 + horizon + 1]
    if len(cont_ids) <= t0:
        raise RuntimeError(f"not enough continuation tokens for {pm['problem_id']}: have {len(cont_ids)} need > {t0}")
    last_t = min(t0 + horizon - 1, len(cont_ids) - 1)
    actual_horizon = max(0, last_t - t0 + 1)
    sched = schedule_indices(cadence, t0, last_t)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    base_input = enc["input_ids"].to(device)
    base_mask = enc.get("attention_mask")
    base_mask = base_mask.to(device) if base_mask is not None else None
    fp_input = base_input.clone()
    fp_mask = base_mask.clone() if base_mask is not None else None
    cfg_input = base_input.clone()
    cfg_mask = base_mask.clone() if base_mask is not None else None
    fp_past = None
    cfg_past = None
    state_err, state_rel, kl, top1 = [], [], [], []
    frozen, actual, postproj, resid_stream = [], [], [], []
    inj_meta = []
    numerical_failures = []
    col = readout.install_readout_hooks(torch, model) if collect_readout else None
    try:
        with torch.inference_mode():
            for t in range(last_t + 1):
                if col:
                    col["records"].clear()
                    col["cfg"] = "FP_STATE"
                fp_out = p1.feed_step(torch, model, fp_input, fp_mask, fp_past)
                fp_past = fp_out.past_key_values
                if col:
                    col["cfg"] = cadence
                cfg_out = p1.feed_step(torch, model, cfg_input, cfg_mask, cfg_past)
                cfg_past = cfg_out.past_key_values
                if t in sched:
                    im = inject_r128(torch, cfg_past, t)
                    im["number_of_injections_so_far"] = len(inj_meta) + 1
                    inj_meta.append(im)
                if col:
                    col["cfg"] = None
                if t >= t0:
                    se, sr = state_totals(torch, cfg_past, fp_past)
                    lm = eff.logits_metrics(torch, fp_out.logits, cfg_out.logits)
                    state_err.append(se)
                    state_rel.append(sr)
                    kl.append(lm["KL"])
                    top1.append(lm["top1_agreement"])
                    if collect_readout:
                        rm = readout_metrics(torch, model, col, cadence, fp_past, cfg_past)
                        frozen.append(rm.get("frozen_query_readout_error_norm"))
                        actual.append(rm.get("actual_preproj_readout_error_norm"))
                        postproj.append(rm.get("postproj_error_norm"))
                        resid_stream.append(rm.get("residual_stream_error_norm"))
                    if not all(math.isfinite(float(x)) for x in [se, sr, lm["KL"]]):
                        numerical_failures.append({"token_index": t, "reason": "nonfinite_metric"})
                if t < len(cont_ids):
                    nxt = torch.tensor([[cont_ids[t]]], dtype=base_input.dtype, device=device)
                    fp_input = nxt
                    cfg_input = nxt.clone()
                    fp_mask = None
                    cfg_mask = None
    finally:
        if col:
            col["close"]()
    residual_norms = [m["residual_norm"] for m in inj_meta]
    return {
        "cadence": cadence,
        "t0": t0,
        "continuation_horizon": actual_horizon,
        "intervention_token_indices": sched,
        "injection_count": len(inj_meta),
        "injection_metadata": inj_meta,
        "state_error_curve": state_err,
        "state_relative_error_curve": state_rel,
        "KL_curve": kl,
        "top1_curve": top1,
        "frozen_readout_error_curve": frozen,
        "actual_readout_error_curve": actual,
        "postproj_error_curve": postproj,
        "residual_stream_error_curve": resid_stream,
        "state_error_auc": summarize_curve(state_rel)["auc"],
        "state_error_norm_auc": summarize_curve(state_err)["auc"],
        "state_error_terminal": summarize_curve(state_rel)["terminal"],
        "state_error_max": summarize_curve(state_rel)["max"],
        "frozen_readout_error_auc": summarize_curve(frozen)["auc"],
        "actual_readout_error_auc": summarize_curve(actual)["auc"],
        "postproj_error_auc": summarize_curve(postproj)["auc"],
        "residual_stream_error_auc": summarize_curve(resid_stream)["auc"],
        "KL_auc": summarize_curve(kl)["auc"],
        "KL_terminal": summarize_curve(kl)["terminal"],
        "KL_max": summarize_curve(kl)["max"],
        "top1_agreement": avg(top1),
        "cumulative_residual_norm_sum": sum(residual_norms),
        "root_sum_square_residual": math.sqrt(sum(x * x for x in residual_norms)),
        "numerical_failures": numerical_failures,
    }


def stage0():
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    pm = selected_prompts(1)[0]
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    enc = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    with torch.inference_mode():
        out_a = p1.feed_step(torch, model, ids, mask, None)
        out_b = p1.feed_step(torch, model, ids.clone(), mask.clone() if mask is not None else None, None)
    shape_ok = True
    scale_ok = True
    max_fp_state_diff = 0.0
    max_fp_logit_diff = float(torch.max(torch.abs(out_a.logits.float() - out_b.logits.float())).item())
    for layer in GDN_LAYERS:
        s = p1.get_state(out_a.past_key_values, layer)
        shape_ok = shape_ok and list(s.shape) == [1, 32, 128, 128]
        _y, _q, _scale, scale_shape, _dyn, _stats = quantize_r128(torch, s)
        scale_ok = scale_ok and list(scale_shape) == [1, 32, 128, 1]
        d = norm(torch, p1.get_state(out_a.past_key_values, layer) - p1.get_state(out_b.past_key_values, layer))
        max_fp_state_diff = max(max_fp_state_diff, d)
    sched = {c: schedule_indices(c, 8, 40) for c in CADENCES}
    quant_identity = "PASS" if shape_ok and scale_ok else "FAIL"
    noninterf = "PASS" if max_fp_logit_diff <= 1e-6 and max_fp_state_diff <= 1e-6 else "FAIL"
    schedule_gate = "PASS" if sched["SINGLE"] == [8] and sched["EVERY_8"] == [8, 16, 24, 32, 40] and sched["EVERY_1"] == list(range(8, 41)) else "FAIL"
    ident_gate = "PASS"
    smoke_cond = run_condition(torch, model, tokenizer, e2e, pm, "EVERY_1", 4, 4, collect_readout=False)
    for im in smoke_cond["injection_metadata"]:
        if im["identity_error_norm"] > 1e-7:
            ident_gate = "FAIL"
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "protocol": "FP32 prompt prefill + teacher-forced continuation + only GDN recurrent state R128 interventions",
        "state_identity": "DynamicCache.layers[layer_idx].recurrent_states[0]",
        "state_shape": [1, 32, 128, 128],
        "quantizer_config": {"orientation": "row", "group_size": 128, "scale_shape": [1, 32, 128, 1], "qrange": [-127, 127], "round": "torch.round", "zero_point": 0},
        "cadence_conditions": CADENCES,
        "intervention_token_indices_audit": sched,
        "PROTOCOL_GATE": "PASS" if shape_ok else "FAIL",
        "QUANTIZER_IDENTITY_GATE": quant_identity,
        "CADENCE_SCHEDULE_GATE": schedule_gate,
        "INTERVENTION_IDENTITY_GATE": ident_gate,
        "INSTRUMENTATION_NONINTERFERENCE_GATE": noninterf,
        "fp_noninterference_max_logit_abs_diff": max_fp_logit_diff,
        "fp_noninterference_max_state_abs_norm_diff": max_fp_state_diff,
        "EVERY_1_identity_note": "EVERY_1 applies canonical R128 after every token from t0 onward.",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    save_json(STAGE0, obj)
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    return obj


def gate_pass(obj):
    keys = ["PROTOCOL_GATE", "QUANTIZER_IDENTITY_GATE", "CADENCE_SCHEDULE_GATE", "INTERVENTION_IDENTITY_GATE", "INSTRUMENTATION_NONINTERFERENCE_GATE"]
    return all(obj.get(k) == "PASS" for k in keys)


def run_stage(stage, t0, horizon):
    st = load_json(STAGE0) or stage0()
    if not gate_pass(st):
        raise RuntimeError("Stage0 gate failed; STOP")
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompts = selected_prompts(1 if stage == "smoke" else 3)
    cadences = ["FP_STATE", "SINGLE", "EVERY_8", "EVERY_1"] if stage == "smoke" else CADENCES
    out = {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "protocol": st["protocol"],
        "prompt_ids": [p["problem_id"] for p in prompts],
        "t0": t0,
        "continuation_horizon": horizon,
        "quantizer_config": st["quantizer_config"],
        "cadence_conditions": cadences,
        "intervention_token_indices": {},
        "injection_counts": {},
        "gate_results": {"stage0": "PASS"},
        "per_prompt": {},
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    for pm in prompts:
        pid = pm["problem_id"]
        out["per_prompt"][pid] = {"role": pm["role"], "per_condition": {}}
        for cadence in cadences:
            print(f"[{now()}] {stage} prompt={pid} cadence={cadence} t0={t0} horizon={horizon}", flush=True)
            rec = run_condition(torch, model, tokenizer, e2e, pm, cadence, t0, horizon, collect_readout=True)
            out["per_prompt"][pid]["per_condition"][cadence] = rec
            out["intervention_token_indices"][f"{pid}|{cadence}"] = rec["intervention_token_indices"]
            out["injection_counts"][f"{pid}|{cadence}"] = rec["injection_count"]
            save_json(CHECKPOINT, out)
    if stage == "smoke":
        failures = [nf for pp in out["per_prompt"].values() for c in pp["per_condition"].values() for nf in c["numerical_failures"]]
        out["Smoke"] = "PASS" if not failures else "FAIL"
        out["gate_results"]["SMOKE_GATE"] = out["Smoke"]
        save_json(SMOKE, out)
    else:
        analyzed = analyze_pilot(out)
        save_json(PILOT, analyzed)
        write_report(analyzed)
        make_figures(analyzed)
        print_terminal_summary(analyzed)
        return analyzed
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def analyze_pilot(obj):
    dose = {}
    adjacent = {}
    per_rows = []
    for pid, pp in obj["per_prompt"].items():
        cond = pp["per_condition"]
        xs = [FREQ[c] for c in PERIODIC_CADENCES]
        state_vals = [cond[c]["state_error_auc"] for c in PERIODIC_CADENCES]
        kl_vals = [cond[c]["KL_auc"] for c in PERIODIC_CADENCES]
        fr_vals = [cond[c]["frozen_readout_error_auc"] for c in PERIODIC_CADENCES]
        ar_vals = [cond[c]["actual_readout_error_auc"] for c in PERIODIC_CADENCES]
        pr_vals = [cond[c]["postproj_error_auc"] for c in PERIODIC_CADENCES]
        dose[pid] = {
            "periodic_spearman_state": spearman(xs, state_vals),
            "periodic_spearman_KL": spearman(xs, kl_vals),
            "periodic_spearman_frozen_readout": spearman(xs, fr_vals),
            "periodic_spearman_actual_readout": spearman(xs, ar_vals),
            "periodic_spearman_postproj": spearman(xs, pr_vals),
        }
        adjacent[pid] = {
            "KL_adjacent_monotonic_count": sum(1 for a, b in zip(ORDERED_CADENCES, ORDERED_CADENCES[1:]) if finite(cond[a]["KL_auc"]) and finite(cond[b]["KL_auc"]) and cond[b]["KL_auc"] > cond[a]["KL_auc"]),
            "state_adjacent_monotonic_count": sum(1 for a, b in zip(ORDERED_CADENCES, ORDERED_CADENCES[1:]) if finite(cond[a]["state_error_auc"]) and finite(cond[b]["state_error_auc"]) and cond[b]["state_error_auc"] > cond[a]["state_error_auc"]),
        }
        per_rows.append({
            "prompt": pid,
            "rho_freq_state_auc": dose[pid]["periodic_spearman_state"],
            "rho_freq_frozen_readout_auc": dose[pid]["periodic_spearman_frozen_readout"],
            "rho_freq_actual_readout_auc": dose[pid]["periodic_spearman_actual_readout"],
            "rho_freq_KL_auc": dose[pid]["periodic_spearman_KL"],
            **adjacent[pid],
        })
    median_state = med([v["periodic_spearman_state"] for v in dose.values()])
    median_kl = med([v["periodic_spearman_KL"] for v in dose.values()])
    median_fr = med([v["periodic_spearman_frozen_readout"] for v in dose.values()])
    median_ar = med([v["periodic_spearman_actual_readout"] for v in dose.values()])
    every1_gt_e32_kl = sum(1 for pp in obj["per_prompt"].values() if pp["per_condition"]["EVERY_1"]["KL_auc"] > pp["per_condition"]["EVERY_32"]["KL_auc"])
    every1_gt_single_state = sum(1 for pp in obj["per_prompt"].values() if pp["per_condition"]["EVERY_1"]["state_error_auc"] > pp["per_condition"]["SINGLE"]["state_error_auc"])
    positive = (
        finite(median_kl) and median_kl >= 0.7
        and sum(1 for v in dose.values() if finite(v["periodic_spearman_KL"]) and v["periodic_spearman_KL"] > 0) >= 2
        and every1_gt_e32_kl >= 2
        and every1_gt_single_state >= 2
    )
    negative = (
        sum(1 for v in dose.values() if finite(v["periodic_spearman_KL"]) and v["periodic_spearman_KL"] <= 0) >= 2
        and sum(1 for v in dose.values() if finite(v["periodic_spearman_state"]) and v["periodic_spearman_state"] <= 0) >= 2
    )
    classification = "POSITIVE" if positive else ("NEGATIVE" if negative else "INCONCLUSIVE")
    obj.update({
        "dose_response": {
            "per_prompt": dose,
            "periodic_spearman_state": median_state,
            "periodic_spearman_KL": median_kl,
            "periodic_spearman_frozen_readout": median_fr,
            "periodic_spearman_actual_readout": median_ar,
            "periodic_spearman_postproj": med([v["periodic_spearman_postproj"] for v in dose.values()]),
            "adjacent_order_counts": adjacent,
        },
        "aggregate": {
            "Median rho(freq, state AUC)": median_state,
            "Median rho(freq, KL AUC)": median_kl,
            "Median rho(freq, frozen readout AUC)": median_fr,
            "Median rho(freq, actual readout AUC)": median_ar,
            "EVERY_1 > EVERY_32 KL": f"{every1_gt_e32_kl} / {len(obj['per_prompt'])}",
            "EVERY_1 > SINGLE state accumulation": f"{every1_gt_single_state} / {len(obj['per_prompt'])}",
        },
        "per_prompt_rho_table": per_rows,
        "pilot_classification": classification,
        "formal_ready": "YES" if classification == "POSITIVE" else "NO",
        "limitations": [
            "Pilot uses 3 prompts and supports only a gate decision, not a paper-level conclusion.",
            "Cadence manipulation may include direct residual accumulation plus network feedback; this experiment does not decompose them.",
        ],
        "next_step": "Formal repeated-accumulation validation" if classification == "POSITIVE" else "Inspect pilot heterogeneity and instrumentation before formal.",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    })
    return obj


def condition_table(obj):
    rows = []
    for pid, pp in obj["per_prompt"].items():
        for c in ORDERED_CADENCES:
            r = pp["per_condition"][c]
            rows.append([pid, c, r["injection_count"], r["state_error_auc"], r["frozen_readout_error_auc"], r["actual_readout_error_auc"], r["KL_auc"], r["KL_terminal"], r["top1_agreement"]])
    return rows


def write_report(obj):
    lines = [
        "# GDN INT8 Repeated Residual Accumulation / Cadence Causal Experiment V1",
        "",
        "## 1. Task",
        f"`{TASK}`",
        "",
        "## 2. Scientific Question",
        "Does increasing the cadence of natural canonical R128 recurrent-state quantization residuals cause systematic state drift, readout-visible drift, and future KL degradation?",
        "",
        "## 3. Prior Evidence",
        "Prior results establish the R128/C128 orientation effect, R128 degradation, R16 rescue, invalidity of old same-codebook lost-update proxies, runtime effective-update relevance, residual strength/direction causality, broad directional sensitivity, low-gain-biased natural R128 residuals, and insufficiency of single-pulse propagation/readout metrics. These are not re-proved here.",
        "",
        "## 4. Intervention Definition",
        "At each scheduled token the model first computes `S_pre`, then writes `S_post = Q_R128(S_pre)` into `DynamicCache.layers[layer_idx].recurrent_states[0]`. The residual is the natural `S_post - S_pre`; no rescaling, replacement, randomization, or fixed residual replay is used.",
        "",
        "## 5. Protocol",
        f"- model: `{obj['model']}`",
        f"- t0: `{obj['t0']}`",
        f"- continuation_horizon: `{obj['continuation_horizon']}`",
        "- continuation_source: `P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED`",
        "- exact_original_generation_token_ids_available: `false`",
        "- causal variable: `R128 intervention cadence`",
        "",
        "## 6. Stage-0 Gates",
        "See `gdn_int8_repeated_residual_accumulation_cadence_v1_stage0.json`; all gates must pass before Pilot.",
        "",
        "## 7. Smoke",
        "Smoke ran FP_STATE / SINGLE / EVERY_8 / EVERY_1 and checks crash-free execution, hook stability, finite metrics, serialization, and checkpoint writes.",
        "",
        "## 8. Pilot Configuration",
        f"- prompts: `{len(obj['prompt_ids'])}` / 3",
        f"- cadence conditions: `{' / '.join(ORDERED_CADENCES)}`",
        "",
        "## 9. Injection Counts",
        "Injection count is an exposure description, not a separate causal variable.",
        "",
        "## 10. State Accumulation Results",
        f"Median rho(freq, state AUC): `{obj['dose_response']['periodic_spearman_state']}`",
        "",
        "## 11. Readout-Visible Accumulation",
        f"Median rho(freq, frozen readout AUC): `{obj['dose_response']['periodic_spearman_frozen_readout']}`",
        f"Median rho(freq, actual readout AUC): `{obj['dose_response']['periodic_spearman_actual_readout']}`",
        "",
        "## 12. KL Dose Response",
        f"Median rho(freq, KL AUC): `{obj['dose_response']['periodic_spearman_KL']}`",
        "",
        "## 13. Per-Prompt Results",
        "| Prompt | Cadence | Injection Count | AUC State Error | AUC Frozen Readout Error | AUC Actual Readout Error | AUC KL | Terminal KL | Top-1 Agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in condition_table(obj):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines += [
        "",
        "## 14. Adjacent Dose Response",
        "| Prompt | rho(freq, state AUC) | rho(freq, frozen-readout AUC) | rho(freq, actual-readout AUC) | rho(freq, KL AUC) | KL adjacent monotonic count | state adjacent monotonic count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in obj["per_prompt_rho_table"]:
        lines.append(f"| {row['prompt']} | {row['rho_freq_state_auc']} | {row['rho_freq_frozen_readout_auc']} | {row['rho_freq_actual_readout_auc']} | {row['rho_freq_KL_auc']} | {row['KL_adjacent_monotonic_count']}/6 | {row['state_adjacent_monotonic_count']}/6 |")
    lines += [
        "",
        "## 15. State Accumulation vs KL",
        "This is reported descriptively only; correlations among state AUC, readout AUC, and KL are not labeled as causal relations.",
        "",
        "## 16. Negative / Corrective Findings",
        "Any noisy, heterogeneous, or non-monotone prompt behavior is preserved in JSON and not smoothed away.",
        "",
        "## 17. Pilot Classification",
        f"`{obj['pilot_classification']}`; FORMAL_READY = `{obj['formal_ready']}`",
        "",
        "## 18. What Is Supported",
        "`REPEATED_RESIDUAL_ACCUMULATION_CADENCE_SIGNAL = SUPPORTED_IN_PILOT` only if Pilot is POSITIVE; otherwise no positive cadence conclusion is claimed.",
        "",
        "## 19. What Is NOT Supported",
        "This does not prove the full mechanism, network feedback, linear error addition, method readiness, bit allocation, mixed precision, high-gain direction protection, or quantization-frequency reduction as a final method.",
        "",
        "## 20. Next Step",
        obj["next_step"],
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(obj):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = [0, *[FREQ[c] for c in PERIODIC_CADENCES]]
    labels = ["SINGLE", *PERIODIC_CADENCES]
    for metric, fname, ylabel in [
        ("state_error_auc", "figure1_cadence_vs_auc_state_error.png", "AUC state relative error"),
        ("KL_auc", "figure2_cadence_vs_auc_kl.png", "AUC KL"),
        ("frozen_readout_error_auc", "figure5_cadence_vs_readout_visible_error.png", "AUC frozen readout error"),
    ]:
        plt.figure()
        for pid, pp in obj["per_prompt"].items():
            ys = [pp["per_condition"][c][metric] for c in labels]
            plt.plot(xs, ys, marker="o", label=pid)
        plt.xlabel("cadence frequency (SINGLE shown as 0)")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=160)
        plt.close()
    for metric, fname, ylabel in [
        ("state_relative_error_curve", "figure3_token_vs_state_error.png", "state relative error"),
        ("KL_curve", "figure4_token_vs_kl.png", "KL"),
    ]:
        plt.figure()
        pid = obj["prompt_ids"][0]
        pp = obj["per_prompt"][pid]
        for c in labels:
            ys = pp["per_condition"][c][metric]
            plt.plot(list(range(len(ys))), ys, label=c)
        plt.xlabel("post-t0 token offset")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=160)
        plt.close()


def print_terminal_summary(obj):
    adj = obj["dose_response"]["adjacent_order_counts"]
    print("\nTASK =")
    print(TASK)
    print("\nStage 0 =")
    print("PASS")
    print("\nSmoke =")
    smoke = load_json(SMOKE)
    print((smoke or {}).get("Smoke", "NOT_RUN"))
    print("\nPilot =")
    print(obj["pilot_classification"])
    print("\nPilot prompts =")
    print(f"{len(obj['prompt_ids'])} / 3")
    print("\nCadence conditions =")
    print("SINGLE / E32 / E16 / E8 / E4 / E2 / E1")
    print("\nMedian rho(freq, state AUC) =")
    print(obj["dose_response"]["periodic_spearman_state"])
    print("\nMedian rho(freq, KL AUC) =")
    print(obj["dose_response"]["periodic_spearman_KL"])
    print("\nMedian rho(freq, frozen readout AUC) =")
    print(obj["dose_response"]["periodic_spearman_frozen_readout"])
    print("\nMedian rho(freq, actual readout AUC) =")
    print(obj["dose_response"]["periodic_spearman_actual_readout"])
    print("\nEVERY_1 > EVERY_32 KL =")
    print(obj["aggregate"]["EVERY_1 > EVERY_32 KL"])
    print("\nEVERY_1 > SINGLE state accumulation =")
    print(obj["aggregate"]["EVERY_1 > SINGLE state accumulation"])
    print("\nKL adjacent dose-response =")
    for pid in obj["prompt_ids"]:
        print(f"{pid}: {adj[pid]['KL_adjacent_monotonic_count']}/6")
    print("\nState adjacent dose-response =")
    for pid in obj["prompt_ids"]:
        print(f"{pid}: {adj[pid]['state_adjacent_monotonic_count']}/6")
    failures = [nf for pp in obj["per_prompt"].values() for cc in pp["per_condition"].values() for nf in cc["numerical_failures"]]
    print("\nNumerical failures =")
    print(len(failures))
    print("\nPILOT_CLASSIFICATION =")
    print(obj["pilot_classification"])
    print("\nFORMAL_READY =")
    print(obj["formal_ready"])
    print("\nMECHANISM_CLOSURE_CANDIDATE =")
    print("NO")
    print("\nMETHOD_DESIGN_READY =")
    print("NO")
    print("\nArtifacts =")
    print(SCRIPT)
    print(PILOT)
    print(REPORT)
    print("\nThen STOP.")


def analyze_existing():
    obj = load_json(PILOT)
    if obj and "pilot_classification" in obj:
        print_terminal_summary(obj)
        return obj
    raise RuntimeError("No pilot result available")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "smoke", "pilot", "analyze"], required=True)
    ap.add_argument("--t0", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=128)
    args = ap.parse_args()
    if args.stage == "stage0":
        stage0()
    elif args.stage == "smoke":
        run_stage("smoke", min(args.t0, 16), min(args.horizon, 16))
    elif args.stage == "pilot":
        run_stage("pilot", args.t0, args.horizon)
    else:
        analyze_existing()


if __name__ == "__main__":
    main()
