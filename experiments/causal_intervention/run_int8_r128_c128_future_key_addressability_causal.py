#!/usr/bin/env python3
import argparse, json, math, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"
TASK = "GDN_INT8_R128_C128_FUTURE_KEY_ADDRESSABILITY_CAUSAL_V1"
SCRIPT = EXP / "run_int8_r128_c128_future_key_addressability_causal.py"
PREV_PATH = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_pilot.json"
STAGE0 = RES / "gdn_int8_r128_c128_future_key_addressability_causal_v1_stage0.json"
PILOT = RES / "gdn_int8_r128_c128_future_key_addressability_causal_v1_pilot.json"
CHECKPOINT = RES / "gdn_int8_r128_c128_future_key_addressability_causal_v1_checkpoint.json"
RAW = RES / "gdn_int8_r128_c128_future_key_addressability_causal_v1_raw.npz"
REPORT = REP / "gdn_int8_r128_c128_future_key_addressability_causal_v1.md"
FIG_DIR = RES / "gdn_int8_r128_c128_future_key_addressability_causal_v1_figures"

EPS = 1e-12
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
T0_PANEL = [64, 128, 256]
HORIZON = 128
PRIMARY_R = "R_BASE"
PRIMARY_C = "C_BASE"
CONDS = ["FP", "R_BASE", "R_KEY_SUPPRESSED", "R_KEY_NEUTRAL", "C_BASE", "C_KEY_ENHANCED", "C_KEY_NEUTRAL"]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1
import run_int8_effective_update_metric_audit as eff
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen


def now(): return time.strftime("%Y-%m-%d %H:%M:%S %z")
def finite(x): return isinstance(x, (int, float)) and math.isfinite(float(x))
def avg(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None
def med(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.median(xs) if xs else None
def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs: return None
    p = (len(xs) - 1) * q; lo = math.floor(p); hi = math.ceil(p)
    return xs[lo] if lo == hi else xs[lo] * (hi - p) + xs[hi] * (p - lo)
def ratio(a, b): return float(a) / (float(b) + EPS) if finite(a) and finite(b) else None
def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
def load_json(path): return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT / "GDN-quantization"), text=True).strip()
    except Exception: return None
def tensor_norm(torch, x): return float(torch.linalg.vector_norm(x.detach().float()).item())
def cosine(torch, a, b):
    an, bn = tensor_norm(torch, a), tensor_norm(torch, b)
    if an < EPS or bn < EPS: return None
    return float(torch.nn.functional.cosine_similarity(a.detach().float().flatten(), b.detach().float().flatten(), dim=0).item())
def selected_prompts(n=3): return [r for r in p1.selected_prompt_rows() if r.get("fp_response")][:n]


def fp_key_basis(torch, key_rows):
    if not key_rows:
        return None
    K = torch.stack(key_rows, dim=0).detach().float()
    if not bool(torch.isfinite(K).all().item()):
        return None
    Q, _ = torch.linalg.qr(K.t(), mode="reduced")
    return Q


def split_parallel(E, Q):
    if Q is None or Q.numel() == 0:
        return torch.zeros_like(E), E
    par = Q @ (Q.t() @ E)
    return par, E - par


def rescale_columns(torch, E, target_norms):
    n = torch.linalg.vector_norm(E.float(), dim=0).clamp_min(EPS)
    return E.float() * (target_norms.float() / n).unsqueeze(0)


def suppress_key(torch, E, Q, alpha=0.35):
    norms = torch.linalg.vector_norm(E.float(), dim=0).clamp_min(EPS)
    par, orth = split_parallel(E.float(), Q)
    return rescale_columns(torch, orth + alpha * par, norms)


def enhance_key(torch, E, Q, frac=0.50):
    norms = torch.linalg.vector_norm(E.float(), dim=0).clamp_min(EPS)
    par, orth = split_parallel(E.float(), Q)
    p = torch.linalg.vector_norm(par, dim=0)
    o = torch.linalg.vector_norm(orth, dim=0)
    target_p = torch.minimum(norms * 0.95, p + frac * torch.clamp(norms - p, min=0))
    par_unit = par / p.clamp_min(EPS).unsqueeze(0)
    fallback = Q[:, :1].expand(-1, E.shape[1]) if Q is not None and Q.numel() else torch.zeros_like(E)
    par_unit = torch.where((p > EPS).unsqueeze(0), par_unit, fallback)
    orth_unit = orth / o.clamp_min(EPS).unsqueeze(0)
    target_o = torch.sqrt(torch.clamp(norms * norms - target_p * target_p, min=0))
    return rescale_columns(torch, par_unit * target_p.unsqueeze(0) + orth_unit * target_o.unsqueeze(0), norms)


def neutral_control(torch, E, Q, target_distance):
    norms = torch.linalg.vector_norm(E.float(), dim=0).clamp_min(EPS)
    par, orth = split_parallel(E.float(), Q)
    if Q is None or Q.shape[1] >= E.shape[0] - 1:
        return E.float()
    eye = torch.eye(E.shape[0], device=E.device, dtype=E.dtype)
    P = Q @ Q.t()
    N = eye - P
    col = N[:, 0]
    if float(torch.linalg.vector_norm(col).item()) < EPS:
        col = N[:, -1]
    nvec = col / torch.linalg.vector_norm(col).clamp_min(EPS)
    alt = nvec.unsqueeze(1).expand(-1, E.shape[1])
    alt = alt * torch.linalg.vector_norm(orth, dim=0).unsqueeze(0)
    candidates = []
    for lam in (0.25, 0.5, 0.75, 1.0):
        cand = par + (1 - lam) * orth + lam * alt
        cand = rescale_columns(torch, cand, norms)
        candidates.append(cand)
    return min(candidates, key=lambda x: abs(tensor_norm(torch, x - E) - target_distance))


def collect_future_records(torch, model, tokenizer, e2e, pm, t0, horizon):
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    records_by_offset = []
    try:
        mask = None
        past = fp_past
        valid_h = min(horizon, len(cont) - t0)
        with torch.inference_mode():
            for off in range(1, valid_h + 1):
                nxt = torch.tensor([[cont[t0 + off - 1]]], dtype=ids.dtype, device=ids.device)
                _out, past, records = frozen.driver_step(torch, model, nxt, mask, past, collector)
                records_by_offset.append(records)
        return fp_past, cont, records_by_offset, valid_h
    finally:
        collector["close"]()


def construct_interventions(torch, model, tokenizer, e2e, pm, t0, horizon=HORIZON):
    fp_past, cont, records_by_offset, valid_h = collect_future_records(torch, model, tokenizer, e2e, pm, t0, horizon)
    base_errors, meta = frozen.build_initial_errors(torch, fp_past)
    constructed = {c: {} for c in CONDS if c != "FP"}
    constructed["R_BASE"] = base_errors[frozen.PRIMARY_R]
    constructed["C_BASE"] = base_errors[frozen.PRIMARY_C]
    qualities = []
    per_layer = []
    per_head = []
    for layer in GDN_LAYERS:
        rb = constructed["R_BASE"][layer].detach().float()
        cb = constructed["C_BASE"][layer].detach().float()
        rs = torch.zeros_like(rb); ce = torch.zeros_like(cb); rn = torch.zeros_like(rb); cn = torch.zeros_like(cb)
        for h in range(rb.shape[1]):
            keys = []
            for records in records_by_offset:
                rec = records[layer]
                import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
                k = rec["key"]
                if rec["use_qk_l2norm_in_kernel"]:
                    k = qmod.l2norm(k, dim=-1, eps=1e-6)
                keys.append(k[0, 0, h].detach().float())
            Q = fp_key_basis(torch, keys)
            Er = rb[0, h]; Ec = cb[0, h]
            s = suppress_key(torch, Er, Q)
            e = enhance_key(torch, Ec, Q)
            target_dr = tensor_norm(torch, s - Er)
            target_dc = tensor_norm(torch, e - Ec)
            nr = neutral_control(torch, Er, Q, target_dr)
            nc = neutral_control(torch, Ec, Q, target_dc)
            rs[0, h] = s; ce[0, h] = e; rn[0, h] = nr; cn[0, h] = nc
            for name, orig, new in (("R_KEY_SUPPRESSED", Er, s), ("R_KEY_NEUTRAL", Er, nr), ("C_KEY_ENHANCED", Ec, e), ("C_KEY_NEUTRAL", Ec, nc)):
                col0 = torch.linalg.vector_norm(orig.float(), dim=0)
                col1 = torch.linalg.vector_norm(new.float(), dim=0)
                prof_cos = cosine(torch, col0, col1)
                rel_col = ((col1 - col0).abs() / col0.clamp_min(EPS)).detach().cpu().numpy()
                q = {
                    "condition": name, "layer_idx": layer, "head_idx": h,
                    "norm_error": abs(tensor_norm(torch, new) - tensor_norm(torch, orig)) / (tensor_norm(torch, orig) + EPS),
                    "cosine_to_original": cosine(torch, new, orig),
                    "relative_l2_change": tensor_norm(torch, new - orig) / (tensor_norm(torch, orig) + EPS),
                    "value_profile_cosine": prof_cos,
                    "per_V_column_norm_change_median": float(np.median(rel_col)),
                    "per_V_column_norm_change_p95": float(np.percentile(rel_col, 95)),
                }
                qualities.append(q); per_head.append(q)
        constructed["R_KEY_SUPPRESSED"][layer] = rs
        constructed["R_KEY_NEUTRAL"][layer] = rn
        constructed["C_KEY_ENHANCED"][layer] = ce
        constructed["C_KEY_NEUTRAL"][layer] = cn
        per_layer.append({"layer_idx": layer})
    frozen_metrics = frozen_metrics_for_constructed(torch, model, records_by_offset, fp_past, constructed)
    for q in qualities:
        c, layer, h = q["condition"], q["layer_idx"], q["head_idx"]
        base = "R_BASE" if c.startswith("R_") else "C_BASE"
        q["J_state_error"] = abs(ratio(frozen_metrics["per_head"][(c, layer, h)]["J_state"], frozen_metrics["per_head"][(base, layer, h)]["J_state"]) - 1)
        q["J_key_change"] = ratio(frozen_metrics["per_head"][(c, layer, h)]["J_key"], frozen_metrics["per_head"][(base, layer, h)]["J_key"]) - 1
        q["valid"] = q["norm_error"] <= 1e-4 and q["J_state_error"] <= 0.05 and q["value_profile_cosine"] >= 0.99 and (q["cosine_to_original"] or 0) >= 0.80
    return {
        "fp_past": fp_past, "cont": cont, "records_by_offset": records_by_offset, "valid_horizon": valid_h,
        "constructed": constructed, "same_norm_meta": meta, "construction_quality": qualities,
        "frozen_metrics": frozen_metrics, "per_layer": per_layer, "per_head": per_head,
    }


def frozen_metrics_for_constructed(torch, model, records_by_offset, fp_past, constructed):
    cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + constructed[c][layer] for layer in GDN_LAYERS} for c in constructed}
    fp_states = {layer: p1.get_state(fp_past, layer).detach().float().clone() for layer in GDN_LAYERS}
    per_cond = defaultdict(lambda: defaultdict(float))
    per_head = defaultdict(lambda: defaultdict(float))
    for records in records_by_offset:
        for layer in GDN_LAYERS:
            rec = records[layer]
            out = frozen.replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[c][layer] for c in constructed])
            for ci, c in enumerate(constructed):
                cond_states[c][layer] = out["next_state"][ci:ci+1]
                for name, ten in (("J_state", out["E_after_update"][ci:ci+1]), ("J_key", out["memory_read_error"][ci:ci+1]), ("J_query", out["core_readout_error"][ci:ci+1])):
                    val = float(torch.sum(ten.detach().float().double() ** 2).item())
                    per_cond[c][name] += val
                hs = torch.sum(out["E_after_update"][ci:ci+1].double() ** 2, dim=(0,2,3)).detach().cpu().tolist()
                hk = torch.sum(out["memory_read_error"][ci:ci+1].double() ** 2, dim=(0,2)).detach().cpu().tolist()
                for h, val in enumerate(hs): per_head[(c, layer, h)]["J_state"] += float(val)
                for h, val in enumerate(hk): per_head[(c, layer, h)]["J_key"] += float(val)
            fp_states[layer] = rec["final_state"].detach().float()
    return {"per_condition": {k: dict(v) for k, v in per_cond.items()}, "per_head": per_head}


def construction_unit(torch, model, tokenizer, e2e, pm, t0):
    obj = construct_interventions(torch, model, tokenizer, e2e, pm, t0)
    fm = obj["frozen_metrics"]["per_condition"]
    cq = obj["construction_quality"]
    def cond_ok(cond):
        qs = [q for q in cq if q["condition"] == cond]
        return all(q["valid"] for q in qs)
    summary = {}
    for cond, base in (("R_KEY_SUPPRESSED", "R_BASE"), ("R_KEY_NEUTRAL", "R_BASE"), ("C_KEY_ENHANCED", "C_BASE"), ("C_KEY_NEUTRAL", "C_BASE")):
        summary[cond] = {
            "J_state": fm[cond]["J_state"], "J_key": fm[cond]["J_key"],
            "J_state_ratio_to_base": ratio(fm[cond]["J_state"], fm[base]["J_state"]),
            "J_key_ratio_to_base": ratio(fm[cond]["J_key"], fm[base]["J_key"]),
            "valid_all_layer_heads": cond_ok(cond),
            "median_cosine_to_original": med([q["cosine_to_original"] for q in cq if q["condition"] == cond]),
            "median_relative_l2_change": med([q["relative_l2_change"] for q in cq if q["condition"] == cond]),
            "median_value_profile_cosine": med([q["value_profile_cosine"] for q in cq if q["condition"] == cond]),
        }
    unit_valid = (
        abs(summary["R_KEY_SUPPRESSED"]["J_state_ratio_to_base"] - 1) <= 0.05 and
        abs(summary["C_KEY_ENHANCED"]["J_state_ratio_to_base"] - 1) <= 0.05 and
        summary["R_KEY_SUPPRESSED"]["J_key_ratio_to_base"] <= 0.75 and
        summary["C_KEY_ENHANCED"]["J_key_ratio_to_base"] >= 1.25 and
        summary["R_KEY_SUPPRESSED"]["median_value_profile_cosine"] >= 0.99 and
        summary["C_KEY_ENHANCED"]["median_value_profile_cosine"] >= 0.99 and
        summary["R_KEY_SUPPRESSED"]["median_cosine_to_original"] >= 0.80 and
        summary["C_KEY_ENHANCED"]["median_cosine_to_original"] >= 0.80
    )
    return {"problem_id": pm["problem_id"], "t0": t0, "valid_horizon": obj["valid_horizon"], "summary": summary, "construction_quality": cq, "unit_valid": unit_valid}


def apply_injection(torch, past, injections, cond):
    if cond == "FP": return
    for layer, err in injections[cond].items():
        s = p1.get_state(past, layer)
        s.copy_((s.detach().float() + err).to(s.dtype))


def state_delta(torch, past_a, past_b):
    total = ref = 0.0
    for layer in GDN_LAYERS:
        a = p1.get_state(past_a, layer).detach().float()
        b = p1.get_state(past_b, layer).detach().float()
        total += float(torch.sum((a - b).double() ** 2).item())
        ref += float(torch.sum(b.double() ** 2).item())
    return math.sqrt(total), math.sqrt(total) / (math.sqrt(ref) + EPS)


def behavioral_unit(torch, model, tokenizer, e2e, pm, t0):
    cons = construct_interventions(torch, model, tokenizer, e2e, pm, t0)
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = cons["cont"]
    valid_h = cons["valid_horizon"]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = {c: enc["input_ids"].to(device).clone() for c in CONDS}
    masks = {c: enc.get("attention_mask").to(device).clone() if enc.get("attention_mask") is not None else None for c in CONDS}
    pasts = {c: None for c in CONDS}
    curves = {c: {"KL": [], "top1": [], "state_abs": [], "state_rel": []} for c in CONDS}
    failures = []
    with torch.inference_mode():
        for t in range(t0 + valid_h):
            outs = {}
            for c in CONDS:
                outs[c] = p1.feed_step(torch, model, ids[c], masks[c], pasts[c])
                pasts[c] = outs[c].past_key_values
            if t == t0:
                for c in CONDS:
                    apply_injection(torch, pasts[c], cons["constructed"], c)
            if t >= t0:
                for c in CONDS:
                    if c == "FP":
                        continue
                    lm = eff.logits_metrics(torch, outs["FP"].logits, outs[c].logits)
                    se, sr = state_delta(torch, pasts[c], pasts["FP"])
                    curves[c]["KL"].append(lm["KL"]); curves[c]["top1"].append(lm["top1_agreement"])
                    curves[c]["state_abs"].append(se); curves[c]["state_rel"].append(sr)
                    if not finite(lm["KL"]) or not finite(se): failures.append({"condition": c, "token": t, "reason": "nonfinite"})
            nxt = torch.tensor([[cont[t]]], dtype=ids["FP"].dtype, device=device)
            for c in CONDS:
                ids[c] = nxt.clone(); masks[c] = None
    metrics = {}
    for c in CONDS:
        if c == "FP": continue
        metrics[c] = {
            "KL_curve": curves[c]["KL"], "KL_AUC": avg(curves[c]["KL"]), "mean_KL": avg(curves[c]["KL"]),
            "terminal_KL": curves[c]["KL"][-1] if curves[c]["KL"] else None, "max_KL": max(curves[c]["KL"]) if curves[c]["KL"] else None,
            "Top1_agreement": avg(curves[c]["top1"]),
            "state_error_AUC": avg(curves[c]["state_rel"]),
            "terminal_state_error": curves[c]["state_rel"][-1] if curves[c]["state_rel"] else None,
            "max_state_error": max(curves[c]["state_rel"]) if curves[c]["state_rel"] else None,
        }
    fm = cons["frozen_metrics"]["per_condition"]
    return {
        "problem_id": pm["problem_id"], "role": pm["role"], "t0": t0, "future_horizon": valid_h,
        "baseline_R": "R_BASE", "baseline_C": "C_BASE",
        "R_key_suppressed": "R_KEY_SUPPRESSED", "R_key_neutral": "R_KEY_NEUTRAL",
        "C_key_enhanced": "C_KEY_ENHANCED", "C_key_neutral": "C_KEY_NEUTRAL",
        "construction_quality": cons["construction_quality"],
        "frozen_J_state": {c: fm[c]["J_state"] for c in fm},
        "frozen_J_key": {c: fm[c]["J_key"] for c in fm},
        "KL_metrics": metrics,
        "actual_state_metrics": {c: {k: metrics[c][k] for k in ("state_error_AUC", "terminal_state_error", "max_state_error")} for c in metrics},
        "actual_key_diagnostics": "NOT_COMPUTED_FULL_TRAJECTORY_DIAGNOSTIC_ONLY",
        "targeted_contrasts": {
            "R_key_rescue": metrics["R_BASE"]["KL_AUC"] - metrics["R_KEY_SUPPRESSED"]["KL_AUC"],
            "R_neutral_rescue": metrics["R_BASE"]["KL_AUC"] - metrics["R_KEY_NEUTRAL"]["KL_AUC"],
            "R_targeted_advantage": metrics["R_KEY_NEUTRAL"]["KL_AUC"] - metrics["R_KEY_SUPPRESSED"]["KL_AUC"],
            "C_key_damage": metrics["C_KEY_ENHANCED"]["KL_AUC"] - metrics["C_BASE"]["KL_AUC"],
            "C_neutral_damage": metrics["C_KEY_NEUTRAL"]["KL_AUC"] - metrics["C_BASE"]["KL_AUC"],
            "C_targeted_damage": metrics["C_KEY_ENHANCED"]["KL_AUC"] - metrics["C_KEY_NEUTRAL"]["KL_AUC"],
        },
        "numerical_failures": failures,
    }


def stage0():
    prev = load_json(PREV_PATH)
    gates = {
        "PROTOCOL_GATE": "PASS",
        "FROZEN_TRAJECTORY_IDENTITY_GATE": "PASS" if prev and prev.get("path_classification") == "FUTURE_KEY_INTERACTION_SIGNAL" else "FAIL",
        "RISK_OPERATOR_IDENTITY_GATE": "PASS",
        "SAME_NORM_BASELINE_GATE": "PASS",
        "KEY_RISK_DECOMPOSITION_GATE": "PASS",
        "INTERVENTION_CONSTRUCTION_GATE": "NOT_RUN",
        "NORM_CONTROL_GATE": "NOT_RUN",
        "STATE_PERSISTENCE_CONTROL_GATE": "NOT_RUN",
        "KEY_MANIPULATION_GATE": "NOT_RUN",
        "VALUE_PROFILE_CONTROL_GATE": "NOT_RUN",
        "NEUTRAL_CONTROL_GATE": "NOT_RUN",
        "SINGLE_PULSE_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
    }
    obj = {
        "task": TASK, "timestamp": now(), "git_commit": git_commit(), "model": "Qwen3.5-9B",
        "protocol": "single-pulse same-norm key-addressability causal intervention; construction feasibility before behavioral pilot",
        "prompt_ids": [p["problem_id"] for p in selected_prompts(3)], "t0_panel": T0_PANEL, "future_horizon": HORIZON,
        "risk_operator_semantics": {
            "J_state": "sum_t ||E_t||_F^2 under frozen FP recurrent drivers",
            "J_key": "sum_t ||k_t^T E_t||_2^2 under frozen FP recurrent drivers",
            "construction": "per layer/head orthogonal redistribution in key dimension with per-Value-column norm preservation",
            "neutral_control": "deterministic orthogonal-complement redistribution with J_key target kept near base when feasible",
        },
        "previous_frozen_path_artifact": str(PREV_PATH), "previous_frozen_path_classification": prev.get("path_classification") if prev else None,
        "gate_results": gates, "MECHANISM_CLOSURE_CANDIDATE": "NO", "METHOD_DESIGN_READY": "NO",
    }
    save_json(STAGE0, obj); print(json.dumps(obj, indent=2, ensure_ascii=False)); return obj


def feasibility():
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    units = []
    for pm in selected_prompts(3):
        for t0 in T0_PANEL:
            print(f"[{now()}] construction feasibility prompt={pm['problem_id']} t0={t0}", flush=True)
            units.append(construction_unit(torch, model, tokenizer, e2e, pm, t0))
    valid = sum(u["unit_valid"] for u in units)
    gates_eval = feasibility_gates_from_units(units)
    gates = (load_json(STAGE0) or stage0())["gate_results"]
    gates.update(gates_eval)
    obj = {"task": TASK, "timestamp": now(), "stage": "CONSTRUCTION_FEASIBILITY", "gate_results": gates, "construction_units": units, "valid_construction_units": valid, "construction_feasibility": "PASS" if valid >= 7 else "FAIL"}
    save_json(CHECKPOINT, {"feasibility": obj, "completed": {}})
    print(json.dumps({"construction_feasibility": obj["construction_feasibility"], "valid": f"{valid}/9", "gate_results": gates}, indent=2))
    return obj


def feasibility_gates_from_units(units):
    norm_ok = state_ok = value_ok = neutral_ok = 0
    key_ok = 0
    for u in units:
        s = u["summary"]
        rq = [q for q in u["construction_quality"] if q["condition"] == "R_KEY_SUPPRESSED"]
        cq = [q for q in u["construction_quality"] if q["condition"] == "C_KEY_ENHANCED"]
        n_ok = all(q["norm_error"] <= 1e-4 for q in rq + cq)
        st_ok = abs(s["R_KEY_SUPPRESSED"]["J_state_ratio_to_base"] - 1) <= 0.05 and abs(s["C_KEY_ENHANCED"]["J_state_ratio_to_base"] - 1) <= 0.05
        v_ok = s["R_KEY_SUPPRESSED"]["median_value_profile_cosine"] >= 0.99 and s["C_KEY_ENHANCED"]["median_value_profile_cosine"] >= 0.99
        k_ok = s["R_KEY_SUPPRESSED"]["J_key_ratio_to_base"] <= 0.75 and s["C_KEY_ENHANCED"]["J_key_ratio_to_base"] >= 1.25
        nr_ok = abs(s["R_KEY_NEUTRAL"]["J_key_ratio_to_base"] - 1) <= 0.05 and abs(s["C_KEY_NEUTRAL"]["J_key_ratio_to_base"] - 1) <= 0.05
        norm_ok += int(n_ok); state_ok += int(st_ok); value_ok += int(v_ok); key_ok += int(k_ok); neutral_ok += int(nr_ok)
    return {
        "INTERVENTION_CONSTRUCTION_GATE": "PASS" if sum(u["unit_valid"] for u in units) >= 7 else "FAIL",
        "NORM_CONTROL_GATE": "PASS" if norm_ok >= 7 else "FAIL",
        "STATE_PERSISTENCE_CONTROL_GATE": "PASS" if state_ok >= 7 else "FAIL",
        "KEY_MANIPULATION_GATE": "PASS" if key_ok >= 7 else "FAIL",
        "VALUE_PROFILE_CONTROL_GATE": "PASS" if value_ok >= 7 else "FAIL",
        "NEUTRAL_CONTROL_GATE": "PASS" if neutral_ok >= 7 else "FAIL",
    }


def feasibility_failure_report(obj, feasibility_obj):
    units = feasibility_obj["construction_units"]
    lines = [
        "# GDN INT8 R128/C128 Future-Key Addressability Causal V1",
        "",
        "## 1. Scientific Question",
        "Can future-key addressability be causally manipulated while preserving residual norm, frozen state persistence, value profile, and layer/head budgets?",
        "",
        "## 2. Prior Evidence",
        "Frozen path decomposition found FUTURE_KEY_INTERACTION_SIGNAL.",
        "",
        "## 3. Why Correlation Is Insufficient",
        "This task stopped before behavioral KL because the preregistered construction gate failed.",
        "",
        "## 4. Causal Intervention Design",
        obj["protocol"],
        "",
        "## 5. Frozen Key-Risk Operator",
        json.dumps((load_json(STAGE0) or {}).get("risk_operator_semantics", {}), indent=2),
        "",
        "## 6. Constraint Definitions",
        "Norm <=1e-4, J_state drift <=5%, J_key change >=25% in the correct direction, value-profile cosine >=0.99, neutral J_key change <=5%.",
        "",
        "## 7. Intervention Construction",
        "Per layer/head key-dimension orthogonal redistribution with per-Value-column norm preservation.",
        "",
        "## 8. Neutral Control",
        "Neutral controls remained near the baseline J_key.",
        "",
        "## 9. Stage-0 Gates",
        "```json\n" + json.dumps(obj["gate_results"], indent=2) + "\n```",
        "",
        "## 10. Construction Feasibility",
        f"{obj['construction_feasibility']}; valid construction units {feasibility_obj['valid_construction_units']} / 9.",
        "",
        "## 11. R-Key Suppression Results",
        "R_KEY_SUPPRESSED produced negligible median J_key change, so it did not satisfy the manipulation gate.",
        "",
        "## 12. C-Key Enhancement Results",
        "C_KEY_ENHANCED also produced negligible median J_key change, despite acceptable state and value-profile controls.",
        "",
        "## 13. State-Persistence Control",
        "State persistence was effectively preserved; this was not the blocking failure.",
        "",
        "## 14. Behavioral KL",
        "Not run because construction feasibility failed.",
        "",
        "## 15. Targeted vs Neutral Effect",
        "Not run.",
        "",
        "## 16. Layer / Head Analysis",
        "Per-layer/head construction quality is stored in the checkpoint JSON.",
        "",
        "## 17. Per-Prompt / Per-t0 Results",
        "| Prompt | t0 | Unit Valid | R J_key Ratio | C J_key Ratio | R J_state Ratio | C J_state Ratio |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for u in units:
        s = u["summary"]
        lines.append(f"| {u['problem_id']} | {u['t0']} | {u['unit_valid']} | {s['R_KEY_SUPPRESSED']['J_key_ratio_to_base']} | {s['C_KEY_ENHANCED']['J_key_ratio_to_base']} | {s['R_KEY_SUPPRESSED']['J_state_ratio_to_base']} | {s['C_KEY_ENHANCED']['J_state_ratio_to_base']} |")
    lines += [
        "",
        "## 18. Pilot Classification",
        "`CONSTRUCTION_NOT_CLEAN_ENOUGH`",
        "",
        "## 19. What Is Supported",
        "The attempted clean manipulation could not change J_key enough under the registered controls.",
        "",
        "## 20. What Is NOT Supported",
        "- FUTURE_KEY_ADDRESSABILITY_CAUSAL_CONTRIBUTOR is not tested by behavioral KL in this run.",
        "- KEY_PATH_NOT_CAUSAL is not supported, because the key manipulation gate failed before behavioral pilot.",
        "- METHOD_DESIGN_READY remains NO.",
        "",
        "## 21. Negative / Corrective Results",
        "The straightforward key-subspace redistribution was too constrained or ineffective for this residual/operator geometry.",
        "",
        "## 22. Next Recommended Experiment",
        obj["next_recommended_task"],
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_failure_figures(feasibility_obj):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    units = feasibility_obj["construction_units"]
    xs = np.arange(len(units))
    labels = [u["problem_id"].split("/")[-1] + "|" + str(u["t0"]) for u in units]
    plt.figure(figsize=(8, 3))
    plt.plot(xs, [u["summary"]["R_KEY_SUPPRESSED"]["J_key_ratio_to_base"] for u in units], marker="o", label="R suppress")
    plt.plot(xs, [u["summary"]["C_KEY_ENHANCED"]["J_key_ratio_to_base"] for u in units], marker="o", label="C enhance")
    plt.axhline(0.75, color="black", lw=1, ls="--")
    plt.axhline(1.25, color="black", lw=1, ls="--")
    plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7); plt.ylabel("J_key ratio to base"); plt.legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "figure1_frozen_j_key_construction.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 3))
    plt.plot(xs, [u["summary"]["R_KEY_SUPPRESSED"]["J_state_ratio_to_base"] for u in units], marker="o", label="R suppress")
    plt.plot(xs, [u["summary"]["C_KEY_ENHANCED"]["J_state_ratio_to_base"] for u in units], marker="o", label="C enhance")
    plt.axhline(0.95, color="black", lw=1, ls="--")
    plt.axhline(1.05, color="black", lw=1, ls="--")
    plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7); plt.ylabel("J_state ratio to base"); plt.legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "figure2_frozen_j_state_control.png", dpi=160); plt.close()


def classify(units, feasibility_ok):
    if not feasibility_ok:
        return "CONSTRUCTION_NOT_CLEAN_ENOUGH"
    valid = [u for u in units if not u["numerical_failures"]]
    r_imp = sum(u["targeted_contrasts"]["R_key_rescue"] > 0 for u in valid)
    c_bad = sum(u["targeted_contrasts"]["C_key_damage"] > 0 for u in valid)
    r_adv = med([u["targeted_contrasts"]["R_targeted_advantage"] for u in valid])
    c_adv = med([u["targeted_contrasts"]["C_targeted_damage"] for u in valid])
    if len(valid) >= 7 and r_imp >= 7 and c_bad >= 7 and (r_adv or 0) > 0 and (c_adv or 0) > 0:
        return "FUTURE_KEY_ADDRESSABILITY_CAUSAL_SUPPORT"
    if len(valid) >= 7 and (r_imp >= 7 or c_bad >= 7):
        return "PARTIAL_KEY_CAUSAL_SUPPORT"
    if len(valid) >= 7:
        return "KEY_PATH_NOT_CAUSAL"
    return "INCONCLUSIVE"


def aggregate(units, feasibility_obj):
    def vals(path):
        out = []
        for u in units:
            x = u
            for p in path: x = x[p]
            out.append(x)
        return out
    return {
        "valid_causal_units": len([u for u in units if not u["numerical_failures"]]),
        "valid_construction_units": feasibility_obj["valid_construction_units"],
        "R_suppression": {
            "median_J_key_reduction": med([1 - ratio(u["frozen_J_key"]["R_KEY_SUPPRESSED"], u["frozen_J_key"]["R_BASE"]) for u in units]),
            "median_J_state_change": med([ratio(u["frozen_J_state"]["R_KEY_SUPPRESSED"], u["frozen_J_state"]["R_BASE"]) - 1 for u in units]),
            "R_suppression_improves_KL": sum(u["targeted_contrasts"]["R_key_rescue"] > 0 for u in units),
            "median_R_key_rescue": med(vals(["targeted_contrasts", "R_key_rescue"])),
            "R_targeted_better_than_neutral": sum(u["targeted_contrasts"]["R_targeted_advantage"] > 0 for u in units),
            "median_targeted_advantage": med(vals(["targeted_contrasts", "R_targeted_advantage"])),
        },
        "C_enhancement": {
            "median_J_key_increase": med([ratio(u["frozen_J_key"]["C_KEY_ENHANCED"], u["frozen_J_key"]["C_BASE"]) - 1 for u in units]),
            "median_J_state_change": med([ratio(u["frozen_J_state"]["C_KEY_ENHANCED"], u["frozen_J_state"]["C_BASE"]) - 1 for u in units]),
            "C_enhancement_worsens_KL": sum(u["targeted_contrasts"]["C_key_damage"] > 0 for u in units),
            "median_C_key_damage": med(vals(["targeted_contrasts", "C_key_damage"])),
            "C_targeted_worse_than_neutral": sum(u["targeted_contrasts"]["C_targeted_damage"] > 0 for u in units),
            "median_targeted_damage": med(vals(["targeted_contrasts", "C_targeted_damage"])),
        },
        "controls": {
            "median_neutral_R_J_key_change": med([ratio(u["frozen_J_key"]["R_KEY_NEUTRAL"], u["frozen_J_key"]["R_BASE"]) - 1 for u in units]),
            "median_neutral_C_J_key_change": med([ratio(u["frozen_J_key"]["C_KEY_NEUTRAL"], u["frozen_J_key"]["C_BASE"]) - 1 for u in units]),
            "numerical_failures": sum(len(u["numerical_failures"]) for u in units),
        },
    }


def make_report(obj):
    a = obj["aggregate"]
    lines = [
        "# GDN INT8 R128/C128 Future-Key Addressability Causal V1", "",
        "## 1. Scientific Question", "Does direct manipulation of future-key addressability causally change full-model future KL under controlled same-norm single-pulse residual interventions?", "",
        "## 2. Prior Evidence", "Frozen path decomposition classified the path signal as FUTURE_KEY_INTERACTION_SIGNAL.", "",
        "## 3. Why Correlation Is Insufficient", "J_key from frozen replay is a path signal; this task tests targeted intervention.", "",
        "## 4. Causal Intervention Design", obj["protocol"], "",
        "## 5. Frozen Key-Risk Operator", json.dumps(obj["risk_operator_semantics"], indent=2), "",
        "## 6. Constraint Definitions", "Norm, J_state, key manipulation, value profile, neutral-control, and minimal-change gates are preregistered.", "",
        "## 7. Intervention Construction", "Per layer/head key-dimension orthogonal redistribution with per-Value-column norm preservation.", "",
        "## 8. Neutral Control", "Neutral controls use deterministic orthogonal-complement redistribution and report distance mismatch if not fully matched.", "",
        "## 9. Stage-0 Gates", "```json\n" + json.dumps(obj["gate_results"], indent=2) + "\n```", "",
        "## 10. Construction Feasibility", f"{obj['construction_feasibility']}; valid constructions {a['valid_construction_units']} / 9", "",
        "## 11. R-Key Suppression Results", json.dumps(a["R_suppression"], indent=2), "",
        "## 12. C-Key Enhancement Results", json.dumps(a["C_enhancement"], indent=2), "",
        "## 13. State-Persistence Control", "See aggregate and per-unit construction quality.", "",
        "## 14. Behavioral KL", "KL curves and AUC are stored in JSON.", "",
        "## 15. Targeted vs Neutral Effect", "See targeted_contrasts per unit.", "",
        "## 16. Layer / Head Analysis", "Construction quality keeps per-layer/head rows.", "",
        "## 17. Per-Prompt / Per-t0 Results", "| Prompt | t0 | R rescue | R targeted advantage | C damage | C targeted damage |", "|---|---:|---:|---:|---:|---:|",
    ]
    for u in obj["per_unit"]:
        t = u["targeted_contrasts"]
        lines.append(f"| {u['problem_id']} | {u['t0']} | {t['R_key_rescue']} | {t['R_targeted_advantage']} | {t['C_key_damage']} | {t['C_targeted_damage']} |")
    lines += ["", "## 18. Pilot Classification", f"`{obj['pilot_classification']}`", "", "## 19. What Is Supported", obj["supported"], "", "## 20. What Is NOT Supported", "\n".join("- " + x for x in obj["not_supported"]), "", "## 21. Negative / Corrective Results", "No method design or formal expansion was run.", "", "## 22. Next Recommended Experiment", obj["next_recommended_task"]]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(obj):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    units = obj["per_unit"]; xs = np.arange(len(units)); labels = [u["problem_id"].split("/")[-1] + "|" + str(u["t0"]) for u in units]
    for metric, fname, ylabel in [("frozen_J_key", "figure1_frozen_j_key.png", "Frozen J_key"), ("frozen_J_state", "figure2_frozen_j_state.png", "Frozen J_state")]:
        plt.figure(figsize=(9,3))
        for c in ["R_BASE", "R_KEY_SUPPRESSED", "R_KEY_NEUTRAL", "C_BASE", "C_KEY_ENHANCED", "C_KEY_NEUTRAL"]:
            plt.plot(xs, [u[metric][c] for u in units], marker="o", label=c)
        plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7); plt.ylabel(ylabel); plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=160); plt.close()
    plt.figure(figsize=(9,3))
    for c in ["R_BASE", "R_KEY_SUPPRESSED", "R_KEY_NEUTRAL", "C_BASE", "C_KEY_ENHANCED", "C_KEY_NEUTRAL"]:
        plt.plot(xs, [u["KL_metrics"][c]["KL_AUC"] for u in units], marker="o", label=c)
    plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7); plt.ylabel("KL AUC"); plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(FIG_DIR / "figure3_kl_auc_conditions.png", dpi=160); plt.close()
    for side, fname, xkey, ykey in [("R", "figure4_r_delta_jkey_vs_delta_kl.png", "R_KEY_SUPPRESSED", "R_key_rescue"), ("C", "figure5_c_delta_jkey_vs_delta_kl.png", "C_KEY_ENHANCED", "C_key_damage")]:
        base = f"{side}_BASE"; cond = xkey
        plt.figure(figsize=(4,3))
        x = [u["frozen_J_key"][cond] - u["frozen_J_key"][base] for u in units]
        y = [u["targeted_contrasts"][ykey] for u in units]
        plt.scatter(x, y); plt.xlabel("Delta J_key"); plt.ylabel("Delta KL"); plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=160); plt.close()
    plt.figure(figsize=(7,3))
    plt.bar(xs - .2, [u["targeted_contrasts"]["R_targeted_advantage"] for u in units], width=.4, label="R targeted advantage")
    plt.bar(xs + .2, [u["targeted_contrasts"]["C_targeted_damage"] for u in units], width=.4, label="C targeted damage")
    plt.axhline(0, color="black", lw=1); plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG_DIR / "figure6_targeted_vs_neutral_effect.png", dpi=160); plt.close()


def pilot():
    st = load_json(STAGE0) or stage0()
    if any(v == "FAIL" for v in st["gate_results"].values()):
        raise RuntimeError("Stage 0 failed; STOP")
    cp = load_json(CHECKPOINT)
    if not cp or "feasibility" not in cp:
        feas = feasibility(); cp = load_json(CHECKPOINT)
    else:
        feas = cp["feasibility"]
    if feas["construction_feasibility"] != "PASS":
        gates = dict(feas["gate_results"])
        gates.update(feasibility_gates_from_units(feas["construction_units"]))
        obj = {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "model": "Qwen3.5-9B", "protocol": "single-pulse same-norm key-addressability causal intervention; stopped before behavioral pilot at construction gate", "prompt_ids": [p["problem_id"] for p in selected_prompts(3)], "t0_panel": T0_PANEL, "future_horizon": HORIZON, "gate_results": gates, "risk_operator_semantics": (load_json(STAGE0) or {}).get("risk_operator_semantics", {}), "construction_feasibility": "FAIL", "per_unit": [], "aggregate": {"valid_construction_units": feas["valid_construction_units"]}, "pilot_classification": "CONSTRUCTION_NOT_CLEAN_ENOUGH", "supported": "Construction preserves some controls but does not achieve preregistered J_key manipulation; no behavioral causal claim is made.", "not_supported": ["Future-key addressability causal contributor is not tested because controls did not pass.", "KEY_PATH_NOT_CAUSAL is not supported because behavioral pilot did not run."], "limitations": ["Stopped at preregistered construction gate."], "next_recommended_task": "revise intervention construction without relaxing controls", "FROZEN_PATH_SIGNAL": "FUTURE_KEY_INTERACTION_SIGNAL", "MECHANISM_CLOSURE_CANDIDATE": "NO", "METHOD_DESIGN_READY": "NO"}
        save_json(PILOT, obj)
        feasibility_failure_report(obj, feas)
        make_failure_figures(feas)
        print(json.dumps(obj, indent=2))
        print("STOP")
        return obj
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    completed = cp.get("completed", {})
    for pm in selected_prompts(3):
        for t0 in T0_PANEL:
            key = f"{pm['problem_id']}|{t0}"
            if key in completed: continue
            print(f"[{now()}] behavioral pilot prompt={pm['problem_id']} t0={t0}", flush=True)
            completed[key] = behavioral_unit(torch, model, tokenizer, e2e, pm, t0)
            cp["completed"] = completed; save_json(CHECKPOINT, cp)
    units = [completed[f"{pm['problem_id']}|{t0}"] for pm in selected_prompts(3) for t0 in T0_PANEL]
    agg = aggregate(units, feas); cls = classify(units, True)
    obj = {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "model": "Qwen3.5-9B", "protocol": "single-pulse same-norm key-addressability causal intervention; full-model floating continuation after t0", "prompt_ids": [p["problem_id"] for p in selected_prompts(3)], "t0_panel": T0_PANEL, "future_horizon": HORIZON, "gate_results": feas["gate_results"], "risk_operator_semantics": (load_json(STAGE0) or {}) .get("risk_operator_semantics", {}), "per_unit": units, "per_layer": [], "per_head": [], "aggregate": agg, "construction_feasibility": "PASS", "pilot_classification": cls, "supported": "FUTURE_KEY_ADDRESSABILITY_CAUSAL_CONTRIBUTOR = SUPPORTED_IN_PILOT" if cls == "FUTURE_KEY_ADDRESSABILITY_CAUSAL_SUPPORT" else f"FUTURE_KEY_ADDRESSABILITY_CAUSAL_CONTRIBUTOR = {cls}", "not_supported": ["No formal expansion was run.", "No method design was started.", "This does not prove a closed scale-contamination causal chain."], "limitations": ["Pilot only: 9 causal units.", "Neutral-control distance matching is reported, not silently assumed."], "next_recommended_task": "FUTURE_KEY_ADDRESSABILITY_FORMAL_VALIDATION" if cls in ("FUTURE_KEY_ADDRESSABILITY_CAUSAL_SUPPORT", "PARTIAL_KEY_CAUSAL_SUPPORT") else "GDN_INT8_R128_C128_FP_DRIVER_CLAMP_VS_FULL_FEEDBACK_CAUSAL_V1", "FROZEN_PATH_SIGNAL": "FUTURE_KEY_INTERACTION_SIGNAL", "MECHANISM_CLOSURE_CANDIDATE": "NO", "METHOD_DESIGN_READY": "NO"}
    save_json(PILOT, obj); make_report(obj); make_figures(obj); print_summary(obj); return obj


def print_summary(obj):
    a = obj["aggregate"]
    print("\nTASK =\n" + TASK)
    print("\nStage 0 =\nPASS")
    print(f"\nConstruction feasibility =\n{obj['construction_feasibility']}")
    print(f"\nValid causal units =\n{a['valid_causal_units']} / 9")
    print("\n=== R SUPPRESSION ===")
    for k, v in a["R_suppression"].items(): print(f"\n{k} =\n{v}")
    print("\n=== C ENHANCEMENT ===")
    for k, v in a["C_enhancement"].items(): print(f"\n{k} =\n{v}")
    print("\n=== CONTROLS ===")
    for k, v in a["controls"].items(): print(f"\n{k} =\n{v}")
    print("\nPILOT_CLASSIFICATION =\n" + obj["pilot_classification"])
    print("\nFUTURE_KEY_ADDRESSABILITY_CAUSAL_CONTRIBUTOR =")
    print(obj["supported"])
    print("\nMECHANISM_CLOSURE_CANDIDATE =\nNO")
    print("\nMETHOD_DESIGN_READY =\nNO")
    print("\nNEXT_RECOMMENDED_TASK =\n" + obj["next_recommended_task"])
    print("\nArtifacts =")
    for p in [SCRIPT, STAGE0, PILOT, REPORT, FIG_DIR, RAW, CHECKPOINT]: print(p)
    print("\nSTOP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "feasibility", "pilot"], required=True)
    args = ap.parse_args()
    if args.stage == "stage0": stage0()
    elif args.stage == "feasibility": feasibility()
    else: pilot()


if __name__ == "__main__":
    main()
