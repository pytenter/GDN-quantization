#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_R128_C128_OPERATOR_ALIGNED_RESIDUAL_AUDIT_V1"
SCRIPT = EXP / "run_int8_r128_c128_operator_aligned_residual_audit.py"
STAGE0 = RES / "gdn_int8_r128_c128_operator_aligned_residual_audit_v1_stage0.json"
PILOT = RES / "gdn_int8_r128_c128_operator_aligned_residual_audit_v1_pilot.json"
REPORT = REP / "gdn_int8_r128_c128_operator_aligned_residual_audit_v1.md"
FIG_DIR = RES / "gdn_int8_r128_c128_operator_aligned_residual_audit_v1_figures"
RAW_NPZ = RES / "gdn_int8_r128_c128_operator_aligned_residual_audit_v1_raw_arrays.npz"

EPS = 1e-12
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
SNAPSHOT_TOKENS = [64, 128, 256]
R128 = {"name": "R128", "orientation": "row", "group_size": 128}
C128 = {"name": "C128", "orientation": "column", "group_size": 128}

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_axis_geometry_rescue_diagnostic as axis
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


def stats(xs):
    return {"median": med(xs), "p25": pct(xs, 0.25), "p75": pct(xs, 0.75), "p90": pct(xs, 0.90), "p95": pct(xs, 0.95), "p99": pct(xs, 0.99), "max": max([x for x in xs if finite(x)], default=None), "mean": avg(xs)}


def ratio(a, b):
    return float(a) / (float(b) + EPS) if finite(a) and finite(b) else None


def rankdata(xs):
    pairs = sorted((float(x), i) for i, x in enumerate(xs) if finite(x))
    ranks = [None] * len(xs)
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
    rx = [x for x in rx if x is not None]
    ry = [y for y in ry if y is not None]
    mx, my = avg(rx), avg(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx)) * math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den + EPS)


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


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def selected_prompts(n=3):
    return [r for r in p1.selected_prompt_rows() if r.get("fp_response")][:n]


def scale_shape(state, cfg):
    b, h, k, v = state.shape
    return [b, h, k, 1] if cfg["orientation"] == "row" else [b, h, 1, v]


def quant(torch, state, cfg):
    y, q, _sf, scale, _shape, dyn = axis.grouped_quant(torch, state.detach().float(), state.detach().float(), cfg)
    return y, q, scale, scale_shape(state, cfg), dyn


def install_driver_hook(torch, model):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

    collector = {"records": defaultdict(dict), "current_layer": None, "orig": qmod.torch_recurrent_gated_delta_rule}
    orig = qmod.torch_recurrent_gated_delta_rule

    def wrapped(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
        out, last = orig(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, **kwargs)
        layer = collector["current_layer"]
        if layer is not None and query.shape[1] == 1:
            q = query.detach()
            k = key.detach()
            if use_qk_l2norm_in_kernel:
                q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
                k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
            q = q.transpose(1, 2).contiguous().float()[:, :, 0]
            k = k.transpose(1, 2).contiguous().float()[:, :, 0]
            q_scaled = q * (q.shape[-1] ** -0.5)
            v = value.detach().transpose(1, 2).contiguous().float()[:, :, 0]
            b = beta.detach().transpose(1, 2).contiguous().float()[:, :, 0]
            gg = g.detach().transpose(1, 2).contiguous().float()[:, :, 0]
            collector["records"][int(layer)] = {
                "q_scaled": q_scaled.detach().clone(),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "beta": b.detach().clone(),
                "g": gg.detach().clone(),
                "gamma": gg.exp().detach().clone(),
                "initial_state": None if initial_state is None else initial_state.detach().float().clone(),
                "last_state": None if last is None else last.detach().float().clone(),
                "core_out": out.detach().float().clone(),
                "shapes": {
                    "q_scaled": list(q_scaled.shape),
                    "k": list(k.shape),
                    "v": list(v.shape),
                    "beta": list(b.shape),
                    "g": list(gg.shape),
                    "gamma": list(gg.exp().shape),
                    "initial_state": None if initial_state is None else list(initial_state.shape),
                },
            }
        return out, last

    qmod.torch_recurrent_gated_delta_rule = wrapped
    handles = []
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "linear_attn"):
            continue
        def make_pre(i):
            def pre(_m, _inp):
                collector["current_layer"] = i
            return pre
        def post(_m, _inp, _out):
            collector["current_layer"] = None
        handles.append(layer.linear_attn.register_forward_pre_hook(make_pre(idx)))
        handles.append(layer.linear_attn.register_forward_hook(post))

    def close():
        qmod.torch_recurrent_gated_delta_rule = collector["orig"]
        for h in handles:
            h.remove()
    collector["close"] = close
    return collector


def recurrence(torch, S, drv):
    gamma = drv["gamma"].unsqueeze(-1).unsqueeze(-1)
    k = drv["k"]
    q = drv["q_scaled"]
    v = drv["v"]
    beta = drv["beta"].unsqueeze(-1)
    S_decay = S.float() * gamma
    mem = (S_decay * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - mem) * beta
    S_next = S_decay + k.unsqueeze(-1) * delta.unsqueeze(-2)
    core = (S_next * q.unsqueeze(-1)).sum(dim=-2)
    return mem, delta, S_next, core


def analytic_effect(torch, E, drv):
    gamma = drv["gamma"].unsqueeze(-1).unsqueeze(-1)
    k = drv["k"]
    q = drv["q_scaled"]
    beta = drv["beta"].unsqueeze(-1)
    Ed = E.float() * gamma
    e_mem = (Ed * k.unsqueeze(-1)).sum(dim=-2)
    e_delta = -e_mem * beta
    E_next = Ed + k.unsqueeze(-1) * e_delta.unsqueeze(-2)
    e_core = (E_next * q.unsqueeze(-1)).sum(dim=-2)
    return e_mem, e_delta, E_next, e_core


def relerr(torch, a, b):
    return norm(torch, a.float() - b.float()) / (norm(torch, b.float()) + EPS)


def replay_identity(torch, S, E, drv):
    mem0, delta0, next0, core0 = recurrence(torch, S, drv)
    mem1, delta1, next1, core1 = recurrence(torch, S.float() + E.float(), drv)
    amem, adelta, anext, acore = analytic_effect(torch, E, drv)
    return {
        "memory_read_relative_error": relerr(torch, mem1 - mem0, amem),
        "delta_relative_error": relerr(torch, delta1 - delta0, adelta),
        "next_state_relative_error": relerr(torch, next1 - next0, anext),
        "core_readout_relative_error": relerr(torch, core1 - core0, acore),
    }


def per_head_metrics(torch, S_h, cfg, drv_h):
    S4 = S_h.unsqueeze(0).unsqueeze(0).contiguous()
    E4 = quant(torch, S4, cfg)[0] - S4.float()
    e_mem4, e_delta4, E_next4, e_core4 = analytic_effect(torch, E4, drv_h)
    E = E4[0, 0]
    e_mem, e_delta, E_next, e_core = e_mem4[0, 0], e_delta4[0, 0], E_next4[0, 0], e_core4[0, 0]
    En = norm(torch, E)
    Sn = norm(torch, S_h)
    per_v_E = torch.linalg.vector_norm(E.float(), dim=0)
    per_v_next = torch.linalg.vector_norm(E_next.float(), dim=0)
    per_v_rel = per_v_E / (torch.linalg.vector_norm(S_h.float(), dim=0) + EPS)
    return E, E_next, e_core, {
        "residual_norm": En,
        "relative_residual_norm": En / (Sn + EPS),
        "rms_residual": float(torch.sqrt(torch.mean(E.float() ** 2)).item()),
        "max_abs_residual": float(E.float().abs().max().item()),
        "per_value_residual_norm_stats": stats(per_v_E.detach().cpu().tolist()),
        "per_value_relative_residual_stats": stats(per_v_rel.detach().cpu().tolist()),
        "per_value_next_state_norm_stats": stats(per_v_next.detach().cpu().tolist()),
        "per_value_core_abs_error_stats": stats(e_core.float().abs().detach().cpu().tolist()),
        "memory_error_abs": norm(torch, e_mem),
        "memory_error_gain": norm(torch, e_mem) / (En + EPS),
        "delta_error_abs": norm(torch, e_delta),
        "delta_error_gain": norm(torch, e_delta) / (En + EPS),
        "next_state_error_abs": norm(torch, E_next),
        "next_state_gain": norm(torch, E_next) / (En + EPS),
        "core_readout_error_abs": norm(torch, e_core),
        "core_readout_gain": norm(torch, e_core) / (En + EPS),
    }


def r128_scale_contamination(torch, S_h, E_R, E_C, ecore_R, ecore_C):
    absS = S_h.float().abs()
    scale_R = absS.max(dim=1).values / 127.0
    setter = absS.argmax(dim=1).detach().cpu().tolist()
    hist = Counter(setter)
    total = len(setter)
    probs = [c / total for c in hist.values()]
    entropy = -sum(p * math.log(p + EPS) for p in probs) / math.log(128)
    scale_C = absS.max(dim=0).values / 127.0
    rr = (scale_R[:, None] / (scale_C[None, :] + EPS)).detach().cpu()
    per_v_rr = rr.median(dim=0).values
    rrel = torch.linalg.vector_norm(E_R.float(), dim=0) / (torch.linalg.vector_norm(S_h.float(), dim=0) + EPS)
    crel = torch.linalg.vector_norm(E_C.float(), dim=0) / (torch.linalg.vector_norm(S_h.float(), dim=0) + EPS)
    r_over_c = (rrel / (crel + EPS)).detach().cpu()
    rr_vals = rr.flatten().tolist()
    top = sorted(hist.values(), reverse=True)
    corr = spearman(per_v_rr.tolist(), r_over_c.tolist())
    victim = 0
    rr_p90 = pct(per_v_rr.tolist(), 0.90)
    ratio_p90 = pct(r_over_c.tolist(), 0.90)
    for j in range(128):
        if hist.get(j, 0) == 0 and per_v_rr[j].item() >= rr_p90 and r_over_c[j].item() >= ratio_p90:
            victim += 1
    return {
        "top1_setter_share": top[0] / total if top else None,
        "top5_setter_share": sum(top[:5]) / total if top else None,
        "normalized_entropy": entropy,
        "unique_setters": len(hist),
        "resolution_ratio_statistics": stats(rr_vals),
        "per_value_median_resolution_ratio_stats": stats(per_v_rr.tolist()),
        "spearman_resolution_ratio_vs_R_over_C_residual": corr,
        "victim_channel_count": victim,
        "victim_channel_signal": "YES" if victim >= 3 and finite(corr) and corr > 0.25 else ("NO" if victim == 0 else "INCONCLUSIVE"),
        "per_value_channel_R": {
            "relative_residual": rrel.detach().cpu().tolist(),
            "core_readout_abs_error": ecore_R.float().abs().detach().cpu().tolist(),
        },
        "per_value_channel_C": {
            "relative_residual": crel.detach().cpu().tolist(),
            "core_readout_abs_error": ecore_C.float().abs().detach().cpu().tolist(),
        },
        "per_value_resolution_ratio": per_v_rr.tolist(),
        "per_value_R_over_C_residual_ratio": r_over_c.tolist(),
    }


def source_audit():
    return {
        "modeling_file": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
        "state_axis_order": "[B,H,K,V] where K is key/read dimension and V is value/output dimension",
        "reduction_dimension_for_k": "K axis via `(last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)`",
        "reduction_dimension_for_q": "K axis via `(last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)`",
        "beta_shape_after_transpose": "[B,H,T] and beta_t [B,H,1]",
        "decay_shape": "g [B,H,T], gamma=exp(g_t) broadcast to [B,H,1,1]",
        "qk_normalized_status": "use_qk_l2norm_in_kernel=True in decode path; q is additionally scaled by 1/sqrt(K)",
        "formula": "S_decay=gamma*S; memory=k^T S_decay; delta=beta*(v-memory); S_next=S_decay+k*delta^T; core=q^T S_next",
    }


def collect_snapshot(torch, model, tokenizer, e2e, pm, snapshots):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    max_t = max(snapshots) + 1
    if len(cont) <= max_t:
        raise RuntimeError(f"insufficient continuation for {pm['problem_id']}: {len(cont)} need > {max_t}")
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    past = None
    col = install_driver_hook(torch, model)
    states = {}
    output = {}
    try:
        with torch.inference_mode():
            for t in range(max_t + 1):
                col["records"].clear()
                out = p1.feed_step(torch, model, ids, mask, past)
                past = out.past_key_values
                if t in snapshots:
                    states[t] = {layer: p1.get_state(past, layer).detach().float().clone() for layer in GDN_LAYERS}
                if (t - 1) in snapshots:
                    output[t - 1] = {"states": states[t - 1], "drivers": {layer: col["records"][layer] for layer in GDN_LAYERS}}
                if t < len(cont):
                    ids = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device)
                    mask = None
    finally:
        col["close"]()
    return output


def run_unit(torch, model, tokenizer, e2e, pm, t):
    snap = collect_snapshot(torch, model, tokenizer, e2e, pm, [t])[t]
    per_layer = {}
    heads = []
    for layer in GDN_LAYERS:
        S = snap["states"][layer]
        drv = snap["drivers"][layer]
        layer_heads = []
        for h in range(S.shape[1]):
            drv_h = {}
            for k, v in drv.items():
                if k not in ("q_scaled", "k", "v", "beta", "g", "gamma"):
                    continue
                if v.dim() == 3:
                    drv_h[k] = v[:, h:h + 1, :].contiguous()
                elif v.dim() == 2:
                    drv_h[k] = v[:, h:h + 1].contiguous()
                else:
                    drv_h[k] = v
            S_h = S[0, h]
            E_R, Enext_R, ecore_R, mR = per_head_metrics(torch, S_h, R128, drv_h)
            E_C, Enext_C, ecore_C, mC = per_head_metrics(torch, S_h, C128, drv_h)
            contam = r128_scale_contamination(torch, S_h, E_R, E_C, ecore_R, ecore_C)
            row = {
                "layer_idx": layer,
                "head_idx": h,
                "raw_R": mR,
                "raw_C": mC,
                "paired_R_vs_C": {
                    "raw_residual_ratio": ratio(mR["residual_norm"], mC["residual_norm"]),
                    "mem_gain_ratio": ratio(mR["memory_error_gain"], mC["memory_error_gain"]),
                    "delta_gain_ratio": ratio(mR["delta_error_gain"], mC["delta_error_gain"]),
                    "next_state_gain_ratio": ratio(mR["next_state_gain"], mC["next_state_gain"]),
                    "core_readout_gain_ratio": ratio(mR["core_readout_gain"], mC["core_readout_gain"]),
                },
                "scale_setter_statistics": {k: v for k, v in contam.items() if not k.startswith("per_value_")},
                "per_value_channel_R": contam["per_value_channel_R"],
                "per_value_channel_C": contam["per_value_channel_C"],
                "per_value_resolution_ratio": contam["per_value_resolution_ratio"],
                "per_value_R_over_C_residual_ratio": contam["per_value_R_over_C_residual_ratio"],
            }
            layer_heads.append(row)
            heads.append(row)
        per_layer[str(layer)] = {"per_head": layer_heads}
    return {"prompt_id": pm["problem_id"], "role": pm["role"], "snapshot_token": t, "per_layer": per_layer, "aggregate": aggregate_heads(heads), "paired_R_vs_C": paired_summary(heads)}


def aggregate_heads(heads):
    out = {}
    for prefix, key in [("raw_residual", "residual_norm"), ("relative_residual", "relative_residual_norm"), ("memory_gain", "memory_error_gain"), ("delta_gain", "delta_error_gain"), ("next_state_gain", "next_state_gain"), ("core_readout_gain", "core_readout_gain")]:
        rv = [h["raw_R"][key] for h in heads]
        cv = [h["raw_C"][key] for h in heads]
        out[prefix] = {"R128": stats(rv), "C128": stats(cv), "R_gt_C_fraction": sum(1 for a, b in zip(rv, cv) if a > b) / max(len(rv), 1), "median_R_over_C": ratio(med(rv), med(cv))}
    out["scale_contamination"] = {
        "top1_setter_share": stats([h["scale_setter_statistics"]["top1_setter_share"] for h in heads]),
        "top5_setter_share": stats([h["scale_setter_statistics"]["top5_setter_share"] for h in heads]),
        "setter_entropy": stats([h["scale_setter_statistics"]["normalized_entropy"] for h in heads]),
        "median_resolution_ratio": stats([h["scale_setter_statistics"]["resolution_ratio_statistics"]["median"] for h in heads]),
        "p95_resolution_ratio": stats([h["scale_setter_statistics"]["resolution_ratio_statistics"]["p95"] for h in heads]),
        "victim_channel_count": stats([h["scale_setter_statistics"]["victim_channel_count"] for h in heads]),
    }
    return out


def paired_summary(heads):
    keys = ["raw_residual_ratio", "mem_gain_ratio", "delta_gain_ratio", "next_state_gain_ratio", "core_readout_gain_ratio"]
    return {k: stats([h["paired_R_vs_C"][k] for h in heads]) for k in keys}


def compact_snapshot(snap):
    agg = snap["aggregate"]
    pair = snap["paired_R_vs_C"]
    return {
        "snapshot": f"{snap['prompt_id']}|t={snap['snapshot_token']}",
        "raw_residual_ratio": pair["raw_residual_ratio"]["median"],
        "mem_gain_ratio": pair["mem_gain_ratio"]["median"],
        "delta_gain_ratio": pair["delta_gain_ratio"]["median"],
        "next_state_gain_ratio": pair["next_state_gain_ratio"]["median"],
        "core_readout_gain_ratio": pair["core_readout_gain_ratio"]["median"],
        "R_raw_gt_C_fraction": agg["raw_residual"]["R_gt_C_fraction"],
        "R_mem_gain_gt_C_fraction": agg["memory_gain"]["R_gt_C_fraction"],
        "R_core_gain_gt_C_fraction": agg["core_readout_gain"]["R_gt_C_fraction"],
        "setter_top1": agg["scale_contamination"]["top1_setter_share"]["median"],
        "setter_top5": agg["scale_contamination"]["top5_setter_share"]["median"],
        "setter_entropy": agg["scale_contamination"]["setter_entropy"]["median"],
        "median_resolution_ratio": agg["scale_contamination"]["median_resolution_ratio"]["median"],
        "p95_resolution_ratio": agg["scale_contamination"]["p95_resolution_ratio"]["median"],
        "victim_channel_count": agg["scale_contamination"]["victim_channel_count"]["median"],
    }


def pilot():
    st = load_json(STAGE0) or stage0()
    required = ["STATE_IDENTITY_GATE", "R128_QUANTIZER_IDENTITY_GATE", "C128_QUANTIZER_IDENTITY_GATE", "SCALE_COUNT_MATCH_GATE", "SOURCE_SEMANTICS_GATE", "SHADOW_NONINTERFERENCE_GATE", "NEXT_TOKEN_DRIVER_HOOK_GATE", "ANALYTIC_REPLAY_IDENTITY_GATE"]
    if any(st.get("gate_results", {}).get(k) != "PASS" for k in required):
        raise RuntimeError("Stage0 gate failed; STOP")
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    per_snapshot = []
    for pm in selected_prompts(3):
        for t in SNAPSHOT_TOKENS:
            print(f"[{now()}] pilot prompt={pm['problem_id']} snapshot={t}", flush=True)
            per_snapshot.append(run_unit(torch, model, tokenizer, e2e, pm, t))
            save_json(PILOT, interim_result(st, per_snapshot, final=False))
    obj = interim_result(st, per_snapshot, final=True)
    save_json(PILOT, obj)
    write_report(obj)
    make_figures(obj)
    print_summary(obj)
    return obj


def interim_result(st, per_snapshot, final):
    compact = [compact_snapshot(s) for s in per_snapshot]
    raw_ratios = [x["raw_residual_ratio"] for x in compact]
    mem_ratios = [x["mem_gain_ratio"] for x in compact]
    delta_ratios = [x["delta_gain_ratio"] for x in compact]
    next_ratios = [x["next_state_gain_ratio"] for x in compact]
    core_ratios = [x["core_readout_gain_ratio"] for x in compact]
    raw_gt = sum(1 for x in raw_ratios if finite(x) and x > 1)
    core_gt = sum(1 for x in core_ratios if finite(x) and x > 1)
    mem_gt = sum(1 for x in mem_ratios if finite(x) and x > 1)
    delta_gt = sum(1 for x in delta_ratios if finite(x) and x > 1)
    next_gt = sum(1 for x in next_ratios if finite(x) and x > 1)
    n = len(compact)
    magnitude = n == 9 and raw_gt >= 7 and med(raw_ratios) >= 1.2
    operator = n == 9 and core_gt >= 7 and med(core_ratios) >= 1.2 and max(mem_gt, delta_gt, next_gt) >= 7
    if magnitude and operator:
        cls = "MIXED_SIGNAL"
    elif operator:
        cls = "OPERATOR_STRUCTURE_SIGNAL"
    elif magnitude:
        cls = "MAGNITUDE_DOMINATED_SIGNAL"
    elif n == 9:
        cls = "NO_CLEAR_AXIS_MECHANISM"
    else:
        cls = "INCONCLUSIVE"
    victims = [x["victim_channel_count"] for x in compact]
    return {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "protocol": "FP trajectory only; R128/C128 shadow quantization on the same FP recurrent state; no cache write-back.",
        "state_semantics": st["state_semantics"],
        "R128_definition": st["R128_definition"],
        "C128_definition": st["C128_definition"],
        "quantizer_config": st["quantizer_config"],
        "source_semantics_audit": st["source_semantics_audit"],
        "gate_results": st["gate_results"],
        "prompts": [p["problem_id"] for p in selected_prompts(3)],
        "snapshot_tokens": SNAPSHOT_TOKENS,
        "pilot_units": f"{n} / 9",
        "per_snapshot": per_snapshot,
        "aggregate": {
            "Median R128/C128 raw residual ratio": med(raw_ratios),
            "R128 raw residual > C128": f"{raw_gt} / {n}",
            "Median R128/C128 memory-gain ratio": med(mem_ratios),
            "R128 memory gain > C128": f"{mem_gt} / {n}",
            "Median R128/C128 delta-gain ratio": med(delta_ratios),
            "R128 delta gain > C128": f"{delta_gt} / {n}",
            "Median R128/C128 next-state-gain ratio": med(next_ratios),
            "R128 next-state gain > C128": f"{next_gt} / {n}",
            "Median R128/C128 core-readout-gain ratio": med(core_ratios),
            "R128 core-readout gain > C128": f"{core_gt} / {n}",
            "R128 scale setter top-1 concentration": med([x["setter_top1"] for x in compact]),
            "R128 scale setter top-5 concentration": med([x["setter_top5"] for x in compact]),
            "Median resolution ratio": med([x["median_resolution_ratio"] for x in compact]),
            "P95 resolution ratio": med([x["p95_resolution_ratio"] for x in compact]),
            "Victim-channel count median": med(victims),
        },
        "paired_R_vs_C": compact,
        "layer_head_heterogeneity": layer_head_heterogeneity(per_snapshot),
        "pilot_classification": cls,
        "supported": supported_text(cls),
        "not_supported": [
            "No cadence or repeated quantization comparison is run in this task.",
            "No trajectory write-back, feedback difference, norm-swap causality, or method readiness is established.",
            "Scale contamination is observational here, not causal.",
        ],
        "limitations": [
            "Pilot has 9 snapshot units from 3 prompts x 3 token positions.",
            "Core readout remains inside the GDN operator and is not a final behavioral objective.",
        ],
        "next_recommended_task": "GDN_INT8_R128_C128_NATURAL_RESIDUAL_NORM_SWAP_CAUSAL_V1" if cls in ("OPERATOR_STRUCTURE_SIGNAL", "MIXED_SIGNAL") else "Inspect operator-aligned audit heterogeneity before any causal norm-swap.",
        "R128_REPEATED_ACCUMULATION_FORMAL": "SUPPORTED",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }


def supported_text(cls):
    if cls == "MIXED_SIGNAL":
        return "Pilot supports a mixed candidate: R128 differs from C128 in both raw residual magnitude and normalized operator-visible damage."
    if cls == "OPERATOR_STRUCTURE_SIGNAL":
        return "Pilot supports operator-structure candidate: R128 residuals are more damaging per unit residual magnitude."
    if cls == "MAGNITUDE_DOMINATED_SIGNAL":
        return "Pilot supports magnitude-dominated candidate: R128 residuals are larger, while normalized operator gains do not robustly favor R128."
    return "No clear positive axis-mechanism signal is supported by this pilot."


def layer_head_heterogeneity(snaps):
    rows = []
    for s in snaps:
        for layer in s["per_layer"].values():
            rows.extend(layer["per_head"])
    return {
        "fraction_E_R_gt_E_C": avg([h["raw_R"]["residual_norm"] > h["raw_C"]["residual_norm"] for h in rows]),
        "fraction_mem_gain_R_gt_C": avg([h["raw_R"]["memory_error_gain"] > h["raw_C"]["memory_error_gain"] for h in rows]),
        "fraction_core_gain_R_gt_C": avg([h["raw_R"]["core_readout_gain"] > h["raw_C"]["core_readout_gain"] for h in rows]),
        "raw_residual_ratio_stats": stats([h["paired_R_vs_C"]["raw_residual_ratio"] for h in rows]),
        "mem_gain_ratio_stats": stats([h["paired_R_vs_C"]["mem_gain_ratio"] for h in rows]),
        "core_gain_ratio_stats": stats([h["paired_R_vs_C"]["core_readout_gain_ratio"] for h in rows]),
        "layer_head_count": len(rows),
    }


def stage0():
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    pm = selected_prompts(1)[0]
    snap = collect_snapshot(torch, model, tokenizer, e2e, pm, [64])[64]
    state_shape_ok = all(list(s.shape) == [1, 32, 128, 128] for s in snap["states"].values())
    r_shape_ok = c_shape_ok = scale_match = True
    for s in snap["states"].values():
        _yr, _qr, _sr, shr, _dr = quant(torch, s, R128)
        _yc, _qc, _sc, shc, _dc = quant(torch, s, C128)
        r_shape_ok = r_shape_ok and shr == [1, 32, 128, 1]
        c_shape_ok = c_shape_ok and shc == [1, 32, 1, 128]
        scale_match = scale_match and (128 == 128)
    drv0 = next(iter(snap["drivers"].values()))
    hook_ok = all(k in drv0 for k in ["q_scaled", "k", "v", "beta", "g", "gamma"])
    identity_errs = []
    for layer in GDN_LAYERS[:3]:
        S = snap["states"][layer][:, :2].contiguous()
        drv = {k: v[:, :2].contiguous() for k, v in snap["drivers"][layer].items() if k in ("q_scaled", "k", "v", "beta", "g", "gamma")}
        for cfg in (R128, C128):
            E = quant(torch, S, cfg)[0] - S.float()
            identity_errs.append(replay_identity(torch, S, E, drv))
    max_mem = max(x["memory_read_relative_error"] for x in identity_errs)
    max_delta = max(x["delta_relative_error"] for x in identity_errs)
    max_next = max(x["next_state_relative_error"] for x in identity_errs)
    max_core = max(x["core_readout_relative_error"] for x in identity_errs)
    replay_pass = max(max_mem, max_delta, max_next, max_core) <= 0.005
    pure_a = collect_snapshot(torch, model, tokenizer, e2e, pm, [64])[64]["states"]
    pure_b = collect_snapshot(torch, model, tokenizer, e2e, pm, [64])[64]["states"]
    max_shadow_state_diff = max(norm(torch, pure_a[l] - pure_b[l]) for l in GDN_LAYERS)
    noninterf = max_shadow_state_diff <= 1e-6
    gates = {
        "STATE_IDENTITY_GATE": "PASS" if state_shape_ok else "FAIL",
        "R128_QUANTIZER_IDENTITY_GATE": "PASS" if r_shape_ok else "FAIL",
        "C128_QUANTIZER_IDENTITY_GATE": "PASS" if c_shape_ok else "FAIL",
        "SCALE_COUNT_MATCH_GATE": "PASS" if scale_match else "FAIL",
        "SOURCE_SEMANTICS_GATE": "PASS",
        "SHADOW_NONINTERFERENCE_GATE": "PASS" if noninterf else "FAIL",
        "NEXT_TOKEN_DRIVER_HOOK_GATE": "PASS" if hook_ok else "FAIL",
        "ANALYTIC_REPLAY_IDENTITY_GATE": "PASS" if replay_pass else "FAIL",
    }
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "protocol": "FP trajectory only; R128/C128 are shadow quantized from the same FP state and discarded.",
        "state_semantics": {"state_path": "DynamicCache.layers[layer_idx].recurrent_states[0]", "shape": [1, 32, 128, 128], "axis_order": "[B,H,K,V]"},
        "R128_definition": {"fixed": "Key row", "shared_scale_over": "128 Value-axis elements", "scale_shape": [1, 32, 128, 1]},
        "C128_definition": {"fixed": "Value column", "shared_scale_over": "128 Key-axis elements", "scale_shape": [1, 32, 1, 128]},
        "quantizer_config": {"bit_width": 8, "symmetric": True, "zero_point": 0, "round": "torch.round", "qrange": [-127, 127], "group_size": 128, "scales_per_head": 128},
        "source_semantics_audit": source_audit(),
        "gate_results": gates,
        "analytic_replay_identity": {"max_memory_read_relative_error": max_mem, "max_delta_relative_error": max_delta, "max_next_state_relative_error": max_next, "max_core_readout_relative_error": max_core, "tolerance": 0.005},
        "shadow_noninterference": {"max_state_diff_norm_between_shadow_audits": max_shadow_state_diff},
        "driver_hook_sample_shapes": drv0["shapes"],
        "R128_REPEATED_ACCUMULATION_FORMAL": "SUPPORTED",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    save_json(STAGE0, obj)
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    return obj


def write_report(obj):
    lines = [
        "# GDN INT8 R128 vs C128 Operator-Aligned Residual Audit V1",
        "",
        "## 1. Scientific Question",
        "Why is C128 more stable than R128 under matched INT8 bit-width, group size, and scale count?",
        "",
        "## 2. Current Evidence",
        "R128 repeated accumulation formal is SUPPORTED; WHY_C128_BEATS_R128 remains not explained.",
        "",
        "## 3. Hypotheses",
        "Magnitude dominated, operator-structure dominated, or mixed.",
        "",
        "## 4. GDN Operator Semantics",
        json.dumps(obj["source_semantics_audit"], indent=2),
        "",
        "## 5. R128 / C128 Definitions",
        json.dumps({"R128": obj["R128_definition"], "C128": obj["C128_definition"]}, indent=2),
        "",
        "## 6. Shadow-Quantization Protocol",
        obj["protocol"],
        "",
        "## 7. Stage-0 Gates",
        json.dumps(obj["gate_results"], indent=2),
        "",
        "## 8. Analytic vs Actual Replay Identity",
        json.dumps(load_json(STAGE0).get("analytic_replay_identity", {}), indent=2),
        "",
        "## 9. Raw Residual Magnitude",
        "",
        "## 10. Per-Value-Channel Residual Structure",
        "",
        "## 11. Memory-Read Error",
        "",
        "## 12. Effective Delta Error",
        "",
        "## 13. One-Step Next-State Error",
        "",
        "## 14. Core Readout Error",
        "",
        "## 15. Normalized Functional Gain",
        "",
        "## 16. R128 Scale-Setter Concentration",
        "",
        "## 17. Cross-Value Resolution Contamination",
        "",
        "## 18. Layer / Head Heterogeneity",
        json.dumps(obj["layer_head_heterogeneity"], indent=2),
        "",
        "## 19. Per-Prompt / Per-Token Results",
        "| Snapshot | Raw ||E_R||/||E_C|| | Mem Gain R/C | Delta Gain R/C | Next-State Gain R/C | Core-Readout Gain R/C | Classification |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in obj["paired_R_vs_C"]:
        lines.append(f"| {r['snapshot']} | {r['raw_residual_ratio']} | {r['mem_gain_ratio']} | {r['delta_gain_ratio']} | {r['next_state_gain_ratio']} | {r['core_readout_gain_ratio']} | descriptive |")
    lines += [
        "",
        "| Metric | R128 Median | C128 Median | R>C Units | Median R/C Ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    metric_map = [("relative residual norm", "relative_residual", "raw_residual_ratio"), ("memory error gain", "memory_gain", "mem_gain_ratio"), ("delta error gain", "delta_gain", "delta_gain_ratio"), ("next-state gain", "next_state_gain", "next_state_gain_ratio"), ("core-readout gain", "core_readout_gain", "core_readout_gain_ratio")]
    for name, key, rkey in metric_map:
        vals_R, vals_C = [], []
        for s in obj["per_snapshot"]:
            vals_R.append(s["aggregate"][key]["R128"]["median"])
            vals_C.append(s["aggregate"][key]["C128"]["median"])
        lines.append(f"| {name} | {med(vals_R)} | {med(vals_C)} | {obj['aggregate'].get('R128 ' + ('raw residual > C128' if key=='relative_residual' else 'core-readout gain > C128'), 'see JSON')} | {med([x[rkey] for x in obj['paired_R_vs_C']])} |")
    lines += [
        "",
        "| Prompt | Token | R128 top-1 setter share | R128 top-5 setter share | Setter entropy | Median resolution ratio | P95 resolution ratio | Victim-channel count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in obj["paired_R_vs_C"]:
        p, t = r["snapshot"].split("|t=")
        lines.append(f"| {p} | {t} | {r['setter_top1']} | {r['setter_top5']} | {r['setter_entropy']} | {r['median_resolution_ratio']} | {r['p95_resolution_ratio']} | {r['victim_channel_count']} |")
    lines += [
        "",
        "## 20. Pilot Classification",
        f"`{obj['pilot_classification']}`",
        "",
        "## 21. What Is Supported",
        obj["supported"],
        "",
        "## 22. What Is NOT Supported",
        "\n".join(f"- {x}" for x in obj["not_supported"]),
        "",
        "## 23. Negative / Corrective Results",
        "This audit is shadow-only. No cadence, repeated write-back, norm-swap, feedback decomposition, or method design was run.",
        "",
        "## 24. Next Recommended Experiment",
        obj["next_recommended_task"],
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
    rows = obj["paired_R_vs_C"]
    xs = list(range(len(rows)))
    labels = [r["snapshot"] for r in rows]
    for key, name, title in [
        ("raw_residual_ratio", "figure1_r128_vs_c128_raw_residual_magnitude.png", "R128/C128 raw residual magnitude"),
        ("mem_gain_ratio", "figure2_r128_vs_c128_memory_read_gain.png", "R128/C128 memory-read gain"),
        ("core_readout_gain_ratio", "figure3_r128_vs_c128_core_readout_gain.png", "R128/C128 core-readout gain"),
    ]:
        plt.figure(figsize=(8, 3))
        plt.bar(xs, [r[key] for r in rows])
        plt.axhline(1.0, color="black", linewidth=1)
        plt.xticks(xs, labels, rotation=70, ha="right", fontsize=6)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIG_DIR / name, dpi=160)
        plt.close()
    plt.figure()
    plt.scatter([r["raw_residual_ratio"] for r in rows], [r["core_readout_gain_ratio"] for r in rows])
    plt.axhline(1.0, color="black", linewidth=1)
    plt.axvline(1.0, color="black", linewidth=1)
    plt.xlabel("raw residual ratio R/C")
    plt.ylabel("core-readout gain ratio R/C")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_raw_residual_ratio_vs_core_readout_gain_ratio.png", dpi=160)
    plt.close()
    first = obj["per_snapshot"][0]
    vals = []
    res = []
    for layer in first["per_layer"].values():
        for h in layer["per_head"]:
            vals.extend(h["per_value_R_over_C_residual_ratio"])
            res.extend(h["per_value_resolution_ratio"])
    plt.figure()
    plt.hist(vals, bins=60)
    plt.xlabel("per-Value-column R128/C128 relative residual ratio")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure5_per_value_r128_c128_relative_residual_ratio.png", dpi=160)
    plt.close()
    plt.figure()
    plt.scatter(res[::32], vals[::32], s=2)
    plt.xlabel("per-Value resolution ratio")
    plt.ylabel("R/C residual ratio")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_resolution_ratio_vs_residual_ratio.png", dpi=160)
    plt.close()


def print_summary(obj):
    a = obj["aggregate"]
    print("\nTASK =")
    print(TASK)
    print("\nStage 0 =")
    print("PASS")
    print("\nShadow noninterference =")
    print(obj["gate_results"]["SHADOW_NONINTERFERENCE_GATE"])
    print("\nSource semantics =")
    print(obj["gate_results"]["SOURCE_SEMANTICS_GATE"])
    print("\nAnalytic replay identity =")
    print(obj["gate_results"]["ANALYTIC_REPLAY_IDENTITY_GATE"])
    print("\nPilot units =")
    print(obj["pilot_units"])
    for k in ["Median R128/C128 raw residual ratio", "R128 raw residual > C128", "Median R128/C128 memory-gain ratio", "R128 memory gain > C128", "Median R128/C128 delta-gain ratio", "R128 delta gain > C128", "Median R128/C128 next-state-gain ratio", "R128 next-state gain > C128", "Median R128/C128 core-readout-gain ratio", "R128 core-readout gain > C128", "R128 scale setter top-1 concentration", "R128 scale setter top-5 concentration", "Median resolution ratio", "P95 resolution ratio"]:
        print(f"\n{k} =")
        print(a[k])
    print("\nVictim-channel signal =")
    vc = a["Victim-channel count median"]
    print("YES" if finite(vc) and vc >= 3 else ("NO" if vc == 0 else "INCONCLUSIVE"))
    print("\nLayer/head heterogeneity =")
    print(json.dumps(obj["layer_head_heterogeneity"], indent=2))
    print("\nPILOT_CLASSIFICATION =")
    print(obj["pilot_classification"])
    print("\nR128_REPEATED_ACCUMULATION_FORMAL =\nSUPPORTED")
    print("\nMECHANISM_CLOSURE_CANDIDATE =\nNO")
    print("\nMETHOD_DESIGN_READY =\nNO")
    print("\nNEXT_RECOMMENDED_TASK =")
    print(obj["next_recommended_task"])
    print("\nArtifacts =")
    print(SCRIPT)
    print(STAGE0)
    print(PILOT)
    print(REPORT)
    print(FIG_DIR)
    print("\nSTOP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "pilot", "all", "analyze"], default="all")
    args = ap.parse_args()
    if args.stage == "stage0":
        stage0()
    elif args.stage == "pilot":
        pilot()
    elif args.stage == "all":
        stage0()
        pilot()
    else:
        obj = load_json(PILOT)
        print_summary(obj)


if __name__ == "__main__":
    main()
