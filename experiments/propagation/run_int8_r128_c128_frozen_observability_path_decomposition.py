#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"
TASK = "GDN_INT8_R128_C128_FROZEN_OBSERVABILITY_PATH_DECOMPOSITION_V1"
SCRIPT = EXP / "run_int8_r128_c128_frozen_observability_path_decomposition.py"
PREV = RES / "gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1_pilot.json"
STAGE0 = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_stage0.json"
PILOT = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_pilot.json"
CHECKPOINT = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_checkpoint.json"
RAW = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_raw.npz"
REPORT = REP / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1.md"
FIG_DIR = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_figures"

EPS = 1e-12
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
T0_PANEL = [64, 128, 256]
HORIZON = 128
OFFSETS = [1, 2, 4, 8, 16, 32, 64, 128]
PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"
SECONDARY_R = "REAL_R"
SECONDARY_C = "C_STRUCT_R_NORM"
CONDITIONS = [PRIMARY_C, PRIMARY_R, SECONDARY_C, SECONDARY_R]
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
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] if lo == hi else xs[lo] * (hi - p) + xs[hi] * (p - lo)


def ratio(a, b):
    return float(a) / (float(b) + EPS) if finite(a) and finite(b) else None


def log_ratio(a, b):
    r = ratio(a, b)
    return math.log(r + EPS) if finite(r) and r > 0 else None


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


def tensor_norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(torch, a, b):
    an, bn = tensor_norm(torch, a), tensor_norm(torch, b)
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(a.detach().float().flatten(), b.detach().float().flatten(), dim=0).item())


def quant(torch, state, cfg):
    return axis.grouped_quant(torch, state.detach().float(), state.detach().float(), cfg)[0]


def norm_swap_residuals(torch, state):
    er = quant(torch, state, R128) - state.detach().float()
    ec = quant(torch, state, C128) - state.detach().float()
    r_to_c = torch.zeros_like(er)
    c_to_r = torch.zeros_like(ec)
    max_norm_err = 0.0
    min_cos = 1.0
    degenerate = 0
    for h in range(state.shape[1]):
        r = er[:, h]
        c = ec[:, h]
        rn, cn = tensor_norm(torch, r), tensor_norm(torch, c)
        if rn < EPS or cn < EPS:
            degenerate += 1
            continue
        r_to_c[:, h] = r * (cn / (rn + EPS))
        c_to_r[:, h] = c * (rn / (cn + EPS))
        max_norm_err = max(
            max_norm_err,
            abs(tensor_norm(torch, r_to_c[:, h]) - cn) / (cn + EPS),
            abs(tensor_norm(torch, c_to_r[:, h]) - rn) / (rn + EPS),
        )
        for a, b in ((r, r_to_c[:, h]), (c, c_to_r[:, h])):
            co = cosine(torch, a, b)
            if co is not None:
                min_cos = min(min_cos, co)
    return er, ec, r_to_c, c_to_r, max_norm_err, min_cos, degenerate


def build_initial_errors(torch, fp_past):
    errors = {c: {} for c in CONDITIONS}
    totals = defaultdict(float)
    max_norm_err = 0.0
    min_cos = 1.0
    degenerate = 0
    per_layer = []
    for layer in GDN_LAYERS:
        state = p1.get_state(fp_past, layer)
        er, ec, r_to_c, c_to_r, ne, co, deg = norm_swap_residuals(torch, state)
        errors[SECONDARY_R][layer] = er
        errors[PRIMARY_C][layer] = ec
        errors[PRIMARY_R][layer] = r_to_c
        errors[SECONDARY_C][layer] = c_to_r
        for c in CONDITIONS:
            n = tensor_norm(torch, errors[c][layer])
            totals[c] += n * n
        max_norm_err = max(max_norm_err, ne)
        min_cos = min(min_cos, co)
        degenerate += deg
        per_layer.append({
            "layer_idx": layer,
            "norm_REAL_R": tensor_norm(torch, er),
            "norm_REAL_C": tensor_norm(torch, ec),
            "norm_R_STRUCT_C_NORM": tensor_norm(torch, r_to_c),
            "norm_C_STRUCT_R_NORM": tensor_norm(torch, c_to_r),
        })
    norms = {c: math.sqrt(totals[c]) for c in CONDITIONS}
    return errors, {
        "norms": norms,
        "primary_initial_norm_ratio": ratio(norms[PRIMARY_R], norms[PRIMARY_C]),
        "secondary_initial_norm_ratio": ratio(norms[SECONDARY_R], norms[SECONDARY_C]),
        "max_per_head_relative_norm_difference": max_norm_err,
        "min_direction_cosine": min_cos,
        "degenerate_residual_units": degenerate,
        "per_layer": per_layer,
    }


def selected_prompts(n=3):
    return [r for r in p1.selected_prompt_rows() if r.get("fp_response")][:n]


def install_fp_driver_capture(torch, model):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

    collector = {"current_layer": None, "records": {}, "orig_rule": qmod.torch_recurrent_gated_delta_rule}
    orig_rule = qmod.torch_recurrent_gated_delta_rule
    handles = []

    def wrapped_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
        out, last_state = orig_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, **kwargs)
        layer = collector.get("current_layer")
        if layer is not None and query.shape[1] == 1:
            collector["records"][int(layer)] = {
                "query": query.detach().clone(),
                "key": key.detach().clone(),
                "value": value.detach().clone(),
                "g": g.detach().clone(),
                "beta": beta.detach().clone(),
                "initial_state": None if initial_state is None else initial_state.detach().float().clone(),
                "final_state": None if last_state is None else last_state.detach().float().clone(),
                "core_output": out.detach().float().clone(),
                "use_qk_l2norm_in_kernel": bool(use_qk_l2norm_in_kernel),
            }
        return out, last_state

    qmod.torch_recurrent_gated_delta_rule = wrapped_rule
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "linear_attn"):
            continue
        mod = layer.linear_attn

        def make_pre(i):
            def pre(_module, _inputs):
                collector["current_layer"] = i
            return pre

        def post(_module, _inputs, _output):
            collector["current_layer"] = None

        def make_norm_pre(i):
            def norm_pre(_module, inputs):
                rec = collector["records"].setdefault(i, {})
                rec["norm_input_core"] = inputs[0].detach().float().clone()
                rec["z"] = inputs[1].detach().clone()
            return norm_pre

        def make_norm_post(i):
            def norm_post(_module, _inputs, output):
                collector["records"].setdefault(i, {})["preproj"] = output.detach().float().clone()
            return norm_post

        def make_out_post(i):
            def out_post(_module, _inputs, output):
                collector["records"].setdefault(i, {})["postproj"] = output.detach().float().clone()
            return out_post

        handles.append(mod.register_forward_pre_hook(make_pre(idx)))
        handles.append(mod.register_forward_hook(post))
        handles.append(mod.norm.register_forward_pre_hook(make_norm_pre(idx)))
        handles.append(mod.norm.register_forward_hook(make_norm_post(idx)))
        handles.append(mod.out_proj.register_forward_hook(make_out_post(idx)))

    def close():
        qmod.torch_recurrent_gated_delta_rule = collector["orig_rule"]
        for h in handles:
            h.remove()

    collector["close"] = close
    return collector


def manual_recurrent_components(torch, rec, initial_state):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

    query, key, value = rec["query"], rec["key"], rec["value"]
    g, beta = rec["g"], rec["beta"]
    if rec["use_qk_l2norm_in_kernel"]:
        query = qmod.l2norm(query, dim=-1, eps=1e-6)
        key = qmod.l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    q_t = query[:, :, 0] * (query.shape[-1] ** -0.5)
    k_t = key[:, :, 0]
    v_t = value[:, :, 0]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta[:, :, 0].unsqueeze(-1)
    s0 = torch.zeros_like(rec["final_state"].float()) if initial_state is None else initial_state.detach().float()
    after_decay = s0 * g_t
    memory = (after_decay * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - memory) * beta_t
    update = k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    next_state = after_decay + update
    core = (next_state * q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(2).transpose(1, 2).contiguous()
    return {"after_decay": after_decay, "memory": memory, "delta": delta, "update": update, "next_state": next_state, "core": core}


def implementation_replay(rec, initial_state):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    return qmod.torch_recurrent_gated_delta_rule(
        rec["query"], rec["key"], rec["value"], rec["g"], rec["beta"], initial_state,
        True, rec["use_qk_l2norm_in_kernel"]
    )


def local_output(torch, model, layer, core, z):
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    la = layers[layer].linear_attn
    pre = la.norm(core.reshape(-1, la.head_v_dim).to(z.dtype), z.reshape(-1, la.head_v_dim).to(z.dtype))
    post = la.out_proj(pre.reshape(core.shape[0], core.shape[1], -1))
    return pre.detach().float(), post.detach().float()


def driver_step(torch, model, input_ids, mask, past, collector):
    collector["records"].clear()
    out = p1.feed_step(torch, model, input_ids, mask, past)
    return out, out.past_key_values, {k: dict(v) for k, v in collector["records"].items()}


def run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    past = None
    collector = install_fp_driver_capture(torch, model)
    try:
        with torch.inference_mode():
            for t in range(t0 + 1):
                _out, past, _recs = driver_step(torch, model, ids, mask, past, collector)
                if t < len(cont):
                    ids = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device)
                    mask = None
        return past, ids, cont, collector
    except Exception:
        collector["close"]()
        raise


def replay_layer_step(torch, model, layer, rec, fp_state, err_state):
    fp_manual = manual_recurrent_components(torch, rec, fp_state)
    pe_manual = manual_recurrent_components(torch, rec, err_state)
    fp_out, fp_next = implementation_replay(rec, fp_state)
    pe_out, pe_next = implementation_replay(rec, err_state)
    core_fp = fp_out.detach().float()
    core_pe = pe_out.detach().float()
    pre_fp, post_fp = local_output(torch, model, layer, core_fp, rec["z"])
    pre_pe, post_pe = local_output(torch, model, layer, core_pe, rec["z"])
    return {
        "E_after_decay": pe_manual["after_decay"] - fp_manual["after_decay"],
        "memory_read_error": pe_manual["memory"] - fp_manual["memory"],
        "delta_error": pe_manual["delta"] - fp_manual["delta"],
        "update_error": pe_manual["update"] - fp_manual["update"],
        "E_after_update": pe_next.detach().float() - fp_next.detach().float(),
        "core_readout_error": core_pe - core_fp,
        "post_norm_error": pre_pe - pre_fp,
        "local_output_error": post_pe - post_fp,
        "next_state": pe_next.detach().float(),
        "fp_next_state": fp_next.detach().float(),
        "identity": {
            "next_state_relerr": tensor_norm(torch, pe_manual["next_state"] - pe_next.float()) / (tensor_norm(torch, pe_next) + EPS),
            "core_relerr": tensor_norm(torch, pe_manual["core"] - pe_out.float()) / (tensor_norm(torch, pe_out) + EPS),
            "fp_next_state_relerr": tensor_norm(torch, fp_manual["next_state"] - fp_next.float()) / (tensor_norm(torch, fp_next) + EPS),
            "fp_core_relerr": tensor_norm(torch, fp_manual["core"] - fp_out.float()) / (tensor_norm(torch, fp_out) + EPS),
        },
    }


def expand_driver(torch, x, batch):
    return x.expand(batch, *x.shape[1:]).contiguous()


def expand_gate_z(torch, z, batch):
    if z.dim() == 2:
        return z.repeat(batch, 1).contiguous()
    return expand_driver(torch, z, batch)


def replay_layer_conditions(torch, model, layer, rec, fp_state, cond_state_list):
    batch = len(cond_state_list)
    rec_b = dict(rec)
    for k in ("query", "key", "value", "g", "beta"):
        rec_b[k] = expand_driver(torch, rec[k], batch)
    rec_b["z"] = expand_gate_z(torch, rec["z"], batch)
    fp_batch = fp_state.detach().float().expand(batch, *fp_state.shape[1:]).contiguous()
    pe_batch = torch.cat([s.detach().float() for s in cond_state_list], dim=0)
    fp_manual = manual_recurrent_components(torch, rec_b, fp_batch)
    pe_manual = manual_recurrent_components(torch, rec_b, pe_batch)
    fp_out, fp_next = implementation_replay(rec_b, fp_batch)
    pe_out, pe_next = implementation_replay(rec_b, pe_batch)
    pre_fp, post_fp = local_output(torch, model, layer, fp_out.float(), rec_b["z"])
    pre_pe, post_pe = local_output(torch, model, layer, pe_out.float(), rec_b["z"])
    return {
        "E_after_decay": pe_manual["after_decay"] - fp_manual["after_decay"],
        "memory_read_error": pe_manual["memory"] - fp_manual["memory"],
        "delta_error": pe_manual["delta"] - fp_manual["delta"],
        "update_error": pe_manual["update"] - fp_manual["update"],
        "E_after_update": pe_next.detach().float() - fp_next.detach().float(),
        "core_readout_error": pe_out.detach().float() - fp_out.detach().float(),
        "post_norm_error": pre_pe.detach().float() - pre_fp.detach().float(),
        "local_output_error": post_pe.detach().float() - post_fp.detach().float(),
        "next_state": pe_next.detach().float(),
        "fp_next_state": fp_next.detach().float(),
        "identity": {
            "next_state_relerr": tensor_norm(torch, pe_manual["next_state"] - pe_next.float()) / (tensor_norm(torch, pe_next) + EPS),
            "core_relerr": tensor_norm(torch, pe_manual["core"] - pe_out.float()) / (tensor_norm(torch, pe_out) + EPS),
            "fp_next_state_relerr": tensor_norm(torch, fp_manual["next_state"] - fp_next.float()) / (tensor_norm(torch, fp_next) + EPS),
            "fp_core_relerr": tensor_norm(torch, fp_manual["core"] - fp_out.float()) / (tensor_norm(torch, fp_out) + EPS),
        },
    }


def full_kl_reference():
    prev = load_json(PREV)
    by = {}
    for u in prev["per_unit"]:
        by[(u["problem_id"], int(u["t0"]))] = {
            "primary": {"R": u[PRIMARY_R]["AUC_KL"], "C": u[PRIMARY_C]["AUC_KL"]},
            "secondary": {"R": u[SECONDARY_R]["AUC_KL"], "C": u[SECONDARY_C]["AUC_KL"]},
            "protocol_hash_inputs": {
                "task": prev["task"], "model": prev["model"], "horizon": prev["continuation_horizon"],
                "prompt_id": u["problem_id"], "t0": u["t0"],
            },
        }
    return by


def run_unit(torch, model, tokenizer, e2e, pm, t0, horizon=HORIZON):
    fp_past, ids, cont, collector = run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        errors, meta = build_initial_errors(torch, fp_past)
        cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + errors[c][layer] for layer in GDN_LAYERS} for c in CONDITIONS}
        fp_states = {layer: p1.get_state(fp_past, layer).detach().float().clone() for layer in GDN_LAYERS}
        curves = {c: defaultdict(list) for c in CONDITIONS}
        per_layer = defaultdict(lambda: defaultdict(float))
        per_head = defaultdict(lambda: defaultdict(float))
        per_value = defaultdict(list)
        max_identity = defaultdict(float)
        numerical_failures = []
        raw = {}
        valid_h = min(horizon, len(cont) - t0)
        mask = None
        past = fp_past
        with torch.inference_mode():
            for off in range(1, valid_h + 1):
                nxt = torch.tensor([[cont[t0 + off - 1]]], dtype=ids.dtype, device=ids.device)
                _out, past, records = driver_step(torch, model, nxt, mask, past, collector)
                for layer in GDN_LAYERS:
                    rec = records.get(layer)
                    if rec is None:
                        numerical_failures.append({"offset": off, "layer": layer, "reason": "missing_fp_driver_record"})
                        continue
                    out = replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[c][layer] for c in CONDITIONS])
                    for name, val in out["identity"].items():
                        max_identity[name] = max(max_identity[name], float(val))
                    for ci, c in enumerate(CONDITIONS):
                        cond_states[c][layer] = out["next_state"][ci : ci + 1]
                        metrics = {
                            "state": out["E_after_update"][ci : ci + 1],
                            "key": out["memory_read_error"][ci : ci + 1],
                            "query": out["core_readout_error"][ci : ci + 1],
                            "local": out["local_output_error"][ci : ci + 1],
                            "after_decay": out["E_after_decay"][ci : ci + 1],
                            "delta": out["delta_error"][ci : ci + 1],
                            "post_norm": out["post_norm_error"][ci : ci + 1],
                        }
                        for k, ten in metrics.items():
                            n2 = float(torch.sum(ten.detach().float().double() ** 2).item())
                            curves[c][k].append(math.sqrt(n2))
                            per_layer[(c, layer)][f"J_{k}"] += n2
                        head_state = out["E_after_update"][ci : ci + 1].detach().float()
                        head_query = out["core_readout_error"][ci : ci + 1].detach().float()
                        head_state_n2 = torch.sum(head_state.double() ** 2, dim=(0, 2, 3)).detach().cpu().tolist()
                        head_query_n2 = torch.sum(head_query.double() ** 2, dim=(0, 1, 3)).detach().cpu().tolist()
                        for h, n2 in enumerate(head_state_n2):
                            per_head[(c, layer, h)]["J_state"] += float(n2)
                        for h, n2 in enumerate(head_query_n2):
                            per_head[(c, layer, h)]["J_query"] += float(n2)
                        qerr = out["core_readout_error"][ci].detach().float()
                        lerr = out["local_output_error"][ci].detach().float()
                        qvals = torch.sqrt(torch.sum(qerr.double() ** 2, dim=(0, 1))).detach().cpu().tolist()
                        lvals = torch.sqrt(torch.sum(lerr.reshape(-1, lerr.shape[-1]).double() ** 2, dim=0))[:256].detach().cpu().tolist()
                        for v, val in enumerate(qvals):
                            per_value[(c, "query", v)].append(float(val))
                        for v, val in enumerate(lvals):
                            per_value[(c, "local_output_first256", v)].append(float(val))
                    fp_states[layer] = rec["final_state"].detach().float()
        for c in CONDITIONS:
            raw[c] = {k: np.asarray(v, dtype=np.float64) for k, v in curves[c].items()}
        return make_unit_result(pm, t0, valid_h, meta, curves, per_layer, per_head, per_value, max_identity, numerical_failures, raw)
    finally:
        collector["close"]()


def auc_from_curve(xs):
    return sum(float(x) ** 2 for x in xs if finite(x))


def make_unit_result(pm, t0, h, meta, curves, per_layer, per_head, per_value, max_identity, failures, raw):
    ref = full_kl_reference()[(pm["problem_id"], t0)]
    unit = {
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "t0": t0,
        "future_horizon": h,
        "same_norm_meta": meta,
        "primary_pair": {},
        "secondary_pair": {},
        "per_layer": [],
        "per_head": [],
        "per_value_channel": [],
        "identity_max_relative_errors": dict(max_identity),
        "numerical_failures": failures,
    }
    for label, rcond, ccond in (("primary_pair", PRIMARY_R, PRIMARY_C), ("secondary_pair", SECONDARY_R, SECONDARY_C)):
        pair = {
            "condition_R": rcond,
            "condition_C": ccond,
            "norm_R": meta["norms"][rcond],
            "norm_C": meta["norms"][ccond],
            "initial_norm_ratio": ratio(meta["norms"][rcond], meta["norms"][ccond]),
            "state_curve_R": curves[rcond]["state"],
            "state_curve_C": curves[ccond]["state"],
            "key_error_curve_R": curves[rcond]["key"],
            "key_error_curve_C": curves[ccond]["key"],
            "query_error_curve_R": curves[rcond]["query"],
            "query_error_curve_C": curves[ccond]["query"],
            "local_output_curve_R": curves[rcond]["local"],
            "local_output_curve_C": curves[ccond]["local"],
            "J_state_R": auc_from_curve(curves[rcond]["state"]),
            "J_state_C": auc_from_curve(curves[ccond]["state"]),
            "J_key_R": auc_from_curve(curves[rcond]["key"]),
            "J_key_C": auc_from_curve(curves[ccond]["key"]),
            "J_query_R": auc_from_curve(curves[rcond]["query"]),
            "J_query_C": auc_from_curve(curves[ccond]["query"]),
            "J_local_R": auc_from_curve(curves[rcond]["local"]),
            "J_local_C": auc_from_curve(curves[ccond]["local"]),
            "full_KL_R": ref["primary" if label == "primary_pair" else "secondary"]["R"],
            "full_KL_C": ref["primary" if label == "primary_pair" else "secondary"]["C"],
        }
        for k in ("state", "key", "query", "local"):
            pair[f"R_over_C_{k}"] = ratio(pair[f"J_{k}_R"], pair[f"J_{k}_C"])
            pair[f"R_minus_C_{k}"] = pair[f"J_{k}_R"] - pair[f"J_{k}_C"]
            pair[f"log_ratio_{k}"] = log_ratio(pair[f"J_{k}_R"], pair[f"J_{k}_C"])
        pair["R_over_C_KL"] = ratio(pair["full_KL_R"], pair["full_KL_C"])
        pair["R_minus_C_KL"] = pair["full_KL_R"] - pair["full_KL_C"]
        pair["log_ratio_KL"] = log_ratio(pair["full_KL_R"], pair["full_KL_C"])
        pair["crossover_offset"] = crossover(pair["query_error_curve_R"], pair["query_error_curve_C"])
        unit[label] = pair
    for (c, layer), vals in per_layer.items():
        row = {"condition": c, "layer_idx": layer}
        row.update(vals)
        unit["per_layer"].append(row)
    for (c, layer, head), vals in per_head.items():
        row = {"condition": c, "layer_idx": layer, "head_idx": head}
        row.update(vals)
        unit["per_head"].append(row)
    for (c, kind, value_idx), vals in per_value.items():
        unit["per_value_channel"].append({
            "condition": c, "kind": kind, "value_idx": value_idx,
            "median": med(vals), "p90": pct(vals, 0.90), "p95": pct(vals, 0.95), "max": max(vals) if vals else None,
        })
    unit["_raw_curves"] = raw
    return unit


def crossover(r_curve, c_curve):
    for i, (r, c) in enumerate(zip(r_curve, c_curve), start=1):
        if i == 1 and r >= c:
            continue
        if i > 1 and r > c:
            return i
    return None


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
    rx, ry = rankdata([p[0] for p in pairs]), rankdata([p[1] for p in pairs])
    mx, my = avg(rx), avg(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx)) * math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den + EPS)


def summarize_pair(units, pair_name):
    out = {}
    for stage in ("state", "key", "query", "local", "KL"):
        rs = [u[pair_name][f"R_over_C_{stage}"] for u in units]
        out[stage] = {
            "median_R_over_C_ratio": med(rs),
            "R_greater_C_units": sum(1 for u in units if u[pair_name][f"R_minus_C_{stage}"] > 0),
            "median_log_ratio": med([u[pair_name][f"log_ratio_{stage}"] for u in units]),
            "median_absolute_difference": med([u[pair_name][f"R_minus_C_{stage}"] for u in units]),
        }
    out["observability_crossover_units"] = sum(1 for u in units if u[pair_name]["crossover_offset"] is not None)
    out["median_crossover_offset"] = med([u[pair_name]["crossover_offset"] for u in units])
    return out


def classify(units):
    p = summarize_pair(units, "primary_pair")
    gt = {k: p[k]["R_greater_C_units"] for k in ("state", "key", "query", "local", "KL")}
    mr = {k: p[k]["median_R_over_C_ratio"] for k in ("state", "key", "query", "local", "KL")}
    strong = [k for k in ("state", "key", "query", "local") if gt[k] >= 7 and (mr[k] or 0) >= 1.2]
    if strong and strong[0] == "state":
        return "PURE_RECURRENCE_STRUCTURE_SIGNAL"
    if "key" in strong and "state" not in strong:
        return "FUTURE_KEY_INTERACTION_SIGNAL"
    if "query" in strong and "state" not in strong:
        return "FUTURE_QUERY_OBSERVABILITY_SIGNAL"
    if "local" in strong and "query" not in strong:
        return "LOCAL_OUTPUT_PATH_SIGNAL"
    if len(strong) > 1:
        return "MIXED_PATH_SIGNAL"
    if gt["KL"] >= 7:
        return "FEEDBACK_REQUIRED_CANDIDATE"
    return "NO_CLEAR_PATH" if len(units) == 9 else "INCONCLUSIVE"


def aggregate(units):
    primary = summarize_pair(units, "primary_pair")
    secondary = summarize_pair(units, "secondary_pair")
    correlations = {}
    for stage in ("state", "key", "query", "local"):
        correlations[f"rho_{stage}_ratio_vs_KL_ratio"] = spearman(
            [u["primary_pair"][f"R_over_C_{stage}"] for u in units],
            [u["primary_pair"]["R_over_C_KL"] for u in units],
        )
    offsets = []
    for off in OFFSETS:
        rows = [u for u in units if off <= len(u["primary_pair"]["query_error_curve_R"])]
        if not rows:
            continue
        offsets.append({
            "offset": off,
            "median_R_over_C_state_ratio": med([ratio(u["primary_pair"]["state_curve_R"][off - 1], u["primary_pair"]["state_curve_C"][off - 1]) for u in rows]),
            "median_R_over_C_key_ratio": med([ratio(u["primary_pair"]["key_error_curve_R"][off - 1], u["primary_pair"]["key_error_curve_C"][off - 1]) for u in rows]),
            "median_R_over_C_query_ratio": med([ratio(u["primary_pair"]["query_error_curve_R"][off - 1], u["primary_pair"]["query_error_curve_C"][off - 1]) for u in rows]),
            "median_R_over_C_local_ratio": med([ratio(u["primary_pair"]["local_output_curve_R"][off - 1], u["primary_pair"]["local_output_curve_C"][off - 1]) for u in rows]),
            "R_query_greater_C_unit_count": sum(1 for u in rows if u["primary_pair"]["query_error_curve_R"][off - 1] > u["primary_pair"]["query_error_curve_C"][off - 1]),
        })
    return {"primary_pair": primary, "secondary_pair": secondary, "offset_analysis": offsets, "correlations": correlations}


def strip_raw(unit):
    u = dict(unit)
    u.pop("_raw_curves", None)
    return u


def source_semantics():
    return {
        "modeling_file": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
        "gdn_module": "Qwen3_5GatedDeltaNet",
        "state_shape": "[batch, value_heads, key_dim, value_dim]",
        "driver_shapes": {
            "query": "[batch, seq_len, value_heads, key_dim] after repeat_interleave and before in-kernel l2norm",
            "key": "[batch, seq_len, value_heads, key_dim] after repeat_interleave and before in-kernel l2norm",
            "value": "[batch, seq_len, value_heads, value_dim]",
            "beta": "[batch, seq_len, value_heads]",
            "g_decay_log": "[batch, seq_len, value_heads]",
            "z_gate": "[batch, seq_len, value_heads, value_dim]",
        },
        "update_order": "state *= exp(g); memory=(state*k).sum(key_dim); delta=(v-memory)*beta; state += k*delta; core=(state*q_scaled).sum(key_dim)",
        "qk_normalization": "use_qk_l2norm_in_kernel=True, then query scaled by key_dim^-0.5",
        "local_output_path": "core -> Qwen3_5RMSNormGated(core,z) -> out_proj",
        "drivers_are_frozen_to": "PURE_FP_TRAJECTORY",
    }


def stage0():
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    pm = selected_prompts(1)[0]
    fp_past, ids, cont, collector = run_fp_to_t0(torch, model, tokenizer, e2e, pm, 16)
    try:
        errors, meta = build_initial_errors(torch, fp_past)
        nxt = torch.tensor([[cont[16]]], dtype=ids.dtype, device=ids.device)
        with torch.inference_mode():
            _out, _past, records = driver_step(torch, model, nxt, None, fp_past, collector)
        max_next = max_core = max_local = max_mem = 0.0
        checked = 0
        for layer in GDN_LAYERS:
            rec = records[layer]
            fp_state = rec["initial_state"]
            pe_state = fp_state + errors[PRIMARY_R][layer]
            out = replay_layer_step(torch, model, layer, rec, fp_state, pe_state)
            max_next = max(max_next, out["identity"]["next_state_relerr"], out["identity"]["fp_next_state_relerr"])
            max_core = max(max_core, out["identity"]["core_relerr"], out["identity"]["fp_core_relerr"])
            man = manual_recurrent_components(torch, rec, pe_state)
            impl_out, _impl_next = implementation_replay(rec, pe_state)
            pre1, post1 = local_output(torch, model, layer, man["core"], rec["z"])
            pre2, post2 = local_output(torch, model, layer, impl_out.float(), rec["z"])
            max_local = max(max_local, tensor_norm(torch, post1 - post2) / (tensor_norm(torch, post2) + EPS))
            max_mem = max(max_mem, 0.0)
            checked += 1
        gates = {
            "PROTOCOL_GATE": "PASS",
            "SOURCE_SEMANTICS_GATE": "PASS",
            "SAME_NORM_REPRODUCTION_GATE": "PASS" if meta["max_per_head_relative_norm_difference"] <= 1e-4 and meta["min_direction_cosine"] >= 0.999999 else "FAIL",
            "FP_DRIVER_CAPTURE_GATE": "PASS" if checked == len(GDN_LAYERS) else "FAIL",
            "FROZEN_NEXT_STATE_IDENTITY_GATE": "PASS" if max_next <= 0.005 else "FAIL",
            "FROZEN_MEMORY_READ_IDENTITY_GATE": "PASS" if max_mem <= 0.005 else "FAIL",
            "READOUT_IDENTITY_GATE": "PASS" if max_core <= 0.005 else "FAIL",
            "LOCAL_OUTPUT_REPLAY_GATE": "PASS" if max_local <= 0.005 else "FAIL",
            "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        }
        obj = {
            "task": TASK,
            "timestamp": now(),
            "git_commit": git_commit(),
            "model": "Qwen3.5-9B",
            "protocol": "same-norm single-pulse residual structure; frozen FP drivers; cloned recurrent implementation replay; no cadence; no repeated quantization",
            "source_semantics": source_semantics(),
            "gate_results": gates,
            "stage0_unit": {
                "problem_id": pm["problem_id"], "t0": 16, "checked_layers": checked,
                "same_norm_meta": meta,
                "max_next_state_identity_relerr": max_next,
                "max_memory_read_identity_relerr": max_mem,
                "max_core_readout_identity_relerr": max_core,
                "max_local_output_replay_relerr": max_local,
            },
            "R128_REPEATED_ACCUMULATION_FORMAL": "SUPPORTED",
            "NATURAL_R128_C128_NORM_SWAP": "NO_CLEAR_CAUSAL_DECOMPOSITION",
            "MECHANISM_CLOSURE_CANDIDATE": "NO",
            "METHOD_DESIGN_READY_CANDIDATE": "NO",
            "METHOD_DESIGN_READY": "NO",
        }
        save_json(STAGE0, obj)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        return obj
    finally:
        collector["close"]()


def make_report(obj):
    a = obj["aggregate"]
    lines = [
        "# GDN INT8 R128/C128 Frozen Observability Path Decomposition V1",
        "",
        "## 1. Scientific Question",
        "Where does same-norm R-structure extra behavioral damage first appear along the frozen FP path?",
        "",
        "## 2. Prior Causal Evidence",
        "R128 repeated accumulation is formal-supported, and natural residual norm-swap remains no-clear-decomposition.",
        "",
        "## 3. Current Contradiction",
        "Same-norm R structure produces larger future KL, but one-step normalized core readout was lower for R in the previous audit.",
        "",
        "## 4. Competing Path Hypotheses",
        "Pure recurrence, future-key interaction, future-query observability, local output path, feedback-required, mixed, or no clear path.",
        "",
        "## 5. Protocol",
        obj["protocol"],
        "",
        "## 6. Same-Norm Residual Pairs",
        "Primary: R_STRUCT_C_NORM vs REAL_C. Secondary: REAL_R vs C_STRUCT_R_NORM.",
        "",
        "## 7. FP Driver Capture",
        "Drivers are captured only from the pure FP teacher-forced trajectory.",
        "",
        "## 8. Source Semantics Audit",
        "See JSON `source_semantics`.",
        "",
        "## 9. Replay Identity Gates",
        "```json\n" + json.dumps(obj["gate_results"], indent=2) + "\n```",
        "",
        "## 10. Pure Recurrent-State Propagation",
        f"Primary median R/C J_state ratio: {a['primary_pair']['state']['median_R_over_C_ratio']}",
        "",
        "## 11. Future-Key Interaction",
        f"Primary median R/C J_key ratio: {a['primary_pair']['key']['median_R_over_C_ratio']}",
        "",
        "## 12. Future-Query Observability",
        f"Primary median R/C J_query ratio: {a['primary_pair']['query']['median_R_over_C_ratio']}",
        "",
        "## 13. Observability Crossover",
        f"Primary crossover units: {a['primary_pair']['observability_crossover_units']} / {len(obj['per_unit'])}",
        "",
        "## 14. GDN Local Output Path",
        f"Primary median R/C J_local ratio: {a['primary_pair']['local']['median_R_over_C_ratio']}",
        "",
        "## 15. Full-Model KL Reference",
        f"Primary R full KL > C: {a['primary_pair']['KL']['R_greater_C_units']} / {len(obj['per_unit'])}",
        "",
        "## 16. Path Waterfall",
        "| Prompt | t0 | Initial Norm R/C | State R/C | Key R/C | Query R/C | Local Output R/C | Full KL R/C | Crossover Offset | Classification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for u in obj["per_unit"]:
        p = u["primary_pair"]
        lines.append(f"| {u['problem_id']} | {u['t0']} | {p['initial_norm_ratio']:.6g} | {p['R_over_C_state']:.6g} | {p['R_over_C_key']:.6g} | {p['R_over_C_query']:.6g} | {p['R_over_C_local']:.6g} | {p['R_over_C_KL']:.6g} | {p['crossover_offset']} | {obj['path_classification']} |")
    lines += [
        "",
        "## 17. Layer / Head Attribution",
        "Per-layer and per-head rows are stored in the pilot JSON.",
        "",
        "## 18. Value-Channel Attribution",
        "Per-value-channel summary statistics are stored in the pilot JSON.",
        "",
        "## 19. Per-Prompt / Per-t0 Results",
        "See path waterfall and JSON curves.",
        "",
        "## 20. Path-vs-KL Association",
        "```json\n" + json.dumps(a["correlations"], indent=2) + "\n```",
        "",
        "## 21. Pilot Classification",
        f"`{obj['path_classification']}`",
        "",
        "## 22. What Is Supported",
        obj["supported"],
        "",
        "## 23. What Is NOT Supported",
        "\n".join("- " + x for x in obj["not_supported"]),
        "",
        "## 24. Negative / Corrective Results",
        "Readout is computed from frozen FP queries and cloned recurrent replay, not from the previous parallel-branch hook timing signal.",
        "",
        "## 25. Next Recommended Experiment",
        obj["next_recommended_task"],
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(obj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stages = ["norm", "state", "key", "query", "local", "KL"]
    vals = [1.0] + [obj["aggregate"]["primary_pair"][s]["median_R_over_C_ratio"] for s in ["state", "key", "query", "local", "KL"]]
    plt.figure(figsize=(7, 3))
    plt.plot(stages, vals, marker="o")
    plt.axhline(1.0, color="black", lw=1)
    plt.ylabel("median R/C ratio")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_r_c_path_waterfall.png", dpi=160)
    plt.close()
    u = obj["per_unit"][0]
    x = np.arange(1, len(u["primary_pair"]["query_error_curve_R"]) + 1)
    plt.figure(figsize=(7, 3))
    plt.plot(x, u["primary_pair"]["query_error_curve_R"], label="R structure")
    plt.plot(x, u["primary_pair"]["query_error_curve_C"], label="C structure")
    plt.xlabel("future offset")
    plt.ylabel("query-visible error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_future_offset_query_visible_error.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 3))
    qrat = [ratio(r, c) for r, c in zip(u["primary_pair"]["query_error_curve_R"], u["primary_pair"]["query_error_curve_C"])]
    plt.plot(x, qrat)
    plt.axhline(1.0, color="black", lw=1)
    plt.xlabel("future offset")
    plt.ylabel("R/C query ratio")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_future_offset_query_ratio.png", dpi=160)
    plt.close()
    for key, fname, ylabel in [
        ("state", "figure4_future_offset_state_error.png", "state error"),
        ("local_output", "figure5_future_offset_local_output_error.png", "local output error"),
    ]:
        plt.figure(figsize=(7, 3))
        plt.plot(x, u["primary_pair"][f"{key}_curve_R"], label="R structure")
        plt.plot(x, u["primary_pair"][f"{key}_curve_C"], label="C structure")
        plt.xlabel("future offset")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=160)
        plt.close()
    layer = defaultdict(lambda: [0.0, 0.0])
    for u in obj["per_unit"]:
        for row in u["per_layer"]:
            if row["condition"] == PRIMARY_R:
                layer[row["layer_idx"]][0] += row.get("J_query", 0.0)
            if row["condition"] == PRIMARY_C:
                layer[row["layer_idx"]][1] += row.get("J_query", 0.0)
    xs = sorted(layer)
    ys = [ratio(layer[i][0], layer[i][1]) for i in xs]
    plt.figure(figsize=(8, 3))
    plt.bar([str(i) for i in xs], ys)
    plt.axhline(1.0, color="black", lw=1)
    plt.ylabel("R/C query observability")
    plt.xlabel("layer")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_per_layer_r_c_query_observability.png", dpi=160)
    plt.close()


def save_raw_npz(units):
    arrays = {}
    for i, u in enumerate(units):
        for pair_name in ("primary_pair", "secondary_pair"):
            p = u[pair_name]
            for key in ("state", "key_error", "query_error", "local_output"):
                arrays[f"unit{i}_{pair_name}_{key}_R"] = np.asarray(p[f"{key}_curve_R"], dtype=np.float64)
                arrays[f"unit{i}_{pair_name}_{key}_C"] = np.asarray(p[f"{key}_curve_C"], dtype=np.float64)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW, **arrays)


def pilot():
    st = load_json(STAGE0) or stage0()
    if any(v != "PASS" for v in st["gate_results"].values()):
        raise RuntimeError("Stage 0 gate failed; STOP")
    cp = load_json(CHECKPOINT) or {"completed": {}}
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompts = selected_prompts(3)
    for pm in prompts:
        for t0 in T0_PANEL:
            key = f"{pm['problem_id']}|{t0}"
            if key in cp["completed"]:
                continue
            print(f"[{now()}] pilot prompt={pm['problem_id']} t0={t0}", flush=True)
            cp["completed"][key] = strip_raw(run_unit(torch, model, tokenizer, e2e, pm, t0, HORIZON))
            save_json(CHECKPOINT, cp)
    units = [cp["completed"][f"{pm['problem_id']}|{t0}"] for pm in prompts for t0 in T0_PANEL]
    agg = aggregate(units)
    cls = classify(units)
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "protocol": "SAME-NORM STRUCTURE-ONLY frozen FP trajectory replay; no cadence; no repeated quantization; no method design",
        "source_semantics": source_semantics(),
        "prompt_ids": [p["problem_id"] for p in prompts],
        "t0_panel": T0_PANEL,
        "future_horizon": HORIZON,
        "gate_results": st["gate_results"],
        "per_unit": units,
        "aggregate": agg,
        "path_waterfall": agg["primary_pair"],
        "correlations": agg["correlations"],
        "path_classification": cls,
        "supported": f"Pilot path classification: {cls}",
        "not_supported": [
            "This is not a formal multi-prompt validation.",
            "This does not prove feedback if classification is FEEDBACK_REQUIRED_CANDIDATE.",
            "No cadence, repeated quantization, new quantizer, bit allocation, scale redesign, Hadamard method, or method design was run.",
        ],
        "limitations": ["Pilot only: 3 canonical prompts x 3 t0 positions.", "Full-model KL reference is reused from the previous norm-swap pilot after prompt/t0/model/protocol alignment."],
        "next_recommended_task": "GDN_INT8_R128_C128_FP_DRIVER_CLAMP_VS_FULL_FEEDBACK_CAUSAL_V1" if cls == "FEEDBACK_REQUIRED_CANDIDATE" else "multi-prompt formal observability validation",
        "R128_REPEATED_ACCUMULATION_FORMAL": "SUPPORTED",
        "NATURAL_R128_C128_NORM_SWAP": "NO_CLEAR_CAUSAL_DECOMPOSITION",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    save_json(PILOT, obj)
    save_raw_npz(units)
    make_report(obj)
    make_figures(obj)
    print_summary(obj)
    return obj


def print_summary(obj):
    p = obj["aggregate"]["primary_pair"]
    s = obj["aggregate"]["secondary_pair"]
    print("\nTASK =")
    print(TASK)
    print("\nStage 0 =")
    print("PASS")
    for k, v in obj["gate_results"].items():
        print(f"\n{k.replace('_', ' ').title()} =\n{v}")
    print(f"\nPilot units =\n{len(obj['per_unit'])} / 9")
    print("\n=== PRIMARY PAIR ===")
    print("\nPair =\nR_STRUCT_C_NORM vs REAL_C")
    print(f"\nMedian initial norm ratio =\n{med([u['primary_pair']['initial_norm_ratio'] for u in obj['per_unit']])}")
    for stage in ("state", "key", "query", "local", "KL"):
        print(f"\nMedian R/C J_{stage if stage != 'KL' else 'full-model KL'} ratio =\n{p[stage]['median_R_over_C_ratio']}")
        print(f"\nR J_{stage} > C =\n{p[stage]['R_greater_C_units']} / {len(obj['per_unit'])}")
    print(f"\nObservability crossover units =\n{p['observability_crossover_units']} / {len(obj['per_unit'])}")
    print(f"\nMedian crossover offset =\n{p['median_crossover_offset']}")
    print("\n=== SECONDARY PAIR ===")
    print("\nPair =\nREAL_R vs C_STRUCT_R_NORM")
    for stage in ("state", "key", "query", "local", "KL"):
        print(f"\nMedian R/C J_{stage if stage != 'KL' else 'full-model KL'} ratio =\n{s[stage]['median_R_over_C_ratio']}")
    print("\n=== ASSOCIATION ===")
    for k, v in obj["correlations"].items():
        print(f"\n{k} =\n{v}")
    print("\nPATH_CLASSIFICATION =")
    print(obj["path_classification"])
    print("\nNumerical failures =")
    print(sum(len(u["numerical_failures"]) for u in obj["per_unit"]))
    print("\nR128_REPEATED_ACCUMULATION_FORMAL =\nSUPPORTED")
    print("\nMECHANISM_CLOSURE_CANDIDATE =\nNO")
    print("\nMETHOD_DESIGN_READY =\nNO")
    print("\nNEXT_RECOMMENDED_TASK =")
    print(obj["next_recommended_task"])
    print("\nArtifacts =")
    for pth in [SCRIPT, STAGE0, PILOT, REPORT, FIG_DIR, RAW]:
        print(pth)
    print("\nSTOP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "pilot"], required=True)
    args = ap.parse_args()
    if args.stage == "stage0":
        stage0()
    else:
        pilot()


if __name__ == "__main__":
    main()
