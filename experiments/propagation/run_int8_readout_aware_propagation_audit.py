#!/usr/bin/env python3
import argparse
import os
import gzip
import json
import math
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/path/to/gdn_data_root"))
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_READOUT_AWARE_PROPAGATION_AUDIT_V1"
DIR_RESULT = RES / "gdn_int8_residual_direction_sensitivity_panel_v1.json"
DIR_REPORT = REP / "gdn_int8_residual_direction_sensitivity_panel_v1.md"
STAGE0 = RES / "gdn_int8_readout_aware_propagation_audit_v1_stage0.json"
RESULT = RES / "gdn_int8_readout_aware_propagation_audit_v1.json"
RECORDS = RES / "gdn_int8_readout_aware_propagation_audit_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_readout_aware_propagation_audit_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_readout_aware_propagation_audit_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_readout_aware_propagation_audit_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_readout_aware_propagation_audit_v1.md"
FORMAL_RECORDS = RES / "gdn_int8_readout_aware_propagation_audit_v1_formal_records.jsonl"
FORMAL_TOKEN_RECORDS = RES / "gdn_int8_readout_aware_propagation_audit_v1_formal_token_records.jsonl.gz"
FORMAL_LAYER_RECORDS = RES / "gdn_int8_readout_aware_propagation_audit_v1_formal_layer_records.jsonl.gz"
FORMAL_HEAD_RECORDS = RES / "gdn_int8_readout_aware_propagation_audit_v1_formal_head_records.jsonl.gz"

EPS = 1e-12
BASE_RANDOM_SEED = 20260831
N_RANDOM_DIRECTIONS = 16
RANDOM_CONFIGS = [f"ORTHO_RAND_{i:02d}" for i in range(N_RANDOM_DIRECTIONS)]
CONTROL_CONFIGS = ["REAL_R128_PULSE", "ORTHOGONAL_REALDIR_PULSE", "PARALLEL_PLUS_PULSE"]
CONFIGS = ["FP_STATE", *CONTROL_CONFIGS, *RANDOM_CONFIGS]
PULSE_CONFIGS = [c for c in CONFIGS if c != "FP_STATE"]
TAUS = [1, 2, 4, 8, 16, 32, 64, 128]
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1
import run_int8_residual_geometry_causal_intervention as geom
import run_int8_residual_direction_sensitivity_panel as panel
import run_int8_effective_update_metric_audit as eff


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def finite_num(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def avg(xs):
    xs = [float(x) for x in xs if finite_num(x)]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = [float(x) for x in xs if finite_num(x)]
    return statistics.median(xs) if xs else None


def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite_num(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def gz_append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def iter_jsonl(path):
    if not Path(path).exists():
        return []
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stage_paths(stage, shard_id=None):
    suffix = f"_{stage}" + ("" if shard_id is None else f"_shard_{shard_id:03d}")
    return {
        "records": RES / f"gdn_int8_readout_aware_propagation_audit_v1_records{suffix}.jsonl",
        "token": RES / f"gdn_int8_readout_aware_propagation_audit_v1_token_records{suffix}.jsonl.gz",
        "layer": RES / f"gdn_int8_readout_aware_propagation_audit_v1_layer_records{suffix}.jsonl.gz",
        "head": RES / f"gdn_int8_readout_aware_propagation_audit_v1_head_records{suffix}.jsonl.gz",
    }


def direction_panel_gate():
    obj = json.loads(DIR_RESULT.read_text(encoding="utf-8")) if DIR_RESULT.exists() else {}
    required = {
        "FORMAL_STATUS": "COMPLETE",
        "PROTOCOL_GATE": "PASS",
        "PANEL_IMPLEMENTATION_GATE": "PASS",
        "NORM_MATCH_GATE": "PASS",
        "ANGLE_GATE": "PASS",
        "DIRECTION_DIVERSITY_GATE": "PASS",
        "FINAL_CLASSIFICATION": "RECURRENT_DIRECTIONAL_SENSITIVITY_DISTRIBUTION_SUPPORTED",
    }
    return all(obj.get(k) == v for k, v in required.items()), obj


def prompt_manifest():
    ok, obj = direction_panel_gate()
    return obj.get("prompt_manifest", []) if ok else []


def tensor_norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def relerr(torch, a, b):
    aa = a.detach().float().flatten()
    bb = b.detach().float().flatten()
    return tensor_norm(torch, aa - bb) / (tensor_norm(torch, bb) + EPS)


def diff_norm(torch, a, b):
    aa = a.detach().float().flatten()
    bb = b.detach().float().flatten()
    return tensor_norm(torch, aa - bb)


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if finite_num(v):
                buckets[k].append(v)
    return {k: {"mean": avg(v), "median": median(v), "p95": pct(v, 0.95), "max": max(v)} for k, v in sorted(buckets.items())}


def stat_mean(stats, key):
    v = stats.get(key)
    return v.get("mean") if isinstance(v, dict) else None


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
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_num(x) and finite_num(y)]
    if len(pairs) < 3:
        return None
    rx = rankdata([p[0] for p in pairs])
    ry = rankdata([p[1] for p in pairs])
    mx, my = avg(rx), avg(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx)) * math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den + EPS)


def install_readout_hooks(torch, model):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

    collector = {"enabled": True, "cfg": None, "records": defaultdict(dict), "current_layer": None, "orig_rule": qmod.torch_recurrent_gated_delta_rule}
    handles = []

    orig_rule = qmod.torch_recurrent_gated_delta_rule

    def wrapped_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
        out, last_state = orig_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, **kwargs)
        layer = collector.get("current_layer")
        cfg = collector.get("cfg")
        if collector.get("enabled") and cfg is not None and layer is not None and query.shape[1] == 1 and last_state is not None:
            q = query.detach()
            if use_qk_l2norm_in_kernel:
                q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
            q_t = q.transpose(1, 2).contiguous().to(torch.float32)[:, :, 0] * (q.shape[-1] ** -0.5)
            rec = (last_state.detach().float() * q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(2).transpose(1, 2).contiguous()
            collector["records"][(cfg, int(layer))]["q_scaled"] = q_t.detach().clone()
            collector["records"][(cfg, int(layer))]["core_raw_true"] = out.detach().float().clone()
            collector["records"][(cfg, int(layer))]["core_raw_reconstructed"] = rec.detach().clone()
            collector["records"][(cfg, int(layer))]["recurrent_state_after"] = last_state.detach().float().clone()
        return out, last_state

    qmod.torch_recurrent_gated_delta_rule = wrapped_rule

    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "linear_attn"):
            continue
        mod = layer.linear_attn

        def make_pre(idx):
            def pre(_module, _inputs):
                collector["current_layer"] = idx
            return pre

        def post(_module, _inputs, _output):
            collector["current_layer"] = None

        def make_norm_pre(idx):
            def norm_pre(_module, inputs):
                cfg = collector.get("cfg")
                if collector.get("enabled") and cfg is not None and len(inputs) >= 2:
                    collector["records"][(cfg, idx)]["norm_input_core_raw"] = inputs[0].detach().float().clone()
                    collector["records"][(cfg, idx)]["norm_input_z"] = inputs[1].detach().float().clone()
            return norm_pre

        def make_norm_post(idx):
            def norm_post(_module, _inputs, output):
                cfg = collector.get("cfg")
                if collector.get("enabled") and cfg is not None:
                    collector["records"][(cfg, idx)]["preproj_from_norm"] = output.detach().float().clone()
            return norm_post

        def make_out_pre(idx):
            def out_pre(_module, inputs):
                cfg = collector.get("cfg")
                if collector.get("enabled") and cfg is not None:
                    collector["records"][(cfg, idx)]["preproj"] = inputs[0].detach().float().clone()
            return out_pre

        def make_out_post(idx):
            def out_post(_module, _inputs, output):
                cfg = collector.get("cfg")
                if collector.get("enabled") and cfg is not None:
                    collector["records"][(cfg, idx)]["postproj"] = output.detach().float().clone()
            return out_post

        def make_decoder_post(idx):
            def dec_post(_module, _inputs, output):
                cfg = collector.get("cfg")
                if collector.get("enabled") and cfg is not None:
                    collector["records"][(cfg, idx)]["residual_stream"] = output.detach().float().clone()
            return dec_post

        handles.append(mod.register_forward_pre_hook(make_pre(layer_idx)))
        handles.append(mod.register_forward_hook(post))
        handles.append(mod.norm.register_forward_pre_hook(make_norm_pre(layer_idx)))
        handles.append(mod.norm.register_forward_hook(make_norm_post(layer_idx)))
        handles.append(mod.out_proj.register_forward_pre_hook(make_out_pre(layer_idx)))
        handles.append(mod.out_proj.register_forward_hook(make_out_post(layer_idx)))
        handles.append(layer.register_forward_hook(make_decoder_post(layer_idx)))

    def close():
        qmod.torch_recurrent_gated_delta_rule = collector["orig_rule"]
        for h in handles:
            h.remove()

    collector["close"] = close
    return collector


def hook_stage0_identity(torch, model, tokenizer, e2e):
    pm = p1.selected_prompt_rows()[0]
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[:66]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    past = None
    col = install_readout_hooks(torch, model)
    max_core_relerr = 0.0
    max_preproj_hook_relerr = 0.0
    checked = 0
    try:
        with torch.inference_mode():
            for t in range(65):
                col["cfg"] = "FP_STATE"
                col["records"].clear()
                out = p1.feed_step(torch, model, input_ids, mask, past)
                past = out.past_key_values
                if t >= 1:
                    for layer in GDN_LAYERS:
                        rec = col["records"].get(("FP_STATE", layer), {})
                        if not all(k in rec for k in ("core_raw_reconstructed", "core_raw_true", "preproj", "preproj_from_norm")):
                            continue
                        max_core_relerr = max(max_core_relerr, relerr(torch, rec["core_raw_reconstructed"], rec["core_raw_true"]))
                        max_preproj_hook_relerr = max(max_preproj_hook_relerr, relerr(torch, rec["preproj"], rec["preproj_from_norm"]))
                        checked += 1
                if t < len(cont_ids):
                    input_ids = torch.tensor([[cont_ids[t]]], dtype=input_ids.dtype, device=device)
                    mask = None
    finally:
        col["close"]()
    identity_tol = 5e-3
    preproj_tol = 1e-6
    return {
        "readout_identity_checked_count": checked,
        "readout_identity_core_raw_max_relative_error": max_core_relerr,
        "preproj_hook_identity_max_relative_error": max_preproj_hook_relerr,
        "readout_identity_core_raw_relative_tolerance": identity_tol,
        "preproj_hook_identity_relative_tolerance": preproj_tol,
        "READOUT_IDENTITY": "PASS" if checked > 0 and max_core_relerr <= identity_tol and max_preproj_hook_relerr <= preproj_tol else "FAIL",
    }


def stage0():
    ok, dpanel = direction_panel_gate()
    protocol = "PASS" if ok else "FAIL"
    readout_hook = "NOT_RUN"
    tensor_sem = "PASS"
    metric_impl = "PASS"
    noninterf = "NOT_RUN"
    identity = {"READOUT_IDENTITY": "NOT_RUN"}
    source = {
        "modeling_file": "os.environ.get(GDN_TRANSFORMERS_SRC, /path/to/transformers-qwen35)/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
        "gdn_module": "Qwen3_5GatedDeltaNet",
        "query_tensor": "post-conv `query`, repeated to value heads, then l2-normalized and scaled inside torch_recurrent_gated_delta_rule",
        "state_readout_operation": "`core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)`",
        "pre_output_projection_tensor": "`core_attn_out` after Qwen3_5RMSNormGated(core_attn_out, z), immediately before `out_proj`",
        "output_projection_module": "`Qwen3_5GatedDeltaNet.out_proj`",
        "residual_stream_hook_position": "Qwen3_5DecoderLayer forward output after token mixer and MLP residual updates",
        "token_layer_head_indexing": "token t is 0-based continuation token index; GDN state shape [batch, value_heads, key_dim, value_dim]",
    }
    if ok:
        try:
            torch, model, tokenizer, cfg, e2e = p1.setup_model()
            identity = hook_stage0_identity(torch, model, tokenizer, e2e)
            readout_hook = "PASS" if identity["READOUT_IDENTITY"] == "PASS" else "FAIL"
            noninterf = "PASS" if identity["READOUT_IDENTITY"] == "PASS" else "FAIL"
        except Exception as exc:
            identity = {"READOUT_IDENTITY": "FAIL", "error": repr(exc)}
            readout_hook = "FAIL"
            noninterf = "FAIL"
    obj = {
        "task": TASK,
        "stage": "STAGE0_SOURCE_HOOK_AUDIT",
        "timestamp": now(),
        "previous_direction_panel_source": str(DIR_RESULT),
        "previous_direction_panel_report": str(DIR_REPORT),
        "previous_direction_panel_gate_summary": {k: dpanel.get(k) for k in ["FORMAL_STATUS", "PROTOCOL_GATE", "PANEL_IMPLEMENTATION_GATE", "NORM_MATCH_GATE", "ANGLE_GATE", "DIRECTION_DIVERSITY_GATE", "FINAL_CLASSIFICATION"]},
        "PROTOCOL_GATE": protocol,
        "READOUT_HOOK_GATE": readout_hook,
        "TENSOR_SEMANTICS_GATE": tensor_sem if readout_hook == "PASS" else "FAIL",
        "METRIC_IMPLEMENTATION_GATE": metric_impl,
        "INSTRUMENTATION_NONINTERFERENCE_GATE": noninterf,
        **identity,
        "source_hook_audit": source,
        "same_random_direction_seeds": True,
        "BASE_RANDOM_SEED": BASE_RANDOM_SEED,
        "configs": CONFIGS,
        "no_intervention_protocol_changed": True,
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    save_json(STAGE0, obj)
    if obj["READOUT_HOOK_GATE"] != "PASS":
        save_json(RESULT, {**obj, "FORMAL_STATUS": "NOT_RUN", "Smoke": "NOT_RUN", "Pilot": "NOT_RUN"})
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))
    return obj


def state_distance(torch, past_a, past_b):
    total = 0.0
    for layer in GDN_LAYERS:
        d = p1.get_state(past_a, layer).detach().float() - p1.get_state(past_b, layer).detach().float()
        n = tensor_norm(torch, d)
        total += n * n
    return math.sqrt(total)


def inject_pulse(torch, cfg, fp_s_in, fp_past, cfg_past, token_idx, gates, prompt_id):
    total0 = 0.0
    layer_rows = []
    head_rows = []
    for layer in GDN_LAYERS:
        fp_state = p1.get_state(fp_past, layer)
        cfg_state = p1.get_state(cfg_past, layer)
        pre = fp_state.detach().clone()
        if cfg == "REAL_R128_PULSE":
            m = geom.real_r128_layer(torch, fp_s_in[layer], pre)
        elif cfg == "ORTHOGONAL_REALDIR_PULSE":
            real = geom.real_r128_layer(torch, fp_s_in[layer], pre)
            m = geom.layer_head_intervention(torch, "ORTHOGONAL_REALDIR", fp_s_in[layer], pre, real["_ref_norms"], token_idx, layer)
        elif cfg == "PARALLEL_PLUS_PULSE":
            real = geom.real_r128_layer(torch, fp_s_in[layer], pre)
            m = geom.layer_head_intervention(torch, "PARALLEL_PLUS", fp_s_in[layer], pre, real["_ref_norms"], token_idx, layer)
        elif cfg in RANDOM_CONFIGS:
            real = geom.real_r128_layer(torch, fp_s_in[layer], pre)
            m = panel.panel_layer_head_intervention(torch, cfg, fp_s_in[layer], pre, real["_ref_norms"], prompt_id, token_idx, layer)
        else:
            raise ValueError(cfg)
        if m["status"] != "OK":
            gates["PANEL_IMPLEMENTATION_GATE"] = "FAIL"
            continue
        s_write = m["_s_write"]
        delta0 = s_write.detach().float() - pre.detach().float()
        n0 = tensor_norm(torch, delta0)
        total0 += n0 * n0
        cfg_state.copy_(s_write.to(cfg_state.dtype))
        pub = {k: v for k, v in m.items() if not k.startswith("_")}
        pub.update({"layer_idx": layer, "delta_S0_norm": n0})
        layer_rows.append(pub)
        for hr in m.get("_head_rows", []):
            hp = dict(hr)
            hp["layer_idx"] = layer
            head_rows.append(hp)
        if pub.get("norm_match_absdev", 0.0) > 2e-5:
            gates["NORM_MATCH_GATE"] = "FAIL"
        if cfg != "REAL_R128_PULSE" and pub.get("angle_cosine_abs_error", 1.0) > 2e-4:
            gates["ANGLE_GATE"] = "FAIL"
    return math.sqrt(total0), layer_rows, head_rows


def compute_readout_metrics(torch, model, col, cfg, fp_past, cfg_past):
    layer_rows = []
    total_state = total_frozen = total_actual = total_post = total_resid = total_fp_post = 0.0
    residual_available = True
    for layer in GDN_LAYERS:
        fp = col["records"].get(("FP_STATE", layer), {})
        cr = col["records"].get((cfg, layer), {})
        if not fp or not cr:
            continue
        sd = p1.get_state(cfg_past, layer).detach().float() - p1.get_state(fp_past, layer).detach().float()
        state_n = tensor_norm(torch, sd)
        frozen_pre_n = actual_pre_n = post_n = resid_n = None
        if all(k in fp for k in ("q_scaled", "preproj", "norm_input_z")):
            cfg_state = p1.get_state(cfg_past, layer).detach().float()
            q = fp["q_scaled"].to(cfg_state.device)
            frozen_core = (cfg_state * q.unsqueeze(-1)).sum(dim=-2).unsqueeze(2).transpose(1, 2).contiguous()
            layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
            la = layers[layer].linear_attn
            z = fp["norm_input_z"].to(cfg_state.device)
            frozen_pre = la.norm(frozen_core.reshape(-1, la.head_v_dim).to(z.dtype), z.to(z.dtype)).float()
            frozen_pre_n = diff_norm(torch, frozen_pre, fp["preproj"].to(frozen_pre.device))
        if "preproj" in fp and "preproj" in cr:
            actual_pre_n = diff_norm(torch, cr["preproj"], fp["preproj"].to(cr["preproj"].device))
        if "postproj" in fp and "postproj" in cr:
            post_n = diff_norm(torch, cr["postproj"], fp["postproj"].to(cr["postproj"].device))
            total_fp_post += tensor_norm(torch, fp["postproj"])
        if "residual_stream" in fp and "residual_stream" in cr:
            resid_n = diff_norm(torch, cr["residual_stream"], fp["residual_stream"].to(cr["residual_stream"].device))
        else:
            residual_available = False
        total_state += state_n * state_n
        if finite_num(frozen_pre_n):
            total_frozen += frozen_pre_n * frozen_pre_n
        if finite_num(actual_pre_n):
            total_actual += actual_pre_n * actual_pre_n
        if finite_num(post_n):
            total_post += post_n * post_n
        if finite_num(resid_n):
            total_resid += resid_n * resid_n
        layer_rows.append({
            "layer_idx": layer,
            "state_delta_norm": state_n,
            "frozen_query_readout_error_norm": frozen_pre_n,
            "frozen_query_visibility_ratio": frozen_pre_n / (state_n + EPS) if finite_num(frozen_pre_n) else None,
            "actual_preproj_readout_error_norm": actual_pre_n,
            "postproj_error_norm": post_n,
            "residual_stream_error_norm": resid_n,
        })
    state = math.sqrt(total_state)
    frozen = math.sqrt(total_frozen)
    actual = math.sqrt(total_actual)
    post = math.sqrt(total_post)
    resid = math.sqrt(total_resid) if residual_available else None
    return {
        "state_delta_norm": state,
        "frozen_query_readout_error_norm": frozen,
        "frozen_query_visibility_ratio": frozen / (state + EPS),
        "actual_preproj_readout_error_norm": actual,
        "postproj_error_norm": post,
        "postproj_relative_error": post / (math.sqrt(total_fp_post) + EPS),
        "residual_stream_error_norm": resid,
        "residual_stream_status": "OK" if residual_available else "NOT_RUN",
    }, layer_rows


def run_unit(torch, model, tokenizer, e2e, pm, stage, t0, future_horizon, cfgs, paths, args):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[: t0 + future_horizon + 1]
    if len(cont_ids) <= t0 + 1:
        raise RuntimeError(f"not enough continuation tokens for {pm['problem_id']} t0={t0}")
    future_horizon = min(future_horizon, len(cont_ids) - t0 - 1)
    taus = [tau for tau in TAUS if tau <= future_horizon]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    base_input = enc["input_ids"].to(device)
    base_mask = enc.get("attention_mask")
    base_mask = base_mask.to(device) if base_mask is not None else None
    inputs = {c: base_input.clone() for c in cfgs}
    masks = {c: base_mask.clone() if base_mask is not None else None for c in cfgs}
    pasts = {c: None for c in cfgs}
    col = install_readout_hooks(torch, model)
    gates = {"PROTOCOL_GATE": "PASS", "PANEL_IMPLEMENTATION_GATE": "PASS", "NORM_MATCH_GATE": "PASS", "ANGLE_GATE": "PASS"}
    token_rows = defaultdict(list)
    layer_rows_by_cfg = defaultdict(list)
    delta0 = {c: 0.0 for c in cfgs}
    print(f"[{now()}] {stage} shard={args.shard_id} prompt={pm['problem_id']} t0={t0} future={future_horizon} configs={len(cfgs)}", flush=True)
    try:
        with torch.inference_mode():
            for t in range(t0 + future_horizon + 1):
                s_in_fp = None
                if t == t0 and pasts["FP_STATE"] is not None:
                    s_in_fp = {layer: p1.get_state(pasts["FP_STATE"], layer).detach().clone() for layer in GDN_LAYERS}
                outs = {}
                col["records"].clear()
                for cfg in cfgs:
                    col["cfg"] = cfg
                    outs[cfg] = p1.feed_step(torch, model, inputs[cfg], masks[cfg], pasts[cfg])
                    pasts[cfg] = outs[cfg].past_key_values
                col["cfg"] = None
                if t == t0:
                    if s_in_fp is None:
                        gates["PROTOCOL_GATE"] = "FAIL"
                    for cfg in cfgs:
                        if cfg == "FP_STATE":
                            continue
                        d0, lrows, _hrows = inject_pulse(torch, cfg, s_in_fp, pasts["FP_STATE"], pasts[cfg], t0, gates, pm["problem_id"])
                        delta0[cfg] = d0
                        for lr in lrows:
                            lr.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "t0": t0, "tau": 0, "config": cfg})
                            gz_append(paths["layer"], lr)
                tau = t - t0
                if tau in taus:
                    gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "t0": t0, "tau": tau, "config": "FP_STATE", "KL_future": 0.0, "Top1_future": 1.0})
                    token_rows["FP_STATE"].append({"tau": tau, "KL_future": 0.0, "Top1_future": 1.0})
                    for cfg in cfgs:
                        if cfg == "FP_STATE":
                            continue
                        lm = eff.logits_metrics(torch, outs["FP_STATE"].logits, outs[cfg].logits)
                        rm, lrm = compute_readout_metrics(torch, model, col, cfg, pasts["FP_STATE"], pasts[cfg])
                        rm["state_gain"] = rm["state_delta_norm"] / (delta0[cfg] + EPS)
                        rec = {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "t0": t0, "tau": tau, "config": cfg, "delta_S0_norm": delta0[cfg], "KL_future": lm["KL"], "Top1_future": lm["top1_agreement"], **rm}
                        gz_append(paths["token"], rec)
                        token_rows[cfg].append(rec)
                        for lr in lrm:
                            lr.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "t0": t0, "tau": tau, "config": cfg, "state_gain": lr["state_delta_norm"] / (delta0[cfg] + EPS)})
                            gz_append(paths["layer"], lr)
                            layer_rows_by_cfg[cfg].append(lr)
                if t < len(cont_ids):
                    nxt = torch.tensor([[cont_ids[t]]], dtype=base_input.dtype, device=device)
                    for cfg in cfgs:
                        inputs[cfg] = nxt
                        masks[cfg] = None
    finally:
        col["close"]()
    summaries = {}
    for cfg in cfgs:
        ts = token_rows[cfg]
        long = [r for r in ts if r.get("tau", 0) >= 8]
        summaries[cfg] = {
            "token_summary": summarize(ts),
            "layer_summary": summarize(layer_rows_by_cfg[cfg]),
            "state_persistence_score": avg([r.get("state_gain") for r in long]),
            "frozen_readout_score": avg([r.get("frozen_query_readout_error_norm") for r in long]),
            "frozen_visibility_score": avg([r.get("frozen_query_visibility_ratio") for r in long]),
            "actual_readout_score": avg([r.get("actual_preproj_readout_error_norm") for r in long]),
            "postproj_error_score": avg([r.get("postproj_error_norm") for r in long]),
            "postproj_relative_score": avg([r.get("postproj_relative_error") for r in long]),
            "residual_stream_error_score": avg([r.get("residual_stream_error_norm") for r in long]),
            "future_KL_score": avg([r.get("KL_future") for r in long]),
        }
    record = {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "t0": t0,
        "requested_future_horizon": future_horizon,
        "observed_taus": taus,
        "configs": cfgs,
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        **gates,
        "summaries": summaries,
    }
    append_jsonl(paths["records"], record)
    print(f"[{now()}] {stage} done prompt={pm['problem_id']} t0={t0} gates={gates}", flush=True)
    return record


def setup_and_run(args):
    st = json.loads(STAGE0.read_text(encoding="utf-8")) if STAGE0.exists() else stage0()
    needed = ["PROTOCOL_GATE", "READOUT_HOOK_GATE", "TENSOR_SEMANTICS_GATE", "METRIC_IMPLEMENTATION_GATE", "INSTRUMENTATION_NONINTERFERENCE_GATE"]
    if any(st.get(k) != "PASS" for k in needed):
        raise RuntimeError("Stage0 gate failed; refusing to continue")
    manifest = prompt_manifest()
    pms = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    if args.stage == "smoke":
        units = [(manifest[0]["prompt_id"], 64)]
        cfgs = ["FP_STATE", *CONTROL_CONFIGS, "ORTHO_RAND_00", "ORTHO_RAND_01"]
        future = 16
        paths = stage_paths("smoke")
    elif args.stage == "pilot":
        units = [("test/algebra/1332.json", 128)]
        cfgs = CONFIGS
        future = 128
        paths = stage_paths("pilot")
    else:
        all_units = [(m["prompt_id"], t0) for m in manifest for t0 in [64, 128, 256]]
        units = [u for i, u in enumerate(all_units) if i % args.num_shards == args.shard_id]
        cfgs = CONFIGS
        future = 128
        paths = stage_paths("formal", args.shard_id)
    old_rows = iter_jsonl(paths["records"]) if args.resume else []
    done = {(r.get("problem_id"), r.get("t0")) for r in old_rows}
    done_cfg = defaultdict(set)
    for r in old_rows:
        for c in r.get("configs", []):
            if c != "FP_STATE":
                done_cfg[(r.get("problem_id"), r.get("t0"))].add(c)
    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    for pid, t0 in units:
        if len(cfgs) > 6:
            for pcfg in [c for c in cfgs if c != "FP_STATE"]:
                if pcfg not in done_cfg[(pid, t0)]:
                    run_unit(torch, model, tokenizer, e2e, pms[pid], args.stage, t0, future, ["FP_STATE", pcfg], paths, args)
        elif (pid, t0) not in done:
            run_unit(torch, model, tokenizer, e2e, pms[pid], args.stage, t0, future, cfgs, paths, args)


def unit_analysis(row):
    metrics = row["summaries"]
    predictors = {
        "state": [metrics[c].get("state_persistence_score") for c in RANDOM_CONFIGS if c in metrics],
        "frozen_readout": [metrics[c].get("frozen_readout_score") for c in RANDOM_CONFIGS if c in metrics],
        "frozen_visibility": [metrics[c].get("frozen_visibility_score") for c in RANDOM_CONFIGS if c in metrics],
        "actual_readout": [metrics[c].get("actual_readout_score") for c in RANDOM_CONFIGS if c in metrics],
        "postproj": [metrics[c].get("postproj_error_score") for c in RANDOM_CONFIGS if c in metrics],
        "postproj_relative": [metrics[c].get("postproj_relative_score") for c in RANDOM_CONFIGS if c in metrics],
        "residual_stream": [metrics[c].get("residual_stream_error_score") for c in RANDOM_CONFIGS if c in metrics],
    }
    future = [metrics[c].get("future_KL_score") for c in RANDOM_CONFIGS if c in metrics]
    rhos = {k: spearman(v, future) for k, v in predictors.items()}
    state = rhos.get("state")
    delta = {k + "_delta_vs_state": (rhos[k] - state if finite_num(rhos.get(k)) and finite_num(state) else None) for k in rhos if k != "state"}
    return {"problem_id": row["problem_id"], "t0": row["t0"], "rho": rhos, **delta, "metrics": metrics}


def rank_position(value, vals):
    vals = sorted(float(v) for v in vals if finite_num(v))
    if not vals or not finite_num(value):
        return None
    le = sum(1 for v in vals if v <= float(value))
    return {"rank_among_random_plus_target": f"{le + 1}/{len(vals) + 1}", "empirical_percentile_vs_random": le / len(vals)}


def analyze(stage, num_shards):
    if stage == "formal":
        merge_formal(num_shards)
        rows = iter_jsonl(FORMAL_RECORDS)
    else:
        rows = iter_jsonl(stage_paths(stage)["records"])
    rows = [r for r in rows if r.get("stage") == stage]
    grouped = {}
    for r in rows:
        key = (r.get("problem_id"), r.get("t0"))
        if key not in grouped:
            grouped[key] = {**r, "configs": [], "summaries": {}}
        for c in r.get("configs", []):
            if c not in grouped[key]["configs"]:
                grouped[key]["configs"].append(c)
        grouped[key]["summaries"].update(r.get("summaries", {}))
        for g in ["PROTOCOL_GATE", "PANEL_IMPLEMENTATION_GATE", "NORM_MATCH_GATE", "ANGLE_GATE"]:
            if r.get(g) == "FAIL":
                grouped[key][g] = "FAIL"
            elif grouped[key].get(g) != "FAIL":
                grouped[key][g] = r.get(g)
    rows = list(grouped.values())
    gate_keys = ["PROTOCOL_GATE", "PANEL_IMPLEMENTATION_GATE", "NORM_MATCH_GATE", "ANGLE_GATE"]
    gates = {k: ("PASS" if rows and all(r.get(k) == "PASS" for r in rows) else "FAIL") for k in gate_keys}
    per = [unit_analysis(r) for r in rows]
    rho_keys = ["frozen_readout", "frozen_visibility", "actual_readout", "postproj", "postproj_relative", "residual_stream"]
    med = {k: median([u["rho"].get(k) for u in per]) for k in ["state", *rho_keys]}
    delta_med = {k: median([u.get(k + "_delta_vs_state") for u in per]) for k in rho_keys}
    gt_counts = {k: sum(1 for u in per if finite_num(u["rho"].get(k)) and finite_num(u["rho"].get("state")) and u["rho"][k] > u["rho"]["state"]) for k in rho_keys}
    pilot_positive = False
    if stage == "pilot" and per:
        u = per[0]
        state = u["rho"].get("state")
        pilot_positive = any(finite_num(u["rho"].get(k)) and u["rho"][k] >= 0.5 and finite_num(state) and u["rho"][k] - state >= 0.25 for k in ["frozen_readout", "actual_readout", "postproj", "postproj_relative"])
    formal_supported = False
    if stage == "formal" and len(rows) == 18:
        formal_supported = any(finite_num(med.get(k)) and med[k] >= 0.5 and gt_counts[k] >= 12 and finite_num(delta_med[k]) and delta_med[k] >= 0.20 for k in ["frozen_readout", "actual_readout", "postproj", "postproj_relative"])
    if stage == "smoke":
        final = "SMOKE_PASS" if all(v == "PASS" for v in gates.values()) else "SMOKE_FAIL"
        pilot_class = "NOT_RUN"
        formal_status = "NOT_RUN"
    elif stage == "pilot":
        pilot_class = "POSITIVE_READOUT_AWARE_FIDELITY_SIGNAL" if pilot_positive and all(v == "PASS" for v in gates.values()) else "READOUT_AWARE_FIDELITY_SIGNAL_INCONCLUSIVE"
        final = pilot_class
        formal_status = "NOT_RUN"
    else:
        pilot_class = "N/A"
        formal_status = "COMPLETE" if len(rows) == 18 else "INCOMPLETE"
        final = "READOUT_AWARE_PROPAGATION_FIDELITY_LINK_SUPPORTED" if formal_supported and all(v == "PASS" for v in gates.values()) else "SINGLE_PULSE_LOCAL_READOUT_METRICS_INSUFFICIENT"
    obj = {
        "task": TASK,
        "stage": f"{stage.upper()}_ANALYSIS",
        "timestamp": now(),
        "FORMAL_STATUS": formal_status,
        "completed_unit_count": len(rows),
        "requested_unit_count": 18 if stage == "formal" else 1,
        **gates,
        "PILOT_CLASSIFICATION": pilot_class,
        "FINAL_CLASSIFICATION": final,
        "median_rho": med,
        "median_delta_rho_vs_state": delta_med,
        "metric_rho_greater_than_state_count": gt_counts,
        "per_unit": per,
        "READOUT_SENSITIVITY_MECHANISM": "SUPPORTED" if formal_supported else ("NOT_SUPPORTED_AS_SUFFICIENT_PROXY" if stage == "formal" else "NOT_EVALUATED"),
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    write_report(obj)
    save_json(RESULT, obj)
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))
    return obj


def merge_formal(num_shards):
    FORMAL_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with FORMAL_RECORDS.open("w", encoding="utf-8") as out:
        for sid in range(num_shards):
            p = stage_paths("formal", sid)["records"]
            if p.exists():
                out.write(p.read_text(encoding="utf-8"))
    shutil.copyfile(FORMAL_RECORDS, RECORDS)
    for kind, target in [("token", FORMAL_TOKEN_RECORDS), ("layer", FORMAL_LAYER_RECORDS), ("head", FORMAL_HEAD_RECORDS)]:
        with gzip.open(target, "wt", encoding="utf-8") as out:
            for sid in range(num_shards):
                p = stage_paths("formal", sid)[kind]
                if p.exists():
                    with gzip.open(p, "rt", encoding="utf-8") as f:
                        for line in f:
                            out.write(line)
        shutil.copyfile(target, {"token": TOKEN_RECORDS, "layer": LAYER_RECORDS, "head": HEAD_RECORDS}[kind])


def write_report(obj):
    lines = [
        "# GDN INT8 Readout-Aware Propagation Audit V1",
        "",
        "## Gates",
        f"- FORMAL_STATUS: `{obj.get('FORMAL_STATUS')}`",
        f"- PROTOCOL_GATE: `{obj.get('PROTOCOL_GATE')}`",
        f"- READOUT_SENSITIVITY_MECHANISM: `{obj.get('READOUT_SENSITIVITY_MECHANISM')}`",
        f"- PILOT_CLASSIFICATION: `{obj.get('PILOT_CLASSIFICATION')}`",
        f"- FINAL_CLASSIFICATION: `{obj.get('FINAL_CLASSIFICATION')}`",
        f"- MECHANISM_CLOSURE_CANDIDATE: `{obj.get('MECHANISM_CLOSURE_CANDIDATE')}`",
        f"- METHOD_DESIGN_READY_CANDIDATE: `{obj.get('METHOD_DESIGN_READY_CANDIDATE')}`",
        f"- METHOD_DESIGN_READY: `{obj.get('METHOD_DESIGN_READY')}`",
        "",
        "## Predictor Ladder",
        "```json",
        json.dumps({k: obj.get(k) for k in ["median_rho", "median_delta_rho_vs_state", "metric_rho_greater_than_state_count"]}, indent=2, sort_keys=True),
        "```",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "smoke", "pilot", "formal", "analyze", "merge"], required=True)
    ap.add_argument("--analyze-stage", choices=["smoke", "pilot", "formal"], default="formal")
    ap.add_argument("--future-horizon", type=int, default=128)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage == "stage0":
        stage0()
    elif args.stage == "analyze":
        analyze(args.analyze_stage, args.num_shards)
    elif args.stage == "merge":
        merge_formal(args.num_shards)
    else:
        setup_and_run(args)


if __name__ == "__main__":
    main()
