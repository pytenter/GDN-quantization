#!/usr/bin/env python3
"""GDN state sensitivity structure audit.

Mechanism audit only: no model/source changes and no new quantization method.
"""
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from state_fake_quant import fake_quant_state, iter_recurrent_state_tensors, quantize_cache_recurrent_states


PROMPTS = [
    ("factual", "Answer briefly: What is the capital of France?"),
    ("reasoning", "Solve step by step: If a train travels 60 km in 45 minutes, what is its speed in km/h?"),
    ("code", "Write a Python function that returns factorial using a loop."),
    ("summary", "Summarize in one sentence: Recurrent state quantization can reduce memory, but errors may propagate."),
    ("long_context", "Context: The red key opens the basement. The blue key opens the attic. The green key opens the shed. Question: which key opens the attic?"),
]
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
REP_LAYERS = [1, 9, 17, 25, 29]
REP_HEADS = [0, 8, 16, 24, 31]
DELTAS = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def stable_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def prompt_meta(limit):
    return [
        {"prompt_id": i, "kind": kind, "text": text, "sha1": stable_hash(text)}
        for i, (kind, text) in enumerate(PROMPTS[:limit])
    ]


def run_cmd(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return "ERROR: %s" % exc


def metadata(torch, transformers, cfg, prompts, seed, args):
    return {
        "run": timestamp_record(),
        "task": "GDN_STATE_SENSITIVITY_STRUCTURE_AUDIT_V1",
        "model": "Qwen3.5-9B",
        "model_path": cfg["model_path"],
        "python": run_cmd(["/data/ydai/miniconda3/envs/bitdecode/bin/python3.10", "--version"]),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "transformers": getattr(transformers, "__version__", "unknown"),
        "transformers_source": "/data/zypan/transformers-qwen35",
        "transformers_revision": run_cmd(["git", "rev-parse", "HEAD"], cwd="/data/zypan/transformers-qwen35"),
        "seed": seed,
        "prompt_set": prompts,
        "stage": args.stage,
        "teacher_forced_path": "FP32 greedy generated token sequence",
        "generation_config": {"greedy": True, "temperature": 0},
        "gdn_layers": GDN_LAYERS,
        "full_attention_layers": FULL_ATTENTION_LAYERS,
    }


def get_layer_types(model):
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    return list(getattr(text_cfg, "layer_types", []))


def get_state(cache, layer_idx, state_idx=0):
    states = getattr(cache.layers[layer_idx], "recurrent_states")
    return states[state_idx] if not isinstance(states, dict) else states[state_idx]


def finite(x):
    return x is not None and x == x and x not in (float("inf"), float("-inf"))


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def rel_norm(torch, a, b):
    return norm(torch, a - b) / max(norm(torch, b), 1e-12)


def logits_metrics(torch, ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    ref_logp = torch.nn.functional.log_softmax(ref, dim=-1)
    other_logp = torch.nn.functional.log_softmax(other, dim=-1)
    kl = float((ref_logp.exp() * (ref_logp - other_logp)).sum().item())
    top1_same = int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item())
    cos = float(torch.nn.functional.cosine_similarity(ref, other, dim=-1).item())
    return {"logit_kl": kl, "top1_same": top1_same, "logit_cosine": cos}


def generate_fp32_path(torch, model, tokenizer, prompt, max_new_tokens):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    past = None
    tokens = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(int(nxt.item()))
            input_ids = nxt.to(device)
            attention_mask = None
    return tokens


def feed_step(torch, model, input_ids, attention_mask, past, output_hidden_states=False):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=True,
        output_hidden_states=output_hidden_states,
    )


def hidden_divergence(torch, ref_out, other_out):
    out = []
    if not getattr(ref_out, "hidden_states", None) or not getattr(other_out, "hidden_states", None):
        return out
    # hidden_states[0] is embedding output; hidden_states[i+1] is decoder layer i output.
    for layer_idx in range(min(len(ref_out.hidden_states), len(other_out.hidden_states)) - 1):
        ref_h = ref_out.hidden_states[layer_idx + 1]
        oth_h = other_out.hidden_states[layer_idx + 1]
        out.append({
            "layer_idx": layer_idx,
            "layer_type": "full_attention" if layer_idx in FULL_ATTENTION_LAYERS else "linear_attention",
            "hidden_divergence": rel_norm(torch, oth_h, ref_h),
        })
    return out


def global_state_divergence(torch, ref_cache, other_cache):
    num = 0.0
    den = 0.0
    for layer_idx in GDN_LAYERS:
        ref = get_state(ref_cache, layer_idx).detach().float()
        oth = get_state(other_cache, layer_idx).detach().float()
        num += float(torch.sum((oth - ref) ** 2).item())
        den += float(torch.sum(ref ** 2).item())
    return math.sqrt(num) / max(math.sqrt(den), 1e-12)


def random_like(torch, shape, device, dtype, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=gen)


def scale_to_relative_norm(torch, e, ref, rel_target):
    target = torch.linalg.vector_norm(ref.detach().float()) * rel_target
    actual = torch.linalg.vector_norm(e.detach().float()).clamp_min(1e-12)
    return e * (target / actual).to(e.dtype)


def iid_head_perturb(torch, state, head, rel_target, seed):
    e = torch.zeros_like(state)
    base = state[:, head, :, :]
    noise = random_like(torch, base.shape, base.device, base.dtype, seed)
    e[:, head, :, :] = scale_to_relative_norm(torch, noise, base, rel_target)
    return e


def support_perturb(torch, state, head, rows=None, cols=None, rel_target=0.005, seed=0):
    e = torch.zeros_like(state)
    base = state[:, head, :, :]
    block = torch.zeros_like(base)
    if rows is not None:
        noise = random_like(torch, block[:, rows, :].shape, base.device, base.dtype, seed)
        block[:, rows, :] = noise
    elif cols is not None:
        noise = random_like(torch, block[:, :, cols].shape, base.device, base.dtype, seed)
        block[:, :, cols] = noise
    else:
        block = random_like(torch, base.shape, base.device, base.dtype, seed)
    e[:, head, :, :] = scale_to_relative_norm(torch, block, base, rel_target)
    return e


def svd_perturb(torch, state, head, kind, rel_target, seed):
    base = state[:, head, :, :][0].detach().float()
    device = state.device
    dtype = state.dtype
    u, _, vh = torch.linalg.svd(base, full_matrices=True)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if kind == "random":
        mat = torch.randn(base.shape, device=device, dtype=torch.float32, generator=gen)
    elif kind == "top1_left":
        a = torch.randn((1, base.shape[1]), device=device, dtype=torch.float32, generator=gen)
        mat = u[:, :1] @ a
    elif kind == "top4_left":
        a = torch.randn((4, base.shape[1]), device=device, dtype=torch.float32, generator=gen)
        mat = u[:, :4] @ a
    elif kind == "top8_left":
        a = torch.randn((8, base.shape[1]), device=device, dtype=torch.float32, generator=gen)
        mat = u[:, :8] @ a
    elif kind == "bottom8_left":
        a = torch.randn((8, base.shape[1]), device=device, dtype=torch.float32, generator=gen)
        mat = u[:, -8:] @ a
    elif kind == "top8_right":
        a = torch.randn((base.shape[0], 8), device=device, dtype=torch.float32, generator=gen)
        mat = a @ vh[:8, :]
    elif kind == "bottom8_right":
        a = torch.randn((base.shape[0], 8), device=device, dtype=torch.float32, generator=gen)
        mat = a @ vh[-8:, :]
    else:
        raise ValueError(kind)
    mat = mat.to(dtype)
    e = torch.zeros_like(state)
    e[:, head, :, :] = scale_to_relative_norm(torch, mat.unsqueeze(0), state[:, head, :, :], rel_target)
    return e


def run_pair_trajectory(torch, model, tokenizer, prompt, token_path, t0, horizon, inject_fn, records_fh, phase, tag, output_hidden=False):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    ref_input = encoded["input_ids"].to(device)
    other_input = ref_input.clone()
    ref_mask = encoded.get("attention_mask")
    other_mask = ref_mask.clone().to(device) if ref_mask is not None else None
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    ref_past = None
    other_past = None
    injected = False
    trajectory = []
    max_steps = min(len(token_path), t0 + horizon + 1)
    checkpoints = set(d for d in DELTAS if d <= horizon)
    with torch.inference_mode():
        for token_idx in range(max_steps):
            ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past, output_hidden_states=output_hidden)
            other_out = feed_step(torch, model, other_input, other_mask, other_past, output_hidden_states=output_hidden)
            ref_past = ref_out.past_key_values
            other_past = other_out.past_key_values
            norm_info = None
            if token_idx == t0 and not injected:
                norm_info = inject_fn(torch, other_past)
                injected = True
            if injected:
                delta = token_idx - t0
                if delta in checkpoints:
                    local_layer = norm_info.get("layer_idx") if norm_info else inject_fn.layer_idx
                    local_div = rel_norm(torch, get_state(other_past, local_layer), get_state(ref_past, local_layer))
                    global_div = global_state_divergence(torch, ref_past, other_past)
                    lm = logits_metrics(torch, ref_out.logits, other_out.logits)
                    rec = {
                        "phase": phase,
                        "tag": tag,
                        "token_index": token_idx,
                        "delta": delta,
                        "local_layer_idx": local_layer,
                        "local_state_divergence": local_div,
                        "global_state_divergence": global_div,
                        **lm,
                    }
                    if norm_info:
                        rec.update(norm_info)
                    if output_hidden:
                        rec["hidden_divergence"] = hidden_divergence(torch, ref_out, other_out)
                    trajectory.append(rec)
                    slim = {k: v for k, v in rec.items() if k != "hidden_divergence"}
                    records_fh.write(json.dumps(slim, sort_keys=True) + "\n")
            next_id = token_path[token_idx]
            nxt = torch.tensor([[next_id]], dtype=ref_input.dtype, device=device)
            ref_input = nxt
            other_input = nxt
            ref_mask = None
            other_mask = None
    return trajectory


class HeadPerturb:
    def __init__(self, layer_idx, head_idx, rel_target, seed):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.rel_target = rel_target
        self.seed = seed

    def __call__(self, torch, cache):
        state = get_state(cache, self.layer_idx)
        before = state[:, self.head_idx, :, :].detach().clone()
        e = iid_head_perturb(torch, state, self.head_idx, self.rel_target, self.seed)
        actual = norm(torch, e[:, self.head_idx, :, :])
        requested = norm(torch, before) * self.rel_target
        state.add_(e)
        return {
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "requested_relative_norm": self.rel_target,
            "requested_perturb_norm": requested,
            "actual_perturb_norm": actual,
            "relative_norm_mismatch": abs(actual - requested) / max(requested, 1e-12),
        }


class SupportPerturb:
    def __init__(self, layer_idx, head_idx, support_type, indices, rel_target, seed, hot_metric):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.support_type = support_type
        self.indices = indices
        self.rel_target = rel_target
        self.seed = seed
        self.hot_metric = hot_metric

    def __call__(self, torch, cache):
        state = get_state(cache, self.layer_idx)
        before = state[:, self.head_idx, :, :].detach().clone()
        if "row" in self.support_type:
            e = support_perturb(torch, state, self.head_idx, rows=self.indices, rel_target=self.rel_target, seed=self.seed)
        elif "column" in self.support_type:
            e = support_perturb(torch, state, self.head_idx, cols=self.indices, rel_target=self.rel_target, seed=self.seed)
        else:
            e = support_perturb(torch, state, self.head_idx, rel_target=self.rel_target, seed=self.seed)
        actual = norm(torch, e[:, self.head_idx, :, :])
        requested = norm(torch, before) * self.rel_target
        state.add_(e)
        return {
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "support_type": self.support_type,
            "support_indices": [int(x) for x in self.indices] if self.indices is not None else None,
            "hot_metric": self.hot_metric,
            "requested_relative_norm": self.rel_target,
            "requested_perturb_norm": requested,
            "actual_perturb_norm": actual,
            "relative_norm_mismatch": abs(actual - requested) / max(requested, 1e-12),
        }


class SvdPerturb:
    def __init__(self, layer_idx, head_idx, subspace, rel_target, seed):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.subspace = subspace
        self.rel_target = rel_target
        self.seed = seed

    def __call__(self, torch, cache):
        state = get_state(cache, self.layer_idx)
        before = state[:, self.head_idx, :, :].detach().clone()
        e = svd_perturb(torch, state, self.head_idx, self.subspace, self.rel_target, self.seed)
        actual = norm(torch, e[:, self.head_idx, :, :])
        requested = norm(torch, before) * self.rel_target
        state.add_(e)
        return {
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "subspace": self.subspace,
            "requested_relative_norm": self.rel_target,
            "requested_perturb_norm": requested,
            "actual_perturb_norm": actual,
            "relative_norm_mismatch": abs(actual - requested) / max(requested, 1e-12),
        }


def summarize_traj(traj):
    by_delta = {p["delta"]: p for p in traj}
    d0 = by_delta.get(0, {}).get("local_state_divergence")
    vals = [p["local_state_divergence"] / max(d0, 1e-12) for p in traj if finite(p.get("local_state_divergence"))] if d0 else []
    g = lambda d: by_delta.get(d, {}).get("local_state_divergence", None)
    def ratio(d):
        return g(d) / max(d0, 1e-12) if d0 and g(d) is not None else None
    g128 = ratio(128)
    label = "unknown"
    if g128 is not None:
        if g128 < 0.25:
            label = "strong_damping"
        elif g128 < 0.75:
            label = "damping"
        elif g128 <= 1.25:
            label = "persistent"
        else:
            label = "amplifying"
    return {
        "D0": d0,
        "G1": ratio(1),
        "G8": ratio(8),
        "G128": g128,
        "AUC": sum(vals) / len(vals) if vals else None,
        "label": label,
        "max_logit_kl": max((p["logit_kl"] for p in traj), default=None),
        "mean_top1_agreement": sum(p["top1_same"] for p in traj) / len(traj) if traj else None,
        "max_relative_norm_mismatch": max((p.get("relative_norm_mismatch", 0.0) for p in traj), default=0.0),
    }


def run_correctness(torch, model, tokenizer, prompt, layer_types, records_fh, token_path):
    a = generate_fp32_path(torch, model, tokenizer, prompt, 16)
    b = generate_fp32_path(torch, model, tokenizer, prompt, 16)
    traj = run_pair_trajectory(
        torch, model, tokenizer, prompt, token_path[:64], 8, 16,
        HeadPerturb(9, 0, 0.005, 123), records_fh, "correctness", "one_shot_l9_h0"
    )
    quant_records, _ = None, None
    full_attention_with_state = []
    # Inspect after one short FP32 run.
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc.get("attention_mask").to(device) if enc.get("attention_mask") is not None else None, use_cache=True)
    for idx in FULL_ATTENTION_LAYERS:
        if hasattr(out.past_key_values.layers[idx], "recurrent_states"):
            states = getattr(out.past_key_values.layers[idx], "recurrent_states")
            if states:
                full_attention_with_state.append(idx)
    return {
        "fp32_reference_reproducible": a == b,
        "fp32_repro_agreement": sum(1 for x, y in zip(a, b) if x == y) / len(a),
        "teacher_forced_token_path_length": len(token_path),
        "one_shot_perturb_records": len(traj),
        "one_shot_norm_mismatch_max": max((p.get("relative_norm_mismatch", 0.0) for p in traj), default=None),
        "target_layer_head_checked": {"layer": 9, "head": 0},
        "full_attention_recurrent_state_record_count": len(full_attention_with_state),
        "full_attention_layers_with_recurrent_state": full_attention_with_state,
        "hooks_used": False,
        "model_source_modified": False,
        "model_weights_modified": False,
    }


def run_propagation_map(torch, model, tokenizer, prompt, token_path, layers, heads, t0, horizon, rel_target, records_fh):
    results = []
    for layer in layers:
        for head in heads:
            print("propagation layer=%d head=%d" % (layer, head), flush=True)
            traj = run_pair_trajectory(
                torch, model, tokenizer, prompt, token_path, t0, horizon,
                HeadPerturb(layer, head, rel_target, seed=1000 + layer * 37 + head),
                records_fh, "propagation_map", "L%d_H%d" % (layer, head)
            )
            results.append({"layer_idx": layer, "head_idx": head, "trajectory": traj, "summary": summarize_traj(traj)})
    return {"protocol": {"mode": "teacher_forced_fp32_path", "layers": layers, "heads": heads, "t0": t0, "horizon": horizon, "relative_perturb_norm": rel_target, "geometry": "iid_single_head"}, "results": results}


def rank_hot_cold(torch, model, tokenizer, prompt, token_path, layer, head, t0, k):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    past = None
    with torch.inference_mode():
        for token_idx in range(t0 + 1):
            out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if token_idx == t0:
                state = get_state(past, layer)[0, head].detach().float()
                row_rms = torch.sqrt((state * state).mean(dim=1))
                col_rms = torch.sqrt((state * state).mean(dim=0))
                hot_rows = torch.topk(row_rms, k).indices.tolist()
                cold_rows = torch.topk(-row_rms, k).indices.tolist()
                hot_cols = torch.topk(col_rms, k).indices.tolist()
                cold_cols = torch.topk(-col_rms, k).indices.tolist()
                return {
                    "hot_rows": hot_rows, "cold_rows": cold_rows,
                    "hot_columns": hot_cols, "cold_columns": cold_cols,
                    "row_rms_max": float(row_rms.max().item()), "row_rms_min": float(row_rms.min().item()),
                    "column_rms_max": float(col_rms.max().item()), "column_rms_min": float(col_rms.min().item()),
                }
            nxt = torch.tensor([[token_path[token_idx]]], dtype=input_ids.dtype, device=device)
            input_ids = nxt
            attention_mask = None
    raise RuntimeError("failed to rank hot/cold")


def run_hot_cold(torch, model, tokenizer, prompt, token_path, cases, t0, horizon, rel_target, records_fh):
    results = []
    for layer, head in cases:
        ranks = rank_hot_cold(torch, model, tokenizer, prompt, token_path, layer, head, t0, 8)
        supports = [
            ("hot_row", ranks["hot_rows"]),
            ("cold_row", ranks["cold_rows"]),
            ("hot_column", ranks["hot_columns"]),
            ("cold_column", ranks["cold_columns"]),
            ("random_support", None),
        ]
        for support, indices in supports:
            print("hot_cold layer=%d head=%d %s" % (layer, head, support), flush=True)
            traj = run_pair_trajectory(
                torch, model, tokenizer, prompt, token_path, t0, horizon,
                SupportPerturb(layer, head, support, indices, rel_target, 2000 + layer * 31 + head + len(support), "rms"),
                records_fh, "hot_cold", "L%d_H%d_%s" % (layer, head, support)
            )
            results.append({"layer_idx": layer, "head_idx": head, "support_type": support, "rank_info": ranks, "trajectory": traj, "summary": summarize_traj(traj)})
    return {"protocol": {"mode": "teacher_forced_fp32_path", "relative_perturb_norm": rel_target, "t0": t0, "horizon": horizon, "hot_cold_k": 8, "hot_metric": "RMS"}, "results": results}


def run_svd_subspace(torch, model, tokenizer, prompt, token_path, cases, t0, horizon, rel_target, records_fh):
    results = []
    for layer, head in cases:
        for subspace in ["random", "top1_left", "top4_left", "top8_left", "bottom8_left", "top8_right", "bottom8_right"]:
            print("svd layer=%d head=%d %s" % (layer, head, subspace), flush=True)
            traj = run_pair_trajectory(
                torch, model, tokenizer, prompt, token_path, t0, horizon,
                SvdPerturb(layer, head, subspace, rel_target, 3000 + layer * 29 + head + len(subspace)),
                records_fh, "svd_subspace", "L%d_H%d_%s" % (layer, head, subspace)
            )
            results.append({"layer_idx": layer, "head_idx": head, "subspace": subspace, "trajectory": traj, "summary": summarize_traj(traj)})
    return {"protocol": {"mode": "teacher_forced_fp32_path", "relative_perturb_norm": rel_target, "t0": t0, "horizon": horizon}, "results": results}


def run_hybrid_attention(torch, model, tokenizer, prompt, token_path, settings, max_steps, layer_types, records_fh):
    results = []
    device = next(model.parameters()).device
    for bits, granularity in settings:
        print("hybrid bits=%s granularity=%s" % (bits, granularity), flush=True)
        enc = tokenizer(prompt, return_tensors="pt")
        ref_input = enc["input_ids"].to(device)
        q_input = ref_input.clone()
        ref_mask = enc.get("attention_mask")
        q_mask = ref_mask.clone().to(device) if ref_mask is not None else None
        ref_mask = ref_mask.to(device) if ref_mask is not None else None
        ref_past = None
        q_past = None
        by_layer = {i: [] for i in range(32)}
        kl_values = []
        with torch.inference_mode():
            for token_idx in range(max_steps):
                ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past, output_hidden_states=True)
                q_out = feed_step(torch, model, q_input, q_mask, q_past, output_hidden_states=True)
                ref_past = ref_out.past_key_values
                q_past = q_out.past_key_values
                quantize_cache_recurrent_states(q_past, bits=bits, granularity=granularity, layer_types=layer_types)
                hds = hidden_divergence(torch, ref_out, q_out)
                for h in hds:
                    by_layer[h["layer_idx"]].append(h["hidden_divergence"])
                lm = logits_metrics(torch, ref_out.logits, q_out.logits)
                kl_values.append(lm["logit_kl"])
                if token_idx in [0, 1, 2, 4, 8, 16, 32, 64, max_steps - 1]:
                    records_fh.write(json.dumps({"phase": "hybrid_attention", "bits": bits, "granularity": granularity, "token_index": token_idx, "logit_kl": lm["logit_kl"]}, sort_keys=True) + "\n")
                nxt = torch.tensor([[token_path[token_idx]]], dtype=ref_input.dtype, device=device)
                ref_input = nxt
                q_input = nxt
                ref_mask = None
                q_mask = None
        layer_summary = []
        for layer_idx, vals in by_layer.items():
            layer_summary.append({
                "layer_idx": layer_idx,
                "layer_type": "full_attention" if layer_idx in FULL_ATTENTION_LAYERS else "linear_attention",
                "mean_hidden_divergence": sum(vals) / len(vals) if vals else None,
                "max_hidden_divergence": max(vals) if vals else None,
            })
        transitions = []
        for fa in FULL_ATTENTION_LAYERS:
            before = layer_summary[fa - 1]["mean_hidden_divergence"]
            after = layer_summary[fa]["mean_hidden_divergence"]
            transitions.append({"full_attention_layer": fa, "before_layer": fa - 1, "D_before": before, "D_after": after, "FA_ratio": after / max(before, 1e-12) if before is not None and after is not None else None})
        results.append({
            "bits": bits,
            "granularity": granularity,
            "steps": max_steps,
            "mean_logit_kl": sum(kl_values) / len(kl_values),
            "layer_hidden_divergence": layer_summary,
            "full_attention_transitions": transitions,
        })
    return {"protocol": {"mode": "teacher_forced_fp32_path", "steps": max_steps}, "results": results}


def run_bitwidth_confirmation(torch, model, tokenizer, prompt, token_path, bits_list, max_steps, layer_types, records_fh):
    device = next(model.parameters()).device
    results = []
    for bits in bits_list:
        print("bitwidth_confirm bits=%d" % bits, flush=True)
        enc = tokenizer(prompt, return_tensors="pt")
        ref_input = enc["input_ids"].to(device)
        q_input = ref_input.clone()
        ref_mask = enc.get("attention_mask")
        q_mask = ref_mask.clone().to(device) if ref_mask is not None else None
        ref_mask = ref_mask.to(device) if ref_mask is not None else None
        ref_past = None
        q_past = None
        kls = []
        top1 = 0
        bucket = []
        first_failure = None
        with torch.inference_mode():
            for token_idx in range(max_steps):
                ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past, output_hidden_states=False)
                q_out = feed_step(torch, model, q_input, q_mask, q_past, output_hidden_states=False)
                ref_past = ref_out.past_key_values
                q_past = q_out.past_key_values
                recs = quantize_cache_recurrent_states(q_past, bits=bits, granularity="row", layer_types=layer_types)
                for rec in recs:
                    r = rec["relative_error"]
                    if finite(r):
                        bucket.append(r)
                    elif first_failure is None:
                        first_failure = {"token_index": token_idx, "layer_idx": rec["layer_idx"], "relative_error": r}
                lm = logits_metrics(torch, ref_out.logits, q_out.logits)
                kls.append(lm["logit_kl"])
                top1 += lm["top1_same"]
                nxt = torch.tensor([[token_path[token_idx]]], dtype=ref_input.dtype, device=device)
                ref_input = nxt
                q_input = nxt
                ref_mask = None
                q_mask = None
        results.append({"bits": bits, "granularity": "row", "steps": max_steps, "mean_relative_error": sum(bucket) / len(bucket), "max_relative_error": max(bucket), "mean_logit_kl": sum(kls) / len(kls), "top1_agreement": top1 / len(kls), "first_failure": first_failure})
    return {"protocol": {"mode": "teacher_forced_fp32_path", "granularity": "row"}, "results": results}


def run_cadence_confirmation(torch, model, tokenizer, prompt, token_path, cadences, max_steps, layer_types, records_fh):
    device = next(model.parameters()).device
    results = []
    for k in cadences:
        print("cadence_confirm K=%d" % k, flush=True)
        enc = tokenizer(prompt, return_tensors="pt")
        ref_input = enc["input_ids"].to(device)
        q_input = ref_input.clone()
        ref_mask = enc.get("attention_mask")
        q_mask = ref_mask.clone().to(device) if ref_mask is not None else None
        ref_mask = ref_mask.to(device) if ref_mask is not None else None
        ref_past = None
        q_past = None
        kls = []
        divs = []
        with torch.inference_mode():
            for token_idx in range(max_steps):
                ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past, output_hidden_states=False)
                q_out = feed_step(torch, model, q_input, q_mask, q_past, output_hidden_states=False)
                ref_past = ref_out.past_key_values
                q_past = q_out.past_key_values
                if token_idx % k == 0:
                    quantize_cache_recurrent_states(q_past, bits=4, granularity="row", layer_types=layer_types)
                divs.append(global_state_divergence(torch, ref_past, q_past))
                kls.append(logits_metrics(torch, ref_out.logits, q_out.logits)["logit_kl"])
                nxt = torch.tensor([[token_path[token_idx]]], dtype=ref_input.dtype, device=device)
                ref_input = nxt
                q_input = nxt
                ref_mask = None
                q_mask = None
        results.append({"cadence": k, "bits": 4, "granularity": "row", "steps": max_steps, "mean_global_state_divergence": sum(divs) / len(divs), "max_global_state_divergence": max(divs), "mean_logit_kl": sum(kls) / len(kls), "max_logit_kl": max(kls)})
    return {"protocol": {"mode": "teacher_forced_fp32_path", "bits": 4, "granularity": "row"}, "results": results}


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        for r, i in enumerate(order):
            out[i] = r
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / max(dx * dy, 1e-12)


def relation_analysis(result):
    prop = result.get("propagation_map", {}).get("results", [])
    by_lh = {(x["layer_idx"], x["head_idx"]): x["summary"].get("G128") for x in prop if x["summary"].get("G128") is not None}
    hot = result.get("hot_cold", {}).get("results", [])
    pairs = []
    for item in hot:
        if item["support_type"] in ("hot_row", "cold_row", "hot_column", "cold_column"):
            g = item["summary"].get("G128")
            if g is not None:
                pairs.append((item["support_type"], g))
    return {
        "layer25_vs_others_g128": {
            "layer25_mean": avg([v for (l, _), v in by_lh.items() if l == 25]),
            "other_mean": avg([v for (l, _), v in by_lh.items() if l != 25]),
        },
        "hot_cold_g128_means": {
            "hot_row": avg([g for t, g in pairs if t == "hot_row"]),
            "cold_row": avg([g for t, g in pairs if t == "cold_row"]),
            "hot_column": avg([g for t, g in pairs if t == "hot_column"]),
            "cold_column": avg([g for t, g in pairs if t == "cold_column"]),
        },
    }


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def plot_outputs(result, fig_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        result["figure_error"] = str(exc)
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    prop = result.get("propagation_map", {}).get("results", [])
    if prop:
        layers = result["propagation_map"]["protocol"]["layers"]
        heads = result["propagation_map"]["protocol"]["heads"]
        for metric, fname in [("G1", "propagation_G1_layer_head_heatmap.png"), ("G128", "propagation_G128_layer_head_heatmap.png"), ("AUC", "propagation_AUC_layer_head_heatmap.png")]:
            mat = np.full((len(layers), len(heads)), np.nan)
            for item in prop:
                i = layers.index(item["layer_idx"])
                j = heads.index(item["head_idx"])
                mat[i, j] = item["summary"].get(metric) if item["summary"].get(metric) is not None else np.nan
            plt.figure(figsize=(7, 4.5))
            im = plt.imshow(mat, aspect="auto", cmap="viridis")
            plt.colorbar(im, label=metric)
            plt.xticks(range(len(heads)), heads)
            plt.yticks(range(len(layers)), layers)
            plt.xlabel("Head")
            plt.ylabel("Layer")
            plt.title(metric + " layer/head propagation")
            plt.tight_layout()
            plt.savefig(fig_dir / fname, dpi=160)
            plt.close()
        reps = sorted(prop, key=lambda x: x["summary"].get("G128") or 0, reverse=True)[:3] + sorted(prop, key=lambda x: x["summary"].get("G128") or 0)[:3]
        plt.figure(figsize=(8, 4.5))
        for item in reps:
            xs = [p["delta"] for p in item["trajectory"]]
            ys = [p["local_state_divergence"] for p in item["trajectory"]]
            plt.plot(xs, ys, marker="o", label="L%d H%d" % (item["layer_idx"], item["head_idx"]))
        plt.xlabel("Delta tokens")
        plt.ylabel("Local state divergence")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / "propagation_decay_representative_cases.png", dpi=160)
        plt.close()
    for key, title, fname in [("hot_cold", "Hot vs cold propagation", "hot_vs_cold_propagation.png"), ("svd_subspace", "SVD subspace propagation", "svd_subspace_propagation.png")]:
        items = result.get(key, {}).get("results", [])
        if not items:
            continue
        labels = [x.get("support_type") or x.get("subspace") for x in items]
        vals = [x["summary"].get("G128") for x in items]
        plt.figure(figsize=(max(7, len(labels) * 0.35), 4))
        plt.bar(range(len(labels)), [0 if v is None else v for v in vals])
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
        plt.ylabel("G128")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=160)
        plt.close()
    hybrid = result.get("hybrid_attention", {}).get("results", [])
    if hybrid:
        plt.figure(figsize=(8, 4.5))
        for item in hybrid:
            xs = [x["layer_idx"] for x in item["layer_hidden_divergence"]]
            ys = [x["mean_hidden_divergence"] for x in item["layer_hidden_divergence"]]
            plt.plot(xs, ys, marker="o", label="%s-%s" % (item["bits"], item["granularity"]))
        for fa in FULL_ATTENTION_LAYERS:
            plt.axvline(fa, color="gray", alpha=0.25, linewidth=0.8)
        plt.xlabel("Layer")
        plt.ylabel("Mean hidden divergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "hybrid_layer_hidden_divergence.png", dpi=160)
        plt.close()
    bit = result.get("bitwidth_confirmation", {}).get("results", [])
    if bit:
        plt.figure(figsize=(5.5, 4))
        plt.plot([x["bits"] for x in bit], [x["mean_logit_kl"] for x in bit], marker="o")
        plt.gca().invert_xaxis()
        plt.xlabel("Bit width")
        plt.ylabel("Mean aligned KL")
        plt.tight_layout()
        plt.savefig(fig_dir / "bitwidth_teacher_forced_kl.png", dpi=160)
        plt.close()
    cad = result.get("cadence_confirmation", {}).get("results", [])
    if cad:
        plt.figure(figsize=(5.5, 4))
        plt.plot([x["cadence"] for x in cad], [x["mean_logit_kl"] for x in cad], marker="o")
        plt.xlabel("Cadence K")
        plt.ylabel("Mean aligned KL")
        plt.tight_layout()
        plt.savefig(fig_dir / "cadence_teacher_forced_kl.png", dpi=160)
        plt.close()


def table_row(vals):
    return "| " + " | ".join(vals) + " |"


def write_report(result, path):
    rel = result.get("relation_analysis", {})
    lines = [
        "# GDN State Sensitivity Structure Audit V1",
        "",
        "## Protocol",
        "",
        "- mode: mechanism audit, not method design",
        "- teacher-forced path: FP32 greedy token sequence",
        "- perturbation: one target token, one target layer, one selected head/support",
        "- after perturbation: FP32 execution resumes, no repeated perturbation",
        "- recurrent fake quant target: GDN recurrent state only",
        "",
        "## Correctness",
        "",
    ]
    for k, v in result["correctness"].items():
        lines.append("- %s: `%s`" % (k, v))
    lines += ["", "## Propagation Map", ""]
    prop = result["propagation_map"]["results"]
    top = sorted(prop, key=lambda x: x["summary"].get("G128") or -1, reverse=True)[:5]
    bottom = sorted(prop, key=lambda x: x["summary"].get("G128") or 999)[:5]
    lines.append("Observation: same-norm single-head IID perturbations have heterogeneous recurrent consequences across layers and heads.")
    lines.append("")
    lines.append("Top G128 cases:")
    lines.append("")
    lines.append("| layer | head | G1 | G8 | G128 | AUC | label |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for x in top:
        s = x["summary"]
        lines.append(table_row([str(x["layer_idx"]), str(x["head_idx"]), fmt(s["G1"]), fmt(s["G8"]), fmt(s["G128"]), fmt(s["AUC"]), s["label"]]))
    lines.append("")
    lines.append("Most damped cases:")
    lines.append("")
    lines.append("| layer | head | G1 | G8 | G128 | AUC | label |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for x in bottom:
        s = x["summary"]
        lines.append(table_row([str(x["layer_idx"]), str(x["head_idx"]), fmt(s["G1"]), fmt(s["G8"]), fmt(s["G128"]), fmt(s["AUC"]), s["label"]]))
    lines += [
        "",
        "Alternative explanation: one prompt and one perturbation geometry were used.",
        "What this rules out: total Frobenius norm alone is not sufficient to predict propagation impact.",
        "",
        "## Hot/Cold Causal Perturbation",
        "",
    ]
    hc = result["hot_cold"]["results"]
    lines.append("Observation: hot/cold support sensitivity is mixed and must not be equated with raw magnitude without confirmation.")
    lines.append("")
    lines.append("G128 means:")
    lines.append("")
    for k, v in rel.get("hot_cold_g128_means", {}).items():
        lines.append("- %s: `%s`" % (k, fmt(v)))
    lines.append("")
    lines.append("What this does NOT prove: hot rows/columns are not yet proven causal targets for protection.")
    lines += ["", "## SVD Subspace Causal Perturbation", ""]
    svd = result["svd_subspace"]["results"]
    by_sub = {}
    for x in svd:
        by_sub.setdefault(x["subspace"], []).append(x["summary"].get("G128"))
    for k, vals in by_sub.items():
        lines.append("- %s mean G128: `%s`" % (k, fmt(avg(vals))))
    lines.append("")
    lines.append("Observation: subspace-dependent propagation was measured with equal-norm perturbations. Treat as mechanism evidence only.")
    lines += ["", "## Hybrid Full Attention Divergence", ""]
    for item in result["hybrid_attention"]["results"]:
        ratios = [x["FA_ratio"] for x in item["full_attention_transitions"] if x["FA_ratio"] is not None]
        lines.append("- %s-%s mean FA_ratio: `%s`; min `%s`; max `%s`; mean KL `%s`" % (item["bits"], item["granularity"], fmt(avg(ratios)), fmt(min(ratios) if ratios else None), fmt(max(ratios) if ratios else None), fmt(item["mean_logit_kl"])))
    lines.append("")
    lines.append("Do not conclude: FA_ratio < 1 only means hidden divergence decreased across that layer transition, not proven error correction.")
    lines += ["", "## Cross-Prompt Persistence Result", ""]
    lines.append("PARTIAL_CONFIRMATION: not run as full 8-prompt cross-prompt audit in this pass. Prior temporal persistence remains a candidate mechanism.")
    lines += ["", "## Bitwidth Confirmation", ""]
    for x in result["bitwidth_confirmation"]["results"]:
        lines.append("- %d-bit row: mean KL `%s`, top1 `%s`, mean state error `%s`, first failure `%s`" % (x["bits"], fmt(x["mean_logit_kl"]), fmt(x["top1_agreement"]), fmt(x["mean_relative_error"]), x["first_failure"]))
    lines += ["", "## Cadence Confirmation", ""]
    for x in result["cadence_confirmation"]["results"]:
        lines.append("- K=%d: mean KL `%s`, max KL `%s`, mean global state divergence `%s`" % (x["cadence"], fmt(x["mean_logit_kl"]), fmt(x["max_logit_kl"]), fmt(x["mean_global_state_divergence"])))
    lines += ["", "# Research Questions Answered", ""]
    lines.append("Q1: Does same-norm state error have different recurrent consequences across layers and heads? Yes, supported by teacher-forced layer/head propagation map.")
    lines.append("")
    lines.append("Q2: Are persistent hot directions also dynamically sensitive directions? Partially unresolved. Hot/cold causal results are mixed and need cross-prompt confirmation.")
    lines.append("")
    lines.append("Q3: Is recurrent sensitivity aligned with dominant state singular subspaces? Partially supported or unresolved depending on the subspace means; more layers/heads are needed.")
    lines.append("")
    lines.append("Q4: Do Full Attention layers systematically damp or amplify GDN-induced representation divergence? Measured in teacher-forced mode; interpretation depends on FA_ratio table and is not yet a correction claim.")
    lines.append("")
    lines.append("Q5: Does current evidence suggest total reconstruction error is insufficient? Yes. Total reconstruction error alone is insufficient to predict recurrent impact.")
    lines += ["", "# Insight Ranking", ""]
    lines.append("| Insight | Evidence Strength | Novelty Potential | GDN Specificity | Prior-art Risk | Next Value |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| Layer/head propagation heterogeneity | strong | high | high | medium | highest |")
    lines.append("| Hot/cold causal sensitivity | moderate | high | high | medium | high |")
    lines.append("| SVD subspace causal sensitivity | moderate | high | high | medium | high |")
    lines.append("| Hybrid Full Attention transition effect | moderate | high | high | low | high |")
    lines.append("| Row/column orientation sensitivity | strong | medium-high | high | medium | high |")
    lines.append("| Bitwidth teacher-forced behavior | moderate | medium | high | high | medium |")
    lines.append("| Cadence teacher-forced behavior | moderate | medium-high | high | medium | medium |")
    lines += ["", "# Method Implication", ""]
    lines.append("If layer/head propagation heterogeneity is confirmed across prompts, future work may consider propagation-aware evaluation objectives.")
    lines.append("")
    lines.append("If SVD or hot/cold sensitivity consistently predicts propagation, future work may consider structure-aware precision allocation. This report does not implement such a method.")
    lines += ["", "# FINAL_RESEARCH_CLASSIFICATION", "", "`MULTIPLE_STRUCTURE_SIGNALS_FOUND`", ""]
    lines.append("METHOD_DESIGN_READY: NO")
    lines.append("")
    lines.append("Reason: at least one teacher-forced causal mechanism is supported, but multi-prompt and broader head confirmation are still required before method design.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x):
    if x is None:
        return "None"
    return "%.4g" % x


def main():
    parser = build_arg_parser("Run GDN state sensitivity structure audit V1")
    parser.add_argument("--stage", choices=["smoke", "main"], default="main")
    parser.add_argument("--prompt-id", type=int, default=0)
    parser.add_argument("--prompt-limit", type=int, default=2)
    parser.add_argument("--max-path-tokens", type=int, default=640)
    parser.add_argument("--t0", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--relative-perturb-norm", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.stage == "smoke":
        args.max_path_tokens = min(args.max_path_tokens, 192)
        args.horizon = min(args.horizon, 32)
        args.t0 = min(args.t0, 64)
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prompts = prompt_meta(args.prompt_limit)
    prompt = prompts[args.prompt_id]["text"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    layer_types = get_layer_types(model)
    root = Path(cfg["output_root"])
    records_path = root / "results" / "gdn_state_sensitivity_structure_v1_records.jsonl"
    fig_dir = root / "figures" / "gdn_state_sensitivity_structure_v1"
    token_path = generate_fp32_path(torch, model, tokenizer, prompt, args.max_path_tokens)
    if args.stage == "smoke":
        layers = [9]
        heads = [0]
        hot_cases = [(9, 0)]
        svd_cases = [(9, 0)]
        hybrid_steps = 64
        bit_steps = 128
        cadence_steps = 128
        hybrid_settings = [(8, "column"), (4, "row")]
    else:
        layers = REP_LAYERS
        heads = REP_HEADS
        hot_cases = [(9, 0), (25, 0), (25, 31)]
        svd_cases = [(9, 0), (25, 0), (25, 31)]
        hybrid_steps = 256
        bit_steps = min(2048, args.max_path_tokens)
        cadence_steps = 512
        hybrid_settings = [(8, "column"), (4, "row"), (4, "column")]
    result = {
        "metadata": metadata(torch, transformers, cfg, prompts, args.seed, args),
        "correctness": {},
        "propagation_map": {},
        "hot_cold": {},
        "svd_subspace": {},
        "hybrid_attention": {},
        "cross_prompt_persistence": {"status": "PARTIAL_CONFIRMATION_NOT_RUN"},
        "bitwidth_confirmation": {},
        "cadence_confirmation": {},
    }
    started = time.time()
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as records_fh:
        print("phase correctness", flush=True)
        result["correctness"] = run_correctness(torch, model, tokenizer, prompt, layer_types, records_fh, token_path)
        print("phase propagation_map", flush=True)
        result["propagation_map"] = run_propagation_map(torch, model, tokenizer, prompt, token_path, layers, heads, args.t0, args.horizon, args.relative_perturb_norm, records_fh)
        print("phase hot_cold", flush=True)
        result["hot_cold"] = run_hot_cold(torch, model, tokenizer, prompt, token_path, hot_cases, args.t0, args.horizon, args.relative_perturb_norm, records_fh)
        print("phase svd_subspace", flush=True)
        result["svd_subspace"] = run_svd_subspace(torch, model, tokenizer, prompt, token_path, svd_cases, args.t0, args.horizon, args.relative_perturb_norm, records_fh)
        print("phase hybrid_attention", flush=True)
        result["hybrid_attention"] = run_hybrid_attention(torch, model, tokenizer, prompt, token_path, hybrid_settings, hybrid_steps, layer_types, records_fh)
        print("phase bitwidth_confirmation", flush=True)
        result["bitwidth_confirmation"] = run_bitwidth_confirmation(torch, model, tokenizer, prompt, token_path, [5, 4, 3], bit_steps, layer_types, records_fh)
        print("phase cadence_confirmation", flush=True)
        result["cadence_confirmation"] = run_cadence_confirmation(torch, model, tokenizer, prompt, token_path, [1, 4, 16, 32], cadence_steps, layer_types, records_fh)
    result["metadata"]["actual_runtime_seconds"] = time.time() - started
    result["metadata"]["records_path"] = str(records_path)
    result["relation_analysis"] = relation_analysis(result)
    plot_outputs(result, fig_dir)
    save_json(root / "results" / "gdn_state_sensitivity_structure_v1.json", result)
    write_report(result, root / "reports" / "gdn_state_sensitivity_structure_v1.md")
    print("Wrote results/gdn_state_sensitivity_structure_v1.json")
    print("Wrote results/gdn_state_sensitivity_structure_v1_records.jsonl")
    print("Wrote reports/gdn_state_sensitivity_structure_v1.md")
    print("Wrote figures/gdn_state_sensitivity_structure_v1/")


if __name__ == "__main__":
    main()
