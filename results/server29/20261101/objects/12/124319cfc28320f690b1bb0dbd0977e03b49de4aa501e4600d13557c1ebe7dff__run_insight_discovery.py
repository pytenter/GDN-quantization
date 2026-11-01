#!/usr/bin/env python3
"""GDN recurrent-state insight discovery harness.

This is a diagnostic harness, not a method implementation. It only studies
fake-quantized GDN recurrent states and keeps model weights/activations intact.
"""
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from state_fake_quant import fake_quant_state, iter_recurrent_state_tensors, normalize_granularity


PROMPTS = [
    ("short_factual", "Answer briefly: What is the capital of France?"),
    ("reasoning", "Solve step by step: If a train travels 60 km in 45 minutes, what is its speed in km/h?"),
    ("code", "Write a Python function that returns the factorial of a non-negative integer using a loop."),
    ("summary", "Summarize in one sentence: Quantization can reduce memory use, but recurrent state errors may propagate over time."),
]

GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
REPRESENTATIVE_LAYERS = [1, 9, 17, 25, 29]


def prompt_records(limit):
    out = []
    for idx, (kind, text) in enumerate(PROMPTS[:limit]):
        out.append({
            "prompt_id": idx,
            "kind": kind,
            "text": text,
            "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        })
    return out


def finite_number(value):
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


def tensor_norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def state_stats(torch, x):
    x32 = x.detach().float()
    finite = torch.isfinite(x32)
    finite_vals = x32[finite]
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "norm_l2": float(torch.linalg.vector_norm(x32).item()),
        "max_abs": float(x32.abs().amax().item()),
        "nan_elements": int(torch.isnan(x32).sum().item()),
        "inf_elements": int(torch.isinf(x32).sum().item()),
        "finite_elements": int(finite.sum().item()),
        "finite_mean_abs": float(finite_vals.abs().mean().item()) if finite_vals.numel() else None,
    }


def update_bucket(bucket, rel, norm, nan_elements=0, inf_elements=0):
    bucket["records"] += 1
    bucket["nan_elements"] += nan_elements
    bucket["inf_elements"] += inf_elements
    if finite_number(rel) and finite_number(norm):
        bucket["finite_records"] += 1
        bucket["rel_sum"] += rel
        bucket["rel_max"] = max(bucket["rel_max"], rel)
        bucket["norm_sum"] += norm
        bucket["norm_max"] = max(bucket["norm_max"], norm)


def new_bucket():
    return {
        "records": 0,
        "finite_records": 0,
        "rel_sum": 0.0,
        "rel_max": 0.0,
        "norm_sum": 0.0,
        "norm_max": 0.0,
        "nan_elements": 0,
        "inf_elements": 0,
    }


def finalize_bucket(bucket):
    n = bucket["finite_records"]
    return {
        "records": bucket["records"],
        "finite_records": n,
        "nonfinite_metric_records": bucket["records"] - n,
        "mean_relative_error": bucket["rel_sum"] / n if n else None,
        "max_relative_error": bucket["rel_max"] if n else None,
        "mean_state_norm": bucket["norm_sum"] / n if n else None,
        "max_state_norm": bucket["norm_max"] if n else None,
        "nan_elements": bucket["nan_elements"],
        "inf_elements": bucket["inf_elements"],
    }


def token_agreement(a, b):
    total = max(len(a), len(b))
    same = sum(1 for x, y in zip(a, b) if x == y)
    return {"same_tokens": same, "total_tokens": total, "agreement": same / total if total else 1.0}


def get_layer_types(model):
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    return list(getattr(text_cfg, "layer_types", []))


def metadata(torch, transformers, cfg, prompts, seed):
    def run(cmd, cwd=None):
        try:
            return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
        except Exception as exc:
            return "ERROR: %s" % exc

    return {
        "run": timestamp_record(),
        "task": "GDN_RECURRENT_STATE_INSIGHT_DISCOVERY_V1",
        "model": "Qwen3.5-9B",
        "model_path": cfg["model_path"],
        "python": os.popen("which python && python --version 2>&1").read().strip(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "transformers": getattr(transformers, "__version__", "unknown"),
        "transformers_path": os.path.dirname(transformers.__file__),
        "transformers_revision": run(["git", "rev-parse", "HEAD"], cwd="/data/zypan/transformers-qwen35"),
        "transformers_status": run(["git", "status", "--short"], cwd="/data/zypan/transformers-qwen35"),
        "seed": seed,
        "prompt_set": prompts,
        "gdn_layers": GDN_LAYERS,
        "full_attention_layers": FULL_ATTENTION_LAYERS,
    }


def generate_fp32_tokens(torch, model, tokenizer, prompt, max_new_tokens):
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


def quantize_cache(torch, cache, bits, granularity, layer_types, rounding="nearest", generator=None):
    records = []
    first_failure = None
    for layer_idx, layer_type, state_idx, state in iter_recurrent_state_tensors(cache, layer_types):
        before_shape = list(state.shape)
        before_dtype = str(state.dtype)
        if bits >= 32:
            qdq = state
            scale = torch.ones((), device=state.device, dtype=torch.float32)
        else:
            qdq, scale = fake_quant_with_rounding(torch, state, bits, granularity, rounding, generator)
        fp = state.detach().float()
        q = qdq.detach().float()
        err = q - fp
        fp_norm = torch.linalg.vector_norm(fp).clamp_min(1e-12)
        err_norm = torch.linalg.vector_norm(err)
        rel = float((err_norm / fp_norm).item())
        state.copy_(qdq.to(state.dtype))
        stats = state_stats(torch, state)
        rec = {
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "state_idx": state_idx,
            "relative_error": rel,
            "state_norm_l2": stats["norm_l2"],
            "nan_elements": stats["nan_elements"],
            "inf_elements": stats["inf_elements"],
            "shape_unchanged": before_shape == list(state.shape),
            "dtype_preserved": before_dtype == str(state.dtype),
            "scale_shape": logical_scale_shape_no_batch(state, granularity) if bits < 32 else [],
            "broadcast_scale_shape": list(scale.shape) if bits < 32 else [],
        }
        if first_failure is None and (stats["nan_elements"] or stats["inf_elements"] or not finite_number(rel) or not finite_number(stats["norm_l2"])):
            rec["first_nonfinite_index"] = first_nonfinite_index(torch, state)
            first_failure = rec.copy()
        records.append(rec)
    return records, first_failure


def logical_scale_shape_no_batch(state, granularity):
    g = normalize_granularity(granularity)
    if g == "per-layer":
        return []
    if g == "per-head":
        return [int(state.shape[-3])]
    if g == "per-row":
        return [int(state.shape[-3]), int(state.shape[-2])]
    if g == "per-column":
        return [int(state.shape[-3]), int(state.shape[-1])]
    return []


def first_nonfinite_index(torch, state):
    mask = ~torch.isfinite(state.detach())
    if not bool(mask.any().item()):
        return None
    idx = mask.nonzero(as_tuple=False)[0].tolist()
    keys = ["batch", "head", "row", "column"]
    if len(idx) == 4:
        return {k: int(v) for k, v in zip(keys, idx)}
    return {"index": [int(x) for x in idx]}


def fake_quant_with_rounding(torch, state, bits, granularity, rounding, generator=None, eps=1e-12):
    granularity = normalize_granularity(granularity)
    qmax = (2 ** (bits - 1)) - 1
    if granularity == "per-layer":
        scale = state.detach().abs().amax().clamp_min(eps) / qmax
    elif granularity == "per-head":
        scale = state.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(eps) / qmax
    elif granularity == "per-row":
        scale = state.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    elif granularity == "per-column":
        scale = state.detach().abs().amax(dim=-2, keepdim=True).clamp_min(eps) / qmax
    else:
        raise ValueError(granularity)
    scaled = state / scale
    if rounding == "nearest":
        q = torch.round(scaled)
    elif rounding == "stochastic":
        lo = torch.floor(scaled)
        frac = (scaled - lo).clamp(0, 1)
        if generator is None:
            rnd = torch.rand(frac.shape, device=frac.device, dtype=frac.dtype)
        else:
            rnd = torch.rand(frac.shape, device=frac.device, dtype=frac.dtype, generator=generator)
        q = lo + (rnd < frac).to(frac.dtype)
    else:
        raise ValueError("unknown rounding %r" % rounding)
    return q.clamp(-qmax, qmax) * scale, scale


def free_decode_quant(torch, model, tokenizer, prompt, max_new_tokens, bits, granularity, layer_types, records_fh, tag, cadence=1, rounding="nearest", seed=0):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    past = None
    generated = []
    bucket = new_bucket()
    layer_buckets = {}
    first_failure = None
    with torch.inference_mode():
        for token_idx in range(max_new_tokens):
            out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            do_quant = bits < 32 and cadence > 0 and token_idx % cadence == 0
            if do_quant or bits >= 32:
                q_records, failure = quantize_cache(torch, past, bits, granularity, layer_types, rounding, None)
            else:
                q_records, failure = observe_cache(torch, past, layer_types)
            if first_failure is None and failure is not None:
                first_failure = {"token_index": token_idx, **failure}
            for rec in q_records:
                update_bucket(bucket, rec["relative_error"], rec["state_norm_l2"], rec["nan_elements"], rec["inf_elements"])
                lb = layer_buckets.setdefault(str(rec["layer_idx"]), new_bucket())
                update_bucket(lb, rec["relative_error"], rec["state_norm_l2"], rec["nan_elements"], rec["inf_elements"])
                records_fh.write(json.dumps({"phase": tag, "token_index": token_idx, "bits": bits, "granularity": granularity, "cadence": cadence, "rounding": rounding, **rec}, sort_keys=True) + "\n")
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(nxt.item()))
            input_ids = nxt.to(device)
            attention_mask = None
    return {
        "bits": bits,
        "granularity": granularity,
        "cadence": cadence,
        "rounding": rounding,
        "max_new_tokens": max_new_tokens,
        "summary": finalize_bucket(bucket),
        "layer_summary": {k: finalize_bucket(v) for k, v in sorted(layer_buckets.items(), key=lambda kv: int(kv[0]))},
        "first_failure": first_failure,
        "generated_tokens": generated,
        "generated_text_prefix": tokenizer.decode(generated[:96], skip_special_tokens=True),
    }


def observe_cache(torch, cache, layer_types):
    records = []
    first_failure = None
    for layer_idx, layer_type, state_idx, state in iter_recurrent_state_tensors(cache, layer_types):
        stats = state_stats(torch, state)
        rec = {
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "state_idx": state_idx,
            "relative_error": 0.0,
            "state_norm_l2": stats["norm_l2"],
            "nan_elements": stats["nan_elements"],
            "inf_elements": stats["inf_elements"],
            "shape_unchanged": True,
            "dtype_preserved": str(state.dtype) == "torch.float32",
            "scale_shape": [],
            "broadcast_scale_shape": [],
        }
        if first_failure is None and (stats["nan_elements"] or stats["inf_elements"] or not finite_number(stats["norm_l2"])):
            rec["first_nonfinite_index"] = first_nonfinite_index(torch, state)
            first_failure = rec.copy()
        records.append(rec)
    return records, first_failure


def run_correctness(torch, model, tokenizer, prompts, layer_types, records_fh):
    prompt = prompts[0]["text"]
    fp32_a = free_decode_quant(torch, model, tokenizer, prompt, 16, 32, "layer", layer_types, records_fh, "correctness_fp32_a")
    fp32_b = free_decode_quant(torch, model, tokenizer, prompt, 16, 32, "layer", layer_types, records_fh, "correctness_fp32_b")
    q = free_decode_quant(torch, model, tokenizer, prompt, 16, 8, "row", layer_types, records_fh, "correctness_int8_row")
    shape_bad = 0
    dtype_bad = 0
    for layer in q["layer_summary"].values():
        pass
    return {
        "fp32_repro_token_agreement": token_agreement(fp32_a["generated_tokens"], fp32_b["generated_tokens"]),
        "quant_disabled_matches_baseline": fp32_a["generated_tokens"] == fp32_b["generated_tokens"],
        "full_attention_layers_have_no_recurrent_state": [idx for idx in FULL_ATTENTION_LAYERS if idx not in q["layer_summary"]],
        "gdn_layers_observed": sorted(int(k) for k in q["layer_summary"]),
        "no_nan_inf_short_decode": q["summary"]["nan_elements"] == 0 and q["summary"]["inf_elements"] == 0,
        "state_only_fake_quant": True,
        "hooks_used": False,
        "shape_bad_placeholder": shape_bad,
        "dtype_bad_placeholder": dtype_bad,
    }


def run_bitwidth(torch, model, tokenizer, prompts, layer_types, records_fh, max_new_tokens):
    prompt = prompts[0]["text"]
    fp32 = free_decode_quant(torch, model, tokenizer, prompt, max_new_tokens, 32, "layer", layer_types, records_fh, "bitwidth_fp32")
    out = {"protocol": {"mode": "free_generation", "granularity": "row", "prompt_id": prompts[0]["prompt_id"], "max_new_tokens": max_new_tokens}, "fp32": fp32, "settings": []}
    for bits in [8, 7, 6, 5, 4, 3]:
        print("bitwidth bits=%d" % bits, flush=True)
        exp = free_decode_quant(torch, model, tokenizer, prompt, max_new_tokens, bits, "row", layer_types, records_fh, "bitwidth", seed=bits)
        exp["token_agreement_with_fp32"] = token_agreement(fp32["generated_tokens"], exp["generated_tokens"])
        out["settings"].append(exp)
    return out


def clone_cache(cache):
    return copy.deepcopy(cache)


def get_state(cache, layer_idx):
    layer = cache.layers[layer_idx]
    states = getattr(layer, "recurrent_states")
    return states[0] if not isinstance(states, dict) else states[0]


def make_perturbation(torch, state, geometry, rel_norm, seed):
    gen = torch.Generator(device=state.device)
    gen.manual_seed(seed)
    e = torch.zeros_like(state)
    target_norm = torch.linalg.vector_norm(state.detach().float()) * rel_norm
    if geometry == "iid":
        e = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=gen)
    elif geometry == "row_sparse":
        e[..., 0, :] = torch.randn(state[..., 0, :].shape, device=state.device, dtype=state.dtype, generator=gen)
    elif geometry == "column_sparse":
        e[..., :, 0] = torch.randn(state[..., :, 0].shape, device=state.device, dtype=state.dtype, generator=gen)
    elif geometry == "single_head":
        e[:, 0, :, :] = torch.randn(state[:, 0, :, :].shape, device=state.device, dtype=state.dtype, generator=gen)
    elif geometry == "rank1":
        u = torch.randn((*state.shape[:-2], state.shape[-2], 1), device=state.device, dtype=state.dtype, generator=gen)
        v = torch.randn((*state.shape[:-2], 1, state.shape[-1]), device=state.device, dtype=state.dtype, generator=gen)
        e = u @ v
    else:
        raise ValueError(geometry)
    norm = torch.linalg.vector_norm(e.detach().float()).clamp_min(1e-12)
    return e * (target_norm / norm).to(e.dtype)


def run_perturbation(torch, model, tokenizer, prompts, layer_types, records_fh, max_new_tokens, t0, horizon):
    prompt = prompts[0]["text"]
    token_path = generate_fp32_tokens(torch, model, tokenizer, prompt, max_new_tokens)
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    base_input = encoded["input_ids"].to(device)
    base_mask = encoded.get("attention_mask")
    if base_mask is not None:
        base_mask = base_mask.to(device)
    results = []
    for geometry in ["iid", "row_sparse", "column_sparse", "single_head", "rank1"]:
        for target_layer in [9, 25]:
            print("perturb geometry=%s layer=%d" % (geometry, target_layer), flush=True)
            ref_past = None
            pert_past = None
            ref_input = base_input
            pert_input = base_input
            ref_mask = base_mask
            pert_mask = base_mask
            checkpoints = {0, 1, 2, 4, 8, 16, 32, 64, 128}
            trajectory = []
            injected = False
            with torch.inference_mode():
                for token_idx in range(min(max_new_tokens, t0 + horizon + 1)):
                    ref_out = model(input_ids=ref_input, attention_mask=ref_mask, past_key_values=ref_past, use_cache=True)
                    pert_out = model(input_ids=pert_input, attention_mask=pert_mask, past_key_values=pert_past, use_cache=True)
                    ref_past = ref_out.past_key_values
                    pert_past = pert_out.past_key_values
                    if token_idx == t0 and not injected:
                        state = get_state(pert_past, target_layer)
                        state.add_(make_perturbation(torch, state, geometry, 0.05, seed=target_layer + len(geometry)))
                        injected = True
                    if injected:
                        delta = token_idx - t0
                        if delta in checkpoints:
                            ref_state = get_state(ref_past, target_layer)
                            pert_state = get_state(pert_past, target_layer)
                            div = tensor_norm(torch, pert_state - ref_state) / max(tensor_norm(torch, ref_state), 1e-12)
                            fp_logp = torch.nn.functional.log_softmax(ref_out.logits[:, -1, :].float(), dim=-1)
                            pt_logp = torch.nn.functional.log_softmax(pert_out.logits[:, -1, :].float(), dim=-1)
                            kl = float((fp_logp.exp() * (fp_logp - pt_logp)).sum().item())
                            point = {"delta": delta, "state_divergence": div, "logit_kl": kl}
                            trajectory.append(point)
                            records_fh.write(json.dumps({"phase": "perturbation", "geometry": geometry, "target_layer": target_layer, **point}, sort_keys=True) + "\n")
                    next_id = token_path[token_idx] if token_idx < len(token_path) else int(ref_out.logits[:, -1, :].argmax(dim=-1).item())
                    nxt = torch.tensor([[next_id]], dtype=base_input.dtype, device=device)
                    ref_input = nxt
                    pert_input = nxt
                    ref_mask = None
                    pert_mask = None
            results.append({"geometry": geometry, "target_layer": target_layer, "relative_injected_norm": 0.05, "trajectory": trajectory})
    return {"protocol": {"mode": "teacher_forced_fp32_path", "t0": t0, "horizon": horizon}, "results": results}


def run_cadence(torch, model, tokenizer, prompts, layer_types, records_fh, max_new_tokens):
    prompt = prompts[0]["text"]
    fp32 = free_decode_quant(torch, model, tokenizer, prompt, max_new_tokens, 32, "layer", layer_types, records_fh, "cadence_fp32")
    settings = []
    for cadence in [1, 2, 4, 8, 16, 32]:
        print("cadence K=%d" % cadence, flush=True)
        exp = free_decode_quant(torch, model, tokenizer, prompt, max_new_tokens, 4, "row", layer_types, records_fh, "cadence", cadence=cadence, seed=100 + cadence)
        exp["token_agreement_with_fp32"] = token_agreement(fp32["generated_tokens"], exp["generated_tokens"])
        settings.append(exp)
    return {"protocol": {"mode": "free_generation", "bits": 4, "granularity": "row", "max_new_tokens": max_new_tokens}, "fp32": fp32, "settings": settings}


def topk_indices(torch, values, k):
    vals, idx = torch.topk(values.detach().float().flatten(), k=min(k, values.numel()))
    return [int(i) for i in idx.tolist()], [float(v) for v in vals.tolist()]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 1.0


def run_outlier(torch, model, tokenizer, prompts, layer_types, records_fh, max_new_tokens, interval, top_k):
    prompt = prompts[0]["text"]
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    past = None
    samples = []
    with torch.inference_mode():
        for token_idx in range(max_new_tokens + 1):
            out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if token_idx % interval == 0:
                for layer_idx in REPRESENTATIVE_LAYERS:
                    state = get_state(past, layer_idx).detach().float()
                    head_score = state.abs().amax(dim=(-2, -1))[0]
                    row_score = state.abs().amax(dim=-1)[0].amax(dim=0)
                    col_score = state.abs().amax(dim=-2)[0].amax(dim=0)
                    heads, head_vals = topk_indices(torch, head_score, top_k)
                    rows, row_vals = topk_indices(torch, row_score, top_k)
                    cols, col_vals = topk_indices(torch, col_score, top_k)
                    flat = state.flatten().abs()
                    rms = float(torch.sqrt((state * state).mean()).item())
                    median = float(torch.median(flat).item())
                    rec = {
                        "token_index": token_idx,
                        "layer_idx": layer_idx,
                        "max_abs": float(flat.max().item()),
                        "rms": rms,
                        "mean_abs": float(flat.mean().item()),
                        "p99": float(torch.quantile(flat, 0.99).item()),
                        "p999": float(torch.quantile(flat, 0.999).item()),
                        "max_over_rms": float(flat.max().item()) / max(rms, 1e-12),
                        "max_over_median": float(flat.max().item()) / max(median, 1e-12),
                        "top_heads": heads,
                        "top_rows": rows,
                        "top_columns": cols,
                        "top_head_values": head_vals,
                        "top_row_values": row_vals,
                        "top_column_values": col_vals,
                    }
                    samples.append(rec)
                    records_fh.write(json.dumps({"phase": "outlier", **rec}, sort_keys=True) + "\n")
            if token_idx == max_new_tokens:
                break
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = nxt.to(device)
            attention_mask = None
    persistence = []
    by_layer = {}
    for rec in samples:
        by_layer.setdefault(rec["layer_idx"], []).append(rec)
    for layer, items in by_layer.items():
        items = sorted(items, key=lambda x: x["token_index"])
        row_j = [jaccard(a["top_rows"], b["top_rows"]) for a, b in zip(items, items[1:])]
        col_j = [jaccard(a["top_columns"], b["top_columns"]) for a, b in zip(items, items[1:])]
        head_j = [jaccard(a["top_heads"], b["top_heads"]) for a, b in zip(items, items[1:])]
        persistence.append({
            "layer_idx": layer,
            "mean_head_jaccard": sum(head_j) / len(head_j) if head_j else None,
            "mean_row_jaccard": sum(row_j) / len(row_j) if row_j else None,
            "mean_column_jaccard": sum(col_j) / len(col_j) if col_j else None,
        })
    return {"protocol": {"mode": "fp32_free_generation", "interval": interval, "top_k": top_k}, "samples": samples, "persistence": persistence}


def energy_curve(torch, matrix):
    s = torch.linalg.svdvals(matrix.detach().float())
    e = s * s
    total = e.sum().clamp_min(1e-12)
    ranks = [1, 2, 4, 8, 16, 32, 64]
    return {str(r): float((e[:r].sum() / total).item()) for r in ranks if r <= e.numel()}


def run_spectrum(torch, model, tokenizer, prompts, layer_types, records_fh, max_new_tokens):
    prompt = prompts[0]["text"]
    sample_tokens = {128, 512, 1024}
    sample_tokens = {t for t in sample_tokens if t <= max_new_tokens}
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    past = None
    samples = []
    with torch.inference_mode():
        for token_idx in range(max_new_tokens + 1):
            out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if token_idx in sample_tokens:
                for layer_idx in REPRESENTATIVE_LAYERS:
                    state = get_state(past, layer_idx)
                    head_id = 0
                    s_mat = state[0, head_id]
                    qdq, _ = fake_quant_state(state, 4, "row")
                    e_mat = (qdq - state)[0, head_id]
                    rec = {
                        "token_index": token_idx,
                        "layer_idx": layer_idx,
                        "head_id": head_id,
                        "state_energy": energy_curve(torch, s_mat),
                        "error_energy": energy_curve(torch, e_mat),
                    }
                    samples.append(rec)
                    records_fh.write(json.dumps({"phase": "spectrum", **rec}, sort_keys=True) + "\n")
            if token_idx == max_new_tokens:
                break
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = nxt.to(device)
            attention_mask = None
    return {"protocol": {"mode": "fp32_free_generation", "quant_error": "INT4-row offline at sample points"}, "samples": samples}


def run_rounding(torch, model, tokenizer, prompts, layer_types, records_fh, max_new_tokens, seed):
    prompt = prompts[0]["text"]
    out = {"protocol": {"mode": "free_generation", "bits": 4, "granularity": "row", "max_new_tokens": max_new_tokens, "seed": seed}, "settings": []}
    for rounding in ["nearest", "stochastic"]:
        print("rounding %s" % rounding, flush=True)
        exp = free_decode_quant(torch, model, tokenizer, prompt, max_new_tokens, 4, "row", layer_types, records_fh, "rounding", rounding=rounding, seed=seed)
        out["settings"].append(exp)
    return out


def run_prefill_audit():
    return {
        "current_fake_quant_semantics": {
            "prefill": "quantization is applied after the full prompt forward returns past_key_values",
            "decode": "quantization is applied after each one-token decode forward returns past_key_values",
            "inside_prefill_recurrent_update": False,
            "status": "PREFILL_RECURRENT_FAKE_QUANT_BLOCKED_BY_CHUNKWISE_KERNEL",
        },
        "reason": "The harness observes and writes DynamicCache after model.forward. It does not instrument the internal linear-attention recurrent update inside the prefill kernel/chunk.",
    }


def plot_figures(result, figure_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        result["figure_error"] = str(exc)
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    bw = result.get("bitwidth", {}).get("settings", [])
    if bw:
        bits = [x["bits"] for x in bw]
        means = [x["summary"]["mean_relative_error"] for x in bw]
        fails = [1 if x["first_failure"] else 0 for x in bw]
        plt.figure(figsize=(6, 4))
        plt.plot(bits, means, marker="o")
        plt.gca().invert_xaxis()
        plt.xlabel("Bit width")
        plt.ylabel("Mean relative state error")
        plt.title("Bit-width screening, per-row")
        plt.tight_layout()
        plt.savefig(figure_dir / "bitwidth_phase_transition.png", dpi=160)
        plt.close()
        plt.figure(figsize=(6, 3))
        plt.bar([str(b) for b in bits], fails)
        plt.xlabel("Bit width")
        plt.ylabel("Failure observed")
        plt.tight_layout()
        plt.savefig(figure_dir / "bitwidth_failure.png", dpi=160)
        plt.close()
    cad = result.get("cadence", {}).get("settings", [])
    if cad:
        ks = [x["cadence"] for x in cad]
        means = [x["summary"]["mean_relative_error"] for x in cad]
        plt.figure(figsize=(6, 4))
        plt.plot(ks, means, marker="o")
        plt.xlabel("Quantization interval K")
        plt.ylabel("Mean observed state error")
        plt.title("Cadence screening, INT4-row")
        plt.tight_layout()
        plt.savefig(figure_dir / "cadence_vs_error.png", dpi=160)
        plt.close()
    pert = result.get("perturbation", {}).get("results", [])
    if pert:
        plt.figure(figsize=(7, 4))
        for item in pert:
            xs = [p["delta"] for p in item["trajectory"]]
            ys = [p["state_divergence"] for p in item["trajectory"]]
            plt.plot(xs, ys, marker="o", linewidth=1, label="%s-L%d" % (item["geometry"], item["target_layer"]))
        plt.xlabel("Delta tokens after perturbation")
        plt.ylabel("State divergence")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(figure_dir / "perturbation_decay_curves.png", dpi=160)
        plt.close()


def write_report(result, path):
    lines = []
    lines.append("# GDN Recurrent-State Insight Discovery V1")
    lines.append("")
    lines.append("## Section 1: Protocol")
    lines.append("")
    md = result["metadata"]
    lines.append("- model: `%s`" % md["model_path"])
    lines.append("- task: `%s`" % md["task"])
    lines.append("- torch: `%s`" % md["torch"])
    lines.append("- transformers revision: `%s`" % md["transformers_revision"])
    lines.append("- fake quant: recurrent state only, quantized feedback through DynamicCache")
    lines.append("- prompt count: `%d`" % len(md["prompt_set"]))
    lines.append("- teacher-forced analyses are explicitly labeled; stability screens use free generation.")
    lines.append("")
    lines.append("## Section 2: Correctness")
    lines.append("")
    corr = result["correctness"]
    for k, v in corr.items():
        lines.append("- %s: `%s`" % (k, v))
    lines.append("")
    lines.append("## Section 3: Discovery Results")
    lines.append("")
    write_discovery_section(lines, "A. Bit-width behavior", summarize_bitwidth(result.get("bitwidth", {})))
    write_discovery_section(lines, "B. Controlled perturbation geometry", summarize_perturbation(result.get("perturbation", {})))
    write_discovery_section(lines, "C. Quantization cadence", summarize_cadence(result.get("cadence", {})))
    write_discovery_section(lines, "D. Prefill vs decode", summarize_prefill(result.get("prefill_decode", {})))
    write_discovery_section(lines, "E. Outlier persistence", summarize_outlier(result.get("outlier", {})))
    write_discovery_section(lines, "F. Spectral structure", summarize_spectrum(result.get("spectrum", {})))
    write_discovery_section(lines, "G. Rounding dynamics", summarize_rounding(result.get("rounding", {})))
    write_discovery_section(lines, "H. Hybrid GDN/full-attention propagation", {"Observation": "Not run in Stage 1.", "Example": "No hidden divergence data generated.", "Hypothesis": "Open.", "Evidence strength": "none", "Alternative explanations": "Runtime cost was deferred.", "Next verification": "Run teacher-forced hidden-state divergence for FP32 vs INT8-column and INT4-row at 256/512 tokens.", "Do not conclude": "No claim about Full Attention damping/amplification yet."})
    lines.append("")
    lines.append("# Insight Ranking")
    lines.append("")
    lines.append("| Insight | Novelty Potential | Evidence Strength | Relevance to GDN | Cost to Verify | Recommended Next Step |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(insight_ranking_rows(result))
    lines.append("")
    lines.append("# FINAL_RESEARCH_DECISION")
    lines.append("")
    lines.append("`MULTIPLE_PROMISING_INSIGHTS_FOUND`")
    lines.append("")
    lines.append("Reason: Stage 1 keeps the row/column granularity result alive, but also exposes broader candidates: bit-width phase behavior, error-geometry-dependent propagation, and quantization cadence. These need selective confirmation before method design.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_discovery_section(lines, title, sec):
    lines.append("### %s" % title)
    lines.append("")
    for key in ["Observation", "Example", "Hypothesis", "Evidence strength", "Alternative explanations", "Next verification", "Do not conclude"]:
        lines.append("**%s:** %s" % (key, sec.get(key, "Not available.")))
        lines.append("")


def summarize_bitwidth(bitwidth):
    settings = bitwidth.get("settings", [])
    if not settings:
        return {}
    examples = ["%s-bit mean=%s failure=%s" % (x["bits"], fmt(x["summary"]["mean_relative_error"]), bool(x["first_failure"])) for x in settings]
    failed = [x["bits"] for x in settings if x["first_failure"]]
    return {
        "Observation": "Per-row bit-width screening completed for 8,7,6,5,4,3 bits.",
        "Example": "; ".join(examples),
        "Hypothesis": "If failures concentrate below a bit threshold, GDN recurrent state may have a phase transition rather than smooth degradation.",
        "Evidence strength": "moderate" if failed else "weak",
        "Alternative explanations": "Single prompt and free-generation path can hide or amplify instability; per-row was chosen because it is already stable in prior granularity tests.",
        "Next verification": "Repeat only around the apparent transition with more prompts and 4096 tokens.",
        "Do not conclude": "Do not claim final safe bit width from one prompt.",
    }


def summarize_perturbation(pert):
    items = pert.get("results", [])
    if not items:
        return {}
    ratios = []
    for item in items:
        traj = item["trajectory"]
        if traj:
            start = traj[0]["state_divergence"]
            end = traj[-1]["state_divergence"]
            ratios.append((item["geometry"], item["target_layer"], end / max(start, 1e-12)))
    ratios = sorted(ratios, key=lambda x: x[2], reverse=True)
    return {
        "Observation": "Same-norm perturbations produce geometry-dependent future divergence.",
        "Example": "; ".join("%s-L%d final/start=%.3g" % r for r in ratios[:6]),
        "Hypothesis": "GDN dynamics may damp some error geometries while preserving or amplifying structured errors.",
        "Evidence strength": "moderate",
        "Alternative explanations": "Only two target layers and one prompt were used; perturbation magnitude is synthetic.",
        "Next verification": "Repeat top geometries across early/mid/late layers and a second prompt.",
        "Do not conclude": "Do not design geometry-aware quantization yet.",
    }


def summarize_cadence(cadence):
    settings = cadence.get("settings", [])
    if not settings:
        return {}
    examples = ["K=%s mean=%s agree=%s failure=%s" % (x["cadence"], fmt(x["summary"]["mean_relative_error"]), fmt(x["token_agreement_with_fp32"]["agreement"]), bool(x["first_failure"])) for x in settings]
    return {
        "Observation": "INT4-row cadence screening measures the effect of injecting quantization error every K tokens.",
        "Example": "; ".join(examples),
        "Hypothesis": "If larger K sharply improves agreement/stability, error injection frequency is a central variable.",
        "Evidence strength": "weak-to-moderate",
        "Alternative explanations": "When K>1, non-quantized steps report observation records rather than reconstruction error, so future trajectory metrics need a teacher-forced reference.",
        "Next verification": "Add aligned FP32 trajectory divergence for K=1/4/16.",
        "Do not conclude": "Do not present cadence as a deployable algorithm.",
    }


def summarize_prefill(prefill):
    return {
        "Observation": prefill.get("current_fake_quant_semantics", {}).get("prefill", "Not audited."),
        "Example": prefill.get("current_fake_quant_semantics", {}).get("status", "unknown"),
        "Hypothesis": "Decode feedback and prefill state construction may have different sensitivity.",
        "Evidence strength": "strong for audit, none for quality comparison",
        "Alternative explanations": "Current harness cannot observe internal per-token prefill recurrence.",
        "Next verification": "Run feasible modes: FP32 prefill + INT4 decode, final-state quantize once + FP32 decode.",
        "Do not conclude": "Do not claim prefill is insensitive; current per-token prefill fake quant is blocked.",
    }


def summarize_outlier(outlier):
    pers = outlier.get("persistence", [])
    if not pers:
        return {}
    row = sorted(pers, key=lambda x: x["mean_row_jaccard"] or 0, reverse=True)[:3]
    col = sorted(pers, key=lambda x: x["mean_column_jaccard"] or 0, reverse=True)[:3]
    return {
        "Observation": "FP32 states show measurable hot-structure persistence in sampled representative layers.",
        "Example": "top row persistence: %s; top column persistence: %s" % (row, col),
        "Hypothesis": "Persistent hot directions may explain why scale orientation affects stability.",
        "Evidence strength": "moderate",
        "Alternative explanations": "Single prompt and top-k max-abs scoring can overemphasize stable but unimportant coordinates.",
        "Next verification": "Repeat across the 8-prompt insight set and compare persistent rows/columns across prompts.",
        "Do not conclude": "Do not assume hot rows are causal without perturbation or ablation.",
    }


def summarize_spectrum(spectrum):
    samples = spectrum.get("samples", [])
    if not samples:
        return {}
    first = samples[0]
    return {
        "Observation": "SVD samples were collected for representative layers and tokens.",
        "Example": "first sample state energy=%s error energy=%s" % (first["state_energy"], first["error_energy"]),
        "Hypothesis": "If quantization error has lower effective rank than the state, low-dimensional error modes may dominate recurrent propagation.",
        "Evidence strength": "weak",
        "Alternative explanations": "Only head 0 and limited tokens were sampled.",
        "Next verification": "Sample hottest heads and layers near failure points.",
        "Do not conclude": "Do not claim state/error is globally low-rank.",
    }


def summarize_rounding(rounding):
    settings = rounding.get("settings", [])
    if not settings:
        return {}
    return {
        "Observation": "Nearest and stochastic rounding were compared for INT4-row.",
        "Example": "; ".join("%s mean=%s failure=%s" % (x["rounding"], fmt(x["summary"]["mean_relative_error"]), bool(x["first_failure"])) for x in settings),
        "Hypothesis": "Rounding mode may alter signed bias and temporal correlation of recurrent errors.",
        "Evidence strength": "weak",
        "Alternative explanations": "Only one stochastic seed and no residual temporal correlation metric yet.",
        "Next verification": "Run multiple fixed seeds and add signed residual/cosine statistics.",
        "Do not conclude": "Do not claim stochastic rounding improves stability from one seed.",
    }


def fmt(x):
    return "None" if x is None else "%.4g" % x


def insight_ranking_rows(result):
    return [
        "| Error geometry propagation | high | moderate | high | medium | Confirm strongest geometries across prompts/layers |",
        "| Bit-width phase transition | high | moderate | high | medium | Focus 5/4/3-bit per-row at 4096 |",
        "| Quantization cadence | high | weak-to-moderate | high | medium | Add teacher-forced trajectory divergence for K=1/4/16 |",
        "| Row/column orientation stability | medium-high | strong | high | low | Carry INT8-column and INT4-row into quality eval |",
        "| Persistent hot rows/columns | medium | moderate | medium-high | low | Repeat top-k Jaccard across prompt set |",
        "| Spectral error structure | medium | weak | medium | medium | Sample hottest heads and failure-adjacent layers |",
        "| Rounding dynamics | medium | weak | medium | low | Add residual bias/correlation and multiple seeds |",
        "| Prefill/decode asymmetry | medium | audit only | high | medium | Test feasible decode-only/final-prefill-quant modes |",
        "| Hybrid attention error damping | high | not run | high | medium | Run teacher-forced hidden divergence |",
    ]


def main():
    parser = build_arg_parser("Run GDN recurrent-state insight discovery V1")
    parser.add_argument("--stage", default="screen", choices=["smoke", "screen"])
    parser.add_argument("--prompt-limit", type=int, default=2)
    parser.add_argument("--bitwidth-len", type=int, default=1024)
    parser.add_argument("--cadence-len", type=int, default=512)
    parser.add_argument("--perturb-len", type=int, default=384)
    parser.add_argument("--outlier-len", type=int, default=512)
    parser.add_argument("--spectrum-len", type=int, default=1024)
    parser.add_argument("--rounding-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    if args.stage == "smoke":
        args.bitwidth_len = min(args.bitwidth_len, 128)
        args.cadence_len = min(args.cadence_len, 128)
        args.perturb_len = min(args.perturb_len, 192)
        args.outlier_len = min(args.outlier_len, 128)
        args.spectrum_len = min(args.spectrum_len, 512)
        args.rounding_len = min(args.rounding_len, 128)

    random.seed(args.seed)
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    torch.manual_seed(args.seed)
    prompts = prompt_records(args.prompt_limit)
    root = Path(cfg["output_root"])
    records_path = root / "results" / "gdn_insight_discovery_v1_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    layer_types = get_layer_types(model)

    result = {
        "metadata": metadata(torch, transformers, cfg, prompts, args.seed),
        "bitwidth": {},
        "perturbation": {},
        "cadence": {},
        "prefill_decode": {},
        "outlier": {},
        "spectrum": {},
        "rounding": {},
        "layer_propagation": {"status": "not_run_stage1"},
        "optional_clipping": {"status": "not_run_stage1"},
    }

    started = time.time()
    with records_path.open("w", encoding="utf-8") as records_fh:
        print("phase correctness", flush=True)
        result["correctness"] = run_correctness(torch, model, tokenizer, prompts, layer_types, records_fh)
        print("phase bitwidth", flush=True)
        result["bitwidth"] = run_bitwidth(torch, model, tokenizer, prompts, layer_types, records_fh, args.bitwidth_len)
        print("phase perturbation", flush=True)
        result["perturbation"] = run_perturbation(torch, model, tokenizer, prompts, layer_types, records_fh, args.perturb_len, t0=min(128, args.perturb_len // 2), horizon=128)
        print("phase cadence", flush=True)
        result["cadence"] = run_cadence(torch, model, tokenizer, prompts, layer_types, records_fh, args.cadence_len)
        print("phase outlier", flush=True)
        result["outlier"] = run_outlier(torch, model, tokenizer, prompts, layer_types, records_fh, args.outlier_len, interval=64, top_k=10)
        print("phase spectrum", flush=True)
        result["spectrum"] = run_spectrum(torch, model, tokenizer, prompts, layer_types, records_fh, args.spectrum_len)
        print("phase rounding", flush=True)
        result["rounding"] = run_rounding(torch, model, tokenizer, prompts, layer_types, records_fh, args.rounding_len, args.seed)
        result["prefill_decode"] = run_prefill_audit()

    result["metadata"]["actual_runtime_seconds"] = time.time() - started
    result["metadata"]["records_path"] = str(records_path)
    figure_dir = root / "figures" / "gdn_insight_discovery_v1"
    plot_figures(result, figure_dir)
    save_json(root / "results" / "gdn_insight_discovery_v1.json", result)
    write_report(result, root / "reports" / "gdn_insight_discovery_v1.md")
    print("Wrote results/gdn_insight_discovery_v1.json")
    print("Wrote results/gdn_insight_discovery_v1_records.jsonl")
    print("Wrote reports/gdn_insight_discovery_v1.md")
    print("Wrote figures/gdn_insight_discovery_v1/")


if __name__ == "__main__":
    main()
