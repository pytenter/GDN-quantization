#!/usr/bin/env python3
"""Dynamic validation for Qwen3.5 GDN recurrent-state sensitivity.

This is a validation script only. It does not change transformers, torch, CUDA,
model files, or model sources. It compares same-norm artificial perturbation
propagation against real fake-quantizer degradation on FP32 teacher-forced token
paths.
"""
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record, write_text
from state_fake_quant import fake_quant_state, logical_scale_shape, normalize_granularity
from run_state_sensitivity_structure_audit import (
    FULL_ATTENTION_LAYERS,
    GDN_LAYERS,
    HeadPerturb,
    feed_step,
    generate_fp32_path,
    get_state,
    global_state_divergence,
    hidden_divergence,
    logits_metrics,
    run_pair_trajectory,
    summarize_traj,
)


PROMPTS = [
    ("factual", "Answer briefly: What is the capital of France?"),
    ("reasoning", "Solve step by step: If a train travels 60 km in 45 minutes, what is its speed in km/h?"),
    ("code", "Write a Python function that returns factorial using a loop."),
    ("summary", "Summarize in one sentence: Recurrent state quantization can reduce memory, but errors may propagate."),
    ("long_context", "Context: The red key opens the basement. The blue key opens the attic. The green key opens the shed. Question: which key opens the attic?"),
    ("math", "Compute 17 * 23 and give only the final integer."),
    ("instruction", "List three practical reasons to run controlled ablations before proposing a quantization method."),
    ("needle", "Context: Alice bought tea. Ben stored the passcode 9137 in a drawer. Carol walked home. Question: what passcode did Ben store?"),
]

DEFAULT_LAYERS = [1, 9, 17, 25, 29]
DEFAULT_HEADS_8 = [0, 4, 8, 12, 16, 20, 24, 31]
DEFAULT_HEADS_16 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31]
QUANTIZERS = [
    {"name": "INT8-column", "bits": 8, "granularity": "column"},
    {"name": "INT5-row", "bits": 5, "granularity": "row"},
    {"name": "INT4-row", "bits": 4, "granularity": "row"},
    {"name": "INT3-row", "bits": 3, "granularity": "row"},
    {"name": "INT4-column", "bits": 4, "granularity": "column"},
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def run_cmd(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return "ERROR: %s" % exc


def parse_int_list(text, default):
    if not text:
        return list(default)
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def finite(x):
    return x is not None and x == x and x not in (float("inf"), float("-inf"))


def avg(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [float(x) for x in xs if finite(x)]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def percentile(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    vy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (vx * vy)


def rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    return pearson(rankdata([x for x, _ in pairs]), rankdata([y for _, y in pairs]))


def prompt_meta(limit):
    import hashlib

    out = []
    for i, (kind, text) in enumerate(PROMPTS[:limit]):
        out.append({"prompt_id": i, "kind": kind, "sha1": hashlib.sha1(text.encode()).hexdigest(), "text": text})
    return out


def classify_head(mean_g, std_g):
    if mean_g is None:
        return "unknown"
    if mean_g < 0.25:
        return "consistently_damping" if (std_g or 0.0) < 0.25 else "trajectory_dependent"
    if mean_g > 0.75:
        return "consistently_sensitive" if (std_g or 0.0) < 0.35 else "trajectory_dependent"
    return "trajectory_dependent"


def metadata(torch, transformers, cfg, args, layers, heads):
    return {
        "task": "GDN_DYNAMIC_SENSITIVITY_VALIDATION_V1",
        "run": timestamp_record(),
        "model": "Qwen3.5-9B",
        "model_path": cfg["model_path"],
        "python": run_cmd(["/data/ydai/miniconda3/envs/bitdecode/bin/python3.10", "--version"]),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": getattr(transformers, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "transformers_source": "/data/zypan/transformers-qwen35",
        "transformers_revision": run_cmd(["git", "rev-parse", "HEAD"], cwd="/data/zypan/transformers-qwen35"),
        "seed": args.seed,
        "stage": args.stage,
        "prompt_limit": args.prompt_limit,
        "prompt_set": prompt_meta(args.prompt_limit),
        "layers": layers,
        "heads": heads,
        "gdn_layers": GDN_LAYERS,
        "full_attention_layers": FULL_ATTENTION_LAYERS,
        "teacher_forced_path": "FP32 greedy generated token sequence",
        "constraints": {
            "model_source_modified": False,
            "model_weights_modified": False,
            "transformers_modified": False,
            "torch_cuda_modified": False,
            "new_quantization_method": False,
        },
    }


def collect_correctness(torch, model, tokenizer, prompt, token_path, records_fh):
    a = generate_fp32_path(torch, model, tokenizer, prompt, 16)
    b = generate_fp32_path(torch, model, tokenizer, prompt, 16)
    traj = run_pair_trajectory(
        torch,
        model,
        tokenizer,
        prompt,
        token_path[:64],
        8,
        16,
        HeadPerturb(9, 0, 0.005, 123),
        records_fh,
        "correctness",
        "one_shot_l9_h0",
    )
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc.get("attention_mask").to(device) if enc.get("attention_mask") is not None else None,
            use_cache=True,
        )
    s0 = get_state(out.past_key_values, GDN_LAYERS[0])
    fa_with_state = []
    for idx in FULL_ATTENTION_LAYERS:
        states = getattr(out.past_key_values.layers[idx], "recurrent_states", None)
        if states:
            fa_with_state.append(idx)
    return {
        "fp32_reference_reproducible": a == b,
        "fp32_repro_agreement": sum(1 for x, y in zip(a, b) if x == y) / max(len(a), 1),
        "state_location": "DynamicCache.layers[layer_idx].recurrent_states[0]",
        "state_shape": list(s0.shape),
        "state_dtype": str(s0.dtype),
        "state_device": str(s0.device),
        "gdn_state_count": len(GDN_LAYERS),
        "full_attention_recurrent_state_record_count": len(fa_with_state),
        "full_attention_layers_with_recurrent_state": fa_with_state,
        "one_shot_norm_mismatch_max": max((p.get("relative_norm_mismatch", 0.0) for p in traj), default=None),
        "teacher_forced_token_path_length": len(token_path),
    }


def run_propagation(torch, model, tokenizer, prompts, paths, layers, heads, t0, horizon, rel_target, records_fh):
    results = []
    for pm, prompt in prompts:
        for layer in layers:
            for head in heads:
                tag = "p%d_L%d_H%d" % (pm["prompt_id"], layer, head)
                print("[%s] propagation %s" % (now(), tag), flush=True)
                traj = run_pair_trajectory(
                    torch,
                    model,
                    tokenizer,
                    prompt,
                    paths[pm["prompt_id"]],
                    t0,
                    horizon,
                    HeadPerturb(layer, head, rel_target, 10000 + pm["prompt_id"] * 997 + layer * 37 + head),
                    records_fh,
                    "multi_prompt_propagation",
                    tag,
                )
                rec = {"prompt_id": pm["prompt_id"], "kind": pm["kind"], "layer_idx": layer, "head_idx": head, "summary": summarize_traj(traj)}
                results.append(rec)
                records_fh.write(json.dumps({"phase": "propagation_summary", **rec}, sort_keys=True) + "\n")
    by_lh = defaultdict(list)
    for r in results:
        by_lh[(r["layer_idx"], r["head_idx"])].append(r["summary"].get("G128"))
    aggs = []
    for (layer, head), vals in sorted(by_lh.items()):
        mean_g = avg(vals)
        std_g = std(vals)
        aggs.append(
            {
                "layer_idx": layer,
                "head_idx": head,
                "mean_G128": mean_g,
                "std_G128": std_g,
                "mean_AUC": avg([r["summary"].get("AUC") for r in results if r["layer_idx"] == layer and r["head_idx"] == head]),
                "label": classify_head(mean_g, std_g),
                "prompt_count": len([v for v in vals if finite(v)]),
            }
        )
    return {
        "protocol": {"mode": "teacher_forced_fp32_path", "layers": layers, "heads": heads, "t0": t0, "horizon": horizon, "relative_perturb_norm": rel_target},
        "results": results,
        "layer_head_aggregates": aggs,
    }


def per_head_errors(torch, state, qdq, layer, heads):
    out = []
    for head in heads:
        ref = state[:, head, :, :].detach().float()
        oth = qdq[:, head, :, :].detach().float()
        diff = ref - oth
        den = float(torch.linalg.vector_norm(ref).item())
        abs_l2 = float(torch.linalg.vector_norm(diff).item())
        out.append(
            {
                "layer_idx": layer,
                "head_idx": head,
                "relative_error": abs_l2 / max(den, 1e-12),
                "abs_l2_error": abs_l2,
                "state_norm_l2": den,
                "rms_error": float(torch.sqrt(torch.mean(diff * diff)).item()),
                "max_abs_error": float(diff.abs().max().item()),
                "nan_count": int(torch.isnan(qdq[:, head, :, :]).sum().item()),
                "inf_count": int(torch.isinf(qdq[:, head, :, :]).sum().item()),
            }
        )
    return out


def quantizer_profile_one(torch, model, tokenizer, pm, prompt, token_path, qspec, layers, heads, steps, record_hidden, records_fh):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    q_input = ref_input.clone()
    ref_mask = enc.get("attention_mask")
    q_mask = ref_mask.clone().to(device) if ref_mask is not None else None
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    ref_past = None
    q_past = None
    token_metrics = []
    lh_values = defaultdict(list)
    scale_shapes = {}
    failure = None
    max_steps = min(steps, len(token_path))
    with torch.inference_mode():
        for token_idx in range(max_steps):
            ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past, output_hidden_states=record_hidden)
            q_out = feed_step(torch, model, q_input, q_mask, q_past, output_hidden_states=record_hidden)
            ref_past = ref_out.past_key_values
            q_past = q_out.past_key_values
            rels = []
            norms = []
            nans = 0
            infs = 0
            for layer in layers:
                state = get_state(q_past, layer)
                qdq, _scale = fake_quant_state(state, qspec["bits"], qspec["granularity"])
                scale_shapes.setdefault(str(layer), logical_scale_shape(state, qspec["granularity"]))
                for hr in per_head_errors(torch, state, qdq, layer, heads):
                    rels.append(hr["relative_error"])
                    norms.append(hr["state_norm_l2"])
                    nans += hr["nan_count"]
                    infs += hr["inf_count"]
                    lh_values[(layer, hr["head_idx"])].append(hr["relative_error"])
                state.copy_(qdq.to(state.dtype))
            lm = logits_metrics(torch, ref_out.logits, q_out.logits)
            hvals = []
            if record_hidden:
                hvals = [x["hidden_divergence"] for x in hidden_divergence(torch, ref_out, q_out)]
            rec = {
                "phase": "quantizer_token",
                "prompt_id": pm["prompt_id"],
                "kind": pm["kind"],
                "quantizer": qspec["name"],
                "bits": qspec["bits"],
                "granularity": normalize_granularity(qspec["granularity"]),
                "token_index": token_idx,
                "mean_relative_error": avg(rels),
                "p95_relative_error": percentile(rels, 0.95),
                "mean_state_norm": avg(norms),
                "global_state_divergence": global_state_divergence(torch, ref_past, q_past),
                "mean_hidden_divergence": avg(hvals),
                "nan_count": nans,
                "inf_count": infs,
                **lm,
            }
            token_metrics.append(rec)
            if token_idx % 32 == 0 or nans or infs:
                records_fh.write(json.dumps(rec, sort_keys=True) + "\n")
            if (nans or infs) and failure is None:
                failure = {"token_index": token_idx, "nan_count": nans, "inf_count": infs}
                break
            nxt = torch.tensor([[token_path[token_idx]]], dtype=ref_input.dtype, device=device)
            ref_input = nxt
            q_input = nxt
            ref_mask = None
            q_mask = None
    layer_head = []
    for (layer, head), vals in sorted(lh_values.items()):
        layer_head.append({"layer_idx": layer, "head_idx": head, "mean_relative_error": avg(vals), "p95_relative_error": percentile(vals, 0.95), "max_relative_error": max(vals) if vals else None})
    summary = {
        "prompt_id": pm["prompt_id"],
        "kind": pm["kind"],
        "quantizer": qspec["name"],
        "bits": qspec["bits"],
        "granularity": normalize_granularity(qspec["granularity"]),
        "steps_requested": steps,
        "steps_completed": len(token_metrics),
        "first_failure": failure,
        "scale_shapes": scale_shapes,
        "mean_relative_error": avg([x["mean_relative_error"] for x in token_metrics]),
        "p95_relative_error": percentile([x["p95_relative_error"] for x in token_metrics], 0.95),
        "mean_logit_kl": avg([x["logit_kl"] for x in token_metrics]),
        "top1_agreement": avg([x["top1_same"] for x in token_metrics]),
        "mean_global_state_divergence": avg([x["global_state_divergence"] for x in token_metrics]),
        "mean_hidden_divergence": avg([x["mean_hidden_divergence"] for x in token_metrics]),
        "nan_count": sum(x["nan_count"] for x in token_metrics),
        "inf_count": sum(x["inf_count"] for x in token_metrics),
        "layer_head_errors": layer_head,
    }
    records_fh.write(json.dumps({"phase": "quantizer_summary", **summary}, sort_keys=True) + "\n")
    return summary


def run_quantizers(torch, model, tokenizer, prompts, paths, layers, heads, steps, record_hidden, records_fh):
    summaries = []
    for pm, prompt in prompts:
        for qspec in QUANTIZERS:
            print("[%s] quantizer prompt=%d %s" % (now(), pm["prompt_id"], qspec["name"]), flush=True)
            summaries.append(quantizer_profile_one(torch, model, tokenizer, pm, prompt, paths[pm["prompt_id"]], qspec, layers, heads, steps, record_hidden, records_fh))
    return {"protocol": {"mode": "teacher_forced_fp32_path", "steps": steps, "record_hidden": record_hidden, "layers": layers, "heads": heads}, "summaries": summaries}


def predictive_validation(propagation, profiles):
    prop = {(r["prompt_id"], r["layer_idx"], r["head_idx"]): r["summary"] for r in propagation["results"]}
    samples = []
    for s in profiles["summaries"]:
        raw = []
        raw_p95 = []
        pw_g = []
        pw_auc = []
        for lh in s["layer_head_errors"]:
            p = prop.get((s["prompt_id"], lh["layer_idx"], lh["head_idx"]), {})
            err = lh.get("mean_relative_error")
            raw.append(err)
            raw_p95.append(lh.get("p95_relative_error"))
            if finite(err) and finite(p.get("G128")):
                pw_g.append(err * p["G128"])
            if finite(err) and finite(p.get("AUC")):
                pw_auc.append(err * p["AUC"])
        samples.append(
            {
                "prompt_id": s["prompt_id"],
                "kind": s["kind"],
                "quantizer": s["quantizer"],
                "raw_mean_error": avg(raw),
                "raw_p95_error": percentile(raw_p95, 0.95),
                "pw_G128_mean": avg(pw_g),
                "pw_AUC_mean": avg(pw_auc),
                "target_mean_KL": s["mean_logit_kl"],
                "target_top1_disagreement": None if s["top1_agreement"] is None else 1.0 - s["top1_agreement"],
                "target_global_divergence": s["mean_global_state_divergence"],
                "target_hidden_divergence": s["mean_hidden_divergence"],
            }
        )
    predictors = ["raw_mean_error", "raw_p95_error", "pw_G128_mean", "pw_AUC_mean"]
    targets = ["target_mean_KL", "target_top1_disagreement", "target_global_divergence", "target_hidden_divergence"]
    corrs = []
    for pred in predictors:
        for target in targets:
            corrs.append({"predictor": pred, "target": target, "pearson": pearson([x[pred] for x in samples], [x[target] for x in samples]), "spearman": spearman([x[pred] for x in samples], [x[target] for x in samples]), "n": sum(1 for x in samples if finite(x[pred]) and finite(x[target]))})
    return {"samples": samples, "correlations": corrs}


def cadence_one(torch, model, tokenizer, pm, prompt, token_path, steps, cadence, records_fh):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    q_input = ref_input.clone()
    ref_mask = enc.get("attention_mask")
    q_mask = ref_mask.clone().to(device) if ref_mask is not None else None
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    ref_past = None
    q_past = None
    metrics = []
    failure = None
    with torch.inference_mode():
        for token_idx in range(min(steps, len(token_path))):
            ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past)
            q_out = feed_step(torch, model, q_input, q_mask, q_past)
            ref_past = ref_out.past_key_values
            q_past = q_out.past_key_values
            if token_idx % cadence == 0:
                for layer in GDN_LAYERS:
                    state = get_state(q_past, layer)
                    qdq, _ = fake_quant_state(state, 4, "row")
                    state.copy_(qdq.to(state.dtype))
            lm = logits_metrics(torch, ref_out.logits, q_out.logits)
            gd = global_state_divergence(torch, ref_past, q_past)
            bad = 0
            for layer in GDN_LAYERS:
                st = get_state(q_past, layer)
                bad += int(torch.isnan(st).sum().item()) + int(torch.isinf(st).sum().item())
            if bad and failure is None:
                failure = {"token_index": token_idx, "bad_count": bad}
                break
            metrics.append({"logit_kl": lm["logit_kl"], "top1_same": lm["top1_same"], "global_state_divergence": gd})
            if token_idx % 64 == 0:
                records_fh.write(json.dumps({"phase": "cadence_token", "prompt_id": pm["prompt_id"], "cadence": cadence, "token_index": token_idx, **metrics[-1]}, sort_keys=True) + "\n")
            nxt = torch.tensor([[token_path[token_idx]]], dtype=ref_input.dtype, device=device)
            ref_input = nxt
            q_input = nxt
            ref_mask = None
            q_mask = None
    return {"prompt_id": pm["prompt_id"], "kind": pm["kind"], "quantizer": "INT4-row", "cadence": cadence, "steps_completed": len(metrics), "first_failure": failure, "mean_logit_kl": avg([m["logit_kl"] for m in metrics]), "top1_agreement": avg([m["top1_same"] for m in metrics]), "mean_global_state_divergence": avg([m["global_state_divergence"] for m in metrics])}


def run_cadence(torch, model, tokenizer, prompts, paths, steps, cadence_values, records_fh):
    summaries = []
    for pm, prompt in prompts:
        for cadence in cadence_values:
            print("[%s] cadence prompt=%d K=%d" % (now(), pm["prompt_id"], cadence), flush=True)
            summaries.append(cadence_one(torch, model, tokenizer, pm, prompt, paths[pm["prompt_id"]], steps, cadence, records_fh))
    by_cadence = []
    for cadence in cadence_values:
        rows = [x for x in summaries if x["cadence"] == cadence]
        by_cadence.append({"cadence": cadence, "prompt_count": len(rows), "mean_KL": avg([r["mean_logit_kl"] for r in rows]), "mean_top1_agreement": avg([r["top1_agreement"] for r in rows]), "mean_global_divergence": avg([r["mean_global_state_divergence"] for r in rows]), "failures": sum(1 for r in rows if r["first_failure"] is not None)})
    return {"protocol": {"quantizer": "INT4-row", "steps": steps, "cadence_values": cadence_values}, "summaries": summaries, "by_cadence": by_cadence}


def static_validation(torch, model, tokenizer, prompts, paths, layers, heads, t0, propagation):
    prop = {(r["prompt_id"], r["layer_idx"], r["head_idx"]): r["summary"].get("G128") for r in propagation["results"]}
    rows = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for pm, prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt")
            inp = enc["input_ids"].to(device)
            mask = enc.get("attention_mask")
            mask = mask.to(device) if mask is not None else None
            past = None
            for token_idx in range(t0 + 1):
                out = model(input_ids=inp, attention_mask=mask, past_key_values=past, use_cache=True)
                past = out.past_key_values
                inp = torch.tensor([[paths[pm["prompt_id"]][token_idx]]], dtype=inp.dtype, device=device)
                mask = None
            for layer in layers:
                st = get_state(past, layer)[0].detach().float()
                for head in heads:
                    mat = st[head]
                    sv = torch.linalg.svdvals(mat)
                    total = float(torch.sum(sv * sv).item())
                    probs = (sv * sv) / max(total, 1e-12)
                    rows.append({"prompt_id": pm["prompt_id"], "layer_idx": layer, "head_idx": head, "state_norm": float(torch.linalg.vector_norm(mat).item()), "rms": float(torch.sqrt(torch.mean(mat * mat)).item()), "max_abs": float(mat.abs().max().item()), "mean_abs": float(mat.abs().mean().item()), "sv_top1_energy": float((sv[0] * sv[0]).item() / max(total, 1e-12)), "sv_top4_energy": float(torch.sum(sv[:4] * sv[:4]).item() / max(total, 1e-12)), "sv_top8_energy": float(torch.sum(sv[:8] * sv[:8]).item() / max(total, 1e-12)), "effective_rank_entropy": float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item()), "target_G128": prop.get((pm["prompt_id"], layer, head))})
    metrics = ["state_norm", "rms", "max_abs", "mean_abs", "sv_top1_energy", "sv_top4_energy", "sv_top8_energy", "effective_rank_entropy"]
    corrs = [{"metric": m, "target": "G128", "pearson": pearson([r[m] for r in rows], [r["target_G128"] for r in rows]), "spearman": spearman([r[m] for r in rows], [r["target_G128"] for r in rows]), "n": len(rows)} for m in metrics]
    return {"protocol": {"t0": t0, "layers": layers, "heads": heads}, "correlations": corrs, "rows_sample": rows[:200]}


def previous_hybrid_summary():
    path = Path("/data/zypan/results/gdn_state_sensitivity_structure_v1.json")
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text())
    return {"available": True, "source": str(path), "hybrid_attention_summary": data.get("hybrid_attention_summary")}


def plot_all(result, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        result["figure_error"] = str(exc)
        return
    aggs = result["propagation_confirmation"]["layer_head_aggregates"]
    layers = result["metadata"]["layers"]
    heads = result["metadata"]["heads"]
    li = {v: i for i, v in enumerate(layers)}
    hi = {v: i for i, v in enumerate(heads)}
    for metric, fname in [("mean_G128", "mean_G128_layer_head_heatmap.png"), ("std_G128", "std_G128_layer_head_heatmap.png")]:
        arr = np.full((len(layers), len(heads)), np.nan)
        for r in aggs:
            arr[li[r["layer_idx"]], hi[r["head_idx"]]] = r.get(metric) if r.get(metric) is not None else np.nan
        plt.figure(figsize=(max(6, len(heads) * 0.45), max(4, len(layers) * 0.55)))
        plt.imshow(arr, aspect="auto", interpolation="nearest")
        plt.colorbar(label=metric)
        plt.xticks(range(len(heads)), heads)
        plt.yticks(range(len(layers)), layers)
        plt.xlabel("head")
        plt.ylabel("layer")
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=160)
        plt.close()
    scatters = [
        ("raw_mean_error", "target_mean_KL", "raw_error_vs_KL.png"),
        ("pw_G128_mean", "target_mean_KL", "propagation_weighted_error_vs_KL.png"),
        ("raw_mean_error", "target_global_divergence", "raw_error_vs_global_divergence.png"),
        ("pw_G128_mean", "target_global_divergence", "propagation_weighted_error_vs_global_divergence.png"),
    ]
    for xk, yk, fname in scatters:
        plt.figure(figsize=(6, 4))
        for s in result["predictive_validation"]["samples"]:
            if finite(s.get(xk)) and finite(s.get(yk)):
                plt.scatter(s[xk], s[yk], s=28)
        plt.xlabel(xk)
        plt.ylabel(yk)
        plt.title(fname.replace("_", " ").replace(".png", ""))
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=160)
        plt.close()
    byq = defaultdict(list)
    for s in result["quantizer_error_profiles"]["summaries"]:
        byq[s["quantizer"]].append(s["mean_relative_error"])
    plt.figure(figsize=(7, 4))
    names = list(byq)
    plt.bar(names, [avg(byq[n]) for n in names])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("mean relative state error")
    plt.title("quantizer layer dynamic risk")
    plt.tight_layout()
    plt.savefig(fig_dir / "quantizer_layer_dynamic_risk.png", dpi=160)
    plt.close()
    cad = result["cadence_confirmation"]["by_cadence"]
    if cad:
        xs = [r["cadence"] for r in cad]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, [r["mean_KL"] for r in cad], marker="o")
        plt.xlabel("cadence K")
        plt.ylabel("mean KL")
        plt.tight_layout()
        plt.savefig(fig_dir / "cadence_multi_prompt_KL.png", dpi=160)
        plt.close()
        plt.figure(figsize=(6, 4))
        plt.plot(xs, [r["mean_global_divergence"] for r in cad], marker="o")
        plt.xlabel("cadence K")
        plt.ylabel("mean global state divergence")
        plt.tight_layout()
        plt.savefig(fig_dir / "cadence_multi_prompt_divergence.png", dpi=160)
        plt.close()


def research_decision(result):
    def corr(pred, target):
        for c in result["predictive_validation"]["correlations"]:
            if c["predictor"] == pred and c["target"] == target:
                return c
        return {}

    raw = corr("raw_mean_error", "target_mean_KL").get("spearman")
    pw = corr("pw_G128_mean", "target_mean_KL").get("spearman")
    improves = finite(raw) and finite(pw) and abs(pw) > abs(raw)
    return {
        "final_classification": "DYNAMIC_SENSITIVITY_VALIDATED" if improves else "DYNAMIC_SENSITIVITY_PARTIALLY_VALIDATED",
        "method_design_ready": "NO",
        "raw_error_prediction_result": raw,
        "propagation_weighted_prediction_result": pw,
        "propagation_weighting_improves_KL_prediction": improves,
        "current_strongest_research_conclusion": "This run validates whether artificial propagation sensitivity explains real quantizer degradation better than raw reconstruction error. It is evidence gathering, not method design.",
        "recommended_next_step": "Repeat with a denser head grid or longer paths before designing adaptive precision.",
    }


def report_text(result):
    aggs = result["propagation_confirmation"]["layer_head_aggregates"]
    top = sorted(aggs, key=lambda x: -1e9 if x.get("mean_G128") is None else x["mean_G128"], reverse=True)[:8]
    bot = sorted(aggs, key=lambda x: 1e9 if x.get("mean_G128") is None else x["mean_G128"])[:8]
    slim_profiles = [{k: s[k] for k in ["prompt_id", "quantizer", "steps_completed", "mean_relative_error", "mean_logit_kl", "top1_agreement", "mean_global_state_divergence", "first_failure"]} for s in result["quantizer_error_profiles"]["summaries"]]
    lines = [
        "# GDN Dynamic Sensitivity Validation V1",
        "",
        "## Goal",
        "Validate whether same-norm artificial recurrent-state perturbation sensitivity predicts real fake-quantizer degradation.",
        "",
        "## Protocol",
        "Teacher-forced FP32 greedy paths; recurrent state only; no model/source/runtime changes.",
        "Layers: `%s`; heads: `%s`." % (result["metadata"]["layers"], result["metadata"]["heads"]),
        "",
        "## Correctness",
        "```json\n%s\n```" % json.dumps(result["correctness"], indent=2, sort_keys=True),
        "",
        "## Multi-Prompt Propagation Map",
        "Top sensitive entries: `%s`." % top,
        "",
        "Strong damping entries: `%s`." % bot,
        "",
        "## Real Quantizer Error Profiles",
        "```json\n%s\n```" % json.dumps(slim_profiles, indent=2, sort_keys=True),
        "",
        "## Raw vs Propagation-Weighted Prediction",
        "```json\n%s\n```" % json.dumps(result["predictive_validation"]["correlations"], indent=2, sort_keys=True),
        "",
        "## Row vs Column Explanation",
        "Row and column differences are reported as measured real fake-quantization error distributions plus KL/top1/global divergence. No new algorithm is proposed.",
        "",
        "## Cadence Multi-Prompt Confirmation",
        "```json\n%s\n```" % json.dumps(result["cadence_confirmation"]["by_cadence"], indent=2, sort_keys=True),
        "",
        "## Static-Metric Negative Results",
        "```json\n%s\n```" % json.dumps(result["static_metric_validation"]["correlations"], indent=2, sort_keys=True),
        "",
        "## Hybrid Attention Summary",
        "```json\n%s\n```" % json.dumps(result["hybrid_attention_summary"], indent=2, sort_keys=True)[:4000],
        "",
        "## Limitations",
        "This is a validation-stage run. It may use a sampled head grid when the full 8 prompt x 5 layer x 32 head map is too expensive. No LongBench is run.",
        "",
        "## Research Decision",
        "```json\n%s\n```" % json.dumps(result["research_decision"], indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def main():
    parser = build_arg_parser("GDN dynamic sensitivity validation")
    parser.add_argument("--stage", default="main", choices=["smoke", "main"])
    parser.add_argument("--prompt-limit", type=int, default=4)
    parser.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    parser.add_argument("--heads", default=",".join(str(x) for x in DEFAULT_HEADS_8))
    parser.add_argument("--head-grid", default="8", choices=["8", "16", "custom"])
    parser.add_argument("--path-tokens", type=int, default=1024)
    parser.add_argument("--prop-t0", type=int, default=128)
    parser.add_argument("--prop-horizon", type=int, default=128)
    parser.add_argument("--rel-perturb", type=float, default=0.005)
    parser.add_argument("--quant-steps", type=int, default=512)
    parser.add_argument("--cadence-steps", type=int, default=256)
    parser.add_argument("--cadence", default="1,4,16,32")
    parser.add_argument("--record-hidden", action="store_true")
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.stage == "smoke":
        args.prompt_limit = min(args.prompt_limit, 1)
        args.layers = "9"
        args.heads = "0,8"
        args.path_tokens = min(args.path_tokens, 192)
        args.prop_t0 = min(args.prop_t0, 32)
        args.prop_horizon = min(args.prop_horizon, 32)
        args.quant_steps = min(args.quant_steps, 96)
        args.cadence_steps = min(args.cadence_steps, 64)

    cfg = resolve_cfg(args)
    layers = parse_int_list(args.layers, DEFAULT_LAYERS)
    if args.head_grid == "16" and args.heads == ",".join(str(x) for x in DEFAULT_HEADS_8):
        heads = DEFAULT_HEADS_16
    else:
        heads = parse_int_list(args.heads, DEFAULT_HEADS_8)
    random.seed(args.seed)
    torch = require("torch")
    transformers = require("transformers")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[%s] loading model %s" % (now(), cfg["model_path"]), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype="auto", device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()

    prompts = prompt_meta(args.prompt_limit)
    prompt_pairs = [(pm, PROMPTS[pm["prompt_id"]][1]) for pm in prompts]
    records_path = Path("/data/zypan/results/gdn_dynamic_sensitivity_validation_v1_records.jsonl")
    results_path = Path("/data/zypan/results/gdn_dynamic_sensitivity_validation_v1.json")
    report_path = Path("/data/zypan/reports/gdn_dynamic_sensitivity_validation_v1.md")
    fig_dir = Path("/data/zypan/figures/gdn_dynamic_sensitivity_validation_v1")
    records_path.parent.mkdir(parents=True, exist_ok=True)

    paths = {}
    with records_path.open("w", encoding="utf-8") as records_fh:
        print("[%s] generating FP32 paths" % now(), flush=True)
        for pm, prompt in prompt_pairs:
            path = generate_fp32_path(torch, model, tokenizer, prompt, args.path_tokens)
            paths[pm["prompt_id"]] = path
            records_fh.write(json.dumps({"phase": "fp32_path", "prompt_id": pm["prompt_id"], "tokens": len(path)}, sort_keys=True) + "\n")
        correctness = collect_correctness(torch, model, tokenizer, prompt_pairs[0][1], paths[prompt_pairs[0][0]["prompt_id"]], records_fh)
        propagation = run_propagation(torch, model, tokenizer, prompt_pairs, paths, layers, heads, args.prop_t0, args.prop_horizon, args.rel_perturb, records_fh)
        profiles = run_quantizers(torch, model, tokenizer, prompt_pairs, paths, layers, heads, args.quant_steps, args.record_hidden, records_fh)
        predictive = predictive_validation(propagation, profiles)
        cadence_values = [int(x) for x in args.cadence.split(",") if x.strip()]
        cadence = run_cadence(torch, model, tokenizer, prompt_pairs, paths, args.cadence_steps, cadence_values, records_fh)
    static = static_validation(torch, model, tokenizer, prompt_pairs, paths, layers, heads, args.prop_t0, propagation)

    result = {
        "metadata": metadata(torch, transformers, cfg, args, layers, heads),
        "correctness": correctness,
        "propagation_confirmation": propagation,
        "quantizer_error_profiles": profiles,
        "predictive_validation": predictive,
        "cadence_confirmation": cadence,
        "static_metric_validation": static,
        "hybrid_attention_summary": previous_hybrid_summary(),
    }
    result["research_decision"] = research_decision(result)
    plot_all(result, fig_dir)
    result["files"] = {"results_json": str(results_path), "records_jsonl": str(records_path), "report_md": str(report_path), "figure_dir": str(fig_dir)}
    save_json(results_path, result)
    write_text(report_path, report_text(result))
    print(json.dumps({"done": True, "result": str(results_path), "report": str(report_path), "classification": result["research_decision"]["final_classification"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
