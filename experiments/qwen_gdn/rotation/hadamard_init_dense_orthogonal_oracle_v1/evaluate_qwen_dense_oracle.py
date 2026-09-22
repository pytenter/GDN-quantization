#!/usr/bin/env python3
"""Held-out local and persistent evaluation for the Qwen dense oracle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "run_qwen_dense_oracle.py"


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = import_file(RUNNER_PATH, "qwen_dense_oracle_runner_for_eval")
HORIZONS = (1, 4, 8, 16, 32, 64, 128)


def load_bank(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank = R.C.PerLayerCayleyRotations(R.GDN_LAYERS).to(device)
    bank.load_state_dict(payload["bank"])
    bank.eval()
    return bank, payload


def matrices(method, banks, device):
    identity = torch.eye(128, device=device, dtype=torch.float32)
    if method in ("Native_INT8", "Hadamard"):
        return {layer: identity for layer in R.GDN_LAYERS}
    bank = banks[method]
    return {layer: bank.layer(layer).matrix().detach() for layer in R.GDN_LAYERS}


def use_rotation(method):
    return method != "Native_INT8"


def local_method(model, traces, method, banks, h, device):
    mats = matrices(method, banks, device)
    accum = {key: [] for key in ("state_error", "core_error", "post_norm_error", "post_gate_error", "out_proj_error", "saturation", "scale_mean", "scale_max")}
    with torch.no_grad():
        for sample in traces:
            per = {key: [] for key in accum}
            for layer in R.GDN_LAYERS:
                rec = R.to_device_record(sample["layers"][layer], device)
                state = rec["state_input"].float()
                matrix = mats[layer]
                if use_rotation(method):
                    rotated = R.transform_state(state, matrix, h)
                    quant = R.C.qwen_c128_qdq(rotated)
                    recovered = R.recover_state(quant.dequant, matrix, h)
                    core, _ = R.rotated_recurrence(rec, quant.dequant, matrix, h)
                else:
                    quant = R.C.qwen_c128_qdq(state)
                    recovered = quant.dequant
                    core, _ = R.native_recurrence(rec, quant.dequant)
                norm, output = R.local_output(model, layer, core, rec["dynamic_gate"])
                per["state_error"].append(float(R.C.relative_mse(recovered, state).cpu()))
                per["core_error"].append(float(R.C.relative_mse(core, rec["core_output"][:, 0]).cpu()))
                per["post_norm_error"].append(float(R.C.relative_mse(norm, rec["post_norm_gate"]).cpu()))
                per["post_gate_error"].append(per["post_norm_error"][-1])
                per["out_proj_error"].append(float(R.C.relative_mse(output, rec["out_proj_output"]).cpu()))
                per["saturation"].append(float((quant.codes.abs() == 127).float().mean().cpu()))
                per["scale_mean"].append(float(quant.scale.mean().cpu()))
                per["scale_max"].append(float(quant.scale.max().cpu()))
            for key in accum:
                accum[key].append(sum(per[key]) / len(per[key]))
    return {key: sum(values) / len(values) for key, values in accum.items()} | {"samples": len(traces)}


def get_state_snapshot(cache):
    return {layer: R.get_state(cache, layer).detach().float().cpu().clone() for layer in R.GDN_LAYERS}


def quantize_cache(cache):
    for layer in R.GDN_LAYERS:
        state = R.get_state(cache, layer)
        qdq = R.C.qwen_c128_qdq(state)
        state.copy_(qdq.dequant.to(state.dtype))


def recover_cache_snapshot(cache, method, mats, h):
    output = {}
    for layer in R.GDN_LAYERS:
        state = R.get_state(cache, layer).detach().float()
        output[layer] = R.recover_state(state, mats[layer], h).cpu() if use_rotation(method) else state.cpu()
    return output


def state_rel(current, reference):
    values = []
    for layer in R.GDN_LAYERS:
        values.append(float(torch.linalg.vector_norm(current[layer] - reference[layer]) / torch.linalg.vector_norm(reference[layer]).clamp_min(R.EPS)))
    return sum(values) / len(values)


def logit_metrics(value, reference):
    x, y = value.float(), reference.float()
    logp, logq = torch.log_softmax(y, -1), torch.log_softmax(x, -1)
    kl = float((logp.exp() * (logp - logq)).sum().cpu())
    rel = float((torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y).clamp_min(R.EPS)).cpu())
    top1 = int(x.argmax(-1).item() == y.argmax(-1).item())
    a = set(torch.topk(x, 20, dim=-1).indices[0].tolist())
    b = set(torch.topk(y, 20, dim=-1).indices[0].tolist())
    return {"future_kl": kl, "logit_relative_l2": rel, "top1_match": top1, "top20_overlap": len(a & b) / 20.0, "nonfinite": int((~torch.isfinite(x)).sum().cpu())}


class RuntimePatch:
    def __init__(self, model, mats, h):
        self.model, self.mats, self.h = model, mats, h
        self.enabled = False
        self.current_layer = None
        self.saved, self.handles = [], []

    def wrap(self, fn):
        def wrapped(query, key, value, *args, **kwargs):
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            q = R.normalize_qk(query) if use_norm else query
            k = R.normalize_qk(key) if use_norm else key
            if self.enabled:
                delta = self.mats[int(self.current_layer)]
                q = q.float().matmul(self.h).matmul(delta)
                k = k.float().matmul(self.h).matmul(delta)
            core, state = fn(q, k, value, *args, use_qk_l2norm_in_kernel=False, **kwargs)
            return core.to(value.dtype), state
        return wrapped

    def install(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        for index, layer in enumerate(R.model_layers(self.model)):
            module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if module is not None:
                self.handles.append(module.register_forward_pre_hook(lambda _m, _a, i=index: setattr(self, "current_layer", i)))
        for name in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
            original = getattr(qmod, name); self.saved.append((qmod, name, original)); setattr(qmod, name, self.wrap(original))

    def close(self):
        for module, name, original in self.saved: setattr(module, name, original)
        for handle in self.handles: handle.remove()


def reference_trajectory(model, ids, device, maximum):
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids[:128]], device=device), use_cache=True)
        cache = out.past_key_values
        logits, states = {}, {}
        for step in range(1, maximum + 1):
            out = model(input_ids=torch.tensor([[ids[127 + step]]], device=device), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if step in HORIZONS or step == maximum:
                logits[step] = out.logits[:, -1].detach().float().cpu()
                states[step] = get_state_snapshot(cache)
    return logits, states


def condition_trajectory(model, patch, ids, device, method, mats, h, reference_logits, reference_states, maximum):
    rows = []
    patch.enabled = use_rotation(method)
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids[:128]], device=device), use_cache=True)
        cache = out.past_key_values; quantize_cache(cache)
        for step in range(1, maximum + 1):
            out = model(input_ids=torch.tensor([[ids[127 + step]]], device=device), past_key_values=cache, use_cache=True)
            cache = out.past_key_values; quantize_cache(cache)
            if step in reference_logits:
                metrics = logit_metrics(out.logits[:, -1].detach().float().cpu(), reference_logits[step])
                metrics["state_relative_l2"] = state_rel(recover_cache_snapshot(cache, method, mats, h), reference_states[step])
                rows.append({"horizon": step, **metrics})
    patch.enabled = False
    return rows


def auc(rows):
    # Horizon-weighted trapezoidal AUC, normalized by the covered interval.
    ordered = sorted(rows, key=lambda row: row["horizon"])
    if len(ordered) == 1: return ordered[0]["future_kl"]
    area = sum((b["horizon"] - a["horizon"]) * (a["future_kl"] + b["future_kl"]) / 2 for a, b in zip(ordered, ordered[1:]))
    return area / (ordered[-1]["horizon"] - ordered[0]["horizon"])


def bootstrap(differences, hadamard, n=10000, seed=20260921):
    generator = random.Random(seed); values, relative = [], []
    for _ in range(n):
        indices = [generator.randrange(len(differences)) for _ in differences]
        effect = sorted(differences[i] for i in indices)[len(indices) // 2]
        base = sorted(hadamard[i] for i in indices)[len(indices) // 2]
        values.append(effect); relative.append(effect / max(base, R.EPS))
    values.sort(); relative.sort()
    return {"resamples": n, "median_effect": values[n // 2], "ci95": [values[int(0.025*n)], values[int(0.975*n)]], "median_relative_reduction": relative[n // 2]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", default="/data/zypan"); p.add_argument("--trace-dir", required=True)
    p.add_argument("--corpus", required=True); p.add_argument("--state-checkpoint", required=True); p.add_argument("--functional-checkpoint", required=True)
    p.add_argument("--output-dir", required=True); args = p.parse_args()
    model, tokenizer, _config, _e2e = R.load_model(args); device = model.get_input_embeddings().weight.device
    state_bank, state_meta = load_bank(args.state_checkpoint, device); func_bank, func_meta = load_bank(args.functional_checkpoint, device)
    banks = {"Dense_State": state_bank, "Dense_Functional": func_bank}
    h = R.hadamard(device); methods = ("Native_INT8", "Hadamard", "Dense_State", "Dense_Functional")
    traces = R.load_traces(Path(args.trace_dir), "VALIDATION")
    local = {method: local_method(model, traces, method, banks, h, device) for method in methods}
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True); R.save_json(outdir / "heldout_local_metrics.json", local)

    documents = R.load_corpus(Path(args.corpus), "HELDOUT")
    per_document, horizon_rows = [], []
    mats_by_method = {method: matrices(method, banks, device) for method in methods}
    patch = RuntimePatch(model, mats_by_method["Hadamard"], h); patch.install()
    try:
        for doc_index, document in enumerate(documents):
            ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
            reference_logits, reference_states = reference_trajectory(model, ids, device, 128)
            for method in methods:
                patch.mats = mats_by_method[method]
                rows = condition_trajectory(model, patch, ids, device, method, patch.mats, h, reference_logits, reference_states, 128)
                for row in rows: horizon_rows.append({"document_id": document["document_id"], "method": method, **row})
                per_document.append({"document_id": document["document_id"], "method": method, "auc": auc(rows)})
            print(f"PERSISTENT {doc_index + 1}/{len(documents)}", flush=True)
    finally: patch.close()
    summary = {method: {str(horizon): {key: sum(row[key] for row in horizon_rows if row["method"] == method and row["horizon"] == horizon) / len([row for row in horizon_rows if row["method"] == method and row["horizon"] == horizon]) for key in ("future_kl", "logit_relative_l2", "top1_match", "top20_overlap", "state_relative_l2", "nonfinite")} for horizon in HORIZONS} for method in methods}
    auc_summary = {method: sum(row["auc"] for row in per_document if row["method"] == method) / len(documents) for method in methods}
    best = min(("Dense_State", "Dense_Functional"), key=lambda method: auc_summary[method])
    by_doc = {(row["document_id"], row["method"]): row["auc"] for row in per_document}
    had = [by_doc[(d["document_id"], "Hadamard")] for d in documents]
    learned = [by_doc[(d["document_id"], best)] for d in documents]
    boot = bootstrap([a-b for a,b in zip(had, learned)], had)
    relative = boot["median_relative_reduction"]
    if boot["median_effect"] > 0 and relative >= 0.10 and boot["ci95"][0] > 0: verdict = "CLEAR_HEADROOM"
    elif boot["median_effect"] > 0: verdict = "PROMISING_HEADROOM"
    else: verdict = "NO_CLEAR_HEADROOM"
    result = {"task": R.TASK, "model": "Qwen3.5-9B/GDN", "horizons": HORIZONS, "summary": summary, "auc": auc_summary, "per_document_auc": per_document, "best_learned": best, "bootstrap_hadamard_minus_best": boot, "persistent_headroom": verdict, "checkpoint_metadata": {"Dense_State": {k: state_meta[k] for k in ("step","lr","objective","seed")}, "Dense_Functional": {k: func_meta[k] for k in ("step","lr","objective","seed")}}, "AIME26_used": False}
    R.save_json(outdir / "persistent_future_kl.json", result)
    R.save_json(outdir / "persistent_horizon_rows.json", horizon_rows)
    print(json.dumps({"auc": auc_summary, "best": best, "bootstrap": boot, "verdict": verdict}, indent=2), flush=True)


if __name__ == "__main__": main()
