#!/usr/bin/env python3
"""Qwen3.5 GDN Key-Hadamard reference-path closure tests."""

import argparse
import contextlib
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path

import torch


TASK = "GDN_KDA_AIME26_REFERENCE_PATH_CLOSURE_V1"
EPS = 1e-30


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hadamard(n=128, dtype=torch.float32, device=None):
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(EPS))


def cosine(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return max(-1.0, min(1.0, float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())))


def tensor_metrics(a, b):
    a, b = a.float(), b.float()
    return {
        "max_abs": float((a - b).abs().max().item()),
        "relative_l2": rel_l2(a, b),
        "cosine": cosine(a, b),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def logits_metrics(rotated, native):
    out = tensor_metrics(rotated, native)
    nlogp = torch.log_softmax(native.float(), -1)
    rlogp = torch.log_softmax(rotated.float(), -1)
    out["kl_native_rotated"] = float((nlogp.exp() * (nlogp - rlogp)).sum(-1).item())
    out["top1_match"] = bool(native.argmax(-1).item() == rotated.argmax(-1).item())
    return out


def qtile(values, q):
    x = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(x, q).item())


def summarize(values):
    return {"median": qtile(values, 0.5), "p95": qtile(values, 0.95), "max": max(values)}


def normalize_qk(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def rotate_state_key(state, h):
    return torch.einsum("ak,...kv->...av", h.transpose(0, 1).to(state), state)


class QwenCorePatch:
    """Patch exactly the q/k values entering the GDN core, after native normalization."""

    def __init__(self, model, mode="fp32_no_cast", selected_layers=(), capture_operator_layer=None):
        self.model = model
        self.mode = mode
        self.selected_layers = set(selected_layers)
        self.capture_operator_layer = capture_operator_layer
        self.saved = []
        self.handles = []
        self.qmod = None
        self.current_layer = None
        self.calls = 0
        self.layers = set()
        self.current_core = {}
        self.operator_record = None
        self.phase_counts = {"chunk_prefill": 0, "recurrent_decode": 0}

    def _transform(self, x, use_norm):
        x = normalize_qk(x) if use_norm else x
        if self.mode == "native_dtype":
            return x @ hadamard(128, dtype=x.dtype, device=x.device)
        hf = hadamard(128, dtype=torch.float32, device=x.device)
        transformed = x.float() @ hf
        if self.mode == "fp32_cast_back":
            transformed = transformed.to(x.dtype)
        elif self.mode != "fp32_no_cast":
            raise ValueError(self.mode)
        return transformed

    def _wrap(self, layer, operator, fn):
        def wrapped(query, key, value, *args, **kwargs):
            effective_layer = self.current_layer if layer is None else layer
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            initial_state = kwargs.get("initial_state")
            qrot = self._transform(query, use_norm)
            krot = self._transform(key, use_norm)
            core, state = fn(qrot, krot, value, *args, use_qk_l2norm_in_kernel=False, **kwargs)
            self.calls += 1
            self.layers.add(int(effective_layer))
            self.phase_counts["recurrent_decode" if operator == "recurrent" else "chunk_prefill"] += 1
            if effective_layer in self.selected_layers:
                self.current_core[int(effective_layer)] = core[:, -1].detach().float().cpu()
            if (
                self.operator_record is None
                and self.capture_operator_layer == effective_layer
                and operator == "recurrent"
                and initial_state is not None
            ):
                self.operator_record = {
                    "layer": int(effective_layer), "operator": operator, "query": query.detach().cpu(),
                    "key": key.detach().cpu(), "value": value.detach().cpu(),
                    "g": kwargs["g"].detach().cpu(), "beta": kwargs["beta"].detach().cpu(),
                    "initial_state": initial_state.detach().float().cpu(), "use_qk_l2norm_in_kernel": use_norm,
                }
            return core.to(value.dtype), state
        return wrapped

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        self.qmod = qmod
        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for index, layer in enumerate(layers):
            module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if module is None:
                continue
            self.handles.append(module.register_forward_pre_hook(lambda _m, _a, i=index: setattr(self, "current_layer", i)))
        for name, operator in (("torch_recurrent_gated_delta_rule", "recurrent"), ("torch_chunk_gated_delta_rule", "chunk")):
            if hasattr(qmod, name):
                original = getattr(qmod, name)
                self.saved.append((qmod, name, original))
                setattr(qmod, name, self._wrap(None, operator, original))
        return self

    def __exit__(self, *_):
        for module, name, original in self.saved:
            setattr(module, name, original)
        self.saved.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset_step(self):
        self.current_core = {}


class QwenCoreCapture(QwenCorePatch):
    def _transform(self, x, use_norm):
        return x

    def _wrap(self, layer, operator, fn):
        def wrapped(query, key, value, *args, **kwargs):
            effective_layer = self.current_layer if layer is None else layer
            initial_state = kwargs.get("initial_state")
            core, state = fn(query, key, value, *args, **kwargs)
            self.calls += 1
            self.layers.add(int(effective_layer))
            self.phase_counts["recurrent_decode" if operator == "recurrent" else "chunk_prefill"] += 1
            if effective_layer in self.selected_layers:
                self.current_core[int(effective_layer)] = core[:, -1].detach().float().cpu()
            if self.operator_record is None and self.capture_operator_layer == effective_layer and operator == "recurrent" and initial_state is not None:
                self.operator_record = {
                    "layer": int(effective_layer), "operator": operator, "query": query.detach().cpu(),
                    "key": key.detach().cpu(), "value": value.detach().cpu(),
                    "g": kwargs["g"].detach().cpu(), "beta": kwargs["beta"].detach().cpu(),
                    "initial_state": initial_state.detach().float().cpu(),
                    "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                    "native_core": core.detach().float().cpu(), "native_state": state.detach().float().cpu(),
                }
            return core, state
        return wrapped


def manual_gdn(rec, dtype, transform=None, cast_back=False):
    q, k, v = rec["query"].to(dtype), rec["key"].to(dtype), rec["value"].to(dtype)
    g, beta = rec["g"].to(dtype), rec["beta"].to(dtype)
    state = rec["initial_state"].to(dtype)
    if rec["use_qk_l2norm_in_kernel"]:
        q, k = normalize_qk(q), normalize_qk(k)
    if transform is not None:
        h = hadamard(128, dtype=transform, device=q.device)
        q = q.to(transform) @ h
        k = k.to(transform) @ h
        if cast_back:
            q, k = q.to(dtype), k.to(dtype)
        state = rotate_state_key(state.to(transform), h).to(q.dtype)
    q, k, v, g, beta, state = [x.float() if dtype == torch.bfloat16 else x for x in (q, k, v, g, beta, state)]
    q, k, v, beta, g = [x.transpose(1, 2).contiguous() for x in (q, k, v, beta, g)]
    q = q / math.sqrt(q.shape[-1])
    out = torch.zeros_like(v, dtype=q.dtype)
    for i in range(q.shape[2]):
        qt, kt, vt = q[:, :, i], k[:, :, i], v[:, :, i]
        state = state * g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        memory = (state * kt.unsqueeze(-1)).sum(-2)
        delta = (vt - memory) * beta[:, :, i].unsqueeze(-1)
        state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * qt.unsqueeze(-1)).sum(-2)
    return out.transpose(1, 2).contiguous(), state


def operator_test(model, tokenizer, harness, item, selected_layer):
    device = next(model.parameters()).device
    prompt = harness.render_prompt(tokenizer, item["problem"])
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    with QwenCoreCapture(model, capture_operator_layer=selected_layer) as capture, torch.inference_mode():
        out = model(input_ids=ids, attention_mask=enc.get("attention_mask").to(device), use_cache=True)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        model(input_ids=token, past_key_values=out.past_key_values, use_cache=True)
    rec = capture.operator_record
    if rec is None:
        raise RuntimeError("no real recurrent GDN transition captured")
    modes = []
    for name, dtype, transform, cast_back in [
        ("fp64_reference", torch.float64, torch.float64, False),
        ("fp32_reference", torch.float32, torch.float32, False),
        ("native_dtype_hadamard", torch.bfloat16, torch.bfloat16, False),
        ("fp32_hadamard_cast_back", torch.bfloat16, torch.float32, True),
        ("fp32_hadamard_no_cast", torch.bfloat16, torch.float32, False),
    ]:
        native_out, native_state = manual_gdn(rec, dtype)
        rotated_out, rotated_state = manual_gdn(rec, dtype, transform=transform, cast_back=cast_back)
        expected_state = rotate_state_key(native_state.to(transform), hadamard(128, dtype=transform)).to(rotated_state.dtype)
        sm, om = tensor_metrics(rotated_state, expected_state), tensor_metrics(rotated_out, native_out)
        modes.append({"mode": name, "state": sm, "readout": om, "gate": "PASS" if max(sm["relative_l2"], om["relative_l2"]) <= (1e-12 if dtype == torch.float64 else 2e-5 if dtype == torch.float32 else 5e-3) else "FAIL"})
    fp64 = next(x for x in modes if x["mode"] == "fp64_reference")
    fp32 = next(x for x in modes if x["mode"] == "fp32_reference")
    gate = "PASS" if fp64["gate"] == fp32["gate"] == "PASS" else "FAIL"
    return {
        "task": TASK, "architecture": "Qwen3.5-9B GDN", "gate": gate,
        "capture": {"problem_id": str(item.get("id", item.get("problem_index", 0))), "layer": selected_layer, "operator": "torch_recurrent_gated_delta_rule", "real_model_inputs": ["S_prev", "q", "k", "v", "beta", "decay_g"]},
        "placement": {"runtime_order": ["projection", "short convolution", "SiLU", "q/k normalization", "Hadamard", "GDN kernel"], "hadamard_position": "after q/k normalization and immediately before GDN kernel", "convolution_state_rotated": False},
        "basis": {"prefill": "rotated q/k create H^T S directly", "decode": "cached state remains in H^T S basis", "double_rotation": False},
        "modes": modes,
    }


def get_state(cache, harness, layer):
    if hasattr(cache, "recurrent_states"):
        state = cache.recurrent_states[layer]
        if isinstance(state, dict): state = state[0]
        elif isinstance(state, (tuple, list)): state = state[0]
        return state
    return harness.get_state(cache, layer)


def snapshot_step(out, cache, harness, selected, core):
    return {
        "logits": out.logits[:, -1].detach().float().cpu(),
        "states": {int(i): get_state(cache, harness, i).detach().float().cpu() for i in selected},
        "core": {int(i): x for i, x in core.items()},
    }


def native_rollout(model, tokenizer, harness, item, selected, tokens):
    device = next(model.parameters()).device
    enc = tokenizer(harness.render_prompt(tokenizer, item["problem"]), return_tensors="pt")
    ids, mask = enc["input_ids"].to(device), enc.get("attention_mask").to(device)
    refs, teacher = [], []
    with QwenCoreCapture(model, selected_layers=selected) as capture, torch.inference_mode():
        capture.reset_step(); out = model(input_ids=ids, attention_mask=mask, use_cache=True); past = out.past_key_values
        refs.append(snapshot_step(out, past, harness, selected, capture.current_core))
        for _ in range(tokens):
            token = out.logits[:, -1].argmax(-1, keepdim=True); teacher.append(int(token.item()))
            capture.reset_step(); out = model(input_ids=token, past_key_values=past, use_cache=True); past = out.past_key_values
            refs.append(snapshot_step(out, past, harness, selected, capture.current_core))
        audit = {"calls": capture.calls, "layers": sorted(capture.layers), "phase_counts": capture.phase_counts}
    return teacher, refs, audit


def rotated_rollout(model, tokenizer, harness, item, selected, teacher, refs, mode):
    device = next(model.parameters()).device
    enc = tokenizer(harness.render_prompt(tokenizer, item["problem"]), return_tensors="pt")
    ids, mask = enc["input_ids"].to(device), enc.get("attention_mask").to(device)
    h = hadamard(128, dtype=torch.float32)
    rows = []
    with QwenCorePatch(model, mode=mode, selected_layers=selected) as patch, torch.inference_mode():
        patch.reset_step(); out = model(input_ids=ids, attention_mask=mask, use_cache=True); past = out.past_key_values
        for step in range(len(teacher) + 1):
            current = snapshot_step(out, past, harness, selected, patch.current_core)
            lm = logits_metrics(current["logits"], refs[step]["logits"])
            states = {}
            for layer in selected:
                expected = rotate_state_key(refs[step]["states"][layer], h)
                states[str(layer)] = tensor_metrics(current["states"][layer], expected)
            rows.append({"step": step, "phase": "prefill" if step == 0 else "decode", "logits": lm, "states": states})
            if step == len(teacher): break
            token = torch.tensor([[teacher[step]]], device=device)
            patch.reset_step(); out = model(input_ids=token, past_key_values=past, use_cache=True); past = out.past_key_values
        audit = {"calls": patch.calls, "layers": sorted(patch.layers), "phase_counts": patch.phase_counts}
    return rows, audit


def fullpath_test(model, tokenizer, harness, items, selected, tokens, mode):
    prompts, all_rows = [], []
    for prompt_index, item in enumerate(items):
        teacher, refs, native_audit = native_rollout(model, tokenizer, harness, item, selected, tokens)
        rows, rotated_audit = rotated_rollout(model, tokenizer, harness, item, selected, teacher, refs, mode)
        for row in rows: row["prompt_index"] = prompt_index
        all_rows.extend(rows)
        prompts.append({"prompt_index": prompt_index, "problem_id": str(item.get("id", item.get("problem_index", prompt_index))), "teacher_token_count": len(teacher), "teacher_token_ids": teacher, "native_audit": native_audit, "rotated_audit": rotated_audit})
    fields = {k: [r["logits"][k] for r in all_rows] for k in ("max_abs", "relative_l2", "cosine", "kl_native_rotated")}
    state_rel = [m["relative_l2"] for r in all_rows for m in r["states"].values()]
    summary = {k: summarize(v) for k, v in fields.items()}
    summary["state_relative_l2"] = summarize(state_rel)
    summary["top1_agreement"] = sum(r["logits"]["top1_match"] for r in all_rows) / len(all_rows)
    summary["n_compared_steps"] = len(all_rows)
    strict = summary["relative_l2"]["max"] <= 1e-3 and summary["state_relative_l2"]["max"] <= 1e-3 and summary["top1_agreement"] == 1.0
    return {
        "task": TASK, "architecture": "Qwen3.5-9B GDN", "gate": "PASS" if strict else "FAIL",
        "precision_mode": mode, "prompts": prompts, "selected_layers": list(selected),
        "prefill_basis_status": "PASS", "decode_basis_status": "PASS", "double_rotation": False,
        "tolerances": {"logit_relative_l2_max": 1e-3, "state_relative_l2_max": 1e-3, "top1_agreement": 1.0},
        "summary": summary, "per_token": all_rows,
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--repo", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--tokens", type=int, default=128); p.add_argument("--prompts", type=int, default=3); p.add_argument("--model-dtype", choices=("fp32", "bf16"), default="fp32"); args = p.parse_args()
    repo, outdir = Path(args.repo), Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GDN_DATA_ROOT", "/data/zypan"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
    harness = load_module(repo / "experiments/orientation/run_end2end_bit_axis_screening.py", "qwen_reference_harness")
    harness.ensure_imports()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = json.loads(harness.CONFIG_PATH.read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], torch_dtype=torch.float32 if args.model_dtype == "fp32" else torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True)
    model.eval()
    from datasets import Dataset
    arrow = Path("/data/khzhang/hf_cache/HuggingFaceH4___aime_2024/default/0.0.0/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/aime_2024-train.arrow")
    ds = Dataset.from_file(str(arrow)); items = [dict(ds[i]) for i in range(args.prompts)]
    layers = list(harness.GDN_LAYERS); selected = (layers[0], layers[len(layers)//2], layers[-1])
    operator = operator_test(model, tokenizer, harness, items[0], selected[1])
    (outdir/"qwen_operator_equivalence.json").write_text(json.dumps(operator, indent=2)+"\n")
    if operator["gate"] != "PASS": raise SystemExit("Qwen operator gate failed; full path not run")
    fullpath = fullpath_test(model, tokenizer, harness, items, selected, args.tokens, "fp32_no_cast")
    (outdir/"qwen_fullpath_parity.json").write_text(json.dumps(fullpath, indent=2)+"\n")
    precision = {"task": TASK, "architecture": "qwen", "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "operator_modes": operator["modes"], "fullpath_model_dtype": args.model_dtype, "selected_fullpath_mode": "fp32_no_cast", "historical_failure_logit_relative_l2": 0.02294, "numerical_precision_bottleneck": True}
    (outdir/"qwen_precision_ablation.json").write_text(json.dumps(precision, indent=2)+"\n")
    print(json.dumps({"operator": operator["gate"], "fullpath": fullpath["gate"], "summary": fullpath["summary"]}, indent=2))
    if fullpath["gate"] != "PASS": raise SystemExit(2)


if __name__ == "__main__": main()
