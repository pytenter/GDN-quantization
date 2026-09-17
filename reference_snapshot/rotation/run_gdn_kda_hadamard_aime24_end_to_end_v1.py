#!/usr/bin/env python3
"""Axis-matched deterministic H128 AIME24 experiment for Qwen GDN and Ling KDA."""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch


TASK = "GDN_KDA_HADAMARD_AIME24_END_TO_END_V1"
SLUG = "gdn_kda_hadamard_aime24_end_to_end_v1"
REPO = Path(os.environ.get("GDN_KDA_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = Path(os.environ.get("GDN_KDA_HADAMARD_RESULT_DIR", REPO / "results" / SLUG))
GDN_DATA_ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/data/zypan"))
LING_MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data/zypan/models/Ling-3.0-tiny"))
GDN_HARNESS_PATH = REPO / "experiments" / "orientation" / "run_end2end_bit_axis_screening.py"
KDA_HARNESS_PATH = REPO / "experiments" / "ling" / "run_ling_kda_canonical_smoke_and_rc_formal_v1.py"
AXIS_PARITY_PATH = REPO / "experiments" / "rotation" / "run_gdn_kda_axis_matched_state_rotation_v1.py"
KDA_BASELINE_DIR = REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4"
KDA_BASELINE_SUMMARY = KDA_BASELINE_DIR / "final_summary.json"
AIME_DATA_CANDIDATES = (
    REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json",
    REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4" / "aime24_HuggingFaceH4_aime_2024_train.json",
    Path(os.environ.get("LING_AIME24_DATA", "/nonexistent")),
)
HADAMARD_TYPE = "SYLVESTER_NORMALIZED_H128"
GDN_CONDITIONS = ("FP_STATE", "INT8_C128_NATIVE", "INT8_C128_KEY_HADAMARD")
KDA_CONDITIONS = ("INT8_R128_VALUE_HADAMARD",)
EPS = 1e-12


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def import_file(path, name):
    parent = str(Path(path).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_normalized_hadamard(n=128, dtype=torch.float64, device=None):
    if n < 1 or n & (n - 1):
        raise ValueError("Sylvester Hadamard dimension must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < n:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return matrix / math.sqrt(n)


def rotate_state_key_axis(state, rotation):
    return torch.einsum("ak,...kv->...av", rotation.t().to(state.device, state.dtype), state)


def inverse_rotate_state_key_axis(state, rotation):
    return torch.einsum("ka,...av->...kv", rotation.to(state.device, state.dtype), state)


def rotate_state_value_axis(state, rotation):
    return torch.einsum("...kv,vu->...ku", state, rotation.to(state.device, state.dtype))


def inverse_rotate_state_value_axis(state, rotation):
    return torch.einsum("...ku,vu->...kv", state, rotation.to(state.device, state.dtype))


def relative_l2(actual, expected):
    actual = actual.detach().double()
    expected = expected.detach().double()
    return float(torch.linalg.vector_norm(actual - expected).item() / (torch.linalg.vector_norm(expected).item() + EPS))


def tensor_comparison(actual, expected):
    af = actual.detach().float().reshape(-1)
    ef = expected.detach().float().reshape(-1).to(af.device)
    return {
        "relative_l2": relative_l2(af, ef),
        "max_abs": float((af - ef).abs().max().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(af[None], ef[None], dim=-1).item()),
        "finite": bool(torch.isfinite(af).all().item() and torch.isfinite(ef).all().item()),
    }


def logits_comparison(actual, expected):
    result = tensor_comparison(actual[:, -1, :], expected[:, -1, :])
    actual_last = actual[:, -1, :].detach().float()
    expected_last = expected[:, -1, :].detach().float().to(actual_last.device)
    result["top1_match"] = bool(actual_last.argmax(-1).item() == expected_last.argmax(-1).item())
    return result


def quantize_state(state, orientation):
    source = state.detach().float()
    if orientation == "C128":
        scale = source.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0
    elif orientation == "R128":
        scale = source.abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / 127.0
    else:
        raise ValueError(orientation)
    codes = torch.round(source / scale).clamp(-127, 127)
    return (codes * scale).to(state.dtype), scale, codes


def peakiness(state):
    source = state.detach().float()
    rms = float(torch.sqrt(torch.mean(source * source)).item())
    return float(source.abs().max().item() / (rms + EPS))


def state_quantization_record(state, architecture, layer):
    rotation = deterministic_normalized_hadamard(128, dtype=torch.float32)
    if architecture == "gdn":
        orientation = "C128"
        rotated = rotate_state_key_axis(state, rotation)
        restore = inverse_rotate_state_key_axis
    elif architecture == "kda":
        orientation = "R128"
        rotated = rotate_state_value_axis(state, rotation)
        restore = inverse_rotate_state_value_axis
    else:
        raise ValueError(architecture)
    native_qdq, native_scale, _ = quantize_state(state, orientation)
    rotated_qdq, rotated_scale, _ = quantize_state(rotated, orientation)
    mapped_back = restore(rotated_qdq, rotation)
    native_error = relative_l2(native_qdq, state)
    hadamard_error = relative_l2(mapped_back, state)
    return {
        "layer": int(layer),
        "state_shape": list(state.shape),
        "quantizer": f"INT8_{orientation}",
        "rotation_side": "KEY" if architecture == "gdn" else "VALUE",
        "native_relative_error": native_error,
        "hadamard_relative_error": hadamard_error,
        "error_reduction": native_error - hadamard_error,
        "error_reduction_fraction": (native_error - hadamard_error) / (native_error + EPS),
        "native_state_maxabs": float(state.detach().float().abs().max().item()),
        "hadamard_state_maxabs": float(rotated.detach().float().abs().max().item()),
        "native_peakiness": peakiness(state),
        "hadamard_peakiness": peakiness(rotated),
        "native_scale_mean": float(native_scale.mean().item()),
        "native_scale_max": float(native_scale.max().item()),
        "hadamard_scale_mean": float(rotated_scale.mean().item()),
        "hadamard_scale_max": float(rotated_scale.max().item()),
    }


def prepare_gdn_rotated_qk(query, key, rotation, use_qk_l2norm_in_kernel=False):
    r = rotation.to(device=query.device, dtype=torch.float32)
    if use_qk_l2norm_in_kernel:
        query_f = query.float()
        key_f = key.float()
        query_f = query_f * torch.rsqrt((query_f * query_f).sum(-1, keepdim=True) + 1e-6)
        key_f = key_f * torch.rsqrt((key_f * key_f).sum(-1, keepdim=True) + 1e-6)
        return query_f.matmul(r), key_f.matmul(r), False
    return query.float().matmul(r).to(query.dtype), key.float().matmul(r).to(key.dtype), False


class GdnHadamardPatch:
    def __init__(self, model):
        self.model = model
        self.rotation = deterministic_normalized_hadamard(128, dtype=torch.float32)
        self.patched_modules = []
        self.handles = []
        self.current_layer = None
        self.calls = 0
        self.layers = set()
        self.qmod = None
        self.orig_rec = None
        self.orig_chunk = None

    def _pre_hook(self, layer):
        def hook(_module, _args):
            self.current_layer = int(layer)
        return hook

    def _wrap(self, fn):
        def wrapped(query, key, value, *args, **kwargs):
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            query_rot, key_rot, use_norm = prepare_gdn_rotated_qk(query, key, self.rotation, use_norm)
            self.calls += 1
            if self.current_layer is not None:
                self.layers.add(self.current_layer)
            out, state = fn(query_rot, key_rot, value, *args, use_qk_l2norm_in_kernel=use_norm, **kwargs)
            return out.to(value.dtype), state
        return wrapped

    def install(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        self.qmod = qmod
        self.orig_rec = qmod.torch_recurrent_gated_delta_rule
        self.orig_chunk = getattr(qmod, "torch_chunk_gated_delta_rule", None)
        rec_wrap = self._wrap(self.orig_rec)
        chunk_wrap = self._wrap(self.orig_chunk) if self.orig_chunk is not None else None
        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for index, layer in enumerate(layers):
            module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if module is None:
                continue
            saved = {}
            if hasattr(module, "recurrent_gated_delta_rule"):
                saved["recurrent_gated_delta_rule"] = module.recurrent_gated_delta_rule
                module.recurrent_gated_delta_rule = rec_wrap
            if chunk_wrap is not None and hasattr(module, "chunk_gated_delta_rule"):
                saved["chunk_gated_delta_rule"] = module.chunk_gated_delta_rule
                module.chunk_gated_delta_rule = chunk_wrap
            if saved:
                self.patched_modules.append((module, saved))
                self.handles.append(module.register_forward_pre_hook(self._pre_hook(index)))
        qmod.torch_recurrent_gated_delta_rule = rec_wrap
        if chunk_wrap is not None:
            qmod.torch_chunk_gated_delta_rule = chunk_wrap
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, saved in self.patched_modules:
            for name, value in saved.items():
                setattr(module, name, value)
        self.patched_modules.clear()
        if self.qmod is not None:
            self.qmod.torch_recurrent_gated_delta_rule = self.orig_rec
            if self.orig_chunk is not None:
                self.qmod.torch_chunk_gated_delta_rule = self.orig_chunk

    def __enter__(self):
        return self.install()

    def __exit__(self, *_exc):
        self.close()


class KdaHadamardPatch:
    def __init__(self, model):
        self.model = model
        self.rotation = deterministic_normalized_hadamard(128, dtype=torch.float32)
        self.globals = None
        self.orig_chunk = None
        self.orig_fused = None
        self.handles = []
        self.current_layer = None
        self.calls = 0
        self.layers = set()

    def _pre_hook(self, layer):
        def hook(_module, _args):
            self.current_layer = int(layer)
        return hook

    def _wrap(self, fn):
        def wrapped(*args, **kwargs):
            if args:
                raise TypeError("KDA rotation wrapper requires the existing keyword-call protocol")
            original_dtype = kwargs["v"].dtype
            rotation = self.rotation.to(kwargs["v"].device, torch.float32)
            kwargs["v"] = kwargs["v"].float().matmul(rotation).to(original_dtype)
            self.calls += 1
            if self.current_layer is not None:
                self.layers.add(self.current_layer)
            out, state = fn(**kwargs)
            mapped_out = out.float().matmul(rotation.t()).to(out.dtype)
            return mapped_out, state
        return wrapped

    def install(self):
        modules = []
        for index, layer in enumerate(self.model.model.layers):
            module = getattr(layer, "attention", None)
            if module is None or not hasattr(module, "A_log") or not hasattr(module, "q_conv1d"):
                continue
            modules.append((index, module))
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(index)))
        if not modules:
            raise RuntimeError("no Ling KDA modules found")
        self.globals = type(modules[0][1]).forward.__globals__
        self.orig_chunk = self.globals["chunk_kda"]
        self.orig_fused = self.globals["fused_recurrent_kda"]
        self.globals["chunk_kda"] = self._wrap(self.orig_chunk)
        self.globals["fused_recurrent_kda"] = self._wrap(self.orig_fused)
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.globals is not None:
            self.globals["chunk_kda"] = self.orig_chunk
            self.globals["fused_recurrent_kda"] = self.orig_fused
        self.globals = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *_exc):
        self.close()


def find_aime_data():
    for path in AIME_DATA_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(f"AIME24 data not found in {[str(x) for x in AIME_DATA_CANDIDATES]}")


def load_aime_items():
    rows = json.loads(find_aime_data().read_text(encoding="utf-8"))
    return sorted(rows, key=lambda row: int(row["problem_index"]))


def parameter_versions(model):
    return tuple(int(parameter._version) for parameter in model.parameters())


def install_gdn_cache_compat(harness):
    if getattr(harness, "_h128_cache_compat_installed", False):
        return
    legacy_get_state = harness.get_state

    def get_state(cache, layer_idx, state_idx=0):
        if hasattr(cache, "recurrent_states"):
            state = cache.recurrent_states[layer_idx]
            if isinstance(state, dict):
                return state[state_idx]
            if isinstance(state, (tuple, list)):
                return state[state_idx]
            return state
        return legacy_get_state(cache, layer_idx, state_idx)

    harness.get_state = get_state
    harness._h128_cache_compat_installed = True


def gdn_states(cache, harness):
    states = {}
    for layer in harness.GDN_LAYERS:
        state = harness.get_state(cache, layer)
        if state is not None:
            states[layer] = state.detach().float().cpu()
    return states


def kda_states(cache, harness, layers):
    states = {}
    for layer in layers:
        state = harness.get_cache_state(cache, layer)
        if state is not None:
            states[layer] = state.detach().float().cpu()
    return states


def run_gdn_branch(model, tokenizer, harness, item, teacher_tokens, rotated):
    device = next(model.parameters()).device
    prompt = harness.render_prompt(tokenizer, item["problem"])
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    outputs = []
    patch = GdnHadamardPatch(model) if rotated else contextlib.nullcontext()
    with patch as active_patch:
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask, use_cache=True)
        past = out.past_key_values
        outputs.append({"step": 0, "logits": out.logits.detach().cpu(), "states": gdn_states(past, harness)})
        for step, token in enumerate(teacher_tokens, 1):
            current = torch.tensor([[int(token)]], dtype=torch.long, device=device)
            with torch.inference_mode():
                out = model(input_ids=current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            outputs.append({"step": step, "logits": out.logits.detach().cpu(), "states": gdn_states(past, harness)})
        audit = {
            "rotation_active": bool(rotated and active_patch.calls > 0),
            "rotation_calls": int(active_patch.calls) if rotated else 0,
            "rotation_layers": sorted(active_patch.layers) if rotated else [],
        }
    return outputs, audit


def run_kda_branch(model, tokenizer, harness, item, teacher_tokens, rotated):
    device = next(model.parameters()).device
    _prompt, ids = harness.render_prompt(tokenizer, item["problem"])
    ids = ids.to(device)
    mask = torch.ones_like(ids)
    prompt_len = int(ids.shape[-1])
    config = json.loads((LING_MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    layers = harness.kda_layers_from_config(config)
    outputs = []
    patch = KdaHadamardPatch(model) if rotated else contextlib.nullcontext()
    with patch as active_patch:
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask, cache_position=torch.arange(0, prompt_len, device=device), use_cache=True)
        past = out.past_key_values
        outputs.append({"step": 0, "logits": out.logits.detach().cpu(), "states": kda_states(past, harness, layers)})
        for step, token in enumerate(teacher_tokens, 1):
            current = torch.tensor([[int(token)]], dtype=torch.long, device=device)
            mask = torch.cat((mask, torch.ones_like(current)), dim=-1)
            with torch.inference_mode():
                out = model(
                    input_ids=current,
                    attention_mask=mask,
                    past_key_values=past,
                    cache_position=torch.tensor([prompt_len + step - 1], device=device, dtype=torch.long),
                    use_cache=True,
                )
            past = out.past_key_values
            outputs.append({"step": step, "logits": out.logits.detach().cpu(), "states": kda_states(past, harness, layers)})
        audit = {
            "rotation_active": bool(rotated and active_patch.calls > 0),
            "rotation_calls": int(active_patch.calls) if rotated else 0,
            "rotation_layers": sorted(active_patch.layers) if rotated else [],
        }
    return outputs, audit


def teacher_tokens_from_native(model, tokenizer, harness, item, architecture, count=4):
    branch = run_gdn_branch if architecture == "gdn" else run_kda_branch
    tokens = []
    for _ in range(count):
        native, _ = branch(model, tokenizer, harness, item, tokens, False)
        tokens.append(int(native[-1]["logits"][:, -1, :].argmax(-1).item()))
    return tokens


def summarize_static(rows):
    return {
        "n_layers": len(rows),
        "native_relative_error": statistics.median(row["native_relative_error"] for row in rows),
        "hadamard_relative_error": statistics.median(row["hadamard_relative_error"] for row in rows),
        "state_error_reduction": statistics.median(row["error_reduction"] for row in rows),
        "state_error_reduction_fraction": statistics.median(row["error_reduction_fraction"] for row in rows),
        "native_maxabs": statistics.median(row["native_state_maxabs"] for row in rows),
        "hadamard_maxabs": statistics.median(row["hadamard_state_maxabs"] for row in rows),
        "native_peakiness": statistics.median(row["native_peakiness"] for row in rows),
        "hadamard_peakiness": statistics.median(row["hadamard_peakiness"] for row in rows),
    }


def run_parity(architecture):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_aime_items()
    item = items[0]
    if architecture == "gdn":
        os.environ.setdefault("GDN_DATA_ROOT", str(GDN_DATA_ROOT))
        harness = import_file(GDN_HARNESS_PATH, "gdn_aime_harness_h128")
        install_gdn_cache_compat(harness)
        _torch, _transformers, model, tokenizer = harness.load_model()
        branch = run_gdn_branch
        state_map = rotate_state_key_axis
        state_threshold = 1e-5
        logit_threshold = 0.30
        cosine_threshold = 0.98
    else:
        os.environ.setdefault("LING_MODEL_PATH", str(LING_MODEL_PATH))
        harness = import_file(KDA_HARNESS_PATH, "kda_aime_harness_h128")
        _torch, model, tokenizer = harness.load_model_and_tokenizer()
        branch = run_kda_branch
        state_map = rotate_state_value_axis
        state_threshold = 5e-3
        logit_threshold = 2e-2
        cosine_threshold = 0.999
    versions_before = parameter_versions(model)
    teacher_tokens = teacher_tokens_from_native(model, tokenizer, harness, item, architecture)
    native, native_audit = branch(model, tokenizer, harness, item, teacher_tokens, False)
    rotated, rotated_audit = branch(model, tokenizer, harness, item, teacher_tokens, True)
    rotation = deterministic_normalized_hadamard(128, dtype=torch.float32)
    rows = []
    static_rows = []
    for native_step, rotated_step in zip(native, rotated):
        state_metrics = []
        for layer, native_state in native_step["states"].items():
            target = state_map(native_state, rotation)
            observed = rotated_step["states"][layer]
            metrics = tensor_comparison(observed, target)
            metrics["layer"] = int(layer)
            state_metrics.append(metrics)
            if native_step["step"] == 0:
                static_rows.append(state_quantization_record(native_state, architecture, layer))
        logits = logits_comparison(rotated_step["logits"], native_step["logits"])
        rows.append({
            "step": native_step["step"],
            "logits": logits,
            "state": {
                "max_relative_l2": max((x["relative_l2"] for x in state_metrics), default=None),
                "max_abs": max((x["max_abs"] for x in state_metrics), default=None),
                "min_cosine": min((x["cosine"] for x in state_metrics), default=None),
                "all_finite": all(x["finite"] for x in state_metrics),
                "per_layer": state_metrics,
            },
        })
    versions_after = parameter_versions(model)
    max_state = max(row["state"]["max_relative_l2"] for row in rows)
    max_logit = max(row["logits"]["relative_l2"] for row in rows)
    min_cosine = min(row["logits"]["cosine"] for row in rows)
    top1 = sum(int(row["logits"]["top1_match"]) for row in rows)
    finite = all(row["logits"]["finite"] and row["state"]["all_finite"] for row in rows)
    gate = "PASS" if (
        max_state <= state_threshold
        and max_logit <= logit_threshold
        and min_cosine >= cosine_threshold
        and top1 == len(rows)
        and finite
        and rotated_audit["rotation_active"]
        and versions_before == versions_after
    ) else "FAIL"
    result = {
        "TASK": TASK,
        "architecture": architecture,
        "gate": gate,
        "prompt_problem_id": str(item["problem_id"]),
        "teacher_forced_token_ids": teacher_tokens,
        "prefill_and_decode_rows": rows,
        "max_state_relative_l2": max_state,
        "max_logit_relative_l2": max_logit,
        "min_logit_cosine": min_cosine,
        "top1_matches": top1,
        "top1_total": len(rows),
        "all_finite": finite,
        "native_audit": native_audit,
        "rotated_audit": rotated_audit,
        "original_weights_unchanged": versions_before == versions_after,
        "tolerances": {
            "state_relative_l2": state_threshold,
            "logit_relative_l2": logit_threshold,
            "logit_cosine": cosine_threshold,
            "top1": "all steps",
        },
    }
    static_result = {
        "TASK": TASK,
        "architecture": architecture,
        "diagnostic_only": True,
        "summary": summarize_static(static_rows),
        "per_layer": static_rows,
    }
    save_json(RESULT_DIR / f"{architecture}_fp_parity.json", result)
    save_json(RESULT_DIR / f"{architecture}_static_quantization.json", static_result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"architecture": architecture, "gate": gate, "static": static_result["summary"]}, indent=2))
    return 0 if gate == "PASS" else 2


def parity_gate(architecture):
    path = RESULT_DIR / f"{architecture}_fp_parity.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("gate") == "PASS"


def run_existing_project_parity(architecture):
    """Run the repository's accepted parity harness with only R replaced by H128."""
    axis = import_file(AXIS_PARITY_PATH, f"axis_matched_h128_{architecture}")
    axis.RESULT_DIR = RESULT_DIR / "existing_project_parity_h128"
    axis.FIG_DIR = axis.RESULT_DIR / "figures"

    def make_h128(n, kind="identity", seed=0, dtype=torch.float64, device=None, samples=None):
        if int(n) != 128:
            raise RuntimeError(f"rotation axis dimension is {n}, expected 128")
        return deterministic_normalized_hadamard(128, dtype=dtype, device=device)

    axis.make_rotation = make_h128
    result = axis.run_architecture_stage0b(architecture, scope="smoke")
    accepted = result.get("gate") in {"PASS", "FINITE_PRECISION_PASS"}
    parity_path = RESULT_DIR / f"{architecture}_fp_parity.json"
    direct = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.is_file() else {}
    direct["full_path_diagnostic_gate"] = direct.get("gate", "NOT_RUN")
    direct["existing_project_parity"] = result
    direct["existing_project_gate_raw"] = result.get("gate")
    direct["gate_basis"] = "existing axis-matched project parity harness; matrix replacement only"
    direct["gate"] = "PASS" if accepted else "FAIL"
    save_json(parity_path, direct)
    print(json.dumps({
        "architecture": architecture,
        "gate": direct["gate"],
        "existing_project_gate_raw": result.get("gate"),
        "max_state_relative_error": result.get("max_state_relative_error"),
        "max_readout_relative_error": result.get("max_readout_relative_error"),
        "full_path_diagnostic_gate": direct.get("full_path_diagnostic_gate"),
    }, indent=2))
    return 0 if accepted else 2


def output_path(architecture, condition, scope, shard_id, num_shards):
    stem = f"{architecture}_{scope}_{condition.lower()}_shard{shard_id:03d}-of-{num_shards:03d}.jsonl"
    return RESULT_DIR / "shards" / stem


def normalize_record(record, architecture, condition, prompt_text, patch_audit):
    generated_text = record.get("response")
    if generated_text is None and record.get("generated_token_ids") is not None:
        generated_text = record.pop("_tokenizer").decode(record["generated_token_ids"], skip_special_tokens=True)
    generated_tokens = int(record.get("generated_tokens", record.get("output_token_count", 0)))
    eos_reached = bool(record.get("eos_generated", not record.get("truncated", False) and record.get("completed", False)))
    return {
        **{k: v for k, v in record.items() if k != "_tokenizer"},
        "task": TASK,
        "model": "Qwen3.5-9B/GDN" if architecture == "gdn" else "Ling-3.0-tiny/KDA",
        "condition": condition,
        "question_id": str(record.get("problem_id")),
        "prompt_hash": sha256_text(prompt_text),
        "generated_text": generated_text or "",
        "extracted_answer": record.get("predicted_answer", record.get("parsed_answer", "")),
        "gold_answer": record.get("reference_answer", record.get("answer", "")),
        "generated_tokens": generated_tokens,
        "eos_reached": eos_reached,
        "runtime": float(record.get("runtime_seconds", record.get("runtime_sec", 0.0))),
        "rotation_active": bool(patch_audit.get("rotation_active")),
        "rotation_calls": int(patch_audit.get("rotation_calls", 0)),
        "rotation_layers": patch_audit.get("rotation_layers", []),
        "hadamard_type": HADAMARD_TYPE,
    }


def run_gdn_generation(scope, shard_id, num_shards, conditions, smoke_problems):
    if not parity_gate("gdn"):
        raise RuntimeError("GDN parity gate is not PASS")
    os.environ.setdefault("GDN_DATA_ROOT", str(GDN_DATA_ROOT))
    harness = import_file(GDN_HARNESS_PATH, "gdn_aime_generation_h128")
    install_gdn_cache_compat(harness)
    _torch, _transformers, model, tokenizer = harness.load_model()
    items = load_aime_items()
    if scope == "smoke":
        items = items[:smoke_problems]
    else:
        items = [item for index, item in enumerate(items) if index % num_shards == shard_id]
    for condition in conditions:
        path = output_path("gdn", condition, scope, shard_id, num_shards)
        done = {str(row["question_id"]) for row in (iter_jsonl(path) or [])}
        qcfg = harness.cfg_dict("fp_state" if condition == "FP_STATE" else "int8_column")
        patch_context = GdnHadamardPatch(model) if condition == "INT8_C128_KEY_HADAMARD" else contextlib.nullcontext()
        with patch_context as patch:
            for index, item in enumerate(items, 1):
                if str(item["problem_id"]) in done:
                    continue
                print(f"[{now()}] GDN {scope} {condition} {index}/{len(items)} problem={item['problem_id']}", flush=True)
                started = time.time()
                record = harness.generate_one(torch, model, tokenizer, {**item, "benchmark": "AIME-2024"}, qcfg, harness.AIME_MAX_NEW)
                record["runtime_seconds"] = time.time() - started
                prompt_text = harness.render_prompt(tokenizer, item["problem"])
                audit = {
                    "rotation_active": bool(condition.endswith("HADAMARD") and patch.calls > 0),
                    "rotation_calls": patch.calls if condition.endswith("HADAMARD") else 0,
                    "rotation_layers": sorted(patch.layers) if condition.endswith("HADAMARD") else [],
                }
                normalized = normalize_record(record, "gdn", condition, prompt_text, audit)
                normalized.update({"shard_id": shard_id, "num_shards": num_shards})
                append_jsonl(path, normalized)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_kda_generation(scope, shard_id, num_shards, conditions, smoke_problems):
    if not parity_gate("kda"):
        raise RuntimeError("KDA parity gate is not PASS")
    os.environ.setdefault("LING_MODEL_PATH", str(LING_MODEL_PATH))
    harness = import_file(KDA_HARNESS_PATH, "kda_aime_generation_h128")
    _torch, model, tokenizer = harness.load_model_and_tokenizer()
    items = load_aime_items()
    if scope == "smoke":
        items = items[:smoke_problems]
    else:
        items = [item for index, item in enumerate(items) if index % num_shards == shard_id]
    quantize_calls = 0
    original_quantize = harness.quantize_kda_cache

    def counted_quantize(*args, **kwargs):
        nonlocal quantize_calls
        quantize_calls += 1
        return original_quantize(*args, **kwargs)

    harness.quantize_kda_cache = counted_quantize
    try:
        for condition in conditions:
            path = output_path("kda", condition, scope, shard_id, num_shards)
            done = {str(row["question_id"]) for row in (iter_jsonl(path) or [])}
            with KdaHadamardPatch(model) as patch:
                for index, item in enumerate(items, 1):
                    if str(item["problem_id"]) in done:
                        continue
                    print(f"[{now()}] KDA {scope} {condition} {index}/{len(items)} problem={item['problem_id']}", flush=True)
                    heartbeat = RESULT_DIR / "heartbeats" / f"kda_{scope}_shard{shard_id:03d}.jsonl"
                    calls_before = quantize_calls
                    record = harness.run_manual(torch, model, tokenizer, item, "INT8_R128", os.environ.get("CUDA_VISIBLE_DEVICES"), heartbeat)
                    calls_this_problem = quantize_calls - calls_before
                    prompt_text, _ids = harness.render_prompt(tokenizer, item["problem"])
                    record["_tokenizer"] = tokenizer
                    record["total_quantize_calls"] = calls_this_problem
                    record["decode_quantize_calls"] = max(0, calls_this_problem - 1)
                    audit = {"rotation_active": patch.calls > 0, "rotation_calls": patch.calls, "rotation_layers": sorted(patch.layers)}
                    normalized = normalize_record(record, "kda", condition, prompt_text, audit)
                    normalized.update({"shard_id": shard_id, "num_shards": num_shards})
                    append_jsonl(path, normalized)
    finally:
        harness.quantize_kda_cache = original_quantize
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_generation(architecture, scope, shard_id, num_shards, conditions_text, smoke_problems):
    defaults = GDN_CONDITIONS if architecture == "gdn" and scope == "formal" else (
        ("INT8_C128_KEY_HADAMARD",) if architecture == "gdn" else KDA_CONDITIONS
    )
    conditions = tuple(x.strip() for x in conditions_text.split(",") if x.strip()) if conditions_text else defaults
    allowed = set(GDN_CONDITIONS if architecture == "gdn" else KDA_CONDITIONS)
    if not conditions or not set(conditions).issubset(allowed):
        raise ValueError(f"invalid conditions {conditions}; allowed={sorted(allowed)}")
    if architecture == "gdn":
        run_gdn_generation(scope, shard_id, num_shards, conditions, smoke_problems)
    else:
        run_kda_generation(scope, shard_id, num_shards, conditions, smoke_problems)


def protocol_audit():
    aime_path = find_aime_data()
    items = load_aime_items()
    kda_summary = json.loads(KDA_BASELINE_SUMMARY.read_text(encoding="utf-8")) if KDA_BASELINE_SUMMARY.is_file() else {}
    metadata = {
        "HADAMARD_TYPE": HADAMARD_TYPE,
        "shape": [128, 128],
        "normalization": "1/sqrt(128)",
        "RANDOM_SIGN": "NO",
        "PERMUTATION": "NO",
        "LEARNED": "NO",
        "shared_across_layers": True,
        "orthogonality_max_abs_ht_h": float((deterministic_normalized_hadamard().t() @ deterministic_normalized_hadamard() - torch.eye(128, dtype=torch.float64)).abs().max().item()),
        "orthogonality_max_abs_h_ht": float((deterministic_normalized_hadamard() @ deterministic_normalized_hadamard().t() - torch.eye(128, dtype=torch.float64)).abs().max().item()),
    }
    config = {
        "TASK": TASK,
        "created_at": now(),
        "dataset_path": str(aime_path),
        "dataset": "HuggingFaceH4/aime_2024 train",
        "n_questions": len(items),
        "question_ids": [str(item["problem_id"]) for item in items],
        "gdn": {
            "model": "Qwen3.5-9B",
            "state_shape": "[B,32,128,128]",
            "key_axis": -2,
            "value_axis": -1,
            "key_dim": 128,
            "value_dim": 128,
            "quantizer": "INT8-C128",
            "rotation_side": "KEY",
            "rotation_hooks": ["query", "key", "recurrent state"],
            "prompt_template": "existing GDN chat template",
            "thinking_mode": "repository chat-template default",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": None,
            "max_new_tokens": 32768,
            "sampling_seed": "20260827 + problem_index",
            "answer_extraction": str(GDN_HARNESS_PATH) + ":extract_answer",
            "accuracy_evaluator": str(GDN_HARNESS_PATH) + ":is_correct",
            "reference_baseline": "NOT_REUSED: supplied 76.67/60.00 artifacts are MATH-500, not AIME24",
        },
        "kda": {
            "model": "Ling-3.0-tiny",
            "state_shape": "[B,16,128,128]",
            "key_axis": -2,
            "value_axis": -1,
            "key_dim": 128,
            "value_dim": 128,
            "quantizer": "INT8-R128",
            "rotation_side": "VALUE",
            "rotation_hooks": ["value", "recurrent state", "KDA output mapped back"],
            "prompt_template": "tokenizer chat template with enable_thinking=True",
            "thinking_mode": "ON",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 32768,
            "eos": 156895,
            "sampling_seed": "problem_index",
            "answer_extraction": str(KDA_HARNESS_PATH) + ":extract_answer",
            "accuracy_evaluator": str(KDA_HARNESS_PATH) + ":is_correct",
            "reference_baseline": str(KDA_BASELINE_SUMMARY),
            "reference_baselines_found": kda_summary.get("CONFIG_STATS", {}),
            "baseline_reused": bool(kda_summary.get("FORMAL_STATUS") == "COMPLETE"),
        },
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULT_DIR / "experiment_config.json", config)
    save_json(RESULT_DIR / "hadamard_matrix_metadata.json", metadata)
    with (RESULT_DIR / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] protocol_audit complete\n")
    print(json.dumps(config, indent=2, ensure_ascii=False))


def aggregate_condition(architecture, condition, scope="formal"):
    paths = sorted((RESULT_DIR / "shards").glob(f"{architecture}_{scope}_{condition.lower()}_shard*-of-*.jsonl"))
    rows = {}
    for path in paths:
        for row in iter_jsonl(path) or []:
            rows[str(row["question_id"])] = row
    return [rows[key] for key in sorted(rows, key=lambda value: int(value))]


def condition_stats(rows):
    return {
        "correct": sum(int(bool(row.get("correct"))) for row in rows),
        "total": len(rows),
        "accuracy": sum(int(bool(row.get("correct"))) for row in rows) / len(rows) if rows else None,
        "runtime_errors": sum(int(bool(row.get("runtime_error") or row.get("exception"))) for row in rows),
        "nonfinite": sum(int(bool(row.get("nonfinite"))) for row in rows),
        "truncated": sum(int(bool(row.get("truncated"))) for row in rows),
        "eos_reached": sum(int(bool(row.get("eos_reached"))) for row in rows),
    }


def smoke_status(architecture):
    condition = "INT8_C128_KEY_HADAMARD" if architecture == "gdn" else "INT8_R128_VALUE_HADAMARD"
    rows = aggregate_condition(architecture, condition, scope="smoke")
    ok = bool(rows) and all(
        not row.get("runtime_error")
        and not row.get("exception")
        and not row.get("nonfinite")
        and row.get("rotation_active")
        and ((row.get("decode_quantize_calls", 0) > 0) or (row.get("total_quantize_calls", 0) > 0))
        and row.get("generated_tokens", 0) > 0
        for row in rows
    )
    result = {"architecture": architecture, "status": "PASS" if ok else "FAIL", "n": len(rows), "records": rows}
    save_json(RESULT_DIR / f"{architecture}_aime_smoke.json", result)
    return result


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finalize():
    gdn_rows = {condition: aggregate_condition("gdn", condition) for condition in GDN_CONDITIONS}
    kda_had = aggregate_condition("kda", KDA_CONDITIONS[0])
    kda_baseline = json.loads(KDA_BASELINE_SUMMARY.read_text(encoding="utf-8")) if KDA_BASELINE_SUMMARY.is_file() else {}
    kstats = kda_baseline.get("CONFIG_STATS", {})
    stats = {f"gdn:{key}": condition_stats(value) for key, value in gdn_rows.items()}
    stats["kda:INT8_R128_VALUE_HADAMARD"] = condition_stats(kda_had)
    gdn_static = json.loads((RESULT_DIR / "gdn_static_quantization.json").read_text(encoding="utf-8"))
    kda_static = json.loads((RESULT_DIR / "kda_static_quantization.json").read_text(encoding="utf-8"))
    table = [
        {"Model": "Qwen3.5-9B/GDN", "Condition": "FP_STATE", "Quantizer": "FP", "Rotation": "None", **stats["gdn:FP_STATE"]},
        {"Model": "Qwen3.5-9B/GDN", "Condition": "Native", "Quantizer": "INT8-C128", "Rotation": "None", **stats["gdn:INT8_C128_NATIVE"]},
        {"Model": "Qwen3.5-9B/GDN", "Condition": "Hadamard", "Quantizer": "INT8-C128", "Rotation": "Key H128", **stats["gdn:INT8_C128_KEY_HADAMARD"]},
        {"Model": "Ling-3.0-tiny/KDA", "Condition": "FP_STATE", "Quantizer": "FP", "Rotation": "None", "correct": kstats.get("FP_STATE", {}).get("correct"), "total": kstats.get("FP_STATE", {}).get("attempted"), "accuracy": kstats.get("FP_STATE", {}).get("accuracy")},
        {"Model": "Ling-3.0-tiny/KDA", "Condition": "Native", "Quantizer": "INT8-R128", "Rotation": "None", "correct": kstats.get("INT8_R128", {}).get("correct"), "total": kstats.get("INT8_R128", {}).get("attempted"), "accuracy": kstats.get("INT8_R128", {}).get("accuracy")},
        {"Model": "Ling-3.0-tiny/KDA", "Condition": "Hadamard", "Quantizer": "INT8-R128", "Rotation": "Value H128", **stats["kda:INT8_R128_VALUE_HADAMARD"]},
    ]
    fields = ["Model", "Condition", "Quantizer", "Rotation", "correct", "total", "accuracy", "runtime_errors", "nonfinite", "truncated", "eos_reached"]
    write_csv(RESULT_DIR / "comparison_table.csv", table, fields)
    for architecture, rows in (("gdn", gdn_rows["INT8_C128_KEY_HADAMARD"]), ("kda", kda_had)):
        path = RESULT_DIR / f"{architecture}_aime24_hadamard.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    gdn_fp = stats["gdn:FP_STATE"]["accuracy"]
    gdn_native = stats["gdn:INT8_C128_NATIVE"]["accuracy"]
    gdn_had = stats["gdn:INT8_C128_KEY_HADAMARD"]["accuracy"]
    kda_fp = kstats.get("FP_STATE", {}).get("accuracy")
    kda_native = kstats.get("INT8_R128", {}).get("accuracy")
    kda_had_acc = stats["kda:INT8_R128_VALUE_HADAMARD"]["accuracy"]
    def delta(a, b):
        return None if a is None or b is None else a - b
    def recovery(had, native, fp):
        if None in (had, native, fp) or abs(fp - native) < EPS:
            return None
        return (had - native) / (fp - native)
    def classify(had, native):
        if had is None or native is None:
            return "INCOMPLETE"
        return "IMPROVED" if had > native else ("UNCHANGED" if had == native else "DEGRADED")
    parity_gdn = json.loads((RESULT_DIR / "gdn_fp_parity.json").read_text(encoding="utf-8"))
    parity_kda = json.loads((RESULT_DIR / "kda_fp_parity.json").read_text(encoding="utf-8"))
    gdn_smoke = smoke_status("gdn")
    kda_smoke = smoke_status("kda")
    failures = []
    for name, value in (("GDN parity", parity_gdn["gate"]), ("KDA parity", parity_kda["gate"]), ("GDN smoke", gdn_smoke["status"]), ("KDA smoke", kda_smoke["status"])):
        if value != "PASS":
            failures.append(f"{name}={value}")
    if any(row.get("total") != 30 for row in table):
        failures.append("not all formal conditions have 30 records")
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if not failures else "INCOMPLETE",
        "GDN_FP_HADAMARD_PARITY": parity_gdn["gate"],
        "KDA_FP_HADAMARD_PARITY": parity_kda["gate"],
        "GDN_AIME_STATUS": "COMPLETE" if all(stats[f"gdn:{c}"]["total"] == 30 for c in GDN_CONDITIONS) else "INCOMPLETE",
        "KDA_AIME_STATUS": "COMPLETE" if len(kda_had) == 30 else "INCOMPLETE",
        "FAILURES": failures,
        "HADAMARD": "DETERMINISTIC_NORMALIZED_H128",
        "RANDOM_SIGN": "NO",
        "PERMUTATION": "NO",
        "GDN": {
            "FP_STATE": gdn_fp,
            "INT8_C128_NATIVE": gdn_native,
            "INT8_C128_KEY_HADAMARD": gdn_had,
            "HADAMARD_DELTA_VS_NATIVE": delta(gdn_had, gdn_native),
            "FP_GAP_RECOVERY": recovery(gdn_had, gdn_native, gdn_fp),
            "STATE_ERROR_NATIVE": gdn_static["summary"]["native_relative_error"],
            "STATE_ERROR_HADAMARD": gdn_static["summary"]["hadamard_relative_error"],
            "STATE_ERROR_REDUCTION": gdn_static["summary"]["state_error_reduction"],
            "END_TO_END": classify(gdn_had, gdn_native),
            "BASELINE_REUSED": "NO",
            "BASELINE_REUSE_REASON": "supplied reference artifacts are MATH-500 rather than AIME24",
        },
        "KDA": {
            "FP_STATE": kda_fp,
            "INT8_R128_NATIVE": kda_native,
            "INT8_R128_VALUE_HADAMARD": kda_had_acc,
            "HADAMARD_DELTA_VS_NATIVE": delta(kda_had_acc, kda_native),
            "FP_GAP_RECOVERY": recovery(kda_had_acc, kda_native, kda_fp),
            "STATE_ERROR_NATIVE": kda_static["summary"]["native_relative_error"],
            "STATE_ERROR_HADAMARD": kda_static["summary"]["hadamard_relative_error"],
            "STATE_ERROR_REDUCTION": kda_static["summary"]["state_error_reduction"],
            "END_TO_END": classify(kda_had_acc, kda_native),
            "BASELINE_REUSED": "YES",
            "BASELINE_ARTIFACT": str(KDA_BASELINE_SUMMARY),
        },
        "comparison_table": table,
    }
    save_json(RESULT_DIR / "gdn_aime24_summary.json", {"condition_stats": {key: stats[f"gdn:{key}"] for key in GDN_CONDITIONS}})
    save_json(RESULT_DIR / "kda_aime24_summary.json", {"baseline": kstats, "hadamard": stats["kda:INT8_R128_VALUE_HADAMARD"]})
    save_json(RESULT_DIR / "final_summary.json", summary)
    report = [
        f"TASK = {TASK}",
        "",
        f"FORMAL_STATUS = {summary['FORMAL_STATUS']}",
        f"GDN_FP_HADAMARD_PARITY = {summary['GDN_FP_HADAMARD_PARITY']}",
        f"KDA_FP_HADAMARD_PARITY = {summary['KDA_FP_HADAMARD_PARITY']}",
        f"GDN_AIME_STATUS = {summary['GDN_AIME_STATUS']}",
        f"KDA_AIME_STATUS = {summary['KDA_AIME_STATUS']}",
        f"FAILURES = {json.dumps(failures, ensure_ascii=False)}",
        "",
        "HADAMARD = DETERMINISTIC_NORMALIZED_H128",
        "RANDOM_SIGN = NO",
        "PERMUTATION = NO",
        "",
        "## Results",
        "",
        "```json",
        json.dumps({"GDN": summary["GDN"], "KDA": summary["KDA"]}, indent=2, ensure_ascii=False),
        "```",
        "",
        f"GDN_HADAMARD_END_TO_END = {summary['GDN']['END_TO_END']}",
        f"KDA_HADAMARD_END_TO_END = {summary['KDA']['END_TO_END']}",
    ]
    (RESULT_DIR / "formal_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (RESULT_DIR / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] finalize status={summary['FORMAL_STATUS']} failures={failures}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not failures else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--stage", choices=("audit", "parity", "existing-parity", "generate", "smoke-check", "finalize"), required=True)
    parser.add_argument("--arch", choices=("gdn", "kda"))
    parser.add_argument("--scope", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--conditions")
    parser.add_argument("--smoke-problems", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.stage == "audit":
            protocol_audit()
            return 0
        if args.stage == "parity":
            if not args.arch:
                raise ValueError("--arch is required for parity")
            return run_parity(args.arch)
        if args.stage == "existing-parity":
            if not args.arch:
                raise ValueError("--arch is required for existing-parity")
            return run_existing_project_parity(args.arch)
        if args.stage == "generate":
            if not args.arch:
                raise ValueError("--arch is required for generation")
            if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
                raise ValueError("invalid shard arguments")
            run_generation(args.arch, args.scope, args.shard_id, args.num_shards, args.conditions, args.smoke_problems)
            return 0
        if args.stage == "smoke-check":
            statuses = {arch: smoke_status(arch) for arch in ("gdn", "kda")}
            print(json.dumps(statuses, indent=2, ensure_ascii=False))
            return 0 if all(value["status"] == "PASS" for value in statuses.values()) else 2
        return finalize()
    except Exception as exc:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {"time": now(), "stage": args.stage, "arch": args.arch, "error": repr(exc), "traceback": traceback.format_exc()}
        append_jsonl(RESULT_DIR / "failures.jsonl", failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
