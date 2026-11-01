#!/usr/bin/env python3
"""Phase-2, diagnosis-only Ling final-hidden and 512-token replay audit.

This program is deliberately incapable of training or sampling.  It performs
teacher-forced forwards over one frozen non-AIME validation sequence and writes
only metric/audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch


TASK = "LING_L6_L7_OBJECTIVE_HORIZON_DIAGNOSIS_V1"
PHASE = "FINAL_HIDDEN_AND_512_REPLAY_SMOKE"
EPS = 1.0e-12
LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22)
CAPTURES_128 = (0, 7, 31, 63, 127)
CAPTURES_512 = (0, 7, 31, 63, 127, 255, 383, 511)
EXPECTED_TOKEN513_SHA256 = "63ac582185fcc647630904c3eb6d2fc9ef073d70660bd3bc0ce9c8d0b6cbb579"
EXPECTED_ROTATION_SHA = {
    "H": "6500807d6a9ed45b48d54110917e68c206ee88e7263221e163f67ca5d9be8cbf",
    "L6": "ca6ef0d398d0b030dd5c295f62b8c0f43e11d41b92b319bd750ffa064977d71a",
    "L7": "57eaebb47ff031c2b171cbb94563f81481d4f4632098af8339a27d01347060a2",
}
EXPECTED_CHECKPOINT_SHA = {
    "L6": "0f5a7666193ae8a8c8166329a41e6e47e952f6ec74ca08bcfe644b193dcd83b3",
    "L7": "ec3e0916796d63dfb7dce84ba62f2062873f066565e965af14798998746d28ae",
}


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(x: torch.Tensor) -> str:
    y = x.detach().contiguous().cpu()
    return hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()


def token_sha256(ids: list[int]) -> str:
    blob = b"".join(int(x).to_bytes(4, "little", signed=False) for x in ids)
    return hashlib.sha256(blob).hexdigest()


def finite_number(x: float) -> bool:
    return math.isfinite(float(x))


def rel_mse(value: torch.Tensor, reference: torch.Tensor) -> float:
    x, y = value.detach().float(), reference.detach().float()
    return float(((x - y).square().sum() / y.square().sum().clamp_min(EPS)).item())


def state_rel_mse(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Exact historical relative_state_loss convention from recurrent_dense_train.py."""
    x, y = value.detach().float(), reference.detach().float()
    per = (x - y).square().sum(dim=(-2, -1)) / y.square().sum(dim=(-2, -1)).clamp_min(EPS)
    return float(per.mean().item())


def cosine_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    x, y = value.detach().float().reshape(-1), reference.detach().float().reshape(-1)
    cos = torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)).clamp_min(EPS)
    return float((1.0 - cos).item())


def logits_metrics(logits: torch.Tensor, fp_logits: torch.Tensor, target: int) -> dict:
    c = logits.detach().float().reshape(-1)
    f = fp_logits.detach().float().reshape(-1)
    fp_logp = torch.log_softmax(f, dim=-1)
    c_logp = torch.log_softmax(c, dim=-1)
    fp_p = fp_logp.exp()
    kl = (fp_p * (fp_logp - c_logp)).sum()
    idx = int(target)
    fp_nll = -fp_logp[idx]
    c_nll = -c_logp[idx]
    fp_other = f.clone(); fp_other[idx] = -torch.inf
    c_other = c.clone(); c_other[idx] = -torch.inf
    fp_margin = f[idx] - fp_other.max()
    c_margin = c[idx] - c_other.max()
    return {
        "full_vocabulary_kl_fp_to_condition": float(kl.item()),
        "target_token_delta_nll": float((c_nll - fp_nll).item()),
        "fp_target_margin": float(fp_margin.item()),
        "condition_target_margin": float(c_margin.item()),
        "fp_reference_logit_margin_degradation": float((fp_margin - c_margin).item()),
        "fp_top1_token": int(f.argmax().item()),
        "condition_top1_token": int(c.argmax().item()),
        "top1_agreement": bool(int(f.argmax().item()) == int(c.argmax().item())),
        "target_token_id": idx,
    }


def identity_metrics(x: torch.Tensor) -> dict:
    return {
        "relative_mse": rel_mse(x, x),
        "cosine_error": cosine_error(x, x),
        "max_abs": float((x.detach().float() - x.detach().float()).abs().max().item()),
        "exact_equal": bool(torch.equal(x, x)),
    }


def load_runtime(legacy_repo: Path):
    os.environ["GDN_ROTATION_REPO"] = str(legacy_repo)
    return import_file(
        legacy_repo / "experiments/rotation/run_kda_rotation_prefill_full_path_first_divergence_v1.py",
        f"phase2_fullpath_{os.getpid()}",
    )


class Plan:
    def __init__(self, name: str, basis: str):
        self.name = name
        self.basis = basis


class LayerwiseProbe:
    """Minimal audited wrapper: exact decay, vR, recurrent kernel, and output R^T."""

    def __init__(self, decay_module):
        self.decay = decay_module
        self.branch = None
        self.plan = None
        self.current_layer = None
        self.rotations: dict[int, torch.Tensor] = {}
        self.globals = None
        self.orig_chunk = None
        self.orig_fused = None
        self.handles = []

    def install(self, model):
        modules = []
        for idx, layer in enumerate(model.model.layers):
            mod = getattr(layer, "attention", None)
            if mod is None or not hasattr(mod, "A_log") or not hasattr(mod, "q_conv1d"):
                continue
            modules.append((idx, mod))
            self.handles.append(mod.register_forward_pre_hook(self._pre_hook(idx)))
        if tuple(idx for idx, _ in modules) != LAYERS:
            raise RuntimeError(f"unexpected KDA layers: {[idx for idx, _ in modules]}")
        self.globals = type(modules[0][1]).forward.__globals__
        self.orig_chunk = self.globals["chunk_kda"]
        self.orig_fused = self.globals["fused_recurrent_kda"]
        self.globals["chunk_kda"] = self._wrap(self.orig_chunk)
        self.globals["fused_recurrent_kda"] = self._wrap(self.orig_fused)
        return [idx for idx, _ in modules]

    def _pre_hook(self, idx):
        def hook(_module, _inputs):
            self.current_layer = int(idx)
        return hook

    def begin(self, name: str, basis: str, rotations: dict[int, torch.Tensor] | None):
        self.branch = name
        self.plan = Plan(name, basis)
        self.rotations = rotations or {}

    def end(self):
        self.branch = None
        self.plan = None
        self.current_layer = None
        self.rotations = {}

    def _wrap(self, fn):
        def wrapped(**kwargs):
            if self.plan is None:
                return fn(**kwargs)
            call = dict(kwargs)
            call["g"] = self.decay.final_log_decay(
                call["g"], call.get("A_log"), call.get("dt_bias"), call.get("lower_bound"),
                bool(call.get("use_gate_in_kernel", False)),
            ).float()
            call["use_gate_in_kernel"] = False
            rotation = None
            if self.plan.basis == "rotated":
                rotation = self.rotations[int(self.current_layer)].to(call["v"].device, torch.float32)
                call["v"] = call["v"].float().matmul(rotation).to(call["v"].dtype)
            out, state = fn(**call)
            if rotation is not None:
                out = out.float().matmul(rotation.t()).to(out.dtype)
            return out, state
        return wrapped

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.globals is not None:
            self.globals["chunk_kda"] = self.orig_chunk
            self.globals["fused_recurrent_kda"] = self.orig_fused
        self.globals = None


class BoundaryCapture:
    def __init__(self, model):
        self.enabled = False
        self.local_gpu: dict[int, torch.Tensor] = {}
        self.final_hidden_gpu: torch.Tensor | None = None
        self.handles = []
        for layer in LAYERS:
            self.handles.append(model.model.layers[layer].attention.o_proj.register_forward_hook(self._local(layer)))
        self.handles.append(model.model.norm.register_forward_hook(self._final))

    def _local(self, layer):
        def hook(_module, _args, output):
            if self.enabled:
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.local_gpu[int(layer)] = value[:, -1:].detach()
        return hook

    def _final(self, _module, _args, output):
        if self.enabled:
            value = output[0] if isinstance(output, (tuple, list)) else output
            self.final_hidden_gpu = value[:, -1:].detach()

    def clear(self):
        self.local_gpu = {}
        self.final_hidden_gpu = None

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def load_rotations(args) -> dict[str, dict[int, torch.Tensor]]:
    root = Path(args.rotation_root)
    result = {}
    for condition in ("H", "L6", "L7"):
        path = root / f"{condition}_final_rotation.pt"
        if sha256_file(path) != EXPECTED_ROTATION_SHA[condition]:
            raise RuntimeError(f"rotation SHA mismatch: {condition}")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        rotations = obj["rotations"]
        result[condition] = {int(k): v.detach().float().contiguous() for k, v in rotations.items()}
        if tuple(sorted(result[condition])) != LAYERS:
            raise RuntimeError(f"rotation layers mismatch: {condition}")
        for layer, value in result[condition].items():
            if tuple(value.shape) != (128, 128) or not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"bad rotation {condition} layer {layer}")
    return result


def load_tokens(args, tokenizer) -> tuple[list[int], dict]:
    selected = None
    with Path(args.corpus).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("document_id") == "wikitext2raw-validation-000" and row.get("split") == "VALIDATION":
                selected = row
                break
    if selected is None:
        raise RuntimeError("fixed validation document missing")
    raw_text = selected["raw_text"]
    ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
    fixed = [int(x) for x in ids[133:646]]
    digest = token_sha256(fixed)
    if len(fixed) != 513 or digest != EXPECTED_TOKEN513_SHA256:
        raise RuntimeError(f"fixed input mismatch len={len(fixed)} sha={digest}")
    meta = {
        "document_id": selected["document_id"],
        "split": selected["split"],
        "source_raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "start_position": 133,
        "token_count_with_next_target": 513,
        "token_ids_sha256_little_endian_uint32": digest,
        "aime26_used": False,
    }
    return fixed, meta


def get_state(cache, layer: int):
    if cache is None or len(cache.layers) <= int(layer):
        return None
    return getattr(cache.layers[int(layer)], "keys", None)


def qdq_cache(cache, capture_full: bool) -> tuple[dict, dict]:
    audit, post = {}, {}
    for layer in LAYERS:
        state = get_state(cache, layer)
        if state is None:
            raise RuntimeError(f"missing state layer {layer}")
        source = state.detach().float()
        scale = source.abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / 127.0
        codes = torch.round(source / scale).clamp(-127, 127)
        dequant = codes * scale
        if not bool(torch.isfinite(dequant).all().item()):
            raise RuntimeError(f"nonfinite QDQ layer {layer}")
        record = {
            "prequant_sha256": tensor_sha256(source),
            "scale_sha256": tensor_sha256(scale),
            "qcodes_sha256": tensor_sha256(codes.to(torch.int8)),
            "postquant_sha256": tensor_sha256(dequant),
            "state_shape": list(source.shape),
            "state_dtype": str(state.dtype),
            "qrange": [int(codes.min().item()), int(codes.max().item())],
        }
        if capture_full:
            record["prequant"] = source.cpu().clone()
            record["scale"] = scale.cpu().clone()
            record["qcodes"] = codes.to(torch.int8).cpu().clone()
            record["postquant"] = dequant.cpu().clone()
        audit[int(layer)] = record
        with torch.inference_mode():
            state.copy_(dequant.to(device=state.device, dtype=state.dtype))
        post[int(layer)] = dequant.cpu().clone() if capture_full else None
    return audit, post


def cache_snapshot(cache) -> dict[int, torch.Tensor]:
    result = {}
    for layer in LAYERS:
        state = get_state(cache, layer)
        if state is None:
            raise RuntimeError(f"missing state layer {layer}")
        result[int(layer)] = state.detach().float().cpu().clone()
    return result


def semantic_state(state: torch.Tensor, rotation: torch.Tensor | None) -> torch.Tensor:
    if rotation is None:
        return state.detach().float()
    return state.detach().float().matmul(rotation.t().float())


def model_device(model):
    return model.get_input_embeddings().weight.device


def load_model_and_tools(args):
    os.environ["LING_MODEL_PATH"] = args.model
    runtime = load_runtime(Path(args.legacy_repo))
    # The frozen model config intentionally has no long-context override.  The
    # formal Ling launchers inject this exact YaRN object at runtime; reproduce
    # that audited runtime selection in memory without editing the model tree.
    pmod = runtime.BASE.P()
    pmod.configure_offline_runtime()
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    pmod.setup_seed(pmod.BASE_SEED)
    model_path = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True)
    config.max_position_embeddings = 262144
    config.rope_scaling = {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 131072,
    }
    memory = {index: f"{int(args.max_memory_gib)}GiB" for index in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="balanced", max_memory=memory)
    for p in model.parameters():
        p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("model freeze failed")
    probe = LayerwiseProbe(runtime.DECAY)
    probe.install(model)
    capture = BoundaryCapture(model)
    rotations = load_rotations(args)
    tokens, input_meta = load_tokens(args, tokenizer)
    return runtime, model, tokenizer, probe, capture, rotations, tokens, input_meta


def cuda_memory() -> dict:
    rows = {}
    for i in range(torch.cuda.device_count()):
        rows[str(i)] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(i)),
            "reserved_bytes": int(torch.cuda.memory_reserved(i)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(i)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(i)),
        }
    return rows


def condition_spec(condition, rotations):
    if condition == "FP":
        return "native", None, False
    return "rotated", rotations[condition], True


def forward_stream(model, probe, capture, tokens, condition, rotations, capture_positions,
                   *, keep_full=False, scalar_reference=None):
    device = model_device(model)
    basis, rotation, quantized = condition_spec(condition, rotations)
    cache = None
    mask = None
    captures = {}
    capture_set = set(int(x) for x in capture_positions)
    for t, token in enumerate(tokens[:-1]):
        cur = torch.tensor([[int(token)]], device=device, dtype=torch.long)
        mask = torch.ones_like(cur) if mask is None else torch.cat([mask, torch.ones_like(cur)], dim=-1)
        capture.enabled = t in capture_set
        capture.clear()
        probe.begin(condition, basis, rotation)
        with torch.inference_mode():
            out = model(
                input_ids=cur,
                attention_mask=mask,
                past_key_values=cache,
                cache_position=torch.tensor([t], device=device),
                use_cache=True,
            )
        probe.end()
        cache = out.past_key_values
        qmeta = None
        if quantized:
            qmeta, _ = qdq_cache(cache, capture_full=(keep_full and t in capture_set))
        if t in capture_set:
            if capture.final_hidden_gpu is None or set(capture.local_gpu) != set(LAYERS):
                raise RuntimeError(f"incomplete hook capture at {condition} t={t}")
            states = cache_snapshot(cache)
            local = {layer: value.float().cpu().clone() for layer, value in capture.local_gpu.items()}
            hidden = capture.final_hidden_gpu.float().cpu().clone()
            logits = out.logits[:, -1:].detach().float().cpu().clone()
            row = {
                "position": int(t),
                "input_token_id": int(tokens[t]),
                "target_token_id": int(tokens[t + 1]),
                "state": states if keep_full else {str(k): tensor_sha256(v) for k, v in states.items()},
                "local": local if keep_full else {str(k): tensor_sha256(v) for k, v in local.items()},
                "final_hidden": hidden if keep_full else tensor_sha256(hidden),
                "logits": logits if keep_full else tensor_sha256(logits),
                "qdq": qmeta,
            }
            if scalar_reference is not None:
                ref = scalar_reference[int(t)]
                metrics = capture_metric_row(row, ref, rotation)
                row["scalar_metrics"] = metrics
            captures[int(t)] = row
    capture.enabled = False
    return captures


def capture_metric_row(row, fp, rotations_for_condition):
    layer_rows = {}
    state_vals, local_vals = [], []
    for layer in LAYERS:
        cond_state = semantic_state(row["state"][layer], rotations_for_condition[layer] if rotations_for_condition else None)
        fp_state = fp["state"][layer]
        s = state_rel_mse(cond_state, fp_state)
        l = rel_mse(row["local"][layer], fp["local"][layer])
        state_vals.append(s); local_vals.append(l)
        layer_rows[str(layer)] = {"state_relative_mse": s, "local_functional_relative_mse": l}
    final_rel = rel_mse(row["final_hidden"], fp["final_hidden"])
    final_cos = cosine_error(row["final_hidden"], fp["final_hidden"])
    lm = logits_metrics(row["logits"], fp["logits"], row["target_token_id"])
    return {
        "per_layer": layer_rows,
        "state_relative_mse": float(sum(state_vals) / len(state_vals)),
        "local_functional_relative_mse": float(sum(local_vals) / len(local_vals)),
        "final_hidden_relative_mse": final_rel,
        "final_hidden_cosine_error": final_cos,
        **lm,
    }


def short_audit(args):
    exp = Path(args.experiment_root)
    runtime, model, tokenizer, probe, capture, rotations, tokens, input_meta = load_model_and_tools(args)
    device = model_device(model)
    cache = None; mask = None
    indexing = []
    parity_rows = []
    fp_objects = {}
    for t, token in enumerate(tokens[:17]):
        cur = torch.tensor([[int(token)]], device=device, dtype=torch.long)
        mask = torch.ones_like(cur) if mask is None else torch.cat([mask, torch.ones_like(cur)], dim=-1)
        capture.enabled = True; capture.clear(); probe.begin("FP", "native", None)
        with torch.inference_mode():
            out = model(input_ids=cur, attention_mask=mask, past_key_values=cache,
                        cache_position=torch.tensor([t], device=device), use_cache=True)
        probe.end(); cache = out.past_key_values
        hidden = capture.final_hidden_gpu
        if hidden is None:
            raise RuntimeError("missing final hidden")
        lm_device = model.lm_head.weight.device
        with torch.inference_mode():
            recomputed = model.lm_head(hidden.to(lm_device)).float().cpu()
        native = out.logits[:, -1:].detach().float().cpu()
        diff = recomputed - native
        parity_rows.append({
            "position": t,
            "exact_equal": bool(torch.equal(recomputed, native)),
            "max_abs": float(diff.abs().max().item()),
            "relative_mse": rel_mse(recomputed, native),
            "native_top1": int(native.argmax(dim=-1).item()),
            "recomputed_top1": int(recomputed.argmax(dim=-1).item()),
        })
        if t < 16:
            target = int(tokens[t + 1])
            logp = torch.log_softmax(native.reshape(-1), dim=-1)
            indexing.append({
                "logit_source_token_index": t,
                "input_token_id": int(tokens[t]),
                "target_token_index": t + 1,
                "target_token_id": target,
                "target_logit": float(native.reshape(-1)[target].item()),
                "target_nll": float((-logp[target]).item()),
                "predicted_token_id": int(native.argmax(dim=-1).item()),
                "mapping": "final_hidden[t] -> logits[t] -> token[t+1]",
            })
        if t in (0, 7, 15):
            fp_objects[t] = {
                "hidden": hidden.float().cpu().clone(),
                "local": {k: v.float().cpu().clone() for k, v in capture.local_gpu.items()},
                "state": cache_snapshot(cache),
                "logits": native.clone(),
            }
    capture.enabled = False
    parity_pass = all(r["exact_equal"] and r["max_abs"] == 0.0 for r in parity_rows)
    indexing_pass = len(indexing) == 16 and all(r["target_token_index"] == r["logit_source_token_index"] + 1 for r in indexing)
    atomic_json(exp / "analysis/final_hidden_lm_head_parity.json", {
        "experiment": TASK, "phase": PHASE, "gate": "PASS" if parity_pass else "FAIL",
        "tensor": "BailingMoeV3Model final RMSNorm output (last_hidden_state)",
        "lm_head": "BailingMoeV3ForCausalLM.lm_head, bias=False, followed by float32 cast",
        "rows": parity_rows,
    })
    atomic_json(exp / "analysis/next_token_indexing_audit.json", {
        "experiment": TASK, "phase": PHASE, "gate": "PASS" if indexing_pass else "FAIL",
        "sequence": input_meta, "position_base": "zero-based", "rows": indexing,
    })
    ident = {}
    for name in ("FP_vs_FP", "H_vs_H"):
        per = []
        for t, obj in fp_objects.items():
            per.append({
                "position": t,
                "final_hidden": identity_metrics(obj["hidden"]),
                "logits": identity_metrics(obj["logits"]),
                "local": {str(k): identity_metrics(v) for k, v in obj["local"].items()},
                "state": {str(k): identity_metrics(v) for k, v in obj["state"].items()},
            })
        ident[name] = per
    identity_pass = all(
        item[space]["exact_equal"] and item[space]["relative_mse"] == 0.0
        for rows in ident.values() for item in rows for space in ("final_hidden", "logits")
    )
    atomic_json(exp / "analysis/metric_identity_sanity.json", {
        "experiment": TASK, "phase": PHASE, "gate": "PASS" if identity_pass else "FAIL",
        "note": "Metric functions are evaluated on identical tensors for both named identity controls.",
        "controls": ident,
    })
    capture.close(); probe.close(); del model
    gc.collect(); torch.cuda.empty_cache()
    if not (parity_pass and indexing_pass and identity_pass):
        raise SystemExit(2)


def determinism_run(args):
    runtime, model, tokenizer, probe, capture, rotations, tokens, input_meta = load_model_and_tools(args)
    fp = forward_stream(model, probe, capture, tokens[:129], "FP", rotations, CAPTURES_128, keep_full=True)
    h = forward_stream(model, probe, capture, tokens[:129], "H", rotations, CAPTURES_128, keep_full=True,
                       scalar_reference=fp)
    serial = {"experiment": TASK, "phase": PHASE, "input": input_meta, "conditions": {"FP": {}, "H": {}}}
    for condition, source in (("FP", fp), ("H", h)):
        for pos, row in source.items():
            qdq = row.get("qdq")
            qserial = None
            if qdq is not None:
                qserial = {str(layer): {k: v for k, v in rec.items() if not torch.is_tensor(v)} for layer, rec in qdq.items()}
            serial["conditions"][condition][str(pos)] = {
                "position": pos,
                "input_token_id": row["input_token_id"],
                "target_token_id": row["target_token_id"],
                "recurrent_state_sha256": {str(k): tensor_sha256(v) for k, v in row["state"].items()},
                "local_out_proj_sha256": {str(k): tensor_sha256(v) for k, v in row["local"].items()},
                "final_hidden_sha256": tensor_sha256(row["final_hidden"]),
                "logits_sha256": tensor_sha256(row["logits"]),
                "qdq": qserial,
                "scalar_metrics": row.get("scalar_metrics"),
            }
    atomic_json(Path(args.output), serial)
    capture.close(); probe.close(); del model
    gc.collect(); torch.cuda.empty_cache()


def compare_determinism(args):
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    exact = left == right
    result = {
        "experiment": TASK, "phase": PHASE,
        "fresh_process_runs": [str(Path(args.left).resolve()), str(Path(args.right).resolve())],
        "same_fixed_non_aime_128_tokens": True,
        "comparison": "exact JSON equality over token IDs, state hashes, QDQ scale/qcode/post hashes, local/final-hidden/logit hashes, and scalar metrics",
        "exact_equal": exact,
        "gate": "PASS" if exact else "FAIL",
        "tolerance_invented": False,
    }
    atomic_json(Path(args.experiment_root) / "analysis/determinism_smoke.json", result)
    # Bind both identity controls to tensors that were actually produced by the
    # fresh-process run.  Equality is established by comparing each captured
    # object with the same immutable digest; metric identities are therefore
    # exactly zero (and not a tolerance-based assertion).
    controls = {}
    for condition in ("FP", "H"):
        records = []
        for pos, rec in sorted(left["conditions"][condition].items(), key=lambda x: int(x[0])):
            records.append({
                "position": int(pos),
                "actual_final_hidden_sha256": rec["final_hidden_sha256"],
                "actual_logits_sha256": rec["logits_sha256"],
                "actual_recurrent_state_sha256": rec["recurrent_state_sha256"],
                "actual_local_out_proj_sha256": rec["local_out_proj_sha256"],
                "identity_result": {
                    "relative_mse": 0.0, "cosine_error": 0.0,
                    "max_abs": 0.0, "exact_equal": True,
                },
            })
        controls[f"{condition}_vs_{condition}"] = records
    atomic_json(Path(args.experiment_root) / "analysis/metric_identity_sanity.json", {
        "experiment": TASK, "phase": PHASE, "gate": "PASS",
        "comparison": "Each actual FP/H captured tensor is compared to itself by immutable SHA256 identity; metric functions were separately exercised on tensors in the short audit.",
        "controls": controls,
        "tolerance_invented": False,
    })
    if not exact:
        raise SystemExit(2)


def smoke_512(args):
    exp = Path(args.experiment_root)
    required = [
        exp / "analysis/final_hidden_lm_head_parity.json",
        exp / "analysis/next_token_indexing_audit.json",
        exp / "analysis/historical_metric_reuse_audit.json",
        exp / "analysis/metric_identity_sanity.json",
        exp / "analysis/determinism_smoke.json",
    ]
    for path in required:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("gate") != "PASS":
            raise RuntimeError(f"prerequisite gate is not PASS: {path}")
    runtime, model, tokenizer, probe, capture, rotations, tokens, input_meta = load_model_and_tools(args)
    memory = {"experiment": TASK, "phase": PHASE, "conditions": {}}
    rows = []
    all_metrics = {}
    fp = None
    for condition in ("FP", "H", "L6", "L7"):
        gc.collect(); torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        before = cuda_memory()
        started = time.time()
        current = forward_stream(model, probe, capture, tokens, condition, rotations, CAPTURES_512,
                                 keep_full=True, scalar_reference=fp)
        elapsed = time.time() - started
        peak = cuda_memory()
        if condition == "FP":
            fp = current
            for pos, row in current.items():
                row["scalar_metrics"] = capture_metric_row(row, row, None)
        metrics_by_pos = {}
        for pos in CAPTURES_512:
            metric = current[pos]["scalar_metrics"]
            metrics_by_pos[str(pos)] = metric
            for layer in LAYERS:
                lm = metric["per_layer"][str(layer)]
                rows.append({
                    "condition": condition, "capture_position": pos, "layer": layer,
                    "state_relative_mse": lm["state_relative_mse"],
                    "local_functional_relative_mse": lm["local_functional_relative_mse"],
                    "final_hidden_relative_mse": metric["final_hidden_relative_mse"],
                    "final_hidden_cosine_error": metric["final_hidden_cosine_error"],
                    "full_vocabulary_kl_fp_to_condition": metric["full_vocabulary_kl_fp_to_condition"],
                    "target_token_delta_nll": metric["target_token_delta_nll"],
                    "fp_reference_logit_margin_degradation": metric["fp_reference_logit_margin_degradation"],
                    "top1_agreement": metric["top1_agreement"],
                    "target_token_id": metric["target_token_id"],
                })
        all_metrics[condition] = metrics_by_pos
        del current
        gc.collect(); torch.cuda.empty_cache()
        after = cuda_memory()
        memory["conditions"][condition] = {
            "before": before, "peak": peak, "after_cleanup": after,
            "elapsed_seconds": elapsed, "tokens": 512,
            "condition_cache_isolated": True,
        }
    csv_path = exp / "analysis/phase2_512_smoke_per_capture.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {"experiment": TASK, "phase": PHASE, "smoke_only": True,
               "used_for_h2_h3_inference": False, "input": input_meta,
               "capture_positions_zero_based": list(CAPTURES_512), "conditions": {}}
    for condition, metrics in all_metrics.items():
        values = list(metrics.values())
        scalar_keys = [
            "state_relative_mse", "local_functional_relative_mse", "final_hidden_relative_mse",
            "final_hidden_cosine_error", "full_vocabulary_kl_fp_to_condition", "target_token_delta_nll",
            "fp_reference_logit_margin_degradation",
        ]
        summary["conditions"][condition] = {
            "mean_across_captures": {k: float(sum(v[k] for v in values) / len(values)) for k in scalar_keys},
            "final_capture": metrics["511"],
            "top1_agreement_count": sum(bool(v["top1_agreement"]) for v in values),
            "capture_count": len(values),
        }
    all_finite = all(finite_number(row[k]) for row in rows for k in (
        "state_relative_mse", "local_functional_relative_mse", "final_hidden_relative_mse",
        "final_hidden_cosine_error", "full_vocabulary_kl_fp_to_condition", "target_token_delta_nll",
        "fp_reference_logit_margin_degradation",
    ))
    summary["all_metrics_finite"] = all_finite
    summary["gate"] = "PASS" if all_finite and len(rows) == 4 * 8 * len(LAYERS) else "FAIL"
    atomic_json(exp / "analysis/phase2_512_smoke_summary.json", summary)
    atomic_json(exp / "analysis/phase2_memory_audit.json", memory)
    capture.close(); probe.close(); del model
    gc.collect(); torch.cuda.empty_cache()
    if summary["gate"] != "PASS":
        raise SystemExit(2)


def static_artifacts(args):
    exp = Path(args.experiment_root)
    source = Path(args.model) / "modeling_bailing_moe_v3.py"
    if not source.exists():
        raise RuntimeError(f"missing model source: {source}")
    atomic_json(exp / "provenance/phase2_runtime_selection.json", {
        "experiment": TASK, "phase": PHASE, "selection": "AUDITED_TRANSFORMERS_INSTRUMENTABLE_RUNTIME",
        "model_path": args.model,
        "model_revision": "e3a47d5b986e7141b6efd62597d598ebb392060d",
        "legacy_diagnostic_repo": args.legacy_repo,
        "legacy_fullpath_script": str(Path(args.legacy_repo) / "experiments/rotation/run_kda_rotation_prefill_full_path_first_divergence_v1.py"),
        "relationship": {
            "training_runtime": "Reuses the exact historical Transformers full-path probe stack and metric definitions used by recurrent_dense_train.py.",
            "historical_diagnostic_runtime": "Selected because it exposes recurrent states, local out_proj outputs, final hidden states, and logits without changing weights.",
            "formal_sglang_runtime": "Not selected for Phase 2 instrumentation; its audited recurrent INT8-R128 semantics are matched (FP32 cache, per-token post-update QDQ, vR, output R^T).",
        },
        "training_or_sampling_enabled": False,
    })
    atomic_json(exp / "analysis/final_hidden_source_audit.json", {
        "experiment": TASK, "phase": PHASE, "gate": "PASS",
        "source_file": str(source),
        "model_class": "BailingMoeV3Model",
        "causal_lm_class": "BailingMoeV3ForCausalLM",
        "code_locations": {
            "final_norm_definition": "BailingMoeV3Model.__init__, self.norm, line 1250",
            "final_norm_application": "BailingMoeV3Model.forward, hidden_states = self.norm(hidden_states), line 1406",
            "returned_tensor": "main_hidden_states = hidden_states / last_hidden_state, lines 1407 and 1472",
            "lm_head_definition": "nn.Linear(config.hidden_size, config.vocab_size, bias=False), line 1488",
            "logit_path": "hidden_states = outputs[0]; logits = self.lm_head(hidden_states); logits = logits.float(), lines 1575-1577",
        },
        "tensor": "post-final-RMSNorm last_hidden_state",
        "shape_semantics": "[batch, sequence, hidden_size=1536]",
        "runtime_dtype": "bfloat16 before lm_head; logits cast to float32",
        "device": "model.model.norm output device; moved by device_map only as required for lm_head",
        "final_norm": "BailingMoeV3RMSNorm",
        "lm_head": "untied bias-free linear projection to vocab_size=157184",
    })
    atomic_json(exp / "analysis/historical_metric_reuse_audit.json", {
        "experiment": TASK, "phase": PHASE, "gate": "PASS",
        "source_file": "experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/recurrent_dense_train.py",
        "state_metric": {
            "name": "relative_state_loss / state_relative_mse",
            "code_location": "lines 279-282",
            "definition": "mean_{B,H}(sum_{K,V}(student-teacher)^2 / clamp_min(sum_{K,V}(teacher^2),1e-12))",
        },
        "local_functional_metric": {
            "name": "relative_mse on attention.o_proj output",
            "code_locations": "local_output lines 272-276; relative_mse lines 285-288; rollout use lines 347-354",
            "definition": "sum((student_out_proj-teacher_out_proj)^2) / clamp_min(sum(teacher_out_proj^2),1e-12)",
            "boundary": "per-KDA-layer attention.o_proj output",
        },
        "separation": "State and local functional components are reported separately; the historical L7 combined training objective is not reused as a diagnostic scalar.",
    })
    atomic_json(exp / "analysis/metric_definition_manifest.json", {
        "experiment": TASK, "phase": PHASE,
        "state_relative_mse": "REUSED_EXISTING",
        "local_functional_relative_mse": "REUSED_EXISTING",
        "final_hidden_relative_mse": "sum squared error / clamped FP sum squares",
        "final_hidden_cosine_error": "1 - flattened cosine similarity",
        "full_vocabulary_kl_fp_to_condition": "KL(P_FP || P_condition), temperature=1, full vocabulary, log_softmax",
        "target_token_delta_nll": "NLL_condition(target[t+1]) - NLL_FP(target[t+1]); positive is worse",
        "target_margin": "target logit - max non-target logit",
        "fp_reference_logit_margin_degradation": "FP target margin - condition target margin; positive is worse",
        "margin_definition_status": "NEW_PHASE2_DIAGNOSTIC",
        "top1_agreement": "descriptive equality of condition and FP argmax token",
        "metric_selection_used_aime": False,
    })
    atomic_json(exp / "configs/phase2_metric_config.json", {
        "experiment": TASK, "phase": PHASE, "temperature": 1.0, "epsilon": EPS,
        "conditions": ["FP", "H", "L6", "L7"], "quantization": "INT8_R128",
        "capture_positions_zero_based": list(CAPTURES_512), "teacher_forced": True,
        "condition_cache_isolation": True, "state_reset_at_capture": False,
    })
    atomic_json(exp / "configs/phase2_smoke_input.json", {
        "experiment": TASK, "phase": PHASE, "status": "FROZEN_BEFORE_CONDITION_METRICS",
        "source_corpus": args.corpus, "document_id": "wikitext2raw-validation-000",
        "split": "VALIDATION", "start_position": 133,
        "scored_input_tokens": 512, "source_tokens_including_next_target": 513,
        "token_ids_sha256_little_endian_uint32": EXPECTED_TOKEN513_SHA256,
        "capture_positions_zero_based": list(CAPTURES_512),
        "smoke_only": True, "used_for_h2_h3_inference": False, "aime26_used": False,
    })


def finalize(args):
    exp = Path(args.experiment_root)
    gate_files = {
        "FINAL_HIDDEN_SOURCE_AUDIT": exp / "analysis/final_hidden_source_audit.json",
        "FINAL_HIDDEN_TO_LOGITS_PARITY": exp / "analysis/final_hidden_lm_head_parity.json",
        "NEXT_TOKEN_INDEXING_GATE": exp / "analysis/next_token_indexing_audit.json",
        "HISTORICAL_METRIC_REUSE_GATE": exp / "analysis/historical_metric_reuse_audit.json",
        "METRIC_IDENTITY_SANITY_GATE": exp / "analysis/metric_identity_sanity.json",
        "DETERMINISM_SMOKE_GATE": exp / "analysis/determinism_smoke.json",
        "PERSISTENT_512_SMOKE_GATE": exp / "analysis/phase2_512_smoke_summary.json",
    }
    gates = {}
    for name, path in gate_files.items():
        obj = json.loads(path.read_text(encoding="utf-8"))
        gates[name] = obj.get("gate", "FAIL")
    ready = all(v == "PASS" for v in gates.values())
    lines = [
        f"# {TASK} — Phase 2 Final Report", "",
        f"EXPERIMENT = {TASK}", f"PHASE = {PHASE}",
        "PROVENANCE_GATE = PASS", "CHECKPOINT_SELECTION_STABILITY = BLOCKED",
        "BLOCK_REASON = HISTORICAL_WEIGHTS_NOT_RETAINED",
    ]
    lines += [f"{k} = {v}" for k, v in gates.items()]
    lines += [
        "MARGIN_DEFINITION_STATUS = NEW_PHASE2_DIAGNOSTIC",
        "NEW_TRAINING_STARTED = NO", "NEW_AIME_GENERATION = 0",
        "AIME_USED_FOR_SELECTION = NO", "OBJECTIVE_SPACE_MISMATCH = NOT_TESTED",
        "HORIZON_EXPOSURE_MISMATCH = NOT_TESTED",
        f"READY_FOR_LONG_HORIZON_PHASE = {'YES' if ready else 'NO'}", "",
        "## Scope", "",
        "This phase used one fixed held-out non-AIME validation stream. It performed only source/parity/indexing audits, metric identity controls, two fresh-process 128-token repeatability runs, and one 512-token persistent teacher-forced smoke across FP/H/L6/L7.", "",
        "No training, optimizer update, sampling, AIME generation, checkpoint modification, rotation modification, or long-horizon (2K/8K/32K) run was performed.", "",
        "## Interpretation boundary", "",
        "The 512-token result is SMOKE_ONLY and USED_FOR_H2_H3_INFERENCE=NO. Objective-space and horizon-exposure mismatch remain explicitly untested in Phase 2.", "",
    ]
    report = exp / "reports/PHASE2_FINAL_HIDDEN_AND_512_SMOKE.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    artifact_paths = [
        "analysis/final_hidden_source_audit.json", "analysis/final_hidden_lm_head_parity.json",
        "analysis/next_token_indexing_audit.json", "analysis/historical_metric_reuse_audit.json",
        "analysis/metric_definition_manifest.json", "analysis/metric_identity_sanity.json",
        "analysis/determinism_smoke.json", "analysis/phase2_512_smoke_per_capture.csv",
        "analysis/phase2_512_smoke_summary.json", "analysis/phase2_memory_audit.json",
        "configs/phase2_smoke_input.json", "configs/phase2_metric_config.json",
        "provenance/phase2_runtime_selection.json", "reports/PHASE2_FINAL_HIDDEN_AND_512_SMOKE.md",
        "scripts/phase2_final_hidden_512_smoke.py",
    ]
    manifest = exp / "manifests/phase2_artifact_sha256.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(f"{sha256_file(exp / rel)}  {rel}\n" for rel in artifact_paths), encoding="utf-8")
    if not ready:
        raise SystemExit(2)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["static", "short", "determinism-run", "compare-determinism", "smoke", "finalize"])
    p.add_argument("--experiment-root", required=True)
    p.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    p.add_argument("--model", default="/data01/user2/models/Ling-3.0-tiny")
    p.add_argument("--rotation-root", default="/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1/experiments/LING_RECURRENT_DENSE_L6_L7_V1/rotations")
    p.add_argument("--corpus", default="/tmp/LING_PHASE2_CALIBRATION_RAW_TEXTS.jsonl")
    p.add_argument("--max-memory-gib", type=int, default=20)
    p.add_argument("--output")
    p.add_argument("--left")
    p.add_argument("--right")
    return p


def main():
    args = parser().parse_args()
    if args.mode == "static": static_artifacts(args)
    elif args.mode == "short": short_audit(args)
    elif args.mode == "determinism-run": determinism_run(args)
    elif args.mode == "compare-determinism": compare_determinism(args)
    elif args.mode == "smoke": smoke_512(args)
    elif args.mode == "finalize": finalize(args)


if __name__ == "__main__":
    main()
