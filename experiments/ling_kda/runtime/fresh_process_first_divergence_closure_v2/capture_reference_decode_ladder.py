#!/usr/bin/env python3
"""Capture compact hashes through the first eight Native reference decode steps."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "repeatability_closure_v1" / "run_ling_repeatability.py"


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNTIME = import_file(V1, "ling_repeatability_for_reference_decode_ladder_v2")


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_fields(record: dict) -> dict:
    return {
        name: tensor_hash(value)
        for name, value in sorted(record.items())
        if torch.is_tensor(value)
    }


def cache_hashes(F, cache, layers) -> dict:
    stack = F.BASE.cache_stack(cache, layers)
    return {str(layer): tensor_hash(stack[layer]) for layer in layers}


def autotune_rows(kernel) -> list[dict]:
    """Return compact, JSON-safe autotuner selections without changing them."""
    current = kernel
    rows = []
    for depth in range(6):
        cache = getattr(current, "cache", None)
        if isinstance(cache, dict):
            rows.append({
                "depth": depth,
                "type": f"{type(current).__module__}.{type(current).__name__}",
                "cache": [{
                    "key": str(key),
                    "kwargs": dict(config.kwargs),
                    "num_warps": config.num_warps,
                    "num_stages": config.num_stages,
                    "num_ctas": getattr(config, "num_ctas", None),
                } for key, config in cache.items()],
            })
        current = getattr(current, "fn", None)
        if current is None:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    parser.add_argument("--max-memory-gib", type=int, default=22)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--state-checkpoint", required=True)
    parser.add_argument("--functional-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--shortconv-driver", type=Path)
    args = parser.parse_args()

    seed_record = RUNTIME.seed_all(20260922)
    RUNTIME.apply_diagnostic_gated_rmsnorm_config()
    F, model, tokenizer, layers, device, h, delta_mats, runtime_mats, _observed = RUNTIME.load_context(args)
    documents = sorted(RUNTIME.R.load_corpus(Path(args.corpus), "HELDOUT"), key=lambda row: int(row["panel_index"]))
    ids = tokenizer(documents[0]["raw_text"], add_special_tokens=False).input_ids[:1024]
    focus_layer = 9
    hook_hashes: dict[int, dict[str, str]] = {}
    current_step = {"value": 0}
    shortconv_driver = {}
    handles = []

    def post(name):
        def hook(_module, _inputs, output):
            if current_step["value"]:
                if isinstance(output, (tuple, list)):
                    output = output[0] if output else None
                if torch.is_tensor(output):
                    hook_hashes.setdefault(current_step["value"], {})[name] = tensor_hash(output)
        return hook

    def pre(name):
        def hook(_module, inputs):
            if current_step["value"] and inputs and torch.is_tensor(inputs[0]):
                hook_hashes.setdefault(current_step["value"], {})[name] = tensor_hash(inputs[0])
        return hook

    layer8 = model.model.layers[focus_layer - 1]
    layer9 = model.model.layers[focus_layer]
    handles.append(layer8.register_forward_hook(post("layer8_output_hidden")))
    handles.append(layer9.register_forward_pre_hook(pre("layer9_input_hidden")))
    handles.append(layer9.input_layernorm.register_forward_hook(post("layer9_pre_norm_output")))
    handles.append(layer9.attention.q_proj.register_forward_hook(post("layer9_q_projection_raw")))
    handles.append(layer9.attention.q_conv1d.register_forward_hook(post("layer9_q_short_conv_output")))

    def q_short_conv_pre(module, inputs, kwargs):
        if not current_step["value"]:
            return
        row = hook_hashes.setdefault(current_step["value"], {})
        x = kwargs.get("x", inputs[0] if inputs else None)
        cache = kwargs.get("cache")
        if torch.is_tensor(x):
            row["layer9_q_short_conv_input"] = tensor_hash(x)
        if torch.is_tensor(cache):
            # Clone before the update kernel mutates this cache in-place.
            row["layer9_q_short_conv_cache_input"] = tensor_hash(cache.clone())
        if args.shortconv_driver is not None and current_step["value"] == 2 and not shortconv_driver:
            shortconv_driver.update({
                "x": x.detach().cpu().clone(),
                "cache": cache.detach().cpu().clone(),
                "weight": module.weight.detach().cpu().clone().reshape(module.weight.shape[0], -1),
                "bias": None if module.bias is None else module.bias.detach().cpu().clone(),
                "residual": None,
                "activation": module.activation,
                "backend": module.backend,
            })

    handles.append(layer9.attention.q_conv1d.register_forward_pre_hook(q_short_conv_pre, with_kwargs=True))
    probe = RUNTIME.E.PerLayerHistoryProbe.build(F, runtime_mats["Native_INT8"])
    probe.install(model)
    identity = torch.eye(128, dtype=torch.float32)
    rows = []
    try:
        item = RUNTIME.E.prefill(F, model, ids, probe, "native", layers, device, "Native_INT8")
        prefill = cache_hashes(F, item["past"], layers)
        for step in range(1, args.steps + 1):
            current_step["value"] = step
            value, records = F.H.advance(
                model, probe, item, int(ids[127 + step]), 127 + step,
                layers, identity, step,
            )
            rows.append({
                "step": step,
                "logits_sha256": tensor_hash(value[:, -1]),
                "cache_layer_sha256": cache_hashes(F, item["past"], layers),
                "record_tensor_sha256": {
                    str(layer): tensor_fields(record)
                    for layer, record in sorted(records.items())
                },
                "hook_tensor_sha256": hook_hashes.get(step, {}),
            })
    finally:
        for handle in handles:
            handle.remove()
        probe.close()
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "scope": "Native reference first decode ladder",
        "run_id": args.run_id,
        "pid": os.getpid(),
        "environment": {key: os.environ.get(key) for key in ("FLA_CACHE_MODE", "FLA_CONFIG_DIR", "LING_GATED_RMSNORM_FIXED_CONFIG")},
        "seed_record": seed_record,
        "document_id": documents[0]["document_id"],
        "layers": list(layers),
        "focus_layer": focus_layer,
        "prefill_cache_layer_sha256": prefill,
        "shortconv_autotune": autotune_rows(
            __import__("fla.modules.conv.triton.kernels", fromlist=["causal_conv1d_update_kernel"])
            .causal_conv1d_update_kernel
        ),
        "steps": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.shortconv_driver is not None:
        if not shortconv_driver:
            raise RuntimeError("step-2 layer-9 q short-convolution driver was not captured")
        args.shortconv_driver.parent.mkdir(parents=True, exist_ok=True)
        torch.save(shortconv_driver, args.shortconv_driver)
    print(json.dumps({"run_id": args.run_id, "steps": len(rows), "last_logit": rows[-1]["logits_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
