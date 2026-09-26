#!/usr/bin/env python3
"""Observation-only dtype audit for Qwen3.5 GDN FP_STATE."""

import inspect
import json
import sys
from pathlib import Path

import torch


REPO = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1")
sys.path.insert(0, str(REPO / "experiments/aime26"))

import run_qwen_aime26_formal as runner  # noqa: E402


def tensor_meta(value):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return {"python_type": type(value).__name__}
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "device": str(value.device),
        "data_ptr": int(value.data_ptr()),
    }


def main():
    out_path = REPO / "artifacts/aime26_v1/formal/qwen/dtype_audit_fp_state.json"
    rows = runner.load_frozen_dataset(runner.DATASET)
    model, tokenizer = runner.load_model_and_tokenizer()
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

    events = []
    current_layer = {"value": None}
    handles = []
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    for index, layer in enumerate(layers):
        module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        if module is not None:
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, _args, layer_index=index: current_layer.__setitem__("value", layer_index)
                )
            )

    originals = {}

    def make_wrapper(phase, function):
        signature = inspect.signature(function)

        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            inputs = {
                name: tensor_meta(bound.arguments.get(name))
                for name in ("query", "key", "value", "g", "beta", "initial_state")
            }
            result = function(*args, **kwargs)
            core, state = result
            events.append(
                {
                    "event": f"{phase}_kernel_call",
                    "layer": current_layer["value"],
                    "inputs": inputs,
                    "core_output": tensor_meta(core),
                    "returned_state": tensor_meta(state),
                }
            )
            return result

        return wrapped

    for name, phase in (
        ("torch_chunk_gated_delta_rule", "prefill"),
        ("torch_recurrent_gated_delta_rule", "first_decode"),
    ):
        original = getattr(qmod, name)
        originals[name] = original
        setattr(qmod, name, make_wrapper(phase, original))

    messages, input_ids = runner.render_input(tokenizer, rows[0]["problem"])
    input_ids = input_ids.to(next(model.parameters()).device)
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.inference_mode():
            prefill = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            cache = prefill.past_key_values
            prefill_cache = {
                str(layer): tensor_meta(runner.get_state(cache, layer)) for layer in runner.GDN_LAYERS
            }
            first_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded = model(input_ids=first_token, past_key_values=cache, use_cache=True)
            decode_cache = {
                str(layer): tensor_meta(runner.get_state(decoded.past_key_values, layer))
                for layer in runner.GDN_LAYERS
            }
    finally:
        for name, original in originals.items():
            setattr(qmod, name, original)
        for handle in handles:
            handle.remove()

    implementation = {}
    for name, original in originals.items():
        implementation[name] = {
            "module": getattr(original, "__module__", None),
            "qualname": getattr(original, "__qualname__", None),
            "source_file": inspect.getsourcefile(original),
            "signature": str(inspect.signature(original)),
        }
    payload = {
        "task": "FP_STATE_ACTUAL_DTYPE_AUDIT",
        "model": "Qwen3.5-9B",
        "architecture": "GDN",
        "quantization": "disabled",
        "rotation": "disabled",
        "problem_id": rows[0]["problem_id"],
        "messages": messages,
        "input_tokens": int(input_ids.shape[-1]),
        "first_decode_token": int(first_token.item()),
        "kernel_implementation": implementation,
        "events": events,
        "prefill_cache": prefill_cache,
        "first_decode_cache": decode_cache,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
