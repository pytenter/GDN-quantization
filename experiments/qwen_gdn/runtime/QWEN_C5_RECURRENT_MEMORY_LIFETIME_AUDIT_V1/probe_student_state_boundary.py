#!/usr/bin/env python3
"""Read-only H32 student-state detach metadata probe for the C5 memory audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch


AUDIT = Path(__file__).resolve().parent
SOURCE = AUDIT.parent / "QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py"
spec = importlib.util.spec_from_file_location("frozen_recurrent", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen recurrent source")
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)


def digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_rows(cache):
    rows = []
    for layer_id in BASE.GDN_LAYERS:
        layer = cache.layers[layer_id]
        states = layer.recurrent_states
        for state_idx, tensor in states.items():
            rows.append({
                "layer_id": layer_id,
                "state_idx": state_idx,
                "requires_grad": tensor.requires_grad,
                "grad_fn": None if tensor.grad_fn is None else type(tensor.grad_fn).__name__,
                "is_leaf": tensor.is_leaf,
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": digest(tensor),
            })
    return rows


def main():
    torch.manual_seed(BASE.TRAIN_SEED)
    model, tokenizer, bank, device = BASE.load_model_bank()
    ids, _positions = BASE.tokenize(tokenizer, BASE.corpus_rows("TRAIN")[0])
    h = BASE.hadamard(device)
    cache = None
    with BASE.RecurrentExperimentPatch(model, bank, h) as patch:
        for token_id in ids[:32]:
            cache = BASE.forward_student(model, patch, token_id, cache, False)
        before = state_rows(cache)
        BASE.detach_cache(cache)
        after = state_rows(cache)
        if len(before) != len(after):
            raise RuntimeError("state count changed across detach")
        by_key = {(row["layer_id"], row["state_idx"]): row for row in after}
        preserved = all(row["sha256"] == by_key[(row["layer_id"], row["state_idx"])]["sha256"] for row in before)
        no_graph = all(not row["requires_grad"] and row["grad_fn"] is None for row in after)
        result = {
            "horizon": 32,
            "state_count": len(after),
            "before_detach": before,
            "after_detach": after,
            "numerical_value_preserved": preserved,
            "autograd_history_detached": no_graph,
            "status": "PASS" if preserved and no_graph else "FAIL",
        }
        path = AUDIT / "analysis/student_state_boundary_probe.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("horizon", "state_count", "numerical_value_preserved", "autograd_history_detached", "status")}, indent=2), flush=True)
        if result["status"] != "PASS":
            raise RuntimeError("state boundary probe failed")


if __name__ == "__main__":
    main()
