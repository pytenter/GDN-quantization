#!/usr/bin/env python3
"""Compact fast-vs-naive KDA operator repeatability check on frozen doc 0 inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as nnf


HERE = Path(__file__).resolve().parent


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MAIN = import_file(HERE / "run_ling_repeatability.py", "ling_repeatability_reference_import")


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def diff(left: torch.Tensor, right: torch.Tensor) -> dict:
    x, y = left.detach().double(), right.detach().double()
    delta = x - y
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(y).clamp_min(1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    parser.add_argument("--max-memory-gib", type=int, default=22)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--state-checkpoint", required=True)
    parser.add_argument("--functional-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from fla.ops.kda import chunk_kda
    from fla.ops.kda.naive import naive_recurrent_kda

    F, model, tokenizer, layers, device, h, delta_mats, runtime_mats, observed = MAIN.load_context(args)
    documents = sorted(MAIN.R.load_corpus(Path(args.corpus), "HELDOUT"), key=lambda row: int(row["panel_index"]))
    token_ids = tokenizer(documents[0]["raw_text"], add_special_tokens=False).input_ids[:1024]
    conditions = {}
    for method in ("Hadamard", "Dense_State"):
        probe = MAIN.E.PerLayerHistoryProbe.build(F, runtime_mats[method])
        probe.install(model)
        try:
            basis = "rotated" if MAIN.E.rotated(method) else "native"
            _item = MAIN.E.prefill(F, model, token_ids, probe, basis, layers, device, method)
            records = dict(probe.records[method])
        finally:
            probe.close()
        layer = min(records)
        record = records[layer]
        q = record["q"].to(device)
        k = record["k"].to(device)
        v = record["v"].to(device)
        g = record["log_decay"].to(device).float()
        beta = record["beta"].to(device)
        initial = record.get("initial_state")
        initial = None if initial is None else initial.to(device).float()
        normalize = bool(record.get("use_qk_l2norm_in_kernel", False))

        fast, reference = [], []
        with torch.inference_mode():
            for _ in range(3):
                output, state = chunk_kda(
                    q=q, k=k, v=v, g=g, beta=beta, initial_state=initial,
                    output_final_state=True, use_qk_l2norm_in_kernel=normalize,
                    use_gate_in_kernel=False,
                )
                torch.cuda.synchronize(device)
                fast.append((output.detach().clone(), state.detach().clone()))

            q_ref, k_ref = q, k
            if normalize:
                q_ref = nnf.normalize(q.float(), p=2, dim=-1).to(q.dtype)
                k_ref = nnf.normalize(k.float(), p=2, dim=-1).to(k.dtype)
            for _ in range(3):
                output, state = naive_recurrent_kda(
                    q=q_ref, k=k_ref, v=v, g=g, beta=beta,
                    initial_state=initial, output_final_state=True,
                )
                torch.cuda.synchronize(device)
                reference.append((output.detach().clone(), state.detach().clone()))

        conditions[method] = {
            "layer": int(layer),
            "sequence_length": int(q.shape[1]),
            "use_qk_l2norm_in_kernel": normalize,
            "fast_output_hashes": [tensor_hash(item[0]) for item in fast],
            "fast_state_hashes": [tensor_hash(item[1]) for item in fast],
            "reference_output_hashes": [tensor_hash(item[0]) for item in reference],
            "reference_state_hashes": [tensor_hash(item[1]) for item in reference],
            "fast_repeatability": "PASS" if len({tensor_hash(item[0]) + tensor_hash(item[1]) for item in fast}) == 1 else "FAIL",
            "reference_repeatability": "PASS" if len({tensor_hash(item[0]) + tensor_hash(item[1]) for item in reference}) == 1 else "FAIL",
            "fast_vs_reference_output": diff(fast[0][0], reference[0][0]),
            "fast_vs_reference_state": diff(fast[0][1], reference[0][1]),
        }

    result = {
        "task": "LING_RUNTIME_REPEATABILITY_CLOSURE_V1",
        "scope": "one frozen document, first KDA layer, exact captured condition driver tensors",
        "document_id": documents[0]["document_id"],
        "conditions": conditions,
        "REFERENCE_REPEATABILITY": (
            "PASS" if all(row["reference_repeatability"] == "PASS" for row in conditions.values()) else "FAIL"
        ),
        "FAST_OPERATOR_REPEATABILITY": (
            "PASS" if all(row["fast_repeatability"] == "PASS" for row in conditions.values()) else "FAIL"
        ),
        "checkpoint_sha256": observed,
        "durable_tensor_bytes": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
