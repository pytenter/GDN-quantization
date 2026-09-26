#!/usr/bin/env python3
"""Held-out runtime-logit parity gate for canonical H and dense theta=0."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import evaluate_ling_dense_oracle as E


def run(F, model, ids, probe, layers, device):
    item = E.prefill(F, model, ids, probe, "rotated", layers, device, "THETA0_GATE")
    E.quantize_cache(F, item["past"], layers)
    identity = torch.eye(128, dtype=torch.float32)
    logits = {}
    for step in range(1, 129):
        value, _ = F.H.advance(model, probe, item, int(ids[127 + step]), 127 + step,
                               layers, identity, step)
        E.quantize_cache(F, item["past"], layers)
        if step in E.HORIZONS:
            logits[step] = value[:, -1].detach().float().cpu()
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    p.add_argument("--max-memory-gib", type=int, default=22)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    F, model, tokenizer, initial_probe, layers = E.R.load_context(args)
    initial_probe.close()
    device = F.model_input_device(model)
    h = E.R.hadamard(device)
    identity = {layer: torch.eye(128, device=device) for layer in layers}
    zero_bank = E.R.C.PerLayerCayleyRotations(layers).to(device)
    dense_zero = {layer: zero_bank.layer(layer).matrix().detach() for layer in layers}
    canonical_runtime = E.runtime_matrices("Hadamard", identity, layers, h)
    dense_zero_runtime = E.runtime_matrices("Dense_State", dense_zero, layers, h)
    document = E.R.load_corpus(Path(args.corpus), "HELDOUT")[0]
    ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
    probe = E.PerLayerHistoryProbe.build(F, canonical_runtime); probe.install(model)
    try:
        canonical = run(F, model, ids, probe, layers, device)
    finally:
        probe.close()
    probe = E.PerLayerHistoryProbe.build(F, dense_zero_runtime); probe.install(model)
    try:
        theta0 = run(F, model, ids, probe, layers, device)
    finally:
        probe.close()
    rows = {}
    for step in E.HORIZONS:
        delta = theta0[step] - canonical[step]
        rows[str(step)] = {"max_abs": float(delta.abs().max()),
                           "relative_l2": float(delta.norm() / canonical[step].norm().clamp_min(E.R.EPS))}
    maximum = max(row["max_abs"] for row in rows.values())
    result = {"task": E.R.TASK, "model": "Ling-3.0-tiny/KDA",
              "document_id": document["document_id"], "horizons": rows,
              "max_abs": maximum,
              "THETA0_HELDOUT_LOGITS_VS_CANONICAL_HADAMARD": "PASS" if maximum == 0.0 else "FAIL",
              "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
              "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO", "AIME26_used": False}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    E.R.save_json(out / "theta0_heldout_logits_gate.json", result)
    print(result)
    if maximum != 0.0:
        raise RuntimeError("THETA0_HELDOUT_LOGITS_PARITY_FAIL")


if __name__ == "__main__":
    main()
