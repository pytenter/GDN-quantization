#!/usr/bin/env python3
"""Held-out runtime-logit parity gate for canonical H and dense theta=0."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import evaluate_qwen_dense_oracle as E


def run(model, patch, ids, device):
    selected = set(E.HORIZONS)
    logits = {}
    patch.enabled = True
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids[:128]], device=device), use_cache=True)
        cache = out.past_key_values
        E.quantize_cache(cache)
        for step in range(1, 129):
            out = model(input_ids=torch.tensor([[ids[127 + step]]], device=device),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            E.quantize_cache(cache)
            if step in selected:
                logits[step] = out.logits[:, -1].detach().float().cpu()
    patch.enabled = False
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", default="/data/zypan")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    model, tokenizer, _config, _e2e = E.R.load_model(args)
    device = model.get_input_embeddings().weight.device
    h = E.R.hadamard(device)
    identity = {layer: torch.eye(128, device=device) for layer in E.R.GDN_LAYERS}
    zero_bank = E.R.C.PerLayerCayleyRotations(E.R.GDN_LAYERS).to(device)
    dense_zero = {layer: zero_bank.layer(layer).matrix().detach() for layer in E.R.GDN_LAYERS}
    document = E.R.load_corpus(Path(args.corpus), "HELDOUT")[0]
    ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
    patch = E.RuntimePatch(model, identity, h)
    patch.install()
    try:
        canonical = run(model, patch, ids, device)
        patch.mats = dense_zero
        theta0 = run(model, patch, ids, device)
    finally:
        patch.close()
    rows = {}
    for step in E.HORIZONS:
        delta = theta0[step] - canonical[step]
        rows[str(step)] = {
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / canonical[step].norm().clamp_min(E.R.EPS)),
        }
    maximum = max(row["max_abs"] for row in rows.values())
    result = {"task": E.R.TASK, "model": "Qwen3.5-9B/GDN",
              "document_id": document["document_id"], "horizons": rows,
              "max_abs": maximum,
              "THETA0_HELDOUT_LOGITS_VS_CANONICAL_HADAMARD": "PASS" if maximum == 0.0 else "FAIL",
              "AIME26_used": False}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    E.R.save_json(out / "theta0_heldout_logits_gate.json", result)
    print(result)
    if maximum != 0.0:
        raise RuntimeError("THETA0_HELDOUT_LOGITS_PARITY_FAIL")


if __name__ == "__main__":
    main()
