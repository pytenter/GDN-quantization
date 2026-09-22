#!/usr/bin/env python3
"""One-document 512-token teacher-forced stress for the Qwen dense oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import evaluate_qwen_dense_oracle as E


def reference_all(model, ids, device, maximum):
    logits, states = {}, {}
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids[:128]], device=device), use_cache=True)
        cache = out.past_key_values
        for step in range(1, maximum + 1):
            out = model(input_ids=torch.tensor([[ids[127 + step]]], device=device),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits[step] = out.logits[:, -1].detach().float().cpu()
            states[step] = E.get_state_snapshot(cache)
    return logits, states


def condition_all(model, patch, ids, device, method, mats, h, references, maximum):
    rows = []
    patch.enabled = E.use_rotation(method)
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids[:128]], device=device), use_cache=True)
        cache = out.past_key_values
        E.quantize_cache(cache)
        for step in range(1, maximum + 1):
            out = model(input_ids=torch.tensor([[ids[127 + step]]], device=device),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            E.quantize_cache(cache)
            row = E.logit_metrics(out.logits[:, -1].detach().float().cpu(), references[0][step])
            row["state_relative_l2"] = E.state_rel(
                E.recover_cache_snapshot(cache, method, mats, h), references[1][step])
            rows.append({"step": step, "method": method, **row})
    patch.enabled = False
    return rows


def summarize(rows):
    keys = ("future_kl", "logit_relative_l2", "state_relative_l2")
    result = {
        key: {
            "mean": sum(float(row[key]) for row in rows) / len(rows),
            "max": max(float(row[key]) for row in rows),
            "final": float(rows[-1][key]),
        }
        for key in keys
    }
    result.update({
        "top1_divergence_count": sum(1 - int(row["top1_match"]) for row in rows),
        "top1_match_rate": sum(int(row["top1_match"]) for row in rows) / len(rows),
        "nonfinite_total": sum(int(row["nonfinite"]) for row in rows),
        "tokens": len(rows),
    })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", default="/data/zypan")
    p.add_argument("--corpus", required=True)
    p.add_argument("--state-checkpoint", required=True)
    p.add_argument("--functional-checkpoint", required=True)
    p.add_argument("--persistent-result", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    model, tokenizer, _config, _e2e = E.R.load_model(args)
    device = model.get_input_embeddings().weight.device
    state_bank, _ = E.load_bank(args.state_checkpoint, device)
    func_bank, _ = E.load_bank(args.functional_checkpoint, device)
    banks = {"Dense_State": state_bank, "Dense_Functional": func_bank}
    best = json.loads(Path(args.persistent_result).read_text(encoding="utf-8"))["best_learned"]
    h = E.R.hadamard(device)
    mats = {
        "Hadamard": E.matrices("Hadamard", banks, device),
        best: E.matrices(best, banks, device),
    }
    document = E.R.load_corpus(Path(args.corpus), "HELDOUT")[0]
    ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
    if len(ids) < 640:
        raise RuntimeError(f"512-token stress needs >=640 tokens, got {len(ids)}")

    references = reference_all(model, ids, device, 512)
    patch = E.RuntimePatch(model, mats["Hadamard"], h)
    patch.install()
    try:
        rows = []
        for method in ("Hadamard", best):
            patch.mats = mats[method]
            rows.extend(condition_all(model, patch, ids, device, method, mats[method], h,
                                      references, 512))
            print(f"STRESS {method} 512/512", flush=True)
    finally:
        patch.close()

    by_method = {method: summarize([row for row in rows if row["method"] == method])
                 for method in ("Hadamard", best)}
    result = {
        "task": E.R.TASK,
        "model": "Qwen3.5-9B/GDN",
        "document_id": document["document_id"],
        "prompt_tokens": 128,
        "teacher_forced_tokens": 512,
        "methods": ["Hadamard", best],
        "summary": by_method,
        "rows": rows,
        "best_learned": best,
        "nonfinite_gate": "PASS" if all(v["nonfinite_total"] == 0 for v in by_method.values()) else "FAIL",
        "AIME26_used": False,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    E.R.save_json(out / "stress_512.json", result)
    print(json.dumps({"summary": by_method, "nonfinite_gate": result["nonfinite_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
