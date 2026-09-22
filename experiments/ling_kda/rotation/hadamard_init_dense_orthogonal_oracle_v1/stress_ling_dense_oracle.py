#!/usr/bin/env python3
"""One-document 512-token teacher-forced stress for the Ling dense oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import evaluate_ling_dense_oracle as E


def trajectory_all(F, model, ids, probe, layers, device, method, mats, h,
                   maximum, references=None):
    basis = "rotated" if E.rotated(method) else "native"
    item = E.prefill(F, model, ids, probe, basis, layers, device, method)
    if references is not None:
        E.quantize_cache(F, item["past"], layers)
    identity = torch.eye(128, dtype=torch.float32)
    logits, states, rows = {}, {}, []
    for step in range(1, maximum + 1):
        value, _ = F.H.advance(model, probe, item, int(ids[127 + step]), 127 + step,
                               layers, identity, step)
        if references is not None:
            E.quantize_cache(F, item["past"], layers)
        last = value[:, -1].detach().float().cpu()
        if references is None:
            logits[step] = last
            states[step] = E.snapshot(F, item["past"], layers)
        else:
            row = E.logit_metrics(last, references[0][step])
            row["state_relative_l2"] = E.state_rel(
                E.recovered_snapshot(F, item["past"], layers, method, mats, h),
                references[1][step], layers)
            rows.append({"step": step, "method": method, **row})
    return (logits, states) if references is None else rows


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
    p.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    p.add_argument("--max-memory-gib", type=int, default=22)
    p.add_argument("--corpus", required=True)
    p.add_argument("--state-checkpoint", required=True)
    p.add_argument("--functional-checkpoint", required=True)
    p.add_argument("--persistent-result", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    F, model, tokenizer, initial_probe, layers = E.R.load_context(args)
    initial_probe.close()
    device = F.model_input_device(model)
    h = E.R.hadamard(device)
    state_bank, _ = E.load_bank(args.state_checkpoint, layers, device)
    func_bank, _ = E.load_bank(args.functional_checkpoint, layers, device)
    banks = {"Dense_State": state_bank, "Dense_Functional": func_bank}
    best = json.loads(Path(args.persistent_result).read_text(encoding="utf-8"))["best_learned"]
    mats = {
        "Hadamard": E.matrices("Hadamard", banks, layers, device),
        best: E.matrices(best, banks, layers, device),
    }
    runtime_mats = {method: E.runtime_matrices(method, mats[method], layers, h)
                    for method in mats}
    document = E.R.load_corpus(Path(args.corpus), "HELDOUT")[0]
    ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
    if len(ids) < 640:
        raise RuntimeError(f"512-token stress needs >=640 tokens, got {len(ids)}")

    identity = {layer: torch.eye(128, device=device) for layer in layers}
    reference_probe = E.PerLayerHistoryProbe.build(F, identity)
    reference_probe.install(model)
    try:
        references = trajectory_all(F, model, ids, reference_probe, layers, device,
                                    "Native_INT8", identity, h, 512, None)
    finally:
        reference_probe.close()

    rows = []
    for method in ("Hadamard", best):
        probe = E.PerLayerHistoryProbe.build(F, runtime_mats[method])
        probe.install(model)
        try:
            rows.extend(trajectory_all(F, model, ids, probe, layers, device, method,
                                       mats[method], h, 512, references))
        finally:
            probe.close()
        print(f"STRESS {method} 512/512", flush=True)

    by_method = {method: summarize([row for row in rows if row["method"] == method])
                 for method in ("Hadamard", best)}
    result = {
        "task": E.R.TASK,
        "model": "Ling-3.0-tiny/KDA",
        "document_id": document["document_id"],
        "prompt_tokens": 128,
        "teacher_forced_tokens": 512,
        "methods": ["Hadamard", best],
        "summary": by_method,
        "rows": rows,
        "best_learned": best,
        "nonfinite_gate": "PASS" if all(v["nonfinite_total"] == 0 for v in by_method.values()) else "FAIL",
        "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
        "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO",
        "AIME26_used": False,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    E.R.save_json(out / "stress_512.json", result)
    print(json.dumps({"summary": by_method, "nonfinite_gate": result["nonfinite_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
