#!/usr/bin/env python3
"""Summarize frozen-driver causal-convolution update runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("run_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    if not records:
        raise RuntimeError("no isolated short-convolution runs found")
    output_hashes = [record["repeats"][0]["output"]["sha256"] for record in records]
    cache_hashes = [record["repeats"][0]["cache_output"]["sha256"] for record in records]
    tensors = [torch.load(path.with_suffix(".temporary.pt"), map_location="cpu", weights_only=False)[0] for path in paths]
    reference = tensors[0].float()
    comparisons = []
    for path, value in zip(paths[1:], tensors[1:]):
        diff = value.float() - reference
        comparisons.append({
            "run": path.stem,
            "max_abs": float(diff.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference)),
            "different_elements": int((value != tensors[0]).sum()),
        })
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "label": args.label,
        "fresh_process_runs": len(records),
        "all_same_process_repeatable": all(record["same_process_repeatable"] for record in records),
        "fresh_process_output_repeatable": len(set(output_hashes)) == 1,
        "fresh_process_cache_repeatable": len(set(cache_hashes)) == 1,
        "output_hash_counts": dict(Counter(output_hashes)),
        "cache_hash_counts": dict(Counter(cache_hashes)),
        "comparisons_to_run0": comparisons,
        "autotune": [record["autotune"] for record in records],
        "environment": [record["environment"] for record in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
