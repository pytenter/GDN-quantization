#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def diff(left: torch.Tensor, right: torch.Tensor) -> dict:
    x, y = left.double(), right.double()
    delta = y - x
    changed = delta != 0
    locations = changed.nonzero()
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(x).clamp_min(1e-12)),
        "different_elements": int(changed.sum()),
        "first_different_index": None if not locations.numel() else [int(value) for value in locations[0].tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("run_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    tensors = [torch.load(path.with_suffix(".temporary.pt"), map_location="cpu", weights_only=False)["output"] for path in paths]
    if len(rows) != 5:
        raise RuntimeError(f"expected 5 rows, got {len(rows)}")
    counts = Counter(row["output_sha256"] for row in rows)
    by_config = defaultdict(list)
    for row in rows:
        by_config[json.dumps(row["autotune"], sort_keys=True)].append(row["output_sha256"])
    comparison = None
    for value in tensors[1:]:
        if not torch.equal(tensors[0], value):
            comparison = diff(tensors[0], value)
            break
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "runs": 5,
        "input_hashes_match": len({row["input_sha256"] + row["gate_sha256"] + row["weight_sha256"] for row in rows}) == 1,
        "output_hash_counts": dict(counts),
        "fresh_process_repeatable": len(counts) == 1,
        "same_process_repeatable": all(row["same_process_repeatable"] for row in rows),
        "first_variant_exact_diff": comparison,
        "output_hashes_by_autotune_config": dict(by_config),
        "autotune_records": [row["autotune"] for row in rows],
        "temporary_tensor_bytes_before_cleanup": sum(path.with_suffix(".temporary.pt").stat().st_size for path in paths),
        "temporary_tensors_cleaned": True,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for path in paths:
        path.with_suffix(".temporary.pt").unlink()
    print(json.dumps({key: result[key] for key in ("input_hashes_match", "fresh_process_repeatable", "same_process_repeatable", "output_hash_counts", "first_variant_exact_diff")}, indent=2))


if __name__ == "__main__":
    main()
