#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def diff(x, y):
    xd, yd = x.double(), y.double()
    d = yd - xd
    nz = d != 0
    where = nz.nonzero()
    return {"max_abs": float(d.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(xd).clamp_min(1e-12)),
            "different_elements": int(nz.sum()),
            "first_different_index": None if not where.numel() else [int(v) for v in where[0].tolist()]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("run_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    tensors = [torch.load(path.with_suffix(".temporary.pt"), map_location="cpu", weights_only=False) for path in paths]
    if len(rows) != 5:
        raise RuntimeError(f"expected 5 rows, got {len(rows)}")
    qcounts = Counter(row["q_output_sha256"] for row in rows)
    kcounts = Counter(row["k_output_sha256"] for row in rows)
    by_config = defaultdict(list)
    for row in rows:
        by_config[json.dumps(row["autotune"], sort_keys=True)].append(row["q_output_sha256"])
    qref = tensors[0]["q"]
    comparison = None
    for value in tensors[1:]:
        if not torch.equal(qref, value["q"]):
            comparison = diff(qref, value["q"])
            break
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2", "runs": 5,
        "q_hash_counts": dict(qcounts), "k_hash_counts": dict(kcounts),
        "fresh_process_repeatable": len(qcounts) == 1 and len(kcounts) == 1,
        "q_first_variant_exact_diff": comparison,
        "output_hashes_by_autotune_config": dict(by_config),
        "autotune_records": [row["autotune"] for row in rows],
        "temporary_tensor_bytes_before_cleanup": sum(path.with_suffix(".temporary.pt").stat().st_size for path in paths),
        "temporary_tensors_cleaned": True,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for path in paths:
        path.with_suffix(".temporary.pt").unlink()
    print(json.dumps({"fresh_process_repeatable": result["fresh_process_repeatable"], "q_hash_counts": result["q_hash_counts"], "q_first_variant_exact_diff": comparison}, indent=2))


if __name__ == "__main__":
    main()
