#!/usr/bin/env python3
"""Require bitwise Step0 parity before mixing RTX 4090 and RTX 3090 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_tensor_files(left: Path, right: Path, pattern: str) -> dict:
    left_files = {path.relative_to(left): path for path in left.rglob(pattern)}
    right_files = {path.relative_to(right): path for path in right.rglob(pattern)}
    rows = []
    for name in sorted(set(left_files) & set(right_files)):
        a = torch.load(left_files[name], map_location="cpu", weights_only=True)
        b = torch.load(right_files[name], map_location="cpu", weights_only=True)
        if isinstance(a, dict):
            tensor_keys = [key for key in sorted(set(a) & set(b)) if torch.is_tensor(a[key]) and torch.is_tensor(b[key])]
            equal = set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in tensor_keys)
            max_abs = max([float((a[key].float() - b[key].float()).abs().max()) for key in tensor_keys] or [0.0])
        else:
            equal = torch.equal(a, b)
            max_abs = float((a.float() - b.float()).abs().max())
        rows.append({"path": str(name), "bitwise_equal": bool(equal), "max_abs": max_abs})
    names_equal = set(left_files) == set(right_files)
    passed = names_equal and bool(rows) and all(row["bitwise_equal"] for row in rows)
    return {"status": "PASS" if passed else "FAIL", "names_equal": names_equal, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--h-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = {
        "fp_state": compare_tensor_files(args.root / "reference_fp", args.root / "candidate_fp", "state_*.pt"),
        "fp_prefill_endpoint": compare_tensor_files(args.root / "reference_fp", args.root / "candidate_fp", "basis_*.pt"),
        "fp_logits": compare_tensor_files(args.root / "reference_fp", args.root / "candidate_fp", "logits_*.pt"),
        "int8_state": compare_tensor_files(args.root / "reference_int8", args.root / "candidate_int8", "state_*.pt"),
        "int8_prefill_endpoint": compare_tensor_files(args.root / "reference_int8", args.root / "candidate_int8", "basis_*.pt"),
        "int8_scale_qcode_post_qdq": compare_tensor_files(args.root / "reference_int8", args.root / "candidate_int8", "runtime_*.pt"),
        "int8_logits": compare_tensor_files(args.root / "reference_int8", args.root / "candidate_int8", "logits_*.pt"),
    }
    passed = all(row["status"] == "PASS" for row in checks.values())
    result = {
        "task": "LING_RECURRENT_DENSE_L6_L7_V1",
        "status": "PASS" if passed else "FAIL",
        "CROSS_HOST_4090_3090_STEP0_PARITY": "PASS" if passed else "FAIL",
        "classification": "BITWISE_EXACT" if passed else "UNRESOLVED_CROSS_HARDWARE_MISMATCH",
        "reference_host_class": "RTX4090",
        "candidate_host_class": "RTX3090",
        "h_final_sha256": sha256(args.h_final),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": {key: value["status"] for key, value in checks.items()}}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
