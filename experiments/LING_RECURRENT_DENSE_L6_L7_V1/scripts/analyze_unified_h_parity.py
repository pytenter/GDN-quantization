#!/usr/bin/env python3
"""Exact non-AIME parity analysis for old L2 and the unified-H final-R path."""

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
    names_equal = set(left_files) == set(right_files)
    rows = []
    for name in sorted(set(left_files) & set(right_files)):
        a = torch.load(left_files[name], map_location="cpu", weights_only=True)
        b = torch.load(right_files[name], map_location="cpu", weights_only=True)
        if isinstance(a, dict):
            keys = sorted(set(a) & set(b))
            tensor_keys = [key for key in keys if torch.is_tensor(a[key]) and torch.is_tensor(b[key])]
            equal = set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in tensor_keys)
            max_abs = max(
                [float((a[key].float() - b[key].float()).abs().max()) for key in tensor_keys] or [0.0]
            )
        else:
            equal = torch.equal(a, b)
            max_abs = float((a.float() - b.float()).abs().max())
        rows.append({"path": str(name), "bitwise_equal": bool(equal), "max_abs": max_abs})
    passed = names_equal and bool(rows) and all(row["bitwise_equal"] for row in rows)
    return {"status": "PASS" if passed else "FAIL", "names_equal": names_equal, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--h-final", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    payload = torch.load(args.h_final, map_location="cpu", weights_only=False)
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h = h / (128 ** 0.5)
    matrix_rows = [
        {"layer_id": layer, "bitwise_equal": torch.equal(payload["rotations"][layer], h)}
        for layer in payload["layer_ids"]
    ]
    matrix_pass = all(row["bitwise_equal"] for row in matrix_rows)
    fp_old, fp_new = root / "old_l2_fp", root / "unified_h_fp"
    q_old, q_new = root / "old_l2_int8", root / "unified_h_int8"
    checks = {
        "matrix_equality": {"status": "PASS" if matrix_pass else "FAIL", "rows": matrix_rows},
        "fp_state_prefill_first_decode_and_short_decode": compare_tensor_files(fp_old, fp_new, "state_*.pt"),
        "fp_prefill_endpoint": compare_tensor_files(fp_old, fp_new, "basis_*.pt"),
        "fp_logits": compare_tensor_files(fp_old, fp_new, "logits_*.pt"),
        "int8_state_prefill_first_decode_and_short_decode": compare_tensor_files(q_old, q_new, "state_*.pt"),
        "int8_prefill_endpoint": compare_tensor_files(q_old, q_new, "basis_*.pt"),
        "int8_scale_qcode_post_qdq": compare_tensor_files(q_old, q_new, "runtime_*.pt"),
        "int8_logits": compare_tensor_files(q_old, q_new, "logits_*.pt"),
    }
    passed = all(value["status"] == "PASS" for value in checks.values())
    result = {
        "task": "LING_RECURRENT_DENSE_L6_L7_V1",
        "UNIFIED_H_VS_L2_PARITY": "PASS" if passed else "FAIL",
        "status": "PASS" if passed else "FAIL",
        "classification": "BITWISE_EXACT" if passed else "UNRESOLVED_MISMATCH",
        "non_aime_prompts": 3,
        "teacher_forced_decode_steps": 8,
        "h_final_path": str(Path(args.h_final).resolve()),
        "h_final_sha256": sha256(Path(args.h_final)),
        "checks": checks,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": {k: v["status"] for k, v in checks.items()}}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
