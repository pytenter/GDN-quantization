#!/usr/bin/env python3
"""Compare canonical 4090/3090 tensor manifests for one final-R INT8 condition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("L6", "L7"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--hardware-assignment", type=Path, required=True)
    parser.add_argument("--fp-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    required_groups = (
        "post_update_recurrent_state",
        "prefill_and_selected_readout",
        "scale_qcode_post_qdq",
        "logits",
    )
    checks = {}
    for group in required_groups:
        left = reference.get("groups", {}).get(group, {})
        right = candidate.get("groups", {}).get(group, {})
        names_equal = set(left) == set(right)
        rows = []
        for name in sorted(set(left) | set(right)):
            left_hash = left.get(name, {}).get("canonical_tensor_sha256")
            right_hash = right.get(name, {}).get("canonical_tensor_sha256")
            rows.append({"path": name, "reference_sha256": left_hash, "candidate_sha256": right_hash, "bitwise_equal": left_hash is not None and left_hash == right_hash})
        passed = names_equal and bool(rows) and all(row["bitwise_equal"] for row in rows)
        checks[group] = {"status": "PASS" if passed else "FAIL", "names_equal": names_equal, "rows": rows}
    rotation_equal = reference.get("rotation_sha256") == candidate.get("rotation_sha256")
    passed = rotation_equal and all(value["status"] == "PASS" for value in checks.values())
    key = f"{args.condition}_4090_3090_INT8_BITWISE_PARITY"
    fp = json.loads(args.fp_audit.read_text(encoding="utf-8"))
    result = {
        "task": "LING_RECURRENT_DENSE_L6_L7_V1",
        "status": "PASS" if passed else "FAIL",
        key: "PASS" if passed else "FAIL",
        "condition": args.condition,
        "runtime_mode": "int8_r128_unified_final_r",
        "rotation_sha256": reference.get("rotation_sha256"),
        "rotation_hash_equal": rotation_equal,
        "reference_host_class": "RTX4090",
        "candidate_host_class": "RTX3090",
        "checks": checks,
        "reference_manifest_sha256": sha256(args.reference),
        "candidate_manifest_sha256": sha256(args.candidate),
        "protocol_amendment_sha256": sha256(args.amendment),
        "hardware_assignment_sha256": sha256(args.hardware_assignment),
        "FP_4090_3090_BITWISE_PARITY": "FAIL",
        "fp_audit_sha256": sha256(args.fp_audit),
        "original_full_cross_host_gate": fp.get("CROSS_HOST_4090_3090_STEP0_PARITY"),
        "scope": "audited model/checkpoint/final-R/runtime/INT8-R128 only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], key: result[key], "checks": {k: v["status"] for k, v in checks.items()}}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
