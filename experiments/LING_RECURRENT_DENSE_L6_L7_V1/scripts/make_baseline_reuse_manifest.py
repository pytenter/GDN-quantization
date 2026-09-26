#!/usr/bin/env python3
"""Freeze the reusable complete L2 last20 baseline and its provenance."""

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
    parser.add_argument("--legacy-audit", required=True)
    parser.add_argument("--old-reuse-audit", required=True)
    parser.add_argument("--parity-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    legacy_path, reuse_path = Path(args.legacy_audit), Path(args.old_reuse_audit)
    legacy, reuse = json.loads(legacy_path.read_text()), json.loads(reuse_path.read_text())
    rows = [
        row
        for row in legacy["rows"]
        if row["condition"] == "L2_INT8_R128_VALUE_HADAMARD_FAST"
        and 11 <= int(row["question_id"].split("_")[-1]) <= 30
    ]
    rows.sort(key=lambda row: row["question_id"])
    checks = {
        "old_reuse_audit_pass": reuse["status"] == "PASS",
        "twenty_unique_last20_rows": len(rows) == 20 and len({row["question_id"] for row in rows}) == 20,
        "all_protocol_valid": all(row["protocol_valid"] for row in rows),
        "all_successful": all(row["successful_sample"] for row in rows),
        "all_seed_1": all(row["seed"] == 1 for row in rows),
        "all_scores_present": all(row["score_present"] for row in rows),
    }
    parity = None
    if args.parity_json and Path(args.parity_json).is_file():
        parity = json.loads(Path(args.parity_json).read_text())
        checks["unified_h_vs_l2_parity"] = parity.get("UNIFIED_H_VS_L2_PARITY") == "PASS"
    passed = all(checks.values()) and parity is not None
    result = {
        "task": "LING_RECURRENT_DENSE_L6_L7_V1",
        "gate": "BASELINE_REUSE_GATE",
        "status": "PASS" if passed else "CONDITIONAL_PENDING_UNIFIED_H_PARITY",
        "decision": "REUSE_COMPLETE_L2_9_OF_20" if passed else "PENDING",
        "legacy_audit_path": str(legacy_path.resolve()),
        "legacy_audit_sha256": sha256(legacy_path),
        "old_reuse_audit_path": str(reuse_path.resolve()),
        "old_reuse_audit_sha256": sha256(reuse_path),
        "parity_path": str(Path(args.parity_json).resolve()) if args.parity_json and parity is not None else None,
        "checks": checks,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
