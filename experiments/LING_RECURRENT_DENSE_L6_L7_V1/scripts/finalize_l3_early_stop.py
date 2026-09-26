#!/usr/bin/env python3
"""Record retained L3 samples without presenting an interim accuracy estimate."""

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
    parser.add_argument("--l3-output-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.l3_output_dir).glob("aime26_*_seed1.json")):
        payload = json.loads(path.read_text())
        if not payload.get("successful_sample"):
            raise RuntimeError(f"unsuccessful retained L3 sample: {path}")
        rows.append(
            {
                "question_id": payload["question_id"],
                "v4_correct": bool(payload["v4_correct"]),
                "generated_tokens": payload["generated_tokens"],
                "finish_reason": payload["finish_reason"],
                "output_path": str(path.resolve()),
                "output_sha256": sha256(path),
            }
        )
    result = {
        "task": "LING_RECURRENT_DENSE_L6_L7_V1",
        "L3_STATUS": "EARLY_STOPPED_FOR_RESOURCE_REALLOCATION",
        "completed_count": len(rows),
        "question_ids": [row["question_id"] for row in rows],
        "correct_count": sum(row["v4_correct"] for row in rows),
        "rows": rows,
        "is_final_accuracy_estimate": False,
        "accuracy_label_forbidden": True,
        "stop_reason": "Audit proved that L3 and L2 use the same H128 mathematical rotation and differ only by a controlled numerical path; limited resources were reallocated to previously untested recurrent-aware L6/L7.",
        "new_l3_samples_after_in_flight": 0,
        "L4_L5_FORMAL": "DO_NOT_START",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("L3_STATUS", "completed_count", "question_ids", "correct_count")}, indent=2))


if __name__ == "__main__":
    main()
