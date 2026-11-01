#!/usr/bin/env python3
"""Compare six new-server FP rows with the frozen 4090 formal reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def first_divergence(left: list[int], right: list[int]):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def finish_type(value):
    return value.get("type") if isinstance(value, dict) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    references = {}
    for line in args.reference.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        references[row["problem_id"]] = row
    rows = []
    for problem_id in sorted(references):
        path = args.records / f"{problem_id}_seed1.json"
        if not path.exists():
            rows.append({"problem_id": problem_id, "status": "MISSING"})
            continue
        old = references[problem_id]
        new = json.loads(path.read_text(encoding="utf-8"))
        old_ids = [int(value) for value in old["generated_token_ids"]]
        new_ids = [int(value) for value in new["generated_token_ids"]]
        divergence = first_divergence(old_ids, new_ids)
        row = {
            "problem_id": problem_id,
            "seed": 1,
            "prompt_ids_equal": old.get("input_ids_hash") == new.get("input_ids_hash"),
            "first_token_equal": old_ids[:1] == new_ids[:1],
            "first_16_equal": old_ids[:16] == new_ids[:16],
            "first_128_equal": old_ids[:128] == new_ids[:128],
            "first_1024_equal": old_ids[:1024] == new_ids[:1024],
            "exact_generated_ids": old_ids == new_ids,
            "first_divergence_token": divergence,
            "tokens_4090": len(old_ids),
            "tokens_3090": len(new_ids),
            "finish_4090": finish_type(old.get("finish_reason")),
            "finish_3090": finish_type(new.get("finish_reason")),
            "eos_4090": old.get("eos_seen"),
            "eos_3090": new.get("eos_generated"),
            "v4_correct_4090": old.get("strict_correct"),
            "v4_correct_3090": new.get("v4_correct"),
            "status": "COMPARED",
        }
        rows.append(row)
    complete = all(row["status"] == "COMPARED" for row in rows)
    if not complete:
        overall = "UNKNOWN"
    elif all(row["exact_generated_ids"] and row["prompt_ids_equal"] for row in rows):
        overall = "EXACT"
    elif all(row["first_1024_equal"] and row["prompt_ids_equal"] for row in rows):
        overall = "EARLY_TOKEN_PARITY"
    else:
        overall = "DIVERGED"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runtime_parity_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "runtime_parity_summary.json").write_text(
        json.dumps({"runtime_parity_status": overall, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# LING 256K runtime parity",
        "",
        f"`RUNTIME_PARITY_STATUS = {overall}`",
        "",
        "| Problem | Prompt | 1 | 16 | 128 | 1024 | Exact | First divergence | 4090 tokens | 3090 tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "COMPARED":
            report.append(f"| {row['problem_id']} | missing | | | | | | | | |")
            continue
        report.append(
            "| {problem_id} | {prompt_ids_equal} | {first_token_equal} | {first_16_equal} | "
            "{first_128_equal} | {first_1024_equal} | {exact_generated_ids} | "
            "{first_divergence_token} | {tokens_4090} | {tokens_3090} |".format(**row)
        )
    (args.output_dir / "LING_256K_RUNTIME_PARITY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_parity_status": overall, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
