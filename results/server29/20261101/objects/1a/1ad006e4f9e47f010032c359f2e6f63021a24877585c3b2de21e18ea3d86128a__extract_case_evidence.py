#!/usr/bin/env python3
"""Extract reproducible token-window evidence for manual trajectory review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIXED_POSITIONS = (0, 4096, 8192, 16384, 32768, 49152, 65536, 77824, 81664)


def read_rows(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    result = {}
    for path in sorted(root.glob("worker*/*.jsonl")):
        condition = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                result[(condition, str(row["problem_id"]), int(row["seed"]))] = row
    return result


def read_metrics(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[(str(row["condition"]), str(row["problem_id"]), int(row["seed"]))] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True,
                        help="condition:problem_id:seed; may be repeated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left", type=int, default=128)
    parser.add_argument("--right", type=int, default=256)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    rows = read_rows(args.input_root)
    metrics = read_metrics(args.sample_metrics)
    output = []
    for selector in args.case:
        condition, problem_id, seed_text = selector.split(":", 2)
        key = (condition, problem_id, int(seed_text))
        if key not in rows:
            output.append({"selector": selector, "status": "MISSING"})
            continue
        row = rows[key]
        detail = metrics.get(key, {})
        tokens = [int(value) for value in row["generated_token_ids"]]
        positions = set(position for position in FIXED_POSITIONS if position < len(tokens))
        for name in (
            "first_gold_candidate_pos", "last_gold_candidate_pos",
            "first_gold_strong_candidate_pos", "last_gold_strong_candidate_pos",
            "longest_repeat_first_pos", "longest_repeat_second_pos",
        ):
            value = detail.get(name)
            if isinstance(value, int):
                positions.add(value)
        for event in detail.get("candidates", []):
            position = event.get("token_position")
            if isinstance(position, int) and event.get("confidence") == "strong":
                positions.add(position)
        case = {
            "selector": selector,
            "status": "FOUND",
            "condition": condition,
            "problem_id": problem_id,
            "seed": int(seed_text),
            "gold": row.get("gold_answer"),
            "generated_token_count": len(tokens),
            "finish_reason": row.get("finish_reason"),
            "eos_seen": row.get("eos_seen"),
            "windows": [],
        }
        for position in sorted(positions):
            start = max(0, position - args.left)
            end = min(len(tokens), position + args.right)
            case["windows"].append({
                "anchor": position,
                "start": start,
                "end": end,
                "text": tokenizer.decode(tokens[start:end], skip_special_tokens=True),
            })
        output.append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(output), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
