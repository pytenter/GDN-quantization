#!/usr/bin/env python3
"""Validate and count partial or complete Ling formal shards."""

import json
from collections import Counter
from pathlib import Path

import run_ling_sglang_aime26 as runner
from aime26_common import load_frozen_dataset


REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
OUT = REPO / "artifacts/aime26_v2/official_sampling_81920/ling/formal"


def main():
    dataset = load_frozen_dataset(DATASET)
    expected = {(runner.METHODS[m][0], row["problem_id"], seed) for m in runner.METHODS for row in dataset for seed in runner.SEEDS}
    actual, invalid = [], 0
    for worker in range(runner.FORMAL_WORKERS):
        for method in runner.METHODS:
            path = OUT / "shards" / f"worker{worker}" / f"{method}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if runner.is_valid_record(row, method):
                    actual.append((row["method"], row["problem_id"], int(row["seed"])))
                else:
                    invalid += 1
    counts = Counter(actual)
    report = {"valid": len(actual), "expected": 180, "invalid": invalid,
              "duplicates": sum(count - 1 for count in counts.values()), "missing": len(expected - set(actual)),
              "unexpected": len(set(actual) - expected), "by_configuration": dict(Counter(m for m, _, _ in actual))}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
