#!/usr/bin/env python3
"""Validate and count partial or complete Qwen formal shards."""

import json
from collections import Counter

import run_qwen_aime26_formal as runner


def main():
    dataset = runner.load_frozen_dataset(runner.DATASET)
    expected = {(runner.METHODS[m], row["problem_id"], seed) for m in runner.METHODS for row in dataset for seed in (1, 2)}
    actual, invalid = [], 0
    for worker in range(runner.FORMAL_WORKERS):
        for method in runner.METHODS:
            path = runner.output_path("formal", method, worker)
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
    report = {
        "valid": len(actual), "expected": 180, "invalid": invalid,
        "duplicates": sum(count - 1 for count in counts.values()),
        "missing": len(expected - set(actual)), "unexpected": len(set(actual) - expected),
        "by_configuration": dict(Counter(method for method, _, _ in actual)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
