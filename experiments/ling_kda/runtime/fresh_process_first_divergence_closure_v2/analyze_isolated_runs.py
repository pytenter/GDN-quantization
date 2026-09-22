#!/usr/bin/env python3
"""Summarize isolated chunk_kda fresh-process results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob("run_*.json"))]
    if len(records) != 5:
        raise RuntimeError(f"expected 5 isolated runs, got {len(records)}")
    hashes = [record["same_process_runs"][0]["state"]["sha256"] for record in records]
    output_hashes = [record["same_process_runs"][0]["output"]["sha256"] for record in records]
    config_by_hash = defaultdict(list)
    for record, state_hash in zip(records, hashes):
        configs = {
            name: value for name, value in record["same_process_runs"][0]["autotune"].items()
            if value.get("best_config") or value.get("cache")
        }
        config_by_hash[state_hash].append(configs)
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "label": args.label,
        "fresh_process_runs": len(records),
        "all_same_process_repeatable": all(record["same_process_repeatable"] for record in records),
        "fresh_process_state_repeatable": len(set(hashes)) == 1,
        "fresh_process_output_repeatable": len(set(output_hashes)) == 1,
        "state_hash_counts": dict(Counter(hashes)),
        "output_hash_counts": dict(Counter(output_hashes)),
        "first_call_runtime_seconds": [record["same_process_runs"][0]["runtime_seconds"] for record in records],
        "autotune_configs_by_state_hash": dict(config_by_hash),
        "environment": [record["environment"] for record in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("label", "fresh_process_state_repeatable", "state_hash_counts", "first_call_runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
