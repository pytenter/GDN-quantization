#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob("run_*.json"))]
    if len(rows) != 5:
        raise RuntimeError(f"expected 5 rows, got {len(rows)}")
    order = rows[0]["order"]
    counts = {name: dict(Counter(row["stages"][name]["sha256"] for row in rows)) for name in order}
    first = next((name for name in order if len(counts[name]) > 1), None)
    result = {"task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2", "runs": 5,
              "stage_order": order, "hash_counts": counts, "FIRST_DIVERGENT_CHUNK_KDA_SUBSTAGE": first}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"FIRST_DIVERGENT_CHUNK_KDA_SUBSTAGE": first, "hash_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
