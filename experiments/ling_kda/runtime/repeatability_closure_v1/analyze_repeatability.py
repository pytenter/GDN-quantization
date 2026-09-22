#!/usr/bin/env python3
"""Aggregate compact Ling KDA repeatability records without tensor traces."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = q * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def repeat_summary(rows: list[dict]) -> dict:
    values = [float(row["auc"]) for row in rows]
    hashes = [row["checkpoint_hashes"] for row in rows]
    common = sorted(set.intersection(*(set(item) for item in hashes))) if hashes else []
    all_hashes_equal = bool(hashes) and all(
        all(item.get(key) == hashes[0].get(key) for key in common) for item in hashes[1:]
    )
    return {
        "n": len(rows),
        "auc_values": values,
        "auc_range": max(values) - min(values) if values else None,
        "auc_std": statistics.pstdev(values) if values else None,
        "all_key_checkpoint_hashes_equal": all_hashes_equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    same_path = args.input_dir / "same_process_runs.jsonl"
    same = read_jsonl(same_path)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in same:
        grouped[(row["document_id"], row["condition"])].append(row)
    same_summary = {f"{doc}::{condition}": repeat_summary(rows) for (doc, condition), rows in sorted(grouped.items())}

    fresh_paths = sorted((args.input_dir / "fresh_process_runs").glob("*.json"))
    fresh_rows = [read_json(path) for path in fresh_paths]
    fresh_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in fresh_rows:
        fresh_grouped[(row["condition"], row["seed_mode"])].append(row)
    fresh_summary = {
        f"{condition}::{seed_mode}": repeat_summary(rows)
        for (condition, seed_mode), rows in sorted(fresh_grouped.items())
    }

    same_diffs = []
    for rows in grouped.values():
        values = [float(row["auc"]) for row in rows]
        same_diffs.extend(abs(a - b) for a, b in itertools.combinations(values, 2))
    fresh_diffs = []
    for rows in fresh_grouped.values():
        values = [float(row["auc"]) for row in rows]
        fresh_diffs.extend(abs(a - b) for a, b in itertools.combinations(values, 2))
    noise = same_diffs + fresh_diffs

    by_doc_repeat = {(row["document_id"], int(row["repeat"]), row["condition"]): float(row["auc"]) for row in same}
    method_effects = {}
    for learned in ("Dense_State", "Dense_Functional"):
        effects = []
        for (doc, repeat, condition), hadamard in by_doc_repeat.items():
            if condition != "Hadamard":
                continue
            key = (doc, repeat, learned)
            if key in by_doc_repeat:
                effects.append(hadamard - by_doc_repeat[key])
        method_effects[f"Hadamard_minus_{learned}"] = {
            "n": len(effects),
            "median": statistics.median(effects),
            "median_absolute": statistics.median(abs(item) for item in effects),
        }

    same_varies = any((item["auc_range"] or 0.0) > 0 or not item["all_key_checkpoint_hashes_equal"] for item in same_summary.values())
    fresh_varies = any((item["auc_range"] or 0.0) > 0 or not item["all_key_checkpoint_hashes_equal"] for item in fresh_summary.values())
    if same_varies and fresh_varies:
        root_class = "BOTH"
    elif same_varies:
        root_class = "PROCESS_INTERNAL_NONDETERMINISM"
    elif fresh_varies:
        root_class = "PROCESS_INITIALIZATION_NONDETERMINISM"
    else:
        root_class = "NONE"

    noise_max = max(noise, default=0.0)
    ratios = {
        key: (
            "INFINITE_ZERO_OBSERVED_NOISE"
            if noise_max == 0 and value["median_absolute"] > 0
            else value["median_absolute"] / noise_max if noise_max else None
        )
        for key, value in method_effects.items()
    }
    result = {
        "task": "LING_RUNTIME_REPEATABILITY_CLOSURE_V1",
        "same_process": same_summary,
        "fresh_process": fresh_summary,
        "SAME_PROCESS_REPEATABILITY": "FAIL" if same_varies else "PASS",
        "FRESH_PROCESS_REPEATABILITY": "FAIL" if fresh_varies else "PASS",
        "ROOT_CAUSE_CLASSIFICATION": root_class,
        "AUC_NOISE_P95": percentile(noise, 0.95),
        "AUC_NOISE_MAX": noise_max,
        "method_effects": method_effects,
        "METHOD_EFFECT_TO_RUNTIME_NOISE_RATIO": ratios,
        "source_sha256": {
            "same_process_runs": sha256(same_path),
            "fresh_process_runs": {path.name: sha256(path) for path in fresh_paths},
        },
        "fresh_process_file_count": len(fresh_rows),
        "durable_tensor_bytes": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "SAME_PROCESS_REPEATABILITY", "FRESH_PROCESS_REPEATABILITY", "ROOT_CAUSE_CLASSIFICATION",
        "AUC_NOISE_P95", "AUC_NOISE_MAX", "METHOD_EFFECT_TO_RUNTIME_NOISE_RATIO",
    )}, indent=2))


if __name__ == "__main__":
    main()
