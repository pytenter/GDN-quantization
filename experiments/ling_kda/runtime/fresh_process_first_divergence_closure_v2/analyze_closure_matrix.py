#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


BEFORE_P95 = 0.023128505481597707
BEFORE_MAX = 0.0400105529926776
METHOD_EFFECT = 0.009027184397862181


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob("*.json"))]
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 closure runs, got {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["document_id"], row["condition"])].append(row)
    group_summary = {}
    spreads = []
    bitwise = []
    for (document, condition), items in sorted(groups.items()):
        values = [float(item["auc"]) for item in items]
        hashes = [item["checkpoint_hashes"] for item in items]
        reference_hashes = [item["reference_hashes"] for item in items]
        spread = max(values) - min(values)
        spreads.append(spread)
        keys = set.intersection(*(set(value) for value in hashes))
        condition_exact = all(len({value[key] for value in hashes}) == 1 for key in keys)
        reference_exact = all(value == reference_hashes[0] for value in reference_hashes[1:])
        exact = condition_exact and reference_exact
        bitwise.append(exact)
        group_summary[f"{document}|{condition}"] = {
            "auc_values": values, "auc_range": spread, "auc_std": statistics.pstdev(values),
            "condition_key_checkpoints_bitwise": condition_exact,
            "reference_trajectory_bitwise": reference_exact,
            "all_key_checkpoints_bitwise": exact,
        }
    p95 = percentile(spreads, .95)
    maximum = max(spreads)
    ratio = METHOD_EFFECT / max(maximum, 1e-12)
    closed = all(bitwise) and maximum == 0.0
    tolerance = (not closed and p95 < BEFORE_P95 * .25 and maximum < BEFORE_MAX * .25 and ratio > 1.0)
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "runs": len(rows), "documents": 3, "conditions": ["Hadamard", "Dense_State"], "repeats": 5,
        "per_document_condition": group_summary,
        "AUC_NOISE_P95_BEFORE": BEFORE_P95, "AUC_NOISE_P95_AFTER": p95,
        "AUC_NOISE_MAX_BEFORE": BEFORE_MAX, "AUC_NOISE_MAX_AFTER": maximum,
        "METHOD_EFFECT": METHOD_EFFECT, "METHOD_EFFECT_TO_RUNTIME_NOISE_RATIO": ratio,
        "all_key_checkpoints_bitwise": all(bitwise),
        "LING_REPEATABILITY_VERDICT": "REPEATABILITY_CLOSED" if closed else ("REPEATABILITY_CLOSED_WITH_NUMERICAL_TOLERANCE" if tolerance else "REPEATABILITY_NOT_CLOSED"),
        "16_doc_authorized": bool(closed or tolerance),
        "64_doc_authorized": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
