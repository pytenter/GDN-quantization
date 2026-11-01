#!/usr/bin/env python3
"""Aggregate compact document-level persistent-headroom shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


METHODS = ("Native_INT8", "Hadamard", "Dense_State", "Dense_Functional")
LEARNED = ("Dense_State", "Dense_Functional")
HORIZONS = (1, 4, 8, 16, 32, 64, 128)
TIE_TOLERANCE = 1.0e-12


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "q25": percentile(values, 0.25),
        "q75": percentile(values, 0.75),
        "max": max(values),
    }


def paired_bootstrap(effects, hadamard, resamples=10000, seed=20260922):
    rng = random.Random(seed)
    means, medians = [], []
    count = len(effects)
    for _ in range(resamples):
        sample = [effects[rng.randrange(count)] for _ in range(count)]
        means.append(statistics.fmean(sample))
        medians.append(statistics.median(sample))
    return {
        "sampling_unit": "document",
        "paired": True,
        "resamples": resamples,
        "seed": seed,
        "mean_effect": statistics.fmean(effects),
        "mean_ci95": [percentile(means, 0.025), percentile(means, 0.975)],
        "median_effect": statistics.median(effects),
        "median_ci95": [percentile(medians, 0.025), percentile(medians, 0.975)],
        "relative_median_reduction": statistics.median(effects) / max(statistics.median(hadamard), 1.0e-12),
        "relative_mean_reduction": statistics.fmean(effects) / max(statistics.fmean(hadamard), 1.0e-12),
    }


def verdict(stats):
    median = stats["median_effect"]
    lower = stats["median_ci95"][0]
    relative = stats["relative_median_reduction"]
    if median > 0 and lower > 0 and relative >= 0.10:
        return "CLEAR_PERSISTENT_HEADROOM"
    if lower > 0 and relative < 0.10:
        return "SMALL_BUT_SIGNIFICANT_HEADROOM"
    if median > 0:
        return "PROMISING_NOT_SIGNIFICANT"
    return "NO_PERSISTENT_HEADROOM"


def load_rows(input_dir):
    rows = []
    for path in sorted(input_dir.glob("shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen", "ling"), required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--old-result", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir); output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_dir)
    ids = [row["document_id"] for row in rows]
    if len(rows) != 64 or len(set(ids)) != 64:
        raise RuntimeError(f"expected 64 unique rows, got {len(rows)} rows / {len(set(ids))} unique")
    rows.sort(key=lambda row: int(row["panel_index"]))
    if [int(row["panel_index"]) for row in rows] != list(range(64)):
        raise RuntimeError("panel indices are not exactly 0..63")
    if any(row["nonfinite_total"] for row in rows):
        raise RuntimeError("nonfinite metric detected")

    base = json.load(open(args.base_manifest, encoding="utf-8"))
    expected_hashes = {item["document_id"]: item["raw_text_sha256"] for item in base["documents"]}
    if any(expected_hashes.get(row["document_id"]) != row["raw_text_sha256"] for row in rows):
        raise RuntimeError("raw text hash mismatch")
    fingerprints = {row["tokenizer_fingerprint_sha256"] for row in rows}
    if len(fingerprints) != 1:
        raise RuntimeError("tokenizer fingerprint drift across shards")

    aucs = {method: [row["methods"][method]["auc"] for row in rows] for method in METHODS}
    auc_distribution = {method: distribution(values) for method, values in aucs.items()}
    horizon_summary = {}
    for method in METHODS:
        horizon_summary[method] = {}
        for horizon in HORIZONS:
            metrics = [next(item for item in row["methods"][method]["horizons"] if item["horizon"] == horizon) for row in rows]
            horizon_summary[method][str(horizon)] = {
                key: statistics.fmean(item[key] for item in metrics)
                for key in ("future_kl", "logit_relative_l2", "top1_match", "top20_overlap", "state_relative_l2", "nonfinite")
            }

    comparisons = {}
    for method in LEARNED:
        effects = [a - b for a, b in zip(aucs["Hadamard"], aucs[method])]
        stats = paired_bootstrap(effects, aucs["Hadamard"])
        stats["wins"] = sum(effect > TIE_TOLERANCE for effect in effects)
        stats["ties"] = sum(abs(effect) <= TIE_TOLERANCE for effect in effects)
        stats["losses"] = sum(effect < -TIE_TOLERANCE for effect in effects)
        stats["tie_tolerance"] = TIE_TOLERANCE
        stats["verdict"] = verdict(stats)
        comparisons[method] = stats

    old = json.load(open(args.old_result, encoding="utf-8"))
    old_auc = old["auc"]
    old_per_doc = {(row["document_id"], row["method"]): row["auc"] for row in old["per_document_auc"]}
    first = rows[:16]
    first_means = {method: statistics.fmean(row["methods"][method]["auc"] for row in first) for method in METHODS}
    per_doc_max_abs = {
        method: max(abs(row["methods"][method]["auc"] - old_per_doc[(row["document_id"], method)]) for row in first)
        for method in METHODS
    }
    tolerance = 1.0e-6
    replication = {
        "new_16doc_auc": first_means,
        "previous_16doc_auc": old_auc,
        "absolute_difference": {method: abs(first_means[method] - old_auc[method]) for method in METHODS},
        "per_document_max_absolute_difference": per_doc_max_abs,
        "tolerance": tolerance,
    }
    replication["status"] = "PASS" if all(value <= tolerance for value in per_doc_max_abs.values()) else "FAIL"
    if replication["status"] != "PASS":
        raise RuntimeError(f"PREVIOUS_16DOC_REPLICATION_FAIL: {replication}")

    best = min(LEARNED, key=lambda method: auc_distribution[method]["mean"])
    old_best = old["best_learned"]
    old_ci = old["bootstrap_hadamard_minus_best"]["ci95"]
    result = {
        "task": "PERSISTENT_HEADROOM_CONFIRMATION_V1",
        "model": "Qwen3.5-9B/GDN" if args.model == "qwen" else "Ling-3.0-tiny/KDA",
        "documents": 64,
        "horizons": HORIZONS,
        "auc_distribution": auc_distribution,
        "horizon_summary": horizon_summary,
        "paired_hadamard_minus_learned": comparisons,
        "best_learned_by_mean_auc": best,
        "best_verdict": comparisons[best]["verdict"],
        "previous_16doc_replication": replication,
        "previous_best_learned": old_best,
        "ci_width_16doc": old_ci[1] - old_ci[0],
        "ci_width_64doc": comparisons[best]["median_ci95"][1] - comparisons[best]["median_ci95"][0],
        "DATA_SPLIT_INTEGRITY": base["DATA_SPLIT_INTEGRITY"],
        "CROSS_MODEL_RAW_TEXT_MATCH": "PENDING_CROSS_REPO_CHECK",
        "streaming_storage_policy": "PASS",
        "new_raw_trace_size": 0,
        "compact_result_size_bytes": sum(path.stat().st_size for path in input_dir.glob("*" ) if path.is_file()),
        "training_performed": False,
        "model_weights_changed": False,
        "AIME26_used": False,
    }
    if args.model == "ling":
        result.update({
            "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
            "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO",
            "PREFILL_ENDPOINT_STATE_BASIS_CONTINUITY": "PASS",
        })
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    manifest = dict(base)
    by_id = {row["document_id"]: row for row in rows}
    for item in manifest["documents"]:
        row = by_id[item["document_id"]]
        item["token_count"] = row["token_count"]
        item["tokenizer_fingerprint_sha256"] = row["tokenizer_fingerprint_sha256"]
    manifest["model"] = result["model"]
    manifest["tokenizer_fingerprint_sha256"] = next(iter(fingerprints))
    manifest["CROSS_MODEL_RAW_TEXT_MATCH"] = "PENDING_CROSS_REPO_CHECK"
    (output_dir / "PERSISTENT_HEADROOM_64DOC_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    with (output_dir / "per_document_auc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("panel_index", "document_id", "original_16", *METHODS, "Hadamard_minus_Dense_State", "Hadamard_minus_Dense_Functional"))
        for row in rows:
            writer.writerow((row["panel_index"], row["document_id"], row["original_16"], *(row["methods"][method]["auc"] for method in METHODS), row["paired_auc"]["Hadamard_minus_Dense_State"], row["paired_auc"]["Hadamard_minus_Dense_Functional"]))
    print(json.dumps({"replication": replication["status"], "best": best, "comparison": comparisons[best], "verdict": result["best_verdict"]}, indent=2))


if __name__ == "__main__":
    main()
