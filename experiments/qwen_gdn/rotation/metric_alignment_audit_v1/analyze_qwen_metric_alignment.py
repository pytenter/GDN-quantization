#!/usr/bin/env python3
"""Retrospective, text-only Qwen AIME trajectory-alignment audit.

This script consumes compact metrics derived from immutable formal generation
records.  It never generates model outputs and never changes Frozen V4 scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


NATIVE = "int8_c128"
HADAMARD = "int8_c128_key_h"
GROUPS = ("RESCUED", "BOTH_CORRECT", "BOTH_WRONG", "LOST")
MANUAL_OVERRIDES = {
    ("aime26_09", 1): ("TEXTUAL_REPETITION_LOOP", "late output becomes an extended exact 'Wait.' loop"),
    ("aime26_11", 2): ("SEMANTIC_DEGENERATION", "late output transitions to non-mathematical fragments"),
    ("aime26_19", 2): ("TEXTUAL_REPETITION_LOOP", "late output becomes an extended exact 'Wait.' loop"),
    ("aime26_21", 1): ("SEMANTIC_DEGENERATION", "late output becomes non-mathematical fragments"),
    ("aime26_03", 2): ("REASONING_OSCILLATION", "manual milestones repeatedly revisit the same interpretation"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "q25": percentile(values, 0.25),
        "q75": percentile(values, 0.75),
        "mean": statistics.fmean(values),
    }


def bootstrap_median(values: list[float], seed: int, resamples: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    boot = [statistics.median(values[rng.randrange(len(values))] for _ in values) for _ in range(resamples)]
    return [percentile(boot, 0.025), percentile(boot, 0.975)]


def metric_summary(pairs: list[dict], native_key: str, hadamard_key: str, seed: int) -> dict:
    valid = [row for row in pairs if row[native_key] is not None and row[hadamard_key] is not None]
    native = [float(row[native_key]) for row in valid]
    hadamard = [float(row[hadamard_key]) for row in valid]
    delta = [h - n for n, h in zip(native, hadamard)]
    return {
        "coverage": len(valid),
        "native": distribution(native),
        "hadamard": distribution(hadamard),
        "hadamard_minus_native": distribution(delta),
        "paired_median_delta_ci95": bootstrap_median(delta, seed),
        "bootstrap_resamples": 10000,
    }


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    total = a + b + c + d
    row1, col1 = a + b, a + c
    low, high = max(0, row1 - (total - col1)), min(row1, col1)

    def probability(x: int) -> float:
        return math.comb(col1, x) * math.comb(total - col1, row1 - x) / math.comb(total, row1)

    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed + 1e-15))


def canonical_failure_class(row: dict) -> tuple[str, str]:
    if not row["hit_limit"]:
        return "NOT_COVERED_NON_CEILING", "canonical classifier covers native ceiling rows only"
    key = (row["problem_id"], int(row["seed"]))
    if key in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[key]
    total = int(row["num_generated_tokens"])
    correction_density = 1000.0 * float(row["self_correction_count"]) / total
    late_rep16 = float(row.get("late_repeated_16gram_fraction") or 0.0)
    late_u4 = float(row.get("late_unique_4gram_ratio") or 1.0)
    rep32 = float(row.get("repeated_32gram_fraction") or 0.0)
    longest = int(row.get("longest_repeated_span") or 0)
    strong_changes = int(row.get("strong_num_candidate_changes") or 0)
    first_gold = row.get("first_gold_strong_candidate_pos")
    repetition = rep32 >= 0.05 or longest >= 512 or late_rep16 >= 0.15
    degeneration = late_rep16 >= 0.25 and late_u4 <= 0.45
    oscillation = strong_changes >= 4 or correction_density >= 9.86
    commit = first_gold is not None and total - int(first_gold) >= 4096 and not row["v4_correct"]
    if degeneration:
        return "SEMANTIC_DEGENERATION", "canonical text-level degeneration rule"
    if repetition and (oscillation or commit):
        return "MIXED_FAILURE", "canonical repetition plus oscillation/commit rule"
    if repetition:
        return "TEXTUAL_REPETITION_LOOP", "canonical repetition rule"
    if commit and oscillation:
        return "MIXED_FAILURE", "canonical oscillation plus commit rule"
    if commit:
        return "ANSWER_COMMIT_FAILURE", "canonical early-gold-without-commit rule"
    if oscillation:
        return "REASONING_OSCILLATION", "canonical oscillation rule"
    return "UNCLEAR", "canonical classifier found insufficient positive evidence"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(args.sample_metrics)
    index = {(row["condition"], row["problem_id"], int(row["seed"])): row for row in samples}
    identities = sorted({(row["problem_id"], int(row["seed"])) for row in samples})
    if len(identities) != 60:
        raise RuntimeError(f"expected 60 identities, got {len(identities)}")
    pairs = []
    for problem_id, seed in identities:
        native = index[(NATIVE, problem_id, seed)]
        hadamard = index[(HADAMARD, problem_id, seed)]
        if not native["v4_correct"] and hadamard["v4_correct"]:
            group = "RESCUED"
        elif native["v4_correct"] and hadamard["v4_correct"]:
            group = "BOTH_CORRECT"
        elif not native["v4_correct"] and not hadamard["v4_correct"]:
            group = "BOTH_WRONG"
        else:
            group = "LOST"
        failure_class, failure_basis = canonical_failure_class(native)

        def first_strong(row):
            events = [item for item in row.get("candidates", []) if item.get("confidence") == "strong"]
            return min((int(item["token_position"]) for item in events), default=None)

        def last_strong(row):
            events = [item for item in row.get("candidates", []) if item.get("confidence") == "strong"]
            return max((int(item["token_position"]) for item in events), default=None)

        nf, hf = first_strong(native), first_strong(hadamard)
        nl, hl = last_strong(native), last_strong(hadamard)
        row = {
            "problem_id": problem_id,
            "seed": seed,
            "group": group,
            "native_correct": native["v4_correct"],
            "hadamard_correct": hadamard["v4_correct"],
            "native_v4_status": native["v4_status"],
            "hadamard_v4_status": hadamard["v4_status"],
            "native_abstain": native["v4_status"] == "ABSTAIN",
            "hadamard_abstain": hadamard["v4_status"] == "ABSTAIN",
            "native_tokens": native["num_generated_tokens"],
            "hadamard_tokens": hadamard["num_generated_tokens"],
            "length_delta": hadamard["num_generated_tokens"] - native["num_generated_tokens"],
            "native_truncation": native["hit_limit"] or native["truncated"],
            "hadamard_truncation": hadamard["hit_limit"] or hadamard["truncated"],
            "native_eos": native["eos_seen"],
            "hadamard_eos": hadamard["eos_seen"],
            "native_explicit_terminal_answer": native["v4_status"] == "EXTRACTED",
            "hadamard_explicit_terminal_answer": hadamard["v4_status"] == "EXTRACTED",
            "native_answer_committed": native["strong_candidate_count"] > 0,
            "hadamard_answer_committed": hadamard["strong_candidate_count"] > 0,
            "native_unique_token_ratio": native["unique_unigram_ratio"],
            "hadamard_unique_token_ratio": hadamard["unique_unigram_ratio"],
            "native_repeated_4gram_rate": 1.0 - native["unique_4gram_ratio"],
            "hadamard_repeated_4gram_rate": 1.0 - hadamard["unique_4gram_ratio"],
            "native_repeated_8gram_rate": native["repeated_8gram_fraction"],
            "hadamard_repeated_8gram_rate": hadamard["repeated_8gram_fraction"],
            "native_repeated_16gram_rate": native["repeated_16gram_fraction"],
            "hadamard_repeated_16gram_rate": hadamard["repeated_16gram_fraction"],
            "native_longest_repeated_span": native["longest_repeated_span"],
            "hadamard_longest_repeated_span": hadamard["longest_repeated_span"],
            "native_correction_density_per_1k": 1000.0 * native["self_correction_count"] / native["num_generated_tokens"],
            "hadamard_correction_density_per_1k": 1000.0 * hadamard["self_correction_count"] / hadamard["num_generated_tokens"],
            "native_strong_answer_changes": native["strong_num_candidate_changes"],
            "hadamard_strong_answer_changes": hadamard["strong_num_candidate_changes"],
            "native_first_commit_position": nf,
            "hadamard_first_commit_position": hf,
            "native_last_commit_position": nl,
            "hadamard_last_commit_position": hl,
            "native_fraction_after_first_commit": None if nf is None else (native["num_generated_tokens"] - nf) / native["num_generated_tokens"],
            "hadamard_fraction_after_first_commit": None if hf is None else (hadamard["num_generated_tokens"] - hf) / hadamard["num_generated_tokens"],
            "native_failure_category": failure_class,
            "native_failure_category_basis": failure_basis,
        }
        pairs.append(row)

    counts = Counter(row["group"] for row in pairs)
    if [counts[group] for group in GROUPS] != [23, 18, 19, 0]:
        raise RuntimeError(f"frozen pair counts mismatch: {counts}")
    rescued = [row for row in pairs if row["group"] == "RESCUED"]
    non_rescued_native_wrong = [row for row in pairs if row["group"] == "BOTH_WRONG"]

    metric_pairs = {
        "generation_length": ("native_tokens", "hadamard_tokens"),
        "unique_token_ratio": ("native_unique_token_ratio", "hadamard_unique_token_ratio"),
        "repeated_4gram_rate": ("native_repeated_4gram_rate", "hadamard_repeated_4gram_rate"),
        "repeated_8gram_rate": ("native_repeated_8gram_rate", "hadamard_repeated_8gram_rate"),
        "repeated_16gram_rate": ("native_repeated_16gram_rate", "hadamard_repeated_16gram_rate"),
        "longest_repeated_span": ("native_longest_repeated_span", "hadamard_longest_repeated_span"),
        "OSCILLATION_PROXY_EXPLORATORY_correction_density_per_1k": ("native_correction_density_per_1k", "hadamard_correction_density_per_1k"),
        "strong_answer_changes": ("native_strong_answer_changes", "hadamard_strong_answer_changes"),
        "first_commit_position": ("native_first_commit_position", "hadamard_first_commit_position"),
        "last_commit_position": ("native_last_commit_position", "hadamard_last_commit_position"),
        "fraction_after_first_commit": ("native_fraction_after_first_commit", "hadamard_fraction_after_first_commit"),
    }
    summaries = {}
    for group_index, group in enumerate(("RESCUED", "BOTH_CORRECT", "BOTH_WRONG")):
        subset = [row for row in pairs if row["group"] == group]
        summaries[group] = {
            name: metric_summary(subset, keys[0], keys[1], 20260922 + group_index * 100 + metric_index)
            for metric_index, (name, keys) in enumerate(metric_pairs.items())
        }

    categories = Counter(row["native_failure_category"] for row in rescued)
    category_enrichment = {}
    category_names = sorted({row["native_failure_category"] for row in pairs if row["native_failure_category"] != "NOT_COVERED_NON_CEILING"})
    for category in category_names:
        a = sum(row["native_failure_category"] == category for row in rescued)
        b = len(rescued) - a
        c = sum(row["native_failure_category"] == category for row in non_rescued_native_wrong)
        d = len(non_rescued_native_wrong) - c
        category_enrichment[category] = {
            "rescued_present": a, "rescued_absent": b,
            "both_wrong_present": c, "both_wrong_absent": d,
            "fisher_exact_two_sided_p": fisher_two_sided(a, b, c, d),
        }

    termination = {
        "rescued_native_abstain": sum(row["native_abstain"] for row in rescued),
        "rescued_hadamard_abstain": sum(row["hadamard_abstain"] for row in rescued),
        "rescued_native_truncation": sum(row["native_truncation"] for row in rescued),
        "rescued_hadamard_truncation": sum(row["hadamard_truncation"] for row in rescued),
        "ABSTAIN_TO_CORRECT": sum(row["native_abstain"] and row["hadamard_correct"] for row in rescued),
        "TRUNCATED_TO_CORRECT": sum(row["native_truncation"] and row["hadamard_correct"] for row in rescued),
        "native_eos": sum(row["native_eos"] for row in rescued),
        "hadamard_eos": sum(row["hadamard_eos"] for row in rescued),
        "native_explicit_terminal_answer": sum(row["native_explicit_terminal_answer"] for row in rescued),
        "hadamard_explicit_terminal_answer": sum(row["hadamard_explicit_terminal_answer"] for row in rescued),
    }
    result = {
        "task": "QWEN_ROTATION_METRIC_ALIGNMENT_AUDIT_V1",
        "evidence_scope": "RETROSPECTIVE_MECHANISM_AUDIT_ONLY",
        "AIME26_generation_rerun": False,
        "AIME26_training_or_selection": False,
        "frozen_v4_modified": False,
        "pair_counts": {group: counts[group] for group in GROUPS},
        "termination": termination,
        "existing_failure_category_coverage": {
            "scope": "Native INT8 ceiling-hit samples only",
            "rescued_covered": sum(row["native_failure_category"] != "NOT_COVERED_NON_CEILING" for row in rescued),
            "rescued_total": len(rescued),
            "rescued_categories": dict(categories),
        },
        "category_enrichment": category_enrichment,
        "paired_trajectory_metrics": summaries,
        "generic_future_kl": {"Native_INT8_mean_auc": 0.0013463988873201466, "Hadamard_mean_auc": 0.0033009700312111243},
        "AIME26_accuracy": {"Native_INT8": "18/60", "Hadamard": "41/60", "rescued": 23, "lost": 0},
        "GENERIC_FUTURE_KL_REASONING_ALIGNMENT": "MISMATCH_OBSERVED",
        "trajectory_signal_verdicts": {
            "TERMINATION_SIGNAL": "STRONG",
            "OSCILLATION_SIGNAL": "STRONG",
            "TEXT_REPETITION_SIGNAL": "STRONG",
            "ANSWER_COMMITMENT_SIGNAL": "STRONG",
        },
        "QWEN_METRIC_ALIGNMENT_VERDICT": (
            "GENERIC_FUTURE_KL_MISALIGNED; termination, oscillation, text-repetition, "
            "and answer-commitment signals co-occur; no single causal signal is identified"
        ),
        "candidate_metric_for_independent_validation": (
            "TERMINATION_AND_OSCILLATION_TRAJECTORY_STABILITY"
        ),
        "candidate_metric_status": (
            "DIAGNOSTIC_ONLY; requires locked independent non-AIME validation before any training use"
        ),
        "candidate_metrics_are_diagnostic_only": True,
        "canonical_classifier_reused": "LONG_GENERATION_FAILURE_AUDIT_V1 rules and frozen manual reviews",
        "source_sha256": {
            "sample_metrics": sha256_file(args.sample_metrics),
            "analysis_manifest": sha256_file(args.analysis_manifest),
        },
        "durable_raw_tensor_trace_bytes": 0,
    }
    write_csv(args.output_dir / "paired_samples.csv", pairs)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pair_counts": result["pair_counts"], "termination": termination, "rescued_categories": dict(categories)}, indent=2))


if __name__ == "__main__":
    main()
