"""CPU/stdlib summaries over already verified scores; no execution decisions.

Input must be completed legal payload scores from two identical independent
FrozenV4 passes. Historical comparisons use only their actual common problems.
"""
from __future__ import annotations

import math
import statistics


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def by_problem(rows):
    result = {}
    for row in rows:
        problem = row["problem_id"]
        require(problem not in result, "Duplicate scored problem")
        require(row["seed"] == 1 and type(row["correct"]) is bool, "Invalid scored seed/correctness")
        result[problem] = row
    return result


def paired(base, candidate, *, expected_n=None):
    left, right = by_problem(base), by_problem(candidate)
    common = sorted(set(left) & set(right))
    if expected_n is not None:
        require(len(common) == expected_n and len(left) == len(right) == expected_n,
                "Complete paired comparison requires identical fixed problem sets")
    both = [p for p in common if left[p]["correct"] and right[p]["correct"]]
    neither = [p for p in common if not left[p]["correct"] and not right[p]["correct"]]
    rescue = [p for p in common if not left[p]["correct"] and right[p]["correct"]]
    lost = [p for p in common if left[p]["correct"] and not right[p]["correct"]]
    discordant = len(rescue) + len(lost)
    pvalue = (1. if discordant == 0 else min(1., 2. * sum(
        math.comb(discordant, k) for k in range(min(len(rescue), len(lost)) + 1)) / (2 ** discordant)))
    n = len(common)
    return dict(status="COMPLETE_PAIRED30" if n == len(left) == len(right) == 30 else "COMMON_SUBSET_EXPLORATORY",
                N=n, common_problems=common, missing_base=sorted(set(right) - set(left)),
                missing_candidate=sorted(set(left) - set(right)),
                both_correct=both, neither_correct=neither, rescue=rescue, lost=lost,
                net=len(rescue) - len(lost),
                delta_percentage_points=(100. * (len(rescue) - len(lost)) / n if n else None),
                exact_mcnemar_p=pvalue if n else None, descriptive_only=True)


def condition_summary(rows):
    require(len(by_problem(rows)) == 30, "Final condition summary requires complete fixedN30")
    lengths = [row["generated_token_count"] for row in rows]
    seconds = [row["wall_seconds"] for row in rows]
    correct = sum(row["correct"] for row in rows)
    return dict(N=30, correct=correct, accuracy=correct / 30.,
                length_mean=statistics.mean(lengths), length_median=statistics.median(lengths),
                length_min=min(lengths), length_max=max(lengths),
                EOS=sum(row["finish_reason"] == "eos" for row in rows),
                length_terminated=sum(row["finish_reason"] == "length" for row in rows),
                generated_tokens=sum(lengths), summed_sample_wall_seconds=sum(seconds),
                effective_sample_TPS=sum(lengths) / sum(seconds),
                peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in rows),
                EXACT_physical_GPUh="UNRESOLVED_USE_INDEPENDENT_MONOTONIC_RESOURCE_LEDGER")


def summarize(current_scores, *, history=None):
    require(len(current_scores) == 60, "No full scientific summary/accuracy for incomplete formal60")
    groups = {condition: [row for row in current_scores if row["condition"] == condition]
              for condition in ("C1", "S3")}
    require(sum(map(len, groups.values())) == 60, "Unexpected current scored condition")
    result = dict(status="COMPLETE60_SINGLE_SEED_EXPLORATORY", conditions={
        condition: condition_summary(rows) for condition, rows in groups.items()},
        S3_vs_C1=paired(groups["C1"], groups["S3"], expected_n=30),
        limitations=["30 fixed questions, one seed, exploratory",
                     "AIME had been observed in preceding experiments",
                     "Resident FP32 decoded State is not packedINT8 storage",
                     "Old V1/V2/V3 STOP outcomes remain unchanged",
                     "Runtime/normalization grouping/UUID differences preclude strict causal interpretation"])
    if history is not None:
        comparisons = {}
        for name in ("P0", "P1"):
            prior = history.get(name, [])
            for condition in ("C1", "S3"):
                comparisons[condition + "_vs_" + name] = paired(prior, groups[condition])
        result["historical_common_question_comparisons"] = comparisons
        maps = {name: by_problem(history.get(name, [])) for name in ("P0", "P1")}
        maps.update({name: by_problem(groups[name]) for name in groups})
        common = sorted(set.intersection(*(set(rows) for rows in maps.values())))
        if common:
            net = lambda a, b: sum(int(maps[a][p]["correct"]) - int(maps[b][p]["correct"]) for p in common)
            result["factorial_interaction"] = dict(
                status="DESCRIPTIVE_COMMON_SUBSET_EXPLORATORY", N=len(common), common_problems=common,
                definition="(S3-C1)-(P1-P0)",
                percentage_points=100. * (net("S3", "C1") - net("P1", "P0")) / len(common),
                mechanism_claim=False, strict_causal_comparison=False)
        else:
            result["factorial_interaction"] = dict(status="NOT_ESTIMABLE_NO_COMMON_FOUR_CONDITION_SET", N=0)
    return result
