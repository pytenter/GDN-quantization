#!/usr/bin/env python3
"""Frozen scoring table and preregistered paired L6/L7 comparisons."""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


CONDITIONS = ("L2", "L6", "L7")
PRIMARY = (("P1", "L6", "L2"), ("P2", "L7", "L2"))
SECONDARY = (("S1", "L7", "L6"),)
ANALYSIS_SEED = 20260926


def finish_type(value):
    return value.get("type") if isinstance(value, dict) else value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mcnemar_exact(rescued: int, regressed: int) -> float:
    n = rescued + regressed
    if n == 0:
        return 1.0
    k = min(rescued, regressed)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def paired(name, treatment, reference, data, selected_ids=None):
    ids = sorted(set(data[treatment]) & set(data[reference]))
    if selected_ids is not None:
        ids = [qid for qid in ids if qid in set(selected_ids)]
    pairs = [(data[treatment][qid]["correct"], data[reference][qid]["correct"]) for qid in ids]
    rescued = [qid for qid, (a, b) in zip(ids, pairs) if a and not b]
    regressed = [qid for qid, (a, b) in zip(ids, pairs) if not a and b]
    rng = random.Random(ANALYSIS_SEED + sum(ord(c) for c in name))
    gains = []
    for _ in range(10000):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        gains.append(sum(int(a) - int(b) for a, b in sample) / len(sample))
    gains.sort()
    return {
        "name": name,
        "treatment": treatment,
        "reference": reference,
        "n": len(ids),
        "treatment_correct": sum(a for a, _ in pairs),
        "reference_correct": sum(b for _, b in pairs),
        "rescued": len(rescued),
        "regressed": len(regressed),
        "net_gain_questions": len(rescued) - len(regressed),
        "both_correct": sum(a and b for a, b in pairs),
        "both_wrong": sum(not a and not b for a, b in pairs),
        "rescued_ids": rescued,
        "regressed_ids": regressed,
        "mcnemar_exact_p": mcnemar_exact(len(rescued), len(regressed)),
        "bootstrap_replicates": 10000,
        "bootstrap_95pct_net_accuracy": [gains[249], gains[9749]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root)
    hardware_manifest = json.loads((root / "manifests/hardware_assignment.json").read_text())
    if hardware_manifest.get("status") != "FROZEN_BEFORE_FORMAL_GENERATION":
        raise RuntimeError("hardware assignment manifest is not frozen")
    hardware = {
        (row["condition"], row["question_id"]): row
        for row in hardware_manifest["assignments"]
    }
    baseline = json.loads((root / "manifests/baseline_reuse_manifest.json").read_text())
    if baseline["status"] != "PASS":
        raise RuntimeError("BASELINE_REUSE_GATE is not PASS")
    rows = []
    for item in baseline["rows"]:
        finish = item["finish_reason"]
        rows.append(
            {
                "condition": "L2",
                "question_id": item["question_id"],
                "seed": 1,
                "source": "REUSED_L2",
                "completed": bool(item["successful_sample"]),
                "correct": bool(item["v4_correct"]),
                "abstain": bool(item["v4_abstain"]),
                "generated_tokens": int(item["generated_tokens"]),
                "finish_type": finish_type(finish),
                "explicit_eos": isinstance(finish, dict) and finish.get("matched") == 156895,
                "max_length_hit": finish_type(finish) == "length",
                "runtime_seconds": None,
                "source_path": item["source_path"],
                "source_hash": item["source_file_hash"],
                "rotation_sha256": None,
                "gpu_architecture": "REUSED_L2",
                "gpu_model": "REUSED_L2",
                "worker_or_gpu_id": "REUSED_L2",
                "retry_count": 0,
                "infrastructure_failure_count": 0,
            }
        )
    for condition in ("L6", "L7"):
        for path in sorted((root / "outputs" / condition).glob("aime26_*_seed1.json")):
            item = json.loads(path.read_text())
            assignment = hardware[(condition, item["question_id"])]
            if item.get("seed") != assignment["generation_seed"]:
                raise RuntimeError(f"seed/assignment mismatch: {condition} {item['question_id']}")
            finish = item["finish_reason"]
            rows.append(
                {
                    "condition": condition,
                    "question_id": item["question_id"],
                    "seed": 1,
                    "source": "NEW_GENERATION",
                    "completed": bool(item["successful_sample"]),
                    "correct": bool(item["v4_correct"]),
                    "abstain": bool(item["v4_abstain"]),
                    "generated_tokens": int(item["generated_tokens"]),
                    "finish_type": finish_type(finish),
                    "explicit_eos": isinstance(finish, dict) and finish.get("matched") == 156895,
                    "max_length_hit": bool(item["hit_context_limit"]),
                    "runtime_seconds": item["runtime_seconds"],
                    "source_path": str(path),
                    "source_hash": file_sha256(path),
                    "rotation_sha256": item["final_rotation_sha256"],
                    "gpu_architecture": assignment["gpu_architecture"],
                    "gpu_model": assignment["gpu_model"],
                    "worker_or_gpu_id": assignment["worker_or_gpu_id"],
                    "retry_count": int(item.get("retry_count", 0)),
                    "infrastructure_failure_count": len(item.get("infrastructure_failures", [])),
                }
            )
    with (root / "analysis/per_sample_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = {condition: {} for condition in CONDITIONS}
    summary = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        for row in selected:
            data[condition][row["question_id"]] = row
        tokens = [row["generated_tokens"] for row in selected if row["completed"]]
        summary.append(
            {
                "condition": condition,
                "status": "COMPLETE" if len(selected) == 20 and all(row["completed"] for row in selected) else "NOT_COMPLETE",
                "completed": sum(row["completed"] for row in selected),
                "correct": sum(row["correct"] for row in selected),
                "incorrect": sum(row["completed"] and not row["correct"] for row in selected),
                "abstain": sum(row["abstain"] for row in selected),
                "length_limit": sum(row["max_length_hit"] for row in selected),
                "explicit_eos": sum(row["explicit_eos"] for row in selected),
                "generated_token_median": statistics.median(tokens) if tokens else None,
                "generated_token_mean": statistics.mean(tokens) if tokens else None,
                "generated_token_total": sum(tokens),
            }
        )
    with (root / "analysis/summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    comparisons = [paired(*spec, data) for spec in PRIMARY + SECONDARY]
    primary = comparisons[:2]
    order = sorted(range(2), key=lambda i: primary[i]["mcnemar_exact_p"])
    adjusted = [None, None]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, primary[index]["mcnemar_exact_p"] * (2 - rank)))
        adjusted[index] = running
    for item, value in zip(primary, adjusted):
        item["holm_adjusted_p_two_primary"] = value
    complete = all(row["status"] == "COMPLETE" for row in summary)
    result = {
        "status": "COMPLETE" if complete else "NOT_COMPLETE",
        "analysis_rng_seed": ANALYSIS_SEED,
        "unit": "question",
        "primary": primary,
        "secondary": comparisons[2:],
    }
    (root / "analysis/paired_comparisons.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    strata = {}
    for model in ("RTX4090", "RTX3090"):
        model_ids = {
            row["question_id"]
            for row in hardware_manifest["assignments"]
            if row["condition"] == "L6" and row["gpu_model"] == model
        }
        strata[model] = {
            "status": "DESCRIPTIVE_NOT_INDEPENDENT_PRIMARY_TEST",
            "n_questions": len(model_ids),
            "primary": [paired(*spec, data, selected_ids=model_ids) for spec in PRIMARY],
            "secondary": [paired(*spec, data, selected_ids=model_ids) for spec in SECONDARY],
            "condition_correct": {
                condition: sum(data[condition][qid]["correct"] for qid in sorted(model_ids))
                for condition in CONDITIONS
            },
        }
    hardware_result = {
        "status": "COMPLETE" if complete else "NOT_COMPLETE",
        "interpretation": "descriptive hardware strata; not independent primary tests",
        "post_hoc_subset_deletion_allowed": False,
        "strata": strata,
    }
    (root / "analysis/hardware_stratified.json").write_text(
        json.dumps(hardware_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "summary": summary, "comparisons": comparisons}, indent=2))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
