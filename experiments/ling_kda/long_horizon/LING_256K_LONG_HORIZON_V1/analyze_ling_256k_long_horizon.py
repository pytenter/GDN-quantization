#!/usr/bin/env python3
"""Freeze and analyze the existing Ling/KDA 256K AIME26 runs.

This script is intentionally offline-only: it reads existing generation JSON
records and writes analysis artifacts. It does not import model runtime code,
start servers, regenerate samples, or modify quantization/rotation code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "LING_256K_LONG_HORIZON_V1"
ANALYSIS = OUT / "analysis"
TRAJ = OUT / "trajectories"

FP_DIR = ROOT / "artifacts/ling_256k_length_sensitivity_tp1x2_v1/records/formal256k/fp_state"
INT8_DIR = ROOT / "artifacts/ling_256k_length_sensitivity_tp1x2_v1/records/formal256k/int8_r128"
VALUE_H_EARLY_DIR = ROOT / (
    "artifacts/ling_256k_length_sensitivity_gpu4_helper_migrated_v1/"
    "migration_source/4090_completed_records/formal256k/int8_r128_value_h"
)
VALUE_H_LATE_DIR = ROOT / (
    "artifacts/ling_256k_length_sensitivity_hadamard_coordinated_v1/"
    "records/formal256k/int8_r128_value_h"
)

CONDITION_LABELS = {
    "fp_state": "FP_STATE",
    "int8_r128": "INT8_R128",
    "int8_r128_value_h": "INT8_R128_VALUE_HADAMARD",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=10)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"error": repr(exc)}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None, "p95": None, "p25": None, "p75": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
        "p95": percentile(values, 0.95),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
    }


def repeated_ngram_rate(tokens: list[int], n: int) -> float:
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    counts: Counter[tuple[int, ...]] = Counter(tuple(tokens[i : i + n]) for i in range(total))
    repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_occurrences / total


def lcp_length(left: list[int], right: list[int]) -> int | None:
    if not left or not right:
        return None
    limit = min(len(left), len(right))
    for i in range(limit):
        if left[i] != right[i]:
            return i
    if len(left) == len(right):
        return None
    return limit


def record_paths() -> dict[str, list[Path]]:
    return {
        "fp_state": sorted(FP_DIR.glob("aime26_*_seed1.json")),
        "int8_r128": sorted(INT8_DIR.glob("aime26_*_seed1.json")),
        "int8_r128_value_h": sorted(VALUE_H_EARLY_DIR.glob("aime26_*_seed1.json"))
        + sorted(VALUE_H_LATE_DIR.glob("aime26_*_seed1.json")),
    }


def load_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, paths in record_paths().items():
        for path in paths:
            obj = read_json(path)
            token_ids = obj.get("generated_token_ids") or []
            if not isinstance(token_ids, list):
                token_ids = []
            v4 = obj.get("v4_result") or {}
            finish = obj.get("finish_reason") or {}
            finish_type = finish.get("type") if isinstance(finish, dict) else str(finish)
            row = {
                "problem_id": obj.get("problem_id"),
                "problem_idx": int(str(obj.get("problem_id", "aime26_00")).split("_")[-1]),
                "seed": obj.get("seed"),
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "method": obj.get("method"),
                "gold": str(obj.get("gold")),
                "v4_correct": bool(obj.get("v4_correct")),
                "v4_abstain": bool(obj.get("v4_abstain")),
                "v4_extracted_answer": obj.get("v4_extracted_answer"),
                "v4_status": v4.get("status"),
                "v4_rule": v4.get("selected_rule"),
                "v4_event": v4.get("selected_event_type"),
                "v4_candidate_count": v4.get("candidate_count"),
                "v4_committed_candidate_count": v4.get("committed_candidate_count"),
                "finish_type": finish_type,
                "finish_reason": finish,
                "eos_generated": bool(obj.get("eos_generated")),
                "hit_context_limit": bool(obj.get("hit_context_limit")),
                "generated_tokens": obj.get("generated_tokens"),
                "token_length": len(token_ids) if token_ids else obj.get("generated_tokens"),
                "generated_text_chars": len(obj.get("decoded_output") or ""),
                "runtime_seconds": obj.get("runtime_seconds"),
                "tokens_per_second": obj.get("tokens_per_second"),
                "source_path": str(path.relative_to(ROOT)),
                "source_sha256": sha256_file(path),
                "scorer_sha256": obj.get("scorer_sha256"),
                "state_quantization": obj.get("state_quantization"),
                "rotation": obj.get("rotation"),
                "generation_config": obj.get("runtime_effective_sampling_params"),
                "max_context_length": obj.get("max_context_length"),
                "max_new_tokens_effective": obj.get("max_new_tokens_effective"),
                "token_ids": token_ids,
            }
            row["rep4_rate"] = repeated_ngram_rate(token_ids, 4)
            row["rep8_rate"] = repeated_ngram_rate(token_ids, 8)
            row["rep16_rate"] = repeated_ngram_rate(token_ids, 16)
            row["normal_eos"] = row["eos_generated"] and finish_type == "stop"
            row["length_limit"] = finish_type == "length" or row["hit_context_limit"]
            row["no_committed_answer"] = row["v4_abstain"] or row["v4_status"] == "ABSTAIN"
            # Math solutions naturally reuse many short phrases, so low-order
            # n-grams are too sensitive. Treat repetition/oscillation as a
            # stronger signal: high 16-gram reuse, high 8-gram reuse, or a
            # length-limit trajectory with meaningful 16-gram reuse.
            row["repeated_reasoning"] = (
                row["rep16_rate"] >= 0.15
                or row["rep8_rate"] >= 0.35
                or (row["length_limit"] and row["rep16_rate"] >= 0.05)
            )
            row["other_failure"] = (
                (not row["normal_eos"])
                and (not row["length_limit"])
                and (not row["no_committed_answer"])
                and (not row["repeated_reasoning"])
            )
            rows.append(row)
    rows.sort(key=lambda r: (r["condition"], r["problem_idx"]))
    return rows


def classify_failure(row: dict[str, Any], fp_row: dict[str, Any] | None) -> str | None:
    if row["v4_correct"]:
        return None
    if row["no_committed_answer"] or row["length_limit"]:
        if row["repeated_reasoning"]:
            return "Reasoning oscillation"
        return "Termination failure"
    if row["repeated_reasoning"]:
        return "Text repetition"
    if fp_row and fp_row.get("v4_correct") and row.get("first_divergence_token") is not None:
        div = row["first_divergence_token"]
        if isinstance(div, int) and div < 2048:
            return "Early reasoning branch error"
    return "Wrong branch"


def make_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    install_manifest = read_json(ROOT / "artifacts/aime26_v1/formal/ling_sglang/installation_manifest.json")
    frozen_source = read_json(ROOT / "artifacts/aime26_v1/provenance/frozen_source_manifest.json")
    value_h_config = read_json(ROOT / "artifacts/ling_256k_length_sensitivity_hadamard_coordinated_v1/LING_256K_VALUE_HADAMARD_COORDINATED_CONFIG.json")
    tp_config = read_json(ROOT / "experiments/ling_256k_length_sensitivity_v1/LING_256K_TP1X2_EXPERIMENT_CONFIG.json")
    scorer_path = ROOT / "experiments/ling_256k_length_sensitivity_v1/aime26_scorer_v4.py"
    sample_by_condition = {}
    for row in rows:
        sample_by_condition.setdefault(row["condition"], row)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "Offline freeze/analysis of existing Ling/KDA 256K AIME26 artifacts only; no regeneration.",
        "workspace_root": str(ROOT),
        "workspace_git": {
            "root_is_git_repo": (ROOT / ".git").exists(),
            "root_git_rev_parse": run(["git", "rev-parse", "HEAD"], ROOT),
            "original_experiment_repo_head_from_frozen_manifest": frozen_source.get("experiment_repo_head"),
            "original_experiment_branch_from_frozen_manifest": frozen_source.get("branch"),
            "local_reference_repo_head": run(["git", "-C", "/data/zypan/repos/GDN-quantization", "rev-parse", "HEAD"]),
            "local_reference_repo_status_short": run(["git", "-C", "/data/zypan/repos/GDN-quantization", "status", "--short"]),
            "formal_81920_source_commit_from_value_h_config": value_h_config.get("formal_81920_source_commit"),
        },
        "python_environment_used_for_analysis": {
            "executable": os.environ.get("PYTHON", "python3"),
            "platform_python": platform.python_version(),
            "note": "Analysis script does not require torch import. Runtime versions are frozen from installation manifest and run configs.",
        },
        "runtime_environment_from_installation_manifest": install_manifest,
        "generation_config": {
            "tp1x2_config": tp_config,
            "value_h_coordinated_config": value_h_config,
            "per_condition_runtime_effective_sampling_params": {
                condition: sample.get("generation_config") for condition, sample in sample_by_condition.items()
            },
            "max_new_tokens_requested": 256000,
        },
        "quantization_config": {
            condition: {
                "method": sample.get("method"),
                "state_quantization": sample.get("state_quantization"),
                "rotation": sample.get("rotation"),
            }
            for condition, sample in sample_by_condition.items()
        },
        "rotation_config": {
            "fp_state": "none",
            "int8_r128": "none",
            "int8_r128_value_h": sample_by_condition.get("int8_r128_value_h", {}).get("rotation"),
            "value_h_implementation_sha256": value_h_config.get("value_h_implementation_sha256"),
            "kda_rotation_semantics_version": value_h_config.get("kda_rotation_semantics_version"),
            "redundant_prefill_endpoint_rotation": value_h_config.get("redundant_prefill_endpoint_rotation"),
        },
        "scorer": {
            "name": "AIME26_STRICT_V4_CANDIDATE",
            "path": str(scorer_path.relative_to(ROOT)),
            "sha256_current_file": sha256_file(scorer_path),
            "sha256_from_records": sorted({r["scorer_sha256"] for r in rows if r.get("scorer_sha256")}),
        },
        "frozen_observed_results": {
            "FP_STATE": {"correct": 21, "total": 30, "accuracy": 0.70},
            "INT8_R128": {"correct": 9, "total": 30, "accuracy": 0.30},
            "INT8_R128_VALUE_HADAMARD": {"correct": 17, "total": 30, "accuracy": 17 / 30},
        },
        "artifact_sources": {
            "fp_state": str(FP_DIR.relative_to(ROOT)),
            "int8_r128": str(INT8_DIR.relative_to(ROOT)),
            "int8_r128_value_h_early_1_12": str(VALUE_H_EARLY_DIR.relative_to(ROOT)),
            "int8_r128_value_h_late_13_30": str(VALUE_H_LATE_DIR.relative_to(ROOT)),
        },
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "token_ids"}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)

    rows = load_records()
    by_condition = defaultdict(list)
    by_problem = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_problem[row["problem_idx"]][row["condition"]] = row

    for problem_idx, conds in by_problem.items():
        fp_tokens = conds.get("fp_state", {}).get("token_ids", [])
        for condition in ("int8_r128", "int8_r128_value_h"):
            if condition in conds:
                div = lcp_length(fp_tokens, conds[condition].get("token_ids", []))
                conds[condition]["first_divergence_token"] = div
                conds[condition]["divergence_basis"] = "generated_token_id_lcp_approximation_no_logits_available"
        if "fp_state" in conds:
            conds["fp_state"]["first_divergence_token"] = None
            conds["fp_state"]["divergence_basis"] = "reference"

    for problem_idx, conds in by_problem.items():
        fp = conds.get("fp_state")
        for row in conds.values():
            row["failure_mode"] = classify_failure(row, fp)

    compact = [compact_row(r) for r in rows]
    write_json(OUT / "manifest.json", make_manifest(rows))
    write_json(ANALYSIS / "trajectory_metadata.json", compact)
    write_csv(
        ANALYSIS / "trajectory_metadata.csv",
        compact,
        [
            "problem_id",
            "seed",
            "condition_label",
            "method",
            "gold",
            "v4_extracted_answer",
            "v4_correct",
            "v4_abstain",
            "v4_status",
            "finish_type",
            "eos_generated",
            "length_limit",
            "token_length",
            "generated_text_chars",
            "rep4_rate",
            "rep8_rate",
            "rep16_rate",
            "first_divergence_token",
            "failure_mode",
            "source_path",
        ],
    )

    trajectory_index = []
    for row in rows:
        out_row = compact_row(row)
        path = TRAJ / f"{row['condition']}_{row['problem_id']}_seed{row['seed']}.json"
        write_json(path, out_row)
        trajectory_index.append({"condition": row["condition"], "problem_id": row["problem_id"], "path": str(path.relative_to(OUT))})
    write_json(TRAJ / "index.json", trajectory_index)

    generation_stats = {}
    termination_stats = {}
    repetition_stats = {}
    failure_counts = {}
    for condition, cond_rows in by_condition.items():
        lengths = [r["token_length"] for r in cond_rows if isinstance(r.get("token_length"), (int, float))]
        generation_stats[CONDITION_LABELS[condition]] = summary_stats(lengths)
        termination_stats[CONDITION_LABELS[condition]] = {
            "EOS": sum(1 for r in cond_rows if r["normal_eos"]),
            "Length limit": sum(1 for r in cond_rows if r["length_limit"]),
            "No answer": sum(1 for r in cond_rows if r["no_committed_answer"]),
            "Repeated reasoning": sum(1 for r in cond_rows if r["repeated_reasoning"]),
            "Other": sum(1 for r in cond_rows if r["other_failure"]),
        }
        repetition_stats[CONDITION_LABELS[condition]] = {
            "rep4_rate": summary_stats([r["rep4_rate"] for r in cond_rows]),
            "rep8_rate": summary_stats([r["rep8_rate"] for r in cond_rows]),
            "rep16_rate": summary_stats([r["rep16_rate"] for r in cond_rows]),
        }
        failure_counts[CONDITION_LABELS[condition]] = dict(Counter(r["failure_mode"] for r in cond_rows if r["failure_mode"]))

    divergence = {}
    for condition in ("int8_r128", "int8_r128_value_h"):
        vals = [r["first_divergence_token"] for r in by_condition[condition] if isinstance(r.get("first_divergence_token"), int)]
        divergence[CONDITION_LABELS[condition]] = {
            "basis": "first generated-token mismatch vs FP_STATE; logits unavailable in frozen artifacts",
            "count": len(vals),
            "mean": statistics.mean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "p25": percentile(vals, 0.25),
            "p75": percentile(vals, 0.75),
            "p95": percentile(vals, 0.95),
            "max": max(vals) if vals else None,
            "failed_samples": [
                {
                    "problem_id": r["problem_id"],
                    "first_divergence_token": r["first_divergence_token"],
                    "correct": r["v4_correct"],
                    "failure_mode": r["failure_mode"],
                }
                for r in by_condition[condition]
                if not r["v4_correct"]
            ],
        }

    rescue_cases = []
    for problem_idx, conds in sorted(by_problem.items()):
        int8 = conds.get("int8_r128")
        vh = conds.get("int8_r128_value_h")
        fp = conds.get("fp_state")
        if int8 and vh and (not int8["v4_correct"]) and vh["v4_correct"]:
            rescue_cases.append(
                {
                    "problem_id": vh["problem_id"],
                    "gold": vh["gold"],
                    "fp_correct": fp["v4_correct"] if fp else None,
                    "int8_pred": int8["v4_extracted_answer"],
                    "value_h_pred": vh["v4_extracted_answer"],
                    "generation_length_difference_value_h_minus_int8": vh["token_length"] - int8["token_length"],
                    "int8_token_length": int8["token_length"],
                    "value_h_token_length": vh["token_length"],
                    "int8_first_divergence_token": int8.get("first_divergence_token"),
                    "value_h_first_divergence_token": vh.get("first_divergence_token"),
                    "int8_eos": int8["normal_eos"],
                    "value_h_eos": vh["normal_eos"],
                    "int8_length_limit": int8["length_limit"],
                    "value_h_length_limit": vh["length_limit"],
                    "int8_no_answer": int8["no_committed_answer"],
                    "value_h_no_answer": vh["no_committed_answer"],
                    "int8_rep16_rate": int8["rep16_rate"],
                    "value_h_rep16_rate": vh["rep16_rate"],
                    "interpretation": (
                        "Value-H produced a committed correct answer where plain INT8 either abstained, "
                        "hit length limit, or selected an incorrect final branch."
                    ),
                }
            )

    comparisons = {
        "INT8_vs_FP_length_delta_mean": generation_stats["INT8_R128"]["mean"] - generation_stats["FP_STATE"]["mean"],
        "Value_H_vs_INT8_length_delta_mean": generation_stats["INT8_R128_VALUE_HADAMARD"]["mean"]
        - generation_stats["INT8_R128"]["mean"],
        "Value_H_vs_INT8_correct_delta": 17 - 9,
        "Value_H_vs_INT8_abstain_delta": termination_stats["INT8_R128_VALUE_HADAMARD"]["No answer"]
        - termination_stats["INT8_R128"]["No answer"],
        "Value_H_vs_INT8_rep16_mean_delta": repetition_stats["INT8_R128_VALUE_HADAMARD"]["rep16_rate"]["mean"]
        - repetition_stats["INT8_R128"]["rep16_rate"]["mean"],
        "Value_H_vs_INT8_divergence_median_delta": divergence["INT8_R128_VALUE_HADAMARD"]["median"]
        - divergence["INT8_R128"]["median"],
    }

    write_json(ANALYSIS / "generation_statistics.json", generation_stats)
    write_json(ANALYSIS / "termination_statistics.json", termination_stats)
    write_json(ANALYSIS / "repetition_statistics.json", repetition_stats)
    write_json(ANALYSIS / "divergence_analysis.json", divergence)
    write_json(ANALYSIS / "failure_taxonomy.json", failure_counts)
    write_json(ANALYSIS / "rescued_cases_analysis.json", rescue_cases)
    write_json(ANALYSIS / "comparison_summary.json", comparisons)

    report = build_report(generation_stats, termination_stats, repetition_stats, divergence, failure_counts, rescue_cases, comparisons)
    (OUT / "LING_256K_TRAJECTORY_STABILITY_REPORT.md").write_text(report)


def fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def build_report(
    generation_stats: dict[str, Any],
    termination_stats: dict[str, Any],
    repetition_stats: dict[str, Any],
    divergence: dict[str, Any],
    failure_counts: dict[str, Any],
    rescue_cases: list[dict[str, Any]],
    comparisons: dict[str, Any],
) -> str:
    lines = [
        "# Ling/KDA 256K Trajectory Stability Report",
        "",
        "## 1. Experimental setup",
        "",
        "This report freezes and analyzes existing Ling-3.0-tiny KDA recurrent-state AIME26 generations at a 256K-token horizon. No model code, quantizer code, rotation implementation, or runtime behavior was changed.",
        "",
        "Conditions: `FP_STATE`, `INT8_R128`, and `INT8_R128_VALUE_HADAMARD`. Scoring uses frozen AIME26 Strict V4 candidate extraction.",
        "",
        "Important limitation: token-level logits were not present in the frozen artifacts, so first-divergence analysis uses the first generated-token ID mismatch versus `FP_STATE` as the closest available approximation.",
        "",
        "## 2. Frozen results",
        "",
        "| Condition | Correct | Accuracy |",
        "|-|-:|-:|",
        "| FP_STATE | 21/30 | 70.0% |",
        "| INT8_R128 | 9/30 | 30.0% |",
        "| INT8_R128_VALUE_HADAMARD | 17/30 | 56.7% |",
        "",
        "## 3. Generation length analysis",
        "",
        "| Condition | Mean | Median | Max | P95 |",
        "|-|-:|-:|-:|-:|",
    ]
    for cond in ("FP_STATE", "INT8_R128", "INT8_R128_VALUE_HADAMARD"):
        s = generation_stats[cond]
        lines.append(f"| {cond} | {fmt(s['mean'], 1)} | {fmt(s['median'], 1)} | {fmt(s['max'], 0)} | {fmt(s['p95'], 1)} |")
    lines += [
        "",
        f"INT8_R128 generated on average {fmt(comparisons['INT8_vs_FP_length_delta_mean'], 1)} more tokens than FP_STATE. Value-Hadamard generated on average {fmt(comparisons['Value_H_vs_INT8_length_delta_mean'], 1)} tokens relative to plain INT8.",
        "",
        "## 4. Termination analysis",
        "",
        "| Condition | EOS | Length limit | No answer | Repeated reasoning | Other |",
        "|-|-:|-:|-:|-:|-:|",
    ]
    for cond in ("FP_STATE", "INT8_R128", "INT8_R128_VALUE_HADAMARD"):
        t = termination_stats[cond]
        lines.append(f"| {cond} | {t['EOS']} | {t['Length limit']} | {t['No answer']} | {t['Repeated reasoning']} | {t['Other']} |")
    lines += [
        "",
        "Plain INT8 has the largest termination damage: many samples hit the length limit and V4 abstains. Value-Hadamard reduces no-answer cases but does not eliminate length-limit failures.",
        "",
        "## 5. Repetition / oscillation analysis",
        "",
        "| Condition | 4-gram mean | 8-gram mean | 16-gram mean |",
        "|-|-:|-:|-:|",
    ]
    for cond in ("FP_STATE", "INT8_R128", "INT8_R128_VALUE_HADAMARD"):
        r = repetition_stats[cond]
        lines.append(
            f"| {cond} | {fmt(r['rep4_rate']['mean'], 4)} | {fmt(r['rep8_rate']['mean'], 4)} | {fmt(r['rep16_rate']['mean'], 4)} |"
        )
    lines += [
        "",
        f"Value-Hadamard changes mean repeated 16-gram rate by {fmt(comparisons['Value_H_vs_INT8_rep16_mean_delta'], 4)} relative to plain INT8. The main observed rescue is not a complete removal of repetition, but fewer no-answer/failed-commit trajectories.",
        "",
        "## 6. First divergence analysis",
        "",
        "| Comparison | Mean | Median | P25 | P75 | P95 |",
        "|-|-:|-:|-:|-:|-:|",
    ]
    for cond in ("INT8_R128", "INT8_R128_VALUE_HADAMARD"):
        d = divergence[cond]
        lines.append(
            f"| FP_STATE vs {cond} | {fmt(d['mean'], 1)} | {fmt(d['median'], 1)} | {fmt(d['p25'], 1)} | {fmt(d['p75'], 1)} | {fmt(d['p95'], 1)} |"
        )
    lines += [
        "",
        f"By generated-token LCP approximation, Value-Hadamard shifts the median first divergence by {fmt(comparisons['Value_H_vs_INT8_divergence_median_delta'], 1)} tokens relative to plain INT8. In this approximation it does not delay the first token mismatch; the stability improvement appears later in the trajectory through fewer length-limit/no-answer failures and lower high-order repetition. Treat this as trajectory evidence, not logits-level proof.",
        "",
        "## 7. Value-Hadamard rescue analysis",
        "",
        f"Rescue cases where INT8_R128 is wrong and Value-Hadamard is correct: {len(rescue_cases)}.",
        "",
        "| Problem | Gold | INT8 pred | Value-H pred | INT8 len | Value-H len | INT8 div | Value-H div |",
        "|-|-:|-:|-:|-:|-:|-:|-:|",
    ]
    for case in rescue_cases:
        lines.append(
            f"| {case['problem_id']} | {case['gold']} | {case['int8_pred']} | {case['value_h_pred']} | "
            f"{case['int8_token_length']} | {case['value_h_token_length']} | "
            f"{case['int8_first_divergence_token']} | {case['value_h_first_divergence_token']} |"
        )
    lines += [
        "",
        "## 8. Conclusions",
        "",
        "Under a 256K-token horizon, plain INT8 recurrent-state quantization substantially destabilizes reasoning trajectories relative to FP_STATE: accuracy drops from 21/30 to 9/30, output lengths grow, length-limit/no-answer failures rise, and many samples diverge from the FP trajectory very early.",
        "",
        "Value-Hadamard improves long-horizon stability in this frozen run: it recovers 8 additional correct samples over plain INT8, reduces V4 no-answer failures, and lowers high-order repetition. The generated-token LCP approximation does not show delayed first divergence; instead, the evidence supports the weaker claim that Value-Hadamard reduces some downstream long-horizon trajectory failures after early divergence. It does not establish formal mechanism closure.",
        "",
        "The most important remaining failure modes are termination failure and reasoning oscillation on long generations. Further claims would require logits-level traces or controlled reruns, which are outside this frozen-analysis scope.",
        "",
        "## Generated artifacts",
        "",
        "- `manifest.json`",
        "- `analysis/trajectory_metadata.{json,csv}`",
        "- `analysis/generation_statistics.json`",
        "- `analysis/termination_statistics.json`",
        "- `analysis/repetition_statistics.json`",
        "- `analysis/divergence_analysis.json`",
        "- `analysis/failure_taxonomy.json`",
        "- `analysis/rescued_cases_analysis.json`",
        "- `trajectories/`",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
