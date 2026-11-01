#!/usr/bin/env python3
"""Offline canonical C5/C6 freeze, Frozen-V4 scoring, paired, and trajectory analysis.

This script is frozen before the C5/C6 outputs are complete.  It never launches
generation and it refuses to score until all four workers exited zero and both
conditions pass the preregistered 60-record identity gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1")
DATASET = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl")
BASELINE_ROOT = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/results/aime26/81920/qwen_gdn")
BASELINE_CSV = BASELINE_ROOT / "frozen_v4_per_sample.csv"
BASELINE_SUMMARY = BASELINE_ROOT / "frozen_v4_summary.json"
SCORER = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/experiments/shared/scoring/aime26_scorer_v4.py")

EXPECTED = {
    "dataset_sha256": "0a744825990f027c6c4455b78776512964bfdbde1e39e8826ad188c6a0d74b9d",
    "scorer_sha256": "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b",
    "formal_config_sha256": "557ac6670fe6e88606be226ccf0dea289ebd34dd4c2be54e7e54b9b6ee9e0836",
    "hardware_assignment_sha256": "75b5df954f64f606f30143df3bf5329e238a6eabe8253bfc287141b0443c475a",
    "runner_sha256": "77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69",
    "C5_rotation_sha256": "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
    "C6_rotation_sha256": "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c",
}
METHOD = {"C5": "QWEN_INT8_C128_KEY_C5", "C6": "QWEN_INT8_C128_KEY_C6"}
H_CONDITION = "QWEN_INT8_C128_KEY_HADAMARD"
PAIR_NAMES = (("C5", "H"), ("C6", "H"), ("C6", "C5"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"corrupt JSON: {path}:{line_no}") from exc
    return rows


def import_scorer():
    if sha256(SCORER) != EXPECTED["scorer_sha256"]:
        raise RuntimeError("Frozen V4 scorer hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_v4", SCORER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_dataset() -> tuple[list[dict], dict[tuple[str, int], dict]]:
    if sha256(DATASET) != EXPECTED["dataset_sha256"]:
        raise RuntimeError("dataset hash mismatch")
    problems = load_jsonl(DATASET)
    if len(problems) != 30:
        raise RuntimeError(f"expected 30 problems, got {len(problems)}")
    jobs = {(r["problem_id"], seed): r for r in problems for seed in (1, 2)}
    if len(jobs) != 60:
        raise RuntimeError("dataset identities are not 60 unique problem/seed pairs")
    return problems, jobs


def output_payload_sha(row: dict) -> str:
    payload = dict(row)
    expected = payload.pop("output_payload_sha256", None)
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if expected != actual:
        raise RuntimeError(f"output payload hash mismatch: {row.get('problem_id')} seed={row.get('seed')}")
    return actual


def validate_and_freeze() -> tuple[dict[str, dict[tuple[str, int], dict]], dict]:
    checks = {
        ROOT / "configs/formal_config.json": EXPECTED["formal_config_sha256"],
        ROOT / "configs/hardware_assignment.json": EXPECTED["hardware_assignment_sha256"],
        ROOT / "scripts/run_qwen_c5_c6_formal.py": EXPECTED["runner_sha256"],
        DATASET: EXPECTED["dataset_sha256"],
        SCORER: EXPECTED["scorer_sha256"],
    }
    for path, expected_hash in checks.items():
        if sha256(path) != expected_hash:
            raise RuntimeError(f"frozen identity mismatch: {path}")
    exit_codes = {}
    for worker in range(4):
        path = ROOT / f"logs/worker{worker}.exitcode"
        if not path.exists():
            raise RuntimeError(f"worker exitcode missing: {worker}")
        exit_codes[str(worker)] = int(path.read_text().strip())
    if set(exit_codes.values()) != {0}:
        raise RuntimeError(f"not all workers exited zero: {exit_codes}")

    _, expected_jobs = load_dataset()
    assignment = json.loads((ROOT / "configs/hardware_assignment.json").read_text())
    assigned = {(j["condition"], j["problem_id"], int(j["seed"])): j for j in assignment["jobs"]}
    if len(assigned) != 120:
        raise RuntimeError("hardware assignment is not 120 unique jobs")

    all_rows: dict[str, dict[tuple[str, int], dict]] = {}
    source_files = []
    per_sample_gpu = []
    for condition in ("C5", "C6"):
        rows = []
        for worker in range(4):
            path = ROOT / f"outputs/formal/shards/worker{worker}/{condition}.jsonl"
            summary_path = path.with_suffix(".summary.json")
            if not path.exists() or not summary_path.exists():
                raise RuntimeError(f"missing output or shard summary: {path}")
            shard_summary = json.loads(summary_path.read_text())
            if shard_summary.get("gate") != "PASS" or shard_summary.get("record_count") != 15:
                raise RuntimeError(f"shard verification failed: {summary_path}")
            shard_rows = load_jsonl(path)
            if len(shard_rows) != 15:
                raise RuntimeError(f"wrong shard row count: {path}")
            rows.extend(shard_rows)
            source_files.append({
                "condition": condition,
                "worker": worker,
                "path": str(path),
                "bytes": path.stat().st_size,
                "rows": len(shard_rows),
                "sha256": sha256(path),
            })
        if len(rows) != 60:
            raise RuntimeError(f"{condition} count != 60")
        indexed = {}
        expected_rotation = EXPECTED[f"{condition}_rotation_sha256"]
        for row in rows:
            key = (row["problem_id"], int(row["seed"]))
            if key in indexed or key not in expected_jobs:
                raise RuntimeError(f"duplicate/unexpected {condition} identity: {key}")
            job = assigned.get((condition, key[0], key[1]))
            if not job:
                raise RuntimeError(f"missing frozen assignment: {condition} {key}")
            expected_problem = expected_jobs[key]
            required = (
                row.get("method") == METHOD[condition]
                and row.get("configuration") == METHOD[condition]
                and row.get("problem") == expected_problem["problem"]
                and str(row.get("gold_answer")) == str(expected_problem["answer"])
                and int(row.get("gpu")) == int(job["gpu"])
                and int(row.get("worker_id")) == int(job["gpu"])
                and int(row.get("retry_count", -1)) == 0
                and row.get("formal_config_sha256") == EXPECTED["formal_config_sha256"]
                and row.get("rotation_sha256") == expected_rotation
                and row.get("runtime_error") is None
                and row.get("nonfinite") is False
                and row.get("scoring_status") == "DEFERRED_UNTIL_ALL_C5_C6_OUTPUTS_FROZEN"
                and row.get("correct") is None
                and row.get("extracted_answer") is None
                and row.get("direct_frozen_load") is True
                and row.get("theta_reconstruction") is False
                and row.get("finish_reason") in {"eos", "length"}
                and int(row.get("generated_token_count")) == len(row.get("generated_token_ids", []))
                and int(row.get("output_tokens")) == len(row.get("generated_token_ids", []))
                and int(row.get("max_new_tokens")) == 81920
            )
            if not required:
                raise RuntimeError(f"formal identity/content gate failed: {condition} {key}")
            output_payload_sha(row)
            indexed[key] = row
            per_sample_gpu.append({
                "condition": condition, "problem_id": key[0], "seed": key[1],
                "gpu": int(row["gpu"]), "worker_id": int(row["worker_id"]),
                "start_utc": row["start_utc"], "end_utc": row["end_utc"],
                "prompt_hash": row["input_ids_hash"], "config_hash": row["formal_config_sha256"],
                "rotation_sha256": row["rotation_sha256"], "output_payload_sha256": row["output_payload_sha256"],
                "generated_token_count": int(row["generated_token_count"]),
                "finish_reason": row["finish_reason"], "retry_count": int(row["retry_count"]),
            })
        if set(indexed) != set(expected_jobs):
            raise RuntimeError(f"{condition} coverage mismatch")
        all_rows[condition] = indexed

    for key in expected_jobs:
        c5, c6 = all_rows["C5"][key], all_rows["C6"][key]
        if c5["input_ids_hash"] != c6["input_ids_hash"] or c5["gpu"] != c6["gpu"]:
            raise RuntimeError(f"paired prompt/GPU mismatch: {key}")

    hash_dir = ROOT / "hashes"
    hash_dir.mkdir(parents=True, exist_ok=True)
    for condition in ("C5", "C6"):
        lines = []
        for item in source_files:
            if item["condition"] == condition:
                rel = os.path.relpath(item["path"], ROOT)
                lines.append(f"{item['sha256']}  {rel}")
        atomic_text(hash_dir / f"{condition}_SHA256SUMS", "\n".join(lines) + "\n")
    atomic_json(ROOT / "analysis/per_sample_gpu_metadata.json", sorted(per_sample_gpu, key=lambda x: (x["condition"], x["problem_id"], x["seed"])))
    freeze = {
        "gate": "PASS", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "FORMAL_PROTOCOL_FROZEN": "PASS", "C5_COMPLETENESS_GATE": "PASS", "C6_COMPLETENESS_GATE": "PASS",
        "C5_records": 60, "C6_records": 60, "workers_exit_codes": exit_codes,
        "source_files": source_files,
        "identities": {k: EXPECTED[k] for k in ("dataset_sha256", "scorer_sha256", "formal_config_sha256", "hardware_assignment_sha256", "runner_sha256")},
        "scoring_performed_before_freeze": False,
    }
    atomic_json(ROOT / "analysis/output_freeze_manifest.json", freeze)
    for item in source_files:
        Path(item["path"]).chmod(0o444)
        Path(item["path"]).with_suffix(".summary.json").chmod(0o444)
    return all_rows, freeze


def load_baseline_csv() -> tuple[dict[tuple[str, int], dict], list[dict]]:
    rows = list(csv.DictReader(BASELINE_CSV.open(encoding="utf-8")))
    h = {(r["problem_id"], int(r["seed"])): r for r in rows if r["condition"] == H_CONDITION}
    if len(h) != 60:
        raise RuntimeError("Hadamard frozen baseline is not 60 unique samples")
    return h, rows


def score_records(condition: str, rows: dict[tuple[str, int], dict], scorer) -> dict:
    scored = []
    for key in sorted(rows):
        row = rows[key]
        result = scorer.extract_v4(row["response"], row["finish_reason"], row["problem"])
        correct = bool(scorer.compare_with_gold(result, row["gold_answer"]))
        scored.append({
            "condition": condition, "problem_id": key[0], "seed": key[1],
            "gold_answer": int(row["gold_answer"]), "v4_status": result.status,
            "v4_extracted_answer": result.normalized_value, "v4_correct": correct,
            "v4_abstain_reason": result.abstain_reason, "finish_reason": row["finish_reason"],
            "truncated": bool(row["truncated"]), "eos_seen": bool(row["eos_seen"]),
            "generated_token_count": int(row["generated_token_count"]),
            "output_payload_sha256": row["output_payload_sha256"],
            "extraction": result.to_dict(),
        })
    correct = sum(r["v4_correct"] for r in scored)
    abstain = sum(r["v4_status"] == "ABSTAIN" for r in scored)
    return {"condition": condition, "scorer": scorer.SCORER_NAME, "scorer_sha256": EXPECTED["scorer_sha256"],
            "samples": 60, "correct": correct, "wrong": 60 - correct, "abstain": abstain,
            "accuracy": correct / 60, "records": scored}


def exact_p(rescue: int, regress: int) -> float:
    n = rescue + regress
    if n == 0:
        return 1.0
    k = min(rescue, regress)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    result, running = {}, 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, raw[name] * (m - rank)))
        result[name] = running
    return result


def paired_table(name: str, left: dict, right: dict, left_name: str, right_name: str) -> tuple[dict, list[dict]]:
    records = []
    rescue_ids, regress_ids = [], []
    for key in sorted(left):
        a, b = bool(left[key]), bool(right[key])
        category = "same_correct" if a and b else "same_wrong" if not a and not b else "rescue" if a else "regress"
        rec = {"problem_id": key[0], "seed": key[1], left_name: a, right_name: b, "category": category}
        records.append(rec)
        sample_id = f"{key[0]}_seed{key[1]}"
        if category == "rescue": rescue_ids.append(sample_id)
        if category == "regress": regress_ids.append(sample_id)
    counts = Counter(r["category"] for r in records)
    result = {
        "comparison": name, "paired_samples": 60,
        "same_correct": counts["same_correct"], "same_wrong": counts["same_wrong"],
        "rescue": counts["rescue"], "regress": counts["regress"],
        "net": counts["rescue"] - counts["regress"],
        "exact_two_sided_p": exact_p(counts["rescue"], counts["regress"]),
        "rescue_sample_ids": rescue_ids, "regress_sample_ids": regress_ids,
    }
    return result, records


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def ngram_metrics(tokens: list[int], n: int) -> tuple[float, float]:
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0.0, 0.0
    counts = Counter(tuple(tokens[i:i+n]) for i in range(total))
    repeated = sum(count - 1 for count in counts.values()) / total
    unique = len(counts) / total
    return repeated, unique


def trajectory_record(condition: str, row: dict, scorer) -> dict:
    tokens = [int(x) for x in row["generated_token_ids"]]
    events = scorer.enumerate_claim_events(row["response"], row["problem"])
    committed = [e for e in events if e.committed and not e.rejection_reasons() and e.normalized_value is not None]
    committed.sort(key=lambda e: (e.start, e.end))
    values = [int(e.normalized_value) for e in committed]
    changes = sum(a != b for a, b in zip(values, values[1:]))
    aba = sum(values[i] == values[i+2] != values[i+1] for i in range(max(0, len(values)-2)))
    retractions = sum(bool(e.retracted) for e in events)
    first = committed[0] if committed else None
    text_len = len(row["response"])
    metrics = {
        "condition": condition, "problem_id": row["problem_id"], "seed": int(row["seed"]),
        "generated_tokens": len(tokens), "length_ceiling": row["finish_reason"] == "length",
        "termination_type": row["finish_reason"], "committed_claim_count": len(committed),
        "first_committed_claim": None if first is None else int(first.normalized_value),
        "first_committed_claim_char": None if first is None else int(first.start),
        "post_commit_fraction": None if first is None or not text_len else (text_len - int(first.end)) / text_len,
        "answer_changes": changes, "retractions": retractions, "oscillation_proxy": changes + aba + retractions,
        "committed_aba_count": aba,
    }
    for n in (4, 8, 16):
        repeated, unique = ngram_metrics(tokens, n)
        metrics[f"repeated_{n}gram_fraction"] = repeated
        metrics[f"unique_{n}gram_ratio"] = unique
    return metrics


def median(values):
    values = [float(x) for x in values if x is not None]
    return statistics.median(values) if values else None


def trajectory_analysis(new_rows, h_rows, comparisons, scorer):
    by_condition = {"C5": new_rows["C5"], "C6": new_rows["C6"], "H": h_rows}
    trajectory = {}
    flat = []
    for condition, rows in by_condition.items():
        trajectory[condition] = {}
        for key, row in sorted(rows.items()):
            rec = trajectory_record(condition, row, scorer)
            trajectory[condition][key] = rec
            flat.append(rec)
    numeric = ["generated_tokens", "length_ceiling", "repeated_4gram_fraction", "repeated_8gram_fraction",
               "repeated_16gram_fraction", "unique_4gram_ratio", "unique_8gram_ratio", "unique_16gram_ratio",
               "post_commit_fraction", "answer_changes", "retractions", "oscillation_proxy"]
    groups = {}
    for comp_name, comp_rows in comparisons.items():
        left_name, right_name = comp_name.split("_vs_")
        groups[comp_name] = {}
        for category in ("rescue", "regress", "same_correct", "same_wrong"):
            keys = [(r["problem_id"], int(r["seed"])) for r in comp_rows if r["category"] == category]
            summary = {"n": len(keys), "sample_ids": [f"{k[0]}_seed{k[1]}" for k in keys]}
            for metric in numeric:
                summary[f"{left_name}_{metric}_median"] = median([trajectory[left_name][k][metric] for k in keys])
                summary[f"{right_name}_{metric}_median"] = median([trajectory[right_name][k][metric] for k in keys])
                diffs = []
                for k in keys:
                    a, b = trajectory[left_name][k][metric], trajectory[right_name][k][metric]
                    if a is not None and b is not None:
                        diffs.append(float(a) - float(b))
                summary[f"delta_{left_name}_minus_{right_name}_{metric}_median"] = median(diffs)
            groups[comp_name][category] = summary
    return flat, {"metric_definition": {
        "ngram_repeated_fraction": "sum(count-1)/number_of_ngrams over generated token IDs",
        "ngram_unique_ratio": "unique ngrams / number_of_ngrams over generated token IDs",
        "claims": "Frozen V4 enumerate_claim_events; committed, unrejected, integer-normalized events only",
        "post_commit_fraction": "characters after first committed claim end / total response characters",
        "answer_changes": "adjacent changes in committed integer claim sequence",
        "retractions": "Frozen V4 claim events flagged retracted",
        "oscillation_proxy": "answer_changes + committed A-B-A count + retractions",
    }, "groups": groups}


def endpoint_signal(comp: dict) -> str:
    if comp["net"] < 0:
        return "NEGATIVE_OR_MIXED"
    if comp["net"] == 0:
        return "NOT_CONFIRMED"
    return "SUPPORTIVE" if comp["holm_p"] <= 0.05 else "SUPPORTIVE_BUT_UNDERPOWERED"


def git_identity() -> tuple[str, str]:
    repo = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
    branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return branch, head


def run_self_test() -> None:
    scorer = import_scorer()
    h_csv, _ = load_baseline_csv()
    paths = sorted({Path(r["source_file"]) for r in h_csv.values()})
    raw = {}
    for path in paths:
        for row in load_jsonl(path):
            raw[(row["problem_id"], int(row["seed"]))] = row
    if set(raw) != set(h_csv):
        raise RuntimeError("self-test baseline raw/CSV identity mismatch")
    mismatches = []
    for key, row in raw.items():
        result = scorer.extract_v4(row["response"], row["finish_reason"], row["problem"])
        correct = scorer.compare_with_gold(result, row["gold_answer"])
        expected = h_csv[key]
        expected_value = expected["v4_extracted_answer"]
        expected_value = None if expected_value == "" else int(expected_value)
        if result.status != expected["v4_status"] or result.normalized_value != expected_value or bool(correct) != (expected["v4_correct"] == "True"):
            mismatches.append({"key": key, "actual_status": result.status, "actual_value": result.normalized_value,
                               "actual_correct": bool(correct), "expected": expected})
    result = {"gate": "PASS" if not mismatches else "FAIL", "baseline": "Hadamard60",
              "records": len(raw), "mismatches": mismatches,
              "scorer_sha256": sha256(SCORER), "baseline_csv_sha256": sha256(BASELINE_CSV)}
    atomic_json(ROOT / "analysis/offline_postprocess_self_test.json", result)
    if mismatches:
        raise RuntimeError(f"Frozen V4 self-test failed: {len(mismatches)} mismatches")
    print(json.dumps(result, indent=2))


def run_final() -> None:
    rows, freeze = validate_and_freeze()
    scorer = import_scorer()
    h_csv, _ = load_baseline_csv()
    h_correct = {k: r["v4_correct"] == "True" for k, r in h_csv.items()}
    h_raw = {}
    for path in sorted({Path(r["source_file"]) for r in h_csv.values()}):
        for row in load_jsonl(path):
            h_raw[(row["problem_id"], int(row["seed"]))] = row
    scored = {c: score_records(c, rows[c], scorer) for c in ("C5", "C6")}
    atomic_json(ROOT / "analysis/C5_scored.json", scored["C5"])
    atomic_json(ROOT / "analysis/C6_scored.json", scored["C6"])
    correctness = {c: {(r["problem_id"], int(r["seed"])): bool(r["v4_correct"]) for r in scored[c]["records"]} for c in ("C5", "C6")}
    correctness["H"] = h_correct

    comparison_records, pair_summary, raw_p = {}, {}, {}
    for left, right in PAIR_NAMES:
        name = f"{left}_vs_{right}"
        summary, records = paired_table(name, correctness[left], correctness[right], left, right)
        comparison_records[name] = records
        pair_summary[name] = summary
        raw_p[name] = summary["exact_two_sided_p"]
    adjusted = holm(raw_p)
    file_names = {"C5_vs_H": "paired_C5_vs_H.csv", "C6_vs_H": "paired_C6_vs_H.csv", "C6_vs_C5": "paired_C6_vs_C5.csv"}
    for name, records in comparison_records.items():
        pair_summary[name]["holm_p"] = adjusted[name]
        write_csv(ROOT / "analysis" / file_names[name], records)
    atomic_json(ROOT / "analysis/paired_summary.json", pair_summary)

    trajectory_rows, trajectory_summary = trajectory_analysis(rows, h_raw, comparison_records, scorer)
    atomic_text(ROOT / "analysis/trajectory_metrics.jsonl", "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in trajectory_rows))
    atomic_json(ROOT / "analysis/trajectory_group_analysis.json", trajectory_summary)

    branch, head = git_identity()
    c5_signal = endpoint_signal(pair_summary["C5_vs_H"])
    c6_signal = endpoint_signal(pair_summary["C6_vs_H"])
    next_action = "MECHANISM_CLOSURE" if c6_signal in {"SUPPORTIVE", "SUPPORTIVE_BUT_UNDERPOWERED"} else "REASSESS_OBJECTIVE"
    final = {
        "FORMAL_PROTOCOL_FROZEN": "PASS", "C5_COMPLETED": "60/60", "C6_COMPLETED": "60/60",
        "C5_CORRECT": f"{scored['C5']['correct']}/60", "C6_CORRECT": f"{scored['C6']['correct']}/60",
        "FP_CORRECT": "52/60", "INT8_C128_CORRECT": "18/60", "HADAMARD_CORRECT": "41/60",
        "comparisons": pair_summary,
        "C5_ENDPOINT_SIGNAL": c5_signal, "C6_ENDPOINT_SIGNAL": c6_signal,
        "C5_STRUCTURED_TRAJECTORY_SIGNAL": "DESCRIPTIVE_ONLY_NO_CAUSAL_CLAIM",
        "C6_STRUCTURED_TRAJECTORY_SIGNAL": "DESCRIPTIVE_ONLY_NO_CAUSAL_CLAIM",
        "CURRENT_RESULT_INTERPRETATION": "Endpoint labels follow the preregistered paired/Holm rule; trajectory summaries are descriptive and do not select checkpoints or metrics.",
        "NEXT_ACTION": next_action,
        "GPU_COUNT": 4, "GPU_MODEL": "NVIDIA GeForce RTX 3090",
        "NEW_GENERATIONS_EXPECTED": 120, "NEW_GENERATIONS_COMPLETED": 120,
        "RETRIES": 0, "INFRA_FAILURES": 1,
        "OUTPUT_HASH_VERIFICATION": "PASS", "FINAL_BRANCH": branch, "FINAL_HEAD": head,
        "freeze_manifest_sha256": sha256(ROOT / "analysis/output_freeze_manifest.json"),
    }
    atomic_json(ROOT / "analysis/final_summary.json", final)
    table = [
        ("FP", 52), ("INT8-C128", 18), ("Hadamard", 41),
        ("C5", scored["C5"]["correct"]), ("C6", scored["C6"]["correct"]),
    ]
    lines = ["# QWEN C5/C6 AIME26 Canonical60 Final Report", "", "## Endpoint table", "",
             "| Condition | Correct / 60 | Accuracy |", "|---|---:|---:|"]
    lines += [f"| {name} | {value}/60 | {100*value/60:.2f}% |" for name, value in table]
    lines += ["", "## Paired comparisons", ""]
    for name in ("C5_vs_H", "C6_vs_H", "C6_vs_C5"):
        p = pair_summary[name]
        lines.append(f"- {name}: rescue={p['rescue']}, regress={p['regress']}, net={p['net']}, exact p={p['exact_two_sided_p']:.8g}, Holm p={p['holm_p']:.8g}")
    lines += ["", "## Interpretation", "", f"- C5_ENDPOINT_SIGNAL = {c5_signal}", f"- C6_ENDPOINT_SIGNAL = {c6_signal}",
              "- Trajectory findings are descriptive only; see `analysis/trajectory_group_analysis.json`.", f"- NEXT_ACTION = {next_action}",
              "", "## Provenance", "", f"- Frozen V4 scorer SHA256: `{EXPECTED['scorer_sha256']}`",
              f"- Formal config SHA256: `{EXPECTED['formal_config_sha256']}`", f"- Final branch: `{branch}`", f"- Final HEAD: `{head}`",
              "- New sample retries: 0", "- Infrastructure incidents: 1 pre-model launcher-hash typo; zero output records were produced by the failed launch.", ""]
    atomic_text(ROOT / "reports/FINAL_REPORT.md", "\n".join(lines))

    atomic_text(ROOT / "analysis/POSTPROCESS_COMPLETE", datetime.now(timezone.utc).isoformat() + "\n")
    # Hash all stable artifacts last.  The manifest cannot hash itself, and the
    # live tmux log remains mutable until the wrapper exits.
    artifact_paths = [p for p in ROOT.rglob("*") if p.is_file() and p != ROOT / "hashes/artifact_sha256.txt" and p.name != "postprocess.log"]
    final["ARTIFACT_COUNT"] = len(artifact_paths) + 1
    atomic_json(ROOT / "analysis/final_summary.json", final)
    artifact_paths = [p for p in ROOT.rglob("*") if p.is_file() and p != ROOT / "hashes/artifact_sha256.txt" and p.name != "postprocess.log"]
    atomic_text(ROOT / "hashes/artifact_sha256.txt", "\n".join(f"{sha256(p)}  {p.relative_to(ROOT)}" for p in sorted(artifact_paths)) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.self_test == args.final:
        raise SystemExit("choose exactly one of --self-test or --final")
    if args.self_test:
        run_self_test()
    else:
        run_final()


if __name__ == "__main__":
    main()
