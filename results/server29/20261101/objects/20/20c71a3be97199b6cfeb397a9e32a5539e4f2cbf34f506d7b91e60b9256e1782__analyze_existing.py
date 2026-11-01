#!/usr/bin/env python3
"""CPU-only evidence synthesis for QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


OUT = Path("experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1")
LEGACY_ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
AIME = LEGACY_ROOT / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1"
DEPLOY = LEGACY_ROOT / "experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1"
TRAIN_OLD = LEGACY_ROOT / "experiments/QWEN_RECURRENT_DENSE_C5_C6_V1"
TF_REPLAY = LEGACY_ROOT / "experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
BASELINE_ROOT = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/results/aime26/81920/qwen_gdn")
BASELINE_CSV = BASELINE_ROOT / "frozen_v4_per_sample.csv"
SCORER = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/experiments/shared/scoring/aime26_scorer_v4.py")

COND_MAP = {
    "QWEN_FP_STATE": "FP",
    "QWEN_INT8_C128": "Native",
    "QWEN_INT8_C128_KEY_HADAMARD": "H",
}
PAIRS = [
    ("C5_vs_H", "C5", "H", "primary_endpoint_family"),
    ("C6_vs_H", "C6", "H", "primary_endpoint_family"),
    ("C5_vs_FP", "C5", "FP", "primary_endpoint_family"),
    ("C6_vs_FP", "C6", "FP", "primary_endpoint_family"),
    ("H_vs_FP", "H", "FP", "primary_endpoint_family"),
    ("C6_vs_C5", "C6", "C5", "auxiliary_endpoint_family"),
]
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_REPS = 10000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, "")
        return
    tmp = path.with_name(path.name + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def bool_csv(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_endpoint_records() -> dict[str, dict[tuple[str, int], dict]]:
    rows: dict[str, dict[tuple[str, int], dict]] = defaultdict(dict)
    for r in csv.DictReader(BASELINE_CSV.open(encoding="utf-8")):
        c = COND_MAP[r["condition"]]
        key = (r["problem_id"], int(r["seed"]))
        rows[c][key] = {
            "correct": bool_csv(r["v4_correct"]),
            "extracted": r["v4_extracted_answer"],
            "status": r["v4_status"],
            "abstain_reason": r["v4_abstain_reason"],
            "finish_reason": r["finish_reason"],
            "truncated": bool_csv(r["truncated"]),
            "eos_seen": bool_csv(r["eos_seen"]),
            "generated_tokens": int(r["generated_token_count"]),
            "runtime_error": bool_csv(r["runtime_error"]),
            "nonfinite": bool_csv(r["nonfinite"]),
            "gold": int(r["gold_answer"]),
            "source_file": r["source_file"],
        }
    for c in ("C5", "C6"):
        data = json.loads((AIME / f"analysis/{c}_scored.json").read_text(encoding="utf-8"))
        for r in data["records"]:
            key = (r["problem_id"], int(r["seed"]))
            rows[c][key] = {
                "correct": bool(r["v4_correct"]),
                "extracted": r["v4_extracted_answer"],
                "status": r["v4_status"],
                "abstain_reason": r["v4_abstain_reason"],
                "finish_reason": r["finish_reason"],
                "truncated": bool(r["truncated"]),
                "eos_seen": bool(r["eos_seen"]),
                "generated_tokens": int(r["generated_token_count"]),
                "runtime_error": False,
                "nonfinite": False,
                "gold": int(r["gold_answer"]),
                "source_file": str(AIME / "outputs/formal/shards"),
            }
    expected = {(f"aime26_{i:02d}", seed) for i in range(1, 31) for seed in (1, 2)}
    for c in ("FP", "Native", "H", "C5", "C6"):
        if set(rows[c]) != expected:
            raise RuntimeError(f"identity mismatch {c}: {len(rows[c])}, missing={sorted(expected-set(rows[c]))}")
    return rows


def exact_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    result: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, raw[name] * (len(ordered) - rank)))
        result[name] = running
    return result


def quantile(values: list[float], q: float) -> float:
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return v[lo]
    return v[lo] * (hi - pos) + v[hi] * (pos - lo)


def cluster_bootstrap(keys: list[tuple[str, int]], left: dict, right: dict) -> dict:
    by_q: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in keys:
        by_q[key[0]].append(key)
    questions = sorted(by_q)
    observed = statistics.mean(float(left[k]["correct"]) - float(right[k]["correct"]) for k in keys)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = [questions[rng.randrange(len(questions))] for _ in questions]
        vals = [float(left[k]["correct"]) - float(right[k]["correct"]) for q in sampled for k in by_q[q]]
        estimates.append(statistics.mean(vals))
    return {
        "risk_difference_candidate_minus_reference": observed,
        "cluster_bootstrap_ci95": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
        "cluster_unit": "question_id (both seeds resampled together)",
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def category(a: bool, b: bool) -> str:
    if a and b:
        return "both_correct"
    if (not a) and b:
        return "reference_only"
    if a and (not b):
        return "candidate_only"
    return "both_wrong"


def build_paired(rows: dict[str, dict]) -> tuple[list[dict], dict]:
    keys = sorted(rows["FP"])
    wide: list[dict] = []
    summaries: dict[str, dict] = {}
    p_primary: dict[str, float] = {}
    for name, left_name, right_name, family in PAIRS:
        cats = Counter(category(rows[left_name][k]["correct"], rows[right_name][k]["correct"]) for k in keys)
        summary = {
            "comparison": name,
            "candidate": left_name,
            "reference": right_name,
            "family": family,
            "n": len(keys),
            "both_correct": cats["both_correct"],
            "reference_only": cats["reference_only"],
            "candidate_only": cats["candidate_only"],
            "both_wrong": cats["both_wrong"],
            "net_candidate_minus_reference": cats["candidate_only"] - cats["reference_only"],
            "exact_mcnemar_two_sided_p": exact_p(cats["candidate_only"], cats["reference_only"]),
            "sample_ids": {
                c: [f"{k[0]}_seed{k[1]}" for k in keys if category(rows[left_name][k]["correct"], rows[right_name][k]["correct"]) == c]
                for c in ("both_correct", "reference_only", "candidate_only", "both_wrong")
            },
        }
        summary.update(cluster_bootstrap(keys, rows[left_name], rows[right_name]))
        summaries[name] = summary
        if family == "primary_endpoint_family":
            p_primary[name] = summary["exact_mcnemar_two_sided_p"]
    adjusted = holm(p_primary)
    for name, p in adjusted.items():
        summaries[name]["holm_p_across_primary_endpoint_family"] = p
    summaries["C6_vs_C5"]["holm_p_across_primary_endpoint_family"] = None
    for key in keys:
        rec: dict[str, object] = {"question_id": key[0], "seed": key[1], "sample_id": f"{key[0]}_seed{key[1]}", "gold": rows["FP"][key]["gold"]}
        for c in ("FP", "Native", "H", "C5", "C6"):
            r = rows[c][key]
            rec.update({
                f"{c}_correct": r["correct"],
                f"{c}_extracted": r["extracted"],
                f"{c}_status": r["status"],
                f"{c}_finish_reason": r["finish_reason"],
                f"{c}_generated_tokens": r["generated_tokens"],
                f"{c}_abstain": r["status"] == "ABSTAIN",
                f"{c}_length_limit": r["finish_reason"] == "length",
                f"{c}_runtime_error": r["runtime_error"],
                f"{c}_nonfinite": r["nonfinite"],
            })
        for name, left_name, right_name, _ in PAIRS:
            rec[f"{name}_category"] = category(rows[left_name][key]["correct"], rows[right_name][key]["correct"])
        rec["four_way_pattern_FP_H_C5_C6"] = "".join("1" if rows[c][key]["correct"] else "0" for c in ("FP", "H", "C5", "C6"))
        wide.append(rec)
    return wide, summaries


def import_scorer():
    spec = importlib.util.spec_from_file_location("frozen_v4", SCORER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ngram_metrics(tokens: list[int], n: int) -> tuple[float, float]:
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0.0, 0.0
    counts = Counter(tuple(tokens[i:i+n]) for i in range(total))
    return sum(v - 1 for v in counts.values()) / total, len(counts) / total


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
    out = {
        "condition": condition,
        "problem_id": row["problem_id"],
        "seed": int(row["seed"]),
        "generated_tokens": len(tokens),
        "length_ceiling": row["finish_reason"] == "length",
        "termination_type": row["finish_reason"],
        "committed_claim_count": len(committed),
        "no_committed_claim": len(committed) == 0,
        "post_commit_fraction": None if first is None or not text_len else (text_len - int(first.end)) / text_len,
        "answer_changes": changes,
        "retractions": retractions,
        "oscillation_proxy": changes + aba + retractions,
    }
    for n in (4, 8, 16):
        repeated, unique = ngram_metrics(tokens, n)
        out[f"repeated_{n}gram_fraction"] = repeated
        out[f"unique_{n}gram_ratio"] = unique
    return out


def load_behavior(endpoint_rows: dict[str, dict]) -> dict[str, dict[tuple[str, int], dict]]:
    by_cond: dict[str, dict[tuple[str, int], dict]] = defaultdict(dict)
    for line in (AIME / "analysis/trajectory_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["no_committed_claim"] = int(r["committed_claim_count"]) == 0
            by_cond[r["condition"]][(r["problem_id"], int(r["seed"]))] = r
    scorer = import_scorer()
    fp_paths = sorted({Path(r["source_file"]) for r in endpoint_rows["FP"].values()})
    raw_fp: dict[tuple[str, int], dict] = {}
    for path in fp_paths:
        for r in load_jsonl(path):
            if r.get("method") == "QWEN_FP_STATE" or r.get("condition") == "QWEN_FP_STATE":
                raw_fp[(r["problem_id"], int(r["seed"]))] = r
    if set(raw_fp) != set(endpoint_rows["FP"]):
        raise RuntimeError(f"FP raw identity mismatch: {len(raw_fp)}")
    for key, r in raw_fp.items():
        by_cond["FP"][key] = trajectory_record("FP", r, scorer)
    return by_cond


def median(values) -> float | None:
    v = [float(x) for x in values if x is not None]
    return statistics.median(v) if v else None


def build_behavior(endpoint_rows: dict[str, dict], paired: dict) -> list[dict]:
    behavior = load_behavior(endpoint_rows)
    metrics = [
        "generated_tokens", "length_ceiling", "no_committed_claim", "post_commit_fraction",
        "answer_changes", "retractions", "oscillation_proxy", "repeated_4gram_fraction",
        "repeated_8gram_fraction", "repeated_16gram_fraction", "unique_4gram_ratio",
        "unique_8gram_ratio", "unique_16gram_ratio",
    ]
    out: list[dict] = []
    keys = sorted(endpoint_rows["FP"])
    for name, left_name, right_name, family in PAIRS:
        for cat in ("both_correct", "reference_only", "candidate_only", "both_wrong"):
            selected = [k for k in keys if category(endpoint_rows[left_name][k]["correct"], endpoint_rows[right_name][k]["correct"]) == cat]
            row: dict[str, object] = {
                "comparison": name, "family": family, "category": cat, "n": len(selected),
                "sample_ids": ";".join(f"{k[0]}_seed{k[1]}" for k in selected),
            }
            for m in metrics:
                lv = [behavior[left_name][k].get(m) for k in selected]
                rv = [behavior[right_name][k].get(m) for k in selected]
                diffs = [float(a) - float(b) for a, b in zip(lv, rv) if a is not None and b is not None]
                row[f"candidate_{m}_median"] = median(lv)
                row[f"reference_{m}_median"] = median(rv)
                row[f"paired_delta_candidate_minus_reference_{m}_median"] = median(diffs)
            out.append(row)
    return out


def sylvester_hadamard(n: int = 128) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def build_rotation_geometry() -> list[dict]:
    H = sylvester_hadamard()
    out = []
    for condition in ("C5", "C6"):
        path = DEPLOY / f"deployment/{condition}/frozen_rotation_fp32.pt"
        payload = torch.load(path, map_location="cpu")
        mats = payload.get("rotations", payload.get("matrices", payload)) if isinstance(payload, dict) else payload
        if isinstance(mats, list):
            iterator = enumerate(mats)
        else:
            iterator = sorted(((int(k), v) for k, v in mats.items()), key=lambda x: x[0])
        for layer, mat in iterator:
            R = mat.detach().to(torch.float64)
            eye = torch.eye(R.shape[0], dtype=torch.float64)
            out.append({
                "condition": condition,
                "layer": int(layer),
                "finite": bool(torch.isfinite(R).all()),
                "orthogonality_max_abs": float((R.T @ R - eye).abs().max()),
                "orthogonality_fro": float(torch.linalg.norm(R.T @ R - eye)),
                "fro_distance_to_hadamard": float(torch.linalg.norm(R - H)),
                "relative_fro_distance_to_hadamard": float(torch.linalg.norm(R - H) / torch.linalg.norm(H)),
                "max_abs_delta_to_hadamard": float((R - H).abs().max()),
            })
    return out


def inventory() -> list[dict]:
    entries = [
        (BASELINE_CSV, "canonical60_baselines", "FP/Native/Hadamard", "60 each", "free-generation condition-specific", "canonical recurrent state per condition", "formal endpoints, identity, finish/token metadata", "no state/logits; behavior text requires raw source"),
        (AIME / "analysis/C5_scored.json", "canonical60_C5_C6", "C5", "60", "C5 free-generation", "canonical C5 persistent recurrence", "Frozen V4 endpoint labels", "no recurrent state/logits"),
        (AIME / "analysis/C6_scored.json", "canonical60_C5_C6", "C6", "60", "C6 free-generation", "canonical C6 persistent recurrence", "Frozen V4 endpoint labels", "no recurrent state/logits"),
        (AIME / "analysis/trajectory_metrics.jsonl", "canonical60_C5_C6", "C5/C6/H", "60 each", "own free-generation tokens", "condition-specific free-generation histories", "descriptive length/repetition/claim/oscillation metrics", "not causal; no FP rows in source table"),
        (AIME / "analysis/paired_summary.json", "canonical60_C5_C6", "C5/C6/H", "3 frozen pairs", "endpoint labels", "N/A", "formal paired results for preregistered pairs", "does not include FP pairs"),
        (DEPLOY / "deployment/C5/frozen_rotation_fp32.pt", "deployment_closure", "C5", "24 layers", "N/A", "N/A", "canonical deployed matrices", "matrix geometry is descriptive, not performance"),
        (DEPLOY / "deployment/C6/frozen_rotation_fp32.pt", "deployment_closure", "C6", "24 layers", "N/A", "N/A", "canonical deployed matrices", "matrix geometry is descriptive, not performance"),
        (DEPLOY / "deployment/C5/deployment_manifest.json", "deployment_closure", "C5", "manifest", "N/A", "N/A", "checkpoint/frozen-R provenance and hashes", "formal training logs/configs not bundled"),
        (DEPLOY / "deployment/C6/deployment_manifest.json", "deployment_closure", "C6", "manifest", "N/A", "N/A", "checkpoint/frozen-R provenance and hashes", "formal training logs/configs not bundled"),
        (TF_REPLAY / "final_report.md", "teacher_forced_replay", "FP/Native/Hadamard", "60 FP token paths", "existing FP token sequences", "condition-specific persistent replay", "historical H/Native state/function mechanism hypotheses", "no C5/C6; descriptive/predictive, not direct learned result"),
        (TF_REPLAY / "correlation_analysis.json", "teacher_forced_replay", "Native/Hadamard relative FP", "60", "existing FP token sequences", "persistent replay", "historical AUROC/correlation summaries", "no C5/C6"),
        (TRAIN_OLD / "configs/recurrent_training_config_c5.json", "legacy_recurrent_training", "C5", "config", "non-AIME calibration docs", "continuous student, resource-selected historical horizon", "legacy objective/hyperparameter provenance", "not the final H64 formal experiment"),
        (TRAIN_OLD / "configs/recurrent_training_config_c6.json", "legacy_recurrent_training", "C6", "config", "non-AIME calibration docs", "continuous student, resource-selected historical horizon", "legacy objective/hyperparameter provenance", "not the final H64 formal experiment"),
        (SCORER, "canonical_scoring", "all", "Frozen V4", "condition output", "N/A", "formal correctness and claim event definitions", "not a mechanism metric"),
    ]
    out = []
    for path, experiment, condition, coverage, token_source, semantics, supports, limitations in entries:
        out.append({
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else "MISSING",
            "bytes": path.stat().st_size if path.is_file() else 0,
            "experiment": experiment,
            "condition": condition,
            "coverage": coverage,
            "token_source": token_source,
            "state_history_semantics": semantics,
            "supports": supports,
            "limitations": limitations,
        })
    for condition in ("C5", "C6"):
        out.extend([
            {"path": f"MISSING::{condition}_formal_H64_training_config_and_logs", "exists": False, "sha256": "MISSING", "bytes": 0, "experiment": "QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1", "condition": condition, "coverage": "MISSING_ON_ANALYSIS_HOST", "token_source": "training documents", "state_history_semantics": "expected H64 continuous numerical recurrence", "supports": "would support exact training/validation trajectory audit", "limitations": "only deployment manifest and lower-priority handoff remain locally"},
            {"path": f"MISSING::{condition}_recurrent_state_cache_logits", "exists": False, "sha256": "MISSING", "bytes": 0, "experiment": "canonical60_C5_C6", "condition": condition, "coverage": "MISSING", "token_source": "N/A", "state_history_semantics": "N/A", "supports": "would support learned recurrent drift/function attribution", "limitations": "cannot be reconstructed from response text"},
        ])
    return out


def main() -> None:
    for d in (OUT / "analysis", OUT / "reports", OUT / "hashes", OUT / "artifacts"):
        d.mkdir(parents=True, exist_ok=True)
    endpoints = load_endpoint_records()
    paired_rows, paired_summary = build_paired(endpoints)
    behavior_rows = build_behavior(endpoints, paired_summary)
    geometry = build_rotation_geometry()
    write_csv(OUT / "artifact_inventory.csv", inventory())
    write_csv(OUT / "paired_outcomes.csv", paired_rows)
    write_csv(OUT / "behavior_gap_analysis.csv", behavior_rows)
    write_csv(OUT / "analysis/rotation_geometry.csv", geometry)
    atomic_json(OUT / "analysis/paired_summary_extended.json", {
        "endpoint_counts": {c: sum(r["correct"] for r in endpoints[c].values()) for c in ("FP", "Native", "H", "C5", "C6")},
        "comparisons": paired_summary,
        "multiplicity": {
            "primary_endpoint_family": [p[0] for p in PAIRS if p[3] == "primary_endpoint_family"],
            "method": "Holm adjustment applied once to exact paired-binomial p-values",
            "auxiliary_endpoint_family": ["C6_vs_C5"],
            "historical_formal_results_preserved": True,
        },
    })
    atomic_json(OUT / "analysis/evidence_analysis_status.json", {
        "EXISTING_EVIDENCE_ANALYSIS": "COMPLETE",
        "FULL_DENSITY_FULL60": "CANCELLED_BY_USER",
        "NEW_FREE_GENERATION": "NOT_STARTED",
        "NEW_TRAINING": "NOT_STARTED",
        "formal_endpoint_fact": {"FP": "52/60", "Native": "18/60", "H": "41/60", "C5": "33/60", "C6": "36/60"},
        "new_analysis": "five-pair endpoint family plus auxiliary C6-vs-C5; question-cluster bootstrap; descriptive behavior groups",
    })
    print(json.dumps({"out": str(OUT), "rows": len(paired_rows), "behavior_rows": len(behavior_rows), "geometry_rows": len(geometry)}, indent=2))


if __name__ == "__main__":
    main()
