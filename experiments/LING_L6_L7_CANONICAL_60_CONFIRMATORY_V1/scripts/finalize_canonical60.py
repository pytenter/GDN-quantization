#!/usr/bin/env python3
"""Finalize LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1 from frozen artifacts only."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONDITIONS = ("H", "L6", "L7")
COMPARISONS = (("L7", "H"), ("L6", "H"), ("L7", "L6"))
EXPECTED_OLD = {"H": 9, "L6": 9, "L7": 10, "FP": 14}
EXPECTED_NEW = {"H": 26, "L6": 27, "L7": 23}
EXPECTED_ROT = {
    "L6": "ca6ef0d398d0b030dd5c295f62b8c0f43e11d41b92b319bd750ffa064977d71a",
    "L7": "57eaebb47ff031c2b171cbb94563f81481d4f4632098af8339a27d01347060a2",
}
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_REPLICATES = 10_000
TRAJECTORY_METRICS = (
    "generated_tokens", "repeated_4gram_fraction",
    "first_committed_claim_position", "num_answer_changes",
    "retracted_event_count",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    os.replace(tmp, path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bool_csv(value: Any) -> bool:
    return str(value).lower() == "true"


def int_or_none(value: Any) -> int | None:
    if value in (None, "", "None", "NOT_AVAILABLE"):
        return None
    return int(float(value))


def float_or_none(value: Any) -> float | None:
    if value in (None, "", "None", "NOT_AVAILABLE"):
        return None
    return float(value)


def exact_p(rescue: int, regress: int) -> float:
    n = rescue + regress
    if n == 0:
        return 1.0
    k = min(rescue, regress)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda x: x[1])
    out: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def paired_summary(deltas: list[float], rng: random.Random) -> dict[str, Any]:
    if not deltas:
        return {"n": 0, "status": "NOT_AVAILABLE"}
    n = len(deltas)
    boot = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_REPLICATES)]
    return {
        "n": n,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "positive": sum(x > 0 for x in deltas),
        "zero": sum(x == 0 for x in deltas),
        "negative": sum(x < 0 for x in deltas),
        "bootstrap_mean_delta_95pct_ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def analyze_raw(path: Path, condition: str, scorer: Any, helper: Any) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("condition") != condition or raw.get("sample_id") != path.stem:
        raise RuntimeError(f"identity mismatch: {path}")
    tokens = [int(x) for x in raw["generated_token_ids"]]
    if len(tokens) != int(raw["generated_tokens"]):
        raise RuntimeError(f"token count mismatch: {path}")
    if not raw.get("successful_sample") or not raw.get("token_provenance_valid") or raw.get("nonfinite") or raw.get("runtime_error") is not None:
        raise RuntimeError(f"runtime/scientific artifact gate failed: {path}")
    infra = raw.get("infrastructure_failures", [])
    infra_count = len(infra) if isinstance(infra, list) else int(infra or 0)
    if int(raw.get("retry_count", -1)) != 0 or infra_count != 0:
        raise RuntimeError(f"retry/infra gate failed: {path}")
    if raw.get("scorer_sha256") != "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b":
        raise RuntimeError(f"scorer hash mismatch: {path}")
    if condition in EXPECTED_ROT and raw.get("final_rotation_sha256") != EXPECTED_ROT[condition]:
        raise RuntimeError(f"rotation hash mismatch: {path}")
    if int(raw.get("context_length", 0)) != 262144 or int(raw.get("safety_margin", 0)) != 512:
        raise RuntimeError(f"budget identity mismatch: {path}")
    if int(raw["max_new_tokens"]) != 262144 - int(raw["prompt_tokens"]) - 512:
        raise RuntimeError(f"dynamic max_new_tokens mismatch: {path}")

    text = raw["decoded_output"]
    result = scorer.extract_v4(text, raw.get("finish_reason"), raw.get("problem", ""))
    recomputed_correct = bool(scorer.compare_with_gold(result, raw.get("gold")))
    if recomputed_correct != bool(raw["v4_correct"]) or result.normalized_value != raw.get("v4_extracted_answer"):
        raise RuntimeError(f"Frozen V4 rescore mismatch: {path}")
    normalized, events, committed = helper.scorer_events(scorer, text, raw.get("finish_reason"), raw.get("problem", ""))
    values = [e.normalized_value for e in committed]
    compressed: list[Any] = []
    for value in values:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    first = committed[0] if committed else None
    first_pos = helper.char_to_token(first.start, len(normalized), len(tokens)) if first else None
    ftype = helper.finish_type(raw.get("finish_reason"))
    hit_max = bool(raw.get("hit_context_limit")) or (ftype == "length" and len(tokens) >= int(raw["max_new_tokens"]))
    return {
        "sample_id": raw["sample_id"], "question_id": raw["question_id"], "seed": int(raw["seed"]),
        "condition": condition, "gold_answer": int(raw["gold"]), "final_answer": raw.get("v4_extracted_answer"),
        "correct": bool(raw["v4_correct"]), "abstain": bool(raw["v4_abstain"]),
        "generated_tokens": len(tokens), "hit_max_length": hit_max,
        "termination_category": "LENGTH_CEILING" if hit_max else ("EXPLICIT_EOS_OR_STOP" if ftype == "stop" else ftype.upper()),
        "explicit_eos": ftype == "stop", "first_committed_claim_position": first_pos,
        "never_formed_stable_commitment": len(committed) == 0,
        "num_answer_changes": max(0, len(compressed) - 1),
        "retracted_event_count": sum(bool(e.retracted) for e in events),
        "repeated_4gram_fraction": helper.repeated_fraction(tokens, 4),
        "source_path": str(path), "source_file_sha256": sha256(path),
        "decoded_output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "hardware": raw["hardware_class"], "host": raw["host"], "physical_gpu": int(raw["physical_gpu"]),
        "instance_id": raw["instance_id"], "prompt_hash": raw["input_ids_hash"],
        "config_hash": raw["frozen_config_sha256"], "rotation_sha256": raw.get("final_rotation_sha256"),
        "scorer_sha256": raw["scorer_sha256"], "retry_count": int(raw["retry_count"]),
        "infrastructure_failures": infra_count,
        "token_provenance_valid": True, "successful_sample": True,
        "official_v4_result": {k: v for k, v in raw["v4_result"].items() if k != "candidate_rejections"},
        "scorer_recomputed_result": {k: v for k, v in dataclasses.asdict(result).items() if k != "candidate_rejections"},
    }


def load_old(old: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    outcomes = list(csv.DictReader((old / "analysis/per_sample_outcomes.csv").open(encoding="utf-8")))
    traj = list(csv.DictReader((old / "analysis/trajectory_metrics.csv").open(encoding="utf-8")))
    traj_idx = {(r["condition"], r["question_id"]): r for r in traj}
    fp = {}
    rows = []
    for r in outcomes:
        c, q = r["condition"], r["question_id"]
        if c == "FP":
            fp[f"{q}_seed1"] = r
            continue
        if c not in CONDITIONS:
            continue
        t = traj_idx[(c, q)]
        rows.append({
            "sample_id": f"{q}_seed1", "question_id": q, "seed": 1, "condition": c,
            "gold_answer": int(r["gold_answer"]), "final_answer": int_or_none(r["final_answer"]),
            "correct": bool_csv(r["correct"]), "abstain": bool_csv(r["abstain"]),
            "generated_tokens": int(r["generated_tokens"]), "hit_max_length": bool_csv(r["hit_max_length"]),
            "termination_category": r["termination_category"], "explicit_eos": bool_csv(r["explicit_eos"]),
            "first_committed_claim_position": int_or_none(t["first_committed_claim_position"]),
            "never_formed_stable_commitment": bool_csv(t["never_formed_stable_commitment"]),
            "num_answer_changes": int(t["num_answer_changes"]),
            "retracted_event_count": int(t["retracted_event_count"]),
            "repeated_4gram_fraction": float_or_none(t["repeated_4gram_fraction"]),
            "source_path": "frozen20:" + q, "source_file_sha256": r["source_file_sha256"],
            "decoded_output_sha256": r["decoded_output_sha256"], "hardware": r["hardware"],
            "host": "frozen20 archived provenance", "physical_gpu": int(r["physical_gpu"]),
            "instance_id": r["instance_id"], "prompt_hash": None, "config_hash": None,
            "rotation_sha256": None, "scorer_sha256": "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b",
            "retry_count": 0, "infrastructure_failures": 0,
            "token_provenance_valid": bool_csv(r["token_provenance_valid"]),
            "successful_sample": bool_csv(r["successful_sample"]),
        })
    return rows, fp


def direction_label(metric: str, mean: float) -> str:
    if mean == 0:
        return "NO_MEAN_DIFFERENCE"
    favorable_negative = metric in {"generated_tokens", "repeated_4gram_fraction", "first_committed_claim_position", "num_answer_changes", "retracted_event_count"}
    return ("LOWER_IN_LEFT" if mean < 0 else "HIGHER_IN_LEFT") + ("_DESCRIPTIVE_ONLY" if favorable_negative else "")


def main() -> None:
    exp = Path(__file__).resolve().parent.parent
    repo = exp.parent.parent
    old = repo / "experiments/LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1"
    helper = load_module("trajectory_closure_helpers", old / "scripts/run_analysis.py")
    scorer = load_module("frozen_v4_canonical60", exp / "scripts/aime26_scorer_v4.py")
    manifest = json.loads((exp / "configs/canonical_sample_manifest.json").read_text())
    expected_ids = [r["sample_id"] for r in manifest["samples"]]
    missing_ids = set(manifest["missing_prospective_sample_ids"])
    existing_ids = set(manifest["existing_frozen_sample_ids"])
    if len(expected_ids) != 60 or set(expected_ids) != missing_ids | existing_ids or missing_ids & existing_ids:
        raise RuntimeError("canonical sample manifest partition failure")
    if sha256(exp / "scripts/aime26_scorer_v4.py") != "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b":
        raise RuntimeError("Frozen V4 source hash mismatch")

    old_rows, fp = load_old(old)
    rows = list(old_rows)
    for condition in CONDITIONS:
        files = sorted((exp / "outputs" / condition).glob("*.json"))
        if len(files) != 40:
            raise RuntimeError(f"{condition}: expected 40 new outputs, got {len(files)}")
        rows.extend(analyze_raw(p, condition, scorer, helper) for p in files)
    by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        key = r["sample_id"]
        if key in by[r["condition"]]:
            raise RuntimeError(f"duplicate condition/sample {r['condition']} {key}")
        by[r["condition"]][key] = r
    for c in CONDITIONS:
        if set(by[c]) != set(expected_ids):
            raise RuntimeError(f"{c} canonical60 coverage failure")
        old_score = sum(by[c][x]["correct"] for x in existing_ids)
        new_score = sum(by[c][x]["correct"] for x in missing_ids)
        if old_score != EXPECTED_OLD[c] or new_score != EXPECTED_NEW[c]:
            raise RuntimeError(f"{c} frozen score mismatch old={old_score} new={new_score}")

    assignment_doc = json.loads((exp / "configs/hardware_assignment.json").read_text())
    assignment = {r["sample_id"]: r for r in assignment_doc["assignments"]}
    if set(assignment) != missing_ids:
        raise RuntimeError("assignment coverage mismatch")
    for c in CONDITIONS:
        for sid in missing_ids:
            r, a = by[c][sid], assignment[sid]
            if r["hardware"] != a["hardware_class"] or r["host"] != a["host"] or r["physical_gpu"] != a["physical_gpu"]:
                raise RuntimeError(f"assignment mismatch {c} {sid}")

    outcome_rows = []
    for sid in expected_ids:
        h, l6, l7 = by["H"][sid], by["L6"][sid], by["L7"][sid]
        if len({h["gold_answer"], l6["gold_answer"], l7["gold_answer"]}) != 1:
            raise RuntimeError(f"gold mismatch {sid}")
        a = assignment.get(sid)
        fpr = fp.get(sid)
        outcome_rows.append({
            "sample_id": sid, "question_id": h["question_id"], "seed": h["seed"], "gold": h["gold_answer"],
            "FP_status": "AVAILABLE_FROZEN20" if fpr else "NOT_AVAILABLE_NO_NEW_FP_AUTHORIZED",
            "FP_correct": bool_csv(fpr["correct"]) if fpr else None,
            "H_extracted_answer": h["final_answer"], "H_outcome": "CORRECT" if h["correct"] else ("ABSTAIN" if h["abstain"] else "WRONG"),
            "L6_extracted_answer": l6["final_answer"], "L6_outcome": "CORRECT" if l6["correct"] else ("ABSTAIN" if l6["abstain"] else "WRONG"),
            "L7_extracted_answer": l7["final_answer"], "L7_outcome": "CORRECT" if l7["correct"] else ("ABSTAIN" if l7["abstain"] else "WRONG"),
            "hardware_assignment": a["hardware_class"] + f" GPU{a['physical_gpu']}" if a else "FROZEN20_ARCHIVED_ASSIGNMENT",
            "H_output_sha256": h["source_file_sha256"], "L6_output_sha256": l6["source_file_sha256"], "L7_output_sha256": l7["source_file_sha256"],
        })
    write_csv(exp / "analysis/canonical60_outcomes.csv", outcome_rows)

    pair_stats: dict[str, Any] = {}
    pair_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    pvals = {}
    for left, right in COMPARISONS:
        name = f"{left}_vs_{right}"
        prs, rescue_ids, regress_ids = [], [], []
        for sid in expected_ids:
            a, b = by[left][sid], by[right][sid]
            category = "same_correct" if a["correct"] and b["correct"] else "same_wrong" if not a["correct"] and not b["correct"] else "rescue" if a["correct"] else "regress"
            if category == "rescue": rescue_ids.append(sid)
            if category == "regress": regress_ids.append(sid)
            prs.append({"sample_id": sid, f"{left}_correct": a["correct"], f"{right}_correct": b["correct"], "category": category})
        p = exact_p(len(rescue_ids), len(regress_ids))
        pvals[name] = p
        pair_stats[name] = {"paired_samples": 60, "rescue": len(rescue_ids), "regress": len(regress_ids), "net": len(rescue_ids)-len(regress_ids), "exact_two_sided_p": p, "rescue_sample_ids": rescue_ids, "regress_sample_ids": regress_ids}
        pair_rows_by_name[name] = prs
        write_csv(exp / f"analysis/paired_{name}.csv", prs)
    adjusted = holm(pvals)
    for name in pair_stats:
        pair_stats[name]["holm_adjusted_p"] = adjusted[name]

    scores = {c: sum(r["correct"] for r in by[c].values()) for c in CONDITIONS}
    endpoint = {
        "gate": "PASS", "primary_endpoint": "Frozen V4 correctness", "canonical_samples": 60,
        "scores": {c: {"correct": scores[c], "total": 60, "accuracy": scores[c]/60} for c in CONDITIONS},
        "FP": {"status": "CANONICAL_AVAILABLE_SUBSET_ONLY", "correct": EXPECTED_OLD["FP"], "total": 20, "new_generation": False},
        "paired_comparisons": pair_stats, "holm_scope": "three frozen paired endpoint comparisons",
    }
    atomic_json(exp / "analysis/endpoint_statistics.json", endpoint)

    trajectory_rows = []
    for c in CONDITIONS:
        for sid in expected_ids:
            r = by[c][sid]
            trajectory_rows.append({k: r.get(k) for k in ("sample_id", "question_id", "seed", "condition", "generated_tokens", "hit_max_length", "termination_category", "explicit_eos", "abstain", "first_committed_claim_position", "never_formed_stable_commitment", "num_answer_changes", "retracted_event_count", "repeated_4gram_fraction")})
    write_csv(exp / "analysis/trajectory_metrics.csv", trajectory_rows)
    rng = random.Random(BOOTSTRAP_SEED)
    traj_stats = {"role": "secondary/exploratory", "sign": "left condition minus right condition", "comparisons": {}}
    for left, right in COMPARISONS:
        name = f"{left}_vs_{right}"
        metrics = {}
        for metric in TRAJECTORY_METRICS:
            deltas = [float(by[left][sid][metric] - by[right][sid][metric]) for sid in expected_ids if by[left][sid].get(metric) is not None and by[right][sid].get(metric) is not None]
            metrics[metric] = paired_summary(deltas, rng)
            if deltas:
                metrics[metric]["direction"] = direction_label(metric, metrics[metric]["mean_delta"])
        for metric in ("hit_max_length", "never_formed_stable_commitment", "abstain"):
            deltas = [int(bool(by[left][sid][metric])) - int(bool(by[right][sid][metric])) for sid in expected_ids]
            metrics[metric] = paired_summary([float(x) for x in deltas], rng)
        metrics["termination_transition_counts"] = dict(sorted(__import__("collections").Counter(f"{by[right][sid]['termination_category']} -> {by[left][sid]['termination_category']}" for sid in expected_ids).items()))
        traj_stats["comparisons"][name] = metrics
    atomic_json(exp / "analysis/trajectory_statistics.json", traj_stats)

    rescue_analysis = {"role": "descriptive association; not causal mechanism", "comparisons": {}}
    for left, right in (("L7", "H"), ("L6", "H")):
        name = f"{left}_vs_{right}"
        groups = {}
        for category in ("rescue", "regress"):
            ids = [r["sample_id"] for r in pair_rows_by_name[name] if r["category"] == category]
            g = {"n": len(ids), "sample_ids": ids}
            for metric in TRAJECTORY_METRICS:
                ds = [float(by[left][sid][metric] - by[right][sid][metric]) for sid in ids if by[left][sid].get(metric) is not None and by[right][sid].get(metric) is not None]
                g[metric] = {"n": len(ds), "mean_delta": statistics.fmean(ds) if ds else None, "median_delta": statistics.median(ds) if ds else None}
            g["termination_transitions"] = dict(sorted(__import__("collections").Counter(f"{by[right][sid]['termination_category']} -> {by[left][sid]['termination_category']}" for sid in ids).items()))
            g["length_ceiling_delta_counts"] = dict(sorted(__import__("collections").Counter(str(int(by[left][sid]["hit_max_length"])-int(by[right][sid]["hit_max_length"])) for sid in ids).items()))
            g["stable_commitment_delta_counts"] = dict(sorted(__import__("collections").Counter(str(int(not by[left][sid]["never_formed_stable_commitment"])-int(not by[right][sid]["never_formed_stable_commitment"])) for sid in ids).items()))
            groups[category] = g
        rescue_analysis["comparisons"][name] = groups
    atomic_json(exp / "analysis/rescue_regression_analysis.json", rescue_analysis)

    old_rotation = json.loads((old / "analysis/rotation_sanity.json").read_text())
    old_rotation["confirmatory_reuse"] = {"gate": "PASS", "L6_expected_sha256": EXPECTED_ROT["L6"], "L7_expected_sha256": EXPECTED_ROT["L7"], "note": "Same archived final rotations; sanity-only metrics reused and identities rechecked on every new output."}
    for c in ("L6", "L7"):
        if old_rotation["conditions"][c]["file_sha256"] != EXPECTED_ROT[c]:
            raise RuntimeError(f"archived rotation sanity identity mismatch: {c}")
    atomic_json(exp / "analysis/rotation_sanity.json", old_rotation)

    gpu_samples = []
    for c in CONDITIONS:
        for sid in expected_ids:
            r = by[c][sid]
            gpu_samples.append({k: r.get(k) for k in ("sample_id", "question_id", "seed", "condition", "host", "hardware", "physical_gpu", "instance_id", "source_file_sha256", "prompt_hash", "config_hash", "rotation_sha256", "scorer_sha256", "generated_tokens", "retry_count", "infrastructure_failures")})
    atomic_json(exp / "analysis/per_sample_gpu_metadata.json", {"samples": gpu_samples, "new40_hardware_counts_per_condition": {"RTX4090": 20, "RTX3090": 20}, "new40_total_condition_outputs": {"RTX4090": 60, "RTX3090": 60}})

    primary = pair_stats["L7_vs_H"]
    l7_label = "NEGATIVE_OR_MIXED" if scores["L7"] < scores["H"] else "NOT_CONFIRMED" if scores["L7"] == scores["H"] else "SUPPORTIVE" if primary["holm_adjusted_p"] < 0.05 and primary["rescue"] > primary["regress"] else "SUPPORTIVE_BUT_UNDERPOWERED"
    l6p = pair_stats["L6_vs_H"]
    l6_label = "NEGATIVE_OR_MIXED" if scores["L6"] < scores["H"] else "NOT_CONFIRMED" if scores["L6"] == scores["H"] else "SUPPORTIVE" if l6p["holm_adjusted_p"] < 0.05 else "SUPPORTIVE_BUT_UNDERPOWERED"
    labels = {
        "PROVENANCE_GATE": "PASS", "CANONICAL_60_COMPLETENESS_GATE": "PASS", "SCORER_GATE": "PASS", "ROTATION_IDENTITY_GATE": "PASS", "INT8_HARDWARE_PARITY_GATE": "PASS_EXISTING_FROZEN_EVIDENCE",
        "FP_CANONICAL_COMPLETENESS": "20/60_CANONICAL_AVAILABLE_SUBSET_ONLY_NO_NEW_FP_AUTHORIZED",
        "L7_ENDPOINT_SIGNAL": l7_label, "L6_ENDPOINT_SIGNAL": l6_label,
        "L7_STRUCTURED_TRAJECTORY_SIGNAL": "INCONCLUSIVE_DESCRIPTIVE_ONLY", "L6_STRUCTURED_TRAJECTORY_SIGNAL": "INCONCLUSIVE_DESCRIPTIVE_ONLY",
        "CURRENT_RESULT_INTERPRETATION": "L7_PRIMARY_NEGATIVE_OR_MIXED; L6_SECONDARY_SUPPORTIVE_BUT_UNDERPOWERED",
        "NEXT_ACTION": "REASSESS_LEARNED_ROTATION_OBJECTIVE",
    }
    atomic_json(exp / "analysis/final_labels.json", labels)

    lines = ["# LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1 Final Report", "", "## Canonical endpoint results", "", "| Condition | Correct / 60 | Accuracy |", "|---|---:|---:|", f"| FP | {EXPECTED_OLD['FP']}/20 available subset | 70.00% on subset |", f"| Hadamard | {scores['H']}/60 | {scores['H']/60:.2%} |", f"| L6 | {scores['L6']}/60 | {scores['L6']/60:.2%} |", f"| L7 | {scores['L7']}/60 | {scores['L7']/60:.2%} |", "", "FP is a reference-only frozen canonical subset. No mixed-hardware FP generation was performed.", "", "## Paired endpoint comparisons", "", "| Comparison | Rescue | Regress | Net | Exact p | Holm p |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("L7_vs_H", "L6_vs_H", "L7_vs_L6"):
        p = pair_stats[name]
        lines.append(f"| {name.replace('_', ' ')} | {p['rescue']} | {p['regress']} | {p['net']:+d} | {p['exact_two_sided_p']:.8g} | {p['holm_adjusted_p']:.8g} |")
        lines.append(f"\n- {name} rescues: {', '.join(p['rescue_sample_ids']) or 'none'}; regressions: {', '.join(p['regress_sample_ids']) or 'none'}.")
    lines += ["", "## Trajectory summaries", "", "All deltas are left minus right and are exploratory. Global shortening or repetition reduction was not a preregistered success criterion.", "", "| Comparison | Metric | Mean Δ | Median Δ | 95% paired-bootstrap CI | Direction |", "|---|---|---:|---:|---|---|"]
    for name in ("L7_vs_H", "L6_vs_H", "L7_vs_L6"):
        for metric in TRAJECTORY_METRICS:
            x = traj_stats["comparisons"][name][metric]
            lines.append(f"| {name.replace('_', ' ')} | {metric} | {x['mean_delta']:.6g} | {x['median_delta']:.6g} | [{x['bootstrap_mean_delta_95pct_ci'][0]:.6g}, {x['bootstrap_mean_delta_95pct_ci'][1]:.6g}] | {x['direction']} |")
    lines += ["", "Rescue/regression termination, length-ceiling, stable-commitment, answer-change, retraction, and repetition summaries are in `analysis/rescue_regression_analysis.json`. They are descriptive associations and do not establish an internal causal mechanism.", "", "## Gates and interpretation", ""]
    lines += [f"- `{k} = {v}`" for k, v in labels.items()]
    lines += ["", "## Resource and audit metadata", "", "- New missing40 assignment per condition: 20 samples on the Ling 2×RTX4090 server and 20 samples on the independent Ling 8×RTX3090 server using only GPU3/GPU4.", "- Across H/L6/L7 new outputs: 60 condition outputs on RTX4090 and 60 condition outputs on RTX3090.", "- 4090 GPU count used: 2. 3090 GPU count used: 2.", "- Retries: 0. Infrastructure failures recorded in canonical outputs: 0.", "- Qwen 4×RTX3090 used: NO. 48GB vGPU used: NO.", "- Protocol: 262,144 total context, dynamic `max_new_tokens = 262144 - prompt_tokens - 512`, Frozen V4, seed 1/2, INT8-R128.", "- Branch: `exp/ling-l6-l7-canonical-60-confirmatory-v1`.", "- Frozen preregistration commit: `69b23bd999dc23a9c852497b98df0f56abdbdff0`.", "", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}", ""]
    atomic_text(exp / "reports/FINAL_REPORT.md", "\n".join(lines))

    inventory = []
    for path in sorted(p for p in exp.rglob("*") if p.is_file() and p != exp / "hashes/SHA256SUMS"):
        if ".tmp" not in path.name and "__pycache__" not in path.parts:
            inventory.append(f"{sha256(path)}  {path.relative_to(exp).as_posix()}")
    atomic_text(exp / "hashes/SHA256SUMS", "\n".join(inventory) + "\n")
    summary = {"gate": "PASS", "scores": endpoint["scores"], "pairwise": pair_stats, "labels": labels, "artifact_hash_entries": len(inventory), "new40_total_generated_tokens": {c: sum(by[c][sid]["generated_tokens"] for sid in missing_ids) for c in CONDITIONS}}
    atomic_json(exp / "analysis/finalization_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
