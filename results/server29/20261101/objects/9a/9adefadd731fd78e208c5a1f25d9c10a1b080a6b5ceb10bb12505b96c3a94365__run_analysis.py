#!/usr/bin/env python3
"""Frozen, read-only paired trajectory closure for FP/H/L6/L7.

The script reads only existing experiment artifacts.  It writes exclusively to
the new LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1 directory supplied via --output.
Official correctness is never redefined: the frozen V4 parser is rerun only as
an integrity check against the stored results.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK = "LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1"
SOURCE_TASK = "LING_RECURRENT_DENSE_L6_L7_V1"
EXPECTED_HEAD = "b058f7cb2144ef245db03c9f2c879546421bc34a"
EXPECTED_ASSIGNMENT = "e2781d2ecaf81f09d0c36b78438bb4d88c8236cebb5bd7a55c72c53a85bb3cf3"
EXPECTED_Q26 = "8a08632e751aaddbf18fbefcb78001bbffa689e0e2c119af4161e8be8dbc89bf"
EXPECTED_SCORES = {"FP": 14, "H": 9, "L6": 9, "L7": 10}
EXPECTED_SWITCHES = {
    "L6-H": {"rescued": ["aime26_23"], "regressed": ["aime26_26"]},
    "L7-H": {"rescued": ["aime26_11", "aime26_12"], "regressed": ["aime26_26"]},
    "L7-L6": {"rescued": ["aime26_11", "aime26_12"], "regressed": ["aime26_23"]},
}
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_REPLICATES = 10_000
NOT_AVAILABLE = "NOT_AVAILABLE"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


# Exact generic definitions copied from the established frozen long-generation
# audit script.  Its path and SHA256 are recorded in provenance.json.
def ngrams(tokens: list[int], n: int) -> Iterable[tuple[int, ...]]:
    return (tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def unique_ratio(tokens: list[int], n: int) -> float | None:
    total = len(tokens) - n + 1
    if total <= 0:
        return None
    return len(set(ngrams(tokens, n))) / total


def repeated_fraction(tokens: list[int], n: int) -> float | None:
    ratio = unique_ratio(tokens, n)
    return None if ratio is None else 1.0 - ratio


def approximate_longest_repeat(tokens: list[int]) -> dict[str, int | None]:
    n = len(tokens)
    if n < 32:
        return {"length": 0, "first": None, "second": None}
    mask = (1 << 64) - 1
    base = 1_000_003
    prefix = [0] * (n + 1)
    powers = [1] * (n + 1)
    for i, token in enumerate(tokens):
        prefix[i + 1] = (prefix[i] * base + int(token) + 1) & mask
        powers[i + 1] = (powers[i] * base) & mask

    def block_hash(start: int, length: int) -> int:
        return (prefix[start + length] - (prefix[start] * powers[length] & mask)) & mask

    candidates = (5000, 4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 100, 64, 32)
    for length in candidates:
        if length * 2 > n:
            continue
        seen: dict[int, int] = {}
        for start in range(0, n - length + 1):
            key = block_hash(start, length)
            previous = seen.get(key)
            if previous is not None and start - previous >= length:
                if tokens[previous:previous + length] == tokens[start:start + length]:
                    return {"length": length, "first": previous, "second": start}
            else:
                seen[key] = start
    return {"length": 0, "first": None, "second": None}


def char_to_token(char_pos: int, text_len: int, token_count: int) -> int | None:
    if text_len <= 0 or token_count <= 0:
        return None
    return min(token_count - 1, max(0, int(char_pos / text_len * token_count)))


def finish_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or value.get("matched") or "unknown")
    if value is None:
        return "unknown"
    return str(value)


def scorer_events(scorer: Any, text: str, finish_reason: Any, problem: str):
    normalized, _ = scorer._normalize_nonsemantic(text)
    events = scorer.enumerate_claim_events(normalized, problem)
    termination = scorer._termination_text(finish_reason)
    if scorer._is_truncated(termination):
        for event in events:
            if not event.committed:
                continue
            explicitly_final = (
                event.strength == "STRONG_FINAL"
                or event.region == "FINAL_ANSWER_REGION"
                or (event.strength == "EXPLICIT" and scorer._near_terminal(normalized, event.end, 1000))
            )
            if not explicitly_final:
                event.truncated_weak = True
                event.committed = False
    committed = [e for e in events if e.committed and not e.rejection_reasons()]
    return normalized, events, committed


def compact_event(event: Any, normalized: str, token_count: int) -> dict[str, Any]:
    return {
        "value": event.normalized_value,
        "expression": event.claimed_expression,
        "event_type": event.event_type,
        "rule": event.rule,
        "char_start": event.start,
        "token_position_estimate": char_to_token(event.start, len(normalized), token_count),
        "retracted": bool(event.retracted),
        "excerpt": normalized[max(0, event.start - 90):min(len(normalized), event.end + 140)].replace("\n", " "),
    }


def analyze_new_sample(path: Path, condition: str, scorer: Any, gpu_meta: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    text = raw["decoded_output"]
    tokens = [int(x) for x in raw["generated_token_ids"]]
    result = scorer.extract_v4(text, raw.get("finish_reason"), raw.get("problem", ""))
    normalized, events, committed = scorer_events(scorer, text, raw.get("finish_reason"), raw.get("problem", ""))
    if bool(raw["v4_correct"]) != bool(scorer.compare_with_gold(result, raw.get("gold"))):
        raise RuntimeError(f"V4 correctness mismatch for {path}")
    if raw.get("v4_extracted_answer") != result.normalized_value:
        raise RuntimeError(f"V4 extraction mismatch for {path}")
    committed_values = [e.normalized_value for e in committed]
    compressed: list[int | None] = []
    for value in committed_values:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    repeat = approximate_longest_repeat(tokens)
    first = committed[0] if committed else None
    last = committed[-1] if committed else None
    first_pos = char_to_token(first.start, len(normalized), len(tokens)) if first else None
    last_pos = char_to_token(last.start, len(normalized), len(tokens)) if last else None
    timeline = [compact_event(e, normalized, len(tokens)) for e in committed]
    if len(timeline) > 12:
        timeline = timeline[:6] + [{"omitted_committed_events": len(timeline) - 12}] + timeline[-6:]
    qid = raw["question_id"]
    meta = gpu_meta[(condition, qid)]
    ftype = finish_type(raw.get("finish_reason"))
    max_new = int(raw.get("max_new_tokens") or 0)
    hit_max = bool(raw.get("hit_context_limit")) or (ftype == "length" and len(tokens) >= max_new)
    gold = int(raw["gold"])
    return {
        "question_id": qid,
        "condition": condition,
        "source_path": str(path),
        "source_file_sha256": sha256_file(path),
        "artifact_storage_host": "lthpc1",
        "seed": int(raw["seed"]),
        "hardware": meta["gpu_model"],
        "hardware_resolution": "FROZEN_PER_SAMPLE_GPU_METADATA",
        "instance_id": raw.get("instance_id"),
        "physical_gpu": raw.get("physical_gpu"),
        "decoded_output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "tail_4096_char_sha256": hashlib.sha256(text[-4096:].encode("utf-8")).hexdigest(),
        "tail_excerpt": text[-800:].replace("\n", " "),
        "generated_tokens": len(tokens),
        "max_new_tokens": max_new,
        "prompt_tokens": raw.get("prompt_tokens"),
        "final_answer": raw.get("v4_extracted_answer"),
        "gold_answer": gold,
        "correct": bool(raw["v4_correct"]),
        "incorrect": not bool(raw["v4_correct"]),
        "abstain": bool(raw["v4_abstain"]),
        "explicit_eos": ftype == "stop",
        "hit_max_length": hit_max,
        "finish_type": ftype,
        "termination_category": "LENGTH_CEILING" if hit_max else ("EXPLICIT_EOS_OR_STOP" if ftype == "stop" else ftype.upper()),
        "successful_sample": bool(raw.get("successful_sample")),
        "token_provenance_valid": bool(raw.get("token_provenance_valid")),
        "runtime_error": raw.get("runtime_error"),
        "runtime_effective_sampling_params": raw.get("runtime_effective_sampling_params"),
        "scorer_sha256": raw.get("scorer_sha256"),
        "model": raw.get("model"),
        "state_quantization": raw.get("state_quantization"),
        "rotation": raw.get("rotation_runtime"),
        "rotation_sha256": raw.get("final_rotation_sha256"),
        "config_hash": None,
        "unique_4gram_ratio": unique_ratio(tokens, 4),
        "repeated_4gram_fraction": repeated_fraction(tokens, 4),
        "repeated_8gram_fraction": repeated_fraction(tokens, 8),
        "repeated_16gram_fraction": repeated_fraction(tokens, 16),
        "longest_repeated_span": repeat["length"],
        "longest_repeat_first_pos": repeat["first"],
        "longest_repeat_second_pos": repeat["second"],
        "first_committed_claim_position": first_pos,
        "final_committed_claim_position": last_pos,
        "num_committed_final_claims": len(committed),
        "num_answer_changes": max(0, len(compressed) - 1),
        "committed_value_sequence": compressed,
        "retracted_event_count": sum(bool(e.retracted) for e in events),
        "explicit_retraction_cue_count": len(list(scorer.RETRACTION_RE.finditer(text))),
        "tokens_after_first_commitment": None if first_pos is None else len(tokens) - first_pos,
        "tokens_after_final_commitment": None if last_pos is None else len(tokens) - last_pos,
        "post_first_commitment_fraction": None if first_pos is None else (len(tokens) - first_pos) / len(tokens),
        "correct_claim_later_retracted": any(e.normalized_value == gold and e.retracted for e in events),
        "never_formed_stable_commitment": len(committed) == 0,
        "committed_timeline": timeline,
        "official_v4_result": {k: v for k, v in raw.get("v4_result", {}).items() if k != "candidate_rejections"},
        "scorer_recomputed_result": {k: v for k, v in dataclasses.asdict(result).items() if k != "candidate_rejections"},
    }


def normalize_baseline_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["decoded_output_sha256"] = row.pop("output_hash")
    row.setdefault("physical_gpu", (row.get("gpu_ids") or [None])[0])
    row.setdefault("rotation_sha256", None)
    row.setdefault("tail_4096_char_sha256", None)
    row.setdefault("tail_excerpt", None)
    row["incorrect"] = not bool(row["correct"])
    return row


def score_switch(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    rescued = sorted(q for q in a if not b[q]["correct"] and a[q]["correct"])
    regressed = sorted(q for q in a if b[q]["correct"] and not a[q]["correct"])
    return {"rescued": rescued, "regressed": regressed}


def exact_mcnemar(rescued: int, regressed: int) -> float:
    n = rescued + regressed
    if n == 0:
        return 1.0
    k = min(rescued, regressed)
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
    if not values:
        raise ValueError("empty quantile")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def paired_summary(deltas: list[float], rng: random.Random) -> dict[str, Any]:
    if not deltas:
        return {"n": 0, "status": NOT_AVAILABLE}
    boot = []
    n = len(deltas)
    for _ in range(BOOTSTRAP_REPLICATES):
        boot.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    return {
        "n": n,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "positive": sum(x > 0 for x in deltas),
        "zero": sum(x == 0 for x in deltas),
        "negative": sum(x < 0 for x in deltas),
        "bootstrap_mean_delta_95pct_ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


NUMERIC_METRICS = [
    "generated_tokens", "unique_4gram_ratio", "repeated_4gram_fraction",
    "repeated_8gram_fraction", "repeated_16gram_fraction", "longest_repeated_span",
    "first_committed_claim_position", "final_committed_claim_position",
    "num_committed_final_claims", "num_answer_changes", "retracted_event_count",
    "explicit_retraction_cue_count", "tokens_after_first_commitment",
    "tokens_after_final_commitment", "post_first_commitment_fraction",
]


def paired_rows(by_condition: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    pairs = [("L6", "H"), ("L7", "H"), ("L7", "L6")]
    rng = random.Random(BOOTSTRAP_SEED)
    stats: dict[str, Any] = {}
    for q in sorted(by_condition["H"]):
        wide: dict[str, Any] = {"question_id": q}
        for c in ("H", "L6", "L7"):
            r = by_condition[c][q]
            wide[f"{c}_correct"] = r["correct"]
            wide[f"{c}_generated_tokens"] = r["generated_tokens"]
            wide[f"{c}_termination"] = r["termination_category"]
        for a, b in pairs:
            name = f"{a}-{b}"
            ar, br = by_condition[a][q], by_condition[b][q]
            wide[f"{name}_correctness_transition"] = int(ar["correct"]) - int(br["correct"])
            wide[f"{name}_termination_transition"] = f"{br['termination_category']} -> {ar['termination_category']}"
            for m in NUMERIC_METRICS:
                av, bv = ar.get(m), br.get(m)
                wide[f"{name}_delta_{m}"] = None if av is None or bv is None else av - bv
            wide[f"{name}_delta_reasoning_oscillation_proxy"] = NOT_AVAILABLE
            wide[f"{name}_delta_textual_loop_indicator"] = NOT_AVAILABLE
            wide[f"{name}_delta_semantic_degeneration_indicator"] = NOT_AVAILABLE
        rows.append(wide)
    for a, b in pairs:
        name = f"{a}-{b}"
        sw = score_switch(by_condition[a], by_condition[b])
        p = exact_mcnemar(len(sw["rescued"]), len(sw["regressed"]))
        metric_stats = {}
        for m in NUMERIC_METRICS:
            ds = [r[f"{name}_delta_{m}"] for r in rows if r[f"{name}_delta_{m}"] is not None]
            metric_stats[m] = paired_summary([float(x) for x in ds], rng)
        stats[name] = {
            **sw,
            "rescued_count": len(sw["rescued"]),
            "regressed_count": len(sw["regressed"]),
            "discordant_count": len(sw["rescued"]) + len(sw["regressed"]),
            "exact_mcnemar_binomial_two_sided_p": p,
            "numeric_metric_deltas": metric_stats,
        }
    adjusted = holm({k: stats[k]["exact_mcnemar_binomial_two_sided_p"] for k in ("L6-H", "L7-H")})
    for name, value in adjusted.items():
        stats[name]["holm_adjusted_p_two_primary"] = value
    return rows, stats


def outcome_associations(by_condition: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    helpful = {
        "generated_tokens": -1, "repeated_4gram_fraction": -1,
        "repeated_8gram_fraction": -1, "repeated_16gram_fraction": -1,
        "first_committed_claim_position": -1, "num_answer_changes": -1,
        "retracted_event_count": -1, "explicit_retraction_cue_count": -1,
    }
    out: dict[str, Any] = {}
    for c in ("L6", "L7"):
        groups: dict[int, list[str]] = defaultdict(list)
        for q, cr in by_condition[c].items():
            groups[int(cr["correct"]) - int(by_condition["H"][q]["correct"])].append(q)
        metric_groups = {}
        for m in helpful:
            mg = {}
            for transition, qs in sorted(groups.items()):
                ds = []
                for q in qs:
                    a, b = by_condition[c][q].get(m), by_condition["H"][q].get(m)
                    if a is not None and b is not None:
                        ds.append(float(a - b))
                mg[str(transition)] = {
                    "n": len(ds),
                    "mean_delta": statistics.fmean(ds) if ds else None,
                    "median_delta": statistics.median(ds) if ds else None,
                }
            metric_groups[m] = mg
        rescue_qs, regress_qs = groups.get(1, []), groups.get(-1, [])
        termination_checks = []
        length_checks = []
        stable_commitment_checks = []
        for q in rescue_qs:
            termination_checks.append(by_condition["H"][q]["hit_max_length"] and not by_condition[c][q]["hit_max_length"])
            length_checks.append(by_condition[c][q]["generated_tokens"] < by_condition["H"][q]["generated_tokens"])
            stable_commitment_checks.append(by_condition["H"][q]["never_formed_stable_commitment"] and not by_condition[c][q]["never_formed_stable_commitment"])
        for q in regress_qs:
            termination_checks.append(not by_condition["H"][q]["hit_max_length"] and by_condition[c][q]["hit_max_length"])
            length_checks.append(by_condition[c][q]["generated_tokens"] > by_condition["H"][q]["generated_tokens"])
            stable_commitment_checks.append(not by_condition["H"][q]["never_formed_stable_commitment"] and by_condition[c][q]["never_formed_stable_commitment"])
        directional = []
        for m, sign in helpful.items():
            rv = [by_condition[c][q].get(m) for q in rescue_qs]
            rb = [by_condition["H"][q].get(m) for q in rescue_qs]
            gv = [by_condition[c][q].get(m) for q in regress_qs]
            gb = [by_condition["H"][q].get(m) for q in regress_qs]
            if rescue_qs and regress_qs and all(x is not None for x in rv + rb + gv + gb):
                rescue_delta = statistics.fmean(float(a - b) for a, b in zip(rv, rb))
                regress_delta = statistics.fmean(float(a - b) for a, b in zip(gv, gb))
                coherent = sign * rescue_delta > sign * regress_delta
                directional.append({"metric": m, "rescue_mean_delta": rescue_delta, "regress_mean_delta": regress_delta, "coherent": coherent})
        out[c] = {
            "transition_question_ids": {str(k): sorted(v) for k, v in groups.items()},
            "grouped_metric_deltas": metric_groups,
            "switch_direction_checks": directional,
            "coherent_metric_count": sum(x["coherent"] for x in directional),
            "checked_metric_count": len(directional),
            "switch_termination_direction": {"coherent": sum(termination_checks), "checked": len(termination_checks)},
            "switch_length_direction": {"coherent": sum(length_checks), "checked": len(length_checks)},
            "switch_stable_commitment_direction": {"coherent": sum(stable_commitment_checks), "checked": len(stable_commitment_checks)},
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            cooked = {k: (json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
            w.writerow(cooked)


def rotation_sanity(source: Path) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {"purpose": "sanity/provenance only; distances are not reasoning metrics", "conditions": {}}
    loaded = {}
    for c in ("L6", "L7"):
        path = source / f"rotations/{c}_final_rotation.pt"
        obj = torch.load(path, map_location="cpu", weights_only=True)
        layers = {}
        for lid, tensor in obj["rotations"].items():
            x = tensor.detach().float().cpu()
            eye = torch.eye(x.shape[0], dtype=x.dtype)
            gram_error = x.T @ x - eye
            sign, logabsdet = torch.linalg.slogdet(x.double())
            layers[str(lid)] = {
                "shape": list(x.shape),
                "dtype": str(tensor.dtype),
                "nan_count": int(torch.isnan(x).sum()),
                "inf_count": int(torch.isinf(x).sum()),
                "orthogonality_error_frobenius": float(torch.linalg.vector_norm(gram_error)),
                "orthogonality_error_max_abs": float(gram_error.abs().max()),
                "parameter_norm_frobenius": float(torch.linalg.vector_norm(x)),
                "determinant_sign": float(sign),
                "log_abs_determinant": float(logabsdet),
            }
        result["conditions"][c] = {
            "file": str(path),
            "file_sha256": sha256_file(path),
            "source_checkpoint": obj["source_checkpoint"],
            "source_checkpoint_sha256": obj["source_checkpoint_sha256"],
            "reconstruction_formula": obj["reconstruction_formula"],
            "entry": obj["entry"],
            "recovery": obj["recovery"],
            "layers": layers,
        }
        loaded[c] = {int(k): v.detach().float().cpu() for k, v in obj["rotations"].items()}
    common = sorted(set(loaded["L6"]) & set(loaded["L7"]))
    result["L7_minus_L6_common_representation"] = {
        "mathematically_valid": True,
        "interpretation": "sanity-only Frobenius distance between final rotation matrices in the same implementation basis",
        "per_layer_frobenius": {str(k): float(torch.linalg.vector_norm(loaded["L7"][k] - loaded["L6"][k])) for k in common},
    }
    result["distance_to_hadamard"] = {
        "status": NOT_AVAILABLE,
        "reason": "not computed here: no need to treat a simple Frobenius distance to H as a functional or reasoning metric",
    }
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def forensic_markdown(by_condition: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = ["# Switch-case forensics", "", "Official V4 correctness is frozen. Token positions below are explicitly estimates from normalized character fraction to stored token count.", ""]
    for q in ("aime26_11", "aime26_12", "aime26_23", "aime26_26"):
        lines += [f"## {q}", "", "| Condition | correct | abstain | tokens | termination | first commitment | final commitment | changes | retractions | rep-4 | longest repeat |", "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for c in ("H", "L6", "L7"):
            r = by_condition[c][q]
            lines.append(f"| {c} | {r['correct']} | {r['abstain']} | {r['generated_tokens']} | {r['termination_category']} | {fmt(r['first_committed_claim_position'])} | {fmt(r['final_committed_claim_position'])} | {r['num_answer_changes']} | {r['retracted_event_count']} | {fmt(r['repeated_4gram_fraction'])} | {r['longest_repeated_span']} |")
        lines += ["", "### Compact claim timelines", ""]
        for c in ("H", "L6", "L7"):
            r = by_condition[c][q]
            lines.append(f"- **{c}:** extracted={r['final_answer']}; gold={r['gold_answer']}; committed sequence={r['committed_value_sequence']}; correct-claim-later-retracted={r['correct_claim_later_retracted']}; never-stable={r['never_formed_stable_commitment']}.")
            for event in r["committed_timeline"]:
                if "omitted_committed_events" in event:
                    lines.append(f"  - ... {event['omitted_committed_events']} committed events omitted ...")
                else:
                    lines.append(f"  - t≈{event['token_position_estimate']}: value={event['value']}, type={event['event_type']}, retracted={event['retracted']}; excerpt=`{event['excerpt'][:240]}`")
        lines += ["", "### Forensic interpretation", ""]
        rows = [by_condition[c][q] for c in ("H", "L6", "L7")]
        if any(r["scorer_recomputed_result"] != r["official_v4_result"] for r in rows):
            lines.append("- Frozen parser recomputation differs from the stored scorer artifact; scorer-artifact explanation remains possible and the provenance gate must fail.")
        else:
            lines.append("- The same frozen V4 parser exactly reproduces every stored extraction/correctness result; there is no parser-version mismatch for this case.")
        lines.append("- A length-ceiling output is treated as trajectory/termination evidence, not automatically as a scorer artifact.")
        if q == "aime26_26":
            r = by_condition["L7"][q]
            if r["hit_max_length"] and r["abstain"] and r["never_formed_stable_commitment"]:
                lines.append("- q26 L7 supports: trajectory divergence/instability → no stable committed final claim → continued generation → length ceiling → abstain. The artifacts cannot establish a causal internal-state mechanism, so ‘divergence’ is descriptive, not mechanistic proof.")
            elif r["hit_max_length"] and r["abstain"]:
                lines.append("- q26 L7 reached the length ceiling and abstained, but it did form at least one parser-recognized committed claim. The stronger ‘never stabilized’ chain is therefore not fully supported.")
            else:
                lines.append("- q26 L7 does not support the proposed length-ceiling/abstain chain; the artifact pattern differs.")
            lines.append(f"- q26 L7 tail-4096-character SHA256: `{r['tail_4096_char_sha256']}`.")
            lines.append(f"- q26 L7 compact tail excerpt: `{r['tail_excerpt'][:800]}`")
        lines.append("")
    lines += ["## Unsupported established indicators", "", "The established 81,920-token segmented definitions for reasoning oscillation proxy, textual loop indicator, and semantic degeneration indicator were not silently extended to the 262,144-token protocol. They are recorded as `NOT_AVAILABLE`.", ""]
    return "\n".join(lines)


def build_report(scores: dict[str, int], switches: dict[str, Any], stats: dict[str, Any], associations: dict[str, Any], by_condition: dict[str, dict[str, dict[str, Any]]], labels: dict[str, str], provenance: dict[str, Any]) -> str:
    lines = [
        f"# {TASK}", "",
        "This is a strict read-only offline analysis. No training or free-generation evaluation was run, and no frozen source artifact was changed.", "",
        "## Outcome integrity", "",
        f"Exact reproduced scores: FP **{scores['FP']}/20**, Hadamard/L2 **{scores['H']}/20**, L6 **{scores['L6']}/20**, L7 **{scores['L7']}/20**.", "",
    ]
    for name in ("L6-H", "L7-H", "L7-L6"):
        s = switches[name]
        lines.append(f"- {name}: rescue={s['rescued']}; regress={s['regressed']}; net={len(s['rescued'])-len(s['regressed'])}; exact p={stats[name]['exact_mcnemar_binomial_two_sided_p']}." + (f" Holm p={stats[name]['holm_adjusted_p_two_primary']}." if "holm_adjusted_p_two_primary" in stats[name] else ""))
    lines += ["", "## All-20 paired trajectory summary", "", "Sign convention throughout is condition A minus condition B. Continuous estimates are exploratory, with paired mean/median, sign counts, and deterministic 10,000-replicate bootstrap CIs in `statistics.json`.", ""]
    for name in ("L6-H", "L7-H", "L7-L6"):
        lines.append(f"### {name}")
        lines.append("")
        for m in ("generated_tokens", "repeated_4gram_fraction", "first_committed_claim_position", "num_answer_changes", "retracted_event_count"):
            x = stats[name]["numeric_metric_deltas"][m]
            lines.append(f"- {m}: n={x['n']}, mean Δ={fmt(x.get('mean_delta'))}, median Δ={fmt(x.get('median_delta'))}, signs +/0/−={x.get('positive')}/{x.get('zero')}/{x.get('negative')}, bootstrap mean-Δ CI={x.get('bootstrap_mean_delta_95pct_ci')}.")
        lines.append("")
    lines += ["## Outcome/trajectory association", ""]
    for c in ("L6", "L7"):
        a = associations[c]
        lines.append(f"- {c}: among {a['checked_metric_count']} prespecified directional checks, {a['coherent_metric_count']} place rescue deltas in a more favorable direction than regression deltas. With only {len(a['transition_question_ids'].get('1', []))} rescues and {len(a['transition_question_ids'].get('-1', []))} regressions, this is descriptive only.")
        lines.append(f"  Switch-direction checks: termination {a['switch_termination_direction']['coherent']}/{a['switch_termination_direction']['checked']}, length {a['switch_length_direction']['coherent']}/{a['switch_length_direction']['checked']}, stable commitment {a['switch_stable_commitment_direction']['coherent']}/{a['switch_stable_commitment_direction']['checked']} coherent.")
    lines += ["", "## Switch-case findings", ""]
    for q in ("aime26_11", "aime26_12", "aime26_23", "aime26_26"):
        bits = []
        for c in ("H", "L6", "L7"):
            r = by_condition[c][q]
            bits.append(f"{c}={'correct' if r['correct'] else ('abstain' if r['abstain'] else 'wrong')}, {r['generated_tokens']} tokens, {r['termination_category']}")
        lines.append(f"- {q}: " + "; ".join(bits) + ". See `switch_case_forensics.md` for compact claim timelines.")
    q26 = by_condition["L7"]["aime26_26"]
    lines += ["", f"q26 L7 independently verifies file SHA256 `{q26['source_file_sha256']}`, {q26['generated_tokens']} generated tokens, length ceiling={q26['hit_max_length']}, incorrect={not q26['correct']}, abstain={q26['abstain']}.", ""]
    lines += ["## Geometry/checkpoint sanity", "", "L6/L7 rotations were independently loaded. Per-layer orthogonality, NaN/Inf, determinant sign/log-absolute-determinant, parameter norm, exact checkpoint/file hashes, and same-basis L6↔L7 Frobenius distances are in `rotation_sanity.json`. Distances are provenance/sanity only, not reasoning metrics.", ""]
    lines += ["## Hardware and provenance caveats", "", f"- Effective assignment SHA256: `{provenance['effective_assignment_sha256']}`.", "- FP is taken from one exact frozen canonical baseline artifact set. The known FP cross-hardware bitwise parity failure is preserved; no cross-hardware FP outputs are pooled as interchangeable observations.", "- H q11/q12 retain their migrated 4090 generation identity; the remaining frozen H samples and FP baseline samples retain their exact artifact paths and generation metadata.", "- L6/L7 condition-level INT8 parity passed in the source audit, permitting the frozen mixed-hardware canonical assignment.", "- Commitment token positions are estimates from normalized character fraction to stored token count; official scoring uses the original validated V4 parser and is not based on those estimates.", "- Established 81,920-token segmented oscillation/loop/semantic-degeneration indicators are `NOT_AVAILABLE`; they were not silently redefined for 262,144 tokens.", ""]
    lines += ["## Diagnostic labels", ""]
    for key in ("PROVENANCE_GATE", "SCORER_ARTIFACT_EXPLANATION", "L7_STRUCTURED_TRAJECTORY_SIGNAL", "L6_STRUCTURED_TRAJECTORY_SIGNAL", "CURRENT_RESULT_INTERPRETATION"):
        lines.append(f"- `{key} = {labels[key]}`")
    lines += ["", "The score ordering alone is not used as evidence. Exact p=1.0 is not interpreted as method identity. The small number of discordant pairs sharply limits inference.", "", f"`NEXT_ACTION = {labels['NEXT_ACTION']}`", "", "No next action was executed.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    source = repo / f"experiments/{SOURCE_TASK}"
    output = args.output.resolve()
    analysis = output / "analysis"
    scripts = output / "scripts"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis directory: {output}")
    analysis.mkdir(parents=True)
    scripts.mkdir(parents=True)

    head = run_git(repo, "rev-parse", "HEAD")
    stage0_status = run_git(repo, "status", "--porcelain")
    scorer_path = source / "scripts/aime26_scorer_v4.py"
    scorer = load_module("frozen_v4_closure", scorer_path)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    gpu_doc = json.loads((source / "analysis/per_sample_gpu_metadata.json").read_text())
    gpu_meta = {(r["condition"], r["question_id"]): r for r in gpu_doc["samples"]}
    rows = [normalize_baseline_row(r) for r in baseline["rows"]]
    for c in ("L6", "L7"):
        for q in range(11, 31):
            rows.append(analyze_new_sample(source / f"outputs/{c}/aime26_{q:02d}_seed1.json", c, scorer, gpu_meta))
    rows.sort(key=lambda r: (r["condition"], r["question_id"]))
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][row["question_id"]] = row
    scores = {c: sum(r["correct"] for r in by_condition[c].values()) for c in ("FP", "H", "L6", "L7")}
    switches = {
        "L6-H": score_switch(by_condition["L6"], by_condition["H"]),
        "L7-H": score_switch(by_condition["L7"], by_condition["H"]),
        "L7-L6": score_switch(by_condition["L7"], by_condition["L6"]),
    }
    integrity_errors = []
    if scores != EXPECTED_SCORES:
        integrity_errors.append(f"score mismatch {scores} != {EXPECTED_SCORES}")
    if switches != EXPECTED_SWITCHES:
        integrity_errors.append(f"switch mismatch {switches} != {EXPECTED_SWITCHES}")
    qids = sorted(by_condition["H"])
    expected_qids = [f"aime26_{q:02d}" for q in range(11, 31)]
    if any(sorted(by_condition[c]) != expected_qids for c in ("FP", "H", "L6", "L7")):
        integrity_errors.append("question set mismatch")
    if any(r["seed"] != 1 for r in rows):
        integrity_errors.append("seed mismatch")
    assignment_path = source / "manifests/effective_hardware_assignment.json"
    assignment_file_sha = sha256_file(assignment_path)
    assignment_doc = json.loads(assignment_path.read_text())
    assignment_sha = assignment_doc["effective_assignment_sha256"]
    q26_sha = sha256_file(source / "outputs/L7/aime26_26_seed1.json")
    if head != EXPECTED_HEAD:
        integrity_errors.append(f"HEAD mismatch {head}")
    if stage0_status:
        integrity_errors.append(f"source repo was dirty before analysis: {stage0_status}")
    if assignment_sha != EXPECTED_ASSIGNMENT:
        integrity_errors.append(f"assignment hash mismatch {assignment_sha}")
    if q26_sha != EXPECTED_Q26:
        integrity_errors.append(f"q26 hash mismatch {q26_sha}")
    if any(r["official_v4_result"] != r["scorer_recomputed_result"] for r in rows):
        integrity_errors.append("stored V4 result differs from frozen parser recomputation")

    pair_rows, stats = paired_rows(by_condition)
    associations = outcome_associations(by_condition)
    stats_payload = {
        "analysis_type": "diagnostic/exploratory",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "scores": scores,
        "pairwise": stats,
        "outcome_transition_relative_to_H": associations,
        "multiple_testing_scope": "Holm correction only for the two source primary comparisons L6-H and L7-H; L7-L6 is secondary exact",
    }
    rotation = rotation_sanity(source)

    # Preregistered labels are assigned conservatively from the frozen evidence.
    scorer_artifact = "NOT_SUPPORTED" if not any(r["official_v4_result"] != r["scorer_recomputed_result"] for r in rows) else "SUPPORTED"
    l7_checks = associations["L7"]["switch_direction_checks"]
    l6_checks = associations["L6"]["switch_direction_checks"]
    l7_ratio = (sum(x["coherent"] for x in l7_checks) / len(l7_checks)) if l7_checks else 0.0
    l6_ratio = (sum(x["coherent"] for x in l6_checks) / len(l6_checks)) if l6_checks else 0.0
    l7_term = associations["L7"]["switch_termination_direction"]
    l7_length = associations["L7"]["switch_length_direction"]
    l7_signal = "SUPPORTIVE" if (
        l7_term["checked"] >= 3 and l7_term["coherent"] == l7_term["checked"]
        and l7_length["coherent"] == l7_length["checked"]
        and l7_ratio >= 0.5
    ) else ("NOT_SUPPORTED" if l7_ratio <= 0.25 and len(l7_checks) >= 4 else "INCONCLUSIVE")
    # L6 has only one rescue and one regression.  Even coherent switch metrics
    # are insufficient to call a structured signal with two discordant cases.
    l6_signal = "INCONCLUSIVE" if len(EXPECTED_SWITCHES["L6-H"]["rescued"]) + len(EXPECTED_SWITCHES["L6-H"]["regressed"]) < 3 else ("SUPPORTIVE" if l6_ratio >= 0.75 else "INCONCLUSIVE")
    if integrity_errors:
        interpretation, next_action = "ARTIFACT_CONFOUNDED", "STOP_AND_FIX_PROVENANCE"
    elif l7_signal == "SUPPORTIVE" and scorer_artifact == "NOT_SUPPORTED":
        interpretation, next_action = "STRUCTURED_BUT_UNDERPOWERED", "CANONICAL_60_EVALUATION"
    elif scorer_artifact == "NOT_SUPPORTED" and (l7_signal != "SUPPORTIVE" or l6_signal != "SUPPORTIVE"):
        interpretation, next_action = "TRAJECTORY_REDISTRIBUTION", "FUNCTIONAL_ALIGNMENT_REPLAY"
    else:
        interpretation, next_action = "INCONCLUSIVE", "FUNCTIONAL_ALIGNMENT_REPLAY"
    labels = {
        "PROVENANCE_GATE": "FAIL" if integrity_errors else "PASS",
        "SCORER_ARTIFACT_EXPLANATION": scorer_artifact if not integrity_errors else "INCONCLUSIVE",
        "L7_STRUCTURED_TRAJECTORY_SIGNAL": l7_signal if not integrity_errors else "INCONCLUSIVE",
        "L6_STRUCTURED_TRAJECTORY_SIGNAL": l6_signal if not integrity_errors else "INCONCLUSIVE",
        "CURRENT_RESULT_INTERPRETATION": interpretation,
        "NEXT_ACTION": next_action,
        "label_rule_details": {
            "L7_coherent_switch_metric_fraction": l7_ratio,
            "L6_coherent_switch_metric_fraction": l6_ratio,
            "L7_switch_termination_direction": l7_term,
            "L7_switch_length_direction": l7_length,
            "L6_underpowered_discordant_pairs": 2,
        },
    }

    relevant_files = [
        "manifests/artifact_manifest.json", "manifests/baseline_reuse_manifest.json",
        "manifests/checkpoint_manifest.json", "manifests/rotation_manifest.json",
        "manifests/effective_hardware_assignment.json", "configs/frozen_eval_config.json",
        "configs/effective_server_config.json", "analysis/per_sample_gpu_metadata.json",
        "analysis/final_hardware_audit.json", "analysis/paired_comparisons.json",
    ]
    frozen_eval = json.loads((source / "configs/frozen_eval_config.json").read_text())
    effective_server = json.loads((source / "configs/effective_server_config.json").read_text())
    rotation_manifest = json.loads((source / "manifests/rotation_manifest.json").read_text())
    artifact_manifest = json.loads((source / "manifests/artifact_manifest.json").read_text())
    condition_identity = {}
    for c in ("FP", "H", "L6", "L7"):
        recorded_models = sorted({str(r.get("model")) for r in by_condition[c].values() if r.get("model") is not None})
        recorded_scorers = sorted({str(r.get("scorer_sha256")) for r in by_condition[c].values() if r.get("scorer_sha256") is not None})
        if c == "FP":
            rotation_ids = ["NONE"]
        elif c == "H":
            rotation_ids = [rotation_manifest["rotations"]["L2_UNIFIED_H128"]["sha256"]]
        else:
            rotation_ids = sorted({str(r.get("rotation_sha256")) for r in by_condition[c].values() if r.get("rotation_sha256") is not None})
        condition_identity[c] = {
            "record_level_models": recorded_models or [NOT_AVAILABLE],
            "resolved_model_from_frozen_config": frozen_eval["model"],
            "resolved_model_revision_from_frozen_config": frozen_eval["model_revision"],
            "record_level_scorer_sha256": recorded_scorers or [NOT_AVAILABLE],
            "resolved_scorer": frozen_eval["scorer"],
            "resolved_scorer_sha256": frozen_eval["scorer_sha256"],
            "max_new_tokens_range": [min(r["max_new_tokens"] for r in by_condition[c].values()), max(r["max_new_tokens"] for r in by_condition[c].values())],
            "max_new_tokens_definition": frozen_eval["max_new_tokens_definition"],
            "state_quantization": sorted({json.dumps(r.get("state_quantization"), sort_keys=True) for r in by_condition[c].values()}),
            "rotation_identity_sha256": rotation_ids,
            "record_config_hashes": sorted({str(r.get("config_hash")) for r in by_condition[c].values() if r.get("config_hash") is not None}),
            "hardware_counts": {h: sum(r["hardware"] == h for r in by_condition[c].values()) for h in sorted({r["hardware"] for r in by_condition[c].values()})},
        }
    provenance = {
        "task": TASK,
        "mode": "STRICT_READ_ONLY_OFFLINE_ANALYSIS",
        "source_task": SOURCE_TASK,
        "source_repo": str(repo),
        "source_git_head_before_analysis": head,
        "source_git_status_before_analysis": "CLEAN" if not stage0_status else stage0_status,
        "expected_frozen_commit": EXPECTED_HEAD,
        "effective_assignment_file": str(source / "manifests/effective_hardware_assignment.json"),
        "effective_assignment_sha256": assignment_sha,
        "effective_assignment_file_sha256": assignment_file_sha,
        "q26_L7_file": str(source / "outputs/L7/aime26_26_seed1.json"),
        "q26_L7_file_sha256": q26_sha,
        "baseline_extract_file": str(args.baseline),
        "baseline_extract_sha256": sha256_file(args.baseline),
        "baseline_source_manifest": baseline["source_manifest"],
        "baseline_source_manifest_sha256": baseline["source_manifest_sha256"],
        "question_ids": qids,
        "seeds": sorted({r["seed"] for r in rows}),
        "condition_runtime_identity": condition_identity,
        "frozen_runtime_identity": {
            "context_length": frozen_eval["context_length"],
            "generation_seed": frozen_eval["generation_seed"],
            "sampling": {k: frozen_eval[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "stop", "thinking", "do_sample")},
            "software": {k: frozen_eval[k] for k in ("sglang_version", "sglang_source_commit", "torch_version", "triton_version", "fla_version")},
            "determinism": {k: frozen_eval[k] for k in ("enable_deterministic_inference", "disable_cuda_graph", "disable_radix_cache", "max_running_requests")},
            "effective_server_config": effective_server,
        },
        "checkpoint_manifest": json.loads((source / "manifests/checkpoint_manifest.json").read_text()),
        "rotation_files": {c: {"path": str(source / f"rotations/{c}_final_rotation.pt"), "sha256": sha256_file(source / f"rotations/{c}_final_rotation.pt")} for c in ("L6", "L7")},
        "scorer_parser": {"path": str(scorer_path), "sha256": sha256_file(scorer_path)},
        "established_metric_script": {"path": baseline["established_metric_path"], "sha256": baseline["established_metric_sha256"], "reused_definitions": baseline["established_metric_implementation"]},
        "commitment_position_mapping": baseline["position_mapping"],
        "unsupported_established_metrics": baseline["unsupported_established_metrics"],
        "source_relevant_file_hashes": {rel: sha256_file(source / rel) for rel in relevant_files},
        "source_artifact_audit": {
            "verified_artifacts": len(artifact_manifest["files"]),
            "manifest_status": artifact_manifest["status"],
            "manifest_sha256": sha256_file(source / "manifests/artifact_manifest.json"),
            "exclusions": artifact_manifest["exclusions"],
            "source": "frozen source artifact manifest and final audit",
        },
        "hardware_audit": json.loads((source / "analysis/final_hardware_audit.json").read_text()),
        "FP_cross_hardware_bitwise_parity": "FAIL; exact frozen canonical FP set used without cross-hardware pooling",
        "L6_L7_INT8_condition_level_parity": "PASS (from frozen source parity artifacts)",
        "integrity_errors": integrity_errors,
        "status": "FAIL" if integrity_errors else "PASS",
    }

    outcome_fields = ["question_id", "condition", "hardware", "instance_id", "physical_gpu", "seed", "source_file_sha256", "decoded_output_sha256", "generated_tokens", "final_answer", "gold_answer", "correct", "incorrect", "abstain", "explicit_eos", "hit_max_length", "termination_category", "successful_sample", "token_provenance_valid"]
    trajectory_fields = ["question_id", "condition", "generated_tokens", *NUMERIC_METRICS[1:], "correct_claim_later_retracted", "never_formed_stable_commitment", "termination_category", "reasoning_oscillation_proxy", "textual_loop_indicator", "semantic_degeneration_indicator"]
    trajectory_rows = []
    for r in rows:
        x = dict(r)
        x.update({"reasoning_oscillation_proxy": NOT_AVAILABLE, "textual_loop_indicator": NOT_AVAILABLE, "semantic_degeneration_indicator": NOT_AVAILABLE})
        trajectory_rows.append(x)
    write_csv(analysis / "per_sample_outcomes.csv", rows, outcome_fields)
    write_csv(analysis / "trajectory_metrics.csv", trajectory_rows, trajectory_fields)
    write_csv(analysis / "pairwise_deltas.csv", pair_rows)
    json_dump(analysis / "provenance.json", provenance)
    json_dump(analysis / "statistics.json", stats_payload)
    json_dump(analysis / "rotation_sanity.json", rotation)
    json_dump(analysis / "diagnostic_labels.json", labels)
    (analysis / "switch_case_forensics.md").write_text(forensic_markdown(by_condition), encoding="utf-8")
    (analysis / "final_report.md").write_text(build_report(scores, switches, stats, associations, by_condition, labels, provenance), encoding="utf-8")
    shutil.copy2(args.baseline, analysis / "baseline_extract_compact.json")
    shutil.copy2(Path(__file__), scripts / "run_analysis.py")

    created = sorted(p for p in output.rglob("*") if p.is_file())
    manifest_lines = [f"{sha256_file(p)}  {p.relative_to(output).as_posix()}" for p in created]
    (output / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    verification = all(sha256_file(output / line.split("  ", 1)[1]) == line.split("  ", 1)[0] for line in manifest_lines)
    print(json.dumps({"status": provenance["status"], "scores": scores, "switches": switches, "labels": labels, "new_artifacts": len(created) + 1, "sha256_manifest_verified": verification}, indent=2))
    if integrity_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
