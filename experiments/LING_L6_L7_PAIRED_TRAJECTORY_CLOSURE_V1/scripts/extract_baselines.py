#!/usr/bin/env python3
"""Read-only extraction of frozen FP/H baseline trajectory diagnostics."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


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


# Exact generic definitions from the frozen long-generation audit script
# (SHA256 recorded in the output and independently verified on the 4090 host).
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


def compact_event(event: Any, text: str, token_count: int) -> dict[str, Any]:
    return {
        "value": event.normalized_value,
        "expression": event.claimed_expression,
        "event_type": event.event_type,
        "rule": event.rule,
        "char_start": event.start,
        "token_position_estimate": char_to_token(event.start, len(text), token_count),
        "retracted": bool(event.retracted),
        "excerpt": text[max(0, event.start - 90):min(len(text), event.end + 140)].replace("\n", " "),
    }


def scorer_committed_events(scorer: Any, text: str, finish_reason: Any, problem: str):
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
    return normalized, events, [e for e in events if e.committed and not e.rejection_reasons()]


def analyze_one(path: Path, condition: str, scorer: Any) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    text = raw["decoded_output"]
    tokens = [int(x) for x in raw["generated_token_ids"]]
    result = scorer.extract_v4(text, raw.get("finish_reason"), raw.get("problem", ""))
    normalized, events, committed = scorer_committed_events(
        scorer, text, raw.get("finish_reason"), raw.get("problem", "")
    )
    committed_values = [e.normalized_value for e in committed]
    compressed: list[int | None] = []
    for value in committed_values:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    changes = max(0, len(compressed) - 1)
    retracted_events = [e for e in events if e.retracted]
    explicit_retraction_cues = len(list(scorer.RETRACTION_RE.finditer(text)))
    # The established long-generation script exposes these exact generic token metrics.
    ratios = {
        "unique_4gram_ratio": unique_ratio(tokens, 4),
        "repeated_4gram_fraction": repeated_fraction(tokens, 4),
        "repeated_8gram_fraction": repeated_fraction(tokens, 8),
        "repeated_16gram_fraction": repeated_fraction(tokens, 16),
    }
    repeat = approximate_longest_repeat(tokens)
    first = committed[0] if committed else None
    last = committed[-1] if committed else None
    first_pos = char_to_token(first.start, len(normalized), len(tokens)) if first else None
    last_pos = char_to_token(last.start, len(normalized), len(tokens)) if last else None
    finish = raw.get("finish_reason") or {}
    finish_type = finish.get("type") if isinstance(finish, dict) else str(finish)
    max_new = int(raw.get("max_new_tokens_effective") or raw.get("max_new_tokens") or 0)
    hit_max = finish_type == "length" and len(tokens) >= max_new
    official_full = raw.get("v4_result", {})
    official = {k: v for k, v in official_full.items() if k != "candidate_rejections"}
    if bool(raw["v4_correct"]) != bool(scorer.compare_with_gold(result, raw.get("gold"))):
        raise RuntimeError(f"V4 score mismatch for {path}")
    if raw.get("v4_extracted_answer") != result.normalized_value:
        raise RuntimeError(f"V4 extraction mismatch for {path}")
    timeline = [compact_event(e, normalized, len(tokens)) for e in committed]
    if len(timeline) > 12:
        timeline = timeline[:6] + [{"omitted_committed_events": len(timeline) - 12}] + timeline[-6:]
    gold = int(raw["gold"])
    correct_then_retracted = any(e.normalized_value == gold and e.retracted for e in events)
    generation_hardware = "RTX4090" if "4090_completed_records" in str(path) else "RTX3090"
    return {
        "question_id": raw.get("problem_id") or raw.get("question_id"),
        "condition": condition,
        "source_path": str(path),
        "source_file_sha256": sha256_file(path),
        "artifact_storage_host": "nlpg-SYS-4029GP-TRT",
        "seed": int(raw["seed"]),
        "hardware": generation_hardware,
        "hardware_resolution": "EXACT_SOURCE_PACKAGE_PATH_AND_FROZEN_MIGRATION_MANIFEST",
        "instance_id": raw.get("parallel_instance_id"),
        "gpu_ids": raw.get("gpu_ids"),
        "output_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "generated_tokens": len(tokens),
        "max_new_tokens": max_new,
        "prompt_tokens": raw.get("prompt_tokens"),
        "final_answer": raw.get("v4_extracted_answer"),
        "gold_answer": gold,
        "correct": bool(raw["v4_correct"]),
        "abstain": bool(raw["v4_abstain"]),
        "explicit_eos": bool(raw.get("eos_generated")),
        "hit_max_length": hit_max,
        "finish_type": finish_type,
        "termination_category": "LENGTH_CEILING" if hit_max else ("EXPLICIT_EOS_OR_STOP" if finish_type == "stop" else str(finish_type).upper()),
        "successful_sample": bool(raw.get("successful_sample")),
        "token_provenance_valid": bool(raw.get("token_provenance_valid")),
        "runtime_error": raw.get("runtime_error"),
        "runtime_effective_sampling_params": raw.get("runtime_effective_sampling_params"),
        "scorer_sha256": raw.get("scorer_sha256"),
        "model": raw.get("model"),
        "state_quantization": raw.get("state_quantization"),
        "rotation": raw.get("rotation"),
        "config_hash": raw.get("experiment_config_hash"),
        **ratios,
        "longest_repeated_span": repeat["length"],
        "longest_repeat_first_pos": repeat["first"],
        "longest_repeat_second_pos": repeat["second"],
        "first_committed_claim_position": first_pos,
        "final_committed_claim_position": last_pos,
        "num_committed_final_claims": len(committed),
        "num_answer_changes": changes,
        "committed_value_sequence": compressed,
        "retracted_event_count": len(retracted_events),
        "explicit_retraction_cue_count": explicit_retraction_cues,
        "tokens_after_first_commitment": None if first_pos is None else len(tokens) - first_pos,
        "tokens_after_final_commitment": None if last_pos is None else len(tokens) - last_pos,
        "post_first_commitment_fraction": None if first_pos is None else (len(tokens) - first_pos) / len(tokens),
        "correct_claim_later_retracted": correct_then_retracted,
        "never_formed_stable_commitment": len(committed) == 0,
        "committed_timeline": timeline,
        "official_v4_result": official,
        "scorer_recomputed_result": {k: v for k, v in dataclasses.asdict(result).items() if k != "candidate_rejections"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path("/data/zypan/worktrees/ling-256k-length-sensitivity-v1")
    manifest = root / "experiments/LING_256K_LONG_HORIZON_V1/manifest.json"
    scorer_path = Path("/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/aime26_scorer_v4.py")
    scorer = load_module("frozen_v4", scorer_path)
    m = json.loads(manifest.read_text())
    sources = m["artifact_sources"]
    rows = []
    for q in range(11, 31):
        fp = root / sources["fp_state"] / f"aime26_{q:02d}_seed1.json"
        hkey = "int8_r128_value_h_early_1_12" if q <= 12 else "int8_r128_value_h_late_13_30"
        hp = root / sources[hkey] / f"aime26_{q:02d}_seed1.json"
        rows.append(analyze_one(fp, "FP", scorer))
        rows.append(analyze_one(hp, "H", scorer))
    scores = {c: sum(r["correct"] for r in rows if r["condition"] == c) for c in ("FP", "H")}
    if scores != {"FP": 14, "H": 9}:
        raise RuntimeError(f"baseline score mismatch: {scores}")
    payload = {
        "task": "LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1_BASELINE_EXTRACT",
        "read_only": True,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "source_workspace_root": str(root),
        "scorer_parser_path": str(scorer_path),
        "scorer_parser_sha256": sha256_file(scorer_path),
        "established_metric_path": "/data01/user2/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/long_generation_audit_static_v1/ling/code/analyze_long_generations.py",
        "established_metric_sha256": "8adc34093e59158a7bd2287f91a3ee167f04c3953ed82a86f18629eee82d14dc",
        "established_metric_implementation": "exact copied definitions: ngrams, unique_ratio, repeated_fraction, approximate_longest_repeat",
        "position_mapping": "ESTIMATED_FROM_NORMALIZED_CHARACTER_FRACTION_TO_STORED_TOKEN_COUNT",
        "unsupported_established_metrics": {
            "reasoning_oscillation_proxy": "NOT_AVAILABLE: established 81920-token segmented definition is not silently extended to 262144 context",
            "textual_loop_indicator": "NOT_AVAILABLE: established late-window definition ends at 81920 and is not silently redefined",
            "semantic_degeneration_indicator": "NOT_AVAILABLE: established late-window definition ends at 81920 and is not silently redefined",
        },
        "scores": scores,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "scores": scores, "rows": len(rows)}))


if __name__ == "__main__":
    main()
