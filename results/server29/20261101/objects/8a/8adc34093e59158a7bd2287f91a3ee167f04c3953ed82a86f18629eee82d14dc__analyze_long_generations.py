#!/usr/bin/env python3
"""Static, read-only audit of AIME26 long-generation trajectories.

The script reads existing formal JSONL shards and never writes into the input
tree.  It uses stored token ids for repetition metrics and the model's actual
tokenizer only to decode fixed token ranges and map text evidence back to token
positions.  Gold answers are used solely for diagnostic candidate trajectories;
they never influence extraction or formal scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SEGMENT_EDGES = (0, 4096, 8192, 16384, 32768, 49152, 65536, 81920)
SELF_CORRECTION_MARKERS = (
    "wait", "check", "verify", "reconsider", "however", "mistake",
    "correction", "recalculate", "let's redo", "another approach",
    "hold on", "actually", "on second thought", "重新检查", "重新计算",
    "等等", "换一种方法", "有误", "修正",
)
PROGRESSION_MARKERS = (
    "therefore", "hence", "thus", "so", "consequently", "we obtain",
    "we get", "所以", "因此", "故", "从而",
)
FINAL_MARKERS = (
    "final answer", "answer is", "the answer", "boxed", "最终答案", "答案是", "答案为",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            row["_source_path"] = str(path)
            row["_source_line"] = line_number
            yield row


def finish_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type", ""))
    return "" if value is None else str(value)


def ngrams(tokens: list[int], n: int) -> Iterable[tuple[int, ...]]:
    return (tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def unique_ratio(tokens: list[int], n: int) -> float | None:
    total = len(tokens) - n + 1
    if total <= 0:
        return None
    return len(set(ngrams(tokens, n))) / total


def repeated_fraction(tokens: list[int], n: int) -> float | None:
    ratio = unique_ratio(tokens, n)
    return None if ratio is None else 1.0 - ratio


def cosine_counter(left: Counter[int], right: Counter[int]) -> float | None:
    if not left or not right:
        return None
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    nl = math.sqrt(sum(value * value for value in left.values()))
    nr = math.sqrt(sum(value * value for value in right.values()))
    return dot / (nl * nr) if nl and nr else None


def approximate_longest_repeat(tokens: list[int]) -> dict[str, int | None]:
    """Find an approximate longest exact repeated token span.

    Lengths are searched from large to small with a 64-bit rolling hash.  This
    intentionally reports a lower bound, not an exact suffix-array result.
    """
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
                if tokens[previous : previous + length] == tokens[start : start + length]:
                    return {"length": length, "first": previous, "second": start}
            else:
                seen[key] = start
    return {"length": 0, "first": None, "second": None}


def marker_positions(text: str, markers: Iterable[str]) -> dict[str, list[int]]:
    lowered = text.lower()
    return {
        marker: [match.start() for match in re.finditer(re.escape(marker), lowered)]
        for marker in markers
    }


def parse_candidate_value(expression: str) -> int | None:
    cleaned = expression.replace(",", "").strip()
    values = re.findall(r"(?<![\w.])[+-]?\d{1,6}(?![\w.])", cleaned)
    if not values:
        return None
    try:
        return int(values[-1])
    except ValueError:
        return None


def candidate_events(text: str) -> list[dict[str, Any]]:
    patterns = (
        ("BOXED", "strong", re.compile(r"\\boxed\s*\{([^{}]{1,160})\}", re.I)),
        ("EXPLICIT_FINAL", "strong", re.compile(
            r"(?:final\s+answer|the\s+answer\s+is|answer\s*(?:is|=|:)|最终答案|答案(?:是|为))"
            r"[^\d+\-]{0,40}([+\-]?\d{1,6})", re.I)),
        ("CONCLUSION", "tentative", re.compile(
            r"(?:therefore|hence|thus|consequently|所以|因此|故|从而)[^\n]{0,180}?"
            r"(?<![\w.])([+\-]?\d{1,6})(?![\w.])", re.I)),
        ("STANDALONE_INTEGER", "tentative", re.compile(r"(?m)^\s*\$?([+\-]?\d{1,6})\$?[.!]?\s*$")),
    )
    events: list[dict[str, Any]] = []
    for event_type, confidence, pattern in patterns:
        for match in pattern.finditer(text):
            expression = match.group(1)
            value = parse_candidate_value(expression)
            events.append({
                "char_start": match.start(),
                "char_end": match.end(),
                "candidate": value,
                "expression": expression,
                "type": event_type,
                "confidence": confidence,
                "excerpt": text[max(0, match.start() - 80) : min(len(text), match.end() + 120)].replace("\n", " "),
            })
    events.sort(key=lambda item: (item["char_start"], item["char_end"], item["type"]))
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int | None]] = set()
    for event in events:
        key = (event["char_start"], event["char_end"], event["candidate"])
        if key not in seen:
            seen.add(key)
            deduplicated.append(event)
    return deduplicated


def map_char_to_token(offsets: list[tuple[int, int]], char_position: int, token_count: int, text_length: int) -> int:
    if offsets:
        starts = [item[0] for item in offsets]
        index = max(0, bisect_right(starts, char_position) - 1)
        return min(index, max(0, token_count - 1))
    if text_length <= 0:
        return 0
    return min(max(0, int(char_position / text_length * token_count)), max(0, token_count - 1))


def encode_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(encoded["input_ids"]), [tuple(item) for item in encoded["offset_mapping"]]
    except Exception:
        return list(tokenizer.encode(text, add_special_tokens=False)), []


def analyze_segment(tokenizer: Any, all_tokens: list[int], start: int, end: int,
                    prior_counters: list[Counter[int]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokens = all_tokens[start:end]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    encoded, offsets = encode_offsets(tokenizer, text)
    token_roundtrip_delta = len(encoded) - len(tokens)
    counter = Counter(tokens)
    adjacent = cosine_counter(prior_counters[-1], counter) if prior_counters else None
    history = [cosine_counter(previous, counter) for previous in prior_counters]
    history = [value for value in history if value is not None]
    prior_counters.append(counter)

    marker_map = marker_positions(text, SELF_CORRECTION_MARKERS + PROGRESSION_MARKERS + FINAL_MARKERS)
    events = candidate_events(text)
    for event in events:
        event["token_position"] = start + map_char_to_token(offsets, event["char_start"], len(tokens), len(text))
        event["segment_start"] = start
        event["segment_end"] = end
        event.pop("char_start", None)
        event.pop("char_end", None)

    row = {
        "segment_start": start,
        "segment_end": end,
        "segment_tokens": len(tokens),
        "unique_unigram_ratio": unique_ratio(tokens, 1),
        "unique_bigram_ratio": unique_ratio(tokens, 2),
        "unique_4gram_ratio": unique_ratio(tokens, 4),
        "repeated_8gram_fraction": repeated_fraction(tokens, 8),
        "repeated_16gram_fraction": repeated_fraction(tokens, 16),
        "repeated_32gram_fraction": repeated_fraction(tokens, 32),
        "adjacent_window_lexical_cosine": adjacent,
        "max_historical_window_lexical_cosine": max(history) if history else None,
        "self_correction_count": sum(len(marker_map[item]) for item in SELF_CORRECTION_MARKERS),
        "progression_marker_count": sum(len(marker_map[item]) for item in PROGRESSION_MARKERS),
        "final_marker_count": sum(len(marker_map[item]) for item in FINAL_MARKERS),
        "candidate_answers": json.dumps([event["candidate"] for event in events], ensure_ascii=False),
        "token_roundtrip_delta": token_roundtrip_delta,
        "text_chars": len(text),
    }
    return row, events


def load_v4(path: str | None):
    if not path:
        return None
    scorer_path = Path(path)
    spec = importlib.util.spec_from_file_location("frozen_aime26_v4", scorer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import V4 scorer: {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def classify(metrics: dict[str, Any]) -> tuple[str, str, list[str]]:
    rep16 = metrics["repeated_16gram_fraction"] or 0.0
    rep32 = metrics["repeated_32gram_fraction"] or 0.0
    longest = metrics["longest_repeated_span"] or 0
    late_rep16 = metrics["late_repeated_16gram_fraction"] or 0.0
    late_u4 = metrics["late_unique_4gram_ratio"]
    late_u4 = 1.0 if late_u4 is None else late_u4
    changes = metrics["strong_num_candidate_changes"]
    corrections = metrics["self_correction_count"]
    aba = metrics["strong_candidate_aba_count"]
    first_gold = metrics["first_gold_strong_candidate_pos"]
    total = metrics["num_generated_tokens"]
    v4_correct = metrics["v4_correct"]

    # The strict repetition signal deliberately ignores ordinary reuse of
    # mathematical phrases.  It requires a repeated long span, very high late
    # 16-gram reuse, or high global 32-gram reuse.
    repetition = longest >= 512 or rep32 >= 0.05 or late_rep16 >= 0.15
    oscillation = (changes >= 4 and corrections >= 400) or aba >= 2
    commit = first_gold is not None and total - first_gold >= 4096 and not v4_correct
    degeneration = longest >= 1024 and late_rep16 >= 0.20 and late_u4 <= 0.50
    signals = []
    if repetition:
        signals.append("TEXTUAL_REPETITION")
    if oscillation:
        signals.append("ANSWER_OSCILLATION")
    if commit:
        signals.append("EARLY_GOLD_WITHOUT_COMMIT")
    if degeneration:
        signals.append("TEXTUAL_DEGENERATION_CANDIDATE")

    if degeneration and (oscillation or commit):
        return "MIXED_FAILURE", "HIGH", signals
    if degeneration:
        return "SEMANTIC_DEGENERATION", "MEDIUM", signals
    if repetition and oscillation:
        return "MIXED_FAILURE", "HIGH", signals
    if repetition:
        return "TEXTUAL_REPETITION_LOOP", "HIGH" if longest >= 1024 else "MEDIUM", signals
    if oscillation:
        return "REASONING_OSCILLATION", "MEDIUM", signals
    if commit:
        return "ANSWER_COMMIT_FAILURE", "MEDIUM", signals
    if rep16 < 0.02 and late_rep16 < 0.03 and corrections < 500 and changes < 3:
        return "PRODUCTIVE_LONG_REASONING", "LOW", signals
    return "UNCLEAR", "LOW", signals


def common_prefix(left: list[int], right: list[int]) -> int:
    size = min(len(left), len(right))
    for index in range(size):
        if left[index] != right[index]:
            return index
    return size


def load_rows(input_root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    for path in sorted(input_root.glob("worker*/*.jsonl")):
        rows = list(iter_jsonl(path))
        by_condition[path.stem].extend(rows)
        sources.append({
            "path": str(path),
            "condition": path.stem,
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return by_condition, sources


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen", "ling"), required=True)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Directory containing worker*/<condition>.jsonl")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v4-scorer")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    v4 = load_v4(args.v4_scorer)
    by_condition, sources = load_rows(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected = ("fp_state", "int8_c128", "int8_c128_key_h") if args.model == "qwen" else (
        "fp_state", "int8_r128", "int8_r128_value_h")
    native_condition = expected[1]
    h_condition = expected[2]
    row_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    sample_details: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for condition, rows in sorted(by_condition.items()):
        identities: set[tuple[str, int]] = set()
        for raw in rows:
            identity = (str(raw["problem_id"]), int(raw["seed"]))
            if identity in identities:
                raise RuntimeError(f"duplicate identity in {condition}: {identity}")
            identities.add(identity)
            tokens = [int(value) for value in raw.get("generated_token_ids", [])]
            if len(tokens) != int(raw.get("generated_token_count", -1)):
                raise RuntimeError(f"token provenance mismatch: {condition} {identity}")
            hit_limit = len(tokens) == int(raw.get("max_new_tokens", 81920)) and finish_type(raw.get("finish_reason")) == "length"
            # Segment every completed trajectory so FP/native/Hadamard pairs
            # have comparable progression and candidate timelines.
            should_segment = True
            candidates: list[dict[str, Any]] = []
            segments: list[dict[str, Any]] = []
            if should_segment:
                prior: list[Counter[int]] = []
                for left, right in zip(SEGMENT_EDGES, SEGMENT_EDGES[1:]):
                    if left >= len(tokens):
                        break
                    end = min(right, len(tokens))
                    segment, events = analyze_segment(tokenizer, tokens, left, end, prior)
                    segment.update({
                        "model": args.model,
                        "condition": condition,
                        "problem": raw["problem_id"],
                        "problem_id": raw["problem_id"],
                        "seed": int(raw["seed"]),
                        "gold_candidate_present": any(event["candidate"] == int(raw["gold_answer"]) for event in events),
                    })
                    segments.append(segment)
                    candidates.extend(events)
                    segment_rows.append(segment)

            candidate_values = [event["candidate"] for event in candidates if event["candidate"] is not None]
            compressed: list[int] = []
            for value in candidate_values:
                if not compressed or compressed[-1] != value:
                    compressed.append(value)
            aba_count = sum(1 for index in range(2, len(compressed)) if compressed[index] == compressed[index - 2] != compressed[index - 1])
            strong_candidates = [event for event in candidates if event["confidence"] == "strong" and event["candidate"] is not None]
            strong_values: list[int] = []
            for event in strong_candidates:
                value = int(event["candidate"])
                if not strong_values or strong_values[-1] != value:
                    strong_values.append(value)
            strong_aba_count = sum(
                1 for index in range(2, len(strong_values))
                if strong_values[index] == strong_values[index - 2] != strong_values[index - 1]
            )
            gold = int(raw["gold_answer"])
            gold_positions = [event["token_position"] for event in candidates if event["candidate"] == gold]
            strong_gold_positions = [
                event["token_position"] for event in strong_candidates if event["candidate"] == gold
            ]
            marker_map = marker_positions(raw.get("response", ""), SELF_CORRECTION_MARKERS)
            repeat = approximate_longest_repeat(tokens) if should_segment else {"length": 0, "first": None, "second": None}
            v4_status = None
            v4_correct = None
            v4_value = None
            if v4 is not None:
                result = v4.extract_v4(raw.get("response", ""), raw.get("finish_reason", ""), raw.get("problem", ""))
                v4_status = result.status
                v4_value = result.normalized_value
                v4_correct = bool(v4.compare_with_gold(result, raw.get("gold_answer")))

            last_segment = segments[-1] if segments else {}
            metrics = {
                "model": args.model,
                "condition": condition,
                "problem_id": raw["problem_id"],
                "seed": int(raw["seed"]),
                "gold": gold,
                "num_generated_tokens": len(tokens),
                "hit_limit": hit_limit,
                "finish_type": finish_type(raw.get("finish_reason")),
                "eos_seen": bool(raw.get("eos_seen")),
                "truncated": bool(raw.get("truncated")),
                "v4_status": v4_status,
                "v4_value": v4_value,
                "v4_correct": v4_correct,
                "int8_candidate_count": len(candidates),
                "first_gold_candidate_pos": min(gold_positions) if gold_positions else None,
                "last_gold_candidate_pos": max(gold_positions) if gold_positions else None,
                "num_candidate_changes": max(0, len(compressed) - 1),
                "candidate_aba_count": aba_count,
                "candidate_integer_sequence": compressed,
                "strong_candidate_count": len(strong_candidates),
                "strong_num_candidate_changes": max(0, len(strong_values) - 1),
                "strong_candidate_aba_count": strong_aba_count,
                "strong_candidate_integer_sequence": strong_values,
                "first_gold_strong_candidate_pos": min(strong_gold_positions) if strong_gold_positions else None,
                "last_gold_strong_candidate_pos": max(strong_gold_positions) if strong_gold_positions else None,
                "self_correction_count": sum(len(value) for value in marker_map.values()),
                "unique_unigram_ratio": unique_ratio(tokens, 1) if should_segment else None,
                "unique_bigram_ratio": unique_ratio(tokens, 2) if should_segment else None,
                "unique_4gram_ratio": unique_ratio(tokens, 4) if should_segment else None,
                "repeated_8gram_fraction": repeated_fraction(tokens, 8) if should_segment else None,
                "repeated_16gram_fraction": repeated_fraction(tokens, 16) if should_segment else None,
                "repeated_32gram_fraction": repeated_fraction(tokens, 32) if should_segment else None,
                "longest_repeated_span": repeat["length"],
                "longest_repeat_first_pos": repeat["first"],
                "longest_repeat_second_pos": repeat["second"],
                "late_repeated_16gram_fraction": last_segment.get("repeated_16gram_fraction"),
                "late_unique_4gram_ratio": last_segment.get("unique_4gram_ratio"),
                "source_path": raw["_source_path"],
                "source_line": raw["_source_line"],
                "candidates": candidates,
                "segments": segments,
                "generated_token_ids": tokens,
            }
            row_index[(condition, identity[0], identity[1])] = metrics
            sample_details.append(metrics)

    # Pair native hit-limit rows to FP and any available Hadamard control.
    pair_rows: list[dict[str, Any]] = []
    for (condition, problem_id, seed), native in sorted(row_index.items()):
        if condition != native_condition or not native["hit_limit"]:
            continue
        fp = row_index.get(("fp_state", problem_id, seed))
        hrow = row_index.get((h_condition, problem_id, seed))
        pair = {
            "model": args.model,
            "problem_id": problem_id,
            "seed": seed,
            "native_tokens": native["num_generated_tokens"],
            "native_v4_correct": native["v4_correct"],
            "native_first_gold_candidate_pos": native["first_gold_candidate_pos"],
            "fp_available": fp is not None,
            "fp_tokens": fp["num_generated_tokens"] if fp else None,
            "fp_finish_type": fp["finish_type"] if fp else None,
            "fp_v4_correct": fp["v4_correct"] if fp else None,
            "fp_first_gold_candidate_pos": fp["first_gold_candidate_pos"] if fp else None,
            "fp_first_gold_strong_candidate_pos": fp["first_gold_strong_candidate_pos"] if fp else None,
            "fp_self_correction_count": fp["self_correction_count"] if fp else None,
            "fp_repeated_16gram_fraction": fp["repeated_16gram_fraction"] if fp else None,
            "fp_common_prefix_tokens": common_prefix(native["generated_token_ids"], fp["generated_token_ids"]) if fp else None,
            "hadamard_available": hrow is not None,
            "hadamard_tokens": hrow["num_generated_tokens"] if hrow else None,
            "hadamard_finish_type": hrow["finish_type"] if hrow else None,
            "hadamard_v4_correct": hrow["v4_correct"] if hrow else None,
            "hadamard_first_gold_candidate_pos": hrow["first_gold_candidate_pos"] if hrow else None,
            "hadamard_first_gold_strong_candidate_pos": hrow["first_gold_strong_candidate_pos"] if hrow else None,
            "hadamard_self_correction_count": hrow["self_correction_count"] if hrow else None,
            "hadamard_repeated_16gram_fraction": hrow["repeated_16gram_fraction"] if hrow else None,
            "hadamard_common_prefix_tokens": common_prefix(native["generated_token_ids"], hrow["generated_token_ids"]) if hrow else None,
        }
        pair_rows.append(pair)
        native["fp_num_tokens"] = pair["fp_tokens"]
        native["fp_correct"] = pair["fp_v4_correct"]
        native["hadamard_available"] = pair["hadamard_available"]
        native["hadamard_tokens"] = pair["hadamard_tokens"]
        native["hadamard_correct"] = pair["hadamard_v4_correct"]

    for metrics in sample_details:
        if not metrics["hit_limit"]:
            continue
        failure_class, confidence, signals = classify(metrics)
        evidence = []
        if metrics["first_gold_candidate_pos"] is not None:
            evidence.append(f"first_gold={metrics['first_gold_candidate_pos']}")
        if metrics["longest_repeated_span"]:
            evidence.append(
                f"repeat={metrics['longest_repeated_span']}@{metrics['longest_repeat_first_pos']},{metrics['longest_repeat_second_pos']}")
        evidence.append(f"corrections={metrics['self_correction_count']}")
        evidence.append(f"candidate_changes={metrics['num_candidate_changes']}")
        label_rows.append({
            "model": metrics["model"],
            "condition": metrics["condition"],
            "problem_id": metrics["problem_id"],
            "seed": metrics["seed"],
            "gold": metrics["gold"],
            "v4_result": "CORRECT" if metrics["v4_correct"] else metrics["v4_status"],
            "num_generated_tokens": metrics["num_generated_tokens"],
            "hit_limit": metrics["hit_limit"],
            "fp_num_tokens": metrics.get("fp_num_tokens"),
            "fp_correct": metrics.get("fp_correct"),
            "int8_candidate_count": metrics["int8_candidate_count"],
            "first_gold_candidate_pos": metrics["first_gold_candidate_pos"],
            "last_gold_candidate_pos": metrics["last_gold_candidate_pos"],
            "num_candidate_changes": metrics["num_candidate_changes"],
            "strong_candidate_count": metrics["strong_candidate_count"],
            "strong_num_candidate_changes": metrics["strong_num_candidate_changes"],
            "strong_candidate_aba_count": metrics["strong_candidate_aba_count"],
            "first_gold_strong_candidate_pos": metrics["first_gold_strong_candidate_pos"],
            "last_gold_strong_candidate_pos": metrics["last_gold_strong_candidate_pos"],
            "self_correction_count": metrics["self_correction_count"],
            "repeated_8gram_fraction": metrics["repeated_8gram_fraction"],
            "repeated_16gram_fraction": metrics["repeated_16gram_fraction"],
            "repeated_32gram_fraction": metrics["repeated_32gram_fraction"],
            "longest_repeated_span": metrics["longest_repeated_span"],
            "failure_class": failure_class,
            "confidence": confidence,
            "evidence_excerpt_positions": ";".join(evidence),
            "signals": "|".join(signals),
            "hadamard_available": metrics.get("hadamard_available"),
            "hadamard_tokens": metrics.get("hadamard_tokens"),
            "hadamard_correct": metrics.get("hadamard_correct"),
        })

    # Token arrays are intentionally omitted from the durable detailed output.
    durable_details = []
    for metrics in sample_details:
        copy = dict(metrics)
        copy.pop("generated_token_ids", None)
        durable_details.append(copy)
    with (args.output_dir / "sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in durable_details:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(args.output_dir / "long_generation_segment_metrics.csv", segment_rows)
    write_csv(args.output_dir / "long_generation_failure_labels.csv", label_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", pair_rows)
    manifest = {
        "model": args.model,
        "input_root": str(args.input_root),
        "tokenizer": args.tokenizer,
        "v4_scorer": args.v4_scorer,
        "v4_scorer_sha256": sha256_file(Path(args.v4_scorer)) if args.v4_scorer else None,
        "sources": sources,
        "conditions": {name: len(by_condition.get(name, [])) for name in expected},
        "hit_limit_counts": {
            name: sum(1 for item in sample_details if item["condition"] == name and item["hit_limit"])
            for name in expected
        },
        "complete_generated_text": all(isinstance(row.get("response"), str) for rows in by_condition.values() for row in rows),
        "token_ids": all(isinstance(row.get("generated_token_ids"), list) for rows in by_condition.values() for row in rows),
        "logits": False,
        "per_step_recurrent_state_statistics": False,
        "eos_metadata": True,
        "matched_fp_int8_possible": True,
        "classification_thresholds": {
            "repetition": "longest_repeat>=512 OR repeated_32gram>=0.05 OR late_repeated_16gram>=0.15",
            "oscillation": "strong_candidate_changes>=4 AND corrections>=400 OR strong_ABA_count>=2",
            "answer_commit_failure": "strong gold candidate >=4096 tokens before end AND V4 not correct",
            "semantic_degeneration_candidate": "longest_repeat>=1024 AND late repeated_16gram>=0.20 AND late unique_4gram<=0.50",
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "model": args.model,
        "conditions": manifest["conditions"],
        "hit_limit_counts": manifest["hit_limit_counts"],
        "labels": dict(Counter(row["failure_class"] for row in label_rows)),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
