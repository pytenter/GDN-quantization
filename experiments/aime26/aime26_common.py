#!/usr/bin/env python3
"""Shared deterministic dataset, scoring, and statistics helpers for AIME 2026."""

import hashlib
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path


TASK = "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2"


def load_frozen_dataset(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 30:
        raise RuntimeError(f"AIME26 dataset must contain 30 rows, found {len(rows)}")
    if [int(row["problem_idx"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError("AIME26 problem_idx must be exactly 1..30")
    return rows


def raw_messages(problem):
    return [{"role": "user", "content": str(problem)}]


def ids_sha256(input_ids):
    values = [int(value) for value in input_ids]
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def last_boxed(text):
    matches = list(re.finditer(r"\\boxed\s*\{", text or ""))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for offset in range(start, len(text)):
        char = text[offset]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:offset]
    return None


def _numeric_fraction(value):
    """Parse a complete integer, decimal, or LaTeX/simple fraction exactly."""
    value = (value or "").strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "").replace(",", "").strip()
    latex_fraction = re.fullmatch(
        r"\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}",
        value,
    )
    if latex_fraction:
        numerator, denominator = map(int, latex_fraction.groups())
        return None if denominator == 0 else Fraction(numerator, denominator)
    simple_fraction = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", value)
    if simple_fraction:
        numerator, denominator = map(int, simple_fraction.groups())
        return None if denominator == 0 else Fraction(numerator, denominator)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        try:
            return Fraction(Decimal(value))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None
    return None


def _canonical_fraction(value):
    parsed = _numeric_fraction(value)
    if parsed is None:
        return ""
    return str(parsed.numerator) if parsed.denominator == 1 else f"{parsed.numerator}/{parsed.denominator}"


def extract_aime_answer(text):
    candidate = last_boxed(text)
    if candidate is None:
        patterns = (r"(?:final answer|answer is|answer:)\s*\$?([^\n]+)",)
        values = []
        for pattern in patterns:
            values.extend(re.findall(pattern, text or "", flags=re.I))
        candidate = values[-1] if values else None
    if candidate is None:
        return ""
    cleaned = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", candidate)
    direct = _canonical_fraction(cleaned)
    if direct:
        return direct
    if re.search(r"\\(?:d?frac)", cleaned):
        return ""
    numeric_tokens = re.findall(
        r"\\(?:d?frac)\s*\{\s*[+-]?\d+\s*\}\s*\{\s*[+-]?\d+\s*\}"
        r"|[+-]?\d+\s*/\s*[+-]?\d+"
        r"|[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
        cleaned,
    )
    return _canonical_fraction(numeric_tokens[-1]) if numeric_tokens else ""


def score_aime(response, gold):
    extracted = extract_aime_answer(response)
    predicted = _numeric_fraction(extracted)
    reference = _numeric_fraction(str(gold))
    correct = predicted is not None and reference is not None and predicted == reference
    return extracted, bool(correct)


def exact_mcnemar_p(rescue, regression):
    discordant = int(rescue) + int(regression)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(rescue, regression) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def problem_bootstrap_delta(rows, baseline_method, method, samples=100000, seed=20260917):
    by_problem = {}
    for row in rows:
        if row["method"] not in (baseline_method, method):
            continue
        key = int(row["problem_id"].rsplit("_", 1)[-1])
        by_problem.setdefault(key, {}).setdefault(row["method"], {})[int(row["seed"])] = int(bool(row["correct"]))
    if sorted(by_problem) != list(range(1, 31)):
        raise RuntimeError("bootstrap requires all 30 problems")
    deltas = []
    rng = random.Random(seed)
    problems = list(range(1, 31))
    for _ in range(samples):
        selected = [rng.choice(problems) for _ in problems]
        base = sum(by_problem[p][baseline_method][s] for p in selected for s in (1, 2))
        rotated = sum(by_problem[p][method][s] for p in selected for s in (1, 2))
        deltas.append((rotated - base) / 60.0)
    deltas.sort()
    lo = deltas[int(0.025 * samples)]
    hi = deltas[min(samples - 1, int(0.975 * samples))]
    return {"samples": samples, "resample_unit": "problem", "seed": seed, "delta_accuracy": [lo, hi]}
