#!/usr/bin/env python3
"""Shared deterministic dataset, scoring, and statistics helpers for AIME 2026."""

import hashlib
import json
import math
import random
import re
from pathlib import Path


TASK = "GDN_KDA_AIME26_2SEED_FORMAL_V1"


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


def extract_aime_answer(text):
    candidate = last_boxed(text)
    if candidate is None:
        patterns = (
            r"(?:final answer|answer is|answer:)\s*\$?([^\n\.]+)",
            r"(?:最终答案|答案是)\s*[:：]?\s*([^\n。]+)",
        )
        values = []
        for pattern in patterns:
            values.extend(re.findall(pattern, text or "", flags=re.I))
        candidate = values[-1] if values else None
    if candidate is None:
        return ""
    cleaned = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", candidate)
    cleaned = cleaned.replace(",", "").replace("$", "").strip()
    integers = re.findall(r"(?<!\d)\d{1,3}(?!\d)", cleaned)
    return integers[-1] if integers else ""


def score_aime(response, gold):
    extracted = extract_aime_answer(response)
    try:
        correct = int(extracted) == int(gold)
    except (TypeError, ValueError):
        correct = False
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
