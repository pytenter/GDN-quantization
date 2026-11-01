#!/usr/bin/env python3
"""Supplementary AIME answer extraction; never replaces the frozen strict metric."""

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from aime26_common import (
    DIAGNOSTIC_SCORER_VERSION,
    STRICT_SCORER_VERSION,
    _numeric_fraction,
    score_aime_strict_v1,
)


PARSED = "PARSED"
MULTIPLE = "MULTIPLE_CANDIDATES"
NO_ANSWER = "NO_ANSWER_FOUND"

_MATH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[+-]\s*)?\\(?:d?frac)\s*\{\s*[+-]?\d+\s*\}\s*\{\s*[+-]?\d+\s*\}"
    r"|(?<![A-Za-z0-9_])[+-]?\d+\s*/\s*[+-]?\d+"
    r"|(?<![A-Za-z0-9_])[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?"
)
_FINAL_LABEL = re.compile(
    r"(?i)(?:\*{1,2}|_{1,2})?\s*final\s+answer"
    r"\s*(?:\*{1,2}|_{1,2})?\s*(?::|=)?\s*(?:\*{1,2}|_{1,2})?"
)
_ANSWER_IS = re.compile(
    r"(?i)(?:(?:therefore|hence|thus|so)\s*,?\s*)?"
    r"(?:the\s+)?(?:final\s+)?answer\s+is\s*"
)


def _canonical_diagnostic(value):
    value = (value or "").strip()
    value = value.replace("−", "-").replace("\\left", "").replace("\\right", "")
    value = re.sub(r"^[\x60*_\s]+|[\x60*_\s.,;:!?]+$", "", value)
    if value.startswith("\\(") and value.endswith("\\)"):
        value = value[2:-2].strip()
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    value = re.sub(r"^[\x60*_\s]+|[\x60*_\s.,;:!?]+$", "", value)
    signed_latex = re.fullmatch(
        r"([+-]?)\s*\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}",
        value,
    )
    if signed_latex:
        sign, numerator, denominator = signed_latex.groups()
        numerator = int(numerator)
        denominator = int(denominator)
        if denominator == 0:
            return ""
        if sign == "-":
            numerator = -numerator
        result = Fraction(numerator, denominator)
        return str(result.numerator) if result.denominator == 1 else f"{result.numerator}/{result.denominator}"
    percentage = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%", value)
    if percentage:
        try:
            result = Fraction(Decimal(percentage.group(1))) / 100
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return ""
        return str(result.numerator) if result.denominator == 1 else f"{result.numerator}/{result.denominator}"
    parsed = _numeric_fraction(value)
    if parsed is None:
        return ""
    return str(parsed.numerator) if parsed.denominator == 1 else f"{parsed.numerator}/{parsed.denominator}"


def _math_candidates(text):
    values = []
    for match in _MATH_TOKEN.finditer(text or ""):
        value = _canonical_diagnostic(match.group(0))
        if value:
            values.append(value)
    return values


def _boxed_candidates(text):
    values = []
    for match in re.finditer(r"\\boxed\s*\{", text or ""):
        start = match.end()
        depth = 1
        for offset in range(start, len(text)):
            char = text[offset]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    value = _canonical_diagnostic(text[start:offset])
                    if value:
                        values.append(value)
                    break
    return values


def _nearby_after(text, start, stop=None):
    fragment = (text or "")[start:stop]
    lines = fragment.splitlines()
    selected = []
    for line in lines:
        if line.strip():
            selected.append(line)
        elif selected:
            break
        if len(selected) >= 4:
            break
    return "\n".join(selected)[:320]


def _explicit_final_candidates(text):
    matches = list(_FINAL_LABEL.finditer(text or ""))
    values = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else None
        nearby = _nearby_after(text, match.end(), stop)
        candidates = _math_candidates(nearby)
        if candidates:
            values.append(candidates[-1])
    return values


def _answer_phrase_candidates(text):
    values = []
    for match in _ANSWER_IS.finditer(text or ""):
        nearby = (text or "")[match.end():match.end() + 120]
        candidates = _math_candidates(nearby)
        if candidates:
            values.append(candidates[0])
    return values


def _standalone_final_candidate(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[-1]
    cleaned = re.sub(r"^[\x60*_\s]+|[\x60*_\s.,;:!?]+$", "", line)
    if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
        cleaned = cleaned[2:-2].strip()
    if cleaned.startswith("$") and cleaned.endswith("$"):
        cleaned = cleaned[1:-1].strip()
    return _canonical_diagnostic(cleaned)


def _result(answer, source, confidence, status, candidates):
    return {
        "diagnostic_extracted_answer": answer,
        "diagnostic_parse_source": source,
        "diagnostic_parse_confidence": confidence,
        "diagnostic_parse_status": status,
        "diagnostic_candidates": list(candidates),
    }


def extract_aime_answer_diagnostic_v1(text):
    boxed = _boxed_candidates(text)
    if boxed:
        unique = list(dict.fromkeys(boxed))
        status = MULTIPLE if len(unique) > 1 else PARSED
        return _result(boxed[-1], "BOXED", "HIGH", status, boxed)

    explicit = _explicit_final_candidates(text)
    if explicit:
        unique = list(dict.fromkeys(explicit))
        if len(unique) > 1:
            return _result(explicit[-1], "LAST_EXPLICIT_FINAL_ANSWER", "HIGH", MULTIPLE, explicit)
        return _result(explicit[-1], "FINAL_ANSWER_LABEL", "HIGH", PARSED, explicit)

    phrases = _answer_phrase_candidates(text)
    if phrases:
        unique = list(dict.fromkeys(phrases))
        if len(unique) > 1:
            return _result(phrases[-1], "ANSWER_IS_PHRASE", "MEDIUM", MULTIPLE, phrases)
        return _result(phrases[-1], "ANSWER_IS_PHRASE", "MEDIUM", PARSED, phrases)

    standalone = _standalone_final_candidate(text)
    if standalone:
        return _result(standalone, "FINAL_STANDALONE_EXPRESSION", "LOW", PARSED, [standalone])

    return _result("", "NO_ANSWER_FOUND", "NONE", NO_ANSWER, [])


def score_aime_layers(response, gold):
    strict_extracted, strict_correct = score_aime_strict_v1(response, gold)
    diagnostic = extract_aime_answer_diagnostic_v1(response)
    predicted = _numeric_fraction(diagnostic["diagnostic_extracted_answer"])
    reference = _numeric_fraction(str(gold))
    diagnostic_correct = predicted is not None and reference is not None and predicted == reference

    strict_parsed = bool(strict_extracted)
    diagnostic_parsed = bool(diagnostic["diagnostic_extracted_answer"])
    if strict_correct and diagnostic_correct:
        disagreement = "AGREE_CORRECT"
    elif strict_correct and not diagnostic_correct:
        disagreement = "STRICT_CORRECT_DIAGNOSTIC_WRONG"
    elif not strict_correct and diagnostic_correct:
        disagreement = "STRICT_WRONG_DIAGNOSTIC_CORRECT"
    elif not strict_parsed and not diagnostic_parsed:
        disagreement = "BOTH_UNPARSED"
    elif not strict_parsed and diagnostic_parsed:
        disagreement = "STRICT_UNPARSED_DIAGNOSTIC_PARSED"
    else:
        disagreement = "AGREE_WRONG"

    boxed_present = bool(re.search(r"\\boxed\s*\{", response or ""))
    source = diagnostic["diagnostic_parse_source"]
    if boxed_present and source == "BOXED":
        compliance = "BOXED_COMPLIANT"
    elif source in {"FINAL_ANSWER_LABEL", "LAST_EXPLICIT_FINAL_ANSWER"}:
        compliance = "EXPLICIT_FINAL_ANSWER_NONBOXED"
    elif diagnostic_parsed:
        compliance = "NONSTANDARD_EXTRACTABLE"
    else:
        compliance = "UNEXTRACTABLE"

    return {
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_extracted_answer": strict_extracted,
        "strict_correct": bool(strict_correct),
        "strict_parse_status": "PARSED" if strict_parsed else "UNPARSED",
        "diagnostic_scorer_version": DIAGNOSTIC_SCORER_VERSION,
        **diagnostic,
        "diagnostic_correct": bool(diagnostic_correct),
        "answer_format_compliance": compliance,
        "boxed_answer_present": boxed_present,
        "strict_vs_diagnostic_status": disagreement,
    }
