#!/usr/bin/env python3
"""AIME26 Strict V4 candidate: rightmost committed final claim.

The extractor is deterministic and gold-blind.  Candidate generation,
classification, selection, normalization, domain validation, and comparison
with gold are deliberately separate stages.  ``</think>`` is only a reasoning
boundary; it never promotes a claim to final-answer status.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Optional


SCORER_NAME = "AIME26_STRICT_V4_CANDIDATE"
STATUS = "CANDIDATE"
DESIGN = "RIGHTMOST_COMMITTED_FINAL_CLAIM"
FIRST_INTEGER_FALLBACK = False
GLOBAL_LAST_INTEGER_FALLBACK = False
GOLD_USED_FOR_CANDIDATE_SELECTION = False
THINK_BOUNDARY_TREATED_AS_FINAL_REGION = False


REJECTION_ORDER = (
    "TENTATIVE",
    "RETRACTED",
    "CONDITIONAL",
    "NEGATED",
    "PROBLEM_RESTATEMENT",
    "INTERMEDIATE_ASSIGNMENT",
    "TRUNCATED_WEAK",
)


@dataclass
class ClaimEvent:
    event_type: str
    start: int
    end: int
    raw_span: str
    claimed_expression: str
    normalized_value: Optional[int]
    strength: str
    region: str
    rule: str
    candidate_sentence: str
    conditional: bool = False
    negated: bool = False
    tentative: bool = False
    retracted: bool = False
    problem_restatement: bool = False
    intermediate_assignment: bool = False
    truncated_weak: bool = False
    committed: bool = False

    def rejection_reasons(self) -> list[str]:
        flags = {
            "TENTATIVE": self.tentative,
            "RETRACTED": self.retracted,
            "CONDITIONAL": self.conditional,
            "NEGATED": self.negated,
            "PROBLEM_RESTATEMENT": self.problem_restatement,
            "INTERMEDIATE_ASSIGNMENT": self.intermediate_assignment,
            "TRUNCATED_WEAK": self.truncated_weak,
        }
        return [reason for reason in REJECTION_ORDER if flags[reason]]

    def rejection_dict(self) -> dict[str, Any]:
        return {
            "expression": self.claimed_expression,
            "reason": self.rejection_reasons(),
            "rule": self.rule,
            "span_start": self.start,
            "span_end": self.end,
            "candidate_sentence": self.candidate_sentence,
        }


@dataclass
class ExtractionResult:
    status: str
    claimed_expression: Optional[str]
    normalized_value: Optional[int]
    aime_domain_valid: Optional[bool]
    selected_event_type: Optional[str]
    selected_rule: Optional[str]
    evidence_span: Optional[str]
    span_start: Optional[int]
    span_end: Optional[int]
    termination_reason: str
    normalization_actions: list[str] = field(default_factory=list)
    candidate_count: int = 0
    committed_candidate_count: int = 0
    candidate_rejections: list[dict[str, Any]] = field(default_factory=list)
    abstain_reason: Optional[str] = None

    @property
    def has_claim(self) -> bool:
        return self.status == "EXTRACTED"

    @property
    def event_type(self) -> Optional[str]:
        return self.selected_event_type

    @property
    def rule(self) -> Optional[str]:
        return self.selected_rule

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _termination_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def _is_truncated(termination_reason: str) -> bool:
    lowered = termination_reason.lower()
    return "max_tokens" in lowered or "\"type\": \"length\"" in lowered or lowered.strip() == "length"


def _normalize_nonsemantic(raw_text: str) -> tuple[str, list[str]]:
    normalized = raw_text
    actions: list[str] = []
    if "\x08oxed{" in normalized:
        normalized = normalized.replace("\x08oxed{", r"\boxed{")
        actions.append("REPAIR_BACKSPACE_BOXED")
    return normalized, actions


def _strip_math_wrappers(expression: str) -> str:
    text = expression.strip().replace("−", "-")
    changed = True
    while changed and text:
        previous = text
        text = re.sub(r"^[`*_\s]+|[`*_\s,;:!?]+$", "", text)
        text = re.sub(r"^\$+|\$+$", "", text).strip()
        if text.endswith(r"\)") or text.endswith(r"\]"):
            text = text[:-2].strip()
        if text.startswith(r"\(") and text.endswith(r"\)"):
            text = text[2:-2].strip()
        if text.startswith(r"\[") and text.endswith(r"\]"):
            text = text[2:-2].strip()
        text = re.sub(r"\\left\b|\\right\b", "", text)
        text = text.rstrip(".,;:!?").strip()
        changed = text != previous
    return text


def _latex_to_arithmetic(expression: str) -> str:
    text = expression
    fraction_pattern = re.compile(
        r"\\(?:d?frac)\s*\{\s*([+-]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([+-]?\d+(?:\.\d+)?)\s*\}"
    )
    previous = None
    while text != previous:
        previous = text
        text = fraction_pattern.sub(r"(\1)/(\2)", text)
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    return text


def _fraction_from_ast(node: ast.AST, source: str) -> Fraction:
    if isinstance(node, ast.Expression):
        return _fraction_from_ast(node.body, source)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        segment = ast.get_source_segment(source, node) or str(node.value)
        try:
            return Fraction(Decimal(segment))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("invalid numeric literal") from exc
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _fraction_from_ast(node.operand, source)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _fraction_from_ast(node.left, source)
        right = _fraction_from_ast(node.right, source)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        if right.denominator != 1 or abs(right.numerator) > 10000:
            raise ValueError("invalid exponent")
        if left.denominator == 0:
            raise ValueError("invalid base")
        return left ** right.numerator
    raise ValueError(f"disallowed arithmetic node: {type(node).__name__}")


def normalize_claim_expression(expression: str, *, strong: bool) -> tuple[str, Optional[int]]:
    """Return the complete cleaned expression and its exact integer value."""
    claimed = _strip_math_wrappers(expression)
    if not claimed:
        return claimed, None
    # An equality chain represents the final asserted RHS, not its first number.
    arithmetic_claim = claimed.rsplit("=", 1)[-1].strip() if "=" in claimed else claimed
    arithmetic_claim = _strip_math_wrappers(arithmetic_claim)
    literal = arithmetic_claim.replace(",", "")
    if re.fullmatch(r"[+-]?\d+", literal):
        return claimed, int(literal)
    if not strong:
        return claimed, None
    arithmetic = _latex_to_arithmetic(arithmetic_claim)
    if not re.fullmatch(r"[\d\s+\-*/().]+", arithmetic):
        return claimed, None
    try:
        tree = ast.parse(arithmetic, mode="eval")
        value = _fraction_from_ast(tree, arithmetic)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return claimed, None
    return claimed, value.numerator if value.denominator == 1 else None


def _balanced_brace(text: str, opening_brace: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(opening_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace + 1 : index], index + 1
    return None


def _scan_arithmetic(text: str, start: int) -> tuple[str, int] | None:
    index = start
    allowed_commands = (r"\frac", r"\dfrac", r"\cdot", r"\times", r"\left", r"\right")
    while index < len(text):
        char = text[index]
        if char in "\r\n":
            break
        if char.isdigit() or char.isspace() or char in "+-*/^().,{}=$*_`":
            index += 1
            continue
        if char == "\\":
            command = next((item for item in allowed_commands if text.startswith(item, index)), None)
            if command:
                index += len(command)
                continue
            if text.startswith(r"\)", index) or text.startswith(r"\]", index):
                index += 2
            break
        break
    raw = text[start:index].strip()
    return (raw, index) if raw and re.search(r"\d", raw) else None


def _extract_expression_at(
    text: str,
    position: int,
    *,
    strong: bool,
    allow_symbolic_lhs: bool = False,
) -> tuple[str, Optional[int], int] | None:
    index = position
    while index < len(text) and text[index].isspace():
        index += 1
    for wrapper in ("**", "__", "`", "$"):
        if text.startswith(wrapper, index):
            index += len(wrapper)
    if text.startswith(r"\(", index) or text.startswith(r"\[", index):
        index += 2
        while index < len(text) and text[index].isspace():
            index += 1
    boxed = re.match(r"\\boxed\s*\{", text[index:])
    if boxed:
        brace = index + boxed.end() - 1
        balanced = _balanced_brace(text, brace)
        if balanced is None:
            return None
        inner, end = balanced
        claimed, value = normalize_claim_expression(inner, strong=True)
        return claimed, value, end

    scanned = _scan_arithmetic(text, index)
    if scanned is None and allow_symbolic_lhs:
        # Support a symbolic LHS followed by an asserted numeric RHS, e.g.
        # ``The answer is \sqrt{N} = 243``.  This is an equality parse, not a
        # first/last-number fallback.
        sentence_end_candidates = [p for p in (text.find("\n", index), text.find(".", index), text.find("!", index), text.find("?", index)) if p >= 0]
        sentence_end = min(sentence_end_candidates) if sentence_end_candidates else min(len(text), index + 240)
        fragment = text[index:sentence_end]
        equals = [m.end() for m in re.finditer(r"(?:=|\bequals\b|\bis\b)\s*", fragment, flags=re.I)]
        if not equals:
            return None
        rhs_start = index + equals[-1]
        while rhs_start < len(text) and text[rhs_start].isspace():
            rhs_start += 1
        scanned = _scan_arithmetic(text, rhs_start)
        if scanned is None:
            return None
        raw, end = scanned
        full = text[index:end].strip()
        claimed, value = normalize_claim_expression(full, strong=strong)
        return (claimed, value, end) if claimed else None

    if scanned is None:
        return None

    raw, end = scanned
    claimed, value = normalize_claim_expression(raw, strong=strong)
    if not claimed:
        return None
    return claimed, value, end


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    left = max(text.rfind("\n", 0, position), text.rfind(".", 0, position), text.rfind("!", 0, position), text.rfind("?", 0, position))
    right_candidates = [p for p in (text.find("\n", position), text.find(".", position), text.find("!", position), text.find("?", position)) if p >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return left + 1, right


def _sentence(text: str, position: int) -> str:
    left, right = _sentence_bounds(text, position)
    return text[left:right].strip()


def _sentence_prefix(text: str, start: int) -> str:
    left, _ = _sentence_bounds(text, start)
    return text[max(left, start - 300) : start]


def _conditional_context(text: str, start: int) -> bool:
    prefix = _sentence_prefix(text, start).lower()
    return bool(re.search(r"\b(?:if|suppose|supposing|assuming|assume|provided|would\s+be|were|earlier\s+i\s+thought)\b", prefix))


def _tentative_context(text: str, start: int, end: int) -> bool:
    sentence = _sentence(text, start).lower()
    prefix = _sentence_prefix(text, start).lower()
    cue = re.compile(
        r"\b(?:maybe|perhaps|possibly|could\s+be|might\s+be|may\s+be|seems?|appears?|"
        r"i\s+think|i\s+suspect|probably|i\s+recall|i\s+remember|i(?:'ve|\s+have)\s+seen)\b"
    )
    if cue.search(sentence) or cue.search(prefix):
        return True
    return "?" in text[start:min(len(text), end + 100)] and bool(re.search(r"\b(?:maybe|could|might|perhaps|possibly)\b", sentence))


def _negated_context(text: str, start: int, end: int) -> bool:
    prefix = _sentence_prefix(text, start).lower()[-100:]
    suffix = text[end:min(len(text), end + 100)].lower()
    if re.search(r"\b(?:not|never|cannot|can't)\b", prefix):
        return True
    return bool(re.match(r"\s*(?:is\s+)?(?:not\s+allowed|invalid|impossible|not\s+the\s+answer|cannot\s+be|can't\s+be)\b", suffix))


def _normalize_overlap(text: str) -> str:
    lowered = text.lower().replace("\\", " ")
    lowered = re.sub(r"[`*_{}$\[\]()]", " ", lowered)
    lowered = re.sub(r"[^a-z0-9+\-=./ ]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _token_overlap(left: str, right: str) -> float:
    a = set(re.findall(r"[a-z0-9]+", left.lower()))
    b = set(re.findall(r"[a-z0-9]+", right.lower()))
    return len(a & b) / max(1, len(a))


def _problem_restatement(text: str, event_start: int, event_end: int, problem_text: str) -> bool:
    if not problem_text:
        return False
    sentence = _sentence(text, event_start)
    sentence_norm = _normalize_overlap(sentence)
    problem_norm = _normalize_overlap(problem_text)
    evidence_norm = _normalize_overlap(text[event_start:event_end])
    if len(sentence_norm) >= 20 and sentence_norm in problem_norm:
        return True
    if len(evidence_norm) >= 12 and evidence_norm in problem_norm:
        return True
    cue = bool(re.search(
        r"(?i)\b(?:the\s+problem\s+(?:asks|states|says)|given\s+that|according\s+to\s+the\s+(?:problem|condition)|"
        r"the\s+condition\s+(?:is|says|states)|we\s+(?:know|are\s+given)\b|we\s+(?:want|need)\b|"
        r"we\s+are\s+looking\s+for|for\s+the\s+final\s+result\s+to\s+be)\b",
        sentence,
    ))
    if cue and _token_overlap(sentence_norm, problem_norm) >= 0.35:
        return True
    # A phrase explicitly named as the problem's final result is especially
    # likely to be a condition restatement, but only when both the prompt and
    # candidate sentence use that phrase and a restatement cue is present.
    return cue and "final result" in sentence_norm and "final result" in problem_norm


RETRACTION_RE = re.compile(
    r"(?i)\b(?:wrong|incorrect|not\s+correct|something\s+is\s+off|does(?:n't|\s+not)\s+work|"
    r"cannot\s+be|can't\s+be|not\s+allowed|invalid|contradiction|contradicts?|impossible|"
    r"reject|discard|this\s+fails|this\s+is\s+false|not\s+the\s+answer|needs?\s+rechecking|"
    r"need\s+to\s+(?:recheck|reconsider)|a\s+mistake)\b"
)


def _retracted_after(text: str, end: int) -> bool:
    # Same-sentence remainder plus at most the next two sentences, capped at
    # 700 characters.  This deliberately avoids unbounded future scanning.
    window = text[end:min(len(text), end + 700)]
    stops = list(re.finditer(r"[.!?\n]", window))
    if len(stops) >= 3:
        window = window[: stops[2].end()]
    return bool(RETRACTION_RE.search(window))


def _terminal_after(text: str, end: int) -> bool:
    tail = text[end:].strip()
    tail = re.sub(r"^[\s$*`\\\]\[().,;:!?-]+", "", tail)
    return not tail


def _near_terminal(text: str, end: int, limit: int = 600) -> bool:
    tail = text[end:].strip()
    return len(tail) <= limit


def _heading_anchor(text: str) -> int:
    pattern = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{0,2})final\s+answer(?:\*{0,2})\s*:?\s*$")
    matches = list(pattern.finditer(text))
    return matches[-1].end() if matches else -1


def _region_for(position: int, final_anchor_end: int) -> str:
    return "FINAL_ANSWER_REGION" if final_anchor_end >= 0 and position >= final_anchor_end else "BODY"


def _conclusion_cue(text: str, start: int, rule: str) -> bool:
    prefix = _sentence_prefix(text, start)
    if rule in {"WE_OBTAIN", "CONCLUSION_TARGET"}:
        return True
    return bool(re.search(
        r"(?i)\b(?:therefore|thus|hence|finally|consequently|so\s+the\s+(?:answer|result)|"
        r"we\s+conclude|this\s+gives\s+the\s+required|the\s+required\s+(?:value|answer|number))\b",
        prefix,
    ))


def _direct_conclusion_cue(text: str, start: int) -> bool:
    prefix = _sentence_prefix(text, start)
    return bool(re.search(r"(?i)\b(?:therefore|thus|hence|finally|consequently)\s*$", prefix))


def _make_event(
    text: str,
    problem_text: str,
    event_type: str,
    start: int,
    end: int,
    expression: str,
    value: Optional[int],
    strength: str,
    rule: str,
    final_anchor_end: int,
) -> ClaimEvent:
    return ClaimEvent(
        event_type=event_type,
        start=start,
        end=end,
        raw_span=text[start:end],
        claimed_expression=expression,
        normalized_value=value,
        strength=strength,
        region=_region_for(start, final_anchor_end),
        rule=rule,
        candidate_sentence=_sentence(text, start),
        conditional=_conditional_context(text, start),
        negated=_negated_context(text, start, end),
        tentative=_tentative_context(text, start, end),
        retracted=_retracted_after(text, end),
        problem_restatement=_problem_restatement(text, start, end, problem_text),
    )


def enumerate_claim_events(normalized_text: str, problem_text: str = "") -> list[ClaimEvent]:
    text = normalized_text
    events: list[ClaimEvent] = []
    final_anchor_end = _heading_anchor(text)

    for match in re.finditer(r"\\boxed\s*\{", text):
        balanced = _balanced_brace(text, match.end() - 1)
        if balanced is None:
            continue
        inner, end = balanced
        expression, value = normalize_claim_expression(inner, strong=True)
        events.append(_make_event(text, problem_text, "BOXED_CLAIM", match.start(), end, expression, value, "STRONG_FINAL", "BALANCED_BOXED", final_anchor_end))

    explicit_patterns = (
        (r"\b(?:the\s+)?(?:correct\s+)?final\s+answer\s*(?:is|equals|=|:)\s*", "FINAL_ANSWER_DIRECT", "STRONG_FINAL"),
        (r"(?<!final\s)\b(?:the\s+)?(?:required\s+)?answer\s*(?:is|equals|=|:)\s*", "ANSWER_DIRECT", "EXPLICIT"),
        # ``final result`` is often a restated condition inside long reasoning.
        # It is explicit, but on truncated output it must also be near-terminal
        # (unlike a literal ``Final Answer`` anchor or a balanced box).
        (r"\b(?:the\s+)?final\s+result\s*(?:is|equals|=|:)\s*", "FINAL_RESULT_DIRECT", "EXPLICIT"),
        (r"\b(?:the\s+)?result\s*(?:is|equals|=|:)\s*", "RESULT_DIRECT", "EXPLICIT"),
    )
    for pattern, rule, strength in explicit_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            extracted = _extract_expression_at(text, match.end(), strong=True, allow_symbolic_lhs=True)
            if extracted is None:
                continue
            expression, value, end = extracted
            events.append(_make_event(text, problem_text, "EXPLICIT_FINAL_ANSWER", match.start(), end, expression, value, strength, rule, final_anchor_end))

    target_patterns = (
        (r"\b(?:the\s+)?(?:required|requested|final)\s+(?:value|number|quantity)\s*(?:is\s+(?:therefore|thus)\s+|is|equals|=|:)\s*", "NAMED_FINAL_TARGET"),
        (r"\b(?:the\s+)?(?:value|sum|remainder)\s+of\s+[^\n.!?]{1,140}?\s+(?:is\s+(?:therefore|thus)\s+|is|equals)\s*", "TARGET_OF_PREDICATE"),
        (r"\b(?:the\s+)?remainder\s+(?:when|after)\s+[^\n.!?]{1,140}?\s+(?:is\s+(?:therefore|thus)\s+|is|equals|=)\s*", "REMAINDER_CONTEXT_PREDICATE"),
        (r"\b(?:the\s+)?(?:value|sum|remainder)\s*(?:is\s+(?:therefore|thus)\s+|is|equals|=|:)\s*", "TARGET_DIRECT"),
        (r"\b(?:the\s+)?integer\s+closest\s+to\s+[^\n.!?]{1,140}?\s+(?:is|equals|=)\s*", "CLOSEST_INTEGER_PREDICATE"),
        (r"\b(?:the\s+)?number\s+of\s+[^\n.!?]{1,180}?\s+(?:is\s+(?:therefore|thus)\s+|is|equals)\s*", "NUMBER_OF_PREDICATE"),
        (r"\b(?:therefore|hence|thus|consequently|finally)\s+(?:the\s+)?(?:answer|result|value|sum|remainder|required\s+value|required\s+number)\s*(?:is|equals|=|:)\s*", "CONCLUSION_TARGET"),
        (r"\b(?:therefore|hence|thus|so)?\s*we\s+(?:finally\s+)?(?:obtain|get|find)\s*", "WE_OBTAIN"),
    )
    for pattern, rule in target_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            extracted = _extract_expression_at(text, match.end(), strong=False)
            if extracted is None:
                continue
            expression, value, end = extracted
            event = _make_event(text, problem_text, "TERMINAL_ASSERTION", match.start(), end, expression, value, "TARGET", rule, final_anchor_end)
            commit_context = event.region == "FINAL_ANSWER_REGION" or _conclusion_cue(text, match.start(), rule) or _terminal_after(text, end)
            event.committed = commit_context
            events.append(event)

    variable_pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:\$|\\\()?([A-Za-z](?:_\{?\d+\}?)?(?:\s*[+\-]\s*[A-Za-z](?:_\{?\d+\}?)?)?)(?:\$|\\\))?\s*(?:is|equals|=)\s*",
        flags=re.I,
    )
    for match in variable_pattern.finditer(text):
        extracted = _extract_expression_at(text, match.end(), strong=True)
        if extracted is None:
            continue
        expression, value, end = extracted
        event = _make_event(text, problem_text, "VARIABLE_ASSERTION", match.start(), end, expression, value, "VARIABLE", "VARIABLE_ASSIGNMENT", final_anchor_end)
        target = re.sub(r"\s+", "", match.group(1).lower())
        explicit_target = target in {"m+n", "p+q"}
        prefix = _sentence_prefix(text, match.start()).lower()
        suffix = text[end:min(len(text), end + 80)]
        continuation = bool(re.search(r"\b(?:continuing|continue|next|now\s+we\s+(?:continue|compute|find))\b", prefix))
        range_member = bool(re.match(r"\s*(?:,|\\dots|\.\.\.)", suffix))
        promoted = event.region == "FINAL_ANSWER_REGION" or _direct_conclusion_cue(text, match.start()) or _terminal_after(text, end)
        if continuation or range_member:
            promoted = False
        event.intermediate_assignment = not (promoted or (explicit_target and _near_terminal(text, end)))
        event.committed = not event.intermediate_assignment
        events.append(event)

    block_match = re.search(r"\\\[\s*([\d\s+\-*/^().,=]+)\s*\\\]\s*$", text)
    if block_match:
        expression, value = normalize_claim_expression(block_match.group(1), strong=True)
        event = _make_event(text, problem_text, "STANDALONE_TERMINAL_ANSWER", block_match.start(), block_match.end(), expression, value, "TERMINAL", "STANDALONE_DISPLAY_INTEGER", final_anchor_end)
        event.committed = True
        events.append(event)
    else:
        lines = list(re.finditer(r"(?m)^([^\r\n]*)$", text))
        nonempty = [item for item in lines if item.group(1).strip()]
        if nonempty:
            last = nonempty[-1]
            expression, value = normalize_claim_expression(last.group(1), strong=False)
            if value is not None and re.fullmatch(r"[+-]?\d+", expression.replace(",", "")):
                event = _make_event(text, problem_text, "STANDALONE_TERMINAL_ANSWER", last.start(), last.end(), expression, value, "TERMINAL", "STANDALONE_LAST_LINE_INTEGER", final_anchor_end)
                event.committed = True
                events.append(event)

    events.sort(key=lambda item: (item.start, item.end, item.event_type, item.rule))
    for event in events:
        if event.event_type in {"BOXED_CLAIM", "EXPLICIT_FINAL_ANSWER", "STANDALONE_TERMINAL_ANSWER"}:
            event.committed = True
        if event.rejection_reasons():
            event.committed = False

    deduplicated: list[ClaimEvent] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        key = (event.start, event.end, event.claimed_expression, event.event_type, event.rule)
        if key not in seen:
            seen.add(key)
            deduplicated.append(event)
    return deduplicated


def extract_v4(raw_text: str, termination_reason: Any = "", problem_text: str = "") -> ExtractionResult:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be str")
    if not isinstance(problem_text, str):
        raise TypeError("problem_text must be str")
    termination = _termination_text(termination_reason)
    normalized_text, actions = _normalize_nonsemantic(raw_text)
    events = enumerate_claim_events(normalized_text, problem_text)

    if _is_truncated(termination):
        for event in events:
            if not event.committed:
                continue
            explicitly_final = (
                event.strength == "STRONG_FINAL"
                or event.region == "FINAL_ANSWER_REGION"
                or (event.strength == "EXPLICIT" and _near_terminal(normalized_text, event.end, 1000))
            )
            if not explicitly_final:
                event.truncated_weak = True
                event.committed = False

    committed = [event for event in events if event.committed and not event.rejection_reasons()]
    rejections = [event.rejection_dict() for event in events if event.rejection_reasons()]
    if not committed:
        reason = "ABSTAIN_TRUNCATED_WITHOUT_EXPLICIT_FINAL" if _is_truncated(termination) else "NO_COMMITTED_FINAL_CLAIM"
        return ExtractionResult(
            status="ABSTAIN",
            claimed_expression=None,
            normalized_value=None,
            aime_domain_valid=None,
            selected_event_type=None,
            selected_rule=None,
            evidence_span=None,
            span_start=None,
            span_end=None,
            termination_reason=termination,
            normalization_actions=actions,
            candidate_count=len(events),
            committed_candidate_count=0,
            candidate_rejections=rejections,
            abstain_reason=reason,
        )

    chosen = max(committed, key=lambda event: (event.start, event.end))
    value = chosen.normalized_value
    domain_valid = isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 999
    return ExtractionResult(
        status="EXTRACTED",
        claimed_expression=chosen.claimed_expression,
        normalized_value=value,
        aime_domain_valid=domain_valid,
        selected_event_type=chosen.event_type,
        selected_rule=chosen.rule,
        evidence_span=raw_text[chosen.start:chosen.end],
        span_start=chosen.start,
        span_end=chosen.end,
        termination_reason=termination,
        normalization_actions=actions,
        candidate_count=len(events),
        committed_candidate_count=len(committed),
        candidate_rejections=rejections,
        abstain_reason=None,
    )


def compare_with_gold(result: ExtractionResult, gold: Any) -> bool:
    """Compare only after extraction; gold never participates in selection."""
    if not result.has_claim or not result.aime_domain_valid:
        return False
    try:
        reference = int(str(gold).strip())
    except (TypeError, ValueError):
        return False
    return result.normalized_value == reference
