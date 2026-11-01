#!/usr/bin/env python3
"""Regression tests for immutable strict scoring and additive diagnostic extraction."""

from aime26_common import (
    STRICT_SCORER_VERSION,
    extract_aime_answer,
    extract_aime_answer_strict_v1,
    score_aime,
)
from aime26_diagnostic_scorer import (
    extract_aime_answer_diagnostic_v1,
    score_aime_layers,
)


CASES = [
    (r"\boxed{277}", "277", "BOXED", "PARSED"),
    (r"\boxed{-13}", "-13", "BOXED", "PARSED"),
    (r"\boxed{\frac{29}{50}}", "29/50", "BOXED", "PARSED"),
    ("**Final Answer:** **277**", "277", "FINAL_ANSWER_LABEL", "PARSED"),
    ("**Final Answer** 277", "277", "FINAL_ANSWER_LABEL", "PARSED"),
    ("Final Answer: 62", "62", "FINAL_ANSWER_LABEL", "PARSED"),
    ("The answer is 277.", "277", "ANSWER_IS_PHRASE", "PARSED"),
    ("Therefore, the final answer is 277.", "277", "FINAL_ANSWER_LABEL", "PARSED"),
    ("Work omitted.\n277", "277", "FINAL_STANDALONE_EXPRESSION", "PARSED"),
    ("Final Answer: 277\nMore text.\nFinal Answer: 62", "62", "LAST_EXPLICIT_FINAL_ANSWER", "MULTIPLE_CANDIDATES"),
    ("No answer is supplied here.", "", "NO_ANSWER_FOUND", "NO_ANSWER_FOUND"),
    (r"Malformed \boxed{", "", "NO_ANSWER_FOUND", "NO_ANSWER_FOUND"),
    ("Reasoning uses 10, 20, and 30. Final Answer: 277", "277", "FINAL_ANSWER_LABEL", "PARSED"),
    (r"**Final Answer:** **\frac{29}{50}**", "29/50", "FINAL_ANSWER_LABEL", "PARSED"),
    (r"Final Answer: -\frac{3}{7}", "-3/7", "FINAL_ANSWER_LABEL", "PARSED"),
    ("Final Answer: `277`", "277", "FINAL_ANSWER_LABEL", "PARSED"),
    ("Final Answer: $277$", "277", "FINAL_ANSWER_LABEL", "PARSED"),
    (r"Final Answer: \(277\)", "277", "FINAL_ANSWER_LABEL", "PARSED"),
    ("Final Answer: 25%", "1/4", "FINAL_ANSWER_LABEL", "PARSED"),
]


def main():
    passed = 0
    for index, (text, expected, source, status) in enumerate(CASES, 1):
        result = extract_aime_answer_diagnostic_v1(text)
        assert result["diagnostic_extracted_answer"] == expected, (index, result)
        assert result["diagnostic_parse_source"] == source, (index, result)
        assert result["diagnostic_parse_status"] == status, (index, result)
        passed += 1

    conflict = extract_aime_answer_diagnostic_v1(
        "Final Answer: 277\nMore text.\nFinal Answer: 62"
    )
    assert conflict["diagnostic_candidates"] == ["277", "62"]

    fraction = score_aime_layers(r"\boxed{\frac{29}{50}}", "29/50")
    assert fraction["strict_extracted_answer"] == "29/50"
    assert fraction["diagnostic_extracted_answer"] == "29/50"
    assert fraction["strict_correct"] is True
    assert fraction["diagnostic_correct"] is True

    markdown = score_aime_layers("**Final Answer**\n\nThe value of m+n is **277**.", "277")
    assert markdown["strict_correct"] is False
    assert markdown["diagnostic_correct"] is True
    assert markdown["diagnostic_parse_source"] == "FINAL_ANSWER_LABEL"
    assert markdown["strict_vs_diagnostic_status"] == "STRICT_WRONG_DIAGNOSTIC_CORRECT"

    assert STRICT_SCORER_VERSION == "AIME26_STRICT_V1"
    assert extract_aime_answer is extract_aime_answer_strict_v1
    assert score_aime(r"\boxed{42}", "42") == ("42", True)
    assert extract_aime_answer("**Final Answer:** **277**") == "277"
    print(f"UNIT_TESTS = {passed} passed / 0 failed")


if __name__ == "__main__":
    main()
