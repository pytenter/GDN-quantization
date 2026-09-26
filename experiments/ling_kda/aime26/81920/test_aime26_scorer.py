#!/usr/bin/env python3
"""Formal-release regression tests for the shared AIME answer scorer."""

from aime26_common import extract_aime_answer, score_aime


CASES = [
    ("The answer is \\boxed{42}.", "42", "42", True, "integer boxed answer"),
    ("Therefore \\boxed{-7}.", "-7", "-7", True, "negative integer"),
    ("Thus \\boxed{\\frac{29}{50}}.", "29/50", "29/50", True, "fraction"),
    ("Thus \\boxed{\\frac{2}{4}}.", "1/2", "1/2", True, "simplified fraction"),
    ("Final answer: \\boxed{0.125}", "1/8", "0.125", True, "decimal"),
    ("First \\boxed{12}, finally \\boxed{34}.", "34", "34", True, "last boxed expression"),
    ("Malformed \\boxed{\\frac{1}{}}", "", "1", False, "malformed answer"),
    ("No final answer is supplied.", "", "0", False, "no boxed answer"),
]


def main():
    for text, expected_extracted, gold, expected_correct, label in CASES:
        extracted = extract_aime_answer(text)
        scored_extracted, correct = score_aime(text, gold)
        assert extracted == expected_extracted, (label, extracted, expected_extracted)
        assert scored_extracted == expected_extracted, (label, scored_extracted, expected_extracted)
        assert correct is expected_correct, (label, correct, expected_correct)
    print("SCORER_CORRECTNESS = PASS")


if __name__ == "__main__":
    main()
