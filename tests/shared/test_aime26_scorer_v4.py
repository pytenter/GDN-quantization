#!/usr/bin/env python3
"""Focused regression checks for the frozen AIME26 Strict V4 scorer."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


SCORER_PATH = Path(__file__).parents[2] / "experiments" / "shared" / "scoring" / "aime26_scorer_v4.py"
SPEC = spec_from_file_location("aime26_scorer_v4", SCORER_PATH)
assert SPEC and SPEC.loader
SCORER = module_from_spec(SPEC)
sys.modules[SPEC.name] = SCORER
SPEC.loader.exec_module(SCORER)


def test_rightmost_committed_claim_wins():
    result = SCORER.extract_v4("The final answer is 12. Correction: the final answer is 34.")
    assert result.status == "EXTRACTED"
    assert result.normalized_value == 34
    assert SCORER.compare_with_gold(result, 34)


def test_tentative_claim_is_rejected():
    result = SCORER.extract_v4("I think the answer is 12.")
    assert result.status == "ABSTAIN"
    assert any("TENTATIVE" in item["reason"] for item in result.candidate_rejections)


def test_retracted_claim_is_rejected():
    result = SCORER.extract_v4("The answer is 12. That answer is wrong; I retract it.")
    assert result.status == "ABSTAIN"
    assert any("RETRACTED" in item["reason"] for item in result.candidate_rejections)


def test_problem_restatement_is_not_selected():
    problem = "The problem states that the answer is 12. Determine the true value."
    result = SCORER.extract_v4(problem, problem_text=problem)
    assert result.status == "ABSTAIN"


def test_truncated_weak_claim_abstains():
    result = SCORER.extract_v4("After simplifying, we obtain 12 and continue", termination_reason="max_tokens")
    assert result.status == "ABSTAIN"
    assert result.abstain_reason == "ABSTAIN_TRUNCATED_WITHOUT_EXPLICIT_FINAL"


def test_out_of_domain_claim_is_not_correct():
    result = SCORER.extract_v4(r"The final answer is \boxed{1000}.")
    assert result.status == "EXTRACTED"
    assert result.aime_domain_valid is False
    assert not SCORER.compare_with_gold(result, 1000)
