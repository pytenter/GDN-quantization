import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_g64_prospective_confirmation_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_kda_g64_prospective", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_rows_and_tokens():
    rows_by_pid = {
        str(i): {"problem_id": str(i), "problem_index": i, "problem": f"Problem {i}"}
        for i in range(60, 90)
    }
    teacher_tokens = {str(i): list(range(i * 1000, i * 1000 + 512)) for i in range(60, 90)}
    return rows_by_pid, teacher_tokens


def test_manifest_freezes_72_units_from_prior_disjoint_prompts_and_four_t0s():
    runner = load_runner()
    rows_by_pid, teacher_tokens = make_rows_and_tokens()
    prior = [
        {"unit_id": f"{pid}|64", "problem_id": str(pid), "problem_index": pid, "t0": 64}
        for pid in [60, 61, 64, 67, 68, 69, 72, 75, 76, 78, 82, 89]
    ]

    manifest = runner.construct_prospective_manifest(prior, rows_by_pid, teacher_tokens)

    assert runner.CONFIGS == ["R64", "C64"]
    assert runner.HISTORY_LENGTHS == [1, 4, 8, 16, 32, 64]
    assert runner.T0_VALUES == [64, 128, 192, 256]
    assert manifest["N_NEW_INDEPENDENT_UNITS"] == 72
    assert len(manifest["units"]) == 72
    assert {u["t0"] for u in manifest["units"]} == {64, 128, 192, 256}
    assert {u["problem_id"] for u in manifest["units"]} == {
        "62", "63", "65", "66", "70", "71", "73", "74", "77",
        "79", "80", "81", "83", "84", "85", "86", "87", "88",
    }
    assert len({u["unit_id"] for u in manifest["units"]}) == 72
    assert all(u["selection_outcome_blind"] is True for u in manifest["units"])


def test_independence_audit_checks_unit_sample_prompt_and_token_overlap():
    runner = load_runner()
    prior_panels = {
        "v2_exact": [
            {
                "unit_id": "60|64",
                "problem_id": "60",
                "sample_id": "60",
                "t0": 64,
                "teacher_forced_token_ids": [1, 2, 3],
            }
        ],
        "v2_independent": [
            {
                "unit_id": "67|64",
                "problem_id": "67",
                "sample_id": "67",
                "t0": 64,
                "teacher_forced_token_ids": [4, 5, 6],
            }
        ],
    }
    new_units = [
        {
            "unit_id": "62|64",
            "problem_id": "62",
            "sample_id": "62",
            "t0": 64,
            "teacher_forced_token_ids": [7, 8, 9],
        }
    ]

    audit = runner.prospective_independence_audit(prior_panels, new_units)

    assert audit["PROSPECTIVE_INDEPENDENCE_GATE"] == "PASS"
    assert audit["TOTAL_UNIT_KEY_OVERLAP_COUNT"] == 0
    assert audit["TOTAL_SAMPLE_ID_OVERLAP_COUNT"] == 0
    assert audit["TOTAL_TOKEN_SEQUENCE_OVERLAP_COUNT"] == 0
    assert audit["TOTAL_PROMPT_OVERLAP_COUNT"] == 0

    bad = runner.prospective_independence_audit(prior_panels, [dict(new_units[0], problem_id="60")])
    assert bad["PROSPECTIVE_INDEPENDENCE_GATE"] == "FAIL"
    assert bad["TOTAL_PROMPT_OVERLAP_COUNT"] == 1


def test_analysis_uses_unit_level_across_history_bootstrap_not_row_bootstrap():
    runner = load_runner()
    rows = []
    for unit_id, deltas in {
        "u0": [0.10, 0.12, 0.11, 0.09, 0.10, 0.08],
        "u1": [0.04, 0.05, 0.03, 0.04, 0.05, 0.04],
        "u2": [-0.02, 0.01, 0.02, 0.03, 0.01, 0.02],
    }.items():
        for history, delta in zip(runner.HISTORY_LENGTHS, deltas):
            rows.append({"unit_id": unit_id, "problem_id": unit_id, "layer": "ALL_KDA", "head": "ALL", "history_length": history, "config": "R64", "future_kl_auc": 1.0})
            rows.append({"unit_id": unit_id, "problem_id": unit_id, "layer": "ALL_KDA", "head": "ALL", "history_length": history, "config": "C64", "future_kl_auc": 1.0 + delta})

    analysis = runner.analyze_prospective_rows(rows, bootstrap_reps=200, bootstrap_seed=123)

    assert len(analysis["per_unit_per_history_effects"]) == 18
    assert analysis["primary_effect_summary"]["N_UNITS"] == 3
    assert analysis["primary_effect_summary"]["PRIMARY_EFFECT_DEFINITION"] == "median over units of mean_L(KL_C64 - KL_R64)"
    assert analysis["primary_effect_summary"]["G64_PRIMARY_EFFECT"] == pytest.approx(0.041666666666666664)
    assert analysis["bootstrap_results"]["BOOTSTRAP_RESAMPLING_LEVEL"] == "canonical_unit"
    assert len(analysis["history_level_summary"]) == 6


def test_decision_classifier_implements_precommitted_cases_and_invalid_gates():
    runner = load_runner()
    gates = runner.pass_gate_summary()
    robust = {
        "primary_effect_summary": {"G64_PRIMARY_EFFECT": 0.05, "G64_BOOTSTRAP_CI": [0.01, 0.09]},
        "history_level_summary": [{"median_delta": 0.01} for _ in range(5)] + [{"median_delta": -0.001}],
        "robustness_summary": {
            "LEAVE_ONE_UNIT_OUT_SIGN_STABLE": "PASS",
            "LEAVE_ONE_LAYER_OUT_SIGN_STABLE": "PASS",
            "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE": "PASS",
            "NO_SINGLE_SUBGROUP_DRIVES_RESULT": "PASS",
        },
    }
    assert runner.classify_prospective_result(robust, gates)["G64_PROSPECTIVE_CLASSIFICATION"] == "ROBUST_R_PREFERENCE"

    boundary = dict(robust)
    boundary["primary_effect_summary"] = {"G64_PRIMARY_EFFECT": 0.05, "G64_BOOTSTRAP_CI": [-0.01, 0.09]}
    assert runner.classify_prospective_result(boundary, gates)["G64_PROSPECTIVE_CLASSIFICATION"] == "ORIENTATION_BOUNDARY"

    column = {
        "primary_effect_summary": {"G64_PRIMARY_EFFECT": -0.05, "G64_BOOTSTRAP_CI": [-0.09, -0.01]},
        "history_level_summary": [{"median_delta": -0.01} for _ in range(4)] + [{"median_delta": 0.001} for _ in range(2)],
        "robustness_summary": {
            "LEAVE_ONE_UNIT_OUT_SIGN_STABLE": "PASS",
            "LEAVE_ONE_LAYER_OUT_SIGN_STABLE": "PASS",
            "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE": "PASS",
            "NO_SINGLE_SUBGROUP_DRIVES_RESULT": "PASS",
        },
    }
    assert runner.classify_prospective_result(column, gates)["G64_PROSPECTIVE_CLASSIFICATION"] == "ROBUST_C_PREFERENCE"

    invalid = runner.pass_gate_summary()
    invalid["PROVENANCE_GATE"] = "FAIL"
    assert runner.classify_prospective_result(robust, invalid)["G64_PROSPECTIVE_CLASSIFICATION"] == "INVALID"


def test_final_console_summary_contains_required_task_fields():
    runner = load_runner()
    summary = {key: "PASS" for key in runner.REQUIRED_CONSOLE_KEYS}
    summary.update({"TASK": runner.TASK, "METHOD_DESIGN_READY": "NO"})

    text = runner.render_console_summary(summary)

    for key in runner.REQUIRED_CONSOLE_KEYS:
        assert f"{key} =" in text
    assert "DO_NOT_RUN_G128_G32_G16" in text
    assert text.rstrip().endswith("STOP.")
