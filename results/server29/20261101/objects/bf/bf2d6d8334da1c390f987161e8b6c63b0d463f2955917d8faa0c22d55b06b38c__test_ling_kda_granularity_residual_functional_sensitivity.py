import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_granularity_residual_functional_sensitivity_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_residual_functional", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_column_residual_uses_row_norm_and_preserves_column_direction():
    runner = load_runner()
    e_r = {0: torch.tensor([[[[3.0, 4.0, 0.0]]]], dtype=torch.float32)}
    e_c = {0: torch.tensor([[[[0.0, 6.0, 8.0]]]], dtype=torch.float32)}

    out = runner.build_matched_column_residual(e_r, e_c)

    assert out["R_error_norm"] == pytest.approx(5.0)
    assert out["C_error_norm"] == pytest.approx(10.0)
    assert out["alpha"] == pytest.approx(0.5)
    assert runner.stack_norm(out["E_C_match"]) == pytest.approx(5.0)
    assert out["magnitude_relative_mismatch"] <= runner.NORM_REL_TOL
    assert out["direction_cosine_Cmatch_C"] >= 1.0 - runner.DIRECTION_COS_TOL
    assert out["MAGNITUDE_MATCH_GATE"] == "PASS"
    assert out["DIRECTION_PRESERVATION_GATE"] == "PASS"


def test_direction_similarity_is_between_same_carrier_row_and_matched_column_errors():
    runner = load_runner()
    e_r = {0: torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)}
    e_c = {0: torch.tensor([[[[0.0, 2.0]]]], dtype=torch.float32)}

    out = runner.build_matched_column_residual(e_r, e_c)

    assert out["R_C_direction_cosine"] == pytest.approx(0.0)


def test_directional_gap_formulas_use_column_minus_row_damage():
    runner = load_runner()

    local = runner.directional_gap_row(
        unit_id="u0",
        group_size=128,
        metric_name="post_out_proj_rel_l2",
        damage_r=0.03,
        damage_c_match=0.05,
    )
    prop = runner.propagation_gap_row(
        unit_id="u0",
        group_size=128,
        history_length=64,
        damage_r=0.04,
        damage_c_match=0.01,
        initial_norm=2.0,
    )

    assert local["local_directional_gap"] == pytest.approx(0.02)
    assert prop["propagation_directional_gap"] == pytest.approx(-0.03)
    assert prop["functional_gain_R"] == pytest.approx(0.02)
    assert prop["functional_gain_C_match"] == pytest.approx(0.005)


def test_unit_summary_joins_stage_a_stage_b_and_stage_c_residual():
    runner = load_runner()
    stage_a = [
        {"unit_id": "u0", "prompt_id": "p0", "sample_id": "s0", "layer": "1", "head": "ALL", "t0": "10", "group_size": 128, "local_directional_gap": 0.02, "R_C_direction_cosine": 0.1},
    ]
    stage_b = [
        {"unit_id": "u0", "group_size": 128, "history_length": 1, "propagation_directional_gap": 0.01},
        {"unit_id": "u0", "group_size": 128, "history_length": 64, "propagation_directional_gap": 0.04},
    ]
    stage_c = [
        {"unit_id": "u0", "group_size": 128, "closed_loop_residual_gap": 0.03},
    ]

    rows = runner.compute_unit_level_residual_summary(stage_a, stage_b, stage_c)

    assert rows == [{
        "unit_id": "u0",
        "prompt_id": "p0",
        "sample_id": "s0",
        "layer": "1",
        "head": "ALL",
        "t0": "10",
        "group_size": 128,
        "local_directional_gap": 0.02,
        "propagation_gap_L1": 0.01,
        "propagation_gap_L4": None,
        "propagation_gap_L8": None,
        "propagation_gap_L16": None,
        "propagation_gap_L32": None,
        "propagation_gap_L64": 0.04,
        "closed_loop_residual_gap": 0.03,
        "R_C_error_direction_cosine": 0.1,
    }]


def test_classification_requires_stable_local_signal_for_local_supported():
    runner = load_runner()
    summary = {
        "blocking_gates_ok": True,
        "LOCAL_TO_CLOSED_LOOP_SPEARMAN": 0.4,
        "LOCAL_TO_CLOSED_LOOP_SPEARMAN_CI": [0.1, 0.6],
    }
    group_rows = [
        {"group_size": 128, "local_gap_ci": [0.01, 0.03], "propagation_gap_L64_ci": [-0.01, 0.01], "closed_loop_residual_gap": 0.02},
        {"group_size": 64, "local_gap_ci": [0.02, 0.04], "propagation_gap_L64_ci": [-0.01, 0.01], "closed_loop_residual_gap": 0.01},
    ]
    robustness = {
        "RESIDUAL_LOO_UNIT_SIGN_STABLE": "PASS",
        "RESIDUAL_LOO_LAYER_SIGN_STABLE": "PASS",
        "RESIDUAL_LOO_PROMPT_SIGN_STABLE": "PASS",
    }

    out = runner.classify_residual_functional_sensitivity(summary, group_rows, robustness)

    assert out["RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS"] == "LOCAL_SUPPORTED"
    assert out["NEXT_RECOMMENDED_TASK"] == "LING_KDA_LOCAL_FUNCTIONAL_GEOMETRY_LOCALIZATION_V1"


def test_recurrent_label_requires_added_separation_not_just_small_significant_gap():
    runner = load_runner()
    summary = {
        "blocking_gates_ok": True,
        "LOCAL_TO_CLOSED_LOOP_SPEARMAN": 0.4,
        "LOCAL_TO_CLOSED_LOOP_SPEARMAN_CI": [0.1, 0.6],
        "PROPAGATION_TO_CLOSED_LOOP_SPEARMAN_CI": [-0.1, 0.3],
    }
    group_rows = [
        {"group_size": 128, "local_directional_gap": 0.003, "local_gap_ci": [0.002, 0.004], "propagated_directional_gap_L64": 0.0002, "propagation_gap_L64_ci": [0.0001, 0.0003], "local_to_recurrent_amplification_L64": -0.0028, "amplification_L64_ci": [-0.003, -0.002], "closed_loop_residual_gap": 0.02},
        {"group_size": 64, "local_directional_gap": 0.003, "local_gap_ci": [0.002, 0.004], "propagated_directional_gap_L64": 0.0002, "propagation_gap_L64_ci": [0.0001, 0.0003], "local_to_recurrent_amplification_L64": -0.0028, "amplification_L64_ci": [-0.003, -0.002], "closed_loop_residual_gap": 0.01},
    ]
    robustness = {
        "RESIDUAL_LOO_UNIT_SIGN_STABLE": "PASS",
        "RESIDUAL_LOO_LAYER_SIGN_STABLE": "PASS",
        "RESIDUAL_LOO_PROMPT_SIGN_STABLE": "PASS",
    }

    out = runner.classify_residual_functional_sensitivity(summary, group_rows, robustness)

    assert out["RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS"] == "LOCAL_SUPPORTED"


def test_layer_robustness_failure_downgrades_global_interpretation():
    runner = load_runner()
    summary = {
        "blocking_gates_ok": True,
        "LOCAL_TO_CLOSED_LOOP_SPEARMAN": 0.4,
        "LOCAL_TO_CLOSED_LOOP_SPEARMAN_CI": [0.1, 0.6],
    }
    group_rows = [
        {"group_size": 128, "local_gap_ci": [0.01, 0.03], "propagation_gap_L64_ci": [-0.01, 0.01], "closed_loop_residual_gap": 0.02},
        {"group_size": 64, "local_gap_ci": [0.02, 0.04], "propagation_gap_L64_ci": [-0.01, 0.01], "closed_loop_residual_gap": 0.01},
    ]
    robustness = {
        "RESIDUAL_LOO_UNIT_SIGN_STABLE": "PASS",
        "RESIDUAL_LOO_LAYER_SIGN_STABLE": "FAIL",
        "RESIDUAL_LOO_PROMPT_SIGN_STABLE": "PASS",
    }

    out = runner.classify_residual_functional_sensitivity(summary, group_rows, robustness)

    assert out["RESIDUAL_FUNCTIONAL_SENSITIVITY_STATUS"] == "MIXED_OR_INCONCLUSIVE"
