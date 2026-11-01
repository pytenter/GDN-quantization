import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_core_output_directional_geometry_localization_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_core_geometry", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transfer_identity_stats_pass_for_small_relative_errors():
    runner = load_runner()
    rows = [
        {"transfer_identity_abs_error": 1e-8, "transfer_identity_rel_error": 1e-7},
        {"transfer_identity_abs_error": 2e-8, "transfer_identity_rel_error": 2e-7},
    ]

    out = runner.transfer_identity_summary(rows)

    assert out["CORE_TRANSFER_IDENTITY_GATE"] == "PASS"
    assert out["CORE_TRANSFER_MODEL"] == "EXACT_LINEAR"
    assert out["CORE_TRANSFER_MAX_REL_ERROR"] == pytest.approx(2e-7)


def test_gain_event_row_uses_transfer_norm_over_error_norm():
    runner = load_runner()

    row = runner.gain_event_row(
        unit_id="u0",
        layer_id=22,
        group_size=128,
        condition="C_MATCHED",
        transfer_norm=4.0,
        error_norm=2.0,
        local_damage=0.5,
    )

    assert row["G_core"] == pytest.approx(2.0)
    assert row["local_damage"] == pytest.approx(0.5)


def test_layer_sign_concordance_uses_frozen_prior_labels():
    runner = load_runner()
    gain_rows = [
        {"layer_id": 0, "delta_G_core": 0.2},
        {"layer_id": 22, "delta_G_core": -0.1},
        {"layer_id": 21, "delta_G_core": -0.1},
    ]
    prior = {
        0: "C_SENSITIVE",
        21: "C_SENSITIVE",
        22: "R_SENSITIVE",
    }

    rows, summary = runner.layer_sign_concordance(gain_rows, prior)

    assert summary["N_LAYER_SIGN_CONCORDANT"] == 2
    assert summary["N_LAYER_SIGN_DISCORDANT"] == 1
    assert summary["DISCORDANT_LAYER_IDS"] == [21]
    assert len(rows) == 3


def test_layer22_reverse_control_requires_negative_delta_g():
    runner = load_runner()

    passed = runner.layer22_reverse_control({"median_G_R": 2.0, "median_G_C": 1.5, "delta_G_core": -0.5, "delta_G_core_CI": [-0.7, -0.1]})
    failed = runner.layer22_reverse_control({"median_G_R": 1.0, "median_G_C": 1.5, "delta_G_core": 0.5, "delta_G_core_CI": [0.1, 0.7]})
    weak = runner.layer22_reverse_control({"median_G_R": 1.0, "median_G_C": 1.1, "delta_G_core": -0.1, "delta_G_core_CI": [-0.3, 0.2]})

    assert passed["LAYER22_REVERSE_CONTROL"] == "PASS"
    assert failed["LAYER22_REVERSE_CONTROL"] == "FAIL"
    assert weak["LAYER22_REVERSE_CONTROL"] == "INCONCLUSIVE"


def test_core_closure_classification_requires_concordance_and_layer22():
    runner = load_runner()
    strong = runner.classify_core_directional_geometry(
        gates_ok=True,
        identity_gate="PASS",
        concordance_rate=17 / 18,
        c_aggregate_ci=[0.1, 0.3],
        layer22_control="PASS",
        c_sensitive_loo_stable=True,
        closed_loop_ci=[-0.1, 0.2],
        metric_independence="MATHEMATICALLY_EQUIVALENT",
    )
    partial = runner.classify_core_directional_geometry(
        gates_ok=True,
        identity_gate="PASS",
        concordance_rate=10 / 18,
        c_aggregate_ci=[0.1, 0.3],
        layer22_control="PASS",
        c_sensitive_loo_stable=True,
        closed_loop_ci=[-0.1, 0.2],
        metric_independence="MATHEMATICALLY_EQUIVALENT",
    )

    assert strong["CORE_DIRECTIONAL_GEOMETRY_STATUS"] == "STRONG_CORE_TRANSFER_CLOSURE"
    assert partial["CORE_DIRECTIONAL_GEOMETRY_STATUS"] == "PARTIAL_SUPPORT"


def test_identity_failure_with_sign_concordance_is_partial_not_invalid():
    runner = load_runner()

    out = runner.classify_core_directional_geometry(
        gates_ok=True,
        identity_gate="FAIL",
        concordance_rate=1.0,
        c_aggregate_ci=[0.1, 0.3],
        layer22_control="PASS",
        c_sensitive_loo_stable=True,
        closed_loop_ci=[0.01, 0.1],
        metric_independence="MATHEMATICALLY_EQUIVALENT",
    )

    assert out["CORE_DIRECTIONAL_GEOMETRY_STATUS"] == "PARTIAL_SUPPORT"
    assert out["FINAL_CLASSIFICATION"] == "KDA_CORE_OUTPUT_TRANSFER_CONTRIBUTES_TO_LOCAL_DIRECTIONAL_SENSITIVITY_BUT_IS_NOT_SUFFICIENT"
