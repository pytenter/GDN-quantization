import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_granularity_magnitude_equalization_causal_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_granularity_mag_eq", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_column_magnitude_equalization_matches_same_carrier_row_norm_and_preserves_direction():
    runner = load_runner()
    e_c = {0: torch.tensor([[[[0.0, 6.0, 8.0]]]], dtype=torch.float32)}
    e_r = {0: torch.tensor([[[[3.0, 4.0, 0.0]]]], dtype=torch.float32)}

    out = runner.build_column_magnitude_equalized_residual(e_c, e_r)

    assert out["C_error_norm"] == pytest.approx(10.0)
    assert out["R_ref_error_norm"] == pytest.approx(5.0)
    assert out["alpha"] == pytest.approx(0.5)
    assert runner.stack_norm(out["delta"]) == pytest.approx(5.0)
    assert out["magnitude_relative_mismatch"] <= runner.NORM_REL_TOL
    assert out["direction_cosine"] >= 1.0 - runner.DIRECTION_COS_TOL
    assert out["MAGNITUDE_MATCH_GATE"] == "PASS"
    assert out["DIRECTION_PRESERVATION_GATE"] == "PASS"
    assert out["N_NEAR_ZERO_C_EVENTS"] == 0


def test_alpha_one_identity_returns_original_column_residual():
    runner = load_runner()
    e_c = {0: torch.tensor([[[[1.0, -2.0]]]], dtype=torch.float32)}
    e_r = {0: torch.tensor([[[[-1.0, 2.0]]]], dtype=torch.float32)}

    out = runner.build_column_magnitude_equalized_residual(e_c, e_r)

    assert out["alpha"] == pytest.approx(1.0)
    assert torch.allclose(out["delta"][0], e_c[0], atol=0.0, rtol=0.0)


def test_zero_column_norm_is_deterministic_and_records_near_zero():
    runner = load_runner()
    e_c = {0: torch.zeros(1, 1, 1, 2)}
    e_r = {0: torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float32)}

    out = runner.build_column_magnitude_equalized_residual(e_c, e_r)

    assert runner.stack_norm(out["delta"]) == pytest.approx(0.0)
    assert out["N_NEAR_ZERO_C_EVENTS"] == 1
    assert out["direction_cosine"] is None
    assert out["MAGNITUDE_MATCH_GATE"] == "PASS"
    assert out["DIRECTION_PRESERVATION_GATE"] == "PASS"


def test_causal_cell_formulas_compute_rescue_and_residual_gap():
    runner = load_runner()
    out = runner.causal_cell_metrics(unit_id="u", group_size=128, history_length=64, kl_r=0.10, kl_c=0.22, kl_ceq=0.13)

    assert out["NativeGap"] == pytest.approx(0.12)
    assert out["Rescue"] == pytest.approx(0.09)
    assert out["ResidualGap"] == pytest.approx(0.03)


def test_mediation_contrast_uses_native_minus_equalized_residual_attenuation():
    runner = load_runner()
    unit_rows = [
        {"unit_id": "u0", "group_size": 128, "Rescue_i_g": 0.09, "NativeGap_i_g": 0.12, "ResidualGap_i_g": 0.03},
        {"unit_id": "u0", "group_size": 16, "Rescue_i_g": 0.01, "NativeGap_i_g": 0.02, "ResidualGap_i_g": 0.01},
        {"unit_id": "u1", "group_size": 128, "Rescue_i_g": 0.08, "NativeGap_i_g": 0.10, "ResidualGap_i_g": 0.02},
        {"unit_id": "u1", "group_size": 16, "Rescue_i_g": 0.02, "NativeGap_i_g": 0.03, "ResidualGap_i_g": 0.01},
    ]

    contrasts = runner.compute_granularity_mediation(unit_rows, bootstrap_reps=200, bootstrap_seed=7)

    assert contrasts["NATIVE_ATTENUATION_128_TO_16"] == pytest.approx(0.085)
    assert contrasts["EQUALIZED_RESIDUAL_ATTENUATION_128_TO_16"] == pytest.approx(0.015)
    assert contrasts["CAUSAL_MEDIATION_CONTRAST_M"] == pytest.approx(0.07)
    assert contrasts["positive_unit_count"] == 2
    assert contrasts["negative_unit_count"] == 0
