import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_kda_o_norm_gated_rms_causal_closure_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("kda_o_norm_gated_rms_causal_closure_v1", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_atomic_metrics_close_stage_transfer_identity():
    r = load_runner()
    prev = r.atomic_stage_metric(torch.tensor([10.0, 0.0]), torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]))
    cur = r.atomic_stage_metric(torch.tensor([6.0, 8.0]), torch.tensor([4.0, 0.0]), torch.tensor([3.0, 0.0]))
    transfer = r.atomic_transfer_metric(prev, cur)

    assert prev["r_j"] == pytest.approx(0.5)
    assert prev["G_j"] == pytest.approx(0.75)
    assert cur["r_j"] == pytest.approx(0.75)
    assert cur["G_j"] == pytest.approx(0.4375)
    assert transfer["Lambda_j"] == pytest.approx(1.5)
    assert transfer["identity_abs_error"] <= 1e-12
    assert transfer["identity_rel_error"] <= 1e-12


def test_aggregation_reconciliation_keeps_atomic_and_median_units_separate():
    r = load_runner()
    prev = [
        {"unit_id": "a", "horizon": 64, "stage": "core", "r_j": 0.25, "G_j": 1.0 - 0.25**2},
        {"unit_id": "b", "horizon": 64, "stage": "core", "r_j": 0.80, "G_j": 1.0 - 0.80**2},
        {"unit_id": "c", "horizon": 64, "stage": "core", "r_j": 0.90, "G_j": 1.0 - 0.90**2},
    ]
    cur = [
        {"unit_id": "a", "horizon": 64, "stage": "rms", "r_j": 0.50, "G_j": 1.0 - 0.50**2},
        {"unit_id": "b", "horizon": 64, "stage": "rms", "r_j": 0.82, "G_j": 1.0 - 0.82**2},
        {"unit_id": "c", "horizon": 64, "stage": "rms", "r_j": 1.80, "G_j": 1.0 - 1.80**2},
    ]

    out = r.aggregation_reconciliation(prev, cur)

    assert out["median_atomic_G_prev"] == pytest.approx(1.0 - 0.80**2)
    assert out["G_prev_from_median_r"] == pytest.approx(1.0 - 0.80**2)
    assert out["median_atomic_Lambda"] == pytest.approx(2.0)
    assert out["Lambda_from_median_r"] == pytest.approx(0.82 / 0.80)
    assert out["median_atomic_Lambda"] != pytest.approx(out["Lambda_from_median_r"])


def test_internal_rms_gamma_sigmoid_gate_stage_order_matches_fused_semantics():
    r = load_runner()
    x = torch.tensor([[[[3.0, 4.0]]]])
    gate_logits = torch.tensor([[[[0.0, math.log(3.0)]]]])
    weight = torch.tensor([2.0, 4.0])

    stages = r.compute_o_norm_internal_stages(x, gate_logits, weight, activation="sigmoid", eps=0.0)

    rms = math.sqrt((3.0**2 + 4.0**2) / 2.0)
    pure = x / rms
    learned = pure * weight.view(1, 1, 1, 2)
    gated = learned * torch.tensor([[[[0.5, 0.75]]]])
    assert list(stages) == ["input_core_readout", "pure_rms", "learned_scale", "effective_gate", "module_output"]
    assert torch.allclose(stages["pure_rms"], pure)
    assert torch.allclose(stages["learned_scale"], learned)
    assert torch.allclose(stages["effective_gate"], torch.tensor([[[[0.5, 0.75]]]]))
    assert torch.allclose(stages["module_output"], gated)


def test_radial_tangential_decomposition_is_orthogonal_and_conservative():
    r = load_runner()
    x = torch.tensor([[3.0, 4.0]])
    delta = torch.tensor([[3.0, 0.0]])

    dec = r.radial_tangential_decomposition(x, delta)

    assert torch.allclose(dec["parallel"] + dec["tangential"], delta, atol=1e-12)
    assert float((dec["parallel"] * dec["tangential"]).sum().abs()) <= 1e-12
    assert dec["radial_fraction"] + dec["tangential_fraction"] == pytest.approx(1.0)


def test_rms_jacobian_matches_small_finite_perturbation_and_frozen_denominator_is_scalar():
    r = load_runner()
    x = torch.tensor([[1.5, -0.5, 2.0, 1.0]], dtype=torch.float64)
    d_native = torch.tensor([[1e-5, -2e-5, 1e-5, 0.0]], dtype=torch.float64)
    d_rotated = torch.tensor([[-1e-5, 1e-5, 2e-5, -1e-5]], dtype=torch.float64)

    audit = r.rms_jacobian_audit(x, d_native, d_rotated, eps=1e-12)
    frozen = r.frozen_denominator_rms_delta(x, d_native, d_rotated, eps=1e-12)

    assert audit["native_linearization_relative_error"] < 1e-5
    assert audit["rotated_linearization_relative_error"] < 1e-5
    assert frozen["Lambda_frozen"] == pytest.approx(1.0, abs=1e-10)


def test_isotropic_scale_and_gate_preserve_rms_and_remove_channel_selectivity():
    r = load_runner()
    weight = torch.tensor([1.0, 2.0, 3.0])
    iso = r.isotropic_multiplier(weight)

    assert iso.shape == weight.shape
    assert torch.allclose(iso, torch.full_like(weight, math.sqrt((1.0 + 4.0 + 9.0) / 3.0)))

    d_native = torch.tensor([[1.0, 0.0, 0.0]])
    d_rotated = torch.tensor([[0.0, 1.0, 0.0]])
    transfer = r.diagonal_transfer(d_native, d_rotated, iso)
    assert transfer["Lambda"] == pytest.approx(1.0)


def test_zero_perturbation_and_matched_context_helpers():
    r = load_runner()
    fp = {"input_core_readout": torch.randn(1, 1, 2, 4), "module_output": torch.randn(1, 1, 2, 4)}
    zero = {k: v.clone() for k, v in fp.items()}

    assert r.zero_perturbation_noninterference(fp, zero)["gate"] == "PASS"
    native = {"unit_id": "u", "horizon": 64, "token_id": 7, "layer": 2, "fp_core_sha256": "a", "gate_sha256": "g", "branch": "native", "delta_o_sha256": "n"}
    rotated = dict(native, branch="rotated", delta_o_sha256="r")
    assert r.check_matched_context(native, rotated)["gate"] == "PASS"

    drifted = dict(rotated, gate_sha256="other")
    assert r.check_matched_context(native, drifted)["gate"] == "FAIL"
