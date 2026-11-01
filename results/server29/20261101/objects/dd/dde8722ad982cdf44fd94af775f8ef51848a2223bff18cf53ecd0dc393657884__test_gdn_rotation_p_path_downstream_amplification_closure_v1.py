import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_rotation_p_path_downstream_amplification_closure_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gdn_p_downstream_amp_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_values():
    fp = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    return {
        "NN": fp + torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
        "RN": fp + torch.tensor([0.4, 0.0, 0.0], dtype=torch.float64),
        "NR": fp + torch.tensor([0.0, 0.2, 0.0], dtype=torch.float64),
        "RR": fp + torch.tensor([0.5, 0.3, 0.0], dtype=torch.float64),
    }, fp


def test_normalized_risk_formula_and_zero_reference():
    r = load_runner()
    y = torch.tensor([2.0, 0.0])
    fp = torch.tensor([1.0, 0.0])

    assert r.normalized_risk(y, fp)["risk"] == pytest.approx(1.0)
    zero = r.normalized_risk(torch.tensor([1.0]), torch.tensor([0.0]))
    assert zero["risk"] > 1e11


def test_stage_rescue_uses_normalized_risks():
    r = load_runner()
    values, fp = stage_values()
    risks = r.stage_normalized_risks(values, fp)
    effects = r.stage_rescue_effects(values, fp)

    assert effects["Rescue_P"] == pytest.approx(risks["RR"]["risk"] - risks["NR"]["risk"])
    assert effects["Rescue_R"] == pytest.approx(risks["RR"]["risk"] - risks["RN"]["risk"])
    assert effects["D"] == pytest.approx(effects["Rescue_P"] - effects["Rescue_R"])


def test_delta_p_delta_r_and_interaction_tensors():
    r = load_runner()
    values, _fp = stage_values()

    delta = r.stage_delta_tensors(values)

    assert torch.equal(delta["DeltaP"], values["RR"] - values["NR"])
    assert torch.equal(delta["DeltaR"], values["RR"] - values["RN"])
    assert torch.equal(delta["interaction"], values["RR"] - values["RN"] - values["NR"] + values["NN"])
    assert r.interaction_strength(values) >= 0.0


def test_operator_amplification_ratio_is_scale_safe():
    r = load_runner()
    prev = {
        "DeltaP": torch.tensor([2.0, 0.0]),
        "DeltaR": torch.tensor([1.0, 0.0]),
        "fp": torch.tensor([10.0, 0.0]),
    }
    cur = {
        "DeltaP": torch.tensor([8.0, 0.0]),
        "DeltaR": torch.tensor([2.0, 0.0]),
        "fp": torch.tensor([20.0, 0.0]),
    }

    out = r.operator_amplification(prev, cur)

    assert out["g_P"] == pytest.approx(4.0)
    assert out["g_R"] == pytest.approx(2.0)
    assert out["Lambda"] == pytest.approx(2.0)
    assert out["gtilde_P"] == pytest.approx(2.0)
    assert out["gtilde_R"] == pytest.approx(1.0)
    assert out["Lambda_tilde"] == pytest.approx(2.0)


def test_gdn_stage_order_is_verified():
    r = load_runner()

    assert r.GDN_DOWNSTREAM_STAGES == [
        "core_readout",
        "rms_unit",
        "learned_rms_weight",
        "dynamic_gate",
        "merge_heads",
        "out_proj",
        "logits_proxy",
    ]


def test_out_proj_exact_linear_transfer():
    r = load_runner()
    delta = torch.randn(1, 1, 4)
    weight = torch.randn(6, 4)

    actual = r.apply_out_proj(delta, weight)
    expected = torch.matmul(delta, weight.t())

    assert torch.allclose(actual, expected)


def test_first_dominance_stage_requires_previous_not_supported():
    r = load_runner()
    rows = [
        {"stage": "core_readout", "horizon": 64, "D_wins": 9, "D_bootstrap95_mean": [-1.0, 1.0], "D_median": 0.1},
        {"stage": "out_proj", "horizon": 64, "D_wins": 18, "D_bootstrap95_mean": [1.0, 2.0], "D_median": 1.5},
    ]

    assert r.first_p_dominance_stage(rows) == "out_proj"


def test_summary_preserves_canonical_stage_order():
    r = load_runner()
    rows = [
        {"horizon": 64, "stage": "out_proj", "D": 2.0},
        {"horizon": 64, "stage": "core_readout", "D": 1.0},
        {"horizon": 64, "stage": "rms_unit", "D": 3.0},
    ]

    out = r.summarize(rows, [64], group_key="stage", keys=["D"], bootstrap_samples=20)

    assert [row["stage"] for row in out] == ["core_readout", "rms_unit", "out_proj"]


def test_logits_diagnostics_include_kl_cosine_top1_and_rescue():
    r = load_runner()
    fp = torch.tensor([[[10.0, 0.0, -1.0]]])
    values = {
        "NN": fp + torch.tensor([[[0.0, 0.2, 0.0]]]),
        "RN": fp + torch.tensor([[[0.1, 0.0, 0.0]]]),
        "NR": fp + torch.tensor([[[-0.2, 0.0, 0.0]]]),
        "RR": fp + torch.tensor([[[-0.1, 0.0, 0.0]]]),
    }

    out = r.logits_condition_diagnostics(values, fp)

    assert out["KL_NN"] >= 0.0
    assert out["Cosine_RR"] <= 1.0
    assert out["Top1Agreement_RR"] == pytest.approx(1.0)
    assert out["KL_Rescue_P"] == pytest.approx(out["KL_RR"] - out["KL_NR"])


def test_operator_closure_metric_records_all_stage_types():
    r = load_runner()

    rms = r.operator_closure_metric("core_readout", "rms_unit")
    proj = r.operator_closure_metric("merge_heads", "out_proj")

    assert rms["operator_specific_exact"] is False
    assert rms["closure_metric"] == "rmsnorm_empirical_frozen_context_transfer"
    assert proj["operator_specific_exact"] is True
    assert proj["closure_metric"] == "exact_linear_transfer"


def test_rms_jacobian_action_matches_finite_difference():
    r = load_runner()
    x = torch.tensor([[[[1.0, -2.0, 0.5, 3.0]]]], dtype=torch.float64)
    delta = torch.tensor([[[[0.2, -0.1, 0.05, 0.4]]]], dtype=torch.float64)
    eps = 1e-6
    step = 1e-2

    actual = r.rms_jacobian_action(x, delta, eps)
    fd = (r.rms_unit(x + step * delta, eps).double() - r.rms_unit(x - step * delta, eps).double()) / (2 * step)

    assert torch.allclose(actual, fd, atol=1e-5, rtol=1e-5)


def test_radial_tangential_parts_reconstruct_delta_and_are_orthogonal():
    r = load_runner()
    x = torch.tensor([[[[1.0, 2.0, 0.0, 0.0]]]], dtype=torch.float64)
    delta = torch.tensor([[[[3.0, 4.0, 5.0, 0.0]]]], dtype=torch.float64)

    parts = r.radial_tangential_parts(x, delta)
    radial = parts["radial"]
    tangential = parts["tangential"]

    assert torch.allclose(radial + tangential, delta)
    assert torch.sum(radial * tangential).abs().item() < 1e-10
    assert parts["radial_fraction"] > 0.0
    assert parts["tangential_fraction"] > 0.0
