import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_query_readout_causal_closure_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("query_readout_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_readout_matches_actual_kda_tensor_orientation():
    runner = load_runner()
    q = torch.tensor([[[1.0, 2.0, -1.0], [0.5, -0.5, 1.0]]])
    u = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 10.0

    got = runner.query_readout(u, q)
    expected = torch.sum(u * q.unsqueeze(-1), dim=-2).unsqueeze(1)

    assert list(got.shape) == [1, 1, 2, 4]
    assert torch.allclose(got, expected)


def test_visible_null_decomposition_preserves_readout_and_annihilates_null():
    runner = load_runner()
    torch.manual_seed(11)
    q = torch.randn(1, 3, 5)
    u = torch.randn(1, 3, 5, 7)

    visible, null = runner.query_visible_null_decomposition(u, q)

    assert torch.allclose(visible + null, u, atol=1e-6)
    assert runner.relative_l2(runner.query_readout(visible, q), runner.query_readout(u, q)) < 1e-6
    assert torch.linalg.vector_norm(runner.query_readout(null, q)) < 1e-6


def test_query_equalization_matches_r_visibility_without_changing_null_component():
    runner = load_runner()
    q = torch.tensor([[[1.0, 0.0, 0.0]]])
    u_r = torch.tensor([[[[2.0, 0.0], [0.0, 9.0], [0.0, -4.0]]]])
    u_c = torch.tensor([[[[8.0, 0.0], [0.0, 9.0], [0.0, -4.0]]]])

    result = runner.query_equalize_visible_component(u_r, u_c, q)

    assert abs(result["gamma"] - 0.25) < 1e-7
    assert result["visibility_match_rel_error"] < 1e-7
    assert result["null_preservation_rel_error"] < 1e-7
    assert torch.allclose(result["u_eq"][..., 1:, :], u_c[..., 1:, :])


def test_pairwise_symmetric_control_reduces_more_visible_direction():
    runner = load_runner()
    q = torch.tensor([[[1.0, 0.0]]])
    u_r = torch.tensor([[[[6.0], [3.0]]]])
    u_c = torch.tensor([[[[2.0], [3.0]]]])

    result = runner.pairwise_readout_match(u_r, u_c, q)

    assert result["reduced_direction"] == "R_TO_C"
    assert result["R_symmetric_damage"] == result["C_native_damage"]
    assert result["C_symmetric_damage"] == result["C_native_damage"]


def test_layer_mapping_keeps_layer22_as_reverse_control_and_pilot_is_preselected():
    runner = load_runner()
    prior = runner.prior_layer_labels()
    rows = [
        {"layer_id": 0, "median_local_gap": 0.10},
        {"layer_id": 1, "median_local_gap": 0.04},
        {"layer_id": 2, "median_local_gap": 0.01},
        {"layer_id": 22, "median_local_gap": -0.05},
    ]

    assert prior[22] == "R_SENSITIVE"
    assert all(label == "C_SENSITIVE" for layer, label in prior.items() if layer != 22)
    assert runner.layer22_symmetric_control(native_gap=-0.2, symmetric_gap=-0.05) == "PASS"
    assert 22 in runner.select_pilot_layers(rows)


def test_upstream_matched_magnitude_gate_uses_layer_local_norms():
    runner = load_runner()

    assert runner.upstream_matched_magnitude_gate(3.0, 3.0 + 1e-8) == "PASS"
    assert runner.upstream_matched_magnitude_gate(3.0, 3.2) == "FAIL"


def test_clustered_bootstrap_and_status_classification_are_positive_only_when_ci_clears_zero():
    runner = load_runner()
    rows = [
        {"unit_id": f"u{i}", "layer_id": 0, "group_size": 128, "rescue": 0.10 + i * 0.001}
        for i in range(36)
    ]

    summary = runner.summarize_rescue(rows, reps=1000, seed=5)

    assert summary["median_rescue"] > 0
    assert summary["ci_low"] > 0
    assert runner.classify_query_readout_status(summary, broad_layer_sign_consistency=True, layer22_control="PASS") == "STRONG_SUPPORT"
    assert runner.classify_query_readout_status({**summary, "ci_low": -0.01}, True, "PASS") == "PARTIAL_SUPPORT"
