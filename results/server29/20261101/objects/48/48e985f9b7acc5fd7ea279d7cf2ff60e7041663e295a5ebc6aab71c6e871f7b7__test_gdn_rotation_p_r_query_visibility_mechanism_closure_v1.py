import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_rotation_p_r_query_visibility_mechanism_closure_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gdn_pr_query_visibility_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_state():
    state = torch.tensor(
        [[[[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]]]],
        dtype=torch.float64,
    )
    key = torch.tensor([[[0.2, -0.4, 0.7]]], dtype=torch.float64)
    query = torch.tensor([[[0.3, 0.1, -0.6]]], dtype=torch.float64)
    gamma = torch.tensor([[0.82]], dtype=torch.float64)
    beta = torch.tensor([[0.35]], dtype=torch.float64)
    return state, key, query, gamma, beta


def test_gdn_transition_apply_matches_explicit_a_matrix():
    r = load_runner()
    error, key, _query, gamma, beta = sample_state()
    actual = r.apply_gdn_error_transition(error, key, gamma, beta)

    eye = torch.eye(error.shape[-2], dtype=torch.float64)
    a = gamma.item() * (eye - beta.item() * torch.outer(key.reshape(-1), key.reshape(-1)))
    expected = torch.einsum("ij,bhjv->bhiv", a, error)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_effective_query_identity_matches_propagated_readout():
    r = load_runner()
    error, key, query, gamma, beta = sample_state()
    propagated = r.apply_gdn_error_transition(error, key, gamma, beta)
    q_eff = r.gdn_effective_query(query, key, gamma, beta)

    lhs = r.readout_vec(propagated, query)
    rhs = r.readout_vec(error, q_eff)

    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_effective_query_factorization_closes_for_paired_branches():
    r = load_runner()
    error, key, query, gamma, beta = sample_state()
    native = r.effective_query_metrics(error, query, key, gamma, beta)
    rotated = r.effective_query_metrics(error * 0.75 + 0.1, query, key, gamma, beta)

    paired = r.p_effective_query_factorization(native, rotated)

    assert paired["relative_error"] <= 1e-12
    assert paired["p_readout_ratio"] == pytest.approx(
        paired["prev_error_norm_ratio"] * paired["effective_visibility_ratio"],
        rel=1e-12,
    )


def test_same_q_eff_required_for_paired_comparison():
    r = load_runner()
    _error, key, query, gamma, beta = sample_state()
    q_eff = r.gdn_effective_query(query, key, gamma, beta)
    changed = q_eff.clone()
    changed[..., 0] += 1e-3

    assert r.assert_same_tensor(q_eff, q_eff.clone(), "q_eff")["gate"] == "PASS"
    assert r.assert_same_tensor(q_eff, changed, "q_eff")["gate"] == "FAIL"


def test_rotated_quantizer_mapped_back_matches_manual_definition():
    r = load_runner()
    state = torch.randn(1, 2, 8, 5, generator=torch.Generator().manual_seed(7))
    rotation = r.exact().make_experiment_rotation("gdn", 8, kind="rht", seed=0, dtype=torch.float32)

    actual = r.rotated_quantizer_mapped_back(state, rotation)
    rotated = r.exact().transform_state_to_branch(state, "gdn", "rotated", rotation)
    q_rot = r.exact().quantize_branch_state(rotated, "gdn", "rotated")
    expected = r.exact().inverse_branch_state(q_rot, "gdn", "rotated", rotation)

    assert torch.equal(actual, expected)


def test_r_factorial_reproduces_nn_and_rr_residuals():
    r = load_runner()
    gen = torch.Generator().manual_seed(11)
    pre_native = torch.randn(1, 2, 8, 5, generator=gen)
    pre_rotated = torch.randn(1, 2, 8, 5, generator=gen)
    rotation = r.exact().make_experiment_rotation("gdn", 8, kind="rht", seed=0, dtype=torch.float32)

    out = r.r_factorial_residuals(pre_native, pre_rotated, rotation)

    assert torch.equal(out["NN"], r.native_quantizer(pre_native) - pre_native)
    assert torch.equal(out["RR"], r.rotated_quantizer_mapped_back(pre_rotated, rotation) - pre_rotated)


def test_r_factorial_effect_bookkeeping():
    r = load_runner()
    risks = {"NN": 1.0, "NR": 1.4, "RN": 1.1, "RR": 2.0}
    effects = r.factorial_effects(risks)

    assert effects["B_N"] == pytest.approx(0.4)
    assert effects["B_R"] == pytest.approx(0.9)
    assert effects["T_N"] == pytest.approx(0.1)
    assert effects["T_R"] == pytest.approx(0.6)
    assert effects["interaction"] == pytest.approx(0.5)


def test_readout_energy_identity_and_counterfactual_bookkeeping():
    r = load_runner()
    p_native = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    r_native = torch.tensor([0.2, 0.4, -0.1], dtype=torch.float64)
    p_rotated = torch.tensor([1.2, -2.3, 0.7], dtype=torch.float64)
    r_rotated = torch.tensor([0.4, 0.1, -0.3], dtype=torch.float64)

    ledger = r.readout_energy_terms(p_native, r_native)
    assert ledger["identity_relative_error"] <= 1e-12

    cf = r.readout_counterfactual_attribution(p_native, p_rotated, r_native, r_rotated)
    expected_delta = cf["L_RR"] - cf["L_NN"]

    assert cf["total_delta"] == pytest.approx(expected_delta)
    assert cf["sum_contributions"] == pytest.approx(expected_delta)
    assert cf["closure_relative_error"] <= 1e-12


def test_zero_error_and_zero_residual_edge_cases_are_finite():
    r = load_runner()
    z = torch.zeros(1, 1, 4, 3, dtype=torch.float64)
    query = torch.randn(1, 1, 4, dtype=torch.float64)
    key = torch.randn(1, 1, 4, dtype=torch.float64)
    gamma = torch.ones(1, 1, dtype=torch.float64)
    beta = torch.ones(1, 1, dtype=torch.float64) * 0.5

    metrics = r.effective_query_metrics(z, query, key, gamma, beta)
    assert metrics["prev_error_norm"] == 0.0
    assert metrics["effective_visibility"] == 0.0
    assert metrics["p_readout_norm"] == 0.0

    ledger = r.readout_energy_terms(torch.zeros(3), torch.zeros(3))
    assert ledger["total_energy"] == 0.0
    assert ledger["identity_relative_error"] == 0.0


def test_short_horizon_tape_hashes_are_not_formal_comparable():
    r = load_runner()

    assert r.tape_hash_comparison_label(64) == "PASS"
    assert r.tape_hash_comparison_label(4) == "NOT_COMPARABLE_SHORT_HORIZON"


def test_symmetric_relerr_uses_absolute_floor_near_zero():
    r = load_runner()
    a = torch.zeros(4)
    b = torch.ones(4) * 1e-10

    assert r.symmetric_relerr(a, b, abs_floor=1e-8) < 0.03
    assert r.symmetric_relerr(a, b, abs_floor=1e-8) < r.relerr(a, b)


def test_identity_gate_error_accepts_tiny_absolute_zero_case():
    r = load_runner()
    a = torch.zeros(4)
    b = torch.ones(4) * 1e-6

    assert r.identity_gate_error(a, b, abs_tol=1e-4) == 0.0


def supported_p_summary_with_r(**overrides):
    row = {
        "horizon": 64,
        "G_E_norm_median": 0.1,
        "G_E_readout_norm_median": -0.05,
        "G_P_norm_median": 0.1,
        "G_P_readout_norm_median": -0.05,
        "G_R_norm_median": 0.1,
        "G_R_readout_norm_median": -0.04,
        "P_effective_visibility_ratio_median": 1.15,
        "P_effective_visibility_ratio_bootstrap95_mean": [1.05, 1.25],
        "P_visibility_reverses_magnitude_wins": 10,
        "P_factorization_relative_error_max": 1e-12,
        "ledger_closure_relative_error_max": 1e-12,
        "counterfactual_closure_relative_error_max": 1e-12,
        "R_readout_norm_B_N_median": -0.001,
        "R_readout_norm_B_R_median": -0.001,
        "R_readout_norm_B_R_bootstrap95_mean": [-0.002, -0.0005],
        "R_readout_norm_B_R_wins": 2,
        "R_readout_norm_T_N_median": 0.003,
        "R_readout_norm_T_R_median": 0.003,
        "R_readout_norm_T_R_bootstrap95_mean": [-0.001, 0.006],
        "R_readout_norm_T_R_wins": 11,
        "R_readout_norm_interaction_median": -0.001,
        "R_readout_norm_interaction_bootstrap95_mean": [-0.003, 0.001],
        "R_readout_norm_interaction_wins": 5,
        "counterfactual_P_contribution_median": 0.001,
        "counterfactual_R_contribution_median": -0.0001,
        "counterfactual_interaction_contribution_median": -0.0001,
    }
    row.update(overrides)
    return [row]


def test_classification_does_not_close_when_r_is_only_partial():
    r = load_runner()

    out = r.classify(supported_p_summary_with_r())

    assert out["GDN_P_EFFECTIVE_QUERY_EXPLAINS_READOUT_DEGRADATION"] == "SUPPORTED"
    assert out["GDN_R_TRAJECTORY_STATE_EFFECT"] == "PARTIAL"
    assert out["GDN_R_QUERY_VISIBILITY_PRIMARY_CAUSE"] == "NO_CLEAR_R_MECHANISM"
    assert out["GDN_P_R_QUERY_VISIBILITY_MECHANISM_CLOSURE"] == "PARTIAL"
    assert out["FAILURE_MECHANISM_IDENTIFIED"] == "PARTIAL"
    assert out["FINAL_CLASSIFICATION"] == "GDN_P_R_QUERY_VISIBILITY_MECHANISM_PARTIAL"
    assert out["NEXT_EXPERIMENT"] == "GDN_REQUANTIZATION_QUERY_GEOMETRY_CLOSURE_V2"


def test_classification_requires_robust_r_factor_for_full_closure():
    r = load_runner()

    out = r.classify(
        supported_p_summary_with_r(
            R_readout_norm_T_R_bootstrap95_mean=[0.001, 0.006],
            R_readout_norm_T_R_wins=13,
        )
    )

    assert out["GDN_R_TRAJECTORY_STATE_EFFECT"] == "ROBUST"
    assert out["GDN_R_QUERY_VISIBILITY_PRIMARY_CAUSE"] != "NO_CLEAR_R_MECHANISM"
    assert out["GDN_P_R_QUERY_VISIBILITY_MECHANISM_CLOSURE"] == "SUPPORTED"
    assert out["FINAL_CLASSIFICATION"] == "GDN_P_R_QUERY_VISIBILITY_MECHANISM_CLOSED"


def test_ledger_closure_alone_is_not_mechanism_evidence():
    r = load_runner()

    out = r.classify(
        supported_p_summary_with_r(
            G_P_readout_norm_median=0.02,
            P_effective_visibility_ratio_median=1.0,
            P_effective_visibility_ratio_bootstrap95_mean=[0.95, 1.05],
            P_visibility_reverses_magnitude_wins=0,
            R_readout_norm_T_R_median=-0.001,
            R_readout_norm_T_R_bootstrap95_mean=[-0.003, 0.001],
            R_readout_norm_T_R_wins=5,
        )
    )

    assert out["GDN_TOTAL_READOUT_REVERSAL_EXPLAINED"] == "YES"
    assert out["GDN_P_EFFECTIVE_QUERY_EXPLAINS_READOUT_DEGRADATION"] == "NOT_SUPPORTED"
    assert out["GDN_R_QUERY_VISIBILITY_PRIMARY_CAUSE"] == "NO_CLEAR_R_MECHANISM"
    assert out["GDN_P_R_QUERY_VISIBILITY_MECHANISM_CLOSURE"] == "NOT_SUPPORTED"
    assert out["FINAL_CLASSIFICATION"] == "GDN_P_R_QUERY_VISIBILITY_MECHANISM_NOT_SUPPORTED"


def test_formal_run_requires_horizon_64():
    r = load_runner()
    args = type("Args", (), {"scope": "formal", "horizon": 32})()

    with pytest.raises(RuntimeError, match="horizon 64"):
        r.validate_run_args(args)


def test_prior_provenance_gate_requires_pass_and_all_units():
    r = load_runner()
    manifest = {"GDN_FP_DRIVER_TAPE_CAPTURE": "PASS", "n_units_completed": r.CANONICAL_N}
    parity = {"GDN_FROZEN_DRIVER_FP_REPLAY": "PASS", "n_units_completed": r.CANONICAL_N}

    r.validate_prior_provenance(manifest, parity)
    with pytest.raises(RuntimeError, match="capture"):
        r.validate_prior_provenance({**manifest, "GDN_FP_DRIVER_TAPE_CAPTURE": "FAIL"}, parity)
    with pytest.raises(RuntimeError, match="replay"):
        r.validate_prior_provenance(manifest, {**parity, "GDN_FROZEN_DRIVER_FP_REPLAY": "FAIL"})
