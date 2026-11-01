import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_kda_rotation_residual_direction_causal_closure_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("residual_direction_closure_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_gdn_residual_mapping_and_frobenius_preservation():
    r = load_runner()
    state = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    rot = r.make_rotation(8, "rht", seed=0, dtype=torch.float64)

    pair = r.residual_pair(state, "gdn", rot)
    native_expected = r.fake_quant_state(state, "C128")[0] - state
    rotated_state = r.rotate_state_key_axis(state, rot)
    rotated_coord_expected = r.fake_quant_state(rotated_state, "C128")[0] - rotated_state

    assert torch.allclose(pair["E_N"], native_expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(pair["E_R_coord"], rotated_coord_expected, atol=1e-12, rtol=1e-12)
    assert pair["frobenius_mapping_relative_error"] <= 1e-12


def test_fresh_kda_residual_mapping_and_frobenius_preservation():
    r = load_runner()
    state = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    rot = r.make_rotation(8, "rht", seed=0, dtype=torch.float64)

    pair = r.residual_pair(state, "kda", rot)
    native_expected = r.fake_quant_state(state, "R128")[0] - state
    rotated_state = r.rotate_state_value_axis(state, rot)
    rotated_coord_expected = r.fake_quant_state(rotated_state, "R128")[0] - rotated_state

    assert torch.allclose(pair["E_N"], native_expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(pair["E_R_coord"], rotated_coord_expected, atol=1e-12, rtol=1e-12)
    assert pair["frobenius_mapping_relative_error"] <= 1e-12


def test_factorial_construction_exact_magnitudes_and_directions():
    r = load_runner()
    e_native = torch.randn(1, 1, 8, 8)
    e_rotated = torch.randn(1, 1, 8, 8) * 0.3

    facts = r.factorial_residuals(e_native, e_rotated)
    checks = r.factorial_sanity_checks(facts, e_native, e_rotated)

    assert set(facts) == {"ZERO", "NN", "RN", "NR", "RR"}
    assert torch.equal(facts["ZERO"], torch.zeros_like(e_native))
    assert torch.linalg.vector_norm(facts["NN"]).item() == pytest.approx(torch.linalg.vector_norm(e_native).item(), rel=1e-6)
    assert torch.linalg.vector_norm(facts["RN"]).item() == pytest.approx(torch.linalg.vector_norm(e_rotated).item(), rel=1e-6)
    assert torch.linalg.vector_norm(facts["NR"]).item() == pytest.approx(torch.linalg.vector_norm(e_native).item(), rel=1e-6)
    assert torch.linalg.vector_norm(facts["RR"]).item() == pytest.approx(torch.linalg.vector_norm(e_rotated).item(), rel=1e-6)
    assert checks["gate"] == "PASS"
    assert checks["cos_RN_EN"] == pytest.approx(1.0, abs=1e-6)
    assert checks["cos_NR_ER"] == pytest.approx(1.0, abs=1e-6)


def test_zero_branch_and_single_pulse_bookkeeping_are_noninterfering():
    r = load_runner()
    cache = {0: torch.ones(1, 1, 2, 2), 1: torch.full((1, 1, 2, 2), 3.0)}
    fp = {k: v.clone() for k, v in cache.items()}
    zero = {k: torch.zeros_like(v) for k, v in cache.items()}
    pulse = {0: torch.ones(1, 1, 2, 2), 1: torch.full((1, 1, 2, 2), -2.0)}

    audit_zero = r.apply_single_pulse_to_stack(cache, zero, already_applied=False)
    assert audit_zero["injection_count"] == 1
    assert audit_zero["future_quantization_count"] == 0
    assert r.stack_max_abs_diff(cache, fp) == 0.0

    audit_pulse = r.apply_single_pulse_to_stack(cache, pulse, already_applied=False)
    assert audit_pulse["injection_count"] == 1
    assert torch.equal(cache[0], fp[0] + pulse[0])
    assert torch.equal(cache[1], fp[1] + pulse[1])
    with pytest.raises(RuntimeError):
        r.apply_single_pulse_to_stack(cache, pulse, already_applied=True)


def test_readout_only_and_memory_only_invariants():
    r = load_runner()
    cache = {0: torch.randn(1, 1, 2, 2)}
    residual = {0: torch.ones(1, 1, 2, 2)}
    logits = torch.randn(1, 1, 11)

    readout = r.readout_only_intervention(cache, residual, query_stack={0: torch.ones(1, 1, 2)})
    assert r.stack_max_abs_diff(cache, readout["cache_after"]) == 0.0
    assert torch.equal(readout["delta_o"][0], torch.full((1, 1, 2), 2.0))

    mem = r.memory_only_intervention(cache, residual, current_logits=logits)
    assert torch.equal(mem["current_logits_after"], logits)
    assert mem["current_logit_noninterference_max_abs"] == 0.0
    assert r.stack_max_abs_diff(mem["future_cache"], cache) > 0.0


def test_teacher_forced_branch_equality_and_horizon_indexing():
    r = load_runner()
    tokens = [10, 11, 12, 13, 14, 15]
    assert r.teacher_token_for_timestep(tokens, 1) == 10
    assert r.teacher_token_for_future_horizon(tokens, 2, 3) == 14
    assert r.selected_horizons(max_horizon=16) == [0, 1, 2, 4, 8, 16]
    assert r.selected_horizons(max_horizon=3) == [0, 1, 2]
    assert r.teacher_forced_sequences_identical({"A": tokens, "B": list(tokens)})
    assert not r.teacher_forced_sequences_identical({"A": tokens, "B": tokens[:-1]})


def test_factorial_effect_formulas_use_unit_level_auc_values():
    r = load_runner()
    eff = r.factorial_effects({"NN": 5.0, "RN": 3.0, "NR": 8.0, "RR": 7.0})

    assert eff["M_N"] == pytest.approx(2.0)
    assert eff["M_R"] == pytest.approx(1.0)
    assert eff["D_N"] == pytest.approx(3.0)
    assert eff["D_R"] == pytest.approx(4.0)
    assert eff["interaction"] == pytest.approx(1.0)


def test_identity_propagation_gain_and_future_readout_gain_normalization():
    r = load_runner()
    error = torch.randn(1, 1, 4, 3)
    operators = [torch.eye(4) for _ in range(4)]
    queries = [torch.randn(1, 1, 4) for _ in range(4)]

    p = r.propagation_gain_curve(error, operators, horizons=[1, 2, 4])
    f = r.future_readout_gain_curve(error, operators, queries, horizons=[1, 2, 4])

    assert p[1] == pytest.approx(1.0, abs=1e-6)
    assert p[2] == pytest.approx(1.0, abs=1e-6)
    assert p[4] == pytest.approx(1.0, abs=1e-6)
    assert set(f) == {1, 2, 4}
    assert all(value >= 0.0 for value in f.values())
