import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[2] / "experiments" / "rotation" / "run_kda_rotation_g_ab_operator_causal_closure_v1.py"
spec = importlib.util.spec_from_file_location("g_ab_runner", RUNNER)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def rotation(n=4):
    q, _ = torch.linalg.qr(torch.tensor([
        [1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 1.0, 2.0], [2.0, 0.0, 0.0, 1.0],
    ])[:n, :n])
    return q


def tensors():
    torch.manual_seed(11)
    state = torch.randn(1, 2, 4, 3, dtype=torch.double)
    a = torch.randn(1, 2, 4, 4, dtype=torch.double)
    b = torch.randn(1, 2, 4, 3, dtype=torch.double)
    q = torch.randn(1, 2, 4, dtype=torch.double)
    return state, a, b, q


def test_g_equals_ga_plus_gb_exactly():
    state, aq, bq, _ = tensors()
    af, bf = aq * 0.9, bq * 1.1
    direct = (torch.einsum("...kl,...lv->...kv", aq, state) + bq) - (torch.einsum("...kl,...lv->...kv", af, state) + bf)
    ga = torch.einsum("...kl,...lv->...kv", aq - af, state)
    gb = bq - bf
    assert torch.allclose(direct, ga + gb, atol=1e-12, rtol=1e-12)


def test_b_canonicalization_identity():
    _, _, b, _ = tensors(); u = rotation(3)
    assert torch.allclose(M.b_to_canonical(M.b_to_basis(b, "rotated", u), "rotated", u), b, atol=1e-12)


def test_native_to_rotated_b_coordinate_conversion():
    _, _, b, _ = tensors(); u = rotation(3).double()
    assert torch.equal(M.b_to_basis(b, "rotated", u), b @ u)


def test_rotated_to_native_b_coordinate_conversion():
    _, _, b, _ = tensors(); u = rotation(3).double(); rotated = b @ u
    assert torch.allclose(M.b_to_canonical(rotated, "rotated", u), b, atol=1e-12)


def test_a_only_intervention_does_not_alter_b():
    _, a, b, _ = tensors(); replacement = a + 1
    before = b.clone(); pair = (replacement.clone(), b.clone())
    assert torch.equal(pair[1], before)


def test_b_only_intervention_does_not_alter_a():
    _, a, b, _ = tensors(); replacement = b + 1
    before = a.clone(); pair = (a.clone(), replacement.clone())
    assert torch.equal(pair[0], before)


@pytest.mark.parametrize("which", ["A", "B", "AB"])
def test_operator_self_restore_identity(which):
    state, a, b, q = tensors()
    reference = M.operator_step(state, a, b, q, False)
    ar = a.clone() if which in {"A", "AB"} else a
    br = b.clone() if which in {"B", "AB"} else b
    restored = M.operator_step(state, ar, br, q, False)
    assert torch.equal(reference[0], restored[0])
    assert torch.equal(reference[1], restored[1])


def test_r_abfp_g_is_zero():
    state, af, bf, _ = tensors()
    ga = torch.einsum("...kl,...lv->...kv", af - af, state)
    gb = bf - bf
    assert torch.count_nonzero(ga + gb) == 0


def test_horizon_indexing_and_alignment():
    assert M.HORIZONS == [1, 2, 4, 8, 16, 32, 64]
    plans = M.plans_for_phase("round1")
    assert [p.name for p in plans] == ["FP", "N", "R", "N_AFP", "N_BFP", "N_ABFP", "R_AFP", "R_BFP", "R_ABFP"]


def test_transplant_trace_alignment():
    plans = {p.name: p for p in M.plans_for_phase("round2")}
    assert (plans["R_AN"].a_source, plans["R_AN"].b_source) == ("N", "R")
    assert (plans["R_BN"].a_source, plans["R_BN"].b_source) == ("R", "N")
    assert (plans["N_BR"].a_source, plans["N_BR"].b_source) == ("N", "R")


def test_no_stale_cached_artifact_mutation():
    state, a, b, q = tensors(); originals = [x.clone() for x in (state, a, b, q)]
    M.operator_step(state, a, b, q, True)
    assert all(torch.equal(x, y) for x, y in zip((state, a, b, q), originals))


def test_query_override_is_independent_of_operators():
    plan = {p.name: p for p in M.plans_for_phase("query")}["R_ABFP_QFP"]
    assert (plan.a_source, plan.b_source, plan.q_source) == ("FP", "FP", "FP")


def test_operator_step_matches_manual_state_and_readout():
    state, a, b, q = tensors()
    out, new_state = M.operator_step(state, a, b, q, False, scale=0.25)
    expected_state = torch.einsum("...kl,...lv->...kv", a, state) + b
    expected_out = torch.einsum("...k,...kv->...v", q * 0.25, expected_state).unsqueeze(1)
    assert torch.allclose(new_state, expected_state)
    assert torch.allclose(out, expected_out)


def test_dedupe_prefers_first_aligned_record():
    rows = [{"unit_id": "u", "horizon": 1, "branch": "R", "x": 1},
            {"unit_id": "u", "horizon": 1, "branch": "R", "x": 2}]
    out = M.dedupe(rows, ("unit_id", "horizon", "branch"))
    assert out == [rows[0]]


def test_b_factorial_endpoint_coordinate_identities_without_running_factorial():
    _, _, b_nnn, _ = tensors(); u = rotation(3).double()
    b_rrr = M.b_to_basis(b_nnn, "rotated", u)
    assert torch.allclose(M.b_to_canonical(b_rrr, "rotated", u), b_nnn, atol=1e-12)


def test_b_internal_plans_keep_a_on_receiving_trajectory():
    plans = {p.name: p for p in M.plans_for_phase("b_internal")}
    assert plans["R_B_betaN"].a_source == "R"
    assert plans["R_B_kN"].a_source == "R"
    assert plans["R_B_vN"].a_source == "R"
    assert plans["N_B_vR"].a_source == "N"


def test_b_internal_factor_sources_change_only_named_factor():
    plans = {p.name: p for p in M.plans_for_phase("b_internal")}
    assert plans["R_B_betaN"].b_factor_sources == {"beta": "N", "k": "R", "v": "R"}
    assert plans["R_B_kN"].b_factor_sources == {"beta": "R", "k": "N", "v": "R"}
    assert plans["R_B_vN"].b_factor_sources == {"beta": "R", "k": "R", "v": "N"}


def test_factorial_contrast_recovers_beta_main_effect():
    rows = []
    for unit in ("u1", "u2"):
        for beta in ("R", "N"):
            for k in ("R", "N"):
                for v in ("R", "N"):
                    rows.append({"unit_id": unit, "operator": f"B_{beta}{k}{v}", "GB_error": 3.0 if beta == "R" else 1.0})
    effects = M.factorial_effects(rows)
    assert effects["beta"]["paired_median"] == pytest.approx(2.0)
    assert effects["k"]["paired_median"] == pytest.approx(0.0)
    assert effects["v"]["paired_median"] == pytest.approx(0.0)
