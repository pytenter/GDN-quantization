import importlib.util
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[2] / "experiments" / "rotation" / "run_kda_rotation_decay_drift_causal_closure_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("kda_decay_closure_v1", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_final_decay_mapping_matches_kernel_equation():
    r = load_runner()
    gate = torch.tensor([[[[-1.0, 0.5], [0.0, 1.0]]]], dtype=torch.float64)
    a_log = torch.tensor([-0.2, 0.3], dtype=torch.float64)
    bias = torch.tensor([0.1, -0.1, 0.2, 0.4], dtype=torch.float64)
    got = r.final_log_decay(gate, a_log, bias)
    expected = -torch.exp(a_log.float()).view(1, 1, 2, 1) * torch.nn.functional.softplus(
        gate.float() + bias.float().view(1, 1, 2, 2)
    )
    assert torch.allclose(got, expected)


def test_decay_self_restore_identity_and_no_trace_mutation():
    r = load_runner()
    plan = r.InterventionPlan("R_DN", "rotated", True, "N", ("decay",))
    raw = {name: torch.randn(2, 3) for name in ("q", "k", "v", "beta", "decay")}
    source = {name: torch.randn(2, 3) for name in raw}
    sources = {"N": source}
    raw_before = {k: v.clone() for k, v in raw.items()}
    source_before = {k: v.clone() for k, v in source.items()}
    out, _ = r.apply_intervention(raw, sources, plan, 8)
    assert torch.equal(out["decay"], source["decay"])
    assert all(torch.equal(raw[k], raw_before[k]) for k in raw)
    assert all(torch.equal(source[k], source_before[k]) for k in source)


def test_native_rotated_decay_coordinates_require_no_rotation():
    r = load_runner()
    decay = torch.randn(1, 1, 2, 4)
    u = torch.flip(torch.eye(4), dims=[0])
    assert torch.equal(r.BASE.driver_to_branch_coordinates("decay", decay, "rotated", u), decay)
    assert torch.equal(r.BASE.driver_to_semantic_coordinates("decay", decay, "rotated", u), decay)


def test_b_update_is_invariant_to_decay_only_change():
    r = load_runner()
    k = torch.randn(1, 3, 4, dtype=torch.float64)
    v = torch.randn(1, 3, 5, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(1, 3, dtype=torch.float64))
    b_before = r.effective_b_update(k, beta, v)
    _unused_decay = torch.rand(1, 3, 4, dtype=torch.float64)
    b_after = r.effective_b_update(k, beta, v)
    assert torch.equal(b_before, b_after)


def test_explicit_a_reconstructs_kda_transition():
    r = load_runner()
    state = torch.randn(1, 2, 4, 5, dtype=torch.float64)
    k = torch.randn(1, 2, 4, dtype=torch.float64)
    v = torch.randn(1, 2, 5, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(1, 2, dtype=torch.float64))
    decay = torch.rand(1, 2, 4, dtype=torch.float64)
    kn = r.normalize_k(k)
    decayed = state * decay.unsqueeze(-1)
    residual_v = v - torch.einsum("...kv,...k->...v", decayed, kn)
    expected = decayed + beta[..., None, None] * torch.einsum("...k,...v->...kv", kn, residual_v)
    assert torch.allclose(r.explicit_kda_update(state, k, v, beta, decay), expected)


def test_single_step_pulse_isolation_and_horizon_alignment():
    r = load_runner()
    plan = r.InterventionPlan("N_DR_PULSE_T4", "native", True, "R", ("decay",), 4)
    raw = {name: torch.ones(2) * i for i, name in enumerate(("q", "k", "v", "beta", "decay"), 1)}
    source = {name: torch.ones(2) * (10 + i) for i, name in enumerate(raw, 1)}
    inactive, _ = r.apply_intervention(raw, {"R": source}, plan, 3)
    active, prov = r.apply_intervention(raw, {"R": source}, plan, 4)
    assert all(torch.equal(inactive[k], raw[k]) for k in raw)
    assert torch.equal(active["decay"], source["decay"])
    assert all(torch.equal(active[k], raw[k]) for k in ("q", "k", "v", "beta"))
    assert prov["horizon"] == 4 and prov["active"] is True


@pytest.mark.parametrize("branch,source,basis", [
    ("R_DN", "N", "rotated"),
    ("N_DR", "R", "native"),
])
def test_persistent_decay_transplant_branch_correctness(branch, source, basis):
    r = load_runner()
    plan = next(p for p in r.formal_plans(False, False) if p.name == branch)
    assert plan.source == source
    assert plan.basis == basis
    assert plan.drivers == ("decay",)
    assert plan.active(0) is False
    assert all(plan.active(h) for h in (1, 2, 8, 64))


def test_factorial_branch_correctness():
    r = load_runner()
    plans = {p.name: p for p in r.formal_plans(False, False)}
    assert plans["R_DFP"].drivers == ("decay",)
    assert plans["R_KBFP"].drivers == ("k", "beta")
    assert plans["R_DKBFP"].drivers == ("decay", "k", "beta")
    assert all(plans[name].source == "FP" for name in ("R_DFP", "R_KBFP", "R_DKBFP"))


def test_dose_interpolation_endpoints_and_no_mutation():
    r = load_runner()
    rotated = torch.randn(2, 3)
    native = torch.randn(2, 3)
    rotated_before, native_before = rotated.clone(), native.clone()
    assert torch.equal(r.interpolate_log_decay(rotated, native, 0.0), rotated)
    assert torch.equal(r.interpolate_log_decay(rotated, native, 1.0), native)
    assert torch.equal(rotated, rotated_before)
    assert torch.equal(native, native_before)


def test_plans_have_unique_names_and_baselines_precede_sources():
    r = load_runner()
    plans = r.formal_plans()
    names = [p.name for p in plans]
    assert len(names) == len(set(names))
    assert names[:3] == ["FP", "N", "R"]
