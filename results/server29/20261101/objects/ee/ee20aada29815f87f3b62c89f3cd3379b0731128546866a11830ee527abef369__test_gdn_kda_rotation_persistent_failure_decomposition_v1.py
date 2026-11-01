import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_kda_rotation_persistent_failure_decomposition_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("failure_decomp_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gdn_rotated_residual_maps_back_to_native_coordinates_and_preserves_norm():
    r = load_runner()
    state = torch.randn(1, 2, 8, 8)
    rot = r.make_rotation(8, "haar", seed=0, dtype=torch.float32)

    native, rotated, rot_coord = r.residual_pair(state, "gdn", rot)

    assert native.shape == state.shape
    assert rotated.shape == state.shape
    assert torch.linalg.vector_norm(rotated).item() == pytest.approx(torch.linalg.vector_norm(rot_coord).item(), rel=1e-6)


def test_kda_rotated_residual_maps_back_to_native_coordinates_and_preserves_norm():
    r = load_runner()
    state = torch.randn(1, 2, 8, 8)
    rot = r.make_rotation(8, "haar", seed=1, dtype=torch.float32)

    native, rotated, rot_coord = r.residual_pair(state, "kda", rot)

    assert native.shape == state.shape
    assert rotated.shape == state.shape
    assert torch.linalg.vector_norm(rotated).item() == pytest.approx(torch.linalg.vector_norm(rot_coord).item(), rel=1e-6)


def test_rho_q_matches_direct_formula():
    r = load_runner()
    err = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    q = torch.tensor([[[5.0, 6.0]]])

    got = r.rho_q(err, q)
    readout = torch.einsum("bhkv,bhk->bhv", err, q)
    expected = torch.linalg.vector_norm(readout) / (torch.linalg.vector_norm(err) * torch.linalg.vector_norm(q) + r.EPS)

    assert got == pytest.approx(float(expected.item()))


def test_factorial_residuals_have_requested_magnitudes_and_directions():
    r = load_runner()
    e_n = torch.randn(1, 1, 4, 4)
    e_r = torch.randn(1, 1, 4, 4)

    fac = r.factorial_residuals(e_n, e_r)

    assert torch.linalg.vector_norm(fac["RN"]).item() == pytest.approx(torch.linalg.vector_norm(e_r).item(), rel=1e-6)
    assert torch.linalg.vector_norm(fac["NR"]).item() == pytest.approx(torch.linalg.vector_norm(e_n).item(), rel=1e-6)
    assert r.flat_cosine(fac["RN"], e_n) == pytest.approx(1.0, abs=1e-6)
    assert r.flat_cosine(fac["NR"], e_r) == pytest.approx(1.0, abs=1e-6)


def test_zero_injection_noninterference():
    r = load_runner()
    state = torch.randn(1, 2, 4, 4)
    injected = r.inject_residual_once(state, torch.zeros_like(state))

    assert torch.equal(injected, state)


def test_propagation_gain_identity_cases():
    r = load_runner()
    err = torch.randn(1, 1, 4, 4)
    gains = r.propagation_gain_curve(err, [torch.eye(4) for _ in range(4)], horizons=[1, 4])

    assert gains[1] == pytest.approx(1.0, rel=1e-6)
    assert gains[4] == pytest.approx(1.0, rel=1e-6)


def test_fresh_error_decomposition_identity():
    r = load_runner()
    parts = [torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)]

    out = r.fresh_error_energy_decomposition(parts)

    assert out["total_squared_norm"] == pytest.approx(out["individual_energy"] + out["cross_time_interaction"], rel=1e-6)
