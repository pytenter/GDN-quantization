import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_rotation_dynamic_gate_p_selectivity_causal_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gdn_gate_causal_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_effective_gate_multiplier_is_silu_of_gate():
    r = load_runner()
    gate = torch.tensor([-2.0, 0.0, 2.0])

    actual = r.effective_gate_multiplier(gate)

    assert torch.allclose(actual, F.silu(gate.float()).double())


def test_branch_gate_provenance_zero_for_shared_gate():
    r = load_runner()
    a = torch.tensor([1.0, 2.0, -3.0])
    branches = {name: a.clone() for name in ["NN", "RN", "NR", "RR"]}

    out = r.gate_provenance_metrics(branches)

    assert out["D_Pgate"] == pytest.approx(0.0)
    assert out["D_Rgate"] == pytest.approx(0.0)
    assert out["max_pairwise_gate_rel_diff"] == pytest.approx(0.0)


def test_fixed_gate_diagonal_transfer_identity_and_energy_normalization():
    r = load_runner()
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    delta_p = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    delta_r = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)

    out = r.gate_energy_alignment_metrics(a, delta_p, delta_r)

    assert out["pi_P_sum"] == pytest.approx(1.0)
    assert out["pi_R_sum"] == pytest.approx(1.0)
    assert out["G_P_squared"] == pytest.approx(out["A_P"])
    assert out["G_R_squared"] == pytest.approx(out["A_R"])
    assert out["identity_relerr_P"] < 1e-12
    assert out["identity_relerr_R"] < 1e-12


def test_isotropic_gate_preserves_rms_and_has_unit_lambda():
    r = load_runner()
    a = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    delta_p = torch.tensor([3.0, 0.0, 4.0], dtype=torch.float64)
    delta_r = torch.tensor([0.0, 5.0, 12.0], dtype=torch.float64)

    iso = r.isotropic_gate_metrics(a, delta_p, delta_r)

    assert iso["c"] == pytest.approx(torch.sqrt((a * a).mean()).item())
    assert iso["G_P_iso"] == pytest.approx(iso["c"])
    assert iso["G_R_iso"] == pytest.approx(iso["c"])
    assert iso["Lambda_iso"] == pytest.approx(1.0)


def test_shuffle_preserves_multiset_rms_and_is_deterministic():
    r = load_runner()
    a = torch.arange(1, 9, dtype=torch.float64)

    s0 = r.shuffle_gate(a, seed=3)
    s1 = r.shuffle_gate(a, seed=3)
    s2 = r.shuffle_gate(a, seed=4)

    assert torch.equal(s0, s1)
    assert not torch.equal(s0, s2)
    assert torch.equal(torch.sort(s0.flatten()).values, torch.sort(a.flatten()).values)
    assert torch.sqrt((s0 * s0).mean()).item() == pytest.approx(torch.sqrt((a * a).mean()).item())


def test_gate_drift_decomposition_is_exact():
    r = load_runner()
    x_base = torch.tensor([1.0, -2.0, 3.0])
    a_base = torch.tensor([0.5, 2.0, -1.0])
    dx = torch.tensor([0.1, 0.2, -0.3])
    da = torch.tensor([0.4, -0.5, 0.6])

    out = r.gate_drift_decomposition(x_base, a_base, x_base + dx, a_base + da)

    expected = ((x_base + dx) * (a_base + da) - x_base * a_base).double()
    assert torch.allclose(out["total"], expected)
    assert torch.allclose(out["fixed_gate"] + out["gate_drift"] + out["interaction"], expected)


def test_downstream_continuation_after_gate_intervention_shapes():
    r = load_runner()
    weighted = torch.randn(1, 1, 2, 3)
    gate = torch.randn(1, 1, 2, 3)
    out_proj_weight = torch.randn(5, 6)

    out = r.continue_after_gate(weighted, gate, out_proj_weight)

    assert out["dynamic_gate"].shape == weighted.shape
    assert out["merge_heads"].shape == (1, 1, 6)
    assert out["out_proj"].shape == (1, 1, 5)


def test_zero_perturbation_noninterference():
    r = load_runner()
    a = torch.randn(8)
    z = torch.zeros(8)

    out = r.gate_energy_alignment_metrics(a, z, z)

    assert out["G_P"] == pytest.approx(0.0)
    assert out["G_R"] == pytest.approx(0.0)
    assert out["A_P"] == pytest.approx(0.0)
    assert out["A_R"] == pytest.approx(0.0)
