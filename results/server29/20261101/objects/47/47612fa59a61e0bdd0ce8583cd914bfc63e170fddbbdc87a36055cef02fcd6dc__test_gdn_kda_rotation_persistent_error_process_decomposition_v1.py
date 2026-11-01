import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_kda_rotation_persistent_error_process_decomposition_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("persistent_error_process_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_reconciliation_identity_on_same_tensor():
    r = load_runner()
    state = torch.randn(1, 2, 8, 8)
    rot = r.make_rotation(8, "rht", seed=0, dtype=torch.float32)

    a = r.same_tensor_metric_reconciliation(state, "gdn", rot)
    b = r.same_tensor_metric_reconciliation(state, "gdn", rot)

    assert a["old_style_ratio"] == pytest.approx(b["fresh_style_ratio"], rel=0, abs=1e-12)
    assert a["relative_gap"] <= 1e-12


def test_native_and_rotated_residuals_map_to_native_coordinates():
    r = load_runner()
    state = torch.randn(1, 2, 8, 8)
    for arch in ["gdn", "kda"]:
        dim = state.shape[-2] if arch == "gdn" else state.shape[-1]
        rot = r.make_rotation(dim, "haar", seed=3, dtype=torch.float32)
        pair = r.residual_pair(state, arch, rot)
        assert pair["native"].shape == state.shape
        assert pair["rotated_native"].shape == state.shape
        assert torch.linalg.vector_norm(pair["rotated_native"]).item() == pytest.approx(
            torch.linalg.vector_norm(pair["rotated_coord"]).item(), rel=1e-6
        )


def test_fresh_sequence_is_independent_from_prior_quantized_state():
    r = load_runner()
    states = [torch.randn(1, 1, 8, 8) for _ in range(3)]
    contaminated = [s + 1000.0 for s in states]
    rot = r.make_rotation(8, "rht", seed=0, dtype=torch.float32)

    clean_rows = r.fresh_sequence_from_states(states, "kda", rot, contaminated_previous_states=None)
    contaminated_rows = r.fresh_sequence_from_states(states, "kda", rot, contaminated_previous_states=contaminated)

    assert [x["m_R_over_m_N"] for x in clean_rows] == pytest.approx(
        [x["m_R_over_m_N"] for x in contaminated_rows], rel=0, abs=0
    )
    assert all(row["fresh_independence_gate"] == "PASS" for row in contaminated_rows)


def test_frozen_driver_branch_requires_fp_driver_flag():
    r = load_runner()
    rows = r.annotate_frozen_driver_rows([{"unit_id": "u0", "branch": "native"}])
    assert rows[0]["transition_driver_source"] == "FP"
    assert rows[0]["FROZEN_DRIVER_GATE"] == "PASS"


def test_pr_identity_and_energy_identity():
    r = load_runner()
    fp = torch.randn(2, 3)
    pre = fp + torch.randn(2, 3) * 0.01
    q = pre + torch.randn(2, 3) * 0.02

    out = r.pr_decomposition(fp, pre, q)

    assert out["identity_relative_error"] <= 1e-6
    assert out["energy_identity_relative_error"] <= 1e-6
    assert torch.allclose(out["E"], out["P"] + out["R"], atol=1e-7)


def test_cross_time_and_readout_interaction_identities():
    r = load_runner()
    parts = [torch.randn(1, 1, 4, 4) * 0.01 for _ in range(4)]
    query = torch.randn(1, 1, 4)

    state = r.cross_time_energy_decomposition(parts)
    readout = r.readout_space_interaction(parts, query)

    assert state["identity_relative_error"] <= 1e-6
    assert state["total_energy"] == pytest.approx(state["individual_energy"] + state["cross_time_interaction"], rel=1e-6)
    assert readout["identity_relative_error"] <= 1e-6
    assert readout["total_energy"] == pytest.approx(readout["individual_energy"] + readout["cross_time_interaction"], rel=1e-6)
