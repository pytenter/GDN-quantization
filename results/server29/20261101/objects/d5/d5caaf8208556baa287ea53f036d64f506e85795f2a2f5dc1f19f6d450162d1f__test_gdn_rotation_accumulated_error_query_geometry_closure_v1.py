import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_rotation_accumulated_error_query_geometry_closure_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gdn_query_geometry_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_error_and_query():
    error = torch.tensor([[[[1.0, -2.0], [3.0, 4.0], [-1.0, 0.5]]]], dtype=torch.float64)
    query = torch.tensor([[[2.0, -1.0, 3.0]]], dtype=torch.float64)
    return error, query


def test_query_projection_reconstructs_error_and_preserves_orthogonal_energy():
    r = load_runner()
    error, query = sample_error_and_query()
    out = r.query_projection(error, query)

    assert torch.allclose(out["parallel"] + out["perpendicular"], error, atol=1e-12, rtol=1e-12)
    assert out["reconstruction_relative_error"] <= 1e-12
    assert out["orthogonal_energy_relative_error"] <= 1e-12
    assert abs(float((out["parallel"] * out["perpendicular"]).sum())) <= 1e-12


def test_exact_readout_norm_identity_and_visibility_equivalence():
    r = load_runner()
    error, query = sample_error_and_query()
    out = r.query_projection(error, query)

    assert out["readout_identity_relative_error"] <= 1e-12
    assert out["visibility"] == pytest.approx(out["rho"], abs=1e-12, rel=1e-12)


def test_same_query_provenance_requires_native_rotated_and_captured_equality():
    r = load_runner()
    _error, query = sample_error_and_query()

    assert r.assert_same_query_provenance(query, query.clone(), query.clone())["gate"] == "PASS"
    changed = query.clone()
    changed[..., 0] += 1e-6
    assert r.assert_same_query_provenance(query, changed, query)["gate"] == "FAIL"


def test_readout_ratio_factorizes_into_state_and_visibility_ratios():
    r = load_runner()
    error, query = sample_error_and_query()
    rotated = error * 0.7 + torch.tensor([[[[0.0, 0.0], [0.2, -0.1], [0.1, 0.3]]]], dtype=torch.float64)
    native_metrics = r.query_projection(error, query)
    rotated_metrics = r.query_projection(rotated, query)
    out = r.readout_ratio_factorization(native_metrics, rotated_metrics)

    assert out["relative_error"] <= 1e-12
    assert out["readout_ratio"] == pytest.approx(out["state_ratio"] * out["visibility_ratio"], rel=1e-12)


def test_pr_readout_vector_and_energy_identities():
    r = load_runner()
    error, query = sample_error_and_query()
    propagated = error * 0.65
    fresh = error - propagated
    out = r.pr_readout_decomposition(propagated, fresh, error, query)

    assert out["vector_identity_relative_error"] <= 1e-12
    assert out["energy_identity_relative_error"] <= 1e-12


def test_zero_error_case_is_finite_and_has_zero_visibility():
    r = load_runner()
    error = torch.zeros(1, 1, 4, 3, dtype=torch.float64)
    query = torch.randn(1, 1, 4, dtype=torch.float64)
    out = r.query_projection(error, query)

    assert out["error_norm"] == 0.0
    assert out["parallel_norm"] == 0.0
    assert out["perpendicular_norm"] == 0.0
    assert out["visibility"] == 0.0
    assert out["rho"] == 0.0
    assert all(torch.isfinite(v).all() for v in [out["parallel"], out["perpendicular"], out["readout"]])


def test_rotation_inverse_map_returns_native_coordinates():
    r = load_runner()
    state = torch.randn(1, 2, 8, 5, dtype=torch.float64)
    rotation = r.exact().make_experiment_rotation("gdn", 8, kind="rht", seed=0, dtype=torch.float64)
    rotated = r.exact().transform_state_to_branch(state, "gdn", "rotated", rotation)
    recovered = r.exact().inverse_branch_state(rotated, "gdn", "rotated", rotation)

    assert torch.allclose(recovered, state, atol=1e-12, rtol=1e-12)


def test_atomic_head_rows_satisfy_all_geometry_identities():
    r = load_runner()
    generator = torch.Generator().manual_seed(19)
    error = torch.randn(1, 3, 8, 4, generator=generator, dtype=torch.float64)
    query = torch.randn(1, 3, 8, generator=generator, dtype=torch.float64)
    rows = r.atomic_query_geometry_rows(error, query)

    assert len(rows) == 3
    assert max(row["reconstruction_relative_error"] for row in rows) <= 1e-12
    assert max(row["orthogonal_energy_relative_error"] for row in rows) <= 1e-12
    assert max(row["readout_identity_relative_error"] for row in rows) <= 1e-12
    assert max(row["visibility_rho_abs_diff"] for row in rows) <= 1e-12


def test_horizon_classification_uses_exact_reversal_threshold():
    r = load_runner()

    assert r.classify_horizon(state_ratio=0.8, visibility_ratio=0.9, readout_ratio=0.72) == "MAGNITUDE_DOMINANT"
    assert r.classify_horizon(state_ratio=0.8, visibility_ratio=1.1, readout_ratio=0.88) == "VISIBILITY_INCREASE_PARTIAL"
    assert r.classify_horizon(state_ratio=0.8, visibility_ratio=1.25, readout_ratio=1.0) == "VISIBILITY_CANCELS_MAGNITUDE_GAIN"
    assert r.classify_horizon(state_ratio=0.8, visibility_ratio=1.4, readout_ratio=1.12) == "VISIBILITY_REVERSES_READOUT_GAIN"


def test_unit_horizon_aggregation_preserves_stack_norm_and_ratio_factorization():
    r = load_runner()
    query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    native_errors = [
        torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]], dtype=torch.float64),
        torch.tensor([[[[1.0, 4.0], [2.0, 0.0]]]], dtype=torch.float64),
    ]
    rotated_errors = [error * scale for error, scale in zip(native_errors, [0.6, 0.9])]

    def atoms(errors):
        rows = []
        for error in errors:
            projection = r.query_projection(error, query)
            pr = r.pr_readout_decomposition(error * 0.75, error * 0.25, error, query)
            rows.append(r.scalar_geometry_metrics(projection, pr))
        return rows

    native = r.aggregate_atomic_metrics(atoms(native_errors))
    rotated = r.aggregate_atomic_metrics(atoms(rotated_errors))
    paired = r.pair_branch_metrics(native, rotated)

    expected_native_norm = math.sqrt(sum(torch.linalg.vector_norm(error).item() ** 2 for error in native_errors))
    assert native["error_norm"] == pytest.approx(expected_native_norm)
    assert paired["factorization_relative_error"] <= 1e-12
    assert paired["readout_ratio"] == pytest.approx(paired["state_ratio"] * paired["visibility_ratio"], rel=1e-12)


def test_identity_gate_uses_fp32_tolerance_only_for_inherited_pr_decomposition():
    r = load_runner()
    clean = {
        "reconstruction_relative_error": 1e-14,
        "orthogonal_energy_relative_error": 1e-14,
        "readout_identity_relative_error": 1e-14,
        "visibility_rho_abs_diff": 1e-14,
        "pr_vector_identity_relative_error": 2e-7,
        "pr_energy_identity_relative_error": 3e-7,
    }
    bad_projection = dict(clean, readout_identity_relative_error=2e-9)

    assert r.validate_atomic_identities([clean])["gate"] == "PASS"
    assert r.validate_atomic_identities([bad_projection])["gate"] == "FAIL"


def test_pr_state_decomposition_promotes_captured_states_before_subtraction():
    r = load_runner()
    fp = torch.randn(1, 2, 8, 5, dtype=torch.float32)
    pre = fp + torch.randn_like(fp) * 0.03
    quantized = pre + torch.randn_like(pre) * 0.02
    out = r.pr_state_decomposition_fp64(fp, pre, quantized)

    assert out["P"].dtype == torch.float64
    assert out["R"].dtype == torch.float64
    assert out["E"].dtype == torch.float64
    assert torch.equal(out["E"], out["P"] + out["R"])


def test_pooled_factorization_uses_query_weighted_visibility_for_heterogeneous_queries():
    r = load_runner()
    q1 = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    q2 = torch.tensor([[[0.0, 2.0]]], dtype=torch.float64)
    native_errors = [
        torch.tensor([[[[2.0], [1.0]]]], dtype=torch.float64),
        torch.tensor([[[[1.0], [3.0]]]], dtype=torch.float64),
    ]
    rotated_errors = [
        torch.tensor([[[[1.0], [1.5]]]], dtype=torch.float64),
        torch.tensor([[[[2.0], [2.0]]]], dtype=torch.float64),
    ]

    def pooled(errors):
        rows = []
        for error, query in zip(errors, [q1, q2]):
            projection = r.query_projection(error, query)
            pr = r.pr_readout_decomposition(error, torch.zeros_like(error), error, query)
            rows.append(r.scalar_geometry_metrics(projection, pr))
        return r.aggregate_atomic_metrics(rows)

    paired = r.pair_branch_metrics(pooled(native_errors), pooled(rotated_errors))

    assert paired["factorization_relative_error"] <= 1e-12
    assert paired["readout_ratio"] == pytest.approx(
        paired["state_ratio"] * paired["query_weighted_visibility_ratio"], rel=1e-12
    )
    assert paired["geometric_factorization_relative_error"] > 1e-3


def test_report_documents_geometric_vs_query_weighted_visibility(tmp_path):
    r = load_runner()
    original_result_dir = r.RESULT_DIR
    try:
        r.RESULT_DIR = tmp_path
        r.write_report({
            "classifications": {
                "GDN_FROZEN_DRIVER_PROVENANCE": "PASS",
                "GDN_STATE_READOUT_REVERSAL_REPRODUCED": "YES",
                "GDN_ACCUMULATED_STATE_MAGNITUDE_GAIN": "ATTENUATED",
                "GDN_QUERY_VISIBILITY_SHIFT": "INCREASED",
                "GDN_QUERY_VISIBILITY_EXPLAINS_READOUT_REVERSAL": "PARTIAL",
                "GDN_READOUT_SPACE_P_COMPONENT_GAIN": -0.1,
                "GDN_READOUT_SPACE_R_COMPONENT_GAIN": -0.2,
                "GDN_READOUT_SPACE_P_R_INTERACTION": "NO_CLEAR_CHANGE",
                "GDN_READOUT_REVERSAL_PRIMARY_COMPONENT": "MULTI_FACTOR",
                "GDN_ACCUMULATED_ERROR_QUERY_GEOMETRY_CLOSURE": "PARTIAL",
                "FAILURE_MECHANISM_IDENTIFIED": "PARTIAL",
                "FINAL_CLASSIFICATION": "GDN_READOUT_REVERSAL_MECHANISM_PARTIAL",
                "NEXT_EXPERIMENT": "GDN_READOUT_REVERSAL_REASSESSMENT_V2",
            },
            "reproduction": {},
            "horizon_summary": [],
            "identity_gates": {},
            "provenance": {},
            "environment": {},
        })
        text = (tmp_path / "report.md").read_text()
    finally:
        r.RESULT_DIR = original_result_dir

    assert "Geometric Vq = ||E_parallel|| / ||E||" in text
    assert "query-weighted visibility ||E^T q|| / ||E||" in text


def test_aggregate_horizons_honors_bootstrap_sample_count(monkeypatch):
    r = load_runner()
    calls = []

    def fake_bootstrap(values, samples=10000, seed=r.BOOTSTRAP_SEED):
        calls.append({"samples": samples, "seed": seed, "n": len(values)})
        return [0.0, 1.0]

    monkeypatch.setattr(r, "bootstrap_mean_ci", fake_bootstrap)
    row = {
        "horizon": 1,
        "G_state": 0.1,
        "G_parallel": 0.2,
        "G_read": -0.3,
        "visibility_ratio": 1.1,
        "query_weighted_visibility_ratio": 1.2,
        "delta_Iq": 0.0,
        "G_Pq": -0.1,
        "G_Rq": -0.2,
        "state_ratio": 0.9,
        "parallel_ratio": 0.8,
        "readout_ratio": 1.3,
        "factorized_readout_ratio": 1.3,
        "factorization_relative_error": 0.0,
        "reversal_threshold_holds": True,
    }
    for branch in ["native", "rotated"]:
        row.update({
            f"{branch}_error_norm": 1.0,
            f"{branch}_parallel_norm": 1.0,
            f"{branch}_perpendicular_norm": 1.0,
            f"{branch}_visibility": 0.1,
            f"{branch}_rho": 0.1,
            f"{branch}_query_weighted_visibility": 0.01,
            f"{branch}_readout_norm": 0.01,
            f"{branch}_query_norm": 0.1,
            f"{branch}_Pq_norm": 0.01,
            f"{branch}_Rq_norm": 0.01,
            f"{branch}_normalized_interference": 0.0,
        })

    r.aggregate_horizons([row], [1], bootstrap_samples=123)

    assert calls
    assert {call["samples"] for call in calls} == {123}
