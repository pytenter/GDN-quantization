import importlib.util
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_rotation_affine_decay_value_unified_causal_closure_v1.py"
SPEC = importlib.util.spec_from_file_location("affine_unified", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def record(seed=0, basis="native"):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, 1, 2, 4, generator=g)
    v = torch.randn(1, 1, 2, 3, generator=g)
    beta = torch.sigmoid(torch.randn(1, 1, 2, generator=g))
    decay = torch.sigmoid(torch.randn(1, 1, 2, 4, generator=g))
    a = M.DECAY.effective_a_operator(k, beta, decay, True)
    b = M.DECAY.effective_b_update(k, beta, v, True)
    return {"k": k, "beta": beta, "v_semantic": v, "effective_decay": decay,
            "effective_a": a.float(), "effective_b": b.float(),
            "use_qk_l2norm_in_kernel": True, "basis": basis}


def synthetic_rows():
    rows = []
    factors = {
        "N": 1.0, "R": 2.0, "R_DN": 1.7, "R_vN": 1.8, "R_BN": 1.6,
        "R_DN_vN": 1.4, "R_DN_BN": 1.2, "R_MN_BN": 1.1, "R_STATE_N": 1.0,
        "N_DR": 1.3, "N_vR": 1.2, "N_BR": 1.4, "N_DR_vR": 1.6,
        "N_DR_BR": 1.8, "N_MR_BR": 1.9,
    }
    for i in range(18):
        for branch, value in factors.items():
            rows.append({"unit_id": f"u{i:02d}", "branch": branch, "FutureKL_AUC": value + i * 1e-4})
    return rows


def test_01_formal_manifest_unique():
    names = [p.name for p in M.formal_plans()]
    assert len(names) == len(set(names)) == 16


def test_02_required_branches_present():
    names = {p.name for p in M.formal_plans()}
    assert {"R_DN_vN", "R_DN_BN", "R_MN_BN", "R_STATE_N", "N_MR_BR"} <= names


def test_03_only_state_plan_transplants_state():
    state = [p.name for p in M.formal_plans() if p.state_source]
    assert state == ["R_STATE_N"]


def test_04_self_transplant_manifest():
    names = {p.name for p in M.stage0_plans()}
    assert {"N_DN", "R_DR", "N_vN", "R_vR", "N_BN", "R_BR", "N_MN", "R_MR"} <= names


def test_05_bootstrap_reproducible():
    x = list(range(18))
    assert M.median_bootstrap_ci(x, n=200, seed=7) == M.median_bootstrap_ci(x, n=200, seed=7)


def test_06_bootstrap_targets_median():
    x = [0.0] * 17 + [1000.0]
    lo, hi = M.median_bootstrap_ci(x, n=500, seed=9)
    assert lo == 0.0 and hi == 0.0


def test_07_sign_test_extreme():
    assert M.sign_test_two_sided([1.0] * 18) < 1e-4


def test_08_m_reconstruction_identity():
    r = record()
    assert torch.allclose(M.GAB.a_from_record(r), r["effective_a"].double(), atol=1e-6)


def test_09_b_reconstruction_identity():
    r = record()
    assert torch.allclose(M.GAB.canonical_b_from_record(r), r["effective_b"].double(), atol=1e-6)


def test_10_m_factor_source_selection():
    records = {"N": record(1), "R": record(2)}
    mixed = M.m_from_records(records, M.m_factors("R", "R", "N"))
    expected = M.DECAY.effective_a_operator(records["R"]["k"], records["R"]["beta"], records["N"]["effective_decay"], True)
    assert torch.allclose(mixed, expected)


def test_11_v_factor_source_selection():
    records = {"N": record(1), "R": record(2)}
    rotation = torch.eye(3)
    mixed = M.b_from_records(records, M.b_factors("R", "R", "N"), "native", rotation)
    expected = M.DECAY.effective_b_update(records["R"]["k"], records["R"]["beta"], records["N"]["v_semantic"], True)
    assert torch.allclose(mixed, expected)


def test_12_operator_update_identity():
    r = record()
    state = torch.randn_like(r["effective_b"])
    q = torch.randn(1, 1, 2, 4)
    _, actual = M.GAB.operator_step(state, r["effective_a"], r["effective_b"], M.GAB._last_token(q), True)
    expected = torch.einsum("...kl,...lv->...kv", r["effective_a"].double(), state.double()) + r["effective_b"].double()
    assert torch.allclose(actual, expected)


def test_13_coordinate_roundtrip():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    state = torch.randn(1, 1, 2, 4, 3)
    rotated = M.state_to_basis(state, "native", "rotated", q)
    restored = M.state_to_basis(rotated, "rotated", "native", q)
    assert torch.allclose(state, restored, atol=1e-5)


def test_14_output_is_finite():
    state = torch.randn(1, 1, 2, 4, 3)
    q = torch.randn(1, 2, 4)
    assert torch.isfinite(M.output_from_state(state, q, True)).all()


def test_15_effect_reports_mean_and_median():
    e = M.effect([0.0] * 17 + [18.0], "x")
    assert e["paired_median"] == 0.0 and e["mean"] == 1.0 and e["n"] == 18


def test_16_per_unit_closure_correct():
    rows = M.closure_rows(synthetic_rows())
    assert len(rows) == 18
    assert rows[0]["closure_DV"] == pytest.approx(0.6)
    assert rows[0]["closure_MB"] == pytest.approx(0.9)


def test_17_no_ratio_of_aggregate_medians():
    rows = M.closure_rows(synthetic_rows())
    e = M.effect([r["closure_DB"] for r in rows], "db")
    assert e["n"] == 18 and e["paired_median"] == pytest.approx(0.8)


def test_18_negative_rotation_gap_rejected():
    rows = synthetic_rows()
    for row in rows:
        if row["unit_id"] == "u00" and row["branch"] == "R":
            row["FutureKL_AUC"] = 0.5
    with pytest.raises(RuntimeError):
        M.closure_rows(rows)


def test_19_summary_has_all_closures():
    rows = synthetic_rows(); closures = M.closure_rows(rows); summary = M.summarize(rows, closures)
    assert all(f"MEDIAN_CLOSURE_{x}" in summary for x in ("D", "V", "B", "DV", "DB", "MB", "STATE"))


def test_20_all_synthetic_values_finite():
    rows = M.closure_rows(synthetic_rows())
    assert all(torch.isfinite(torch.tensor([v for k, v in row.items() if k != "unit_id"])).all() for row in rows)
