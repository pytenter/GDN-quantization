import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_gdn_kda_temporal_exposure_residual_audit_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("temporal_audit_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_impulse_response_sets_p0_to_one():
    runner = load_runner()

    response = runner.normalize_impulse_response({0: 2.0, 1: 4.0, 4: 1.0})

    assert response[0] == 1.0
    assert response[1] == 2.0
    assert response[4] == 0.5


def test_piecewise_linear_interpolation_and_cumulative_exposure():
    runner = load_runner()
    response = {0: 1.0, 4: 3.0}

    assert runner.piecewise_linear_value(response, 2) == 2.0
    assert runner.cumulative_exposure(response, 4, mode="linear_sum") == 7.0
    assert runner.cumulative_exposure(response, 4, mode="terminal") == 3.0
    assert runner.cumulative_exposure(response, 4, mode="checkpoint_auc") == 8.0
    assert runner.cumulative_exposure(response, 4, mode="stepwise") == 4.0


def test_log_temporal_ratio_and_mft_identity():
    runner = load_runner()

    t = runner.log_temporal_ratio(8.0, 2.0)
    score = runner.mft_score(m=0.25, f=-0.5, t=t)

    assert abs(t - runner.log_ratio(8.0, 2.0)) < 1e-12
    assert abs(score - (0.25 - 0.5 + t)) < 1e-12


def test_outcome_sign_and_boundary_classification_are_reused():
    runner = load_runner()

    assert runner.preference_from_score(0.2) == "C"
    assert runner.preference_from_score(-0.2) == "R"
    assert runner.preference_from_score(0.01) == "B"
    assert runner.preference_from_y(0.2) == "C"
    assert runner.preference_from_y(-0.2) == "R"


def test_no_outcome_leakage_rejects_kl_columns_for_t_sources():
    runner = load_runner()

    clean = [{"unit_id": "u1", "functional_gain_R": "1.0", "functional_gain_C_match": "2.0"}]
    leaky = [{"unit_id": "u1", "Y": "0.1", "functional_gain_R": "1.0"}]

    assert runner.no_outcome_leakage_gate(clean, allowed_outcome_columns={"future_KL_R", "future_KL_C_match"}) == "PASS"
    assert runner.no_outcome_leakage_gate(leaky, allowed_outcome_columns={"future_KL_R", "future_KL_C_match"}) == "FAIL"


def test_unit_cluster_bootstrap_reports_unit_count():
    runner = load_runner()
    rows = [
        {"unit_id": "u1", "value": 1.0},
        {"unit_id": "u1", "value": 3.0},
        {"unit_id": "u2", "value": 5.0},
    ]

    boot = runner.unit_cluster_bootstrap(rows, lambda sample: sum(r["value"] for r in sample) / len(sample), reps=50, seed=9)

    assert boot["n_units"] == 2
    assert boot["n_rows"] == 3
    assert boot["ci_low"] <= boot["estimate"] <= boot["ci_high"]


def test_mf_reuse_requires_exact_previous_columns():
    runner = load_runner()
    row = {"architecture": "KDA", "unit_id": "u", "group_size": "16", "m": "-1", "f": "-2", "S_MF": "-3"}

    assert runner.mf_reuse_gate([row]) == "PASS"
    assert runner.mf_reuse_gate([{k: v for k, v in row.items() if k != "S_MF"}]) == "FAIL"
