import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_gdn_kda_unified_orientation_risk_audit_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gdn_kda_unified_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_log_ratio_and_mf_equivalence():
    runner = load_runner()

    row = runner.make_risk_row(
        architecture="X",
        unit_id="u1",
        group_size=128,
        history_length=4,
        m_r=4.0,
        m_c=2.0,
        f_r=3.0,
        f_c=6.0,
        kl_r=0.3,
        kl_c=0.1,
        provenance=["artifact"],
    )

    assert abs(row["m"] - runner.log_ratio(4.0, 2.0)) < 1e-12
    assert abs(row["f"] - runner.log_ratio(3.0, 6.0)) < 1e-12
    assert abs(row["S_MF"] - (row["m"] + row["f"])) < 1e-12
    assert abs(row["S_MF"] - runner.log_ratio(12.0, 12.0)) < 1e-12


def test_outcome_sign_convention_and_boundary_classification():
    runner = load_runner()

    assert runner.actual_preference_from_y(0.2, effect_tolerance=1e-6) == "C"
    assert runner.actual_preference_from_y(-0.2, effect_tolerance=1e-6) == "R"
    assert runner.actual_preference_from_y(1e-8, effect_tolerance=1e-6) == "B"
    assert runner.predicted_preference_from_score(0.2, score_tolerance=1e-6) == "C"
    assert runner.predicted_preference_from_score(-0.2, score_tolerance=1e-6) == "R"
    assert runner.predicted_preference_from_score(1e-8, score_tolerance=1e-6) == "B"


def test_missing_row_handling_excludes_incomplete_mf_rows():
    runner = load_runner()

    row = runner.make_missing_row("KDA", "u2", 64, "ALL", "missing F metric")

    assert row["complete_mf"] == "NO"
    assert row["S_MF"] == "MISSING"
    assert runner.complete_rows([row]) == []


def test_unit_cluster_bootstrap_resamples_units_not_rows():
    runner = load_runner()
    rows = [
        {"unit_id": "u1", "value": 1.0},
        {"unit_id": "u1", "value": 100.0},
        {"unit_id": "u2", "value": 3.0},
        {"unit_id": "u2", "value": 5.0},
    ]

    result = runner.unit_cluster_bootstrap(
        rows,
        lambda sample: sum(r["value"] for r in sample) / len(sample),
        reps=100,
        seed=7,
    )

    assert result["n_units"] == 2
    assert result["n_rows"] == 4
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_orientation_accuracy_is_boundary_aware():
    runner = load_runner()
    rows = [
        {"actual_preference": "C", "m_pred": "C", "f_pred": "R", "mf_pred": "C"},
        {"actual_preference": "R", "m_pred": "C", "f_pred": "R", "mf_pred": "R"},
        {"actual_preference": "B", "m_pred": "R", "f_pred": "B", "mf_pred": "B"},
    ]

    summary = runner.orientation_accuracy(rows)

    assert summary["M_ONLY"] == 1 / 3
    assert summary["F_ONLY"] == 2 / 3
    assert summary["MF"] == 1.0


def test_panel_join_marks_available_and_missing_groups():
    runner = load_runner()
    magnitude = {"u1": {128: {"m_r": 1.0, "m_c": 2.0, "Y": -0.1}}}
    functional = {"u1": {128: {"f_r": 4.0, "f_c": 8.0}}}

    rows = runner.join_architecture_panel("KDA", magnitude, functional, [128, 64])

    assert rows[0]["complete_mf"] == "YES"
    assert rows[0]["actual_preference"] == "R"
    assert rows[1]["complete_mf"] == "NO"
    assert rows[1]["missing_reason"] == "missing magnitude and functional metrics"
