import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_layer_conditioned_local_functional_geometry_audit_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_layer_audit", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layer_classification_requires_ci_excluding_zero_and_enough_n():
    runner = load_runner()

    assert runner.classify_layer_effect(0.2, [0.1, 0.3], 8) == "C_SENSITIVE"
    assert runner.classify_layer_effect(-0.2, [-0.3, -0.1], 8) == "R_SENSITIVE"
    assert runner.classify_layer_effect(0.002, [-0.01, 0.01], 8) == "NEUTRAL_OR_WEAK"
    assert runner.classify_layer_effect(0.2, [0.1, 0.3], 2) == "INSUFFICIENT_N"


def test_heterogeneity_distinguishes_strength_from_sign_reversal():
    runner = load_runner()

    strength_rows = [
        {"layer_classification": "C_SENSITIVE"},
        {"layer_classification": "C_SENSITIVE"},
        {"layer_classification": "NEUTRAL_OR_WEAK"},
    ]
    sign_rows = [
        {"layer_classification": "C_SENSITIVE"},
        {"layer_classification": "R_SENSITIVE"},
        {"layer_classification": "NEUTRAL_OR_WEAK"},
    ]

    assert runner.classify_layer_heterogeneity(strength_rows) == "STRENGTH_ONLY"
    assert runner.classify_layer_heterogeneity(sign_rows) == "SIGN_HETEROGENEITY"


def test_leave_one_layer_counts_sign_reversing_removals():
    runner = load_runner()
    rows = [
        {"layer_id": 0, "local_gap": 1.0},
        {"layer_id": 0, "local_gap": 1.0},
        {"layer_id": 1, "local_gap": -3.0},
        {"layer_id": 1, "local_gap": -3.0},
    ]

    loo, details = runner.leave_one_layer_out(rows, reps=100)

    assert details["N_SIGN_REVERSING_LAYER_REMOVALS"] == 1
    assert details["SIGN_REVERSING_LAYER_IDS"] == [1]
    assert {row["removed_layer"] for row in loo} == {0, 1}


def test_attenuation_classification_uses_layer_gap_difference():
    runner = load_runner()

    assert runner.classify_attenuation(0.4, [0.1, 0.7], [0.5, 0.3, 0.2, 0.1]) == "ATTENUATES"
    assert runner.classify_attenuation(-0.4, [-0.7, -0.1], [0.1, 0.2, 0.3, 0.5]) == "INCREASES"
    assert runner.classify_attenuation(0.01, [-0.05, 0.05], [0.2, 0.2, 0.2, 0.2]) == "STABLE"
    assert runner.classify_attenuation(0.2, [0.1, 0.3], [0.2, 0.5, 0.1, 0.3]) == "NON_MONOTONIC"


def test_earliest_stage_and_best_proxy_are_data_driven():
    runner = load_runner()
    stage_rows = [
        {"stage": "READOUT", "median_gap": 0.1, "bootstrap_CI_low": 0.02, "bootstrap_CI_high": 0.2},
        {"stage": "CORE_OUTPUT", "median_gap": 0.1, "bootstrap_CI_low": 0.02, "bootstrap_CI_high": 0.2},
    ]
    assoc_rows = [
        {"stage": "READOUT", "spearman": 0.2, "spearman_CI_low": -0.1, "spearman_CI_high": 0.5},
        {"stage": "CORE_OUTPUT", "spearman": -0.4, "spearman_CI_low": -0.7, "spearman_CI_high": -0.1},
    ]

    assert runner.earliest_local_divergence_stage(stage_rows) == "READOUT"
    best = runner.best_proxy_stage(assoc_rows)
    assert best["stage"] == "CORE_OUTPUT"
    assert best["spearman"] == -0.4
