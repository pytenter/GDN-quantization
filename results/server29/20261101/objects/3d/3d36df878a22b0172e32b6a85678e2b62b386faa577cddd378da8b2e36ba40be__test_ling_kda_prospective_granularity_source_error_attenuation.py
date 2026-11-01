import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_prospective_granularity_source_error_attenuation_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("source_attenuation", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_rows(panel="MATCHED_FP_CARRIER", state_hash="same"):
    rows = []
    for unit in ["u0", "u1", "u2"]:
        for group, c_rel, r_rel in [(128, 0.08, 0.02), (64, 0.06, 0.025), (32, 0.04, 0.03), (16, 0.03, 0.028)]:
            for step in [1, 4, 8, 16, 32, 64]:
                base = {
                    "unit_id": unit,
                    "prompt_id": unit,
                    "sample_id": unit,
                    "layer": "ALL_KDA",
                    "head": "ALL",
                    "t0": 64,
                    "panel": panel,
                    "group_size": group,
                    "history_or_step": step,
                    "recurrent_step": step,
                    "state_norm": 10.0,
                    "source_abs_norm": r_rel * 10.0,
                    "source_rel_norm": r_rel,
                    "source_mse": r_rel * r_rel,
                    "source_max_abs": r_rel,
                    "scale_mean": 1.0,
                    "scale_std": 0.1,
                    "scale_max": 2.0,
                    "scale_min": 0.1,
                    "state_hash": f"{unit}-{step}-{state_hash}",
                }
                rows.append({**base, "orientation": "R", "source_error_hash": f"r-{unit}-{group}-{step}"})
                rows.append({**base, "orientation": "C", "source_abs_norm": c_rel * 10.0, "source_rel_norm": c_rel, "source_mse": c_rel * c_rel, "source_max_abs": c_rel, "source_error_hash": f"c-{unit}-{group}-{step}"})
    return rows


def test_panel_a_event_pairing_requires_matched_carrier_and_computes_log_ratio():
    runner = load_runner()

    paired = runner.pair_source_events(event_rows())

    assert paired["MATCHED_CARRIER_IDENTITY_GATE"] == "PASS"
    assert len(paired["paired_rows"]) == 72
    first = next(row for row in paired["paired_rows"] if row["group_size"] == 128)
    assert first["source_log_ratio"] == pytest.approx(runner.math.log((0.08 + runner.EPS) / (0.02 + runner.EPS)))
    assert first["source_c_over_r_ratio"] == pytest.approx(4.0)
    assert first["source_rel_gap_C_minus_R"] == pytest.approx(0.06)


def test_panel_a_mismatched_carrier_fails_gate():
    runner = load_runner()
    rows = event_rows()
    for row in rows:
        if row["orientation"] == "C":
            row["state_hash"] += "-changed"

    paired = runner.pair_source_events(rows)

    assert paired["MATCHED_CARRIER_IDENTITY_GATE"] == "FAIL"


def test_unit_level_source_asymmetry_and_contrasts_are_unit_bootstrapped():
    runner = load_runner()
    paired = runner.pair_source_events(event_rows())["paired_rows"]
    unit_rows = runner.compute_unit_level_source_asymmetry(paired)
    contrasts = runner.compute_source_attenuation_contrasts(unit_rows, "MATCHED_FP_CARRIER", bootstrap_reps=200, bootstrap_seed=11)

    assert len(unit_rows) == 3
    assert unit_rows[0]["PANEL_A_A_g128"] > unit_rows[0]["PANEL_A_A_g16"]
    assert contrasts["PANEL_A_D128_TO_D16"]["median_contrast"] > 0
    assert contrasts["BOOTSTRAP_RESAMPLING_LEVEL"] == "canonical_unit"


def test_source_kl_join_and_association_use_unit_level_rows():
    runner = load_runner()
    paired = runner.pair_source_events(event_rows())["paired_rows"]
    source_units = runner.compute_unit_level_source_asymmetry(paired)
    kl_units = [
        {"unit_id": row["unit_id"], "delta_g128": 0.10, "delta_g64": 0.07, "delta_g32": 0.04, "delta_g16": 0.01}
        for row in source_units
    ]

    joined = runner.join_source_with_kl(source_units, kl_units)
    assoc = runner.compute_source_functional_association(joined, bootstrap_reps=200, bootstrap_seed=13)

    assert joined["SOURCE_KL_JOIN_GATE"] == "PASS"
    assert len(joined["unit_rows"]) == 3
    assert "source_kl_attenuation_spearman" in assoc
    assert "cell_level_source_vs_kl_spearman" in assoc


def test_classification_and_next_task_for_strong_source_support():
    runner = load_runner()
    contrasts = {
        "PANEL_A_D128_TO_D64": {"median_contrast": 0.1, "bootstrap_ci": [0.01, 0.2]},
        "PANEL_A_D64_TO_D32": {"median_contrast": 0.1, "bootstrap_ci": [-0.01, 0.2]},
        "PANEL_A_D32_TO_D16": {"median_contrast": 0.1, "bootstrap_ci": [0.01, 0.2]},
        "PANEL_A_D128_TO_D16": {"median_contrast": 0.3, "bootstrap_ci": [0.1, 0.5]},
    }
    robustness = {
        "SOURCE_ATTENUATION_LOO_UNIT_SIGN_STABLE": "PASS",
        "SOURCE_ATTENUATION_LOO_LAYER_SIGN_STABLE": "PASS",
        "SOURCE_ATTENUATION_LOO_PROMPT_SIGN_STABLE": "PASS",
        "NO_SINGLE_SUBGROUP_DRIVES_RESULT": "PASS",
    }

    out = runner.classify_source_attenuation(contrasts, robustness, "BROAD_ACROSS_HISTORY")

    assert out["SOURCE_ERROR_ATTENUATION_STATUS"] == "STRONG_SUPPORT"
    assert out["NEXT_RECOMMENDED_TASK"] == "LING_KDA_GRANULARITY_MAGNITUDE_EQUALIZATION_CAUSAL_V1"
