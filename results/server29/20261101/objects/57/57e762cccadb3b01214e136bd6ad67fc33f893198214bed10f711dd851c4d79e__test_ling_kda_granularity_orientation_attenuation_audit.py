import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_granularity_orientation_attenuation_audit_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("attenuation_audit", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_rows(unit_ids, group_effects):
    rows = []
    histories = [1, 4, 8, 16, 32, 64]
    for unit_id in unit_ids:
        pid, t0 = unit_id.split("|")
        for group, delta in group_effects[unit_id].items():
            for history in histories:
                rows.append({
                    "unit_id": unit_id,
                    "prompt_id": pid,
                    "sample_id": pid,
                    "layer": "ALL_KDA",
                    "head": "ALL",
                    "t0": int(t0),
                    "group_size": group,
                    "history_length": history,
                    "config": f"R{group}",
                    "future_kl_auc": 1.0,
                })
                rows.append({
                    "unit_id": unit_id,
                    "prompt_id": pid,
                    "sample_id": pid,
                    "layer": "ALL_KDA",
                    "head": "ALL",
                    "t0": int(t0),
                    "group_size": group,
                    "history_length": history,
                    "config": f"C{group}",
                    "future_kl_auc": 1.0 + delta,
                })
    return rows


def test_reconstructs_common_effects_from_raw_kl_and_counts_rows():
    runner = load_runner()
    units = [{"unit_id": f"u{i}|64", "prompt_id": f"u{i}", "sample_id": f"u{i}", "layer": "ALL_KDA", "head": "ALL", "t0": 64} for i in range(36)]
    effects = {u["unit_id"]: {128: 0.08, 64: 0.06, 32: 0.04, 16: 0.01} for u in units}

    result = runner.build_common_effect_tables(units, fake_rows([u["unit_id"] for u in units], effects))

    assert result["COMMON_COHORT_GATE"] == "PASS"
    assert len(result["per_history"]) == 864
    assert len(result["unit_group"]) == 36
    assert result["unit_group"][0]["delta_g128"] == pytest.approx(0.08)
    assert result["unit_group"][0]["delta_g16"] == pytest.approx(0.01)


def test_detects_missing_common_cohort_data_without_imputation():
    runner = load_runner()
    units = [{"unit_id": "u0|64", "prompt_id": "u0", "sample_id": "u0", "layer": "ALL_KDA", "head": "ALL", "t0": 64}]
    rows = fake_rows(["u0|64"], {"u0|64": {128: 0.08, 64: 0.06, 32: 0.04, 16: 0.01}})
    rows = [r for r in rows if not (r["group_size"] == 16 and r["config"] == "C16" and r["history_length"] == 64)]

    result = runner.build_common_effect_tables(units, rows)

    assert result["COMMON_COHORT_GATE"] == "FAIL"
    assert result["missing_cells"]


def test_primary_contrasts_monotonicity_slopes_and_ratios():
    runner = load_runner()
    unit_rows = [
        {"unit_id": "a", "delta_g128": 0.08, "delta_g64": 0.06, "delta_g32": 0.04, "delta_g16": 0.01},
        {"unit_id": "b", "delta_g128": 0.07, "delta_g64": 0.05, "delta_g32": 0.03, "delta_g16": 0.02},
        {"unit_id": "c", "delta_g128": 0.06, "delta_g64": 0.07, "delta_g32": 0.02, "delta_g16": 0.00},
    ]

    contrasts = runner.compute_paired_contrasts(unit_rows, bootstrap_reps=200, bootstrap_seed=3)
    mono = runner.compute_monotonicity(unit_rows)
    slopes = runner.compute_log2_slopes(unit_rows, bootstrap_reps=200, bootstrap_seed=4)
    ratios = runner.compute_attenuation_ratios({"128": {"median_effect": 0.08}, "64": {"median_effect": 0.06}, "32": {"median_effect": 0.04}, "16": {"median_effect": 0.01}})

    assert contrasts["D128_TO_D16"]["median_contrast"] == pytest.approx(0.06)
    assert mono["N_DELTA128_GT_DELTA16"] == 3
    assert mono["N_STRICT_MONOTONIC_UNITS"] == 2
    assert slopes["positive_slope_unit_count"] == 3
    assert ratios["A64"] == pytest.approx(0.75)
    assert ratios["Reduction16"] == pytest.approx(0.875)


def test_classification_prefers_strong_support_for_positive_primary_ci_and_robustness():
    runner = load_runner()
    group_summary = {
        "128": {"median_effect": 0.08},
        "64": {"median_effect": 0.06},
        "32": {"median_effect": 0.04},
        "16": {"median_effect": 0.01},
    }
    contrasts = {
        "D128_TO_D64": {"median_contrast": 0.02, "bootstrap_ci": [0.001, 0.04]},
        "D64_TO_D32": {"median_contrast": 0.02, "bootstrap_ci": [-0.001, 0.04]},
        "D32_TO_D16": {"median_contrast": 0.03, "bootstrap_ci": [0.001, 0.05]},
        "D128_TO_D16": {"median_contrast": 0.07, "bootstrap_ci": [0.03, 0.10]},
    }
    robustness = {
        "LEAVE_ONE_UNIT_OUT_SIGN_STABLE": "PASS",
        "LEAVE_ONE_LAYER_OUT_SIGN_STABLE": "PASS",
        "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE": "PASS",
        "NO_SINGLE_SUBGROUP_DRIVES_RESULT": "PASS",
    }

    out = runner.classify_attenuation(group_summary, contrasts, robustness, "BROAD_ACROSS_HISTORY", "GLOBAL")

    assert out["ORIENTATION_ATTENUATION_STATUS"] == "STRONG_SUPPORT"
    assert out["FINAL_CLASSIFICATION"] == "KDA_ORIENTATION_ASYMMETRY_ATTENUATES_WITH_FINER_GRANULARITY"
    assert out["NEXT_RECOMMENDED_TASK"] == "LING_KDA_PROSPECTIVE_GRANULARITY_SOURCE_ERROR_ATTENUATION_V1"


def test_single_layer_leave_one_out_is_not_a_false_failure():
    runner = load_runner()
    unit_rows = [
        {"unit_id": f"u{i}", "prompt_id": f"p{i}", "layer": "ALL_KDA", "delta_g128": 0.08, "delta_g64": 0.06, "delta_g32": 0.04, "delta_g16": 0.01}
        for i in range(4)
    ]

    _rows, summary = runner.compute_robustness(unit_rows)

    assert summary["LEAVE_ONE_LAYER_OUT_SIGN_STABLE"] == "PASS"
