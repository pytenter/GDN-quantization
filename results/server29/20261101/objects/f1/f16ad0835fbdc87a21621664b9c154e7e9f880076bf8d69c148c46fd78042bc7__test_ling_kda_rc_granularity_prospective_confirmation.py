import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_rc_granularity_prospective_confirmation_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_kda_rc_prospective", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_g64_manifest():
    units = []
    for prompt in range(18):
        pid = str(100 + prompt)
        for t0 in [64, 128, 192, 256]:
            units.append({
                "unit_id": f"{pid}|{t0}",
                "problem_id": pid,
                "prompt_id": pid,
                "sample_id": pid,
                "problem_index": prompt,
                "t0": t0,
                "layer": "ALL_KDA",
                "head": "ALL",
                "token_sequence_sha256": f"tok-{pid}",
            })
    return {"sha256": "g64-sha", "N_NEW_INDEPENDENT_UNITS": 72, "units": units}


def rows_for_group(group_size, unit_ids, deltas_by_history):
    rows = []
    for unit_id in unit_ids:
        pid = unit_id.split("|")[0]
        for history, delta in zip([1, 4, 8, 16, 32, 64], deltas_by_history):
            rows.append({"unit_id": unit_id, "problem_id": pid, "layer": "ALL_KDA", "head": "ALL", "group_size": group_size, "history_length": history, "config": f"R{group_size}", "future_kl_auc": 1.0})
            rows.append({"unit_id": unit_id, "problem_id": pid, "layer": "ALL_KDA", "head": "ALL", "group_size": group_size, "history_length": history, "config": f"C{group_size}", "future_kl_auc": 1.0 + delta})
    return rows


def test_primary_subset_reuses_exact_g64_manifest_order_without_rerunning_g64():
    runner = load_runner()
    manifest = fake_g64_manifest()

    audit, primary = runner.freeze_primary_subset_from_g64_manifest(manifest)

    assert runner.CONFIGS == ["R128", "C128", "R32", "C32", "R16", "C16"]
    assert runner.GROUP_SIZES == [128, 32, 16]
    assert runner.HISTORY_LENGTHS == [1, 4, 8, 16, 32, 64]
    assert audit["PROSPECTIVE_MANIFEST_REUSE_GATE"] == "PASS"
    assert audit["NO_OUTCOME_BASED_SUBSET_SELECTION_GATE"] == "PASS"
    assert len(primary["units"]) == 36
    assert [u["unit_id"] for u in primary["units"]] == [u["unit_id"] for u in manifest["units"][:36]]
    assert "R64" not in runner.CONFIGS and "C64" not in runner.CONFIGS


def test_expected_primary_and_expansion_run_counts_are_fixed():
    runner = load_runner()

    assert runner.expected_primary_runs() == 1296
    assert runner.expected_expansion_runs(["128"]) == 432
    assert runner.expected_expansion_runs(["128", "32", "16"]) == 1296


def test_analysis_is_per_group_unit_level_and_preserves_history_summaries():
    runner = load_runner()
    unit_ids = ["u0", "u1", "u2"]
    rows = []
    rows += rows_for_group(128, unit_ids, [0.10, 0.11, 0.12, 0.09, 0.08, 0.10])
    rows += rows_for_group(32, unit_ids, [0.03, 0.04, 0.05, 0.03, 0.04, 0.05])
    rows += rows_for_group(16, unit_ids, [-0.05, -0.04, -0.03, -0.02, -0.01, -0.02])

    analysis = runner.analyze_granularity_rows(rows, bootstrap_reps=200, bootstrap_seed=7)

    assert len(analysis["per_unit_per_history_effects"]) == 54
    assert len(analysis["history_level_summary"]) == 18
    assert analysis["group_level_summary"]["128"]["N_UNITS"] == 3
    assert analysis["group_level_summary"]["128"]["PRIMARY_EFFECT_DEFINITION"] == "median over units of mean_L(KL_Cg - KL_Rg)"
    assert analysis["group_level_summary"]["128"]["PRIMARY_EFFECT"] == pytest.approx(0.10)
    assert analysis["bootstrap_results"]["BOOTSTRAP_RESAMPLING_LEVEL"] == "canonical_unit"


def test_adaptive_decisions_expand_only_groups_with_primary_ci_crossing_zero():
    runner = load_runner()
    group_summary = {
        "128": {"BOOTSTRAP_CI": [0.01, 0.03]},
        "32": {"BOOTSTRAP_CI": [-0.01, 0.02]},
        "16": {"BOOTSTRAP_CI": [-0.03, -0.01]},
    }

    decisions = runner.decide_adaptive_expansions(group_summary)

    assert decisions["G128_EXPANDED_TO_72"] == "NO"
    assert decisions["G32_EXPANDED_TO_72"] == "YES"
    assert decisions["G16_EXPANDED_TO_72"] == "NO"
    assert decisions["N_ADDITIONAL_EXPANSION_RUNS"] == 432


def test_classifiers_cover_group_and_cross_granularity_cases():
    runner = load_runner()
    gates = runner.pass_gate_summary()
    robust = {
        "PRIMARY_EFFECT": 0.05,
        "BOOTSTRAP_CI": [0.01, 0.08],
        "R_PREFERRED_HISTORY_COUNT": 5,
        "C_PREFERRED_HISTORY_COUNT": 1,
        "LEAVE_ONE_UNIT_OUT_SIGN_STABLE": "PASS",
        "LEAVE_ONE_LAYER_OUT_SIGN_STABLE": "PASS",
        "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE": "PASS",
        "NO_SINGLE_SUBGROUP_DRIVES_RESULT": "PASS",
    }
    boundary = dict(robust, BOOTSTRAP_CI=[-0.01, 0.08])
    column = dict(robust, PRIMARY_EFFECT=-0.05, BOOTSTRAP_CI=[-0.08, -0.01], R_PREFERRED_HISTORY_COUNT=1, C_PREFERRED_HISTORY_COUNT=5)

    assert runner.classify_group(robust, gates) == "ROBUST_R_PREFERENCE"
    assert runner.classify_group(boundary, gates) == "ORIENTATION_BOUNDARY"
    assert runner.classify_group(column, gates) == "ROBUST_C_PREFERENCE"

    bad_gates = runner.pass_gate_summary()
    bad_gates["PIPELINE_REPEATABILITY_GATE"] = "FAIL"
    assert runner.classify_group(robust, bad_gates) == "INVALID"

    full = {"128": "ROBUST_R_PREFERENCE", "64": "ROBUST_R_PREFERENCE", "32": "ROBUST_R_PREFERENCE", "16": "ROBUST_R_PREFERENCE"}
    assert runner.classify_cross_granularity(full)["LING_KDA_PROSPECTIVE_GRANULARITY_PATTERN"] == "R/R/R/R"
    mixed = dict(full, **{"32": "ORIENTATION_BOUNDARY"})
    assert runner.classify_cross_granularity(mixed)["LING_KDA_ORIENTATION_GRANULARITY_RESULT"] == "PARTIALLY_STABLE_WITH_BOUNDARY"
    changed = dict(full, **{"16": "ROBUST_C_PREFERENCE"})
    assert runner.classify_cross_granularity(changed)["LING_KDA_ORIENTATION_GRANULARITY_RESULT"] == "GRANULARITY_DEPENDENT_ORIENTATION_CHANGE"


def test_final_console_summary_contains_required_keys_and_stop_rule():
    runner = load_runner()
    summary = {key: "PASS" for key in runner.REQUIRED_CONSOLE_KEYS}
    summary.update({"TASK": runner.TASK, "METHOD_DESIGN_READY": "NO", "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW"})

    text = runner.render_console_summary(summary)

    for key in runner.REQUIRED_CONSOLE_KEYS:
        assert f"{key} =" in text
    assert "NO_NEXT_EXPERIMENT_LAUNCHED = YES" in text
    assert text.rstrip().endswith("STOP.")
