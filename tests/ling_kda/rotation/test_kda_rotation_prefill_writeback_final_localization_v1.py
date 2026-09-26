import importlib.util
import inspect
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_rotation_prefill_writeback_final_localization_v1.py"
SPEC = importlib.util.spec_from_file_location("prefill_writeback", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def synthetic_rows(noq_full=0.02, noq_prefill=0.03, noq_decode=0.10):
    return [{
        "unit_id": f"u{i}",
        "auc_N_HISTORY": 0.01,
        "auc_R_LIVE_FULL": 0.11,
        "auc_R_NOQ_FULL": noq_full,
        "auc_R_NOQ_PREFILL_ONLY": noq_prefill,
        "auc_R_NOQ_DECODE_ONLY": noq_decode,
    } for i in range(18)]


def test_01_task_and_result_path_are_new():
    assert M.TASK == "KDA_ROTATION_PREFILL_WRITEBACK_FINAL_LOCALIZATION_V1"
    assert M.RESULT_DIR != M.HISTORY_RESULTS


def test_02_exactly_five_main_conditions():
    assert M.CONDITIONS == (
        "N_HISTORY", "R_LIVE_FULL", "R_NOQ_FULL",
        "R_NOQ_PREFILL_ONLY", "R_NOQ_DECODE_ONLY")


def test_03_native_policy_is_int8_for_both_phases():
    assert M.policy("N_HISTORY") == ("native", True, True)


def test_04_live_rotated_policy_is_int8_for_both_phases():
    assert M.policy("R_LIVE_FULL") == ("rotated", True, True)


def test_05_noq_full_uses_fp_state_for_both_phases():
    assert M.policy("R_NOQ_FULL") == ("rotated", False, False)


def test_06_noq_prefill_only_switches_at_decode():
    assert M.policy("R_NOQ_PREFILL_ONLY") == ("rotated", False, True)


def test_07_noq_decode_only_switches_at_decode():
    assert M.policy("R_NOQ_DECODE_ONLY") == ("rotated", True, False)


def test_08_switch_rebuilds_live_plan():
    source = inspect.getsource(M.set_quantization)
    assert 'H.plan(item["name"], item["basis"], bool(enabled))' in source


def test_09_no_fake_quantizer_or_manual_float_cache():
    source = inspect.getsource(M.prepare_histories)
    assert "tiny" not in source and "fake" not in source and "float32 custom" not in source


def test_10_existing_fp_state_path_is_used():
    source = inspect.getsource(M.prepare_histories)
    assert "H.prefill_branch" in source and "H.advance" in source


def test_11_value_rotation_is_preserved_for_all_noq_conditions():
    for name in ("R_NOQ_FULL", "R_NOQ_PREFILL_ONLY", "R_NOQ_DECODE_ONLY"):
        assert M.policy(name)[0] == "rotated"


def test_12_all_conditions_use_same_native_future():
    source = inspect.getsource(M.run_unit)
    assert "H.evaluate_native_future" in source


def test_13_t0_conversion_uses_validated_history_runner():
    source = inspect.getsource(M.run_unit)
    assert "H.evaluate_native_future" in source
    assert "inverse_stack" not in source


def test_14_boundary_has_runtime_assertions():
    source = inspect.getsource(M.prepare_histories)
    assert 'assert item["quantized"] is prefill_q' in source
    assert 'assert items[name]["quantized"] is policy(name)[2]' in source


def test_15_history_composition_is_packed_prefill():
    row = M.history_composition({"unit_id": "x", "t0": 64}, 152)
    assert row["packed_prefill"] and not row["chunked_prefill"]
    assert row["prefill_token_count"] == 152


def test_16_l64_boundary_is_exact():
    row = M.history_composition({"unit_id": "x", "t0": 128}, 100)
    assert row["L64_START_POSITION"] == 164
    assert row["L64_END_POSITION"] == 227


def test_17_full_minus_l64_components_are_explicit():
    row = M.history_composition({"unit_id": "x", "t0": 128}, 100)
    assert row["EXTRA_HISTORY_FULL_MINUS_L64"] == {
        "packed_prefill_tokens": 100, "early_decode_tokens": 64}


def test_18_prefill_quantization_frequency_is_endpoint_only():
    row = M.history_composition({"unit_id": "x", "t0": 64}, 100)
    assert row["prefill_quantization_frequency"] == "once at packed-prefill endpoint"


def test_19_unitwise_auc_aggregation():
    rows = [{"unit_id": unit, "condition": condition, "future_kl": horizon}
            for unit in ("a", "b") for condition in M.CONDITIONS for horizon in (1, 2, 3)]
    out = M.aggregate_auc(rows)
    assert len(out) == 2 and out[0]["auc_N_HISTORY"] == 2


def test_20_closures_are_calculated_per_unit():
    rows = M.closure_rows(synthetic_rows())
    assert len(rows) == 18
    assert rows[0]["full_noq_closure"] == pytest.approx(0.9)
    assert rows[0]["prefill_noq_closure"] == pytest.approx(0.8)
    assert rows[0]["decode_noq_closure"] == pytest.approx(0.1)


def test_21_unstable_denominators_are_retained():
    rows = synthetic_rows(); rows[0]["auc_R_LIVE_FULL"] = 0.01
    derived = M.closure_rows(rows)
    assert len(derived) == 18 and derived[0]["unstable_denominator"]


def test_22_strong_support_uses_bootstrap_lower_bound():
    assert M.support({"bootstrap_ci_low": 0.3, "positive": 18, "n": 18, "paired_median": 0.8}) == "STRONG"
    assert M.support({"bootstrap_ci_low": -0.1, "positive": 18, "n": 18, "paired_median": 0.8}) == "PARTIAL"


def test_23_median_bootstrap_is_inherited_from_verified_runner():
    stat = M.effect([0.5] * 18, "X")
    assert stat["paired_median"] == 0.5 and stat["bootstrap_ci_low"] == 0.5


def test_24_horizon_rows_are_not_bootstrap_units():
    source = inspect.getsource(M.summarize)
    assert "closure_rows(unit_rows)" in source
    assert "prefill_diagnostics" not in inspect.getsource(M.effect)


def test_25_old_formal_artifacts_are_read_only():
    source = inspect.getsource(M)
    assert "HISTORY_RESULTS" in source
    assert "write_rows(HISTORY_RESULTS" not in source
    assert "save_json(HISTORY_RESULTS" not in source


def test_26_validation_requires_all_formal_rows():
    source = inspect.getsource(M.validate)
    assert "EXPECTED_UNITS * PRIMARY_HORIZON * (len(CONDITIONS) + 1)" in source
