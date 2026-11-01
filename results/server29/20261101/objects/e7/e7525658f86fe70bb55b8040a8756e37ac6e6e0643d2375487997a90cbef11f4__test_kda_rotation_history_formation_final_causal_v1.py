import importlib.util
import inspect
import math
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_rotation_history_formation_final_causal_v1.py"
SPEC = importlib.util.spec_from_file_location("history_formation", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def orthogonal(n=4):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(17)))
    return q


def stage_a_rows():
    rows = []
    for i in range(18):
        row = {"unit_id": f"u{i}", "auc_L0": 0.01, "auc_L1": 0.011, "auc_L2": 0.012,
               "auc_L4": 0.014, "auc_L8": 0.018, "auc_L16": 0.025, "auc_L32": 0.04,
               "auc_L64": 0.07, "auc_FULL": 0.12}
        rows.append(row)
    return rows


def stage_b_rows():
    return [{"unit_id": f"u{i}", "auc_N_HISTORY": 0.01, "auc_R_LIVE_HISTORY": 0.11,
             "auc_R_FP32_TRANSITION_HISTORY": 0.03} for i in range(18)]


def stage_c_rows():
    return [{"unit_id": f"u{i}", "auc_N_HISTORY": 0.01, "auc_R_LIVE_HISTORY": 0.11,
             "auc_R_NO_STATE_QUANT_HISTORY": 0.04} for i in range(18)]


def test_01_task_and_result_path_are_new():
    assert M.TASK == "KDA_ROTATION_HISTORY_FORMATION_FINAL_CAUSAL_V1"
    assert M.RESULT_DIR != M.PREVIOUS_RESULTS


def test_02_timeline_begins_from_empty_cache():
    row = M.timeline_row({"unit_id": "x", "t0": 64}, 100)
    assert row["common_start_token"] == 0 and row["state_cache_initialization"] == "model empty cache at common start"


def test_03_timeline_includes_prefill_and_decode():
    row = M.timeline_row({"unit_id": "x", "t0": 64}, 100)
    assert row["history_includes_prefill"] and row["history_includes_decode"]
    assert row["number_of_history_transitions"] == 164


def test_04_l0_is_native_history_endpoint():
    source = inspect.getsource(M.prepare_exposure_histories)
    assert 'conditions = {"L0": native}' in source


def test_05_full_is_live_rotated_endpoint():
    source = inspect.getsource(M.prepare_exposure_histories)
    assert 'conditions["FULL"] = full' in source and '"R_FULL", "rotated", True' in source


def test_06_switch_occurs_before_intended_token():
    source = inspect.getsource(M.prepare_exposure_histories)
    assert "t == t0 - dose" in source and "to_rotated(item" in source


def test_07_no_boundary_only_qr_operator():
    source = inspect.getsource(M.prepare_exposure_histories)
    assert "quantize_operator" not in source and "Q_R" not in source


def test_08_t0_conversion_is_inverse_value_rotation():
    source = inspect.getsource(M.to_native)
    assert "inverse_stack" in source


def test_09_all_conditions_use_native_future():
    source = inspect.getsource(M.evaluate_native_future)
    assert 'plan(item["name"], "native", True)' in source


def test_10_exact_step_matches_kda_einsum():
    s = torch.randn(1, 2, 3, 4); a = torch.randn(1, 2, 3, 3); b = torch.randn(1, 2, 3, 4)
    expected = torch.einsum("...kl,...lv->...kv", a.double(), s.double()) + b.double()
    assert torch.equal(M.exact_step(s, a, b), expected)


def test_11_exact_step_empty_state_is_b():
    b = torch.randn(1, 2, 3, 4)
    assert torch.equal(M.exact_step(None, torch.empty(0), b), b.double())


def test_12_fp32_history_changes_only_probe_transition():
    source = inspect.getsource(M.HistoryFormationProbe._wrap)
    assert "fp32_rotated_sequence" in source and "precision_intervention" in source


def test_13_fp32_int8_condition_keeps_original_quantizer():
    specs = dict((x[0], x[2]) for x in M.phase_specs("stageB"))
    assert specs["R_FP32_TRANSITION_HISTORY"] is True


def test_14_noq_uses_nonquantized_cache_path():
    specs = dict((x[0], x[2]) for x in M.phase_specs("stageC"))
    assert specs["R_NO_STATE_QUANT_HISTORY"] is False


def test_15_quantizer_semantics_unchanged():
    q = M.COMM.quantizer_semantics()
    assert (q["CONFIG"], q["GROUP_SIZE"], q["QMIN"], q["QMAX"]) == ("INT8_R128", 128, -127, 127)


def test_16_no_native_donor_or_driver_transplant():
    for phase in ("stageB", "stageC", "stageD"):
        for name, basis, quantized in M.phase_specs(phase):
            p = M.plan(name, basis, quantized)
            assert p.state_source is None and p.m_source is None and p.b_source is None


def test_17_unitwise_auc_aggregation():
    rows = [{"unit_id": u, "condition": c, "future_kl": h} for u in ("a", "b")
            for c in ("L0", "FULL") for h in (1, 2, 3)]
    out = M.aggregate_auc(rows, ("L0", "FULL"))
    assert len(out) == 2 and out[0]["auc_L0"] == 2


def test_18_median_bootstrap_uses_unit_values():
    assert M.median_bootstrap_ci([0] * 17 + [1000], n=500, seed=3) == [0.0, 0.0]


def test_19_exposure_summary_preserves_all_units():
    conditions = ("L0", "L1", "L2", "L4", "L8", "L16", "L32", "L64", "FULL")
    units, summary = M.summarize_stage_a(stage_a_rows(), conditions)
    assert len(units) == 18 and summary["effects"]["FULL"]["n"] == 18


def test_20_exposure_status_detects_supported_dose():
    conditions = ("L0", "L1", "L2", "L4", "L8", "L16", "L32", "L64", "FULL")
    _, summary = M.summarize_stage_a(stage_a_rows(), conditions)
    assert summary["ROTATED_HISTORY_EXPOSURE_STATUS"] == "STRONG_SUPPORT"


def test_21_fp32_closure_is_paired_per_unit():
    units, summary = M.summarize_stage_b(stage_b_rows())
    assert len(units) == 18 and summary["FP32_HISTORY_CLOSURE"]["paired_median"] == pytest.approx(0.8)


def test_22_unstable_denominators_are_retained():
    rows = stage_b_rows(); rows[0] = {**rows[0], "auc_R_LIVE_HISTORY": 0.01}
    units, summary = M.summarize_stage_b(rows)
    assert len(units) == 18 and "u0" in summary["unstable_units"]


def test_23_noq_closure_is_paired_per_unit():
    units, summary = M.summarize_stage_c(stage_c_rows())
    assert len(units) == 18 and summary["NOQ_HISTORY_CLOSURE"]["paired_median"] == pytest.approx(0.7)


def test_24_interaction_uses_consistent_encoding():
    b, _ = M.summarize_stage_b(stage_b_rows()); c, _ = M.summarize_stage_c(stage_c_rows())
    x = [{"unit_id": f"u{i}", "auc_R_FP32_NOQ_HISTORY": 0.015} for i in range(18)]
    rows, summary = M.summarize_stage_d(b, c, x)
    assert len(rows) == 18 and summary["factor_encoding"].startswith("I=LIVE_INT8")


def test_25_no_nan_in_synthetic_summaries():
    _, b = M.summarize_stage_b(stage_b_rows()); _, c = M.summarize_stage_c(stage_c_rows())
    assert all(math.isfinite(x) for x in (b["FP32_HISTORY_CLOSURE"]["paired_median"], c["NOQ_HISTORY_CLOSURE"]["paired_median"]))


def test_26_old_formal_artifacts_are_read_only_references():
    source = inspect.getsource(M)
    assert "PREVIOUS_RESULTS" in source
    assert "write_rows(PREVIOUS_RESULTS" not in source and "save_json(PREVIOUS_RESULTS" not in source
