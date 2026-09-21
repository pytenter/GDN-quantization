import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_value_hadamard_postfix_scientific_rebaseline_v1.py"
SPEC = importlib.util.spec_from_file_location("postfix_rebaseline", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_01_task_and_output_are_new():
    assert M.TASK == "KDA_VALUE_HADAMARD_POSTFIX_SCIENTIFIC_REBASELINE_V1"
    assert M.RESULT_DIR != M.HISTORICAL_RESULTS


def test_02_exact_three_main_conditions():
    assert M.CONDITIONS == ("NATIVE_R128", "BUGGY_HADAMARD_R128", "CORRECTED_HADAMARD_R128")


def test_03_corrected_and_buggy_differ_only_at_endpoint_flag():
    buggy = M.BRANCH_SPECS["BUGGY_HADAMARD_R128"]
    fixed = M.BRANCH_SPECS["CORRECTED_HADAMARD_R128"]
    assert buggy[1] == fixed[1] == "rotated"
    assert buggy[2] is True and fixed[2] is False


def test_04_corrected_prefill_cache_is_direct():
    source = inspect.getsource(M.run_formal_unit)
    assert "apply_endpoint_rotation=extra_rotation" in source
    assert '"CORRECTED_HADAMARD_R128": ("RS_FIXED", "rotated", False)' in inspect.getsource(M)


def test_05_kernel_return_state_is_rotated():
    assert M.code_audit()["PREFILL_KERNEL_RETURN_STATE_BASIS"] == "ROTATED_VALUE"


def test_06_first_decode_keeps_rotated_basis():
    source = inspect.getsource(M.run_formal_unit)
    assert "H.to_native" not in source
    assert "item[\"basis\"]" in source


def test_07_core_output_maps_back_once():
    assert M.corrected_semantics()["core_output_mapback_count"] == 1


def test_08_rmsnorm_and_gate_receive_native_representation():
    semantics = M.corrected_semantics()
    assert semantics["rmsnorm_input_basis"] == "NATIVE_VALUE"
    assert semantics["dynamic_gate_downstream_basis"] == "NATIVE_VALUE"


def test_09_quantizer_is_int8_r128_v_axis():
    q = M.corrected_semantics()["quantizer"]
    assert (q["CONFIG"], q["GROUP_AXIS"], q["GROUP_SIZE"]) == ("INT8_R128", "V", 128)


def test_10_group_scale_shape_is_bhK1():
    assert M.corrected_semantics()["quantizer"]["SCALE_SHAPE"] == "[B,H,K,1]"


def test_11_hadamard_is_formal_seed_zero_rht128():
    semantics = M.corrected_semantics()
    assert "RHT128" in semantics["rotation"] and "seed 0" in semantics["rotation"]


def test_12_model_weights_are_hashed_before_and_after():
    source = inspect.getsource(M.run_formal)
    assert "weight_hash_before" in source and "MODEL_WEIGHTS_UNCHANGED" in source


def test_13_teacher_forced_fp_panel_is_3_by_128():
    assert (M.FP_PROMPTS, M.FP_TOKENS) == (3, 128)


def test_14_future_protocol_uses_identical_tokens_and_horizons():
    source = inspect.getsource(M.run_formal_unit)
    assert "for condition, item in items.items()" in source
    assert "tokens[int(unit[\"t0\"]) + h - 1]" in source


def test_15_quant_metrics_include_codes_saturation_and_clipping():
    source = inspect.getsource(M.quantize_cache_with_rows)
    for name in ("saturation_fraction", "clipping_fraction", "code_mean_abs", "code_zero_fraction"):
        assert name in source


def test_16_quantizer_writes_qdq_once_per_boundary():
    source = inspect.getsource(M.quantize_cache_with_rows)
    assert "fake_quant_ling_state" in source
    assert "state.copy_(qdq" in source


def test_17_statistics_aggregate_horizons_within_unit_first():
    source = inspect.getsource(M.run_formal_unit)
    assert "BASE.mean" in source
    assert "future_kl" in source


def test_18_median_bootstrap_matches_median_estimand():
    values = [1.0, 2.0, 100.0]
    item = M.effect(values, "x")
    assert item["paired_median"] == 2.0
    assert item["estimand"] == "paired canonical-unit median"


def test_19_nonfinite_validation_is_explicit():
    source = inspect.getsource(M.validate)
    assert "future_finite" in source and "static_finite" in source


def test_20_previous_artifacts_are_not_written():
    source = inspect.getsource(M)
    assert "save_json(HISTORICAL_RESULTS" not in source
    assert "write_rows(HISTORICAL_RESULTS" not in source


def test_21_static_metric_identity():
    stat = M.tensor_metrics(torch.ones(8), torch.ones(8))
    assert stat["relative_l2"] == 0.0 and stat["max_abs"] == 0.0


def test_22_stage0_proves_old_and_corrected_paths():
    audit = M.code_audit()
    assert audit["STAGE0_CODE_AUDIT"] == "PASS"
    assert audit["CORRECTED_PREFILL_ENDPOINT_EXTRA_STATE_ROTATION"] == "NO"
    assert audit["BUGGY_PREFILL_ENDPOINT_EXTRA_STATE_ROTATION"] == "YES"


def test_23_formal_is_gated_by_fp_parity():
    assert "Stage 1 did not authorize formal run" in inspect.getsource(M.run_formal)


def test_24_stop_rule_is_complete_and_conservative():
    source = inspect.getsource(M.finalize)
    for term in ("CORRECTED_FP_EQUIVALENCE", "STATIC_QUANTIZATION_GAIN_PRESERVED",
                 "CORRECTED_HADAMARD_LONG_HORIZON_HARM",
                 "HISTORICAL_SEVERE_FAILURE_ATTRIBUTABLE_TO_DOUBLE_ROTATION"):
        assert term in source


def test_25_required_reinterpretation_categories_exist():
    source = inspect.getsource(M.write_reinterpretation)
    assert "Still Valid" in source
    assert "Valid Only As Bug-Propagation Evidence" in source
    assert "Invalid As Intrinsic KDA Rotation Mechanism" in source
