import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_rotation_prefill_full_path_first_divergence_v1.py"
SPEC = importlib.util.spec_from_file_location("first_divergence", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_01_task_and_output_are_new():
    assert M.TASK == "KDA_ROTATION_PREFILL_FULL_PATH_FIRST_DIVERGENCE_V1"
    assert M.RESULT_DIR != M.KERNEL_RESULTS


def test_02_primary_trace_is_ns_rs():
    assert {"NS", "RS"}.issubset(M.TRACE_BRANCHES)


def test_03_model_parallel_is_balanced_and_bounded():
    source = inspect.getsource(M.load_sharded_model)
    assert 'device_map="balanced"' in source
    assert "max_memory=memory" in source


def test_04_input_device_comes_from_embeddings():
    assert "get_input_embeddings" in inspect.getsource(M.model_input_device)


def test_05_only_chunk_prefill_is_replaced():
    source = inspect.getsource(M.FullPathForensicProbe._wrap)
    assert 'operator != "chunk_kda"' in source


def test_06_sequential_token_order_is_exact():
    source = inspect.getsource(M.FullPathForensicProbe._wrap)
    assert "for token_index in range" in source
    assert "token_index:token_index + 1" in source


def test_07_sequential_state_is_carried_exactly():
    source = inspect.getsource(M.FullPathForensicProbe._wrap)
    assert 'step["initial_state"] = state' in source
    assert "output, state = self.orig_fused" in source


def test_08_runtime_value_rotation_uses_validated_orientation():
    source = inspect.getsource(M.FullPathForensicProbe._wrap)
    assert "BASE.driver_to_branch_coordinates" in source


def test_09_runtime_mapback_uses_r_transpose():
    source = inspect.getsource(M.FullPathForensicProbe._wrap)
    assert "self.rotation.t()" in source


def test_10_kernel_state_and_endpoint_transform_are_separate():
    source = inspect.getsource(M.prefill_with_boundaries)
    assert '"kernel_return"' in source
    assert '"endpoint_basis_handling"' in source
    assert "apply_endpoint_rotation" in source


def test_11_postquant_state_is_separate():
    source = inspect.getsource(M.prefill_with_boundaries)
    assert '"postquant"' in source
    assert "quantize_branch_cache" in source


def test_12_state_is_mapped_before_native_comparison():
    source = inspect.getsource(M.free_running_rows)
    assert "map_state_to_native" in source
    assert '"INVERSE_STATE_VALUE_ROTATION"' in source


def test_13_value_is_mapped_before_native_comparison():
    source = inspect.getsource(M.free_running_rows)
    assert "map_value_to_native" in source
    assert '"INVERSE_VALUE_ROTATION"' in source


def test_14_trace_rows_have_required_basis_tags():
    row = M.row_from_pair("u", "p", 0, 0, "x", torch.ones(2), torch.ones(2),
                          "NATIVE_BASIS", "ROTATED_VALUE_BASIS", torch.ones(2), "R.T")
    assert row["tensor_basis_native"] == "NATIVE_BASIS"
    assert row["tensor_basis_rotated"] == "ROTATED_VALUE_BASIS"
    assert row["comparison_basis"] == "NATIVE_BASIS"
    assert row["map_applied"] == "R.T"


def test_15_metric_identity_is_exact():
    stat = M.metrics(torch.ones(4), torch.ones(4))
    assert stat["max_abs"] == 0 and stat["relative_l2"] == 0
    assert stat["cosine"] == pytest.approx(1.0)


def test_16_none_state_identity_is_exact():
    stat = M.metrics(None, None)
    assert stat["relative_l2"] == 0 and stat["cosine"] == 1


def test_17_value_theory_check_uses_actual_runtime_semantic_value():
    source = inspect.getsource(M.free_running_rows)
    assert 'rs["v_semantic"].float().matmul(rotation.float())' in source


def test_18_mapback_theory_check_uses_actual_raw_output():
    source = inspect.getsource(M.free_running_rows)
    assert 'rs["raw_core_output"].float().matmul(rotation.t().float())' in source


def test_19_frozen_replay_uses_native_state_and_value():
    source = inspect.getsource(M.frozen_replay)
    assert 'state_native = rec["state_in"]' in source
    assert 'rec["v_semantic"]' in source


def test_20_frozen_replay_transforms_only_basis_dependent_inputs():
    source = inspect.getsource(M.frozen_replay)
    assert "BASE.state_to_branch_coordinates" in source
    assert "value_rot" in source
    assert "fused_call(probe, rec, state_rot, value_rot" in source


def test_21_frozen_call_uses_exact_recorded_drivers():
    source = inspect.getsource(M.fused_call)
    for name in ("q", "k", "log_decay", "beta"):
        assert f'rec["{name}"]' in source


def test_22_hidden_freeze_regenerates_real_projections_and_convs():
    source = inspect.getsource(M.hidden_driver_regeneration)
    for name in ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d"):
        assert name in source


def test_23_native_instrumentation_replay_sets_empirical_floor():
    source = inspect.getsource(M.native_replay_floors)
    assert "native_instrumentation_replay_relative_L2" in source
    assert "high_precision_rotation_relative_L2" in source


def test_24_first_divergence_is_floor_relative():
    source = inspect.getsource(M.locate_divergence)
    assert "10.0 * max(floor" in source
    assert '"AMPLIFICATION_RATIO"' in source


def test_25_endpoint_stages_are_in_execution_order():
    source = inspect.getsource(M.locate_divergence)
    assert '"prefill_endpoint_kernel_return": 20' in source
    assert '"prefill_endpoint_endpoint_basis_handling": 21' in source
    assert '"prefill_endpoint_postquant": 22' in source


def test_26_hooks_cover_full_post_core_path():
    source = inspect.getsource(M.FullPathForensicProbe.install)
    for name in ("input_layernorm", "o_norm", "o_proj", "post_attention_layernorm", "mlp"):
        assert name in source


def test_27_trace_audit_does_not_invent_fused_substages():
    audit = M.trace_audit()
    assert audit["FULL_PATH_TRACE_AUDIT"] == "PASS"
    assert any("fused" in value for value in audit["nonexistent_separate_stages"])


def test_28_previous_artifacts_are_read_only():
    source = inspect.getsource(M)
    assert "KERNEL_RESULTS" in source and "PRECISION_RESULTS" in source
    assert "save_json(KERNEL_RESULTS" not in source
    assert "write_rows(KERNEL_RESULTS" not in source


def test_29_optional_phase2_placeholders_are_explicit():
    source = inspect.getsource(M.placeholder_phase2)
    assert "NOT_RUN_BY_STOP_RULE" in source


def test_30_validation_requires_coordinate_schema():
    source = inspect.getsource(M.validate)
    for name in ("tensor_basis_native", "tensor_basis_rotated", "map_applied", "comparison_basis"):
        assert name in source


def test_31_phase2_has_exact_three_causal_conditions():
    source = inspect.getsource(M.evaluate_phase2_unit)
    assert '(("NS", False), ("RS", True), ("RS_FIXED", False))' in source


def test_32_intervention_removes_only_endpoint_rotation():
    source = inspect.getsource(M.evaluate_phase2_unit)
    assert "apply_endpoint_rotation=endpoint_rotation" in source
    assert 'basis = "native" if name == "NS" else "rotated"' in source


def test_33_fixed_branch_uses_same_sequential_operator():
    assert "RS_FIXED" in M.TRACE_BRANCHES


def test_34_all_causal_conditions_map_to_native_boundary():
    source = inspect.getsource(M.evaluate_phase2_unit)
    assert 'for name in ("RS", "RS_FIXED")' in source
    assert "H.to_native(items[name]" in source


def test_35_all_causal_conditions_use_native_continuation():
    source = inspect.getsource(M.evaluate_phase2_unit)
    assert '"_NATIVE_CONTINUATION"' in source
    assert 'H.plan(name + "_NATIVE_CONTINUATION", "native", True)' in source


def test_36_causal_statistics_are_unitwise_median_bootstrap():
    source = inspect.getsource(M.phase2)
    assert 'H.effect([row["CAUSAL_RESCUE"] for row in unit_rows]' in source
    assert 'H.effect([row["CAUSAL_CLOSURE"] for row in unit_rows]' in source


def test_37_phase2_requires_phase1_authorization():
    source = inspect.getsource(M.phase2)
    assert "Phase I did not authorize" in source


def test_38_phase2_does_not_change_weights():
    source = inspect.getsource(M.phase2)
    assert '"MODEL_WEIGHTS_UNCHANGED"' in source
    assert "weight_hash ==" in source


def test_39_validation_checks_formal_unit_and_horizon_counts():
    source = inspect.getsource(M.validate)
    assert "EXPECTED_UNITS * PRIMARY_HORIZON * 3" in source
    assert '"formal_complete"' in source


def test_40_validation_rejects_nonfinite_formal_metrics():
    source = inspect.getsource(M.validate)
    assert "formal_finite" in source
    assert "math.isfinite" in source
