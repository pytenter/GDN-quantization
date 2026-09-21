import importlib.util
import inspect
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_rotation_prefill_kernel_equivariance_causal_v1.py"
SPEC = importlib.util.spec_from_file_location("prefill_kernel", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def synthetic_rows():
    return [{"unit_id": f"u{i}", "auc_NP": 0.01, "auc_NS": 0.012,
             "auc_RP": 0.12, "auc_RS": 0.015} for i in range(18)]


def synthetic_diagnostics():
    rows = []
    for i in range(18):
        for layer in range(18):
            for pair, value in (("NP_NS", 0.001), ("NP_RP", 0.2),
                                ("NS_RS", 0.002), ("RP_RS", 0.19)):
                rows.append({"unit_id": f"u{i}", "layer": layer, "pair": pair,
                             "state_relative_l2": value})
    return rows


def test_01_task_and_result_path_are_new():
    assert M.TASK == "KDA_ROTATION_PREFILL_KERNEL_EQUIVARIANCE_CAUSAL_V1"
    assert M.RESULT_DIR != M.WRITEBACK_RESULTS


def test_02_exact_four_factorial_conditions():
    assert M.CONDITIONS == ("NP", "NS", "RP", "RS")


def test_03_np_is_native_packed():
    assert M.condition_spec("NP") == ("native", "packed")


def test_04_ns_is_native_sequential():
    assert M.condition_spec("NS") == ("native", "sequential")


def test_05_rp_is_rotated_packed():
    assert M.condition_spec("RP") == ("rotated", "packed")


def test_06_rs_is_rotated_sequential():
    assert M.condition_spec("RS") == ("rotated", "sequential")


def test_07_sequential_intercepts_only_chunk_kernel():
    source = inspect.getsource(M.SequentialPrefillProbe._wrap)
    assert 'operator != "chunk_kda"' in source


def test_08_sequential_uses_original_decode_kernel():
    source = inspect.getsource(M.SequentialPrefillProbe._wrap)
    assert "self.orig_fused(**step)" in source


def test_09_sequential_preserves_token_order():
    source = inspect.getsource(M.SequentialPrefillProbe._wrap)
    assert "for index in range(sequence_length)" in source
    assert "index:index + 1" in source


def test_10_sequential_carries_exact_returned_state():
    source = inspect.getsource(M.SequentialPrefillProbe._wrap)
    assert 'step["initial_state"] = state' in source
    assert "output, state = self.orig_fused" in source


def test_11_same_sequential_implementation_for_ns_rs():
    assert M.SEQUENTIAL_CONDITIONS == {"NS", "RS"}


def test_12_rs_uses_validated_value_rotation():
    source = inspect.getsource(M.SequentialPrefillProbe._wrap)
    assert "BASE.driver_to_branch_coordinates" in source
    assert "self.rotation.t()" in source


def test_13_driver_identity_covers_required_drivers():
    source = inspect.getsource(M.driver_identity_rows)
    for name in ("q", "k", "v_semantic", "beta", "log_decay"):
        assert f'"{name}"' in source


def test_14_reference_validation_compares_full_fused_recurrence():
    source = inspect.getsource(M.SequentialPrefillProbe._wrap)
    assert "reference_output, reference_state = self.orig_fused" in source


def test_15_full_model_remains_packed():
    source = inspect.getsource(M.prepare_prefill)
    assert "H.prefill_branch" in source
    assert "model(input_ids" not in source


def test_16_rotated_prefill_maps_to_native_at_boundary():
    source = inspect.getsource(M.prepare_prefill)
    assert "H.to_native(items[name]" in source


def test_17_all_history_decode_is_native():
    source = inspect.getsource(M.prepare_prefill)
    assert '"native", True' in source


def test_18_all_future_evaluation_is_shared():
    source = inspect.getsource(M.continue_native)
    assert "H.evaluate_native_future" in source


def test_19_factorial_effects_are_per_unit():
    row = M.factorial_rows(synthetic_rows())[0]
    assert row["packed_rotation_effect"] == pytest.approx(0.11)
    assert row["sequential_rotation_effect"] == pytest.approx(0.003)
    assert row["packed_kernel_effect_native"] == pytest.approx(-0.002)
    assert row["packed_kernel_effect_rotated"] == pytest.approx(0.105)
    assert row["rotation_x_packed_interaction"] == pytest.approx(0.107)


def test_20_unitwise_auc_aggregation():
    rows = [{"unit_id": unit, "condition": condition, "future_kl": horizon}
            for unit in ("a", "b") for condition in M.CONDITIONS for horizon in (1, 2, 3)]
    out = M.aggregate_auc(rows)
    assert len(out) == 2 and out[0]["auc_NP"] == 2


def test_21_median_bootstrap_uses_unit_values():
    stat = M.effect([0.5] * 18, "X")
    assert stat["paired_median"] == 0.5 and stat["bootstrap_ci_low"] == 0.5


def test_22_layer_excess_uses_packed_minus_sequential_rotation_gap():
    earliest, rows = M.layer_excess_summary(synthetic_diagnostics())
    assert earliest == 0 and rows[0]["paired_median"] == pytest.approx(0.198)


def test_23_strong_factorial_classification():
    _, summary = M.summarize_factorial(synthetic_rows(), synthetic_diagnostics())
    assert summary["FINAL_KDA_PREFILL_CLASSIFICATION"] == "PACKED_PREFILL_KERNEL_ROTATION_INTERACTION_STRONGLY_SUPPORTED"


def test_24_state_carrier_replaces_only_kda_stack():
    source = inspect.getsource(M.run_unit)
    assert "H.clone_branch(source" in source
    assert "H.replace_stack(item[\"past\"], stacks[name])" in source


def test_25_no_driver_transplant_contamination():
    source = inspect.getsource(M.SequentialPrefillProbe._wrap)
    assert "source_records" not in source


def test_26_previous_formal_artifacts_are_read_only():
    source = inspect.getsource(M)
    assert "WRITEBACK_RESULTS" in source
    assert "write_rows(WRITEBACK_RESULTS" not in source
    assert "save_json(WRITEBACK_RESULTS" not in source
