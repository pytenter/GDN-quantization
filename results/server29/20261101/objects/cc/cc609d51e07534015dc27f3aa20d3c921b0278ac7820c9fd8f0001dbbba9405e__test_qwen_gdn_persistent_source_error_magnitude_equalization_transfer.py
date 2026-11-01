import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "propagation" / "run_qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("qwen_gdn_mag_eq", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen_gamma_axis_uses_c_magnitude_at_zero_and_r_magnitude_at_one():
    runner = load_runner()
    e_r = {0: torch.tensor([[[[3.0, 4.0]]]])}
    e_c = {0: torch.tensor([[[[0.0, 10.0]]]])}

    r_zero = runner.build_gamma_intervention(e_r, e_c, "R_DIR", 0.0)
    r_one = runner.build_gamma_intervention(e_r, e_c, "R_DIR", 1.0)
    c_zero = runner.build_gamma_intervention(e_r, e_c, "C_DIR", 0.0)
    c_one = runner.build_gamma_intervention(e_r, e_c, "C_DIR", 1.0)

    assert r_zero["m_target"] == pytest.approx(10.0)
    assert r_one["m_target"] == pytest.approx(5.0)
    assert c_zero["m_target"] == pytest.approx(10.0)
    assert c_one["m_target"] == pytest.approx(5.0)
    assert runner.stack_norm(r_zero["delta"]) == pytest.approx(10.0)
    assert runner.stack_norm(r_one["delta"]) == pytest.approx(5.0)
    assert runner.stack_norm(c_zero["delta"]) == pytest.approx(10.0)
    assert runner.stack_norm(c_one["delta"]) == pytest.approx(5.0)


def test_same_state_counterfactual_residual_generation_uses_one_source_state():
    runner = load_runner()
    state = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])

    out = runner.extract_same_state_residuals(torch, {0: state})

    assert set(out["e_R"]) == {0}
    assert set(out["e_C"]) == {0}
    assert torch.allclose(out["q_R"][0] - state.float(), out["e_R"][0])
    assert torch.allclose(out["q_C"][0] - state.float(), out["e_C"][0])
    assert out["source_identity_max_abs_error"] <= runner.IDENTITY_TOL
    assert out["SAME_STATE_COUNTERFACTUAL_GATE"] == "PASS"


def test_qwen_endpoint_identities_are_r_gamma_one_and_c_gamma_zero():
    runner = load_runner()
    rows = [
        {"unit_id": "u", "history_length": 64, "source_direction": "R_DIR", "gamma": "0", "future_kl_auc": 0.10, "state_error_t0": 1.0},
        {"unit_id": "u", "history_length": 64, "source_direction": "R_DIR", "gamma": "1", "future_kl_auc": 0.25, "state_error_t0": 2.0},
        {"unit_id": "u", "history_length": 64, "source_direction": "C_DIR", "gamma": "0", "future_kl_auc": 0.08, "state_error_t0": 1.2},
        {"unit_id": "u", "history_length": 64, "source_direction": "C_DIR", "gamma": "1", "future_kl_auc": 0.20, "state_error_t0": 2.5},
    ]

    out = runner.closed_loop_contrasts(rows)

    assert out["NATIVE_R_MINUS_C_GAP"] == pytest.approx(0.17)
    assert out["R_MAG_EQUALIZATION_RESCUE"] == pytest.approx(0.15)
    assert out["C_RMAG_GRAFT_DAMAGE"] == pytest.approx(0.12)
    assert out["R_EQ_VS_C_NATIVE"] == pytest.approx(0.02)
    assert out["C_GRAFT_VS_R_NATIVE"] == pytest.approx(-0.05)
    assert out["MAGNITUDE_MAIN"] == pytest.approx(0.135)
    assert out["STATE_ERROR_RESCUE"] == pytest.approx(1.0)
    assert out["STATE_ERROR_GRAFT_DAMAGE"] == pytest.approx(1.3)


def test_global_stack_norm_matching_preserves_direction():
    runner = load_runner()
    e_r = {
        0: torch.tensor([[[[3.0, 4.0]]]]),
        2: torch.tensor([[[[0.0, 12.0]]]]),
    }
    e_c = {
        0: torch.tensor([[[[0.0, 0.0]]]]),
        2: torch.tensor([[[[0.0, 13.0]]]]),
    }

    out = runner.build_gamma_intervention(e_r, e_c, "R_DIR", 0.0)

    assert out["m_R_same_state"] == pytest.approx(13.0)
    assert out["m_C_same_state"] == pytest.approx(13.0)
    assert out["applied_magnitude_relative_error"] <= runner.NORM_TOL
    assert out["direction_cosine"] >= 1.0 - runner.IDENTITY_TOL
    assert out["APPLIED_MAGNITUDE_GATE"] == "PASS"
    assert out["SOURCE_DIRECTION_PRESERVATION_GATE"] == "PASS"


def test_degenerate_direction_with_nonzero_target_fails_gate():
    runner = load_runner()
    e_r = {0: torch.zeros(1, 1, 1, 2)}
    e_c = {0: torch.tensor([[[[0.0, 7.0]]]])}

    out = runner.build_gamma_intervention(e_r, e_c, "R_DIR", 0.0)

    assert out["degenerate_residual_count"] == 1
    assert out["APPLIED_MAGNITUDE_GATE"] == "FAIL"
    assert out["SOURCE_DIRECTION_PRESERVATION_GATE"] == "FAIL"


def test_timing_schedule_is_active_history_only_and_future_is_full_precision():
    runner = load_runner()

    schedule = runner.history_schedule(t0=64, history_length=8, horizon=128)

    assert schedule["active_timesteps"] == list(range(57, 65))
    assert schedule["future_timesteps"] == list(range(65, 193))
    assert schedule["INTERVENTION_TIMING_GATE"] == "PASS"
    assert schedule["NO_FURTHER_QUANTIZATION_GATE"] == "PASS"
    assert schedule["CURRENT_LOGIT_NONRETROACTIVITY_GATE"] == "PASS"


def test_qwen_timestep_zero_is_prefill_and_future_h1_consumes_token_t0():
    runner = load_runner()
    tokens = [101, 102, 103, 104]

    assert runner.teacher_token_for_timestep(tokens, 1) == 101
    assert runner.teacher_token_for_timestep(tokens, 3) == 103
    assert runner.teacher_token_for_future_horizon(tokens, t0=2, horizon=1) == 103
    assert runner.teacher_token_for_future_horizon(tokens, t0=2, horizon=2) == 104


def test_branch_configs_share_one_gamma_axis_and_native_endpoints():
    runner = load_runner()

    configs = runner.branch_configs([0.0, 0.5, 1.0])
    by_branch = {row["branch"]: row for row in configs}

    assert by_branch["FP"]["source_direction"] == "FP"
    assert by_branch["R_DIR_G1"]["native_endpoint"] == "R128"
    assert by_branch["C_DIR_G0"]["native_endpoint"] == "C128"
    assert by_branch["R_DIR_G0"]["native_endpoint"] is None
    assert by_branch["C_DIR_G1"]["native_endpoint"] is None


def test_apply_delta_reproduces_native_quantized_state_after_dtype_cast():
    runner = load_runner()
    state = torch.tensor([[[[0.01, 0.24], [-0.37, 0.91]]]], dtype=torch.bfloat16)
    stack = {0: state.clone()}
    residuals = runner.extract_same_state_residuals(torch, stack)

    r_native = runner.build_gamma_intervention(
        residuals["e_R"], residuals["e_C"], "R_DIR", 1.0
    )
    runner.apply_delta_to_stack(stack, r_native["delta"])

    expected = residuals["q_R"][0].to(state.dtype)
    assert torch.equal(stack[0], expected)
    assert runner.max_stack_abs_error(stack, {0: expected}) == 0.0


def test_unit_sharding_is_deterministic_and_complete():
    runner = load_runner()
    units = [{"unit_id": str(i)} for i in range(18)]

    shard0 = runner.select_shard_units(units, shard_id=0, num_shards=2)
    shard1 = runner.select_shard_units(units, shard_id=1, num_shards=2)

    assert [u["unit_id"] for u in shard0] == [str(i) for i in range(0, 18, 2)]
    assert [u["unit_id"] for u in shard1] == [str(i) for i in range(1, 18, 2)]
    assert {u["unit_id"] for u in shard0 + shard1} == {str(i) for i in range(18)}


def test_full_logit_metrics_use_forward_kl_and_preserve_identity():
    runner = load_runner()
    ref = torch.tensor([[[2.0, 0.0, -1.0]]])
    shifted = torch.tensor([[[0.0, 2.0, -1.0]]])

    same = runner.full_logit_metrics(torch, ref, ref)
    changed = runner.full_logit_metrics(torch, ref, shifted)

    assert same["KL"] == pytest.approx(0.0, abs=1e-8)
    assert same["top1_match"] == 1
    assert changed["KL"] > 0.0
    assert changed["top1_match"] == 0


def test_global_state_metrics_cover_absolute_relative_and_cosine():
    runner = load_runner()
    ref = {0: torch.tensor([1.0, 2.0]), 2: torch.tensor([3.0])}
    branch = {0: torch.tensor([2.0, 2.0]), 2: torch.tensor([3.0])}

    metrics = runner.state_metrics_from_stacks(torch, branch, ref)

    assert metrics["state_error_norm"] == pytest.approx(1.0)
    assert metrics["state_relative_L2"] == pytest.approx(1.0 / (14.0 ** 0.5))
    assert 0.0 < metrics["state_cosine"] < 1.0


def test_stage_protocol_matches_required_smoke_pilot_and_formal_panels():
    runner = load_runner()

    assert runner.stage_protocol("smoke") == {
        "history_lengths": [1, 8],
        "gamma_values": [0.0, 0.5, 1.0],
        "unit_limit": 1,
    }
    assert runner.stage_protocol("pilot")["history_lengths"] == [1, 8, 32, 64]
    assert runner.stage_protocol("pilot")["gamma_values"] == runner.GAMMA_VALUES
    assert runner.stage_protocol("pilot")["unit_limit"] == 6
    assert runner.stage_protocol("formal")["history_lengths"] == runner.HISTORY_LENGTHS
    assert runner.stage_protocol("formal")["unit_limit"] is None


def test_manifest_units_receive_stable_ids_without_changing_order():
    runner = load_runner()
    manifest = [
        {"problem_id": "a.json", "t0": 64, "role": "x"},
        {"problem_id": "b.json", "t0": 128, "role": "y"},
    ]

    units = runner.canonical_units_with_ids(manifest)

    assert [u["unit_id"] for u in units] == ["a.json|64", "b.json|128"]
    assert [u["problem_id"] for u in units] == ["a.json", "b.json"]


def test_qwen_cache_stack_supports_transformers_5_direct_state_lists():
    runner = load_runner()

    class DirectCache:
        recurrent_states = [torch.tensor([0]), None, torch.tensor([2])]

    class LegacyP1:
        @staticmethod
        def get_state(cache, layer):
            raise AssertionError("legacy accessor must not be used for direct-list cache")

    stack = runner.qwen_cache_stack(LegacyP1, DirectCache(), layers=[0, 2])

    assert stack[0].item() == 0
    assert stack[2].item() == 2


def test_apply_delta_can_update_cache_tensor_created_in_inference_mode():
    runner = load_runner()
    with torch.inference_mode():
        state = torch.tensor([1.0, 2.0])

    runner.apply_delta_to_stack({0: state}, {0: torch.tensor([0.5, -0.5])})

    assert torch.equal(state, torch.tensor([1.5, 1.5]))


def test_successful_worker_update_clears_stale_failure_records(tmp_path):
    runner = load_runner()
    path = tmp_path / "failures.json"
    runner.save_json(path, [{"error": "stale"}])

    runner.save_worker_failures(path, [])

    assert path.read_text(encoding="utf-8").strip() == "[]"


def test_exact_result_coverage_detects_duplicate_and_missing_branches():
    runner = load_runner()
    units = [{"unit_id": "u"}]
    configs = runner.branch_configs([0.0, 1.0])
    rows = [
        {
            "unit_id": "u",
            "history_length": 1,
            "source_direction": config["source_direction"],
            "gamma": config["gamma"],
        }
        for config in configs
    ]

    assert runner.exact_result_coverage(rows, units, [1], [0.0, 1.0])
    assert not runner.exact_result_coverage(rows[:-1] + [rows[0]], units, [1], [0.0, 1.0])


def test_branch_success_counts_report_each_branch_against_expected_pairs():
    runner = load_runner()
    units = [{"unit_id": "u1"}, {"unit_id": "u2"}]
    configs = runner.branch_configs([0.0, 1.0])
    rows = [
        {
            "unit_id": unit["unit_id"],
            "history_length": history_length,
            "branch": config["branch"],
            "source_direction": config["source_direction"],
            "gamma": config["gamma"],
        }
        for unit in units
        for history_length in (1, 8)
        for config in configs
    ]

    counts = runner.branch_success_counts(rows, units, [1, 8], [0.0, 1.0])

    assert set(counts) == {config["branch"] for config in configs}
    assert all(item == {"n_success": 4, "n_expected": 4, "status": "4 / 4"} for item in counts.values())


def test_source_norm_summary_weights_each_unit_history_cell_equally():
    runner = load_runner()
    rows = [
        {"unit_id": "u1", "history_length": 1, "m_R_same_state": 10.0, "m_C_same_state": 1.0},
        *[
            {"unit_id": "u2", "history_length": 64, "m_R_same_state": 1.0, "m_C_same_state": 2.0}
            for _ in range(100)
        ],
    ]

    summary = runner.summarize_source_norms(rows)

    assert summary["R_SOURCE_NORM"] == pytest.approx(5.5)
    assert summary["C_SOURCE_NORM"] == pytest.approx(1.5)
    assert summary["R_NORM_GT_C"] == "1 / 2"
    assert summary["C_NORM_GT_R"] == "1 / 2"
    assert len(summary["per_unit_history_length"]) == 2
    assert summary["per_unit_history_length"][1]["fraction_R_norm_gt_C_norm"] == 0.0


def test_checkpoint_validation_rejects_wrong_horizon_and_incomplete_endpoint_data():
    runner = load_runner()
    unit = {"unit_id": "u", "problem_id": "p", "t0": 64}
    rows = []
    schedule_rows = []
    audits = []
    for config in runner.branch_configs([0.0, 1.0]):
        rows.append({
            "unit_id": "u",
            "history_length": 1,
            "source_direction": config["source_direction"],
            "gamma": config["gamma"],
            "manifest_sha256": "manifest",
            "stage": "formal",
            **{f"KL_h{horizon}": 0.0 for horizon in runner.FUTURE_KL_HORIZONS},
            "future_kl_auc": 0.0,
            "state_error_t0": 0.0,
            "state_error_future": 0.0,
        })
        if config["source_direction"] != "FP":
            schedule_rows.append({
                "unit_id": "u",
                "history_length": 1,
                "branch": config["branch"],
                "active_timestep": 64,
                "SAME_STATE_COUNTERFACTUAL_GATE": "PASS",
                "APPLIED_MAGNITUDE_GATE": "PASS",
                "SOURCE_DIRECTION_PRESERVATION_GATE": "PASS",
                "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
            })
            audits.append({
                "branch": config["branch"],
                "history_length": 1,
                "native_endpoint": config["native_endpoint"],
                "endpoint_identity_max_abs_error": 0.0 if config["native_endpoint"] else None,
                "intervention_count": 1,
                "future_quantization_count": 0,
            })
    result = {
        "task": runner.TASK,
        "stage": "formal",
        "manifest_sha256": "manifest",
        "unit": unit,
        "rows": rows,
        "schedule_rows": schedule_rows,
        "state_rows": [
            {"history_length": 1, "branch": config["branch"], "horizon": horizon}
            for config in runner.branch_configs([0.0, 1.0], include_fp=False)
            for horizon in runner.STATE_HORIZONS
        ],
        "audits": audits,
    }

    assert runner.validate_unit_result(result, unit, "formal", "manifest", [1], [0.0, 1.0], 128) == []
    result["rows"][1]["KL_h128"] = None
    errors = runner.validate_unit_result(result, unit, "formal", "manifest", [1], [0.0, 1.0], 128)

    assert "missing_or_nonfinite_KL_h128" in errors
    result["rows"][1]["KL_h128"] = 0.0
    result["rows"][1].pop("KL_h8", None)
    errors = runner.validate_unit_result(result, unit, "formal", "manifest", [1], [0.0, 1.0], 128)
    assert "missing_or_nonfinite_KL_h8" in errors
    result["rows"][1]["KL_h8"] = 0.0
    result["state_rows"][-1] = dict(result["state_rows"][0])
    errors = runner.validate_unit_result(result, unit, "formal", "manifest", [1], [0.0, 1.0], 128)
    assert "state_trajectory_coverage_mismatch" in errors
    result["state_rows"] = [
        {"history_length": 1, "branch": config["branch"], "horizon": horizon}
        for config in runner.branch_configs([0.0, 1.0], include_fp=False)
        for horizon in runner.STATE_HORIZONS
    ]
    result["runner_sha256"] = "source-a"
    errors = runner.validate_unit_result(result, unit, "formal", "manifest", [1], [0.0, 1.0], 128)
    assert "missing_observed_execution_traces" in errors
    result.pop("runner_sha256")
    next(audit for audit in audits if audit["native_endpoint"] == "R128")["endpoint_identity_max_abs_error"] = None
    errors = runner.validate_unit_result(result, unit, "formal", "manifest", [1], [0.0, 1.0], 128)
    assert "blocking_gate_R_NATIVE_ENDPOINT_IDENTITY_GATE" in errors
    r_audit = next(audit for audit in audits if audit["native_endpoint"] == "R128")
    r_audit["endpoint_identity_max_abs_error"] = 0.0
    r_audit["native_endpoint"] = None
    errors = runner.validate_unit_result(result, unit, "formal", "manifest", [1], [0.0, 1.0], 128)
    assert "native_endpoint_metadata_mismatch" in errors


def test_observed_gates_require_exact_timing_and_nonvacuous_native_endpoints():
    runner = load_runner()
    schedule_rows = [
        {
            "unit_id": "u",
            "history_length": 2,
            "branch": "R_DIR_G1",
            "active_timestep": timestep,
            "SAME_STATE_COUNTERFACTUAL_GATE": "PASS",
            "APPLIED_MAGNITUDE_GATE": "PASS",
            "SOURCE_DIRECTION_PRESERVATION_GATE": "PASS",
            "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
        }
        for timestep in (63, 64)
    ]
    audits = [{
        "unit_id": "u",
        "branch": "R_DIR_G1",
        "history_length": 2,
        "native_endpoint": "R128",
        "endpoint_identity_max_abs_error": None,
        "intervention_count": 2,
        "future_quantization_count": 0,
    }]

    gates = runner.observed_execution_gates(schedule_rows, audits, [{"unit_id": "u", "t0": 64}], 128)

    assert gates["INTERVENTION_TIMING_GATE"] == "PASS"
    assert gates["R_NATIVE_ENDPOINT_IDENTITY_GATE"] == "FAIL"
    assert gates["C_NATIVE_ENDPOINT_IDENTITY_GATE"] == "FAIL"
    schedule_rows[-1]["active_timestep"] = 65
    gates = runner.observed_execution_gates(schedule_rows, audits, [{"unit_id": "u", "t0": 64}], 128)
    assert gates["INTERVENTION_TIMING_GATE"] == "FAIL"


def test_formal_prerequisite_rejects_failed_gate_even_with_positive_pilot_label():
    runner = load_runner()
    _obj, _manifest, manifest_hash = runner.load_canonical_manifest()
    summary = {
        "task": runner.TASK,
        "stage": "pilot",
        "canonical_manifest_sha256": manifest_hash,
        "execution_runner_sha256": ["source-a"],
        "protocol_errors": [],
        "checkpoint_validation_errors": [],
        "formal_status": "NOT_RUN_PILOT_COMPLETE",
        "PILOT": "POSITIVE_OR_MIXED_SUPPORTS_FORMAL",
        "FRESH_OUTPUT_VERIFICATION": "PASS",
        "gates": {key: "PASS" for key in runner.BLOCKING_GATE_NAMES},
    }
    assert runner.formal_prerequisite_status(summary) == (True, "PASS")

    summary["gates"]["APPLIED_MAGNITUDE_GATE"] = "FAIL"
    allowed, reason = runner.formal_prerequisite_status(summary)

    assert not allowed
    assert reason == "BLOCKING_GATES_FAILED"


def test_stage_prerequisites_require_fresh_valid_previous_stage():
    runner = load_runner()
    stage0 = {
        "stage": "stage0",
        "formal_status": "NOT_RUN_STAGE0_COMPLETE",
        "FRESH_OUTPUT_VERIFICATION": "PASS",
        "gates": {key: "PASS" for key in runner.BLOCKING_GATE_NAMES},
        "task": runner.TASK,
        "canonical_manifest_sha256": "manifest",
        "execution_runner_sha256": ["source-a"],
        "protocol_errors": [],
        "checkpoint_validation_errors": [],
    }
    assert runner.stage_prerequisite_status("smoke", stage0, "manifest") == (True, "PASS")
    stage0["FRESH_OUTPUT_VERIFICATION"] = "FAIL"
    assert runner.stage_prerequisite_status("smoke", stage0, "manifest")[0] is False


def test_freshness_failure_masks_every_scientific_interpretation_field():
    runner = load_runner()
    summary = {
        "formal_status": "COMPLETE",
        "QWEN_PERSISTENT_SOURCE_MAGNITUDE_CAUSAL": "STRONG",
        "LING_TO_QWEN_MAGNITUDE_MECHANISM_TRANSFER": "SUPPORTED",
        "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED": "NO",
        "QWEN_FUNCTIONAL_GEOMETRY_DOMINANCE": "NOT_REQUIRED_AS_DOMINANT",
        "CROSS_ARCH_MAGNITUDE_ACCUMULATION_COMMON_MECHANISM": "SUPPORTED",
        "CROSS_ARCH_MECHANISM_HYPOTHESIS": "H1_COMMON_MAGNITUDE_ACCUMULATION_MECHANISM",
        "FINAL_CLASSIFICATION": "POSITIVE",
        "NEXT_RECOMMENDED_TASK": "NEXT",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES",
    }

    runner.invalidate_scientific_interpretation(summary, "INVALID_FRESH_OUTPUT_VERIFICATION")

    assert summary["formal_status"] == "INVALID"
    assert summary["FINAL_CLASSIFICATION"] == "INVALID_FRESH_OUTPUT_VERIFICATION"
    assert summary["QWEN_PERSISTENT_SOURCE_MAGNITUDE_CAUSAL"] == "NOT_INTERPRETABLE_INVALID_OUTPUT"
    assert summary["LING_TO_QWEN_MAGNITUDE_MECHANISM_TRANSFER"] == "NOT_INTERPRETABLE_INVALID_OUTPUT"
    assert summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] == "NO"


def test_scientific_classification_cannot_promote_smoke_or_infinitesimal_effects():
    runner = load_runner()

    def stat(value, low=None, positive=18):
        return {
            "median": value,
            "bootstrap_ci": [value if low is None else low, value],
            "positive": positive,
            "negative": 18 - positive,
            "n": 18,
        }

    contrasts = {
        "NATIVE_R_MINUS_C_GAP_BY_L": {"64": stat(1.0, 0.8)},
        "R_MAG_EQUALIZATION_RESCUE_BY_L": {"64": stat(1e-300, 0.0)},
        "C_RMAG_GRAFT_DAMAGE_BY_L": {"64": stat(1e-300, 0.0)},
        "R_EQ_VS_C_NATIVE_BY_L": {"64": stat(0.0, 0.0)},
        "STATE_ERROR_RESCUE_BY_L": {"64": stat(1e-300, 0.0)},
        "STATE_ERROR_GRAFT_DAMAGE_BY_L": {"64": stat(1e-300, 0.0)},
    }
    dose = {
        direction: {"median_spearman_rho": 1.0, "positive_rho_count": 18, "n_units": 18}
        for direction in runner.SOURCE_DIRECTIONS
    }
    source = {"QWEN_SOURCE_MAGNITUDE_ORIENTATION": "R_GT_C"}

    smoke = runner.scientific_classification(
        contrasts, dose, source, stage="smoke", blocking_pass=True, complete_units=True
    )
    formal = runner.scientific_classification(
        contrasts, dose, source, stage="formal", blocking_pass=True, complete_units=True
    )

    assert smoke["final"] == "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL"
    assert formal["causal"] != "STRONG"
    assert formal["functional_geometry_dominance"] == "PRIORITY_RAISED"
    assert formal["common_magnitude_mechanism"] == "NOT_SUPPORTED"


def test_resume_metadata_rejects_changed_protocol_or_runner_source():
    runner = load_runner()
    manifest = {
        "task": runner.TASK,
        "stage": "formal",
        "shard_id": 0,
        "num_shards": 2,
        "manifest_sha256": "manifest",
        "history_lengths": runner.HISTORY_LENGTHS,
        "gamma_values": runner.GAMMA_VALUES,
        "horizon": 128,
    }
    provenance = {
        "task": runner.TASK,
        "stage": "formal",
        "shard_id": 0,
        "num_shards": 2,
        "runner_sha256": "source-a",
    }

    assert runner.validate_resume_metadata(
        manifest, provenance, "formal", 0, 2, "manifest",
        runner.HISTORY_LENGTHS, runner.GAMMA_VALUES, 128, "source-a"
    ) == []
    assert "runner_sha256" in runner.validate_resume_metadata(
        manifest, provenance, "formal", 0, 2, "manifest",
        runner.HISTORY_LENGTHS, runner.GAMMA_VALUES, 128, "source-b"
    )
    manifest["horizon"] = 64
    assert "manifest_horizon" in runner.validate_resume_metadata(
        manifest, provenance, "formal", 0, 2, "manifest",
        runner.HISTORY_LENGTHS, runner.GAMMA_VALUES, 128, "source-a"
    )


def test_checkpoint_source_provenance_supports_attested_active_legacy_run_only():
    runner = load_runner()

    assert runner.checkpoint_source_provenance_errors(
        {"runner_sha256": "source-a"}, {"runner_sha256": "source-a"}
    ) == []
    assert runner.checkpoint_source_provenance_errors(
        {}, {"runner_sha256": "source-a", "captured_during_active_execution": "2026-09-09T18:15:42+08:00"}
    ) == []
    assert runner.checkpoint_source_provenance_errors(
        {}, {"runner_sha256": "source-a"}
    ) == ["legacy_checkpoint_without_active_execution_attestation"]
    assert runner.checkpoint_source_provenance_errors(
        {"runner_sha256": "source-b"}, {"runner_sha256": "source-a"}
    ) == ["checkpoint_runner_sha256_mismatch"]


def test_execution_provenance_is_bound_to_an_archived_executable(tmp_path):
    runner = load_runner()
    artifact = tmp_path / "runner.pyc"
    artifact.write_bytes(b"exact executed bytecode")
    digest = runner.hashlib.sha256(artifact.read_bytes()).hexdigest()
    provenance = {
        "task": runner.TASK,
        "stage": "formal",
        "shard_id": 0,
        "num_shards": 2,
        "runner_sha256": "source-a",
        "execution_artifact_path": str(artifact),
        "execution_artifact_sha256": digest,
    }

    assert runner.validate_execution_provenance(provenance, "formal", 0, 2) == []
    artifact.write_bytes(b"tampered")
    assert "execution_artifact_sha256" in runner.validate_execution_provenance(provenance, "formal", 0, 2)


def test_cross_arch_table_does_not_claim_native_orientation_before_formal():
    runner = load_runner()
    rows = runner.cross_arch_rows({
        "stage": "smoke",
        "formal_status": "NOT_RUN_SMOKE_COMPLETE",
        "FINAL_CLASSIFICATION": "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL",
        "source_norms": {},
    })

    native = next(row for row in rows if row["factor"] == "Native worse orientation")
    assert native["qwen_gdn"] == "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL"
