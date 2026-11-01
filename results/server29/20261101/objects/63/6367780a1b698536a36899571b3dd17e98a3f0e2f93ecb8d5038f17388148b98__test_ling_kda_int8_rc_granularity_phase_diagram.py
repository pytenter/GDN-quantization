import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_granularity", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_config_order_and_formal_row_count_are_fixed():
    runner = load_runner()

    assert runner.CONFIGS == ["R128", "R64", "R32", "R16", "C128", "C64", "C32", "C16"]
    assert runner.GROUP_SIZES == [128, 64, 32, 16]
    assert runner.HISTORY_LENGTHS == [1, 4, 8, 16, 32, 64]
    assert runner.expected_unit_rows("formal", n_units=18) == 18 * 6 * 8
    assert runner.expected_jsonl_rows("formal", n_units=18, horizon=128) == 18 * 6 * 8
    assert runner.expected_horizon_rows("formal", n_units=18, horizon=128) == 18 * 6 * 8 * 128
    assert runner.expected_source_rows("formal", n_units=18) == 18 * 8 * sum(runner.HISTORY_LENGTHS)
    assert runner.expected_state_rows("formal", n_units=18) == 18 * 6 * 8 * len(runner.STATE_HORIZONS)


def test_grouped_quantizer_r_splits_value_axis_only():
    runner = load_runner()
    state = torch.zeros(1, 1, 2, 8, dtype=torch.float32)
    state[0, 0, 0, :] = torch.arange(1, 9)
    state[0, 0, 1, :] = torch.arange(10, 90, 10)

    _qdq, meta = runner.fake_quant_ling_state_grouped(state, "R4")

    assert meta["orientation"] == "R"
    assert meta["group_size"] == 4
    assert meta["group_axis"] == "V"
    assert meta["crosses_key_channels"] is False
    assert list(meta["scale"].shape) == [1, 1, 2, 2, 1]
    assert torch.allclose(meta["scale"][0, 0, 0, :, 0], torch.tensor([4.0 / 127.0, 8.0 / 127.0]))
    assert torch.allclose(meta["scale"][0, 0, 1, :, 0], torch.tensor([40.0 / 127.0, 80.0 / 127.0]))


def test_grouped_quantizer_c_splits_key_axis_only():
    runner = load_runner()
    state = torch.zeros(1, 1, 8, 2, dtype=torch.float32)
    state[0, 0, :, 0] = torch.arange(1, 9)
    state[0, 0, :, 1] = torch.arange(10, 90, 10)

    _qdq, meta = runner.fake_quant_ling_state_grouped(state, "C4")

    assert meta["orientation"] == "C"
    assert meta["group_size"] == 4
    assert meta["group_axis"] == "K"
    assert meta["crosses_key_channels"] is True
    assert meta["key_channels_per_scale"] == 4
    assert list(meta["scale"].shape) == [1, 1, 2, 1, 2]
    assert torch.allclose(meta["scale"][0, 0, :, 0, 0], torch.tensor([4.0 / 127.0, 8.0 / 127.0]))
    assert torch.allclose(meta["scale"][0, 0, :, 0, 1], torch.tensor([40.0 / 127.0, 80.0 / 127.0]))


def test_quantizer_identity_gate_uses_own_dequantized_tensor():
    runner = load_runner()
    state = torch.randn(1, 1, 4, 4)
    qdq, _meta = runner.fake_quant_ling_state_grouped(state, "R4")

    gate = runner.quantizer_identity_gate(qdq, qdq)

    assert gate["gate"] == "PASS"
    assert gate["max_abs_error"] == 0.0


def test_quantize_cache_grouped_can_update_inference_cache_tensor():
    runner = load_runner()

    class Layer:
        pass

    class Cache:
        pass

    layer = Layer()
    with torch.inference_mode():
        layer.keys = torch.randn(1, 1, 4, 4)
    cache = Cache()
    cache.layers = [layer]

    meta = runner.quantize_cache_grouped(torch, cache, "R4", [0])

    assert meta["finite"] is True
    assert meta["touched_layers"] == [0]


def test_delta_func_is_c_auc_minus_r_auc_and_positive_prefers_r():
    runner = load_runner()

    row = runner.phase_row_from_values(128, 64, kl_r=0.2, kl_c=0.5, norm_r=1.0, norm_c=2.0, mse_r=0.1, mse_c=0.4)

    assert row["C_MINUS_R_FUNCTIONAL_GAP"] == 0.3
    assert row["preferred"] == "R_PREFERRED"
    row = runner.phase_row_from_values(16, 64, kl_r=0.5, kl_c=0.2, norm_r=1.0, norm_c=0.5, mse_r=0.1, mse_c=0.05)
    assert row["preferred"] == "C_PREFERRED"


def test_decay_heterogeneity_uses_backend_log_decay_by_c_groups():
    runner = load_runner()
    rec = {
        0: {
            "g": torch.tensor([[[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]]),
            "A_log": torch.zeros(1),
            "dt_bias": torch.zeros(8),
        }
    }

    c = runner.decay_heterogeneity_for_records(torch, rec, "C4")
    r = runner.decay_heterogeneity_for_records(torch, rec, "R4")

    assert c["C_GROUP_DECAY_HETEROGENEITY"] > 0
    assert c["C_GROUP_DECAY_RANGE"] > 0
    assert r["R_ANALOG_DECAY_HETEROGENEITY"] == 0.0


def test_phase_classifier_covers_declared_cases():
    runner = load_runner()

    assert runner.NEAR_TIE_AUC <= 1e-5
    assert runner.classify_orientation({128: 0.30, 64: 0.25, 32: 0.22, 16: 0.21})["result"] == "R_PREFERENCE_STABLE"
    assert runner.classify_orientation({128: 0.30, 64: 0.20, 32: 0.10, 16: 0.04})["result"] == "R_PREFERENCE_WITH_C_GRANULARITY_RESCUE"
    assert runner.classify_orientation({128: 0.30, 64: 0.12, 32: 0.03, 16: 0.0})["result"] == "ORIENTATION_BOUNDARY_APPROACHED"
    assert runner.classify_orientation({128: 0.30, 64: 0.10, 32: -0.01, 16: -0.02})["result"] == "ORIENTATION_FLIP_CONFIRMED"


def test_stage_protocols_and_required_gate_names_are_predeclared():
    runner = load_runner()

    assert runner.stage_configs("smoke") == ["R128", "R64", "R32", "C128", "C64", "C32"]
    assert runner.stage_history_lengths("pilot") == [1, 8, 32, 64]
    assert runner.stage_unit_limit("pilot") == 3
    assert set(runner.REQUIRED_GATES) >= {
        "CANONICAL_MANIFEST_GATE",
        "REPLAY_INPUT_GATE",
        "STATE_SEMANTICS_GATE",
        "ALL_QUANTIZER_SEMANTICS_GATES",
        "ALL_QUANTIZER_IDENTITY_GATES",
        "R128_BASELINE_REPRODUCTION_GATE",
        "C128_BASELINE_REPRODUCTION_GATE",
        "INSTRUMENTATION_NONINTERFERENCE_GATE",
    }


def test_timing_and_noninterference_checks_use_row_evidence():
    runner = load_runner()

    ok_source = [{
        "t0": 64,
        "history_length": 4,
        "active_timestep": t,
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
        "current_logit_nonretroactivity_max_abs": 0.0,
    } for t in [61, 62, 63, 64]]
    ok_horizon = [{"future_quantize_calls": 0, "no_further_quantization": True}]

    assert runner.timing_gate_from_source_rows(ok_source)["gate"] == "PASS"
    assert runner.noninterference_gate_from_rows(ok_source, ok_horizon)["gate"] == "PASS"

    bad_source = list(ok_source)
    bad_source[-1] = dict(bad_source[-1], active_timestep=65)
    assert runner.timing_gate_from_source_rows(bad_source)["gate"] == "FAIL"
    bad_horizon = [{"future_quantize_calls": 1, "no_further_quantization": True}]
    assert runner.noninterference_gate_from_rows(ok_source, bad_horizon)["gate"] == "FAIL"


def test_stage0_gate_requires_all_evidence_gates_to_pass():
    runner = load_runner()
    gates = {
        "REPLAY_INPUT_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "ALL_QUANTIZER_SEMANTICS_GATES": "PASS",
        "ALL_QUANTIZER_IDENTITY_GATES": "PASS",
        "R128_BASELINE_REPRODUCTION_GATE": "PASS",
        "C128_BASELINE_REPRODUCTION_GATE": "PASS",
        "TIMING_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
    }

    assert runner.stage0_gate_from_evidence(gates, failed_units=[]) == "PASS"
    assert runner.stage0_gate_from_evidence({**gates, "INSTRUMENTATION_NONINTERFERENCE_GATE": "FAIL"}, failed_units=[]) == "FAIL"
    assert runner.stage0_gate_from_evidence(gates, failed_units=[{"unit_id": "u"}]) == "FAIL"


def test_summarize_uses_row_noninterference_when_no_prior_stage_summary():
    runner = load_runner()
    source_rows = [{
        "unit_id": "u",
        "config": "R128",
        "history_length": 1,
        "t0": 64,
        "active_timestep": 64,
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
        "current_logit_nonretroactivity_max_abs": 0.0,
    }]
    horizon_rows = [{"future_quantize_calls": 0, "no_further_quantization": True}]

    summary = runner.summarize_aggregate("stage0", [], horizon_rows, source_rows, [])

    assert summary["INSTRUMENTATION_NONINTERFERENCE_GATE"] == "PASS"
    assert summary["TIMING_GATE"] == "PASS"


def test_fresh_output_verification_reports_missing_and_failed_units(tmp_path):
    runner = load_runner()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "FORMAL_STATUS": "COMPLETE",
        "N_FORMAL_UNITS": 18,
        "CANONICAL_MANIFEST_GATE": "PASS",
        "REPLAY_INPUT_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "ALL_QUANTIZER_SEMANTICS_GATES": "PASS",
        "ALL_QUANTIZER_IDENTITY_GATES": "PASS",
        "R128_BASELINE_REPRODUCTION_GATE": "PASS",
        "C128_BASELINE_REPRODUCTION_GATE": "PASS",
        "TIMING_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "COVERAGE_GATE": "PASS",
        "failed_units": [],
    }
    (run_dir / "aggregate_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name in [
        "results.jsonl",
        "granularity_phase_diagram.csv",
        "source_quantizability_by_granularity.csv",
        "decay_heterogeneity_by_granularity.csv",
        "report.md",
    ]:
        (run_dir / name).write_text("x\n", encoding="utf-8")
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps({"i": i}) for i in range(864)) + "\n", encoding="utf-8")

    ok = runner.verify_fresh_outputs(run_dir, expected_records=864)
    assert ok["FRESH_OUTPUT_VERIFICATION"] == "PASS"

    summary["failed_units"] = ["bad-unit"]
    (run_dir / "aggregate_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    bad = runner.verify_fresh_outputs(run_dir, expected_records=864)
    assert bad["FRESH_OUTPUT_VERIFICATION"] == "FAIL"
    assert bad["failed_units"] == ["bad-unit"]


def test_aggregate_writes_required_phase_source_and_decay_outputs(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    monkeypatch.setattr(runner, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path / "reports")
    runner.RESULT_DIR.mkdir()
    runner.REPORT_DIR.mkdir()

    unit_rows = []
    horizon_rows = []
    source_rows = []
    state_rows = []
    for unit_idx in range(18):
        for L in runner.HISTORY_LENGTHS:
            for config in runner.CONFIGS:
                parsed = runner.parse_config(config)
                g = parsed["group_size"]
                is_c = parsed["orientation"] == "C"
                auc = (0.1 if is_c else 0.05) * (g / 128.0) * (L / 64.0)
                source_norm = (2.0 if is_c else 1.0) * (g / 128.0)
                source_mse = (4.0 if is_c else 1.0) * (g / 128.0)
                row = {
                    "task": runner.TASK,
                    "stage": "formal",
                    "unit_id": f"u{unit_idx}",
                    "prompt_id": str(unit_idx),
                    "problem_index": unit_idx,
                    "t0": 64,
                    "history_length": L,
                    "config": config,
                    "orientation": parsed["orientation"],
                    "group_size": g,
                    "future_kl_auc": auc,
                    "source_norm": source_norm,
                    "source_mse": source_mse,
                    "state_error_t0": auc,
                    "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
                    "NO_FURTHER_QUANTIZATION_GATE": "PASS",
                }
                for h in runner.FUTURE_KL_HORIZONS:
                    row[f"KL_h{h}"] = auc
                unit_rows.append(row)
                runner.append_jsonl(tmp_path / "results_formal.jsonl", row)
                for h in range(1, 129):
                    horizon_rows.append({
                        "unit_id": f"u{unit_idx}",
                        "history_length": L,
                        "config": config,
                        "horizon": h,
                        "future_quantize_calls": 0,
                        "no_further_quantization": True,
                    })
                for h in runner.STATE_HORIZONS:
                    state_rows.append({"unit_id": f"u{unit_idx}", "history_length": L, "config": config, "horizon": h})
                source_rows.extend({
                    "unit_id": f"u{unit_idx}",
                    "history_length": L,
                    "config": config,
                    "orientation": parsed["orientation"],
                    "group_size": g,
                    "active_timestep": t,
                    "t0": 64,
                    "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
                    "current_logit_nonretroactivity_max_abs": 0.0,
                } for t in range(64 - L + 1, 65))
    runner.write_rows(tmp_path / "unit_summary_formal.csv", runner.merged_fields(unit_rows), unit_rows)
    pass_summary = {
        "CANONICAL_MANIFEST_GATE": "PASS",
        "REPLAY_INPUT_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "ALL_QUANTIZER_SEMANTICS_GATES": "PASS",
        "ALL_QUANTIZER_IDENTITY_GATES": "PASS",
        "R128_BASELINE_REPRODUCTION_GATE": "PASS",
        "C128_BASELINE_REPRODUCTION_GATE": "PASS",
        "TIMING_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "failed_units": [],
    }
    for name in ["stage0", "smoke", "pilot", "formal"]:
        (tmp_path / f"{name}_run_summary.json").write_text(json.dumps(pass_summary), encoding="utf-8")
    runner.write_rows(tmp_path / "horizon_formal.csv", runner.merged_fields(horizon_rows), horizon_rows)
    runner.write_rows(tmp_path / "source_error_formal.csv", runner.merged_fields(source_rows), source_rows)
    runner.write_rows(tmp_path / "state_error_trajectory_formal.csv", runner.merged_fields(state_rows), state_rows)

    summary = runner.aggregate("formal")

    assert summary["FORMAL_STATUS"] == "COMPLETE"
    assert summary["N_FORMAL_UNITS"] == 18
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "granularity_phase_diagram.csv").exists()
    assert (tmp_path / "source_quantizability_by_granularity.csv").exists()
    assert (tmp_path / "decay_heterogeneity_by_granularity.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert summary["C_MINUS_R_FUNCTIONAL_GAP_BY_GROUP_SIZE"]["128"] > 0
    assert summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] == "YES"
    assert summary["DELTA_FUNC_PAIRED_STATS_BY_GROUP_SIZE"]["128"]["n"] == 18
    assert summary["STATE_ERROR_R_BY_GROUP_SIZE"]["128"] is not None
    assert summary["STATE_ERROR_C_BY_GROUP_SIZE"]["128"] is not None
    assert summary["R_STATE_GRANULARITY_GAIN"] is not None
    assert summary["C_STATE_GRANULARITY_GAIN"] is not None


def test_aggregate_preserves_failed_units_from_failure_jsonl(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    monkeypatch.setattr(runner, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path / "reports")
    runner.RESULT_DIR.mkdir()
    runner.REPORT_DIR.mkdir()

    rows = []
    for unit_idx in range(17):
        for L in runner.HISTORY_LENGTHS:
            for config in runner.CONFIGS:
                parsed = runner.parse_config(config)
                rows.append({
                    "unit_id": f"u{unit_idx}",
                    "history_length": L,
                    "config": config,
                    "orientation": parsed["orientation"],
                    "group_size": parsed["group_size"],
                    "future_kl_auc": 0.1,
                    "source_norm": 1.0,
                    "source_mse": 1.0,
                    "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
                    "NO_FURTHER_QUANTIZATION_GATE": "PASS",
                })
    runner.write_rows(tmp_path / "unit_summary_formal.csv", runner.merged_fields(rows), rows)
    runner.write_rows(tmp_path / "horizon_formal.csv", ["no_further_quantization"], [{"no_further_quantization": True}])
    runner.write_rows(tmp_path / "source_error_formal.csv", ["CURRENT_LOGIT_NONRETROACTIVITY_GATE"], [{"CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS"}])
    runner.write_rows(tmp_path / "state_error_trajectory_formal.csv", ["unit_id"], [{"unit_id": "u0"}])
    (tmp_path / "failed_units_formal.jsonl").write_text(json.dumps({"unit_id": "u17", "error": "boom"}) + "\n", encoding="utf-8")

    summary = runner.aggregate("formal")

    assert summary["FORMAL_STATUS"] == "INCOMPLETE"
    assert summary["failed_units"] == [{"unit_id": "u17", "error": "boom"}]


def test_aggregate_preserves_formal_shard_gate_failures(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    monkeypatch.setattr(runner, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path / "reports")
    runner.RESULT_DIR.mkdir()
    runner.REPORT_DIR.mkdir()

    (tmp_path / "formal_run_summary_shard0.json").write_text(json.dumps({
        "CANONICAL_MANIFEST_GATE": "PASS",
        "REPLAY_INPUT_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "ALL_QUANTIZER_SEMANTICS_GATES": "PASS",
        "ALL_QUANTIZER_IDENTITY_GATES": "PASS",
        "R128_BASELINE_REPRODUCTION_GATE": "PASS",
        "C128_BASELINE_REPRODUCTION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "FAIL",
        "failed_units": [{"unit_id": "u_bad"}],
    }), encoding="utf-8")
    runner.write_rows(tmp_path / "unit_summary_formal.csv", ["unit_id"], [])
    runner.write_rows(tmp_path / "horizon_formal.csv", ["unit_id"], [])
    runner.write_rows(tmp_path / "source_error_formal.csv", ["unit_id"], [])
    runner.write_rows(tmp_path / "state_error_trajectory_formal.csv", ["unit_id"], [])

    summary = runner.aggregate("formal")

    assert summary["INSTRUMENTATION_NONINTERFERENCE_GATE"] == "FAIL"
    assert summary["failed_units"] == [{"unit_id": "u_bad"}]


def test_aggregate_requires_stage_gate_evidence_for_formal(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    monkeypatch.setattr(runner, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path / "reports")
    runner.RESULT_DIR.mkdir()
    runner.REPORT_DIR.mkdir()

    rows = []
    horizon_rows = []
    source_rows = []
    for unit_idx in range(18):
        for L in runner.HISTORY_LENGTHS:
            for config in runner.CONFIGS:
                parsed = runner.parse_config(config)
                rows.append({
                    "unit_id": f"u{unit_idx}",
                    "history_length": L,
                    "config": config,
                    "orientation": parsed["orientation"],
                    "group_size": parsed["group_size"],
                    "future_kl_auc": 0.1,
                    "source_norm": 1.0,
                    "source_mse": 1.0,
                    "state_error_t0": 0.1,
                    "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
                    "NO_FURTHER_QUANTIZATION_GATE": "PASS",
                })
                horizon_rows.append({
                    "unit_id": f"u{unit_idx}",
                    "history_length": L,
                    "config": config,
                    "horizon": 1,
                    "future_quantize_calls": 0,
                    "no_further_quantization": True,
                })
                source_rows.extend({
                    "unit_id": f"u{unit_idx}",
                    "history_length": L,
                    "config": config,
                    "orientation": parsed["orientation"],
                    "group_size": parsed["group_size"],
                    "active_timestep": t,
                    "t0": 64,
                    "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
                    "current_logit_nonretroactivity_max_abs": 0.0,
                } for t in range(64 - L + 1, 65))
    runner.write_rows(tmp_path / "unit_summary_formal.csv", runner.merged_fields(rows), rows)
    runner.write_rows(tmp_path / "horizon_formal.csv", runner.merged_fields(horizon_rows), horizon_rows)
    runner.write_rows(tmp_path / "source_error_formal.csv", runner.merged_fields(source_rows), source_rows)
    runner.write_rows(tmp_path / "state_error_trajectory_formal.csv", ["unit_id"], [{"unit_id": "u0"}])

    summary = runner.aggregate("formal")

    assert summary["CANONICAL_MANIFEST_GATE"] == "FAIL"
    assert summary["FORMAL_STATUS"] == "INCOMPLETE"


def test_formal_completion_fails_when_horizon_or_state_rows_are_incomplete(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    monkeypatch.setattr(runner, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path / "reports")
    runner.RESULT_DIR.mkdir()
    runner.REPORT_DIR.mkdir()

    rows = []
    source_rows = []
    for unit_idx in range(18):
        for L in runner.HISTORY_LENGTHS:
            for config in runner.CONFIGS:
                parsed = runner.parse_config(config)
                rows.append({
                    "unit_id": f"u{unit_idx}",
                    "history_length": L,
                    "config": config,
                    "orientation": parsed["orientation"],
                    "group_size": parsed["group_size"],
                    "future_kl_auc": 0.1,
                    "source_norm": 1.0,
                    "source_mse": 1.0,
                    "state_error_t0": 0.1,
                    "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
                    "NO_FURTHER_QUANTIZATION_GATE": "PASS",
                })
                source_rows.extend({
                    "unit_id": f"u{unit_idx}",
                    "history_length": L,
                    "config": config,
                    "orientation": parsed["orientation"],
                    "group_size": parsed["group_size"],
                    "active_timestep": t,
                    "t0": 64,
                    "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
                    "current_logit_nonretroactivity_max_abs": 0.0,
                } for t in range(64 - L + 1, 65))
    pass_summary = {
        "CANONICAL_MANIFEST_GATE": "PASS",
        "REPLAY_INPUT_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "ALL_QUANTIZER_SEMANTICS_GATES": "PASS",
        "ALL_QUANTIZER_IDENTITY_GATES": "PASS",
        "R128_BASELINE_REPRODUCTION_GATE": "PASS",
        "C128_BASELINE_REPRODUCTION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
    }
    for name in ["stage0", "smoke", "pilot", "formal"]:
        (tmp_path / f"{name}_run_summary.json").write_text(json.dumps(pass_summary), encoding="utf-8")
    runner.write_rows(tmp_path / "unit_summary_formal.csv", runner.merged_fields(rows), rows)
    runner.write_rows(tmp_path / "horizon_formal.csv", ["future_quantize_calls", "no_further_quantization"], [{"future_quantize_calls": 0, "no_further_quantization": True}])
    runner.write_rows(tmp_path / "source_error_formal.csv", runner.merged_fields(source_rows), source_rows)
    runner.write_rows(tmp_path / "state_error_trajectory_formal.csv", ["unit_id"], [{"unit_id": "u0"}])

    summary = runner.aggregate("formal")

    assert summary["FORMAL_STATUS"] == "INCOMPLETE"
    assert summary["COVERAGE_GATE"] == "FAIL"


def test_stage_protocol_snapshot_is_not_named_manifest(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)

    runner.write_stage_protocol_snapshot("stage0", {"stage": "stage0"})

    assert (tmp_path / "protocol_snapshot_stage0.json").exists()
    assert not (tmp_path / "manifest_stage0.json").exists()
