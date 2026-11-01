import csv
import ast
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "propagation" / "run_qwen_gdn_mag_matched_residual_functional_geometry_closure_v1.py"
OFFLINE_SCRIPT = ROOT / "experiments" / "propagation" / "offline_finish_qwen_gdn_mag_matched_residual_functional_geometry_closure_v1.sh"
SPEC = importlib.util.spec_from_file_location("qwen_gdn_magmatched_geometry", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_normalized_functional_metrics_are_scale_invariant():
    base = runner.normalized_functional_metrics(m0=2.0, m4=4.0, m5=6.0)
    scaled = runner.normalized_functional_metrics(m0=7.0, m4=14.0, m5=21.0)

    assert base == {"M0": 2.0, "M4": 4.0, "M5": 6.0, "F4": 2.0, "F5": 3.0}
    assert scaled["F4"] == pytest.approx(base["F4"])
    assert scaled["F5"] == pytest.approx(base["F5"])


def test_normalized_functional_metrics_reject_zero_norm():
    with pytest.raises(ValueError, match="zero-norm"):
        runner.normalized_functional_metrics(m0=0.0, m4=0.0, m5=0.0)


def test_factorial_interventions_use_literal_direction_magnitude_cells():
    e_r = {0: torch.tensor([[3.0, 4.0]])}
    e_c = {0: torch.tensor([[0.0, 2.0]])}

    cells = runner.build_factorial_interventions(e_r, e_c)

    assert runner.stack_norm(cells["RR"]["delta"]) == pytest.approx(5.0)
    assert runner.stack_norm(cells["RC"]["delta"]) == pytest.approx(2.0)
    assert runner.stack_norm(cells["CR"]["delta"]) == pytest.approx(5.0)
    assert runner.stack_norm(cells["CC"]["delta"]) == pytest.approx(2.0)
    assert runner.stack_cos(cells["RR"]["delta"], e_r) == pytest.approx(1.0)
    assert runner.stack_cos(cells["RC"]["delta"], e_r) == pytest.approx(1.0)
    assert runner.stack_cos(cells["CR"]["delta"], e_c) == pytest.approx(1.0)
    assert runner.stack_cos(cells["CC"]["delta"], e_c) == pytest.approx(1.0)
    assert cells["RR"]["source_direction"] == "R"
    assert cells["RC"]["source_direction"] == "R"
    assert cells["CR"]["source_direction"] == "C"
    assert cells["CC"]["source_direction"] == "C"
    assert cells["RR"]["source_magnitude_family"] == "R_MAG"
    assert cells["RC"]["source_magnitude_family"] == "C_MAG"
    assert cells["CR"]["source_magnitude_family"] == "R_MAG"
    assert cells["CC"]["source_magnitude_family"] == "C_MAG"


def test_factorial_interventions_reject_degenerate_source():
    e_r = {0: torch.zeros(2)}
    e_c = {0: torch.ones(2)}

    with pytest.raises(ValueError, match="degenerate"):
        runner.build_factorial_interventions(e_r, e_c)


def test_require_matching_source_state_rejects_fp_replay_drift():
    assert runner.require_matching_source_state("same", "same") == "same"
    with pytest.raises(RuntimeError, match="FP source-state replay drift"):
        runner.require_matching_source_state("precomputed", "runtime")


def test_state_signature_hashes_full_tensor_contents_not_just_norms():
    left = {0: torch.tensor([1.0, 0.0, 0.0])}
    right = {0: torch.tensor([0.0, 1.0, 0.0])}

    assert runner.state_signature(left) != runner.state_signature(right)


def test_offline_script_prints_required_summary_on_pilot_stop_and_formal_completion():
    text = OFFLINE_SCRIPT.read_text()

    assert "pilot did not support formalization; printing pilot summary and stopping" in text
    assert 'cat "$LOG_DIR/merge_pilot.log"' in text
    assert 'cat "$LOG_DIR/merge_formal.log"' in text


def test_frozen_driver_step_runs_under_inference_mode():
    calls = []

    class _FakeInferenceMode:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    class _FakeTorch:
        @staticmethod
        def inference_mode():
            return _FakeInferenceMode()

    class _FakeFrozenDriver:
        @staticmethod
        def driver_step(*args):
            calls.append(("driver_step", args))
            return "out", "past", {"records": True}

    result = runner.frozen_driver_step_inference(
        _FakeTorch, _FakeFrozenDriver, "model", "ids", "mask", "past", "collector"
    )

    assert result == ("out", "past", {"records": True})
    assert calls == [
        "enter",
        ("driver_step", (_FakeTorch, "model", "ids", "mask", "past", "collector")),
        "exit",
    ]


def test_all_frozen_driver_step_inference_call_sites_pass_frozen_module():
    tree = ast.parse(RUNNER.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozen_driver_step_inference"
    ]

    assert calls
    for call in calls:
        assert len(call.args) == 7
        assert isinstance(call.args[1], ast.Name)
        assert call.args[1].id == "frozen"


def test_formal_merge_requires_positive_pilot_prerequisite(monkeypatch):
    calls = []

    def _require(stage):
        calls.append(stage)
        raise RuntimeError("formal blocked: pilot is negative or inconclusive")

    def _unexpected(*args, **kwargs):
        raise AssertionError("formal merge collected results before checking pilot")

    monkeypatch.setattr(runner, "require_stage_prerequisite", _require)
    monkeypatch.setattr(runner, "collect_stage_results", _unexpected)

    with pytest.raises(RuntimeError, match="pilot is negative or inconclusive"):
        runner.aggregate_stage("formal", 2)

    assert calls == ["formal"]


def test_stage_protocols_have_exact_prespecified_sizes():
    assert runner.stage_protocol("stage0") == {"history_lengths": [1], "unit_indices": [0]}
    assert runner.stage_protocol("smoke") == {"history_lengths": [1, 8], "unit_indices": [0]}
    assert runner.stage_protocol("pilot") == {
        "history_lengths": [1, 8, 32, 64],
        "unit_indices": [0, 4, 8, 9, 13, 17],
    }
    assert runner.stage_protocol("formal") == {
        "history_lengths": [1, 4, 8, 16, 32, 64],
        "unit_indices": list(range(18)),
    }
    assert [cell["branch"] for cell in runner.branch_configs()] == ["FP", "RR", "RC", "CR", "CC"]


def test_exact_result_coverage_rejects_missing_and_duplicate_rows():
    units = [{"unit_id": "u0"}, {"unit_id": "u1"}]
    rows = [
        {"unit_id": unit["unit_id"], "history_length": length, "branch": branch["branch"]}
        for unit in units
        for length in [1, 8]
        for branch in runner.branch_configs()
    ]
    assert runner.exact_result_coverage(rows, units, [1, 8])
    assert not runner.exact_result_coverage(rows[:-1], units, [1, 8])
    assert not runner.exact_result_coverage(rows + [dict(rows[0])], units, [1, 8])


def test_direction_effect_rows_use_required_factorial_contrasts():
    rows = [
        {"unit_id": "u0", "history_length": 8, "branch": branch, "future_kl_auc": value}
        for branch, value in {"FP": 0.0, "RR": 10.0, "RC": 7.0, "CR": 4.0, "CC": 2.0}.items()
    ]
    effect = runner.build_direction_effect_rows(rows)[0]
    assert effect["DIRECTION_RISK_AT_C_MAG"] == pytest.approx(5.0)
    assert effect["DIRECTION_RISK_AT_R_MAG"] == pytest.approx(6.0)
    assert effect["DIRECTION_MAIN"] == pytest.approx(5.5)


def test_paired_summary_is_deterministic_and_reports_sign_test():
    first = runner.paired_summary([1.0, 2.0, 3.0, -1.0], seed=17)
    second = runner.paired_summary([1.0, 2.0, 3.0, -1.0], seed=17)
    assert first == second
    assert first["median"] == pytest.approx(1.5)
    assert first["positive"] == 3
    assert first["negative"] == 1
    assert first["n"] == 4
    assert 0.0 <= first["paired_sign_p"] <= 1.0


def test_history_amplification_uses_log2_history_length():
    by_l = {str(length): {"median": float(index)} for index, length in enumerate([1, 4, 8, 16, 32, 64])}
    assert runner.history_amplification_rho(by_l) == pytest.approx(1.0)


def test_m0_exact_matching_reports_no_discriminatory_signal():
    rows = []
    for index, unit in enumerate(["u0", "u1", "u2"]):
        for branch, auc, m0 in [
            ("RC", 3.0 + index, 2.0),
            ("CC", 1.0, 2.0),
            ("RR", 4.0 + index, 5.0),
            ("CR", 2.0, 5.0),
        ]:
            rows.append({"unit_id": unit, "history_length": 1, "branch": branch, "future_kl_auc": auc, "M0": m0})
    signal = runner.metric_predictive_signal(rows, "M0")
    assert signal["C_MAG"]["status"] == "NO_DISCRIMINATORY_SIGNAL_EXACT_MATCHING"
    assert signal["R_MAG"]["status"] == "NO_DISCRIMINATORY_SIGNAL_EXACT_MATCHING"
    assert signal["C_MAG"]["spearman"] is None
    assert signal["R_MAG"]["spearman"] is None


def test_closed_loop_consistency_thresholds_are_prespecified():
    closed = {str(length): {"median": value} for length, value in zip([1, 4, 8, 16, 32, 64], [1, 2, 3, 4, 5, 6])}
    strong = {str(length): {"median": value} for length, value in zip([1, 4, 8, 16, 32, 64], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])}
    partial = {str(length): {"median": value} for length, value in zip([1, 4, 8, 16, 32, 64], [1, 1, 1, -1, -1, -1])}
    none = {str(length): {"median": -1.0} for length in [1, 4, 8, 16, 32, 64]}
    assert runner.closed_loop_direction_consistency(strong, closed) == "STRONG"
    assert runner.closed_loop_direction_consistency(partial, closed) == "PARTIAL"
    assert runner.closed_loop_direction_consistency(none, closed) == "NOT_SUPPORTED"


def _cell(median, positive=18, n=18):
    return {"median": median, "positive": positive, "negative": n - positive, "ties": 0, "n": n}


def test_scientific_classification_covers_strong_partial_and_not_supported():
    lengths = ["1", "4", "8", "16", "32", "64"]
    same_state = {
        carrier: {length: _cell(1.0) for length in lengths}
        for carrier in ["C_NATIVE", "R_MAG_EQUALIZED"]
    }
    direction = {
        name: {length: _cell(1.0) for length in lengths}
        for name in ["DIRECTION_RISK_AT_C_MAG_BY_L", "DIRECTION_RISK_AT_R_MAG_BY_L"]
    }
    predictive = {"C_MAG": {"spearman": 0.6}, "R_MAG": {"spearman": 0.5}}

    strong = runner.scientific_classification(same_state, direction, predictive, "STRONG")
    assert strong["FINAL_CLASSIFICATION"] == "QWEN_MAG_MATCHED_RESIDUAL_FUNCTIONAL_GEOMETRY_SUPPORTED"
    assert strong["QWEN_MAG_MATCHED_FUNCTIONAL_DIRECTION_CAUSAL"] == "SUPPORTED"

    partial_direction = {name: dict(values) for name, values in direction.items()}
    for name in partial_direction:
        partial_direction[name] = {length: _cell(1.0 if length == "1" else -1.0, 18 if length == "1" else 0) for length in lengths}
    partial = runner.scientific_classification(same_state, partial_direction, predictive, "PARTIAL")
    assert partial["FINAL_CLASSIFICATION"] == "LOCAL_FUNCTIONAL_GEOMETRY_SUPPORTED_RECURRENCE_INTERACTION_REQUIRED"

    weak_same_state = {
        carrier: {length: _cell(-1.0, 0) for length in lengths}
        for carrier in ["C_NATIVE", "R_MAG_EQUALIZED"]
    }
    unsupported = runner.scientific_classification(weak_same_state, direction, predictive, "PARTIAL")
    assert unsupported["FINAL_CLASSIFICATION"] == "M5_NOT_SUPPORTED_FOR_CLOSED_LOOP_GAP"
    assert unsupported["M5_RESIDUAL_FUNCTIONAL_GEOMETRY_EXPLANATION"] == "NOT_SUPPORTED_FOR_CLOSED_LOOP_GAP"


class _FakeRealRC:
    @staticmethod
    def q_for_readout(_torch, record):
        return record["q"]

    @staticmethod
    def readout_from_state(_torch, q, residual):
        return residual * q


class _FakeFrozen:
    @staticmethod
    def implementation_replay(record, initial_state):
        assert initial_state is record["initial_state"]
        return record["o"], None


class _FakeM5:
    @staticmethod
    def apply_linear_parts(_torch, _model, _layer, record, o, readout, stage):
        assert torch.equal(o, record["o"][:, 0])
        return readout * (2.0 if stage == "G2" else 3.0)


def test_canonical_functional_metrics_use_one_context_and_global_scope():
    records = {
        0: {"q": torch.tensor([2.0]), "o": torch.ones(1, 1), "initial_state": object()},
        1: {"q": torch.tensor([1.0]), "o": torch.ones(1, 1), "initial_state": object()},
    }
    residual = {0: torch.tensor([3.0]), 1: torch.tensor([4.0])}

    metrics = runner.canonical_functional_metrics(
        torch, object(), _FakeM5, _FakeFrozen, _FakeRealRC, records, residual
    )

    assert metrics["M0"] == pytest.approx(5.0)
    assert metrics["M4"] == pytest.approx((12.0 ** 2 + 8.0 ** 2) ** 0.5)
    assert metrics["M5"] == pytest.approx((18.0 ** 2 + 12.0 ** 2) ** 0.5)
    assert metrics["F4"] == pytest.approx(metrics["M4"] / 5.0)
    assert metrics["F5"] == pytest.approx(metrics["M5"] / 5.0)


def _valid_result_rows(unit_id="u0", histories=(1,)):
    rows = []
    for length in histories:
        for config in runner.branch_configs():
            row = {
                "task": runner.TASK,
                "git_commit": "commit",
                "manifest_sha256": "manifest",
                "replay_runner_sha256": runner.EXPECTED_REPLAY_SHA256,
                "m5_source_sha256": runner.EXPECTED_M5_SHA256,
                "runner_sha256": "runner",
                "unit_id": unit_id,
                "prompt": unit_id,
                "t0": 1,
                "history_length": length,
                "carrier": "FP",
                "branch": config["branch"],
                "source_direction": config["source_direction"],
                "source_magnitude_family": config["source_magnitude_family"],
                "raw_residual_norm": 0.0 if config["branch"] == "FP" else 1.0,
                "M0": 0.0 if config["branch"] == "FP" else 1.0,
                "M4": 0.0 if config["branch"] == "FP" else 2.0,
                "M5": 0.0 if config["branch"] == "FP" else 3.0,
                "F4": 0.0 if config["branch"] == "FP" else 2.0,
                "F5": 0.0 if config["branch"] == "FP" else 3.0,
                "future_kl_auc": 0.0,
                "direction_cosine": 1.0 if config["branch"] != "FP" else None,
                "norm_matching_error": 0.0,
                "future_quantization_count": 0,
            }
            for horizon in runner.FUTURE_KL_HORIZONS:
                row[f"KL_h{horizon}"] = 0.0
            rows.append(row)
    return rows


def test_validate_unit_result_rejects_missing_horizon_and_wrong_source_hash():
    same_state_rows = [
        {
            "unit_id": "u0",
            "history_length": 1,
            "carrier": carrier,
            "active_timestep": 10,
            "source_direction": direction,
            "context_signature": f"{carrier}-state",
            "M0": 1.0,
            "M4": 2.0,
            "M5": 3.0,
            "F4": 2.0,
            "F5": 3.0,
        }
        for carrier in ["C_NATIVE", "R_MAG_EQUALIZED"]
        for direction in ["R", "C"]
    ]
    factorial_rows = [
        {
            "unit_id": "u0",
            "history_length": 1,
            "active_timestep": 10,
            "branch": branch,
            "source_state_signature": "fp-state",
            "direction_cosine": 1.0,
            "norm_matching_error": 0.0,
            "M0": 1.0,
            "M4": 2.0,
            "M5": 3.0,
            "F4": 2.0,
            "F5": 3.0,
        }
        for branch in ["RR", "RC", "CR", "CC"]
    ]
    audits = [
        {
            "unit_id": "u0",
            "t0": 10,
            "history_length": 1,
            "branch": branch,
            "quantization_event_timesteps": [10],
            "future_forward_horizons": list(range(1, 129)),
            "future_quantization_count": 0,
            "current_logit_nonretroactivity_max_abs": 0.0,
        }
        for branch in ["RR", "RC", "CR", "CC"]
    ]
    result = {
        "task": runner.TASK,
        "stage": "stage0",
        "manifest_sha256": "manifest",
        "runner_sha256": "runner",
        "git_commit": "commit",
        "unit": {"unit_id": "u0"},
        "rows": _valid_result_rows(),
        "protocol": {"history_lengths": [1], "branches": ["FP", "RR", "RC", "CR", "CC"], "horizon": 128},
        "same_state_rows": same_state_rows,
        "factorial_rows": factorial_rows,
        "audits": audits,
    }
    assert runner.validate_unit_result(result, {"unit_id": "u0"}, "stage0", [1], "manifest", "runner", "commit") == []

    missing = dict(result, rows=[dict(row) for row in result["rows"]])
    missing["rows"][1].pop("KL_h128")
    assert "missing_or_nonfinite_KL_h128" in runner.validate_unit_result(
        missing, {"unit_id": "u0"}, "stage0", [1], "manifest", "runner", "commit"
    )

    wrong = dict(result, rows=[dict(row) for row in result["rows"]])
    wrong["rows"][1]["m5_source_sha256"] = "wrong"
    assert "row_m5_source_sha256" in runner.validate_unit_result(
        wrong, {"unit_id": "u0"}, "stage0", [1], "manifest", "runner", "commit"
    )

    missing_trace = dict(result, factorial_rows=result["factorial_rows"][:-1])
    assert "factorial_trace_coverage" in runner.validate_unit_result(
        missing_trace, {"unit_id": "u0"}, "stage0", [1], "manifest", "runner", "commit"
    )


def test_fresh_output_verification_requires_exact_rows_and_current_files(tmp_path):
    output_by_name = {name: tmp_path / name for name in runner.REQUIRED_OUTPUT_NAMES}
    units = [{"unit_id": f"u{index}"} for index in range(18)]
    rows = _valid_result_rows(histories=runner.HISTORY_LENGTHS)
    rows = [dict(row, unit_id=unit["unit_id"]) for unit in units for row in rows]
    same_state_rows = [
        {
            "task": runner.TASK,
            "git_commit": "commit",
            "manifest_sha256": "manifest",
            "prompt": unit["unit_id"],
            "t0": 1,
            "history_length": length,
            "carrier": carrier,
            "branch": f"{carrier}_{direction}",
            "active_timestep": timestep,
            "source_direction": direction,
            "raw_residual_norm": 1.0,
            "M0": 1.0,
            "M4": 2.0,
            "M5": 3.0,
            "F4": 2.0,
            "F5": 3.0,
            "context_signature": "state",
        }
        for unit in units
        for length in runner.HISTORY_LENGTHS
        for timestep in range(length)
        for carrier in ["C_NATIVE", "R_MAG_EQUALIZED"]
        for direction in ["R", "C"]
    ]
    effect_rows = [
        {
            "task": runner.TASK,
            "git_commit": "commit",
            "manifest_sha256": "manifest",
            "prompt": unit["unit_id"],
            "t0": 1,
            "unit_id": unit["unit_id"],
            "history_length": length,
            "DIRECTION_RISK_AT_C_MAG": 0.0,
            "DIRECTION_RISK_AT_R_MAG": 0.0,
            "DIRECTION_MAIN": 0.0,
        }
        for unit in units
        for length in runner.HISTORY_LENGTHS
    ]
    output_by_name["results.jsonl"].write_text("".join(f"{row}\n" for row in rows))
    output_by_name["aggregate_summary.json"].write_text("{}\n")
    output_by_name["report.md"].write_text("# report\n")
    runner.write_csv(output_by_name["same_state_f5.csv"], same_state_rows)
    runner.write_csv(output_by_name["direction_factorial.csv"], [row for row in rows if row["branch"] != "FP"])
    runner.write_csv(output_by_name["history_direction_effect.csv"], effect_rows)
    outputs = list(output_by_name.values())
    merge_started = min(path.stat().st_mtime for path in outputs) - 0.01

    status, checks = runner.fresh_output_verification(
        rows, units, runner.HISTORY_LENGTHS, outputs, merge_started
    )
    assert status == "PASS"
    assert all(checks.values())

    status, checks = runner.fresh_output_verification(
        rows[:-1], units, runner.HISTORY_LENGTHS, outputs, merge_started
    )
    assert status == "FAIL"
    assert not checks["exact_result_coverage"]


def test_fresh_output_verification_rejects_malformed_csv(tmp_path):
    output_by_name = {name: tmp_path / name for name in runner.REQUIRED_OUTPUT_NAMES}
    units = [{"unit_id": f"u{index}"} for index in range(18)]
    rows = _valid_result_rows(histories=runner.HISTORY_LENGTHS)
    rows = [dict(row, unit_id=unit["unit_id"]) for unit in units for row in rows]
    for path in output_by_name.values():
        path.write_text("fresh\n")
    with output_by_name["direction_factorial.csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["unit_id", "history_length"])
        writer.writeheader()
        writer.writerow({"unit_id": "u0", "history_length": 1})

    status, checks = runner.fresh_output_verification(
        rows, units, runner.HISTORY_LENGTHS, list(output_by_name.values()),
        min(path.stat().st_mtime for path in output_by_name.values()) - 0.01,
    )

    assert status == "FAIL"
    assert not checks["direction_factorial_csv"]


def test_same_state_summary_reduces_time_within_unit_before_inference():
    rows = []
    for unit, r_values, c_values in [
        ("u0", [4.0, 2.0], [1.0, 1.0]),
        ("u1", [2.0, 2.0], [1.0, 1.0]),
    ]:
        for timestep, (r_value, c_value) in enumerate(zip(r_values, c_values)):
            for direction, value in [("R", r_value), ("C", c_value)]:
                rows.append({
                    "unit_id": unit,
                    "history_length": 2,
                    "carrier": "C_NATIVE",
                    "active_timestep": timestep,
                    "source_direction": direction,
                    "F5": value,
                })
    summary, cells = runner.summarize_same_state_f5(rows)
    assert len(cells) == 2
    assert sorted(cell["F5_R_MINUS_C"] for cell in cells) == pytest.approx([1.0, 2.0])
    assert summary["C_NATIVE"]["2"]["n"] == 2
    assert summary["C_NATIVE"]["2"]["median"] == pytest.approx(1.5)


def test_pilot_decision_is_positive_for_any_prespecified_signal():
    same_state = {
        "C_NATIVE": {"1": _cell(-1.0, 0, 6)},
        "R_MAG_EQUALIZED": {"1": _cell(-1.0, 0, 6)},
    }
    direction = {
        "DIRECTION_RISK_AT_C_MAG_BY_L": {"1": _cell(-1.0, 0, 6), "8": _cell(-1.0, 0, 6)},
        "DIRECTION_RISK_AT_R_MAG_BY_L": {"1": _cell(-1.0, 0, 6), "8": _cell(-1.0, 0, 6)},
    }
    predictive = {"C_MAG": {"spearman": -0.2}, "R_MAG": {"spearman": 0.4}}
    amplification = {"C_MAG": -0.4, "R_MAG": -0.1}
    assert runner.pilot_decision(same_state, direction, predictive, amplification) == "POSITIVE_SUPPORTS_FORMAL"
    predictive["R_MAG"]["spearman"] = -0.2
    assert runner.pilot_decision(same_state, direction, predictive, amplification) == "NEGATIVE_OR_INCONCLUSIVE"


def test_manifest_errors_reject_assignment_and_source_hash_changes():
    expected = {
        "task": runner.TASK,
        "stage": "formal",
        "git_commit": "abc",
        "shard_id": 0,
        "num_shards": 2,
        "assigned_unit_ids": ["u0"],
        "history_lengths": runner.HISTORY_LENGTHS,
        "branches": ["FP", "RR", "RC", "CR", "CC"],
        "horizon": 128,
        "manifest_sha256": runner.EXPECTED_MANIFEST_SHA256,
        "runner_sha256": "runner",
        "replay_runner_sha256": runner.EXPECTED_REPLAY_SHA256,
        "m5_source_sha256": runner.EXPECTED_M5_SHA256,
        "model_config_sha256": runner.EXPECTED_MODEL_CONFIG_SHA256,
        "tokenizer_config_sha256": runner.EXPECTED_TOKENIZER_CONFIG_SHA256,
        "tokenizer_sha256": runner.EXPECTED_TOKENIZER_SHA256,
    }
    assert runner.manifest_errors(dict(expected), expected) == []
    wrong = dict(expected, assigned_unit_ids=["u1"], m5_source_sha256="wrong")
    assert runner.manifest_errors(wrong, expected) == ["assigned_unit_ids", "m5_source_sha256"]
