import importlib.util
import csv
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_persistent_source_error_magnitude_equalization_causal_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_persistent_source_mag_eq", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_closed_loop_intervention_uses_same_state_gamma_magnitude_and_direction():
    runner = load_runner()
    e_r = {0: torch.tensor([[[[3.0, 4.0]]]])}
    e_c = {0: torch.tensor([[[[0.0, 10.0]]]])}

    out = runner.build_gamma_intervention(e_r, e_c, "R_DIR", 0.25)

    assert abs(out["m_R_same_state"] - 5.0) < 1e-6
    assert abs(out["m_C_same_state"] - 10.0) < 1e-6
    assert abs(out["m_target"] - 6.25) < 1e-6
    assert abs(runner.stack_norm(out["delta"]) - 6.25) < 1e-6
    assert runner.stack_cos(out["delta"], e_r) > 1.0 - 1e-6
    assert out["APPLIED_MAGNITUDE_GATE"] == "PASS"
    assert out["SOURCE_DIRECTION_PRESERVATION_GATE"] == "PASS"


def test_closed_loop_contrasts_keep_signed_over_rescue_and_ratios():
    runner = load_runner()
    rows = [
        {"unit_id": "u", "history_length": 64, "source_direction": "R_DIR", "gamma": "0", "future_kl_auc": 0.10, "state_error_t0": 1.0},
        {"unit_id": "u", "history_length": 64, "source_direction": "R_DIR", "gamma": "1", "future_kl_auc": 0.25, "state_error_t0": 2.0},
        {"unit_id": "u", "history_length": 64, "source_direction": "C_DIR", "gamma": "0", "future_kl_auc": 0.08, "state_error_t0": 1.2},
        {"unit_id": "u", "history_length": 64, "source_direction": "C_DIR", "gamma": "1", "future_kl_auc": 0.20, "state_error_t0": 2.5},
    ]

    out = runner.closed_loop_contrasts(rows)

    assert abs(out["NATIVE_GAP"] - 0.10) < 1e-12
    assert abs(out["C_MAG_EQ_RESCUE"] - 0.12) < 1e-12
    assert abs(out["R_CMAG_GRAFT_DAMAGE"] - 0.15) < 1e-12
    assert abs(out["C_EQ_VS_R_NATIVE"] + 0.02) < 1e-12
    assert abs(out["R_GRAFT_VS_C_NATIVE"] - 0.05) < 1e-12
    assert abs(out["C_GAP_CLOSURE_RATIO"] - 1.2) < 1e-12
    assert abs(out["R_GRAFT_TO_GAP_RATIO"] - 1.5) < 1e-12
    assert abs(out["STATE_ERROR_RESCUE"] - 1.3) < 1e-12
    assert abs(out["STATE_ERROR_GRAFT_DAMAGE"] - 1.0) < 1e-12


def test_spearman_and_dose_response_detect_monotonic_gamma_effects():
    runner = load_runner()
    rows = [
        {"unit_id": "u", "history_length": 8, "source_direction": "C_DIR", "gamma": g, "future_kl_auc": auc}
        for g, auc in zip(["0", "0.25", "0.5", "0.75", "1"], [0.1, 0.2, 0.3, 0.4, 0.5])
    ]

    out = runner.dose_response(rows, "C_DIR")

    assert abs(out["median_spearman_rho"] - 1.0) < 1e-12
    assert out["monotonic_unit_count"] == 1
    assert out["n_units"] == 1
    assert abs(out["endpoint_effect"]["median"] - 0.4) < 1e-12


def test_merge_gate_requires_all_shards_to_pass():
    runner = load_runner()
    shards = [
        {"APPLIED_MAGNITUDE_GATE": "PASS"},
        {"APPLIED_MAGNITUDE_GATE": "PASS"},
    ]

    assert runner.merge_gate(shards, "APPLIED_MAGNITUDE_GATE") == "PASS"
    shards[1]["APPLIED_MAGNITUDE_GATE"] = "FAIL"
    assert runner.merge_gate(shards, "APPLIED_MAGNITUDE_GATE") == "FAIL"


def test_merge_stage_files_combines_outputs(tmp_path):
    runner = load_runner()
    shard_dirs = [tmp_path / "shard0", tmp_path / "shard1"]
    for i, shard in enumerate(shard_dirs):
        shard.mkdir()
        write_csv(shard / "unit_summary_pilot.csv", [{"unit_id": f"u{i}", "history_length": "1"}])
        write_csv(shard / "horizon_pilot.csv", [{"unit_id": f"u{i}", "history_length": "1"}])
        write_csv(shard / "per_step_residual_schedule_pilot.csv", [{"unit_id": f"u{i}", "history_length": "1"}])
        write_csv(shard / "state_error_trajectory_pilot.csv", [{"unit_id": f"u{i}", "history_length": "1"}])
        (shard / "results_pilot.jsonl").write_text(json.dumps({"unit_id": f"u{i}"}) + "\n", encoding="utf-8")
        (shard / "pilot_run_summary.json").write_text(json.dumps({"APPLIED_MAGNITUDE_GATE": "PASS"}), encoding="utf-8")

    merged = runner.merge_stage_files("pilot", shard_dirs, tmp_path / "out")

    assert [r["unit_id"] for r in merged["unit_rows"]] == ["u0", "u1"]
    assert [r["unit_id"] for r in merged["horizon_rows"]] == ["u0", "u1"]
    assert len((tmp_path / "out" / "results_pilot.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "out" / "pilot_run_summary_shard0.json").exists()
    assert (tmp_path / "out" / "pilot_run_summary_shard1.json").exists()


def test_aggregate_summary_preserves_gamma_dose_rows(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    monkeypatch.setattr(runner, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path / "reports")
    runner.RESULT_DIR.mkdir()
    runner.REPORT_DIR.mkdir()

    unit_rows = [{"unit_id": "u", "history_length": "1", "source_direction": "FP", "gamma": "", "future_kl_auc": "0", "state_error_t0": "0"}]
    for source_direction in ["R_DIR", "C_DIR"]:
        for gamma, auc in zip(["0", "0.25", "0.5", "0.75", "1"], ["0.1", "0.2", "0.3", "0.4", "0.5"]):
            unit_rows.append({
                "unit_id": "u",
                "history_length": "1",
                "source_direction": source_direction,
                "gamma": gamma,
                "future_kl_auc": auc,
                "state_error_t0": auc,
            })
    write_csv(tmp_path / "unit_summary_formal.csv", unit_rows)
    write_csv(tmp_path / "horizon_formal.csv", [{"unit_id": "u", "history_length": "1"}])
    write_csv(tmp_path / "state_error_trajectory_formal.csv", [{"unit_id": "u", "history_length": "1"}])
    write_csv(tmp_path / "per_step_residual_schedule_formal.csv", [
        {
            "source_direction": "R_DIR",
            "gamma": "0",
            "endpoint_identity_abs_error": "0",
            "SAME_STATE_COUNTERFACTUAL_GATE": "PASS",
            "APPLIED_MAGNITUDE_GATE": "PASS",
            "SOURCE_DIRECTION_PRESERVATION_GATE": "PASS",
            "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
            "degenerate_residual_count": "0",
        },
        {
            "source_direction": "C_DIR",
            "gamma": "1",
            "endpoint_identity_abs_error": "0",
            "SAME_STATE_COUNTERFACTUAL_GATE": "PASS",
            "APPLIED_MAGNITUDE_GATE": "PASS",
            "SOURCE_DIRECTION_PRESERVATION_GATE": "PASS",
            "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
            "degenerate_residual_count": "0",
        },
    ])

    summary = runner.aggregate("formal")

    assert len(summary["GAMMA_DOSE_ROWS"]) == 2
    assert {row["source_direction"] for row in summary["GAMMA_DOSE_ROWS"]} == {"R_DIR", "C_DIR"}
