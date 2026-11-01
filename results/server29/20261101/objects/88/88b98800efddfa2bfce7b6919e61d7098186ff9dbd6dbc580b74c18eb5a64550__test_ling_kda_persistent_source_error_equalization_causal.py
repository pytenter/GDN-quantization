import importlib.util
import csv
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_persistent_source_error_equalization_causal_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_persistent_source_equalization", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_sequence_branches_match_per_pulse_global_norm_and_direction():
    runner = load_runner()
    e_r = {
        0: torch.tensor([[[[3.0, 0.0]]]]),
        1: torch.tensor([[[[4.0, 0.0]]]]),
    }
    e_c = {
        0: torch.tensor([[[[0.0, 6.0]]]]),
        1: torch.tensor([[[[0.0, 8.0]]]]),
    }

    out = runner.build_sequence_branches(e_r, e_c)

    assert out["SEQUENCE_NORM_SCHEDULE_GATE"] == "PASS"
    assert out["SEQUENCE_DIRECTION_GATE"] == "PASS"
    assert out["REAL_SOURCE_BRANCH_IDENTITY_GATE"] == "PASS"
    assert abs(runner.stack_norm(out["deltas"]["C_DIR_R_MAG_SEQ"]) - runner.stack_norm(e_r)) < 1e-6
    assert abs(runner.stack_norm(out["deltas"]["R_DIR_C_MAG_SEQ"]) - runner.stack_norm(e_c)) < 1e-6
    assert runner.stack_cos(out["deltas"]["C_DIR_R_MAG_SEQ"], e_c) > 1.0 - 1e-6
    assert runner.stack_cos(out["deltas"]["R_DIR_C_MAG_SEQ"], e_r) > 1.0 - 1e-6


def test_temporal_energy_metrics_separates_constructive_and_destructive_terms():
    runner = load_runner()
    pulses = [
        {0: torch.tensor([1.0, 0.0])},
        {0: torch.tensor([1.0, 0.0])},
    ]
    constructive = runner.temporal_energy_metrics(pulses)
    assert abs(constructive["individual_propagated_energy"] - 2.0) < 1e-12
    assert abs(constructive["total_sequence_energy"] - 4.0) < 1e-12
    assert abs(constructive["temporal_cross_term"] - 2.0) < 1e-12
    assert abs(constructive["temporal_coherence_ratio"] - 1.0) < 1e-12

    destructive = runner.temporal_energy_metrics([
        {0: torch.tensor([1.0, 0.0])},
        {0: torch.tensor([-1.0, 0.0])},
    ])
    assert abs(destructive["individual_propagated_energy"] - 2.0) < 1e-12
    assert abs(destructive["total_sequence_energy"]) < 1e-12
    assert abs(destructive["temporal_cross_term"] + 2.0) < 1e-12
    assert abs(destructive["temporal_coherence_ratio"] + 1.0) < 1e-12


def test_history_length_contrasts_use_expected_branch_formulas():
    runner = load_runner()
    row = runner.sequence_contrasts("60|64", "60", 0, 64, 8, {
        "FP": 0.0,
        "R_NATIVE_SEQ": 0.10,
        "C_NATIVE_SEQ": 0.30,
        "C_DIR_R_MAG_SEQ": 0.15,
        "R_DIR_C_MAG_SEQ": 0.22,
    })

    assert abs(row["NATIVE_GAP"] - 0.20) < 1e-12
    assert abs(row["C_MAG_EQUALIZATION_RESCUE"] - 0.15) < 1e-12
    assert abs(row["R_CMAG_GRAFT_EFFECT"] - 0.12) < 1e-12
    assert abs(row["D_RMAG"] - 0.05) < 1e-12
    assert abs(row["D_CMAG"] - 0.08) < 1e-12
    assert abs(row["M_MAIN"] - 0.135) < 1e-12
    assert abs(row["D_MAIN"] - 0.065) < 1e-12


def test_l1_branch_mapping_matches_previous_single_pulse_names():
    runner = load_runner()

    assert runner.L1_SINGLE_PULSE_BRANCH_MAP == {
        "R_NATIVE_SEQ": "AUC_RR",
        "C_DIR_R_MAG_SEQ": "AUC_CR",
        "R_DIR_C_MAG_SEQ": "AUC_RC",
        "C_NATIVE_SEQ": "AUC_CC",
    }


def test_additive_injection_uses_float32_sum_before_cache_write():
    runner = load_runner()
    state = torch.tensor([1.0], dtype=torch.bfloat16)
    delta = torch.tensor([0.002], dtype=torch.float32)
    expected = (state.float() + delta).to(torch.bfloat16)

    actual = runner.cache_add_float32_then_cast(state.clone(), delta)

    assert torch.equal(actual, expected)


def test_cache_write_helper_accepts_inference_tensors():
    runner = load_runner()
    with torch.inference_mode():
        state = torch.zeros(2, dtype=torch.bfloat16)
    delta = torch.tensor([0.25, -0.5], dtype=torch.float32)

    runner.copy_add_float32_then_cast_(state, delta)

    assert torch.equal(state, delta.to(torch.bfloat16))


def test_capture_record_clones_tensor_payloads_to_cpu():
    runner = load_runner()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = {
        0: {
            "q": torch.ones(1, device=device),
            "initial_state": torch.ones(1, device=device),
            "final_state": torch.zeros(1, device=device),
            "operator": "fused_recurrent_kda",
        }
    }

    cloned = runner.clone_capture_records(records)

    assert cloned[0]["q"].device.type == "cpu"
    assert cloned[0]["initial_state"].device.type == "cpu"
    assert cloned[0]["final_state"].device.type == "cpu"
    assert cloned[0]["q"].data_ptr() != records[0]["q"].data_ptr()


def test_select_units_by_ids_preserves_manifest_order():
    runner = load_runner()
    units = [{"unit_id": "a"}, {"unit_id": "b"}, {"unit_id": "c"}]

    selected = runner.select_units_by_ids(units, "c,a")

    assert [u["unit_id"] for u in selected] == ["a", "c"]


def test_stack_state_metrics_can_skip_expensive_cosine():
    runner = load_runner()
    diff = {0: torch.tensor([1.0, 0.0])}
    ref = {0: torch.tensor([1.0, 0.0])}

    metrics = runner.stack_delta_metrics(diff, ref, include_cosine=False)

    assert metrics["state_error_norm"] == 1.0
    assert abs(metrics["state_relative_L2"] - 1.0) < 1e-9
    assert metrics["state_cosine"] is None


def test_temporal_superposition_gate_uses_relative_and_scaled_absolute_tolerance():
    runner = load_runner()

    assert runner.temporal_superposition_gate(5.7e-4, 7.2e-5) == "PASS"
    assert runner.temporal_superposition_gate(1.2e-3, 7.2e-5) == "FAIL"
    assert runner.temporal_superposition_gate(5.7e-4, 1.2e-4) == "FAIL"


def test_merge_blocking_gates_requires_all_shards_to_pass():
    runner = load_runner()
    shards = [
        {"SOURCE_ERROR_IDENTITY_GATE": "PASS", "SEQUENCE_DIRECTION_GATE": "PASS"},
        {"SOURCE_ERROR_IDENTITY_GATE": "PASS", "SEQUENCE_DIRECTION_GATE": "PASS"},
    ]

    assert runner.merge_gate(shards, "SOURCE_ERROR_IDENTITY_GATE") == "PASS"
    assert runner.merge_gate(shards, "SEQUENCE_DIRECTION_GATE") == "PASS"

    shards[1]["SEQUENCE_DIRECTION_GATE"] = "FAIL"
    assert runner.merge_gate(shards, "SEQUENCE_DIRECTION_GATE") == "FAIL"


def test_summarize_merged_stage_uses_shard_gates_and_recomputed_temporal_gate():
    runner = load_runner()
    shards = [{
        "SOURCE_ERROR_IDENTITY_GATE": "PASS",
        "SEQUENCE_NORM_SCHEDULE_GATE": "PASS",
        "SEQUENCE_DIRECTION_GATE": "PASS",
        "REAL_SOURCE_BRANCH_IDENTITY_GATE": "PASS",
        "INTERVENTION_TIMING_GATE": "PASS",
        "CURRENT_LOGIT_IDENTITY_GATE": "PASS",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "L1_REPRODUCTION_GATE": "PASS",
        "TEMPORAL_SUPERPOSITION_GATE": "FAIL",
        "max_source_identity_abs_error": 0.0,
        "max_source_identity_relative_L2_error": 0.0,
        "max_sequence_norm_schedule_error": 0.0,
        "min_same_R_direction_cosine": 1.0,
        "min_same_C_direction_cosine": 1.0,
    }]
    unit_rows = [{
        "unit_id": "synthetic",
        "history_length": 1,
        "AUC_FP": 0.0,
        "L1_REPRODUCTION_SOURCE": "VALIDATED_SINGLE_PULSE_RUNNER",
        "max_p_only_superposition_abs_error": 5.7e-4,
        "max_p_only_superposition_relative_error": 7.2e-5,
    }]

    summary = runner.summarize_merged_stage("pilot", unit_rows, [], shards, [1], 128, [])

    assert summary["SOURCE_ERROR_IDENTITY_GATE"] == "PASS"
    assert summary["TEMPORAL_SUPERPOSITION_GATE"] == "PASS"
    assert summary["PILOT"] == "NEGATIVE_OR_INCONCLUSIVE"

    shards[0]["SEQUENCE_DIRECTION_GATE"] = "FAIL"
    summary = runner.summarize_merged_stage("pilot", unit_rows, [], shards, [1], 128, [])
    assert summary["SEQUENCE_DIRECTION_GATE"] == "FAIL"
    assert summary["PILOT"] == "FAIL"


def test_merge_stage_files_combines_shard_tables_and_jsonl(tmp_path):
    runner = load_runner()
    shard_dirs = [tmp_path / "shard0", tmp_path / "shard1"]
    for i, shard in enumerate(shard_dirs):
        shard.mkdir()
        write_csv(shard / "sequence_horizon_pilot.csv", [{"unit_id": f"u{i}", "history_length": "1"}])
        write_csv(shard / "sequence_unit_auc_pilot.csv", [{"unit_id": f"u{i}", "history_length": "1", "AUC_FP": "0"}])
        write_csv(shard / "temporal_composition_pilot.csv", [{"unit_id": f"u{i}", "history_length": "1", "branch": "R_NATIVE_SEQ"}])
        write_csv(shard / "temporal_coherence_pilot.csv", [{"unit_id": f"u{i}", "history_length": "1", "branch": "R_NATIVE_SEQ"}])
        (shard / "results_pilot.jsonl").write_text(json.dumps({"unit_id": f"u{i}"}) + "\n", encoding="utf-8")
        (shard / "pilot_run_summary.json").write_text(json.dumps({"SOURCE_ERROR_IDENTITY_GATE": "PASS"}), encoding="utf-8")

    merged = runner.merge_stage_files("pilot", shard_dirs, tmp_path / "out")

    assert [r["unit_id"] for r in merged["unit_rows"]] == ["u0", "u1"]
    assert [r["unit_id"] for r in merged["horizon_rows"]] == ["u0", "u1"]
    assert len((tmp_path / "out" / "results_pilot.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "out" / "pilot_run_summary_shard0.json").exists()
    assert (tmp_path / "out" / "pilot_run_summary_shard1.json").exists()
