import hashlib
import importlib.util
import csv
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_persistent_error_decomposition_causal_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_persistent_decomp", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decompose_components_satisfy_exact_identity():
    runner = load_runner()
    fp_next = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    fp_on_q_prev = torch.tensor([[[[1.5, 1.0], [2.0, 5.0]]]])
    q_pre = torch.tensor([[[[2.0, 1.25], [2.5, 4.0]]]])
    q_next = torch.tensor([[[[2.25, 1.0], [2.25, 4.5]]]])

    parts = runner.decompose_error(fp_next, fp_on_q_prev, q_pre, q_next)

    assert torch.equal(parts["P"], fp_on_q_prev - fp_next)
    assert torch.equal(parts["G"], q_pre - fp_on_q_prev)
    assert torch.equal(parts["R"], q_next - q_pre)
    assert torch.equal(parts["E"], q_next - fp_next)
    assert torch.allclose(parts["E"], parts["P"] + parts["G"] + parts["R"])
    assert parts["identity"]["max_abs_error"] == 0.0
    assert parts["identity"]["relative_L2_error"] == 0.0


def test_component_metrics_record_delta_r_energy_and_pairwise_terms():
    runner = load_runner()
    fp_state = torch.ones(1, 1, 1, 4)
    P = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    G = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])
    R = torch.tensor([[[[-0.5, -0.5, 0.0, 0.0]]]])
    E = P + G + R

    row = runner.component_metrics(E=E, P=P, G=G, R=R, fp_state=fp_state)

    assert row["E_norm"] == torch.linalg.vector_norm(E).item()
    assert row["B_norm"] == torch.linalg.vector_norm(P + G).item()
    assert row["delta_r_energy"] == torch.linalg.vector_norm(E).item() ** 2 - torch.linalg.vector_norm(P + G).item() ** 2
    assert row["r_cancellation_flag"] is True
    assert row["dot_P_G"] == 0.0
    assert row["dot_P_R"] == -0.5
    assert row["dot_G_R"] == -0.5
    assert row["energy_identity_abs_error"] < 1e-7


def test_component_metrics_record_leave_one_out_and_requested_energy_names():
    runner = load_runner()
    fp_state = torch.ones(1, 1, 1, 4)
    P = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    G = torch.tensor([[[[0.0, 2.0, 0.0, 0.0]]]])
    R = torch.tensor([[[[-0.25, -0.5, 0.0, 0.0]]]])
    E = P + G + R

    row = runner.component_metrics(E=E, P=P, G=G, R=R, fp_state=fp_state)

    assert row["predicted_total_energy"] == row["TOTAL_predicted"]
    assert row["actual_total_energy"] == row["TOTAL_actual"]
    assert row["energy_identity_error"] == row["energy_identity_abs_error"]
    assert row["loo_P"] == torch.linalg.vector_norm(E).item() ** 2 - torch.linalg.vector_norm(G + R).item() ** 2
    assert row["loo_G"] == torch.linalg.vector_norm(E).item() ** 2 - torch.linalg.vector_norm(P + R).item() ** 2
    assert row["loo_R"] == torch.linalg.vector_norm(E).item() ** 2 - torch.linalg.vector_norm(P + G).item() ** 2
    assert row["loo_R"] < 0
    assert row["R_error_role"] == "CANCELLATION"


def test_all_kda_all_head_units_expand_to_each_kda_layer():
    runner = load_runner()
    unit = {"layer": "ALL_KDA", "head": "ALL"}

    assert runner.iter_unit_layer_heads(unit, [0, 2, 4]) == [(0, "ALL"), (2, "ALL"), (4, "ALL")]


def test_quantizer_scale_shapes_match_ling_bhkv_semantics():
    runner = load_runner()
    state = torch.arange(1, 1 + 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)

    _r_qdq, r_meta = runner.fake_quant_ling_state(state, "INT8_R128")
    _c_qdq, c_meta = runner.fake_quant_ling_state(state, "INT8_C128")

    assert list(r_meta["scale"].shape) == [1, 2, 3, 1]
    assert r_meta["scale_shape_semantics"] == "[B,H,K,1]"
    assert list(c_meta["scale"].shape) == [1, 2, 1, 4]
    assert c_meta["scale_shape_semantics"] == "[B,H,1,V]"


def test_missing_canonical_manifest_blocks_formal_status(tmp_path):
    runner = load_runner()

    audit = runner.load_canonical_units(tmp_path / "missing.json")

    assert audit["status"] == "MISSING"
    assert audit["formal_units"] == []
    assert audit["gate"] == "FAIL"


def test_canonical_manifest_gate_validates_hash_and_fixed_unit_set(tmp_path):
    runner = load_runner()
    manifest = {
        "units": [
            {
                "unit_id": f"{pid}|{t0}",
                "problem_id": pid,
                "problem_index": idx,
                "t0": t0,
                "future_horizon": 128,
                "layer": "ALL_KDA",
                "head": "ALL",
            }
            for pid, idx in [("60", 0), ("61", 1), ("64", 4), ("68", 8), ("69", 9), ("76", 16)]
            for t0 in [64, 128, 256]
        ]
    }
    path = tmp_path / "canonical_units.json"
    path.write_text(runner.json.dumps(manifest), encoding="utf-8")
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    audit = runner.load_canonical_units(path, expected_sha256=expected_hash)

    assert audit["gate"] == "PASS"
    assert audit["sha256"] == expected_hash
    assert audit["prompt_set"] == ["60", "61", "64", "68", "69", "76"]
    assert audit["t0_set"] == [64, 128, 256]
    assert audit["future_horizon_set"] == [128]


def test_aggregate_writes_layer_summary_and_final_gate_fields(tmp_path):
    runner = load_runner()
    runner.RUN_DIR = tmp_path
    runner.RESULT_DIR = tmp_path
    path = tmp_path / "decomposition_horizon_formal.csv"
    rows = []
    for unit_id in ["60|64"]:
        for orientation, e_norm, p_norm, g_norm, r_norm, loo_r in [
            ("R128", 1.0, 0.4, 0.2, 0.8, -0.1),
            ("C128", 2.0, 1.2, 0.3, 0.9, -0.2),
        ]:
            for layer in [0, 1]:
                rows.append({
                    "unit_id": unit_id,
                    "orientation": orientation,
                    "prompt_id": "60",
                    "problem_index": "0",
                    "layer": str(layer),
                    "head": "ALL",
                    "t0": "64",
                    "horizon": "1",
                    "E_norm": str(e_norm),
                    "P_norm": str(p_norm),
                    "G_norm": str(g_norm),
                    "R_norm": str(r_norm),
                    "B_norm": "1.0",
                    "E_rel": str(e_norm / 10),
                    "P_rel": str(p_norm / 10),
                    "G_rel": str(g_norm / 10),
                    "R_rel": str(r_norm / 10),
                    "B_rel": "0.1",
                    "cos_B_R": "-0.5",
                    "delta_r_energy": str(loo_r),
                    "delta_r_energy_normalized": str(loo_r),
                    "loo_P": "0.3",
                    "loo_G": "0.2",
                    "loo_R": str(loo_r),
                    "loo_P_normalized": "0.03",
                    "loo_G_normalized": "0.02",
                    "loo_R_normalized": str(loo_r / 10),
                    "dot_P_R": "-0.1",
                    "dot_G_R": "-0.1",
                    "readout_E": "0.01",
                    "readout_P": "0.02",
                    "readout_G": "0.03",
                    "readout_R": "0.04",
                    "r_cancellation_flag": "True",
                    "R_error_role": "CANCELLATION",
                    "decomp_max_abs_error": "0.0",
                    "decomp_relative_L2_error": "0.0",
                    "energy_identity_error": "0.0",
                    "fp_replay_relative_L2_error": "0.0",
                    "q_replay_relative_L2_error": "0.0",
                })
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = runner.aggregate(path)

    assert Path(summary["decomposition_layer_formal_csv"]).exists()
    assert summary["DECOMPOSITION_IDENTITY_GATE"] == "PASS"
    assert summary["PER_LAYER_DECOMPOSITION_IDENTITY_GATE"] == "PASS"
    assert summary["LOCAL_REQUANT_IDENTITY_GATE"] == "PASS"
    assert summary["R128_LOO_R"] < 0
    assert summary["C128_LOO_R"] < 0
    assert summary["REQUANTIZATION_ERROR_ROLE"] == "SYSTEMATIC_CANCELLATION_SIGNAL"


def test_reset_stage_outputs_removes_append_targets(tmp_path):
    runner = load_runner()
    runner.RUN_DIR = tmp_path
    for name in ["decomposition_horizon_stage0.csv", "decomposition_raw_stage0.jsonl"]:
        (tmp_path / name).write_text("stale\n", encoding="utf-8")

    runner.reset_stage_outputs("stage0")

    assert not (tmp_path / "decomposition_horizon_stage0.csv").exists()
    assert not (tmp_path / "decomposition_raw_stage0.jsonl").exists()
