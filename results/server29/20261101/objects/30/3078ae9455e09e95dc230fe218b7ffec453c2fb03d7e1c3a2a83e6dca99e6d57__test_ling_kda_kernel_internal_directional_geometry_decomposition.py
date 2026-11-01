import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "experiments" / "ling" / "pytorch_kda_reference.py"
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_kernel_internal_directional_geometry_decomposition_v1.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_step_reference_exposes_named_kernel_checkpoints():
    ref = load_module(REF, "pytorch_kda_reference")
    q = torch.tensor([[[[3.0, 4.0]]]])
    k = torch.tensor([[[[1.0, 0.0]]]])
    v = torch.tensor([[[[2.0, -1.0]]]])
    g = torch.zeros(1, 1, 1, 2)
    beta = torch.ones(1, 1, 1)
    state = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)

    out = ref.kda_one_step_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
    )

    assert list(out.output.shape) == [1, 1, 1, 2]
    assert list(out.final_state.shape) == [1, 1, 2, 2]
    assert {"K0_INPUT_ERROR", "K1_DECAYED_ERROR", "K2_UPDATE_ERROR", "K3_READOUT_ERROR", "K4_CORE_OUTPUT"}.issubset(
        ref.kernel_stage_ids()
    )


def test_exact_state_error_transfer_matches_finite_difference():
    ref = load_module(REF, "pytorch_kda_reference")
    torch.manual_seed(7)
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 1, 4)
    v = torch.randn(1, 1, 1, 3)
    g = -torch.rand(1, 1, 1, 4)
    beta = torch.rand(1, 1, 1)
    state = torch.randn(1, 1, 4, 3)
    err = torch.randn_like(state) * 0.01

    fd = ref.kda_one_step_reference(q, k, v, g, beta, initial_state=state + err).output - ref.kda_one_step_reference(
        q, k, v, g, beta, initial_state=state
    ).output
    transfer = ref.kda_exact_state_error_transfer(q, k, v, g, beta, err).stage_tensors["K4_CORE_OUTPUT"]

    assert torch.linalg.vector_norm(fd.float() - transfer.float()) / torch.linalg.vector_norm(fd.float()) < 1e-4


def test_phase3_gate_requires_strong_localization_and_parity():
    runner = load_module(RUNNER, "kernel_runner")

    assert runner.phase3_gate("PASS", "PASS", "STRONG", "PASS", "K2_UPDATE_ERROR") == "RUN"
    assert runner.phase3_gate("FAIL", "PASS", "STRONG", "PASS", "K2_UPDATE_ERROR") == "NOT_RUN"
    assert runner.phase3_gate("PASS", "PASS", "PARTIAL", "PASS", "K2_UPDATE_ERROR") == "NOT_RUN"
    assert runner.phase3_gate("PASS", "PASS", "STRONG", "FAIL", "K2_UPDATE_ERROR") == "NOT_RUN"


def test_earliest_stage_and_layer22_reverse_helpers():
    runner = load_module(RUNNER, "kernel_runner")
    rows = [
        {"stage_id": "K0_INPUT_ERROR", "N_SIGN_CONCORDANT": 10, "LAYER22_REVERSE_CONTROL": "FAIL"},
        {"stage_id": "K1_DECAYED_ERROR", "N_SIGN_CONCORDANT": 17, "LAYER22_REVERSE_CONTROL": "FAIL"},
        {"stage_id": "K2_UPDATE_ERROR", "N_SIGN_CONCORDANT": 18, "LAYER22_REVERSE_CONTROL": "PASS"},
    ]

    assert runner.earliest_stage(rows, 17) == "K1_DECAYED_ERROR"
    assert runner.earliest_stage(rows, 18) == "K2_UPDATE_ERROR"
    assert runner.earliest_layer22_reverse(rows) == "K2_UPDATE_ERROR"


def test_layer_sign_concordance_keeps_layer22_as_reverse_control():
    runner = load_module(RUNNER, "kernel_runner")
    layer_rows = [
        {"stage_id": "K2_UPDATE_ERROR", "layer_id": 0, "delta_G": 0.2},
        {"stage_id": "K2_UPDATE_ERROR", "layer_id": 21, "delta_G": 0.1},
        {"stage_id": "K2_UPDATE_ERROR", "layer_id": 22, "delta_G": -0.3},
    ]
    prior = {0: "C_SENSITIVE", 21: "C_SENSITIVE", 22: "R_SENSITIVE"}

    row = runner.kernel_stage_layer_sign_concordance(layer_rows, prior)[0]

    assert row["N_SIGN_CONCORDANT"] == 3
    assert row["DISCORDANT_LAYER_IDS"] == ""
    assert row["LAYER22_REVERSE_CONTROL"] == "PASS"


def test_source_provenance_gate_requires_hashes_for_local_runtime_files(tmp_path):
    runner = load_module(RUNNER, "kernel_runner")
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1\n", encoding="utf-8")

    assert runner.local_source_provenance_gate([a]) == "PASS"
    assert runner.local_source_provenance_gate([a, b]) == "FAIL"
