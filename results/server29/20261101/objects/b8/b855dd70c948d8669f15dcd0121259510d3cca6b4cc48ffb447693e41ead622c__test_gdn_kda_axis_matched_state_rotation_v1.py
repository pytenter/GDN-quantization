import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_kda_axis_matched_state_rotation_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("axis_rotation_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rht_is_deterministic_and_orthogonal():
    runner = load_runner()

    r0a = runner.make_rotation(8, "rht", seed=0, dtype=torch.float64)
    r0b = runner.make_rotation(8, "rht", seed=0, dtype=torch.float64)
    r1 = runner.make_rotation(8, "rht", seed=1, dtype=torch.float64)

    assert torch.equal(r0a, r0b)
    assert not torch.equal(r0a, r1)
    assert runner.orthogonality_error(r0a) <= 1e-12


def test_haar_and_hadamard_fallback_are_orthogonal():
    runner = load_runner()

    haar = runner.make_rotation(7, "haar", seed=3, dtype=torch.float64)
    fallback = runner.make_rotation(6, "rht", seed=2, dtype=torch.float64)

    assert runner.orthogonality_error(haar) <= 1e-12
    assert runner.orthogonality_error(fallback) <= 1e-12


def test_rotation_roundtrip_on_key_and_value_axes():
    runner = load_runner()
    state = torch.randn(2, 3, 5, 7, dtype=torch.float64)
    r = runner.make_rotation(5, "haar", seed=0, dtype=torch.float64)
    u = runner.make_rotation(7, "haar", seed=1, dtype=torch.float64)

    key_rot = runner.rotate_state_key_axis(state, r)
    val_rot = runner.rotate_state_value_axis(state, u)

    assert torch.allclose(runner.inverse_rotate_state_key_axis(key_rot, r), state, atol=1e-12, rtol=1e-12)
    assert torch.allclose(runner.inverse_rotate_state_value_axis(val_rot, u), state, atol=1e-12, rtol=1e-12)


def test_value_axis_tensor_rotation_roundtrip():
    runner = load_runner()
    value = torch.randn(2, 3, 4, 8)
    u = runner.make_rotation(8, "haar", seed=2, dtype=torch.float32)

    rotated = runner.rotate_value_axis_tensor(value, u)

    assert torch.allclose(runner.inverse_rotate_value_axis_tensor(rotated, u), value.float(), atol=1e-5, rtol=1e-5)


def test_stage0a_fp64_algebra_parity_passes_strict_gate():
    runner = load_runner()

    result = runner.run_stage0a_fp64_parity(k_dim=8, v_dim=8, batch=2, heads=3, steps=32)

    assert result["gate"] == "PASS"
    assert result["gdn"]["max_relative_state_error"] <= 1e-10
    assert result["gdn"]["max_relative_readout_error"] <= 1e-10
    assert result["kda"]["max_relative_state_error"] <= 1e-10
    assert result["kda"]["max_relative_readout_error"] <= 1e-10
    assert result["nonfinite"] == 0


def test_quantizer_axis_semantics_match_verified_bhkv_convention():
    runner = load_runner()
    state = torch.zeros(1, 2, 3, 4)

    _rq, rmeta = runner.fake_quant_state(state, "R128")
    _cq, cmeta = runner.fake_quant_state(state, "C128")

    assert rmeta["scale_shape_semantics"] == "[B,H,K,1]"
    assert cmeta["scale_shape_semantics"] == "[B,H,1,V]"
    assert list(rmeta["scale"].shape) == [1, 2, 3, 1]
    assert list(cmeta["scale"].shape) == [1, 2, 1, 4]
    assert rmeta["group_axis"] == "V"
    assert cmeta["group_axis"] == "K"


def test_normalized_peakiness_exact_formula_and_group_metrics():
    runner = load_runner()
    group = torch.tensor([3.0, 4.0])
    assert runner.normalized_peakiness(group) == pytest.approx((2.0 ** 0.5) * 4.0 / 5.0)

    state = torch.tensor([[[[1.0, -2.0, 0.0, 4.0], [0.5, -0.5, 1.0, -1.0]]]])
    rows = runner.quantization_group_metrics(state, "R128", arch="TEST", unit_id="u", basis="identity", rotation_axis="none")

    assert len(rows) == 2
    assert all(row["normalized_peakiness"] >= 1.0 for row in rows)
    assert all(row["quant_scale"] > 0.0 for row in rows)
    assert all(0.0 <= row["code_utilization_fraction"] <= 1.0 for row in rows)


def test_rotation_l2_invariance_metrics_are_numerical_precision():
    runner = load_runner()
    state = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    rotation = runner.make_rotation(8, "rht", seed=0, dtype=torch.float64)

    errors = runner.rotation_l2_invariance_errors(state, runner.rotate_state_key_axis(state, rotation), "C128")

    assert errors["median"] <= 1e-12
    assert errors["p99"] <= 1e-12
    assert errors["max"] <= 1e-12


def test_architecture_stage_directories_are_isolated(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(runner, "FIG_DIR", tmp_path / "figures")

    gdn = runner.arch_stage_dir("gdn", "stage0b")
    kda = runner.arch_stage_dir("kda", "stage0b")

    assert gdn == tmp_path / "gdn" / "stage0b"
    assert kda == tmp_path / "kda" / "stage0b"
    assert gdn != kda


def test_cli_accepts_architecture_and_stage_arguments():
    runner = load_runner()

    args = runner.parse_args(["--architecture", "gdn", "--stage", "stage0b", "--scope", "smoke"])

    assert args.architecture == "gdn"
    assert args.stage == "stage0b"
    assert args.scope == "smoke"


def test_identity_rotation_noninterference_for_both_axes():
    runner = load_runner()
    state = torch.randn(1, 2, 4, 4)
    eye = runner.make_rotation(4, "identity", dtype=state.dtype)

    assert torch.equal(runner.transform_state(state, "key", eye), state)
    assert torch.equal(runner.transform_state(state, "value", eye), state)


def test_gdn_qk_rotation_preserves_native_l2norm_precision_boundary():
    runner = load_runner()
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    r = runner.make_rotation(8, "rht", seed=0, dtype=torch.float32)

    q_rot, k_rot, use_norm = runner.prepare_gdn_rotated_qk(q, k, r, use_qk_l2norm_in_kernel=True)
    native_q = runner.gdn_kernel_l2norm(q, dim=-1).float().matmul(r)
    native_k = runner.gdn_kernel_l2norm(k, dim=-1).float().matmul(r)

    assert use_norm is False
    assert torch.equal(q_rot, native_q)
    assert torch.equal(k_rot, native_k)


def test_stage0b_gate_uses_recurrent_parity_not_downstream_logit_diagnostic():
    runner = load_runner()
    rows = [{
        "state_relative_error": 2e-7,
        "readout_relative_error": 3e-7,
        "logit_relative_l2": 2e-2,
        "logit_cosine": 0.999,
        "top1_match": 1,
        "nonfinite": 0,
    }]

    assert runner.classify_stage0b(rows) == "PASS"
