import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_kda_rotation_exact_frozen_driver_replay_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("exact_frozen_driver_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_gdn_driver_tape_replays_deterministically():
    r = load_runner()
    tape = r.synthetic_driver_tape("gdn", steps=4, k_dim=8, v_dim=8, heads=2)
    out1 = r.replay_synthetic_fp(tape, "gdn")
    out2 = r.replay_synthetic_fp(tape, "gdn")

    assert out1["max_state_relative_error"] <= 1e-12
    assert out2["max_state_relative_error"] <= 1e-12
    assert out1["driver_tape_sha256"] == out2["driver_tape_sha256"]


def test_synthetic_kda_driver_tape_replays_deterministically():
    r = load_runner()
    tape = r.synthetic_driver_tape("kda", steps=4, k_dim=8, v_dim=8, heads=2)
    out = r.replay_synthetic_fp(tape, "kda")

    assert out["max_state_relative_error"] <= 1e-12
    assert out["driver_tape_sha256"] == r.driver_tape_hash(tape)


def test_native_and_rotated_replay_consume_identical_driver_ids():
    r = load_runner()
    tape = r.synthetic_driver_tape("gdn", steps=3, k_dim=8, v_dim=8, heads=1)
    rot = r.make_rotation(8, "rht", seed=0, dtype=torch.float64)
    native = r.replay_synthetic_quantized(tape, "gdn", branch="native", rotation=rot, quantizer_off=True)
    rotated = r.replay_synthetic_quantized(tape, "gdn", branch="rotated", rotation=rot, quantizer_off=True)

    assert r.assert_identical_driver_provenance(native["rows"], rotated["rows"])["gate"] == "PASS"
    assert max(abs(n["state_error_norm"] - rr["state_error_norm"]) for n, rr in zip(native["rows"], rotated["rows"])) <= 1e-9


def test_rotated_quantizer_off_parity_for_both_architectures():
    r = load_runner()
    for arch in ["gdn", "kda"]:
        tape = r.synthetic_driver_tape(arch, steps=5, k_dim=8, v_dim=8, heads=2)
        dim = 8
        rot = r.make_rotation(dim, "haar", seed=2, dtype=torch.float64)
        native = r.replay_synthetic_quantized(tape, arch, branch="native", rotation=rot, quantizer_off=True)
        rotated = r.replay_synthetic_quantized(tape, arch, branch="rotated", rotation=rot, quantizer_off=True)
        assert native["max_state_relative_error"] <= 1e-10
        assert rotated["max_state_relative_error"] <= 1e-10


def test_pr_identity_quantizer_off_and_fp_previous_state_identity():
    r = load_runner()
    fp = torch.randn(1, 1, 4, 4)
    pre = fp.clone()
    q = pre.clone()
    decomp = r.pr_decomposition(fp, pre, q)

    assert decomp["P_norm"] == pytest.approx(0.0)
    assert decomp["R_norm"] == pytest.approx(0.0)
    assert decomp["identity_relative_error"] <= 1e-12
    assert decomp["energy_identity_relative_error"] <= 1e-12


def test_readout_space_state_error_mapping():
    r = load_runner()
    err = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    q = torch.tensor([[[5.0, 6.0]]])
    got = r.readout_error_norm(err, q)
    expected = torch.linalg.vector_norm(torch.einsum("bhkv,bhk->bhv", err, q)).item()

    assert got == pytest.approx(expected)


def test_driver_tape_hash_supports_bfloat16_driver_tensors():
    r = load_runner()
    tape = {
        "architecture": "gdn",
        "unit_id": "bf16_hash_regression",
        "steps": [
            {
                "t": 0,
                "records": {
                    "layer0": {
                        "q": torch.randn(1, 1, 4, dtype=torch.bfloat16),
                        "k": torch.randn(1, 1, 4, dtype=torch.bfloat16),
                    }
                },
            }
        ],
    }

    first = r.driver_tape_hash(tape)
    second = r.driver_tape_hash(tape)

    assert isinstance(first, str)
    assert len(first) == 64
    assert first == second


def test_fp_replay_parity_gate_accepts_exact_zero_relative_error():
    r = load_runner()
    parity_summary = {
        "n_units_completed": 1,
        "max_state_relative_error": 0.0,
        "nonfinite": 0,
    }

    assert r.fp_replay_parity_status(parity_summary, expected_units=1, tol=1e-5) == "PASS"


def test_kda_rht_uses_validated_value_side_convention():
    r = load_runner()
    base = r.make_rotation(8, "rht", seed=0, dtype=torch.float64)
    got = r.make_experiment_rotation("kda", 8, kind="rht", seed=0, dtype=torch.float64)

    assert torch.equal(got, base.t().contiguous())
