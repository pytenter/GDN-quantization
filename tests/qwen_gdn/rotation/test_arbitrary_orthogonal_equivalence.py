import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[3] / "experiments/shared/rotation/orthogonal_matrix_generator.py"
SPEC = importlib.util.spec_from_file_location("orthogonal_matrix_generator", MODULE)
ROT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROT)


def test_dense_generation_is_deterministic_and_in_so128():
    for seed in (0, 1, 2):
        left = ROT.dense_so_matrix(seed)
        right = ROT.dense_so_matrix(seed)
        assert torch.equal(left, right)
        audit = ROT.matrix_audit(f"R_seed{seed}", left, seed)
        assert audit["orthogonality_gate"] == "PASS"
        assert audit["max_abs_rt_r_minus_i"] <= 1e-12
        assert audit["determinant"] > 0.0


def test_forward_inverse_are_explicit_and_roundtrip():
    value = torch.randn((2, 3, 128), dtype=torch.float32)
    for rotation in (ROT.OrthogonalRotation128.identity(), ROT.OrthogonalRotation128.hadamard(), ROT.OrthogonalRotation128.dense(0)):
        recovered = rotation.inverse(rotation.forward(value, axis=-1), axis=-1)
        assert torch.allclose(recovered, value, atol=2e-6, rtol=2e-6)


def test_dense_inverse_is_transpose_not_forward():
    rotation = ROT.OrthogonalRotation128.dense(1)
    assert not torch.equal(rotation.matrix_fp64, rotation.matrix_fp64.transpose(0, 1))
    value = torch.randn((4, 128), dtype=torch.float32)
    correct = rotation.inverse(rotation.forward(value))
    incorrect = rotation.forward(rotation.forward(value))
    assert torch.allclose(correct, value, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(incorrect, value, atol=1e-3, rtol=1e-3)
