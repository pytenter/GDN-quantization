import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[3] / "experiments/shared/rotation/orthogonal_matrix_generator.py"
SPEC = importlib.util.spec_from_file_location("orthogonal_matrix_generator_ling", MODULE)
ROT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROT)


def test_dense_rotation_roundtrip_uses_transpose():
    value = torch.randn((2, 7, 128), dtype=torch.float32)
    for seed in (0, 1, 2):
        rotation = ROT.OrthogonalRotation128.dense(seed)
        recovered = rotation.inverse(rotation.forward(value))
        assert torch.allclose(recovered, value, atol=2e-6, rtol=2e-6)
        assert ROT.matrix_audit(f"R_seed{seed}", rotation.matrix_fp64, seed)["orthogonality_gate"] == "PASS"


def test_dense_rotation_is_not_accidentally_self_inverse():
    rotation = ROT.OrthogonalRotation128.dense(0)
    assert not torch.equal(rotation.matrix_fp64, rotation.matrix_fp64.transpose(0, 1))
