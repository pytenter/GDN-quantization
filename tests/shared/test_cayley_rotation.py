import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "experiments/shared/rotation/cayley_rotation.py"
SPEC = importlib.util.spec_from_file_location("cayley_rotation_test_target", PATH)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def normalized_hadamard(n=128):
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / n**0.5


def test_theta_zero_is_identity_and_hadamard_path_is_unchanged():
    layer = C.CayleyDenseRotation()
    identity = torch.eye(128)
    assert torch.equal(layer.matrix(), identity)
    x = torch.randn(3, 128)
    hx = x @ normalized_hadamard()
    assert torch.equal(layer.forward_correction(hx), hx)


def test_cayley_is_orthogonal_and_proper():
    torch.manual_seed(4)
    layer = C.CayleyDenseRotation()
    with torch.no_grad():
        layer.theta.normal_(std=0.02)
    matrix = layer.matrix()
    residual = matrix.T @ matrix - torch.eye(128)
    assert residual.abs().max().item() < 1.0e-5
    assert torch.linalg.det(matrix).item() > 0


def test_inverse_forward_roundtrip_fp32():
    torch.manual_seed(5)
    layer = C.CayleyDenseRotation()
    with torch.no_grad():
        layer.theta.uniform_(-0.01, 0.01)
    x = torch.randn(2, 7, 128)
    recovered = layer.inverse_correction(layer.forward_correction(x))
    assert torch.allclose(recovered, x, atol=2e-6, rtol=2e-6)


def test_gradient_reaches_only_theta_not_frozen_model():
    torch.manual_seed(6)
    model = torch.nn.Linear(128, 32, bias=False)
    model.requires_grad_(False)
    layer = C.CayleyDenseRotation()
    x = torch.randn(2, 128)
    layer.forward_correction(x).square().mean().backward()
    assert layer.theta.grad is not None
    assert torch.isfinite(layer.theta.grad).all()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_ste_forward_exact_and_backward_identity_qwen_c128():
    x = torch.randn(2, 3, 128, 9, requires_grad=True)
    exact = C.qwen_c128_qdq(x)
    ste = C.qwen_c128_ste(x)
    assert torch.equal(ste.dequant, exact.dequant)
    assert ste.scale.shape == (2, 3, 1, 9)
    ste.dequant.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_ste_forward_exact_and_axis_semantics_ling_r128():
    x = torch.randn(2, 3, 11, 128, requires_grad=True)
    exact = C.ling_r128_qdq(x)
    ste = C.ling_r128_ste(x)
    assert torch.equal(ste.dequant, exact.dequant)
    assert ste.scale.shape == (2, 3, 11, 1)
    ste.dequant.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_layer_shared_inventory_and_runtime_gate():
    bank = C.PerLayerCayleyRotations((0, 2, 4))
    assert bank.total_trainable_parameters == 3 * 8128
    assert bank.runtime_gate()["status"] == "PASS"

