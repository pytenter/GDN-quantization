#!/usr/bin/env python3
import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location("proxy", Path(__file__).with_name("run_short_horizon_proxy.py"))
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


def test_readout_and_p2():
    # One layer/head: E is identity; q is already the prepared query.
    e = torch.eye(2).reshape(1, 1, 2, 2)
    q = torch.tensor([[[1.0, 0.0]]])
    m = proxy.state_metrics(torch, e, q, 0)
    assert abs(m["state_mse"] - 0.5) < 1e-12
    assert abs(m["readout_energy"] - 0.5) < 1e-12
    cov = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    p2, aux = proxy.historical_sensitivity(torch, e, cov, 0)
    assert abs(p2 - 0.5) < 1e-12 and abs(aux - 0.5) < 1e-12


def test_zero_and_scaling():
    e = torch.zeros(2, 3, 4, 5)
    q = torch.randn(2, 3, 4)
    m = proxy.state_metrics(torch, e, q, 0)
    assert m["state_mse"] == 0 and m["readout_energy"] == 0
    e = torch.randn(2, 3, 4, 5)
    a = proxy.state_metrics(torch, e, q, 0)
    b = proxy.state_metrics(torch, 3 * e, q, 0)
    assert abs(b["state_mse"] / a["state_mse"] - 9) < 1e-5
    assert abs(b["readout_energy"] / a["readout_energy"] - 9) < 1e-5


def test_shapes_history_and_coordinates():
    torch.manual_seed(0)
    e, q = torch.randn(2, 3, 4, 5), torch.randn(2, 7, 3, 4)
    cov = torch.einsum("lthk,lthj->lhkj", q, q) / 7
    p2, _ = proxy.historical_sensitivity(torch, e, cov, 0)
    explicit = torch.einsum("lhkv,lthk->lthv", e, q).square().mean().item()
    assert abs(p2 - explicit) < 1e-6
    rotation, _ = torch.linalg.qr(torch.randn(5, 5))
    assert torch.allclose((e @ rotation) @ rotation.T, e, atol=1e-6)
    x, y = torch.randn(1, 11), torch.randn(1, 11)
    assert proxy.functional_kl(torch, x, y) == proxy.functional_kl(torch, x[:, None], y[:, None])
    assert proxy.functional_kl(torch, x, x) == 0
    assert proxy.functional_kl(torch, x, y) > 0


if __name__ == "__main__":
    test_readout_and_p2()
    test_zero_and_scaling()
    test_shapes_history_and_coordinates()
    print("CPU_PROXY_MATH_TESTS_PASS")
