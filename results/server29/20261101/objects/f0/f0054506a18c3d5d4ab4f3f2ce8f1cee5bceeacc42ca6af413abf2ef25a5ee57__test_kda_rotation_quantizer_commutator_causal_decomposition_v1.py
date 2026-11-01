import importlib.util
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_rotation_quantizer_commutator_causal_decomposition_v1.py"
SPEC = importlib.util.spec_from_file_location("commutator", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def orthogonal(n=4):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(7)))
    return q


def state():
    return torch.randn(1, 2, 3, 4, generator=torch.Generator().manual_seed(11))


def synthetic_stage_a():
    rows = []
    for i in range(18):
        for lam in M.LAMBDAS:
            rows.append({"unit_id": f"u{i:02d}", "condition": M.lambda_name(lam), "future_kl_auc": 0.1 + lam * (1 + i / 100)})
    return rows


def synthetic_stage_b():
    rows = []
    values = {"NN": 0.0, "NR": 0.1, "RN": 0.9, "RR": 1.0}
    for i in range(18):
        for condition, value in values.items():
            rows.append({"unit_id": f"u{i:02d}", "condition": condition, "future_kl_auc": value + i * 1e-5})
    return rows


def test_01_quantizer_semantics_are_formal_r128():
    q = M.quantizer_semantics()
    assert q["GROUP_AXIS"] == "V" and q["GROUP_SIZE"] == 128 and q["QMIN"] == -127 and q["QMAX"] == 127


def test_02_manual_quantization_reconstructs_official():
    x = state()
    official, meta = M.official_quantize(x)
    manual, _ = M.quantize_with_scale(x, meta["scale"])
    assert torch.equal(official, manual)


def test_03_scale_shape_is_strict():
    with pytest.raises(ValueError):
        M.quantize_with_scale(state(), torch.ones(1, 2, 1, 1))


def test_04_fp_rotation_inverse_identity():
    x, r = state(), orthogonal()
    assert torch.allclose(M.inverse_rotate_state(M.rotate_state(x, r), r), x, atol=1e-5)


def test_05_nn_reproduces_native():
    e = M.construct_endpoints(state(), orthogonal())
    assert torch.equal(e["NN"], e["NN_official"])


def test_06_rr_reproduces_rotated_mapback():
    e = M.construct_endpoints(state(), orthogonal())
    assert torch.allclose(e["RR"], e["RR_official"], atol=1e-6)


def test_07_lambda_zero_reproduces_nn():
    e = M.construct_endpoints(state(), orthogonal())
    assert torch.equal(M.interpolate_endpoint(e["NN"], e["RR"], 0), e["NN"])


def test_08_lambda_one_reproduces_rr():
    e = M.construct_endpoints(state(), orthogonal())
    assert torch.equal(M.interpolate_endpoint(e["NN"], e["RR"], 1), e["RR"])


def test_09_lambda_interpolation_shape_dtype():
    e = M.construct_endpoints(state(), orthogonal())
    z = M.interpolate_endpoint(e["NN"], e["RR"], 0.5)
    assert z.shape == e["NN"].shape and z.dtype == e["NN"].dtype


def test_10_nr_uses_rotated_scale_in_native_coordinates():
    e = M.construct_endpoints(state(), orthogonal())
    expected, _ = M.quantize_with_scale(e["S"], e["rotated_meta"]["scale"])
    assert torch.equal(e["NR"], expected)


def test_11_rn_mapback_semantics():
    e = M.construct_endpoints(state(), orthogonal())
    rotated, _ = M.quantize_with_scale(e["SR"], e["native_meta"]["scale"])
    assert torch.allclose(e["RN"], M.inverse_rotate_state(rotated, orthogonal()), atol=1e-6)


def test_12_input_is_not_modified():
    x, before = state(), state().clone()
    M.construct_endpoints(x, orthogonal())
    assert torch.equal(x, before)


def test_13_median_bootstrap_reproducible_and_targets_median():
    x = [0.0] * 17 + [1000.0]
    assert M.median_bootstrap_ci(x, n=500, seed=9) == [0.0, 0.0]


def test_14_unitwise_horizon_aggregation():
    rows = []
    for unit in ("a", "b"):
        for h in range(1, 4):
            rows.append({"unit_id": unit, "condition": "L0", "future_kl": h})
    out = M.aggregate_units(rows)
    assert len(out) == 2 and all(r["future_kl_auc"] == 2.0 for r in out)


def test_15_linear_slope():
    assert M.linear_slope([0, .25, .5, .75, 1], [0, .25, .5, .75, 1]) == pytest.approx(1.0)


def test_16_stage_a_strong_support():
    audit = {k: "PASS" for k in ("NATIVE_ENDPOINT_IDENTITY", "ROTATED_ENDPOINT_IDENTITY", "LAMBDA_0_ENDPOINT_IDENTITY", "LAMBDA_1_ENDPOINT_IDENTITY")}
    _, summary = M.summarize_stage_a(synthetic_stage_a(), audit)
    assert summary["COMMUTATOR_CAUSAL_STATUS"] == "STRONG_SUPPORT" and summary["positive_slope_units"] == 18


def test_17_stage_b_lattice_classification():
    _, summary = M.summarize_stage_b(synthetic_stage_b())
    assert summary["PRIMARY_MECHANISM"] == "ROTATED_QUANTIZATION_LATTICE_OR_ERROR_GEOMETRY"


def test_18_no_nonfinite_endpoints():
    e = M.construct_endpoints(state(), orthogonal())
    assert all(torch.isfinite(e[k]).all() for k in ("NN", "NR", "RN", "RR"))
