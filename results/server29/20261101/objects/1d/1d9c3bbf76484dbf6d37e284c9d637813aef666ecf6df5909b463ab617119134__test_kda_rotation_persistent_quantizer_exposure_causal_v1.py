import importlib.util
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_rotation_persistent_quantizer_exposure_causal_v1.py"
SPEC = importlib.util.spec_from_file_location("persistent_exposure", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def orthogonal(n=4):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(7)))
    return q


def state():
    return torch.randn(1, 2, 3, 4, generator=torch.Generator().manual_seed(11))


def synthetic_stage_a(kind="strong"):
    rows = []
    names = [M.exposure_name(x) for x in M.EXPOSURES]
    for i in range(18):
        row = {"unit_id": f"u{i:02d}"}
        for j, name in enumerate(names):
            if kind == "strong":
                row[f"auc_{name}"] = 0.10 if j <= 1 else 0.10 + 0.01 * (j - 1) + i * 1e-5
            else:
                row[f"auc_{name}"] = 0.10 - 0.001 * j + i * 1e-5
        rows.append(row)
    return rows


def synthetic_stage_b(strong=True):
    rows = []
    for i in range(18):
        nn = 0.10 + i * 1e-5
        nr = nn - 0.01 if strong else nn + 0.01
        rn = 0.20 + i * 1e-5
        rr = rn + 0.04 if strong else rn + 0.01
        en, er = nr - nn, rr - rn
        rows.append({"unit_id": f"u{i:02d}", "auc_NN": nn, "auc_NR": nr, "auc_RN": rn, "auc_RR": rr,
                     "E_NATIVE_HISTORY": en, "E_ROTATED_HISTORY": er,
                     "HISTORY_X_QUANTIZER_INTERACTION": er - en})
    return rows


def test_01_exposure_names_roundtrip():
    assert [M.exposure_value(M.exposure_name(x)) for x in M.EXPOSURES] == list(M.EXPOSURES)


def test_02_l0_is_always_native_quantizer():
    assert all(M.operator_for_boundary(0, j) == "QN" for j in range(1, 10))


def test_03_l1_changes_only_t0_boundary():
    assert [M.operator_for_boundary(1, j) for j in range(1, 5)] == ["QR", "QN", "QN", "QN"]


def test_04_l4_exact_exposure_counter():
    assert [M.operator_for_boundary(4, j) for j in range(1, 7)] == ["QR"] * 4 + ["QN"] * 2


def test_05_full_is_always_rotated_quantizer():
    assert all(M.operator_for_boundary("FULL", j) == "QR" for j in range(1, 100))


def test_06_qn_reproduces_official_native_endpoint():
    x = state(); actual, _ = M.quantize_operator(x, orthogonal(), "QN")
    expected, _ = M.COMM.official_quantize(x)
    assert torch.equal(actual, expected)


def test_07_qr_reproduces_official_mapped_back_endpoint():
    x, r = state(), orthogonal(); actual, _ = M.quantize_operator(x, r, "QR")
    rotated, _ = M.COMM.official_quantize(M.COMM.rotate_state(x, r))
    assert torch.allclose(actual, M.COMM.inverse_rotate_state(rotated, r), atol=1e-6)


def test_08_operator_endpoint_identity_passes():
    result = M.operator_endpoint_identity(state(), orthogonal())
    assert result["QN"]["max_abs_error"] == 0 and result["QR"]["max_abs_error"] < 1e-6


def test_09_quantizer_does_not_modify_input():
    x = state(); before = x.clone(); M.quantize_operator(x, orthogonal(), "QR")
    assert torch.equal(x, before)


def test_10_quantizer_codes_and_scale_keep_r128_shape():
    x = state(); _, meta = M.quantize_operator(x, orthogonal(), "QN")
    assert meta["codes"].shape == x.shape and meta["scale"].shape == x.shape[:-1] + (1,)


def test_11_quantizer_semantics_remain_unchanged():
    q = M.COMM.quantizer_semantics()
    assert (q["CONFIG"], q["GROUP_AXIS"], q["GROUP_SIZE"], q["QMIN"], q["QMAX"]) == ("INT8_R128", "V", 128, -127, 127)


def test_12_diagnostics_retain_raw_layer_step_fields():
    x, r = state(), orthogonal(); q, meta = M.quantize_operator(x, r, "QN")
    rows = M.diagnostic_rows("u", "L0", 0, 1, {0: x}, {0: q}, {0: meta},
                             {0: x}, {0: q}, {0: meta}, {0: x})
    required = {"prequant_state_norm", "scale_raw_json", "quant_error_relative_norm",
                "code_churn_vs_native_same_step", "postquant_state_rel_to_native"}
    assert len(rows) == 1 and required.issubset(rows[0])


def test_13_stack_hash_is_stable_and_state_sensitive():
    x = state(); assert M.stack_hash({0: x}) == M.stack_hash({0: x.clone()})
    assert M.stack_hash({0: x}) != M.stack_hash({0: x + 1})


def test_14_median_bootstrap_targets_median():
    assert M.median_bootstrap_ci([0.0] * 17 + [1000.0], n=500, seed=9) == [0.0, 0.0]


def test_15_stage_a_aggregation_is_unitwise():
    rows = []
    for unit in ("a", "b"):
        for name in [M.exposure_name(x) for x in M.EXPOSURES]:
            for h in range(1, 4): rows.append({"unit_id": unit, "condition": name, "future_kl": float(h)})
    result = M.aggregate_stage_a(rows)
    assert len(result) == 2 and all(row["auc_L0"] == 2.0 for row in result)


def test_16_stage_a_strong_support_requires_full_endpoint():
    op = {"FULL_QR_REPRODUCES_ROTATED": "PASS"}
    summary = M.summarize_stage_a(synthetic_stage_a("strong"), op)
    assert summary["PERSISTENT_EXPOSURE_CAUSAL_STATUS"] == "STRONG_SUPPORT"


def test_17_stage_a_not_supported_for_beneficial_full():
    op = {"FULL_QR_REPRODUCES_ROTATED": "PASS"}
    assert M.summarize_stage_a(synthetic_stage_a("negative"), op)["PERSISTENT_EXPOSURE_CAUSAL_STATUS"] == "NOT_SUPPORTED"


def test_18_failed_full_endpoint_prevents_classification():
    op = {"FULL_QR_REPRODUCES_ROTATED": "FAIL"}
    assert M.summarize_stage_a(synthetic_stage_a("strong"), op)["PERSISTENT_EXPOSURE_CAUSAL_STATUS"] == "NOT_SUPPORTED"


def test_19_history_depth_comes_from_first_significant_exposure():
    assert M.choose_history_depth({"FIRST_L_WITH_CI_ABOVE_ZERO": "L8"}) == 8


def test_20_stage_b_aggregation_never_resamples_horizons():
    rows = []
    for unit in ("a", "b"):
        for condition, value in {"NN": 1, "NR": 2, "RN": 3, "RR": 5}.items():
            for h in range(4): rows.append({"unit_id": unit, "condition": condition, "future_kl": value + h})
    out = M.aggregate_stage_b(rows)
    assert len(out) == 2 and out[0]["HISTORY_X_QUANTIZER_INTERACTION"] == pytest.approx(1.0)


def test_21_stage_b_strong_history_interaction():
    summary = M.summarize_stage_b(synthetic_stage_b(True), 8)
    assert summary["HISTORY_CONDITIONED_QUANTIZER_STATUS"] == "STRONG_SUPPORT"


def test_22_history_pairs_share_exact_prequant_hash():
    x, y = state(), state() + 0.1
    hashes = {"NN": M.stack_hash({0: x}), "NR": M.stack_hash({0: x}),
              "RN": M.stack_hash({0: y}), "RR": M.stack_hash({0: y})}
    assert hashes["NN"] == hashes["NR"] and hashes["RN"] == hashes["RR"]


def test_23_reconciliation_is_explicit_and_noncontradictory():
    r = M.reconciliation_payload()
    assert r["PROTOCOL_RECONCILIATION"] == "PASS" and r["PROTOCOL_CONTRADICTION"] == "NO"
    assert r["OLD_NATIVE_STATE_CEILING_INTERVENTION"] == "PERSISTENT"
    assert r["NEW_COMMUTATOR_INTERVENTION"] == "ONE_SHOT"


def test_24_no_nonfinite_operator_outputs():
    for operator in ("QN", "QR"):
        out, meta = M.quantize_operator(state(), orthogonal(), operator)
        assert torch.isfinite(out).all() and torch.isfinite(meta["scale"]).all()


def test_25_instrumentation_helpers_do_not_mutate_state_or_rotation():
    x, r = state(), orthogonal(); xb, rb = x.clone(), r.clone()
    M.operator_endpoint_identity(x, r); M.tensor_hash(x); M.stack_hash({0: x})
    assert torch.equal(x, xb) and torch.equal(r, rb)


def test_26_monotonic_score_does_not_assume_linearity():
    assert M.monotonic_score([0, 1, 1, 3]) == 1.0
    assert M.monotonic_score([0, 2, 1, 3]) == pytest.approx(2 / 3)
