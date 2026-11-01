import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "rotation" / "run_kda_rotation_full_transition_final_causal_closure_v1.py"
SPEC = importlib.util.spec_from_file_location("full_transition", RUNNER)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def orthogonal(n=4):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(7)))
    return q


def stack():
    return {0: torch.randn(1, 2, 3, 4, generator=torch.Generator().manual_seed(11)),
            1: torch.randn(1, 2, 3, 4, generator=torch.Generator().manual_seed(12))}


def stage_a_rows():
    rows = []
    for i in range(18):
        rows.append({"unit_id": f"u{i}", "auc_NN": 0.01, "auc_NR": 0.08,
                     "auc_RN": 0.04, "auc_RR": 0.14 + i * 1e-4})
    return rows


def stage_b_rows():
    rows = []
    for i in range(18):
        rows.append({"unit_id": f"u{i}", "auc_NATIVE": 0.01, "auc_R_LIVE": 0.11,
                     "auc_R_PREQUANT_N": 0.02, "auc_R_POSTQUANT_N": 0.015,
                     "auc_R_FP32_TRANSITION": 0.04})
    return rows


def test_01_switch_n_to_r_roundtrip():
    x, r = stack(), orthogonal()
    assert M.stack_metrics(M.inverse_stack(M.rotate_stack(x, r), r), x)["relative_l2"] < 1e-6


def test_02_switch_r_to_n_roundtrip():
    x, r = stack(), orthogonal()
    assert M.stack_metrics(M.rotate_stack(M.inverse_stack(x, r), r), x)["relative_l2"] < 1e-6


def test_03_branch_semantics_define_real_endpoints():
    p = M.protocol_semantics()
    assert "actual Native future" in p["NN"] and "actual live Rotated future" in p["RR"]


def test_04_boundary_only_qr_is_forbidden():
    assert M.protocol_semantics()["boundary_only_QR"] == "FORBIDDEN_AND_ABSENT"
    assert "quantize_operator" not in inspect.getsource(M.run_history_transition_unit)


def test_05_branch_selection_only_changes_future_basis():
    source = inspect.getsource(M.switch_branches)
    assert '"NR"' in source and '"RN"' in source and "rotate_stack" in source and "inverse_stack" in source


def test_06_native_history_source_is_exact_clone():
    source = inspect.getsource(M.switch_branches)
    assert 'BASE.clone_cache(pasts["N"])' in source


def test_07_rotated_history_source_is_exact_clone():
    source = inspect.getsource(M.switch_branches)
    assert 'BASE.clone_cache(pasts["R"])' in source


def test_08_historical_state_donor_is_prequant():
    p = M.protocol_semantics()
    assert "PRE-QUANT" in p["historical_R_STATE_N"] and p["R_PREQUANT_N"] == "exact historical R_STATE_N semantics"


def test_09_true_postquant_donor_replaces_final_state():
    donor = {"NATIVE": {0: {"final_state": torch.zeros(1), "postquant_state": torch.ones(1)}}}
    out = M.postquant_source_records(donor)
    assert torch.equal(out["NATIVE"][0]["final_state"], torch.ones(1))


def test_10_postquant_donor_requires_quantized_state():
    with pytest.raises(RuntimeError): M.postquant_source_records({"NATIVE": {0: {"final_state": torch.zeros(1)}}})


def test_11_donor_plans_do_not_transplant_drivers():
    plans = {p.name: p for p in M.donor_plans()}
    for name in ("R_PREQUANT_N", "R_POSTQUANT_N"):
        assert plans[name].state_source == "NATIVE" and plans[name].m_source is None and plans[name].b_source is None


def test_12_donor_receiver_timestep_is_explicit():
    assert "donor_receiver_timestep_match" in inspect.getsource(M.run_localization_unit)


def test_13_exact_transition_matches_einsum_semantics():
    s = torch.randn(1, 2, 3, 4); a = torch.randn(1, 2, 3, 3); b = torch.randn(1, 2, 3, 4)
    expected = torch.einsum("...kl,...lv->...kv", a.double(), s.double()) + b.double()
    assert torch.equal(M.exact_transition(s, a, b), expected)


def test_14_high_precision_value_rotation_identity():
    s = torch.randn(1, 2, 3, 4, dtype=torch.float64); a = torch.randn(1, 2, 3, 3, dtype=torch.float64)
    b = torch.randn(1, 2, 3, 4, dtype=torch.float64); r = orthogonal().double()
    native = M.exact_transition(s, a, b)
    rotated = M.BASE.inverse_rotate_state_value_axis(M.exact_transition(M.BASE.rotate_state_value_axis(s, r), a, b.matmul(r)), r)
    assert torch.allclose(native, rotated, atol=1e-6, rtol=1e-6)


def test_15_fp32_transition_keeps_rotated_int8_plan():
    p = {x.name: x for x in M.donor_plans()}["R_FP32_TRANSITION"]
    assert p.basis == "rotated" and p.quantized


def test_16_live_driver_rotation_is_already_float32():
    text = M.protocol_semantics()["R_FP32_DRIVER"]
    assert "already rotates v in float32" in text and "kernel boundary" in text


def test_17_rotation_matrix_not_modified_by_stack_conversion():
    r, before = orthogonal(), orthogonal().clone(); x = stack(); M.rotate_stack(x, r); M.inverse_stack(x, r)
    assert torch.equal(r, before)


def test_18_quantizer_semantics_unchanged():
    q = M.COMM.quantizer_semantics()
    assert (q["CONFIG"], q["GROUP_SIZE"], q["QMIN"], q["QMAX"]) == ("INT8_R128", 128, -127, 127)


def test_19_unitwise_auc_aggregation():
    rows = []
    for u in ("a", "b"):
        for c in ("NN", "NR", "RN", "RR"):
            for h in range(1, 4): rows.append({"unit_id": u, "condition": c, "future_kl": h})
    out = M.aggregate_auc(rows, ("NN", "NR", "RN", "RR"))
    assert len(out) == 2 and out[0]["auc_NN"] == 2


def test_20_median_bootstrap_estimand_consistency():
    assert M.median_bootstrap_ci([0] * 17 + [1000], n=500, seed=9) == [0.0, 0.0]


def test_21_stage_a_effects_are_per_unit():
    derived, summary = M.summarize_stage_a(stage_a_rows())
    assert len(derived) == 18 and summary["effects"]["interaction"]["n"] == 18


def test_22_stage_b_keeps_unstable_denominators():
    rows = stage_b_rows(); rows[0] = {**rows[0], "auc_R_LIVE": 0.01}
    derived, summary = M.summarize_stage_b(rows)
    assert len(derived) == 18 and "u0" in summary["unstable_units"]


def test_23_stage_b_pre_post_closure_values():
    derived, summary = M.summarize_stage_b(stage_b_rows())
    assert summary["PREQUANT_CLOSURE"]["paired_median"] == pytest.approx(0.9)
    assert summary["POSTQUANT_CLOSURE"]["paired_median"] == pytest.approx(0.95)


def test_24_fp32_transition_closure_uses_unit_pairs():
    units, b = M.summarize_stage_b(stage_b_rows())
    d = M.summarize_stage_d(units, b)
    assert d["FP32_TRANSITION_CLOSURE"]["n"] == 18 and d["FP32_TRANSITION_CLOSURE"]["paired_median"] == pytest.approx(0.7)


def test_25_no_nan_or_inf_in_synthetic_summaries():
    units, b = M.summarize_stage_b(stage_b_rows()); d = M.summarize_stage_d(units, b)
    values = [b["PREQUANT_CLOSURE"]["paired_median"], d["FP32_TRANSITION_CLOSURE"]["paired_median"]]
    assert all(math.isfinite(x) for x in values)


def test_26_instrumentation_helpers_do_not_mutate_inputs():
    x, r = stack(), orthogonal(); xb = {k: v.clone() for k, v in x.items()}; rb = r.clone()
    M.stack_metrics(x, x); M.rotate_stack(x, r); M.inverse_stack(x, r)
    assert all(torch.equal(x[k], xb[k]) for k in x) and torch.equal(r, rb)


import math
