import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_kda_rotation_closed_loop_driver_drift_causal_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("kda_closed_loop_driver_drift_v1", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_value_side_restore_v_transform_uses_u_transpose_semantics():
    r = load_runner()
    u = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    v_native = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)

    got = r.driver_to_branch_coordinates("v", v_native, "rotated", u)

    assert torch.equal(got, v_native.matmul(u))
    assert torch.equal(r.driver_to_semantic_coordinates("v", got, "rotated", u), v_native)


def test_non_value_drivers_remain_in_semantic_coordinates_for_rotated_branch():
    r = load_runner()
    u = torch.flip(torch.eye(4), dims=[0])
    q = torch.randn(1, 1, 2, 4)

    assert torch.equal(r.driver_to_branch_coordinates("q", q, "rotated", u), q)
    assert torch.equal(r.driver_to_semantic_coordinates("k", q, "rotated", u), q)


def test_restore_plan_modifies_only_requested_driver_and_is_persistent():
    r = load_runner()
    plan = r.RestorationPlan(name="restore-v", drivers=("v",), start_horizon=1)
    branch = {"q": torch.ones(1), "k": torch.ones(1) * 2, "v": torch.ones(1) * 3}
    fp = {"q": torch.ones(1) * 10, "k": torch.ones(1) * 20, "v": torch.ones(1) * 30}

    out_h1, prov_h1 = r.apply_driver_restoration(branch, fp, plan, horizon=1)
    out_h8, prov_h8 = r.apply_driver_restoration(branch, fp, plan, horizon=8)
    out_h0, prov_h0 = r.apply_driver_restoration(branch, fp, plan, horizon=0)

    assert torch.equal(out_h1["v"], fp["v"])
    assert torch.equal(out_h1["q"], branch["q"])
    assert torch.equal(out_h8["v"], fp["v"])
    assert torch.equal(out_h0["v"], branch["v"])
    assert prov_h1["restored_drivers"] == ["v"]
    assert prov_h1["non_restored_drivers"] == ["k", "q"]
    assert prov_h0["restored_drivers"] == []


def test_futurekl_rescue_and_delta_rescue_formulas():
    r = load_runner()
    row = r.restoration_rescue_row("restore-q", baseline_native=0.40, baseline_rotated=0.38, restored_native=0.35, restored_rotated=0.25)

    assert row["Rescue_N"] == pytest.approx(0.05)
    assert row["Rescue_R"] == pytest.approx(0.13)
    assert row["DeltaRescue"] == pytest.approx(0.08)


def test_effective_a_and_b_match_kda_synthetic_transition():
    r = load_runner()
    state = torch.arange(1, 1 + 1 * 1 * 2 * 2, dtype=torch.float64).reshape(1, 1, 2, 2)
    k = torch.tensor([[[0.25, 0.5]]], dtype=torch.float64)
    v = torch.tensor([[[2.0, -1.0]]], dtype=torch.float64)
    beta = torch.tensor([[[0.4, 0.6]]], dtype=torch.float64)
    decay = torch.tensor([[[0.9, 0.8]]], dtype=torch.float64)

    got = r.apply_effective_kda_update(state, k, v, beta, decay)
    expected = state * decay.unsqueeze(-1) + torch.einsum("...k,...v->...kv", k * beta, v)
    a = r.effective_a_operator(k, beta, decay)
    b = r.effective_b_update(k, beta, v)

    assert torch.allclose(got, expected)
    assert torch.allclose(torch.einsum("...kl,...lv->...kv", a, state), state * decay.unsqueeze(-1))
    assert torch.allclose(b, torch.einsum("...k,...v->...kv", k * beta, v))


def test_effective_b_broadcasts_ling_per_head_beta_over_key_axis():
    r = load_runner()
    k = torch.tensor([[[0.25, 0.5], [0.1, 0.2], [0.3, 0.4]]], dtype=torch.float64)
    v = torch.tensor([[[2.0, -1.0], [0.5, 1.5], [-2.0, 3.0]]], dtype=torch.float64)
    beta = torch.tensor([[0.4, 0.6, 0.8]], dtype=torch.float64)

    b = r.effective_b_update(k, beta, v)

    assert torch.allclose(b, torch.einsum("...k,...v->...kv", k * beta.unsqueeze(-1), v))


def test_pgr_identity_and_energy_identity_close():
    r = load_runner()
    fp_next = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    fp_on_q_prev = torch.tensor([[[[1.5, 2.0], [3.0, 5.0]]]])
    q_pre = torch.tensor([[[[1.6, 2.2], [2.8, 5.1]]]])
    q_next = torch.tensor([[[[1.4, 2.4], [2.7, 5.4]]]])

    parts = r.pgr_decomposition(fp_next, fp_on_q_prev, q_pre, q_next)

    assert parts["identity_max_abs_error"] == pytest.approx(0.0)
    assert parts["identity_relative_l2_error"] == pytest.approx(0.0)
    assert parts["energy_identity_relative_error"] < 1e-6


def test_zero_restoration_branch_reproduces_baseline_schedule():
    r = load_runner()
    branch = {"q": torch.randn(2), "v": torch.randn(2)}
    fp = {"q": torch.randn(2), "v": torch.randn(2)}
    plan = r.RestorationPlan(name="baseline", drivers=(), start_horizon=1)

    out, prov = r.apply_driver_restoration(branch, fp, plan, horizon=16)

    assert all(torch.equal(out[k], branch[k]) for k in branch)
    assert prov["restored_drivers"] == []
    assert prov["provenance_gate"] == "PASS"


def test_select_unit_subset_supports_two_gpu_striding():
    r = load_runner()
    units = [{"unit_id": str(i)} for i in range(18)]

    left = r.select_unit_subset(units, shard_index=0, shard_count=2)
    right = r.select_unit_subset(units, shard_index=1, shard_count=2)

    assert [u["unit_id"] for u in left] == [str(i) for i in range(0, 18, 2)]
    assert [u["unit_id"] for u in right] == [str(i) for i in range(1, 18, 2)]
    assert sorted(int(u["unit_id"]) for u in left + right) == list(range(18))


def test_completed_unit_ids_require_primary_horizon_all_branches(tmp_path):
    r = load_runner()
    path = tmp_path / "horizon_rescue" / "horizon_rescue.jsonl"
    path.parent.mkdir()
    branches = [s["branch"] for s in r.branch_specs(include_restorations=True)]
    for branch in branches:
        r.append_jsonl(path, {"unit_id": "done", "horizon": r.PRIMARY_HORIZON, "branch": branch})
    for branch in branches[:-1]:
        r.append_jsonl(path, {"unit_id": "partial", "horizon": r.PRIMARY_HORIZON, "branch": branch})

    got = r.completed_unit_ids_from_output(tmp_path)

    assert got == {"done"}


def test_quantize_branch_cache_can_update_inference_mode_cache(monkeypatch):
    r = load_runner()

    with torch.inference_mode():
        state = torch.ones(1, 1, 2, 2)
    cache = {"state": state}

    class FakePersistent:
        @staticmethod
        def get_cache_state(cache_obj, layer):
            assert layer == 0
            return cache_obj["state"]

        @staticmethod
        def fake_quant_ling_state(x, mode):
            assert mode == "INT8_R128"
            return x * 2.0, {"mode": mode}

    monkeypatch.setattr(r, "P", lambda: FakePersistent)

    meta = r.quantize_branch_cache(torch, cache, [0], "native", torch.eye(2))

    assert meta["finite"] is True
    assert torch.equal(cache["state"], torch.ones_like(state) * 2.0)


def test_full_logit_metrics_is_self_contained(monkeypatch):
    r = load_runner()

    def fail_p():
        raise AssertionError("full_logit_metrics should not depend on imported persistent runner")

    monkeypatch.setattr(r, "P", fail_p)
    ref = torch.tensor([[[2.0, 0.0, -1.0]]])
    other = torch.tensor([[[1.5, 0.5, -1.0]]])

    metrics = r.full_logit_metrics(torch, ref, other)

    assert metrics["KL"] > 0.0
    assert metrics["top1_match"] == pytest.approx(1.0)
    assert 0.0 <= metrics["top20_overlap"] <= 1.0
