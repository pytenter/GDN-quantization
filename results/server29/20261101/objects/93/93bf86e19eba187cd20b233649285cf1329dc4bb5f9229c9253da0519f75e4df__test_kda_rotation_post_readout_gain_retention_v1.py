import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_kda_rotation_post_readout_gain_retention_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("kda_post_readout_gain_retention_v1", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readout_perturbation_uses_canonical_kda_state_semantics():
    r = load_runner()
    err = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    q = torch.tensor([[[5.0, 6.0]]])

    got = r.readout_perturbation(err, q)
    expected = torch.einsum("bhkv,bhk->bhv", err, q)

    assert got.shape == (1, 1, 2)
    assert torch.equal(got, expected)


def test_normalized_stage_risk_and_gain_are_scale_safe():
    r = load_runner()
    fp = torch.tensor([3.0, 4.0])
    native = torch.tensor([0.3, 0.4])
    rotated = torch.tensor([0.15, 0.2])

    rn = r.normalized_stage_risk(fp, native)
    rr = r.normalized_stage_risk(fp, rotated)
    gain = r.stage_gain(rn, rr)

    assert rn == pytest.approx(0.01)
    assert rr == pytest.approx(0.0025)
    assert gain == pytest.approx(0.75)


def test_delta_g_and_operator_transfer_metrics():
    r = load_runner()

    assert r.delta_gain(0.25, 0.60) == pytest.approx(-0.35)
    transfer = r.operator_transfer(
        prev_native=torch.tensor([2.0, 0.0]),
        next_native=torch.tensor([4.0, 0.0]),
        prev_rotated=torch.tensor([1.0, 0.0]),
        next_rotated=torch.tensor([3.0, 0.0]),
    )

    assert transfer["g_native"] == pytest.approx(2.0)
    assert transfer["g_rotated"] == pytest.approx(3.0)
    assert transfer["Lambda"] == pytest.approx(1.5)


def test_static_kda_post_readout_audit_order_names_real_stages():
    r = load_runner()
    audit = r.build_static_kda_post_readout_audit()
    names = [stage["stage_name"] for stage in audit["stages"]]

    assert names[:5] == [
        "core_readout",
        "o_norm_gated_rms",
        "head_merge",
        "o_proj",
        "attention_residual_add",
    ]
    assert names[-2:] == ["model_norm", "lm_head_logits"]
    assert audit["KDA_POST_READOUT_PATH_AUDIT"] == "PASS"


def test_matched_pair_provenance_allows_only_readout_delta_difference():
    r = load_runner()
    fp = {"unit_id": "u0", "t": 64, "token_id": 7, "layer": 2, "q_sha256": "same", "cache_sha256": "same"}
    native = dict(fp, branch="native", delta_o_sha256="n")
    rotated = dict(fp, branch="rotated", delta_o_sha256="r")

    gate = r.check_matched_pair_provenance(native, rotated)

    assert gate["gate"] == "PASS"
    assert gate["differing_fields"] == ["branch", "delta_o_sha256"]


def test_matched_pair_provenance_rejects_query_or_cache_drift():
    r = load_runner()
    native = {"unit_id": "u0", "t": 64, "token_id": 7, "layer": 2, "q_sha256": "a", "cache_sha256": "same", "branch": "native"}
    rotated = {"unit_id": "u0", "t": 64, "token_id": 7, "layer": 2, "q_sha256": "b", "cache_sha256": "same", "branch": "rotated"}

    gate = r.check_matched_pair_provenance(native, rotated)

    assert gate["gate"] == "FAIL"
    assert "q_sha256" in gate["differing_fields"]


def test_zero_perturbation_noninterference_and_cache_metadata():
    r = load_runner()
    fp_stages = {"core_readout": torch.randn(1, 1, 2, 3), "logits": torch.randn(1, 5)}
    zero_stages = {k: v.clone() for k, v in fp_stages.items()}

    out = r.zero_perturbation_noninterference(fp_stages, zero_stages)

    assert out["gate"] == "PASS"
    assert out["max_abs_error"] == 0.0
    assert r.cache_modification_gate(before_sha256="abc", after_sha256="abc") == "PASS"
    assert r.cache_modification_gate(before_sha256="abc", after_sha256="def") == "FAIL"
