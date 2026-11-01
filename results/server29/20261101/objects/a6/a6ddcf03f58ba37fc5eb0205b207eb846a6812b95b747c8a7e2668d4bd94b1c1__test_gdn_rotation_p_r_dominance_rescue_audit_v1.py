import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "rotation" / "run_gdn_rotation_p_r_dominance_rescue_audit_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gdn_pr_dominance_rescue_v1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_components():
    p_n = torch.tensor([[[[1.0, -2.0], [0.5, 0.25]]]], dtype=torch.float64)
    r_n = torch.tensor([[[[0.2, 0.1], [-0.1, 0.3]]]], dtype=torch.float64)
    p_r = torch.tensor([[[[1.5, -2.5], [0.25, 0.75]]]], dtype=torch.float64)
    r_r = torch.tensor([[[[0.4, -0.2], [-0.3, 0.5]]]], dtype=torch.float64)
    q = torch.tensor([[[0.6, -0.8]]], dtype=torch.float64)
    return p_n, r_n, p_r, r_r, q


def test_rescue_panel_uses_required_nn_rn_nr_rr_names():
    r = load_runner()
    p_n, r_n, p_r, r_r, _q = sample_components()

    panel = r.build_component_panel(p_n, r_n, p_r, r_r)

    assert set(panel) == {"NN", "RN", "NR", "RR"}
    assert torch.equal(panel["NN"], p_n + r_n)
    assert torch.equal(panel["RN"], p_r + r_n)
    assert torch.equal(panel["NR"], p_n + r_r)
    assert torch.equal(panel["RR"], p_r + r_r)


def test_component_restoration_changes_only_named_component():
    r = load_runner()
    p_n, r_n, p_r, r_r, _q = sample_components()
    parts = r.build_component_parts(p_n, r_n, p_r, r_r)

    assert torch.equal(parts["RN"]["P"], p_r)
    assert torch.equal(parts["RN"]["R"], r_n)
    assert torch.equal(parts["NR"]["P"], p_n)
    assert torch.equal(parts["NR"]["R"], r_r)


def test_readout_risk_and_rescue_bookkeeping():
    r = load_runner()
    p_n, r_n, p_r, r_r, q = sample_components()
    risks = r.readout_panel_risks(r.build_component_panel(p_n, r_n, p_r, r_r), q)
    effects = r.rescue_effects(risks)

    assert effects["Rescue_P"] == pytest.approx(risks["RR"] - risks["NR"])
    assert effects["Rescue_R"] == pytest.approx(risks["RR"] - risks["RN"])
    assert effects["Excess"] == pytest.approx(risks["RR"] - risks["NN"])
    assert effects["interaction"] == pytest.approx(risks["RR"] - risks["RN"] - risks["NR"] + risks["NN"])


def test_zero_excess_denominator_suppresses_fraction():
    r = load_runner()
    effects = r.rescue_effects({"NN": 1.0, "RN": 0.9, "NR": 0.8, "RR": 1.0})

    assert effects["Excess"] == pytest.approx(0.0)
    assert effects["F_P"] is None
    assert effects["F_R"] is None


def test_same_query_hash_required_for_all_conditions():
    r = load_runner()
    q = torch.randn(1, 2, 4)
    changed = q.clone()
    changed[..., 0] += 1e-3

    assert r.same_query_gate({"NN": q, "RN": q.clone(), "NR": q.clone(), "RR": q.clone()})["gate"] == "PASS"
    assert r.same_query_gate({"NN": q, "RN": changed, "NR": q.clone(), "RR": q.clone()})["gate"] == "FAIL"


def test_downstream_transform_does_not_mutate_input():
    r = load_runner()
    delta = torch.randn(1, 1, 3, 2)
    original = delta.clone()
    gate = torch.randn(1, 1, 3, 2)
    weight = torch.ones(2)
    out_proj_weight = torch.randn(5, 6)

    out = r.gdn_same_token_downstream(delta, gate, weight, 1e-6, out_proj_weight)

    assert torch.equal(delta, original)
    assert out["post_norm"].shape == delta.shape
    assert out["post_out_proj"].shape == (1, 1, 5)


def test_classification_marks_p_dominant_only_with_downstream_support():
    r = load_runner()
    t64 = {
        "horizon": 64,
        "Rescue_P_median": 3.0,
        "Rescue_P_wins": 15,
        "Rescue_R_median": 0.2,
        "Rescue_R_wins": 7,
        "Rescue_P_minus_R_median": 2.8,
        "Rescue_P_minus_R_wins": 14,
        "Rescue_P_minus_R_bootstrap95_mean": [0.5, 4.0],
    }
    downstream = {
        "post_out_proj": dict(t64),
        "logits": dict(t64),
    }

    out = r.classify_dominance([t64], downstream)

    assert out["GDN_P_VS_R_RESCUE"] == "P_DOMINANT"
    assert out["GDN_P_DOMINANCE_DOWNSTREAM"] == "SUPPORTED"
    assert out["GDN_MECHANISM_CLOSURE_STATUS"] == "SUFFICIENT_FOR_CURRENT_PAPER"
    assert out["NEXT_EXPERIMENT"] == "KDA_ROTATION_POST_READOUT_GAIN_RETENTION_V1"


def test_classification_stays_mixed_when_raw_readout_not_dominant():
    r = load_runner()
    raw = {
        "horizon": 64,
        "Rescue_P_median": 7e-4,
        "Rescue_P_wins": 10,
        "Rescue_R_median": 2e-4,
        "Rescue_R_wins": 11,
        "Rescue_P_minus_R_median": 5e-4,
        "Rescue_P_minus_R_wins": 9,
        "Rescue_P_minus_R_bootstrap95_mean": [-2e-3, 2e-3],
    }
    downstream = {
        "post_out_proj": {
            **raw,
            "Rescue_P_gt_Rescue_R_wins": 18,
            "Rescue_P_minus_R_bootstrap95_mean": [1.0, 2.0],
        },
        "logits": {
            **raw,
            "Rescue_P_gt_Rescue_R_wins": 18,
            "Rescue_P_minus_R_bootstrap95_mean": [1.0, 2.0],
        },
    }

    out = r.classify_dominance([raw], downstream)

    assert out["GDN_P_DOMINANCE_AT_READOUT"] == "PARTIAL"
    assert out["GDN_P_DOMINANCE_DOWNSTREAM"] == "SUPPORTED"
    assert out["GDN_P_VS_R_RESCUE"] == "MIXED_OR_UNRESOLVED"
    assert out["GDN_READOUT_FAILURE_DOMINANT_MECHANISM"] == "UNRESOLVED"
