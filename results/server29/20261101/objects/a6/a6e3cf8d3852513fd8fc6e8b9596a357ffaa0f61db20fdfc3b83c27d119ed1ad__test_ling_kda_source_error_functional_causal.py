import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_source_error_functional_causal_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_source_error", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_error_identity_and_global_norms():
    runner = load_runner()
    stack = {
        0: torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        1: torch.tensor([[[[0.5, -1.5], [2.5, -3.5]]]]),
    }

    out = runner.extract_source_errors(stack)

    assert out["SOURCE_ERROR_IDENTITY_GATE"] == "PASS"
    assert out["m_R"] > 0
    assert out["m_C"] > 0
    for layer, state in stack.items():
        assert torch.allclose(state + out["e_R"][layer], out["q_R"][layer])
        assert torch.allclose(state + out["e_C"][layer], out["q_C"][layer])


def test_factorial_branches_match_global_magnitude_and_direction():
    runner = load_runner()
    e_r = {
        0: torch.tensor([[[[3.0, 0.0]]]]),
        1: torch.tensor([[[[4.0, 0.0]]]]),
    }
    e_c = {
        0: torch.tensor([[[[0.0, 6.0]]]]),
        1: torch.tensor([[[[0.0, 8.0]]]]),
    }
    branches = runner.build_factorial_branches(e_r, e_c)

    assert branches["FACTORIAL_NORM_GATE"] == "PASS"
    assert branches["FACTORIAL_DIRECTION_GATE"] == "PASS"
    assert branches["REAL_SOURCE_BRANCH_IDENTITY_GATE"] == "PASS"
    assert torch.allclose(branches["deltas"]["R_DIR@R_MAG"][0], e_r[0])
    assert torch.allclose(branches["deltas"]["C_DIR@C_MAG"][1], e_c[1])
    assert abs(runner.stack_norm(branches["deltas"]["C_DIR@R_MAG"]) - branches["m_R"]) < 1e-6
    assert abs(runner.stack_norm(branches["deltas"]["R_DIR@C_MAG"]) - branches["m_C"]) < 1e-6


def test_factorial_contrasts_use_per_unit_auc_not_horizon_rows():
    runner = load_runner()
    auc = {
        "FP": 0.0,
        "R_DIR@R_MAG": 0.10,
        "C_DIR@R_MAG": 0.15,
        "R_DIR@C_MAG": 0.20,
        "C_DIR@C_MAG": 0.30,
    }

    row = runner.factorial_contrasts("60|64", "60", 0, 64, auc, 2.0, 4.0)

    assert abs(row["REAL_GAP"] - 0.20) < 1e-12
    assert abs(row["D_R"] - 0.05) < 1e-12
    assert abs(row["D_C"] - 0.10) < 1e-12
    assert abs(row["D_MAIN"] - 0.075) < 1e-12
    assert abs(row["M_R"] - 0.10) < 1e-12
    assert abs(row["M_C"] - 0.15) < 1e-12
    assert abs(row["M_MAIN"] - 0.125) < 1e-12
    assert abs(row["INTERACTION"] - 0.05) < 1e-12
