import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/ling_kda/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_ling_dense_oracle.py"


def text(): return RUNNER.read_text(encoding="utf-8")


def test_syntax_and_corrected_endpoint_policy():
    ast.parse(text())
    assert "CORRECTED_PREFILL_ENDPOINT_V2" in text()
    assert '"REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO"' in text()


def test_value_axis_and_bf16_boundary_are_explicit():
    source = text()
    assert "C.ling_r128_ste" in source
    assert "rotated_fp32.to(torch.bfloat16)" in source
    assert "state.float().matmul(h).matmul(rotation)" in source


def test_model_is_frozen_and_adam_only_sees_bank():
    source = text()
    assert "freeze_model(model)" in source
    assert "torch.optim.Adam(bank.parameters()" in source
    assert "weight_decay=0.0" in source


def test_aime_is_excluded():
    assert '"AIME26_used": False' in text()
