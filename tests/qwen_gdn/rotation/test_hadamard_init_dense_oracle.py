import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py"


def source():
    return RUNNER.read_text(encoding="utf-8")


def test_runner_is_syntax_valid_and_forbids_aime_data():
    ast.parse(source())
    assert '"AIME26_used": False' in source()


def test_qwen_axis_and_two_stage_order_are_explicit():
    text = source()
    assert 'C.qwen_c128_ste' in text
    assert 'h_state = torch.einsum' in text
    assert 'rotation, h_state' in text


def test_only_theta_bank_is_given_to_adam():
    text = source()
    assert 'torch.optim.Adam(bank.parameters()' in text
    assert 'freeze_model(model)' in text


def test_protocol_optimizer_and_early_stop_are_fixed():
    text = source()
    assert 'weight_decay=0.0' in text
    assert 'clip_grad_norm_(bank.parameters(), 1.0)' in text
    assert 'stale >= 10' in text
