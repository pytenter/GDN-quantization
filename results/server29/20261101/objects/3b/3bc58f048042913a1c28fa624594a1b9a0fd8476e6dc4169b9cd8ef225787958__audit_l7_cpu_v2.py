"""CPU-only source identity, fixed-input helper parity, and historical audit."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from objective import original_helpers, digest, combined_scores

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
ORIGINAL = Path("/data/zypan/cleanroom/LING_L6_L7_NEW_MEMBER_HANDOFF_V1_e1256bf_full/experiments/LING_RECURRENT_DENSE_L6_L7_V1")
INHERITED = Path("/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY/inputs")


def publish(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261011)
    src = ORIGINAL / "scripts/recurrent_dense_train.py" if args.remote else ROOT / "sources/recurrent_dense_train.py"
    h = original_helpers(torch, src)
    fixtures = []
    for dtype in [torch.float32, torch.bfloat16]:
        for scale in [0.0, 1e-9, 1.0, 10.0]:
            reference = (torch.randn(1, 1, 12) * scale).to(dtype)
            value = (reference.float() + torch.randn_like(reference.float()) * 1e-3).to(dtype)
            actual = h["relative_mse"](value, reference)
            gold = ((value.float()-reference.float()).square().sum()
                    / reference.float().square().sum().clamp_min(1e-12))
            assert torch.equal(actual, gold) and torch.isfinite(actual)
            assert torch.equal(actual, h["relative_mse"](value, reference))
            assert float(h["relative_mse"](reference, reference)) == 0
            fixtures.append({"dtype": str(dtype), "teacher_scale": scale,
                             "loss": float(actual), "exact_formula_and_repeat": True})
    teacher = torch.randn(1, 16, 128, 128)
    student = teacher + torch.randn_like(teacher) * 0.03
    gold_state = (((student-teacher).square().sum((-2,-1)))
                  / teacher.square().sum((-2,-1)).clamp_min(1e-12)).mean()
    assert torch.equal(h["relative_state_loss"](student, teacher), gold_state)
    # A nonuniform-head fixture catches accidental global relative-MSE replacement.
    t = torch.ones(1, 2, 2, 2); t[:, 1] *= 10
    s = t.clone(); s[:, 0] += 1
    assert float(h["relative_state_loss"](s, t)) == 0.5
    assert float(h["relative_mse"](s, t)) != 0.5
    assert float(h["relative_state_loss"](teacher, teacher)) == 0
    # Call-site parity only; the real gated normalization remains a GPU gate.
    calls = []
    class RecordingNorm:
        def __call__(self, x, gate):
            calls.append((x.dtype, gate.dtype, tuple(x.shape)))
            return x + gate
    projection = torch.nn.Linear(8, 5, bias=False).to(torch.bfloat16)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(
        attention=SimpleNamespace(o_proj=projection, o_norm=RecordingNorm()))]))
    core, gate = torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4)
    output = h["local_output"](model, 0, core, gate)
    expected = projection((core.to(torch.bfloat16)+gate.to(torch.bfloat16)).reshape(1,1,-1))
    assert torch.equal(output, expected)
    assert calls == [(torch.bfloat16, torch.bfloat16, (1,1,2,4))]
    f, st, total = combined_scores(torch, h, {0: output+1}, {0: output}, student, teacher, [0])
    assert torch.equal(total, f + 0.1 * st)
    config_path = ORIGINAL / "configs/recurrent_training_config.json" if args.remote else ROOT / "sources/recurrent_training_config.json"
    summary_path = ORIGINAL / "checkpoints/l7/training_summary.json" if args.remote else ROOT / "sources/l7_training_summary.json"
    cfg, summary = json.loads(config_path.read_text()), json.loads(summary_path.read_text())
    assert cfg["detach_interval"] == cfg["gradient_horizon"] == 128
    assert cfg["sequence_length"] == summary["numerical_recurrent_horizon"] == 512
    assert not cfg["student_state_reset_within_sequence"] and summary["steps_completed"] == 100
    history = [dict(step=x["step"], **x["validation"]) for x in summary["history"] if "validation" in x]
    assert all(x["primary"] == x["functional"] for x in history)
    best_functional = min(history, key=lambda x: x["functional"])["step"]
    best_total = min(history, key=lambda x: x["combined"])["step"]
    quantizer = None
    if args.remote:
        path = INHERITED / "runtime_repo/experiments/shared/rotation/cayley_rotation.py"
        spec = importlib.util.spec_from_file_location("l7_original_cayley_cpu", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        source = torch.randn(1,16,128,128)
        result = h["exact_qdq"](module, source, False)
        boundary = source.to(torch.bfloat16).float()
        scales = boundary.abs().amax(-1,keepdim=True).clamp_min(1e-12)/127
        codes = torch.round(boundary/scales).clamp(-127,127)
        assert torch.equal(result.scale, scales) and torch.equal(result.dequant, codes*scales)
        assert torch.equal(result.codes, codes)
        quantizer = {"source_sha256": digest(path), "original_boundary_bf16": True,
                     "R128_V_axis_exact": True, "ties_to_even": True,
                     "historical_codes_dtype": str(result.codes.dtype),
                     "no_actual_gpu_writeback_test_claimed": True}
    publish(ROOT / ("CPU_OBJECTIVE_PARITY_REMOTE.json" if args.remote else "CPU_OBJECTIVE_PARITY_LOCAL.json"), {
        "status": "CPU_DEFINITION_PARITY_PASS", "matched_gpu_forward_parity": "NOT_EXECUTED_GPU_OCCUPIED",
        "torch": torch.__version__, "python": platform.python_version(), "gpu_hours": 0,
        "source_sha256": digest(src), "config_sha256": digest(config_path),
        "training_summary_sha256": digest(summary_path), "fixed_inputs": fixtures,
        "per_head_state_weighting_exact": True, "post_o_proj_helper_callsite_exact": True,
        "real_o_norm_kernel_validated": False, "loss_combination_exact": True,
        "historical_quantizer_cpu_test": quantizer, "validation_history": history,
        "best_functional_step": best_functional, "best_total_step": best_total,
        "checkpoint_selection_disagreement_observed": best_total != best_functional,
        "no_training_or_backward_executed": True, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    print("CPU_DEFINITION_PARITY_PASS; GPU matched parity remains NOT_EXECUTED")


if __name__ == "__main__":
    main()
