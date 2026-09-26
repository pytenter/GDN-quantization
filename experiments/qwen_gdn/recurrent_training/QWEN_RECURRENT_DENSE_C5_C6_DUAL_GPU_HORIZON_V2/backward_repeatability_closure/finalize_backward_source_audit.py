#!/usr/bin/env python3
"""Record the strict deterministic diagnostic and source audit, without changing runtime."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent
REPO = V2.parents[1]
ANALYSIS = ROOT / "analysis"
CONFIGS = ROOT / "configs"
PROTOCOL = ROOT / "protocol_amendments"
TRANSFORMERS = Path("/data/zypan/transformers-qwen35")
MODEL = Path("/data/zypan/modelscope_models/Qwen3.5-9B")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git_head(path: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    error_path = ANALYSIS / "single_deterministic_01.error.json"
    error = json.loads(error_path.read_text())
    first_op = "cumsum_cuda_kernel"
    if first_op not in error["message"] or "torch_chunk_gated_delta_rule" not in error["traceback"]:
        raise RuntimeError("unexpected deterministic error; inspect before classifying")
    modeling = TRANSFORMERS / "src/transformers/models/qwen3_5/modeling_qwen3_5.py"
    cayley = REPO / "experiments/shared/rotation/cayley_rotation.py"
    runtime = {
        "qwen_experiment_git_commit": git_head(REPO),
        "transformers_qwen35_git_commit": git_head(TRANSFORMERS),
        "transformers_modeling_qwen35_sha256": sha(modeling),
        "model_path": str(MODEL),
        "model_config_sha256": sha(MODEL / "config.json"),
        "v1_training_code_sha256": sha(REPO / "experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py"),
        "v2_topology_code_sha256": sha(V2 / "run_dual_gpu_v2.py"),
        "qdq_rotation_code_sha256": sha(cayley),
        "backward_closure_code_sha256": sha(ROOT / "backward_closure.py"),
        "initial_reference_config_sha256": sha(CONFIGS / "frozen_backward_reference.json"),
        "note": "This supplements initial preregistration fields that were null because server Git does not support git -C.",
    }
    new(CONFIGS / "runtime_source_revisions.json", runtime)
    new(PROTOCOL / "deterministic_backward_candidate.json", {
        "candidate": "STRICT_PYTORCH_DETERMINISTIC_DIAGNOSTIC_ONLY",
        "torch_use_deterministic_algorithms": True,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_precision": "highest",
        "result": "REJECTED_NOT_EXECUTABLE_ON_FROZEN_PATH",
        "first_reported_nondeterministic_op": first_op,
        "first_error_sha256": sha(error_path),
        "no_operator_bypass_or_replacement": True,
        "not_adopted_for_formal_training": True,
    })
    new(ANALYSIS / "deterministic_mode_diagnostic.json", {
        "candidate_sha256": sha(PROTOCOL / "deterministic_backward_candidate.json"),
        "error_sha256": sha(error_path),
        "FIRST_REPORTED_NONDETERMINISTIC_OP": first_op,
        "location": "torch_chunk_gated_delta_rule, modeling_qwen3_5.py line 291, g.cumsum(dim=-1)",
        "phase": "forward_during_teacher_target_collection_before_backward",
        "SINGLE_GPU_EXACT_BACKWARD_CLOSURE": "NOT_ACHIEVABLE_IN_TESTED_STRICT_RUNTIME",
        "deterministic_forward_vs_original": "NOT_RUN_OPERATOR_ERROR",
        "formal_runtime_changed": False,
    })
    source = modeling.read_text()
    cayley_source = cayley.read_text()
    new(ANALYSIS / "backward_nondeterminism_source_audit.json", {
        "BACKWARD_NONDETERMINISM_SOURCE": "UNRESOLVED",
        "FIRST_REPORTED_NONDETERMINISTIC_OP": first_op,
        "finding": "PyTorch flags CUDA cumsum in the Qwen3.5 gated-delta fallback as nondeterministic before backward; this does not by itself prove it caused the observed gradient variability.",
        "evidence": {
            "single_forward_exact_five_runs": True,
            "single_gradient_bitwise_repeatable_five_runs": False,
            "deterministic_error_sha256": sha(error_path),
            "runtime_source_revisions_sha256": sha(CONFIGS / "runtime_source_revisions.json"),
            "gated_delta_fallback_has_cumsum": "g = g.cumsum(dim=-1)" in source,
            "qdq_ste_uses_detach_identity_gradient": "value + (exact_dequant - value).detach()" in cayley_source,
            "custom_autograd_Function_in_qdq_source": "torch.autograd.Function" in cayley_source,
            "modeling_source_explicit_triton_autotune": "@triton.autotune" in source,
            "modeling_source_explicit_cuda_graph": "CUDAGraph" in source,
            "modeling_source_explicit_scatter_add": "scatter_add" in source or "index_add" in source,
            "flash_attn_installed": True,
            "fla_distribution_installed": False,
            "hub_fallback_stack_observed": "hub_kernels.py" in error["traceback"] and "torch_chunk_gated_delta_rule" in error["traceback"],
        },
        "candidate_mechanisms_not_proven": [
            "CUDA cumsum kernel or its backward reduction",
            "CUDA reduction or shared-parameter gradient accumulation ordering",
            "other model backward kernels, including full-attention blocks",
        ],
        "diagnostic_limits": "Strict deterministic mode halts at the first reported forward operator, so later backward operators cannot be audited by that mode without changing the frozen path.",
    })
    (ROOT / "reports/source_determinism_audit.md").write_text(
        "# Backward source/determinism audit\n\n"
        "Five independent single-GPU runs had bitwise-identical forward/QDQ/teacher targets/loss "
        "but non-identical raw rotation gradients. The first divergent trainable parameter is "
        "`bank.layer(0).theta`; 21 of 24 GDN layers varied in at least one pair.\n\n"
        "A diagnostic runtime enabled PyTorch deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, "
        "disabled TF32, and selected deterministic cuDNN. It failed during teacher-target forward at "
        "`g.cumsum(dim=-1)` (`cumsum_cuda_kernel`) in Qwen3.5's gated-delta fallback, before backward. "
        "No operator was bypassed/replaced and these settings were not adopted for formal training. "
        "This flags a nondeterministic operator but does not prove it caused the observed gradient variation; "
        "backward source classification remains UNRESOLVED. The QDQ STE is a detach-based identity gradient, "
        "not a custom autograd Function. See analysis JSON for source/runtime hashes and other candidates.\n"
    )
    print(json.dumps({"FIRST_REPORTED_NONDETERMINISTIC_OP": first_op,
                      "BACKWARD_NONDETERMINISM_SOURCE": "UNRESOLVED",
                      "numerical_protocol_sha256": sha(PROTOCOL / "backward_numerical_equivalence_v1.json")}, indent=2))


if __name__ == "__main__":
    main()
