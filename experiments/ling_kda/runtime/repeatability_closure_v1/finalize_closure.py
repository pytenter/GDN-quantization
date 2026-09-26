#!/usr/bin/env python3
"""Produce the compact, machine-readable Ling repeatability closure verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = read(args.input_dir / "repeatability_summary.json")
    reference = read(args.input_dir / "reference_path_summary.json")
    divergence = read(args.input_dir / "first_divergence_exact.json")
    manifest = read(args.input_dir / "LING_REPEATABILITY_3DOC_MANIFEST.json")
    ratios = summary["METHOD_EFFECT_TO_RUNTIME_NOISE_RATIO"]
    scientifically_resolved = (
        summary["SAME_PROCESS_REPEATABILITY"] == "PASS"
        and summary["FRESH_PROCESS_REPEATABILITY"] == "PASS"
        and all(isinstance(value, (int, float)) and value > 1 for value in ratios.values())
    )
    verdict = "REPEATABILITY_CLOSED_WITH_NUMERICAL_TOLERANCE" if scientifically_resolved else "REPEATABILITY_NOT_CLOSED"
    result = {
        "task": "LING_RUNTIME_REPEATABILITY_CLOSURE_V1",
        "manifest_sha256": manifest["manifest_sha256"],
        "SAME_PROCESS_REPEATABILITY": summary["SAME_PROCESS_REPEATABILITY"],
        "FRESH_PROCESS_REPEATABILITY": summary["FRESH_PROCESS_REPEATABILITY"],
        "ROOT_CAUSE_CLASSIFICATION": summary["ROOT_CAUSE_CLASSIFICATION"],
        "FIRST_DIVERGENT_LAYER": divergence["layer"],
        "FIRST_DIVERGENT_MODULE": "KDA prefill recurrent state/cache before quantization",
        "FIRST_DIVERGENCE_MAGNITUDE": {
            "max_abs": divergence["max_abs"],
            "relative_l2": divergence["relative_l2"],
            "different_elements": divergence["different_elements"],
        },
        "AUC_NOISE_P95": summary["AUC_NOISE_P95"],
        "AUC_NOISE_MAX": summary["AUC_NOISE_MAX"],
        "METHOD_EFFECT_TO_RUNTIME_NOISE_RATIO": ratios,
        "deterministic_algorithm_factor": {
            "status": "UNSUPPORTED_OPERATION",
            "operation": "CuBLAS linear/GEMM without preconfigured CUBLAS_WORKSPACE_CONFIG",
            "environment_changed": False,
        },
        "runtime_factor_findings": {
            "cuda_graph": False,
            "radix_cache": False,
            "mamba_radix_cache": False,
            "torch_compile": False,
            "tf32_matmul": False,
            "fla_fused_kernel": True,
            "fla_cache_mode": "unset/default-disabled",
            "triton_autotuning_in_kda_source": True,
            "explicit_kda_forward_atomic_found_in_source_audit": False,
            "workspace_reuse": "not controlled in canonical runtime",
        },
        "REFERENCE_REPEATABILITY": reference["REFERENCE_REPEATABILITY"],
        "FAST_OPERATOR_REPEATABILITY": reference["FAST_OPERATOR_REPEATABILITY"],
        "FAST_PATH_ONLY_NONDETERMINISM": "INCONCLUSIVE",
        "GPU0_REPEATABILITY": "NOT_RUN_EXTERNAL_GPU_JOB_ACTIVE",
        "GPU1_REPEATABILITY": "same-process PASS; fresh-process FAIL",
        "CROSS_GPU_NUMERICAL_VARIATION": "NOT_RUN_GPU0_BUSY",
        "LING_REPEATABILITY_VERDICT": verdict,
        "PERSISTENT_EVALUATION_NOT_SCIENTIFICALLY_RESOLVED": not scientifically_resolved,
        "16_doc_rerun_authorized": False,
        "PREVIOUS_16DOC_REPLICATION": "NOT_RERUN",
        "LING_PERSISTENT_EVALUATION_READY": False,
        "temporary_tensors_cleaned": divergence["temporary_tensors_cleaned"],
        "durable_tensor_bytes": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
