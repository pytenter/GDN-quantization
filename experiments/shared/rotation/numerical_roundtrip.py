#!/usr/bin/env python3
"""Compact finite-precision roundtrip diagnostics for orthogonal rotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


EPS = 1e-30


def tensor_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict:
    left = value.detach().to(device="cpu", dtype=torch.float64)
    right = reference.detach().to(device="cpu", dtype=torch.float64)
    difference = left - right
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    return {
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(difference) / (right_norm + EPS)).item()),
        "cosine": float(((left * right).sum() / (left_norm * right_norm + EPS)).item()),
        "nonfinite": int((~torch.isfinite(left)).sum().item()),
    }


def _matmul_roundtrip(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return value.matmul(matrix).matmul(matrix.transpose(0, 1))


def precision_roundtrip(value: torch.Tensor, matrix_fp64: torch.Tensor, device: str) -> dict:
    """Measure P0/P1/P2/P3 without claiming an unobservable accumulator dtype."""
    source = value.detach().to(device="cpu", dtype=torch.float64)
    matrix64 = matrix_fp64.detach().to(device="cpu", dtype=torch.float64)

    p0 = _matmul_roundtrip(source, matrix64)

    source32 = source.to(device=device, dtype=torch.float32)
    matrix32 = matrix64.to(device=device, dtype=torch.float32)
    p1 = _matmul_roundtrip(source32, matrix32)

    stored_bf16 = source.to(device=device, dtype=torch.bfloat16)
    bf16_reference = stored_bf16.float()
    p2_before = _matmul_roundtrip(bf16_reference, matrix32)
    p2_after = p2_before.to(torch.bfloat16)
    before_metric = tensor_metrics(p2_before, bf16_reference)
    after_metric = tensor_metrics(p2_after, stored_bf16)

    # Some model paths store the forward-rotated driver in BF16 before the
    # inverse/readout. Keep this distinct from native-BF16 matmul: both
    # multiplications remain FP32 and only the intermediate is quantized.
    forward_fp32 = bf16_reference.matmul(matrix32)
    forward_stored_bf16 = forward_fp32.to(torch.bfloat16)
    intermediate_before_final = forward_stored_bf16.float().matmul(matrix32.transpose(0, 1))
    intermediate_after_final = intermediate_before_final.to(torch.bfloat16)
    intermediate_before_metric = tensor_metrics(intermediate_before_final, bf16_reference)
    intermediate_after_metric = tensor_metrics(intermediate_after_final, stored_bf16)

    native = {
        "status": "UNAVAILABLE",
        "input_dtype": "torch.bfloat16",
        "weight_dtype": "torch.bfloat16",
        "accumulation_dtype": "ACCUMULATION_DTYPE_UNKNOWN",
        "output_dtype": None,
        "metrics": None,
    }
    try:
        matrix_bf16 = matrix64.to(device=device, dtype=torch.bfloat16)
        native_value = _matmul_roundtrip(stored_bf16, matrix_bf16)
        native.update({
            "status": "PASS",
            "output_dtype": str(native_value.dtype),
            "metrics": tensor_metrics(native_value, stored_bf16),
        })
    except (RuntimeError, TypeError) as error:
        native["error"] = f"{type(error).__name__}: {error}"

    return {
        "p0_fp64": {
            "input_dtype": "torch.float64", "weight_dtype": "torch.float64",
            "matmul_dtype": "torch.float64", "output_dtype": str(p0.dtype),
            "metrics": tensor_metrics(p0, source),
        },
        "p1_fp32": {
            "input_dtype": "torch.float32", "weight_dtype": "torch.float32",
            "matmul_dtype": "torch.float32", "output_dtype": str(p1.dtype),
            "metrics": tensor_metrics(p1, source32),
        },
        "p2_bf16_storage_fp32_rotation": {
            "input_storage_dtype": "torch.bfloat16",
            "rotation_dtype": "torch.float32",
            "before_final_bf16_cast": before_metric,
            "after_final_bf16_cast": after_metric,
            "roundtrip_error_relative_l2": before_metric["relative_l2"],
            "cast_error_relative_l2_delta": after_metric["relative_l2"] - before_metric["relative_l2"],
        },
        "p2b_bf16_intermediate_storage_fp32_rotation": {
            "input_storage_dtype": "torch.bfloat16",
            "rotation_dtype": "torch.float32",
            "intermediate_storage_dtype": "torch.bfloat16",
            "forward_storage_cast": tensor_metrics(forward_stored_bf16, forward_fp32),
            "before_final_bf16_cast": intermediate_before_metric,
            "after_final_bf16_cast": intermediate_after_metric,
            "intermediate_cast_contribution_relative_l2_delta": (
                intermediate_before_metric["relative_l2"] - before_metric["relative_l2"]
            ),
            "final_cast_contribution_relative_l2_delta": (
                intermediate_after_metric["relative_l2"] - intermediate_before_metric["relative_l2"]
            ),
        },
        "p3_native_bf16": native,
    }


def run_suite(tensors: dict[str, torch.Tensor], rotations: dict[str, torch.Tensor], device: str) -> dict:
    return {
        tensor_name: {
            rotation_name: precision_roundtrip(tensor, matrix, device)
            for rotation_name, matrix in rotations.items()
        }
        for tensor_name, tensor in tensors.items()
    }


def manifest_hashes(manifest: dict) -> dict[str, str]:
    hashes = {"H": "ANALYTIC_NORMALIZED_HADAMARD_128"}
    for item in manifest["matrices"]:
        hashes[item["name"].replace("R_seed", "R")] = item["sha256_fp64_c_order_bytes"]
    return hashes


def validate_manifest_hashes(previous_path: Path, current_manifest: dict) -> dict:
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    expected = manifest_hashes(previous)
    actual = manifest_hashes(current_manifest)
    return {
        "expected": expected,
        "actual": actual,
        "status": "PASS" if expected == actual else "MATRIX_PROVENANCE_MISMATCH",
    }


def aggregate_error_contribution(
    suite: dict, tensor_names: Iterable[str], *, uses_bf16_intermediate_storage: bool = False
) -> dict:
    roundtrip, intermediate_cast, final_cast = [], [], []
    for tensor_name in tensor_names:
        for item in suite[tensor_name].values():
            p2 = item["p2_bf16_storage_fp32_rotation"]
            roundtrip.append(float(p2["roundtrip_error_relative_l2"]))
            if uses_bf16_intermediate_storage:
                p2b = item["p2b_bf16_intermediate_storage_fp32_rotation"]
                intermediate_cast.append(float(p2b["intermediate_cast_contribution_relative_l2_delta"]))
                final_cast.append(float(p2b["final_cast_contribution_relative_l2_delta"]))
            else:
                intermediate_cast.append(0.0)
                final_cast.append(float(p2["cast_error_relative_l2_delta"]))
    roundtrip_mean = sum(roundtrip) / len(roundtrip)
    intermediate_mean = sum(intermediate_cast) / len(intermediate_cast)
    final_mean = sum(final_cast) / len(final_cast)
    contributions = {
        "ROTATION_ROUNDTRIP": max(0.0, roundtrip_mean),
        "BF16_INTERMEDIATE_CAST": max(0.0, intermediate_mean),
        "BF16_FINAL_CAST": max(0.0, final_mean),
    }
    return {
        "roundtrip_error_relative_l2_mean": roundtrip_mean,
        "bf16_intermediate_cast_error_relative_l2_delta_mean": intermediate_mean,
        "bf16_final_cast_error_relative_l2_delta_mean": final_mean,
        "dominant": max(contributions, key=contributions.get),
    }
