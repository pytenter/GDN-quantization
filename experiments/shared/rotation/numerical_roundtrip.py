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


def aggregate_error_contribution(suite: dict, tensor_names: Iterable[str]) -> dict:
    roundtrip, cast = [], []
    for tensor_name in tensor_names:
        for item in suite[tensor_name].values():
            p2 = item["p2_bf16_storage_fp32_rotation"]
            roundtrip.append(float(p2["roundtrip_error_relative_l2"]))
            cast.append(float(p2["cast_error_relative_l2_delta"]))
    return {
        "roundtrip_error_relative_l2_mean": sum(roundtrip) / len(roundtrip),
        "bf16_cast_error_relative_l2_delta_mean": sum(cast) / len(cast),
        "dominant": "BF16_CAST" if sum(cast) > sum(roundtrip) else "ROTATION_ROUNDTRIP",
    }
