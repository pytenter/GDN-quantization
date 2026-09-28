#!/usr/bin/env python3
"""Deterministic SO(128) matrices and an explicit R/R.T rotation API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


DIMENSION = 128
ORTHOGONALITY_GATE = 1e-12
GENERATION_METHOD = "CPU_FP64_GAUSSIAN_QR_CANONICAL_DIAGONAL_DET_POSITIVE"


def normalized_hadamard(dimension: int = DIMENSION) -> torch.Tensor:
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float64, device="cpu")
    while matrix.shape[0] < dimension:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return matrix / math.sqrt(dimension)


def dense_so_matrix(seed: int, dimension: int = DIMENSION) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    gaussian = torch.randn((dimension, dimension), generator=generator, dtype=torch.float64)
    q, upper = torch.linalg.qr(gaussian, mode="reduced")
    diagonal = torch.diagonal(upper)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * signs.unsqueeze(0)
    if float(torch.linalg.det(q).item()) < 0.0:
        q[:, -1].neg_()
    return q.contiguous()


def matrix_sha256(matrix: torch.Tensor) -> str:
    value = matrix.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def matrix_audit(name: str, matrix: torch.Tensor, seed: int | None) -> dict:
    value = matrix.detach().to(device="cpu", dtype=torch.float64)
    identity = torch.eye(value.shape[0], dtype=torch.float64)
    residual = value.transpose(0, 1) @ value - identity
    determinant = float(torch.linalg.det(value).item())
    maximum = float(residual.abs().max().item())
    return {
        "name": name,
        "seed": seed,
        "dimension": int(value.shape[0]),
        "dtype": "torch.float64",
        "device": "cpu",
        "generation_method": GENERATION_METHOD if seed is not None else name.upper(),
        "determinant": determinant,
        "max_abs_rt_r_minus_i": maximum,
        "frobenius_rt_r_minus_i": float(torch.linalg.vector_norm(residual).item()),
        "sha256_fp64_c_order_bytes": matrix_sha256(value),
        "orthogonality_gate_threshold": ORTHOGONALITY_GATE,
        "orthogonality_gate": "PASS" if maximum <= ORTHOGONALITY_GATE else "FAIL",
    }


def build_manifest(seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    matrices = [matrix_audit(f"R_seed{seed}", dense_so_matrix(seed), seed) for seed in seeds]
    status = "PASS" if all(item["orthogonality_gate"] == "PASS" for item in matrices) else "FAIL"
    return {
        "schema_version": "ARBITRARY_ORTHOGONAL_ROTATION_MANIFEST_V1",
        "torch_version": torch.__version__,
        "dimension": DIMENSION,
        "generator": GENERATION_METHOD,
        "status": status,
        "matrices": matrices,
    }


def _apply_on_axis(value: torch.Tensor, matrix: torch.Tensor, axis: int) -> torch.Tensor:
    normalized_axis = axis if axis >= 0 else value.ndim + axis
    if normalized_axis < 0 or normalized_axis >= value.ndim:
        raise IndexError(f"axis {axis} is invalid for rank-{value.ndim} tensor")
    if value.shape[normalized_axis] != matrix.shape[0]:
        raise ValueError(
            f"axis size {value.shape[normalized_axis]} does not match rotation {matrix.shape[0]}"
        )
    moved = value.movedim(normalized_axis, -1)
    rotated = moved.float().matmul(matrix.to(device=value.device, dtype=torch.float32))
    return rotated.movedim(-1, normalized_axis).to(value.dtype)


class OrthogonalRotation128:
    """Row-vector convention: forward uses R and inverse uses R.T."""

    def __init__(self, name: str, matrix_fp64: torch.Tensor):
        self.name = name
        self.matrix_fp64 = matrix_fp64

    @classmethod
    def identity(cls) -> "OrthogonalRotation128":
        return cls("identity", torch.eye(DIMENSION, dtype=torch.float64))

    @classmethod
    def hadamard(cls) -> "OrthogonalRotation128":
        return cls("hadamard", normalized_hadamard())

    @classmethod
    def dense(cls, seed: int) -> "OrthogonalRotation128":
        return cls(f"R_seed{seed}", dense_so_matrix(seed))

    def forward(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return _apply_on_axis(value, self.matrix_fp64, axis)

    def inverse(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return _apply_on_axis(value, self.matrix_fp64.transpose(0, 1), axis)

    def fp32(self, device: torch.device | str) -> torch.Tensor:
        return self.matrix_fp64.to(device=device, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    if manifest["status"] != "PASS":
        raise RuntimeError("ORTHOGONAL_GENERATOR_FAIL")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
