#!/usr/bin/env python3
"""Hadamard-initialized, dense SO(d) corrections for recurrent-state INT8.

The public API deliberately keeps the canonical Hadamard operation separate:
callers first execute their existing H128 implementation and only then call
``forward_correction``.  The inverse order is correction transpose followed by
the existing canonical Hadamard inverse.  This prevents accidentally replacing
the deployed two-stage numerical path with one precomposed dense matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


DIMENSION = 128
DOF_128 = DIMENSION * (DIMENSION - 1) // 2
EPS = 1.0e-12


def _axis_to_last(value: torch.Tensor, axis: int) -> tuple[torch.Tensor, int]:
    axis = axis if axis >= 0 else value.ndim + axis
    if not 0 <= axis < value.ndim:
        raise IndexError(f"axis {axis} invalid for rank {value.ndim}")
    return value.movedim(axis, -1), axis


class CayleyDenseRotation(nn.Module):
    """A proper dense orthogonal correction parameterized by d(d-1)/2 FP32s."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        super().__init__()
        self.dimension = int(dimension)
        rows, cols = torch.triu_indices(self.dimension, self.dimension, offset=1)
        self.register_buffer("upper_rows", rows, persistent=False)
        self.register_buffer("upper_cols", cols, persistent=False)
        self.theta = nn.Parameter(torch.zeros(rows.numel(), dtype=torch.float32))

    @property
    def degrees_of_freedom(self) -> int:
        return int(self.theta.numel())

    def skew(self) -> torch.Tensor:
        # The FP32 master parameter is never silently promoted from model dtype.
        theta = self.theta.float()
        matrix = torch.zeros(
            (self.dimension, self.dimension), device=theta.device, dtype=torch.float32
        )
        matrix[self.upper_rows, self.upper_cols] = theta
        matrix[self.upper_cols, self.upper_rows] = -theta
        return matrix

    def matrix(self) -> torch.Tensor:
        """Return (I-A)(I+A)^-1 without constructing a matrix inverse."""
        skew = self.skew()
        identity = torch.eye(self.dimension, device=skew.device, dtype=torch.float32)
        left = identity - skew
        right = identity + skew
        # Solve X right = left by transposing to the solve(B, A) convention.
        return torch.linalg.solve(right.transpose(0, 1), left.transpose(0, 1)).transpose(0, 1)

    def forward_correction(self, hadamard_value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """Apply only DeltaR after an already executed canonical Hadamard."""
        moved, original_axis = _axis_to_last(hadamard_value, axis)
        corrected = moved.float().matmul(self.matrix())
        return corrected.movedim(-1, original_axis)

    def inverse_correction(self, rotated_value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """Apply only DeltaR.T before the caller's canonical H inverse."""
        moved, original_axis = _axis_to_last(rotated_value, axis)
        corrected = moved.float().matmul(self.matrix().transpose(0, 1))
        return corrected.movedim(-1, original_axis)

    def audit(self) -> dict[str, float | int | bool]:
        matrix = self.matrix()
        identity = torch.eye(self.dimension, device=matrix.device, dtype=torch.float32)
        residual = matrix.transpose(0, 1).matmul(matrix) - identity
        sign, logabsdet = torch.linalg.slogdet(matrix)
        return {
            "dimension": self.dimension,
            "degrees_of_freedom": self.degrees_of_freedom,
            "max_abs_rt_r_minus_i": float(residual.abs().max().detach().cpu()),
            "frobenius_rt_r_minus_i": float(torch.linalg.vector_norm(residual).detach().cpu()),
            "determinant_sign": float(sign.detach().cpu()),
            "log_abs_determinant": float(logabsdet.detach().cpu()),
            "theta_norm": float(torch.linalg.vector_norm(self.theta).detach().cpu()),
            "distance_from_identity_frobenius": float(
                torch.linalg.vector_norm(matrix - identity).detach().cpu()
            ),
            "finite": bool(torch.isfinite(matrix).all().detach().cpu()),
        }


class PerLayerCayleyRotations(nn.Module):
    """One dense correction per recurrent layer, shared across all heads."""

    def __init__(self, layer_ids: Iterable[int], dimension: int = DIMENSION) -> None:
        super().__init__()
        ids = tuple(int(layer_id) for layer_id in layer_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("layer ids must be unique")
        self.layer_ids = ids
        self.rotations = nn.ModuleDict(
            {str(layer_id): CayleyDenseRotation(dimension) for layer_id in ids}
        )

    def layer(self, layer_id: int) -> CayleyDenseRotation:
        return self.rotations[str(int(layer_id))]

    @property
    def total_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def runtime_gate(self, threshold: float = 1.0e-4) -> dict:
        layers = {layer_id: self.layer(layer_id).audit() for layer_id in self.layer_ids}
        maximum = max((row["max_abs_rt_r_minus_i"] for row in layers.values()), default=0.0)
        finite = all(row["finite"] for row in layers.values())
        proper = all(row["determinant_sign"] > 0 for row in layers.values())
        return {
            "threshold": float(threshold),
            "max_abs_rt_r_minus_i": float(maximum),
            "all_finite": finite,
            "all_proper": proper,
            "layers": {str(key): value for key, value in layers.items()},
            "status": "PASS" if finite and proper and maximum <= threshold else "FAIL",
        }


@dataclass(frozen=True)
class QuantizationResult:
    dequant: torch.Tensor
    scale: torch.Tensor
    codes: torch.Tensor


def canonical_symmetric_int8_qdq(value: torch.Tensor, group_axis: int) -> QuantizationResult:
    """Exact canonical symmetric INT8 QDQ (ties-to-even via torch.round)."""
    source = value.float()
    scale = source.abs().amax(dim=group_axis, keepdim=True).clamp_min(EPS) / 127.0
    codes = torch.round(source / scale).clamp(-127, 127)
    return QuantizationResult(codes * scale, scale, codes)


def qwen_c128_qdq(value: torch.Tensor) -> QuantizationResult:
    """Qwen [B,H,K,V]: groups K (dim=-2), scale [B,H,1,V]."""
    return canonical_symmetric_int8_qdq(value, group_axis=-2)


def ling_r128_qdq(value: torch.Tensor) -> QuantizationResult:
    """Ling [B,H,K,V]: groups V (dim=-1), scale [B,H,K,1]."""
    return canonical_symmetric_int8_qdq(value, group_axis=-1)


def ste_from_exact(value: torch.Tensor, exact_dequant: torch.Tensor) -> torch.Tensor:
    """Exact QDQ in the forward pass and identity gradient in the backward pass."""
    return value + (exact_dequant - value).detach()


def qwen_c128_ste(value: torch.Tensor) -> QuantizationResult:
    exact = qwen_c128_qdq(value)
    return QuantizationResult(ste_from_exact(value, exact.dequant), exact.scale, exact.codes)


def ling_r128_ste(value: torch.Tensor) -> QuantizationResult:
    exact = ling_r128_qdq(value)
    return QuantizationResult(ste_from_exact(value, exact.dequant), exact.scale, exact.codes)


def relative_mse(value: torch.Tensor, reference: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    numerator = (value.float() - reference.float()).square().sum()
    denominator = reference.float().square().sum().clamp_min(eps)
    return numerator / denominator

