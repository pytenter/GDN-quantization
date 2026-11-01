"""CPU-ONLY PROPOSED FORMAT: optional INT8 exponent on FP16 state scales.

Not the frozen original R2/R3 method and not authorized for GPU/formal use.
Buffer keys retain the original codec unless explicitly requested otherwise.

Packed, zero-inclusive affine G32 PACE state/key codecs.

These are declared reimplementation conventions, not verified official code.
All scale arithmetic uses float64 before storage in FP16. Encoding and decoding
use that stored scale. Ties use torch.round (nearest even), before adding zp.
INT4 payload and zero points are physically packed, low nibble first.
"""
from dataclasses import dataclass
import math
from typing import Dict, Sequence, Tuple

import torch


GROUP_SIZE = 32
FP16_MIN_SUBNORMAL = 2.0 ** -24
FP16_MAX = 65504.0


class FP16ScaleOverflowError(ValueError):
    """A real stored-scale failure, with bounded JSON-safe technical evidence."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("affine group scale is not representable as positive finite FP16")


def _scale_diagnostic_value(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Infinity" if value > 0 else "-Infinity"
    return value


def _invalid_scale_diagnostics(groups, bits, lo, hi, raw_scale, effective):
    """Only called after the original finite-input check and a failed cast."""
    bad = ~torch.isfinite(effective) | (effective <= 0)
    # A single scalar scale has group index []; do not depend on nonzero's
    # special zero-dimensional indexing convention for that case.
    first = ([[]] if bool(bad.item()) else []) if bad.ndim == 0 else (
        bad.nonzero(as_tuple=False)[:8].detach().cpu().tolist())
    rows = []
    for index in first:
        key = tuple(index)
        rows.append({"index": index,
                     "lo": _scale_diagnostic_value(lo[key].item()),
                     "hi": _scale_diagnostic_value(hi[key].item()),
                     "raw_scale": _scale_diagnostic_value(raw_scale[key].item()),
                     "effective_scale": _scale_diagnostic_value(effective[key].item())})
    return {"status": "UNREPRESENTABLE_STORED_FP16_SCALE",
            "bits": int(bits), "group_shape": list(groups.shape),
            "scale_shape": list(raw_scale.shape), "input_dtype": str(groups.dtype),
            "input_classification": "FINITE_FLOATING_INPUT_ALREADY_VALIDATED",
            "bad_group_count": int(bad.sum().item()),
            "first_bad_group_indices": first, "groups": rows,
            "diagnostic_group_limit": 8, "diagnostic_only_no_recovery": True}


def _require_finite(x: torch.Tensor, name: str) -> None:
    if not x.is_floating_point() or not bool(torch.isfinite(x).all().item()):
        raise ValueError(f"{name} must contain finite floating-point values")


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Flatten UINT4 codes into independently owned UINT8 bytes."""
    if codes.dtype != torch.uint8 or bool((codes > 15).any().item()):
        raise ValueError("nibble codes must be uint8 in [0,15]")
    flat = codes.reshape(-1)
    if flat.numel() % 2:
        flat = torch.cat((flat, flat.new_zeros(1)))
    return (flat[0::2] | (flat[1::2] << 4)).contiguous()


def unpack_nibbles(payload: torch.Tensor, count: int) -> torch.Tensor:
    if payload.dtype != torch.uint8 or payload.ndim != 1:
        raise ValueError("packed payload must be a uint8 vector")
    if count < 0 or payload.numel() != (count + 1) // 2:
        raise ValueError("packed payload length mismatch")
    out = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(-1)
    return out[:count]


@dataclass(frozen=True)
class PackedAffineGroups:
    """A tensor of independent groups, with 32 values on the last axis."""

    bits: int
    shape: Tuple[int, ...]
    codes: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    scale_exponents: object = None  # Optional actual INT8 tensor, one/group.

    @classmethod
    @torch.no_grad()
    def from_tensor(cls, groups: torch.Tensor, bits: int = 4, *, allow_scale_exponents=False):
        if bits not in (4, 8):
            raise ValueError("only INT4 and INT8 are supported")
        if groups.ndim < 1 or groups.shape[-1] != GROUP_SIZE or groups.numel() == 0:
            raise ValueError("each group must have exactly 32 values")
        _require_finite(groups, "groups")
        x = groups.to(torch.float64)
        lo = x.amin(dim=-1).clamp(max=0)
        hi = x.amax(dim=-1).clamp(min=0)
        qmax = (1 << bits) - 1
        raw_scale = (hi - lo) / qmax
        # All-zero groups use scale=1, code=zp=0. Other tiny ranges use the
        # smallest representable nonzero FP16 scale, never a zero denominator.
        effective = torch.where(raw_scale == 0, torch.ones_like(raw_scale),
                                raw_scale.clamp(min=FP16_MIN_SUBNORMAL)).to(torch.float16)
        invalid = ~torch.isfinite(effective) | (effective <= 0)
        scale_exponents = None
        if bool(invalid.any().item()):
            if not allow_scale_exponents:
                raise FP16ScaleOverflowError(
                    _invalid_scale_diagnostics(groups, bits, lo, hi, raw_scale, effective))
            # This extension is deliberately restricted to finite raw scales.
            # In particular it never hides nonfinite input/span arithmetic.
            if not bool(torch.isfinite(raw_scale).all().item()):
                raise ValueError("extended scale requires finite raw affine scales")
            _, binary_exp = torch.frexp(raw_scale)
            tentative = (binary_exp - 16).clamp(min=1)
            if bool((tentative[invalid] > 127).any().item()):
                raise ValueError("INT8 scale exponent exceeds supported range")
            first = torch.ldexp(raw_scale, -tentative).to(torch.float16)
            # At a binade edge, e.g. 131040/2=65520, one more exponent is
            # required. This bounded correction is derived before encoding.
            correction = invalid & (~torch.isfinite(first) | (first <= 0))
            chosen = tentative + correction.to(tentative.dtype)
            if bool((chosen[invalid] > 127).any().item()):
                raise ValueError("INT8 scale exponent exceeds supported range")
            chosen = torch.where(invalid, chosen, torch.zeros_like(chosen))
            scale_exponents = chosen.to(torch.int8).contiguous()
            replacement = torch.ldexp(raw_scale, -chosen).to(torch.float16)
            # Ordinary groups retain the exact initial cast, including every
            # finite-positive value in the 65504..65520 rounding band.
            effective = torch.where(invalid, replacement, effective)
            if not bool(torch.isfinite(effective).all().item()) or bool((effective <= 0).any().item()):
                raise ValueError("extended FP16 mantissa remains invalid")
        scale64 = effective.to(torch.float64)
        if scale_exponents is not None:
            scale64 = torch.ldexp(scale64, scale_exponents.to(torch.int32))
            scale32 = scale64.to(torch.float32)
            if not bool(torch.isfinite(scale32).all().item()) or bool((scale32 <= 0).any().item()):
                raise ValueError("extended scale is not positive finite FP32")
        zp = torch.round(-lo / scale64).clamp(0, qmax).to(torch.uint8)
        q = (torch.round(x / scale64.unsqueeze(-1)) + zp.to(torch.float64).unsqueeze(-1))
        q = q.clamp(0, qmax).to(torch.uint8)
        payload = pack_nibbles(q) if bits == 4 else q.reshape(-1).contiguous()
        zeros = pack_nibbles(zp) if bits == 4 else zp.reshape(-1).contiguous()
        encoded = cls(bits, tuple(groups.shape), payload, effective.contiguous(), zeros,
                      scale_exponents)
        if scale_exponents is not None:
            # Validate actual decoded FP32 carrier, not just the scale field.
            encoded.dequantize()
        return encoded

    def integer_codes(self) -> torch.Tensor:
        n = self.scales.numel() * GROUP_SIZE
        q = unpack_nibbles(self.codes, n) if self.bits == 4 else self.codes
        return q.reshape(self.shape)

    def integer_zero_points(self) -> torch.Tensor:
        n = self.scales.numel()
        zp = unpack_nibbles(self.zero_points, n) if self.bits == 4 else self.zero_points
        return zp.reshape(self.shape[:-1])

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        # Ordinary groups keep the original FP32 arithmetic. Extended groups
        # additionally verify the reconstructed scale and actual FP32 carrier.
        if self.scale_exponents is None:
            values = ((self.integer_codes().to(torch.float32) -
                       self.integer_zero_points().to(torch.float32).unsqueeze(-1)) *
                      self.scales.to(torch.float32).unsqueeze(-1))
            return values.to(dtype)
        exponents = self.scale_exponents
        if (exponents.dtype != torch.int8 or exponents.shape != self.scales.shape or
                exponents.device != self.scales.device or bool((exponents < 0).any().item())):
            raise ValueError("invalid INT8 state scale exponent metadata")
        effective32 = torch.ldexp(self.scales.to(torch.float32), exponents.to(torch.int32))
        if not bool(torch.isfinite(effective32).all().item()) or bool((effective32 <= 0).any().item()):
            raise ValueError("extended scale is not positive finite FP32")
        values = ((self.integer_codes().to(torch.float32) -
                   self.integer_zero_points().to(torch.float32).unsqueeze(-1)) *
                  effective32.unsqueeze(-1))
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("extended decoded FP32 carrier is nonfinite")
        return values.to(dtype)

    def resident_tensors(self) -> Dict[str, torch.Tensor]:
        result = {"codes": self.codes, "scales": self.scales, "zero_points": self.zero_points}
        if self.scale_exponents is not None:
            result["scale_exponents"] = self.scale_exponents
        return result

    def storage_nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.resident_tensors().values())


@dataclass(frozen=True)
class PackedAffineState:
    """Packed mixed-head S[B,H,K,V], grouped only along K within each V."""

    logical_shape: Tuple[int, int, int, int]
    int8_mask: Tuple[bool, ...]
    int4_heads: Tuple[int, ...]
    int8_heads: Tuple[int, ...]
    int4: object
    int8: object

    @classmethod
    @torch.no_grad()
    def from_tensor(cls, state: torch.Tensor, int8_mask: Sequence[bool], group_size: int = 32):
        if group_size != GROUP_SIZE or state.ndim != 4 or state.shape[2] % GROUP_SIZE:
            raise ValueError("state must be [B,H,K,V] with K divisible by fixed G32")
        _require_finite(state, "state")
        if any(d <= 0 for d in state.shape):
            raise ValueError("state dimensions must be positive")
        if isinstance(int8_mask, torch.Tensor):
            if int8_mask.dtype != torch.bool or int8_mask.ndim != 1:
                raise ValueError("int8_mask must be a one-dimensional bool mask")
            mask = tuple(int8_mask.detach().cpu().tolist())
        else:
            mask = tuple(int8_mask)
            if any(type(v) is not bool for v in mask):
                raise ValueError("int8_mask entries must be bool")
        if len(mask) != state.shape[1]:
            raise ValueError("head mask length mismatch")
        b, h, k, v = state.shape
        head4 = tuple(i for i, high in enumerate(mask) if not high)
        head8 = tuple(i for i, high in enumerate(mask) if high)

        def encode(heads, bits):
            if not heads:
                return None
            # [B,H,V,K/32,32]: never group across adjacent value columns.
            groups = state[:, list(heads)].permute(0, 1, 3, 2).contiguous().reshape(b, len(heads), v, k // 32, 32)
            return PackedAffineGroups.from_tensor(groups, bits, allow_scale_exponents=True)

        return cls(tuple(state.shape), mask, head4, head8, encode(head4, 4), encode(head8, 8))

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        sample = self.int4 if self.int4 is not None else self.int8
        b, h, k, v = self.logical_shape
        out = torch.empty(self.logical_shape, device=sample.codes.device, dtype=dtype)
        for heads, part in ((self.int4_heads, self.int4), (self.int8_heads, self.int8)):
            if part is not None:
                values = part.dequantize(dtype).reshape(b, len(heads), v, k).permute(0, 1, 3, 2)
                out[:, list(heads)] = values
        return out

    def resident_tensors(self) -> Dict[str, torch.Tensor]:
        return {f"int{bits}.{name}": tensor
                for bits, part in ((4, self.int4), (8, self.int8)) if part is not None
                for name, tensor in part.resident_tensors().items()}

    def storage_nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.resident_tensors().values())


def encode_affine(state: torch.Tensor, bits: int = 4) -> PackedAffineState:
    """Uniform G32 state codec, also usable by held-out Q4 calibration."""
    if bits not in (4, 8):
        raise ValueError("only INT4 and INT8 are supported")
    return PackedAffineState.from_tensor(state, [bits == 8] * state.shape[1])
