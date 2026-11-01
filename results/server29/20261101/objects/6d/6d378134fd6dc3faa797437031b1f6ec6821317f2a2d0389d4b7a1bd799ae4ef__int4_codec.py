"""Version 1, independent logical UINT4 Key32 codec; eager FP32 operation graph."""
from dataclasses import dataclass
import torch

RANGE_FLOOR = 1e-12

@dataclass
class Encoded:
    method: str
    shape: tuple
    codes: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    negative_zero_flags: None = None

    def cpu(self):
        return Encoded(self.method, self.shape, self.codes.detach().cpu().clone(),
                       self.scales.detach().cpu().clone(), self.zero_points.detach().cpu().clone())

def tensor_bytes(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

def encode(x, method='INT4', *, debug=False):
    if method != 'INT4' or x.dtype != torch.float32 or x.ndim != 4 or tuple(x.shape[-2:]) != (128,128):
        raise ValueError('INT4 expects FP32 [B,H,Key128,Value128]')
    if not bool(torch.isfinite(x).all().item()):
        raise FloatingPointError('nonfinite preQDQ')
    y = x.detach().reshape(x.shape[0], x.shape[1], 4, 32, 128)
    a = torch.minimum(y.amin(dim=-2), torch.zeros_like(y[...,0,:]))
    b = torch.maximum(y.amax(dim=-2), torch.zeros_like(y[...,0,:]))
    a = torch.where(a == 0, torch.zeros_like(a), a)
    b = torch.where(b == 0, torch.zeros_like(b), b)
    span = b - a
    safe = span.clamp_min(RANGE_FLOOR)
    scale = safe / torch.full_like(safe, 15.0)
    if not bool(torch.isfinite(scale).all().item()) or not bool((scale > 0).all().item()):
        raise FloatingPointError('invalid FP32 scale (including span overflow)')
    zero_ratio = -a / scale
    zp = torch.round(zero_ratio).clamp(0,15).to(torch.uint8)
    levels = y / scale.unsqueeze(-2)
    if not bool(torch.isfinite(levels).all().item()):
        raise FloatingPointError('nonfinite FP32 normalized level')
    rounded = torch.round(levels)
    shifted = rounded + zp.unsqueeze(-2).float()
    codes = shifted.clamp(0,15).to(torch.uint8)
    encoded = Encoded('INT4', tuple(x.shape), codes, scale, zp)
    if not debug:
        return encoded
    fields = dict(minimum=a, maximum=b, span=span, safe_range=safe, scales=scale,
                  zero_ratio=zero_ratio, zero_points=zp, levels=levels, rounded=rounded,
                  shifted=shifted, codes=codes, decoded=decode(encoded))
    return encoded, fields

def decode(e):
    if e.method != 'INT4' or e.codes.dtype != torch.uint8 or e.zero_points.dtype != torch.uint8 or e.scales.dtype != torch.float32:
        raise ValueError('INT4 encoded field dtype/method mismatch')
    centered = e.codes.to(torch.int16) - e.zero_points.unsqueeze(-2).to(torch.int16)
    out = centered.float() * e.scales.unsqueeze(-2)
    out = torch.where(out == 0, torch.zeros_like(out), out)
    if not bool(torch.isfinite(out).all().item()):
        raise FloatingPointError('nonfinite FP32 decode')
    return out.reshape(e.shape)

def require_consumed(actual, expected_encoded):
    expected = decode(expected_encoded)
    actual_cpu = actual.detach().cpu()
    if actual_cpu.dtype != expected.dtype or actual_cpu.shape != expected.shape:
        raise RuntimeError('next-consumption shape/dtype differs')
    if not torch.equal(actual_cpu, expected) or tensor_bytes(actual_cpu) != tensor_bytes(expected):
        raise RuntimeError('next kernel did not consume actual INT4 QDQ state bytes')
    return True
