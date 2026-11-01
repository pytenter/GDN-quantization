"""Key-axis INT8 codecs. Eager FP32 only; no model/GPU initialization."""
from dataclasses import dataclass
import torch

EPS = 1e-12
METHODS = ('B0', 'B1', 'B2')

@dataclass
class Encoded:
    method: str
    shape: tuple
    codes: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    negative_zero_flags: torch.Tensor | None

    def cpu(self):
        return Encoded(self.method, self.shape, self.codes.detach().cpu().clone(),
                       self.scales.detach().cpu().clone(),
                       None if self.zero_points is None else self.zero_points.detach().cpu().clone(),
                       None if self.negative_zero_flags is None else self.negative_zero_flags.detach().cpu().clone())

def validate_input(x):
    if x.ndim != 4 or tuple(x.shape[-2:]) != (128,128) or x.dtype != torch.float32:
        raise ValueError('Expected FP32 [B,H,Key128,Value128]')
    if not bool(torch.isfinite(x).all().item()):
        raise FloatingPointError('nonfinite preQDQ')

def encode(x, method, *, group_size=32, cpu_semantics='CPU_CANONICAL', debug=False):
    validate_input(x)
    if method not in METHODS: raise ValueError('unknown condition')
    if cpu_semantics not in ('CPU_CANONICAL','CUDA_LEGACY_EMULATED'):
        raise ValueError('unknown arithmetic contract')
    if group_size not in (32, 64): raise ValueError('Unsupported versioned Key group size')
    group = 128 if method == 'B0' else group_size
    y = x.detach().reshape(x.shape[0], x.shape[1], 128//group, group, 128)
    if method != 'B2':
        peak = y.abs().amax(dim=-2)
        safe = peak.clamp_min(EPS)
        if x.device.type == 'cpu' and cpu_semantics == 'CUDA_LEGACY_EMULATED':
            # Mirrors PyTorch v2.5.1 CUDA scalar-divisor dispatch. GPU must verify.
            reciprocal = torch.tensor(1.0/127.0, dtype=torch.float32, device=x.device)
            scale = safe * reciprocal
        else:
            scale = safe / 127.0  # Exact inherited CUDA operation graph.
        levels = y / scale.unsqueeze(-2)
        rounded = torch.round(levels)
        clipped = rounded.clamp(-127,127)
        codes = clipped.to(torch.int8)
        negzero = (clipped == 0) & torch.signbit(clipped)
        result = Encoded(method, tuple(x.shape), codes, scale, None, negzero)
        fields = dict(peak=peak,safe_range=safe,scales=scale,levels=levels,
                      rounded=rounded,codes=codes,negative_zero_flags=negzero)
    else:
        a = torch.minimum(y.amin(dim=-2), torch.zeros_like(y[...,0,:]))
        b = torch.maximum(y.amax(dim=-2), torch.zeros_like(y[...,0,:]))
        # Canonical +0 extrema, distinct from original symmetric signed-zero rule.
        a = torch.where(a == 0, torch.zeros_like(a), a)
        b = torch.where(b == 0, torch.zeros_like(b), b)
        span = b-a
        safe = span.clamp_min(EPS)
        scale = safe / torch.full_like(safe, 255.0)
        zero_ratio = -a/scale
        zp = torch.round(zero_ratio).clamp(0,255).to(torch.uint8)
        levels = y / scale.unsqueeze(-2)
        rounded = torch.round(levels)
        shifted = rounded + zp.unsqueeze(-2).float()
        codes = shifted.clamp(0,255).to(torch.uint8)
        result = Encoded(method,tuple(x.shape),codes,scale,zp,None)
        fields = dict(minimum=a,maximum=b,span=span,safe_range=safe,scales=scale,
                      zero_ratio=zero_ratio,zero_points=zp,levels=levels,rounded=rounded,codes=codes)
    if not bool(torch.isfinite(scale).all().item()) or not bool((scale>0).all().item()):
        raise FloatingPointError('invalid FP32 scale')
    if not bool(torch.isfinite(levels).all().item()):
        raise FloatingPointError('nonfinite level')
    if debug: fields['decoded'] = decode(result)
    return (result, fields) if debug else result

def decode(e):
    if e.method == 'B2':
        if e.codes.dtype != torch.uint8 or e.zero_points.dtype != torch.uint8:
            raise ValueError('asymmetric integer dtype')
        centered = e.codes.to(torch.int16) - e.zero_points.unsqueeze(-2).to(torch.int16)
        out = centered.float() * e.scales.unsqueeze(-2)
        out = torch.where(out == 0, torch.zeros_like(out), out)
    else:
        if e.codes.dtype != torch.int8: raise ValueError('symmetric integer dtype')
        out = e.codes.float() * e.scales.unsqueeze(-2)
        out = torch.where(e.negative_zero_flags, torch.full_like(out,-0.0), out)
    if not bool(torch.isfinite(out).all().item()): raise FloatingPointError('nonfinite decode')
    return out.reshape(e.shape)

def qdq_writeback(state, method, **kwargs):
    encoded = encode(state, method, **kwargs)
    state.copy_(decode(encoded))
    return encoded

def tensor_bytes(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

def require_consumed(actual, expected_encoded):
    expected = decode(expected_encoded)
    actual_cpu = actual.detach().cpu()
    if actual_cpu.dtype != expected.dtype or actual_cpu.shape != expected.shape:
        raise RuntimeError('next-consumption shape/dtype mismatch')
    # Actual values and actual bytes, not pointer/hash identity.
    if not torch.equal(actual_cpu,expected) or tensor_bytes(actual_cpu) != tensor_bytes(expected):
        raise RuntimeError('next kernel did not consume QDQ writeback')
    return True
