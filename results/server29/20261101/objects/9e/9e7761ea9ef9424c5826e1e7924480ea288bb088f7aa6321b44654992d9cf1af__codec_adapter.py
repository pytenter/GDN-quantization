"""Qualified Dual-M AST adapter for native KDA [B,H,Key,Value] state.

Rotation is deliberately outside this codec. No CUDA initialization or model import.
FP32 decoded carrier is functional QDQ, not a physical packed-INT8 state claim.
"""
import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
LOW, HIGH = 2.0 ** -24, 65504.0
NAMES = ('C128_CANONICAL', 'C128_FP16_SCALE', 'C128_LS_FP16', 'DUAL_M_INT8')
QUALIFIED_CODEC_SHA256 = 'cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6'
OFFICIAL_QUANTIZATION_SHA256 = '0c39fe6da2b5ad68e2a081471b1ec22465a9c9f914c367ed984a78ad3717ee69'


def source(x):
    if not isinstance(x, torch.Tensor) or x.device.type not in ('cpu', 'cuda') or x.ndim != 4 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError('Require finite floating native [B,H,Key,Value] state')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Nonfinite input')
    x = x.detach().float()
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Nonfinite FP32 input conversion')
    return x


@dataclass(frozen=True)
class EncodedState:
    codec: str
    shape: tuple
    codes: torch.Tensor
    columns: torch.Tensor
    rows: torch.Tensor | None = None

    def validate(self):
        b, h, k, v = self.shape
        assert self.codec == NAMES[3]
        assert self.codes.dtype == torch.int8 and tuple(self.codes.shape) == self.shape
        assert bool((self.codes != -128).all())
        assert self.rows is not None and self.rows.dtype == torch.float16
        assert self.columns.dtype == torch.float16
        assert tuple(self.rows.shape) == (b, h, k, 1)
        assert tuple(self.columns.shape) == (b, h, 1, v)
        for scale in (self.rows, self.columns):
            assert bool(torch.isfinite(scale).all()) and bool((scale > 0).all())
            assert scale.device == self.codes.device
        assert bool(torch.isfinite(self.decode()).all())
        return self

    def decode(self):
        # Preserve original fit_state association; x/(r*c) would change rounding.
        return (self.rows.float() * self.columns.float()) * self.codes.float()

    def tensor_payload_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.codes, self.rows, self.columns))


def _extract(path, names, namespace):
    data = path.read_bytes()
    expected = QUALIFIED_CODEC_SHA256 if path.name == 'qualified_dual_axis_codec.py' else OFFICIAL_QUANTIZATION_SHA256
    assert hashlib.sha256(data).hexdigest() == expected, 'REFERENCE_SOURCE_CHANGED'
    tree = ast.parse(data)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


_ns = dict(torch=torch, CANON={'EPS': 1e-8}, LOW=LOW, HIGH=HIGH, NAMES=NAMES,
           source=source, EncodedState=EncodedState)
_extract(ROOT / 'reference_sources/qualified_dual_axis_codec.py', ('safe_half', 'levels', 'encode'), _ns)
safe_half, levels = _ns['safe_half'], _ns['levels']
_encode = _ns['encode']


def encode_dualm(x):
    """All eight bits, unit importance, zero pivots; one column LS + reencode."""
    return _encode(x, NAMES[3])


def trace_encode_dualm(x):
    """Low-frequency observer; original encode AST and returned output unchanged.

    Raw scale inputs and diagnostic tensors stay on the source device until the
    caller offloads them. This must never be retained as next-token state.
    The adapter supports one serial worker, not concurrent encode calls.
    """
    raw_scales = []
    original_safe_half = _ns['safe_half']
    def observed_safe_half(value):
        raw_scales.append(value.detach().clone())
        return original_safe_half(value)
    _ns['safe_half'] = observed_safe_half
    try:
        encoded = encode_dualm(x)
    finally:
        _ns['safe_half'] = original_safe_half
    assert len(raw_scales) == 3, 'QUALIFIED_SCALE_CALL_LAYOUT_CHANGED'
    native = x.detach().float()
    levels_unrounded = (native / encoded.rows.float()) / encoded.columns.float()
    levels_rounded = torch.round(levels_unrounded)
    scale_names = ('rows', 'initial_columns', 'LS_columns')
    scale_records = []
    affected = torch.zeros_like(native, dtype=torch.bool)
    for name, raw in zip(scale_names, raw_scales):
        outside = (raw < LOW) | (raw > HIGH)
        affected = affected | outside.expand_as(native)
        scale_records.append(dict(name=name, raw_input=raw,
            raw_min=raw.amin(), raw_max=raw.amax(),
            raw_floor_hits=(raw < LOW).sum(), raw_ceiling_hits=(raw > HIGH).sum()))
    affected = affected & (native != 0)
    energy = native.double().square()
    error = (encoded.decode().double() - native.double()).square()
    energy_total = energy.sum((-2, -1))
    energy_affected = (energy * affected).sum((-2, -1))
    error_affected = (error * affected).sum((-2, -1))
    energy_fraction = torch.where(energy_total > 0,
        energy_affected / energy_total.clamp_min(1e-300), torch.zeros_like(energy_total))
    affected_relative_mse = torch.where(energy_affected > 0,
        error_affected / energy_affected.clamp_min(1e-300), torch.zeros_like(energy_affected))
    trace = dict(scale_records=scale_records,
        unrounded_over127_count=(levels_unrounded.abs() > 127).sum(),
        rounded_clamp_count=(levels_rounded.abs() > 127).sum(),
        unrounded_nonfinite_count=(~torch.isfinite(levels_unrounded)).sum(),
        code_count=encoded.codes.numel(),
        effective_scale_min=(encoded.rows.float() * encoded.columns.float()).amin(),
        effective_scale_max=(encoded.rows.float() * encoded.columns.float()).amax(),
        affected_energy_fraction_per_batch_head=energy_fraction,
        affected_relative_mse_per_batch_head=affected_relative_mse,
        local_relative_mse_per_batch_head=torch.where(energy_total > 0,
            error.sum((-2, -1)) / energy_total.clamp_min(1e-300), torch.zeros_like(energy_total)),
        observational_only=True)
    return encoded, trace


def official_reference(x, architecture='kda'):
    """Fixed official fit_state AST; all8/ones/pivots0, no model/calibration import."""
    ns = dict(torch=torch, dataclass=dataclass)
    _extract(ROOT / 'reference_sources/stepquant_quantization.py',
             ('QuantizationPlan', '_fp16_scale', 'fit_state'), ns)
    x = source(x)
    plan = ns['QuantizationPlan'](architecture,
        torch.full(x.shape[1:3], 8, dtype=torch.int64, device=x.device),
        torch.ones(x.shape[1:3], dtype=torch.float32, device=x.device))
    return ns['fit_state'](x, plan)


def direct_equation_reference(x):
    """Separate short equation implementation, preserving all frozen operations."""
    x = source(x)
    half = lambda v: v.clamp(LOW, HIGH).half()
    nearest = lambda v: torch.round(v).clamp(-127, 127)
    rows = half(x.abs().mean(-1, keepdim=True).clamp_min(1e-20).sqrt())
    normalized = x / rows.float()
    columns = half((normalized.abs() / 127.).amax(-2, keepdim=True))
    codes = nearest(normalized / columns.float())
    weighted = rows.float() * codes
    denominator = weighted.square().sum(-2, keepdim=True)
    columns = half(torch.where(denominator > 0,
        (weighted * x).sum(-2, keepdim=True) / denominator.clamp_min(1e-30), columns.float()))
    codes = nearest(normalized / columns.float()).to(torch.int8)
    return codes, rows, columns, (rows.float() * columns.float()) * codes.float()
