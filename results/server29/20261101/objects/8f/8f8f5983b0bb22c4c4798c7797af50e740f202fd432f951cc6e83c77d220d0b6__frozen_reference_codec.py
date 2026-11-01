"""CPU-only common interfaces. Original Dual-M fitting belongs to model adapters.

These utilities never detect accelerators and never import a model.  Decoded
floating tensors are temporary evaluations, not persistent packed INT8 states.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from typing import Iterable
import torch

LOW, HIGH = 2.0 ** -24, 65504.0
_CPU_GUARD_ATTEMPTS: list[str] = []
_CPU_GUARD_INSTALLED = False


def install_cpu_guard() -> None:
    """Make accidental torch CUDA probes/initialization fail before hardware use."""
    global _CPU_GUARD_INSTALLED
    if _CPU_GUARD_INSTALLED:
        return
    def blocked(name):
        def fail(*args, **kwargs):
            _CPU_GUARD_ATTEMPTS.append(name)
            raise RuntimeError("ALLOW_GPU=NO: forbidden CUDA operation " + name)
        return fail
    for name in ("init", "_lazy_init", "is_available", "device_count",
                 "current_device", "synchronize", "memory_allocated",
                 "memory_reserved", "max_memory_allocated", "max_memory_reserved",
                 "reset_peak_memory_stats"):
        if hasattr(torch.cuda, name):
            setattr(torch.cuda, name, blocked("torch.cuda." + name))
    torch.Tensor.cuda = blocked("Tensor.cuda")
    _CPU_GUARD_INSTALLED = True


def cpu_guard_status() -> dict:
    return dict(ALLOW_GPU=False, NEW_GPUh=0, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                guard_installed=_CPU_GUARD_INSTALLED,
                forbidden_attempts=list(_CPU_GUARD_ATTEMPTS),
                hardware_probes_performed=0)


def cpu_state(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
        raise ValueError("Require CPU Tensor; accelerator probing is forbidden")
    if x.ndim != 4 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError("Require nonempty floating [B,H,Key,Value] state")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite input state")
    result = x.detach().to(dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite conversion to FP32")
    return result


def architecture_name(architecture: str) -> str:
    name = str(architecture).lower()
    if name not in ("gdn", "kda"):
        raise ValueError("architecture must be gdn or kda")
    return name


def metadata_precision(name) -> str:
    if name in ("fp16", "float16", "deployment_FP16", torch.float16):
        return "fp16"
    if name in ("fp32", "float32", "canonical_FP32", "ideal_FP32", torch.float32):
        return "fp32"
    raise ValueError("metadata_dtype must explicitly specify fp16 or fp32")


def owned(x: torch.Tensor) -> torch.Tensor:
    return x.detach().contiguous().clone()


def validate_owned_tensor(x: torch.Tensor) -> None:
    if x.device.type != "cpu" or not x.is_contiguous() or x._base is not None:
        raise ValueError("Encoded tensors must be owned contiguous CPU storage")
    if x.untyped_storage().nbytes() != x.numel() * x.element_size():
        raise ValueError("Encoded tensor has hidden backing-storage bytes")


def payload_audit(codes: torch.Tensor, metadata: Iterable[tuple[str, torch.Tensor | None]]) -> dict:
    validate_owned_tensor(codes)
    parts = {"codes": codes.untyped_storage().nbytes()}
    dtypes = {"codes": str(codes.dtype)}
    shapes = {"codes": list(codes.shape)}
    for name, tensor in metadata:
        if tensor is not None:
            validate_owned_tensor(tensor)
            parts[name] = tensor.untyped_storage().nbytes()
            dtypes[name] = str(tensor.dtype)
            shapes[name] = list(tensor.shape)
    total = sum(parts.values())
    return dict(tensor_storage_bytes=parts, code_bytes=parts["codes"],
                metadata_bytes=total-parts["codes"], tensor_payload_bytes=total,
                effective_bits_per_element=8.0*total/codes.numel(),
                dtypes=dtypes, shapes=shapes, integer_code_storage=True,
                floating_decoded_carrier_persisted=False,
                excludes="serialization header, Python/allocator overhead, shared transform matrices",
                deployment_status="CPU packed-representation accounting; no deployed INT8 runtime claim")


def round_scales(raw: torch.Tensor, precision: str) -> tuple[torch.Tensor, dict]:
    precision = metadata_precision(precision)
    if raw.device.type != "cpu" or raw.dtype != torch.float32:
        raise ValueError("Scale arithmetic must be CPU FP32")
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise ArithmeticError("Unsupported nonfinite or negative raw fitted scale")
    if precision == "fp16":
        result = owned(raw.clamp(LOW, HIGH).to(torch.float16))
    else:
        result = owned(raw)
    if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise ArithmeticError("Stored scale must be finite and strictly positive")
    decoded = result.float()
    abs_error = (decoded.double()-raw.double()).abs()
    rel_error = abs_error / raw.double().abs().clamp_min(1e-300)
    return result, dict(raw_scale_min=float(raw.min()), raw_scale_max=float(raw.max()),
        stored_scale_min=float(decoded.min()), stored_scale_max=float(decoded.max()),
        raw_scale_zero_count=int((raw == 0).sum()),
        fp16_underflow_floor_hits=int((raw < LOW).sum()) if precision == "fp16" else 0,
        fp16_overflow_ceiling_hits=int((raw > HIGH).sum()) if precision == "fp16" else 0,
        metadata_max_absolute_error=float(abs_error.max()),
        metadata_max_relative_error=float(rel_error.max()),
        scale_arithmetic="FP32", stored_metadata=precision,
        ideal_FP32_and_stored_FP16_are_distinct_conditions=True)


def validate_fit_scales(x: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor) -> None:
    b,h,k,v=x.shape
    for scale,shape in ((rows,(b,h,k,1)),(columns,(b,h,1,v))):
        if not isinstance(scale,torch.Tensor) or scale.device.type != "cpu":
            raise ValueError("Original fitted scales must be CPU Tensors")
        if tuple(scale.shape) != shape or scale.dtype != torch.float16:
            raise ValueError("Original Dual-M rows/columns require exact FP16 [B,H,K,1]/[B,H,1,V]")
        if not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):
            raise ValueError("Invalid original fitted rows/columns")


def hadamard(n: int = 128, dtype=torch.float32) -> torch.Tensor:
    if not isinstance(n,int) or n < 1 or n & (n-1):
        raise ValueError("Hadamard order must be a positive power of two")
    if dtype not in (torch.float32,torch.float64):
        raise ValueError("Hadamard dtype must be explicitly FP32 or FP64")
    h=torch.ones((1,1),dtype=dtype,device="cpu")
    while h.shape[0] < n:
        h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return owned(h/(n**0.5))


def _transform(x,architecture,H,inverse):
    x=cpu_state(x); architecture=architecture_name(architecture)
    n=x.shape[-2] if architecture=="gdn" else x.shape[-1]
    H=hadamard(n) if H is None else H
    if H.device.type!="cpu" or H.dtype!=torch.float32 or tuple(H.shape)!=(n,n):
        raise ValueError("Transform requires an explicit matching CPU FP32 Hadamard")
    if not bool(torch.isfinite(H).all()):raise ValueError("Nonfinite Hadamard")
    if architecture=="gdn":return H@x if inverse else H.T@x
    return x@H.T if inverse else x@H


def to_hadamard(x, architecture: str, H=None):
    return _transform(x,architecture,H,False)


def from_hadamard(x, architecture: str, H=None):
    return _transform(x,architecture,H,True)

