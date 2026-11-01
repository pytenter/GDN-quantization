"""Fixed matched FP16-scale students; canonical FP32 R128 is identity only."""
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
import ast
import hashlib
import sys
import types


@dataclass(frozen=True)
class Condition:
    identifier: str
    group_size: int
    asymmetric: bool
    scale_dtype: str = "fp16"


CONDITIONS = MappingProxyType({
    "B0": Condition("B0", 128, False),
    "B1": Condition("B1", 32, False),
    "B2": Condition("B2", 32, True),
    "B3": Condition("B3", 64, True),
})
CANONICAL_R128_FP32_ROLE = "HISTORICAL_SOURCE_IDENTITY_ONLY_NOT_MATCHED_B0"


def condition_spec(identifier):
    if identifier not in CONDITIONS:
        raise ValueError("Only frozen B0/B1/B2/B3 matched FP16 students are allowed")
    return CONDITIONS[identifier]


def condition_identity(identifier):
    value = condition_spec(identifier)
    groups = 128 * (128 // value.group_size)
    code_bytes = 128 * 128
    scale_bytes = groups * 2
    z_bytes = groups if value.asymmetric else 0
    return dict(condition=identifier,group_size=value.group_size,asymmetric=value.asymmetric,
                Hadamard="fixed Value-H128",metadata_dtype="fp16",code_dtype="uint8" if value.asymmetric else "int8",
                codes_bytes_per_head=code_bytes,scale_bytes_per_head=scale_bytes,zero_point_bytes_per_head=z_bytes,
                integer_payload_bytes_per_head=code_bytes+scale_bytes+z_bytes,
                payload_bits_per_element=8*(code_bytes+scale_bytes+z_bytes)/16384,
                physical_packed_GPU_persistent=False,unquantized_continuation_allowed=False)


def build_reference_encoder(torch):
    """Load exact sealed CPU AST functions without its CUDA/env import side effect.

    No model or legacy module is imported. Only the SHA-verified math, CPU
    validators and encoded dataclass are compiled unchanged in an isolated
    namespace. In particular this loader does not set CVD or probe a GPU.
    """
    root=Path(__file__).resolve().parents[1]
    specs=(("frozen_reference_codec.py","8f8f5983b0bb22c4c4798c7797af50e740f202fd432f951cc6e83c77d220d0b6",
            {"cpu_state","architecture_name","metadata_precision","owned","validate_owned_tensor","payload_audit","round_scales"}),
           ("frozen_group32_core.py","1104634db33ae2eb34e9ec4a661f13858949648f8b93763b141357cfa713f862",
            {"_group_view","EncodedGroup","_encode_value_groups"}))
    module=types.ModuleType("_sealed_groupwise_cpu_reference_ast")
    sys.modules[module.__name__]=module
    module.__dict__.update(torch=torch,dataclass=dataclass,LOW=2.**-24,HIGH=65504.)
    for filename,expected,names in specs:
        path=root/"inherited"/filename
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise AssertionError("SEALED_CPU_REFERENCE_SOURCE_CHANGED")
        tree=ast.parse(raw.decode(),filename=str(path))
        selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
        if {node.name for node in selected}!=names:raise AssertionError("MISSING_SEALED_CPU_REFERENCE_AST")
        future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
        body=ast.fix_missing_locations(ast.Module(body=[future]+selected,type_ignores=[]))
        exec(compile(body,str(path),"exec"),module.__dict__)
    def encode_cpu(value,size,asymmetric):
        return module._encode_value_groups(value,"kda",size,asymmetric,"fp16",1e-12)
    return encode_cpu


@dataclass(frozen=True)
class HeadwiseReference:
    codes: object
    scales: object
    zero_points: object
    group_size: int
    shape: tuple

    def decode(self,torch):
        b,h,k,v=self.shape
        q=self.codes.reshape(b,h,k,v//self.group_size,self.group_size)
        centered=q.float() if self.zero_points is None else (q.to(torch.int16)-self.zero_points.to(torch.int16).unsqueeze(-1)).float()
        return (centered*self.scales.float().unsqueeze(-1)).reshape(self.shape).contiguous()


def headwise_reference_encode(torch,x,condition,future_gpu_authorized=False):
    """Independent per-head evaluation of the already frozen device equations.

    Same FP32 fit/safe-half/round-before-z operations, copied from the sealed
    reference. The head loops are an independent grouping/layout path, not a
    faster or alternative quantizer, and this result never enters continuation.
    """
    spec=condition_spec(condition)
    if x.device.type=="cuda" and future_gpu_authorized is not True:raise PermissionError("GPU_STAGE_NOT_AUTHORIZED")
    if x.ndim!=4 or tuple(x.shape[-2:])!=(128,128) or not bool(torch.isfinite(x).all()):raise ValueError("INVALID_REFERENCE_INPUT")
    b,h,k,v=x.shape
    codes=torch.empty(x.shape,dtype=torch.uint8 if spec.asymmetric else torch.int8,device=x.device)
    scales=torch.empty((b,h,k,v//spec.group_size),dtype=torch.float16,device=x.device)
    zp=torch.empty_like(scales,dtype=torch.uint8) if spec.asymmetric else None
    for batch in range(b):
        for head in range(h):
            group=x[batch,head].detach().float().reshape(k,v//spec.group_size,spec.group_size)
            if spec.asymmetric:
                zero=torch.zeros((),dtype=torch.float32,device=x.device)
                lower=torch.minimum(group.amin(-1),zero)
                upper=torch.maximum(group.amax(-1),zero)
                span=upper-lower
                if not bool(torch.isfinite(span).all()):raise ArithmeticError("REFERENCE_SPAN_OVERFLOW")
                raw=span.clamp_min(1e-12)/255.
            else:raw=group.abs().amax(-1).clamp_min(1e-12)/127.
            stored=raw.clamp(2.**-24,65504.).half()
            levels=group/stored.float().unsqueeze(-1)
            if not bool(torch.isfinite(levels).all()):raise ArithmeticError("REFERENCE_NONFINITE_QUOTIENT")
            if spec.asymmetric:
                z=torch.round(-lower/stored.float()).clamp(0,255).to(torch.uint8)
                q=(torch.round(levels)+z.float().unsqueeze(-1)).clamp(0,255).to(torch.uint8)
                zp[batch,head].copy_(z)
            else:q=torch.round(levels).clamp(-127,127).to(torch.int8)
            codes[batch,head].copy_(q.reshape(k,v));scales[batch,head].copy_(stored)
    return HeadwiseReference(codes,scales,zp,spec.group_size,tuple(x.shape))
