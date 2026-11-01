"""Device-capable migration of sealed groupwise equations; no torch/model import.

Import is inert. CPU fixtures are legal now. Any CUDA Tensor requires a later
explicit authorization passed by the independently gated worker. This code
performs no device inventory, resource probe, model construction or kernel call.
"""
from dataclasses import dataclass
import hashlib
from matched_baseline import condition_spec

LOW, HIGH, EPS = 2.0 ** -24, 65504.0, 1e-12
GPU_RUNTIME_STATUS = "NOT_RUN"


@dataclass(frozen=True)
class EncodedGroup:
    codes: object
    scales: object
    zero_points: object
    condition: str
    group_size: int
    asymmetric: bool
    original_shape: tuple
    diagnostics: dict

    def decode(self, torch):
        b,h,k,v = self.original_shape
        q = self.codes.reshape(b,h,k,v//self.group_size,self.group_size)
        if self.asymmetric:
            centered = q.to(torch.int16) - self.zero_points.to(torch.int16).unsqueeze(-1)
            values = centered.float() * self.scales.float().unsqueeze(-1)
        else:
            values = q.float() * self.scales.float().unsqueeze(-1)
        result = values.reshape(self.original_shape).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise ArithmeticError("NONFINITE_GROUPWISE_DECODE")
        return result

    def payload_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.codes,self.scales,self.zero_points) if t is not None)

    def metadata_audit(self):
        parts = {"codes":self.codes.numel()*self.codes.element_size(),"scales":self.scales.numel()*self.scales.element_size(),
                 "zero_points":0 if self.zero_points is None else self.zero_points.numel()*self.zero_points.element_size()}
        return dict(condition=self.condition,group_size=self.group_size,actual_tensor_payload_bytes=parts,
                    total_integer_and_metadata_payload_bytes=sum(parts.values()),
                    payload_bits_per_element=8*sum(parts.values())/self.codes.numel(),
                    code_dtype=str(self.codes.dtype),scale_dtype=str(self.scales.dtype),
                    zero_point_dtype=None if self.zero_points is None else str(self.zero_points.dtype),
                    encoded_object_retains_unquantized_state=False,physical_packed_GPU_persistent=False,
                    excludes="FP32 native cache carrier, H, model weights, workspace, allocator and headers")


def encode(torch, x, condition, future_gpu_authorized=False):
    """Encode already Value-H-rotated [B,H,128,128] state, always FP16 scale.

    Same inherited FP32 fit arithmetic and safe-half floor/ceiling. Every group
    is contiguous along Value; batch/head/Key-row identities never mix. No
    Float32 metadata alternative is permitted in this matched runtime matrix.
    """
    spec = condition_spec(condition)
    if x.device.type not in ("cpu","cuda"):
        raise ValueError("Unsupported execution device")
    if x.device.type == "cuda" and future_gpu_authorized is not True:
        raise PermissionError("GPU_STAGE_NOT_AUTHORIZED")
    if x.ndim != 4 or tuple(x.shape[-2:]) != (128,128) or min(x.shape)<1:
        raise ValueError("Require actual native-axis KDA [B,H,128,128] Hadamard coordinates")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("Require finite floating state")
    value = x.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ArithmeticError("FP32_STATE_CONVERSION_OVERFLOW")
    b,h,k,v = value.shape
    grouped = value.reshape(b,h,k,v//spec.group_size,spec.group_size)
    if spec.asymmetric:
        zero = torch.zeros((),dtype=torch.float32,device=value.device)
        lower = torch.minimum(grouped.amin(-1),zero)
        upper = torch.maximum(grouped.amax(-1),zero)
        span = upper-lower
        if not bool(torch.isfinite(span).all()):
            raise ArithmeticError("FP32_ASYMMETRIC_SPAN_OVERFLOW")
        raw_scale = span.clamp_min(EPS)/255.0
    else:
        maximum = grouped.abs().amax(-1)
        raw_scale = maximum.clamp_min(EPS)/127.0
    if not bool(torch.isfinite(raw_scale).all()):
        raise ArithmeticError("NONFINITE_RAW_SCALE")
    stored = raw_scale.clamp(LOW,HIGH).half().detach().contiguous().clone()
    if not bool(torch.isfinite(stored).all()) or not bool((stored>0).all()):
        raise ArithmeticError("INVALID_STORED_FP16_SCALE")
    levels = grouped/stored.float().unsqueeze(-1)
    if not bool(torch.isfinite(levels).all()):
        raise ArithmeticError("NONFINITE_CODE_QUOTIENT")
    if spec.asymmetric:
        ratio = -lower/stored.float()
        if not bool(torch.isfinite(ratio).all()):
            raise ArithmeticError("NONFINITE_ZERO_POINT_QUOTIENT")
        z = torch.round(ratio).clamp(0,255).to(torch.uint8).detach().contiguous().clone()
        shifted = torch.round(levels)+z.float().unsqueeze(-1)
        clipping = (shifted<0)|(shifted>255)
        q = shifted.clamp(0,255).to(torch.uint8)
        saturation = (q==0)|(q==255)
    else:
        z = None
        rounded = torch.round(levels)
        clipping = (rounded < -127)|(rounded>127)
        q = rounded.clamp(-127,127).to(torch.int8)
        saturation = q.abs()==127
    q = q.reshape(value.shape).detach().contiguous().clone()
    diagnostics = dict(raw_scale_min=float(raw_scale.min()),raw_scale_max=float(raw_scale.max()),
        stored_scale_min=float(stored.float().min()),stored_scale_max=float(stored.float().max()),
        FP16_underflow_floor_hits=int((raw_scale<LOW).sum()),FP16_overflow_ceiling_hits=int((raw_scale>HIGH).sum()),
        rounded_clipping_count=int(clipping.sum()),saturation_code_count=int(saturation.sum()),
        group_count=stored.numel(),nonfinite_count=0,rounding="ties-to-even; round before integer zero-point addition",
        metadata_dtype="FP16_ACTUALLY_STORED_AND_USED",group_axis="Value",dynamic_qdq_idempotence="NOT_ASSUMED")
    if z is not None:
        diagnostics.update(zero_point_min=int(z.min()),zero_point_max=int(z.max()),
                           zero_point_boundary_count=int(((z==0)|(z==255)).sum()))
    return EncodedGroup(q,stored,z,condition,spec.group_size,spec.asymmetric,tuple(value.shape),diagnostics)


def fingerprint(torch, value):
    """Audit bytes only; never retain source Tensor as continuation state."""
    host = value.detach().to(device="cpu",copy=True).contiguous()
    raw = host.view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def compare_to_cpu_reference(torch, rotated, actual, reference_encoder):
    """Strict future Canary hook: same already-rotated input, zero tolerance.

    Caller injects the separately SHA-verified historical CPU source encoder.
    No fallback is allowed when codes/scales/zero-point/decode differ in bytes.
    This routine alone does not qualify an H matmul or a GPU full-model run.
    """
    host = rotated.detach().to(device="cpu",copy=True)
    ref = reference_encoder(host,actual.group_size,actual.asymmetric)
    result = {}
    for name, a, b in (("codes",actual.codes,ref.codes),("scales",actual.scales,ref.scales),
                       ("zero_points",actual.zero_points,ref.zero_points),
                       ("decoded",actual.decode(torch),ref.decode())):
        if a is None or b is None:
            result[name] = dict(torch_equal=a is None and b is None,byte_equal=a is None and b is None,
                                max_absolute_error=0.0,max_relative_error=0.0)
            continue
        a = a.detach().to(device="cpu",copy=True).contiguous()
        b = b.detach().to(device="cpu",copy=True).contiguous()
        diff = (a.double()-b.double()).abs()
        result[name] = dict(torch_equal=bool(torch.equal(a,b)),byte_equal=fingerprint(torch,a)==fingerprint(torch,b),
                           max_absolute_error=float(diff.max()),max_relative_error=float((diff/b.double().abs().clamp_min(1e-300)).max()))
    passed = all(r["torch_equal"] and r["byte_equal"] and r["max_absolute_error"]==0 and r["max_relative_error"]==0 for r in result.values())
    if not passed:
        raise AssertionError(("STRICT_CPU_REFERENCE_PARITY_FAILED",actual.condition,result))
    headwise=[]
    for batch in range(host.shape[0]):
        for head in range(host.shape[1]):
            piece=reference_encoder(host[batch:batch+1,head:head+1],actual.group_size,actual.asymmetric)
            checks={}
            for name,source,reference in (("codes",actual.codes,piece.codes),("scales",actual.scales,piece.scales),
                                         ("zero_points",actual.zero_points,piece.zero_points)):
                if source is None:
                    checks[name]=reference is None
                else:
                    checks[name]=fingerprint(torch,source[batch:batch+1,head:head+1])==fingerprint(torch,reference)
            if not all(checks.values()):
                raise AssertionError(("HEADWISE_CPU_REFERENCE_PARITY_FAILED",actual.condition,batch,head,checks))
            headwise.append(dict(batch=batch,head=head,fields_byte_equal=checks))
    return dict(status="PASS_EXACT",fields=result,headwise_reference_checks=headwise,headwise_byteexact=True,
                headwise_scope="separately invoked frozen CPU reference on exact rotated input; not GPU H matmul parity",
                tolerance=0,GPU_RUNTIME_STATUS=GPU_RUNTIME_STATUS)
