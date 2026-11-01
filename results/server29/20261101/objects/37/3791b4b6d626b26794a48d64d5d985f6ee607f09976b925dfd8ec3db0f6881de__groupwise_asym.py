"""Fixed-axis, CPU FP32 arithmetic Sym/Asym INT8 final encoders."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from reference_codec import (cpu_state, architecture_name, metadata_precision,
    owned, payload_audit, round_scales, validate_owned_tensor)


def _group_view(x,architecture,group_size):
    b,h,k,v=x.shape
    if architecture=="gdn":
        if k%group_size:raise ValueError("Key axis is not divisible by frozen group_size")
        return x.reshape(b,h,k//group_size,group_size,v),3
    if v%group_size:raise ValueError("Value axis is not divisible by frozen group_size")
    return x.reshape(b,h,k,v//group_size,group_size),4


@dataclass(frozen=True)
class EncodedGroup:
    architecture: str
    shape: tuple
    group_size: int
    asymmetric: bool
    metadata_dtype: str
    codes: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    diagnostics: dict

    def validate(self):
        architecture_name(self.architecture)
        precision=metadata_precision(self.metadata_dtype)
        if tuple(self.codes.shape)!=self.shape:raise ValueError("Code shape mismatch")
        expected=torch.uint8 if self.asymmetric else torch.int8
        if self.codes.dtype!=expected:raise ValueError("Integer code dtype mismatch")
        if not self.asymmetric and bool((self.codes==-128).any()):raise ValueError("Sym -128 is forbidden")
        view,axis=_group_view(self.codes,self.architecture,self.group_size)
        shape=list(view.shape);shape.pop(axis)
        if tuple(self.scales.shape)!=tuple(shape):raise ValueError("Compact group scale shape mismatch")
        if self.scales.dtype!=(torch.float16 if precision=="fp16" else torch.float32):raise ValueError("Scale dtype mismatch")
        if not bool(torch.isfinite(self.scales).all()) or not bool((self.scales>0).all()):raise ValueError("Invalid scale")
        if self.asymmetric:
            if self.zero_points is None or self.zero_points.dtype!=torch.uint8 or self.zero_points.shape!=self.scales.shape:
                raise ValueError("Asym requires UINT8 compact group zero_points")
        elif self.zero_points is not None:raise ValueError("Sym has no zero_points")
        for t in (self.codes,self.scales,self.zero_points):
            if t is not None:validate_owned_tensor(t)
        return self

    def decode(self):
        self.validate();view,axis=_group_view(self.codes,self.architecture,self.group_size)
        if self.asymmetric:
            centered=view.to(torch.int16)-self.zero_points.to(torch.int16).unsqueeze(axis)
            y=centered.float()*self.scales.float().unsqueeze(axis)
        else:y=view.float()*self.scales.float().unsqueeze(axis)
        result=owned(y.reshape(self.shape))
        if not bool(torch.isfinite(result).all()):raise ArithmeticError("Nonfinite decoded state")
        return result

    def signed_equivalent(self):
        if not self.asymmetric:raise ValueError("signed_equivalent is only defined for UINT8 Asym")
        return owned((self.codes.to(torch.int16)-128).to(torch.int8)),owned((self.zero_points.to(torch.int16)-128).to(torch.int8))

    def byte_audit(self):
        self.validate()
        return payload_audit(self.codes,(("group_scales",self.scales),("zero_points",self.zero_points)))


@torch.no_grad()
def encode_groupwise(x, architecture="gdn", group_size=128, asymmetric=False,
                     metadata_dtype="fp16", eps=1e-12):
    """Encode without head mixing; grouping follows [B,H,Key,Value] semantics.

    Sym follows max(abs).clamp_min(eps)/127.  Asym includes zero in its
    interval, applies the same model eps to the span, then divides by 255.
    The zero-point is fitted AFTER actual metadata rounding.  Both precision
    contracts use FP32 encoding arithmetic; FP64 is only for diagnostics.
    """
    x=cpu_state(x);architecture=architecture_name(architecture)
    precision=metadata_precision(metadata_dtype)
    if group_size not in (64,128):raise ValueError("Only preregistered 64/128 groups are allowed")
    if not isinstance(eps,(int,float)) or not 0<float(eps)<float("inf"):raise ValueError("eps must be finite positive model-source value")
    eps=float(eps);view,axis=_group_view(x,architecture,group_size)
    if asymmetric:
        a=torch.minimum(view.amin(axis),torch.zeros((),device="cpu",dtype=torch.float32))
        b=torch.maximum(view.amax(axis),torch.zeros((),device="cpu",dtype=torch.float32))
        span=b-a
        if not bool(torch.isfinite(span).all()):raise ArithmeticError("FP32 Asym span overflow")
        raw=span.clamp_min(eps)/255.0
        scales,diag=round_scales(raw,precision)
        zero_ratio=-a/scales.float()
        if not bool(torch.isfinite(zero_ratio).all()):raise ArithmeticError("Nonfinite zero-point arithmetic")
        z_float=torch.round(zero_ratio).clamp(0,255)
        zp=owned(z_float.to(torch.uint8))
        levels=view/scales.float().unsqueeze(axis)
        if not bool(torch.isfinite(levels).all()):raise ArithmeticError("Nonfinite FP32 code quotient")
        rounded=torch.round(levels)
        shifted=rounded+zp.float().unsqueeze(axis)
        clipping=(shifted<0)|(shifted>255)
        codes=owned(shifted.clamp(0,255).to(torch.uint8).reshape(x.shape))
        diag.update(raw_span_floor_hits=int((span<eps).sum()),zero_group_count=int((span==0).sum()),
                    zero_point_min=int(zp.min()),zero_point_max=int(zp.max()),
                    zero_point_boundary_count=int(((zp==0)|(zp==255)).sum()),
                    unrounded_outside_code_range_count=int(((levels+zp.float().unsqueeze(axis)<0)|(levels+zp.float().unsqueeze(axis)>255)).sum()),
                    rounded_clipping_count=int(clipping.sum()),
                    saturation_code_count=int(((codes==0)|(codes==255)).sum()))
    else:
        maximum=view.abs().amax(axis)
        raw=maximum.clamp_min(eps)/127.0
        scales,diag=round_scales(raw,precision)
        levels=view/scales.float().unsqueeze(axis)
        if not bool(torch.isfinite(levels).all()):raise ArithmeticError("Nonfinite FP32 code quotient")
        rounded=torch.round(levels)
        clipping=(rounded < -127)|(rounded > 127)
        codes=owned(rounded.clamp(-127,127).to(torch.int8).reshape(x.shape));zp=None
        diag.update(raw_amplitude_floor_hits=int((maximum<eps).sum()),zero_group_count=int((maximum==0).sum()),
                    unrounded_outside_code_range_count=int((levels.abs()>127).sum()),
                    rounded_clipping_count=int(clipping.sum()),
                    saturation_code_count=int((codes.abs()==127).sum()))
    diag.update(architecture=architecture,group_axis="Key" if architecture=="gdn" else "Value",
        group_size=group_size,asymmetric=bool(asymmetric),eps=eps,
        finite_input=True,nonfinite_raw_parameter_count=0,nonfinite_code_quotient_count=0,
        rounding="round-to-nearest ties-to-even BEFORE adding integer zero-point",
        zero_points_derived_from="actual stored scale recovered to FP32",
        dynamic_reestimate_idempotence="NOT_ASSUMED")
    return EncodedGroup(architecture,tuple(x.shape),group_size,bool(asymmetric),precision,
                        codes,scales,zp,diag).validate()

