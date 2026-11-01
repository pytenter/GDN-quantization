"""Conditional final encoders using ORIGINAL fitted FP16 Dual-M row/column scales.

No row/column fitting occurs here.  Model-specific, source-verified adapters fit
the original codec. T1/T2 replace its last encoding step directly from original
X, not from T0's decoded values. Row/column fitting remains conditioned on q0.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from reference_codec import (cpu_state, metadata_precision, owned, payload_audit,
                            round_scales, validate_fit_scales,validate_owned_tensor)


def _tiles(x,tile_size):
    b,h,k,v=x.shape
    if k%tile_size or v%tile_size:raise ValueError("State dimensions must divide exactly into frozen 32x32 tiles")
    return x.reshape(b,h,k//tile_size,tile_size,v//tile_size,tile_size).permute(0,1,2,4,3,5)


def _untile(x,shape):
    return x.permute(0,1,2,4,3,5).reshape(shape)


@dataclass(frozen=True)
class EncodedDualM:
    shape: tuple
    codes: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor
    diagnostics: dict

    def validate(self):
        if tuple(self.codes.shape)!=self.shape or self.codes.dtype!=torch.int8 or bool((self.codes==-128).any()):raise ValueError("Invalid T0 signed INT8 codes")
        validate_fit_scales(self.codes,self.rows,self.columns)
        for t in (self.codes,self.rows,self.columns):validate_owned_tensor(t)
        return self

    def decode(self):
        self.validate()
        result=owned((self.rows.float()*self.columns.float())*self.codes.float())
        if not bool(torch.isfinite(result).all()):raise ArithmeticError("Nonfinite T0 decode")
        return result

    def byte_audit(self):
        self.validate()
        return payload_audit(self.codes,(("row_scales",self.rows),("column_scales",self.columns)))


@dataclass(frozen=True)
class EncodedTile:
    shape: tuple
    tile_size: int
    asymmetric: bool
    metadata_dtype: str
    codes: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor
    tile_scales: torch.Tensor
    zero_points: torch.Tensor | None
    diagnostics: dict

    @property
    def scales(self):return self.tile_scales

    def validate(self):
        expected=torch.uint8 if self.asymmetric else torch.int8
        if tuple(self.codes.shape)!=self.shape or self.codes.dtype!=expected:raise ValueError("Invalid tile integer codes")
        if not self.asymmetric and bool((self.codes==-128).any()):raise ValueError("Invalid Sym tile -128")
        validate_fit_scales(self.codes,self.rows,self.columns)
        b,h,k,v=self.shape;shape=(b,h,k//self.tile_size,v//self.tile_size)
        if self.tile_size!=32 or tuple(self.tile_scales.shape)!=shape:raise ValueError("Invalid fixed tile layout")
        precision=metadata_precision(self.metadata_dtype)
        if self.tile_scales.dtype!=(torch.float16 if precision=="fp16" else torch.float32):raise ValueError("Tile scale dtype mismatch")
        if not bool(torch.isfinite(self.tile_scales).all()) or not bool((self.tile_scales>0).all()):raise ValueError("Invalid tile scales")
        if self.asymmetric:
            if self.zero_points is None or self.zero_points.dtype!=torch.uint8 or tuple(self.zero_points.shape)!=shape:raise ValueError("Invalid UINT8 tile zero_points")
        elif self.zero_points is not None:raise ValueError("Unexpected Sym tile zero_point")
        for t in (self.codes,self.rows,self.columns,self.tile_scales,self.zero_points):
            if t is not None:validate_owned_tensor(t)
        return self

    def decode(self):
        self.validate();view=_tiles(self.codes,self.tile_size)
        if self.asymmetric:centered=view.to(torch.int16)-self.zero_points.to(torch.int16)[...,None,None]
        else:centered=view
        normalized=_untile(centered.float()*self.tile_scales.float()[...,None,None],self.shape)
        result=owned((self.rows.float()*self.columns.float())*normalized)
        if not bool(torch.isfinite(result).all()):raise ArithmeticError("Nonfinite tile decode")
        return result

    def signed_equivalent(self):
        if not self.asymmetric:raise ValueError("Only Asym has a UINT8/signed offset mapping")
        return owned((self.codes.to(torch.int16)-128).to(torch.int8)),owned((self.zero_points.to(torch.int16)-128).to(torch.int8))

    def byte_audit(self):
        self.validate()
        return payload_audit(self.codes,(("row_scales",self.rows),("column_scales",self.columns),
            ("tile_scales",self.tile_scales),("zero_points",self.zero_points)))


@torch.no_grad()
def dualm_t0_from_fit(x,rows,columns):
    x=cpu_state(x);validate_fit_scales(x,rows,columns)
    normalized=(x/rows.float())/columns.float()
    if not bool(torch.isfinite(normalized).all()):raise ArithmeticError("Nonfinite original Dual-M normalized state")
    rounded=torch.round(normalized)
    codes=owned(rounded.clamp(-127,127).to(torch.int8))
    return EncodedDualM(tuple(x.shape),codes,owned(rows),owned(columns),dict(
        condition="T0_original_DualM_final_encode",fit_performed_here=False,
        original_rowcol_fit_conditioned_on_original_codes=True,
        normalized_operation="(original_X / rows.float()) / columns.float()",
        rounded_clipping_count=int((rounded.abs()>127).sum()),
        unrounded_outside_code_range_count=int((normalized.abs()>127).sum()),
        original_T0_reference_parity="MODEL_ADAPTER_MUST_VERIFY_INDEPENDENTLY")).validate()


@torch.no_grad()
def encode_tiles(x,rows,columns,asymmetric=False,tile_size=32,metadata_dtype="fp16",eps=1e-12):
    x=cpu_state(x);validate_fit_scales(x,rows,columns)
    precision=metadata_precision(metadata_dtype)
    if tile_size!=32:raise ValueError("Only preregistered 32x32 final tiles are allowed")
    if not isinstance(eps,(float,int)) or not 0<float(eps)<float("inf"):raise ValueError("eps must be finite positive")
    normalized=(x/rows.float())/columns.float()
    if not bool(torch.isfinite(normalized).all()):raise ArithmeticError("Nonfinite original normalized state")
    view=_tiles(normalized,tile_size)
    if asymmetric:
        a=torch.minimum(view.amin((-2,-1)),torch.zeros((),dtype=torch.float32,device="cpu"))
        b=torch.maximum(view.amax((-2,-1)),torch.zeros((),dtype=torch.float32,device="cpu"))
        span=b-a
        if not bool(torch.isfinite(span).all()):raise ArithmeticError("FP32 tile span overflow")
        raw=span.clamp_min(float(eps))/255.0
        scales,diag=round_scales(raw,precision)
        z=torch.round(-a/scales.float()).clamp(0,255)
        if not bool(torch.isfinite(z).all()):raise ArithmeticError("Nonfinite tile zero_point")
        zp=owned(z.to(torch.uint8));levels=view/scales.float()[...,None,None]
        if not bool(torch.isfinite(levels).all()):raise ArithmeticError("Nonfinite tile code quotient")
        shifted=torch.round(levels)+zp.float()[...,None,None]
        clip=(shifted<0)|(shifted>255)
        codes=owned(_untile(shifted.clamp(0,255).to(torch.uint8),x.shape))
        diag.update(zero_group_count=int((span==0).sum()),raw_span_floor_hits=int((span<float(eps)).sum()),
            zero_point_min=int(zp.min()),zero_point_max=int(zp.max()),rounded_clipping_count=int(clip.sum()),
            saturation_code_count=int(((codes==0)|(codes==255)).sum()))
    else:
        maximum=view.abs().amax((-2,-1));raw=maximum.clamp_min(float(eps))/127.0
        scales,diag=round_scales(raw,precision);levels=view/scales.float()[...,None,None]
        if not bool(torch.isfinite(levels).all()):raise ArithmeticError("Nonfinite tile code quotient")
        rounded=torch.round(levels);zp=None
        codes=owned(_untile(rounded.clamp(-127,127).to(torch.int8),x.shape))
        diag.update(zero_group_count=int((maximum==0).sum()),raw_amplitude_floor_hits=int((maximum<float(eps)).sum()),
                    rounded_clipping_count=int((rounded.abs()>127).sum()),saturation_code_count=int((codes.abs()==127).sum()))
    diag.update(condition="T2_32x32_ASYM" if asymmetric else "T1_32x32_SYM",
        tile_size=32,fit_performed_here=False,original_rowcol_fit_conditioned_on_original_codes=True,
        original_rowcol_metadata="FP16 from model-specific original fitter",
        normalized_operation="(original_X / rows.float()) / columns.float()",
        final_encode_input="ORIGINAL_NORMALIZED_X_NOT_T0_RECONSTRUCTION",
        decode_operation="(rows.float()*columns.float()) * (tile_scale.float()*(wide_int(q)-wide_int(z)))",
        eps=float(eps),nonfinite_code_quotient_count=0)
    return EncodedTile(tuple(x.shape),32,bool(asymmetric),precision,codes,owned(rows),owned(columns),scales,zp,diag).validate()

