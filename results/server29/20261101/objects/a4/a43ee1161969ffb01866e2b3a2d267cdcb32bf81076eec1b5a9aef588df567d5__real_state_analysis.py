"""CPU scoring for matched native states; no model or accelerator imports.

Readout vectors are source-specific, time-aligned M_1^T...M_h^T q_h
prepared from actual saved coefficients.  They are not fresh random queries.
The FP32-vs-FP64 difference includes reducer arithmetic and the effective
query transport's FP64-to-FP32 representation sensitivity. It is empirical,
not a bound on model forward error or on the linearization approximation.
"""
from __future__ import annotations
import math
import torch


@torch.no_grad()
def score_native(native, decoded, query_vectors=None):
    if native.device.type != "cpu" or decoded.device.type != "cpu":
        raise ValueError("CPU tensors required")
    if native.ndim != 4 or native.shape != decoded.shape:
        raise ValueError("Expected matched [batch,head,key,value] states")
    if native.shape[-2:] != (128,128):
        raise ValueError("Frozen experiment requires 128x128 heads")
    if not bool(torch.isfinite(native).all() & torch.isfinite(decoded).all()):
        raise ArithmeticError("Nonfinite native or reconstruction")
    # Scoring uses the actual FP32 reconstruction, not ideal unrounded metadata.
    source = native.float()
    restored = decoded.float()
    e64 = restored.double() - source.double()
    e32 = restored - source
    energy = source.double().square().sum((-2,-1))
    sse = e64.square().sum((-2,-1))
    nmse = sse / energy.clamp_min(1e-12)
    n32 = e32.square().sum((-2,-1)) / source.square().sum((-2,-1)).clamp_min(1e-12)
    result = {"state_sse":sse, "state_mse":sse / (128*128),
              "state_nmse":nmse,"native_energy":energy,
              "state_scorer_delta":(n32.double()-nmse).abs()}
    for h, vector in sorted((query_vectors or {}).items()):
        if h not in (1,16,64,128):
            raise ValueError("Horizon outside preregistered set")
        if vector.device.type != "cpu" or tuple(vector.shape) != tuple(native.shape[:2])+(128,):
            raise ValueError("Query transport shape/CPU mismatch")
        if not bool(torch.isfinite(vector).all()):
            raise ArithmeticError("Nonfinite actual query transport")
        response = torch.einsum("bhkv,bhk->bhv", e64, vector.double())
        value = response.square().sum(-1)
        value32 = torch.einsum("bhkv,bhk->bhv",e32,vector.float()).square().sum(-1)
        result["readout_h%d"%h] = value
        result["readout_scorer_delta_h%d"%h] = (value32.double()-value).abs()
    for key,value in result.items():
        if not bool(torch.isfinite(value).all()):
            raise ArithmeticError("Nonfinite score: "+key)
    return result


@torch.no_grad()
def asymmetry_summary(x, architecture, group_size=128):
    """Per-head statistics of actual Hadamard groups; no candidate selection."""
    if x.device.type!="cpu" or x.ndim!=4 or x.shape[-2:]!=(128,128):
        raise ValueError("Expected CPU [B,H,128,128]")
    if architecture not in ("gdn","kda") or group_size not in (64,128):
        raise ValueError("Unregistered architecture/group")
    if not bool(torch.isfinite(x).all()):raise ArithmeticError("Nonfinite input")
    b,h,_,_=x.shape
    if architecture=="gdn":
        view=x.reshape(b,h,128//group_size,group_size,128).movedim(3,-1)
    else:view=x.reshape(b,h,128,128//group_size,group_size)
    # Flatten groups within a head, preserving head isolation.
    v=view.double().reshape(b,h,-1,group_size)
    lo=v.amin(-1); hi=v.amax(-1)
    pos=hi.clamp_min(0);neg=(-lo).clamp_min(0)
    amplitude=torch.maximum(pos,neg)
    span=pos+neg
    balance=(pos-neg).abs()/span.clamp_min(1e-12)
    ratio=pos/neg.clamp_min(1e-12)
    abs_v=v.abs()
    peak=abs_v.amax(-1)
    second=abs_v.topk(2,dim=-1).values[...,1]
    q99=torch.quantile(abs_v,0.99,dim=-1)
    specs={"group_min_mean":lo.mean(-1),"group_max_mean":hi.mean(-1),
           "positive_range_mean":pos.mean(-1),"negative_range_mean":neg.mean(-1),
           "positive_negative_ratio_mean_denominator_floor_1e_minus12":ratio.mean(-1),
           "one_sided_group_fraction":((pos==0)|(neg==0)).double().mean(-1),
           "range_imbalance_mean":balance.mean(-1),
           "range_imbalance_p95":torch.quantile(balance,0.95,dim=-1),
           "sym_ideal_step_mean":(amplitude.clamp_min(1e-12)/127).mean(-1),
           "asym_ideal_step_mean":(span.clamp_min(1e-12)/255).mean(-1),
           "top1_top2_ratio_mean":(peak/second.clamp_min(1e-12)).mean(-1),
           "top1_p99_ratio_mean":(peak/q99.clamp_min(1e-12)).mean(-1),
           "group_count":torch.full((b,h),v.shape[-2],dtype=torch.float64)}
    return specs


def distribution(values):
    vals=[float(x) for x in values if x is not None]
    if not vals:return {"count":0,"mean":None,"median":None,"p95":None,"p99":None,"max":None}
    if not all(math.isfinite(x) for x in vals):raise ArithmeticError("Nonfinite aggregation")
    t=torch.tensor(vals,dtype=torch.float64)
    return {"count":len(vals),"mean":math.fsum(vals)/len(vals),"median":float(torch.quantile(t,.5)),
            "p95":float(torch.quantile(t,.95)),"p99":float(torch.quantile(t,.99)),"max":max(vals)}
