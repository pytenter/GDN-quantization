"""Independent CPU FP64 packed decode + direct-formula oracle for actual GPU records.

Validation only. Dense CPU base exists here, never in the production fast path.
NumPy decoder is independent of the production codec's dequantize method.
Only error summaries are saved; no state values/tensors are downloaded to Mac.
"""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import lq_unroll_buffer as L
from pace_quantizer import PackedAffineState


def independent_base(base):
    b,h,k,v=base.logical_shape
    result=np.empty((b,h,k,v),dtype=np.float64)
    for heads,part in [(base.int4_heads,base.int4),(base.int8_heads,base.int8)]:
        if part is None:continue
        codes=part.codes.cpu().numpy()
        zeros=part.zero_points.cpu().numpy()
        if part.bits==4:
            codes=np.stack([codes&15,codes>>4],axis=-1).reshape(-1)
            zeros=np.stack([zeros&15,zeros>>4],axis=-1).reshape(-1)
        codes=codes[:math.prod(part.shape)].reshape(part.shape).astype(np.int16)
        zeros=zeros[:math.prod(part.shape[:-1])].reshape(part.shape[:-1]).astype(np.int16)
        scale=part.scales.cpu().numpy().astype(np.float64)
        if part.scale_exponents is not None:
            scale=np.ldexp(scale,part.scale_exponents.cpu().numpy().astype(np.int32))
        decoded=(codes-zeros[...,None])*scale[...,None]
        result[:,list(heads)]=decoded.reshape(b,len(heads),v,k).transpose(0,1,3,2)
    return result


def metric(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    diff=a-b;rmse=float(np.sqrt(np.mean(diff*diff)))
    return dict(max_absolute_error=float(np.max(np.abs(diff))),RMSE=rmse,
        normalized_RMSE=rmse/max(float(np.sqrt(np.mean(b*b))),1e-30),
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),elements=a.size)


def oracle(base,pk,pu,alpha,q,k,value,beta):
    B=independent_base(base)
    def array(t):return t.float().cpu().numpy().astype(np.float64)
    keys,updates,aa,qq,kk,vv,bb=map(array,(pk,pu,alpha,q,k,value,beta))
    f_q=np.sum(B*qq[...,None],axis=-2)
    f_k=np.sum(B*kk[...,None],axis=-2)
    weight=np.ones_like(aa[0])
    for j in range(len(keys)):weight=weight*aa[j]
    f_q=f_q*weight[...,None];f_k=f_k*weight[...,None]
    for i in range(len(keys)):
        tail=np.ones_like(aa[0])
        for j in range(i+1,len(keys)):tail=tail*aa[j]
        f_q=f_q+updates[i]*(np.sum(keys[i]*qq,axis=-1)*tail)[...,None]
        f_k=f_k+updates[i]*(np.sum(keys[i]*kk,axis=-1)*tail)[...,None]
    u=bb[...,None]*(vv-aa[-1,...,None]*f_k)
    out=aa[-1,...,None]*f_q+u*np.sum(kk*qq,axis=-1)[...,None]
    return u,out


def run(output):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    masks=json.loads((ROOT/'calibration/head_mask.json').read_text())['layers']
    original=L.packed_readout;rows=[];context={}
    def checked(base,pk,pu,alpha,q,k,value,beta,maps):
        u,out=original(base,pk,pu,alpha,q,k,value,beta,maps)
        if pk.shape[0] in (0,1,7,15):
            expected_u,expected_y=oracle(base,pk,pu,alpha,q,k,value,beta)
            actual_u=u.cpu().numpy()
            actual_y=out.float().cpu().numpy()
            rounded=torch.from_numpy(expected_y).to(torch.bfloat16).float().numpy()
            row=dict(context,pending=pk.shape[0],u_FP32_vs_FP64=metric(actual_u,expected_u),
                output_BF16_vs_FP64=metric(actual_y,expected_y),
                output_BF16_vs_oracle_BF16=metric(actual_y,rounded),
                BF16_mismatch_count=int(np.count_nonzero(actual_y!=rounded)))
            rows.append(row)
        return u,out
    L.packed_readout=checked
    start=time.monotonic()
    for period in (8,16):
        for layer,desc in masks.items():
            mask=[i in desc['int8_heads'] for i in range(32)]
            obj=L.LQUnrollBuffer(period,mask);obj.set_prompt_length(0)
            context.update(period=period,layer=int(layer),scenario='normal')
            for t in range(17):
                q=torch.randn((1,16,128),device='cuda',dtype=torch.bfloat16)*.2
                k=torch.randn_like(q)*.2
                v=torch.randn((1,32,128),device='cuda',dtype=torch.bfloat16)*.2
                beta=torch.full((1,32),.5,device='cuda',dtype=torch.bfloat16)
                g=torch.full((1,32),-.001,device='cuda',dtype=torch.float32)
                obj.step(q,k,v,beta,g)
            assert obj.validation_dense_materializations==0
            print(json.dumps(dict(phase='SAME_STATE_ORACLE',period=period,layer=int(layer),comparisons=len(rows))),flush=True)
        # Actual optional-exponent metadata is also decoded independently.
        obj=L.LQUnrollBuffer(period,mask);obj.set_prompt_length(0)
        state=torch.randn((1,32,128,128),device='cuda')*1e8
        obj._shape=tuple(state.shape);obj.packed_state=PackedAffineState.from_tensor(state,mask)
        context.update(period=period,layer=int(layer),scenario='exponent')
        for t in range(17):obj.step(q,k,v,beta,g)
    worst={field:max(x[field]['normalized_RMSE'] for x in rows) for field in ['u_FP32_vs_FP64','output_BF16_vs_FP64','output_BF16_vs_oracle_BF16']}
    assert all(x[f]['finite'] for x in rows for f in worst)
    result=dict(status='COMPLETE_INDEPENDENT_SAME_STATE_FP64_COMPARISONS',comparisons=len(rows),
        rows=rows,worst_normalized_RMSE=worst,all24_actual_head_masks=True,
        production_source_modified=False,new_AIME_samples=0,model_loads=0,
        elapsed_s=time.monotonic()-start,oracle_dense_CPU_only=True,
        scope='Same actual GPU packed base, normalized keys/queries, FP32 u records and actual GPU decay values. This isolates readout correctness from quantized-trajectory divergence.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],comparisons=len(rows),worst=worst)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
