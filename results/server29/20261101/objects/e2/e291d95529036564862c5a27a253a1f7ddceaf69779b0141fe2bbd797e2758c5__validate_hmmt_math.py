"""Non-HMMT mathematical and rotated direct-state gates, actual CUDA."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from hmmt_adapters import hadamard128,rotate_bf16
from pace_buffer import native_l2norm
from pace_quantizer import PackedAffineState
from lq_unroll_buffer import LQUnrollBuffer
from validate_unroll import ReplayReference,metrics,packed_diff,explain_rounding
from model_engineering import runtime_identity

def emit(phase,**kw):print(json.dumps(dict(phase=phase,monotonic=time.monotonic(),**kw)),flush=True)
def step(state,q,k,v,beta,g):
    state=state*g.exp()[...,None,None]
    u=(v.float()-(state*k[...,None]).sum(-2))*beta.float()[...,None]
    state=state+k[...,None]*u[...,None,:]
    return state,(state*q[...,None]/math.sqrt(128)).sum(-2),u
def fp_case(length,headcount=4):
    h=hadamard128('cuda');gen=torch.Generator(device='cpu').manual_seed(132001+length)
    s=torch.zeros((1,headcount,128,128),device='cuda');ideal=s.clone();actual=s.clone()
    records=[]
    for index in range(length):
        q=(torch.randn((1,headcount,128),generator=gen)*.2).to('cuda',torch.bfloat16)
        k=(torch.randn((1,headcount,128),generator=gen)*.2).to('cuda',torch.bfloat16)
        v=(torch.randn((1,headcount,128),generator=gen)*.2).to('cuda',torch.bfloat16)
        beta=torch.rand((1,headcount),generator=gen).to('cuda',torch.bfloat16)
        g=-(torch.rand((1,headcount),generator=gen)*.1+.01).to('cuda')
        nq,nk=native_l2norm(q).float(),native_l2norm(k).float()
        s,y,u=step(s,nq,nk,v,beta,g)
        ideal,yi,ui=step(ideal,nq@h,nk@h,v,beta,g)
        actual,ya,ua=step(actual,native_l2norm(rotate_bf16(q,h)).float(),native_l2norm(rotate_bf16(k,h)).float(),v,beta,g)
        expected=h@s
        records.append(dict(token=index+1,ideal_output=metrics(yi,y),ideal_state=metrics(ideal,expected),
                            actual_output=metrics(ya.to(torch.bfloat16),y.to(torch.bfloat16)),actual_state=metrics(actual,expected),
                            delta=metrics(ua,u)))
    return dict(length=length,heads=headcount,rows=records)

def direct_case(length,prompt=0,scenario='normal',heads=4):
    mask=[i%4==0 for i in range(heads)];h=hadamard128('cuda')
    a=ReplayReference(8,mask,key_heads=heads//2)
    b=LQUnrollBuffer(8,mask,key_heads=heads//2,capture_validation=True,verify_batched_keys=True)
    for obj in [a,b]:obj.set_prompt_length(prompt)
    gen=torch.Generator(device='cpu').manual_seed(132123+length*31+prompt)
    if scenario=='exponent':
        init=(torch.randn((1,heads,128,128),generator=gen)*1e8).to('cuda')
        for obj in [a,b]:obj._shape=tuple(init.shape);obj.packed_state=PackedAffineState.from_tensor(init,mask)
    rows=[]
    for index in range(prompt+length):
        def bf(shape):return (torch.randn(shape,generator=gen)*.2).to('cuda',torch.bfloat16)
        q,k=rotate_bf16(bf((1,heads//2,128)),h),rotate_bf16(bf((1,heads//2,128)),h)
        v=bf((1,heads,128));beta=torch.rand((1,heads),generator=gen).to('cuda',torch.bfloat16)
        g=-(torch.rand((1,heads),generator=gen)*.3+.001).to('cuda')
        if scenario=='weak_decay':g=g*.00001
        if scenario=='strong_decay':g=g*100
        ya=a.step(q,k,v,beta,g);yb=b.step(q,k,v,beta,g)
        assert a.tokens==b.tokens and a.pending_tokens==b.pending_tokens and a.writeback_count==b.writeback_count
        row=dict(forward=index+1,decode_forward=max(0,index+1-prompt),output=metrics(yb,ya),pending=b.pending_tokens,writebacks=b.writeback_count)
        if index>=prompt:
            row['delta']=metrics(b.last_delta,a.last_delta)
            state=b._delta_state(validation=True) if b.pending_tokens else b.last_boundary_state
            query=native_l2norm(q.repeat_interleave(2,1)).float()/math.sqrt(128)
            row['same_actual_state_readout']=metrics(yb,(state*query[...,None]).sum(-2).to(torch.bfloat16))
            if not b.pending_tokens:
                row['boundary_state']=metrics(b.last_boundary_state,a.last_boundary_state)
                row['packed_cross']=packed_diff(b.packed_state,a.packed_state)
                row['cross_rounding_labels']=explain_rounding(b.packed_state,a.packed_state,b.last_boundary_state,a.last_boundary_state)
                encoded=PackedAffineState.from_tensor(b.last_boundary_state,mask)
                for name,t in b.packed_state.resident_tensors().items():assert torch.equal(t,encoded.resident_tensors()[name])
                row['unchanged_encoder_reproduction_exact']=True
        rows.append(row)
    previous=(b.tokens,b.pending_tokens,b.writeback_count);a.finalize('preserve');storage=b.finalize('preserve')
    assert previous==(b.tokens,b.pending_tokens,b.writeback_count)
    assert b.decode_direct_calls==length and b.decode_dense_materializations==length//8 and b.pending_tokens==length%8
    return dict(length=length,prompt=prompt,scenario=scenario,heads=heads,rows=rows,storage=storage,no_tail_flush=True)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    spec=json.loads((ROOT/'validation/NUMERICAL_PREREGISTRATION.json').read_text())
    emit('MATHEMATICAL_FP_H_GATES')
    fp=[fp_case(n) for n in spec['lengths']]+[fp_case(32,32)]
    ideal=max(row[k]['normalized_RMSE'] for c in fp for row in c['rows'] for k in ['ideal_output','ideal_state'])
    actual_output=max(row['actual_output']['normalized_RMSE'] for c in fp for row in c['rows'])
    actual_state=max(row['actual_state']['normalized_RMSE'] for c in fp for row in c['rows'])
    emit('FP_H_GATES_DONE',ideal=ideal,actual_output=actual_output,actual_state=actual_state)
    direct=[]
    for n in spec['lengths']:
        direct.append(direct_case(n));emit('ROTATED_DIRECT_CASE',length=n,prompt=0)
    for prompt in [1,7,17,257]:
        direct.append(direct_case(33,prompt));emit('ROTATED_DIRECT_CASE',length=33,prompt=prompt)
    for scenario in ['weak_decay','strong_decay','exponent']:direct.append(direct_case(33,scenario=scenario))
    direct.append(direct_case(33,prompt=3,heads=32))
    cross=max(row[k]['normalized_RMSE'] for c in direct for row in c['rows'] for k in ['output','delta','boundary_state'] if k in row)
    self_error=max(row['same_actual_state_readout']['normalized_RMSE'] for c in direct for row in c['rows'] if 'same_actual_state_readout' in row)
    passed=dict(ideal_coordinate=ideal<=spec['ideal_FP32_Hadamard']['normalized_RMSE'],
                production_BF16_FP=(actual_output<=.005 and actual_state<=.005),
                rotated_direct=(cross<=.005 and self_error<=.001))
    output=dict(status='PASS_ALL_NON_HMMT_MATH_GATES' if all(passed.values()) else 'NON_HMMT_MATH_GATES_FAILED',
                passed=passed,runtime=runtime_identity(),preregistration=spec,fp=fp,rotated_direct=direct,
                maxima=dict(ideal=ideal,production_FP_output=actual_output,production_FP_state=actual_state,cross_direct=cross,self_actual_state=self_error),
                new_HMMT_generations=0,thresholds_changed=False)
    p=ROOT/'validation/MATH_CUDA_V1.json'
    with p.open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    emit('MATH_VALIDATION_FINISHED',status=output['status'],passed=passed,maxima=output['maxima'])
    return 0 if all(passed.values()) else 2
if __name__=='__main__':raise SystemExit(main())
