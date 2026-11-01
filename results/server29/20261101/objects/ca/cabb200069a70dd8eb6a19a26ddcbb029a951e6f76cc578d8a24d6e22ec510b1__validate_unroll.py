"""Non-AIME synthetic correctness, structure and lifecycle verification.

No model, tokenizer, answers or generation. CPU is a prototype/reference check;
CUDA executes the same packed kernel intended for formal decode.
"""
import argparse, json, math, sys, time, traceback, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from pace_buffer import PACEBuffer, native_l2norm
from pace_quantizer import PackedAffineState
from lq_unroll_buffer import LQUnrollBuffer


class ReplayReference(PACEBuffer):
    def __init__(self,period,mask,**kwargs):
        super().__init__(8,mask,**kwargs)
        self.decode_period=period
        self.last_delta=None
        self.last_boundary_state=None
    def _replay(self):
        state=self._base()
        for i in range(self.pending_tokens):
            token=self._slots[i]
            key=self._token_keys(token)
            state=state*token.log_decay.exp().unsqueeze(-1).unsqueeze(-1)
            delta=token.value.float()-(state*key.unsqueeze(-1)).sum(-2)
            delta=delta*token.beta.float().unsqueeze(-1)
            state=state+key.unsqueeze(-1)*delta.unsqueeze(-2)
            self.last_delta=delta.clone()
        return state
    def _commit(self,state,reason):
        self.last_boundary_state=state.clone()
        return super()._commit(state,reason)


def metrics(a,b):
    aa,bb=a.double().reshape(-1),b.double().reshape(-1)
    diff=aa-bb
    rms=float(diff.square().mean().sqrt())
    denom=float(bb.square().mean().sqrt())
    return dict(max_absolute_error=float(diff.abs().max()),
        max_relative_error=float((diff.abs()/bb.abs().clamp_min(1e-12)).max()),
        RMSE=rms,normalized_RMSE=rms/max(denom,1e-30),
        cosine_similarity=float(torch.nn.functional.cosine_similarity(aa,bb,dim=0)) if denom else None,
        mismatch_count=int((aa!=bb).sum()),elements=aa.numel(),
        nonfinite_count=int((~torch.isfinite(aa)|~torch.isfinite(bb)).sum()))


def packed_diff(a,b):
    rows={}
    for bits in (4,8):
        x,y=getattr(a,'int'+str(bits)),getattr(b,'int'+str(bits))
        for name in ['codes','scales','zero_points','scale_exponents']:
            u,v=getattr(x,name),getattr(y,name)
            label=str(bits)+'.'+name
            if u is None or v is None:
                rows[label]=dict(both_absent=u is None and v is None,shape_mismatch=(u is None)!=(v is None))
            else:
                rows[label]=metrics(u,v)
        rows[str(bits)+'.integer_codes']=metrics(x.integer_codes(),y.integer_codes())
    return rows


def explain_rounding(actual,reference,sa,sr):
    """Explain EVERY unequal code by the unchanged encoder's exact inputs.

    Explanations do not assert bit equivalence. Near a rounding boundary,
    FP32 expansion changes a stored code/scale and creates a new numerical path.
    Only bounded scalar counterexamples are retained, never dense raw state.
    """
    counts={}; examples=[]; unexplained=0; code_diffs=0
    for bits,heads in [(4,actual.int4_heads),(8,actual.int8_heads)]:
        x,y=getattr(actual,'int'+str(bits)),getattr(reference,'int'+str(bits))
        xa=sa[:,list(heads)].permute(0,1,3,2).contiguous().reshape(x.shape).double()
        xr=sr[:,list(heads)].permute(0,1,3,2).contiguous().reshape(y.shape).double()
        ca,cr=x.integer_codes(),y.integer_codes()
        indices=(ca!=cr).nonzero().cpu().tolist()
        za,zr=x.integer_zero_points(),y.integer_zero_points()
        for ix in indices:
            key=tuple(ix);group=tuple(ix[:-1]);code_diffs+=1
            ea=0 if x.scale_exponents is None else int(x.scale_exponents[group])
            er=0 if y.scale_exponents is None else int(y.scale_exponents[group])
            scalea=float(x.scales[group])*2.0**ea;scaler=float(y.scales[group])*2.0**er
            va=float(xa[key]/scalea);vr=float(xr[key]/scaler)
            zeroa,zeror=int(za[group]),int(zr[group])
            qa=max(0,min((1<<bits)-1,round(va)+zeroa))
            qr=max(0,min((1<<bits)-1,round(vr)+zeror))
            assert qa==int(ca[key]) and qr==int(cr[key]),'Encoder reproduction error'
            scale_ulp=abs(int(x.scales[group].view(torch.int16))-int(y.scales[group].view(torch.int16)))
            if ea==er and scale_ulp==0 and zeroa==zeror and abs(qa-qr)==1 and abs(va-vr)<1e-2:
                reason='INTEGER_CODE_RNE_BOUNDARY'
            elif ea==er and scale_ulp<=1 and abs(zeroa-zeror)<=1 and abs(qa-qr)<=2:
                reason='ADJACENT_FP16_SCALE_OR_INTEGER_ZERO_ROUNDING_BOUNDARY'
            else:
                reason='UNEXPLAINED';unexplained+=1
            counts[reason]=counts.get(reason,0)+1
            if len(examples)<8:
                examples.append(dict(bits=bits,index=ix,reason=reason,code_actual=qa,
                    code_reference=qr,scaled_input_actual=va,scaled_input_reference=vr,
                    scale_FP16_ULP_difference=scale_ulp,exponent_actual=ea,exponent_reference=er,
                    zero_point_actual=zeroa,zero_point_reference=zeror,
                    absolute_preencode_state_difference=abs(float(xa[key]-xr[key]))))
    return dict(code_mismatches=code_diffs,unexplained=unexplained,reasons=counts,examples=examples)


def run_case(period,length,prompt,device,scenario='normal',heads=4):
    mask=[i%4==0 for i in range(heads)]
    a=ReplayReference(period,mask,key_heads=heads//2)
    b=LQUnrollBuffer(period,mask,key_heads=heads//2,
        validation_cpu=device=='cpu',capture_validation=True,verify_batched_keys=True)
    a.set_prompt_length(prompt);b.set_prompt_length(prompt)
    generator=torch.Generator(device='cpu').manual_seed(6192026+length*31+prompt)
    if scenario=='exponent':
        init=(torch.randn((1,heads,128,128),generator=generator)*1e8).to(device)
        for obj in (a,b):
            obj._shape=tuple(init.shape);obj.packed_state=PackedAffineState.from_tensor(init,mask)
    rows=[];first=None
    for t in range(prompt+length):
        def bf(shape):return (torch.randn(shape,generator=generator)*.2).to(device=device,dtype=torch.bfloat16)
        q=bf((1,heads//2,128));k=bf((1,heads//2,128));v=bf((1,heads,128))
        beta=torch.rand((1,heads),generator=generator).to(device=device,dtype=torch.bfloat16)
        logg=-(torch.rand((1,heads),generator=generator)*.3+.001)
        if scenario=='strong_decay':logg=logg*100
        if scenario=='weak_decay':logg=logg*.00001
        g=logg.to(device)
        ya=a.step(q,k,v,beta,g);yb=b.step(q,k,v,beta,g)
        assert a.tokens==b.tokens and a.pending_tokens==b.pending_tokens
        assert a.writeback_count==b.writeback_count
        row=dict(forward=t+1,decode_forward=max(0,t+1-prompt),output=metrics(yb,ya),
                 pending=b.pending_tokens,writebacks=b.writeback_count)
        if t>=prompt:
            row['delta']=metrics(b.last_delta,a.last_delta)
            # Separate algebraic self-consistency from cross-codec trajectory drift.
            # Validation-only materialization of the actual new delta records.
            if b.pending_tokens:
                state=b._delta_state(validation=True)
                query=native_l2norm(q.repeat_interleave(2,dim=1)).float()/math.sqrt(128)
                expected=(state*query.unsqueeze(-1)).sum(-2).to(torch.bfloat16)
                row['actual_delta_state_readout']=metrics(yb,expected)
            if b.pending_tokens==0:
                row['boundary_state']=metrics(b.last_boundary_state,a.last_boundary_state)
                row['packed']=packed_diff(b.packed_state,a.packed_state)
                row['rounding_explanation']=explain_rounding(b.packed_state,a.packed_state,b.last_boundary_state,a.last_boundary_state)
        elif b.pending_tokens==0:
            for n,tensor in b.packed_state.resident_tensors().items():
                assert torch.equal(tensor,a.packed_state.resident_tensors()[n]),'Unchanged prefill metadata differs'
        if first is None and row['output']['mismatch_count']:
            first=t+1
        rows.append(row)
    prev=(a.tokens,a.writeback_count,a.pending_tokens,b.tokens,b.writeback_count,b.pending_tokens)
    ra=a.finalize('preserve');rb=b.finalize('preserve')
    assert prev==(a.tokens,a.writeback_count,a.pending_tokens,b.tokens,b.writeback_count,b.pending_tokens)
    assert rb['decode_dense_materializations']==length//period
    assert rb['decode_direct_calls']==length
    assert b.pending_tokens==length%period
    expected_bytes=17696*(heads/32)
    assert rb['decode_buffer_capacity_nbytes']==int(expected_bytes*period)
    assert rb['dense_state_persistent'] is False
    production=[(n,t) for n,t in b.resident_tensors().items() if not n.startswith('VALIDATION_ONLY')]
    assert not any(tuple(t.shape)==tuple(b._shape) and t.is_floating_point() for n,t in production)
    return dict(period=period,length=length,prompt=prompt,scenario=scenario,heads=heads,
        first_BF16_output_divergence=first,rows=rows,reference_storage=ra,direct_storage=rb,
        no_tail_flush=True,no_extra_forward=True,
        actual_decode_boundary_materializations=b.decode_dense_materializations,
        terminal_pending=b.pending_tokens)


def run(device,output,quick=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start=time.monotonic();cases=[]
    lengths=[1,2,7,8,9,15,16,17,31,32,33]
    for p in [8,16]:
        for n in lengths:
            cases.append(run_case(p,n,0,device))
            print(json.dumps(dict(phase='NON_AIME_CASE',period=p,length=n,completed=len(cases))),flush=True)
        for prompt,n in [(1,17),(7,33),(9,17)]:
            cases.append(run_case(p,n,prompt,device))
        for scenario in ['strong_decay','weak_decay','exponent']:
            cases.append(run_case(p,33,0,device,scenario))
        if not quick:
            cases.append(run_case(p,33,257,device))
            cases.append(run_case(p,33,3,device,heads=32))
            cases.append(run_case(p,257,0,device,'weak_decay'))
    # Explicit invalid inputs must fail closed before a production readout.
    invalid=[]
    for bad in ['nan_q','inf_v','positive_g','bad_beta']:
        obj=LQUnrollBuffer(8,[True,False,False,False],key_heads=2,validation_cpu=device=='cpu')
        obj.set_prompt_length(0)
        q=torch.ones((1,2,128),dtype=torch.bfloat16,device=device);k=q.clone()
        v=torch.ones((1,4,128),dtype=torch.bfloat16,device=device)
        beta=torch.ones((1,4),dtype=torch.bfloat16,device=device)*.5
        g=torch.ones((1,4),dtype=torch.float32,device=device)*-.1
        if bad=='nan_q':q[0,0,0]=float('nan')
        if bad=='inf_v':v[0,0,0]=float('inf')
        if bad=='positive_g':g[0,0]=.1
        if bad=='bad_beta':beta[0,0]=2
        rejected=False
        try:obj.step(q,k,v,beta,g)
        except (ValueError,FloatingPointError):rejected=True
        assert rejected,bad
        invalid.append(dict(case=bad,rejected=True))
    flat=[r for c in cases for r in c['rows']]
    worst={}
    for field in ['output','delta','boundary_state','actual_delta_state_readout']:
        subset=[r[field] for r in flat if field in r]
        worst[field]={k:max(x[k] for x in subset) for k in ['max_absolute_error','normalized_RMSE','mismatch_count']}
        assert all(x['nonfinite_count']==0 for x in subset)
    # This collector does not silently grant numerical acceptance from completion.
    # A separate preregistered numerical-admission review consumes all metrics.
    result=dict(status='COMPLETE_MEASUREMENTS_PENDING_NUMERIC_ADMISSION',device=device,
        actual_CUDA=device.startswith('cuda'),new_AIME_samples=0,model_loads=0,
        elapsed_s=time.monotonic()-start,cases=cases,invalid_cases=invalid,worst=worst,
        numerical_gate_granted=False,prototype_only=device=='cpu')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],device=device,cases=len(cases),worst=worst,elapsed_s=result['elapsed_s']),allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    try:run(args.device,args.output,args.quick)
    except BaseException as exc:
        failure=dict(status='VALIDATION_EXECUTION_FAILED',error=repr(exc),traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if not args.output.exists():args.output.write_text(json.dumps(failure,indent=2))
        print(json.dumps(failure),flush=True);raise
