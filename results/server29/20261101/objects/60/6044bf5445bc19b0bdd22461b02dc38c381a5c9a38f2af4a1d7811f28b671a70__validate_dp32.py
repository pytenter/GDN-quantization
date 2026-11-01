"""New period only: original collector and same-base causal review; no model."""
import copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from validate_unroll import run_case,ReplayReference,metrics,explain_rounding
from lq_unroll_buffer import LQUnrollBuffer
from pace_quantizer import PackedAffineState

def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def lineage(case):
    period=32;heads=case['heads'];prompt=case['prompt'];length=case['length'];scenario=case['scenario'];mask=[i%4==0 for i in range(heads)]
    ref=ReplayReference(period,mask,key_heads=heads//2)
    direct=LQUnrollBuffer(period,mask,key_heads=heads//2,capture_validation=True)
    for x in [ref,direct]:x.set_prompt_length(prompt)
    gen=torch.Generator(device='cpu').manual_seed(6192026+length*31+prompt)
    if scenario=='exponent':
        init=(torch.randn((1,heads,128,128),generator=gen)*1e8).cuda()
        for x in [ref,direct]:x._shape=tuple(init.shape);x.packed_state=PackedAffineState.from_tensor(init,mask)
    rows=[]
    for t in range(prompt+length):
        if t>=prompt and (t-prompt)%period==0:
            shadow=ReplayReference(period,mask,key_heads=heads//2);shadow.set_prompt_length(0)
            shadow.prompt_length=prompt;shadow.tokens=t;shadow._shape=direct._shape;shadow.packed_state=copy.deepcopy(direct.packed_state)
        def bf(shape):return (torch.randn(shape,generator=gen)*.2).to(device='cuda',dtype=torch.bfloat16)
        q,k,v=bf((1,heads//2,128)),bf((1,heads//2,128)),bf((1,heads,128))
        beta=torch.rand((1,heads),generator=gen).to(device='cuda',dtype=torch.bfloat16)
        logg=-(torch.rand((1,heads),generator=gen)*.3+.001)
        if scenario=='strong_decay':logg=logg*100
        if scenario=='weak_decay':logg=logg*.00001
        g=logg.cuda();yr=ref.step(q,k,v,beta,g);yd=direct.step(q,k,v,beta,g)
        if t<prompt:continue
        ys=shadow.step(q,k,v,beta,g)
        row=dict(forward=t+1,same_base_output=metrics(yd,ys),cross_output=metrics(yd,yr))
        if direct.pending_tokens==0:
            local=explain_rounding(direct.packed_state,shadow.packed_state,direct.last_boundary_state,shadow.last_boundary_state)
            legacy=explain_rounding(direct.packed_state,ref.packed_state,direct.last_boundary_state,ref.last_boundary_state)
            row.update(local_rounding=local,legacy_cross_rounding=legacy,same_base_delta=metrics(direct.last_delta,shadow.last_delta),same_base_boundary=metrics(direct.last_boundary_state,shadow.last_boundary_state))
        rows.append(row)
    return dict(period=period,length=length,prompt=prompt,scenario=scenario,heads=heads,rows=rows)

def main():
    Path(sys.argv[1]).parent.mkdir(parents=True,exist_ok=True)
    from compile_telemetry import install
    install(Path(sys.argv[1]).with_suffix('.compile.json'))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(False)
    cfg=json.loads((ROOT/'CONFIG.json').read_text());limits=cfg['tolerances'];start=time.monotonic();cases=[]
    roster=[(n,0,'normal',4) for n in [1,2,7,8,9,15,16,17,31,32,33,63,64,65,97]]
    roster += [(65,p,'normal',4) for p in [1,7,9,257]]
    roster += [(65,0,s,4) for s in ['strong_decay','weak_decay','exponent']]
    roster += [(65,3,'normal',32),(257,0,'weak_decay',4)]
    for n,p,s,h in roster:
        cases.append(run_case(32,n,p,'cuda',s,h))
        print(json.dumps(dict(phase='DP32_NUMERIC_CASE',length=n,prompt=p,scenario=s,heads=h,completed=len(cases))),flush=True)
    flat=[x for c in cases for x in c['rows']];worst={}
    for field in ['output','delta','boundary_state','actual_delta_state_readout']:
        subset=[x[field] for x in flat if field in x]
        worst[field]=max(x['normalized_RMSE'] for x in subset)
        assert all(x['nonfinite_count']==0 for x in subset)
    invalid=[]
    for bad in ['nan_q','inf_v','positive_g','bad_beta']:
        obj=LQUnrollBuffer(32,[True,False,False,False],key_heads=2);obj.set_prompt_length(0)
        q=torch.ones((1,2,128),dtype=torch.bfloat16,device='cuda');k=q.clone()
        v=torch.ones((1,4,128),dtype=torch.bfloat16,device='cuda');beta=torch.full((1,4),.5,dtype=torch.bfloat16,device='cuda');g=torch.full((1,4),-.1,device='cuda')
        if bad=='nan_q':q[0,0,0]=float('nan')
        if bad=='inf_v':v[0,0,0]=float('inf')
        if bad=='positive_g':g[0,0]=.1
        if bad=='bad_beta':beta[0,0]=2
        try:obj.step(q,k,v,beta,g)
        except (ValueError,FloatingPointError):invalid.append(bad)
        else:raise AssertionError('Invalid input accepted: '+bad)
    selected=[c for c in cases if (c['length']==257 and c['scenario']=='weak_decay') or any(x.get('rounding_explanation',{}).get('unexplained',0) for x in c['rows'])]
    lineage_cases=[]
    for c in selected:
        lineage_cases.append(lineage(c));print(json.dumps(dict(phase='DP32_LINEAGE_CASE',length=c['length'],prompt=c['prompt'],scenario=c['scenario'],heads=c['heads'])),flush=True)
    local=[x for c in lineage_cases for x in c['rows']];unexplained=sum(x['local_rounding']['unexplained'] for x in local if 'local_rounding' in x)
    local_max={f:max(x[f]['normalized_RMSE'] for x in local if f in x) for f in ['same_base_output','same_base_delta','same_base_boundary']}
    accepted=(all(worst[f]<=limits['cross_replay_max_normalized_RMSE'] for f in ['output','delta','boundary_state']) and worst['actual_delta_state_readout']<=limits['self_actual_delta_state_readout_max_normalized_RMSE'] and unexplained==0 and max(local_max.values())<=limits['cross_replay_max_normalized_RMSE'])
    from model_engineering import runtime_identity
    result=dict(status='PASS_DP32_ORIGINAL_NUMERICAL_GATES' if accepted else 'STOP_NUMERICAL_VALIDATION_FAILED',runtime=runtime_identity(),period=32,cases=cases,worst_NRMSE=worst,original_unchanged_limits=limits,same_base_lineage=lineage_cases,same_base_worst_NRMSE=local_max,local_unexplained_code_mismatches=unexplained,all_cases_with_legacy_UNEXPLAINED_reviewed=True,invalid_rejected=invalid,elapsed_s=time.monotonic()-start,new_AIME_samples=0,dense_reference_validation_only=True,original_cross_heuristic_labels_retained=True)
    save(Path(sys.argv[1]),result)
    print(json.dumps(dict(phase='DP32_VALIDATION_DONE',status=result['status'],worst_NRMSE=worst,local_unexplained=unexplained)),flush=True)
    return 0 if accepted else 2
if __name__=='__main__':raise SystemExit(main())
