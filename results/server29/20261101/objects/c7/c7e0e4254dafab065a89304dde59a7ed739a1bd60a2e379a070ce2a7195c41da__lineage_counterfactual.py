"""Non-AIME same-window-base interventions; no production method changes.

Each shadow sequential replay receives the actual Direct base at the start of
each window. This distinguishes inherited quantized-base differences from local
algebra/reduction differences. Original local heuristic failures are retained.
"""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from validate_unroll import ReplayReference,metrics,explain_rounding
from lq_unroll_buffer import LQUnrollBuffer

def run(output):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[]
    for period in (8,16):
        mask=[True,False,False,False];heads=4
        ref=ReplayReference(period,mask,key_heads=2)
        direct=LQUnrollBuffer(period,mask,key_heads=2,capture_validation=True)
        for x in (ref,direct):x.set_prompt_length(0)
        gen=torch.Generator(device='cpu').manual_seed(6192026+257*31)
        shadow=None;last_base_equal=True;first_base_split=None
        for t in range(257):
            if t%period==0:
                shadow=ReplayReference(period,mask,key_heads=2);shadow.set_prompt_length(0)
                shadow.tokens=t;shadow._shape=direct._shape
                shadow.packed_state=copy.deepcopy(direct.packed_state)
                if direct.packed_state is not None:
                    last_base_equal=all(torch.equal(v,ref.packed_state.resident_tensors()[n]) for n,v in direct.packed_state.resident_tensors().items())
            def bf(shape):return (torch.randn(shape,generator=gen)*.2).to(device='cuda',dtype=torch.bfloat16)
            q=bf((1,2,128));k=bf((1,2,128));v=bf((1,4,128))
            beta=torch.rand((1,4),generator=gen).to(device='cuda',dtype=torch.bfloat16)
            logg=-(torch.rand((1,4),generator=gen)*.3+.001)*.00001
            g=logg.cuda()
            ya=ref.step(q,k,v,beta,g);yb=direct.step(q,k,v,beta,g);ys=shadow.step(q,k,v,beta,g)
            row=dict(period=period,forward=t+1,same_window_base_output=metrics(yb,ys),cross_trajectory_output=metrics(yb,ya))
            if direct.pending_tokens==0:
                legacy=explain_rounding(direct.packed_state,ref.packed_state,direct.last_boundary_state,ref.last_boundary_state)
                local=explain_rounding(direct.packed_state,shadow.packed_state,direct.last_boundary_state,shadow.last_boundary_state)
                inherited=local_and_inherited=0
                for bits in (4,8):
                    actual=getattr(direct.packed_state,'int'+str(bits)).integer_codes()
                    historical=getattr(ref.packed_state,'int'+str(bits)).integer_codes()
                    intervened=getattr(shadow.packed_state,'int'+str(bits)).integer_codes()
                    different=actual!=historical
                    inherited+=int((different & (actual==intervened)).sum())
                    local_and_inherited+=int((different & (actual!=intervened)).sum())
                if legacy['code_mismatches'] and first_base_split is None:first_base_split=t+1
                row.update(window_start_packed_bases_equal=last_base_equal,first_quantized_split=first_base_split,
                    legacy_local_heuristic=legacy,same_base_local_rounding=local,
                    cross_mismatches_removed_by_base_intervention=inherited,
                    cross_mismatches_also_present_in_same_base_local_rounding=local_and_inherited,
                    same_window_base_delta=metrics(direct.last_delta,shadow.last_delta),
                    same_window_base_boundary=metrics(direct.last_boundary_state,shadow.last_boundary_state))
                assert inherited+local_and_inherited==legacy['code_mismatches']
                assert local['unexplained']==0,'Same-base local mismatch not explained by original rounding test'
            rows.append(row)
        print(json.dumps(dict(phase='LINEAGE_PERIOD_COMPLETE',period=period)),flush=True)
    local_max={name:max(x[name]['normalized_RMSE'] for x in rows if name in x) for name in ['same_window_base_output','same_window_base_delta','same_window_base_boundary']}
    assert max(local_max.values())<.005
    result=dict(status='PASS_SAME_BASE_INTERVENTION_WITH_ORIGINAL_TOLERANCE',new_AIME_samples=0,
        local_max_NRMSE=local_max,rows=rows,
        legacy_UNEXPLAINED_labels_retained=True,
        scope='Runtime counterfactual on exactly the257-step weak-decay input fixtures, both periods. Every window shadow uses actual Direct base; original replay trajectory remains unchanged. Not80K or full-model equivalence.')
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
