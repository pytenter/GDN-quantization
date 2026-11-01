"""Primary equivalence: unchanged production chunk vs single-line patched chunk."""
import argparse, sys, traceback
import closure_common as C
sys.path.insert(0,str(C.ROOT/'runtime_patch'))
from patched_gdn import build
from qualify_operator import compare

def run(mode):
    import torch
    C.check(C.read(C.ROOT/'provenance/operator_parity.json')['CUMSUM_NUMERICAL_PARITY_GATE']=='PASS','operator parity required')
    C.configure(reference=mode=='reference')
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    fn=qmod.torch_chunk_gated_delta_rule if mode=='reference' else build(qmod,emit=True)
    outputs={}; rows=[]
    for f in C.read(C.ROOT/'provenance/frozen_cumsum_inputs.json')['fixtures']:
        if f['name']=='long_prefill': continue
        C.check(C.sha(C.ROOT/f['path'])==f['sha256'],'fixture drift')
        payload=C.cuda(C.load_pt(f['path'])['chunk'])
        with torch.inference_mode(): out,state=fn(**payload)
        outputs[f['name']]=dict(output=out.cpu(),state=state.cpu())
        rows.append(dict(name=f['name'],length=payload['query'].shape[1],output=C.record(out),state=C.record(state)))
    artifact=C.pt_save('chunk/'+mode+'.pt',outputs)
    C.save('chunk/'+mode+'.json',dict(rows=rows,artifact=artifact,strict=torch.are_deterministic_algorithms_enabled(),
           REFERENCE_ONLY_NONDETERMINISTIC_ALLOWED=mode=='reference'))

def analyze():
    ref=C.load_pt('chunk/reference.pt'); out=C.load_pt('chunk/strict.pt'); rows=[]
    for n in ref:
        for k in ['output','state']:
            rows.append(dict(fixture=n,object=k,**compare(ref[n][k],out[n][k],'chunk_fp32' if k=='state' else 'chunk_bf16')))
    gate='PASS' if all(r['PASS'] for r in rows) else 'FAIL'
    C.save('provenance/chunk_gdn_parity.json',dict(PATCHED_CHUNK_GDN_PARITY_GATE=gate,rows=rows,
        primary_reference='original production chunk, exact captured real q/k/v/g/beta; only cumsum schedule differs',
        policy_sha256=C.sha(C.ROOT/'protocol/numerical_parity_policy.json')))
    C.check(gate=='PASS','chunk numerical parity failed; STOP')
    print('PATCHED_CHUNK_GDN_PARITY_GATE=PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reference','strict','analyze'],required=True);a=p.parse_args()
    try: analyze() if a.phase=='analyze' else run(a.phase)
    except Exception as e:
        C.save('provenance/chunk_'+a.phase+'.error.json',dict(message=str(e),traceback=traceback.format_exc(),stop=True));raise
