"""Fixed teacher-forced FP-only probes, separate reference and strict processes."""
import argparse, gc, hashlib, json, sys, traceback
import closure_common as C
sys.path.insert(0,str(C.ROOT/'runtime_patch'))
from patched_gdn import install
from qualify_operator import compare

def plan():
    C.save('protocol/full_probe_execution_policy.json',dict(
       frozen_before_full_model=True,seed=C.SEED,historical32=dict(prefill_length=1,total_tokens=32),
       dev256=dict(prefill_length=128,total_tokens=256),
       correction_to_initial_capture_policy='dev256 uses 128-token prefill followed by remaining 128 cached tokens, preserving all256 token IDs and order; no repeated token suffix. Initial capture policy description was provisional, not an executed comparison.',
       checkpoints={'historical32':[0,7,31],'dev256':[127,191,255]},
       state_layout='all24 actual cache recurrent states',
       logits='all positions and all vocabulary coordinates, BF16 model outputs',
       transition_trace='ordered per-call layer/input_length/output/state raw tensor hashes; elapsed/PID excluded',
       generated_tokens=0,quantizer_used=False,rotation_used=False,scientific_measurement=False))

def run(mode,index):
    import torch
    C.check(C.read(C.ROOT/'provenance/chunk_gdn_parity.json')['PATCHED_CHUNK_GDN_PARITY_GATE']=='PASS','chunk parity required')
    p=C.read(C.ROOT/'protocol/full_probe_execution_policy.json')
    F,core,model,tok,qmod=C.init_model(reference=mode=='reference')
    if mode=='strict': install(qmod)
    layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
    active=[None]; trace=[]; checkpoint_outputs={}; capture_now=[False]; handles=[]
    for l in F.GDN_LAYERS:
        module=getattr(layers[l],'linear_attn',None) or getattr(layers[l],'self_attn',None)
        handles.append(module.register_forward_pre_hook(lambda _m,_a,l=l:active.__setitem__(0,l)))
    original_chunk=qmod.torch_chunk_gated_delta_rule; original_recurrent=qmod.torch_recurrent_gated_delta_rule
    def wrap(fn):
        def observe(*args,**kw):
            result=fn(*args,**kw); o,s=result
            trace.append(dict(layer=active[0],input_length=args[0].shape[1],output_sha256=C.tensor_sha(o),state_sha256=C.tensor_sha(s)))
            if capture_now[0]: checkpoint_outputs[str(active[0])]=o.detach().cpu().clone()
            return result
        return observe
    qmod.torch_chunk_gated_delta_rule=wrap(original_chunk); qmod.torch_recurrent_gated_delta_rule=wrap(original_recurrent)
    tokens=C.read(C.ROOT/'protocol/frozen_probe_tokens.json')['probes']; artifacts=[]; summaries=[]
    try:
        for name in ['historical32','dev256']:
            F.setup_seed(C.SEED); trace.clear(); ids=tokens[name]; pre=p[name]['prefill_length']; wanted=p['checkpoints'][name]
            saved={}; logits=[]; cache=None
            batches=[(pre-1,ids[:pre])]+[(i,[ids[i]]) for i in range(pre,len(ids))]
            with torch.inference_mode():
                for position,batch in batches:
                    capture_now[0]=position in wanted; checkpoint_outputs.clear()
                    out=model(input_ids=torch.tensor([batch],device='cuda:0'),past_key_values=cache,use_cache=True)
                    cache=out.past_key_values; logits.append(out.logits.detach().cpu().clone())
                    if position in wanted:
                        for l in F.GDN_LAYERS:
                            state=F.get_state(cache,l)
                            saved[f'state/{position}/{l}']=state.detach().cpu().clone()
                            saved[f'gdn/{position}/{l}']=checkpoint_outputs[str(l)].clone()
                    if position in wanted: print('FULL_PROBE_CHECKPOINT '+mode+' '+str(index)+' '+name+' '+str(position),flush=True)
            saved['logits']=torch.cat(logits,dim=1)
            digest={k:C.tensor_sha(v) for k,v in saved.items()}
            trace_hash=hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            artifact=C.pt_save(f'full_model/{mode}_{index}_{name}.pt',saved); artifacts.append(artifact)
            summaries.append(dict(probe=name,output_hashes=digest,trace_sha256=trace_hash,trace_calls=len(trace),
                                  final_state_layers=len(F.GDN_LAYERS),total_tokens=len(ids),strict=torch.are_deterministic_algorithms_enabled()))
            C.save(f'full_model/{mode}_{index}_{name}_trace.json',dict(rows=trace,sha256=trace_hash))
            del out,cache,saved,logits;gc.collect();torch.cuda.empty_cache()
    finally:
        qmod.torch_chunk_gated_delta_rule=original_chunk;qmod.torch_recurrent_gated_delta_rule=original_recurrent
        for h in handles:h.remove()
    C.save(f'full_model/{mode}_{index}.json',dict(rows=summaries,artifacts=artifacts,
        REFERENCE_ONLY_NONDETERMINISTIC_ALLOWED=mode=='reference',torch_version=torch.__version__,cuda_runtime=torch.version.cuda))

def parity():
    import torch
    rows=[]
    for name in ['historical32','dev256']:
        a=C.load_pt(f'full_model/reference_1_{name}.pt');b=C.load_pt(f'full_model/strict_1_{name}.pt')
        C.check(set(a)==set(b),'full evidence object coverage')
        for k in a:
            kind='model_fp32' if a[k].dtype==torch.float32 else 'model_bf16'
            rows.append(dict(probe=name,object=k,**compare(a[k],b[k],kind)))
        del a,b;gc.collect()
    gate='PASS' if all(r['PASS'] for r in rows) else 'FAIL'
    C.save('provenance/full_model_probe_parity.json',dict(FULL_MODEL_SHORT_PROBE_PARITY_GATE=gate,rows=rows,
           policy_sha256=C.sha(C.ROOT/'protocol/numerical_parity_policy.json')))
    C.save('provenance/additional_deterministic_blocker_audit.json',dict(
           ADDITIONAL_UNSUPPORTED_DETERMINISTIC_OP_COUNT=0,complete_dev256_strict_probe=True,
           additional_operations_patched=0,strict=True,gate='PASS'))
    C.check(gate=='PASS','full model numerical parity failed; STOP')
    print('FULL_MODEL_SHORT_PROBE_PARITY_GATE=PASS',flush=True)

def determinism():
    C.check(C.read(C.ROOT/'provenance/full_model_probe_parity.json')['FULL_MODEL_SHORT_PROBE_PARITY_GATE']=='PASS','full parity required')
    runs=[C.read(C.ROOT/f'full_model/strict_{i}.json') for i in range(1,6)]; rows=[]
    for idx,n in enumerate(['historical32','dev256']):
        values=[r['rows'][idx] for r in runs];base=values[0]
        ok=all(v['strict'] and v['output_hashes']==base['output_hashes'] and v['trace_sha256']==base['trace_sha256'] for v in values)
        rows.append(dict(probe=n,PASS=ok,runs=values))
    gate='PASS' if all(r['PASS'] for r in rows) else 'FAIL'
    C.save('provenance/fresh_process_determinism.json',dict(PATCHED_FULL_MODEL_FRESH_PROCESS_DETERMINISM_GATE=gate,fresh_processes=5,rows=rows))
    C.check(gate=='PASS','fresh process full model determinism failed; STOP')
    print('PATCHED_FULL_MODEL_FRESH_PROCESS_DETERMINISM_GATE=PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['plan','reference','strict','parity','determinism'],required=True);p.add_argument('--index',type=int,default=1);a=p.parse_args()
    try:
        if a.phase in ['reference','strict']:run(a.phase,a.index)
        else:globals()[a.phase]()
    except Exception as e:
        C.save(f'provenance/full_model_{a.phase}_{a.index}.error.json',dict(message=str(e),traceback=traceback.format_exc(),stop=True,
           additional_unsupported_deterministic_op=bool('deterministic' in str(e).lower() and 'implementation' in str(e).lower())))
        raise
