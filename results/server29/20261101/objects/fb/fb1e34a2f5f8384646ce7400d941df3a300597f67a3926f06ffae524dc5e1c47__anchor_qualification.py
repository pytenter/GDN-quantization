"""Exactly one engineering 8192 persistent warmup; fresh-process Parent restore."""
import gc,json,math,os,sys,time,traceback
from pathlib import Path
import pilot_engineering as P

def pt_save(name,payload):
    path=P.ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    P.check(not path.exists() and not temporary.exists(),'write-once artifact identity')
    with temporary.open('xb') as f:
        P.torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.rename(temporary,path)
    return {'path':str(path),'sha256':P.sha(path),'bytes':path.stat().st_size}

def memory(cache,token):
    t=P.torch;free,total=t.cuda.mem_get_info()
    return {'consumed':token,'allocated':t.cuda.memory_allocated(),'reserved':t.cuda.memory_reserved(),
            'free':free,'total':total,'cache_bytes':sum(x.numel()*x.element_size() for _,x in P.S.tree_tensors(cache)),
            'process_peak_allocated':t.cuda.max_memory_allocated(),'process_peak_reserved':t.cuda.max_memory_reserved()}

def receipt(started):
    return {'seconds':time.monotonic()-started,'GPUh':(time.monotonic()-started)/3600,
            'peak_allocated':P.torch.cuda.max_memory_allocated(),'peak_reserved':P.torch.cuda.max_memory_reserved(),
            'GPU_UUID':P.S.UUIDS[0],'code_sha256':P.sha(__file__)}

def real_interventions(h,F,model,ids):
    t=P.torch;clean=[];cache=None
    for p,token in enumerate(ids[:36]):
        out,cache=h.forward(token,cache,p,mode='CLEAN_FP_REFERENCE')
        clean.append(t.stack([h.quant.original(F.get_state(cache,l))[0] for l in P.S.LAYERS]).cpu())
    del cache;gc.collect();t.cuda.empty_cache()
    cache=None
    for p,token in enumerate(ids[:32]):out,cache=h.forward(token,cache,p)
    anchor=P.snapshot(cache,model,32);del cache;gc.collect();t.cuda.empty_cache()
    traces={};activated={};previous={};seen=set()
    recurrent=h.qmod.torch_recurrent_gated_delta_rule
    def checked(query,key,value,*args,**kw):
        layer=h.active
        P.check(t.equal(kw['initial_state'],previous[layer]),'real intervention next-token consumes prior quantized branch state')
        seen.add(layer)
        return recurrent(query,key,value,*args,**kw)
    h.qmod.torch_recurrent_gated_delta_rule=checked
    try:
        for arm in ['NO_SCALE_INFLATION','NO_SCALE_DEFLATION']:
            cache=P.restore(anchor,model);trace=[];count=0
            for p in range(32,36):
                previous={l:F.get_state(cache,l).detach().clone() for l in P.S.LAYERS};seen=set()
                c=clean[p].to('cuda:0')
                out,cache=h.forward(ids[p],cache,p,mode=arm,clean=c,capture=True)
                P.check(seen==set(P.S.LAYERS),'real intervention current cache consumed at all24 layers')
                for i,l in enumerate(P.S.LAYERS):
                    source=h.pre[l];adaptive,_,_=h.quant.original(source)
                    effective=t.minimum(adaptive,c[i]) if arm=='NO_SCALE_INFLATION' else t.maximum(adaptive,c[i])
                    count+=int((effective!=adaptive).sum())
                    expected=t.round(source.detach().float()/effective).clamp(-127,127)*effective
                    P.check(t.equal(F.get_state(cache,l),expected),'real full-model current-token min/max original QDQ writeback')
                trace.append(P.digest(P.fingerprint(cache,model,out.logits)))
            traces[arm]=trace;activated[arm]=count
            del cache;gc.collect();t.cuda.empty_cache()
    finally:h.qmod.torch_recurrent_gated_delta_rule=recurrent
    P.save('QUANTIZER_INTERVENTION_VALIDITY.json',{'status':'PASS','engineering_tokens':4,'anchor_consumed':32,
        'same_anchor':True,'real_closed_loop_fullmodel':True,'direct_current_clean_C6R_scales':True,
        'original_AST_codes_round_clamp_dequant':True,'next_token_consumption_all24_layers':True,
        'engineering_activated_group_counts':activated,'trajectory_hashes':traces,'formal_activations':'NOT_RUN'})

def produce():
    for k in [1,2]:P.check(json.loads((P.ROOT/f'qualification/small_{k}.json').read_text())['status']=='PASS','short prerequisite')
    a=json.loads((P.ROOT/'qualification/small_1.json').read_text());b=json.loads((P.ROOT/'qualification/small_2.json').read_text())
    for key in ['oracle_compensated_trace','parent_observer_trace','checkpoint_short','KL_max_abs_FP32_vs_FP64_error','prospective_delta_resolution']:
        P.check(a[key]==b[key],'fresh-process short byte/scalar determinism '+key)
    P.save('qualification/FRESH_SMALL_PROCESS_PARITY.json',{'status':'PASS','processes':2,'compared_fields':
        ['oracle_compensated_trace','parent_observer_trace','checkpoint_short','KL_max_abs_FP32_vs_FP64_error','prospective_delta_resolution']})
    F,core,policy,model,tokenizer,started=P.setup()
    ids=P.N.read(policy['documents'][0]['token_path'])[:8200]
    P.check(len(ids)==8200,'old engineering complete input')
    h=P.Harness(model,F,core,P.torch);h.previous={};cache=None;memory_rows=[]
    with P.torch.inference_mode():
        real_interventions(h,F,model,ids)
        for p in range(8192):
            out,cache=h.forward(ids[p],cache,p)
            if (p+1)%256==0:
                row=memory(cache,p+1);row['seconds']=time.monotonic()-started;memory_rows.append(row)
                print('ANCHOR_PROGRESS',p+1,json.dumps(row,sort_keys=True),flush=True)
        identity=P.fingerprint(cache,model)
        payload=P.snapshot(cache,model,8192)
        P.check(P.S.snapshot_hash(payload['cache'])==identity['cache_tensors'],'8192 complete CPU snapshot bit identity')
        artifact=pt_save('qualification/engineering_anchor_08192.pt',payload)
        h.previous['C6R']={l:F.get_state(cache,l).detach().clone() for l in P.S.LAYERS}
        baseline=[]
        for p in range(8192,8200):
            out,cache=h.forward(ids[p],cache,p,qual=True)
            baseline.append(P.digest(P.fingerprint(cache,model,out.logits,h.codes)))
        memory_rows.append(memory(cache,8200))
        last=memory_rows[-1];first=memory_rows[0]
        slope=(last['cache_bytes']-first['cache_bytes'])/(last['consumed']-first['consumed'])
        extra=math.ceil(slope*(32768-last['consumed']))
        transient=max(r['process_peak_allocated']-r['allocated'] for r in memory_rows)
        available=last['free']+last['reserved']-last['allocated']
        feasible=extra+transient<=available
        record={'status':'PASS' if feasible else 'FAIL_AND_STOP','anchor':artifact,'anchor_fingerprint':identity,
                'uninterrupted_next8':baseline,'memory_measurements':memory_rows,'cache_bytes_per_token':slope,
                'additional_cache_bytes_to32768':extra,'measured_transient_bytes':transient,
                'available_device_bytes_including_reusable_pool':available,
                'projection_role':'closed-loop actual persistent Parent cache; endpoint32768 still not executed',**receipt(started)}
        P.save('qualification/ANCHOR_PRODUCTION_RECEIPT.json',record)
        P.check(feasible,'confirmed32768 cache projection exceeds actual24GB capacity')
    h.close();print('ANCHOR_8192_PRODUCED_AND_CONTINUOUS_PARENT_BASELINE_PASS',flush=True)

def resume():
    r=json.loads((P.ROOT/'qualification/ANCHOR_PRODUCTION_RECEIPT.json').read_text());P.check(r['status']=='PASS','physical production prerequisite')
    F,core,policy,model,tokenizer,started=P.setup();ids=P.N.read(policy['documents'][0]['token_path'])[:8200]
    path=Path(r['anchor']['path']);P.check(P.sha(path)==r['anchor']['sha256'],'anchor serializedSHA exact')
    payload=P.torch.load(path,map_location='cpu',weights_only=False)
    h=P.Harness(model,F,core,P.torch)
    wanted=P.S.snapshot_hash(payload['cache'])
    with P.torch.inference_mode():
        for branch in ['PARENT','NO_SCALE_INFLATION','NO_SCALE_DEFLATION']:
            # All3 complete restoration/isolation checks; interventions were semantically
            # tested on short engineered paired states. No unpaired oracle is used here.
            cache=P.restore(payload,model)
            P.check(P.fingerprint(cache,model)==r['anchor_fingerprint'],'complete8192 restore identity '+branch)
            source_storages={(t.device.type,t.untyped_storage().data_ptr()) for _,t in P.S.tree_tensors(payload['cache'])}
            target_storages={(t.device.type,t.untyped_storage().data_ptr()) for _,t in P.S.tree_tensors(cache)}
            P.check(not source_storages.intersection(target_storages),'isolated complete branch mutable storage '+branch)
            if branch=='PARENT':
                h.previous['C6R']={l:F.get_state(cache,l).detach().clone() for l in P.S.LAYERS};trace=[]
                for p in range(8192,8200):
                    out,cache=h.forward(ids[p],cache,p,qual=True)
                    trace.append(P.digest(P.fingerprint(cache,model,out.logits,h.codes)))
                P.check(trace==r['uninterrupted_next8'],'fresh-process8192 Parent continuous/resume bit parity')
            P.check(P.S.snapshot_hash(payload['cache'])==wanted,'immutable source anchor after branch')
            del cache;gc.collect();P.torch.cuda.empty_cache()
    P.save('ANCHOR_CHECKPOINT_AND_FORK_PARITY.json',{'status':'PASS','engineering_only':True,'anchor_consumed':8192,
        'artifact':r['anchor'],'complete_GDN_KV_conv_metadata_RoPE_RNG':True,'restored_branches':
        ['PARENT','NO_SCALE_INFLATION','NO_SCALE_DEFLATION'],'fresh_process_Parent_next8':'BYTE_EXACT',
        'student_next_token_consumption':'verified against previous actual quantized writeback',
        'no_shared_mutable_storage':True,'formal_document_anchors':'NOT_CREATED_YET',**receipt(started)})
    h.close();print('ANCHOR_FRESH_PROCESS_PARENT_PARITY_PASS',flush=True)

if __name__=='__main__':
    try:
        if sys.argv[1]=='produce':produce()
        elif sys.argv[1]=='resume':resume()
        else:raise ValueError('phase')
    except Exception as exc:
        if not (P.ROOT/'FIRST_FAILURE_GATE.json').exists():
            details={'status':'FAIL_AND_STOP','stage':'STAGE0_ANCHOR_'+sys.argv[1].upper(),'utc':P.N.now(),
                'exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                'formal_outcomes_produced':False,'no_retry':True}
            if getattr(P,'torch',None) is not None and P.torch.cuda.is_initialized():
                details.update(peak_allocated=P.torch.cuda.max_memory_allocated(),peak_reserved=P.torch.cuda.max_memory_reserved())
            P.save('FIRST_FAILURE_GATE.json',details)
        raise
