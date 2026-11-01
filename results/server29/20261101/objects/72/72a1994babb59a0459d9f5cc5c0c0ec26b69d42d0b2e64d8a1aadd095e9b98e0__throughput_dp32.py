"""Same original fixed_run for short comparison; common fixed-stream long run."""
import gc,json,math,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import pace_runtime as R
import model_engineering as M
from lq_unroll_buffer import LQUnrollBuffer,DeltaRecord
from pace_buffer import native_l2norm
import pace_quantizer as Q
from packed_direct_read import packed_readout

def save(name,x):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def emit(phase,**kw):print(json.dumps(dict(phase=phase,monotonic=time.monotonic(),**kw)),flush=True)

def micro(mask):
    """Separate diagnostic instrumentation; never enabled in model benchmark."""
    rows=[]
    def inputs():
        return (torch.randn((1,16,128),device='cuda',dtype=torch.bfloat16)*.1,
                torch.randn((1,16,128),device='cuda',dtype=torch.bfloat16)*.1,
                torch.randn((1,32,128),device='cuda',dtype=torch.bfloat16)*.1,
                torch.full((1,32),.3,device='cuda',dtype=torch.bfloat16),
                torch.full((1,32),-.01,device='cuda',dtype=torch.float32))
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261028)
        for period in [8,16,32]:
            readouts=[]
            for pending in sorted(set([0,period//2,period-2,period-1])):
                b=LQUnrollBuffer(period,mask);b.set_prompt_length(0)
                for _ in range(max(1,pending)):b.step(*inputs())
                if pending==0:
                    # Obtain a normal zero-window base without retaining records.
                    for _ in range(period-1):b.step(*inputs())
                assert b.pending_tokens==pending
                q,k,v,beta,g=inputs();enc=Q.PackedAffineGroups.from_tensor(k.reshape(1,16,4,32),4)
                records=b._slots[:pending];kk=b._batched_keys(records+[DeltaRecord(enc,None,g)])
                pk=kk[:-1].contiguous();key=kk[-1].contiguous()
                pu=torch.stack([x.u for x in records]).contiguous() if records else torch.empty((0,1,32,128),device='cuda')
                alphas=torch.stack([x.log_decay for x in records]+[g]).exp().contiguous()
                query=native_l2norm(q.repeat_interleave(2,1)).float()/math.sqrt(128)
                def call():return packed_readout(b.packed_state,pk,pu,alphas,query,key,v,beta,b._head_maps)
                for _ in range(10):call()
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=call()
                events=[];host=[]
                for _ in range(10):
                    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(50):graph.replay()
                    e.record();e.synchronize();events.append(s.elapsed_time(e)*1000/50)
                    torch.cuda.synchronize();t=time.perf_counter();last=call();host.append((time.perf_counter()-t)*1e6);torch.cuda.synchronize()
                readouts.append(dict(pending_records=pending,query_token_is_boundary=pending==period-1,
                    fused_two_kernel_CUDA_graph_us=events,median_fused_us=statistics.median(events),
                    ordinary_host_dispatch_us=host,median_host_dispatch_us=statistics.median(host),
                    fused_kernel_launches=2,full_preparation_kernel_launches='NOT_PROFILED',
                    component_split='Base matvec, delta and pending readout fused; separate exclusive latencies UNMEASURED without changing kernel'))
                del b,graph,captured,last,pk,pu,alphas,kk,key,query;gc.collect();torch.cuda.empty_cache()
            boundary=[]
            for repeat in range(10):
                b=LQUnrollBuffer(period,mask);b.set_prompt_length(0)
                for _ in range(period-1):b.step(*inputs())
                components=[];pack_events=[];in_state_encode=[False]
                orig_delta=b._delta_state;orig_encode=Q.PackedAffineState.from_tensor;orig_pack=Q.pack_nibbles
                peak={}
                def instrument(name,fn,args,kwargs):
                    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    s.record();t=time.perf_counter();out=fn(*args,**kwargs);wall=time.perf_counter()-t;e.record()
                    components.append((name,s,e,wall));return out
                def delta(*args,**kwargs):
                    tensors=b.resident_tensors();peak.update(inventory=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size()) for n,t in tensors.items()],pending=b.pending_tokens)
                    return instrument('state_reconstruction',orig_delta,args,kwargs)
                def encode(*args,**kwargs):
                    in_state_encode[0]=True
                    try:return instrument('state_encode_total_including_packing',orig_encode,args,kwargs)
                    finally:in_state_encode[0]=False
                def pack(*args,**kwargs):
                    if not in_state_encode[0]:return orig_pack(*args,**kwargs)
                    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    s.record();out=orig_pack(*args,**kwargs);e.record();pack_events.append((s,e));return out
                q=inputs();b._delta_state=delta;Q.PackedAffineState.from_tensor=encode;Q.pack_nibbles=pack
                try:
                    torch.cuda.synchronize();t=time.perf_counter();b.step(*q);torch.cuda.synchronize();whole=(time.perf_counter()-t)*1e6
                finally:
                    b._delta_state=orig_delta;Q.PackedAffineState.from_tensor=orig_encode;Q.pack_nibbles=orig_pack
                assert b.pending_tokens==0 and b.decode_dense_materializations==1
                c={n:dict(event_stream_elapsed_us=s.elapsed_time(e)*1000,host_submit_wall_us=w*1e6) for n,s,e,w in components}
                packing=sum(s.elapsed_time(e)*1000 for s,e in pack_events)
                boundary.append(dict(repeat=repeat,whole_boundary_host_synchronized_us=whole,components=c,packing_event_stream_us=packing,
                    encode_remainder_event_us=c['state_encode_total_including_packing']['event_stream_elapsed_us']-packing,
                    encode_remainder_scope='INFERRED residual includes metadata/scales/zp/exponent/RNE/layout/checks/dispatch; not isolated SM execution',
                    actual_precommit_inventory=peak,post_storage=b.storage_report()))
                del b;gc.collect();torch.cuda.empty_cache()
            rows.append(dict(period=period,readouts=readouts,boundaries=boundary))
            emit('MICRO_PERIOD_DONE',period=period)
    return dict(status='COMPLETE_COMMON_DIAGNOSTIC_PROFILING',rows=rows,actual_SM_occupancy='UNMEASURED',
        timing_scope='Separate identical instrumentation for all periods; excluded from primary full-model measurements. Event stream elapsed can include host dispatch gaps. Fused component sub-times unavailable without rewriting kernel.',new_AIME_samples=0)

def long_run(model,native,masks,f,tokenizer,inputs,period):
    adapter=R.PACEAdapter(model,native,masks,period,len(inputs),buffer_factory=LQUnrollBuffer)
    cache=None;count=0;gen=torch.Generator(device='cuda:0').manual_seed(1)
    warpers=f.build_warpers();mask=torch.zeros((1,model.get_output_embeddings().weight.shape[0]),dtype=torch.bool,device='cuda:0')
    def forward(token):
        nonlocal cache,count
        out=model(input_ids=torch.tensor([[token]],dtype=torch.long,device='cuda:0'),past_key_values=cache,use_cache=True)
        cache=out.past_key_values;count+=1;assert bool(torch.isfinite(out.logits).all());adapter.assert_cache(cache,count);return out
    marks={};host_iterations=[]
    with torch.inference_mode(),adapter:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        for token in inputs:out=forward(token)
        torch.cuda.synchronize();prefill=time.perf_counter()-t
        for i in range(16384):
            t=time.perf_counter();sampled,_=f.sample_next(warpers,out.logits[:,-1,:],mask,gen);mask.scatter_(1,sampled,True)
            out=forward(inputs[i%len(inputs)]);host_iterations.append((time.perf_counter()-t)*1000)
            n=i+1
            if n in [32,2048,8192,16384]:
                torch.cuda.synchronize();marks[n]=time.perf_counter()
            if n%128==0:emit('LONG_FIXED_STREAM',period=period,decode_forwards=n)
        for b in adapter.buffers.values():b.finalize('preserve')
        audit=adapter.audit()
        assert all(x['decode_dense_materializations']==16384//period and x['validation_dense_materializations']==0 and not x['dense_state_persistent'] for x in audit['storage_by_layer'].values())
        intervals=[dict(start_decode_forward=lo,end_decode_forward=hi,forwards=hi-lo,wall_s=marks[hi]-marks[lo],TPS=(hi-lo)/(marks[hi]-marks[lo])) for lo,hi in [(32,16384),(2048,8192),(8192,16384)]]
        vals=sorted(host_iterations[32:]);row=dict(path='direct',decode_period=period,scope='LONG_TEACHER_FORCED_IDENTICAL_STREAM',prompt_tokens=len(inputs),warmup_decode_forwards=32,measured_decode_forwards=16384-32,
            decode_wall_s=marks[16384]-marks[32],decode_TPS=(16384-32)/(marks[16384]-marks[32]),mean_decode_token_ms=1000*(marks[16384]-marks[32])/(16384-32),
            prefill_latency_s=prefill,intervals=intervals,host_iteration_ms=dict(P50=statistics.median(vals),P95=vals[math.ceil(.95*len(vals))-1],scope='Host iteration with original checks, no added per-token CUDA synchronization; not isolated CUDA kernel latency'),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),quantizer_audit=audit,input_sha256=R.canonical_sha(inputs),input_stream='Cyclic original16-token non-AIME fixture from decode index0; identical to original warmup+measured stream',fixed_replayed_inputs=True,original_sampler_executed=True,free_generation=False,no_AIME=True)
    del out,cache,adapter;gc.collect();torch.cuda.empty_cache();return row

def main():
    from compile_telemetry import install
    install(ROOT/'COMPILER_TELEMETRY.json')
    import execution_common as C
    before=C.model_pins(ROOT,full=True)
    save('MODEL17_FRESH_SHA.json',dict(files=before))
    emit('LOAD_FROZEN_MODEL');t=time.perf_counter();model,tokenizer,native=R._load_frozen_model(ROOT)
    load=time.perf_counter()-t;identity=M.runtime_identity();masks,masksha,_=R.load_head_mask(ROOT/'calibration/head_mask.json')
    save('LOADED_REGULAR_MAPPED_SHA.json',dict(files=C.regular_maps(),runtime=identity))
    fixture=json.loads((ROOT/'protocol/non_aime_fixture.json').read_text())['tokens'][:16]
    binding=R._module(ROOT/'vendor/protocol_binding.py','dp32_benchmark_binding');f,_=binding.sampler(ROOT,torch)
    t=time.perf_counter();save('KERNEL_RAW.json',micro(masks[next(iter(masks))]));profile_wall=time.perf_counter()-t
    # Each period touches every specialization before primary timing.
    warm=[]
    for period in [8,16,32]:
        t=time.perf_counter();row=M.fixed_run(model,native,masks,f,tokenizer,fixture,'direct',period,32,32)
        warm.append(dict(period=period,first_touch_wall_s=time.perf_counter()-t,measurement=row))
        emit('FIRST_TOUCH_DONE',period=period)
    save('FIRST_TOUCH_AND_COMPILE.json',dict(model_load_wall_s=load,diagnostic_profile_wall_s=profile_wall,first_touches=warm,scope='Cold import/compile/profile and first-touch wall recorded separately; not included in primary decode TPS'))
    rows=[]
    for mode in ['short','long']:
        for repeat,order in enumerate([[8,16,32],[16,32,8],[32,8,16]]):
            for position,period in enumerate(order):
                emit('BENCHMARK_START',mode=mode,repeat=repeat,period=period,position=position)
                start=time.monotonic()
                row=M.fixed_run(model,native,masks,f,tokenizer,fixture,'direct',period,32,128) if mode=='short' else long_run(model,native,masks,f,tokenizer,fixture,period)
                row.update(mode=mode,repeat=repeat,position=position,request_start_monotonic=start,request_end_monotonic=time.monotonic())
                save('measurements/'+mode+'_r'+str(repeat)+'_dp'+str(period)+'.json',row);rows.append(row)
                emit('BENCHMARK_DONE',mode=mode,repeat=repeat,period=period,TPS=row['decode_TPS'])
    save('BENCHMARK_RESULTS.json',dict(status='COMPLETE_FIXED_WORKLOAD_ALL_THREE_PERIODS',runtime=identity,head_mask_sha256=masksha,model_load_wall_s=load,measurements=rows,new_AIME_samples=0,new_accuracy_evaluations=0,same_kernel_all_periods=True))
    assert C.model_pins(ROOT)==[dict(x,fresh_full_SHA=False) for x in before]
    save('POSTFLIGHT_REGULAR_MAPPED_SHA.json',dict(files=C.regular_maps()))
    emit('ALL_BENCHMARKS_DONE');return 0
if __name__=='__main__':raise SystemExit(main())
