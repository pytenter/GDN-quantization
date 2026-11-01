"""Non-AIME full-model smoke and matched, teacher-forced throughput.

All paths use the same BF16 model, native attention and original sampler.
Fixed replay inputs prevent output length/answer differences from changing work.
No AIME input is read by this program.
"""
import argparse,gc,hashlib,json,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
import pace_runtime as R
from lq_unroll_buffer import LQUnrollBuffer
from unroll_runtime import UnrollEngine


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def runtime_identity():
    import ctypes,uuid
    lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0
    d=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(d),0)==0
    raw=(ctypes.c_ubyte*16)();assert lib.cuDeviceGetUuid(ctypes.byref(raw),d)==0
    driver_uuid='GPU-'+str(uuid.UUID(bytes=bytes(raw)))
    assert driver_uuid==os.environ['CUDA_VISIBLE_DEVICES']
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,6)
    return dict(driver_uuid=driver_uuid,CVD=os.environ['CUDA_VISIBLE_DEVICES'],device_name=torch.cuda.get_device_name(0),torch=torch.__version__,CUDA=torch.version.cuda,threads=[torch.get_num_threads(),torch.get_num_interop_threads()],deterministic=torch.are_deterministic_algorithms_enabled(),TF32=torch.backends.cuda.matmul.allow_tf32)


def fixed_run(model,native,masks,f,tokenizer,inputs,kind,period,warmup,measured):
    factory=LQUnrollBuffer if kind=='direct' else None
    adapter=R.PACEAdapter(model,native,masks,period,len(inputs),buffer_factory=factory)
    cache=None;count=0
    gen=torch.Generator(device='cuda:0').manual_seed(1)
    warpers=f.build_warpers();mask=torch.zeros((1,model.get_output_embeddings().weight.shape[0]),dtype=torch.bool,device='cuda:0')
    def forward(token):
        nonlocal cache,count
        out=model(input_ids=torch.tensor([[token]],dtype=torch.long,device='cuda:0'),past_key_values=cache,use_cache=True)
        cache=out.past_key_values;count+=1
        assert bool(torch.isfinite(out.logits).all())
        adapter.assert_cache(cache,count)
        return out
    with torch.inference_mode(),adapter:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();p0=time.perf_counter()
        for token in inputs:out=forward(token)
        torch.cuda.synchronize();prefill=time.perf_counter()-p0
        for i in range(warmup):
            sampled,_=f.sample_next(warpers,out.logits[:,-1,:],mask,gen);mask.scatter_(1,sampled,True)
            out=forward(inputs[i%len(inputs)])
        before={str(k):v.storage_report() for k,v in adapter.buffers.items()}
        torch.cuda.synchronize();t0=time.perf_counter()
        for i in range(measured):
            sampled,_=f.sample_next(warpers,out.logits[:,-1,:],mask,gen);mask.scatter_(1,sampled,True)
            out=forward(inputs[(warmup+i)%len(inputs)])
        torch.cuda.synchronize();seconds=time.perf_counter()-t0
        for b in adapter.buffers.values():b.finalize('preserve')
        audit=adapter.audit()
        if kind=='direct':
            assert all(b.decode_dense_materializations==(warmup+measured)//period for b in adapter.buffers.values())
            assert all(b.validation_dense_materializations==0 for b in adapter.buffers.values())
        result=dict(path=kind,decode_period=period,prompt_tokens=len(inputs),warmup_decode_forwards=warmup,
            measured_decode_forwards=measured,prefill_latency_s=prefill,decode_wall_s=seconds,
            decode_TPS=measured/seconds,mean_decode_token_ms=1000*seconds/measured,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            before_measurement_storage=before,quantizer_audit=audit,
            input_sha256=R.canonical_sha(inputs),fixed_replayed_inputs=True,original_sampler_executed=True,
            free_generation=False,no_AIME=True,compile_and_model_load_excluded=True)
    del out,cache,adapter;gc.collect();torch.cuda.empty_cache()
    return result


def run(a):
    print(json.dumps(dict(phase='LOAD_FROZEN_BF16_MODEL')),flush=True)
    model,tokenizer,native=R._load_frozen_model(ROOT)
    ident=runtime_identity();masks,mask_sha,_=R.load_head_mask(ROOT/'calibration/head_mask.json')
    fixture=json.loads((ROOT/'protocol/non_aime_fixture.json').read_text())['tokens'][:16]
    binding=R._module(ROOT/'vendor/protocol_binding.py','unroll_fixed_binding');f,_=binding.sampler(ROOT,torch)
    result=dict(runtime=ident,head_mask_sha256=mask_sha,mode=a.mode,new_AIME_samples=0)
    if a.mode=='smoke':
        outputs={}
        for period in [8,16]:
            engine=UnrollEngine(ROOT,period,model=model,tokenizer=tokenizer,native_module=native)
            engine.buffer_factory=lambda p,m,**kw:LQUnrollBuffer(p,m,verify_batched_keys=True,**kw)
            outputs[str(period)]=engine.generate(fixture,problem_id='NON_AIME_FROZEN_FIXTURE',max_new_tokens=33)
            assert outputs[str(period)]['status']=='COMPLETE' and outputs[str(period)]['generated_token_count']>0
            for row in outputs[str(period)]['quantizer_audit']['storage_by_layer'].values():
                assert row['validation_dense_materializations']==0 and not row['dense_state_persistent']
                assert row['decode_dense_materializations']==row['decode_tokens']//period
            print(json.dumps(dict(phase='SMOKE_COMPLETE',period=period,generated=outputs[str(period)]['generated_token_count'])),flush=True)
            del engine;gc.collect();torch.cuda.empty_cache()
        result.update(status='PASS_NON_AIME_FULL_MODEL_SMOKE',outputs=outputs)
    else:
        rows=[]
        roster=[('replay',8),('direct',8),('direct',16)]
        for repeat in range(a.repeats):
            order=roster[repeat%3:]+roster[:repeat%3]
            for kind,period in order:
                print(json.dumps(dict(phase='MATCHED_BENCH_START',path=kind,period=period,repeat=repeat)),flush=True)
                row=fixed_run(model,native,masks,f,tokenizer,fixture,kind,period,a.warmup,a.measured)
                row['repeat']=repeat;rows.append(row)
                print(json.dumps(dict(phase='MATCHED_BENCH_DONE',path=kind,period=period,repeat=repeat,TPS=row['decode_TPS'])),flush=True)
        result.update(status='COMPLETE_MATCHED_FIXED_WORKLOAD',measurements=rows,
            scope='Synchronized model decode + unchanged sampler/checks; teacher-forced same non-AIME inputs. Model load, compilation and prefill excluded from decode TPS; prefill separately measured.')
    save(a.output,result)
    print(json.dumps(dict(status=result['status'],output=str(a.output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['smoke','benchmark'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=32);p.add_argument('--measured',type=int,default=128)
    run(p.parse_args())
