"""Bounded engineering diagnosis of 26183/FP. No Primary or new science keys."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

sys.dont_write_bytecode = True
import functional_worker as w

D = Path(os.environ['LING_PROSPECTIVE_RESULTS'])
OLD = D.parent/'execution_04'
DOC = 'pg19-test-26183'
UUID = 'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'

def publish(path, value):
    w.publish(path, value)

def tensor_row(torch, x):
    return dict(shape=list(x.shape), dtype=str(x.dtype), stride=list(x.stride()),
                offset=x.storage_offset(), inference=x.is_inference(),
                content_sha256=hashlib.sha256(w.engine.tensor_bytes(torch,x)).hexdigest())

def layouts(torch, cache):
    return {k:tensor_row(torch,x) for k,x in w.engine.cache_tensors(torch,cache)}

def buffers(torch, model):
    return {k:tensor_row(torch,x) for k,x in model.named_buffers()}

def convolution_configs():
    from fla.modules.conv.triton import kernels
    kernel = kernels.causal_conv1d_update_kernel
    rows=[]
    while kernel is not None:
        row=dict(type=type(kernel).__module__+'.'+type(kernel).__qualname__)
        if hasattr(kernel,'cache'):
            row['selected_configs']={str(k):str(v) for k,v in kernel.cache.items()}
        if hasattr(kernel,'best_config'):row['best_config']=str(kernel.best_config)
        rows.append(row);kernel=getattr(kernel,'fn',None)
    return rows

def restore(torch, payload, capture):
    cache=w.transfer(torch,payload['cache'],capture.device)
    assert w.object_fingerprint(torch,cache)==payload['cache_fingerprint']
    torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state(payload['cuda_rng'])
    random.setstate(payload['python_rng']);__import__('numpy').random.set_state(payload['numpy_rng'])
    return cache

def suffix(torch, direction, capture, tokens, cache, observer, name):
    trace=[];h=hashlib.sha256();saved=[]
    before=dict(cache_fingerprint=w.object_fingerprint(torch,cache),
                cache_layout=layouts(torch,cache),model_buffers=buffers(torch,capture.model),
                conv_configs=convolution_configs())
    for position in range(8192,8256):
        assert not (D/'STOP.json').exists()
        cache,logits,_=direction.model_step(torch,capture,cache,tokens[position],position)
        assert bool(torch.isfinite(logits).all()) and logits.shape[-1]==157184
        data=w.engine.tensor_bytes(torch,logits)
        h.update(position.to_bytes(8,'little'));h.update(data)
        trace.append(dict(position=position,logits_sha256=hashlib.sha256(data).hexdigest()))
        saved.append(logits.detach().cpu().contiguous())
        observer.value=None
        del logits,data
    fp=w.object_fingerprint(torch,cache)
    payload_path=D/'diagnostic_payloads'/(name+'.pt')
    torch.save(dict(positions=list(range(8192,8256)),full_vocab_logits=saved),payload_path)
    return dict(name=name,begin=8192,endpoint=8256,observer_enabled=observer.enabled,
                full64_logit_sha256=h.hexdigest(),final_native_cache_sha256=fp,
                trace=trace,before=before,after_model_buffers=buffers(torch,capture.model),
                after_conv_configs=convolution_configs(),
                diagnostic_logits=dict(path=str(payload_path),sha256=w.sha(payload_path),bytes=payload_path.stat().st_size))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['fresh_branches','parent_and_restore'],required=True)
    ap.add_argument('--result',type=Path,required=True);a=ap.parse_args()
    capture=audit=observer=None;started=time.time()
    try:
        assert not a.result.exists() and not (D/'STOP.json').exists()
        assert os.environ['PYTHONHASHSEED']=='0' and os.environ['CUDA_VISIBLE_DEVICES']=='3'
        from prospective_common import verify,load_inputs
        cfg=verify();assert cfg['selected_gpu_uuid']==UUID
        gpu=w.route(cfg);assert gpu==3
        import torch
        runtime=[w.initialize(torch,0)]
        identity=w.proc_identity(os.getpid())
        assert w.device(gpu,cfg)['exclusive']
        base=w.load_module(w.PRIOR/'run_short_horizon_proxy_v2.py','diagnostic_verified_base')
        direction=w.load_module(base.DIRECTION_RUNNER,'diagnostic_verified_direction')
        direction.GPU_UUIDS[gpu]=UUID
        host=direction.gpu_guard(torch,gpu)
        _,_,module,core=direction.load_assets(gpu)
        _,tokens,token_path,_,_=load_inputs(torch,cfg,DOC,['FP'])
        capture=module.Capture();capture.model.eval()
        audit=w.engine.PersistentAudit(torch,direction,core,capture,cfg['layers'],['FP'])
        audit.active=None
        observer=w.HiddenObserver(torch,capture,False)
        runtime.append(w.snapshot(torch,'after_original_helpers',0))
        (D/'diagnostic_payloads').mkdir(exist_ok=True)
        cp=OLD/'runs'/f'{DOC}__FP__8192.pt'
        parent=w.read(OLD/'runs'/f'{DOC}__FP__parent.json')
        assert w.sha(cp)==parent['checkpoint_sha256']
        branches=[];prefix_compare=None
        torch.cuda.reset_peak_memory_stats()
        if a.mode=='fresh_branches':
            payload=torch.load(cp,map_location='cpu',weights_only=False)
            for name,on in [('cold_OFF',False),('warm_OFF',False),('warm_ON',True)]:
                observer.enabled=on
                cache=restore(torch,payload,capture)
                branches.append(suffix(torch,direction,capture,tokens,cache,observer,name))
                del cache;gc.collect()
        else:
            observer.enabled=True;cache=None;prefix=hashlib.sha256()
            for position in range(8192):
                assert not (D/'STOP.json').exists()
                cache,logits,_=direction.model_step(torch,capture,cache,tokens[position],position)
                assert bool(torch.isfinite(logits).all())
                prefix.update(position.to_bytes(8,'little'));prefix.update(w.engine.tensor_bytes(torch,logits))
                if (position+1)%512==0:
                    guard=w.device(gpu,cfg,os.getpid(),binding=identity,require_context=True,
                                   evidence=a.result.with_suffix('.GPU_EXCLUSIVITY.jsonl'))
                    assert guard['exclusive'],guard['failure_conditions']
                    print(json.dumps(dict(event='diagnostic_parent_progress',position=position+1,
                                          elapsed_seconds=time.time()-started)),flush=True)
                observer.value=None;del logits
            spec_sha=w.sha(D/'FROZEN_SPEC.json')
            payload=w.checkpoint(torch,cache,audit,'FP',8191,cfg,spec_sha,w.sha(token_path),None,prefix.hexdigest())
            prefix_compare=dict(new_cache_fingerprint=payload['cache_fingerprint'],
                 old_cache_fingerprint=torch.load(cp,map_location='cpu',weights_only=False)['cache_fingerprint'],
                 new_prefix_full_logit_sha256=prefix.hexdigest(),
                 old_prefix_full_logit_sha256=torch.load(cp,map_location='cpu',weights_only=False)['prefix_full_logit_sha256'])
            torch.save(payload,D/'diagnostic_payloads/new_parent_8192.pt')
            branches.append(suffix(torch,direction,capture,tokens,cache,observer,'uninterrupted_ON'))
            del cache;gc.collect();observer.enabled=False
            cache=restore(torch,payload,capture)
            branches.append(suffix(torch,direction,capture,tokens,cache,observer,'same_process_restore_OFF'))
            del cache
        guard=w.device(gpu,cfg,os.getpid(),binding=identity,require_context=True,
                       evidence=a.result.with_suffix('.GPU_EXCLUSIVITY.jsonl'))
        assert guard['exclusive'],guard['failure_conditions']
        publish(a.result,dict(status='PASS',meaning='DIAGNOSTIC_COLLECTION_COMPLETED_NOT_NUMERICAL_QUALIFICATION',
            mode=a.mode,document=DOC,condition='FP',numerical_qualification=False,scientific_trajectory_count=0,
            underlying_math_unchanged=True,checkpoint_source_sha256=w.sha(cp),branches=branches,
            prefix_comparison=prefix_compare,host=host,runtime=runtime,
            memory=[dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())],
            started_epoch=started,completed_epoch=time.time()))
    except BaseException as e:
        if not (D/'STOP.json').exists():publish(D/'STOP.json',dict(status='STOP',stage='BOUNDED_ENGINEERING_DIAGNOSIS',
             reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        raise
    finally:
        if observer is not None:observer.close()
        if audit is not None:audit.close()
        if capture is not None:capture.close()

if __name__=='__main__':main()
