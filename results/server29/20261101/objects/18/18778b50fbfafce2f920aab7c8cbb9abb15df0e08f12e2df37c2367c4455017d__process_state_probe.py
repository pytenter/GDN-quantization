"""One bounded, same-key cross-process probe after the initial two diagnostics."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import time
import traceback

D=Path(os.environ['LING_PROSPECTIVE_RESULTS'])
W=Path(__file__).resolve().parent
UUID='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'

def autotuners():
    rows={}
    for name,module in sorted(sys.modules.items()):
        if not name.startswith('fla.') or module is None:continue
        for key,value in sorted(vars(module).items()):
            typename=type(value).__module__+'.'+type(value).__qualname__
            if not (typename.startswith('triton.') or 'CachedAutotuner' in typename):continue
            seen=set();chain=[];v=value
            while v is not None and id(v) not in seen:
                seen.add(id(v));r={'type':type(v).__module__+'.'+type(v).__qualname__}
                cache=getattr(v,'cache',None)
                if isinstance(cache,dict):r['selected']={str(k):str(x) for k,x in cache.items()}
                if hasattr(v,'best_config'):r['best_config']=str(v.best_config)
                if hasattr(v,'configs'):r['configs']=[str(x) for x in v.configs]
                chain.append(r);v=getattr(v,'fn',None)
            if any('selected' in x or 'configs' in x for x in chain):rows[name+'.'+key]=chain
    return rows

def worker():
    import functional_worker as w
    import resume_diagnostic as diag
    from prospective_common import verify,load_inputs
    cfg=verify();gpu=w.route(cfg);assert gpu==3
    plan=w.read(D/'CROSS_PROCESS_PLAN.json')
    assert os.environ['FLA_CACHE_MODE']==('disabled' if plan['variant']=='cross_process_auto' else 'default')
    assert w.sha(Path(os.environ['FLA_CONFIG_DIR'])/'causal_conv1d_update_kernel.json')=='0a3b2436f8c4be23d1aaf425199a0ce7be9d4742ab9c5085f36b9ebbf34eb8cb'
    import torch
    runtime=[w.initialize(torch,0)];capture=audit=observer=None;started=time.time()
    try:
        identity=w.proc_identity(os.getpid());assert w.device(gpu,cfg)['exclusive']
        base=w.load_module(w.PRIOR/'run_short_horizon_proxy_v2.py','process_probe_base')
        direction=w.load_module(base.DIRECTION_RUNNER,'process_probe_direction');direction.GPU_UUIDS[gpu]=UUID
        host=direction.gpu_guard(torch,gpu);_,_,module,core=direction.load_assets(gpu)
        _,tokens,_,_,_=load_inputs(torch,cfg,'pg19-test-26183',['FP'])
        capture=module.Capture();capture.model.eval();audit=w.engine.PersistentAudit(torch,direction,core,capture,cfg['layers'],['FP'])
        audit.active=None;observer=w.HiddenObserver(torch,capture,False)
        runtime.append(w.snapshot(torch,'after_original_helpers',0));torch.cuda.reset_peak_memory_stats()
        branches=[]
        for name,cp in [('newCP_cold_OFF',plan['new_checkpoint']),('oldCP_warm_OFF',plan['old_checkpoint']),('newCP_warm_OFF',plan['new_checkpoint'])]:
            assert w.sha(cp['path'])==cp['sha256']
            z=torch.load(cp['path'],map_location='cpu',weights_only=False)
            cache=diag.restore(torch,z,capture);before=autotuners()
            b=diag.suffix(torch,direction,capture,tokens,cache,observer,name)
            b.update(checkpoint_sha256=cp['sha256'],all_autotuners_before=before,all_autotuners_after=autotuners())
            branches.append(b);del z,cache
        guard=w.device(gpu,cfg,os.getpid(),binding=identity,require_context=True,evidence=D/'process_state_probe.GPU_EXCLUSIVITY.jsonl')
        assert guard['exclusive'],guard['failure_conditions']
        source=Path(inspect.getfile(type(capture.model)))
        runtime_state={'model_class':type(capture.model).__module__+'.'+type(capture.model).__qualname__,
           'actual_model_class_source':str(source),'actual_model_class_source_sha256':w.sha(source),
           'matmul_precision':torch.get_float32_matmul_precision(),
           'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
           'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
           'model_attention_implementation':capture.model.config._attn_implementation}
        import fla.ops.utils.cache as cache_module
        runtime_state['FLA_CACHE_MODE_observed']=cache_module.FLA_CACHE_MODE.value
        runtime_state['FLA_CONFIG_DIR_observed']=str(cache_module.get_fla_config_dir())
        runtime_state['environment']={k:os.environ.get(k) for k in ['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG','FLA_CACHE_MODE','FLA_CONFIG_DIR','CUDA_VISIBLE_DEVICES']}
        w.publish(D/'process_state_probe.json',dict(status='PASS',meaning='DIAGNOSTIC_COLLECTION_ONLY',
           numerical_qualification=False,scientific_trajectory_count=0,branches=branches,
           runtime_state=runtime_state,runtime=runtime,host=host,
           memory=[dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())],
           started_epoch=started,completed_epoch=time.time()))
    finally:
        if observer is not None:observer.close()
        if audit is not None:audit.close()
        if capture is not None:capture.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');a=ap.parse_args()
    if a.worker:return worker()
    from prospective_common import verify
    from gpu_admission import admit
    from recovery_state import atomic_json,lock
    from execution04_launcher import worker_environment
    held=lock(D/'controller.lock')
    try:
        cfg=verify();plan=json.loads((D/'CROSS_PROCESS_PLAN.json').read_text())
        assert not (D/'process_state_probe.INTENT.json').exists(),'NO_RETRY'
        admit(cfg,D/'admission/process_state_probe.window.jsonl')
        base=D/'process_state_probe'
        ident=dict(document='pg19-test-26183',rotation='FP',stage='engineering_process_diagnostic',target_gpu_uuid=UUID,execution=str(D))
        command=[sys.executable,'-B','-u',str(__file__),'--worker']
        intent=base.with_suffix('.INTENT.json')
        atomic_json(intent,dict(base=str(base),command=command,unit_identity=ident,source_commit=plan['source_commit'],tag='process_state_probe',stage='engineering_process_diagnostic'))
        child=subprocess.Popen([sys.executable,'-B','-u',str(W/'unit_runner.py'),str(intent)],env=worker_environment(os.environ,0,UUID,3))
        try:rc=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            atomic_json(D/'STOP.json',dict(status='STOP',stage='PROCESS_STATE_PROBE_TIMEOUT',epoch=time.time(),no_retry=True))
            child.wait(timeout=120)
            raise
        assert rc==0,('DIAGNOSTIC_FAILED',rc)
    except BaseException as e:
        if not (D/'STOP.json').exists():atomic_json(D/'STOP.json',dict(status='STOP',stage='PROCESS_STATE_PROBE',reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        raise
    finally:held.close()

if __name__=='__main__':main()
