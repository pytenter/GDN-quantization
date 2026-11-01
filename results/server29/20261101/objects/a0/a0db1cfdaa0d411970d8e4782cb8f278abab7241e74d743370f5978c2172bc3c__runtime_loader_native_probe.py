#!/usr/bin/env python3
"""One pinned engineering load-only observation; never scientific qualification."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'operations/loader_native_probe'
UUID='GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def new(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode()+bytes([10])
    with p.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(p.relative_to(ROOT)),bytes=len(raw),sha256=sha(p))

def identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttick=int(fields[19]),cmdline=[v.decode() for v in (p/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))])

def alive(expected):
    try:return identity(expected['pid'])==expected
    except (OSError,ValueError):return False

def pins():
    assert not any((ROOT/n).exists() for n in ['SOURCE_FREEZE.json','CONTROLLER_STARTED.json','GPU_PERMIT.json'])
    frozen=read(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json')
    permit=read(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_PERMIT.json')
    assert permit['source_freeze_sha256']==sha(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json')
    assert permit['repair_authorization_sha256']==sha(ROOT/'HUMAN_REPAIR_AUTHORIZATION.json')
    assert permit['gpu_uuid']==UUID and permit['model_forward_calls_allowed']==0
    assert permit['counts_as_scientific_qualification'] is False
    for row in frozen['files']:
        p=ROOT/row['path'];assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
    return frozen,permit

def worker():
    frozen,permit=pins()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
    import qualification_guards as guards
    assert not (TARGET/'STARTED.json').exists()
    started=new(TARGET/'STARTED.json',dict(identity=identity(os.getpid()),monotonic=time.monotonic(),hostname=socket.gethostname(),
        probe_source_freeze_sha256=sha(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json'),
        probe_permit_sha256=sha(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_PERMIT.json'),scientific_qualification=False))
    observations=[];forward_calls=0
    def capture(stage):
        record=guards.capture_native_map_evidence(Path('/proc/self/maps').read_text())
        record.update(stage=stage,identity=identity(os.getpid()),monotonic=time.monotonic(),observation_only=True)
        observations.append(new(TARGET/('MAPS_'+stage+'.json'),record))
        assert not record['errors'],record['errors']
    try:
        rows=guards.verify_model_and_external(ROOT,True)
        guards.p1_queue_released(ROOT,1)
        before=guards._own_gpu(ROOT,1)
        driver=guards.driver_identity(ROOT,1,rows)
        capture('PRE_TORCH')
        import torch
        import resident_rotation
        capture('POST_TORCH')
        spec=importlib.util.spec_from_file_location('pinned_qualifier_load_only',ROOT/'scripts/qualify_gpu.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.torch=torch;module.resident=resident_rotation
        def forbid_forward(mod,args):
            nonlocal forward_calls
            forward_calls+=1
            raise RuntimeError('Engineering load-only probe forbids every module forward')
        hook=torch.nn.modules.module.register_module_forward_pre_hook(forbid_forward)
        runtime=module.Runtime(ROOT,TARGET,1,'LEGACY_H')
        nvml=guards._own_gpu(ROOT,1,os.getpid())
        capture('MODEL_LOADED')
        # Observe the already authorized profiler provider once. Only a single
        # scalar allocation/add is executed: no GDN, QDQ, prompt or model call.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            value=torch.zeros(1,device='cuda:0');value.add_(1);torch.cuda.synchronize()
        events=[dict(key=e.key,count=e.count) for e in profile.key_averages()]
        capture('POST_SCALAR_PROFILER')
        assert forward_calls==0
        new(TARGET/'RESULT.json',dict(status='PASS_ENGINEERING_LOAD_ONLY_OBSERVATION_NOT_QUALIFICATION',
            identity=identity(os.getpid()),hostname=socket.gethostname(),monotonic=time.monotonic(),
            environment=runtime.environment,driver_identity=driver,NVML_before=before,NVML_loaded=nvml,
            observations=observations,scalar_profiler_events=events,model_forward_calls=forward_calls,
            QDQ_calls=0,AIME_generations=0,counts_as_scientific_qualification=False,
            native_maps_are_observations_not_admission=True,started=started))
        hook.remove()
    except BaseException as exc:
        new(TARGET/'FAILURE.json',dict(status='ENGINEERING_LOAD_ONLY_PROBE_FAILURE_NO_RETRY',error=repr(exc),
            traceback=traceback.format_exc(),identity=identity(os.getpid()),monotonic=time.monotonic(),
            observations=observations,model_forward_calls=forward_calls,QDQ_calls=0,AIME_generations=0))
        raise

def supervise():
    pins();TARGET.mkdir(exist_ok=True)
    assert not any((TARGET/n).exists() for n in ['DISPATCH.json','STARTED.json','RESULT.json','FAILURE.json'])
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
    import qualification_guards as guards
    observation=guards._own_gpu(ROOT,1)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
    with (TARGET/'worker.log').open('xb') as log:
        lower=time.monotonic()
        process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--worker'],env=env,stdout=log,stderr=log)
        who=identity(process.pid);upper=time.monotonic()
        new(TARGET/'DISPATCH.json',dict(worker_identity=who,supervisor_identity=identity(os.getpid()),hostname=socket.gethostname(),
            dispatch_monotonic_lower=lower,identity_monotonic_upper=upper,NVML_before=observation,
            scientific_qualification=False,probe_source_freeze_sha256=sha(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json'),
            probe_permit_sha256=sha(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_PERMIT.json')))
        rc=process.wait();exited=time.monotonic()
    deadline=exited+120;obs=guards.vendor(ROOT).query_gpu()
    while any(x['pid']==who['pid'] and x['uuid']==UUID for x in obs['apps']) and time.monotonic()<deadline:
        time.sleep(1);obs=guards.vendor(ROOT).query_gpu()
    released=time.monotonic()
    gone=not alive(who);released_ok=not any(x['pid']==who['pid'] and x['uuid']==UUID for x in obs['apps'])
    result=TARGET/'RESULT.json'
    payload=read(result) if result.exists() else None
    okay=rc==0 and gone and released_ok and payload and payload['model_forward_calls']==0 and payload['QDQ_calls']==0 and payload['AIME_generations']==0
    new(ROOT/'operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json',dict(
        status='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED' if okay else 'FAIL_ENGINEERING_LOAD_ONLY_PROBE',
        worker_identity=who,supervisor_identity=identity(os.getpid()),actual_wait_returncode=rc,
        all_exact_own_processes_exited=gone,own_GPU_UUID_released=released_ok,final_NVML=obs,
        model_forward_calls=payload['model_forward_calls'] if payload else None,QDQ_calls=0,AIME_generations=0,
        counts_as_scientific_qualification=False,probe_source_freeze_sha256=sha(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json'),
        probe_permit_sha256=sha(ROOT/'ENGINEERING_LOAD_ONLY_PROBE_PERMIT.json'),
        result=dict(path=str(result.relative_to(ROOT)),bytes=result.stat().st_size,sha256=sha(result)) if result.exists() else None,
        resource_interval=dict(dispatch_monotonic_lower=lower,identity_monotonic_upper=upper,
            exit_first_observed_monotonic=exited,release_first_observed_monotonic=released,monotonic_same_host=True),
        hostname=socket.gethostname(),EXACT_physical_GPUh='UNRESOLVED',cost_scope='ENGINEERING_LOADER_ONLY_SEPARATE_FROM_58_QUALIFICATION_AND_60_FORMAL'))
    assert okay,'Actual engineering probe failed; evidence preserved, no implicit retry'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args();assert args.worker!=args.supervise,'Exactly one explicit mode required'
    worker() if args.worker else supervise()
