#!/usr/bin/env python3
"""Fixed tiny baseline/profiler native-exit diagnosis; no model or scientific tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'operations/cupti_exit_diagnostic_v1'
UUID='GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(v,sort_keys=True,indent=2,allow_nan=False).encode()+bytes([10])
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
def who(pid):
    p=Path('/proc')/str(pid);x=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttick=int(x[19]),cmdline=[s.decode()for s in (p/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))])
def same(v):
    try:return who(v['pid'])==v
    except OSError:return False
def verify():
    p=read(OUT/'PERMIT.json');assert p['gpu_uuid']==UUID and p['model_loads_allowed']==0
    assert p['human_authorization_sha256']==sha(ROOT/'HUMAN_NATIVE_EXIT_REPAIR_AUTHORIZATION.json')
    assert p['script_sha256']==sha(__file__)
    assert p['prior_failed_completion_sha256']==sha(ROOT/'operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json')
    assert read(ROOT/'operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json')['actual_wait_returncode']==-11
    assert not (ROOT/'SOURCE_FREEZE.json').exists()and not(ROOT/'CONTROLLER_STARTED.json').exists()

def worker(case):
    verify();assert os.environ['CUDA_VISIBLE_DEVICES']==UUID
    new(OUT/(case+'_STARTED.json'),dict(identity=who(os.getpid()),monotonic=time.monotonic(),case=case,
        profiler_environment={k:os.environ.get(k)for k in ['TEARDOWN_CUPTI','DISABLE_CUPTI_LAZY_REINIT','CUDA_MODULE_LOADING','LD_PRELOAD']},
        model_loads=0,model_forward_calls=0,QDQ_calls=0,AIME_generations=0))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.__version__=='2.5.1+cu121'and torch.cuda.device_count()==1
    sys.path.insert(0,str(ROOT/'src'));import qualification_guards as g
    operations=[]
    if case=='profile':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as p:
            x=torch.zeros(1,device='cuda:0');x.add_(1);torch.cuda.synchronize()
        operations=[dict(key=e.key,count=e.count)for e in p.key_averages()]
    else:
        x=torch.zeros(1,device='cuda:0');x.add_(1);torch.cuda.synchronize()
    own=g._own_gpu(ROOT,1,os.getpid())
    new(OUT/(case+'_BODY.json'),dict(status='BODY_COMPLETE_NOT_EXIT_PROOF',case=case,identity=who(os.getpid()),monotonic=time.monotonic(),
        operations=operations,NVML=own,mapped_library_observation=g.capture_native_map_evidence(Path('/proc/self/maps').read_text()),
        model_loads=0,model_forward_calls=0,QDQ_calls=0,AIME_generations=0))

GDB=r'''set pagination off
set confirm off
set print thread-events off
handle SIG32 nostop noprint pass
handle SIGSEGV stop print pass
python
import gdb,json
def stopped(event):
    if isinstance(event,gdb.SignalEvent):
        print('NATIVE_DIAGNOSTIC_SIGNAL '+json.dumps({'signal':event.stop_signal,'inferior_pid':gdb.selected_inferior().pid}))
def exited(event):
    print('NATIVE_DIAGNOSTIC_EXIT '+json.dumps({'exit_code':getattr(event,'exit_code',None)}))
gdb.events.stop.connect(stopped)
gdb.events.exited.connect(exited)
gdb.execute('run')
if gdb.selected_inferior().pid:
    gdb.execute('thread apply all bt 20')
    gdb.execute('info sharedlibrary')
    gdb.execute('continue')
end
'''

def supervise():
    verify();sys.path.insert(0,str(ROOT/'src'));import qualification_guards as g
    g.verify_native_closure(ROOT,fresh=True)
    new(OUT/'SUPERVISOR_STARTED.json',dict(identity=who(os.getpid()),monotonic=time.monotonic(),native_manifest_sha256=sha(ROOT/g.NATIVE_CLOSURE)))
    assert not(OUT/'gdb_commands.gdb').exists()
    (OUT/'gdb_commands.gdb').write_text(GDB)
    results=[]
    for case in ['baseline','profile']:
        before=g._own_gpu(ROOT,1);env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--case',case]
        if case=='profile':command=['gdb','-q','-batch','-x',str(OUT/'gdb_commands.gdb'),'--args']+command
        with (OUT/(case+'.log')).open('xb')as log:
            lower=time.monotonic();p=subprocess.Popen(command,env=env,stdout=log,stderr=log);parent_identity=who(p.pid);upper=time.monotonic();rc=p.wait();exit_time=time.monotonic()
        started=read(OUT/(case+'_STARTED.json'))if(OUT/(case+'_STARTED.json')).exists()else None
        child=started['identity']if started else parent_identity
        deadline=time.monotonic()+120
        while True:
            obs=g.vendor(ROOT).query_gpu()
            own=[x for x in obs['apps']if x['pid']==child['pid']and x['uuid']==UUID]
            if not own or time.monotonic()>=deadline:break
            time.sleep(1)
        text=(OUT/(case+'.log')).read_text(errors='replace')
        signals=[json.loads(x.split('NATIVE_DIAGNOSTIC_SIGNAL ',1)[1])for x in text.splitlines()if 'NATIVE_DIAGNOSTIC_SIGNAL 'in x]
        exits=[json.loads(x.split('NATIVE_DIAGNOSTIC_EXIT ',1)[1])for x in text.splitlines()if 'NATIVE_DIAGNOSTIC_EXIT 'in x]
        entry=dict(case=case,actual_wrapper_wait_returncode=rc,wrapper_identity=parent_identity,inferior_identity=child,
            native_signal_events=signals,native_exit_events=exits,own_exact_process_exited=not same(child),own_GPU_UUID_released=not own,
            resource_interval=dict(dispatch_monotonic_lower=lower,identity_monotonic_upper=upper,exit_first_observed_monotonic=exit_time,
                release_first_observed_monotonic=time.monotonic(),monotonic_same_host=True),NVML_before=before,NVML_after=obs,
            gdb_is_not_inferior_exit_proof=case=='profile',model_loads=0,model_forward_calls=0,QDQ_calls=0,AIME_generations=0)
        new(OUT/(case+'_COMPLETION.json'),entry);results.append(entry)
        assert entry['own_exact_process_exited']and entry['own_GPU_UUID_released'],'Own diagnostic not released; never touch foreign process'
    new(OUT/'RESULT.json',dict(status='ENGINEERING_DIAGNOSIS_FINISHED_NOT_SCIENTIFIC_QUALIFICATION',cases=results,
        no_model_or_QDQ_or_AIME=True,original_failed_probe_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['baseline','profile']);p.add_argument('--supervise',action='store_true');a=p.parse_args()
    assert bool(a.case)!=a.supervise
    worker(a.case)if a.case else supervise()
