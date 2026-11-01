"""Task-bound CPU admission. Importing this module never queries a GPU."""
import os,json,hashlib,time
from pathlib import Path
from inherited_device_guards import (UUID,require,read,sha,atomic_new,process_identity,
    same_process,query_gpu,validate_gpu,require_driver_ordinal_identity,require_own_gpu_identity,
    terminate_exact,worker_environment,Ledger,binding_rows)
TASK='QWEN_GDN_HISTORICAL_PORT_INT8_GROUP32_QUALIFICATION_V1'

def verify_sources(root):
    root=Path(root);v=read(root/'SOURCE_FREEZE.json')
    for x in v['files']:
        p=root/x['path'];require(p.is_file() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Frozen source mismatch:'+x['path'])
    return sha(root/'SOURCE_FREEZE.json')

def authorize(root,permit):
    root=Path(root);p=read(permit);s=verify_sources(root)
    require(p.get('task_id')==TASK and p.get('ALLOW_GPU_CANARY')=='YES','New task GPU authorization absent')
    require(p.get('source_freeze_sha256')==s and p.get('plan_sha256')==sha(root/'GPU_PLAN.json'),'Permit source/plan binding differs')
    require(p.get('device')=={'physical_index':0,'uuid':UUID},'Permit device differs')
    require(p.get('resource_policy')=='NO_GPU_TIME_BUDGET' and all(k in p and p[k] is None for k in ('max_GPU_hours','hard_seconds','protective_seconds')),'Unlimited-time authorization not explicitly bound')
    require(p.get('no_progress_watchdog_seconds')==1800 and p.get('cleanup_seconds')==120,'Progress watchdog/cleanup contract differs')
    require(p.get('human_separate_authorization') is True and bool(p.get('human_authorization_text')),'Missing separate human grant')
    return p

def worker_admit(root,permit,unit):
    p=authorize(root,permit);d=read(Path(root)/'execution'/unit/'DISPATCH.json')
    require(d['unit']==unit and d['permit_sha256']==sha(permit),'Worker dispatch mismatch')
    require(os.getppid()==d['controller_identity']['pid'] and same_process(d['controller_identity']),'Actual parent controller mismatch')
    require(d.get('resource_policy')=='NO_GPU_TIME_BUDGET' and d.get('execution_run_id')==p.get('execution_run_id'),'Dispatch resource/run binding differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Worker CVD differs')
    return d

def bindings(root):
    base=Path(root)/'inherited/historical_port_payload/analysis/environment'
    rows=[]
    for name in ('PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json','CPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json'):
        rows+=binding_rows(read(base/name))
    return {str(Path(x['path']).resolve()):x for x in rows}

def verify_external(root):
    rows=bindings(root)
    for path,x in rows.items():
        p=Path(path);require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'External dependency changed:'+path)
    return len(rows)

def library_check(root):
    allowed=bindings(root);full=Path('/proc/self/maps').read_text();selected=[];missing=[]
    words=('libtorch','libc10','libcudart','libcublas','libcudnn','libnvrtc','libcuda.','libtorchaudio')
    for line in full.splitlines():
        f=line.split(maxsplit=5)
        if len(f)!=6 or not f[5].startswith('/'):continue
        p=Path(f[5]);key=str(p.resolve())
        relevant=('.so' in p.name and '/site-packages/torch/' in key) or any(x in p.name for x in words)
        if not relevant:continue
        x=allowed.get(key)
        if x is None:missing.append(key);continue
        a=p.stat();h=sha(p);z=p.stat()
        selected.append(dict(path=key,sha256=h,bytes=z.st_size,match=h==x['sha256'] and z.st_size==x['bytes'] and int(f[4])==z.st_ino and (a.st_ino,a.st_mtime_ns)==(z.st_ino,z.st_mtime_ns)))
    result=dict(status='PASS' if selected and not missing and all(x['match'] for x in selected) else 'FAIL',full_maps=full,selected=selected,missing=missing,identity=require_own_gpu_identity(os.getpid()))
    return result
