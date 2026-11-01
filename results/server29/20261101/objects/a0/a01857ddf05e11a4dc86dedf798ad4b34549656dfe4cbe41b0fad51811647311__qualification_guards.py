"""No-CUDA admission for one explicitly named, never-repeated qualification unit.

This module is safe to import before torch. It never waits for, signals, or
modifies another task. An idle gap in a still-running P1 queue is not availability.
"""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

TASK = 'QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1'
UUIDS = {1:'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
         2:'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',
         3:'GPU-30b07aba-a728-7649-4899-e09d33f8bc75'}
P1 = Path('/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022')
P1_FREEZE = '3955f03f2eaa56f83245a85ccad1c56d7d98b70eaa046154a1d95f5b79a7fa14'
_LOCKS = []

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def vendor(root):
    sys.path.insert(0,str(Path(root)/'vendor'))
    import inherited_device_guards as guards
    return guards

def verify_sources(root):
    root=Path(root)
    freeze=read(root/'SOURCE_FREEZE.json')
    for row in freeze['files']:
        p=root/row['path']
        require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], 'Frozen source/asset changed: '+row['path'])
    return sha(root/'SOURCE_FREEZE.json')

def verify_model_and_external(root, fresh_external=True):
    root=Path(root)
    for row in read(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
        s=Path(row['path']).stat()
        require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']), 'Model/tokenizer stat changed')
    rows=read(root/'analysis/EXTERNAL_BINDINGS.json')['files']
    require(len(rows)==61,'Expected actual61 external dependency bindings')
    if fresh_external:
        for row in rows:
            require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'], 'External dependency changed: '+row['path'])
    return rows

def p1_queue_released(root,gpu):
    guards=vendor(root)
    require(sha(P1/'SOURCE_FREEZE.json')==P1_FREEZE,'P1 source admission identity changed')
    jobs=[j for j in read(P1/'FORMAL_JOB_MANIFEST.json')['jobs'] if j['condition']=='P1' and j['gpu_uuid']==UUIDS[gpu]]
    require(len(jobs)==10, 'P1 frozen UUID queue differs')
    for job in jobs:
        d=P1/'formal'/job['id']
        require((d/'RESULT.json').is_file() and (d/'RESOURCE_INTERVAL.json').is_file(), 'RESOURCE_BLOCKED_NO_DISPATCH: P1 UUID queue still has pending samples')
        x=read(d/'RESOURCE_INTERVAL.json')
        require(x['exit_returncode']==0 and x['release']=='PASS' and not guards.same_process(x['identity']), 'RESOURCE_BLOCKED_NO_DISPATCH: P1 own process not released')
    return [j['id'] for j in jobs]

def driver_identity(root,gpu,rows):
    guards=vendor(root)
    source=Path(root)/'vendor/inherited_device_guards.py'
    tree=ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_driver_ordinal_identity')
    replacements=0
    for node in ast.walk(fn):
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Attribute) and node.left.func.attr=='get' and node.left.args and isinstance(node.left.args[0],ast.Constant) and node.left.args[0].value=='CUDA_VISIBLE_DEVICES':
            require(len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value=='0','Driver enumeration AST changed')
            node.comparators[0]=ast.Constant(UUIDS[gpu]);replacements+=1
    require(replacements==1,'Exactly one driver CVD selector change required')
    ns=dict(Path=Path,require=require,sha=sha,os=os,UUID=UUIDS[gpu])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),ns)
    return ns[fn.name](rows)

def _own_gpu(root,gpu,pid=None):
    observation=vendor(root).query_gpu()
    devices=[x for x in observation['devices'] if x['physical_index']==gpu]
    require(len(devices)==1 and devices[0]['uuid']==UUIDS[gpu] and '3090' in devices[0]['name'],'Actual physical GPU UUID/type differs')
    foreign=[x for x in observation['apps'] if x['uuid']==UUIDS[gpu] and x['pid']!=pid]
    require(not foreign,'RESOURCE_BLOCKED_NO_DISPATCH: foreign GPU owner; never terminate it')
    if pid is None:
        require(devices[0]['memory_free_MiB']>=23000,'Insufficient free VRAM for fresh9B')
    else:
        require(any(x['pid']==pid and x['uuid']==UUIDS[gpu] for x in observation['apps']),'Actual own CUDA UUID/PID missing')
    return observation

def admit_single_unit(root,unit_id,gpu,pid):
    root=Path(root).resolve()
    require(gpu in UUIDS and pid==os.getpid(),'Only selectedUUID1/2/3 current process admitted')
    permit=read(root/'GPU_PERMIT.json')
    require(permit['task_id']==TASK and permit['ALLOW_GPU']=='QUALIFICATION_ONLY','NO_GPU_PERMIT: qualification has not been admitted; do not initialize CUDA')
    require(permit['source_freeze_sha256']==verify_sources(root),'Permit/frozen source binding differs')
    require(permit['qualification_plan_sha256']==sha(root/'QUALIFICATION_JOB_MANIFEST.json'),'Qualification plan binding differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==UUIDS[gpu],'CVD must be actual selectedUUID')
    units=read(root/'QUALIFICATION_JOB_MANIFEST.json')['units']
    unit=next(x for x in units if x['id']==unit_id)
    require(unit['gpu']==gpu,'Named unit frozen GPU binding differs')
    target=root/'qualification'/unit_id
    require(not target.exists(),'Unit already attempted; no restart/repeated dispatch')
    completed_queue=p1_queue_released(root,gpu)
    observation=_own_gpu(root,gpu)
    rows=verify_model_and_external(root)
    free=os.statvfs('/data')
    require(free.f_bavail*free.f_frsize>=8*1024**3,'Data headroom below8GiB')
    lock=(root/'operations'/('GPU_'+str(gpu)+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);_LOCKS.append(lock)
    # Recheck ownership after acquiring the new-task lock; never race a pending P1 queue.
    _own_gpu(root,gpu)
    receipt=driver_identity(root,gpu,rows)
    target.mkdir(exist_ok=False)
    identity=vendor(root).process_identity(pid)
    admission=dict(unit=unit_id,identity=identity,source_freeze_sha256=permit['source_freeze_sha256'],
        permit_sha256=sha(root/'GPU_PERMIT.json'),gpu=gpu,uuid=UUIDS[gpu],driver=receipt,
        completed_prior_queue=completed_queue,monotonic=time.monotonic(),pre_context_NVML=observation)
    vendor(root).atomic_new(target/'DISPATCH_AND_ADMISSION.json',admission)
    return dict(unit=unit,target=target,identity=identity,admission=admission)

def check_after_model(root,gpu,pid):
    root=Path(root);verify_model_and_external(root,fresh_external=False)
    observation=_own_gpu(root,gpu,pid)
    mapped=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split()
        if len(fields)<6 or '.so' not in fields[-1]:
            continue
        path=Path(fields[-1])
        require(path.is_file(),'Mapped native library absent/deleted')
        require(int(fields[4])==path.stat().st_ino,'Mapped library inode changed')
        mapped.append(str(path.resolve()))
    rows=read(root/'analysis/EXTERNAL_BINDINGS.json')['files']
    required={str(Path(r['path']).resolve()) for r in rows if '.so' in r['path']}
    controlled=[p for p in mapped if any(s in Path(p).name.lower() for s in ['cuda','cublas','cudnn','cusparse','cusolver','curand','nvrtc','nccl','libtorch','libc10'])]
    require(all(p in required for p in controlled),'Unexpected mapped controlled native library')
    return dict(status='PASS_OWN_UUID_MODEL_STAT_AND_MAPPED_LIBRARY_IDENTITY',NVML=observation,mapped_controlled=sorted(set(controlled)))
