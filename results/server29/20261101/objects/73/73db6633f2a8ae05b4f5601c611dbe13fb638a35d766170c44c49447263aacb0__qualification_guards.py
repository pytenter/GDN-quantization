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

def p1_queue_released(root, gpu):
    """Human explicitly cancelled P1; require real stop, never fake10/10 PASS."""
    root = Path(root)
    guards = vendor(root)
    require(sha(P1/'SOURCE_FREEZE.json') == P1_FREEZE, 'Prior P1 identity changed')
    binding = read(root/'PRIOR_P1_USER_STOP_ADMISSION.json')
    grant = read(root/'HUMAN_CONTINUATION_AUTHORIZATION.json')
    require(grant['P1_stop_authorized'] is True and grant['P1_restart_authorized'] is False,
            'Human P1 cancellation scope absent')
    require(binding['human_authorization_sha256'] == sha(root/'HUMAN_CONTINUATION_AUTHORIZATION.json')
            and binding['P1_source_freeze_sha256'] == P1_FREEZE,
            'Human-stop source/grant binding differs')
    receipt_path = root / binding['receipt_relative_path']
    require(receipt_path.resolve().is_relative_to(root.resolve()) and sha(receipt_path) == binding['receipt_sha256'],
            'Prior P1 actual stop receipt differs')
    receipt = read(receipt_path)
    require(receipt['status'] == 'PASS_HUMAN_AUTHORIZED_P1_STOP_EXACT_IDENTITIES_EXITED_OWN_GPU_RELEASED'
            and receipt['human_authorization_sha256'] == binding['human_authorization_sha256']
            and receipt['P1_source_freeze_sha256'] == P1_FREEZE
            and receipt['no_future_original_scheduler_dispatch'] is True
            and receipt['all_exact_own_processes_exited'] is True
            and receipt['own_GPU_UUIDs_released'] is True
            and receipt['original_finisher_absent_independently_verified'] is True,
            'Actual P1 cancellation/exit/release is incomplete')
    identities = [receipt['original_coordinator'], receipt['original_launcher']] + receipt['stopped_workers']
    require(all(not guards.same_process(x) for x in identities), 'An exact old P1 process is still alive')
    paths = {str(P1/'scripts'/name) for name in
             ['gpu_controller.py','gpu_unit_worker.py','run_offline.sh','finalize_offline.py']}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            raw = (p/'stat').read_text(); fields = raw[raw.rfind(')')+2:].split()
            if fields[0] in ('Z','X','x'):
                continue
            args = (p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(not paths.intersection(args), 'An original P1 worker/scheduler/finisher remains')
    obs = guards.query_gpu()
    old_by_pid = {x['pid']: x for x in identities}
    for app in obs['apps']:
        expected = old_by_pid.get(app['pid'])
        if expected is not None:
            current = guards.process_identity(app['pid'])
            require(current is not None and current != expected,
                    'Old PID still appears in NVML without a distinct reused process identity')
    return dict(status='PASS_HUMAN_CANCELLED_P1_RELEASE_NOT_30_COMPLETE',
                P1_source_freeze_sha256=P1_FREEZE, stop_receipt_sha256=binding['receipt_sha256'],
                prior_partial_completed=receipt['original_completed_candidate_count'],
                actual_completed_payloads=receipt['actual_RESULT_payload_count'], gpu=gpu)

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
    dispatch = read(root/'dispatch/qualification'/(unit_id+'.json'))
    parent = vendor(root).process_identity(os.getppid())
    require(dispatch['unit'] == unit_id and dispatch['unit_record'] == unit
            and dispatch['controller_identity'] == parent and vendor(root).same_process(parent)
            and dispatch['source_freeze_sha256'] == permit['source_freeze_sha256']
            and dispatch['permit_sha256'] == sha(root/'GPU_PERMIT.json'),
            'Qualification requires exact current controller/manifest dispatch')
    require(permit['human_authorization_sha256'] == sha(root/'HUMAN_CONTINUATION_AUTHORIZATION.json')
            and permit['prior_p1_stop_admission_sha256'] == sha(root/'PRIOR_P1_USER_STOP_ADMISSION.json'),
            'Qualification permit human-stop admission differs')
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
        prior_P1_user_cancelled_release=completed_queue,monotonic=time.monotonic(),pre_context_NVML=observation)
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
