"""One bounded local-only observer; no GPU/finisher dispatch or delivery.

Remote code reads whitelisted gold-blind status/identity files and NVML only.
Existing SSH masters are reused without credential input or persistence.
"""
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

TASK='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025'
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/tmp/'+TASK
PYTHON55='/data/ydai/miniconda3/envs/bitdecode/bin/python'
UUIDS={1:'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
       2:'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',
       3:'GPU-30b07aba-a728-7649-4899-e09d33f8bc75'}

REMOTE_CODE=r'''
from pathlib import Path
import csv,hashlib,json,os,signal,subprocess,time
R=Path(REMOTE_ROOT);UUIDS=EXPECTED_UUIDS
signal.alarm(32)
began=time.monotonic();issues=[]

def read_json(path,limit=4*1024*1024):
    # One opened inode is a complete old/new atomic-replace snapshot.
    # Never use stable-path hashing/mtime assertions for mutable PROGRESS.
    try:
        with Path(path).open('rb') as f:
            raw=f.read(limit+1)
        if len(raw)>limit:return None,dict(status='READ_BOUND_EXCEEDED',path=str(path))
        return json.loads(raw),dict(status='READ_OK_OPENED_FD',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    except FileNotFoundError:return None,dict(status='NOT_PRESENT')
    except (OSError,ValueError) as e:
        return None,dict(status='READ_WAIT',error_type=type(e).__name__,path=str(path))

def identity(pid):
    p=Path('/proc')/str(int(pid))
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        raw=(p/'cmdline').read_bytes().rstrip(b'\0')
        argv=[] if not raw else [x.decode(errors='replace') for x in raw.split(b'\0')]
        return dict(pid=int(pid),starttick=int(fields[19]),cmdline=argv,state=fields[0],ppid=int(fields[1]))
    except (OSError,ValueError,IndexError):return None

def compare(expected):
    if not isinstance(expected,dict) or not all(k in expected for k in ('pid','starttick','cmdline')):
        return dict(status='IDENTITY_NOT_RECORDED',owned_alive=False)
    actual=identity(expected['pid'])
    if actual is None:return dict(status='ORIGINAL_PROCESS_ABSENT',owned_alive=False,expected=expected)
    same=(actual['starttick']==expected['starttick'] and actual['cmdline']==expected['cmdline'])
    if actual['starttick']!=expected['starttick']:status='PID_REUSED_FOREIGN'
    elif actual['state'] in ('Z','X','x') or not actual['cmdline']:status='EXIT_TRANSITION_OBSERVED'
    elif same:status='EXACT_PID_STARTTICK_ARGV_ALIVE'
    else:status='SAME_STARTTICK_ARGV_DIFFERS_UNRESOLVED'
    return dict(status=status,owned_alive=same and actual['state'] not in ('Z','X','x'),
                expected=expected,actual=actual)

def scope(argv):
    scripts={'controller':'gpu_controller.py','worker':'gpu_unit_worker.py',
             'finisher':'finalize_offline.py','CPU_score_reader':'score_pass_cpu.py',
             'offline_shell':'run_offline.sh'}
    for kind,name in scripts.items():
        script=str(R/'scripts'/name)
        if script not in argv:continue
        if kind in ('controller','worker'):
            if '--root' not in argv:continue
            at=argv.index('--root')
            if at+1>=len(argv) or argv[at+1]!=str(R):continue
        elif kind!='offline_shell' and str(R) not in argv:continue
        return kind
    return None

# Enumerate only this Unix user's processes and return only exact task paths.
owned=[];scanned=0;scan_deadline=time.monotonic()+4.
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    if scanned>=5000 or time.monotonic()>scan_deadline:
        issues.append('OWN_PROCESS_SCAN_BOUNDED');break
    try:
        if p.stat().st_uid!=os.getuid():continue
    except OSError:continue
    scanned+=1;actual=identity(int(p.name))
    if actual is None:continue
    kind=scope(actual['cmdline'])
    if kind is not None:owned.append(dict(kind=kind,identity=actual))

progress,pmeta=read_json(R/'FORMAL_PROGRESS.json')
session,smeta=read_json(R/'SESSION_RESULT.json')
started,cmeta=read_json(R/'GPU_RUN_STARTED.json')
fin_start,fmeta=read_json(R/'FINALIZER_STARTED.json')
source,srcmeta=read_json(R/'SOURCE_FREEZE.json')
permit,permitmeta=read_json(R/'GPU_PERMIT.json')
ledger,lmeta=read_json(R/'RESOURCE_LEDGER.json')
qualification,qmeta=read_json(R/'GPU_CODEC_QUALIFICATION.json')
admission,ameta=read_json(R/'AUTHORIZATION_FORMAL.json')
seal,sealmeta=read_json(R/'FINAL_SEAL_RECEIPT.json')
archive,archmeta=read_json(R/'FINAL_ARCHIVE_VERIFICATION.json')
terminal,tmeta=read_json(R/'FINAL_TASK_STATUS.json')
for label,meta in (('formal_progress',pmeta),('session',smeta),('controller_start',cmeta),
    ('finisher_start',fmeta),('source',srcmeta),('permit',permitmeta),('ledger',lmeta),
    ('qualification',qmeta),('admission',ameta),('seal',sealmeta),('archive',archmeta),('terminal',tmeta)):
    if meta['status'] not in ('READ_OK_OPENED_FD','NOT_PRESENT'):
        issues.append('INCOMPLETE_READ_'+label+'_'+meta['status'])
controller=compare(None if started is None else started.get('identity'))
finisher=compare(None if fin_start is None else fin_start.get('identity'))
failures={}
for name in ('FIRST_FAILURE.json','FINALIZER_FAILURE.json'):
    failure,meta=read_json(R/name)
    failures[name]=dict(exists=failure is not None,file=meta)
    if failure is not None:
        # Do not download traceback, partial tokens, answer text, or scores.
        failures[name].update(status=failure.get('status'),unit=failure.get('unit'),
            error_type=str(failure.get('error','')).split('(',1)[0][:100])

progress_keys=('phase','monotonic','token','total','generated_tokens','generated_token_count',
    'input_forwards','condition','document','anchor','complete_anchor_batches','propagation_steps')
active=[];expected_identities=[]
if isinstance(ledger,dict):
    expected_identities.extend(x['identity'] for x in ledger.get('intervals',[]) if isinstance(x.get('identity'),dict))
for row in owned:
    if row['kind']!='worker' or row['identity']['state'] in ('Z','X','x'):continue
    actual=row['identity'];argv=actual['cmdline']
    if '--unit' not in argv or argv.index('--unit')+1>=len(argv):continue
    unit=argv[argv.index('--unit')+1]
    if not unit or Path(unit).name!=unit or unit in ('.','..'):
        issues.append('UNSAFE_UNIT_ARGV');continue
    formal='--formal' in argv;folder=R/('formal' if formal else 'execution')/unit
    start,meta=read_json(folder/'STARTED.json');dispatch,dmeta=read_json(folder/'DISPATCH.json')
    current,curmeta=read_json(folder/'PROGRESS.json')
    interval,imeta=read_json(folder/'RESOURCE_INTERVAL.json')
    expected=None if start is None else start.get('identity')
    verified=compare(expected)
    if expected is not None:expected_identities.append(expected)
    selected_gpu=None if dispatch is None else dispatch.get('gpu')
    expected_uuid=UUIDS.get(selected_gpu)
    binding=(dispatch is not None and dispatch.get('unit')==unit and expected_uuid is not None and
             dispatch.get('gpu_uuid')==expected_uuid and dispatch.get('formal') is formal and
             dispatch.get('permit_sha256')==permitmeta.get('sha256'))
    fields={} if not isinstance(current,dict) else {k:current[k] for k in progress_keys if k in current}
    age=None
    if isinstance(fields.get('monotonic'),(int,float)):age=max(0.,time.monotonic()-fields['monotonic'])
    active.append(dict(unit=unit,formal=formal,gpu=selected_gpu,expected_uuid=expected_uuid,
        process=actual,started_identity=verified,dispatch_binding_matches=binding,
        progress=fields,progress_read=curmeta,host_monotonic_progress_age_seconds=age,
        resource_interval_present=interval is not None))

nvml=dict(status='WAIT_NVML_READONLY_UNAVAILABLE',devices=[],apps=[])
try:
    selected=','.join(UUIDS[g] for g in sorted(UUIDS))
    p=subprocess.run(['nvidia-smi','--id='+selected,
        '--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],
        text=True,capture_output=True,timeout=9)
    if p.returncode:raise RuntimeError('NVML readonly device query rc'+str(p.returncode))
    devices=[]
    for fields in csv.reader(p.stdout.strip().splitlines()):
        fields=[x.strip() for x in fields]
        if len(fields)!=5:raise ValueError('malformed NVML device row')
        if fields[1] not in UUIDS.values():continue
        devices.append(dict(physical_index=int(fields[0]),uuid=fields[1],name=fields[2],
            memory_total_MiB=float(fields[3]),memory_free_MiB=float(fields[4])))
    p=subprocess.run(['nvidia-smi','--id='+selected,
        '--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],
        text=True,capture_output=True,timeout=9)
    if p.returncode:raise RuntimeError('NVML readonly process query rc'+str(p.returncode))
    apps=[]
    for fields in csv.reader(p.stdout.strip().splitlines()):
        fields=[x.strip() for x in fields]
        if len(fields)!=3:raise ValueError('malformed NVML process row')
        if fields[1] in UUIDS.values():apps.append(dict(pid=int(fields[0]),uuid=fields[1],memory_MiB=fields[2]))
    expected_by_pid={x['pid']:x for x in expected_identities}
    own=[];foreign=[]
    for app in apps:
        exp=expected_by_pid.get(app['pid']);proof=compare(exp)
        (own if proof['owned_alive'] else foreign).append(dict(app,identity_match=proof['status']))
    device_match=(len(devices)==3 and all(d['physical_index'] in UUIDS and
        UUIDS[d['physical_index']]==d['uuid'] and '3090' in d['name'] for d in devices))
    nvml=dict(status='PASS_READONLY_NVML',devices=devices,apps=apps,authorized_device_map_matches=device_match,
        exact_owned_apps=own,foreign_apps_on_authorized_UUIDs=foreign,GPU0_not_in_scope=True,
        global_NVML_equals_task_not_required=True)
except (OSError,ValueError,subprocess.SubprocessError,RuntimeError) as exc:
    nvml.update(error_type=type(exc).__name__,message=str(exc)[:500])

all_original_exited=('OWN_PROCESS_SCAN_BOUNDED' not in issues and not controller['owned_alive'] and not finisher['owned_alive'] and
    not any(x['identity']['state'] not in ('Z','X','x') for x in owned if x['kind'] in
            ('controller','worker','finisher','CPU_score_reader','offline_shell')))
readiness=bool(session is not None and session.get('own_all_UUID_release') is True and
    fin_start is not None and seal is not None and seal.get('status')=='PASS_FINAL_SEAL' and
    archive is not None and archive.get('status')=='PASS_INDEPENDENT29_ALL_FILE_SHA_BYTES_ATOMIC_SEAL' and
    all_original_exited and nvml['status']=='PASS_READONLY_NVML' and not nvml['exact_owned_apps'])
status=('TERMINAL_SEAL_READY_FOR_INDEPENDENT_AUDIT' if readiness else
    'TRUE_FAILURE_OBSERVED_WAIT_ORIGINAL_TERMINAL_PIPELINE' if any(v['exists'] for v in failures.values()) else
    'WAIT_READONLY_NVML_UNAVAILABLE' if nvml['status']!='PASS_READONLY_NVML' else
    'READONLY_AUTHORIZED_DEVICE_MAP_DIFFERS' if not nvml['authorized_device_map_matches'] else
    'WAIT_READONLY_STATE_READ_INCOMPLETE' if issues else
    'PASS_READONLY_RUNNING' if started is not None else 'PREPARATION_NO_GPU_DISPATCH_OBSERVED')
qualified_count=None
if isinstance(qualification,dict):qualified_count=qualification.get('inspection',{}).get('count')
out=dict(status=status,task_id=R.name,remote_root=str(R),observed_remote_unix=time.time(),
    source_freeze_sha256=srcmeta.get('sha256'),permit_sha256=permitmeta.get('sha256'),
    gold_blind=True,source_file_present=source is not None,permit_present=permit is not None,
    formal_progress=None if progress is None else {k:progress.get(k) for k in
        ('status','completed','planned','completed_by_method','source_freeze_sha256','gold_blind')},
    formal_progress_read=pmeta,qualification_completed=qualified_count,formal_admission_present=admission is not None,
    session=None if session is None else dict(status=session.get('status'),completed_units=len(session.get('completed',[])),
        own_all_UUID_release=session.get('own_all_UUID_release')),
    controller=controller,finisher=finisher,owned_task_processes=owned,active_units=active,
    failures=failures,nvml=nvml,terminal_ready_observed=readiness,all_original_processes_exited_observed=all_original_exited,
    final_status=None if terminal is None else terminal.get('status'),
    seal_status=None if seal is None else seal.get('status'),seal_sha256=sealmeta.get('sha256'),
    archive_verification_status=None if archive is None else archive.get('status'),
    phase='FORMAL' if admission is not None else 'QUALIFICATION' if started is not None else 'PREPARATION',
    process_scan_count=scanned,observer_issues=issues,seconds=time.monotonic()-began,
    readonly_NVML=True,GPU_kernel_calls=0,model_loads=0,scientific_source_mutations=0,
    dispatches=0,signals=0,finisher_calls=0,scoring_calls=0,audit_or_delivery_calls=0,
    readiness_is_independent_delivery_PASS=False)
print(json.dumps(out,sort_keys=True,allow_nan=False))
'''


def _query_remote():
    path=ROOT.parent/'QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/operations/remote.py'
    spec=importlib.util.spec_from_file_location('existing_readonly_ssh_transport',path)
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    code=REMOTE_CODE.replace('REMOTE_ROOT',repr(REMOTE)).replace('EXPECTED_UUIDS',repr(UUIDS))
    result=subprocess.run(transport.command(role='55',python=PYTHON55),input=code,
                          text=True,capture_output=True,timeout=40)
    if result.returncode:
        return dict(status='WAIT_READONLY_SSH_UNAVAILABLE',ssh_returncode=result.returncode,
                    error=result.stderr[-1500:],GPU_retries=0)
    return json.loads(result.stdout)


def _local_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run():
    operations=ROOT/'operations';operations.mkdir(exist_ok=True)
    lock_path=operations/'READONLY_OBSERVER.lock'
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='SKIP_ACTIVE_READONLY_OBSERVER',task_id=TASK,
                        lock_path=str(lock_path),GPU_retries=0,dispatches=0)
        started=datetime.now(timezone.utc);stamp=started.strftime('%Y%m%dT%H%M%S_%fZ')
        lock.seek(0);lock.truncate();lock.write(json.dumps(dict(pid=os.getpid(),started_UTC=started.isoformat(),
            tool_sha256=_local_sha(Path(__file__)),local_root=str(ROOT))));lock.flush()
        before=time.monotonic()
        try:result=_query_remote()
        except (subprocess.TimeoutExpired,TimeoutError) as exc:
            result=dict(status='WAIT_READONLY_SSH_TIMEOUT',error_type=type(exc).__name__,
                        SSH_timeout_seconds=40,GPU_retries=0)
        except (OSError,ValueError,RuntimeError) as exc:
            result=dict(status='WAIT_READONLY_OBSERVER_ERROR',error_type=type(exc).__name__,
                        error=str(exc)[:1500],GPU_retries=0)
        result.update(local_observation_UTC=started.isoformat(),local_root=str(ROOT),
            observer_pid=os.getpid(),observer_tool_sha256=_local_sha(Path(__file__)),
            bounded_readonly_once=True,total_local_seconds=time.monotonic()-before,
            dispatched_GPU_or_finisher=False)
        for key,name in (('source_freeze_sha256','SOURCE_FREEZE.json'),('permit_sha256','GPU_PERMIT.json')):
            local=_local_sha(ROOT/name)
            result['local_'+key]=local
            result['local_remote_'+key+'_match']=None if local is None or result.get(key) is None else local==result[key]
        directory=operations/'READONLY_OBSERVATIONS';directory.mkdir(exist_ok=True)
        path=directory/(stamp+'_'+str(os.getpid())+'.json')
        result['observation_path']=str(path)
        with path.open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        return result


if __name__=='__main__':
    result=run()
    progress=result.get('formal_progress') or {}
    print(json.dumps(dict(status=result['status'],phase=result.get('phase'),
        completed=progress.get('completed'),planned=progress.get('planned'),
        active_units=len(result.get('active_units',[])),terminal_ready=result.get('terminal_ready_observed',False),
        observation_path=result.get('observation_path'),dispatches=0,finisher_calls=0),
        ensure_ascii=False,sort_keys=True))
