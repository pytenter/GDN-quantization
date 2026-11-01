"""Explicitly authorized control-only continuation; never rerun completed units.

This standalone stdlib supervisor imports the unchanged frozen coordinator's
hardware/process/source helpers, and changes only their I/O ROOT binding.
It is not a new science protocol or a repair of the original sealed execution.
--deploy-static is explicit; --prepare only validates a previously staged root.
--run must be invoked by Root inside a real detached tmux server. No SSH code.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback

TASK='QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1'
CONTINUATION='CONTINUATION_01'
INITIAL_MANIFEST_SHA='eaf0395728a0c57efbdbc28cb6d062803896c8c132cc4c5dd35687267f6eb1d4'
INITIAL_GPUH=1.5062975196954276
FP_SCORE_SHA='98e3e2befa8a0e3aab3dc948c0991d82126fdba39848d55fb9019ee4d5d2a4a0'
FP_SCORE='scores/science_doc1_FP_NATIVE.pt'
INHERITED_PHASES=('canary_initial','canary_fresh','science_doc1_FP_NATIVE')
ALL_SCIENCE=tuple(f'science_doc{d}_{c}' for d in (1,2) for c in ('FP_NATIVE','H_C128','H_DUAL_M'))
REMAINING=ALL_SCIENCE[1:]
CANARY_TAGS=('N_OFF','A_REFERENCE','A_OFF','A_ON','D_OFF','D_ON','N_FRESH','A_FRESH','D_FRESH')
REQUIRED_BINDINGS=('hashes/artifact_sha256.txt','hashes/DELIVERY_VERIFICATION.json','source_manifest.json',
    'FIRST_FAILURE.json','GATE_STATUS_FINAL.json','gpu_resource_ledger.json','CANARY_INITIAL.json','G4_RUNTIME_BUDGET.json',
    'analysis/science_doc1_FP_NATIVE.json','analysis/RECEIPT_science_doc1_FP_NATIVE.json',FP_SCORE)
FORBIDDEN_ACTIVE=('FIRST_FAILURE.json','GATE_STATUS_FINAL.json','gpu_resource_ledger.json','COORDINATOR_STARTED.json',
    'FINAL_STATUS.json','FINAL_SCIENTIFIC_REPORT.md','science_results.json','science_results.csv','window_results.csv',
    'scale_safety.json','codec_parity.json','numerical_qualification.json','runtime_cost.json',
    'analysis/POSTPROCESS_STATUS.json','checkpoints/CURRENT.pt','checkpoints/CATALOG.jsonl')
TENSOR_SUFFIXES={'.pt','.pth','.ckpt','.bin','.safetensors','.npy','.npz','.pkl','.pickle'}


def need(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def safe(root,relative):
    p=Path(relative)
    need(not p.is_absolute() and '..' not in p.parts and not any(ord(c)<32 for c in str(p)),'Unsafe relative artifact path')
    target=(root/p).resolve();need(target.is_relative_to(root.resolve()),'Artifact escapes continuation root')
    return target


def publish(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        need(path.is_file() and path.read_bytes()==data,'Existing artifact conflicts; no overwrite: '+str(path));return False
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    return True


def new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def load_config(path,initial_root=None,continuation_root=None):
    cfg=read(path)
    if initial_root is not None:cfg['original_root']=str(initial_root)
    if continuation_root is not None:cfg['continuation_root']=str(continuation_root)
    old=Path(cfg['original_root']).resolve();new=Path(cfg['continuation_root']).resolve()
    need(cfg['task_id']==TASK and cfg['continuation_id']==CONTINUATION,'Wrong continuation identity')
    need(old.name==new.name==TASK and old!=new and not new.is_relative_to(old) and not old.is_relative_to(new),'Require distinct isolated I/O roots retaining the task basename')
    need(cfg['initial_source_manifest_sha256']==INITIAL_MANIFEST_SHA,'Original prospective source identity must not change')
    need(cfg['prior_GPU_hours']==INITIAL_GPUH,'No budget reset or omission of the original qualifications')
    for rel in REQUIRED_BINDINGS:
        value=cfg['original_bindings'].get(rel)
        need(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Missing exact original binding: '+rel)
    need(cfg['original_bindings']['source_manifest.json']==INITIAL_MANIFEST_SHA and cfg['original_bindings'][FP_SCORE]==FP_SCORE_SHA,'Frozen original manifest/FP identity mismatch')
    auth=cfg['human_authorization']
    need(auth['message_verbatim']=='你重新连接啊' and auth['human_was_informed_of_original_FAIL_and_BrokenPipe'] is True,'Missing actual informed follow-up authorization provenance')
    need(isinstance(auth.get('root_interpretation_disclosed_to_human'),str) and auth['root_interpretation_disclosed_to_human'].strip(),
         'Record Root interpretation separately from the exact human quote; do not fabricate a five-phase user quote')
    need(cfg['resource_gate_relative_path']=='CONTINUATION_RESOURCE_GATE.json','Use an explicit independently reviewed continuation resource gate')
    need(len(cfg['resource_gate_sha256'])==64,'Independent resource receipt SHA is required; no automatic resource waiver')
    return cfg,old,new


def parse_manifest(path):
    rows={}
    for line in Path(path).read_text().splitlines():
        digest,rel=line.split('  ',1);need(rel not in rows and len(digest)==64,'Malformed/duplicate original seal entry');rows[rel]=digest
    return rows


def verify_original(cfg,old):
    for rel,digest in cfg['original_bindings'].items():need(sha(safe(old,rel))==digest,'Original bound evidence changed: '+rel)
    receipt=read(old/'hashes/DELIVERY_VERIFICATION.json');manifest=parse_manifest(old/'hashes/artifact_sha256.txt')
    need(receipt['status']=='PASS' and receipt['manifest_sha256']==cfg['original_bindings']['hashes/artifact_sha256.txt'],'Original execution must first have an immutable complete STOP delivery seal')
    need({k:v['sha256'] for k,v in receipt['all_payload_assets'].items()}==manifest,'Original receipt and checksum inventory disagree')
    checked=[]
    for rel,digest in manifest.items():
        p=safe(old,rel);before=p.stat();actual=sha(p);after=p.stat()
        need((before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)==(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev)
             and actual==digest,'Original sealed asset changed during/current hash: '+rel)
        checked.append({'path':rel,'sha256':actual,'bytes':after.st_size})
    gate=read(old/'GATE_STATUS_FINAL.json');ledger=read(old/'gpu_resource_ledger.json');failure=read(old/'FIRST_FAILURE.json')
    need(gate['status']=='FAIL_AND_STOP' and gate['GPU_release_verified'] is True and gate['no_further_GPU_work'] is True,'Original FAIL/release barrier not preserved')
    need(failure['no_later_GPU_work'] is True and 'BrokenPipeError' in failure['error'],'This continuation authorizes only the disclosed original stdout-control STOP')
    need(ledger['GPU_HOURS_USED']==gate['GPU_HOURS_USED']==INITIAL_GPUH and ledger['final_GPU_release_verified'] is True,'Original cumulative GPU budget/release identity failed')
    need(tuple(p['phase'] for p in ledger['phases'])==INHERITED_PHASES and all(p['released'] is True and p['returncode']==0 for p in ledger['phases']),
         'Exactly the two canary phases and one complete FP phase must be inherited')
    need(all(gate['gates'][p]=='PASS' for p in INHERITED_PHASES) and all(gate['gates'][p]=='NOT_RUN' for p in REMAINING),'Never restart a completed/active scientific unit')
    need(read(old/'analysis/RECEIPT_science_doc1_FP_NATIVE.json')['result']==dict(status='PASS',condition='FP_NATIVE',doc=1,consumed=32768,scores=384),'Original FP must have a real complete worker receipt')
    return dict(status='PASS',verified_original_sealed_entries=len(checked),sealed_assets=checked,
                original_manifest_sha256=cfg['original_bindings']['hashes/artifact_sha256.txt'],
                original_receipt_sha256=cfg['original_bindings']['hashes/DELIVERY_VERIFICATION.json'],
                original_failure_sha256=cfg['original_bindings']['FIRST_FAILURE.json'],
                original_ledger_sha256=cfg['original_bindings']['gpu_resource_ledger.json'],original_GPU_hours=INITIAL_GPUH)


def active_static_paths(old):
    paths={x['path'] for x in read(old/'source_manifest.json')['execution_sources']}|{'source_manifest.json','CANARY_INITIAL.json','G4_RUNTIME_BUDGET.json','analysis/CANARY_REDUCER.json'}
    paths|={f'analysis/{tag}.json' for tag in CANARY_TAGS+('science_doc1_FP_NATIVE',)}
    paths|={f'scores/{tag}.pt' for tag in CANARY_TAGS+('science_doc1_FP_NATIVE',)}
    paths|={f'analysis/{prefix}{phase}.json' for phase in INHERITED_PHASES for prefix in ('RECEIPT_','ENV_','PROCESS_')}
    return sorted(paths)


def deploy_static(config_path,initial_root=None,continuation_root=None):
    cfg,old,new=load_config(config_path,initial_root,continuation_root);before=verify_original(cfg,old)
    new.mkdir(parents=True,exist_ok=True)
    for rel in FORBIDDEN_ACTIVE:need(not safe(new,rel).exists(),'Active continuation root must not inherit old mutable/final status: '+rel)
    deployed=[]
    for rel in active_static_paths(old):
        src=safe(old,rel);dst=safe(new,rel);need(src.is_file(),'Required inherited asset absent: '+rel)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():need(sha(dst)==sha(src),'Existing inherited copy conflicts: '+rel)
        elif rel.startswith('scores/'):
            try:os.link(src,dst)
            except FileExistsError:need(sha(dst)==sha(src),'Concurrent inherited tensor publication conflicts')
            except OSError:
                with src.open('rb') as incoming,dst.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing,8*1024**2)
        else:publish(dst,src.read_bytes())
        need(sha(dst)==sha(src),'Inherited copy verification failed: '+rel)
        deployed.append(dict(path=rel,sha256=sha(src),bytes=src.stat().st_size,hardlink=src.stat().st_dev==dst.stat().st_dev and src.stat().st_ino==dst.stat().st_ino))
    history=new/'inherited_initial';history.mkdir(exist_ok=True)
    # All actual small historical evidence is copied byte-for-byte, never a
    # fabricated placeholder. Big tensors stay at their sealed original paths.
    locators=[]
    historical={x['path']:x for x in before['sealed_assets']}
    historical.update({rel:dict(path=rel,sha256=sha(old/rel),bytes=(old/rel).stat().st_size) for rel in ('hashes/artifact_sha256.txt','hashes/DELIVERY_VERIFICATION.json')})
    for rel,row in sorted(historical.items()):
        src=safe(old,rel)
        if src.suffix.lower() in TENSOR_SUFFIXES:
            locators.append(dict(row,actual_path=str(src),retained_at_original_root=True,copied_into_history=False));continue
        publish(safe(history,rel),src.read_bytes())
    publish(history/'REMOTE_TENSOR_LOCATORS.json',(json.dumps(locators,ensure_ascii=False,indent=2)+'\n').encode())
    if 'resource_gate_source' in cfg:
        resource_source=Path(cfg['resource_gate_source']);need(sha(resource_source)==cfg['resource_gate_sha256'],'Reviewed resource source receipt changed')
        publish(new/cfg['resource_gate_relative_path'],resource_source.read_bytes())
    after=verify_original(cfg,old);need(before==after,'Original sealed bytes changed during deployment')
    publish(new/'CONTINUATION_DEPLOYMENT.json',(json.dumps(dict(status='PASS',task_id=TASK,continuation_id=CONTINUATION,
        original_root=str(old),active_static_assets=deployed,original_snapshot=before,
        original_CURRENT_preserved_only_at=str(old/'checkpoints/CURRENT.pt'),original_catalog_only_in_history=True,
        no_GPU_work=True),ensure_ascii=False,indent=2)+'\n').encode())
    return dict(status='PASS',mode='DEPLOY_STATIC_ONLY',continuation_root=str(new),files=len(deployed),GPU_work_performed=False)


def validate_resource_receipt(cfg,new):
    p=safe(new,cfg['resource_gate_relative_path']);need(p.is_file() and sha(p)==cfg['resource_gate_sha256'],'Independent current continuation resource receipt absent/changed')
    gate=read(p);need(gate.get('status')=='PASS','Independent resource review has not passed; no GPU dispatch')
    need(gate.get('GPU_UUID')=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422' and gate.get('original_resource_ledger_sha256')==cfg['original_bindings']['gpu_resource_ledger.json'],
         'Resource review must bind the actual GPU and original completed reservation')
    need(gate.get('scope')=='FROZEN_G4_ADMISSION_REUSED_WITH_DISCLOSED_UNCERTAINTY'
         and gate.get('D32K_capacity_empirically_established') is False
         and gate.get('new_reserved_upper_bound_established') is False
         and gate.get('original_memory_threshold_headroom_allocator_dtype_device_unchanged') is True,
         'Resource continuation must disclose uncertainty and preserve original memory/allocator/device rules')
    return dict(path=cfg['resource_gate_relative_path'],sha256=sha(p),receipt=gate)


def prepare(config_path,initial_root=None,continuation_root=None):
    cfg,old,new=load_config(config_path,initial_root,continuation_root);need(new.is_dir(),'Deploy static assets before prepare')
    need(not (new/'CONTINUATION_PREPARED.json').exists(),'Prepared continuation is immutable; do not overwrite')
    original=verify_original(cfg,old)
    for rel in FORBIDDEN_ACTIVE:need(not safe(new,rel).exists(),'Old final/mutable execution state must remain only in history: '+rel)
    inherited=[]
    for rel in active_static_paths(old):
        p=safe(new,rel);need(p.is_file() and sha(p)==sha(safe(old,rel)),'Inherited active asset differs: '+rel)
        inherited.append(dict(path=rel,sha256=sha(p),bytes=p.stat().st_size))
    resource=validate_resource_receipt(cfg,new)
    publish(new/'continuation_control/CONTROLLER_SOURCE.py',Path(__file__).read_bytes())
    publish(new/'continuation_control/CONFIG_SNAPSHOT.json',Path(config_path).read_bytes())
    recovery=dict(task_id=TASK,continuation_id=CONTINUATION,original_root=str(old),continuation_root=str(new),
        human_authorization=cfg['human_authorization'],original_execution_classification='FAIL_AND_STOP_PRESERVED',
        original_failure_sha256=cfg['original_bindings']['FIRST_FAILURE.json'],original_bindings=cfg['original_bindings'],
        original_prospective_source_manifest_sha256=INITIAL_MANIFEST_SHA,science_source_files_modified=False,
        cumulative_prior_GPU_hours=INITIAL_GPUH,MAX_GPU_HOURS=24.0,MAX_NEW_ARCHIVE_GIB_COMBINED=10,
        inherited_PHASES=list(INHERITED_PHASES),remaining_PHASES=list(REMAINING),
        no_repeated_qualification_or_FP_prefix=True,core_math_codecs_documents_scoring_unchanged=True,
        checkpoint_policy='Original CURRENT and catalog stay at original root/history; active CURRENT/catalog begin with new remaining trajectories only',
        inherited_phase_annotations='Exact prior reservation objects are preserved in ledger.phases; provenance is a parallel annotation, never a rewrite of the old ledger',
        resource_review=resource,controller_source_sha256=sha(__file__),config_sha256=sha(config_path))
    new_json(new/'CONTINUATION_RECOVERY_MANIFEST.json',recovery)
    new_json(new/'CONTINUATION_PREPARED.json',dict(status='PASS',task_id=TASK,continuation_id=CONTINUATION,
        prepared_unix=time.time(),controller_source_sha256=sha(__file__),config_sha256=sha(config_path),
        recovery_manifest_sha256=sha(new/'CONTINUATION_RECOVERY_MANIFEST.json'),original_verification=original,
        inherited_static_assets=inherited,resource_gate_sha256=resource['sha256'],no_GPU_work=True))
    publish(new/'CONTINUATION_README.txt',(
        'This is an explicitly authorized control-only continuation after a sealed stdout BrokenPipe STOP.\n'
        'Original FAIL_AND_STOP and FIRST_FAILURE remain immutable in the original root and inherited_initial.\n'
        'A future COMPLETE here means five remaining trajectories plus exact inherited PASS assets; it never rewrites the original FAIL as PASS.\n'
        'Human verbatim: 你重新连接啊 . Root interpretation and prior failure disclosure are separately recorded in the recovery manifest.\n'
        'The original two canary phases and Doc1 FP_NATIVE are not rerun. Total budget carries all prior 1.5062975196954276 GPUh.\n'
        'All five science source files, documents, preregistration, codecs, runtime arithmetic and KL reducer remain byte-identical.\n'
        'No new attempt retry after a continuation GPU failure. Active checkpoint/catalog contain only new remaining trajectories.\n'
    ).encode())
    need(verify_original(cfg,old)==original,'Original seal changed during CPU prepare')
    return dict(status='PASS',mode='PREPARE_VALIDATE_ONLY',recovery_manifest_sha256=sha(new/'CONTINUATION_RECOVERY_MANIFEST.json'),GPU_work_performed=False)


def import_helpers(new):
    p=new/'src/coordinator.py';spec=importlib.util.spec_from_file_location('unchanged_frozen_coordinator_helpers',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=new
    return module


def logical_bytes(root):
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        need(not any((Path(directory)/d).is_symlink() for d in dirs),'No symlink directories in combined archive')
        for name in files:
            p=Path(directory)/name;total+=p.lstat().st_size if p.is_symlink() else p.stat().st_size
    return total


def combined_storage(C,old):
    current=C.artifact_usage();prior=logical_bytes(old);total=current['new_archive_bytes']+prior
    C.require(total<C.MAX_ARCHIVE_BYTES,'Original plus continuation logical archive reaches10GiB cap','BLOCKED_AND_STOP')
    C.require(current['free_disk_bytes']>=C.MIN_FREE_DISK_BYTES,'Filesystem free below original1GiB reserve','BLOCKED_AND_STOP')
    return dict(new_archive_bytes=current['new_archive_bytes'],original_archive_bytes=prior,combined_archive_bytes=total,
                free_disk_bytes=current['free_disk_bytes'],hardlinks_counted_twice_conservatively=True)


def projection(C,g4,unfinished,ledger):
    original,method=C.remaining_science_projection(g4,unfinished,ledger)
    observed=[p['reservation_seconds'] for p in ledger['phases'] if p['phase'] in ALL_SCIENCE]
    floor=max(observed)*1.25*len(unfinished) if observed else 0.0
    return max(original,floor),dict(original_method=method,original_G4_remaining_seconds=original,
        observed_completed_science_reservation_floor_seconds=floor,observed_margin=1.25)


def resident_stdio(new):
    p=new/'logs/CONTINUATION_COORDINATOR_STDOUT.log';p.parent.mkdir(exist_ok=True)
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600);null=os.open('/dev/null',os.O_RDONLY)
    os.dup2(null,0);os.dup2(fd,1);os.dup2(fd,2);os.close(null);os.close(fd)
    sys.stdout=open(1,'w',buffering=1,closefd=False);sys.stderr=open(2,'w',buffering=1,closefd=False)
    need(stat.S_ISREG(os.fstat(1).st_mode) and stat.S_ISREG(os.fstat(2).st_mode),'Supervisor stdout/stderr must be resident regular files, never SSH pipes')
    return str(p)


def tmux_residency():
    need(os.environ.get('TMUX') and os.environ.get('TMUX_PANE'),'Run must actually originate in a tmux pane')
    pane=os.environ['TMUX_PANE']
    # tmux1.8 supports pane_pid/session_name but not the newer #{pid} format.
    raw=subprocess.check_output(['tmux','display-message','-p','-t',pane,'#{pane_pid} #{session_name}'],text=True,timeout=10).strip().split(maxsplit=1)
    need(len(raw)==2 and raw[0].isdigit(),'Cannot independently query the live tmux pane/session')
    fields=os.environ['TMUX'].rsplit(',',2)
    need(len(fields)==3 and fields[0] and fields[1].isdigit() and fields[2].isdigit(),'Malformed standard TMUX socket,serverPID,session environment')
    pane_pid=int(raw[0]);server_pid=int(fields[1]);proc=Path('/proc')/str(server_pid)
    need(proc.stat().st_uid==os.getuid(),'tmux server is not owned by the current user')
    comm=(proc/'comm').read_text().strip();cmdline=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    need(comm.startswith('tmux') and cmdline and Path(cmdline[0]).name.startswith('tmux'),'TMUX serverPID is not an actual tmux process')
    executable=Path(shutil.which('tmux') or '')
    need(executable.is_file() and os.path.samefile(proc/'exe',executable),'TMUX serverPID executable does not match the queried tmux binary')
    p=os.getpid();ancestors=[]
    while p>1 and len(ancestors)<64:
        ancestors.append(p)
        if p==server_pid:break
        text=(Path('/proc')/str(p)/'stat').read_text();fields=text[text.rfind(')')+2:].split();p=int(fields[1])
    need(pane_pid in ancestors and server_pid in ancestors and ancestors.index(pane_pid)<=ancestors.index(server_pid),
         'Controller ancestry does not reach the queried pane and its actual tmux server; a log-view pane is insufficient')
    return dict(status='PASS',pane=pane,pane_PID=pane_pid,server_PID=server_pid,server_comm=comm,server_cmdline=cmdline,
                server_same_UID=True,server_executable_matches_queried_binary=True,session=raw[1],controller_PID=os.getpid(),controller_ancestor_chain_to_tmux_server=ancestors)


def emit(value,ledger):
    # Informational output can never downgrade an already persisted PASS unit.
    try:print(json.dumps(value,ensure_ascii=False),flush=True)
    except (BrokenPipeError,OSError) as exc:
        ledger.setdefault('informational_stdout_errors',[]).append(dict(unix=time.time(),error=repr(exc),phase=value.get('phase'),scientific_PASS_preserved=True))


def run(config_path,initial_root=None,continuation_root=None):
    cfg,old,new=load_config(config_path,initial_root,continuation_root);prepared=read(new/'CONTINUATION_PREPARED.json')
    need(prepared['status']=='PASS' and prepared['controller_source_sha256']==sha(__file__) and prepared['config_sha256']==sha(config_path),'Prepared controller/config changed')
    need(prepared['recovery_manifest_sha256']==sha(new/'CONTINUATION_RECOVERY_MANIFEST.json'),'Recovery authorization binding changed')
    recovery=read(new/'CONTINUATION_RECOVERY_MANIFEST.json')
    need(recovery['original_root']==str(old) and recovery['continuation_root']==str(new),'Explicit run roots differ from the prepared recovery binding')
    tmux=tmux_residency()
    # Single-file exclusive guard wins before any model or GPU dispatch.
    new_json(new/'CONTINUATION_RUN_GUARD.json',dict(pid=os.getpid(),unix=time.time(),task_id=TASK,
        continuation_id=CONTINUATION,source_manifest_sha256=INITIAL_MANIFEST_SHA,recovery_manifest_sha256=prepared['recovery_manifest_sha256']))
    logfile=resident_stdio(new);C=import_helpers(new);C.require(C.MAX_GPU_SECONDS==24*3600 and C.SCIENCE_RESERVE_SECONDS==900 and C.EXIT_RESERVE_SECONDS==300,'Original safety budgets changed')
    new_json(new/'CONTINUATION_TMUX_RESIDENCY.json',tmux)
    for name in ('logs','analysis','scores','checkpoints'):(new/name).mkdir(exist_ok=True)
    old_snapshot=verify_original(cfg,old);need(old_snapshot==prepared['original_verification'],'Original sealed evidence changed since prepare')
    validate_resource_receipt(cfg,new)
    inherited=read(old/'gpu_resource_ledger.json');gates=copy.deepcopy(read(old/'GATE_STATUS_FINAL.json')['gates'])
    ledger=copy.deepcopy(inherited)
    for name in ('final_status','final_GPU_release_verified','final_GPU_release_probe','ended_unix'):ledger.pop(name,None)
    ledger.update(task_id=TASK,continuation_id=CONTINUATION,started_unix=time.time(),GPU_HOURS_USED=INITIAL_GPUH,
        MAX_GPU_HOURS=24.0,prior_execution_GPU_hours=INITIAL_GPUH,inherited_phase_count=len(INHERITED_PHASES),
        inherited_original_ledger_sha256=cfg['original_bindings']['gpu_resource_ledger.json'],
        inherited_phase_provenance={str(i):dict(original_root=str(old),original_phase=p['phase'],original_entry_sha256=canonical(p)) for i,p in enumerate(inherited['phases'])},
        historical_initial_execution_status='FAIL_AND_STOP_PRESERVED',
        historical_initial_failure_sha256=cfg['original_bindings']['FIRST_FAILURE.json'],
        accounting='Exact prior3 reservations unchanged plus only new remaining5; no budget reset. New exclusive reservations include load/compute/IO/until verified release.',
        supervisor_stdio_file=logfile)
    phase=None;last_accounted=None;failure=None;final_status='FAIL_AND_STOP'
    def termination_signal(signum,frame):raise C.StopExecution(f'Continuation received signal {signum}; stop exact own worker without retry')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,termination_signal)
    frozen=C.read(new/'source_manifest.json');g4=C.read(new/'G4_RUNTIME_BUDGET.json');g4sha=C.sha(new/'G4_RUNTIME_BUDGET.json')
    try:
        C.require(not (new/'FIRST_FAILURE.json').exists(),'Existing continuation failure prohibits any later GPU work')
        C.require(C.sha(new/'source_manifest.json')==INITIAL_MANIFEST_SHA,'Frozen science manifest changed')
        C.source_check(frozen);C.frozen_document_check();combined_storage(C,old)
        for phase0 in INHERITED_PHASES:C.require(gates[phase0]=='PASS','Required inherited phase is not PASS')
        for phase0 in REMAINING:C.require(gates[phase0]=='NOT_RUN' and not (new/f'analysis/RECEIPT_{phase0}.json').exists(),'An intended remaining unit already ran; refuse duplicate')
        C.require(g4.get('status')=='PASS','Original qualified G4 is not PASS')
        C.write('COORDINATOR_STARTED.json',dict(task_id=TASK,pid=os.getpid(),unix=time.time(),phases=list(REMAINING),
            source_manifest_sha256=INITIAL_MANIFEST_SHA,recovery_manifest_sha256=prepared['recovery_manifest_sha256'],
            continuation_id=CONTINUATION,original_FAIL_preserved=True,prior_GPU_hours=INITIAL_GPUH,single_attempt_no_retries=True),exclusive=True)
        admission=C.idle_admission();C.write('gpu_admission.json',dict(status='PASS',probe=admission,GPU_UUID=C.GPU_UUID,
            physical_GPU_ID=0,CUDA_mapping='CUDA_VISIBLE_DEVICES=0 -> cuda:0',MAX_GPU_COUNT=1,MAX_CONCURRENT_TRAJECTORIES=1,
            MAX_GPU_HOURS=24.0,independent_continuation_resource_review_sha256=cfg['resource_gate_sha256']),exclusive=True)
        for phase in REMAINING:
            C.require(not (new/'FIRST_FAILURE.json').exists(),'Continuation FIRST_FAILURE forbids later dispatch')
            C.require(C.sha(new/'source_manifest.json')==INITIAL_MANIFEST_SHA and C.sha(new/'G4_RUNTIME_BUDGET.json')==g4sha,'Frozen science/G4 identity changed')
            C.source_check(frozen);C.frozen_document_check();validate_resource_receipt(cfg,new)
            C.require(not (new/f'analysis/RECEIPT_{phase}.json').exists() and not (new/f'scores/{phase}.pt').exists(),'Completed artifact/unit must never be overwritten or rerun')
            remaining=C.MAX_GPU_SECONDS-ledger['GPU_HOURS_USED']*3600
            unfinished=[p for p in REMAINING if gates[p]!='PASS'];prediction,method=projection(C,g4,unfinished,ledger)
            C.require(prediction<=remaining-C.SCIENCE_RESERVE_SECONDS,'Conservative forecast cannot complete all remaining pairs within carried24GPUh and900s reserve','BLOCKED_AND_STOP')
            storage=combined_storage(C,old);probe=C.idle_admission()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
            command=[C.PYTHON,'-u',str(new/'src/worker.py'),phase];child=None;recorded=None;phase_error=None;cleanup_error=None;log=None
            start=time.monotonic();entry=dict(phase=phase,dispatch_unix=time.time(),command=command,budget_remaining_before_seconds=remaining,
                remaining_science_prediction_seconds=prediction,forecast_method=method,storage_before=storage,peak_sampled_NVML_MiB=probe['used_MiB'],NVML_sample_count=0,coarse_NVML_samples=[],released=False,
                continuation_id=CONTINUATION)
            try:
                log=(new/f'logs/{phase}.log').open('x');child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                entry['pid']=child.pid;recorded=C.process_identity(child.pid);C.require(C.exact_child(recorded,phase),'Exact new worker PID/cmdline mismatch')
                entry['process_identity_at_dispatch']=recorded;C.write(f'analysis/PROCESS_{phase}.json',recorded,exclusive=True)
                last_progress=start;last_size=-1;last_storage=start;last_live=start;last_coarse=start-30
                while child.poll() is None:
                    time.sleep(C.SAMPLE_SECONDS);probe=C.nvml_probe();C.hardware_identity(probe);elapsed=time.monotonic()-start
                    entry['NVML_sample_count']+=1;entry['peak_sampled_NVML_MiB']=max(entry['peak_sampled_NVML_MiB'],probe['used_MiB'])
                    C.require(all(a['pid']==child.pid for a in probe['apps']),'Foreign process on reservedGPU0; terminate only own exactPID')
                    C.require(remaining-elapsed>C.EXIT_RESERVE_SECONDS,'Carried24GPUh reached300s safe-stop reserve','BLOCKED_AND_STOP')
                    C.require(not (new/'FIRST_FAILURE.json').exists(),'Worker recorded FIRST_FAILURE; no dependent phase')
                    size=(new/f'logs/{phase}.log').stat().st_size
                    if size!=last_size:last_progress=time.monotonic();last_size=size
                    C.require(time.monotonic()-last_progress<C.WATCHDOG_SECONDS,'30-minute log no-progress watchdog; no retry')
                    if time.monotonic()-last_storage>=5:entry['storage_latest']=combined_storage(C,old);last_storage=time.monotonic()
                    if time.monotonic()-last_coarse>=30:entry['coarse_NVML_samples'].append(probe);last_coarse=time.monotonic()
                    if time.monotonic()-last_live>=10:
                        C.write('analysis/GPU_RESOURCE_LEDGER_LIVE.json',dict(ledger,active_phase=phase,active_PID=child.pid,active_reserved_seconds=elapsed,
                            active_cumulative_GPUh=ledger['GPU_HOURS_USED']+elapsed/3600,active_peak_NVML_MiB=entry['peak_sampled_NVML_MiB']));last_live=time.monotonic()
                entry['returncode']=child.wait()
            except BaseException as exc:phase_error=exc;entry['phase_exception']=repr(exc)
            finally:
                if child is not None:
                    try:entry['cleanup']=C.terminate_own_child(child,recorded,phase)
                    except BaseException as exc:cleanup_error=exc;entry['cleanup_error']=repr(exc)
                    entry['returncode']=child.poll();released,release_probe=C.wait_for_release(child.pid)
                else:release_probe=C.safe_probe();released=release_probe['status']=='PASS' and not release_probe['apps']
                elapsed=time.monotonic()-start;entry.update(reservation_seconds=elapsed,GPU_hours=elapsed/3600,release_observation_unix=time.time(),
                    release_probe=release_probe,released=released,reservation_duration_is_lower_bound_if_release_unverified=not released)
                if log is not None:log.close()
                ledger['phases'].append(entry);ledger['GPU_HOURS_USED']+=elapsed/3600;last_accounted=time.monotonic();C.write('analysis/GPU_RESOURCE_LEDGER_LIVE.json',ledger)
            C.require(entry['released'],'Own GPU release not verified');C.require(cleanup_error is None,'Exact PID cleanup failed: '+str(cleanup_error))
            if phase_error is not None:raise phase_error
            C.require(entry['returncode']==0,'Worker failed; no patch/retry');C.require(not (new/'FIRST_FAILURE.json').exists(),'Worker FIRST_FAILURE exists after exit')
            receipt=C.read(new/f'analysis/RECEIPT_{phase}.json');C.require(receipt.get('status')=='PASS','Valid completed worker PASS receipt absent')
            C.require(C.sha(new/'source_manifest.json')==INITIAL_MANIFEST_SHA,'Frozen source manifest changed during phase');C.source_check(frozen);combined_storage(C,old)
            gates[phase]='PASS';entry['receipt_sha256']=C.sha(new/f'analysis/RECEIPT_{phase}.json')
            C.write('analysis/GPU_RESOURCE_LEDGER_LIVE.json',ledger)
            C.write(f'analysis/CONTINUATION_PASS_{phase}.json',dict(status='PASS',phase=phase,receipt_sha256=entry['receipt_sha256'],cumulative_GPUh=ledger['GPU_HOURS_USED']),exclusive=True)
            emit(dict(phase=phase,status='PASS',reservation_seconds=elapsed,cumulative_GPUh=ledger['GPU_HOURS_USED'],release_verified=True),ledger)
        final_status='COMPLETE_AND_STOP'
    except BaseException as exc:
        if phase is not None and gates[phase]!='PASS':gates[phase]='FAIL' if any(e['phase']==phase and e.get('continuation_id')==CONTINUATION for e in ledger['phases']) else 'NOT_RUN'
        final_status=getattr(exc,'classification','FAIL_AND_STOP');failure=dict(status=final_status,phase=phase,error=repr(exc),traceback=traceback.format_exc(),unix=time.time(),
            no_GPU_retry=True,no_later_GPU_work=True,continuation_id=CONTINUATION,original_FAIL_preserved=True)
        if not (new/'FIRST_FAILURE.json').exists():C.write('FIRST_FAILURE.json',failure,exclusive=True)
    final_probe=C.safe_probe()
    if len(ledger['phases'])>len(INHERITED_PHASES) and not ledger['phases'][-1]['released'] and last_accounted is not None:
        e=ledger['phases'][-1];extension=max(0.0,time.monotonic()-last_accounted);e['reservation_seconds']+=extension;e['GPU_hours']+=extension/3600;ledger['GPU_HOURS_USED']+=extension/3600;e['final_release_probe']=final_probe
        if final_probe['status']=='PASS' and 'pid' in e and not any(a['pid']==e['pid'] for a in final_probe['apps']):e.update(released=True,release_confirmed_late=True,release_observation_unix=time.time(),reservation_duration_is_lower_bound_if_release_unverified=False)
    own=[e['pid'] for e in ledger['phases'] if 'pid' in e];release_verified=final_probe['status']=='PASS' and not any(a['pid'] in own for a in final_probe['apps'])
    try:
        final_original=verify_original(cfg,old);C.require(final_original==old_snapshot,'Original sealed STOP evidence changed during continuation')
    except BaseException as exc:
        final_original=dict(status='FAIL',error=repr(exc));final_status='FAIL_AND_STOP'
        if failure is None:
            failure=dict(status=final_status,phase=phase,error='Original seal final verification failed: '+repr(exc),
                no_GPU_retry=True,no_later_GPU_work=True,continuation_id=CONTINUATION,original_FAIL_preserved=True)
            if not (new/'FIRST_FAILURE.json').exists():C.write('FIRST_FAILURE.json',failure,exclusive=True)
    C.require(ledger['phases'][:len(INHERITED_PHASES)]==inherited['phases'],'Inherited actual reservation objects were altered')
    ledger.update(final_status=final_status,final_GPU_release_verified=release_verified,final_GPU_release_probe=final_probe,ended_unix=time.time(),original_seal_post_run_verification=final_original)
    C.write('gpu_resource_ledger.json',ledger,exclusive=True)
    C.write('GATE_STATUS_FINAL.json',dict(task_id=TASK,status=final_status,gates=gates,coordinator_failure=failure,
        first_failure_file='FIRST_FAILURE.json' if (new/'FIRST_FAILURE.json').exists() else None,
        original_execution_status='FAIL_AND_STOP_PRESERVED',original_failure_path=str(old/'FIRST_FAILURE.json'),original_failure_sha256=cfg['original_bindings']['FIRST_FAILURE.json'],
        recovery_manifest_sha256=prepared['recovery_manifest_sha256'],continuation_id=CONTINUATION,
        GPU_HOURS_USED=ledger['GPU_HOURS_USED'],MAX_GPU_HOURS=24.0,GPU_release_verified=release_verified,GPU_release_query=final_probe,no_further_GPU_work=True),exclusive=True)
    emit(dict(status=final_status,GPU_HOURS_USED=ledger['GPU_HOURS_USED'],GPU_release_verified=release_verified,original_FAIL_preserved=True),ledger)
    return 0 if final_status=='COMPLETE_AND_STOP' else 1


def self_test():
    import tempfile
    from types import SimpleNamespace
    source={'phase':'science_doc1_FP_NATIVE','reservation_seconds':3600.0,'GPU_hours':1.0,'released':True}
    inherited=[dict(phase='canary_initial',GPU_hours=0.1),dict(phase='canary_fresh',GPU_hours=0.2),source]
    ledger=dict(GPU_HOURS_USED=INITIAL_GPUH,phases=copy.deepcopy(inherited))
    fake=SimpleNamespace(remaining_science_projection=lambda g,u,l:(10.0,'SYNTHETIC'))
    predicted,method=projection(fake,{},list(REMAINING),ledger)
    need(predicted==3600*1.25*5 and ledger['phases']==inherited,'Observed cost floor/inherited identity mock failed')
    need(tuple(REMAINING)==('science_doc1_H_C128','science_doc1_H_DUAL_M','science_doc2_FP_NATIVE','science_doc2_H_C128','science_doc2_H_DUAL_M'),'Remaining phases include a duplicate prefix')
    class BrokenStdout:
        def write(self,*args):raise BrokenPipeError('SYNTHETIC_ONLY')
        def flush(self):raise BrokenPipeError('SYNTHETIC_ONLY')
    prior=sys.stdout
    try:
        sys.stdout=BrokenStdout();emit(dict(phase='science_doc1_H_C128',status='PASS'),ledger)
    finally:sys.stdout=prior
    need(ledger['informational_stdout_errors'][0]['scientific_PASS_preserved'] is True,'Broken stdout incorrectly became a scientific failure')
    with tempfile.TemporaryDirectory(prefix='qwen_continuation_cpu_mock_') as td:
        root=Path(td);guard=root/'guard.json';new_json(guard,dict(mock=True))
        try:new_json(guard,dict(mock=False))
        except FileExistsError:pass
        else:raise AssertionError('Exclusive guard allowed duplicate run')
        original=root/'original';continued=root/'continued';original.mkdir();continued.mkdir();(original/'x').write_bytes(b'ORIGINAL');os.link(original/'x',continued/'x')
        need(logical_bytes(original)+logical_bytes(continued)==16,'Hardlinks must be counted twice under conservative total archive cap')
    return dict(status='PASS',scope='Pure CPU disposable/mock control tests; no SSH/model/GPU/science scores or actual run',
        inherited_reservations_unmodified=True,remaining_phases=len(REMAINING),carried_GPU_hours=INITIAL_GPUH,
        stdout_BrokenPipe_preserves_PASS=True,duplicate_guard_rejected=True,observed_actual_cost_floor=True,
        hardlinks_counted_twice_conservatively=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--initial-root');parser.add_argument('--root');group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--deploy-static',action='store_true');group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true');group.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),ensure_ascii=False))
    else:
        need(args.config,'Explicit continuation binding config is required')
        if args.deploy_static:print(json.dumps(deploy_static(args.config,args.initial_root,args.root),ensure_ascii=False))
        elif args.prepare:print(json.dumps(prepare(args.config,args.initial_root,args.root),ensure_ascii=False))
        else:sys.exit(run(args.config,args.initial_root,args.root))
