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
import re
import stat
import sys
import time

TASK = 'QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1'
UUIDS = {1:'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
         2:'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',
         3:'GPU-30b07aba-a728-7649-4899-e09d33f8bc75'}
P1 = Path('/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022')
P1_FREEZE = '3955f03f2eaa56f83245a85ccad1c56d7d98b70eaa046154a1d95f5b79a7fa14'
_LOCKS = []
NATIVE_CLOSURE = 'analysis/RUNTIME_NATIVE_CLOSURE.json'
NATIVE_CLOSURE_STATUS = 'PASS_TWO_FRESH_CPU_IMPORTS_AND_PROVENANCED_NATIVE_CLOSURE_NO_GPU'
_NATIVE_AUDIT_SEQUENCE = 0

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
    checked = {}
    if fresh_external:
        for row in rows:
            digest=sha(row['path'])
            require(Path(row['path']).stat().st_size==row['bytes'] and digest==row['sha256'], 'External dependency changed: '+row['path'])
            checked[str(Path(row['path']).resolve())]=digest
    # The original61 contract remains unchanged. A separately frozen, fully
    # provenanced native closure adds exact file identities, including lazy
    # provider/profiler libraries; it does not replace or weaken those61 hashes.
    verify_native_closure(root, fresh=fresh_external, checked_hashes=checked)
    return rows


def _stat_record(value):
    return dict(bytes=value.st_size,inode=value.st_ino,device=value.st_dev,mtime_ns=value.st_mtime_ns)


def _stable_file(path, *, fresh=True, checked_hashes=None):
    path=Path(path)
    before=path.stat()
    require(stat.S_ISREG(before.st_mode),'Native closure member is not a regular file: '+str(path))
    with path.open('rb') as stream:
        magic=stream.read(4)
    require(magic==b'\x7fELF','Native closure member is not an ELF file: '+str(path))
    digest=None
    if fresh:
        digest=(checked_hashes or {}).get(str(path))
        if digest is None:
            digest=sha(path)
    after=path.stat()
    require(_stat_record(before)==_stat_record(after),'Native file changed during observation: '+str(path))
    return dict(path=str(path),**_stat_record(after),sha256=digest,ELF_magic=magic.hex())


def load_native_closure(root):
    root=Path(root).resolve()
    path=root/NATIVE_CLOSURE
    require(path.is_file() and not path.is_symlink(),'Missing ordinary pre-frozen native closure manifest')
    manifest=read(path)
    require(manifest.get('schema_version')==1 and manifest.get('status')==NATIVE_CLOSURE_STATUS,
            'Native closure is not a two-fresh-CPU provenanced pre-compute contract')
    require(manifest.get('base_external_bindings_sha256')==sha(root/'analysis/EXTERNAL_BINDINGS.json'),
            'Native closure no longer binds the unchanged original61 external contract')
    files=manifest.get('files')
    require(isinstance(files,list) and files and len(files)==len({row['path'] for row in files}),
            'Native closure exact file roster is empty or duplicated')
    for row in files:
        p=Path(row['path'])
        require(p.is_absolute() and str(p.resolve())==str(p) and not p.is_symlink(),
                'Native closure must use exact canonical file paths: '+str(p))
        require(all(type(row.get(name)) is int and row[name]>=0 for name in ('bytes','inode','device','mtime_ns'))
                and row['inode']>0 and isinstance(row.get('sha256'),str)
                and len(row['sha256'])==64 and all(x in '0123456789abcdef' for x in row['sha256'])
                and isinstance(row.get('provenance'),(dict,list)) and bool(row['provenance']),
                'Native closure lacks a pinned stat/SHA/provenance row: '+str(p))
    by_path={row['path']:row for row in files}
    baseline=read(root/'analysis/EXTERNAL_BINDINGS.json')['files']
    require(len(baseline)==61,'Original native closure base must remain exactly61 files')
    for row in baseline:
        p=Path(row['path']).resolve()
        with p.open('rb') as stream:magic=stream.read(4)
        if magic==b'\x7fELF':
            require(str(p) in by_path and by_path[str(p)]['bytes']==row['bytes']
                    and by_path[str(p)]['sha256']==row['sha256'],
                    'Complete closure must also pin every original61 native ELF: '+str(p))
    probes=manifest.get('cpu_probe_receipts')
    require(isinstance(probes,list) and len(probes)==2 and len({row['path'] for row in probes})==2,
            'Native closure requires two distinct fresh CPU probe receipts')
    for row in probes:
        rel=Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts,'CPU closure receipt escaped new root')
        probe=(root/rel).resolve()
        require(probe.is_relative_to(root) and not probe.is_symlink() and probe.is_file()
                and probe.stat().st_size==row['bytes'] and sha(probe)==row['sha256'],
                'Bound fresh CPU native-closure receipt changed: '+str(rel))
    return manifest


def verify_native_closure(root, *, fresh=True, checked_hashes=None):
    manifest=load_native_closure(root)
    verified=[]
    for row in manifest['files']:
        actual=_stable_file(row['path'],fresh=fresh,checked_hashes=checked_hashes)
        require(all(actual[name]==row[name] for name in ('bytes','inode','device','mtime_ns')),
                'Native closure stat identity changed: '+row['path'])
        if fresh:
            require(actual['sha256']==row['sha256'],'Native closure fresh SHA changed: '+row['path'])
        verified.append(actual)
    return dict(status='PASS_EXACT_NATIVE_CLOSURE_STAT_AND_FRESH_SHA' if fresh else 'PASS_EXACT_NATIVE_CLOSURE_STAT',
                manifest_sha256=sha(Path(root)/NATIVE_CLOSURE),count=len(verified),files=verified,
                original61_unchanged=True,full_fresh_SHA=fresh)


def parse_native_maps(text):
    """Parse complete Linux pathname fields, retaining deleted/space evidence.

No library-name keyword or directory-prefix allowlist exists here. File-backed
ELF mappings (plus unreadable executable/so candidates) are checked below;
anonymous kernel/JIT mappings are preserved as a separately stated scope.
"""
    rows=[]
    for line in text.splitlines():
        fields=line.split(None,5)
        if len(fields)<5:
            continue
        raw=fields[5] if len(fields)==6 else ''
        deleted=raw.endswith(' (deleted)')
        spelling=raw[:-10] if deleted else raw
        spelling=re.sub(r'\\([0-7]{3})',lambda match:chr(int(match.group(1),8)),spelling)
        rows.append(dict(raw=line,address=fields[0],permissions=fields[1],offset=fields[2],
                         mapped_device=fields[3],mapped_inode=int(fields[4]),raw_path=raw,
                         path=spelling if spelling.startswith('/') else None,deleted=deleted))
    return rows


def _inspect_native_maps(text, allowed=None):
    errors=[];observed=[]
    parsed=parse_native_maps(text)
    grouped={}
    for item in parsed:
        if item['path'] is not None:
            grouped.setdefault(item['path'],[]).append(item)
    for spelling,mappings in sorted(grouped.items()):
        path=Path(spelling);record=dict(path=spelling,canonical_path=str(path.resolve()),mappings=mappings)
        candidate=any('x' in row['permissions'] for row in mappings) or '.so' in path.name
        if not candidate:
            # Do not read tensor/weight mmap data, /dev/nvidia devices or other
            # non-executable data mappings while identifying native code.
            continue
        try:
            info=path.lstat();record['lstat']=dict(_stat_record(info),mode=info.st_mode,symlink=stat.S_ISLNK(info.st_mode))
            before=path.stat();record['stat']=dict(_stat_record(before),mode=before.st_mode)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('Executable/shared-library mapping is not an ordinary file')
            with path.open('rb') as stream:magic=stream.read(4)
            record['ELF_magic']=magic.hex()
            if magic!=b'\x7fELF':
                if candidate:
                    record['error']='Executable/shared-library candidate is not readable ELF'
                    errors.append(dict(path=spelling,error=record['error']))
                    observed.append(record)
                continue
            record['sha256']=sha(path)
            after=path.stat()
            failures=[]
            if _stat_record(before)!=_stat_record(after):failures.append('file changed during fresh hash')
            if any(row['deleted'] for row in mappings):failures.append('deleted mapping')
            for row in mappings:
                major,minor=(int(part,16) for part in row['mapped_device'].split(':'))
                if row['mapped_inode']!=after.st_ino or (major,minor)!=(os.major(after.st_dev),os.minor(after.st_dev)):
                    failures.append('mapped device/inode differs from current actual file')
            expected=None if allowed is None else allowed.get(record['canonical_path'])
            record['expected']=expected
            if allowed is not None and expected is None:
                failures.append('unknown exact native path outside original61 plus frozen closure')
            elif expected is not None:
                for name in ('bytes','sha256'):
                    actual=record['sha256'] if name=='sha256' else after.st_size
                    if actual!=expected[name]:failures.append('actual '+name+' differs from frozen file')
                for name in ('inode','device','mtime_ns'):
                    if name in expected and _stat_record(after)[name]!=expected[name]:failures.append('actual '+name+' differs from frozen file')
            record['verdict']=('OBSERVED_ONLY_NO_ADMISSION' if allowed is None else 'PASS') if not failures else 'FAIL'
            record['errors']=failures
            errors.extend(dict(path=spelling,canonical_path=record['canonical_path'],error=error) for error in failures)
            observed.append(record)
        except (OSError,ValueError) as error:
            if candidate or any(row['deleted'] for row in mappings):
                record['error']=repr(error);record['verdict']='FAIL'
                errors.append(dict(path=spelling,canonical_path=record['canonical_path'],error=repr(error)))
                observed.append(record)
    status=('OBSERVATION_ONLY_NO_NATIVE_ADMISSION' if allowed is None else 'PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED')
    return dict(status=status if not errors else 'FAIL_NATIVE_MAPPING_IDENTITY',
                mapped_native=observed,errors=errors,raw_maps=text,parsed_mapping_count=len(parsed),
                anonymous_or_kernel_executable_mappings=[row for row in parsed if row['path'] is None and 'x' in row['permissions']],
                scope='Every file-backed mapped native ELF; no basename keyword or directory-prefix exemption. Anonymous/kernel code is recorded but not claimed byte-hashed.',
                exact_allowlist_enforced=allowed is not None)


def capture_native_map_evidence(text):
    """Observation only for an authorized load-only engineering probe.

This never returns an admission PASS and changes no manifest. It can be called
offline on saved maps for complete path/lstat/device/inode/freshSHA evidence.
No model, CUDA, Torch or native library is imported/loaded by this function.
"""
    return _inspect_native_maps(text)


def assess_native_maps(root, text):
    """Return full native evidence/errors before caller durably enforces STOP."""
    root=Path(root).resolve()
    baseline=read(root/'analysis/EXTERNAL_BINDINGS.json')['files']
    require(len(baseline)==61,'Expected unchanged original61 external dependencies')
    closure=load_native_closure(root)
    allowed={str(Path(row['path']).resolve()):dict(row,origin='original61') for row in baseline}
    for row in closure['files']:
        old=allowed.get(row['path'])
        if old is not None:
            require(old['bytes']==row['bytes'] and old['sha256']==row['sha256'],
                    'Native closure contradicts the original61 contract: '+row['path'])
        allowed[row['path']]=dict(row,origin='separately_frozen_native_closure')
    record=_inspect_native_maps(text,allowed)
    record.update(manifest_sha256=sha(root/NATIVE_CLOSURE),
                  base_external_bindings_sha256=sha(root/'analysis/EXTERNAL_BINDINGS.json'),original61_preserved=True)
    return record


def persist_native_audit(root, record):
    global _NATIVE_AUDIT_SEQUENCE
    root=Path(root).resolve();directory=root/'operations/native_library_audit'
    current=root
    for part in Path('operations/native_library_audit').parts:
        current=current/part
        if os.path.lexists(current):
            require(current.is_dir() and not current.is_symlink(),'Native evidence directory is not ordinary')
        else:current.mkdir()
    _NATIVE_AUDIT_SEQUENCE+=1
    writer=dict(pid=os.getpid(),starttick=None,cmdline=list(sys.argv),identity_source='CURRENT_CPU_HOST_NO_PROC_IDENTITY')
    try:
        raw=Path('/proc/self/stat').read_text()
        writer.update(starttick=int(raw[raw.rfind(')')+2:].split()[19]),
                      cmdline=Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),
                      identity_source='ACTUAL_PROC_PID_STARTTICK_ARGV')
    except (OSError,ValueError,UnicodeError) as error:
        # macOS CPU fixtures have no /proc. If a real identity observation
        # fails, preserve that error and still save the primary native fault.
        writer['identity_error']=repr(error)
    observed_ns=time.monotonic_ns()
    name='NATIVE_MAPS_PID%d_T%s_N%d_%03d.json'%(os.getpid(),str(writer['starttick']),observed_ns,_NATIVE_AUDIT_SEQUENCE)
    path=directory/name
    document=dict(record,writer_identity=writer,writer_monotonic_ns=observed_ns)
    data=json.dumps(document,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
    with path.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    directory_fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    return dict(path=str(path.relative_to(root)),bytes=len(data),sha256=sha(path))

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
    # Keep the exact installed CUPTI ELF alive before cuInit. The opposite
    # order reproducibly segfaults in CUPTI during libcudart exit callbacks.
    # This ordinary ELF load calls no CUDA/CUPTI API and does not import Torch.
    from native_profiler_lifecycle import preload_cupti_before_driver
    native_initialization=preload_cupti_before_driver(root)
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
    receipt=ns[fn.name](rows)
    receipt['native_profiler_initialization']=native_initialization
    return receipt

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
    root=Path(root).resolve();require(pid==os.getpid(),'Only the actual current worker maps can be audited')
    raw=Path('/proc/self/maps').read_text();record=None;observation=None;failures=[]
    try:record=assess_native_maps(root,raw)
    except BaseException as error:
        try:record=capture_native_map_evidence(raw)
        except BaseException as capture_error:
            record=dict(raw_maps=raw,mapped_native=[],errors=[dict(scope='observation_capture',error=repr(capture_error))])
        record.update(status='FAIL_NATIVE_MANIFEST_OR_MAPPING_AUDIT',error=repr(error))
        record['errors'].append(dict(scope='frozen_manifest_or_audit',error=repr(error)))
    try:verify_model_and_external(root,fresh_external=False)
    except BaseException as error:failures.append(dict(scope='model_external_native_stat',error=repr(error)))
    try:observation=_own_gpu(root,gpu,pid)
    except BaseException as error:failures.append(dict(scope='actual_own_NVML',error=repr(error)))
    record.update(pid=pid,gpu=gpu,gpu_uuid=UUIDS[gpu],monotonic=time.monotonic(),NVML=observation,additional_failures=failures)
    passed=record['status']=='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED' and not failures
    record['guard_status']='PASS' if passed else 'FAIL_CLOSED_STOP'
    evidence=persist_native_audit(root,record)
    require(passed,'Actual native runtime identity failed; full path/stat/inode/freshSHA evidence durably retained at '+evidence['path'])
    from profiler_detach_lifecycle import detach_qualification_profile_if_completed
    profiler_lifecycle = detach_qualification_profile_if_completed(root, gpu, pid)
    return dict(status='PASS_OWN_UUID_MODEL_STAT_AND_MAPPED_LIBRARY_IDENTITY',NVML=observation,
                mapped_controlled=sorted({row['canonical_path'] for row in record['mapped_native']}),
                exact_native_closure=True,native_audit=evidence,original61_unchanged=True,
                profiler_native_lifecycle=profiler_lifecycle)
