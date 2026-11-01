"""Durable read-only resource evidence around the unchanged admission predicate.

Classification is diagnostic only. Only the original assess_fn decides acceptance;
children, same-UID processes and unknown identities are never auto-authorized.
"""
from pathlib import Path
import fcntl,hashlib,json,os,tempfile,time,traceback


def canonical_bytes(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def read_process_metadata(pid):
    result=dict(pid=int(pid),status='UNKNOWN')
    try:
        root=Path('/proc')/str(pid);stat=(root/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        status=(root/'status').read_text().splitlines()
        result.update(status='PASS',tgid=int(next(x for x in status if x.startswith('Tgid:')).split()[1]),uid=int(next(x for x in status if x.startswith('Uid:')).split()[1]),ppid=int(fields[1]),start_ticks=int(fields[19]),comm=(root/'comm').read_text().strip(),nspid=[int(x) for x in next(x for x in status if x.startswith('NSpid:')).split()[1:]],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        try:result['pid_namespace']=os.readlink(root/'ns/pid')
        except Exception as e:result['status']='UNKNOWN';result['pid_namespace']=None;result['pid_namespace_error']=repr(e)
    except Exception as e:result['error']=repr(e)
    return result


def classify_snapshot(snapshot,owner):
    union=set();parse_errors=[]
    try:
        for p in snapshot['direct_NVML_processes']['compute']['processes']:union.add(int(p['pid']))
    except Exception as e:parse_errors.append(dict(source='NVML_compute',error=repr(e)))
    try:
        for p in snapshot['SMI_all_processes']['processes']:
            if p['type']!='G':union.add(int(p['pid']))
    except Exception as e:parse_errors.append(dict(source='SMI_compute_union',error=repr(e)))
    # Additional diagnostic source emitted by the new sequential sampler.
    # Historical snapshots without this field keep their original two-source
    # classification; the live lifecycle guard requires the parsed third source.
    early=snapshot.get('early_SMI_compute_processes')
    if early is not None:
        try:
            assert early.get('status')=='PASS','EARLY_SMI_COMPUTE_UNKNOWN'
            for p in early['processes']:
                assert p['uuid']==snapshot['uuid'],'EARLY_SMI_UUID_CONFLICT'
                union.add(int(p['pid']))
        except Exception as e:parse_errors.append(dict(source='early_SMI_compute_union',error=repr(e)))
    owner_pid=owner.get('pid') if isinstance(owner,dict) else None
    allowed=set() if owner_pid is None else {int(owner_pid)}
    snapshot_metadata={}
    for p in snapshot.get('process_ownership',[]):
        try:snapshot_metadata[int(p['pid'])]=p
        except Exception as e:parse_errors.append(dict(source='snapshot_process_ownership',error=repr(e)))
    processes=[]
    for pid in sorted(union):
        live=read_process_metadata(pid);captured=snapshot_metadata.get(pid)
        if pid in allowed:label='EXPECTED_OWNER_PID_ONLY_ORIGINAL_ASSESS_CAN_VALIDATE_IDENTITY'
        elif live.get('status')!='PASS':label='UNSEEN_OR_UNKNOWN_NONOWNER_FORBIDDEN'
        elif owner_pid is not None and live.get('tgid')==owner_pid:label='THREAD_TID_OF_OWNER_FORBIDDEN'
        elif owner_pid is not None and live.get('ppid')==owner_pid:label='CHILD_OF_OWNER_FORBIDDEN'
        elif isinstance(owner,dict) and live.get('uid')==owner.get('uid'):label='SAME_UID_NONOWNER_FORBIDDEN'
        else:label='FOREIGN_NONOWNER_FORBIDDEN'
        processes.append(dict(pid=pid,classification=label,forbidden_by_exact_compute_union=pid not in allowed,snapshot_metadata=captured,live_metadata=live))
    return dict(status='COMPLETE_EXACT_COMPUTE_UNION' if not parse_errors else 'UNKNOWN_OR_PARTIAL_COMPUTE_UNION',compute_union=sorted(union),allowed_owner_pid=owner_pid,forbidden_pids=sorted(union-allowed),processes=processes,parse_errors=parse_errors,ancestry_or_same_UID_never_authorizes=True,acceptance_predicate_unchanged=True)


def append_raw_snapshot(snapshot,uuid,owner,baseline,raw_path):
    path=Path(raw_path);path.parent.mkdir(parents=True,exist_ok=True)
    snapshot_sha=hashlib.sha256(canonical_bytes(snapshot)).hexdigest()
    record=dict(schema='resource_raw_evidence_v1',uuid=uuid,epoch=time.time(),owner=owner,baseline=baseline,snapshot=snapshot,snapshot_sha256=snapshot_sha,controller_process=read_process_metadata(os.getpid()))
    raw=canonical_bytes(record)+b'\n'
    with path.open('ab') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        handle.seek(0,os.SEEK_END);offset=handle.tell();handle.write(raw);handle.flush();os.fsync(handle.fileno())
    return json.loads(raw),dict(path=str(path),offset=offset,bytes=len(raw),line_sha256=hashlib.sha256(raw).hexdigest())


def query_failures(snapshot):
    result=[]
    for key in ['physical','compute','SMI_XML']:
        row=snapshot.get(key)
        if isinstance(row,dict) and row.get('returncode')!=0:result.append(dict(source=key,raw=row))
    for key in ['direct_NVML','SMI_all_processes']:
        row=snapshot.get(key)
        if isinstance(row,dict) and row.get('status')!='PASS':result.append(dict(source=key,raw=row))
    early=snapshot.get('early_SMI_compute_processes')
    if early is not None and (not isinstance(early,dict) or early.get('status')!='PASS'):result.append(dict(source='early_SMI_compute_processes',raw=early))
    nvml=snapshot.get('direct_NVML',{})
    if isinstance(nvml,dict):
        for key in ['memory','memory_v2','utilization']:
            row=nvml.get(key)
            if isinstance(row,dict) and row.get('status') not in [None,'PASS']:result.append(dict(source='direct_NVML.'+key,raw=row))
    for key,row in snapshot.get('direct_NVML_processes',{}).items():
        if not isinstance(row,dict) or row.get('status')!='PASS':result.append(dict(source='direct_NVML_processes.'+str(key),raw=row))
    return result


def pin_failure(snapshot,uuid,owner,baseline,raw_reference,original_error,pinned_path,status='FAIL'):
    path=Path(pinned_path);path.parent.mkdir(parents=True,exist_ok=True)
    classification=classify_snapshot(snapshot,owner);query_errors=query_failures(snapshot)
    if query_errors or classification['parse_errors'] or snapshot.get('status')=='UNKNOWN_SAMPLING_EXCEPTION':status='UNKNOWN'
    record=dict(telemetry_query_errors=query_errors,schema='failed_resource_snapshot_v1',status=status,epoch=time.time(),uuid=uuid,snapshot=snapshot,snapshot_sha256=hashlib.sha256(canonical_bytes(snapshot)).hexdigest(),owner=owner,baseline=baseline,raw_reference=raw_reference,classification=classification,forbidden_pids=classification['forbidden_pids'],original_error=original_error,controller_process=read_process_metadata(os.getpid()),acceptance_predicate_unchanged=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.partial')
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(canonical_bytes(record)+b'\n');handle.flush();os.fsync(handle.fileno())
        try:os.link(tmp,path)
        except FileExistsError:pass # first pinned evidence is permanently retained
        finally:os.unlink(tmp)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    sealed=path.read_bytes();return json.loads(sealed),hashlib.sha256(sealed).hexdigest()


class ResourceEvidenceError(RuntimeError):
    def __init__(self,original_error,evidence_path,evidence_sha256,forbidden_pids):
        self.original_error=original_error;self.evidence_path=str(evidence_path);self.evidence_sha256=evidence_sha256;self.forbidden_pids=list(forbidden_pids)
        super().__init__('RESOURCE_ASSESSMENT_FAILED_WITH_PINNED_EVIDENCE: '+str(original_error))


def assess_with_evidence(snapshot,uuid,owner=None,baseline=None,*,raw_path,pinned_path,assess_fn=None):
    # This fsync completes before the existing predicate is invoked, even on FAIL.
    preserved,reference=append_raw_snapshot(snapshot,uuid,owner,baseline,raw_path)
    try:
        if assess_fn is None:
            from gpu_admission import assess as assess_fn
        return assess_fn(snapshot,uuid,owner,baseline)
    except BaseException as error:
        original=dict(type=type(error).__name__,repr=repr(error),traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)))
        pinned,digest=pin_failure(preserved['snapshot'],uuid,preserved['owner'],preserved['baseline'],reference,original,pinned_path)
        raise ResourceEvidenceError(original,str(pinned_path),digest,pinned['forbidden_pids']) from error


def record_sampling_failure(error,uuid,owner=None,baseline=None,*,raw_path,pinned_path):
    detail=dict(type=type(error).__name__,repr=repr(error),traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)))
    for name in ['stdout','stderr','returncode','cmd']:
        value=getattr(error,name,None)
        if value is not None:
            try:canonical_bytes(value);detail[name]=value
            except (TypeError,ValueError):detail[name]=repr(value)
    snapshot=dict(status='UNKNOWN_SAMPLING_EXCEPTION',uuid=uuid,error=detail,sampled_epoch=time.time(),no_successful_snapshot_available=True)
    _,reference=append_raw_snapshot(snapshot,uuid,owner,baseline,raw_path)
    pinned,digest=pin_failure(snapshot,uuid,owner,baseline,reference,detail,pinned_path,status='UNKNOWN')
    raise ResourceEvidenceError(detail,str(pinned_path),digest,pinned['forbidden_pids']) from error
