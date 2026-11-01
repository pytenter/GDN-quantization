"""CPU verification and result delivery; no additional GPU execution."""
import csv,hashlib,io,json,math,os,statistics,subprocess,time
from pathlib import Path
from cleanroom_common import D,W,TASK,read,sha,publish

def text(path,value):
 with Path(path).open('x') as f:f.write(value);f.flush();os.fsync(f.fileno())
def csv_export(path,rows):
 if not rows:return
 out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);text(path,out.getvalue())
UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'
TARGET_INDEX=6
EXPECTED_PRIOR_SAME_TASK_GPUH=1.007654354178602

def resource_budget_context(prereg,current):
 prior=prereg.get('prior_same_task_GPUh',prereg.get('prior_same_task_RTX3090_GPUh',0.0))
 assert type(prior) in (int,float) and math.isfinite(prior) and prior==EXPECTED_PRIOR_SAME_TASK_GPUH,'PRIOR_SAME_TASK_3090_GPUH_NOT_BOUND_TO_ACTUAL_PREVIOUS_V2'
 assert math.isfinite(current) and current>=0 and prior+current<=36.0,'SAME_TASK_3090_HARD_BUDGET_EXCEEDED'
 other=prereg.get('other_task_resources',{})
 separate=other.get('rtx4090_qualification',{}).get('cumulative_RTX4090_GPUh',prereg.get('independent_RTX4090_qualification_GPUh',3.4473476))
 return dict(prior_same_task_RTX3090_GPUh=prior,current_execution_RTX3090_GPUh=current,
  same_task_cumulative_RTX3090_GPUh=prior+current,remaining_same_task_RTX3090_GPUh=36.0-prior-current,
  MAX_SAME_TASK_RTX3090_GPUh=36.0,current_execution_cap_GPUh=36.0-prior,protective_cumulative_GPUh=35.8,independent_RTX4090_qualification_GPUh=separate,
  other_task_resources=other,independent_RTX4090_budget_scope='separate qualification-only task; reported for provenance, not charged against this3090 CleanRoom cap')

def optional(path,default=None):
 return read(path) if Path(path).is_file() else default


def progress_log_audit(jobs, exits):
    """Read real progress only; a failed worker's unlogged terminal cursor is unknown."""
    records=[]
    for path in sorted(Path(jobs).glob('*.log')):
        tag=path.name[:-4];receipt=exits.get(tag,{})
        launch=optional(path.with_name(tag+'.LAUNCH.json'),{})
        kind=receipt.get('kind',launch.get('kind','UNKNOWN'))
        progress=[];invalid=[]
        with path.open(errors='replace') as stream:
            for line_number,line in enumerate(stream,1):
                try:value=json.loads(line)
                except (ValueError,TypeError):continue
                if not isinstance(value,dict) or value.get('event') not in ['qualification_progress','formal_progress']:continue
                position=value.get('position')
                peaks=[value.get('peak_allocated'),value.get('peak_reserved')]
                if (type(position) is not int or position<0 or
                        any(x is not None and (type(x) not in (int,float) or not math.isfinite(x) or x<0) for x in peaks)):
                    invalid.append(line_number);continue
                progress.append(dict(line_number=line_number,**value))
        measured=lambda key:max((x[key] for x in progress if x.get(key) is not None),default=None)
        failed=receipt.get('status')!='PASS'
        records.append(dict(tag=tag,kind=kind,document=receipt.get('document',launch.get('document')),
            condition=receipt.get('condition',launch.get('condition')),
            worker_status=receipt.get('status','UNKNOWN_NO_EXIT_RECEIPT'),
            failed_or_incomplete=failed,source_log_path=str(path),source_log_sha256=sha(path),
            progress_row_count=len(progress),invalid_progress_line_numbers=invalid,
            last_recorded_progress=progress[-1] if progress else None,
            last_confirmed_consumed_tokens=max((x['position'] for x in progress),default=None),
            exact_terminal_cursor=None,
            exact_terminal_cursor_status='UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT',
            max_observed_peak_allocated_bytes=measured('peak_allocated'),
            max_observed_peak_reserved_bytes=measured('peak_reserved'),
            CUDA_scope='logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks',
            unlogged_tokens_not_inferred=True,science_qualification_claimed=False))
    return records

def cuda_observations(profiles, progress):
    allocated=[x['peak_allocated'] for x in profiles if x.get('peak_allocated') is not None]
    allocated += [x['max_observed_peak_allocated_bytes'] for x in progress if x.get('max_observed_peak_allocated_bytes') is not None]
    reserved=[x['peak_reserved'] for x in profiles if x.get('peak_reserved') is not None]
    reserved += [x['max_observed_peak_reserved_bytes'] for x in progress if x.get('max_observed_peak_reserved_bytes') is not None]
    maximum_allocated=max(allocated,default=None);maximum_reserved=max(reserved,default=None)
    return dict(CUDA_peak_records=profiles,CUDA_successful_PROFILE_records=profiles,
        CUDA_peak_progress_records=progress,
        CUDA_partial_failed_worker_peak_records=[x for x in progress if x['failed_or_incomplete']],
        max_observed_CUDA_peak_allocated_bytes=maximum_allocated,
        max_observed_CUDA_peak_reserved_bytes=maximum_reserved,
        CUDA_peak_allocated_bytes=maximum_allocated,CUDA_peak_reserved_bytes=maximum_reserved,
        CUDA_scope='maximum observed recorded counters across separately scoped verified successful or artifact-only PROFILEs and all actual progress logs; failed-worker observations are lower bounds through last logged position, not terminal or lifetime peaks; model loading/restore excluded after frozen-loop resets; canary has its separate pre-model-load reset; no full32K peak measured')

def resource_stop_evidence(first_stop, root, admission_assess_window=None):
    """Verify the first pinned failure and its exact fsynced JSONL record."""
    root=Path(root).resolve()
    if first_stop is None:return dict(status='NO_FIRST_STOP',first_STOP=None,is_resource_gate_STOP=False,artifacts=[],pinned_failures=[],observed_process_records=[],errors=[])
    linked={};missing=[];errors=[];pinned=[];observations=[];query_unknown=[];in_worker=[];admission=[]
    canonical=lambda value:json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if isinstance(item,str) and any(token in key.lower() for token in ('path','file','artifact','evidence','snapshot')) and item.endswith(('.json','.jsonl')):
                    candidate=Path(item)
                    if not candidate.is_absolute():candidate=root/candidate
                    candidate=candidate.resolve()
                    if not candidate.is_relative_to(root):continue
                    expected=value.get(key.replace('_path','_sha256').replace('_file','_sha256'))
                    if key in ('path','file','artifact_path'):expected=value.get('sha256',expected)
                    if candidate.is_file():linked[str(candidate)]=expected if isinstance(expected,str) and len(expected)==64 else None
                    else:missing.append(str(candidate))
                elif isinstance(item,(dict,list)):visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(first_stop)
    own=None
    tag=first_stop.get('tag')
    if tag:own=optional(root/'jobs'/(tag+'.LAUNCH.json'),{}).get('identity')
    def inspect(value,path,owner):
        if isinstance(value,dict):
            owner=value.get('owner',owner)
            is_unknown=value.get('no_successful_snapshot_available') is True or value.get('status')=='UNKNOWN_SAMPLING_EXCEPTION'
            if 'direct_NVML_processes' in value or ('direct_NVML' in value and 'target' in value) or is_unknown:
                groups=value.get('direct_NVML_processes',{})
                compute=groups.get('compute',{});graphics=groups.get('graphics',{})
                smi=value.get('SMI_all_processes',{})
                early=value.get('early_SMI_compute_processes')
                early_present='early_SMI_compute_processes' in value
                early_valid=(not early_present or (isinstance(early,dict) and early.get('status')=='PASS' and isinstance(early.get('processes'),list) and all(isinstance(p,dict) and p.get('uuid')==UUID and str(p.get('pid','')).isdigit() for p in early['processes'])))
                visibility=(not is_unknown and compute.get('status')=='PASS' and graphics.get('status')=='PASS' and smi.get('status')=='PASS' and early_valid)
                early_pids={int(p['pid']) for p in early['processes']} if early_present and early_valid else set()
                nvml_pids={p['pid'] for p in compute.get('processes',[]) if type(p.get('pid')) is int}
                smi_pids={int(p['pid']) for p in smi.get('processes',[]) if str(p.get('pid','')).isdigit() and p.get('type')!='G'}
                union=nvml_pids|smi_pids|early_pids
                owners={p['pid']:p for p in value.get('process_ownership',[]) if type(p.get('pid')) is int}
                owner_metadata=owners.get(owner.get('pid')) if isinstance(owner,dict) else None
                identity_status='N/A_NO_OWNER'
                if isinstance(owner,dict):
                    identity_status='UNKNOWN_METADATA_NOT_CAPTURED'
                    if owner_metadata and owner_metadata.get('status')=='PASS':
                        identity_status='MATCH_PID_UID_STARTTIME' if all(owner_metadata.get(k)==owner.get(k) for k in ['pid','uid','start_ticks']) else 'MISMATCH_PID_UID_OR_STARTTIME'
                own_bytes=[p['usedGpuMemory'] for p in compute.get('processes',[]) if p.get('pid')==owner.get('pid') and type(p.get('usedGpuMemory')) is int and 0<=p['usedGpuMemory']<2**64-1] if compute.get('status')=='PASS' and identity_status=='MATCH_PID_UID_STARTTIME' else None
                foreign=sorted(union-({owner['pid']} if isinstance(owner,dict) else set())) if visibility else None
                observations.append(dict(artifact_path=path,epoch=value.get('epoch',value.get('sampled_epoch')),uuid=value.get('uuid'),
                    observation_status='OBSERVED_COMPLETE_PROCESS_APIS' if visibility else 'UNKNOWN_QUERY_OR_PROCESS_VISIBILITY_FAILED',
                    NVML_compute_PIDs=sorted(nvml_pids) if compute.get('status')=='PASS' else None,
                    SMI_non_graphics_PIDs=sorted(smi_pids) if smi.get('status')=='PASS' else None,
                    early_SMI_compute_source_present=early_present,
                    early_SMI_compute_PIDs=(sorted(early_pids) if early_valid else None) if early_present else None,
                    early_SMI_compute_status='PASS_TARGET_UUID_PID_LIST' if early_present and early_valid else 'UNKNOWN_OR_INVALID_TARGET_UUID_PID_LIST' if early_present else 'NOT_CAPTURED_LEGACY_TWO_SOURCE_SNAPSHOT',
                    owned_NVML_usedGpuMemory_bytes=own_bytes,owned_process_snapshot_identity_status=identity_status,foreign_process_memory_excluded=True,
                    unauthorized_PIDs_under_unchanged_exact_owner_policy=foreign,
                    unauthorized_process_metadata=[owners.get(pid,dict(pid=pid,status='UNKNOWN_METADATA_NOT_CAPTURED')) for pid in foreign] if foreign is not None else None,
                    exact_owned_process_identity=owner,posthoc_data_used=False))
                if not visibility:query_unknown.append(path)
            for item in value.values():
                if isinstance(item,(dict,list)):inspect(item,path,owner)
        elif isinstance(value,list):
            for item in value:inspect(item,path,owner)
    artifacts=[];processed=set()
    while set(linked)-processed:
        name=sorted(set(linked)-processed)[0];processed.add(name)
        expected=linked[name];path=Path(name);actual=sha(path)
        if expected is not None and expected!=actual:errors.append(dict(stage='resource_evidence',reason='STOP_LINKED_ARTIFACT_SHA_MISMATCH',path=name,expected=expected,actual=actual))
        artifacts.append(dict(path=name,bytes=path.stat().st_size,sha256=actual,expected_SHA_from_STOP=expected))
        try:
            if path.suffix=='.jsonl':
                # The exact failed line is checked from the pinned byte offset.
                # Do not reinterpret every earlier successful line as the failure.
                continue
            payload=read(path);inspect(payload,name,own);visit(payload)
            if payload.get('schema')=='failed_resource_snapshot_v1':
                snapshot=payload['snapshot'];snapshot_sha=hashlib.sha256(canonical(snapshot)).hexdigest()
                assert snapshot_sha==payload['snapshot_sha256'],'PINNED_SNAPSHOT_CANONICAL_SHA_MISMATCH'
                reference=payload['raw_reference'];raw_path=Path(reference['path']).resolve()
                assert raw_path.is_relative_to(root) and raw_path.is_file(),'PINNED_RAW_REFERENCE_OUTSIDE_EXECUTION_OR_MISSING'
                with raw_path.open('rb') as handle:
                    handle.seek(reference['offset']);line=handle.read(reference['bytes'])
                assert len(line)==reference['bytes'] and hashlib.sha256(line).hexdigest()==reference['line_sha256'],'PINNED_RAW_LINE_SHA_MISMATCH'
                raw_record=json.loads(line)
                assert raw_record['snapshot_sha256']==snapshot_sha and hashlib.sha256(canonical(raw_record['snapshot'])).hexdigest()==snapshot_sha,'PINNED_AND_RAW_SNAPSHOT_CONFLICT'
                assert raw_record['uuid']==payload['uuid']==UUID and raw_record['owner']==payload['owner'],'PINNED_RAW_UUID_OR_OWNER_CONFLICT'
                classification=payload.get('classification',{})
                forbidden=classification.get('forbidden_pids')
                assert forbidden==payload.get('forbidden_pids'),'PINNED_CLASSIFICATION_FORBIDDEN_PID_CONFLICT'
                compute=snapshot.get('direct_NVML_processes',{}).get('compute',{})
                smi=snapshot.get('SMI_all_processes',{})
                early=snapshot.get('early_SMI_compute_processes')
                early_present='early_SMI_compute_processes' in snapshot
                early_valid=(not early_present or (isinstance(early,dict) and early.get('status')=='PASS' and isinstance(early.get('processes'),list) and all(isinstance(x,dict) and x.get('uuid')==UUID and str(x.get('pid','')).isdigit() for x in early['processes'])))
                if compute.get('status')=='PASS' and smi.get('status')=='PASS' and early_valid:
                    union={int(x['pid']) for x in compute.get('processes',[])}
                    union.update(int(x['pid']) for x in smi.get('processes',[]) if x['type']!='G')
                    if early_present:union.update(int(x['pid']) for x in early['processes'])
                    expected_forbidden=sorted(union-({payload['owner']['pid']} if isinstance(payload['owner'],dict) else set()))
                    assert classification.get('compute_union')==sorted(union) and forbidden==expected_forbidden,'PINNED_CLASSIFICATION_DIFFERS_FROM_UNCHANGED_RAW_UNION'
                elif classification.get('status')=='COMPLETE_EXACT_COMPUTE_UNION':
                    raise AssertionError('QUERY_UNKNOWN_CANNOT_HAVE_COMPLETE_COMPUTE_CLASSIFICATION')
                if 'forbiddenPIDs' in first_stop:assert first_stop['forbiddenPIDs']==forbidden,'STOP_AND_PINNED_FORBIDDEN_PID_CONFLICT'
                if own is not None:assert payload['owner']==own,'STOP_PINNED_OWNER_DIFFERS_FROM_LAUNCH'
                pinned.append(dict(path=name,sha256=actual,status=payload.get('status'),snapshot_sha256=snapshot_sha,
                    raw_reference=reference,raw_line_and_snapshot_SHA_verified=True,
                    original_error=payload.get('original_error'),diagnostic_classification=classification,
                    forbidden_pids=forbidden,owner=payload.get('owner'),
                    ancestry_or_same_UID_does_not_authorize=True))
        except Exception as e:
            errors.append(dict(stage='resource_evidence',reason='STOP_LINKED_ARTIFACT_OR_RAW_BINDING_VERIFICATION_FAILED',path=name,error=repr(e)))
    normalize=lambda text:str(text).replace('\\','')
    reason=normalize(first_stop.get('reason',''))
    stop_epoch=first_stop.get('epoch',first_stop.get('stopped_epoch'))
    stage=first_stop.get('stage');doc=first_stop.get('document');condition=first_stop.get('condition')
    kind={'parent':'parent','resumed_parity':'fresh','fresh':'fresh','canary':'canary'}.get(stage)
    if stage in ['formal','formal1','formal2'] and doc and condition:
        candidates=[]
        for candidate_kind in ['formal1','formal2']:
            launch=optional(root/'jobs'/(str(doc)+'__'+str(condition)+'__'+candidate_kind+'.LAUNCH.json'),{})
            receipt=optional(root/'jobs'/(str(doc)+'__'+str(condition)+'__'+candidate_kind+'.EXIT.json'),{})
            if launch and receipt.get('status')!='PASS':candidates.append(candidate_kind)
        if len(candidates)==1:kind=candidates[0]
    # Frozen in-worker device() already fsyncs a complete JSONL record before
    # asserting. Its first failed line is usable only for the same unit/reason.
    if not pinned and kind and doc and condition:
        path=root/('canary/gpu6.GPU_EXCLUSIVITY.jsonl' if kind=='canary' else
            ('formal/'+str(doc)+'__'+str(condition)+'__formal.GPU_EXCLUSIVITY.jsonl' if kind in ['formal1','formal2'] else 'runs/'+str(doc)+'__'+str(condition)+'__'+('resumed_parity' if kind=='fresh' else kind)+'.GPU_EXCLUSIVITY.jsonl'))
        if path.resolve().is_relative_to(root) and path.is_file():
            owner=optional(root/'jobs'/(str(doc)+'__'+str(condition)+'__'+kind+'.LAUNCH.json'),{}).get('identity')
            offset=0;failed=None
            with path.open('rb') as stream:
                for line_number,line in enumerate(stream,1):
                    current_offset=offset;offset+=len(line)
                    try:record=json.loads(line)
                    except (ValueError,TypeError):continue
                    if isinstance(record,dict) and record.get('exclusive') is False:
                        failed=(line_number,current_offset,line,record);break
            if failed is not None:
                line_number,offset,line,record=failed
                conditions=record.get('failure_conditions',[])
                matched=any(normalize(x) and normalize(x) in reason for x in conditions)
                epoch=record.get('epoch')
                temporal=(type(stop_epoch) in (int,float) and type(epoch) in (int,float) and epoch<=stop_epoch)
                identity=(record.get('gpu_uuid')==UUID and record.get('gpu')==TARGET_INDEX and isinstance(owner,dict))
                if matched and temporal and identity:
                    snapshot=record.get('snapshot')
                    if isinstance(snapshot,dict):
                        inspect(record,str(path),owner)
                        binding=dict(source='FROZEN_IN_WORKER_GPU_EXCLUSIVITY_JSONL',path=str(path),sha256=sha(path),
                            line_number=line_number,offset=offset,bytes=len(line),line_sha256=hashlib.sha256(line).hexdigest(),
                            snapshot_sha256=hashlib.sha256(canonical(snapshot)).hexdigest(),
                            unit=dict(document=doc,condition=condition,stage=stage),owner=owner,
                            first_failed_line=True,reason_matches_first_STOP=True,record_epoch=epoch,STOP_epoch=stop_epoch,
                            failure_conditions=conditions,no_posthoc_substitution=True)
                        in_worker.append(binding)
                        artifacts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=binding['sha256'],source=binding['source']))
    # Admission has no worker yet. A unique write-once three-sample file without
    # its PASS receipt is replayed using the unchanged CPU-only window predicate.
    traceback_text=str(first_stop.get('traceback',''))
    admission_context=stage=='DEPENDENCY_FAIL_CLOSED' and any(x in traceback_text for x in ['g.admit(', 'in admit', 'in assess_window'])
    if not pinned and not in_worker and admission_context:
        candidates=[]
        for path in sorted((root/'admission').glob('*.json')):
            if path.name.endswith('.PASS.json') or path.with_name(path.stem+'.PASS.json').exists():continue
            try:
                value=read(path)
                if isinstance(value,dict) and isinstance(value.get('samples'),list) and len(value['samples'])==3:
                    candidates.append((path,value))
            except (ValueError,OSError):continue
        if len(candidates)==1:
            path,value=candidates[0];epoch=value.get('epoch')
            temporal=(type(stop_epoch) in (int,float) and type(epoch) in (int,float) and epoch<=stop_epoch)
            replay_error=None
            if temporal:
                if admission_assess_window is None:
                    from gpu_admission import assess_window as admission_assess_window
                try:admission_assess_window(value['samples'],UUID)
                except BaseException as e:replay_error=repr(e)
            if temporal and replay_error is not None and normalize(replay_error)==reason:
                inspect(value,str(path),None)
                binding=dict(source='UNIQUE_UNPASSED_FROZEN_ADMISSION_THREE_SAMPLE_FILE',path=str(path),sha256=sha(path),
                    sample_count=3,owner=None,no_worker_launched=True,original_CPU_window_predicate_reproduced_error=replay_error,
                    unique_unpassed_admission=True,reason_matches_first_STOP=True,record_epoch=epoch,STOP_epoch=stop_epoch,
                    no_posthoc_substitution=True)
                admission.append(binding);artifacts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=binding['sha256'],source=binding['source']))
            else:errors.append(dict(stage='resource_evidence',reason='UNPASSED_ADMISSION_DOES_NOT_REPRODUCE_FIRST_STOP',path=str(path),replayed_error=replay_error))
        else:errors.append(dict(stage='resource_evidence',reason='ADMISSION_FAILURE_ASSOCIATION_UNKNOWN',unpassed_three_sample_candidates=[str(p) for p,v in candidates]))
    text=str(first_stop.get('stage',''))+' '+str(first_stop.get('reason',''))
    needs_raw=(any(x in text for x in ['RUNNING_RESOURCE','RESOURCE_ADMISSION','UNAUTHORIZED_EXTERNAL_COMPUTE','PROCESS_VISIBILITY','INSUFFICIENT_FREE_MEMORY','NVML_','GRAPHICS_','MEMORY_ARITHMETIC','DEVICE_IDENTITY']) and 'BUDGET' not in text)
    if missing:errors.append(dict(stage='resource_evidence',reason='STOP_REFERENCED_EVIDENCE_ARTIFACT_MISSING',paths=sorted(set(missing))))
    if in_worker or admission:needs_raw=True
    if needs_raw and not (pinned or in_worker or admission):
        errors.append(dict(stage='resource_evidence',reason='RESOURCE_STOP_WITHOUT_VERIFIED_PINNED_RAW_SNAPSHOT',query_or_trigger_PID_identity='UNKNOWN'))
    if query_unknown:errors.append(dict(stage='resource_evidence',reason='RESOURCE_SNAPSHOT_QUERY_OR_VISIBILITY_UNKNOWN',paths=sorted(set(query_unknown)),missing_process_information_not_treated_as_empty=True))
    return dict(status='PASS_FIRST_STOP_EVIDENCE_BOUND' if not errors else 'INCOMPLETE_OR_FAILED_RESOURCE_EVIDENCE',
        first_STOP=first_stop,first_STOP_sha256=sha(root/'STOP.json') if (root/'STOP.json').is_file() else None,
        is_resource_gate_STOP=needs_raw,artifacts=artifacts,pinned_failures=pinned,in_worker_failure_bindings=in_worker,admission_failure_bindings=admission,observed_process_records=observations,
        exact_trigger_PID_or_ownership_not_captured='UNKNOWN' if not observations else 'SEE_LINKED_RAW_OBSERVATIONS_WITH_EXPLICIT_API_STATUS',
        current_or_posthoc_samples_substituted=False,errors=errors)

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']==''
 complete=(D/'COMPLETE.json').exists();assert complete!=(D/'STOP.json').exists()
 from gpu_admission import proc_identity
 receipts=[read(p) for p in sorted((D/'jobs').glob('*.EXIT.json'))]
 for launch in (D/'jobs').glob('*.LAUNCH.json'):
  launched=read(launch);receipt=optional(launch.with_name(launch.name.replace('.LAUNCH.json','.EXIT.json')),{});ident=launched.get('identity') or receipt.get('cleanup_identity')
  if ident is None:
   assert receipt.get('registration_failure') and receipt.get('wait_confirmed_child_exit') and receipt.get('returncode') is not None,'UNIDENTIFIED_CHILD_NOT_CONFIRMED_EXITED'
   live=False
  else:
   try:live=proc_identity(ident['pid'])==ident
   except (OSError,ValueError):live=False
  assert not live,'OWN_GPU_WORKER_STILL_ACTIVE'
  assert launch.with_name(launch.name.replace('.LAUNCH.json','.EXIT.json')).exists(),'EXIT_RECEIPT_MISSING'
 freeze=read(D/'SOURCE_CODE_FREEZE.json');assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
 for n,h in freeze['files'].items():assert sha(W/n)==h,n
 status_files={r['tag']:r for r in receipts}
 for r in receipts:
  assert r['wall_seconds']==r['completed_monotonic']-r['started_monotonic']
  if r['status']=='PASS':
   assert sha(r['result_path'])==r['result_sha256']
   assert sha(Path(r['result_path']).with_suffix('.PROFILE.json'))==r['profile_sha256']
 ledger=read(D/'gpu_resource_ledger.json');assert not ledger['active_workers']
 assert read(D/'preregistration.json')['prior_same_task_GPUh']+ledger['actual_new_GPU_hours']<=36
 positions=[8192+64*k for k in range(384)]
 spec=read(D/'FROZEN_SPEC.json');selection=read(D/'FROZEN_4DOC_SELECTION.json');routing=read(D/'EXECUTION_ROUTING.json')
 assert spec['primary']['positions']==positions and positions[-1]==32704
 qrows=[];FP=students=0;units=[];docrows=[];posrows=[];CP=[];resolution=(D/'resolution_contract.json').exists()
 for u in spec['new_units']:
  key=u['document']+'__'+u['rotation'];p=D/'qualification_units'/(key+'.json');cp=D/'runs'/(key+'__8192.pt')
  if p.exists():
   q=read(p)
   from cleanroom_common import compare_exact
   compare_exact(read(D/'runs'/(key+'__parent.json')),read(D/'runs'/(key+'__resumed_parity.json')),q['qualified_gpu_uuid'])
   assert sha(cp)==q['checkpoint_sha256'] and sha(D/'runs'/(key+'__parent.json'))==q['parent_sha256'] and sha(D/'runs'/(key+'__resumed_parity.json'))==q['resumed_sha256']
   qrows.append(q)
  else:qrows.append(dict(document=u['document'],rotation=u['rotation'],status='NOT_COMPLETED',INT8_QDQ='INCOMPLETE' if u['rotation']!='FP' else 'N/A_FP'))
  if cp.exists():CP.append(dict(document=u['document'],condition=u['rotation'],cursor=8192,path=str(cp),resolved_path=str(cp.resolve()),bytes=cp.stat().st_size,sha256=sha(cp),prefix_source='REUSED_VERIFIED_FP' if cp.is_symlink() else 'NEW_OWN_HISTORY'))
  long=D/'checkpoints'/(key+'__16384.pt')
  if long.exists():
   metadata=read(long.with_suffix('.json'));assert metadata['sha256']==sha(long)
   CP.append(dict(document=u['document'],condition=u['rotation'],cursor=16384,path=str(long),bytes=long.stat().st_size,sha256=sha(long),logical_reducer_and_native_state=True))
 import torch
 assert not torch.cuda.is_initialized()
 for doc in selection['documents']:
  name=doc['document'];valid={};token_path=Path(doc['document_identity']['token_file_path']);tokens=read(token_path)
  for condition in ['FP',doc['P1_predicted_rotation'],doc['L7_predicted_rotation']]:
   tag=name+'__'+condition+'__formal2';receipt=status_files.get(tag)
   if not receipt or receipt['status']!='PASS':continue
   j=read(receipt['result_path']);assert j['status']=='PASS' and j['all_finite'] and j['endpoint']==32768
   assert j['sample_positions']==positions and j['vocabulary_dimension']==157184
   assert j['token_sha256']==sha(token_path) and j['primary_spec_sha256']==sha(D/'FROZEN_SPEC.json')
   assert j['gpu_uuid']==routing['document_devices'][name]
   assert j['formal_observer_uninterrupted_parity_PASS'] and j['observer_qualified_before_first_primary_reduction']
   assert j['final_cache_inventory']['cache_bytes']==2033025024
   proof=read(D/'segments'/(name+'__'+condition+'__long_resume_exact.json'));assert proof['status']=='PASS_EXACT_64_LOGITS_NATIVE_CACHE_AUDIT_RNG'
   if condition=='FP':
    assert len([r for r in j['references'] if r['logits']])==384
    assert [r['position'] for r in j['references']]==sorted(set(positions)|{8192,16383,24575,32767})
    digest=hashlib.sha256()
    for r in j['references']:
     path=Path(r['path']);assert sha(path)==r['sha256'] and path.stat().st_size==r['bytes']
     payload=torch.load(path,map_location='cpu',weights_only=False);assert payload['position']==r['position'] and payload['token']==tokens[r['position']]
     if r['logits']:
      x=payload['logits'];assert tuple(x.shape)==(1,1,157184) and x.dtype==torch.float32 and bool(torch.isfinite(x).all())
      digest.update(r['position'].to_bytes(8,'little'));digest.update(x.contiguous().numpy().tobytes())
     del payload
    assert digest.hexdigest()==j['sampled_logits_content_sha256'];FP+=1
   else:
    samples=j['primary_samples'];assert len(samples)==384 and [v['position'] for v in samples]==positions
    assert all(math.isfinite(v['KL']) and v['vocabulary_dimension']==157184 for v in samples)
    assert sum(v['KL'] for v in samples)/384==j['primary_mean_KL']
    integrity=j['persistent_integrity'];assert integrity['passed'] and integrity['every_token_layer_qdq_count']==32768*18 and integrity['actual_next_kernel_equal_post_qdq_count']==32767*18
    students+=1
    for x in samples:posrows.append(dict(document=name,condition=condition,position=x['position'],KL=x['KL'],vocabulary_dimension=157184,result_sha256=sha(receipt['result_path'])))
   units.append(dict(document=name,condition=condition,status='COMPLETE',UUID=j['gpu_uuid'],result_path=receipt['result_path'],sha256=sha(receipt['result_path']),checkpoint_sha256=j['checkpoint_sha256'],long_resume_proof_sha256=sha(D/'segments'/(name+'__'+condition+'__long_resume_exact.json'))))
   valid[condition]=j
  p,l=doc['P1_predicted_rotation'],doc['L7_predicted_rotation'];base=dict(Doc=name,P1_Rotation=p,Original_L7_Rotation=l,K_P1=None,K_L7=None,Delta=None,Winner='NOT_RUN',status='NOT_RUN')
  if len(valid)==3:
   kp,kl=valid[p]['primary_mean_KL'],valid[l]['primary_mean_KL'];delta=kp-kl
   threshold=3.3300064494019496e-6
   winner='P1' if delta < -threshold else 'L7' if delta>threshold else 'NUMERICALLY_UNRESOLVED'
   base.update(K_P1=kp,K_L7=kl,Delta=delta,Winner=winner,status='COMPLETE',relative_delta_over_K_L7=delta/kl if kl!=0 else None,numerical_resolution=threshold,resolution_evidence='new exact replay, frozen profile and historical arithmetic threshold; no new32K error calibration')
  docrows.append(base)
 valid_docs=[r for r in docrows if r['status']=='COMPLETE'];P1=sum(r['Winner']=='P1' for r in valid_docs);L7=sum(r['Winner']=='L7' for r in valid_docs);ties=len(valid_docs)-P1-L7
 qcount=sum(r['status']=='PASS' for r in qrows);int8q=[r for r in qrows if r['rotation']!='FP']
 classification='P1_DIRECTIONALLY_SUPPORTED' if len(valid_docs)==4 and P1>=3 else 'P1_INCREMENTAL_VALUE_NOT_SUPPORTED' if len(valid_docs)==4 and L7>=3 else 'MIXED_OR_INCONCLUSIVE'
 if complete:assert qcount==12 and FP==4 and students==8 and len(valid_docs)==4 and len(posrows)==8*384
 evidence=resource_stop_evidence(optional(D/'STOP.json'),D)
 budget=resource_budget_context(read(D/'preregistration.json'),ledger['actual_new_GPU_hours'])
 totals=dict(TASK=TASK,EXECUTION_ID=D.name,MAX_CONCURRENT_GPUS=1,SCIENTIFIC_PROTOCOL_IDENTITY='PASS',ENGINEERING_IMPLEMENTATION='CLEANROOM_CONTROL_ONLY',GPU_RESOURCE_ADMISSION='PASS' if len([p for p in (D/'admission').glob('*.json') if not p.name.endswith('.PASS.json')])==len(list((D/'admission').glob('*.PASS.json'))) and (complete or not str(read(D/'STOP.json').get('stage','')).startswith('RUNNING_RESOURCE')) and not evidence['errors'] and not evidence['is_resource_gate_STOP'] else 'FAIL',FLA_NUMERICAL_PROFILE='PASS' if (D/'G1_CANARY_COMPLETE.json').exists() else 'INCOMPLETE',MODEL_AND_ROTATION_IDENTITY='PASS' if read(D/'audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json')['status']=='PASS_ACTUAL_BYTES' else 'FAIL',FRESH64_QUALIFICATION=f'{qcount}/12',INT8_PERSISTENT_QDQ_QUALIFICATION='PASS' if len(int8q)==8 and all(r['status']=='PASS' for r in int8q) else 'INCOMPLETE',FORMAL_FP_COMPLETE=f'{FP}/4',FORMAL_STUDENT_COMPLETE=f'{students}/8',ACTUAL_VALID_DOCUMENTS=f'{len(valid_docs)}/4',PRIMARY_384_SCORE_REPRODUCTION='PASS' if students else 'NOT_RUN',P1_WINS=P1,L7_WINS=L7,TIE_UNRESOLVED=ties,MISSING_DOCUMENTS=4-len(valid_docs),NEW_GPU_HOURS_USED=ledger['actual_new_GPU_hours'],PRIOR_SAME_TASK_RTX3090_GPUH=budget['prior_same_task_RTX3090_GPUh'],INDEPENDENT_RTX4090_QUALIFICATION_GPUH=budget['independent_RTX4090_qualification_GPUh'],MAX_GPU_HOURS=36.,FINAL_SCIENTIFIC_CLASSIFICATION=classification,FINAL_EXECUTION_STATUS='COMPLETE_AND_STOP' if complete else 'PARTIAL_STOP' if qcount or units else 'FAIL_CLOSED_STOP')
 deltas=[r['Delta'] for r in valid_docs]
 judgment=('本次没有完成四篇，尚不能回答P1增量预测价值；工程/资源停止不作为selector科学负例。' if len(valid_docs)<4 else '三篇或四篇支持P1，仅是有限文档方向性信号；应结合平均/中位Delta和每篇效应，不等于总体平均优势或泛化确认。' if P1>=3 else '三篇或四篇Original L7胜出，本次不支持当前P1 selector的稳定增量；建议暂停该版本推进。' if L7>=3 else '结果混合或数值未分辨，未建立稳定增量；不支持据此扩大selector训练，也不能证明P1普遍无效。')
 next_science=('若继续，先以独立更大预注册文档样本确认方向与效应稳定性。' if len(valid_docs)==4 and P1>=3 else '当前不自动开展训练或新实验；不足或混合的证据应先明确未回答问题，避免把工程失败或n4小样本当普遍结论。')
 summary=dict(machine_status=totals,first_STOP=optional(D/'STOP.json'),first_STOP_sha256=sha(D/'STOP.json') if (D/'STOP.json').exists() else None,STOP_resource_evidence=evidence,resource_budget=budget,documents=docrows,valid_n=len(valid_docs),mean_Delta=statistics.mean(deltas) if deltas else None,median_Delta=statistics.median(deltas) if deltas else None,exact_sign_test='NOT_REPORTED: document independence and sign-symmetry are not established; no confirmatory p-value',four_of_four_theoretical_one_sided_p_if_independent_symmetric_null=0.0625,old_two_document_MIXED='PERMANENTLY_PRESERVED_NOT_COMBINED_N6')
 publish(D/'final_scientific_results.json',summary);csv_export(D/'final_scientific_results.csv',valid_docs);csv_export(D/'positional_KL.csv',posrows)
 publish(D/'qualification_matrix.json',dict(qualified=qcount,planned=12,units=qrows))
 publish(D/'checkpoint_manifest.json',dict(checkpoints=CP,large_payloads_remote_only=True))
 publish(D/'run_manifest.json',dict(task=TASK,source_commit=freeze['source_commit'],formal_units=units,GPU_workers=receipts,planned_formal_trajectories=12,actual_GPU_workers=len(receipts),extra_overlap_tokens_per_formal_trajectory=64,unique_logical_consumed_tokens_per_complete_trajectory=32768,actual_formal_worker_calls_per_complete_trajectory=2))
 peaks=[];NVML=[];profile_records=[];artifact_profile_records=[]
 for r in receipts:
  if r.get('NVML_periodic_peak_owned_bytes') is not None:NVML.append(r['NVML_periodic_peak_owned_bytes'])
  if r['status']=='PASS':
   profile_path=Path(r['result_path']).with_suffix('.PROFILE.json');profile=read(profile_path)
   assert sha(profile_path)==r['profile_sha256'] and profile['status']=='PASS_ACTUAL_FROZEN_PROFILE'
   measured=profile['measured_CUDA_peak'];peaks.append(measured)
   profile_records.append(dict(tag=r['tag'],kind=r['kind'],profile_path=str(profile_path),profile_sha256=r['profile_sha256'],scope=profile.get('CUDA_scope','UNKNOWN_RESET_SCOPE'),**measured))
   j=read(r['result_path']);memory=j.get('memory',[])
   if isinstance(memory,dict):memory=[memory]
   peaks.extend(memory)
  elif r.get('artifact_status')=='PASS_ARTIFACT_BINDINGS_ONLY_RESOURCE_NOT_QUALIFIED':
   # Actual measured resource counters remain evidence; no failed receipt can
   # enter the unchanged qualification or formal scientific PASS loops above.
   assert r['status']!='PASS' and r['returncode']==0
   result_path=Path(r['result_path']);profile_path=result_path.with_suffix('.PROFILE.json');env_path=D/'jobs'/(r['tag']+'.ENV.json')
   assert result_path.resolve().is_relative_to(D.resolve()) and profile_path.resolve().is_relative_to(D.resolve())
   assert sha(result_path)==r['artifact_result_sha256'] and sha(profile_path)==r['artifact_profile_sha256'] and sha(env_path)==r['artifact_environment_sha256']
   profile=read(profile_path);artifact=read(result_path)
   assert profile['status']=='PASS_ACTUAL_FROZEN_PROFILE' and artifact['status']=='PASS' and profile['actual_environment']['LING_TARGET_GPU_UUID']==UUID
   measured=profile['measured_CUDA_peak']
   assert all(type(measured[k]) in (int,float) and math.isfinite(measured[k]) and measured[k]>=0 for k in ['peak_allocated','peak_reserved'])
   peaks.append(measured)
   artifact_profile_records.append(dict(tag=r['tag'],kind=r['kind'],receipt_status=r['status'],artifact_status=r['artifact_status'],profile_path=str(profile_path),profile_sha256=r['artifact_profile_sha256'],result_sha256=r['artifact_result_sha256'],resource_evidence_only=True,numerical_or_formal_qualification_counted=False,scope=profile.get('CUDA_scope','UNKNOWN_RESET_SCOPE'),**measured))
 progress=progress_log_audit(D/'jobs',status_files)
 evidence=resource_stop_evidence(optional(D/'STOP.json'),D)
 publish(D/'audits/TERMINAL_RESOURCE_EVIDENCE_AUDIT.json',evidence)
 publish(D/'audits/TERMINAL_PARTIAL_PROGRESS_AUDIT.json',dict(scope='actual completed/partial worker logs only; no terminal cursor inferred',workers=progress))
 for x in evidence.get('observed_process_records',[]):
  NVML.extend(x.get('owned_NVML_usedGpuMemory_bytes') or [])
 counters=cuda_observations(peaks,progress)
 budget=resource_budget_context(read(D/'preregistration.json'),ledger['actual_new_GPU_hours'])
 resource=dict(new_GPU_hours=ledger['actual_new_GPU_hours'],max_GPU_hours=36.,actual_GPU_worker_count=len(receipts),
  peak_measured_CUDA_allocated=counters['max_observed_CUDA_peak_allocated_bytes'],
  peak_measured_CUDA_reserved=counters['max_observed_CUDA_peak_reserved_bytes'],
  successful_PROFILE_records=profile_records,artifact_only_PROFILE_records=artifact_profile_records,partial_progress_records=progress,
  max_observed_CUDA_peak_allocated_bytes=counters['max_observed_CUDA_peak_allocated_bytes'],
  max_observed_CUDA_peak_reserved_bytes=counters['max_observed_CUDA_peak_reserved_bytes'],
  CUDA_scope=counters['CUDA_scope'],NVML_periodic_peak_owned_bytes=max(NVML) if NVML else None,
  NVML_scope='periodic exact-owned PID samples plus exact PID/UID/starttime owned-memory observations from the failed raw evidence; foreign/child/graphics memory excluded; not an instantaneous peak',
  STOP_resource_evidence=evidence,resource_evidence_errors=evidence['errors'],
  launch_to_exit_includes_load_restore_teardown=True,**budget)
 publish(D/'gpu_resource_summary.json',resource)
 table='|Doc|P1 Rotation|Original L7 Rotation|K_P1|K_L7|Delta|Winner|\n|---|---|---|---:|---:|---:|---|\n'
 for r in docrows:
  v=[r[k] for k in ['Doc','P1_Rotation','Original_L7_Rotation','K_P1','K_L7','Delta','Winner']]
  table+='|'+'|'.join('NOT_RUN' if x is None else format(x,'.17g') if isinstance(x,float) else str(x) for x in v)+'|\n'
 blocker=json.dumps(read(D/'STOP.json'),ensure_ascii=False,indent=2) if not complete else 'NONE'
 report=f'''# Ling/KDA 四文档 Clean-Room Final

独立TASK：{TASK}；本次单卡执行身份：{D.name}；原科学ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1。
终态：{totals['FINAL_EXECUTION_STATUS']}。本轮仅更换控制层，旧execution01至05及前次Clean-Room STOP全部保留，未覆盖STOP/模型/CP。本次由用户新明确授权仅GPU6串行运行，不自动增加后续尝试。

## 科学身份与执行

原prereg54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722，四篇freeze a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b，FROZEN_SPEC 8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1均实际SHA PASS。100/20/22/58输入流程与4篇×16=64冻结评分不变，无重排、重选或新训练。32份权重、raw/token、旋转及数学源码实际SHA见G0/source manifest。源码commit：{freeze['source_commit']}。

冻结数学worker文件字节保持；正式编译入口仅添加已登记的checkpoint/cursor/输出原子化控制节点，逐项可逆还原整个原始源码，未改model_step/QDQ/rotation/observer/KL/原Python sum384。旧执行05的双零13点规则不追溯改变；新任务使用3次间隔采样，≤2%差异只提示，前提是无外部compute/身份异常、显存稳定且按历史max(reserved,NVML峰值)+至少2GiB空余准入。实际资源安全仍独立监督，每个模型进程来源/profile/UUID验证。

每卡仅一次真实模型8-token canary，非科学轨迹。数值资格{qcount}/12；INT8资格{totals['INT8_PERSISTENT_QDQ_QUALIFICATION']}，FP audit=None为N/A。同一所选UUID上的既有FP8192 CP只在当前GPU fresh64合同实际PASS后复用，设备不同时重建其FP前缀；INT8各有自己的从token0 persistent-QDQ历史。FLA Conv BD8/W8、Gated RMSNorm BT16/W8固定，五个启动变量在新解释器前设置。8192原fresh64、observer/nativecache/RNG精确合同比较未放宽。

正式逻辑顺序41603→2544→7987→26183，每篇FP→P1→OriginalL7，全局串行。新增一次计划16384 checkpoint，原cache/RNG/QDQ state及完整已处理评分列表/原reducer状态原子保存。第一段继续64步收集未中断对照，第二段fresh恢复16384，完整64 logits/cache/audit/RNG必须exact，重放评分不重复计入384数组。每条完整轨迹唯一逻辑consumed tokens32768，计划恢复验证额外64token单列计时；两个正式segment不冒充两条科学样本。非预期进程退出无静默重试。

## 实际结果

{table}

原V157184，t=8192+64k (k0..383、末32704)，消费ids[t]预测ids[t+1]；原FP32 full-vocabulary KL(FP||student)，原Python sum/384。未完成项NOT_RUN，不填0、无空壳科学CSV。实际有效文档{len(valid_docs)}/4，FP{FP}/4、student{students}/8。P1胜{P1}、L7胜{L7}、数值未分辨{ties}、缺失{4-len(valid_docs)}。平均Delta={summary['mean_Delta']}，中位Delta={summary['median_Delta']}。每篇相对效应在真实JSON/CSV中，独立单位document，最多n4；不把384点/层/head伪重复，不合并旧2doc为n6。

原阈值3.3300064494019496e-6为历史经验分辨率。正式结果揭示前依据冻结数学/软件/实际FLA profile、所选GPU self-KL零和各条件fresh64精确重复性冻结适用证据；这不是新32K误差上界校准。阈值内标NUMERICALLY_UNRESOLVED。原始完整精度Delta始终保留。未计算确认性sign-test；独立/对称假设未建立，即使4/4同向在该假设下单侧理论p也为0.0625。

## 资源与文件

{json.dumps(resource,ensure_ascii=False,indent=2)}

预算硬上限36同任务3090新增GPUh；此前同任务3090实测1.007654354178602GPUh。独立4090资格任务已报告{budget["independent_RTX4090_qualification_GPUh"]!r}GPUh仅单列历史身份，不扣本任务3090额度。含canary、加载、parent、fresh64、正式两段和恢复的每worker launch-to-exit，仅GPU6按真实launch-to-exit累计；旧V2 GPUh计入同任务累计额度（本轮剩余34.992345645821395GPUh）。运行前和第一批真实parent后更新实际测量预测，保护性35.8GPUh提前停止留退出余量，没有提高36上限。所有实际CP和FP payload由manifest真实SHA绑定，大模型/CP只留服务器，不下载模型。仅一张GPU6，未杀其他用户/重置GPU。

服务器目录：{D}；工作树：{W}。source_manifest、engineering_change_manifest、resource_admission、qualification_matrix、run_manifest、gpu_resource_ledger和hashes保存真实记录。完整8×384逐点校验仅在实际完成时为PASS，部分状态详见机器字段与原始数组。

## 首失败及证据边界

失败partial worker的最后确认消费位置与counter下界见TERMINAL_PARTIAL_PROGRESS_AUDIT；未记录的实际终止cursor为UNKNOWN，不按耗时推断。资源摘要max_observed合并成功PROFILE和失败日志累计counter，不能冒充完整生命周期峰值。失败pin与精确raw字节offset/line SHA、原worker或唯一admission关联见TERMINAL_RESOURCE_EVIDENCE_AUDIT；查询失败/丢失保持UNKNOWN及error，不用posthoc替代。

```json
{blocker}
```

出现实质身份/数值/资源失败后保留首STOP及已合格资产，不扩大诊断，不修数学或改变容差，本次单卡授权以外不启动后续Clean-Room尝试。E_future/Hidden KL及新proxy不计算；FP32载体仅声明persistent-QDQ/writeback，不声称physical resident INT8 storage。

## 五项科研判断

1. P1是否在四篇预先冻结文档上更好？本轮实际分类 **{classification}**，胜负、完整Delta与缺失如上；不以工程通过代替科学答案。
2. 与原2doc MIXED 1:1的关系？旧MIXED和posthoc永久保留。新四篇是同一冻结问题的独立前瞻扩展，条件筛选且n小，不合并确认性n6。
3. 是否有足够增量值得继续？{judgment}
4. 没有稳定增量是否停止当前路线？若完整结果多为L7胜，应暂停当前P1 selector推进；若混合或不完整，尚不能作普遍无效结论，但没有继续训练授权。
5. 下一步科学验证？{next_science} 仅科研建议，未执行任何后续训练/新数据/旋转/L8/L9/STE/Cayley/AIME/free generation/Qwen。

```json
{json.dumps(totals,ensure_ascii=False,indent=2)}
```
'''
 text(D/'FINAL_SCIENTIFIC_REPORT.md',report)
 text(D/'FAILURE_HISTORY.md','# Failure history\n\nHistorical executions01-05 and previous Clean-Room STOP are read-only. This GPU6-only execution has new explicit human authorization. Current first STOP:\n\n```json\n'+blocker+'\n```\nNo further attempt beyond this explicitly authorized GPU6-only execution is launched.\n')
 files={}
 for p in sorted(D.rglob('*')):
  if not p.is_file() or p.suffix=='.pt' or p.name.endswith('.lock') or '.partial' in p.name or p.name in ['artifact_sha256.txt','DELIVERY_VERIFICATION.json'] or p.name=='controller.log':continue
  files[str(p.relative_to(D))]=dict(bytes=p.stat().st_size,sha256=sha(p))
 manifest=dict(status='COMPLETE' if complete else 'STOP_ACTUAL_PARTIAL',task=TASK,source_commit=freeze['source_commit'],files=files,large_payloads=dict(checkpoints=str(D/'checkpoint_manifest.json'),FP_manifests=[u['result_path'] for u in units if u['condition']=='FP']),self_exclusion='manifest and SHA list excluded from manifest; SHA list includes manifest itself')
 publish(D/'hashes/DELIVERY_VERIFICATION.json',manifest)
 text(D/'hashes/artifact_sha256.txt',''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items()))+sha(D/'hashes/DELIVERY_VERIFICATION.json')+'  hashes/DELIVERY_VERIFICATION.json\n')
 print(json.dumps(dict(status='CPU_DELIVERY_SEALED',final=totals,small_files=len(files))),flush=True)

if __name__=='__main__':main()
