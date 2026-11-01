"""CPU-only lifecycle classification; no CUDA/model/scientific implementation."""
import csv,io,math

def early_compute_processes(snapshot,uuid):
 raw=snapshot.get('compute');assert isinstance(raw,dict) and raw.get('returncode')==0,'EARLY_SMI_COMPUTE_QUERY_UNKNOWN'
 text=raw.get('stdout');assert isinstance(text,str),'EARLY_SMI_COMPUTE_QUERY_UNKNOWN'
 rows=[]
 for row in csv.reader(io.StringIO(text)):
  if not row or all(not x.strip() for x in row):continue
  assert len(row)==3,'EARLY_SMI_COMPUTE_CSV_UNKNOWN'
  actual,pid,memory=(x.strip() for x in row)
  assert actual==uuid and pid.isdigit() and int(pid)>0,'EARLY_SMI_COMPUTE_IDENTITY_UNKNOWN'
  rows.append(dict(uuid=actual,pid=int(pid),used_gpu_memory=memory))
 return rows

def validate_timing_envelope(snapshot):
 values=snapshot.get('query_timings');assert isinstance(values,dict),'PER_QUERY_TIMING_MISSING'
 for key in ['physical','compute','fuser','direct_NVML','direct_NVML_processes','SMI_XML','process_ownership']:
  row=values.get(key);assert isinstance(row,dict),'PER_QUERY_TIMING_MISSING:'+key
  a,b=row.get('started_monotonic'),row.get('completed_monotonic')
  assert type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and 0<=a<=b,'PER_QUERY_TIMING_INVALID:'+key
 return True

def strict_snapshot_safety(snapshot,uuid,owner,baseline):
 import gpu_admission as g
 s=snapshot
 for key in ['physical','compute','SMI_XML']:
  assert isinstance(s.get(key),dict) and s[key].get('returncode')==0,'SAFETY_QUERY_UNKNOWN:'+key
 assert s.get('uuid')==uuid and len(s.get('target',[]))==1 and s['target'][0]['uuid']==uuid,'DEVICE_IDENTITY_FAILURE'
 t=s['target'][0];assert t['name']=='NVIDIA GeForce RTX 3090' and float(t['total_MiB'])>=24576,'UNAUTHORIZED_DEVICE_TYPE'
 n=s.get('direct_NVML');assert isinstance(n,dict) and n.get('status')=='PASS','NVML_UNAVAILABLE'
 for key in ['memory','memory_v2','utilization','temperature','power_mW','pstate','SM_clock_MHz']:assert key in n,'NVML_API_MISSING:'+key
 for key,value in n.items():
  if isinstance(value,dict) and key!='query_timings':assert value.get('status','PASS')=='PASS','NVML_API_UNKNOWN:'+key
 assert isinstance(n['memory'],dict) and all(type(n['memory'].get(k)) is int and n['memory'][k]>=0 for k in ['total','used','free']),'NVML_MEMORY_V1_UNKNOWN'
 for key in ['gpu','memory']:
  v=n['utilization'].get(key) if isinstance(n['utilization'],dict) else None
  assert type(v) in (int,float) and math.isfinite(v) and 0<=v<=100,'NVML_UTILIZATION_UNKNOWN'
 for key in ['temperature','power_mW','pstate','SM_clock_MHz']:
  v=n[key];assert type(v) in (int,float) and math.isfinite(v) and v>=0,'NVML_API_UNKNOWN:'+key
 for key in ['total_MiB','used_MiB','GPU_util']:
  v=float(t[key]);assert math.isfinite(v) and v>=0,'PHYSICAL_API_UNKNOWN:'+key
 m=n.get('memory_v2');assert isinstance(m,dict) and all(type(m.get(k)) is int and m[k]>=0 for k in ['total','free','used','reserved']),'NVML_MEMORY_UNKNOWN'
 assert m['total']==m['free']+m['used']+m['reserved'],'MEMORY_ARITHMETIC_FAILURE'
 assert m['free']>=g.HEADROOM,'INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM'
 groups=s.get('direct_NVML_processes');assert isinstance(groups,dict) and all(isinstance(groups.get(k),dict) and groups[k].get('status')=='PASS' and isinstance(groups[k].get('processes'),list) for k in ['compute','graphics']),'PROCESS_VISIBILITY_UNKNOWN'
 smi=s.get('SMI_all_processes');assert isinstance(smi,dict) and smi.get('status')=='PASS' and isinstance(smi.get('processes'),list),'PROCESS_VISIBILITY_UNKNOWN'
 graphics=g.graphics_identity(s)
 if baseline is not None:assert graphics==baseline,'GRAPHICS_IDENTITY_CHANGED'
 early=early_compute_processes(s,uuid)
 captured=s.get('early_SMI_compute_processes');assert isinstance(captured,dict) and captured.get('status')=='PASS' and captured.get('processes')==early,'EARLY_SMI_COMPUTE_CAPTURE_UNKNOWN_OR_CONFLICT'
 union={int(p['pid']) for p in groups['compute']['processes']}
 union.update(int(p['pid']) for p in smi['processes'] if p['type']!='G');union.update(p['pid'] for p in early)
 assert all(p>0 for p in union),'COMPUTE_PID_UNKNOWN'
 allowed=set() if owner is None else {owner['pid']};foreign=sorted(union-allowed)
 assert not foreign,('UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS_ALL_THREE_SOURCES',foreign)
 if owner is not None:
  for p in s.get('process_ownership',[]):
   if p.get('pid')==owner['pid']:
    assert p.get('status')=='PASS' and all(p.get(k)==owner.get(k) for k in ['pid','uid','start_ticks']),'OWN_CAPTURED_PROCESS_IDENTITY_UNKNOWN_OR_CHANGED'
 return dict(status='SAFETY_FIELDS_VERIFIED_NOT_RESOURCE_PASS',compute_union=sorted(union),early_SMI_compute_PIDs=sorted(p['pid'] for p in early),graphics=graphics)

def assess_lifecycle_sample(snapshot,uuid,owner,baseline,poll_before,poll_after,assess_fn=None):
 assert poll_before is None,'SAMPLE_NOT_STARTED_DURING_LIVE_CHILD'
 safety=strict_snapshot_safety(snapshot,uuid,owner,baseline)
 if poll_after is None:
  if assess_fn is None:
   from gpu_admission import assess as assess_fn
  value=assess_fn(snapshot,uuid,owner,baseline)
  return dict(lifecycle_state='RUNNING_RESOURCE_PASS',resource=value,safety=safety)
 assert type(poll_after) is int and poll_after==0,'NONZERO_CHILD_EXIT_DURING_SAMPLE'
 return dict(lifecycle_state='NATURAL_EXIT_TRANSITION_NOT_RESOURCE_PASS',resource=None,safety=safety)

def assess_terminal_window(rows,uuid,baseline,assess_window_fn=None):
 assert len(rows)==3,'THREE_SAMPLES_REQUIRED'
 for row in rows:strict_snapshot_safety(row,uuid,None,baseline)
 if assess_window_fn is None:
  from gpu_admission import assess_window as assess_window_fn
 value=assess_window_fn(rows,uuid)
 assert value['graphics_baseline']==baseline,'GRAPHICS_IDENTITY_CHANGED'
 return value

def validate_completion_assets(result,profile,environment_record,*,kind,document,condition,uuid,expected_environment,source_verified,source_commit=None):
 assert source_verified is True,'COMPLETION_SOURCE_IDENTITY_NOT_VERIFIED'
 assert isinstance(result,dict) and result.get('status')=='PASS','COMPLETION_RESULT_NOT_PASS'
 assert isinstance(profile,dict) and profile.get('status')=='PASS_ACTUAL_FROZEN_PROFILE','COMPLETION_PROFILE_NOT_PASS'
 assert isinstance(environment_record,dict) and environment_record.get('environment')==expected_environment,'COMPLETION_LAUNCH_ENVIRONMENT_MISMATCH'
 assert profile.get('actual_environment')==expected_environment and profile.get('hash_randomization')==0,'COMPLETION_PROFILE_ENVIRONMENT_MISMATCH'
 inner=profile.get('profile',{});assert inner.get('status')=='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE' and inner.get('forward_kernels_executed') is True,'COMPLETION_ACTUAL_FORWARD_PROFILE_MISSING'
 assert all(inner.get('environment',{}).get(k)==expected_environment[k] for k in ['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG','FLA_CACHE_MODE','FLA_CONFIG_DIR']),'COMPLETION_PROFILE_INPUT_ENVIRONMENT_MISMATCH'
 assert result.get('document')==document,'COMPLETION_DOCUMENT_MISMATCH'
 if kind=='canary':assert result.get('UUID')==uuid,'COMPLETION_UUID_MISMATCH'
 else:
  assert result.get('condition')==condition,'COMPLETION_CONDITION_MISMATCH'
  if kind=='formal1':
   identity=result.get('identity',{});assert identity.get('UUID')==uuid and identity.get('source_commit')==source_commit,'COMPLETION_UUID_OR_SOURCE_MISMATCH'
  elif kind=='formal2':assert result.get('gpu_uuid')==uuid,'COMPLETION_UUID_MISMATCH'
  else:assert result.get('host',{}).get('gpu_uuid')==uuid,'COMPLETION_UUID_MISMATCH'
 return dict(status='PASS_ARTIFACT_BINDING_ONLY_NOT_JOB_COMPLETE',kind=kind,document=document,condition=condition,uuid=uuid)
