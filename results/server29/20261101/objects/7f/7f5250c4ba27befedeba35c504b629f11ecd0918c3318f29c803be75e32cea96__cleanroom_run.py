"""One bounded Clean-Room controller: canary, 12 qualifications, serial science."""
import concurrent.futures,copy,ctypes,fcntl,json,math,os,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
from cleanroom_common import D,W,PYTHON,read,publish,sha,stop,event,environment,compare_exact
import gpu_admission as g
from gpu_admission import sample
from resource_evidence import assess_with_evidence,record_sampling_failure,ResourceEvidenceError
from lifecycle_guard import assess_lifecycle_sample,assess_terminal_window,strict_snapshot_safety,validate_timing_envelope,validate_completion_assets
TASK='LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1'


TARGET_UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'
TARGET_INDEX=6
def validate_single_device_routing(routing):
 assert routing['authorized_devices']=={TARGET_UUID:TARGET_INDEX},'GPU6_ONLY_AUTHORIZATION'
 expected={'pg19-test-41603','pg19-test-2544','pg19-test-7987','pg19-test-26183'}
 assert set(routing['document_devices'])==expected and set(routing['document_devices'].values())=={TARGET_UUID},'ALL_DOCUMENTS_MUST_USE_GPU6'
 assert routing['formal_serial'] is True and routing['qualification_parallel_lanes']==1,'SINGLE_DEVICE_SERIAL_REQUIRED'
 return [TARGET_UUID]

def _child_parent_death():
 # Linux-only safety: a lost controller cannot leave an unbounded orphan worker.
 expected_parent=os.getppid();lib=ctypes.CDLL(None,use_errno=True)
 if lib.prctl(1,signal.SIGTERM,0,0,0)!=0:os._exit(120)
 if expected_parent<=1 or os.getppid()!=expected_parent:os._exit(121)

def _actual_own_child_identity(process,expected=None):
 # Independent /proc fallback for a failed metadata/registration call. Popen's
 # unreaped direct child cannot have its PID reused before this process waits.
 assert process.poll() is None,'CHILD_ALREADY_EXITED'
 root=Path('/proc')/str(process.pid)
 fields=(root/'stat').read_text().rsplit(') ',1)[1].split()
 status=(root/'status').read_text().splitlines()
 value=dict(pid=process.pid,start_ticks=int(fields[19]),uid=int(next(x for x in status if x.startswith('Uid:')).split()[1]),nspid=[int(x) for x in next(x for x in status if x.startswith('NSpid:')).split()[1:]],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 assert value['uid']==os.getuid() and int(fields[1])==os.getpid(),'NOT_OWN_UNREAPED_DIRECT_CHILD'
 if expected is not None:assert value==expected,'CHILD_IDENTITY_CHANGED_DURING_REGISTRATION'
 return value

def _exclusive_emergency(path,value):
 # Used only after a failed ordinary publication, never overwrites evidence.
 path=Path(path)
 try:
  with path.open('x') as handle:
   json.dump(value,handle,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
 except FileExistsError:pass

def verify_launch_registration(freeze=None):
 # Small registered controls and CPU evidence only; model weights are not rehashed.
 freeze=read(D/'SOURCE_CODE_FREEZE.json') if freeze is None else freeze
 assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==freeze['source_commit'],'SOURCE_COMMIT_CHANGED'
 for name,digest in freeze['files'].items():assert sha(W/name)==digest,('FROZEN_SOURCE_CHANGED',name)
 assert {'cleanroom_run.py','resource_evidence.py','lifecycle_guard.py','gpu_telemetry.py'}<=set(freeze['files']),'RESOURCE_CONTROL_SOURCE_NOT_FROZEN'
 for name,key in [('CALIBRATION_CONFIG.json','config_sha256'),('FROZEN_SPEC.json','scientific_spec_sha256'),('EXECUTION_ROUTING.json','routing_sha256'),('preregistration.json','preregistration_sha256')]:
  assert sha(D/name)==freeze[key],('FROZEN_CONTROL_IDENTITY_CHANGED',name)
 cpu_path=D/freeze['CPU_test_path'];assert cpu_path.resolve().is_relative_to(D.resolve())
 assert sha(cpu_path)==freeze['CPU_test_sha256'],'CPU_CONTROL_REGRESSION_CHANGED'
 assert read(cpu_path)['status'] in ['PASS','PASS_CPU_CONTROL_ONLY'],'CPU_CONTROL_REGRESSION_NOT_PASS'
 resource_path=D/'audits/CPU_CONTROLLER_RESOURCE_REGRESSION.json'
 assert sha(resource_path)==freeze['controller_resource_cpu_sha256'],'CONTROLLER_RESOURCE_CPU_REGRESSION_CHANGED'
 assert read(resource_path)['status']=='PASS','CONTROLLER_RESOURCE_CPU_REGRESSION_NOT_PASS'
 prereg=read(D/'preregistration.json')
 assert prereg['FULL_32K_EXECUTION_AUTHORIZED'] is True and prereg['MAX_CONCURRENT_GPUS']==1 and prereg['MAX_TOTAL_NEW_GPUH']==36.0,'FROZEN_SINGLE_GPU_FULL32K_BUDGET_CHANGED'
 return freeze

def protected(fn,*args):
 try:return fn(*args)
 except BaseException as e:
  if isinstance(e,ResourceEvidenceError):stop(repr(e),stage='RESOURCE_ADMISSION',evidence_path=e.evidence_path,evidence_sha256=e.evidence_sha256,forbiddenPIDs=e.forbidden_pids,original_resource_error=e.original_error)
  else:stop(repr(e),stage='DEPENDENCY_FAIL_CLOSED',traceback=traceback.format_exc())
  raise

class Controller:
 def __init__(self):
  self.lock=threading.RLock();self.active={};self.completed=[];self.routing=read(D/'EXECUTION_ROUTING.json')
  self.prior_GPUh=read(D/'preregistration.json')['prior_same_task_GPUh'];assert type(self.prior_GPUh) in (int,float) and math.isfinite(self.prior_GPUh) and 0<=self.prior_GPUh<35.8,'PRIOR_CUMULATIVE_GPUH_INVALID'
 def ledger(self):
  now=time.time();done=sum(r['wall_seconds'] for r in self.completed);live=sum(time.monotonic()-r['started_monotonic'] for r in self.active.values())
  j=dict(task='LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1',execution_id=D.name,MAX_CONCURRENT_GPUS=1,MAX_GPU_HOURS=36.0,protective_STOP_at_GPUh=35.8,actual_new_GPU_hours=(done+live)/3600,prior_same_task_GPUh=self.prior_GPUh,cumulative_same_task_GPU_hours=self.prior_GPUh+(done+live)/3600,remaining_same_task_GPU_hours=36.0-self.prior_GPUh-(done+live)/3600,completed_launch_to_exit_GPU_hours=done/3600,active_workers=copy.deepcopy(list(self.active.values())),completed_workers=copy.deepcopy(self.completed),scope='actual GPU-assigned worker launch-to-exit includes model loading/restore/teardown; CPU/admission excluded; concurrent devices summed',historical_same_task_GPU_hours_charged_to_cumulative_budget=True,epoch=now)
  publish(D/'gpu_resource_ledger.json',j,replace=True)
  if self.prior_GPUh+(done+live)/3600>=35.8:stop('GPU_BUDGET_PROTECTIVE_STOP',used_GPUh=self.prior_GPUh+(done+live)/3600,current_execution_GPUh=(done+live)/3600,prior_GPUh=self.prior_GPUh,hard_cap=36.0)
  return j
 def _completion_bindings(self,result,identity,freeze,env,kind,doc,condition,uuid,base):
  assert read(D/'SOURCE_CODE_FREEZE.json')==freeze,'SOURCE_MANIFEST_CHANGED_DURING_WORKER'
  verify_launch_registration(freeze)
  profile=result.with_suffix('.PROFILE.json');envpath=Path(str(base)+'.ENV.json')
  observed=read(envpath);assert observed['identity']==identity,'COMPLETION_ENV_IDENTITY_MISMATCH'
  keys=['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG','FLA_CACHE_MODE','FLA_CONFIG_DIR','CUDA_VISIBLE_DEVICES','LING_TARGET_GPU_UUID']
  validate_completion_assets(read(result),read(profile),observed,kind=kind,document=doc,condition=condition,uuid=uuid,expected_environment={k:env[k] for k in keys},source_verified=True,source_commit=freeze['source_commit'])
  return dict(result_sha256=sha(result),profile_sha256=sha(profile),environment_sha256=sha(envpath),source_manifest_sha256=sha(D/'SOURCE_CODE_FREEZE.json'))
 def _terminal_gate(self,uuid,tag,baseline):
  # One predetermined three-point idle gate after confirmed natural exit. No retry.
  gate_begin=time.time();gate_mono=time.monotonic();rows=[];raw_path=D/'jobs'/(tag+'.TERMINAL_RESOURCE_RAW.jsonl');pin=D/'FAILED_RESOURCE_SNAPSHOT.json'
  for i in range(3):
   assert not (D/'STOP.json').exists(),'GLOBAL_STOP_BEFORE_TERMINAL_GATE'
   if i:time.sleep(5)
   try:snap=sample(uuid)
   except BaseException as e:record_sampling_failure(e,uuid,None,baseline,raw_path=raw_path,pinned_path=pin)
   snap['terminal_gate_sample_index']=i
   def checked(s,u,o,b):
    validate_timing_envelope(s);strict_snapshot_safety(s,u,None,b)
    return g.assess(s,u,None,b)
   assess_with_evidence(snap,uuid,None,baseline,raw_path=raw_path,pinned_path=pin,assess_fn=checked);rows.append(snap)
  envelope=copy.deepcopy(rows[-1]);envelope['terminal_window_snapshots']=rows
  result=assess_with_evidence(envelope,uuid,None,baseline,raw_path=raw_path,pinned_path=pin,assess_fn=lambda s,u,o,b:assess_terminal_window(s['terminal_window_snapshots'],u,b))
  path=D/'jobs'/(tag+'.TERMINAL_RESOURCE_PASS.json');publish(path,dict(status='PASS_ORIGINAL_THREE_POINT_TERMINAL_IDLE_GATE',samples=rows,original_window_result=result,science_trajectory_count=0,epoch=time.time()))
  return dict(path=str(path),sha256=sha(path),status='PASS_ORIGINAL_THREE_POINT_TERMINAL_IDLE_GATE',started_epoch=gate_begin,completed_epoch=time.time(),wall_seconds=time.monotonic()-gate_mono,scope='post-wait CPU-only idle admission time excluded from GPU worker launch-to-exit')
 def job(self,uuid,kind,doc,condition='FP'):
  assert not (D/'STOP.json').exists(),'EXISTING_FIRST_STOP'
  validate_single_device_routing(self.routing)
  assert uuid==TARGET_UUID,'UNAUTHORIZED_NON_GPU6_LAUNCH'
  freeze=read(D/'SOURCE_CODE_FREEZE.json')
  verify_launch_registration(freeze)
  tag=doc+'__'+condition+'__'+kind
  if kind=='canary':result=D/'canary'/('gpu'+str(self.routing['authorized_devices'][uuid])+'.json')
  elif kind in ['parent','fresh']:result=D/'runs'/(doc+'__'+condition+'__'+('parent' if kind=='parent' else 'resumed_parity')+'.json')
  elif kind=='formal1':result=D/'segments'/(doc+'__'+condition+'__part1.json')
  else:result=D/'formal'/(doc+'__'+condition+'__formal.json')
  base=D/'jobs'/tag;base.parent.mkdir(exist_ok=True)
  assert not Path(str(base)+'.LAUNCH.json').exists() and not result.exists(),'NO_SILENT_JOB_RETRY'
  with self.lock:assert not self.active,'SINGLE_GPU_WORKER_ALREADY_ACTIVE'
  admitted=g.admit(uuid,tag);assert admitted['graphics_baseline']==read(D/'resource_baselines.json')[uuid]
  # Preserve original admission and additionally reject early CSV-only conflicts.
  admission_rows=read(admitted['sample_path'])['samples']
  for row in admission_rows:
   def checked_admission(s,u,o,b):
    validate_timing_envelope(s);strict_snapshot_safety(s,u,None,b)
    return g.assess(s,u,None,b)
   assess_with_evidence(row,uuid,None,admitted['graphics_baseline'],raw_path=D/'admission'/(tag+'.STRICT_RAW.jsonl'),pinned_path=D/'FAILED_RESOURCE_SNAPSHOT.json',assess_fn=checked_admission)
  assert not (D/'STOP.json').exists(),'GLOBAL_STOP_BEFORE_LAUNCH'
  env=environment(uuid);command=[PYTHON,'-B','-u',str(W/'cleanroom_worker.py'),'--kind',kind,'--document',doc,'--condition',condition,'--result',str(result)]
  source_commit=freeze['source_commit']
  log=Path(str(base)+'.log').open('x');start=time.time();start_mono=time.monotonic()
  try:process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=_child_parent_death)
  except BaseException:log.close();raise
  identity=None
  launch=dict(tag=tag,kind=kind,document=doc,condition=condition,UUID=uuid,identity=None,spawned_PID=process.pid,started_epoch=start,started_monotonic=start_mono,command=command,result_path=str(result),source_commit=source_commit)
  try:
   identity=g.proc_identity(process.pid);assert identity['uid']==os.getuid(),'SPAWNED_WORKER_UID_CONFLICT';launch['identity']=identity
   publish(str(base)+'.LAUNCH.json',launch)
   with self.lock:
    assert uuid not in self.active,'DUPLICATE_ACTIVE_UUID';self.active[uuid]=launch;self.ledger()
   event('WORKER_LAUNCHED',tag=tag,UUID=uuid,PID=process.pid)
  except BaseException as registration_error:
   failure=repr(registration_error);trace=traceback.format_exc()
   try:stop(failure,stage='POST_SPAWN_REGISTRATION',tag=tag,traceback=trace)
   except BaseException as publication_error:
    _exclusive_emergency(D/'STOP.json',dict(status='STOP',task=TASK,reason=failure,stage='POST_SPAWN_REGISTRATION',tag=tag,traceback=trace,epoch=time.time(),no_retry=True,ordinary_STOP_publication_error=repr(publication_error)))
   cleanup_identity=identity
   try:
    try:rc=process.wait(timeout=60)
    except subprocess.TimeoutExpired:
     cleanup_identity=_actual_own_child_identity(process,identity);process.terminate()
     try:rc=process.wait(timeout=15)
     except subprocess.TimeoutExpired:
      _actual_own_child_identity(process,cleanup_identity);process.kill();rc=process.wait(timeout=10)
    end=time.time();end_mono=time.monotonic()
    receipt=dict(**launch,status='FAILED_OR_ABORTED',reason=failure,registration_failure=True,registration_traceback=trace,cleanup_identity=cleanup_identity,returncode=rc,completed_epoch=end,completed_monotonic=end_mono,wall_seconds=end_mono-start_mono,epoch_span_seconds=end-start,wait_confirmed_child_exit=True,NVML_periodic_peak_owned_bytes=None,NVML_sample_count=0,NVML_scope='registration failed before periodic sampling; not measured')
    _exclusive_emergency(str(base)+'.LAUNCH.json',launch)
    _exclusive_emergency(str(base)+'.EXIT.json',receipt)
    with self.lock:
     self.completed.append(receipt);self.active.pop(uuid,None);self.ledger()
   finally:log.close()
   raise RuntimeError('POST_SPAWN_REGISTRATION_FAILED_CHILD_EXITED') from registration_error
  stop_seen=None;last_guard=0.;max_position=-1;last_progress=time.monotonic();nvml_peak=None;allocated_peak=None;reserved_peak=None;sample_count=0;timeout=900 if kind=='canary' else 7200 if kind in ['parent','fresh','formal1'] else 10800
  failure=None
  try:
   raw=(Path('/proc')/str(process.pid)/'environ').read_bytes().decode(errors='surrogateescape').split('\0');observed=dict(x.split('=',1) for x in raw if '=' in x)
   keys=['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG','FLA_CACHE_MODE','FLA_CONFIG_DIR','CUDA_VISIBLE_DEVICES','LING_TARGET_GPU_UUID']
   assert {k:observed.get(k) for k in keys}=={k:env[k] for k in keys},'ACTUAL_START_ENV_MISMATCH'
   publish(str(base)+'.ENV.json',dict(environment={k:observed[k] for k in keys},identity=identity))
   while process.poll() is None:
    with self.lock:self.ledger()
    now=time.time()
    if now-last_guard>=10 and not (D/'STOP.json').exists():
     try:
      raw_path=Path(str(base)+'.RESOURCE_RAW.jsonl');pinned_path=D/'FAILED_RESOURCE_SNAPSHOT.json'
      poll_before=process.poll()
      if poll_before is not None:break
      before_mono=time.monotonic();before_epoch=time.time();assert g.proc_identity(process.pid)==identity,'OWN_PROCESS_IDENTITY_CHANGED_BEFORE_SAMPLE'
      try:snap=sample(uuid)
      except BaseException as sampling_error:record_sampling_failure(sampling_error,uuid,identity,admitted['graphics_baseline'],raw_path=raw_path,pinned_path=pinned_path)
      poll_after=process.poll();after_mono=time.monotonic()
      snap['child_lifecycle_bracket']=dict(poll_before=poll_before,poll_after=poll_after,owner=identity,started_monotonic=before_mono,completed_monotonic=after_mono,started_epoch=before_epoch,completed_epoch=time.time())
      def checked(s,u,o,b):
       validate_timing_envelope(s)
       for q in s['query_timings'].values():assert before_mono<=q['started_monotonic']<=q['completed_monotonic']<=after_mono,'QUERY_OUTSIDE_CHILD_BRACKET'
       return assess_lifecycle_sample(s,u,o,b,poll_before,poll_after)
      assessed=assess_with_evidence(snap,uuid,identity,admitted['graphics_baseline'],raw_path=raw_path,pinned_path=pinned_path,assess_fn=checked)
      if assessed['lifecycle_state']=='NATURAL_EXIT_TRANSITION_NOT_RESOURCE_PASS':
       publish(str(base)+'.NATURAL_EXIT_TRANSITION.json',dict(status='TRANSITION_NOT_RUNNING_RESOURCE_PASS',bracket=snap['child_lifecycle_bracket'],raw_path=str(raw_path),science_qualification=False));break
      value=assessed['resource'];sample_count+=1
      own=[p['usedGpuMemory'] for p in snap['direct_NVML_processes']['compute']['processes'] if p['pid']==identity['pid']]
      if own:nvml_peak=max(own+[nvml_peak or 0])
      with Path(str(base)+'.RESOURCE.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=now,**value,owned_NVML_bytes=own))+'\n');f.flush();os.fsync(f.fileno())
     except ResourceEvidenceError as e:
      failure=repr(e);stop(failure,stage='RUNNING_RESOURCE_SAFETY',tag=tag,evidence_path=e.evidence_path,evidence_sha256=e.evidence_sha256,forbiddenPIDs=e.forbidden_pids,original_resource_error=e.original_error)
     except BaseException as e:failure=repr(e);stop(failure,stage='RUNNING_RESOURCE_EVIDENCE_OR_PUBLICATION_ERROR',tag=tag,traceback=traceback.format_exc())
     last_guard=now
    # A model's CUDA context may still be absent during CPU model setup; only
    # foreign processes and headroom matter until its frozen in-worker guard.
    tail=Path(str(base)+'.log').read_text()[-12000:]
    for line in reversed(tail.splitlines()):
     try:j=json.loads(line)
     except (ValueError,TypeError):continue
     if isinstance(j,dict) and 'position' in j:
      pos=j['position']
      if pos>max_position:max_position=pos;last_progress=time.monotonic()
      break
    if time.time()-start>timeout or time.monotonic()-last_progress>900:
     failure='BOUNDED_WORKER_TIMEOUT';stop(failure,tag=tag,elapsed=time.time()-start)
    if (D/'STOP.json').exists():
     if stop_seen is None:stop_seen=time.monotonic()
     waited=time.monotonic()-stop_seen
     if waited>60 and process.poll() is None:
      assert g.proc_identity(process.pid)==identity
      process.send_signal(signal.SIGTERM)
     if waited>75 and process.poll() is None:
      assert g.proc_identity(process.pid)==identity
      process.kill()
    time.sleep(2)
   rc=process.wait();end=time.time();end_mono=time.monotonic()
  except BaseException as e:
   failure=repr(e);stop(failure,stage='CONTROL_SUPERVISION',tag=tag,traceback=traceback.format_exc())
   try:rc=process.wait(timeout=60)
   except subprocess.TimeoutExpired:
    assert g.proc_identity(process.pid)==identity;process.terminate()
    try:rc=process.wait(timeout=15)
    except subprocess.TimeoutExpired:
     assert g.proc_identity(process.pid)==identity;process.kill();rc=process.wait(timeout=10)
   end=time.time();end_mono=time.monotonic()
  finally:log.close()
  receipt=dict(**launch,completed_epoch=end,completed_monotonic=end_mono,wall_seconds=end_mono-start_mono,epoch_span_seconds=end-start,returncode=rc,NVML_periodic_peak_owned_bytes=nvml_peak,NVML_sample_count=sample_count,NVML_scope='periodic owned PID samples every approximately10s, not instantaneous peak')
  bindings=None
  try:
   assert rc==0 and result.exists(),('WORKER_NONZERO_OR_MISSING_RESULT',rc)
   bindings=self._completion_bindings(result,identity,freeze,env,kind,doc,condition,uuid,base)
   assert not (D/'STOP.json').exists(),'EXISTING_FIRST_STOP_PREVENTS_RESOURCE_QUALIFIED_COMPLETION'
   terminal=self._terminal_gate(uuid,tag,admitted['graphics_baseline'])
   assert self._completion_bindings(result,identity,freeze,env,kind,doc,condition,uuid,base)==bindings,'COMPLETED_ASSETS_CHANGED_DURING_TERMINAL_GATE'
   assert not (D/'STOP.json').exists(),'GLOBAL_STOP_PREVENTS_JOB_PASS'
   receipt.update(status='PASS',terminal_resource_gate=terminal,**bindings)
  except BaseException as e:
   receipt.update(status='FAILED_OR_ABORTED',reason=repr(e),supervision_failure=failure)
   if bindings is not None:receipt.update(artifact_status='PASS_ARTIFACT_BINDINGS_ONLY_RESOURCE_NOT_QUALIFIED',artifact_result_sha256=bindings['result_sha256'],artifact_profile_sha256=bindings['profile_sha256'],artifact_environment_sha256=bindings['environment_sha256'])
   if isinstance(e,ResourceEvidenceError):stop(repr(e),stage='TERMINAL_RESOURCE_SAFETY',tag=tag,evidence_path=e.evidence_path,evidence_sha256=e.evidence_sha256,forbiddenPIDs=e.forbidden_pids,original_resource_error=e.original_error)
   else:stop(repr(e),stage=kind,tag=tag)
  publish(str(base)+'.EXIT.json',receipt)
  with self.lock:self.completed.append(receipt);del self.active[uuid];self.ledger()
  assert receipt['status']=='PASS',receipt
  event('WORKER_COMPLETE',tag=tag,UUID=uuid,wall_seconds=receipt['wall_seconds'])
  return read(result)
 def lane(self,uuid):
  units=read(D/'FROZEN_SPEC.json')['new_units']
  for u in units:
   doc,cond=u['document'],u['rotation']
   if self.routing['document_devices'][doc]!=uuid:continue
   key=doc+'__'+cond
   if key in read(D/'reused_FP_bindings.json')['units']:
    b=read(D/'reused_FP_bindings.json')['units'][key];assert sha(b['original_checkpoint'])==b['checkpoint_sha256'] and sha(b['original_parent'])==b['parent_sha256']
    parent=read(D/'runs'/(key+'__parent.json'))
   else:parent=self.job(uuid,'parent',doc,cond)
   fresh=self.job(uuid,'fresh',doc,cond);compare_exact(parent,fresh,uuid)
   row=dict(status='PASS',document=doc,rotation=cond,qualified_gpu_uuid=uuid,parent_sha256=sha(D/'runs'/(key+'__parent.json')),resumed_sha256=sha(D/'runs'/(key+'__resumed_parity.json')),checkpoint_sha256=sha(D/'runs'/(key+'__8192.pt')),observer_ON_OFF_exact=True,full64_logits_native_cache_checkpoint_RNG='PASS_FROZEN_CONTRACT',INT8_QDQ='N/A_FP' if cond=='FP' else 'PASS_EVERY_TOKEN_WRITEBACK_AND_NEXT_KERNEL_CONSUMPTION')
   publish(D/'qualification_units'/(key+'.json'),row);event('CONDITION_QUALIFIED',document=doc,condition=cond,UUID=uuid)
   self.forecast()
 def forecast(self):
  with self.lock:ledger=self.ledger()
  parents=[read(p) for p in (D/'runs').glob('*__parent.json')]
  existing_cp=len(list((D/'runs').glob('*__8192.pt')));qualified=len(list((D/'qualification_units').glob('*.json')))
  parent_est=max([1500.]+[p['elapsed_seconds'] for p in parents])*1.25
  hist=read(D/'preregistration.json')['historical_formal_worker_seconds']
  late_rates=[]
  for parent in parents:
   memories=parent.get('memory',[])
   for a,b in zip(memories[-4:],memories[-4:][1:]):
    if b['position']>a['position']:late_rates.append((b['elapsed']-a['elapsed'])/(b['position']-a['position']))
  current_formal_prediction=max(late_rates,default=0.)*24576*1.35+max([60.]+[p.get('loading_seconds',0.) for p in parents])
  formal_est=max(max(hist or [3600.])*1.5,current_formal_prediction)
  done_formal=len(list((D/'formal').glob('*__formal.json')))
  projected=ledger['cumulative_same_task_GPU_hours']+(12-existing_cp)*parent_est/3600+(12-qualified)*90/3600+(12-done_formal)*formal_est/3600
  report=dict(epoch=time.time(),status='PASS_BUDGET_FORECAST' if projected<=36 else 'STOP_BUDGET_FORECAST',used_GPUh=ledger['cumulative_same_task_GPU_hours'],current_execution_GPUh=ledger['actual_new_GPU_hours'],prior_same_task_GPUh=self.prior_GPUh,projected_total_GPUh=projected,cap_GPUh=36.,unbuilt_prefixes=12-existing_cp,qualifications_remaining=12-qualified,formal_remaining=12-done_formal,parent_seconds_estimate=parent_est,formal_continuation_seconds_estimate=formal_est,new_measured_late_token_seconds=late_rates,basis='actual completed parent launch-to-exit and previous same frozen full-model formal receipts; 25% parent/50% historical formal margins and new qualification late-token throughput*1.35 plus loading; not a deadline')
  publish(D/'budget_forecast.json',report,replace=True)
  if projected>36:stop('PROJECTED_BUDGET_EXCEEDS_36_GPUH',forecast=report);raise RuntimeError('BUDGET_FORECAST_STOP')

def main():
 hold=(D/'controller.lock').open('a');fcntl.flock(hold,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (D/'STOP.json').exists() and not (D/'RUN_CONTROLLER.json').exists(),'INDEPENDENT_TASK_ALREADY_STARTED_NO_RETRY'
 freeze=protected(verify_launch_registration)
 publish(D/'RUN_CONTROLLER.json',dict(identity=g.proc_identity(os.getpid()),source_commit=freeze['source_commit'],epoch=time.time(),max_concurrent_GPU_workers=1,max_new_GPUh=36.0,execution_id=D.name))
 ctl=Controller();devices=validate_single_device_routing(ctl.routing);event('G1_REAL_MODEL_CANARY')
 def interrupted(number,frame):
  stop('CONTROLLER_SIGNAL',signal=number,stage='CONTROL_INTERRUPTION');raise SystemExit(128+number)
 old_handlers={sig:signal.signal(sig,interrupted) for sig in [signal.SIGTERM,signal.SIGINT]}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
   futures=[pool.submit(protected,ctl.job,u,'canary',next(d for d,v in ctl.routing['document_devices'].items() if v==u)) for u in devices]
   for f in concurrent.futures.as_completed(futures):f.result()
  publish(D/'G1_CANARY_COMPLETE.json',dict(status='PASS_ALL_SELECTED_REAL_MODEL_FORWARD_CANARIES',science_trajectory_count=0))
  ctl.forecast();event('G2_QUALIFICATION_RUNNING')
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
   futures=[pool.submit(protected,ctl.lane,u) for u in devices]
   for f in concurrent.futures.as_completed(futures):f.result()
  units=[read(D/'qualification_units'/(u['document']+'__'+u['rotation']+'.json')) for u in read(D/'FROZEN_SPEC.json')['new_units']]
  assert len(units)==12 and all(x['status']=='PASS' for x in units)
  publish(D/'LONG_HORIZON_QUALIFICATION.json',dict(status='PASS',count=12,units=units))
  publish(D/'QUALIFICATION_GATES_COMPLETED.json',dict(status='ALL_EIGHT_GATES_PASS',source='newCleanRoom G0/G1/G2 identity, environment, FP/INT8 fresh64 and physical gates',full32768='PENDING'))
  publish(D/'resolution_contract.json',dict(status='SUPPORTED_BY_IDENTICAL_FROZEN_MATH_RUNTIME_PROFILE_AND_EXACT_FRESH64_REPLAY',threshold=3.3300064494019496e-6,threshold_origin='historical empirical arithmetic resolution, not statistical significance',new_evidence='the single authorized GPU6 actual frozen profile and zero self fullV FP32 KL; every FP/student condition exact full64 fresh replay on its own device before outcomes',limitations='not a new32K numerical error calibration; differences within the historical resolution remain NUMERICALLY_UNRESOLVED; no cross-device equality assumed'))
  event('ALL_12_QUALIFIED_FORMAL_STARTING')
  for doc in read(D/'FROZEN_4DOC_SELECTION.json')['execution_order']:
   uuid=ctl.routing['document_devices'][doc['document']]
   for condition in doc['conditions']:
    ctl.forecast();ctl.job(uuid,'formal1',doc['document'],condition);ctl.job(uuid,'formal2',doc['document'],condition)
  publish(D/'COMPLETE.json',dict(status='COMPLETE',FP=4,students=8,endpoint=32768,epoch=time.time()));event('SCIENCE_COMPLETE_CPU_VERIFICATION')
 except BaseException as e:
  stop(repr(e),stage='CLEANROOM_CONTROLLER',traceback=traceback.format_exc());event('FAIL_CLOSED_CPU_DELIVERY',reason=repr(e))
 finally:
  assert not ctl.active,'ACTIVE_WORKERS_MUST_EXIT_BEFORE_CPU_SEAL'
  with ctl.lock:ctl.ledger()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
  r=subprocess.run([PYTHON,'-B','-u',str(W/'cleanroom_finalize.py')],env=env)
  assert r.returncode==0,'CPU_DELIVERY_ERROR_PRESERVE_FIRST_STOP'
  for sig,handler in old_handlers.items():signal.signal(sig,handler)
  hold.close()

if __name__=='__main__':main()
