"""Single offline finite coordinator, three disjoint per-UUID serial queues."""
import sys,os,argparse,time,json,subprocess,threading,traceback,concurrent.futures,statistics,signal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
from record_validation import validate
from qualification_reuse import admit_reused48,remaining_units,require_current_dispatch
from durable_output29 import publish_unit
plan=C.read(R/'GPU_EXECUTION_PLAN.json')
if not a.execute:print(json.dumps(dict(status='CPU_CHECK_ONLY',qualification_units=len(plan['qualification_units']),formal_samples=90,GPU_dispatches=0)));raise SystemExit
permit=R/'GPU_PERMIT.json';C.authorize(R,permit);C.require(not (R/'GPU_RUN_STARTED.json').exists(),'Already dispatched; no relaunch or resets')
identity=C.process_identity(os.getpid());C.atomic_new(R/'GPU_RUN_STARTED.json',dict(identity=identity,monotonic=time.monotonic(),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
STOP=threading.Event();lock=threading.RLock();usage=[];complete=[];all_children=[];active={}
C.event(R/'GPU_RESOURCE_LEDGER.jsonl','SESSION_START',identity=identity,resource_policy='NO_GPU_TIME_BUDGET')
def first_failure(exc,unit):
 STOP.set()
 with lock:
  if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),unit=unit))
def controlled_signal(signum,_frame):
 first_failure(RuntimeError('External controller signal requests controlled owned-only STOP: '+signal.Signals(signum).name),'CONTROLLER_SIGNAL')
signal.signal(signal.SIGTERM,controlled_signal);signal.signal(signal.SIGINT,controlled_signal)
def publish_ledger():
 with lock:
  jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];byid={j['id']:j for j in jobs};legal=[x for x in usage if x['unit'] in byid and x['exit_returncode']==0 and x['release']=='PASS'];counts={m:sum(byid[x['unit']]['condition']==m for x in legal) for m in C.METHODS}
  C.status(R/'FORMAL_PROGRESS.json',dict(status='FORMAL_COMPLETE90_WAIT_CPU_FINISHER' if len(legal)==90 else 'TRUE_STOP_PENDING_CPU_REPORT' if (R/'FIRST_FAILURE.json').exists() else 'FORMAL_RUNNING' if (R/'AUTHORIZATION_FORMAL.json').exists() else 'QUALIFICATION_PENDING_FORMAL_GATE',completed=len(legal),planned=90,completed_by_method=counts,active=[dict(unit=k,**v) for k,v in active.items() if k in byid],source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),gold_blind=True))
  C.status(R/'RESOURCE_LEDGER.json',dict(resource_policy='NO_GPU_TIME_BUDGET',intervals=usage,completed_units=complete,reservation_GPUh_lower=sum(x['reservation_lower_seconds'] for x in usage)/3600,reservation_GPUh_upper=sum(x['reservation_upper_seconds'] for x in usage)/3600,observed_physical_occupancy_lower_GPUh=sum(x['observed_occupancy_lower_seconds'] for x in usage)/3600,EXACT_physical_GPUh='UNRESOLVED'))
def reap_identity_transition(proc,child):
 # Linux may clear /proc/cmdline while an exiting process has not become
 # waitpid-waitable yet. A changed identity is never a signal target.
 if proc.poll() is not None:return True
 p=Path('/proc')/str(proc.pid);exiting=False;detail={}
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes();detail=dict(state=fields[0],cmdline_empty=not cmd)
  exiting=fields[0] in ('Z','X','x') or not cmd
 except (FileNotFoundError,ProcessLookupError):exiting=True;detail=dict(proc_entry_absent=True)
 if not exiting:raise RuntimeError('Active worker PID/starttick/argv changed without exit transition; never signal replacement')
 C.event(R/'GPU_RESOURCE_LEDGER.jsonl','BOUNDED_OWN_POPEN_REAP_AFTER_IDENTITY_TRANSITION',identity=child,observation=detail,max_wait_seconds=1.)
 try:proc.wait(timeout=1.)
 except subprocess.TimeoutExpired:raise RuntimeError('Identity transition did not reach real owned Popen exit within1s; never signal mismatchedPID')
 return True
def run(unit,formal=False):
 require_current_dispatch(R,unit,formal)
 name=unit['id'];gpu=unit['gpu'];proc=None;child=None;log=None;dispatch=None;last_alive=None;gpu_first=None;gpu_last=None;exit_seen=None;release_seen=None;last_gpu_owned=None
 try:
  C.require(not STOP.is_set(),'Global STOP prevents dispatch');C.device(gpu);C.storage_guard(R);target=R/('formal' if formal else 'execution')/name;target.mkdir(parents=True,exist_ok=True)
  C.require(not (target/'DISPATCH.json').exists(),'Unit already dispatched; no retry')
  C.atomic_new(target/'DISPATCH.json',dict(unit=name,controller_identity=identity,permit_sha256=C.sha(permit),gpu=gpu,gpu_uuid=C.UUIDS[gpu],formal=formal))
  log=(target/'worker.log').open('x');dispatch=time.monotonic();cmd=[sys.executable,'-u','-B',str(R/'scripts/gpu_unit_worker.py'),'--root',str(R),'--unit',name]+(['--formal'] if formal else [])
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=C.worker_environment(gpu),start_new_session=True);child=C.process_identity(proc.pid)
  with lock:all_children.append(child);active[name]=dict(identity=child,gpu=gpu,gpu_uuid=C.UUIDS[gpu],formal=formal)
  C.event(R/'GPU_RESOURCE_LEDGER.jsonl','DISPATCH',unit=name,gpu=gpu,identity=child,formal=formal);publish_ledger();last_alive=dispatch;last_progress=dispatch;signature=None;last_meaning=None;lastpoll=0
  while proc.poll() is None:
   if not C.same_process(child):
    if reap_identity_transition(proc,child):break
   now=time.monotonic();last_alive=now
   p=target/'PROGRESS.json'
   try:s=p.stat();sig=(s.st_ino,s.st_mtime_ns,s.st_size)
   except FileNotFoundError:sig=None
   if sig!=signature:
    # PROGRESS is mutable atomic-replace data, not an immutable sourcepin.
    # Reading one opened inode yields a complete old/new JSON snapshot even
    # while the worker replaces the pathname; C.read's stable-path check would
    # incorrectly turn this legitimate update into a scientific STOP.
    try:
     with p.open('rb') as stream:current=json.load(stream)
    except FileNotFoundError:current={}
    signature=sig;meaning=(current.get('phase'),current.get('generated_tokens'),current.get('input_forwards'),current.get('token'),current.get('total'),current.get('condition'))
    if meaning!=last_meaning:last_meaning=meaning;last_progress=now
   if STOP.is_set():raise RuntimeError('Shared first failure stops remaining own units')
   C.require(now-last_progress<1800,'No real phase/token progress for1800s')
   if now-lastpoll>=15:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==child['pid']]
    if own:
     # Validate exactly this fresh observation. A second query can race with
     # healthy child exit; a foreign/UUID violation in this observation must
     # remain a failure even when the child subsequently exits successfully.
     devices=[x for x in obs['devices'] if x['physical_index']==gpu]
     C.require(len(devices)==1 and devices[0]['uuid']==C.UUIDS[gpu] and '3090' in devices[0]['name'] and devices[0]['memory_total_MiB']>=24000,'Observed physical device identity differs')
     C.require(all(x['uuid']==C.UUIDS[gpu] for x in own),'Observed active worker used unauthorized UUID')
     C.require(not any(x['uuid']==C.UUIDS[gpu] and x['pid']!=child['pid'] for x in obs['apps']),'Observed foreign process overlapped assigned UUID; preserve first cause')
     gpu_first=now if gpu_first is None else gpu_first;gpu_last=now;last_gpu_owned=now
    else:C.require(not any(x['uuid']==C.UUIDS[gpu] for x in obs['apps']),'Foreign GPU process appeared')
    lastpoll=now;C.storage_guard(R)
   time.sleep(.5)
  exit_seen=time.monotonic();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_PROCESS_EXIT',unit=name,lower=last_alive,upper=exit_seen,returncode=proc.returncode)
  C.require(proc.returncode==0,'Unit worker exited unsuccessfully; inspect immutable log, no retry')
  out=C.read(target/'RESULT.json');copy=dict(out);payload=copy.pop('payload_sha256');C.require(C.objsha(copy)==payload and out['identity']==child and out['status']=='PASS' and out['permit_sha256']==C.sha(permit),'Process/result/permit mismatch')
  if formal:validate(out,unit,R)
  with lock:complete.append(name)
 except BaseException as exc:first_failure(exc,name);raise
 finally:
  if proc is not None:
   if proc.poll() is None:
    if child is None or not C.same_process(child):
     reap_identity_transition(proc,child)
    else:
     proc.terminate()
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:
      if not C.same_process(child):reap_identity_transition(proc,child)
      elif proc.poll() is None:proc.kill();proc.wait(timeout=15)
   if exit_seen is None:
    exit_seen=time.monotonic();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_PROCESS_EXIT',unit=name,lower=last_alive or dispatch,upper=exit_seen,returncode=proc.returncode)
   began=time.monotonic()
   while time.monotonic()-began<120:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==proc.pid];now=time.monotonic()
    if own and proc.poll() is not None:
     try:replacement=C.process_identity(proc.pid)
     except (OSError,ValueError):replacement=None
     if replacement is not None and replacement!=child:
      C.event(R/'GPU_RESOURCE_LEDGER.jsonl','CONFIRMED_POPEN_EXIT_REUSED_PID_IS_FOREIGN',unit=name,original_identity=child,replacement_identity=replacement,NVML_records=own,signals_to_replacement=0)
      own=[]
    if not own:release_seen=now;C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_OWN_UUID_RELEASE',unit=name,lower=last_gpu_owned or dispatch,upper=now,all_UUIDs_checked=True);break
    C.require(all(x['uuid']==C.UUIDS[gpu] for x in own),'Own process used unauthorized GPU');last_gpu_owned=now;time.sleep(.5)
   if log:log.close()
   record=dict(unit=name,gpu=gpu,identity=child,exit_returncode=proc.returncode,release='PASS' if release_seen else 'UNRESOLVED',reservation_lower_seconds=max(0,(last_alive or dispatch)-dispatch),reservation_upper_seconds=(release_seen or time.monotonic())-dispatch,observed_occupancy_lower_seconds=max(0,gpu_last-gpu_first) if gpu_first is not None else 0,physical_occupancy_upper_seconds=(release_seen or time.monotonic())-dispatch)
   with lock:usage.append(record);active.pop(name,None)
   C.atomic_new(target/'RESOURCE_INTERVAL.json',record);C.event(R/'GPU_RESOURCE_LEDGER.jsonl','CLOSED_RESOURCE_INTERVAL',**record);publish_ledger()
   if release_seen is None:first_failure(RuntimeError('GPU release unresolved in cleanup observation'),name)
   try:
    durable=publish_unit(R,target,unit=name,formal=formal)
    C.require(durable['status']=='PASS_DURABLE29_CLOSED_UNIT','Output29 durable receipt missing')
    C.event(R/'GPU_RESOURCE_LEDGER.jsonl','DURABLE29_UNIT_COMMITTED',unit=name,formal=formal,receipt_sha256=C.sha(target/'DURABLE29_RECEIPT.json'))
   except BaseException as durable_error:first_failure(durable_error,name);raise
def queues(units,formal=False):
 by={g:[u for u in units if u['gpu']==g] for g in C.AUTHORIZED_GPUS}
 def serial(g):
  for u in by[g]:
   if STOP.is_set():return
   run(u,formal)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(serial,g) for g in C.AUTHORIZED_GPUS]
  for f in concurrent.futures.as_completed(futures):
   try:f.result()
   except BaseException as exc:first_failure(exc,'STAGE_QUEUE')
 C.require(not STOP.is_set(),'Global failed qualification/formal stage; no continuation')
def require_released():
 obs=C.query_gpu();live=[i for i in all_children if C.same_process(i)];C.require(not live,'Actual owned worker PID/starttick/argv still alive');C.require(all(x['release']=='PASS' for x in usage),'Own UUID release interval unresolved');return obs
publish_ledger()
try:
 C.model_guard(R);C.atomic_new(R/'analysis/INHERITED_QUALIFICATION_ADMISSION.json',C.admit_inherited(R));C.validate_jobs(R,C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']);C.atomic_new(R/'analysis/EXECUTION_EXTERNAL_SHA_ADMISSION.json',dict(status='PASS',bound_files=C.verify_external(R)))
 for g in C.AUTHORIZED_GPUS:C.device(g)
 reused=admit_reused48(R,plan);C.atomic_new(R/'analysis/QUALIFICATION_REUSE_RUNTIME_ADMISSION.json',reused)
 # All original fields/persistence/free/context qualifications are reused.
 # The only new qualification stage is the original four GPU1 matched cases;
 # they never use ByteReference, so no historical RAM fixtures are recreated.
 units=remaining_units(plan);queues(units)
 require_released();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','QUALIFICATION_STAGE_COMPLETE',kinds=['matched_performance'],units=4,reused_original_units=48)
 C.require(complete==[u['id'] for u in units],'Finite4 new qualification roster incomplete or reordered');C.verify_sources(R);C.model_guard(R);finalobs=require_released()
 from control_qualification import inspect_qualification
 inspected=inspect_qualification(R,plan)
 qualified=dict(status='PASS_ALL_NEW_METHODS_THREE_UUID_REQUIRED_GATES',units=[u['id'] for u in plan['qualification_units']],reused_qualification_count=48,fresh_qualification_count=4,current_units=complete.copy(),reused_admission_sha256=C.sha(R/'analysis/QUALIFICATION_REUSE_RUNTIME_ADMISSION.json'),devices=[C.UUIDS[g] for g in C.AUTHORIZED_GPUS],historical_base_readonly=True,numerical_81K_claim=False,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),inspection=inspected)
 C.atomic_new(R/'GPU_CODEC_QUALIFICATION.json',qualified)
 C.atomic_new(R/'PERSISTENT_QDQ_NEXT_CONSUMPTION.json',dict(status='PASS_TOKEN0_ALL24LAYER_WRITEBACK_ACTUAL_NEXT_BYTES',qualification_inspection=inspected,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),unquantized_continuation_shadow=False,persistent_carrier='FP32_post_QDQ',short_actual_next_values_and_bytes=True,formal_sparse_actual_next_values_and_bytes=True))
 rates={u['condition']:C.read(R/'execution'/u['id']/'RESULT.json')['result'] for u in plan['qualification_units'] if u['kind']=='matched_performance'}
 C.atomic_new(R/'GPU_THROUGHPUT_AND_MEMORY.json',dict(status='PASS_ACTUAL_MATCHED_GPU1_NONAIME_FOUR_METHODS',same_GPU_uuid=C.UUIDS[1],same_frozen_input=True,results=rates,full_context_is_synthetic_native_SDPA_pressure=True,numerical_81K_claim=False))
 jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];cap_seconds={str(g):sum(81920/min(rates[u['condition']]['timing']['short']['TPS'],rates[u['condition']]['timing']['late']['TPS']) for u in jobs if u['gpu']==g) for g in C.AUTHORIZED_GPUS}
 estimate=dict(status='ACTUAL_NONAIME_RATES_CONDITIONAL_CAP_SCENARIO',rates={k:v['timing'] for k,v in rates.items()},theoretical_max_new_tokens=7372800,all_remaining_hit_cap_seconds_by_GPU=cap_seconds,all_cap_parallel_wall_seconds=max(cap_seconds.values()),all_cap_GPUh=sum(cap_seconds.values())/3600,limitations='MatchedGPU1 fixed4096 input, observer-off early/latewindows, not81K speed guarantee. Othercards share3090 model but no same-input percard speed claim. EOS/load/prompt/logging overhead separate. No hardcap imposed.')
 C.atomic_new(R/'RESOURCE_AND_COST_ESTIMATE_ACTUAL.json',estimate)
 C.validate_jobs(R,jobs)
 C.atomic_new(R/'AUTHORIZATION_FORMAL.json',dict(status='PASS_ALL10_GATES_ALLOW_FORMAL_ONCE',GPU_qualification_sha256=C.sha(R/'GPU_CODEC_QUALIFICATION.json'),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(permit),jobs_sha256=C.sha(R/'FORMAL_JOB_MANIFEST.json'),required_context=82293,max_new_tokens=81920))
 C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FORMAL_STAGE_START',samples=90);publish_ledger();queues(jobs,formal=True);obs=require_released();C.require(all(j['id'] in complete for j in jobs),'Formal90 incomplete')
 C.atomic_new(R/'FORMAL_GENERATIONS_COMPLETE.json',dict(status='PASS_90_LEGAL_GENERATIONS_PROCESSES_RELEASED',count=90,observation=obs,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
 C.atomic_new(R/'SESSION_RESULT.json',dict(status='GPU_COMPLETE_REUSE48_PLUS4_QUALIFICATION_90_FORMAL',completed=complete,reused_qualification_count=48,fresh_qualification_count=4,new_formal_count=90,reuse_admission_sha256=C.sha(R/'analysis/QUALIFICATION_REUSE_RUNTIME_ADMISSION.json'),own_all_UUID_release=True,final_observation=obs))
except BaseException as exc:
 first_failure(exc,'CONTROLLER');obs=require_released();C.atomic_new(R/'SESSION_RESULT.json',dict(status='FAIL_CLOSED_STOP',completed=complete,reused_qualification_count=48 if (R/'analysis/QUALIFICATION_REUSE_RUNTIME_ADMISSION.json').exists() else 0,fresh_qualification_count=sum(x in [u['id'] for u in remaining_units(plan)] for x in complete),own_all_UUID_release=True,final_observation=obs));raise
finally:publish_ledger()
