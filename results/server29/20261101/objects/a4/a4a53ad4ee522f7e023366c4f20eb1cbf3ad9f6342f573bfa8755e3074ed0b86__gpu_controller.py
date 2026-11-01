"""Single offline finite coordinator, two disjoint per-UUID serial queues."""
import sys,os,argparse,time,json,subprocess,threading,traceback,concurrent.futures,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
from record_validation import validate
plan=C.read(R/'GPU_EXECUTION_PLAN.json')
if not a.execute:print(json.dumps(dict(status='CPU_CHECK_ONLY',qualification_units=len(plan['qualification_units']),formal_samples=120,GPU_dispatches=0)));raise SystemExit
permit=R/'GPU_PERMIT.json';C.authorize(R,permit);C.require(not (R/'GPU_RUN_STARTED.json').exists(),'Already dispatched; no relaunch or resets')
identity=C.process_identity(os.getpid());C.atomic_new(R/'GPU_RUN_STARTED.json',dict(identity=identity,monotonic=time.monotonic(),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
STOP=threading.Event();lock=threading.RLock();usage=[];complete=[];all_children=[]
C.event(R/'RESOURCE_EVENTS.jsonl','SESSION_START',identity=identity,resource_policy='NO_GPU_TIME_BUDGET')
def first_failure(exc,unit):
 STOP.set()
 with lock:
  if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),unit=unit))
def publish_ledger():
 with lock:
  C.status(R/'RESOURCE_LEDGER.json',dict(resource_policy='NO_GPU_TIME_BUDGET',intervals=usage,completed_units=complete,reservation_GPUh_lower=sum(x['reservation_lower_seconds'] for x in usage)/3600,reservation_GPUh_upper=sum(x['reservation_upper_seconds'] for x in usage)/3600,observed_physical_occupancy_lower_GPUh=sum(x['observed_occupancy_lower_seconds'] for x in usage)/3600,EXACT_physical_GPUh='UNRESOLVED'))
def run(unit,formal=False):
 name=unit['id'];gpu=unit['gpu'];proc=None;child=None;log=None;dispatch=None;last_alive=None;gpu_first=None;gpu_last=None;exit_seen=None;release_seen=None;last_gpu_owned=None
 try:
  C.require(not STOP.is_set(),'Global STOP prevents dispatch');C.device(gpu);C.storage_guard(R);target=R/('formal' if formal else 'execution')/name;target.mkdir(parents=True,exist_ok=True)
  C.require(not (target/'DISPATCH.json').exists(),'Unit already dispatched; no retry')
  C.atomic_new(target/'DISPATCH.json',dict(unit=name,controller_identity=identity,permit_sha256=C.sha(permit),gpu=gpu,gpu_uuid=C.UUIDS[gpu],formal=formal))
  log=(target/'worker.log').open('x');dispatch=time.monotonic();cmd=[sys.executable,'-u','-B',str(R/'scripts/gpu_unit_worker.py'),'--root',str(R),'--unit',name]+(['--formal'] if formal else [])
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=C.worker_environment(gpu),start_new_session=True);child=C.process_identity(proc.pid)
  with lock:all_children.append(child)
  C.event(R/'RESOURCE_EVENTS.jsonl','DISPATCH',unit=name,gpu=gpu,identity=child,formal=formal);last_alive=dispatch;last_progress=dispatch;signature=None;lastpoll=0
  while proc.poll() is None:
   now=time.monotonic();last_alive=now
   p=target/'PROGRESS.json';sig=(p.stat().st_mtime_ns,p.stat().st_size) if p.exists() else None
   if sig!=signature:signature=sig;last_progress=now
   if STOP.is_set():raise RuntimeError('Shared first failure stops remaining own units')
   C.require(now-last_progress<1800,'No real phase/token progress for1800s')
   if now-lastpoll>=15:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==child['pid']]
    if own:
     C.device(gpu,child['pid']);gpu_first=now if gpu_first is None else gpu_first;gpu_last=now;last_gpu_owned=now
    else:C.require(not any(x['uuid']==C.UUIDS[gpu] for x in obs['apps']),'Foreign GPU process appeared')
    lastpoll=now;C.storage_guard(R)
   time.sleep(.5)
  exit_seen=time.monotonic();C.event(R/'RESOURCE_EVENTS.jsonl','FIRST_PROCESS_EXIT',unit=name,lower=last_alive,upper=exit_seen,returncode=proc.returncode)
  C.require(proc.returncode==0,'Unit worker exited unsuccessfully; inspect immutable log, no retry')
  out=C.read(target/'RESULT.json');payload=dict(out);saved=payload.pop('payload_sha256');C.require(C.objsha(payload)==saved and out['gpu']==gpu and out['gpu_uuid']==C.UUIDS[gpu],'Actual qualification payload/GPU differs');C.require(out['identity']==child and out['status']=='PASS' and out['permit_sha256']==C.sha(permit),'Process/result/permit mismatch')
  if formal:validate(out,unit,R)
  with lock:complete.append(name)
  C.status(R/'FORMAL_PROGRESS.json',dict(status='RUNNING',qualification_completed=len([n for n in complete if n in {u['id'] for u in plan['qualification_units']}]),formal_completed=len([n for n in complete if n in {u['id'] for u in C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']}]),formal_planned=120,completed_units=complete.copy()))
 except BaseException as exc:first_failure(exc,name);raise
 finally:
  if proc is not None:
   if proc.poll() is None:
    C.require(child is not None and C.same_process(child),'Cleanup process identity differs; never kill foreign PID');proc.terminate()
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
     C.require(C.same_process(child),'Own PID changed before SIGKILL');proc.kill();proc.wait(timeout=15)
   if exit_seen is None:
    exit_seen=time.monotonic();C.event(R/'RESOURCE_EVENTS.jsonl','FIRST_PROCESS_EXIT',unit=name,lower=last_alive or dispatch,upper=exit_seen,returncode=proc.returncode)
   began=time.monotonic()
   while time.monotonic()-began<120:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==proc.pid];now=time.monotonic()
    if not own:release_seen=now;C.event(R/'RESOURCE_EVENTS.jsonl','FIRST_OWN_UUID_RELEASE',unit=name,lower=last_gpu_owned or dispatch,upper=now,all_UUIDs_checked=True);break
    C.require(all(x['uuid']==C.UUIDS[gpu] for x in own),'Own process used unauthorized GPU');last_gpu_owned=now;time.sleep(.5)
   if log:log.close()
   record=dict(unit=name,gpu=gpu,identity=child,exit_returncode=proc.returncode,release='PASS' if release_seen else 'UNRESOLVED',reservation_lower_seconds=max(0,(last_alive or dispatch)-dispatch),reservation_upper_seconds=(release_seen or time.monotonic())-dispatch,observed_occupancy_lower_seconds=max(0,gpu_last-gpu_first) if gpu_first is not None else 0,physical_occupancy_upper_seconds=(release_seen or time.monotonic())-dispatch)
   with lock:usage.append(record)
   C.atomic_new(target/'RESOURCE_INTERVAL.json',record);C.event(R/'GPU_RESOURCE_LEDGER.jsonl','ACTUAL_UNIT_INTERVAL',**record);publish_ledger()
   if release_seen is None:first_failure(RuntimeError('GPU release unresolved in cleanup observation'),name)
def queues(units,formal=False):
 by={g:[u for u in units if u['gpu']==g] for g in C.UUIDS}
 def serial(g):
  for u in by[g]:
   if STOP.is_set():return
   run(u,formal)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(serial,g) for g in C.UUIDS]
  for f in concurrent.futures.as_completed(futures):
   try:f.result()
   except BaseException as exc:first_failure(exc,'STAGE_QUEUE')
 C.require(not STOP.is_set(),'Global failed qualification/formal stage; no continuation')
def require_released():
 obs=C.query_gpu();C.require(not any(x['pid'] in {i['pid'] for i in all_children} for x in obs['apps']),'Own worker NVML not released');C.require(all(not C.same_process(i) for i in all_children),'Actual worker still alive');return obs
try:
 C.model_guard(R);C.validate_jobs(R,C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']);C.atomic_new(R/'analysis/EXECUTION_EXTERNAL_SHA_ADMISSION.json',dict(status='PASS',bound_files=C.verify_external(R)))
 for g in C.UUIDS:C.device(g)
 # Only reserved tmpfs comparison fixtures; never historical data, no models.
 ref=C.reference_root();C.require(not ref.exists(),'Existing task reference RAM root; inspect, no reuse');ref.mkdir();s=os.statvfs(ref);C.require(s.f_bavail*s.f_frsize>=24*1024**3,'Temporary RAM filesystem must have24GiB available')
 stages=[['gpu_codec_micro'],['group64_persistent_fresh'],['nonAIME_free_generation_fresh'],['nonAIME_throughput'],['real_model_memory_pressure']]
 for kinds in stages:
  units=[u for u in plan['qualification_units'] if u['kind'] in kinds];queues(units);require_released();C.event(R/'RESOURCE_EVENTS.jsonl','QUALIFICATION_STAGE_COMPLETE',kinds=kinds,units=len(units))
 C.require(len(complete)==len(plan['qualification_units']),'Finite qualification roster incomplete');C.verify_sources(R);C.model_guard(R);finalobs=require_released()
 qualified=dict(status='PASS_ALL_FOUR_G64_CODECS_TWO_AUTHORIZED_GPUS',units=complete.copy(),devices=list(C.UUIDS.values()),numerical_81K_claim=False,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'))
 C.atomic_new(R/'GROUP64_GPU_CODEC_QUALIFICATION.json',qualified)
 C.atomic_new(R/'GROUP64_FREE_GENERATION_CANARY.json',dict(status='PASS_FRESH_SAME_GPU_ACTUAL_SAMPLER_FULL_CHECKPOINT_BYTES',units=[u['id'] for u in plan['qualification_units'] if u['kind']=='nonAIME_free_generation_fresh'],source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
 C.atomic_new(R/'GROUP64_PERSISTENT_QDQ_AUDIT.json',dict(status='PASS_TOKEN0_SINGLE_TOKEN_PROMPT_AND_DECODE',CPU_qualified=True,real_9B_fresh_GDN_and_sampler=True,formal_prompt_tokens_each_QDQ=True,unquantized_continuation_shadow=False,persistent_carrier='FP32_post_QDQ',short_actual_next_values_and_bytes=True,frozen_full_checkpoint_repeats=[0,31,127,511],all_step_fresh_cache_byte_equality_claim=False))
 C.atomic_new(R/'GPU_UUID_AND_RUNTIME_AUDIT.json',dict(status='PASS_BOTH_DRIVER_UUID_AND_PRE_POST_CLOSED_MAPS',devices=list(C.UUIDS.values()),final_worker_release=finalobs,per_unit_receipts='execution/*/DRIVER_IDENTITY.json and PREFLIGHT/POSTFLIGHT_LIBRARY.json'))
 rates={str(g):{} for g in C.UUIDS}
 for u in plan['qualification_units']:
  if u['kind']=='nonAIME_throughput':rates[str(u['gpu'])]=C.read(R/'execution'/u['id']/'RESULT.json')['result']
 C.atomic_new(R/'RESOURCE_AND_COST_ESTIMATE_ACTUAL.json',dict(status='ACTUAL_NONAIME_OBSERVER_OFF_REPLAY_RATES',rates=rates,limitation='Teacher inputs and actual original sampling; free64 TPS separately measured. Not81K speed guarantee. EOS and loading are separate; no elapsed or GPUh cap.'))
 jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];C.validate_jobs(R,jobs)
 C.atomic_new(R/'AUTHORIZATION_FORMAL.json',dict(status='PASS_ALL10_GATES_ALLOW_FORMAL_ONCE',GPU_qualification_sha256=C.sha(R/'GROUP64_GPU_CODEC_QUALIFICATION.json'),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(permit),jobs_sha256=C.sha(R/'FORMAL_JOB_MANIFEST.json'),required_context=82293,max_new_tokens=81920))
 C.event(R/'RESOURCE_EVENTS.jsonl','FORMAL_STAGE_START',samples=120);queues(jobs,formal=True);obs=require_released();C.require(all(j['id'] in complete for j in jobs),'Formal120 incomplete')
 C.atomic_new(R/'FORMAL_GENERATIONS_COMPLETE.json',dict(status='PASS_120_LEGAL_GENERATIONS_PROCESSES_RELEASED',count=120,observation=obs,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
 C.atomic_new(R/'SESSION_RESULT.json',dict(status='GPU_COMPLETE_GROUP64_QUALIFICATION_120_FORMAL',completed=complete,own_all_UUID_release=True,final_observation=obs))
except BaseException as exc:
 first_failure(exc,'CONTROLLER');obs=require_released();C.atomic_new(R/'SESSION_RESULT.json',dict(status='FAIL_CLOSED_STOP',completed=complete,own_all_UUID_release=True,final_observation=obs));raise
finally:publish_ledger()
