"""Single offline finite coordinator, four disjoint per-UUID serial queues."""
import sys,os,argparse,time,json,subprocess,threading,traceback,concurrent.futures,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
from record_validation import validate
plan=C.read(R/'GPU_EXECUTION_PLAN.json')
if not a.execute:print(json.dumps(dict(status='CPU_CHECK_ONLY',qualification_units=len(plan['qualification_units']),formal_samples=60,GPU_dispatches=0)));raise SystemExit
permit=R/'GPU_PERMIT.json';C.authorize(R,permit);C.require(not (R/'GPU_RUN_STARTED.json').exists(),'Already dispatched; no relaunch or resets')
identity=C.process_identity(os.getpid());C.atomic_new(R/'GPU_RUN_STARTED.json',dict(identity=identity,monotonic=time.monotonic(),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
STOP=threading.Event();lock=threading.RLock();usage=[];complete=[];all_children=[]
C.event(R/'GPU_RESOURCE_LEDGER.jsonl','SESSION_START',identity=identity,resource_policy='NO_GPU_TIME_BUDGET')
def first_failure(exc,unit):
 STOP.set()
 with lock:
  if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),unit=unit))
def publish_ledger():
 with lock:
  C.status(R/'RESOURCE_LEDGER.json',dict(resource_policy='NO_GPU_TIME_BUDGET',intervals=usage,completed_units=complete,reservation_GPUh_lower=sum(x['reservation_lower_seconds'] for x in usage)/3600,reservation_GPUh_upper=sum(x['reservation_upper_seconds'] for x in usage)/3600,observed_physical_occupancy_lower_GPUh=sum(x['observed_occupancy_lower_seconds'] for x in usage)/3600,EXACT_physical_GPUh='UNRESOLVED'))
  C.status(R/'FORMAL_PROGRESS.json',dict(task_id=C.TASK,completed=sum(x.startswith('aime26_') for x in complete),total=60,active_own_worker_identities=[i for i in all_children if C.same_process(i)],failure=(R/'FIRST_FAILURE.json').exists(),monotonic=time.monotonic()))
def run(unit,formal=False):
 name=unit['id'];gpu=unit['gpu'];proc=None;child=None;log=None;dispatch=None;last_alive=None;gpu_first=None;gpu_last=None;exit_seen=None;release_seen=None;last_gpu_owned=None
 try:
  C.require(not STOP.is_set(),'Global STOP prevents dispatch');C.device(gpu);C.storage_guard(R);target=R/('formal' if formal else 'execution')/name;target.mkdir(parents=True,exist_ok=True)
  C.require(not (target/'DISPATCH.json').exists(),'Unit already dispatched; no retry')
  C.atomic_new(target/'DISPATCH.json',dict(unit=name,controller_identity=identity,permit_sha256=C.sha(permit),gpu=gpu,gpu_uuid=C.UUIDS[gpu],formal=formal))
  log=(target/'worker.log').open('x');dispatch=time.monotonic();cmd=[sys.executable,'-u','-B',str(R/'scripts/gpu_unit_worker.py'),'--root',str(R),'--unit',name]+(['--formal'] if formal else [])
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=C.worker_environment(gpu),start_new_session=True);child=C.process_identity(proc.pid)
  with lock:all_children.append(child)
  C.event(R/'GPU_RESOURCE_LEDGER.jsonl','DISPATCH',unit=name,gpu=gpu,identity=child,formal=formal);last_alive=dispatch;last_progress=dispatch;signature=None;lastpoll=0
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
  exit_seen=time.monotonic();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_PROCESS_EXIT',unit=name,lower=last_alive,upper=exit_seen,returncode=proc.returncode)
  C.require(proc.returncode==0,'Unit worker exited unsuccessfully; inspect immutable log, no retry')
  out=C.read(target/'RESULT.json');C.require(out['identity']==child and out['status']=='PASS' and out['permit_sha256']==C.sha(permit),'Process/result/permit mismatch')
  if formal:validate(out,unit,R)
  with lock:complete.append(name)
 except BaseException as exc:first_failure(exc,name);raise
 finally:
  if proc is not None:
   if proc.poll() is None:
    C.require(child is not None and C.same_process(child),'Cleanup process identity differs; never kill foreign PID');proc.terminate()
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
     C.require(C.same_process(child),'Own PID changed before SIGKILL');proc.kill();proc.wait(timeout=15)
   if exit_seen is None:
    exit_seen=time.monotonic();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_PROCESS_EXIT',unit=name,lower=last_alive or dispatch,upper=exit_seen,returncode=proc.returncode)
   began=time.monotonic()
   while time.monotonic()-began<120:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==proc.pid];now=time.monotonic()
    if not own:release_seen=now;C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_OWN_UUID_RELEASE',unit=name,lower=last_gpu_owned or dispatch,upper=now,all_UUIDs_checked=True);break
    C.require(all(x['uuid']==C.UUIDS[gpu] for x in own),'Own process used unauthorized GPU');last_gpu_owned=now;time.sleep(.5)
   if log:log.close()
   record=dict(unit=name,gpu=gpu,identity=child,exit_returncode=proc.returncode,release='PASS' if release_seen else 'UNRESOLVED',reservation_lower_seconds=max(0,(last_alive or dispatch)-dispatch),reservation_upper_seconds=(release_seen or time.monotonic())-dispatch,observed_occupancy_lower_seconds=max(0,gpu_last-gpu_first) if gpu_first is not None else 0,physical_occupancy_upper_seconds=(release_seen or time.monotonic())-dispatch)
   with lock:usage.append(record)
   C.atomic_new(target/'RESOURCE_INTERVAL.json',record);publish_ledger()
   if release_seen is None:first_failure(RuntimeError('GPU release unresolved in cleanup observation'),name)
def require_released():
 obs=C.query_gpu();C.require(not any(x['pid'] in {i['pid'] for i in all_children} for x in obs['apps']),'Own worker NVML not released');C.require(all(not C.same_process(i) for i in all_children),'Actual worker still alive');return obs
try:
 C.model_guard(R);C.atomic_new(R/'analysis/INHERITED_QUALIFICATION_ADMISSION.json',C.admit_inherited(R))
 jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];C.validate_jobs(R,jobs)
 C.atomic_new(R/'analysis/EXECUTION_EXTERNAL_SHA_ADMISSION.json',dict(status='PASS',bound_files=C.verify_external(R)))
 C.device(C.SELECTED_GPU)
 ref=C.reference_root();C.require(not ref.exists(),'New task RAM reference root already exists; no reuse');ref.mkdir();s=os.statvfs(ref);C.require(s.f_bavail*s.f_frsize>=100*1024**3,'100GiB initial RAM reference admission failed')
 for unit in plan['qualification_units']:
  run(unit,False);require_released()
 C.require(len(complete)==22,'New INT4 finite qualification roster incomplete')
 C.verify_sources(R);C.model_guard(R);obs=require_released()
 math_result=C.read(R/'execution/GPU3_INT4_CODEC_MATH/RESULT.json')['result']
 C.require(math_result['status']=='PASS_INDEPENDENT_EXACT_INT4_CODEC','GPU Codec qualification missing')
 C.atomic_new(R/'INT4_CPU_GPU_CODEC_QUALIFICATION.json',dict(status='PASS_BOTH_COORDINATES_SHARED_CANONICAL_INT4',GPU=math_result,CPU_independent=C.read(R/'qualification/INDEPENDENT_CPU_COMPARISON.json'),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
 C.atomic_new(R/'NATIVE_HADAMARD_ROTATION_AUDIT.json',dict(status='PASS_ORIGINAL_COORDINATES_REUSED_NATIVE_TRUE_FRESH_CONTROL',H_source_sha256=C.sha(R/'inherited/group32/src/canonical_h128.py'),H_original_matrix_sha256=math_result['H_original_sha256'],H_orthogonality=math_result['H_orthogonality_max_abs'],H_inverse=math_result['H_inverse_max_abs'],native_control_units=['GPU3_NATIVE_CONTROL_R0','GPU3_NATIVE_CONTROL_R1'],original_INT8_H_native_sources_unchanged=C.read(R/'analysis/UNCHANGED_PARENT_NUMERICAL_SOURCES.json'),conditions={'I4-N':'NATIVE','I4-H':'H128'}))
 C.atomic_new(R/'PERSISTENT_QDQ_CANARY.json',dict(status='PASS_32_128_512_FRESH_FULL_VALUES_BYTES_FREE64_AND4096',units=complete.copy(),short_next_consumption='Every token actual values and bytes; all24GDNlayers',long_next_consumption='Frozen sparse actual complete values/bytes, not all positions',persistent_dtype='FP32_decoded_QDQ',code_dtype='UINT8_0_TO_15',physical_INT4_packing=False,numerical_81920_bound=False))
 qualification=dict(status='PASS_ALL_SINGLE_GPU_INT4_FORMAL_GATES',units=complete.copy(),device=C.UUIDS[C.SELECTED_GPU],physical_index=C.SELECTED_GPU,all_own_processes_exited=True,all_own_NVML_released=True,observation=obs,required_context=82293,context_pressure=C.read(R/'execution/GPU3_INT4_CONTEXT82293/RESULT.json')['result'],source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'))
 C.atomic_new(R/'ONE_GPU_RUNTIME_QUALIFICATION.json',qualification)
 rates={c:C.read(R/'execution'/('GPU3_'+c+'_LONG4096')/'RESULT.json')['result']['timing'] for c in ['I4-N','I4-H']}
 measured_rates={c:min(v['short']['TPS'],v['late']['TPS']) for c,v in rates.items()}
 measured_load_seconds={c:C.read(R/'execution'/('GPU3_'+c+'_LONG4096')/'RESULT.json')['environment']['model_load_seconds'] for c in rates}
 estimate=dict(status='ACTUAL_INT4_NONAIME_RATES_CONDITIONAL_NOT_GUARANTEE',model_load_seconds=measured_load_seconds,rates=rates,all60_cap_new_tokens=60*81920,all60_cap_generation_hours=sum(30*81920/t for t in measured_rates.values())/3600,limitations='Teacher4096 with original sampler, four short/late windows; future free-generation lengths and81K speed unknown. Qualification/loading/prompt/postflight/archive separate. Not a hard completion bound or exactphysicalGPUh.')
 C.atomic_new(R/'RESOURCE_AND_COST_ESTIMATE_ACTUAL.json',estimate)
 with (R/'RESOURCE_AND_COST_ESTIMATE.md').open('x') as f:f.write('# 实测单GPU资源条件预测\n\n'+json.dumps(estimate,ensure_ascii=False,indent=2)+'\n\n同卡串行时，生成墙钟与生成占用GPU小时的估算数值接近；物理精确GPUh仍UNRESOLVED。实际加载、资格、退出释放见append-only账本，不能借用INT8吞吐当INT4实测。\n')
 C.validate_jobs(R,jobs)
 C.atomic_new(R/'AUTHORIZATION_FORMAL.json',dict(status='PASS_ALL_INT4_SINGLE_GPU_GATES_ALLOW_FORMAL_ONCE',source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),qualification_sha256=C.sha(R/'ONE_GPU_RUNTIME_QUALIFICATION.json'),permit_sha256=C.sha(permit),jobs_sha256=C.sha(R/'FORMAL_JOB_MANIFEST.json'),required_context=82293,max_new_tokens=81920))
 C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FORMAL_STAGE_START',samples=60)
 for unit in jobs:run(unit,True)
 obs=require_released();C.require(all(j['id'] in complete for j in jobs),'Formal60 incomplete')
 C.atomic_new(R/'FORMAL_GENERATIONS_COMPLETE.json',dict(status='PASS_60_LEGAL_GENERATIONS_PROCESSES_RELEASED',count=60,observation=obs,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
 C.atomic_new(R/'SESSION_RESULT.json',dict(status='GPU_COMPLETE_22_QUALIFICATION_60_FORMAL',completed=complete,own_all_UUID_release=True,final_observation=obs))
except BaseException as exc:
 first_failure(exc,'CONTROLLER');obs=require_released()
 C.atomic_new(R/'SESSION_RESULT.json',dict(status='FAIL_CLOSED_STOP',completed=complete,own_all_UUID_release=True,final_observation=obs));raise
finally:publish_ledger()
