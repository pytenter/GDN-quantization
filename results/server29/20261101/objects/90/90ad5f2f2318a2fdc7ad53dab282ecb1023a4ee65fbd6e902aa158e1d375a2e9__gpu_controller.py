"""One finite V2 coordinator: three serial UUID queues and independent branches."""
import sys,os,argparse,time,subprocess,threading,traceback,concurrent.futures,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
from record_validation import validate
plan=C.read(R/'GPU_EXECUTION_PLAN.json');jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];CONDITIONS=('P1','C1','S3')
if not a.execute:print(json.dumps(dict(status='CPU_CHECK_ONLY',units=len(plan['qualification_units']),formal=90)));raise SystemExit
permit=R/'GPU_PERMIT.json';C.authorize(R,permit);C.validate_jobs(R,jobs);C.require(not (R/'GPU_RUN_STARTED.json').exists(),'Already dispatched; never restart/reset coordinator')
identity=C.process_identity(os.getpid());C.atomic_new(R/'GPU_RUN_STARTED.json',dict(identity=identity,monotonic=time.monotonic(),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
REFERENCE_PAIR_LOCK=threading.Lock()
LOCK=threading.RLock();GLOBAL_STOP=threading.Event();usage=[];complete=[];children=[];failure_history=[]
u_cpu=C.read(R/'branches/U_CPU_ELIGIBILITY.json') if (R/'branches/U_CPU_ELIGIBILITY.json').exists() else dict(eligible=False)
alive_branch={'P1':True,'C1':bool(u_cpu['eligible']),'S3':bool(u_cpu['eligible'])};eligible={c:False for c in CONDITIONS}
C.event(R/'GPU_RESOURCE_LEDGER.jsonl','SESSION_START',identity=identity,resource_policy='NO_GPU_TIME_LIMITS',branch_policy='P1_INDEPENDENT_OF_U')
def branches_for(unit):
 if unit.get('kind') in ('fp_rotation_equivalence','numerical_diagnostics'):return ('C1','S3')
 if unit.get('kind')=='codec_math':return CONDITIONS
 return (unit['condition'],)
def record_failure(exc,unit,scope=None):
 scope=tuple(scope or branches_for(unit));failure=dict(unit=unit['id'],conditions=list(scope),error=repr(exc),traceback=traceback.format_exc(),monotonic=time.monotonic())
 with LOCK:
  failure_history.append(failure)
  for c in scope:
   alive_branch[c]=False;eligible[c]=False
   p=R/'branches'/('FIRST_FAILURE_'+c+'.json')
   if not p.exists():C.atomic_new(p,dict(status='FAIL_CLOSED_BRANCH_NO_RETRY',**failure))
  if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='FIRST_BRANCH_FAILURE_PRESERVED_OTHER_LEGAL_BRANCHES_MAY_CONTINUE',**failure))
 C.event(R/'GPU_RESOURCE_LEDGER.jsonl','BRANCH_FAIL_CLOSED',**failure)
def scope_alive(unit):
 return not GLOBAL_STOP.is_set() and any(alive_branch[c] for c in branches_for(unit))
def publish():
 with LOCK:
  C.status(R/'RESOURCE_LEDGER.json',dict(resource_policy='NO_GPU_TIME_LIMITS',intervals=usage,completed_units=complete,reservation_GPUh_lower=sum(x['reservation_lower_seconds'] for x in usage)/3600,reservation_GPUh_upper=sum(x['reservation_upper_seconds'] for x in usage)/3600,observed_physical_occupancy_lower_GPUh=sum(x['observed_occupancy_lower_seconds'] for x in usage)/3600,EXACT_physical_GPUh='UNRESOLVED'))
  counts={c:sum(j['id'] in complete for j in jobs if j['condition']==c) for c in CONDITIONS}
  C.status(R/'FORMAL_PROGRESS.json',dict(task_id=C.TASK,completed=sum(counts.values()),total=90,completed_by_condition=counts,eligible=eligible,branch_alive=alive_branch,active_own_identities=[x for x in children if C.same_process(x)],failure_history=failure_history,monotonic=time.monotonic()))
def run(unit,formal=False):
 name=unit['id'];gpu=unit['gpu'];target=R/('formal' if formal else 'execution')/name;proc=None;child=None;log=None;dispatch=time.monotonic();last_alive=dispatch;first_gpu=None;last_gpu=None;last_gpu_owned=None;released=None;exit_seen=None;success=False
 try:
  C.require(scope_alive(unit),'Branch already failed before dispatch');C.device(gpu);C.storage_guard(R);target.mkdir(parents=True,exist_ok=True);C.require(not (target/'DISPATCH.json').exists(),'Unit was dispatched; never retry it')
  C.atomic_new(target/'DISPATCH.json',dict(unit=name,controller_identity=identity,permit_sha256=C.sha(permit),gpu=gpu,gpu_uuid=C.UUIDS[gpu],formal=formal,scope=list(branches_for(unit))))
  command=[sys.executable,'-u','-B',str(R/'scripts/gpu_unit_worker.py'),'--root',str(R),'--unit',name]+(['--formal'] if formal else []);log=(target/'worker.log').open('x');dispatch=time.monotonic();proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=C.worker_environment(gpu),start_new_session=True);child=C.process_identity(proc.pid)
  with LOCK:children.append(child)
  C.event(R/'GPU_RESOURCE_LEDGER.jsonl','DISPATCH',unit=name,gpu=gpu,identity=child,formal=formal);last_alive=dispatch;last_progress=dispatch;signature=None;last_poll=0
  while proc.poll() is None:
   now=time.monotonic();last_alive=now;p=target/'PROGRESS.json';sig=p.read_bytes() if p.exists() else None
   if sig!=signature:signature=sig;last_progress=now
   C.require(scope_alive(unit),'First failure of this branch stops its own active unit')
   C.require(now-last_progress<1800,'1800s without actual phase/token progress')
   if now-last_poll>=15:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==child['pid']]
    if own:C.device(gpu,child['pid']);first_gpu=now if first_gpu is None else first_gpu;last_gpu=now;last_gpu_owned=now
    else:C.require(not any(x['uuid']==C.UUIDS[gpu] for x in obs['apps']),'Foreign process appeared on assigned UUID; never signal it')
    C.storage_guard(R);last_poll=now
   time.sleep(.5)
  exit_seen=time.monotonic();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_PROCESS_EXIT',unit=name,identity=child,lower=last_alive,upper=exit_seen,returncode=proc.returncode)
  C.require(proc.returncode==0,'Worker failed; preserve real first difference and no GPU retry')
  wrapper=C.read(target/'RESULT.json');C.require(wrapper['identity']==child and wrapper['status']=='PASS' and wrapper['permit_sha256']==C.sha(permit),'Result/process/source permit differs')
  if formal:validate(wrapper,unit,R)
  success=True
 except BaseException as exc:record_failure(exc,unit)
 finally:
  if proc is not None:
   if proc.poll() is None:
    C.require(C.same_process(child),'Cleanup own PID/starttick/cmdline differs; never signal foreign process');proc.terminate()
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:C.require(C.same_process(child),'Own identity changed before SIGKILL');proc.kill();proc.wait(timeout=15)
   if exit_seen is None:exit_seen=time.monotonic();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_PROCESS_EXIT',unit=name,identity=child,lower=last_alive,upper=exit_seen,returncode=proc.returncode)
   began=time.monotonic()
   while time.monotonic()-began<120:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==child['pid']];now=time.monotonic()
    if not own:released=now;C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_OWN_UUID_RELEASE',unit=name,identity=child,lower=last_gpu_owned or dispatch,upper=now);break
    C.require(all(x['uuid']==C.UUIDS[gpu] for x in own),'Own process consumed unauthorized UUID');last_gpu_owned=now;time.sleep(.5)
   if log:log.close()
   interval=dict(unit=name,identity=child,gpu=gpu,condition=unit.get('condition'),formal=formal,exit_returncode=proc.returncode,release='PASS' if released else 'UNRESOLVED',reservation_lower_seconds=max(0,last_alive-dispatch),reservation_upper_seconds=(released or time.monotonic())-dispatch,observed_occupancy_lower_seconds=max(0,last_gpu-first_gpu) if first_gpu is not None else 0,physical_occupancy_upper_seconds=(released or time.monotonic())-dispatch)
   C.atomic_new(target/'RESOURCE_INTERVAL.json',interval)
   with LOCK:
    usage.append(interval)
    if success and released:complete.append(name)
   if released is None:success=False;record_failure(RuntimeError('Own GPU release unresolved after120s observation'),unit);GLOBAL_STOP.set()
  publish()
 return success
def reference_pair_admission(unit):
 # Global lock is acquired BEFORE a worker/model exists. Keep it through R0/R1
 # and paired cleanup; waiting threads consume no GPU model or RAM references.
 C.require(unit.get('repeat')==0 and unit.get('reference_unit')==unit['id'],'Expected exact reference R0 identity')
 st=os.statvfs(C.reference_root());free=st.f_bavail*st.f_frsize
 C.require(free>=100*1024**3,'Global reference pair requires80GiB reservation plus20GiB physical margin')
 C.event(R/'GPU_RESOURCE_LEDGER.jsonl','REFERENCE_PAIR_GLOBAL_ADMISSION',unit=unit['id'],free_bytes=free,reservation_bytes=80*1024**3,margin_bytes=20*1024**3)
def queues(units,formal=False):
 def serial(g):
  done=set()
  for u in units:
   if u['id'] in done or u['gpu']!=g or not scope_alive(u):continue
   if formal and not eligible[u['condition']]:continue
   if not formal and u.get('reference_unit') is not None:
    C.require(u.get('repeat')==0,'Reference R1 must only dispatch within its admitted pair')
    mates=[v for v in units if v['gpu']==g and v.get('reference_unit')==u['id'] and v.get('repeat')==1]
    C.require(len(mates)==1,'Every admitted reference must have exactly one frozen R1')
    mate=mates[0]
    with REFERENCE_PAIR_LOCK:
     if not scope_alive(u):continue
     try:reference_pair_admission(u)
     except BaseException as exc:
      record_failure(exc,u);done.update([u['id'],mate['id']]);continue
     first=run(u,formal)
     if first and scope_alive(mate):run(mate,formal)
     done.update([u['id'],mate['id']])
     C.event(R/'GPU_RESOURCE_LEDGER.jsonl','REFERENCE_PAIR_GLOBAL_RELEASE',unit=u['id'],paired_reference_still_exists=(C.reference_root()/u['id']).exists())
   else:run(u,formal)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(serial,g) for g in C.SELECTED_GPUS]
  for f in concurrent.futures.as_completed(fs):
   try:f.result()
   except BaseException as exc:
    GLOBAL_STOP.set();record_failure(exc,dict(id='GLOBAL_QUEUE_ERROR',kind='codec_math'),CONDITIONS)
def released_observation():
 obs=C.query_gpu();C.require(all(not C.same_process(i) for i in children),'An exact own worker is still alive');C.require(not any(x['pid'] in {i['pid'] for i in children} for x in obs['apps']),'Own historical worker PID still occupies NVML');return obs
try:
 C.model_guard(R);C.atomic_new(R/'analysis/INHERITED_QUALIFICATION_ADMISSION.json',C.admit_inherited(R));C.atomic_new(R/'analysis/EXECUTION_EXTERNAL_FRESH_SHA_ADMISSION.json',dict(status='PASS',files=C.verify_external(R)))
 for g in C.SELECTED_GPUS:C.device(g)
 for x in C.read(R/'P1_EXECUTABLE_IDENTITY.json')['files']:C.require(C.sha(R/x['path'])==x['sha256'],'Independent immutable P1 executable source differs')
 ref=C.reference_root();C.require(not ref.exists(),'New task reference RAM root already exists');ref.mkdir();stat=os.statvfs(ref);C.require(stat.f_bavail*stat.f_frsize>=100*1024**3,'100GiB initial reference RAM admission failed')
 units=plan['qualification_units'];queues([u for u in units if u['kind']=='codec_math']);released_observation()
 # P1 owns no U/FP gate. Its eligibility survives every U-family failure.
 queues([u for u in units if u.get('condition')=='P1']);released_observation()
 p1_required=[u['id'] for u in units if u.get('condition')=='P1']+[u['id'] for u in units if u['kind']=='codec_math']
 eligible['P1']=alive_branch['P1'] and all(i in complete for i in p1_required)
 if u_cpu['eligible'] and (alive_branch['C1'] or alive_branch['S3']):queues([u for u in units if u['kind']=='fp_rotation_equivalence']);released_observation()
 fp_required=[u['id'] for u in units if u['kind']=='fp_rotation_equivalence'];fp_pass=all(i in complete for i in fp_required)
 if not fp_pass:
  for c in ('C1','S3'):alive_branch[c]=False
 for c in ('C1','S3'):
  if alive_branch[c]:queues([u for u in units if u.get('condition')==c]);released_observation()
  required=[u['id'] for u in units if u.get('condition')==c]+fp_required+[u['id'] for u in units if u['kind']=='codec_math'];eligible[c]=alive_branch[c] and u_cpu['eligible'] and all(i in complete for i in required)
 if eligible['C1'] and eligible['S3']:queues([u for u in units if u['kind']=='numerical_diagnostics']);released_observation()
 C.verify_sources(R);C.model_guard(R);obs=released_observation()
 fp={str(g):C.read(R/'execution'/('GPU'+str(g)+'_FP_EQ')/'RESULT.json')['result'] for g in C.SELECTED_GPUS if (R/'execution'/('GPU'+str(g)+'_FP_EQ')/'RESULT.json').exists()}
 if fp:C.atomic_new(R/'REAL_9B_FP_ROTATION_EQUIVALENCE.json',dict(status='PASS_ALL3_ACTUAL_9B_FP' if fp_pass else 'FAIL_FP_U_BRANCH_ONLY',devices=fp,all_devices_pass=fp_pass))
 diag=R/'execution/GPU1_HELDOUT_NUMERICAL_DIAGNOSTICS/RESULT.json'
 if diag.exists():C.atomic_new(R/'CALIBRATION_NUMERICAL_DIAGNOSTICS.json',C.read(diag)['result'])
 qualification=dict(status='ALL_BRANCH_GATES_EVALUATED',eligible=eligible,branch_alive=alive_branch,completed_qualification=[x for x in complete if not x.startswith('aime26_')],failure_history=failure_history,own_all_processes_released=True,observation=obs,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),P1_U_independent=True)
 C.atomic_new(R/'P1_C1_S3_GPU_QUALIFICATION.json',qualification)
 rates={str(g):{} for g in C.SELECTED_GPUS}
 for u in units:
  p=R/'execution'/u['id']/'RESULT.json'
  if u['kind']=='nonAIME_persistent_long' and p.exists():rates[str(u['gpu'])][u['condition']]=C.read(p)['result']
 scenarios={}
 for n in (8192,20000,40000,81920):
  per={str(g):sum(n/min(rates[str(g)][j['condition']]['timing']['short']['TPS'],rates[str(g)][j['condition']]['timing']['late']['TPS']) for j in jobs if j['gpu']==g and eligible[j['condition']]) for g in C.SELECTED_GPUS};scenarios[str(n)]=dict(per_GPU_seconds=per,parallel_generation_wall_hours=max(per.values())/3600,occupancy_GPUh_estimate=sum(per.values())/3600)
 C.atomic_new(R/'RESOURCE_AND_COST_ESTIMATE_ACTUAL.json',dict(status='ACTUAL_NEW_ROTATION_NONAIME_TEACHER4096_WITH_SAMPLER_CONDITIONAL',rates=rates,conditional_mean_tokens=scenarios,not_81K_or_free_AIME_speed_guarantee=True,EXACT_physical_GPUh='UNRESOLVED'))
 for c in CONDITIONS:
  if eligible[c]:C.atomic_new(R/'branches'/('AUTHORIZATION_FORMAL_'+c+'.json'),dict(status=c+'_ELIGIBLE_ALLOW_ORIGINAL30_ONCE',condition=c,eligible=True,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(permit),jobs_sha256=C.sha(R/'FORMAL_JOB_MANIFEST.json'),qualification_sha256=C.sha(R/'P1_C1_S3_GPU_QUALIFICATION.json'),P1_executable_sha256=C.sha(R/'P1_EXECUTABLE_IDENTITY.json') if c=='P1' else None,U_manifest_sha256=C.sha(R/'CANONICAL_UH_MATRICES_AND_SHA.json') if c!='P1' else None))
 C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FORMAL_STAGE_START',eligible=eligible);publish();queues(jobs,True);obs=released_observation();counts={c:sum(j['id'] in complete for j in jobs if j['condition']==c) for c in CONDITIONS}
 C.atomic_new(R/'FORMAL_GENERATIONS_COMPLETE.json',dict(status='LEGAL_GENERATIONS_PROCESSES_RELEASED',count=sum(counts.values()),counts=counts,eligible_branches=eligible,own_UUID_release=obs))
 C.atomic_new(R/'SESSION_RESULT.json',dict(status='GPU_COMPLETE_90' if all(v==30 for v in counts.values()) else 'P1_COMPLETE_U_BRANCH_FAILED' if counts['P1']==30 and not (counts['C1']==30 and counts['S3']==30) else 'PARTIAL_RESULTS_PRESERVED',counts=counts,completed=complete,failures=failure_history,own_all_UUID_release=True,final_observation=obs))
except BaseException as exc:
 GLOBAL_STOP.set();record_failure(exc,dict(id='CONTROLLER',kind='codec_math'),CONDITIONS);obs=released_observation();counts={c:sum(j['id'] in complete for j in jobs if j['condition']==c) for c in CONDITIONS}
 if not (R/'FORMAL_GENERATIONS_COMPLETE.json').exists():C.atomic_new(R/'FORMAL_GENERATIONS_COMPLETE.json',dict(status='PARTIAL_LEGAL_PAYLOADS_PROCESSES_RELEASED',count=sum(counts.values()),counts=counts,eligible_branches=eligible,own_UUID_release=obs))
 C.atomic_new(R/'SESSION_RESULT.json',dict(status='PARTIAL_RESULTS_PRESERVED',counts=counts,completed=complete,failures=failure_history,own_all_UUID_release=True,final_observation=obs));raise
finally:publish()
