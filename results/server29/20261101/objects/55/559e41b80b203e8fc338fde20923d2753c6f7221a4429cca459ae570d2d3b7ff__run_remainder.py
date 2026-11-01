"""One explicit GPU0 remainder attempt. Original science and old cancellation stay immutable."""
import argparse, os, sys, time, subprocess, traceback, signal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
R=a.root.resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
from record_validation import validate
import native_map_guards as G
jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];plan=C.read(R/'GPU_EXECUTION_PLAN.json')
if not a.execute:
 C.validate_jobs(R,jobs);print('PASS_CPU_REMAINDER_ROSTER');raise SystemExit
C.authorize(R,R/'GPU_PERMIT.json');C.validate_jobs(R,jobs)
identity=C.process_identity(os.getpid());C.atomic_new(R/'CONTROLLER_STARTED.json',dict(identity=identity,monotonic=time.monotonic()))
intervals=[];completed=[];failed=None
def publish(active=None):
 C.status(R/'PHASE_STATUS.json',dict(status='RUNNING' if failed is None else 'FAIL_CLOSED_STOP',formal_completed=len(completed),formal_total=15,preserved_original=15,active=active,monotonic=time.monotonic(),gold_blind=True))
def run(unit,formal):
 global failed
 target=R/('formal' if formal else 'execution')/unit['id'];proc=None;child=None;log=None;exit_time=None;released=None;last_alive=None;first_gpu=None;last_gpu=None;last_gpu_seen=None;dispatch=None;failure=None
 try:
  C.device(0);C.storage_guard(R)
  C.require(not os.path.lexists(target),'Already-existing unit leaf is never reused');target.mkdir()
  C.atomic_new(target/'DISPATCH.json',dict(unit=unit['id'],controller_identity=identity,permit_sha256=C.sha(R/'GPU_PERMIT.json'),gpu=0,gpu_uuid=C.UUIDS[0],formal=formal))
  log=(target/'worker.log').open('x');dispatch=time.monotonic()
  cmd=[sys.executable,'-u','-B',str(R/'scripts/gpu_unit_worker.py'),'--root',str(R),'--unit',unit['id']]+(['--formal'] if formal else [])
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=C.worker_environment(0),start_new_session=True);child=C.process_identity(proc.pid);identity_upper=time.monotonic()
  C.event(R/'GPU_RESOURCE_LEDGER.jsonl','DISPATCH',unit=unit['id'],identity=child,gpu_uuid=C.UUIDS[0],lower=dispatch,upper=identity_upper,formal=formal)
  last_alive=dispatch;last_progress=dispatch;signature=None;last_poll=0
  while proc.poll() is None:
   now=time.monotonic();last_alive=now;p=target/'PROGRESS.json';raw=p.read_bytes() if p.exists() else None
   if raw!=signature:signature=raw;last_progress=now
   C.require(now-last_progress<1800,'1800s without actual phase/token progress')
   if now-last_poll>=15:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==child['pid']]
    if own:
     C.device(0,child['pid']);first_gpu=now if first_gpu is None else first_gpu;last_gpu=now;last_gpu_seen=now
    else:C.require(not any(x['uuid']==C.UUIDS[0] for x in obs['apps']),'Foreign GPU0 process appeared; never signal it')
    C.storage_guard(R);publish(dict(unit=unit['id'],formal=formal,identity=child,progress=C.read(p) if p.exists() else None));last_poll=now
   time.sleep(.5)
  exit_time=time.monotonic();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_EXIT',unit=unit['id'],identity=child,actual_wait_returncode=proc.returncode,lower=last_alive,upper=exit_time)
  C.require(proc.returncode==0,'Real worker failure: no automatic retry')
  wrapper=C.read(target/'RESULT.json');C.require(wrapper['identity']==child and wrapper['status']=='PASS' and wrapper['source_freeze_sha256']==C.sha(R/'SOURCE_FREEZE.json') and wrapper['permit_sha256']==C.sha(R/'GPU_PERMIT.json'),'Actual worker receipt binding differs')
  if formal:validate(wrapper,unit,R)
 except BaseException as exc:
  failure=dict(unit=unit['id'],error=repr(exc),traceback=traceback.format_exc(),monotonic=time.monotonic())
 finally:
  if proc is not None:
   if proc.poll() is None:
    C.require(C.same_process(child),'Exact identity changed; refuse cleanup');proc.terminate()
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
     C.require(C.same_process(child),'Exact identity changed; refuse kill');proc.kill();proc.wait(timeout=15)
   if exit_time is None:
    exit_time=time.monotonic();C.event(R/'GPU_RESOURCE_LEDGER.jsonl','FIRST_EXIT',unit=unit['id'],identity=child,actual_wait_returncode=proc.returncode,lower=last_alive or dispatch,upper=exit_time)
   began=time.monotonic()
   while time.monotonic()-began<120:
    obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==child['pid']];now=time.monotonic()
    if not C.same_process(child) and not own:released=now;break
    C.require(all(x['uuid']==C.UUIDS[0] for x in own),'Own process on unauthorized UUID');last_gpu_seen=now;time.sleep(.5)
   log.close()
   interval=dict(unit=unit['id'],identity=child,gpu=0,gpu_uuid=C.UUIDS[0],formal=formal,exit_returncode=proc.returncode,release='PASS' if released else 'UNRESOLVED',monotonic_same_host=True,dispatch_monotonic_lower=dispatch,identity_monotonic_upper=identity_upper,exit_last_alive_monotonic=last_alive,exit_first_observed_monotonic=exit_time,release_first_observed_monotonic=released,reservation_lower_seconds=max(0,(last_alive or dispatch)-dispatch),reservation_upper_seconds=(released or time.monotonic())-dispatch,observed_occupancy_lower_seconds=max(0,last_gpu-first_gpu) if first_gpu is not None else 0,EXACT_physical_GPUh='UNRESOLVED')
   C.atomic_new(target/'RESOURCE_INTERVAL.json',interval);intervals.append(interval);C.event(R/'GPU_RESOURCE_LEDGER.jsonl','RELEASE_OBSERVED',interval=interval)
   if released is None:failure=failure or dict(unit=unit['id'],error='Own GPU release unresolved after120s')
  if failure is not None:
   failed=failure
   if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',**failure))
  elif formal:completed.append(unit['id'])
  C.status(R/'RESOURCE_LEDGER.json',dict(intervals=intervals,completed_units=completed,EXACT_physical_GPUh='UNRESOLVED'));publish()
 return failure is None
try:
 G.verify_model_and_external(R,fresh_external=True)
 C.atomic_new(R/'INITIAL_DEPENDENCY_ADMISSION.json',dict(status='PASS_FRESH61_PLUS293_AND17STAT',monotonic=time.monotonic()))
 C.device(0);C.storage_guard(R)
 if run(plan['qualification_units'][0],False):
  C.atomic_new(R/'branches/AUTHORIZATION_FORMAL_P1.json',dict(condition='P1',eligible=True,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(R/'GPU_PERMIT.json'),GPU0_P1_32_result_sha256=C.sha(R/'execution/GPU0_P1_32/RESULT.json'),qualification_exit_release_sha256=C.sha(R/'execution/GPU0_P1_32/RESOURCE_INTERVAL.json'),prior_P1_27_qualification_readonly=True))
  for job in jobs:
   if not run(job,True):break
except BaseException as exc:
 failed=dict(error=repr(exc),traceback=traceback.format_exc())
 if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',**failed))
finally:
 C.atomic_new(R/'SESSION_RESULT.json',dict(status='P1_REMAINDER_COMPLETE_15_15' if len(completed)==15 and failed is None else 'P1_REMAINDER_PARTIAL_FAIL_CLOSED_STOP',completed=completed,count=len(completed),preserved_original=15,combined_legal_count=15+len(completed),failure=failed,controller_identity=identity,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(R/'GPU_PERMIT.json'),monotonic=time.monotonic()))
raise SystemExit(0 if failed is None else 1)
