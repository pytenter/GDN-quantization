"""Only one calibration dispatch. No retries, no signal to another task."""
import sys,os,time,subprocess,json,traceback
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
T=R/'calibration/execution';T.mkdir(exist_ok=True);identity=C.process_identity(os.getpid())
C.require(not (R/'CALIBRATION_RUN_STARTED.json').exists(),'Calibration already started; never reset/relaunch')
C.atomic_new(R/'CALIBRATION_RUN_STARTED.json',dict(identity=identity,monotonic=time.monotonic(),source_freeze_sha256=C.verify_sources(R)))
proc=None;child=None;start=time.monotonic();last_alive=start;last_gpu=None;first_gpu=None;released=None
try:
 C.model_guard(R);C.device(1);C.storage_guard(R)
 C.atomic_new(R/'analysis/CALIBRATION_EXTERNAL_FRESH_SHA.json',dict(status='PASS',actual_dependencies=C.verify_external(R)))
 model=[]
 for row in C.read(R/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
  C.require(C.sha(Path(row['path']))==row['sha256'],'Model fresh SHA differs:'+row['path']);model.append(row)
 C.atomic_new(R/'analysis/CALIBRATION_MODEL_FRESH_SHA.json',dict(status='PASS_17_ACTUAL_MODEL_TOKENIZER_SHA',files=model))
 C.atomic_new(R/'THREE_GPU_UUID_AUDIT.json',dict(status='PASS_THREE_AUTHORIZED_IDLE_3090',selected=[1,2,3],observations={str(g):C.device(g) for g in C.SELECTED_GPUS},other_GPU0='Observed only; old task protected; never selected'))
 log=(T/'worker.log').open('x');proc=subprocess.Popen([sys.executable,'-u','-B',str(R/'scripts/collect_query_moments.py'),str(R)],stdout=log,stderr=subprocess.STDOUT,env=C.worker_environment(1),start_new_session=True);child=C.process_identity(proc.pid)
 C.atomic_new(T/'DISPATCH.json',dict(controller_identity=identity,worker_identity=child,gpu=1,gpu_uuid=C.UUIDS[1],source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')));C.event(R/'GPU_RESOURCE_LEDGER.jsonl','CALIBRATION_DISPATCH',identity=child,gpu=1)
 previous=None;last_progress=time.monotonic();last_poll=0
 while proc.poll() is None:
  now=time.monotonic();last_alive=now;p=T/'PROGRESS.json';sig=p.read_bytes() if p.exists() else None
  if sig!=previous:previous=sig;last_progress=now
  C.require(now-last_progress<1800,'1800s no actual phase/token progress')
  if now-last_poll>15:
   obs=C.query_gpu();own=[a for a in obs['apps'] if a['pid']==proc.pid]
   if own:C.device(1,proc.pid);first_gpu=now if first_gpu is None else first_gpu;last_gpu=now
   else:C.require(not [a for a in obs['apps'] if a['uuid']==C.UUIDS[1]],'Foreign GPU1 process appeared; never stop it')
   C.storage_guard(R);last_poll=now
  time.sleep(.5)
 C.event(R/'GPU_RESOURCE_LEDGER.jsonl','CALIBRATION_FIRST_EXIT',identity=child,lower=last_alive,upper=time.monotonic(),returncode=proc.returncode)
 C.require(proc.returncode==0,'Calibration failed; preserve original worker log and no retry')
 C.require(C.read(R/'QUERY_SECOND_MOMENT_AUDIT.json')['status']=='PASS_ACTUAL_NATIVE_READOUT_QUERY_FP64_SECOND_MOMENT','Missing native query collection receipt')
except BaseException as exc:
 if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),unit='CALIBRATION'))
 raise
finally:
 if proc is not None:
  if proc.poll() is None:
   C.require(C.same_process(child),'Own calibration process identity changed');proc.terminate()
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:C.require(C.same_process(child),'Own identity changed before SIGKILL');proc.kill();proc.wait(timeout=15)
  began=time.monotonic()
  while time.monotonic()-began<120:
   own=[a for a in C.query_gpu()['apps'] if a['pid']==child['pid']]
   if not own:released=time.monotonic();break
   C.require(all(a['uuid']==C.UUIDS[1] for a in own),'Own calibration used unauthorized GPU');time.sleep(.5)
  record=dict(unit='QUERY_CALIBRATION_4096',identity=child,gpu=1,exit_returncode=proc.returncode,release='PASS' if released else 'UNRESOLVED',reservation_lower_seconds=max(0,last_alive-start),reservation_upper_seconds=(released or time.monotonic())-start,observed_physical_occupancy_lower_seconds=max(0,last_gpu-first_gpu) if first_gpu is not None else 0,EXACT_physical_GPUh='UNRESOLVED')
  C.atomic_new(T/'RESOURCE_INTERVAL.json',record);C.event(R/'GPU_RESOURCE_LEDGER.jsonl','CALIBRATION_RELEASE',**record)
  C.require(released is not None,'Own calibration release unresolved')
  if proc.returncode==0:C.atomic_new(R/'CALIBRATION_SESSION_RESULT.json',dict(status='PASS_COLLECTION_EXIT_AND_RELEASE',record=record))
