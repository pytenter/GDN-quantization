"""Original finite remaining matrix, no GPU time budget; immutable parent reuse."""
import argparse,os,sys,json,time,subprocess,traceback
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--permit',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'))
import gpu_control as C
import continuation_control as K
source=C.verify_sources(R);plan=C.read(R/'GPU_PLAN.json')
if not a.execute:
 print(json.dumps(dict(status='CPU_CHECK_ONLY_GPU_NOT_AUTHORIZED',source_sha256=source,finite_units=len(plan['units']),GPU_queries=0,GPU_dispatches=0)));sys.exit(0)
C.require(a.permit is not None,'New continuation permit required');permit=C.authorize(R,a.permit)
C.require(not (R/'execution/CONTROLLER_STARTED.json').exists(),'Already dispatched; no relaunch')
inherited,parent_admission=K.admit_parent(R,plan)
C.atomic_new(R/'execution/PARENT_ADMISSION.json',parent_admission)
external=C.verify_external(R)
from inherited_archive_transport import SSHArchiveTransport
archive_config=dict(host='zypan@202.38.247.29',control_path='/tmp/qwen_replication_archive29.sock',remote_python='/usr/bin/python3',archive_root=plan['archive_root'],card_key='single_gpu0',required_quota_bytes=plan['archive_max_bytes'],global_reservation_bytes=plan['archive_max_bytes'],global_headroom_bytes=plan['archive_headroom_bytes'],io_timeout_seconds=120)
archive_admission=SSHArchiveTransport(archive_config).preflight()
C.atomic_new(R/'execution/ARCHIVE_ADMISSION.json',archive_admission)
obs=C.query_gpu();C.validate_gpu(obs)
identity=C.process_identity(os.getpid());start=time.monotonic()
C.atomic_new(R/'execution/CONTROLLER_STARTED.json',dict(identity=identity,start=start,resource_policy='NO_GPU_TIME_BUDGET',no_progress_watchdog_seconds=1800,external_bindings=external))
ledger=C.Ledger(R/'execution/RESOURCE_EVENTS.jsonl');ledger.event('SESSION_START',source_sha256=source,resource_policy='NO_GPU_TIME_BUDGET')
complete=dict(inherited);new_complete={};stop=None;usage=[];dispatched=[];proc=None;child=None;current=None

def close_current():
 global proc,child,current
 if current is None or current['recorded']:return
 name=current['name'];dispatch=current['dispatch'];last_alive=current['last_alive']
 if proc.poll() is None:
  C.require(C.same_process(child),'Own process identity changed; refusing foreign cleanup')
  C.terminate_exact(proc,child)
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:
   C.require(C.same_process(child),'Own process identity changed before kill');proc.kill();proc.wait(timeout=15)
 exit_seen=time.monotonic();ledger.once((name,'exit'),'first_process_exit',unit=name,lower=last_alive,upper=exit_seen,returncode=proc.returncode)
 last_owned=dispatch;release_start=time.monotonic();released=False
 while time.monotonic()-release_start<plan['cleanup_seconds']:
  obs=C.query_gpu();own=[x for x in obs['apps'] if x['pid']==child['pid']]
  C.require(not any(x['uuid']!=C.UUID for x in own),'Unauthorized UUID owner process')
  now=time.monotonic()
  if not own:
   ledger.once((name,'release'),'first_owned_resource_release',unit=name,lower=last_owned,upper=now,all_UUIDs_checked=True);released=True;break
  last_owned=now;time.sleep(.5)
 current['log'].close()
 record=dict(unit=name,kind=current['unit']['kind'],tokens=current['unit']['tokens'],seconds=now-dispatch,identity=child,exit_returncode=proc.returncode,release='PASS' if released else 'UNRESOLVED',**K.reservation(dispatch,last_alive,now))
 usage.append(record);current['recorded']=True
 C.atomic_new(R/'execution'/name/'RESOURCE_INTERVAL.json',record)
 C.require(released,'Own GPU release unresolved during cleanup observation')

try:
 for unit in plan['units']:
  name=unit['id'];C.require(name not in complete,'Attempt to repeat completed unit')
  C.require(K.ready(unit,complete),'Original finite prior-stage gate missing:'+name)
  C.validate_gpu(C.query_gpu())
  target=R/'execution'/name;target.mkdir(parents=True,exist_ok=True)
  C.atomic_new(target/'DISPATCH.json',dict(unit=name,controller_identity=identity,permit_sha256=C.sha(a.permit),resource_policy='NO_GPU_TIME_BUDGET',execution_run_id=permit['execution_run_id']))
  log=(target/'worker.log').open('x');dispatch=time.monotonic()
  cmd=[sys.executable,'-B',str(R/'scripts/gpu_worker.py'),'--root',str(R),'--permit',str(a.permit),'--unit',name]
  proc=subprocess.Popen(cmd,env=C.worker_environment(dict(id=name)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  child=C.process_identity(proc.pid);dispatched.append(child);ledger.event('dispatch',unit=name,identity=child)
  current=dict(name=name,unit=unit,dispatch=dispatch,last_alive=dispatch,recorded=False,log=log)
  signature=K.progress_signature(target);last_progress=dispatch;last_gpu_poll=dispatch
  while proc.poll() is None:
   now=time.monotonic();current['last_alive']=now
   new=K.progress_signature(target)
   if new!=signature:
    signature=new;last_progress=now;ledger.event('REAL_PHASE_OR_TOKEN_PROGRESS',unit=name,signature=new)
   C.require(not K.stalled(last_progress,now,plan['no_progress_watchdog_seconds']),'NO_PROGRESS_WATCHDOG_STOP:'+name)
   if now-last_gpu_poll>=15:
    C.validate_gpu(C.query_gpu(),own_pid=child['pid']);last_gpu_poll=now
   time.sleep(.5)
  close_current()
  C.require(proc.returncode==0,'Worker failed: inspect immutable evidence; no retry')
  result=C.read(target/'RESULT.json');C.require(result['identity']==child and result['permit_sha256']==C.sha(a.permit),'Result process/permit identity')
  C.require(result['status']=='PASS','Unit not PASS')
  complete[name]=result;new_complete[name]=result
except BaseException as exc:
 stop='FAIL_CLOSED_STOP';C.atomic_new(R/'execution/FIRST_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),unit=current['name'] if current else None))
finally:
 try:close_current()
 except BaseException as exc:
  stop='FAIL_CLOSED_STOP'
  if not (R/'execution/CLEANUP_FAILURE.json').exists():C.atomic_new(R/'execution/CLEANUP_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
 try:
  final_observation=C.query_gpu();release=not any(x['pid'] in {p['pid'] for p in dispatched} for x in final_observation['apps'])
 except BaseException:final_observation=None;release='UNRESOLVED'
 if release is not True:stop='FAIL_CLOSED_STOP'
 ledger.event('SESSION_TERMINAL',stop=stop,completed=list(new_complete))
 C.atomic_new(R/'execution/SESSION_RESULT.json',dict(status=stop or 'FINITE_ROSTER_TERMINAL',completed=list(new_complete),inherited_complete=list(inherited),skipped=[u['id'] for u in plan['units'] if u['id'] not in new_complete],usage=usage,own_all_UUID_release=release,final_observation=final_observation,EXACT_GPUh='UNRESOLVED',reservation_GPUh_lower=sum(x['reservation_lower_seconds'] for x in usage)/3600,reservation_GPUh_upper=sum(x['reservation_upper_seconds'] for x in usage)/3600,cleanup_usage_incomplete=len(usage)!=len(dispatched),resource_policy='NO_GPU_TIME_BUDGET'))
sys.exit(0 if stop is None else 1)
