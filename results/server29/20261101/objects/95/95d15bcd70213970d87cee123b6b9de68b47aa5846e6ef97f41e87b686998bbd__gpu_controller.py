"""Finite new task runner; --check/default NO cannot import/query GPU APIs."""
import argparse,os,sys,json,time,subprocess,traceback
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--permit',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'))
import gpu_control as C
source=C.verify_sources(R);plan=C.read(R/'GPU_PLAN.json')
if not a.execute:
 print(json.dumps(dict(status='CPU_CHECK_ONLY_GPU_NOT_AUTHORIZED',source_sha256=source,finite_units=len(plan['units']),GPU_queries=0,GPU_dispatches=0)));sys.exit(0)
C.require(a.permit is not None,'New task permit required');permit=C.authorize(R,a.permit)
C.require(not (R/'execution/CONTROLLER_STARTED.json').exists(),'Already dispatched; no relaunch')
# CPU hashing precedes GPU query and never loads model weights.
external=C.verify_external(R)
from inherited_archive_transport import SSHArchiveTransport
archive_config=dict(host='zypan@202.38.247.29',control_path='/tmp/qwen_replication_archive29.sock',remote_python='/usr/bin/python3',archive_root=plan['archive_root'],card_key='single_gpu0',required_quota_bytes=plan['archive_max_bytes'],global_reservation_bytes=plan['archive_max_bytes'],global_headroom_bytes=plan['archive_headroom_bytes'],io_timeout_seconds=120)
archive_admission=SSHArchiveTransport(archive_config).preflight()
obs=C.query_gpu();C.validate_gpu(obs)
identity=C.process_identity(os.getpid());start=time.monotonic();protect=start+1650;hard=start+1800
C.atomic_new(R/'execution/CONTROLLER_STARTED.json',dict(identity=identity,start=start,protective_deadline=protect,hard_deadline=hard,external_bindings=external))
ledger=C.Ledger(R/'execution/RESOURCE_EVENTS.jsonl');ledger.event('SESSION_START',source_sha256=source)
eligible={'B0':True,'B1':True,'B2':True};complete={};stop=None;usage=[];micro=None;proc=None;child=None
try:
 for unit in plan['units']:
  name=unit['id'];method=unit.get('method')
  if method and not eligible[method]:continue
  if unit['kind']!='micro':
   C.require(micro is not None,'Micro gate missing')
   if micro['conditions'][method]['status']!='GPU_CODEC_QUALIFIED':eligible[method]=False;continue
  if unit.get('repeat')==1 and unit['reference'] not in complete:continue
  if unit['kind']=='canary' and unit['tokens']>32:
   prior=32 if unit['tokens']==128 else 128
   if not all(f'{method}_{prior}_R{i}' in complete for i in (0,1)):continue
  if unit['kind']=='benchmark' and not all(f'{method}_512_R{i}' in complete for i in (0,1)):continue
  remaining=protect-time.monotonic()
  # Freeze conservative adaptive admission: most recent actual same-kind cost,
  # length proportional, plus one observed load cost allowance. No science results.
  estimate=plan['initial_admission_seconds'][unit['kind']]
  prior_cost=[x for x in usage if x['kind']=='canary']
  if prior_cost and unit['kind']=='canary':estimate=max(estimate,prior_cost[-1]['seconds']*unit['tokens']/prior_cost[-1]['tokens']+30)
  if remaining<estimate:stop='BUDGET_INSUFFICIENT_FOR_NEXT_LEGAL_UNIT';break
  target=R/'execution'/name;target.mkdir(parents=True,exist_ok=True)
  C.atomic_new(target/'DISPATCH.json',dict(unit=name,controller_identity=identity,permit_sha256=C.sha(a.permit),protective_deadline_monotonic=protect))
  log=(target/'worker.log').open('x');dispatch=time.monotonic()
  cmd=[sys.executable,'-B',str(R/'scripts/gpu_worker.py'),'--root',str(R),'--permit',str(a.permit),'--unit',name]
  proc=subprocess.Popen(cmd,env=C.worker_environment(dict(id=name)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  child=C.process_identity(proc.pid);ledger.event('dispatch',unit=name,identity=child)
  previous_alive=dispatch;last_obs=dispatch
  while proc.poll() is None:
   if time.monotonic()>=protect:
    if C.same_process(child):C.terminate_exact(proc,child)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
     if C.same_process(child):proc.kill()
     proc.wait(timeout=15)
    stop='GPU_PROTECTIVE_BUDGET_STOP';break
   previous_alive=time.monotonic();time.sleep(.5)
  exit_seen=time.monotonic();ledger.once((name,'exit'),'first_process_exit',unit=name,lower=previous_alive,upper=exit_seen,returncode=proc.returncode)
  while True:
   obs=C.query_gpu();owned=[x for x in obs['apps'] if x['pid']==child['pid']]
   C.require(not any(x['uuid']!=C.UUID for x in owned),'Unauthorized UUID owner process')
   now=time.monotonic()
   if not owned:
    ledger.once((name,'release'),'first_owned_resource_release',unit=name,lower=last_obs,upper=now,all_UUIDs_checked=True);break
   C.require(now<hard,'GPU release unresolved at hard deadline');last_obs=now;time.sleep(.5)
  log.close();seconds=now-dispatch
  usage.append(dict(unit=name,kind=unit['kind'],tokens=unit.get('tokens'),seconds=seconds,reservation_lower_seconds=max(0,previous_alive-dispatch),reservation_upper_seconds=seconds))
  if stop:break
  C.require(proc.returncode==0,'Worker failed: inspect immutable log/receipt; no retry')
  result=C.read(target/'RESULT.json');C.require(result['identity']==child and result['permit_sha256']==C.sha(a.permit),'Result process/permit identity')
  if unit['kind']=='micro':
   C.require(result['status']=='PASS_MICRO_TERMINAL','Micro not terminal');micro=result
  else:C.require(result['status']=='PASS','Unit not PASS')
  complete[name]=result
except BaseException as exc:
 stop='FAIL_CLOSED_STOP';C.atomic_new(R/'execution/FIRST_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
finally:
 if proc is not None and proc.poll() is None and child is not None and C.same_process(child):
  C.terminate_exact(proc,child)
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:
   if C.same_process(child):proc.kill()
   proc.wait(timeout=15)
  ledger.once((child['pid'],'cleanup_exit'),'cleanup_owned_process_exit',identity=child,monotonic_observed=time.monotonic())
 # A query failure leaves release UNRESOLVED, never assumes zero usage.
 try:
  final_observation=C.query_gpu();release=not any(x['pid']==child['pid'] for x in final_observation['apps']) if child else True
 except BaseException:final_observation=None;release='UNRESOLVED'
 ledger.event('SESSION_TERMINAL',stop=stop,completed=list(complete))
 C.atomic_new(R/'execution/SESSION_RESULT.json',dict(status=stop or 'FINITE_ROSTER_TERMINAL',completed=list(complete),skipped=[u['id'] for u in plan['units'] if u['id'] not in complete],usage=usage,own_all_UUID_release=release,final_observation=final_observation,EXACT_GPUh='UNRESOLVED',reservation_GPUh_lower=sum(x['reservation_lower_seconds'] for x in usage)/3600,reservation_GPUh_upper=sum(x['reservation_upper_seconds'] for x in usage)/3600,cleanup_usage_incomplete=bool(child and not any(x['unit']==name for x in usage))))
