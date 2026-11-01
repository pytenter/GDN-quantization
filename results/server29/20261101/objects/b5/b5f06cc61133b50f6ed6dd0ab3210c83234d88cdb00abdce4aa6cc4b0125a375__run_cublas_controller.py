#!/usr/bin/env python3
"""Prepared controller. --run requires an independently supplied human authorization.
Default --prepare only validates local frozen files; no GPU queries or imports.
"""
import argparse,csv,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'
UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def once(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:
  json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def append_event(root,obj):
 p=root/'execution'/'RESOURCE_EVENTS.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
 row=dict(unix=time.time(),monotonic_seconds=time.monotonic(),**obj)
 with p.open('a') as f:
  f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 return row

def update_resource_ledger(root):
 sys.path.insert(0,str(root/'qualification'))
 from resource_event_reducer import reduce_events
 path=root/'execution/RESOURCE_EVENTS.jsonl'
 events=[] if not path.exists() else [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
 if not events:return
 reduced=reduce_events(events,as_of_monotonic=time.monotonic())
 upper=reduced['GPU_seconds_upper_bound']
 lower=sum(x.get('occupied_seconds_lower',0) for x in reduced['units'])
 value={'task_id':TASK,'ALLOW_GPU_DIAG':'YES','NEW_GPUh':'UNRESOLVED',
  'derived_view_as_of_unix':time.time(),'derived_view_as_of_monotonic':time.monotonic(),
  'live_view_may_lag_up_to_seconds':5,'immutable_events_are_primary':True,
  'EXACT_GPUh':'UNRESOLVED','GPU_dispatches':sum(e['kind']=='dispatch_before_spawn' for e in events),
  'observed_dispatch_reservation_GPUh_lower':lower/3600,
  'observed_dispatch_reservation_GPUh_upper':'UNKNOWN' if upper=='UNKNOWN' else upper/3600,
  'resource_status':'LIVE_OR_TERMINAL_OBSERVATION_BOUNDS','events_file':'execution/RESOURCE_EVENTS.jsonl',
  'reduced_units':reduced['units'],'old_ledgers_unchanged':True,
  'physical_release_instant_not_fabricated':True}
 # Only the derived live view is replaced; immutable events/first receipts remain append-only.
 temp=root/'RESOURCE_LEDGER.next.json'
 with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,root/'RESOURCE_LEDGER.json')

def local_identity(root):
 m=read(root/'SOURCE_MANIFEST.json')
 if m['task_id']!=TASK or not m['execution_sources'] or not m['external_dependencies']:raise RuntimeError('source task mismatch or empty closure')
 for row in m['execution_sources']:
  p=(root/row['path']).resolve()
  if not p.is_relative_to(root):raise RuntimeError('source path escape')
  if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise RuntimeError('frozen source mismatch: '+row['path'])
 return m

def admission(root,auth_path,allow):
 if allow!='YES':raise RuntimeError('GPU not authorized: explicit --allow-gpu-diag YES required')
 a=read(auth_path)
 for k,v in [('task_id',TASK),('status','PASS'),('ALLOW_GPU_DIAG','YES'),('root',str(root)),('source_manifest_sha256',sha(root/'SOURCE_MANIFEST.json')),('contract_sha256',sha(root/'analysis/STRICT_BASELINE_CONTRACT.json')),('preregistration_sha256',sha(root/'qualification/CUBLAS_PREREGISTRATION.json'))]:
  if a.get(k)!=v:raise RuntimeError('authorization binding mismatch: '+k)
 if a.get('device')!={'physical_index':0,'uuid':UUID}:raise RuntimeError('device admission mismatch')
 plan=read(root/'qualification/CUBLAS_PREREGISTRATION.json')
 if str(root)!=plan['future_execution_root']:raise RuntimeError('execution root must be the frozen independent GPU run, never CPU seal')
 origin=read(root/'RUN_COPY_ORIGIN.json')
 if origin.get('task_id')!=TASK or origin.get('source_manifest_sha256')!=sha(root/'SOURCE_MANIFEST.json'):raise RuntimeError('missing or changed independent run bundle origin')
 if a.get('units')!=plan['units']:raise RuntimeError('unit allowlist differs from preregistration')
 if (root/'FIRST_FAILURE.json').exists() or (root/'execution/CONTROLLER_STARTED.json').exists():raise RuntimeError('failed/started task cannot be retried')
 return a,plan

def verify_external(m):
 for row in m['external_dependencies']:
  p=Path(row['path'])
  if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise RuntimeError('external identity mismatch: '+row['path'])

# These functions are invoked only in the explicit authorized --run branch.
# Merely importing or using --prepare cannot query GPU/driver/process state.
def device_rows():
 cmd=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits']
 text=subprocess.check_output(cmd,text=True,timeout=5)
 rows=[]
 for r in csv.reader(text.splitlines()):
  if len(r)!=6:raise RuntimeError('GPU visibility schema mismatch')
  rows.append({'index':int(r[0]),'uuid':r[1].strip(),'name':r[2].strip(),'total_MiB':float(r[3]),'free_MiB':float(r[4]),'driver':r[5].strip()})
 return rows

def gpu_processes():
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=5)
 rows=[]
 for r in csv.reader(text.splitlines()):
  if not r:continue
  if len(r)!=2:raise RuntimeError('GPU process visibility schema mismatch')
  rows.append((int(r[0]),r[1].strip()))
 return rows

def proc_identity(pid):
 p=Path('/proc')/str(pid)
 raw=(p/'stat').read_text();suffix=raw[raw.rfind(')')+2:].split()
 return {'starttick':int(suffix[19]),'cmdline':[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]}

def cleanup_own(proc,identity,unit,worker):
 if proc.poll() is not None:return
 now=proc_identity(proc.pid)
 if now['starttick']!=identity['starttick'] or str(worker) not in now['cmdline'] or unit not in now['cmdline']:
  raise RuntimeError('own-process identity lost; refuse signal')
 os.killpg(proc.pid,signal.SIGTERM)
 try:proc.wait(timeout=15)
 except subprocess.TimeoutExpired:
  now=proc_identity(proc.pid)
  if now['starttick']!=identity['starttick']:raise RuntimeError('PID reused; refuse signal')
  os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)

def enough_time(first,occupied,allowance,now=None):
 now=time.monotonic() if now is None else now
 wall=0 if first is None else now-first
 return occupied+allowance<=1650 and wall+allowance<=1650

def unit_environment(unit):
 from cublas_harness import fresh_environment
 return dict(fresh_environment(unit['condition']),ALLOW_GPU_DIAG='YES',HF_DATASETS_OFFLINE='1')

def command(root,unit,authorization):
 from cublas_harness import build_command
 return build_command(sys.executable,root,authorization,unit)

def run_unit(root,unit,authorization,first,occupied):
 if not enough_time(first,occupied,unit['allowance_seconds']):raise RuntimeError('remaining budget cannot admit next unit')
 if any(uuid==UUID for pid,uuid in gpu_processes()):raise RuntimeError('GPU not exclusive before unit')
 rows=[r for r in device_rows() if r['index']==0]
 if len(rows)!=1 or rows[0]['uuid']!=UUID or '3090' not in rows[0]['name'] or rows[0]['free_MiB']<22528:raise RuntimeError('GPU identity/memory admission failed')
 worker=root/'scripts/cublas_worker.py';target=root/'execution'/unit['unit_id'];target.mkdir(parents=True,exist_ok=True)
 if (target/'STARTED.json').exists() or (target/'DISPATCH.json').exists():raise RuntimeError('unit already dispatched; no retry')
 controller_identity=proc_identity(os.getpid())
 once(target/'DISPATCH.json',{'unit':unit,'device_observation':rows[0],'wall_unix':time.time(),
  'controller_pid':os.getpid(),'controller_starttick':controller_identity['starttick'],
  'authorization_sha256':sha(authorization),'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
  'budget_snapshot':{'completed_observed_seconds_upper':occupied,'first_dispatch_monotonic':first,'unit_allowance_seconds':unit['allowance_seconds'],'protective_seconds':1650,'hard_seconds':1800}})
 t0=time.monotonic();first=t0 if first is None else first
 append_event(root,{'unit_id':unit['unit_id'],'GPU_UUID':UUID,'kind':'dispatch_before_spawn','command':command(root,unit,authorization)})
 update_resource_ledger(root)
 last_ledger=time.monotonic()
 proc=None;identity=None;exit_seen=False;released=False;reason=None
 log=(target/'stdout.log').open('xb')
 try:
  child_env=dict(unit_environment(unit),QWEN_CONTROLLER_DISPATCH_SHA256=sha(target/'DISPATCH.json'))
  proc=subprocess.Popen(command(root,unit,authorization),cwd=root,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  append_event(root,{'unit_id':unit['unit_id'],'GPU_UUID':UUID,'kind':'spawn_observed','pid':proc.pid})
  try:identity=proc_identity(proc.pid)
  except FileNotFoundError:
   if proc.poll() is None:raise
   identity=None
  while True:
   now=time.monotonic();code=proc.poll();rows=gpu_processes()
   if now-last_ledger>=5:
    update_resource_ledger(root);last_ledger=now
   own_visible=any(pid==proc.pid and uuid==UUID for pid,uuid in rows)
   if any(uuid==UUID and pid!=proc.pid for pid,uuid in rows):reason='foreign GPU process observed'
   if code is None and own_visible:append_event(root,{'unit_id':unit['unit_id'],'GPU_UUID':UUID,'kind':'busy_observed','pid':proc.pid})
   if code is not None and not exit_seen:
    append_event(root,{'unit_id':unit['unit_id'],'GPU_UUID':UUID,'kind':'first_process_exit_observed','pid':proc.pid,'returncode':code});exit_seen=True
   if code is not None and not own_visible:
    append_event(root,{'unit_id':unit['unit_id'],'GPU_UUID':UUID,'kind':'first_exit_and_release_observed','pid':proc.pid});released=True;break
   if now-t0>=unit['allowance_seconds']:reason='unit allowance reached'
   if occupied+now-t0>=1650 or now-first>=1650:reason='protective task limit reached'
   if reason:
    if identity is not None:cleanup_own(proc,identity,unit['unit_id'],worker)
    else:raise RuntimeError('cleanup identity unavailable')
    # Continue bounded release observation, preserving the first exit receipt.
    deadline=time.monotonic()+min(120,max(0,1800-(time.monotonic()-first)))
    while time.monotonic()<deadline:
     code=proc.poll();visible=any(pid==proc.pid and uuid==UUID for pid,uuid in gpu_processes())
     if code is not None and not exit_seen:
      append_event(root,{'unit_id':unit['unit_id'],'GPU_UUID':UUID,'kind':'first_process_exit_observed','pid':proc.pid,'returncode':code});exit_seen=True
     if code is not None and not visible:
      append_event(root,{'unit_id':unit['unit_id'],'GPU_UUID':UUID,'kind':'first_exit_and_release_observed','pid':proc.pid});released=True;break
     time.sleep(1)
    raise RuntimeError(reason)
   time.sleep(1)
 finally:
  log.close()
  if proc is not None and proc.poll() is None and identity is not None:cleanup_own(proc,identity,unit['unit_id'],worker)
 if not released:raise RuntimeError('exit/release not proven; exact usage unresolved')
 elapsed=time.monotonic()-t0
 result=read(target/'RESULT.json')
 if proc.returncode!=0 or result.get('status')!='PASS':raise RuntimeError('worker failed; no retry')
 once(target/'CONTROLLER_RECEIPT.json',{'status':'PASS','returncode':proc.returncode,'pid':proc.pid,'identity':identity,
  'occupied_seconds_observed_upper':elapsed,'exact_physical_GPU_seconds':'UNRESOLVED_POLLING_BOUNDS_ONLY','result_sha256':sha(target/'RESULT.json')})
 update_resource_ledger(root)
 return first,occupied+elapsed

def summarize_first_difference(root):
 paths=list((root/'execution').glob('*/EARLIEST_NUMERICAL_FAILURE.json'))
 if not paths:return
 paths.sort(key=lambda p:p.stat().st_mtime_ns)
 target=root/'execution/FIRST_NUMERICAL_DIFFERENCE.json'
 if target.exists():return
 receipt=read(paths[0]);a=Path(receipt.get('actual_receipt',''));b=Path(receipt.get('reference_receipt',''))
 try:
  if not(a.resolve().is_relative_to(root) and b.resolve().is_relative_to(root)):raise RuntimeError('numerical receipt outside run')
  from cublas_compare import compare_failure
  value=compare_failure(b,a)
 except BaseException as err:
  value={'status':'DIFFERENCE_DETAILS_UNRESOLVED','error':str(err),'original_receipt':receipt}
 once(target,value)

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',action='store_true');ap.add_argument('--allow-gpu-diag',default='NO',choices=['NO','YES']);ap.add_argument('--authorization',type=Path);a=ap.parse_args(argv);root=a.root.resolve();admitted=False
 try:
  manifest=local_identity(root)
  if not a.run:
   print(json.dumps({'status':'PREPARED_CPU_ONLY_NO_GPU','GPU_queries':0,'GPU_dispatches':0,'root':str(root),'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json')}));return 0
  if not a.authorization:raise RuntimeError('authorization file required')
  auth,plan=admission(root,a.authorization.resolve(),a.allow_gpu_diag)
  verify_external(manifest)
  admitted=True
  # GPU calls first become reachable here, after all authorization/source gates.
  if any(uuid==UUID for pid,uuid in gpu_processes()):raise RuntimeError('GPU not exclusive')
  disk=os.statvfs(root)
  if disk.f_bavail*disk.f_frsize<plan['minimum_free_new_root_bytes']:raise RuntimeError('insufficient disk for raw qualification reference')
  ci=proc_identity(os.getpid())
  once(root/'execution/CONTROLLER_STARTED.json',{'task_id':TASK,'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'authorization_sha256':sha(a.authorization),'unix':time.time(),'controller_pid':os.getpid(),'controller_starttick':ci['starttick'],'controller_cmdline':ci['cmdline']})
  from cublas_harness import numeric_group_verdict,performance_verdict
  first=None;occupied=0.;unitmap={u['unit_id']:u for u in plan['units']}
  for key in plan['mandatory_numeric_units']:first,occupied=run_unit(root,unitmap[key],a.authorization.resolve(),first,occupied)
  numeric=numeric_group_verdict(root,'FP')
  once(root/'execution/FP_NUMERIC_GROUP.json',numeric)
  once(root/'qualification/FP_NUMERICAL_VERDICT.json',numeric)
  if numeric.get('status')!='PASS':raise RuntimeError('FP byte-level numerical qualification failed')
  for key in plan['performance_units']:first,occupied=run_unit(root,unitmap[key],a.authorization.resolve(),first,occupied)
  perf=performance_verdict(root);once(root/'execution/FP_PERFORMANCE_GROUP.json',perf)
  once(root/'qualification/PERFORMANCE_VERDICT.json',perf)
  if perf.get('status') not in ['PASS_BENEFIT','PASS_NO_BENEFIT']:raise RuntimeError('performance receipt incomplete/invalid')
  optional=[]
  if perf['status']=='PASS_BENEFIT':
   for group in plan['conditional_groups']:
    keys=group['units'];allowance=sum(unitmap[k]['allowance_seconds'] for k in keys)
    if not enough_time(first,occupied,allowance):
     optional.append({'group':group['name'],'status':'NOT_RUN_INSUFFICIENT_RESERVED_BUDGET'});continue
    for key in keys:first,occupied=run_unit(root,unitmap[key],a.authorization.resolve(),first,occupied)
    if group['codec_qualification']:
     verdict=numeric_group_verdict(root,group['codec_qualification']);once(root/'execution'/(group['name']+'_VERDICT.json'),verdict)
     once(root/'qualification'/(group['codec_qualification']+'_NUMERICAL_VERDICT.json'),verdict)
     if verdict.get('status')!='PASS':raise RuntimeError('Codec numerical qualification failed')
    optional.append({'group':group['name'],'status':'PASS_EXECUTED'})
  else:optional=[{'group':g['name'],'status':'NOT_RUN_NO_FP_BENEFIT'} for g in plan['conditional_groups']]
  once(root/'execution/CONTROLLER_RESULT.json',{'status':'PASS_FINITE_SCHEDULE_COMPLETE','GPU_seconds_observed_upper':occupied,'exact_GPUh':'UNRESOLVED_POLLING_BOUNDS_ONLY','FP_numeric':numeric,'FP_performance':perf,'optional':optional})
  return 0
 except BaseException as e:
  if a.run and admitted:
   try:update_resource_ledger(root)
   except BaseException as ledger_error:
    if not (root/'RESOURCE_ACCOUNTING_FAILURE.json').exists():once(root/'RESOURCE_ACCOUNTING_FAILURE.json',{'status':'UNRESOLVED','error':str(ledger_error),'exact_GPUh':'UNRESOLVED','do_not_treat_unknown_as_zero':True})
   p=root/'FIRST_FAILURE.json'
   if not p.exists():once(p,{'status':'FAIL_CLOSED_STOP','error':str(e),'unix':time.time(),'no_retry':True})
   summarize_first_difference(root)
  print(json.dumps({'status':'BLOCKED_OR_FAIL_CLOSED','error':str(e),'GPU_not_authorized_by_default':True}),file=sys.stderr)
  return 1
if __name__=='__main__':raise SystemExit(main())
