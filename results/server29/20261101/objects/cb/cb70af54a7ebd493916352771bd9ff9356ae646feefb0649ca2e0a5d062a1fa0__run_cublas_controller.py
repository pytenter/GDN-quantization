#!/usr/bin/env python3
"""Amended finite per-card controller; explicit human authorization only.
No cumulative GPU-hour or unit-wall cap. A committed-progress watchdog remains.
Import/default preparation is CPU-only and never queries a GPU.
"""
import argparse,csv,fcntl,hashlib,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
sys.dont_write_bytecode=True
TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'
RUN='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_GPU_20261019_V1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def once(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
 try:
  with tmp.open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(tmp,p)
 finally:tmp.unlink(missing_ok=True)
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def local_identity(root):
 m=read(root/'SOURCE_MANIFEST.json')
 if m['task_id']!=TASK or not m['execution_sources'] or not m['external_dependencies']:raise RuntimeError('source task/closure invalid')
 for row in m['execution_sources']:
  p=(root/row['path']).resolve()
  if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise RuntimeError('frozen source mismatch:'+row['path'])
 return m
def verify_external(m):
 for row in m['external_dependencies']:
  p=Path(row['path'])
  if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise RuntimeError('external identity mismatch:'+row['path'])
def admission(root,auth_path,allow):
 if allow!='YES':raise RuntimeError('explicit --allow-gpu-diag YES required')
 a=read(auth_path);p=read(root/'qualification/CUBLAS_PREREGISTRATION.json')
 checks={'task_id':TASK,'execution_run_id':RUN,'status':'PASS','ALLOW_GPU_DIAG':'YES','root':str(root),'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'contract_sha256':sha(root/'analysis/STRICT_BASELINE_CONTRACT.json'),'preregistration_sha256':sha(root/'qualification/CUBLAS_PREREGISTRATION.json'),'archive_config_sha256':sha(root/'ARCHIVE_CONFIG.json'),'human_authorization_sha256':sha(root/'HUMAN_GPU_AUTHORIZATION.json'),'amendment_sha256':sha(root/'GPU_EXECUTION_AMENDMENT.json'),'run_copy_origin_sha256':sha(root/'RUN_COPY_ORIGIN.json'),'no_GPU_time_limit':True}
 for k,v in checks.items():
  if a.get(k)!=v:raise RuntimeError('authorization binding mismatch:'+k)
 device=a.get('device',{})
 if type(device.get('physical_index')) is not int or device['physical_index'] not in range(4) or not isinstance(device.get('uuid'),str) or not device['uuid'].startswith('GPU-'):raise RuntimeError('invalid admitted device')
 if read(root/'ARCHIVE_CONFIG.json').get('card_key')!='gpu'+str(device['physical_index']):raise RuntimeError('archive card/device mismatch')
 if p.get('device')!=device or p.get('no_GPU_time_limit') is not True or p.get('progress_watchdog_seconds')!=1800 or str(root)!=p['future_execution_root']:raise RuntimeError('amended device/run/progress contract changed')
 from cublas_harness import build_stage_schedule
 if a.get('units')!=p['units'] or p['units']!=build_stage_schedule():raise RuntimeError('finite canonical unit allowlist changed')
 shared=Path(a['shared_control_root']).resolve();cards={k:str(Path(v).resolve())for k,v in a['card_roots'].items()}
 if set(cards)!={'gpu0','gpu1','gpu2','gpu3'} or cards.get('gpu'+str(device['physical_index']))!=str(root) or any(not Path(v).is_relative_to(shared) or Path(v)==shared for v in cards.values()):raise RuntimeError('sibling card roots not bound')
 if p.get('shared_control_root')!=str(shared) or p.get('card_roots')!=cards:raise RuntimeError('sibling plan/auth mismatch')
 origin=read(root/'RUN_COPY_ORIGIN.json')
 if origin.get('task_id')!=TASK or origin.get('source_manifest_sha256')!=sha(root/'SOURCE_MANIFEST.json'):raise RuntimeError('independent run copy origin mismatch')
 if (root/'FIRST_FAILURE.json').exists() or (shared/'FIRST_FAILURE.json').exists() or (root/'execution/CONTROLLER_STARTED.json').exists():raise RuntimeError('failed/started run cannot restart')
 return a,p
def append_event(root,obj):
 p=root/'execution/RESOURCE_EVENTS.jsonl';p.parent.mkdir(parents=True,exist_ok=True);row=dict(unix=time.time(),monotonic_seconds=time.monotonic(),**obj)
 with p.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 return row
def update_resource_ledger(root):
 sys.path.insert(0,str(root/'qualification'));from resource_event_reducer import reduce_events
 p=root/'execution/RESOURCE_EVENTS.jsonl'
 if not p.exists():return
 events=[json.loads(s)for s in p.read_text().splitlines()if s.strip()]
 if not events:return
 r=reduce_events(events,as_of_monotonic=time.monotonic());u=r['GPU_seconds_upper_bound']
 v=dict(task_id=TASK,execution_run_id=RUN,ALLOW_GPU_DIAG='YES',NEW_GPUh='UNRESOLVED',EXACT_GPUh='UNRESOLVED',no_GPU_time_limit=True,GPU_dispatches=sum(e['kind']=='dispatch_before_spawn'for e in events),derived_view_as_of_unix=time.time(),derived_view_as_of_monotonic=time.monotonic(),live_view_may_lag_up_to_seconds=5,immutable_events_are_primary=True,observed_dispatch_reservation_GPUh_lower=sum(x.get('occupied_seconds_lower',0)for x in r['units'])/3600,observed_dispatch_reservation_GPUh_upper='UNKNOWN'if u=='UNKNOWN'else u/3600,reduced_units=r['units'],physical_release_instant_not_fabricated=True)
 tmp=root/'RESOURCE_LEDGER.next.json'
 with tmp.open('w')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,root/'RESOURCE_LEDGER.json')
def shared_check(auth):
 p=Path(auth['shared_control_root'])/'FIRST_FAILURE.json'
 if p.exists():raise RuntimeError('shared fail-closed STOP:'+str(p))
def publish_failure(root,auth,error,unit=None):
 value=dict(status='FAIL_CLOSED_STOP',task_id=TASK,execution_run_id=RUN,root=str(root),device=auth['device'],unit_id=unit,error=str(error),unix=time.time(),no_retry=True)
 for p in [root/'FIRST_FAILURE.json',Path(auth['shared_control_root'])/'FIRST_FAILURE.json']:
  try:once(p,value)
  except FileExistsError:pass

def device_rows():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True,timeout=5);rows=[]
 for r in csv.reader(out.splitlines()):
  if len(r)!=6:raise RuntimeError('GPU visibility schema changed')
  rows.append(dict(index=int(r[0]),uuid=r[1].strip(),name=r[2].strip(),total_MiB=float(r[3]),free_MiB=float(r[4]),driver=r[5].strip()))
 return rows
def gpu_processes():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=5);rows=[]
 for r in csv.reader(out.splitlines()):
  if r:
   if len(r)!=2:raise RuntimeError('GPU process schema changed')
   rows.append((int(r[0]),r[1].strip()))
 return rows
def proc_identity(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();return dict(starttick=int(raw[raw.rfind(')')+2:].split()[19]),cmdline=[x.decode(errors='replace')for x in(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])
def cleanup_own(proc,identity,unit,worker):
 if proc.poll()is not None:return
 n=proc_identity(proc.pid)
 if n['starttick']!=identity['starttick']or str(worker)not in n['cmdline']or unit not in n['cmdline']:raise RuntimeError('own-process identity lost; refuse signal')
 os.killpg(proc.pid,signal.SIGTERM)
 try:proc.wait(timeout=15)
 except subprocess.TimeoutExpired:
  n=proc_identity(proc.pid)
  if n['starttick']!=identity['starttick']:raise RuntimeError('PID reused; refuse signal')
  os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)
def unit_environment(unit,device):
 from cublas_harness import fresh_environment
 return dict(fresh_environment(unit['condition'],device=device),ALLOW_GPU_DIAG='YES',HF_DATASETS_OFFLINE='1')
def command(root,unit,authorization):
 from cublas_harness import build_command
 return build_command(sys.executable,root,authorization,unit)
def progress_cursor(target):
 p=target/'PROGRESS.jsonl'
 if not p.exists():return -1
 try:
  with p.open('rb')as f:
   f.seek(max(0,p.stat().st_size-8192));lines=f.read().splitlines()
  for line in reversed(lines):
   try:r=json.loads(line)
   except (ValueError,UnicodeDecodeError):continue
   if r.get('progress_kind')in['phase_transition','bytes_hashed','token_committed']and type(r.get('sequence'))is int:return r['sequence']
 except FileNotFoundError:pass
 return -1

def run_unit(root,unit,authorization,auth,plan):
 shared_check(auth);device=auth['device'];uuid=device['uuid']
 if any(g==uuid for pid,g in gpu_processes()):raise RuntimeError('selected GPU not exclusive')
 rows=[r for r in device_rows()if r['index']==device['physical_index']]
 if len(rows)!=1 or rows[0]['uuid']!=uuid or '3090'not in rows[0]['name']or rows[0]['free_MiB']<22528:raise RuntimeError('GPU identity/memory admission failed')
 target=root/'execution'/unit['unit_id'];target.mkdir(parents=True,exist_ok=True)
 if (target/'STARTED.json').exists()or(target/'DISPATCH.json').exists():raise RuntimeError('unit already dispatched; no retry')
 ci=proc_identity(os.getpid());once(target/'DISPATCH.json',dict(unit=unit,device=device,device_observation=rows[0],wall_unix=time.time(),controller_pid=os.getpid(),controller_starttick=ci['starttick'],authorization_sha256=sha(authorization),source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),progress_policy=dict(no_GPU_time_limit=True,watchdog_seconds=1800)))
 t0=time.monotonic();append_event(root,dict(unit_id=unit['unit_id'],GPU_UUID=uuid,kind='dispatch_before_spawn',command=command(root,unit,authorization)));update_resource_ledger(root)
 proc=None;identity=None;exit_seen=False;released=False;last_progress=t0;cursor=-1;last_ledger=t0;reason=None;worker=root/'scripts/cublas_worker.py'
 log=(target/'stdout.log').open('xb')
 try:
  env=dict(unit_environment(unit,device),QWEN_CONTROLLER_DISPATCH_SHA256=sha(target/'DISPATCH.json'))
  proc=subprocess.Popen(command(root,unit,authorization),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);append_event(root,dict(unit_id=unit['unit_id'],GPU_UUID=uuid,kind='spawn_observed',pid=proc.pid))
  try:identity=proc_identity(proc.pid)
  except FileNotFoundError:
   if proc.poll()is None:raise
  while True:
   now=time.monotonic();code=proc.poll();live=gpu_processes();own=any(pid==proc.pid and g==uuid for pid,g in live);new=progress_cursor(target)
   if new>cursor:cursor=new;last_progress=now
   if now-last_ledger>=5:update_resource_ledger(root);last_ledger=now
   if any(g==uuid and pid!=proc.pid for pid,g in live):reason='foreign process observed on selected GPU'
   if code is None and own:append_event(root,dict(unit_id=unit['unit_id'],GPU_UUID=uuid,kind='busy_observed',pid=proc.pid))
   if code is not None and not exit_seen:append_event(root,dict(unit_id=unit['unit_id'],GPU_UUID=uuid,kind='first_process_exit_observed',pid=proc.pid,returncode=code));exit_seen=True
   if code is not None and not own:append_event(root,dict(unit_id=unit['unit_id'],GPU_UUID=uuid,kind='first_exit_and_release_observed',pid=proc.pid));released=True;break
   if (Path(auth['shared_control_root'])/'FIRST_FAILURE.json').exists():reason='shared sibling fail-closed STOP'
   if now-last_progress>=plan['progress_watchdog_seconds']:reason='no committed phase/token/hash progress for watchdog interval'
   if reason:
    publish_failure(root,auth,reason,unit['unit_id'])
    if identity is None:raise RuntimeError('cleanup identity unavailable')
    cleanup_own(proc,identity,unit['unit_id'],worker)
    # Cleanup observation grace is not a GPU execution-time budget.
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
     code=proc.poll();own=any(pid==proc.pid and g==uuid for pid,g in gpu_processes())
     if code is not None and not exit_seen:append_event(root,dict(unit_id=unit['unit_id'],GPU_UUID=uuid,kind='first_process_exit_observed',pid=proc.pid,returncode=code));exit_seen=True
     if code is not None and not own:append_event(root,dict(unit_id=unit['unit_id'],GPU_UUID=uuid,kind='first_exit_and_release_observed',pid=proc.pid));released=True;break
     update_resource_ledger(root);time.sleep(1)
    raise RuntimeError(reason)
   time.sleep(1)
 finally:
  log.close()
  if proc is not None and proc.poll()is None and identity is not None:cleanup_own(proc,identity,unit['unit_id'],worker)
  update_resource_ledger(root)
 if not released:raise RuntimeError('exit/release not proven; cost unresolved')
 result=read(target/'RESULT.json')
 if proc.returncode!=0 or result.get('status')!='PASS':raise RuntimeError('worker failed; no retry')
 once(target/'CONTROLLER_RECEIPT.json',dict(status='PASS',returncode=proc.returncode,pid=proc.pid,identity=identity,occupied_seconds_observed_upper=time.monotonic()-t0,exact_physical_GPU_seconds='UNRESOLVED_POLLING_BOUNDS_ONLY',result_sha256=sha(target/'RESULT.json')))
 return time.monotonic()-t0

def barrier(auth,relative,statuses):
 while True:
  shared_check(auth);pending=[]
  for key,name in auth['card_roots'].items():
   p=Path(name)/relative
   if not p.exists():pending.append(key);continue
   v=read(p)
   if v.get('status')not in statuses:raise RuntimeError('sibling phase failed:'+str(p))
  if not pending:return
  time.sleep(1)
def performance_lock(auth):
 p=Path(auth['shared_control_root'])/'performance.lock';p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a')
 try:
  while True:
   shared_check(auth)
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
   except BlockingIOError:time.sleep(1)
 except BaseException:f.close();raise

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',action='store_true');ap.add_argument('--allow-gpu-diag',default='NO',choices=['NO','YES']);ap.add_argument('--authorization',type=Path);a=ap.parse_args(argv);root=a.root.resolve();admitted=False;auth=None
 try:
  manifest=local_identity(root)
  if not a.run:print(json.dumps(dict(status='PREPARED_CPU_ONLY_NO_GPU',GPU_queries=0,GPU_dispatches=0)));return 0
  if not a.authorization:raise RuntimeError('authorization file required')
  auth,plan=admission(root,a.authorization.resolve(),a.allow_gpu_diag);verify_external(manifest);admitted=True;shared_check(auth)
  if any(g==auth['device']['uuid']for pid,g in gpu_processes()):raise RuntimeError('selected GPU not exclusive')
  disk=os.statvfs(root);required=plan['minimum_free_new_root_bytes']
  if required<4*1024**3 or disk.f_bavail*disk.f_frsize<required:raise RuntimeError('local streaming reserve below4GiB')
  from archive_io import archive_preflight
  archive=archive_preflight(root)
  if archive.get('status')!='PASS' or archive.get('required_quota_bytes',0)<200*1024**3:raise RuntimeError('remote archive capacity qualification failed')
  ci=proc_identity(os.getpid());once(root/'execution/CONTROLLER_STARTED.json',dict(task_id=TASK,execution_run_id=RUN,device=auth['device'],source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),authorization_sha256=sha(a.authorization),unix=time.time(),controller_pid=os.getpid(),controller_starttick=ci['starttick'],controller_cmdline=ci['cmdline'],archive_preflight=archive))
  from cublas_harness import numeric_group_verdict,performance_verdict
  units={u['unit_id']:u for u in plan['units']};seconds=0
  for key in plan['mandatory_numeric_units']:seconds+=run_unit(root,units[key],a.authorization.resolve(),auth,plan)
  numeric=numeric_group_verdict(root,'FP');once(root/'execution/FP_NUMERIC_GROUP.json',numeric);once(root/'qualification/FP_NUMERICAL_VERDICT.json',numeric)
  if numeric.get('status')!='PASS':raise RuntimeError('FP byte qualification failed')
  barrier(auth,'qualification/FP_NUMERICAL_VERDICT.json',{'PASS'})
  lock=performance_lock(auth)
  try:
   for key in plan['performance_units']:seconds+=run_unit(root,units[key],a.authorization.resolve(),auth,plan)
   perf=performance_verdict(root);once(root/'execution/FP_PERFORMANCE_GROUP.json',perf);once(root/'qualification/PERFORMANCE_VERDICT.json',perf)
  finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  if perf.get('status')not in ['PASS_BENEFIT','PASS_NO_BENEFIT']:raise RuntimeError('invalid performance receipt')
  barrier(auth,'qualification/PERFORMANCE_VERDICT.json',{'PASS_BENEFIT','PASS_NO_BENEFIT'})
  optional=[]
  if perf['status']=='PASS_BENEFIT':
   for group in plan['conditional_groups']:
    # Also serialize optional numerical/profiling groups to avoid host/I/O confounds.
    lock=performance_lock(auth)
    try:
     for key in group['units']:seconds+=run_unit(root,units[key],a.authorization.resolve(),auth,plan)
     if group['codec_qualification']:
      v=numeric_group_verdict(root,group['codec_qualification']);once(root/'execution'/(group['name']+'_VERDICT.json'),v);once(root/'qualification'/(group['codec_qualification']+'_NUMERICAL_VERDICT.json'),v)
      if v.get('status')!='PASS':raise RuntimeError('Codec numerical qualification failed')
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    optional.append(dict(group=group['name'],status='PASS_EXECUTED'))
  else:optional=[dict(group=g['name'],status='NOT_RUN_NO_FP_BENEFIT')for g in plan['conditional_groups']]
  once(root/'execution/CONTROLLER_RESULT.json',dict(status='PASS_FINITE_SCHEDULE_COMPLETE',device=auth['device'],GPU_seconds_observed_upper=seconds,exact_GPUh='UNRESOLVED_POLLING_BOUNDS_ONLY',no_GPU_time_limit=True,FP_numeric=numeric,FP_performance=perf,optional=optional));return 0
 except BaseException as e:
  if a.run and admitted:
   publish_failure(root,auth,e)
   try:update_resource_ledger(root)
   except BaseException as err:
    try:once(root/'RESOURCE_ACCOUNTING_FAILURE.json',dict(status='UNRESOLVED',error=str(err),do_not_treat_unknown_as_zero=True))
    except FileExistsError:pass
  print(json.dumps(dict(status='BLOCKED_OR_FAIL_CLOSED',error=str(e))),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
