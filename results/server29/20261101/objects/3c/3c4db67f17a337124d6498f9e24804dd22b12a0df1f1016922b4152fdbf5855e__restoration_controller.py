#!/usr/bin/env python3
"""Future single-card restoration controller. Import/--check are CPU stdlib only.

No GPU query occurs until CLI, ROOT, independent human authorization, plan,
request, source freeze and external source/model bindings all pass. No retries.
"""
import argparse,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
TASK='QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1'
SCOPE='SINGLE_GPU_HISTORICAL_PORT_RESTORATION_QUALIFICATION'
UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
UNITS=[{'id':'NUMERIC_FRESH0','mode':'numeric','optional':False},{'id':'NUMERIC_FRESH1','mode':'numeric','optional':False}]+[{'id':'OLD_FP_R'+str(i),'mode':'benchmark','optional':False}for i in range(3)]+[{'id':'NEW_FP_R0','mode':'benchmark','optional':True}]
LIMITS={'hard_seconds':1800,'protective_seconds':1650,'cleanup_reserve_seconds':150,'dispatch_max_seconds':240,'cleanup_wait_seconds':120}
def require(ok,message):
 if not ok:raise RuntimeError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def stable(path):
 p=Path(path);require(p.is_file()and not p.is_symlink(),'Missing/unsafe file:'+str(p));a=p.stat();b=p.read_bytes();z=p.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'File changed:'+str(p));return b
def sha(path):
 p=Path(path);require(p.is_file(),'File missing:'+str(p));a=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 z=p.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'Changed while SHA:'+str(p));return h.hexdigest()
def read(path):return json.loads(stable(path))
def relative(root,name):
 p=Path(name);require(not p.is_absolute()and'..'not in p.parts,'Unsafe relative path:'+str(name));q=(Path(root)/p).resolve();require(Path(root).resolve()in q.parents,'Path outside root');return q
def atomic_new(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);raw=canonical(obj)+b'\n';tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));require(not p.exists()and not tmp.exists(),'Existing immutable receipt:'+str(p))
 with tmp.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.link(tmp,p);tmp.unlink();directory=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(directory)
 finally:os.close(directory)
 return hashlib.sha256(raw).hexdigest()
def process_identity(pid):
 p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split();argv=(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0');return {'pid':int(pid),'starttick':int(fields[19]),'cmdline':[x.decode()for x in argv]}
def same_process(expected,reader=process_identity):
 try:return reader(expected['pid'])==expected
 except (OSError,ValueError):return False

def validate_plan(plan):
 require(plan.get('schema')=='QWEN_HISTORICAL_RUNTIME_GPU_PLAN_V1'and plan.get('task_id')==TASK,'Wrong plan schema/task')
 require(plan.get('device')=={'physical_index':0,'uuid':UUID},'Device/UUID changed')
 for k,v in LIMITS.items():require(type(plan.get(k))in (int,float)and plan[k]==v,'Frozen budget changed:'+k)
 require(plan.get('units')==UNITS,'Frozen six-unit roster/order changed')
 require(plan.get('max_active_GPUs')==1,'Only one active GPU permitted')
 require(plan.get('minimum_disk_free_bytes')==10*1024**3 and plan.get('max_numeric_raw_bytes')==8*1024**3 and plan.get('min_gpu_free_bytes')==22*1024**3,'Storage/GPU memory minimum changed')
 require(plan.get('numeric_tokens')==32 and plan.get('warmup_tokens')==128 and plan.get('measured_tokens')==128,'Workload window changed')
 require(plan.get('binding_files')and isinstance(plan['binding_files'],list),'Binding files absent')
 return plan

def binding_rows(obj):
 rows=[]
 for key in ['model_weight_shards','model_small_files','native_library_inventory','native_library_rows','torch_python_files','external_source_files']:
  rows+=obj.get(key,[])
 if isinstance(obj.get('python_executable'),dict):rows.append(obj['python_executable'])
 return [x for x in rows if isinstance(x,dict)and all(k in x for k in ['path','bytes','sha256'])]

def check_inputs(root,plan_path=None):
 root=Path(root).resolve();plan_path=Path(plan_path).resolve()if plan_path else root/'qualification/GPU_PLAN.json';require(root in plan_path.parents,'Plan outside root');plan=validate_plan(read(plan_path));freeze_path=root/'SOURCE_FREEZE.json';freeze=read(freeze_path);rows=freeze.get('files');require(isinstance(rows,list)and rows,'Source freeze files empty');seen=set()
 for row in rows:
  require(row['path']not in seen,'Duplicate frozen path');seen.add(row['path']);p=relative(root,row['path']);require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'Frozen source changed:'+row['path'])
 require({'scripts/restoration_controller.py','worker/restoration_worker.py',str(plan_path.relative_to(root)),'USER_REQUEST.txt'}<=seen,'Controller/worker/plan/request not frozen')
 require(plan.get('request_sha256')==sha(root/'USER_REQUEST.txt'),'Plan request identity changed');bindings={}
 for name in plan['binding_files']:
  require(name in seen,'Binding file not source-frozen:'+name)
  for row in binding_rows(read(relative(root,name))):
   p=Path(row.get('resolved_path',row['path']));require(p.is_absolute(),'External dependency not absolute');key=str(p.resolve());old=bindings.get(key)
   if old:require((old['bytes'],old['sha256'])==(row['bytes'],row['sha256']),'Conflicting external binding:'+key)
   else:bindings[key]=row
 require(len([x for x in bindings.values()if str(x['path']).endswith('.safetensors')])==4,'Exactly four model shards required')
 for name,row in bindings.items():
  p=Path(name);require(p.is_file()and p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'External source/model/native bytes/SHA changed:'+name)
 return {'root':root,'plan':plan,'plan_path':plan_path,'plan_sha256':sha(plan_path),'freeze_path':freeze_path,'source_freeze_sha256':sha(freeze_path),'request_sha256':sha(root/'USER_REQUEST.txt'),'source_files':len(rows),'external_files':len(bindings)}

def authorize(root,cli_yes,authorization,plan_path=None):
 root=Path(root).resolve();require(cli_yes=='YES','GPU_NOT_AUTHORIZED: explicit CLI YES missing');state=read(root/'ROOT.json');require(state.get('task_id')==TASK and state.get('ALLOW_GPU_RESTORE')=='YES','GPU_NOT_AUTHORIZED: ROOT ALLOW_GPU_RESTORE is not YES')
 ap=Path(authorization).resolve();require(ap==root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','Independent authorization path required');auth=read(ap);require(auth.get('ALLOW_GPU_RESTORE')=='YES'and auth.get('task_id')==TASK and auth.get('scope')==SCOPE,'GPU_NOT_AUTHORIZED: wrong human scope')
 require(isinstance(auth.get('human_request_quote'),str)and bool(auth['human_request_quote'].strip()),'Explicit human request quote missing')
 require(auth.get('device_uuid')==UUID and auth.get('max_active_GPUs')==1 and type(auth.get('max_GPU_hours'))in(int,float)and auth['max_GPU_hours']==0.5,'Human device/budget differs')
 require(auth.get('allowed_units')==[u['id']for u in UNITS],'Human allowed-unit roster differs');checked=check_inputs(root,plan_path)
 for key in ['plan_sha256','source_freeze_sha256','request_sha256']:require(auth.get(key)==checked[key],'Human authorization identity changed:'+key)
 checked.update(authorization=auth,authorization_path=ap,authorization_sha256=sha(ap));return checked

def require_worker_permit(root,mode,unit_id,permit_path,*,process_reader=None,parent_pid=None,monotonic=None):
 root=Path(root).resolve();reader=process_reader or process_identity;clock=monotonic or time.monotonic;phase=root/'execution'/unit_id;pp=Path(permit_path).resolve();require(pp==phase/'PERMIT.json','Permit path/unit root mismatch');permit=read(pp)
 require(permit.get('task_id')==TASK and permit.get('scope')==SCOPE and permit.get('root')==str(root),'Wrong worker permit scope');checked=authorize(root,'YES',root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',root/permit['plan_relative_path'])
 unit=next((u for u in UNITS if u['id']==unit_id),None);require(unit and permit.get('unit')==unit and mode==unit['mode'],'Worker mode/unit changed')
 meta=read(root/'execution/CONTROLLER_STARTED.json');pid=os.getppid()if parent_pid is None else parent_pid;identity=permit['controller_identity'];require(pid==identity['pid']and identity==meta['controller_identity']and reader(pid)==identity,'Worker not direct child of exact admitted controller')
 require(str(root/'scripts/restoration_controller.py')in identity['cmdline']and'--run'in identity['cmdline'],'Unexpected controller cmdline')
 for k in ['plan_sha256','source_freeze_sha256','request_sha256','authorization_sha256']:require(permit.get(k)==checked[k]==meta[k],'Permit identity mismatch:'+k)
 require(permit.get('device')==checked['plan']['device'],'Permit GPU binding mismatch')
 for key,value in permit.get('environment_controls',{}).items():require((key not in os.environ)if value is None else os.environ.get(key)==value,'Worker startup environment differs:'+key)
 require(set(permit.get('environment_controls',{}))==set(ENV_KEYS),'Incomplete worker environment binding');require(clock()<=permit['dispatch_deadline_monotonic'],'Worker dispatch deadline expired');dispatch_path=phase/'DISPATCH.json';until=time.monotonic()+2
 while not dispatch_path.exists()and time.monotonic()<until:time.sleep(.01)
 dispatch=read(dispatch_path);own=reader(os.getpid());require(dispatch['worker_identity']==own and own['cmdline']==permit['worker_command'],'Worker PID/starttick/cmdline not dispatcher-bound');require(dispatch['permit_sha256']==sha(pp),'Dispatch permit changed');require(not(root/'FIRST_FAILURE.json').exists(),'Prior failure prohibits worker')
 return dict(permit,permit_sha256=sha(pp),worker_identity=own,checked_identity={k:checked[k]for k in ['plan_sha256','source_freeze_sha256','request_sha256','authorization_sha256']})

def disk_guard(root,plan):
 v=os.statvfs(root);free=v.f_bavail*v.f_frsize;raw=sum(p.stat().st_size for p in (Path(root)/'execution').rglob('*.bin')if p.is_file());require(free>=plan['minimum_disk_free_bytes'],'Disk reserve insufficient');require(raw<=plan['max_numeric_raw_bytes'],'Numeric raw cap exceeded');return {'available_bytes':free,'numeric_raw_bytes':raw}

def query_gpu(timeout=10):
 # This function is called only after authorize/check_inputs succeeds.
 deadline=time.monotonic()+timeout;cmd=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'];p=subprocess.run(cmd,capture_output=True,text=True,timeout=max(.01,deadline-time.monotonic()));require(p.returncode==0,'GPU inventory query failed');devices=[]
 for line in p.stdout.strip().splitlines():
  f=[x.strip()for x in line.split(',')];require(len(f)==5,'Malformed GPU inventory');devices.append({'physical_index':int(f[0]),'uuid':f[1],'name':f[2],'memory_total_MiB':float(f[3]),'memory_free_MiB':float(f[4])})
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=max(.01,deadline-time.monotonic()));require(p.returncode==0,'GPU process query failed');apps=[]
 for line in p.stdout.strip().splitlines():
  f=[x.strip()for x in line.split(',')];require(len(f)==3,'Malformed GPU process inventory');apps.append({'pid':int(f[0]),'uuid':f[1],'memory_MiB':f[2]})
 return {'devices':devices,'apps':apps}
def validate_gpu(observation,own_pid=None,minimum_free=23000):
 rows=[x for x in observation['devices']if x['physical_index']==0];require(len(rows)==1 and rows[0]['uuid']==UUID and'3090'in rows[0]['name']and rows[0]['memory_total_MiB']>=24000,'Actual GPU identity/memory differs');foreign=[x for x in observation['apps']if x['uuid']==UUID and x['pid']!=own_pid];require(not foreign,'Selected GPU has foreign compute process; do not kill it')
 if own_pid is None:require(rows[0]['memory_free_MiB']>=minimum_free,'Insufficient exclusive GPU memory')
 return rows[0]
def budget_action(unit,elapsed,plan):
 require(math.isfinite(elapsed)and elapsed>=0,'Invalid monotonic elapsed');fits=plan['hard_seconds']-elapsed>=plan['dispatch_max_seconds']+plan['cleanup_reserve_seconds']and elapsed<plan['protective_seconds']
 if fits:return 'DISPATCH'
 return 'STOP_MINIMUM_NUMERIC_BUDGET'if unit['mode']=='numeric'else'SKIP_BUDGET'
class Ledger:
 def __init__(self,path,clock=time.monotonic):self.path=Path(path);self.clock=clock;self.first=set()
 def event(self,kind,**values):
  self.path.parent.mkdir(parents=True,exist_ok=True)
  with self.path.open('a')as f:f.write(json.dumps(dict(event=kind,monotonic=self.clock(),unix=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 def once(self,key,kind,**values):
  if key in self.first:return False
  self.first.add(key);self.event(kind,**values);return True

def terminate_exact(proc,identity,reader=process_identity):
 require(identity['pid']==proc.pid and same_process(identity,reader),'Refuse termination: PID/starttick/cmdline differs');proc.terminate();return True

def validate_result(phase,unit,identity,permit_sha):
 p=Path(phase)/'RESULT.json';require(p.is_file(),'Worker exited without RESULT: inspect logs/process, never retry');r=read(p);require(r.get('status')=='PASS'and r.get('unit_id')==unit['id']and r.get('mode')==unit['mode'],'Worker failed or wrong receipt');require(r.get('process_identity')==identity and r.get('permit_sha256')==permit_sha,'Worker receipt identity/permit differs');return r

ENV_KEYS=['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']
def worker_environment(unit):
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 for key in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']:env.pop(key,None)
 if unit['id']=='NEW_FP_R0':env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_DEVICE_ORDER='PCI_BUS_ID')
 return env

def run_controller(root,authorization,cli_yes,plan_path=None,*,gpu_reader=query_gpu,popen=subprocess.Popen,reader=process_identity,clock=time.monotonic,sleeper=time.sleep):
 checked=authorize(root,cli_yes,authorization,plan_path);root=checked['root'];plan=checked['plan'];execution=root/'execution';require(not(execution/'CONTROLLER_STARTED.json').exists()and not(execution/'CONTROLLER_ATTEMPT_STARTED.json').exists()and not(root/'FIRST_FAILURE.json').exists(),'Prior started/failure controller cannot restart')
 start=clock();ledger=Ledger(execution/'RESOURCE_EVENTS.jsonl',clock);identity=reader(os.getpid());atomic_new(execution/'CONTROLLER_ATTEMPT_STARTED.json',dict(task_id=TASK,controller_identity=identity,start_monotonic=start,authorization_sha256=checked['authorization_sha256']));results=[];skipped=[];processes=[];active=None;worker_id=None;unit=None;before=after=last_alive=last_seen=exit_seen=released=None
 def observe_gpu(deadline=None):
  if gpu_reader is query_gpu:return query_gpu(timeout=max(.01,min(5,(deadline-clock())if deadline is not None else 5)))
  return gpu_reader()
 def exit_event(rc):
  nonlocal exit_seen
  if exit_seen is None:exit_seen=clock();ledger.once(unit['id']+':exit','FIRST_PROCESS_EXIT_OBSERVED',unit=unit,returncode=rc,observed_monotonic=exit_seen,worker_identity=worker_id)
 def release_event():
  nonlocal released
  if released is None:released=clock();ledger.once(unit['id']+':release','FIRST_NVML_RELEASE_OBSERVED',unit=unit,observed_monotonic=released,worker_identity=worker_id)
 def cost_event():
  if unit is None or before is None or after is None:return
  ledger.once(unit['id']+':cost','UNIT_COST_BOUNDS',unit_id=unit['id'],worker_device_reservation_seconds_lower=max(0,(last_alive or after)-after),worker_device_reservation_seconds_upper=max(0,released-before)if released is not None else'UNRESOLVED',last_process_alive_monotonic=last_alive,process_exit_observed_monotonic=exit_seen,NVML_release_observed_monotonic=released,actual_physical_GPU_hours='UNRESOLVED_NOT_ZERO')
 def cleanup():
  nonlocal last_alive,last_seen
  errors=[]
  if active is None:return {'status':'NO_WORKER_DISPATCHED','NVML_release_verified':'NOT_DISPATCHED','errors':errors}
  deadline=clock()+plan['cleanup_wait_seconds'];cleanup_start=clock();killed=False
  if active.poll()is None:
   if worker_id is not None and same_process(worker_id,reader):terminate_exact(active,worker_id,reader)
   else:errors.append('REFUSED_SIGNAL_PROCESS_IDENTITY_NOT_VERIFIED')
  while clock()<deadline:
   poll_time=clock();rc=active.poll()
   if rc is None:last_alive=poll_time
   else:exit_event(rc)
   try:
    obs=observe_gpu(deadline);own=[x for x in obs['apps']if x['uuid']==UUID and x['pid']==active.pid]
    if own:last_seen=clock()
    if rc is not None and not own:release_event();break
   except Exception as exc:errors.append(type(exc).__name__+':'+str(exc))
   if not killed and clock()-cleanup_start>=10 and active.poll()is None and worker_id is not None and same_process(worker_id,reader):active.kill();killed=True
   sleeper(.25)
  cost_event();return {'status':'PASS_EXIT_AND_RELEASE_VERIFIED'if released is not None else'UNRESOLVED_CLEANUP_RELEASE','NVML_release_verified':released is not None,'process_exit_observed_monotonic':exit_seen,'release_observed_monotonic':released,'errors':errors}
 try:
  disk_guard(root,plan);obs=observe_gpu();validate_gpu(obs,minimum_free=plan['min_gpu_free_bytes']/1024**2);ledger.event('GPU_ADMISSION_PASS',observation=obs)
  meta={k:checked[k]for k in ['plan_sha256','source_freeze_sha256','request_sha256','authorization_sha256']};meta.update(task_id=TASK,controller_identity=identity,device=plan['device'],start_monotonic=start,clock_host='THIS_CONTROLLER_HOST');atomic_new(execution/'CONTROLLER_STARTED.json',meta)
  for unit in UNITS:
   action=budget_action(unit,clock()-start,plan)
   if action=='SKIP_BUDGET':skipped.append(unit['id']);ledger.event('UNIT_NOT_RUN_BUDGET',unit=unit);continue
   require(action=='DISPATCH','Insufficient budget for minimum two fresh numeric runs')
   if unit['mode']=='benchmark':require(all(x in [y['unit_id']for y in results]for x in ['NUMERIC_FRESH0','NUMERIC_FRESH1']),'Both numeric fresh processes must PASS before performance')
   disk_guard(root,plan);validate_gpu(observe_gpu(),minimum_free=plan['min_gpu_free_bytes']/1024**2);phase=execution/unit['id'];require(not phase.exists(),'Existing unit cannot restart');phase.mkdir(parents=True);permit_path=phase/'PERMIT.json';command=[sys.executable,'-B',str(root/'worker/restoration_worker.py'),'--root',str(root),'--mode',unit['mode'],'--unit',unit['id'],'--permit',str(permit_path)];environment=worker_environment(unit);before=clock();after=last_alive=last_seen=exit_seen=released=None
   permit=dict(task_id=TASK,scope=SCOPE,root=str(root),unit=unit,device=plan['device'],controller_identity=identity,worker_command=command,environment_controls={k:environment.get(k)for k in ENV_KEYS},plan_relative_path=str(checked['plan_path'].relative_to(root)),dispatch_monotonic=before,dispatch_deadline_monotonic=min(before+plan['dispatch_max_seconds'],start+plan['protective_seconds']),**{k:checked[k]for k in ['plan_sha256','source_freeze_sha256','request_sha256','authorization_sha256']});permit_sha=atomic_new(permit_path,permit)
   with(phase/'worker.log').open('xb')as log:active=popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
   after=clock();worker_id=reader(active.pid);require(worker_id['cmdline']==command and worker_id not in processes,'Observed worker cmdline/fresh identity differs');processes.append(worker_id);atomic_new(phase/'DISPATCH.json',dict(worker_identity=worker_id,permit_sha256=permit_sha,dispatch_before_monotonic=before,dispatch_after_monotonic=after));ledger.once(unit['id']+':dispatch','FIRST_DISPATCH',unit=unit,worker_identity=worker_id,before_monotonic=before,after_monotonic=after)
   while True:
    poll_time=clock();rc=active.poll()
    if rc is None:last_alive=poll_time
    else:exit_event(rc)
    require(clock()<permit['dispatch_deadline_monotonic'],'DISPATCH_OR_PROTECTIVE_LIMIT');disk_guard(root,plan);current=observe_gpu(permit['dispatch_deadline_monotonic']);validate_gpu(current,own_pid=active.pid);own=[x for x in current['apps']if x['uuid']==UUID and x['pid']==active.pid]
    if own:last_seen=clock();ledger.event('OWN_GPU_PID_PRESENT',unit_id=unit['id'],worker_identity=worker_id)
    if exit_seen is not None and not own:release_event();break
    sleeper(.25)
   cost_event();require(active.returncode==0,'Worker nonzero exit: inspect log and receipt; no retry');require(not same_process(worker_id,reader),'Worker still alive after exit');result=validate_result(phase,unit,worker_id,permit_sha);results.append(result);active=None;worker_id=None;require(clock()-start<=plan['hard_seconds'],'Hard total budget exceeded')
  final=dict(status='COMPLETE_FROZEN_RESTORATION_UNITS',task_id=TASK,completed_units=[x['unit_id']for x in results],skipped_units=skipped,minimum_numeric_complete=all(x in [y['unit_id']for y in results]for x in ['NUMERIC_FRESH0','NUMERIC_FRESH1']),benchmark_repeats=len([x for x in results if x['unit_id'].startswith('OLD_FP_R')]),elapsed_controller_seconds=clock()-start,actual_physical_GPU_hours='UNRESOLVED_NOT_ZERO',no_retry=True);atomic_new(execution/'CONTROLLER_RESULT.json',final);return final
 except BaseException as exc:
  cleanup_result=cleanup();failure=dict(status='FAIL_CLOSED_STOP',task_id=TASK,error=type(exc).__name__+':'+str(exc),unit=unit,cleanup=cleanup_result,traceback=traceback.format_exc(),no_retry=True)
  if not(root/'FIRST_FAILURE.json').exists():atomic_new(root/'FIRST_FAILURE.json',failure)
  if not(execution/'CONTROLLER_RESULT.json').exists():atomic_new(execution/'CONTROLLER_RESULT.json',failure)
  ledger.event('FAIL_CLOSED_STOP',failure=failure);raise

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--check',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--allow-gpu-restore',choices=['NO','YES'],default='NO');p.add_argument('--authorization',type=Path);p.add_argument('--plan',type=Path);a=p.parse_args();require(not(a.check and a.run),'Choose check or run')
 if not a.run:
  x=check_inputs(a.root,a.plan);print(json.dumps(dict(status='PASS_CPU_SOURCE_AND_BINDINGS_CHECK_NOT_GPU_QUALIFICATION',ALLOW_GPU_RESTORE=read(Path(a.root)/'ROOT.json').get('ALLOW_GPU_RESTORE','NO'),source_files=x['source_files'],external_files=x['external_files'],GPU_queries=0,GPU_dispatches=0)));return
 require(a.authorization is not None,'Independent human authorization required');print(json.dumps(run_controller(a.root,a.authorization,a.allow_gpu_restore,a.plan)))
if __name__=='__main__':
 try:main()
 except BaseException as exc:print(json.dumps({'status':'DENIED_OR_FAIL_CLOSED_STOP','error':type(exc).__name__+':'+str(exc)}),file=sys.stderr);sys.exit(2)
