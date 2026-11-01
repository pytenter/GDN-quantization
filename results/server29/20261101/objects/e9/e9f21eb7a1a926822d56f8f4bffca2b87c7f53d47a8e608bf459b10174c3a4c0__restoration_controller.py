#!/usr/bin/env python3
"""Finite same-card recovery controller; no elapsed GPU/dispatch limit.

CPU --check never imports a GPU runtime or queries GPU/Driver/NVML. A separately
bound new-task authorization and real source/model closure precede GPU queries.
Only genuine committed phase/token/hash facts reset the 30min watchdog.
"""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
TASK='QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1'
SCOPE='QWEN_HISTORICAL_RUNTIME_GPU_RECOVERY_FINITE_FP_QUALIFICATION_RESOURCE_AMENDMENT_20261019_V1'
UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
PARENT_FREEZE='64ca5853e4ed65ab3e2c59d9deb85b4aa026f0f712abd5496495b7b912fcb457'
REPLACED={'scripts/restoration_controller.py','worker/restoration_worker.py','qualification/GPU_PLAN.json','USER_REQUEST.txt'}
UNITS=[{'id':'NUMERIC_FRESH0','mode':'numeric','optional':False},{'id':'NUMERIC_FRESH1','mode':'numeric','optional':False}]+[{'id':'OLD_FP_R'+str(i),'mode':'benchmark','optional':False}for i in range(3)]+[{'id':'NEW_FP_R0','mode':'benchmark','optional':True}]
ENV_KEYS=['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']
MATH_GATE={'state_atol': 1e-05, 'state_scale_rtol': 0.0001, 'output_atol': 0.001, 'output_scale_rtol': 0.01, 'normalization_atol': 0.0001, 'normalization_scale_rtol': 0.02, 'scope': '32-token actual-activation-driven FP64 state recurrence from token0; bounded normalization/core rounding checks, not full-model FP64/long-horizon proof'}

def require(ok,message):
 if not ok:raise RuntimeError(message)

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def stable(path):
 p=Path(path);require(p.is_file()and not p.is_symlink(),'Missing/unsafe file:'+str(p));a=p.stat();b=p.read_bytes();z=p.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'File changed:'+str(p));return b

def sha(path,progress=None):
 p=Path(path);require(p.is_file(),'File missing:'+str(p));a=p.stat();h=hashlib.sha256();count=0;next_notice=64*1024**2
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024**2),b''):
   h.update(block);count+=len(block)
   if progress is not None and count>=next_notice:progress('BOUND_FILE_HASH_READ',dict(path=str(p),read_bytes=count,prefix_sha256=h.hexdigest()));next_notice=count+64*1024**2
 z=p.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'Changed while SHA:'+str(p))
 if progress is not None:progress('BOUND_FILE_HASH_COMPLETE',dict(path=str(p),read_bytes=count,sha256=h.hexdigest()))
 return h.hexdigest()

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

def binding_rows(obj):
 rows=[]
 for key in ['model_weight_shards','model_small_files','native_library_inventory','native_library_rows','torch_python_files','external_source_files']:
  rows+=obj.get(key,[])
 if isinstance(obj.get('python_executable'),dict):rows.append(obj['python_executable'])
 return [x for x in rows if isinstance(x,dict)and all(k in x for k in ['path','bytes','sha256'])]

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
 if own_pid is not None:require(not any(x['pid']==own_pid and x['uuid']!=UUID for x in observation['apps']),'Own worker PID is present on an unauthorized GPU UUID')
 if own_pid is None:require(rows[0]['memory_free_MiB']>=minimum_free,'Insufficient exclusive GPU memory')
 return rows[0]

def require_own_gpu_identity(own_pid,observation=None):
 # PRE/POST worker admission: actual process records must be nonempty and
 # exclusively the authorized physical UUID. Keep full failed evidence.
 receipt=dict(status='FAIL_CLOSED_STOP',own_pid=int(own_pid),expected_uuid=UUID,expected_physical_index=0,observation=None,own_records=[],actual_UUIDs=[],actual_identity='UNRESOLVED')
 try:
  obs=query_gpu()if observation is None else observation;receipt['observation']=obs;own=[x for x in obs['apps']if x['pid']==own_pid];receipt.update(own_records=own,actual_UUIDs=sorted({x['uuid']for x in own}));validate_gpu(obs,own_pid=own_pid);require(own,'Own worker PID absent from actual NVML compute records');require(all(x['uuid']==UUID for x in own),'Own worker actual UUID differs');receipt.update(status='PASS',actual_identity='ONLY_AUTHORIZED_UUID');return receipt
 except BaseException as exc:
  receipt['error']=type(exc).__name__+':'+str(exc);exc.gpu_identity_receipt=receipt;raise

def require_driver_ordinal_identity(library_rows,*,driver_factory=None,environment=None):
 # Future authorized child only. This performs driver initialization/enumeration
 # without requesting a context, retaining exact driver-file and UUID evidence.
 # It is never called by --check or prior to worker permit admission.
 import ctypes,uuid
 env=os.environ if environment is None else environment;receipt=dict(status='FAIL_CLOSED_STOP',expected_uuid=UUID,CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES'),logical_ordinal=0,visible_device_count=None,actual_uuid=None,driver=None,calls=[],context_creation_API_calls=0,kernel_launch_API_calls=0)
 try:
  require(env.get('CUDA_VISIBLE_DEVICES')=='0','Driver ordinal check requires exact CUDA_VISIBLE_DEVICES=0');candidates={}
  for row in library_rows:
   p=Path(row.get('resolved_path',row['path'])).resolve()
   if p.name.startswith('libcuda.so'):candidates[str(p)]=row
  require(len(candidates)==1,'Exactly one frozen resolved libcuda.so driver required');path,row=next(iter(candidates.items()));p=Path(path);a=p.stat();h=sha(p);z=p.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)and z.st_size==row['bytes']and h==row['sha256'],'Driver file differs from bound SHA/bytes');receipt['driver']=dict(path=path,bytes=z.st_size,sha256=h,device=z.st_dev,inode=z.st_ino,mtime_ns=z.st_mtime_ns)
  lib=(driver_factory or ctypes.CDLL)(path)
  class CUuuid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
  init=lib.cuInit;countfn=lib.cuDeviceGetCount;get=lib.cuDeviceGet;uuid_name='cuDeviceGetUuid_v2'if hasattr(lib,'cuDeviceGetUuid_v2')else'cuDeviceGetUuid';getuuid=getattr(lib,uuid_name)
  for fn,args in [(init,[ctypes.c_uint]),(countfn,[ctypes.POINTER(ctypes.c_int)]),(get,[ctypes.POINTER(ctypes.c_int),ctypes.c_int]),(getuuid,[ctypes.POINTER(CUuuid),ctypes.c_int])]:fn.argtypes=args;fn.restype=ctypes.c_int
  def call(name,fn,*args):
   rc=int(fn(*args));receipt['calls'].append(dict(api=name,return_code=rc));require(rc==0,'CUDA Driver identity API failure:'+name+':'+str(rc))
  call('cuInit',init,0);count=ctypes.c_int();call('cuDeviceGetCount',countfn,ctypes.byref(count));receipt['visible_device_count']=count.value;require(count.value==1,'Driver visible count is not one under CVD=0; ordinal interpretation unqualified');dev=ctypes.c_int();call('cuDeviceGet',get,ctypes.byref(dev),0);value=CUuuid();call(uuid_name,getuuid,ctypes.byref(value),dev.value);actual='GPU-'+str(uuid.UUID(bytes=bytes(value.bytes)));receipt['actual_uuid']=actual;require(actual==UUID,'Driver logical ordinal0 UUID differs from authorized physical UUID');a=p.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)and sha(p)==h,'Driver changed during ordinal observation');receipt.update(status='PASS',scope='DRIVER_ENUMERATION_ONLY_NO_CONTEXT_OR_NUMERIC_PROOF');return receipt
 except BaseException as exc:
  receipt['error']=type(exc).__name__+':'+str(exc);exc.driver_identity_receipt=receipt;raise

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

def worker_environment(unit):
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 for key in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']:env.pop(key,None)
 if unit['id']=='NEW_FP_R0':env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_DEVICE_ORDER='PCI_BUS_ID')
 return env

def validate_plan(plan):
 require(plan.get('schema')=='QWEN_HISTORICAL_RUNTIME_GPU_PLAN_RESOURCE_AMENDMENT_V1'and plan.get('task_id')==TASK,'Wrong amended plan task/schema');require(plan.get('parent_effective_source_freeze_sha256')==PARENT_FREEZE and plan.get('parent_effective_manifest_path')=='provenance/PARENT_EFFECTIVE_SOURCE_FREEZE.json','Wrong parent source lineage');require(plan.get('device')=={'physical_index':0,'uuid':UUID},'Matched GPU mapping changed');require(plan.get('units')==UNITS,'Finite science roster changed')
 require(plan.get('available_authorized_GPU_count')==4 and plan.get('max_active_GPUs')==1,'Four available cards do not change single-card matched roster')
 for key in ['max_GPU_hours','hard_seconds','protective_seconds','cleanup_reserve_seconds','dispatch_max_seconds']:require(key in plan and plan[key]is None,'Elapsed resource cap must be explicitly absent:'+key)
 require(plan.get('no_progress_watchdog_seconds')==1800 and plan.get('cleanup_wait_seconds')==120,'Watchdog/cleanup observation contract changed');require(plan.get('numeric_tokens')==32 and plan.get('warmup_tokens')==128 and plan.get('measured_tokens')==128 and plan.get('math_gate')==MATH_GATE,'Frozen numerical contract changed');require(plan.get('minimum_disk_free_bytes')==10*1024**3 and plan.get('max_numeric_raw_bytes')==8*1024**3 and plan.get('min_gpu_free_bytes')==22*1024**3,'Storage/memory safety contract changed');require(isinstance(plan.get('binding_files'),list)and plan['binding_files'],'External bindings missing');return plan

def emit_progress(root,unit_id,phase,details):
 """Emit facts only when an actual phase/token/hash/capture advances."""
 require(unit_id in [x['id']for x in UNITS]and isinstance(phase,str)and phase and isinstance(details,dict),'Invalid real-progress fact');require(not any(k in details for k in ['heartbeat','elapsed_seconds','wall_clock_heartbeat']),'Timer heartbeat is not real progress');base=Path(root).resolve()/'execution'/unit_id;base.mkdir(parents=True,exist_ok=True);fact=dict(phase=phase,details=details);row=dict(event='ACTUAL_PROGRESS',task_id=TASK,unit_id=unit_id,process_identity=process_identity(os.getpid()),monotonic=time.monotonic(),fact_sha256=hashlib.sha256(canonical(fact)).hexdigest(),**fact)
 with(base/'PROGRESS_EVENTS.jsonl').open('a')as out:out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())

class ProgressObserver:
 def __init__(self,phase,unit,identity,clock=time.monotonic):self.phase=Path(phase);self.unit=unit;self.identity=identity;self.clock=clock;self.last_progress=clock();self.offset=0;self.pending=b'';self.seen=set();self.artifacts={};self.log_offset=0;self.log_pending=b''
 def poll(self):
  accepted=[];path=self.phase/'PROGRESS_EVENTS.jsonl'
  if path.exists():
   require(not path.is_symlink(),'Progress path symlink');size=path.stat().st_size;require(size>=self.offset,'Progress file truncated')
   with path.open('rb')as f:f.seek(self.offset);data=f.read();self.offset=f.tell()
   chunks=(self.pending+data).split(b'\n');self.pending=chunks.pop()
   for raw in chunks:
    if not raw:continue
    row=json.loads(raw);require(row.get('event')=='ACTUAL_PROGRESS'and row.get('task_id')==TASK and row.get('unit_id')==self.unit['id']and row.get('process_identity')==self.identity,'Unowned/malformed progress event');fact=dict(phase=row['phase'],details=row['details']);fingerprint=hashlib.sha256(canonical(fact)).hexdigest();require(row.get('fact_sha256')==fingerprint,'Progress fact hash differs')
    if fingerprint not in self.seen:self.seen.add(fingerprint);accepted.append(dict(kind='WORKER_PHASE_TOKEN_HASH_FACT',fact_sha256=fingerprint,phase=row['phase'],details=row['details']))
  log=self.phase/'worker.log'
  if log.exists():
   require(not log.is_symlink()and log.stat().st_size>=self.log_offset,'Worker stdout log truncated/unsafe')
   with log.open('rb')as stream:stream.seek(self.log_offset);data=stream.read();self.log_offset=stream.tell()
   lines=(self.log_pending+data).split(b'\n');self.log_pending=lines.pop()
   for raw in lines:
    try:row=json.loads(raw)
    except (ValueError,UnicodeError):continue
    if not(self.unit['mode']=='benchmark'and isinstance(row,dict)and row.get('event')=='PROGRESS'and row.get('unit')==self.unit['id']and type(row.get('consumed'))is int and row['consumed']in [128,256]):continue
    fact=dict(phase='FROZEN_BENCHMARK_TOKEN_PROGRESS',details=dict(consumed=row['consumed']));fingerprint=hashlib.sha256(canonical(fact)).hexdigest()
    if fingerprint not in self.seen:self.seen.add(fingerprint);accepted.append(dict(kind='FROZEN_BENCHMARK_STDOUT_TOKEN_FACT',fact_sha256=fingerprint,**fact))
  # Independent small committed numerical JSONs are scientific progress, not
  # monitor/ledger heartbeats. Never scan or rehash large raw tensors here.
  for path in sorted(self.phase.rglob('*.json')):
   if not(path.name.startswith('TOKEN_')or path.parent.name=='math'):continue
   require(not path.is_symlink()and path.stat().st_size<=2*1024**2,'Unsafe numerical progress receipt');metadata=(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)
   if self.artifacts.get(str(path),{}).get('metadata')==metadata:continue
   try:data=stable(path);json.loads(data)
   except (OSError,RuntimeError,ValueError,UnicodeError):continue # Writer has not committed a complete stable JSON yet.
   content=hashlib.sha256(data).hexdigest();prior=self.artifacts.get(str(path));self.artifacts[str(path)]=dict(metadata=metadata,sha256=content)
   if prior is None or prior['sha256']!=content:accepted.append(dict(kind='COMMITTED_NUMERICAL_JSON',path=str(path),bytes=len(data),sha256=content))
  if accepted:self.last_progress=self.clock()
  return accepted

def check_inputs(root,plan_path=None,progress=None):
 root=Path(root).resolve();pp=Path(plan_path).resolve()if plan_path else root/'qualification/GPU_PLAN.json';require(root in pp.parents,'Plan outside run');plan=validate_plan(read(pp));freeze_path=root/'SOURCE_FREEZE.json';fr=read(freeze_path);rows=fr.get('files');require(fr.get('task_id')==TASK and isinstance(rows,list)and len(rows)==407,'New effective407 required');seen={}
 for row in rows:
  require(row['path']not in seen,'Duplicate source path');seen[row['path']]=row;p=relative(root,row['path']);require(p.stat().st_size==row['bytes']and sha(p,progress)==row['sha256'],'Amended frozen source changed:'+row['path'])
 require(REPLACED<=set(seen),'Required four replacements absent');parent_path=relative(root,plan['parent_effective_manifest_path']);require(sha(parent_path)==PARENT_FREEZE,'Original effective407 parent differs');parent=read(parent_path);parent_rows={x['path']:x for x in parent['files']};require(len(parent_rows)==407 and set(parent_rows)==set(seen),'Parent/new source roster differs');require(all(seen[k]==parent_rows[k]for k in set(seen)-REPLACED),'An inherited403 source was modified');require(all(seen[k]!=parent_rows[k]for k in REPLACED),'All four amended source rows must replace the old versions');require(plan.get('request_sha256')==sha(root/'USER_REQUEST.txt'),'New request SHA differs')
 resource_path=root/'HUMAN_GPU_RESOURCE_AUTHORIZATION.json';resource=read(resource_path);resource_sha=sha(resource_path);require(plan.get('resource_authorization_sha256')==resource_sha,'Resource amendment file SHA differs');require(resource.get('task_id')==TASK and resource.get('ALLOW_GPU_RESTORE')=='YES'and resource.get('authorization_received')is True,'Latest explicit human resource authorization missing');require(resource.get('human_GPU_authorization_quote')and resource.get('latest_resource_amendment_quote'),'Both human authorization quotes required');require(resource.get('task_request_sha256')==sha(root/'USER_REQUEST.txt')and resource.get('parent_effective_source_freeze_sha256')==PARENT_FREEZE,'Resource request/lineage differs');require(resource.get('authorized_available_GPU_count')==4 and resource.get('max_active_GPUs_for_this_finite_comparison')==1 and resource.get('chosen_UUID_for_matched_comparison')==UUID,'Resource UUID/finite comparison scope differs');require(all(k in resource and resource[k]is None for k in ['GPU_elapsed_hard_limit_seconds','GPU_total_hours_hard_limit','dispatch_elapsed_hard_limit_seconds']),'Old elapsed cap reintroduced');require(resource.get('no_progress_watchdog_seconds')==1800 and resource.get('finite_science_units')==[u['id']for u in UNITS],'Resource watchdog/roster differs')
 bindings={}
 for name in plan['binding_files']:
  require(name in seen,'External binding metadata not frozen')
  for row in binding_rows(read(relative(root,name))):
   p=Path(row.get('resolved_path',row['path']));require(p.is_absolute(),'External source path not absolute');key=str(p.resolve());prior=bindings.get(key)
   if prior:require((prior['bytes'],prior['sha256'])==(row['bytes'],row['sha256']),'Conflicting dependency identity')
   else:bindings[key]=row
 require(len([x for x in bindings.values()if x['path'].endswith('.safetensors')])==4,'Four model shards required')
 for name,row in bindings.items():
  p=Path(name);require(p.is_file()and p.stat().st_size==row['bytes']and sha(p,progress)==row['sha256'],'Current external model/source/native identity differs:'+name)
 return dict(root=root,plan=plan,plan_path=pp,plan_sha256=sha(pp),source_freeze_sha256=sha(freeze_path),parent_effective_source_freeze_sha256=PARENT_FREEZE,request_sha256=sha(root/'USER_REQUEST.txt'),resource_authorization_sha256=resource_sha,source_files=len(rows),inherited_files=403,replaced_files=4,external_files=len(bindings))

IDENTITIES=['plan_sha256','source_freeze_sha256','parent_effective_source_freeze_sha256','request_sha256','resource_authorization_sha256','authorization_sha256']
def authorize(root,cli_yes,authorization,plan_path=None,progress=None):
 root=Path(root).resolve();require(cli_yes=='YES','GPU_NOT_AUTHORIZED: CLI YES absent');state=read(root/'ROOT.json');require(state.get('task_id')==TASK and state.get('ALLOW_GPU_RESTORE')=='YES','GPU_NOT_AUTHORIZED: new-task ROOT not YES');ap=Path(authorization).resolve();require(ap==root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','Independent authorization path differs');auth=read(ap);require(auth.get('task_id')==TASK and auth.get('scope')==SCOPE and auth.get('ALLOW_GPU_RESTORE')=='YES','New task/scope authorization missing');require(auth.get('device_uuid')==UUID and auth.get('available_authorized_GPU_count')==4 and auth.get('max_active_GPUs')==1,'Authorization device/available/active scope differs');require(all(k in auth and auth[k]is None for k in ['max_GPU_hours','dispatch_max_seconds','hard_seconds']),'No elapsed hard cap is authorized by latest amendment');require(auth.get('allowed_units')==[u['id']for u in UNITS],'Authorization science roster differs');require(auth.get('execution_root')==str(root),'Authorization exact new execution root differs');require(auth.get('human_request_quote')and auth.get('resource_amendment_quote'),'Original and latest human quotes missing');checked=check_inputs(root,plan_path,progress)
 for key in IDENTITIES[:-1]:require(auth.get(key)==checked[key],'Authorization source/request/resource identity differs:'+key)
 resource=read(root/'HUMAN_GPU_RESOURCE_AUTHORIZATION.json');require(auth['human_request_quote']==resource['human_GPU_authorization_quote']and auth['resource_amendment_quote']==resource['latest_resource_amendment_quote'],'Human quotes do not match independent resource amendment');checked.update(authorization=auth,authorization_path=ap,authorization_sha256=sha(ap));return checked

def require_worker_permit(root,mode,unit_id,permit_path,*,process_reader=None,parent_pid=None,monotonic=None):
 root=Path(root).resolve();reader=process_reader or process_identity;clock=monotonic or time.monotonic;phase=root/'execution'/unit_id;pp=Path(permit_path).resolve();require(pp==phase/'PERMIT.json','Wrong unit permit path');permit=read(pp);require(permit.get('task_id')==TASK and permit.get('scope')==SCOPE and permit.get('root')==str(root),'Wrong permit new-task/root/scope');checked=authorize(root,'YES',root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',root/permit['plan_relative_path'],progress=lambda phase,details:emit_progress(root,unit_id,phase,details));unit=next((u for u in UNITS if u['id']==unit_id),None);require(unit and permit.get('unit')==unit and mode==unit['mode'],'Unit/mode differs');meta=read(root/'execution/CONTROLLER_STARTED.json');parent=os.getppid()if parent_pid is None else parent_pid;identity=permit['controller_identity'];require(parent==identity['pid']and identity==meta['controller_identity']and reader(parent)==identity,'Worker is not exact direct child of controller');require(str(root/'scripts/restoration_controller.py')in identity['cmdline']and'--run'in identity['cmdline'],'Wrong actual controller cmdline')
 for key in IDENTITIES:require(permit.get(key)==checked[key]==meta[key],'Worker source/request/resource/full authorization binding differs:'+key)
 require(permit.get('device')==checked['plan']['device']and permit.get('no_progress_watchdog_seconds')==1800 and permit.get('dispatch_elapsed_hard_limit_seconds')is None,'Permit device/watchdog/resource amendment differs')
 for key,value in permit.get('environment_controls',{}).items():require((key not in os.environ)if value is None else os.environ.get(key)==value,'Worker startup environment differs:'+key)
 require(set(permit.get('environment_controls',{}))==set(ENV_KEYS),'Worker environment binding incomplete');require(clock()>=permit['dispatch_monotonic'],'Invalid dispatch monotonic identity');dispatch_path=phase/'DISPATCH.json';until=time.monotonic()+2
 while not dispatch_path.exists()and time.monotonic()<until:time.sleep(.01)
 dispatch=read(dispatch_path);own=reader(os.getpid());require(dispatch['worker_identity']==own and own['cmdline']==permit['worker_command']and dispatch['permit_sha256']==sha(pp),'Own PID/starttick/cmdline/permit differs');require(not(root/'FIRST_FAILURE.json').exists(),'Prior failure; no retry');return dict(permit,permit_sha256=sha(pp),worker_identity=own,checked_identity={k:checked[k]for k in IDENTITIES})
def run_controller(root,authorization,cli_yes,plan_path=None,*,gpu_reader=query_gpu,popen=subprocess.Popen,reader=process_identity,clock=time.monotonic,sleeper=time.sleep,progress_factory=ProgressObserver):
 checked=authorize(root,cli_yes,authorization,plan_path);root=checked['root'];plan=checked['plan'];execution=root/'execution';require(not(execution/'CONTROLLER_STARTED.json').exists()and not(execution/'CONTROLLER_ATTEMPT_STARTED.json').exists()and not(root/'FIRST_FAILURE.json').exists(),'Existing controller/STOP cannot retry');start=clock();ledger=Ledger(execution/'RESOURCE_EVENTS.jsonl',clock);identity=reader(os.getpid());atomic_new(execution/'CONTROLLER_ATTEMPT_STARTED.json',dict(task_id=TASK,controller_identity=identity,start_monotonic=start,authorization_sha256=checked['authorization_sha256']));results=[];identities=[];active=None;worker_id=None;unit=None;before=after=last_alive=exit_seen=released=None
 def observation(stage):
  obs=query_gpu(timeout=5)if gpu_reader is query_gpu else gpu_reader();ledger.event('NVML_OBSERVATION',stage=stage,unit_id=unit['id']if unit else None,worker_identity=worker_id,observation=obs);return obs
 def exit_event(rc):
  nonlocal exit_seen
  if exit_seen is None:exit_seen=clock();ledger.once(unit['id']+':exit','FIRST_PROCESS_EXIT_OBSERVED',unit=unit,returncode=rc,observed_monotonic=exit_seen,worker_identity=worker_id)
 def release_event():
  nonlocal released
  if released is None:released=clock();ledger.once(unit['id']+':release','FIRST_ALL_UUID_NVML_RELEASE_OBSERVED',unit=unit,observed_monotonic=released,worker_identity=worker_id)
 def cost():
  if unit is None or before is None or after is None:return
  ledger.once(unit['id']+':cost','UNIT_RESERVATION_INTERVAL_BOUNDS',unit_id=unit['id'],reservation_seconds_lower=max(0,(last_alive if last_alive is not None else after)-after),reservation_seconds_upper=max(0,released-before)if released is not None else'UNRESOLVED',last_alive_monotonic=last_alive,exit_observed_monotonic=exit_seen,all_UUID_release_observed_monotonic=released,actual_physical_GPU_hours='UNRESOLVED_NOT_ZERO')
 def cleanup():
  nonlocal last_alive
  errors=[]
  if active is None:return dict(status='NO_WORKER_DISPATCHED',NVML_release_verified='NOT_DISPATCHED',errors=errors)
  began=clock();end=began+plan['cleanup_wait_seconds'];killed=False
  if active.poll()is None:
   if worker_id is not None and same_process(worker_id,reader):terminate_exact(active,worker_id,reader)
   else:errors.append('NO_SIGNAL_UNVERIFIED_OR_REUSED_PID')
  while clock()<end:
   tick=clock();rc=active.poll()
   if rc is None:last_alive=tick
   else:exit_event(rc)
   try:
    obs=observation('CLEANUP');own=[x for x in obs['apps']if x['pid']==active.pid]
    if rc is not None and not own:release_event();break
   except Exception as exc:errors.append(type(exc).__name__+':'+str(exc))
   if not killed and clock()-began>=10 and active.poll()is None and worker_id is not None and same_process(worker_id,reader):active.kill();killed=True
   sleeper(.25)
  cost();return dict(status='PASS_EXIT_AND_ALL_UUID_RELEASE_VERIFIED'if released is not None else'UNRESOLVED_CLEANUP_RELEASE',NVML_release_verified=released is not None,exit_observed_monotonic=exit_seen,all_UUID_release_observed_monotonic=released,errors=errors)
 try:
  disk_guard(root,plan);obs=observation('INITIAL_ADMISSION');validate_gpu(obs,minimum_free=plan['min_gpu_free_bytes']/1024**2);meta={k:checked[k]for k in IDENTITIES};meta.update(task_id=TASK,scope=SCOPE,controller_identity=identity,device=plan['device'],start_monotonic=start,no_elapsed_GPU_or_dispatch_cap=True,no_progress_watchdog_seconds=1800,clock_host='THIS_CONTROLLER_HOST');atomic_new(execution/'CONTROLLER_STARTED.json',meta)
  for unit in UNITS:
   if unit['mode']=='benchmark':require(all(u in [r['unit_id']for r in results]for u in ['NUMERIC_FRESH0','NUMERIC_FRESH1']),'Both numeric fresh passes required before performance')
   if unit['id']=='NEW_FP_R0':require(all('OLD_FP_R'+str(i)in [r['unit_id']for r in results]for i in range(3)),'Three OLD results required before optional NEW')
   disk_guard(root,plan);validate_gpu(observation('PRE_DISPATCH'),minimum_free=plan['min_gpu_free_bytes']/1024**2);phase=execution/unit['id'];require(not phase.exists(),'Unit path exists; no retry');phase.mkdir(parents=True);pp=phase/'PERMIT.json';command=[sys.executable,'-B',str(root/'worker/restoration_worker.py'),'--root',str(root),'--mode',unit['mode'],'--unit',unit['id'],'--permit',str(pp)];env=worker_environment(unit);before=clock();after=last_alive=exit_seen=released=None;worker_id=None;permit=dict(task_id=TASK,scope=SCOPE,root=str(root),unit=unit,device=plan['device'],controller_identity=identity,worker_command=command,environment_controls={k:env.get(k)for k in ENV_KEYS},plan_relative_path=str(checked['plan_path'].relative_to(root)),dispatch_monotonic=before,dispatch_elapsed_hard_limit_seconds=None,no_progress_watchdog_seconds=1800,**{k:checked[k]for k in IDENTITIES});permit_sha=atomic_new(pp,permit)
   with(phase/'worker.log').open('xb')as log:active=popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
   after=clock();candidate=reader(active.pid);require(candidate['pid']==active.pid and candidate['cmdline']==command and candidate not in identities,'Actual worker PID/cmdline not verified; do not signal foreign process');worker_id=candidate;identities.append(worker_id);atomic_new(phase/'DISPATCH.json',dict(worker_identity=worker_id,permit_sha256=permit_sha,dispatch_before_monotonic=before,dispatch_after_monotonic=after));ledger.once(unit['id']+':dispatch','FIRST_DISPATCH',unit=unit,worker_identity=worker_id,before_monotonic=before,after_monotonic=after);progress=progress_factory(phase,unit,worker_id,clock)
   while True:
    tick=clock();rc=active.poll()
    if rc is None:last_alive=tick
    else:exit_event(rc)
    facts=progress.poll()
    for fact in facts:ledger.event('REAL_PROGRESS_OBSERVED',unit_id=unit['id'],worker_identity=worker_id,fact=fact)
    require(clock()-progress.last_progress<plan['no_progress_watchdog_seconds'],'NO_REAL_PROGRESS_WATCHDOG_1800_SECONDS');disk_guard(root,plan);obs=observation('WORKER_POLL');validate_gpu(obs,own_pid=active.pid);own=[x for x in obs['apps']if x['pid']==active.pid]
    if exit_seen is not None and not own:release_event();break
    sleeper(.25)
   cost();require(active.returncode==0,'Worker failed; preserve true failure, no retry');require(not same_process(worker_id,reader),'Worker still exists after exit observation');result=validate_result(phase,unit,worker_id,permit_sha);results.append(result);active=None;worker_id=None
  final=dict(status='COMPLETE_FINITE_RECOVERY_UNITS',task_id=TASK,scope=SCOPE,completed_units=[x['unit_id']for x in results],benchmark_repeats=3,minimum_numeric_complete=True,elapsed_controller_seconds=clock()-start,elapsed_hard_cap=None,dispatch_elapsed_hard_cap=None,actual_physical_GPU_hours='UNRESOLVED_NOT_ZERO',NVML_sample_peak_scope='DERIVE_FROM_FULL_RECORDED_OBSERVATIONS_NOT_CONTINUOUS_EXACT_PEAK',no_retry=True);atomic_new(execution/'CONTROLLER_RESULT.json',final);return final
 except BaseException as exc:
  cleanup_result=cleanup();failure=dict(status='FAIL_CLOSED_STOP',task_id=TASK,scope=SCOPE,error=type(exc).__name__+':'+str(exc),unit=unit,cleanup=cleanup_result,traceback=traceback.format_exc(),no_retry=True)
  if not(root/'FIRST_FAILURE.json').exists():atomic_new(root/'FIRST_FAILURE.json',failure)
  if not(execution/'CONTROLLER_RESULT.json').exists():atomic_new(execution/'CONTROLLER_RESULT.json',failure)
  ledger.event('FAIL_CLOSED_STOP',failure=failure);raise

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--check',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--allow-gpu-restore',choices=['NO','YES'],default='NO');ap.add_argument('--authorization',type=Path);ap.add_argument('--plan',type=Path);args=ap.parse_args();require(not(args.check and args.run),'Choose check or run')
 if not args.run:
  checked=check_inputs(args.root,args.plan);print(json.dumps(dict(status='PASS_CPU_NEW_RESOURCE_SOURCE_BINDINGS_NOT_GPU_QUALIFICATION',task_id=TASK,source_files=checked['source_files'],inherited_files=403,replaced_files=4,external_files=checked['external_files'],GPU_queries=0,Driver_calls=0,GPU_dispatches=0)));return
 require(args.authorization is not None,'Independent new-task human authorization required');print(json.dumps(run_controller(args.root,args.authorization,args.allow_gpu_restore,args.plan)))
if __name__=='__main__':
 try:main()
 except BaseException as exc:print(json.dumps(dict(status='DENIED_OR_FAIL_CLOSED_STOP',error=type(exc).__name__+':'+str(exc))),file=sys.stderr);sys.exit(2)
