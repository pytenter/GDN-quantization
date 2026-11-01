"""Task/permit/source/device admission. Import has no torch or GPU API calls."""
import os,sys,json,hashlib,subprocess,time,ast,threading
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'inherited/group32/src'))
from inherited_device_guards import require,sha,read,atomic_new,process_identity,same_process,binding_rows,query_gpu
UUIDS=('GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75')
TASK='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
EXECUTION_REVISION=TASK+'_REDUCTION_REPAIR_20261025'
EXECUTION_ROOT='/tmp/'+EXECUTION_REVISION
ORIGIN_SOURCE_ROOT='/tmp/'+TASK
ORIGIN_SOURCE_SHA256='ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7'
CALIBRATION_REUSE_STATUS='PASS_REUSED_FROZEN_NONAIME_CALIBRATION_DUAL_HOST_FRESH_SHA_NO_RECOLLECT_NO_RESELECT'
AUTHORIZED_GPUS=(1,2,3)
METHODS=('W','R','RW')
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def objsha(x):return hashlib.sha256(canonical(x)).hexdigest()
def status(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid())+'.'+str(threading.get_ident()))
 with t.open('xb') as f:f.write(canonical(obj)+b'\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def event(path,kind,**v):
 import fcntl
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(dict(event=kind,monotonic=time.monotonic(),unix=time.time(),**v),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def verify_sources(root):
 rows=read(root/'SOURCE_FREEZE.json')['files']
 for x in rows:
  p=root/x['path'];require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Frozen execution source differs:'+x['path'])
 return sha(root/'SOURCE_FREEZE.json')
def authorize(root,permit):
 root=Path(root).resolve();p=read(permit)
 require(str(root)==EXECUTION_ROOT and p.get('execution_root')==str(root),'Exact new reduction-repair execution root required')
 h=read(root/'operations/HUMAN_TASK_AUTHORIZATION.json');repair=read(root/'operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json');reuse=read(root/'CALIBRATION_REUSE_ADMISSION.json')
 require(p.get('reduction_repair_authorization_sha256')==sha(root/'operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json') and p.get('calibration_reuse_admission_sha256')==sha(root/'CALIBRATION_REUSE_ADMISSION.json'),'Repair authorization or calibration-reuse pin differs')
 require(repair['task_id']==TASK and repair['new_remote_root']==EXECUTION_ROOT and repair['execution_revision']==EXECUTION_REVISION and repair['authorized_physical_GPUs']==list(AUTHORIZED_GPUS) and repair['old_qualification_credit']==0 and repair['old_formal_credit']==0,'Repair grant identity differs')
 require(reuse['task_id']==TASK and reuse['status']==CALIBRATION_REUSE_STATUS and reuse['origin_source_root']==ORIGIN_SOURCE_ROOT and reuse['origin_source_sha256']==ORIGIN_SOURCE_SHA256,'Calibration reuse origin/status differs')
 require(reuse['selected_a_b']==[.5,.25] and reuse['new_qualification_credit']==0 and reuse['new_formal_credit']==0 and reuse['no_reselection'] is True,'Calibration reuse selection/credit differs')
 require(p['task_id']==TASK and h['task_id']==TASK and p['ALLOW_GPU']=='YES','Task-specific grant missing')
 require(p['devices']==[UUIDS[g] for g in AUTHORIZED_GPUS] and p['physical_indices']==list(AUTHORIZED_GPUS),'Only55 GPU1/2/3 are authorized')
 require(p['source_freeze_sha256']==verify_sources(root) and p['plan_sha256']==sha(root/'GPU_EXECUTION_PLAN.json') and p['human_grant_sha256']==sha(root/'operations/HUMAN_TASK_AUTHORIZATION.json'),'Permit identities changed')
 require(all(p[k] is None for k in ('GPUh_hard_cap','total_wall_cap','unit_elapsed_cap')) and p['watchdog_seconds']==1800 and p['cleanup_seconds']==120,'Resource policy differs')
 require(p['max_new_tokens']==81920 and p['formal_new_samples']==90 and p['seed']==1,'Formal protocol differs')
 require(p['calibration_freeze_sha256']==sha(root/'CALIBRATION_DATA_FREEZE.json') and p['omega_manifest_sha256']==sha(root/'TRANSPORTED_QUERY_OMEGA.json') and p['parameter_selection_sha256']==sha(root/'CALIBRATION_PARAMETER_SELECTION.json'),'Frozen shared calibration/selection differs')
 return p

def model_guard(root):
 for x in read(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
  s=Path(x['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns']),'Model/tokenizer stat changed:'+x['path'])
def bindings(root):
 base=root/'inherited/group32/inherited/historical_port_payload/analysis/environment';rows=[]
 for n in ['PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json','CPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json']:rows+=binding_rows(read(base/n))
 return {str(Path(x['path']).resolve()):x for x in rows}
def verify_external(root):
 rows=bindings(root)
 for p,x in rows.items():require(Path(p).stat().st_size==x['bytes'] and sha(p)==x['sha256'],'External binding changed:'+p)
 return len(rows)
def device(gpu,own_pid=None,minimum_free=23000):
 require(type(gpu)==int and gpu in AUTHORIZED_GPUS,'Invalid selected physical GPU');obs=query_gpu();d=[x for x in obs['devices'] if x['physical_index']==gpu]
 require(len(d)==1 and d[0]['uuid']==UUIDS[gpu] and '3090' in d[0]['name'] and d[0]['memory_total_MiB']>=24000,'Physical device differs')
 require(not [x for x in obs['apps'] if x['uuid']==UUIDS[gpu] and x['pid']!=own_pid],'Foreign process on assigned GPU; never kill it')
 if own_pid is None:require(d[0]['memory_free_MiB']>=minimum_free,'Insufficient exclusive VRAM')
 else:
  own=[x for x in obs['apps'] if x['pid']==own_pid];require(own and all(x['uuid']==UUIDS[gpu] for x in own),'Actual worker owner-PID UUID differs or absent')
 return obs

def driver_identity(root,gpu,driver_factory=None,environment=None):
 # Same qualified driver function, only selected physical CVD/UUID admission changes.
 p=root/'inherited/group32/src/inherited_device_guards.py';tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_driver_ordinal_identity');changed=0
 for n in ast.walk(fn):
  if isinstance(n,ast.Compare) and isinstance(n.left,ast.Call) and isinstance(n.left.func,ast.Attribute) and n.left.func.attr=='get' and n.left.args and isinstance(n.left.args[0],ast.Constant) and n.left.args[0].value=='CUDA_VISIBLE_DEVICES':
   require(len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value=='0','Driver admission AST unexpected');n.comparators[0]=ast.Constant(UUIDS[gpu]);changed+=1
 require(changed==1,'Driver physical selector adapter must change exactly1 comparison')
 ns=dict(Path=Path,require=require,sha=sha,os=os,UUID=UUIDS[gpu]);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),ns)
 return ns[fn.name](list(bindings(root).values()),driver_factory=driver_factory,environment=environment)

def library_check(root,gpu):
 allowed=bindings(root);maps=Path('/proc/self/maps').read_text();selected={};segments=[];missing=[]
 words=('libtorch','libc10','libcudart','libcublas','libcudnn','libnvrtc','libcuda.','libtorchaudio')
 for line in maps.splitlines():
  fields=line.split(maxsplit=5)
  if len(fields)!=6 or not fields[5].startswith('/'):continue
  p=Path(fields[5]);key=str(p.resolve());relevant=('.so' in p.name and '/site-packages/torch/' in key) or any(x in p.name for x in words)
  if not relevant:continue
  x=allowed.get(key)
  if x is None:missing.append(key);continue
  a=p.stat();require(int(fields[4])==a.st_ino,'Mapped library inode differs')
  segments.append(dict(path=key,mapped_inode=int(fields[4]),actual_inode=a.st_ino))
  if key not in selected:
   h=sha(p);z=p.stat();selected[key]=dict(path=key,sha256=h,bytes=z.st_size,match=h==x['sha256'] and z.st_size==x['bytes'] and (a.st_ino,a.st_mtime_ns)==(z.st_ino,z.st_mtime_ns))
 v=dict(status='PASS' if selected and not missing and all(x['match'] for x in selected.values()) else 'FAIL',full_maps=maps,segments=segments,selected=list(selected.values()),missing=sorted(set(missing)),identity=device(gpu,os.getpid()))
 return v

def worker_environment(gpu):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 for k in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']:env.pop(k,None)
 return env

def progress(target,phase,**v):status(target/'PROGRESS.json',dict(phase=phase,monotonic=time.monotonic(),**v))
def storage_guard(root):
 s=os.statvfs(root);require(s.f_bavail*s.f_frsize>=4*1024**3,'Independent output filesystem headroom below4GiB')
 shared=os.statvfs('/data');require(shared.f_bavail*shared.f_frsize>=8*1024**3,'Shared read-only dependency filesystem headroom below8GiB')
 # Scientific output bound, excluding the immutable source and CPU preparation.
 used=sum(p.stat().st_blocks*512 for folder in ['execution','formal'] for p in (root/folder).rglob('*') if p.is_file())
 require(used<=2*1024**3,'New execution small-output2GiB bound reached; stop before exhausting disk')
 return dict(free_bytes=s.f_bavail*s.f_frsize,shared_data_free_bytes=shared.f_bavail*shared.f_frsize,allocated_output_bytes=used)
def reference_root():return Path('/dev/shm/qwen_sym8_g32_wscale_rowbal_aime26_seed1_v1_reduction_repair_20261025')

def admit_inherited(root):
 # Historical source/results remain outside this task's execution and are never
 # relabelled fresh. An independent fresh reader must populate this receipt.
 p=root/'HISTORICAL_G32_BASELINE_REUSE.json';r=read(p)
 require(r['status']=='PASS_READONLY_HISTORICAL_IDENTITY_AND_BASELINES','Historical independent audit is not PASS')
 require(r['task_id']==TASK and r['historical_root_readonly'] is True,'Historical provenance scope differs')
 require(r['formal_generations_reused_as_new']==0 and r['historical_scores']=={'H+G32+Sym8':21,'H+G32+Asym8':25},'Historical read-only controls differ')
 require(r['runtime_math_qualified'] is True and r['historical_formal_complete']==120,'Original qualified full runtime missing')
 return dict(status='PASS_READONLY_ORIGINAL_G32_RUNTIME_NOT_NEW_QUALIFICATION',receipt_sha256=sha(p),new_formal_credit=0)

def validate_jobs(root,jobs):
 from control_plan import validate_formal_jobs
 return validate_formal_jobs(read(root/'AIME26_DATASET_FREEZE.json'),read(root/'QUESTION_GPU_ASSIGNMENT.json'),jobs,UUIDS,AUTHORIZED_GPUS)
