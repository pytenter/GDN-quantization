"""Task/permit/source/device admission. Import has no torch or GPU API calls."""
import os,sys,json,hashlib,subprocess,time,ast,threading
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'inherited/group32/src'))
from inherited_device_guards import require,sha,read,atomic_new,process_identity,same_process,binding_rows,query_gpu
UUIDS=('GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75')
TASK='QWEN_GDN_ASYM_INT4_GROUP32_AIME26_SEED1_V1'
SELECTED_GPU=3
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
 p=read(permit);h=read(root/'HUMAN_GPU_AUTHORIZATION.json')
 require(p['task_id']==TASK and p['ALLOW_GPU']=='YES' and h['task_id']==TASK,'Task-specific authorization missing')
 require(p['devices']==[UUIDS[SELECTED_GPU]] and p['physical_index']==SELECTED_GPU and h['selected_uuid']==UUIDS[SELECTED_GPU],'Only the authorized idle GPU3 may be used')
 require(h['request_sha256']==sha(root/'USER_REQUEST.txt'),'Human attached instruction identity changed')
 require(read(root/'analysis/FINAL_PRELAUNCH_CPU_REVIEW.json')['status']=='PASS_FINAL_PRELAUNCH_CPU_REVIEW' and read(root/'qualification/INDEPENDENT_CPU_COMPARISON.json')['status']=='PASS_TWO_SEPARATE_PROCESS_SUITES_EXACT','New INT4 independent CPU qualification missing')
 require(p['source_freeze_sha256']==verify_sources(root) and p['plan_sha256']==sha(root/'GPU_EXECUTION_PLAN.json') and p['human_grant_sha256']==sha(root/'HUMAN_GPU_AUTHORIZATION.json'),'Frozen permit identities changed')
 require(all(p[k] is None for k in ('GPUh_hard_cap','total_wall_cap','unit_elapsed_cap')) and p['watchdog_seconds']==1800 and p['cleanup_seconds']==120,'Resource policy differs')
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
 require(type(gpu)==int and gpu==SELECTED_GPU,'Invalid selected physical GPU');obs=query_gpu();d=[x for x in obs['devices'] if x['physical_index']==gpu]
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
 s=os.statvfs(root);require(s.f_bavail*s.f_frsize>=8*1024**3,'Compute disk headroom below8GiB')
 # Scientific output bound, excluding the immutable source and CPU preparation.
 used=sum(p.stat().st_blocks*512 for folder in ['execution','formal'] for p in (root/folder).rglob('*') if p.is_file())
 require(used<=2*1024**3,'New execution small-output2GiB bound reached; stop before exhausting disk')
 return dict(free_bytes=s.f_bavail*s.f_frsize,allocated_output_bytes=used)
def reference_root():return Path('/dev/shm/qwen_gdn_asym_int4_group32_aime26_seed1_v1')

def admit_inherited(root):
 p=root/'evidence/group32/delivery/SHA256_MANIFEST.json';require(sha(p)=='8aadaba31dee663b72a59abe8ee48fdce609f1919e469af41eb61a4c0c63d6ba','Final old qualification manifest differs')
 m=read(p);x=next(x for x in m['files'] if x['path']=='TASK_RESULT.json');q=root/'evidence/group32/delivery/TASK_RESULT.json';require(q.stat().st_size==x['bytes'] and sha(q)==x['sha256'],'Bound original qualification result differs');r=read(q);require(r['status']=='INT8_SHORT_RUNTIME_QUALIFIED','Original not qualified')
 for b in ['B1','B2']:require(r['conditions'][b]['passed_fresh_lengths']==[32,128,512] and r['conditions'][b]['GPU_codec']=='GPU_CODEC_QUALIFIED','Inherited Q2/Q3 GPU0 evidence incomplete')
 return dict(status='PASS_BOUND_ORIGINAL_Q2_Q3_GPU0_512_REUSE',old_manifest_sha256=sha(p),old_task_result_sha256=sha(q))

def validate_jobs(root,jobs):
 records={x['problem_id']:x for x in read(root/'AIME26_DATASET_FREEZE.json')['records']}
 require(len(jobs)==60 and len(records)==30,'Expected exactly30x2 seed1 samples')
 require(len({(j['problem_id'],j['condition'],j['seed']) for j in jobs})==60,'Repeated/missing sample')
 for n,(pid,rec) in enumerate(sorted(records.items()),1):
  rows=[j for j in jobs if j['problem_id']==pid]
  require([j['condition'] for j in rows]==(['I4-N','I4-H'] if n%2 else ['I4-H','I4-N']),'Frozen alternating order changed')
  for j in rows:
   require(j['gpu']==SELECTED_GPU and j['gpu_uuid']==UUIDS[SELECTED_GPU] and j['seed']==1 and j['input_ids_sha256']==rec['input_ids_sha256'] and j['input_tokens']==rec['input_tokens'] and j['max_new_tokens']==81920,'SingleUUID/prompt/seed/cap binding changed')
 return True
