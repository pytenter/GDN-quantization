"""Task/permit/source/device admission. Import has no torch or GPU API calls."""
import os,sys,json,hashlib,subprocess,time,ast,threading
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from inherited_device_guards import require,sha,read,atomic_new,process_identity,same_process,binding_rows,query_gpu
NODE=read(R/'NODE.json')
UUIDS={NODE['physical_gpu']:NODE['uuid']}
ENV_ROOT=Path(NODE['environment_root'])
USER_SITE=Path(NODE['user_site'])
TASK='QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1'
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
 require(Path(root).resolve()==R.resolve(),'Only unique current science root may dispatch')
 p=read(permit);h=read(root/'HUMAN_GPU_AUTHORIZATION.json');n=read(root/'NODE.json')
 require(n==NODE and any(x['id']==n['id']and x['host']==n['host']and x['physical_gpu']==n['physical_gpu']and x['uuid']==n['uuid']for x in h['only_nodes']),'Host-specific human grant differs')
 require(p['task_id']==TASK and p['ALLOW_GPU']=='YES' and p['node_id']==n['id']and p['host']==n['host']and p['devices']==list(UUIDS.values()),'Node task/device permit missing')
 require(p['source_freeze_sha256']==verify_sources(root)and p['plan_sha256']==sha(root/'GPU_EXECUTION_PLAN.json')and p['global_plan_sha256']==sha(root/'CROSS_HOST_PLAN.json')and p['human_grant_sha256']==sha(root/'HUMAN_GPU_AUTHORIZATION.json'),'Permit/source/global plan identities changed')
 require(all(p[k]is None for k in ('GPUh_hard_cap','total_wall_cap','unit_elapsed_cap'))and p['watchdog_seconds']==1800 and p['cleanup_seconds']==120,'Resource policy differs')
 return p

def model_guard(root):
 for x in read(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
  s=Path(x['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns']),'Model/tokenizer stat changed:'+x['path'])
def bindings(root):return {str(Path(x['path']).resolve()):x for x in read(root/'analysis/EXTERNAL_NATIVE_BINDINGS.json')['files']}
def verify_external(root):
 rows=bindings(root);other=read(root/'analysis/EXTERNAL_PYTHON_BINDINGS.json')['files']
 for x in list(rows.values())+other:
  p=Path(x['path']);require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'External binding changed:'+str(p))
 return len(rows)+len(other)

def device(gpu,own_pid=None,minimum_free=23000):
 require(type(gpu)==int and gpu in UUIDS,'Invalid selected physical GPU');obs=query_gpu();d=[x for x in obs['devices'] if x['physical_index']==gpu]
 require(len(d)==1 and d[0]['uuid']==UUIDS[gpu] and '3090' in d[0]['name'] and d[0]['memory_total_MiB']>=24000,'Physical device differs')
 require(not [x for x in obs['apps'] if x['uuid']==UUIDS[gpu] and x['pid']!=own_pid],'Foreign process on assigned GPU; never kill it')
 if own_pid is None:require(d[0]['memory_free_MiB']>=minimum_free,'Insufficient exclusive VRAM')
 else:
  own=[x for x in obs['apps'] if x['pid']==own_pid];require(own and all(x['uuid']==UUIDS[gpu] for x in own),'Actual worker owner-PID UUID differs or absent')
 return obs

def driver_identity(root,gpu,driver_factory=None,environment=None):
 # Same qualified driver function, only selected physical CVD/UUID admission changes.
 p=root/'src/inherited_device_guards.py';tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_driver_ordinal_identity');changed=0
 for n in ast.walk(fn):
  if isinstance(n,ast.Compare) and isinstance(n.left,ast.Call) and isinstance(n.left.func,ast.Attribute) and n.left.func.attr=='get' and n.left.args and isinstance(n.left.args[0],ast.Constant) and n.left.args[0].value=='CUDA_VISIBLE_DEVICES':
   require(len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value=='0','Driver admission AST unexpected');n.comparators[0]=ast.Constant(UUIDS[gpu]);changed+=1
 require(changed==1,'Driver physical selector adapter must change exactly1 comparison')
 ns=dict(Path=Path,require=require,sha=sha,os=os,UUID=UUIDS[gpu]);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),ns)
 return ns[fn.name]([x for p,x in bindings(root).items() if p.startswith('/usr/') and Path(p).name.startswith('libcuda.so')],driver_factory=driver_factory,environment=environment)

def library_check(root,gpu):
 allowed=bindings(root);maps=Path('/proc/self/maps').read_text();selected={};segments=[];missing=[]
 words=('libtorch','libc10','libcudart','libcublas','libcudnn','libnvrtc','libcuda.','libtorchaudio')
 for line in maps.splitlines():
  fields=line.split(maxsplit=5)
  if len(fields)!=6 or not fields[5].startswith('/'):continue
  p=Path(fields[5]);key=str(p.resolve());relevant=('.so' in p.name and '/site-packages/torch/' in key) or any(x in p.name for x in words) or ('.so' in p.name and (str(ENV_ROOT) in key or str(USER_SITE) in key)) or p.name.startswith('libnvidia')
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
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(USER_SITE))
 for k in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']:env.pop(k,None)
 return env

def progress(target,phase,**v):status(target/'PROGRESS.json',dict(phase=phase,monotonic=time.monotonic(),**v))
def storage_guard(root):
 s=os.statvfs(root);require(s.f_bavail*s.f_frsize>=8*1024**3,'Compute disk headroom below8GiB')
 # Scientific output bound, excluding the immutable source and CPU preparation.
 used=sum(p.stat().st_blocks*512 for folder in ['execution','formal'] for p in (root/folder).rglob('*') if p.is_file())
 require(used<=2*1024**3,'New execution small-output2GiB bound reached; stop before exhausting disk')
 return dict(free_bytes=s.f_bavail*s.f_frsize,allocated_output_bytes=used)
def reference_root():return Path('/dev/shm/qwen_group64_four_codec_aime26_seed1_v1_cross_host_v4_'+NODE['id'])

def validate_jobs(root,jobs):
 assignment={x['problem_id']:x for x in read(root/'QUESTION_GPU_ASSIGNMENT.json')['assignments']if x['node_id']==NODE['id']};records={x['problem_id']:x for x in read(root/'AIME26_DATASET_FREEZE.json')['records']}
 require(len(jobs)==60 and len(assignment)==15,'Node question roster must15bundles/60samples')
 require(len({(j['problem_id'],j['condition'],j['seed'])for j in jobs})==60,'Repeated/missing condition')
 for pid,a in assignment.items():
  rows=[j for j in jobs if j['problem_id']==pid]
  require([j['condition']for j in rows]==a['conditions']and set(a['conditions'])=={'G64-0','G64-1','G64-2','G64-3'},'Frozen rotated bundle order differs')
  for j in rows:require(j['node_id']==NODE['id']and j['compute_host']==NODE['host']and j['gpu']==NODE['physical_gpu']and j['gpu_uuid']==NODE['uuid']and j['seed']==1 and j['max_new_tokens']==81920 and j['input_ids_sha256']==records[pid]['input_ids_sha256']and j['input_tokens']==records[pid]['input_tokens'],'Question/host/UUID/prompt bundle differs')
 require(all(j['problem_id']in assignment for j in jobs),'Node dispatched a different node question')
 return True
