import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
UUID="GPU-a78176ed-f97f-c54f-6201-5a01b570f422"


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

def worker_environment(unit):
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 for key in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']:env.pop(key,None)
 if unit['id']=='NEW_FP_R0':env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_DEVICE_ORDER='PCI_BUS_ID')
 return env
