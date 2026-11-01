"""Prospective V2 lifecycle helpers. No scientific numerical implementation."""
import ctypes,ctypes.util,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
TASK='QWEN_GDN_C6R_C7_4X3090_PERSISTENT_WORKER_READINESS_V2'
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
V1=REPO/'experiments/QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1'
BASE=Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
OLD=BASE/'experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1'
FORMAL=OLD/'scripts/run_qwen_c5_c6_formal.py'
CORE=BASE/'experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
ROTATION=BASE/'experiments/shared/rotation/cayley_rotation.py'
PRIOR=Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1')
MODEL=Path('/data/zypan/modelscope_models/Qwen3.5-9B')
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
PARENT='f17d69df1a7281dccfd82f21c1bee45cf791b6b0'
BRANCH='exp/qwen-gdn-c6r-c7-persistent-worker-readiness-v2'
UUIDS=['GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75']
SHAS={'C6R':'a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6','C7':'57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8'}
def require(v,m):
 if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
 require(not p.exists(),'refuse overwrite '+str(p))
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def cmd(a):return subprocess.check_output(a,cwd=REPO,text=True).strip()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def gmodule():
 for p in ('/data/zypan/transformers-qwen35/src','/data/zypan/experiments/qwen35_gdn_quant'):
  if p not in sys.path:sys.path.insert(0,p)
 os.environ['QWEN_TRANSFORMERS_SRC']='/data/zypan/transformers-qwen35/src'
 return mod('v2_unchanged_formal',FORMAL)
def verify_manifest():
 for line in (ROOT/'manifests/v2_premeasurement_sha256.txt').read_text().splitlines():
  d,p=line.split('  ',1);require(sha(p)==d,'premeasurement drift '+p)
 require(read(ROOT/'analysis/production_runner_scientific_path_identity.json')['status']=='PASS','scientific path audit failed')
def rotation_paths(arm):return V1/'rotations'/arm/'frozen_rotation_fp32.pt',V1/'configs'/f'{arm.lower()}_rotation_manifest.json'
class Nvml:
 class Memory(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
 class Process(ctypes.Structure):_fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]
 def __init__(self):
  self.lib=ctypes.CDLL(ctypes.util.find_library('nvidia-ml'));self.check(self.lib.nvmlInit_v2())
 def check(self,rc):require(rc==0,'NVML rc='+str(rc))
 def snapshot(self,gpu):
  handle=ctypes.c_void_p();self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(gpu),ctypes.byref(handle)))
  u=ctypes.create_string_buffer(96);self.check(self.lib.nvmlDeviceGetUUID(handle,u,96))
  mem=self.Memory();self.check(self.lib.nvmlDeviceGetMemoryInfo(handle,ctypes.byref(mem)))
  n=ctypes.c_uint(1024);arr=(self.Process*1024)();self.check(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(handle,ctypes.byref(n),arr))
  rows=[dict(pid=int(p.pid),used_bytes=int(p.usedGpuMemory)) for p in arr[:n.value]]
  return dict(gpu=gpu,uuid=u.value.decode(),used_bytes=int(mem.used),free_bytes=int(mem.free),processes=rows,resolution='NVML_BYTES',time=time.time())
