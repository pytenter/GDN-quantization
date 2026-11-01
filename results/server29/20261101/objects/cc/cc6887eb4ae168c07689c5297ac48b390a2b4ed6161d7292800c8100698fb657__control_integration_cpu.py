"""Real production controller/worker/strict-inspector CPU integration fixtures.

Only model/device/native/runtime endpoints are explicit CPU mocks. Production
controller, worker, payload validator, plan, authorization/source verification,
strict qualification inspector, Popen lifecycle, signals and formal barrier run.
Every fixture is outside the science root. No mocked output is GPU/AIME credit.
"""
import argparse,ast,copy,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,tempfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
PHASE_A_SHA='c7365ecc1b6bae3252c1e13ffde41cc5aaa604768f662aa60f6de15a246f0849'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def require(ok,msg):
 if not ok:raise RuntimeError(msg)

CONTROL_SHIM=r'''
import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
assert os.environ.get('WSCALE_CPU_FIXTURE_ONLY')=='YES','CPU fixture cannot run in science environment'
ROOT=Path(os.environ['WSCALE_CPU_FIXTURE_ROOT']);sys.path.insert(0,str(ROOT/'inherited/group32/src'))
spec=importlib.util.spec_from_file_location('_actual_frozen_execution_control',ROOT/'production/execution_control.py');real=importlib.util.module_from_spec(spec);spec.loader.exec_module(real)
def actual_identity(pid):
 p=Path('/proc')/str(pid)
 if p.exists():
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),starttick=int(fields[19]),cmdline=[b.decode() for b in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])
 line=''
 for attempt in range(40):
  line=subprocess.check_output(['ps','-ww','-p',str(pid),'-o','lstart=','-o','command='],text=True).strip()
  if not line:raise ProcessLookupError(pid)
  command=line[24:].strip()
  complete=(not command.startswith('(')) and ('.py' in command or ' -c ' in command)
  if 'gpu_unit_worker.py' in command:complete=complete and ' --unit ' in command
  if 'gpu_controller.py' in command:complete=complete and ' --root ' in command and ' --execute' in command
  if complete:break
  time.sleep(.002)
 command=line[24:].strip();parts=command.split(' ',1)
 # macOS python3 launcher execs its framework binary after Popen returns.
 # Normalize only this CPU fixture's Python executable spelling; argument
 # vector and process start identity remain checked. Linux uses actual/proc.
 if 'python' in parts[0].lower():command='CPU_FIXTURE_PYTHON '+(parts[1] if len(parts)>1 else '')
 return dict(pid=int(pid),starttick=int(hashlib.sha256(line[:24].encode()).hexdigest()[:12],16),cmdline=[command])
def identity(pid):
 p=ROOT/'state/REUSED_PID.json'
 if p.exists():
  x=json.loads(p.read_text())
  if int(pid)==x['original']['pid']:return x['replacement']
 return actual_identity(pid)
def same(expected):
 if (ROOT/'state'/('exit_transition_'+str(expected['pid'])+'.json')).exists():return False
 if (ROOT/'state'/('closed_'+str(expected['pid'])+'.json')).exists():return False
 try:
  current=identity(expected['pid'])
  if current!=expected:
   try:real.atomic_new(ROOT/'state'/('IDENTITY_DIFF_'+str(os.getpid())+'_'+str(time.time_ns())+'.json'),dict(expected=expected,current=current))
   except BaseException:pass
  return current==expected
 except (OSError,ValueError,subprocess.SubprocessError):return False
def actual_same(expected):
 try:return actual_identity(expected['pid'])==expected
 except (OSError,ValueError,subprocess.SubprocessError):return False
def register(gpu):
 p=ROOT/'state'/('worker_'+str(os.getpid())+'.json');p.parent.mkdir(exist_ok=True)
 real.atomic_new(p,dict(identity=actual_identity(os.getpid()),gpu=gpu))
def query():
 devices=[dict(physical_index=g,uuid=real.UUIDS[g],name='CPU_MOCK_RTX3090',memory_total_MiB=24576,memory_free_MiB=24576) for g in range(4)];apps=[];retired=[]
 for p in sorted((ROOT/'state').glob('worker_*.json')):
  x=json.loads(p.read_text())
  closed=ROOT/'state'/('closed_'+str(x['identity']['pid'])+'.json')
  if not closed.exists() and actual_same(x['identity']):apps.append(dict(pid=x['identity']['pid'],uuid=real.UUIDS[x['gpu']],memory_MiB='CPU_MOCK'))
  else:
   retired.append(x)
   if not closed.exists():
    try:
     with closed.open('x') as marker:marker.write('CPU_MOCK_RETIRED_IDENTITY\n')
    except FileExistsError:pass
 cfg=json.loads((ROOT/'CASE.json').read_text())
 if cfg['case']=='foreign_pid_reuse' and retired:
  p=ROOT/'state/REUSED_PID.json'
  with (ROOT/'state/reuse_creation.lock').open('a') as creation_lock:
   __import__('fcntl').flock(creation_lock,__import__('fcntl').LOCK_EX)
   if not p.exists():
    original=retired[0]['identity'];replacement=dict(original,starttick=original['starttick']+1,cmdline=['CPU_MOCK_FOREIGN_REUSED_PID'])
    real.atomic_new(p,dict(original=original,replacement=replacement,scope='MOCKED_NVML_PID_REUSE_NOT_A_REAL_GPU_OBSERVATION'))
  x=json.loads(p.read_text());apps.append(dict(pid=x['replacement']['pid'],uuid=real.UUIDS[0],memory_MiB='CPU_MOCK_FOREIGN_REUSE'))
 foreign=cfg.get('foreign_identity')
 if foreign and actual_same(foreign):apps.append(dict(pid=foreign['pid'],uuid=real.UUIDS[0],memory_MiB='CPU_MOCK_FOREIGN_GPU0'))
 return dict(devices=devices,apps=apps,CPU_MOCK_ONLY=True,GPU_calls=0)
def storage(root):
 s=os.statvfs(root);return dict(free_bytes=s.f_bavail*s.f_frsize,CPU_MOCK_SHARED_DATA_FILESYSTEM=True)
def driver(root,gpu,**kwargs):
 real.require(os.environ['CUDA_VISIBLE_DEVICES']==real.UUIDS[gpu],'Actual inherited worker environment binding failed')
 return dict(status='PASS',actual_uuid=real.UUIDS[gpu],visible_device_count=1,CPU_MOCK_ONLY=True,GPU_calls=0)
def library(root,gpu):return dict(status='PASS',CPU_MOCK_ONLY=True,identity=real.device(gpu,os.getpid()),GPU_calls=0)
def bindings(root):
 p=Path(root)/'CPU_MOCK_MODEL.bin';return {str(p):dict(path=str(p),bytes=p.stat().st_size,sha256=real.sha(p))}
original_atomic=real.atomic_new
def fixture_atomic(path,obj):
 value=original_atomic(path,obj)
 if Path(path).name=='RESULT.json' and Path(path).parent.name=='GPU1_W_FIELDS' and json.loads((ROOT/'CASE.json').read_text())['case']=='exit_transition':
  original_atomic(ROOT/'state'/('exit_transition_'+str(os.getpid())+'.json'),dict(CPU_MOCK_ONLY=True,scope='SIMULATE_EMPTY_CMDLINE_BEFORE_WAITPID_WAITABLE'))
  __import__('threading').Event().wait(.15)
 return value
real.atomic_new=fixture_atomic
# CPU fixture I/O mapping only. The exact production authorize function and its
# constant root checks execute unchanged, against an explicit virtual new root.
_original_authorize=real.authorize
class AuthorizationRoot:
 def __init__(self,p):self.actual=Path(p)
 def resolve(self):return self
 def __str__(self):return real.EXECUTION_ROOT
 def __fspath__(self):return str(self.actual)
 def __truediv__(self,name):return self.actual/name
def authorize(root,permit):
 actual=Path(root).resolve();real.require(actual==ROOT.resolve(),'CPU fixture root mismatch')
 original_path=real.Path
 def mapped_path(value):
  p=Path(value)
  return AuthorizationRoot(p) if p.resolve()==ROOT.resolve() else p
 real.Path=mapped_path
 try:return _original_authorize(root,permit)
 finally:real.Path=original_path
real.authorize=authorize
real.process_identity=identity;real.same_process=same;real.query_gpu=query;real.storage_guard=storage;real.driver_identity=driver;real.library_check=library;real.bindings=bindings;real.reference_root=lambda:ROOT/'ram_references'
globals().update({k:v for k,v in vars(real).items() if not k.startswith('__')})
# Explicit fixture hooks; never copied into production source.
fixture_register=register
'''

TORCH_MOCK="""CPU_FIXTURE_ONLY=True
class _Cuda:
 @staticmethod
 def max_memory_allocated():return 0
 @staticmethod
 def max_memory_reserved():return 0
cuda=_Cuda()
"""

RUNTIME_MOCK=r'''
import hashlib,json,os,sys,time
from pathlib import Path
import execution_control as C
import torch
class FakeIds:
 def __init__(self,x):self.x=x
 def __getitem__(self,i):assert i==0;return self
 def tolist(self):return list(self.x)
class SamplerInterface:
 def __init__(self,root):self.root=root
 def render_input(self,tokenizer,problem):
  for r in C.read(self.root/'AIME26_DATASET_FREEZE.json')['records']+[C.read(self.root/'protocol/FREE_CANARY_INPUT.json')]:
   if r['problem']==problem:return 'CPU_MOCK_PROMPT',FakeIds(r['input_ids'])
  raise ValueError('Unknown frozen fixture prompt')
 def ids_sha256(self,ids):
  for r in C.read(self.root/'AIME26_DATASET_FREEZE.json')['records']+[C.read(self.root/'protocol/FREE_CANARY_INPUT.json')]:
   if r['input_ids']==ids:return r.get('input_ids_sha256',hashlib.sha256(json.dumps(ids).encode()).hexdigest())
  raise ValueError('Unknown frozen fixture token IDs')
class Engine:
 def __init__(self,root,gpu,target):
  self.root=root;self.gpu=gpu;self.target=target;self.layers=[i for i in range(32) if i%4!=3];self.torch=torch;self.f=SamplerInterface(root);self.tokenizer=object();self.environment=dict(CPU_MOCK_MODEL_AND_GPU=True,GPU_calls=0);C.fixture_register(gpu);C.progress(target,'CPU_MOCK_MODEL_LOADED')
 def teacher(self,unit,ids,reference=None):
  cfg=C.read(self.root/'CASE.json')
  if cfg['case']=='external_sigterm' and unit['id']=='GPU1_W_PERSIST512_R0':
   C.atomic_new(self.root/'state/READY_FOR_SIGNAL.json',dict(identity=C.process_identity(os.getpid()),CPU_MOCK_ONLY=True))
   while True:time.sleep(.005)
  n=unit['tokens'];audit=dict(qdq_layer_updates=24*n,next_consumption_checks=24*(n-1),CPU_MOCK_ONLY=True)
  if reference:reference.finish(n)
  timing={w:dict(TPS=100.,measured=128,observer=False,sampling_invoked=True) for w in ('short','late')}
  C.progress(self.target,'CPU_MOCK_TEACHER_COMPLETE',token=n)
  return dict(status='PASS',tokens=n,audit=audit,writebacks_by_layer={str(i):n for i in self.layers},timing=timing,CPU_MOCK_ONLY=True,GPU_calls=0)
 def generate(self,condition,input_ids,cap,item,reference=None):
  cfg=C.read(self.root/'CASE.json')
  if cfg['case']=='formal_failure' and item['id']==cfg['fail_formal_id']:raise RuntimeError('EXPECTED_CPU_MOCK_FORMAL_FAILURE')
  p=len(input_ids);n=1;total=p+n-1;tokens=[2];trace=reference is not None;checks=24*(total-1) if trace else 24*sum(t>0 and (t in range(1,128) or t%2048==0) for t in range(total))
  if reference:reference.finish(total)
  C.progress(self.target,'CPU_MOCK_GENERATION_COMPLETE',generated_tokens=n,input_forwards=total)
  return dict(status='COMPLETE',task_id=C.TASK,condition=condition,problem_id=item['problem_id'],seed=1,gpu=self.gpu,gpu_uuid=C.UUIDS[self.gpu],input_ids_sha256=self.f.ids_sha256(input_ids),input_tokens=p,generated_token_ids=tokens,generated_token_ids_sha256=C.objsha(tokens),generated_token_count=n,response='CPU_FIXTURE_ONLY_NOT_AIME_GENERATION',finish_reason='eos',stop_token_ids=[2],nonfinite=False,runtime_error=None,actual_model_forward_calls=total,quantizer_audit=dict(tokens=total,qdq_layer_updates=24*total,next_consumption_checks=checks,CPU_MOCK_ONLY=True),writebacks_by_layer={str(i):total for i in self.layers},prompt_layer_QDQ=24*p,decode_layer_QDQ=24*(n-1),sampler_calls=n,initial_generator_sha256='0'*64,final_generator_sha256='1'*64,wall_seconds=.001,TPS=1000.,sampler_source_sha256=C.sha(self.root/'protocol/sampler.py'),scorer_source_sha256=C.sha(self.root/'protocol/aime26_scorer_v4.py'),dataset_freeze_sha256=C.sha(self.root/'AIME26_DATASET_FREEZE.json'),CPU_MOCK_ONLY=True,GPU_calls=0)
 def pressure(self):return dict(status='PASS',cases=[dict(condition=q,total_context=82293,actual_short_QDQ_tokens=16,free_bytes=2*1024**3,pressure_scope='SYNTHETIC_KV_NATIVE_SDPA_WITH_LIVE_MODEL_NOT81K_GDN_TRAJECTORY',CPU_MOCK_ONLY=True) for q in C.METHODS],CPU_MOCK_ONLY=True,GPU_calls=0)
 def profile(self,unit,ids):return dict(status='PASS_ACTUAL16TOKEN_CUDA_PROFILE',actual_CUDA_kernel_calls=1,quantizer_CUDA_milliseconds=1.,CPU_MOCK_ONLY=True,GPU_calls=0)
class ByteReference:
 def __init__(self,root,target,unit):self.root=root;self.target=target;self.unit=unit
 def finish(self,n):C.atomic_new(self.target/'TRACE.json',dict(status='REFERENCE_COLLECTED' if self.unit['repeat']==0 else 'PASS_FULL_ACTUAL_TENSOR_BYTES',tokens=[dict(token=t,bytes=0,CPU_MOCK_ONLY=True) for t in range(n)],CPU_MOCK_ONLY=True,GPU_calls=0))
 def cleanup_success(self):pass
'''
FIELDS_MOCK=r'''
import execution_control as C

def run(root,target,unit):
 C.fixture_register(unit['gpu']);C.progress(target,'CPU_MOCK_CODEC_FIELDS')
 if C.read(root/'CASE.json')['case']=='qualification_failure' and unit['id']=='GPU2_R_FIELDS':raise RuntimeError('EXPECTED_CPU_MOCK_QUALIFICATION_FAILURE')
 return dict(status='PASS',actual_cuda=True,independent_reference_pass=True,CPU_MOCK_ONLY=True,wire_actual_cuda_flag_is_mocked=True,GPU_calls=0)
'''
SITE_CUSTOMIZE="""import os,time
from pathlib import Path
if os.environ.get('WSCALE_CPU_FIXTURE_ONLY')=='YES':
 _original_sleep=time.sleep
 time.sleep=lambda seconds:_original_sleep(min(seconds,.005))
 _original_read_bytes=Path.read_bytes
 def fixture_read_bytes(path):
  if path.name=='cmdline' and str(path).startswith('/proc/'):
   marker=Path(os.environ['WSCALE_CPU_FIXTURE_ROOT'])/'state'/('exit_transition_'+path.parent.name+'.json')
   if marker.exists():return b''
  return _original_read_bytes(path)
 Path.read_bytes=fixture_read_bytes
"""

def actual_identity(pid):
 p=Path('/proc')/str(pid)
 if p.exists():
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),starttick=int(fields[19]),cmdline=[b.decode() for b in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])
 line=''
 for attempt in range(40):
  line=subprocess.check_output(['ps','-ww','-p',str(pid),'-o','lstart=','-o','command='],text=True).strip()
  if not line:raise ProcessLookupError(pid)
  command=line[24:].strip()
  complete=(not command.startswith('(')) and ('.py' in command or ' -c ' in command)
  if 'gpu_unit_worker.py' in command:complete=complete and ' --unit ' in command
  if 'gpu_controller.py' in command:complete=complete and ' --root ' in command and ' --execute' in command
  if complete:break
  time.sleep(.002)
 command=line[24:].strip();parts=command.split(' ',1)
 # macOS python3 launcher execs its framework binary after Popen returns.
 # Normalize only this CPU fixture's Python executable spelling; argument
 # vector and process start identity remain checked. Linux uses actual/proc.
 if 'python' in parts[0].lower():command='CPU_FIXTURE_PYTHON '+(parts[1] if len(parts)>1 else '')
 return dict(pid=int(pid),starttick=int(hashlib.sha256(line[:24].encode()).hexdigest()[:12],16),cmdline=[command])
def same_actual(x):
 try:return actual_identity(x['pid'])==x
 except (OSError,ValueError,subprocess.SubprocessError):return False

def stage_fixture(parent,case):
 f=parent/case;f.mkdir();(f/'state').mkdir();(f/'src').mkdir();(f/'scripts').mkdir();(f/'production').mkdir();(f/'protocol').mkdir();(f/'analysis').mkdir();(f/'operations').mkdir()
 production=['scripts/gpu_controller.py','scripts/gpu_unit_worker.py','src/control_plan.py','src/control_qualification.py','src/record_validation.py']
 pins=[]
 for name in production:
  p=f/name;p.write_bytes((R/name).read_bytes());pins.append(dict(path=name,sha256=sha(p),exact_production_source=True))
 p=f/'production/execution_control.py';p.write_bytes((R/'src/execution_control.py').read_bytes());pins.append(dict(path='production/execution_control.py',sha256=sha(p),exact_production_source=True))
 inherited=f/'inherited/group32/src';inherited.mkdir(parents=True);(inherited/'inherited_device_guards.py').write_bytes((R/'inherited/group32/src/inherited_device_guards.py').read_bytes())
 for name,text in [('execution_control.py',CONTROL_SHIM),('runtime_engine.py',RUNTIME_MOCK),('gpu_codec_fields.py',FIELDS_MOCK),('torch.py',TORCH_MOCK),('triton_runtime_guard.py',"import os,hashlib\nfrom pathlib import Path\ndef dependency_receipt():\n r=Path(os.environ['WSCALE_CPU_FIXTURE_ROOT']).resolve();return dict(status='PASS_FRESH_TRITON_DEPENDENCIES_ACTUAL_IMPORT',CPU_MOCK_ONLY=True,GPU_calls=0,manifest_sha256=hashlib.sha256((r/'analysis/TRITON_RUNTIME_FRESH_SHA.json').read_bytes()).hexdigest(),task_root=str(r),actual_triton_version='3.1.0',version='3.1.0')\n"),('non_aime_diagnostics.py',"def run(root,condition):return dict(status='PASS_ACTUAL_NONAIME_DIAGNOSTICS_NO_PROXY_THRESHOLD',proxy_MSE=1e99,CPU_MOCK_ONLY=True,GPU_calls=0)\n"),('sitecustomize.py',SITE_CUSTOMIZE)]: (f/'src'/name).write_text(text)
 for name in ['AIME26_DATASET_FREEZE.json','HISTORICAL_G32_BASELINE_REUSE.json','inherited/group32/inherited/historical_port_payload/candidate/benchmark_replay/fixtures/doc1_tokens.json']:
  (f/name).parent.mkdir(parents=True,exist_ok=True);(f/name).write_bytes((R/name).read_bytes())
 for name in ['sampler.py','aime26_scorer_v4.py','FREE_CANARY_INPUT.json']:(f/'protocol'/name).write_bytes((R/'protocol'/name).read_bytes())
 (f/'operations/HUMAN_TASK_AUTHORIZATION.json').write_bytes((R/'operations/HUMAN_TASK_AUTHORIZATION.json').read_bytes())
 (f/'operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json').write_bytes((R/'operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json').read_bytes())
 (f/'operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json').write_bytes((R/'operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json').read_bytes())
 dump(f/'CALIBRATION_REUSE_ADMISSION.json',dict(task_id='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1',status='PASS_REUSED_FROZEN_NONAIME_CALIBRATION_DUAL_HOST_FRESH_SHA_NO_RECOLLECT_NO_RESELECT',origin_source_root='/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1',origin_source_sha256='ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7',selected_a_b=[.5,.25],new_qualification_credit=0,new_formal_credit=0,no_reselection=True,CPU_MOCK_ONLY=True,not_scientific_calibration_evidence=True))
 for name in ['CALIBRATION_DATA_FREEZE.json','TRANSPORTED_QUERY_OMEGA.json','CALIBRATION_PARAMETER_SELECTION.json']:dump(f/name,dict(CPU_FIXTURE_ONLY=True,not_scientific_evidence=True))
 dump(f/'analysis/TRITON_RUNTIME_FRESH_SHA.json',dict(version='3.1.0',CPU_MOCK_ONLY=True,not_actual_compiler_evidence=True))
 (f/'CPU_MOCK_MODEL.bin').write_bytes(b'EXPLICIT_CPU_MOCK_MODEL_NO_WEIGHTS')
 stat=(f/'CPU_MOCK_MODEL.bin').stat();dump(f/'analysis/MODEL_TOKENIZER_FRESH_SHA.json',dict(CPU_FIXTURE_ONLY=True,files=[dict(path=str(f/'CPU_MOCK_MODEL.bin'),bytes=stat.st_size,inode=stat.st_ino,mtime_ns=stat.st_mtime_ns,sha256=sha(f/'CPU_MOCK_MODEL.bin'))]))
 spec=importlib.util.spec_from_file_location('fixture_plan_builder_'+case,R/'src/control_plan.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 uuids=('GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75');assignment,jobs=mod.build_formal_plan(read(f/'AIME26_DATASET_FREEZE.json'),uuids);plan=mod.build_qualification_plan(uuids)
 dump(f/'QUESTION_GPU_ASSIGNMENT.json',assignment);dump(f/'FORMAL_JOB_MANIFEST.json',jobs);dump(f/'GPU_EXECUTION_PLAN.json',plan)
 cfg=dict(case=case,CPU_FIXTURE_ONLY=True,fail_formal_id=jobs['jobs'][0]['id']);foreign=None
 if case=='foreign_pid_reuse':
  foreign=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(120)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);cfg['foreign_identity']=actual_identity(foreign.pid)
 dump(f/'CASE.json',cfg)
 paths=[p for p in sorted(f.rglob('*')) if p.is_file()];dump(f/'SOURCE_FREEZE.json',dict(CPU_FIXTURE_ONLY=True,not_GPU_qualification=True,files=[dict(path=str(p.relative_to(f)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]))
 dump(f/'GPU_PERMIT.json',dict(task_id=mod.TASK,execution_root='/tmp/'+mod.TASK+'_BYTEVIEW_REPAIR_20261025',byteview_repair_authorization_sha256=sha(f/'operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json'),reduction_repair_authorization_sha256=sha(f/'operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json'),calibration_reuse_admission_sha256=sha(f/'CALIBRATION_REUSE_ADMISSION.json'),ALLOW_GPU='YES',CPU_FIXTURE_ONLY=True,not_valid_for_scientific_dispatch=True,devices=[uuids[g] for g in (1,2,3)],physical_indices=[1,2,3],source_freeze_sha256=sha(f/'SOURCE_FREEZE.json'),plan_sha256=sha(f/'GPU_EXECUTION_PLAN.json'),human_grant_sha256=sha(f/'operations/HUMAN_TASK_AUTHORIZATION.json'),GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,watchdog_seconds=1800,cleanup_seconds=120,max_new_tokens=81920,formal_new_samples=90,seed=1,calibration_freeze_sha256=sha(f/'CALIBRATION_DATA_FREEZE.json'),omega_manifest_sha256=sha(f/'TRANSPORTED_QUERY_OMEGA.json'),parameter_selection_sha256=sha(f/'CALIBRATION_PARAMETER_SELECTION.json')))
 return f,pins,foreign

def env_for(f):return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(f/'src'),WSCALE_CPU_FIXTURE_ONLY='YES',WSCALE_CPU_FIXTURE_ROOT=str(f))
def run_controller(f,case):
 log=(f/'controller.log').open('wb');proc=subprocess.Popen([sys.executable,'-u','-B',str(f/'scripts/gpu_controller.py'),'--root',str(f),'--execute'],stdout=log,stderr=subprocess.STDOUT,env=env_for(f));signal_sent=False
 try:
  if case=='external_sigterm':
   deadline=time.monotonic()+40
   while proc.poll() is None and not (f/'state/READY_FOR_SIGNAL.json').exists() and time.monotonic()<deadline:time.sleep(.01)
   require((f/'state/READY_FOR_SIGNAL.json').exists(),'Actual worker did not reach signalfixture point')
   proc.send_signal(signal.SIGTERM);signal_sent=True
  rc=proc.wait(timeout=90)
 finally:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
  log.close()
 return rc,signal_sent

def authorization_negative_checks(f):
 # Exercise the unchanged real authorize body through the same explicit CPU
 # virtual-path I/O fixture before any controller Popen is possible.
 code=r"""
import json,os,sys
from pathlib import Path
import execution_control as C
root=Path(os.environ['WSCALE_CPU_FIXTURE_ROOT']);permit=root/'GPU_PERMIT.json';original=permit.read_bytes();rows=[]
for key,value in [('execution_root','/tmp/'+C.TASK),('execution_root',C.PARENT_EXECUTION_ROOT),('byteview_repair_authorization_sha256','2'*64),('reduction_repair_authorization_sha256','0'*64),('calibration_reuse_admission_sha256','1'*64)]:
 p=json.loads(original);p[key]=value;permit.write_text(json.dumps(p))
 try:C.authorize(root,permit)
 except RuntimeError as e:rows.append(dict(case=key,status='PASS_REJECTED',error=str(e)))
 else:raise AssertionError('Invalid repair admission accepted:'+key)
 finally:permit.write_bytes(original)
reuse=root/'CALIBRATION_REUSE_ADMISSION.json';old=reuse.read_bytes()
for key,value in [('status','BLOCKED'),('origin_source_root','/tmp/FOREIGN_SOURCE'),('origin_source_sha256','f'*64),('new_qualification_credit',1),('no_reselection',False)]:
 r=json.loads(old);r[key]=value;reuse.write_text(json.dumps(r));p=json.loads(original);p['calibration_reuse_admission_sha256']=C.sha(reuse);permit.write_text(json.dumps(p))
 try:C.authorize(root,permit)
 except RuntimeError as e:rows.append(dict(case='reuse_'+key,status='PASS_REJECTED',error=str(e)))
 else:raise AssertionError('Invalid reuse origin/status accepted:'+key)
 finally:reuse.write_bytes(old);permit.write_bytes(original)
grant=root/'operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json';saved_grant=grant.read_bytes()
for key,value in [('new_remote_root',C.PARENT_EXECUTION_ROOT),('prior_true_STOP_source_sha256','f'*64),('old_qualification_credit',1),('no_infinite_GPU_retry',False)]:
 g=json.loads(saved_grant);g[key]=value;grant.write_text(json.dumps(g));p=json.loads(original);p['byteview_repair_authorization_sha256']=C.sha(grant);permit.write_text(json.dumps(p))
 try:C.authorize(root,permit)
 except RuntimeError as e:rows.append(dict(case='byteview_'+key,status='PASS_REJECTED',error=str(e)))
 else:raise AssertionError('Invalid byteview scope accepted:'+key)
 finally:grant.write_bytes(saved_grant);permit.write_bytes(original)
C.authorize(root,permit)
print(json.dumps(dict(status='PASS_REAL_AUTHORIZE_REPAIR_NEGATIVES_ONLY_CPU_FIXTURE',checks=rows,GPU_calls=0,model_loads=0,production_dispatches=0)))
"""
 result=subprocess.run([sys.executable,'-B','-c',code],env=env_for(f),capture_output=True,text=True,timeout=20)
 require(result.returncode==0,'Repair authorization negative CPU checks failed:'+result.stderr)
 report=json.loads(result.stdout);dump(f/'AUTHORIZATION_NEGATIVE_CPU.json',report)
 return report

def assert_no_owned_children(f):
 live=[]
 for p in (f/'state').glob('worker_*.json'):
  x=read(p)['identity']
  if same_actual(x):live.append(x)
 require(not live,'Actual CPU fixture child orphaned:'+str(live))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence-root',type=Path);ap.add_argument('--cases',nargs='+',choices=['success','exit_transition','qualification_failure','formal_failure','external_sigterm','foreign_pid_reuse'],default=['success','exit_transition','qualification_failure','formal_failure','external_sigterm','foreign_pid_reuse']);args=ap.parse_args()
 production_hashes={n:sha(R/n) for n in ['scripts/gpu_controller.py','scripts/gpu_unit_worker.py','src/control_qualification.py','src/control_plan.py','src/execution_control.py','src/record_validation.py']}
 phase=R/'CALIBRATION_SOURCE_FREEZE.json';before=sha(phase);require(before==PHASE_A_SHA,'PhaseA sourcefreeze hash differs')
 phasefiles={x['path']:x['sha256'] for x in read(phase)['files']};origin=R/'provenance/PRIOR_TRUE_STOP/source'
 require(all(sha(origin/k)==v for k,v in phasefiles.items()),'Original frozenphaseA provenance changed beforetests')
 base=args.evidence_root or Path(tempfile.mkdtemp(prefix='wscale_byteview_repair_control_cpu_'));base=base.resolve();base.mkdir(parents=True,exist_ok=True);results=[]
 for case in args.cases:
  f,pins,foreign=stage_fixture(base,case)
  try:
   authorization_negatives=authorization_negative_checks(f) if case=='success' else None
   rc,sent=run_controller(f,case);terminal=read(f/'SESSION_RESULT.json');ledger=read(f/'RESOURCE_LEDGER.json');formal=list((f/'formal').glob('*/DISPATCH.json')) if (f/'formal').exists() else [];gate=(f/'AUTHORIZATION_FORMAL.json').exists();successful=case in ('success','exit_transition','foreign_pid_reuse')
   if successful:
    require(rc==0 and terminal['status']=='GPU_COMPLETE_52_QUALIFICATION_90_FORMAL' and len(formal)==90,'Success path failed actual strictbarrier:'+str((rc,terminal)))
    q=read(f/'GPU_CODEC_QUALIFICATION.json');require(q['inspection']['count']==52 and len(ledger['intervals'])==142,'Actual52strictinspect/90formal count differs');require(read(f/'FORMAL_PROGRESS.json')['completed']==90,'Actual legalprogress count differs')
   else:
    require(rc!=0 and terminal['status']=='FAIL_CLOSED_STOP' and (f/'FIRST_FAILURE.json').exists(),'Expected failure didnotfailclosed')
    if case in ('qualification_failure','external_sigterm'):require(not gate and not formal,'Qualification failure orsignal wrongly admittedformal')
    if case=='formal_failure':require(gate and len(formal)<90,'Formal failure path didnotpreserve gate/stopqueue')
    if case=='external_sigterm':require(sent and read(f/'FIRST_FAILURE.json')['unit']=='CONTROLLER_SIGNAL' and 'SIGTERM' in read(f/'FIRST_FAILURE.json')['error'],'ExternalSIGTERM didnotentercontrolledSTOP')
   require(all(x['release']=='PASS' for x in ledger['intervals']),'Actual own CPU release observation missing');assert_no_owned_children(f)
   if case=='exit_transition':require('BOUNDED_OWN_POPEN_REAP_AFTER_IDENTITY_TRANSITION' in (f/'GPU_RESOURCE_LEDGER.jsonl').read_text(),'Actual transientexit boundedreap branch not exercised')
   if case=='foreign_pid_reuse':require((f/'state/REUSED_PID.json').exists() and foreign.poll() is None and same_actual(read(f/'CASE.json')['foreign_identity']),'ForeignPIDreuse fixture missing or actualforeignprocess signalled')
   row=dict(case=case,status='PASS',returncode=rc,formal_dispatches=len(formal),formal_barrier_created=gate,actual_Popen_leaves=len(ledger['intervals']),production_source=pins,first_failure=read(f/'FIRST_FAILURE.json') if (f/'FIRST_FAILURE.json').exists() else None,CPU_MOCK_ONLY=True,GPU_qualification_credit=0,AIME_samples=0,remote_writes=0,fixture_root=str(f),receipt_sha256={name:sha(f/name) for name in ['SESSION_RESULT.json','RESOURCE_LEDGER.json','FORMAL_PROGRESS.json']})
   if case=='success':
    row['repair_authorization_negatives']=authorization_negatives
    count=len(list(f.glob('execution/*/DISPATCH.json')))+len(formal);rc2,_=run_controller(f,'restart');after=len(list(f.glob('execution/*/DISPATCH.json')))+len(list(f.glob('formal/*/DISPATCH.json')));require(rc2!=0 and after==count and 'Already dispatched; no relaunch or resets' in (f/'controller.log').read_text(),'Actualrestart allowedredispatch')
    row['restart']=dict(status='PASS_REJECTED_BEFORE_NEW_POPEN',returncode=rc2,dispatch_count_unchanged=count)
   results.append(row);print(json.dumps(dict(case=case,status='PASS',CPU_MOCK_ONLY=True,GPU_credit=0)),flush=True)
  finally:
   if foreign is not None:
    # This is only the test-owned CPU sentinel, never a GPU/foreignsciencePID.
    if foreign.poll() is None:foreign.terminate();foreign.wait(timeout=5)
 require(sha(phase)==before and all(sha(origin/k)==v for k,v in phasefiles.items()),'FrozenphaseA source changed duringtests')
 require(all(sha(R/k)==v for k,v in production_hashes.items()),'Production source changed duringfreshsuite')
 report=dict(production_sha256=production_hashes,status='PASS_ACTUAL_PRODUCTION_CONTROLLER_WORKER_STRICT_INSPECT_FORMAL_BARRIER_CPU_FIXTURES',cases=results,case_count=len(results),restart_negative=True,phaseA_freeze_sha256=before,phaseA_31_original_frozen_sources_unchanged=True,phaseA_original_source_root=str(origin),new_execution_control_is_independent_repair=True,explicit_mocks=['model/runtime','CUDA/device/native observations','fixtureidentity helper for portability/PIDreuse','poll sleep acceleration'],actual_components=['controller source','worker source','execution_control source/auth/source verification','finite plan source','payloadvalidator source','strict inspector source','real Popen/freshPython children','real signals andownedcleanup','formal admission barrier'],GPU_calls=0,GPU_qualification_credit=0,AIME_samples=0,science_GPU_runs_started=0,fixture_root=str(base),not_scientific_execution_evidence=True)
 dump(base/'CPU_CONTROL_INTEGRATION_REPORT.json',report);print(json.dumps(dict(status=report['status'],report=str(base/'CPU_CONTROL_INTEGRATION_REPORT.json'),case_count=len(results),GPU_calls=0)),flush=True)
if __name__=='__main__':main()
