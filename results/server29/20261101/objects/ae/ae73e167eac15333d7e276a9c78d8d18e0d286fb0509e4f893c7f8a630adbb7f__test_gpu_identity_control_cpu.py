#!/usr/bin/env python3
"""Synthetic CPU-only ownership tests. Never calls GPU/NVML or Torch."""
import ast,copy,ctypes,hashlib,importlib.util,json,os,sys,tempfile,types,uuid
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
REV=Path(__file__).resolve().parents[1];ROOT=REV.parent
spec=importlib.util.spec_from_file_location('revision_controller_cpu',REV/'scripts/restoration_controller.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
checks=[]
def ok(name,value=True):assert value,name;checks.append(dict(name=name,status='PASS'))
def reject(name,fn):
 try:fn()
 except RuntimeError as exc:ok(name);return exc
 raise AssertionError(name+' unexpectedly accepted')
def observation(apps):return dict(devices=[dict(physical_index=0,uuid=C.UUID,name='RTX 3090',memory_total_MiB=24576,memory_free_MiB=24256),dict(physical_index=1,uuid='GPU-other',name='RTX 3090',memory_total_MiB=24576,memory_free_MiB=24256)],apps=apps)
PID=700
correct=observation([dict(pid=PID,uuid=C.UUID)]);wrong=observation([dict(pid=PID,uuid='GPU-other')]);missing=observation([]);multiple=observation([dict(pid=PID,uuid=C.UUID),dict(pid=PID,uuid='GPU-other')]);foreign_selected=observation([dict(pid=PID,uuid=C.UUID),dict(pid=999,uuid=C.UUID)]);foreign_other=observation([dict(pid=PID,uuid=C.UUID),dict(pid=999,uuid='GPU-other')])
with mock.patch.object(C,'query_gpu',side_effect=AssertionError('No actual query in CPU tests')):
 good=C.require_own_gpu_identity(PID,correct);ok('own_exact_actual_UUID_PASS_and_full_observation_retained',good['status']=='PASS'and good['observation']==correct and good['actual_UUIDs']==[C.UUID]);ok('foreign_process_on_other_card_allowed',C.require_own_gpu_identity(PID,foreign_other)['status']=='PASS')
 for name,obs in [('wrong_UUID',wrong),('own_missing',missing),('own_cross_card',multiple),('foreign_selected',foreign_selected)]:
  exc=reject(name+'_worker_admission_rejects',lambda obs=obs:C.require_own_gpu_identity(PID,obs));receipt=exc.gpu_identity_receipt;ok(name+'_full_failed_identity_receipt_retained',receipt['status']=='FAIL_CLOSED_STOP'and receipt['observation']==obs)
 for name,obs in [('wrong_UUID',wrong),('own_cross_card',multiple)]:reject('controller_monitor_'+name+'_rejects',lambda obs=obs:C.validate_gpu(obs,own_pid=PID))
 ok('controller_monitor_before_GPU_allocation_can_have_no_NVML_own_row',C.validate_gpu(missing,own_pid=PID)['uuid']==C.UUID);ok('controller_monitor_foreign_other_card_allowed',C.validate_gpu(foreign_other,own_pid=PID)['uuid']==C.UUID)
 wrongindex=copy.deepcopy(correct);wrongindex['devices'][0]['uuid']='GPU-wrong-physical';reject('physical_index0_UUID_mismatch_rejects',lambda:C.require_own_gpu_identity(PID,wrongindex))
# Driver ordinal qualification is exercised exclusively with injected Python
# callables. The real ctypes.CDLL is prohibited even in every negative case.
with tempfile.TemporaryDirectory(prefix='DRIVER_ORDINAL_CPU_FAKE_')as temporary:
 dp=Path(temporary).resolve()/'libcuda.so.fixture';dp.write_bytes(b'CPU fake driver identity');row=dict(path=str(dp),resolved_path=str(dp),bytes=dp.stat().st_size,sha256=C.sha(dp));events=[]
 class Fn:
  def __init__(self,fn):self.fn=fn
  def __call__(self,*args):return self.fn(*args)
 class FakeDriver:
  def __init__(self,actual=C.UUID,count=1,v2=True,fail=None):
   self.actual=actual;self.count=count;self.fail=fail;self.cuInit=Fn(self.init);self.cuDeviceGetCount=Fn(self.device_count);self.cuDeviceGet=Fn(self.device)
   setattr(self,'cuDeviceGetUuid_v2'if v2 else'cuDeviceGetUuid',Fn(self.device_uuid))
  def rc(self,name):events.append(name);return 7 if self.fail==name else 0
  def init(self,flag):assert flag==0;return self.rc('cuInit')
  def device_count(self,pointer):pointer._obj.value=self.count;return self.rc('cuDeviceGetCount')
  def device(self,pointer,ordinal):assert ordinal==0;pointer._obj.value=17;return self.rc('cuDeviceGet')
  def device_uuid(self,pointer,device):
   assert device==17
   for i,value in enumerate(uuid.UUID(self.actual.removeprefix('GPU-')).bytes):pointer._obj.bytes[i]=value
   return self.rc('cuDeviceGetUuid')
 with mock.patch.object(ctypes,'CDLL',side_effect=AssertionError('Real driver load/cuInit forbidden during CPU tests')):
  for v2 in [True,False]:
   events.clear();receipt=C.require_driver_ordinal_identity([row],driver_factory=lambda path:FakeDriver(v2=v2),environment={'CUDA_VISIBLE_DEVICES':'0'});ok('driver_fake_UUID_'+('v2'if v2 else'fallback')+'_PASS',receipt['status']=='PASS'and receipt['actual_uuid']==C.UUID and receipt['visible_device_count']==1);ok('driver_fake_only_four_no_context_APIs_'+str(v2),events==['cuInit','cuDeviceGetCount','cuDeviceGet','cuDeviceGetUuid']and receipt['context_creation_API_calls']==receipt['kernel_launch_API_calls']==0)
  for name,driver,env,rows in [('wrong_UUID',FakeDriver(actual='GPU-11111111-1111-1111-1111-111111111111'),{'CUDA_VISIBLE_DEVICES':'0'},[row]),('visible_count4',FakeDriver(count=4),{'CUDA_VISIBLE_DEVICES':'0'},[row]),('visible_count0',FakeDriver(count=0),{'CUDA_VISIBLE_DEVICES':'0'},[row]),('wrong_CVD',FakeDriver(),{'CUDA_VISIBLE_DEVICES':'1'},[row]),('driver_missing',FakeDriver(),{'CUDA_VISIBLE_DEVICES':'0'},[]),('driver_SHA_changed',FakeDriver(),{'CUDA_VISIBLE_DEVICES':'0'},[dict(row,sha256='0'*64)]),('cuInit_error',FakeDriver(fail='cuInit'),{'CUDA_VISIBLE_DEVICES':'0'},[row])]:
   events.clear();exc=reject('driver_'+name+'_rejects',lambda driver=driver,env=env,rows=rows:C.require_driver_ordinal_identity(rows,driver_factory=lambda path:driver,environment=env));ok('driver_'+name+'_failure_receipt_retained',exc.driver_identity_receipt['status']=='FAIL_CLOSED_STOP'and exc.driver_identity_receipt['context_creation_API_calls']==0)
   if name in ['wrong_CVD','driver_missing','driver_SHA_changed']:ok('driver_'+name+'_before_any_driver_API',not events)
   if name=='visible_count4':ok('unqualified_driver_CVD_semantics_do_not_proceed_to_ordinal',events==['cuInit','cuDeviceGetCount'])
# AST-extract the actual revised worker library gate. Standard-library fixture
# files stand in for a mapped native library, with exact real inode/SHA bytes.
wtree=ast.parse((REV/'worker/restoration_worker.py').read_text());main=next(x for x in wtree.body if isinstance(x,ast.FunctionDef)and x.name=='main');gate=next(x for x in main.body if isinstance(x,ast.FunctionDef)and x.name=='library_check')
with tempfile.TemporaryDirectory(prefix='GPU_IDENTITY_CPU_MOCK_')as temporary:
 tmp=Path(temporary);lib=tmp/'libtorch_fixture.so';lib.write_bytes(b'CPU fixture native byte identity');maps=tmp/'maps.txt';maps.write_text('00000000-00000010 r-xp 00000000 00:00 '+str(lib.stat().st_ino)+' '+str(lib)+'\n');records={}
 def path_proxy(value):return maps if str(value)=='/proc/self/maps'else Path(value)
 def save(path,value):records[Path(path).name]=copy.deepcopy(value)
 env=dict(Path=path_proxy,controller=C,os=types.SimpleNamespace(getpid=lambda:PID),target=tmp,allowed={str(lib.resolve()):dict(bytes=lib.stat().st_size,sha256=C.sha(lib))},digest=C.sha,environment_snapshot=lambda:dict(CPU_FAKE=True),save=save,check=C.require,a=types.SimpleNamespace(unit='NUMERIC_FRESH0'),r=tmp)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[gate],type_ignores=[])),'revised_worker_library_check_CPU_only','exec'),env)
 for name,obs,success in [('correct',correct,True),('wrong',wrong,False),('missing',missing,False),('cross_card',multiple,False),('foreign_selected',foreign_selected,False),('foreign_other',foreign_other,True)]:
  records.clear()
  with mock.patch.object(C,'query_gpu',return_value=obs):
   if success:env['library_check']('PREFLIGHT')
   else:reject('actual_worker_library_PRE_'+name+'_raises',lambda:env['library_check']('PREFLIGHT'))
  own=records['PREFLIGHT_OWN_GPU_IDENTITY.json'];gate_receipt=records['PREFLIGHT_LIBRARY_IDENTITY.json'];ok('actual_worker_'+name+'_persists_complete_own_receipt',own['observation']==obs and gate_receipt['own_gpu_identity']==own);ok('actual_worker_'+name+'_combined_gate_status',gate_receipt['status']==('PASS'if success else'FAIL_CLOSED_STOP'))
 records.clear()
 with mock.patch.object(C,'query_gpu',return_value=correct):env['library_check']('POSTFLIGHT')
 ok('worker_POST_identity_gate_records_actual_process',records['POSTFLIGHT_OWN_GPU_IDENTITY.json']['own_pid']==PID)
 # Exact-PID cleanup must still wait when its compute record is on another UUID.
 def cleanup_case(name,eventually_release):
  rr=tmp/name;rr.mkdir();execution=rr/'execution';calls=[0];clockvalue=[0.];children={};signals=[];ctrl=dict(pid=os.getpid(),starttick=101,cmdline=['CPU fake controller']);checked=dict(root=rr,plan=dict(C.LIMITS,min_gpu_free_bytes=22*1024**3,minimum_disk_free_bytes=10*1024**3,max_numeric_raw_bytes=8*1024**3,device=dict(physical_index=0,uuid=C.UUID)),plan_path=rr/'qualification/GPU_PLAN.json',authorization_sha256='a'*64,plan_sha256='b'*64,source_freeze_sha256='c'*64,request_sha256='d'*64)
  def clock():clockvalue[0]+=.1;return clockvalue[0]
  def sleep(s):clockvalue[0]+=s
  class Child:
   def __init__(self,command,**kw):self.pid=PID;self.returncode=None;self.identity=dict(pid=PID,starttick=707,cmdline=command);self.exited=False;children[PID]=self
   def poll(self):return 0 if self.exited else None
   def terminate(self):signals.append('own_terminate');self.exited=True;self.returncode=0
   def kill(self):signals.append('own_kill');self.exited=True;self.returncode=-9
  def reader(pid):
   if pid==os.getpid():return ctrl
   return children[pid].identity
  def fake_gpu():
   calls[0]+=1
   if calls[0]<3:return missing
   if eventually_release and calls[0]>=5:return missing
   return wrong
  with mock.patch.object(C,'authorize',return_value=checked),mock.patch.object(C,'disk_guard',return_value=dict(available_bytes=20*1024**3,numeric_raw_bytes=0)):
   reject(name+'_actual_controller_foreign_own_UUID_FAIL_CLOSED',lambda:C.run_controller(rr,rr/'auth','YES',gpu_reader=fake_gpu,popen=Child,reader=reader,clock=clock,sleeper=sleep))
  fail=C.read(rr/'FIRST_FAILURE.json');events=[json.loads(x)for x in(execution/'RESOURCE_EVENTS.jsonl').read_text().splitlines()];ok(name+'_exact_own_process_only_signalled',signals==['own_terminate']);ok(name+'_release_verified_only_after_all_UUID_rows_disappear',fail['cleanup']['NVML_release_verified']is eventually_release);ok(name+'_other_UUID_row_not_false_release',calls[0]>=5 if eventually_release else fail['cleanup']['status']=='UNRESOLVED_CLEANUP_RELEASE');ok(name+'_exit_observation_once',sum(x['event']=='FIRST_PROCESS_EXIT_OBSERVED'for x in events)==1);ok(name+'_release_event_count_correct',sum(x['event']=='FIRST_NVML_RELEASE_OBSERVED'for x in events)==(1 if eventually_release else 0))
 cleanup_case('wrong_UUID_lingers_then_releases',True);cleanup_case('wrong_UUID_lingers_past_cleanup_deadline',False)
# All worker AST outside the I/O admission library_check function is identical.
def without_library_gate(tree):
 tree=copy.deepcopy(tree);m=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='main');m.body=[x for x in m.body if not(isinstance(x,ast.FunctionDef)and x.name in ['library_check','driver_ordinal_check'])and not(isinstance(x,ast.Import)and any(a.name=='torch'for a in x.names))]
 for x in m.body:
  if isinstance(x,ast.Try):x.body=[b for b in x.body if not(isinstance(b,ast.Expr)and isinstance(b.value,ast.Call)and isinstance(b.value.func,ast.Name)and b.value.func.id=='driver_ordinal_check')and not(isinstance(b,ast.Import)and any(a.name=='torch'for a in b.names))]
 return ast.dump(tree,include_attributes=False)
old_worker=ast.parse((ROOT/'worker/restoration_worker.py').read_text());ok('complete_worker_AST_identical_except_two_identity_admission_gates',without_library_gate(old_worker)==without_library_gate(wtree))
# Numeric PREFLIGHT is the same existing call preceding all token forwards;
# benchmark checked loaders retain the original before-m.main timing ordering.
worker_text=(REV/'worker/restoration_worker.py').read_text();ok('driver_ordinal_check_called_before_Torch_import_model_seed_and_benchmark',worker_text.index('        driver_ordinal_check()')<worker_text.index('        import torch')<worker_text.index("        if a.mode=='benchmark':")<worker_text.index('f.setup_seed(20261015)'));ok('numeric_PRE_is_before_first_scientific_model_forward',worker_text.index("assert_old_environment();library_check('PREFLIGHT');N=")<worker_text.index('position[\'t\']=t;out=model('));ok('benchmark_checked_loader_has_PRE_before_return_to_legacy_main',"out=original();assert_old_environment();library_check('PREFLIGHT');return out"in worker_text)
for path,want in [('scripts/restoration_controller.py','3c4db67f17a337124d6498f9e24804dd22b12a0df1f1016922b4152fdbf5855e'),('worker/restoration_worker.py','e3f63c2535bf00e563f391f570df63c8da5a083720f2c4c5abe74af8aa36730c')]:ok('original_frozen_'+path+'_unchanged',hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==want)
ok('no_Torch_or_GPU_runtime_import',not any(x=='torch'or x.startswith('torch.')or x=='transformers'or x.startswith('transformers.')for x in sys.modules))
print(json.dumps(dict(status='PASS_CPU_CONTROL_REVISION_01_SYNTHETIC_TESTS_NOT_GPU_QUALIFICATION',checks_passed=len(checks),checks=checks,GPU_queries=0,GPU_dispatches=0,all_GPU_observations='INJECTED_SYNTHETIC_ONLY',original_sources_unchanged=True,controller_sha256=C.sha(REV/'scripts/restoration_controller.py'),worker_sha256=C.sha(REV/'worker/restoration_worker.py')),indent=2,sort_keys=True))
