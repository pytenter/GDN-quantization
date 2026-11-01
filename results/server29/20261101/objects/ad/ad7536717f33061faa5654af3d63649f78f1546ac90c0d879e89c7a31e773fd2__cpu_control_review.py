"""Actual production admission/UUID/roster rejection tests; GPU APIs mocked explicitly."""
import sys,os,json,tempfile,shutil,copy,uuid,ast
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));import execution_control as C
checks=[]
def ok(name):checks.append(dict(name=name,status='PASS'))
def reject(name,fn):
 try:fn()
 except (RuntimeError,ValueError,KeyError,FileNotFoundError):ok(name);return
 raise AssertionError('Unsafe case accepted:'+name)
def put(p,x):p.write_text(json.dumps(x))
with tempfile.TemporaryDirectory(prefix='g64_admission_cpu_')as d:
 f=Path(d);(f/'src').mkdir();shutil.copyfile(R/'src/inherited_device_guards.py',f/'src/inherited_device_guards.py');put(f/'NODE.json',C.NODE);put(f/'CROSS_HOST_PLAN.json',C.read(R/'CROSS_HOST_PLAN.json'));put(f/'HUMAN_GPU_AUTHORIZATION.json',C.read(R/'HUMAN_GPU_AUTHORIZATION.json'));put(f/'GPU_EXECUTION_PLAN.json',C.read(R/'GPU_EXECUTION_PLAN.json'));put(f/'SOURCE_FREEZE.json',dict(files=[dict(path='src/inherited_device_guards.py',bytes=(f/'src/inherited_device_guards.py').stat().st_size,sha256=C.sha(f/'src/inherited_device_guards.py'))]))
 permit=dict(node_id=C.NODE['id'],host=C.NODE['host'],global_plan_sha256=C.sha(f/'CROSS_HOST_PLAN.json'),task_id=C.TASK,ALLOW_GPU='YES',devices=list(C.UUIDS.values()),source_freeze_sha256=C.sha(f/'SOURCE_FREEZE.json'),plan_sha256=C.sha(f/'GPU_EXECUTION_PLAN.json'),human_grant_sha256=C.sha(f/'HUMAN_GPU_AUTHORIZATION.json'),GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,watchdog_seconds=1800,cleanup_seconds=120);put(f/'GPU_PERMIT.json',permit);oldR=C.R;C.R=f
 C.authorize(f,f/'GPU_PERMIT.json');ok('Actual production permit positive with fake CPU project')
 reject('Wrong root cannot dispatch',lambda:C.authorize(R,f/'GPU_PERMIT.json'))
 for k,val in [('task_id','OTHER'),('ALLOW_GPU','NO'),('node_id','FOREIGN_NODE'),('host','OTHER_HOST'),('global_plan_sha256','bad'),('devices',['GPU-UNAUTHORIZED']),('source_freeze_sha256','bad'),('plan_sha256','bad'),('human_grant_sha256','bad'),('GPUh_hard_cap',.5),('unit_elapsed_cap',60),('watchdog_seconds',60)]:
  q=dict(permit);q[k]=val;put(f/'GPU_PERMIT.json',q);reject('Reject altered '+k,lambda:C.authorize(f,f/'GPU_PERMIT.json'))
 put(f/'GPU_PERMIT.json',permit);(f/'src/inherited_device_guards.py').write_text('tampered');reject('Reject changed frozen source',lambda:C.authorize(f,f/'GPU_PERMIT.json'));C.R=oldR
obs=dict(devices=[dict(physical_index=g,uuid=u,name='RTX 3090',memory_total_MiB=24576,memory_free_MiB=24110)for g,u in C.UUIDS.items()],apps=[dict(pid=123,uuid='FOREIGN_GPU0',memory_MiB=20000)])
original_query=C.query_gpu;C.query_gpu=lambda:copy.deepcopy(obs)
for g in C.UUIDS:C.device(g);ok('Only selected physical UUID idle admission '+str(g))
for g in [i for i in range(8)if i not in C.UUIDS]+[-1,True]:reject('Unauthorized physical index '+str(g),lambda:C.device(g))
for g in C.UUIDS:
 obs['apps'].append(dict(pid=777,uuid=C.UUIDS[g]));reject('Foreign assignedGPU refused '+str(g),lambda:C.device(g));obs['apps'].pop()
 obs['apps'].append(dict(pid=777,uuid=C.UUIDS[g]));C.device(g,777);ok('Actual owner nonempty ownUUID '+str(g));obs['apps'].append(dict(pid=777,uuid='FOREIGN_GPU0'));reject('Own PID on unauthorizedUUID refused '+str(g),lambda:C.device(g,777));obs['apps'].pop();obs['apps'].pop()
C.query_gpu=original_query
jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];C.validate_jobs(R,jobs);ok('Node60 unique15 rotating samequestionUUID roster')
for name,mutate in [('missing sample',lambda j:j.pop()),('duplicated sample',lambda j:j.append(j[0])),('wrong condition',lambda j:j[0].update(condition='Q0')),('wrong GPU',lambda j:j[0].update(gpu=4)),('wrong UUID',lambda j:j[0].update(gpu_uuid='FOREIGN')),('wrong seed',lambda j:j[0].update(seed=2)),('wrong inputIDs',lambda j:j[0].update(input_ids_sha256='bad'))]:
 j=copy.deepcopy(jobs);mutate(j);reject('Manifest '+name,lambda:C.validate_jobs(R,j))
# The same real driver identity adapter is exercised with mocked CUDA calls only.
class Fn:
 def __init__(self,fn):self.fn=fn
 def __call__(self,*args):return self.fn(*args)
class Driver:
 def __init__(self,expected,count=1):
  self.cuInit=Fn(lambda *_:0)
  self.cuDeviceGetCount=Fn(lambda p:(setattr(p._obj,'value',count)or 0))
  self.cuDeviceGet=Fn(lambda p,i:(setattr(p._obj,'value',0)or 0))
  def get(p,i):
   for n,b in enumerate(uuid.UUID(expected[4:]).bytes):p._obj.bytes[n]=b
   return 0
  self.cuDeviceGetUuid_v2=Fn(get)
for g in C.UUIDS:
 r=C.driver_identity(R,g,driver_factory=lambda p:Driver(C.UUIDS[g]),environment={'CUDA_VISIBLE_DEVICES':C.UUIDS[g]});assert r['status']=='PASS';ok('Production CVD UUID driver adapter mocked '+str(g))
 reject('Wrong driver actualUUID '+str(g),lambda:C.driver_identity(R,g,driver_factory=lambda p:Driver('GPU-11111111-2222-3333-4444-555555555555'),environment={'CUDA_VISIBLE_DEVICES':C.UUIDS[g]}))
 reject('Wrong driver visible count '+str(g),lambda:C.driver_identity(R,g,driver_factory=lambda p:Driver(C.UUIDS[g],2),environment={'CUDA_VISIBLE_DEVICES':C.UUIDS[g]}))
 reject('Numeric CVD forbidden '+str(g),lambda:C.driver_identity(R,g,driver_factory=lambda p:Driver(C.UUIDS[g]),environment={'CUDA_VISIBLE_DEVICES':str(g)}))
# Production score contrast extracted without importing executable finisher.
s=R/'scripts/finalize_global_cpu.py';fn=next(n for n in ast.parse(s.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='pair');ns={'C':C,'math':__import__('math')};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(s),'exec'),ns)
a={str(i):i<20 for i in range(30)};b={str(i):i<18 or i in(21,22)for i in range(30)};p=ns['pair'](a,b);assert p['rescued']==2 and p['lost']==2 and p['net_correct']==0 and p['exact_McNemar_two_sided_p']==1;ok('Paired McNemar real production contrast two-way flips')
assert 'torch'not in sys.modules
print(json.dumps(dict(status='PASS_GROUP64_PRODUCTION_CONTROL_CPU',count=len(checks),checks=checks,CUDA_native_calls='MOCKED_CPU_ONLY',actual_GPU_dispatches=0)))
