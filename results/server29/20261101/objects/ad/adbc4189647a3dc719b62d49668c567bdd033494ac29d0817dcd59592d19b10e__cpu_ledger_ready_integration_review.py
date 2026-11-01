"""Real controller -> real ready receipt -> real qualification inspector/barrier.

GPU, model, native and child processes are explicit CPU mocks. Reused original
qualification files and inspector field checks are real, read only, never credits.
The closed v5 source must reproduce the missing-ledger stop; v6 must pass.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import time
import types

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
import execution_control as real
import continuation_admission as actual_reuse

layout=real.read(R/'CROSS_HOST_PLAN.json')
node=next(n for n in layout['nodes']if n['id']=='server55_gpu0')
plan=dict(qualification_units=[u for u in layout['qualification_units']if u['node_id']==node['id']])
jobs=[j for j in real.read(R/'GLOBAL_FORMAL_JOB_MANIFEST.json')['jobs']if j['node_id']==node['id']]
qids=[u['id']for u in plan['qualification_units']]
origin=actual_reuse.origin_root(R,node['id'])
assert len(qids)==19 and len(jobs)==40
original_popen=subprocess.Popen
original_statvfs=os.statvfs
checks=[]
with tempfile.TemporaryDirectory(prefix='G64_LEDGER_READY_REAL_BOUNDARY_CPU_')as tmp:
 for old in [True,False]:
  F=Path(tmp)/('closed_v5_reproduces'if old else'new_v6_passes');F.mkdir()
  for name in ['CROSS_HOST_PLAN.json','GLOBAL_FORMAL_JOB_MANIFEST.json','QUESTION_GPU_ASSIGNMENT.json','AIME26_DATASET_FREEZE.json','HUMAN_GPU_AUTHORIZATION.json']:
   (F/name).write_bytes((R/name).read_bytes())
  def put(n,v):real.atomic_new(F/n,v)
  put('GPU_EXECUTION_PLAN.json',plan);put('FORMAL_JOB_MANIFEST.json',dict(jobs=jobs))
  put('SOURCE_FREEZE.json',dict(CPU_FIXTURE=True));put('GPU_PERMIT.json',dict(CPU_FIXTURE=True))
  put('analysis/MODEL_TOKENIZER_FRESH_SHA.json',real.read(R/'analysis/MODEL_TOKENIZER_FRESH_SHA.json'))
  fake=types.ModuleType('execution_control')
  for name in dir(real):setattr(fake,name,getattr(real,name))
  fake.NODE=node;fake.UUIDS={0:node['uuid']}
  fake.authorize=lambda *args:dict(EXPLICIT_CPU_PERMIT_MOCK=True)
  fake.verify_sources=lambda *args:real.sha(F/'SOURCE_FREEZE.json')
  fake.model_guard=lambda *args:None
  fake.verify_external=lambda *args:0
  fake.storage_guard=lambda *args:None
  fake.query_gpu=lambda:dict(apps=[],EXPLICIT_CPU_NVML_MOCK=True)
  fake.same_process=lambda i:False
  fake.reference_root=lambda:F/'reference'
  fake.device=lambda g,*args,**kwargs:real.require(g==0,'CPU fixture foreign device')or dict(EXPLICIT_CPU_DEVICE_MOCK=True)
  fake.worker_environment=lambda g:dict(os.environ,CUDA_VISIBLE_DEVICES=node['uuid'],EXPLICIT_CPU_MODEL_MOCK='1')
  fake.validate_jobs=lambda root,rows:real.require(rows==jobs,'CPU fixture jobs changed')
  children={};calls=[]
  fake.process_identity=lambda pid:children.get(pid,dict(pid=pid,starttick=1,cmdline=['CPU_MOCK_CONTROLLER']))
  a=types.ModuleType('continuation_admission')
  a.admit_local_reuse=lambda root:dict(qualification_ids=qids,formal_ids=[])
  a.qualification_source=lambda root,unit:origin if unit in qids else F
  a.host_reused_formal=lambda root:{}
  v=types.ModuleType('record_validation')
  v.validate=lambda w,j,root:real.require(w['unit']==j['id']and w['gpu']==0 and w['gpu_uuid']==node['uuid'],'CPU child job identity')
  inp=types.ModuleType('qualification_input_admission');inp.validate=lambda root:dict(EXPLICIT_CPU_INPUT_MOCK=True)
  for n,m in [('execution_control',fake),('continuation_admission',a),('record_validation',v),('qualification_input_admission',inp)]:sys.modules[n]=m
  spec=importlib.util.spec_from_file_location('cross_host_contract',R/'src/cross_host_contract.py')
  contract=importlib.util.module_from_spec(spec);sys.modules[spec.name]=contract;spec.loader.exec_module(contract)
  observer=[];stop=threading.Event()
  def inspect_and_admit():
   try:
    deadline=time.monotonic()+15
    while not(F/'LOCAL_QUALIFICATION_READY.json').exists():
     assert time.monotonic()<deadline and not stop.is_set(),'Controller did not reach actual ready'
     time.sleep(.01)
    # This is the original production CPU observer, not an inspection mock.
    receipt=contract.inspect_qualification(F)
    ledger=real.read(F/'RESOURCE_LEDGER.json')
    assert ledger['intervals']==[] and set(ledger['completed_units'])==set(qids)
    assert ledger['EXACT_physical_GPUh']=='UNRESOLVED'
    assert not calls and not list((F/'formal').glob('*/DISPATCH.json'))
    # No new formal can run while the other host's qualification is pending.
    time.sleep(.15);assert not calls
    peer=copy.deepcopy(receipt);peer.update(node_id='server29_gpu6',host='202.38.247.29',gpu_uuid=next(n['uuid']for n in layout['nodes']if n['id']=='server29_gpu6'),gpu_uuids=[n['uuid']for n in layout['nodes']if n['host']=='202.38.247.29'],qualification_count=38,reused_qualification_count=19,fresh_qualification_count=19)
    admission=dict(status='PASS_THREE_GPU57_MIXED_ORIGIN_QUALIFICATIONS_ALLOW_UNIQUE120',global_plan_sha256=real.sha(F/'CROSS_HOST_PLAN.json'),global_jobs_sha256=real.sha(F/'GLOBAL_FORMAL_JOB_MANIFEST.json'),human_grant_sha256=real.sha(F/'HUMAN_GPU_AUTHORIZATION.json'),nodes={'server55_gpu0':receipt,'server29_gpu6':peer},EXPLICIT_OTHER_HOST_QUALIFICATION_CPU_FIXTURE_NOT_CREDIT=True)
    contract.require_global_formal_admission(F,admission)
    put('GLOBAL_FORMAL_ADMISSION.json',admission)
    observer.append(dict(status='PASS_ACTUAL_INSPECTOR_ZERO_INTERVAL_READY',count=receipt['qualification_count']))
   except BaseException as e:
    observer.append(dict(status='OBSERVER_REJECTED',error=repr(e)))
    put('GLOBAL_STOP_REQUEST.json',dict(EXPLICIT_CPU_OBSERVER_STOP=True,error=repr(e)))
  class Child:
   def __init__(self,cmd,stdout,stderr,env,start_new_session):
    assert '--formal'in cmd and env['CUDA_VISIBLE_DEVICES']==node['uuid']
    assert (F/'AUTHORIZATION_FORMAL.json').is_file()and(F/'GLOBAL_FORMAL_ADMISSION.json').is_file()
    contract.require_global_formal_admission(F,real.read(F/'GLOBAL_FORMAL_ADMISSION.json'))
    unit=cmd[cmd.index('--unit')+1];job=next(j for j in jobs if j['id']==unit)
    assert unit not in calls;calls.append(unit);self.pid=900000+len(calls);self.returncode=0
    identity=dict(pid=self.pid,starttick=self.pid,cmdline=cmd);children[self.pid]=identity
    w=dict(unit=unit,status='PASS',gpu=0,gpu_uuid=node['uuid'],identity=identity,permit_sha256=real.sha(F/'GPU_PERMIT.json'),result=dict(EXPLICIT_CPU_MODEL_FIXTURE=True,condition=job['condition']))
    w['payload_sha256']=real.objsha(w);real.atomic_new(F/'formal'/unit/'RESULT.json',w)
   def poll(self):return self.returncode
   def wait(self,timeout=None):return self.returncode
   def terminate(self):raise AssertionError('Exited CPU child was signalled')
   def kill(self):raise AssertionError('Exited CPU child was signalled')
  subprocess.Popen=Child
  os.statvfs=lambda p:types.SimpleNamespace(f_bavail=100*1024**3,f_frsize=1)if Path(p)==F/'reference'else original_statvfs(p)
  t=threading.Thread(target=inspect_and_admit);t.start()
  script=(R.parent/'three_gpu_continuation_v5/scripts/gpu_controller.py')if old else R/'scripts/gpu_controller.py'
  sys.argv=[str(script),'--root',str(F),'--execute']
  try:runpy.run_path(str(script),run_name='__main__')
  except RuntimeError:
   assert old
  else:assert not old
  finally:stop.set();t.join(timeout=20);assert not t.is_alive()
  if old:
   assert observer[0]['status']=='OBSERVER_REJECTED'and 'RESOURCE_LEDGER.json'in observer[0]['error']
   assert len(calls)==0 and real.read(F/'SESSION_RESULT.json')['status']=='FAIL_CLOSED_STOP'
   checks.append(dict(case='CLOSED_V5_TRUE_MISSING_LEDGER_BUG_REPRODUCED_WITH_PRODUCTION_INSPECTOR',formal_CPU_children=0,old_source_not_modified=True))
  else:
   assert observer[0]['status']=='PASS_ACTUAL_INSPECTOR_ZERO_INTERVAL_READY'
   assert len(calls)==40 and set(calls)=={j['id']for j in jobs}
   intervals=real.read(F/'RESOURCE_LEDGER.json')['intervals'];assert len(intervals)==40 and all(i['release']=='PASS'for i in intervals)
   assert not any((F/'execution'/u/'DISPATCH.json').exists()for u in qids)
   assert real.read(F/'SESSION_RESULT.json')['status']=='GPU_COMPLETE_GROUP64_NODE_ALL_PLANNED'
   checks.append(dict(case='V6_ZERO_NEW_INTERVAL_REAL_CONTROLLER_READY_INSPECTOR_AND_BARRIER_PASS',reused19_never_dispatched=True,formal_CPU_children=40,formal_before_global_barrier=0,ledger_contains_only_new_CPU_child_intervals=True))
   saved=(F/'RESOURCE_LEDGER.json').read_bytes();(F/'RESOURCE_LEDGER.json').unlink()
   try:contract.inspect_qualification(F)
   except RuntimeError:pass
   else:raise AssertionError('Missing ledger was silently accepted')
   (F/'RESOURCE_LEDGER.json').write_bytes(saved)
   checks.append(dict(case='PRODUCTION_INSPECTOR_STILL_REJECTS_MISSING_LEDGER'))
subprocess.Popen=original_popen;os.statvfs=original_statvfs
assert 'torch'not in sys.modules
print(json.dumps(dict(status='PASS_REAL_CONTROLLER_QUALIFICATION_READY_INSPECTOR_LEDGER_REPAIR_CPU_INTEGRATION',checks=checks,count=len(checks),GPU_calls=0,model_forwards=0,qualification_credit=0,formal_credit=0,scope='CPU device/native/child/model/other-host-admission explicit mocks; actual original19 qualification payload/trace field reads and actual production inspector/barrier'))) 
