"""CPU production-controller regression: delayed wait status, real child exits, positive identity changes."""
import argparse,copy,importlib.util,io,json,os,subprocess,sys,tempfile,time,types
from pathlib import Path
from unittest.mock import patch
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();R=a.root.resolve();checks=[]
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v
new=load('production_repaired_controller',R/'scripts/resident_controller.py');old=load('preserved_previous_controller',R/'provenance/PRIOR_295_STOP/resident_controller.py')
def ok(name,condition):
 assert condition,name
 checks.append(name)
def reject(name,fn):
 try:fn()
 except (RuntimeError,FileNotFoundError):checks.append(name);return
 raise AssertionError('Expected rejection: '+name)
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f)
class Child:
 pid=987654321
 def __init__(self,waits):self.waits=iter(waits);self.returncode=None;self.wait_calls=0
 def poll(self):return self.returncode
 def wait(self,timeout):
  self.wait_calls+=1;v=next(self.waits)
  if v is None:raise subprocess.TimeoutExpired(['CPU_FIXTURE_ONLY'],timeout)
  self.returncode=v;return v
def setup(root,module,child,current=None):
 expected=dict(pid=child.pid,starttick=123,cmdline=['CPU_FIXTURE_CHILD_ONLY']);own=dict(pid=os.getpid(),starttick=1,cmdline=['CPU_FIXTURE_CONTROLLER_ONLY'])
 class C:
  TASK='CPU_ONLY_FIXTURE';UUIDS={1:'CPU_UUID_1'}
  process_identity=staticmethod(lambda pid:own if pid==os.getpid() else current)
  same_process=staticmethod(lambda v: C.process_identity(v['pid'])==v)
  query_gpu=staticmethod(lambda:dict(apps=[]))
  read=staticmethod(lambda p:json.loads(p.read_text()))
  sha=staticmethod(lambda p:'CPU_FIXTURE_HASH')
  atomic_new=staticmethod(write)
  event=staticmethod(lambda *args,**kwargs:None)
  validate_qualification_result=staticmethod(lambda root,unit:C.read(root/'qualification'/unit['id']/'RESULT.json'))
 controller=module.Controller(root,C);target=root/'qualification/CPU_UNIT';target.mkdir(parents=True)
 now=time.monotonic();record=dict(process=child,identity=expected,uuid='CPU_UUID_1',gpu=1,phase='qualification',unit=dict(id='CPU_UNIT'),target=target,exit_observed=None,exit_returncode=None,last_progress_monotonic=now,progress_signature=None,latest_progress=None,dispatch=dict(dispatch_monotonic=now,source_freeze_sha256='CPU_FIXTURE_HASH',permit_sha256='CPU_FIXTURE_HASH'),identity_observed_monotonic=now,log=io.StringIO())
 controller.active[1]=record;released=[]
 def complete(row,release,**kwargs):released.append((row['exit_returncode'],release));del controller.active[1]
 controller.complete_released=complete
 return controller,record,released,C
with tempfile.TemporaryDirectory(prefix='resident_exit_transition_cpu_') as tmp:
 t=Path(tmp)
 c,rec,done,_=setup(t/'old',old,Child([None,0]))
 reject('old_two_nonblocking_polls_reproduce_false_STOP_under_missing_identity',c.poll)
 c,rec,done,_=setup(t/'actual_wait',new,Child([0]));c.poll()
 ok('new_uses_actual_owned_Popen_wait0',done==[(0,'PASS')] and rec['process'].wait_calls==1)
 c,rec,done,_=setup(t/'pending',new,Child([None,None,0]))
 c.poll();ok('first_pending_wait_does_not_invent_exit_or_release',not done and 1 in c.active and rec['exit_observed'] is None)
 c.poll();ok('second_pending_wait_does_not_admit_new_work',not done and 1 in c.active and rec['exit_observed'] is None)
 c.poll();ok('eventual_actual_wait_status_drives_release',done==[(0,'PASS')])
 proof=json.loads(Path(rec['exit_transition_observation']).read_text());ok('durable_expected_current_raw_proc_evidence',proof['expected']==rec['identity'] and 'raw_proc' in proof and proof['no_exit_or_PASS_inferred_from_proc'])
 c,rec,done,_=setup(t/'empty_argv',new,Child([0]),dict(pid=Child.pid,starttick=123,cmdline=['']));c.poll()
 ok('same_starttick_empty_teardown_argv_requires_actual_wait',done==[(0,'PASS')] and rec['process'].wait_calls==1)
 for name,current in [('starttick',dict(pid=Child.pid,starttick=124,cmdline=['CPU_FIXTURE_CHILD_ONLY'])),('argv',dict(pid=Child.pid,starttick=123,cmdline=['CPU_FOREIGN_COMMAND']))]:
  child=Child([0]);c,rec,done,_=setup(t/name,new,child,current)
  reject('positive_live_'+name+'_change_still_fails',c.poll)
  ok('positive_'+name+'_mismatch_never_waited_as_success',not done and child.wait_calls==0 and Path(rec['exit_transition_observation']).exists())
 c,rec,done,C=setup(t/'nonzero',new,Child([7]));c.complete_released=types.MethodType(new.Controller.complete_released,c)
 reject('actual_nonzero_wait7_is_not_turned_into_PASS',c.poll)
 interval=json.loads((rec['target']/'RESOURCE_INTERVAL.json').read_text());ok('actual_nonzero7_persisted_in_interval',interval['exit_returncode']==7 and not c.completed['qualification'])
 c,rec,done,C=setup(t/'missing_result',new,Child([0]));c.complete_released=types.MethodType(new.Controller.complete_released,c)
 reject('exit0_without_actual_result_still_fails_validation',c.poll)
 ok('missing_result_not_counted_as_qualification_PASS',not c.completed['qualification'])
 c,rec,done,C=setup(t/'watchdog',new,Child([None,None]));write(rec['target']/'PROGRESS.json',dict(phase='UNIT_PASS',token=64,monotonic=10))
 with patch.object(new.time,'monotonic',return_value=1000.):c.poll()
 (rec['target']/'PROGRESS.json').write_text(json.dumps(dict(phase='UNIT_PASS',token=64,monotonic=2000)))
 with patch.object(new.time,'monotonic',return_value=2801.):reject('pending_identity_does_not_disable1800_real_progress_watchdog',c.poll)
 ok('watchdog_did_not_fake_child_exit',not done and rec['exit_returncode'] is None)
 for i in range(12):
  code=0 if i%2==0 else 7
  p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.04);raise SystemExit('+str(code)+')'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  try:
   c,rec,done,C=setup(t/('real_%02d'%i),new,p)
   deadline=time.monotonic()+5
   while c.active and time.monotonic()<deadline:c.poll();time.sleep(.001)
   actual=p.wait(timeout=5);ok('real_fresh_CPU_child_%02d_actual_wait%d'%(i,code),done==[(actual,'PASS')] and actual==code and not c.active)
  finally:
   if p.poll() is None:p.terminate();p.wait(timeout=5)
ok('no_Torch_or_CUDA_import','torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='')
import hashlib
result=dict(status='PASS_CPU_PRODUCTION_CONTROLLER_EXIT_TRANSITION_REGRESSION',count=len(checks),checks=checks,controller_sha256=hashlib.sha256((R/'scripts/resident_controller.py').read_bytes()).hexdigest(),old_controller_sha256=hashlib.sha256((R/'provenance/PRIOR_295_STOP/resident_controller.py').read_bytes()).hexdigest(),GPU_dispatches=0,model_loads=0,real_fresh_CPU_child_processes=12,all_negative_fixtures_outside_root=True,no_live_identity_guard_weakened=True,actual_child_exit0_not_inferred=True)
write(a.output,result);print(json.dumps(dict(status=result['status'],checks=len(checks))))
