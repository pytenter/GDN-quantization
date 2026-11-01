"""Execute actual coordinator with CPU-only transport/node-process mocks. Three finite dispatch/STOP scenarios."""
import ast,copy,json,os,runpy,subprocess,sys,tempfile,time,types
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));import execution_control as C
checks=[];saved={n:getattr(C,n)for n in ['NODE','authorize','device','process_identity']};save_run=subprocess.run;save_popen=subprocess.Popen;save_sleep=time.sleep;real_rpc=sys.modules.get('node_rpc')
C.NODE=next(n for n in C.read(R/'CROSS_HOST_PLAN.json')['nodes']if n['id']=='server55_gpu0');C.authorize=lambda *a:None;C.device=lambda *a:dict(CPU_DEVICE_MOCK=True);C.process_identity=lambda pid:dict(pid=pid,CPU_PROCESS_MOCK=True)
class Child:
 pid=900001
 def __init__(self):self.returncode=None
 def poll(self):return self.returncode
 def wait(self):self.returncode=0;return 0
with tempfile.TemporaryDirectory(prefix='g64_COORDINATOR_CPU_ONLY_NO_GPU_')as tmp:
 for scenario in ['both_pass','peer_failure','unknown_launch_reply']:
  F=Path(tmp)/scenario;F.mkdir();(F/'operations').mkdir();child=Child();actions=[];remote_stopped=False
  for name in ['CROSS_HOST_PLAN.json','GLOBAL_FORMAL_JOB_MANIFEST.json','QUESTION_GPU_ASSIGNMENT.json','AIME26_DATASET_FREEZE.json','HUMAN_GPU_AUTHORIZATION.json']:(F/name).write_bytes((R/name).read_bytes())
  for name in ['SOURCE_FREEZE.json','GPU_PERMIT.json','GROUP64_GPU_CODEC_QUALIFICATION.json']:(F/name).write_text('{}')
  def state(node):
   stopped=(F/'GLOBAL_STOP_REQUEST.json').exists();formal=(F/'GLOBAL_FORMAL_ADMISSION.json').exists()
   if formal or stopped:child.returncode=0
   v=dict(controller_alive=not(formal or stopped),active=[],node_parent_alive=False,node_tmux_alive=False)
   if stopped and scenario=='unknown_launch_reply':return v
   v['GPU_RUN_STARTED.json']=dict(identity=dict(pid=900002))
   if formal:v['SESSION_RESULT.json']=dict(status='GPU_COMPLETE_GROUP64_HOST19_FORMAL60');v['operations/NODE_ACTUAL_EXIT.json']=dict(exit=0)
   elif scenario=='peer_failure'and node=='server29_gpu6':v['FIRST_FAILURE.json']=dict(error='CPU_INJECTED_PEER_FAILURE')
   else:v['LOCAL_QUALIFICATION_READY.json']=dict(CPU_FIXTURE=True)
   return v
  def receipt(node):
   n=next(n for n in C.read(F/'CROSS_HOST_PLAN.json')['nodes']if n['id']==node)
   return dict(node_id=node,host=n['host'],gpu_uuid=n['uuid'],qualification_count=19,all19_actual_exit0_release_PASS=True,own_worker_processes_and_UUID_released=True,source_sha256=C.sha(F/'SOURCE_FREEZE.json'),permit_sha256=C.sha(F/'GPU_PERMIT.json'),qualification_sha256=C.sha(F/'GROUP64_GPU_CODEC_QUALIFICATION.json'),model_bytes_identity_sha256='1'*64)
  class Pending(RuntimeError):pass
  def rpc(code,timeout=55):
   actions.append(code)
   if code=='CPU_STATE':return state('server29_gpu6')
   if 'inspect_qualification'in code:return receipt('server29_gpu6')
   if "tmux','new-session'"in code:
    if scenario=='unknown_launch_reply':raise Pending('CPU lost reply after possible unique launch')
    return dict(status='CPU_LAUNCH_MOCK')
   if 'value='in code:
    value=next(ast.literal_eval(n.value)for n in ast.parse(code).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='value')
    assert value['status']in ['GLOBAL_TRUE_STOP_REQUEST','PASS_BOTH_HOST19_TRUE_QUALIFICATIONS_ALLOW_FORMAL120_ONCE'];return dict(status='PASS_EXACT_COORDINATION_RECEIPT')
   return dict(status='CPU_ONLY_IDLE_CHECK')
  fake=types.ModuleType('node_rpc');fake.prefix=lambda:'';fake.state_code=lambda:'CPU_STATE';fake.query29=rpc;fake.ConnectionPending=Pending;sys.modules['node_rpc']=fake
  def localrun(cmd,**kw):
   code=kw['input'];v=state('server55_gpu0')if code=='CPU_STATE'else receipt('server55_gpu0');return subprocess.CompletedProcess(cmd,0,stdout=json.dumps(v),stderr='')
  subprocess.run=localrun;subprocess.Popen=lambda *a,**kw:child
  sleeps=[0]
  def sleep(n):
   sleeps[0]+=1;assert sleeps[0]<10,'CPU coordinator unexpectedly fails to reach terminal'
  time.sleep=sleep;sys.argv=[str(R/'scripts/global_coordinator.py'),'--root',str(F)]
  try:runpy.run_path(str(R/'scripts/global_coordinator.py'),run_name='__main__')
  except (RuntimeError,Pending):assert scenario!='both_pass'
  else:assert scenario=='both_pass'
  terminal=C.read(F/'GLOBAL_SESSION_RESULT.json')
  if scenario=='both_pass':
   assert terminal['status']=='GPU_COMPLETE_CROSS_HOST_GROUP64_120_FORMAL'and terminal['formal']==120 and(F/'GLOBAL_FORMAL_ADMISSION.json').exists();checks.append('Actual coordinator both19 -> common admission -> both60 -> actual node exits')
  else:
   assert terminal['status']=='FAIL_CLOSED_CROSS_HOST_STOP'and(F/'GLOBAL_FIRST_FAILURE.json').exists()and not(F/'GLOBAL_FORMAL_ADMISSION.json').exists()
   assert any('GLOBAL_STOP_REQUEST.json'in s for s in actions);checks.append('Actual coordinator '+scenario+' preserves cause and publishes peer STOP without relaunch')
  assert sum("tmux','new-session'"in s for s in actions)==1;checks.append('No repeated launch in '+scenario)
subprocess.run=save_run;subprocess.Popen=save_popen;time.sleep=save_sleep
for k,v in saved.items():setattr(C,k,v)
if real_rpc is None:sys.modules.pop('node_rpc',None)
else:sys.modules['node_rpc']=real_rpc
assert 'torch'not in sys.modules
print(json.dumps(dict(status='PASS_ACTUAL_CROSS_HOST_COORDINATOR_CPU_SCENARIOS',count=len(checks),checks=checks,GPU_dispatches=0,transport_and_processes='EXPLICIT_CPU_MOCKS',qualification_credit=0,formal_credit=0)))
