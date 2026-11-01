import contextlib,hashlib,io,json,sys,tempfile,types,subprocess
from pathlib import Path
path=Path('/Users/pytenter/Documents/临时项目1/qwen_hadamard_dual_axis_direct_32k_pilot_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1/src/coordinator.py');code=path.read_text();compile(code,str(path),'exec')
def module():
 m=types.ModuleType('coordinator_cpu_test');m.__file__=str(path);exec(compile(code,str(path),'exec'),m.__dict__);return m
m=module();assert len(m.PHASES)==8 and len(m.SCIENCE_PHASES)==6
pred,_=m.remaining_science_projection({'projected_remaining_GPU_seconds':660,'phase_predictions_seconds':{p:100 for p in m.SCIENCE_PHASES}},m.SCIENCE_PHASES[2:],{'phases':[]});assert pred==460
assert m.exact_child({'pid':42,'cmdline':['python',str(m.ROOT/'src/worker.py'),'canary_initial']},'canary_initial')
assert not m.exact_child({'pid':42,'cmdline':['python','/other/worker.py','canary_initial']},'canary_initial')
class Clock:
 def __init__(self):self.t=0.
 def monotonic(self):return self.t
 def time(self):return 1700000000+self.t
 def sleep(self,s):self.t+=s

def run_case(mode):
 m=module();clock=Clock();m.time=clock
 with tempfile.TemporaryDirectory(prefix='qwen_coordinator_cpu_mock_') as tmp:
  r=Path(tmp);m.ROOT=r
  def save(rel,value):
   p=r/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+'\n')
  (r/'src').mkdir();(r/'src/coordinator.py').write_text(code);(r/'src/worker.py').write_text('# CPU mock placeholder\n')
  save('preregistration.json',{'GPU_EXECUTION_AUTHORIZED':'YES'});save('frozen_documents.json',{'mock':True})
  save('source_manifest.json',{'execution_sources':[{'path':x,'sha256':hashlib.sha256((r/x).read_bytes()).hexdigest()} for x in ['src/coordinator.py','src/worker.py','preregistration.json','frozen_documents.json']]})
  for x in ['ENGINEERING_ADAPTER_CPU.json','CPU_STRUCTURE_001.json','MODEL_CONTENT_IDENTITY.json']:save('qualification_cpu/'+x,{'status':'PASS'})
  m.frozen_document_check=lambda:None
  called=[];children={}
  class Child:
   def __init__(self,args,**kwargs):
    self.args=args;self.phase=args[-1];self.pid=41000+len(called);called.append(self.phase);children[self.pid]=self;self.returncode=None if mode=='foreign' else 0
    save('analysis/RECEIPT_'+self.phase+'.json',{'status':'PASS','phase':self.phase})
    if self.phase=='canary_fresh':save('G4_RUNTIME_BUDGET.json',{'status':'RESOURCE_BLOCKED' if mode=='g4_block' else 'PASS','projected_remaining_GPU_seconds':90000 if mode=='budget_block' else 6000,'phase_predictions_seconds':{p:1000 for p in m.SCIENCE_PHASES}})
    if mode=='worker_failure':save('FIRST_FAILURE.json',{'status':'FAIL_AND_STOP','phase':self.phase,'worker_failure':True})
   def poll(self):return self.returncode
   def wait(self,timeout=None):return self.returncode
   def send_signal(self,sig):self.returncode=-sig
  m.subprocess=types.SimpleNamespace(Popen=Child,TimeoutExpired=subprocess.TimeoutExpired,STDOUT=subprocess.STDOUT)
  def probe():
   apps=[]
   if mode=='foreign' and children:apps=[{'pid':99999,'gpu_uuid':m.GPU_UUID,'process_name':'foreign','used_gpu_memory_MiB':'20'}]
   return {'status':'PASS','unix':clock.time(),'physical_id':0,'uuid':m.GPU_UUID,'name':'NVIDIA GeForce RTX 3090','total_MiB':24576,'free_MiB':24256,'used_MiB':3,'utilization_percent':0,'apps':apps}
  m.nvml_probe=probe;m.process_identity=lambda pid:{'pid':pid,'start_time_ticks':'123','cmdline':children[pid].args}
  if mode=='release_unverified':m.safe_probe=lambda:{'status':'QUERY_FAILED','unix':clock.time(),'error':'CPU mock unavailable NVML'}
  with contextlib.redirect_stdout(io.StringIO()):rc=m.main()
  final=json.loads((r/'GATE_STATUS_FINAL.json').read_text());ledger=json.loads((r/'gpu_resource_ledger.json').read_text())
  if mode=='complete':assert rc==0 and called==m.PHASES and final['status']=='COMPLETE_AND_STOP' and final['GPU_release_verified'] is True
  elif mode in ['g4_block','budget_block']:
   assert rc==1 and called==m.PHASES[:2] and final['status']=='BLOCKED_AND_STOP',final
   assert all(final['gates'][p]=='NOT_RUN' for p in m.SCIENCE_PHASES)
  elif mode=='foreign':
   assert called==m.PHASES[:1] and final['status']=='FAIL_AND_STOP',final
   assert children[41000].returncode==-15 and ledger['phases'][0]['cleanup']['termination_requested'] is True
   assert ledger['phases'][0]['release_probe']['apps'][0]['pid']==99999
  elif mode=='release_unverified':
   assert called==m.PHASES[:1] and final['GPU_release_verified'] is False,final
   assert ledger['phases'][0]['released'] is False and ledger['phases'][0]['reservation_seconds']>=30
   assert ledger['phases'][0]['reservation_duration_is_lower_bound_if_release_unverified'] is True
  elif mode=='worker_failure':assert called==m.PHASES[:1] and json.loads((r/'FIRST_FAILURE.json').read_text())['worker_failure'] is True
  before=(r/'GATE_STATUS_FINAL.json').read_bytes()
  with contextlib.redirect_stdout(io.StringIO()):assert m.main()==2
  assert (r/'GATE_STATUS_FINAL.json').read_bytes()==before
  return {'case':mode,'status':'PASS','dispatched_phases':len(called),'result':final['status']}
results=[run_case(x) for x in ['complete','g4_block','budget_block','foreign','release_unverified','worker_failure']]
assert not any(x in sys.modules for x in ['torch','transformers'])
print(json.dumps({'status':'PASS_CPU_STATIC_AND_MOCK_CONTROL_TESTS','coordinator_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parse_compile':'PASS','external_GPU_commands_executed':0,'CUDA_libraries_imported':False,'pure_policy_and_exact_PID_tests':'PASS','cases':results},indent=2))
