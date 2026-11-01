#!/usr/bin/env python3
"""Synthetic CPU-only controller authorization/resource/PID tests. No GPU APIs.

YES records exist only inside temporary fake roots with fake one-byte model
files and injected GPU/process readers. They are not real human authorization.
"""
import argparse,copy,hashlib,importlib.util,json,os,sys,tempfile,types
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,default=Path(__file__).resolve().parents[1]/'scripts/restoration_controller.py');args=p.parse_args();spec=importlib.util.spec_from_file_location('_restoration_controller_CPU_tests',args.controller);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
checks=[];gpu_calls=0
def ok(name,condition=True):
 assert condition,name;checks.append({'name':name,'status':'PASS'})
def reject(name,fn,text=None):
 try:fn()
 except RuntimeError as e:
  if text:assert text in str(e),(name,str(e),text)
  ok(name);return
 raise AssertionError(name+' unexpectedly admitted')
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def make(parent):
 root=(parent/'task').resolve();root.mkdir();external=parent/'external';external.mkdir();(root/'scripts').mkdir();(root/'worker').mkdir();(root/'qualification').mkdir();(root/'USER_REQUEST.txt').write_text('SYNTHETIC CPU ONLY REQUEST');(root/'scripts/restoration_controller.py').write_text('# synthetic frozen controller\n');(root/'worker/restoration_worker.py').write_text('# synthetic worker, never executed\n');weights=[]
 for i in range(4):
  q=external/('model'+str(i)+'.safetensors');q.write_bytes(bytes([i]));weights.append({'path':str(q),'bytes':1,'sha256':C.sha(q)})
 bindings='analysis/environment/PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json';save(root/bindings,{'model_weight_shards':weights});plan=dict(schema='QWEN_HISTORICAL_RUNTIME_GPU_PLAN_V1',task_id=C.TASK,device={'physical_index':0,'uuid':C.UUID},max_active_GPUs=1,numeric_tokens=32,warmup_tokens=128,measured_tokens=128,minimum_disk_free_bytes=10*1024**3,max_numeric_raw_bytes=8*1024**3,min_gpu_free_bytes=22*1024**3,units=copy.deepcopy(C.UNITS),binding_files=[bindings],request_sha256=C.sha(root/'USER_REQUEST.txt'),**C.LIMITS);save(root/'qualification/GPU_PLAN.json',plan);save(root/'ROOT.json',{'task_id':C.TASK,'ALLOW_GPU_RESTORE':'NO'});files=[]
 for name in ['scripts/restoration_controller.py','worker/restoration_worker.py','qualification/GPU_PLAN.json','USER_REQUEST.txt',bindings]:
  q=root/name;files.append({'path':name,'bytes':q.stat().st_size,'sha256':C.sha(q)})
 save(root/'SOURCE_FREEZE.json',{'files':files});auth={'task_id':C.TASK,'scope':C.SCOPE,'ALLOW_GPU_RESTORE':'YES','request_sha256':C.sha(root/'USER_REQUEST.txt'),'plan_sha256':C.sha(root/'qualification/GPU_PLAN.json'),'source_freeze_sha256':C.sha(root/'SOURCE_FREEZE.json'),'device_uuid':C.UUID,'max_active_GPUs':1,'max_GPU_hours':.5,'allowed_units':[u['id']for u in C.UNITS],'human_request_quote':'SYNTHETIC CPU FIXTURE ONLY, NO REAL GPU AUTHORIZATION','synthetic_non_executable_CPU_fixture':True};save(root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',auth);return root,plan,auth

def no_gpu():
 global gpu_calls;gpu_calls+=1;raise AssertionError('Physical GPU query forbidden in CPU tests')
with tempfile.TemporaryDirectory(prefix='restoration_controller_CPU_ONLY_')as temporary:
 base=Path(temporary);root,plan,auth=make(base);authpath=root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json'
 with mock.patch.object(C,'check_inputs',side_effect=AssertionError('Heavy external read must not occur under ROOT NO')):
  reject('ROOT_NO_stops_before_external_hash_reads',lambda:C.run_controller(root,authpath,'YES',gpu_reader=no_gpu),'ROOT ALLOW_GPU_RESTORE')
 reject('ROOT_NO_denies_before_any_GPU_query',lambda:C.run_controller(root,authpath,'YES',gpu_reader=no_gpu), 'ROOT ALLOW_GPU_RESTORE')
 save(root/'ROOT.json',{'task_id':C.TASK,'ALLOW_GPU_RESTORE':'YES'});reject('CLI_default_NO_denies_before_any_GPU_query',lambda:C.run_controller(root,authpath,'NO',gpu_reader=no_gpu),'CLI YES')
 for field,value,name in [('ALLOW_GPU_RESTORE','NO','human_NO'),('task_id','wrong','wrong_task'),('scope','old_four_card_authorization','old_scope'),('human_request_quote',' ','empty_human_quote'),('device_uuid','GPU-foreign','wrong_UUID'),('max_GPU_hours',1,'budget_expansion'),('max_active_GPUs',4,'four_GPU_authorization_not_accepted'),('allowed_units',['NUMERIC_FRESH0'],'incomplete_unit_roster'),('source_freeze_sha256','0'*64,'wrong_source_freeze_sha'),('plan_sha256','0'*64,'wrong_plan_sha'),('request_sha256','0'*64,'wrong_request_sha')]:
  bad=dict(auth);bad[field]=value;save(authpath,bad);reject(name+'_before_GPU',lambda:C.run_controller(root,authpath,'YES',gpu_reader=no_gpu))
 save(authpath,auth);before=(root/'worker/restoration_worker.py').read_bytes();(root/'worker/restoration_worker.py').write_text('mutated');reject('changed_frozen_source_denies_before_GPU',lambda:C.run_controller(root,authpath,'YES',gpu_reader=no_gpu));(root/'worker/restoration_worker.py').write_bytes(before)
 external=base/'external/model0.safetensors';external.write_bytes(b'x');reject('changed_model_SHA_denies_before_GPU',lambda:C.run_controller(root,authpath,'YES',gpu_reader=no_gpu));external.write_bytes(b'\0')
 ok('historical_external_source_binding_schema_supported',C.binding_rows({'external_source_files':[{'path':'/synthetic/source.py','bytes':7,'sha256':'f'*64}]})==[{'path':'/synthetic/source.py','bytes':7,'sha256':'f'*64}]);checked=C.check_inputs(root);old_env=C.worker_environment(C.UNITS[0]);new_env=C.worker_environment(C.UNITS[-1]);ok('OLD_startup_known_four_keys_genuinely_unset',all(k not in old_env for k in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']));ok('NEW_optional_exact_Strict_four_keys',all(new_env[k]==v for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}.items()));ok('CPU_check_passes_without_Torch_or_GPU',checked['external_files']==4 and gpu_calls==0)
 for key in C.LIMITS:
  bad=copy.deepcopy(plan);bad[key]+=1;reject('budget_boundary_'+key,lambda b=bad:C.validate_plan(b))
 bad=copy.deepcopy(plan);bad['units']=bad['units'][1:];reject('cannot_skip_first_numeric_fresh',lambda:C.validate_plan(bad));bad=copy.deepcopy(plan);bad['numeric_tokens']=31;reject('cannot_shorten_numeric_window',lambda:C.validate_plan(bad));bad=copy.deepcopy(plan);bad['max_numeric_raw_bytes']+=1;reject('cannot_expand_numeric_raw_cap',lambda:C.validate_plan(bad))
 ok('dispatch_exact240plus150_remaining_boundary',C.budget_action(C.UNITS[0],1410,plan)=='DISPATCH');ok('numeric_below_budget_STOPS_not_skips',C.budget_action(C.UNITS[0],1410.0001,plan)=='STOP_MINIMUM_NUMERIC_BUDGET');ok('benchmark_below_budget_skips_without_repeats',C.budget_action(C.UNITS[2],1410.0001,plan)=='SKIP_BUDGET');ok('optional_NEW_below_budget_skips',C.budget_action(C.UNITS[-1],1410.0001,plan)=='SKIP_BUDGET');reject('NaN_clock_not_budget_admitted',lambda:C.budget_action(C.UNITS[0],float('nan'),plan))
 observation={'devices':[{'physical_index':0,'uuid':C.UUID,'name':'RTX 3090','memory_total_MiB':24576,'memory_free_MiB':24256}],'apps':[]};ok('synthetic_single_card_inventory_accepted',C.validate_gpu(observation)['uuid']==C.UUID);bad=copy.deepcopy(observation);bad['apps']=[{'pid':777,'uuid':C.UUID}];reject('foreign_GPU_process_rejects_no_foreign_kill',lambda:C.validate_gpu(bad));bad=copy.deepcopy(observation);bad['devices'][0]['uuid']='GPU-foreign';reject('physical_UUID_mismatch_rejects',lambda:C.validate_gpu(bad))
 fake_stat=types.SimpleNamespace(f_bavail=9*1024**3,f_frsize=1)
 with mock.patch.object(C.os,'statvfs',return_value=fake_stat):reject('disk_10GiB_reserve_enforced',lambda:C.disk_guard(root,plan))
 fake_path=types.SimpleNamespace(is_file=lambda:True,stat=lambda:types.SimpleNamespace(st_size=8*1024**3+1))
 with mock.patch.object(C.os,'statvfs',return_value=types.SimpleNamespace(f_bavail=20*1024**3,f_frsize=1)),mock.patch.object(C.Path,'rglob',return_value=[fake_path]):reject('8GiB_raw_cap_enforced',lambda:C.disk_guard(root,plan))
 ledger=C.Ledger(base/'events.jsonl',clock=lambda:100);ledger.once('id:exit','FIRST_EXIT');ledger.once('id:exit','FIRST_EXIT');ok('first_exit_event_written_only_once',len((base/'events.jsonl').read_text().splitlines())==1)
 class Term:
  pid=5;terminated=False
  def terminate(self):self.terminated=True
 term=Term();ident={'pid':5,'starttick':7,'cmdline':['only-own']};reject('PID_reuse_does_not_kill_foreign',lambda:C.terminate_exact(term,ident,lambda pid:dict(ident,starttick=8)));ok('PID_reuse_no_signal_sent',not term.terminated);C.terminate_exact(term,ident,lambda pid:ident);ok('exact_owned_process_can_be_terminated',term.terminated)
 # Durable permit/parent/worker binding, all temporary synthetic processes.
 meta=C.authorize(root,'YES',authpath);controller_id={'pid':222,'starttick':77,'cmdline':[sys.executable,str(root/'scripts/restoration_controller.py'),'--root',str(root),'--run']};phase=root/'execution/NUMERIC_FRESH0';phase.mkdir(parents=True);pp=phase/'PERMIT.json';command=[sys.executable,'-B',str(root/'worker/restoration_worker.py'),'--root',str(root),'--mode','numeric','--unit','NUMERIC_FRESH0','--permit',str(pp)];own={'pid':os.getpid(),'starttick':888,'cmdline':command};permit=dict(task_id=C.TASK,scope=C.SCOPE,root=str(root),unit=C.UNITS[0],device=plan['device'],controller_identity=controller_id,worker_command=command,plan_relative_path='qualification/GPU_PLAN.json',dispatch_deadline_monotonic=340,environment_controls={k:C.worker_environment(C.UNITS[0]).get(k)for k in C.ENV_KEYS},**{k:meta[k]for k in ['plan_sha256','source_freeze_sha256','request_sha256','authorization_sha256']});save(pp,permit);save(root/'execution/CONTROLLER_STARTED.json',dict(controller_identity=controller_id,**{k:meta[k]for k in ['plan_sha256','source_freeze_sha256','request_sha256','authorization_sha256']}));save(phase/'DISPATCH.json',dict(worker_identity=own,permit_sha256=C.sha(pp)))
 def reader(pid):return controller_id if pid==222 else own
 with mock.patch.dict(C.os.environ,C.worker_environment(C.UNITS[0]),clear=True):
  verified=C.require_worker_permit(root,'numeric','NUMERIC_FRESH0',pp,process_reader=reader,parent_pid=222,monotonic=lambda:100);ok('worker_permit_exact_direct_parent_and_own_identity',verified['permit_sha256']==C.sha(pp));reject('worker_wrong_mode_denied',lambda:C.require_worker_permit(root,'benchmark','NUMERIC_FRESH0',pp,process_reader=reader,parent_pid=222,monotonic=lambda:100));reject('worker_expired_dispatch_denied',lambda:C.require_worker_permit(root,'numeric','NUMERIC_FRESH0',pp,process_reader=reader,parent_pid=222,monotonic=lambda:400));reject('worker_foreign_parent_denied',lambda:C.require_worker_permit(root,'numeric','NUMERIC_FRESH0',pp,process_reader=reader,parent_pid=223,monotonic=lambda:100));reject('worker_outside_permit_path_denied',lambda:C.require_worker_permit(root,'numeric','NUMERIC_FRESH0',root/'fake.json',process_reader=reader,parent_pid=222,monotonic=lambda:100));reject('ended_worker_missing_receipt_no_retry',lambda:C.validate_result(phase,C.UNITS[0],own,C.sha(pp)))
 # Complete finite scheduling is exercised through fake children and fake NVML.
 second=base/'second';second.mkdir();root2,plan2,auth2=make(second);save(root2/'ROOT.json',{'task_id':C.TASK,'ALLOW_GPU_RESTORE':'YES'});children={};next_pid=[9000];now=[0.]
 def clock():now[0]+=.1;return now[0]
 controller2={'pid':os.getpid(),'starttick':55,'cmdline':[sys.executable,str(root2/'scripts/restoration_controller.py'),'--root',str(root2),'--run']}
 class FakeChild:
  def __init__(self,command,**kw):
   next_pid[0]+=1;self.pid=next_pid[0];self.returncode=None;assert kw['env']['CUDA_VISIBLE_DEVICES']=='0'and kw['env']['PYTHONDONTWRITEBYTECODE']=='1';self.alive=True;self.identity={'pid':self.pid,'starttick':self.pid+1,'cmdline':command};children[self.pid]=self;permit=read_json(Path(command[-1]));phase=Path(command[-1]).parent;save(phase/'RESULT.json',{'status':'PASS','unit_id':permit['unit']['id'],'mode':permit['unit']['mode'],'process_identity':self.identity,'permit_sha256':C.sha(Path(command[-1]))})
  def poll(self):self.alive=False;self.returncode=0;return 0
  def terminate(self):raise AssertionError('No fake child should need termination')
  def kill(self):raise AssertionError('No fake child should need kill')
 def read_json(p):return json.loads(p.read_text())
 def reader2(pid):
  if pid==os.getpid():return controller2
  child=children[pid]
  if child.alive:return child.identity
  raise FileNotFoundError('synthetic exited')
 def fake_gpu():return copy.deepcopy(observation)
 with mock.patch.object(C,'disk_guard',return_value={'available_bytes':20*1024**3,'numeric_raw_bytes':0}):complete=C.run_controller(root2,root2/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=fake_gpu,popen=FakeChild,reader=reader2,clock=clock,sleeper=lambda _:None)
 ok('mock_finite_two_numeric_then_three_OLD_and_optional_NEW',complete['completed_units']==[u['id']for u in C.UNITS]and complete['minimum_numeric_complete']and complete['benchmark_repeats']==3)
 events=[json.loads(x)for x in(root2/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines()];ok('each_mock_dispatch_exit_release_first_event_once',all(sum(x['event']==name for x in events)==6 for name in ['FIRST_DISPATCH','FIRST_PROCESS_EXIT_OBSERVED','FIRST_NVML_RELEASE_OBSERVED']));ok('mock_cost_bounds_keep_physical_GPUh_UNKNOWN',all(x.get('actual_physical_GPU_hours')=='UNRESOLVED_NOT_ZERO'for x in events if x['event']=='UNIT_COST_BOUNDS'));reject('started_controller_cannot_restart',lambda:C.run_controller(root2,root2/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=no_gpu))
 # Predispatch resource failure is immutable, and does not query GPU or launch.
 third=base/'third';third.mkdir();root3,_,_=make(third);save(root3/'ROOT.json',{'task_id':C.TASK,'ALLOW_GPU_RESTORE':'YES'});ctrl3={'pid':os.getpid(),'starttick':101,'cmdline':['synthetic-controller']}
 with mock.patch.object(C,'disk_guard',side_effect=RuntimeError('synthetic disk reserve admission failure')):
  reject('predispatch_disk_failure_STOP_receipt',lambda:C.run_controller(root3,root3/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=no_gpu,popen=lambda *a,**kw:(_ for _ in()).throw(AssertionError('No dispatch')),reader=lambda _:ctrl3,clock=lambda:1.),'synthetic disk')
 stop3=read_json(root3/'FIRST_FAILURE.json');ok('predispatch_failure_retains_attempt_and_no_worker',stop3['cleanup']['status']=='NO_WORKER_DISPATCHED'and(root3/'execution/CONTROLLER_ATTEMPT_STARTED.json').is_file()and not(root3/'execution/CONTROLLER_STARTED.json').exists());reject('predispatch_failed_controller_no_retry',lambda:C.run_controller(root3,root3/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=no_gpu))
 # An already exited process may retain a compute PID. Cleanup must observe its
 # disappearance, and must not signal this exited process or assume release.
 def cleanup_case(label,release):
  top=base/label;top.mkdir();rr,_,_=make(top);save(rr/'ROOT.json',{'task_id':C.TASK,'ALLOW_GPU_RESTORE':'YES'});ticks=[0.];children={};calls={'disk':0,'gpu':0};signals=[];controller={'pid':os.getpid(),'starttick':202,'cmdline':['synthetic-controller']}
  def tick():ticks[0]+=.1;return ticks[0]
  def sleep(seconds):ticks[0]+=seconds
  class Exited:
   def __init__(self,command,**kw):self.pid=9200;self.returncode=0;self.identity={'pid':self.pid,'starttick':777,'cmdline':command};children[self.pid]=self
   def poll(self):return 0
   def terminate(self):signals.append('terminate')
   def kill(self):signals.append('kill')
  def rd(pid):
   if pid==os.getpid():return controller
   return children[pid].identity
  def dg(*a):
   calls['disk']+=1
   if calls['disk']==3:raise RuntimeError('synthetic post-dispatch disk failure')
   return {'available_bytes':20*1024**3,'numeric_raw_bytes':0}
  def gp():
   calls['gpu']+=1;out=copy.deepcopy(observation)
   if calls['gpu']>=3 and(not release or calls['gpu']==3):out['apps']=[{'pid':9200,'uuid':C.UUID}]
   return out
  with mock.patch.object(C,'disk_guard',side_effect=dg):reject(label+'_raises_without_retry',lambda:C.run_controller(rr,rr/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=gp,popen=Exited,reader=rd,clock=tick,sleeper=sleep),'synthetic post-dispatch disk failure')
  failure=read_json(rr/'FIRST_FAILURE.json');ev=[json.loads(x)for x in(rr/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines()];cost=next(x for x in ev if x['event']=='UNIT_COST_BOUNDS');ok(label+'_no_signal_to_exited_process',not signals);ok(label+'_release_only_observation',failure['cleanup']['NVML_release_verified']is release);ok(label+'_upper_bound_resolution',isinstance(cost['worker_device_reservation_seconds_upper'],float)if release else cost['worker_device_reservation_seconds_upper']=='UNRESOLVED');ok(label+'_exit_once',sum(x['event']=='FIRST_PROCESS_EXIT_OBSERVED'for x in ev)==1);return failure
 cleanup_case('exited_PID_lingers_then_releases',True);cleanup_case('exited_PID_never_releases',False)
 # Lower bounds use a known-alive observation, never the later exit observation.
 fourth=base/'fourth';fourth.mkdir();root4,_,_=make(fourth);save(root4/'ROOT.json',{'task_id':C.TASK,'ALLOW_GPU_RESTORE':'YES'});children4={};pids4=[9400];t4=[0.];controller4={'pid':os.getpid(),'starttick':303,'cmdline':['synthetic-controller']}
 def tick4():t4[0]+=.1;return t4[0]
 class AliveThenExited:
  def __init__(self,command,**kw):
   pids4[0]+=1;self.pid=pids4[0];self.returncode=None;self.polls=0;self.alive=True;self.identity={'pid':self.pid,'starttick':self.pid+2,'cmdline':command};children4[self.pid]=self;pp=Path(command[-1]);pm=read_json(pp);save(pp.parent/'RESULT.json',dict(status='PASS',unit_id=pm['unit']['id'],mode=pm['unit']['mode'],process_identity=self.identity,permit_sha256=C.sha(pp)))
  def poll(self):
   self.polls+=1
   if self.polls<3:return None
   self.alive=False;self.returncode=0;return 0
  def terminate(self):raise AssertionError('No termination expected')
  def kill(self):raise AssertionError('No kill expected')
 def rd4(pid):
  if pid==os.getpid():return controller4
  if children4[pid].alive:return children4[pid].identity
  raise FileNotFoundError('synthetic exit')
 with mock.patch.object(C,'disk_guard',return_value={'available_bytes':20*1024**3,'numeric_raw_bytes':0}):C.run_controller(root4,root4/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=fake_gpu,popen=AliveThenExited,reader=rd4,clock=tick4,sleeper=lambda _:None)
 ev4=[json.loads(x)for x in(root4/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines()];costs4=[x for x in ev4 if x['event']=='UNIT_COST_BOUNDS'];dispatch4={x['unit']['id']:x for x in ev4 if x['event']=='FIRST_DISPATCH'};ok('reservation_lower_bound_exact_last_alive_minus_dispatch_after',all(abs(x['worker_device_reservation_seconds_lower']-(x['last_process_alive_monotonic']-dispatch4[x['unit_id']]['after_monotonic']))<1e-9 for x in costs4));ok('reservation_lower_bound_strictly_before_exit_observation',all(x['last_process_alive_monotonic']<x['process_exit_observed_monotonic']for x in costs4));ok('reservation_lower_le_upper',all(x['worker_device_reservation_seconds_lower']<=x['worker_device_reservation_seconds_upper']for x in costs4))
ok('no_real_GPU_query_or_Torch_import',gpu_calls==0 and 'torch'not in sys.modules and not any(x.startswith('transformers')for x in sys.modules))
print(json.dumps({'status':'PASS_CPU_SYNTHETIC_CONTROLLER_TESTS_NOT_GPU_QUALIFICATION','checks':checks,'checks_passed':len(checks),'GPU_queries':0,'GPU_dispatches':0,'real_Torch_imports':False,'synthetic_YES_records_only_in_deleted_temporary_fake_roots':True,'controller_sha256':hashlib.sha256(args.controller.read_bytes()).hexdigest()},sort_keys=True,indent=2))
