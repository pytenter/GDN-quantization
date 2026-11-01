#!/usr/bin/env python3
"""Deleted temporary CPU fixtures only. No physical GPU/runtime imports."""
import ast,copy,hashlib,importlib.util,json,os,sys,tempfile,types
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
RUN=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('amended_controller_CPU',RUN/'drafts/controller.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
checks=[]
def ok(name,value=True):assert value,name;checks.append(dict(name=name,status='PASS'))
def reject(name,fn):
 try:fn()
 except RuntimeError as exc:ok(name);return exc
 raise AssertionError(name+' unexpectedly admitted')
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True))
def describe(p,name):return dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def fixture(top):
 root=(top/'payload').resolve();root.mkdir(parents=True);external=top/'external';external.mkdir();base=json.loads((RUN/'drafts/GPU_PLAN.json').read_bytes());(root/'USER_REQUEST.txt').write_text('SYNTHETIC NEW RECOVERY REQUEST, NO ACTUAL AUTHORIZATION');base['request_sha256']=C.sha(root/'USER_REQUEST.txt');base['binding_files']=['analysis/bindings.json'];weights=[]
 for i in range(4):
  p=external/('model'+str(i)+'.safetensors');p.write_bytes(bytes([i]));weights.append(describe(p,str(p)))
 save(root/'analysis/bindings.json',dict(model_weight_shards=weights));resource=json.loads((RUN/'HUMAN_GPU_RESOURCE_AUTHORIZATION.json').read_bytes());resource['task_request_sha256']=base['request_sha256'];save(root/'HUMAN_GPU_RESOURCE_AUTHORIZATION.json',resource);base['resource_authorization_sha256']=C.sha(root/'HUMAN_GPU_RESOURCE_AUTHORIZATION.json')
 for rel in ['scripts/restoration_controller.py','worker/restoration_worker.py']:
  p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# SYNTHETIC CPU NONEXECUTABLE SOURCE\n')
 save(root/'qualification/GPU_PLAN.json',base);inherited=['analysis/bindings.json']
 for i in range(402):
  name='base/source_'+str(i)+'.py';p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# inherited\n');inherited.append(name)
 current=[describe(root/name,name)for name in sorted(inherited+list(C.REPLACED))];assert len(current)==407
 parent=[dict(row,bytes=1,sha256='0'*64)if row['path']in C.REPLACED else row for row in current];save(root/'provenance/PARENT_EFFECTIVE_SOURCE_FREEZE.json',dict(files=parent));save(root/'SOURCE_FREEZE.json',dict(task_id=C.TASK,files=current));save(root/'ROOT.json',dict(task_id=C.TASK,ALLOW_GPU_RESTORE='NO'));auth=dict(task_id=C.TASK,scope=C.SCOPE,ALLOW_GPU_RESTORE='YES',device_uuid=C.UUID,available_authorized_GPU_count=4,max_active_GPUs=1,max_GPU_hours=None,hard_seconds=None,dispatch_max_seconds=None,allowed_units=[x['id']for x in C.UNITS],execution_root=str(root),human_request_quote=resource['human_GPU_authorization_quote'],resource_amendment_quote=resource['latest_resource_amendment_quote'],plan_sha256=C.sha(root/'qualification/GPU_PLAN.json'),source_freeze_sha256=C.sha(root/'SOURCE_FREEZE.json'),parent_effective_source_freeze_sha256=C.PARENT_FREEZE,request_sha256=base['request_sha256'],resource_authorization_sha256=base['resource_authorization_sha256'],synthetic_deleted_CPU_fixture_not_authorization=True);save(root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',auth);return root,base,auth
real_sha=C.sha
def fake_parent_sha(path,progress=None):
 if Path(path).name=='PARENT_EFFECTIVE_SOURCE_FREEZE.json':return C.PARENT_FREEZE
 return real_sha(path,progress)
def no_gpu():raise AssertionError('Physical GPU query forbidden')
with tempfile.TemporaryDirectory(prefix='AMENDMENT_CONTROLLER_CPU_ONLY_')as tmp, mock.patch.object(C,'sha',side_effect=fake_parent_sha):
 top=Path(tmp);root,plan,auth=fixture(top/'one');ap=root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json'
 with mock.patch.object(C,'check_inputs',side_effect=AssertionError('ROOT NO must stop before external hash')):reject('ROOT_NO_denies_before_model_hash_or_GPU',lambda:C.run_controller(root,ap,'YES',gpu_reader=no_gpu))
 save(root/'ROOT.json',dict(task_id=C.TASK,ALLOW_GPU_RESTORE='YES'));checked=C.authorize(root,'YES',ap);ok('new_task_full_authorization_parent407_resourceSHA_PASS',all(checked[k]==auth[k]for k in C.IDENTITIES[:-1])and checked['source_files']==407 and checked['inherited_files']==403 and checked['replaced_files']==4)
 for key,value in [('task_id','OLD_TASK'),('scope','old_scope'),('execution_root','/wrong'),('device_uuid','GPU-other'),('max_GPU_hours',.5),('hard_seconds',1800),('dispatch_max_seconds',240),('resource_authorization_sha256','0'*64),('parent_effective_source_freeze_sha256','0'*64),('human_request_quote','invented'),('resource_amendment_quote','invented')]:
  bad=dict(auth);bad[key]=value;save(ap,bad);reject('auth_'+key+'_mismatch_denies_before_GPU',lambda:C.authorize(root,'YES',ap))
 save(ap,auth)
 for key in ['hard_seconds','protective_seconds','dispatch_max_seconds','max_GPU_hours']:
  bad=copy.deepcopy(plan);bad[key]=123;reject('plan_cannot_reintroduce_elapsed_cap_'+key,lambda bad=bad:C.validate_plan(bad))
 for key,value in [('no_progress_watchdog_seconds',1801),('max_active_GPUs',4),('available_authorized_GPU_count',1),('numeric_tokens',31),('warmup_tokens',127)]:
  bad=copy.deepcopy(plan);bad[key]=value;reject('plan_fixed_finite_contract_'+key,lambda bad=bad:C.validate_plan(bad))
 bad=copy.deepcopy(plan);bad['units']=bad['units']+[dict(id='EXTRA',mode='benchmark',optional=True)];reject('four_available_GPUs_do_not_expand_science_roster',lambda:C.validate_plan(bad));bad=copy.deepcopy(plan);bad['math_gate']['state_atol']*=2;reject('numeric_thresholds_cannot_change',lambda:C.validate_plan(bad))
 frozen=C.read(root/'SOURCE_FREEZE.json');target=root/'base/source_0.py';before=target.read_bytes();target.write_text('# changed inherited math\n');changed=copy.deepcopy(frozen)
 for row in changed['files']:
  if row['path']=='base/source_0.py':row.update(describe(target,row['path']))
 save(root/'SOURCE_FREEZE.json',changed);reject('even_rehashed_current_manifest_cannot_modify_inherited403',lambda:C.check_inputs(root));target.write_bytes(before);save(root/'SOURCE_FREEZE.json',frozen)
 # The real helper validates a genuine controller-emitted-shaped new-task
 # permit including both full authorization/resource hashes; only processes and
 # SHA-progress sink are faked. No permit is created outside this temp root.
 meta=C.authorize(root,'YES',ap);unit=C.UNITS[0];phase0=root/'execution'/unit['id'];phase0.mkdir(parents=True);pp0=phase0/'PERMIT.json';cmd0=[sys.executable,'-B',str(root/'worker/restoration_worker.py'),'--root',str(root),'--mode','numeric','--unit',unit['id'],'--permit',str(pp0)];parent0=dict(pid=707,starttick=88,cmdline=[sys.executable,str(root/'scripts/restoration_controller.py'),'--root',str(root),'--run']);own0=dict(pid=os.getpid(),starttick=99,cmdline=cmd0);permit0=dict(task_id=C.TASK,scope=C.SCOPE,root=str(root),unit=unit,device=plan['device'],controller_identity=parent0,worker_command=cmd0,environment_controls={k:C.worker_environment(unit).get(k)for k in C.ENV_KEYS},plan_relative_path='qualification/GPU_PLAN.json',dispatch_monotonic=0,dispatch_elapsed_hard_limit_seconds=None,no_progress_watchdog_seconds=1800,**{k:meta[k]for k in C.IDENTITIES});save(pp0,permit0);save(root/'execution/CONTROLLER_STARTED.json',dict(controller_identity=parent0,**{k:meta[k]for k in C.IDENTITIES}));save(phase0/'DISPATCH.json',dict(worker_identity=own0,permit_sha256=real_sha(pp0)));facts=[]
 def permit_reader(pid):return parent0 if pid==707 else own0
 with mock.patch.dict(C.os.environ,C.worker_environment(unit),clear=True),mock.patch.object(C,'emit_progress',side_effect=lambda root,unit,phase,details:facts.append((phase,details))):
  accepted=C.require_worker_permit(root,'numeric',unit['id'],pp0,process_reader=permit_reader,parent_pid=707,monotonic=lambda:10);ok('new_worker_permit_binds_all_six_source_request_resource_auth_identifiers',accepted['checked_identity']=={k:meta[k]for k in C.IDENTITIES});ok('worker_permit_real_external_hash_emits_actual_completion_progress',any(x[0]=='BOUND_FILE_HASH_COMPLETE'for x in facts));changed=dict(permit0,resource_authorization_sha256='0'*64);save(pp0,changed);reject('worker_permit_changed_resourceSHA_cannot_start',lambda:C.require_worker_permit(root,'numeric',unit['id'],pp0,process_reader=permit_reader,parent_pid=707,monotonic=lambda:10));save(pp0,permit0);reject('worker_foreign_parent_cannot_start',lambda:C.require_worker_permit(root,'numeric',unit['id'],pp0,process_reader=permit_reader,parent_pid=708,monotonic=lambda:10))
 # Real progress accepts true facts and committed numerical JSON, never
 # an arbitrary timestamp/ledger update or duplicate fact.
 phase=top/'progress';phase.mkdir();identity=dict(pid=7,starttick=8,cmdline=['CPU fixture']);now=[0.];observer=C.ProgressObserver(phase,C.UNITS[0],identity,clock=lambda:now[0]);progress_path=phase/'PROGRESS_EVENTS.jsonl'
 def add_fact(phase_name,details,who=identity):
  fact=dict(phase=phase_name,details=details);row=dict(event='ACTUAL_PROGRESS',task_id=C.TASK,unit_id='NUMERIC_FRESH0',process_identity=who,fact_sha256=hashlib.sha256(C.canonical(fact)).hexdigest(),**fact)
  with progress_path.open('a')as f:f.write(json.dumps(row)+'\n')
 now[0]=100;add_fact('HASH_READ',dict(path='model0',read_bytes=64,sha256='a'));ok('actual_hash_progress_refreshes_watchdog',len(observer.poll())==1 and observer.last_progress==100);now[0]=1700;add_fact('HASH_READ',dict(path='model0',read_bytes=64,sha256='a'));ok('duplicate_fact_heartbeat_does_not_refresh_watchdog',observer.poll()==[]and observer.last_progress==100);save(phase/'IGNORED_MONITOR.json',dict(heartbeat=1700));ok('monitor_JSON_is_not_scientific_progress',observer.poll()==[]and observer.last_progress==100);save(phase/'TOKEN_000.json',dict(token=0,cursor=1,actual_sha='abc'));ok('committed_token_JSON_advances_watchdog',len(observer.poll())==1 and observer.last_progress==1700);now[0]=1800;os.utime(phase/'TOKEN_000.json',None);ok('touch_without_content_change_does_not_refresh',observer.poll()==[]and observer.last_progress==1700);save(phase/'TOKEN_001.json',dict(token=1));(phase/'TOKEN_001.json').write_text('{partial');ok('partial_JSON_write_does_not_fake_commit_or_STOP',observer.poll()==[]);(phase/'worker.log').write_text(json.dumps(dict(event='PROGRESS',unit='NUMERIC_FRESH0',consumed=128,elapsed=3))+'\n');ok('numeric_unit_cannot_fake_128token_benchmark_progress',observer.poll()==[]);benchmark_phase=top/'benchmark_progress';benchmark_phase.mkdir();benchmark_observer=C.ProgressObserver(benchmark_phase,C.UNITS[2],identity,clock=lambda:now[0]);(benchmark_phase/'worker.log').write_text(json.dumps(dict(event='PROGRESS',unit='OLD_FP_R0',consumed=128,elapsed=3))+'\n');ok('legacy_stdout_real_consumed_token_advances_progress',len(benchmark_observer.poll())==1);last=benchmark_observer.last_progress;now[0]=1900
 with(benchmark_phase/'worker.log').open('a')as log:log.write(json.dumps(dict(event='PROGRESS',unit='OLD_FP_R0',consumed=128,elapsed=99))+'\n')
 ok('changed_elapsed_same_consumed_token_not_progress',benchmark_observer.poll()==[]and benchmark_observer.last_progress==last)
 add_fact('TOKEN',dict(consumed=1),dict(identity,pid=99));reject('foreign_progress_writer_FAIL_CLOSED',observer.poll)
 # Unlimited elapsed is tested through six finite fake children, with actual
 # committed phase/token facts throughout; total fake wall exceeds all old caps.
 inventory=dict(devices=[dict(physical_index=0,uuid=C.UUID,name='RTX 3090',memory_total_MiB=24576,memory_free_MiB=24256)],apps=[])
 def schedule_case(name,mode):
  rr,pp,aa=fixture(top/name);save(rr/'ROOT.json',dict(task_id=C.TASK,ALLOW_GPU_RESTORE='YES'));children={};counter=[8000];clockvalue=[0.];signals=[];controller=dict(pid=os.getpid(),starttick=11,cmdline=['CPU controller']);gpu_calls=[0]
  def clock():return clockvalue[0]
  def sleep(seconds):clockvalue[0]+=700 if mode=='progress_long'else 900
  class Child:
   def __init__(self,command,**kw):
    counter[0]+=1;self.pid=counter[0];self.identity=dict(pid=self.pid,starttick=self.pid+1,cmdline=command);self.returncode=None;self.alive=True;self.count=0;self.phase=Path(command[-1]).parent;self.unit=C.read(Path(command[-1]))['unit'];children[self.pid]=self;save(self.phase/'RESULT.json',dict(status='PASS',unit_id=self.unit['id'],mode=self.unit['mode'],process_identity=self.identity,permit_sha256=real_sha(Path(command[-1]))))
   def poll(self):
    if not self.alive:return self.returncode
    self.count+=1
    if mode=='progress_long':
     fact=dict(phase='TOKEN_COMMITTED',details=dict(consumed=self.count));row=dict(event='ACTUAL_PROGRESS',task_id=C.TASK,unit_id=self.unit['id'],process_identity=self.identity,fact_sha256=hashlib.sha256(C.canonical(fact)).hexdigest(),**fact)
     with(self.phase/'PROGRESS_EVENTS.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
     if self.count>=3:self.alive=False;self.returncode=0;return 0
    return None
   def terminate(self):signals.append(('terminate',self.pid));self.alive=False;self.returncode=-15
   def kill(self):signals.append(('kill',self.pid));self.alive=False;self.returncode=-9
  def reader(pid):
   if pid==os.getpid():return controller
   child=children[pid]
   if child.alive:return dict(child.identity,cmdline=['FOREIGN CMD'])if mode=='bad_cmdline'else child.identity
   raise FileNotFoundError('synthetic exit')
  def gpu():
   gpu_calls[0]+=1;obs=copy.deepcopy(inventory)
   for child in children.values():
    if child.alive or mode=='lingering_other_UUID':obs['apps'].append(dict(pid=child.pid,uuid='GPU-other'if mode=='lingering_other_UUID'else C.UUID,memory_MiB=str(100+gpu_calls[0])))
   return obs
  with mock.patch.object(C,'disk_guard',return_value=dict(available_bytes=20*1024**3,numeric_raw_bytes=0)):
   if mode=='progress_long':result=C.run_controller(rr,rr/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=gpu,popen=Child,reader=reader,clock=clock,sleeper=sleep)
   else:reject(name+'_FAIL_CLOSED_no_retry',lambda:C.run_controller(rr,rr/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=gpu,popen=Child,reader=reader,clock=clock,sleeper=sleep));result=C.read(rr/'FIRST_FAILURE.json')
  events=[json.loads(x)for x in(rr/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines()]
  if mode=='progress_long':
   ok('six_finite_units_complete_beyond_old_1800s_cap',result['completed_units']==[x['id']for x in C.UNITS]and result['elapsed_controller_seconds']>1800);ok('each_unit_can_exceed_old240_when_true_progress_continues',all(x['reservation_seconds_upper']>240 for x in events if x['event']=='UNIT_RESERVATION_INTERVAL_BOUNDS'));ok('every_NVML_poll_full_observation_retained_for_sample_peak',sum(x['event']=='NVML_OBSERVATION'for x in events)==gpu_calls[0]and all('observation'in x for x in events if x['event']=='NVML_OBSERVATION'));ok('first_dispatch_exit_all_UUID_release_once_perunit',all(sum(x['event']==event for x in events)==6 for event in ['FIRST_DISPATCH','FIRST_PROCESS_EXIT_OBSERVED','FIRST_ALL_UUID_NVML_RELEASE_OBSERVED']));ok('reservation_interval_never_claims_exact_physical_GPUh',all(x['actual_physical_GPU_hours']=='UNRESOLVED_NOT_ZERO'for x in events if x['event']=='UNIT_RESERVATION_INTERVAL_BOUNDS'))
  elif mode=='idle':ok('1800s_without_real_progress_triggers_watchdog','NO_REAL_PROGRESS_WATCHDOG'in result['error']);ok('idle_exact_owned_child_cleaned',signals and signals[0][0]=='terminate')
  elif mode=='bad_cmdline':ok('unverified_cmdline_never_signalled',not signals)
  elif mode=='lingering_other_UUID':ok('other_UUID_still_own_PID_never_marked_released',result['cleanup']['NVML_release_verified']is False)
  return rr
 done=schedule_case('long','progress_long');schedule_case('idle','idle');schedule_case('wrongcmd','bad_cmdline');schedule_case('otherUUID','lingering_other_UUID');reject('completed_new_controller_cannot_restart',lambda:C.run_controller(done,done/'HUMAN_GPU_RESTORE_AUTHORIZATION.json','YES',gpu_reader=no_gpu))
 # Driver/NVML identity helpers themselves were copied byte/AST exact from the
 # independently reviewed previous amendment; no new device algorithm.
 old=ast.parse((RUN.parents[0]/'execution_payload/scripts/restoration_controller.py').read_text());new=ast.parse((RUN/'drafts/controller.py').read_text());names=['query_gpu','validate_gpu','require_own_gpu_identity','require_driver_ordinal_identity','worker_environment'];oldnodes={x.name:x for x in old.body if isinstance(x,ast.FunctionDef)};newnodes={x.name:x for x in new.body if isinstance(x,ast.FunctionDef)};ok('device_driver_owner_environment_helpers_AST_unchanged',all(ast.dump(oldnodes[n],include_attributes=False)==ast.dump(newnodes[n],include_attributes=False)for n in names));ok('no_GPU_Torch_or_Driver_runtime_imports',not any(x=='torch'or x.startswith('torch.')or x=='transformers'or x.startswith('transformers.')for x in sys.modules))
print(json.dumps(dict(status='PASS_CPU_ONLY_RESOURCE_AMENDMENT_CONTROLLER_SYNTHETIC_TESTS_NOT_GPU_QUALIFICATION',checks_passed=len(checks),checks=checks,GPU_queries=0,Driver_calls=0,GPU_dispatches=0,synthetic_temporary_auth_only=True,parent_SHA_simulated_only_for_temporary_fixture=True,controller_sha256=hashlib.sha256((RUN/'drafts/controller.py').read_bytes()).hexdigest()),indent=2,sort_keys=True))
