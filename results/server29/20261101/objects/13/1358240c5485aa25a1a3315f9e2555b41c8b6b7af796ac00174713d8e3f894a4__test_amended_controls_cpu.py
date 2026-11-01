"""CPU-only control tests. All GPU/process APIs are replaced by fakes."""
from pathlib import Path
import ast,contextlib,hashlib,io,json,os,sys,tempfile,types
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import run_cublas_controller as C
import cublas_worker as W
import cublas_harness as H
checks=[]
def ck(n,v):
 if not v:raise AssertionError(n)
 checks.append(dict(name=n,status='PASS'))
def deny(n,f):
 try:f()
 except (RuntimeError,ValueError,FileNotFoundError):ck(n,True)
 else:ck(n,False)
ck('CPU_imports_did_not_load_engine','torch'not in sys.modules and not any(n.startswith('transformers')for n in sys.modules))
units=H.build_stage_schedule();ck('finite16_unique_units',len(units)==len({u['unit_id']for u in units})==16)
ck('all_nominal_estimates_nonbinding',all('estimated_seconds'in u and 'allowance_seconds'not in u for u in units))
core=(R.parent/'QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/evidence/28f10b0f2a7e75dc606b38f63d526a91e4b220d6af98a28e536dfef4f10ebe8f/core.py').read_text()
oldenv=dict(os.environ)
for card in range(4):
 for condition in ['B0','B1']:
  device=dict(physical_index=card,uuid='GPU-SYNTHETIC-'+str(card));env=H.fresh_environment(condition,{},device)
  ck('physical_environment_'+str(card)+condition,env['CUDA_VISIBLE_DEVICES']==str(card)and env['CUBLAS_WORKSPACE_CONFIG']==H.WORKSPACES[condition])
  tree=W.transform_core(core,H.WORKSPACES[condition],card);ck('reverse_AST_'+str(card)+condition,isinstance(tree,ast.Module))
  calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and ast.unparse(n.func)=='torch.cuda.set_device'];ck('logical_cuda0_unchanged_'+str(card)+condition,len(calls)==1 and calls[0].args[0].value==0)
ck('parent_environment_unchanged',dict(os.environ)==oldenv)
# The qualified numerical/codec path differs from sealed worker only by progress recording.
sealed=ast.parse((R.parent/'QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1/scripts/cublas_worker.py').read_text());new=ast.parse((R/'scripts/cublas_worker.py').read_text())
def fn(tree,name):return next(x for x in tree.body if getattr(x,'name',None)==name)
class StripProgress(ast.NodeTransformer):
 def visit_Expr(self,n):
  if isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='mark_progress':return None
  return self.generic_visit(n)
ck('qualify_math_and_order_AST_unchanged',ast.dump(fn(sealed,'qualify'),include_attributes=False)==ast.dump(StripProgress().visit(fn(new,'qualify')),include_attributes=False))
ck('DeepRecorder_AST_unchanged',ast.dump(fn(sealed,'DeepRecorder'),include_attributes=False)==ast.dump(fn(new,'DeepRecorder'),include_attributes=False))
ck('no_worker_signal_alarm',not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in ['alarm','setitimer']for n in ast.walk(new)))
ck('controller_no_elapsed_budget_function',not any(getattr(n,'name',None)=='enough_time'for n in ast.parse((R/'scripts/run_cublas_controller.py').read_text()).body))
q=R/'qualification/source_cpu_synthetic';q.mkdir(exist_ok=True);shared=Path(tempfile.mkdtemp(prefix='cards_',dir=q));roots={f'gpu{i}':str(shared/f'gpu{i}')for i in range(4)};root=Path(roots['gpu2']);root.mkdir();(root/'scripts').mkdir();(root/'analysis').mkdir();(root/'qualification').mkdir();(root/'execution').mkdir()
for n in ['HUMAN_GPU_AUTHORIZATION.json','GPU_EXECUTION_AMENDMENT.json']: (root/n).write_text('{}')
(root/'ARCHIVE_CONFIG.json').write_text(json.dumps(dict(card_key='gpu2')));(root/'analysis/STRICT_BASELINE_CONTRACT.json').write_text('{}');dummy=root/'scripts/dummy.txt';dummy.write_text('CPU synthetic source only')
manifest=dict(task_id=C.TASK,execution_sources=[dict(path='scripts/dummy.txt',bytes=dummy.stat().st_size,sha256=C.sha(dummy))],external_dependencies=[dict(path=str(dummy),bytes=dummy.stat().st_size,sha256=C.sha(dummy))]);(root/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest));(root/'RUN_COPY_ORIGIN.json').write_text(json.dumps(dict(task_id=C.TASK,source_manifest_sha256=C.sha(root/'SOURCE_MANIFEST.json'))))
device=dict(physical_index=2,uuid='GPU-SYNTHETIC-2');plan=dict(future_execution_root=str(root),device=device,no_GPU_time_limit=True,progress_watchdog_seconds=1800,units=units,shared_control_root=str(shared),card_roots=roots);pp=root/'qualification/CUBLAS_PREREGISTRATION.json';pp.write_text(json.dumps(plan));auth=dict(task_id=C.TASK,execution_run_id=C.RUN,status='PASS',ALLOW_GPU_DIAG='YES',root=str(root),source_manifest_sha256=C.sha(root/'SOURCE_MANIFEST.json'),contract_sha256=C.sha(root/'analysis/STRICT_BASELINE_CONTRACT.json'),preregistration_sha256=C.sha(pp),archive_config_sha256=C.sha(root/'ARCHIVE_CONFIG.json'),human_authorization_sha256=C.sha(root/'HUMAN_GPU_AUTHORIZATION.json'),amendment_sha256=C.sha(root/'GPU_EXECUTION_AMENDMENT.json'),run_copy_origin_sha256=C.sha(root/'RUN_COPY_ORIGIN.json'),device=device,no_GPU_time_limit=True,units=units,shared_control_root=str(shared),card_roots=roots);ap=root/'AUTH.json';ap.write_text(json.dumps(auth))
ck('amended_authorization_admitted_without_probe',C.admission(root,ap,'YES')[0]==auth)
deny('defaultNO_denied',lambda:C.admission(root,ap,'NO'))
oldGPU,oldDevice=C.gpu_processes,C.device_rows
try:
 C.gpu_processes=lambda:(_ for _ in()).throw(AssertionError('GPU query reached'));C.device_rows=C.gpu_processes
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):rc=C.main(['--root',str(root),'--run','--authorization',str(ap)])
 ck('unauthorized_defaultNO_no_gpu_no_mutation',rc==1 and not(root/'FIRST_FAILURE.json').exists())
finally:C.gpu_processes,C.device_rows=oldGPU,oldDevice
# Exact root/parent/device receipts are checked using injected CPU-only reader.
unit=units[0];target=root/'execution'/unit['unit_id'];target.mkdir();argv=['/fake/python','-B',str(root/'scripts/run_cublas_controller.py'),'--root',str(root),'--run'];meta=dict(controller_pid=900,controller_starttick=123,controller_cmdline=argv,device=device,authorization_sha256=C.sha(ap),source_manifest_sha256=auth['source_manifest_sha256']);(root/'execution/CONTROLLER_STARTED.json').write_text(json.dumps(meta));dp=target/'DISPATCH.json';dispatch=dict(unit=unit,device=device,controller_pid=900,controller_starttick=123,authorization_sha256=C.sha(ap),source_manifest_sha256=auth['source_manifest_sha256'],progress_policy=dict(no_GPU_time_limit=True,watchdog_seconds=1800));dp.write_text(json.dumps(dispatch));before=os.environ.get('QWEN_CONTROLLER_DISPATCH_SHA256');os.environ['QWEN_CONTROLLER_DISPATCH_SHA256']=C.sha(dp)
try:
 ck('exact_parent_device_dispatch_admitted',W.controller_admission(root,ap,auth,unit,plan,reader=lambda p:(123,argv),parent_pid=900)==dispatch)
 deny('reused_parent_PID_denied',lambda:W.controller_admission(root,ap,auth,unit,plan,reader=lambda p:(124,argv),parent_pid=900))
finally:
 if before is None:os.environ.pop('QWEN_CONTROLLER_DISPATCH_SHA256',None)
 else:os.environ['QWEN_CONTROLLER_DISPATCH_SHA256']=before
# Committed progress only; stdout/heartbeat does not count.
progress=target/'PROGRESS.jsonl';progress.write_text(json.dumps(dict(progress_kind='phase_transition',sequence=1))+'\n'+json.dumps(dict(progress_kind='heartbeat',sequence=999))+'\n');ck('heartbeat_does_not_reset_watchdog',C.progress_cursor(target)==1)
with progress.open('a')as f:f.write(json.dumps(dict(progress_kind='bytes_hashed',sequence=2))+'\n')
ck('committed_hash_progress_counts',C.progress_cursor(target)==2)
# Synthetic long-progressing unit exceeds the old wall cap without any GPU operation.
saved={k:getattr(C,k)for k in ['time','subprocess','gpu_processes','device_rows','proc_identity','progress_cursor','update_resource_ledger','cleanup_own']}
class Clock:
 def __init__(self):self.t=0
 def monotonic(self):return self.t
 def time(self):return 1792400000+self.t
 def sleep(self,n):self.t+=900
class FakeProcess:
 pid=123456789
 def __init__(self):self.n=0;self.returncode=None
 def poll(self):
  self.n+=1
  if self.n>=6:self.returncode=0
  return self.returncode
clock=Clock();box={'proc':None,'cleanup':0,'ledger':0}
def fake_spawn(*args,**kwargs):box['proc']=FakeProcess();return box['proc']
u=next(x for x in units if x['unit_id']=='B0_FP_P0');ut=root/'execution'/u['unit_id'];ut.mkdir();(ut/'RESULT.json').write_text(json.dumps(dict(status='PASS',synthetic_CPU_fixture_only=True)))
try:
 C.time=clock;C.subprocess=types.SimpleNamespace(Popen=fake_spawn,TimeoutExpired=saved['subprocess'].TimeoutExpired,STDOUT=saved['subprocess'].STDOUT)
 C.gpu_processes=lambda:[]if box['proc']is None or box['proc'].returncode is not None else [(box['proc'].pid,device['uuid'])]
 C.device_rows=lambda:[dict(index=2,uuid=device['uuid'],name='RTX3090',free_MiB=24000)]
 C.proc_identity=lambda pid:dict(starttick=123,cmdline=argv)
 C.progress_cursor=lambda p:box['proc'].n
 C.update_resource_ledger=lambda p:box.__setitem__('ledger',box['ledger']+1)
 C.cleanup_own=lambda *a:box.__setitem__('cleanup',box['cleanup']+1)
 elapsed=C.run_unit(root,u,ap,auth,plan)
 ck('progressing_unit_over_old1800_seconds_not_killed',elapsed>1800 and box['cleanup']==0)
 ev=[json.loads(x)for x in(root/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines()]
 ck('first_exit_recorded_once',sum(x['kind']=='first_process_exit_observed'for x in ev)==1)
 ck('first_release_recorded_once',sum(x['kind']=='first_exit_and_release_observed'for x in ev)==1)
 ck('ledger_refreshed_during_long_unit',box['ledger']>2)
finally:
 for k,v in saved.items():setattr(C,k,v)
# A stalled committed cursor, unlike long elapsed time with progress, does stop.
stallshared=Path(tempfile.mkdtemp(prefix='stall_',dir=q));stallroot=stallshared/'gpu2';(stallroot/'execution').mkdir(parents=True);(stallroot/'SOURCE_MANIFEST.json').write_text('{}');sap=stallroot/'AUTH.json';sap.write_text('{}');sauth=dict(auth,shared_control_root=str(stallshared));su=next(x for x in units if x['unit_id']=='B0_FP_P1');clock=Clock();box={'proc':None,'cleanup':0,'ledger':0}
saved={k:getattr(C,k)for k in ['time','subprocess','gpu_processes','device_rows','proc_identity','progress_cursor','update_resource_ledger','cleanup_own']}
def stall_cleanup(proc,*a):box['cleanup']+=1;proc.returncode=-15
try:
 C.time=clock;C.subprocess=types.SimpleNamespace(Popen=fake_spawn,TimeoutExpired=saved['subprocess'].TimeoutExpired,STDOUT=saved['subprocess'].STDOUT)
 C.gpu_processes=lambda:[]if box['proc']is None or box['proc'].returncode is not None else [(box['proc'].pid,device['uuid'])]
 C.device_rows=lambda:[dict(index=2,uuid=device['uuid'],name='RTX3090',free_MiB=24000)]
 C.proc_identity=lambda pid:dict(starttick=123,cmdline=argv);C.progress_cursor=lambda p:0
 C.update_resource_ledger=lambda p:box.__setitem__('ledger',box['ledger']+1);C.cleanup_own=stall_cleanup
 try:C.run_unit(stallroot,su,sap,sauth,plan)
 except RuntimeError as e:ck('no_committed_progress_watchdog_failsclosed','no committed' in str(e)and box['cleanup']==1)
 else:ck('no_committed_progress_watchdog_failsclosed',False)
 ck('watchdog_publishes_shared_STOP',(stallshared/'FIRST_FAILURE.json').is_file())
 ev=[json.loads(x)for x in(stallroot/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines()]
 ck('watchdog_exit_release_each_once',sum(x['kind']=='first_process_exit_observed'for x in ev)==sum(x['kind']=='first_exit_and_release_observed'for x in ev)==1)
finally:
 for k,v in saved.items():setattr(C,k,v)
# Root failure publication is atomic first-writer and stops sibling admissions.
C.publish_failure(root,auth,'synthetic first failure');old=(shared/'FIRST_FAILURE.json').read_bytes();C.publish_failure(root,auth,'synthetic later failure');ck('global_first_failure_not_overwritten',(shared/'FIRST_FAILURE.json').read_bytes()==old);deny('shared_failure_stops_sibling',lambda:C.shared_check(auth))
# Own-process cleanup must refuse PID reuse; all signals are fake.
savedOS,savedProc=C.os,C.proc_identity;signals=[]
class Proc:
 pid=777
 def __init__(self):self.returncode=None
 def poll(self):return self.returncode
 def wait(self,timeout):self.returncode=0
try:
 C.os=types.SimpleNamespace(killpg=lambda p,s:signals.append((p,s)));C.proc_identity=lambda pid:dict(starttick=124,cmdline=['worker','unit'])
 deny('PID_reuse_never_signalled',lambda:C.cleanup_own(Proc(),dict(starttick=123),'unit',Path('worker')));ck('no_foreign_signal',signals==[])
 C.proc_identity=lambda pid:dict(starttick=123,cmdline=['worker','unit']);C.cleanup_own(Proc(),dict(starttick=123),'unit',Path('worker'));ck('only_own_process_group_signal',len(signals)==1 and signals[0][0]==777)
finally:C.os,C.proc_identity=savedOS,savedProc
ck('tests_never_imported_GPU_engine','torch'not in sys.modules)
out=dict(status='CPU_PASS_NOT_GPU_QUALIFICATION',GPU_queries=0,GPU_dispatches=0,checks=checks,no_GPU_time_limit=True,only_progress_watchdog_seconds=1800,source_sha256={n:C.sha(R/'scripts'/n)for n in ['run_cublas_controller.py','cublas_worker.py','cublas_harness.py']});(R/'qualification/AMENDED_SOURCE_CPU_TESTS.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],len(checks),'checks')
