"""Synthetic control tests. No torch imports, GPU commands, process kills or dispatch."""
import hashlib,json,os,sys,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import run_cublas_controller as ctl
from cublas_harness import build_stage_schedule
checks=[]
def check(name,fn):fn();checks.append({'name':name,'status':'PASS'})
def require(ok):
 if not ok:raise AssertionError('test requirement failed')
def rejects(fn):
 try:fn()
 except (RuntimeError,FileExistsError,ValueError):return
 raise AssertionError('expected rejection')
# Any accidental actual probe/dispatch/signal causes an immediate test failure.
def forbidden(*a,**k):raise AssertionError('No real GPU/process operation allowed in CPU tests')
ctl.subprocess.check_output=forbidden;ctl.subprocess.Popen=forbidden;ctl.os.killpg=forbidden
check('defaultNO_stops_before_authfile_and_GPU',lambda:rejects(lambda:ctl.admission(R,Path('/not/read'),'NO')))
check('B0_env_before_process',lambda:require(ctl.unit_environment({'condition':'B0'})['CUBLAS_WORKSPACE_CONFIG']==':4096:8'))
check('B1_env_before_process',lambda:require(ctl.unit_environment({'condition':'B1'})['CUBLAS_WORKSPACE_CONFIG']==':16:8'))
check('parent_environment_unchanged',lambda:require(os.environ.get('ALLOW_GPU_DIAG')!='YES'))
check('child_environment_only_explicit_run',lambda:require(ctl.unit_environment({'condition':'B0'})['ALLOW_GPU_DIAG']=='YES'))
check('budget_rejects_pending_group',lambda:require(not ctl.enough_time(0.,1000.,720.,now=1000.)))
check('wall_budget_even_when_occupiedlow',lambda:require(not ctl.enough_time(0.,100.,180.,now=1500.)))
check('exact1650_admitted_then_cleanupreserve',lambda:require(ctl.enough_time(0.,1000.,650.,now=1000.)))
check('over1650_rejected',lambda:require(not ctl.enough_time(0.,1000.,650.001,now=1000.)))
check('plan_allowlist_equals_worker',lambda:require(json.loads((R/'qualification/CUBLAS_PREREGISTRATION.json').read_text())['units']==build_stage_schedule()))
check('mandatoryallowances1260',lambda:require(sum(x['allowance_seconds'] for x in build_stage_schedule() if not x['optional'])==1260))
with tempfile.TemporaryDirectory(prefix='qwen_fast_control_fixture_') as td:
 t=Path(td);(t/'qualification').mkdir();(t/'analysis').mkdir()
 (t/'SOURCE_MANIFEST.json').write_text(json.dumps({'task_id':ctl.TASK,'execution_sources':[],'external_dependencies':[]}))
 (t/'qualification/CUBLAS_PREREGISTRATION.json').write_text(json.dumps({'units':build_stage_schedule(),'future_execution_root':str(t)}));(t/'analysis/STRICT_BASELINE_CONTRACT.json').write_text('{}')
 auth=dict(task_id=ctl.TASK,status='PASS',ALLOW_GPU_DIAG='YES',root=str(t),source_manifest_sha256=ctl.sha(t/'SOURCE_MANIFEST.json'),contract_sha256=ctl.sha(t/'analysis/STRICT_BASELINE_CONTRACT.json'),preregistration_sha256=ctl.sha(t/'qualification/CUBLAS_PREREGISTRATION.json'),device={'physical_index':0,'uuid':ctl.UUID},units=build_stage_schedule())
 ctl.once(t/'RUN_COPY_ORIGIN.json',{'task_id':ctl.TASK,'source_manifest_sha256':ctl.sha(t/'SOURCE_MANIFEST.json')})
 ap=t/'auth.json';ap.write_text(json.dumps(auth))
 check('bound_synthetic_authorization_admitted_without_probe',lambda:require(ctl.admission(t,ap,'YES')[0]['task_id']==ctl.TASK))
 ap.write_text(json.dumps(dict(auth,contract_sha256='bad')))
 check('tampered_contract_rejected',lambda:rejects(lambda:ctl.admission(t,ap,'YES')))
 ap.write_text(json.dumps(dict(auth,units=auth['units'][:-1])))
 check('partial_or_changed_allowlist_rejected',lambda:rejects(lambda:ctl.admission(t,ap,'YES')))
 ap.write_text(json.dumps(auth));(t/'FIRST_FAILURE.json').write_text('{}')
 check('failed_task_no_retry',lambda:rejects(lambda:ctl.admission(t,ap,'YES')));(t/'FIRST_FAILURE.json').unlink()
 ctl.once(t/'first.json',{'t':1})
 check('first_receipt_not_overwritten',lambda:rejects(lambda:ctl.once(t/'first.json',{'t':2})))
 check('original_first_receipt_retained',lambda:require(json.loads((t/'first.json').read_text())['t']==1))
 # Ledger gets actual unknown/bounds as soon as a dispatch event exists, never zero.
 (t/'qualification/resource_event_reducer.py').write_bytes((R/'qualification/resource_event_reducer.py').read_bytes())
 ctl.append_event(t,dict(unit_id='SYNTHETIC',GPU_UUID=ctl.UUID,kind='dispatch_before_spawn'))
 ctl.update_resource_ledger(t);ledger=json.loads((t/'RESOURCE_LEDGER.json').read_text())
 check('incomplete_dispatch_not_reported_zero',lambda:require(ledger['NEW_GPUh']=='UNRESOLVED' and ledger['GPU_dispatches']==1 and ledger['observed_dispatch_reservation_GPUh_upper']>0))
check('worker_commands_no_unaccepted_repeat_argument',lambda:require('--repeat' not in ctl.command(R,build_stage_schedule()[0],R/'fake-auth.json')))
check('CPU_import_did_not_load_engine',lambda:require('torch' not in sys.modules and not any(n.startswith('transformers') for n in sys.modules)))
report={'status':'PASS_CPU_CONTROL_TESTS_NOT_GPU_QUALIFICATION','GPU_queries':0,'GPU_dispatches':0,'checks':checks,'controller_sha256':hashlib.sha256((R/'scripts/run_cublas_controller.py').read_bytes()).hexdigest()}
(R/'qualification/CONTROLLER_CPU_TEST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'GPU_dispatches':0}))
