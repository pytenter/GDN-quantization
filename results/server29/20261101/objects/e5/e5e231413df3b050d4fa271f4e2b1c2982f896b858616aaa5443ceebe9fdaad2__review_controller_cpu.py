"""Independent CPU-only controller review. All device/process operations mocked."""
from pathlib import Path
import argparse,ast,contextlib,hashlib,importlib.util,io,json,os,sys,tempfile,types
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'));sys.path.insert(0,str(R/'qualification'))
import run_cublas_controller as c
import cublas_harness as h
from resource_event_reducer import reduce_events
checks=[]
def check(n,b):
 if not b:raise AssertionError(n)
 checks.append(dict(name=n,status='PASS'))
p=json.loads((R/'qualification/CUBLAS_PREREGISTRATION.json').read_text())
check('preregistration_allowlist_matches_worker_full_dictionaries',p['units']==h.build_stage_schedule())
for u in p['units']:
 e=c.unit_environment(u)
 check('child_env_'+u['unit_id'],all(e.get(k)==v for k,v in h.fresh_environment(u['condition'],{}).items()) and e.get('ALLOW_GPU_DIAG')=='YES')
 cmds=c.command(R,u,R/'qualification/AUTHORIZATION_TEMPLATE.json');check('no_unaccepted_repeat_'+u['unit_id'],'--repeat' not in cmds)
check('budgetgroup_reserved',not c.enough_time(0,1260,480,now=1200))
check('wallbudget_reserved',not c.enough_time(0,10,480,now=1200))
check('1800minusprotect_is150',p['GPU_hard_seconds']-p['protective_seconds']==p['cleanup_seconds']==150)
check('numeric_first_order',p['mandatory_numeric_units']==['B0_FP_Q0','B0_FP_Q1','B1_FP_Q0','B1_FP_Q1'])
events=[dict(unit_id='synthetic',GPU_UUID='syntheticUUID',monotonic_seconds=t,kind=k) for t,k in [(0.,'dispatch_before_spawn'),(.1,'spawn_observed'),(5.,'busy_observed')]]
v=reduce_events(events,as_of_monotonic=7.)
check('livefailurecostnotzero',v['has_incomplete_units'] and v['GPU_seconds_upper_bound']==7.)
terminal=events+[dict(unit_id='synthetic',GPU_UUID='syntheticUUID',monotonic_seconds=t,kind=k) for t,k in [(5.1,'first_process_exit_observed'),(5.2,'first_exit_and_release_observed')]]
for kind in ['first_process_exit_observed','first_exit_and_release_observed']:
 try:reduce_events(terminal+[dict(unit_id='synthetic',GPU_UUID='syntheticUUID',monotonic_seconds=7.,kind=kind)])
 except ValueError:check('duplicate_'+kind+'_refused',True)
 else:check('duplicate_'+kind+'_refused',False)
# No real kill/proc lookup: mismatch identity must abort before fakekill, matchtargetsownpgid.
real_os,real_ident=c.os,c.proc_identity;signals=[]
class Proc:
 pid=123456789
 def __init__(self):self.returncode=None
 def poll(self):return self.returncode
 def wait(self,timeout):self.returncode=0
try:
 c.os=types.SimpleNamespace(killpg=lambda pid,sig:signals.append((pid,sig)))
 c.proc_identity=lambda pid:dict(starttick='wrong',cmdline='worker synthetic')
 try:c.cleanup_own(Proc(),dict(starttick='correct'),'synthetic',Path('worker'))
 except RuntimeError:check('identitymismatch_no_signal',signals==[])
 else:check('identitymismatch_no_signal',False)
 c.proc_identity=lambda pid:dict(starttick='correct',cmdline='worker synthetic')
 c.cleanup_own(Proc(),dict(starttick='correct'),'synthetic',Path('worker'));check('mockcleanup_exact_own_pgid',len(signals)==1 and signals[0][0]==123456789)
finally:c.os,c.proc_identity=real_os,real_ident
# Default prepare/denial remains devicefree. Temporaryroot is explicitlynewsyntheticfixture.
q=R/'qualification/controller_review_synthetic';q.mkdir(exist_ok=True)
root=Path(tempfile.mkdtemp(prefix='denial_',dir=q));dummy=root/'source_fixture.txt';dummy.write_text('Synthetic CPU source identity only. No engine.')
row=dict(path=dummy.name,bytes=dummy.stat().st_size,sha256=hashlib.sha256(dummy.read_bytes()).hexdigest());external=dict(row,path=str(dummy))
(root/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(task_id=c.TASK,execution_sources=[row],external_dependencies=[external])))
from prepare_run_bundle import prepare
try:prepare(root,root/'analysis/nested_run')
except RuntimeError:check('staging_inside_source_refused',not(root/'analysis/nested_run').exists())
else:check('staging_inside_source_refused',False)
real_gpu,real_dev=c.gpu_processes,c.device_rows
try:
 c.gpu_processes=lambda:(_ for _ in()).throw(AssertionError('device query reached'))
 c.device_rows=lambda:(_ for _ in()).throw(AssertionError('device query reached'))
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):a=c.main(['--root',str(root)])
 check('default_prepare_noGPU',a==0 and not(root/'FIRST_FAILURE.json').exists())
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):a=c.main(['--root',str(root),'--run'])
 denied_mutation=(root/'FIRST_FAILURE.json').exists()
 check('run_withoutauthorization_doesnotqueryGPU',a==1)
finally:c.gpu_processes,c.device_rows=real_gpu,real_dev
files=['scripts/run_cublas_controller.py','scripts/prepare_run_bundle.py','qualification/resource_event_reducer.py','qualification/CUBLAS_PREREGISTRATION.json','scripts/cublas_harness.py']
out=dict(status='PASS_CPU_REVIEW_WITH_OPEN_FINDINGS' if denied_mutation else 'PASS_CPU_REVIEW_NO_DENIAL_MUTATION',GPU_queries=0,GPU_dispatches=0,checks=checks,unauthorized_denial_mutated_synthetic_root=denied_mutation,files=[dict(path=n,bytes=(R/n).stat().st_size,sha256=hashlib.sha256((R/n).read_bytes()).hexdigest())for n in files],not_GPU_runtime_qualification=True)
(R/'analysis/controller_independent_mock_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],len(checks),'checks','denialmutates',denied_mutation)
