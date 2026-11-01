"""Run the complete unchanged repaired controller on a CPU sandbox.

Only environment/admission/NVML and the worker executable are substitutes. The
actual controller launches a real fresh CPU child that exits1, records/releases
it, closes the shared branch, and must still publish its honest0-result SESSION.
No model, Torch, CUDA context or real experiment directory is used.
"""
from pathlib import Path
import ast,sys,json,tempfile,subprocess,os
root=Path(sys.argv[1])
tree=ast.parse((root/'src/execution_control.py').read_text())
helpers='\n\n'.join(ast.unparse(x) for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in {'canonical','objsha','status','event'})
guards=root/'inherited/group32/src/inherited_device_guards.py'
with tempfile.TemporaryDirectory(prefix='oscar_worker_failure_') as tmp:
 r=Path(tmp)
 for d in ['src','scripts','analysis','branches','execution','formal']:(r/d).mkdir()
 def put(n,x):(r/n).write_text(json.dumps(x))
 put('GPU_EXECUTION_PLAN.json',dict(qualification_units=[dict(id='CPU_ONLY_SHARED_WORKER_FAIL',kind='codec_math',gpu=1)]))
 put('FORMAL_JOB_MANIFEST.json',dict(jobs=[]))
 put('SOURCE_FREEZE.json',dict(count=0,files=[]))
 put('GPU_PERMIT.json',dict(CPU_SANDBOX=True))
 put('P1_EXECUTABLE_IDENTITY.json',dict(files=[]))
 put('branches/U_CPU_ELIGIBILITY.json',dict(eligible=True))
 (r/'scripts/gpu_unit_worker.py').write_text("import os;assert os.environ['CUDA_VISIBLE_DEVICES']=='';raise SystemExit(1)\n")
 (r/'src/record_validation.py').write_text('def validate(*a): raise AssertionError("No formal sample may run in failure test")\n')
 fake='''import sys,os,json,time,hashlib,threading
from pathlib import Path
sys.path.insert(0,GUARD_DIR)
from inherited_device_guards import require,sha,read,atomic_new,process_identity,same_process
TASK='CPU_FAILURE_TEST';SELECTED_GPUS=(1,2,3);UUIDS=('FORBIDDEN','CPU_UUID1','CPU_UUID2','CPU_UUID3')
'''.replace('GUARD_DIR',repr(str(guards.parent)))+helpers+'''
def authorize(*args):return None
def validate_jobs(*args):return True
def device(*args):return dict(CPU_SANDBOX=True)
def model_guard(*args):return None
def storage_guard(*args):return None
def admit_inherited(*args):return dict(status='CPU_SANDBOX_NO_MODEL')
def verify_external(*args):return 61
def verify_sources(root):return sha(root/'SOURCE_FREEZE.json')
def reference_root():return Path(REF_PATH)
def query_gpu():return dict(apps=[],devices=[])
def worker_environment(*args):return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
'''.replace('REF_PATH',repr(str(r/'ram')))
 (r/'src/execution_control.py').write_text(fake)
 wrapper=r/'run_controller_cpu.py'
 wrapper.write_text('''import sys,os,types
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.statvfs=lambda p:types.SimpleNamespace(f_bavail=125*1024**3,f_frsize=1)
sys.argv=[CONTROLLER,'--root',SANDBOX,'--execute']
exec(compile(Path(CONTROLLER).read_bytes(),CONTROLLER,'exec'),dict(__name__='__main__',__file__=CONTROLLER))
'''.replace('CONTROLLER',repr(str(root/'scripts/gpu_controller.py'))).replace('SANDBOX',repr(str(r))))
 p=subprocess.run([sys.executable,'-u','-B',str(wrapper)],capture_output=True,text=True,timeout=30,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 assert p.returncode==0,(p.returncode,p.stdout,p.stderr)
 session=json.loads((r/'SESSION_RESULT.json').read_text());progress=json.loads((r/'FORMAL_PROGRESS.json').read_text());ledger=json.loads((r/'RESOURCE_LEDGER.json').read_text());failure=json.loads((r/'FIRST_FAILURE.json').read_text());events=[json.loads(l) for l in (r/'GPU_RESOURCE_LEDGER.jsonl').read_text().splitlines()]
 assert session['counts']==dict(P1=0,C1=0,S3=0) and session['status']=='PARTIAL_RESULTS_PRESERVED'
 assert len(ledger['intervals'])==1 and ledger['intervals'][0]['exit_returncode']==1 and ledger['intervals'][0]['release']=='PASS'
 assert not any(progress['branch_alive'].values()) and not progress['active_own_identities']
 assert any(x['event']=='BRANCH_FAIL_CLOSED' for x in events) and 'Worker failed' in failure['error']
 assert not list((r/'formal').glob('*/DISPATCH.json'))
 print(json.dumps(dict(status='PASS_ACTUAL_CONTROLLER_FAILED_CPU_CHILD_STILL_PUBLISHES_HONEST_SESSION',real_CPU_child_exit=1,observed_intervals=1,own_release='PASS',formal_counts=session['counts'],original_failure_recorded=True,error_handler_crash=False,SESSION_RESULT_written=True,model_loads=0,GPU_forwards=0),sort_keys=True))
