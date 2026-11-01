"""Actual production Bash owning waits and late recorder on CPU-only children.

On the target host the actual fixed production Python is used directly. A
Bash function boundary is used only on macOS when that fixed path is absent;
production shell source and filesystem paths are never edited.
Controller/finisher science and root admission are explicit CPU fixture mocks.
"""
import hashlib,json,os,shlex,subprocess,sys,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x)+'\n')
C_SOURCE=r'''
import hashlib,json,os,sys
from pathlib import Path
EXECUTION_ROOT=str(Path(__file__).resolve().parents[1])
def require(v,m):
 if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_new(p,x):
 with Path(p).open('x') as f:json.dump(x,f,sort_keys=True)
def process_identity(pid):
 if sys.platform.startswith('linux'):
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,starttick=int(s[19]),cmdline=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
 return dict(pid=pid,starttick=12345,cmdline=[sys.executable]+sys.argv)
def same_process(x):
 if not sys.platform.startswith('linux'):return False
 try:return process_identity(x['pid'])==x
 except FileNotFoundError:return False
def verify_sources(root):
 for row in read(root/'SOURCE_FREEZE.json')['files']:
  p=root/row['path'];require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'CPU fixture source mutation')
'''
CHILD=r'''
import sys,os,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));import execution_control as C
cfg=C.read(R/'CASE.json');kind=Path(__file__).name
if kind=='gpu_controller.py':
 C.atomic_new(R/'GPU_RUN_STARTED.json',dict(identity=C.process_identity(os.getpid())))
 if cfg['session']:C.atomic_new(R/'SESSION_RESULT.json',dict(CPU_MOCK_ONLY=True))
 raise SystemExit(cfg['controller_rc'])
else:
 C.atomic_new(R/'FINALIZER_STARTED.json',dict(identity=C.process_identity(os.getpid())))
 raise SystemExit(cfg['finisher_rc'])
'''
def main():
 rows=[]
 fixed=Path('/data/ydai/miniconda3/envs/bitdecode/bin/python')
 if fixed.is_file():
  assert fixed.resolve()==Path(sys.executable).resolve(),'Run this target CPU test with the actual fixed production interpreter'
  boundary='REAL_FIXED_PRODUCTION_PYTHON';alias=False
 else:
  assert sys.platform=='darwin','Fixed interpreter absent; Mac-only alias fixture is not permitted here'
  boundary='MAC_ONLY_ALIAS_FUNCTION_FIXED_INTERPRETER_ABSENT';alias=True
 for name,controller,finisher,session,expected in [('success',0,0,True,0),('controller_fail',1,0,True,1),('finisher_fail',0,3,True,3),('no_session',2,None,False,2)]:
  with tempfile.TemporaryDirectory(prefix='wscale-offline-wait-cpu-') as temp:
   f=Path(temp).resolve();(f/'scripts').mkdir();(f/'src').mkdir();(f/'scripts/run_offline.sh').write_bytes((R/'scripts/run_offline.sh').read_bytes());(f/'scripts/record_offline_chain_exit.py').write_bytes((R/'scripts/record_offline_chain_exit.py').read_bytes());(f/'src/execution_control.py').write_text(C_SOURCE)
   for n in ('gpu_controller.py','finalize_offline.py'):(f/'scripts'/n).write_text(CHILD)
   dump(f/'CASE.json',dict(controller_rc=controller,finisher_rc=finisher,session=session,CPU_MOCK_ONLY=True));source_files=[p for p in f.rglob('*') if p.is_file()];dump(f/'SOURCE_FREEZE.json',dict(files=[dict(path=p.relative_to(f).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in source_files]));dump(f/'GPU_PERMIT.json',dict(execution_root=str(f),source_freeze_sha256=sha(f/'SOURCE_FREEZE.json')))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');env.pop('BASH_ENV',None)
   if alias:
    rcfile=f/'bash_env';rcfile.write_text('function /data/ydai/miniconda3/envs/bitdecode/bin/python() {\n case " $* " in\n *record_offline_chain_exit.py*) '+shlex.quote(sys.executable)+' "$@" ;;\n *) exec '+shlex.quote(sys.executable)+' "$@" ;;\n esac\n}\n');env['BASH_ENV']=str(rcfile)
   p=subprocess.run(['bash',str(f/'scripts/run_offline.sh')],capture_output=True,text=True,env=env,timeout=15);assert p.returncode==expected,(name,p.returncode,p.stderr);result=json.loads((f/'OFFLINE_CHAIN_OBSERVED_EXIT.json').read_text());assert result['controller_shell_wait_status']==controller and result['finisher_shell_wait_status']==finisher;assert result['controller']['identity']['pid']>0 and result['controller']['status']=='ACTUAL_OWNED_WAIT_COMPLETED_ORIGINAL_IDENTITY_ABSENT'
   if session:assert result['finisher']['identity']['pid']>0 and result['finisher']['status']=='ACTUAL_OWNED_WAIT_COMPLETED_ORIGINAL_IDENTITY_ABSENT'
   else:assert result['finisher'] is None
   rows.append(dict(case=name,status='PASS',actual_shell_returncode=p.returncode,controller_shell_wait_status=controller,finisher_shell_wait_status=finisher,CPU_MOCK_ONLY=True))
 print(json.dumps(dict(status='PASS_ACTUAL_OFFLINE_BASH_OWNING_WAIT_AND_LATE_CPU_RECORDER',cases=rows,actual_Linux_proc_identity=sys.platform.startswith('linux'),GPU_calls=0,model_loads=0,real_science_controller_or_finisher_calls=0,interpreter_boundary=boundary,BASH_ENV_function_injected=alias,actual_interpreter_realpath=str(Path(sys.executable).resolve()),production_shell_sha256=sha(R/'scripts/run_offline.sh'),production_recorder_sha256=sha(R/'scripts/record_offline_chain_exit.py')),sort_keys=True))
if __name__=='__main__':main()
