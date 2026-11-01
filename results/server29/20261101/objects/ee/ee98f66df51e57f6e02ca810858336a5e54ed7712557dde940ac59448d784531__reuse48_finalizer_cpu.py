"""Fresh CPU full/partial execution of production finisher with explicit mocks.

Mocks: device/process/reuse admission/scored payloads/SSH transport. The actual
production finisher performs its full/partial accounting, report, CSV, allfile
SHA verification, fsync and NOREPLACE publishing on isolated CPU fixture files.
No GPU, actual AIME generation, model load or scientific finisher is invoked.
"""
import builtins
import contextlib
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
R=Path(__file__).resolve().parents[1];SOURCE=R/'scripts/finalize_offline.py'
REAL_RUN=subprocess.run;REAL_IMPORT=builtins.__import__;REAL_STATVFS=os.statvfs

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True)+'\n')
def require(v,m):
 if not v:raise RuntimeError(m)
def identity(pid):
 if sys.platform!='linux':return dict(pid=pid,starttick=12345,cmdline=['CPU_MOCK_ONLY'])
 p=Path('/proc')/str(pid);return dict(pid=pid,starttick=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def alive(x):
 if not isinstance(x,dict) or 'pid' not in x or sys.platform!='linux':return False
 try:return identity(x['pid'])==x
 except (FileNotFoundError,ProcessLookupError):return False
class Fixture:
 def __init__(self,full):
  self.tmp=tempfile.TemporaryDirectory(prefix='wscale-reuse48-finalizer-cpu-');self.root=Path(self.tmp.name)/'compute';self.root.mkdir();self.archive=Path(self.tmp.name)/'archive';self.full=full;self.scores=[];self.score_calls=[];self.ssh_calls=[]
  self.C=types.ModuleType('execution_control');c=self.C;c.TASK='CPU_FIXTURE_ONLY';c.METHODS=('W','R','RW');c.read=read;c.sha=sha;c.require=require;c.atomic_new=lambda p,x:self.new(p,x);c.same_process=alive;c.query_gpu=lambda:dict(CPU_MOCK_ONLY=True);c.process_identity=identity;c.verify_sources=lambda root:None;c.model_guard=lambda root:None
  self.reuse=types.ModuleType('qualification_reuse');self.reuse.admit_reused48=lambda root,plan:dict(status='PASS_REUSE48_EXACT_BYTEVIEW_ORIGIN_EXIT0_RELEASE',count=48);self.reuse.remaining_units=lambda plan:plan['qualification_units']
  write(self.root/'ROOT.json',dict(archive_root=str(self.archive)));write(self.root/'SOURCE_FREEZE.json',dict(CPU_MOCK_ONLY=True));write(self.root/'GPU_PERMIT.json',dict(CPU_MOCK_ONLY=True));write(self.root/'QUALIFICATION_REUSE_ADMISSION.json',dict(CPU_MOCK_ONLY=True));write(self.root/'GPU_RUN_STARTED.json',dict(identity=dict(CPU_MOCK_ONLY=True)))
  units=[dict(id='GPU1_'+q+'_MATCHED4096',gpu=1) for q in ('base','W','R','RW')];write(self.root/'GPU_EXECUTION_PLAN.json',dict(qualification_units=units));jobs=[];ledger=[]
  for pid in range(1,31):
   name='aime26_'+str(pid).zfill(2)
   for q in c.METHODS:
    unit=name+'_'+q+'_seed1';jobs.append(dict(id=unit,condition=q,problem_id=name));row=dict(problem_id=name,condition=q,correct=pid<=20,extracted_answer='1',generated_token_count=2,finish_reason='eos',gpu_uuid='CPU_MOCK_UUID',finite=True,wall_seconds=1.,payload_sha256='0'*64);self.scores.append(row)
  write(self.root/'FORMAL_JOB_MANIFEST.json',dict(jobs=jobs));write(self.root/'MOCK_SCORE_INPUT.json',dict(status='CPU_MOCK_SCORE_ONLY_NO_ACTUAL_AIME',scores=self.scores))
  score_script=self.root/'scripts/score_pass_cpu.py';score_script.parent.mkdir(parents=True)
  score_script.write_text('import json,sys,time\nfrom pathlib import Path\ntime.sleep(.2)\nr=Path(sys.argv[1]);Path(sys.argv[2]).write_bytes((r/"MOCK_SCORE_INPUT.json").read_bytes())\n')
  n=94 if full else 2
  for index,unit in enumerate(units+jobs):
   if index>=n:break
   name=unit['id'];folder='execution' if index<4 else 'formal';target=self.root/folder/name;target.mkdir(parents=True)
   interval=dict(unit=name,identity=dict(CPU_MOCK_ONLY=True),exit_returncode=0,release='PASS');ledger.append(interval);write(target/'RESOURCE_INTERVAL.json',interval);write(target/'RESULT.json',dict(CPU_MOCK_ONLY=True))
   write(target/'DURABLE29_RECEIPT.json',dict(status='PASS_DURABLE29_CLOSED_UNIT',unit=name,actual_exit_returncode=0,source_freeze_sha256=sha(self.root/'SOURCE_FREEZE.json'),permit_sha256=sha(self.root/'GPU_PERMIT.json')))
  write(self.root/'RESOURCE_LEDGER.json',dict(intervals=ledger));write(self.root/'SESSION_RESULT.json',dict(status='GPU_COMPLETE_REUSE48_PLUS4_QUALIFICATION_90_FORMAL' if full else 'FAIL_CLOSED_STOP'))
  if not full:write(self.root/'FIRST_FAILURE.json',dict(error='CPU_FIXTURE_EXPECTED_FAIL',unit=units[2]['id']))
  else:
   vector={'aime26_'+str(i).zfill(2):i<=21 for i in range(1,31)};other={'aime26_'+str(i).zfill(2):i<=25 for i in range(1,31)}
   write(self.root/'HISTORICAL_G32_BASELINE_REUSE.json',dict(historical_per_problem={'H+G32+Sym8':vector,'H+G32+Asym8':other},historical_scores={'H+G32+Sym8':21,'H+G32+Asym8':25}))
 def new(self,p,x):
  p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('x') as f:json.dump(x,f,sort_keys=True)
 def transport(self,cmd,*args,**kwargs):
  if cmd[0]=='ssh':
   self.ssh_calls.append(cmd);tokens=shlex.split(cmd[-1]);code=tokens[tokens.index('-c')+1];argv=tokens[tokens.index('-c')+2:];old=sys.argv;sys.argv=['remote_cpu_fixture']+argv;out=io.StringIO()
   try:
    with contextlib.redirect_stdout(out):exec(compile(code,'exact_production_remote_archive_code','exec'),{'__name__':'__main__'})
    return types.SimpleNamespace(returncode=0,stdout=out.getvalue().encode(),stderr=b'')
   except BaseException as e:return types.SimpleNamespace(returncode=1,stdout=out.getvalue().encode(),stderr=repr(e).encode())
   finally:sys.argv=old
  if cmd[0]=='rsync':
   src=Path(cmd[-2]);dest=Path(cmd[-1].split(':',1)[1]);shutil.copytree(src,dest,dirs_exist_ok=True,ignore=lambda path,names:['finisher.log'] if 'finisher.log' in names else []);return types.SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
  if any(str(v).endswith('score_pass_cpu.py') for v in cmd):
   self.score_calls.append(cmd);write(Path(cmd[-1]),dict(status='CPU_MOCK_SCORE_ONLY_NO_ACTUAL_AIME',scores=self.scores));return types.SimpleNamespace(returncode=0,stdout=b'CPU_SCORE_MOCK\n',stderr=b'')
  raise AssertionError('Unexpected production finisher subprocess:'+repr(cmd))
 def execute(self):
  oldargv=sys.argv;oldpath=sys.path[:];oldC=sys.modules.get('execution_control');oldReuse=sys.modules.get('qualification_reuse');oldimport=builtins.__import__;sys.argv=[str(SOURCE),str(self.root)];sys.modules['execution_control']=self.C;sys.modules['qualification_reuse']=self.reuse;subprocess.run=self.transport
  class MockFn:
   def __call__(self,a,old,b,new,flags):
    assert flags==1
    if Path(os.fsdecode(new)).exists():ctypes.set_errno(errno.EEXIST);return -1
    os.rename(old,new);return 0
  fake_ctypes=types.SimpleNamespace(CDLL=lambda *a,**k:types.SimpleNamespace(renameat2=MockFn()),c_int=ctypes.c_int,c_char_p=ctypes.c_char_p,c_uint=ctypes.c_uint,get_errno=ctypes.get_errno)
  def imp(name,*a,**k):return fake_ctypes if name=='ctypes' and sys.platform!='linux' else REAL_IMPORT(name,*a,**k)
  builtins.__import__=imp;os.statvfs=lambda path:types.SimpleNamespace(f_bavail=200*1024**3,f_frsize=1)
  try:
   with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(SOURCE),run_name='__main__')
  finally:
   builtins.__import__=oldimport;os.statvfs=REAL_STATVFS;subprocess.run=REAL_RUN;sys.argv=oldargv;sys.path[:]=oldpath
   for name,old in [('execution_control',oldC),('qualification_reuse',oldReuse)]:
    if old is None:sys.modules.pop(name,None)
    else:sys.modules[name]=old
 def close(self):self.tmp.cleanup()
def main():
 rows=[]
 for full in (True,False):
  f=Fixture(full)
  try:
   f.execute();a=read(f.root/'QUALIFICATION_REUSE48_FRESH4_TERMINAL_ACCOUNTING.json');assert a['reused_qualification_count']==48 and a['fresh_qualification_legal_exit0_release']==(4 if full else 2);assert a['current_intervals']==(94 if full else 2);assert sum((f.root/('SCORE_PASS_'+str(rep)+'_ACTUAL_EXIT.json')).exists() for rep in (0,1))==(2 if full else 0)
   expected='FORMAL_COMPLETE_90_90_AND_STOP' if full else 'PARTIAL_RESULTS_PRESERVED_AND_STOP';assert read(f.root/'FINAL_TASK_STATUS.json')['status']==expected;assert (f.root/'AIME26_PER_PROBLEM_RESULTS.csv').exists()==full;assert (f.root/'FROZEN_V4_SCORES.json').exists()==full
   manifest=read(f.archive/'SHA256_MANIFEST.json')
   for x in manifest['files']:p=f.archive/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
   assert read(f.archive/'FINAL_SEAL_RECEIPT.json')['scientific_status']==expected
   if full:
    for rep in (0,1):
     start=read(f.root/('SCORE_PASS_'+str(rep)+'_STARTED.json'));exit=read(f.root/('SCORE_PASS_'+str(rep)+'_ACTUAL_EXIT.json'));assert start['identity']==exit['identity'] and exit['actual_returncode']==0 and exit['same_identity_alive'] is False and start['identity']['pid']>0
   try:f.execute()
   except RuntimeError as exc:assert 'Finisher already started' in str(exc)
   else:raise AssertionError('restart accepted')
   rows.append(dict(mode='full90' if full else 'partial',status='PASS',source_SHA=sha(SOURCE),archive_files=len(manifest['files']),actual_CPU_mock_score_children=2 if full else 0,qualification_accounting=a,actual_production_verifier_executed=True,restart_rejected=True))
  finally:f.close()
 for bad in ('missing_durable','old_status','missing_reuse'):
  f=Fixture(True)
  try:
   if bad=='missing_durable':next((f.root/'formal').glob('*/DURABLE29_RECEIPT.json')).unlink()
   elif bad=='old_status':write(f.root/'SESSION_RESULT.json',dict(status='GPU_COMPLETE_52_QUALIFICATION_90_FORMAL'))
   else:f.reuse.admit_reused48=lambda *a:dict(status='FAIL',count=47)
   try:f.execute()
   except RuntimeError:pass
   else:raise AssertionError('invalid full terminal accepted:'+bad)
   assert not f.score_calls and not f.archive.exists() and (f.root/'FINALIZER_FAILURE.json').exists();rows.append(dict(mode=bad,status='PASS_FAIL_CLOSED_NO_SCORING'))
  finally:f.close()
 print(json.dumps(dict(status='PASS_ACTUAL_REUSE48_FINISHER_FULL_PARTIAL_CPU_CONTROL',cases=rows,renameat2_actual_linux=sys.platform=='linux',CPU_MOCK_ONLY=True,GPU_calls=0,SSH_calls=0,AIME_samples=0),sort_keys=True))
if __name__=='__main__':main()
