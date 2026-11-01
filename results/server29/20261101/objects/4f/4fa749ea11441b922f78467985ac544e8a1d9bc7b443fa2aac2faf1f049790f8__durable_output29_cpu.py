"""Isolated CPU tests of the exact production29 publisher; no SSH or GPU calls.

Path/server disk boundaries are CPU fixtures. On Linux the production ctypes
renameat2 NOREPLACE executes for real; macOS uses an explicitly labeled syscall
mock because it does not provide that Linux syscall.
"""
import base64
import builtins
import contextlib
import ctypes
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import shutil
import types

R=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('durable_output29_cpu_module',R/'src/durable_output29.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
REAL_IMPORT=builtins.__import__;REAL_STATVFS=os.statvfs
class Fixture:
 def __init__(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='wscale-durable29-cpu-');self.root=Path(self.tmp.name)/'live';self.low=False
  self.cfg=dict(task_id=M.TASK,execution_revision=M.REVISION,compute_root=M.REMOTE_ROOT,live_root=M.LIVE_ROOT,archive_root=M.ARCHIVE_ROOT,source_freeze_sha256='1'*64,permit_sha256='2'*64,human_authorization_sha256='3'*64,unit='GPU1_W_MATCHED4096',formal=False)
 def execute(self,action,cfg=None):
  cfg=dict(self.cfg if cfg is None else cfg);argv=sys.argv;sys.argv=['cpu_mock_remote',action,base64.b64encode(M._canonical(cfg)).decode()]
  fixture=self
  class MockFn:
   def __call__(self,a,old,b,new,flags):
    assert flags==1
    old,new=Path(os.fsdecode(old)),Path(os.fsdecode(new))
    if new.exists():ctypes.set_errno(errno.EEXIST);return -1
    os.rename(old,new);return 0
  fake_ctypes=types.SimpleNamespace(CDLL=lambda *a,**k:types.SimpleNamespace(renameat2=MockFn()),c_int=ctypes.c_int,c_char_p=ctypes.c_char_p,c_uint=ctypes.c_uint,get_errno=ctypes.get_errno)
  def imp(name,*a,**k):
   if name=='pathlib':return types.SimpleNamespace(Path=lambda value:fixture.root if str(value)==M.LIVE_ROOT else Path(value))
   if name=='ctypes' and sys.platform!='linux':return fake_ctypes
   return REAL_IMPORT(name,*a,**k)
  def statvfs(path):
   if self.low:return types.SimpleNamespace(f_bavail=1,f_frsize=4096)
   return types.SimpleNamespace(f_bavail=200*1024**3,f_frsize=1)
  capture=io.StringIO();oldimport=builtins.__import__;builtins.__import__=imp;os.statvfs=statvfs
  try:
   with contextlib.redirect_stdout(capture):exec(compile(M.REMOTE_PROGRAM,'exact_production_REMOTE_PROGRAM','exec'),{'__name__':'__main__'})
   return json.loads(capture.getvalue())
  finally:builtins.__import__=oldimport;os.statvfs=REAL_STATVFS;sys.argv=argv
 def prepare(self):
  data=b'CPU_FIXTURE_ONLY_NO_GPU_NO_SCIENCE\n';rows=[dict(path='worker.log',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())]
  binding={k:v for k,v in self.cfg.items() if k not in ('unit','formal')}
  manifest=dict(binding=binding,unit=self.cfg['unit'],formal=False,actual_exit_returncode=0,release='PASS',files=rows)
  raw=M._canonical(manifest)+b'\n';self.cfg['manifest_sha256']=hashlib.sha256(raw).hexdigest();reserved=self.execute('reserve')
  self.stage=Path(reserved['stage']);self.stage.joinpath('worker.log').write_bytes(data);self.stage.joinpath('DURABLE29_MANIFEST.json').write_bytes(raw)
  return reserved
 def close(self):self.tmp.cleanup()
def main():
 rows=[]
 def case(name,fn):
  f=Fixture()
  try:fn(f);rows.append(dict(name=name,status='PASS'))
  finally:f.close()
 def blocked(f,action='seal',contains='',cfg=None):
  try:f.execute(action,cfg)
  except (RuntimeError,FileExistsError) as exc:assert contains in str(exc),(contains,str(exc))
  else:raise AssertionError('Expected failclosed '+contains)
 def good(f):
  f.prepare();r=f.execute('seal');assert r['status']=='PASS_DURABLE29_CLOSED_UNIT' and r['files']==1 and r['rename_policy']=='renameat2_NOREPLACE';assert not f.stage.exists() and Path(r['remote_root']).exists();blocked(f,'reserve','Already published')
 case('closed unit exactSHAbytes publication and repeat refused',good)
 def badsha(f):f.prepare();(f.stage/'worker.log').write_bytes(b'changed');blocked(f,contains='SHA/byte mismatch')
 case('badSHA or bytes stops publication',badsha)
 def extra(f):f.prepare();(f.stage/'extra').write_text('no');blocked(f,contains='exact file inventory')
 case('extra file stops publication',extra)
 def link(f):f.prepare();(f.stage/'extra').symlink_to('/no-target');blocked(f,contains='Symlink')
 case('symlink stops publication',link)
 def man(f):f.prepare();(f.stage/'DURABLE29_MANIFEST.json').write_text('{}');blocked(f,contains='manifest SHA')
 case('manifest pin mismatch stops publication',man)
 def low(f):f.cfg['manifest_sha256']='4'*64;f.low=True;blocked(f,'reserve','headroom');assert not list((f.root/'staging').iterdir())
 case('low archive storage refuses staging',low)
 def ident(f):f.prepare();x=dict(f.cfg,permit_sha256='f'*64);blocked(f,'reserve','Live root identity',x)
 case('another source or permit cannot reuse live root',ident)
 def root(f):x=dict(f.cfg,live_root='/tmp/foreign');blocked(f,'reserve','Exact authorized',x)
 case('foreign storage root refused',root)
 def compute(f):x=dict(f.cfg,compute_root='/tmp/old');blocked(f,'reserve','Exact authorized',x)
 case('old execution root refused',compute)
 def unit(f):x=dict(f.cfg,unit='../foreign');blocked(f,'reserve','Unsafe unit',x)
 case('path traversal refused',unit)
 def collision(f):
  f.prepare();final=f.root/'units'/f.cfg['unit'];final.mkdir();(final/'original').write_text('preserved');blocked(f,contains='NOREPLACE');assert (final/'original').read_text()=='preserved' and f.stage.exists()
 case('actual NOREPLACE preserves existing directory and collision stage',collision)
 def partial(f):
  f.prepare();p=f.stage/'DURABLE29_MANIFEST.json';m=json.loads(p.read_text());m['actual_exit_returncode']=-11;p.write_bytes(M._canonical(m)+b'\n');f.cfg['manifest_sha256']=M._sha(p);newstage=f.root/'staging'/(f.cfg['unit']+'_'+f.cfg['manifest_sha256'][:16]);f.stage.rename(newstage);f.stage=newstage;r=f.execute('seal');assert r['actual_exit_returncode']==-11 and r['release']=='PASS'
 case('failed exited partial durably preserved without changing exit',partial)
 # Execute the production controller-facing API as well. Only CPU process,
 # exact-root admission and SSH/rsync boundaries are mocked; filesystem hashing,
 # claims, receipts, duplicate refusal and the exact remote program execute.
 original=(M._configuration,M._identity,M._remote,M.subprocess.run);old_reuse=sys.modules.get('qualification_reuse')
 def api_case(name,mode):
  f=Fixture();root=Path(f.tmp.name)/'compute';target=root/'execution'/f.cfg['unit'];target.mkdir(parents=True)
  write=lambda p,x:Path(p).write_bytes(M._canonical(x)+b'\n')
  permit=f.cfg['permit_sha256'];child=dict(pid=987654321,starttick=123,cmdline=['CPU_FIXTURE_ONLY_WORKER'])
  write(root/'GPU_EXECUTION_PLAN.json',dict(qualification_units=[dict(id=f.cfg['unit'],gpu=1)]));write(target/'DISPATCH.json',dict(unit=f.cfg['unit'],formal=False,gpu=1,permit_sha256=permit));write(target/'RESOURCE_INTERVAL.json',dict(unit=f.cfg['unit'],identity=child,release='PASS',exit_returncode=0));write(target/'RESULT.json',dict(CPU_FIXTURE_ONLY=True));(target/'worker.log').write_text('closed CPU fixture\n')
  binding={k:v for k,v in f.cfg.items() if k not in ('unit','formal')};calls=[]
  def identity(pid):
   if pid==child['pid']:
    if mode=='alive':return child
    raise FileNotFoundError('CPU fixture worker actually exited')
   return dict(pid=pid,starttick=234,cmdline=['CPU_FIXTURE_ONLY_PUBLISHER'])
  def remote(action,cfg):
   calls.append(action)
   if mode=='network' and action=='reserve':raise RuntimeError('CPU_FIXTURE_NETWORK_FAILURE')
   result=f.execute(action,cfg)
   if mode=='receipt' and action=='seal':result['permit_sha256']='f'*64
   return result
  def transfer(cmd,**kwargs):
   assert cmd[0]=='rsync';dest=Path(cmd[-1].split(':',1)[1]);shutil.copytree(Path(cmd[-2]),dest,dirs_exist_ok=True)
   if mode=='mutation':(target/'worker.log').write_text('mutated after transfer')
   if mode=='transfer':return types.SimpleNamespace(returncode=1,stderr=b'CPU_FIXTURE_TRANSFER_FAILURE')
   return types.SimpleNamespace(returncode=0,stderr=b'')
  M._configuration=lambda r:binding;M._identity=identity;M._remote=remote;M.subprocess.run=transfer;sys.modules['qualification_reuse']=types.SimpleNamespace(remaining_units=lambda plan:plan['qualification_units'])
  try:
   error=None
   try:M.publish_unit(root,target,unit=f.cfg['unit'],formal=False)
   except RuntimeError as exc:error=str(exc)
   assert (error is None)==(mode=='good'),(name,error)
   if mode=='good':
    assert (target/'DURABLE29_RECEIPT.json').exists() and calls==['reserve','seal']
    try:M.publish_unit(root,target,unit=f.cfg['unit'],formal=False)
    except RuntimeError as exc:assert 'already claimed' in str(exc)
    else:raise AssertionError('API accepted duplicate')
   elif mode=='alive':assert not (target/'DURABLE29_STARTED.json').exists() and not calls
   else:
    assert (target/'DURABLE29_STARTED.json').exists() and not (target/'DURABLE29_RECEIPT.json').exists();before=list(calls)
    try:M.publish_unit(root,target,unit=f.cfg['unit'],formal=False)
    except RuntimeError as exc:assert 'already claimed' in str(exc)
    else:raise AssertionError('API retried failed publication')
    assert calls==before
   rows.append(dict(name=name,status='PASS'))
  finally:
   M._configuration,M._identity,M._remote,M.subprocess.run=original
   if old_reuse is None:sys.modules.pop('qualification_reuse',None)
   else:sys.modules['qualification_reuse']=old_reuse
   f.close()
 for mode in ('good','alive','network','transfer','receipt','mutation'):api_case('controller API '+mode+' once-only boundary',mode)
 print(json.dumps(dict(status='PASS_DURABLE29_CPU_EXACT_REMOTE_PUBLISHER',cases=rows,count=len(rows),CPU_MOCK_ONLY=True,GPU_calls=0,SSH_calls=0,renameat2_actual_linux=sys.platform=='linux',production_source_sha256=M._sha(R/'src/durable_output29.py')),sort_keys=True))
if __name__=='__main__':main()
