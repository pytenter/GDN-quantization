"""Synthetic actual-controller receipt -> worker binding test. No /proc orGPU access."""
from pathlib import Path
import hashlib,json,os,sys,tempfile
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import run_cublas_controller as c
import cublas_worker as w
from cublas_harness import build_stage_schedule
q=R/'qualification/controller_review_synthetic';q.mkdir(exist_ok=True);root=Path(tempfile.mkdtemp(prefix='dispatch_',dir=q));(root/'execution/B0_FP_Q0').mkdir(parents=True);(root/'scripts').mkdir();(root/'SOURCE_MANIFEST.json').write_text('{}');(root/'AUTH.json').write_text('{}');(root/'RUN_COPY_ORIGIN.json').write_text('{}')
argv=['/synthetic/python','-B',str(root/'scripts/run_cublas_controller.py'),'--root',str(root),'--run'];suffix=['R']+['0']*18+['123']+['0']*4;stat='900 (python) '+' '.join(suffix)
class FakePath:
 def __init__(self,s):self.s=str(s)
 def __truediv__(self,x):return FakePath(self.s+'/'+str(x))
 def read_text(self):assert self.s.endswith('/stat');return stat
 def read_bytes(self):assert self.s.endswith('/cmdline');return ('\0'.join(argv)+'\0').encode()
oldpath=c.Path
try:c.Path=FakePath;identity=c.proc_identity(900)
finally:c.Path=oldpath
unit=build_stage_schedule()[0];auth={'source_manifest_sha256':c.sha(root/'SOURCE_MANIFEST.json')};meta=dict(controller_pid=900,controller_starttick=identity['starttick'],controller_cmdline=identity['cmdline'],authorization_sha256=c.sha(root/'AUTH.json'),source_manifest_sha256=auth['source_manifest_sha256']);dispatch=dict(unit=unit,controller_pid=900,controller_starttick=identity['starttick'],authorization_sha256=c.sha(root/'AUTH.json'),source_manifest_sha256=auth['source_manifest_sha256'],budget_snapshot=dict(completed_observed_seconds_upper=0,unit_allowance_seconds=180,protective_seconds=1650,hard_seconds=1800));(root/'execution/CONTROLLER_STARTED.json').write_text(json.dumps(meta));dp=root/'execution/B0_FP_Q0/DISPATCH.json';dp.write_text(json.dumps(dispatch));oldenv=os.environ.get('QWEN_CONTROLLER_DISPATCH_SHA256');checks=[]
def expect(name,reader=None,pid=900,deny=False):
 try:w.controller_admission(root,root/'AUTH.json',auth,unit,dict(future_execution_root=str(root)),reader=reader or(lambda p:(123,argv)),parent_pid=pid)
 except RuntimeError:
  if not deny:raise
  checks.append(dict(name=name,status='PASS_REJECTED'))
 else:
  if deny:raise AssertionError(name+'accepted')
  checks.append(dict(name=name,status='PASS'))
try:
 os.environ['QWEN_CONTROLLER_DISPATCH_SHA256']=c.sha(dp)
 expect('actual_controller_receipt_shape_admitted_by_worker')
 expect('PID_reuse_rejected',reader=lambda p:(124,argv),deny=True)
 expect('foreign_parent_rejected',pid=901,deny=True)
 expect('controller_cmdline_change_rejected',reader=lambda p:(123,argv+['extra']),deny=True)
 os.environ['QWEN_CONTROLLER_DISPATCH_SHA256']='wrong';expect('dispatch_hash_tamper_rejected',deny=True)
finally:
 if oldenv is None:os.environ.pop('QWEN_CONTROLLER_DISPATCH_SHA256',None)
 else:os.environ['QWEN_CONTROLLER_DISPATCH_SHA256']=oldenv
out=dict(status='PASS_CPU_CROSS_RECEIPT_BINDING_NOT_GPU',GPU_queries=0,GPU_dispatches=0,checks=checks,actual_controller_receipt_value_types={k:type(v).__name__ for k,v in identity.items()},controller_sha256=c.sha(R/'scripts/run_cublas_controller.py'),worker_sha256=c.sha(R/'scripts/cublas_worker.py'),not_GPU_qualification=True);(R/'analysis/controller_dispatch_parent_mock.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],len(checks),'checks')
