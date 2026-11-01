"""One new CPU-only two-fresh admission call and bounded proof fetch; no GPU."""
from pathlib import Path
import base64,hashlib,importlib.util,json,datetime,sys
R=Path(__file__).resolve().parents[1]
HELPER_SHA='48ac884f57a20a58d9d0b0fc27881b2ce7ee6ea0b7acba41087df6d6c904c787'
TRANSPORT_SHA='db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 with p.open('x')as f:json.dump(d,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
REMOTE=r'''
from pathlib import Path
import os,json,runpy,io,contextlib,hashlib,base64,traceback
R=Path('/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_exponent_repair_20261025');target=R/'operations/run_exponent_repair_final_cpu.py'
assert hashlib.sha256(target.read_bytes()).hexdigest()==EXPECTED_SHA
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (R/'GPU_PERMIT.json').exists()
assert not (R/'operations/TARGET_TWO_FRESH_FINAL_EXPONENT_CPU_INTEGRATION.json').exists()
assert not list((R/'engineering').glob('EXPONENT_FINAL_CPU_INTEGRATION_fresh*'))
buf=io.StringIO();error=None
try:
 with contextlib.redirect_stdout(buf):runpy.run_path(str(target),run_name='__main__')
except BaseException:error=traceback.format_exc()
report=R/'operations/TARGET_TWO_FRESH_FINAL_EXPONENT_CPU_INTEGRATION.json'
paths=([report]if report.exists()else[])+[p for directory in sorted((R/'engineering').glob('EXPONENT_FINAL_CPU_INTEGRATION_fresh*'))for p in sorted(directory.rglob('*'))if p.is_file()and '__pycache__'not in p.parts and p.suffix in('.json','.log')]
rows=[]
for p in paths:
 b=p.read_bytes();rows.append(dict(path=p.relative_to(R).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
assert len(rows)<=160 and sum(x['bytes']for x in rows)<16*1024**2
print(json.dumps(dict(status='PASS_TWO_FRESH_TARGET_CPU_FINAL'if error is None and report.exists()else'FAIL_TARGET_CPU_ADMISSION_PRESERVED',error=error,captured_progress_stdout=buf.getvalue(),files=rows,GPU_calls=0,model_loads=0,real_AIME_or_scorer_calls=0),sort_keys=True))
'''
def main():
 assert sha(R/'operations/run_exponent_repair_final_cpu.py')==HELPER_SHA
 p=R/'operations/remote.py';assert sha(p)==TRANSPORT_SHA
 claim=R/'operations/ACTUAL_TARGET_TWO_FRESH_EXPONENT_CPU_EXECUTION_CLAIM.json';receipt=R/'operations/ACTUAL_TARGET_TWO_FRESH_EXPONENT_CPU_EXECUTION_RECEIPT.json';failure=R/'operations/ACTUAL_TARGET_TWO_FRESH_EXPONENT_CPU_EXECUTION_FAILURE.json'
 assert not any(p.exists()for p in [claim,receipt,failure])
 save(claim,dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Two planned fresh CPU passes at distinct new prefix after preserved pre-test namespace rejection; no GPU/source/finisher execution',helper_sha256=HELPER_SHA,no_automatic_repeat=True))
 sp=importlib.util.spec_from_file_location('exponent_final_target_CPU_transport',p);t=importlib.util.module_from_spec(sp);sp.loader.exec_module(t)
 try:
  out=t.run('EXPECTED_SHA='+repr(HELPER_SHA)+'\n'+REMOTE,role='55',python='/data/ydai/miniconda3/envs/bitdecode/bin/python',timeout=600)
  rows=out.pop('files');meta=[{k:v for k,v in x.items()if k!='base64'}for x in rows];out['proof_files']=meta
  save(receipt,out)  # Retain actual remote failure/progress BEFORE any local file copy.
  for row in rows:
   rel=Path(row['path']);assert not rel.is_absolute()and '..'not in rel.parts and str(rel)==row['path'];q=R/rel
   b=base64.b64decode(row['base64'],validate=True);assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256']
   q.parent.mkdir(parents=True,exist_ok=True)
   if q.exists():assert q.is_file()and sha(q)==row['sha256']and q.stat().st_size==row['bytes']
   else:
    with q.open('xb')as f:f.write(b)
  save(R/'operations/ACTUAL_TARGET_EXPONENT_CPU_PROOF_FILES_VERIFICATION.json',dict(status='PASS_ALL_CPU_PROOF_FILES_SHA_BYTES',files=meta,count=len(meta),GPU_calls=0,model_loads=0))
  print(json.dumps({'status':out['status'],'error':out['error'],'receipt_sha256':sha(receipt),'proof_files':len(rows),'progress':out['captured_progress_stdout']},ensure_ascii=False),flush=True)
  assert out['status']=='PASS_TWO_FRESH_TARGET_CPU_FINAL','Actual CPU admission failed; preserve all receipts'
 except BaseException as e:
  if not failure.exists():save(failure,dict(status='FAIL_CPU_ADMISSION_OR_CPU_PROOF_TRANSFER_NO_AUTOMATIC_REPEAT',error=repr(e),claim_sha256=sha(claim),GPU_calls=0))
  raise
if __name__=='__main__':main()
