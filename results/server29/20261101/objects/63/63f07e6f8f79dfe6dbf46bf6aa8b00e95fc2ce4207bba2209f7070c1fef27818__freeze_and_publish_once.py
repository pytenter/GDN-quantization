"""Publish one immutable, prevalidated new execution; never dispatches GPU work."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os

ROOT=Path(__file__).resolve().parents[1]
TASK='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
spec=importlib.util.spec_from_file_location('staging',ROOT/'operations/stage_reduction_cpu.py')
staging=importlib.util.module_from_spec(spec);spec.loader.exec_module(staging)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def new(p,d):
 with Path(p).open('x') as f:
  json.dump(d,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
 assert ROOT.name==TASK+'_REDUCTION_REPAIR_20261025'
 assert not any((ROOT/n).exists() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','GPU_RUN_STARTED.json'])
 assert read(ROOT/'CPU_CODEC_QUALIFICATION.json')['status']=='PASS_CPU_CONTRACTS_AND_INTEGRATION_NOT_GPU'
 assert read(ROOT/'operations/PREFREEZE_FRESH_ENVIRONMENT_ADMISSION.json')['status']=='PASS_DUAL_HOST_PREFREEZE_ENVIRONMENT_ADMISSION'
 proof=read(ROOT/'operations/SM86_CPU_COMPILE_R3_PROCESS_RECEIPT.json')
 assert proof['returncode']==0 and proof['failure'] is None and proof['report']['variant_count']==20
 assert sha(ROOT/'src/unified_codec.py')=='3331a4f7ebd4118a078e70326bf07d291629a766cd06f50fff12e6c9985e5c34'
 assert sha(ROOT/'src/gpu_codec_fields.py')=='7e0430b04c354988b0ac66c2334236c976b0987055b9acf1d7f210d1815ec891'
 assert sha(ROOT/'operations/INDEPENDENT_R3_COMPILED_REDUCTION_DAG_REVIEW.json')=='5a3cc33f14e02f142abc1e14ce4a2149c66dcc154d9b5c9db5b329da3cd5927e'
 # Terminal delivery is a separate local-only tool, bound by a later tool manifest.
 # Its development does not change or extend this execution's science source.
 excluded={'operations/deliver_terminal_cpu_once.py','operations/TERMINAL_DELIVERY_TOOL_MANIFEST.json',
           'operations/TERMINAL_DELIVERY_HELPER_CPU_REVIEW.json','tests/terminal_delivery_cpu.py'}
 rows=[]
 allowed={'.json','.jsonl','.py','.sh','.md','.txt','.log','.ttir','.ttgir','.llir','.ptx','.patch'}
 for p in sorted(ROOT.rglob('*')):
  assert not p.is_symlink(),str(p)
  if not p.is_file():continue
  name=str(p.relative_to(ROOT))
  if '__pycache__' in p.parts or name in excluded or 'terminal_delivery' in name.lower():continue
  assert p.suffix in allowed,name
  s=p.stat();rows.append(dict(path=name,bytes=s.st_size,sha256=sha(p)))
 assert len({x['path'] for x in rows})==len(rows)
 # Stage every frozen byte while no freeze/permit/dispatch exists remotely.
 batches=[];batch=[];size=0
 for row in rows:
  if batch and size+row['bytes']>2*1024**2:batches.append(batch);batch=[];size=0
  batch.append(ROOT/row['path']);size+=row['bytes']
 if batch:batches.append(batch)
 results=[staging.stage(batch) for batch in batches]
 new(ROOT/'operations/FINAL_PREFREEZE_ASSET_STAGING.json',dict(status='PASS_FINAL_PREFREEZE_ASSET_STAGING',batches=results,count=len(rows),GPU_dispatches=0))
 source=dict(task_id=TASK,execution_revision=ROOT.name,status='FROZEN_ONCE_INDEPENDENT_REDUCTION_REPAIR_QUALIFICATION_THEN_CONDITIONAL_FORMAL',
  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,count=len(rows),bytes=sum(x['bytes'] for x in rows),
  qualifications_current=52,formal_new_samples=90,shared_row_parameters={'a':.5,'b':.25},
  inherited_calibration_only=True,old_qualification_credit=0,old_formal_credit=0,
  prior_true_STOP_source_sha256='ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7',
  phase_A_original31_preserved_in_origin_and_provenance=True,new_ROOT_is_independent_revision=True,
  no_GPU_retry=True,no_accuracy_or_proxy_elimination=True)
 new(ROOT/'SOURCE_FREEZE.json',source)
 permit=read(ROOT.parent/TASK/'GPU_PERMIT.json')
 permit.update(execution_root=staging.REMOTE,execution_revision=ROOT.name,
  source_freeze_sha256=sha(ROOT/'SOURCE_FREEZE.json'),
  reduction_repair_authorization_sha256=sha(ROOT/'operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json'),
  calibration_reuse_admission_sha256=sha(ROOT/'CALIBRATION_REUSE_ADMISSION.json'),
  old_qualification_credit=0,old_formal_credit=0,
  scope='ONE_NEW_INDEPENDENT_REDUCTION_REPAIR_52_REQUIRED_QUALIFICATION_THEN90_CONDITIONAL_FORMAL_ONCE')
 new(ROOT/'GPU_PERMIT.json',permit)
 payload={n:(ROOT/n).read_text() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json']}
 code='ROOT='+repr(staging.REMOTE)+'\npayload='+repr(payload)+'\n'+r'''
from pathlib import Path
import hashlib,json,os,sys
r=Path(ROOT);assert not r.is_symlink()
assert not any((r/n).exists() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','GPU_RUN_STARTED.json','SESSION_RESULT.json'])
source=json.loads(payload['SOURCE_FREEZE.json']);permit=json.loads(payload['GPU_PERMIT.json'])
assert source['count']==len(source['files']) and source['qualifications_current']==52 and source['formal_new_samples']==90
for x in source['files']:
 p=r/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
sys.path.insert(0,str(r/'src'));import execution_control as C
C.model_guard(r);storage=C.storage_guard(r);devices=[C.device(g) for g in C.AUTHORIZED_GPUS]
assert not list((r/'execution').glob('*/DISPATCH.json')) and not list((r/'formal').glob('*/DISPATCH.json'))
for n,raw in payload.items():
 p=r/n;t=p.with_name(n+'.publish_'+str(os.getpid()))
 with t.open('xb') as f:f.write(raw.encode());f.flush();os.fsync(f.fileno())
 os.link(t,p);t.unlink()
fd=os.open(r,os.O_RDONLY);os.fsync(fd);os.close(fd)
p=C.authorize(r,r/'GPU_PERMIT.json')
assert not (r/'GPU_RUN_STARTED.json').exists()
print(json.dumps(dict(status='PASS_ONCE_FROZEN_SOURCE_PERMIT_PUBLISHED_ACTUAL_AUTHORIZE_NO_DISPATCH',
 source_sha256=C.sha(r/'SOURCE_FREEZE.json'),permit_sha256=C.sha(r/'GPU_PERMIT.json'),
 files=source['count'],bytes=source['bytes'],storage=storage,selected_devices=devices,GPU_dispatches=0)))
'''
 result=staging.transport.run(code,role='55',python='/data/ydai/miniconda3/envs/bitdecode/bin/python',timeout=120)
 assert result['source_sha256']==sha(ROOT/'SOURCE_FREEZE.json') and result['permit_sha256']==sha(ROOT/'GPU_PERMIT.json')
 new(ROOT/'operations/FROZEN_PUBLICATION_RECEIPT.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='selected_devices'}))

if __name__=='__main__':main()
