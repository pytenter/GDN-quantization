"""Single bounded CPU finish after the offline controller exits. Never retries GPU.
Generates actual report, streams seal, copies only this task to permanent29,
then independently verifies every archive asset with SHA256 and byte count.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

def read(p):return json.loads(Path(p).read_text())
def need(v,s):
 if not v:raise ValueError(s)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def files(root):
 for base,dirs,names in os.walk(root,followlinks=True):
  dirs[:]=sorted(x for x in dirs if x!='__pycache__')
  for name in sorted(names):
   if name.endswith('.pyc'):continue
   p=Path(base)/name;rel=p.relative_to(root)
   need(not p.is_symlink(),'Unexpected symlink file')
   if p.resolve().is_relative_to(root.resolve()):pass
   else:need((rel.parts[0]=='scores' or (rel.parts[0]=='qualification_gpu' and 'scores' in rel.parts)) and p.resolve().is_relative_to(Path('/tmp/qwen_5doc_scores_20261016')),'Unexpected asset escape')
   yield rel,p

def run(root,parent,archive):
 root=Path(root);gate=read(root/'GATE_STATUS_FINAL.json');ledger=read(root/'gpu_resource_ledger.json')
 need(gate['no_further_GPU_work'],'Controller must have stopped dispatching')
 released=bool(gate.get('GPU_release_verified') and ledger.get('final_GPU_release_verified') and all(x.get('released') for x in ledger['phases']))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 stages=[]
 commands=[('current_dependencies',[sys.executable,str(root/'qualification_cpu/audit_parent.py'),'--root',str(root),'--parent',parent,'--stage','POSTRUN']),
  ('postprocess',[sys.executable,str(root/'analysis_scripts/postprocess_scores.py'),'--root',str(root)]),
  ('report',[sys.executable,str(root/'analysis_scripts/finalize_replication.py'),'--root',str(root)])]
 for name,argv in commands if released else []:
  if name=='postprocess' and stages and stages[0]['returncode']!=0:
   stages.append(dict(stage=name,returncode=None,status='NOT_RUN_DEPENDENCY_FAILED'));continue
  start=time.time()
  with (root/('logs/CPU_'+name+'.log')).open('xb') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=env)
  stages.append(dict(stage=name,returncode=p.returncode,started_unix=start,ended_unix=time.time()))
  if p.returncode:
   failure=dict(status='FAIL_AND_STOP',phase='CPU_'+name,returncode=p.returncode,no_GPU_retry=True,log='logs/CPU_'+name+'.log')
   write(root/('analysis/CPU_'+name+'_FAILURE.json'),failure)
   if not (root/'FIRST_FAILURE.json').exists():write(root/'FIRST_FAILURE.json',failure)
 if not (root/'FINAL_SCIENTIFIC_REPORT.md').is_file():
  # CPU reporting failures must themselves remain reviewable. Never invent KL
  # or a science CSV to fill an absent/failed analysis.
  fallback=dict(TASK=read(root/'preregistration.json')['TASK_ID'],FINAL_EXECUTION_STATUS='FAIL_AND_STOP',
   PROSPECTIVE_SCIENTIFIC_RESULT='INCOMPLETE_NOT_CONFIRMATORY',GPU_HOURS_USED=ledger['GPU_HOURS_USED'],
   GPU_RELEASE_VERIFIED=released,GPU_HOURS_ACCOUNTING='Actual verified reservations' if released else 'LOWER_BOUND_UNTIL_RELEASE_VERIFIED',completed_GPU_phases=[x['phase'] for x in ledger['phases'] if x.get('returncode')==0 and x.get('released')],
   CPU_stages=stages,first_failure=read(root/'FIRST_FAILURE.json') if (root/'FIRST_FAILURE.json').exists() else None,
   note='CPU report/identity/analysis failed or GPU release remains unverified; no qualified full5doc scientific answer is claimed',
   tensor_delivery='NOT_VERIFIED/POSSIBLY_ACTIVE; small failure delivery only' if not released else 'Actual final assets independently sealed below')
  if not (root/'FINAL_STATUS.json').exists():write(root/'FINAL_STATUS.json',fallback)
  with (root/'FINAL_SCIENTIFIC_REPORT.md').open('x') as f:
   f.write('# 真实失败交付\n\nFAIL_AND_STOP；INCOMPLETE_NOT_CONFIRMATORY。\n\n'+json.dumps(fallback,ensure_ascii=False,indent=2)+'\n\nQ1–Q4：未获得本轮完整合格科学答案，不将缺失视为平局。Q5–Q6：仅已存实际安全/资源/首失败回执可查。Q7：项目独立性UNKNOWN，不作显著性宣称。Q8：不能据此声称交互收益，不运行后续实验。Q9：物理packedINT8、FP/H等价、机制归因、泛化、AIME与learnedrotation互补均未建立。\n\nVERIFIED_ENGINEERING仅限真实PASS回执；PROSPECTIVE_SCIENTIFIC_RESULT=INCOMPLETE；POSTHOC_DIAGNOSTIC仅实际产物；NOT_ESTABLISHED如上。\n')
 if not (root/'GPU_RESOURCE_LEDGER.json').exists():
  with (root/'GPU_RESOURCE_LEDGER.json').open('xb') as f:f.write((root/'gpu_resource_ledger.json').read_bytes())
 write(root/'analysis/OFFLINE_FINISH_STAGES.json',dict(status='PASS' if released and all(x['returncode']==0 for x in stages) else 'FAIL_AND_STOP_WITH_ACTUAL_REPORT',stages=stages,no_GPU_executed=True,no_further_research=True))
 # All writers except this bounded finish have stopped. Its own live stdout log
 # is explicitly excluded to keep the payload manifest acyclic and immutable.
 remote_script='''import hashlib,json,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);expected=sys.argv[2];m=r/'hashes/SHA256SUMS.txt'
assert hashlib.sha256(m.read_bytes()).hexdigest()==expected
c=json.loads((r/'hashes/PAYLOAD_CATALOG.json').read_text());sizes={x['path']:x['bytes'] for x in c['assets']};checked=[]
for line in m.read_text().splitlines():
 h,rel=line.split('  ',1);p=r/rel;assert p.resolve().is_relative_to(r.resolve());a=p.stat();v=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16777216),b''):v.update(b)
 independent=subprocess.check_output(['sha256sum','--',str(p)],text=True).split()[0]
 z=p.stat();assert v.hexdigest()==h==independent and (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
 if rel in sizes:assert z.st_size==sizes[rel]
 checked.append(dict(path=rel,sha256=h,bytes=z.st_size,python_stream_SHA=True,independent_sha256sum=True))
out=dict(status='PASS',permanent_archive_root=str(r),manifest_sha256=expected,verified_assets=checked,verified_count=len(checked),all_current_tensors_verified=c['GPU_release_verified'],delivery_scope=c['delivery_scope'],tensor_scope=c['tensor_scope'],verified_unix=time.time(),no_GPU=True)
with (r/'hashes/FINAL_ARCHIVE_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2);f.write('\\n')
print(json.dumps(dict(status='PASS',count=len(checked),manifest_sha256=expected)))
'''
 verifier=root/'hashes/ARCHIVE_VERIFY_SOURCE.py'
 with verifier.open('x') as f:f.write(remote_script)
 exclusions={'logs/OFFLINE_FINISH.log','hashes/SHA256SUMS.txt','hashes/FINAL_ARCHIVE_VERIFICATION.json'}
 small_failure_paths=set()
 if not released:
  small_failure_paths={e['path'] for e in read(root/'source_manifest.json')['execution_sources']}
  small_failure_paths.update(['source_manifest.json','SOURCE_MANIFEST.json','GATE_STATUS_FINAL.json','gpu_resource_ledger.json','GPU_RESOURCE_LEDGER.json','FIRST_FAILURE.json','FINAL_STATUS.json','FINAL_SCIENTIFIC_REPORT.md','analysis/OFFLINE_FINISH_STAGES.json','hashes/ARCHIVE_VERIFY_SOURCE.py'])
 assets=[]
 for rel,p in files(root):
  if str(rel) in exclusions:continue
  if not released and str(rel) not in small_failure_paths:continue
  before=p.stat();h=sha(p);after=p.stat()
  need((before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)==(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev),'Payload changed while sealing')
  assets.append(dict(path=str(rel),sha256=h,bytes=after.st_size))
 write(root/'hashes/PAYLOAD_CATALOG.json',dict(assets=assets,explicit_nonpayload_live_log='logs/OFFLINE_FINISH.log',no_GPU=True,GPU_release_verified=released,
   delivery_scope='FULL_ACTUAL_ASSET_DELIVERY' if released else 'SMALL_FAILURE_DELIVERY_ONLY',
   tensor_scope='Every actual retained current tensor hashed and independently archive verified' if released else 'NOT_VERIFIED/POSSIBLY_ACTIVE; intentionally omitted all score/checkpoint tensors and live logs'))
 catalog=root/'hashes/PAYLOAD_CATALOG.json';assets.append(dict(path='hashes/PAYLOAD_CATALOG.json',sha256=sha(catalog),bytes=catalog.stat().st_size))
 manifest=root/'hashes/SHA256SUMS.txt'
 with manifest.open('x') as f:
  for a in sorted(assets,key=lambda a:a['path']):f.write(a['sha256']+'  '+a['path']+'\n')
 manifest_sha=sha(manifest)
 ssh=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.29']
 # A fresh archive directory is required. No merge/overwrite of a prior seal.
 subprocess.run(ssh+[shlex.join(['/bin/mkdir','-p',str(Path(archive).parent)])],check=True)
 subprocess.run(ssh+['test ! -e '+shlex.quote(archive)+' && mkdir '+shlex.quote(archive)],check=True)
 copy_command=['rsync','-aL','--exclude=__pycache__','--exclude=*.pyc','--exclude=logs/OFFLINE_FINISH.log','-e',shlex.join(ssh[:-1])]
 if not released:
  names=root/'hashes/SMALL_FAILURE_ARCHIVE_PATHS.txt'
  with names.open('x') as f:f.write('\n'.join(a['path'] for a in assets)+'\nhashes/SHA256SUMS.txt\n')
  copy_command.append('--files-from='+str(names))
 subprocess.run(copy_command+[str(root)+'/',ssh[-1]+':'+archive+'/'],check=True)
 command=shlex.join(['/data/ydai/miniconda3/envs/bitdecode/bin/python',archive+'/hashes/ARCHIVE_VERIFY_SOURCE.py',archive,manifest_sha])
 # Python path may differ on29; use its verified available python3 via explicit
 # remote discovery, without changing model/runtime on55.
 probe=subprocess.check_output(ssh+['command -v python3'],text=True).strip()
 need(probe.startswith('/'),'No archive Python3 available')
 command=shlex.join([probe,archive+'/hashes/ARCHIVE_VERIFY_SOURCE.py',archive,manifest_sha])
 subprocess.run(ssh+[command],check=True)
 subprocess.run(['rsync','-a','-e',shlex.join(ssh[:-1]),ssh[-1]+':'+archive+'/hashes/FINAL_ARCHIVE_VERIFICATION.json',str(root/'hashes/FINAL_ARCHIVE_VERIFICATION.json')],check=True)
 rec=read(root/'hashes/FINAL_ARCHIVE_VERIFICATION.json');need(rec['status']=='PASS' and rec['manifest_sha256']==manifest_sha,'Permanent archive receipt mismatch')
 print(json.dumps(dict(status='DELIVERY_SEALED_AND_STOP',actual_scientific_execution=read(root/'FINAL_STATUS.json')['FINAL_EXECUTION_STATUS'],archive=archive,manifest_sha256=manifest_sha,verified_assets=rec['verified_count'])),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--parent',required=True);p.add_argument('--archive',required=True);a=p.parse_args()
 try:run(a.root,a.parent,a.archive)
 except BaseException as e:
  f=Path(a.root)/'analysis/OFFLINE_FINISH_FAILURE.json'
  if not f.exists():write(f,dict(status='DELIVERY_INCOMPLETE',error=repr(e),traceback=traceback.format_exc(),no_GPU_retry=True))
  raise
