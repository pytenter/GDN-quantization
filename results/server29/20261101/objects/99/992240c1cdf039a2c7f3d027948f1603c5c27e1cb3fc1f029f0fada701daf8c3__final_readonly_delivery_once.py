"""One-shot independent CPU audit/delivery. No GPU, finisher, source/seal mutation."""
import os,sys,json,base64,hashlib,subprocess,fcntl,time,tarfile,traceback
from pathlib import Path
import remote
R=remote.ROOT;SOURCE='dd3c3cf9ba95a83d956e024025173e0e1e9e835428d495d2ce0af1c32098a6be';PERMIT='9091f50bc95972ccbc933d6dfc294598a46b2b93a901c883392719785b52f2b1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
lock=(R/'operations/LOCAL_DELIVERY.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(R/'operations/LOCAL_VERIFIED_DELIVERY.json').exists():print('ALREADY_VERIFIED_NO_TRANSFER');raise SystemExit
if(R/'operations/LOCAL_DELIVERY_CLAIM.json').exists():raise RuntimeError('Existing transfer/audit claim: inspect actual local process/failure; never blindly repeat transfer')
ready=remote.run(r'''import json,sys
from pathlib import Path
R=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1');sys.path.insert(0,str(R/'src'));import execution_control as C
out={n:C.read(R/n)for n in ['GPU_RUN_STARTED.json','FINALIZER_STARTED.json','FINAL_SEAL_RECEIPT.json','operations/OFFLINE_ACTUAL_EXIT.json']if(R/n).exists()}
alive=any(C.same_process(out[n]['identity'])for n in ['GPU_RUN_STARTED.json','FINALIZER_STARTED.json']if n in out)
print(json.dumps(dict(ready=bool(out.get('FINAL_SEAL_RECEIPT.json')and out.get('operations/OFFLINE_ACTUAL_EXIT.json')and not alive),state=out)))
''',timeout=55)
if not ready['ready']:print(json.dumps(dict(status='WAIT_NOT_TERMINAL_NO_TRANSFER',readiness=ready)));raise SystemExit
remote.save('operations/LOCAL_DELIVERY_CLAIM.json',dict(pid=os.getpid(),start_observation=subprocess.run(['ps','-p',str(os.getpid()),'-o','lstart=,command='],capture_output=True,text=True).stdout,source_sha256=SOURCE,permit_sha256=PERMIT,started_time=time.time(),scope='CPU independent audit plus exactly one transfer, no GPU/finisher/source/seal mutation'))
try:
 code=r'''import os,sys,json
from pathlib import Path
R=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1');sys.path.insert(0,str(R/'src'));import execution_control as C
assert C.sha(R/'SOURCE_FREEZE.json')=='SOURCE_PIN' and C.sha(R/'GPU_PERMIT.json')=='PERMIT_PIN'
C.verify_sources(R);C.model_guard(R);external=C.verify_external(R)
archive=Path(C.read(R/'ROOT.json')['archive_root']);manifest=C.read(R/'SHA256_MANIFEST.json');assert C.sha(R/'SHA256_MANIFEST.json')==C.sha(archive/'SHA256_MANIFEST.json')
seal=C.read(R/'FINAL_SEAL_RECEIPT.json');assert seal==C.read(archive/'FINAL_SEAL_RECEIPT.json')and seal['manifest_sha256']==C.sha(R/'SHA256_MANIFEST.json')and seal['archive_verification_sha256']==C.sha(R/'FINAL_ARCHIVE_VERIFICATION.json')
assert C.sha(R/'FINAL_ARCHIVE_VERIFICATION.json')==C.sha(archive/'FINAL_ARCHIVE_VERIFICATION.json')
for x in manifest['files']:
 for base in [R,archive]:
  p=base/x['path'];assert p.is_file()and not p.is_symlink()and p.stat().st_size==x['bytes']and C.sha(p)==x['sha256'],str(p)
ledger=C.read(R/'RESOURCE_LEDGER.json');intervals=ledger['intervals'];identities=[x['identity']for x in intervals if x['identity']]
for n in ['GPU_RUN_STARTED.json','FINALIZER_STARTED.json']:
 identities.append(C.read(R/n)['identity'])
identities.append(C.read(R/'operations/OFFLINE_ONCE_STARTED.json')['identity'])
assert all(not C.same_process(i)for i in identities)
obs=C.query_gpu();assert not any(x['pid']in{i['pid']for i in identities}for x in obs['apps'])
assert all(x['release']=='PASS'for x in intervals)
exit=C.read(R/'operations/OFFLINE_ACTUAL_EXIT.json');assert exit['finisher_exit']==0
formal=[x for x in intervals if (R/'formal'/x['unit']/'RESULT.json').exists()and x['exit_returncode']==0];qual=[x for x in intervals if(R/'execution'/x['unit']/'RESULT.json').exists()and x['exit_returncode']==0]
from record_validation import validate
jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];lookup={j['id']:j for j in jobs}
for x in qual+formal:
 d=R/('formal'if x in formal else'execution')/x['unit'];w=C.read(d/'RESULT.json');p=dict(w);saved=p.pop('payload_sha256');assert C.objsha(p)==saved and w['identity']==x['identity']and w['gpu_uuid']==C.UUIDS[x['gpu']]and w['source_freeze_sha256']==C.sha(R/'SOURCE_FREEZE.json')and w['permit_sha256']==C.sha(R/'GPU_PERMIT.json')
 assert C.read(d/'STARTED.json')['identity']==x['identity']
 driver=C.read(d/'DRIVER_IDENTITY.json');assert driver['status']=='PASS'and driver['actual_uuid']==C.UUIDS[x['gpu']]
 if x in formal:validate(w,lookup[x['unit']],R)
 for n in ['PREFLIGHT_LIBRARY.json','POSTFLIGHT_LIBRARY.json']:
  if n.startswith('POST')and not(d/n).exists():assert next(u for u in C.read(R/'GPU_EXECUTION_PLAN.json')['qualification_units']if u['id']==x['unit'])['kind']=='gpu_codec_micro';continue
  assert C.read(d/n)['status']=='PASS'
assert C.sha(R/'history/group32/AIME26_FROZEN_V4_SCORES.json')==C.read(R/'GROUP32_HISTORY_REUSE_AND_IDENTITY.json')['sealed_scores_sha256']
if manifest['status']=='FORMAL_COMPLETE_120_120_AND_STOP':
 assert len(formal)==120 and len(qual)==38 and exit['controller_exit']==0
 assert C.read(R/'SCORE_PASS_0.json')==C.read(R/'SCORE_PASS_1.json')==C.read(R/'FROZEN_V4_SCORES.json')
else:assert not(R/'FROZEN_V4_SCORES.json').exists()
# Transfer only current small science/control files. All raw tensors remain in verified archive.
selected=[];excluded=[]
for x in manifest['files']:
 p=Path(x['path']);head=p.parts[0]
 if head in ('history','runtime','fixtures')or p.suffix.lower()in('.bin','.npz','.npy','.pt','.pth','.safetensors'):excluded.append(x);continue
 selected.append(x)
for n in ['SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','operations/OFFLINE_ACTUAL_EXIT.json']:
 p=R/n
 if n not in {x['path']for x in selected}:selected.append(dict(path=n,bytes=p.stat().st_size,sha256=C.sha(p)))
print(json.dumps(dict(status='PASS_INDEPENDENT_CURRENT_SOURCE_NATIVE_MODEL_RAW_ARCHIVE_AND_OWN_RELEASE',science_status=manifest['status'],source_sha256=C.sha(R/'SOURCE_FREEZE.json'),source_count=C.read(R/'SOURCE_FREEZE.json')['count'],external_files_verified=external,archive_file_count=manifest['count'],all_root_archive_SHA_bytes=True,formal_valid_count=len(formal),qualification_valid_count=len(qual),selected_files=selected,excluded_files=excluded,excluded_scope='Remote full source/history/raw fixtures retained and all hashes audited; env/model external preserved, never downloaded toMac',final_observation=obs)))
'''.replace('SOURCE_PIN',SOURCE).replace('PERMIT_PIN',PERMIT)
 audit=remote.run(code,timeout=1800);remote.save('operations/FINAL_INDEPENDENT_READONLY_VERIFICATION.json',audit)
 stage=R/'delivery.partial';assert not stage.exists()and not(R/'delivery').exists();stage.mkdir()
 names=[x['path']for x in audit['selected_files']]
 transfer=r'''import sys,tarfile
from pathlib import Path
R=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')as t:
 for name in NAMES:
  p=R/name;assert p.is_file()and not p.is_symlink();t.add(p,arcname=name,recursive=False)
'''.replace('NAMES',repr(names))
 # One stream, no remote temporary tar or second publication.
 err=(R/'operations/READONLY_TRANSFER.stderr').open('xb');proc=subprocess.Popen(remote.command(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err);proc.stdin.write(transfer.encode());proc.stdin.close()
 try:
  with tarfile.open(fileobj=proc.stdout,mode='r|gz')as t:
   for m in t:
    p=Path(m.name);assert m.isfile()and not p.is_absolute()and'..'not in p.parts and str(p)in names
    dest=stage/p;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb')as f:
     source=t.extractfile(m)
     while True:
      b=source.read(8*1024**2)
      if not b:break
      f.write(b)
  assert proc.wait(timeout=60)==0
 finally:
  err.close()
  if proc.poll()is None:proc.terminate();proc.wait(timeout=15)
 for x in audit['selected_files']:
  p=stage/x['path'];assert p.is_file()and p.stat().st_size==x['bytes']and sha(p)==x['sha256'],str(p)
 stage.rename(R/'delivery');remote.save('operations/LOCAL_VERIFIED_DELIVERY.json',dict(status='PASS_SELECTED_SMALL_SCIENCE_FILES_SHA_BYTES',selected_count=len(audit['selected_files']),science_status=audit['science_status'],independent_remote_audit_sha256=sha(R/'operations/FINAL_INDEPENDENT_READONLY_VERIFICATION.json'),delivery_root=str(R/'delivery'),raw_model_tensor_downloads=0,source_sha256=SOURCE,permit_sha256=PERMIT))
 print(json.dumps(dict(status='PASS_COMPLETE_VERIFIED_DELIVERY',science_status=audit['science_status'],formal_valid_count=audit['formal_valid_count'],qualification_valid_count=audit['qualification_valid_count'])))
except BaseException as e:
 remote.save('operations/LOCAL_DELIVERY_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),GPU_retries=0,repeat_transfer_allowed=False));raise
