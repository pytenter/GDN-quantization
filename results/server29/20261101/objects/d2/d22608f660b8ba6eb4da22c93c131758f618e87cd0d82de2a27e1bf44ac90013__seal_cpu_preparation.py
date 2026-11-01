"""Immutable local CPU preparation snapshot, never GPU/scientific completion."""
from pathlib import Path
import json,hashlib,shutil,time
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((R/'CPU_CODEC_QUALIFICATION.json').read_text())['Native_actual_full_logits_state_bytes_compared'] is True
assert json.loads((R/'analysis/CPU_RESULT_DOWNLOAD_VERIFICATION.json').read_text())['status']=='PASS_ALL_REMOTE_CPU_RESULT_SMALL_SHA'
sources=[]
for p in sorted(R.rglob('*')):
 rel=p.relative_to(R)
 if not p.is_file() or 'cpu_preparation_delivery' in rel.parts or 'qualification' in rel.parts:continue
 if rel.parts[0] in ['src','scripts','protocol','inherited'] or str(rel) in ['USER_REQUEST.txt','GPU_EXECUTION_PLAN.json','AIME26_SEED1_PROTOCOL.json','AIME26_DATASET_FREEZE.json','QUESTION_GPU_ASSIGNMENT.json','FOUR_CODEC_NUMERICAL_CONTRACT.md']:
  sources.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
f=R/'CPU_PREPARATION_SOURCE_FREEZE.json';assert not f.exists();f.write_text(json.dumps(dict(status='CPU_ONLY_SOURCE_FREEZE_NOT_GPU_ADMISSION',files=sources,count=len(sources),GPU_dispatches=0),indent=2)+'\n')
D=R/'cpu_preparation_delivery';D.mkdir(exist_ok=False);rows=[]
for p in sorted(R.rglob('*')):
 rel=p.relative_to(R)
 if not p.is_file() or 'cpu_preparation_delivery' in rel.parts:continue
 q=D/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert p.read_bytes()==q.read_bytes();rows.append(dict(path=str(rel),bytes=q.stat().st_size,sha256=sha(q)))
manifest=dict(status='CPU_PREPARATION_SEAL_ONLY',files=rows,count=len(rows),bytes=sum(x['bytes'] for x in rows),GPU_dispatches=0,formal_generations=0)
(D/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(D/'SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in rows))
seal=dict(status='PASS_IMMUTABLE_CPU_PREPARATION_SEAL',manifest_sha256=sha(D/'SHA256_MANIFEST.json'),source_freeze_sha256=sha(D/'CPU_PREPARATION_SOURCE_FREEZE.json'),count=len(rows),bytes=manifest['bytes'],GPU_dispatches=0,formal_generations=0,unix=time.time())
(D/'FINAL_SEAL_RECEIPT.json').write_text(json.dumps(seal,indent=2)+'\n')
for x in rows:assert (D/x['path']).stat().st_size==x['bytes'] and sha(D/x['path'])==x['sha256']
(R/'analysis/CPU_SEAL_INDEPENDENT_REREAD_VERIFICATION.json').write_text(json.dumps(dict(status='PASS_ALL_CPU_SNAPSHOT_SMALL_SHA_REREAD',**seal),indent=2)+'\n')
print(json.dumps(seal))
