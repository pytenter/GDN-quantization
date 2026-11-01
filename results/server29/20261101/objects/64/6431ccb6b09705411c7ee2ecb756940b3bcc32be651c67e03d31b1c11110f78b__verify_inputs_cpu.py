"""Read-only CPU content verification; no torch import or GPU initialization."""
import hashlib,json,pathlib,time,os
R=pathlib.Path(__file__).resolve().parents[1];P=R.parent
start=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
assert sha(P/'hashes/artifact_sha256.txt')=='ad024f3b4406680f1d4f7a6de6294f6f889c7667139b86f378bcec1b7ce9add2'
assert sha(P/'hashes/DELIVERY_VERIFICATION.json')=='23f26b85c800156b5546b580fa21f3af482a5b4bfc756c8112b17f3bfd9e488b'
parent=[]
for line in (P/'hashes/artifact_sha256.txt').read_text().splitlines():
 h,rel=line.split('  ',1);f=P/rel;assert sha(f)==h,rel;parent.append(dict(path=rel,sha256=h,bytes=f.stat().st_size))
old=json.loads((P/'execution_01/analysis/PARENT_AND_MODEL_CONTENT_VERIFICATION.json').read_text())
model=[];runtime=[]
for a in old['model_weight_and_config_current_content_hashes']:
 f=pathlib.Path(a['path']);h=sha(f);assert h==a['sha256'] and f.stat().st_size==a['bytes'],str(f)
 model.append(dict(path=str(f),bytes=f.stat().st_size,sha256=h,status='PASS',fresh_full_stream_hash=True))
for a in old['current_runtime_source_recheck']:
 f=pathlib.Path(a['path']);h=sha(f);assert h==a['sha256'],str(f);runtime.append(dict(path=str(f),sha256=h,status='PASS'))
auth=json.loads((R/'execution_authorization_addendum.json').read_text());docs=[]
for d in auth['frozen_documents']:
 f=R/d['relative_token_path'];v=json.loads(f.read_text());ids=v if isinstance(v,list) else v['input_ids']
 h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest();assert len(ids)==128 and h==d['token_ids_sha256'],str(f)
 docs.append(dict(document_id=d['document_id'],token_count=len(ids),canonical_token_ids_sha256=h,status='PASS'))
v=dict(status='PASS',stage='FRESH_CPU_READONLY_SOURCE_WEIGHT_PARENT_AND_DOCUMENT_VERIFICATION',parent_files_verified=len(parent),parent_files=parent,model_weight_and_config_current_content_hashes=model,current_runtime_source_recheck=runtime,documents=docs,CUDA_MODEL_LOAD=False,GPU_HOURS_USED=0,CPU_wall_seconds=time.monotonic()-start,verified_unix=time.time(),disk_free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize)
with (R/'analysis/PARENT_AND_MODEL_CONTENT_VERIFICATION.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',parent_files=len(parent),model_files=len(model),runtime_files=len(runtime),docs=len(docs),CPU_wall_seconds=v['CPU_wall_seconds'])))
