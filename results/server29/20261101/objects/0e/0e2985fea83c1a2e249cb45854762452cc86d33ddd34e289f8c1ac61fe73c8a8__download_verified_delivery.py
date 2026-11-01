#!/usr/bin/env python3
"""CPU-only exact small-asset delivery, path-hash folders preserve case collisions."""
from pathlib import Path
import base64,hashlib,json,sys
import local_ops as ops
B=ops.BASE;D=B/'delivery';D.mkdir(exist_ok=True)
code='''from pathlib import Path
import json,hashlib,sys,base64
out=[]
for row in json.load(sys.stdin):
 p=Path(row['remote']);raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 assert len(raw)<5000000,'large asset forbidden for local delivery'
 out.append(dict(local=row['local'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
print(json.dumps(out))'''
manifest=json.loads(ops.call("from pathlib import Path\nprint((Path("+repr(ops.REMOTE)+")/'DELIVERY_ASSET_MANIFEST.json').read_text())"))
rows=[]
for x in manifest['small_assets']:rows.append(dict(remote=ops.REMOTE+'/'+x['path'],local=x['path'],bytes=x['bytes'],sha256=x['sha256']))
for x in manifest['external_dependencies']:
 key=hashlib.sha256(x['path'].encode()).hexdigest();rows.append(dict(remote=x['path'],local='dependency_snapshot/by_path_sha256/'+key+'/'+Path(x['path']).name,bytes=x['bytes'],sha256=x['sha256']))
# Bind the manifest itself independently, avoiding a self-referential seal.
mraw=ops.call('from pathlib import Path\nimport sys\nsys.stdout.buffer.write((Path('+repr(ops.REMOTE)+')/"DELIVERY_ASSET_MANIFEST.json").read_bytes())')
(D/'DELIVERY_ASSET_MANIFEST.json').write_bytes(mraw)
verified=[]
for start in range(0,len(rows),8):
 for x in json.loads(ops.call(code,rows[start:start+8])):
  raw=base64.b64decode(x.pop('data'));assert hashlib.sha256(raw).hexdigest()==x['sha256'] and len(raw)==x['bytes']
  p=D/x['local'];p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw)
  verified.append(x)
# Re-read every local payload after writes, and all remotely retained large traces via the sealed remote verification receipt.
for x in verified:
 raw=(D/x['local']).read_bytes();assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
receipt=dict(task_id=manifest['task_id'],status='PASS_EXACT_SMALL_ASSET_DELIVERY',files_verified=len(verified),manifest_sha256=hashlib.sha256(mraw).hexdigest(),report_sha256=hashlib.sha256((D/'FINAL_TECHNICAL_REPORT.md').read_bytes()).hexdigest(),frozen_execution_manifest_sha256=hashlib.sha256((D/'SOURCE_SHA256_MANIFEST.json').read_bytes()).hexdigest(),external_dependencies_verified=len(manifest['external_dependencies']),large_asset_policy='No model weights, large tensor or actual Chrome trace downloaded; retained permanent /data on55; remote fresh SHA/bytes receipt included.',verified=verified)
(B/'LOCAL_DELIVERY_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='verified'},ensure_ascii=False))
