import hashlib,json,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);expected=json.loads((r/'hashes/DELIVERY_PAYLOAD_EXPECTED.json').read_text());refs=json.loads((r/'hashes/EXTERNAL_ARCHIVED_ASSETS.json').read_text())['assets']
def need(v,s):
 if not v:raise RuntimeError(s)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16777216),b''):h.update(b)
 return h.hexdigest()
def check(a,p):
 need(p.is_file() and p.stat().st_size==a['bytes'],'Archive bytes '+str(p));before=p.stat();h=sha(p);s=subprocess.check_output(['sha256sum','--',str(p)],text=True).split()[0];after=p.stat();need(h==s==a['sha256'] and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Archive doubleSHA '+str(p));return dict(path=str(p),sha256=h,bytes=after.st_size,python_stream_SHA=True,independent_sha256sum=True)
need(not (r/'hashes/FINAL_ARCHIVE_VERIFICATION.json').exists(),'Existing final seal; no overwrite')
need(sha(r/'hashes/DELIVERY_PAYLOAD_EXPECTED.json')==sys.argv[2],'Expected payload document content mismatch')
for a in expected['assets']:check(a,r/a['path'])
checked=[];current=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=str(p.relative_to(r));need(not p.name.endswith(('.partial','.tmp')),'Unfinished archive file '+rel)
 if rel in ['SHA256_MANIFEST.json','hashes/SHA256SUMS.txt','hashes/FINAL_ARCHIVE_VERIFICATION.json']:continue
 a=dict(path=rel,sha256=sha(p),bytes=p.stat().st_size);checked.append(check(a,p));current.append(a)
external=[check(a,pathlib.Path(a['archive_path'])) for a in refs]
manifest=dict(TASK_ID=r.name,status='PASS',assets=current,external_archived_assets=refs,all_current_tensors_verified=True,verified_unix=time.time())
with (r/'SHA256_MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
with (r/'hashes/SHA256SUMS.txt').open('x') as f:
 for a in current:f.write(a['sha256']+'  '+a['path']+'\n')
 f.write(sha(r/'SHA256_MANIFEST.json')+'  SHA256_MANIFEST.json\n')
out=dict(status='PASS',archive_root=str(r),verified_asset_count=len(checked),verified_external_assets=len(external),all_current_tensors_verified=True,current_payload=checked,referenced_external_assets=external,SHA256_MANIFEST_SHA256=sha(r/'SHA256_MANIFEST.json'),SHA256SUMS_SHA256=sha(r/'hashes/SHA256SUMS.txt'),actual_report_sha256=sha(r/'FINAL_SCIENTIFIC_REPORT.md'),no_GPU=True)
with (r/'hashes/FINAL_ARCHIVE_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',verified_asset_count=len(checked),verified_external_assets=len(external),report_SHA256=out['actual_report_sha256'])))
