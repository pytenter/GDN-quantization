import hashlib,json,os,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);expected=json.loads(sys.argv[2]);need=lambda v,s:None if v else (_ for _ in ()).throw(RuntimeError(s))
need(not (r/'hashes/FINAL_ARCHIVE_VERIFICATION.json').exists(),'Final seal already exists; no overwrite')
def h(p):
 q=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16777216),b''):q.update(b)
 return q.hexdigest()
for a in expected:
 p=r/a['path'];need(p.is_file() and p.stat().st_size==a['bytes'] and h(p)==a['sha256'],'Final copied asset mismatch '+a['path'])
current=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or p.is_symlink() or p.name.endswith(('.lock','.tmp','.pyc')) or '__pycache__' in p.parts:continue
 rel=str(p.relative_to(r))
 if rel in ('hashes/FINAL_ARCHIVE_VERIFICATION.json','SHA256_MANIFEST.json','hashes/SHA256SUMS.txt'):continue
 before=p.stat();digest=h(p);independent=subprocess.check_output(['sha256sum','--',str(p)],text=True).split()[0];after=p.stat()
 need(digest==independent and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Archive streaming verification failed '+rel)
 current.append(dict(path=rel,sha256=digest,bytes=after.st_size,python_stream_SHA=True,independent_sha256sum=True))
# Archived science scores must exactly match the original per-phase location
# receipts; a bare unreferenced tensor cannot pass as a scientific artifact.
for loc in sorted((r/'archive').glob('DOC_*_CURRENT_ASSET_LOCATIONS.json')):
 m=json.loads(loc.read_text());need(m['status']=='PASS','Invalid score location map')
 for a in m['assets']:
  p=pathlib.Path(a['archive_path']);need(p.resolve().is_relative_to(r.resolve()),'Archive escape')
  need(p.stat().st_size==a['bytes'] and h(p)==a['sha256'],'Archived score mapping mismatch')
manifest=dict(TASK_ID=r.name,status='PASS',assets=current,all_current_tensors_verified=True,tensor_count=sum(a['path'].endswith('.pt') for a in current),source_of_truth='permanent archive .29; current actual artifacts only; retired rolling CPs remain catalogued',explicit_nonpayload_live_log='logs/OFFLINE_FINISH.log',verified_unix=time.time())
with (r/'SHA256_MANIFEST.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
md=h(r/'SHA256_MANIFEST.json')
with (r/'hashes/SHA256SUMS.txt').open('x') as f:
 for a in current:f.write(a['sha256']+'  '+a['path']+'\n')
 f.write(md+'  SHA256_MANIFEST.json\n')
receipt=dict(status='PASS',archive_root=str(r),all_current_tensors_verified=True,verified_asset_count=len(current),verified_tensor_count=manifest['tensor_count'],SHA256_MANIFEST_SHA256=md,SHA256SUMS_SHA256=h(r/'hashes/SHA256SUMS.txt'),actual_report_sha256=h(r/'FINAL_SCIENTIFIC_REPORT.md'),no_GPU=True,verified_unix=time.time())
tmp=r/'hashes/FINAL_ARCHIVE_VERIFICATION.tmp'
with tmp.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
os.replace(tmp,r/'hashes/FINAL_ARCHIVE_VERIFICATION.json')
print(json.dumps(receipt))
