import hashlib,json,pathlib,subprocess,sys,time
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
with (r/'hashes/FINAL_ARCHIVE_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',count=len(checked),manifest_sha256=expected)))
