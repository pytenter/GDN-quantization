"""Fresh independent streaming hashes, current plus immutable referenced parent raw."""
import json
from pathlib import Path
from collect_terminal import R,cfg,ssh
D=R/'delivery';plan=json.loads((D/'GPU_PLAN.json').read_text());parent=D/plan['parent_delivery_relative']
old_plan=json.loads((parent/'GPU_PLAN.json').read_text());roots=[plan['archive_root'],old_plan['archive_root']];objects={}
def walk(v,origin):
 if isinstance(v,dict):
  if all(k in v for k in ('path','bytes','sha256')) and isinstance(v['path'],str) and any(v['path'].startswith(r+'/') for r in roots):
   x={k:v[k] for k in ('path','bytes','sha256')};key=x['path']
   if key in objects:assert all(objects[key][k]==x[k] for k in x)
   else:objects[key]=dict(x,receipt_sources=[])
   objects[key]['receipt_sources'].append(origin)
  for x in v.values():walk(x,origin)
 elif isinstance(v,list):
  for x in v:walk(x,origin)
for base in (D/'execution',parent/'execution'):
 for p in sorted(base.rglob('*.json')):walk(json.loads(p.read_text()),str(p.relative_to(D)))
rows=[objects[k] for k in sorted(objects)];(R/'operations/EXPECTED_CURRENT_RAW_OBJECTS.json').write_text(json.dumps(rows,indent=2)+'\n')
remote29=r'''import json,hashlib,os,sys,uuid,time
from pathlib import Path
v=json.load(sys.stdin);out=[];root=Path(v['root']);root.mkdir(parents=True,exist_ok=True);roots=v['allowed_roots'];expected={x['path']:x for x in v['objects']}
pc=Path(v['parent_catalog']['path']);pb=pc.read_bytes();assert len(pb)==v['parent_catalog']['bytes'] and hashlib.sha256(pb).hexdigest()==v['parent_catalog']['sha256']
progress=root/'FINAL_HASH_PROGRESS.jsonl'
for i,x in enumerate(v['objects']):
 p=Path(x['path']);assert any(str(p).startswith(r+'/') for r in roots) and p.is_file() and not p.is_symlink()
 a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 z=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 assert z.st_size==x['bytes'] and h.hexdigest()==x['sha256'],str(p)
 out.append(dict(path=str(p),bytes=z.st_size,sha256=h.hexdigest(),fresh_stream_SHA=True))
 if i%16==0 or i==len(v['objects'])-1:
  with progress.open('a') as f:f.write(json.dumps(dict(verified=i+1,total=len(v['objects']),monotonic=time.monotonic()))+'\n');f.flush();os.fsync(f.fileno())
orphans=[dict(path=str(p),bytes=p.stat().st_size) for p in root.rglob('*') if p.is_file() and '.tmp.' in p.name]
unindexed=[dict(path=str(p),bytes=p.stat().st_size) for p in root.rglob('*.bin') if str(p) not in expected and '.tmp.' not in p.name]
result=dict(status='PASS_FRESH_INDEPENDENT_REMOTE_RAW_SHA_BYTES',objects=out,object_count=len(out),total_bytes=sum(x['bytes'] for x in out),parent_catalog_unchanged=v['parent_catalog'],orphan_tmp=orphans,unindexed_bin=unindexed,allowed_roots=roots)
p=root/'FINAL_ASSET_CATALOG.json';assert not p.exists(),'Final catalog already published: inspect existing seal'
b=(json.dumps(result,sort_keys=True)+'\n').encode();t=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
os.link(t,p);t.unlink();assert p.read_bytes()==b
result['catalog']=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),atomic_no_clobber=True)
print(json.dumps(result))'''
forward=r'''import subprocess,shlex,sys
command=shlex.join(['/usr/bin/python3','-B','-c',sys.argv[1]])
p=subprocess.run(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','zypan@202.38.247.29',command],input=sys.stdin.buffer.read(),capture_output=True,timeout=14400)
assert p.returncode==0,p.stderr.decode();sys.stdout.buffer.write(p.stdout)'''
pc=json.loads((parent/'archive/FINAL_ARCHIVE_VERIFICATION.json').read_text())['catalog']
v=json.loads(ssh(forward,[remote29],json.dumps(dict(root=plan['archive_root'],allowed_roots=roots,objects=rows,parent_catalog=pc)).encode(),timeout=14460))
assert len(v['objects'])==len(rows) and not v['unindexed_bin']
o=D/'archive';o.mkdir(exist_ok=True)
(o/'FINAL_ARCHIVE_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n')
(o/'ASSET_CATALOG.json').write_text(json.dumps(dict(objects=rows,remote_catalog=v['catalog']),indent=2)+'\n')
print(json.dumps({k:v[k] for k in ('status','object_count','total_bytes','orphan_tmp','unindexed_bin','catalog')}))
