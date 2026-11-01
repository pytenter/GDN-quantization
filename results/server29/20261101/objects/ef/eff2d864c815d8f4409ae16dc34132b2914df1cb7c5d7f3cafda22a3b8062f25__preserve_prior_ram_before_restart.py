"""Human-authorized repair prerequisite: durably verify prior RAM refs on29,
then release only exact verified own files. No CUDA or scientific re-execution.
"""
import os,sys,json,hashlib,subprocess,shlex,time,re
from pathlib import Path
R=Path(sys.argv[1]);cfg=json.loads((R/'ROOT.json').read_text());old=Path(cfg['prior_failed_remote_root'])
def put(n,x):
 p=R/'operations'/n
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def remote(code,*args):
 p=subprocess.run(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','zypan@202.38.247.29',shlex.join(['/usr/bin/python3','-u','-B','-c',code,*args])],capture_output=True,text=True,timeout=900)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
def same(x):
 p=Path('/proc')/str(x['pid'])
 try:return int((p/'stat').read_text().rsplit(')',1)[1].split()[19])==x['starttick'] and (p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')==x['cmdline']
 except (FileNotFoundError,ProcessLookupError):return False
assert not (R/'operations/PRIOR_RAM_PRESERVATION_STARTED.json').exists()
put('PRIOR_RAM_PRESERVATION_STARTED.json',dict(pid=os.getpid(),monotonic=time.monotonic(),CPU_only=True))
try:
 catalog=old/'operations/STOP_RAM_REFERENCE_SHA_CATALOG.json';m=json.loads(catalog.read_text());ref=Path(m['root'])
 assert str(ref)=='/dev/shm/qwen_gdn_oscar_sym32_unified_recovery_aime26_v2'
 assert m['file_count']==1313 and m['total_bytes']==113347957760 and m['total_bytes']<=128*1024**3
 ledger=json.loads((old/'RESOURCE_LEDGER.json').read_text());ids=[x['identity'] for x in ledger['intervals']]
 ids += [json.loads((old/n).read_text())['identity'] for n in ['GPU_RUN_STARTED.json','FINALIZER_STARTED.json']]
 assert all(x['release']=='PASS' for x in ledger['intervals']) and not any(same(x) for x in ids)
 rels=[]
 for x in m['files']:
  p=Path(x['path']);rel=p.relative_to(ref);assert re.fullmatch(r'GPU[123]_P1_512_R0/[0-9]{5}\.bin',str(rel));s=p.stat()
  assert (s.st_ino,s.st_size,s.st_mtime_ns)==(x['inode'],x['bytes'],x['mtime_ns'])
  rels.append(str(rel))
 listing=R/'operations/PRIOR_RAM_FILES.txt';listing.write_text('\n'.join(rels)+'\n')
 final='/data/zypan/qwen_oscar_sym32_unified_v2_archive/FAILED_V2_RAM_REFERENCES_20261022_'+sha(catalog)[:16];stage=final+'.staging'
 remote('''import sys,json,os
from pathlib import Path
stage=Path(sys.argv[1]);final=Path(sys.argv[2]);assert not stage.exists() and not final.exists();s=os.statvfs('/data/zypan');assert s.f_bavail*s.f_frsize>=int(sys.argv[3])+100*1024**3;stage.mkdir(parents=True);print(json.dumps(dict(status='RESERVED')))
''',stage,final,str(m['total_bytes']))
 print('Copying1313 RAM engineering references55->29; no Mac raw download.',flush=True)
 ssh=shlex.join(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes'])
 p=subprocess.run(['rsync','-a','--files-from='+str(listing),'-e',ssh,str(ref)+'/', 'zypan@202.38.247.29:'+stage+'/'],capture_output=True,text=True,timeout=7200);assert p.returncode==0,p.stderr
 p=subprocess.run(['scp','-o','ControlPath=/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes',str(catalog),'zypan@202.38.247.29:'+stage+'/REFERENCE_CATALOG.json'],capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr
 print('Verifying all original bytes on29 before any RAM release.',flush=True)
 archived=remote('''import sys,json,hashlib,os,ctypes
from pathlib import Path
stage=Path(sys.argv[1]);final=Path(sys.argv[2]);c=stage/'REFERENCE_CATALOG.json';assert hashlib.sha256(c.read_bytes()).hexdigest()==sys.argv[3];m=json.loads(c.read_text());root=Path(m['root']);rows=[]
for x in m['files']:
 p=stage/Path(x['path']).relative_to(root);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert p.stat().st_size==x['bytes'] and h.hexdigest()==x['sha256'];rows.append(dict(path=str(p.relative_to(stage)),bytes=x['bytes'],sha256=x['sha256']))
v=dict(status='PASS_ALL1313_ORIGINAL_RAM_BYTES_DURABLY_PRESERVED_ON29',archive_root=str(final),files=rows,count=len(rows),bytes=sum(x['bytes'] for x in rows),catalog_sha256=sys.argv[3]);raw=json.dumps(v,indent=2)+chr(10);(stage/'VERIFICATION.json').write_text(raw)
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;assert fn(-100,os.fsencode(stage),-100,os.fsencode(final),1)==0,os.strerror(ctypes.get_errno());fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(v))
''',stage,final,sha(catalog))
 put('PRIOR_RAM_ARCHIVE_VERIFICATION.json',archived)
 assert not any(same(x) for x in ids)
 print('Fresh source SHA then release only exact archived engineering files.',flush=True)
 released=[]
 for x in m['files']:
  p=Path(x['path']);s=p.stat();assert s.st_size==x['bytes'] and not p.is_symlink() and sha(p)==x['sha256'];p.unlink();released.append(x)
  if len(released)%100==0:print('Verified RAM release %d/1313'%len(released),flush=True)
 for d in sorted(ref.iterdir()):
  assert d.is_dir() and not list(d.iterdir());d.rmdir()
 ref.rmdir();s=os.statvfs('/dev/shm');assert s.f_bavail*s.f_frsize>=100*1024**3
 put('PRIOR_RAM_RELEASE_AFTER_DURABLE_ARCHIVE.json',dict(status='PASS_ARCHIVED_ALL_BYTES_THEN_EXACT_RAM_ONLY_RELEASE',archive_root=final,files=len(released),bytes=sum(x['bytes'] for x in released),free_bytes=s.f_bavail*s.f_frsize,science_seal_rewritten=False,GPU_retries=0,old_GPU0_touched=False))
 print('PASS old RAM preserved on29 and released; repair admission may continue.',flush=True)
except BaseException as exc:
 import traceback
 put('PRIOR_RAM_PRESERVATION_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0));raise
