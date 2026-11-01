"""Synchronous CPU-only closed-unit publication to29, with atomic NOREPLACE."""
import json,os,shlex,subprocess
from pathlib import Path
import pace_control as C

HOST='zypan@202.38.247.29'
SOCKET='/tmp/qwen_replication_archive29.sock'

def remote(code,timeout=300):
    p=subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,
                      'env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -'],
                     input=code,text=True,capture_output=True,timeout=timeout)
    C.require(p.returncode==0,'Archive29 CPU/connection failure: '+p.stderr[-4000:])
    return json.loads(p.stdout)

def publish_unit(root,target,interval):
    root=Path(root);target=Path(target)
    C.require(str(root.resolve())==C.REMOTE and target.resolve().is_relative_to(root.resolve()),'Publisher may write only own exact execution subtree')
    C.require(isinstance(interval['unit'],str) and interval['unit'].replace('_','').isalnum() and interval['stage'] in ('formal','scale_repair_gates'),'Safe fixed unit/stage required')
    C.require(interval['released'] and type(interval['returncode']) is int,'Closed actual exit/release required before publication')
    rows=[]
    for p in sorted(target.rglob('*')):
        if p.is_file() and p.name not in ('DURABLE29_MANIFEST.json','DURABLE29_RECEIPT.json'):
            C.require(not p.is_symlink(),'Unit symlink forbidden')
            rows.append(dict(path=p.relative_to(target).as_posix(),bytes=p.stat().st_size,sha256=C.sha(p)))
    manifest=dict(unit=interval['unit'],stage=interval['stage'],files=rows,count=len(rows),bytes=sum(x['bytes'] for x in rows),
                  source_manifest_sha256=C.sha(root/C.SOURCE),permit_sha256=C.sha(root/'GPU_PERMIT.json'),actual_exit_returncode=interval['returncode'],released=True)
    C.atomic_new(target/'DURABLE29_MANIFEST.json',manifest)
    digest=C.sha(target/'DURABLE29_MANIFEST.json')
    final=C.ARCHIVE.replace('/'+C.REVISION,'/LIVE_'+C.REVISION)+'/units/'+interval['unit']
    stage=final+'.staging_'+digest[:16]
    remote('FINAL='+repr(final)+'\nSTAGE='+repr(stage)+'\n'+'''import pathlib,json,os
f=pathlib.Path(FINAL);s=pathlib.Path(STAGE);assert not f.exists() and not s.exists();s.parent.mkdir(parents=True,exist_ok=True)
v=os.statvfs(s.parent);assert v.f_bavail*v.f_frsize>=8*1024**3;s.mkdir()
print(json.dumps(dict(status='STAGE_CREATED_ONCE')))
''')
    shell=shlex.join(['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=15'])
    p=subprocess.run(['rsync','-a','-e',shell,str(target)+'/',HOST+':'+stage+'/'],capture_output=True,timeout=900)
    C.require(p.returncode==0,'Closed-unit CPU transfer failure: '+p.stderr.decode()[-4000:])
    receipt=remote('FINAL='+repr(final)+'\nSTAGE='+repr(stage)+'\nDIGEST='+repr(digest)+'\n'+'''import pathlib,json,hashlib,os,ctypes
s=pathlib.Path(STAGE);f=pathlib.Path(FINAL);p=s/'DURABLE29_MANIFEST.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==DIGEST;m=json.loads(p.read_bytes())
expected={x['path'] for x in m['files']}|{'DURABLE29_MANIFEST.json'};assert {p.relative_to(s).as_posix() for p in s.rglob('*') if p.is_file()}==expected
for x in m['files']:
 p=s/x['path'];a=p.stat();h=hashlib.sha256()
 with p.open('rb') as q:
  for b in iter(lambda:q.read(8*1024**2),b''):h.update(b)
 z=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns) and z.st_size==x['bytes'] and h.hexdigest()==x['sha256']
receipt=dict(status='PASS_DURABLE29_CLOSED_UNIT',unit=m['unit'],actual_exit_returncode=m['actual_exit_returncode'],released=True,manifest_sha256=DIGEST,source_manifest_sha256=m['source_manifest_sha256'],permit_sha256=m['permit_sha256'],files=m['count'],bytes=m['bytes'],final_root=str(f))
with (s/'DURABLE29_RECEIPT.json').open('x') as p:json.dump(receipt,p,sort_keys=True);p.write('\\n');p.flush();os.fsync(p.fileno())
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
assert fn(-100,os.fsencode(s),-100,os.fsencode(f),1)==0,os.strerror(ctypes.get_errno());fd=os.open(f.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(receipt))
''',timeout=900)
    C.atomic_new(target/'DURABLE29_RECEIPT.json',receipt)
    return receipt
