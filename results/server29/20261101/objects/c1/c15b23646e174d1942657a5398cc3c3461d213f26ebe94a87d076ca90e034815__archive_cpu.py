"""CPU-only immutable SHA/bytes publication on29, renameat2 NOREPLACE."""
import base64,hashlib,json,shlex,subprocess
from pathlib import Path
ARCHIVE='/data/zypan/qwen_gdn_pace48_archive/PACE48_LEAP_UNROLL_DP8_DP16_V1'
SSH=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=15','zypan@202.38.247.29']
def publish(root,paths,relative):
    assert len(paths)==len(set(paths)),'Duplicate archive paths rejected before publication'
    root=Path(root);records=[]
    for rel in paths:
        b=(root/rel).read_bytes();records.append(dict(path=str(rel),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
    code='''import base64,ctypes,hashlib,json,os,pathlib,uuid
root=pathlib.Path(ARCHIVE);root.mkdir(parents=True,exist_ok=True)
dest=root/RELATIVE
assert not dest.exists(),'NOREPLACE archive already exists'
tmp=dest.parent/('.incoming_'+uuid.uuid4().hex);tmp.mkdir(parents=True)
rows=RECORDS
assert os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>8*1024**3
for r in rows:
 p=tmp/r['path'];assert '..' not in pathlib.PurePosixPath(r['path']).parts and not pathlib.PurePosixPath(r['path']).is_absolute()
 p.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(r['data'])
 assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
manifest=[{k:v for k,v in r.items() if k!='data'} for r in rows]
with (tmp/'ARCHIVE_MANIFEST.json').open('x') as f:json.dump(manifest,f,sort_keys=True);f.flush();os.fsync(f.fileno())
for p in sorted([x for x in tmp.rglob('*') if x.is_dir()]+[tmp],key=lambda x:len(x.parts),reverse=True):
 fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
dest.parent.mkdir(parents=True,exist_ok=True)
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
assert fn(-100,os.fsencode(tmp),-100,os.fsencode(dest),1)==0,ctypes.get_errno()
fd=os.open(str(dest.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
for r in manifest:
 p=dest/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
print(json.dumps(dict(status='PASS_29_DURABLE_NOREPLACE',root=str(dest),files=len(manifest),bytes=sum(x['bytes'] for x in manifest),manifest_sha256=hashlib.sha256((dest/'ARCHIVE_MANIFEST.json').read_bytes()).hexdigest())))
'''.replace('ARCHIVE',repr(ARCHIVE),1).replace('RELATIVE',repr(relative),1).replace('RECORDS',repr(records),1)
    p=subprocess.run(SSH+['exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -'],input=code,text=True,capture_output=True)
    if p.returncode:raise RuntimeError('CPU archive failed: '+p.stderr[-3000:]+p.stdout[-1000:])
    return json.loads(p.stdout)
