"""CPU-only frozen-source bootstrap to a new29 subtree before any GPU launch."""
import json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pace_control as C
from pace_four_gpu_publish import remote,HOST,SOCKET

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and str(ROOT)==C.REMOTE
    C.authorize(ROOT);assert not (ROOT/'SESSION_STARTED.json').exists()
    names=[x['path'] for x in C.read(ROOT/C.SOURCE)['files']]+[C.SOURCE,'GPU_PERMIT.json']
    rows=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=C.sha(ROOT/n)) for n in names];assert len(names)==len(set(names))
    m=dict(status='FROZEN_SOURCE_BOOTSTRAP_ONLY',files=rows,count=len(rows),bytes=sum(x['bytes'] for x in rows),source_manifest_sha256=C.sha(ROOT/C.SOURCE),permit_sha256=C.sha(ROOT/'GPU_PERMIT.json'))
    C.atomic_new(ROOT/'SOURCE_BOOTSTRAP_MANIFEST.json',m);digest=C.sha(ROOT/'SOURCE_BOOTSTRAP_MANIFEST.json')
    final=C.ARCHIVE.replace('/'+C.REVISION,'/LIVE_'+C.REVISION)+'/SOURCE_BOOTSTRAP';stage=final+'.staging_'+digest
    remote('FINAL='+repr(final)+'\nSTAGE='+repr(stage)+'\n'+'''import pathlib,os,json
f=pathlib.Path(FINAL);s=pathlib.Path(STAGE);assert not f.exists() and not s.exists();s.parent.mkdir(parents=True,exist_ok=True);v=os.statvfs(s.parent);assert v.f_bavail*v.f_frsize>=8*1024**3;s.mkdir();print(json.dumps(dict(status='EXCLUSIVE_SOURCE_STAGE_CREATED')))
''')
    listing=ROOT/'operations/BOOTSTRAP_FILE_LIST.txt'
    with listing.open('x') as f:f.write('\n'.join(names+['SOURCE_BOOTSTRAP_MANIFEST.json'])+'\n')
    p=subprocess.run(['rsync','-rlt','--files-from='+str(listing),'-e',shlex.join(['ssh','-S',SOCKET,'-o','BatchMode=yes']),str(ROOT)+'/',HOST+':'+stage+'/'],capture_output=True,timeout=900)
    C.require(p.returncode==0,'Frozen source bootstrap transfer failed: '+p.stderr.decode()[-2000:])
    receipt=remote('FINAL='+repr(final)+'\nSTAGE='+repr(stage)+'\nDIGEST='+repr(digest)+'\n'+'''import pathlib,os,json,hashlib,ctypes
s=pathlib.Path(STAGE);f=pathlib.Path(FINAL);p=s/'SOURCE_BOOTSTRAP_MANIFEST.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==DIGEST;m=json.loads(p.read_bytes())
assert {p.relative_to(s).as_posix() for p in s.rglob('*') if p.is_file()}=={x['path'] for x in m['files']}|{'SOURCE_BOOTSTRAP_MANIFEST.json'}
for x in m['files']:
 p=s/x['path'];a=p.stat();h=hashlib.sha256()
 with p.open('rb') as z:
  for b in iter(lambda:z.read(4194304),b''):h.update(b)
  os.fsync(z.fileno())
 assert p.stat().st_size==a.st_size==x['bytes'] and h.hexdigest()==x['sha256'] and not p.is_symlink()
receipt=dict(status='PASS_SOURCE_BOOTSTRAP29_ALL_SHA_BYTES_NOREPLACE',source_manifest_sha256=m['source_manifest_sha256'],permit_sha256=m['permit_sha256'],manifest_sha256=DIGEST,files=m['count'],bytes=m['bytes'],archive_root=str(f),GPU_calls=0)
with (s/'SOURCE_BOOTSTRAP_RECEIPT.json').open('x') as p:json.dump(receipt,p,sort_keys=True);p.write('\\n');p.flush();os.fsync(p.fileno())
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;assert fn(-100,os.fsencode(s),-100,os.fsencode(f),1)==0,os.strerror(ctypes.get_errno());fd=os.open(f.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(receipt))
''',timeout=900)
    C.atomic_new(ROOT/'SOURCE_BOOTSTRAP_RECEIPT.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
