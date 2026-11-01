"""Before GPU dispatch, seal the static source/calibration snapshot on29 once."""
from pathlib import Path
import importlib.util,json,os

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage',ROOT/'operations/reuse48_stage.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)

def main():
    code='ROOT='+repr(stage.REMOTE)+'\n'+r'''
from pathlib import Path
import sys,json,os,subprocess,shlex,base64,hashlib
r=Path(ROOT);sys.path.insert(0,str(r/'src'));import execution_control as C
from durable_output29 import _configuration
C.authorize(r,r/'GPU_PERMIT.json');assert not (r/'GPU_RUN_STARTED.json').exists()
assert not (r/'operations/BOOTSTRAP29_STARTED.json').exists(),'Bootstrap already attempted; no automatic repeat'
cfg=_configuration(r);C.atomic_new(r/'operations/BOOTSTRAP29_STARTED.json',dict(status='CPU_STATIC_BOOTSTRAP_STARTED_ONCE',binding=cfg,identity=C.process_identity(os.getpid())))
rows=[]
for p in sorted(r.rglob('*')):
 assert not p.is_symlink()
 if not p.is_file() or '__pycache__' in p.parts or p.name=='BOOTSTRAP29_MANIFEST.json':continue
 rows.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=C.sha(p)))
C.atomic_new(r/'BOOTSTRAP29_MANIFEST.json',dict(status='PRE_GPU_STATIC_SOURCE_CALIBRATION_SNAPSHOT',binding=cfg,count=len(rows),bytes=sum(x['bytes'] for x in rows),files=rows))
digest=C.sha(r/'BOOTSTRAP29_MANIFEST.json');cfg=dict(cfg,bootstrap_sha256=digest)
SOCKET='/tmp/qwen_replication_archive29.sock';HOST='zypan@202.38.247.29'
def remote(body):
 code='cfg='+repr(cfg)+'\n'+body
 p=subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,shlex.join(['/usr/bin/python3','-B','-c',code])],capture_output=True,timeout=180)
 C.require(p.returncode==0,'CPU bootstrap29 failed:'+p.stderr.decode(errors='replace')[-2000:]);return json.loads(p.stdout)
reserved=remote(r"""
from pathlib import Path
import json,os
root=Path(cfg['live_root']);assert str(root)=='/data/zypan/qwen_gdn_wscale_rowbal_archive/LIVE_QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_REUSE48_ARCHIVE29_20261025'
assert not root.exists(),'Live root already exists before first bootstrap; preserve and inspect'
s=os.statvfs(root.parent);assert s.f_bavail*s.f_frsize>=120*1024**3
root.mkdir();ident={k:cfg[k] for k in ('task_id','execution_revision','compute_root','live_root','archive_root','source_freeze_sha256','permit_sha256','human_authorization_sha256')}
with (root/'LIVE_IDENTITY.json').open('x') as f:json.dump(ident,f,sort_keys=True,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
(root/'units').mkdir();(root/'staging').mkdir();dest=root/'staging'/('SOURCE_BOOTSTRAP_'+cfg['bootstrap_sha256'][:16]);dest.mkdir()
print(json.dumps(dict(stage=str(dest),final=str(root/'SOURCE_BOOTSTRAP'))))
""")
p=subprocess.run(['rsync','-a','-e',shlex.join(['ssh','-S',SOCKET,'-o','BatchMode=yes']),str(r)+'/',HOST+':'+reserved['stage']+'/'],capture_output=True,timeout=1200)
C.require(p.returncode==0,'CPU bootstrap rsync failed:'+p.stderr.decode(errors='replace')[-2000:])
receipt=remote(r"""
from pathlib import Path
import json,os,hashlib,ctypes
root=Path(cfg['live_root']);stage=root/'staging'/('SOURCE_BOOTSTRAP_'+cfg['bootstrap_sha256'][:16]);final=root/'SOURCE_BOOTSTRAP'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
assert sha(stage/'BOOTSTRAP29_MANIFEST.json')==cfg['bootstrap_sha256']
m=json.loads((stage/'BOOTSTRAP29_MANIFEST.json').read_text());expected={x['path'] for x in m['files']}|{'BOOTSTRAP29_MANIFEST.json'}
actual=set()
for p in stage.rglob('*'):
 assert not p.is_symlink()
 if p.is_file():actual.add(p.relative_to(stage).as_posix())
assert actual==expected
for x in m['files']:
 p=stage/x['path'];assert p.is_file() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
 with p.open('rb') as f:os.fsync(f.fileno())
v=dict(status='PASS_STATIC_SOURCE_CALIBRATION_DURABLE29_BEFORE_GPU',count=m['count'],bytes=m['bytes'],manifest_sha256=cfg['bootstrap_sha256'],source_freeze_sha256=cfg['source_freeze_sha256'],permit_sha256=cfg['permit_sha256'],archive_root=str(final),GPU_calls=0)
with (stage/'BOOTSTRAP29_VERIFICATION.json').open('x') as f:json.dump(v,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
assert fn(-100,os.fsencode(stage),-100,os.fsencode(final),1)==0,os.strerror(ctypes.get_errno())
fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(v))
""")
C.atomic_new(r/'operations/BOOTSTRAP29_VERIFICATION.json',receipt)
print(json.dumps(receipt))
'''
    compile(code,'<prevalidated-bootstrap29>','exec')
    result=stage.transport.run(code,role='55',python=stage.PYTHON,timeout=1500)
    path=ROOT/'operations/BOOTSTRAP29_VERIFICATION_LOCAL.json'
    with path.open('x') as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(result))

if __name__=='__main__':main()
