import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,sys
known=set(json.load(sys.stdin));base=Path('/data/ydai/miniconda3/envs/bitdecode/lib');out=[];seen={}
paths=sorted(base.glob('libcudart.so*'))
if not paths:out.append(dict(role='conda_CUDA_runtime',directory=str(base),pattern='libcudart.so*',status='MISSING'))
for p in paths:
 if not p.is_file() or str(p) in known:continue
 a=p.stat();target=str(p.resolve())
 if target not in seen:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  seen[target]=h.hexdigest()
 b=p.stat();stable=(a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns) and target==str(p.resolve())
 out.append(dict(role='conda_CUDA_runtime',path=str(p),resolved_path=target,bytes=b.st_size,sha256=seen[target],symlink=p.is_symlink(),status='STABLE_CURRENT_CPU_HASH' if stable else 'UNSTABLE',historical_dynamic_selection='UNKNOWN',prospective_selection='REQUIRES_LOADED_MAP_VERIFICATION_AFTER_AUTHORIZATION',mtime_ns=b.st_mtime_ns,inode=b.st_ino))
print(json.dumps(out))