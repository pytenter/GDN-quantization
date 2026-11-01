import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,sys,hashlib,base64
out=[]
for name in json.load(sys.stdin)['paths']:
 p=Path(name)
 if not p.is_file():out.append(dict(remote=name,status='MISSING'));continue
 a=p.stat()
 if a.st_size>8000000:out.append(dict(remote=name,status='SKIPPED_SIZE_LIMIT',bytes=a.st_size));continue
 raw=p.read_bytes();b=p.stat()
 stable=(a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
 if not stable:out.append(dict(remote=name,status='UNKNOWN_UNSTABLE'));continue
 out.append(dict(remote=name,status='STABLE_READ',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime_ns=b.st_mtime_ns,inode=b.st_ino,data=base64.b64encode(raw).decode()))
print(json.dumps(out))