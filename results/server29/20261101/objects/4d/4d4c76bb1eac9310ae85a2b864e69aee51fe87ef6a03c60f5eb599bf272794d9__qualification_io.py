"""CPU-only durable IO and launch identity helpers; no torch or device API."""
from pathlib import Path
import json,os,hashlib,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return ref(p)
def read(p):return json.loads(Path(p).read_text())
def verify(r,c):
 for rel,digest in c['sources'].items():assert sha(Path(r)/rel)==digest,('FROZEN_SOURCE_MISMATCH',rel)
 return True
def wait(p,*,job,stop,seconds=1800):
 start=time.monotonic()
 while not Path(p).exists():
  if Path(stop).exists():raise RuntimeError('FIRST_FAILURE_LATCHED')
  if time.monotonic()-start>seconds:raise RuntimeError('NO_PROGRESS_WATCHDOG:'+str(p))
  time.sleep(.2)
 return read(p)
def event(p,v):
 b=(json.dumps(dict(monotonic_ns=time.monotonic_ns(),wall_ns=time.time_ns(),**v),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
 try:os.write(fd,b);os.fsync(fd)
 finally:os.close(fd)
