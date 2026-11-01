"""Independent read-only collection after real process and NVML release only."""
import io,json,hashlib,shlex,subprocess,tarfile,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'ROOT.json').read_text())
def ssh(code,args=(),payload=None,timeout=120):
 cmd=shlex.join([cfg['remote_python'],'-B','-c',code]+[str(x) for x in args])
 p=subprocess.run(['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',cmd],input=payload,capture_output=True,timeout=timeout)
 assert p.returncode==0,p.stderr.decode()[-4000:]
 return p.stdout
def main():
 remote=cfg['remote_root'];D=R/'delivery';assert not D.exists(),'Collection already exists: inspect instead of overwriting'
 probe=r'''import sys,json,subprocess
from pathlib import Path
r=Path(sys.argv[1]);s=json.loads((r/'execution/SESSION_RESULT.json').read_text());live=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  c=(p/'cmdline').read_bytes().decode().split('\0')
  if any(x.startswith(str(r)+'/scripts/gpu_') for x in c):live.append(dict(pid=int(p.name),cmdline=c))
 except (OSError,UnicodeError):pass
assert not live,'Original controller/worker still running'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);assert p.returncode==0
ids={json.loads(x)['identity']['pid'] for x in (r/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines() if json.loads(x)['event']=='dispatch'}
apps=p.stdout.strip();assert not any(int(x.split(',')[0]) in ids for x in apps.splitlines()),'Owned GPU resource still present'
sys.path.insert(0,str(r/'src'));import gpu_control as C;source=C.verify_sources(r)
print(json.dumps(dict(status='PASS_TERMINAL_REAL_PROCESS_AND_NVML_RELEASE',session=s,actual_NVML_apps=apps,source_freeze_sha256=source,worker_PIDs=sorted(ids))))'''
 v=json.loads(ssh(probe,[remote],timeout=60));(R/'operations/TERMINAL_ADMISSION.json').write_text(json.dumps(v,indent=2)+'\n')
 # Inventory hashes all bounded metadata/source/fixture files, excluding any raw.
 inventory=r'''import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);rows=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(r)
 if any(x in rel.parts for x in ['delivery','FINAL_DELIVERY','__pycache__']):continue
 if p.suffix in ('.bin','.tmp'):continue
 b=p.read_bytes();assert len(b)<4*1024**2,'Large metadata excluded, investigate:'+str(p)
 rows.append(dict(path=str(rel),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(rows))'''
 rows=json.loads(ssh(inventory,[remote],timeout=120));(R/'operations/REMOTE_SMALL_FILE_INVENTORY.json').write_text(json.dumps(rows,indent=2)+'\n')
 stream=r'''import sys,json,tarfile
from pathlib import Path
r=Path(sys.argv[1]);rows=json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for x in rows:t.add(r/x['path'],arcname=x['path'])'''
 raw=ssh(stream,[remote],json.dumps(rows).encode(),120);D.mkdir()
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as t:
  for x in rows:
   name=x['path'];p=D/name;assert not Path(name).is_absolute() and '..' not in Path(name).parts
   b=t.extractfile(name).read();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 print(json.dumps(dict(status='PASS_EXACT_SMALL_DOWNLOAD',files=len(rows),bytes=sum(x['bytes'] for x in rows),local_delivery=str(D))))
if __name__=='__main__':main()
