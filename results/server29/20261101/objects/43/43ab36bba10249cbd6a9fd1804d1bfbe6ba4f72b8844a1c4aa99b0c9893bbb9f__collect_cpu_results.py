import json,subprocess,shlex,base64,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'ROOT.json').read_text())
code='''import sys,json,hashlib,base64
from pathlib import Path
R=Path(sys.argv[1]);paths=[]
for p in R.rglob('*'):
 if not p.is_file() or 'inherited' in p.relative_to(R).parts or 'evidence' in p.relative_to(R).parts:continue
 if p.suffix not in ['.json','.log']:continue
 b=p.read_bytes();assert len(b)<4*1024**2
 paths.append(dict(path=str(p.relative_to(R)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),content=base64.b64encode(b).decode()))
print(json.dumps(dict(files=paths)))
'''
p=subprocess.run(['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join([cfg['remote_python'],'-u','-B','-c',code,cfg['remote_root']])],capture_output=True,timeout=60);assert p.returncode==0,p.stderr.decode();v=json.loads(p.stdout);rows=[]
for x in v['files']:
 name=Path(x['path']);assert not name.is_absolute() and '..' not in name.parts;b=base64.b64decode(x['content']);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'];q=R/name;q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==b,'Local/remote existing content differs:'+str(name)
 else:q.write_bytes(b)
 rows.append({k:x[k] for k in ['path','bytes','sha256']})
(R/'analysis/CPU_RESULT_DOWNLOAD_VERIFICATION.json').write_text(json.dumps(dict(status='PASS_ALL_REMOTE_CPU_RESULT_SMALL_SHA',files=rows,count=len(rows),raw_downloads=0,GPU_dispatches=0),indent=2)+'\n');print(json.dumps(dict(status='PASS_CPU_OUTPUT_DOWNLOAD',files=len(rows))))
