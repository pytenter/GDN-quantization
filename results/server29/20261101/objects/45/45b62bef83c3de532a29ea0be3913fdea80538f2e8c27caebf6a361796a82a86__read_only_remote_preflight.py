#!/usr/bin/env python3
"""Read-only current environment and full frozen dependency verification."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / 'QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'

def main():
    manifest = json.loads((PARENT / 'SOURCE_MANIFEST.json').read_text())
    program = '''import hashlib,json,os,subprocess,time
from pathlib import Path
rows=ROWS
out=[]
for row in rows:
 p=Path(row['path']); h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 actual=dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
 actual['pass']=actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'];out.append(actual)
 if not actual['pass']:break
disks={}
for path in ['/data','/tmp','/home']:
 s=os.statvfs(path);disks[path]={'available_bytes':s.f_bavail*s.f_frsize}
print(json.dumps(dict(status='PASS' if len(out)==len(rows) and all(x['pass'] for x in out) else 'FAIL',checked=len(out),expected=len(rows),files=out,disks=disks,gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True),compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True),unix=time.time()),allow_nan=False))
'''.replace('rows=ROWS', 'rows=' + repr(manifest['external_dependencies']))
    command = ['ssh', '-S', '/tmp/qwen_gdn_replication_20261013.sock', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'zypan@202.38.247.55', '/data/ydai/miniconda3/envs/bitdecode/bin/python -B -']
    completed = subprocess.run(command, input=program, text=True, capture_output=True, timeout=900)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    value = json.loads(completed.stdout)
    value['received_utc'] = datetime.now(timezone.utc).isoformat()
    value['parent_source_manifest_sha256'] = hashlib.sha256((PARENT/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()
    target = ROOT/'analysis'/'REMOTE_PREFLIGHT_FRESH_DEPENDENCIES.json'
    with target.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='files'}, ensure_ascii=False))
    return 0 if value['status']=='PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
