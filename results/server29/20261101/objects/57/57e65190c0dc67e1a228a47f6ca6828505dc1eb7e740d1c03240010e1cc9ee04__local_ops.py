"""Local CPU-only SSH transport and raw-byte snapshots; no credentials."""
from pathlib import Path
import base64, hashlib, json, shlex, subprocess
BASE=Path(__file__).resolve().parents[1]
REMOTE='/data/zypan/analyses/qwen_gdn_hadamard_dual_m_aime26_formal_60_v1/repair_sampler_scope_v1'
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S','/tmp/qwen_gdn_replication_20261013.sock','zypan@202.38.247.55']
def call(code, payload=None):
    compile(code,'remote','exec')
    a=subprocess.run(SSH+[shlex.join([PY,'-B','-c',code])],input=None if payload is None else json.dumps(payload).encode(),capture_output=True)
    if a.returncode:raise RuntimeError(a.stderr.decode(errors='replace'))
    return a.stdout
def fetch(paths):
    rows=json.loads(call('''from pathlib import Path
import sys,json,hashlib,base64
out=[]
for name in json.load(sys.stdin)['paths']:
 p=Path(name);raw=p.read_bytes();assert len(raw)<2000000
 out.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
print(json.dumps(out))''',{'paths':paths}))
    cat_path=BASE/'CPU_SNAPSHOT_CATALOG.json'
    cat=json.loads(cat_path.read_text()) if cat_path.exists() else []
    for row in rows:
        raw=base64.b64decode(row.pop('data'));assert hashlib.sha256(raw).hexdigest()==row['sha256']
        p=BASE/'snapshots'/row['sha256']/Path(row['path']).name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert p.read_bytes()==raw
        else:p.write_bytes(raw)
        row['local_path']=str(p)
        if row not in cat:cat.append(row)
    cat_path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n')
    return rows
def upload(rel, raw):
    return call('''from pathlib import Path
import json,sys,base64,hashlib
j=json.load(sys.stdin);p=Path(j['path']);raw=base64.b64decode(j['data']);p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():assert p.read_bytes()==raw,'Refuse overwrite: '+str(p)
else:
 with p.open('xb') as f:f.write(raw)
print(hashlib.sha256(p.read_bytes()).hexdigest())''',{'path':REMOTE+'/'+rel,'data':base64.b64encode(raw).decode()}).decode().strip()
