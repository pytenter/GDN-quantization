"""V3-only orchestration helpers. Parent artifacts are immutable inputs."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = 'LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY'
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
V1 = REPO/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V1'
V2 = REPO/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM'
PARENT = REPO/'experiments/LING_L6_L7_OBJECTIVE_HORIZON_DIAGNOSIS_V1'
MODEL = Path('/data01/user2/models/Ling-3.0-tiny')
LEGACY = ROOT/'inputs/runtime_repo'
TRACES = ROOT/'inputs/traces'
PYTHON = V2/'runtime/env/bin/python'
BASE_HEAD = 'b6ee87c08e2ab8a06c52443177d62ec096dbbf5b'
LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
REGIONS = dict(early=LAYERS[:6],middle=LAYERS[6:12],late=LAYERS[12:])
SOURCE_SHA = 'd1c433598875ea84c80c6f6d4fb5fe48088d5ca0bf44ccff9d950a77560fd9ae'
UUIDS = ['GPU-283734e9-d350-ed3b-a9f2-e3ec1494e186','GPU-936da32c-c539-627e-7f50-bd12fb7e9521']

def load(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def emit(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def command(args):return subprocess.check_output(args,cwd=REPO,text=True).strip()
def tensor_sha(x):
    import torch
    return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def object_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def import_file(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
def trainer():
    p=ROOT/'inputs/historical_trainer.py';assert sha(p)==SOURCE_SHA
    return import_file(p,'v3_immutable_historical_trainer')
def verify_inventory(p):
    inv=load(p)
    for r in inv['files']:
        q=Path(r['path']);assert q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],str(q)
    for r in inv.get('trees',[]):
        actual=[str(q) for q in sorted(Path(r['root']).rglob('*')) if q.is_file() and '__pycache__' not in q.parts and not any(x in q.parts for x in r.get('exclude_parts',[]))]
        assert actual==r['paths'],r['root']
    return {'gate':'PASS','files_checked':len(inv['files']),'inventory_sha256':sha(p)}
def hardware():
    rows=[]
    for line in command(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,driver_version','--format=csv,noheader,nounits']).splitlines():
        a=[x.strip() for x in line.split(',')]
        rows.append(dict(physical_gpu=int(a[0]),gpu_uuid=a[1],name=a[2],used_mib=int(a[3]),total_mib=int(a[4]),driver=a[5]))
    assert len(rows)==2 and [r['gpu_uuid'] for r in rows]==UUIDS and all('4090' in r['name'] for r in rows)
    return dict(host=command(['hostname']),gpus=rows)
def stop(stage,exc):
    p=ROOT/'run/STOP.json'
    if not p.exists():emit(p,dict(gate='FAIL',stage=stage,error=repr(exc),time_unix=time.time(),no_retry=True,no_patch=True))
