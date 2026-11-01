"""New surrogate experiment only; all historical inputs are read-only."""
from pathlib import Path
import hashlib, importlib.util, json, os, platform, subprocess, sys, time

TASK = 'LING_PROPAGATION_SURROGATE_VALIDATION_V1'
D = Path(__file__).resolve().parents[1]
R = D.parents[1]
P = Path('/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1')
OLD = Path('/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1')
V4 = P/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V4_DETERMINISTIC'
V3 = P/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY'
ENV = P/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM/runtime/env'
PYTHON = ENV/'bin/python'
MODEL = Path('/data01/user2/models/Ling-3.0-tiny')
LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
UUIDS = ['GPU-283734e9-d350-ed3b-a9f2-e3ec1494e186','GPU-936da32c-c539-627e-7f50-bd12fb7e9521']
PARENTS = [OLD/'experiments/LING_L6_L7_TF_REPLAY_MECHANISM_V1',
 OLD/'experiments/LING_L6_L7_DATAADAPTIVE_ROTATION_V1',
 P/'experiments/LING_L6_L7_OBJECTIVE_HORIZON_DIAGNOSIS_V1',
 P/'experiments/LING_LAYERWISE_PROPAGATION_FRESH_CONFIRM_V1', V4]

def load(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def row(p):
    p=Path(p); return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p))
def emit(p,x):
    p=Path(p); p=p if p.is_absolute() else D/p
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(x,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
def command(a): return subprocess.check_output(a,cwd=R,text=True).strip()
def imp(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s)
    sys.modules[n]=m;s.loader.exec_module(m);return m
def files(root): return sorted(p for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
def tensor_sha(x):
    import torch
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def verify_parent():
    x=load(D/'provenance/parent_inventory_before.json')
    for t in x['trees']: assert [str(p) for p in files(t['root'])]==t['paths'], t['root']
    for f in x['files']: assert row(f['path'])==f, f['path']
    return dict(gate='PASS',files_checked=len(x['files']),inventory_sha256=sha(D/'provenance/parent_inventory_before.json'))
def runtime():
    import importlib.metadata
    manifest=V4/'runtime/v4_deterministic_runtime_manifest.json';x=load(manifest)
    assert x['qualification']=='PASS'
    expected=x['package_environment']['current'];got={p:importlib.metadata.version(p) for p in expected['packages']}
    assert got==expected['packages'] and sys.version==expected['python']
    return dict(gate='PASS',authority=row(manifest),python=sys.version,packages=got)
def hardware():
    out=[]
    for l in command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader,nounits']).splitlines():
        a=[s.strip() for s in l.split(',')];out.append(dict(physical_gpu=int(a[0]),gpu_uuid=a[1],name=a[2],total_mib=int(a[3]),used_mib=int(a[4]),driver=a[5]))
    assert len(out)==2 and [x['gpu_uuid'] for x in out]==UUIDS and all('4090' in x['name'] for x in out)
    return dict(host=command(['hostname']),gpus=out)
def stop(stage,e):
    import traceback
    f=D/'run/STOP.json'
    if not f.exists():emit(f,dict(gate='FAIL',stage=stage,error=repr(e),traceback=traceback.format_exc(),time_unix=time.time(),no_auto_retry=True,NEW_TRAINING='NO',NEW_AIME_GENERATION=0))

