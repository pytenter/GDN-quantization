#!/usr/bin/env python3
"""Stage and run one separately pinned loader-only engineering observation."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REMOTE=json.loads((ROOT/'ROOT.json').read_text())['remote_root']
PYTHON='/data/ydai/miniconda3/envs/bitdecode/bin/python'
SSH=['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=20','zypan@202.38.247.55']

def raw(v):return json.dumps(v,sort_keys=True,indent=2,allow_nan=False).encode()+bytes([10])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

RECEIVE=r'''
import base64,hashlib,json,os,pathlib,subprocess,sys,tarfile,time
r=pathlib.Path(sys.argv[1]).resolve()
assert not any((r/n).exists() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','CONTROLLER_STARTED.json','ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json'])
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for m in tar:
  rel=pathlib.PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..'not in rel.parts
  p=r/str(rel);assert p.resolve().is_relative_to(r) and not p.is_symlink()
  data=tar.extractfile(m).read()
  if p.exists():assert p.read_bytes()==data, 'Unexpected pre-probe changed staged byte:'+str(rel)
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
freeze=json.loads((r/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json').read_text())
for x in freeze['files']:
 p=r/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
sys.path.insert(0,str(r/'src'));import qualification_guards as g
assert 'torch'not in sys.modules
obs=g._own_gpu(r,1)
target=r/'operations/loader_native_probe';target.mkdir(exist_ok=False)
log=(target/'supervisor.log').open('xb')
p=subprocess.Popen([sys.executable,'-u','-B',str(r/'operations/runtime_loader_native_probe.py'),'--supervise'],stdout=log,stderr=log,start_new_session=True)
identity=g.vendor(r).process_identity(p.pid)
receipt=dict(status='LAUNCHED_ONE_PINNED_ENGINEERING_LOAD_ONLY_PROBE_NOT_QUALIFICATION',supervisor_identity=identity,monotonic=time.monotonic(),NVML_before=obs,
 probe_source_freeze_sha256=sha(r/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json'),probe_permit_sha256=sha(r/'ENGINEERING_LOAD_ONLY_PROBE_PERMIT.json'))
b=json.dumps(receipt,sort_keys=True,indent=2).encode()+bytes([10]);out=r/'operations/ENGINEERING_LOAD_ONLY_PROBE_LAUNCH.json'
with out.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(receipt=receipt,base64=base64.b64encode(b).decode())))
'''

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    selected=set()
    for directory in ('src','vendor'):
        selected.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and not p.is_symlink() and p.suffix=='.py' and '__pycache__'not in p.parts)
    for name in ['scripts/qualify_gpu.py','operations/runtime_loader_native_probe.py',
                 'HUMAN_REPAIR_AUTHORIZATION.json','HUMAN_CONTINUATION_AUTHORIZATION.json','PRIOR_P1_USER_STOP_ADMISSION.json',
                 'operations/P1_AUTHORIZED_STOP_COMPLETION/USER_STOP_RECEIPT.json',
                 'analysis/EXTERNAL_BINDINGS.json','analysis/MODEL_TOKENIZER_FRESH_SHA.json','analysis/RUNTIME_NATIVE_CLOSURE.json']:
        selected.add(ROOT/name)
    closure=json.loads((ROOT/'analysis/RUNTIME_NATIVE_CLOSURE.json').read_text())
    for x in closure['cpu_probe_receipts']:selected.add(ROOT/x['path'])
    copy=ROOT/'provenance/ENGINEERING_PROBE_INPUTS/RUNTIME_NATIVE_CLOSURE_CPU_293.json'
    copy.parent.mkdir(parents=True,exist_ok=True)
    if not copy.exists():copy.write_bytes((ROOT/'analysis/RUNTIME_NATIVE_CLOSURE.json').read_bytes())
    assert copy.read_bytes()==(ROOT/'analysis/RUNTIME_NATIVE_CLOSURE.json').read_bytes()
    selected.add(copy)
    rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)]
    freeze=dict(status='SEPARATELY_PINNED_ENGINEERING_LOADER_PROBE_NO_SCIENTIFIC_QUALIFICATION',files=rows,count=len(rows),
        preserved_CPU_native_manifest='provenance/ENGINEERING_PROBE_INPUTS/RUNTIME_NATIVE_CLOSURE_CPU_293.json',
        model_forward_calls_allowed=0,QDQ_calls_allowed=0,AIME_generations_allowed=0,scientific_qualification=False)
    b=raw(freeze)
    permit=dict(status='ALLOW_ONE_ENGINEERING_LOADER_ONLY_OBSERVATION',source_freeze_sha256=hashlib.sha256(b).hexdigest(),
        repair_authorization_sha256=sha(ROOT/'HUMAN_REPAIR_AUTHORIZATION.json'),gpu_uuid='GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
        model_forward_calls_allowed=0,QDQ_calls_allowed=0,AIME_generations_allowed=0,counts_as_scientific_qualification=False)
    if not args.execute:
        print(json.dumps(dict(files=len(rows),execute=False,permit=permit)));return
    for name,data in [('ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json',b),('ENGINEERING_LOAD_ONLY_PROBE_PERMIT.json',raw(permit))]:
        with (ROOT/name).open('xb')as f:f.write(data)
        selected.add(ROOT/name)
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w')as tar:
        for p in sorted(selected):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    result=subprocess.run(SSH+[shlex.join([PYTHON,'-u','-B','-c',RECEIVE,REMOTE])],input=stream.getvalue(),capture_output=True,timeout=180)
    with (ROOT/'operations/ENGINEERING_LOAD_ONLY_PROBE_LAUNCH_TRANSPORT.json').open('x')as f:
        json.dump(dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),f,indent=2);f.write(chr(10))
    assert result.returncode==0,result.stderr.decode()
    report=json.loads(result.stdout)
    with (ROOT/'operations/ENGINEERING_LOAD_ONLY_PROBE_LAUNCH.json').open('xb')as f:f.write(base64.b64decode(report['base64'],validate=True))
    print(json.dumps(report['receipt']))

if __name__=='__main__':main()
