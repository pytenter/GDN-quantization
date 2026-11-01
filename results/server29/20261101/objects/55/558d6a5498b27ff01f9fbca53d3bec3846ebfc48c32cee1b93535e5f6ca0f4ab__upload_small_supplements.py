#!/usr/bin/env python3
"""Upload bounded independent reports/tools; never alters the frozen payload."""
import argparse, base64, hashlib, json, pathlib, shlex, subprocess
ROOT=pathlib.Path(__file__).resolve().parent.parent
REMOTE_ROOT='/data/zypan/analyses/qwen_gdn_historical_runtime_gpu_recovery_v1/gpu_execution_20261019_v1'
RECEIVER=r'''
import base64,hashlib,json,os,pathlib,sys
r=pathlib.Path('/data/zypan/analyses/qwen_gdn_historical_runtime_gpu_recovery_v1/gpu_execution_20261019_v1');v=json.load(sys.stdin);out=[]
for x in v:
 rel=pathlib.PurePosixPath(x['path']);assert not rel.is_absolute()and'..'not in rel.parts and(rel.parts[0]in{'scripts','tests','analysis','qualification','provenance','operations'}or str(rel)in{'INT8_COMPATIBILITY_AUDIT.md','HANDOFF.md'})
 p=r/rel;assert r in p.resolve().parents and not p.is_symlink();b=base64.b64decode(x['data']);assert len(b)==x['bytes']and len(b)<16*1024**2 and hashlib.sha256(b).hexdigest()==x['sha256']
 if p.exists():
  a=p.read_bytes();assert a==b,'Existing supplement differs; do not overwrite';status='EXISTING_EXACT_BYTES_RETAINED'
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  assert p.read_bytes()==b;status='PASS_NEW_SUPPLEMENT_SHA_BYTES'
 out.append(dict(path=str(rel),status=status,bytes=len(b),sha256=x['sha256']))
print(json.dumps(dict(status='PASS_NO_PAYLOAD_SOURCE_MUTATION',files=out)))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');a=p.parse_args();rows=[]
 for name in a.paths:
  rel=pathlib.PurePosixPath(name)
  assert not rel.is_absolute()and '..'not in rel.parts and(rel.parts[0]in{'scripts','tests','analysis','qualification','provenance','operations'}or str(rel)in{'INT8_COMPATIBILITY_AUDIT.md','HANDOFF.md'})
  q=ROOT/rel;assert q.is_file()and not q.is_symlink();b=q.read_bytes();assert len(b)<16*1024**2
  rows.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
 body=json.dumps(rows).encode();assert len(body)<64*1024**2
 cmd=['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','zypan@202.38.247.55',shlex.join(['/data/ydai/miniconda3/envs/bitdecode/bin/python','-B','-c',RECEIVER])]
 r=subprocess.run(cmd,input=body,capture_output=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
 result=json.loads(r.stdout);directory=ROOT/'operations';directory.mkdir(exist_ok=True)
 digest=hashlib.sha256(body).hexdigest();out=directory/('SUPPLEMENT_UPLOAD_'+digest[:20]+'.json')
 if not out.exists():
  with out.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
