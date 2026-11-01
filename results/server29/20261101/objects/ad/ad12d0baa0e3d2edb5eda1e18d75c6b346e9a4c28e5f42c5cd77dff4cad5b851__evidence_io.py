"""Audit-only stable read/copy transport. No GPU/runtime imports or mutations."""
import base64, fcntl, hashlib, json, os, shlex, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
AIME='/data/zypan/analyses/qwen_gdn_hadamard_dual_m_aime26_formal_60_v1/repair_sampler_scope_v1'
PARENT='/data/zypan/analyses/qwen_gdn_hadamard_dual_m_aime26_formal_60_v1'
FUTURE='/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
C6R='/data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1'
HIST='/data/zypan/analyses/qwen_gdn_historical_throughput_reconciliation_v1'
SPEED='/data/zypan/analyses/qwen_gdn_dual_m_equivalent_speedup_v1'
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S','/tmp/qwen_gdn_replication_20261013.sock','zypan@202.38.247.55']
def digest(raw):return hashlib.sha256(raw).hexdigest()
def event(value):
 with (ROOT/'READ_ONLY_ACTIONS.jsonl').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(dict(unix=time.time(),**value),ensure_ascii=False)+'\n');f.flush()
def call(code,payload=None):
 pre="import os,sys\nos.environ['CUDA_VISIBLE_DEVICES']=''\nos.environ['PYTHONDONTWRITEBYTECODE']='1'\nsys.dont_write_bytecode=True\n"
 script=pre+code;compile(script,'audit_read_only','exec')
 key=digest(script.encode());p=ROOT/'scripts'/('remote_'+key+'.py')
 if not p.exists():p.write_text(script)
 result=subprocess.run(SSH+[shlex.join([PY,'-B','-c',script])],input=None if payload is None else json.dumps(payload).encode(),capture_output=True)
 event(dict(kind='CPU_STDLIB_REMOTE_READ',script=str(p.relative_to(ROOT)),script_sha256=key,returncode=result.returncode))
 if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
 return result.stdout
def fetch(paths):
 data=json.loads(call('''from pathlib import Path
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
print(json.dumps(out))''',{'paths':paths}))
 for x in data:
  if x['status']!='STABLE_READ':event(dict(kind='EVIDENCE_READ',**x));continue
  raw=base64.b64decode(x.pop('data'));assert digest(raw)==x['sha256']
  p=ROOT/'evidence'/digest(x['remote'].encode())/Path(x['remote']).name;p.parent.mkdir(exist_ok=True)
  if p.exists():assert p.read_bytes()==raw,'Evidence changed; keep first snapshot'
  else:p.write_bytes(raw)
  x['local']=str(p);event(dict(kind='EVIDENCE_READ',**x))
 return data
def local_for(remote):return ROOT/'evidence'/digest(remote.encode())/Path(remote).name
