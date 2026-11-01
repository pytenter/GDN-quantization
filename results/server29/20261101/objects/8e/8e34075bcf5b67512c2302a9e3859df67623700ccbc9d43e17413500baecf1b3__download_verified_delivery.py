#!/usr/bin/env python3
"""Independent one-shot CPU read-only waiter, verifier and small-file delivery.

No GPU dispatch, remote mutation, scientific retry, finisher invocation or raw
Tensor/model download. Only readiness SSH255/timeouts are retried. Any actual
verification or transfer error is preserved and fails closed without retry.
"""
from __future__ import annotations
import argparse
import base64
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

TASK = "QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1"
ATTEMPT = TASK + "_ENGINEERING_REPAIR_20261022"
TOOL_MANIFEST = "operations/READONLY_DELIVERY_TOOL_MANIFEST.json"
CONTROLS = ("SHA256_MANIFEST.json", "SHA256SUMS", "FINAL_ARCHIVE_VERIFICATION.json", "FINAL_SEAL_RECEIPT.json")
RAW_SUFFIXES = (".bin", ".npz", ".npy", ".pt", ".pth", ".f32", ".safetensor", ".safetensors", ".ckpt", ".onnx", ".gguf", ".pyc", ".pyo")
STARTED = "operations/LOCAL_DOWNLOADER_STARTED.json"
FAILURE = "operations/LOCAL_DOWNLOADER_FAILURE.json"
SSH29 = ["ssh", "-S", "/tmp/qwen_replication_archive29.sock", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", "zypan@202.38.247.29"]


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())


def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_new(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())


def save(root, name, value): write_new(Path(root) / name, canonical(value))


def safe_relative(name):
    require(isinstance(name, str) and name and not PurePosixPath(name).is_absolute()
        and str(PurePosixPath(name)) == name and all(x not in (".", "..", "") for x in name.split("/"))
        and not any(x in name for x in ("\n", "\r", "\0")), "Unsafe relative science filename")
    return Path(name)


def raw_path(name): return str(name).lower().endswith(RAW_SUFFIXES)


def ssh55(config):
    return ["ssh", "-S", config["ssh_control_path"], "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", "zypan@202.38.247.55"]


def command(config, role, *, code_argument=None):
    python = config["remote_python"] if role == "55" else "/usr/bin/python3"
    invocation = ["env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", python, "-B"]
    invocation += ["-"] if code_argument is None else ["-c", code_argument]
    remote = "exec " + " ".join(shlex.quote(x) for x in invocation)
    if role == "29": remote = "exec " + " ".join(shlex.quote(x) for x in SSH29 + [remote])
    return ssh55(config) + [remote]


class ConnectionPending(RuntimeError): pass


def remote_json(config, role, code, *, readiness=False):
    try:
        result = subprocess.run(command(config, role), input=code, text=True, capture_output=True,
                                timeout=50 if readiness else None)
    except subprocess.TimeoutExpired as error:
        if readiness: raise ConnectionPending("Readiness connection timeout") from error
        raise
    if readiness and result.returncode == 255:
        raise ConnectionPending("Readiness SSH255: " + result.stderr[-3000:])
    require(result.returncode == 0, "Read-only verification/transport failed; no automatic retry: " + role + ":" + result.stderr[-6000:] + result.stdout[-1000:])
    return json.loads(result.stdout)


REMOTE_HELPERS = r'''
from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,sys,time
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe(n):
 p=Path(n);assert isinstance(n,str) and n and not p.is_absolute() and str(p)==n and all(x not in ('.','..','')for x in n.split('/'))
 assert not any(x in n for x in ('\n','\r','\0'));return p
def stable(r,row):
 p=r/safe(row['path']);assert p.resolve().is_relative_to(r.resolve())and not p.is_symlink()and p.is_file()
 a=p.stat();h=sha(p);b=p.stat()
 assert(a.st_size,a.st_ino,a.st_mtime_ns)==(b.st_size,b.st_ino,b.st_mtime_ns)
 assert b.st_size==row['bytes']and h==row['sha256'];return dict(path=row['path'],bytes=b.st_size,sha256=h)
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
  if f[0]in('Z','X','x'):return None
  return dict(pid=int(pid),starttick=int(f[19]),cmdline=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
 except(FileNotFoundError,ProcessLookupError):return None
 except PermissionError:
  assert(Path('/proc')/str(pid)).stat().st_uid!=os.geteuid(),'Own-user process identity unreadable'
  return None
def belongs(x,r):return x is not None and any(a==str(r)or a.startswith(str(r)+'/')for a in x['cmdline'])
def execution_process(x,r):return x is not None and any(a.startswith(str(r/'scripts')+'/')for a in x['cmdline'])
def inventory(r,expected=None,allow_extras=False):
 found=set();dirs=set();allowed={'.'}
 if expected is not None:
  for n in expected:
   safe(n);allowed.update(str(p)for p in Path(n).parents)
 for p in r.rglob('*'):
  assert not p.is_symlink(),'Symlink in science root'
  n=str(p.relative_to(r))
  if p.is_file():found.add(n)
  else:assert p.is_dir();dirs.add(n)
 if expected is not None and not allow_extras:
  assert found==set(expected),'Unknown/missing sealed file';assert dirs<=allowed,'Unknown sealed directory'
 return found,dirs
def bound_evidence(root,row):
 name=row['path'];rel=Path(name)
 if rel.is_absolute():
  owners=(root,globals().get('remote_root',root));matched=False
  for owner in owners:
   if rel.is_relative_to(owner):rel=rel.relative_to(owner);matched=True;break
  assert matched,'Evidence path escapes the exact task root'
 rel=safe(str(rel));p=root/rel;assert p.resolve().is_relative_to(root.resolve())and not p.is_symlink()and p.is_file()
 h=sha(p);size=p.stat().st_size;assert h==row['sha256']
 if 'bytes'in row:assert size==row['bytes']
 return dict(path=str(rel),bytes=size,sha256=h),read(p)
def exited_probe(expected):
 assert isinstance(expected,dict)and type(expected['pid'])is int and expected['pid']>0 and type(expected['starttick'])is int and expected['starttick']>0
 assert isinstance(expected.get('cmdline'),(str,list))and expected['cmdline']
 current=identity(expected['pid'])
 assert current is None or current['starttick']!=expected['starttick'],'Probe PID/starttick remains alive even if argv representation differs'
 return dict(expected=expected,current=current,exact_identity_exited=True)
def native_contract(root):
 path=root/'analysis/RUNTIME_NATIVE_CLOSURE.json';assert path.is_file()and not path.is_symlink()
 m=read(path);assert m['schema_version']==1 and m['status']=='PASS_TWO_FRESH_CPU_IMPORTS_AND_PROVENANCED_NATIVE_CLOSURE_NO_GPU'
 assert m['base_external_bindings_sha256']==sha(root/'analysis/EXTERNAL_BINDINGS.json')
 rows=m['files'];assert rows and len(rows)==m['count']==len({x['path']for x in rows})and sum(x['bytes']for x in rows)==m['bytes']
 for x in rows:
  assert Path(x['path']).is_absolute()and all(type(x[k])is int and x[k]>=0 for k in('bytes','inode','device','mtime_ns'))and x['inode']>0
  assert len(x['sha256'])==64 and all(c in'0123456789abcdef'for c in x['sha256'])and x['provenance']
 probes=m['cpu_probe_receipts'];assert len(probes)==len({x['path']for x in probes})==2
 receipts=[]
 for x in probes:
  meta,record=bound_evidence(root,x)
  assert x['CPU_only']is True and x['CUDA_initialized']is False and x['CPU_PID_absent_from_NVML']is True
  assert record['CUDA_VISIBLE_DEVICES']==''and record['Torch_CUDA_initialized']is False and record['CPU_PID_absent_from_NVML']is True
  assert all(record[k]==0 for k in('GPU_contexts_created','GPU_dispatches','model_instances_created','Tensor_forwards'))
  assert record['identity']==x['identity'];receipts.append(dict(file=meta,identity=x['identity'],scope='CPU_ONLY_NO_MODEL_NO_FORWARD_NO_CUDA'))
 assert receipts[0]['identity']!=receipts[1]['identity']
 extension=m.get('loader_only_extension');extension_evidence=None
 if extension is not None:
  meta,value=bound_evidence(root,extension)
  assert value['status']=='PASS_EXACT_LOADER_PROBE_ELF_PROVENANCE_CPU_REVIEW'
  assert value['base_external_bindings_sha256']==m['base_external_bindings_sha256']
  assert extension['probe_receipt']==value['probe_receipt']
  probe_meta,_=bound_evidence(root,extension['probe_receipt'])
  extension_evidence=dict(extension=meta,probe_receipt=probe_meta,additional_file_count=extension['additional_file_count'])
 return m,dict(manifest_sha256=sha(path),count=len(rows),bytes=sum(x['bytes']for x in rows),CPU_receipts=receipts,loader_extension=extension_evidence)
def fresh_native_files(root,manifest,contract,cache):
 import stat
 actual=[]
 for x in manifest['files']:
  p=Path(x['path']);assert str(p.resolve())==str(p)and not p.is_symlink()
  a=p.stat();statrow=dict(bytes=a.st_size,inode=a.st_ino,device=a.st_dev,mtime_ns=a.st_mtime_ns)
  assert stat.S_ISREG(a.st_mode)and all(statrow[k]==x[k]for k in statrow)
  with p.open('rb')as stream:magic=stream.read(4)
  assert magic==b'\x7fELF','Closure contains a non-ELF row'
  old=cache.get(str(p));reused=old is not None and all(old[k]==statrow[k]for k in statrow)
  h=old['sha256']if reused else sha(p);b=p.stat()
  assert(a.st_size,a.st_ino,a.st_dev,a.st_mtime_ns)==(b.st_size,b.st_ino,b.st_dev,b.st_mtime_ns)and h==x['sha256']
  row=dict(path=str(p),**statrow,sha256=h,ELF_magic=magic.hex(),fresh_SHA_reused_from_this_audit_original61=reused)
  actual.append(row);cache[str(p)]=row
 aliases=manifest.get('aliases',[]);assert len(aliases)==len({x['path']for x in aliases});observed=[];allowed={x['path']for x in manifest['files']}
 for x in aliases:
  p=Path(x['path']);resolved=str(p.resolve());assert resolved==x['canonical_path']and resolved in allowed
  observed.append(dict(path=str(p),canonical_path=resolved,match=True))
 exits=[exited_probe(x['identity'])for x in contract['CPU_receipts']]
 return dict(status='PASS_DYNAMIC_COMPLETE_FROZEN_NATIVE_ELF_FRESH_SHA_STAT_BYTES',manifest_sha256=contract['manifest_sha256'],count=len(actual),bytes=sum(x['bytes']for x in actual),files=actual,aliases=observed,CPU_probe_exit=exits,loader_extension=contract['loader_extension'],scope='EVERY_EXACT_FROZEN_ELF_ROW_DYNAMIC_COUNT_NO_DIRECTORY_ALLOW')
def loader_probe_scope(root,by_path,native_sha,native_count):
 import math
 completion_path='operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json';review_path='operations/ENGINEERING_NATIVE_MAPS_FINAL_REVIEW.json'
 assert completion_path in by_path and review_path in by_path,'Loader-only probe evidence omitted from science archive'
 compmeta,c=bound_evidence(root,by_path[completion_path]);reviewmeta,review=bound_evidence(root,by_path[review_path])
 assert c['status']=='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED'and c['actual_wait_returncode']==0
 assert c['all_exact_own_processes_exited']is True and c['own_GPU_UUID_released']is True and c['counts_as_scientific_qualification']is False
 assert c['model_forward_calls']==c['QDQ_calls']==c['AIME_generations']==0
 worker=c['worker_identity'];exitproof=exited_probe(worker)
 resultmeta,result=bound_evidence(root,c['result'])
 pins={}
 for key in('probe_source_freeze_sha256','probe_permit_sha256'):
  digest=c[key];assert len(digest)==64
  hits=[dict(path=x['path'],bytes=x['bytes'],sha256=x['sha256'])for x in by_path.values()if x['sha256']==digest]
  assert hits,'Original exact probe sourcefreeze/permit bytes omitted from final science archive'
  for hit in hits:bound_evidence(root,hit)
  pins[key]=hits
 assert review['status']=='PASS_ALL_LOAD_ONLY_NATIVE_MAP_STAGES_EXACTLY_PINNED'
 assert review['native_manifest_sha256']==native_sha and review['completion_sha256']==compmeta['sha256']and review['actual_fresh_native_closure_count']==native_count
 stages=review['stages'];assert stages and len(stages)==len({x['path']for x in stages})
 stageproof=[]
 for stage in stages:
  assert stage['assessment_status']=='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED'and type(stage['mapped_native_count'])is int and stage['mapped_native_count']>=0
  meta,_=bound_evidence(root,stage);assert meta['path']in by_path
  stageproof.append(dict(file=meta,mapped_native_count=stage['mapped_native_count'],assessment_status=stage['assessment_status']))
 cost=c['resource_interval'];assert cost['monotonic_same_host']is True
 keys=('dispatch_monotonic_lower','identity_monotonic_upper','exit_first_observed_monotonic','release_first_observed_monotonic')
 values=[cost[k]for k in keys];assert all(type(v)in(int,float)and math.isfinite(v)for v in values)and values==sorted(values)
 upper=values[-1]-values[0]
 return dict(status='PASS_LOADER_ONLY_ENGINEERING_PROBE_EXIT_RELEASE_SCOPE_AND_OBSERVED_COST',completion=compmeta,review=reviewmeta,worker_identity=worker,exit=exitproof,result=resultmeta,source_and_permit=pins,native_map_stages=stageproof,resource_interval=cost,observed_dispatch_to_release_upper_seconds=upper,observed_dispatch_to_release_upper_GPUh=upper/3600,EXACT_physical_GPUh='UNRESOLVED',cost_scope='ENGINEERING_LOADER_ONLY_SEPARATE_FROM_58_SCIENTIFIC_QUALIFICATION_AND_60_FORMAL',scientific_qualification_PASS_from_probe=0,formal_generations_from_probe=0,model_forward_calls=0,QDQ_calls=0,AIME_generations=0)

'''


def readiness_code(config, source_sha):
    return REMOTE_HELPERS + "\nr=Path(" + repr(config["remote_root"]) + ");source_sha=" + repr(source_sha) + "\n" + r'''
out=dict(status='WAIT_READONLY_RUNNING_OR_NOT_STARTED',monotonic=time.monotonic())
if (r/'SOURCE_FREEZE.json').exists():assert sha(r/'SOURCE_FREEZE.json')==source_sha,'Remote frozen source identity changed'
else:raise AssertionError('Bound source freeze missing')
for n in ('FIRST_FAILURE.json','FINISHER_FAILURE.json','SESSION_RESULT.json','CONTROLLER_STARTED.json','FINISHER_STARTED.json','FINISHER_RESULT.json'):
 if(r/n).exists():out[n]=read(r/n)
if 'FINISHER_FAILURE.json'in out:raise AssertionError('Real finisher failure retained; independent CPU recovery required')
controller=out.get('CONTROLLER_STARTED.json');finisher=out.get('FINISHER_STARTED.json')
if controller:
 assert isinstance(controller.get('identity'),dict),'Controller exact identity missing'
 out['controller_exact_alive']=identity(controller['identity']['pid'])==controller['identity']
 if not out['controller_exact_alive']and 'SESSION_RESULT.json'not in out:raise AssertionError('Controller exited without SESSION; never fabricate or retry')
if finisher:
 assert isinstance(finisher.get('identity'),dict),'Finisher exact identity missing'
 out['finisher_exact_alive']=identity(finisher['identity']['pid'])==finisher['identity']
 if not out['finisher_exact_alive']and 'FINISHER_RESULT.json'not in out:raise AssertionError('Finisher exited without real result; no automatic retry')
if all((r/n).is_file()for n in ('SESSION_RESULT.json','CONTROLLER_STARTED.json','FINISHER_STARTED.json','FINISHER_RESULT.json','FINAL_SEAL_RECEIPT.json')):
 if not out['controller_exact_alive']and not out['finisher_exact_alive']:
  assert out['FINISHER_RESULT.json']['status']=='PASS_REAL_TERMINAL_SCIENCE_ARCHIVED_ON29'
  out['status']='READY_TRUE_CONTROLLER_AND_FINISHER_EXITED_AND_SEALED'
print(json.dumps(out))
'''


def audit_code(config, source_sha, role):
    remote = config["remote_root"]; archive = config["archive_root"]
    prefix = REMOTE_HELPERS + "\nr=Path(" + repr(remote if role == "55" else archive) + ")\n"
    prefix += "remote_root=Path(" + repr(remote) + ");archive_root=" + repr(archive) + ";source_sha=" + repr(source_sha) + ";role=" + repr(role) + "\n"
    prefix += "task=" + repr(TASK) + ";controls=" + repr(CONTROLS) + ";raw_suffixes=" + repr(RAW_SUFFIXES) + "\n"
    return prefix + r'''
assert r.is_dir()and not r.is_symlink()
control_data={n:(r/n).read_bytes()for n in controls};m=json.loads(control_data['SHA256_MANIFEST.json']);seal=json.loads(control_data['FINAL_SEAL_RECEIPT.json']);v=json.loads(control_data['FINAL_ARCHIVE_VERIFICATION.json'])
assert m['task_id']==seal['task_id']==task and m['execution_attempt']==seal['execution_attempt']==remote_root.name
assert seal['status']=='PASS_ATOMIC_NOREPLACE_SEALED_SCIENCE'and seal['atomic']=='renameat2_RENAME_NOREPLACE'and seal['archive_root']==archive_root
assert m['source_freeze_sha256']==seal['source_freeze_sha256']==v['source_freeze_sha256']==source_sha==sha(r/'SOURCE_FREEZE.json')
assert seal['manifest_sha256']==v['manifest_sha256']==hashlib.sha256(control_data['SHA256_MANIFEST.json']).hexdigest()
assert seal['SHA256SUMS_sha256']==hashlib.sha256(control_data['SHA256SUMS']).hexdigest()
assert seal['archive_verification_sha256']==hashlib.sha256(control_data['FINAL_ARCHIVE_VERIFICATION.json']).hexdigest()
assert v['status']=='PASS_INDEPENDENT_ARCHIVE29_EVERY_FILE_FRESH_SHA_AND_BYTES'
assert seal['terminal']==m['terminal']==v['terminal']and seal['formal_completed']==m['formal_completed']and seal['planned_formal']==60
rows=m['files'];names={x['path']for x in rows}
assert len(rows)==len(names)==m['count']and sum(x['bytes']for x in rows)==m['bytes']<=20*1024**3
assert not names&set(controls)
assert control_data['SHA256SUMS']==''.join(x['sha256']+'  '+x['path']+'\n'for x in rows).encode()
actual=[stable(r,row)for row in rows]
assert actual==v['files']and len(actual)==v['count']and sum(x['bytes']for x in actual)==v['bytes']
source=read(r/'SOURCE_FREEZE.json')['files'];assert source and len(source)==len({x['path']for x in source})
by_path={x['path']:x for x in actual}
for x in source:assert x['path']in by_path and all(x[k]==by_path[x['path']][k]for k in ('path','bytes','sha256'))
assert 'analysis/RUNTIME_NATIVE_CLOSURE.json'in{z['path']for z in source},'Native closure must be in the final scientific sourcefreeze'
native_manifest,native_contract_proof=native_contract(r)
raw=[x for x in actual if x['path'].lower().endswith(raw_suffixes)]
archiver_raw=[x for x in actual if Path(x['path']).suffix.lower()in{'.bin','.npz','.npy','.pt','.pth','.f32','.safetensors'}]
assert archiver_raw==m['raw_remote_only_files']and len(archiver_raw)==m['raw_remote_only_count']
session=read(r/'SESSION_RESULT.json');task_result=read(r/'TASK_RESULT.json')
assert session['status']==task_result['status']==m['terminal']and session['formal_completed']==m['formal_completed']==task_result['actual_formal_completed']
assert session['own_release_all_PASS']is True and session['all_own_processes_released']is True
assert len(session['formal_receipts'])==session['formal_completed']and len({x['unit']for x in session['formal_receipts']})==session['formal_completed']
if m['terminal']=='RESIDENT_ROTATION_QUALIFIED_FORMAL_COMPLETE_60_60_AND_STOP':assert m['formal_completed']==60
else:assert m['terminal']in('FAIL_CLOSED_STOP','RESIDENT_ROTATION_MATH_FAIL_CLOSED_STOP','RESOURCE_BLOCKED_NO_DISPATCH')and 0<=m['formal_completed']<=60
extras=[];external=[];models=[];processes=[];observation=None;excluded_supplements=[];native_fresh=None;probe_scope=None;hash_cache={}
if role=='29':inventory(r,names|set(controls))
else:
 exclusions=set(m['excluded_exact_relative_paths']);live=m['excluded_live_descriptors']
 expected_exclusions=set(controls)|{'FINISHER_RESULT.json','FINISHER_FAILURE.json','operations/ATOMIC_ARCHIVE_PUBLICATION.json'}|set(live)
 assert exclusions==expected_exclusions,'Unapproved exclusion path'
 for n in live:safe(n);assert n not in names and n not in{'FINISHER_STARTED.json','TASK_RESULT.json'}
 found,_=inventory(r,allow_extras=True);extras=sorted(found-names-set(controls));assert set(extras)<=exclusions,'Unknown unsealed compute supplement'
 for n in extras:excluded_supplements.append(stable(r,dict(path=n,bytes=(r/n).stat().st_size,sha256=sha(r/n))))
 finish=read(r/'FINISHER_RESULT.json');assert finish['status']=='PASS_REAL_TERMINAL_SCIENCE_ARCHIVED_ON29'and finish['terminal']==m['terminal']and finish['actual_formal_completed']==m['formal_completed']
 assert finish['identity']==m['finisher_identity']==read(r/'FINISHER_STARTED.json')['identity']
 identities=[]
 for n in('CONTROLLER_STARTED.json','FINISHER_STARTED.json'):
  x=read(r/n)['identity'];assert isinstance(x,dict);identities.append(x)
 controller_identity=read(r/'CONTROLLER_STARTED.json')['identity']
 units={x['id']:x for x in read(r/'QUALIFICATION_JOB_MANIFEST.json')['units']}
 jobs={x['id']:x for x in read(r/'FORMAL_JOB_MANIFEST.json')['jobs']}
 for folder in('qualification','formal'):
  for started in sorted((r/folder).glob('*/STARTED.json')):
   x=read(started)['identity'];assert isinstance(x,dict);identities.append(x)
  for path in sorted((r/folder).glob('*/RESOURCE_INTERVAL.json')):
   row=read(path);roster=units if folder=='qualification'else jobs;unit=roster[path.parent.name]
   assert row['unit']==unit['id']and row['gpu']==unit['gpu']and row['gpu_uuid']==unit['gpu_uuid']
   assert row['controller_identity']==controller_identity and row['release']=='PASS'
   assert row['source_freeze_sha256']==source_sha and row['phase']==folder
   x=row['identity'];assert isinstance(x,dict)and belongs(x,remote_root)
   assert str(r/'scripts'/('qualify_gpu.py'if folder=='qualification'else'resident_formal_worker.py'))in x['cmdline']
   identities.append(x)
 probe_scope=loader_probe_scope(r,by_path,native_contract_proof['manifest_sha256'],native_contract_proof['count']);identities.append(probe_scope['worker_identity'])
 unique={json.dumps(x,sort_keys=True):x for x in identities};identities=list(unique.values())
 for x in identities:
  current=identity(x['pid']);assert current!=x,'An exact own process is still alive'
  processes.append(dict(expected=x,current=current,exact_identity_exited=True))
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   x=identity(int(p.name));assert not execution_process(x,remote_root),'Unknown own scientific root process is alive'
 bindings=read(r/'analysis/EXTERNAL_BINDINGS.json')['files'];assert len(bindings)==len({x['path']for x in bindings})==61
 for x in bindings:
  p=Path(x['path']);a=p.stat();h=sha(p);b=p.stat();assert(a.st_size,a.st_ino,a.st_mtime_ns)==(b.st_size,b.st_ino,b.st_mtime_ns)
  assert b.st_size==x['bytes']and h==x['sha256'];external.append(dict(path=x['path'],bytes=b.st_size,sha256=h))
  hash_cache[str(p.resolve())]=dict(bytes=b.st_size,inode=b.st_ino,device=b.st_dev,mtime_ns=b.st_mtime_ns,sha256=h)
 native_fresh=fresh_native_files(r,native_manifest,native_contract_proof,hash_cache)
 model_rows=read(r/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files'];assert len(model_rows)==len({x['path']for x in model_rows})==17
 for x in model_rows:
  b=Path(x['path']).stat();assert(b.st_size,b.st_ino,b.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns'])
  models.append(dict(path=x['path'],bytes=b.st_size,inode=b.st_ino,mtime_ns=b.st_mtime_ns))
 result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20);assert result.returncode==0,result.stderr
 apps=[];known={x['pid']:[]for x in identities}
 for x in identities:known[x['pid']].append(x)
 for line in result.stdout.splitlines():
  if not line.strip():continue
  parts=[x.strip()for x in line.split(',')];pid=int(parts[1]);current=identity(pid)
  assert not belongs(current,remote_root),'Own GPU task remains alive'
  if pid in known:assert current is not None and all(current!=x for x in known[pid]),'Known own PID still in NVML without positively foreign current identity'
  apps.append(dict(uuid=parts[0],pid=pid,used_memory_MiB=parts[2],identity=current))
 observation=dict(status='PASS_NO_OWN_ROOT_NVML_FOREIGN_ALLOWED',apps=apps,own_task_processes=[])
out=dict(status='PASS_INDEPENDENT_FRESH_READONLY_SCIENCE_AND_SOURCE_CATALOG',role=role,task_id=task,terminal=m['terminal'],formal_completed=m['formal_completed'],host=socket.gethostname(),monotonic=time.monotonic(),
 count=len(actual),bytes=sum(x['bytes']for x in actual),files=actual,source_count=len(source),source_files=source,source_freeze_sha256=source_sha,raw_remote_only_files=raw,
 native_closure_contract=native_contract_proof,native_closure_fresh_SHA_stat=native_fresh,loader_probe_scope=probe_scope,external_fresh_SHA=external,model_stat=models,exact_process_exit=processes,NVML=observation,compute_excluded_supplements=excluded_supplements,archive29_exact_inventory=role=='29',
 controls={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())for n,b in control_data.items()})
print(json.dumps(out,sort_keys=True))
'''


def decode_controls(audit):
    require(set(audit["controls"]) == set(CONTROLS), "Wrong original control roster")
    result = {}
    for name, row in audit["controls"].items():
        data = base64.b64decode(row["base64"], validate=True)
        require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "Original control byte/SHA mismatch")
        result[name] = data
    return result


def stream_program(config, rows):
    require(all(not raw_path(x["path"]) for x in rows), "Raw/model stream forbidden")
    # Keep remote argv bounded even when the dynamic science catalog has many
    # thousands of files. Bind the complete small-file roster by canonical SHA;
    # read it from the already verified sealed manifest on the remote host.
    catalog_sha = hashlib.sha256(canonical(rows)).hexdigest()
    return REMOTE_HELPERS + "\nimport tarfile\nr=Path(" + repr(config["remote_root"]) + ")\nraw_suffixes=" + repr(RAW_SUFFIXES) + "\nexpected_catalog_sha=" + repr(catalog_sha) + "\nexpected_count=" + repr(len(rows)) + "\n" + r'''
rows=[x for x in read(r/'SHA256_MANIFEST.json')['files']if not x['path'].lower().endswith(raw_suffixes)]
assert len(rows)==expected_count and hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n').hexdigest()==expected_catalog_sha
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|')as archive:
 for row in rows:
  stable(r,row);p=r/safe(row['path']);before=p.stat()
  info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o444
  with p.open('rb')as stream:archive.addfile(info,stream)
  after=p.stat();assert(before.st_size,before.st_ino,before.st_mtime_ns)==(after.st_size,after.st_ino,after.st_mtime_ns)
  stable(r,row)
'''


def stream_small(config, rows, partial):
    expected = {x["path"]: x for x in rows}; require(len(expected) == len(rows), "Duplicate requested file")
    code = stream_program(config, rows)
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(command(config, "55", code_argument=code), stdout=subprocess.PIPE, stderr=errors)
        seen = set()
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
                for member in archive:
                    name = member.name; safe_relative(name)
                    require(member.isfile() and name in expected and name not in seen and not raw_path(name), "Unknown/nonregular/raw stream member")
                    row = expected[name]; require(member.size == row["bytes"], "Stream declared size differs")
                    source = archive.extractfile(member); require(source is not None, "Missing regular payload")
                    path = partial / safe_relative(name); path.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256(); total = 0
                    with path.open("xb") as dest:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            dest.write(chunk); digest.update(chunk); total += len(chunk)
                        dest.flush(); os.fsync(dest.fileno())
                    require(total == row["bytes"] and digest.hexdigest() == row["sha256"], "Actual streamed bytes/SHA differ")
                    seen.add(name)
            proc.stdout.close(); code = proc.wait(); errors.seek(0)
            require(code == 0, "Actual stream failed, preserve/no retry: " + errors.read().decode(errors="replace")[-6000:])
            require(seen == set(expected), "Stream missing expected files")
        except BaseException:
            proc.stdout.close()
            if proc.poll() is None: proc.terminate()
            proc.wait()
            raise


def rename_noreplace(source, destination):
    lib = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        fn = lib.renamex_np; fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]; fn.restype = ctypes.c_int
        result = fn(os.fsencode(source), os.fsencode(destination), 4)
    elif hasattr(lib, "renameat2"):
        fn = lib.renameat2; fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]; fn.restype = ctypes.c_int
        result = fn(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else: raise RuntimeError("No atomic NOREPLACE primitive; refuse publication")
    if result: raise OSError(ctypes.get_errno(), "Atomic NOREPLACE failed")


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def local_atomic_selftest(root):
    with tempfile.TemporaryDirectory(prefix="resident_local_atomic_", dir=str(root / "operations")) as directory:
        p = Path(directory); a, b, c = p / "source", p / "existing", p / "new"
        a.write_bytes(b"SOURCE_KEEP"); b.write_bytes(b"EXISTING_KEEP")
        try: rename_noreplace(a, b)
        except OSError as error: require(error.errno == errno.EEXIST, "Unexpected NOREPLACE error")
        else: raise RuntimeError("NOREPLACE overwrote existing path")
        require(a.read_bytes() == b"SOURCE_KEEP" and b.read_bytes() == b"EXISTING_KEEP", "Negative atomic test changed bytes")
        rename_noreplace(a, c); require(not a.exists() and c.read_bytes() == b"SOURCE_KEEP", "Positive atomic test failed")
        fsync_dir(p)
    return dict(status="PASS_ACTUAL_LOCAL_NOREPLACE_POSITIVE_AND_NEGATIVE", platform=sys.platform,
                primitive="renamex_np_RENAME_EXCL4" if sys.platform == "darwin" else "renameat2_RENAME_NOREPLACE1")


def connection(root, status, **fields):
    row = dict(status=status, local_monotonic=time.monotonic(), **fields)
    path = root / "operations/LOCAL_DOWNLOADER_CONNECTION_STATUS.json"
    temp = path.with_suffix(".tmp." + str(os.getpid()))
    write_new(temp, canonical(row)); os.replace(temp, path)
    with (root / "operations/LOCAL_DOWNLOADER_CONNECTION_EVENTS.jsonl").open("ab") as stream:
        stream.write(canonical(row)); stream.flush(); os.fsync(stream.fileno())


def bind_local(root):
    require(root.name == ATTEMPT and Path(__file__).resolve() == root / "operations/download_verified_delivery.py", "Wrong independent local root/tool")
    config = read(root / "ROOT.json"); manifest = read(root / TOOL_MANIFEST)
    require(config["remote_root"] == manifest["remote_root"] and config["archive_root"] == manifest["archive_root"], "Remote/archive binding differs")
    source_sha = manifest["source_freeze_sha256"]
    require(sha(root / "SOURCE_FREEZE.json") == source_sha and len(source_sha) == 64, "Bound final source freeze differs")
    mine = [x for x in manifest["files"] if x["path"] == "operations/download_verified_delivery.py"]
    require(len(mine) == 1, "Unique independent tool identity required")
    for row in manifest["files"]:
        path = root / safe_relative(row["path"])
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Delivery tool manifest byte binding differs")
    return config, source_sha, mine[0]


def verify_local_sources(root, remote):
    present, missing_raw = [], []
    for row in remote["source_files"]:
        path = root / safe_relative(row["path"])
        if path.exists():
            require(not path.is_symlink() and path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Actual local frozen source differs")
            present.append(row)
        else:
            require(raw_path(row["path"]), "Missing nonraw frozen source is not a remote-only exception")
            missing_raw.append(row)
    return dict(status="PASS_ALL_LOCAL_PRESENT_SOURCES_AND_REMOTE_ONLY_RAW_EXCEPTION", source_count=len(remote["source_files"]), local_present_count=len(present), remote_only_missing_raw_count=len(missing_raw), missing_raw=missing_raw)


def deliver(root, config, source_sha, mine):
    partial, final = root / "delivery.partial_once", root / "delivery"
    require(not os.path.lexists(partial) and not os.path.lexists(final), "Prior delivery/stage exists; preserve/no implicit retry")
    audit55 = remote_json(config, "55", audit_code(config, source_sha, "55")); save(root, "operations/REMOTE55_FINAL_READONLY_AUDIT.json", audit55)
    audit29 = remote_json(config, "29", audit_code(config, source_sha, "29")); save(root, "operations/REMOTE29_FINAL_READONLY_AUDIT.json", audit29)
    require(audit55["files"] == audit29["files"] and audit55["source_files"] == audit29["source_files"] and audit55["raw_remote_only_files"] == audit29["raw_remote_only_files"], "Fresh55/29 science/source/raw catalogs differ")
    c55, c29 = decode_controls(audit55), decode_controls(audit29); require(c55 == c29, "Original29 control bytes differ from55 mirror")
    require(len(audit55["external_fresh_SHA"]) == 61 and len(audit55["model_stat"]) == 17 and audit55["NVML"]["status"] == "PASS_NO_OWN_ROOT_NVML_FOREIGN_ALLOWED", "Compute dependency/process evidence incomplete")
    native = audit55["native_closure_fresh_SHA_stat"]
    require(native["status"] == "PASS_DYNAMIC_COMPLETE_FROZEN_NATIVE_ELF_FRESH_SHA_STAT_BYTES" and native["count"] == len(native["files"])
        and native["count"] == audit55["native_closure_contract"]["count"] == audit29["native_closure_contract"]["count"]
        and native["manifest_sha256"] == audit29["native_closure_contract"]["manifest_sha256"], "Full dynamically frozen native closure is not independently fresh-verified")
    require(audit55["loader_probe_scope"]["scientific_qualification_PASS_from_probe"] == 0 and audit55["loader_probe_scope"]["formal_generations_from_probe"] == 0,
            "Engineering loader probe was conflated with scientific qualification/formal")
    save(root, "operations/FINAL_NATIVE_CLOSURE_FRESH_VERIFICATION.json", native)
    save(root, "operations/FINAL_LOADER_ONLY_PROBE_SCOPE_AND_COST_VERIFICATION.json", audit55["loader_probe_scope"])
    local_source = verify_local_sources(root, audit55); save(root, "operations/SOURCE_LOCAL_DELIVERY_VERIFICATION.json", local_source)
    raw = audit29["raw_remote_only_files"]; small = [x for x in audit29["files"] if not raw_path(x["path"])]
    require(sum(x["bytes"] for x in small) <= 2 * 1024**3 and all(x["bytes"] <= 256 * 1024**2 for x in small), "Small report delivery bound exceeded; no model/raw transfer allowed")
    require(os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 4 * 1024**3 + sum(x["bytes"] for x in small), "Insufficient Mac small-file delivery headroom")
    save(root, "operations/REMOTE_ONLY_TENSOR_CATALOG.json", dict(status="PASS_ALL_NEW_AND_REUSED_SCIENCE_RAW_DUAL_HOST_FRESH_SHA_BYTES", files=raw, count=len(raw), bytes=sum(x["bytes"] for x in raw), downloaded_to_Mac=False))
    atomic = local_atomic_selftest(root); save(root, "operations/LOCAL_ATOMIC_NOREPLACE_SELFTEST.json", atomic)
    partial.mkdir(exist_ok=False); fsync_dir(root)
    save(root, "operations/LOCAL_DOWNLOAD_TRANSFER_STARTED.json", dict(monotonic=time.monotonic(), once_only=True, small_count=len(small), small_bytes=sum(x["bytes"] for x in small)))
    stream_small(config, small, partial)
    control_rows = []
    for name in CONTROLS:
        data = c29[name]; write_new(partial / name, data)
        control_rows.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    expected = small + control_rows; actual = set()
    for path in partial.rglob("*"):
        require(not path.is_symlink(), "Symlink in local delivery")
        if path.is_file(): actual.add(str(path.relative_to(partial)))
        else: require(path.is_dir(), "Nonregular local member")
    require(actual == {x["path"] for x in expected}, "Unexpected or missing local file")
    for row in expected:
        path = partial / safe_relative(row["path"])
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Local fresh final SHA/bytes differs")
    require(not any(raw_path(n) for n in actual), "Raw/model reached Mac delivery")
    require(sha(root / mine["path"]) == mine["sha256"], "Delivery tool changed during operation")
    fsync_dir(partial); rename_noreplace(partial, final); fsync_dir(root)
    local = dict(status="PASS_ORIGINAL29_CONTROLS_AND_SMALL_FILES_LOCALLY_FRESH_SHA_VERIFIED", task_id=TASK, execution_attempt=root.name,
        terminal=audit29["terminal"], formal_completed=audit29["formal_completed"], delivery=str(final), files=len(expected), bytes=sum(x["bytes"] for x in expected), file_catalog=expected,
        source_freeze_sha256=source_sha, dynamic_source_count=audit29["source_count"], remote_only_raw_count=len(raw), raw_or_model_downloads=0,
        exact_original29_control_bytes_equal55=True, atomic_publication=atomic)
    save(root, "operations/LOCAL_DELIVERY_VERIFICATION.json", local)
    independent = dict(status="PASS_INDEPENDENT_FINAL_READONLY_ALL_CATALOGS_PROCESSES_AND_LOCAL_DELIVERY", task_id=TASK, terminal=audit29["terminal"],
        source_freeze_sha256=source_sha, dynamic_science_count=audit29["count"], dynamic_source_count=audit29["source_count"], dynamic_raw_count=len(raw),
        actual61_external_fresh_SHA=True, actual17_model_stat=True, complete_native_closure_fresh_SHA_stat_count=native["count"], native_manifest_sha256=native["manifest_sha256"], loader_probe_scope_and_cost_separate=True, all_source_and_raw_fresh55_and29=True, exact_controller_finisher_and_workers_exited=True,
        no_own_task_NVML_foreign_allowed=True, archive29_all_files_and_directories_exact=True, original_four_control_bytes_equal55_29=True,
        local_small_files_fresh_SHA_bytes=True, remote_mutations=0, GPU_or_formal_retries=0, old_science_seals_changed=False,
        local_verification_sha256=sha(root / "operations/LOCAL_DELIVERY_VERIFICATION.json"))
    save(root, "operations/FINAL_INDEPENDENT_READONLY_VERIFICATION.json", independent)
    summary = dict(status="PASS_LEGAL_COMPLETE60_OR_TRUE_PARTIAL_STOP_INDEPENDENT_DELIVERY", task_id=TASK, execution_attempt=root.name,
        terminal=audit29["terminal"], actual_formal_completed=audit29["formal_completed"], planned_formal=60, archive_root=config["archive_root"], local_delivery=str(final),
        source_freeze_sha256=source_sha, source_count=audit29["source_count"], science_count=audit29["count"], remote_only_raw_count=len(raw), local_verified=True,
        independent_verified=True, complete_native_closure_fresh_SHA_stat_count=native["count"], native_manifest_sha256=native["manifest_sha256"], engineering_loader_probe_not_scientific_qualification=True, raw_or_model_downloads=0, GPU_retries=0, EXACT_physical_GPUh="UNRESOLVED_UNLESS_SCIENCE_COST_REPORT_PROVES_OTHERWISE")
    save(root, "operations/FINAL_DELIVERY_SUMMARY.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv); root = args.root.resolve()
    require(root.name == ATTEMPT and Path(__file__).resolve() == root / "operations/download_verified_delivery.py", "Wrong independent tool root")
    (root / "operations").mkdir(parents=True, exist_ok=True)
    try:
        config, source_sha, mine = bind_local(root)
    except BaseException as error:
        name = "operations/LOCAL_DOWNLOADER_PREFLIGHT_FAILURE.json"
        if not (root / name).exists():
            save(root, name, dict(status="REAL_LOCAL_TOOL_OR_SOURCE_MANIFEST_PREFLIGHT_FAILURE_NO_RETRY", error=repr(error),
                                 traceback=traceback.format_exc(), monotonic=time.monotonic(), remote_mutations=0))
        raise
    lock = (root / "operations/LOCAL_DOWNLOADER.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not any((root / n).exists() for n in (STARTED, FAILURE, "operations/LOCAL_DELIVERY_VERIFICATION.json", "operations/FINAL_DELIVERY_SUMMARY.json"))
        and not os.path.lexists(root / "delivery.partial_once") and not os.path.lexists(root / "delivery"), "Downloader/delivery previously attempted; inspect/no auto restart")
    save(root, STARTED, dict(status="READ_ONLY_CPU_WAITER_STARTED", pid=os.getpid(), argv=sys.argv, monotonic=time.monotonic(), source_freeze_sha256=source_sha, tool_sha256=mine["sha256"], remote_mutations=0))
    try:
        while True:
            try:
                ready = remote_json(config, "55", readiness_code(config, source_sha), readiness=True)
            except ConnectionPending as error:
                connection(root, "CONNECTION_WAIT_READONLY_NO_GPU_CHANGE", reason=str(error))
                if not args.wait: raise
                time.sleep(30); continue
            connection(root, "CONNECTED_READONLY_POLLING", remote_status=ready["status"])
            if ready["status"] == "READY_TRUE_CONTROLLER_AND_FINISHER_EXITED_AND_SEALED": break
            if not args.wait: raise RuntimeError("Not ready; no download or GPU work started")
            time.sleep(30)
        save(root, "operations/LOCAL_DOWNLOADER_TRUE_READINESS.json", ready)
        summary = deliver(root, config, source_sha, mine); print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        if not (root / FAILURE).exists(): save(root, FAILURE, dict(status="REAL_READONLY_DELIVERY_FAILURE_PRESERVE_NO_AUTO_RETRY", error=repr(error), traceback=traceback.format_exc(),
            monotonic=time.monotonic(), source_freeze_sha256=source_sha, partial_exists=os.path.lexists(root / "delivery.partial_once"), delivery_exists=os.path.lexists(root / "delivery"),
            remote_mutations=0, raw_or_model_downloads=0))
        raise

if __name__ == "__main__": raise SystemExit(main())
