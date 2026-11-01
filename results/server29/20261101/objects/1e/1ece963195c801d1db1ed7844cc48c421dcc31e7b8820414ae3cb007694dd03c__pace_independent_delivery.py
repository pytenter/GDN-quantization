"""One bounded blind observation, then one independently claimed delivery.

This program runs on the Mac. Its transport is an externally supplied, SHA-pinned
existing SSH helper; no credentials are copied into this scientific artifact.
It never starts/stops/restarts a scientific process, imports torch, or scores.
The finisher owns the primary archive. This helper verifies it and seals only a
new sibling supplement with late exit receipts and independent delivery proofs.
"""
import argparse
import base64
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

TASK = "QWEN_GDN_PACE48_AIME26_SEED1_60_V1"
REVISION = TASK + "_SCALE_EXPONENT_REPAIR_20261025"
LOCAL = Path("/Users/pytenter/Documents/临时项目1") / REVISION
REMOTE = "/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_exponent_repair_20261025"
ARCHIVE = "/data/zypan/qwen_gdn_pace48_archive/" + REVISION
UUID = "GPU-a78176ed-f97f-c54f-6201-5a01b570f422"
METHOD_VARIANT = "PACE48_FP16_EXPONENT_V1"
STATE_SCALE_FORMAT = "FP16_MANTISSA_PLUS_OPTIONAL_INT8_EXPONENT"
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".so", ".cubin", ".pyc",
                      ".npy", ".npz", ".ckpt", ".pkl", ".pickle", ".onnx")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def safe_path(root, relative):
    p = Path(relative)
    require(isinstance(relative, str) and not p.is_absolute() and ".." not in p.parts
            and str(p) == relative and relative != "." and not any(c in relative for c in "\x00\r\n"),
            "Unsafe artifact path")
    base = Path(root).resolve()
    result = base / p
    current = result
    while current != base:
        require(not current.is_symlink(), "Symlink artifact path forbidden")
        current = current.parent
    return result


def verify_rows(root, rows, *, external=False):
    seen = set()
    total = 0
    for row in rows:
        name = row["path"]
        require(name not in seen, "Duplicate manifest path")
        seen.add(name)
        p = Path(name) if external else safe_path(root, name)
        a = p.stat()
        require(p.is_file() and a.st_size == row["bytes"] and digest(p) == row["sha256"],
                "Fresh SHA/bytes mismatch: " + name)
        b = p.stat()
        require((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) == (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns),
                "File changed while auditing: " + name)
        if external and "inode" in row:
            require(b.st_ino == row["inode"], "Frozen external inode changed: " + name)
        if external and "mtime_ns" in row:
            require(b.st_mtime_ns == row["mtime_ns"], "Frozen external mtime changed: " + name)
        if external and "device" in row:
            require(b.st_dev == row["device"], "Frozen external device changed: " + name)
        if external and "resolved_path" in row:
            require(str(p.resolve()) == row["resolved_path"], "Frozen external resolved path changed: " + name)
        total += a.st_size
    return dict(count=len(rows), bytes=total, status="PASS_ALL_FRESH_SHA_BYTES")


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def acquire_reader_lock(root):
    path = Path(root) / "operations/INDEPENDENT_DELIVERY_READER.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def existing_delivery(root, source_sha, permit_sha):
    output = Path(root) / "independent_delivery"
    result = output / "RESULT.json"
    failure = output / "FIRST_FAILURE.json"
    claim = Path(root) / "operations/INDEPENDENT_TERMINAL_DELIVERY_CLAIM.json"
    if result.is_file():
        row = json.loads(result.read_text())
        require(row["status"] == "PASS_INDEPENDENT_TERMINAL_DELIVERY_AND_STOP"
                and row["source_manifest_sha256"] == source_sha and row["permit_sha256"] == permit_sha,
                "Existing independent delivery belongs to different pins or is invalid")
        require(row.get('execution_revision') == REVISION and row.get('method_variant') == METHOD_VARIANT
                and row.get('state_scale_format') == STATE_SCALE_FORMAT, 'Existing delivery format differs')
        return dict(status="ALREADY_DELIVERED_NO_REPEAT", result=str(result))
    if failure.is_file():
        return dict(status="PRIOR_DELIVERY_FAILURE_NO_AUTOMATIC_RETRY", failure=str(failure),
                    claim_preserved=claim.is_file())
    if claim.is_file():
        return dict(status="ALREADY_CLAIMED_NO_AUTOMATIC_RETRY", claim=str(claim))
    require(not output.exists(), "Independent output directory already exists without a valid terminal receipt")
    return None


def connection_failure(exc):
    if isinstance(exc, subprocess.TimeoutExpired):
        return True
    text = str(exc)
    patterns = [r"ssh: connect to host [^\n]+(?:Connection timed out|Operation timed out|Connection refused|No route to host)",
                r"ssh: Could not resolve hostname [^\n]+", r"Permission denied \((?:publickey|password|keyboard-interactive|gssapi)[^)]*\)",
                r"Permission denied, please try again", r"kex_exchange_identification: (?:read: )?Connection (?:reset|closed)"]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


REMOTE_COMMON = r'''
import base64,collections,ctypes,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
TASK="QWEN_GDN_PACE48_AIME26_SEED1_60_V1"
REVISION=TASK+"_SCALE_EXPONENT_REPAIR_20261025"
REMOTE="/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_exponent_repair_20261025"
ARCHIVE="/data/zypan/qwen_gdn_pace48_archive/"+REVISION
UUID="GPU-a78176ed-f97f-c54f-6201-5a01b570f422"
METHOD_VARIANT="PACE48_FP16_EXPONENT_V1"
STATE_SCALE_FORMAT="FP16_MANTISSA_PLUS_OPTIONAL_INT8_EXPONENT"
def req(x,m):
 if not x:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text())
def raw(p):return Path(p).read_bytes()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def safe(root,name):
 p=Path(name);req(isinstance(name,str) and not p.is_absolute() and '..' not in p.parts and str(p)==name and name!='.' and not any(c in name for c in '\x00\r\n'),'Unsafe artifact path')
 base=Path(root).resolve();q=base/p;current=q
 while current!=base:
  req(not current.is_symlink(),'Symlink artifact path');current=current.parent
 return q
def rows_check(root,rows,external=False):
 seen=set();total=0
 for x in rows:
  name=x['path'];req(name not in seen,'Duplicate manifest path');seen.add(name)
  p=Path(name) if external else safe(root,name);a=p.stat()
  req(p.is_file() and a.st_size==x['bytes'] and sha(p)==x['sha256'],'Fresh SHA/bytes mismatch: '+name)
  b=p.stat();req((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'Changed during audit: '+name)
  if external and 'inode' in x:req(b.st_ino==x['inode'],'External inode changed: '+name)
  if external and 'mtime_ns' in x:req(b.st_mtime_ns==x['mtime_ns'],'External mtime changed: '+name)
  if external and 'device' in x:req(b.st_dev==x['device'],'External device changed: '+name)
  if external and 'resolved_path' in x:req(str(p.resolve())==x['resolved_path'],'External resolved path changed: '+name)
  total+=b.st_size
 return {'status':'PASS_ALL_FRESH_SHA_BYTES','count':len(rows),'bytes':total}
def identity(pid):
 p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 return {'pid':int(pid),'starttick':int(fields[19]),'cmdline':[s.decode() for s in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]}
def compare(old):
 try:
  now=identity(old['pid'])
  if now==old:return {'status':'OWNED_ALIVE','owned_alive':True,'expected':old,'actual':now}
  if now['starttick']!=old['starttick']:return {'status':'PID_REUSED_FOREIGN','owned_alive':False,'expected':old,'actual':now}
  return {'status':'SAME_STARTTICK_ARGV_DIFFERENT_UNRESOLVED','owned_alive':True,'expected':old,'actual':now}
 except FileNotFoundError:return {'status':'GONE','owned_alive':False,'expected':old}
 except (OSError,ValueError) as e:return {'status':'IDENTITY_UNRESOLVED','owned_alive':True,'expected':old,'error':repr(e)}
def gpu_observation(identities):
 p=subprocess.run(['nvidia-smi','--id='+UUID,'--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 req(p.returncode==0,'NVML inventory unavailable');devices=[]
 for line in p.stdout.strip().splitlines():
  f=[x.strip() for x in line.split(',')];req(len(f)==5 and f[1]==UUID and int(f[0])==0,'GPU0 UUID differs');devices.append(f)
 req(len(devices)==1,'Exact physicalGPU0 missing')
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 req(p.returncode==0,'NVML process inventory unavailable');own=[];foreign=0;old={i['pid']:i for i in identities}
 for line in p.stdout.strip().splitlines():
  f=[x.strip() for x in line.split(',')];req(len(f)==3,'Malformed NVML app');pid=int(f[0])
  if pid in old:
   match=compare(old[pid])
   if match['status']=='PID_REUSED_FOREIGN':foreign+=1
   else:own.append({'pid':pid,'uuid':f[1],'used_memory':f[2],'identity_status':match['status']})
  else:foreign+=1
 return {'device0':devices[0],'own_apps':own,'foreign_app_count':foreign,'foreign_processes_not_signalled':True}
def scoped_processes(root):
 expected={'scripts/offline_parent.py','scripts/pace_controller.py','scripts/pace_worker.py','scripts/finalize_cpu.py','scripts/run_offline.sh'}
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   x=identity(int(p.name));argv=x['cmdline']
   if any(t in expected or t in {str(Path(root)/s) for s in expected} for t in argv):
    if str(root) in argv or str((p/'cwd').resolve())==str(root):found.append(x)
  except (OSError,ValueError):pass
 return found

OLD_REMOTE='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1'
OLD_ARCHIVE='/data/zypan/qwen_gdn_pace48_archive/QWEN_GDN_PACE48_AIME26_SEED1_60_V1'
OLD_SOURCE='700b707d450a14fb5dbd52aef3df6c0fa6ac59649359eb0712cac87a951113ca'
OLD_PERMIT='3cb0900e85fea2124fbcbfd3b1c764ce833e3de3e037cadd06f990c230914c31'
OLD_SCIENCE='9943928865d7259e200bc4b2df0e368ee6d230821ae23e230a5b278ed26b4a2e'
OLD_MANIFEST='a803c18c7aa8bf90e69b9d01e9ece3d1599a6bf51fa371f4ff1250d08c332c18'
OLD_SUPPLEMENT=OLD_ARCHIVE+'.independent_terminal_supplement_529389b2de188a0662c86b46a7185c56ecab478fa393146f95a38fbad8168b26'
OLD_SUPPLEMENT_MANIFEST='da6e44754828439e1a0a0798d14ea17d6715ef9176c1003f5658512546fa39b9'
def exact_inventory(root,expected):
 actual=set()
 for p in root.rglob('*'):
  req(not p.is_symlink(),'Archive symlink forbidden')
  if p.is_file():actual.add(p.relative_to(root).as_posix())
 req(actual==set(expected),'Exact immutable archive inventory differs')
def formal_mask_sha(root,current):
 return sha(Path(root)/('calibration/head_mask.json' if current else 'provenance/PRIOR_PACE_TRUE_STOP/calibration/head_mask.json'))
def prior_source_science(root):
 req(sha(root/'source_manifest.json')==OLD_SOURCE and sha(root/'GPU_PERMIT.json')==OLD_PERMIT and sha(root/'SCIENCE_FREEZE.json')==OLD_SCIENCE,'Original PACE source/permit/science changed')
 source=read(root/'source_manifest.json');req(source['count']==len(source['files'])==133,'Original source133 count differs')
 sourceproof=rows_check(root,source['files'])
 req(sha(root/'DELIVERY_MANIFEST.json')==OLD_MANIFEST,'Original primary manifest changed')
 manifest=read(root/'DELIVERY_MANIFEST.json');req(manifest['count']==len(manifest['files'])==317 and manifest['total_bytes']==sum(x['bytes'] for x in manifest['files'])==18360743,'Original317 science count differs')
 proof=rows_check(root,manifest['files']);science=read(root/'SCIENCE_FREEZE.json');rows_check(root,science['files'])
 verdict=read(root/'final_verdict.json');req(verdict['status']=='FAIL_CLOSED_STOP' and verdict['formal_completed']==3 and verdict['full_accuracy_reported'] is False,'Old trueSTOP reclassified')
 failed=read(root/'R3_PACE48_DP1/trajectories/aime26_02_R3_seed1/RESOURCE_INTERVAL.json');req(failed['returncode']==1 and failed['released'] is True,'Old failed partial identity/credit changed')
 return {'status':'PASS_FRESH_FIXED_ORIGINAL_PACE_SOURCE133_SCIENCE317_TRUE_STOP','source':sourceproof,'science':proof,'source_manifest_sha256':OLD_SOURCE,'permit_sha256':OLD_PERMIT,'science_freeze_sha256':OLD_SCIENCE,'manifest_sha256':OLD_MANIFEST,'failed_q02_R3_completion_credit':0}

PRIOR_REPAIR_REMOTE='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_repair_20261025'
PRIOR_REPAIR_ARCHIVE='/data/zypan/qwen_gdn_pace48_archive/QWEN_GDN_PACE48_AIME26_SEED1_60_V1_SCALE_REPAIR_20261025'
PRIOR_REPAIR_SOURCE='f51840870f017786bf761a53804a81bfa5c4a941a2207f136349761e0272256f'
PRIOR_REPAIR_PERMIT='32ca98b22157ec6b5c03ea648c39cc8f6e88b3d773e3ee245380fd53c5dbf2d7'
PRIOR_REPAIR_MANIFEST='de5e8f1df149106a82a1b5121b282beb987e58f34bebdde2db921ee0624caca7'
def prior_repair_pins(current):
 meta=current/'provenance/PRIOR_SCALE_REPAIR_TRUE_STOP_METADATA'
 req(sha(meta/'source_manifest.json')==PRIOR_REPAIR_SOURCE and sha(meta/'GPU_PERMIT.json')==PRIOR_REPAIR_PERMIT and sha(meta/'DELIVERY_MANIFEST.json')==PRIOR_REPAIR_MANIFEST,'Prior genuine-overflow metadata changed')
 official=read(meta/'independent_delivery/RESULT.json');extra=read(meta/'operations/FINAL_EXTRA_REVIEW_SUPPLEMENT_VERIFICATION.json')
 req(official['status']=='PASS_INDEPENDENT_TERMINAL_DELIVERY_AND_STOP' and official['core_archive']==PRIOR_REPAIR_ARCHIVE and official['core_manifest_sha256']==PRIOR_REPAIR_MANIFEST and official['source_manifest_sha256']==PRIOR_REPAIR_SOURCE and official['permit_sha256']==PRIOR_REPAIR_PERMIT and official['formal_completed']==3 and official['final_status']=='FAIL_CLOSED_STOP','Prior genuine-overflow STOP delivery changed')
 req(extra['status']=='PASS_FINAL_EXTRA_REVIEWS_ATOMIC_SIBLING_SUPPLEMENT' and extra['science_seals_modified'] is False,'Prior extra independent review binding changed')
 return meta,official,extra
def prior_actual_identities(actual):
 identities=[]
 for name in ['OFFLINE_PARENT_STARTED.json','SESSION_STARTED.json','FINISHER_STARTED.json']:
  path=actual/name
  if path.is_file():identities.append(read(path)['identity'])
 identities.extend(x['identity'] for x in read(actual/'RESOURCE_LEDGER.json')['intervals'] if x.get('identity'))
 states=[compare(x) for x in {canonical(x):x for x in identities}.values()]
 req(not any(x['owned_alive'] for x in states),'Prior exact PID/starttick/argv is still alive or unresolved')
 return dict(status='PASS_FRESH_PRIOR_EXACT_IDENTITIES_GONE_OR_PID_REUSED_FOREIGN',identities=states,identity_count=len(states),signals=0)
def prior_repair_source_science(actual,current,archive=False):
 meta,official,extra=prior_repair_pins(current)
 req(sha(actual/'source_manifest.json')==PRIOR_REPAIR_SOURCE and sha(actual/'GPU_PERMIT.json')==PRIOR_REPAIR_PERMIT and sha(actual/'DELIVERY_MANIFEST.json')==PRIOR_REPAIR_MANIFEST,'Prior genuine-overflow source/permit/primary manifest differs')
 source=read(actual/'source_manifest.json');manifest=read(actual/'DELIVERY_MANIFEST.json')
 req(source['count']==len(source['files']) and manifest['count']==len(manifest['files']) and manifest['total_bytes']==sum(x['bytes'] for x in manifest['files']),'Prior genuine-overflow dynamic source/science inventory differs')
 proof={'source':rows_check(actual,source['files']),'science':rows_check(actual,manifest['files'])}
 req(read(actual/'final_verdict.json')==read(meta/'final_verdict.json') and read(actual/'SESSION_RESULT.json')==read(meta/'SESSION_RESULT.json') and read(actual/'RESOURCE_LEDGER.json')==read(meta/'RESOURCE_LEDGER.json'),'Prior STOP verdict/session/real intervals changed')
 verdict=read(actual/'final_verdict.json');req(verdict['status']=='FAIL_CLOSED_STOP' and verdict['formal_completed']==3 and verdict['continuation_accounting']['current_formal_completed']==0 and verdict['full_accuracy_reported'] is False,'Prior failed formal was reclassified')
 intervals=read(actual/'RESOURCE_LEDGER.json')['intervals'];req(len(intervals)==2 and all(x['released'] is True for x in intervals) and [x['returncode'] for x in intervals]==[0,1],'Prior gate/pass or failed formal realexit changed')
 diagnostic='R3_PACE48_DP1/trajectories/aime26_02_R3_seed1/SCALE_RANGE_DIAGNOSTIC.json'
 req(sha(actual/diagnostic)==sha(meta/diagnostic),'Prior genuine-overflow diagnostic changed')
 proof.update(status='PASS_FRESH_PRIOR_SCALE_REPAIR_TRUE_STOP_ZERO_NEW_CREDIT',source_manifest_sha256=PRIOR_REPAIR_SOURCE,permit_sha256=PRIOR_REPAIR_PERMIT,manifest_sha256=PRIOR_REPAIR_MANIFEST,failed_formal_completion_credit=0)
 if archive:
  exact_inventory(actual,[x['path'] for x in manifest['files']]+['DELIVERY_MANIFEST.json','ARCHIVE_VERIFICATION.json'])
  receipt=read(actual/'ARCHIVE_VERIFICATION.json');req(receipt['status']=='PASS_INDEPENDENT_ARCHIVE29_ALL_SHA_BYTES' and receipt['manifest_sha256']==PRIOR_REPAIR_MANIFEST and receipt['files']==manifest['count'] and receipt['bytes']==manifest['total_bytes'],'Prior primary real archive receipt differs')
  supplements=[]
  for expected in [official['supplement'],extra['verification']]:
   path=Path(expected['archive_root']);req(str(path).startswith(PRIOR_REPAIR_ARCHIVE+'.independent_terminal_supplement_') and sha(path/'SUPPLEMENT_MANIFEST.json')==expected['manifest_sha256'],'Prior independent sibling manifest changed')
   sm=read(path/'SUPPLEMENT_MANIFEST.json');req(sm['core_archive']==PRIOR_REPAIR_ARCHIVE and sm['core_manifest_sha256']==PRIOR_REPAIR_MANIFEST and sm['count']==len(sm['files'])==expected['files'] and sm['total_bytes']==sum(x['bytes'] for x in sm['files'])==expected['bytes'],'Prior independent sibling dynamic inventory differs')
   exact_inventory(path,[x['path'] for x in sm['files']]+['SUPPLEMENT_MANIFEST.json']);supplements.append(dict(archive_root=str(path),manifest_sha256=expected['manifest_sha256'],verification=rows_check(path,sm['files'])))
  proof['late_and_final_independent_supplements']=supplements
 else:
  audit=read(meta/'independent_delivery/AUDIT55.json');proof['late55_mirrors']=rows_check(actual,audit['late_files'])
  proof['actual_prior_identities']=prior_actual_identities(actual)
 return proof

def fresh_triton(current):
 path=current/'analysis/SHARED_BITDECODE_TRITON322_INHERITED_INVENTORY.json';data=read(path)
 rows=data['files']+data['distribution_metadata']+data['host_c_compiler']
 req(len({x['path'] for x in rows})==len(rows) and data['count']==len(data['files']),'Inherited Triton inventory schema differs')
 return dict(status='PASS_FRESH_BYTES_AGAINST_SOURCE_BOUND_INHERITED_INVENTORY',inventory_sha256=sha(path),verification=rows_check(current,rows,True),historical_metadata_is_not_new_execution_evidence=True)

def mapped_native_artifacts(current):
 artifacts=[]
 for path in sorted(current.rglob('*.json')):
  rel=path.relative_to(current)
  if rel.parts[0]=='provenance' or not (path.name.startswith('LOADED_LIBRARY_') or path.name=='POSTFLIGHT_LIBRARY.json'):continue
  data=read(path);req(data['status']=='PASS_ACTUAL_LOADED_NATIVE_SHA_INODE' and data['selected'],'Loaded native artifact malformed')
  rows=data['selected'];proof=rows_check(current,rows,True);mapped={}
  for line in data['full_maps'].splitlines():
   fields=line.split(None,5)
   if len(fields)==6 and fields[5].startswith('/'):
    mapped.setdefault(str(Path(fields[5]).resolve()),set()).add(int(fields[4]))
  for row in rows:req(row['inode'] in mapped.get(str(Path(row['path']).resolve()),set()),'Recorded native maps inode differs: '+row['path'])
  artifacts.append(dict(path=rel.as_posix(),sha256=sha(path),selected_count=len(rows),verification=proof))
 return dict(status='PASS_ALL_RECORDED_ACTUAL_NATIVE_SHA_INODE_MAPS',artifacts=artifacts,artifact_count=len(artifacts),native_count_is_dynamic=True)
def driver_artifacts(current):
 artifacts=[]
 for path in sorted(current.rglob('DRIVER_IDENTITY.json')):
  rel=path.relative_to(current)
  if rel.parts[0]=='provenance':continue
  data=read(path);req(data['status']=='PASS' and data['actual_uuid']==data['expected_uuid']==data['CUDA_VISIBLE_DEVICES']==UUID and data['visible_device_count']==1 and data['logical_ordinal']==0,'Actual driver UUID/visibility receipt differs')
  req(data['context_creation_API_calls']==data['kernel_launch_API_calls']==0 and all(x['return_code']==0 for x in data['calls']),'Driver enumeration receipt malformed')
  artifacts.append(dict(path=rel.as_posix(),sha256=sha(path),verification=rows_check(current,[data['driver']],True)))
 return dict(status='PASS_ALL_ACTUAL_DRIVER_RECEIPTS_AND_FRESH_DRIVER_SHA_INODE',artifacts=artifacts,artifact_count=len(artifacts),driver_API_calls_in_this_audit=0)

def root_pins(p):
 r=Path(REMOTE);req(p['root']==REMOTE and p['archive']==ARCHIVE,'Exact PACE roots required')
 req(sha(r/'source_manifest.json')==p['source_sha256'] and sha(r/'GPU_PERMIT.json')==p['permit_sha256'],'Source/permit pin differs')
 metadata=read(r/'ROOT.json');req(metadata['task_id']==TASK and metadata['execution_revision']==REVISION and metadata['remote_root']==REMOTE and metadata['archive_root']==ARCHIVE,'Root task/revision differs')
 grant=r/'operations/HUMAN_PACE_SCALE_EXPONENT_REPAIR_AUTHORIZATION.json';req(sha(grant)=='9090e4344c7e2bbd462046c9ae9b3e50ef1701e98016ccab572dc22964c1dd97','New human scale-exponent grant differs')
 permit=read(r/'GPU_PERMIT.json');req(permit['execution_revision']==REVISION and permit['formal_new_samples']==57 and permit['reused_formal_samples']==3 and permit['formal_total_unique']==60,'New finite permit scope differs')
 req(permit.get('method_variant')==METHOD_VARIANT and permit.get('state_scale_format')==STATE_SCALE_FORMAT,'Explicit exponent method permit missing')
 method=read(r/'method_spec.json');req(method['common']['method_variant']==METHOD_VARIANT and method['common']['state_scale_format']==STATE_SCALE_FORMAT,'Declared exponent method differs')
 return r
def terminal_snapshot(p):
 r=root_pins(p);names=['OFFLINE_PARENT_STARTED.json','OFFLINE_PARENT_RESULT.json','PARENT_CONTROLLER_EXIT.json','PARENT_FINISHER_EXIT.json','SESSION_RESULT.json','FINISHER_STARTED.json','FINISHER_EXIT.json','DELIVERY_MANIFEST.json','ARCHIVE_VERIFICATION.json','DELIVERY_READY.json','final_verdict.json','RESOURCE_LEDGER.json']
 exists={n:(r/n).is_file() for n in names};ledger=read(r/'RESOURCE_LEDGER.json') if exists['RESOURCE_LEDGER.json'] else {}
 identities=[]
 for n in ['OFFLINE_PARENT_STARTED.json','FINISHER_STARTED.json']:
  if (r/n).is_file():identities.append(read(r/n)['identity'])
 for n in ['SESSION_STARTED.json','PARENT_CONTROLLER_ACTUAL_CHILD_IDENTITY.json','PARENT_FINISHER_ACTUAL_CHILD_IDENTITY.json']:
  if (r/n).is_file():
   value=read(r/n);identities.append(value.get('identity',value))
 if exists['SESSION_RESULT.json']:identities.append(read(r/'SESSION_RESULT.json')['controller'])
 active=ledger.get('active_unit')
 if active is not None:
  jobs={x['id']:x for x in read(r/'FORMAL_JOB_MANIFEST.json')['jobs']}
  if active in jobs:
   job=jobs[active];target=r/(job['condition']+'_PACE48_DP'+str(job['decode_period']))/'trajectories'/active
  else:
   req(active=='SCALE_REPAIR_GATES','Unknown active unit');target=r/'execution'/active
  dp=target/'DISPATCH.json';cp=target/'ACTUAL_CHILD_IDENTITY.json'
  if cp.is_file():
   child=read(cp);dispatch=read(dp);req(dispatch['unit']==active and dispatch['gpu_uuid']==UUID and dispatch['source_manifest_sha256']==p['source_sha256'] and dispatch['permit_sha256']==p['permit_sha256'],'Active dispatch binding differs');identities.append(child)
  elif (target/'STARTED.json').is_file():identities.append(read(target/'STARTED.json')['identity'])
 for q in r.glob('FROZEN_V4_FRESH_PASS_*.PROCESS_STARTED.json'):
  identities.append(read(q)['identity'])
 for q in r.glob('FROZEN_V4_FRESH_PASS_*.STARTED.json'):
  identities.append(read(q)['identity'])
 identities.extend(x['identity'] for x in ledger.get('intervals',[]) if x.get('identity'))
 unique={canonical(x):x for x in identities};identities=list(unique.values());states=[compare(x) for x in identities]
 nvml=gpu_observation(identities);scoped=scoped_processes(r)
 ready=all(exists.values()) and not any(x['owned_alive'] for x in states) and not nvml['own_apps'] and not scoped
 if ready:
  parent=read(r/'OFFLINE_PARENT_RESULT.json');sess=read(r/'SESSION_RESULT.json');fin=read(r/'FINISHER_EXIT.json');delivery=read(r/'DELIVERY_READY.json')
  ready=(parent['status']=='PARENT_TERMINAL' and parent.get('failure') is None and sess['own_workers_exit_and_NVML_release'] is True and fin['status']=='PASS_TERMINAL_DELIVERY' and fin['returncode']==0 and delivery['status']=='VERIFIED_COMPLETE_OR_REAL_STOP')
  for n in ['PARENT_CONTROLLER_EXIT.json','PARENT_FINISHER_EXIT.json']:
   x=read(r/n);ready=ready and x['actual_exited'] is True and x['identity_gone'] is True
   if n=='PARENT_FINISHER_EXIT.json':ready=ready and x['returncode']==0
 failure=(r/'FINISHER_FAILURE.json').is_file() or (exists['OFFLINE_PARENT_RESULT.json'] and read(r/'OFFLINE_PARENT_RESULT.json').get('failure') is not None)
 return {'status':'READY_FOR_INDEPENDENT_AUDIT' if ready else 'TERMINAL_DELIVERY_NEEDS_ATTENTION' if failure else 'PENDING_NO_ACTION','blind':True,'task_id':TASK,'execution_revision':REVISION,'method_variant':METHOD_VARIANT,'state_scale_format':STATE_SCALE_FORMAT,'formal_completed':3+sum(x.get('stage')=='formal' for x in ledger.get('completed_units',[])),'formal_reused':3,'historical_formal_completed':3,'runtime_reuse_admitted':(r/'CONTINUATION_ADMISSION.json').is_file(),'formal_current_completed':sum(x.get('stage')=='formal' for x in ledger.get('completed_units',[])),'active_unit':ledger.get('active_unit'),'terminal_exists':exists,'identities':states,'NVML':nvml,'same_root_processes':scoped,'source_manifest_sha256':p['source_sha256'],'permit_sha256':p['permit_sha256'],'unix':time.time(),'scientific_dispatches':0,'signals':0,'remote_mutations':0}
'''


REMOTE_AUDIT55 = r'''
r=root_pins(P);snap=terminal_snapshot(P);req(snap['status']=='READY_FOR_INDEPENDENT_AUDIT','Terminal readiness changed')
source=read(r/'source_manifest.json');req(source['task_id']==TASK and source['count']==len(source['files']),'Source manifest schema differs')
checks={'source':rows_check(r,source['files'])}
checks['prior55_fixed_origin']=prior_source_science(Path(OLD_REMOTE))
checks['prior55_fixed_origin']['actual_prior_identities']=prior_actual_identities(Path(OLD_REMOTE))
checks['prior55_genuine_overflow_true_stop']=prior_repair_source_science(Path(PRIOR_REPAIR_REMOTE),r)
for name in ['EXTERNAL_PYTHON_BINDINGS.json','EXTERNAL_NATIVE_BINDINGS.json','MODEL_TOKENIZER_FRESH_SHA.json']:
 data=read(r/'analysis'/name)
 if name=='MODEL_TOKENIZER_FRESH_SHA.json':req(len(data['files'])==17,'Full model/tokenizer17 roster differs')
 checks[name]=rows_check(r,data['files'],True)
checks['shared_Triton_runtime']=fresh_triton(r)
checks['actual_loaded_native']=mapped_native_artifacts(r)
checks['actual_driver_artifacts']=driver_artifacts(r)
manifest=read(r/'DELIVERY_MANIFEST.json');req(manifest['task_id']==TASK and manifest['count']==len(manifest['files']) and manifest['total_bytes']==sum(x['bytes'] for x in manifest['files']),'Delivery manifest counts differ')
req(manifest['source_manifest_sha256']==P['source_sha256'] and manifest['permit_sha256']==P['permit_sha256'],'Delivery source pins differ')
checks['primary_delivery_files']=rows_check(r,manifest['files'])
manifest_sha=sha(r/'DELIVERY_MANIFEST.json');ready=read(r/'DELIVERY_READY.json');archive=read(r/'ARCHIVE_VERIFICATION.json');verdict=read(r/'final_verdict.json')
req(ready['manifest_sha256']==manifest_sha==archive['manifest_sha256'] and ready['archive_verification_sha256']==sha(r/'ARCHIVE_VERIFICATION.json') and ready['archive_root']==ARCHIVE and archive['archive_root']==ARCHIVE and archive['status']=='PASS_INDEPENDENT_ARCHIVE29_ALL_SHA_BYTES','Archive/ready receipt differs')
req(verdict['status'] in ['FORMAL_COMPLETE_60_60_AND_STOP','FAIL_CLOSED_STOP'] and verdict['status']==ready['final_status'],'Scientific terminal status differs')
ledger=read(r/'RESOURCE_LEDGER.json');intervals=ledger['intervals'];req(ledger['active_unit'] is None and ledger['all_recorded_workers_released'] is True,'Ledger still active')
req(all(x['released'] is True and x['accounting_final'] is True and x['lower_bound_only'] is False and type(x['returncode'])==int and x['GPU_seconds']>=0 for x in intervals),'Final OS exit/accounting/release missing')
req(abs(ledger['cumulative_GPU_seconds']-sum(x['GPU_seconds'] for x in intervals))<1e-5,'Ledger cumulative accounting differs')
jobs=read(r/'FORMAL_JOB_MANIFEST.json')['jobs'];req(len(jobs)==60 and len({x['id'] for x in jobs})==60,'Fixed60 job roster differs')
data=read(r/'AIME26_DATASET_FREEZE.json')['records'];req(len(data)==30,'Canonical30 dataset missing')
expected=[]
for row in data:
 for arm,period in [('R2',8),('R3',1)]:expected.append((row['problem_id']+'_'+arm+'_seed1',row['problem_id'],arm,period,row['input_ids_sha256'],len(row['input_ids'])))
req([(x['id'],x['problem_id'],x['condition'],x['decode_period'],x['input_ids_sha256'],x['input_tokens']) for x in jobs]==expected,'Fixed problem/configuration/order differs')
indexed={(x['stage'],x['unit']):x for x in intervals};req(len(indexed)==len(intervals),'Repeated worker interval identity')
complete=[];invalid=[]
science_sha=None
if (r/'SCIENCE_FREEZE.json').is_file():
 sci=read(r/'SCIENCE_FREEZE.json');rows_check(r,sci['files']);science_sha=sha(r/'SCIENCE_FREEZE.json')
sys.path.insert(0,str(r/'src'));import pace_continuation as K
K.reuse_admission(r);union_invalid=[];union=K.completed_rows(r,invalid=union_invalid);continuation=K.continuation_summary(r)
for job,admitted,target in union:
 result=target/'RESULT.json'
 try:
  interval=read(target/'RESOURCE_INTERVAL.json');current=admitted['source_manifest_sha256']==P['source_sha256'];req((not current or interval==indexed[('formal',job['id'])]) and interval['returncode']==0 and interval['released'] is True,'Trajectory OS exit/release differs')
  wrap=read(result);payload_sha=wrap.pop('payload_sha256');req(hashlib.sha256(canonical(wrap)).hexdigest()==payload_sha,'Worker canonical payload SHA differs');wrap['payload_sha256']=payload_sha
  req(wrap['status']=='PASS' and wrap['stage']=='formal' and wrap['unit']==job['id'] and wrap['identity']==interval['identity'] and wrap['task_id']==TASK and wrap==admitted and (not current or (wrap['source_manifest_sha256']==P['source_sha256'] and wrap['permit_sha256']==P['permit_sha256'] and wrap['science_freeze_sha256']==science_sha)),'Worker final identity differs')
  z=wrap['result'];n=len(z['generated_token_ids']);total=job['input_tokens']+n-1
  if current:
   req(wrap.get('execution_revision')==REVISION and wrap.get('method_variant')==METHOD_VARIANT and wrap.get('state_scale_format')==STATE_SCALE_FORMAT,'Current wrapper does not declare new exponent revision')
   req(z.get('execution_revision')==REVISION and z.get('method_variant')==METHOD_VARIANT and z.get('state_scale_format')==STATE_SCALE_FORMAT,'Current formal payload silently claims original format')
  req(z['status']=='COMPLETE' and z['nonfinite'] is False and z['runtime_error'] is None and z['deterministic_algorithms'] is False,'Invalid numeric/protocol result')
  req(all(z[k]==job[k] for k in ['condition','problem_id','seed','input_ids_sha256','input_tokens']) and z['head_mask_sha256']==formal_mask_sha(r,current),'Trajectory source/mask identity differs')
  req(0<n<=81920 and z['generated_token_count']==n and all(type(t)==int and 0<=t<248320 for t in z['generated_token_ids']) and z['generated_token_ids_sha256']==hashlib.sha256(canonical(z['generated_token_ids'])).hexdigest(),'Token payload/count differs')
  req(not any(t in z['stop_token_ids'] for t in z['generated_token_ids'][:-1]),'Earlier EOS hidden in trajectory');eos=z['generated_token_ids'][-1] in z['stop_token_ids'];req(z['finish_reason']==('eos' if eos else 'length') and (eos or n==81920),'EOS/cap condition differs')
  req(z['actual_model_forward_calls']==total and z['sampler_calls']==n,'p+n-1 invariant differs')
  audit=z['quantizer_audit'];req(audit['dense_recurrent_carrier'] is False and audit['decode_period']==job['decode_period'] and audit['prefill_period']==256 and audit['actual_layer_coverage']==24 and all(x==total for x in audit['helper_calls'].values()),'Packed recurrent/helper coverage differs')
  if current:req(audit.get('execution_revision')==REVISION and audit.get('method_variant')==METHOD_VARIANT and audit.get('state_scale_format')==STATE_SCALE_FORMAT and audit.get('state_exponent_metadata_counted') is True,'Exponent audit annotation/counting missing')
  layers={str(i) for i in range(32) if i%4!=3}
  for key in ['native_first_chunk_compat_calls','native_first_chunk_state_checked']:req(set(audit[key])==layers and all(x==1 for x in audit[key].values()),'Native firstchunk compatibility/state-check roster differs')
  req(audit['native_first_chunk_state_retained'] is False and audit['native_first_chunk_state_discarded'] is True,'Native firstchunk dense state was not discarded')
  req(len(audit['storage_by_layer'])==24 and all(x['dense_state_persistent'] is False and x['fixture_only'] is False and x['tokens']==total and x['pending_tokens']==(n-1)%job['decode_period'] and x['quantization_count']==math.ceil(job['input_tokens']/256)+(n-1)//job['decode_period'] and x['finalization_policy']=='preserve' for x in audit['storage_by_layer'].values()),'Periodic packed storage differs')
  exponent_bytes=0
  for storage in audit['storage_by_layer'].values():
   req(storage['total_resident_tensor_nbytes']==sum(x['nbytes'] for x in storage['inventory']),'Actual tensor inventory total differs')
   for tensor in storage['inventory']:
    if tensor['name'].endswith('.scale_exponents'):
     req(tensor['name'] in ['base.int4.scale_exponents','base.int8.scale_exponents'] and tensor['dtype'] in ['torch.int8','int8'],'Exponent metadata escaped state head-parts or changed dtype')
     req(tensor['nbytes']==math.prod(tensor['shape']),'INT8 exponent physical bytes differ from shape');exponent_bytes+=tensor['nbytes']
  req(0<=exponent_bytes<=393216 and (current or exponent_bytes==0),'Actual optional state exponent allocation exceeds bound or rewrites historical format')
  for name,d in z['protocol_identity'].items():req(d==sha(r/'protocol'/name),'Sampler/scorer/data binding differs')
  complete.append({'unit':job['id'],'condition':job['condition'],'problem_id':job['problem_id'],'result_sha256':sha(result),'relative_result_path':str(result.relative_to(r)),'evidence_origin':'CURRENT_SCALE_EXPONENT_REPAIR' if current else 'REUSED_ORIGINAL_PACE_TRUE_STOP','source_manifest_sha256':wrap['source_manifest_sha256'],'permit_sha256':wrap['permit_sha256'],'actual_state_exponent_nbytes':exponent_bytes,'evidence_stored_format':STATE_SCALE_FORMAT if current else 'LEGACY_FP16_ONLY_E0_EQUIVALENT'})
 except Exception as exc:invalid.append({'unit':job['id'],'error':repr(exc)})
invalid.extend(union_invalid)
req(len([x for x in complete if x['evidence_origin']=='REUSED_ORIGINAL_PACE_TRUE_STOP'])==3,'Exact prior3 reuse missing')
req(verdict['formal_completed']==len(complete) and verdict['R2_completed']==sum(x['condition']=='R2' for x in complete) and verdict['R3_completed']==sum(x['condition']=='R3' for x in complete),'Real completed count differs from verdict')
full=verdict['status']=='FORMAL_COMPLETE_60_60_AND_STOP'
req(verdict.get('method_variant')==METHOD_VARIANT and verdict.get('state_scale_format')==STATE_SCALE_FORMAT and verdict.get('original_unmodified_R2_R3') is False,'Terminal report mislabels new numerical format')
if full:
 req(len(complete)==60 and not invalid and science_sha is not None and len([x for x in complete if x['evidence_origin']=='CURRENT_SCALE_EXPONENT_REPAIR'])==57 and len(intervals)==58 and continuation['runtime_reuse_admitted'] is True and continuation['admitted_reused_formal_count']==3,'Incomplete strict3+57 full success')
 gate,gate_interval,gate_target=K.verify_current_unit(r,'scale_repair_gates','SCALE_REPAIR_GATES');req(gate_interval==indexed[('scale_repair_gates','SCALE_REPAIR_GATES')],'New actual repair gate interval differs')
 gr=gate['result'];fmt=gr['exponent_format_tests'];req(gr['method_variant']==METHOD_VARIANT and gr['state_scale_format']==STATE_SCALE_FORMAT and gr['actual_cuda'] is True and gr['no_AIME_or_calibration_or_formal_credit'] is True and fmt['status']=='PASS' and fmt['actual_cuda'] is True and fmt['accepted_domain_exact'] is True and fmt['exponent_dtype']=='int8' and fmt['key_buffer_extension'] is False and fmt['CPU_GPU_parity']['status']=='PASS','New exponent CUDA qualification proof incomplete')
 passes=[read(r/('FROZEN_V4_FRESH_PASS_'+str(i)+'.json')) for i in [0,1]]
 req(passes[0]['identity']['pid']!=passes[1]['identity']['pid'] and passes[0]['payload']==passes[1]['payload'],'Two fresh FrozenV4 payloads differ')
 req(len(passes[0]['payload']['rows'])==60 and passes[0]['payload']['scorer_sha256']==sha(r/'protocol/aime26_scorer_v4.py'),'Frozen scorer roster/pin differs')
 for x in passes:
  req(x['status']=='PASS_FRESH_CPU_FROZEN_V4_EXACT_SAVED_PAYLOAD' and not compare(x['identity'])['owned_alive'],'Scorer process/status unresolved')
 for index,x in enumerate(passes):
  exit=read(r/('FROZEN_V4_FRESH_PASS_'+str(index)+'.PROCESS_EXIT.json'));req(exit['identity']==x['identity'] and exit['returncode']==0 and exit['actual_exited'] is True and exit['identity_gone'] is True and exit['source_manifest_sha256']==P['source_sha256'] and exit['permit_sha256']==P['permit_sha256'],'Actual owning-wait scorer exit proof differs')
 byunit={x['relative_result_path']:x for x in complete}
 for x in passes[0]['payload']['rows']:req(x['raw_result_path'] in byunit and x['raw_result_sha256']==byunit[x['raw_result_path']]['result_sha256'],'Scored raw result SHA differs')
 import csv
 with (r/'comparison/paired_results.csv').open() as f:paired=list(csv.DictReader(f))
 req(len(paired)==30 and len({x['problem_id'] for x in paired})==30,'Paired30 table incomplete')
 for arm,period in [('R2',8),('R3',1)]:
  summary=read(r/(arm+'_PACE48_DP'+str(period))/'summary.json');score=[x for x in passes[0]['payload']['rows'] if x['condition']==arm]
  req(summary['N']==30 and summary['correct']==sum(x['correct'] for x in score) and summary['accuracy']==summary['correct']/30,'Saved arm summary differs')
else:
 req(verdict['full_accuracy_reported'] is False and verdict['partial_scientific_CSV_created'] is False,'Partial status invented a complete report')
 req(not (r/'comparison/paired_results.csv').exists() and not (r/'R2_PACE48_DP8/results.jsonl').exists() and not (r/'R3_PACE48_DP1/results.jsonl').exists(),'Partial stop contains full scientific CSV/results')
late_names=['FINISHER_EXIT.json','PARENT_FINISHER_EXIT.json','OFFLINE_PARENT_RESULT.json','DELIVERY_READY.json','ARCHIVE_VERIFICATION.json','finisher.log']
late=[]
for name in late_names:
 p=r/name;req(p.is_file(),'Late exit artifact missing: '+name);late.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
after=terminal_snapshot(P);req(after['status']=='READY_FOR_INDEPENDENT_AUDIT','Terminal identities/NVML changed during audit')
print(json.dumps({'status':'PASS_INDEPENDENT55_ALL_SOURCE_OUTPUT_EXTERNAL_MODEL_SHA_BYTES','task_id':TASK,'execution_revision':REVISION,'method_variant':METHOD_VARIANT,'state_scale_format':STATE_SCALE_FORMAT,'checks':checks,'fresh_full_model17_SHA':True,'terminal_snapshot':after,'final_status':verdict['status'],'formal_completed':len(complete),'formal_reused':3,'formal_current_completed':len(complete)-3,'continuation':continuation,'invalid_raw_results':invalid,'source_manifest_sha256':P['source_sha256'],'permit_sha256':P['permit_sha256'],'science_freeze_sha256':science_sha,'manifest_sha256':manifest_sha,'manifest_base64':base64.b64encode(raw(r/'DELIVERY_MANIFEST.json')).decode(),'late_files':late,'kernel_calls':0,'model_loads':0,'scores_recomputed':0,'signals':0,'remote_mutations':0,'physical_GPU_active_seconds':'UNRESOLVED_NOT_MEASURED'},sort_keys=True))
'''


REMOTE_AUDIT29 = r'''
req(P['archive']==ARCHIVE,'Exact primary archive required');r=Path(ARCHIVE);req(r.is_dir() and not r.is_symlink(),'Primary archive missing')
req(sha(r/'DELIVERY_MANIFEST.json')==P['manifest_sha256'],'Primary archive manifest SHA differs')
m=read(r/'DELIVERY_MANIFEST.json');req(m['task_id']==TASK and m['source_manifest_sha256']==P['source_sha256'] and m['permit_sha256']==P['permit_sha256'] and m['count']==len(m['files']) and m['total_bytes']==sum(x['bytes'] for x in m['files']),'Archive manifest identity/count differs')
exact_inventory(r,[x['path'] for x in m['files']]+['DELIVERY_MANIFEST.json','ARCHIVE_VERIFICATION.json'])
proof=rows_check(r,m['files']);receipt=read(r/'ARCHIVE_VERIFICATION.json');req(receipt['manifest_sha256']==P['manifest_sha256'] and receipt['status']=='PASS_INDEPENDENT_ARCHIVE29_ALL_SHA_BYTES' and receipt['files']==m['count'] and receipt['bytes']==m['total_bytes'],'Original archive receipt differs')
oldroot=Path(OLD_ARCHIVE);prior=prior_source_science(oldroot);oldm=read(oldroot/'DELIVERY_MANIFEST.json');exact_inventory(oldroot,[x['path'] for x in oldm['files']]+['DELIVERY_MANIFEST.json','ARCHIVE_VERIFICATION.json'])
oldreceipt=read(oldroot/'ARCHIVE_VERIFICATION.json');req(oldreceipt['status']=='PASS_INDEPENDENT_ARCHIVE29_ALL_SHA_BYTES' and oldreceipt['manifest_sha256']==OLD_MANIFEST and oldreceipt['files']==317 and oldreceipt['bytes']==18360743,'Original sealed archive receipt differs')
sp=Path(OLD_SUPPLEMENT);req(sha(sp/'SUPPLEMENT_MANIFEST.json')==OLD_SUPPLEMENT_MANIFEST,'Original late exit supplement changed');sm=read(sp/'SUPPLEMENT_MANIFEST.json');req(sm['core_archive']==OLD_ARCHIVE and sm['core_manifest_sha256']==OLD_MANIFEST and sm['count']==len(sm['files'])==9 and sm['total_bytes']==sum(x['bytes'] for x in sm['files'])==58435,'Original late supplement scope differs');exact_inventory(sp,[x['path'] for x in sm['files']]+['SUPPLEMENT_MANIFEST.json']);prior['late_exit_supplement']=rows_check(sp,sm['files'])
prior_repair=prior_repair_source_science(Path(PRIOR_REPAIR_ARCHIVE),r,archive=True)
print(json.dumps({'status':'PASS_SECOND_INDEPENDENT_ARCHIVE29_ALL_SHA_BYTES','archive_root':ARCHIVE,'method_variant':METHOD_VARIANT,'state_scale_format':STATE_SCALE_FORMAT,'manifest_sha256':P['manifest_sha256'],'source_manifest_sha256':P['source_sha256'],'permit_sha256':P['permit_sha256'],'verification':proof,'original_archive_verification_sha256':sha(r/'ARCHIVE_VERIFICATION.json'),'prior_original_source_science_and_late_exit_supplement':prior,'prior_genuine_overflow_stop_source_science_and_all_supplements':prior_repair,'GPU_queries':0,'remote_mutations':0},sort_keys=True))
'''


REMOTE_FETCH = r'''
base=Path(P['base']);req(str(base) in [REMOTE,ARCHIVE],'Fetch root forbidden')
total=0;out=[]
for x in P['files']:
 p=safe(base,x['path']);req(p.suffix.lower() not in ['.pt','.pth','.bin','.safetensors','.so','.cubin','.pyc','.npy','.npz','.ckpt','.pkl','.pickle','.onnx'],'Binary/tensor download forbidden')
 req(x['bytes']<=P['per_file_limit'],'Small-file per-file bound exceeded');b=raw(p);total+=len(b)
 req(len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'] and total<=P['batch_limit'],'Fetch SHA/bytes/batch bound differs')
 out.append({'path':x['path'],'bytes':len(b),'sha256':x['sha256'],'base64':base64.b64encode(b).decode()})
print(json.dumps({'status':'PASS_SMALL_BYTES_FETCH','files':out},sort_keys=True))
'''


REMOTE_SUPPLEMENT = r'''
prefix=ARCHIVE+'.independent_terminal_supplement_';req(P['final'].startswith(prefix) and len(P['final'])==len(prefix)+64,'Exact sibling supplement name required')
f=Path(P['final']);s=Path(P['final']+'.staging');req(not f.exists() and not s.exists(),'Supplement already exists; no overwrite or automatic retry')
req(sha(Path(ARCHIVE)/'DELIVERY_MANIFEST.json')==P['core_manifest_sha256'],'Core archive changed before supplement')
req(len(P['files'])<=32 and sum(x['bytes'] for x in P['files'])<=33554432,'Supplement size/file-count bound exceeded')
s.mkdir();rows=[]
for x in P['files']:
 b=base64.b64decode(x['base64'],validate=True);req(len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],'Supplement payload differs')
 q=safe(s,x['path']);q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('xb') as z:z.write(b);z.flush();os.fsync(z.fileno())
 rows.append({'path':x['path'],'bytes':len(b),'sha256':x['sha256']})
m={'task_id':TASK,'status':'LATE_EXIT_AND_INDEPENDENT_DELIVERY_SUPPLEMENT','core_archive':ARCHIVE,'core_manifest_sha256':P['core_manifest_sha256'],'files':rows,'count':len(rows),'total_bytes':sum(x['bytes'] for x in rows)}
with (s/'SUPPLEMENT_MANIFEST.json').open('xb') as z:z.write(canonical(m)+b'\n');z.flush();os.fsync(z.fileno())
proof=rows_check(s,rows);manifest_sha=sha(s/'SUPPLEMENT_MANIFEST.json')
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rc=fn(-100,os.fsencode(s),-100,os.fsencode(f),1)
if rc:raise OSError(ctypes.get_errno(),'Supplement renameat2 NOREPLACE failed')
req(sha(f/'SUPPLEMENT_MANIFEST.json')==manifest_sha,'Supplement manifest changed after seal');rows_check(f,rows)
print(json.dumps({'status':'PASS_NEW_SIBLING_SUPPLEMENT_ALL_SHA_BYTES_ATOMIC_SEAL','archive_root':str(f),'manifest_sha256':manifest_sha,'core_manifest_sha256':P['core_manifest_sha256'],'files':proof['count'],'bytes':proof['bytes'],'GPU_queries':0,'core_archive_modified':False},sort_keys=True))
'''


def load_transport(path, expected_sha):
    require(digest(path) == expected_sha, "External transport SHA differs")
    spec = importlib.util.spec_from_file_location("pace_existing_pinned_transport", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def call(transport, body, payload, *, role="55", timeout=60):
    # The existing transport supplies this complete program on Python stdin.
    # No shell interpolation of the JSON payload is performed here.
    code = "import json\nP=json.loads(" + repr(json.dumps(payload)) + ")\n" + REMOTE_COMMON + body
    python = {"55": "/data/ydai/miniconda3/envs/bitdecode/bin/python", "29": "/usr/bin/python3"}[role]
    return transport.run(code, role=role, python=python, timeout=timeout)


def fetch(transport, base, rows, destination, *, role, maximum_total=2147483648):
    total = sum(x["bytes"] for x in rows)
    require(total <= maximum_total, "Mac small-file total bound exceeded")
    verified = []
    batch = []
    size = 0
    batches = []
    for row in rows:
        require(row["bytes"] <= 134217728, "Mac small-file per-file bound exceeded")
        if batch and size + row["bytes"] > 16777216:
            batches.append(batch)
            batch, size = [], 0
        batch.append(row)
        size += row["bytes"]
    if batch:
        batches.append(batch)
    for batch in batches:
        result = call(transport, REMOTE_FETCH, dict(base=base, files=batch,
                       per_file_limit=134217728, batch_limit=134217728), role=role, timeout=180)
        require(result["status"] == "PASS_SMALL_BYTES_FETCH", "Remote small-file fetch failed")
        require([x["path"] for x in result["files"]] == [x["path"] for x in batch], "Fetched roster differs")
        for row in result["files"]:
            data = base64.b64decode(row["base64"], validate=True)
            require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                    "Mac received SHA/bytes differ")
            target = safe_path(destination, row["path"])
            write_new(target, data)
            require(digest(target) == row["sha256"], "Mac saved SHA differs")
            verified.append({k: row[k] for k in ("path", "bytes", "sha256")})
    return verified


def self_test():
    with tempfile.TemporaryDirectory(prefix="pace_independent_delivery_cpu_") as name:
        root = Path(name)
        write_new(root / "x.txt", b"abc")
        row = dict(path="x.txt", bytes=3, sha256=digest(root / "x.txt"))
        require(verify_rows(root, [row])["count"] == 1, "Valid fresh SHA fixture failed")
        for action in (lambda: safe_path(root, "../escape"),
                       lambda: verify_rows(root, [row, row]),
                       lambda: verify_rows(root, [dict(row, sha256="0"*64)]),
                       lambda: write_new(root / "x.txt", b"overwrite")):
            try:
                action()
            except (RuntimeError, FileExistsError):
                pass
            else:
                raise RuntimeError("Unsafe/duplicate/stale fixture was accepted")
        namespace = {}
        exec(REMOTE_COMMON, namespace)
        require(namespace["rows_check"](root, [row])["count"] == 1, "Remote independent verifier fixture failed")
        old = dict(pid=42, starttick=100, cmdline=["python", "owned.py"])
        namespace["identity"] = lambda pid: dict(old)
        require(namespace["compare"](old)["status"] == "OWNED_ALIVE", "Live owned identity fixture failed")
        namespace["identity"] = lambda pid: dict(old, starttick=101)
        require(namespace["compare"](old)["status"] == "PID_REUSED_FOREIGN", "Reused PID fixture failed")
        namespace["identity"] = lambda pid: dict(old, cmdline=["unresolved"])
        require(namespace["compare"](old)["owned_alive"] is True, "Same-tick changed argv must remain unresolved")
        def gone(_pid):
            raise FileNotFoundError()
        namespace["identity"] = gone
        require(namespace["compare"](old)["status"] == "GONE", "Exited identity fixture failed")
        lock = acquire_reader_lock(root)
        require(lock is not None, "Fresh reader lock failed")
        try:
            require(acquire_reader_lock(root) is None, "Busy reader lock allowed another active operation")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        write_new(root / "operations/INDEPENDENT_TERMINAL_DELIVERY_CLAIM.json", b"{}\n")
        require(existing_delivery(root, "a"*64, "b"*64)["status"] == "ALREADY_CLAIMED_NO_AUTOMATIC_RETRY", "Existing claim was not terminal")
        write_new(root / "independent_delivery/FIRST_FAILURE.json", b"{}\n")
        require(existing_delivery(root, "a"*64, "b"*64)["status"] == "PRIOR_DELIVERY_FAILURE_NO_AUTOMATIC_RETRY", "Existing failure was not terminal")
        write_new(root / "independent_delivery/RESULT.json", canonical(dict(status="PASS_INDEPENDENT_TERMINAL_DELIVERY_AND_STOP", source_manifest_sha256="a"*64, permit_sha256="b"*64, execution_revision=REVISION, method_variant=METHOD_VARIANT, state_scale_format=STATE_SCALE_FORMAT)))
        require(existing_delivery(root, "a"*64, "b"*64)["status"] == "ALREADY_DELIVERED_NO_REPEAT", "Existing success allowed repeated transfer")
        try:
            existing_delivery(root, "c"*64, "b"*64)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Existing success with changed source pin was accepted")
        require(connection_failure(RuntimeError("SSH/CPU operation failed: ssh: connect to host 202.38.247.55 port 22: Connection timed out")), "SSH connect failure not classified WAIT")
        require(connection_failure(RuntimeError("zypan@host: Permission denied (publickey,password).")), "SSH authentication failure not classified WAIT")
        require(connection_failure(subprocess.TimeoutExpired(["ssh"], 60)), "Transport timeout not classified WAIT")
        require(not connection_failure(RuntimeError("SSH/CPU operation failed: RuntimeError: Source/permit pin differs")), "Real remote source error incorrectly classified WAIT")
        require(not connection_failure(PermissionError("Permission denied: '/source_manifest.json'")), "Remote filesystem permission error incorrectly classified SSH WAIT")
        # Compile each remote mode without executing network, NVML, or writes.
        for source in (REMOTE_AUDIT55, REMOTE_AUDIT29, REMOTE_FETCH, REMOTE_SUPPLEMENT):
            compile(REMOTE_COMMON+source, "pace_remote_cpu_helper", "exec")
        class CaptureTransport:
            def run(self, code, **options):
                return options
        require(call(CaptureTransport(), "", {}, role="55")["python"] ==
                "/data/ydai/miniconda3/envs/bitdecode/bin/python", "55 interpreter selector differs")
        require(call(CaptureTransport(), "", {}, role="29")["python"] ==
                "/usr/bin/python3", "29 interpreter selector differs")
    return dict(status="PASS_CPU_ONLY_HELPER_SELFTEST", cases=26, network_calls=0,
                GPU_calls=0, old_assets_modified=False)


def run_once(args, root):
    payload = dict(root=REMOTE, archive=ARCHIVE, source_sha256=args.source_sha256,
                   permit_sha256=args.permit_sha256)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    try:
        prior = existing_delivery(root, args.source_sha256, args.permit_sha256)
        if prior:
            print(json.dumps(prior))
            return 0
        require(digest(root / "source_manifest.json") == args.source_sha256
                and digest(root / "GPU_PERMIT.json") == args.permit_sha256, "Local frozen source/permit differs")
        source = json.loads((root / "source_manifest.json").read_text())
        self_row = next(x for x in source["files"] if x["path"] == "operations/pace_independent_delivery.py")
        require(digest(__file__) == self_row["sha256"], "Delivery helper is not the frozen implementation")
        transport = load_transport(args.transport, args.transport_sha256)
        observation = call(transport, "print(json.dumps(terminal_snapshot(P),sort_keys=True))", payload, timeout=60)
    except BaseException as exc:
        waiting = connection_failure(exc)
        value = dict(status="WAIT_REMOTE_CONNECTION_NO_CLAIM" if waiting else "FAIL_READONLY_ADMISSION_NO_CLAIM",
                     error=repr(exc), claim_created=False, scientific_dispatches=0, signals=0,
                     source_manifest_sha256=args.source_sha256, permit_sha256=args.permit_sha256)
        path = root / "monitoring" / (("CONNECTION_WAIT_" if waiting else "READONLY_ADMISSION_FAILURE_")+stamp+".json")
        write_new(path, canonical(value)+b"\n")
        print(json.dumps(dict(status=value["status"], receipt=str(path), error=value["error"])))
        return 0 if waiting else 1
    receipt = root / "monitoring" / ("INDEPENDENT_BLIND_OBSERVATION_"+stamp+".json")
    write_new(receipt, canonical(observation)+b"\n")
    if observation["status"] != "READY_FOR_INDEPENDENT_AUDIT" or not args.deliver_if_ready:
        print(json.dumps(dict(status=observation["status"], observation=str(receipt),
                              formal_completed=observation["formal_completed"], delivery_started=False)))
        return 0
    claim = root / "operations/INDEPENDENT_TERMINAL_DELIVERY_CLAIM.json"
    write_new(claim, canonical(dict(status="CLAIMED_ONCE", source_manifest_sha256=args.source_sha256,
              permit_sha256=args.permit_sha256, observation_sha256=digest(receipt), unix=time.time()))+b"\n")
    output = root / "independent_delivery"
    output.mkdir(exist_ok=False)
    try:
        audit55 = call(transport, REMOTE_AUDIT55, payload, timeout=7200)
        require(audit55["status"] == "PASS_INDEPENDENT55_ALL_SOURCE_OUTPUT_EXTERNAL_MODEL_SHA_BYTES", "Independent55 audit failed")
        manifest_raw = base64.b64decode(audit55.pop("manifest_base64"), validate=True)
        require(hashlib.sha256(manifest_raw).hexdigest() == audit55["manifest_sha256"], "Received primary manifest SHA differs")
        manifest = json.loads(manifest_raw)
        write_new(output / "AUDIT55.json", canonical(audit55)+b"\n")
        audit29 = call(transport, REMOTE_AUDIT29, dict(payload, manifest_sha256=audit55["manifest_sha256"]),
                       role="29", timeout=7200)
        require(audit29["status"] == "PASS_SECOND_INDEPENDENT_ARCHIVE29_ALL_SHA_BYTES", "Independent29 audit failed")
        expected_archive_receipt = next(x["sha256"] for x in audit55["late_files"]
                                        if x["path"] == "ARCHIVE_VERIFICATION.json")
        require(audit29["original_archive_verification_sha256"] == expected_archive_receipt,
                "Archive29 verification receipt is not byte-identical to the55 mirror")
        write_new(output / "AUDIT29.json", canonical(audit29)+b"\n")
        late = output / "late_exit_receipts"
        late.mkdir()
        late_rows = fetch(transport, REMOTE, audit55["late_files"], late, role="55", maximum_total=33554432)
        core = output / "primary_archive_small_files"
        core.mkdir()
        small = [x for x in manifest["files"] if x.get("download_small") is True
                 and not x["path"].lower().endswith(FORBIDDEN_SUFFIXES)]
        copied = fetch(transport, ARCHIVE, small, core, role="29",
                       maximum_total=2147483648+sum(x["bytes"] for x in source["files"]))
        write_new(core / "DELIVERY_MANIFEST.json", manifest_raw)
        local = dict(status="PASS_MAC_SMALL_FILES_ALL_RECEIVED_SHA_BYTES", task_id=TASK,
                     execution_revision=REVISION, method_variant=METHOD_VARIANT, state_scale_format=STATE_SCALE_FORMAT,
                     core_manifest_sha256=audit55["manifest_sha256"], files=copied,
                     count=len(copied), total_bytes=sum(x["bytes"] for x in copied),
                     omitted_binary_or_non_small_count=len(manifest["files"])-len(copied),
                     no_model_or_GPU_tensor_download=True, late_receipts=late_rows)
        write_new(output / "MAC_SMALL_FILES_VERIFICATION.json", canonical(local)+b"\n")
        bundle = []
        for p in sorted(output.rglob("*")):
            if p.is_file() and not core in p.parents:
                data = p.read_bytes()
                bundle.append(dict(path=str(p.relative_to(output)), bytes=len(data),
                                   sha256=hashlib.sha256(data).hexdigest(), base64=base64.b64encode(data).decode()))
        bundle_sha = hashlib.sha256(canonical([{k:v for k,v in x.items() if k!="base64"} for x in bundle])).hexdigest()
        require(len(bundle)<=32 and sum(x["bytes"] for x in bundle)<=33554432,
                "Supplement size/file-count bound exceeded")
        supplement = call(transport, REMOTE_SUPPLEMENT, dict(final=ARCHIVE+".independent_terminal_supplement_"+bundle_sha,
                          core_manifest_sha256=audit55["manifest_sha256"], files=bundle), role="29", timeout=300)
        require(supplement["status"] == "PASS_NEW_SIBLING_SUPPLEMENT_ALL_SHA_BYTES_ATOMIC_SEAL", "Supplement seal failed")
        write_new(output / "SUPPLEMENT_ARCHIVE_VERIFICATION.json", canonical(supplement)+b"\n")
        result = dict(status="PASS_INDEPENDENT_TERMINAL_DELIVERY_AND_STOP", task_id=TASK,
                      execution_revision=REVISION, method_variant=METHOD_VARIANT, state_scale_format=STATE_SCALE_FORMAT,
                      final_status=audit55["final_status"], formal_completed=audit55["formal_completed"],
                      source_manifest_sha256=args.source_sha256, permit_sha256=args.permit_sha256,
                      core_archive=ARCHIVE, core_manifest_sha256=audit55["manifest_sha256"],
                      supplement=supplement, Mac_delivery=str(core), small_files=len(copied),
                      audit55_sha256=digest(output / "AUDIT55.json"), audit29_sha256=digest(output / "AUDIT29.json"),
                      Mac_verification_sha256=digest(output / "MAC_SMALL_FILES_VERIFICATION.json"),
                      scientific_dispatches=0, signals=0, score_calls=0, GPU_kernel_calls=0,
                      remote55_mutations=0, core_archive_modified=False)
        write_new(output / "RESULT.json", canonical(result)+b"\n")
        print(json.dumps(dict(status=result["status"], final_status=result["final_status"],
                              formal_completed=result["formal_completed"], result=str(output / "RESULT.json"))))
        return 0
    except BaseException as exc:
        import traceback
        write_new(output / "FIRST_FAILURE.json", canonical(dict(status="FAIL_CLOSED_INDEPENDENT_DELIVERY",
                  error=repr(exc), traceback=traceback.format_exc(), claim_preserved=True,
                  no_automatic_retry=True, scientific_dispatches=0, signals=0))+b"\n")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--transport", type=Path)
    parser.add_argument("--transport-sha256")
    parser.add_argument("--source-sha256")
    parser.add_argument("--permit-sha256")
    parser.add_argument("--deliver-if-ready", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    for value in (args.transport_sha256, args.source_sha256, args.permit_sha256):
        require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "Explicit frozen SHA pins required")
    require(args.local_root is not None and args.transport is not None, "Local root and pinned transport are required")
    root = args.local_root.resolve()
    require(root == LOCAL.resolve(), "Only the exact new local scale-exponent revision may observe or deliver")
    lock = acquire_reader_lock(root)
    if lock is None:
        print(json.dumps(dict(status="LOCAL_READER_LOCK_BUSY_NO_ACTION")))
        return 0
    try:
        return run_once(args, root)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
