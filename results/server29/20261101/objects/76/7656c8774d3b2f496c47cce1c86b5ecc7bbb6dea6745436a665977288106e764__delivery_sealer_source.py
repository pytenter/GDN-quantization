"""Immutable CPU delivery/seal for the independent direct32K task.

Reads ROOT.json. No CUDA/model import, no GPU dispatch, no tensor deserialization.
Remote payloads are streamed twice: hashlib first, an independent sha256sum/
shasum process second. Large tensors remain remote and are listed explicitly.
Existing files are compared or reused exactly; conflicting bytes are never
overwritten. --self-test uses disposable synthetic files and no SSH connection.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

TASK = 'QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1'
BASE = pathlib.Path(__file__).resolve().parent
PYTHON = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
SSH = ['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','zypan@202.38.247.55']
CAP = 10 * 1024**3
TENSOR_LIMIT = 16 * 1024**2
MANIFEST = 'hashes/artifact_sha256.txt'
RECEIPT = 'hashes/DELIVERY_VERIFICATION.json'
SOURCE_SNAPSHOT = 'analysis_scripts/delivery_sealer_source.py'
CONFIG_SNAPSHOT = 'analysis_scripts/DELIVERY_ROOT_CONFIG.json'
DEPENDENCY_SOURCE = 'analysis_scripts/final_dependency_verify.py'
DEPENDENCY_RECEIPT = 'analysis/FINAL_DEPENDENCY_VERIFICATION.json'
SEALS = {
    'inherited_cpu_v1': ('d659b34a8d8f7dc6196192bcad9d2f95a0aad3d9b1237f58f2e501777c58b015',70),
    '.': ('ad024f3b4406680f1d4f7a6de6294f6f889c7667139b86f378bcec1b7ce9add2',140),
    'execution_01': ('0afd31ce3992fc302556a88a3e6cb2f5e539850303efe824999c57ce7366ff54',24),
    'execution_02': ('efece7e96e301f1a7bc40c66e1b89788a0d1387c17dd566083d7bf980b7a5af6',43),
}

REMOTE_HELPER = r'''
import hashlib,json,os,pathlib,shutil,subprocess,sys,tempfile
TASK='QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1'
CAP=10*1024**3
EXCLUDED={'hashes/artifact_sha256.txt','hashes/DELIVERY_VERIFICATION.json'}
def need(ok,why):
 if not ok:raise ValueError(why)
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 return h.hexdigest()
def independent_digest(p):
 command=([shutil.which('sha256sum'),'--',str(p)] if shutil.which('sha256sum') else
          [shutil.which('shasum'),'-a','256','--',str(p)] if shutil.which('shasum') else None)
 need(command is not None,'Independent sha256sum/shasum executable unavailable; no substitute PASS')
 value=subprocess.check_output(command,text=True).split()[0]
 need(len(value)==64 and all(c in '0123456789abcdef' for c in value),'Bad independent SHA256 output')
 return value,command[0]
def safe(root,rel):
 need(isinstance(rel,str) and bool(rel) and not any(ord(c)<32 for c in rel),'Unsafe/control-character asset path')
 q=pathlib.PurePosixPath(rel);need(not q.is_absolute() and '..' not in q.parts,'Asset path escapes root')
 p=(root/pathlib.Path(*q.parts));need(p.resolve().is_relative_to(root.resolve()),'Asset symlink escapes root')
 return p
def ignored(rel):
 p=pathlib.PurePosixPath(rel)
 return '__pycache__' in p.parts or p.suffix=='.pyc' or p.name.startswith('._')
def scan(root,second=False):
 out={};ignored_rows=[];total=0;tools=set()
 for directory,dirs,files in os.walk(root,followlinks=False):
  for d in dirs:need(not (pathlib.Path(directory)/d).is_symlink(),'Symlink directory must not be followed or silently excluded')
  for name in files:
   p=pathlib.Path(directory)/name;need(p.is_file() and not p.is_symlink(),'Only regular in-root assets may be sealed')
   rel=str(p.relative_to(root));safe(root,rel);s=p.stat();total+=s.st_size
   if ignored(rel):ignored_rows.append(dict(path=rel,bytes=s.st_size));continue
   if rel in EXCLUDED:continue
   before=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
   if second:h,tool=independent_digest(p);tools.add(tool)
   else:h=digest(p)
   s=p.stat();after=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
   need(before==after,'Asset changed while streaming: '+rel)
   out[rel]=dict(sha256=h,bytes=s.st_size)
 need(total<=CAP,'Physical new archive exceeds10GiB including excluded bytecode/seal files')
 return dict(assets=out,ignored_non_delivery_files=ignored_rows,total_regular_bytes=total,
             payload_bytes=sum(v['bytes'] for v in out.values()),independent_hash_executables=sorted(tools))
def frozen(root):
 p=root/'source_manifest.json';m=json.loads(p.read_text());need(m['task']==TASK and m['frozen_before_any_new_GPU_forward'] is True,'Wrong prospective source manifest')
 source_sha=digest(p);dispatch=root/'COORDINATOR_STARTED.json'
 if dispatch.is_file():
  started=json.loads(dispatch.read_text());need(started['task_id']==TASK and started['source_manifest_sha256']==source_sha,'Current source manifest differs from the original coordinator dispatch binding')
 else:
  ledger=json.loads((root/'gpu_resource_ledger.json').read_text());need(not ledger['phases'],'GPU work exists without its original source-manifest dispatch binding')
 entries=m['execution_sources'];need(len(entries)==56,'Expected frozen56 execution assets; refusal rather than changing scope')
 need({'src/worker.py','src/coordinator.py','preregistration.json','frozen_documents.json'}<={x['path'] for x in entries},'Mandatory frozen execution assets absent')
 for row in entries:
  q=safe(root,row['path']);need(q.is_file() and q.stat().st_size==row['bytes'] and digest(q)==row['sha256'],'Frozen source identity changed: '+row['path'])
 return dict(status='PASS',verified_execution_sources=len(entries),source_manifest_sha256=source_sha,
             original_dispatch_source_binding='PASS' if dispatch.is_file() else 'NO_GPU_DISPATCH',
             old_execution02_seal_sha256=m['old_execution02_seal_sha256'],model_identity_receipt_sha256=m['model_identity_receipt_sha256'])
def boundary(root):
 g=json.loads((root/'GATE_STATUS_FINAL.json').read_text());l=json.loads((root/'gpu_resource_ledger.json').read_text())
 need(g['task_id']==l['task_id']==TASK,'Wrong final task identity')
 need(g.get('no_further_GPU_work') is True and g.get('GPU_release_verified') is True and l.get('final_GPU_release_verified') is True,'Final GPU release barrier missing')
 need(all(p.get('released') is True for p in l['phases']),'An own GPU reservation lacks verified release')
 need(g['GPU_HOURS_USED']==l['GPU_HOURS_USED'] and 0<=l['GPU_HOURS_USED']<=24.0,'GPUh accounting mismatch/cap')
 final=json.loads((root/'FINAL_STATUS.json').read_text());need(final['TASK']==TASK and final['FINAL_EXECUTION_STATUS'] in ('COMPLETE_AND_STOP','PARTIAL_STOP','FAIL_AND_STOP','BLOCKED_AND_STOP'),'Final Chinese report/status not completed')
 need((root/'FINAL_SCIENTIFIC_REPORT.md').is_file(),'Final scientific report is absent')
 return dict(status='PASS',GPU_HOURS_USED=l['GPU_HOURS_USED'],GPU_release_verified=True,
   final_execution_status=final['FINAL_EXECUTION_STATUS'],gate_sha256=digest(root/'GATE_STATUS_FINAL.json'),
   ledger_sha256=digest(root/'gpu_resource_ledger.json'),final_status_sha256=digest(root/'FINAL_STATUS.json'),
   first_failure_sha256=digest(root/'FIRST_FAILURE.json') if (root/'FIRST_FAILURE.json').is_file() else None)
def historical(parent,seals):
 root=pathlib.Path(parent);out=[]
 for rel,(h,count) in seals.items():
  p=root/rel;m=p/'hashes/artifact_sha256.txt';need(digest(m)==h,'Historical manifest changed: '+rel)
  lines=m.read_text().splitlines();need(len(lines)==count,'Historical file count changed')
  for line in lines:
   expected,name=line.split('  ',1);need(digest(safe(p,name))==expected,'Historical sealed bytes changed: '+rel+'/'+name)
  out.append(dict(root=str(p),manifest_sha256=h,sealed_files=count,receipt_sha256=digest(p/'hashes/DELIVERY_VERIFICATION.json'),status='PASS_UNCHANGED'))
 return out
def post_dependencies(root):
 p=root/'analysis/FINAL_DEPENDENCY_VERIFICATION.json';need(p.is_file(),'Fresh post-GPU current dependency stream verification is absent')
 r=json.loads(p.read_text());need(r['status']=='PASS' and r['task_id']==TASK and r['stage']=='POST_GPU_FINAL_CURRENT_DEPENDENCY_CONTENT','Post-run dependency verification did not pass')
 ledger=json.loads((root/'gpu_resource_ledger.json').read_text())
 need(r['GPU_coordinator_ended_unix']==ledger['ended_unix'] and r['verification_started_unix']>=ledger['ended_unix'],'Dependency verification was not after this task final GPU end')
 need(r['source_manifest_sha256']==digest(root/'source_manifest.json') and r['final_gate_sha256']==digest(root/'GATE_STATUS_FINAL.json') and r['final_resource_ledger_sha256']==digest(root/'gpu_resource_ledger.json'),'Dependency receipt final identities changed')
 source=safe(root,r['verification_source_relative']);need(digest(source)==r['verification_source_sha256'],'Current dependency verifier source differs from its receipt')
 model=json.loads((root/'qualification_cpu/MODEL_CONTENT_IDENTITY.json').read_text())['model_weight_and_config_current_content_hashes']
 expected={e['path']:{k:e[k] for k in ('bytes','sha256')} for e in model}
 rows=r['model_weight_and_config_files'];need(len(rows)==r['model_files_verified']==14 and len({x['path'] for x in rows})==14,'Fourteen current model files were not freshly streamed')
 need({x['path']:{k:x[k] for k in ('bytes','sha256')} for x in rows}==expected,'Post-run model expected content identities differ')
 runtime=r['runtime_source_files'];need(bool(runtime) and len(runtime)==r['runtime_source_files_verified'],'Current actual runtime dependencies absent')
 for x in rows+runtime:
  need(x['status']=='PASS' and x['fresh_full_stream_hash'] is True and x['unchanged_during_hash'] is True,'Dependency does not carry a genuine fresh stream/stat result')
  q=pathlib.Path(x['path']);s=q.stat();current=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
  need(current==x['stat'],'Dependency stat changed after final full stream verification: '+str(q))
 return dict(status='PASS',receipt='analysis/FINAL_DEPENDENCY_VERIFICATION.json',receipt_sha256=digest(p),
             model_files_freshly_streamed=14,runtime_source_files_freshly_streamed=len(runtime),
             post_stream_current_stat_recheck='PASS',verification_source_sha256=r['verification_source_sha256'],
             verification_started_unix=r['verification_started_unix'],verified_unix=r['verified_unix'])
def upload(root,args):
 p=safe(root,args['relative']);need(not p.exists(),'Refuse overwrite of existing remote artifact')
 p.parent.mkdir(parents=True,exist_ok=True);tmp=None
 try:
  with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.delivery_upload_',delete=False) as f:
   tmp=pathlib.Path(f.name);h=hashlib.sha256();n=0
   for b in iter(lambda:sys.stdin.buffer.read(1024**2),b''):f.write(b);h.update(b);n+=len(b)
   f.flush();os.fsync(f.fileno())
  need(n==args['bytes'] and h.hexdigest()==args['sha256'],'Uploaded bytes do not match requested identity')
  os.link(tmp,p)
 finally:
  if tmp is not None and tmp.exists():tmp.unlink()
 return dict(path=args['relative'],sha256=digest(p),bytes=p.stat().st_size,created_exclusively=True)
def main(args):
 root=pathlib.Path(args['root']);need(root.is_dir() and root.name==TASK and root.resolve().name==TASK,'Root must be this independent new task')
 mode=args['mode']
 if mode=='readiness':
  b=boundary(root);p=root/'analysis_scripts/final_dependency_verify.py'
  return dict(boundary=b,remote_receipt_exists=(root/'hashes/DELIVERY_VERIFICATION.json').exists(),
              dependency_receipt_exists=(root/'analysis/FINAL_DEPENDENCY_VERIFICATION.json').exists(),
              dependency_source=dict(sha256=digest(p),bytes=p.stat().st_size) if p.is_file() else None)
 if mode=='stream':
  p=safe(root,args['relative']);need(p.is_file() and not p.is_symlink(),'Download source not a regular asset')
  s=p.stat();before=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):sys.stdout.buffer.write(b)
  sys.stdout.buffer.flush();s=p.stat();need(before==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed during download');return None
 if mode=='upload':return upload(root,args)
 if mode in ('prepare','verify'):
  b=boundary(root);f=frozen(root);hist=historical(args['old_parent'],args['historical_seals']);result=scan(root,second=mode=='verify')
  result.update(boundary=b,frozen=f,historical_seals=hist,post_run_dependencies=post_dependencies(root),remote_receipt_exists=(root/'hashes/DELIVERY_VERIFICATION.json').exists(),
                remote_manifest_sha256=digest(root/'hashes/artifact_sha256.txt') if (root/'hashes/artifact_sha256.txt').is_file() else None)
  if mode=='verify':
   need(result['assets']==args['expected_assets'],'Independent second full-stream asset identity failed')
   m=root/'hashes/artifact_sha256.txt';need(m.is_file() and digest(m)==args['manifest_sha256'],'Remote manifest SHA mismatch')
   parsed={}
   for line in m.read_text().splitlines():
    h,rel=line.split('  ',1);need(rel not in parsed,'Duplicate manifest path');parsed[rel]=h
   need(parsed=={k:v['sha256'] for k,v in result['assets'].items()},'Manifest inventory differs from actual complete payload')
  return result
 if mode=='final_metadata':
  b=boundary(root);need(b==args['boundary'],'Release/failure/status evidence changed during delivery')
  for rel,h in args['seal_hashes'].items():need(digest(safe(root,rel))==h,'Seal metadata SHA mismatch')
  current=[];total=0
  for d,dirs,files in os.walk(root,followlinks=False):
   for name in dirs:need(not (pathlib.Path(d)/name).is_symlink(),'Symlink directory appeared during final seal')
   for name in files:
    p=pathlib.Path(d)/name;need(not p.is_symlink(),'Symlink appeared during final seal');total+=p.stat().st_size
    rel=str(p.relative_to(root))
    if not ignored(rel):current.append(rel)
  need(set(current)==set(args['expected_paths']),'Asset set changed after second verification')
  need(total<=CAP,'Final physical archive with receipt exceeds10GiB')
  return dict(status='PASS',actual_total_regular_bytes=total,seal_hashes=args['seal_hashes'],boundary=b)
 raise ValueError('Unsupported helper mode')
if __name__=='__main__':
 result=main(json.loads(sys.argv[1]))
 if result is not None:print(json.dumps(result,separators=(',',':')),flush=True)
'''


def need(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def item(path):
    p=pathlib.Path(path)
    return dict(sha256=sha(p),bytes=p.stat().st_size)


def ignored(relative):
    p=pathlib.PurePosixPath(relative)
    return '__pycache__' in p.parts or p.suffix=='.pyc' or p.name.startswith('._')


def safe(root,relative):
    need(isinstance(relative,str) and relative and not any(ord(c)<32 for c in relative),'Unsafe artifact path')
    r=pathlib.PurePosixPath(relative)
    need(not r.is_absolute() and '..' not in r.parts,'Artifact escapes task root')
    p=root/pathlib.Path(*r.parts)
    need(p.resolve().is_relative_to(root.resolve()),'Artifact symlink escapes task root')
    return p


def local_assets(root):
    out={}
    for directory,dirs,files in os.walk(root,followlinks=False):
        need(all(not (pathlib.Path(directory)/d).is_symlink() for d in dirs),'Local symlink directory is not silently followed')
        for name in files:
            p=pathlib.Path(directory)/name;need(p.is_file() and not p.is_symlink(),'Local asset is not a regular in-root file')
            rel=str(p.relative_to(root));safe(root,rel)
            if not ignored(rel) and rel not in (MANIFEST,RECEIPT):
                out[rel]=item(p)
    return out


def download_policy(relative,record):
    p=pathlib.PurePosixPath(relative);suffix=p.suffix.lower();n=record['bytes']
    tensor=(suffix in ('.pt','.pth','.ckpt','.bin','.safetensors','.npy','.npz','.pkl','.pickle')
            or (p.parts and p.parts[0] in ('scores','checkpoints') and suffix not in ('.json','.jsonl','.txt','.csv')))
    if suffix=='.safetensors':
        return False,'MODEL_OR_TENSOR_SAFETENSORS_NOT_DOWNLOADED'
    if tensor and n>TENSOR_LIMIT:
        return False,'LARGE_TENSOR_GT_16MiB_REMOTE_ONLY'
    return True,'SMALL_ACTUAL_ASSET'


def runtime_owned_asset(relative):
    """Scientific/runtime evidence must exist on the authoritative compute host."""
    p=pathlib.PurePosixPath(relative)
    return (bool(p.parts) and p.parts[0] in ('scores','checkpoints','analysis')) or relative in (
        'science_results.json','science_results.csv','window_results.csv','FIRST_FAILURE.json',
        'gpu_resource_ledger.json','GATE_STATUS_FINAL.json','gpu_admission.json','CANARY_INITIAL.json',
        'COORDINATOR_STARTED.json','G4_RUNTIME_BUDGET.json','FINAL_STATUS.json','FINAL_SCIENTIFIC_REPORT.md',
        'scale_safety.json','runtime_cost.json','numerical_qualification.json','codec_parity.json')


def command(payload):
    return SSH+[shlex.quote(PYTHON)+' -c '+shlex.quote(REMOTE_HELPER)+' '+shlex.quote(json.dumps(payload,separators=(',',':')))]


def remote(payload):
    return json.loads(subprocess.check_output(command(payload),text=True))


def publish_bytes(path,data):
    """New-only atomic publication on the target filesystem; exact reuse allowed."""
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        need(path.is_file() and path.read_bytes()==data,'Refuse overwrite of existing local artifact: '+str(path))
        return False
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.delivery_local_',delete=False) as f:
            tmp=pathlib.Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return True


def upload_new(local_root,remote_root,relative,record):
    payload=dict(mode='upload',root=remote_root,relative=relative,**record)
    with safe(local_root,relative).open('rb') as stream:
        result=subprocess.run(command(payload),stdin=stream,stdout=subprocess.PIPE,text=False,check=True)
    actual=json.loads(result.stdout)
    need({k:actual[k] for k in ('sha256','bytes')}==record,'Remote new-only upload identity mismatch')
    return record


def download_new(root,remote_root,relative,record):
    target=safe(root,relative)
    if target.exists():
        need(item(target)==record,'Existing local artifact conflicts with authoritative remote bytes: '+relative)
        return False
    target.parent.mkdir(parents=True,exist_ok=True);tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.delivery_download_',delete=False) as f:
            tmp=pathlib.Path(f.name)
            subprocess.run(command(dict(mode='stream',root=remote_root,relative=relative)),stdout=f,check=True)
            f.flush();os.fsync(f.fileno())
        need(item(tmp)==record,'Downloaded asset SHA/bytes failed: '+relative)
        os.link(tmp,target)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return True


def verify_local_history(parent):
    out=[]
    for relative,(digest,count) in SEALS.items():
        root=pathlib.Path(parent)/relative;m=root/'hashes/artifact_sha256.txt'
        need(sha(m)==digest,'Local historical manifest changed: '+relative)
        lines=m.read_text().splitlines();need(len(lines)==count,'Local historical count changed')
        for line in lines:
            expected,name=line.split('  ',1)
            need(sha(safe(root,name))==expected,'Local historical sealed bytes changed: '+relative+'/'+name)
        out.append(dict(root=str(root),status='PASS_UNCHANGED',manifest_sha256=digest,sealed_files=count,
                        receipt_sha256=sha(root/'hashes/DELIVERY_VERIFICATION.json')))
    return out


def verify_local_stepquant_origins(parent):
    manifest=pathlib.Path(parent)/'inherited_cpu_v1/source_manifest.json'
    entries=[e for e in json.loads(manifest.read_text())['files'] if e['path'].startswith('/Users/')]
    need(len(entries)==4,'Four explicit local STEP reference origins expected')
    out=[]
    for e in entries:
        p=pathlib.Path(e['path']);before=p.stat();actual=item(p);after=p.stat()
        need(actual=={k:e[k] for k in ('sha256','bytes')}
             and (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)==(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev),
             'Current local STEP original source changed: '+str(p))
        out.append(dict(path=str(p),**actual,status='PASS_UNCHANGED',fresh_stream_hash=True))
    return out


def manifest_bytes(assets):
    return ''.join(v['sha256']+'  '+name+'\n' for name,v in sorted(assets.items())).encode()


def write_receipt(root,remote_root,receipt,second):
    data=(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    need(second['total_regular_bytes']+len(data)<=CAP,'Manifest+receipt would exceed10GiB archive cap')
    need(not (root/RECEIPT).exists(),'Existing local final seal receipt may not be overwritten')
    publish_bytes(root/RECEIPT,data)
    upload_new(root,remote_root,RECEIPT,item(root/RECEIPT))
    return sha(root/RECEIPT)


def seal(config_path):
    config_path=pathlib.Path(config_path).resolve();cfg=json.loads(config_path.read_text())
    need(cfg.get('task_id')==TASK,'ROOT.json is not this independent new task')
    root=pathlib.Path(cfg['local_root']).resolve();rr=cfg['remote_root']
    need(root.is_dir() and root.name==TASK and pathlib.PurePosixPath(rr).name==TASK,'New task root identity failed')
    need(not (root/RECEIPT).exists(),'Existing final local seal receipt; use an independent read-only verifier instead of overwriting')
    request=dict(root=rr,old_parent=cfg['old_parent_remote'],historical_seals=SEALS)
    started=time.time();ready=remote(dict(mode='readiness',root=rr))
    need(ready['remote_receipt_exists'] is False,'Existing remote final seal receipt may not be overwritten')
    dependency_source=BASE/'final_dependency_verify.py';need(dependency_source.is_file(),'Companion final_dependency_verify.py is missing')
    publish_bytes(root/DEPENDENCY_SOURCE,dependency_source.read_bytes())
    dependency_identity=item(root/DEPENDENCY_SOURCE)
    if ready['dependency_source'] is None:
        upload_new(root,rr,DEPENDENCY_SOURCE,dependency_identity)
    else:need(ready['dependency_source']==dependency_identity,'Existing remote current-dependency verifier source conflicts; no overwrite')
    if not ready['dependency_receipt_exists']:
        cmd=shlex.quote(PYTHON)+' '+shlex.quote(rr+'/'+DEPENDENCY_SOURCE)+' --root '+shlex.quote(rr)+' --old-parent '+shlex.quote(cfg['old_parent_remote'])
        subprocess.run(SSH+[cmd],check=True)
    first=remote(dict(request,mode='prepare'))
    need(first['remote_receipt_exists'] is False,'Existing remote final seal receipt may not be overwritten')
    local_history=verify_local_history(cfg['old_parent_local'])
    local_stepquant=verify_local_stepquant_origins(cfg['old_parent_local'])
    # Archive this independent delivery program/config as ordinary new assets.
    # Neither file is a frozen GPU execution source or a historical asset.
    publish_bytes(root/SOURCE_SNAPSHOT,pathlib.Path(__file__).read_bytes())
    publish_bytes(root/CONFIG_SNAPSHOT,config_path.read_bytes())
    assets=dict(first['assets']);local=local_assets(root);uploaded=[]
    for name,value in sorted(local.items()):
        if name in assets:
            need(value==assets[name],'Existing remote/local bytes conflict; neither copy will be overwritten: '+name)
        else:
            eligible,reason=download_policy(name,value)
            need(eligible,'Unexpected local-only large tensor cannot be uploaded as a small delivery asset: '+name)
            need(not runtime_owned_asset(name),'Scientific/runtime evidence is absent remotely; refuse to manufacture it from a local-only file: '+name)
            need(first['total_regular_bytes']+sum(x['bytes'] for x in uploaded)+value['bytes']<CAP,
                 'Adding local-only delivery asset would exhaust10GiB cap')
            upload_new(root,rr,name,value);assets[name]=value;uploaded.append(dict(path=name,**value))
    downloaded=[];mirrored=[];remote_only=[];large_already_local=[]
    for name,value in sorted(assets.items()):
        eligible,reason=download_policy(name,value);path=safe(root,name)
        if not eligible:
            if path.exists():
                need(item(path)==value,'Pre-existing local large asset differs from remote: '+name)
                large_already_local.append(dict(path=name,**value,reason='ALREADY_PRESENT_BEFORE_TOOL; no new tensor download'))
            else:remote_only.append(dict(path=name,**value,reason=reason))
            continue
        if download_new(root,rr,name,value):
            downloaded.append(name)
        mirrored.append(name)
    manifest=manifest_bytes(assets)
    publish_bytes(root/MANIFEST,manifest)
    manifest_sha=sha(root/MANIFEST)
    # Upload only when absent; a pre-existing partial manifest must already
    # agree exactly. No existing sealed manifest is rewritten.
    if first['remote_manifest_sha256'] is not None:
        need(first['remote_manifest_sha256']==manifest_sha,'Existing remote partial manifest differs; no overwrite')
    else:
        # Initial in-process metadata proved absence. Atomic exclusive upload
        # still refuses a concurrent publication, without replacing anything.
        upload_new(root,rr,MANIFEST,item(root/MANIFEST))
    second=remote(dict(request,mode='verify',expected_assets=assets,manifest_sha256=manifest_sha))
    need(second['assets']==assets and second['boundary']==first['boundary'],'Independent remote verification or release/failure identity changed')
    need(second['frozen']==first['frozen'],'Frozen source identity changed during delivery')
    need(second['post_run_dependencies']==first['post_run_dependencies'],'Final dependency proof identity/stat changed during delivery')
    fresh_local=local_assets(root)
    expected_local=set(mirrored)|{x['path'] for x in large_already_local}
    need(set(fresh_local)==expected_local,'Local payload asset set changed during delivery')
    for name in expected_local:
        need(fresh_local[name]==assets[name],'Independent second local SHA/bytes verification failed: '+name)
    need(sha(root/MANIFEST)==manifest_sha,'Local manifest changed after independent checks')
    need(sha(root/SOURCE_SNAPSHOT)==sha(__file__),'Delivery tool source changed during sealing')
    receipt=dict(status='PASS',task_id=TASK,local_root=str(root),remote_root=rr,
        manifest=MANIFEST,manifest_sha256=manifest_sha,remote_payload_files_verified=len(assets),
        small_files_verified_both_copies=len(mirrored),new_small_downloads=len(downloaded),downloaded_paths=downloaded,
        new_local_only_small_assets_uploaded=uploaded,remote_only_tensors=remote_only,
        large_assets_already_present_locally=large_already_local,all_payload_assets=assets,
        total_payload_bytes=sum(v['bytes'] for v in assets.values()),
        actual_regular_archive_bytes_before_final_receipt=second['total_regular_bytes'],MAX_NEW_ARCHIVE_BYTES=CAP,
        ignored_non_delivery_files=second['ignored_non_delivery_files'],
        remote_full_stream_verifications=2,remote_first_method='Python hashlib SHA256,16MiB chunks',
        remote_second_independent_executables=second['independent_hash_executables'],
        local_download_verification='All downloads hash+bytes before atomic publish; second independent re-read of every local mirrored asset',
        frozen_source_verification=second['frozen'],GPU_release_barrier=second['boundary'],
        post_run_current_dependency_verification=second['post_run_dependencies'],current_local_STEP_original_sources=local_stepquant,
        GPU_HOURS_USED=second['boundary']['GPU_HOURS_USED'],GPU_work_performed=False,tensor_deserialization=False,
        local_historical_seals=local_history,remote_historical_seals=second['historical_seals'],
        historical_scope='CPU70/parent140/oldexecution01=24/oldexecution02=43 sealed files read-only; no new claim of rehashing all external NoInflation/AIME large tensors',
        failure_history_preserved=True,first_failure_sha256=second['boundary']['first_failure_sha256'],
        tool_source_snapshot=SOURCE_SNAPSHOT,tool_source_sha256=sha(__file__),ROOT_config_snapshot=CONFIG_SNAPSHOT,ROOT_config_sha256=sha(config_path),
        self_reference_exclusion='Checksum manifest excludes itself and this final receipt. Receipt binds manifest SHA. Its own SHA is independently compared at both endpoints and returned by tool.',
        receipt_and_manifest_publication='New-only atomic creation; existing conflicting files and completed seals are never overwritten',
        started_unix=started,verified_unix=time.time())
    receipt_sha=write_receipt(root,rr,receipt,second)
    seal_hashes={MANIFEST:manifest_sha,RECEIPT:receipt_sha}
    final=remote(dict(mode='final_metadata',root=rr,boundary=second['boundary'],seal_hashes=seal_hashes,
        expected_paths=sorted(list(assets)+[MANIFEST,RECEIPT])))
    need(all(sha(root/name)==digest for name,digest in seal_hashes.items()),'Final local seal metadata identity failed')
    return dict(status='PASS',task_id=TASK,remote_payload_files=len(assets),small_files_verified_both_copies=len(mirrored),
        remote_only_tensors=len(remote_only),manifest_sha256=manifest_sha,receipt_sha256=receipt_sha,
        actual_remote_regular_bytes_including_seal=final['actual_total_regular_bytes'],GPU_work_performed=False,
        delivery_receipt=str(root/RECEIPT),science_scores_deserialized=False)


def self_test():
    """Real tiny files + synthetic metadata, no scientific assets or network."""
    ns={'__name__':'delivery_helper_selftest'};exec(compile(REMOTE_HELPER,'<embedded_readonly_helper>','exec'),ns)
    with tempfile.TemporaryDirectory(prefix='qwen_delivery_mock_') as td:
        root=pathlib.Path(td)/TASK;root.mkdir();entries=[]
        names=['src/worker.py','src/coordinator.py','preregistration.json','frozen_documents.json']+[f'src/fixture_{i}.txt' for i in range(52)]
        for name in names:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SYNTHETIC_SELFTEST_ONLY '+name)
            entries.append(dict(path=name,**item(p)))
        source=dict(task=TASK,frozen_before_any_new_GPU_forward=True,execution_sources=entries,
            old_execution02_seal_sha256='SYNTHETIC_OLD_IDENTITY',model_identity_receipt_sha256='SYNTHETIC_MODEL_RECEIPT')
        (root/'source_manifest.json').write_text(json.dumps(source))
        gate=dict(task_id=TASK,status='BLOCKED_AND_STOP',GPU_HOURS_USED=0.,no_further_GPU_work=True,GPU_release_verified=True)
        ledger=dict(task_id=TASK,GPU_HOURS_USED=0.,final_GPU_release_verified=True,phases=[])
        (root/'GATE_STATUS_FINAL.json').write_text(json.dumps(gate));(root/'gpu_resource_ledger.json').write_text(json.dumps(ledger))
        (root/'FINAL_STATUS.json').write_text(json.dumps(dict(TASK=TASK,FINAL_EXECUTION_STATUS='BLOCKED_AND_STOP')))
        (root/'FINAL_SCIENTIFIC_REPORT.md').write_text('SYNTHETIC_SELFTEST_ONLY; no scientific result')
        (root/'__pycache__').mkdir();(root/'__pycache__/ignored.pyc').write_bytes(b'BYTECODE_MOCK')
        assert ns['frozen'](root)['verified_execution_sources']==56
        assert ns['boundary'](root)['GPU_release_verified'] is True
        first=ns['scan'](root);second=ns['scan'](root,second=True)
        assert first['assets']==second['assets'] and second['independent_hash_executables']
        assert all(not ignored(n) for n in first['assets']) and first['ignored_non_delivery_files']
        bad=[]
        for name in ('../escape','/escape','line\nname'):
            try:safe(root,name)
            except ValueError:bad.append(name)
            else:raise AssertionError('unsafe path was accepted')
        p=root/'new_only.txt';assert publish_bytes(p,b'ORIGINAL') is True and publish_bytes(p,b'ORIGINAL') is False
        try:publish_bytes(p,b'ALTERED')
        except ValueError:pass
        else:raise AssertionError('existing local artifact overwritten')
        assert p.read_bytes()==b'ORIGINAL'
        assert download_policy('scores/science.pt',dict(bytes=TENSOR_LIMIT+1))[0] is False
        assert download_policy('checkpoints/CURRENT.tmp',dict(bytes=TENSOR_LIMIT+1))[0] is False
        assert download_policy('analysis/large_report.json',dict(bytes=TENSOR_LIMIT+1))[0] is True
        assert download_policy('scores/small_canary.pt',dict(bytes=1024))[0] is True
        assert download_policy('model.safetensors',dict(bytes=100))[0] is False
        assert runtime_owned_asset('science_results.csv') and runtime_owned_asset('analysis/science_doc1_H_C128.json')
        assert runtime_owned_asset('analysis/N_OFF.json') and runtime_owned_asset('analysis/ENV_canary_initial.json')
        assert runtime_owned_asset('analysis/FINAL_DEPENDENCY_VERIFICATION.json') and runtime_owned_asset('FINAL_STATUS.json')
        assert runtime_owned_asset('scale_safety.json') and runtime_owned_asset('numerical_qualification.json')
        assert not runtime_owned_asset('analysis_scripts/finalize_report.py')
        payload=manifest_bytes(first['assets']).decode();assert MANIFEST not in payload and RECEIPT not in payload
        original=(root/'src/worker.py').read_bytes();(root/'src/worker.py').write_bytes(b'TAMPERED')
        try:ns['frozen'](root)
        except ValueError:pass
        else:raise AssertionError('frozen source tamper was accepted')
        (root/'src/worker.py').write_bytes(original)
        gate['GPU_release_verified']=False;(root/'GATE_STATUS_FINAL.json').write_text(json.dumps(gate))
        try:ns['boundary'](root)
        except ValueError:pass
        else:raise AssertionError('unreleased GPU barrier accepted')
    return dict(status='PASS',scope='Disposable synthetic metadata and tiny files only; no network, GPU, real scores, archive, or seal',
        frozen_source_count=56,independent_stream_passes=2,unsafe_paths_rejected=len(bad),
        refuses_existing_file_overwrite=True,refuses_unreleased_GPU=True,detects_frozen_source_tamper=True,
        remote_only_tensor_policy='PASS',self_reference_exclusion='PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(BASE/'ROOT.json'));parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    print(json.dumps(self_test() if args.self_test else seal(args.config),ensure_ascii=False),flush=True)
