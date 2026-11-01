"""One bounded CPU analysis/report/seal pass after controller exit; no GPU."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE='/data/zypan/qwen_c6r_dual_m_interaction_archive/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1'
SSH=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.29']
LIVE_EXCLUSIONS={'logs/OFFLINE_FINISH.log','logs/COORDINATOR_STDOUT.log','hashes/FINAL_ARCHIVE_VERIFICATION.json','hashes/SHA256SUMS.txt','SHA256_MANIFEST.json'}


def read(p):return json.loads(Path(p).read_text())
def need(v,s):
    if not v:raise ValueError(s)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(rel,x):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def error(stage,exc):
    item=dict(status='FAIL_CLOSED_STOP',stage=stage,error=repr(exc),traceback=traceback.format_exc(),unix=time.time(),no_GPU_retry=True)
    save('analysis/CPU_'+stage+'_FAILURE.json',item)
    if not (ROOT/'FIRST_FAILURE.json').exists():save('FIRST_FAILURE.json',item)


def verify_sources():
    m=read(ROOT/'SOURCE_MANIFEST.json')
    checked=[]
    for a in m['execution_sources']+m.get('external_dependencies',[]):
        p=Path(a['path']) if Path(a['path']).is_absolute() else ROOT/a['path']
        need(p.is_file() and p.stat().st_size==a['bytes'] and sha(p)==a['sha256'],'Frozen dependency changed: '+str(p))
        checked.append(dict(path=str(p),sha256=a['sha256'],bytes=a['bytes']))
    need((ROOT/'source_manifest.json').read_bytes()==(ROOT/'SOURCE_MANIFEST.json').read_bytes(),'Manifest alias changed')
    need((ROOT/'preregistration.json').read_bytes()==(ROOT/'PREREGISTRATION.json').read_bytes(),'Preregistration alias changed')
    save('analysis/POSTRUN_SOURCE_VERIFICATION.json',dict(status='PASS',actual_stream_hashed_dependencies=checked,no_GPU=True))


def payload_files():
    seen=set()
    for base,dirs,names in os.walk(ROOT,followlinks=True):
        here=Path(base)
        dirs[:]=sorted(d for d in dirs if d!='__pycache__')
        # Global rolling CP is included once. Per-lane aliases do not claim
        # independent checkpoint retention after their turn has ended.
        if 'qualification_gpu' in here.parts:
            dirs[:]=[d for d in dirs if d!='checkpoints']
        for n in sorted(names):
            p=here/n;rel=str(p.relative_to(ROOT))
            if rel in LIVE_EXCLUSIONS or n.endswith('.pyc') or n.endswith('.tmp') or n.endswith('.lock'):continue
            if p.is_symlink() and not p.exists():continue
            need(p.is_file(),'Nonregular final payload '+rel)
            need(rel not in seen,'Duplicate payload '+rel);seen.add(rel)
            yield rel,p


ARCHIVE_VERIFY_SOURCE=r'''import hashlib,json,os,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);expected=json.loads(sys.argv[2]);need=lambda v,s:None if v else (_ for _ in ()).throw(RuntimeError(s))
need(not (r/'hashes/FINAL_ARCHIVE_VERIFICATION.json').exists(),'Final seal already exists; no overwrite')
def h(p):
 q=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16777216),b''):q.update(b)
 return q.hexdigest()
for a in expected:
 p=r/a['path'];need(p.is_file() and p.stat().st_size==a['bytes'] and h(p)==a['sha256'],'Final copied asset mismatch '+a['path'])
current=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or p.is_symlink() or p.name.endswith(('.lock','.tmp','.pyc')) or '__pycache__' in p.parts:continue
 rel=str(p.relative_to(r))
 if rel in ('hashes/FINAL_ARCHIVE_VERIFICATION.json','SHA256_MANIFEST.json','hashes/SHA256SUMS.txt'):continue
 before=p.stat();digest=h(p);independent=subprocess.check_output(['sha256sum','--',str(p)],text=True).split()[0];after=p.stat()
 need(digest==independent and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Archive streaming verification failed '+rel)
 current.append(dict(path=rel,sha256=digest,bytes=after.st_size,python_stream_SHA=True,independent_sha256sum=True))
# Archived science scores must exactly match the original per-phase location
# receipts; a bare unreferenced tensor cannot pass as a scientific artifact.
for loc in sorted((r/'archive').glob('DOC_*_CURRENT_ASSET_LOCATIONS.json')):
 m=json.loads(loc.read_text());need(m['status']=='PASS','Invalid score location map')
 for a in m['assets']:
  p=pathlib.Path(a['archive_path']);need(p.resolve().is_relative_to(r.resolve()),'Archive escape')
  need(p.stat().st_size==a['bytes'] and h(p)==a['sha256'],'Archived score mapping mismatch')
manifest=dict(TASK_ID=r.name,status='PASS',assets=current,all_current_tensors_verified=True,tensor_count=sum(a['path'].endswith('.pt') for a in current),source_of_truth='permanent archive .29; current actual artifacts only; retired rolling CPs remain catalogued',explicit_nonpayload_live_log='logs/OFFLINE_FINISH.log',verified_unix=time.time())
with (r/'SHA256_MANIFEST.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
md=h(r/'SHA256_MANIFEST.json')
with (r/'hashes/SHA256SUMS.txt').open('x') as f:
 for a in current:f.write(a['sha256']+'  '+a['path']+'\n')
 f.write(md+'  SHA256_MANIFEST.json\n')
receipt=dict(status='PASS',archive_root=str(r),all_current_tensors_verified=True,verified_asset_count=len(current),verified_tensor_count=manifest['tensor_count'],SHA256_MANIFEST_SHA256=md,SHA256SUMS_SHA256=h(r/'hashes/SHA256SUMS.txt'),actual_report_sha256=h(r/'FINAL_SCIENTIFIC_REPORT.md'),no_GPU=True,verified_unix=time.time())
tmp=r/'hashes/FINAL_ARCHIVE_VERIFICATION.tmp'
with tmp.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
os.replace(tmp,r/'hashes/FINAL_ARCHIVE_VERIFICATION.json')
print(json.dumps(receipt))
'''


def seal():
    script=ARCHIVE_VERIFY_SOURCE
    verifier=ROOT/'hashes/FINAL_ARCHIVE_VERIFY_SOURCE.py'
    with verifier.open('x') as f:f.write(script)
    subprocess.run(SSH+[shlex.join(['mkdir','-p',ARCHIVE])],check=True,timeout=30)
    # First bind every still-resident artifact to its pre-copy SHA. All worker
    # and CPU stage writers have stopped; only this excluded live log is open.
    expected=[]
    for rel,p in payload_files():
        a=p.stat();h=sha(p);b=p.stat()
        need((a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino),'Payload changed during final hash')
        expected.append(dict(path=rel,sha256=h,bytes=b.st_size))
    save('hashes/LOCAL_CURRENT_PAYLOAD.json',dict(status='PASS',assets=expected,live_exclusions=sorted(LIVE_EXCLUSIONS),no_GPU=True))
    p=ROOT/'hashes/LOCAL_CURRENT_PAYLOAD.json'
    expected.append(dict(path='hashes/LOCAL_CURRENT_PAYLOAD.json',sha256=sha(p),bytes=p.stat().st_size))
    # Metadata listing is fixed; rsync does not replace a previously archived
    # file. Any disagreement is a real seal failure in the verification below.
    listing=ROOT/'hashes/FINAL_COPY_PATHS.txt'
    with listing.open('x') as f:f.write('\n'.join(a['path'] for a in expected)+'\n')
    expected.append(dict(path='hashes/FINAL_COPY_PATHS.txt',sha256=sha(listing),bytes=listing.stat().st_size))
    argv=['rsync','-aL','--ignore-existing','--files-from='+str(listing),'-e',shlex.join(SSH[:-1]),str(ROOT)+'/',SSH[-1]+':'+ARCHIVE+'/']
    subprocess.run(argv,check=True,timeout=1200)
    # The copy-list is part of the archive too.
    subprocess.run(['scp','-o','BatchMode=yes','-o','ControlPath=/tmp/qwen_replication_archive29.sock',str(listing),SSH[-1]+':'+ARCHIVE+'/hashes/FINAL_COPY_PATHS.txt'],check=True,timeout=60)

    python29=read(ROOT/'archive/ARCHIVE_STORAGE_PREFLIGHT.json')['python']
    need(python29.startswith('/'),'Archive Python identity missing')
    command=shlex.join([python29,'-c',script,ARCHIVE,json.dumps(expected,separators=(',',':'))])
    out=subprocess.run(SSH+[command],capture_output=True,text=True,check=True,timeout=1800)
    for name in ['SHA256_MANIFEST.json','hashes/SHA256SUMS.txt','hashes/FINAL_ARCHIVE_VERIFICATION.json']:
        subprocess.run(['scp','-o','BatchMode=yes','-o','ControlPath=/tmp/qwen_replication_archive29.sock',SSH[-1]+':'+ARCHIVE+'/'+name,str(ROOT/name)],check=True,timeout=120)
    need(read(ROOT/'hashes/FINAL_ARCHIVE_VERIFICATION.json')['status']=='PASS','Archive verification missing')
    print(out.stdout,flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    gpu=read(ROOT/'FINAL_GPU_STATUS.json')
    need(gpu['no_further_GPU_work'] and gpu['GPU_release_verified'],'CPU closeout requires verified release of all own GPU workers')
    stages=[]
    try:verify_sources();stages.append(dict(stage='POSTRUN_SOURCE_VERIFICATION',status='PASS'))
    except BaseException as exc:error('sources',exc);stages.append(dict(stage='POSTRUN_SOURCE_VERIFICATION',status='FAIL'))
    # No retry loop, including CPU analysis. Preserve the first real failure.
    qualified=(ROOT/'GPU_QUALIFICATION.json').is_file() and read(ROOT/'GPU_QUALIFICATION.json').get('status')=='PASS'
    if stages[-1]['status']=='PASS' and qualified:
        try:
            with (ROOT/'logs/CPU_ANALYSIS.log').open('xb') as f:
                p=subprocess.run([sys.executable,str(ROOT/'analysis_scripts/analyze_interaction.py')],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2700)
            stages.append(dict(stage='CPU_ANALYSIS',status='PASS',returncode=p.returncode))
        except BaseException as exc:error('analysis',exc);stages.append(dict(stage='CPU_ANALYSIS',status='FAIL'))
    elif not qualified:
        stages.append(dict(stage='CPU_ANALYSIS',status='NOT_RUN_NEW_GPU_CONDITIONS_NOT_QUALIFIED'))
    with (ROOT/'logs/CPU_REPORT.log').open('xb') as f:
        subprocess.run([sys.executable,str(ROOT/'analysis_scripts/final_report.py')],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    stages.append(dict(stage='REPORT',status='PASS'))
    save('analysis/CPU_CLOSEOUT_STAGES.json',dict(status='PASS' if all(s['status']=='PASS' for s in stages) else 'STOP_WITH_ACTUAL_REPORT',stages=stages,no_GPU=True,no_additional_experiments=True))
    seal()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        traceback.print_exc()
        p=ROOT/'analysis/DELIVERY_STOP.json'
        if not p.exists():save('analysis/DELIVERY_STOP.json',dict(status='FAIL_CLOSED_STOP',error=repr(exc),no_GPU_retry=True,no_automatic_next_experiment=True))
        raise
