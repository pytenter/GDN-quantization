"""Fresh dual-host provenance review and ordinary copy, no GPU or calibration rerun."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[1]
TASK='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
spec=importlib.util.spec_from_file_location('stage',ROOT/'operations/stage_reduction_cpu.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)

CODE=r'''
from pathlib import Path
import hashlib,json,os,shutil,time
def sha(p):
 p=Path(p);a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 z=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(root,rows):
 for x in rows:
  n=Path(x['path']);assert not n.is_absolute() and '..' not in n.parts
  p=root/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
origin=Path(ORIGIN);parent=Path(PARENT)
assert sha(origin/'SOURCE_FREEZE.json')=='ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7'
assert sha(parent/'SOURCE_FREEZE.json')=='6e87495d2dac7ccc7d97363256be3862ab7a52159271b848a58f06e2eabd1f4e'
source=read(origin/'SOURCE_FREEZE.json');assert source['count']==len(source['files'])==100;check(origin,source['files'])
ps=read(parent/'SOURCE_FREEZE.json');assert ps['count']==len(ps['files'])==299;check(parent,ps['files'])
cal=read(origin/'CALIBRATION_SOURCE_FREEZE.json');assert len(cal['files'])==31;check(origin,cal['files'])
old_manifest=read(origin/'SHA256_MANIFEST.json');by={x['path']:x for x in old_manifest['files']}
for name in ('calibration/RAW_CLONE_VERIFICATION.json','CALIBRATION_PARAMETER_SELECTION.json','TRANSPORTED_QUERY_OMEGA.json','calibration/FP_COLLECTION_EXIT_RELEASE.json','calibration/PARENT_FAILURE.json'):
 check(origin,[by[name]])
raw=read(origin/'calibration/RAW_CLONE_VERIFICATION.json');assert raw['count']==len(raw['files'])==4323
check(origin,raw['files']);assert sum(x['bytes'] for x in raw['files'])==raw['bytes']==1300245388
fp=read(origin/'calibration/FP_COLLECTION_MANIFEST.json');exit=read(origin/'calibration/FP_COLLECTION_EXIT_RELEASE.json')
assert exit['status']=='PASS_ACTUAL_EXIT0_AND_RELEASE' and exit['exit_returncode']==0 and exit['release']=='PASS'
assert exit['worker_identity']==fp['identity'] and exit['collection_manifest_sha256']==sha(origin/'calibration/FP_COLLECTION_MANIFEST.json')
sel=read(origin/'CALIBRATION_PARAMETER_SELECTION.json');assert sel['selected']['a']==.5 and sel['selected']['b']==.25
assert sha(origin/'TRANSPORTED_QUERY_OMEGA.json')=='7061082f1f85e988e5f3de299d2fbdda61c553aeb8499f507e62711ab8820a67'
assert read(parent/'FINAL_TASK_STATUS.json')['status']=='PARTIAL_RESULTS_PRESERVED_AND_STOP'
assert not (parent/'FINALIZER_FAILURE.json').exists()
assert read(parent/'FINAL_SEAL_RECEIPT.json')['status']=='PASS_FINAL_SEAL'
assert read(parent/'FINAL_ARCHIVE_VERIFICATION.json')['manifest_sha256']==sha(parent/'SHA256_MANIFEST.json')
assert read(parent/'SHA256_MANIFEST.json')['count']==4915
if ROLE=='55':
 new=Path(NEWROOT);assert not new.is_symlink()
 assert not any((new/n).exists() for n in ('SOURCE_FREEZE.json','GPU_PERMIT.json','GPU_RUN_STARTED.json'))
 assert not (new/'calibration/raw').exists(),'New raw destination exists; preserve and inspect instead of duplicate copy'
 space=os.statvfs(new.parent);assert space.f_bavail*space.f_frsize>=raw['bytes']+4*1024**3
 new.mkdir(exist_ok=True)
 for x in raw['files']:
  p=new/x['path'];p.parent.mkdir(parents=True,exist_ok=True)
  with (origin/x['path']).open('rb') as src,p.open('xb') as dst:
   shutil.copyfileobj(src,dst,4*1024**2);dst.flush();os.fsync(dst.fileno())
 check(new,raw['files'])
 copy=dict(status='PASS_NEW_BYTEVIEW_REGULAR_COPY_OF_FROZEN_SHARED_CALIBRATION_RAW',count=len(raw['files']),bytes=raw['bytes'],catalog_sha256=sha(origin/'calibration/RAW_CLONE_VERIFICATION.json'),origin=str(origin),original_calibration_not_recollected=True,model_weight_copies=0,GPU_calls=0)
 p=new/'operations/CALIBRATION_RAW_COPY_FROM_ORIGIN.json';p.parent.mkdir(exist_ok=True)
 with p.open('x') as f:json.dump(copy,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
else:
 target=Path(NEWARCHIVE);assert not target.exists()
 space=os.statvfs(target.parent);assert space.f_bavail*space.f_frsize>=120*1024**3
 copy=None
names=('SOURCE_FREEZE.json','GPU_PERMIT.json','SHA256_MANIFEST.json','FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','CALIBRATION_SOURCE_FREEZE.json','CALIBRATION_DATA_FREEZE.json','CALIBRATION_PARAMETER_SELECTION.json','TRANSPORTED_QUERY_OMEGA.json','calibration/FP_COLLECTION_MANIFEST.json','calibration/FP_COLLECTION_EXIT_RELEASE.json','calibration/PARENT_FAILURE.json','calibration/RAW_CLONE_VERIFICATION.json')
out=dict(status='PASS_FRESH_FIXED_ORIGIN_SHARED_CALIBRATION_PROVENANCE',root=str(origin),source_count=100,calibration_source_count=31,raw_clone_count=4323,raw_clone_bytes=raw['bytes'],raw_catalog_sha256=sha(origin/'calibration/RAW_CLONE_VERIFICATION.json'),original_collector_exit0_release=True,actual_original_collector_identity=exit['worker_identity'],old_CPU_parent_failure_preserved=True,calibration_or_selection_reruns=0,formal_or_qualification_credit=0,selected_a_b=[.5,.25],pins={n:sha(origin/n) for n in names},parent299_source_unchanged=True,parent299_source_sha256=sha(parent/'SOURCE_FREEZE.json'),parent_STOP_manifest_sha256=sha(parent/'SHA256_MANIFEST.json'),copy=copy,monotonic=time.monotonic(),GPU_calls=0,finisher_calls=0,signals=0)
print(json.dumps(out))
'''

def call(role):
    origin=('/tmp/' if role=='55' else '/data/zypan/qwen_gdn_wscale_rowbal_archive/')+TASK
    parent=origin+'_REDUCTION_REPAIR_20261025'
    prefix='ROLE='+repr(role)+'\nORIGIN='+repr(origin)+'\nPARENT='+repr(parent)+'\nNEWROOT='+repr(stage.REMOTE)+'\nNEWARCHIVE='+repr('/data/zypan/qwen_gdn_wscale_rowbal_archive/'+ROOT.name)+'\n'
    return stage.transport.run(prefix+CODE,role=role,python='/data/ydai/miniconda3/envs/bitdecode/bin/python' if role=='55' else '/usr/bin/python3',timeout=240)

if __name__=='__main__':
    output=ROOT/'CALIBRATION_REUSE_ADMISSION.json'
    prior=ROOT/'provenance/PARENT_REDUCTION_TRUE_STOP/CALIBRATION_REUSE_ADMISSION.json'
    assert prior.is_file() and output.read_bytes()==prior.read_bytes()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={role:pool.submit(call,role) for role in ('55','29')}
        hosts={role:job.result() for role,job in jobs.items()}
    assert hosts['55']['raw_catalog_sha256']==hosts['29']['raw_catalog_sha256']
    result=dict(task_id=TASK,execution_revision=ROOT.name,status='PASS_REUSED_FROZEN_NONAIME_CALIBRATION_DUAL_HOST_FRESH_SHA_NO_RECOLLECT_NO_RESELECT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),origin_source_root='/tmp/'+TASK,origin_archive_root='/data/zypan/qwen_gdn_wscale_rowbal_archive/'+TASK,origin_source_sha256='ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7',shared_omega_sha256='7061082f1f85e988e5f3de299d2fbdda61c553aeb8499f507e62711ab8820a67',selected_a_b=[.5,.25],new_qualification_credit=0,new_formal_credit=0,no_reselection=True,shared_raw_original_shm='/dev/shm/qwen_sym8_g32_wscale_rowbal_calibration_v1',original_failure_not_reclassified=True,parent_reduction_STOP_not_rejudged=True,hosts=hosts)
    output.write_text(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    copy=ROOT/'operations/CALIBRATION_RAW_COPY_FROM_ORIGIN.json'
    # The copied parent's same-named engineering receipt is history, not this copy.
    old=ROOT/'provenance/PARENT_REDUCTION_TRUE_STOP/CALIBRATION_RAW_COPY_FROM_ORIGIN.json'
    old.write_bytes(copy.read_bytes())
    copy.write_text(json.dumps(hosts['55']['copy'],sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],receipt_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),GPU_dispatches=0)))
