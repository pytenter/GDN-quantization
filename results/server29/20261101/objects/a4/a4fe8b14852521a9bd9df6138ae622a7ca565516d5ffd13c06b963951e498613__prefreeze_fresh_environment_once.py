"""Fresh, bounded CPU/read-only environment admission for this revision only."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage', ROOT/'operations/stage_reduction_cpu.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

COMMON = r'''
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
digest_cache={}
def digest(p):
 p=Path(p);a=p.stat();key=(str(p.resolve()),a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)
 if key in digest_cache:return digest_cache[key]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 z=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)
 digest_cache[key]=h.hexdigest()
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert os.environ['CUDA_VISIBLE_DEVICES']==''
'''

CODE55 = COMMON + r'''
r=Path('/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025')
old=Path('/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1')
assert r.is_dir() and not r.is_symlink()
assert not any((r/n).exists() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','GPU_RUN_STARTED.json','SESSION_RESULT.json'])
assert digest(old/'SOURCE_FREEZE.json')=='ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7'
for row in read(old/'SOURCE_FREEZE.json')['files']:
 p=old/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
groups={}
for name in ['EXTERNAL_INPUTS_FRESH_SHA','MODEL_TOKENIZER_FRESH_SHA','TRITON_RUNTIME_FRESH_SHA']:
 d=read(r/('analysis/'+name+'.json'));rows=d['files']+d.get('distribution_metadata',[])+d.get('host_c_compiler',[])
 groups[name]=[]
 for x in rows:
  p=Path(x['path']);s=p.stat();assert s.st_size==x['bytes'] and digest(p)==x['sha256'],str(p)
  if name=='MODEL_TOKENIZER_FRESH_SHA':assert (s.st_ino,s.st_mtime_ns)==(x['inode'],x['mtime_ns']),str(p)
  groups[name].append(dict(path=str(p),bytes=s.st_size,sha256=x['sha256'],inode=s.st_ino,mtime_ns=s.st_mtime_ns))
sys.path.insert(0,str(r/'src'))
import execution_control as C
import non_aime_diagnostics as D
docs,anchors,drives,raw=D._raw_inventory(r)
C.model_guard(r)  # All bound external rows are freshly SHA-verified above.
inventory=[C.device(g) for g in C.AUTHORIZED_GPUS]
storage=C.storage_guard(r)
mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
shm=os.statvfs('/dev/shm');shmfree=shm.f_bavail*shm.f_frsize
assert mem['MemAvailable']>=100*1024**3 and shmfree>=100*1024**3
assert not C.reference_root().exists(),'Independent teacher reference path already exists'
assert not list((r/'execution').glob('*/DISPATCH.json')) and not list((r/'formal').glob('*/DISPATCH.json'))
identities=[]
for app in inventory[0]['apps']:
 try:identities.append(C.process_identity(app['pid']))
 except (OSError,ValueError):identities.append(dict(pid=app['pid'],identity='ALREADY_TERMINAL_OR_UNRESOLVED'))
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True,timeout=10)
assert 'qwen-wscale-rowbal-byteview-repair-20261025' not in sessions.stdout.splitlines()
assert 'torch' not in sys.modules and 'triton' not in sys.modules
print(json.dumps(dict(status='PASS_PREFREEZE_FRESH55_SOURCE_EXTERNAL_MODEL_TRITON_SHM_STORAGE_SELECTED_UUIDS',
 groups=groups,prior100_source_unchanged=True,shared_calibration_raw=raw,storage=storage,
 MemAvailable=mem['MemAvailable'],shm_free_bytes=shmfree,selected_UUID_admission=inventory,
 actual_foreign_GPU_identities=identities,tmux_sessions=sessions.stdout.splitlines(),
 compute_GPU_calls=0,driver_contexts=0,model_loads=0,model_downloads=0,NVML='READ_ONLY_INVENTORY',monotonic=time.monotonic())))
'''

CODE29 = COMMON + r'''
parent=Path('/data/zypan/qwen_gdn_wscale_rowbal_archive')
target=parent/'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025'
assert parent.is_dir() and not target.exists()
s=os.statvfs(parent);free=s.f_bavail*s.f_frsize
assert free>=120*1024**3
old=parent/'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
assert digest(old/'SOURCE_FREEZE.json')=='ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7'
for x in read(old/'SOURCE_FREEZE.json')['files']:
 p=old/x['path'];assert p.stat().st_size==x['bytes'] and digest(p)==x['sha256']
print(json.dumps(dict(status='PASS_PREFREEZE29_ARCHIVE_TARGET_ABSENT_HEADROOM_PRIOR100_UNCHANGED',
 archive_target=str(target),free_bytes=free,prior100_source_unchanged=True,compute_GPU_calls=0,monotonic=time.monotonic())))
'''


if __name__ == '__main__':
    output = ROOT/'operations/BYTEVIEW_FRESH_ENVIRONMENT_ADMISSION.json'
    assert not output.exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f55 = pool.submit(stage.transport.run, CODE55, role='55', python='/data/ydai/miniconda3/envs/bitdecode/bin/python', timeout=240)
        f29 = pool.submit(stage.transport.run, CODE29, role='29', python='/usr/bin/python3', timeout=90)
        result = dict(status='PASS_DUAL_HOST_PREFREEZE_ENVIRONMENT_ADMISSION',
                      UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      host55=f55.result(), host29=f29.result(), GPU_dispatches=0)
    with output.open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],receipt=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
