"""Once-only CPU STOP publication, independent29SHA and atomic no-clobber."""
import hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
R=Path(sys.argv[1]).resolve()
P=Path('/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022')
ARCH='/data/zypan/qwen_oscar_sym32_c1_s3_v3_archive/'+R.name.upper()
# The root's exact scientific name contains the all-caps model identifier.
ARCH='/data/zypan/qwen_oscar_sym32_c1_s3_v3_archive/QWEN_GDN_OSCAR_SYM32_C1_S3_CANONICAL_RECOVERY_AIME26_V3'
STAGE=ARCH+'.partial_once'
SSH=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','zypan@202.38.247.29']
def load(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 return h.hexdigest()
def save(name,value):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def on29(code):
 result=subprocess.run(SSH+['/usr/bin/python3','-'],input=code,text=True,capture_output=True)
 if result.returncode:raise RuntimeError('Archive29CPU failed:'+result.stderr[-4000:]+result.stdout[-4000:])
 return result.stdout
def actual_identity():
 raw=Path('/proc/self/stat').read_text();return dict(pid=os.getpid(),starttick=int(raw[raw.rfind(')')+2:].split()[19]),cmdline=Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (R/'operations/CPU_STOP_PUBLISHER_STARTED.json').exists()
save('operations/CPU_STOP_PUBLISHER_STARTED.json',dict(identity=actual_identity(),monotonic=time.monotonic(),GPU_dispatches=0,original_tasks_untouched=True))
try:
 assert load(R/'TASK_RESULT.json')['status']=='V3_NUMERICAL_CONTRACT_UNRESOLVED_AND_STOP'
 assert not load(R/'V3_NUMERICAL_QUALIFICATION_CONTRACT.json')['GPU_admission']
 assert load(R/'qualification/INDEPENDENT_CPU_VERIFICATION.json')['status'].startswith('PASS')
 assert load(R/'TOKEN22_FIRST_DIFFERENCE_ANALYSIS.json')['status'].endswith('UNRESOLVED')
 assert not (R/'formal').exists() and not (R/'execution').exists()
 # Full fresh parent source, external and modelstat audit, with no parent writes.
 sys.path.insert(0,str(P/'src'));import execution_control as C
 before=load(R/'P1_RUNNING_TASK_IMMUTABILITY_AUDIT.json')
 for f in before['source_files']:
  q=P/f['path'];assert q.stat().st_size==f['bytes'] and digest(q)==f['sha256']
 assert C.verify_external(P)==61;C.model_guard(P)
 gpu=C.query_gpu();own_gpu=[x for x in gpu['apps'] if str(R) in ' '.join(C.process_identity(x['pid']).get('cmdline',[]))]
 assert own_gpu==[]
 active=[]
 for q in (P/'formal').glob('*/STARTED.json'):
  identity=load(q).get('identity')
  if identity and C.same_process(identity):active.append(dict(unit=q.parent.name,identity=identity,progress=load(q.parent/'PROGRESS.json')))
 save('P1_RUNNING_TASK_POST_AUDIT.json',dict(status='PASS_FRESH622_SOURCE61_DEPENDENCIES17_MODELSTAT_UNCHANGED',source_freeze_sha256=digest(P/'SOURCE_FREEZE.json'),source_files=622,actual_external_SHA=61,model_stat=17,active_processes=active,actual_NVML=gpu,own_V3_GPU_processes=own_gpu,original_root_writes=0,new_GPU_dispatches=0,monotonic=time.monotonic()))
 # Actual atomic primitive positive/negative checks, isolated from old assets.
 prep=on29('''import os,ctypes,errno,json,tempfile
from pathlib import Path
base=Path('''+repr(str(Path(ARCH).parent))+''');base.mkdir(exist_ok=True,parents=True)
assert not Path('''+repr(ARCH)+''').exists() and not Path('''+repr(STAGE)+''').exists()
st=os.statvfs(base);assert st.f_bavail*st.f_frsize>=120*1024**3
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
with tempfile.TemporaryDirectory(prefix='v3_atomic_selftest_',dir=str(base)) as d:
 p=Path(d);a=p/'a';b=p/'b';c=p/'c';a.write_bytes(b'ORIGINAL_A');b.write_bytes(b'KEEP_B')
 ret=fn(-100,os.fsencode(a),-100,os.fsencode(b),1);e=ctypes.get_errno();assert ret==-1 and e==errno.EEXIST and a.read_bytes()==b'ORIGINAL_A' and b.read_bytes()==b'KEEP_B'
 assert fn(-100,os.fsencode(a),-100,os.fsencode(c),1)==0 and c.read_bytes()==b'ORIGINAL_A' and not a.exists()
print(json.dumps(dict(status='PASS_ACTUAL_RENAMEAT2_NOREPLACE_POSITIVE_AND_NEGATIVE',available_bytes=st.f_bavail*st.f_frsize,host='archive29')))
''')
 save('ATOMIC_NOREPLACE_TEST.json',json.loads(prep))
 inputs=load(R/'CPU_ATTRIBUTION_SOURCE_FREEZE.json')
 for f in inputs['files']:
  q=R/f['path'];assert q.stat().st_size==f['bytes'] and digest(q)==f['sha256']
 source_paths={f['path'] for f in inputs['files']}
 source_paths.update(str(q.relative_to(R)) for q in (R/'scripts').glob('*.py'))
 source_paths.update(['CPU_ATTRIBUTION_SOURCE_FREEZE.json','V3_NUMERICAL_QUALIFICATION_CONTRACT.json','V3_UNIFIED_EXECUTION_PLAN.md','FP32_ROUNDING_AND_RECURRENCE_ERROR_BUDGET.md','AIME26_C1_S3_PROTOCOL_FREEZE.json','AIME26_C1_S3_GPU_ASSIGNMENT.json','FORMAL_JOB_MANIFEST.json'])
 sources=[dict(path=n,bytes=(R/n).stat().st_size,sha256=digest(R/n)) for n in sorted(source_paths)]
 save('SOURCE_FREEZE.json',dict(status='FINAL_CPU_ONLY_INPUT_SOURCE_AND_CONTRACT_FREEZE_NO_GPU_PERMIT',count=len(sources),files=sources,CPU_initial_source_freeze_sha256=digest(R/'CPU_ATTRIBUTION_SOURCE_FREEZE.json'),GPU_dispatches=0))
 exclude={'SHA256_MANIFEST.json','SHA256SUMS','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json'}
 science=[q for q in sorted(R.rglob('*')) if q.is_file() and str(q.relative_to(R)).split('/')[0]!='operations' and str(q.relative_to(R)) not in exclude]
 assert all(not q.is_symlink() for q in science)
 rows=[dict(path=str(q.relative_to(R)),bytes=q.stat().st_size,sha256=digest(q)) for q in science]
 total=sum(q['bytes'] for q in rows);assert total<20*1024**3
 save('SHA256_MANIFEST.json',dict(status='FROZEN_V3_CPU_STOP_SCIENCE',task=R.name,count=len(rows),bytes=total,files=rows,excluded_exact_top_level=sorted(exclude),supplemental_operations_not_science_seal=True))
 with (R/'SHA256SUMS').open('x') as f:f.write(''.join(x['sha256']+'  '+x['path']+'\n' for x in rows))
 # New path only; no implicit archive retries or overwrites.
 subprocess.run(['rsync','-a','--ignore-existing','--exclude=/operations/','-e','ssh -S /tmp/qwen_replication_archive29.sock -o BatchMode=yes',str(R)+'/', 'zypan@202.38.247.29:'+STAGE+'/'],check=True)
 manifest_sha=digest(R/'SHA256_MANIFEST.json');sums_sha=digest(R/'SHA256SUMS')
 verifier='''from pathlib import Path
import json,hashlib,socket,time
r=Path('''+repr(STAGE)+''')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(r/'SHA256_MANIFEST.json')=='''+repr(manifest_sha)+'''
assert sha(r/'SHA256SUMS')=='''+repr(sums_sha)+'''
m=json.loads((r/'SHA256_MANIFEST.json').read_text());verified=[]
for x in m['files']:
 p=r/x['path'];assert not p.is_symlink() and p.resolve().is_relative_to(r.resolve())
 assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];verified.append(x)
allowed={x['path'] for x in m['files']}|{'SHA256_MANIFEST.json','SHA256SUMS'}
assert {str(p.relative_to(r)) for p in r.rglob('*') if p.is_file()}==allowed
print(json.dumps(dict(status='PASS_INDEPENDENT_ARCHIVE29_EVERY_FILE_SHA_AND_BYTES',host=socket.gethostname(),monotonic=time.monotonic(),manifest_sha256=sha(r/'SHA256_MANIFEST.json'),count=len(verified),bytes=sum(x['bytes'] for x in verified),files=verified),sort_keys=True))
'''
 raw_verification=on29(verifier)
 with (R/'FINAL_ARCHIVE_VERIFICATION.json').open('x') as f:f.write(raw_verification);f.flush();os.fsync(f.fileno())
 save('FINAL_SEAL_RECEIPT.json',dict(status='PASS_V3_NUMERICAL_CONTRACT_UNRESOLVED_AND_STOP_SEALED',terminal='V3_NUMERICAL_CONTRACT_UNRESOLVED_AND_STOP',formal_completed=0,formal_planned=60,GPU_dispatches=0,manifest_sha256=manifest_sha,SHA256SUMS_sha256=sums_sha,archive_verification_sha256=digest(R/'FINAL_ARCHIVE_VERIFICATION.json'),archive_root=ARCH,atomic='renameat2_RENAME_NOREPLACE',source_freeze_sha256=digest(R/'SOURCE_FREEZE.json')))
 for n in ['FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json']:
  subprocess.run(['rsync','-a','--ignore-existing','-e','ssh -S /tmp/qwen_replication_archive29.sock -o BatchMode=yes',str(R/n),'zypan@202.38.247.29:'+STAGE+'/'+n],check=True)
 controls={n:digest(R/n) for n in ['SHA256_MANIFEST.json','SHA256SUMS','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json']}
 final=on29('''from pathlib import Path
import json,hashlib,ctypes,os
r=Path('''+repr(STAGE)+''');final=Path('''+repr(ARCH)+''');controls='''+repr(controls)+'''
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for n,h in controls.items():assert sha(r/n)==h
m=json.loads((r/'SHA256_MANIFEST.json').read_text())
for x in m['files']:assert (r/x['path']).stat().st_size==x['bytes'] and sha(r/x['path'])==x['sha256']
assert {str(p.relative_to(r)) for p in r.rglob('*') if p.is_file()}=={x['path'] for x in m['files']}|set(controls)
assert not final.exists()
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
ret=fn(-100,os.fsencode(r),-100,os.fsencode(final),1)
if ret:raise OSError(ctypes.get_errno(),'renameat2_NOREPLACE')
fd=os.open(final.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='PASS_ATOMIC_NOREPLACE_PUBLISHED_NO_GPU',archive_root=str(final),count=len(m['files']),bytes=m['bytes'],controls=controls)))
''')
 save('operations/ATOMIC_PUBLICATION_RECEIPT.json',json.loads(final))
 save('operations/CPU_STOP_PUBLISHER_RESULT.json',dict(status='PASS',identity=actual_identity(),monotonic=time.monotonic(),archive_root=ARCH,files=len(rows),bytes=total,source_count=len(sources),GPU_dispatches=0))
 print(final,flush=True)
except BaseException as error:
 if not (R/'operations/CPU_STOP_PUBLISHER_FAILURE.json').exists():save('operations/CPU_STOP_PUBLISHER_FAILURE.json',dict(status='CPU_DELIVERY_FAILED_PRESERVE_AND_INSPECT_NO_AUTORETRY',error=repr(error),traceback=traceback.format_exc(),GPU_dispatches=0))
 raise
