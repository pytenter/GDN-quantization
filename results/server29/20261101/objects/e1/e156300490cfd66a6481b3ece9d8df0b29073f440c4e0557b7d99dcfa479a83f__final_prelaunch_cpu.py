"""Final CPU/SSH-only checks, source freeze, permit. Never dispatches CUDA work."""
import os,sys,json,subprocess,shlex,ast,hashlib,time
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));import execution_control as C
C.require(not (R/'GPU_RUN_STARTED.json').exists() and not (R/'SOURCE_FREEZE.json').exists(),'Already frozen/started; inspect, no overwrite')
for folder in ['src','scripts']:
 for p in (R/folder).glob('*.py'):ast.parse(p.read_text())
log=R/'qualification/FINAL_EXECUTION_CPU_TESTS.log'
with log.open('xb') as f:p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/test_execution_cpu.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,timeout=240)
C.require(p.returncode==0,'Execution CPU integration failed; inspect final log')
v=json.loads(next(l[12:] for l in log.read_text().splitlines() if l.startswith('RESULT_JSON=')));C.require(v['status']=='PASS_NEW_EXECUTION_CPU_INTEGRATION','CPU integration not PASS');C.atomic_new(R/'qualification/FINAL_EXECUTION_CPU_TESTS.json',v)
C.admit_inherited(R);C.validate_jobs(R,C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']);C.model_guard(R);C.storage_guard(R)
s=os.statvfs(R);C.require(s.f_bavail*s.f_frsize>=12*1024**3,'Need2GiBsmall+2GiBfailure+8GiBheadroom beforeGPU')
external=C.verify_external(R);C.atomic_new(R/'analysis/FINAL_PRELAUNCH_EXTERNAL_61_SHA.json',dict(status='PASS_FRESH_EXTERNAL_SHA',count=external,CPU_only=True))
code='''import sys,json,os,ctypes,tempfile,hashlib
from pathlib import Path
base=Path(sys.argv[1]);base.parent.mkdir(parents=True,exist_ok=True);assert not base.exists(),'Final sciencearchive mustnotalreadyexist';s=os.statvfs(base.parent);assert s.f_bavail*s.f_frsize>=120*1024**3,'20GiBquota+100GiBheadroom'
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
with tempfile.TemporaryDirectory(prefix='_CPU_ATOMIC_NO_GPU_',dir=base.parent) as d:
 p=Path(d);a=p/'source';a.mkdir();b=b'QWEN_AIME26_CPU_ATOMIC_ARCHIVE_TEST';(a/'blob').write_bytes(b);blocked=p/'existing';blocked.mkdir();rc=fn(-100,os.fsencode(a),-100,os.fsencode(blocked),1);assert rc!=0 and a.exists() and blocked.exists(),'No-clobber negativefixturefailed';final=p/'published';assert fn(-100,os.fsencode(a),-100,os.fsencode(final),1)==0 and (final/'blob').read_bytes()==b
print(json.dumps(dict(status='PASS_SAME_FILESYSTEM_ATOMIC_NO_CLOBBER_CPU_ONLY',archive_target=str(base),available_bytes=s.f_bavail*s.f_frsize,probe_bytes=len(b),probe_sha256=hashlib.sha256(b).hexdigest(),GPU_calls=0,temporary_fixture_removed=True)))
'''
archive=C.read(R/'ROOT.json')['archive_root'];p=subprocess.run(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','zypan@202.38.247.29',shlex.join(['/usr/bin/python3','-u','-B','-c',code,archive])],capture_output=True,timeout=90);C.require(p.returncode==0,'Archive CPU admission failed:'+p.stderr.decode());C.atomic_new(R/'analysis/ARCHIVE_ATOMIC_CPU_PREFLIGHT.json',json.loads(p.stdout))
# Only execution sources and immutable binding data, never live statuses or old snapshots.
rows=[]
fixed={'USER_REQUEST.txt','HUMAN_GPU_AUTHORIZATION.json','GPU_EXECUTION_PLAN.json','FORMAL_JOB_MANIFEST.json','AIME26_SEED1_PROTOCOL.json','AIME26_DATASET_FREEZE.json','QUESTION_GPU_ASSIGNMENT.json','FOUR_CODEC_NUMERICAL_CONTRACT.md','CPU_CODEC_QUALIFICATION.json'}
for p in sorted(R.rglob('*')):
 rel=p.relative_to(R)
 if not p.is_file() or p.is_symlink() or 'cpu_preparation_delivery' in rel.parts:continue
 include=rel.parts[0] in ['src','scripts','protocol','inherited'] or str(rel) in fixed or str(rel) in ['analysis/MODEL_TOKENIZER_FRESH_SHA.json','qualification/FINAL_EXECUTION_CPU_TESTS.json','analysis/FINAL_PRELAUNCH_EXTERNAL_61_SHA.json','analysis/ARCHIVE_ATOMIC_CPU_PREFLIGHT.json','evidence/group32/delivery/SHA256_MANIFEST.json','evidence/group32/delivery/TASK_RESULT.json']
 if include:rows.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=C.sha(p)))
C.atomic_new(R/'SOURCE_FREEZE.json',dict(task_id=C.TASK,status='GPU_QUALIFICATION_AND_FORMAL_FULL_SOURCE_FROZEN_BEFORE_FIRST_DISPATCH',files=rows,count=len(rows),parent_500_source_sha256='be6786c37ab508f6749fce33cfd00606a2c2ac5bbbd2fb429a673143b40dfddc',no_runtime_math_changes=True))
C.atomic_new(R/'GPU_PERMIT.json',dict(task_id=C.TASK,ALLOW_GPU='YES',devices=list(C.UUIDS),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),plan_sha256=C.sha(R/'GPU_EXECUTION_PLAN.json'),human_grant_sha256=C.sha(R/'HUMAN_GPU_AUTHORIZATION.json'),GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,watchdog_seconds=1800,cleanup_seconds=120))
C.authorize(R,R/'GPU_PERMIT.json')
p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/gpu_controller.py'),'--root',str(R)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True);C.require(p.returncode==0,'Final check-only controller failed');print(p.stdout.decode(),flush=True)
print(json.dumps(dict(status='PASS_FROZEN_GPU_PRELAUNCH_CPU_ONLY',source_count=len(rows),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(R/'GPU_PERMIT.json'),GPU_dispatches=0)),flush=True)
