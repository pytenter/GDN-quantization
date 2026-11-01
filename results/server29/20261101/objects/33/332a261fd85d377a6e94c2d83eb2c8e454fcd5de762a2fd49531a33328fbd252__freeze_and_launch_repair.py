"""One CPU preflight/freeze then one human-authorized new execution launch."""
from pathlib import Path
import sys,json,os,hashlib,time,subprocess,shlex
R=Path(sys.argv[1]);sys.path.insert(0,str(R/'src'));import execution_control as C
old=Path(C.read(R/'ROOT.json')['prior_failed_remote_root'])
def dump(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def replace(n,x):(R/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
assert not (R/'SOURCE_FREEZE.json').exists() and not (R/'GPU_PERMIT.json').exists()
assert not (R/'GPU_RUN_STARTED.json').exists()
assert not list((R/'execution').iterdir()) and not list((R/'formal').iterdir())
release=C.read(R/'operations/PRIOR_RAM_RELEASE_AFTER_DURABLE_ARCHIVE.json')
assert release['status']=='PASS_ARCHIVED_ALL_BYTES_THEN_EXACT_RAM_ONLY_RELEASE'
assert release['files']==1313 and release['bytes']==113347957760
archive=C.read(R/'operations/PRIOR_RAM_ARCHIVE_VERIFICATION.json')
assert archive['status']=='PASS_ALL1313_ORIGINAL_RAM_BYTES_DURABLY_PRESERVED_ON29'
assert not Path('/dev/shm/qwen_gdn_oscar_sym32_unified_recovery_aime26_v2').exists()
assert not C.reference_root().exists()
st=os.statvfs('/dev/shm');assert st.f_bavail*st.f_frsize>=100*1024**3
print('RAM preservation/release and100GiB admission PASS.',flush=True)
prior_manifest=C.read(old/'SOURCE_FREEZE.json')
assert C.sha(old/'SOURCE_FREEZE.json')=='84a60e2fa0bde4443bca1f3401355df8259a8301d97d42f9802790aa9010044f'
for x in prior_manifest['files']:
 p=old/x['path'];assert p.stat().st_size==x['bytes'] and C.sha(p)==x['sha256']
assert C.read(old/'FINAL_TASK_STATUS.json')['status']=='PARTIAL_RESULTS_PRESERVED_AND_STOP'
assert C.read(old/'FINAL_TASK_STATUS.json')['formal_completed']==0
prior_seal=C.read(old/'FINAL_SEAL_RECEIPT.json')
assert C.sha(old/'SHA256_MANIFEST.json')==prior_seal['manifest_sha256']
assert C.sha(old/'FINAL_ARCHIVE_VERIFICATION.json')==prior_seal['archive_verification_sha256']
for base in ['REPAIRED_CONTROL_TESTS','WORKER_FAILURE_SESSION_CPU']:
 a=C.read(R/'qualification'/(base+'_0.json'));b=C.read(R/'qualification'/(base+'_1.json'));assert a==b and a['status'].startswith('PASS')
invariants=C.read(R/'REPAIR_CHANGES_AND_INVARIANTS.json')
for x in invariants['unchanged_scientific_files']:
 assert C.sha(R/x['path'])==x['sha256']==C.sha(old/x['path'])
for g in C.SELECTED_GPUS:C.device(g)
obs=C.query_gpu();C.model_guard(R);assert C.verify_external(R)==61
print('UUID/17modelstat/61actualexternal freshSHA PASS.',flush=True)
assert C.validate_jobs(R,C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'])
for x in C.read(R/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
 s=Path(x['path']).stat();assert (s.st_size,s.st_ino,s.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns'])
replace('THREE_GPU_UUID_AUDIT.json',dict(status='PASS_CURRENT_THREE_IDLE_UUIDS_BEFORE_AUTHORIZED_RESTART',selected=list(C.SELECTED_GPUS),observation=obs,old_GPU0_foreign_job_untouched=True,monotonic=time.monotonic()))
cfg=C.read(R/'ROOT.json');cfg.update(status='CPU_REPAIR_ADMITTED_NEW_ONCE_EXECUTION',GPU_dispatches=0,new_scientific_branch=False,prior_failure_preserved=True);replace('ROOT.json',cfg)
p1=C.read(R/'P1_EXECUTABLE_IDENTITY.json')
for x in p1['files']:
 p=R/x['path'];x.update(bytes=p.stat().st_size,sha256=C.sha(p))
p1['status']='FROZEN_INDEPENDENT_P1_EXECUTABLE_BEFORE_AUTHORIZED_REPAIR_GPU';replace('P1_EXECUTABLE_IDENTITY.json',p1)
admission=dict(status='PASS_HUMAN_REPAIR_PREFLIGHT_BEFORE_NEW_FREEZE',prior_failed_source_sha256=C.sha(old/'SOURCE_FREEZE.json'),prior_failed_manifest_sha256=prior_seal['manifest_sha256'],prior_failure_AIME=0,prior_RAM_engineering_archive=release,CPU_control_checks_each=10,CPU_full_controller_failed_child_SESSION_passes=2,external_fresh_SHA_count=61,model_stat_count=17,UUIDs_idle=True,observation=obs,old_GPU0_untouched=True,math_codec_sampler_roster_cap_unchanged=True,monotonic=time.monotonic())
dump('analysis/REPAIR_PRE_GPU_ADMISSION.json',admission)
extra=['RESTART_AUTHORIZATION.json','REPAIR_CHANGES_AND_INVARIANTS.json','scripts/test_repaired_control_cpu.py','scripts/test_worker_failure_session_cpu.py','qualification/REPAIRED_CONTROL_TESTS_0.json','qualification/REPAIRED_CONTROL_TESTS_1.json','qualification/WORKER_FAILURE_SESSION_CPU_0.json','qualification/WORKER_FAILURE_SESSION_CPU_1.json','operations/preserve_prior_ram_before_restart.py','operations/freeze_and_launch_repair.py','operations/PRIOR_RAM_ARCHIVE_VERIFICATION.json','operations/PRIOR_RAM_RELEASE_AFTER_DURABLE_ARCHIVE.json','operations/CODE_REPAIR_DIFF.diff','analysis/REPAIR_PRE_GPU_ADMISSION.json']
names=sorted({x['path'] for x in prior_manifest['files']}|set(extra));rows=[]
for n in names:
 p=R/n;assert p.is_file() and not p.is_symlink()
 if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
 rows.append(dict(path=n,bytes=p.stat().st_size,sha256=C.sha(p)))
manifest=dict(task_id=C.TASK,status='FROZEN_ONCE_BEFORE_HUMAN_AUTHORIZED_REPAIR_RESTART',files=rows,count=len(rows),P1_independent_identity_sha256=C.sha(R/'P1_EXECUTABLE_IDENTITY.json'),canonical_route=prior_manifest['canonical_route'],V1_unchanged=True,prior_failed_V2_unchanged=True,repair_authorization_sha256=C.sha(R/'RESTART_AUTHORIZATION.json'))
dump('SOURCE_FREEZE.json',manifest)
h=C.read(R/'HUMAN_GPU_AUTHORIZATION.json')
permit=dict(task_id=C.TASK,ALLOW_GPU='YES',devices=[C.UUIDS[g] for g in C.SELECTED_GPUS],source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),plan_sha256=C.sha(R/'GPU_EXECUTION_PLAN.json'),human_grant_sha256=C.sha(R/'HUMAN_GPU_AUTHORIZATION.json'),GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,watchdog_seconds=1800,cleanup_seconds=120)
dump('GPU_PERMIT.json',permit);C.authorize(R,R/'GPU_PERMIT.json');C.storage_guard(R)
final=dict(status='PASS_REPAIRED_SOURCE_AND_PERMIT_FREEZE_BEFORE_ONCE_LAUNCH',source_count=len(rows),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(R/'GPU_PERMIT.json'),P1_executable_sha256=C.sha(R/'P1_EXECUTABLE_IDENTITY.json'),human_authorization_sha256=C.sha(R/'HUMAN_GPU_AUTHORIZATION.json'),plan_sha256=C.sha(R/'GPU_EXECUTION_PLAN.json'),prior_source_unchanged=True,old_GPU0_untouched=True)
dump('operations/FINAL_CPU_ADMISSION.json',final)
tmux='qwen-oscar-sym32-unified-v2-rerun'
p=subprocess.run(['tmux','has-session','-t',tmux],capture_output=True);assert p.returncode!=0,'Already live tmux; never duplicate'
command=shlex.join(['bash',str(R/'scripts/run_offline.sh')]);p=subprocess.run(['tmux','new-session','-d','-s',tmux,command],capture_output=True,text=True);assert p.returncode==0,p.stderr
dump('operations/LAUNCH_RECEIPT.json',dict(tmux=tmux,command=command,once_only=True,source_freeze_sha256=final['source_freeze_sha256'],permit_sha256=final['permit_sha256'],new_execution_not_prior_process_restart=True,monotonic=time.monotonic()))
print(json.dumps(final),flush=True);print('ONCE GPU pipeline started.',flush=True)
