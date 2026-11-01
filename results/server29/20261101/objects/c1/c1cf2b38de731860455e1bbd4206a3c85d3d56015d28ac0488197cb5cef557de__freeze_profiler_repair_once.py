"""Once-only CPU final admission and primary source freeze; never launches a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def enc(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b'\n'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


PROGRAM = r'''
import pathlib,sys,os,json,hashlib,time
r=pathlib.Path(sys.argv[1]).resolve();f=json.load(sys.stdin)['freeze']
def enc(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2).encode()+b'\n'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
 with p.open('xb') as s:s.write(enc(v));s.flush();os.fsync(s.fileno())
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not any(os.path.lexists(r/n) for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','CONTROLLER_STARTED.json','OFFLINE_STARTED.json'])
cfg=read(r/'ROOT.json');auth=read(r/'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json')
assert cfg['remote_root']==str(r) and cfg['remote_execution_slug']==r.name
assert auth['human_request']=='检查为什么异常，修复后，继续完成测试' and auth['GPU0_P1_must_continue_unmodified'] and not auth['GPU0_authorized']
assert f['human_profiler_repair_authorization_sha256']==sha(r/'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json')
assert f['count']==len(f['files'])==len({x['path'] for x in f['files']})
for row in f['files']:
 p=r/row['path'];assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(r)
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
 if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
scope=read(r/'PROFILER_REPAIR_SOURCE_SCOPE.json');old=pathlib.Path(scope['remote_prior_root'])
assert sha(old/'SOURCE_FREEZE.json')==auth['prior_source_sha256']=='5be0a7471840d29579560685f98952f2a74ac9bb9b1b6ad85c6e6e0f78563092'
assert sha(old/'FINAL_SEAL_RECEIPT.json')==auth['prior_seal_sha256']
changes={'HANDOFF.md','ROOT.json','operations/download_verified_delivery.py','operations/test_readonly_delivery_cpu.py','scripts/resident_finalize_offline.py','src/qualification_guards.py'}
for row in scope['prior_sources']:
 p=old/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Prior science modified:'+row['path']
 if row['path'] not in changes:
  p=r/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Unexpected non-lifecycle change:'+row['path']
assert sha(r/'src/resident_rotation.py')=='4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95'
assert sha(r/'scripts/qualify_gpu.py')=='a41b84fbecfe262b37211ba27804329955f23577ef46babade06a115ec88fff2'
sys.path[:0]=[str(r/'src'),str(r/'scripts'),str(r/'operations')]
import resident_execution_control as C,qualification_guards as G,resident_controller as controller,resident_archive as archive,engineering_gpu_evidence as E,profiler_engineering_evidence as P
suite=read(r/'analysis/PROFILER_REPAIR_CPU_V2_TWO_FRESH_SUITE.json');assert len(suite['runs'])==12
for name,count in [('DETACH',21),('GUARD',74),('NATIVE',47),('CONTROL',79),('EXIT',30)]:
 rows=[x for x in suite['runs'] if x['case']==name];assert len(rows)==2 and len({x['pid'] for x in rows})==2
 for row in rows:
  result=row['result'];assert row['actual_wait_returncode']==0 and result.get('check_count',result.get('count'))==count
  for p,digest in result.get('source_pins',{}).items():assert sha(r/p)==digest,p
fin=read(r/'analysis/PROFILER_FINAL_FINISHER_TWO_FRESH.json');assert len(fin['runs'])==2
for row in fin['runs']:
 v=row['result'];assert row['actual_wait_returncode']==0 and v.get('count',v.get('check_count'))==45
 assert v['finisher_sha256']==sha(r/'scripts/resident_finalize_offline.py')
for i in range(2):
 v=read(r/('analysis/PROFILER_READONLY_MAC_R'+str(i)+'.json'));assert v['check_count']==88 and not v['Torch_imported']
 for p,digest in v['source_pins'].items():assert sha(r/p)==digest,p
inherited=E.validate(r);assert inherited['total_GPU_work_dispatches']==12
current=P.validate(r,C);assert current==read(r/'analysis/PROFILER_ENGINEERING_EVIDENCE.json') and current['engineering_dispatch_count']==15
assert current['guard_sha256']==sha(r/'src/qualification_guards.py') and current['lifecycle_sha256']==sha(r/'src/profiler_detach_lifecycle.py')
prior=read(old/'SESSION_RESULT.json');assert prior['qualification_completed']==48 and prior['formal_completed']==0 and prior['own_release_all_PASS']
assert not C.same_process(prior['controller_identity'])
external=C.verify_model_and_external(r,True);assert len(external)==61 and len(G.load_native_closure(r)['files'])==293
prior_p1=C.validate_prior_p1_stop(r)
units=read(r/'QUALIFICATION_JOB_MANIFEST.json')['units'];jobs=read(r/'FORMAL_JOB_MANIFEST.json')['jobs']
assert len(units)==len({x['id'] for x in units})==58 and len(jobs)==len({x['id'] for x in jobs})==60
assert {x['condition'] for x in jobs}=={'C1','S3'}
assert all(x['seed']==1 and x['max_new_tokens']==81920 and x['gpu_uuid']==C.UUIDS[x['gpu']] for x in jobs)
for row in units:
 assert not os.path.lexists(r/'qualification'/row['id']) and not os.path.lexists(r/'dispatch/qualification'/(row['id']+'.json'))
assert all(not os.path.lexists(r/'formal'/x['id']) for x in jobs)
prepared=controller.prepare_execution_directories(r);obs=C.query_gpu()
assert not any(x['uuid'] in C.UUIDS.values() for x in obs['apps'])
for gpu,uuid in C.UUIDS.items():
 d=next(x for x in obs['devices'] if x['physical_index']==gpu);assert d['uuid']==uuid and '3090' in d['name'] and d['memory_free_MiB']>=23000
ram=os.statvfs('/dev/shm');data=os.statvfs('/data')
assert ram.f_bavail*ram.f_frsize>=100*1024**3 and data.f_bavail*data.f_frsize>=8*1024**3
cat,exclusions=archive.catalog(r,{})
write(r/'operations/FINAL_PREFREEZE_REAL_ARCHIVE_CATALOG.json',dict(status='PASS_ACTUAL_REAL_ROOT_ARCHIVE_CATALOG_NO_SYMLINKS',count=len(cat),bytes=sum(x['bytes'] for x in cat),files=cat,excluded_exact_paths=exclusions))
assert 'torch' not in sys.modules
source_sha=hashlib.sha256(enc(f)).hexdigest()
admission=dict(status='PASS_HUMAN_AUTHORIZED_PROFILER_REPAIR_FINAL_CPU_ADMISSION_NOT_GPU_PROOF',source_freeze_sha256=source_sha,source_count=f['count'],human_profiler_repair_authorization_sha256=sha(r/'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json'),prior324_science_fresh_SHA_immutable=True,only_six_exact_old_source_changes=sorted(changes),controller_sha256=sha(r/'scripts/resident_controller.py'),guard_sha256=sha(r/'src/qualification_guards.py'),lifecycle_sha256=sha(r/'src/profiler_detach_lifecycle.py'),CPU_detach_checks_each=21,CPU_checks_each=79,actual_guard_path_CPU_checks_each=74,actual_native_closure_guard_CPU_checks_each=47,CPU_exit_checks_each=30,CPU_finisher_checks_each=45,CPU_Mac_readonly_checks_each=88,two_fresh_target_CPU_runs_each=True,actual_external_dependency_fresh_SHA_count=61,actual_fresh_native_closure_count=293,model_stat_count=17,actual_required_parent_directory_preparation=prepared,prior_P1_stop=prior_p1,NVML=obs,free_RAM_bytes=ram.f_bavail*ram.f_frsize,free_data_bytes=data.f_bavail*data.f_frsize,inherited_engineering_GPU_dispatches=12,current_engineering_GPU_dispatches=15,current_actual_nonAIME_engineering_forwards=816,profiler_evidence_sha256=sha(r/'analysis/PROFILER_ENGINEERING_EVIDENCE.json'),engineering_probe_is_scientific_qualification=False,scientific_qualification_GPU_dispatches=0,formal_generations=0,prior48_qualifications_reused_as_new_PASS=False,Torch_imported=False,archive_catalog_sha256=sha(r/'operations/FINAL_PREFREEZE_REAL_ARCHIVE_CATALOG.json'),monotonic=time.monotonic())
write(r/'operations/FINAL_CPU_ADMISSION.json',admission);write(r/'SOURCE_FREEZE.json',f)
permit=dict(task_id=C.TASK,execution_attempt=r.name,ALLOW_GPU='QUALIFICATION_ONLY',source_freeze_sha256=source_sha,devices=list(C.UUIDS.values()),human_profiler_repair_authorization_sha256=sha(r/'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json'),human_authorization_sha256=sha(r/'HUMAN_CONTINUATION_AUTHORIZATION.json'),repair_authorization_sha256=sha(r/'HUMAN_REPAIR_AUTHORIZATION.json'),native_exit_repair_authorization_sha256=sha(r/'HUMAN_NATIVE_EXIT_REPAIR_AUTHORIZATION.json'),prior_p1_stop_admission_sha256=sha(r/'PRIOR_P1_USER_STOP_ADMISSION.json'),runtime_native_closure_sha256=sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json'),qualification_plan_sha256=sha(r/'QUALIFICATION_JOB_MANIFEST.json'),execution_plan_sha256=sha(r/'GPU_EXECUTION_PLAN.json'),final_CPU_admission_sha256=sha(r/'operations/FINAL_CPU_ADMISSION.json'),profiler_engineering_evidence_sha256=admission['profiler_evidence_sha256'],engineering_gpu_evidence_binding_sha256=sha(r/'operations/ENGINEERING_GPU_EVIDENCE_BINDING.json'),engineering_probe_completion_sha256=sha(inherited['active_completion']),formal_permission=False,conditional_formal_separate_permit_required=True,GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,no_progress_watchdog_seconds=1800,own_cleanup_observation_seconds=120,no_GPU_retry_after_STARTED=True)
write(r/'GPU_PERMIT.json',permit);assert C.verify_sources(r)==source_sha
print(json.dumps(dict(status=admission['status'],source_count=f['count'],source_sha256=source_sha,permit_sha256=sha(r/'GPU_PERMIT.json'),admission_sha256=sha(r/'operations/FINAL_CPU_ADMISSION.json'))))
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--execute', action='store_true'); args = ap.parse_args()
    assert args.execute and not (ROOT / 'SOURCE_FREEZE.json').exists()
    cfg = json.loads((ROOT / 'ROOT.json').read_text())
    names = {}
    for p in sorted(ROOT.rglob('*')):
        assert not p.is_symlink()
        if p.is_file():
            names[p.relative_to(ROOT).as_posix()] = dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
    # Every raw matrix preserved remotely is a source. Never transfer it to Mac.
    inventory_code = "import pathlib,json,hashlib; r=pathlib.Path(" + repr(cfg['remote_root']) + "); print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in r.rglob('*') if p.is_file() and p.suffix in ('.bin','.f32','.npz','.npy','.pt','.pth','.safetensors')]))"
    ssh = ['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55']
    result = subprocess.run(ssh + [shlex.join([cfg['remote_python'],'-B','-c',inventory_code])], capture_output=True,text=True,check=True)
    remote_raw = json.loads(result.stdout)
    for row in remote_raw:
        if row['path'] in names: assert names[row['path']] == row
        names[row['path']] = row
    freeze = dict(task_id=cfg['task_id'],execution_attempt=cfg['remote_execution_slug'],status='ONCE_FROZEN_PROFILER_REPAIR_QUALIFICATION_ONLY',human_profiler_repair_authorization_sha256=sha(ROOT/'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json'),prior_source_freeze_sha256='5be0a7471840d29579560685f98952f2a74ac9bb9b1b6ad85c6e6e0f78563092',count=len(names),files=sorted(names.values(),key=lambda x:x['path']),scientific_qualification_GPU_dispatches_at_freeze=0,formal_generations_at_freeze=0,new_engineering_GPU_dispatches_at_freeze=15,inherited_prior_engineering_dispatches=12,prior48_qualifications_count_as_new_PASS=False,remote_only_raw_count=len(remote_raw),generated_remote_traces_not_local_source_members=True)
    result = subprocess.run(ssh + [shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',cfg['remote_python'],'-B','-c',PROGRAM,cfg['remote_root']])],input=json.dumps(dict(freeze=freeze)),capture_output=True,text=True,timeout=300)
    with (ROOT/'operations/FREEZE_PROFILER_REPAIR_TRANSPORT.json').open('x') as stream:
        json.dump(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),stream,indent=2)
    assert result.returncode==0,result.stderr
    print(result.stdout)
    for name in ['SOURCE_FREEZE.json','GPU_PERMIT.json','operations/FINAL_CPU_ADMISSION.json','operations/FINAL_PREFREEZE_REAL_ARCHIVE_CATALOG.json']:
        data = subprocess.run(ssh+[shlex.join(['cat',cfg['remote_root']+'/'+name])],capture_output=True,check=True).stdout
        with (ROOT/name).open('xb') as stream:stream.write(data)
    assert sha(ROOT/'SOURCE_FREEZE.json') == hashlib.sha256(enc(freeze)).hexdigest()


if __name__ == '__main__':
    main()
