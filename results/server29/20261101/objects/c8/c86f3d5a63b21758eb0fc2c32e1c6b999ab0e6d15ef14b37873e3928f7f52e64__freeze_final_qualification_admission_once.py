#!/usr/bin/env python3
"""Reviewable once-only scientific admission after the separate loader probe.

Default is a local plan. --execute performs CPU/source/resource checks and
creates the scientific freeze/qualification permit once; it launches no GPU.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pre_gpu_deploy',ROOT/'operations/deploy_final_pre_gpu_sources.py')
deploy=importlib.util.module_from_spec(spec);spec.loader.exec_module(deploy)


def encoded(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'


def source_payload(root=ROOT):
    root=Path(root);assert not(root/'SOURCE_FREEZE.json').exists()
    rows=[]
    for p in deploy.paths(root):
        data=p.read_bytes();rows.append(dict(path=p.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    remote=json.loads((root/'operations/REMOTE_ONLY25_ASSET_COPY.json').read_text())
    assert remote['status']=='PASS_REMOTE_ONLY_ORIGINAL25_ASSET_INDEPENDENT_COPY'and len(remote['files'])==25
    rows+=remote['files'];assert len(rows)==len({x['path']for x in rows})
    counts=deploy.probe_counts(root)
    assert counts['engineering_native_loader_GPU_probes']==4 and counts['total_GPU_work_dispatches']==12,'All engineering history and two repaired full-loader exits required'
    freeze=dict(task_id='QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1',execution_attempt=root.name,
        status='ONCE_PRE_SCIENTIFIC_GPU_SOURCE_FROZEN_QUALIFICATION_ONLY',count=len(rows),files=sorted(rows,key=lambda x:x['path']),
        original_resident_core_unchanged=True,original68_source_freeze_preserved=True,prior153_source_freeze_sha256=deploy.PRIOR153,
        GPU_PERMIT_excluded_to_avoid_circular_binding=True,
        scientific_qualification_GPU_dispatches_at_freeze=0,engineering_native_loader_GPU_probes_at_freeze=4,engineering_diagnostic_GPU_dispatches_at_freeze=8,
        GPU_dispatches_at_freeze=12,total_GPU_work_dispatches_at_freeze=12,
        formal_generations_at_freeze=0,CPU_PASS_is_GPU_PASS=False,loader_probe_is_scientific_qualification=False)
    return dict(freeze=freeze,source_sha256=hashlib.sha256(encoded(freeze)).hexdigest())


PROGRAM=r'''
import base64,hashlib,json,os,pathlib,stat,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);assert not r.is_symlink();r=r.resolve();payload=json.load(sys.stdin)
def encoded(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write_new(p,b):
 with pathlib.Path(p).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
def safe(name):
 p=pathlib.Path(name);assert isinstance(name,str)and name and not p.is_absolute()and str(p)==name and all(x not in('','..','.')for x in name.split('/'));return p
def bound(row):
 p=r/safe(row['path']);assert p.resolve().is_relative_to(r)and p.is_file()and not p.is_symlink()
 assert sha(p)==row['sha256']and('bytes'not in row or p.stat().st_size==row['bytes']),row['path'];return p
def no_symlinks():
 for directory,dirs,files in os.walk(r,followlinks=False):
  for name in dirs+files:
   p=pathlib.Path(directory)/name;s=p.lstat();assert not stat.S_ISLNK(s.st_mode)and(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)),str(p)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not any(os.path.lexists(r/x)for x in('SOURCE_FREEZE.json','GPU_PERMIT.json','CONTROLLER_STARTED.json','OFFLINE_STARTED.json'))
no_symlinks();repair=read(r/'HUMAN_REPAIR_AUTHORIZATION.json');freeze=payload['freeze']
assert repair['execution_attempt']==freeze['execution_attempt']and repair['qualification_authorized']and repair['formal_conditionally_authorized']
prior_sha='f0c27e48442e924b1a49fa254c4f262f49be85da00a3ec52c3073d7e73e77d5c'
assert repair['prior_source_freeze_sha256']==freeze['prior153_source_freeze_sha256']==prior_sha
assert repair.get('prior153_original_science_seal_exists')is False
raw=encoded(freeze);source_sha=hashlib.sha256(raw).hexdigest();assert source_sha==payload['source_sha256']and freeze['count']==len(freeze['files'])
assert freeze['scientific_qualification_GPU_dispatches_at_freeze']==0 and freeze['engineering_native_loader_GPU_probes_at_freeze']==4
assert freeze['GPU_dispatches_at_freeze']==freeze['total_GPU_work_dispatches_at_freeze']==12 and freeze['loader_probe_is_scientific_qualification']is False
by={x['path']:x for x in freeze['files']};assert len(by)==len(freeze['files'])
for row in freeze['files']:
 p=bound(row)
 if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
assert sha(r/'src/resident_rotation.py')=='4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95'
# Read the preserved prior153 contract; no old file is written or rejudged.
prior=r.parent/'qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_directory_repair_rerun_20261022'
assert sha(prior/'SOURCE_FREEZE.json')==prior_sha
old=read(prior/'SOURCE_FREEZE.json');assert len(old['files'])==153
old_by={x['path']:x for x in old['files']}
math_paths=[n for n in old_by if n.startswith(('vendor/','calibration/','protocol/'))or n in('src/resident_rotation.py','RESIDENT_ROTATION_MATRICES_SHA.json','P1_QUESTION_GPU_ASSIGNMENT.json')]
for n in math_paths:
 assert n in by and by[n]['sha256']==old_by[n]['sha256']and by[n]['bytes']==old_by[n]['bytes'],'Scientific math/input identity changed: '+n
raw25=read(r/'operations/REMOTE_ONLY25_ASSET_COPY.json');assert len(raw25['files'])==25
for x in raw25['files']:
 assert x['path']in old_by and all(x[k]==old_by[x['path']][k]for k in('bytes','sha256'))
 assert by[x['path']]==x
sys.path.insert(0,str(r/'src'));import resident_execution_control as C
import qualification_guards as G
assert 'torch'not in sys.modules
external=C.verify_model_and_external(r,True);assert len(external)==61
closure=G.load_native_closure(r);native_count=len(closure['files']);assert native_count>0
assert all(x['path']in by for x in closure['cpu_probe_receipts'])
prior_P1=C.validate_prior_p1_stop(r);assert len(read(r/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files'])==17
for row in read(r/'HISTORY_INPUT_FREEZE.json')['external_sources']:
 p=pathlib.Path(row['path']);assert p.stat().st_size==row['bytes']and sha(p)==row['sha256']
control=read(r/'analysis/CONTROL_TARGET_FINAL_TWO_FRESH_RESULT.json')
assert control['status']=='PASS_FINAL_TWO_ACTUAL_TARGET_PY310_FRESH_CPU_CONTROL_PROCESSES'
assert len(control['processes'])==len({x['pid']for x in control['processes']})==2
for x in control['processes']:
 v=x['result'];assert x['actual_wait_returncode']==0 and v['check_count']==79
 assert v['real_model_loaded']is False and v['GPU_units_dispatched']==0
 for n,h in v['source_pins'].items():assert sha(r/n)==h,n
guard=read(r/'analysis/ACTUAL_GUARD_NATIVE_EXIT_TARGET_FINAL_TWO_FRESH_RESULT.json')
assert guard['status']=='PASS_TWO_FRESH_REAL_GUARD_PATH_DIRECTORY_REPAIR_CPU_ONLY'
assert len(guard['processes'])==len({x['pid']for x in guard['processes']})==2
for x in guard['processes']:
 v=x['result'];assert x['actual_wait_returncode']==0 and v['check_count']==74
 assert v['Torch_imported']is False and v['GPU_dispatches']==0
 assert v['controller_sha256']==sha(r/'scripts/resident_controller.py')and v['actual_guard_sha256']==sha(r/'src/qualification_guards.py')
 assert v['negative_fixture_storage']['outside_science_root']is True
native=read(r/'analysis/CPU_RUNTIME_NATIVE_CLOSURE_GUARD_NATIVE_EXIT_TARGET_TWO_FRESH_RESULT.json')
assert native['status']=='PASS_TWO_FRESH_TARGET_CPU_NATIVE_CLOSURE_GUARD'
assert len(native['processes'])==len({x['pid']for x in native['processes']})==2
native_ids=[]
for x in native['processes']:
 v=x['result'];assert x['actual_wait_returncode']==0 and v['check_count']==47
 assert v['Torch_imported']is False and v['GPU_dispatches']==v['model_loads']==0 and v['CUDA_visible_devices']==''
 assert v['guard_sha256']==sha(r/'src/qualification_guards.py')and v['test_script_sha256']==sha(r/'scripts/test_runtime_native_closure_cpu.py')
 assert pathlib.Path(v['source_root']).resolve()==r and v['external_negative_fixtures_destroyed_with_temporary_directory']is True
 ident=v['identity'];assert ident['pid']==x['pid']and not C.same_process(ident);native_ids.append(ident)
assert len({(x['pid'],x['starttick'])for x in native_ids})==2
sys.path.insert(0,str(r/'operations'));import engineering_gpu_evidence as E
engineering=E.validate(r)
completion_path=engineering['active_completion'];completion=read(completion_path)
for proof in engineering['successful_completions']:
 assert not C.same_process(proof['worker_identity'])and not C.same_process(proof['supervisor_identity'])
assert completion['status']=='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED'and completion['actual_wait_returncode']==0
assert completion['all_exact_own_processes_exited']is True and completion['own_GPU_UUID_released']is True
assert completion['model_forward_calls']==completion['QDQ_calls']==completion['AIME_generations']==0 and completion['counts_as_scientific_qualification']is False
assert not C.same_process(completion['worker_identity'])
interval=completion['resource_interval'];assert interval['monotonic_same_host']is True
assert interval['dispatch_monotonic_lower']<=interval['identity_monotonic_upper']<=interval['exit_first_observed_monotonic']<=interval['release_first_observed_monotonic']
# Both separately authorized probe contracts must be retained as exact inputs.
for key in('probe_source_freeze_sha256','probe_permit_sha256'):
 assert any(x['sha256']==completion[key]for x in freeze['files']),key+' omitted from scientific source inputs'
result=read(bound(completion['result']));observations=result['observations'];assert observations
obs={x['path']:x for x in observations};assert len(obs)==len(observations)
review_path=engineering['active_review'];review=read(review_path)
assert review['status']=='PASS_ALL_LOAD_ONLY_NATIVE_MAP_STAGES_EXACTLY_PINNED'
assert review['native_manifest_sha256']==sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json')and review['completion_sha256']==sha(completion_path)
assert review['actual_fresh_native_closure_count']==native_count and len(review['stages'])==len(observations)
stage_verdicts=[]
for stage in review['stages']:
 assert stage['path']in obs and stage['sha256']==obs[stage['path']]['sha256']
 assert stage['assessment_status']=='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED'
 p=bound(obs[stage['path']]);saved=read(p)
 assert isinstance(saved['raw_maps'],str)and saved['raw_maps']
 actual=G.assess_native_maps(r,saved['raw_maps'])
 assert actual['status']=='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED'and not actual['errors']
 assert len(actual['mapped_native'])==stage['mapped_native_count']
 stage_verdicts.append(dict(path=stage['path'],sha256=sha(p),mapped_native_count=len(actual['mapped_native']),assessment_status=actual['status']))
assert len({x['path']for x in stage_verdicts})==len(observations)
sys.path.insert(0,str(r/'scripts'));import resident_controller as controller
prepared=controller.prepare_execution_directories(r);assert prepared['unit_targets_created']is False
units=read(r/'QUALIFICATION_JOB_MANIFEST.json')['units'];jobs=read(r/'FORMAL_JOB_MANIFEST.json')['jobs']
assert len(units)==len({x['id']for x in units})==58 and len(jobs)==len({x['id']for x in jobs})==60
assert {x['condition']for x in jobs}=={'C1','S3'}and all(x['seed']==1 and x['max_new_tokens']==81920 and x['gpu_uuid']==C.UUIDS[x['gpu']]for x in jobs)
assert not any(os.path.lexists(r/'qualification'/x['id'])or os.path.lexists(r/'dispatch/qualification'/(x['id']+'.json'))for x in units)
assert not any(os.path.lexists(r/'formal'/x['id'])for x in jobs)
gpu=C.query_gpu();assert not any(x['uuid']in C.UUIDS.values()for x in gpu['apps'])
for index,uuid in C.UUIDS.items():
 device=next(x for x in gpu['devices']if x['physical_index']==index)
 assert device['uuid']==uuid and '3090'in device['name']and device['memory_free_MiB']>=23000
assert set(repair['selected_uuids'])==set(C.UUIDS.values())and repair['GPU0_authorized']is False
ram=os.statvfs('/dev/shm');data=os.statvfs('/data');assert ram.f_bavail*ram.f_frsize>=100*1024**3 and data.f_bavail*data.f_frsize>=8*1024**3
no_symlinks()
receipt=dict(status='PASS_FINAL_CPU_SOURCE_RESOURCE_ADMISSION_QUALIFICATION_ONLY_NOT_GPU_PROOF',
 source_freeze_sha256=source_sha,source_count=freeze['count'],prior153_source_freeze_sha256=prior_sha,
 original25_asset_fresh_SHA=True,unchanged_math_input_files=len(math_paths),resident_core_unchanged=True,
 actual_external_dependency_fresh_SHA_count=61,actual_fresh_native_closure_count=native_count,
 native_closure_manifest_sha256=sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json'),model_stat_count=17,prior_P1_user_cancelled_release=prior_P1,
 actual_target_CPU_two_fresh_exit0=True,CPU_checks_each=79,actual_guard_path_CPU_checks_each=74,
 actual_native_closure_guard_CPU_checks_each=47,actual_native_closure_guard_two_fresh_exited=native_ids,
 all_python_compile_Py310=True,selected_UUIDs_observed_empty=gpu,shared_RAM_available_bytes=ram.f_bavail*ram.f_frsize,
 data_available_bytes=data.f_bavail*data.f_frsize,scientific_qualification_GPU_dispatches=0,
 engineering_native_loader_GPU_probes=4,engineering_diagnostic_GPU_dispatches=8,total_GPU_work_dispatches=12,GPU_dispatches=12,formal_generations=0,
 engineering_probe_completion_sha256=sha(completion_path),engineering_native_maps_final_review_sha256=sha(review_path),
 engineering_probe_worker_identity=completion['worker_identity'],all_saved_probe_maps_strictly_reassessed=stage_verdicts,
 engineering_probe_is_scientific_qualification=False,Torch_imported='torch'in sys.modules,
 actual_required_parent_directory_preparation=prepared,monotonic=time.monotonic())
assert receipt['Torch_imported']is False;admission_raw=encoded(receipt)
permit=dict(task_id=C.TASK,execution_attempt=r.name,ALLOW_GPU='QUALIFICATION_ONLY',source_freeze_sha256=source_sha,
 qualification_plan_sha256=sha(r/'QUALIFICATION_JOB_MANIFEST.json'),execution_plan_sha256=sha(r/'GPU_EXECUTION_PLAN.json'),
 human_authorization_sha256=sha(r/'HUMAN_CONTINUATION_AUTHORIZATION.json'),prior_p1_stop_admission_sha256=sha(r/'PRIOR_P1_USER_STOP_ADMISSION.json'),
 final_CPU_admission_sha256=hashlib.sha256(admission_raw).hexdigest(),devices=list(C.UUIDS.values()),repair_authorization_sha256=sha(r/'HUMAN_REPAIR_AUTHORIZATION.json'),
 runtime_native_closure_sha256=sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json'),engineering_probe_completion_sha256=sha(completion_path),
 engineering_gpu_evidence_binding_sha256=sha(r/E.BINDING),native_exit_repair_authorization_sha256=sha(r/'HUMAN_NATIVE_EXIT_REPAIR_AUTHORIZATION.json'),
 formal_permission=False,conditional_formal_separate_permit_required=True,GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,
 no_progress_watchdog_seconds=1800,own_cleanup_observation_seconds=120,no_GPU_retry_after_STARTED=True)
files={'SOURCE_FREEZE.json':raw,'operations/FINAL_CPU_ADMISSION.json':admission_raw,'GPU_PERMIT.json':encoded(permit)}
for name,b in files.items():write_new(r/name,b)
assert C.verify_sources(r)==source_sha
print(json.dumps(dict(status='PASS_ONCE_FROZEN_QUALIFICATION_ONLY',files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())for n,b in files.items()})))
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    payload=source_payload()
    if not args.execute:
        print(json.dumps(dict(status='PLAN_ONLY_NO_REMOTE_ACTION',source_count=payload['freeze']['count'],source_sha256=payload['source_sha256'],
                              scientific_qualification_GPU_dispatches=0,engineering_native_loader_GPU_probes=4)));return
    command='exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 '+shlex.join([deploy.PYTHON,'-u','-B','-c',PROGRAM,deploy.REMOTE])
    result=subprocess.run(deploy.SSH+[command],input=json.dumps(payload).encode(),capture_output=True,timeout=900)
    with(ROOT/'operations/FINAL_FREEZE_ADMISSION_TRANSPORT.json').open('x')as f:
        json.dump(dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),f,indent=2);f.write('\n')
    assert result.returncode==0,result.stderr.decode();report=json.loads(result.stdout)
    for name,row in report['files'].items():
        b=base64.b64decode(row['base64'],validate=True);assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256']
        with(ROOT/name).open('xb')as f:f.write(b)
    print(json.dumps(dict(status=report['status'],source_count=payload['freeze']['count'],source_sha256=payload['source_sha256'],permit_sha256=report['files']['GPU_PERMIT.json']['sha256'])))


if __name__=='__main__':main()
