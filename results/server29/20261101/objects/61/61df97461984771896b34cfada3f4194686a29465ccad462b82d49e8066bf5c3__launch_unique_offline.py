#!/usr/bin/env python3
"""Launch this newly frozen engineering repair exactly once; default is PLAN."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PROGRAM=r'''
import json,os,pathlib,shlex,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);assert not r.is_symlink();r=r.resolve()
name='qwen-oscar-resident-eng-repair-20261022'
assert r.name=='qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_engineering_repair_20261022'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path.insert(0,str(r/'src'));import resident_execution_control as C
assert not any(os.path.lexists(r/p)for p in('OFFLINE_STARTED.json','CONTROLLER_STARTED.json','operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json','operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json'))
source=C.verify_sources(r);C.verify_model_and_external(r,fresh_external=False);C.validate_prior_p1_stop(r)
permit=C.read(r/'GPU_PERMIT.json');admission=C.read(r/'operations/FINAL_CPU_ADMISSION.json');repair=C.read(r/'HUMAN_REPAIR_AUTHORIZATION.json')
assert repair['prior_source_freeze_sha256']=='f0c27e48442e924b1a49fa254c4f262f49be85da00a3ec52c3073d7e73e77d5c'
assert repair['prior153_original_science_seal_exists']is False and repair['qualification_authorized']and repair['formal_conditionally_authorized']
assert permit['ALLOW_GPU']=='QUALIFICATION_ONLY'and permit['source_freeze_sha256']==source and permit['formal_permission']is False
assert permit['repair_authorization_sha256']==C.sha(r/'HUMAN_REPAIR_AUTHORIZATION.json')
assert permit['final_CPU_admission_sha256']==C.sha(r/'operations/FINAL_CPU_ADMISSION.json')
assert permit['runtime_native_closure_sha256']==C.sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json')
assert admission['source_freeze_sha256']==source and admission['actual_external_dependency_fresh_SHA_count']==61 and admission['model_stat_count']==17
assert admission['CPU_checks_each']==79 and admission['actual_guard_path_CPU_checks_each']==74 and admission['actual_native_closure_guard_CPU_checks_each']==47
assert admission['scientific_qualification_GPU_dispatches']==0 and admission['engineering_native_loader_GPU_probes']==4 and admission['GPU_dispatches']==12
assert admission['engineering_probe_is_scientific_qualification']is False
assert admission['actual_required_parent_directory_preparation']['status']=='PASS_ORDINARY_EXECUTION_PARENT_DIRECTORIES'
assert all((r/n).is_dir()and not(r/n).is_symlink()for n in admission['actual_required_parent_directory_preparation']['directories'])
sys.path.insert(0,str(r/'operations'));import engineering_gpu_evidence as E
engineering=E.validate(r)
completion_path=engineering['active_completion'];completion=C.read(completion_path)
assert completion['status']=='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED'and completion['actual_wait_returncode']==0
assert completion['all_exact_own_processes_exited']and completion['own_GPU_UUID_released']and completion['counts_as_scientific_qualification']is False
assert completion['model_forward_calls']==completion['QDQ_calls']==completion['AIME_generations']==0
assert not C.same_process(completion['worker_identity'])
if 'supervisor_identity'in completion:assert not C.same_process(completion['supervisor_identity'])
assert permit['engineering_probe_completion_sha256']==admission['engineering_probe_completion_sha256']==C.sha(completion_path)
review_path=r/'operations/ENGINEERING_NATIVE_MAPS_FINAL_REVIEW.json';review=C.read(review_path)
assert review['status']=='PASS_ALL_LOAD_ONLY_NATIVE_MAP_STAGES_EXACTLY_PINNED'
assert review['completion_sha256']==C.sha(completion_path)and review['native_manifest_sha256']==C.sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json')
assert admission['engineering_native_maps_final_review_sha256']==C.sha(review_path)
assert review['actual_fresh_native_closure_count']==admission['actual_fresh_native_closure_count']==len(C.read(r/'analysis/RUNTIME_NATIVE_CLOSURE.json')['files'])
assert set(repair['selected_uuids'])==set(C.UUIDS.values())and repair['GPU0_authorized']is False
assert not any(os.path.lexists(r/'qualification'/x['id'])for x in C.read(r/'QUALIFICATION_JOB_MANIFEST.json')['units'])
assert not any(os.path.lexists(r/'formal'/x['id'])for x in C.read(r/'FORMAL_JOB_MANIFEST.json')['jobs'])
obs=C.query_gpu();assert not any(a['uuid']in C.UUIDS.values()for a in obs['apps'])
probe=subprocess.run(['tmux','has-session','-t',name],capture_output=True);assert probe.returncode==1,'tmux session already exists or inventory command failed'
C.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json',dict(status='ONE_ACTUAL_LAUNCH_ATTEMPT',tmux=name,source_freeze_sha256=source,
 permit_sha256=C.sha(r/'GPU_PERMIT.json'),repair_authorization_sha256=C.sha(r/'HUMAN_REPAIR_AUTHORIZATION.json'),
 engineering_probe_completion_sha256=C.sha(completion_path),scientific_GPU_dispatches_before_launch=0,prior_engineering_GPU_probes=4,prior_engineering_diagnostic_dispatches=8,monotonic=time.monotonic()))
command=['tmux','new-session','-d','-s',name,shlex.join(['bash',str(r/'scripts/run_resident_offline.sh'),str(r)])]
x=subprocess.run(command,capture_output=True,text=True);assert x.returncode==0,x.stderr
for i in range(120):
 if(r/'CONTROLLER_STARTED.json').exists():break
 time.sleep(.25)
started=C.read(r/'CONTROLLER_STARTED.json');actual=C.process_identity(started['identity']['pid'])
assert actual==started['identity']and C.same_process(actual),started
receipt=dict(status='PASS_UNIQUE_OFFLINE_CONTROLLER_ACTUALLY_STARTED',command=command,tmux=name,controller_identity=actual,
 source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),repair_authorization_sha256=C.sha(r/'HUMAN_REPAIR_AUTHORIZATION.json'),
 engineering_probe_completion_sha256=C.sha(completion_path),prior_engineering_GPU_probes=4,prior_engineering_diagnostic_dispatches=8,
 offline=C.read(r/'OFFLINE_STARTED.json'),NVML=C.query_gpu(),monotonic=time.monotonic())
C.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json',receipt);print(json.dumps(receipt,sort_keys=True))
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    cfg=json.loads((ROOT/'ROOT.json').read_text())
    if not args.execute:
        print(json.dumps(dict(status='PLAN_ONLY_NO_GPU_OR_REMOTE_ACTION',remote_root=cfg['remote_root'],tmux='qwen-oscar-resident-eng-repair-20261022')));return
    command=['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=20','zypan@202.38.247.55',
             'exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 '+shlex.join([cfg['remote_python'],'-u','-B','-c',PROGRAM,cfg['remote_root']])]
    result=subprocess.run(command,capture_output=True,timeout=180)
    with(ROOT/'operations/RESIDENT_OFFLINE_LAUNCH_TRANSPORT.json').open('x')as f:
        json.dump(dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),f,indent=2);f.write('\n')
    assert result.returncode==0,result.stderr.decode();receipt=json.loads(result.stdout)
    with(ROOT/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json').open('xb')as f:f.write(result.stdout)
    print(json.dumps(dict(status=receipt['status'],controller_identity=receipt['controller_identity'],offline=receipt['offline'])))


if __name__=='__main__':main()
