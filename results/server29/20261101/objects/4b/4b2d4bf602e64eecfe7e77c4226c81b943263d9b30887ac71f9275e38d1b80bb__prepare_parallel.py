"""Append-only scheduling amendment; no mathematical-source changes."""
import ast
import copy
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

REPO=Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
ROOT=REPO/'experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
BUG=ROOT/'amendments/CAPTURE_KEY_SHADOWING_BUGFIX_V1'
PAR=ROOT/'amendments/FOUR_GPU_ANCHOR_PARALLELISM_V1'
import sys
sys.path.insert(0,str(BUG/'scripts'))
from capture_original_drive import sha,read,save

def exact_copy(src,dst):
    assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst);assert sha(src)==sha(dst)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (BUG/'scientific_execution_authorization.json').exists()
    assert not list((BUG/'scientific').rglob('drive_manifest.json'))
    receipt=read(BUG/'bugfix_commit_receipt.json')
    for rel,h in receipt['code_sha256'].items():assert sha(BUG/rel)==h
    reused=[]
    for name in ['capture_original_drive.py','capture_extended_drive.py','isolated_replay_engineering.py',
                 'persistent_anchor_replay.py','qualify_long_and_streaming.py','scientific_lifecycle.py']:
        exact_copy(BUG/'scripts'/name,PAR/'scripts'/name)
        ast.parse((PAR/'scripts'/name).read_text())
        reused.append(dict(name=name,sha256=sha(PAR/'scripts'/name),byte_identical=True))
    for name in ['capture_engineering_preregistration.json','remaining_engineering_gate_policy.json','engineering_fixture_256_token_ids.json']:
        exact_copy(BUG/'protocol'/name,PAR/'protocol'/name)
    for name in ['capture_freeze_receipt.json','short_qualification_freeze.json','storage_lifecycle_report.json']:
        exact_copy(BUG/name,PAR/name)
    exact_copy(BUG/'engineering/capture_runtime_variability.json',PAR/'engineering/capture_runtime_variability.json')
    for f in sorted((BUG/'engineering/gates').iterdir()):
        if f.is_file():exact_copy(f,PAR/'engineering/gates'/f.name)
    for name in ['capture_01','capture_02','capture_2048','repeatability']:
        (PAR/'engineering'/name).symlink_to(BUG/'engineering'/name,target_is_directory=True)
    old=read(BUG/'protocol/extended_capture_policy.json');new=copy.deepcopy(old)
    for i,d in enumerate(new['documents'],1):d['output_directory']=str(PAR/'scientific'/('doc_%02d'%i))
    # Engineering2048 refers to the same already captured input, never recaptured.
    norm=copy.deepcopy(new)
    for d,o in zip(norm['documents'],old['documents']):d['output_directory']=o['output_directory']
    assert norm==old
    save(PAR/'protocol/extended_capture_policy.json',new)
    budget=copy.deepcopy(read(BUG/'engineering2048_storage_supplement.json'))
    extra=3*134217728+16777216
    base=budget['BASE_PEAK_BYTES']+extra
    margin=max(5*1024**3,(base+4)//5)
    assert margin>=budget['SAFETY_MARGIN_BYTES']
    available=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
    budget.update(BASE_PEAK_BYTES=base,SAFETY_MARGIN_BYTES=margin,PEAK_REQUIRED_STORAGE_BYTES=base+margin,
                  AVAILABLE_DATA_BYTES=available,HEADROOM_BYTES=available-base-margin,
                  additional_parallel_result_temporary_allowance_bytes=extra,
                  margin_counted_once=True,STORAGE_CAPACITY_GATE='PASS' if base+margin<=available else 'FAIL',
                  scope='One full document drive shared read-only by four independent anchor processes; no replicated drive',
                  previous_report_sha256=sha(BUG/'engineering2048_storage_supplement.json'))
    save(PAR/'engineering2048_storage_supplement.json',budget)
    assert budget['STORAGE_CAPACITY_GATE']=='PASS','FAIL_AND_STOP_STORAGE_CAPACITY'
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True)
    save(PAR/'protocol/parallel_schedule.json',dict(authorization='Human: 在保证磁盘空间的前提下，改为四卡并行',
         scheduling_only=True,documents=[1,2,3,4],capture_gpu=0,condition_gpu={'Native':0,'H':1,'C6R':2,'C7':3},
         max_concurrent_full_document_drives=1,shared_drive_read_only=True,max_concurrent_replays=4,
         next_document_requires_all_four_results_sealed_and_current_drive_released=True,
         topology_parity_fixture_tokens=256,topology_parity_conditions=['Native','H','C6R','C7'],
         topology_parity_criterion='Exact identity.outputs (every-token FP/Q hashes, metric-byte hashes, endpoint/horizon values and closure maxima) and runtime call counts vs existing GPU0 reference',
         gate_order=['existing_G11_2048_run01_natural_completion','same_frozen_G11_run02','G11_compare','G12',
                     'four_GPU_256_fixture_PARITY','full_premeasurement_commit','per_document_capture_parallel_replay_seal_release'],
         new_failure='FAIL_AND_STOP; no retry, no threshold relaxation; no incomplete drive deletion',
         runtime_gpu_inventory=devices,physical_devices_per_process='CUDA_VISIBLE_DEVICES restricts each worker to exactly one physical GPU; unchanged local cuda:0',
         scientific_inputs_and_computations_changed=False,training=False,AIME_generation=False))
    scripts={str(p.relative_to(PAR)):sha(p) for p in sorted((PAR/'scripts').glob('*')) if p.is_file()}
    # P.execute validates the unchanged mathematical source set. Administrative files are separately frozen.
    mathematics={r['name']:r['sha256'] for r in reused}
    qualified=read(BUG/'long_qualification_freeze.json')
    qualified['code_sha256']={'scripts/'+n:h for n,h in mathematics.items()}
    qualified['rules_sha256']=sha(PAR/'protocol/remaining_engineering_gate_policy.json')
    save(PAR/'long_qualification_freeze.json',qualified)
    extended=copy.deepcopy(read(BUG/'extended_capture_freeze.json'))
    # Preserve the receipt schema; route only the policy identity to the append-only namespace.
    for k in list(extended):
        if k in ['policy_sha256','capture_policy_sha256']:extended[k]=sha(PAR/'protocol/extended_capture_policy.json')
    save(PAR/'extended_capture_freeze.json',extended)
    control=copy.deepcopy(read(BUG/'continuation_control_freeze.json'));control['code_sha256']=scripts
    save(PAR/'continuation_control_freeze.json',control)
    save(PAR/'static_parallel_source_audit.json',dict(gate='PASS',mathematical_modules=reused,
         production_operations_changed=0,policy_comparison_excluding_only_scientific_output_paths='EXACT',
         documents_reselected=False,anchors_changed=False,release_implementation_byte_identical=True,
         previous_G0_G10_reexecuted=False,previous_G11_256_reexecuted=False,new_GPU_mapping_qualification_separate=True))
    runner=BUG/'scripts/run_repaired_stage.py'
    parent=Path('/proc/5663');child=Path('/proc/6662')
    parent_cmd=parent.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
    child_cmd=child.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
    cs=child.joinpath('stat').read_text().split()
    assert 'resume_bugfix_workflow.sh' in parent_cmd and '--stage replay2048 --run 01' in child_cmd
    assert cs[3]=='5663' and cs[2] not in ['Z','T']
    mode=stat.S_IMODE(runner.stat().st_mode)
    save(PAR/'scheduler_handoff_intent.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         parent_pid=5663,parent_start=parent.joinpath('stat').read_text().split()[21],
         worker_pid=6662,worker_start=cs[21],parent_cmd=parent_cmd,worker_cmd=child_cmd,
         runner_path=str(runner),runner_sha256=sha(runner),original_mode=mode,
         action='Temporarily deny subsequent old entrypoint reads; current already-loaded worker finishes naturally; restore identical mode/content after old scheduler exits',
         SIGSTOP_did_not_persist=True,no_GPU_worker_signal=True,no_process_group_signal=True,
         no_scientific_result_or_numerical_gate_failure=True,expected_old_next_launch='Permission denied before loading code/output; administrative supersession only'))
    runner.chmod(0)
    assert not os.access(runner,os.R_OK)
    save(PAR/'scheduler_launch_block_applied.json',dict(applied=True,current_worker_running=child.exists(),
         no_GPU_signal=True,old_controller_cannot_read_next_entrypoint=True,source_content_unchanged=True))
    print(json.dumps(dict(STATIC_AUDIT='PASS',STORAGE_CAPACITY_GATE='PASS',available=available,required=base+margin,
                         headroom=available-base-margin,old_next_launch_blocked=True)),flush=True)

if __name__=='__main__':main()
