"""CPU launcher. Default only verifies readiness; GPU run requires explicit flag."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

D=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_03')
W=Path('/data/zypan/worktrees/ling-kda-prospective-qualified-completion')
os.environ['LING_PROSPECTIVE_RESULTS']=str(D)
sys.path.insert(0,str(W));sys.dont_write_bytecode=True
from recovery_state import atomic_json,lock,sha
from prospective_common import read

def readiness():
    assert not (D/'STOP.json').exists(),'FIRST_FAILURE_NO_RETRY'
    freeze=read(D/'SOURCE_CODE_FREEZE.json')
    assert subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()==freeze['source_commit']
    assert subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True)==''
    for n,h in freeze['files'].items():assert sha(W/n)==h,('SOURCE_CHANGED',n)
    assert sha(D/'CALIBRATION_CONFIG.json')==freeze['config_sha256']
    assert sha(D/'EXECUTION_ROUTING.json')==freeze['routing_sha256']
    assert sha(D/'FROZEN_SPEC.json')==freeze['scientific_spec_sha256']
    assert read(D/'ENGINEERING_READINESS.json')['total_current_CPU_tests']==37
    assert read(D/'CHECKPOINT_REUSE_CPU_AUDIT.json')['CUDA_initialized'] is False
    assert read(D/'new_recovery_cpu_reverification_02/NEW_RECOVERY_TEST_REPORT.json')['source_sha256']==sha(W/'test_qualified_completion.py')
    return freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-authorized',action='store_true',help='Use only after the human explicitly authorizes running the prepared experiment')
    args=parser.parse_args();readiness()
    if not args.run_authorized:
        print(json.dumps(dict(status='CODE_READY_AWAITING_EXPLICIT_USER_RUN',GPU_workers_launched=0,resource_polling=False)),flush=True);return
    held=lock(D/'launch.lock')
    # Check the controller lock without taking over a live controller.
    check=lock(D/'controller.lock');check.close()
    binding=read(D/'LAUNCHER_SOURCE_BINDING.json');assert sha(__file__)==binding['sha256']
    # Recheck genuine reused prefix bytes before obtaining fresh resource evidence.
    for b in read(D/'REUSED_PARENT_BINDINGS.json')['units'].values():
        assert sha(b['original_manifest_path'])==b['original_manifest_sha256']
    for p in (D/'runs').glob('*__parent.json'):
        parent=read(p);cp=D/'runs'/p.name.replace('__parent.json','__8192.pt');assert sha(cp)==parent['checkpoint_sha256']
    from gpu_admission import classify,classify_window
    from gpu_telemetry import observe
    routing=read(D/'EXECUTION_ROUTING.json');uuids=list(routing['authorized_devices']);ready=False
    # Window01 is immutable historical preparation evidence. Only two more
    # windows are available in this execution; no unbounded waiting/retry.
    for number in [2,3]:
        path=D/f'GPU_PREFLIGHT_WINDOW_{number:02d}.jsonl'
        assert not path.exists(),'EXISTING_PREFLIGHT_NOT_REPEATED_OR_OVERWRITTEN'
        rows=observe(uuids,path,seconds=60,interval=5)
        results=[]
        for uuid in uuids:
            samples=[r for r in rows if r['uuid']==uuid]
            results.append(dict(uuid=uuid,status=classify_window(samples),sample_count=len(samples),states=[classify(r) for r in samples]))
        atomic_json(D/f'GPU_PREFLIGHT_RESULT_{number:02d}.json',dict(epoch=time.time(),window_number=number,maximum_total_windows=3,window_sha256=sha(path),results=results,science_GPU_workers_launched=0))
        if all(r['status']=='AVAILABLE_VERIFIED' for r in results):ready=True;break
    if not ready:
        atomic_json(D/'STOP.json',dict(status='RESOURCE_UNRESOLVED_STOP',stage='BOUNDED_READONLY_PREFLIGHT',epoch=time.time(),reason='Configured target devices did not pass the frozen resource gate within the remaining two bounded windows',no_retry=True,GPU_workers_launched=0))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        r=subprocess.run([sys.executable,'-B','-u',str(W/'finalize_cpu.py')],env=env);assert r.returncode==0,'CPU_DELIVERY_ERROR_PRESERVE_SCIENCE';return
    assert not (D/'STOP.json').exists()
    uuid=uuids[0]
    env=dict(os.environ,LING_TARGET_GPU_UUID=uuid,CUDA_VISIBLE_DEVICES=str(routing['authorized_devices'][uuid]),LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    env.pop('LING_EXECUTION_LANE',None)
    atomic_json(D/'USER_AUTHORIZED_RUN_LAUNCH.json',dict(epoch=time.time(),launcher_sha256=sha(__file__),source_commit=read(D/'SOURCE_CODE_FREEZE.json')['source_commit'],explicit_run_flag=True,resource_preflight='PASS',formal_order_unchanged=True))
    os.set_inheritable(held.fileno(),True)
    os.execve(sys.executable,[sys.executable,'-B','-u',str(W/'recovery_controller.py')],env)

if __name__=='__main__':main()
