"""CPU admission and one tmux launch of qualification -> conditional formal."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main(root):
    sys.path.insert(0,str(root/'src'))
    import execution_control as C
    C.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Offline launcher must be CPU-only')
    C.require(root==Path('/tmp')/C.EXECUTION_REVISION and root==Path(__file__).resolve().parents[1],
              'Exact independent execution root required')
    permit=C.authorize(root,root/'GPU_PERMIT.json')
    C.model_guard(root);storage=C.storage_guard(root)
    C.require(not(root/'GPU_RUN_STARTED.json').exists() and not(root/'SESSION_RESULT.json').exists(),
              'Already dispatched/terminal; never restart')
    C.require(not list((root/'execution').glob('*/DISPATCH.json')) and
              not list((root/'formal').glob('*/DISPATCH.json')),'Existing unit dispatch; never repeat')
    C.require(C.read(root/'CPU_CODEC_QUALIFICATION.json')['status']=='PASS_CPU_CONTRACTS_AND_INTEGRATION_NOT_GPU',
              'Preparatory CPU qualification missing')
    C.require(C.read(root/'calibration/FP_COLLECTION_EXIT_RELEASE.json')['status']=='PASS_ACTUAL_EXIT0_AND_RELEASE',
              'Actual calibration exit/release missing')
    calibration_archive=C.read(root/'calibration/ARCHIVE29_VERIFICATION.json')
    C.require(calibration_archive['status']=='PASS_INDEPENDENT29_NONAIME_RAW_SOURCE_SHA_BYTES_NOREPLACE',
              'Durable independent calibration evidence missing')
    C.validate_jobs(root,C.read(root/'FORMAL_JOB_MANIFEST.json')['jobs'])
    inventory=[C.device(g) for g in C.AUTHORIZED_GPUS]
    session='qwen-wscale-rowbal-byteview-repair-20261025'
    check=subprocess.run(['tmux','has-session','-t',session],capture_output=True,text=True)
    C.require(check.returncode==1,'Existing or unresolved tmux session; do not relaunch')
    C.require(not(root/'finisher.log').exists(),'Existing offline output; inspect rather than truncate')
    argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1',
          'TRANSFORMERS_OFFLINE=1','bash',str(root/'scripts/run_offline.sh')]
    C.atomic_new(root/'operations/OFFLINE_LAUNCH_CPU_ADMISSION.json',dict(
        status='PASS_ACTUAL_PRE_TMUTX_CPU_ADMISSION_NO_GPU_DISPATCH',
        source_freeze_sha256=C.sha(root/'SOURCE_FREEZE.json'),permit_sha256=C.sha(root/'GPU_PERMIT.json'),
        model17_stat='PASS',storage=storage,selected_devices_fresh=inventory,
        shared_nonAIME_archive=calibration_archive,monotonic=time.monotonic()))
    C.atomic_new(root/'operations/OFFLINE_LAUNCH_ATTEMPT.json',dict(
        session=session,argv=argv,source_freeze_sha256=C.sha(root/'SOURCE_FREEZE.json'),
        permit_sha256=C.sha(root/'GPU_PERMIT.json'),CPU_launcher_identity=C.process_identity(os.getpid()),
        conditional_formal_count=90,monotonic=time.monotonic(),no_automatic_retry=True))
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(root/'finisher.log'))+' 2>&1'
    proc=subprocess.run(['tmux','new-session','-d','-s',session,command],capture_output=True,text=True)
    receipt=dict(status='TMUX_LAUNCH_COMMAND_RETURNED',session=session,actual_returncode=proc.returncode,
        stderr=proc.stderr,argv=argv,source_freeze_sha256=C.sha(root/'SOURCE_FREEZE.json'),
        permit_sha256=C.sha(root/'GPU_PERMIT.json'),monotonic=time.monotonic())
    if proc.returncode==0:
        pane=subprocess.run(['tmux','display-message','-p','-t',session,'#{pane_pid}'],capture_output=True,text=True)
        receipt['pane_query_returncode']=pane.returncode
        if pane.returncode==0:
            try:receipt['actual_offline_shell_identity']=C.process_identity(int(pane.stdout.strip()))
            except (OSError,ValueError):receipt['actual_offline_shell_identity']='TERMINAL_OR_NOT_YET_OBSERVED; inspect actual receipts'
    C.atomic_new(root/'operations/OFFLINE_LAUNCH_RECEIPT.json',receipt)
    C.require(proc.returncode==0,'tmux launch failed; preserve attempt, no automatic retry')
    import json
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve())
