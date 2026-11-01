#!/usr/bin/env python3
"""Launch the newly authorized frozen execution exactly once, then observe."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PROGRAM=r'''
import json,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);name='qwen-oscar-resident-dir-repair-20261022'
sys.path.insert(0,str(r/'src'));import resident_execution_control as C
assert not any((r/p).exists()for p in ['OFFLINE_STARTED.json','CONTROLLER_STARTED.json','operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json'])
source=C.verify_sources(r);C.verify_model_and_external(r,fresh_external=False);C.validate_prior_p1_stop(r)
permit=C.read(r/'GPU_PERMIT.json');admission=C.read(r/'operations/FINAL_CPU_ADMISSION.json')
assert permit['ALLOW_GPU']=='QUALIFICATION_ONLY'and permit['source_freeze_sha256']==source
assert permit['repair_authorization_sha256']==C.sha(r/'HUMAN_REPAIR_AUTHORIZATION.json')
assert admission['actual_guard_path_CPU_checks_each']==74
assert admission['actual_required_parent_directory_preparation']['status']=='PASS_ORDINARY_EXECUTION_PARENT_DIRECTORIES'
assert all((r/n).is_dir()and not(r/n).is_symlink()for n in admission['actual_required_parent_directory_preparation']['directories'])
obs=C.query_gpu();assert not any(a['uuid']in C.UUIDS.values()for a in obs['apps'])
probe=subprocess.run(['tmux','has-session','-t',name],capture_output=True);assert probe.returncode!=0
C.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json',dict(status='ONE_ACTUAL_LAUNCH_ATTEMPT',tmux=name,source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),monotonic=time.monotonic()))
command=['tmux','new-session','-d','-s',name,'bash '+str(r/'scripts/run_resident_offline.sh')+' '+str(r)]
x=subprocess.run(command,capture_output=True,text=True);assert x.returncode==0,x.stderr
for i in range(120):
 if (r/'CONTROLLER_STARTED.json').exists():break
 time.sleep(.25)
started=C.read(r/'CONTROLLER_STARTED.json');actual=C.process_identity(started['identity']['pid'])
assert actual==started['identity']and C.same_process(actual),started
receipt=dict(status='PASS_UNIQUE_OFFLINE_CONTROLLER_ACTUALLY_STARTED',command=command,tmux=name,controller_identity=actual,source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),offline=C.read(r/'OFFLINE_STARTED.json'),NVML=C.query_gpu(),monotonic=time.monotonic())
C.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json',receipt)
print(json.dumps(receipt,sort_keys=True))
'''


def main():
    cfg=json.loads((ROOT/'ROOT.json').read_text())
    command=['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=20',
             'zypan@202.38.247.55',shlex.join([cfg['remote_python'],'-u','-B','-c',PROGRAM,cfg['remote_root']])]
    result=subprocess.run(command,capture_output=True,timeout=180)
    with (ROOT/'operations/RESIDENT_OFFLINE_LAUNCH_TRANSPORT.json').open('x')as f:
        json.dump(dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),f,indent=2);f.write(chr(10))
    assert result.returncode==0,result.stderr.decode()
    receipt=json.loads(result.stdout)
    with (ROOT/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json').open('xb')as f:f.write(result.stdout)
    print(json.dumps(dict(status=receipt['status'],controller_identity=receipt['controller_identity'],offline=receipt['offline'])))


if __name__=='__main__':main()
