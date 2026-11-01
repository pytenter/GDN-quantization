"""Launch the newly admitted profiler repair once. No implicit retry."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = r'''
import pathlib,os,sys,json,subprocess,shlex,time
r=pathlib.Path(sys.argv[1]).resolve();name='qwen-resident-profiler-repair-20261023'
assert r.name=='qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_profiler_exit_repair_20261023'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=[str(r/'src'),str(r/'operations')]
import resident_execution_control as C,profiler_engineering_evidence as P
assert not any(os.path.lexists(r/p) for p in ['OFFLINE_STARTED.json','CONTROLLER_STARTED.json','operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json','operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json'])
source=C.verify_sources(r);permit=C.read(r/'GPU_PERMIT.json');admission=C.read(r/'operations/FINAL_CPU_ADMISSION.json')
assert permit['ALLOW_GPU']=='QUALIFICATION_ONLY' and permit['source_freeze_sha256']==source and permit['formal_permission'] is False
assert permit['human_profiler_repair_authorization_sha256']==C.sha(r/'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json')
assert permit['final_CPU_admission_sha256']==C.sha(r/'operations/FINAL_CPU_ADMISSION.json') and admission['source_freeze_sha256']==source
assert P.validate(r,C)==C.read(r/'analysis/PROFILER_ENGINEERING_EVIDENCE.json')
assert admission['current_engineering_GPU_dispatches']==15 and admission['scientific_qualification_GPU_dispatches']==admission['formal_generations']==0
assert admission['prior48_qualifications_reused_as_new_PASS'] is False
C.verify_model_and_external(r,fresh_external=False);C.validate_prior_p1_stop(r)
obs=C.query_gpu();assert not any(x['uuid'] in C.UUIDS.values() for x in obs['apps'])
assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==1
C.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json',dict(status='ONE_ACTUAL_PROFILER_REPAIR_LAUNCH_ATTEMPT',tmux=name,source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),human_authorization_sha256=permit['human_profiler_repair_authorization_sha256'],monotonic=time.monotonic(),no_automatic_retry=True))
command=['tmux','new-session','-d','-s',name,shlex.join(['bash',str(r/'scripts/run_resident_offline.sh'),str(r)])]
x=subprocess.run(command,capture_output=True,text=True);assert x.returncode==0,x.stderr
for i in range(120):
 if (r/'CONTROLLER_STARTED.json').exists():break
 time.sleep(.25)
started=C.read(r/'CONTROLLER_STARTED.json');actual=C.process_identity(started['identity']['pid'])
assert actual==started['identity'] and C.same_process(actual)
receipt=dict(status='PASS_UNIQUE_PROFILER_REPAIR_OFFLINE_CONTROLLER_ACTUALLY_STARTED',tmux=name,command=command,controller_identity=actual,source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),offline=C.read(r/'OFFLINE_STARTED.json'),NVML=C.query_gpu(),monotonic=time.monotonic(),inherited_engineering_dispatches=12,current_engineering_dispatches=15,formal_permission=False)
C.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json',receipt);print(json.dumps(receipt,sort_keys=True))
'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    cfg=json.loads((ROOT/'ROOT.json').read_text())
    if not args.execute:
        print(json.dumps(dict(status='PLAN_ONLY_NO_LAUNCH',remote_root=cfg['remote_root'])));return
    command=['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',cfg['remote_python'],'-B','-c',PROGRAM,cfg['remote_root']])]
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    with (ROOT/'operations/RESIDENT_OFFLINE_LAUNCH_TRANSPORT.json').open('x') as stream:
        json.dump(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),stream,indent=2)
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    with (ROOT/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(dict(status=receipt['status'],controller_identity=receipt['controller_identity'])))


if __name__=='__main__':
    main()
