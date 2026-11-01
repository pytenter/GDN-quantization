"""Launch the newly admitted profiler repair once. No implicit retry."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = "\nimport pathlib,os,sys,json,subprocess,shlex,time\nr=pathlib.Path(sys.argv[1]).resolve();name='qwen-resident-qualification-continuation-20261023'\nassert r.name=='qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_qualification_continuation_20261023'\nassert os.environ.get('CUDA_VISIBLE_DEVICES')==''\nsys.path[:0]=[str(r/'src'),str(r/'operations')]\nimport resident_execution_control as C,profiler_engineering_evidence as P,qualification_reuse as Q\nassert not any(os.path.lexists(r/p) for p in ['OFFLINE_STARTED.json','CONTROLLER_STARTED.json','operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json','operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json'])\nsource=C.verify_sources(r);permit=C.read(r/'GPU_PERMIT.json');admission=C.read(r/'operations/FINAL_CPU_ADMISSION.json')\nassert permit['ALLOW_GPU']=='QUALIFICATION_ONLY' and permit['source_freeze_sha256']==source and permit['formal_permission'] is False\nassert permit['human_profiler_repair_authorization_sha256']==C.sha(r/'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json')\nassert permit['final_CPU_admission_sha256']==C.sha(r/'operations/FINAL_CPU_ADMISSION.json') and admission['source_freeze_sha256']==source\nassert P.validate(r,C)==C.read(r/'analysis/PROFILER_ENGINEERING_EVIDENCE.json')\nassert admission['current_engineering_GPU_dispatches']==0 and admission['qualification_reused']==48 and admission['qualification_remaining']==10 and admission['scientific_qualification_GPU_dispatches']==admission['formal_generations']==0\nassert admission['prior48_qualifications_reused_as_new_PASS'] is False\nassert len(Q.seed(C,r))==48 and permit['qualification_reuse_admission_sha256']==C.sha(r/'QUALIFICATION_REUSE_ADMISSION.json')\nassert permit['human_qualification_reuse_authorization_sha256']==C.sha(r/'HUMAN_QUALIFICATION_REUSE_AUTHORIZATION.json')\nC.verify_model_and_external(r,fresh_external=False);C.validate_prior_p1_stop(r)\nobs=C.query_gpu();assert not any(x['uuid'] in C.UUIDS.values() for x in obs['apps'])\nassert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==1\nC.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_ATTEMPT.json',dict(status='ONE_ACTUAL_ORIGIN_AWARE_QUALIFICATION_CONTINUATION_LAUNCH_ATTEMPT',tmux=name,source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),human_authorization_sha256=permit['human_profiler_repair_authorization_sha256'],monotonic=time.monotonic(),no_automatic_retry=True))\ncommand=['tmux','new-session','-d','-s',name,shlex.join(['bash',str(r/'scripts/run_resident_offline.sh'),str(r)])]\nx=subprocess.run(command,capture_output=True,text=True);assert x.returncode==0,x.stderr\nfor i in range(120):\n if (r/'CONTROLLER_STARTED.json').exists():break\n time.sleep(.25)\nstarted=C.read(r/'CONTROLLER_STARTED.json');actual=C.process_identity(started['identity']['pid'])\nassert actual==started['identity'] and C.same_process(actual)\nreceipt=dict(status='PASS_UNIQUE48_PLUS10_CONTINUATION_OFFLINE_CONTROLLER_ACTUALLY_STARTED',tmux=name,command=command,controller_identity=actual,source_freeze_sha256=source,permit_sha256=C.sha(r/'GPU_PERMIT.json'),offline=C.read(r/'OFFLINE_STARTED.json'),NVML=C.query_gpu(),monotonic=time.monotonic(),inherited_engineering_dispatches=27,current_engineering_dispatches=0,qualification_reused=48,qualification_current_remaining=10,formal_permission=False)\nC.atomic_new(r/'operations/RESIDENT_OFFLINE_LAUNCH_RECEIPT.json',receipt);print(json.dumps(receipt,sort_keys=True))\n"


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
