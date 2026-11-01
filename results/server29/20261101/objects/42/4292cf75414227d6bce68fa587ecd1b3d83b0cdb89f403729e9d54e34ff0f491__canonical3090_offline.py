#!/usr/bin/env python3
"""tmux coordinator: no GPU launch until A800 closure and all four GPUs are idle."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import canonical3090_pipeline as P

ROOT=P.ROOT
A800='/root/autodl-tmp/GDN-A800-cleanroom/experiments/'+ROOT.name
SOCKET='/tmp/qwen_a800_cleanroom_control'
HOST='root@connect.nma1.seetacloud.com'
SSH=['ssh','-S',SOCKET,'-p','49629','-o','BatchMode=yes','-o','ConnectTimeout=20',HOST]
RSYNC_SSH='ssh -S '+SOCKET+' -p 49629 -o BatchMode=yes -o ConnectTimeout=20'
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'


def remote(cmd):return subprocess.check_output(SSH+[cmd],text=True,timeout=60).strip()


def idle():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return not out.strip()


def enough():
    if shutil.disk_usage('/data').free<15*1024**3:raise RuntimeError('below 15GiB; no next stage; no scientific artifact deletion')


def copy(source,dest,options=()):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['nice','-n','19','ionice','-c','3','rsync','-a','--partial','--bwlimit=20000','-e',RSYNC_SSH,*options,HOST+':'+source,str(dest)],check=True)


def main():
    P.check_freeze()
    for name in ('offline_pipeline.completion.json','offline_pipeline.error.json'):
        if (ROOT/'logs'/name).exists():raise RuntimeError('existing coordinator evidence; no automatic retry')
    print('Waiting for A800 matched training and fixed Shared16 posttraining closure.',flush=True)
    while True:
        code=remote('if test -f '+A800+'/logs/posttraining_pipeline.exitcode; then cat '+A800+'/logs/posttraining_pipeline.exitcode; else echo WAIT; fi')
        if code=='WAIT':time.sleep(60);continue
        if code!='0':raise RuntimeError('A800 training/posttraining stopped; preserve evidence and do not start deployment')
        break
    enough()
    # Transfer only small provenance/evidence plus the exact three selected matrices, not massive teachers or model weights.
    for directory in ('provenance','qualification','teacher','readiness','training','analysis','deployment'):
        copy(A800+'/'+directory+'/',ROOT/directory,('--exclude=targets_data/','--exclude=historical_rotations/','--exclude=*.pt','--exclude=*.log'))
    # A800 training configs must not overwrite the separately frozen inference protocol.
    copy(A800+'/protocol/',ROOT/'protocol/a800_training',('--exclude=*.pt',))
    copy(A800+'/logs/',ROOT/'logs/a800',('--exclude=*.log',))
    selected=P.read(ROOT/'deployment/selected_artifacts_pre_endpoint.json')
    for arm in P.ARMS:
        r=selected['conditions'][arm];dest=ROOT/'deployment'/arm/'frozen_rotation_fp32.pt'
        if not r['rotation_path'].startswith(A800+'/training/'+arm+'/frozen_rotations/'):raise RuntimeError('selected source path out of task scope')
        copy(r['rotation_path'],dest)
        if P.sha(dest)!=r['file_sha256']:raise RuntimeError('byte-for-byte selected R transfer failure')
    P.save(ROOT/'deployment/transfer_readback.json',{'status':'PASS','source':HOST,'selected_artifacts':selected['conditions'],'source_scientific_artifacts_deleted':False,'theta_rematerialization':False})
    while not idle():time.sleep(60)
    enough();P.check_freeze()
    print('All four canonical GPUs idle. Starting fixed non-AIME deployment probe.',flush=True)
    with (ROOT/'logs/deployment_smoke.log').open('x') as log:
        subprocess.run([PY,str(ROOT/'scripts/canonical3090_pipeline.py'),'--phase','smoke'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
    # One formal condition at a time. Four fresh workers, static inherited assignments. No automatic rerun.
    for arm in P.ARMS:
        while not idle():time.sleep(60)
        enough();P.check_freeze();workers=[]
        for worker in range(4):
            path=ROOT/f'logs/{arm}_worker{worker}.exitcode'
            if path.exists():raise RuntimeError('existing worker evidence; no automatic restart')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(worker),PYTHONUNBUFFERED='1')
            log=(ROOT/f'logs/{arm}_worker{worker}.log').open('x')
            process=subprocess.Popen([PY,str(ROOT/'scripts/canonical3090_endpoint.py'),'--arm',arm,'--worker',str(worker)],env=env,stdout=log,stderr=subprocess.STDOUT)
            workers.append((worker,process,log))
        # Do not kill other workers on a failure; let currently running units preserve their natural evidence.
        codes={}
        for worker,process,log in workers:
            code=process.wait();log.close();codes[str(worker)]=code
            (ROOT/f'logs/{arm}_worker{worker}.exitcode').write_text(str(code)+'\n')
        P.save(ROOT/f'logs/{arm}_condition_exit.json',{'exitcodes':codes,'status':'PASS' if set(codes.values())=={0} else 'FAIL'})
        if set(codes.values())!={0}:raise RuntimeError('formal condition failed; no downstream condition or semantic retry')
    with (ROOT/'logs/endpoint_finish.log').open('x') as log:
        subprocess.run([PY,str(ROOT/'scripts/canonical3090_pipeline.py'),'--phase','finish'],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['sha256sum','-c',str(ROOT/'SHA256SUMS')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    # Additive readback copy to training host; no deletion or overwrite of source scientific evidence.
    for name in ('aime26','deployment','analysis'):
        target=A800+'/canonical3090_endpoint/'+name+'/'
        subprocess.run(['rsync','-a','--partial','--bwlimit=20000','-e',RSYNC_SSH,str(ROOT/name)+'/',HOST+':'+target],check=True)
    for name in ('FINAL_REPORT.md','NEXT_EXPERIMENT_DECISION.md','SHA256SUMS'):
        subprocess.run(['rsync','-a','-e',RSYNC_SSH,str(ROOT/name),HOST+':'+A800+'/canonical3090_endpoint/'+name],check=True)
    P.save(ROOT/'logs/offline_pipeline.completion.json',{'status':'PASS','endpoint_records':180,'scientific_retries':0,'next_experiment_started':False,'human_qualitative_evidence_review_required':True})
    print('COMPLETE: report and frozen outputs available; no follow-on experiment launched.',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        P.save(ROOT/'logs/offline_pipeline.error.json',{'status':'FAIL','error':str(exc),'traceback':traceback.format_exc(),'do_not_retry':True,'no_worker_killed':True});raise
