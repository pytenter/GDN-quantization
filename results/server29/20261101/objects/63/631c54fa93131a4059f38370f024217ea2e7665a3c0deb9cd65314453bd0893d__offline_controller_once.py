#!/usr/bin/env python3
"""One native finite controller invocation; no retries, workers or permits made here."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode=True

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def new(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);a=p.parse_args();run=a.root.resolve();root=run/'payload'
    state=read(root/'ROOT.json');admission=read(run/'analysis/REMOTE_PRE_GPU_CPU_ADMISSION.json')
    assert state['ALLOW_GPU_RESTORE']=='YES' and admission['status']=='PASS_AUTHORIZED_NEW407_CPU_ADMISSION_NO_GPU'
    for name,key in [('SOURCE_FREEZE.json','source_freeze_sha256'),('HUMAN_GPU_RESTORE_AUTHORIZATION.json','authorization_sha256')]:assert sha(root/name)==admission[key]
    assert not(root/'execution/CONTROLLER_ATTEMPT_STARTED.json').exists() and not(root/'FIRST_FAILURE.json').exists()
    command=[sys.executable,'-B',str(root/'scripts/restoration_controller.py'),'--root',str(root),'--run','--allow-gpu-restore','YES','--authorization',str(root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json')]
    identity=dict(pid=os.getpid(),starttick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
    new(run/'operations/OFFLINE_NATIVE_INVOCATION_STARTED.json',dict(task_id=state['task_id'],command=command,supervisor_identity=identity,source_freeze_sha256=admission['source_freeze_sha256'],authorization_sha256=admission['authorization_sha256'],monotonic=time.monotonic(),retry=False))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with(run/'operations/native_controller.log').open('xb') as log:
        before=time.monotonic();rc=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,env=env);after=time.monotonic()
    new(run/'operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json',dict(command=command,returncode=rc,before_monotonic=before,after_monotonic=after,wall_seconds=after-before,retry=False,finisher_separate_read_only=True))
    print(json.dumps(dict(event='NATIVE_FINITE_CONTROLLER_EXITED',returncode=rc)),flush=True)
    return rc
if __name__=='__main__':sys.exit(main())
