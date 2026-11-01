"""Append-only stage supervision, no retries, no outcome reads or formal launch.

Stored in results, outside the frozen measurement worktree. Calls only the V2
controller, waits for each child to EXIT, then executes preregistered fresh-1.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

CODE=Path('/data/zypan/worktrees/ling-kda-p1-l7-full-persistent-prospective-v2')
sys.path.insert(0,str(CODE))
from v2_common import OUT,RUNS,LATCH,MODES,checked,emit,read,tag,stop,ledger,digest
PYTHON='/data/zypan/envs/ling-kda/bin/python'
DOCS=[('pg19-test-860',6),('pg19-test-5956',3)]

def command(*args):
    subprocess.run([PYTHON,'-B',str(CODE/'v2_controller.py'),*args],check=True)

def wait(mode,rep):
    while True:
        if LATCH.exists(): raise RuntimeError('V2_EXISTING_STOP_NO_RETRY')
        done=True
        for doc,gpu in DOCS:
            t=tag(mode,doc,gpu,rep); started=RUNS/(t+'.STARTED.json'); p=RUNS/(t+'.json')
            receipt=checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))
            if not started.exists():
                assert time.time()-receipt['epoch']<120,'WORKER_DID_NOT_START'
                done=False; continue
            st=checked(started); alive=Path(f"/proc/{st['pid']}").exists()
            if alive:
                log=RUNS/(t+'.log')
                assert log.exists() and time.time()-log.stat().st_mtime<receipt['no_progress_seconds']+120,'NO_PROGRESS_LOG_WATCHDOG'
                done=False
            elif not p.exists():
                raise RuntimeError(f'WORKER_EXIT_WITHOUT_COMPLETE_OR_STOP:{t}')
            else:
                result=checked(p)
                assert result['status']=='PASS' and result['endpoint']==MODES[mode]['endpoint']
        if done: return
        time.sleep(30)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    emit(OUT/'QUALIFICATION_MICRO_SUPERVISOR_STARTED.json',dict(status='STARTED',epoch=time.time(),pid=os.getpid(),
        code_sha256=digest(__file__),scope='preplanned fresh1 after fresh0; both exact quals before micro; NEVER formal',
        retry=False,scientific_outcome_access=False))
    try:
        wait('qualification',0)
        for doc,gpu in DOCS: command('launch','--gpu',str(gpu),'--mode','qualification','--repeat','1')
        wait('qualification',1)
        command('aggregate')
        print(json.dumps(dict(event='BOTH_GPU_QUALIFICATION_PASS',epoch=time.time())),flush=True)
        for doc,gpu in DOCS: command('launch','--gpu',str(gpu),'--mode','micro')
        print(json.dumps(dict(event='BOTH_8448_MICRO_LAUNCHED',epoch=time.time())),flush=True)
        wait('micro',0)
        emit(OUT/'MICRO_COMPLETE_PENDING_PHYSICAL_RESOURCE_GATE.json',dict(status='COMPLETE',epoch=time.time(),
            formal_launched=False,near_zero_rule_requires_resolution_before_formal=True,ledger=ledger()))
        print(json.dumps(dict(event='BOTH_MICRO_COMPLETE_RESOURCE_GATE_PENDING',epoch=time.time())),flush=True)
    except Exception as e:
        stop(repr(e),'qualification_micro_supervisor',traceback=traceback.format_exc())
        emit(OUT/'QUALIFICATION_MICRO_SUPERVISOR_STOP.json',dict(status='STOP',epoch=time.time(),
            error=repr(e),traceback=traceback.format_exc(),no_retries=True,no_signals=True))
        raise

if __name__=='__main__': main()
