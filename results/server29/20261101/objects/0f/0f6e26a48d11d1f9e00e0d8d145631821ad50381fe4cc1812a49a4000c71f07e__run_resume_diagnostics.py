"""Two predeclared, serial diagnostic GPU workers. No scientific execution."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback
from recovery_state import atomic_json, lock, sha
from execution04_launcher import worker_environment
from prospective_common import verify
from gpu_admission import admit

D=Path(os.environ['LING_PROSPECTIVE_RESULTS'])
W=Path(__file__).resolve().parent
UUID='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'

def main():
    held=lock(D/'diagnostic_controller.lock')
    try:
        cfg=verify();plan=json.loads((D/'BOUNDED_GPU_DIAGNOSTIC_PLAN.json').read_text())
        assert plan['units']==['fresh_branches','parent_and_restore']
        for mode in plan['units']:
            assert not (D/'STOP.json').exists()
            base=D/'diagnostics'/mode;assert not Path(str(base)+'.INTENT.json').exists(),'NO_RETRY'
            atomic_json(D/'DIAGNOSTIC_STATE.json',dict(state='ORIGINAL_60_SECOND_ADMISSION',mode=mode,epoch=time.time()),replace=True)
            admit(cfg,D/'admission'/(mode+'.window.jsonl'))
            ident=dict(document='pg19-test-26183',rotation='FP',stage='engineering_diagnostic',
                       target_gpu_uuid=UUID,execution=str(D),source_commit=plan['source_commit'],
                       numerical_qualification=False,scientific_trajectory_count=0)
            command=[sys.executable,'-B','-u',str(W/'resume_diagnostic.py'),'--mode',mode,'--result',str(base)+'.json']
            intent=Path(str(base)+'.INTENT.json')
            atomic_json(intent,dict(base=str(base),command=command,unit_identity=ident,
                        source_commit=plan['source_commit'],tag=mode,stage='engineering_diagnostic'))
            env=worker_environment(os.environ,0,UUID,3)
            p=subprocess.Popen([sys.executable,'-B','-u',str(W/'unit_runner.py'),str(intent)],env=env)
            atomic_json(D/'DIAGNOSTIC_STATE.json',dict(state='WORKER_RUNNING',mode=mode,
                        supervisor_pid=p.pid,epoch=time.time()),replace=True)
            start=time.monotonic();limit=600 if mode=='fresh_branches' else 2700
            while p.poll() is None:
                if time.monotonic()-start>limit:
                    if not (D/'STOP.json').exists():atomic_json(D/'STOP.json',dict(status='STOP',stage='DIAGNOSTIC_TIMEOUT',mode=mode,epoch=time.time(),no_retry=True))
                    # The owned worker checks this STOP at every safe token.
                    p.wait(timeout=120);raise TimeoutError('BOUNDED_DIAGNOSTIC_LIMIT')
                time.sleep(2)
            assert p.returncode==0,('DIAGNOSTIC_WORKER_FAILED',mode,p.returncode)
            result=json.loads(Path(str(base)+'.json').read_text())
            assert result['numerical_qualification'] is False and result['scientific_trajectory_count']==0
        atomic_json(D/'DIAGNOSTIC_COLLECTION_COMPLETE.json',dict(status='TWO_PREDECLARED_DIAGNOSTICS_COMPLETE',epoch=time.time(),numerical_qualification=False,formal_science_started=False))
        atomic_json(D/'DIAGNOSTIC_STATE.json',dict(state='AWAITING_CPU_EVIDENCE_ANALYSIS',epoch=time.time()),replace=True)
    except BaseException as e:
        if not (D/'STOP.json').exists():atomic_json(D/'STOP.json',dict(status='STOP',stage='BOUNDED_ENGINEERING_DIAGNOSIS',reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        raise
    finally:held.close()

if __name__=='__main__':main()
