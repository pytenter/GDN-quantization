"""Run the four frozen students and two references serially after all gates."""
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import functional_worker as w

D = Path(__file__).resolve().parent


def main():
    assert not (D/'STOP.json').exists()
    while not (D/'QUALIFICATION_GATES_COMPLETED.json').exists():
        assert not (D/'STOP.json').exists(), 'QUALIFICATION_STOP_NO_FORMAL'
        time.sleep(5)
    gate = w.read(D/'QUALIFICATION_GATES_COMPLETED.json')
    assert gate['status'] == 'ALL_EIGHT_GATES_PASS'
    assert w.read(D/'SERIAL_OBSERVER_CPU_CHECK.json')['status'] == 'PASS_CPU_ONLY'
    assert w.read(D/'SECONDARY_SPEC.json')['status'] == 'FROZEN_BY_USER_CLARIFICATION_BEFORE_NEW_SCIENCE'
    spec = w.read(D/'FROZEN_SPEC.json')
    root = D/'formal'; root.mkdir(exist_ok=True)
    w.publish(D/'FORMAL_STARTED.json',dict(started_epoch=time.time(),single_GPU6_serial=True,
        frozen_spec_sha256=w.sha(D/'FROZEN_SPEC.json'),formal_code_freeze_sha256=w.sha(D/'FORMAL_CODE_FREEZE.json'),
        secondary_spec_sha256=w.sha(D/'SECONDARY_SPEC.json'),no_training=True))
    for entry in spec['documents']:
        for condition in ('FP',entry['P1_predicted_winner'],entry['L7_predicted_winner']):
            assert not (D/'STOP.json').exists()
            tag=f"{entry['document']}__{condition}__formal"
            assert not (root/(tag+'.STARTED.json')).exists(), 'ALREADY_LAUNCHED_NO_RETRY'
            with (root/(tag+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,'-B','-u',str(D/'formal_worker.py'),
                    '--document',entry['document'],'--condition',condition],stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode==0, ('FORMAL_WORKER_STOP',tag,result.returncode)
            path=root/(tag+'.json'); v=w.read(path)
            assert v['status']=='PASS' and v['endpoint']==32768
            w.publish(root/(tag+'.NATURAL_EXIT.json'),dict(epoch=time.time(),natural_exit=True,result_sha256=w.sha(path)))
            # Status exposes completion/resource counts; it makes no adaptive
            # launch choices from the first document's results.
            print('FORMAL_TRAJECTORY_COMPLETE',tag,flush=True)
    w.publish(D/'FORMAL_COMPLETE.json',dict(status='PASS',completed_epoch=time.time(),
        students=4,FP_references=2,no_training=True))
    print('ALL_FROZEN_FORMAL_TRAJECTORIES_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as e:
        if not (D/'STOP.json').exists():
            w.publish(D/'STOP.json',dict(status='STOP',stage='formal_pipeline',reason=repr(e),
                traceback=traceback.format_exc(),stopped_epoch=time.time(),no_retry=True,
                science_verdict='INCOMPLETE_NOT_EVALUATED'))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(D/'seal_status.py')],check=False)
