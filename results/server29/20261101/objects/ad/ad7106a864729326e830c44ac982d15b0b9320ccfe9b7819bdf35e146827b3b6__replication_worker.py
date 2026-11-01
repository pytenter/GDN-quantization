"""I/O/roster adapter for the unchanged, qualified production numerical engine."""
import os, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OLD = Path('/data/zypan/worktrees/qwen-gdn-scale-feedback-fullmodel-functional-causal-pilot-v1/experiments/QWEN_GDN_SCALE_FEEDBACK_FULLMODEL_FUNCTIONAL_CAUSAL_PILOT_V1')
sys.path.insert(0,str(OLD/'scripts'))
import pilot_engineering as P
import anchor_qualification as A
import formal_worker as W
import observer_reducer_qualification as O
P.ROOT=ROOT; P.REPO=REPO; P.TASK='QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1'
W.ARMS=['PARENT','NO_SCALE_INFLATION']

def main():
    action=sys.argv[1]
    if action=='observer':O.run(int(sys.argv[2]))
    elif action=='resume':A.resume()
    elif action=='reference':W.reference(int(sys.argv[2]))
    elif action=='warmup':W.warmup(int(sys.argv[2]))
    elif action=='branch':W.branch(int(sys.argv[2]),sys.argv[3])
    else:raise ValueError(action)

if __name__=='__main__':
    started=time.monotonic()
    try:main()
    except Exception as e:
        if not (ROOT/'FIRST_FAILURE_GATE.json').exists():
            P.save('FIRST_FAILURE_GATE.json',{'status':'FAIL_AND_STOP','stage':'INCREMENTAL_OR_FORMAL_WORKER','arguments':sys.argv[1:],
                'exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'no_retry':True,
                'unit_wall_seconds':time.monotonic()-started,'unit_GPUh_including_initialization':(time.monotonic()-started)/3600})
        raise
