"""Dependency coordinator: finish current B, then qualified A/G, then seal.

Does not restart B, does not alter frozen source, does not read partial scores.
No scheduler or follow-up research; only the user's authorized V3 workflow.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numerical_recovery as N

ROOT=N.ROOT
RUNTIME=Path('/data/zypan/experiment_runs/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3_runtime')


def event(phase,**data):
    path=ROOT/'execution/PIPELINE_EVENTS.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(dict(utc=N.now(),phase=phase,**data),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())


def process_identity(pid):
    path=Path('/proc')/str(pid)/'cmdline'
    if not path.exists():return False
    command=path.read_bytes().replace(b'\0',b' ').decode()
    N.check(str(ROOT/'scripts/scale_science.py')+' controller' in command,'current B process exact identity; no guessed orchestration')
    return True


def main(pid):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    numerical=ROOT/'scripts/finalize_discovery.py'
    if not (ROOT/'MULTITOKEN_FULL_MODEL_PARITY.json').exists():
        subprocess.run([sys.executable,str(numerical),'numerical'],env=env,check=True)
    event('WAIT_EXISTING_TRACK_B',pid=pid,no_restart=True)
    log=RUNTIME/'science_controller_001.log'
    while not (ROOT/'TRACK_B_SHORT_SCORE_FREEZE.json').exists():
        failures=list((ROOT/'execution').glob('SCIENCE_*STOP.json'))
        N.check(not failures,'existing B stopped; preserve failure and do not launch history')
        N.check(process_identity(pid),'B process ended without required complete score seal')
        N.check(time.time()-log.stat().st_mtime<1800,'confirmed no-progress for30min; no new stage')
        time.sleep(15)
    N.check(not N.resources()['processes_csv'],'all existing B GPU workers naturally completed before A')
    history=ROOT/'scripts/matched_history.py'
    subprocess.run([sys.executable,str(history),'prepare'],env=env,check=True)
    tracked=[str((ROOT/x).relative_to(N.REPO)) for x in ['scripts/matched_history.py','scripts/finalize_discovery.py',
        'scripts/continue_pipeline.py','HISTORY_PREMEASUREMENT_SPEC.json','OBSERVER_PARITY_REPORT.json',
        'TRACK_B_ENGINEERING_GATE.json','TRACK_B_SHORT_SCORE_FREEZE.json','TRACK_B_SHORT_SCORES.csv','ORIGINAL_C5_C6_BASELINES.csv',
        'MULTITOKEN_FULL_MODEL_PARITY.json','FRESH_PROCESS_DETERMINISM.json','FIRST_DIVERGENCE_REPORT.json',
        'HISTORICAL_NATIVE_OPERATOR_DESCRIPTIVE_DIFFERENCES.json']]
    subprocess.run(['git','add','--']+tracked,cwd=N.REPO,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Seal qualified Track B and freeze matched persistent history continuation'],cwd=N.REPO,check=True)
    event('TRACK_A_AND_LONG_ENGINEERING_FREEZE',commit=N.cmd(['git','rev-parse','HEAD'],N.REPO),no_old_G_pairing=True)
    with (RUNTIME/'history_controller_001.log').open('x') as loghandle:
        result=subprocess.run([sys.executable,str(history),'controller'],env=env,stdout=loghandle,stderr=subprocess.STDOUT)
    N.check(result.returncode==0,'new matched history controller STOP; no retry')
    event('MATCHED_PANEL_EXECUTION_COMPLETE')
    event('FINAL_SEAL_STARTED',operational_log_outside_scientific_manifest=True)
    subprocess.run([sys.executable,str(numerical),'seal'],env=env,check=True)
    print('V3_COMPLETE_AND_SEALED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--existing-B-pid',type=int,required=True);args=parser.parse_args()
    try:main(args.existing_B_pid)
    except Exception as exc:
        rel='execution/WORKFLOW_FAIL_AND_STOP.json'
        if not (ROOT/rel).exists():N.save(rel,{'utc':N.now(),'status':'FAIL_AND_STOP','message':str(exc),'exception':type(exc).__name__,
            'traceback':traceback.format_exc(),'no_retry':True,'completed_stage_evidence_preserved':True})
        event('FAIL_AND_STOP',message=str(exc))
        if not (ROOT/'FINAL_REPORT.md').exists():
            result=subprocess.run([sys.executable,str(ROOT/'scripts/finalize_discovery.py'),'seal'])
            if result.returncode:print('FINAL_DELIVERY_ENGINEERING_FAILURE_PRESERVED',flush=True)
        raise
