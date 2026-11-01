"""Exactly two predeclared variants of the same engineering checkpoint probe."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from recovery_state import atomic_json, lock
from execution04_launcher import worker_environment

def main():
    root=Path(sys.argv[1]);w=Path(__file__).resolve().parent
    held=lock(root/'phase02_controller.lock')
    plan=json.loads((root/'PHASE02_CAUSAL_PLAN.json').read_text())
    try:
        for name in plan['ordered_variants']:
            d=root/name
            assert not (d/'STOP.json').exists() and not (d/'process_state_probe.INTENT.json').exists()
            env=worker_environment(os.environ,0,'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae',3)
            env['LING_PROSPECTIVE_RESULTS']=str(d)
            env['FLA_CACHE_MODE']='disabled' if name=='cross_process_auto' else 'default'
            env['FLA_CONFIG_DIR']=str(root/'historical_conv_only')
            atomic_json(root/'PHASE02_STATE.json',dict(state='VARIANT_RUNNING',variant=name,epoch=time.time()),replace=True)
            rc=subprocess.run([sys.executable,'-B','-u',str(w/'process_state_probe.py')],env=env).returncode
            assert rc==0,('VARIANT_FAILED_NO_RETRY',name,rc)
        atomic_json(root/'PHASE02_COMPLETE.json',dict(status='TWO_FINITE_VARIANTS_COLLECTED_NOT_SCIENCE',epoch=time.time(),numerical_qualification=False,formal_science_started=False))
    finally:held.close()

if __name__=='__main__':main()
