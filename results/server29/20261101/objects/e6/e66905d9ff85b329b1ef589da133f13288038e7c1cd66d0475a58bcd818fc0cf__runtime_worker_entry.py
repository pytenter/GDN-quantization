"""Delegate unchanged scientific worker, then attest its actual runtime profile."""
from pathlib import Path
import json
import os
import runpy
import sys
import time
import traceback
from recovery_state import atomic_json
from frozen_runtime_profile import validate_environment,validate_loaded_profile

def main():
    script=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]);args=sys.argv[3:]
    assert args[0]=='--';args=args[1:]
    root=Path(__file__).resolve().parent
    assert script.parent==root and script.name in ['functional_worker.py','formal_worker.py']
    d=Path(os.environ['LING_PROSPECTIVE_RESULTS'])
    try:
        assert not (d/'STOP.json').exists()
        validate_environment()
        assert sys.flags.hash_randomization==0,'HASH_SEED_NOT_EFFECTIVE_AT_INTERPRETER_START'
        sys.argv=[str(script),*args]
        runpy.run_path(str(script),run_name='__main__')
        profile=validate_loaded_profile(require_forward=True)
        profile.update(worker_script=str(script),pid=os.getpid(),epoch=time.time(),
                       delegation='unchanged original scientific worker; profile inspected after completion')
        atomic_json(output,profile)
    except BaseException as e:
        if not (d/'STOP.json').exists():
            atomic_json(d/'STOP.json',dict(status='STOP',stage='FROZEN_RUNTIME_PROFILE_WORKER_ENVELOPE',
                       reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        raise

if __name__=='__main__':main()
