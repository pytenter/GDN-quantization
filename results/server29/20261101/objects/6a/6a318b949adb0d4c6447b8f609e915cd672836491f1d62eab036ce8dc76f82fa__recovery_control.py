"""Append-only engineering recovery authority; never erases the first STOP."""
from common import *

def active_stop():
    if (ROOT/'run/SCIENTIFIC_STOP.json').exists():return True
    first=ROOT/'run/STOP.json'
    if not first.exists():return False
    recovery=ROOT/'provenance/PREMEASUREMENT_ENGINEERING_RECOVERY_ENVELOPE_V1.json'
    if not recovery.exists():return True
    r=load(recovery)
    attempt=r['CURRENT_SUCCESSFUL_PREPARATION_ATTEMPT']
    complete=ROOT/f'provenance/preparation_attempts/attempt{attempt}/CPU_PREPARATION_COMPLETE.json'
    if not complete.exists():return True
    assert sha(first)==sha(ROOT/'provenance/STOP_PREPARATION.json')==r['first_stop_sha256']
    assert load(first)['stage']=='PREPARATION' and load(first)['error']=="KeyError('gate')"
    assert r[f'V3_PREPARATION_ATTEMPT_{attempt}']=='PASS' and load(complete)['gate']=='PASS'
    return False

def scientific_stop(stage,exc):
    p=ROOT/'run/SCIENTIFIC_STOP.json'
    if not p.exists():emit(p,dict(gate='FAIL',stage=stage,error=repr(exc),time_unix=time.time(),no_retry=True,
        no_runtime_or_numerical_patch=True,first_STOP_preserved=True,first_STOP_sha256=sha(ROOT/'provenance/STOP_PREPARATION.json')))
