"""CPU-only identity, atomic output and fail-closed recovery decisions."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic_json(p,value,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if p.exists() and not replace:
        assert p.read_bytes()==data,('IMMUTABLE_OUTPUT_CONFLICT',str(p));return
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',suffix='.partial',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if replace:os.replace(tmp,p)
        else:os.link(tmp,p);os.unlink(tmp)
        d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def lock(p):
    f=open(p,'a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:f.close();raise
    return f

def complete(result,receipt,identity=None):
    r=json.loads(Path(result).read_text());e=json.loads(Path(receipt).read_text())
    assert r['status']=='PASS' and e['natural_exit'] and e['result_sha256']==sha(result),'CORRUPT_COMPLETED_UNIT'
    if identity is not None:assert e['unit_identity']==identity,'UNIT_IDENTITY_CHANGED'
    return r

def recovery_action(result_exists,receipt_exists,live_runner,checkpoint_valid,scientific_failure=False):
    if scientific_failure:return 'STOP_FIRST_SCIENTIFIC_FAILURE'
    if result_exists and receipt_exists:return 'VERIFY_AND_REUSE'
    if live_runner:return 'ATTACH_EXISTING_RUNNER'
    if result_exists:return 'STOP_UNOBSERVED_EXIT'
    if checkpoint_valid:return 'RESTORE_ONLY_UNDER_FROZEN_8192_CONTRACT'
    return 'STOP_INCOMPLETE_NO_QUALIFIED_CHECKPOINT'

def validate_positions(rows):
    assert len(rows)==384 and [r['position'] for r in rows]==[8192+64*k for k in range(384)]
    import math
    assert all(math.isfinite(r['kl']) for r in rows)
    return sum(r['kl'] for r in rows)/384
