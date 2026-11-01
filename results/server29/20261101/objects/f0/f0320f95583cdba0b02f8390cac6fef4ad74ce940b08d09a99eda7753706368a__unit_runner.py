"""Independent worker wait/exit receipt survives controller interruption."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from recovery_state import atomic_json,lock,sha
from gpu_admission import proc_identity

def main():
    intent=Path(sys.argv[1]);v=json.loads(intent.read_text());base=Path(v['base'])
    held=lock(str(base)+'.unit.lock')
    assert not Path(str(base)+'.NATURAL_EXIT.json').exists()
    atomic_json(str(base)+'.RUNNER.json',dict(identity=proc_identity(os.getpid()),intent_sha256=sha(intent)))
    start=time.time()
    with Path(str(base)+'.log').open('x') as log:
        child=subprocess.Popen(v['command'],stdout=log,stderr=subprocess.STDOUT)
        atomic_json(str(base)+'.LAUNCH.json',dict(worker=proc_identity(child.pid),started_epoch=start,command=v['command'],unit_identity=v['unit_identity']))
        rc=child.wait();end=time.time()
    if rc==0:
        r=json.loads(Path(str(base)+'.json').read_text());assert r['status']=='PASS'
        atomic_json(str(base)+'.NATURAL_EXIT.json',dict(natural_exit=True,started_epoch=start,completed_epoch=end,
            wall_seconds=end-start,result_sha256=sha(Path(str(base)+'.json')),unit_identity=v['unit_identity'],
            source_commit=v['source_commit'],tag=v['tag'],stage=v['stage']))
    else:
        atomic_json(str(base)+'.FAILED_EXIT.json',dict(started_epoch=start,completed_epoch=end,wall_seconds=end-start,
            returncode=rc,unit_identity=v['unit_identity'],source_commit=v['source_commit'],tag=v['tag'],stage=v['stage']))
    held.close()
    return rc

if __name__=='__main__':sys.exit(main())
