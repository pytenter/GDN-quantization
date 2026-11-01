"""Independent exit receipt and frozen-runtime launch envelope; no model math."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from recovery_state import atomic_json,lock,sha
from gpu_admission import proc_identity

def main():
    intent=Path(sys.argv[1]);v=json.loads(intent.read_text());base=Path(v['base'])
    held=lock(str(base)+'.unit.lock');child=None;start=end=None;worker_rc=None
    real_gpu=v['unit_identity'].get('target_gpu_uuid','').startswith('GPU-')
    output=Path(str(base)+'.RUNTIME_PROFILE.json');command=v['command'];delegated=False
    try:
        assert not Path(str(base)+'.NATURAL_EXIT.json').exists()
        atomic_json(str(base)+'.RUNNER.json',dict(identity=proc_identity(os.getpid()),intent_sha256=sha(intent)))
        if real_gpu:
            from frozen_runtime_profile import validate_environment,ENVIRONMENT_KEYS
            profile=validate_environment()
            locations=[i for i,arg in enumerate(command) if Path(arg).name in ['functional_worker.py','formal_worker.py']]
            if locations:
                assert len(locations)==1
                i=locations[0];delegated=True
                command=[command[0],'-B','-u',str(Path(__file__).resolve().parent/'runtime_worker_entry.py'),command[i],str(output),'--',*command[i+1:]]
        start=time.time()
        with Path(str(base)+'.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            identity=proc_identity(child.pid)
            atomic_json(str(base)+'.LAUNCH.json',dict(worker=identity,started_epoch=start,command=command,
                        delegated_original_command=v['command'] if delegated else None,unit_identity=v['unit_identity']))
            if real_gpu:
                raw=Path('/proc')/str(child.pid)/'environ'
                values=dict(x.split('=',1) for x in raw.read_bytes().decode(errors='surrogateescape').split('\0') if '=' in x)
                observed={k:values.get(k) for k in ENVIRONMENT_KEYS}
                assert observed==profile['environment'],'ACTUAL_WORKER_START_ENVIRONMENT_MISMATCH'
                atomic_json(str(base)+'.WORKER_ENVIRONMENT.json',dict(status='PASS_ACTUAL_PROC_START_ENVIRONMENT',
                            worker=identity,environment=observed,fixed_FLA_config_SHA256=profile['fixed_FLA_config_SHA256']))
            worker_rc=child.wait();end=time.time()
        if worker_rc==0:
            r=json.loads(Path(str(base)+'.json').read_text());assert r['status']=='PASS'
            extras={}
            if real_gpu:
                assert validate_environment()==profile,'FROZEN_PROFILE_CHANGED_DURING_WORKER'
                if delegated:
                    q=json.loads(output.read_text());assert q['status']=='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE'
                    assert q['environment']==profile['environment'] and q['forward_kernels_executed']
                    extras['runtime_profile_sha256']=sha(output)
                else:
                    assert r['frozen_FLA_profile']['status']=='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE'
                extras['worker_start_environment_sha256']=sha(str(base)+'.WORKER_ENVIRONMENT.json')
            atomic_json(str(base)+'.NATURAL_EXIT.json',dict(natural_exit=True,started_epoch=start,completed_epoch=end,
                        wall_seconds=end-start,result_sha256=sha(Path(str(base)+'.json')),unit_identity=v['unit_identity'],
                        source_commit=v['source_commit'],tag=v['tag'],stage=v['stage'],**extras))
        else:
            atomic_json(str(base)+'.FAILED_EXIT.json',dict(started_epoch=start,completed_epoch=end,wall_seconds=end-start,
                        returncode=worker_rc,worker_returncode=worker_rc,unit_identity=v['unit_identity'],source_commit=v['source_commit'],tag=v['tag'],stage=v['stage']))
        return worker_rc
    except BaseException as e:
        if real_gpu:
            d=Path(os.environ['LING_PROSPECTIVE_RESULTS'])
            if not (d/'STOP.json').exists():atomic_json(d/'STOP.json',dict(status='STOP',stage='FROZEN_RUNTIME_PROFILE_SUPERVISOR',
                        reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        if child is not None and child.poll() is None:
            worker_rc=child.wait(timeout=120);end=time.time()
        if child is not None and not Path(str(base)+'.FAILED_EXIT.json').exists():
            end=time.time() if end is None else end
            atomic_json(str(base)+'.FAILED_EXIT.json',dict(started_epoch=start,completed_epoch=end,wall_seconds=end-start,
                        returncode=1,worker_returncode=worker_rc,unit_verdict='PROFILE_OR_SUPERVISION_FAILURE',
                        reason=repr(e),unit_identity=v['unit_identity'],source_commit=v['source_commit'],tag=v['tag'],stage=v['stage']))
        raise
    finally:held.close()

if __name__=='__main__':sys.exit(main())
