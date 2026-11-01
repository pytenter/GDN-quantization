"""Once-only CPU owner for bounded FP collection and actual exit/release receipt."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def main(root, gpu):
    sys.path.insert(0,str(root/'src'))
    sys.path.insert(0,str(root/'calibration'))
    import execution_control as C
    from collect_fp_once import validate_calibration_permit
    validate_calibration_permit(root,gpu,C)
    C.device(gpu)
    target=root/'calibration'
    C.atomic_new(target/'PARENT_STARTED.json',dict(identity=C.process_identity(os.getpid()),
        source_freeze_sha256=C.sha(root/'CALIBRATION_SOURCE_FREEZE.json'),monotonic=time.monotonic()))
    child=None;identity=None;requested=[];reason=None
    def stop(signum,frame):requested.append(signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        log=(target/'FP_COLLECTION.log').open('xb')
        argv=[sys.executable,'-u','-B',str(root/'calibration/collect_fp_once.py'),
              '--root',str(root),'--gpu',str(gpu)]
        child=subprocess.Popen(argv,env=C.worker_environment(gpu),stdout=log,stderr=subprocess.STDOUT)
        identity=C.process_identity(child.pid)
        C.atomic_new(target/'PARENT_DISPATCH.json',dict(identity=identity,argv=argv,
            parent_identity=C.process_identity(os.getpid()),monotonic=time.monotonic()))
        last_progress=None;last_change=time.monotonic()
        while child.poll() is None:
            if not C.same_process(identity):
                if child.poll() is not None:break
                raise RuntimeError('Owned collector identity changed while Popen active')
            p=target/'execution/PROGRESS.json'
            progress=C.sha(p) if p.exists() else None
            if progress!=last_progress:last_progress=progress;last_change=time.monotonic()
            obs=C.query_gpu()
            if any(a['uuid']==C.UUIDS[gpu] and a['pid']!=child.pid for a in obs['apps']):
                reason='Foreign process on assigned GPU; no foreign signals';break
            if requested:reason='Explicit external parent STOP signal';break
            if time.monotonic()-last_change>1800:reason='Actual phase/token no progress for1800seconds';break
            time.sleep(2)
        if reason and child.poll() is None:
            if not C.same_process(identity):raise RuntimeError('Refuse signal after owned identity mismatch')
            child.terminate()
        code=child.wait();log.close()
        end=time.monotonic();released=False;observations=[]
        while time.monotonic()-end<=120:
            obs=C.query_gpu();live=C.same_process(identity)
            owned=[a for a in obs['apps'] if a['uuid']==C.UUIDS[gpu] and a['pid']==identity['pid']]
            observations.append(dict(monotonic=time.monotonic(),exact_identity_alive=live,owned_NVML=owned))
            if not live and not owned:released=True;break
            time.sleep(2)
        C.atomic_new(target/'PARENT_RESULT.json',dict(exit_returncode=code,release='PASS' if released else 'FAIL',
            identity=identity,stop_reason=reason,observations=observations,monotonic=time.monotonic()))
        if code!=0 or not released or reason:raise RuntimeError('Bounded collection actual exit/release failed')
        manifest=C.read(target/'FP_COLLECTION_MANIFEST.json')
        if manifest['identity']!=identity:raise RuntimeError('Collector manifest exact identity differs')
        C.atomic_new(target/'FP_COLLECTION_EXIT_RELEASE.json',dict(status='PASS_ACTUAL_EXIT0_AND_RELEASE',
            exit_returncode=0,release='PASS',worker_identity=identity,
            collection_manifest_sha256=C.sha(target/'FP_COLLECTION_MANIFEST.json'),
            observations=observations,monotonic=time.monotonic()))
    except BaseException as error:
        cleanup=dict(attempted=False,actual_exit=None,release='UNRESOLVED')
        if child is not None:
            cleanup['attempted']=True
            if child.poll() is None and identity is not None and C.same_process(identity):
                child.terminate();cleanup['signal']='SIGTERM_EXACT_OWNED_POPEN'
            try:
                cleanup['actual_exit']=child.wait(timeout=120)
                observations=[];deadline=time.monotonic()+120
                while True:
                    obs=C.query_gpu();owned=[a for a in obs['apps'] if a['uuid']==C.UUIDS[gpu] and a['pid']==identity['pid']]
                    live=C.same_process(identity);observations.append(dict(monotonic=time.monotonic(),exact_identity_alive=live,owned_NVML=owned))
                    if not live and not owned:cleanup['release']='PASS';break
                    if time.monotonic()>=deadline:break
                    time.sleep(2)
                cleanup['observations']=observations
            except BaseException as cleanup_error:cleanup['error']=repr(cleanup_error)
        C.atomic_new(target/'PARENT_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),
            preserve_all=True,no_automatic_retry=True,identity=identity,stop_reason=reason,cleanup=cleanup))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args();main(a.root.resolve(),a.gpu)
