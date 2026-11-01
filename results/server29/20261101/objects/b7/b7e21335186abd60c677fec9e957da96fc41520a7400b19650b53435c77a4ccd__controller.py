"""Conditional frozen execution to terminal report, with no automatic AIME/training."""
from common import *
from worker import frozen_guard
from surrogate_statistics import analyze
import signal,traceback

def phase(name):
    processes=[];handles=[]
    try:
        for gpu in [0,1]:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            logfile=D/'logs'/f'{name}_GPU{gpu}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
            log=logfile.open('x');handles.append(log)
            p=subprocess.Popen([str(PYTHON),str(D/'scripts/worker.py'),'--gpu',str(gpu),'--phase',name],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            processes.append(p)
        emit('run/'+name+'_WORKERS.json',dict(pids=[p.pid for p in processes],physical_gpus=[0,1],phase=name))
        while True:
            status=[p.poll() for p in processes]
            if any(v is not None and v!=0 for v in status):raise RuntimeError(('WORKER_FAILURE',name,status))
            if all(v==0 for v in status):break
            time.sleep(1)
    finally:
        # Only directly created worker process groups are terminated on failure.
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for h in handles:h.close()
    return analyze(name)

def main():
    receipt=frozen_guard();runtime();hardware()
    assert not (D/'run/DISCOVERY_STARTED.json').exists()
    assert not list((D/'raw').rglob('*.json'))
    assert command(['git','status','--porcelain'])==''
    parent=verify_parent()
    emit('run/DISCOVERY_STARTED.json',dict(commit=receipt['commit'],parent_validation=parent,time_unix=time.time(),no_training=True,no_AIME=True))
    phase('DISCOVERY')
    selected=load(D/'analysis/discovery_selection.json')
    if selected['DISCOVERY_SURROGATE_GATE']=='PASS':
        frozen_guard()
        assert not (D/'run/CONFIRMATION_STARTED.json').exists()
        emit('run/CONFIRMATION_STARTED.json',dict(commit=receipt['commit'],selection=row(D/'analysis/discovery_selection.json'),time_unix=time.time()))
        phase('CONFIRMATION')
    subprocess.run([str(PYTHON),str(D/'scripts/finalize.py')],cwd=R,check=True)
    subprocess.run(['git','diff','--check'],cwd=R,check=True)
    subprocess.run(['git','add','--',str(D.relative_to(R))],cwd=R,check=True)
    subprocess.run(['git','commit','-m','experiment: finalize prospective propagation surrogate validation'],cwd=R,check=True)
    emit('run/final_commit_receipt.json',dict(commit=command(['git','rev-parse','HEAD']),worktree=command(['git','status','--porcelain']),FINAL_ACTION='STOP',push_performed=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (D/'run/FORMAL_STOP.json').exists():emit('run/FORMAL_STOP.json',dict(gate='FAIL',error=repr(e),traceback=traceback.format_exc(),no_auto_retry=True,NEW_TRAINING='NO',NEW_AIME_GENERATION=0,FINAL_ACTION='STOP'))
        raise
