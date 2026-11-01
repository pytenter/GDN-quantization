"""One serial state machine; numerical worker code remains qualified originals."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from recovery_state import atomic_json,lock,sha,complete
from gpu_admission import admit,proc_identity,route
from prospective_common import entry_for
import prospective_pipeline as p

D=p.OUT;W=p.WT

def event(state,**fields):
    row=dict(state=state,epoch=time.time(),execution=str(D),**fields)
    with (D/'STATE_EVENTS.jsonl').open('a') as f:
        __import__('fcntl').flock(f,__import__('fcntl').LOCK_EX)
        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    name='CURRENT_STATE.json' if not os.environ.get('LING_EXECUTION_LANE') else 'CURRENT_STATE_'+os.environ['LING_EXECUTION_LANE']+'.json'
    atomic_json(D/name,row,replace=True)
    print(json.dumps(row),flush=True)

def persist(path,value):atomic_json(path,value)

def runner_live(base):
    f=Path(str(base)+'.RUNNER.json')
    if not f.exists():return False
    identity=p.read(f)['identity']
    try:return proc_identity(identity['pid'])==identity
    except (OSError,ValueError):return False

def run(document,condition=None,stage='screening'):
    cfg=p.verify();freeze=p.read(D/'SOURCE_CODE_FREEZE.json')
    if stage=='parent':
        from qualified_completion import reuse_parent
        reused=reuse_parent(document,condition)
        if reused is not None:return reused
    folder=D/('screening' if stage=='screening' else 'formal' if stage=='formal' else 'runs');folder.mkdir(exist_ok=True)
    tag=document if stage=='screening' else f'{document}__{condition}__{stage}'
    base=folder/tag;result=Path(str(base)+'.json');receipt=Path(str(base)+'.NATURAL_EXIT.json')
    identity=dict(document=document,rotation=condition,stage=stage,source_commit=freeze['source_commit'],
        config_sha256=freeze['config_sha256'],token_sha256=entry_for(document)['token_file_sha256'],
        contract_sha256=sha(D/'preregistration/DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json'),execution=str(D),target_gpu_uuid=cfg['selected_gpu_uuid'])
    if result.exists() and receipt.exists():
        r=complete(result,receipt,identity);event('SUCCESSFUL_UNIT_REUSED',tag=tag);return r
    assert not Path(str(base)+'.FAILED_EXIT.json').exists(),'FIRST_FAILED_UNIT_NO_RETRY'
    intent=Path(str(base)+'.INTENT.json')
    if intent.exists():
        assert p.read(intent)['unit_identity']==identity,'INTENT_IDENTITY_CHANGED'
        if result.exists() and receipt.exists():return complete(result,receipt,identity)
        assert runner_live(base),'INCOMPLETE_UNIT_NO_LIVE_RUNNER_NO_AUTOMATIC_RETRY'
        event('EXISTING_RUNNER_REATTACHED',tag=tag)
    else:
        assert not result.exists() and not Path(str(base)+'.STARTED.json').exists(),'ORPHAN_UNIT_NOT_RELAUNCHED'
        admission_path=D/'admission';admission_path.mkdir(exist_ok=True)
        admit(cfg,admission_path/(tag+'.window.jsonl'))
        gpu=route(cfg);assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu),'CUDA_UUID_ROUTING_CHANGED'
        if stage!='screening':
            s=p.read(D/'FROZEN_4DOC_SELECTION.json');assert sha(D/'FROZEN_4DOC_SELECTION.json')==p.read(D/'SELECTION_FREEZE_RECEIPT.json')['sha256']
            assert any(d['document']==document and condition in ['FP',d['P1_predicted_rotation'],d['L7_predicted_rotation']] for d in s['documents'])
        command=[sys.executable,'-B','-u',str(W/('short_worker.py' if stage=='screening' else 'formal_worker.py' if stage=='formal' else 'functional_worker.py')),'--document',document]
        if stage!='screening':command+=['--condition',condition]
        if stage not in ['screening','formal']:command+=['--stage',stage]
        persist(intent,dict(command=command,base=str(base),unit_identity=identity,source_commit=freeze['source_commit'],tag=tag,stage=stage))
        subprocess.Popen([sys.executable,'-B','-u',str(W/'unit_runner.py'),str(intent)],start_new_session=True)
        for _ in range(20):
            if Path(str(base)+'.RUNNER.json').exists():break
            time.sleep(.1)
        assert runner_live(base),'RUNNER_NOT_STARTED'
        event('WORKER_LAUNCHED',tag=tag)
    last_position=-1;last_progress=time.monotonic()
    while not receipt.exists():
        if Path(str(base)+'.FAILED_EXIT.json').exists():raise RuntimeError(('WORKER_FAILED',tag))
        if not runner_live(base):
            if receipt.exists():break
            raise RuntimeError('RUNNER_LOST_NO_EXIT_RECEIPT')
        log=Path(str(base)+'.log')
        if log.exists():
            for line in reversed(log.read_text().splitlines()):
                try:r=json.loads(line)
                except (ValueError,TypeError):continue
                if isinstance(r,dict) and 'position' in r:
                    value=r['position'];assert value>=last_position,'PROGRESS_REGRESSED'
                    if value>last_position:last_position=value;last_progress=time.monotonic()
                    break
        if time.monotonic()-last_progress>900:
            persist(D/'STOP.json',dict(status='STOP',stage=stage,reason='CONFIRMED_NO_PROGRESS',tag=tag,epoch=time.time(),no_retry=True))
            # Worker consumes STOP at next safe step. Never kill unrelated processes.
            raise RuntimeError('CONFIRMED_NO_PROGRESS')
        time.sleep(2)
    r=complete(result,receipt,identity)
    event('WORKER_NATURAL_EXIT_PASS',tag=tag,stage=stage)
    return r

def frozen_selection():
    f=D/'FROZEN_4DOC_SELECTION.json'
    if not f.exists():return p.select()
    s=p.read(f);ds=sorted((p.read(x) for x in (D/'screening').glob('*.SELECTION_DECISION.json')),key=lambda x:x['scan_index'])
    assert s['documents']==[d for d in ds if d['eligible_pairs']][:4] and len(s['documents'])==4
    receipt=D/'SELECTION_FREEZE_RECEIPT.json'
    if not receipt.exists():
        assert not list((D/'runs').glob('*.STARTED.json')) and not list((D/'formal').glob('*.STARTED.json'))
        persist(receipt,dict(sha256=sha(f),epoch=time.time(),long_GPU_workers_launched=0,long_science_units_launched=0,recovered_partial_freeze=True))
    assert sha(f)==p.read(receipt)['sha256'];return s

def main():
    from qualified_completion import run_main
    run_main()

def atomic_text(path,s):
    from tempfile import mkstemp
    path=Path(path)
    if path.exists():
        assert path.read_text()==s,'IMMUTABLE_TEXT_CONFLICT';return
    fd,tmp=mkstemp(dir=path.parent,suffix='.partial')
    with os.fdopen(fd,'w') as f:f.write(s);f.flush();os.fsync(f.fileno())
    os.link(tmp,path);os.unlink(tmp)

if __name__=='__main__':
    try:
        main()
    except BlockingIOError:
        print('EXISTING_CONTROLLER_LOCK_NO_DUPLICATE');sys.exit(0)
    except BaseException as e:
        if not (D/'STOP.json').exists():persist(D/'STOP.json',dict(status='STOP',stage='RECOVERY_CONTROLLER',reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        event('FAIL_CLOSED_STOP',reason=repr(e))
    if os.environ.get('LING_EXECUTION_LANE'):
        sys.exit(1 if (D/'STOP.json').exists() else 0)
    # One controller owns CPU verification/delivery after every scientific worker exits.
    if (D/'PROSPECTIVE_COMPLETE.json').exists() or (D/'STOP.json').exists():
        if (D/'RESULT_AND_DELIVERY_HASHES.json').exists():
            m=p.read(D/'RESULT_AND_DELIVERY_HASHES.json')
            for n,v in m['files'].items():assert sha(D/n)==v['sha256'],n
            print('ALREADY_SEALED_VERIFIED_NO_ADDITIONAL_WORK')
        else:
            for launch in D.rglob('*.LAUNCH.json'):
                ident=p.read(launch)['worker']
                for _ in range(30):
                    try:live=proc_identity(ident['pid'])==ident
                    except (OSError,ValueError):live=False
                    if not live:break
                    time.sleep(2)
                assert not live,'GPU_WORKER_STILL_ACTIVE_DO_NOT_SEAL'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            r=subprocess.run([sys.executable,'-B','-u',str(W/'finalize_cpu.py')],env=env)
            if r.returncode:raise RuntimeError('CPU_DELIVERY_ERROR_PRESERVE_SCIENCE_AND_DIAGNOSE')
