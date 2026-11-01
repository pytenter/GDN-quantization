"""Offline, fail-closed sequential-document/four-GPU-anchor orchestrator."""
import json
import os
import signal
import stat
import subprocess
import time
import traceback
from pathlib import Path
import parallel_stage as B
from parallel_preregister import preregister
C=B.C;S=B.S;PAR=B.PAR;ROOT=C.ROOT;REPO=C.REPO
BUG=ROOT/'amendments/CAPTURE_KEY_SHADOWING_BUGFIX_V1'
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
ACTIVE=[]
STAGE='EXISTING_G11_RUN01_HANDOFF'

def proc(pid):
    p=Path('/proc')/str(pid)/'stat'
    try:text=p.read_text()
    except FileNotFoundError:return None
    parts=text[text.rindex(')')+2:].split()
    return dict(state=parts[0],ppid=int(parts[1]),start=parts[19],exit_raw=int(parts[49]))

def environment(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/data/zypan/transformers-qwen35/src',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return e

def disk_full_gate(label):
    v=os.statvfs(ROOT);available=v.f_bavail*v.f_frsize
    budget=C.read(PAR/'engineering2048_storage_supplement.json')
    required=budget['PEAK_REQUIRED_STORAGE_BYTES']
    C.save(PAR/'capacity'/('%s.json'%label),dict(AVAILABLE_DATA_BYTES=available,PEAK_REQUIRED_STORAGE_BYTES=required,
          HEADROOM_BYTES=available-required,SAFETY_MARGIN_BYTES=budget['SAFETY_MARGIN_BYTES'],margin_counted_once=True,
          STORAGE_CAPACITY_GATE='PASS' if available>=required else 'FAIL'))
    assert available>=required,'FAIL_AND_STOP_STORAGE_CAPACITY'

def gpu_idle(gpu):
    s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not s,'GPU%d occupied by existing compute: %s; never signal unrelated processes'%(gpu,s)

def batch(jobs,label):
    global ACTIVE
    ACTIVE=[];handles=[];start=time.monotonic()
    try:
        for gpu,args in jobs:
            if gpu is not None:gpu_idle(gpu)
            log=PAR/'logs'/('%s_gpu%s.log'%(label,gpu));log.parent.mkdir(parents=True,exist_ok=True)
            f=log.open('x');handles.append(f)
            p=subprocess.Popen([PY]+args,cwd=REPO,env=environment(gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            ACTIVE.append(p)
            print('LAUNCHED '+label+' GPU='+str(gpu)+' PID='+str(p.pid),flush=True)
        minimum=None
        while True:
            statuses=[p.poll() for p in ACTIVE]
            assert not any(x is not None and x!=0 for x in statuses),'New worker failure in '+label
            v=os.statvfs(ROOT);available=v.f_bavail*v.f_frsize;minimum=available if minimum is None else min(minimum,available)
            assert available>=C.read(PAR/'engineering2048_storage_supplement.json')['SAFETY_MARGIN_BYTES'],'Runtime storage safety floor; STOP and retain drive'
            if all(x==0 for x in statuses):break
            time.sleep(10)
        C.save(PAR/'execution'/('%s.json'%label),dict(stage=label,exitcodes=statuses,pids=[p.pid for p in ACTIVE],
             elapsed_seconds=time.monotonic()-start,minimum_available_bytes=minimum,
             gpu_assignments=[j[0] for j in jobs],all_pass=True))
    except Exception:
        # Cancel only newly created workers owned by this controller after a new failure.
        for p in ACTIVE:
            if p.poll() is None:p.terminate()
        for p in ACTIVE:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if p.poll() is None:p.kill();p.wait()
        raise
    finally:
        for f in handles:f.close()
        ACTIVE=[]

def handoff():
    intent=C.read(PAR/'scheduler_handoff_intent.json');worker=intent['worker_pid'];parent=intent['parent_pid']
    while True:
        w=proc(worker)
        if w is None or w['start']!=intent['worker_start'] or w['state']=='Z':break
        assert not list(BUG.glob('*.error.json')),'Existing G11 worker failed; STOP'
        time.sleep(10)
    if w is not None and w['start']==intent['worker_start']:assert w['exit_raw']==0,'Existing G11 exited nonzero'
    result=BUG/'engineering/repeatability/length_02048_run_01/replay_result.json'
    seal=result.with_name('result_seal.json')
    assert C.read(result)['status']=='PASS' and C.read(seal)['replay_result_sha256']==C.sha(result)
    # The old parent may exit from the planned unreadable next launcher. If genuinely
    # stopped after its worker is dead, terminate only that stopped shell, never resume it.
    while True:
        p=proc(parent)
        if p is None or p['start']!=intent['parent_start'] or p['state']=='Z':break
        if p['state'] in ['T','t']:
            assert proc(worker) is None or proc(worker)['state']=='Z'
            os.kill(parent,signal.SIGKILL)
        time.sleep(2)
    runner=Path(intent['runner_path']);runner.chmod(intent['original_mode'])
    assert C.sha(runner)==intent['runner_sha256'] and stat.S_IMODE(runner.stat().st_mode)==intent['original_mode']
    C.save(PAR/'scheduler_handoff_complete.json',dict(existing_worker_naturally_completed=True,
         existing_worker_result_sha256=C.sha(result),old_scheduler_no_longer_active=True,
         current_GPU_worker_signals=0,old_runner_content_and_mode_restored=True,
         old_next_launch_block='Planned administrative permission block, zero computation/output from run02 attempted by old controller',
         scientific_measurements=0))
    print('EXISTING_G11_RUN01_NATURAL_COMPLETION_HANDOFF_PASS',flush=True)

def import_new_passes():
    for name in ['determinism_2048.json','determinism.json','streaming.json']:
        src=BUG/'engineering/gates'/name;dst=PAR/'engineering/gates'/name
        assert C.read(src)['gate']=='PASS'
        import shutil
        shutil.copy2(src,dst);assert C.sha(src)==C.sha(dst)

def topology_compare():
    baseline=BUG/'engineering/repeatability/length_00256_run_01/replay_result.json'
    ref=C.read(baseline);records=[]
    for gpu in range(4):
        path=PAR/'engineering'/('topology_gpu%d'%gpu)/'engineering/repeatability/length_00256_run_01/replay_result.json'
        current=C.read(path);seal=C.read(path.with_name('result_seal.json'))
        assert current['status']=='PASS' and seal['replay_result_sha256']==C.sha(path)
        exact=current['identity']['outputs']==ref['identity']['outputs'] and current['identity']['calls']==ref['identity']['calls']
        assert current['identity']['conditions']==ref['identity']['conditions']
        assert exact,'Cross-GPU bitwise/metric parity failed; no relaxation or retry'
        records.append(dict(gpu=gpu,result_sha256=C.sha(path),reference_sha256=C.sha(baseline),all_conditions_exact=True,
                            elapsed_seconds=current['elapsed_seconds']))
    C.save(B.E.GATES/'cross_gpu_parity.json',dict(gate='PASS',CROSS_GPU_EXACT_PARITY_GATE='PASS',
         criterion=C.read(PAR/'protocol/parallel_schedule.json')['topology_parity_criterion'],records=records,
         fixture_tokens=256,scientific_measurements=0,threshold_relaxed=False))

def final_scheduling_report(action):
    units=list((PAR/'scientific').rglob('replay_result.json'));releases=list((PAR/'scientific').rglob('drive_release_record.json'))
    auth=PAR/'scientific_execution_authorization.json'
    status=dict(FINAL_ACTION=action,DISCOVERY_COMPLETE=str(len(units))+'/16',
       PREMEASUREMENT_COMMIT_SHA=C.read(auth)['PREMEASUREMENT_COMMIT_SHA'] if auth.exists() else None,
       GPU_MAPPING=C.read(PAR/'protocol/parallel_schedule.json')['condition_gpu'],MAX_CONCURRENT_FULL_DOCUMENT_DRIVES=1,
       completed_drive_releases=len(releases),mathematical_source_changes=0,training=0,AIME_generation=0,
       failed_stage=None if action=='DISCOVERY_COMPLETE_REPORT_ONLY' else STAGE,
       BUG_CLASSIFICATION='IMPLEMENTATION_VARIABLE_SHADOWING',BUGFIX_SCOPE='VARIABLE_RENAME_OR_SCOPE_CORRECTION_ONLY',
       FAILED_ATTEMPT_PRESERVED='YES',SCIENTIFIC_MEASUREMENT_BEFORE_FIX=0,G0_G10_PREVIOUS_PASS_PRESERVED='YES',
       G11_256_TOKEN_STATUS='PASS',G11_2048_TOKEN_PREVIOUS_STATUS='FAIL_IMPLEMENTATION_BUG',
       G11_2048_TOKEN_RETEST_STATUS=C.read(B.E.GATES/'determinism.json')['gate'] if (B.E.GATES/'determinism.json').exists() else 'NOT_COMPLETED',
       G12_STATUS=C.read(B.E.GATES/'streaming.json')['gate'] if (B.E.GATES/'streaming.json').exists() else 'NOT_COMPLETED')
    C.save(PAR/'PARALLEL_FINAL_STATUS.json',status)
    S.write_text(PAR/'PARALLEL_FINAL_REPORT.md','# Four-GPU execution amendment outcome\n\n'+json.dumps(status,indent=2)+
        '\n\nSee FINAL_REPORT.md for the unchanged scientific analysis; this supplement is authoritative for scheduling, actual GPU mapping and the expanded storage budget.\n')
    files=[p for p in sorted(PAR.rglob('*')) if p.is_file() and p.name!='PARALLEL_FINAL_SHA256SUMS' and '__pycache__' not in p.parts]
    S.write_text(PAR/'PARALLEL_FINAL_SHA256SUMS',''.join(C.sha(p)+'  '+str(p.relative_to(PAR))+'\n' for p in files))

def main():
    global STAGE
    (PAR/'logs').mkdir(exist_ok=True);B.verify_admin_freeze()
    lock=PAR/'controller.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        handoff()
        STAGE='G11_2048_RUN02'
        batch([(0,[str(BUG/'scripts/run_repaired_stage.py'),'--stage','replay2048','--run','02'])],'G11_run02')
        STAGE='G11_COMPARE2048'
        batch([(None,[str(BUG/'scripts/run_repaired_stage.py'),'--stage','compare2048'])],'G11_compare')
        STAGE='G12_STREAMING'
        batch([(None,[str(BUG/'scripts/run_repaired_stage.py'),'--stage','streaming'])],'G12_streaming')
        import_new_passes();disk_full_gate('before_topology')
        STAGE='CROSS_GPU_EXACT_PARITY'
        stage=str(PAR/'scripts/parallel_stage.py')
        batch([(gpu,[stage,'--stage','topology','--gpu',str(gpu)]) for gpu in range(4)],'topology256')
        topology_compare()
        STAGE='FULL_PREMEASUREMENT_FREEZE';preregister()
        for doc in range(1,5):
            STAGE='DOC_%d_CAPTURE'%doc;disk_full_gate('before_doc%d'%doc)
            batch([(0,[stage,'--stage','capture','--gpu','0','--document',str(doc)])],'doc%d_capture'%doc)
            STAGE='DOC_%d_PARALLEL_REPLAYS'%doc
            jobs=[(gpu,[stage,'--stage','replay','--gpu',str(gpu),'--document',str(doc),'--condition',condition])
                  for condition,gpu in C.read(PAR/'protocol/parallel_schedule.json')['condition_gpu'].items()]
            batch(jobs,'doc%d_parallel_replay'%doc)
            # A successful join plus original S.seal_release is mandatory; no release on partial failure.
            STAGE='DOC_%d_SEAL_RELEASE'%doc
            batch([(None,[stage,'--stage','release','--document',str(doc)])],'doc%d_release'%doc)
        STAGE='FINALIZE'
        batch([(None,[stage,'--stage','finalize'])],'finalize')
        final_scheduling_report('DISCOVERY_COMPLETE_REPORT_ONLY')
        C.save(PAR/'controller_exit.json',dict(exitcode=0,FINAL_ACTION='DISCOVERY_COMPLETE_REPORT_ONLY'))
    except Exception as e:
        C.save(PAR/'analysis/controller.error.json',dict(FINAL_ACTION='FAIL_AND_STOP',stage=STAGE,
             exception=type(e).__name__,message=str(e),traceback=traceback.format_exc(),retry=False))
        # Restore entrypoint permissions only if no old scheduler remains; never unblock an active one.
        intent=C.read(PAR/'scheduler_handoff_intent.json');p=proc(intent['parent_pid'])
        if p is None or p['start']!=intent['parent_start'] or p['state']=='Z':
            Path(intent['runner_path']).chmod(intent['original_mode'])
        if not (PAR/'FINAL_REPORT.md').exists():S.stop(STAGE)
        if not (PAR/'PARALLEL_FINAL_STATUS.json').exists():final_scheduling_report('FAIL_AND_STOP')
        C.save(PAR/'controller_exit.json',dict(exitcode=1,FINAL_ACTION='FAIL_AND_STOP',stage=STAGE))
        raise

if __name__=='__main__':main()
