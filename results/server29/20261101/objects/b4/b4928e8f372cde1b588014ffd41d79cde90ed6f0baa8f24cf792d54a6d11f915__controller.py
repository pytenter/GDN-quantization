"""Independent, single-attempt controller for the two remaining Doc5 units.

Control only. Parent STOP/seal stay immutable; no qualification, model, kernel,
codec, or scientific reducer is changed here. --self-test never imports torch.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TASK = 'QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1'
RUN_ID = 'C6R_DOC5_CONTINUATION_20261017_V1'
PARENT = Path('/data/zypan/analyses/qwen_gdn_c6r_dual_m_interaction_reuse_5doc_v1/experiments/qwen_gdn/quantizer') / TASK
PARENT_CONTROLLER_SHA = 'bd2b6fbd925da630ba0e5429417668b6b5dc9e98565a7c400b7e866d540363e7'
PARENT_LEDGER_SHA = '62587f327511e664ac749a591b68059f38ee8a1bd876036b574564e4a876203f'
PARENT_SOURCE_SHA = 'd76fcb4057eaaa781806ab6865e91998705ecb773449cf8851d2f0641969f9fc'
PARENT_PREREG_SHA = '721038d6f9a48a7903eaa9339fe8823e95d5763e9b435b55199213625cbf5763'
CHECKPOINT_SHA = '0d9c2e22a634b4327c03e34343087fd3d796276035f653b4b22ead3d8cf442ce'
CHECKPOINT_BYTES = 715984046
OLD_HOURS = 14.467484712157813
HARD, PROTECT = 20 * 3600, 19.5 * 3600
GPU0 = 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
ARCHIVE = '/data/zypan/qwen_c6r_dual_m_interaction_archive/CONTINUATION_20261017/' + TASK
PHASES = ('science_doc5_C6R_C128', 'science_doc5_C6R_DUAL_M')


def need(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    else:
        temp = path.with_name(path.name + '.control_tmp')
        with temp.open('w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        os.replace(temp, path)


def original_control():
    path = PARENT / 'control/coordinator.py'
    need(sha(path) == PARENT_CONTROLLER_SHA, 'Original frozen control module changed')
    spec = importlib.util.spec_from_file_location('original_c6r_control_readonly', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def used_seconds(phases, active=None, now=None):
    now = time.monotonic() if now is None else now
    return OLD_HOURS * 3600 + math.fsum(x['reservation_seconds'] for x in phases) + (max(0, now-active['start']) if active else 0)


def projection(budget, completed, active=None, now=None):
    now = time.monotonic() if now is None else now
    done = {x['phase'] for x in completed if x.get('unit_status') == 'PASS'}
    elapsed = max(0, now-active['start']) if active else 0
    remaining = math.fsum(max(0, budget['phase_predictions_seconds'][p] - (elapsed if active and active['phase']==p else 0)) for p in PHASES if p not in done)
    reserves = budget['additional_GPU_held_IO_reserve_seconds'] + budget['safety_exit_reserve_seconds'] + budget['restore_load_extra_reserve_seconds']
    return used_seconds(completed, active, now) + remaining + reserves


def release_observed(returncode, probe, pid):
    # An empty NVML list alone does not prove that a CPU-loading worker exited.
    return (returncode is not None and probe['status']=='PASS' and
            not any(p['pid']==pid for p in probe['GPUs']['0']['apps']))


def storage(root, budget, O):
    """Original bounds, combined when checkpoint and scores share /data."""
    b = budget['storage']; cp = Path(b['checkpoint_directory'])
    need(cp == root/'checkpoints' and cp.is_dir() and not cp.is_symlink(), 'Only independent new-root checkpoint directory allowed')
    need((root/'scores').is_dir() and not (root/'scores').is_symlink(), 'Only new-root score directory allowed')
    snapshots=[]
    for p in cp.glob('CURRENT*'):
        try:
            if p.is_file(): snapshots.append(dict(path=str(p), bytes=p.stat().st_size))
        except FileNotFoundError:
            pass  # Original atomic replacement can retire CURRENT.tmp.
    occupied=sum(x['bytes'] for x in snapshots)
    need(all(x['bytes'] <= b['checkpoint_file_upper_bound_bytes'] for x in snapshots), 'Original checkpoint file bound exceeded')
    cp_need=max(0, 2*b['checkpoint_file_upper_bound_bytes']-occupied)+b['checkpoint_free_reserve_bytes']
    metadata=max(0,O.tree_bytes(root)-O.tree_bytes(root/'inputs')-O.tree_bytes(cp)); scores=O.tree_bytes(root/'scores')
    # Inputs are already physically included in current free space. They do
    # not replace or relax the frozen reserve for future metadata growth.
    non_scores=max(0,metadata-scores)
    score_peak=4*b['score_file_upper_bound_bytes']+b['one_score_write_allowance_bytes']+b['canary_score_allowance_bytes']
    data_need=max(0,score_peak-scores)+max(0,b['metadata_allowance_bytes']-non_scores)+b['metadata_free_reserve_bytes']
    same=cp.stat().st_dev==root.stat().st_dev
    need(O.available(root) >= (cp_need+data_need if same else data_need), 'Frozen combined data/checkpoint atomic capacity insufficient')
    need(same or O.available(cp)>=cp_need, 'Frozen checkpoint atomic capacity insufficient')
    return dict(status='PASS',checkpoint_files=snapshots,checkpoint_required_bytes=cp_need,data_required_bytes=data_need,same_filesystem=same,
                combined_required_bytes=cp_need+data_need if same else None,free_data_bytes=O.available(root),free_checkpoint_bytes=O.available(cp))


def preflight(root, O):
    need(root.name==TASK and root!=PARENT and not root.is_relative_to(PARENT), 'Independent continuation root required')
    need(not (root/'FIRST_FAILURE.json').exists(), 'New-root prior failure forbids launch')
    a,b,m=read(root/'CONTINUATION_ADMISSION.json'),read(root/'CONTINUATION_BUDGET.json'),read(root/'CONTINUATION_SOURCE_MANIFEST.json')
    need(a['status']=='PASS' and a['GPU_EXECUTION_AUTHORIZED']=='YES', 'Explicit continuation admission required')
    need(a['RUN_ID']==b['RUN_ID']==m['RUN_ID']==RUN_ID and a['TASK_ID']==b['TASK_ID']==m['TASK_ID']==TASK, 'Continuation identity mismatch')
    need(m['frozen_before_any_new_GPU_forward'] is True, 'New control source must be frozen before continuation')
    need(Path(a['parent_root'])==PARENT and a['parent_source_manifest_sha256']==PARENT_SOURCE_SHA and a['parent_preregistration_sha256']==PARENT_PREREG_SHA, 'Parent source identity mismatch')
    need(a['old_GPU_hours']==b['old_GPU_hours']==OLD_HOURS and a['MAX_TOTAL_GPU_HOURS']==b['MAX_TOTAL_GPU_HOURS']==20 and a['PROTECTIVE_STOP_GPUH']==b['PROTECTIVE_STOP_GPUH']==19.5, 'Shared original budget changed')
    need(a['expected_GPU_UUID']==GPU0 and a['allowed_phases']==list(PHASES) and b['physical_GPU_ID']==0, 'Only original Doc5/GPU0 phases allowed')
    need(b['storage']['permanent_archive_root']==ARCHIVE, 'Only independent continuation archive namespace allowed')
    need(sha(PARENT/'GPU_RESOURCE_LEDGER.json')==PARENT_LEDGER_SHA==b['parent_ledger_sha256'], 'Immutable original GPU ledger changed')
    parent=read(PARENT/'GPU_RESOURCE_LEDGER.json')
    need(parent['GPU_HOURS_USED']==OLD_HOURS and parent['final_GPU_release_verified'] is True and all(p['released'] for p in parent['phases']), 'Original GPU releases/accounting not established')
    need(sha(PARENT/'SOURCE_MANIFEST.json')==sha(root/'SOURCE_MANIFEST.json')==PARENT_SOURCE_SHA and sha(root/'source_manifest.json')==PARENT_SOURCE_SHA, 'Original83-source manifest must stay byte exact')
    need(sha(PARENT/'PREREGISTRATION.json')==sha(root/'PREREGISTRATION.json')==PARENT_PREREG_SHA and sha(root/'preregistration.json')==PARENT_PREREG_SHA, 'Original scientific preregistration must stay byte exact')
    original=read(root/'SOURCE_MANIFEST.json'); O.source_check(root,original); O.source_check(root,m)
    c=a['input_checkpoint']; p=root/c['path']
    need(c['path']=='inputs/CURRENT_DOC5_C128.pt' and c['sha256']==CHECKPOINT_SHA and c['bytes']==CHECKPOINT_BYTES and c['cursor']==16384 and
         p.is_file() and not p.is_symlink() and p.stat().st_size==CHECKPOINT_BYTES and sha(p)==CHECKPOINT_SHA, 'Exact archived Doc5 checkpoint required')
    doc=read(root/'frozen_documents.json')['selected_documents'][4]; artifact=doc['artifacts']['input_ids_32768']
    ids=read(root/artifact['path'])
    need(a['doc_id']==doc['document_id'] and a['input_ids_sha256']==artifact['canonical_json_sha256'] and len(ids)==32768 and
         hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==a['input_ids_sha256'], 'Exact original Doc5 token identity required')
    need(set(b['phase_predictions_seconds'])==set(PHASES) and all(math.isfinite(v) and v>0 for v in b['phase_predictions_seconds'].values()), 'Exactly two positive phase forecasts required')
    original_budget=read(PARENT/'RESOURCE_BUDGET.json')
    for key in ('checkpoint_file_upper_bound_bytes','checkpoint_free_reserve_bytes','score_file_upper_bound_bytes','scores_max_local_formal_count',
                'one_score_write_allowance_bytes','canary_score_allowance_bytes','metadata_allowance_bytes','metadata_free_reserve_bytes'):
        need(b['storage'][key]==original_budget['storage'][key], 'Frozen resource bound changed: '+key)
    floors={PHASES[0]:7068.987457180023,PHASES[1]:6670.799368232489}
    need(all(b['phase_predictions_seconds'][p]>=floors[p] for p in PHASES) and b['additional_GPU_held_IO_reserve_seconds']>=360 and
         b['safety_exit_reserve_seconds']>=900 and b['restore_load_extra_reserve_seconds']>=120, 'Conservative forecast/reserves were reduced')
    need(projection(b,[])<PROTECT, 'Whole remaining task cannot fit original protective budget')
    need(not any((root/n).exists() for n in ('CONTROL_RUN_GUARD.json','COORDINATOR_STARTED.json','FINAL_GPU_STATUS.json','GPU_RESOURCE_LEDGER.json')), 'Existing continuation attempt cannot repeat')
    for phase in PHASES:
        outputs=['analysis/'+phase+'.json','analysis/ENV_'+phase+'.json','analysis/RECEIPT_'+phase+'.json',
                 'analysis/CHECKPOINT_LOCK_'+phase+'.json','analysis/CONTINUATION_ADAPTER_'+phase+'.json',
                 'analysis/RESTORE_'+phase+'.json','scores/'+phase+'.pt','logs/'+phase+'.log']
        need(not any((root/p).exists() for p in outputs), 'Existing new-phase output cannot be overwritten or retried: '+phase)
    return a,b,m,original,parent


def verify_receipt(root, phase):
    p=root/'analysis'/('RECEIPT_'+phase+'.json'); need(p.is_file(), 'Worker exited without receipt')
    r=read(p); a=read(root/'analysis'/(phase+'.json')); condition=phase.split('_',2)[2]
    need(r['status']==r['result']['status']==a['status']=='PASS', 'Worker receipt not PASS')
    for k,v in dict(condition=condition,doc=5,consumed=32768,scores=384).items(): need(r['result'][k]==v, 'Worker core receipt mismatch: '+k)
    expected=(dict(resume_cursor=16384,new_model_tokens=16384,prefix_score_count=128,new_score_count=256,rotation_calls=393216,
                   new_writebacks=393216,new_next_token_consumption_checks=393216,restore_audit_status='PASS') if phase==PHASES[0] else
              dict(resume_cursor=0,new_model_tokens=32768,prefix_score_count=0,new_score_count=384,rotation_calls=786432,
                   new_writebacks=786432,new_next_token_consumption_checks=786408))
    for k,v in expected.items():
        need(a[k]==v and r['result'][k]==v, 'Actual continuation counter mismatch: '+k)
    need(a['consumed_tokens']==32768 and a['condition']==condition and len(a['mechanism'])==expected['new_score_count'], 'Combined trajectory/suffix diagnostic scope mismatch')
    if phase==PHASES[0]:
        restore=read(root/'analysis'/('RESTORE_'+phase+'.json'))
        need(restore['status']=='PASS' and restore['checkpoint_sha256']==CHECKPOINT_SHA and restore['cursor']==16384 and restore['first_suffix_token_QDQ_previous_seeded'] is True, 'Actual restore/first-suffix QDQ proof missing')
        need(isinstance(restore['CPU_cache_deep_hash'],str) and len(restore['CPU_cache_deep_hash'])==64 and
             restore['CPU_cache_deep_hash']==restore['GPU_cache_deep_hash'], 'Actual restored cache CPU/GPU deep bytes differ')
    return dict(receipt_path=str(p.relative_to(root)),receipt_sha256=sha(p),actual_counters=expected)


def run(root):
    root=Path(root).resolve(); O=original_control(); C=O.helpers(PARENT)
    if any((root/n).exists() for n in ('CONTROL_RUN_GUARD.json','COORDINATOR_STARTED.json','FINAL_GPU_STATUS.json')):
        return dict(status='REFUSED_EXISTING_ATTEMPT',no_GPU_dispatch=True)
    logpath=O.stdio(root); completed=[]; active=None; failure=None; phase=None; budget=None; parent=None; uuids={}; manifest=None; original=None
    status='FAIL_CLOSED_STOP'; gates={p:'NOT_RUN' for p in PHASES}; registry={}; children={}; final_probe=None
    def stop(signum,frame): raise RuntimeError('Continuation supervisor signal '+str(signum))
    stop_signals=(signal.SIGHUP,signal.SIGTERM,signal.SIGINT)
    for sig in stop_signals: signal.signal(sig,stop)
    C.exact_child=lambda identity,label: identity.get('pid',0)>0 and identity.get('cmdline')==registry.get(label)
    def live():
        now=time.monotonic()
        write(root/'analysis/GPU_RESOURCE_LEDGER_LIVE.json',dict(RUN_ID=RUN_ID,old_GPU_hours=OLD_HOURS,
            active_cumulative_GPUh=used_seconds(completed,active,now)/3600,continuation_phases=completed,
            active_reservations=[] if active is None else [dict(phase=active['phase'],pid=active['child'].pid,physical_GPU_ID=0,active_reserved_seconds=now-active['start'])],
            conservative_total_projection_GPUh=projection(budget,completed,active,now)/3600 if budget else None,
            MAX_GPU_HOURS=20.0,PROTECTIVE_STOP_GPUH=19.5,updated_unix=time.time()))
    def finish(permit_failure=False):
        nonlocal active
        job=active; entry=job['entry']; child=job['child']; cleanup_error=None
        try: entry['cleanup']=C.terminate_own_child(child,job['identity'],job['phase'])
        except BaseException as exc: cleanup_error=repr(exc);entry['cleanup_error']=cleanup_error
        deadline=time.monotonic()+30; probe=O.safe_inventory(uuids)
        while time.monotonic()<deadline and not release_observed(child.poll(),probe,child.pid):
            time.sleep(.5); probe=O.safe_inventory(uuids)
        released=release_observed(child.poll(),probe,child.pid)
        elapsed=time.monotonic()-job['start']; entry.update(returncode=child.poll(),reservation_seconds=elapsed,GPU_hours=elapsed/3600,
            released=released,release_probe=probe,release_observation_unix=time.time(),reservation_duration_is_lower_bound_if_release_unverified=not released,
            process_exit_verified=child.poll() is not None,_last_accounted=time.monotonic())
        valid=child.poll()==0 and released and cleanup_error is None and not (root/'FIRST_FAILURE.json').exists()
        if valid:
            try:entry.update(verify_receipt(root,job['phase']))
            except BaseException as exc:valid=False;entry['receipt_error']=repr(exc)
        entry['unit_status']='PASS' if valid else 'FAIL'; gates[job['phase']]=entry['unit_status']
        job['log'].close(); completed.append(entry);active=None;live()
        if not permit_failure:need(valid,'Continuation worker/receipt/release failed: '+job['phase'])
        return entry
    try:
        admission,budget,manifest,original,parent=preflight(root,O); uuids=parent['GPU_UUIDS']
        hashes={p:sha(root/p) for p in ('CONTINUATION_ADMISSION.json','CONTINUATION_BUDGET.json','CONTINUATION_SOURCE_MANIFEST.json')}
        residency=O.tmux_identity();write(root/'CONTROL_RUN_GUARD.json',dict(RUN_ID=RUN_ID,pid=os.getpid(),unix=time.time(),single_attempt=True),True)
        write(root/'control/TMUX_RESIDENCY.json',residency,True)
        archive=O.archive_preflight(budget);write(root/'archive/ARCHIVE_STORAGE_PREFLIGHT.json',archive,True)
        subprocess.run(O.SSH29+[shlex.join([archive['python'],'-c','import pathlib,sys;assert not pathlib.Path(sys.argv[1]).exists(), "Continuation archive already exists; no reuse/overwrite"',ARCHIVE])],check=True,timeout=20)
        probe=O.inventory(uuids);O.idle(probe,0); disk=storage(root,budget,O)
        write(root/'CONTINUATION_RESOURCE_ADMISSION.json',dict(status='PASS',RUN_ID=RUN_ID,GPU_UUID=GPU0,physical_GPU_ID=0,
            actual_GPU_probe=probe,storage=disk,conservative_total_projection_GPUh=projection(budget,[])/3600,
            scope='Only two original Doc5 units; C128 resumes16384, DualMfrom0; conservative full-reservation forecast is not permission to repeat prefix'),True)
        write(root/'COORDINATOR_STARTED.json',dict(RUN_ID=RUN_ID,pid=os.getpid(),unix=time.time(),allowed_phases=list(PHASES),hashes=hashes,stdio_log=logpath),True)
        for phase in PHASES:
            need(not (root/'FIRST_FAILURE.json').exists(),'Any first failure prohibits next dispatch')
            need(all(sha(root/p)==v for p,v in hashes.items()),'Frozen continuation metadata changed')
            O.source_check(root,original);O.source_check(root,manifest);disk=storage(root,budget,O)
            need(projection(budget,completed)<PROTECT,'Remaining two units cannot fit cumulative protective19.5GPUh')
            probe=O.inventory(uuids);O.idle(probe,0)
            need(gates[phase]=='NOT_RUN' and not (root/'analysis'/('RECEIPT_'+phase+'.json')).exists(),'Continuation unit cannot repeat')
            command=[C.PYTHON,'-u',str(root/'control/recovery_worker_entry.py'),'--gpu','0',phase];registry[phase]=command
            logfile=(root/'logs'/(phase+'.log')).open('x');start=time.monotonic()
            # Defer raising a stop only across Popen/registration. This closes
            # the signal window that could otherwise leave an untracked child.
            # Signals are not blocked, so the exec'ed worker inherits normal
            # default termination behavior rather than a blocked signal mask.
            pending_stop=[]
            def defer_stop(signum,frame): pending_stop.append(signum)
            for sig in stop_signals:signal.signal(sig,defer_stop)
            try:
                need(not pending_stop,'Stop before child dispatch')
                child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),stdout=logfile,stderr=subprocess.STDOUT)
                children[child.pid]=child
                entry=dict(RUN_ID=RUN_ID,phase=phase,physical_GPU_ID=0,GPU_UUID=GPU0,pid=child.pid,command=command,dispatch_unix=time.time(),released=False)
                active=dict(phase=phase,child=child,identity=None,entry=entry,log=logfile,start=start,last_progress=start,last_log_size=-1)
                identity=C.process_identity(child.pid);active['identity']=identity;entry['process_identity_at_dispatch']=identity
                need(C.exact_child(identity,phase),'Exact continuation child identity failed')
            finally:
                for sig in stop_signals:signal.signal(sig,stop)
            need(not pending_stop,'Stop received during child registration: '+str(pending_stop))
            last_disk=last_live=0
            while True:
                now=time.monotonic(); used=used_seconds(completed,active,now)
                need(used<PROTECT and used+300<HARD,'Original cumulative19.5/20GPUh stop')
                need(projection(budget,completed,active,now)<PROTECT,'Conservative remaining forecast no longer fits original budget')
                need(not (root/'FIRST_FAILURE.json').exists(),'Worker numerical/identity failure')
                need(all(sha(root/p)==v for p,v in hashes.items()),'Frozen continuation metadata changed during execution')
                probe=O.inventory(uuids)
                need(all(p['pid']==child.pid for p in probe['GPUs']['0']['apps']),'Foreign GPU0 process; stop only exact own worker')
                size=(root/'logs'/(phase+'.log')).stat().st_size
                if size!=active['last_log_size']:active['last_log_size']=size;active['last_progress']=now
                need(now-active['last_progress']<1800,'30-minute no-progress watchdog')
                entry['peak_sampled_NVML_MiB']=max(entry.get('peak_sampled_NVML_MiB',0),probe['GPUs']['0']['used_MiB'])
                if child.poll() is not None:finish();break
                if now-last_disk>=5:
                    disk=storage(root,budget,O);write(root/'analysis/STORAGE_GUARD_LIVE.json',dict(disk,unix=time.time()));last_disk=now
                if now-last_live>=10:live();last_live=now
                time.sleep(.5)
            O.source_check(root,original);O.source_check(root,manifest)
            a=read(root/'analysis'/(phase+'.json'))
            need(a['score_file']=='scores/'+phase+'.pt' and sha(root/a['score_file'])==a['score_file_sha256'],'New score identity changed before archival')
            # These original archive functions enforce independent Python SHA +
            # external sha256sum, fresh staging, atomic seal, exact-score unlink.
            rels=[Path(a['score_file']),Path('analysis')/(phase+'.json'),Path('analysis')/('RECEIPT_'+phase+'.json'),Path('checkpoints/CURRENT.pt'),Path('checkpoints/CATALOG.jsonl')]
            result=O.archive_assets(root,budget,archive['python'],phase,rels,True)
            write(root/'archive'/('PASS_'+phase+'.json'),dict(RUN_ID=RUN_ID,**result),True)
        need(all(gates[p]=='PASS' for p in PHASES),'Continuation incomplete');status='COMPLETE_AND_STOP'
    except BaseException as exc:
        failure=dict(status='FAIL_CLOSED_STOP',RUN_ID=RUN_ID,phase=phase,error=repr(exc),traceback=traceback.format_exc(),unix=time.time(),
            no_GPU_retry=True,parent_STOP_unchanged=True)
        if not (root/'FIRST_FAILURE.json').exists():write(root/'FIRST_FAILURE.json',failure,True)
        if active:
            if active['identity'] is None and active['child'].poll() is None:
                try:
                    identity=C.process_identity(active['child'].pid)
                    need(C.exact_child(identity,active['phase']), 'Cannot prove newly registered own child for cleanup')
                    active['identity']=identity
                except BaseException as identity_error:
                    failure['cleanup_identity_error']=repr(identity_error)
            try:finish(True)
            except BaseException as cleanup:failure['cleanup_error']=repr(cleanup)
    final_probe=O.safe_inventory(uuids) if uuids else dict(status='QUERY_FAILED',error='No admitted UUID map')
    now=time.monotonic()
    if active and not any(e is active['entry'] for e in completed):
        e=active['entry'];elapsed=max(0,now-active['start']);e.update(returncode=active['child'].poll(),reservation_seconds=elapsed,GPU_hours=elapsed/3600,
            released=False,unit_status='FAIL',reservation_duration_is_lower_bound_if_release_unverified=True,_last_accounted=now);completed.append(e)
    for e in completed:
        if not e['released']:
            extension=max(0,now-e['_last_accounted']);e['reservation_seconds']+=extension;e['GPU_hours']+=extension/3600
            if release_observed(children[e['pid']].poll(),final_probe,e['pid']):
                e.update(released=True,returncode=children[e['pid']].poll(),release_verified_late=True,process_exit_verified=True,reservation_duration_is_lower_bound_if_release_unverified=False)
        e.pop('_last_accounted',None)
    own={x['pid'] for x in completed}
    released=(active is None and all(e['released'] and children[e['pid']].poll() is not None for e in completed) and
              (not own or final_probe['status']=='PASS' and not any(p['pid'] in own for p in final_probe['GPUs']['0']['apps'])))
    hours=used_seconds(completed)/3600
    if not released or hours>=20:
        status='FAIL_CLOSED_STOP'
        if not (root/'FIRST_FAILURE.json').exists():write(root/'FIRST_FAILURE.json',dict(status=status,phase='FINAL_RELEASE_OR_BUDGET',RUN_ID=RUN_ID,no_GPU_retry=True),True)
    parent_rows=parent['phases'] if parent else []
    ledger=dict(task_id=TASK,RUN_ID=RUN_ID,status=status,final_status=status,GPU_COUNT=1,physical_GPU_ID=0,GPU_UUID=GPU0,GPU_UUIDS=uuids,
        old_GPU_HOURS_USED=OLD_HOURS,NEW_GPU_HOURS_USED=math.fsum(e['GPU_hours'] for e in completed),GPU_HOURS_USED=hours,MAX_GPU_HOURS=20.0,PROTECTIVE_STOP_GPUH=19.5,
        parent_ledger_path=str(PARENT/'GPU_RESOURCE_LEDGER.json'),parent_ledger_sha256=PARENT_LEDGER_SHA,parent_phases=parent_rows,
        continuation_phases=completed,phases=parent_rows+completed,final_GPU_release_verified=released,final_GPU_release_probe=final_probe,ended_unix=time.time(),
        accounting='Original cumulative cost plus every new exact child dispatch through independently verified ownGPU release, including load/checkpoint/IO and cleanup; postrelease archive is CPU walltime')
    write(root/'GPU_RESOURCE_LEDGER.json',ledger,True)
    result=dict(task_id=TASK,RUN_ID=RUN_ID,status=status,COORDINATOR_STATUS=status,gates=gates,GPU_HOURS_USED=hours,NEW_GPU_HOURS_USED=ledger['NEW_GPU_HOURS_USED'],
        old_GPU_HOURS_USED=OLD_HOURS,MAX_GPU_HOURS=20.0,PROTECTIVE_STOP_GPUH=19.5,GPU_release_verified=released,GPU_release_query=final_probe,
        no_further_GPU_work=True,parent_STOP_unchanged=True,coordinator_failure=failure)
    write(root/'FINAL_GPU_STATUS.json',result,True)
    return result


def self_test():
    b=dict(phase_predictions_seconds={PHASES[0]:7068.987457180023,PHASES[1]:6670.799368232489},additional_GPU_held_IO_reserve_seconds=360,
           safety_exit_reserve_seconds=900,restore_load_extra_reserve_seconds=120)
    initial=projection(b,[],now=100)
    a=dict(phase=PHASES[0],start=0)
    need(projection(b,[],a,100)==initial,'Active elapsed must not be double charged in remaining forecast')
    c=[dict(phase=PHASES[0],reservation_seconds=3500,unit_status='PASS')]
    need(used_seconds(c,now=0)==OLD_HOURS*3600+3500,'Parent cost lost/double counted')
    need(projection(b,c,now=0)<initial and initial<PROTECT,'Whole remaining task does not fit protective budget')
    need(used_seconds([],dict(start=0),300)>used_seconds([],dict(start=0),200),'Live reservations not charged')
    empty=dict(status='PASS',GPUs={'0':dict(apps=[])})
    need(not release_observed(None,empty,123), 'CPU-loading live child incorrectly considered released')
    need(release_observed(0,empty,123) and release_observed(-15,empty,123), 'Exited child without GPU app must qualify for release')
    need(not release_observed(0,dict(status='PASS',GPUs={'0':dict(apps=[dict(pid=123)])}),123), 'Own GPU PID still present')
    need(not release_observed(0,dict(status='QUERY_FAILED'),123), 'Missing GPU observation incorrectly passes')
    return dict(status='PASS_CPU_ONLY',GPU_work=False,allowed_phases=list(PHASES),initial_cumulative_forecast_GPUh=initial/3600,
                old_GPUh=OLD_HOURS,remaining_hard_GPUh=20-OLD_HOURS,remaining_protective_GPUh=19.5-OLD_HOURS,
                full_C128_forecast_is_resource_allowance_not_prefix_rerun_permission=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:sys.exit(0 if run(args.root)['status']=='COMPLETE_AND_STOP' else 1)
