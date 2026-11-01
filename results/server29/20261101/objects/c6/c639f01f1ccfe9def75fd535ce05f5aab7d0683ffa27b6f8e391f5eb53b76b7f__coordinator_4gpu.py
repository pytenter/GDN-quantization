"""Four physical GPUs; one shared 24GPUh budget; unchanged scientific worker.

Each document's three complete prefixes stay on its frozen physical GPU. Canary
outputs are isolated; scientific output is shared with task-wide checkpoint I/O
serialized by the entry adapter. No automatic retry/resume or scientific reads.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
TASK='QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
CONDITIONS=('FP_NATIVE','H_C128','H_DUAL_M')
ASSIGNMENT={1:0,2:1,3:2,4:3,5:0}
SCIENCE=tuple(f'science_doc{d}_{c}' for d in range(1,6) for c in CONDITIONS)
QUAL=tuple(f'gpu{g}_{p}' for g in range(4) for p in ('canary_initial','canary_fresh'))
UNITS=QUAL+SCIENCE
STAGE=Path('/tmp/qwen_5doc_scores_20261016')


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value); return value


def helpers(root):
    S=module(root/'control/coordinator.py','five_doc_preserved_single_control_helpers')
    C=S.import_helpers(root)
    return S,C


def inventory(C,uuids):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10)
    out={}
    for row in csv.reader(raw.splitlines()):
        v=[x.strip() for x in row]
        if v and v[0].isdigit() and int(v[0]) in range(4):
            g=int(v[0]); C.require(str(g) not in out,'Duplicate physical GPU identity')
            C.require(v[1]==uuids[str(g)] and 'RTX 3090' in v[2] and int(v[3])==24576,'Frozen physical GPU/UUID/type/capacity changed')
            out[str(g)]=dict(physical_GPU_ID=g,GPU_UUID=v[1],name=v[2],total_MiB=int(v[3]),
                            free_MiB=int(v[4]),used_MiB=int(v[5]),utilization_percent=int(v[6]),apps=[])
    C.require(set(out)==set(uuids)=={'0','1','2','3'},'Four exact frozen physical GPUs required')
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10)
    by_uuid={uuid:g for g,uuid in uuids.items()}
    for row in csv.reader(raw.splitlines()):
        if row and row[0].strip() in by_uuid:
            v=[x.strip() for x in row]; C.require(len(v)>=4,'Malformed compute-app inventory')
            out[by_uuid[v[0]]]['apps'].append(dict(pid=int(v[1]),process_name=v[2],used_MiB=v[3]))
    return dict(status='PASS',unix=time.time(),GPUs=out)


def safe_inventory(C,uuids):
    try:return inventory(C,uuids)
    except BaseException as exc:return dict(status='QUERY_FAILED',unix=time.time(),error=repr(exc))


def idle(C,probe,gpu):
    p=probe['GPUs'][str(gpu)]
    C.require(not p['apps'] and p['free_MiB']>=23000 and p['utilization_percent']<=1,
              f'Physical GPU{gpu} not truly idle with23000MiB free','BLOCKED_AND_STOP')
    return p


def command_for(C,root,phase,gpu,uuid):
    worker_phase=phase.split('_',1)[1] if phase in QUAL else phase
    output=root/'qualification_gpu'/f'gpu{gpu}' if phase in QUAL else root
    return [C.PYTHON,'-u',str(root/'control/worker_entry_4gpu.py'),worker_phase,str(gpu),uuid,str(output)],worker_phase,output


def verify(C,S,root):
    C.require(not (root/'FIRST_FAILURE.json').exists(),'Prior failure forbids any new GPU dispatch')
    m=C.read(root/'source_manifest.json'); p=C.read(root/'preregistration.json'); b=C.read(root/'RESOURCE_BUDGET_4GPU.json')
    C.require(m['task']==p['TASK_ID']==b['task_id']==TASK and m['frozen_before_any_new_GPU_forward'] is True,'Wrong frozen new-task identity')
    C.require(p.get('GPU_EXECUTION_AUTHORIZED')=='YES' and p.get('GPU_COUNT',p.get('MAX_GPU_COUNT'))==4,'Explicit four-GPU authorization absent')
    C.require(b.get('status')=='PASS' and b['MAX_NEW_GPU_HOURS']==24.0 and b['GPU_COUNT']==4,'Wrong global resource budget identity')
    uuids=b['GPU_UUIDS']; C.require(set(uuids)=={'0','1','2','3'} and len(set(uuids.values()))==4,'Four distinct true UUIDs required')
    expected={str(k):v for k,v in ASSIGNMENT.items()}
    C.require(p['doc_GPU_assignment']==expected,'Frozen document/GPU assignment changed')
    C.require(p.get('GPU_UUIDS',p.get('G1',{}).get('GPU_UUIDS'))==uuids,'Preregistered UUID map differs')
    C.require(p['score_positions']==list(range(8192,32768,64)),'Scoring positions changed')
    C.source_check(m); S.frozen_documents(C,root)
    C.require(p['documents_manifest_sha256']==C.sha(root/'frozen_documents.json'),'Preregistered document manifest changed')
    paths={row['path'] for row in m['execution_sources']}
    C.require({'control/coordinator_4gpu.py','control/worker_entry_4gpu.py','control/coordinator.py','RESOURCE_BUDGET_4GPU.json'}<=paths,
              'Active control and budget must be in frozen source manifest')
    for rel,digest in S.CORE_SHAS.items():C.require(C.sha(root/rel)==digest,'Qualified science bytes changed: '+rel)
    for rel in ('qualification_cpu/ENGINEERING_ADAPTER_CPU.json','qualification_cpu/CPU_STRUCTURE_001.json',
                'qualification_cpu/MODEL_CONTENT_IDENTITY.json','qualification_cpu/DEVICE_ADAPTER_CPU.json',
                'qualification_cpu/INDEPENDENT_SCIENCE_DELIVERY_REVIEW.json',
                'qualification_cpu/CONTROL_4GPU_RESOURCE_REVIEW.json'):
        C.require(C.read(root/rel)['status']=='PASS','CPU source/model prerequisite absent: '+rel)
    device_audit=C.read(root/'qualification_cpu/DEVICE_ADAPTER_CPU.json')
    C.require(device_audit['source_sha256']['control/worker_entry_4gpu.py']==C.sha(root/'control/worker_entry_4gpu.py'),
              'Device adapter CPU qualification source changed')
    base=Path(b['projection_basis_source_path'])
    C.require(base.is_absolute() and C.sha(base)==b['projection_basis_source_sha256'],'Historical32K resource source changed')
    old=C.read(base); rows=[r for r in old['phases'] if r['phase'].startswith('science_doc')]
    C.require(old['final_status']=='COMPLETE_AND_STOP' and old['final_GPU_release_verified'] is True and len(rows)==6,'Historical completed resource basis invalid')
    C.require(set(b['phase_predictions_seconds'])==set(SCIENCE),'Exactly15 formal phase predictions required')
    for condition in CONDITIONS:
        values=[S.positive(r['reservation_seconds'],r['phase']) for r in rows if r['phase'].split('_',2)[2]==condition]
        C.require(len(values)==2,'Historical same-condition resource sample incomplete')
        expected=max(values)*1.10
        C.require(all(b['phase_predictions_seconds'][phase]==expected for phase in SCIENCE if phase.endswith('_'+condition)),
                  'Four-GPU base must be historical worst same-condition reservation times1.10')
    canary_basis=Path(b['canary_baseline_source_path'])
    C.require(canary_basis.is_absolute() and C.sha(canary_basis)==b['canary_baseline_source_sha256'],
              'Historical canary median baseline identity changed')
    cb=C.read(canary_basis)
    for condition in CONDITIONS:S.positive(cb['throughput'][condition]['median_seconds'],'historical canary median '+condition)
    S.positive(b['qualification_prediction_seconds'],'qualification_prediction_seconds')
    C.require(b['checkpoint_lock_wait_reserve_seconds']==360 and b['safety_exit_reserve_seconds']==900,'Shared IO/exit reserve changed')
    C.require(sum(b['phase_predictions_seconds'].values())+b['qualification_prediction_seconds']+360+900<86400,
              'Full15 plus four-card qualification+IO+exit cannot fit shared24GPUh','BLOCKED_AND_STOP')
    for gpu in range(4):
        lane=root/'qualification_gpu'/f'gpu{gpu}'
        C.require(lane.is_dir(),'Canary output scope absent')
        for rel in ('source_manifest.json','preregistration.json'):
            C.require(C.sha(lane/rel)==C.sha(root/rel),'Canary source/protocol copies differ')
        C.require((lane/'scores').is_symlink() and (lane/'scores').resolve()==STAGE/f'gpu{gpu}','Canary score filesystem mapping changed')
    C.require((root/'scores').is_symlink() and (root/'scores').resolve()==STAGE and STAGE.is_dir() and not any(STAGE.rglob('*.pt')),
              'New score staging must have exact mapping and no inherited tensor files')
    C.require(not (root/'checkpoints/CURRENT.pt').exists() and not (root/'checkpoints/CATALOG.jsonl').exists(),'New task cannot inherit a prefix checkpoint')
    return m,p,b,uuids


def storage(C,S,root,budget):
    cfg=budget['storage']; C.require(Path(cfg['score_directory'])==STAGE,'Score staging path changed')
    C.require((root/'scores').is_symlink() and (root/'scores').resolve()==STAGE and STAGE.is_dir()
              and root.stat().st_dev!=STAGE.stat().st_dev,'Exact separate filesystem mapping required')
    for gpu in range(4):
        link=root/'qualification_gpu'/f'gpu{gpu}'/'scores'
        C.require(link.is_symlink() and link.resolve()==STAGE/f'gpu{gpu}','Canary score symlink changed')
    metadata=S.tree_bytes(root); scores=S.tree_bytes(STAGE); cp=0
    for path in (root/'checkpoints').glob('CURRENT*') if (root/'checkpoints').exists() else []:
        try:cp+=path.stat().st_size
        except FileNotFoundError:pass
    # Complete original function under one lock permits only CURRENT + tmp.
    C.require(cfg['checkpoint_file_upper_bound_bytes']>=1507209134,'Actual qualified checkpoint upper bound reduced')
    cp_peak=2*cfg['checkpoint_file_upper_bound_bytes']
    small_peak=512*1024**2+cfg['canary_checkpoint_total_bound_bytes']
    C.require(cfg['canary_checkpoint_total_bound_bytes']>=4*60000000,'Four canary checkpoint allowance reduced')
    non_cp=max(0,metadata-cp)
    metadata_remaining=max(0,cp_peak-cp)+max(0,small_peak-non_cp)
    C.require(cfg['canary_score_upper_bound_bytes']>=200000000 and cfg['score_one_write_allowance_bytes']>=382000000,
              'Four-card canary or current score-write allowance reduced')
    score_peak=15*cfg['score_file_upper_bound_bytes']+cfg['canary_score_upper_bound_bytes']+cfg['score_one_write_allowance_bytes']
    C.require(cfg['score_file_upper_bound_bytes']>=381536430 and cfg['metadata_reserve_bytes']>=1024**3
              and cfg['score_reserve_bytes']>=1024**3,'Qualified storage bounds/reserves reduced')
    mf,sf=S.free_bytes(root),S.free_bytes(STAGE)
    C.require(mf>=metadata_remaining+cfg['metadata_reserve_bytes'],'Metadata/globalCP/canaryCP filesystem lacks peak+reserve','BLOCKED_AND_STOP')
    C.require(sf>=max(0,score_peak-scores)+cfg['score_reserve_bytes'],'All15 plus four-card canary scores lack disk+reserve','BLOCKED_AND_STOP')
    C.require(metadata+scores<C.MAX_ARCHIVE_BYTES,'New combined archive reaches original10GiB cap','BLOCKED_AND_STOP')
    return dict(status='PASS',metadata_bytes=metadata,score_bytes=scores,global_CURRENT_and_tmp_bytes=cp,
                metadata_free_bytes=mf,score_free_bytes=sf,metadata_remaining_peak_bytes=metadata_remaining,
                canary_checkpoint_total_bound_bytes=cfg['canary_checkpoint_total_bound_bytes'],
                task_global_single_CURRENT=True,separate_filesystems=True)


def total_seconds(ledger,active,now):
    return math.fsum(p['reservation_seconds'] for p in ledger['phases'])+math.fsum(now-j['start'] for j in active.values())


def forecast(budget,unfinished,ledger,active,now):
    observed={}
    for row in ledger['phases']:
        if row['phase'] in SCIENCE:
            cond=row['phase'].split('_',2)[2]
            observed[cond]=max(observed.get(cond,0),row['reservation_seconds'])
    base=budget.get('adjusted_phase_predictions_seconds',budget['phase_predictions_seconds'])
    predicted={p:max(base[p],observed.get(p.split('_',2)[2],0)) for p in unfinished}
    by_phase={j['phase']:max(0,now-j['start']) for j in active.values()}
    remaining=math.fsum(max(0,v-by_phase.get(p,0)) for p,v in predicted.items())
    return remaining,dict(method='monotone_same_condition_full32K_reservation_floor; active elapsed counted exactly once',
                          phase_predictions_seconds=predicted,observed_condition_max_seconds=observed,
                          downward_updates=False,active_elapsed_seconds=by_phase)


def adjust_after_fresh(C,root,budget):
    old=C.read(Path(budget['canary_baseline_source_path']))
    old_rows=C.read(Path(budget['projection_basis_source_path']))['phases']
    adjusted={}; detail={}
    for condition in CONDITIONS:
        previous=old['throughput'][condition]['median_seconds']
        current=[C.read(root/'qualification_gpu'/f'gpu{g}'/'G4_RUNTIME_BUDGET.json')['throughput'][condition]['median_seconds'] for g in range(4)]
        C.require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in [previous]+current),'Invalid actual OFF postwarm median')
        ratio=max(current)/previous
        prior_allowance=budget['phase_predictions_seconds'][next(p for p in SCIENCE if p.endswith('_'+condition))]
        historical_worst=max(r['reservation_seconds'] for r in old_rows
                             if r['phase'].startswith('science_doc') and r['phase'].split('_',2)[2]==condition)
        value=max(prior_allowance,historical_worst*ratio)
        for phase in SCIENCE:
            if phase.endswith('_'+condition):adjusted[phase]=value
        detail[condition]=dict(historical_OFF_postwarm_median_seconds=previous,
                actual_four_OFF_postwarm_median_seconds=current,max_current_to_historical_ratio=ratio,
                historical_worst_full32K_reservation_seconds=historical_worst,
                frozen_10percent_base_seconds=prior_allowance,adjusted_per_phase_seconds=value)
    result=dict(status='MEASURED_RESOURCE_ADJUSTMENT',rule='max(old worst full32K reservation *1.10, old worst reservation *max4 current OFF postwarm median / old OFF median); then observed full32K monotone floor',
                no_repeated_10percent_multiplier=True,no_scientific_outcomes_read=True,
                canary_baseline_source_sha256=budget['canary_baseline_source_sha256'],conditions=detail,
                current_canary_G4_sources={str(g):dict(path=f'qualification_gpu/gpu{g}/G4_RUNTIME_BUDGET.json',
                    sha256=C.sha(root/'qualification_gpu'/f'gpu{g}'/'G4_RUNTIME_BUDGET.json')) for g in range(4)},
                adjusted_phase_predictions_seconds=adjusted)
    C.write('analysis/FOUR_GPU_PREDICTION_ADJUSTMENT.json',result,exclusive=True)
    return dict(budget,adjusted_phase_predictions_seconds=adjusted),result


def cross_gpu_canary(C,root):
    keys=('t','input_id','logits','cache','cache_cursor')
    checked=[]
    for tag,helper in (('N_FRESH',False),('A_FRESH',True),('D_FRESH',True)):
        compare_keys=keys+('helper','QDQ') if helper else keys
        values=[]
        for gpu in range(4):
            result=C.read(root/'qualification_gpu'/f'gpu{gpu}'/'analysis'/f'{tag}.json')
            C.require(result['status']=='PASS' and result['consumed_tokens']==128 and len(result['rows'])==128,'Cross-card canary missing exact128 observations')
            C.require([r['t'] for r in result['rows']]==list(range(128)),'Cross-card canary token order mismatch')
            values.append([{k:r[k] for k in compare_keys} for r in result['rows']])
        C.require(all(v==values[0] for v in values[1:]),'Cross-physical-GPU128 cache/logit/helper exact qualification failed')
        checked.append(tag)
    return dict(status='PASS',scope='Four physical3090 GPUs,128 tokens only; no32K forward-uncertainty bound',
                compared_fresh_tags=checked,all128_input_cache_cursor_logits_and_applicable_helper_QDQ_exact=True)


def canary_memory(C,root,gpu):
    lane=root/'qualification_gpu'/f'gpu{gpu}'; g4=C.read(lane/'G4_RUNTIME_BUDGET.json')
    C.require(g4['projected_peak_reserved_bytes']<g4['admissible_memory_bytes'],'Original measured canary memory threshold failed','BLOCKED_AND_STOP')
    C.require(C.read(lane/'analysis/CANARY_REDUCER.json')['status']=='PASS','Canary reducer qualification failed')
    return dict(status='PASS',physical_GPU_ID=gpu,original_G4_sha256=C.sha(lane/'G4_RUNTIME_BUDGET.json'),
                original_memory_admission_enforced=True,six_prefix_time_scope='DIAGNOSTIC_ONLY_NOT15_PHASE_BUDGET')


def run(root):
    root=Path(root).resolve(); S,C=helpers(root)
    C.require(root.name==TASK,'Wrong independent new-task root')
    if any((root/f).exists() for f in ('CONTROL_4GPU_RUN_GUARD.json','COORDINATOR_STARTED.json','GATE_STATUS_FINAL.json')):
        return dict(status='REFUSED_EXISTING_ATTEMPT',no_GPU_dispatch=True)
    logfile=S.resident_stdio(root)
    gates={p:'NOT_RUN' for p in UNITS}; gates.update(canary_initial='NOT_RUN',canary_fresh='NOT_RUN',cross_gpu_canary_exact='NOT_RUN')
    ledger=dict(task_id=TASK,GPU_HOURS_USED=0.0,MAX_GPU_HOURS=24.0,GPU_COUNT=4,old_task_GPU_hours_included=False,
                phases=[],started_unix=time.time(),supervisor_stdio_file=logfile,
                accounting='SUM of each physicalGPU worker dispatch through independently observed release; all live reservations added; shared24GPUh, no per-card budget reset')
    active={}; registry={}; phase=None; failure=None; uuids={}; manifest_sha=budget_sha=None
    def signal_stop(signum,frame):raise C.StopExecution('Supervisor signal '+str(signum)+'; no retry')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal_stop)
    def exact(identity,label):return bool(identity.get('pid',0)>0 and label in registry and identity.get('cmdline')==registry[label])
    C.exact_child=exact
    def live_write():
        now=time.monotonic(); used=total_seconds(ledger,active,now)
        C.write('analysis/GPU_RESOURCE_LEDGER_LIVE.json',dict(ledger,active_cumulative_GPUh=used/3600,
                active_reservations=[dict(phase=j['phase'],pid=j['child'].pid,physical_GPU_ID=g,
                       GPU_UUID=uuids[str(g)],active_reserved_seconds=now-j['start']) for g,j in active.items()]))
    def finish(gpu,allow_failure=False):
        job=active[gpu]; child=job['child']; entry=job['entry']; output=job['output']; label=job['phase']
        cleanup_error=None
        try:entry['cleanup']=C.terminate_own_child(child,job['identity'],label)
        except BaseException as exc:cleanup_error=exc; entry['cleanup_error']=repr(exc)
        deadline=time.monotonic()+30; probe=safe_inventory(C,uuids)
        while time.monotonic()<deadline and not (probe['status']=='PASS' and not any(a['pid']==child.pid for a in probe['GPUs'][str(gpu)]['apps'])):
            time.sleep(.5);probe=safe_inventory(C,uuids)
        released=probe['status']=='PASS' and not any(a['pid']==child.pid for a in probe['GPUs'][str(gpu)]['apps'])
        elapsed=time.monotonic()-job['start']; entry.update(returncode=child.poll(),reservation_seconds=elapsed,
                GPU_hours=elapsed/3600,released=released,release_probe=probe,release_observation_unix=time.time(),
                reservation_duration_is_lower_bound_if_release_unverified=not released)
        job['log'].close(); ledger['phases'].append(entry); ledger['GPU_HOURS_USED']=math.fsum(r['GPU_hours'] for r in ledger['phases'])
        entry['_last_accounted_monotonic']=time.monotonic(); del active[gpu]
        rp=output/'analysis'/f'RECEIPT_{job["worker_phase"]}.json'; valid=False
        if child.poll()==0 and rp.is_file() and released and cleanup_error is None:
            receipt=C.read(rp); valid=receipt.get('status')==receipt.get('result',{}).get('status')=='PASS'
            try:
                C.require(C.sha(root/'source_manifest.json')==manifest_sha,'Source manifest changed during worker execution')
                C.source_check(manifest)
            except BaseException as source_error:
                valid=False;entry['source_identity_error']=repr(source_error)
            if label in SCIENCE:
                parts=label.split('_',2)
                valid=valid and receipt['result']==dict(status='PASS',condition=parts[2],doc=int(parts[1][3:]),consumed=32768,scores=384)
            if valid:
                entry['receipt_sha256']=C.sha(rp); gates[label]='PASS'
                C.write(f'analysis/PASS_{label}.json',dict(status='PASS',phase=label,worker_phase=job['worker_phase'],
                        physical_GPU_ID=gpu,GPU_UUID=uuids[str(gpu)],receipt_path=str(rp.relative_to(root)),
                        receipt_sha256=entry['receipt_sha256']),exclusive=True)
        if not valid:gates[label]='FAIL'
        live_write()
        if not allow_failure:
            C.require(valid and released and cleanup_error is None,'Worker/receipt/release failed: '+label)
            C.require(not (output/'FIRST_FAILURE.json').exists(),'Worker recorded FIRST_FAILURE: '+label)
            if label.endswith('canary_fresh'):
                memory=canary_memory(C,root,gpu)
                C.write(f'analysis/CANARY_MEMORY_GPU{gpu}.json',memory,exclusive=True)
        return entry
    try:
        manifest,prereg,budget,uuids=verify(C,S,root)
        manifest_sha=C.sha(root/'source_manifest.json');budget_sha=C.sha(root/'RESOURCE_BUDGET_4GPU.json');ledger['GPU_UUIDS']=uuids
        residency=S.tmux_residency()
        try:C.write('CONTROL_4GPU_RUN_GUARD.json',dict(pid=os.getpid(),unix=time.time(),source_manifest_sha256=manifest_sha),exclusive=True)
        except FileExistsError:return dict(status='REFUSED_SIMULTANEOUS_COORDINATOR',no_GPU_dispatch=True)
        C.write('control/TMUX_RESIDENCY_4GPU.json',residency,exclusive=True);storage(C,S,root,budget)
        initial=inventory(C,uuids)
        for gpu in range(4):idle(C,initial,gpu)
        C.write('gpu_admission.json',dict(status='PASS',GPU_COUNT=4,GPU_UUIDS=uuids,probe=initial,
                MAX_TOTAL_GPU_HOURS=24.0,per_card_budget_not_24=True),exclusive=True)
        C.write('COORDINATOR_STARTED.json',dict(task_id=TASK,pid=os.getpid(),unix=time.time(),phases=list(UNITS),GPU_COUNT=4,
                GPU_UUIDS=uuids,source_manifest_sha256=manifest_sha,resource_budget_sha256=budget_sha,
                doc_GPU_assignment={str(k):v for k,v in ASSIGNMENT.items()},single_attempt_no_retries=True),exclusive=True)
        qual_queues={g:[f'gpu{g}_canary_initial',f'gpu{g}_canary_fresh'] for g in range(4)}
        science_queues={g:[p for p in SCIENCE if ASSIGNMENT[int(p.split('_',2)[1][3:])]==g] for g in range(4)}
        for stage,queues in (('QUALIFICATION',qual_queues),('SCIENCE',science_queues)):
            if stage=='SCIENCE':
                C.require(all(gates[p]=='PASS' for p in QUAL),'All four physicalGPU initial/fresh gates must PASS')
                cross=cross_gpu_canary(C,root);C.write('analysis/CROSS_GPU_CANARY_EXACT.json',cross,exclusive=True)
                gates.update(canary_initial='PASS',canary_fresh='PASS',cross_gpu_canary_exact='PASS')
                budget,adjustment=adjust_after_fresh(C,root,budget)
                future,method=forecast(budget,SCIENCE,ledger,active,time.monotonic())
                C.require(total_seconds(ledger,active,time.monotonic())+future+360+900<86400,
                          'Actual four-card qualification plus15 trajectories+IO+exit cannot fit shared24GPUh','BLOCKED_AND_STOP')
                C.write('analysis/FOUR_GPU_RESOURCE_ADMISSION.json',dict(status='PASS',actual_qualification_GPUh=ledger['GPU_HOURS_USED'],
                        remaining_science_seconds=future,checkpoint_IO_reserve_seconds=360,safety_exit_reserve_seconds=900,
                        resource_budget_sha256=budget_sha,prediction_adjustment_sha256=C.sha(root/'analysis/FOUR_GPU_PREDICTION_ADJUSTMENT.json'),
                        forecast_method=method),exclusive=True)
            last_storage=last_live=time.monotonic();last_coarse=time.monotonic()-30
            while active or any(queues.values()):
                C.require(not (root/'FIRST_FAILURE.json').exists(),'Master FIRST_FAILURE forbids later work')
                C.require(C.sha(root/'source_manifest.json')==manifest_sha and C.sha(root/'RESOURCE_BUDGET_4GPU.json')==budget_sha,'Frozen source/budget identity changed')
                probe=inventory(C,uuids);now=time.monotonic()
                for gpu in range(4):
                    allowed=active[gpu]['child'].pid if gpu in active else None
                    C.require(all(a['pid']==allowed for a in probe['GPUs'][str(gpu)]['apps']),f'Foreign GPU{gpu} process; stop only exact own children')
                if active:
                    C.require(total_seconds(ledger,active,now)+300*len(active)<86400,'Shared24GPUh reaches300seconds per active-card cleanup reserve','BLOCKED_AND_STOP')
                if stage=='SCIENCE':
                    unfinished=[p for p in SCIENCE if gates[p]!='PASS'];remaining,method=forecast(budget,unfinished,ledger,active,now)
                    C.require(total_seconds(ledger,active,now)+remaining+360+900<86400,'Monotone total four-card forecast cannot finish all15','BLOCKED_AND_STOP')
                for gpu in range(4):
                    if gpu in active or not queues[gpu]:continue
                    phase=queues[gpu].pop(0);idle(C,probe,gpu);storage(C,S,root,budget)
                    C.source_check(manifest);S.frozen_documents(C,root)
                    C.require(gates[phase]=='NOT_RUN','Registered unit cannot be repeated')
                    command,worker_phase,output=command_for(C,root,phase,gpu,uuids[str(gpu)])
                    C.require(not (output/'analysis'/f'RECEIPT_{worker_phase}.json').exists(),'Existing receipt forbids unit restart')
                    registry[phase]=command; log=(root/'logs'/f'{phase}.log').open('x');start=time.monotonic()
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',
                             MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
                    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    entry=dict(phase=phase,worker_phase=worker_phase,physical_GPU_ID=gpu,GPU_UUID=uuids[str(gpu)],
                        output_root_relative='.' if output==root else str(output.relative_to(root)),pid=child.pid,
                        command=command,dispatch_unix=time.time(),released=False,
                        peak_sampled_NVML_MiB=probe['GPUs'][str(gpu)]['used_MiB'],NVML_sample_count=0,coarse_NVML_samples=[])
                    active[gpu]=dict(phase=phase,worker_phase=worker_phase,output=output,child=child,log=log,start=start,
                                     identity=None,entry=entry,last_progress=start,last_log_size=-1)
                    identity=C.process_identity(child.pid);active[gpu]['identity']=identity
                    entry['process_identity_at_dispatch']=identity
                    C.require(exact(identity,phase),'Exact dispatched physicalGPU entry identity failed')
                    C.write(f'analysis/PROCESS_{phase}.json',identity,exclusive=True)
                for gpu,job in list(active.items()):
                    phase=job['phase']; output=job['output']
                    C.require(not (output/'FIRST_FAILURE.json').exists(),'Worker FIRST_FAILURE in '+phase)
                    size=(root/'logs'/f'{phase}.log').stat().st_size
                    if size!=job['last_log_size']:job['last_progress']=time.monotonic();job['last_log_size']=size
                    C.require(time.monotonic()-job['last_progress']<1800,'30-minute no-progress watchdog: '+phase)
                    job['entry']['peak_sampled_NVML_MiB']=max(job['entry']['peak_sampled_NVML_MiB'],probe['GPUs'][str(gpu)]['used_MiB'])
                    job['entry']['NVML_sample_count']+=1
                    if now-last_coarse>=30:job['entry']['coarse_NVML_samples'].append(probe['GPUs'][str(gpu)])
                    if job['child'].poll() is not None:
                        finish(gpu);S.emit(dict(phase=phase,status='PASS',physical_GPU_ID=gpu,
                                               completed_GPUh=ledger['GPU_HOURS_USED']),ledger)
                if now-last_storage>=5:storage(C,S,root,budget);last_storage=now
                if now-last_live>=10:live_write();last_live=now
                if now-last_coarse>=30:last_coarse=now
                if active:time.sleep(.5)
        C.require(all(gates[p]=='PASS' for p in UNITS),'Incomplete registered units cannot become complete')
        final_status='COMPLETE_AND_STOP'
    except BaseException as exc:
        guard=root/'CONTROL_4GPU_RUN_GUARD.json'
        if not ledger['phases'] and not active and guard.is_file() and C.read(guard).get('pid')!=os.getpid():
            # Another launch may acquire the exclusive guard while this launch
            # is still doing CPU preflight. Do not poison the owner's attempt.
            return dict(status='REFUSED_SIMULTANEOUS_COORDINATOR',no_GPU_dispatch=True)
        final_status=getattr(exc,'classification','FAIL_AND_STOP')
        failure=dict(status=final_status,phase=phase,error=repr(exc),traceback=traceback.format_exc(),unix=time.time(),
                     no_GPU_retry=True,no_later_GPU_work=True,active_phases=[j['phase'] for j in active.values()])
        lane_failures=[]
        for gpu in range(4):
            fp=root/'qualification_gpu'/f'gpu{gpu}'/'FIRST_FAILURE.json'
            if fp.is_file():lane_failures.append(dict(path=str(fp.relative_to(root)),sha256=C.sha(fp)))
        failure['preserved_lane_first_failures']=lane_failures
        if not (root/'FIRST_FAILURE.json').exists():C.write('FIRST_FAILURE.json',failure,exclusive=True)
        # Signal all exact own workers first. Never wait for one while leaving
        # the other physical cards doing unbounded additional scientific work.
        for gpu,job in list(active.items()):
            if job['child'].poll() is None:
                try:
                    identity=C.process_identity(job['child'].pid)
                    # A live unreaped Popen child cannot have its PID reused.
                    # If its first /proc read failed, bind its exact cmdline now.
                    if job['identity'] is None and exact(identity,job['phase']):
                        job['identity']=identity;job['entry']['late_dispatch_identity_binding']=identity
                    C.require(job['identity'] is not None and identity['start_time_ticks']==job['identity']['start_time_ticks'] and exact(identity,job['phase']),
                              'Exact own PID identity changed; no broad kill')
                    job['child'].send_signal(signal.SIGTERM);job['entry']['stop_all_exact_SIGTERM_sent']=True
                except BaseException as cleanup:job['entry']['stop_all_signal_error']=repr(cleanup)
        for gpu in list(active):
            try:finish(gpu,allow_failure=True)
            except BaseException as cleanup:
                failure.setdefault('cleanup_errors',[]).append(dict(physical_GPU_ID=gpu,error=repr(cleanup)))
    final=safe_inventory(C,uuids) if uuids else dict(status='QUERY_FAILED',error='No qualified UUID map')
    now=time.monotonic()
    for row in ledger['phases']:
        if not row['released']:
            extension=max(0,now-row['_last_accounted_monotonic']);row['reservation_seconds']+=extension;row['GPU_hours']+=extension/3600
            if final['status']=='PASS' and not any(a['pid']==row['pid'] for a in final['GPUs'][str(row['physical_GPU_ID'])]['apps']):
                row.update(released=True,release_confirmed_late=True,reservation_duration_is_lower_bound_if_release_unverified=False)
        row.pop('_last_accounted_monotonic',None)
    ledger['GPU_HOURS_USED']=math.fsum(r['GPU_hours'] for r in ledger['phases'])
    own={r['pid'] for r in ledger['phases']}
    released=final['status']=='PASS' and not any(a['pid'] in own for gpu in final['GPUs'].values() for a in gpu['apps']) and not active
    if final_status=='COMPLETE_AND_STOP' and not released:
        final_status='FAIL_AND_STOP'
        failure=dict(status=final_status,phase='FINAL_GPU_RELEASE_VERIFICATION',
                     error='Final independent current GPU release verification failed; completed worker PASS units preserved',
                     unix=time.time(),no_GPU_retry=True,no_later_GPU_work=True,final_GPU_release_probe=final)
        if not (root/'FIRST_FAILURE.json').exists():C.write('FIRST_FAILURE.json',failure,exclusive=True)
    ledger.update(final_status=final_status,final_GPU_release_verified=released,final_GPU_release_probe=final,ended_unix=time.time())
    C.write('gpu_resource_ledger.json',ledger,exclusive=True)
    result=dict(task_id=TASK,status=final_status,gates=gates,GPU_COUNT=4,GPU_UUIDS=uuids,GPU_HOURS_USED=ledger['GPU_HOURS_USED'],
                MAX_GPU_HOURS=24.0,GPU_release_verified=released,GPU_release_query=final,no_further_GPU_work=True,
                source_manifest_sha256=manifest_sha,resource_budget_sha256=budget_sha,coordinator_failure=failure,
                first_failure_file='FIRST_FAILURE.json' if (root/'FIRST_FAILURE.json').exists() else None,
                per_gpu_final_gate={str(g):{p:gates[f'gpu{g}_{p}'] for p in ('canary_initial','canary_fresh')} for g in range(4)})
    C.write('GATE_STATUS_FINAL.json',result,exclusive=True);S.emit(result,ledger);return result


def self_test():
    assert len(QUAL)==8 and len(SCIENCE)==15 and len(UNITS)==23
    queues={g:[p for p in SCIENCE if ASSIGNMENT[int(p.split('_',2)[1][3:])]==g] for g in range(4)}
    assert [len(queues[g]) for g in range(4)]==[6,3,3,3]
    assert queues[0][:3]==list(SCIENCE[:3]) and all('doc5' in p for p in queues[0][3:])
    ledger=dict(phases=[dict(phase='gpu0_canary_initial',reservation_seconds=100,GPU_hours=100/3600)])
    active={0:dict(phase='science_doc1_FP_NATIVE',start=10),1:dict(phase='science_doc2_FP_NATIVE',start=20)}
    assert total_seconds(ledger,active,50)==170
    budget=dict(phase_predictions_seconds={p:1000 for p in SCIENCE})
    future,_=forecast(budget,SCIENCE,ledger,active,50)
    assert future==14930 and total_seconds(ledger,active,50)+future==15100
    slower=dict(phases=[dict(phase='science_doc1_H_C128',reservation_seconds=1200,GPU_hours=1200/3600)])
    future,details=forecast(budget,[p for p in SCIENCE if p!='science_doc1_H_C128'],slower,{},0)
    assert future==14800 and all(details['phase_predictions_seconds'][p]==1200 for p in details['phase_predictions_seconds'] if p.endswith('H_C128'))
    from types import SimpleNamespace
    old_medians={c:dict(median_seconds=1.0) for c in CONDITIONS}
    old_rows=[dict(phase=f'science_doc{d}_{c}',reservation_seconds=1000) for d in (1,2) for c in CONDITIONS]
    written=[]
    def synthetic_read(path):
        if str(path)=='/SYNTHETIC_OLD_G4':return dict(throughput=old_medians)
        if str(path)=='/SYNTHETIC_OLD_LEDGER':return dict(phases=old_rows)
        return dict(throughput={c:dict(median_seconds=1.20 if c=='H_DUAL_M' else 1.05) for c in CONDITIONS})
    fake=SimpleNamespace(read=synthetic_read,require=lambda ok,msg:None if ok else (_ for _ in ()).throw(AssertionError(msg)),
                         sha=lambda p:'SYNTHETIC_NO_FILES',write=lambda *a,**k:written.append(a))
    b=dict(phase_predictions_seconds={p:1100.0 for p in SCIENCE},canary_baseline_source_path='/SYNTHETIC_OLD_G4',
           projection_basis_source_path='/SYNTHETIC_OLD_LEDGER',canary_baseline_source_sha256='SYNTHETIC_NO_FILES')
    adjusted,audit=adjust_after_fresh(fake,Path('/SYNTHETIC_ROOT'),b)
    assert all(value==(1200 if p.endswith('H_DUAL_M') else 1100) for p,value in adjusted['adjusted_phase_predictions_seconds'].items())
    assert written and audit['no_repeated_10percent_multiplier'] is True
    return dict(status='PASS',scope='CPU scheduler/budget algebra only; no SSH/NVML/Torch/model/GPU',
                actual_units=23,canaries=8,science=15,frozen_same_card_doc_assignment=True,
                all_active_physical_occupancy_summed=True,active_elapsed_not_double_counted=True,
                monotone_same_condition_forecast=True,four_card_median_slowdown_raises_without_double_margin=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(ROOT));parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:sys.exit(0 if run(args.root)['status']=='COMPLETE_AND_STOP' else 1)
