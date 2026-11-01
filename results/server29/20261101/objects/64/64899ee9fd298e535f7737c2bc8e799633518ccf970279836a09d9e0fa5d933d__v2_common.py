"""Isolated V2 namespaces and identity checks; never calls any V1 guard."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from run_matched_l7 import publish, load_module, PRIOR
from objective import digest

OLD=Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1')
OUT=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_FULL_PERSISTENT_PROSPECTIVE_V2')
CODE=Path(__file__).resolve().parent
RUNS=OUT/'runs'
FREEZE=OUT/'V2_CODE_AND_ASSET_FREEZE.json'
LATCH=OUT/'V2_FAIL_CLOSED_STOP.json'
MODES={'qualification':dict(endpoint=512,primary=256,secondary=0),
       'micro':dict(endpoint=8448,primary=8192,secondary=2048),
       'formal':dict(endpoint=32768,primary=8192,secondary=2048)}

def read(p): return json.loads(Path(p).read_bytes())

def emit(p,value):
    publish(p,value)
    publish(p.with_suffix('.sha256.json'),dict(path=str(p),sha256=digest(p)))

def checked(p):
    assert digest(p)==read(p.with_suffix('.sha256.json'))['sha256'],('SIDECAR_MISMATCH',str(p))
    return read(p)

def stop(reason,stage,**extra):
    try: emit(LATCH,dict(status='STOP',classification='ENGINEERING_OR_NUMERICAL_STOP',
        reason=reason,stage=stage,epoch=time.time(),no_retry=True,no_signals_used=True,**extra))
    except FileExistsError: pass

def verify():
    assert not LATCH.exists(),'V2_STOP_NO_RETRY'
    freeze=checked(FREEZE)
    for p,h in freeze['code_sha256'].items(): assert digest(p)==h,('V2_CODE_CHANGED',p)
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==freeze['code_commit']
    identity=checked(OUT/'SOURCE_AND_WEIGHT_IDENTITY.json')
    assert identity['status']=='PASS'
    for p,row in identity['actual_identities'].items():
        stat=Path(p).stat()
        assert stat.st_size==row['bytes'] and stat.st_mtime_ns==row['mtime_ns'],('FROZEN_ASSET_METADATA_CHANGED',p)
        if not p.endswith('.safetensors'): assert digest(p)==row['sha256'],('FROZEN_ASSET_CHANGED',p)
    assert checked(OUT/'CPU_REGRESSION_TEST_RESULTS.json')['status']=='PASS'
    policy=checked(OUT/'RESOURCE_POLICY_AMENDMENT_V2.json')
    assert policy['GPU_HOURS_HARD_CAP'] is None and policy['FIXED_VRAM_PERCENT_CAP'] is None
    assert digest(OUT/'RESOURCE_POLICY_AMENDMENT_V2.json')==freeze['resource_policy_sha256']
    assert not Path('/var/run/reboot-required').exists() and not Path('/run/systemd/shutdown/scheduled').exists()
    assert shutil.disk_usage(OUT).free>=2*1024**3,'PHYSICAL_STORAGE_UNAVAILABLE'
    cfg=read(OLD/'PHASE_B_CALIBRATION_CONFIG.json')
    sel=read(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json')
    assert sel['route']=='A' and len(sel['documents'])==2
    return cfg,sel,freeze

def device(gpu,cfg,pid=None):
    uuid=cfg['gpu_uuids'][str(gpu)]
    q=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    fields=[x.strip() for x in q.split(',')]
    assert fields[0]==str(gpu) and fields[1]==uuid and fields[3]==cfg['runtime']['driver'],'PHYSICAL_UUID_OR_DRIVER_MISMATCH'
    apps=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    holders=subprocess.run(['fuser',f'/dev/nvidia{gpu}'],capture_output=True,text=True)
    assert holders.returncode in (0,1),'DEVICE_HOLDER_CHECK_FAILED'
    pids=[int(x.split(',')[1]) for x in apps.splitlines() if x.strip()]
    holder_ids=[int(x) for x in holders.stdout.split()]
    allowed=set() if pid is None else {pid}
    exclusive=all(x in allowed for x in pids+holder_ids)
    if pid is None: exclusive=exclusive and float(fields[5])<=64 and float(fields[6])==0
    return dict(gpu=gpu,gpu_uuid=uuid,query=q,compute_apps=apps,holders_stdout=holders.stdout,
        holders_stderr=holders.stderr,holder_returncode=holders.returncode,exclusive=exclusive,
        observed_epoch=time.time())

def fingerprint(value):
    keys=('endpoint','fp_reference_full_logit_sha256','fp_replay_parity_tokens','hybrid_cache_inventories',
          'results','full_model_teacher_forced','every_token_qdq_including_after_anchors','same_gpu_live_fp_reference')
    return hashlib.sha256(json.dumps({k:value[k] for k in keys},sort_keys=True,
        allow_nan=False,separators=(',',':')).encode()).hexdigest()

def tag(mode,doc,gpu,rep=0): return f'{mode}_{doc}_gpu{gpu}_fresh{rep}'

def complete(mode,doc,gpu,rep=0):
    p=RUNS/(tag(mode,doc,gpu,rep)+'.json'); v=checked(p)
    cfg,sel,freeze=verify()
    assert v['status']=='PASS' and v['endpoint']==MODES[mode]['endpoint']
    assert v['gpu_uuid']==cfg['gpu_uuids'][str(gpu)] and v['mode']==mode and v['document']==doc
    assert v['freeze_sha256']==digest(FREEZE) and v['runner_sha256']==digest(CODE/'v2_worker.py')
    assert v['pair_selection_sha256']==digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json')
    assert all(x['passed'] for x in v['persistent_integrity'].values())
    return v,p

def ledger():
    rows=[]; now=time.time()
    for p in sorted(RUNS.glob('*.LAUNCH_AUTHORIZATION.json')):
        r=read(p); t=p.name.removesuffix('.LAUNCH_AUTHORIZATION.json')
        canonical,fail,started=(RUNS/(t+s) for s in ('.json','.STOP.json','.STARTED.json'))
        end,status=now,'ACTIVE_OR_UNKNOWN'
        if fail.exists(): end,status=read(fail)['stopped_epoch'],'STOP'
        elif canonical.exists(): end,status=checked(canonical)['completed_epoch'],'COMPLETE'
        row=dict(tag=t,gpu=r['gpu'],gpu_uuid=r['gpu_uuid'],mode=r['mode'],launch_epoch=r['epoch'],
            receipt_sha256=digest(p),end_epoch=end,status=status,device_hours=max(0,end-r['epoch'])/3600)
        if started.exists(): row['pid']=read(started)['pid']
        rows.append(row)
    historical=read(OLD/'final_stop_20261011/GPU_COST_LEDGER.json')
    return dict(status='COMPLETE' if all(x['status']!='ACTIVE_OR_UNKNOWN' for x in rows) else 'IN_PROGRESS',
        historical_v1_gpu_hours=2.0895585917101966,historical_ledger_path=str(OLD/'final_stop_20261011/GPU_COST_LEDGER.json'),
        historical_ledger_sha256=digest(OLD/'final_stop_20261011/GPU_COST_LEDGER.json'),
        v2_gpu_hours=sum(x['device_hours'] for x in rows),jobs=rows,GPU_HOURS_HARD_CAP=None,
        waiting_external_jobs_cpu_excluded=True,loading_fp_jit_all_included=True,epoch=now)
