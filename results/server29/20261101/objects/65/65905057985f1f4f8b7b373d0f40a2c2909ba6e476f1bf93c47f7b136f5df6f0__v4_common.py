"""V4 append-only namespace; old numerical sources and evidence stay immutable."""
import copy
import os
import shutil
import subprocess
import time
from v4_stage1_common import CODE,OUT,RUNS,OLD,V3,LATCH,UUID,ENGINE_SHA,digest,read,emit,checked
from v3_guard import device as physical_device,proc_identity
from run_matched_l7 import PRIOR,load_module
from v2_common import fingerprint

FREEZE=OUT/'V4_PORTABILITY_CODE_FREEZE.json'
MODES={'qualification':dict(endpoint=512,primary=256,secondary=0),
       'micro':dict(endpoint=8448,primary=8192,secondary=2048),
       'formal':dict(endpoint=32768,primary=8192,secondary=2048)}

def tag(mode,doc,gpu=6,rep=0): return f'{mode}_{doc}_gpu{gpu}_fresh{rep}'

def verify():
    assert not LATCH.exists(),'V4_STOP_NO_RETRY'
    freeze=checked(FREEZE)
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==freeze['code_commit']
    for p,h in freeze['code_sha256'].items(): assert digest(p)==h,('V4_FROZEN_CODE_CHANGED',p)
    stage1=checked(OUT/'V4_STAGE1_CODE_FREEZE.json')
    for p,h in stage1['code_sha256'].items(): assert digest(p)==h,('V4_STAGE1_CODE_CHANGED',p)
    assert checked(OUT/'CROSS_DEVICE_FP_REPEATABILITY.json')['status']=='PASS'
    ids=checked(OUT/'V4_REUSED_SOURCE_IDENTITY.json')
    for p,row in ids['assets'].items():
        s=__import__('pathlib').Path(p).stat()
        assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'],('ASSET_METADATA_CHANGED',p)
        if not p.endswith('.safetensors'): assert digest(p)==row['sha256'],('ASSET_CHANGED',p)
    for p,h in ids['historical_STOP_sha256'].items(): assert digest(p)==h,('HISTORICAL_STOP_CHANGED',p)
    assert not __import__('pathlib').Path('/var/run/reboot-required').exists()
    assert not __import__('pathlib').Path('/run/systemd/shutdown/scheduled').exists()
    assert shutil.disk_usage(OUT).free>=2*1024**3,'PHYSICAL_STORAGE_UNAVAILABLE'
    cfg=read(OLD/'PHASE_B_CALIBRATION_CONFIG.json')
    sel=copy.deepcopy(checked(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'))
    assert sel['route']=='A' and len(sel['documents'])==2
    # This prospective map is not launch authority until the amendment passes.
    for e in sel['documents']: e['gpu']=6
    cfg['document_gpu_map']={e['document']:6 for e in sel['documents']}
    return cfg,sel,freeze

def device(gpu,cfg,pid=None):
    assert gpu==6,'V4_FIXED_SERIAL_GPU6_ONLY'
    binding=proc_identity(pid) if pid is not None else None
    if pid is not None:
        evidence=next(p for p in RUNS.glob('*.PROCESS_IDENTITY.json')
                      if checked(p)['identity']==binding)
        stem=evidence.name.removesuffix('.PROCESS_IDENTITY.json')
    else: stem='v4_prelaunch'
    return physical_device(6,cfg,pid,binding=binding,
        evidence=RUNS/(stem+'.GPU_EXCLUSIVITY.jsonl'),require_context=pid is not None)

def stop(reason,stage,**extra):
    if not LATCH.exists(): emit(LATCH,dict(status='STOP',reason=reason,stage=stage,epoch=time.time(),no_retry=True,no_signals_used=True,**extra))

def alive(t):
    p=RUNS/(t+'.PROCESS_IDENTITY.json')
    if not p.exists(): return False
    identity=checked(p)['identity']
    try: return proc_identity(identity['pid'])==identity
    except FileNotFoundError: return False

def complete(mode,doc,gpu=6,rep=0):
    p=RUNS/(tag(mode,doc,gpu,rep)+'.json'); v=checked(p)
    assert v['status']=='PASS' and v['document']==doc and v['gpu_uuid']==UUID
    assert v['endpoint']==MODES[mode]['endpoint'] and all(x['passed'] for x in v['persistent_integrity'].values())
    return v,p

def ledger():
    now=time.time(); rows=[]
    for p in sorted(RUNS.glob('*.LAUNCH_AUTHORIZATION.json')):
        receipt=checked(p); t=p.name.removesuffix('.LAUNCH_AUTHORIZATION.json')
        end,status=now,'ACTIVE_OR_UNKNOWN'
        done,fail=RUNS/(t+'.json'),RUNS/(t+'.STOP.json')
        if fail.exists(): end,status=checked(fail)['stopped_epoch'],'STOP'
        elif done.exists(): end,status=checked(done)['completed_epoch'],'COMPLETE'
        exit_path=RUNS/(t+'.EXIT_OBSERVED.json')
        upper=checked(exit_path)['observed_epoch'] if exit_path.exists() else (now if alive(t) else None)
        rows.append(dict(tag=t,gpu=6,gpu_uuid=UUID,status=status,launch_epoch=receipt['epoch'],end_epoch=end,
            receipt_sha256=digest(p),receipt_to_canonical_gpu_hours=max(0,end-receipt['epoch'])/3600,
            observed_exit_upper_GPUh=None if upper is None else max(0,upper-receipt['epoch'])/3600))
    return dict(epoch=now,status='COMPLETE' if all(x['status']!='ACTIVE_OR_UNKNOWN' for x in rows) else 'IN_PROGRESS',
        jobs=rows,V4_receipt_to_canonical_GPUh=sum(x['receipt_to_canonical_gpu_hours'] for x in rows),
        observed_exit_upper_GPUh=sum(x['observed_exit_upper_GPUh'] or x['receipt_to_canonical_gpu_hours'] for x in rows),
        Stage1_exit_not_exactly_recorded=True,method='Includes startup/loading/FP/JIT; canonical time is not exact process exit; later observed exits are conservative upper bounds.',
        GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None,waiting_external_jobs_CPU_historical_excluded=True)
