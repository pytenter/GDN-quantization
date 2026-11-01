"""Independent V3 namespace and stage gates; historical directories remain read-only."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from objective import digest
from run_matched_l7 import publish,load_module,PRIOR
from v2_common import MODES,fingerprint,tag
from v3_guard import device as hardware

CODE=Path(__file__).resolve().parent
OLD=Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1')
V2=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_FULL_PERSISTENT_PROSPECTIVE_V2')
OUT=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_FAST_RECOVERY_AND_FORMAL_32K_V3')
RUNS=OUT/'runs'; FREEZE=OUT/'V3_CODE_FREEZE.json'; LATCH=OUT/'V3_FAIL_CLOSED_STOP.json'
ENGINE_SHA='d448121735fd5ed7eff3a62689d359f17eca4b5467972ff69d299073a2b865c8'

def read(p): return json.loads(Path(p).read_bytes())
def emit(p,value):
    publish(p,value)
    publish(p.with_suffix('.sha256.json'),dict(path=str(p),sha256=digest(p)))
def checked(p):
    assert digest(p)==read(p.with_suffix('.sha256.json'))['sha256'],('SIDECAR_MISMATCH',str(p))
    return read(p)
def stop(reason,stage,**extra):
    if not LATCH.exists():
        emit(LATCH,dict(status='STOP',reason=reason,stage=stage,epoch=time.time(),no_retry=True,no_signals_used=True,**extra))

def verify():
    assert not LATCH.exists(),'V3_STOP_NO_RETRY'
    freeze=checked(FREEZE)
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==freeze['code_commit']
    for p,h in freeze['code_sha256'].items(): assert digest(p)==h,('CODE_IDENTITY_CHANGED',p)
    ids=checked(OUT/'SOURCE_AND_FROZEN_SELECTION_IDENTITY.json')
    for p,v in ids['assets'].items():
        s=Path(p).stat(); assert s.st_size==v['bytes'] and s.st_mtime_ns==v['mtime_ns'],('ASSET_METADATA_CHANGED',p)
        if not p.endswith('.safetensors'): assert digest(p)==v['sha256'],('ASSET_HASH_CHANGED',p)
    assert checked(OUT/'CPU_REGRESSION_RESULTS.json')['status']=='PASS'
    assert digest(CODE/'persistent_engine_v1.py')==ENGINE_SHA
    assert not Path('/var/run/reboot-required').exists() and not Path('/run/systemd/shutdown/scheduled').exists()
    assert shutil.disk_usage(OUT).free>=2*1024**3,'PHYSICAL_STORAGE_UNAVAILABLE'
    return read(OLD/'PHASE_B_CALIBRATION_CONFIG.json'),read(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),freeze

def device(gpu,cfg,pid=None):
    binding=None; name='prelaunch'
    if pid is not None:
        matches=[p for p in RUNS.glob('*.PROCESS_IDENTITY.json') if read(p)['identity']['pid']==pid]
        assert len(matches)==1,'PROCESS_BINDING_UNKNOWN'
        v=checked(matches[0]); assert v['gpu']==gpu and v['gpu_uuid']==cfg['gpu_uuids'][str(gpu)]
        binding=v['identity']; name=matches[0].name.removesuffix('.PROCESS_IDENTITY.json')
    return hardware(gpu,cfg,pid,binding=binding,evidence=RUNS/(name+'.GPU_EXCLUSIVITY.jsonl'),require_context=pid is not None)

def complete(mode,doc,gpu,rep=0):
    p=RUNS/(tag(mode,doc,gpu,rep)+'.json'); v=checked(p); cfg,sel,freeze=verify()
    assert v['status']=='PASS' and v['endpoint']==MODES[mode]['endpoint'] and v['gpu_uuid']==cfg['gpu_uuids'][str(gpu)]
    assert v['mode']==mode and v['document']==doc and v['repeat']==rep
    assert v['freeze_sha256']==digest(FREEZE) and v['runner_sha256']==digest(CODE/'v2_worker.py')
    assert v['pair_selection_sha256']==digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json')
    assert all(x['passed'] for x in v['persistent_integrity'].values())
    assert checked(RUNS/(tag(mode,doc,gpu,rep)+'.PROCESS_IDENTITY.json'))['wrapper_sha256']==digest(CODE/'v3_worker.py')
    return v,p

def ledger():
    rows=[]; now=time.time()
    for p in sorted(RUNS.glob('*.LAUNCH_AUTHORIZATION.json')):
        r=checked(p); t=p.name.removesuffix('.LAUNCH_AUTHORIZATION.json'); end=now; status='RUNNING'
        done,fail=RUNS/(t+'.json'),RUNS/(t+'.STOP.json')
        if fail.exists(): end=read(fail)['stopped_epoch']; status='STOP'
        elif done.exists():
            v=checked(done); end=v['completed_epoch']; status=v['status']
        rows.append(dict(tag=t,gpu=r['gpu'],gpu_uuid=r['gpu_uuid'],mode=r['mode'],status=status,
            launch_epoch=r['epoch'],end_epoch=end,device_hours=(end-r['epoch'])/3600,receipt_sha256=digest(p)))
    return dict(status='IN_PROGRESS' if any(x['status']=='RUNNING' for x in rows) else 'COMPLETE',epoch=now,
        v3_gpu_hours=sum(x['device_hours'] for x in rows),jobs=rows,GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None,
        cpu_wait_and_external_work_excluded=True,loading_FP_JIT_and_probes_included=True,
        historical_V2_gpu_hours=checked(V2/'GPU_COST_LEDGER.json')['v2_gpu_hours'])
