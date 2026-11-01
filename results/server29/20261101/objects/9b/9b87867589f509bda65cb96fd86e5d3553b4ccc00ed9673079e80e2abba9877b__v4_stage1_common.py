"""Independent Stage1 namespace, minimal fresh identity and physical UUID guard."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from v3_guard import device as physical_device, proc_identity

CODE = Path(__file__).resolve().parent
OUT = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_EFFICIENT_RECOVERY_V4')
RUNS = OUT/'runs'
V3 = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_FAST_RECOVERY_AND_FORMAL_32K_V3')
OLD = Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1')
FREEZE = OUT/'V4_STAGE1_CODE_FREEZE.json'
LATCH = OUT/'V4_FAIL_CLOSED_STOP.json'
ENGINE_SHA = 'd448121735fd5ed7eff3a62689d359f17eca4b5467972ff69d299073a2b865c8'
UUID = 'GPU-de53f243-b271-521c-d875-d96611ed8574'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def emit(path, value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    side=path.with_suffix('.sha256.json')
    with side.open('x',encoding='utf-8') as f:
        json.dump(dict(path=str(path),sha256=digest(path)),f,sort_keys=True,indent=2)


def checked(path):
    assert digest(path)==read(Path(path).with_suffix('.sha256.json'))['sha256'],('SIDECAR_MISMATCH',str(path))
    return read(path)


def verify():
    assert not LATCH.exists(),'V4_STOP_NO_RETRY'
    freeze=checked(FREEZE)
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==freeze['code_commit']
    for p,h in freeze['code_sha256'].items():
        assert digest(p)==h,('STAGE1_CODE_CHANGED',p)
    assert digest(CODE/'persistent_engine_v1.py')==ENGINE_SHA
    ids=checked(OUT/'V4_REUSED_SOURCE_IDENTITY.json')
    for path,row in ids['assets'].items():
        p=Path(path); s=p.stat()
        assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'],('FROZEN_ASSET_METADATA_CHANGED',path)
        if p.suffix!='.safetensors':
            assert digest(p)==row['sha256'],('FROZEN_ASSET_CHANGED',path)
    for p,h in ids['historical_STOP_sha256'].items():
        assert digest(p)==h,('HISTORICAL_STOP_CHANGED',p)
    assert not Path('/var/run/reboot-required').exists() and not Path('/run/systemd/shutdown/scheduled').exists()
    assert shutil.disk_usage(OUT).free>=4*1024**3,'STAGE1_LOGIT_STORAGE_UNAVAILABLE'
    return read(OLD/'PHASE_B_CALIBRATION_CONFIG.json'),freeze


def name(rep):
    return f'fp512_pg19-test-5956_gpu6_fresh{rep}'


def device(cfg, rep=None, pid=None):
    binding=checked(RUNS/(name(rep)+'.PROCESS_IDENTITY.json'))['identity'] if pid is not None else None
    return physical_device(6,cfg,pid,binding=binding,
        evidence=RUNS/((name(rep) if rep is not None else 'prelaunch')+'.GPU_EXCLUSIVITY.jsonl'),
        require_context=pid is not None)


def stop(reason, **extra):
    if not LATCH.exists():
        emit(LATCH,dict(status='STOP',reason=reason,epoch=time.time(),no_signals_used=True,no_retry=True,**extra))


def alive(rep):
    p=RUNS/(name(rep)+'.PROCESS_IDENTITY.json')
    if not p.exists(): return False
    identity=checked(p)['identity']
    try: return proc_identity(identity['pid'])==identity
    except FileNotFoundError: return False


def ledger():
    rows=[]
    for p in sorted(RUNS.glob('fp512_*.LAUNCH_AUTHORIZATION.json')):
        if '.sha256.' in p.name: continue
        receipt=checked(p); t=p.name.removesuffix('.LAUNCH_AUTHORIZATION.json')
        done,fail=RUNS/(t+'.json'),RUNS/(t+'.STOP.json')
        end,status=time.time(),'RUNNING_OR_UNKNOWN'
        if fail.exists(): end,status=checked(fail)['stopped_epoch'],'STOP'
        elif done.exists(): end,status=checked(done)['completed_epoch'],'COMPLETE'
        rows.append(dict(tag=t,gpu=6,gpu_uuid=UUID,receipt_sha256=digest(p),
            launch_epoch=receipt['epoch'],end_epoch=end,status=status,GPU_hours=(end-receipt['epoch'])/3600))
    return dict(stage='FP512_CROSS_DEVICE_ONLY',epoch=time.time(),jobs=rows,
        actual_receipt_to_canonical_GPUh=sum(x['GPU_hours'] for x in rows),
        GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None,
        CPU_wait_external_jobs_excluded=True,exact_process_exit_not_recorded=True)
