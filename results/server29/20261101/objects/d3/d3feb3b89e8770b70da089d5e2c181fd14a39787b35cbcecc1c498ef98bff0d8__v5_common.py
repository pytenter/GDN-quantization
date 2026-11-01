"""Isolated V5 evidence; all historical assets are read-only."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
OUT = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_3090_24GB_RESOURCE_CLOSURE_V5')
V4 = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_EFFICIENT_RECOVERY_V4')
OLD = Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1')
RUNS = OUT / 'runs'
LATCH = OUT / 'V5_FAIL_CLOSED_STOP.json'
FREEZE = OUT / 'V5_CODE_AND_ASSET_FREEZE.json'
ENGINE_SHA = 'd448121735fd5ed7eff3a62689d359f17eca4b5467972ff69d299073a2b865c8'

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_bytes())

def emit(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(v, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    if p.suffix == '.json':
        with p.with_suffix('.sha256.json').open('x', encoding='utf-8') as f:
            json.dump(dict(path=str(p), sha256=digest(p)), f, sort_keys=True); f.write('\n')

def checked(p):
    assert digest(p) == read(Path(p).with_suffix('.sha256.json'))['sha256'], ('SIDECAR_CHANGED', str(p))
    return read(p)

def query(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, ('QUERY_FAILED', args, p.stderr)
    return p.stdout.strip()

def stop(reason, stage, **kw):
    if not LATCH.exists(): emit(LATCH, dict(status='STOP', first_failure=str(reason), stage=stage, epoch=time.time(), no_retry=True, no_signals_used=True, **kw))

def verify():
    assert not LATCH.exists(), 'V5_STOP_NO_RETRY'
    f = checked(FREEZE)
    assert query(['git','-C',str(CODE),'rev-parse','HEAD']) == f['code_commit']
    for p,h in f['code_sha256'].items(): assert digest(p)==h, ('FROZEN_CODE_CHANGED',p)
    for p,h in f['evidence_sha256'].items(): assert digest(p)==h, ('EVIDENCE_CHANGED',p)
    for p,v in f['source_identities']['assets'].items():
        s=Path(p).stat(); assert s.st_size==v['bytes'] and s.st_mtime_ns==v['mtime_ns'], ('ASSET_METADATA_CHANGED',p)
        if not p.endswith('.safetensors'): assert digest(p)==v['sha256'], ('ASSET_CHANGED',p)
    for p,h in f['source_identities']['historical_STOP_sha256'].items(): assert digest(p)==h, ('HISTORY_STOP_CHANGED',p)
    assert digest(CODE/'persistent_engine_v1.py')==ENGINE_SHA
    assert not Path('/var/run/reboot-required').exists() and not Path('/run/systemd/shutdown/scheduled').exists()
    assert shutil.disk_usage(OUT).free>=2*1024**3
    return read(OLD/'PHASE_B_CALIBRATION_CONFIG.json'), checked(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'), f

def tag(mode, doc, repeat=0): return f'{mode}_{doc}_fresh{repeat}'

def device_guard(pid=None, evidence=None, require_context=False):
    from v3_guard import device, proc_identity
    mapping=checked(OUT/'MODEL_PARALLEL_DEVICE_MAP.json')
    cfg=read(OLD/'PHASE_B_CALIBRATION_CONFIG.json'); cfg['gpu_uuids']=mapping['physical_gpu_uuids']
    binding=None if pid is None else proc_identity(pid)
    rows=[device(int(g),cfg,pid,binding=binding,evidence=evidence,require_context=require_context) for g in mapping['physical_gpus']]
    return dict(exclusive=all(r['exclusive'] for r in rows), devices=rows)

def ledger():
    rows=[]; now=time.time()
    for p in sorted(RUNS.glob('*.LAUNCH_AUTHORIZATION.json')):
        r=checked(p); stem=p.name.removesuffix('.LAUNCH_AUTHORIZATION.json'); end=now; status='ACTIVE_OR_UNKNOWN'
        for suffix,field,state in [('.json','completed_epoch','COMPLETE'),('.STOP.json','stopped_epoch','STOP')]:
            q=RUNS/(stem+suffix)
            if q.exists(): end=checked(q)[field]; status=state
        x=RUNS/(stem+'.EXIT_OBSERVED.json'); upper=checked(x)['observed_epoch'] if x.exists() else None
        rows.append(dict(tag=stem,physical_gpus=r['physical_gpus'],launch_epoch=r['epoch'],canonical_or_stop_epoch=end,
            status=status,receipt_to_canonical_device_hours=(end-r['epoch'])/3600*len(r['physical_gpus']),
            exit_observed_upper_device_hours=None if upper is None else (upper-r['epoch'])/3600*len(r['physical_gpus'])))
    return dict(epoch=now,jobs=rows,receipt_to_canonical_GPUh=sum(r['receipt_to_canonical_device_hours'] for r in rows),
        conservative_observed_exit_upper_GPUh=sum(r['exit_observed_upper_device_hours'] or r['receipt_to_canonical_device_hours'] for r in rows),
        includes_loading_FP_JIT=True,waiting_CPU_old_experiments_excluded=True,GPUh_administrative_cap=None,VRAM_percentage_cap=None,
        final=all(r['exit_observed_upper_device_hours'] is not None for r in rows))
