"""CPU-only heartbeat audit; no scientific scores are printed or used."""
import json
import subprocess
import time
from pathlib import Path
import v4_common as v4
from v3_guard import device as physical_device,proc_identity

def main():
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')==''
    now=time.time(); failures=[]; active=[]; statuses={}; freeze_path=v4.OUT/'V4_PERSISTENT_CODE_FREEZE.json'
    if not freeze_path.exists(): freeze_path=v4.OUT/'V4_PORTABILITY_CODE_FREEZE.json'
    freeze=v4.checked(freeze_path)
    actual_commit=subprocess.check_output(['git','-C',str(v4.CODE),'rev-parse','HEAD'],text=True).strip()
    # Transition creates a new commit before publishing its corresponding freeze.
    transitioning=(v4.OUT/'V4_CPU_TRANSITION_STARTED.json').exists() and not (v4.OUT/'V4_CPU_TRANSITION_COMPLETE.json').exists()
    if actual_commit!=freeze['code_commit'] and not transitioning: failures.append('UNKNOWN_MEASUREMENT_HEAD')
    for p,h in freeze['code_sha256'].items():
        if v4.digest(p)!=h: failures.append('FROZEN_CODE_CHANGED:'+p)
    identities=v4.checked(v4.OUT/'V4_REUSED_SOURCE_IDENTITY.json')
    for p,h in identities['historical_STOP_sha256'].items():
        if v4.digest(p)!=h: failures.append('HISTORICAL_STOP_CHANGED:'+p)
    for p,row in identities['assets'].items():
        s=Path(p).stat()
        if s.st_size!=row['bytes'] or s.st_mtime_ns!=row['mtime_ns']: failures.append('FROZEN_ASSET_METADATA_CHANGED:'+p)
    for p in v4.RUNS.glob('*.PROCESS_IDENTITY.json'):
        identity=v4.checked(p)['identity']
        try:
            actual=proc_identity(identity['pid'])
            if actual==identity: active.append(identity)
        except FileNotFoundError: pass
    cfg=v4.read(v4.OLD/'PHASE_B_CALIBRATION_CONFIG.json')
    assert len(active)<=1,'V4_SERIAL_MAPPING_MORE_THAN_ONE_MEASUREMENT'
    hardware=physical_device(6,cfg,active[0]['pid'] if active else None,binding=active[0] if active else None,
        evidence=v4.OUT/'MONITOR_GPU_EXCLUSIVITY.jsonl',require_context=False)
    if not hardware['exclusive']:
        busy={'UNAUTHORIZED_TARGET_COMPUTE_PID','UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION'}
        if active or not set(hardware['failure_conditions']).issubset(busy): failures.extend(hardware['failure_conditions'])
    for p in (Path('/var/run/reboot-required'),Path('/run/systemd/shutdown/scheduled')):
        if p.exists(): failures.append('MAINTENANCE:'+str(p))
    free=__import__('shutil').disk_usage(v4.OUT).free
    if free<2*1024**3: failures.append('PHYSICAL_DISK_UNAVAILABLE')
    for p in v4.OUT.rglob('*.json'):
        if p.name.endswith('.sha256.json'): continue
        side=p.with_suffix('.sha256.json')
        # Publication may be in progress between canonical and sidecar writes.
        if side.exists():
            if v4.digest(p)!=v4.read(side)['sha256']: failures.append('CANONICAL_SIDECAR_CHANGED:'+str(p))
    for name in ('CROSS_DEVICE_FP_REPEATABILITY.json','P1_L7_PORTABILITY_GATE.json','HARDWARE_MAPPING_AMENDMENT.json',
                 'GPU_512X2_QUALIFICATION.json','MICRO_8448_RESULTS.json','PHYSICAL_RESOURCE_FEASIBILITY.json','FINAL_STATUS.json'):
        p=v4.OUT/name
        if p.exists() and p.with_suffix('.sha256.json').exists():
            value=v4.checked(p)
            statuses[name]=value.get('status',value.get('FORMAL_32K_STATUS','UNKNOWN'))
    if failures: v4.stop('MONITOR_IDENTITY_RESOURCE_OR_OWNERSHIP_FAIL','MONITOR',failures=failures)
    value=dict(epoch=now,actual_commit=actual_commit,freeze_path=str(freeze_path),frozen_code_sha256=v4.digest(freeze_path),
        active_measurement_identities=active,hardware=hardware,disk_free_bytes=free,statuses=statuses,
        independent_STOP_paths=[str(p.relative_to(v4.OUT)) for p in v4.OUT.rglob('*STOP.json')],
        fail_closed_latch=v4.read(v4.LATCH) if v4.LATCH.exists() else None,ledger=v4.ledger(),
        scientific_KL_values_or_rankings_not_accessed_for_decisions=True)
    p=v4.OUT/f'MONITOR_SNAPSHOT_{time.time_ns()}.json'; v4.emit(p,value)
    print(json.dumps(dict(snapshot_path=str(p),statuses=statuses,active_PIDs=[x['pid'] for x in active],
        disk_free_bytes=free,target_compute_PIDs=hardware.get('target_compute_pids'),exclusive=hardware['exclusive'],
        failures=failures,STOP_paths=value['independent_STOP_paths'],ledger_GPUh=value['ledger']['V4_receipt_to_canonical_GPUh']),ensure_ascii=False))

if __name__=='__main__': main()
