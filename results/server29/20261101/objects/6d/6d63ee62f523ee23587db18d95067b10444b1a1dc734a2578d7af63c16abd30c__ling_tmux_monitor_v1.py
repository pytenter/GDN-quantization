"""Read-only server monitor; no launch, signal, repair, or scientific mutation."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

R = Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
A = Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1/hardware_expansion_gpu3_v1')
FD = R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
P = A/'prospective_parallel'
MON = Path('/data/zypan/ling_monitoring_v1')
PARENT = 'e428b5deb0579b926b2ad098a8699b3f4f2233f0'
UUIDS = {3:'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae',6:'GPU-de53f243-b271-521c-d875-d96611ed8574'}
ASSIGNMENT = {'pg19-test-37702':6,'pg19-test-28988':6,'pg19-test-5396':3,'pg19-test-26493':3}
PIDS = [2726684,2726685,2738631,2739499,2740299]
LOGS = [Path('/data/zypan/historical-rebase-discovery.log'),A/'gpu3_reference_capture.log',
        Path('/data/zypan/gpu3-reference-preparation-supervisor.log'),
        Path('/data/zypan/gpu3-parallel-preparation-supervisor.log'),A/'gpu3_parallel_postcommit.log']
PROTECTED = {
 R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay/advancement_inputs/historical_backend_discovery_runner.py':
 'f228ef31a73b0a8829bf082c6db9aef8a0984f4a7eab69592b3a480f007824a9',
 Path('/data/zypan/experiment_runs/ling_historical_runtime_replay_v1/checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/scripts/bridge.py'):
 'e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd'}

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(args): return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
def append(path,value):
    with path.open('a') as f: f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
def finite(value):
    if isinstance(value,float): assert math.isfinite(value),'NONFINITE_CANONICAL_VALUE'
    elif isinstance(value,dict):
        for x in value.values(): finite(x)
    elif isinstance(value,list):
        for x in value: finite(x)

def validate(unit,doc,rotation,commits):
    assert unit['gate']=='PASS' and unit['task']=='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
    gpu=ASSIGNMENT[unit['document']]
    assert unit['hardware']==dict(physical_gpu=gpu,gpu_uuid=UUIDS[gpu]),'CANONICAL_HARDWARE_MISMATCH'
    assert unit['anchor'] in (8192,2048) and unit['endpoint']==32768
    assert unit['premeasurement_commit'] in commits,'CANONICAL_COMMIT_MISMATCH'
    assert unit['token_sha256']==doc['token_ids_sha256']
    assert unit['rotation_sha256']==rotation['artifact']['sha256']
    assert unit['scores']['E_anchor_sha256']==unit['target']['E_anchor_sha256']
    for key in ('FP_REFERENCE_PAIRING_GATE','INITIAL_PULSE_IDENTITY_GATE','NO_FURTHER_QDQ_GATE'):
        assert unit['target'][key]=='PASS'
    assert unit['target']['teacher_tokens']==32768-unit['anchor']
    finite(unit)

def snapshot(seen):
    result={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'monitor_pid':os.getpid(),'mode':'READ_ONLY_NO_AUTOMATIC_FORMAL_ADVANCEMENT',
            'cleanup_ledger_bookkeeping_remaining':1,'actual_task_processes':{},'alerts':[]}
    for pid in PIDS:
        proc=Path('/proc')/str(pid)
        result['actual_task_processes'][str(pid)]={'present':proc.exists(),
            'command':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc.exists() else None}
    result['actual_gpu_processes']=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])
    hardware=command(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'])
    result['hardware']=hardware
    for gpu,uuid in UUIDS.items():
        assert any(line.split(',')[0].strip()==str(gpu) and line.split(',')[1].strip()==uuid for line in hardware.splitlines()),'GPU_UUID_MISMATCH'
    result['free_bytes']=os.statvfs('/data/zypan').f_bavail*os.statvfs('/data/zypan').f_frsize
    if result['free_bytes']<20*1024**3: result['alerts'].append('STORAGE_BELOW_20_GIB')
    result['active_parent_head']=command(['git','-C',str(R),'rev-parse','HEAD'])
    assert result['active_parent_head']==PARENT,'ACTIVE_PARENT_HEAD_CHANGED'
    for path,expected in PROTECTED.items(): assert sha(path)==expected,('PROTECTED_SOURCE_MISMATCH',str(path))
    stops=[]
    for root in (FD,A):
        for path in root.rglob('*STOP*.json'):
            stops.append({'path':str(path),'sha256':sha(path)})
    result['stop_receipts']=stops
    if stops: result['alerts'].append('STOP_RECEIPTS_PRESENT_REQUIRE_REVIEW_NO_AUTO_RETRY')
    docs={d['document_id']:d for d in read(FD/'document_manifest.json')['documents']}
    bank={c['condition_id']:c for c in read(FD/'rotation_manifest.json')['bank']['conditions']}
    commits={PARENT}
    seal=P/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json'
    if seal.exists(): commits.add(read(seal)['commit'])
    for previous,expected in seen.items():
        assert Path(previous).exists() and sha(previous)==expected,'PREVIOUS_CANONICAL_REMOVED_OR_CHANGED'
    units={}
    for folder in [FD/'discovery/unit_results',A/'discovery_shards/gpu3/discovery/unit_results',A/'discovery_shards/gpu6/discovery/unit_results']:
        for path in folder.glob('*.json'):
            if path.name.endswith('.sha256.json'): continue
            side=path.with_suffix('.sha256.json')
            if not side.exists(): continue # writer has not yet sealed the unit
            identity=read(side);actual=sha(path)
            assert identity['sha256']==actual and identity['bytes']==path.stat().st_size,('CANONICAL_SIDECAR_MISMATCH',str(path))
            if str(path) in seen: assert seen[str(path)]==actual,'PREVIOUS_CANONICAL_CHANGED'
            unit=read(path);validate(unit,docs[unit['document']],bank[unit['condition']],commits)
            key=(unit['document'],unit['condition'],unit['anchor'])
            assert key not in units,('DUPLICATE_CANONICAL_UNIT',key)
            units[key]=unit;seen[str(path)]=actual
    result['canonical_primary_complete']=sum(k[2]==8192 for k in units)
    result['canonical_secondary_complete']=sum(k[2]==2048 for k in units)
    result['canonical_by_gpu']={str(g):sum(u['hardware']['physical_gpu']==g for u in units.values()) for g in UUIDS}
    result['log_tail']={str(p):command(['tail','-n','3',str(p)]) for p in LOGS if p.exists()}
    result['gpu3_fp_manifest_complete']=(A/'reference_preparation/FP_REFERENCE_MANIFEST.json').exists()
    result['gpu3_postcommit_complete']=(A/'postcommit_parallel/gpu3/MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json').exists()
    return result

def main():
    MON.mkdir(exist_ok=True)
    lock=MON/'monitor.pid'
    if lock.exists():
        old=int(lock.read_text())
        assert not Path('/proc',str(old)).exists(),'MONITOR_ALREADY_RUNNING'
        raise RuntimeError('STALE_MONITOR_PID_REQUIRES_EXPLICIT_APPEND_ONLY_RECOVERY')
    with lock.open('x') as f:f.write(str(os.getpid())+'\n')
    seen={};last=None
    while True:
        try:
            state=snapshot(seen)
            append(MON/'snapshots.jsonl',state)
            signature=(state['canonical_primary_complete'],state['canonical_secondary_complete'],
                       state['gpu3_fp_manifest_complete'],state['gpu3_postcommit_complete'],tuple(state['alerts']))
            print(state['time_utc'],'primary',signature[0],'/64 secondary',signature[1],'/64',
                  'GPU3 refs',signature[2],'postcommit',signature[3],'alerts',signature[4],flush=True)
            if signature!=last:
                append(MON/'events.jsonl',state);last=signature
        except Exception as error:
            incident={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'error':repr(error),'action':'ALERT_NO_KILL_NO_RETRY_NO_SCIENTIFIC_MUTATION'}
            append(MON/'monitor_incidents.jsonl',incident);print('MONITOR_ALERT',incident,flush=True)
        time.sleep(60)

if __name__=='__main__': main()
