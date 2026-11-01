"""Append-only orchestration; original scientific functions are never changed."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,subprocess,time,traceback
W=Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1')
A=W/'hardware_expansion_gpu3_v1'
P=A/'prospective_parallel'
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
FD=R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
PARENT='e428b5deb0579b926b2ad098a8699b3f4f2233f0'
RUNNER='gpu3_parallel_discovery_v1_guard_identity_v2.py'
EXPECTED_RUNNER='3b0a8d9e583cbd34e2a9d91f3a42d93357f080f7f1c10edacb28dfb155f4d94f'
PRESERVED_ATTEMPT01_STOP_SHA256='93e9db17436e5282c4e7b6b5c3224f381c7f7ccbce0dca0f6e20e8f63be01fc3'
ORIGINAL_PIDS=[2726684,2726685]
UUIDS={3:'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae',6:'GPU-de53f243-b271-521c-d875-d96611ed8574'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def row(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*args):return subprocess.check_output(['git','-C',str(W),*args],text=True).strip()
def finite(v):
    if isinstance(v,float):assert math.isfinite(v),'NONFINITE_OUTPUT'
    elif isinstance(v,dict):
        for x in v.values():finite(x)
    elif isinstance(v,list):
        for x in v:finite(x)
def assert_expected_guard_exit(text):
    assert text.rstrip().endswith('AssertionError: PRESERVED_FORMAL_STOP'),'UNEXPECTED_ORIGINAL_EXCEPTION'
    assert text.count('Traceback (most recent call last):')==1,'MULTIPLE_ORIGINAL_EXCEPTIONS'
def classify_engineering_stop(name,digest):
    if name=='ENGINEERING_STOP.json' and digest==PRESERVED_ATTEMPT01_STOP_SHA256:
        return 'PRESERVED_ATTEMPT01_NONBLOCKING'
    return 'BLOCKING'
def validate_engineering_stop_state():
    preserved=P/'ENGINEERING_STOP.json'
    assert preserved.exists(),'PRESERVED_ATTEMPT01_STOP_MISSING'
    assert classify_engineering_stop(preserved.name,sha(preserved))=='PRESERVED_ATTEMPT01_NONBLOCKING','PRESERVED_ATTEMPT01_STOP_CHANGED'
    for candidate in P.glob('*STOP*.json'):
        if candidate==preserved:
            continue
        assert classify_engineering_stop(candidate.name,sha(candidate))!='BLOCKING','NEW_BLOCKING_STOP_PRESENT:'+candidate.name
def sealed():
    seal=read(P/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json');commit=seal['commit']
    assert git('rev-parse','HEAD')==commit and not git('status','--porcelain=v1','--untracked-files=no')
    assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()==PARENT
    for item in seal['input_files']:assert sha(item['path'])==item['sha256'],'FROZEN_INPUT_CHANGED'
    assert sha(P/'inputs'/RUNNER)==EXPECTED_RUNNER
    plan=read(P/'SCIENTIFIC_PLAN.json')
    assert plan['scientific_protocol_changed']=='NO' and plan['FP_REFERENCES_MIXED_ACROSS_HARDWARE']=='NO'
    for item in plan['identity_files']:assert sha(item['path'])==item['sha256'],'FROZEN_IDENTITY_CHANGED'
    validate_engineering_stop_state()
    sync=read(P/'SOURCE_PREMEASUREMENT_SYNC.json')
    assert sync['gate']=='PASS' and sync['commit']==commit and sync['source_tracked_clean'] is True
    receipt=P/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json'
    assert sync['target_seal_sha256']==sha(receipt)
    gate=read(A/'postcommit_parallel/gpu3/MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json')
    assert gate['gate']=='PASS' and gate['premeasurement_commit']==commit
    assert gate['physical_gpu']==3 and gate['gpu_uuid']==UUIDS[3]
    return seal,plan
def request():
    seal,plan=sealed()
    assert all(Path('/proc',str(pid)).exists() for pid in ORIGINAL_PIDS),'ORIGINAL_RUNNER_NOT_ACTIVE'
    assert not (FD/'STOP.json').exists(),'PRESERVE_EXISTING_STOP'
    log=Path('/data/zypan/historical-rebase-discovery.log')
    assert 'Traceback (most recent call last):' not in log.read_text(),'ORIGINAL_ALREADY_FAILED'
    started=[row(p) for p in sorted((FD/'discovery/logs').glob('*.started.json'))]
    assert started,'NO_ACTUAL_STARTED_CONDITION'
    canonical=[row(p) for p in sorted((FD/'discovery/unit_results').glob('*.json')) if not p.name.endswith('.sha256.json')]
    value=dict(kind='ENGINEERING_CONTROLLED_HANDOFF_REQUEST',scientific_failure=False,
        time_unix=time.time(),parent_premeasurement_commit=PARENT,new_premeasurement_commit=seal['commit'],
        requested_by='user-authorized prospective hardware amendment',
        boundary='original condition/document guard; complete both anchors and endpoint; no signal',
        started_at_request=started,canonical_at_request=canonical,original_task_pids=ORIGINAL_PIDS,
        source_sync_receipt=row(P/'SOURCE_PREMEASUREMENT_SYNC.json'),
        gpu3_postcommit=row(A/'postcommit_parallel/gpu3/MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json'),
        no_active_anchor_interrupted=True,cleanup_ledger_bookkeeping_remaining=1)
    emit(FD/'STOP.json',value)
    emit(P/'HANDOFF_REQUEST_AUDIT.json',dict(gate='REQUESTED_NOT_COMPLETED',control_receipt=row(FD/'STOP.json'),
        original_runner_source_unchanged=True,signals_sent=False,scientific_failure=False))
    print('HANDOFF_REQUESTED_CURRENT_PAIR_NOT_INTERRUPTED',flush=True)
def validate():
    seal,plan=sealed();stop=read(FD/'STOP.json')
    assert stop['kind']=='ENGINEERING_CONTROLLED_HANDOFF_REQUEST' and stop['scientific_failure'] is False
    assert stop['new_premeasurement_commit']==seal['commit']
    assert all(not Path('/proc',str(pid)).exists() for pid in ORIGINAL_PIDS),'ORIGINAL_PROCESS_STILL_ACTIVE'
    source=read(P/'SOURCE_ORIGINAL_PROCESS_EXIT.json')
    assert source['gate']=='PASS' and source['source_pid']==3211839 and source['source_process_present'] is False
    assert source['expected_advancement_resume_stop'] is True
    log=Path('/data/zypan/historical-rebase-discovery.log');assert_expected_guard_exit(log.read_text())
    spec=importlib.util.spec_from_file_location('validated_parallel_identity',P/'inputs'/RUNNER)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    docs={d['document_id']:d for d in read(FD/'document_manifest.json')['documents']}
    bank={c['condition_id']:c for c in read(FD/'rotation_manifest.json')['bank']['conditions']}
    unit_files=[];files=[row(log),row(P/'SOURCE_ORIGINAL_PROCESS_EXIT.json')]
    keys=set()
    started=list((FD/'discovery/logs').glob('*.started.json'))
    for entry in started:
        meta=read(entry);doc=meta['document'];cond=meta['condition']
        assert plan['document_assignment'][doc]==6,'ORIGINAL_RAN_GPU3_ASSIGNED_DOCUMENT'
        for anchor in [8192,2048]:
            path=FD/'discovery/unit_results'/(doc+'__'+cond+'__'+str(anchor)+'.json')
            side=path.with_suffix('.sha256.json');identity=read(side)
            assert identity['sha256']==sha(path) and identity['bytes']==path.stat().st_size
            unit=read(path);m.validate_unit(unit,6,doc,cond,anchor,docs[doc]['token_ids_sha256'],bank[cond]['artifact']['sha256'],[PARENT])
            finite(unit);key=(doc,cond,anchor);assert key not in keys;keys.add(key)
            unit_files.append(row(path));files.extend([row(path),row(side)])
        endpoint=FD/'discovery/summaries'/(doc+'__'+cond+'__endpoint32K.json')
        finite(read(endpoint));files.extend([row(endpoint),row(entry)])
    actual=[p for p in (FD/'discovery/unit_results').glob('*.json') if not p.name.endswith('.sha256.json')]
    assert len(actual)==len(unit_files),'CANONICAL_OUTSIDE_STARTED_PAIR_INVENTORY'
    assert all(sha(item['path'])==item['sha256'] for item in stop['canonical_at_request']),'OLD_OUTPUT_CHANGED'
    assert all(sha(item['path'])==item['sha256'] for item in stop['started_at_request']),'OLD_STARTED_RECEIPT_CHANGED'
    emit(P/'HANDOFF_COMPLETION.json',dict(gate='PASS',premeasurement_commit=seal['commit'],
        all_original_task_processes_exited=True,actual_gpu6_original_process_present=False,
        source_original_supervisor_exited=True,no_active_anchor_interrupted=True,signals_sent=False,
        controlled_handoff_receipt_sha256=sha(FD/'STOP.json'),files=files,unit_files=unit_files,
        complete_pairs=len(unit_files)//2,scientific_failure=False,cleanup_ledger_bookkeeping_remaining=1))
    print('ACTUAL_SAFE_HANDOFF_VALIDATION_PASS',flush=True)
def no_gpu_process(gpu):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert not any(UUIDS[gpu] in line for line in text.splitlines()),'GPU_OCCUPIED_NO_INTERFERENCE_ALLOWED'
def environment(plan,gpu):
    env=os.environ.copy()
    for k,v in plan['launch_profiles'][str(gpu)]['environment'].items():
        if v is None:env.pop(k,None)
        else:env[k]=v
    return env
def advance():
    seal,plan=sealed();handoff=read(P/'HANDOFF_COMPLETION.json');assert handoff['gate']=='PASS'
    assert handoff['controlled_handoff_receipt_sha256']==sha(FD/'STOP.json')
    assert all(not Path('/proc',str(pid)).exists() for pid in ORIGINAL_PIDS)
    for item in handoff['files']:assert sha(item['path'])==item['sha256']
    for gpu in [3,6]:no_gpu_process(gpu)
    emit(P/'CONTROLLER_ADVANCEMENT_STARTED.json',dict(premeasurement_commit=seal['commit'],time_unix=time.time(),
        controller_sha256=sha(Path(__file__)),stage='GPU6_POSTCOMMIT_ONLY',formal_launch_not_yet_started=True))
    with (A/'gpu6_parallel_postcommit.log').open('xb') as f:
        result=subprocess.run(['/data/zypan/envs/ling-kda/bin/python','-B',str(P/'inputs'/RUNNER),'postcommit','--gpu','6'],
            cwd=W,env=environment(plan,6),stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,'GPU6_POSTCOMMIT_FAIL'
    for gpu in [3,6]:
        gate=A/('postcommit_parallel/gpu'+str(gpu)+'/MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json')
        receipt=read(gate)
        assert receipt['gate']=='PASS' and receipt['premeasurement_commit']==seal['commit']
        assert receipt['physical_gpu']==gpu and receipt['gpu_uuid']==UUIDS[gpu]
        no_gpu_process(gpu)
        emit(P/('GPU'+str(gpu)+'_FORMAL_AUTHORIZATION.json'),dict(gate='PASS',premeasurement_commit=seal['commit'],
            physical_gpu=gpu,gpu_uuid=UUIDS[gpu],postcommit_receipt=row(gate),handoff_receipt=row(P/'HANDOFF_COMPLETION.json'),
            source_sync_receipt=row(P/'SOURCE_PREMEASUREMENT_SYNC.json'),FORMAL_AUTHORIZED='YES'))
    children=[]
    for gpu in [6,3]:
        path=A/('gpu'+str(gpu)+'_parallel_discovery.log')
        with path.open('xb') as f:
            proc=subprocess.Popen(['/data/zypan/envs/ling-kda/bin/python','-B',str(P/'inputs'/RUNNER),'discovery','--gpu',str(gpu)],
                cwd=W,env=environment(plan,gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append((gpu,proc));emit(P/('GPU'+str(gpu)+'_FORMAL_PROCESS_STARTED.json'),dict(pid=proc.pid,
            gpu=gpu,gpu_uuid=UUIDS[gpu],premeasurement_commit=seal['commit'],log=str(path),time_unix=time.time()))
        print('FROZEN_SHARD_STARTED',gpu,proc.pid,flush=True)
    for gpu,proc in children:
        assert proc.wait()==0,('SHARD_EXECUTION_FAILED',gpu)
        receipt=read(A/('discovery_shards/gpu'+str(gpu)+'/discovery/raw_shard_freeze.json'))
        assert receipt['gate']=='PASS' and receipt['complete']==64
    emit(P/'TWO_SHARDS_COMPUTE_COMPLETE.json',dict(gate='PASS',unique_total_expected=128,analysis_not_yet_executed=True,
        FINAL_ACTION='WAIT_FOR_CANONICAL_MERGE_AND_ORIGINAL_ANALYSIS',L9_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO'))
    print('BOTH_SHARDS_COMPLETE_REQUIRES_MERGE_ORIGINAL_ANALYSIS_FINAL_VALIDATION',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['request','validate','advance']);args=parser.parse_args()
    try:globals()[args.mode]()
    except BaseException as error:
        path=P/('CONTROLLED_HANDOFF_'+args.mode+'_STOP.json')
        if not path.exists():emit(path,dict(error=repr(error),traceback=traceback.format_exc(),FINAL_ACTION='FAIL_AND_STOP',
            signals_sent=False,scientific_protocol_changed='NO'))
        raise
