"""Wait for GPU3 references; freeze prospective science; run GPU3 postcommit only."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time,traceback
W=Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1')
A=W/'hardware_expansion_gpu3_v1'
P=A/'prospective_parallel'
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
FD=R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
OLD=Path('/data/zypan/worktrees/ling-error-coherence-diagnosis-v1/experiments/LING_ERROR_COHERENCE_DIAGNOSIS_V1')
BASE='f57fb963e9f71e0ae12a4a9a0939c44bd6d2287e'
PARENT='e428b5deb0579b926b2ad098a8699b3f4f2233f0'
GPU6_UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'
GPU3_UUID='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'
RUNNER=Path('/data/zypan/gpu3_parallel_discovery_v1_guard_identity_v2.py')
TESTS=Path('/data/zypan/gpu3_parallel_discovery_tests.py')
EXPECTED_RUNNER_SHA='3b0a8d9e583cbd34e2a9d91f3a42d93357f080f7f1c10edacb28dfb155f4d94f'
EXPECTED_TEST_SHA='69fdd435cf1da52dd3f17532208ff451385d3e1e1fe5c99d500fb9274f7a450c'
OLD_RAW_SHA='e5b99c632d053813488bacbef2b659f979747d5099ecf74f9f9c02cdbe090a2c6'
OLD_SOURCE_HEAD='d9e23eeee086f527e7ad4e488ab9a9c99d71cc93'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def row(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*args,cwd=W):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(source.read_bytes())
    assert sha(dest)==sha(source)
def regression(runner,tests):
    # Same pure suite, explicitly bound to the exact v2 candidate; no model imported.
    code="import importlib.util,unittest; s=importlib.util.spec_from_file_location('fixture',"+repr(str(tests))+"); t=importlib.util.module_from_spec(s); s.loader.exec_module(t); s=importlib.util.spec_from_file_location('candidate',"+repr(str(runner))+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); t.m=m; result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(t)); assert result.wasSuccessful()"
    # The fixture's original top-level import also remains a frozen engineering input.
    assert sha(Path('/data/zypan/gpu3_parallel_discovery_v1.py'))!=''
    result=subprocess.run(['/data/zypan/envs/ling-kda/bin/python','-B','-c',code],capture_output=True,text=True)
    assert result.returncode==0,('PARALLEL_REGRESSION_FAIL',result.stdout,result.stderr)
    return dict(gate='PASS',test_count=11,stdout=result.stdout,stderr=result.stderr,
        candidate_sha256=sha(runner),fixture_sha256=sha(tests),no_model_loaded=True,FORMAL_MEASUREMENTS=0)

def main():
    print('WAITING_FOR_TWO_ACTUAL_GPU3_32K_FP_REFERENCES',flush=True)
    manifest=A/'reference_preparation/FP_REFERENCE_MANIFEST.json'
    while not manifest.exists():
        assert not (A/'reference_preparation/STOP.json').exists()
        assert not (A/'REFERENCE_PREPARATION_SUPERVISOR_STOP.json').exists()
        assert Path('/proc/2738631').exists(),'REFERENCE_SUPERVISOR_EXITED_WITHOUT_COMPLETION'
        time.sleep(10)
    while Path('/proc/2738631').exists() or Path('/proc/2739499').exists():time.sleep(1)
    assert not (A/'reference_preparation/STOP.json').exists()
    assert not (A/'REFERENCE_PREPARATION_SUPERVISOR_STOP.json').exists()
    assert not (P/'ENGINEERING_STOP.json').exists()
    assert git('rev-parse','HEAD')==BASE and not git('status','--porcelain=v1','--untracked-files=no')
    assert git('rev-parse','HEAD',cwd=R)==PARENT
    fp=read(manifest);assert fp['gate']=='PASS' and fp['FP_REFERENCES_MIXED_ACROSS_HARDWARE']=='NO'
    assert fp['hardware']['physical_gpu']==3 and fp['hardware']['gpu_uuid']==GPU3_UUID
    assert {r['document'] for r in fp['references']}=={'pg19-test-5396','pg19-test-26493'}
    determinism=read(A/'FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json')
    assert determinism['gate']=='PASS' and determinism['repeat_count']==3
    assert all(sha(r['path'])==r['sha256'] for r in determinism['files'])
    assert len({r['sha256'] for r in determinism['files']})==1
    amendment=read(A/'HARDWARE_ASSIGNMENT_AMENDMENT.json')
    assignment=amendment['proposed_document_assignment']
    assert assignment=={'pg19-test-37702':6,'pg19-test-5396':3,'pg19-test-28988':6,'pg19-test-26493':3}
    for item in amendment['parent_protocol_identity']:assert sha(item['path'])==item['sha256']
    assert sha(RUNNER)==EXPECTED_RUNNER_SHA and sha(TESTS)==EXPECTED_TEST_SHA
    copy(RUNNER,P/'inputs'/RUNNER.name);copy(TESTS,P/'inputs'/TESTS.name)
    # Preserve the first orchestration draft as fixture-only input; never execute it scientifically.
    copy(Path('/data/zypan/gpu3_parallel_discovery_v1.py'),P/'inputs/gpu3_parallel_discovery_v1.py')
    copy(Path(__file__),P/'inputs'/Path(__file__).name)
    emit(P/'PARALLEL_REGRESSION.json',regression(P/'inputs'/RUNNER.name,P/'inputs'/TESTS.name))
    references={}
    for ref in fp['references']:
        assert ref['physical_gpu']==3 and ref['gpu_uuid']==GPU3_UUID and ref['FP_CAPTURE_REPLAY_BITWISE_GATE']=='PASS'
        trace=Path(ref['trace_path']);assert sha(trace)==ref['trace_sha256']
        files=[row(trace)]+[chunk['artifact'] for chunk in ref['chunks']]
        for item in files:assert sha(item['path'])==item['sha256']
        references[ref['document']]=dict(physical_gpu=3,gpu_uuid=GPU3_UUID,trace_path=str(trace),trace_sha256=sha(trace),
            files=files,FP_CAPTURE_REPLAY_BITWISE_GATE='PASS',origin='new independent GPU3 capture',origin_receipt=row(A/'reference_preparation'/(ref['document']+'_FP_REFERENCE.json')))
    raw=OLD/'discovery/raw_freeze.json'
    # Independently checked against the clean historical source Git HEAD and disk.
    assert sha(raw)==OLD_RAW_SHA,'HISTORICAL_GPU6_REFERENCE_ORIGIN_IDENTITY_CHANGED'
    raw_frozen=read(raw);assert raw_frozen['complete']==64
    old_files={item['path']:item['sha256'] for item in raw_frozen['files']}
    for doc in ['pg19-test-37702','pg19-test-28988']:
        trace=OLD/'discovery/traces'/doc/'trace_freeze.json';drive=read(trace)
        original=OLD/'discovery/unit_results'/(doc+'__H.json')
        assert sha(original)==old_files[str(original)]
        unit=read(original)
        identity=unit['identity'];assert unit['gate']=='PASS' and unit['FP_CAPTURE_REPLAY_BITWISE']=='PASS'
        assert identity['physical_gpu']==6 and identity['gpu_uuid']==GPU6_UUID and identity['doc_id']==doc
        assert identity['trace_sha256']==sha(trace)
        files=[row(raw),row(trace),row(original)]+[chunk['artifact'] for chunk in drive['chunks']]
        for item in files:assert sha(item['path'])==item['sha256']
        references[doc]=dict(physical_gpu=6,gpu_uuid=GPU6_UUID,trace_path=str(trace),trace_sha256=sha(trace),files=files,
            FP_CAPTURE_REPLAY_BITWISE_GATE='PASS',origin='retained same-GPU6 historical drive',origin_receipt=row(original),
            historical_source_head=OLD_SOURCE_HEAD,historical_source_tracked_clean=True,historical_source_git_blob_raw_freeze_sha256=OLD_RAW_SHA)
    identity_files=list(amendment['parent_protocol_identity'])
    for path in [manifest,A/'GPU3_LAUNCH_IDENTITY.json',A/'HARDWARE_ASSIGNMENT_AMENDMENT.json',A/'FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json',
        Q/'HISTORICAL_RUNNER_LAUNCH_IDENTITY.json',Q/'HISTORICAL_RUNNER_SELF_REPLAY_GATE.json',
        Q/'phase_b_qualification_evidence/FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json',
        R/'provenance_capsule_v3/scientific_asset_manifest.json',Q/'HISTORICAL_EXECUTED_PACKAGE_IDENTITY.json',
        FD/'historical_assets/executed_dependency_closure.json']:
        identity_files.append(row(path))
    blocking=['SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY_GATE','RUNTIME_ENVIRONMENT_IDENTITY_GATE','BASIS_SEMANTICS_GATE',
        'HISTORICAL_K3_PARITY_GATE','HISTORICAL_JPROP_PARITY_GATE','FUTURE_KL_HARNESS_PARITY_GATE','FP_REFERENCE_PAIRING_GATE',
        'HARDWARE_ASSIGNMENT_GATE','STORAGE_CAPACITY_GATE']
    for run in ['fresh_process_01','fresh_process_02','fresh_process_03']:
        path=A/'qualification'/run/'QUALIFICATION_COMPLETE.json';receipt=read(path)
        assert receipt['gate']=='PASS' and all(receipt[key]=='PASS' for key in blocking)
        identity_files.append(row(path));identity_files.append(row(A/'qualification'/run/'NUMERIC_FINGERPRINT.json'))
    plan=dict(experiment='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',parent_premeasurement_commit=PARENT,
        scientific_protocol_changed='NO',FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',document_assignment=assignment,
        primary_anchor=8192,secondary_anchor=2048,endpoint=32768,total_units=128,units_per_gpu=64,
        all_rotations_and_anchors_of_document_same_gpu=True,metrics_and_thresholds='exact byte-identical parent preregistration',
        references=references,identity_files=identity_files,
        launch_profiles={'3':read(A/'GPU3_LAUNCH_IDENTITY.json'),'6':read(Q/'HISTORICAL_RUNNER_LAUNCH_IDENTITY.json')},
        deployment_semantics='persistent INT8; every-token low-bit writeback; no delayed QDQ/FP buffer/mixed precision',
        FINAL_ACTION='STOP_AFTER_DISCOVERY',L9_TRAINING_EXECUTED='NO',NEW_ROTATION_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO',FREE_GENERATION_EXECUTED='NO',
        advancement='GPU3 postcommit first; controlled original GPU6 pair boundary; actual handoff validation; GPU6 postcommit; authorize frozen shards')
    emit(P/'SCIENTIFIC_PLAN.json',plan)
    for item in identity_files:assert sha(item['path'])==item['sha256']
    paths=[str(path.relative_to(W)) for path in sorted(A.rglob('*')) if path.is_file() and path.suffix in ['.py','.json','.log']]
    subprocess.run(['git','add','--sparse','--',*paths],cwd=W,check=True)
    diff=git('diff','--cached','--name-status').splitlines()
    assert diff and all(s.startswith('A\thardware_expansion_gpu3_v1/') for s in diff),'APPEND_ONLY_SCOPE_VIOLATION'
    tree=git('write-tree','--missing-ok')
    assert all(s.startswith('A\thardware_expansion_gpu3_v1/') for s in git('diff-tree','--no-commit-id','--name-status','-r',BASE,tree).splitlines())
    env=os.environ.copy();env.update(GIT_AUTHOR_NAME='Codex',GIT_AUTHOR_EMAIL='codex@local',GIT_COMMITTER_NAME='Codex',GIT_COMMITTER_EMAIL='codex@local')
    commit=subprocess.check_output(['git','commit-tree',tree,'-p',BASE,'-m','ling: freeze prospective two-GPU scientific plan and same-GPU FP inventories'],cwd=W,env=env,text=True).strip()
    subprocess.run(['git','update-ref','refs/heads/exp/ling-observability-gpu3-expansion-v1',commit,BASE],cwd=W,check=True)
    assert not git('status','--porcelain=v1','--untracked-files=no') and git('rev-parse','HEAD',cwd=R)==PARENT
    inputs=[row(path) for path in sorted((P/'inputs').glob('*'))]+[row(P/'SCIENTIFIC_PLAN.json'),row(P/'PARALLEL_REGRESSION.json')]
    emit(P/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json',dict(commit=commit,parent=BASE,input_files=inputs,
        formal_measurements_gpu3=0,GPU3_FORMAL_AUTHORIZED='NO',parent_GPU6_measurements_preserved=True))
    print('PARALLEL_SCIENTIFIC_PREMEASUREMENT_COMMIT',commit,flush=True)
    launch=plan['launch_profiles']['3']['environment'];env=os.environ.copy()
    for k,v in launch.items():
        if v is None:env.pop(k,None)
        else:env[k]=v
    with (A/'gpu3_parallel_postcommit.log').open('xb') as f:
        result=subprocess.run(['/data/zypan/envs/ling-kda/bin/python','-B',str(P/'inputs'/RUNNER.name),'postcommit','--gpu','3'],cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,('GPU3_POSTCOMMIT_FAIL',result.returncode)
    assert read(A/'postcommit_parallel/gpu3/MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json')['gate']=='PASS'
    print('GPU3_POSTCOMMIT_PASS_WAITING_CONTROLLED_HANDOFF_NO_FORMAL_LAUNCH',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        stop=P/'ENGINEERING_STOP.json'
        if not stop.exists():emit(stop,dict(error=repr(error),traceback=traceback.format_exc(),FINAL_ACTION='FAIL_AND_STOP',GPU3_FORMAL_AUTHORIZED='NO'))
        raise
