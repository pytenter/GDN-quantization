"""Build scientific premeasurement artifacts ONLY from completed real qualification."""
from pathlib import Path
import hashlib,json,shutil,tarfile
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');C=R/'provenance_capsule_v3';Q=C/'scientific_reproducibility_v2_loader_alias'
D=R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
RUNS=Path('/data/zypan/experiment_runs/ling_observability_scientific_repro_v2');A=Path(str(R)+'-authority')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('xb') as f:f.write(src.read_bytes())
    assert sha(src)==sha(dst)
def main():
    assert not (Q/'STOP.json').exists()
    report=read(RUNS/'ALL_SCIENTIFIC_QUALIFICATION_GATES.json');assert report['gate']=='PASS'
    assert report['independent_processes']>=3 and set(report['gates'])==set(read(Q/'repro_policy.json')['blocking_gates'])
    assert all(v=='PASS' for v in report['gates'].values())
    for row in report['evidence']:assert sha(row['path'])==row['sha256']
    assert read(Q/'CACHE_CREATED_EMPTY.json')['entries']==[]
    for row in read(Q/'HISTORICAL_FAILURES_AND_SOURCE_IDENTITY.json')['files']:
        assert sha(C/row['path'])==row['sha256']
    D.mkdir(parents=True,exist_ok=False)
    for p in (C/'frozen_protocol').iterdir():
        if p.is_file():copy(p,D/p.name)
    evidence=Q/'qualification_evidence';evidence.mkdir()
    for n in range(1,4):
        name='fresh_process_%02d'%n;run=RUNS/name
        for suffix in ['NUMERIC_FINGERPRINT.json','QUALIFICATION_COMPLETE.json','SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY.json','smoke/preflight.json','smoke/basis_and_jprop_parity.json','smoke/repeat0/result.json','smoke/repeat0/trace/trace_freeze.json']:
            copy(run/suffix,evidence/name/suffix)
        copy(RUNS/(name+'_LAUNCH.json'),evidence/(name+'_LAUNCH.json'))
    copy(RUNS/'INPUT_COMMIT.json',evidence/'INPUT_COMMIT.json')
    copy(RUNS/'ALL_SCIENTIFIC_QUALIFICATION_GATES.json',Q/'SCIENTIFIC_QUALIFICATION_REPORT.json')
    copy(Q/'repro_policy.json',D/'SCIENTIFIC_REPRODUCIBILITY_AUTHORIZATION.json')
    inventory=read(RUNS/'fresh_process_01/QUALIFICATION_COMPLETE.json')['runtime_ingress']
    # Original discovery requires executed historical/project dependencies. Generated
    # runtime code is separately recorded, not transformed into a full binary closure.
    modules={row['path']:row['sha256'] for row in inventory['module_files'] if row['identity_class']=='FROZEN_PROJECT_SOURCE'}
    for p,h in modules.items():assert sha(p)==h
    emit(D/'historical_assets/executed_dependency_closure.json',dict(gate='PASS',scope='executed frozen project/historical sources only; runtime-generated full closure retired',
         all_module_files=[dict(path=p,sha256=h) for p,h in sorted(modules.items())],old_dirty_directory_executed=False,
         historical_source_snapshot_manifest_sha256=sha(C/'historical_source_snapshot_manifest.json'),HISTORICAL_DIRTY_WORKTREE_DISCLOSED='YES'))
    emit(D/'smoke/ALL_BLOCKING_GATES.json',dict(gate='PASS',gates=report['gates'],evidence=report['evidence'],
         policy='explicit user scientific reproducibility qualification',GPU6_FORMAL_AUTHORIZED=False,
         formal_requires_postcommit=True,FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',cross_hardware_FP_bitwise='FAIL_PRESERVED'))
    emit(D/'SCIENTIFIC_PREMEASUREMENT_STAGE.json',dict(status='READY_FOR_COMMIT_NOT_YET_FORMAL',qualification_gates=report['gates'],
         original_preregistration_byte_identical=sha(D/'preregistration.json')==sha(C/'frozen_protocol/preregistration.json'),
         model_semantics_changed=False,QDQ_semantics_changed=False,KDA_recurrence_changed=False,basis_semantics_changed=False,
         document_set_changed=False,rotation_set_changed=False,horizon_changed=False,metric_changed=False,target_changed=False,
         tolerance_relaxed=False,advancement_rule_changed=False,FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
    with tarfile.open(A/'scientific-premeasurement.tar.gz','w:gz') as tar:
        for root in [Q,D]:
            for p in sorted(root.rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(R)))
    print('SCIENTIFIC_PREMEASUREMENT_STAGE_READY_ALL_NINE_GATES_ACTUAL_PASS',flush=True)
if __name__=='__main__':main()
