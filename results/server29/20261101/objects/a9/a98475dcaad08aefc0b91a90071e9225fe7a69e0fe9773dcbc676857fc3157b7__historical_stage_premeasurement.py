"""Stage scientific preregistration only after every actual qualification receipt PASS."""
from pathlib import Path
import hashlib,json,tarfile
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
E=Q/'phase_b_qualification_evidence'
H=Path('/data/zypan/experiment_runs/ling_historical_runtime_replay_v1/checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1')
D=R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def cp(a,b):
    b.parent.mkdir(parents=True,exist_ok=True)
    with b.open('xb') as f:f.write(a.read_bytes())
    assert sha(a)==sha(b)
def main():
    assert read(Q/'HISTORICAL_RUNNER_SELF_REPLAY_GATE.json')['gate']=='PASS'
    assert read(E/'FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json')['gate']=='PASS'
    assert not (E/'SUPERVISOR_STOP.json').exists()
    gate_names=['SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY_GATE','RUNTIME_ENVIRONMENT_IDENTITY_GATE','BASIS_SEMANTICS_GATE',
        'HISTORICAL_K3_PARITY_GATE','HISTORICAL_JPROP_PARITY_GATE','FUTURE_KL_HARNESS_PARITY_GATE','HARDWARE_ASSIGNMENT_GATE','STORAGE_CAPACITY_GATE',
        'OBSERVABILITY_INSTRUMENTATION_TRANSPARENCY_GATE']
    for i in range(1,4):
        receipt=read(E/f'fresh_process_{i:02d}/QUALIFICATION_COMPLETE.json')
        assert receipt['gate']=='PASS' and all(receipt[n]=='PASS' for n in gate_names)
    assert read(E/'fresh_process_01/NUMERIC_FINGERPRINT.json')==read(E/'fresh_process_02/NUMERIC_FINGERPRINT.json')==read(E/'fresh_process_03/NUMERIC_FINGERPRINT.json')
    D.mkdir(parents=True)
    for name in ['preregistration.json','preregistration.md','document_manifest.json','rotation_manifest.json','runtime_manifest.json','source_identity.json','pulse_definition.md','basis_semantics_audit.md']:
        cp(H/name,D/name)
    scripts=Q/'advancement_inputs';scripts.mkdir()
    for name in ['historical_backend_discovery_runner.py','historical_stage_premeasurement.py','historical_finalize_discovery.py','historical_auto_advance_source.py']:
        cp(Path('/data/zypan')/name,scripts/name)
    dep_rows=read(E/'fresh_process_03/smoke/repeat0/result.json')['dependencies']['execution_module_files']
    selected=[row for row in dep_rows if str(row['path']).startswith('/data/zypan/worktrees/') or '/ling_historical_runtime_replay_v1/checkout/' in row['path']]
    # Compatibility field read by original discovery; no claim of full generated
    # or OS/runtime ingress closure. Scientific/project source only.
    emit(D/'historical_assets/executed_dependency_closure.json',dict(all_module_files=selected,
        scope='Frozen actually executed scientific/project source only; generated/runtime-native full closure NOT REQUIRED',
        HISTORICAL_DIRTY_WORKTREE_DISCLOSED='YES',old_dirty_worktree_execution=False))
    gates={n:'PASS' for n in gate_names};gates['FRESH_PROCESS_NUMERIC_DETERMINISM_GATE']='PASS';gates['HISTORICAL_RUNNER_SELF_REPLAY_GATE']='PASS'
    emit(D/'smoke/ALL_BLOCKING_GATES.json',dict(gate='PASS',gates=gates,
         evidence_root=str(E),not_original_bridge_auto_gate=True,FORMAL_MEASUREMENTS=0))
    files=[dict(path=str(p),sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file()]
    files += [dict(path=str(p),sha256=sha(p)) for p in sorted(scripts.rglob('*')) if p.is_file()]
    files += [dict(path=str(H/'scripts/bridge.py'),sha256=sha(H/'scripts/bridge.py'))]
    emit(D/'WHOLE_RUNTIME_PREMEASUREMENT_IDENTITY.json',dict(files=files,qualified_historical_runtime=str(H),
        asset_manifest_sha256=sha(R/'provenance_capsule_v3/scientific_asset_manifest.json'),
        original_preregistration_byte_identity=True,MODEL_SEMANTICS_CHANGED='NO',QDQ_SEMANTICS_CHANGED='NO',
        KDA_RECURRENCE_CHANGED='NO',BASIS_SEMANTICS_CHANGED='NO',DOCUMENT_SET_CHANGED='NO',ROTATION_SET_CHANGED='NO',
        HORIZON_CHANGED='NO',METRIC_CHANGED='NO',TARGET_CHANGED='NO',TOLERANCE_RELAXED='NO',ADVANCEMENT_RULE_CHANGED='NO'))
    with tarfile.open(Path(str(R)+'-authority')/'scientific-repro-historical-premeasurement.tar.gz','w:gz') as t:
        for root in [Q,D]:
            for p in sorted(root.rglob('*')):
                if p.is_file():t.add(p,arcname=str(p.relative_to(R)))
    print('SCIENTIFIC_PREMEASUREMENT_READY_NO_FORMAL_AUTHORIZATION',flush=True)
if __name__=='__main__':main()
