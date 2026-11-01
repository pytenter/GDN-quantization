"""Reuse the fresh V2 deep audit; refresh only execution-relevant identities."""
import os
import time
from pathlib import Path
from v3_common import *

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    RUNS.mkdir(exist_ok=True)
    assert not list(RUNS.glob('*.LAUNCH_AUTHORIZATION.json'))
    baseline=checked(V2/'SOURCE_AND_WEIGHT_IDENTITY.json'); vf=checked(V2/'V2_CODE_AND_ASSET_FREEZE.json')
    assert vf['code_commit']=='4748c03a3ebd427a9dce7e6e0c6b120c72af4ad5'
    for p,h in vf['code_sha256'].items():
        assert digest(p)==h and digest(CODE/Path(p).name)==h,'V2_BASELINE_CHANGED'
    manifest=read(V2/'RESULT_AND_DELIVERY_HASHES.json')['files']
    preserved=[p for p in V2.rglob('*') if p.is_file() and (p.name.endswith('.STOP.json') or p.name=='V2_FAIL_CLOSED_STOP.json')]
    preserved+=[OLD/'PERSISTENT_FAIL_CLOSED_STOP.json']
    assets={}; history={}
    for p in preserved:
        h=digest(p)
        if p.is_relative_to(V2): assert h==manifest[str(p.relative_to(V2))]
        else: assert h==baseline['actual_identities'][str(p)]['sha256']
        history[str(p)]=h
    pf=read(OLD/'PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json')
    paths={p for p in pf['dependencies'] if p.endswith('.py') or '/models/' in p or '/diagnostic_inputs/' in p}
    paths.update(str(OLD/n) for n in ('PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json','PROSPECTIVE_DOCUMENT_SELECTION.json',
        'PHASE_B_PAIR_METHODS_FREEZE.json','PHASE_B_CALIBRATION_MATCHED_TABLES.json','PHASE_B_CALIBRATION_CONFIG.json','SOURCE_IDENTITIES.json'))
    sel=read(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json')
    expected={'pg19-test-860':(6,'H_to_L7_a25'),'pg19-test-5956':(3,'H_to_L7_a50')}
    assert len(sel['documents'])==2
    for e in sel['documents']:
        g,b=expected[e['document']]
        assert e['gpu']==g and e['pair']['condition_a']=='H_to_L6_a50' and e['pair']['condition_b']==b
        assert e['predicted_winners']['P1']=='H_to_L6_a50' and e['predicted_winners']['L7_TOTAL_ORIGINAL']==b
        paths.update(sel['model_and_rotation_identities']['rotations'][c]['path'] for c in ('H_to_L6_a50',b))
    weights={p for p in baseline['actual_identities'] if p.endswith('.safetensors')}
    assert len(weights)==32
    paths.update(weights)
    for p in sorted(paths):
        previous=baseline['actual_identities'][p]; s=Path(p).stat()
        assert s.st_size==previous['bytes'] and s.st_mtime_ns==previous['mtime_ns'],('V2_AUDIT_FRESHNESS_CHANGED',p)
        if not p.endswith('.safetensors'): assert digest(p)==previous['sha256'],('EXECUTION_ASSET_CHANGED',p)
        assets[p]=previous
    assert digest(CODE/'persistent_engine_v1.py')==ENGINE_SHA
    emit(OUT/'SOURCE_AND_FROZEN_SELECTION_IDENTITY.json',dict(status='PASS',epoch=time.time(),assets=assets,
        historical_STOP_sha256=history,v2_final_audit_sha256=digest(V2/'FINAL_PRESERVATION_AND_RELEASE_AUDIT.json'),
        v2_identity_sha256=digest(V2/'SOURCE_AND_WEIGHT_IDENTITY.json'),v2_measurement_commit=vf['code_commit'],
        base_numeric_engine_sha256=ENGINE_SHA,weight_policy='Reuse V2 actual full SHA; verify all32 size/mtime identity, no repeated deep rehash',
        source_rotation_data_selection_fresh_rehash=True,selection_reused_not_recomputed=True,selection=sel))
    emit(OUT/'GPU_GUARD_REPAIR_REPORT.json',dict(status='CODE_REPAIR_PENDING_CPU_AND_REAL_GPU_TESTS',
        historical_failure_cause_not_proven=True,v2_problem='target compute PIDs + raw fuser holder PIDs required to be only own PID; no failing hw snapshot saved',
        v3_change='Exact UUID NVML compute PID/starttime binding; target and shared fuser are raw diagnostic evidence, not compute ownership',
        no_cross_GPU_peer_compute_allowlist=True,query_failure_or_unknown_target_context_rejected=True,
        evidence_persisted_before_assertion=True,strict_deterministic_and_engine_unchanged=True,
        GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None))
    print('V3_RELEVANT_ASSET_IDENTITIES_PASS',len(assets),'WEIGHT_METADATA',len(weights))

if __name__=='__main__': main()
