"""Explicit historical fixture, cached-code and storage identities; no numerics."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess
C=Path(__file__).resolve().parent
OLD=Path('/data/zypan/worktrees/ling-observability-provenance-amendment-v1/checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def row(p):return dict(path=str(p),size=p.stat().st_size,sha256=sha(p))
def emit(name,x):
    with (C/name).open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
def main():
    assert not (C/'STOP.json').exists(),'NO_GATE_RETRY'
    expected={r['path']:r['sha256'] for r in read(C/'historical_preservation_before.json')['files']}
    fixture=OLD/'smoke/repeat0/result.json'
    source_name=str(fixture).replace(str(OLD),'/data01/user2/worktrees/ling-long-horizon-observability-bridge-v1/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',1)
    assert sha(fixture)==expected[source_name],'FROZEN_HISTORICAL_FIXTURE_IDENTITY_FAILURE'
    baseline=read(fixture);assert baseline['gate']=='PASS'
    emit('known_fixture_manifest.json',dict(result=row(fixture),historical_source_inventory_path=source_name,scientific_fields=['scores','target','endpoint','FP_step_hashes'],same_GPU_UUID=baseline['hardware']['gpu_uuid'],expected_FUTURE_KL_AUC=baseline['target']['TARGET_FUTURE_KL_AUC'],zero_new_formal_measurements=True))
    mods={};parents=set()
    for r in baseline['dependencies']['execution_module_files']:
        p=Path(r['path'])
        if '/.cache/huggingface/modules/' in str(p) and p.suffix=='.py':
            assert sha(p)==r['sha256'],'HISTORICAL_MODEL_REMOTE_CODE_CHANGED'
            mods[str(p)]=dict(**row(p),role='historically_imported_frozen_model_remote_code_alias')
            for parent in p.parents:
                if '/.cache/huggingface/modules' not in str(parent):break
                parents.add(parent)
    for parent in parents:
        p=parent/'__init__.py'
        if p.is_file():mods[str(p)]=dict(**row(p),role='existing_HF_remote_code_package_support_source; explicitly frozen, not OS metadata')
    emit('cached_environment_module_manifest.json',dict(files=list(mods.values()),source_snapshot_third_party_source_copied=False))
    env={k:v for k,v in baseline['hardware']['environment'].items() if v is not None}
    env['TOKENIZERS_PARALLELISM']='false'
    emit('measurement_launcher_environment.json',dict(set=env,unset=['CUBLAS_WORKSPACE_CONFIG'] if baseline['hardware']['environment']['CUBLAS_WORKSPACE_CONFIG'] is None else [],locale='record without changing numerical process locale',expected_historical_torch_flags={k:baseline['hardware'][k] for k in ['cudnn_benchmark','cudnn_deterministic','deterministic_algorithms','matmul_tf32']},NUMERICAL_POLICY_CHANGED=False,GPU6_FORMAL_AUTHORIZED='NO'))
    free=shutil.disk_usage(C).free
    # Original coherence capacity rule: remaining traces * 13,325,495,104 B
    # + 2 GiB non-trace overhead; original bridge preserves a 20 GiB reserve.
    assert len([r for r in read(C/'scientific_asset_manifest.json')['files'] if r.get('scientific_role')=='same_GPU6_frozen_FP_drive_manifest'])==4
    peak=2*1024**3;reserve=20*1024**3
    assert free>=peak+reserve,('STORAGE_CAPACITY_FAILURE',free,peak,reserve)
    emit('storage_capacity_identity.json',dict(gate='PASS',available_bytes=free,remaining_FP_drives_to_capture=0,required_new_peak_bytes=peak,safety_margin_bytes=reserve,required_total_bytes=peak+reserve,existing_FP_drive_bytes=sum(r['size'] for r in read(C/'scientific_asset_manifest.json')['files'] if r.get('scientific_role')=='same_GPU6_frozen_teacher_forced_FP_drive_chunk'),estimate_source='Historical coherence storage rule non-trace overhead plus unchanged original bridge 20 GiB reserve; four frozen drives reused, not regenerated.',full_future_logits_saved=False))
    # Preserve every old receipt, including untracked V2 evidence, by exact bytes.
    old=[]
    for p in sorted(OLD.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:old.append(row(p))
    emit('target_historical_preservation_before_qualification.json',dict(files=old,V1_STATUS='FAIL_PRESERVED',V2_STATUS='FAIL_PRESERVED',V2_RUNTIME_INGRESS_ARCHITECTURE='RETIRED_AS_OVERBROAD_FOR_FORMAL_EXECUTION',SUPERVISOR_CLEANUP_LEDGER_REMAINING=1))
    print('V3_CANDIDATE_INPUTS_READY_NOT_QUALIFIED',len(old),'protected target historical files',flush=True)
if __name__=='__main__':main()
