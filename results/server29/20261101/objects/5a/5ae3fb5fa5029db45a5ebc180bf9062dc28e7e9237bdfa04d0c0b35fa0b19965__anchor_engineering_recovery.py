"""CPU-only immutable anchor, executed BEFORE patching the V3 metadata reader."""
from common import *
import shutil

def main():
    assert command(['git','rev-parse','HEAD'])=='ba96365f34098b17b64b6a88220a8c85295053f5'
    assert not (ROOT/'run/training').exists() and not (ROOT/'run/fresh_b').exists()
    assert not list((ROOT/'runtime').glob('*smoke.json'))
    assert not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
    old=load(ROOT/'reports/FINAL_VERDICT.json')
    for k in ['formal_optimizer_updates','disposable_optimizer_updates','FP_H_smoke_forwards','Fresh_B_treatment_forwards','AIME_generation']:assert old[k]==0
    inventory=load(ROOT/'provenance/protected_parent_identity_before.json')
    rows={r['path']:dict(r,role='protected_parent') for r in inventory['files']}
    objects=[(ROOT/'provenance/v1_v2_parent_reference.json','V3_parent_reference'),
        (ROOT/'provenance/STOP_PREPARATION.json','immutable_first_STOP'),
        (ROOT/'reports/FINAL_VERDICT.json','immutable_attempt1_verdict'),
        (ROOT/'reports/FINAL_REPORT.md','immutable_attempt1_report'),
        (V1/'provenance/historical_l7_recipe_audit_recovery_v1.json','historical_recipe'),
        (V1/'provenance/historical_l7_recipe_audit.json','canonical_recipe'),
        (V1/'reports/PRETRAINING_RUNTIME_STOP_VERDICT_V1.json','formal_recipe_identity_verdict'),
        (V2/'runtime/cleanroom_environment_manifest.json','V2_runtime_manifest'),
        (V2/'runtime/cleanroom_config_observation.json','V2_runtime_observation')]
    for d in ['protocol','configs']:
        objects += [(p,'frozen_'+d) for p in sorted((V2/d).glob('*.json'))]
    objects += [(p,'frozen_Fresh_B_tokens') for p in sorted((V2/'inputs/fresh_b_tokens').glob('*.json'))]
    objects += [(p,'frozen_training_validation_traces') for p in sorted((V2/'inputs/traces').glob('*.pt'))]
    objects += [(p,'scientific_source') for p in sorted((V2/'inputs/runtime_repo').rglob('*.py')) if '__pycache__' not in p.parts]
    objects += [(p,'user_execution_authority') for p in sorted((ROOT/'provenance/user_authority').glob('*.txt'))]
    for p,role in objects:rows[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),role=role)
    for r in rows.values():assert sha(r['path'])==r['sha256'],r['path']
    oldsrc=ROOT/'scripts/prepare_phase_a.py'
    dest=ROOT/'provenance/preparation_attempts/attempt1_prepare_phase_a.py';dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists();shutil.copyfile(oldsrc,dest);assert sha(dest)==sha(oldsrc)
    emit(ROOT/'provenance/v3_scientific_objects_before_engineering_recovery.json',dict(gate='PASS',files=list(rows.values()),
        V3_PREPARATION_ATTEMPT_1='FAIL_ENGINEERING_METADATA_READER',SCIENTIFIC_FIELDS_EVALUATED='NO',
        SCIENTIFIC_GPU_MEASUREMENT_STARTED='NO',SCIENTIFIC_GPU_FORWARD_COUNT=0,
        formal_optimizer_updates=0,disposable_optimizer_updates=0,Fresh_B_treatment_forwards=0,AIME_generation=0,
        engineering_source=dict(path=str(oldsrc),old_sha256=sha(oldsrc),immutable_snapshot=str(dest)),
        model_revision='e3a47d5b986e7141b6efd62597d598ebb392060d',
        tokenizer_revision='same frozen Ling model revision; bytes enumerated in protected inventory',
        V3_scientific_files_not_yet_copied='source V2 exact-byte objects anchored; no replacement by a newly designed object'))
    print('SCIENTIFIC_OBJECT_BEFORE_RECOVERY_MANIFEST=PASS; GPU_FORWARD_COUNT=0',flush=True)

if __name__=='__main__':main()
