"""Read-only integrity verification and V3 preparation-stop reporting only."""
from common import *
import shutil

def main():
    failure=load(ROOT/'run/STOP.json')
    assert failure['stage']=='PREPARATION' and failure['error']=="KeyError('gate')"
    assert not (ROOT/'run/training').exists() and not (ROOT/'run/fresh_b').exists()
    assert not list((ROOT/'runtime').glob('*smoke.json'))
    identity=verify_inventory(ROOT/'provenance/protected_parent_identity_before.json')
    for r in load(V2/'manifests/phase_a_stop_sha256_v2.json')['files']:
        assert sha(REPO/r['path'])==r['sha256']
    historical=load(V1/'provenance/historical_l7_recipe_audit_recovery_v1.json')
    assert historical['RECOVERY_HISTORICAL_AUDIT_RESULT']=='PASS'
    apps=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'])
    assert not apps,'GPU_RECLAIM_NOT_CONFIRMED'
    shutil.copyfile(ROOT/'run/STOP.json',ROOT/'provenance/STOP_PREPARATION.json')
    parent=dict(V1_STATUS='PRETRAINING_INFRASTRUCTURE_STOP',V2_STATUS='HISTORICAL_EXACT_PARITY_FAIL',
        V2_VERDICT_CHANGED='NO',V3_HISTORICAL_EXACT_PARITY_REQUIRED='NO',
        V1_V2_PARENT_IMMUTABILITY_GATE='PASS',v1_final_commit='f24dc978c87b110933c3d13b72556978a74316ab',
        v2_final_commit=BASE_HEAD,starting_branch='exp/ling-l7r-l8-layer-aware-matched-training-v2-cleanroom',
        starting_HEAD=BASE_HEAD,V1_historical_recipe_audit='PASS',V2_runtime_reconstruction='PASS',
        V2_config_semantics='PASS',V2_model_initialization='PASS',V2_historical_exact_parity='FAIL',
        V2_Fresh_B_treatment_forwards=0,V2_formal_optimizer_updates=0,V2_AIME_generation=0,
        v1_recipe_audit_sha256=sha(V1/'provenance/historical_l7_recipe_audit_recovery_v1.json'),
        v2_verdict_sha256=sha(V2/'reports/FINAL_VERDICT.json'),protected_inventory=identity,
        current_hardware=hardware(),new_V3_scientific_outcomes=None)
    emit(ROOT/'provenance/v1_v2_parent_reference.json',parent)
    verdict=dict(EXPERIMENT=TASK,V1_STATUS='PRETRAINING_INFRASTRUCTURE_STOP',V2_STATUS='HISTORICAL_EXACT_PARITY_FAIL',
        V2_VERDICT_CHANGED='NO',V3_HISTORICAL_EXACT_PARITY_REQUIRED='NO',V1_V2_PARENT_IMMUTABILITY_GATE='PASS')
    for key in ['V3_RUNTIME_IDENTITY_GATE','V3_HISTORICAL_RECIPE_PROVENANCE_GATE','V3_LAYER_REGION_IDENTITY_GATE',
        'FRESH_B_V3_IDENTITY_GATE','GPU0_CONTEMPORARY_DETERMINISM_GATE','GPU1_CONTEMPORARY_DETERMINISM_GATE',
        'DUAL_4090_CONTEMPORARY_READINESS_GATE','V3_L7R_L8_SINGLE_FACTOR_GATE','V3_UPDATE0_MATCHED_IDENTITY_GATE',
        'L7R_SCIENTIFIC_PATH_GATE','L8_SCIENTIFIC_PATH_GATE','L8_EARLY_FIXED_H_GATE','L8_MIDDLE_LATE_TRAINABLE_GATE',
        'BASE_WEIGHT_FREEZE_GATE','V3_CHECKPOINT_RULE_IDENTITY_GATE']:
        verdict[key]='NOT_EVALUATED'
    verdict.update(FRESH_B_PRIOR_TREATMENT_FORWARD=0,PREMEASUREMENT_COMMIT_GATE='NOT_REACHED',PREMEASUREMENT_COMMIT_SHA='NOT_CREATED',
        TRAINING_HARDWARE='GPU0_ONLY_SEQUENTIAL',TRAINING_ORDER='L7R_THEN_L8',L7R_TRAINING='NOT_STARTED',L8_TRAINING='NOT_STARTED',
        L7R_SELECTED_CHECKPOINT='NOT_CREATED',L8_SELECTED_CHECKPOINT='NOT_CREATED',L7R_ROTATION_SHA256='NOT_CREATED',L8_ROTATION_SHA256='NOT_CREATED',
        FRESH_B_COMPLETE='0/16',POSITIVE_FHGAIN_COUNT='NOT_EVALUATED',MEDIAN_FHGAIN='NOT_EVALUATED',PRIMARY_ONE_SIDED_EXACT_P='NOT_EVALUATED',
        L8_DOWNSTREAM_TRANSMISSION_CONFIRMATION='NOT_EVALUATED',AIME_ADVANCEMENT_GATE='NOT_EVALUATED',V3_256K_ENDPOINT_RUNTIME_GATE='NOT_EVALUATED',
        H_AIME_COMPLETE='0/60',L7R_AIME_COMPLETE='0/60',L8_AIME_COMPLETE='0/60')
    for key in ['H_AIME_ACCURACY','L7R_AIME_ACCURACY','L8_AIME_ACCURACY','L8_VS_L7R_RESCUE','L8_VS_L7R_REGRESSION',
        'L8_VS_L7R_NET','L8_VS_L7R_EXACT_P','L8_VS_H_RESCUE','L8_VS_H_REGRESSION','L8_VS_H_NET','L8_VS_H_EXACT_P']:
        verdict[key]='NOT_EVALUATED'
    verdict.update(FROZEN_V4='NOT_REACHED',FRESH16_VERDICT_CHANGED='NO',FORMAL_PHASE3_VERDICTS_CHANGED='NO',FINAL_ACTION='STOP',
        V3_PHASE_A_STATUS='PREPARATION_IMPLEMENTATION_STOP',formal_optimizer_updates=0,disposable_optimizer_updates=0,
        FP_H_smoke_forwards=0,Fresh_B_treatment_forwards=0,AIME_generation=0,workers_exited='PASS',GPU_reclaim='PASS',
        no_scientific_retry=True,no_tolerance_created=True,no_parent_changes=True,error=failure['error'],error_stage=failure['stage'])
    emit(ROOT/'reports/FINAL_VERDICT.json',verdict)
    paragraphs=[
        ('Q1 — Contemporary repeatability','NOT EVALUATED. Neither GPU0 nor GPU1 A/B process was launched. This stop is not evidence of GPU non-determinism.'),
        ('Q2 — V2 historical FAIL','Preserved exactly. V1/V2 and protected parent artifacts were rehashed before and after; V3 does not reinterpret the V2 result or introduce tolerance.'),
        ('Q3 — Single factor','NOT EVALUATED in V3. The inherited L7R/L8 design is early learnability versus exact fixed H, but no V3 runtime-loaded matched identity was established.'),
        ('Q4 — Matched training','NOT STARTED. Both formal and disposable optimizer update counts are zero.'),
        ('Q5 — Fresh-B Final Hidden','NOT EVALUATED. Zero treatment forwards; no positive count, median or exact p exists.'),
        ('Q6 — Early H-like / middle-late retention','NOT EVALUATED; no V3 treatment effects are available.'),
        ('Q7 — AIME advancement','NOT EVALUATED. No Fresh-B verdict, so AIME was not authorized by its advancement gate.'),
        ('Q8 — L8 versus contemporary L7R AIME','NOT EVALUATED. No new generations.'),
        ('Q9 — L8 versus contemporary H AIME','NOT EVALUATED. No new generations.'),
        ('Q10 — Evidence chain','Historical Fresh16 association remains SUPPORTED and unchanged. V3 intervention, downstream fidelity and reasoning endpoints are all NOT EVALUATED. No causal claim is made.')]
    report=['# V3 Contemporary Matched Experiment — Preparation STOP Report','',
        '```text']+[f'{k} = {v}' for k,v in verdict.items()]+['```','',
        '## Stop cause','',
        'The new V3 CPU preparation script assumed a top-level `gate` key in the immutable V1 recovery recipe report. The actual report records its result under `RECOVERY_HISTORICAL_AUDIT_RESULT`. Reading the absent key raised `KeyError(\'gate\')` before the parent reference was emitted and before runtime validation, data copying, source freezing or GPU computation.',
        '',
        'This is a defect in the new V3 metadata reader, not a scientific-path numerical mismatch or a failed same-device A/B test. The failed invocation and source are retained. No automatic patch/retry, historical output comparison or numerical tolerance was applied. This STOP commit is not a PREMEASUREMENT commit and cannot authorize training.',
        '',
        'The V1 recipe audit result is independently read under its actual key for this reporting-only receipt; that does not resume preparation or satisfy the unexecuted V3 recipe gate.',
        '',
        '## Integrity and resources','',
        f"- Starting HEAD: `{BASE_HEAD}`; V3 branch: `exp/ling-l7r-l8-layer-aware-matched-training-v3-contemporary`.",
        f"- Protected file identities verified after STOP: {identity['files_checked']}; inventory SHA256 `{identity['inventory_sha256']}`.",
        '- V2 stop inventory fully revalidated; no protected parent was modified.',
        '- Compute process inventory is empty. No V3 GPU worker was started.',
        '- Runtime environment identity: not reached, so no new claim that the V2 environment remains identical is made.',
        '- Unreached gates are NOT_EVALUATED rather than falsely PASS or scientifically FAIL.',
        '',
        '## Next authority required','',
        'A continuation needs explicit authorization to correct only the V3 metadata-reading defect, preserve this failed invocation, and resume CPU preparation before any GPU computation. That correction must not change V1/V2, scientific recipe, runtime semantics, inputs, comparison strictness or outcomes.',
        '']
    for title,body in paragraphs:report.extend(['## '+title,'',body,''])
    report.extend(['## Evidence files','',
        '`scripts/prepare_phase_a.py:31` — failed top-level key lookup.',
        '`provenance/STOP_PREPARATION.json` — retained failed invocation.',
        '`provenance/v1_v2_parent_reference.json` — reporting-only immutable parent reference.',
        '`provenance/protected_parent_identity_before.json` — byte inventory.',
        '`reports/FINAL_VERDICT.json` — complete reached/unreached verdict accounting.',''])
    p=ROOT/'reports/FINAL_REPORT.md';p.parent.mkdir(exist_ok=True)
    with p.open('x') as f:f.write('\n'.join(report))
    emit(ROOT/'provenance/preparation_stop_integrity.json',dict(gate='PASS',protected=identity,compute_processes_empty=True,
        no_model_or_GPU_computation=True,formal_updates=0,disposable_updates=0,Fresh_B_treatment_forwards=0,AIME_generation=0,
        source_sha256=sha(ROOT/'scripts/prepare_phase_a.py'),historical_recipe_report_sha256=parent['v1_recipe_audit_sha256'],
        failure_sha256=sha(ROOT/'provenance/STOP_PREPARATION.json'),report_sha256=sha(p)))
    rows=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and not any(x in p.parts for x in ['run','inputs','logs','__pycache__','manifests']):
            rows.append(dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)))
    emit(ROOT/'manifests/preparation_stop_sha256.json',dict(gate='PASS',files=rows,self_excluded=True))
    print(json.dumps(dict(FINAL_ACTION='STOP',protected_files=identity['files_checked'],manifest_files=len(rows),formal_updates=0)),flush=True)

if __name__=='__main__':main()
