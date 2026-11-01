"""CPU-only provenance, protected inventory, byte-preserving copies and preregistration."""
from common import *
import random
import recipe_schema

ATTEMPT=3
ATTEMPT_ROOT=ROOT/'provenance/preparation_attempts'/f'attempt{ATTEMPT}'

def receipt(relative,value):
    p=ROOT/relative
    if p.exists():
        assert load(p)==value,('EXISTING_RECEIPT_CHANGED',str(p))
    else:emit(p,value)

def copy(p,relative):
    q=ROOT/relative;q.parent.mkdir(parents=True,exist_ok=True)
    if not q.exists():subprocess.run(['cp','--reflink=auto','--preserve=timestamps',str(p),str(q)],check=True)
    a=sha(p);assert a==sha(q)
    return dict(source=str(p),destination=str(q),bytes=q.stat().st_size,sha256=a)

def main():
    assert command(['git','rev-parse','HEAD'])=='ba96365f34098b17b64b6a88220a8c85295053f5'
    assert command(['git','branch','--show-current'])=='exp/ling-l7r-l8-layer-aware-matched-training-v3-contemporary'
    assert not (ROOT/'run/training').exists() and not (ROOT/'run/fresh_b').exists()
    assert not list((ROOT/'runtime').glob('*smoke.json'))
    anchor=ROOT/'provenance/v3_scientific_objects_before_engineering_recovery.json'
    before_identity=verify_inventory(anchor)
    schema=recipe_schema.actual()
    receipt(str((ATTEMPT_ROOT/'recipe_schema_validation.json').relative_to(ROOT)),schema)
    receipt(str((ATTEMPT_ROOT/'schema_unit_tests.json').relative_to(ROOT)),recipe_schema.tests())
    for r in load(V2/'manifests/phase_a_stop_sha256_v2.json')['files']:assert sha(REPO/r['path'])==r['sha256']
    parent_identity=verify_inventory(ROOT/'provenance/protected_parent_identity_before.json')
    protected=load(ROOT/'provenance/protected_parent_identity_before.json')['files']
    receipt(str((ATTEMPT_ROOT/'parent_identity.json').relative_to(ROOT)),parent_identity)
    receipt(str((ATTEMPT_ROOT/'v1_v2_parent_reference.json').relative_to(ROOT)),dict(V1_STATUS='PRETRAINING_INFRASTRUCTURE_STOP',V2_STATUS='HISTORICAL_EXACT_PARITY_FAIL',
        V2_VERDICT_CHANGED='NO',V3_HISTORICAL_EXACT_PARITY_REQUIRED='NO',HISTORICAL_EXACT_BITWISE_PARITY='NOT_A_V3_GATE',
        V1_V2_PARENT_IMMUTABILITY_GATE='PASS',v1_final_commit='f24dc978c87b110933c3d13b72556978a74316ab',v2_final_commit=BASE_HEAD,
        starting_branch='exp/ling-l7r-l8-layer-aware-matched-training-v2-cleanroom',starting_HEAD=BASE_HEAD,
        V1_historical_recipe_audit=schema['RECOVERY_HISTORICAL_AUDIT_RESULT'],
        V2_runtime_reconstruction='PASS',V2_config_semantics='PASS',V2_model_initialization='PASS',
        V2_historical_exact_parity='FAIL',V2_Fresh_B_treatment_forwards=0,V2_formal_optimizer_updates=0,V2_AIME_generation=0,
        v1_recipe_audit_sha256=sha(V1/'provenance/historical_l7_recipe_audit_recovery_v1.json'),v2_verdict_sha256=sha(V2/'reports/FINAL_VERDICT.json')))
    if not (ROOT/'provenance/hardware_identity.json').exists():emit(ROOT/'provenance/hardware_identity.json',hardware())
    rows=[]
    for d in ['protocol','configs']:
        for p in sorted((V2/d).glob('*.json')):
            # Scientific parent bytes retained as provenance; V3 authority is separate.
            rows.append(copy(p,('protocol/parent_reference/' if d=='protocol' else 'configs/')+p.name))
    for n in ['frozen_layer_regions.json','train_manifest.json','validation_manifest.json','checkpoint_selection_rule.json','fresh_b_selection_manifest.json','hardware_assignment.json','statistical_plan.json']:
        copy(V2/'protocol'/n,'protocol/'+n)
    for d in ['traces','runtime_repo','fresh_b_tokens']:
        for p in sorted((V2/'inputs'/d).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:rows.append(copy(p,'inputs/'+d+'/'+str(p.relative_to(V2/'inputs'/d))))
    rows.append(copy(V2/'inputs/historical_trainer.py','inputs/historical_trainer.py'))
    trains=sorted(TRACES.glob('wikitext2raw-train-*.pt'));vals=sorted(TRACES.glob('wikitext2raw-validation-*.pt'))
    assert len(trains)==64 and len(vals)==16
    for name,paths in [('train',trains),('validation',vals)]:
        frozen=load(ROOT/'protocol'/f'{name}_manifest.json')
        expected={r['document_id']:r for r in frozen['rows']}
        assert set(expected)=={p.stem for p in paths}
        for p in paths:assert sha(p)==expected[p.stem]['sha256']
    ca=load(ROOT/'configs/l7r.json');cb=load(ROOT/'configs/l8.json')
    assert {k for k in ca if ca[k]!=cb[k]}=={'condition','learnable_layers','fixed_H_layers'}
    assert ca['draw_order']==[p.stem for p in _draws(trains)]
    assert ca['steps']==100 and ca['gradient_horizon']==ca['detach_interval']==128 and ca['sequence_length']==512
    rule=load(ROOT/'protocol/checkpoint_selection_rule.json');assert rule['metric']=='primary = local_functional_relative_mse'
    assert load(ROOT/'protocol/frozen_layer_regions.json')['regions']==REGIONS
    assert ca['learnable_layers']==LAYERS and cb['fixed_H_layers']==REGIONS['early'] and cb['learnable_layers']==REGIONS['middle']+REGIONS['late']
    receipt('protocol/historical_recipe_provenance.json',dict(gate='PASS',V3_HISTORICAL_RECIPE_PROVENANCE_GATE='PASS',
         V3_LAYER_REGION_IDENTITY_GATE='PASS',V3_CHECKPOINT_RULE_IDENTITY_GATE='PASS',training_traces=64,validation_traces=16,
         copies=rows,recipe_audit_sha256=sha(V1/'provenance/historical_l7_recipe_audit_recovery_v1.json'),regions=REGIONS,
         checkpoint_rule_sha256=sha(ROOT/'protocol/checkpoint_selection_rule.json'),numerical_historical_parity_required=False,
         actual_schema_validation=schema))
    assert not (V1/'run/fresh_b').exists() and not (V2/'run/fresh_b').exists()
    fresh=load(ROOT/'protocol/fresh_b_selection_manifest.json')
    assert fresh['gate']=='PASS' and fresh['n']==16 and len(fresh['streams'])==16
    for r in fresh['streams']:
        p=ROOT/'inputs/fresh_b_tokens'/(r['document_id']+'.tokens.json');assert sha(p)==r['token_file_sha256']
        assert all(v==0 for v in r['overlap_32word_counts'].values())
    assert not fresh['selected_id_overlap'] and not fresh['selected_exact_passage_overlap']
    receipt('protocol/fresh_b_v3_identity.json',dict(FRESH_B_V3_IDENTITY_GATE='PASS',FRESH_B_PRIOR_TREATMENT_FORWARD=0,
         selection_manifest_sha256=sha(ROOT/'protocol/fresh_b_selection_manifest.json'),
         parent_selection_manifest_sha256=sha(V2/'protocol/fresh_b_selection_manifest.json'),
         tokens=[r for r in rows if '/fresh_b_tokens/' in r['destination']],exclusions_changed=False,reselection=False))
    plan=dict(task=TASK,phase='PHASE_A_CONTEMPORARY_DETERMINISM_CLOSURE',input='wikitext2raw-train-000.pt',
        input_sha256=sha(TRACES/'wikitext2raw-train-000.pt'),input_token_ids=[296,289,8023,1639],tokens=4,
        source='existing historical TRAIN/development smoke only; no Fresh16/Fresh-B/AIME/new PG19',seed=0,
        dtype='bfloat16 base/FP32 Cayley',model_config='raw immutable model config; no rope/config override',
        max_memory_gib=22,model=str(MODEL),runtime_identity='runtime/v3_runtime_identity.json',
        comparison='EXACT structured numerical evidence within each GPU; no cross-GPU bitwise gate',
        fresh_processes=['gpu0_A','gpu0_B','gpu1_A','gpu1_B'],process_order_per_gpu=['A','B'],
        forward_order=['FP','H','L7R_disposable_update0'],full_forward_scope='native 4-token cold-cache teacher forcing with immutable Phase2 layerwise wrappers and immediate H QDQ; training replay uses original trace initial teacher state',
        training_replay='immutable historical rollout_chunk, all18 layers, first4 tokens, LocalFunctional+.1State',
        update='Adam lr=.003 wd0, global clip1, exactly one disposable update',
        deterministic_policy='inherit original configure_offline_runtime/setup_seed; no new backend/algorithm changes; record actual flags',
        relevant_exact_fields=['input IDs','losses','all-layer per-token recurrent-state hashes','FP/H logits/final-hidden/local hashes','parameter init','rotation matrices','raw and clipped gradients','updated parameters','optimizer state','base before/after'],
        excluded_comparison_fields=['PID','process label','timing','allocator memory','filesystem path'],
        scientific_failure='STOP; no patch/retry/tolerance',formal_optimizer_updates_authorized=0,
        Fresh_B_forwards_authorized=0,AIME_generations_authorized=0)
    receipt('protocol/contemporary_determinism_plan.json',plan)
    receipt('protocol/v3_execution_authority.json',dict(task=TASK,training_gpu=0,training_order=['L7R','L8'],fresh_process_and_reclaim_required=True,
        V2_HISTORICAL_PARITY_VERDICT='FAIL',V3_HISTORICAL_EXACT_PARITY_REQUIRED='NO',
        formal_training_requires='all Phase A gates PASS + clean PREMEASUREMENT commit',
        Fresh_B_primary='median(FHError_L7R-FHError_L8)>0 AND strictly_positive>=12/16 AND exact one-sided p<.05',
        Fresh_B_not_supported='STOP, no AIME',AIME_advancement='SUPPORTED only',
        AIME_conditions=['H','L7R','L8'],AIME_counts=[60,60,60],AIME_new_raw=180,
        AIME_order=['H60 raw freeze','L7R60 raw freeze','L860 raw freeze','180 closure','exact Frozen V4'],
        AIME_hardware='problem-major then seed1/seed2 ordered rank modulo2; same unit allconditions same physical GPU',
        AIME_primary='L8 vs L7R',AIME_key_secondary='L8 vs H',AIME_context=262144,
        cluster_bootstrap=dict(unit='30 question clusters retaining both seeds',replicates=10000,seed=20261005,CI='95 percentile'),
        bootstrap_seed_provenance='mechanically inherit V1 frozen statistical seed before any V3 numerical computation',
        no_auto_variants=True,no_auto_retry=True,no_AIME_pilot=True,no_selection_on_confirmatory_data=True))
    runtime_env=os.environ.copy();runtime_env['PYTHONNOUSERSITE']='1'
    subprocess.run([str(PYTHON),str(ROOT/'scripts/verify_runtime.py'),'--attempt',str(ATTEMPT)],env=runtime_env,check=True)
    after_identity=verify_inventory(anchor)
    old=load(anchor)['engineering_source']
    envelope=dict(RECOVERY_TYPE='PREMEASUREMENT_ENGINEERING_ONLY',TRIGGER='V3_METADATA_READER_SCHEMA_MISMATCH',
        FIRST_STOP_PRESERVED='YES',SCIENTIFIC_PROTOCOL_CHANGED='NO',RUNTIME_SEMANTICS_CHANGED='NO',MODEL_SOURCE_CHANGED='NO',
        DATA_CHANGED='NO',FRESH_B_CHANGED='NO',STATISTICS_CHANGED='NO',AIME_POLICY_CHANGED='NO',
        V3_PREPARATION_ATTEMPT_1='FAIL_ENGINEERING_METADATA_READER',V3_PREPARATION_ATTEMPT_2='FAIL_ENGINEERING_LAUNCH_PROFILE',
        V3_PREPARATION_ATTEMPT_3='PASS',CURRENT_SUCCESSFUL_PREPARATION_ATTEMPT=ATTEMPT,
        SCIENTIFIC_FIELDS_EVALUATED_DURING_ATTEMPT_1='NO',SCIENTIFIC_GPU_MEASUREMENT_BEFORE_ATTEMPT_2='NO',
        ENGINEERING_RECOVERY_ATTEMPTS=2,SCIENTIFIC_OBJECT_IDENTITY_AFTER_ENGINEERING_FIX='PASS',
        attempt2_classification=load(ROOT/'provenance/preparation_attempts/attempt2/ENGINEERING_DEFECT_CLASSIFICATION.json'),
        attempt2_preserved_sources={n:sha(ROOT/'provenance/preparation_attempts/attempt2'/('source_'+n)) for n in ['prepare_phase_a.py','verify_runtime.py']},
        new_runtime_reader_sha256=sha(ROOT/'scripts/verify_runtime.py'),
        old_source_sha256=old['old_sha256'],new_source_sha256=sha(__file__),
        changes=['main: explicit actual V1 recovery/companion schema validation','append-only attempt paths and idempotent exact-byte copies',
                 'starting commit validation for authorized continuation','full CPU prerequisite/runtime/hash closure',
                 'restore V2 frozen PYTHONNOUSERSITE=1 launch option; append-only runtime observation paths'],
        before_identity=before_identity,after_identity=after_identity,schema_tests=recipe_schema.tests(),
        first_stop_sha256=sha(ROOT/'provenance/STOP_PREPARATION.json'),
        authority_sha256=sha(ROOT/'provenance/user_authority/ENGINEERING_ENVELOPE_V1.txt'),
        measurements=dict(formal_optimizer_updates=0,disposable_optimizer_updates=0,Fresh_B_treatment_forwards=0,AIME_generation=0,SCIENTIFIC_GPU_FORWARD_COUNT=0))
    receipt('provenance/PREMEASUREMENT_ENGINEERING_RECOVERY_ENVELOPE_V1.json',envelope)
    gates={k:'PASS' for k in ['V1_V2_PARENT_IMMUTABILITY_GATE','V3_RUNTIME_IDENTITY_GATE','V3_HISTORICAL_RECIPE_PROVENANCE_GATE',
        'V3_LAYER_REGION_IDENTITY_GATE','FRESH_B_V3_IDENTITY_GATE','V3_CHECKPOINT_RULE_IDENTITY_GATE','V3_L7R_L8_CONFIG_SINGLE_FACTOR_GATE',
        'PROTECTED_DATA_OVERLAP_GATE','SCIENTIFIC_OBJECT_HASH_CLOSURE_GATE']}
    receipt(str((ATTEMPT_ROOT/'CPU_PREPARATION_COMPLETE.json').relative_to(ROOT)),dict(gate='PASS',attempt=ATTEMPT,gates=gates,
        SCIENTIFIC_GPU_MEASUREMENT_STARTED='NO',formal_optimizer_updates=0,Fresh_B_treatment_forwards=0,AIME_generation=0))
    print(json.dumps(dict(PHASE_A_PREPARATION='PASS',protected=len(protected),copies=len(rows))),flush=True)

def _draws(paths):
    r=random.Random(0);return [r.choice(paths) for _ in range(100)]

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        p=ATTEMPT_ROOT/'FAILED_INVOCATION.json'
        if not p.exists():emit(p,dict(error=repr(e),traceback=traceback.format_exc(),scientific_GPU_measurement_started=False))
        raise
