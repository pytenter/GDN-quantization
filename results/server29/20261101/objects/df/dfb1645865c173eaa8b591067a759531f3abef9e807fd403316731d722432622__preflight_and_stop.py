"""CPU-only, fail-closed inventory. Never imports Torch or reads document content.

This is a new experiment's preflight, not a parent resume or cumsum repair.
The authoritative predecessor evidence stays at its original absolute paths.
"""
import hashlib, json, os, subprocess, time
from pathlib import Path

TASK='QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1'
REPO=Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1')
ROOT=REPO/'experiments'/TASK
OLD_REPO=Path('/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1')
PARENT=OLD_REPO/'experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1'
PREDECESSOR=OLD_REPO/'experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,data):
    path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def text(p,data):
    path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(data)
def seal_check(root):
    manifest=root/'FINAL_SHA256SUMS';rows=[]
    for line in manifest.read_text().splitlines():
        expected,relative=line.split(None,1);relative=relative.lstrip('*')
        path=root/relative
        actual=sha(path) if path.is_file() else None
        rows.append(dict(path=str(path),relative_path=relative,expected_sha256=expected,actual_sha256=actual,PASS=actual==expected))
    return dict(root=str(root),manifest_path=str(manifest),manifest_sha256=sha(manifest),
                sealed_artifact_count=len(rows),all_exact=all(r['PASS'] for r in rows),rows=rows)

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only preflight requires CUDA_VISIBLE_DEVICES empty')
    parent_check=seal_check(PARENT);old_check=seal_check(PREDECESSOR)
    seal_ok=parent_check['all_exact'] and old_check['all_exact']
    checks=dict(PARENT_SEAL_INTEGRITY_GATE='PASS' if seal_ok else 'FAIL',parent=parent_check,engineering_predecessor=old_check,
                inspected_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),original_paths_are_authoritative=True)
    save('parent_seal_check.json',checks)
    parent_verdict=read(PARENT/'analysis/final_verdict.json')
    old_verdict=read(PREDECESSOR/'provenance/final_verdict.json')['verdict']
    preserve_parent=parent_verdict['TASK_STATUS']=='STOP_STRICT_DETERMINISTIC_RUNTIME_UNAVAILABLE' and parent_verdict['QWEN_RUNTIME_IDENTITY_GATE']=='FAIL'
    preserve_old=old_verdict['DETERMINISTIC_CUMSUM_COMPATIBILITY_GATE']=='FAIL' and old_verdict['PARENT_PROPAGATION_EXPERIMENT_AUTHORIZED_TO_RESUME']=='NO' and old_verdict['FINAL_ACTION']=='FAIL_AND_STOP'
    docs=read(PARENT/'protocol/frozen_fresh_documents.json')
    bank=read(PARENT/'protocol/frozen_rotation_bank.json')
    phase_a=read(PARENT/'protocol/preregistration_repaired_001.json')
    metrics=read(PARENT/'protocol/metric_definitions.json')
    recurrence=read(PARENT/'provenance/GDN_RECURRENCE_SOURCE_AUDIT.json')
    quantizer=read(PARENT/'provenance/quantizer_identity.json')
    native=read(PARENT/'provenance/native_basis_source_mapping.json')
    runtime=read(PARENT/'provenance/runtime_identity.json')
    anchors=read(PARENT/'provenance/anchor_recovery.json')
    rotation=read(PARENT/'provenance/rotation_parameterization_identity.json')
    source_checks=[]
    for path,expected in phase_a['source_sha256'].items():
        actual=sha(path)
        source_checks.append(dict(path=path,expected_sha256=expected,actual_sha256=actual,PASS=actual==expected))
    source_ok=all(r['PASS'] for r in source_checks)
    # These are frozen manifest metadata, not confirmation documents or their tokens.
    documents_frozen=(docs.get('status') not in ['NOT_SELECTED',None] and docs.get('count',0)>=4 and phase_a.get('scientific_docs_selected') is True)
    bank_frozen=(bank.get('ROTATION_BANK_COUNT')==12 and bank.get('status') not in ['NOT_CREATED',None] and phase_a.get('rotation_bank_created') is True)
    all_inputs_frozen=documents_frozen and bank_frozen and phase_a.get('full_premeasurement_freeze_complete') is True and source_ok
    inventory=dict(EXPERIMENT=TASK,PARENT_EXPERIMENT=parent_verdict['EXPERIMENT'],ENGINEERING_PREDECESSOR=old_verdict['EXPERIMENT'],
        parent_experiment_path=str(PARENT),engineering_predecessor_path=str(PREDECESSOR),
        parent_STOP_verdict=parent_verdict,old_cumsum_closure_verdict=old_verdict,
        sealed_artifact_counts=dict(parent=parent_check['sealed_artifact_count'],old_closure=old_check['sealed_artifact_count']),
        sealed_hash_manifests={str(PARENT/'FINAL_SHA256SUMS'):parent_check['manifest_sha256'],str(PREDECESSOR/'FINAL_SHA256SUMS'):old_check['manifest_sha256']},
        full_sealed_hash_rows='parent_seal_check.json',
        rotation_bank_definition=dict(expected_count=12,actual_frozen_manifest=bank,
            source_task_definition='R00 H;R01 C5R;R02 C6R;R03 C7;R04..07 0.2/0.4/0.6/0.8 A_C6R;R08 0.5 A_C7;R09 0.5 A_C5R;R10 0.5 A_C6R+0.5 A_C7;R11 0.5 A_C6R+0.5 A_C5R',
            definition_is_not_an_artifact_freeze=True),
        discovery_doc_definition=dict(source='PG19 TEST / independent long text; original deterministic exclusion ordering; first4 of8 eligible docs; at least32769 frozen Qwen token IDs/doc',actual_manifest=docs,actual_selected_count=docs.get('count',0)),
        confirmation_doc_definition=dict(source='next4 of the original eligible8; protected, no raw documents or token files read',content_read=False,executed=0),
        parent_phase_a_preregistration=phase_a,
        model_checkpoint_identity=runtime.get('model_identity'),runtime_identity_reused=runtime,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_files_sha256=source_checks,quantizer_identity=quantizer,
        rotation_identity=dict(anchor_recovery=anchors,parameterization_audit=rotation,derived_bank_created=False),
        recurrent_layer_ids=recurrence['recurrent_layer_ids'],state_shape=recurrence['state_shape'],state_dtype='torch.float32',
        native_basis_mapping=native,production_recurrence_source_ranges=[{k:v for k,v in r.items() if k!='source'} for r in recurrence['source_sections']],
        parent_metric_and_reducer_definition=metrics,
        model_or_rotation_numerical_audits_repeated=False,model_loaded=False,capture_started=False,replay_started=False,
        CONFIRMATION_DOCS_READ='NO',CONFIRMATION_DOCS_EXECUTED=0)
    save('preflight_parent_inventory.json',inventory)
    frozen=dict(FROZEN_INPUT_IDENTITY_GATE='PASS' if all_inputs_frozen else 'FAIL',
        discovery_documents_prospectively_frozen=documents_frozen,rotation_bank_prospectively_frozen=bank_frozen,
        full_parent_premeasurement_freeze_complete=phase_a.get('full_premeasurement_freeze_complete'),
        PREMEASUREMENT_COMMIT_SHA=phase_a.get('PREMEASUREMENT_COMMIT_SHA'),
        documents_manifest=dict(path=str(PARENT/'protocol/frozen_fresh_documents.json'),sha256=sha(PARENT/'protocol/frozen_fresh_documents.json'),content=docs),
        rotations_manifest=dict(path=str(PARENT/'protocol/frozen_rotation_bank.json'),sha256=sha(PARENT/'protocol/frozen_rotation_bank.json'),content=bank),
        distinction='Frozen anchor checkpoints and a written12-condition mathematical design are not an instantiated/hash-frozen12-condition bank. A document-selection rule is not4 frozen Discovery documents.',
        no_document_selection_performed=True,no_rotation_bank_generated=True,confirmation_content_read=False)
    save('frozen_input_identity.json',frozen)
    if all_inputs_frozen and seal_ok:
        raise RuntimeError('This inventory-only stop script must not claim a scientific run; stop and request next engineering implementation.')
    # Stop at G0/G1. No later phase artifacts may be mistaken for executed gates.
    later=['DRIVE_FACTORIZATION_GATE','CAPTURE_INTEGRITY_GATE','FP_REPLAY_PARITY_GATE',
           'PHYSICAL_TRANSITION_ROTATION_INVARIANCE_GATE','COUNTERFACTUAL_DECOMPOSITION_GATE',
           'REPLAY_DETERMINISM_GATE','STREAMING_REDUCER_PARITY_GATE']
    verdict=dict(EXPERIMENT=TASK,PARENT_EXPERIMENT=parent_verdict['EXPERIMENT'],ENGINEERING_PREDECESSOR=old_verdict['EXPERIMENT'],
        EVIDENCE_TYPE='PROSPECTIVE_CONTROLLED_RECURRENT_MECHANISM_DISCOVERY',
        PARENT_STOP_PRESERVED='YES' if preserve_parent and parent_check['all_exact'] else 'NO',
        OLD_CUMSUM_CLOSURE_FAIL_PRESERVED='YES' if preserve_old and old_check['all_exact'] else 'NO',
        PARENT_SEAL_INTEGRITY_GATE='PASS' if seal_ok else 'FAIL',FROZEN_INPUT_IDENTITY_GATE='FAIL',
        **{k:'NOT_EVALUATED' for k in later},DISCOVERY_COMPLETE='0/48',HORIZON_2048_COMPLETE='0/48',
        HORIZON_8192_COMPLETE='0/48',HORIZON_32768_COMPLETE='0/48',
        SCIENTIFIC_SCOPE='INTRINSIC_RECURRENT_ERROR_DYNAMICS_UNDER_FROZEN_EXTERNAL_DRIVE',
        FULL_MODEL_CLOSED_LOOP_PROPAGATION='NOT_EVALUATED',PARENT_PROPAGATION_EXPERIMENT_AUTHORIZED_TO_RESUME='NO',
        CONFIRMATION_DOCS_READ='NO',CONFIRMATION_DOCS_EXECUTED=0,NEW_TRAINING='NO',OPTIMIZER_STEP=0,
        NEW_C8_TRAINING_AUTHORIZED='NO',NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,
        DISCOVERY_ONLY='YES',CONFIRMATION_AUTHORIZED_AUTOMATICALLY='NO',FINAL_ACTION='FAIL_AND_STOP')
    reason='Parent sealed evidence does not contain selected/hash-frozen Discovery documents or an instantiated/hash-frozen12-condition rotation bank. New task sections4, G1 and31 require immediate stop and forbid selecting/building replacements.'
    save('final_verdict.json',dict(verdict=verdict,stop_reason=reason,stop_phase='Phase0 / G1',gpu_computation_started=False,
        parent_scientific_resume_started=False,scientific_measurements=0,old_experiment_files_modified=0))
    save('execution_status.json',dict(phase0='COMPLETE_WITH_G1_FAILURE',phase1_through12='NOT_STARTED',
        preregistration_written=False,drive_schema_written=False,drive_capture_attempts=0,drive_shards=0,
        replay_units=0,reducer_metrics=0,result_CSVs_created=False,missing_later_deliverables_reason='Fail-closed before execution, not silent truncation or missing scientific data'))
    report='\n\n'.join(k+' =\n'+str(v) for k,v in verdict.items())+'\n\n'
    report+='# Independent frozen-drive experiment: preflight STOP\n\n'+reason+'\n\n'
    report+='## G0: predecessor evidence preserved\n\n'
    report+=f'Parent: {parent_check["sealed_artifact_count"]} sealed files, all SHA256 exact. Cumsum closure: {old_check["sealed_artifact_count"]} sealed files, all SHA256 exact. Both original verdicts are unchanged. New branch and directory are isolated; copied metadata in the clone is not substituted for authoritative original artifacts. No old report or manifest was edited.\n\n'
    report+='## G1: frozen-input evidence\n\n'
    report+='- protocol/frozen_fresh_documents.json: status NOT_SELECTED, count0.\n- protocol/frozen_rotation_bank.json: status NOT_CREATED, ROTATION_BANK_COUNT0, expected_count12.\n- repaired Phase-A preregistration: scientific_docs_selected=false, rotation_bank_created=false, full_premeasurement_freeze_complete=false, PREMEASUREMENT_COMMIT_SHA=null.\n\n'
    report+='The four anchor definitions H/C5R/C6R/C7 and the proposed Cayley interpolation coefficients were specified previously, and checkpoint recovery/parameterization identity passed. Neither fact proves that the required12-condition scientific bank was prospectively materialized and sealed. Likewise a deterministic selection rule does not identify the actual four Discovery documents. Parent metric metadata also retains epsilon/probe_count as NOT_FROZEN; no defaults were invented.\n\n'
    report+='## Read-only source inventory\n\n'
    report+='Production recurrent helper: FP32 state; decay, key-state erase, beta-scaled delta write, query readout. Canonical rotations normalize q/k before separate H and Delta transforms; common native mapping is separate Delta then H on stored states. C128 QDQ uses key axis-2, amax floor1e-12, /127, ties-to-even round, clip[-127,127], cast/copy writeback. All source SHAs are retained and checked; these descriptions are source inventory, not a passed drive-factorization or numerical replay gate. Parent J_ACC/J_INJ definitions are preserved, including the original inclusive[1024,T] averaging window. No new reducer or FD metric was implemented or measured.\n\n'
    report+='## Questions\n\n'
    answers={
      'Q1':'NOT_EVALUATED. No replay implementation or GPU execution; no claim of cumsum-free deterministic qualification.',
      'Q2':'NOT_EVALUATED. No capture/replay parity run.',
      'Q3':'NOT_EVALUATED. No vector or energy decomposition measurements.',
      'Q4':'NOT_EVALUATED at2K/8K/32K; no rotation ranking exists.',
      'Q5':'NOT_EVALUATED. No horizon-dependent history measurements.',
      'Q6':'NOT_EVALUATED. No alignment/cancellation measurements.',
      'Q7':'NOT_EVALUATED. No discordant-pair results.',
      'Q8':'NOT_EVALUATED. Available anchors are not a frozen12-condition bank and no comparison was run.',
      'Q9':'No scientific mechanism result is supported. The authorized scope remains intrinsic recurrent dynamics under frozen external drive; full-model closed-loop reasoning is NOT_EVALUATED.',
      'Q10':'No outcome-driven confirmation hypothesis can be justified without Discovery. A separate explicit input-preparation authorization would be required if the user wishes to establish new prospective Discovery inputs; this report authorizes none of selection, bank construction, confirmation, training, C8 or AIME.'}
    for k,v in answers.items():report+=k+': '+v+'\n\n'
    report+='## Deliverables and action\n\nInventory, both seal checks, exact frozen-input failure evidence, execution status and this final report are provided. Preregistration, capture/schema, later parity gates, scientific CSVs and horizon analyses were deliberately NOT_STARTED because G1 failed. No empty CSVs or invented metrics were emitted. The desired horizons remain2048/8192/32768; no Ling result, endpoint accuracy or rotation ranking was used to choose inputs, caps or metrics.\n\nFINAL_ACTION=FAIL_AND_STOP. No parent resume. No automated continuation.\n'
    text('FINAL_REPORT.md',report)
    # Final readback: verify both seals again after all new artifacts were written.
    after_parent=seal_check(PARENT);after_old=seal_check(PREDECESSOR)
    save('final_parent_seal_revalidation.json',dict(parent=after_parent,engineering_predecessor=after_old,
        all_exact=after_parent['all_exact'] and after_old['all_exact']))
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS']
    text('SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    print(json.dumps(verdict),flush=True)

if __name__=='__main__':main()
