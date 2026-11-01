"""Preserve prepared inputs and append the prospective storage STOP evidence.

No matrix operations, model forward, GPU work, scientific metrics or cleanup.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent))
import prepare_local_documents as A

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    docs=A.read(A.ROOT/'protocol/frozen_discovery_documents.json')
    assert docs['DISCOVERY_DOCUMENT_SELECTION_GATE']=='PASS' and docs['DISCOVERY_DOC_COUNT']==4
    for d in docs['documents']:
        assert A.sha(A.ROOT/d['text_path'])==d['text_sha256']
        assert A.sha(A.ROOT/d['token_path'])==d['token_sha256']
        assert len(A.read(A.ROOT/d['token_path']))==32768
    config=A.read(A.MODEL/'config.json')
    text_config=config.get('text_config',config)
    assert text_config['linear_num_key_heads']==16 and text_config['linear_num_value_heads']==32
    assert text_config['linear_key_head_dim']==128 and text_config['linear_value_head_dim']==128
    A.save('resource_model_configuration.json',dict(config_path=str(A.MODEL/'config.json'),sha256=A.sha(A.MODEL/'config.json'),text_config=text_config))
    layers=A.read(A.PARENT/'provenance/GDN_RECURRENCE_SOURCE_AUDIT.json')['recurrent_layer_ids']
    assert len(layers)==24
    # Original production input: repeated key heads32, value heads32, K=V=128;
    # key repetition can be losslessly encoded as16 unique source heads.
    # Query is not used by the recurrent STATE update (only readout) and can be
    # omitted from a minimum state-only drive; this is an optimistic budget,
    # not an implemented or passed drive-factorization/replay gate.
    source_keys=16;value_heads=32;dimension=128;steps=32768;doc_count=4
    minimum_bytes_per_token_per_layer=source_keys*dimension*2+value_heads*dimension*2+value_heads*4+value_heads*2
    minimum_drive_bytes=minimum_bytes_per_token_per_layer*steps*len(layers)*doc_count
    raw_state_bytes=32*128*128*4
    sparse_reference_bytes=raw_state_bytes*len(layers)*3*doc_count
    metadata_and_shard_overhead=128*1024**2
    reserve=5*1024**3
    required=minimum_drive_bytes+sparse_reference_bytes+metadata_and_shard_overhead+reserve
    full_query_key_value_bytes=(3*32*128*2+32*4+32*2)*steps*len(layers)*doc_count
    current=os.statvfs(A.ROOT);available=current.f_bavail*current.f_frsize
    assert available<required, 'Capacity changed: do not emit a stale resource FAIL; inspect before continuing'
    resource=dict(STORAGE_CAPACITY_GATE='FAIL',failure_kind='PREMEASUREMENT_RESOURCE_FEASIBILITY',
        available_bytes=available,required_available_bytes=required,shortfall_bytes=required-available,
        doc_count=4,tokens_per_doc=32768,recurrent_layers=24,
        state_only_drive_minimum_uncompressed_bytes=minimum_drive_bytes,
        full_query_key_value_original_dtype_bytes=full_query_key_value_bytes,
        sparse_reference_state_bytes=sparse_reference_bytes,metadata_and_shard_overhead_bytes=metadata_and_shard_overhead,
        safe_free_space_reserve_bytes=reserve,
        minimum_drive_components=dict(key_unique16_heads_BF16=4096,value32_heads_BF16=8192,g32_FP32=128,beta32_BF16=64),
        budget_interpretation='Original-dtype uncompressed sufficient representation, assuming exact source duplicate-key-head encoding and state-only query omission; no measured compression ratio or speculative data-dependent savings assumed',
        query_omission_is_a_design_estimate_not_a_passed_drive_factorization=True,
        source=dict(architecture=str(A.PARENT/'provenance/GDN_RECURRENCE_SOURCE_AUDIT.json'),sha256=A.sha(A.PARENT/'provenance/GDN_RECURRENCE_SOURCE_AUDIT.json')),
        inspected_local_filesystems=A.cmd(['df','-B1','/data','/tmp','/home'])['stdout'],
        home_root_writable=os.access('/home',os.W_OK),own_home_on_data=True,
        A800_read_only_check=dict(host='connect.nma1.seetacloud.com',port=49629,gpu_mode='NO_GPU',
            root_available_bytes=12788109312,data_available_bytes=12101947392,
            root_path='/root',data_path='/root/autodl-tmp',source='Authenticated read-only df -B1 and login resource inventory during this task',
            files_written=0,processes_interrupted=0,compute_work_started=False),
        no_capture_started=True,no_old_scientific_artifacts_deleted=True,no_unsupported_multi_host_storage_pipeline_created=True)
    A.save('premeasurement_resource_feasibility.json',resource)
    summary=A.read(A.ROOT/'eligible_document_manifest.json')
    chosen=next(c for c in summary['corpora'] if c['dataset_name']==docs['CANDIDATE_CORPUS'])
    A.save('protocol/future_confirmation_rule.json',dict(CONFIRMATION_RULE_FROZEN='YES',
        rule='Continue the same namespace|candidate_pool_revision|canonical_doc_id SHA256 ordering after the four Discovery documents',
        CANDIDATE_POOL_REVISION=docs['CANDIDATE_POOL_REVISION'],confirmation_inputs_materialized=False,CONFIRMATION_EXECUTED=0,
        pure_candidate_eligibility_tokenization_is_not_confirmation_execution=True,separate_confirmation_authorization_required=True))
    verdict=dict(EXPERIMENT=A.TASK,OLD_PARENT_STOP_PRESERVED='YES',OLD_CUMSUM_CLOSURE_FAIL_PRESERVED='YES',
        OLD_PREFLIGHT_FAIL_PRESERVED='YES',OLD_INPUT_FREEZE_AMENDMENT_FAIL_PRESERVED='YES',OLD_EVIDENCE_INTEGRITY_GATE='PASS',
        SCIENTIFIC_MEASUREMENT_COUNT_BEFORE_NEW_EXPERIMENT=0,NEW_PROSPECTIVE_DOCUMENT_PROTOCOL='YES',
        PROSPECTIVE_DOCUMENT_PROTOCOL_GATE='PASS',CANDIDATE_CORPUS=docs['CANDIDATE_CORPUS'],
        CANDIDATE_POOL_REVISION=docs['CANDIDATE_POOL_REVISION'],ELIGIBLE_DOCUMENT_COUNT=chosen['eligible_count'],
        DISCOVERY_DOCUMENT_SELECTION_GATE='PASS',DISCOVERY_DOC_COUNT=4,ANCHOR_CONDITION_COUNT=0,ANCHOR_IDENTITY_GATE='NOT_EVALUATED',
        PREMEASUREMENT_COMMIT_SHA=None,STORAGE_CAPACITY_GATE='FAIL',
        DRIVE_FACTORIZATION_GATE='NOT_EVALUATED',CAPTURE_SCHEMA_GATE='NOT_EVALUATED',FP_REPLAY_PARITY_GATE='NOT_EVALUATED',
        NO_QDQ_ROTATION_INVARIANCE_GATE='NOT_EVALUATED',COUNTERFACTUAL_DECOMPOSITION_GATE='NOT_EVALUATED',
        REPLAY_ISOLATION_GATE='NOT_EVALUATED',REPLAY_DETERMINISM_GATE='NOT_EVALUATED',STREAMING_REDUCER_PARITY_GATE='NOT_EVALUATED',
        DISCOVERY_EXECUTION_AUTHORIZED='NO',DISCOVERY_COMPLETE='0/16',HORIZON_2048_COMPLETE='0/16',HORIZON_8192_COMPLETE='0/16',HORIZON_32768_COMPLETE='0/16',
        CONFIRMATION_RULE_FROZEN='YES',CONFIRMATION_EXECUTED=0,FULL_MODEL_CLOSED_LOOP_PROPAGATION='NOT_EVALUATED',
        NEW_C8_TRAINING_AUTHORIZED='NO',NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,NEW_TRAINING='NO',OPTIMIZER_STEP=0,
        PARENT_PROPAGATION_RESUME_AUTHORIZED='NO',PUSH_PERFORMED='NO',FINAL_ACTION='FAIL_AND_STOP')
    A.save('final_verdict.json',dict(verdict=verdict,stop_phase='Premeasurement resource readiness; after deterministic document preparation, before anchor/preregistration/GPU stages',
        missing_downstream_artifacts_reason='Resource gate fails; no successful full premeasurement freeze or engineering qualification claimed',
        desired_anchor_labels=['R0 Native INT8-C128 identity','R1 Hadamard','R2 C6R','R3 C7'],
        anchor_count_semantics='0 newly frozen/qualified anchor identities; the four intended labels above remain specified, not dropped',
        scientific_metric_execution_count=0,drive_capture_count=0,no_inputs_substituted=True,
        full_freeze_commit_created=False,audit_commit_is_not_preregistration=True))
    A.save('execution_status.json',dict(stage='PREMEASUREMENT_RESOURCE_STOP',document_preparation='PASS',
        model_loaded=False,GPU_compute_started=False,model_forward_count=0,drive_capture_count=0,replay_units=0,
        scientific_metric_execution_count=0,training_steps=0,confirmation_inputs_materialized=0,
        actual_anchor_identity_freeze_count=0,formal_preregistration_complete=False,
        no_cumsum_patch=True,no_interpolated_rotation_bank=True,old_artifacts_modified=0))
    report='# QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1: premeasurement resource STOP\n\n'
    report+='New prospective document preparation succeeded. Four Discovery documents were mechanically selected by the new namespace/revision/ID hash ranking, without semantic selection or model forward. '
    report+='The experiment is stopped before anchor qualification, full preregistration, capture and replay because the existing writable scientific storage cannot safely retain four original-dtype canonical drive trajectories. '
    report+='This is NOT another legacy document-rule failure, NOT a scientific mechanism failure and NOT a parent resume.\n\n## Historical evidence\n\n'
    report+='All148 sealed predecessor files (47parent+80cumsum+8Attempt0+13failed-amendment) pass SHA256. The failed amendment audit commit196e1da and all four original verdicts remain unchanged. '
    report+='Scientific measurements before this experiment and in this attempt are both0.\n\n## Deterministic corpus/doc selection\n\n'
    report+='Local PG19 was not found in surveyed own/project-referenced roots. Existing Qwen WikiText calibration and reserved confirmation/mechanism corpora were excluded wholesale without reading raw contents. '
    report+='Priority C evaluated the8 existing local LongBench configurations using only context and stable IDs, never answers or endpoint outcomes. '
    report+='The selected repobench-p corpus has8 eligible distinct contexts out of500 records; lcc has1 and all other surveyed configurations0. No corpus was downloaded or concatenated.\n\n'
    report+='Corpus revision: '+docs['CANDIDATE_POOL_REVISION']+'\n\n'
    for d in docs['documents']:
        report+=f"- Rank{d['selection_rank']}: `{d['canonical_doc_id']}`; original{d['token_count']} tokens; frozen first32768; key`{d['selection_key']}`.\n"
    report+='\nAll four text/token file hashes were revalidated. Future Confirmation only has a frozen continuation rule; no Confirmation input was materialized or executed. Candidate-pool eligibility tokenization is not holdout model evaluation.\n\n## Storage feasibility\n\n'
    report+=f"The state-only original-dtype drive estimate is {minimum_drive_bytes:,} bytes, before sparse reference states and shard metadata. This optimistic minimum assumes lossless16-source-key-head encoding and omits query because it is not used in the state-update equations; this design estimate is not a passed factorization gate. "
    report+=f"Keeping all query/key/value tensors instead costs {full_query_key_value_bytes:,} bytes. With sparse states, metadata allowance and5GiB safety reserve, required available capacity is {required:,} bytes; /data currently has {available:,}; shortfall {required-available:,} bytes.\n\n"
    report+='The /home filesystem has additional free capacity but no writable own-user directory or write permission on its root. The already-authorized A800 is currently GPU-less; its root/data mounts have about12.8/12.1GB free each. '
    report+='No unproven cross-host storage/sharding pipeline was introduced, no old scientific artifact deleted, no partition filled near exhaustion, no process interrupted. '
    report+=f'A separately authorized continuation needs at least about{required/1e9:.1f}GB available capacity on a safe writable drive-artifact destination for the minimum representation, or about{(full_query_key_value_bytes+sparse_reference_bytes+metadata_and_shard_overhead+reserve)/1e9:.1f}GB if all repeated-head raw query/key/value tensors are retained. '
    report+='Inputs, corpus revisions and selection remain sealed; this report does not authorize changing documents, horizons or scientific definitions.\n\n## Required questions\n\n'
    answers={'Q1':'YES: new prospective outcome-independent hierarchy and SHA selection implemented before tokenization/measurement.',
        'Q2':'YES: four uniquely selected Discovery documents/texts/32768-token files frozen and hash-verified.',
        'Q3':'NOT_EVALUATED: four intended anchors unchanged, but identity qualification not started after resource STOP.',
        'Q4':'NOT_EVALUATED: scientific replay not implemented or run; no cumsum patch attempted.',
        'Q5':'NOT_EVALUATED: no capture or FP replay parity.',
        'Q6':'NOT_EVALUATED: no E/H/injection vector or energy measurement.',
        'Q7':'NOT_EVALUATED: no C6R/H mechanism contrasts.',
        'Q8':'NOT_EVALUATED: no C7/C6R mechanism pattern.',
        'Q9':'NOT_EVALUATED: no Native/H degradation result.',
        'Q10':'NOT_EVALUATED: all2K/8K/32K scientific counts0.',
        'Q11':'Only the stated intrinsic frozen-drive scope is designed; no full-model reasoning causality evidence exists.',
        'Q12':'No scientific Confirmation hypothesis established; Confirmation NOT_RUN.'}
    for q,a in answers.items():report+=q+': '+a+'\n\n'
    report+='## Final fields\n\n```text\n'+'\n'.join(k+' = '+('null' if v is None else str(v)) for k,v in verdict.items())+'\n```\n'
    report+='\nMissing downstream manifests, engineering results and scientific CSVs are deliberate consequences of the premeasurement resource failure, not empty/unreported completed results. '
    report+='Any commit preserving this stopped audit is audit-only, not PREMEASUREMENT_COMMIT_SHA. FINAL_ACTION=FAIL_AND_STOP.\n'
    A.write('FINAL_REPORT.md',report)
    A.save('final_historical_seal_revalidation.json',A.history())
    files=[p for p in sorted(A.ROOT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts]
    A.write('SHA256SUMS',''.join(A.sha(p)+'  '+str(p.relative_to(A.ROOT))+'\n' for p in files))
    print(json.dumps(verdict),flush=True)

if __name__=='__main__':main()
