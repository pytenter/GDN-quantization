"""Append-only authoritative accounting correction and prospective G4 freeze."""
import ast
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO=Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
ROOT=REPO/'experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
OLD=ROOT/'amendments/STORAGE_LIFECYCLE_AMENDMENT_V1'
OUT=ROOT/'amendments/STORAGE_ACCOUNTING_CORRECTION_V1'
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_local_documents as A

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n')
    return p

def write(name,text):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(text)
    return p

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    hist=A.history();existing=A.seal(ROOT,'SHA256SUMS');assert existing['all_exact']
    old_seal=[]
    for line in (OLD/'SHA256SUMS').read_text().splitlines():
        expected,rel=line.split(None,1);path=(OLD/rel).resolve()
        assert path.is_relative_to(ROOT.resolve()) and A.sha(path)==expected
        old_seal.append(dict(path=str(path),sha256=expected))
    old_verdict=A.read(OLD/'final_verdict.json')
    assert old_verdict['verdict']['FINAL_ACTION']=='FAIL_AND_STOP_STORAGE_CAPACITY'
    save('old_evidence_revalidation.json',dict(historical=hist,current_stop_seal=existing,
        storage_double_count_stop_seal=old_seal,old_stop_preserved=True))
    anchor=A.read(ROOT/'protocol/frozen_anchor_conditions.json')
    assert anchor['ANCHOR_IDENTITY_GATE']=='PASS' and anchor['ANCHOR_CONDITION_COUNT']==4
    docs=A.read(ROOT/'protocol/frozen_discovery_documents.json')
    assert docs['DISCOVERY_DOC_COUNT']==4
    for d in docs['documents']:
        assert A.sha(ROOT/d['text_path'])==d['text_sha256']
        assert A.sha(ROOT/d['token_path'])==d['token_sha256']
    budget=A.read(OLD/'storage_lifecycle_report.json')
    base=budget['OPERATIONAL_LIFECYCLE_PEAK_BYTES'];margin=max(5*1024**3,(base+4)//5)
    required=base+margin
    assert required==26090930176 and margin==budget['SAFETY_MARGIN_BYTES']
    st=os.statvfs(ROOT);available=st.f_bavail*st.f_frsize
    correction=dict(AUTHORITATIVE_STORAGE_ACCOUNTING_CORRECTION=True,old_fail_preserved=True,
        previous_report_sha256=A.sha(OLD/'FINAL_REPORT.md'),previous_storage_sha256=A.sha(OLD/'storage_lifecycle_report.json'),
        classification='ACCOUNTING_SPECIFICATION_BUG; NOT_SCIENTIFIC_PROTOCOL_CHANGE; NOT_SAFETY_MARGIN_RELAXATION',
        formula='BASE_PEAK_BYTES + SAFETY_MARGIN_BYTES <= AVAILABLE_DATA_BYTES; margin counted exactly once',
        BASE_PEAK_BYTES=base,SAFETY_MARGIN_BYTES=margin,PEAK_REQUIRED_STORAGE_BYTES=required,
        AVAILABLE_DATA_BYTES=available,HEADROOM_BYTES=available-required,
        STORAGE_CAPACITY_GATE='PASS' if required<=available else 'FAIL',
        prior_audited_available_bytes=30559752192,prior_audited_correct_headroom_bytes=4468822016,
        prior_audit_commit=A.cmd(['git','rev-parse','HEAD'])['stdout'].strip(),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        scientific_protocol_changed=False,documents_reselected=False,old_artifacts_modified=0,
        unchanged_components=budget['result_components'],single_doc_drive_bytes=budget['ESTIMATED_SINGLE_DOC_DRIVE_BYTES'])
    save('storage_lifecycle_report.json',correction)
    assert correction['STORAGE_CAPACITY_GATE']=='PASS','current storage insufficient; STOP'
    first=A.read(ROOT/docs['documents'][0]['token_path'])
    assert len(first)==32768
    fixture=save('protocol/engineering_fixture_256_token_ids.json',first[:256])
    parent=A.read(A.PARENT/'protocol/preregistration_repaired_001.json')
    policy=dict(EXPERIMENT=A.TASK,stage='G4_CAPTURE_EMPIRICAL_REPRODUCIBILITY',
        fixture_rule='First256 token IDs of the already frozen first Discovery document; engineering only, no reselection',
        fixture_doc_id=docs['documents'][0]['canonical_doc_id'],fixture_file=str(fixture),fixture_file_sha256=A.sha(fixture),
        source_document_token_file_sha256=docs['documents'][0]['token_sha256'],fixture_token_count=256,
        acquisition_role='FROZEN_INPUT_ACQUISITION_ONLY',canonical_capture_rule='FIRST_SUCCESSFUL_VALID_CAPTURE',
        original_production_cumsum=True,strict_determinism_enforcement=False,
        capture_seed=parent['test_seed'],fresh_process_count=2,physical_gpu=0,device_map={'':'cuda:0'},
        topology='Unchanged formal single-GPU BF16 loader; first token original chunk helper, subsequent cached one-token original recurrent helper',
        tensors=['query','key','value','g','beta'],drop_tensors=False,compression=False,dtype_conversion=False,
        shard_tokens=64,FP_reference_tensor_checkpoints=[32,256],FP_reference_state_hashes='every token, every24 GDN layers',
        required_qkv_shape=[1,1,32,128],required_g_beta_shape=[1,1,32],
        expected_dtype={'query':'torch.bfloat16','key':'torch.bfloat16','value':'torch.bfloat16','g':'torch.float32','beta':'torch.bfloat16','state':'torch.float32'},
        finite_required=True,expected_layers=anchor['layer_ids'],
        criterion='Byte-exact across both fresh processes: all tensor identities, every reference state hash, checkpoint tensor hashes; not torch.save container bytes',
        no_capture_quality_selection=True,release_if_byte_exact='YES',release_if_not_byte_exact='NO',
        numerical_difference_reporting_policy=parent['numerical_policy'],
        any_gate_failure_action='FAIL_AND_STOP; preserve both captures; no retries or threshold changes',
        nonexact_retention_action='FAIL_AND_STOP_STORAGE_RETENTION_REQUIRED if four full raw drives cannot be retained',
        forbidden=dict(training=False,optimizer_step=False,AIME_generation=False,free_generation=False,confirmation=False),
        anchor_manifest_sha256=A.sha(ROOT/'protocol/frozen_anchor_conditions.json'),
        discovery_manifest_sha256=A.sha(ROOT/'protocol/frozen_discovery_documents.json'),
        tokenizer_manifest_sha256=A.sha(ROOT/'protocol/tokenizer_identity.json'),
        source_sha256={row['path']:row['sha256'] for row in anchor['canonical_sources']},
        model_identity= A.read(A.PARENT/'provenance/runtime_identity.json')['model_identity'])
    save('protocol/capture_engineering_preregistration.json',policy)
    numerical=parent['numerical_policy']
    remaining=dict(frozen_before_GPU_capture_results=True,
        gate_order=['CAPTURE_EMPIRICAL_REPRODUCIBILITY_GATE','DRIVE_FACTORIZATION_GATE','CAPTURE_SCHEMA_GATE',
            'FP_REPLAY_PARITY_GATE','NO_QDQ_ROTATION_INVARIANCE_GATE','COUNTERFACTUAL_DECOMPOSITION_GATE',
            'REPLAY_ISOLATION_GATE','REPLAY_DETERMINISM_GATE','STREAMING_REDUCER_PARITY_GATE'],
        FP_REPLAY_PARITY_GATE=dict(lengths=[32,256],criterion='Exact tensor SHA across all24 layers at every replay token, inherited parent FP exact check',
            numerical_policy_for_reporting_only=numerical,no_posthoc_fallback=True),
        NO_QDQ_ROTATION_INVARIANCE_GATE=dict(conditions=['Native','H','C6R','C7'],steps=256,layers=24,
            criterion=numerical['comparison'],gamma128=numerical['gamma128'],multiplier=8,
            comparison_operands='condition-specific unquantized native state and native FP state; max(1,abs operands)',
            exact_arithmetic_invariance_not_claimed=True),
        COUNTERFACTUAL_DECOMPOSITION_GATE=dict(steps=256,conditions=['Native','H','C6R','C7'],
            vector='FP64 E=SQ-FP; H=F(SQ)-F(FP); I_PATH=SQ-F(SQ); I_REF=Q(FP)-F(FP); INTERACTION=I_PATH-I_REF',
            vector_absolute_bound='8*2^-53*max(1,max_abs(original FP32 states converted to FP64))',
            energy_relative_bound='16*gamma_N_FP64*max(1,ACC,HISTORY,PATH_INJECTION,abs(ALIGNMENT)); N=32*128*128',
            common_denominator='FP native state squared Frobenius norm accumulated in FP64 +1e-12 per layer/token',
            normalized_window='inclusive1024..T for science; inclusive1..T for short engineering checks only',
            energy_accumulator='FP64; signed alignment retained'),
        REPLAY_ISOLATION_GATE=dict(blocked_cumsum_call_count=0,full_model_forward_count=0,
            instrument='Fail-closed torch.cumsum/Tensor.cumsum and torch.nn.Module.__call__ guards; only original helper and formal transform/QDQ allowed'),
        REPLAY_DETERMINISM_GATE=dict(strict_algorithms=True,length256_fresh_processes=3,length2048_fresh_processes=2,
            all_conditions=True,criterion='All key state hashes and metric hashes byte-exact',
            fixture2048='First2048 tokens of already frozen Doc1 captured once only after short gates PASS; engineering only',
            unsupported_operator='FAIL_AND_STOP; do not patch'),
        STREAMING_REDUCER_PARITY_GATE=dict(short_length=256,
            full_materialization='All per-token per-layer FP64 energy/normalized observations; not full hidden/state trajectory',
            criterion='All six normalized terms and six raw energy means match within16*gamma_256_FP64*max(1,abs values)',
            no_changed_reduction_or_metric_definition=True),
        persistent_state='Production stored basis across all tokens; physical state is a derived read-only separate H/Delta mapping',
        no_inverse_remapping_of_persistent_path=True,
        clean_reference_Q='Separate condition-specific inverse-map of FP state then original helper, formal quantizer/writeback and physical map',
        source_transition='Unchanged original torch_recurrent_gated_delta_rule reused, no copied recurrence formula',
        scientific_metric_names=['J_ACC_FD','J_HISTORY_FD','J_INJ_PATH_FD','J_INJ_REF_FD','J_ALIGN_FD','J_INTERACTION_FD'],
        weighting='Equal layer weight; per-layer per-token Frobenius norm includes all32 heads and128x128 coordinates; equal token weight',
        epsilon=1e-12,horizons=[2048,8192,32768],scientific_first_token=1024,
        parent_metric_source_sha256=A.sha(A.PARENT/'protocol/metric_definitions.json'),
        parent_J_INJ_denominator_not_relabelled_as_FD=True,
        any_new_failure='FAIL_AND_STOP; preserve input/results; no retry, no threshold change, no failed drive deletion')
    save('protocol/remaining_engineering_gate_policy.json',remaining)
    write('CORRECTION_REPORT.md',
        '# Authoritative storage accounting correction\n\n'
        'The old double-counting FAIL report, seals and audit commits remain byte-identical. '
        'Safety margin is counted exactly once as authorized; no scientific definitions or inputs changed.\n\n'
        f'BASE_PEAK_BYTES = {base}\n\nSAFETY_MARGIN_BYTES = {margin}\n\n'
        f'PEAK_REQUIRED_STORAGE_BYTES = {required}\n\nAVAILABLE_DATA_BYTES = {available}\n\n'
        f'HEADROOM_BYTES = {available-required}\n\nSTORAGE_CAPACITY_GATE = PASS\n\n'
        'Next unfinished gate is G4, original production capture on the fixed256-token engineering fixture in two fresh processes. '
        'The fixture is the prefix of the existing first frozen document, not new selection. Tensor and every-token state hashes must match exactly. '
        'This engineering policy is frozen before any GPU results. It is not the completed scientific premeasurement preregistration. '
        'Scientific Discovery remains0/16 and is not authorized until all later engineering gates and the full premeasurement freeze pass.\n')
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts]
    write('SHA256SUMS',''.join(A.sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in files))
    print(json.dumps(correction),flush=True)

if __name__=='__main__':main()
