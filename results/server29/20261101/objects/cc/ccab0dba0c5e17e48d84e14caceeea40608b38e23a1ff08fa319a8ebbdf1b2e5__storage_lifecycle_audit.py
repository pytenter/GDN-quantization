"""Append-only CPU audit of the authorized storage lifecycle amendment.

No model forward, capture, replay, compression, cleanup or scientific measurement.
Sections 5 and 6 are applied literally; a one-margin interpretation is reported
separately, never used to silently authorize execution.
"""
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
ROOT = REPO/'experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
OUT = ROOT/'amendments/STORAGE_LIFECYCLE_AMENDMENT_V1'
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_local_documents as A

FORMAL = A.CANON/'experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py'
ARCH = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
PHYSICAL = Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/sparse_probe.py')
EXPECTED_SOURCES = {
    FORMAL: '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69',
    ARCH: '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a',
    PHYSICAL: 'c214b3e7d2dacabb445378ef85cda5ed4ffe8fc2679d56d7c78217a8a95780b9',
}
EXPECTED_EXPORTS = {
    'C6R': 'a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6',
    'C7': '57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8',
}
EXPECTED_IDS = [
    'd318e101c2458d5e844d488bad7f6ce0e147f5075a880514',
    'bd788a58347b1655aec6622c6ba1d8e6687199abb90a510d',
    '3a4de092b20b1c9dd64199a1d382ddd2a906a37ed2dcda96',
    '5eab3c8573bc773090ab7ea27211875c61890ca4e2f92711',
]

def save(name, value):
    p=OUT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    return p

def write(name, text):
    p=OUT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(text)
    return p

def tensor_sha(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def documents():
    d=A.read(ROOT/'protocol/frozen_discovery_documents.json')
    assert d['DISCOVERY_DOC_COUNT']==4 and d['DISCOVERY_DOCUMENT_SELECTION_GATE']=='PASS'
    assert d['CANDIDATE_CORPUS']=='long_bench/repobench-p'
    revision=d['CANDIDATE_POOL_REVISION']
    assert revision=='7dcaafcb4974ad5893d9f22a7f36fd70fa811bfa8c5fc5448a931ddc0917eb9a'
    assert A.sha(ROOT/'corpora/repobench-p/candidate_pool_manifest.json')==revision
    source=A.read(ROOT/'corpora/repobench-p/candidate_pool_manifest.json')
    assert A.sha(source['source_container_path'])==source['source_container_sha256']
    eligible=A.read(ROOT/'corpora/repobench-p/eligible_document_manifest.json')
    assert eligible['CANDIDATE_POOL_REVISION']==revision and eligible['eligible_count']==8
    ranking=[]
    for row in eligible['records']:
        if row['eligible']:
            key=A.hash_bytes((A.TASK+'|'+revision+'|'+row['canonical_doc_id']).encode())
            ranking.append((key,row['canonical_doc_id']))
    assert len(set(k for k,_ in ranking))==8
    ranking.sort()
    expected=[('long_bench/repobench-p/test/'+x) for x in EXPECTED_IDS]
    assert [x['canonical_doc_id'] for x in d['documents']]==expected
    assert [x[1] for x in ranking[:4]]==expected
    tok=A.read(ROOT/'protocol/tokenizer_identity.json')
    for f in tok['files']:assert A.sha(f['path'])==f['sha256']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(A.MODEL),local_files_only=True,trust_remote_code=True)
    assert type(tokenizer).__name__==tok['tokenizer_class']
    rows=[]
    for i,row in enumerate(d['documents'],1):
        assert row['selection_rank']==i and row['selection_key']==ranking[i-1][0]
        assert A.sha(ROOT/row['text_path'])==row['text_sha256']==row['content_SHA256']
        assert A.sha(ROOT/row['token_path'])==row['token_sha256']
        text=(ROOT/row['text_path']).read_bytes().decode('utf-8')
        ids=tokenizer.encode(text,add_special_tokens=False)
        frozen=A.read(ROOT/row['token_path'])
        assert len(ids)==row['token_count'] and len(frozen)==32768 and ids[:32768]==frozen
        rows.append(dict(doc_id=row['canonical_doc_id'],selection_rank=i,selection_key=row['selection_key'],
            content_sha256=row['text_sha256'],token_sha256=row['token_sha256'],retokenization_exact=True))
    return dict(FROZEN_DISCOVERY_DOCUMENT_INTEGRITY_GATE='PASS',frozen_document_count=4,
        selection_manifest_sha256=A.sha(ROOT/'protocol/frozen_discovery_documents.json'),
        tokenizer_manifest_sha256=A.sha(ROOT/'protocol/tokenizer_identity.json'),
        candidate_revision=revision,rows=rows,original_files_modified=0,reselection=False,model_forward_count=0)

def anchors():
    import torch
    torch.set_num_threads(1)
    source_rows=[]
    for p,expected in EXPECTED_SOURCES.items():
        actual=A.sha(p);assert actual==expected, 'canonical source identity mismatch'
        source_rows.append(dict(path=str(p),sha256=actual))
    nodes=ast.parse(FORMAL.read_text()).body
    hnode=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='hadamard')
    namespace={'torch':torch,'math':math}
    exec(compile(ast.Module(body=[hnode],type_ignores=[]),str(FORMAL),'exec'),namespace)
    H=namespace['hadamard'](128,dtype=torch.float32,device='cpu')
    I=torch.eye(128,dtype=torch.float32)
    layer_ids=A.read(A.PARENT/'provenance/GDN_RECURRENCE_SOURCE_AUDIT.json')['recurrent_layer_ids']
    assert layer_ids==[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]
    previous=A.read(A.PARENT/'provenance/rotation_parameterization_identity.json')
    bound=previous['policy']['orthogonality_max_abs'];assert bound==1e-4
    recovery=A.read(A.PARENT/'provenance/anchor_recovery.json')
    records=[]
    for code,label,matrix in [('A0','Native INT8-C128',I),('A1','Hadamard',H)]:
        residual=float((matrix.T@matrix-I).abs().max());assert residual<=bound
        records.append(dict(anchor_id=code,condition=label,matrix_sha256=tensor_sha(matrix),
            matrix_dtype=str(matrix.dtype),matrix_shape=list(matrix.shape),
            per_layer_matrix_sha256={str(l):tensor_sha(matrix) for l in layer_ids},
            orthogonality_max_abs=residual,
            identity_rule='No rotation; exact identity map' if code=='A0' else 'Unchanged formal hadamard(128) function; FP32 normalized Sylvester H128',
            native_basis='S_native = S_stored' if code=='A0' else 'S_native = H128 @ S_stored',
            qk_rule='Original normalize_qk then no rotation' if code=='A0' else 'Original normalize_qk then q/k.float @ H128'))
    for code,label in [('A2','C6R'),('A3','C7')]:
        provenance=next(r for r in recovery['anchors'] if r['condition']==label)
        p=Path(provenance['matrix'])
        assert A.sha(p)==EXPECTED_EXPORTS[label]==provenance['matrix_sha256']
        payload=torch.load(p,map_location='cpu',weights_only=False)
        assert list(payload['layer_ids'])==layer_ids
        matrices=payload['matrices'];assert set(matrices)==set(map(str,layer_ids))
        rows=[]
        for layer in layer_ids:
            t=matrices[str(layer)]
            assert t.device.type=='cpu' and t.dtype==torch.float32 and list(t.shape)==[128,128]
            assert bool(torch.isfinite(t).all())
            expected=next(r for r in previous['rows'] if r['condition']==label and r['layer']==layer)
            actual=tensor_sha(t);assert actual==expected['frozen_sha256']
            residual=float((t.T@t-I).abs().max());assert residual<=bound
            rows.append(dict(layer=layer,matrix_sha256=actual,orthogonality_max_abs=residual))
        records.append(dict(anchor_id=code,condition=label,export_path=str(p),export_file_sha256=A.sha(p),
            original_provenance=provenance,per_layer_matrices=rows,matrix_dtype='torch.float32',matrix_shape=[128,128],
            orthogonality_max_abs=max(r['orthogonality_max_abs'] for r in rows),
            native_basis='S_native = H128 @ (Delta @ S_stored); separate products, never fused',
            stored_from_native='Delta.T @ (H128 @ S_native); separate products',
            qk_rule='Original normalize_qk then q/k.float @ H128 @ Delta; separate products',
            direct_frozen_export_load=True,theta_reconstruction=False))
    result=dict(ANCHOR_CONDITION_COUNT=4,ANCHOR_IDENTITY_GATE='PASS',layer_ids=layer_ids,
        anchors=records,canonical_sources=source_rows,orthogonality_bound_inherited=bound,
        source_mapping=A.read(A.PARENT/'provenance/native_basis_source_mapping.json'),
        no_new_learned_rotation=True,no_interpolation=True,model_loaded=False,numerical_replay_invariance='NOT_EVALUATED')
    A.save('protocol/frozen_anchor_conditions.json',result)
    return result

def storage():
    # Complete raw original helper arguments: q/k/v repeated32 heads BF16,
    # g32 FP32, beta32 BF16. No query omission or head deduplication.
    steps=32768;layers=24;heads=32;dim=128
    per_token_layer=3*heads*dim*2+heads*4+heads*2
    drive=per_token_layer*steps*layers
    state_checkpoint=heads*dim*dim*4*layers
    # All four documents' sparse FP horizon states are retained. No trajectory.
    fp_references=state_checkpoint*3*4
    # Two original-dtype256 fixture captures and32/256 FP checkpoints per run.
    engineering=per_token_layer*256*layers*2+state_checkpoint*2*2
    metadata=128*1024**2
    # Same-filesystem rename has no second full copy. Shard payload temporary
    # is already included in full drive; in-memory replay has no disk temp.
    operational=drive+fp_references+engineering+metadata
    margin=max(5*1024**3,(operational+4)//5)
    section5_required=operational+margin
    literal_total=section5_required+margin
    st=os.statvfs(ROOT);available=st.f_bavail*st.f_frsize
    gate='PASS' if section5_required<=available-margin else 'FAIL'
    lower_bound=drive+fp_references+2*5*1024**3
    return dict(OLD_TOTAL_STORAGE_ESTIMATE_BYTES=45365592064,
        STORAGE_STRATEGY='SEQUENTIAL_PER_DOCUMENT_CAPTURE_REPLAY_RELEASE',MAX_CONCURRENT_FULL_DOCUMENT_DRIVES=1,
        ESTIMATED_SINGLE_DOC_DRIVE_BYTES=drive,TEMPORARY_CAPTURE_BYTES=0,ATOMIC_FINALIZE_OVERHEAD_BYTES=0,
        REPLAY_TEMPORARY_BYTES=0,RESULT_BYTES=fp_references+engineering+metadata,
        result_components=dict(retained_sparse_fp_reference_bytes=fp_references,
            retained_256_engineering_fixture_and_reference_bytes=engineering,metadata_results_serialization_allowance_bytes=metadata),
        SAFETY_MARGIN_BYTES=margin,OPERATIONAL_LIFECYCLE_PEAK_BYTES=operational,
        PEAK_REQUIRED_STORAGE_BYTES=section5_required,AVAILABLE_STORAGE_BYTES=available,
        LITERAL_SECTION6_TOTAL_FREE_REQUIREMENT_BYTES=literal_total,
        HEADROOM_BYTES=available-literal_total,SHORTFALL_BYTES=max(0,literal_total-available),
        STORAGE_CAPACITY_GATE=gate,
        literal_gate_formula='(operational_peak + safety_margin) <= (available - safety_margin)',
        minimal_raw_drive_plus_retained_fp_checkpoints_double_margin_lower_bound_bytes=lower_bound,
        lower_bound_shortfall_bytes=max(0,lower_bound-available),
        ONE_MARGIN_INTERPRETATION=dict(required_total_bytes=section5_required,
            headroom_bytes=available-section5_required,gate='PASS' if section5_required<=available else 'FAIL',
            execution_authorization='NO; requires explicit resolution of Sections5/6, not silently adopted'),
        raw_formula=dict(bytes_per_layer_token=per_token_layer,layers=layers,tokens=steps,
            query_BF16_bytes=8192,key_BF16_bytes=8192,value_BF16_bytes=8192,g_FP32_bytes=128,beta_BF16_bytes=64),
        compression='NONE; no speculative compression ratio or new compression implementation',
        query_dropped=False,head_deduplication=False,dtype_conversion=False,
        peak_observed_drive_disk_usage_bytes=0,bytes_released_per_document=[],
        actual_scientific_usage='NOT_RUN; estimates above are not measured capture/replay sizes',
        physical_capacity_note='The complete raw representation is chosen before factorization; no claim that every legal future lossless encoding would fail',
        audit_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('HF_HUB_OFFLINE')=='1'
    status=A.cmd(['git','status','--porcelain'])
    # Only this newly uploaded amendment namespace is allowed untracked.
    for row in status['stdout'].splitlines():
        assert 'experiments/'+A.TASK+'/amendments/STORAGE_LIFECYCLE_AMENDMENT_V1' in row
    base=A.cmd(['git','rev-parse','HEAD'])['stdout'].strip()
    assert base=='6b88fdf1ed70661ec7f8e94615f729f13d3ec564'
    history=A.history();current=A.seal(ROOT,'SHA256SUMS')
    assert current['all_exact'] and current['sealed_artifact_count']==41
    save('old_evidence_integrity.json',dict(historical=history,current_experiment_stop_seal=current,
        total_sealed_files=189,OLD_EVIDENCE_INTEGRITY_GATE='PASS',old_audit_commit=base,audit_commit_only=True))
    print('G0 PASS: 189 unchanged sealed files',flush=True)
    save('frozen_discovery_document_integrity.json',documents())
    print('G1 PASS: identical four documents and exact retokenization',flush=True)
    anchor=anchors();save('anchor_identity_report.json',anchor)
    print('G2 PASS: four anchors, authoritative exports, source SHA, orthogonality',flush=True)
    budget=storage();save('storage_lifecycle_report.json',budget)
    print('G3 '+budget['STORAGE_CAPACITY_GATE']+'; literal_shortfall_bytes='+str(budget['SHORTFALL_BYTES']),flush=True)
    if budget['STORAGE_CAPACITY_GATE']=='PASS':
        raise RuntimeError('Capacity now passes: this CPU-only audit must not emit a stale STOP; continue engineering separately')
    verdict=dict(EXPERIMENT=A.TASK,STORAGE_LIFECYCLE_AMENDMENT='SEQUENTIAL_PER_DOCUMENT_CAPTURE_REPLAY_RELEASE',
        OLD_TOTAL_STORAGE_ESTIMATE_GB=45.4,MAX_CONCURRENT_FULL_DOCUMENT_DRIVES=1,
        OLD_EVIDENCE_INTEGRITY_GATE='PASS',FROZEN_DISCOVERY_DOCUMENT_INTEGRITY_GATE='PASS',DISCOVERY_DOC_COUNT=4,
        ANCHOR_CONDITION_COUNT=4,ANCHOR_IDENTITY_GATE='PASS',CAPTURE_EMPIRICAL_REPRODUCIBILITY_GATE='NOT_EVALUATED',
        DRIVE_RELEASE_AFTER_SEAL_ALLOWED='NOT_EVALUATED',
        ESTIMATED_SINGLE_DOC_DRIVE_GB=budget['ESTIMATED_SINGLE_DOC_DRIVE_BYTES']/1e9,
        PEAK_REQUIRED_STORAGE_GB=budget['PEAK_REQUIRED_STORAGE_BYTES']/1e9,
        AVAILABLE_STORAGE_GB=budget['AVAILABLE_STORAGE_BYTES']/1e9,STORAGE_CAPACITY_GATE='FAIL',PREMEASUREMENT_COMMIT_SHA=None,
        DRIVE_FACTORIZATION_GATE='NOT_EVALUATED',CAPTURE_SCHEMA_GATE='NOT_EVALUATED',FP_REPLAY_PARITY_GATE='NOT_EVALUATED',
        NO_QDQ_ROTATION_INVARIANCE_GATE='NOT_EVALUATED',COUNTERFACTUAL_DECOMPOSITION_GATE='NOT_EVALUATED',
        REPLAY_ISOLATION_GATE='NOT_EVALUATED',REPLAY_DETERMINISM_GATE='NOT_EVALUATED',STREAMING_REDUCER_PARITY_GATE='NOT_EVALUATED',
        DISCOVERY_EXECUTION_AUTHORIZED='NO',DISCOVERY_COMPLETE='0/16',HORIZON_2048_COMPLETE='0/16',HORIZON_8192_COMPLETE='0/16',
        HORIZON_32768_COMPLETE='0/16',COMPLETED_DOC_DRIVES_RELEASED='0/4',CONFIRMATION_EXECUTED=0,
        FULL_MODEL_CLOSED_LOOP_PROPAGATION='NOT_EVALUATED',NEW_C8_TRAINING_AUTHORIZED='NO',NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,
        FINAL_ACTION='FAIL_AND_STOP_STORAGE_CAPACITY')
    save('final_verdict.json',dict(verdict=verdict,reason='Literal Section5+Section6 storage inequality fails; one-margin alternative reported, not authorized',
        scientific_measurement_count=0,model_forward_count=0,GPU_work_started=False,drive_capture_count=0,
        old_artifacts_modified=0,files_deleted=0,full_preregistration_created=False))
    save('final_integrity_revalidation.json',dict(historical=A.history(),old_experiment=A.seal(ROOT,'SHA256SUMS'),
        anchor_manifest_sha256=A.sha(ROOT/'protocol/frozen_anchor_conditions.json')))
    text='# STORAGE_LIFECYCLE_AMENDMENT_V1 — exact byte audit and resource STOP\n\n'
    text+='Same scientific experiment; sequential per-document policy accepted. No document, anchor, metric, dtype, horizon or original seal was changed. '
    text+='All189 prior sealed files PASS. Four documents PASS exact content/token hashes, frozen order, source container, tokenizer identity and retokenization. '
    text+='Four anchors are now frozen in protocol/frozen_anchor_conditions.json: Native identity, unchanged formal H128, authoritative C6R and C7 exports; all24 layers and inherited1e-4 orthogonality limit PASS. No model was loaded.\n\n'
    text+='## Exact storage arithmetic\n\n'
    for k in ['ESTIMATED_SINGLE_DOC_DRIVE_BYTES','TEMPORARY_CAPTURE_BYTES','ATOMIC_FINALIZE_OVERHEAD_BYTES','REPLAY_TEMPORARY_BYTES',
              'RESULT_BYTES','OPERATIONAL_LIFECYCLE_PEAK_BYTES','SAFETY_MARGIN_BYTES','PEAK_REQUIRED_STORAGE_BYTES',
              'AVAILABLE_STORAGE_BYTES','LITERAL_SECTION6_TOTAL_FREE_REQUIREMENT_BYTES','SHORTFALL_BYTES']:
        text+=k+' = '+str(budget[k])+'\n\n'
    text+='The full original-dtype q/k/v/g/beta raw drive costs19,478,347,776 bytes per document. No query omitted, heads deduplicated or new compression invented. '
    text+='Final shard payload is written once and renamed on the same filesystem; it is already counted in the raw drive, not counted again as a full temporary copy. Replay disk temporary is0. '
    text+='RESULT_BYTES contains four documents\' sparse FP horizon checkpoints, two256-token engineering captures with32/256 FP states, and128MiB serialization/metadata allowance. These are estimates, not successful implementation or observed scientific runtime.\n\n'
    text+='## The specification has two different safety-margin expressions\n\n'
    text+='Section5 defines PEAK_REQUIRED_STORAGE_BYTES including SAFETY_MARGIN_BYTES. Section6 additionally subtracts that same margin from available space. '
    text+='Applied literally, operational_peak +2*margin must fit. It does not. Even raw full drive + retained sparse FP horizon references +2*5GiB exceeds current free space before engineering or metadata allowances. '
    text+='This is a storage/control-policy STOP, not a scientific or numerical failure, and not proof that sequential storage is physically infeasible under a one-margin policy.\n\n'
    alternative=budget['ONE_MARGIN_INTERPRETATION']
    text+=f"Counting the safety margin once would require{alternative['required_total_bytes']:,} bytes, leaving{alternative['headroom_bytes']:,} bytes beyond that reserved margin: capacity PASS under that alternative only. "
    text+='The audit does not silently adopt it. User direction is needed to resolve the two expressions, or additional free capacity is needed under the literal rule. No cleanup of old artifacts is authorized/performed.\n\n'
    text+='## Scope and evidence\n\nNo capture, empirical reproducibility, factorization implementation, replay, numerical engineering qualification or scientific Discovery has started. '
    text+='Preregistration and PREMEASUREMENT_COMMIT_SHA remain absent. Any commit sealing this report is AUDIT_COMMIT_ONLY. '
    text+='No scientific comparison or Confirmation hypothesis can be reported. Peak observed new drive disk usage0; completed drives released0/4. '
    text+='The next scientific stages remain blocked; this report does not authorize retrying old experiments or relaxing numerical thresholds.\n\n'
    text+='## Required final fields\n\n```text\n'+'\n'.join(k+' = '+('null' if v is None else str(v)) for k,v in verdict.items())+'\n```\n'
    write('FINAL_REPORT.md',text)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts]
    lines=[A.sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in files]
    lines.append(A.sha(ROOT/'protocol/frozen_anchor_conditions.json')+'  ../../protocol/frozen_anchor_conditions.json\n')
    write('SHA256SUMS',''.join(lines))
    print(json.dumps(verdict,sort_keys=True),flush=True)

if __name__=='__main__':main()
