"""Verify saved completed artifacts without re-running model arithmetic."""
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
import functional_worker as w

D=Path(__file__).resolve().parent
spec=w.read(D/'FROZEN_SPEC.json')
assert not (D/'STOP.json').exists()
completion=w.read(D/'FORMAL_COMPLETE.json')
assert completion['status']=='PASS' and completion['students']==4 and completion['FP_references']==2
assert w.read(D/'QUALIFICATION_GATES_COMPLETED.json')['status']=='ALL_EIGHT_GATES_PASS'
freeze=w.read(D/'FORMAL_CODE_FREEZE.json')
for name,h in freeze['files'].items(): assert w.sha(D/name)==h, name
assert w.sha(D/'FORMAL_CODE_FREEZE.json')==w.read(D/'FORMAL_STARTED.json')['formal_code_freeze_sha256']
for name,row in spec['source_assets'].items():
    s=Path(name).stat()
    assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'],name
    if not name.endswith('.safetensors'): assert w.sha(name)==row['sha256'],name
for name,h in spec['historical_STOP_sha256'].items():assert w.sha(name)==h,name
parity=w.read(D/'CHECKPOINT_FORK_PARITY_COMPLETED.json')
assert parity['status']=='PASS' and len(parity['pairs'])==6
for pair in parity['pairs']:
    assert pair['status']=='PASS'
    for key in ('uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256',
                'audit_state_fingerprint','checkpoint_sha256'):
        assert pair['parent'][key]==pair['resumed'][key]
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
positions=spec['primary']['positions']
assert positions==[8192+64*k for k in range(384)]
manifest=w.read(D/'delivery/FP_REFERENCE_MANIFEST.json')
verdict=w.read(D/'delivery/DOCUMENT_VERDICT.json')
audits=[]
for entry in spec['documents']:
    doc=entry['document']
    fp=w.read(D/'formal'/f'{doc}__FP__formal.json')
    assert fp['status']=='PASS' and fp['endpoint']==32768 and fp['sample_positions']==positions
    assert fp['formal_observer_uninterrupted_parity_PASS'] and fp['all_finite']
    digest=hashlib.sha256(); sampled=[]
    for item in fp['references']:
        assert w.sha(item['path'])==item['sha256']
        payload=torch.load(item['path'],map_location='cpu',weights_only=False)
        assert payload['position']==item['position']
        if item['logits']:
            value=payload['logits']
            assert value.dtype==torch.float32 and tuple(value.shape)==(1,1,157184)
            assert bool(torch.isfinite(value).all())
            sampled.append(item['position'])
            digest.update(item['position'].to_bytes(8,'little'))
            digest.update(w.engine.tensor_bytes(torch,value))
        if item['state']:assert bool(torch.isfinite(payload['state']).all())
    assert sampled==positions and digest.hexdigest()==fp['sampled_logits_content_sha256']
    assert len(fp['references'])==387
    reference=next(x for x in manifest['references'] if x['document']==doc)
    assert reference['actual_sample_count']==384
    assert reference['sampled_logits_content_sha256']==digest.hexdigest()
    assert reference['reference_result_sha256']==w.sha(D/'formal'/f'{doc}__FP__formal.json')
    means={}
    for condition in ('FP',entry['P1_predicted_winner'],entry['L7_predicted_winner']):
        result=w.read(D/'formal'/f'{doc}__{condition}__formal.json')
        assert result['status']=='PASS' and result['endpoint']==32768 and result['all_finite']
        assert result['observer_qualified_before_first_primary_reduction']
        assert result['token_sha256']==entry['token_file']['sha256']==fp['token_sha256']
        assert result['gpu_uuid']==fp['gpu_uuid']==spec['execution']['gpu_uuid']
        assert result['primary_spec_sha256']==w.sha(D/'FROZEN_SPEC.json')
        assert result['secondary_spec_sha256']==w.sha(D/'SECONDARY_SPEC.json')
        assert result['checkpoint_sha256']==w.sha(D/'runs'/f'{doc}__{condition}__8192.pt')
        assert w.read(D/'formal'/f'{doc}__{condition}__formal.NATURAL_EXIT.json')['result_sha256']==w.sha(D/'formal'/f'{doc}__{condition}__formal.json')
        if condition=='FP':continue
        samples=result['primary_samples']
        assert len(samples)==384 and [x['position'] for x in samples]==positions
        assert all(x['vocabulary_dimension']==157184 and math.isfinite(x['KL']) for x in samples)
        assert sum(x['KL'] for x in samples)/384==result['primary_mean_KL']
        assert result['persistent_integrity']['passed']
        assert result['persistent_integrity']['every_token_layer_qdq_count']==32768*18
        assert result['persistent_integrity']['actual_next_kernel_equal_post_qdq_count']==32767*18
        assert result['quantization_diagnostics']['codes']['out_of_range']==0
        assert result['quantization_diagnostics']['scale']['count']==24576*18
        assert result['quantization_diagnostics']['J_INJ_original_replay_norm']['count']==24576*18
        assert sorted(map(int,result['state_diagnostics']))==[8192,16383,24575,32767]
        assert result['E_future']==result['Hidden_KL']=='NOT_FROZEN_NOT_COMPUTED'
        means[condition]=result['primary_mean_KL']
    delta=means[entry['P1_predicted_winner']]-means[entry['L7_predicted_winner']]
    winner='UNRESOLVED' if abs(delta)<=spec['resolution']['threshold'] else 'P1' if delta<0 else 'L7'
    delivered=next(x for x in verdict['documents'] if x['document']==doc)
    assert delivered['Delta']==delta and delivered['prediction_winner']==winner
    audits.append(dict(document=doc,sampled_FP_logits_SHA=digest.hexdigest(),
        primary_samples_per_student=384,all_full_vocabulary=True,Delta=delta,winner=winner))
    print('DOCUMENT_ARTIFACT_AUDIT_PASS',doc,flush=True)
for line in (D/'delivery/SHA256SUMS.txt').read_text().splitlines():
    h,name=line.split('  ',1);assert w.sha(D/'delivery'/name)==h,name
assert not torch.cuda.is_initialized()
receipt=dict(status='PASS',CUDA_initialized=False,all_6_trajectories_32768=True,
    all_6_checkpoint_fork_pairs_exact=True,full_FP_reference_payloads_SHA_checked=True,
    original_384_sample_sum_reproduced_exactly=True,source_metadata_and_nonweight_SHA_preserved=True,
    historical_STOP_SHA_preserved=True,delivery_SHA256SUMS_PASS=True,documents=audits,
    completion_time_Asia_Shanghai=datetime.fromtimestamp(completion['completed_epoch'],timezone(timedelta(hours=8))).isoformat(),
    no_model_or_KL_reexecution=True)
w.publish(D/'COMPLETED_ARTIFACT_AUDIT.json',receipt)
print(json.dumps(receipt,ensure_ascii=False),flush=True)
