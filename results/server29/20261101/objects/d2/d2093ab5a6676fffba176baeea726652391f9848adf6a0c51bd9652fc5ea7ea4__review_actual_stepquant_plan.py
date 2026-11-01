"""Independent CPU-only, read-only validation of the realized official Plan.

Writes only two new derived JSON files in independent_review. Does not load a
model or large statistics tensor, execute a forward, fit a new Plan, or use CUDA.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def store(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    return {'path':str(path),'sha256':sha(path),'bytes':Path(path).stat().st_size}


def run(root):
    import torch
    assert not torch.cuda.is_initialized()
    torch.cuda._lazy_init=lambda *a,**k: (_ for _ in ()).throw(AssertionError('CUDA forbidden'))
    torch.set_num_threads(1)
    root=root.resolve()
    out=root/'independent_review';out.mkdir(exist_ok=True)
    assert not (root/'FIRST_FAILURE.json').exists()
    auth=read(root/'AUTHORIZATION_CALIBRATION.json')
    source=read(root/'SOURCE_MANIFEST.json')
    prereg=read(root/'PREREGISTRATION.json')
    inp=read(root/'CALIBRATION_INPUTS.json')
    gate=read(root/'calibration/PLAN_CPU_QUALIFICATION.json')
    result=read(root/'calibration/CALIBRATION_RESULT.json')
    coordinator=read(root/'execution/calibration/RESULT.json')
    assert auth['status']=='PASS' and prereg['frozen'] is True
    for name,key in [('SOURCE_MANIFEST.json','source_manifest_sha256'),('RUNTIME_SOURCE_MANIFEST.json','runtime_source_manifest_sha256'),
                     ('PREREGISTRATION.json','prereg_sha256'),('MODEL_TOKENIZER_FRESH_IDENTITY.json','model_tokenizer_identity_sha256'),
                     ('CALIBRATION_INPUTS.json','calibration_inputs_sha256')]:
        assert sha(root/name)==auth[key],name
    source_count=0
    for row in source['execution_sources']:
        p=root/row['path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
        source_count+=1
    official=read(root/'OFFICIAL_STEPQUANT_SOURCE_MANIFEST.json')
    assert official['commit']=='f2deac58131219e6aecff274f8eabe88e08d2e3c'
    for row in official['assets']:
        p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    path=root/'calibration/qwen35_9b_stepquant6_plan.pt'
    digest=sha(path)
    assert gate['status']==result['status']==coordinator['status']=='PASS'
    assert coordinator['all_GPUs_released'] is True
    assert digest==gate['plan_sha256']==result['plan']['sha256'] and path.stat().st_size==result['plan']['bytes']
    # CPU mapping, no model modules imported and no native forward execution.
    plan=torch.load(path,map_location='cpu',weights_only=False)
    assert plan['format_version']==2 and plan['state_format']=='stepquant'
    assert plan['official_commit']==official['commit']
    settings=plan['settings']
    for key,val in {'nominal_bits':6,'pivots':32,'horizon':2048,'value_group_size':32,'candidates':[4,6,8],
                    'integer_budget':4352,'integer_bits_used':4352}.items():
        assert settings[key]==val,(key,settings[key])
    layers=(0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30)
    expected={'model.layers.'+str(l)+'.linear_attn':l for l in layers}
    assert plan['layer_name_to_index']==expected
    assert set(plan['plans'])==set(expected)==set(settings['calibration_tokens'])==set(plan['state_shape_per_layer'])
    assert all(v==65536 for v in settings['calibration_tokens'].values())
    assert all(v==[1,32,128,128] for v in plan['state_shape_per_layer'].values())
    assert plan['data']==inp and inp['AIME_used'] is False
    assert inp['revision']=='b08601e04326c79dfdd32d625aee71d232d685c3'
    assert inp['segments']==32 and inp['sequence_length']==2048 and inp['seed']==0
    assert inp['sample_every']==8 and inp['max_snapshots']==64 and inp['gamma']==.25
    token_path=root/inp['segments_path'];assert sha(token_path)==inp['segments_sha256']
    aggregate=hashlib.sha256();rows=[json.loads(s) for s in token_path.read_text().splitlines() if s]
    assert len(rows)==32
    for i,row in enumerate(rows):
        raw=struct.pack('<'+'q'*len(row['input_ids']),*row['input_ids'])
        assert row['segment_index']==i and len(row['input_ids'])==2048
        assert row['start']==inp['window_start_indices'][i]
        assert hashlib.sha256(raw).hexdigest()==row['token_ids_sha256']
        aggregate.update(raw)
    assert aggregate.hexdigest()==inp['official_combined_token_ids_sha256']==gate['data_token_sha256']==result['data_token_sha256']
    parquet=root/'calibration_data/wikitext2_raw_train_b08601e0.parquet'
    assert parquet.stat().st_size==inp['parquet_bytes'] and sha(parquet)==inp['parquet_sha256']
    fresh=read(root/'MODEL_TOKENIZER_FRESH_IDENTITY.json')
    assert plan['full_model_file_SHA256_identity']==fresh and fresh['status']=='PASS' and len(fresh['files'])==17
    assert all(r['PASS'] is True and r['sha256']==r['expected_sha256'] for r in fresh['files'])
    # Compare metadata files/weight stat to original complete-content SHA proof;
    # do not repeat a ~20GB model hash or load model weights in this review.
    basenames={Path(r['path']).name:r for r in fresh['files']}
    for name,value in plan['checkpoint_identity']['metadata_sha256'].items():
        assert basenames[name]['sha256']==value
    for name,value in plan['checkpoint_identity']['weights_stat'].items():
        p=Path(basenames[name]['path'])
        assert p.stat().st_size==value['size']==basenames[name]['bytes']
        assert p.stat().st_mtime_ns==value['mtime_ns']
    fp=hashlib.sha256()
    for name in ('config.json','model.safetensors.index.json'):
        fp.update(Path(basenames[name]['path']).read_bytes())
    assert fp.hexdigest()==plan['checkpoint_fingerprint']
    official_root=root/'official_source/STEPQuant_f2deac58/stepquant'
    for rel,value in plan['calibration_source_sha256'].items():
        assert sha(official_root/rel)==value
    native=Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
    assert sha(native)=='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
    statistics=root/'calibration/qwen35_9b_statistics.pt'
    statistics_digest=sha(statistics)  # Read-only streaming hash, never torch.load.
    assert statistics_digest==plan['statistics_sha256']==gate['statistics_sha256']==result['statistics']['sha256']
    assert statistics.stat().st_size==result['statistics']['bytes']
    counts={b:0 for b in (4,6,8,16)};integer_sum=0;payload=0;layer_meta=[];pivots=[]
    max_geometric_error=0.
    for name,l in sorted(expected.items(),key=lambda p:p[1]):
        p=plan['plans'][name];bits=p['bits'];impact=p['impact']
        assert p['architecture']=='gdn' and p['value_group_size']==32
        assert bits.device.type==impact.device.type=='cpu'
        assert bits.dtype==torch.int64 and impact.dtype==torch.float32
        assert tuple(bits.shape)==tuple(impact.shape)==(32,128)
        assert torch.equal(bits,bits[:,0,None].expand_as(bits))
        assert set(bits[:,0].tolist())<={4,6,8,16}
        assert bool(torch.isfinite(impact).all()) and bool((impact>0).all())
        geometric=impact.double().log().mean(-1).exp()
        ge=float((geometric-1).abs().max());max_geometric_error=max(max_geometric_error,ge)
        assert ge<1e-6
        per_counts={b:int((bits[:,0]==b).sum()) for b in counts}
        for b,n in per_counts.items():counts[b]+=n
        integer_sum+=int(bits[:,0][bits[:,0]!=16].sum())
        # B1 bitstream plus original FP16 pivot, row and column scale payload.
        code=sum(per_counts[b]*128*128*b//8 for b in (4,6,8))
        piv=per_counts[16]*128*128*2;scales=(32-per_counts[16])*128*2*2
        layer_payload=code+piv+scales;payload+=layer_payload
        assert per_counts=={int(b):n for b,n in result['per_layer'][name]['heads_per_bit'].items()}
        assert layer_payload==result['per_layer'][name]['encoded_request_payload_bytes']
        heads=[]
        for h in range(32):
            row=impact[h].contiguous()
            heads.append({'head':h,'bits':int(bits[h,0]),'FP16_pivot':int(bits[h,0])==16,
                'impact_min':float(row.min()),'impact_max':float(row.max()),'impact_mean':float(row.mean()),
                'impact_geometric_mean':float(geometric[h]),
                'impact_128_FP32_SHA256':hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).hexdigest()})
            if int(bits[h,0])==16:pivots.append({'layer_index':l,'module_name':name,'head':h})
        layer_meta.append({'module_name':name,'layer_index':l,'heads_per_bit':per_counts,
            'state_shape':[1,32,128,128],'bits_shape':[32,128],'impact_shape':[32,128],
            'calibration_tokens':settings['calibration_tokens'][name],'encoded_request_payload_bytes_B1':layer_payload,
            'payload_components':{'integer_codes_bytes':code,'FP16_pivots_bytes':piv,'FP16_row_and_column_scales_bytes':scales},
            'heads':heads})
    assert counts=={4:299,6:170,8:267,16:32} and sum(counts.values())==768
    assert integer_sum==settings['integer_bits_used']==4352
    assert integer_sum+8*32==6*768
    assert payload==result['all_layers_encoded_request_payload_bytes_B1']==10338304
    assert not torch.cuda.is_initialized()
    meta={'status':'ACTUAL_PLAN_CPU_METADATA_VERIFIED','plan_sha256':digest,'GDN_layers':24,'heads':768,
        'heads_per_bit':counts,'integer_budget':4352,'integer_bits_used':integer_sum,
        'nominal_budget_with_pivots_counted_as8':integer_sum+8*32,'nominal_bits':6,
        'actual_payload_data_mean_bits_per_element':sum(b*n for b,n in counts.items())/768,
        'encoded_request_tensor_payload_bytes_B1':payload,
        'physical_persistent_cache':'FP32 reconstructed QDQ carrier; transient packed payload only, no persistent packed claim',
        'lifetime_horizon':2048,'value_group_size':32,'impact_gamma':.25,
        'max_impact_head_geometric_mean_abs_error':max_geometric_error,'FP16_pivot_locations':pivots,'layers':layer_meta}
    saved=store(out/'REALIZED_PLAN_METADATA.json',meta)
    review={'task_id':auth['task_id'],'RUN_ID':auth.get('RUN_ID'),'status':'PASS',
        'scope':'Independent CPU inspection of the actual realized Plan and read-only streaming SHA validation of statistics. No model load, CUDA, forward, new calibration fit, generation or correctness scoring.',
        'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'review_source_sha256':sha(__file__),'plan':{'path':str(path),'sha256':digest,'bytes':path.stat().st_size},
        'statistics':{'path':str(statistics),'sha256':statistics_digest,'bytes':statistics.stat().st_size,'inspection':'streaming SHA/bytes only; not deserialized or downloaded'},
        'checks':{'actual_completed_calibration_and_CPU_gate':'PASS','all24_expected_GDN_modules':'PASS','768_whole_head_mixed_precision':'PASS',
            '32_FP16_global_pivots':'PASS','nominal6_residual_integer_budget4352_exact':'PASS','impact_FP32_positive_finite_geometric_normalization':'PASS',
            'official_frozen_source_and_lifetime2048_definition':'PASS','same32x2048_seed0_non_AIME_calibration_tokens':'PASS',
            'full17_model_tokenizer_content_SHA_receipt_binding':'PASS','actual_statistics_streaming_SHA':'PASS','coordinator_release_receipt':'PASS'},
        'bindings':{'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'runtime_source_manifest_sha256':sha(root/'RUNTIME_SOURCE_MANIFEST.json'),
            'prereg_sha256':sha(root/'PREREGISTRATION.json'),'authorization_sha256':sha(root/'AUTHORIZATION_CALIBRATION.json'),
            'calibration_inputs_sha256':sha(root/'CALIBRATION_INPUTS.json'),'model_identity_receipt_sha256':sha(root/'MODEL_TOKENIZER_FRESH_IDENTITY.json'),
            'official_source_files_fresh_SHA':len(official['assets']),'all_frozen_source_files_fresh_SHA':source_count,
            'official_commit':plan['official_commit'],'data_token_sha256':aggregate.hexdigest(),'model_file_count':17},
        'realized_metadata':saved,'heads_per_bit':counts,'encoded_request_tensor_payload_bytes_B1':payload,
        'definition':'Unchanged official GDN calibration uses lifetime-weighted per-head mixed4/6/8 distortions, initial global DP, stable residual-risk32 FP16 pivots, residual integer DP. Horizon2048 and gamma.25 weighted-row/column-LS codec remain original source.',
        'limits':['Does not recompute all statistics-derived distortion costs, pivot ranking or allocator optimality from the3GiB tensor. Frozen official source, actual fit receipt, Plan structure and full statistics hash are independently checked.',
            '17 model/tokenizer SHA values are bound to the earlier fresh complete-content proof and current weight stat; this independent review does not repeat the ~20GB model stream hash.',
            'Official FP32 shadow calibration states are not claimed byte-identical to native chunk intermediate states.',
            'This is Plan qualification, not four-GPU canary, formal AIME or scientific quality completion.'],
        'CUDA_initialized':False,'full_generation':'NOT_RUN_BY_REVIEWER'}
    saved_review=store(out/'CALIBRATION_PLAN_INDEPENDENT_REVIEW.json',review)
    print(json.dumps({'status':'PASS','review':saved_review,'metadata':saved,'heads_per_bit':counts},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    run(p.parse_args().root)
