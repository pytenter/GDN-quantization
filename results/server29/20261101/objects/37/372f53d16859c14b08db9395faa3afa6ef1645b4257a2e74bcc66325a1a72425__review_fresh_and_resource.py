"""Read-only stdlib audit of completed fresh/resource receipts and raw traces."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def asset(p,root):
    return {'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size}


def save(p,obj):
    with p.open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}


def bindings(record,auth,root,mode):
    for k in ('prereg_sha256','source_manifest_sha256','runtime_source_manifest_sha256','model_tokenizer_identity_sha256','plan_sha256'):
        assert record[k]==auth[k],(mode,k)
    assert record['authorization_sha256']==sha(root/('AUTHORIZATION_'+mode.upper()+'.json'))


def run(root):
    root=root.resolve();out=root/'independent_review'
    assert not (root/'FIRST_FAILURE.json').exists()
    ledger=read(root/'GPU_RESOURCE_LEDGER.json')
    nvml=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    active_pids={int(line.split(',')[0]) for line in nvml.splitlines() if line.strip()}
    layers={str(i) for i in (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30)}
    auths={mode:read(root/('AUTHORIZATION_'+mode.upper()+'.json')) for mode in ('canary','fresh','resource')}
    for mode,auth in auths.items():
        assert auth['dispatch_GPUs']==[0,1,2,3] and auth['status']=='PASS'
        assert sha(root/'PREREGISTRATION.json')==auth['prereg_sha256']
        assert sha(root/'SOURCE_MANIFEST.json')==auth['source_manifest_sha256']
        assert sha(root/'RUNTIME_SOURCE_MANIFEST.json')==auth['runtime_source_manifest_sha256']
        assert sha(root/'calibration/qwen35_9b_stepquant6_plan.pt')==auth['plan_sha256']
    def completion(mode):
        p=root/'execution'/mode/'RESULT.json';receipt=read(p)
        assert receipt['status']=='PASS' and receipt['all_GPUs_released'] is True
        rows=[r for r in ledger['dispatches'] if r['stage']==mode]
        assert len(rows)==4 and {r['gpu'] for r in rows}==set(range(4))
        for row in rows:
            assert row['returncode']==0 and row['process_exited'] is True and row['NVML_released'] is True
            assert row['pid'] not in active_pids
            assert row['gpu_uuid']==auths[mode]['gpu_uuid_map'][str(row['gpu'])]
            assert row['release_monotonic']>=row['dispatch_monotonic']
            assert row['occupied_seconds']>=row['release_monotonic']-row['dispatch_monotonic']-1e-6
            assert str(root/'source/shared_runtime_worker.py') in row['identity']['cmdline']
        return {'coordinator_result':asset(p,root),'recorded_ledger_sha256_at_stage_completion':receipt['ledger_sha256'],
            'release_proofs':rows,'occupied_GPU_seconds':sum(r['occupied_seconds'] for r in rows),
            'note':'Global ledger is append-refreshed by later stages; prior per-stage ledger SHA is retained as recorded, not falsely compared to the later mutable ledger.'}
    comparison=('generated_token_ids','input_ids_hash','initial_generator_state_sha256','final_generator_state_sha256',
                'sampling_audit','hadamard_audit','qualification_trace')
    pairs=[]
    for gpu in range(4):
        for mode in ('canary','fresh'):
            p=root/'qualification'/('gpu'+str(gpu))/mode/'RESULT.json';res=read(p)
            assert res['status']=='PASS' and res['units']==2 and res['mode']==mode
            bindings(res,auths[mode],root,mode)
            assert res['gpu']['UUID']==auths[mode]['gpu_uuid_map'][str(gpu)]
        for method in ('STEPQUANT6','H_DUALM'):
            paths=[root/'qualification'/('gpu'+str(gpu))/mode/('-1_'+method+'.json') for mode in ('canary','fresh')]
            rows=[read(p) for p in paths]
            for mode,p,row in zip(('canary','fresh'),paths,rows):
                assert row['status']=='COMPLETE' and row['runtime_error'] is None and row['nonfinite'] is False
                assert row['method']==method and row['gpu']['UUID']==auths[mode]['gpu_uuid_map'][str(gpu)]
                assert row['output_tokens']==row['generated_token_count']==len(row['generated_token_ids'])==128
                assert row['retry_count']==0
                bindings(row,auths[mode],root,mode)
                check=dict(row);stored=check.pop('output_payload_sha256');assert canonical(check)==stored
                q=row['quantizer_audit'];assert q['calls']==128 and q['prefill_calls']==1 and q['decode_calls']==127
                assert q['prefill_quantized'] is True and q['layer_writebacks']==24*128 and q['next_consumption_checks']==24*127
                trace=row['qualification_trace'];assert len(trace['forward'])==len(trace['QDQ'])==128
                assert {v['step'] for v in trace['QDQ']}==set(range(128))
                assert all(set(t['layers'])==layers for t in trace['QDQ'])
                assert all(set(t['state_sha256'])==set(t['helper'])==layers for t in trace['forward'])
            assert all(rows[0][key]==rows[1][key] for key in comparison),(gpu,method)
            pairs.append({'gpu':gpu,'UUID':rows[1]['gpu']['UUID'],'method':method,'output_tokens':128,
                'token_ids_sha256':canonical(rows[1]['generated_token_ids']),
                'initial_generator_state_sha256':rows[1]['initial_generator_state_sha256'],
                'final_generator_state_sha256':rows[1]['final_generator_state_sha256'],
                'full_forward_and_QDQ_trace_sha256':canonical(rows[1]['qualification_trace']),
                'canary':asset(paths[0],root),'fresh':asset(paths[1],root),
                'canary_output_payload_sha256':rows[0]['output_payload_sha256'],
                'fresh_output_payload_sha256':rows[1]['output_payload_sha256'],
                'all_compared_fields_exact':True})
    fresh={'task_id':auths['fresh']['task_id'],'RUN_ID':auths['fresh'].get('RUN_ID'),'status':'PASS',
        'qualification_scope':'PREFIX_ONLY','actual_new_method_numeric_prefix_tokens':128,
        'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_source_sha256':sha(__file__),
        'review_method':'Independent stdlib comparison of complete same-GPU canary/fresh token, generator, sampling, H, forward and QDQ traces plus canonical raw-payload SHA. No forward/GPU computation or scoring.',
        'pair_comparisons':pairs,'canary_completion':completion('canary'),'fresh_completion':completion('fresh'),
        'checks':{'4GPUs_x2methods':'PASS','128_complete_tokens_each':'PASS','complete_same_GPU_fresh_process_bytes':'PASS',
                  'payload_SHA_and_source_model_tokenizer_plan_protocol_GPU_identity':'PASS','actual_exit_and_NVML_release':'PASS'},
        'scientific_81920_numeric_status':'NOT_TESTED_NEW_METHODS','cross_GPU_exactness':'NOT_CLAIMED_BY_THIS_SAME_GPU_REVIEW',
        'limit':'This128-token qualification does not establish long-horizon81920 GDN stability or AIME quality.'}
    saved_fresh=save(out/'FRESH_STAGE_INDEPENDENT_REVIEW.json',fresh)
    auth=auths['resource'];assert auth['resource_min_free_bytes']==1024**3
    config=read('/data/zypan/modelscope_models/Qwen3.5-9B/config.json')['text_config']
    layer_types=config.get('layer_types') or ['linear_attention' if (i+1)%4 else 'full_attention' for i in range(config['num_hidden_layers'])]
    assert layer_types.count('full_attention')==8 and config['num_key_value_heads']==4
    assert config['num_attention_heads']==16 and config['head_dim']==256
    n=81920+auth['max_prompt_tokens'];assert n==82293
    kv=8*2*1*4*n*256*2;workspace=2*1*4*(n+1)*256*2+2*1*16*(n+1)*256*2
    resource_rows=[]
    for gpu in range(4):
        p=root/'qualification'/('gpu'+str(gpu))/'resource/RESULT.json';row=read(p)
        assert row['status']=='PASS_RESOURCE_PRESSURE_ONLY'
        bindings(row,auth,root,'resource')
        assert row['total_context_slots']==n and row['output_cap']==81920
        assert row['KV_BF16_bytes']==kv and row['additional_workspace_bytes']==workspace
        assert row['long_SDPA_query_shape']==[1,16,1,256] and row['long_SDPA_KV_shape']==[1,16,n+1,256]
        assert row['long_SDPA_output_dtype']=='torch.bfloat16' and row['long_SDPA_finite'] is True
        assert row['CUDA_driver_free_bytes']>=1024**3
        assert row['peak_reserved_bytes']>=row['peak_allocated_bytes']>=row['model_before_bytes']+kv+workspace
        assert row['science_numeric_81920']=='NOT_TESTED_NEW_METHODS'
        resource_rows.append({'gpu':gpu,'UUID':auth['gpu_uuid_map'][str(gpu)],'receipt':asset(p,root),**row})
    resource={'task_id':auth['task_id'],'RUN_ID':auth.get('RUN_ID'),'status':'PASS',
        'qualification_scope':'RESOURCE_PRESSURE_ONLY','new_method_numerical_qualification':'PREFIX_ONLY_128_TOKENS',
        'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_source_sha256':sha(__file__),
        'actual_model_attention_layout':{'full_attention_layers':8,'KV_heads':4,'query_heads':16,'head_dim':256},
        'allocation_context_slots':n,'attention_workspace_context_slots':n+1,'max_output_tokens':81920,
        'expected_KV_BF16_bytes':kv,'expected_additional_workspace_bytes':workspace,'required_minimum_driver_free_bytes':1024**3,
        'GPU_results':resource_rows,'completion':completion('resource'),
        'checks':{'four_actual_live_model_resource_workers':'PASS','native_long_SDPA_finite':'PASS','shape_and_allocated_bytes':'PASS',
                  'at_least1GiB_free_on_each_GPU':'PASS','actual_CUDA_peak_receipts':'PASS','actual_exit_and_NVML_release':'PASS'},
        'science_numeric_81920':'NOT_TESTED_NEW_METHODS','AIME_quality':'NOT_SCORED_BY_REVIEW',
        'limits':['Real live-model allocation of synthetic long KV/workspace and one native SDPA per GPU; no81920-token GDN trajectory.',
                  'Persistent recurrent carrier remains FP32 QDQ; no persistent physical packed-state claim.',
                  'PASS concerns resource pressure only; numeric qualification remains128-token PREFIX_ONLY until formal completion.']}
    saved_resource=save(out/'RESOURCE_STAGE_INDEPENDENT_REVIEW.json',resource)
    print(json.dumps({'fresh':saved_fresh,'resource':saved_resource,'status':'PASS'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    run(p.parse_args().root)
