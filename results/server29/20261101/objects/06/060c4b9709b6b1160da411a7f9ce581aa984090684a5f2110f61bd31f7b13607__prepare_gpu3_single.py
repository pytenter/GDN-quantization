"""CPU-only identity registration for the newly authorized GPU3-only attempt.

Copies frozen scientific bytes; never starts a model, worker, or CUDA context.
The initial read-only admission window is single-shot and cannot select another GPU.
"""
from pathlib import Path
import ast, hashlib, json, os, shutil, subprocess, sys, time, traceback
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent
D=Path(os.environ['LING_PROSPECTIVE_RESULTS']).resolve()
C=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1')
O=Path('/data/zypan/worktrees/ling-kda-prospective-execution05')
H=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05')
TARGET='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'
EXECUTION='LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V1'
BASE='d03bd1b1122e93f89c5bc06b212b0b52b4974eed'
from cleanroom_common import read, sha, publish as atomic_publish, compare_exact
from frozen_runtime_profile import validate_files, CONFIG_SHA256

def publish(path,value):
    path=Path(path)
    if path.exists():
        assert read(path)==value,('PREPARATION_IDENTITY_CONFLICT',str(path))
    else: atomic_publish(path,value)

def copy_exact(source,target):
    source,target=Path(source),Path(target)
    assert source.is_file() and not source.is_symlink(),('NOT_PLAIN_FROZEN_ASSET',str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): assert target.read_bytes()==source.read_bytes(),('COPY_CONFLICT',str(target))
    else:
        with target.open('xb') as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())
    return dict(source=str(source),destination=str(target),sha256=sha(target),bytes=target.stat().st_size)

def main():
    begin=time.time()
    assert D.name==EXECUTION and W.name=='ling-kda-4doc-cleanroom-gpu3-single-v1'
    assert not (D/'RUN_CONTROLLER.json').exists() and not (D/'STOP.json').exists()
    assert not (D/'audits/PREPARATION_COMPLETE.json').exists(),'SINGLE_SHOT_PREPARATION_NO_REPEAT'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==BASE
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=O,text=True).strip()=='902d7974c89465df882403e96cc8dcc8dc1c0e85'
    old_stop='b831befc5b34d80143a5152ef85da1b84a8b904be0f14af72f13583e77d0afda'
    old_manifest='1a14787eb60b7385cee5b37f1662feff2fc45aad2cff343b4fb331a3424b1889'
    assert sha(C/'STOP.json')==old_stop and sha(C/'hashes/DELIVERY_VERIFICATION.json')==old_manifest
    copied=[]
    for n in ['CALIBRATION_CONFIG.json','FROZEN_SPEC.json','FROZEN_4DOC_SELECTION.json','SECONDARY_SPEC.json','MODEL_WEIGHT_SHA256.txt','SELECTION_FREEZE_RECEIPT.json']:
        copied.append(copy_exact(C/n,D/n))
    for p in sorted((C/'preregistration').rglob('*')):
        if p.is_file():copied.append(copy_exact(p,D/'preregistration'/p.relative_to(C/'preregistration')))
    for n,h in [('FROZEN_SPEC.json','8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1'),('FROZEN_4DOC_SELECTION.json','a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b'),('preregistration/NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md','54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722')]:
        assert sha(D/n)==h,('FROZEN_SCIENCE_SHA_CONFLICT',n)
    publish(D/'audits/FROZEN_ASSET_COPY.json',dict(status='PASS_BYTE_IDENTICAL',files=copied,previous_attempt_read_only=True,no_historical_STOP_or_current_results_copied=True))
    auth='用户在当前聊天明确授权：你就只使用一张GPU3来跑就行了。该授权在旧Clean-Room STOP交付及只读GPU空闲监控之后给出；本轮独立身份，仅GPU3，最多1个GPU worker，36.0新增GPUh硬预算及全部冻结科学/数值/资源门禁不变。不得复用旧STOP为当前状态，不覆盖历史、不自动迁移或开启失败后的第二轮。'
    publish(D/'audits/GPU3_SINGLE_USER_AUTHORIZATION.json',dict(status='EXPLICIT_HUMAN_AUTHORIZATION',user_text='你就只使用一张GPU3来跑就行了',execution_id=EXECUTION,scientific_protocol='LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1',previous_cleanroom_STOP=dict(path=str(C/'STOP.json'),sha256=old_stop,delivery_manifest_sha256=old_manifest),MAX_CONCURRENT_GPUS=1,MAX_TOTAL_NEW_GPUH=36.0,authorized_devices={TARGET:3},interpretation=auth,scope='same frozen four-document32K experiment with independent new execution identity; not permission to relax any scientific or physical gate'))
    validate_files()
    math_names=['functional_worker.py','formal_worker.py','persistent_engine_v1.py','v2_runtime.py','objective.py','run_matched_l7.py','frozen_runtime_profile.py','prospective_common.py']
    maths={}
    for n in math_names:
        assert (W/n).read_bytes()==(O/n).read_bytes(),('MATHEMATICAL_IDENTITY_CONFLICT',n)
        maths[n]=sha(W/n)
    from formal_checkpoint_control import adapted_source
    adapted=adapted_source();assert "primary = sum(x['KL'] for x in kl)/384" in adapted
    publish(D/'audits/MATHEMATICAL_IDENTITY.json',dict(status='PASS',unchanged_file_SHA256=maths,formal_control_adapter='unchanged CleanRoom reversible control hooks; source reversal restores original mathematical worker byte-for-byte',formal_adapter_source_sha256=sha(W/'formal_checkpoint_control.py'),adapted_formal_AST_SHA256=hashlib.sha256(ast.dump(ast.parse(adapted),include_attributes=False).encode()).hexdigest(),mathematical_kernel_changes=False))
    actual=read(C/'audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json')
    assert actual['status']=='PASS_ACTUAL_BYTES' and len(actual['model_files'])==32 and len(actual['input_and_rotation_files'])==24
    checked={}
    for name in ['model_files','input_and_rotation_files']:
        checked[name]=[]
        for row in actual[name]:
            p=Path(row['path']);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],('ACTUAL_ASSET_SHA_FAILURE',str(p))
            checked[name].append(dict(row))
    spec=read(D/'FROZEN_SPEC.json');model=Path(spec['model']['path']);model_metadata=[]
    for filename,key in [('config.json','config_sha256'),('generation_config.json','generation_config_sha256'),('tokenizer_config.json','tokenizer_config_sha256'),('tokenizer.json','tokenizer_sha256'),('model.safetensors.index.json','weight_index_sha256')]:
        p=model/filename;assert sha(p)==spec['model'][key],('MODEL_METADATA_CHANGED',filename)
        model_metadata.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    publish(D/'audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json',dict(status='PASS_ACTUAL_BYTES',**checked,model_metadata=model_metadata,elapsed_CPU_seconds=time.time()-begin,new_GPU_worker_count=0))
    audit=read(C/'PREREGISTRATION_RECONCILIATION.json')
    old_history=read(C/'audits/HISTORICAL_STOP_AND_MANIFEST_IDENTITIES.json')
    for row in old_history['history']:
        assert sha(Path(row['execution'])/'STOP.json')==row['STOP_sha256']
        matches=[p for p in [Path(row['execution'])/'RESULT_AND_DELIVERY_HASHES.json',Path(row['execution'])/'delivery/RESULT_AND_DELIVERY_HASHES.json'] if p.exists() and sha(p)==row['manifest_sha256']]
        assert matches,('HISTORICAL_MANIFEST_CHANGED',row['execution'])
    old_history['history'].append(dict(execution=str(C),STOP_sha256=old_stop,manifest_sha256=old_manifest,manifest_path=str(C/'hashes/DELIVERY_VERIFICATION.json')))
    publish(D/'audits/HISTORICAL_STOP_AND_MANIFEST_IDENTITIES.json',old_history)
    for n,h in audit['frozen_protocol_files'].items():assert sha(Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')/n)==h,n
    for n,h in audit['scorer_source_SHA'].items():assert sha(n)==h,n
    for row in audit['source_assets']:
        p=Path(row['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']) and sha(p)==row['sha256'],('HISTORICAL_SOURCE_ASSET_CHANGED',str(p))
    for n,h in audit['historical_STOP_preserved'].items():assert sha(n)==h,n
    audit['historical_STOP_preserved'][str(C/'STOP.json')]=old_stop
    audit.update(status='PASS',current_execution=str(D),authorization_reconciliation=auth,selected_GPU=3,selected_GPU_UUID=TARGET,selected_documents=[d['document'] for d in spec['documents']],CUDA_initialized=False,new_long_GPU_scientific_units=0,disk_free_bytes=shutil.disk_usage(D).free,elapsed_seconds=time.time()-begin,actual_gpu_inventory='Not inherited: current GPU3-only actual read-only samples are in audits/INITIAL_RESOURCE_SAMPLES.json',actual_compute_inventory='Not inherited: current GPU3-only actual process evidence is in audits/INITIAL_RESOURCE_SAMPLES.json')
    assert audit['disk_free_bytes']>=audit['required_storage_plan_bytes']
    publish(D/'PREREGISTRATION_RECONCILIATION.json',audit)
    selection=read(D/'FROZEN_4DOC_SELECTION.json')
    expected=[('pg19-test-41603','H_to_L7_a50','H'),('pg19-test-2544','H_to_L6_a50','H_to_L7_a50'),('pg19-test-7987','H_to_L7_a50','H_to_L7R_a25'),('pg19-test-26183','H_to_L6_a50','H_to_L7R_a25')]
    assert [(x['document'],x['P1_predicted_rotation'],x['L7_predicted_rotation']) for x in selection['documents']]==expected
    mapping={doc:TARGET for doc,_,_ in expected}
    publish(D/'EXECUTION_ROUTING.json',dict(execution=str(D),execution_id=EXECUTION,source_worktree=str(W),reference_experiment=str(D),authorized_devices={TARGET:3},document_devices=mapping,formal_global_order=selection['execution_order'],formal_serial=True,qualification_parallel_lanes=1,MAX_CONCURRENT_GPUS=1,MAX_TOTAL_NEW_GPUH=36.0,original_checkpoint_contract_sha256=sha(D/'FROZEN_SPEC.json'),initial_device_selection='explicit new user authorization fixes GPU3 only before any current model worker; no discovery/migration/fallback to any other card'))
    publish(D/'FP_PAIRING_CONTRACT_GATE.json',dict(status='PASS',reference_experiment=str(D),spec_sha256=sha(D/'FROZEN_SPEC.json'),mode='CLEANROOM_OWN_FP_SAME_DOC_UUID_FROZEN_RUNTIME_GPU3_ONLY'))
    import torch
    import functional_worker as f
    assert not torch.cuda.is_initialized()
    key='pg19-test-26183__FP';cp=H/'runs'/(key+'__8192.pt');parent=H/'runs'/(key+'__parent.json');fresh=H/'runs'/(key+'__resumed_parity.json');qpath=H/'qualification_units'/(key+'.json');oldq=read(qpath)
    h='2e6abbe74fc5266fd26c3b74da4977cdd3849382534852503fb68a10e0f77f4f'
    assert sha(cp)==h==oldq['checkpoint_sha256'] and sha(parent)==oldq['parent_sha256'] and sha(fresh)==oldq['resumed_sha256']
    p=read(parent);oldfresh=read(fresh);compare_exact(p,oldfresh,TARGET)
    profiles={}
    for stage in ['parent','resumed_parity']:
        path=H/'runs'/(key+'__'+stage+'.RUNTIME_PROFILE.json');profile=read(path)
        assert profile['status']=='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE' and profile['forward_kernels_executed']
        assert profile['fixed_FLA_config_SHA256']==CONFIG_SHA256
        assert profile['actually_used_forward_configurations']['causal_conv1d_update_kernel']==dict(kwargs={'BD':8},num_ctas=1,num_stages=3,num_warps=8)
        assert profile['actually_used_forward_configurations']['layer_norm_gated_fwd_kernel']==dict(kwargs={'BT':16},num_ctas=1,num_stages=3,num_warps=8)
        profiles[stage]=dict(path=str(path),sha256=sha(path))
    saved=torch.load(cp,map_location='cpu',weights_only=False);entry=next(x for x in selection['documents'] if x['document']=='pg19-test-26183')
    assert saved['schema']=='full_native_hybrid_cache_v1' and saved['next_position']==8192 and saved['last_position']==8191
    assert saved['frozen_spec_sha256']==sha(D/'FROZEN_SPEC.json') and saved['token_sha256']==entry['document_identity']['token_file_sha256']
    assert saved['recurrent_layers']==read(D/'CALIBRATION_CONFIG.json')['layers']
    assert saved['audit_state'] is None and saved['quantizer_state']['rotation_sha256'] is None
    assert f.object_fingerprint(torch,saved['cache'])==saved['cache_fingerprint']
    rng={k:f.object_fingerprint(torch,saved[k]) for k in ['torch_rng','cuda_rng','python_rng','numpy_rng']}
    assert torch.is_tensor(saved['torch_rng']) and saved['torch_rng'].dtype==torch.uint8 and saved['torch_rng'].device.type=='cpu'
    assert torch.is_tensor(saved['cuda_rng']) and saved['cuda_rng'].dtype==torch.uint8 and saved['cuda_rng'].device.type=='cpu'
    cache_fingerprint=saved['cache_fingerprint'];prefix=saved['prefix_full_logit_sha256'];del saved
    reuse={key:dict(original_checkpoint=str(cp),checkpoint_sha256=h,original_parent=str(parent),parent_sha256=sha(parent),original_qualification_sha256=sha(qpath),current_fresh64_required=True)}
    for suffix,source in [('__8192.pt',cp),('__parent.json',parent)]:
        target=D/'runs'/(key+suffix);assert not target.exists() and not target.is_symlink();target.symlink_to(source)
    publish(D/'audits/REUSED_FP_CPU_IDENTITY.json',dict(status='PASS_ACTUAL_CPU_BYTES_NATIVE_CACHE_PROFILE_RNG_IDENTITY',key=key,UUID=TARGET,checkpoint_sha256=h,cache_fingerprint=cache_fingerprint,prefix_full_logit_sha256=prefix,RNG_fingerprints=rng,profiles=profiles,old_exact64_historical_only=True,current_fresh64_required=True,current_qualification_not_yet_passed=True,CUDA_initialized=torch.cuda.is_initialized()))
    publish(D/'reused_FP_bindings.json',dict(status='ELIGIBLE_SAME_UUID_FP_PREFIX_VERIFIED_PENDING_CURRENT_FRESH64',units=reuse,not_reused=[dict(document='pg19-test-41603',reason='old prefix UUID is GPU2; user authorizes only GPU3; rebuild own prefix instead of cross-device grandfathering',old_UUID='GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca',new_UUID=TARGET,old_checkpoint_sha256='367ad2083285fd5a74c69ead6b84e9dbe443dfd22e628ab038cf0966cc10a0a7')],other_historical_checkpoint_reuse=False))
    hist=[]
    for path in Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1').rglob('*__formal.NATURAL_EXIT.json'):
        j=read(path)
        if j.get('natural_exit'):hist.append(dict(path=str(path),sha256=sha(path),seconds=j['wall_seconds']))
    publish(D/'preregistration.json',dict(TASK='LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1',execution_id=EXECUTION,scientific_protocol='LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1',preregistration_sha256='54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722',selection_sha256=sha(D/'FROZEN_4DOC_SELECTION.json'),FROZEN_SPEC_sha256=sha(D/'FROZEN_SPEC.json'),frozen_epoch=selection['frozen_epoch'],FULL_32K_EXECUTION_AUTHORIZED=True,MAX_CONCURRENT_GPUS=1,MAX_TOTAL_NEW_GPUH=36.0,initial_devices={TARGET:3},historical_formal_measurements=hist,historical_formal_worker_seconds=[r['seconds'] for r in hist],new_qualification_workers_planned=23,new_parent_workers=11,current_fresh64_workers=12,canary_model_workers=1,formal_workers=24,formal_scientific_trajectories=12,planned_long_checkpoint_cursor=16384,planned_overlap_tokens=64,global_formal_order=selection['execution_order'],all_GPU_workers_serial=True,no_new_scientific_keys=True,stats_unit='document',old_MIXED_not_combined_n6=True,new_human_authorization=auth))
    publish(D/'engineering_change_manifest.json',dict(status='REGISTERED_BEFORE_GPU',base_commit=BASE,changes=['fixed authorized GPU3 UUID for all four documents and every worker','maximum one scheduler GPU worker for canary,qualification,and formal','CPU registration for new explicit authorization and preserved oldSTOP; math/FLA/quantizer/KL/strict comparison unchanged'],unchanged_math=maths,formal_adapter_exact_source_reversibility=True,extra_headroom_bytes=2*2**30,low_util_threshold_percent=2,resource_window_samples=3,resource_sample_interval_seconds=5,hard_new_GPUh=36.,protective_stop_GPUh=35.8,automatic_failed_science_retry=False,automatic_second_attempt=False,allowlisted_devices_frozen_before_gpu={TARGET:3},epoch=time.time()))
    assert not torch.cuda.is_initialized()
    # One fixed read-only window. A busy result is retained and never changes cards.
    from gpu_telemetry import sample,inventory
    from gpu_admission import assess_window
    inventory_rows,_=inventory();target=[r for r in inventory_rows if r['uuid']==TARGET]
    assert len(target)==1 and int(target[0]['index'])==3,'AUTHORIZED_GPU3_UUID_MAPPING_CHANGED'
    publish(D/'audits/INITIAL_DEVICE_DISCOVERY.json',dict(status='EXPLICIT_GPU3_ONLY_NO_SELECTION',target=target,human_steering='你就只使用一张GPU3来跑就行了',other_GPU_use_authorized=False))
    rows=[]
    for i in range(3):
        rows.append(sample(TARGET))
        if i<2:time.sleep(5)
    publish(D/'audits/INITIAL_RESOURCE_SAMPLES.json',dict(samples={TARGET:rows}))
    assert all(len(s['target'])==1 and int(s['target'][0]['index'])==3 for s in rows)
    result=assess_window(rows,TARGET)
    publish(D/'resource_admission.json',dict(status='PASS_INITIAL_THREE_SAMPLE_WINDOW',results={TARGET:result},samples={TARGET:rows}))
    publish(D/'resource_baselines.json',{TARGET:result['graphics_baseline']})
    assert not torch.cuda.is_initialized()
    publish(D/'audits/PREPARATION_COMPLETE.json',dict(status='PASS_CPU_IDENTITIES_AND_FIXED_GPU3_READ_ONLY_INITIAL_WINDOW',execution_id=EXECUTION,CUDA_initialized=False,new_GPU_workers=0,parent_workers_planned=11,fresh64_workers_planned=12,all_workers_serial=True,elapsed_seconds=time.time()-begin,initial_window_min_free_bytes=result['min_free_bytes'],CPU_preparation_source_sha256=sha(__file__)))
    print('GPU3_SINGLE_CPU_PREPARATION_PASS; GPU_MODEL_WORKERS_NOT_STARTED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (D/'audits/CPU_PREPARATION_FAILED.json').exists():
            atomic_publish(D/'audits/CPU_PREPARATION_FAILED.json',dict(status='FAIL',reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_GPU_model_workers_launched=True,no_retry_or_GPU_migration=True))
        raise
