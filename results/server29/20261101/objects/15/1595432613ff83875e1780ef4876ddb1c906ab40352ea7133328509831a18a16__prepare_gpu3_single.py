"""CPU-only identity registration for the newly authorized GPU6-only attempt.

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
TARGET='GPU-de53f243-b271-521c-d875-d96611ed8574'
TARGET_INDEX=6
PRIOR_SAME_TASK_GPUH=1.007654354178602
P=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2')
EXECUTION='LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1'
BASE='3c5837dcf61701661a935f3bbb3d2170a00f8d59'
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
    assert D.name==EXECUTION and W.name=='ling-kda-4doc-cleanroom-gpu6-single-v1'
    assert not (D/'RUN_CONTROLLER.json').exists() and not (D/'STOP.json').exists()
    assert not (D/'audits/PREPARATION_COMPLETE.json').exists(),'SINGLE_SHOT_PREPARATION_NO_REPEAT'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==BASE
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=O,text=True).strip()=='902d7974c89465df882403e96cc8dcc8dc1c0e85'
    old_stop='b831befc5b34d80143a5152ef85da1b84a8b904be0f14af72f13583e77d0afda'
    old_manifest='1a14787eb60b7385cee5b37f1662feff2fc45aad2cff343b4fb331a3424b1889'
    assert sha(C/'STOP.json')==old_stop and sha(C/'hashes/DELIVERY_VERIFICATION.json')==old_manifest
    previous_hashes={'STOP.json':'0acd47e7c2317207751e42f487be5dc2bdf251807bf3c497ecd667084b126b41','hashes/DELIVERY_VERIFICATION.json':'810366226d9b8480895ce2c07f0ba5b791a1b718ea677e17d53d12fd6ea55d52','gpu_resource_ledger.json':'8c49ac90178afcb659115d01b6ac98d4abc1dcefb94a48ff8b876ab74113d134','PREREGISTRATION_RECONCILIATION.json':'876c6fadde7aede053262a6b82044d354c71d545414e7d20d635d9bbbfb030ed','audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json':'c1b523e076b1dd1e2fca588576f11f0e6ee4cf3f1e1f79f5385df3a8f6f82955'}
    for name,digest in previous_hashes.items():assert sha(P/name)==digest,('PREVIOUS_SEALED_IDENTITY_CHANGED',name)
    previous_ledger=read(P/'gpu_resource_ledger.json')
    assert previous_ledger['actual_new_GPU_hours']==PRIOR_SAME_TASK_GPUH and not previous_ledger['active_workers']
    prior_audit=dict(status='PASS_ACTUAL_PREVIOUS_SEALED_LAUNCH_TO_EXIT',prior_same_task_GPUh=PRIOR_SAME_TASK_GPUH,previous_execution=str(P),previous_sealed_files={name:dict(path=str(P/name),sha256=digest) for name,digest in previous_hashes.items()},earlier_same_task_GPUh=0.0,cumulative_cap_GPUh=36.0,current_remaining_GPUh=36.0-PRIOR_SAME_TASK_GPUH,protective_cumulative_GPUh=35.8,independent_4090_not_inherited=True)
    publish(D/'audits/SAME_TASK_BUDGET_BASELINE.json',prior_audit)
    copied=[]
    for n in ['CALIBRATION_CONFIG.json','FROZEN_SPEC.json','FROZEN_4DOC_SELECTION.json','SECONDARY_SPEC.json','MODEL_WEIGHT_SHA256.txt','SELECTION_FREEZE_RECEIPT.json']:
        copied.append(copy_exact(C/n,D/n))
    for p in sorted((C/'preregistration').rglob('*')):
        if p.is_file():copied.append(copy_exact(p,D/'preregistration'/p.relative_to(C/'preregistration')))
    for n,h in [('FROZEN_SPEC.json','8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1'),('FROZEN_4DOC_SELECTION.json','a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b'),('preregistration/NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md','54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722')]:
        assert sha(D/n)==h,('FROZEN_SCIENCE_SHA_CONFLICT',n)
    publish(D/'audits/FROZEN_ASSET_COPY.json',dict(status='PASS_BYTE_IDENTICAL',files=copied,previous_attempt_read_only=True,no_historical_STOP_or_current_results_copied=True))
    auth='用户最新明确授权新独立3090完整实验，优先空闲高编号卡。运行前只读选择GPU6并冻结其UUID；本轮仅GPU6全局串行。原同任务3090累计硬预算36.0GPUh、累计保护35.8GPUh不变；先前V2实测1.007654354178602GPUh必须计入，剩余34.992345645821395GPUh。新GPU6与旧GPU3不同，12条件全部新建own-history parent及fresh64，不继承3090或4090资格/CP/PASS。所有旧STOP永久只读，不自动换卡、重试或增科学条件。'
    publish(D/'audits/GPU6_SINGLE_USER_AUTHORIZATION.json',dict(status='EXPLICIT_HUMAN_AUTHORIZATION',human_authorization_summary='新独立3090完整实验，优先空闲高编号卡',authorization_summary_is_verbatim_quote=False,execution_id=EXECUTION,scientific_protocol='LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1',previous_cleanroom_STOP=dict(path=str(C/'STOP.json'),sha256=old_stop,delivery_manifest_sha256=old_manifest),previous_GPU3_V2=dict(path=str(P),STOP_sha256=previous_hashes['STOP.json'],manifest_sha256=previous_hashes['hashes/DELIVERY_VERIFICATION.json'],prior_same_task_GPUh=PRIOR_SAME_TASK_GPUH),MAX_CONCURRENT_GPUS=1,MAX_TOTAL_NEW_GPUH=36.0,authorized_devices={TARGET:TARGET_INDEX},interpretation=auth,scope='same frozen four-document32K experiment with independent new execution identity; not permission to relax any scientific or physical gate'))
    validate_files()
    math_names=['functional_worker.py','formal_worker.py','persistent_engine_v1.py','v2_runtime.py','objective.py','run_matched_l7.py','frozen_runtime_profile.py','prospective_common.py']
    maths={}
    for n in math_names:
        assert (W/n).read_bytes()==(O/n).read_bytes(),('MATHEMATICAL_IDENTITY_CONFLICT',n)
        maths[n]=sha(W/n)
    from formal_checkpoint_control import adapted_source
    adapted=adapted_source();assert "primary = sum(x['KL'] for x in kl)/384" in adapted
    publish(D/'audits/MATHEMATICAL_IDENTITY.json',dict(status='PASS',unchanged_file_SHA256=maths,formal_control_adapter='unchanged CleanRoom reversible control hooks; source reversal restores original mathematical worker byte-for-byte',formal_adapter_source_sha256=sha(W/'formal_checkpoint_control.py'),adapted_formal_AST_SHA256=hashlib.sha256(ast.dump(ast.parse(adapted),include_attributes=False).encode()).hexdigest(),mathematical_kernel_changes=False))
    actual=read(P/'audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json')
    cached_source=read(P/'PREREGISTRATION_RECONCILIATION.json')
    source_by_path={row['path']:row for row in cached_source['source_assets']}
    assert actual['status']=='PASS_ACTUAL_BYTES' and len(actual['model_files'])==32 and len(actual['input_and_rotation_files'])==24
    checked={'model_files':[],'input_and_rotation_files':[]}
    for row in actual['model_files']:
        asset=source_by_path[row['path']];path=Path(row['path']);stat=path.stat()
        assert path.suffix=='.safetensors' and asset['sha256']==row['sha256'] and stat.st_size==row['bytes']==asset['bytes'] and stat.st_mtime_ns==asset['mtime_ns'],('CACHED_MODEL_ACTUAL_SHA_METADATA_CHANGED',str(path))
        checked['model_files'].append(dict(row,mtime_ns=stat.st_mtime_ns,verification='PREVIOUS_ACTUAL_FULL_SHA_REUSED_WITH_UNCHANGED_SIZE_MTIME',actual_SHA_authority_path=str(P/'audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json'),actual_SHA_authority_sha256=previous_hashes['audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json'],current_full_payload_rehashed=False))
    for row in actual['input_and_rotation_files']:
        path=Path(row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],('ACTUAL_SMALL_ASSET_SHA_FAILURE',str(path))
        checked['input_and_rotation_files'].append(dict(row,verification='CURRENT_ACTUAL_FULL_SHA'))
    spec=read(D/'FROZEN_SPEC.json');model=Path(spec['model']['path']);model_metadata=[]
    for filename,key in [('config.json','config_sha256'),('generation_config.json','generation_config_sha256'),('tokenizer_config.json','tokenizer_config_sha256'),('tokenizer.json','tokenizer_sha256'),('model.safetensors.index.json','weight_index_sha256')]:
        p=model/filename;assert sha(p)==spec['model'][key],('MODEL_METADATA_CHANGED',filename)
        model_metadata.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    publish(D/'audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json',dict(status='PASS_ACTUAL_BYTES',**checked,model_metadata=model_metadata,elapsed_CPU_seconds=time.time()-begin,new_GPU_worker_count=0,model_full_payloads_rehashed=False,model_SHA_scope='original full-byte SHA authority plus unchanged source_assets size/mtime; current inputs/rotations and model metadata actually rehashed',previous_model_audit_sha256=previous_hashes['audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json']))
    audit=read(P/'PREREGISTRATION_RECONCILIATION.json')
    old_history=read(C/'audits/HISTORICAL_STOP_AND_MANIFEST_IDENTITIES.json')
    for row in old_history['history']:
        assert sha(Path(row['execution'])/'STOP.json')==row['STOP_sha256']
        matches=[p for p in [Path(row['execution'])/'RESULT_AND_DELIVERY_HASHES.json',Path(row['execution'])/'delivery/RESULT_AND_DELIVERY_HASHES.json'] if p.exists() and sha(p)==row['manifest_sha256']]
        assert matches,('HISTORICAL_MANIFEST_CHANGED',row['execution'])
    old_history['history'].append(dict(execution=str(C),STOP_sha256=old_stop,manifest_sha256=old_manifest,manifest_path=str(C/'hashes/DELIVERY_VERIFICATION.json')))
    old_history['history'].append(dict(execution=str(P),STOP_sha256=previous_hashes['STOP.json'],manifest_sha256=previous_hashes['hashes/DELIVERY_VERIFICATION.json'],manifest_path=str(P/'hashes/DELIVERY_VERIFICATION.json'),actual_GPUh=PRIOR_SAME_TASK_GPUH))
    publish(D/'audits/HISTORICAL_STOP_AND_MANIFEST_IDENTITIES.json',old_history)
    for n,h in audit['frozen_protocol_files'].items():assert sha(Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')/n)==h,n
    for n,h in audit['scorer_source_SHA'].items():assert sha(n)==h,n
    for row in audit['source_assets']:
        p=Path(row['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']),('HISTORICAL_SOURCE_ASSET_METADATA_CHANGED',str(p))
        if p.suffix=='.safetensors':
            assert any(x['path']==str(p) and x['sha256']==row['sha256'] for x in checked['model_files']),('UNBOUND_CACHED_MODEL_SHA',str(p))
        else:assert sha(p)==row['sha256'],('HISTORICAL_SMALL_SOURCE_ASSET_CHANGED',str(p))
    for n,h in audit['historical_STOP_preserved'].items():assert sha(n)==h,n
    audit['historical_STOP_preserved'][str(C/'STOP.json')]=old_stop
    audit['historical_STOP_preserved'][str(P/'STOP.json')]=previous_hashes['STOP.json']
    audit.update(status='PASS',current_execution=str(D),authorization_reconciliation=auth,selected_GPU=TARGET_INDEX,selected_GPU_UUID=TARGET,selected_documents=[d['document'] for d in spec['documents']],CUDA_initialized=False,new_long_GPU_scientific_units=0,disk_free_bytes=shutil.disk_usage(D).free,elapsed_seconds=time.time()-begin,actual_gpu_inventory='Not inherited: current GPU6-only actual read-only samples are in audits/INITIAL_RESOURCE_SAMPLES.json',actual_compute_inventory='Not inherited: current GPU6-only actual process evidence is in audits/INITIAL_RESOURCE_SAMPLES.json')
    assert audit['disk_free_bytes']>=audit['required_storage_plan_bytes']
    publish(D/'PREREGISTRATION_RECONCILIATION.json',audit)
    selection=read(D/'FROZEN_4DOC_SELECTION.json')
    expected=[('pg19-test-41603','H_to_L7_a50','H'),('pg19-test-2544','H_to_L6_a50','H_to_L7_a50'),('pg19-test-7987','H_to_L7_a50','H_to_L7R_a25'),('pg19-test-26183','H_to_L6_a50','H_to_L7R_a25')]
    assert [(x['document'],x['P1_predicted_rotation'],x['L7_predicted_rotation']) for x in selection['documents']]==expected
    mapping={doc:TARGET for doc,_,_ in expected}
    publish(D/'EXECUTION_ROUTING.json',dict(execution=str(D),execution_id=EXECUTION,source_worktree=str(W),reference_experiment=str(D),authorized_devices={TARGET:TARGET_INDEX},document_devices=mapping,formal_global_order=selection['execution_order'],formal_serial=True,qualification_parallel_lanes=1,MAX_CONCURRENT_GPUS=1,MAX_TOTAL_NEW_GPUH=36.0,original_checkpoint_contract_sha256=sha(D/'FROZEN_SPEC.json'),initial_device_selection='explicit new user authorization fixes GPU6 only before any current model worker; no discovery/migration/fallback to any other card'))
    publish(D/'FP_PAIRING_CONTRACT_GATE.json',dict(status='PASS',reference_experiment=str(D),spec_sha256=sha(D/'FROZEN_SPEC.json'),mode='CLEANROOM_OWN_FP_SAME_DOC_UUID_FROZEN_RUNTIME_GPU6_ONLY'))
    # Different frozen UUID: no historical parent/CP or PASS can enter this run.
    assert TARGET!='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'
    assert not list((D/'runs').glob('*__parent.json')) and not list((D/'runs').glob('*__8192.pt'))
    publish(D/'reused_FP_bindings.json',dict(status='NO_SAME_UUID_PREFIX_REUSE_ALL_OWN_HISTORY_REQUIRED',units={},reason='new authorized GPU6 UUID differs from previous GPU3 and all4090 qualifications; twelve own-history parents and twelve current fresh64 workers required; no inherited PASS or checkpoint',historical_candidate_GPU3_UUID='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae',target_UUID=TARGET,other_historical_checkpoint_reuse=False))
    publish(D/'audits/REUSED_FP_CPU_IDENTITY.json',dict(status='NOT_REUSED_DIFFERENT_FROZEN_GPU_UUID',units={},current_parent_workers_required=12,current_fresh64_workers_required=12,no_historical_parent_CP_symlinks_created=True,no_qualification_labels_inherited=True,CUDA_initialized=False))
    hist=[]
    for path in Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1').rglob('*__formal.NATURAL_EXIT.json'):
        j=read(path)
        if j.get('natural_exit'):hist.append(dict(path=str(path),sha256=sha(path),seconds=j['wall_seconds']))
    publish(D/'preregistration.json',dict(TASK='LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1',execution_id=EXECUTION,scientific_protocol='LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1',preregistration_sha256='54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722',selection_sha256=sha(D/'FROZEN_4DOC_SELECTION.json'),FROZEN_SPEC_sha256=sha(D/'FROZEN_SPEC.json'),frozen_epoch=selection['frozen_epoch'],FULL_32K_EXECUTION_AUTHORIZED=True,MAX_CONCURRENT_GPUS=1,MAX_TOTAL_NEW_GPUH=36.0,initial_devices={TARGET:TARGET_INDEX},historical_formal_measurements=hist,historical_formal_worker_seconds=[r['seconds'] for r in hist],new_qualification_workers_planned=24,new_parent_workers=12,current_fresh64_workers=12,canary_model_workers=1,formal_workers=24,formal_scientific_trajectories=12,planned_long_checkpoint_cursor=16384,planned_overlap_tokens=64,global_formal_order=selection['execution_order'],all_GPU_workers_serial=True,no_new_scientific_keys=True,stats_unit='document',old_MIXED_not_combined_n6=True,new_human_authorization=auth,prior_same_task_GPUh=PRIOR_SAME_TASK_GPUH,remaining_GPUh=36.0-PRIOR_SAME_TASK_GPUH,protective_cumulative_GPUh=35.8,prior_same_task_resource_audit_sha256=sha(D/'audits/SAME_TASK_BUDGET_BASELINE.json'),previous_execution_resource=prior_audit,other_task_resources=read(P/'preregistration.json').get('other_task_resources',{}),no_4090_qualification_or_checkpoint_inheritance=True,budget_scope='same3090 full-science task cumulative36GPUh including priorV2 actual launch-to-exit1.007654354178602; independent4090 qualification reported separately'))
    publish(D/'engineering_change_manifest.json',dict(status='REGISTERED_BEFORE_GPU',base_commit=BASE,changes=['fixed authorized GPU6 UUID for all four documents and every worker','maximum one scheduler GPU worker for canary,qualification,and formal','CPU registration for new explicit authorization and preserved oldSTOP; math/FLA/quantizer/KL/strict comparison unchanged'],unchanged_math=maths,formal_adapter_exact_source_reversibility=True,extra_headroom_bytes=2*2**30,low_util_threshold_percent=2,resource_window_samples=3,resource_sample_interval_seconds=5,hard_cumulative_GPUh=36.,prior_same_task_GPUh=PRIOR_SAME_TASK_GPUH,current_remaining_GPUh=36.0-PRIOR_SAME_TASK_GPUH,protective_cumulative_GPUh=35.8,automatic_failed_science_retry=False,automatic_second_attempt=False,allowlisted_devices_frozen_before_gpu={TARGET:TARGET_INDEX},epoch=time.time()))
    assert 'torch' not in sys.modules,'CPU_PREPARATION_MUST_NOT_IMPORT_TORCH_OR_INITIALIZE_CUDA'
    # One fixed read-only window. A busy result is retained and never changes cards.
    from gpu_telemetry import sample,inventory
    from gpu_admission import assess_window
    inventory_rows,_=inventory();target=[r for r in inventory_rows if r['uuid']==TARGET]
    assert len(target)==1 and int(target[0]['index'])==TARGET_INDEX,'AUTHORIZED_GPU6_UUID_MAPPING_CHANGED'
    publish(D/'audits/INITIAL_DEVICE_DISCOVERY.json',dict(status='EXPLICIT_GPU6_ONLY_NO_SELECTION',target=target,human_steering='新独立3090完整实验，优先空闲高编号卡',other_GPU_use_authorized=False))
    rows=[]
    for i in range(3):
        rows.append(sample(TARGET))
        if i<2:time.sleep(5)
    publish(D/'audits/INITIAL_RESOURCE_SAMPLES.json',dict(samples={TARGET:rows}))
    assert all(len(s['target'])==1 and int(s['target'][0]['index'])==TARGET_INDEX for s in rows)
    result=assess_window(rows,TARGET)
    publish(D/'resource_admission.json',dict(status='PASS_INITIAL_THREE_SAMPLE_WINDOW',results={TARGET:result},samples={TARGET:rows}))
    publish(D/'resource_baselines.json',{TARGET:result['graphics_baseline']})
    assert 'torch' not in sys.modules,'CPU_PREPARATION_MUST_NOT_IMPORT_TORCH_OR_INITIALIZE_CUDA'
    publish(D/'audits/PREPARATION_COMPLETE.json',dict(status='PASS_CPU_IDENTITIES_AND_FIXED_GPU6_READ_ONLY_INITIAL_WINDOW',execution_id=EXECUTION,CUDA_initialized=False,new_GPU_workers=0,parent_workers_planned=12,fresh64_workers_planned=12,all_workers_serial=True,elapsed_seconds=time.time()-begin,initial_window_min_free_bytes=result['min_free_bytes'],CPU_preparation_source_sha256=sha(__file__)))
    print('GPU6_SINGLE_CPU_PREPARATION_PASS; GPU_MODEL_WORKERS_NOT_STARTED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (D/'audits/CPU_PREPARATION_FAILED.json').exists():
            atomic_publish(D/'audits/CPU_PREPARATION_FAILED.json',dict(status='FAIL',reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_GPU_model_workers_launched=True,no_retry_or_GPU_migration=True))
        raise
