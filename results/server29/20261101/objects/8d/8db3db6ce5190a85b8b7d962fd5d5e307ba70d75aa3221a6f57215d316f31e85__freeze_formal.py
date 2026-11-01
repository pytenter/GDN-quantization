"""CPU-only scientific/source admission before the one authorized launch."""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import execution_common as C
from formal_controller import validate_jobs

def read(name):return json.loads((ROOT/name).read_text())
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='' and __debug__
    for n in ['FORMAL_SOURCE_MANIFEST.json','GPU_PERMIT.json','FORMAL_STARTED.json','operations/UNIQUE_LAUNCH_CLAIM.json']:assert not (ROOT/n).exists(),'One source freeze/launch only'
    jobs=read('FORMAL_JOBS.json')['jobs'];validate_jobs(jobs)
    a,b=read('configs/DP8.json'),read('configs/DP16.json');assert {k for k in a if a[k]!=b[k]}=={'decode_period'}
    assert a['decode_period']==8 and b['decode_period']==16
    for rel in ['src/pace_quantizer.py','src/pace_buffer.py','src/pace_runtime.py','calibration/head_mask.json','vendor/canonical_h128.py','vendor/modeling_qwen3_5.py','vendor/cache_utils.py','vendor/protocol_binding.py','protocol/sampler.py','protocol/common.py','protocol/aime26_scorer_v4.py','protocol/aime26_frozen.jsonl']:
        assert C.sha(ROOT/rel)==C.sha(ROOT/'baseline'/rel),'Inherited frozen bytes changed: '+rel
    assert C.sha(ROOT/'src/pace_quantizer.py')=='89f4801aac1498eb68d285fb40ea3eb922b52645782853367e25558042798807'
    assert C.sha(ROOT/'calibration/head_mask.json')=='09483fb4debca26e9561778adea987d80dfcadb1ef0d4dd1ec311e6915bf7ca6'
    evidence=['validation/CUDA_FRESH_V4.json','validation/FAST_BATCH_KEYS_V5.json','validation/SAME_STATE_ORACLE_V1.json','validation/LINEAGE_COUNTERFACTUAL_V1.json','validation/FULL_MODEL_PARITY_V1.json','performance/FAST_MATCHED_THROUGHPUT_V2.json','validation/CPU_CONTROL_FRESH4.json','validation/CPU_CONTROL_FRESH5.json','validation/CPU_DELIVERY_PIPELINE2.json','validation/CPU_ARCHIVE_CANARY_RECEIPT.json']
    for n in ['CPU_CONTROL_FRESH4','CPU_CONTROL_FRESH5','CPU_DELIVERY_PIPELINE2']:assert read('validation/'+n+'.json')['status'].startswith('PASS')
    assert read('validation/CPU_ARCHIVE_CANARY_RECEIPT.json')['status'].startswith('PASS_REAL29')
    prep=read('configs/NUMERICAL_ADMISSION_PREREGISTRATION.json')['floating_gate']
    for n in ['CUDA_FRESH_V4','FAST_BATCH_KEYS_V5']:
        x=read('validation/'+n+'.json');assert len(x['cases'])==40 and x['actual_CUDA'] and x['new_AIME_samples']==0
        for key in ['output','delta','boundary_state']:assert x['worst'][key]['normalized_RMSE']<=prep['cross_replay_max_normalized_RMSE']
        assert x['worst']['actual_delta_state_readout']['normalized_RMSE']<=prep['self_actual_delta_state_readout_max_normalized_RMSE']
        assert all(z[key]['nonfinite_count']==0 for c in x['cases'] for z in c['rows'] for key in ['output','delta','boundary_state','actual_delta_state_readout'] if key in z)
    lineage=read('validation/LINEAGE_COUNTERFACTUAL_V1.json');assert lineage['status'].startswith('PASS')
    assert all(z['same_base_local_rounding']['unexplained']==0 for z in lineage['rows'] if 'same_base_local_rounding' in z)
    downstream=read('validation/FULL_MODEL_PARITY_V1.json');assert downstream['unchanged_prefill_bitwise'] and downstream['actual_markers_equal'] and downstream['new_AIME_samples']==0
    for gpu in range(4):
        n='validation/FAST_MODEL_SMOKE_GPU'+str(gpu)+'_V2.json';x=read(n);assert x['status']=='PASS_NON_AIME_FULL_MODEL_SMOKE' and x['runtime']['driver_uuid']==C.UUIDS[gpu]
        receipt='runtime/FAST_MODEL_SMOKE_GPU'+str(gpu)+'_V2/OWNING_WAIT_AND_RELEASE.json';z=read(receipt);assert z['owning_Popen_exit']==0 and z['release_observed']
        evidence.extend([n,receipt])
        snapshot=read('runtime/FAST_MODEL_SMOKE_GPU'+str(gpu)+'_V2/SOURCE_SNAPSHOT.json')
        by={z['path']:z for z in snapshot['files']}
        for rel in ['src/lq_unroll_buffer.py','src/packed_direct_read.py','src/unroll_runtime.py']:assert by[rel]['sha256']==C.sha(ROOT/rel),'Production mathematical source differs from four-card actual smoke: '+rel
    for name in ['FAST_MATCHED_THROUGHPUT_V2','FULL_MODEL_PARITY_V1','LINEAGE_COUNTERFACTUAL_V1','FAST_BATCH_KEYS_V5']:
        rel='runtime/'+name+'/OWNING_WAIT_AND_RELEASE.json';z=read(rel);assert z['owning_Popen_exit']==0 and z['release_observed'];evidence.append(rel)
    cpu=read('validation/CPU_FULL_V3.json');assert cpu['worst']['output']['normalized_RMSE']>prep['cross_replay_max_normalized_RMSE']
    admission=dict(status='ADMITTED_TARGET_CUDA_INDEPENDENT_NUMERICAL_IMPLEMENTATION',formal_admission=True,new_formal_target=60,reuse=0,
        inherited_quantizer_mask_protocol_exact=True,only_configuration_difference='decode_period8/16',
        target_CUDA_operator_original_bounds_pass=True,tolerances_unchanged=prep,
        CPU_prototype_cross_trajectory_screen='FAIL_RETAINED_NOT_PRODUCTION_BACKEND',CPU_prototype_not_relabelled_PASS=True,
        packed_difference_scope='Original687 UNEXPLAINED local labels retained. Actual same-window-base intervention removes1077 cross-trajectory differences;2 remaining local differences satisfy original adjacent-rounding test. Local unexplained0. No change to codec or tolerance.',
        downstream_logit_worst_NRMSE=downstream['max_logits_normalized_RMSE'],downstream_layer_output_worst_NRMSE=downstream['max_layer_output_normalized_RMSE'],
        downstream_equivalence_claim=False,long80K_stability_and_accuracy='UNRESOLVED_BEFORE_ACTUAL_FORMAL',
        admission_scope_clarification='Local same-input/target-backend operator checks are distinct from CPU prototype and multilayer different-input numerical trajectories. User explicitly authorizes an independent FP32 numerical implementation after documented differences; no old score reuse.',
        CPU_control_and_finisher_real_waits_pass=True,real29_NOREPLACE_canary_pass=True,
        evidence=[dict(path=n,sha256=C.sha(ROOT/n),bytes=(ROOT/n).stat().st_size) for n in evidence],client_independent_will_recheck_terminal=True)
    C.atomic(ROOT/'validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json',admission)
    selected=[]
    for folder in ['src','scripts','configs','vendor','protocol','calibration','baseline','analysis']:
        selected += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    selected += [ROOT/n for n in ['FORMAL_JOBS.json','AIME26_DATASET_FREEZE.json','operations/HUMAN_REQUEST.txt','operations/MINIMAL_PINNED_RUNTIME_VENDOR.json','operations/BASELINE_SOURCE_VERIFICATION.json','operations/PRE_FORMAL_CPU_SCOPE_NOTES.md','operations/launch_formal.sh','validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json']+evidence]
    selected=sorted(set(selected))
    for p in selected:assert p.is_file()
    source=dict(status='FROZEN_BEFORE_UNIQUE_FORMAL_START',files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=C.sha(p)) for p in selected],scientific_new_configs=2,formal_new_target=60,reuse=0)
    C.atomic(ROOT/'FORMAL_SOURCE_MANIFEST.json',source);digest=C.sha(ROOT/'FORMAL_SOURCE_MANIFEST.json')
    C.atomic(ROOT/'GPU_PERMIT.json',dict(status='AUTHORIZED_BY_SAVED_FULL_HUMAN_REQUEST',task='PACE48_LEAP_UNROLL_DP8_DP16_V1',formal_target=60,reuse=0,source_sha256=digest,admission_sha256=C.sha(ROOT/'validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json'),human_request_sha256=C.sha(ROOT/'operations/HUMAN_REQUEST.txt'),physical55_UUIDs=C.UUIDS,TP=1,archive29_CPU_only=True,no_preemption_no_foreign_signals=True,formal_backend='TRITON_PACKED_DIRECT_ONLY',replay_reference='ENGINEERING_FIXED_WORKLOAD_ONLY',no_automatic_GPU_retry=True,no_GPUh_or_totalwall_cap=True,phase_no_progress_seconds=1800,own_exit_release_seconds=120))
    C.source(ROOT,digest)
    print(json.dumps(dict(status='PASS_SOURCE_FREEZE_AND_TARGET_CUDA_ADMISSION',source_sha256=digest,permit_sha256=C.sha(ROOT/'GPU_PERMIT.json'),files=len(source['files']),bytes=sum(x['bytes'] for x in source['files']))),flush=True)
if __name__=='__main__':run()
