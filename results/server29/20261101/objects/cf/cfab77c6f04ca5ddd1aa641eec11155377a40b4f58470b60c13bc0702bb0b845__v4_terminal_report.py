"""Read science only after the planned roster ends; no unmeasured CSV shells."""
import json
import math
import time
from v4_persistent_common import *

def status(name):
    path=OUT/name
    return checked(path)['status'] if path.exists() else 'NOT_RUN'

def terminal_report(error=None):
    identities=list(RUNS.glob('*.PROCESS_IDENTITY.json'))
    active=[p.name.removesuffix('.PROCESS_IDENTITY.json') for p in identities if alive(p.name.removesuffix('.PROCESS_IDENTITY.json'))]
    if active:
        emit(OUT/f'ACTIVE_STOP_SNAPSHOT_{time.time_ns()}.json',dict(status='STOP_WITH_ACTIVE_WORKER_REQUIRING_MONITORING',active_workers=active,error=error,ledger=ledger()))
        return False
    account=ledger(); emit(OUT/'RESOURCE_USAGE_LEDGER.json',account)
    freeze=checked(FREEZE); selection=read(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json')
    spec=checked(OUT/'FROZEN_FORMAL_32K_SPEC.json') if (OUT/'FROZEN_FORMAL_32K_SPEC.json').exists() else None
    units=[]; references=[]; rows=[]; comparisons=[]
    for e in selection['documents']:
        doc=e['document']; t=tag('formal',doc); path=RUNS/(t+'.json'); receipt=RUNS/(t+'.LAUNCH_AUTHORIZATION.json')
        if not receipt.exists(): continue
        common=dict(document=doc,gpu=6,gpu_uuid=UUID,receipt_sha256=digest(receipt),code_commit=freeze['code_commit'],
            mapping_amendment_sha256=digest(OUT/'HARDWARE_MAPPING_AMENDMENT.json'),formal_spec_sha256=digest(OUT/'FROZEN_FORMAL_32K_SPEC.json'),
            original_pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
            asset_identity_sha256=digest(OUT/'V4_REUSED_SOURCE_IDENTITY.json'),engine_sha256=ENGINE_SHA)
        if path.exists() and path.with_suffix('.sha256.json').exists() and not (RUNS/(t+'.STOP.json')).exists():
            v=checked(path); assert v['status']=='PASS' and v['endpoint']==32768 and v['engine_sha256']==ENGINE_SHA
            by={r['condition']:r for r in v['stream']['results']}
            for c in v['conditions']:
                r=by[c]; trace=r['kl_trajectory']
                assert len(trace)==30720 and all(math.isfinite(x) for x in trace)
                assert r['primary']['window']==[8192,32768] and r['secondary']['window']==[2048,32768]
                assert sum(trace[6144:])/24576==r['primary']['mean'] and sum(trace)/30720==r['secondary']['mean']
                assert sum(trace[6144:])==r['primary']['auc'] and sum(trace)==r['secondary']['auc']
                units.append(dict(common,condition=c,status='COMPLETE',endpoint=32768,canonical_path=str(path),canonical_sha256=digest(path),
                    full_logit_sha256=r['full_logit_sha256'],token_sha256=v['token_file']['sha256'],rotation_sha256=v['rotation_artifacts'][c]['sha256'],
                    persistent_integrity=r['persistent_integrity'],gpu_memory=v['gpu_memory'],same_GPU_FP_sha256=v['stream']['fp_reference_full_logit_sha256']))
                rows.append(dict(document=doc,gpu=6,gpu_uuid=UUID,condition=c,primary_mean_KL=r['primary']['mean'],primary_AUC_KL=r['primary']['auc'],primary_token_count=24576,
                    secondary_mean_KL=r['secondary']['mean'],secondary_AUC_KL=r['secondary']['auc'],secondary_token_count=30720,
                    canonical_sha256=digest(path),FP_full_logit_sha256=v['stream']['fp_reference_full_logit_sha256'],token_sha256=v['token_file']['sha256'],
                    rotation_sha256=v['rotation_artifacts'][c]['sha256'],code_commit=freeze['code_commit'],engine_sha256=ENGINE_SHA,
                    mapping_amendment_sha256=common['mapping_amendment_sha256'],asset_identity_sha256=common['asset_identity_sha256'],science_unit_status='COMPLETE'))
            references.append(dict(common,status='COMPLETE',endpoint=32768,canonical_sha256=digest(path),
                token_sha256=v['token_file']['sha256'],full_logit_sha256=v['stream']['fp_reference_full_logit_sha256'],
                paired_conditions=v['conditions'],live_same_GPU_FP=True,formal_FP_tensor_chunks_archived=False))
            p1,l7=e['predicted_winners']['P1'],e['predicted_winners']['L7_TOTAL_ORIGINAL']
            pa,pb=by[p1]['primary']['mean'],by[l7]['primary']['mean']; delta=pa-pb; threshold=spec['uncertainty_rule']['threshold']
            verdict='TIE' if delta==0 else 'UNRESOLVED' if abs(delta)<=threshold else 'P1' if delta<0 else 'L7'
            comparisons.append(dict(document=doc,gpu=6,gpu_uuid=UUID,P1_predicted_rotation=p1,L7_predicted_rotation=l7,
                P1_primary_KL=pa,L7_primary_KL=pb,signed_Delta_primary=delta,
                P1_secondary_KL=by[p1]['secondary']['mean'],L7_secondary_KL=by[l7]['secondary']['mean'],
                signed_Delta_secondary=by[p1]['secondary']['mean']-by[l7]['secondary']['mean'],
                numerical_resolution_threshold=threshold,verdict=verdict,resolution_is_empirical_not_statistical=True,
                canonical_sha256=digest(path),code_commit=freeze['code_commit']))
        else:
            for c in (e['pair']['condition_a'],e['pair']['condition_b']): units.append(dict(common,condition=c,status='FAILED_OR_INCOMPLETE'))
    completed=sum(x['status']=='COMPLETE' for x in units); started=bool(units)
    scientific='INCOMPLETE_NOT_EVALUATED' if completed<4 else 'UNRESOLVED'
    if completed==4:
        winners=[x['verdict'] for x in comparisons]
        if winners==['P1','P1']: scientific='DIRECTIONALLY_SUPPORTIVE_SMALL_PILOT'
        elif winners==['L7','L7']: scientific='NOT_SUPPORTED_IN_THIS_PILOT'
        elif set(winners)=={'P1','L7'}: scientific='MIXED'
    if started:
        emit(OUT/'FP_REFERENCE_MANIFEST.json',dict(status='COMPLETE' if len(references)==2 else 'INCOMPLETE',references=references,completed_count=len(references),expected_count=2))
        emit(OUT/'FORMAL_32K_RUN_MANIFEST.json',dict(status='COMPLETE' if completed==4 else 'INCOMPLETE',units=units,completed_students=completed,expected_students=4))
        if rows: emit(OUT/'FORMAL_NUMERIC_TABLES.json',dict(status='COMPLETE' if completed==4 else 'INCOMPLETE',results=rows,comparisons=comparisons,
            original_reducer_verified=True,scientific_verdict=scientific))
    # Fill NOT_RUN artifacts before reading final statuses, unlike old V3 bug.
    for n in ('GPU_512X2_QUALIFICATION.json','MICRO_8448_RESULTS.json'):
        if not (OUT/n).exists(): emit(OUT/n,dict(status='NOT_RUN',reason=error or 'NOT_EXECUTED',no_PASS_claim=True))
    measured=[checked(p) for p in RUNS.glob('*.json') if not any(x in p.name for x in ('.sha256.','.LAUNCH_AUTHORIZATION.','.PROCESS_IDENTITY.','.STARTED.','.EXIT_OBSERVED.','.PREFIX_FINGERPRINTS.','.REDUCER_DIAGNOSTIC.','.STOP.'))]
    peaks=[x['gpu_memory'] for x in measured if 'gpu_memory' in x]
    memory=dict(peak_allocated_bytes=max((x['peak_allocated_bytes'] for x in peaks),default=None),
        peak_reserved_bytes=max((x['peak_reserved_bytes'] for x in peaks),default=None))
    statuses=dict(ENGINEERING_STATUS='STOP' if error else 'PASS',FP512_CROSS_DEVICE_STATUS=status('CROSS_DEVICE_FP_REPEATABILITY.json'),
        PORTABILITY_STATUS=status('P1_L7_PORTABILITY_GATE.json'),HARDWARE_MAPPING_STATUS=status('HARDWARE_MAPPING_AMENDMENT.json'),
        NUMERICAL_QUALIFICATION_STATUS=status('GPU_512X2_QUALIFICATION.json'),MICROBENCHMARK_STATUS=status('MICRO_8448_RESULTS.json'),
        PHYSICAL_RESOURCE_STATUS=status('PHYSICAL_RESOURCE_FEASIBILITY.json'),FORMAL_32K_STATUS='COMPLETE' if completed==4 else 'INCOMPLETE' if started else 'NOT_RUN',
        P1_VS_L7_SCIENTIFIC_VERDICT=scientific,completed_student_32K=completed,completed_same_GPU_FP=len(references),
        V4_gpu_hours=account['V4_receipt_to_canonical_GPUh'],process_exit_upper_GPUh=account['observed_exit_upper_GPUh'],
        actual_memory_peaks=memory,error=error,active_measurement_workers=[],epoch=time.time(),code_commit=freeze['code_commit'])
    emit(OUT/'FINAL_STATUS.json',statuses)
    emit(OUT/'GPU_EXCLUSIVITY_EVIDENCE.json',dict(raw_files={str(p.relative_to(OUT)):digest(p) for p in RUNS.glob('*.GPU_EXCLUSIVITY.jsonl')},
        physical_target_UUID_compute_PID_start_boot_ownership=True,shared_fuser_FD_diagnostic_only=True,no_signals=True))
    storage=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    report=f'''# Ling/KDA P1-vs-L7 V4 中文科研汇报

最终状态：{json.dumps(statuses,ensure_ascii=False)}

1. GPU6/doc5956 真实512-token FP双独立进程：{statuses['FP512_CROSS_DEVICE_STATUS']}，不以短前缀代替资格。
2. 合法设备修订：{statuses['HARDWARE_MAPPING_STATUS']}。仅V4 prospective serial GPU6；V1/V2/V3历史FAIL/STOP原样保留，不retroactive PASS。
3. 原P1/L7相反预测：{statuses['PORTABILITY_STATUS']}；不改doc/pair/方向，原window256/1024 H128/原数学，未重新执行selector。
4. 两篇完整512×2：{statuses['NUMERICAL_QUALIFICATION_STATUS']}；FP+两students、full-logit byte hash、KL exact fingerprint、18KDA6MLAconv/QDQ/inverse writeback/next-kernel/finite/storage均是真实门禁。
5. 8448 micro：{statuses['MICROBENCHMARK_STATUS']}；资源门禁{statuses['PHYSICAL_RESOURCE_STATUS']}。机械/性能盲态，无额外8448 observer重复；同模型调用形状/缓存下原byte payload复用做前缀核验。
6. 正式student完成{completed}/4，sameGPU fullFP完成{len(references)}/2。每token持续QDQ，无anchor关闭或FP state reset。formal仅在线full-logit SHA，未归档FP tensor chunks。
7. 两文档P1/L7 primary/secondary及signed Delta：{json.dumps(comparisons,ensure_ascii=False)}。没有完成的值是未测得，不能填0。总体判断{scientific}；仅两独立文档，不作统计确认/普遍排名/gradient训练/AIME主张。
8. GPU3问题是否绕开：V4资格和正式完成时可在本pilot操作上绕开；不证明GPU3硬件故障或其底层根因，未执行完整GPU3 forensic。
9. 未解决数值问题：{error or '未见本轮门禁内数值失败；经验数值resolution不是普适误差界。'}。V3历史GPU3/doc5956 exact-repeat FAIL仍有效。未做任何operator修复或quantizer改动。
10. 本轮记录{account['V4_receipt_to_canonical_GPUh']:.12f} GPUh，含startup/loading/FP/JIT/student，退出观察保守上界{account['observed_exit_upper_GPUh']:.12f}；Stage1仅canonical时间、非精确process exit。显存峰值{json.dumps(memory)}。报告生成前实际结果存储{storage} bytes，最终ZIP/代码archive额外大小另见交付清单。CPU/外部等待/历史实验不计本轮。

Primary原full-vocabulary FP32 KL(FP||student)[8192,32768)，secondary同轨迹[2048,32768)，Python sum reducer/原索引。数值判定规则在首条formal outcome前冻结，经验资格期同logit arithmetic resolution非统计显著性/普适bound；near-zero标UNRESOLVED。第一doc outcome不改变第二doc计划。

代码commit：{freeze['code_commit']}。原engine SHA256：{ENGINE_SHA}。结果目录：{OUT}。
'''
    with (OUT/('FINAL_SCIENTIFIC_REPORT.md' if started else 'FINAL_STOP_REPORT.md')).open('x',encoding='utf-8') as f: f.write(report)
    emit(OUT/'V4_TERMINAL_REPORT_COMPLETED.json',dict(status='COMPLETE',formal_students=completed,epoch=time.time(),delivery_archive_still_pending=True))
    print('V4_FINAL_STATUS',json.dumps(statuses),flush=True); return True

if __name__=='__main__': terminal_report(read(LATCH)['reason'] if LATCH.exists() else None)
