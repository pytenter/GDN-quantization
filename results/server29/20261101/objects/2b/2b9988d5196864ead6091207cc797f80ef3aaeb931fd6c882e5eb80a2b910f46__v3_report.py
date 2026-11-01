"""Truthful status/JSON tables; CSV export waits for real formal results."""
import math
import time
from v3_common import *

def artifact_status(name):
    p=OUT/name
    return checked(p)['status'] if p.exists() else 'NOT_RUN'

def terminal_report(error=None):
    account=ledger(); emit(OUT/'RESOURCE_USAGE_LEDGER.json',account)
    cfg=read(OLD/'PHASE_B_CALIBRATION_CONFIG.json'); sel=read(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json')
    freeze=checked(FREEZE); units=[]; fp=[]; rows=[]; comparisons=[]
    spec=checked(OUT/'FROZEN_FORMAL_32K_SPEC.json') if (OUT/'FROZEN_FORMAL_32K_SPEC.json').exists() else None
    for e in sel['documents']:
        t=tag('formal',e['document'],e['gpu'],0); p=RUNS/(t+'.json'); receipt=RUNS/(t+'.LAUNCH_AUTHORIZATION.json')
        if not receipt.exists(): continue
        common=dict(document=e['document'],gpu=e['gpu'],gpu_uuid=cfg['gpu_uuids'][str(e['gpu'])],
            receipt_path=str(receipt),receipt_sha256=digest(receipt),code_commit=freeze['code_commit'],
            original_pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
            source_model_weight_identity_sha256=digest(OUT/'SOURCE_AND_FROZEN_SELECTION_IDENTITY.json'),
            quantizer='original core.qdq_r128 + original direction.quantize_l6_cache; symmetric INT8-R128 native inverse writeback',
            qualification_sha256=digest(OUT/'GPU_512X2_QUALIFICATION.json'),formal_spec_sha256=digest(OUT/'FROZEN_FORMAL_32K_SPEC.json'))
        if p.exists() and p.with_suffix('.sha256.json').exists():
            v=checked(p); assert v['status']=='PASS' and v['endpoint']==32768
            assert v['engine_sha256']==ENGINE_SHA
            by={r['condition']:r for r in v['stream']['results']}
            for c in v['conditions']:
                r=by[c]; trace=r['kl_trajectory']
                assert len(trace)==32768-2048 and all(math.isfinite(x) for x in trace)
                assert r['primary']['window']==[8192,32768] and r['primary']['count']==24576
                assert r['secondary']['window']==[2048,32768] and r['secondary']['count']==30720
                assert sum(trace[6144:])/24576==r['primary']['mean'] and sum(trace)==r['secondary']['auc']
                unit=dict(common,condition=c,status='COMPLETE',endpoint=32768,canonical_path=str(p),canonical_sha256=digest(p),
                    full_logit_sha256=r['full_logit_sha256'],token_sha256=v['token_file']['sha256'],rotation_sha256=v['rotation_artifacts'][c]['sha256'],
                    same_GPU_FP_full_logit_sha256=v['stream']['fp_reference_full_logit_sha256'],actual_runtime=v['host'],
                    persistent_integrity=r['persistent_integrity'],gpu_memory=v['gpu_memory'],all_finite=True)
                units.append(unit)
                rows.append(dict(document=e['document'],gpu=e['gpu'],gpu_uuid=common['gpu_uuid'],condition=c,
                    primary_mean_KL=r['primary']['mean'],primary_AUC_KL=r['primary']['auc'],primary_token_count=24576,
                    secondary_mean_KL=r['secondary']['mean'],secondary_AUC_KL=r['secondary']['auc'],secondary_token_count=30720,
                    canonical_sha256=digest(p),FP_full_logit_sha256=unit['same_GPU_FP_full_logit_sha256'],
                    token_sha256=unit['token_sha256'],rotation_sha256=unit['rotation_sha256'],
                    code_commit=freeze['code_commit'],source_model_weight_identity_sha256=common['source_model_weight_identity_sha256'],
                    quantizer='original_INT8_R128_rotate_QDQ_inverse_writeback',qualification_sha256=common['qualification_sha256'],
                    science_unit_status='COMPLETE'))
            fp.append(dict(common,status='COMPLETE',endpoint=32768,full_logit_sha256=v['stream']['fp_reference_full_logit_sha256'],
                canonical_path=str(p),canonical_sha256=digest(p),paired_conditions=v['conditions'],token_sha256=v['token_file']['sha256'],
                live_matched_token_indexing=True,logits_archived=False))
            a=e['predicted_winners']['P1']; b=e['predicted_winners']['L7_TOTAL_ORIGINAL']
            pa,pb=by[a]['primary']['mean'],by[b]['primary']['mean']; delta=pa-pb; threshold=spec['uncertainty_rule']['threshold']
            verdict='TIE' if delta==0 else 'NUMERICALLY_UNRESOLVED' if abs(delta)<=threshold else 'P1' if delta<0 else 'L7'
            comparisons.append(dict(document=e['document'],gpu=e['gpu'],gpu_uuid=common['gpu_uuid'],P1_predicted_rotation=a,L7_predicted_rotation=b,
                P1_primary_KL=pa,L7_primary_KL=pb,signed_Delta_primary=delta,relative_paired_effect_vs_L7=delta/pb if pb!=0 else None,
                P1_secondary_KL=by[a]['secondary']['mean'],L7_secondary_KL=by[b]['secondary']['mean'],
                numerical_uncertainty_threshold=threshold,verdict=verdict,independent_unit='document',
                canonical_sha256=digest(p),code_commit=freeze['code_commit']))
        else:
            for c in (e['pair']['condition_a'],e['pair']['condition_b']): units.append(dict(common,condition=c,status='FAILED_OR_INCOMPLETE',endpoint=None))
    formal_started=bool(units); completed=sum(u['status']=='COMPLETE' for u in units)
    scientific='NOT_EVALUATED_NO_QUALIFIED_32K_RESULT'
    if formal_started:
        emit(OUT/'FP_REFERENCE_MANIFEST.json',dict(status='COMPLETE' if len(fp)==2 else 'INCOMPLETE',references=fp,completed_count=len(fp),expected_count=2))
        emit(OUT/'FORMAL_32K_RUN_MANIFEST.json',dict(status='COMPLETE' if completed==4 else 'INCOMPLETE',units=units,completed_students=completed,expected_students=4))
        winners=[x['verdict'] for x in comparisons]
        if completed<4: scientific='INCOMPLETE_NO_TWO_DOCUMENT_VERDICT'
        elif winners==['P1','P1']: scientific='DIRECTIONALLY_SUPPORTIVE_SMALL_PILOT'
        elif winners==['L7','L7']: scientific='P1_NOT_SUPPORTED_IN_THIS_PILOT'
        elif set(winners)=={'P1','L7'}: scientific='MIXED'
        else: scientific='TIE_OR_NUMERICALLY_UNRESOLVED'
        emit(OUT/'DOCUMENT_LEVEL_VERDICT.json',dict(status='COMPLETE' if completed==4 else 'INCOMPLETE',documents=comparisons,
            P1_VS_L7_SCIENTIFIC_VERDICT=scientific,independent_document_count=2,
            no_general_ranking_gradient_training_or_AIME_claim=True))
        # The requested CSV tables are populated only from these real records.
        # They are exported/validated locally via the spreadsheet skill, then
        # exclusive-uploaded. No empty CSV shell for unexecuted results.
        if rows: emit(OUT/'FORMAL_NUMERIC_TABLES.json',dict(status='COMPLETE' if completed==4 else 'INCOMPLETE',results=rows,comparisons=comparisons))
    statuses=dict(ENGINEERING_STATUS='STOP' if error else 'PASS',NUMERICAL_QUALIFICATION_STATUS=artifact_status('GPU_512X2_QUALIFICATION.json'),
        MICROBENCHMARK_STATUS=artifact_status('MICRO_8448_RESULTS.json'),FORMAL_32K_STATUS='COMPLETE' if completed==4 else 'INCOMPLETE' if formal_started else 'NOT_RUN',
        P1_VS_L7_SCIENTIFIC_VERDICT=scientific,completed_student_32k=completed,completed_same_GPU_FP=len(fp),
        error=error,code_commit=freeze['code_commit'],v3_gpu_hours=account['v3_gpu_hours'],epoch=time.time())
    emit(OUT/'V3_FINAL_STATUS.json',statuses)
    for n in ('GPU_512X2_QUALIFICATION.json','MICRO_8448_RESULTS.json','FROZEN_FORMAL_32K_SPEC.json'):
        if not (OUT/n).exists(): emit(OUT/n,dict(status='STOP' if n=='GPU_512X2_QUALIFICATION.json' else 'NOT_RUN',reason=error or 'NOT_EXECUTED',no_claim_of_PASS=True))
    raw=[str(p) for p in RUNS.glob('*.GPU_EXCLUSIVITY.jsonl')]
    emit(OUT/'GPU_EXCLUSIVITY_EVIDENCE.json',dict(status='COMPLETE',raw_files={p:digest(p) for p in raw},
        first_failure=read(LATCH) if LATCH.exists() else None,evidence_precedes_failure_assertion=True,
        target_context_only_current_verified_worker=True,shared_fd_diagnostic_not_target_compute=True))
    report=f'''# Ling/KDA P1-vs-L7 V3 中文汇报

最终状态：{json.dumps(statuses,ensure_ascii=False)}

1. V2 guard具体问题：把目标compute与fuser所有holder合并，仅允许自己PID；失败瞬间hw未保存。历史首因未知，不把当前空闲当作历史原因证明。
2. V3逐次核验物理UUID、NVML compute PID、worker /proc starttime/boot_id/UID，失败前保存raw NVML/fuser和每PID分类。
3. 修复仅涉及guard、namespace、launcher和必要telemetry；V2 worker/runtime及原engine字节不变。CPU结果见CPU_REGRESSION_RESULTS.json；实际CUDA guard probe不是512数值资格。
4. 资格始终GPU6→GPU3串行；后续是否并行只由一次轻量dual guard probe决定，详见GPU_GUARD_SCHEDULING.json（若已执行）。
5. 两卡512×2：{statuses['NUMERICAL_QUALIFICATION_STATUS']}。仅完整canonical、fresh-process exact fingerprint及全部机械门禁通过才标PASS。
6. 8448 micro：{statuses['MICROBENCHMARK_STATUS']}。真实速度/allocated/reserved/cache/block记录在MICRO_8448_RESULTS.json；未运行则无数值。
7. 真实32K物理资源：{artifact_status('PHYSICAL_RESOURCE_FEASIBILITY.json')}。无16GPUh/85%门槛，依据实际native/workspace growth和容量。
8. 正式student：{completed}/4；没有拼接或更换样本。
9. same-GPU完整FP：{len(fp)}/2。原engine每文档一条新FP流在线配对两student，full-logit字节hash识别，未声称FP tensor文件已归档。
10. 文档860 P1/L7 KL：见下方实际文档表；没有完成则未测得。
11. 文档5956 P1/L7 KL：见下方实际文档表；没有完成则未测得。
12. 各文档支持谁：{json.dumps(comparisons,ensure_ascii=False)}。
13. P1独立预测增量：{scientific}。仅两独立文档；不能宣称普遍排名优势、统计充分确认或梯度等价。
14. 本轮成本：{account['v3_gpu_hours']:.12f} GPUh，含probe/loading/JIT/FP和student，按两设备求和。显存实际peak见各canonical/micro；CPU、等待、其他任务不计。
15. 是否做梯度资格：本轮未执行任何训练。只有方向支持才值得单独提出P1_GRADIENT_TRAINABILITY_QUALIFICATION；任何结果均不能直接证明STE/Cayley可优化或改善AIME。

经验数值uncertainty只基于资格期同logit输入的CPU64/GPU32 arithmetic及reducer验证，正式前冻结，非普适误差证明或统计显著性。微小Delta不强判胜负。V1/V2 STOP保留；本轮无自动patch/retry。

远端结果：{OUT}

代码worktree：{CODE}，commit {freeze['code_commit']}；engine {ENGINE_SHA}。
'''
    name='FINAL_SCIENTIFIC_REPORT.md' if formal_started else 'FINAL_STOP_REPORT.md'
    with (OUT/name).open('x',encoding='utf-8') as f: f.write(report)
    refresh_hashes()
    print('V3_TERMINAL_REPORT',json.dumps(statuses),flush=True)

def refresh_hashes():
    # Hash snapshots are append-only. Final filenames are reserved for final
    # local CSV validation/delivery; an unfinished stage is not final delivery.
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and not p.name.startswith(('RESULT_AND_DELIVERY_HASHES','SHA256SUMS'))}
    emit(OUT/f'RESULT_AND_DELIVERY_HASHES_SNAPSHOT_{time.time_ns()}.json',dict(files=files,status='SNAPSHOT'))

if __name__=='__main__': terminal_report(read(LATCH)['reason'] if LATCH.exists() else None)
