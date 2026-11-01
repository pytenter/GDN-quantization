"""Render a report from real receipts; never authorize additional experiments."""
import collections
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis_scripts'))
from analyze_interaction import read, save, sha


def optional(rel):
    p=ROOT/rel
    return read(p) if p.is_file() else {}


def measured_mechanism(runtime):
    rows=[]
    for trajectory in runtime.get('trajectories',[]):
        samples=[]
        for point in trajectory['mechanism']:
            stats=point['stats']
            samples.extend(stats if isinstance(stats,list) else [stats])
        values=[x['relative_local_QDQ_mse'] for x in samples if 'relative_local_QDQ_mse' in x]
        ranges={k:dict(min=min(x[k] for x in samples if k in x),max=max(x[k] for x in samples if k in x)) for k in ('scale_min','scale_max','effective_scale_min','effective_scale_max','encoded_row_scale_min','encoded_row_scale_max','encoded_column_scale_min','encoded_column_scale_max','raw_row_scale_min','raw_row_scale_max','raw_first_column_scale_min','raw_first_column_scale_max','raw_second_column_scale_min','raw_second_column_scale_max','unrounded_over127_fraction','rounded_clamp_fraction') if any(k in x for x in samples)}
        rows.append(dict(document_index=trajectory['document_index'],condition=trajectory['condition'],MSE_sample_count=len(values),mean_measured_local_relative_MSE=statistics.mean(values) if values else None,
                         aggregation='C128 equalweight layer/score samples; Dual-M pooled24layer per-score samples; different histories/aggregation, no causal improvement ratio',
                         measured_scale_and_clipping_ranges=ranges,all_token_scale_risk_counts=trajectory.get('all_token_scale_risk_counts'),
                         post_warmup_tokens_per_second=trajectory['post_warmup_tokens_per_second'],wall_seconds=trajectory['wall_seconds'],peak_allocated=trajectory['peak_allocated'],peak_reserved=trajectory['peak_reserved']))
    return rows


def main():
    s=optional('SCIENCE_RESULTS.json')
    inter=optional('INTERACTION_RESULTS.json')
    runtime=optional('RUNTIME_COST.json')
    gpu=optional('FINAL_GPU_STATUS.json')
    ledger=optional('GPU_RESOURCE_LEDGER.json')
    hist=optional('HISTORICAL_ASSET_AUDIT.json')
    rotation=optional('C6R_ROTATION_IDENTITY.json')
    qualification=optional('GPU_QUALIFICATION.json')
    failure=optional('FIRST_FAILURE.json')
    released=gpu.get('GPU_release_verified',False)
    complete=bool(s.get('status')=='COMPLETE' and inter.get('status')=='COMPLETE_EXPLORATORY' and qualification.get('status')=='PASS' and gpu.get('COORDINATOR_STATUS')=='COMPLETE_AND_STOP' and not failure and released)
    docs=s.get('documents',[])
    interactions=inter.get('documents',[])
    means=s.get('summary',{'status':'NOT_RUN'})
    counts=s.get('direction_counts',{})
    icount=collections.Counter(d['Interaction_direction'] for d in interactions)
    learned=[d['LearnedGain_DualM'] for d in interactions]
    learned_wins=sum(d['LearnedGain_DualM']>d['LearnedGain_DualM_arithmetic_resolution'] for d in interactions)
    learned_losses=sum(d['LearnedGain_DualM'] < -d['LearnedGain_DualM_arithmetic_resolution'] for d in interactions)
    cases=[]
    if not complete:cases=['F_INCOMPLETE_OR_ENGINEERING_STOP']
    elif counts.get('D_WIN')==5:
        if learned_wins==5:cases.append('A_DUAL_M_IMPROVES_C6R_AND_ALL5_BETTER_THAN_H_DUAL_M')
        elif learned_losses==5:cases.append('B_DUAL_M_IMPROVES_C6R_BUT_ALL5_WORSE_THAN_H_DUAL_M')
        else:cases.append('DUAL_M_IMPROVES_C6R_BUT_ABSOLUTE_LEARNED_ADVANTAGE_MIXED_OR_UNRESOLVED')
        if icount.get('NEGATIVE')==5:cases.append('C_NEGATIVE_INTERACTION_ALL5')
        elif icount.get('POSITIVE')==5:cases.append('POSITIVE_INTERACTION_ALL5_EXPLORATORY')
        else:cases.append('INTERACTION_MIXED_OR_UNRESOLVED')
    elif counts.get('C_WIN')==5:cases=['D_NO_C6R_INCREMENTAL_GAIN_ALL5_OPPOSITE']
    else:cases=['E_MIXED_OR_NUMERICALLY_UNRESOLVED']
    mechanism=measured_mechanism(runtime)
    window_summary={}
    for name in ('Early','Middle','Late'):
        observed=[d['windows'][name] for d in docs]
        if observed:window_summary[name]=dict(documents=len(observed),mean_Delta=statistics.mean(x['Delta'] for x in observed),D_WIN=sum(x['Winner']=='H_DUAL_M' for x in observed),C_WIN=sum(x['Winner']=='H_C128' for x in observed),UNRESOLVED=sum(x['Winner']=='NUMERICALLY_UNRESOLVED' for x in observed))
    late_largest=sum(max(d['windows'],key=lambda w:d['windows'][w]['Delta'])=='Late' for d in docs)
    cs={c:sum((ROOT/f'analysis/RECEIPT_science_doc{d}_{c}.json').is_file() and optional(f'analysis/RECEIPT_science_doc{d}_{c}.json').get('status')=='PASS' for d in range(1,6)) for c in ('C6R_C128','C6R_DUAL_M')}
    status=dict(TASK_ID=ROOT.name,GPU_AUTHORIZATION='YES_HUMAN_FOUR_GPU_OVERRIDE_TOTAL20',
                HISTORICAL_REUSE_GATE=hist.get('status','NOT_RUN'),C6R_NUMERICAL_QUALIFICATION=qualification.get('status','NOT_RUN'),
                PERSISTENT_QDQ_QUALIFICATION='QUALIFIED' if qualification.get('status')=='PASS' else 'NOT_RUN_OR_NOT_QUALIFIED',
                FP_NATIVE_REUSED='5/5_VERIFIED_FROZEN_ASSETS' if hist.get('status')=='PASS' else 'NOT_QUALIFIED',
                H_C128_REUSED='5/5_CPU_REPRODUCED' if hist.get('status')=='PASS' else 'NOT_QUALIFIED',
                H_DUAL_M_REUSED='5/5_CPU_REPRODUCED' if hist.get('status')=='PASS' else 'NOT_QUALIFIED',
                C6R_C128_32K_COMPLETED=f"{cs['C6R_C128']}/5",C6R_DUAL_M_32K_COMPLETED=f"{cs['C6R_DUAL_M']}/5",
                VALID_PRIMARY_PAIRED_DOCS=f'{len(docs)}/5',PRIMARY_RESULT=s.get('PRIMARY_RESULT','NOT_RUN'),
                INTERACTION_RESULT=dict(status=inter.get('status','NOT_RUN'),directions=dict(icount)) if interactions else 'NOT_RUN',
                CROSS_GPU_COMPARABILITY=inter.get('device_strata','NOT_RUN'),
                GPU_HOURS_USED=ledger.get('GPU_HOURS_USED',0.0),MAX_GPU_HOURS=20.0,PROTECTIVE_STOP_GPUH=19.5,
                GPU_release_verified=released,PHYSICAL_PACKED_INT8='NOT_QUALIFIED',PROJECT_INDEPENDENCE='UNKNOWN',
                DESIGN='POSTHOC_ON_HISTORICALLY_OBSERVED_DOCUMENTS',
                FINAL_SCIENTIFIC_CLASSIFICATION=s.get('PRIMARY_RESULT','NOT_RUN') if complete else 'INCOMPLETE_OR_ENGINEERING_STOP',
                SCIENTIFIC_CASES=cases,
                FINAL_EXECUTION_STATUS='COMPLETE_AND_STOP' if complete else 'FAIL_CLOSED_STOP',
                ABSOLUTE_REPORT_PATH=str(ROOT/'FINAL_SCIENTIFIC_REPORT.md'),no_further_GPU_work=True,no_automatic_next_experiment=True,
                old_FP_NATIVE_FP_H_failure='PRESERVED_token1_relL2_0.006718461729707917_gt0.002',
                old_BrokenPipe_failure='PRESERVED_NOT_OVERWRITTEN',first_failure=failure or None)
    save('FINAL_STATUS.json',status)
    q={}
    q[1]=f"本次Primary={status['PRIMARY_RESULT']}。唯一Primary为Delta_C6R=K_C6R_C128−K_C6R_DUAL_M。" if docs else 'NOT_RUN：未取得完整合法新C6R配对评分，无法判断Dual-M对C6R的32K收益。'
    q[2]=f"完整合法配对{len(docs)}/5；可分辨方向{json.dumps(counts,ensure_ascii=False)}，缺失{5-len(docs)}。正式N固定5，不以评分位置为样本。"
    q[3]=json.dumps(means,ensure_ascii=False)+'；mean per-document RelativeGain与ratio of mean KL分别报告，不混用。'
    q[4]=f"C6R+Dual-M相较H+Dual-M可分辨优于{learned_wins}/5、劣于{learned_losses}/5；其余未分辨或缺失。完整逐doc LearnedGain_DualM见表。" if interactions else 'NOT_RUN：尚无合法绝对性能对比；不能把Delta_C6R>0当作C6R+Dual-M胜过H+Dual-M。'
    q[5]=f"Interaction方向计数{dict(icount)}。逐doc值与算术范围见表；Delta_C6R正值与正交互是不同命题。" if interactions else 'NOT_RUN：Interaction=Delta_C6R−Delta_H，无新合法数据不能确定方向。'
    q[6]='原五篇有明确后续四卡人类授权；本次也收到新的四卡授权，总20GPUh未提高。冻结映射Doc1/5→GPU0、Doc2→GPU1、Doc3→GPU2、Doc4→GPU3。只有真实同物理UUID且数学/软件/runtime资格匹配才归A；不同设备归B描述；身份不完整归C排除。128-token跨卡exact不能界定全32K数值误差，即使全A仍是同设备探索而非独立确认性因果2×2。'
    q[7]=json.dumps(s.get('sensitivity',{'status':'NOT_RUN'}),ensure_ascii=False)+'；去Doc1与LOO均为事前要求的描述性敏感性，不从Primary剔除Doc1。'
    q[8]=json.dumps([dict(document_index=m['document_index'],condition=m['condition'],mean_measured_local_relative_MSE=m['mean_measured_local_relative_MSE'],sample_count=m['MSE_sample_count']) for m in mechanism],ensure_ascii=False)+'。C128为逐层/评分点等权局部relative-MSE，Dual-M为每评分点24层汇总relative-MSE，聚合口径和历史不同，不能给同输入codec因果改善比。长期KL方向见Primary，未识别局部误差是否为唯一中介。' if mechanism else 'NOT_RUN：没有真实新轨迹机制统计，不伪造MSE/scale或中介结论。'
    q[9]=json.dumps(window_summary,ensure_ascii=False)+f'。Early/Middle/Late各128点；Late绝对Delta在{late_largest}/{len(docs)}篇最大，不假设晚窗统一最大。完整逐doc方向见WINDOW_RESULTS.csv。' if window_summary else 'NOT_RUN：未形成完整384位置窗口结果。'
    q[10]=f"实际新增GPU占用{status['GPU_HOURS_USED']}GPUh；上限20，protective19.5。占用涵盖加载、forward、QDQ、checkpoint锁等待及释放前IO。实际吞吐/整轨迹wall、峰值与scale/clipping统计如下："+json.dumps(mechanism,ensure_ascii=False)+'。未触发严重风险不等于没有floor事件；relative-MSE和range仅对应注明的采样/调用口径。physical packed INT8仍NOT_QUALIFIED。'
    q[11]='本次历史H结果已观察，N=5且项目独立性UNKNOWN。即使Primary和Interaction方向一致，也仅可提供独立新doc、同GPU完整2×2研究的探索动机，不能宣称独立确认性显著性；不在本任务执行。'
    q[12]=('当前5/5可分辨增量收益可支持讨论未来codec-aware Loss，但是否有learned绝对优势及正向交互需分别看'+str(cases)+'；应先有独立新doc/同设备2×2证据再把候选目标作为方法开发依据。' if complete and counts.get('D_WIN')==5 else '当前结果不足以直接支持开发或训练新codec-aware rotation Loss，应保留完整混合/未分辨/失败证据。')+' 候选FunctionalLoss(R,DualM)+lambda*StateLoss(R,DualM)尚未优化或验证；本任务不训练C8、不跑后续研究。'
    q[13]='尚未验证：广泛泛化、项目独立性、全32K forward不确定性上界、FP_NATIVE/FP_H等价、packed INT8部署速度、AIME增益、唯一scale/clipping/query-readout因果中介及新训练目标。旧2doc只作背景，不并入本轮N。'
    lines=['# Qwen/GDN C6R × Dual-M 32K 交互探索报告','',f"最终执行状态：**{status['FINAL_EXECUTION_STATUS']}**；科学分类：**{status['FINAL_SCIENTIFIC_CLASSIFICATION']}**。",'',
           '本报告只来自真实历史复核、新轨迹与实际回执。未完成的科学项标NOT_RUN，未构造空壳科学CSV。', '', '## 执行与身份','',
           '```json',json.dumps(status,ensure_ascii=False,indent=2),'```','',
           'C6R冻结矩阵以原manifest和逐层SHA直接加载，H128后Delta保持原两个矩阵乘法；未训练、插值、筛选矩阵，未改CanonicalC128、Dual-M核心、GDN递归算子或Primary reducer。原源码/差异与资格见SOURCE_MANIFEST及C6R审计。','',
           '本次四卡是用户对原附件单卡要求的新明确覆盖，详情见HUMAN_4GPU_STORAGE_OVERRIDE.txt和预注册；原历史协议审计中对初始单卡附件的描述保留其时点，不代表最终新授权。','', '## 实际逐文档科学结果','']
    if docs:
        lines += ['|Doc|K_C6R_C128|K_C6R_DualM|Delta_C6R|每doc相对改善|算术分辨率|方向|','|---|---:|---:|---:|---:|---:|---|']
        for d in docs:
            gain='NOT_RESOLVED' if d['RelativeGain'] is None else f"{100*d['RelativeGain']:.4f}%"
            lines.append(f"|{d['document_index']}|{d['K_A']:.12g}|{d['K_D']:.12g}|{d['Delta']:.12g}|{gain}|{d['Delta_resolution']:.9g}|{d['Winner']}|")
    else:lines += ['NOT_RUN：无完整合法配对；不填0或科学方向。']
    lines += ['', '## 交互及设备可比性','']
    if interactions:
        lines += ['|Doc|设备层|Delta_H|Delta_C6R|Interaction|LearnedGain_C128|LearnedGain_DualM|','|---|---|---:|---:|---:|---:|---:|']
        for d in interactions:lines.append('|'+ '|'.join(str(d[k]) for k in ('document_index','stratum','Delta_H','Delta_C6R','Interaction','LearnedGain_C128','LearnedGain_DualM'))+'|')
    else:lines += ['NOT_RUN：不能从工程停止解释方法无效。']
    lines += ['', '算术分辨率来自相同保存logits的FP32评分与独立标量binary64审计，依赖原IEEE64/libm条件；它不界定模型前向32K误差。历史已观察H基线与新C6R的证据层级不能升级为新样本确认性2×2。','', '## 十三项科学问题','']
    titles=['Dual-M是否改善C6R的32K Future-KL？','五篇支持/反对/未分辨/缺失多少？','平均绝对与相对改善是多少？','C6R+Dual-M是否优于H+Dual-M？','Interaction方向是否稳定？','跨GPU历史比较有何影响？','去Doc1后的描述性结论如何？','局部MSE是否对应长期KL？','三窗口有什么差异？','系统开销与scale/clipping风险怎样？','是否足以支持独立2×2确认？','是否需要开发codec-aware rotation Loss？','哪些问题尚未验证？']
    for i,title in enumerate(titles,1):lines += [f'### {i}. {title}','',q[i],'']
    lines += ['## 停止与交付','', '全部实际任务结束后停止，不追加任何GPU研究、训练、AIME或独立确认。大tensor永久位置及SHA/bytes见archive映射与最终SHA256_MANIFEST；只有独立校验通过后才移除新任务临时副本。历史封存不覆盖。','']
    with (ROOT/'FINAL_SCIENTIFIC_REPORT.md').open('x') as f:f.write('\n'.join(lines))
    save('GATE_STATUS_FINAL.json',dict(status=status['FINAL_EXECUTION_STATUS'],historical_reuse=hist.get('status','NOT_RUN'),C6R_identity=rotation.get('status','NOT_RUN'),GPU_qualification=qualification.get('status','NOT_RUN'),GPU_release_verified=released,all_actual_current_assets_to_be_verified_at_seal=True,no_further_GPU_work=True,all_scientific_tasks_stop=True))


if __name__=='__main__':main()
