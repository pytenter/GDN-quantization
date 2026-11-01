"""Document-unit analysis after the whole fixed panel, or truthful fail-closed seal."""
import csv, hashlib, json, math, os, statistics, subprocess, time
from pathlib import Path
from archive_assets import ROOT, sha, save, load, REMOTE_BASE
ARMS=['PARENT','NO_SCALE_INFLATION']

def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csvout(name,data):
    if not data:return
    with (ROOT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def combine(name,files):
    data=[]
    for p in files:data+=rows(p)
    csvout(name,data)
def small_and_local_files():
    paths=list(ROOT.rglob('*'))
    for top in ['anchors','students']:
        if (ROOT/top).exists():paths+=list((ROOT/top).rglob('*'))
    return sorted({p for p in paths if p.is_file() and not any(x in p.parts for x in ['.cache','__pycache__','logs']) and p.name not in ['ACTIVE_COORDINATOR_LOCK','COORDINATOR_EVENTS.jsonl','RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']})

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (ROOT/'RESULT_AND_DELIVERY_HASHES.json').exists()
    failed=load(ROOT/'FIRST_FAILURE_GATE.json') if (ROOT/'FIRST_FAILURE_GATE.json').exists() else None
    spec=load(ROOT/'FROZEN_SCIENCE_SPEC.json') if (ROOT/'FROZEN_SCIENCE_SPEC.json').exists() else None
    refs={};anchors={};branches={};archives={}
    for d in range(1,9):
        for target,filename in [(refs,f'references/doc_{d:02d}/FP_REFERENCE_MANIFEST.json'),(anchors,f'anchors/doc_{d:02d}/ANCHOR_MANIFEST.json'),(archives,f'archive/DOC_{d:02d}_ARCHIVE_VERIFICATION.json')]:
            p=ROOT/filename
            if p.exists():target[d]=load(p)
        for a in ARMS:
            p=ROOT/f'students/doc_{d:02d}/{a}/BRANCH_MANIFEST.json'
            if p.exists():branches[(d,a)]=load(p)
    complete=len(refs)==len(anchors)==len(archives)==8 and len(branches)==16 and not failed
    if complete:
        assert all(x['status']=='COMPLETE' for x in list(refs.values())+list(anchors.values())+list(branches.values()))
        assert all(x['status']=='PASS_ALL_FILE_SHA_BYTES' for x in archives.values())
        assert all(load(ROOT/f'archive/DOC_{d:02d}_NUMERICAL_AUDIT.json')['status']=='PASS' for d in range(1,9))
    if refs:save('FP_REFERENCE_MANIFEST.json',{'status':'COMPLETE' if len(refs)==8 else 'INCOMPLETE','documents':refs})
    if anchors:save('ANCHOR_MANIFEST.json',{'status':'COMPLETE' if len(anchors)==8 else 'INCOMPLETE','documents':anchors})
    if branches:save('PARENT_NOINFLATION_RUN_MANIFEST.json',{'status':'COMPLETE' if len(branches)==16 else 'INCOMPLETE','branches':[dict(document_index=d,**b) for (d,a),b in sorted(branches.items())]})
    for filename,source in [('PER_POSITION_FULLVOCAB_KL.csv','SAMPLED_FULLVOCAB_FUTURE_KL.csv'),('INTERVENTION_ACTIVATION.csv','SCALE_INTERVENTION_ACTIVATION.csv'),('CLIPPING_SATURATION.csv','SCALE_INTERVENTION_ACTIVATION.csv'),('STATE_DAMAGE_ENDPOINTS.csv','STATE_DAMAGE_ENDPOINTS.csv')]:
        combine(filename,[ROOT/f'students/doc_{d:02d}/{a}/{source}' for d,a in sorted(branches)])
    effects=[];windows=[]
    for d in range(1,9):
        if not all((d,a) in branches for a in ARMS):continue
        parent,cap=[branches[(d,a)] for a in ARMS];delta=parent['K']-cap['K'];fraction=cap['activation_groups']/cap['total_groups']
        direction='NOT_CLASSIFIED_INCOMPLETE'
        if complete:direction='POSITIVE' if delta>spec['DELTA_resolution'] else 'NEGATIVE' if delta<-spec['DELTA_resolution'] else 'UNRESOLVED'
        effect=dict(parent['identity'],document_index=d,K_PARENT=parent['K'],K_NOINFLATION=cap['K'],Delta=delta,
            RelativeGain=delta/parent['K'] if math.isfinite(parent['K']) and parent['K']>0 else 'UNDEFINED',direction=direction,
            resolution=spec['DELTA_resolution'],activated_groups=cap['activation_groups'],group_count=cap['total_groups'],activation_fraction=fraction,
            weak_activation=fraction<spec['weak_activation_fraction'],Parent_source_overflow=parent['clipped_values']/parent['state_values'],NoInflation_source_overflow=cap['clipped_values']/cap['state_values'],
            Parent_code_saturation=parent['saturated_codes']/parent['state_values'],NoInflation_code_saturation=cap['saturated_codes']/cap['state_values'])
        effects.append(effect)
        for start in [8192,16384,24576]:
            pk=parent['window_K'][str(start)];ck=cap['window_K'][str(start)]
            diag=rows(ROOT/f'students/doc_{d:02d}/NO_SCALE_INFLATION/SCALE_INTERVENTION_ACTIVATION.csv')
            diag=[x for x in diag if int(x['window_start'])==start]
            windows.append({'document_index':d,'document_id':parent['identity']['document_id'],'window_start':start,'window_end':start+8192,'position_count':128,
                 'Parent_KL':pk,'NoInflation_KL':ck,'paired_Delta':pk-ck,'contribution_to_overall_Delta':(pk-ck)/3,
                 'activation_fraction':math.fsum(float(x['activated_groups']) for x in diag)/math.fsum(float(x['groups']) for x in diag),
                 'source_overflow_fraction':math.fsum(float(x['clipped_values']) for x in diag)/math.fsum(float(x['state_values']) for x in diag)})
    csvout('DOCUMENT_LEVEL_PAIRED_EFFECT.csv',effects);csvout('TEMPORAL_EFFECT_BY_DOCUMENT.csv',windows)
    stats={'status':'INCOMPLETE','planned_N':8,'completed_paired_documents':len(effects),'old_pilot_merged':False,'statistical_unit':'DOCUMENT','document_results':effects,
           'project_independence':'authoritative project metadata absent; conditional inference, possible within-project correlation','actual_sign_test':None,'CI':None}
    verdict='INCOMPLETE_OR_ENGINEERING_STOP';nextstep='RESOLVE_FIRST_FAILURE_BEFORE_ANY_NEW_RESEARCH'
    if complete:
        positive=sum(x['direction']=='POSITIVE' for x in effects);negative=sum(x['direction']=='NEGATIVE' for x in effects);unresolved=8-positive-negative
        resolved=positive+negative
        p=math.fsum(math.comb(resolved,k) for k in range(positive,resolved+1))/2**resolved if resolved else None
        values=[x['Delta'] for x in effects];gains=[x['RelativeGain'] for x in effects if x['RelativeGain']!='UNDEFINED']
        import numpy as np
        rng=np.random.default_rng(20261013);boot=np.array(values)[rng.integers(0,8,size=(100000,8))].mean(axis=1)
        ci=np.percentile(boot,[2.5,97.5],method='linear').tolist()
        supported=unresolved==0 and positive>=7
        stats.update(status='DIRECTIONAL_REPLICATION_SUPPORTED_IN_8DOC_PANEL' if supported else 'DIRECTIONAL_REPLICATION_NOT_CONFIRMED',positive=positive,negative=negative,UNRESOLVED=unresolved,
           denominator=8,positive_fraction=positive/8,negative_fraction=negative/8,unresolved_fraction=unresolved/8,
           Delta_mean=math.fsum(values)/8,Delta_median=statistics.median(values),RelativeGain_values=[x['RelativeGain'] for x in effects],
           RelativeGain_mean=math.fsum(gains)/len(gains) if gains else None,RelativeGain_median=statistics.median(gains) if gains else None,
           RelativeGain_range=[min(gains),max(gains)] if gains else None,
           actual_sign_test={'one_sided_p':p,'n':resolved,'positive':positive,'interpretation':'primary N8 test only noUNRESOLVED; otherwise resolved-only descriptive sign statistic, full8 denominator retained','independence_assumption_verified_by_repository_metadata':False},
           CI={'method':'100000 whole-document bootstrap mean Delta, percentile linear interpolation','seed':20261013,'confidence':.95,'lower':ci[0],'upper':ci[1],'limits':'smallN; conditional on representative independent documents, within-project dependence unknown; diagnostic oracle/data/model/rotation scope only'},
           weak_activation_documents=[x['document_index'] for x in effects if x['weak_activation']],negative_documents=[x['document_index'] for x in effects if x['direction']=='NEGATIVE'],temporal_windows=windows)
        verdict='REPLICATION_DIRECTIONALLY_SUPPORTED' if supported else 'FUNCTIONAL_EFFECT_NOT_GENERAL_IN_THIS_PANEL' if negative>=2 else 'MIXED_OR_HETEROGENEOUS_FUNCTIONAL_EFFECT'
        nextstep='QWEN_GDN_TEMPORAL_READOUT_RISK_CAUSAL_DIAGNOSIS_V1_RECOMMENDED_ONLY_NOT_EXECUTED' if supported else 'DOCUMENT_AND_TIME_HETEROGENEITY_DIAGNOSIS_RECOMMENDED_ONLY_NOT_EXECUTED'
    save('REPLICATION_STATISTICAL_REPORT.json',stats)
    statuses={'ENGINEERING_STATUS':'PASS' if spec else 'INCOMPLETE_OR_ENGINEERING_STOP','NUMERICAL_QUALIFICATION_STATUS':'PASS' if spec and not failed else 'FAIL_OR_INCOMPLETE' if failed else 'NOT_RUN',
              'PHYSICAL_RESOURCE_STATUS':'PASS_FULL32768_CONFIRMED_WITH_VERIFIED_REMOTE_ARCHIVE' if complete else 'INCOMPLETE_OR_STOP','FORMAL_8DOC_STATUS':'COMPLETE' if complete else 'INCOMPLETE',
              'REPLICATION_STATISTICAL_STATUS':stats['status'],'FUNCTIONAL_CAUSAL_REPLICATION_VERDICT':verdict,'NEXT_RECOMMENDED_EXPERIMENT':nextstep}
    save('SCIENTIFIC_VERDICT.json',dict(statuses,first_failure=failed,completed_FP=len(refs),completed_anchors=len(anchors),completed_continuations=len(branches),completed_remote_archives=len(archives),
         statistical_independence_limitation=stats['project_independence'],new_C8_training='NOT_AUTHORIZED_NOT_EXECUTED',AIME_free_generation_new_rotation_surrogate_search=False,NoDeflation_new_formal_branches=0,all_document_raw_effects=effects))
    units=[]
    for n in ['ANCHOR_CHECKPOINT_AND_FORK_PARITY.json','qualification/observer_reducer_1.json','qualification/observer_reducer_2.json']:
        if (ROOT/n).exists():
            x=load(ROOT/n);units.append({'path':n,'GPUh':x['GPUh'],'seconds':x['seconds'],'peak_allocated':x['peak_allocated'],'peak_reserved':x['peak_reserved'],'SHA':sha(ROOT/n),'role':'new_incremental_qualification'})
    for d,x in refs.items():units.append(dict(x,receipt_path=f'references/doc_{d:02d}/FP_REFERENCE_MANIFEST.json',role='formal_reference'))
    for d,x in anchors.items():units.append(dict(x,receipt_path=f'anchors/doc_{d:02d}/ANCHOR_MANIFEST.json',role='formal_anchor'))
    for (d,a),x in branches.items():units.append(dict(x,receipt_path=f'students/doc_{d:02d}/{a}/BRANCH_MANIFEST.json',role='formal_branch'))
    failed_h=failed.get('unit_GPUh_including_initialization',0) if failed else 0
    total=math.fsum(x['GPUh'] for x in units)+failed_h
    retained_remote=sum(x['bytes'] for x in archives.values());local_bytes=sum(p.stat().st_size for p in small_and_local_files())
    progress=[]
    for p in (ROOT/'execution').glob('*_progress.jsonl'):
        for line in p.read_text().splitlines():progress.append(json.loads(line))
    ledger={'actual_GPUh':total,'completed_unit_GPUh':math.fsum(x['GPUh'] for x in units),'failed_unit_GPUh':failed_h,'definition':'one occupied GPU x original unit wall time including initialization; excludes CPU archive/audit and idle gaps',
        'copied_historical_qualification_GPUh_excluded':True,'unit_receipts':units,'peak_allocated_bytes':max([x['peak_allocated'] for x in units]+[0]),'peak_reserved_bytes':max([x['peak_reserved'] for x in units]+[0]),
        'actual_retained_archive29_bytes':retained_remote,'actual_current_compute_file_bytes_before_seal':local_bytes,'compute_data_free_bytes':__import__('shutil').disk_usage(ROOT).free,
        'cache_growth_and_actual_VRAM_observations':'execution/doc_*_progress.jsonl; snapshotted at seal','max_observed_cache_bytes':max([x['cache_bytes'] for x in progress]+[0]),
        'min_observed_physical_free_GPU_bytes':min([x['free'] for x in progress]+[0]) if not progress else min(x['free'] for x in progress),'plan_not_substituted_for_actual':True}
    save('GPU_COST_LEDGER.json',ledger)
    history=load(ROOT/'HISTORICAL_TEMPORAL_ANALYSIS.json');hw=[x for x in history['windows'] if x['condition']=='NO_SCALE_INFLATION']
    text=['# QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1 最终科学报告','',json.dumps(statuses,ensure_ascii=False,indent=2),'',
          '本轮在既有兼容RepoBench-P / RepoBench-P_e缓存上冻结8篇新文本/token去重文档。所有文档先固定再运行，历史2-doc不进入新统计。原数据没有权威repository/project字段；跨项目独立性无法核实，符号检验与bootstrap区间的解释均依赖独立性假设。','',
          '## 20项回答','',
          '1. 历史时间收益：'+ '; '.join('Doc%d [%d,%d) 平均改善=%+.12g，对总体Delta贡献=%+.12g'%(x['document_index'],x['window_start'],x['window_end'],x['mean_paired_improvement'],x['contribution_to_overall_Delta']) for x in hw)+'。Doc1第一窗口略恶化，后两窗口改善；Doc2三窗口改善。POSTHOC_DESCRIPTIVE_ANALYSIS。',
          '2. outcome-blind：是。789个不同原文候选、21个长度合格新候选在任何新GPU outcome前冻结；seed20261013与SHA排序，未使用KL/G/激活/收益。额外缓存发现造成的执行前暂停及独立审计记录完整保留。',
          '3. 长度/独立性：冻结8篇均提供32769原context tokens，排除真实历史registry 6个本域已用ID及150个raw SHA，并通过冻结文本/token近重复检查。repository/project身份UNKNOWN，不能宣称已证实8个独立项目。',
          '4. 数值合同：'+('PASS；原compensated scan、量化器、数学engine、全词表KL/reducer源文件不改；增量资格两新observer进程和8192旧anchor新进程next8逐字节一致；分辨率0.000690448106979602。' if spec else '未完成全部增量资格，未授权越门禁正式执行。'),
          '5. RTX3090完整轨迹：'+('全部完成。' if complete else '未全部完成；见FIRST_FAILURE_GATE与实际回执。'),
          f'6. FP references：{len(refs)}/8 有完成回执。',f'7. 共同8192 anchors：{len(anchors)}/8。',f'8. 学生完整continuations：{len(branches)}/16。',
          '9. Parent Future-KL：见下面所有实际成对文档逐项数值与DOCUMENT_LEVEL_PAIRED_EFFECT.csv。',
          '10. NoInflation Future-KL：同表；未测单位无虚构值。','11. signed Delta/relative gain：同表；Parent非正或非finite时UNDEFINED。',
          '12. 可分辨改善：'+(f"{stats['positive']}/8；负向{stats['negative']}/8；UNRESOLVED {stats['UNRESOLVED']}/8。" if complete else 'INCOMPLETE，不进行成功率确认。'),
          '13. exact sign test：'+(json.dumps(stats['actual_sign_test'],ensure_ascii=False) if complete else '未运行；N8正式未完成，不能引用7/8或8/8示例p值充作结果。'),
          '14. 反例/弱激活：'+(json.dumps({'negative_documents':stats['negative_documents'],'weak_activation_documents':stats['weak_activation_documents']},ensure_ascii=False)+'；弱激活真实保留，不更换或删除。' if complete else '未完整执行，不判科学方向。'),
          '15. 时间一致性：'+('所有8篇三窗口见TEMPORAL_EFFECT_BY_DOCUMENT.csv；不将窗口或384个位置作为独立样本，不根据窗口结果修改Primary。' if complete else '本轮未确认；历史仅两篇，Doc1与Doc2时间异质性明显。'),
          '16. clipping与改善共存：min scale会收紧范围，从而增加source overflow或饱和；长期KL反映整个独立闭环传播的总效应，两者不矛盾，也未证明clipping是唯一中介。实际新文档触发/overflow/saturation见下表及INTERVENTION_ACTIVATION.csv。历史NoInf两doc触发率22.6853%/21.9178%，source overflow约0.7022%/0.7285%，Parent约0.0341%/0.0341%；同时总体KL下降。',
          f"17. 本轮实际GPUh={total:.12g}，peak allocated={ledger['peak_allocated_bytes']} bytes，reserved={ledger['peak_reserved_bytes']} bytes；已验证远端保留={retained_remote} bytes；当前本地文件={local_bytes} bytes（封存前）。复制的旧资格GPUh不重复计入。cache/free VRAM/wall-clock见原单位回执、进度及GPU_COST_LEDGER。",
          '18. 独立复制：'+verdict+'；只解释本模型/C6R/已有代码数据域/oracle干预/冻结协议，不是部署方法或全GDN模型结论。',
          '19. 下一步：'+nextstep+'。仅建议，本次不会启动新实验；历史已存在状态与功能不同步及时间异质性，不能绕过新8-doc统计。',
          '20. C8训练证据：不充分，NEW_C8_TRAINING=NOT_AUTHORIZED，未执行。','',
          '|Doc|Parent KL|NoInflation KL|Delta|RelativeGain|方向|触发率|Parent overflow|NoInf overflow|','|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for x in effects:text.append('|%d|%.12g|%.12g|%+.12g|%s|%s|%.6f%%|%.6f%%|%.6f%%|'%(x['document_index'],x['K_PARENT'],x['K_NOINFLATION'],x['Delta'],('%.6f%%'%(100*x['RelativeGain'])) if x['RelativeGain']!='UNDEFINED' else 'UNDEFINED',x['direction'],100*x['activation_fraction'],100*x['Parent_source_overflow'],100*x['NoInflation_source_overflow']))
    text += ['','文档bootstrap区间与小N/项目相关性限制：'+json.dumps(stats.get('CI'),ensure_ascii=False),'',
             '科学决策：'+verdict+'。NoInflation为clean-FP oracle诊断干预，没有验证可部署scale cap、S1/S2训练Loss、旋转排名或AIME改善，也没有运行新NoDeflation以复制其方向特异性。',
             '稀疏状态诊断仅16384/24576/32768三个预冻结终点；state-MSE不替代KL。所有大tensor保存在202.38.247.29的逐doc封存目录，位置映射及SHA见archive/DOC_*_CURRENT_ASSET_LOCATIONS.json。旧正式资产未删除或修改。',
             '第一失败：'+json.dumps(failed,ensure_ascii=False),'']
    with (ROOT/'FINAL_SCIENTIFIC_REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    with (ROOT/'RESEARCH_DECISION_REPORT.md').open('x') as f:f.write('# 科研决策\n\n'+verdict+'\n\n'+nextstep+'\n\n只建议，不启动后续研究；项目独立性限制、oracle部署差距及小N限制仍然成立。C8训练未授权。\n')
    events=ROOT/'execution/COORDINATOR_EVENTS.jsonl'
    if events.exists():
        snapshot=ROOT/'execution/COORDINATOR_EVENTS_SEALED_SNAPSHOT.jsonl'
        with snapshot.open('xb') as f:f.write(events.read_bytes())
    files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in small_and_local_files()]
    external=[]
    for d in archives:
        external+=load(ROOT/f'archive/DOC_{d:02d}_CURRENT_ASSET_LOCATIONS.json')['assets']
    save('RESULT_AND_DELIVERY_HASHES.json',{'status':'SEALED_COMPLETE' if complete else 'SEALED_INCOMPLETE_OR_ENGINEERING_STOP','time_unix':time.time(),'files':files,'archived_tensor_files':external,
         'live_coordinator_events_excluded_snapshot_included':True,'all_remote_asset_SHA_verified_before_local_relocation':True,'large_tensors_not_for_local_download':True,'scientific_verdict':statuses})
    with (ROOT/'SHA256SUMS.txt').open('x') as f:
        for x in files:f.write(x['sha256']+'  '+x['path']+'\n')
        f.write(sha(ROOT/'RESULT_AND_DELIVERY_HASHES.json')+'  RESULT_AND_DELIVERY_HASHES.json\n')
    print('REPLICATION_COMPLETE_AND_SEALED' if complete else 'REPLICATION_REAL_STOP_SEALED',json.dumps(statuses),flush=True)

if __name__=='__main__':main()
