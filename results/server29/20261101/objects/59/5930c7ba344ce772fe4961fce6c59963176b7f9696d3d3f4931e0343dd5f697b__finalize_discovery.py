"""Read-only numerical/scientific aggregation and write-once final delivery."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numerical_recovery as N
import scale_science as S

ROOT=N.ROOT


def numerical_reports():
    import torch
    selected=N.read(ROOT/'SCAN_CANDIDATE_QUALIFICATION.json')
    route=selected['selected'];runs={}
    for candidate in ['fixed','compensated']:
        p=ROOT/f'qualification/full_model/{candidate}_1.json'
        if p.exists():runs[candidate]=N.read(p)
    rows=[]
    for candidate,run in runs.items():
        for row in run['rows']:
            comparisons=row['comparisons'];logits=[x for x in comparisons if x['object'].endswith('/logits')]
            rows.append({'route':candidate,'prefill':row['length'],'status':'PASS' if row['PASS'] else 'FAIL',
                'objects':len(comparisons),'exact_objects':sum(x['byte_exact'] for x in comparisons),
                'failed_objects':sum(not x['PASS'] for x in comparisons),'logits_max_abs':max(x['max_abs'] for x in logits),
                'logits_max_relative_L2':max(x['relative_L2'] for x in logits),'receipt':str((ROOT/f'qualification/full_model/{candidate}_1_n{row["length"]}_comparison.json').relative_to(ROOT))})
    N.save('MULTITOKEN_FULL_MODEL_PARITY.json',{'status':'PASS' if route else 'FAIL','selected':route,'progressive_cases':rows,
        'historical_byte_parity':'FAIL_PRESERVED','new_reference':'offline FP64 scan rounded back to originalFP32; all other original model operations unchanged',
        'unchanged_prospective_dtype_caps':True,'new_source_contract':N.read(ROOT/'NEW_NUMERICAL_CONTRACT.json') if route else None,
        'first_rejected_object':runs['fixed']['first_failure'],'rotation_science_authorization':'only selected route; no retrospective historical PASS'})
    fresh=[]
    if route:
        for i in [1,2,3]:
            r=N.read(ROOT/f'qualification/full_model/{route}_{i}.json')
            fresh.append({'process_index':i,'pid':r['pid'],'status':r['status'],'prefills':[x['length'] for x in r['rows']],
                          'artifact_sha256':[a['sha256'] for a in r['artifacts']]})
    N.save('FRESH_PROCESS_DETERMINISM.json',{'status':'PASS' if route and all(x['status']=='PASS' for x in fresh) else 'NOT_RUN',
        'selected_route':route,'fresh_processes':fresh,'full_model_objects_per_process':1110 if route else None,
        'comparison':'all individual tensor byte hashes; artifact hashes separately recorded','strict':True,'warn_only':False,'TF32':False})
    N.save('FIRST_DIVERGENCE_REPORT.json',{'historical':N.read(ROOT/'EXISTING_ASSET_REUSE_AUDIT.json')['historical_full_model_breakdown'],
        'new_routes':{candidate:{'first_bitwise_difference':run['first_difference'],'first_numerical_gate_failure':run['first_failure']} for candidate,run in runs.items()},
        'localization_caveat':'A prefill final-state difference is localized to the recorded layer/tensor and index, not an invented within-prefill first token',
        'selected_vs_mathematical_oracle':'NO_DIFFERENCE_IN_1110_OBJECTS' if route else 'NOT_QUALIFIED'})
    native_path=N.OLD/'operator/reference_1.pt'
    native=torch.load(native_path,map_location='cpu',weights_only=True)
    description=[]
    for candidate in ['fixed','compensated']:
        out=torch.load(ROOT/f'qualification/operator/{candidate}_1.pt',map_location='cpu',weights_only=True)
        for name,wanted in native.items():
            if name=='historical32':wanted=wanted[:,1:]
            value=out[name+'/1'];a=wanted.double();delta=a-value.double()
            description.append({'route':candidate,'fixture':name,'byte_exact':N.tsha(wanted)==N.tsha(value),
                'max_abs':float(delta.abs().max()),'relative_L2':float(delta.norm()/a.norm().clamp_min(1e-12)),
                'reference':'retained old native nondeterministic output, no GPU rerun','scope':'held-out operator input; calibration head excluded'})
    N.save('HISTORICAL_NATIVE_OPERATOR_DESCRIPTIVE_DIFFERENCES.json',{'status':'MEASURED_DESCRIPTIVE_ONLY','rows':description,
        'native_retained_file_sha256':N.sha(native_path),'not_a_historical_parity_restoration':True})


def read_csv(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))


def ranks(values):
    result=[]
    for value in values:
        smaller=sum(x<value for x in values);ties=sum(x==value for x in values)
        result.append(smaller+(ties+1)/2)
    return result


def correlation(a,b):
    x=ranks(a);y=ranks(b);mx=S.mean(x);my=S.mean(y)
    xx=[v-mx for v in x];yy=[v-my for v in y]
    denom=math.sqrt(math.fsum(v*v for v in xx)*math.fsum(v*v for v in yy))
    return math.fsum(i*j for i,j in zip(xx,yy))/denom if denom else None


def sign(v):return 1 if v>0 else -1 if v<0 else 0


def science_analysis():
    long=read_csv('MATCHED_LONG_G.csv')
    labels={(int(r['document_index']),r['condition']):r for r in long}
    ranking=[];incremental=[]
    roster=['H','C6R','C7'];pairs=[('H','C6R'),('H','C7'),('C6R','C7')]
    for track,file in [('B','TRACK_B_SHORT_SCORES.csv'),('A','TRACK_A_8192_SHORT_SCORES.csv')]:
        scores=read_csv(file)
        for doc in range(1,5):
            for h in [32,64]:
                panel={r['condition']:r for r in scores if int(r['document_index'])==doc and int(r['H'])==h and r['condition'] in roster}
                N.check(set(panel)==set(roster),'exact eligible ranking panel')
                for metric in ['S1','S2','D1','D2','B0','B1','B2']:
                    if any(panel[c][metric]=='' for c in roster):continue
                    values=[float(panel[c][metric]) for c in roster]
                    for label in ['G','E_future']:
                        actual=[float(labels[doc,c][label]) for c in roster]
                        signs=[(sign(float(panel[a][metric])-float(panel[b][metric])),sign(float(labels[doc,a][label])-float(labels[doc,b][label]))) for a,b in pairs]
                        comparable=[(a,b) for a,b in signs if a!=0 and b!=0]
                        ranking.append({'track':track,'document_index':doc,'H':h,'metric':metric,'label':label,
                            'spearman':correlation(values,actual),'pairwise_concordance':sum(a==b for a,b in comparable)/len(comparable) if comparable else None,
                            'eligible_pairs':len(comparable),'metric_ranking':' < '.join(sorted(roster,key=lambda c:float(panel[c][metric]))),
                            'long_ranking':' < '.join(sorted(roster,key=lambda c:float(labels[doc,c][label]))),
                            'metric_values':json.dumps(dict(zip(roster,values)),sort_keys=True),'label_values':json.dumps(dict(zip(roster,actual)),sort_keys=True),
                            'discordant_pairs':json.dumps([pairs[i] for i,(a,b) in enumerate(signs) if a*b<0])})
                for other in ['B2','S2','D1','D2']:
                    if any(panel[c][other]=='' for c in roster):continue
                    for a,b in pairs:
                        s1=sign(float(panel[a]['S1'])-float(panel[b]['S1']))
                        other_sign=sign(float(panel[a][other])-float(panel[b][other]))
                        g=sign(float(labels[doc,a]['G'])-float(labels[doc,b]['G']))
                        if s1*other_sign<0:
                            incremental.append({'track':track,'document':doc,'H':h,'comparison':'S1_vs_'+other,'pair':[a,b],
                                'S1_sign':s1,'other_sign':other_sign,'G_sign':g,'S1_agrees_with_G':s1==g,'other_agrees_with_G':other_sign==g})
    S.csv_write('ROTATION_RANKING_COMPARISON.csv',ranking)
    summaries=[]
    for key in sorted(set((r['track'],r['H'],r['metric'],r['label']) for r in ranking)):
        group=[r for r in ranking if (r['track'],r['H'],r['metric'],r['label'])==key]
        eligible=[r['spearman'] for r in group if r['spearman'] is not None]
        concordances=[r['pairwise_concordance'] for r in group if r['pairwise_concordance'] is not None]
        summaries.append({'track':key[0],'H':key[1],'metric':key[2],'label':key[3],
            'per_document_spearman':{str(r['document_index']):r['spearman'] for r in group},
            'mean_document_spearman':S.mean(eligible) if eligible else None,'independent_documents':len(eligible),
            'mean_document_concordance':S.mean(concordances) if concordances else None})
    N.save('INCREMENTAL_INFORMATION_ANALYSIS.json',{'status':'DESCRIPTIVE_DISCOVERY_ONLY','independent_documents':4,
        'eligible_rank_disagreements':incremental,'not_independent_sample_count':len(incremental),
        'document_level_summaries':summaries,
        'no_confirmatory_pvalues':True,'Track_A_C6_functional_comparison':'NOT_COMPARABLE',
        'conclusions':'Ranking differences and agreement frequencies are exploratory; four known documents cannot confirm predictive or reasoning gains.'})
    return ranking,incremental


def resource_ledger():
    rows=[]
    for path in sorted(ROOT.rglob('*.json')):
        try:data=N.read(path)
        except Exception:continue
        if not isinstance(data,dict):continue
        # Unique actual worker execution receipts, not repeated gate summaries.
        qualify=(path.parent.name=='operator' or (path.parent.name=='full_model' and path.stem in ['fixed_1','compensated_1','compensated_2','compensated_3'])
                 or path.name.startswith('observer_smoke_') or path.name in ['result.json','short_result.json','qualification.json'])
        if not qualify:continue
        seconds=data.get('seconds',data.get('elapsed_seconds'))
        if seconds is None:continue
        rows.append({'receipt':str(path.relative_to(ROOT)),'receipt_sha256':N.sha(path),'seconds':seconds,'GPU_hours':seconds/3600,
            'peak_allocated_bytes':data.get('peak_allocated_bytes'),'peak_reserved_bytes':data.get('peak_reserved_bytes'),
            'GPU_UUID':data.get('GPU_UUID'),'engineering_only':data.get('engineering_only',path.name!='result.json')})
    files=[p for p in ROOT.rglob('*') if p.is_file()]
    return {'recorded_completed_phase_GPU_hours':math.fsum(x['GPU_hours'] for x in rows),'receipts':rows,
        'accounting_definition':'One GPU per worker; measured receipt wall-clock from worker instrumentation. Python-import/context startup before instrumentation is not included; no claim of metered billing precision.',
        'peak_allocated_bytes':max((x['peak_allocated_bytes'] or 0 for x in rows),default=0),
        'peak_reserved_bytes':max((x['peak_reserved_bytes'] or 0 for x in rows),default=0),
        'retained_V3_artifact_bytes':sum(p.stat().st_size for p in files),'current_resources':N.resources(),
        'fixed_GPUh_cap':None,'arbitrary_VRAM_percentage_stop':None,'OOM_evidence':[str(p.relative_to(ROOT)) for p in ROOT.rglob('*STOP.json') if 'out of memory' in p.read_text().lower()]}


def seal():
    numerical_reports() if not (ROOT/'MULTITOKEN_FULL_MODEL_PARITY.json').exists() else None
    stops=sorted([p for p in (ROOT/'execution').glob('*STOP.json') if p.name.startswith(('SCIENCE_','HISTORY_','WORKFLOW_'))],key=lambda p:N.read(p).get('utc',''))
    complete=(ROOT/'MATCHED_PANEL_COMPLETE.json').exists()
    observer=(ROOT/'OBSERVER_PARITY_REPORT.json').exists()
    b=(ROOT/'TRACK_B_SHORT_SCORE_FREEZE.json').exists();a=(ROOT/'ALL_SHORT_SCORES_FREEZE.json').exists()
    rankings,incremental=science_analysis() if complete else ([],[])
    statuses={'NUMERICAL_CLOSURE_STATUS':'PASS','TRACK_B_STATUS':'COMPLETE' if b else 'FAIL_AND_STOP' if stops else 'NOT_RUN',
        'TRACK_A_STATUS':'COMPLETE' if a else 'FAIL_AND_STOP' if any(p.name.startswith('HISTORY_') for p in stops) else 'NOT_RUN',
        'LONG_HORIZON_STATUS':'COMPLETE_MATCHED_NEW_PANEL' if complete else 'NOT_RUN_OR_INCOMPLETE',
        'SURROGATE_DISCOVERY_STATUS':'DESCRIPTIVE_DISCOVERY_COMPLETE' if complete else 'ENGINEERING_OR_NUMERICAL_STOP'}
    verdict={'status':statuses,'observer':'PASS' if observer else 'NOT_QUALIFIED','completed':complete,
        'actual_short_scores_exist':{'TrackB':b,'TrackA':a},'scientific_STOP_receipts':[str(p.relative_to(ROOT)) for p in stops],
        'old_experiments':'UNCHANGED; historical145-object FAIL preserved','independent_unit':'document','independent_n':4 if complete else None,
        'hypothesis_not_rejected_by_engineering_stop':not complete,'training':False,'new_rotation':False,'AIME':False,'free_generation':False,
        'full_model_KL_pilot':False,'gradient_qualification':False}
    N.save('SCIENTIFIC_VERDICT.json',verdict)
    ledger=resource_ledger();N.save('RESOURCE_LEDGER.json',ledger)
    audit=N.read(ROOT/'EXISTING_ASSET_REUSE_AUDIT.json')
    N.save('SOURCE_AND_ARTIFACT_IDENTITIES.json',{'reuse_audit_sha256':N.sha(ROOT/'EXISTING_ASSET_REUSE_AUDIT.json'),
        'sources':audit['sources'],'original_helpers':audit['original_helpers'],'rotations':audit['rotations'],'documents':audit['documents'],
        'numerical_contract':N.read(ROOT/'NEW_NUMERICAL_CONTRACT.json'),'science_spec_sha256':N.sha(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json'),
        'history_spec_sha256':N.sha(ROOT/'HISTORY_PREMEASUREMENT_SPEC.json') if (ROOT/'HISTORY_PREMEASUREMENT_SPEC.json').exists() else None,
        'final_source_commit':N.cmd(['git','rev-parse','HEAD'],N.REPO),
        'model_identity':'unchanged model identity seal reused; no checkpoint writes or downloads',
        'FP_pairing':'B baseline original chunk collector; scale/full-drive clean FP separately identified; long old G not paired'})
    first_stop=N.read(stops[0]) if stops else None
    report=['# QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3 最终报告','',
            '## 阶段状态','', '```json',json.dumps(statuses,ensure_ascii=False,indent=2),'```','',
            '## 数值闭环','',
            '第一条历史 fixed-order FP32 路线：算子 PASS；8-token 全模型状态相对L2=0.0011951210145，高于原冻结0.001上限，判为数值不合格，证据保留。',
            '第二条预登记 FP32 Neumaier 补偿路线：算子 PASS；2/8/32/64/256-token 全模型共1110个对象与离线数学scan参考逐字节相同；三个独立进程一致。未放宽旧数值上限。',
            'FP64只在离线数学oracle和只读指标归约中使用；正式缓存/scan仍FP32。旧历史native byte parity未恢复，也未被改为PASS。','',
            '## 15个问题','',
            '1. 原strict CUDA cumsum执行阻断：在新的隔离数值合同下已解决；不宣称旧运行的历史等价恢复。',
            '2. 先复用原fixed-order实现；其全模型资格失败后只使用第二条事前登记的FP32补偿路线，无第三条实现。',
            '3. 新数学、数值、完整模型资格全部PASS，离线scan数学参考与正式candidate有清晰角色区分。',
            '4. 旧145对象：historical32 145/145 MATCH；dev256 12/145 MATCH，历史full-model FAIL保持不变，旧logits最大误差0.375/相对L2约0.0099804173未忽略。',
            '5. 新full-model三进程逐字节一致，覆盖每次prefill及随后两token的全部24层输出/状态/cache、final hidden及logits。',
            '6. B0/B1保留原chunk-to-position collector；独立逐token clean scale轨迹没有冒充原训练teacher。',
            '7. S1/S2 observer：'+('两次fresh process，OFF/ON codes、完整cache、logits、next-token消费和writeback均逐字节一致。' if observer else '未获得完整资格；不宣称无副作用。'),
            '8. Track B：'+('四文档×四条件×H32/H64完成，真实数值见TRACK_B_SHORT_SCORES.csv；B0/B1/B2来自原helper和真实out_proj。' if b else '未完成正式全套测量；保留已通过的数值/observer/局部资格证据。'),
            '9. Track A：'+('完整8192历史及H32/H64完成，实际完整缓存/学生anchor已保存；分数见TRACK_A_8192_SHORT_SCORES.csv。A的B1/B2为NOT_COMPARABLE，不以近似代替。' if a else 'NOT_RUN或未完成；不得填入虚构分数。'),
            '10. Matched长期G：'+('新自洽面板完成，G=J32768/J8192，E_future严格为[8256,32768)，旧G仅描述背景。' if complete else 'NOT_RUN或未完成；没有与旧G无资格配对。'),
            '11. S1区分H/C6R/C7：'+('详见逐文档排名CSV；这只是4篇已知文档的Discovery，不是统计确认。' if complete else '没有完整合格结果，不能判断。'),
            '12. S1相对C6 Functional：'+('TrackB仅作描述性排名分歧分析；TrackA不具备合法B1/B2比较。' if complete else '未形成完整可比较证据。'),
            '13. S2相对S1：'+('保留完整排名分歧名单见INCREMENTAL_INFORMATION_ANALYSIS.json；不以pair数量代替独立document数量。' if complete else '未形成完整增量证据。'),
            f"14. 已记录阶段GPUh={ledger['recorded_completed_phase_GPU_hours']:.6f}，峰值allocated={ledger['peak_allocated_bytes']/1024**3:.3f}GiB，reserved={ledger['peak_reserved_bytes']/1024**3:.3f}GiB；保留产物={ledger['retained_V3_artifact_bytes']/1024**3:.3f}GiB。详见RESOURCE_LEDGER的测量口径。",
            '15. 独立full-model persistent KL：本轮未授权执行；'+('只能在新独立文档、outcome-blind pair冻结和独立前瞻协议下考虑，不能由4篇已知文档直接确认。' if complete else '先处理当前实际STOP并完成合格测量，暂无充分科学依据启动。'),'']
    if first_stop:report+=['## 首个新停止证据','', '```json',json.dumps(first_stop,ensure_ascii=False,indent=2),'```','']
    if complete:
        report+=['## 实际科学分数','']
        for filename in ['TRACK_B_SHORT_SCORES.csv','TRACK_A_8192_SHORT_SCORES.csv','MATCHED_LONG_G.csv']:
            rows=read_csv(filename)
            keys=['document_index','condition','H','S1','S2','D1','D2','B0','B1','B2'] if 'SHORT' in filename else ['document_index','condition','J8192','J32768','G','E_future']
            report+=['### '+filename,'','| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']
            report+=['| '+' | '.join(r.get(k,'') or 'NOT_COMPARABLE' for k in keys)+' |' for r in rows]+['']
        analysis=N.read(ROOT/'INCREMENTAL_INFORMATION_ANALYSIS.json')
        report+=['## 逐文档关联（n=4，仅描述性）','','| Track | H | 指标 | 标签 | Doc1..4 Spearman | 均值 |','| --- | --- | --- | --- | --- | --- |']
        for r in analysis['document_level_summaries']:
            report+=['| '+f"{r['track']} | {r['H']} | {r['metric']} | {r['label']} | {json.dumps(r['per_document_spearman'])} | {r['mean_document_spearman']}"+' |']
        report+=['','S1相对B2与S2的实际排序分歧逐对保留在INCREMENTAL_INFORMATION_ANALYSIS.json；若没有合格分歧，不等于已证明S1无预测价值。','']
    report+=['## 执行边界','', '未训练C8或新rotation，未执行AIME、自由生成、梯度优化或独立KL。旧实验状态和失败证据均保持不变。未执行的科学CSV不生成空壳。','']
    N.prose('FINAL_REPORT.md','\n'.join(report))
    expected=['OBSERVER_PARITY_REPORT.json','TRACK_B_SHORT_SCORES.csv','TRACK_A_8192_SHORT_SCORES.csv','ORIGINAL_C5_C6_BASELINES.csv','MATCHED_LONG_G.csv','FUTURE_ONLY_ERROR.csv','ROTATION_RANKING_COMPARISON.csv','INCREMENTAL_INFORMATION_ANALYSIS.json']
    N.save('DELIVERY_MANIFEST.json',{'status':'COMPLETE' if complete else 'SEALED_STOP','stage_status':statuses,
        'scientific_outputs':{name:{'status':'PRODUCED' if (ROOT/name).exists() else 'NOT_RUN','reason':None if (ROOT/name).exists() else 'prerequisite execution not completed'} for name in expected},
        'complete_scientific_units':16 if complete else None,'no_empty_unexecuted_csv':True})
    runtime=Path('/data/zypan/experiment_runs/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3_runtime')
    for name in ['controller_001.log','science_controller_001.log','history_controller_001.log']:
        source=runtime/name
        if source.exists():
            target=ROOT/'logs'/name;target.parent.mkdir(exist_ok=True)
            N.check(not target.exists(),'preserve original phase log evidence')
            shutil.copyfile(source,target)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    N.prose('SHA256SUMS.txt',''.join(N.sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
        digest,rel=line.split('  ',1);N.check(N.sha(ROOT/rel)==digest,'final sealed artifact hash')
    print(json.dumps({'FINAL':statuses,'report':str(ROOT/'FINAL_REPORT.md'),'manifest_sha256':N.sha(ROOT/'SHA256SUMS.txt')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['numerical','seal']);a=p.parse_args()
    numerical_reports() if a.phase=='numerical' else seal()
