"""CPU-only, posthoc descriptive analysis of the sealed historical2-doc pilot."""
import ast
import csv
import hashlib
import json
import math
import os
import pathlib
import types

ROOT = pathlib.Path(__file__).resolve().parent
OLD = pathlib.Path('/data/zypan/worktrees/qwen-gdn-scale-feedback-fullmodel-functional-causal-pilot-v1/experiments/QWEN_GDN_SCALE_FEEDBACK_FULLMODEL_FUNCTIONAL_CAUSAL_PILOT_V1')
LABEL = 'POSTHOC_DESCRIPTIVE_ANALYSIS'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(n, value):
    p = ROOT / n
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')

def csv_save(n, data):
    assert data
    with (ROOT / n).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)

def read_rows(n):
    with (OLD / n).open(newline='') as f:
        return list(csv.DictReader(f))

def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return ast.unparse(fn)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    spec = json.loads((OLD / 'FROZEN_SCIENCE_SPEC.json').read_text())
    seal = json.loads((OLD / 'RESULT_AND_DELIVERY_HASHES.json').read_text())
    assert sha(OLD / 'FROZEN_SCIENCE_SPEC.json') == '1f24740272d941433afb1e4f8f616e72c3bc2dc29993b89130bdef41b8c9c200'
    assert sha(OLD / 'RESULT_AND_DELIVERY_HASHES.json') == 'bdfe7f26f2ffdd083cfe1dab85cc915362a3e6d7f744bdc0e81f6782d0344a7c'
    assert seal['status'] == 'SEALED_COMPLETE'
    hashes = {x['path']: x['sha256'] for x in seal['files']}
    inputs = ['FROZEN_SCIENCE_SPEC.json', 'SAMPLED_FULLVOCAB_FUTURE_KL.csv', 'SCALE_INTERVENTION_ACTIVATION.csv', 'CLIPPING_AND_SATURATION.csv', 'STATE_DAMAGE_ENDPOINTS.csv', 'DOCUMENT_LEVEL_CAUSAL_EFFECT.csv', 'SCIENTIFIC_VERDICT.json']
    for n in inputs:
        assert sha(OLD / n) == hashes[n]
    mean_source = pathlib.Path(next(n for n in spec['source_SHA'] if n.endswith('/scale_science.py')))
    check_source = pathlib.Path(next(n for n in spec['source_SHA'] if n.endswith('/numerical_recovery.py')))
    assert sha(mean_source) == spec['source_SHA'][str(mean_source)]
    assert sha(check_source) == spec['source_SHA'][str(check_source)]
    ns = {}; check_ast = extract(check_source, 'check', ns)
    reduction_ns = {'N': types.SimpleNamespace(check=ns['check']), 'math': math}
    mean_ast = extract(mean_source, 'mean', reduction_ns); mean = reduction_ns['mean']
    kl = read_rows('SAMPLED_FULLVOCAB_FUTURE_KL.csv')
    scale = read_rows('SCALE_INTERVENTION_ACTIVATION.csv')
    state = read_rows('STATE_DAMAGE_ENDPOINTS.csv')
    assert len(kl) == 2304
    positions = list(range(8192, 32768, 64)); assert positions == spec['positions']
    paired = []; windows = []; extremes = []; largest = []; docs = []
    for d in spec['documents']:
        idx = d['document_index']; uid = d['document_id']
        series = {}
        for arm in ['PARENT', 'NO_SCALE_INFLATION', 'NO_SCALE_DEFLATION']:
            r = [x for x in kl if x['document_id'] == uid and x['condition'] == arm]
            assert [int(x['token_index']) for x in r] == positions
            series[arm] = [float(x['KL_FP32']) for x in r]
        di = [a-b for a,b in zip(series['PARENT'], series['NO_SCALE_INFLATION'])]
        dd = [a-b for a,b in zip(series['PARENT'], series['NO_SCALE_DEFLATION'])]
        primary = mean(series['PARENT'])-mean(series['NO_SCALE_INFLATION'])
        assert math.isclose(mean(di), primary, rel_tol=1e-12, abs_tol=1e-14)
        threshold = sorted(series['PARENT'])[math.ceil(.95*384)-1]
        for k,t in enumerate(positions):
            row = {'analysis_status':LABEL, 'document_index':idx, 'document_id':uid, 'token_index':t,
                   'Parent_KL':series['PARENT'][k], 'NoInflation_KL':series['NO_SCALE_INFLATION'][k], 'NoDeflation_KL':series['NO_SCALE_DEFLATION'][k],
                   'DeltaInf':di[k], 'DeltaDef':dd[k], 'cumulative_mean_DeltaInf':mean(di[:k+1]), 'cumulative_mean_DeltaDef':mean(dd[:k+1])}
            paired.append(row)
            if series['PARENT'][k] >= threshold:
                extremes.append(dict(row, definition='Parent KL >= per-document nearest-rank95th percentile', Parent_KL_95pct_threshold=threshold, window_start=8192+8192*((t-8192)//8192)))
        for arm,delta in [('NO_SCALE_INFLATION',di),('NO_SCALE_DEFLATION',dd)]:
            for k in sorted(range(384),key=lambda j:(-abs(delta[j]),positions[j]))[:20]:
                largest.append({'analysis_status':LABEL,'document_index':idx,'condition':arm,'token_index':positions[k],'paired_difference':delta[k],'absolute_difference':abs(delta[k])})
            for start in [8192,16384,24576]:
                inds=[k for k,t in enumerate(positions) if start<=t<start+8192]
                delta_w=[delta[k] for k in inds]
                stats=[x for x in scale if x['document_id']==uid and x['condition']==arm and int(x['window_start'])==start]
                parent_stats=[x for x in scale if x['document_id']==uid and x['condition']=='PARENT' and int(x['window_start'])==start]
                assert len(stats)==len(parent_stats)==24
                groups=math.fsum(float(x['groups']) for x in stats); values=math.fsum(float(x['state_values']) for x in stats)
                contribution=math.fsum(delta_w)/384
                total_delta=mean(series['PARENT'])-mean(series[arm])
                row={'analysis_status':LABEL,'document_index':idx,'document_id':uid,'condition':arm,'window_start':start,'window_end':start+8192,'position_count':128,
                     'mean_Parent_KL':mean([series['PARENT'][k] for k in inds]),'mean_intervention_KL':mean([series[arm][k] for k in inds]),'mean_paired_improvement':mean(delta_w),
                     'contribution_to_overall_Delta':contribution,'contribution_fraction':contribution/total_delta if total_delta!=0 else 'UNDEFINED',
                     'positive_position_fraction':sum(x>0 for x in delta_w)/128,'negative_position_fraction':sum(x<0 for x in delta_w)/128,'zero_position_fraction':sum(x==0 for x in delta_w)/128,
                     'extreme_Parent_event_count':sum(series['PARENT'][k]>=threshold for k in inds),
                     'intervention_activated_groups':int(math.fsum(float(x['activated_groups']) for x in stats)),
                     'intervention_activation_fraction':math.fsum(float(x['activated_groups']) for x in stats)/groups,
                     'intervention_source_overflow_fraction':math.fsum(float(x['clipped_values']) for x in stats)/values,
                     'Parent_source_overflow_fraction':math.fsum(float(x['clipped_values']) for x in parent_stats)/values,
                     'intervention_code_saturation_fraction':math.fsum(float(x['saturated_codes']) for x in stats)/values}
                windows.append(row)
        state_means={}
        for arm in series:
            r=[x for x in state if x['document_id']==uid and x['condition']==arm]
            assert [int(x['consumed_tokens']) for x in r]==[16384,24576,32768]
            state_means[arm]=mean([float(x['relative_state_MSE']) for x in r])
        docs.append({'document_index':idx,'document_id':uid,'primary_Delta_unchanged':primary,
                     'NoInflation_positive_position_fraction':sum(x>0 for x in di)/384,
                     'NoInflation_negative_position_fraction':sum(x<0 for x in di)/384,
                     'NoDeflation_mean_Delta':mean(series['PARENT'])-mean(series['NO_SCALE_DEFLATION']),
                     'state_three_sparse_endpoint_observation_means':state_means,
                     'Parent_KL_95pct_threshold':threshold})
    csv_save('HISTORICAL_PAIRED_POSITION_DIFFERENCES.csv',paired)
    csv_save('HISTORICAL_TEMPORAL_WINDOW_EFFECTS.csv',windows)
    csv_save('HISTORICAL_EXTREME_KL_EVENTS.csv',extremes)
    csv_save('HISTORICAL_LARGEST_PAIRED_DIFFERENCES.csv',largest)
    save('HISTORICAL_TEMPORAL_ANALYSIS.json',{'status':LABEL,'new_formal_GPU_forward_count':0,'historical_primary_unchanged':True,
         'historical_data_not_added_to_new8doc_statistics':True,'input_files':[{'path':str(OLD/n),'SHA':sha(OLD/n)} for n in inputs],
         'original_reducer_source_SHA':sha(mean_source),'original_reducer_AST':mean_ast,'original_check_AST':check_ast,
         'positive_position_count_unit':'time position; descriptive only, not independent statistical samples',
         'extreme_event_definition':'per-doc Parent KL >= nearest-rank95th percentile of384 positions, chosen for posthoc descriptive reporting',
         'clipping_window_source':'actual original per-window/layer counts; not inferred from global totals',
         'documents':docs,'windows':windows})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for d,ax in zip(docs,axes):
        r=[x for x in paired if x['document_index']==d['document_index']]
        ax.plot(positions,[x['cumulative_mean_DeltaInf'] for x in r],label='Parent - NoInflation',color='#087e8b')
        ax.plot(positions,[x['cumulative_mean_DeltaDef'] for x in r],label='Parent - NoDeflation',color='#c44900')
        ax.axhline(0,color='gray',linewidth=.8)
        for t in [16384,24576]:ax.axvline(t,color='gray',linestyle=':',linewidth=.8)
        ax.set(title='Historical pilot, document %d'%d['document_index'],xlabel='Zero-based input token index',ylabel='Cumulative mean paired KL difference')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Posthoc descriptive analysis: positive means lower intervention KL')
    fig.tight_layout();fig.savefig(ROOT/'HISTORICAL_CUMULATIVE_MEAN_EFFECT.png',dpi=160);fig.savefig(ROOT/'HISTORICAL_CUMULATIVE_MEAN_EFFECT.svg');plt.close(fig)
    text=['# 历史2-doc时间机制分析', '', LABEL, '',
          '仅重用旧2304条全词表KL及原窗口/layer诊断；没有任何新GPU forward，不计入新8-doc独立复制。原Primary及数值分辨率保持不变。', '',
          '|文档/干预|窗口|平均KL改善|对总体Delta贡献|正向位置比例|Parent极端事件数|激活率|干预source overflow|',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for x in windows:
        text.append('|%d/%s|[%d,%d)|%.12g|%.12g|%.4f%%|%d|%.4f%%|%.6f%%|'%(x['document_index'],x['condition'],x['window_start'],x['window_end'],x['mean_paired_improvement'],x['contribution_to_overall_Delta'],100*x['positive_position_fraction'],x['extreme_Parent_event_count'],100*x['intervention_activation_fraction'],100*x['intervention_source_overflow_fraction']))
    text += ['', 'Doc1 NoInflation首个8K窗口轻微恶化，之后两个窗口改善；Doc2三个窗口均改善但幅度不同。各窗口均128点评分，对总体Delta的贡献为该窗口paired差值之和/384，负贡献允许保留。', '',
             '两篇NoInflation均有巨大稀疏终点state-MSE改善及充分scale干预激活，同时source overflow/code saturation增加而总体KL下降。该描述不能确认唯一中介、逐token状态因果映射或可训练Loss。状态诊断仅16384/24576/32768三个终点的24层relativeMSE均值。', '',
             '极端事件定义：Parent KL达到该doc384点的nearest-rank95%分位及以上；完整时间位置见HISTORICAL_EXTREME_KL_EVENTS.csv。正负位置比例是描述统计，不将384个时间点当作独立样本。原数据确实保存了每窗口/layer clipping计数，窗口clipping由真实计数汇总。', '',
             '![累计平均改善曲线](HISTORICAL_CUMULATIVE_MEAN_EFFECT.png)', '',
             '384点paired差值与累计曲线共768行，另有12条窗口汇总和绝对paired差值最大的20位置/条件。所有产物绑定旧spec/源文件SHA。']
    with (ROOT/'HISTORICAL_TEMPORAL_ANALYSIS.md').open('x') as f:f.write('\n'.join(text)+'\n')
    print('P0_POSTHOC_CPU_ANALYSIS_COMPLETE',json.dumps(docs,ensure_ascii=False))

if __name__=='__main__':main()
