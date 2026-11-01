"""Posthoc descriptions from real saved scalars; preserve original Primary."""
import csv
import hashlib
import json
import math
from pathlib import Path

D=Path(__file__).resolve().parent
OLD=D.parent/'ling_functional_validation'
spec=json.loads((OLD/'FROZEN_SPEC.json').read_text())
audit=json.loads((D/'EXISTING_RESULTS_ASSET_AUDIT.json').read_text())
positions=spec['primary']['positions']
blocks=[('Early',8192,16384),('Middle',16384,24576),('Late',24576,32768)]


def write_csv(name,rows):
    with (D/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


summaries=[];curves=[];extremes=[];state_rows=[];facts=[]
for entry in spec['documents']:
    doc=entry['document'];a=entry['P1_predicted_winner'];b=entry['L7_predicted_winner']
    values={}
    for r in (a,b):
        p=OLD/'evidence/completed_formal'/f'{doc}__{r}__formal.json'
        expected=next(x for x in audit['old_student_records'] if x['document']==doc and x['rotation']==r)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256']
        v=json.loads(p.read_text());points=v['primary_samples']
        assert len(points)==384 and [x['position'] for x in points]==positions
        assert all(x['vocabulary_dimension']==157184 and math.isfinite(x['KL']) for x in points)
        assert sum(x['KL'] for x in points)/384==v['primary_mean_KL']
        values[r]=v
    xs=[x['KL'] for x in values[a]['primary_samples']]
    ys=[x['KL'] for x in values[b]['primary_samples']]
    ds=[x-y for x,y in zip(xs,ys)]
    original=values[a]['primary_mean_KL']-values[b]['primary_mean_KL']
    cumulative=0.;doc_curves=[]
    for i,(t,x,y,z) in enumerate(zip(positions,xs,ys,ds)):
        cumulative+=z
        row=dict(document=doc,position=t,P1_rotation=a,L7_rotation=b,P1_KL=x,L7_KL=y,
            paired_delta=z,cumulative_mean_delta=cumulative/(i+1),scored_positions_so_far=i+1,
            interpretation='POSTHOC_DESCRIPTIVE_NOT_INDEPENDENT_SAMPLES')
        curves.append(row);doc_curves.append(row)
    assert abs(cumulative/384-original)<1e-14
    doc_blocks=[]
    for name,start,end in blocks:
        indices=[i for i,t in enumerate(positions) if start<=t<end]
        assert len(indices)==128
        delta=sum(ds[i] for i in indices)/128
        row=dict(document=doc,block=name,start=start,end_exclusive=end,scored_positions=128,
            P1_rotation=a,L7_rotation=b,P1_block_mean_KL=sum(xs[i] for i in indices)/128,
            L7_block_mean_KL=sum(ys[i] for i in indices)/128,Delta_block_mean_paired=delta,
            contribution_to_original_384_mean_delta=sum(ds[i] for i in indices)/384,
            negative_point_count=sum(ds[i]<0 for i in indices),positive_point_count=sum(ds[i]>0 for i in indices),
            zero_point_count=sum(ds[i]==0 for i in indices),minimum_paired_delta=min(ds[i] for i in indices),
            maximum_paired_delta=max(ds[i] for i in indices),
            descriptive_mean_direction='P1_lower' if delta<0 else 'L7_lower' if delta>0 else 'equal',
            formal_segment_resolution='NOT_QUALIFIED_384_POINT_THRESHOLD_NOT_APPLIED',
            inferential_status='POSTHOC_DESCRIPTIVE')
        summaries.append(row);doc_blocks.append(row)
    for sign in ('negative','positive'):
        eligible=[x for x in doc_curves if x['paired_delta']<0] if sign=='negative' else [x for x in doc_curves if x['paired_delta']>0]
        ordered=sorted(eligible,key=lambda x:(-abs(x['paired_delta']),x['position']))[:10]
        for rank,row in enumerate(ordered,1):
            extremes.append(dict(document=doc,sign=sign,within_sign_rank_by_magnitude=rank,
                position=row['position'],paired_delta=row['paired_delta'],P1_KL=row['P1_KL'],L7_KL=row['L7_KL'],
                block=next(n for n,s,e in blocks if s<=row['position']<e),
                display_rule='top10_each_sign_posthoc_descriptive_not_a_new_metric'))
    for s in (8192,16383,24575,32767):
        sa=values[a]['state_diagnostics'][str(s)];sb=values[b]['state_diagnostics'][str(s)]
        nearest=min(range(384),key=lambda i:(abs(positions[i]-s),positions[i]))
        adjacent=[x for x in doc_blocks if x['start']-1<=s<x['end_exclusive']]
        state_rows.append(dict(document=doc,state_snapshot_position=s,
            P1_state_MSE=sa['accumulated_error_mse'],L7_state_MSE=sb['accumulated_error_mse'],
            state_MSE_lower='P1' if sa['accumulated_error_mse']<sb['accumulated_error_mse'] else 'L7',
            nearest_actual_KL_position=positions[nearest],time_distance=abs(positions[nearest]-s),
            nearest_actual_paired_KL_delta=ds[nearest],
            adjacent_blocks=';'.join(x['block'] for x in adjacent),
            adjacent_block_mean_deltas=';'.join(str(x['Delta_block_mean_paired']) for x in adjacent),
            causal_interpretation='NOT_IDENTIFIABLE_FROM_4_STATE_SNAPSHOTS'))
    strongest=max(doc_blocks,key=lambda x:abs(x['contribution_to_original_384_mean_delta']))
    facts.append(dict(document=doc,original_primary_delta=original,
        summed_paired_delta_over_384=cumulative/384,
        paired_vs_original_reducer_roundoff=cumulative/384-original,
        largest_absolute_segment_contribution=strongest['block'],blocks=doc_blocks,
        block_mean_sign_reversal=len(set(x['descriptive_mean_direction'] for x in doc_blocks))>1,
        causal_early_damage_dominance='NOT_IDENTIFIABLE',
        descriptive_early_dominance_observed=strongest['block']=='Early',
        late_state_vs_late_KL='P1_state_MSE_lower_at_24575_and_32767_but_L7_Late_KL_lower',
        independent_document_units=1))
write_csv('TEMPORAL_KL_DIAGNOSIS.csv',summaries)
write_csv('PAIRED_KL_TEMPORAL_POINTS.csv',curves)
write_csv('TEMPORAL_KL_EXTREMES.csv',extremes)
distribution=[]
for doc in ('pg19-test-860','pg19-test-5956'):
    for sign in ('negative','positive'):
        rows=[x for x in extremes if x['document']==doc and x['sign']==sign]
        strongest=rows[0]
        distribution.append(dict(document=doc,sign=sign,posthoc_top_count=len(rows),
            Early_count=sum(x['block']=='Early' for x in rows),Middle_count=sum(x['block']=='Middle' for x in rows),
            Late_count=sum(x['block']=='Late' for x in rows),largest_magnitude_position=strongest['position'],
            largest_magnitude_paired_delta=strongest['paired_delta'],
            single_position_contribution_to_original_384_mean=strongest['paired_delta']/384,
            interpretation='POSTHOC_DISPLAY_OF_EXISTING_SCALARS_NO_NEW_FORMAL_METRIC'))
write_csv('TEMPORAL_EXTREME_POSITION_DISTRIBUTION.csv',distribution)
write_csv('STATE_MSE_VS_NEARBY_KL.csv',state_rows)
(D/'TEMPORAL_MECHANISM_FACTS.json').write_text(json.dumps(dict(status='COMPLETE_CPU_ONLY',
    original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',documents=facts,
    formal_secondary_formulas_added=False,no_independent_token_p_value=True),indent=2)+'\n')
lines=['# State MSE 与时间 KL：事后诊断',
    '\n只使用原 V4 四个 state MSE 快照；不新增 E_future、Hidden KL 或任何正式 Secondary。',
    '\n|文档|state 时点|MSE 更低|最近真实 KL 时点|该点 Δ|相邻分段及 mean Δ|',
    '|---|---:|---|---:|---:|---|']
for row in state_rows:
    lines.append(f"|{row['document']}|{row['state_snapshot_position']}|{row['state_MSE_lower']}|{row['nearest_actual_KL_position']}|{row['nearest_actual_paired_KL_delta']:.9g}|{row['adjacent_blocks']}: {row['adjacent_block_mean_deltas']}|")
lines.extend(['\n最近 KL 是已保存评分位置，不插值，也不生成未测位置；完整逐点信息见 PAIRED_KL_TEMPORAL_POINTS.csv。',
    '\n两篇文档在24575和32767的 state MSE均为P1更低，但 Late区间的平均Δ均为正，描述性地偏向L7。较小 state MSE 不保证较小区间 functional KL。',
    '\n860 最大绝对区间贡献来自 Middle；5956来自 Late。原 MIXED不能在描述层面归结为早期差异主导。两篇的分段 KL 方向均发生反转。',
    '\n这些是时间对应性。四个瞬时 state 快照不能代表区间积分损伤；无法据此识别误差方向、历史累积、未来Query或router等具体因果路径。',
    '\n原3.3300064494e-06只用于原384点Primary；128点分段没有独立资格阈值，只报告符号和数值。浮点求和次序差记录在TEMPORAL_MECHANISM_FACTS.json中，不等于FP32 KL误差界。',
    '\n384个位置不是独立文档，不计算逐位置独立样本p-value。'])
(D/'STATE_MSE_VS_KL_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for i,fact in enumerate(facts):
        doc=fact['document'];rows=[x for x in curves if x['document']==doc];x=[r['position'] for r in rows]
        axes[i,0].plot(x,[r['P1_KL'] for r in rows],lw=.8,label=rows[0]['P1_rotation'])
        axes[i,0].plot(x,[r['L7_KL'] for r in rows],lw=.8,label=rows[0]['L7_rotation']);axes[i,0].legend(fontsize=8)
        axes[i,1].plot(x,[r['paired_delta'] for r in rows],lw=.8,color='#7750a1')
        axes[i,2].plot(x,[r['cumulative_mean_delta'] for r in rows],lw=1.5,color='#146c94')
        axes[i,2].axhline(fact['original_primary_delta'],lw=.8,ls='--',color='#555',label='Original Primary delta');axes[i,2].legend(fontsize=8)
        for j in (1,2):axes[i,j].axhline(0,color='#333',lw=.7)
        for j,title in enumerate(('Saved full-vocab KL','Paired delta (P1 - L7)','Cumulative mean paired delta')):
            axes[i,j].set_title(doc+' / '+title,fontsize=10);axes[i,j].set_xlabel('Exact scored token position')
            axes[i,j].set_ylabel('KL' if j==0 else 'Delta');axes[i,j].grid(alpha=.2)
            for boundary in (16384,24576):axes[i,j].axvline(boundary,color='#888',ls=':',lw=.7)
    fig.suptitle('Posthoc temporal diagnosis / 384 correlated positions per document / no token-level inference',fontsize=12)
    fig.savefig(D/'TEMPORAL_KL_TRAJECTORIES.png',dpi=180)
    fig.savefig(D/'TEMPORAL_KL_TRAJECTORIES.svg')
    plt.close(fig)
except ImportError:pass
print(json.dumps(dict(status='COMPLETE_CPU_ONLY',blocks=summaries),ensure_ascii=False))
