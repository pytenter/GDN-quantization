"""CPU-only immutable 23-panel closure before descriptive category analysis."""
import collections, csv, json, math, statistics, traceback
from pathlib import Path
import c6r_h_diagnosis as S
ROOT=S.ROOT
def med(x):return statistics.median(x) if x else None
def avg(x):return statistics.mean(x) if x else None
def finite(x):
    if isinstance(x,float):S.require(math.isfinite(x),'nonfinite raw metric')
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
def describe(rows,key):
    x=[r[key] for r in rows if r[key] is not None]
    return dict(n=len(x),positive=sum(v>0 for v in x),negative=sum(v<0 for v in x),zero=sum(v==0 for v in x),median=med(x),mean=avg(x),sample_ids_positive=[r['sample_id'] for r in rows if r[key] is not None and r[key]>0],sample_ids_negative=[r['sample_id'] for r in rows if r[key] is not None and r[key]<0])
def region_flags(rows,families,order,control_categories):
    summaries=[];family_results={}
    for family in families:
        for region in order:
            for c in 'BACD':
                rr=[r for r in rows if r['pair_category']==c and r['family']==family and r['region']==region]
                summaries.append(dict(pair_category=c,family=family,region=region,n=len(rr),positive_count=sum(r['delta']>0 for r in rr),negative_count=sum(r['delta']<0 for r in rr),median=med([r['delta'] for r in rr]),mean=avg([r['delta'] for r in rr]),sample_ids_positive=[r['sample_id'] for r in rr if r['delta']>0]))
        group={r['region']:r for r in summaries if r['family']==family and r['pair_category']=='B'}
        worst=max(order,key=lambda n:group[n]['median'])
        b=group[worst];controls={c:next(r for r in summaries if r['family']==family and r['region']==worst and r['pair_category']==c) for c in 'CD'}
        stable=b['n']==6 and b['positive_count']>=5 and b['median']>0
        clean=all(controls[c]['positive_count']<5 for c in control_categories)
        flag='PRESENT' if stable and clean else 'MIXED' if stable else 'NOT_PRESENT'
        family_results[family]=dict(group_worst_region=worst,status=flag,H_WIN_positive=b['positive_count'],H_WIN_median=b['median'],control_positive={c:controls[c]['positive_count'] for c in 'CD'},controls_clean_CD=all(controls[c]['positive_count']<5 for c in 'CD'),exploratory=True)
    flag='PRESENT' if any(x['status']=='PRESENT' for x in family_results.values()) else 'MIXED' if any(x['status']=='MIXED' for x in family_results.values()) else 'NOT_PRESENT'
    return summaries,family_results,flag
def run():
    S.require(not (ROOT/'logs/STOP.json').exists(),'worker STOP no analysis')
    S.verify(S.manifest_rows(ROOT,ROOT/'protocol/PREMEASUREMENT_SHA256SUMS'))
    cfg=S.read(ROOT/'protocol/preregistration.json');panel=S.read(ROOT/'protocol/panel.json')['records'];identity=S.read(ROOT/'provenance/input_identity.json')
    S.require(len(panel)==23 and collections.Counter(r['pair_category'] for r in panel)==dict(A=5,B=6,C=6,D=6),'frozen panel')
    for gpu in range(4):
        w=S.read(ROOT/f'logs/worker{gpu}.complete.json');expected=[r['sample_id'] for r in panel if r['gpu']==gpu]
        S.require(w['status']=='PASS' and w['completed']==w['assigned']==len(expected) and w['measurement_sample_ids']==expected and w['normal_exit'] and w['retries']==0 and w['lm_head_unchanged'],'worker closure')
        S.require((ROOT/f'logs/worker{gpu}.exitcode').read_text().strip()=='0','worker exit nonzero')
    paths=[ROOT/f'raw/{r["panel_rank"]:02d}.json' for r in panel]
    S.require(sorted((ROOT/'raw').glob('*.json'))==paths,'raw count or identity')
    # No partial metric reading. First freeze all 23 closed outputs, then read values.
    rawhash=[dict(path=str(p),sha256=S.sha(p)) for p in paths]
    frozen=S.read(ROOT/'protocol/raw_measurement_freeze.json')
    S.require(frozen['status']=='PASS' and frozen['count']==23 and frozen['files']==rawhash and frozen['aggregation_started'] is False,'existing raw freeze differs')
    S.verify(S.manifest_rows(ROOT,ROOT/'protocol/RAW_SHA256SUMS'))
    S.verify(rawhash)
    modelrows=[];layerrows=[];qdqrows=[];audits=[];paired=[];layerdelta=[];qdqdelta=[];regionrows=[];qdqregion=[];worstrows=[];windowrows=[]
    layerids=sorted(sum(cfg['regions'].values(),[]))
    for p,r in zip(paths,panel):
        x=S.read(p);finite(x)
        S.require(x['status']=='PASS' and x['sample_id']==r['sample_id'] and x['panel_rank']==r['panel_rank'] and x['pair_category']==r['pair_category'] and x['gpu']==r['gpu'] and x['uuid']==r['GPU_uuid'],'raw identity/hardware')
        S.require(x['t_star']==r['t_star'] and x['shared_prefix_sha256']==r['shared_prefix_sha256'] and x['window_positions']==r['window_positions'],'prefix/window identity')
        S.require(x['runtime']==cfg['runtime'] and x['new_free_generation']==x['new_training']==x['scorer_calls']==0,'runtime/forbidden')
        expected={(a,j) for a in cfg['conditions'] for j in r['window_positions']+[r['t_star']]}
        S.require({(v['condition'],v['predicted_generated_token_index']) for v in x['model_metrics']}==expected and len(x['model_metrics'])==len(expected),'model completeness')
        S.require({(v['condition'],v['layer']) for v in x['layer_metrics']}=={(a,l) for a in cfg['conditions'] for l in layerids} and len(x['layer_metrics'])==72,'all24 layer completeness')
        S.require({(v['condition'],v['layer']) for v in x['qdq_metrics']}=={(a,l) for a in ['H','C6R'] for l in layerids} and len(x['qdq_metrics'])==48,'QDQ observation completeness')
        S.require(len(x['condition_audits'])==3 and {a['condition'] for a in x['condition_audits']}==set(cfg['conditions']),'audit arms')
        for a in x['condition_audits']:
            S.require(a['prefix_tokens_consumed']==r['t_star'] and a['quantizer_calls']==(0 if a['condition']=='FP' else r['t_star']),'QDQ call audit')
            if a['condition']!='FP':S.require(a['quantizer_layers']==layerids and a['patch']['calls']==24*(1+r['t_star']) and a['patch']['layers']==layerids,'canonical QDQ/rotation coverage')
            audits.append(dict(sample_id=r['sample_id'],gpu=r['gpu'],**a))
        modelrows+=x['model_metrics'];layerrows+=x['layer_metrics'];qdqrows+=x['qdq_metrics']
        m={(v['condition'],v['predicted_generated_token_index']):v for v in x['model_metrics']};t=r['t_star'];h=m['H',t];c=m['C6R',t]
        for j in r['window_positions']:
            S.require(m['FP',j]['kl_fp_to_condition']==0 and m['FP',j]['margin_degradation']==0,'FP self-reference')
            S.require(m['H',j]['fp_reference_token']==m['C6R',j]['fp_reference_token']==m['FP',j]['fp_reference_token'],'FP reference token')
            windowrows.append(dict(sample_id=r['sample_id'],pair_category=r['pair_category'],token_index=j,DeltaKL=m['C6R',j]['kl_fp_to_condition']-m['H',j]['kl_fp_to_condition'],DeltaMargin=m['C6R',j]['condition_margin_for_fp_reference']-m['H',j]['condition_margin_for_fp_reference']))
        ww=[v for v in windowrows if v['sample_id']==r['sample_id']]
        out=dict(sample_id=r['sample_id'],panel_rank=r['panel_rank'],pair_category=r['pair_category'],category_name=r['category_name'],gpu=r['gpu'],GPU_uuid=r['GPU_uuid'],t_star=t,window_count=len(ww),KL_H=h['kl_fp_to_condition'],KL_C6R=c['kl_fp_to_condition'],DeltaKL=c['kl_fp_to_condition']-h['kl_fp_to_condition'],Margin_H=h['condition_margin_for_fp_reference'],Margin_C6R=c['condition_margin_for_fp_reference'],DeltaMargin=c['condition_margin_for_fp_reference']-h['condition_margin_for_fp_reference'],FH_H=h['Final_Hidden_relative_mse'],FH_C6R=c['Final_Hidden_relative_mse'],DeltaFH=c['Final_Hidden_relative_mse']-h['Final_Hidden_relative_mse'],median_DeltaKL_LAST64=med([v['DeltaKL'] for v in ww]),mean_DeltaKL_LAST64=avg([v['DeltaKL'] for v in ww]),median_DeltaMargin_LAST64=med([v['DeltaMargin'] for v in ww]),mean_DeltaMargin_LAST64=avg([v['DeltaMargin'] for v in ww]),PairMargin_H=h['PairMargin'],PairMargin_C6R=c['PairMargin'],DeltaPairMargin=None if h['PairMargin'] is None else c['PairMargin']-h['PairMargin'],DeltaEOSProb=c['EOS_probability']-h['EOS_probability'])
        paired.append(out)
        lm={(v['condition'],v['layer']):v for v in x['layer_metrics']};qm={(v['condition'],v['layer']):v for v in x['qdq_metrics']}
        for l in layerids:
            region=next(n for n,ls in cfg['regions'].items() if l in ls)
            layerdelta.append(dict(sample_id=r['sample_id'],pair_category=r['pair_category'],layer=l,region=region,DeltaState=lm['C6R',l]['State_relative_l2']-lm['H',l]['State_relative_l2'],DeltaFunctional=lm['C6R',l]['Functional_relative_l2']-lm['H',l]['Functional_relative_l2']))
            qdqdelta.append(dict(sample_id=r['sample_id'],pair_category=r['pair_category'],layer=l,region=region,DeltaQDQ=qm['C6R',l]['post_QDQ_relative_l2']-qm['H',l]['post_QDQ_relative_l2'],QDQ_H=qm['H',l]['post_QDQ_relative_l2'],QDQ_C6R=qm['C6R',l]['post_QDQ_relative_l2']))
        for family,field,data,output in [('State','DeltaState',layerdelta,regionrows),('Functional','DeltaFunctional',layerdelta,regionrows),('QDQ','DeltaQDQ',qdqdelta,qdqregion)]:
            vals={n:med([v[field] for v in data if v['sample_id']==r['sample_id'] and v['region']==n]) for n in cfg['region_order']}
            worst=max(cfg['region_order'],key=lambda n:vals[n])
            worstrows.append(dict(sample_id=r['sample_id'],pair_category=r['pair_category'],family=family,worst_region=worst,worst_penalty=vals[worst],**vals))
            for n,v in vals.items():output.append(dict(sample_id=r['sample_id'],pair_category=r['pair_category'],family=family,region=n,delta=v,positive=v>0,sample_worst_region=worst,evidence='EXPLORATORY_ONLY'))
    keys=['DeltaKL','DeltaMargin','DeltaFH','median_DeltaKL_LAST64','mean_DeltaKL_LAST64','median_DeltaMargin_LAST64','mean_DeltaMargin_LAST64','DeltaPairMargin','DeltaEOSProb']
    groups={c:{k:describe([r for r in paired if r['pair_category']==c],k) for k in keys} for c in 'BACD'};B=groups['B']
    primary=[('H_WIN_PREFIX_KL_SIGNAL','DeltaKL','positive'),('H_WIN_PREFIX_MARGIN_SIGNAL','DeltaMargin','negative'),('H_WIN_FH_SIGNAL','DeltaFH','positive')]
    flags={name:'PRESENT' if B[k][direction]>=5 and (B[k]['median']>0 if direction=='positive' else B[k]['median']<0) else 'NOT_PRESENT' for name,k,direction in primary}
    regionsummary,layerfamilies,layerflag=region_flags(regionrows,['State','Functional'],cfg['region_order'],'C')
    qsummary,qfamilies,qdqflag=region_flags(qdqregion,['QDQ'],cfg['region_order'],'CD')
    active=[]
    for name,k,direction in primary:
        if flags[name]=='PRESENT':active.append(dict(signal=name,family=k,direction=direction,H_WIN=B[k][direction],C=groups['C'][k][direction],D=groups['D'][k][direction]))
    for prefix,families in [('Layer',layerfamilies),('QDQ',qfamilies)]:
        for f,z in families.items():
            if z['status']!='NOT_PRESENT':active.append(dict(signal=prefix,family=f,direction='positive',region=z['group_worst_region'],H_WIN=z['H_WIN_positive'],C=z['control_positive']['C'],D=z['control_positive']['D']))
    specific=[x for x in active if x['C']<5 and x['D']<5]
    specificity='NOT_PRESENT' if not active else 'MIXED' if any(x['C']>=5 or x['D']>=5 for x in active) else 'PRESENT'
    eligiblelayer=any(z['status']=='PRESENT' and z['controls_clean_CD'] for z in layerfamilies.values())
    primaryspecific=any(x['signal'].startswith('H_WIN_') for x in specific)
    nextstep='LAYER_AWARE_C8_HYPOTHESIS_READY' if eligiblelayer else 'READOUT_TRAJECTORY_HYPOTHESIS_NEXT' if primaryspecific and layerflag!='PRESENT' and qdqflag!='PRESENT' else 'NO_STABLE_C6R_H_MECHANISM_STOP_NEW_QWEN_TRAINING'
    categorygap=bool(specific) and not eligiblelayer and not (primaryspecific and layerflag!='PRESENT' and qdqflag!='PRESENT')
    S.verify(identity['protected_files']);S.verify(rawhash)
    S.save('provenance/parent_revalidation.json',dict(status='PASS',files=len(identity['protected_files']),parent_write_operations=0,checked_utc=S.utc(),files_before_after_unchanged=identity['protected_files']),True)
    modelcols=['sample_id','panel_rank','pair_category','gpu','uuid','condition','predicted_generated_token_index','role','kl_fp_to_condition','fp_reference_token','condition_margin_for_fp_reference','delta_nll_fixed_fp_next_token','EOS_probability','EOS_logit','EOS_rank','condition_top1','Final_Hidden_relative_mse','PairMargin']
    S.table('analysis/model_metrics.csv',[{k:r.get(k) for k in modelcols} for r in modelrows],freeze=True)
    layercols=['sample_id','panel_rank','pair_category','condition','layer','region','predicted_generated_token_index','State_relative_l2','Functional_relative_l2']
    S.table('analysis/layer_metrics.csv',[{k:r[k] for k in layercols} for r in layerrows],freeze=True)
    qflat=[]
    for r in qdqrows:
        z={k:r[k] for k in ['sample_id','pair_category','condition','layer','region','predicted_generated_token_index','post_QDQ_relative_l2','pre_QDQ_L2','pre_QDQ_mean_abs','pre_QDQ_max_abs','evidence']}
        for group in ['group_amax','group_scale']:
            for k,v in r[group].items():z[group+'_'+k]=v
        qflat.append(z)
    S.table('analysis/qdq_metrics.csv',qflat,freeze=True)
    for name,rows in [('paired_prefix_summary',paired),('last64_paired_per_token',windowrows),('layer_paired_delta',layerdelta),('qdq_paired_delta',qdqdelta),('layer_region_per_sample',regionrows),('qdq_region_per_sample',qdqregion),('per_sample_worst_regions',worstrows),('layer_region_summary',regionsummary),('qdq_region_summary',qsummary)]:S.table('analysis/'+name+'.csv',rows,freeze=True)
    S.save('analysis/category_summaries.json',groups,True);S.save('analysis/location_and_specificity.json',dict(layer=layerfamilies,qdq=qfamilies,active_signals=active,specific_signals=specific,H_WIN_SIGNAL_SPECIFICITY=specificity,category_gap=categorygap),True)
    hours=sum(a['elapsed_seconds'] for a in audits)/3600
    S.save('analysis/measurement_closure.json',dict(status='PASS',panel_count=23,divergent=len(paired),condition_runs=len(audits),model_metric_rows=len(modelrows),layer_metric_rows=len(layerrows),qdq_metric_rows=len(qdqrows),raw_freeze_sha256=S.sha(ROOT/'protocol/RAW_SHA256SUMS'),all_conditions_same_sample_GPU=True,shared_prefix_only=True,full_vocab_saved=False,measured_condition_gpu_hours=hours,quantizer_calls=sum(a['quantizer_calls'] for a in audits),condition_audits=audits),True)
    verdict=dict(EXPERIMENT=S.TASK,EVIDENCE_TYPE='PROSPECTIVE_POSTENDPOINT_MECHANISM_DIAGNOSIS',NEW_MODEL_FORWARD='YES',NEW_GPU_MEASUREMENT='YES',NEW_FREE_GENERATION=0,NEW_AIME_GENERATION=0,NEW_TRAINING='NO',**{k:v for k,v in identity.items() if k.endswith('_GATE')},PANEL_COUNT=23,MEASURED_SAMPLE_COUNT='23/23',H_WIN_COUNT=6,C6R_WIN_COUNT=5,BOTH_CORRECT_CONTROL_COUNT=6,BOTH_WRONG_CONTROL_COUNT=6,H_WIN_DELTAKL_POSITIVE_COUNT=B['DeltaKL']['positive'],H_WIN_MEDIAN_DELTAKL=B['DeltaKL']['median'],H_WIN_DELTAMARGIN_NEGATIVE_COUNT=B['DeltaMargin']['negative'],H_WIN_MEDIAN_DELTAMARGIN=B['DeltaMargin']['median'],H_WIN_DELTAFH_POSITIVE_COUNT=B['DeltaFH']['positive'],H_WIN_MEDIAN_DELTAFH=B['DeltaFH']['median'],H_WIN_LAST64_KL_POSITIVE_COUNT=B['median_DeltaKL_LAST64']['positive'],H_WIN_LAST64_MARGIN_NEGATIVE_COUNT=B['median_DeltaMargin_LAST64']['negative'],**flags,QWEN_C6R_H_LAYER_LOCATION_SIGNAL=layerflag,QWEN_C6R_H_QDQ_LOCALIZATION_SIGNAL=qdqflag,H_WIN_SIGNAL_SPECIFICITY=specificity,FULL_TRAJECTORY_REPLAY='NOT_RUN',PRIMARY_NEXT_STEP_CATEGORY=None if categorygap else nextstep,NEW_C8_TRAINING_AUTHORIZED='NO',FINAL_ACTION='STOP',CATEGORY_MAPPING_GATE='BLOCKED_USER_DIRECTION_REQUIRED' if categorygap else 'PASS',HISTORICAL_H='HISTORICAL_CONTEXTUAL_ONLY',READOUT_AWARE_METRIC='NOT_AVAILABLE',RAW_M5='NOT_CAPTURED')
    S.save('analysis/final_verdict.json',verdict,True)
    report='```text\n'+'\n'.join(f'{k} = {v}' for k,v in verdict.items())+'\n```\n\n'
    report+='## Scope and interpretation\n\nHistorical H=41/60; C6R=40/60. Exact endpoint pairs: C6R wins5, H wins6, both correct35, both wrong14. Historical H is HISTORICAL_CONTEXTUAL_ONLY, not upgraded to a contemporaneous endpoint. New measurements are controlled FP/H/C6R teacher-forced replay, same sample/prefix/GPU and manual HF BF16 runtime. All23 include11 discordants and mechanically chosen6+6 controls. No p-values, new scoring, generation, training or C7 study. Differences below are C6R−H. Endpoint-selected small-panel evidence is descriptive, not causal.\n\n'
    report+='## Separate categories\n\n| Category | n | ΔKL>0 | median ΔKL | Δmargin<0 | median Δmargin | ΔFH>0 | median ΔFH | LAST64 KL>0 | LAST64 margin<0 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    names=dict(B='H_WIN',A='C6R_WIN',C='BOTH_CORRECT',D='BOTH_WRONG')
    for cat in 'BACD':
        q=groups[cat];report+=f"| {names[cat]} | {q['DeltaKL']['n']} | {q['DeltaKL']['positive']} | {q['DeltaKL']['median']:.9g} | {q['DeltaMargin']['negative']} | {q['DeltaMargin']['median']:.9g} | {q['DeltaFH']['positive']} | {q['DeltaFH']['median']:.9g} | {q['median_DeltaKL_LAST64']['positive']} | {q['median_DeltaMargin_LAST64']['negative']} |\n"
    report+='\nFull exact IDs, means, medians, zeros and signs are in category_summaries.json and paired_prefix_summary.csv. LAST64 uses frozen [max(0,t*−64),t*); insufficient prefixes use all available positions. Margin refers to the FP-preferred token and raw logits, never sampling-warper logits. Secondary PairMargin uses endpoint-success-associated token, NOT ground truth. FH is model-level normalized lm_head input relative MSE, never layerwise. State error uses post-QDQ physical-basis state; the same step logits and Functional output precede its QDQ writeback. QDQ is measured native post-vs-pre and is descriptive.\n\n'
    answers=[
        f"Q1. H_WIN_PREFIX_KL_SIGNAL={flags['H_WIN_PREFIX_KL_SIGNAL']}: ΔKL positive {B['DeltaKL']['positive']}/6; median {B['DeltaKL']['median']:.9g}. LAST64 median ΔKL positive {B['median_DeltaKL_LAST64']['positive']}/6. This does not prove why H wins.",
        f"Q2. H_WIN_PREFIX_MARGIN_SIGNAL={flags['H_WIN_PREFIX_MARGIN_SIGNAL']}: negative {B['DeltaMargin']['negative']}/6; median {B['DeltaMargin']['median']:.9g}; LAST64 negative {B['median_DeltaMargin_LAST64']['negative']}/6.",
        f"Q3. H_WIN_FH_SIGNAL={flags['H_WIN_FH_SIGNAL']}: positive {B['DeltaFH']['positive']}/6; median {B['DeltaFH']['median']:.9g}.",
        f"Q4. Control same-direction counts: both-correct KL {groups['C']['DeltaKL']['positive']}/6, margin {groups['C']['DeltaMargin']['negative']}/6, FH {groups['C']['DeltaFH']['positive']}/6; both-wrong KL {groups['D']['DeltaKL']['positive']}/6, margin {groups['D']['DeltaMargin']['negative']}/6, FH {groups['D']['DeltaFH']['positive']}/6. Specificity={specificity}; all active location control counts in location_and_specificity.json.",
        f"Q5. C6R_WIN reversed directions: KL lower {groups['A']['DeltaKL']['negative']}/5; margin higher {groups['A']['DeltaMargin']['positive']}/5; FH lower {groups['A']['DeltaFH']['negative']}/5. n=5 descriptive only.",
        f"Q6. Layer-location flag={layerflag}, EXPLORATORY_ONLY. Frozen H_WIN group-worst regions: "+json.dumps(layerfamilies,sort_keys=True)+'. Per-sample worst regions and every fixed-region median are reported, not hidden.',
        f"Q7. QDQ localization flag={qdqflag}, DESCRIPTIVE_ONLY. "+json.dumps(qfamilies,sort_keys=True)+'. Native QDQ error, actual amax/scale, pre-magnitude in qdq_metrics.csv; no quantizer copy or M5 ranking.',
        f"Q8. Layer-aware hypothesis eligibility={eligiblelayer}. "+('Qwen-specific exploratory location justifies a separately preregistered matched C6R-control vs C8 hypothesis, not causal attribution.' if eligiblelayer else 'No eligible specific layer evidence for assigning fixed-H layers and learned others.'),
        f"Q9. PRIMARY_NEXT_STEP_CATEGORY={verdict['PRIMARY_NEXT_STEP_CATEGORY']}. "+('Recommend stopping new Qwen rotation training: no stable specific signal under frozen rules.' if nextstep=='NO_STABLE_C6R_H_MECHANISM_STOP_NEW_QWEN_TRAINING' and not categorygap else 'Only a next hypothesis is recommended; no new rotation training is authorized.')]
    report+='## Q1–Q9\n\n'+'\n\n'.join(answers)+'\n\n## Exploratory fixed-region summaries\n\n| Category | Family | Region | median Δ | positive |\n|---|---|---|---:|---:|\n'
    for z in regionsummary+qsummary:report+=f"| {names[z['pair_category']]} | {z['family']} | {z['region']} | {z['median']:.9g} | {z['positive_count']}/{z['n']} |\n"
    report+='\nExact ordinal thirds: '+json.dumps(cfg['regions'],sort_keys=True)+'. Tie order early/middle/late; reduction and sign rules were frozen before GPU measurement. No boundary, threshold, panel or metric adjustment followed observations.\n\n## Existing-output behavior triage\n\n11 discordants ×2 conditions are in existing_output_behavior_triage.csv. All values reuse saved original output length/termination and precomputed frozen claim/repetition definitions; V4 was never called. First committed claim offsets are character positions, not fabricated token positions. Oscillation is a proxy (changes+ABA+retractions), not a new failure threshold. Behavior differences and t* pair margins cannot exclude all downstream reasoning effects.\n\n'
    with (ROOT/'analysis/existing_output_behavior_triage.csv').open() as f:behavior=list(csv.DictReader(f))
    for cat in 'BA':
        for arm in ['H','C6R']:
            rr=[r for r in behavior if r['pair_category']==cat and r['condition']==arm]
            report+=f"{names[cat]} {arm}: n={len(rr)}, abstain={sum(S.boolval(r['abstain']) for r in rr)}, length-hit={sum(S.boolval(r['length_hit']) for r in rr)}, EOS={sum(S.boolval(r['EOS']) for r in rr)}, median length={med([int(r['generated_length']) for r in rr])}, median oscillation={med([float(r['oscillation_proxy']) for r in rr])}, median repeat8={med([float(r['repeated_8gram_fraction']) for r in rr]):.9g}.\n\n"
    report+=f'## Integrity, resource and reproduction\n\n23/23 closure,69 condition runs, {len(layerrows)} layer rows, {len(qdqrows)} QDQ rows, {len(modelrows)} compact model metric rows. Actual timed condition GPU hours={hours:.9g}. All source/parent/model/tokenizer/rotation/scorer hashes unchanged before/after, zero parent writes. Raw frozen SHA256 manifest precedes aggregate reading. Four RTX3090, fixed zero-based rank%4, UUIDs in panel/paired output; no cross-GPU equality claim. lm_head exact CPU checkpoint readback before and hash unchanged afterward. No memory creep over64MiB guard; no scientific retry. TorchDispatchMode observes actual original amax/div outputs without changing returned tensors; CPU exact writeback/audit parity PASS. Original quantize_c128 is called once per decode, not a duplicate quantizer.\n\nREADOUT_AWARE_METRIC=NOT_AVAILABLE as a distinct validated reused metric. Optional LAST64 layer tensors and raw M5 NOT_CAPTURED; all24 t* State/Functional and actual QDQ metrics are present. No full-vocab trajectories stored. Blank PairMargin values in controls are not applicable, not zero. Frozen code and commands are scripts/run_c6r_h_diagnosis.sh, interpreter /data/ydai/miniconda3/envs/bitdecode/bin/python. Existing results must not be overwritten or workers retried. FINAL_ACTION=STOP; full trajectory NOT_RUN; C8 NOT_AUTHORIZED.\n'
    if categorygap:report+='\nCategory mapping gap: a specific QDQ/location signal does not satisfy the supplied A/B definitions. No invented category; STOP and request user direction.\n'
    report+='\n## Preserved CPU aggregation incident 001\n\nOriginal aggregation stopped after raw freeze and before aggregate outputs because a layer-list guard compared sorted runtime layer IDs with dictionary iteration order early/late/middle from sort_keys JSON. All46 non-FP audits have the exact original24 layers and exact calls. CPU fix001 sorts only the expected layer-list identity guard and reuses/verifies the existing raw freeze. Original analyzer, traceback, log and exitcodes are preserved. Region boundaries, metrics, sign rules, measurements and thresholds are unchanged; no GPU replay retry. The corrected source and unified diff are retained in provenance.\n'
    S.write('FINAL_REPORT.md',report,True)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.lock') and '.tmp.' not in p.name and '__pycache__' not in p.parts and p.name not in ['controller.log','analysis.log'])
    S.write('SHA256SUMS',''.join(f'{S.sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),True)
    S.verify(S.manifest_rows(ROOT,ROOT/'SHA256SUMS'))
    print(json.dumps(dict(status='PASS',verdict=verdict,report=str(ROOT/'FINAL_REPORT.md'),manifest_sha256=S.sha(ROOT/'SHA256SUMS'))),flush=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        S.save('logs/analysis_cpu_fix_001.error.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),utc=S.utc(),no_retry=True),True);raise
