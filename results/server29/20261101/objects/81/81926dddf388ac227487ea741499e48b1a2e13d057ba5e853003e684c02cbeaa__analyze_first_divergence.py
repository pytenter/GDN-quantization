"""All-25 raw freeze, then prospective descriptive analysis. CPU only."""
import collections, json, math, statistics, sys, time, traceback
from pathlib import Path
import first_divergence_replay as S

ROOT=S.ROOT
def median(x):return statistics.median(x) if x else None
def mean(x):return statistics.mean(x) if x else None
def finite(x):
    if isinstance(x,float):S.require(math.isfinite(x),'nonfinite raw metric')
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
def describe(rows,key):
    x=[r[key] for r in rows if r[key] is not None]
    return dict(n=len(x),positive=sum(v>0 for v in x),negative=sum(v<0 for v in x),zero=sum(v==0 for v in x),median=median(x),mean=mean(x),sample_ids_positive=[r['sample_id'] for r in rows if r[key] is not None and r[key]>0],sample_ids_negative=[r['sample_id'] for r in rows if r[key] is not None and r[key]<0])
def run():
    S.require(not (ROOT/'logs/STOP.json').exists(),'worker STOP, no analysis')
    S.verify(S.manifest_rows(ROOT,ROOT/'protocol/PREMEASUREMENT_SHA256SUMS'))
    cfg=S.read(ROOT/'protocol/preregistration.json');panel=S.read(ROOT/'protocol/panel.json')['records'];identity=S.read(ROOT/'provenance/input_identity.json')
    for gpu in range(4):
        w=S.read(ROOT/f'logs/worker{gpu}.complete.json')
        S.require(w['status']=='PASS' and w['completed']==sum(r['gpu']==gpu for r in panel) and w['normal_exit'] and w['retries']==0,'worker closure')
        S.require((ROOT/f'logs/worker{gpu}.exitcode').read_text().strip()=='0','worker not exited zero')
    paths=[ROOT/f'raw/{r["panel_rank"]:02d}.json' for r in panel]
    S.require(sorted((ROOT/'raw').glob('*.json'))==paths,'raw count/identity')
    # Freeze the complete raw set before reading or aggregating any values.
    rawhash=[dict(path=str(p),sha256=S.sha(p)) for p in paths]
    S.save('protocol/raw_measurement_freeze.json',dict(status='PASS',count=25,files=rawhash,closed_utc=S.utc(),aggregation_started=False),True)
    S.write('protocol/RAW_SHA256SUMS',''.join(f'{r["sha256"]}  {Path(r["path"]).relative_to(ROOT).as_posix()}\n' for r in rawhash),True)
    S.verify(rawhash)
    modelrows=[];layerrows=[];paired=[];regionrows=[];audits=[]
    for p,r in zip(paths,panel):
        x=S.read(p);finite(x)
        S.require(x['sample_id']==r['sample_id'] and x['panel_rank']==r['panel_rank'] and x['gpu']==r['gpu'] and x['uuid']==r['GPU_uuid'],'sample hardware/identity')
        if r['t_star'] is None:
            S.require(x['status']=='NO_COMPARABLE_DIVERGENCE_ACCOUNTED','no-divergence accounting');continue
        S.require(x['status']=='PASS' and x['t_star']==r['t_star'] and x['shared_prefix_sha256']==r['shared_prefix_sha256'],'prefix identity')
        S.require(x['new_free_generation']==x['new_training']==x['scorer_calls']==0,'forbidden action')
        expect={(arm,j) for arm in cfg['conditions'] for j in r['window_positions']+[r['t_star']]}
        got={(m['condition'],m['predicted_generated_token_index']) for m in x['model_metrics']}
        S.require(got==expect and len(got)==len(x['model_metrics']),'condition/token completeness')
        S.require({(m['condition'],m['layer']) for m in x['layer_metrics']}=={(arm,l) for arm in cfg['conditions'] for l in sum(cfg['regions'].values(),[])} and len(x['layer_metrics'])==96,'24-layer completeness')
        for a in x['condition_audits']:
            S.require(a['prefix_tokens_consumed']==r['t_star'] and a['quantizer_calls']==(0 if a['condition']=='FP' else r['t_star']),'quantizer audit')
            audits.append(dict(sample_id=r['sample_id'],gpu=r['gpu'],condition=a['condition'],**a))
        S.require({a['condition'] for a in x['condition_audits']}==set(cfg['conditions']),'audits incomplete')
        modelrows+=x['model_metrics'];layerrows+=x['layer_metrics']
        m={(v['condition'],v['predicted_generated_token_index']):v for v in x['model_metrics']};t=r['t_star']
        a=m['C6R',t];b=m['C7',t]
        kl=[m['C7',j]['kl_fp_to_condition']-m['C6R',j]['kl_fp_to_condition'] for j in r['window_positions']]
        margin=[m['C7',j]['condition_margin_for_fp_reference']-m['C6R',j]['condition_margin_for_fp_reference'] for j in r['window_positions']]
        out=dict(sample_id=r['sample_id'],panel_rank=r['panel_rank'],pair_category=r['pair_category'],gpu=r['gpu'],GPU_uuid=r['GPU_uuid'],t_star=t,window_count=len(kl),DeltaKL=b['kl_fp_to_condition']-a['kl_fp_to_condition'],DeltaMargin=b['condition_margin_for_fp_reference']-a['condition_margin_for_fp_reference'],median_DeltaKL_LAST64=median(kl),mean_DeltaKL_LAST64=mean(kl),median_DeltaMargin_LAST64=median(margin),mean_DeltaMargin_LAST64=mean(margin),DeltaEOSProb=b['EOS_probability']-a['EOS_probability'],DeltaPairMargin=None if a['PairMargin'] is None else b['PairMargin']-a['PairMargin'],DeltaFinalHidden=b['Final_Hidden_relative_mse']-a['Final_Hidden_relative_mse'])
        paired.append(out)
        lm={(v['condition'],v['layer']):v for v in x['layer_metrics']}
        for family,key in [('State','State_relative_l2'),('Functional','Functional_relative_l2')]:
            vals={region:median([lm['C7',l][key]-lm['C6R',l][key] for l in ids]) for region,ids in cfg['regions'].items()}
            for region,v in vals.items():regionrows.append(dict(sample_id=r['sample_id'],pair_category=r['pair_category'],family=family,region=region,delta=v,positive=v>0,unique_largest=all(v>other for name,other in vals.items() if name!=region)))
    keys=['DeltaKL','DeltaMargin','median_DeltaKL_LAST64','mean_DeltaKL_LAST64','median_DeltaMargin_LAST64','mean_DeltaMargin_LAST64','DeltaEOSProb','DeltaPairMargin','DeltaFinalHidden']
    groups={c:{key:describe([r for r in paired if r['pair_category']==c],key) for key in keys} for c in 'BACD'}
    B=groups['B'];flags={
        'PREFIX_KL_SIGNAL':'PRESENT' if B['DeltaKL']['n']==10 and B['DeltaKL']['positive']>=7 and B['DeltaKL']['median']>0 else 'NOT_PRESENT',
        'PREFIX_MARGIN_SIGNAL':'PRESENT' if B['DeltaMargin']['n']==10 and B['DeltaMargin']['negative']>=7 and B['DeltaMargin']['median']<0 else 'NOT_PRESENT',
        'PRE_DIVERGENCE_ACCUMULATION_SIGNAL':'PRESENT' if B['median_DeltaKL_LAST64']['positive']>=7 or B['median_DeltaMargin_LAST64']['negative']>=7 else 'NOT_PRESENT'}
    active=[]
    if flags['PREFIX_KL_SIGNAL']=='PRESENT':active.append(('DeltaKL','positive'))
    if flags['PREFIX_MARGIN_SIGNAL']=='PRESENT':active.append(('DeltaMargin','negative'))
    if B['median_DeltaKL_LAST64']['positive']>=7:active.append(('median_DeltaKL_LAST64','positive'))
    if B['median_DeltaMargin_LAST64']['negative']>=7:active.append(('median_DeltaMargin_LAST64','negative'))
    specificity='NOT_PRESENT' if not active else 'PRESENT' if all(groups['C'][k][direction]<=3 for k,direction in active) else 'MIXED'
    regionsummary=[];candidates={family:[] for family in ['State','Functional']}
    for family in candidates:
        for region in cfg['regions']:
            for c in 'BACD':
                rr=[r for r in regionrows if r['pair_category']==c and r['family']==family and r['region']==region]
                row=dict(pair_category=c,family=family,region=region,n=len(rr),positive_count=sum(r['positive'] for r in rr),unique_largest_count=sum(r['unique_largest'] for r in rr),positive_and_largest_count=sum(r['positive'] and r['unique_largest'] for r in rr),median=median([r['delta'] for r in rr]),mean=mean([r['delta'] for r in rr]))
                regionsummary.append(row)
                if c=='B' and row['positive_count']>=7 and row['unique_largest_count']>=7 and row['median']>0:candidates[family].append(region)
    def controls_clean(family,region):return all(r['positive_and_largest_count']<=3 for r in regionsummary if r['family']==family and r['region']==region and r['pair_category'] in 'CD')
    same=len(candidates['State'])==len(candidates['Functional'])==1 and candidates['State']==candidates['Functional']
    layer='PRESENT' if same and all(controls_clean(f,candidates[f][0]) for f in candidates) else 'MIXED' if any(candidates.values()) else 'NOT_PRESENT'
    nextstep='FULL_TRAJECTORY_REPLAY_NEXT' if active and specificity=='PRESENT' and not all(groups['D'][k][direction]>=4 for k,direction in active) else 'LAYER_STRUCTURE_FOCUSED_REPLAY_NEXT' if layer=='PRESENT' else 'NO_STABLE_PREFIX_MECHANISM_STOP'
    S.verify(identity['protected_files']);S.verify(rawhash)
    S.save('provenance/read_only_parent_revalidation.json',dict(status='PASS',PARENT_ARTIFACT_IDENTITY_GATE='PASS',files=len(identity['protected_files']),parent_write_operations=0,checked_utc=S.utc()),True)
    # Flat typed CSVs; nested raw audit and Top32 remain in immutable JSON.
    modelfields=['sample_id','panel_rank','pair_category','gpu','uuid','condition','predicted_generated_token_index','role','kl_fp_to_condition','fp_reference_token','condition_margin_for_fp_reference','delta_nll_fixed_fp_next_token','EOS_probability','EOS_logit','EOS_rank','condition_top1','Final_Hidden_relative_mse','PairMargin']
    S.table('analysis/model_metrics.csv',[{k:r.get(k) for k in modelfields} for r in modelrows],freeze=True)
    layerfields=['sample_id','panel_rank','pair_category','condition','layer','region','predicted_generated_token_index','State_relative_l2','Functional_relative_l2']
    S.table('analysis/layer_metrics.csv',[{k:r[k] for k in layerfields} for r in layerrows],freeze=True)
    S.table('analysis/paired_prefix_summary.csv',paired,freeze=True);S.table('analysis/layer_region_per_sample.csv',regionrows,freeze=True);S.table('analysis/layer_region_summary.csv',regionsummary,freeze=True)
    S.save('analysis/category_summaries.json',groups,True);S.save('analysis/measurement_closure.json',dict(status='PASS',panel_count=25,divergent=len(paired),exact_sequence_controls=sum(r['first_divergence']=='NONE' for r in panel),model_metric_rows=len(modelrows),layer_metric_rows=len(layerrows),condition_runs=len(audits),measured_gpu_hours=sum(a['elapsed_seconds'] for a in audits)/3600,raw_freeze_sha256=S.sha(ROOT/'protocol/RAW_SHA256SUMS'),all_conditions_same_sample_GPU=True,shared_prefix_only=True,full_vocab_saved=False),True)
    verdict=dict(EXPERIMENT=S.TASK,EVIDENCE_TYPE='PROSPECTIVE_POSTENDPOINT_MECHANISM_DIAGNOSIS',NEW_MODEL_FORWARD='YES',NEW_GPU_MEASUREMENT='YES',NEW_FREE_GENERATION=0,NEW_AIME_GENERATION=0,NEW_TRAINING='NO',PANEL_IDENTITY_GATE='PASS',PANEL_COUNT=25,PARENT_ARTIFACT_IDENTITY_GATE='PASS',ROTATION_IDENTITY_GATE='PASS',MODEL_IDENTITY_GATE='PASS',MECHANISM_METRIC_IDENTITY_GATE='PASS',MEASURED_SAMPLE_COUNT='25/25',SAMPLES_WITH_FIRST_DIVERGENCE=len(paired),SAMPLES_WITH_NO_DIVERGENCE=sum(r['first_divergence']=='NONE' for r in panel),REGRESSION_COUNT=10,RESCUE_COUNT=3,BOTH_CORRECT_CONTROL_COUNT=6,BOTH_WRONG_CONTROL_COUNT=6,REGRESSION_DELTAKL_POSITIVE_COUNT=B['DeltaKL']['positive'],REGRESSION_MEDIAN_DELTAKL=B['DeltaKL']['median'],REGRESSION_DELTAMARGIN_NEGATIVE_COUNT=B['DeltaMargin']['negative'],REGRESSION_MEDIAN_DELTAMARGIN=B['DeltaMargin']['median'],REGRESSION_LAST64_KL_WORSE_COUNT=B['median_DeltaKL_LAST64']['positive'],REGRESSION_LAST64_MARGIN_WORSE_COUNT=B['median_DeltaMargin_LAST64']['negative'],**flags,REGRESSION_SIGNAL_SPECIFICITY=specificity,QWEN_LAYER_LOCATION_SIGNAL=layer,FULL_TRAJECTORY_REPLAY='NOT_RUN',PRIMARY_NEXT_STEP_CATEGORY=nextstep,NEW_C8_TRAINING_AUTHORIZED='NO',FINAL_ACTION='STOP')
    S.save('analysis/final_verdict.json',verdict,True)
    report='```text\n'+'\n'.join(f'{k} = {v}' for k,v in verdict.items())+'\n```\n\n'
    report+='## Scope and identity\n\nThe parent endpoint remains C6R=40/60, C7=33/60, rescue=3, regression=10, net=-7; unchanged. This 25-panel diagnostic is not a new reasoning endpoint or causal identification. No p-values, sampling, scoring, training or condition-specific prefix. Four RTX3090 GPUs, fixed rank%4 assignment, canonical manual HF BF16 path; UUIDs in panel and paired CSV. Raw model logits are used without generation warpers. Prefill is unquantized; canonical QDQ is applied after decode. Layer state is post-QDQ, while that step’s logits/functional output precede the current writeback. Exact numerical source functions, commits and hashes are frozen in preregistration. Final Hidden is model-level normalized lm_head input relative MSE, not layerwise.\n\n'
    report+='## Separate categories\n\n| Category | n | ΔKL>0 | median ΔKL | mean ΔKL | Δmargin<0 | median Δmargin | mean Δmargin | LAST64 KL worse | LAST64 margin worse |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for c in 'BACD':
        q=groups[c];report+=f"| {c} | {q['DeltaKL']['n']} | {q['DeltaKL']['positive']} | {q['DeltaKL']['median']:.8g} | {q['DeltaKL']['mean']:.8g} | {q['DeltaMargin']['negative']} | {q['DeltaMargin']['median']:.8g} | {q['DeltaMargin']['mean']:.8g} | {q['median_DeltaKL_LAST64']['positive']} | {q['median_DeltaMargin_LAST64']['negative']} |\n"
    report+='\nA=rescue; B=regression; C=both correct; D=both wrong. Complete per-sample values/sign IDs/means are in paired_prefix_summary.csv and category_summaries.json. Hadamard is a mechanism anchor only; all four arms are in model_metrics.csv. Winner tokens are endpoint-success-associated, never ground truth; PairMargin and EOS are secondary, not termination-failure evidence.\n\n'
    questions=[
        f"Q1: First-divergence KL signal {flags['PREFIX_KL_SIGNAL']}; B positive {B['DeltaKL']['positive']}/10, median {B['DeltaKL']['median']:.8g}.",
        f"Q2: FP-preferred margin signal {flags['PREFIX_MARGIN_SIGNAL']}; B negative {B['DeltaMargin']['negative']}/10, median {B['DeltaMargin']['median']:.8g}.",
        f"Q3: LAST64 accumulation flag {flags['PRE_DIVERGENCE_ACCUMULATION_SIGNAL']}; KL worse {B['median_DeltaKL_LAST64']['positive']}/10, margin worse {B['median_DeltaMargin_LAST64']['negative']}/10. Window sign rules describe sustained differences, not causal accumulation or monotonic drift.",
        f"Q4: Both-correct control counts KL {groups['C']['DeltaKL']['positive']}/6; margin {groups['C']['DeltaMargin']['negative']}/6; window KL {groups['C']['median_DeltaKL_LAST64']['positive']}/6; window margin {groups['C']['median_DeltaMargin_LAST64']['negative']}/6. Specificity {specificity} under the frozen conservative operationalization.",
        f"Q5: Both-wrong counts KL {groups['D']['DeltaKL']['positive']}/6; margin {groups['D']['DeltaMargin']['negative']}/6; window KL {groups['D']['median_DeltaKL_LAST64']['positive']}/6; window margin {groups['D']['median_DeltaMargin_LAST64']['negative']}/6. Common difficulty signals cannot establish regression-specific mechanisms.",
        f"Q6: Rescues reversed direction: KL lower {groups['A']['DeltaKL']['negative']}/3; margin higher {groups['A']['DeltaMargin']['positive']}/3; window KL lower {groups['A']['median_DeltaKL_LAST64']['negative']}/3; window margin higher {groups['A']['median_DeltaMargin_LAST64']['positive']}/3. n=3 descriptive only.",
        f"Q7: Qwen layer-location flag {layer}, EXPLORATORY_ONLY; State candidates {candidates['State']}, Functional candidates {candidates['Functional']}. No evidence borrowed from another model and no layer-aware C8 authorization.",
        f"Q8: Recommended category {nextstep}. This is a recommendation only: full trajectory replay NOT_RUN; all computation stops here."
    ]
    report+='## Q1–Q8\n\n'+'\n\n'.join(questions)+'\n\n## Exploratory layer regions\n\n| Category | Family | Region | median C7−C6R error | positive | uniquely largest |\n|---|---|---|---:|---:|---:|\n'
    for z in regionsummary:report+=f"| {z['pair_category']} | {z['family']} | {z['region']} | {z['median']:.8g} | {z['positive_count']}/{z['n']} | {z['unique_largest_count']}/{z['n']} |\n"
    report+='\nRegion boundaries are frozen ordinal thirds, 8 layers each; reduction is per-sample median of paired layer-error differences, followed by category distribution. Neither thresholds nor boundaries were adjusted after measurement.\n\n## Integrity, hardware, reproduction and boundary\n\n25/25 immutable raw files were closed before aggregate analysis. Parent raw/scorer/model/rotation/source hash audits passed before and after. GPUs were not reassigned between conditions. No scientific retries. No huge vocabulary trajectories written. Saved Top32 is only at t*. Blank CSV secondary values mean not applicable, never zero. The selected endpoint-discordant panel cannot prove the mechanism of every regression or prove a Final-Hidden objective caused divergence.\n\nReproduce CPU analysis only in a separate new output directory; existing artifacts are overwrite-protected. The archived scripts record the full run. Command used: `bash scripts/run_first_divergence.sh`; interpreter `/data/ydai/miniconda3/envs/bitdecode/bin/python`. Do not rerun existing workers. `SHA256SUMS` covers all scientific files including frozen raw/protocol/source scripts. FINAL_ACTION=STOP.\n'
    S.write('FINAL_REPORT.md',report,True)
    # Include worker audit evidence and stable logs as well; controller can still append.
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.lock') and '.tmp.' not in p.name and p.name not in ['controller.log','analysis.log'])
    S.write('SHA256SUMS',''.join(f'{S.sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),True)
    S.verify(S.manifest_rows(ROOT,ROOT/'SHA256SUMS'))
    print(json.dumps(dict(status='PASS',verdict=verdict,report=str(ROOT/'FINAL_REPORT.md'),manifest_sha256=S.sha(ROOT/'SHA256SUMS'))),flush=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        S.save('logs/analysis.error.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),utc=S.utc(),no_retry=True),True);raise
