"""Raw120-gated Frozen V4 scoring, single primary test, descriptive closure."""
import argparse, collections, csv, io, statistics
import canonical60_common as A

DIAG=A.SOURCE.parent/'qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/analyze_existing.py'

def table(path,rows):
    A.require(bool(rows),'empty required CSV')
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    A.write_text(path,f.getvalue(),True)

def clustered_ci(keys,deltas):
    import numpy as np
    questions=sorted({k[0] for k in keys})
    A.require(len(questions)==30 and len(keys)==60,'bootstrap question/sample cardinality')
    clusters=np.array([sum(deltas[(q,s)] for s in (1,2)) for q in questions],dtype=np.int64)
    rng=np.random.default_rng(20261003)
    draws=rng.integers(0,30,size=(10000,30))
    effects=clusters[draws].sum(axis=1)/60
    lo,hi=np.percentile(effects,[2.5,97.5],method='linear')
    return dict(method='QUESTION_CLUSTER_BOOTSTRAP',questions=30,seeds_per_question=2,replicates=10000,seed=20261003,rng='numpy.random.default_rng',percentile_method='linear',confidence=0.95,point_estimate=sum(deltas.values())/60,lower=float(lo),upper=float(hi),bootstrap_effects_sha256=A.digest(effects.tolist()))

def label(net,p):
    if net==0:return 'NO_OBSERVED_NET_GAIN'
    if net>0:return 'SUPPORTED' if p<0.05 else 'DIRECTIONALLY_SUPPORTIVE_BUT_UNDERPOWERED'
    return 'CONTRADICTED' if p<0.05 else 'NEGATIVE_DIRECTION_INCONCLUSIVE'

def paired(left,right,name):
    A.require(set(left)==set(right) and len(left)==60,'paired exact60 key mismatch')
    rows=[];counts=collections.Counter()
    for k in sorted(left):
        x,y=left[k],right[k]
        category='BOTH_CORRECT' if x and y else 'BOTH_WRONG' if not x and not y else 'RESCUE' if x else 'REGRESSION'
        counts[category]+=1;rows.append(dict(problem_id=k[0],seed=k[1],left_correct=x,right_correct=y,category=category))
    for category in ('RESCUE','REGRESSION','BOTH_CORRECT','BOTH_WRONG'):counts.setdefault(category,0)
    A.require(sum(counts.values())==60,'paired category count')
    return rows,dict(comparison=name,samples=60,**counts,net=counts['RESCUE']-counts['REGRESSION'],accuracy_difference=(sum(left.values())-sum(right.values()))/60,rescue_sample_ids=[f"{r['problem_id']}:seed{r['seed']}" for r in rows if r['category']=='RESCUE'],regression_sample_ids=[f"{r['problem_id']}:seed{r['seed']}" for r in rows if r['category']=='REGRESSION'])

def self_test():
    pp=A.mod('canonical60_metrics_only_test',A.POST)
    A.require(pp.exact_p(0,0)==1 and pp.exact_p(3,3)==1 and pp.exact_p(6,0)==0.03125,'exact primary test self-test')
    keys=[(f'synthetic_{i:02d}',s) for i in range(30) for s in (1,2)]
    for v in (0,1,-1):
        c=clustered_ci(keys,{k:v for k in keys});A.require(c['lower']==v==c['upper']==c['point_estimate'],'cluster bootstrap self-test')
    A.require(label(0,1)=='NO_OBSERVED_NET_GAIN' and label(1,0.03)=='SUPPORTED' and label(1,1)=='DIRECTIONALLY_SUPPORTIVE_BUT_UNDERPOWERED' and label(-1,0.03)=='CONTRADICTED' and label(-1,1)=='NEGATIVE_DIRECTION_INCONCLUSIVE','mechanical labels')
    a={k:i%3==0 for i,k in enumerate(keys)};b={k:i%4==0 for i,k in enumerate(keys)}
    _,p=paired(a,b,'SYNTHETIC');A.require(p['net']==sum(a.values())-sum(b.values()),'paired count self-test')
    print('SYNTHETIC_STATISTICS_SELF_TEST_PASS_NO_AIME_SCORING',flush=True)

def execute():
    closure=A.read(A.ROOT/'analysis/canonical60_raw_output_closure.json')
    A.require(closure['CANONICAL60_RAW_OUTPUT_CLOSURE_GATE']=='PASS' and closure['SCORING_AUTHORIZED']=='YES' and closure['total']==120,'scoring not authorized')
    A.verify_freeze();A.verify_manifest(A.ROOT/'manifests/canonical60_120_raw_outputs_sha256.txt')
    frozen=A.read(A.ROOT/'manifests/canonical60_120_raw_outputs.json')
    A.require(frozen['count']==120 and len(frozen['records'])==120,'raw manifest count')
    A.require(not list((A.ROOT/'scores').glob('*')),'scoring already exists; no automatic second scoring')
    plan=A.read(A.ROOT/'configs/canonical60_failure_analysis_plan.json')
    A.require(A.sha(A.POST)==plan['canonical_utility_sha256'],'canonical metrics source drift')
    A.require(A.sha(A.SCORER)==A.SCORER_SHA,'Frozen V4 source drift')
    pp=A.mod('canonical60_existing_postprocessor',A.POST);scorer=pp.import_scorer()
    goldpath=A.ROOT/'manifests/canonical60_gold_manifest.json'
    gold={(r['problem_id'],r['seed']):int(r['answer']) for r in A.read(goldpath)['records']}
    jobs=A.samples();keys={(j['problem_id'],j['seed']) for j in jobs}
    A.require(set(gold)==keys and len(keys)==60,'gold/sample key mismatch')
    A.save('provenance/frozen_v4_identity.json',dict(FROZEN_V4_IDENTITY_GATE='PASS',scorer_path=str(A.SCORER),scorer_sha256=A.SCORER_SHA,scorer_version=scorer.SCORER_NAME,gold_manifest_path=str(goldpath),gold_manifest_sha256=A.sha(goldpath),original_dataset_sha256=A.DATASET_SHA,raw120_frozen_before_scorer=True),True)
    raws={};scored={};outcomes={}
    for arm in ('C6R','C7'):
        rows={}
        for j in jobs:
            r=A.read(A.final_path(arm,j));A.validate_raw(r,arm,j);rows[(j['problem_id'],j['seed'])]=r
        raws[arm]=rows
        # Only scoring-local copies receive gold; immutable raw files retain None.
        scoring_rows={k:dict(r,gold_answer=gold[k]) for k,r in rows.items()}
        result=pp.score_records(arm,scoring_rows,scorer)
        A.require(result['samples']==60 and len(result['records'])==60,'scored cardinality')
        A.save(f'scores/{arm.lower()}_frozen_v4.json',result,True);scored[arm]=result
        outcomes[arm]={(r['problem_id'],r['seed']):r['v4_correct'] for r in result['records']}
    rows,primary=paired(outcomes['C7'],outcomes['C6R'],'C7_vs_C6R')
    scoremaps={a:{(r['problem_id'],r['seed']):r for r in scored[a]['records']} for a in scored}
    pairrows=[]
    for r in rows:
        k=(r['problem_id'],r['seed']);x=scoremaps['C6R'][k];y=scoremaps['C7'][k]
        pairrows.append(dict(problem_id=k[0],seed=k[1],gold=gold[k],C6R_extracted_answer=x['v4_extracted_answer'],C6R_correct=x['v4_correct'],C7_extracted_answer=y['v4_extracted_answer'],C7_correct=y['v4_correct'],pair_category=r['category']))
    table('scores/canonical60_pair_table.csv',pairrows)
    primary.update(evidence='PRIMARY',C6R_correct=scored['C6R']['correct'],C7_correct=scored['C7']['correct'],exact_two_sided_p=pp.exact_p(primary['RESCUE'],primary['REGRESSION']),alpha=0.05,multiplicity_adjustment='NONE_SINGLE_CONFIRMATORY_COMPARISON')
    primary['PRIMARY_ENDPOINT_SIGNAL']=label(primary['net'],primary['exact_two_sided_p'])
    ci=clustered_ci(keys,{k:int(outcomes['C7'][k])-int(outcomes['C6R'][k]) for k in keys})
    A.save('analysis/primary_question_cluster_bootstrap.json',ci,True)
    A.save('analysis/primary_c7_vs_c6r.json',primary,True)
    # Exact stage barrier: historical endpoint artifacts are loaded only NOW.
    A.require((A.ROOT/'analysis/primary_c7_vs_c6r.json').exists(),'primary not frozen')
    A.require(A.sha(DIAG)==plan['historical_reader_sha256'],'historical reader identity drift')
    diag=A.mod('canonical60_historical_reader',DIAG);historical=diag.load_endpoint_records()
    histfiles=[diag.BASELINE_CSV,A.OLD/'analysis/C5_scored.json',A.OLD/'analysis/C6_scored.json']
    historical_summary={a:dict(samples=len(v),correct=sum(r['correct'] for r in v.values()),accuracy=sum(r['correct'] for r in v.values())/60,SECONDARY_CONTEXT_ONLY=True) for a,v in historical.items()}
    secondary=[]
    for arm in ('C6R','C7'):
        for h in ('FP','Native','H','C5','C6'):
            A.require(set(historical[h])==keys,'historical exact60 match')
            _,s=paired(outcomes[arm],{k:r['correct'] for k,r in historical[h].items()},arm+'_vs_'+h)
            s.update(evidence='SECONDARY_CONTEXTUAL',left_accuracy=scored[arm]['accuracy'],right_accuracy=historical_summary[h]['accuracy'],NO_SECONDARY_P_VALUES=True)
            secondary.append(s)
    A.save('analysis/secondary_historical_context.json',dict(status='PASS',load_after_primary=True,NO_SECONDARY_P_VALUES=True,historical=historical_summary,comparisons=secondary,sources=[dict(path=str(p),sha256=A.sha(p)) for p in histfiles]),True)
    behavior=[]
    for arm in raws:
        behavior.extend(pp.trajectory_record(arm,r,scorer) for r in raws[arm].values())
    A.require(len(behavior)==120,'behavior cardinality')
    table('analysis/generation_behavior_per_sample.csv',behavior)
    aggregates={}
    for arm in ('C6R','C7'):
        rs=[r for r in behavior if r['condition']==arm];tokens=[r['generated_tokens'] for r in rs]
        summary=dict(samples=60,total_generated_tokens=sum(tokens),mean_generated_tokens=statistics.mean(tokens),median_generated_tokens=statistics.median(tokens),normal_EOS_stop=sum(r['termination_type']=='eos' for r in rs),length_ceiling=sum(r['length_ceiling'] for r in rs),abstain=scored[arm]['abstain'])
        metricnames=[k for k in rs[0] if k not in ('condition','problem_id','seed','generated_tokens','length_ceiling','termination_type')]
        for k in metricnames: summary[k+'_median']=pp.median([r[k] for r in rs])
        summary['answer_changes_total']=sum(r['answer_changes'] for r in rs);summary['retractions_total']=sum(r['retractions'] for r in rs)
        aggregates[arm]=summary
    A.save('analysis/generation_behavior_summary.json',dict(evidence='DESCRIPTIVE_ONLY',canonical_utility=str(A.POST),canonical_utility_sha256=A.sha(A.POST),conditions=aggregates,question_split='NOT_RUN_NO_ESTABLISHED_CANONICAL_CONVENTION',new_model_execution=False),True)
    behavior_maps={a:{(r['problem_id'],r['seed']):r for r in behavior if r['condition']==a} for a in ('C6R','C7')}
    transitions=[]
    for k in sorted(keys):
        x=raws['C6R'][k];y=raws['C7'][k];sx=scoremaps['C6R'][k];sy=scoremaps['C7'][k]
        cat=next(r['pair_category'] for r in pairrows if (r['problem_id'],r['seed'])==k)
        transitions.append(dict(problem_id=k[0],seed=k[1],pair_category=cat,termination_transition=('ceiling' if x['finish_reason']=='length' else 'normal')+'->'+('ceiling' if y['finish_reason']=='length' else 'normal'),answer_transition=('abstain' if sx['v4_status']=='ABSTAIN' else 'committed')+'->'+('abstain' if sy['v4_status']=='ABSTAIN' else 'committed'),C6R_first_committed_claim_char=behavior_maps['C6R'][k]['first_committed_claim_char'],C7_first_committed_claim_char=behavior_maps['C7'][k]['first_committed_claim_char'],rescue_C6R_ceiling=cat=='RESCUE' and x['finish_reason']=='length',regression_C7_ceiling=cat=='REGRESSION' and y['finish_reason']=='length',C7_rescue_H_ceiling=outcomes['C7'][k] and not historical['H'][k]['correct'] and historical['H'][k]['finish_reason']=='length',C6R_rescue_H_ceiling=outcomes['C6R'][k] and not historical['H'][k]['correct'] and historical['H'][k]['finish_reason']=='length'))
    table('analysis/failure_transition_table.csv',transitions)
    counts=dict(termination=dict(collections.Counter(r['termination_transition'] for r in transitions)),answer=dict(collections.Counter(r['answer_transition'] for r in transitions)))
    association={k:sum(r[k] for r in transitions) for k in ('rescue_C6R_ceiling','regression_C7_ceiling','C7_rescue_H_ceiling','C6R_rescue_H_ceiling')}
    A.save('analysis/failure_transition_analysis.json',dict(evidence='DESCRIPTIVE_ASSOCIATION_NOT_CAUSAL',counts=counts,ceiling_associations=association,rows=transitions),True)
    A.verify_manifest(A.ROOT/'manifests/canonical60_120_raw_outputs_sha256.txt')
    protected=A.protected_revalidate()
    A.verify_manifest(A.ROOT/'manifests/canonical60_premeasurement_sha256.txt')
    A.require(A.sha(A.SCORER)==A.SCORER_SHA,'scorer changed after analysis')
    protected.update(PROTECTED_ARTIFACT_REVALIDATION_GATE='PASS',raw120='PASS',gold_manifest_sha256=A.sha(goldpath),scorer_sha256=A.SCORER_SHA)
    A.save('provenance/protected_artifact_revalidation.json',protected,True)
    report(scored,primary,ci,historical_summary,secondary,aggregates,counts,association)
    A.save('analysis/endpoint_completion.json',dict(status='PASS',SCORING_COMPLETE='YES',raw_outputs=120,primary_complete=True,secondary_context_complete=True,descriptive_complete=True,NEW_TRAINING_STARTED='NO',C5R_AIME_GENERATIONS=0,PUSH_PERFORMED='NO',STOP=True,utc=A.utc()),True)
    print('CANONICAL60_ENDPOINT_COMPLETE_STOP',primary['PRIMARY_ENDPOINT_SIGNAL'],flush=True)

def report(scored,p,ci,hist,secondary,behavior,transitions,associations):
    header=dict(EXPERIMENT=A.TASK,PARENT_READINESS='QWEN_GDN_C6R_C7_4X3090_PERSISTENT_WORKER_READINESS_V2',CANONICAL60_PREMEASUREMENT_FREEZE_GATE='PASS',CANONICAL60_RUNTIME_IDENTITY_GATE='PASS',C6R_ROTATION_SHA=A.SHAS['C6R'],C7_ROTATION_SHA=A.SHAS['C7'],CONDITION_ORDER='C6R_THEN_C7',WORKER_ASSIGNMENT='RANK_MOD_4_SAME_GPU_ACROSS_CONDITIONS',C6R_RAW_COMPLETE='60 / 60',C7_RAW_COMPLETE='60 / 60',TOTAL_NEW_AIME_GENERATIONS='120 / 120',C6R_RAW_OUTPUT_FREEZE_GATE='PASS',C7_RAW_OUTPUT_FREEZE_GATE='PASS',CANONICAL60_RAW_OUTPUT_CLOSURE_GATE='PASS',FROZEN_V4_IDENTITY_GATE='PASS',SCORING_COMPLETE='YES',C6R_CORRECT=str(scored['C6R']['correct'])+' / 60',C7_CORRECT=str(scored['C7']['correct'])+' / 60',C7_RESCUE_C6R=p['RESCUE'],C7_REGRESSION_C6R=p['REGRESSION'],BOTH_CORRECT=p['BOTH_CORRECT'],BOTH_WRONG=p['BOTH_WRONG'],C7_MINUS_C6R_ACCURACY=p['accuracy_difference'],PRIMARY_EXACT_P=p['exact_two_sided_p'],PRIMARY_95CI_CLUSTER_BOOTSTRAP=[ci['lower'],ci['upper']],PRIMARY_ENDPOINT_SIGNAL=p['PRIMARY_ENDPOINT_SIGNAL'],PROTECTED_ARTIFACT_REVALIDATION_GATE='PASS',NEW_TRAINING_STARTED='NO',TRAINING_RERUN='NO',C5R_AIME_GENERATIONS=0,PUSH_PERFORMED='NO')
    lines=['# '+A.TASK,'','```text']+[f'{k} = {v}' for k,v in header.items()]+['```','','## PRIMARY: matched objective endpoint','','C6R is the matched clean-A800 Local-Functional control; C7 is the Final-Hidden objective. Frozen exports are reused without re-export/reselection. C5R reproduction remains INCONCLUSIVE and its AIME endpoint is SKIP.','',f"C6R: {scored['C6R']['correct']}/60 ({100*scored['C6R']['accuracy']:.2f}%). C7: {scored['C7']['correct']}/60 ({100*scored['C7']['accuracy']:.2f}%). Difference: {100*p['accuracy_difference']:.2f} percentage points.",'',f"Rescue {p['RESCUE']}; regression {p['REGRESSION']}; both correct {p['BOTH_CORRECT']}; both wrong {p['BOTH_WRONG']}. Two-sided exact discordant binomial p={p['exact_two_sided_p']:.8g}; alpha=0.05; one primary comparison, no multiplicity adjustment.",'',f"95% question-cluster percentile CI: [{100*ci['lower']:.2f}, {100*ci['upper']:.2f}] percentage points. Resample 30 questions carrying both seeds; 10000 draws; seed20261003; numpy default_rng, linear percentiles.",'','Mechanical signal: '+p['PRIMARY_ENDPOINT_SIGNAL'], '', 'Rescue sample IDs: '+', '.join(p['rescue_sample_ids']),'','Regression sample IDs: '+', '.join(p['regression_sample_ids']),'','## SECONDARY: historical context','','No historical result reader ran until the primary artifacts had been written. These are contextual comparisons, with no secondary p-values or confirmatory claims.','','|Condition|Correct/60|Accuracy|','|---|---:|---:|']
    for a,s in hist.items():lines.append(f"|{a}|{s['correct']}/60|{100*s['accuracy']:.2f}%|")
    lines+=['','|Comparison|Difference (pp)|Rescue|Regression|Both correct|Both wrong|','|---|---:|---:|---:|---:|---:|']
    for s in secondary:
        if s['comparison'].endswith('_vs_H'):lines.append(f"|{s['comparison']}|{100*s['accuracy_difference']:.2f}|{s['RESCUE']}|{s['REGRESSION']}|{s['BOTH_CORRECT']}|{s['BOTH_WRONG']}|")
    lines+=['','## DESCRIPTIVE: output behavior','','All metrics reuse the existing canonical trajectory_record utility; no new model execution. Underlying120 rows and60 paired transitions are retained. No causal mechanism claim.','','```json',__import__('json').dumps(dict(behavior=behavior,transitions=transitions,ceiling_associations=associations),indent=2),'```','','## Protocol, provenance, incidents and reproducibility','', 'Premeasurement commit: '+A.read(A.ROOT/'logs/premeasurement_commit.json')['commit'], '', 'Source V2 final commit: '+A.PARENT+'. New branch: '+A.BRANCH+'. Isolated shared local Git clone (remote Git1.8.3.1 lacks worktree support), with no source checkout modification.', '', 'Four independent single-GPU workers, 15 samples each. All C6R workers exit and exact byte-resolution NVML reclaim precede C6R freeze and fresh C7 processes. Same problem/seed uses the same UUID; no dynamic balancing. Manual HF81920 with source-derived sampler, unchanged QDQ/rotation/writeback. Gold-free generation manifest and separate scorer gold manifest. Atomic per-sample final files; all120 final outputs immutable and rehashed. No endpoint-dependent reruns or automatic retries.', '', 'Preparation audit disclosure: the source runtime JSON embeds already-frozen historical aggregate baseline fields and these were printed once during CPU source-only review, before any new output existed. No new endpoint scoring or per-sample historical result loading occurred. No protocol was chosen using new outcomes. See provenance/preparation_audit.json.', '', 'V1 full in-process unload remains FAIL at8519680 allocated bytes, unresolved. V2 persistent worker readiness is a different lifecycle and does not overwrite V1.', '', 'Reproduction entry (single-use; existing errors/outputs must be inspected, not blindly rerun):', '', '```sh',f'cd {A.REPO}',f'bash {A.ROOT}/scripts/canonical60_offline.sh','```','','## Evidence scope and STOP','','AIME26 participated in earlier method development and diagnosis; this is preregistered matched-objective endpoint evidence, not a fully untouched external final benchmark. A fresh reasoning holdout is recommended after a final method freeze. Text failure associations are descriptive, not proof that a mechanism caused accuracy changes.','','Potential follow-up only: QWEN_GDN_C6R_C7_ENDPOINT_MECHANISM_V1. NOT_STARTED. No training, C8, replay, new benchmark or push is authorized by this report. STOP and await explicit user decision.']
    A.write_text('reports/'+A.TASK+'.md','\n'.join(lines)+'\n',True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    elif a.execute:execute()
    else:p.error('explicit --execute or --self-test required')
