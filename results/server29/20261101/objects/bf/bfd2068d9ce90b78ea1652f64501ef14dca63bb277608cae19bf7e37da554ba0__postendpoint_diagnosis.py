"""Existing-artifact-only CPU diagnosis. Never imports a model or invokes a scorer."""
import argparse, collections, csv, hashlib, json, math, statistics, subprocess
from pathlib import Path

TASK = 'QWEN_C6R_C7_CANONICAL60_POSTENDPOINT_DIAGNOSIS_V1'
NA = 'NOT_AVAILABLE_FROM_EXISTING_ARTIFACTS'

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def boolval(v):
    require(v in (True, False, 'True', 'False'), 'Invalid boolean '+str(v))
    return v is True or v == 'True'
def key(r): return r['problem_id'], int(r['seed'])
def sid(k): return f'{k[0]}:seed{k[1]}'
def quantile(xs, q):
    xs=sorted(xs); pos=(len(xs)-1)*q; i=math.floor(pos); j=math.ceil(pos)
    return xs[i]+(xs[j]-xs[i])*(pos-i)
def dist(xs):
    xs=[float(x) for x in xs if x is not None]
    if not xs: return {'n':0, 'mean':None,'median':None,'p25':None,'p75':None,'max':None,'min':None}
    require(all(math.isfinite(x) for x in xs),'nonfinite diagnostic')
    return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),p25=quantile(xs,.25),p75=quantile(xs,.75),max=max(xs),min=min(xs))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--endpoint',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); src=Path(args.endpoint).resolve(); out=Path(args.output).resolve()
    require(out.name==TASK and out.parent==src.parent and not out.exists(),'unsafe or existing output target')
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=src,text=True).strip())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()
    consumed={}; final_manifest=src/'manifests/final_artifact_sha256.txt'
    # Every input in the existing final manifest is checked before and after analysis.
    protected=[]
    for line in final_manifest.read_text().splitlines():
        digest,name=line.split(None,1); path=Path(name.strip()); path=path if path.is_absolute() else src/path
        require(sha(path)==digest,'final manifest mismatch '+str(path)); protected.append((path,digest))
    consumed[str(final_manifest)]=sha(final_manifest)
    def load(rel):
        p=src/rel; consumed[str(p)]=sha(p); return read(p)
    def save(rel,data):
        p=out/rel; p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
    def csvsave(rel,rows):
        p=out/rel; p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest=load('manifests/canonical60_120_raw_outputs.json')
    jobs=load('manifests/canonical60_sample_manifest.json')['samples']
    expected={key(r) for r in jobs}; require(len(jobs)==len(expected)==60,'sample manifest duplicates')
    require({k[1] for k in expected}=={1,2} and len({k[0] for k in expected})==30,'question seed layout')
    gold={key(r):int(r['answer']) for r in load('manifests/canonical60_gold_manifest.json')['records']}
    scorer_identity=load('provenance/frozen_v4_identity.json')
    scorer=Path(scorer_identity['scorer_path']); require(sha(scorer)==scorer_identity['scorer_sha256'],'scorer identity')
    consumed[str(scorer)]=sha(scorer)
    plan=load('configs/canonical60_failure_analysis_plan.json')
    utility=Path(plan['canonical_utility']); require(sha(utility)==plan['canonical_utility_sha256'],'historical metric implementation changed')
    consumed[str(utility)]=sha(utility)
    scores={}; raw={}; raw_inventory=[]; rotation_inventory={}
    for arm in ('C6R','C7'):
        s=load(f'scores/{arm.lower()}_frozen_v4.json'); require(s['scorer_sha256']==sha(scorer),'scored result identity')
        require(s['scorer']==scorer_identity['scorer_version'],'scorer version')
        scores[arm]={key(r):r for r in s['records']}
        require(len(s['records'])==60 and set(scores[arm])==expected,'score alignment')
        recs=[r for r in manifest['records'] if r['condition']==arm]
        require(len(recs)==60 and {key(r) for r in recs}==expected,'raw manifest alignment')
        raw[arm]={}
        for rec in recs:
            p=Path(rec['path']); require(p.resolve().is_relative_to(src/'raw'),'raw outside endpoint')
            require(sha(p)==rec['sha256'],'raw SHA'); consumed[str(p)]=rec['sha256']; r=read(p); k=key(r)
            require(r['condition']==arm and k in expected and not p.stat().st_mode&0o222,'raw identity/immutability')
            require(r['prompt_sha256']==next(j['prompt_sha256'] for j in jobs if key(j)==k),'prompt identity')
            sc=scores[arm][k]; require(sc['output_payload_sha256']==r['output_payload_sha256'],'score payload identity')
            require(int(sc['gold_answer'])==gold[k],'gold alignment')
            require(sc['generated_token_count']==r['generated_token_count']==len(r['generated_token_ids']),'token count')
            require(sc['finish_reason']==r['finish_reason'] and sc['eos_seen']==r['eos_seen'],'termination alignment')
            raw[arm][k]=r
            raw_inventory.append({**rec,'sample_id':sid(k),'model_identity_sha256':r['model_identity_sha256'],'theta_checkpoint_sha256':r['learned_rotation_audit']['theta_checkpoint_sha256']})
        audit=raw[arm][sorted(expected)[0]]['learned_rotation_audit']; rp=Path(audit['rotation_file']); mp=Path(audit['deployment_manifest'])
        require(sha(rp)==audit['rotation_sha256'] and sha(mp)==audit['deployment_manifest_sha256'],'rotation/deployment identity')
        consumed[str(rp)]=sha(rp); consumed[str(mp)]=sha(mp)
        rotation_inventory[arm]=dict(rotation_file=str(rp),rotation_sha256=sha(rp),deployment_manifest=str(mp),deployment_manifest_sha256=sha(mp),deployment_identity=read(mp),theta_checkpoint_sha256=audit['theta_checkpoint_sha256'],checkpoint_identity_basis='frozen export provenance + 60 raw audit records; no checkpoint load/export')
        require(all(r['rotation_sha256']==sha(rp) and r['learned_rotation_audit']['theta_checkpoint_sha256']==audit['theta_checkpoint_sha256'] for r in raw[arm].values()),'within-arm rotation identity')
    bpath=src/'analysis/generation_behavior_per_sample.csv'; consumed[str(bpath)]=sha(bpath)
    behavior={}
    integer=['generated_tokens','committed_claim_count','first_committed_claim','first_committed_claim_char','answer_changes','retractions','oscillation_proxy','committed_aba_count']
    with bpath.open() as f:
        brows=list(csv.DictReader(f))
    require(len(brows)==120,'behavior row count')
    for r in brows:
        for name in integer:r[name]=int(r[name]) if r[name]!='' else None
        r['length_ceiling']=boolval(r['length_ceiling'])
        for name in r:
            if 'fraction' in name or 'ratio' in name:r[name]=float(r[name]) if r[name]!='' else None
        k=key(r); arm=r['condition']; require((arm,k) not in behavior,'duplicate behavior')
        require(r['generated_tokens']==raw[arm][k]['generated_token_count'],'behavior length')
        require(r['oscillation_proxy']==r['answer_changes']+r['committed_aba_count']+r['retractions'],'oscillation definition')
        for n in (4,8,16):require(abs(r[f'repeated_{n}gram_fraction']+r[f'unique_{n}gram_ratio']-1)<1e-12,'ngram complement')
        behavior[arm,k]=r
    primary=load('analysis/primary_c7_vs_c6r.json'); pairs=[]
    for k in sorted(expected):
        a=scores['C6R'][k]; b=scores['C7'][k]; ca=a['v4_correct']; cb=b['v4_correct']
        cat='A' if cb and not ca else 'B' if ca and not cb else 'C' if ca else 'D'
        row=dict(problem_id=k[0],seed=k[1],sample_id=sid(k),pair_category=cat,delta_generation_length=b['generated_token_count']-a['generated_token_count'])
        for arm,s in [('C6R',a),('C7',b)]:
            t=behavior[arm,k]; r=raw[arm][k]
            row.update({arm+'_correct':s['v4_correct'],arm+'_abstain':s['v4_status']=='ABSTAIN',arm+'_abstain_reason':s['v4_abstain_reason'],arm+'_extracted_answer':s['v4_extracted_answer'],arm+'_generation_tokens':s['generated_token_count'],arm+'_length_hit':s['finish_reason']=='length',arm+'_EOS':s['eos_seen'],arm+'_finish_reason':s['finish_reason'],arm+'_problem':r['problem'],arm+'_first_commit_token':NA,arm+'_post_commit_token_count':NA,arm+'_final_retraction_present':NA,arm+'_semantic_answer_flip_present':NA,arm+'_answer_flip_present_proxy':t['answer_changes']>0,arm+'_oscillation_classification':'NONZERO_EXISTING_PROXY' if t['oscillation_proxy']>0 else 'ZERO_EXISTING_PROXY'})
            for name,value in t.items():
                if name not in ('condition','problem_id','seed'):row[arm+'_'+name]=value
        for n in (4,8,16):
            delta=row[f'C7_repeated_{n}gram_fraction']-row[f'C6R_repeated_{n}gram_fraction']; row[f'delta_repeated_{n}gram_fraction']=delta
            row[f'repetition_{n}gram_classification']='HIGHER_THAN_PAIRED_C6R' if delta>0 else 'LOWER_THAN_PAIRED_C6R' if delta<0 else 'EQUAL_TO_PAIRED_C6R'
        pairs.append(row)
    groups={c:[r for r in pairs if r['pair_category']==c] for c in 'ABCD'}
    require([len(groups[c]) for c in 'ABCD']==[3,10,30,17],'pair counts do not reproduce endpoint')
    require(sum(r['C6R_correct'] for r in pairs)==40 and sum(r['C7_correct'] for r in pairs)==33,'correctness counts')
    require({r['sample_id'] for r in groups['A']}==set(primary['rescue_sample_ids']) and {r['sample_id'] for r in groups['B']}==set(primary['regression_sample_ids']),'discordant IDs')
    out.mkdir(); save('provenance/input_artifact_inventory.json',dict(endpoint_root=str(src),repository_root=str(repo),endpoint_commit=commit,C6R_sample_count=60,C7_sample_count=60,paired_sample_ids=[sid(k) for k in sorted(expected)],problem_ids=sorted({k[0] for k in expected}),seed_ids=[1,2],raw_outputs=raw_inventory,rotations=rotation_inventory,scorer_identity=scorer_identity,source_hashes=consumed,model_identity_sha256=sorted({r['model_identity_sha256'] for arm in raw.values() for r in arm.values()})))
    csvsave('analysis/c6r_c7_paired_endpoint_table.csv',pairs)
    taxonomy=dict(definitions={'A':'C7 rescue C6R','B':'C7 regression C6R','C':'both correct','D':'both wrong'},counts={c:len(v) for c,v in groups.items()},sample_ids={c:[r['sample_id'] for r in v] for c,v in groups.items()})
    save('analysis/pair_taxonomy.json',taxonomy); csvsave('analysis/pair_taxonomy.csv',[{k:r[k] for k in ['problem_id','seed','sample_id','pair_category']} for r in pairs])
    ab={arm:{r['sample_id'] for r in pairs if r[arm+'_abstain']} for arm in ('C6R','C7')}
    newab=ab['C7']-ab['C6R']; lostab=ab['C6R']-ab['C7']
    abst=dict(regression_count=10,regression_with_C7_abstain=sum(r['C7_abstain'] for r in groups['B']),regression_without_C7_abstain=sum(not r['C7_abstain'] for r in groups['B']),rescue_with_C6R_abstain=sum(r['C6R_abstain'] for r in groups['A']),C7_unique_abstain_count=len(newab),C6R_unique_abstain_count=len(lostab),C7_unique_abstain_ids=sorted(newab),C6R_unique_abstain_ids=sorted(lostab),shared_abstain_ids=sorted(ab['C7']&ab['C6R']),C7_abstain=9,C6R_abstain=4,net_abstain_increase=5,C7_new_abstain_by_category=dict(collections.Counter(r['pair_category'] for r in pairs if r['sample_id'] in newab)),C7_abstain_reasons=dict(collections.Counter(r['C7_abstain_reason'] for r in pairs if r['C7_abstain'])),note='Net increase 5 is not the number of newly abstaining trajectories: 7 new and 2 resolved.')
    save('analysis/abstain_decomposition.json',abst)
    metrics=['generation_tokens','oscillation_proxy','answer_changes','retractions','committed_aba_count','committed_claim_count','first_committed_claim_char','post_commit_fraction']+[f'{t}_{n}gram_{suffix}' for n in (4,8,16) for t,suffix in [('repeated','fraction'),('unique','ratio')]]
    summaries={}
    for cat,rows in {'ALL':pairs,**groups}.items():
        summaries[cat]=dict(samples=len(rows),conditions={arm:{m:dist([r[arm+'_'+m] for r in rows]) for m in metrics} for arm in ('C6R','C7')},paired_deltas={m:dist([r['C7_'+m]-r['C6R_'+m] for r in rows if r['C7_'+m] is not None and r['C6R_'+m] is not None]) for m in metrics},observed_counts={arm:dict(abstain=sum(r[arm+'_abstain'] for r in rows),EOS=sum(r[arm+'_EOS'] for r in rows),length_hit=sum(r[arm+'_length_hit'] for r in rows),nonzero_oscillation_proxy=sum(r[arm+'_oscillation_proxy']>0 for r in rows),answer_changes_present=sum(r[arm+'_answer_changes']>0 for r in rows),ABA_present=sum(r[arm+'_committed_aba_count']>0 for r in rows),any_retraction=sum(r[arm+'_retractions']>0 for r in rows)) for arm in ('C6R','C7')},C7_higher_than_C6R={m:sum(r['C7_'+m]>r['C6R_'+m] for r in rows if r['C7_'+m] is not None and r['C6R_'+m] is not None) for m in metrics})
    save('analysis/generation_length_analysis.json',{k:{'samples':v['samples'],'C6R':v['conditions']['C6R']['generation_tokens'],'C7':v['conditions']['C7']['generation_tokens'],'delta_length':v['paired_deltas']['generation_tokens'],'counts':v['observed_counts']} for k,v in summaries.items()})
    save('analysis/trajectory_diagnostic_distributions.json',summaries)
    definition=dict(source_csv=str(bpath),source_csv_sha256=sha(bpath),source_implementation=str(utility),source_implementation_sha256=sha(utility),no_scorer_call=True,no_reextraction=True,oscillation_proxy='answer_changes + committed_aba_count + retractions; retractions count all V4 events, including rejected/intermediate events',answer_changes='adjacent changes among committed unrejected normalized integer claims; not semantic reasoning switching',repetition='existing 4/8/16 token ngrams; count duplicates / total ngrams and unique / total',nonzero_indicator='mechanical zero/nonzero description only; no validated failure threshold',repetition_classification='paired relative comparison only; no new n/window/threshold',first_committed_claim='integer answer VALUE, not a position; position is first_committed_claim_char',post_commit_fraction='character fraction, not token fraction',unavailable={n:NA for n in ['first_commit_token','post_commit_token_count','final_retraction_present','semantic_answer_flip','long_repeated_spans']},missing_reason='Existing summaries lack token-to-character offsets, per-event final retraction labels and validated semantic flip/long-span taxonomy; no new extraction or thresholds introduced.')
    save('provenance/diagnostic_definition_reuse.json',definition)
    save('analysis/first_commitment_analysis.json',dict(status='CHARACTER_POSITION_ONLY_TOKEN_POSITION_UNAVAILABLE',definitions=definition,groups={k:{arm:{m:v['conditions'][arm][m] for m in ['committed_claim_count','first_committed_claim_char','post_commit_fraction']} for arm in ('C6R','C7')} for k,v in summaries.items()}))
    save('analysis/reasoning_oscillation_analysis.json',dict(evidence='EXISTING_TEXT_PROXY_ONLY_NOT_CAUSAL',groups={k:dict(counts=v['observed_counts'],paired_deltas={m:v['paired_deltas'][m] for m in ['oscillation_proxy','answer_changes','retractions','committed_aba_count']}) for k,v in summaries.items()}))
    save('analysis/repetition_diagnostics.json',dict(evidence='EXISTING_FROZEN_NGRAM_DEFINITIONS_DESCRIPTIVE',groups={k:dict(conditions={arm:{m:v['conditions'][arm][m] for m in metrics if 'gram' in m} for arm in ('C6R','C7')},paired_deltas={m:v['paired_deltas'][m] for m in metrics if 'gram' in m},higher_counts={m:v['C7_higher_than_C6R'][m] for m in metrics if 'gram' in m}) for k,v in summaries.items()},long_repeated_spans=NA))
    questions=[]
    for problem in sorted({r['problem_id'] for r in pairs}):
        rs=sorted([r for r in pairs if r['problem_id']==problem],key=lambda r:r['seed']); require([r['seed'] for r in rs]==[1,2],'two-seed layout')
        cats=[r['pair_category'] for r in rs]
        questions.append(dict(problem_id=problem,seed1_pair_category=cats[0],seed2_pair_category=cats[1],C6R_correct_seeds=sum(r['C6R_correct'] for r in rs),C7_correct_seeds=sum(r['C7_correct'] for r in rs),regression_seeds=cats.count('B'),rescue_seeds=cats.count('A'),stable_C7_regression=cats==['B','B'],stable_C7_rescue=cats==['A','A'],mixed_seed_question=cats[0]!=cats[1] and any(c in 'AB' for c in cats)))
    cluster=dict(questions=questions,num_questions_with_any_C7_regression=sum(q['regression_seeds']>0 for q in questions),num_questions_with_both_seed_C7_regression=sum(q['stable_C7_regression'] for q in questions),num_questions_with_any_C7_rescue=sum(q['rescue_seeds']>0 for q in questions),num_questions_with_both_seed_C7_rescue=sum(q['stable_C7_rescue'] for q in questions),num_mixed_seed_questions=sum(q['mixed_seed_question'] for q in questions),mixed_definition='different categories across seeds and at least one A or B; includes opposing A/B',num_opposing_rescue_regression_questions=sum({q['seed1_pair_category'],q['seed2_pair_category']}=={'A','B'} for q in questions),stable_regression_ids=[q['problem_id'] for q in questions if q['stable_C7_regression']],stable_rescue_ids=[q['problem_id'] for q in questions if q['stable_C7_rescue']],regressions_per_question={q['problem_id']:q['regression_seeds'] for q in questions if q['regression_seeds']},seed_sensitive_single_regression_questions=[q['problem_id'] for q in questions if q['regression_seeds']==1])
    save('analysis/question_cluster_stability.json',cluster); csvsave('analysis/question_cluster_stability.csv',questions)
    save('analysis/discordance_concentration.json',dict(unique_regression_questions=cluster['num_questions_with_any_C7_regression'],regressions_per_question=cluster['regressions_per_question'],two_seed_regression_questions=cluster['stable_regression_ids'],single_seed_regression_count=len(cluster['seed_sensitive_single_regression_questions']),description='Mixed structure: two stable double-seed regression questions plus six single-seed regression questions.'))
    for cat,name in [('B','regression'),('A','rescue')]:
        save(f'analysis/c7_{name}_mechanism_summary.json',dict(count=len(groups[cat]),rows=groups[cat],aggregate=summaries[cat],interpretation='Observed text/endpoint associations only; proxy positivity is not proof of oscillation or cause.'))
        csvsave(f'analysis/c7_{name}_mechanism_summary.csv',groups[cat])
    protected_sources=load('provenance/protected_artifact_revalidation.json')['files']
    hp=next(Path(r['path']) for r in protected_sources if r['path'].endswith('/frozen_v4_per_sample.csv'))
    hd=next(r['sha256'] for r in protected_sources if r['path']==str(hp)); require(sha(hp)==hd,'historical H SHA')
    consumed[str(hp)]=hd
    with hp.open() as f:hrows=[r for r in csv.DictReader(f) if r['condition']=='QWEN_INT8_C128_KEY_HADAMARD']
    hm={key(r):r for r in hrows}; require(len(hrows)==60 and set(hm)==expected,'H strict sample alignment')
    htable=[]
    for r in pairs:
        k=key(r); h=hm[k]; require(int(h['gold_answer'])==gold[k],'H gold alignment')
        x=r['C6R_correct']; y=boolval(h['v4_correct']); cat='RESCUE' if x and not y else 'REGRESSION' if y and not x else 'BOTH_CORRECT' if x else 'BOTH_WRONG'
        htable.append(dict(problem_id=k[0],seed=k[1],sample_id=sid(k),C6R_correct=x,H_correct=y,pair_category=cat,evidence='HISTORICAL_CONTEXTUAL_ONLY'))
    require(sum(r['H_correct'] for r in htable)==41,'H total')
    hcontext=dict(status='AVAILABLE',evidence='HISTORICAL_CONTEXTUAL_ONLY',source_path=str(hp),source_sha256=hd,counts=dict(collections.Counter(r['pair_category'] for r in htable)),sample_ids={c:[r['sample_id'] for r in htable if r['pair_category']==c] for c in ['RESCUE','REGRESSION','BOTH_CORRECT','BOTH_WRONG']},same_runtime_primary=False)
    save('analysis/c6r_vs_h_contextual_taxonomy.json',hcontext); csvsave('analysis/c6r_vs_h_contextual_taxonomy.csv',htable)
    # No post-hoc statistical tests. Ratings are transparent qualitative triage, not a validated classifier.
    verdict=dict(EXPERIMENT=TASK,EVIDENCE_TYPE='EXISTING_ARTIFACT_ONLY_POSTHOC_DIAGNOSIS',NEW_MODEL_FORWARD='NO',NEW_GPU_MEASUREMENT='NO',NEW_GENERATION=0,NEW_AIME_GENERATION=0,NEW_TRAINING='NO',NEW_CHECKPOINT_SELECTION='NO',NEW_ROTATION_EXPORT='NO',INPUT_ARTIFACT_IDENTITY_GATE='PASS',PAIRED_SAMPLE_ALIGNMENT_GATE='PASS',FROZEN_V4_IDENTITY_GATE='PASS',PRIMARY_PAIR_COUNT_REPRODUCTION_GATE='PASS',C6R_ACCURACY='40/60',C7_ACCURACY='33/60',C7_RESCUE=3,C7_REGRESSION=10,C7_NET=-7,C6R_ABSTAIN=4,C7_ABSTAIN=9,C7_REGRESSION_WITH_ABSTAIN=abst['regression_with_C7_abstain'],C7_REGRESSION_WITH_LENGTH_HIT=sum(r['C7_length_hit'] for r in groups['B']),C7_REGRESSION_WITH_OSCILLATION=sum(r['C7_oscillation_proxy']>0 for r in groups['B']),C7_REGRESSION_WITH_ANSWER_FLIP=sum(r['C7_answer_changes']>0 for r in groups['B']),C7_RESCUE_WITH_C6R_ABSTAIN=abst['rescue_with_C6R_abstain'],C7_UNIQUE_REGRESSION_QUESTIONS=cluster['num_questions_with_any_C7_regression'],C7_BOTH_SEED_REGRESSION_QUESTIONS=cluster['num_questions_with_both_seed_C7_regression'],C7_UNIQUE_RESCUE_QUESTIONS=cluster['num_questions_with_any_C7_rescue'],C7_BOTH_SEED_RESCUE_QUESTIONS=cluster['num_questions_with_both_seed_C7_rescue'],C7_TERMINATION_FAILURE_SIGNAL='PRESENT',C7_TRAJECTORY_INSTABILITY_SIGNAL='PRESENT',C7_QUESTION_LEVEL_FAILURE_STABILITY='PRESENT',C6R_VS_H_CONTEXTUAL_PAIRING='AVAILABLE',PRIMARY_NEXT_STEP_CATEGORY='TRAJECTORY_TERMINATION_DIAGNOSIS_NEXT',RECOMMENDED_NEXT_EXPERIMENT='QWEN_C6R_C7_DISCORDANT_TRAJECTORY_REPLAY_V1',NEW_C8_TRAINING_AUTHORIZED='NO',FULLY_BLINDED_EVALUATION='NO',DO_NOT_RUN_REPLAY='YES',NEW_CONFIRMATORY_TESTS=0,FINAL_ACTION='STOP')
    save('analysis/diagnostic_verdicts.json',dict(verdict=verdict,rationale={'termination':'Abstain signal present, but '+str(abst['regression_with_C7_abstain'])+'/10 regressions abstain; all C7 regressions EOS, no ceiling. Not a majority explanation or hardware termination failure.','trajectory':'Six single-seed regression questions, one opposite A/B question, plus paired repetition changes support a limited follow-up hypothesis; existing oscillation proxy is not a failure classifier.','question':'Two double-seed regression questions are present, but 10 regressions span 8 questions; not predominantly a few stable failures.','next_step':'B: shared-prefix replay hypothesis only. Endpoint artifacts do not contain state/logit/layerwise trajectories; do not choose layer-aware training or new loss.'}))
    controls={c:[r['sample_id'] for r in sorted(groups[c],key=lambda r:(r['problem_id'],r['seed']))[:6]] for c in ('C','D')}
    panel=[dict(sample_id=r['sample_id'],pair_category=r['pair_category'],reference_condition='C7' if r['pair_category']=='A' else 'C6R',selection='ALL_DISCORDANTS' if r['pair_category'] in 'AB' else 'LEXICOGRAPHIC_FIRST_6_WITHIN_CATEGORY') for r in pairs if r['pair_category'] in 'AB' or r['sample_id'] in controls.get(r['pair_category'],[])]
    save('proposals/discordant_replay_panel.json',dict(status='PROPOSAL_ONLY_NOT_RUN',authorization='NONE',samples=25,rows=panel,conditions=['FP','H','C6R','C7'],controls_rule='sort (problem_id ascending, seed ascending); first six both-correct and first six both-wrong; no replay outcomes used',teacher_forcing='reference correct C6R trajectory for B; correct C7 for A; fixed C6R trajectory for C/D controls (D reference is not claimed correct)',future_metrics=['state error','functional error','final hidden error','KL(FP||condition)','target delta NLL','FP-reference top1','logit margin','EOS probability','EOS margin','layerwise/tokenwise errors'],first_divergence_prefix='PROPOSED_NOT_RUN; compare t<first differing generated token, measure distribution before divergence; fixed saved tokens only; context budgets require separate authorization',training_authorized=False))
    save('provenance/limitations.json',dict(FULLY_BLINDED_EVALUATION='NO',reason='User-authorized correctness/historical peeking before completion. Fixed 60/condition completed; does not invalidate raw outputs.',obsolete_endpoint_report_sentence='No historical result reader ran until primary artifacts written is superseded by the two saved protocol amendment JSONs; preserved, not edited.',no_causal_claim=True,no_significance_fishing=True,historical_C6_improvement='NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS',Ling_used_as_Qwen_evidence=False,token_commitment_positions=NA,final_retraction=NA,semantic_answer_flip=NA,oscillation_positive_not_diagnosed_failure=True))
    for p,d in protected:require(sha(p)==d,'source changed during diagnosis '+str(p))
    for p,d in consumed.items():require(sha(p)==d,'external input changed '+p)
    save('provenance/read_only_revalidation.json',dict(status='PASS',endpoint_final_manifest_files_checked=len(protected),all_inputs_unchanged=True,no_original_artifact_written=True,consumed_source_hashes=consumed))
    save('analysis/diagnosis_summary.json',dict(verdict=verdict,taxonomy=taxonomy,abstain=abst,cluster=cluster,historical_H=hcontext,distributions=summaries))
    print(json.dumps(dict(output=str(out),verdict=verdict,abstain=abst,cluster={k:v for k,v in cluster.items() if k!='questions'},regression_summary=summaries['B'],rescue_summary=summaries['A'],all_summary=summaries['ALL']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
