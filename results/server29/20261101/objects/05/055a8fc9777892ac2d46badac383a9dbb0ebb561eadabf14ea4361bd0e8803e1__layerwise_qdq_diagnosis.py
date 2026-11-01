"""CPU-only existing-artifact exploratory layerwise QDQ candidate discovery."""
import argparse, ast, collections, csv, hashlib, io, json, math, os, statistics, subprocess, sys, time, traceback
from fractions import Fraction
from pathlib import Path
TASK='QWEN_C6R_H_LAYERWISE_QDQ_FINE_DIAGNOSIS_V1'
PARENT=Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1/experiments/QWEN_C6R_H_DISCORDANT_MECHANISM_DIAGNOSIS_V1')
ROOT=PARENT.parent/TASK
NAMES=dict(B='H_WIN',A='C6R_WIN',C='BOTH_CORRECT',D='BOTH_WRONG')
COUNTS=dict(B=6,A=5,C=6,D=6)
EXPECTED=[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]
C6R_SHA='a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6'
ARCH=Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
def require(v,m):
    if not v:raise RuntimeError(m)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(rel,text):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);require(not p.exists(),'refuse overwrite '+str(p))
    with p.open('x',encoding='utf-8',newline='') as f:f.write(text);f.flush();os.fsync(f.fileno())
    p.chmod(0o444)
def save(rel,obj):write(rel,json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(rel,rows,fields=None):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows);write(rel,s.getvalue())
def csvrows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def manifest(p):
    return [dict(path=str(p.parent/n.strip()),sha256=h) for h,n in [s.split(None,1) for s in p.read_text().splitlines() if s.strip()]]
def verify(entries):
    for x in entries:require(sha(x['path'])==x['sha256'],'SHA drift '+x['path'])
def source_function(path,name):
    s=Path(path).read_text();n=next(n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name);return ast.get_source_segment(s,n)
def describe(v):return dict(n=len(v),positive_count=sum(x>0 for x in v),negative_count=sum(x<0 for x in v),zero_count=sum(x==0 for x in v),median=statistics.median(v),mean=statistics.mean(v),minimum=min(v),maximum=max(v))
def candidate(z):
    return dict(HWIN_count=z['B']['positive_count']>=5,HWIN_median=z['B']['median']>0,C6RWIN_count=z['A']['positive_count']<=2,C6RWIN_median=z['A']['median']<=0,BC_count=z['C']['positive_count']<=3,BC_median=z['C']['median']<=0,BW_count=z['D']['positive_count']<=3,BW_median=z['D']['median']<=0)
def selftest():
    good={c:dict(positive_count=n,median=v) for c,n,v in [('B',5,1.),('A',2,0.),('C',3,0.),('D',3,-1.)]}
    require(all(candidate(good).values()),'candidate boundary test')
    for c,k,v in [('B','positive_count',4),('B','median',0),('A','positive_count',3),('A','median',1e-30),('C','positive_count',4),('C','median',1e-30),('D','positive_count',4),('D','median',1e-30)]:
        bad={x:y.copy() for x,y in good.items()};bad[c][k]=v;require(not all(candidate(bad).values()),'candidate boundary false pass')
    require(sorted([4,1,2],key=lambda l:(-1,l))==[1,2,4],'top3 tie rule')
def prepare():
    require(not ROOT.exists(),'existing experiment, do not duplicate');ROOT.mkdir(parents=True)
    # Freeze the exact rule before parsing any single-layer numerical payload.
    save('protocol/preregistration.json',dict(task=TASK,evidence_type='EXISTING_ARTIFACT_ONLY_EXPLORATORY_CANDIDATE_DISCOVERY',parent=str(PARENT),primary='saved native post-vs-pre QDQ relative_l2; DeltaQDQ=C6R-H',categories={NAMES[c]:COUNTS[c] for c in 'BACD'},strict_rule=dict(HWIN_positive_count_at_least=5,HWIN_median_strictly_positive=True,C6RWIN_positive_count_at_most=2,C6RWIN_median_at_most=0,BC_positive_count_at_most=3,BC_median_at_most=0,BW_positive_count_at_most=3,BW_median_at_most=0),ranking=['specificity_gap descending','median_gap descending','layer_id ascending'],specificity_gap='HWIN_positive/6 - max(C6RWIN_positive/5,BC_positive/6,BW_positive/6); exact rational for ranking',median_gap='median_HWIN - max(median_C6RWIN,median_BC,median_BW)',worst_top3='descending DeltaQDQ, ascending layer_id ties; retain signed values even if all negative',max_prioritized_candidates=4,no_posthoc_threshold=True,case_mapping='strict>=1 ENDPOINT_SPECIFIC; strict0 and any HWIN layer>=5 positive/median>0 DIFFUSE_NOT_ENDPOINT_SPECIFIC under failed separation (report exact failed tests); no enriched HWIN layer NO_STABLE_SINGLE_LAYER_SIGNAL',component_capture='Only existing group_amax/group_scale min/mean/median/max and pre_QDQ_L2/mean_abs/max_abs. Paired differences and category summaries; no invented energy/outlier ratio.',component_consistency='within already selected candidates, descriptive nonzero sign agreement count vs DeltaQDQ; no causal or objective claim, no candidate selection by components',State_Functional='candidate-only contextual summaries after QDQ-only candidate selection',HEAD_AUDIT='static config/source/cache-axis audit only; no model construction/import',RAW_M5='NOT_AVAILABLE unless saved in parent; never cross-rotation ranking',fresh_confirmation='required before C8; independent teacher-forced data, excluding this23, C6R/C7 replay and discovery data; no dataset selection',zero_gpu=True,NEW_MODEL_FORWARD='NO',NEW_GPU_MEASUREMENT='NO',NEW_GENERATION=0,NEW_AIME_GENERATION=0,NEW_TRAINING='NO',NEW_CHECKPOINT_SELECTION='NO',NEW_ROTATION_EXPORT='NO',FINEGRAINED_GPU_CAPTURE='NOT_RUN',NEW_C8_TRAINING_AUTHORIZED='NO',FINAL_ACTION='STOP',script_sha256=sha(__file__),created_utc=utc()))
    write('scripts/layerwise_qdq_diagnosis.py',Path(__file__).read_text())
    save('protocol/rule_freeze.json',dict(status='FROZEN_BEFORE_SINGLE_LAYER_RESULT_READ',rule_sha256=sha(ROOT/'protocol/preregistration.json'),utc=utc()))
    selftest()
    entries=manifest(PARENT/'FINAL_SHA256SUMS');verify(entries)
    require(len(entries)==106,'parent final sealing identity')
    rawfreeze=read(PARENT/'protocol/raw_measurement_freeze.json');require(rawfreeze['status']=='PASS' and rawfreeze['count']==23,'parent raw freeze count');verify(rawfreeze['files'])
    # RAW manifest names are parent-relative, not protocol-relative.
    rawmanifest=[]
    for line in (PARENT/'protocol/RAW_SHA256SUMS').read_text().splitlines():
        h,n=line.split(None,1);rawmanifest.append(dict(path=str(PARENT/n.strip()),sha256=h))
    verify(rawmanifest);require(rawmanifest==rawfreeze['files'],'raw frozen manifests differ')
    panel=read(PARENT/'protocol/frozen_panel.json')['records'];require(len(panel)==23 and collections.Counter(r['pair_category'] for r in panel)==COUNTS,'23 panel categories')
    require(len({r['sample_id'] for r in panel})==23,'panel duplicate')
    cfg=read(PARENT/'protocol/preregistration.json');ident=read(PARENT/'provenance/input_identity.json');inventory=read(PARENT/'provenance/input_artifact_inventory.json')
    require(cfg['conditions']==['FP','H','C6R'] and ident['rotations']['C6R']['sha256']==C6R_SHA and sha(ident['rotations']['C6R']['path'])==C6R_SHA,'condition/rotation identity')
    verify(ident['model_identity']['files'])
    sources=inventory['metric_sources']
    for s in sources:
        require(sha(s['path'])==s['file_sha256'] and hashlib.sha256(source_function(s['path'],s['function']).encode()).hexdigest()==s['function_sha256'],'metric function identity')
    require(any(s['function']=='quantize_c128' and s['path']==ident['runtime']['generation_source']['path'] for s in sources),'native QDQ source provenance')
    require(read(PARENT/'provenance/qdq_observer_cpu_parity.json')['status']=='PASS','actual QDQ observer parity')
    rawlayers=[];available=None
    for r in panel:
        x=read(PARENT/f"raw/{r['panel_rank']:02d}.json")
        require(x['status']=='PASS' and x['sample_id']==r['sample_id'] and x['gpu']==r['gpu'] and x['uuid']==r['GPU_uuid'],'raw identity')
        ls=sorted({v['layer'] for v in x['qdq_metrics']});rawlayers.append(ls)
        require(len(x['qdq_metrics'])==48 and {(v['condition'],v['layer']) for v in x['qdq_metrics']}=={(c,l) for c in ['H','C6R'] for l in ls},'raw QDQ coverage')
        for q in x['qdq_metrics']:
            require(math.isfinite(q['post_QDQ_relative_l2']) and q['post_QDQ_relative_l2']==q['QDQ_error_metrics']['relative_l2'],'QDQ relative_l2 identity')
            require(q['group_shape']==[1,32,1,128] and q['group_amax']['count']==q['group_scale']['count']==4096,'actual group shape/count')
        if available is None:available=sorted(x['qdq_metrics'][0])
    layers=rawlayers[0];require(all(x==layers for x in rawlayers) and layers==EXPECTED and layers==[l for n in cfg['region_order'] for l in cfg['regions'][n]],'actual layer IDs vs parent')
    modelconfig=Path(ident['model_identity']['path'])/'config.json';modelj=read(modelconfig);text=modelj.get('text_config',modelj)
    require([i for i,t in enumerate(text['layer_types']) if t=='linear_attention']==layers,'architecture layer identity')
    require(ARCH.is_file(),'architecture source missing')
    ext=list({str(p):dict(path=str(p),sha256=sha(p)) for p in [Path(s['path']) for s in sources]+[modelconfig,ARCH,Path(ident['rotations']['C6R']['path'])]}.values())
    before=entries+[dict(path=str(PARENT/'FINAL_SHA256SUMS'),sha256=sha(PARENT/'FINAL_SHA256SUMS'))]+ext
    save('provenance/input_artifact_inventory.json',dict(PARENT_ARTIFACT_IDENTITY_GATE='PASS',QDQ_RAW_IDENTITY_GATE='PASS',PANEL_ALIGNMENT_GATE='PASS',LAYER_IDENTITY_GATE='PASS',parent=str(PARENT),parent_final_manifest_sha256=sha(PARENT/'FINAL_SHA256SUMS'),raw_manifest_sha256=sha(PARENT/'protocol/RAW_SHA256SUMS'),panel_manifest_sha256=sha(PARENT/'protocol/frozen_panel.json'),parent_final_verdict=read(PARENT/'analysis/final_verdict.json'),raw_count=23,qdq_rows=1104,layer_ids=layers,panel_ids=[r['sample_id'] for r in panel],category_counts=COUNTS,C6R=ident['rotations']['C6R'],H='same-runtime canonical H mechanism comparator; endpoint HISTORICAL_CONTEXTUAL_ONLY',available_QDQ_schema=available,metric_sources=sources,architecture_source=str(ARCH),architecture_source_sha256=sha(ARCH),model_config_path=str(modelconfig),model_config_sha256=sha(modelconfig),before_files=before))
    save('protocol/frozen_layer_ids.json',dict(status='PASS',layer_count=24,layer_ids=layers,evidence='23 raw QDQ layer sets; actual model config layer_types; parent runtime regions; all exact',regions=cfg['regions']))
    save('provenance/cpu_rule_boundary_tests.json',dict(status='PASS',eight_independent_failed_boundary_tests=True,exact_rational_ranking=True,ascending_layer_ties=True,no_GPU=True))
    write('protocol/PREPARED_SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in sorted(ROOT.rglob('*')) if p.is_file()))
    print(json.dumps(dict(status='PASS',phase='prepare',root=str(ROOT),rule_frozen_before_layer_reads=True,layers=24,panel=23)),flush=True)
def assert_prepared():
    for line in (ROOT/'protocol/PREPARED_SHA256SUMS').read_text().splitlines():
        h,n=line.split(None,1);require(sha(ROOT/n.strip())==h,'prepared drift '+n)
    require(sha(__file__)==read(ROOT/'protocol/preregistration.json')['script_sha256'],'analysis source drift')
def architecture():
    inv=read(ROOT/'provenance/input_artifact_inventory.json');cfg=read(inv['model_config_path']);tc=cfg.get('text_config',cfg)
    source=Path(inv['architecture_source']);text=source.read_text();tree=ast.parse(text)
    wanted=['Qwen3_5GatedDeltaNet','torch_recurrent_gated_delta_rule'];nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    require(len(nodes)==2,'GDN architecture AST evidence')
    snippets={n.name:ast.get_source_segment(text,n) for n in nodes}
    valid=tc['linear_num_key_heads']==16 and tc['linear_num_value_heads']==32 and tc['linear_key_head_dim']==tc['linear_value_head_dim']==128
    valid=valid and all(s in snippets['Qwen3_5GatedDeltaNet'] for s in ['self.num_v_heads = config.linear_num_value_heads','self.num_k_heads = config.linear_num_key_heads','query.repeat_interleave','self.out_proj(core_attn_out)'])
    recurrent=snippets['torch_recurrent_gated_delta_rule'];valid=valid and 'last_recurrent_state' in recurrent and 'torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim' in recurrent
    audit=dict(HEADWISE_DECOMPOSITION_AVAILABLE='YES' if valid else 'NO',inspection='CPU static source and actual saved group shape, no model import',state_shape=[1,32,128,128],axes=['batch','32 value heads','128 key coordinates','128 value coordinates'],key_heads=16,value_heads=32,key_to_value_head_repeat=2,QDQ_group_axes='(batch,value_head,value_coordinate); reduce128 key-coordinate axis (-2);4096 groups per layer',headwise_QDQ_state='structurally valid32 value-head decomposition' if valid else 'not proven',headwise_Functional='NOT_AVAILABLE_FROM_PARENT_POST_OUT_PROJ_MIXED_CAPTURE; do not relabel4096 hidden channels as heads',energy_concentration='NOT_AVAILABLE_FROM_EXISTING_SUMMARY_SCHEMA',source_path=str(source),source_sha256=sha(source),source_ranges={n.name:dict(start=n.lineno,end=n.end_lineno) for n in nodes},config_path=inv['model_config_path'],config_sha256=inv['model_config_sha256'])
    save('provenance/headwise_architecture_audit.json',audit)
    write('provenance/architecture_source_excerpts.txt','\n\n'.join(n+'\n'+s for n,s in snippets.items())+'\n')
    return audit
def analyze():
    assert_prepared();inv=read(ROOT/'provenance/input_artifact_inventory.json');verify(inv['before_files'])
    cfg=read(ROOT/'protocol/preregistration.json');panel=read(PARENT/'protocol/frozen_panel.json')['records'];layers=read(ROOT/'protocol/frozen_layer_ids.json')['layer_ids'];region={l:n for n,ls in read(ROOT/'protocol/frozen_layer_ids.json')['regions'].items() for l in ls}
    raw={r['sample_id']:read(PARENT/f"raw/{r['panel_rank']:02d}.json") for r in panel}
    q={(sid,v['condition'],v['layer']):v for sid,x in raw.items() for v in x['qdq_metrics']}
    # Cross-check the frozen parent CSV, not a replacement metric.
    qc=csvrows(PARENT/'analysis/qdq_metrics.csv');require(len(qc)==1104,'parent CSV rows')
    require({(v['sample_id'],v['condition'],int(v['layer'])) for v in qc}==set(q),'parent CSV exact unique coverage')
    for v in qc:
        qq=q[v['sample_id'],v['condition'],int(v['layer'])];require(float(v['post_QDQ_relative_l2'])==qq['post_QDQ_relative_l2'],'parent CSV/JSON mismatch')
    delta={(sid,l):q[sid,'C6R',l]['post_QDQ_relative_l2']-q[sid,'H',l]['post_QDQ_relative_l2'] for sid in raw for l in layers}
    parentdelta=csvrows(PARENT/'analysis/qdq_paired_delta.csv');require(len(parentdelta)==552,'parent paired rows')
    require({(v['sample_id'],int(v['layer'])) for v in parentdelta}==set(delta),'parent paired exact unique coverage')
    for v in parentdelta:require(float(v['DeltaQDQ'])==delta[v['sample_id'],int(v['layer'])],'paired delta consistency')
    groups={c:[r['sample_id'] for r in panel if r['pair_category']==c] for c in 'BACD'};stats={};rows=[];ranking=[]
    for l in layers:
        z={c:describe([delta[sid,l] for sid in groups[c]]) for c in 'BACD'};stats[l]=z;checks=candidate(z)
        gap=Fraction(z['B']['positive_count'],6)-max(Fraction(z['A']['positive_count'],5),Fraction(z['C']['positive_count'],6),Fraction(z['D']['positive_count'],6))
        mg=z['B']['median']-max(z[c]['median'] for c in 'ACD');strict=all(checks.values())
        row=dict(layer_id=l,region=region[l],QDQ_ENDPOINT_SPECIFIC_LAYER_CANDIDATE='YES' if strict else 'NO',specificity_gap=float(gap),specificity_gap_exact=str(gap),median_gap=mg,HWIN_enriched=checks['HWIN_count'] and checks['HWIN_median'],failed_criteria=';'.join(k for k,v in checks.items() if not v))
        for c in 'BACD':
            for k,v in z[c].items():row[f'{k}_{NAMES[c]}']=v
        for k,v in checks.items():row['criterion_'+k]=v
        rows.append(row);ranking.append((gap,mg,l,row))
    ranking.sort(key=lambda x:(-x[0],-x[1],x[2]));rankrows=[dict(rank=i+1,**x[3]) for i,x in enumerate(ranking)];candidates=[x[2] for x in ranking if x[3]['QDQ_ENDPOINT_SPECIFIC_LAYER_CANDIDATE']=='YES'];prioritized=candidates[:4]
    enriched=[l for l in layers if stats[l]['B']['positive_count']>=5 and stats[l]['B']['median']>0]
    geometry='ENDPOINT_SPECIFIC' if candidates else 'DIFFUSE_NOT_ENDPOINT_SPECIFIC' if enriched else 'NO_STABLE_SINGLE_LAYER_SIGNAL'
    table('analysis/per_layer_qdq_specificity.csv',rows);table('analysis/ranked_exploratory_layers.csv',rankrows)
    paired=[dict(sample_id=r['sample_id'],pair_category=r['pair_category'],category_name=NAMES[r['pair_category']],layer_id=l,region=region[l],QDQError_H=q[r['sample_id'],'H',l]['post_QDQ_relative_l2'],QDQError_C6R=q[r['sample_id'],'C6R',l]['post_QDQ_relative_l2'],DeltaQDQ=delta[r['sample_id'],l]) for r in panel for l in layers]
    table('analysis/per_sample_layer_qdq.csv',paired)
    worst=[];conc={};top3={}
    for c in 'BACD':
        winners=collections.Counter();tops=collections.Counter()
        for sid in groups[c]:
            order=sorted(layers,key=lambda l:(-delta[sid,l],l));winners[order[0]]+=1;tops.update(order[:3]);worst.append(dict(sample_id=sid,category=NAMES[c],worst_layer=order[0],worst_delta=delta[sid,order[0]],top3_layer_ids=order[:3],top3_deltas=[delta[sid,l] for l in order[:3]]))
        conc[NAMES[c]]=dict(n=len(groups[c]),counts={str(l):winners[l] for l in layers},most_common_layers=[dict(layer_id=l,count=n) for l,n in sorted(winners.items(),key=lambda x:(-x[1],x[0]))]);top3[NAMES[c]]=dict(n=len(groups[c]),top3_total=3*len(groups[c]),counts={str(l):tops[l] for l in layers})
    save('analysis/worst_layer_concentration.json',dict(tie_break='layer ID ascending',categories=conc,per_sample=worst));save('analysis/top3_layer_concentration.json',dict(tie_break='layer ID ascending',categories=top3,per_sample=worst))
    # Only actual saved summaries, with precise schema names and units.
    compnames={f'Delta_{g}_{k}':(g,k) for g in ['group_amax','group_scale'] for k in ['minimum','mean','median','maximum']}
    compnames.update({f'Delta_{k}':(k,None) for k in ['pre_QDQ_L2','pre_QDQ_mean_abs','pre_QDQ_max_abs']})
    components=[];componentsummary=[]
    for r in panel:
        sid=r['sample_id']
        for l in layers:
            vals={name:(q[sid,'C6R',l][g][k]-q[sid,'H',l][g][k]) if k else (q[sid,'C6R',l][g]-q[sid,'H',l][g]) for name,(g,k) in compnames.items()}
            require(all(math.isfinite(v) for v in vals.values()),'nonfinite component')
            components.append(dict(sample_id=sid,pair_category=r['pair_category'],category_name=NAMES[r['pair_category']],layer_id=l,DeltaQDQ=delta[sid,l],**vals))
    for l in layers:
        for c in 'BACD':
            rr=[r for r in components if r['layer_id']==l and r['pair_category']==c]
            for name in compnames:
                z=describe([r[name] for r in rr]);agreement=sum(r[name]*r['DeltaQDQ']>0 for r in rr)
                componentsummary.append(dict(layer_id=l,category=NAMES[c],component=name,**z,nonzero_sign_agreement_with_DeltaQDQ=agreement,candidate_layer=l in candidates,evidence='DESCRIPTIVE_ONLY'))
    table('analysis/qdq_component_paired.csv',components);table('analysis/qdq_component_summary.csv',componentsummary)
    contextual=[];contextfields=['layer_id','category','family','n','positive_count','negative_count','zero_count','median','mean','minimum','maximum']
    # State/Functional cannot affect the candidate rule or ranking above.
    lm={(sid,v['condition'],v['layer']):v for sid,x in raw.items() for v in x['layer_metrics']}
    for l in candidates:
        for c in 'BACD':
            for family in ['State','Functional']:
                z=describe([lm[sid,'C6R',l][family+'_relative_l2']-lm[sid,'H',l][family+'_relative_l2'] for sid in groups[c]])
                contextual.append(dict(layer_id=l,category=NAMES[c],family=family,**z))
    table('analysis/candidate_state_functional_context.csv',contextual,fields=contextfields)
    head=architecture();top=ranking[0][2];z=stats[top]
    verdict=dict(EXPERIMENT=TASK,EVIDENCE_TYPE='EXISTING_ARTIFACT_ONLY_EXPLORATORY_CANDIDATE_DISCOVERY',NEW_MODEL_FORWARD='NO',NEW_GPU_MEASUREMENT='NO',NEW_GENERATION=0,NEW_AIME_GENERATION=0,NEW_TRAINING='NO',NEW_CHECKPOINT_SELECTION='NO',NEW_ROTATION_EXPORT='NO',**{k:inv[k] for k in ['PARENT_ARTIFACT_IDENTITY_GATE','QDQ_RAW_IDENTITY_GATE','PANEL_ALIGNMENT_GATE','LAYER_IDENTITY_GATE']},LAYER_COUNT=24,PANEL_COUNT=23,STRICT_CANDIDATE_LAYER_COUNT=len(candidates),STRICT_CANDIDATE_LAYER_IDS=sorted(candidates),PRIORITIZED_CANDIDATE_LAYERS=prioritized,QWEN_SINGLE_LAYER_QDQ_SPECIFICITY_SIGNAL='PRESENT' if candidates else 'NOT_PRESENT',QWEN_QDQ_GEOMETRY_SIGNAL=geometry,TOP_SPECIFICITY_LAYER=top,TOP_SPECIFICITY_GAP=float(ranking[0][0]),TOP_SPECIFICITY_GAP_EXACT=str(ranking[0][0]),TOP_LAYER_HWIN_POSITIVE=f"{z['B']['positive_count']}/6",TOP_LAYER_C6RWIN_POSITIVE=f"{z['A']['positive_count']}/5",TOP_LAYER_BOTH_CORRECT_POSITIVE=f"{z['C']['positive_count']}/6",TOP_LAYER_BOTH_WRONG_POSITIVE=f"{z['D']['positive_count']}/6",HEADWISE_DECOMPOSITION_AVAILABLE=head['HEADWISE_DECOMPOSITION_AVAILABLE'],FINEGRAINED_GPU_CAPTURE='NOT_RUN',FRESH_CONFIRM_REQUIRED='YES',PRIMARY_NEXT_STEP_CATEGORY='QDQ_FINEGRAINED_CAPTURE_NEXT' if candidates else 'STOP_QDQ_INTERVENTION',NEW_C8_TRAINING_AUTHORIZED='NO',RAW_M5='NOT_AVAILABLE',FINAL_ACTION='STOP')
    save('analysis/final_verdict.json',verdict);save('analysis/candidate_selection.json',dict(strict_candidate_layer_ids=sorted(candidates),ranked_candidates=candidates,prioritized=prioritized,all_HWIN_enriched_layers=enriched,ranking='specificity_gap,median_gap,layer_id',per_layer_criteria=rows))
    if candidates:
        save('proposals/QWEN_C6R_H_QDQ_FINEGRAINED_CAPTURE_V1.json',dict(status='PROPOSAL_ONLY_NOT_AUTHORIZED_NOT_RUN',candidate_layers=prioritized,candidate_ranking=[r for r in rankrows if r['layer_id'] in candidates],panel='same frozen23 endpoint-selected samples',panel_path=str(PARENT/'protocol/frozen_panel.json'),panel_sha256=sha(PARENT/'protocol/frozen_panel.json'),conditions=['H','C6R'],FP='only if needed for saved-definition State reference, not needed for native QDQ error',same_GPU_pairing='original physical UUID/rank%4; H/C6R same sample sameGPU',replay='exact same frozen shared prefix[0:t*); t* snapshot only; no condition-specific prefix or sampling',metric_source=inv['metric_sources'],priorities=['C128 group-wise','optional true32 value-head QDQ/state'],group_axes=head['QDQ_group_axes'],group_metrics=['existing pre-QDQ magnitude summaries','actual amax','actual scale','post-vs-pre absolute_l2','existing relative_l2 applied to exact128-key group'],headwise=head,Functional_headwise='NOT_PROPOSED: parent output capture mixes heads through out_proj',no_free_generation=True,no_AIME_generation=True,no_training=True,FRESH_CONFIRM_REQUIRED='YES',future_confirmation='independent teacher-forced data; exclude current23, C6R/C7 replay, discovery data; no dataset chosen',evidence_chain=['discovery','freeze location+metric+direction','independent fresh confirmation','only then separately consider C8']))
    figures(rows,layers)
    verify(inv['before_files']);save('provenance/parent_revalidation.json',dict(status='PASS',before_after_unchanged=True,parent_writes=0,files=inv['before_files'],checked_utc=utc()))
    report(verdict,rows,rankrows,stats,enriched,conc,top3,componentsummary,contextual,head)
    print(json.dumps(dict(status='PASS',verdict=verdict,report=str(ROOT/'FINAL_REPORT.md'))),flush=True)
def figures(rows,layers):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    out=ROOT/'figures';out.mkdir(exist_ok=True);x=list(range(24));colors=['#d55e00','#0072b2','#009e73','#777777'];styles=['-','--','-.',':']
    for number,field,title,ylabel in [(1,'median','Per-layer native QDQ error difference','Median C6R - H relative L2'),(2,'positive_count','Per-layer positive DeltaQDQ fractions','Positive fraction')]:
        fig,ax=plt.subplots(figsize=(11,4.6))
        for c,color,style in zip('BACD',colors,styles):
            y=[r[field+'_'+NAMES[c]]/(COUNTS[c] if field=='positive_count' else 1) for r in rows]
            ax.plot(x,y,linestyle=style,marker='o',markersize=3,label=f'{NAMES[c]} (n={COUNTS[c]})',color=color)
        ax.set_xticks(x,labels=layers);ax.set_xlabel('Recurrent layer ID (ordinal positions)');ax.set_ylabel(ylabel);ax.set_title(title);ax.axhline(0,color='black',lw=.7);ax.grid(axis='y',alpha=.2);ax.legend(ncol=2,fontsize=9)
        if number==1:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        else:ax.set_ylim(-.03,1.03)
        fig.tight_layout();p=out/f'figure{number}.png';require(not p.exists(),'figure overwrite');fig.savefig(p,dpi=170);plt.close(fig);p.chmod(0o444)
    fig,ax=plt.subplots(figsize=(11,4.6));bars=ax.bar(x,[r['specificity_gap'] for r in rows],color='#577890')
    for bar,r in zip(bars,rows):
        if r['QDQ_ENDPOINT_SPECIFIC_LAYER_CANDIDATE']=='YES':bar.set_edgecolor('black');bar.set_linewidth(2)
    ax.set_xticks(x,labels=layers);ax.set_xlabel('Recurrent layer ID (ordinal positions)');ax.set_ylabel('H-win sign rate minus maximum other-group rate');ax.set_title('Exploratory specificity gap (outline: strict candidates)');ax.axhline(0,color='black',lw=.7);ax.grid(axis='y',alpha=.2);fig.tight_layout();p=out/'figure3.png';require(not p.exists(),'figure overwrite');fig.savefig(p,dpi=170);plt.close(fig);p.chmod(0o444)
def report(v,rows,rankrows,stats,enriched,conc,top3,components,context,head):
    text='```text\n'+'\n'.join(f'{k} = {x}' for k,x in v.items())+'\n```\n\n'
    text+='## Scope and frozen method\n\nCPU-only analysis of immutable parent23×24 native QDQ measurements. No model construction, torch import, new forward, GPU measurement, generation, scorer call, training, checkpoint selection or export. Parent current-runtime H is a mechanism comparator; historical H endpoint remains HISTORICAL_CONTEXTUAL_ONLY. Strict eight-test rule was frozen before numerical layer reads in protocol/preregistration.json; all24 layers tested, no post-hoc thresholds or region shifts. DeltaQDQ is the original native post-vs-pre relative L2 C6R minus H, not State/Functional/M5. Counts are exact, medians use all signed values including zeros. Four endpoint categories are never pooled for inference. All results are EXPLORATORY_CANDIDATE_DISCOVERY, no p-values or causal claim.\n\n'
    text+='## All24 layers\n\n| Layer | H_WIN +/6 | H median | C6R_WIN +/5 | C6R median | BC +/6 | BC median | BW +/6 | BW median | specificity gap | Strict |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n'
    for r in rows:
        text+=f"| {r['layer_id']} | {r['positive_count_H_WIN']} | {r['median_H_WIN']:.8g} | {r['positive_count_C6R_WIN']} | {r['median_C6R_WIN']:.8g} | {r['positive_count_BOTH_CORRECT']} | {r['median_BOTH_CORRECT']:.8g} | {r['positive_count_BOTH_WRONG']} | {r['median_BOTH_WRONG']:.8g} | {r['specificity_gap']:.6g} | {r['QDQ_ENDPOINT_SPECIFIC_LAYER_CANDIDATE']} |\n"
    text+='\nPer-layer means/min/max, all eight pass/fail checks, exact rational specificity gaps and median gaps are in per_layer_qdq_specificity.csv. ranked_exploratory_layers.csv ranks every layer without promoting failed candidates. Worst and top3 retain negative values when necessary; layer-ID ascending resolves ties.\n\n'
    text+='## Q1–Q8\n\n'
    early=read(ROOT/'protocol/frozen_layer_ids.json')['regions']['early'];earlyen=[l for l in enriched if l in early]
    text+=f'Q1. Early region has {len(earlyen)} H_WIN-enriched layers: {earlyen}; the full24 H_WIN-enriched set is {enriched}. No additive attribution is inferred: the parent region statistic is a median across layers, so it cannot be decomposed as an error sum. Layer medians, worst-layer and top3 counts identify concentration descriptively.\n\n'
    text+=f"Q2. Strict endpoint-specific candidate count={v['STRICT_CANDIDATE_LAYER_COUNT']}, IDs={v['STRICT_CANDIDATE_LAYER_IDS']}; prioritized={v['PRIORITIZED_CANDIDATE_LAYERS']}. All eight H_WIN enrichment and three-category separation constraints must pass simultaneously.\n\n"
    text+='Q3. Worst-layer distributions: '+json.dumps({k:z['most_common_layers'] for k,z in conc.items()},sort_keys=True)+'. Exact sample IDs/top3 deltas and all-layer counts are saved in the concentration JSON files. This is not a concentration significance test.\n\n'
    if v['STRICT_CANDIDATE_LAYER_COUNT']:
        text+='Q4. Candidate-only paired components are reported below. Sign agreement indicates descriptive consistency, not mechanism identification. amax and scale are algebraically linked by the existing canonical path and are not independent evidence. Energy/outlier concentration is NOT_AVAILABLE; no ratio was invented.\n\n'
        for l in v['STRICT_CANDIDATE_LAYER_IDS']:
            for z in components:
                if z['layer_id']==l and z['category']=='H_WIN':text+=f"Layer{l} {z['component']}: positive {z['positive_count']}/6, median {z['median']:.9g}, same nonzero sign with DeltaQDQ {z['nonzero_sign_agreement_with_DeltaQDQ']}/6.\n\n"
        text+='Q5. Candidate contextual State/Functional summaries (never used for candidate selection):\n\n| Layer | Category | Family | Positive/n | Median delta |\n|---|---|---|---:|---:|\n'
        for z in context:text+=f"| {z['layer_id']} | {z['category']} | {z['family']} | {z['positive_count']}/{z['n']} | {z['median']:.9g} |\n"
    else:text+='Q4. No strict candidate, so a candidate-specific component explanation is NOT_APPLICABLE. Actual saved amax/scale/pre-magnitude differences are retained for every layer and each category, but not used to rescue failed candidates. amax/scale are linked; energy concentration is not available.\n\nQ5. No strict QDQ candidate, so candidate State/Functional contextual table is empty with explicit headers; not zero error and not evidence for intervention.\n\n'
    text+=f"\nQ6. Geometry verdict={v['QWEN_QDQ_GEOMETRY_SIGNAL']}. "+('At least one layer separates the selected endpoint categories under the frozen discovery rule; fresh confirmation is still absent.' if v['STRICT_CANDIDATE_LAYER_COUNT'] else 'The region signal does not resolve into an endpoint-specific single-layer candidate. H_WIN-stable layers that also fail C6R_WIN or controls are compatible with a rotation-related quantization-geometry difference; this does not prove its cause.')+'\n\n'
    text+=f"Q7. Recommended category={v['PRIMARY_NEXT_STEP_CATEGORY']}. "+('A proposal only is stored under proposals/QWEN_C6R_H_QDQ_FINEGRAINED_CAPTURE_V1.json, up to4 strict layers, same23 panel and same-GPU pairing; prioritize real C128 groups and only structurally valid heads. No GPU work has run.' if v['STRICT_CANDIDATE_LAYER_COUNT'] else 'Stop this QDQ-intervention route. No head/group capture proposal is promoted from failed layers.')+'\n\n'
    text+='Q8. NO: current endpoint-selected discovery evidence is insufficient for C8. Mandatory future chain is discovery, freeze location+metric+direction, independent fresh confirmation, then separately consider C8. Fresh confirmation excludes these23, C6R/C7 replay and discovery data. No future dataset/docs were selected. NEW_C8_TRAINING_AUTHORIZED=NO.\n\n'
    text+='## Architecture and component limits\n\n'+json.dumps(head,indent=2)+'\n\nState axis32 is the actual value-head axis, not an arbitrary split;16 key/query heads are repeated twice before the recurrent rule. Each native C128 group fixes value-head and value-coordinate and spans128 key coordinates. Parent group summaries contain4096 actual amax/scale values per layer, but individual group measurements were not saved. Functional capture after out_proj is mixed across heads and cannot be relabeled per-head. No M5 values are available, none used in ranking/objectives.\n\n'
    text+='## Figures\n\n![Median per layer](figures/figure1.png)\n\n![Positive fractions](figures/figure2.png)\n\n![Specificity gap](figures/figure3.png)\n\nThe three curves/bar views are descriptive; figures never determine candidate eligibility.\n\n'
    text+='## Integrity and reproduction\n\nParent sealed106-file manifest and raw23-file freeze were checked before/after. Original raw JSON and parent CSV agree exactly for1104 native QDQ rows and552 paired layer differences. Model/config/rotation and numerical source function/file hashes were verified without importing/constructing a model. All24 runtime layer IDs match each raw payload, configuration layer_types and frozen ordinal thirds. No source was changed. scripts/layerwise_qdq_diagnosis.py records prepare/analyze/seal commands with /data/ydai/miniconda3/envs/bitdecode/bin/python; analyses are overwrite-protected. FINAL_SHA256SUMS is emitted only after analysis exit to avoid hashing a live stdout log. FINAL_ACTION=STOP.\n'
    write('FINAL_REPORT.md',text)
def seal():
    require((ROOT/'FINAL_REPORT.md').exists(),'report incomplete');require((ROOT/'logs/analysis.exitcode').read_text().strip()=='0','analysis nonzero')
    verify(read(ROOT/'provenance/input_artifact_inventory.json')['before_files']);assert_prepared()
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.tmp.' not in p.name)
    write('FINAL_SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files))
    verify(manifest(ROOT/'FINAL_SHA256SUMS'));print(json.dumps(dict(status='PASS',sealed_files=len(files),sha256=sha(ROOT/'FINAL_SHA256SUMS'),root=str(ROOT))),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','analyze','seal'],required=True);a=ap.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only: explicitly hide all CUDA devices')
    require('torch' not in sys.modules,'model/torch import forbidden');selftest()
    try:globals()[a.phase]()
    except BaseException as e:
        if ROOT.exists():save(f'logs/{a.phase}.error.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),utc=utc(),no_measurement_retry=True))
        raise
if __name__=='__main__':main()
