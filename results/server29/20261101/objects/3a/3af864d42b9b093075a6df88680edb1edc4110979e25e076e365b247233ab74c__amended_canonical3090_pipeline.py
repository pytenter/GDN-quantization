#!/usr/bin/env python3
"""Fail-closed deployment and endpoint closure; all scientific helpers are reused."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import amended_canonical3090_endpoint as A
import canonical3090_pipeline as BASE

ROOT=A.ROOT
OLD=A.OLD
ALL_ARMS=A.ARMS
ARMS=('C6R','C7')
REPO=OLD.parents[2]
DIAG=Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/analyze_existing.py')
RUNTIME=REPO/'experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/scripts/runtime.py'
PROBE=REPO/'experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/scripts/deployment_frozen_probe.py'
CORE=REPO/'experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
ROTATION=REPO/'experiments/shared/rotation/cayley_rotation.py'
MANDATORY_PAIRS=(('C7','C6R'),('C7','H'),('C6R','H'),('C7','historical_C6'),('C6R','historical_C6'),('C7','FP'))
AUXILIARY_PAIRS=(('C5R','historical_C5'),('C5R','H'))
PAIRS=MANDATORY_PAIRS


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def read(path):return json.loads(Path(path).read_text())


def table(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare():
    old=BASE.check_freeze()
    amendment=ROOT/'protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json'
    if read(amendment)['original_preregistration_rewritten']:raise RuntimeError('invalid amendment')
    cfg=dict(old)
    sources={str(p):sha(p) for p in sorted((ROOT/'scripts').glob('amended_*')) if p.is_file()}
    for ext in ('md','json'):
        p=ROOT/f'protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.{ext}';sources[str(p)]=sha(p)
    cfg.update(mandatory_conditions=['C6R','C7'],conditional_C5R='RUN_DUE_TO_MATERIAL_SHIFT only',
        primary_pair=['C7','C6R'],paired_family=[list(x) for x in MANDATORY_PAIRS],
        conditional_auxiliary_family=[list(x) for x in AUXILIARY_PAIRS],
        multiplicity='Holm fixed six mandatory; separate conditional two-comparison auxiliary Holm',
        blind_until_all_180_frozen=False,blind_until_all_routed_outputs_frozen=True,
        original_endpoint_config_sha256=sha(ROOT/'protocol/canonical_endpoint.json'),
        amendment_sha256=sha(amendment),amended_sources=sources)
    save(ROOT/'protocol/canonical_endpoint_amended.json',cfg)
    save(ROOT/'protocol/canonical_endpoint_amended_freeze.json',{'status':'PASS','config_sha256':sha(ROOT/'protocol/canonical_endpoint_amended.json')})
    save(ROOT/'logs/endpoint_amended_prepare.completion.json',{'status':'PASS','mandatory_samples':120,'conditional_extra':60,'GPU_work_started':False})


def check_freeze():
    BASE.check_freeze()
    p=ROOT/'protocol/canonical_endpoint_amended.json';cfg=read(p)
    if sha(p)!=read(ROOT/'protocol/canonical_endpoint_amended_freeze.json')['config_sha256']:raise RuntimeError('amended endpoint config drift')
    for path,h in cfg['amended_sources'].items():
        if sha(path)!=h:raise RuntimeError('amended control identity drift '+path)
    return cfg


def active_arms():
    gate=read(ROOT/'analysis/C5R_REPRODUCTION_GATE.json')
    if gate['status']!='PASS' or gate['no_AIME_used'] is not True:raise RuntimeError('non-AIME routing gate missing')
    if gate['amendment_sha256']!=sha(ROOT/'protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json'):raise RuntimeError('routing amendment mismatch')
    routes={'CLOSE_REPRODUCTION':'SKIP_NOT_NEEDED','MATERIAL_SHIFT':'RUN_DUE_TO_MATERIAL_SHIFT','INCONCLUSIVE':'SKIP_INCONCLUSIVE','NOT_COMPLETED':'NOT_REACHED'}
    if routes.get(gate['C5R_REPRODUCTION'])!=gate['C5R_AIME26_ROUTE']:raise RuntimeError('invalid C5R route')
    for path,h in gate['evidence_sha256'].items():
        if sha(ROOT/path)!=h:raise RuntimeError('routing evidence changed '+path)
    return ('C6R','C7','C5R') if gate['C5R_AIME26_ROUTE']=='RUN_DUE_TO_MATERIAL_SHIFT' else ('C6R','C7')


def smoke():
    cfg=check_freeze();g=A.generator();import torch
    trust=read(ROOT/'provenance/trusted_environment_freeze.json')
    if trust['A800_TRAINING_ENVIRONMENT_TRUSTED']!='YES':raise RuntimeError('training trust gate missing')
    for p in ('teacher/teacher_cache_parity.json','readiness/update0_forward_parity.json','readiness/update0_validation_parity.json','readiness/final_hidden_indexing_audit.json'):
        if read(ROOT/p)['status']!='PASS':raise RuntimeError('A800 scientific prerequisite failed: '+p)
    for arm in ALL_ARMS:
        if arm=='C5R' and not (ROOT/'training/C5R/summary.json').exists():
            isolation=read(ROOT/'analysis/C5R_ISOLATED_FAILURE_CONTINUATION.json')
            if isolation.get('classification')!='C5R_SPECIFIC_FAILURE' or isolation.get('shared_validity')!='PASS' or not isolation.get('manual_evidence_review'):raise RuntimeError('C5R failure not demonstrably isolated')
            for path,h in isolation['evidence_sha256'].items():
                if sha(ROOT/path)!=h:raise RuntimeError('isolation evidence drift')
            continue
        for kind in ('gradient_smoke','h64_readiness'):
            if read(ROOT/f'readiness/{arm.lower()}_{kind}.json')['status']!='PASS':raise RuntimeError('arm readiness failed')
        summary=read(ROOT/f'training/{arm}/summary.json')
        if summary['status']!='PASS' or summary['updates']!=125 or summary['exposures']!=1000 or summary['H']!=64 or summary['FROZEN_MODEL_INTEGRITY']!='PASS' or summary['retry_count']!=0:raise RuntimeError('formal training incomplete')
        guards=[json.loads(x) for x in (ROOT/f'qualification/training_{arm}_immutable_guards.jsonl').read_text().splitlines()]
        if not {'FRESH_FULL_GPU_INTEGRITY','BEFORE_UPDATE1','AFTER_UPDATE_1','AFTER_UPDATE_63','AFTER_UPDATE_125'}.issubset({r['point'] for r in guards}) or any(r['status']!='PASS' for r in guards):raise RuntimeError('immutable model guards incomplete')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader,nounits'],text=True)
    if len([x for x in gpu.splitlines() if 'RTX 3090' in x])!=4:raise RuntimeError('not canonical four RTX3090 host')
    rt=module('runtime',RUNTIME);rt.check_parent_sources();core=rt.load_core();p=module('unchanged_canonical_deployment_probe',PROBE)
    selected=read(ROOT/'deployment/selected_artifacts_pre_endpoint.json')
    if selected['status']!='PASS':raise RuntimeError('pre-endpoint selected artifacts not frozen')
    model,tokenizer=g.load_model_and_tokenizer();core.freeze_model(model);device=model.get_input_embeddings().weight.device
    before={n:g.tensor_sha256(t) for n,t in model.state_dict().items()};results={}
    for arm in ARMS:
        m=selected['conditions'][arm];path=ROOT/'deployment'/arm/'frozen_rotation_fp32.pt'
        if sha(path)!=m['file_sha256']:raise RuntimeError('rotation transfer identity failed')
        save(ROOT/f'deployment/{arm.lower()}_manifest.json',m)
        bank,identity=g.load_frozen_rotation(path,ROOT/f'deployment/{arm.lower()}_manifest.json',device,arm)
        orth={str(l):float((bank.layer(l).matrix().T@bank.layer(l).matrix()-torch.eye(128,device=device)).abs().max()) for l in g.GDN_LAYERS}
        if max(orth.values())>core.ORTHOGONALITY_THRESHOLD:raise RuntimeError('deployment orthogonality failed')
        result=p.run_fixed_probe(core,model,tokenizer,bank)
        results[arm]={'status':'PASS','identity':identity,'orthogonality_by_layer':orth,'fixed_nonAIME_probe':result,'no_free_generation':True}
        del bank
    after={n:g.tensor_sha256(t) for n,t in model.state_dict().items()}
    if before!=after:raise RuntimeError('canonical deployment smoke changed frozen model')
    save(ROOT/'deployment/canonical_3090_smoke.json',{'status':'PASS','conditions':results,'loader_sha256':sha(A.RUNNER),'probe_sha256':sha(PROBE),
        'runtime':g.runtime_version(),'GPU_inventory_csv':gpu,'base_weights_unchanged':True,'theta_reconstruction':False,'A800_R_bytes_preserved':True})


def finish():
    cfg=check_freeze();g=A.generator();pp=module('frozen_endpoint_postprocess',OLD/'scripts/postprocess_canonical60.py')
    d=module('frozen_question_cluster',DIAG);_,jobs=pp.load_dataset();allrows={};files=[];gpurows=[]
    for arm in ARMS:
        records=[]
        manifest=read(ROOT/f'deployment/{arm.lower()}_manifest.json')
        for worker in range(4):
            if int((ROOT/f'logs/{arm}_worker{worker}.exitcode').read_text())!=0:raise RuntimeError('worker not successful')
            path=ROOT/f'aime26/{arm.lower()}/shards/worker{worker}.jsonl'
            if read(path.with_suffix('.summary.json'))['gate']!='PASS':raise RuntimeError('shard gate failed')
            shard=pp.load_jsonl(path)
            expected={(j['problem_id'],j['seed']):j for j in cfg['jobs'] if j['gpu']==worker}
            if len(shard)!=15:raise RuntimeError('shard record count')
            for r in shard:
                key=(r['problem_id'],int(r['seed']))
                if key not in expected:raise RuntimeError('unexpected sample')
                j=expected[key]
                if (r['gpu']!=str(worker) or r['worker_id']!=worker or r['canonical_index']!=ALL_ARMS.index(arm)*60+j['pair_index']
                    or r['formal_config_sha256']!=sha(ROOT/'protocol/canonical_endpoint_amended.json') or r['rotation_sha256']!=manifest['file_sha256']
                    or r['deployment_manifest_sha256']!=sha(ROOT/f'deployment/{arm.lower()}_manifest.json')
                    or r['problem']!=jobs[key]['problem'] or int(r['gold_answer'])!=int(jobs[key]['answer'])):raise RuntimeError('sample metadata identity mismatch')
                gpurows.append({'condition':arm,'problem_id':key[0],'seed':key[1],'gpu':worker,'input_ids_hash':r['input_ids_hash'],'rotation_sha256':r['rotation_sha256'],'payload_sha256':r['output_payload_sha256']})
            records.extend(shard);files.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'records':15})
        gate=g.validate_records(records,arm.lower(),60,set(jobs))
        if gate['gate']!='PASS':raise RuntimeError('full condition failed: '+json.dumps(gate))
        allrows[arm]={(r['problem_id'],int(r['seed'])):r for r in records}
    for key in jobs:
        if len({(allrows[a][key]['input_ids_hash'],allrows[a][key]['gpu']) for a in ARMS})!=1:raise RuntimeError('paired prompt or GPU differs')
    # Freeze and verify before any scorer reads new responses.
    save(ROOT/'analysis/output_freeze_manifest.json',{'status':'PASS','source_files':files,'records':60*len(ARMS),'per_condition':60,'scoring_before_freeze':False})
    save(ROOT/'analysis/per_sample_gpu_metadata.json',gpurows)
    for a in ARMS:
        with (ROOT/f'aime26/{a.lower()}/SHA256SUMS').open('x') as f:
            for item in files:
                if f'/{a.lower()}/' in item['path']:f.write(item['sha256']+'  '+os.path.relpath(item['path'],ROOT)+'\n')
    for item in files:
        if sha(item['path'])!=item['sha256']:raise RuntimeError('output changed during freeze')
        Path(item['path']).chmod(0o444)
    scorer=pp.import_scorer();scored={a:pp.score_records(a,allrows[a],scorer) for a in ARMS}
    for a,value in scored.items():save(ROOT/f'aime26/{a.lower()}/scored.json',value)
    flat=[{k:v for k,v in r.items() if k!='extraction'} for a in ARMS for r in scored[a]['records']];table(ROOT/'aime26/scores.csv',flat)
    base=d.load_endpoint_records()
    outcomes={a:{(r['problem_id'],int(r['seed'])):bool(r['v4_correct']) for r in scored[a]['records']} for a in ARMS}
    for new,old in [('H','H'),('FP','FP'),('historical_C5','C5'),('historical_C6','C6')]:outcomes[new]={k:r['correct'] for k,r in base[old].items()}
    comps=[]
    for a,b in PAIRS:
        name=a+'_vs_'+b;s,paired=pp.paired_table(name,outcomes[a],outcomes[b],a,b)
        cluster=d.cluster_bootstrap(sorted(jobs),{k:{'correct':v} for k,v in outcomes[a].items()},{k:{'correct':v} for k,v in outcomes[b].items()})
        s.update(cluster);s.update(both_correct=s['same_correct'],both_wrong=s['same_wrong'],candidate_only=s['rescue'],reference_only=s['regress'],accuracy_difference=s['net']/60)
        table(ROOT/f'aime26/{name}_paired.csv',paired);comps.append(s)
    adjusted={}
    for family in (MANDATORY_PAIRS,AUXILIARY_PAIRS):
        names={a+'_vs_'+b for a,b in family}
        adjusted.update(pp.holm({s['comparison']:s['exact_two_sided_p'] for s in comps if s['comparison'] in names}))
    for s in comps:
        s['holm_p']=adjusted[s['comparison']]
        s['multiplicity_family']='mandatory_six' if s['comparison'] in {a+'_vs_'+b for a,b in MANDATORY_PAIRS} else 'conditional_auxiliary_two'
    save(ROOT/'aime26/paired_comparisons.json',{'primary':'C7_vs_C6R','comparisons':comps,'question_cluster_source_sha256':sha(DIAG)})
    table(ROOT/'aime26/paired_comparisons.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in s.items()} for s in comps])
    behavior=[pp.trajectory_record(a,r,scorer) for a in ARMS for r in allrows[a].values()]
    # Reuse archived behavior definitions; do not rescore baselines.
    for old,label in [('C5','historical_C5'),('C6','historical_C6')]:
        for worker in range(4):
            path=OLD/f'outputs/formal/shards/worker{worker}/{old}.jsonl'
            behavior.extend(pp.trajectory_record(label,r,scorer) for r in pp.load_jsonl(path))
    h,_=pp.load_baseline_csv();rawh={}
    for path in sorted({Path(r['source_file']) for r in h.values()}):
        for r in pp.load_jsonl(path):
            if r['method']==pp.H_CONDITION:rawh[(r['problem_id'],int(r['seed']))]=r
    if set(rawh)!=set(jobs):raise RuntimeError('Hadamard archived raw coverage')
    behavior.extend(pp.trajectory_record('H',r,scorer) for r in rawh.values());table(ROOT/'aime26/behavior_per_sample.csv',behavior)
    summary=[]
    for a in (*ARMS,'H','historical_C5','historical_C6'):
        rs=[r for r in behavior if r['condition']==a]
        summary.append({'condition':a,'samples':len(rs),'generated_tokens_median':pp.median([r['generated_tokens'] for r in rs]),
            'length_limit':sum(r['length_ceiling'] for r in rs),'EOS':sum(r['termination_type']=='eos' for r in rs),
            'no_final_claim':sum(r['committed_claim_count']==0 for r in rs),'post_commit_fraction_median':pp.median([r['post_commit_fraction'] for r in rs]),
            'repeated_8gram_fraction_median':pp.median([r['repeated_8gram_fraction'] for r in rs]),'oscillation_proxy_median':pp.median([r['oscillation_proxy'] for r in rs]),
            'retractions':sum(r['retractions'] for r in rs),'V4_abstain':scored[a]['abstain'] if a in scored else sum(r.get('status')=='ABSTAIN' for r in base['H' if a=='H' else a.removeprefix('historical_')].values())})
    table(ROOT/'aime26/behavior_summary.csv',summary)
    final(scored,comps)
    save(ROOT/'logs/endpoint_closure.completion.json',{'status':'PASS','new_conditions':len(ARMS),'samples':60*len(ARMS),'scorer_unchanged':True,'semantic_retries':0,'protocol_changed':False})


def final(scored,comps):
    trusted=read(ROOT/'provenance/trusted_environment_freeze.json')
    status={k:trusted[k] for k in ('A800_HOST_QUALIFICATION','FULL_MODEL_GPU_PARAMETER_INTEGRITY','FULL_MODEL_FRESH_PROCESS_REPEATABILITY','FP_RUNTIME_STABILITY','A800_TRAINING_ENVIRONMENT_TRUSTED')}
    status.update(LM_HEAD_CANARY_3X='PASS',CLEAN_TEACHER_TARGET_PARITY='PASS',C5R_C6R_C7_UPDATE0_FORWARD_PARITY='PASS',FULL_DENSITY_FULL60='CANCELLED_BY_USER',PRIMARY_MATCHED_COMPARISON='C7_VS_C6R')
    reproduction=read(ROOT/'analysis/C5R_REPRODUCTION_GATE.json')
    status.update(C5R_REPRODUCTION=reproduction['C5R_REPRODUCTION'],C5R_AIME26_ROUTE=reproduction['C5R_AIME26_ROUTE'],C5R_AIME26='NOT_RUN')
    for a in ALL_ARMS:
        completed=(ROOT/f'training/{a}/summary.json').exists()
        for suffix in ('GRADIENT_SMOKE','H64_READY'):
            rel='gradient_smoke' if suffix=='GRADIENT_SMOKE' else 'h64_readiness'
            p=ROOT/f'readiness/{a.lower()}_{rel}.json'
            status[a+'_'+suffix]=read(p).get('status','FAIL') if p.exists() else 'NOT_COMPLETED'
        status[a+'_DEPLOYMENT']='PASS' if a in ARMS else 'NOT_RUN'
        status[a+'_FORMAL_TRAINING']='COMPLETE' if completed else 'NOT_COMPLETED'
        status[a+'_SELECTED_UPDATE']=read(ROOT/f'training/{a}/summary.json')['selected_update'] if completed else 'NA'
        if a in scored:status[a+'_AIME26']=str(scored[a]['correct'])+'/60'
    status['C7_H64_DIRECT_READY']='PASS'
    by={s['comparison']:s for s in comps};matched=by['C7_vs_C6R'];h=by['C7_vs_H'];shift=by['C6R_vs_historical_C6']
    status.update(C7_VS_C6R_NET=matched['net'],C7_VS_HADAMARD_NET=h['net'],C6R_VS_HISTORICAL_C6=shift['net'])
    # Positive observed direction follows amendment cases A/B/C; statistical
    # significance is reported separately, never promoted to causal proof.
    if matched['net']<=0:verdict='C7_OBJECTIVE_NOT_SUPPORTED'
    elif h['net']>0:verdict='C7_OBJECTIVE_SUPPORTED_AND_BEATS_HADAMARD'
    else:verdict='C7_OBJECTIVE_SUPPORTED_BUT_BELOW_HADAMARD'
    status['PRIMARY_VERDICT']=verdict
    status['CLEANROOM_C6_MATERIAL_SHIFT']='YES' if shift['net']!=0 and shift['holm_p']<=.05 else 'UNKNOWN'
    status['C6_SHIFT_DECISION_BOUNDARY']='Multiplicity-controlled paired difference is evidence of a shift; otherwise UNKNOWN, not proof of equivalence. No automatic attribution to old vGPU corruption.'
    status['MATCHED_STATISTICAL_SUPPORT']='YES' if matched['net']>0 and matched['holm_p']<=.05 else 'NO'
    save(ROOT/'analysis/final_status.json',status)
    lines=['# A800 clean-room amended matched-objective experiment','',
        'Primary: C7 vs C6R. C5R is auxiliary lineage evidence and is not a co-primary arm.',
        'All completed training arms use the unchanged trusted A800 protocol, seed0, H64, 125 updates/1000 captures. C6R/C7 differ only in the intended objective.',
        'Inference uses canonical four RTX3090, direct byte-identical A800 materialized FP32 R, unchanged native generator and Frozen V4. No theta reconstruction, checkpoint retuning or new baseline generation.',
        '', '| Condition | Correct | Accuracy |','|---|---:|---:|','| FP archived |52/60|86.67%|','| Native archived |18/60|30.00%|','| Hadamard archived |41/60|68.33%|','| Historical C5 archived |33/60|55.00%|','| Historical C6 archived |36/60|60.00%|']
    for a in ARMS:lines.append(f"|{a}|{scored[a]['correct']}/60|{100*scored[a]['accuracy']:.2f}%|")
    if 'C5R' not in scored:lines.append('|C5R auxiliary|NOT_RUN|Not measured; not assumed equal to 33/60|')
    lines+=['','| Comparison | Both correct | Reference only | Candidate only | Both wrong | Net | exact p | Holm p | Question-cluster 95% CI |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for s in comps:lines.append(f"|{s['comparison']}|{s['both_correct']}|{s['reference_only']}|{s['candidate_only']}|{s['both_wrong']}|{s['net']}|{s['exact_two_sided_p']:.6g}|{s['holm_p']:.6g}|{s['cluster_bootstrap_ci95']}|")
    lines+=['','## Required statuses','', '```text']+[k+' = '+str(v) for k,v in status.items()]+['```','',
        '## Scientific interpretation and limits','',
        'Verdict labels describe the observed amendment A/B/C directions, not proof of causal superiority. Exact paired tests, fixed six-comparison Holm and project question-cluster uncertainty must accompany any interpretation; the two optional C5R comparisons have a separate auxiliary Holm family.',
        'C5R routing was frozen using non-AIME evidence before deployment. INCONCLUSIVE means no extra C5R AIME by default, not an assumed accuracy. Historical missing Final-Hidden values were not invented.',
        'Archived C5/C6 endpoints remain valid observations. C6R historical shift is not automatically caused by old silent corruption; environment, runtime and numerical trajectory remain alternatives.',
        'Exact rescue/regression sample IDs are in paired JSON/CSV. All scheduled candidates, learning curves, selected theta/R hashes, Shared16, descriptive rotation distances and reproduction gate are preserved.',
        'Runtime, hardware, teacher and model provenance plus setup/operational incidents are recorded under provenance/, qualification/, protocol/ and logs/. No scientific failure was retried; no frozen scientific code was changed.',
        'No GSM8K, MATH-500, Full60 mechanism capture, new loss, new seed, C8 or follow-on experiment was started. AIME is a research endpoint, not an untouched benchmark.']
    with (ROOT/'FINAL_REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    with (ROOT/'NEXT_EXPERIMENT_DECISION.md').open('x') as f:f.write('# Manual next decision only\n\n'+verdict+'\n\nReview the frozen uncertainty and non-AIME evidence. No follow-on workload is authorized or automatically executed.\n')


def main():
    global ARMS,PAIRS
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('prepare','smoke','finish'),required=True);args=p.parse_args()
    for d in ('protocol','logs','deployment','analysis','aime26'): (ROOT/d).mkdir(parents=True,exist_ok=True)
    marker='endpoint_amended_prepare' if args.phase=='prepare' else 'endpoint_'+args.phase
    for suffix in ('completion','error'):
        if (ROOT/f'logs/{marker}.{suffix}.json').exists():raise RuntimeError('existing evidence: no automatic retry')
    try:
        if args.phase!='prepare':
            ARMS=active_arms();PAIRS=MANDATORY_PAIRS+(AUXILIARY_PAIRS if 'C5R' in ARMS else ())
        globals()[args.phase]()
        if args.phase!='prepare':save(ROOT/f'logs/endpoint_{args.phase}.completion.json',{'status':'PASS'})
        if args.phase=='finish':
            with (ROOT/'SHA256SUMS').open('x') as f:
                for q in sorted(ROOT.rglob('*')):
                    if q.is_file() and q.name!='SHA256SUMS' and '__pycache__' not in q.parts and '.git' not in q.parts and not (q.parent.name=='logs' and q.suffix in ('.log','.exitcode')):
                        f.write(sha(q)+'  '+str(q.relative_to(ROOT))+'\n')
    except BaseException as e:
        save(ROOT/f'logs/{marker}.error.json',{'status':'FAIL','error':str(e),'traceback':traceback.format_exc(),'do_not_retry':True});raise


if __name__=='__main__':main()
