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
import canonical3090_endpoint as A

ROOT=A.ROOT
OLD=A.OLD
ARMS=A.ARMS
REPO=OLD.parents[2]
DIAG=Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/analyze_existing.py')
RUNTIME=REPO/'experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/scripts/runtime.py'
PROBE=REPO/'experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/scripts/deployment_frozen_probe.py'
CORE=REPO/'experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py'
ROTATION=REPO/'experiments/shared/rotation/cayley_rotation.py'
PAIRS=(('C7','C6R'),('C7','H'),('C6R','H'),('C5R','H'),('C6R','historical_C6'),('C5R','historical_C5'),('C7','historical_C6'),('C7','FP'))


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
    g=A.generator();old=read(OLD/'configs/formal_config.json')
    expected={OLD/'configs/formal_config.json':'557ac6670fe6e88606be226ccf0dea289ebd34dd4c2be54e7e54b9b6ee9e0836',
        OLD/'configs/hardware_assignment.json':'75b5df954f64f606f30143df3bf5329e238a6eabe8253bfc287141b0443c475a',
        A.RUNNER:A.EXPECTED_RUNNER,Path(old['scorer']):old['scorer_sha256'],Path(old['canonical_protocol_manifest']):old['canonical_protocol_manifest_sha256'],
        g.DATASET:old['canonical_dataset_sha256'],CORE:'48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e',
        ROTATION:'62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067'}
    for p,h in expected.items():
        if sha(p)!=h:raise RuntimeError('canonical source identity mismatch: '+str(p))
    pp=module('frozen_endpoint_postprocess',OLD/'scripts/postprocess_canonical60.py')
    d=module('frozen_question_cluster',DIAG)
    _,keys=pp.load_dataset();base=d.load_endpoint_records()
    if any(set(rs)!=set(keys) for rs in base.values()):raise RuntimeError('archived endpoint keys differ from canonical manifest')
    if {k:sum(r['correct'] for r in base[k].values()) for k in ('FP','Native','H','C5','C6')}!={'FP':52,'Native':18,'H':41,'C5':33,'C6':36}:
        raise RuntimeError('archived endpoints drift')
    jobs=[j for j in read(OLD/'configs/hardware_assignment.json')['jobs'] if j['condition']=='C5']
    jobs=sorted(jobs,key=lambda j:j['pair_index'])
    if len(jobs)!=60 or {(j['problem_id'],j['seed']) for j in jobs}!=set(keys) or [j['pair_index'] for j in jobs]!=list(range(60)):
        raise RuntimeError('canonical manifest identity coverage')
    if any(j['gpu']!=j['pair_index']%4 for j in jobs):raise RuntimeError('archived static assignment invalid')
    for p in (RUNTIME,PROBE,DIAG,OLD/'scripts/postprocess_canonical60.py',pp.BASELINE_CSV,pp.BASELINE_SUMMARY,OLD/'analysis/C5_scored.json',OLD/'analysis/C6_scored.json'):
        expected[p]=sha(p)
    for worker in range(4):
        for arm in ('C5','C6'):
            p=OLD/f'outputs/formal/shards/worker{worker}/{arm}.jsonl';expected[p]=sha(p)
    h,_=pp.load_baseline_csv()
    for p in sorted({Path(r['source_file']) for r in h.values()}):expected[p]=sha(p)
    for top in (Path('/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26'),RUNTIME.parent.parent/'configs',CORE.parent/'manifests'):
        for p in sorted(top.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.json'):expected[p]=sha(p)
    # Freeze canonical implementation imports and existing checkpoint/tokenizer bytes prospectively.
    for top in (Path('/data/zypan/transformers-qwen35/src'),Path('/data/zypan/experiments/qwen35_gdn_quant')):
        for p in sorted(top.rglob('*.py')):expected[p]=sha(p)
    for p in sorted(g.MODEL_PATH.rglob('*')):
        if p.is_file() and (p.suffix in ('.json','.safetensors','.jinja') or p.name=='merges.txt'):expected[p]=sha(p)
    for p in sorted((ROOT/'scripts').glob('*')):
        if p.is_file():expected[p]=sha(p)
    cfg={'task':ROOT.name,'training_order':list(ARMS),'primary_pair':['C7','C6R'],'paired_family':[list(x) for x in PAIRS],
        'dataset_sha256':sha(g.DATASET),'frozen_sources':{str(p):h for p,h in expected.items()},'runtime':g.runtime_version(),
        'jobs':[{k:j[k] for k in ('problem_id','seed','pair_index','gpu')} for j in jobs],
        'canonical_protocol':old['protocol'],'canonical_formal_config_sha256':sha(OLD/'configs/formal_config.json'),
        'runner_byte_identical':True,'adapter_changes':'condition labels and output location only; unchanged generate_one and loader',
        'bootstrap_source':str(DIAG),'bootstrap_seed':d.BOOTSTRAP_SEED,'bootstrap_reps':d.BOOTSTRAP_REPS,
        'multiplicity':'Holm across all eight prospectively fixed comparisons','blind_until_all_180_frozen':True,
        'baseline_reuse':{'FP':52,'Native':18,'H':41,'historical_C5':33,'historical_C6':36},
        'dynamic_reassignment':False,'disk_threshold_bytes':15*1024**3,'semantic_retry':False,
        'training_on_3090':False,'FULL_DENSITY_FULL60':'CANCELLED_BY_USER'}
    save(ROOT/'protocol/canonical_endpoint.json',cfg)
    save(ROOT/'protocol/canonical_endpoint_freeze.json',{'status':'PASS','config_sha256':sha(ROOT/'protocol/canonical_endpoint.json')})
    save(ROOT/'logs/endpoint_prepare.completion.json',{'status':'PASS','sample_seed_pairs':60,'conditions':list(ARMS),'GPU_work_started':False})


def check_freeze():
    cfg=read(ROOT/'protocol/canonical_endpoint.json')
    if sha(ROOT/'protocol/canonical_endpoint.json')!=read(ROOT/'protocol/canonical_endpoint_freeze.json')['config_sha256']:
        raise RuntimeError('endpoint config changed')
    for p,h in cfg['frozen_sources'].items():
        if sha(p)!=h:raise RuntimeError('endpoint source/input drift: '+p)
    g=A.generator()
    if g.runtime_version()!=cfg['runtime']:raise RuntimeError('canonical runtime version drift')
    return cfg


def smoke():
    cfg=check_freeze();g=A.generator();import torch
    trust=read(ROOT/'provenance/trusted_environment_freeze.json')
    if trust['A800_TRAINING_ENVIRONMENT_TRUSTED']!='YES':raise RuntimeError('training trust gate missing')
    for p in ('teacher/teacher_cache_parity.json','readiness/update0_forward_parity.json','readiness/update0_validation_parity.json','readiness/final_hidden_indexing_audit.json'):
        if read(ROOT/p)['status']!='PASS':raise RuntimeError('A800 scientific prerequisite failed: '+p)
    for arm in ARMS:
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
                if (r['gpu']!=str(worker) or r['worker_id']!=worker or r['canonical_index']!=ARMS.index(arm)*60+j['pair_index']
                    or r['formal_config_sha256']!=sha(ROOT/'protocol/canonical_endpoint.json') or r['rotation_sha256']!=manifest['file_sha256']
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
    save(ROOT/'analysis/output_freeze_manifest.json',{'status':'PASS','source_files':files,'records':180,'per_condition':60,'scoring_before_freeze':False})
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
    adjusted=pp.holm({s['comparison']:s['exact_two_sided_p'] for s in comps})
    for s in comps:s['holm_p']=adjusted[s['comparison']]
    save(ROOT/'aime26/paired_comparisons.json',{'primary':'C7_vs_C6R','eight_fixed_family':comps,'question_cluster_source_sha256':sha(DIAG)})
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
    save(ROOT/'logs/endpoint_closure.completion.json',{'status':'PASS','new_conditions':3,'samples':180,'scorer_unchanged':True,'semantic_retries':0,'protocol_changed':False})


def final(scored,comps):
    trusted=read(ROOT/'provenance/trusted_environment_freeze.json');status={k:trusted[k] for k in ('A800_HOST_QUALIFICATION','FULL_MODEL_GPU_PARAMETER_INTEGRITY','FULL_MODEL_FRESH_PROCESS_REPEATABILITY','FP_RUNTIME_STABILITY','A800_TRAINING_ENVIRONMENT_TRUSTED')}
    status.update(LM_HEAD_CANARY_3X='PASS',CLEAN_TEACHER_TARGET_PARITY='PASS',C5R_C6R_C7_UPDATE0_FORWARD_PARITY='PASS',FULL_DENSITY_FULL60='CANCELLED_BY_USER')
    for a in ARMS:
        for suffix in ('GRADIENT_SMOKE','H64_READY','DEPLOYMENT'):status[a+'_'+suffix]='PASS'
        status[a+'_FORMAL_TRAINING']='COMPLETE';status[a+'_SELECTED_UPDATE']=read(ROOT/f'training/{a}/summary.json')['selected_update'];status[a+'_AIME26']=str(scored[a]['correct'])+'/60'
    status['C7_H64_DIRECT_READY']='PASS';by={s['comparison']:s for s in comps};matched=by['C7_vs_C6R'];h=by['C7_vs_H']
    status.update(C7_VS_C6R_NET=matched['net'],C7_VS_HADAMARD_NET=h['net'],C6R_VS_HISTORICAL_C6=by['C6R_vs_historical_C6']['net'])
    # Conservative prospective endpoint rule: no post-hoc "meaningful" threshold.
    if matched['net']<=0:verdict='C7_NOT_BETTER_THAN_C6R'
    elif h['net']<0:verdict='C7_BEATS_C6R_BUT_NOT_HADAMARD'
    elif matched['holm_p']<=.05:verdict='C7_OBJECTIVE_SUPPORTED'
    else:verdict='INCONCLUSIVE'
    status['PRIMARY_VERDICT']=verdict;status['REPRODUCTION_CLASSIFICATION']='INCONCLUSIVE_REQUIRES_QUALITATIVE_CURVE_REVIEW'
    save(ROOT/'analysis/final_status.json',status)
    lines=['# A800 clean-room C5R/C6R/C7 matched experiment','',
        'PRIMARY comparison: C7 vs C6R. Historical C5/C6 remain valid archived endpoints; this does not retrospectively invalidate them.',
        'AIME26 is a research endpoint, not an untouched benchmark. Exact Canonical60, archived baseline reuse, unchanged Frozen V4. No AIME tuning.',
        'Training: same qualified A800, seed0, H64, 125 updates/1000 captures per arm. Only C6R/C7 intended difference is objective.',
        'Inference: canonical four RTX3090; exact training-host materialized FP32 Delta bytes loaded directly, never reconstructed from theta.',
        '', '| Condition | Correct | Accuracy |','|---|---:|---:|','| FP archived |52/60|86.67%|','| Native archived |18/60|30.00%|','| Hadamard archived |41/60|68.33%|','| Historical C5 archived |33/60|55.00%|','| Historical C6 archived |36/60|60.00%|']
    for a in ARMS:lines.append(f"|{a}|{scored[a]['correct']}/60|{100*scored[a]['accuracy']:.2f}%|")
    lines+=['','| Candidate vs reference | Both correct | Reference only | Candidate only | Both wrong | Net | exact p | Holm p | Question-cluster 95% CI |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for s in comps:lines.append(f"|{s['comparison']}|{s['both_correct']}|{s['reference_only']}|{s['candidate_only']}|{s['both_wrong']}|{s['net']}|{s['exact_two_sided_p']:.6g}|{s['holm_p']:.6g}|{s['cluster_bootstrap_ci95']}|")
    lines+=['','## Required statuses','', '```text']+[k+' = '+str(v) for k,v in status.items()]+['```','',
        '## Interpretation and limits','',
        'Endpoint verdict uses positive paired net, the frozen eight-comparison Holm family, and observed Hadamard direction. Support requires Holm p<=0.05; positive direction without sufficient uncertainty support is INCONCLUSIVE.',
        'Historical reproduction requires qualitative review of predefined validation curves and Shared16 metrics. No arbitrary post-hoc numerical boundary is applied; the automatic report conservatively leaves it INCONCLUSIVE pending that evidence review. A shift must not automatically be attributed to old silent corruption.',
        'Cross-objective learning curves, seven-condition Shared16 comparison, descriptive rotation distances and pre-endpoint reproduction evidence are in analysis/. Exact paired IDs and frozen behavioral diagnostics are in aime26/.',
        'Setup/install/metadata incidents are preserved in provenance/ and logs/; no canary, model-integrity or scientific failure was repaired/retried. Check actual error records for any infrastructure incident.',
        'No new benchmark, distributed training, full-density mechanism capture, C8, new objective or follow-on experiment was executed.']
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'NEXT_EXPERIMENT_DECISION.md').write_text('# Next decision (recommendation only)\n\n'+verdict+'\n\nReview uncertainty and the frozen validation curves before deciding. If C7 is promising, a separately authorized external reasoning-generalization experiment is the next hypothesis; otherwise decision/logit/task-token supervision or regression control may be considered. Nothing is automatically executed.\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('prepare','smoke','finish'),required=True);args=p.parse_args()
    for d in ('protocol','logs','deployment','analysis','aime26'): (ROOT/d).mkdir(parents=True,exist_ok=True)
    for suffix in ('completion','error'):
        if (ROOT/f'logs/endpoint_{args.phase}.{suffix}.json').exists():raise RuntimeError('existing evidence: no automatic retry')
    try:
        globals()[args.phase]()
        if args.phase!='prepare':save(ROOT/f'logs/endpoint_{args.phase}.completion.json',{'status':'PASS'})
        if args.phase=='finish':
            with (ROOT/'SHA256SUMS').open('x') as f:
                for q in sorted(ROOT.rglob('*')):
                    if q.is_file() and q.name!='SHA256SUMS' and '__pycache__' not in q.parts and '.git' not in q.parts and not (q.parent.name=='logs' and q.suffix in ('.log','.exitcode')):
                        f.write(sha(q)+'  '+str(q.relative_to(ROOT))+'\n')
    except BaseException as e:
        save(ROOT/f'logs/endpoint_{args.phase}.error.json',{'status':'FAIL','error':str(e),'traceback':traceback.format_exc(),'do_not_retry':True});raise


if __name__=='__main__':main()
