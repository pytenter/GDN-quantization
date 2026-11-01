#!/usr/bin/env python3
"""Offline exact Frozen V4 scoring; no model forward or GPU execution.

No scientific CSV/JSONL is published until all120 frozen paired generations,
their immutable source/Plan/input identities and decoded text pass validation.
Incomplete/failed admission publishes only SCORING_STATUS.json.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

TASK = 'QWEN_GDN_AIME26_STEPQUANT6_VS_H_DUALM_2SEED_V1'
METHODS = ('STEPQUANT6', 'H_DUALM')
LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]
SCORER_SHA = 'fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b'
SCORER_REL = 'protocol_recovery/snapshots/' + SCORER_SHA + '_aime26_scorer_v4.py'
CAP = 81920
VOCAB = 248320
OUTPUTS = ('GENERATION_RESULTS.jsonl','PER_PROBLEM_RESULTS.csv','PER_SEED_RESULTS.csv','ACCURACY_SUMMARY.json')


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes().decode('utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',',':'), allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def atomic_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('xb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    try:
        os.link(tmp,path)
    finally:
        tmp.unlink()


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def verify_source_manifest(root, name, expected):
    p = root/name
    need(sha(p)==expected,'Frozen source manifest SHA mismatch: '+name)
    value = read(p)
    rows = value.get('execution_sources',value.get('files',[]))+value.get('external_dependencies',[])
    need(isinstance(rows,list) and rows,'Empty or non-list source manifest')
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute():
            path = root/path
            need(path.resolve().is_relative_to(root.resolve()),'Relative source path escapes root')
        need(path.is_file() and sha(path)==row['sha256'],'Current frozen source mismatch: '+str(path))
        if 'bytes' in row:
            need(path.stat().st_size==row['bytes'],'Source byte count mismatch')


def validate_units(value, gpu_map):
    units = value['units']
    need(len(units)==60 and [u['canonical_index'] for u in units]==list(range(60)),'Exactly60 canonical paired units required')
    need(len({(u['problem_id'],u['seed']) for u in units})==60,'Duplicate problem/seed unit')
    need(len({u['problem_id'] for u in units})==30,'Exactly30 distinct problems required')
    for i,u in enumerate(units):
        need(u['problem_idx']==i//2+1 and u['seed']==[1,2][i%2],'Frozen original problem/seed ordering changed')
        need(u['gpu']==i%4 and u['gpu_uuid']==gpu_map[str(i%4)],'Frozen unit GPU assignment changed')
        need(u['method_order']==(list(METHODS) if (i//4)%2==0 else list(reversed(METHODS))),'Frozen alternating method order changed')
        need(is_sha(u['input_ids_hash']),'Missing canonical prompt token SHA')
    return units


def expected_generation_config(seed):
    return dict(enable_thinking=True,do_sample=True,temperature=1.,top_p=.95,top_k=20,min_p=0.,
        presence_penalty=1.5,repetition_penalty=1.,presence_penalty_scope='generated_output_tokens_only',
        max_new_tokens=CAP,sampling_seed=seed)


def validate_record(row, unit, method, bindings, stops, expected_assets=None):
    need(row.get('status')=='COMPLETE' and row.get('runtime_error') is None and row.get('nonfinite') is False,
         'Technical failure is INCOMPLETE, never a scientific wrong answer')
    need(row.get('task')==TASK and row.get('method')==method,'Row task/method mismatch')
    for key in ('problem_id','problem_idx','canonical_index','seed','input_ids_hash'):
        need(row.get(key)==unit[key],'Row frozen identity mismatch: '+key)
    need(row.get('problem')==unit['problem'] and str(row.get('gold_answer'))==str(unit['answer']),'Original problem/gold changed')
    need(row.get('raw_messages')==[dict(role='user',content=unit['problem'])],'Prompt messages changed')
    need(row.get('input_tokens')==unit['input_tokens'],'Prompt token length changed')
    gpu = row.get('gpu',{})
    need(gpu.get('physical_index')==unit['gpu'] and gpu.get('UUID')==unit['gpu_uuid'] and gpu.get('logical_index')==0,
         'Pair physical GPU/UUID changed')
    need('3090' in gpu.get('name',''),'Unexpected GPU class')
    need(type(row.get('retry_count')) is int and row.get('retry_count')==0,'Answer-based or replacement retry forbidden')
    identity = dict(task=TASK,model='Qwen3.5-9B',configuration=method,problem_id=unit['problem_id'],seed=unit['seed'])
    need(row.get('formal_identity')==identity,'Formal unit identity mismatch')
    for key,value in bindings.items():
        need(row.get(key)==value,'Row frozen binding mismatch: '+key)
    need(row.get('correct') is None and row.get('extracted_answer') is None and
         row.get('scoring_status')=='DEFERRED_FROZEN_V4_UNTIL_ALL120_LEGAL','Worker performed unauthorized/simple scoring')
    need(row.get('generation_config')==expected_generation_config(unit['seed']),'Actual generation configuration changed')
    tokens = row.get('generated_token_ids')
    need(isinstance(tokens,list) and 1<=len(tokens)<=CAP and all(type(t) is int and 0<=t<VOCAB for t in tokens),
         'Invalid generated token vector')
    n = len(tokens)
    need(row.get('generated_token_count')==n and row.get('output_tokens')==n,'Generated token length mismatch')
    need(row.get('stop_token_ids')==sorted(stops),'EOS/stop token contract changed')
    reason = row.get('finish_reason')
    need(reason in ('eos','length'),'Invalid scientific completion reason')
    need(not any(t in stops for t in tokens[:-1]),'Generation continued after original stop token')
    need((reason=='eos' and tokens[-1] in stops) or (reason=='length' and n==CAP and tokens[-1] not in stops),
         'EOS/length-cap completion is inconsistent')
    need(isinstance(row.get('response'),str),'Missing decoded response')
    audit = row.get('sampling_audit',{})
    m = audit.get('max_survivors')
    need(type(m) is int and 1<=m<=VOCAB and audit.get('audit_version')=='HF_TOPK_TIE_AWARE_V2'
         and audit.get('implementation')=='transformers.TopKLogitsWarper' and audit.get('requested_top_k')==20
         and audit.get('tie_policy')=='include_scores_equal_to_kth_cutoff' and audit.get('top_k_applied') is True
         and audit.get('tie_expansion_observed')==(m>20),'Native HF tie-aware sampling audit failed')
    q = row.get('quantizer_audit',{})
    need(q.get('calls')==n and q.get('prefill_calls')==1 and q.get('decode_calls')==n-1 and q.get('prefill_quantized') is True,
         'Prefill/per-token persistent QDQ count mismatch')
    need(q.get('layer_writebacks')==24*n and q.get('next_consumption_checks')==24*(n-1)
         and q.get('persistent_recurrent_tensor_bytes')==50331648,'QDQ/next-kernel state-consumption audit failed')
    need(q.get('cache_representation')=='FP32 reconstructed QDQ carrier; no unquantized shadow',
         'Packed INT8 versus FP32 carrier semantics changed')
    h = row.get('hadamard_audit',{})
    if method=='H_DUALM':
        need(h.get('calls')==24*n and h.get('layers')==LAYERS and
             h.get('phase_counts')==dict(chunk_prefill=24,recurrent_decode=24*(n-1)),'H128 transform boundary/count mismatch')
    else:
        need(h.get('calls')==0 and h.get('layers')==[],'Unexpected non-native STEPQuant rotation')
    for key in ('initial_generator_state_sha256','final_generator_state_sha256'):
        need(is_sha(row.get(key)),'Missing generator state identity')
    need(isinstance(row.get('latency_s'),(float,int)) and math.isfinite(row['latency_s']) and row['latency_s']>0,'Invalid generation wall time')
    need(type(row.get('peak_allocated_bytes')) is int and type(row.get('peak_reserved_bytes')) is int and
         0<=row['peak_allocated_bytes']<=row['peak_reserved_bytes'],'Invalid CUDA memory record')
    need(expected_assets is not None,'Expected model/tokenizer/codec content identity required')
    aliases=dict(physical_gpu_uuid=gpu['UUID'],prompt_token_sha=row['input_ids_hash'],generated_text=row['response'],
        wall_time=row['latency_s'],peak_cuda_allocated=row['peak_allocated_bytes'],peak_cuda_reserved=row['peak_reserved_bytes'],
        sampling_config=row['generation_config'],top_k_audit=row['sampling_audit'],q_d_q_writeback_audit=row['quantizer_audit'],
        model_sha=expected_assets['model_sha'],tokenizer_sha=expected_assets['tokenizer_sha'],
        codec_or_plan_sha=expected_assets[method],protocol_sha=bindings['prereg_sha256'])
    for key,value in aliases.items():need(row.get(key)==value,'Required result metadata alias mismatch: '+key)
    payload = dict(row); saved = payload.pop('output_payload_sha256',None)
    need(is_sha(saved) and canonical_sha(payload)==saved,'Raw result canonical payload SHA mismatch')
    return row


def load_exact_scorer(root):
    path = root/SCORER_REL
    need(sha(path)==SCORER_SHA,'Frozen V4 source was changed or substituted')
    spec = importlib.util.spec_from_file_location('exact_frozen_aime26_v4_finisher',path)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    need(module.SCORER_NAME=='AIME26_STRICT_V4_CANDIDATE' and module.STATUS=='CANDIDATE',
         'Original candidate-named Frozen V4 identity changed')
    return module


def complete_inputs(root, authorization):
    root = root.resolve()
    need(authorization.resolve()==root/'AUTHORIZATION_FORMAL.json','Explicit final formal authorization file required')
    auth = read(authorization)
    need(auth.get('task_id')==TASK and auth.get('status')=='PASS' and 'formal' in auth.get('allowed_modes',[]),'No formal authorization')
    prereg = read(root/'PREREGISTRATION.json')
    need(prereg.get('TASK_ID')==TASK and prereg.get('frozen') is True and sha(root/'PREREGISTRATION.json')==auth['prereg_sha256'],
         'Preregistration is not the frozen admitted plan')
    verify_source_manifest(root,auth.get('source_manifest','SOURCE_MANIFEST.json'),auth['source_manifest_sha256'])
    verify_source_manifest(root,'RUNTIME_SOURCE_MANIFEST.json',auth['runtime_source_manifest_sha256'])
    need(prereg.get('SOURCE_MANIFEST_sha256')==auth['source_manifest_sha256'],'Preregistered source binding changed')
    need(sha(root/'FROZEN_UNITS.json')==auth['units_sha256'],'Frozen units file SHA changed')
    units = validate_units(read(root/'FROZEN_UNITS.json'),auth['gpu_uuid_map'])
    identity = read(root/'FROZEN_AIME26_IDENTITY.json')
    need(sha(root/'FROZEN_AIME26_IDENTITY.json')==prereg['FROZEN_AIME26_IDENTITY_sha256'],'Frozen AIME identity SHA changed')
    need(identity['units']['SHA256']==auth['units_sha256'] and identity['FrozenV4']['SHA256']==SCORER_SHA,'Unit/scorer identity mismatch')
    model_path = root/'MODEL_TOKENIZER_FRESH_IDENTITY.json'
    need(sha(model_path)==auth['model_tokenizer_identity_sha256'],'Model/tokenizer identity receipt changed')
    need(identity['model']['fresh_identity_receipt_SHA256']==auth['model_tokenizer_identity_sha256'],
         'AIME and runtime model/tokenizer identity receipts differ')
    model = read(model_path)
    need(model.get('status')=='PASS' and len(model.get('files',[]))==17 and all(x.get('PASS') for x in model['files']),
         'Model/tokenizer identity not qualified')
    model_files={Path(x['path']).name:x['sha256'] for x in model['files']
                 if Path(x['path']).name=='config.json' or Path(x['path']).name.startswith('model')}
    tokenizer_files={Path(x['path']).name:x['sha256'] for x in model['files']
                 if Path(x['path']).name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja')}
    need(len(model_files)==6 and len(tokenizer_files)==5,'Model/tokenizer content digest file roster changed')
    assets=dict(model_sha=canonical_sha(model_files),tokenizer_sha=canonical_sha(tokenizer_files),
        STEPQUANT6=auth['plan_sha256'],H_DUALM=sha(root/'H_DUAL_M_IDENTITY.json'))
    need(sha(root/'calibration/qwen35_9b_stepquant6_plan.pt')==auth['plan_sha256'],'Full6 Plan bytes changed')
    plan_gate = read(root/'calibration/PLAN_CPU_QUALIFICATION.json')
    need(plan_gate.get('status')=='PASS' and plan_gate.get('plan_sha256')==auth['plan_sha256'],'Full6 Plan qualification missing')
    for name,digest in auth.get('required_gate_receipts',{}).items():
        need(sha(root/name)==digest and read(root/name).get('status')=='PASS','Required qualification gate failed: '+name)
    need(not (root/'FIRST_FAILURE.json').exists(),'Task first failure is present; no scoring admission')
    need(not list((root/'formal').rglob('FIRST_FAILURE.json')),'Formal worker failure remains INCOMPLETE')
    ledger = read(root/'GPU_RESOURCE_LEDGER.json')
    need(ledger.get('all_GPUs_released') is True and ledger.get('own_NVML_pids_remaining')==[],
         'GPU release not independently verified before CPU scoring')
    bindings = {key:auth[key] for key in ('prereg_sha256','plan_sha256','source_manifest_sha256',
        'runtime_source_manifest_sha256','model_tokenizer_identity_sha256')}
    bindings['authorization_sha256'] = sha(authorization)
    bindings['units_sha256'] = auth['units_sha256']
    for gpu in range(4):
        phase = read(root/'formal'/('worker'+str(gpu))/'RESULT.json')
        need(phase.get('status')=='PASS' and phase.get('mode')=='formal' and phase.get('units')==30,
             'Worker did not finish its30 formal generations')
        need(phase.get('gpu',{}).get('physical_index')==gpu and
             phase.get('gpu',{}).get('UUID')==auth['gpu_uuid_map'][str(gpu)],'Worker completion GPU changed')
        for key,value in bindings.items():need(phase.get(key)==value,'Worker completion identity changed: '+key)
    rows = {}
    paths = {}
    expected_paths = set()
    for unit in units:
        for method in METHODS:
            path = root/'formal'/('worker'+str(unit['gpu']))/(str(unit['canonical_index']).zfill(2)+'_'+method+'.json')
            expected_paths.add(path.resolve())
            need(path.is_file() and not path.is_symlink(),'Missing complete formal unit: '+str(path.relative_to(root)))
            key = (unit['canonical_index'],method)
            rows[key] = validate_record(read(path),unit,method,bindings,set(identity['tokenizer']['stop_ids']),assets)
            paths[key] = path
        left,right = (rows[(unit['canonical_index'],method)] for method in METHODS)
        need(left['initial_generator_state_sha256']==right['initial_generator_state_sha256'],
             'Matched methods did not start the same independent generator state')
    actual_paths = {p.resolve() for p in (root/'formal').glob('worker*/*.json')
                    if any(p.name.endswith('_'+m+'.json') for m in METHODS)}
    need(actual_paths==expected_paths and len(rows)==120,'Extra, duplicate or missing formal result files')
    return units,rows,paths,identity,model,bindings,ledger


def validate_decoded_text(rows, model, identity):
    # This runs only after all120 unit guards. It loads a tokenizer, never a model.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    for item in identity['tokenizer']['identity_files']:
        need(sha(item['path'])==item['sha256'],'Tokenizer current bytes differ from frozen identity')
    sys.path.insert(0,'/data/zypan/transformers-qwen35/src')
    import torch
    need(not torch.cuda.is_initialized(),'CPU scoring found initialized CUDA')
    torch.cuda._lazy_init=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('CUDA forbidden in offline scoring'))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(identity['model']['path'],trust_remote_code=True,local_files_only=True)
    for row in rows.values():
        need(tok.decode(row['generated_token_ids'],skip_special_tokens=True)==row['response'],'Saved response is not exact frozen tokenizer decode(ids)')
    need(not torch.cuda.is_initialized(),'Offline scoring initialized CUDA')


def csv_bytes(rows, fields):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def score_complete(root, authorization):
    need(not (root/'SCORING_STATUS.json').exists(),'Prior scoring status exists; no automatic scoring retry')
    for name in OUTPUTS:
        need(not (root/name).exists(),'Refuse replacing scientific result: '+name)
    units,rows,paths,identity,model,bindings,ledger=complete_inputs(root,authorization)
    validate_decoded_text(rows,model,identity)
    scorer=load_exact_scorer(root)
    scored={};jsonl=[];raw_catalog=[]
    for unit in units:
        for method in METHODS:
            key=(unit['canonical_index'],method);row=rows[key]
            extraction=scorer.extract_v4(row['response'],row['finish_reason'],unit['problem'])
            correct=scorer.compare_with_gold(extraction,unit['answer'])
            scored[key]=dict(correct=bool(correct),extraction=extraction.to_dict())
            record=dict(row);record['frozen_v4_scoring']=dict(scored[key],source_sha256=SCORER_SHA)
            jsonl.append(json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False))
            raw_catalog.append(dict(path=str(paths[key].relative_to(root)),sha256=sha(paths[key])))
    per_problem=[]
    for problem_idx in range(1,31):
        us=[u for u in units if u['problem_idx']==problem_idx]
        rec=dict(problem_idx=problem_idx,problem_id=us[0]['problem_id'])
        for u in us:
            seed=u['seed'];a=scored[(u['canonical_index'],METHODS[0])]['correct'];h=scored[(u['canonical_index'],METHODS[1])]['correct']
            rec.update({f'STEPQUANT6_seed{seed}_correct':int(a),f'H_DUALM_seed{seed}_correct':int(h),
                f'seed{seed}_paired_difference':int(a)-int(h),f'seed{seed}_disagreement':int(a!=h)})
        per_problem.append(rec)
    per_seed=[];method_summary={}
    for method in METHODS:
        for seed in (1,2):
            count=sum(scored[(u['canonical_index'],method)]['correct'] for u in units if u['seed']==seed)
            per_seed.append(dict(method=method,seed=seed,correct=count,total=30,accuracy_percent=100*count/30))
        values=[rows[(u['canonical_index'],method)] for u in units]
        lengths=sorted(v['output_tokens'] for v in values)
        total=sum(scored[(u['canonical_index'],method)]['correct'] for u in units)
        method_summary[method]=dict(total_correct=total,total=60,accuracy_percent=100*total/60,
            seed1_correct=next(x['correct'] for x in per_seed if x['method']==method and x['seed']==1),
            seed2_correct=next(x['correct'] for x in per_seed if x['method']==method and x['seed']==2),
            length=dict(mean=statistics.fmean(lengths),median=statistics.median(lengths),p95=lengths[math.ceil(.95*len(lengths))-1],p95_rule='nearest_rank'),
            EOS=sum(v['finish_reason']=='eos' for v in values),length_truncation=sum(v['finish_reason']=='length' for v in values),
            unable_to_extract=sum(scored[(u['canonical_index'],method)]['extraction']['status']=='ABSTAIN' for u in units),
            invalid_AIME_domain=sum(scored[(u['canonical_index'],method)]['extraction']['status']=='EXTRACTED' and not scored[(u['canonical_index'],method)]['extraction']['aime_domain_valid'] for u in units),
            technical_or_numerical_failures=0,generation_wall_seconds_sum=sum(v['latency_s'] for v in values),
            wall_scope='Generation-only sum; model loading, calibration and qualification GPU hours are separately inGPU_RESOURCE_LEDGER.json',
            persistent_recurrent_tensor_bytes=50331648,
            representation='FP32 reconstructed QDQ carrier; not physical packed INT8',
            transient_encoded_layout_examples=[v['quantizer_audit'].get('transient_STEP_payload_layout') for v in values[:1]])
    a_only=h_only=both_right=both_wrong=0
    for unit in units:
        a=scored[(unit['canonical_index'],METHODS[0])]['correct'];h=scored[(unit['canonical_index'],METHODS[1])]['correct']
        a_only+=a and not h;h_only+=h and not a;both_right+=a and h;both_wrong+=not a and not h
    summary=dict(task_id=TASK,status='COMPLETE_ALL120_LEGAL_FROZEN_V4_SCORED',methods=method_summary,
        primary=dict(name='Accuracy_STEPQuant6-Accuracy_H_DualM',difference=method_summary[METHODS[0]]['total_correct']/60-method_summary[METHODS[1]]['total_correct']/60,
            percentage_points=method_summary[METHODS[0]]['accuracy_percent']-method_summary[METHODS[1]]['accuracy_percent']),
        paired_units=dict(total=60,STEPQUANT6_only_correct=a_only,H_DUALM_only_correct=h_only,both_correct=both_right,both_wrong=both_wrong,disagreements=a_only+h_only),
        uncertainty=dict(status='NOT_ESTIMATED_DESCRIPTIVE_ONLY',independent_problems=30,seeds_per_problem=2,
            note='60paired outputs are not60independent math problems; any future interval must resample30problem clusters retaining bothseeds'),
        scorer_sha256=SCORER_SHA,bindings=bindings,raw_results=raw_catalog,
        GPU_resource_ledger=dict(path='GPU_RESOURCE_LEDGER.json',sha256=sha(root/'GPU_RESOURCE_LEDGER.json'),all_GPUs_released=ledger['all_GPUs_released']),
        no_technical_failure_counted_as_scientific_wrong=True,model_forward_count=0,GPU_execution_by_finisher=False,
        historical_results='REFERENCE_ONLY; newprefill timing differs; no historical generation is merged',created_unix=time.time())
    payloads={'GENERATION_RESULTS.jsonl':('\n'.join(jsonl)+'\n').encode(),
        'PER_PROBLEM_RESULTS.csv':csv_bytes(per_problem,list(per_problem[0])),
        'PER_SEED_RESULTS.csv':csv_bytes(per_seed,['method','seed','correct','total','accuracy_percent']),
        'ACCURACY_SUMMARY.json':json_bytes(summary)}
    for name,payload in payloads.items():atomic_new(root/name,payload)
    return dict(status='COMPLETE_ALL120_LEGAL_FROZEN_V4_SCORED',outputs=[dict(path=n,sha256=sha(root/n)) for n in OUTPUTS],GPU_execution=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--authorization',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        result=score_complete(args.root.resolve(),args.authorization)
    except Exception as exc:
        result=dict(task_id=TASK,status='INCOMPLETE_OR_FAIL_CLOSED_STOP',error=str(exc),
            scientific_outputs_present=[n for n in OUTPUTS if (args.root/n).exists()],
            technical_failure_not_scored_as_wrong=True,model_forward_count=0,GPU_execution=False)
        target=args.root/'SCORING_STATUS.json'
        if not target.exists():atomic_new(target,json_bytes(result))
        print(json.dumps(result,ensure_ascii=False));return 2
    atomic_new(args.root/'SCORING_STATUS.json',json_bytes(result))
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':
    raise SystemExit(main())
