"""Frozen outcome-blind CPU selection audit; no model forward or outcome read."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OLD = Path('/data/zypan/worktrees/qwen-gdn-scale-feedback-fullmodel-functional-causal-pilot-v1/experiments/QWEN_GDN_SCALE_FEEDBACK_FULLMODEL_FUNCTIONAL_CAUSAL_PILOT_V1')
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
SOURCE = Path('/data/ydai/qwen_model_cache/datasets/long_bench/repobench-p/1.0.0/0384ffc1cabf36a2322f408bfdc21ea54d6a2704f2b1e17ca49d89035e666f1e/long_bench-test.arrow')
TASK = 'QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1'
SEED = 20261013
SKIP_KEYS = {'eligibility', 'candidates', 'candidate_pool', 'candidate_docs', 'exclusion_registry', 'source_manifests', 'near_duplicate_comparisons', 'excluded_candidates', 'all_candidates'}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def save(n, x):
    p = ROOT / n; p.parent.mkdir(exist_ok=True, parents=True)
    with p.open('x') as f: json.dump(x, f, ensure_ascii=False, sort_keys=True, indent=2); f.write('\n')

def load(p): return json.loads(Path(p).read_text())
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    start = time.time()
    spec = load(OLD/'FROZEN_SCIENCE_SPEC.json')
    assert sha(OLD/'FROZEN_SCIENCE_SPEC.json') == '1f24740272d941433afb1e4f8f616e72c3bc2dc29993b89130bdef41b8c9c200'
    assert sha(OLD/'RESULT_AND_DELIVERY_HASHES.json') == 'bdfe7f26f2ffdd083cfe1dab85cc915362a3e6d7f744bdc0e81f6782d0344a7c'
    inherited = []
    for p, s in spec['source_SHA'].items():
        actual = sha(p); inherited.append({'path':p, 'expected_SHA':s, 'actual_SHA':actual, 'status':'PASS' if actual==s else 'FAIL'})
        assert actual == s, ('SOURCE_IDENTITY_FAILURE',p)
    # Hash only the tokenizer and source files here. Weights are not used by this CPU audit.
    # A prospective GPU run would still require incremental identity/runtime qualification.
    model_stats = []
    for f in spec['model_files']:
        p = Path(f['path']); st = p.stat()
        model_stats.append({'path':str(p), 'historical_SHA':f['sha256'], 'bytes':st.st_size, 'mtime_ns':st.st_mtime_ns,
                            'matches_historical_size_mtime':st.st_size==f['bytes'] and st.st_mtime_ns==f['mtime_ns'],
                            'new_weight_content_hash_performed':False})
    old_selection = load(OLD/'DOCUMENT_SELECTION_AND_EXCLUSIONS.json')
    dataset_identity = old_selection['dataset_identity']
    assert sha(SOURCE) == dataset_identity['snapshot_SHA']
    # Inventory existing compatible files within relevant known cache roots, never acquire new data.
    cached = []; errors = []
    bases = ['/data/ydai/qwen_model_cache/datasets', '/data/zypan/.cache', '/data/zypan/artifacts', '/data/zypan/research_archive', '/data/zypan/consolidation']
    for base in bases:
        if not Path(base).exists(): continue
        for parent, dirs, names in os.walk(base, onerror=lambda e:errors.append(str(e))):
            dirs[:] = [d for d in dirs if len(Path(parent).relative_to(base).parts)<9 and d not in ['.git','__pycache__','node_modules']]
            for n in names:
                p = Path(parent)/n
                if 'repobench' in str(p).lower() and p.suffix in ['.arrow','.parquet','.jsonl']:
                    cached.append({'path':str(p),'bytes':p.stat().st_size,'SHA':sha(p)})
    assert {x['SHA'] for x in cached} == {dataset_identity['snapshot_SHA'], '83c6bde23707f190b4ba04cf5fefc9c3779b26be5aed333bc5ac98d5952b233b', '919a4439e2a84ebb25bacc39ac3b3269a7641af6e02ae205ed78d8c53dfe3568'}, ('UNAUDITED_CACHE_IDENTITY',cached)
    compatibility = load(ROOT/'ADDITIONAL_RAW_CACHE_IDENTITY_AUDIT.json')
    assert next(x for x in compatibility['extra_cached_files'] if x['path'].endswith('/repobench-p.jsonl'))['novel_context_count'] == 0
    sys.path.insert(0, '/data/zypan/transformers-qwen35')
    from datasets import Dataset
    from transformers import AutoTokenizer
    data = Dataset.from_file(str(SOURCE))
    records = [(i, row, str(SOURCE), 'repobench-p') for i,row in enumerate(data)]
    extra_file = next(x for x in compatibility['extra_cached_files'] if x['path'].endswith('/repobench-p_e.jsonl'))
    extra_rows = [json.loads(line) for line in Path(extra_file['path']).read_text().splitlines() if line.strip()]
    assert len(extra_rows) == extra_file['count'] and set(extra_rows[0]) == set(data.column_names)
    assert {x['dataset'] for x in extra_rows} == {'repobench-p_e'} and {x['language'] for x in extra_rows} <= {'python','java'}
    records += [(i,row,extra_file['path'],'repobench-p_e') for i,row in enumerate(extra_rows)]
    candidates = []; lookup = {}; raw_seen = {}; exact_pool_duplicates = []
    for i, row, source_file, source_variant in records:
        uid = 'long_bench/repobench-p/test/'+str(row['_id'])
        text = row['context']; raw = hashlib.sha256(text.encode()).hexdigest()
        project = next((str(row[k]) for k in ['repository','repo','project','project_id','repository_id'] if k in row and row[k]), 'UNKNOWN_NOT_IN_SNAPSHOT_SCHEMA')
        if raw in raw_seen:
            exact_pool_duplicates.append({'source_file':source_file,'row':i,'document_id':uid,'duplicate_of':raw_seen[raw],'raw_SHA':raw});continue
        assert str(row['_id']) not in lookup, ('ID_CONTEXT_COLLISION',uid)
        raw_seen[raw] = uid
        c = {'document_id':uid,'dataset_id':str(row['_id']),'row':i,'source_file':source_file,'source_variant':source_variant,'raw_text_SHA':raw,'repository_project_identity':project,
             'language':row.get('language'),'text':text}
        candidates.append(c); lookup[str(row['_id'])]=c
    excluded_ids = set(); excluded_raw = set(); sources = []; skipped = []
    registry = old_selection['exclusion_registry']
    # Retain the historical registry only after rechecking every source manifest SHA.
    verified_registry = []
    for f in registry['source_manifests']:
        actual = sha(f['path']); assert actual==f['sha256'], ('HISTORICAL_REGISTRY_IDENTITY_FAILURE',f['path'])
        verified_registry.append({'path':f['path'],'SHA':actual,'status':'PASS','identified_exclusion_count':len(f['identified_exclusions'])})
    excluded_ids.update(registry['ids']); excluded_raw.update(registry['raw_SHA'])
    def extract(obj):
        found = []
        def walk(v, key=''):
            if isinstance(v,dict):
                for k,x in v.items():
                    if str(k).lower() not in SKIP_KEYS: walk(x,str(k).lower())
            elif isinstance(v,list):
                for x in v: walk(x,key)
            elif isinstance(v,str):
                if v.startswith('long_bench/repobench-p/test/'):
                    excluded_ids.add(v); found.append({'document_id':v})
                elif key in ['document_id','canonical_doc_id','dataset_document_id','_id'] and v in lookup:
                    excluded_ids.add(lookup[v]['document_id']); found.append({'document_id':lookup[v]['document_id']})
                if key in ['raw_text_sha256','raw_text_sha','text_sha256','content_sha256','text_sha','document_raw_sha'] and re.fullmatch('[0-9a-f]{64}',v):
                    excluded_raw.add(v);found.append({'raw_text_SHA':v})
        walk(obj); return found
    for base in ['/data/zypan/worktrees','/data/zypan/results','/data/zypan/GDN-quantization']:
        for parent,dirs,names in os.walk(base,onerror=lambda e:errors.append(str(e))):
            dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','node_modules'] and len(Path(parent).relative_to(base).parts)<10]
            if REPO==Path(parent) or REPO in Path(parent).parents: dirs[:]=[];continue
            for n in names:
                p=Path(parent)/n; lower=n.lower()
                if p.suffix!='.json' or any(k in lower for k in ['candidate_pool','eligible_document','corpus_inventory','frozen_cumsum','tokens.json']): continue
                if not any(k in lower for k in ['document','manifest','frozen','panel','audit','identity','checkpoint']):continue
                if p.stat().st_size>20*1024**2:
                    skipped.append({'path':str(p),'reason':'large metadata; used explicit selected manifests'});continue
                try: obj=load(p)
                except (ValueError,UnicodeError):continue
                hits=extract(obj)
                if hits: sources.append({'path':str(p),'SHA':sha(p),'identified_exclusions':hits})
    # Explicitly assert both actual old panels were discovered, not candidates from inventories.
    v3=Path('/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/SCIENCE_PREMEASUREMENT_SPEC.json')
    for d in load(v3)['documents']:
        assert d['canonical_doc_id'] in excluded_ids
    for d in spec['documents']:
        assert d['document_id'] in excluded_ids and d['raw_text_SHA'] in excluded_raw
    reg={'status':'AUDITED','utc':utc(),'rule':'actual selected/used identities; candidate/eligibility inventories excluded from discovery',
         'ids':sorted(excluded_ids),'raw_SHA':sorted(excluded_raw),'prior_registry_source':str(OLD/'DOCUMENT_SELECTION_AND_EXCLUSIONS.json'),
         'prior_registry_container_SHA':sha(OLD/'DOCUMENT_SELECTION_AND_EXCLUSIONS.json'),'prior_registry_verified_sources':verified_registry,
         'source_manifests':sources,'skipped':skipped,'scan_errors':errors}
    save('EXCLUSION_REGISTRY.json',reg)
    files=[{'name':n,'sha256':sha(MODEL/n)} for n in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']]
    tokenizer_sha=digest(files); assert tokenizer_sha==old_selection['tokenizer_identity']['SHA']
    pool=[{k:v for k,v in c.items() if k!='text'} for c in candidates]
    save('CANDIDATE_POOL_BEFORE_TOKENIZATION.json',{'status':'FROZEN_OUTCOME_BLIND_POOL','dataset_identity':dataset_identity,'schema':str(data.features),'candidates':pool,'count':len(pool),'exact_duplicate_source_rows':exact_pool_duplicates,'compatible_raw_cache_audit_SHA':sha(ROOT/'ADDITIONAL_RAW_CACHE_IDENTITY_AUDIT.json'),'source_catalog':cached,'compatibility_basis':'existing same-schema RepoBench-P task variants; primary JSONL exact equivalent; _e additional contexts admitted before tokenizer/outcome; authoritative project identity absent'})
    rules={'status':'FROZEN_BEFORE_TOKENIZATION','utc':utc(),'seed':SEED,'task':TASK,'requested_N':8,
           'source_SHA':dataset_identity['snapshot_SHA'],'source_catalog_SHA':digest(cached),'tokenizer_SHA':tokenizer_sha,'exclusion_registry_SHA':sha(ROOT/'EXCLUSION_REGISTRY.json'),
           'candidate_pool_SHA':sha(ROOT/'CANDIDATE_POOL_BEFORE_TOKENIZATION.json'),'context_only':True,'add_special_tokens':False,'chat_template':False,'minimum_tokens':32769,
           'rank':'SHA256(task|seed|canonical_document_id|raw_text_SHA)','exact_duplicate':'raw text or first32769 token sequence equal, or raw text containment',
           'near_duplicate':'aligned64 token-shingle Jaccard >=0.2 against historical used and new accepted documents',
           'repository_rule':'one per known repository/project; unknown metadata explicit correlation limitation; no inferred repository ID',
           'no_outcome_features':['G','KL','activation','benefit','S1','S2'],'stop_if_fewer_than8':True,
           'code_commit_before_execution':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    save('DOCUMENT_SELECTION_RULE_BEFORE_TOKENIZATION.json',rules)
    print('POOL_AND_RULES_FROZEN',len(pool),len(excluded_ids),len(excluded_raw),flush=True)
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
    audit=[]; historical=[]; ranked=[]
    for number,c in enumerate(candidates,1):
        ids=tok.encode(c['text'],add_special_tokens=False)
        entry={k:v for k,v in c.items() if k!='text'};entry['token_count']=len(ids)
        excluded=c['document_id'] in excluded_ids or c['raw_text_SHA'] in excluded_raw
        entry['status']='EXCLUDED_PREVIOUS_RELATED_OUTCOME' if excluded else ('INSUFFICIENT_LENGTH' if len(ids)<32769 else 'LENGTH_ELIGIBLE')
        audit.append(entry)
        if excluded or len(ids)>=32769:
            c['ids']=ids[:32769]
            c['shingles']={digest(ids[p:p+64]) for p in range(0,min(len(ids),32769)-63,64)}
            if excluded: historical.append(c)
            elif len(ids)>=32769:
                c['selection_key']=hashlib.sha256(f"{TASK}|{SEED}|{c['document_id']}|{c['raw_text_SHA']}".encode()).hexdigest(); ranked.append(c)
        if number%100==0:print('TOKENIZED_ROWS',number,flush=True)
    ranked.sort(key=lambda c:(c['selection_key'],c['document_id']));accepted=[];comparisons=[]
    for c in ranked:
        rejected=[]
        for other in historical+accepted:
            union=c['shingles']|other['shingles'];j=len(c['shingles']&other['shingles'])/len(union) if union else 0
            exact=c['text']==other['text'] or c['ids']==other['ids'] or c['text'] in other['text'] or other['text'] in c['text']
            comparisons.append({'a':c['document_id'],'b':other['document_id'],'aligned64token_Jaccard':j,'exact_or_containment':exact})
            if exact or j>=.2:rejected.append(other['document_id'])
        known=c['repository_project_identity']!='UNKNOWN_NOT_IN_SNAPSHOT_SCHEMA'
        if known and any(x['repository_project_identity']==c['repository_project_identity'] for x in accepted):rejected.append('SAME_KNOWN_PROJECT')
        c['dedup_status']='REJECTED' if rejected else 'PASS_TEXT_TOKEN_DEDUP_PROJECT_UNKNOWN' if not known else 'PASS'
        c['rejection_matches']=rejected
        if not rejected:accepted.append(c)
    status='PASS' if len(accepted)>=8 else 'DOCUMENT_SELECTION_INSUFFICIENT_STOP'
    def metadata(c):
        x={k:v for k,v in c.items() if k not in ['text','ids','shingles']}
        x['input_count']=32769;x['token_sequence_canonical_SHA']=digest(c['ids']);x['tokenizer_SHA']=tokenizer_sha
        return x
    result={'status':status,'utc':utc(),'dataset_identity':dataset_identity,'dataset_schema':str(data.features),'existing_cache_inventory':cached,'cache_search_bases':bases,'source_rows_before_raw_dedup':len(records),'exact_pool_duplicate_rows':exact_pool_duplicates,'compatible_raw_cache_audit_SHA':sha(ROOT/'ADDITIONAL_RAW_CACHE_IDENTITY_AUDIT.json'),
            'tokenizer_identity':dict(old_selection['tokenizer_identity'],verified_SHA=tokenizer_sha),'requested_N':8,'candidate_count':len(candidates),
            'length_eligible_before_history_exclusions':sum(x['token_count']>=32769 for x in audit),
            'new_length_candidates':len(ranked),'accepted_after_all_available_dedup_rules':len(accepted),
            'project_independence_verified':False if any(x['repository_project_identity']=='UNKNOWN_NOT_IN_SNAPSHOT_SCHEMA' for x in accepted) else True,
            'outcome_blind':True,'counts_by_status':dict(collections.Counter(x['status'] for x in audit)),'all_rows':audit,
            'ranked_new_candidates':[metadata(c) for c in ranked],'near_duplicate_comparisons':comparisons,
            'historical_comparison_document_count':len(historical),'exclusion_registry_SHA':sha(ROOT/'EXCLUSION_REGISTRY.json'),
            'selection_rules_SHA':sha(ROOT/'DOCUMENT_SELECTION_RULE_BEFORE_TOKENIZATION.json'),'new_GPU_forward_count':0}
    save('DATASET_SELECTION_AUDIT.json',result)
    save('FROZEN_8DOC_SELECTION.json',{'status':status,'requested_N':8,'selected_documents':[metadata(c) for c in accepted[:8]] if status=='PASS' else [],
         'audited_remaining_candidates':[metadata(c) for c in accepted],'candidate_count_upper_bound':len(ranked),
         'no_partial_panel_execution':True,'audit_SHA':sha(ROOT/'DATASET_SELECTION_AUDIT.json'),'exclusion_registry_SHA':sha(ROOT/'EXCLUSION_REGISTRY.json'),
         'seed':SEED,'dataset_identity':dataset_identity,'tokenizer_SHA':tokenizer_sha})
    if status!='PASS':
        save('FIRST_FAILURE_GATE.json',{'status':status,'first_failure_gate':'G1_DOCUMENT_SELECTION','utc':utc(),'required':8,'available_upper_bound':len(ranked),
             'accepted_after_available_dedup_rules':len(accepted),'repository_identity_limitation':'snapshot has no authoritative repository/project fields',
             'new_outcomes_produced':False,'new_GPU_forward_count':0,'no_retry_or_rule_relaxation':True,'P2_P3_P4':'NOT_EXECUTED'})
    save('SOURCE_AND_RUNTIME_IDENTITY.json',{'status':'SOURCE_SHA_AUDIT_PASS_NEW_RUNTIME_NOT_QUALIFIED','base_commit':spec['code_commit'],
         'historical_spec_SHA':sha(OLD/'FROZEN_SCIENCE_SPEC.json'),'historical_seal_SHA':sha(OLD/'RESULT_AND_DELIVERY_HASHES.json'),
         'source_files':inherited,'tokenizer_identity':old_selection['tokenizer_identity'],'model_files_stat_audit':model_stats,
         'model_SHA_historical':spec['model_SHA'],'rotation_SHA_historical':spec['C6R_rotation_SHA'],'quantizer_SHA_historical':spec['quantizer_SHA'],
         'historical_runtime':spec['runtime'],'historical_runtime_identity':spec['runtime_identity'],'new_runtime_qualification':'NOT_RUN_SELECTION_STOP',
         'weight_content_SHA_new_audit':'NOT_NEEDED_NO_FORMAL_FORWARD','new_source_SHA':{'selection_audit_attempt002.py':sha(ROOT/'selection_audit_attempt002.py'),'selection_audit_attempt001.py':sha(ROOT/'selection_audit.py'),'historical_temporal.py':sha(ROOT/'historical_temporal.py')},
         'old_FAIL_STOP_history_retained_in_original_paths':True})
    save('P1_EXECUTION_RECEIPT.json',{'status':status,'CPU_wall_seconds':time.time()-start,'GPUh':0,'new_GPU_forward_count':0,'completed_rows':len(audit),'utc':utc()})
    print('SELECTION_AUDIT_COMPLETE',json.dumps({k:result[k] for k in ['status','candidate_count','length_eligible_before_history_exclusions','new_length_candidates','accepted_after_all_available_dedup_rules','counts_by_status']}),flush=True)

if __name__=='__main__':main()
