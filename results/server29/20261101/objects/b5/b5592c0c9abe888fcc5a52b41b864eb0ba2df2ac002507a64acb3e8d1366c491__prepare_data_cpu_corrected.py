"""Bounded, outcome-blind CPU isolation of new paired-training and future data.

Only new-task files are written. Original preprocessing is preserved; the
explicitly authorized new raw-data source is cached RepoBench rather than Wiki.
"""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent
CFG = json.loads((DATA / 'SELECTION_CONFIG.json').read_text())
SEED = 20261017
START = time.time()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    p = Path(p)
    a = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(16*1024**2), b''):
            h.update(x)
    z = p.stat()
    need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns), 'Source changed during hash: '+str(p))
    return h.hexdigest()


def canon(x):
    return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()


def text_sha(x):
    return hashlib.sha256(x.encode()).hexdigest()


def put(p,x):
    with Path(p).open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def event(stage,**kw):
    print(json.dumps(dict(stage=stage,elapsed_seconds=time.time()-START,**kw),ensure_ascii=False),flush=True)


def packed(ids):
    return struct.pack('<'+str(len(ids))+'I',*ids)


def contains(needle,haystack):
    pos=haystack.find(needle)
    while pos>=0:
        if pos%4==0:
            return True
        pos=haystack.find(needle,pos+1)
    return False


def run():
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only visible devices must be empty')
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('CPU data audit exceeded1800seconds')))
    signal.alarm(1800)
    rule=read(DATA/'SELECTION_RULE.json')
    need(rule['status']=='FROZEN_BEFORE_NEW_CANDIDATE_TOKENIZATION' and rule['seed']==SEED,'Prewritten selection rule missing')
    rule_sha=sha(DATA/'SELECTION_RULE.json')
    freeze=read(DATA/'CORRECTION_SOURCE_FREEZE.json')
    for item in freeze['execution_sources']:need(sha(DATA/item['path'])==item['sha256'],'Correction frozen source changed: '+item['path'])
    ids=set(CFG['known_Ling_ids']);raws=set();token_hashes=set();sources=[]
    def collect(x):
        if isinstance(x,list):
            for a in x:collect(a)
        elif isinstance(x,dict):
            for key,value in x.items():
                name=key.lower()
                if isinstance(value,str):
                    if name in ['document_id','doc_id','sample_id']:
                        ids.add(value)
                    if len(value)==64 and all(c in '0123456789abcdef' for c in value):
                        if 'raw' in name and ('sha' in name or 'hash' in name):raws.add(value)
                        elif any(s in name for s in ['token','input','ids']) and any(s in name for s in ['sha','hash']):token_hashes.add(value)
                collect(value)
    old=Path(CFG['latest_5doc_root'])
    paths={a['path']:a['sha256'] for a in CFG['historical_registry_sources']}
    for a in CFG['inherited_sources']:
        if not any(k in a['role'] for k in ['SCRIPT','AUDIT','SOURCE_MANIFEST']):paths.setdefault(a['path'],a['sha256'])
    for name in ['data_selection/EXCLUSION_UNION.json','frozen_documents.json']:
        p=old/name;paths[str(p)]=sha(p)
    c6=Path(CFG['C6R_reuse_root'])/'FROZEN_DOCUMENT_REUSE_MANIFEST.json'
    paths[str(c6)]=sha(c6)
    for path,expected in paths.items():
        p=Path(path)
        actual=sha(p)
        need(actual==expected,'Historical registry source SHA changed: '+path)
        x=read(p)
        correction=read(DATA/'CPU_EXCLUSION_CLASSIFICATION_CORRECTION_V1.json')
        exception=correction['exact_exception']
        if str(p)==exception['source']:
            need(actual==exception['source_SHA'],'Correction source identity changed')
            expected={r['document_id']:r for r in exception['remove_only_these_six_unselected_records']}
            selected_ids={r['document_id'] for r in x['selected_documents']}
            need(not selected_ids.intersection(expected),'Correction attempted to drop selected history')
            dropped=[r for r in x['audited_remaining_candidates'] if r['document_id'] in expected]
            need(len(dropped)==6 and {r['document_id'] for r in dropped}==set(expected),'Correction exception record count changed')
            for r in dropped:need(r['raw_text_SHA']==expected[r['document_id']]['raw_SHA'],'Correction raw identity changed')
            x=dict(x)
            x['audited_remaining_candidates']=[r for r in x['audited_remaining_candidates'] if r['document_id'] not in expected]
        collect(x)
        if p.name=='EXCLUSION_UNION.json':
            ids.update(x.get('document_ids',[]));raws.update(x.get('raw_text_sha256',[]));token_hashes.update(x.get('canonical_32768_input_sha256',[]))
        if p.name in ['EXCLUSION_REGISTRY.json','SELECTION_EXCLUSION_REGISTRY.json']:
            ids.update(x.get('ids',[]));raws.update(x.get('raw_SHA',[]))
        sources.append(dict(path=path,sha256=actual,bytes=p.stat().st_size,status='PASS_FRESH_HASH'))
    need(sha(CFG['original_corpus'])==CFG['original_corpus_SHA'],'Original96corpus SHA changed')
    corpus=[json.loads(x) for x in Path(CFG['original_corpus']).read_text().splitlines() if x.strip()]
    need(collections.Counter(x['split'] for x in corpus)=={'TRAIN':64,'VALIDATION':16,'HELDOUT':16},'Original96data contract changed')
    need(sha(CFG['aime_dataset'])==CFG['aime_dataset_SHA'],'AIME26 frozen dataset SHA changed')
    aime=[json.loads(x) for x in Path(CFG['aime_dataset']).read_text().splitlines() if x.strip()]
    need(len(aime)==30,'AIME unique problem count changed')
    sys.path.insert(0,CFG['transformers_source'])
    from transformers import AutoTokenizer
    for a in CFG['tokenizer_files']:
        need(sha(Path(CFG['tokenizer_path'])/a['name'])==a['sha256'],'Tokenizer source changed: '+a['name'])
    tokenizer=AutoTokenizer.from_pretrained(CFG['tokenizer_path'],local_files_only=True)
    observed=[];old_rows=[]
    for x in corpus:
        raw=x['raw_text'];need(text_sha(raw)==x['raw_text_sha256'],'Original Wiki raw identity changed')
        v=tokenizer(raw,add_special_tokens=False).input_ids
        ids.add(x['document_id']);raws.add(x['raw_text_sha256']);token_hashes.update([canon(v),canon(v[:1024])])
        observed.append(packed(v[:1024]));old_rows.append(dict(document_id=x['document_id'],split=x['split'],raw_SHA=x['raw_text_sha256'],full_token_SHA=canon(v),input1024_SHA=canon(v[:1024]),tokens=min(1024,len(v))))
    aime_rows=[]
    for x in aime:
        raw=x['problem'];v=tokenizer(raw,add_special_tokens=False).input_ids;raws.add(text_sha(raw));token_hashes.update([canon(v),canon(v[:1024])]);ids.add(str(x['problem_id']));aime_rows.append(dict(problem_id=x['problem_id'],raw_problem_SHA=text_sha(raw),canonical_token_SHA=canon(v),token_count=len(v)))
    # Directly inspect legal cached sources; only context text is a candidate.
    rows=[];schemas={};source_meta=[]
    for a in CFG['pool_sources']:
        p=Path(a['path']);need(p.stat().st_size==a['bytes'] and sha(p)==a['SHA'],'Cached pool source changed: '+str(p));source_meta.append(dict(path=str(p),sha256=a['SHA'],bytes=a['bytes']))
        if p.suffix!='.jsonl':continue
        values=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];schemas[p.name]=sorted(set(k for x in values for k in x))
        for n,x in enumerate(values):
            raw=x['context'];did='long_bench/repobench-p/test/'+x['_id']
            rows.append(dict(document_id=did,dataset_id=x['_id'],raw_text_sha256=text_sha(raw),raw=raw,source_file=str(p),source_file_SHA=a['SHA'],source_variant=p.stem,source_row=n,language=x.get('language')))
    import pyarrow as pa
    import pyarrow.ipc as ipc
    ap=next(Path(a['path']) for a in CFG['pool_sources'] if a['path'].endswith('.arrow'))
    with pa.memory_map(str(ap),'r') as f:arrow=ipc.open_stream(f).read_all().to_pylist()
    need({(x['_id'],text_sha(x['context'])) for x in arrow}=={(x['dataset_id'],x['raw_text_sha256']) for x in rows if x['source_variant']=='repobench-p'},'Original Arrow/JSONL cached identity no longer equivalent')
    unique={}
    for x in rows:unique.setdefault(x['raw_text_sha256'],x)
    candidates=[]
    for n,x in enumerate(unique.values(),1):
        v=tokenizer(x['raw'],add_special_tokens=False).input_ids
        x.update(full_token_count=len(v),full_token_SHA=canon(v),input1024_SHA=canon(v[:1024]),input32768_SHA=canon(v[:32768]) if len(v)>=32768 else None,input32769_SHA=canon(v[:32769]) if len(v)>=32769 else None,ids=v)
        reasons=[]
        if x['document_id'] in ids or x['dataset_id'] in ids:reasons.append('HISTORICAL_DOCUMENT_ID')
        if x['raw_text_sha256'] in raws:reasons.append('HISTORICAL_RAW_SHA')
        if any(x[k] in token_hashes for k in ['full_token_SHA','input1024_SHA','input32768_SHA','input32769_SHA'] if x[k]):reasons.append('HISTORICAL_CANONICAL_TOKEN_SHA')
        if reasons:
            observed.append(packed(v[:32769]))
        x['history_exclusions']=reasons;x['rank_key']=text_sha(str(SEED)+'|'+x['document_id']+'|'+x['raw_text_sha256']);candidates.append(x)
        if n%100==0:event('CPU_CANDIDATE_TOKENIZATION',count=n,total=len(unique))
    ranked=sorted(candidates,key=lambda x:(x['rank_key'],x['raw_text_sha256'],x['document_id']))
    reserved=[];future_shas=set()
    for x in ranked:
        if not x['history_exclusions'] and x['full_token_count']>=32769 and x['input32768_SHA'] not in future_shas:
            reserved.append(x);future_shas.add(x['input32768_SHA'])
    need(len(reserved)>=2,'Fewer than2 lawful new future32769-token documents')
    future=reserved[:2];future_vectors=[packed(x['ids'][:32769]) for x in reserved]
    train_dev=[];prefixes=set();skip_counts=collections.Counter()
    for x in ranked:
        if x['history_exclusions'] or not 1024<=x['full_token_count']<32769:continue
        if x['input1024_SHA'] in prefixes:skip_counts['DUPLICATE_TRAIN_DEV_INPUT1024']+=1;continue
        needle=packed(x['ids'][:1024])
        if any(contains(needle,h) for h in future_vectors):skip_counts['INPUT1024_PRESENT_IN_RESERVED_FUTURE']+=1;continue
        if any(contains(needle,h) for h in observed):skip_counts['INPUT1024_PRESENT_IN_KNOWN_HISTORY']+=1;continue
        train_dev.append(x);prefixes.add(x['input1024_SHA'])
        if len(train_dev)==80:break
    need(len(train_dev)==80,'Insufficient64TRAIN+16DEV after all identity/subsequence checks')
    selected={'TRAIN':train_dev[:64],'DEVELOPMENT_VALIDATION':train_dev[64:],'FUTURE_32K_TEST':future}
    selected_records={};all_artifacts=[]
    for split,values in selected.items():
        selected_records[split]=[]
        for order,x in enumerate(values,1):
            p=DATA/'documents'/split/f'doc_{order:03d}';p.mkdir(parents=True,exist_ok=False)
            (p/'raw_context.txt').write_bytes(x['raw'].encode());vectors={'input_ids_1024':x['ids'][:1024]} if split!='FUTURE_32K_TEST' else {'input_ids_32768':x['ids'][:32768],'input_ids_32769':x['ids'][:32769]}
            record={k:v for k,v in x.items() if k not in ['raw','ids','history_exclusions']};record.update(split=split,split_order=order,project_identity='UNKNOWN',project_identity_evidence='No repository/project fields in actual cached source schema',raw_path=str((p/'raw_context.txt').relative_to(ROOT)),artifacts={})
            for name,value in vectors.items():
                f=p/(name+'.json');put(f,value);record['artifacts'][name]=dict(path=str(f.relative_to(ROOT)),file_SHA=sha(f),canonical_SHA=canon(value),count=len(value))
            if split!='FUTURE_32K_TEST':record['capture_positions']=sorted(random.Random(int(x['raw_text_sha256'][:16],16)^20260921).sample(range(128,1024),8))
            else:record['score_positions']=list(range(8192,32768,64))
            put(p/'IDENTITY.json',record);selected_records[split].append(record)
            all_artifacts.extend(dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in p.iterdir() if f.is_file())
    clean_candidates=[{k:v for k,v in x.items() if k not in ['raw','ids']} for x in ranked]
    put(DATA/'CANDIDATE_AUDIT.json',dict(status='PASS_CPU_OUTCOME_BLIND',rule_SHA=rule_sha,candidates=clean_candidates,source_rows=len(rows),distinct_raw_contexts=len(ranked),reserved_future_candidate_count=len(reserved),train_dev_prefix_skip_counts=dict(skip_counts),source_files=source_meta,source_schema=schemas))
    put(DATA/'HISTORICAL_EXCLUSION_AUDIT.json',dict(status='PASS_KNOWN_HISTORY_EXCLUSIONS',verified_sources=sources,document_ids=sorted(ids),raw_SHA=sorted(raws),canonical_token_SHA=sorted(token_hashes),original96=old_rows,original_corpus_SHA=CFG['original_corpus_SHA'],original_builder_path=CFG['original_builder'],original_builder_SHA=sha(CFG['original_builder']),AIME26_unique_problem_count=30,AIME26_dataset_SHA=CFG['aime_dataset_SHA'],AIME26_problem_identities=aime_rows,scope_limitation='All identifiable registered history excluded; unrecorded historical usage cannot be reconstructed or claimed absent'))
    # A second source/raw/token identity pass runs after materialization.
    for split,values in selected_records.items():
        for a in values:
            raw=(ROOT/a['raw_path']).read_text();need(text_sha(raw)==a['raw_text_sha256'],'New materialized raw identity mismatch')
            fresh=tokenizer(raw,add_special_tokens=False).input_ids;need(canon(fresh)==a['full_token_SHA'],'Fresh second tokenization mismatch')
            for name,item in a['artifacts'].items():
                value=read(ROOT/item['path']);need(sha(ROOT/item['path'])==item['file_SHA'] and canon(value)==item['canonical_SHA'] and value==fresh[:item['count']],'Frozen selected input mismatch')
    need(sha(DATA/'SELECTION_RULE.json')==rule_sha,'Rule changed during selection')
    manifest=dict(correction_id='CPU_EXCLUSION_CLASSIFICATION_CORRECTION_V1',correction_source_freeze_SHA=sha(DATA/'CORRECTION_SOURCE_FREEZE.json'),first_failure_retained_SHA=correction['original_first_failure_SHA'],TASK_ID=CFG['TASK_ID'],status='PASS_FROZEN_64_16_2_NEW_INPUTS',rule_SHA=rule_sha,rule_seed=SEED,source_files=source_meta,tokenizer=dict(path=CFG['tokenizer_path'],actual_class=type(tokenizer).__name__,identity_SHA=CFG['tokenizer_identity_sha256'],files=CFG['tokenizer_files'],add_special_tokens=False),splits=selected_records,split_counts={k:len(v) for k,v in selected_records.items()},reserved_future_pool_all_excluded_from_training=[{k:x[k] for k in ['document_id','raw_text_sha256','full_token_count','input32768_SHA']} for x in reserved],original_preprocessing_preserved=dict(max_tokens=1024,capture_points=8,capture_seed='int(rawSHA[:16],16) XOR20260921',capture_min128=True),source_change=rule['data_source_change'],paired_arms_share_order=True,old_C6R_G_adapt_interpretation='Absolute adaptation comparison includes source-data change; only paired G_specific can isolate planned codec variable under all other matched conditions',PROJECT_INDEPENDENCE='UNKNOWN',GPU_work=False,model_forward_count=0,new_model_results_used=False,created_unix=time.time())
    put(DATA/'DATA_SPLIT_MANIFEST.json',manifest)
    leak=dict(TASK_ID=CFG['TASK_ID'],status='PASS_KNOWN_IDENTITY_AND_ACTUAL_SPLIT_EXCLUSION',split_counts=manifest['split_counts'],historical_registry_sources_fresh_SHA=len(sources),historical_raw_SHA_count=len(raws),historical_canonical_token_SHA_count=len(token_hashes),historical_document_ID_count=len(ids),old96_and_AIME30_excluded=True,all_6_or_actual_reserved_long_contexts_excluded_from_TRAIN_DEV=len(reserved),no_raw_or_full_or_canonical_input_duplicate_across_splits=True,no_TRAIN_DEV_input1024_subsequence_in_any_reserved_future32769_input=True,no_TRAIN_DEV_input1024_subsequence_in_available_observed_vectors=True,all_selected_retokenized_fresh_twice=True,PROJECT_INDEPENDENCE='UNKNOWN',limitations=['Unknown repositories/projects prevent claims of project-level independence','Not a proof against arbitrary semantic paraphrase or unrecorded historical usage','Original WikiText training source differs from new paired RepoBench source; both A/D arms share exactly the same new inputs and order'],manifest_SHA=sha(DATA/'DATA_SPLIT_MANIFEST.json'),selection_rule_SHA=rule_sha,artifact_hashes=all_artifacts,GPU_work=False)
    put(DATA/'DATA_LEAKAGE_AUDIT.json',leak)
    signal.alarm(0);event('DATA_FREEZE_COMPLETE',status=manifest['status'],split_counts=manifest['split_counts'],reserved_future=len(reserved),manifest_SHA=sha(DATA/'DATA_SPLIT_MANIFEST.json'),leakage_SHA=sha(DATA/'DATA_LEAKAGE_AUDIT.json'))


if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=DATA/'DATA_SELECTION_FIRST_FAILURE.json'
        if not p.exists():put(p,dict(status='FAIL_CLOSED_STOP',error=repr(e),traceback=traceback.format_exc(),GPU_work=False,no_automatic_retry=True))
        raise
