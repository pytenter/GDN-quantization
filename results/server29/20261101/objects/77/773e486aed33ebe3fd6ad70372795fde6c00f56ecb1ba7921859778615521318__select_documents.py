"""Outcome-blind local RepoBench-P identity selection. No model or outcomes used."""
import hashlib,json,os,re,sys,traceback
from pathlib import Path
import pilot_engineering as P

SOURCE=Path('/data/ydai/qwen_model_cache/datasets/long_bench/repobench-p/1.0.0/0384ffc1cabf36a2322f408bfdc21ea54d6a2704f2b1e17ca49d89035e666f1e/long_bench-test.arrow')
SEED=20261012

def main():
    P.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only selection')
    source_sha=P.sha(SOURCE)
    P.save('DOCUMENT_SELECTION_RULE_BEFORE_TOKENIZATION.json',{'utc':P.N.now(),'seed':SEED,
        'source_path':str(SOURCE),'source_SHA':source_sha,'context_only':True,'minimum_tokens':32769,
        'rank':'SHA256(task|seed|canonical_document_id|raw_text_SHA)','near_duplicate_rule':'64-token aligned shingles Jaccard >=0.2 rejected; exact prefix/content duplicate rejected',
        'selection_does_not_use':'G,KL,S1,S2,activation,intervention outcomes','count':2})
    sys.path.insert(0,'/data/zypan/transformers-qwen35')
    from transformers import AutoTokenizer
    from datasets import Dataset
    model=Path('/data/zypan/modelscope_models/Qwen3.5-9B')
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True)
    files=[{'name':name,'sha256':P.sha(model/name)} for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']]
    tokenizer_sha=P.digest(files)
    dataset=Dataset.from_file(str(SOURCE)).select_columns(['context','_id'])
    candidates=[];uid_map={}
    for i,row in enumerate(dataset):
        text=row['context'];raw=text.encode('utf-8');raw_sha=hashlib.sha256(raw).hexdigest()
        uid=str(row['_id']);docid='long_bench/repobench-p/test/'+uid
        uid_map[uid]={'document_id':docid,'raw_text_SHA':raw_sha}
        candidates.append({'document_id':docid,'dataset_id':uid,'row':i,'raw_text_SHA':raw_sha,'text':text})
    excluded_ids=set();excluded_raw=set();sources=[]
    def extract(obj):
        found=[]
        def walk(value,key=''):
            if isinstance(value,dict):
                for k,v in value.items():walk(v,str(k).lower())
            elif isinstance(value,list):
                for v in value:walk(v,key)
            elif isinstance(value,str):
                if value.startswith('long_bench/repobench-p/test/'):
                    excluded_ids.add(value);found.append({'document_id':value})
                elif key in ['document_id','canonical_doc_id','dataset_document_id','_id'] and value in uid_map:
                    excluded_ids.add(uid_map[value]['document_id']);found.append(uid_map[value])
                if key in ['raw_text_sha256','raw_text_sha','text_sha256','content_sha256','text_sha'] and re.fullmatch('[0-9a-f]{64}',value):
                    excluded_raw.add(value);found.append({'raw_text_SHA':value})
        walk(obj)
        return found
    survey=[Path('/data/zypan/worktrees'),Path('/data/zypan/results'),Path('/data/zypan/GDN-quantization')]
    skipped=[]
    for base in survey:
        for parent,dirs,names in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','node_modules'] and len(Path(parent).relative_to(base).parts)<10]
            if P.REPO==Path(parent) or P.REPO in Path(parent).parents:dirs[:]=[];continue
            for name in names:
                path=Path(parent)/name;lower=name.lower()
                if path.suffix!='.json':continue
                if any(s in lower for s in ['candidate_pool','eligible_document','corpus_inventory','frozen_cumsum','tokens.json']):continue
                if not any(s in lower for s in ['document','manifest','frozen','panel','audit','identity','checkpoint']):continue
                if path.stat().st_size>20*1024**2:
                    skipped.append({'path':str(path),'reason':'large non-selected metadata; explicit selected document manifests used'});continue
                try:obj=json.loads(path.read_text())
                except (ValueError,UnicodeError):continue
                hits=extract(obj)
                if hits:sources.append({'path':str(path),'sha256':P.sha(path),'identified_exclusions':hits})
    old=json.loads((P.V3/'SCIENCE_PREMEASUREMENT_SPEC.json').read_text())['documents']
    for d in old:
        excluded_ids.add(d['canonical_doc_id']);excluded_raw.add(d['text_sha256'])
    eligibility=[];ranked=[]
    for row in candidates:
        if row['document_id'] in excluded_ids or row['raw_text_SHA'] in excluded_raw:
            eligibility.append({'document_id':row['document_id'],'row':row['row'],'raw_text_SHA':row['raw_text_SHA'],'status':'EXCLUDED_PREVIOUS_RELATED_PANEL'});continue
        ids=tokenizer.encode(row['text'],add_special_tokens=False)
        entry={k:row[k] for k in ['document_id','dataset_id','row','raw_text_SHA']};entry['token_count']=len(ids)
        if len(ids)<32769:entry['status']='INSUFFICIENT_LENGTH';eligibility.append(entry);continue
        key=hashlib.sha256(f"{P.TASK}|{SEED}|{row['document_id']}|{row['raw_text_SHA']}".encode()).hexdigest()
        entry.update(status='LENGTH_ELIGIBLE',selection_key=key);eligibility.append(entry)
        row.update(ids=ids[:32769],selection_key=key);ranked.append(row)
    ranked.sort(key=lambda r:(r['selection_key'],r['document_id']));selected=[];overlap=[]
    for row in ranked:
        shingles={hashlib.sha256(json.dumps(row['ids'][p:p+64],separators=(',',':')).encode()).hexdigest() for p in range(0,32769-63,64)}
        reject=False
        for other in selected:
            j=len(shingles & other['shingles'])/len(shingles | other['shingles'])
            overlap.append({'a':row['document_id'],'b':other['document_id'],'aligned64token_Jaccard':j})
            if j>=.2 or row['ids']==other['ids'] or row['text']==other['text']:reject=True
        if reject:continue
        row['shingles']=shingles;selected.append(row)
        if len(selected)==2:break
    result={'status':'PASS' if len(selected)==2 else 'DOCUMENT_ELIGIBILITY_STOP','utc':P.N.now(),'seed':SEED,
        'dataset_identity':{'name':'long_bench/repobench-p','split':'test','snapshot':str(SOURCE),'snapshot_SHA':source_sha,'bytes':SOURCE.stat().st_size},
        'tokenizer_identity':{'files':files,'SHA':tokenizer_sha,'class':type(tokenizer).__name__,'add_special_tokens':False,'chat_template':False},
        'exclusion_registry':{'ids':sorted(excluded_ids),'raw_SHA':sorted(excluded_raw),'source_manifests':sources,'skipped':skipped,
            'rule':'selected/used related document identities, not all candidates in an eligibility pool'},
        'eligibility':eligibility,'near_duplicate_comparisons':overlap,'documents':[],'formal_GPU_forward_count':0}
    for i,row in enumerate(selected,1):
        text_path=P.ROOT/f'inputs/prospective_{i:02d}.txt';text_path.parent.mkdir(exist_ok=True)
        with text_path.open('xb') as f:f.write(row['text'].encode())
        P.save(f'inputs/prospective_{i:02d}_tokens.json',row['ids'])
        token_path=P.ROOT/f'inputs/prospective_{i:02d}_tokens.json'
        result['documents'].append({'document_index':i,'document_id':row['document_id'],'row':row['row'],
            'raw_text_SHA':P.sha(text_path),'token_SHA':P.sha(token_path),'tokenizer_SHA':tokenizer_sha,
            'text_path':str(text_path),'token_path':str(token_path),'input_definition':'32769 raw-context tokens; consume first32768; extra token is prediction alignment',
            'input_count':32769,'selection_key':row['selection_key']})
    P.save('DOCUMENT_SELECTION_AND_EXCLUSIONS.json',result)
    P.check(len(selected)==2,'DOCUMENT_ELIGIBILITY_STOP: fewer than2 independent new32769-token documents')
    print('PROSPECTIVE_DOCUMENT_SELECTION_PASS',[(d['document_id'],d['raw_text_SHA']) for d in result['documents']],flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if not (P.ROOT/'FIRST_FAILURE_GATE.json').exists():P.save('FIRST_FAILURE_GATE.json',{'status':'FAIL_AND_STOP',
            'stage':'DOCUMENT_ELIGIBILITY','utc':P.N.now(),'message':str(exc),'exception':type(exc).__name__,
            'traceback':traceback.format_exc(),'formal_outcomes_produced':False,'no_retry':True})
        raise
