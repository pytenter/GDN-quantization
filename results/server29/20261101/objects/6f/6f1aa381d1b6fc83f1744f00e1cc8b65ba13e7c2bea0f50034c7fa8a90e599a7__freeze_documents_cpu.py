"""Freeze two existing-cache test documents. CPU tokenizer only, no model loads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
import hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha_bytes(pathlib.Path(p).read_bytes())
def read(rel):return json.loads((ROOT/rel).read_text())
def save(rel,data):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
LING=['pg19-test-41603','pg19-test-2544','pg19-test-7987','pg19-test-26183']
rule=dict(seed=20261015,requested_N=2,source='verified existing Qwen LongBench RepoBench-P/P_e test caches',
     ordering='SHA256(str(seed).encode()+raw_text_SHA.encode()) ascending; exact raw-SHA dedup; ties by document ID',
     exclusions='all previous known IDs/raw SHAs + frozen NoInflation8doc + human-provided Ling four IDs',
     min_tokens=128,input_slice='first128 tokens with add_special_tokens=False and no chat template; no leading/padding/BOS insertion',
     no_calibration_or_parameter_fitting=True,no_model_or_GPU_execution=True,project_independence='UNKNOWN; distinct documents/texts only',
     invalid_candidate_behavior='preserve failure and STOP; no outcome-based reselection; insufficient token length is a predeclared eligibility rejection')
save('analysis/DOCUMENT_SELECTION_RULE_BEFORE_SELECTION.json',rule)
save('history/HUMAN_LING_EXCLUSION_ADDENDUM.json',dict(source='Direct human reply in this chat',task='LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1',freeze_date='2026-10-14',excluded_ids=LING,original_manifest_path='NOT_PROVIDED',original_manifest_verification='NOT_CLAIMED',authorization='Human explicitly permits using these known IDs without locating original manifest; GPU switch unchanged NO'))
prior=read('runtime_sources/SELECTION_EXCLUSION_REGISTRY.json');old=read('runtime_sources/FROZEN_8DOC_SELECTION.json');audit=read('runtime_sources/DOCUMENT_SELECTION_AUDIT.json')
excluded_ids=set(prior['ids'])|set(LING);excluded_sha=set(prior['raw_SHA'])
for d in old['selected_documents']:
    excluded_ids.add(d['document_id']);excluded_sha.add(d['raw_text_SHA'])
unique={}
for d in audit['all_rows']:
    if d['document_id'] in excluded_ids or d['dataset_id'] in excluded_ids or d['raw_text_SHA'] in excluded_sha:continue
    if d.get('token_count',0)<128:continue
    key=d['raw_text_SHA']
    if key not in unique or d['document_id']<unique[key]['document_id']:unique[key]=d
ranked=sorted(unique.values(),key=lambda d:(sha_bytes(str(rule['seed']).encode()+d['raw_text_SHA'].encode()),d['document_id']))
save('analysis/RANKED_DOCUMENT_POOL_BEFORE_TOKENIZATION.json',dict(rule_sha256=sha_file(ROOT/'analysis/DOCUMENT_SELECTION_RULE_BEFORE_SELECTION.json'),candidates=[dict(document_id=d['document_id'],raw_text_SHA=d['raw_text_SHA'],selection_key=sha_bytes(str(rule['seed']).encode()+d['raw_text_SHA'].encode()),source_file=d['source_file'],row=d['row']) for d in ranked],excluded_ids=sorted(excluded_ids),excluded_raw_sha_count=len(excluded_sha),known_Ling_ID_source='history/HUMAN_LING_EXCLUSION_ADDENDUM.json'))
sys.path.insert(0,'/data/zypan/transformers-qwen35/src')
from transformers import AutoTokenizer
import torch
tokenizer=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
assert tokenizer.__class__.__name__=='Qwen2Tokenizer',tokenizer.__class__.__name__
modeldir=pathlib.Path('/data/zypan/modelscope_models/Qwen3.5-9B')
tokenizer_files=[]
for a in audit['tokenizer_identity']['files']:
    got=sha_file(modeldir/a['name']);assert got==a['sha256'];tokenizer_files.append(dict(name=a['name'],sha256=got))
inventory={a['path']:a for a in audit['existing_cache_inventory']};verified_sources={};selections=[];prefixes=set()
for d in ranked:
    if len(selections)==2:break
    path=pathlib.Path(d['source_file']);expected=inventory[str(path)]
    if str(path) not in verified_sources:
        got=sha_file(path);assert got==expected['SHA'],str(path)
        verified_sources[str(path)]=dict(sha256=got,bytes=path.stat().st_size)
    if path.suffix=='.arrow':
        import pyarrow as pa
        with pa.memory_map(str(path),'r') as f:table=pa.ipc.open_stream(f).read_all()
        row=table.slice(d['row'],1).to_pylist()[0]
    else:row=json.loads(path.read_text().splitlines()[d['row']])
    raw=row['context'];rawsha=sha_bytes(raw.encode('utf-8'));assert rawsha==d['raw_text_SHA']
    found=str(row.get('_id',row.get('id','')));assert found==d['dataset_id'],(found,d['dataset_id'])
    assert d['document_id'] not in excluded_ids and found not in excluded_ids and rawsha not in excluded_sha
    ids=tokenizer(raw,add_special_tokens=False)['input_ids']
    if len(ids)<128:
        save(f'analysis/LENGTH_REJECTION_{len(selections)}.json',dict(document_id=d['document_id'],actual_tokens=len(ids)));continue
    selected=list(map(int,ids[:128]));canonical=json.dumps(selected,separators=(',',':')).encode();tokenhash=sha_bytes(canonical)
    assert tokenhash not in prefixes,'Duplicate128-token input: STOP without reselection'
    prefixes.add(tokenhash);i=len(selections)+1
    tokenpath=f'documents/doc_{i:02d}/input_ids_128.json';save(tokenpath,selected)
    textpath=ROOT/f'documents/doc_{i:02d}/raw_context.txt';textpath.write_text(raw,encoding='utf-8')
    rowinfo=dict(document_id=d['document_id'],dataset_id=d['dataset_id'],source_file=str(path),source_file_sha256=verified_sources[str(path)]['sha256'],source_row=d['row'],source_variant=d['source_variant'],raw_text_sha256=rawsha,raw_text_bytes=len(raw.encode()),raw_text_copy=str(textpath.relative_to(ROOT)),full_token_count=len(ids),consumed_token_count=128,token_ids_path=tokenpath,token_ids_file_sha256=sha_file(ROOT/tokenpath),token_ids_canonical_json_sha256=tokenhash,tokenizer_class=tokenizer.__class__.__name__,add_special_tokens=False,chat_template=False,selection_key=sha_bytes(str(rule['seed']).encode()+rawsha.encode()),history_ID_raw_SHA_exclusions='PASS',known_Ling_ID_exclusion='PASS_HUMAN_PROVIDED_SET',AIME26_source='NOT_AIME; RepoBench test',calibration_evaluation_separation='No calibration, query weighting, tuning or parameter fitting uses these documents; evaluation-only frozen inputs',repository_project_independence='UNKNOWN_NOT_IN_SCHEMA')
    selections.append(rowinfo)
assert len(selections)==2,len(selections)
assert not torch.cuda.is_initialized()
save('analysis/FROZEN_CANARY_DOCUMENTS.json',dict(status='PASS_DOCUMENT_IDENTITY_AND_SELECTION_ONLY',selected_documents=selections,requested_N=2,distinct_raw_documents=2,project_independence='UNKNOWN',known_Ling_ids=LING,Ling_original_manifest_verification='NOT_CLAIMED_HUMAN_OVERRIDE',verified_sources=verified_sources,tokenizer_files=tokenizer_files,tokenizer_identity_sha256=audit['tokenizer_identity']['SHA'],rule_sha256=sha_file(ROOT/'analysis/DOCUMENT_SELECTION_RULE_BEFORE_SELECTION.json'),candidate_count=len(ranked),outcome_blind=True,calibration_or_tuning_used=False,model_loaded=False,CUDA_initialized=False,GPU_HOURS_USED=0,GPU_AUTHORIZATION='NO'))
print(json.dumps(dict(status='PASS_DOCUMENT_IDENTITY_AND_SELECTION_ONLY',documents=[dict(document_id=d['document_id'],input_token_hash=d['token_ids_canonical_json_sha256']) for d in selections],GPU_HOURS_USED=0)))
