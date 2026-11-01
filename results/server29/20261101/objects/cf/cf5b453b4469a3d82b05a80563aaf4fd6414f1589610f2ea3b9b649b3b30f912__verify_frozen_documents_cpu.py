"""Fresh CPU process independently rereads source rows and tokenizes all five selected docs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import json,hashlib,csv,sys,datetime
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data_selection'
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((OUT/'SELECTION_CONFIG.json').read_text());sys.path.insert(0,cfg['transformers_source'])
from transformers import AutoTokenizer
from datasets import Dataset
import torch
assert torch.cuda.is_initialized() is False
assert (ROOT/'FROZEN_DOCUMENTS.json').read_bytes()==(ROOT/'frozen_documents.json').read_bytes()
m=json.loads((ROOT/'FROZEN_DOCUMENTS.json').read_text());audit=json.loads((OUT/'CANDIDATE_AUDIT.json').read_text());ex=json.loads((OUT/'EXCLUSION_UNION.json').read_text())
assert m['DOCS_FROZEN']==5 and m['status']=='PASS_FROZEN_5_NEW_DOCUMENTS' and m['new_science_results_seen'] is False
assert m['selection_rule_sha256']==sha(OUT/'SELECTION_RULE.json') and m['selection_config_sha256']==sha(OUT/'SELECTION_CONFIG.json') and m['selection_code_sha256']==sha(OUT/'select_documents_cpu.py')
assert m['candidate_audit_json_sha256']==sha(OUT/'CANDIDATE_AUDIT.json') and m['candidate_audit_csv_sha256']==sha(OUT/'CANDIDATE_AUDIT.csv')
assert m['exclusion_union_sha256']==sha(OUT/'EXCLUSION_UNION.json') and m['project_metadata_audit_sha256']==sha(OUT/'PROJECT_METADATA_AUDIT.json')
assert audit['reliable_project_mapping'] is False and m['PROJECT_INDEPENDENCE']==m['project_independence']=='UNKNOWN'
eligible=[x for x in audit['candidates'] if x['status']=='ELIGIBLE'];assert len(eligible)==11
assert [x['document_id'] for x in eligible[:5]]==[x['document_id'] for x in m['selected_documents']]
assert [x['selection_key'] for x in audit['candidates']]==sorted(x['selection_key'] for x in audit['candidates'])
with (OUT/'CANDIDATE_AUDIT.csv').open(newline='') as f:csv_rows=list(csv.DictReader(f))
assert len(csv_rows)==len(audit['candidates'])==789 and len([x for x in csv_rows if x['selected_order']])==5
for x in m['source_files']:assert sha(x['path'])==x['sha256']
for x in m['tokenizer']['files']:assert sha(Path(cfg['tokenizer_path'])/x['name'])==x['sha256']
assert hashlib.sha256(canon(m['tokenizer']['files'])).hexdigest()==m['tokenizer']['identity_sha256']==cfg['tokenizer_identity_sha256']
tok=AutoTokenizer.from_pretrained(cfg['tokenizer_path'],local_files_only=True);checked=[];rawset=set();inputset=set()
for d in m['selected_documents']:
 p=Path(d['source_file']);assert sha(p)==d['source_file_sha256']
 r=Dataset.from_file(str(p))[d['source_row']] if p.suffix=='.arrow' else [json.loads(x) for x in p.open() if x.strip()][d['source_row']]
 assert r['_id']==d['dataset_id'] and 'long_bench/repobench-p/test/'+r['_id']==d['document_id']
 raw=r['context'].encode();rawsha=hashlib.sha256(raw).hexdigest();assert rawsha==d['raw_text_sha256'] and (ROOT/d['raw_text_copy']).read_bytes()==raw
 ids=tok(r['context'],add_special_tokens=False,return_attention_mask=False)['input_ids'];assert len(ids)==d['full_token_count']>=32769
 for key,expected in [('full_token_ids',ids),('input_ids_32769',ids[:32769]),('input_ids_32768',ids[:32768])]:
  a=d['artifacts'][key];p=ROOT/a['path'];actual=json.loads(p.read_text())
  assert actual==expected and a['count']==len(actual) and sha(p)==a['file_sha256'] and hashlib.sha256(canon(actual)).hexdigest()==a['canonical_json_sha256']
 inputsha=hashlib.sha256(canon(ids[:32768])).hexdigest()
 assert d['document_id'] not in ex['document_ids'] and rawsha not in ex['raw_text_sha256'] and inputsha not in ex['canonical_32768_input_sha256']
 assert rawsha not in rawset and inputsha not in inputset;rawset.add(rawsha);inputset.add(inputsha)
 positions=ROOT/'documents'/f'doc_{d["document_index"]:02d}'/'score_positions.json';assert sha(positions)==d['score_positions_file_sha256']
 points=json.loads(positions.read_text())['positions'];assert len(points)==384
 for k,row in enumerate(points):
  t=8192+64*k;assert row['token_index_zero_based']==t and row['predicted_next_token_index']==t+1 and row['consumed_token_id']==ids[t] and row['target_token_id']==ids[t+1]
 checked.append({'document_id':d['document_id'],'full_token_count':len(ids),'raw_sha256':rawsha,'canonical_input32768_sha256':inputsha,'status':'PASS_FRESH_SOURCE_TOKEN_PREFIX_POSITIONS_AND_EXCLUSIONS'})
assert torch.cuda.is_initialized() is False
result={'status':'PASS_FRESH_PROCESS_FROZEN_DOCUMENT_VERIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_documents_sha256':sha(ROOT/'FROZEN_DOCUMENTS.json'),'manifest_sha256':sha(ROOT/'FROZEN_DOCUMENTS.json'),'uppercase_lowercase_alias_same_bytes':True,'selected_count':5,'eligible_count':len(eligible),'candidate_rows':len(audit['candidates']),'fixed_rank_first_five':'PASS','selected_documents':checked,'project_independence':'UNKNOWN','known_exclusion_ID_count':ex['document_id_count'],'known_exclusion_raw_SHA_count':ex['raw_sha_count'],'known_exclusion_canonical_input_count':ex['historical_canonical_input_count'],'model_weights_loaded':False,'CUDA_initialized':False,'GPU_forward_count':0,'verification_code_sha256':sha(__file__)}
with (OUT/'FROZEN_DOCUMENT_VERIFICATION.json').open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
