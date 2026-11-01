"""Fresh-process, CPU-only source reread and retokenization of frozen documents."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import hashlib,json,sys,datetime
ROOT=Path(__file__).resolve().parents[1]
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def fsha(p):return sha(Path(p).read_bytes())
cfg=json.loads((ROOT/'data_selection/SELECTION_CONFIG.json').read_text())
sys.path.insert(0,cfg['transformers_source'])
from transformers import AutoTokenizer
from datasets import Dataset
import torch
assert not torch.cuda.is_initialized()
man=json.loads((ROOT/'frozen_documents.json').read_text())
audit=json.loads((ROOT/'data_selection/DATASET_SELECTION_AUDIT.json').read_text())
exc=json.loads((ROOT/'data_selection/EXCLUSION_UNION.json').read_text())
assert man['selection_audit_sha256']==fsha(ROOT/'data_selection/DATASET_SELECTION_AUDIT.json')
assert man['selection_rule_sha256']==fsha(ROOT/'data_selection/SELECTION_RULES.json')
assert man['exclusion_union_sha256']==fsha(ROOT/'data_selection/EXCLUSION_UNION.json')
assert man['DOCS_FROZEN']==2 and man['new_science_results_seen'] is False
eligible=[r for r in audit['candidates'] if r['status']=='ELIGIBLE']
assert not audit['project_fields_reliable'] # actual inspected snapshot has no repository/project field
assert [r['document_id'] for r in eligible[:2]]==[r['document_id'] for r in man['selected_documents']]
assert [r['raw_text_sha256'] for r in eligible]==sorted(r['raw_text_sha256'] for r in eligible)
tok=AutoTokenizer.from_pretrained(cfg['tokenizer_path'],local_files_only=True)
rows=[]
for d in man['selected_documents']:
 p=Path(d['source_file']);assert fsha(p)==d['source_file_sha256']
 if p.suffix=='.arrow':r=Dataset.from_file(str(p))[d['source_row']]
 else:
  r=[json.loads(x) for x in p.open() if x.strip()][d['source_row']]
 assert r['_id']==d['dataset_id']
 raw=r['context'].encode();assert sha(raw)==d['raw_text_sha256']
 assert (ROOT/d['raw_text_copy']).read_bytes()==raw
 assert d['document_id'] not in exc['document_ids'] and d['raw_text_sha256'] not in exc['raw_text_sha256']
 ids=tok(r['context'],add_special_tokens=False,return_attention_mask=False)['input_ids']
 assert len(ids)==d['full_token_count'] and len(ids)>=32769
 assert sha(canon(ids))==d['full_token_ids_canonical_sha256']
 for key,expected in [('full_token_ids',ids),('input_ids_32769',ids[:32769]),('input_ids_32768',ids[:32768])]:
  a=d['artifacts'][key];p=ROOT/a['path'];actual=json.loads(p.read_text())
  assert actual==expected and len(actual)==a['count']
  assert fsha(p)==a['file_sha256'] and sha(canon(actual))==a['canonical_json_sha256']
 sp=ROOT/'documents'/f'doc_{d["document_index"]:02d}'/'score_positions.json';assert fsha(sp)==d['score_positions_file_sha256']
 positions=json.loads(sp.read_text())['positions'];assert len(positions)==384
 for k,p in enumerate(positions):
  t=8192+64*k
  assert p['token_index_zero_based']==t and p['predicted_next_token_index']==t+1
  assert p['consumed_token_id']==ids[t] and p['target_token_id']==ids[t+1]
 rows.append({'document_id':d['document_id'],'raw_text_sha256':d['raw_text_sha256'],'full_token_count':len(ids),'full_tokens_sha256':sha(canon(ids)),'input_ids_32769_sha256':d['artifacts']['input_ids_32769']['canonical_json_sha256'],'input_ids_32768_sha256':d['artifacts']['input_ids_32768']['canonical_json_sha256'],'score_position_count':384,'source_reread':'PASS','independent_fresh_CPU_retokenization':'PASS','known_exclusions':'PASS','prefix_and_position_semantics':'PASS'})
assert not torch.cuda.is_initialized()
result={'status':'PASS_FRESH_PROCESS_FROZEN_DOCUMENT_VERIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':fsha(ROOT/'frozen_documents.json'),'verification_code_sha256':fsha(__file__),'selection_is_exact_first_two_raw_SHA_order':True,'selected_documents':rows,'project_independence':'UNKNOWN','known_exclusion_id_count':exc['id_count'],'known_exclusion_raw_sha_count':exc['raw_sha_count'],'CUDA_initialized':False,'model_weights_loaded':False,'GPU_HOURS_USED':0}
p=ROOT/'data_selection/FROZEN_DOCUMENT_VERIFICATION.json'
with p.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,indent=2))
