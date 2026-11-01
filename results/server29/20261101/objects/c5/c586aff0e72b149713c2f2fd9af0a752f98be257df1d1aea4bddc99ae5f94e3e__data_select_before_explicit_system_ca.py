"""One bounded CPU-only PG19 validation selection. NO model/CUDA/score imports.

Before any book payload: pin full50-object metadata, exclusions, tokenizer identities,
and deterministic ordering. Each generation-pinned book is requested once, no retry.
All historical files are read only. Only the current new task paths may be created.
"""
import base64
import csv
import datetime
import hashlib
import json
import os
import re
import struct
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

TASK='LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1'
D=Path('/data/zypan/experiment_runs')/TASK
OLD=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')
MODEL=Path('/data/zypan/models/Ling-3.0-tiny')
TRAIN=Path('/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1/source_traces/CALIBRATION_RAW_TEXTS.jsonl')
FRESH=Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY/protocol/fresh_b_selection_manifest.json')
TOTAL_CAP=512*1024**2
EXPECTED={OLD/'PG19_TEST_GCS_LISTING.json':'2aa8c57a6d0d17156bb10f3455f3e2bc4301d19a6cdc5754de3b40ed5932581d',OLD/'PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json':'6ea30dae0a79ccb4b540e9fc670c23a2996ee29345a5956e390f6befaa204f67',OLD/'historical_path_identity_inventory.json':'4bc1da0da22a6b43185ba6a5507b0d4ade95b3a9bad2b3c51e015e34c023bf97',TRAIN:'37f38795847da8daa776be9d7dd3b6083442dd7a27e0d14f0dedbeaa3d4884d0',FRESH:'97b19570b1a615a7e6301a9f74eab50b176395adf0d3c28d62d4c9b00ca69adf'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(ids):return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()
def norm(s):return hashlib.sha256(' '.join(unicodedata.normalize('NFC',s).split()).encode()).hexdigest()
def put(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def event(v):print(json.dumps(v,ensure_ascii=False),flush=True)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ONLY_VISIBLE_DEVICE_BOUNDARY'
 for x in ['FROZEN_DOCUMENTS.json','DATA_SELECTION_AUDIT.csv','audits/DATA_SELECTION_PROTOCOL.json','audits/DATA_SELECTION_MANIFEST.json','audits/DATA_SELECTION_FIRST_FAILURE.json']:
  assert not (D/x).exists(),('NO_OVERWRITE_OR_RETRY',x)
 assert not (D/'inputs').exists(),'NO_EXISTING_SELECTION_INPUTS'
 started=time.time()
 sources=[]
 for p,h in EXPECTED.items():
  actual=sha(p);assert actual==h,('HISTORICAL_MANIFEST_SHA',str(p));sources.append({'path':str(p),'sha256':actual})
 pool=read(OLD/'PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json')
 original_docs=pool['inputs'];assert len(original_docs)==100
 fresh=read(FRESH);old20=set(read(OLD/'historical_path_identity_inventory.json')['path_identity_evidence'])
 protected=old20|{x for x in fresh['exclusion_sets'] if x.startswith('pg19-')}|{x['document_id'] for x in fresh['streams']}|{'pg19-test-41603','pg19-test-2544','pg19-test-7987','pg19-test-26183'}
 assert len(protected)==58
 train=[json.loads(x) for x in TRAIN.read_text().splitlines()];assert len(train)==96
 historical_raw=set();historical_norm=set();historical_full_tokens=set();historical_prefix_tokens=set()
 historical_cache_checks=[]
 for x in original_docs:
  raw=Path(x['raw_text_path']).read_bytes();ids=read(x['token_file_path'])
  assert hashlib.sha256(raw).hexdigest()==x['raw_sha256']
  assert sha(x['token_file_path'])==x['token_file_sha256']
  assert canonical(ids)==x['canonical_prefix_token_sha256']
  historical_raw.add(x['raw_sha256']);historical_norm.add(x['normalized_text_sha256'])
  historical_full_tokens.add(x['canonical_full_token_sha256'])
  historical_prefix_tokens.add(canonical(ids[:8449]) if len(ids)>=8449 else canonical(ids))
  historical_cache_checks.append({'document':x['document'],'raw_sha256':x['raw_sha256'],'canonical_full_token_sha256':x['canonical_full_token_sha256'],'current_cached_token_sha256':x['token_file_sha256'],'actual_sha_checks':'PASS'})
 for x in train:
  assert hashlib.sha256(x['raw_text'].encode()).hexdigest()==x['raw_text_sha256'],('TRAIN_RAW_INNER_IDENTITY',x['document_id'])
  historical_raw.add(x['raw_text_sha256']);historical_norm.add(norm(x['raw_text']))
 # Recover validation IDs from already persisted historical metadata, with explicit bounds.
 scan_roots=[Path(x) for x in ['/data/zypan/experiment_runs','/data/zypan/worktrees','/data/zypan/cleanroom','/data/zypan/diagnostic_inputs','/data/zypan/experiments','/data/zypan/repos/GDN-quantization']]
 skip={'.git','.cache','__pycache__','envs','models','node_modules','raw_text','tokens','rotations','traces','checkpoints','sources','source_assets','source_snapshot','source','inputs','historical_assets','provenance_capsule_v3'}
 historical_validation={};metadata_scanned=0
 pattern=re.compile(r'pg19-validation-[0-9]+')
 for root in scan_roots:
  for base,dirs,files in os.walk(root):
   if str(D) in base or 'ling-kda-rotation-dual-axis-functional-v1' in base:
    dirs[:]=[];continue
   dirs[:]=[v for v in dirs if v not in skip]
   for name in dirs+files:
    for doc in pattern.findall(str(Path(base)/name)):
     historical_validation.setdefault(doc,[]).append(str(Path(base)/name))
   for name in files:
    p=Path(base)/name
    if p.suffix!='.json':continue
    try:
     if p.stat().st_size>1024**2:continue
     text=p.read_text()
    except (OSError,UnicodeError):continue
    metadata_scanned+=1
    for doc in set(pattern.findall(text)):historical_validation.setdefault(doc,[]).append(str(p))
 assert metadata_scanned<=100000,'BOUNDED_HISTORY_SCAN'
 q=urllib.parse.urlencode({'prefix':'validation/','maxResults':'100','fields':'items(name,size,generation,md5Hash),nextPageToken'})
 listing_url='https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?'+q
 with urllib.request.urlopen(listing_url,timeout=30) as response:listing=json.load(response)
 assert 'nextPageToken' not in listing and len(listing['items'])==50,'OFFICIAL_FULL_VALIDATION50_REQUIRED'
 objects=sorted(listing['items'],key=lambda x:x['name'])
 assert len({x['name'] for x in objects})==50
 assert all(re.fullmatch(r'validation/[0-9]+\.txt',x['name']) and x['generation'] and x['md5Hash'] for x in objects)
 assert sum(int(x['size']) for x in objects)<=TOTAL_CAP,'RAW_DOWNLOAD_BUDGET'
 model_identity={name:sha(MODEL/name) for name in ['config.json','tokenizer.json','tokenizer_config.json']}
 assert model_identity==pool['model_tokenizer_identity'],'ORIGINAL_LING_TOKENIZER_IDENTITY'
 expected_ids=set(x['document'] for x in original_docs)|protected|set(historical_validation)
 protocol={'TASK_ID':TASK,'status':'PREFROZEN_BEFORE_ANY_VALIDATION_PAYLOAD_OR_TOKENIZATION','frozen_utc':now(),'source_script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'official_source':'https://github.com/google-deepmind/pg19','dataset':'PG19','source_split':'validation','RepoBench_P':'NOT_LOCATED; PG19 accurately named alternative authorized by root','project_independence':'UNKNOWN','candidate_pool':{'listing_url':listing_url,'listing':listing,'complete_objects':50,'declared_total_raw_bytes':sum(int(x['size']) for x in objects)},'seed':0,'ordering':'ascending SHA256(UTF8(TASK_ID+\"|seed=0|validation|\"+document_id+\"|\"+raw_sha256)); document_id tie-break','eligibility':{'minimum_actual_Ling_tokens':8449,'tokenizer':'AutoTokenizer.from_pretrained(local original model, local_files_only=True, trust_remote_code=False).encode(original UTF8 text, add_special_tokens=False)','normalization':'none for model inputs; NFC+whitespace for duplicate hashes only','token_canonicalization':'uint32 little-endian bytes without header','excluded_all_previous_test100_even_if_CPU_only':True,'excluded_protected58':sorted(protected),'excluded_calibration96_raw_manifest':str(TRAIN),'history_metadata_scan':{'roots':[str(x) for x in scan_roots],'skipped_directory_names':sorted(skip),'JSON_size_cap_bytes':1024**2,'JSON_files_scanned':metadata_scanned,'found_validation_ID_paths':historical_validation,'boundary':'bounded identifiable historical records; base-model pretraining overlap UNKNOWN'},'duplicate_rule':'historical raw/normalized/full canonical tokens/8449-prefix always excluded; within new candidates keep earliest under frozen hash ordering','raw_requests_per_object':1,'application_retries':0,'total_raw_bytes_cap':TOTAL_CAP,'source_generation_and_MD5_required':True},'selection':'first6 eligible formal + seventh eligible independent canary; no outcome reads; insufficient7 STOP without replacement dataset','condition_order':['FP_NATIVE','H_R128','H_DUAL_M','L6_R128','L6_DUAL_M'],'trajectories':{'formal':30,'consumed_tokens':8448,'minimum_input_tokens':8449,'score_positions':list(range(8192,8448))},'model_tokenizer_identity':model_identity,'historical_manifest_bindings':sources,'historical_cache_actual_SHA_checks':historical_cache_checks,'historical_raw_or_token_payloads_modified':False,'GPU_forward_before_freeze':0}
 put(D/'audits/DATA_SELECTION_PROTOCOL.json',protocol)
 protocol_sha=sha(D/'audits/DATA_SELECTION_PROTOCOL.json')
 put(D/'audits/PG19_VALIDATION_GCS_LISTING.json',listing)
 event({'stage':'PROTOCOL_FROZEN','sha256':protocol_sha,'objects':50,'total_raw_bytes':sum(int(x['size']) for x in objects),'historical_validation_seen_ids':sorted(historical_validation),'metadata_files_scanned':metadata_scanned})
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
 for x in train:
  ids=tokenizer.encode(x['raw_text'],add_special_tokens=False)
  historical_full_tokens.add(canonical(ids));historical_prefix_tokens.add(canonical(ids[:8449]))
 raw_dir=D/'inputs/raw';tok_dir=D/'inputs/tokens';raw_dir.mkdir(parents=True);tok_dir.mkdir()
 rows=[];downloaded=0
 for index,obj in enumerate(objects):
  name=obj['name'];book_id=Path(name).stem;doc='pg19-validation-'+book_id
  request_url='https://storage.googleapis.com/download/storage/v1/b/deepmind-gutenberg/o/'+urllib.parse.quote(name,safe='')+'?'+urllib.parse.urlencode({'alt':'media','generation':obj['generation']})
  before=time.time()
  with urllib.request.urlopen(request_url,timeout=60) as response:
   generation=response.headers.get('x-goog-generation')
   if generation is not None:assert generation==obj['generation'],'GENERATION_HEADER_MISMATCH'
   raw=response.read(int(obj['size'])+1)
  downloaded+=len(raw)
  assert downloaded<=TOTAL_CAP and len(raw)==int(obj['size']),'DOWNLOADED_SIZE_MISMATCH'
  assert base64.b64encode(hashlib.md5(raw).digest()).decode()==obj['md5Hash'],'OBJECT_MD5_MISMATCH'
  text=raw.decode('utf-8',errors='strict');raw_path=raw_dir/(doc+'.txt')
  with raw_path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  ids=tokenizer.encode(text,add_special_tokens=False)
  assert all(type(x) is int and 0<=x<157184 for x in ids),'CANONICAL_TOKEN_RANGE'
  token_path=tok_dir/(doc+'.json')
  with token_path.open('x') as f:json.dump(ids,f,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
  raw_sha=hashlib.sha256(raw).hexdigest();nsha=norm(text);full_sha=canonical(ids);prefix_sha=canonical(ids[:8449]) if len(ids)>=8449 else None
  reasons=[]
  if len(ids)<8449:reasons.append('INSUFFICIENT_8449_TOKENS')
  if doc in expected_ids:reasons.append('IDENTIFIABLE_HISTORICAL_DOCUMENT_ID')
  if 'pg19-test-'+book_id in expected_ids:reasons.append('OLD_TEST100_OR_PROTECTED_BOOK_ID')
  if raw_sha in historical_raw:reasons.append('HISTORICAL_RAW_SHA_DUPLICATE')
  if nsha in historical_norm:reasons.append('HISTORICAL_NORMALIZED_SHA_DUPLICATE')
  if full_sha in historical_full_tokens:reasons.append('HISTORICAL_FULL_TOKEN_DUPLICATE')
  if prefix_sha in historical_prefix_tokens:reasons.append('HISTORICAL_8449_PREFIX_TOKEN_DUPLICATE')
  order=hashlib.sha256((TASK+'|seed=0|validation|'+doc+'|'+raw_sha).encode()).hexdigest()
  row={'document':doc,'document_id':doc,'book_id':book_id,'dataset':'PG19','source_split':'validation','project_identity':None,'PROJECT_INDEPENDENCE':'UNKNOWN','source_url':request_url,'generation':obj['generation'],'object_md5_base64':obj['md5Hash'],'raw_text_path':str(raw_path),'raw_sha256':raw_sha,'raw_bytes':len(raw),'normalized_text_sha256':nsha,'token_file_path':str(token_path),'token_file_sha256':sha(token_path),'full_token_count':len(ids),'canonical_full_token_sha256':full_sha,'canonical_prefix_token_sha256':prefix_sha,'canonical_prefix_count':8449,'order_sha256':order,'exclusion_reasons':reasons,'eligible_before_new_pool_dedup':not reasons,'raw_download_request_count':1,'download_seconds':time.time()-before}
  rows.append(row);event({'stage':'CPU_TOKENIZED_VALIDATION','index':index+1,'document':doc,'tokens':len(ids),'eligible_before_dedup':not reasons,'downloaded_total_bytes':downloaded})
 ordered=sorted(rows,key=lambda x:(x['order_sha256'],x['document']))
 seen_raw=set(historical_raw);seen_norm=set(historical_norm);seen_tokens=set(historical_full_tokens);seen_prefix=set(historical_prefix_tokens);eligible=[]
 for rank,row in enumerate(ordered):
  row['candidate_order_rank']=rank
  if row['eligible_before_new_pool_dedup']:
   if row['raw_sha256'] in seen_raw:row['exclusion_reasons'].append('NEW_POOL_RAW_DUPLICATE')
   if row['normalized_text_sha256'] in seen_norm:row['exclusion_reasons'].append('NEW_POOL_NORMALIZED_DUPLICATE')
   if row['canonical_full_token_sha256'] in seen_tokens:row['exclusion_reasons'].append('NEW_POOL_FULL_TOKEN_DUPLICATE')
   if row['canonical_prefix_token_sha256'] in seen_prefix:row['exclusion_reasons'].append('NEW_POOL_PREFIX_TOKEN_DUPLICATE')
   if not row['exclusion_reasons']:
    row['eligible_rank']=len(eligible);eligible.append(row)
    seen_raw.add(row['raw_sha256']);seen_norm.add(row['normalized_text_sha256']);seen_tokens.add(row['canonical_full_token_sha256']);seen_prefix.add(row['canonical_prefix_token_sha256'])
  row['eligible']=not row['exclusion_reasons'];row['selection_role']='NOT_SELECTED'
 assert len(eligible)>=7,'INSUFFICIENT_7_UNSEEN_LONG_DOCUMENTS_STOP'
 formal=eligible[:6];canary=eligible[6]
 for i,row in enumerate(formal):row['selection_role']='FORMAL';row['formal_document_order']=i
 canary['selection_role']='QUALIFICATION_CANARY_ONLY'
 assert len({x['document'] for x in formal+[canary]})==7
 fields=['candidate_order_rank','document','source_split','generation','raw_bytes','raw_sha256','normalized_text_sha256','full_token_count','canonical_full_token_sha256','canonical_prefix_token_sha256','order_sha256','eligible','eligible_rank','selection_role','formal_document_order','exclusion_reasons']
 audit=D/'DATA_SELECTION_AUDIT.csv'
 with audit.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for row in ordered:writer.writerow({k:json.dumps(row[k]) if k=='exclusion_reasons' else row.get(k,'') for k in fields})
  f.flush();os.fsync(f.fileno())
 frozen={'TASK_ID':TASK,'status':'PASS_CPU_DATA_COHORT_FROZEN_BEFORE_GPU_FORWARD','frozen_utc':now(),'formal_documents':formal,'canary_document':canary,'target_documents':6,'formal_trajectories':30,'tokens_per_trajectory':8448,'minimum_input_tokens':8449,'positions':list(range(8192,8448)),'condition_order':protocol['condition_order'],'PROJECT_INDEPENDENCE':'UNKNOWN','source_dataset':'PG19','source_split':'validation','RepoBench_P_used':False,'data_selection_protocol_path':str(D/'audits/DATA_SELECTION_PROTOCOL.json'),'data_selection_protocol_sha256':protocol_sha,'data_selection_audit_sha256':sha(audit),'eligible_pool_count':len(eligible),'frozen_prefix':{'count':8449,'hash_semantics':'SHA256(uint32-little-endian first8449 IDs)','token_file_semantics':'full original CPU-tokenized sequence; consume first8448, retain ids8448 for target identity'},'GPU_model_forward_before_data_freeze':0,'pretraining_contamination':'UNKNOWN; research-task unseen is not model-pretraining unseen','historical_exclusion_scope':protocol['eligibility']['history_metadata_scan']}
 put(D/'FROZEN_DOCUMENTS.json',frozen)
 files=[D/'audits/DATA_SELECTION_PROTOCOL.json',D/'audits/PG19_VALIDATION_GCS_LISTING.json',audit,D/'FROZEN_DOCUMENTS.json',Path(__file__).resolve()]+sorted(raw_dir.iterdir())+sorted(tok_dir.iterdir())
 manifest={'TASK_ID':TASK,'status':'PASS_CPU_ONLY_DATA_SELECTION','finished_utc':now(),'CPU_seconds':time.time()-started,'GPU_forwards':0,'GPU_hours':0.0,'raw_download_requests':len(rows),'downloaded_raw_bytes':downloaded,'declared_cap_bytes':TOTAL_CAP,'application_retries':0,'candidate_objects':50,'eligible':len(eligible),'formal_documents':[x['document'] for x in formal],'canary_document':canary['document'],'PROJECT_INDEPENDENCE':'UNKNOWN','historical_sources':sources,'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
 put(D/'audits/DATA_SELECTION_MANIFEST.json',manifest)
 event({'stage':'DATA_COHORT_FROZEN','formal_documents':manifest['formal_documents'],'canary':manifest['canary_document'],'eligible':len(eligible),'downloaded_raw_bytes':downloaded,'CPU_seconds':manifest['CPU_seconds'],'FROZEN_DOCUMENTS_sha256':sha(D/'FROZEN_DOCUMENTS.json'),'manifest_sha256':sha(D/'audits/DATA_SELECTION_MANIFEST.json'),'GPU_forwards':0})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  failure={'TASK_ID':TASK,'status':'FAIL_CLOSED_DATA_SELECTION_STOP','type':type(e).__name__,'error':str(e),'observed_utc':now(),'GPU_forwards':0,'no_retry':True,'history_modified':False,'source_script_sha256':sha(__file__)}
  p=D/'audits/DATA_SELECTION_FIRST_FAILURE.json'
  if not p.exists():put(p,failure)
  event(failure)
  raise
