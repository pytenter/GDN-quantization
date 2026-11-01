"""CPU-only prospective selection. No model outputs, weights, codecs, or GPU calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import collections,csv,datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data_selection'
CFG=json.loads((OUT/'SELECTION_CONFIG.json').read_text());RULE=OUT/'SELECTION_RULE.json'
SEED=20261016

def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def digest(x):return hashlib.sha256(x).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ids_from(x):
 if isinstance(x,list) and all(isinstance(v,int) for v in x):return x
 if isinstance(x,dict):
  for key in ['input_ids','tokens','token_ids','ids']:
   if key in x:
    value=x[key]
    if isinstance(value,list) and len(value)==1 and isinstance(value[0],list):value=value[0]
    if isinstance(value,list) and all(isinstance(v,int) for v in value):return value
 raise ValueError('Unknown historical token artifact schema')

def main():
 assert not (ROOT/'FROZEN_DOCUMENTS.json').exists() and not (ROOT/'frozen_documents.json').exists()
 rule=json.loads(RULE.read_text());assert rule['seed']==SEED
 audit={'TASK_ID':CFG['TASK_ID'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_rule_sha256':sha(RULE),'selection_config_sha256':sha(OUT/'SELECTION_CONFIG.json'),'selection_code_sha256':sha(__file__),'outcome_blind':True,'model_weights_loaded':False,'GPU_forward_count':0,'source_verification':[],'pool_verification':[]}
 inherited={};source_by_role={}
 for x in CFG['inherited_sources']:
  p=Path(x['path']);actual=sha(p);assert actual==x['sha256'],('Inherited source changed',str(p),actual,x['sha256'])
  inherited[x['role']]=json.loads(p.read_text()) if p.suffix=='.json' else None;source_by_role[x['role']]=p
  audit['source_verification'].append({'role':x['role'],'path':str(p),'sha256':actual,'bytes':p.stat().st_size,'status':'PASS'})
 known_ids=set();known_raw=set();reason=collections.defaultdict(set)
 def add(docid,raw,why):
  if docid:known_ids.add(docid);reason['ID:'+docid].add(why)
  if raw:known_raw.add(raw);reason['RAW:'+raw].add(why)
 for key in ['EXCLUSION_REGISTRY.json','SELECTION_EXCLUSION_REGISTRY.json']:
  for value in inherited[key]['ids']:add(value,None,key)
  for value in inherited[key]['raw_SHA']:add(None,value,key)
 for key in ['FROZEN_8DOC_SELECTION.json','DOCUMENT_INPUT_ARTIFACTS.json']:
  for d in inherited[key].get('selected_documents',inherited[key].get('documents',[])):add(d['document_id'],d['raw_text_SHA'],key)
 for d in inherited['FROZEN_CANARY_DOCUMENTS.json']['selected_documents']:add(d['document_id'],d['raw_text_sha256'],'Historical 128-token canary')
 for d in inherited['PILOT_FROZEN_DOCUMENTS']['selected_documents']:add(d['document_id'],d['raw_text_sha256'],'Completed previous2doc32K Pilot')
 prior=inherited['PILOT_EXCLUSION_UNION']
 for value in prior['document_ids']:add(value,None,'Inherited verified Pilot exclusion union')
 for value in prior['raw_text_sha256']:add(None,value,'Inherited verified Pilot exclusion union')
 for value in rule['known_Ling_ids']:add(value,None,'Direct human-provided Ling exclusion set; original manifest not supplied')
 # Preserve the actual selected/used records in the sealed historical registry.
 # Candidate/eligibility inventories never become exclusions merely by existing.
 history_sources=[]
 for x in inherited['EXCLUSION_REGISTRY.json']['source_manifests']:
  p=Path(x['path']);row={'path':str(p),'recorded_sha256':x['SHA'],'identified_exclusion_count':len(x['identified_exclusions'])}
  if p.is_file():row.update(current_sha256=sha(p),status='CURRENT_BYTES_MATCH_RECORDED' if sha(p)==x['SHA'] else 'CURRENT_DIFFERS_PRIOR_IDENTITY_RETAINED')
  else:row['status']='NOT_CURRENTLY_ACCESSIBLE_PRIOR_IDENTITY_RETAINED'
  for e in x['identified_exclusions']:add(e.get('document_id'),e.get('raw_text_SHA'),'Verified historical selected/used registry: '+str(p))
  history_sources.append(row)
 save(OUT/'HISTORICAL_SOURCE_AUDIT.json',{'status':'AUDITED','sources':history_sources,'recorded_registry_sha256':sha(source_by_role['EXCLUSION_REGISTRY.json']),'current_status_counts':dict(collections.Counter(x['status'] for x in history_sources)),'rule':'Historical known-used identities remain excluded even if old source container is unavailable or has changed; never silently discard them','candidate_inventories_excluded_from_history':True})
 for x in CFG['pool_sources']:
  p=Path(x['path']);actual=sha(p);assert actual==x['SHA'] and p.stat().st_size==x['bytes']
  audit['pool_verification'].append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size,'status':'PASS'})
 sys.path.insert(0,CFG['transformers_source'])
 from transformers import AutoTokenizer
 from datasets import Dataset
 import torch
 assert torch.cuda.is_initialized() is False
 tokenizer_files=[]
 for x in CFG['tokenizer_files']:
  p=Path(CFG['tokenizer_path'])/x['name'];actual=sha(p);assert actual==x['sha256'];tokenizer_files.append({'name':x['name'],'sha256':actual})
 tokenizer_sha=digest(canon(tokenizer_files));assert tokenizer_sha==CFG['tokenizer_identity_sha256']
 tok=AutoTokenizer.from_pretrained(CFG['tokenizer_path'],local_files_only=True)
 tokenizer={'path':CFG['tokenizer_path'],'class':type(tok).__name__,'files':tokenizer_files,'identity_sha256':tokenizer_sha,'add_special_tokens':False,'chat_template':False}
 audit['tokenizer']=tokenizer
 # Inspect metadata in actual cached files, not inferred repository names from code.
 metadata_files=[];walked_dirs=0
 for directory in CFG['project_metadata_cache_bases']:
  base=Path(directory)
  if not base.exists():continue
  for current,dirs,files in os.walk(base):
   walked_dirs+=1
   for name in files:
    p=Path(current)/name
    if 'repobench' in name.lower() or ('repobench' in str(p).lower() and p.suffix.lower() in ['.json','.py','.yaml','.yml','.md']):metadata_files.append({'path':str(p),'bytes':p.stat().st_size})
 info_path=Path(CFG['dataset_info_path']);info=json.loads(info_path.read_text())
 project_audit={'status':'PROJECT_INDEPENDENCE_UNKNOWN','attempted_recovery':True,'cache_bases':CFG['project_metadata_cache_bases'],'walked_directories':walked_dirs,'matching_cached_files':metadata_files,'dataset_info_path':str(info_path),'dataset_info_sha256':sha(info_path),'dataset_info_features':sorted(info['features']),'original_project_mapping_found':False,'identity_rule':'No inference from raw hash, filenames, imports or comments; no unprovided upstream manifest claimed verified','no_new_data_download':True}
 arrow=next(x for x in CFG['pool_sources'] if x['path'].endswith('.arrow'));ep=next(x for x in CFG['pool_sources'] if x['path'].endswith('repobench-p_e.jsonl'));pj=next(x for x in CFG['pool_sources'] if x['path'].endswith('/repobench-p.jsonl'))
 raw_rows=[];schemas={}
 for variant,source in [('repobench-p',arrow),('repobench-p_e',ep)]:
  path=Path(source['path']);rows=Dataset.from_file(str(path)).to_list() if path.suffix=='.arrow' else [json.loads(x) for x in path.open() if x.strip()]
  schemas[variant]=sorted(set().union(*(x.keys() for x in rows)))
  for i,r in enumerate(rows):
   assert isinstance(r['_id'],str) and isinstance(r['context'],str)
   text=r['context'];did='long_bench/repobench-p/test/'+r['_id'];raw=digest(text.encode())
   project={k:r[k] for k in ['repository','repo','project','repository_id','project_id','repo_name','repository_name','repository_url'] if k in r}
   raw_rows.append({'document_id':did,'dataset_id':r['_id'],'raw_text_sha256':raw,'source_file':str(path),'source_file_sha256':source['SHA'],'source_variant':variant,'source_row':i,'raw_bytes':len(text.encode()),'language':r.get('language'),'project_fields':project,'text':text})
 jrows=[json.loads(x) for x in Path(pj['path']).open() if x.strip()]
 assert {(x['_id'],digest(x['context'].encode())) for x in jrows}=={(x['dataset_id'],x['raw_text_sha256']) for x in raw_rows if x['source_variant']=='repobench-p'}
 audit['P_Arrow_JSONL_duplicate_verification']={'status':'PASS','Arrow_rows':sum(x['source_variant']=='repobench-p' for x in raw_rows),'main_JSONL_rows':len(jrows)}
 project_audit['actual_row_schemas']=schemas;project_audit['explicit_project_field_rows']=sum(bool(x['project_fields']) for x in raw_rows)
 save(OUT/'PROJECT_METADATA_AUDIT.json',project_audit)
 seen={};id_raw={};duplicates=[]
 for r in raw_rows:
  if r['document_id'] in id_raw:assert id_raw[r['document_id']]==r['raw_text_sha256'],'Conflicting raw text for same ID'
  id_raw[r['document_id']]=r['raw_text_sha256']
  if r['raw_text_sha256'] in seen:duplicates.append({k:r[k] for k in ['document_id','raw_text_sha256','source_file','source_row']});continue
  seen[r['raw_text_sha256']]=r
 unique=list(seen.values());historical_input=set();historical_rows=[];by_id={}
 for i,r in enumerate(unique):
  ids=tok(r['text'],add_special_tokens=False,return_attention_mask=False)['input_ids'];assert all(isinstance(x,int) and x>=0 for x in ids)
  r['full_token_count']=len(ids);r['full_token_ids_canonical_sha256']=digest(canon(ids));r['input_ids_32768_canonical_sha256']=digest(canon(ids[:32768])) if len(ids)>=32768 else None
  r['known_history_reasons']=sorted(reason.get('ID:'+r['document_id'],set())|reason.get('RAW:'+r['raw_text_sha256'],set()))
  if len(ids)>=32769:r['ids']=ids
  if r['known_history_reasons']:
   if len(ids)>=32768:historical_input.add(r['input_ids_32768_canonical_sha256'])
   historical_rows.append({k:r[k] for k in ['document_id','raw_text_sha256','full_token_count','input_ids_32768_canonical_sha256','known_history_reasons']})
  by_id[r['document_id']]=r
  if i%100==0:print(json.dumps({'stage':'CPU_FRESH_TOKENIZATION','processed':i+1,'total':len(unique)}),flush=True)
 # Verify actual historical input artifacts when identified manifests supply them.
 history_artifacts=[]
 for d in inherited['DOCUMENT_INPUT_ARTIFACTS.json']['documents']:
  p=Path(d['token_path']);assert p.is_file() and sha(p)==d['token_SHA'],str(p)
  ids=ids_from(json.loads(p.read_text()));assert len(ids)>=32769
  value=digest(canon(ids[:32768]));historical_input.add(value)
  assert value==by_id[d['document_id']]['input_ids_32768_canonical_sha256']
  history_artifacts.append({'document_id':d['document_id'],'path':str(p),'file_sha256':sha(p),'canonical_32768_sha256':value,'status':'PASS_SOURCE_FRESH_TOKEN_IDENTITY'})
 pilot_root=source_by_role['PILOT_FROZEN_DOCUMENTS'].parent
 for d in inherited['PILOT_FROZEN_DOCUMENTS']['selected_documents']:
  a=d['artifacts']['input_ids_32768'];p=pilot_root/a['path'];assert sha(p)==a['file_sha256']
  ids=ids_from(json.loads(p.read_text()));value=digest(canon(ids));assert len(ids)==32768 and value==a['canonical_json_sha256']==by_id[d['document_id']]['input_ids_32768_canonical_sha256'];historical_input.add(value)
  history_artifacts.append({'document_id':d['document_id'],'path':str(p),'file_sha256':sha(p),'canonical_32768_sha256':value,'status':'PASS_SOURCE_FRESH_TOKEN_IDENTITY'})
 exclusion={'status':'PASS_KNOWN_HISTORY_UNION','document_ids':sorted(known_ids),'raw_text_sha256':sorted(known_raw),'canonical_32768_input_sha256':sorted(historical_input),'document_id_count':len(known_ids),'raw_sha_count':len(known_raw),'historical_canonical_input_count':len(historical_input),'reasons':{k:sorted(v) for k,v in sorted(reason.items())},'actual_used_pool_rows_fresh_tokenized':historical_rows,'verified_input_artifacts':history_artifacts,'known_Ling_ids':rule['known_Ling_ids'],'Ling_original_manifest':'NOT_PROVIDED_NOT_CLAIMED_VERIFIED','provenance_registry_sha256':sha(source_by_role['EXCLUSION_REGISTRY.json'])}
 save(OUT/'EXCLUSION_UNION.json',exclusion)
 ranked=sorted(unique,key=lambda r:(digest((str(SEED)+'|'+r['document_id']+'|'+r['raw_text_sha256']).encode()),r['raw_text_sha256'],r['document_id']))
 seen_input={};eligible=[];records=[]
 for rank,r in enumerate(ranked,1):
  meta={k:v for k,v in r.items() if k not in ['text','ids','project_fields']};meta['selection_rank']=rank;meta['selection_key']=digest((str(SEED)+'|'+r['document_id']+'|'+r['raw_text_sha256']).encode());meta['project_identity']='UNKNOWN' if not r['project_fields'] else r['project_fields'];meta['project_identity_verifiability']='NOT_IN_SNAPSHOT_SCHEMA' if not r['project_fields'] else 'EXPLICIT_SOURCE_FIELD'
  if r['known_history_reasons']:meta['status']='EXCLUDED_HISTORY_ID_OR_RAW'
  elif r['full_token_count']<32769:meta['status']='INSUFFICIENT_LENGTH'
  elif r['input_ids_32768_canonical_sha256'] in historical_input:meta['status']='EXCLUDED_HISTORY_CANONICAL_INPUT'
  elif r['input_ids_32768_canonical_sha256'] in seen_input:meta.update(status='EXCLUDED_DUPLICATE_CANONICAL_INPUT',duplicate_input_of=seen_input[r['input_ids_32768_canonical_sha256']])
  else:
   meta['status']='ELIGIBLE';seen_input[r['input_ids_32768_canonical_sha256']]=r['document_id'];r['meta']=meta;eligible.append(r)
  records.append(meta)
 reliable=bool(eligible) and all(bool(r['project_fields']) and all(isinstance(v,str) and v.strip() for v in r['project_fields'].values()) for r in eligible)
 selected=[]
 if reliable:
  projects=set()
  for r in eligible:
   key=canon(r['project_fields'])
   if key not in projects:selected.append(r);projects.add(key)
   if len(selected)==5:break
  for r in eligible:
   if len(selected)==5:break
   if r not in selected:selected.append(r)
 else:selected=eligible[:5]
 ordinal={r['document_id']:i+1 for i,r in enumerate(selected)}
 for row in records:row['selected_order']=ordinal.get(row['document_id'])
 audit.update(status='PASS' if len(selected)==5 else 'STOP_INSUFFICIENT_LEGAL_DOCUMENTS',raw_source_rows=len(raw_rows),unique_raw_candidates=len(unique),exact_raw_duplicate_rows=duplicates,candidate_status_counts=dict(collections.Counter(x['status'] for x in records)),eligible_count=len(eligible),selected_count=len(selected),selected_ids=[x['document_id'] for x in selected],source_schemas=schemas,PROJECT_INDEPENDENCE='UNKNOWN' if not reliable else 'EXPLICIT_PROJECT_MAPPING_RECORDED',reliable_project_mapping=reliable,candidates=records,exclusion_union_sha256=sha(OUT/'EXCLUSION_UNION.json'),project_metadata_audit_sha256=sha(OUT/'PROJECT_METADATA_AUDIT.json'),CUDA_initialized=torch.cuda.is_initialized())
 save(OUT/'CANDIDATE_AUDIT.json',audit)
 fields=['selection_rank','selection_key','selected_order','status','document_id','dataset_id','source_variant','source_file','source_file_sha256','source_row','raw_text_sha256','raw_bytes','full_token_count','full_token_ids_canonical_sha256','input_ids_32768_canonical_sha256','project_identity','project_identity_verifiability','known_history_reasons','duplicate_input_of','language']
 with (OUT/'CANDIDATE_AUDIT.csv').open('x',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
  for row in records:
   clean=dict(row);clean['known_history_reasons']=json.dumps(clean.get('known_history_reasons',[]),ensure_ascii=False);writer.writerow(clean)
 if len(selected)!=5:print(json.dumps({'status':audit['status'],'eligible':len(eligible)}),flush=True);return
 docs=[]
 for i,r in enumerate(selected,1):
  directory=ROOT/'documents'/f'doc_{i:02d}';directory.mkdir(parents=True,exist_ok=False);raw=directory/'raw_context.txt';raw.write_bytes(r['text'].encode())
  artifacts={}
  for label,values in [('full_token_ids',r['ids']),('input_ids_32769',r['ids'][:32769]),('input_ids_32768',r['ids'][:32768])]:
   p=directory/(label+'.json');p.write_bytes(canon(values)+b'\n');artifacts[label]={'path':str(p.relative_to(ROOT)),'remote_path':str(p),'count':len(values),'file_sha256':sha(p),'canonical_json_sha256':digest(canon(values))}
  scoremap=[{'k':k,'token_index_zero_based':8192+64*k,'consumed_token_id':r['ids'][8192+64*k],'predicted_next_token_index':8193+64*k,'target_token_id':r['ids'][8193+64*k],'window':['Early','Middle','Late'][k//128]} for k in range(384)]
  save(directory/'score_positions.json',{'semantics':'After consuming ids[t], logits predict ids[t+1]','positions':scoremap})
  doc=dict(r['meta']);doc.update(document_index=i,selected_order=i,raw_text_copy=str(raw.relative_to(ROOT)),artifacts=artifacts,consumed_tokens=32768,retained_minimum_ids=32769,score_position_count=384,score_positions_file_sha256=sha(directory/'score_positions.json'),history_ID_raw_and_input_SHA_exclusions='PASS',known_Ling_ID_exclusion='PASS_HUMAN_PROVIDED_SET',calibration_or_parameter_fitting_used=False,outcome_blind=True)
  save(directory/'DOCUMENT_IDENTITY.json',doc);docs.append(doc)
 manifest={'TASK_ID':CFG['TASK_ID'],'task_id':CFG['TASK_ID'],'status':'PASS_FROZEN_5_NEW_DOCUMENTS','seed':SEED,'DOCS_FROZEN':5,'requested_N':5,'selected_documents':docs,'distinct_raw_documents':5,'distinct_canonical_32768_inputs':5,'PROJECT_INDEPENDENCE':audit['PROJECT_INDEPENDENCE'],'project_independence':audit['PROJECT_INDEPENDENCE'],'repository_project_identity_note':'Distinct raw/input hashes do not prove project-level independence','selection_rule_sha256':sha(RULE),'selection_config_sha256':sha(OUT/'SELECTION_CONFIG.json'),'selection_code_sha256':sha(__file__),'candidate_audit_json_sha256':sha(OUT/'CANDIDATE_AUDIT.json'),'candidate_audit_csv_sha256':sha(OUT/'CANDIDATE_AUDIT.csv'),'exclusion_union_sha256':sha(OUT/'EXCLUSION_UNION.json'),'project_metadata_audit_sha256':sha(OUT/'PROJECT_METADATA_AUDIT.json'),'tokenizer':tokenizer,'source_files':audit['pool_verification'],'verified_historical_sources':audit['source_verification'],'known_Ling_ids':rule['known_Ling_ids'],'Ling_original_manifest_verification':'NOT_PROVIDED_NOT_CLAIMED_VERIFIED','GPU_forward_count':0,'model_weights_loaded':False,'CUDA_initialized':False,'new_science_results_seen':False,'position_convention':'Zero-based consume ids[t], logits predict ids[t+1]; exactly32768 inputs and at least32769 retained','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 data=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
 for name in ['FROZEN_DOCUMENTS.json','frozen_documents.json']:
  with (ROOT/name).open('xb') as f:f.write(data)
 print(json.dumps({'status':manifest['status'],'FROZEN_DOCUMENTS_sha256':sha(ROOT/'FROZEN_DOCUMENTS.json'),'eligible_count':len(eligible),'selected':[{'order':d['selected_order'],'document_id':d['document_id'],'full_token_count':d['full_token_count'],'raw_sha256':d['raw_text_sha256'],'input32768_sha256':d['artifacts']['input_ids_32768']['canonical_json_sha256']} for d in docs],'PROJECT_INDEPENDENCE':manifest['PROJECT_INDEPENDENCE']}),flush=True)
 assert torch.cuda.is_initialized() is False

if __name__=='__main__':main()
