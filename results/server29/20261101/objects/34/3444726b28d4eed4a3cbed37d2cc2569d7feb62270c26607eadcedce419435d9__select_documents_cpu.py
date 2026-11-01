"""Prospective outcome-blind cached-pool selection; CPU only, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import sys,json,hashlib,datetime,collections
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_selection'
CONFIG=json.loads((OUT/'SELECTION_CONFIG.json').read_text())
RULE=OUT/'SELECTION_RULES.json'

def sha(b):return hashlib.sha256(b).hexdigest()
def fsha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
 assert not (ROOT/'frozen_documents.json').exists(),'No overwrite of frozen selection'
 rule=json.loads(RULE.read_text());assert rule['seed']==20261015
 audit={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_rule_sha256':fsha(RULE),'code_sha256':fsha(__file__),'config_sha256':fsha(OUT/'SELECTION_CONFIG.json'),'status':'IN_PROGRESS','source_verification':[],'pool_verification':[],'CUDA_initialized':False,'model_weights_loaded':False,'GPU_HOURS_USED':0}
 inherited={}
 for d in CONFIG['inherited_sources']:
  p=Path(d['path']);actual=fsha(p);assert actual==d['sha256'],(p,actual,d['sha256'])
  inherited[d['role']]=json.loads(p.read_text())
  audit['source_verification'].append({'role':d['role'],'path':str(p),'sha256':actual,'bytes':p.stat().st_size,'status':'PASS'})
 # Historical registries contain actual selected/used identities, not their candidate inventories.
 ids=set();raw=set();reasons=collections.defaultdict(list)
 def add(docid,rawsha,reason):
  if docid:ids.add(docid);reasons['ID:'+docid].append(reason)
  if rawsha:raw.add(rawsha);reasons['RAW:'+rawsha].append(reason)
 for key in ['EXCLUSION_REGISTRY.json','SELECTION_EXCLUSION_REGISTRY.json']:
  d=inherited[key]
  for x in d['ids']:add(x,None,key)
  for x in d['raw_SHA']:add(None,x,key)
 for key in ['FROZEN_8DOC_SELECTION.json','DOCUMENT_INPUT_ARTIFACTS.json']:
  d=inherited[key]
  docs=d.get('selected_documents',d.get('documents',[]))
  for x in docs:add(x['document_id'],x['raw_text_SHA'],key)
 for x in inherited['FROZEN_CANARY_DOCUMENTS.json']['selected_documents']:
  add(x['document_id'],x['raw_text_sha256'],'old 128-token canary')
 ling=['pg19-test-41603','pg19-test-2544','pg19-test-7987','pg19-test-26183']
 for x in ling:add(x,None,'direct human known exclusion set; original manifest path not supplied, not claimed verified')
 exclusion={'status':'PASS_KNOWN_EXCLUSION_UNION','document_ids':sorted(ids),'raw_text_sha256':sorted(raw),'id_count':len(ids),'raw_sha_count':len(raw),'reasons':dict(sorted(reasons.items())),'inherited_verified_sources':audit['source_verification'],'known_Ling_ids':ling,'Ling_original_manifest':'NOT_PROVIDED_NOT_CLAIMED_VERIFIED','candidate_inventories_not_treated_as_used_documents':True}
 save(OUT/'EXCLUSION_UNION.json',exclusion)
 # Source identities are verified before reading cached text.
 for d in CONFIG['pool_sources']:
  p=Path(d['path']);actual=fsha(p);assert actual==d['SHA'],(p,actual,d['SHA']);assert p.stat().st_size==d['bytes']
  audit['pool_verification'].append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size,'status':'PASS'})
 sys.path.insert(0,CONFIG['transformers_source'])
 from transformers import AutoTokenizer
 from datasets import Dataset
 import torch
 assert not torch.cuda.is_initialized()
 tokenizer_files=[]
 for d in CONFIG['tokenizer_files']:
  p=Path(CONFIG['tokenizer_path'])/d['name'];actual=fsha(p);assert actual==d['sha256']
  tokenizer_files.append({'name':d['name'],'sha256':actual})
 token_identity=sha(canon(tokenizer_files));assert token_identity==CONFIG['tokenizer_identity_sha256']
 tok=AutoTokenizer.from_pretrained(CONFIG['tokenizer_path'],local_files_only=True)
 audit['tokenizer']={'path':CONFIG['tokenizer_path'],'class':type(tok).__name__,'files':tokenizer_files,'identity_sha256':token_identity,'add_special_tokens':False,'chat_template':False}
 # P Arrow and P_e only. The cached P JSONL is verified but not needed: old audit showed exact duplication of the P Arrow 500 rows.
 arrow=next(x for x in CONFIG['pool_sources'] if x['path'].endswith('.arrow'))
 ep=next(x for x in CONFIG['pool_sources'] if x['path'].endswith('repobench-p_e.jsonl'))
 pjson=next(x for x in CONFIG['pool_sources'] if x['path'].endswith('/repobench-p.jsonl'))
 rows=[];source_schemas={}
 for variant,source in [('repobench-p',arrow),('repobench-p_e',ep)]:
  path=Path(source['path'])
  source_rows=Dataset.from_file(str(path)).to_list() if path.suffix=='.arrow' else [json.loads(line) for line in path.open() if line.strip()]
  source_schemas[variant]=sorted(set().union(*(r.keys() for r in source_rows)))
  for idx,r in enumerate(source_rows):
   assert isinstance(r.get('context'),str) and isinstance(r.get('_id'),str)
   text=r['context'];data=text.encode('utf-8');did='long_bench/repobench-p/test/'+r['_id']
   rows.append({'document_id':did,'dataset_id':r['_id'],'raw_text_sha256':sha(data),'source_file':str(path),'source_file_sha256':source['SHA'],'source_variant':variant,'source_row':idx,'text':text,'raw_bytes':len(data),'language':r.get('language'),'project_fields':{k:r[k] for k in ['repository','project','repo','repository_id','project_id','repo_name','repository_name'] if k in r}})
 # Independently ensure verified main JSONL really duplicates the P Arrow; no new candidate source added.
 jrows=[json.loads(x) for x in Path(pjson['path']).open() if x.strip()]
 apairs={(r['dataset_id'],r['raw_text_sha256']) for r in rows if r['source_variant']=='repobench-p'}
 jpairs={(r['_id'],sha(r['context'].encode())) for r in jrows}
 assert apairs==jpairs,'P Arrow/JSONL identity changed'
 audit['P_Arrow_JSONL_exact_duplicate_check']={'status':'PASS','P_Arrow_unique_pairs':len(apairs),'P_JSONL_unique_pairs':len(jpairs),'JSONL_row_count':len(jrows)}
 audit['source_schema']=source_schemas
 seenraw={};idraw={};duplicates=[]
 for r in rows:
  if r['document_id'] in idraw:assert idraw[r['document_id']]==r['raw_text_sha256'],'Conflicting identical document ID'
  idraw[r['document_id']]=r['raw_text_sha256']
  if r['raw_text_sha256'] in seenraw:
   duplicates.append({k:r[k] for k in ['document_id','raw_text_sha256','source_file','source_row']});continue
  seenraw[r['raw_text_sha256']]=r
 candidates=[];eligible=[]
 for i,r in enumerate(sorted(seenraw.values(),key=lambda x:(x['raw_text_sha256'],sha((str(rule['seed'])+'|'+x['document_id']).encode()),x['document_id']))):
  meta={k:v for k,v in r.items() if k not in ['text','project_fields']};meta['selection_rank']=i+1
  meta['project_identity']='UNKNOWN_NOT_IN_SNAPSHOT_SCHEMA' if not r['project_fields'] else r['project_fields']
  meta['exclusion_matches']=reasons.get('ID:'+r['document_id'],[])+reasons.get('RAW:'+r['raw_text_sha256'],[])
  if meta['exclusion_matches']:meta['status']='EXCLUDED_KNOWN_HISTORY'
  else:
   tids=tok(r['text'],add_special_tokens=False,return_attention_mask=False)['input_ids']
   assert all(isinstance(v,int) and v>=0 for v in tids)
   meta['full_token_count']=len(tids);meta['full_token_ids_canonical_sha256']=sha(canon(tids))
   meta['status']='ELIGIBLE' if len(tids)>=32769 else 'INSUFFICIENT_LENGTH'
   if meta['status']=='ELIGIBLE':r['full_ids']=tids;r['meta']=meta;eligible.append(r)
  candidates.append(meta)
  if i%100==0:print('CPU candidate',i+1,'/',len(seenraw),'eligible',len(eligible),flush=True)
 audit['candidate_count']=len(candidates);audit['raw_rows_count']=len(rows);audit['duplicate_rows']=duplicates;audit['candidates']=candidates
 audit['status_counts']=dict(collections.Counter(x['status'] for x in candidates));audit['known_exclusion_id_count']=len(ids);audit['known_exclusion_raw_sha_count']=len(raw)
 # Only explicit field-presence in ALL eligible rows enables deterministic cross-project priority.
 reliable=bool(eligible) and all(bool(r['project_fields']) and all(isinstance(x,str) and x.strip() for x in r['project_fields'].values()) for r in eligible)
 audit['project_fields_reliable']=reliable
 if len(eligible)<2:
  audit['status']='STOP_INSUFFICIENT_ELIGIBLE_DOCUMENTS';save(OUT/'DATASET_SELECTION_AUDIT.json',audit);print(audit['status'],flush=True);return
 chosen=[eligible[0]]
 if reliable:
  project=lambda x:canon(x['project_fields'])
  chosen.append(next((r for r in eligible[1:] if project(r)!=project(chosen[0])),eligible[1]))
 else:chosen.append(eligible[1])
 docs=[]
 for index,r in enumerate(chosen,1):
  directory=ROOT/'documents'/f'doc_{index:02d}';directory.mkdir(parents=True,exist_ok=False)
  rawpath=directory/'raw_context.txt';rawpath.write_bytes(r['text'].encode('utf-8'))
  full=r['full_ids'];artifacts={}
  for label,values in [('full_token_ids',full),('input_ids_32769',full[:32769]),('input_ids_32768',full[:32768])]:
   p=directory/(label+'.json');p.write_bytes(canon(values)+b'\n')
   artifacts[label]={'path':str(p.relative_to(ROOT)),'remote_path':str(p),'count':len(values),'file_sha256':fsha(p),'canonical_json_sha256':sha(canon(values))}
  scoremap=[{'k':k,'token_index_zero_based':8192+64*k,'consumed_token_id':full[8192+64*k],'predicted_next_token_index':8193+64*k,'target_token_id':full[8193+64*k],'window':['Early','Middle','Late'][k//128]} for k in range(384)]
  save(directory/'score_positions.json',{'semantics':'After consuming ids[t], logits predict ids[t+1]; no BOS or chat template','positions':scoremap})
  d=dict(r['meta']);d.update({'document_index':index,'raw_text_copy':str(rawpath.relative_to(ROOT)),'artifacts':artifacts,'consumed_tokens':32768,'retained_minimum_ids':32769,'score_position_count':384,'first_scoring_index':8192,'last_scoring_index':32704,'score_positions_file_sha256':fsha(directory/'score_positions.json'),'history_ID_raw_SHA_exclusions':'PASS','known_Ling_ID_exclusion':'PASS_HUMAN_PROVIDED_SET','calibration_or_parameter_fitting_used':False,'outcome_blind':True})
  save(directory/'DOCUMENT_IDENTITY.json',d);docs.append(d)
 assert docs[0]['raw_text_sha256']!=docs[1]['raw_text_sha256']
 audit['status']='PASS';audit['selected_document_ids']=[x['document_id'] for x in docs];audit['CUDA_initialized']=torch.cuda.is_initialized();assert audit['CUDA_initialized'] is False
 save(OUT/'DATASET_SELECTION_AUDIT.json',audit)
 manifest={'task_id':CONFIG['task_id'],'status':'PASS_FROZEN_2_DOCUMENTS','seed':20261015,'selected_documents':docs,'DOCS_FROZEN':2,'distinct_raw_documents':2,'project_independence':'UNKNOWN' if not reliable else 'EXPLICIT_DIFFERENT_PROJECT_CHECK_RECORDED','repository_project_identity_note':'Distinct text is not evidence of project-level independence','selection_rule_sha256':fsha(RULE),'selection_config_sha256':fsha(OUT/'SELECTION_CONFIG.json'),'selection_code_sha256':fsha(__file__),'selection_audit_sha256':fsha(OUT/'DATASET_SELECTION_AUDIT.json'),'exclusion_union_sha256':fsha(OUT/'EXCLUSION_UNION.json'),'tokenizer':audit['tokenizer'],'source_files':audit['pool_verification'],'old_identity_sources':audit['source_verification'],'known_Ling_ids':ling,'Ling_original_manifest_verification':'NOT_PROVIDED_NOT_CLAIMED_VERIFIED','GPU_forward_count':0,'model_weights_loaded':False,'CUDA_initialized':False,'position_convention':'Zero-based: consume ids[t], score logits predict ids[t+1]; full32769 retained; exactly32768 consumed','new_science_results_seen':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 save(ROOT/'frozen_documents.json',manifest)
 print(json.dumps({'status':manifest['status'],'manifest_sha256':fsha(ROOT/'frozen_documents.json'),'selected_documents':[{'document_id':x['document_id'],'raw_text_sha256':x['raw_text_sha256'],'full_token_count':x['full_token_count'],'input_32768_sha256':x['artifacts']['input_ids_32768']['canonical_json_sha256']} for x in docs],'eligible':len(eligible)},indent=2),flush=True)

if __name__=='__main__':main()
