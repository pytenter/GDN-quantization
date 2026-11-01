"""CPU-only bounded discovery; no document selection, model import, or historical writes.

Default reproduces local checks and records the captured first public-directory query.
--refresh-public-listings performs at most one bounded 8-object query per new split.
This script NEVER downloads book payloads and NEVER freezes a scientific cohort.
"""
import argparse
import collections
import datetime
import hashlib
import json
import struct
import urllib.parse
import urllib.request
from pathlib import Path

TASK = 'LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1'
D = Path('/data/zypan/experiment_runs') / TASK
OLD = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')
TRAIN = Path('/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1/source_traces/CALIBRATION_RAW_TEXTS.jsonl')
FRESH = Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY/protocol/fresh_b_selection_manifest.json')
OFFICIAL_README = 'https://raw.githubusercontent.com/google-deepmind/pg19/master/README.md'
CAPTURED = {
 'validation': {'nextPageToken':'ChR2YWxpZGF0aW9uLzI0MzYwLnR4dA==','items':[
 {'name':'validation/1022.txt','generation':'1568646527821385','size':'68996','md5Hash':'CWkjqgy0nI6nGnndBEtIcw=='},
 {'name':'validation/11155.txt','generation':'1568646527851144','size':'502104','md5Hash':'acMXPDyjgqpXkzaWd1P5wQ=='},
 {'name':'validation/13089.txt','generation':'1568646527832458','size':'576815','md5Hash':'W2quxC7hKXpBQVmynRQsIA=='},
 {'name':'validation/16959.txt','generation':'1568646527818253','size':'188790','md5Hash':'fopsJAtB7WBEdWInBB9l+w=='},
 {'name':'validation/1925.txt','generation':'1568646527829662','size':'325788','md5Hash':'42ULsSWUhuDM6i40bWTVeg=='},
 {'name':'validation/2383.txt','generation':'1568646527834934','size':'1701939','md5Hash':'VctzYGVlGrWRSR7Nw8XFzg=='},
 {'name':'validation/23956.txt','generation':'1568646527898261','size':'172098','md5Hash':'4dSPSKk1HbzOHxUpumFMrg=='},
 {'name':'validation/24360.txt','generation':'1568646527847384','size':'855066','md5Hash':'hmByBVsaP+gOLY4tWIpFXA=='}]},
 'train': {'nextPageToken':'Cg90cmFpbi8xMDAwNy50eHQ=','items':[
 {'name':'train/10.txt','generation':'1568646216462931','size':'4432484','md5Hash':'dPFt8Cg30+r0cE+Jr5twyQ=='},
 {'name':'train/100.txt','generation':'1568646216296164','size':'5568169','md5Hash':'rmyKAPk5jib/dVHdOMWDwg=='},
 {'name':'train/10002.txt','generation':'1568646215338523','size':'285611','md5Hash':'/411M+1rte4DyEPwglXWEA=='},
 {'name':'train/10003.txt','generation':'1568646215432949','size':'359387','md5Hash':'J7Ss29+dbe0xJN+6Gaxo+g=='},
 {'name':'train/10004.txt','generation':'1568646215640855','size':'281543','md5Hash':'WGMrs3uFGG5mPyS04oP1pg=='},
 {'name':'train/10005.txt','generation':'1568646215431537','size':'413545','md5Hash':'7+6TySStCQjhzFLh5OySpw=='},
 {'name':'train/10006.txt','generation':'1568646215345292','size':'74677','md5Hash':'ruRDJA1eRkRBTVUpccq7yw=='},
 {'name':'train/10007.txt','generation':'1568646215673401','size':'158941','md5Hash':'NqLhl2ZY2ap6resnmr4BGQ=='}]}}

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
 return json.loads(Path(path).read_text())
def canonical(values):
 return hashlib.sha256(struct.pack('<' + 'I' * len(values), *values)).hexdigest()
def url(split):
 return 'https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?' + urllib.parse.urlencode({'prefix': split+'/', 'maxResults':'8', 'fields':'items(name,size,generation,md5Hash),nextPageToken'})
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--output',required=True)
 parser.add_argument('--refresh-public-listings',action='store_true')
 args=parser.parse_args()
 output=Path(args.output).resolve()
 if output != (D/'audits/DATA_POOL_DISCOVERY.json').resolve():
  raise ValueError('Output is restricted to the new task audit path')
 if output.exists(): raise FileExistsError(output)
 pool=read(OLD/'PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json')
 historical=read(OLD/'historical_path_identity_inventory.json')
 fresh=read(FRESH)
 train=[json.loads(x) for x in TRAIN.read_text().splitlines()]
 prior=set(historical['path_identity_evidence'])
 protected={x for x in fresh['exclusion_sets'] if x.startswith('pg19-')}
 fresh16={x['document_id'] for x in fresh['streams']}
 original4={'pg19-test-41603','pg19-test-2544','pg19-test-7987','pg19-test-26183'}
 excluded=prior|protected|fresh16|original4
 cache=[]
 ids_by_doc={}
 for row in pool['inputs']:
  raw=Path(row['raw_text_path']); token=Path(row['token_file_path']); ids=read(token)
  checks={'raw_sha256':sha(raw)==row['raw_sha256'], 'token_file_sha256':sha(token)==row['token_file_sha256'], 'canonical_cached_prefix_sha256':canonical(ids)==row['canonical_prefix_token_sha256']}
  cache.append({'document':row['document'],'cached_token_count':len(ids),'raw_sha256':sha(raw),'canonical_8449_prefix_sha256':canonical(ids[:8449]) if len(ids)>=8449 else None,'checks':checks,'previous_CPU_catalog_seen':True,'known_scientific_exclusion':row['document'] in excluded})
  ids_by_doc[row['document']]=ids
 assert len(cache)==100
 remainder=[x for x in cache if x['cached_token_count']>=8449 and not x['known_scientific_exclusion']]
 train_raw={x['raw_text_sha256'] for x in train}
 protected_raw={x['raw_sha256'] for x in cache if x['known_scientific_exclusion']}
 protected_prefix={x['canonical_8449_prefix_sha256'] for x in cache if x['known_scientific_exclusion'] and x['canonical_8449_prefix_sha256']}
 duplicate={'raw_vs_known58_or_train96':[x['document'] for x in remainder if x['raw_sha256'] in protected_raw|train_raw], 'prefix8449_vs_known58':[x['document'] for x in remainder if x['canonical_8449_prefix_sha256'] in protected_prefix], 'remainder_raw_unique':len({x['raw_sha256'] for x in remainder})==len(remainder), 'remainder_prefix8449_unique':len({x['canonical_8449_prefix_sha256'] for x in remainder})==len(remainder)}
 public=[]
 for split in ['validation','train']:
  if args.refresh_public_listings:
   with urllib.request.urlopen(url(split),timeout=30) as f: data=json.load(f)
   origin='CURRENT_BOUNDED_READ_ONLY_METADATA_QUERY'
  else: data=CAPTURED[split]; origin='CAPTURED_FROM_FIRST_BOUNDED_PUBLIC_DIRECTORY_QUERY_THIS_TASK'
  public.append({'split':split,'url':url(split),'origin':origin,'response':data,'canonical_response_sha256':hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'payload_downloaded':False,'token_count_qualified':False,'cohort_selected':False})
 report={'TASK_ID':TASK,'phase':'0C_DISCOVERY_ONLY','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_initializations':0,'model_forwards':0,'history_files_modified':False,'documents_selected':0,'documents_frozen':0,'PROJECT_INDEPENDENCE':'UNKNOWN','REPObench_P_asset_status':'NOT_LOCATED_IN_BOUNDED_TRUSTED_LOCAL_ASSET_SEARCH; not a claim of global absence','existing_pool_identity':{'dataset':'PG19','split':'test','official_source':pool['official_source'],'all100_previously_CPU_catalogued':True,'at_least8449_tokens':sum(x['cached_token_count']>=8449 for x in cache),'raw_and_token_cache_hash_all_PASS':all(all(x['checks'].values()) for x in cache)},'exclusions':{'minimum_known_ids':sorted(excluded),'count':len(excluded),'legacy_registry20_incomplete':True,'supplement':'Fresh-B protected exclusion_sets plus16 streams and original4','complete_historical_exclusion_qualification':'NOT_YET_ASSERTED'},'old_test_pool_feasibility':{'after_minimum_known58_count':len(remainder),'candidate_ids_are_NOT_a_selection':[x['document'] for x in remainder],'duplicate_checks':duplicate,'strict_never_CPU_catalogued_count':0,'never_method_forward_status':'REQUIRES_COMPLETE_HISTORICAL_MANIFEST_CONTENT_AUDIT'},'new_split_discovery':{'official_readme':OFFICIAL_README,'official_readme_declared_books':{'train':28602,'validation':50,'test':100},'public_directory_queries':public,'feasibility':'OFFICIAL_NEW_SPLITS_REACHABLE; >=6_QUALIFIED_UNSEEN_LONG_DOCUMENTS_NOT_YET_PROVEN','validation_preferred_over_train':'Unseen by this research task is separate from base-model pretraining contamination; neither split proves no pretraining overlap','PROJECT_INDEPENDENCE':'UNKNOWN; book/title metadata do not establish repository/project independence'},'recommended_predeclared_selection':{'seed':0,'domain':TASK,'sort':'ascending SHA256(UTF8(TASK_ID+\"|seed=0|\"+source_split+\"|\"+document_id+\"|\"+raw_sha256)); document_id tie-break','filters':'all identified historical training/selection/tuning/diagnostic/canary/formal exclusions; raw + canonical tokens + shared-passage dedup; length>=8449; no method-error reads','formal_cohort':'first6 eligible only after all gates; not executed by this discovery','canary':'separate preregistered hash-domain; disjoint from formal6'},'sources':[{'path':str(p),'sha256':sha(p)} for p in [OLD/'PG19_TEST_GCS_LISTING.json',OLD/'PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json',OLD/'historical_path_identity_inventory.json',FRESH,TRAIN]],'source_script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'cache_checks':cache,'status':'CPU_ONLY_DATA_DISCOVERY_COMPLETE_NO_COHORT_FREEZE'}
 output.parent.mkdir(exist_ok=True)
 with output.open('x') as f:json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
 print(json.dumps({'output':str(output),'sha256':sha(output),'source_sha256':sha(__file__),'old_pool_all100_SHA_PASS':report['existing_pool_identity']['raw_and_token_cache_hash_all_PASS'],'known_excluded_count':len(excluded),'remainder_count_NOT_selected':len(remainder),'formal_documents_selected':0,'new_split_token_qualification':'NOT_RUN','GPU_forwards':0}))
if __name__=='__main__':main()
