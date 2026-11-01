"""Independent CPU-only data freeze verification, no tokenizer/model imports."""
import csv,datetime,hashlib,json
from pathlib import Path
D=Path('/data/zypan/experiment_runs/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stream_token_sha(ids):
 h=hashlib.sha256()
 for x in ids:
  assert type(x) is int and 0<=x<157184
  h.update(x.to_bytes(4,'little',signed=False))
 return h.hexdigest()
frozen=read(D/'FROZEN_DOCUMENTS.json');protocol=read(D/'audits/DATA_SELECTION_PROTOCOL.json');manifest=read(D/'audits/DATA_SELECTION_MANIFEST.json')
checks={};checks['all105_recorded_files_current_SHA_match']=all(sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes'] for x in manifest['files'])
checks['historical_bound_sources_unchanged']=all(sha(x['path'])==x['sha256'] for x in manifest['historical_sources'])
checks['corrected_CPU_job_natural_rc0']=read(D/'audits/CPU_DATA_TLS_CORRECTION_OUTCOME.json')['rc']==0
checks['original_TLS_preparation_failure_preserved']=sha(D/'audits/DATA_SELECTION_FIRST_FAILURE.json')=='3d72ca22569ce9a36365c964a6d3dcbbeaec2f7b73a99a4c2106eb5bf472fe1d'
checks['data_protocol_bound']=sha(D/'audits/DATA_SELECTION_PROTOCOL.json')==frozen['data_selection_protocol_sha256']
with (D/'DATA_SELECTION_AUDIT.csv').open() as f:rows=list(csv.DictReader(f))
checks['exactly50_complete_candidates']=len(rows)==50 and len({x['document'] for x in rows})==50
checks['CSV_bound']=sha(D/'DATA_SELECTION_AUDIT.csv')==frozen['data_selection_audit_sha256']
checks['fixed_seed0_validation_dataset']=protocol['seed']==0 and protocol['source_split']=='validation' and frozen['source_split']=='validation'
checks['minimum_model_token_length_all50']=all(int(x['full_token_count'])>=8449 for x in rows)
checks['hash_order_reproduced']=all(hashlib.sha256((frozen['TASK_ID']+'|seed=0|validation|'+x['document']+'|'+x['raw_sha256']).encode()).hexdigest()==x['order_sha256'] for x in rows) and [x['order_sha256'] for x in rows]==sorted(x['order_sha256'] for x in rows)
eligible=[x for x in rows if x['eligible']=='True']
checks['first6_seventh_canary_exact']=[x['document'] for x in frozen['formal_documents']]==[x['document'] for x in eligible[:6]] and frozen['canary_document']['document']==eligible[6]['document']
checks['formal6_canary_disjoint']=len({x['document'] for x in frozen['formal_documents']+[frozen['canary_document']]})==7
selected=[]
for row in frozen['formal_documents']+[frozen['canary_document']]:
 ids=read(row['token_file_path'])
 c={'document':row['document'],'actual_token_count':len(ids),'full_token_count_bound':len(ids)==row['full_token_count'],'raw_sha_bound':sha(row['raw_text_path'])==row['raw_sha256'],'token_file_SHA_bound':sha(row['token_file_path'])==row['token_file_sha256'],'canonical_8449_U32LE_independent_serializer':stream_token_sha(ids[:8449])==row['canonical_prefix_token_sha256'],'canonical_full_U32LE_independent_serializer':stream_token_sha(ids)==row['canonical_full_token_sha256']}
 selected.append(c)
checks['selected7_all_actual_identity_checks']=all(all(v for k,v in x.items() if k not in {'document','actual_token_count'}) for x in selected)
raw_files=list((D/'inputs/raw').iterdir());token_files=list((D/'inputs/tokens').iterdir())
checks['protocol_frozen_before_any_book_payload']=(D/'audits/DATA_SELECTION_PROTOCOL.json').stat().st_mtime_ns<min(x.stat().st_mtime_ns for x in raw_files)
checks['cohort_frozen_after_all50_tokenizations']=(D/'FROZEN_DOCUMENTS.json').stat().st_mtime_ns>=max(x.stat().st_mtime_ns for x in token_files)
checks['raw_exact_one50_no_retry']=manifest['raw_download_requests']==50 and manifest['application_retries']==0 and len(raw_files)==50 and len(token_files)==50
checks['raw_under512MiB']=manifest['downloaded_raw_bytes']==17733002 and manifest['downloaded_raw_bytes']<=512*1024**2
checks['no_project_independence_claim']=frozen['PROJECT_INDEPENDENCE']=='UNKNOWN'
checks['no_GPU_forward_by_selection']=frozen['GPU_model_forward_before_data_freeze']==0 and manifest['GPU_forwards']==0
report={'TASK_ID':frozen['TASK_ID'],'status':'PASS_INDEPENDENT_CPU_DATA_SELECTION_VERIFICATION' if all(checks.values()) else 'FAIL_CLOSED_INDEPENDENT_CPU_DATA_SELECTION_VERIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'selected7':selected,'source_script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'FROZEN_DOCUMENTS_sha256':sha(D/'FROZEN_DOCUMENTS.json'),'DATA_SELECTION_MANIFEST_sha256':sha(D/'audits/DATA_SELECTION_MANIFEST.json'),'actual_manifest_file_count':len(manifest['files']),'raw_payload_bytes':manifest['downloaded_raw_bytes'],'GPU_forwards':0,'scientific_results_read':False}
output=D/'audits/DATA_SELECTION_VERIFICATION.json'
with output.open('x') as f:json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'checks_count':len(checks),'checks':checks,'selected7':selected,'output_sha256':sha(output),'source_sha256':sha(__file__),'FROZEN_DOCUMENTS_sha256':report['FROZEN_DOCUMENTS_sha256'],'manifest_sha256':report['DATA_SELECTION_MANIFEST_sha256']}))
if not all(checks.values()):raise SystemExit(1)
