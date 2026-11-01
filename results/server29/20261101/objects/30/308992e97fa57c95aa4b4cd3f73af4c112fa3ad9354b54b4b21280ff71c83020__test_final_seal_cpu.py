"""Bounded synthetic CPU seal integration; never uses the real formal output tree."""
import copy
import csv
import datetime
import hashlib
import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import analysis as A
import metrics_cpu as M
import torch

REAL_D=Path('/data/zypan/experiment_runs/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1')
UUID='CPU_SYNTHETIC_SEAL_UUID_NO_GPU'
SCOPE='same_FP32_logits_scoring_only_not_forward_error_bound'
CHECKPOINT='CPU_FIXTURE_TOKEN_SHA_NOT_A_REAL_INPUT'
PREFIX='CPU_FIXTURE_CANONICAL_PREFIX_SHA_NOT_A_REAL_INPUT'
ROOT=Path(tempfile.mkdtemp(prefix='ling_dual_axis_final_seal_cpu_'))
SOURCE_PATHS=[HERE/'analysis.py',HERE/'metrics_cpu.py',HERE/'test_analysis_cpu.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCE_BEFORE={str(p):sha(p) for p in SOURCE_PATHS}
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def read(p):return json.loads(p.read_text())
def update(p,fn):
 v=read(p);fn(v);p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def make_result(doc,cond):
 k={'FP_NATIVE':0.0,'H_R128':8.0,'H_DUAL_M':7.0,'L6_R128':6.0,'L6_DUAL_M':3.0}[cond]
 scores=[{'position':8192+i,'kl_fp32':k,'kl_reference_extended':k,'arithmetic_error_radius':1e-10,'vocabulary_size':157184,'arithmetic_scope':SCOPE} for i in range(256)]
 out={'status':'PASS','kind':'formal','document':doc,'condition':cond,'UUID':UUID,'begin':0,'endpoint':8448,'scores':scores,'aggregate':M.aggregate(scores),'token_file_sha256':CHECKPOINT,'canonical_prefix_token_sha256':PREFIX,'fixture_role':'SYNTHETIC_CPU_SEAL_PARSER_FIXTURE_NOT_SCIENTIFIC_RESULT'}
 if cond!='FP_NATIVE':
  v=.4 if cond.endswith('R128') else .1
  out['local_diagnostics']=[{'native_local_QDQ_relative_mse_per_head':[[v,v],[v,v]]}]
 return out

def make_case(name):
 d=ROOT/name;d.mkdir()
 docs=[{'document':'CPU_SYNTHETIC_DOC_'+str(i),'token_file_sha256':CHECKPOINT,'canonical_prefix_token_sha256':PREFIX} for i in range(6)]
 put(d/'PREREGISTRATION.json',{'TASK_ID':A.TASK,'identity':{'target_uuid':UUID},'fixture':True})
 put(d/'FROZEN_DOCUMENTS.json',{'formal_documents':docs,'fixture':True,'PROJECT_INDEPENDENCE':'UNKNOWN'})
 put(d/'SOURCE_MANIFEST.json',{'fixture':True,'source_sha256':SOURCE_BEFORE[str(HERE/'analysis.py')]})
 put(d/'GPU_RESOURCE_LEDGER.json',{'charged_GPU_hours':1.25,'active_workers':[],'fixture_not_real_GPU_hours':True})
 put(d/'GPU_QUALIFICATION.json',{'status':'PASS_SYNTHETIC_FIXTURE_NOT_REAL_GPU_QUALIFICATION'})
 (d/'controller.log').write_text('SYNTHETIC controller before seal\n')
 (d/'CONTROLLER.lock').write_text('synthetic lock\n');(d/'controller.lock').write_text('synthetic lock\n');(d/'runner.lock').write_text('synthetic lock\n')
 for doc in docs:
  for cond in A.CONDITIONS:
   tag=doc['document']+'__'+cond+'.formal';rpath=d/'formal'/(tag+'.json');job=d/'jobs'/tag
   put(rpath,make_result(doc['document'],cond));put(rpath.with_suffix('.PROFILE.json'),{'fixture':True,'status':'PASS'})
   put(job/'ENV.json',{'fixture':True,'UUID':UUID})
   put(job/'ASSET_BINDING.json',{'fixture':True,'token_file_sha256':CHECKPOINT,'canonical_prefix_token_sha256':PREFIX})
   put(job/'TERMINAL_RESOURCE.PASS.json',{'fixture':True,'status':'PASS_SYNTHETIC_TERMINAL_GATE'})
   receipt={'status':'PASS','qualified_completion':True,'result_path':str(rpath),'result_sha256':sha(rpath),'source_manifest_sha256':sha(d/'SOURCE_MANIFEST.json'),'environment_sha256':sha(job/'ENV.json'),'asset_binding_sha256':sha(job/'ASSET_BINDING.json'),'terminal_gate_sha256':sha(job/'TERMINAL_RESOURCE.PASS.json'),'profile_sha256':sha(rpath.with_suffix('.PROFILE.json')),'fixture_role':'CPU_SYNTHETIC_RECEIPT_NOT_GPU_EXIT'}
   put(job/'EXIT.json',receipt)
 return d,docs

def paths(d,doc=0,cond='L6_DUAL_M'):
 tag='CPU_SYNTHETIC_DOC_'+str(doc)+'__'+cond+'.formal'
 return d/'formal'/(tag+'.json'),d/'jobs'/tag

def verify_manifest(d,receipt):
 m=read(d/'SHA256_MANIFEST.json')
 for group in ['files','large_payloads_server_only']:
  for name,row in m[group].items():
   assert sha(d/name)==row['sha256'] and (d/name).stat().st_size==row['bytes'],(group,name)
 sums={}
 for line in (d/'SHA256SUMS').read_text().splitlines():
  h,name=line.split('  ',1);assert sha(d/name)==h;sums[name]=h
 assert set(sums)==set(m['files'])|{'SHA256_MANIFEST.json'}
 assert receipt['manifest_sha256']==sha(d/'SHA256_MANIFEST.json') and receipt['sums_sha256']==sha(d/'SHA256SUMS')
 assert receipt['report_sha256']==sha(d/'FINAL_SCIENTIFIC_REPORT.md')
 assert 'SHA256_MANIFEST.json' not in m['files'] and 'SHA256SUMS' not in m['files']
 return m

def validate_incomplete(d,expected):
 receipt=A.seal(d);data=read(d/'SCIENCE_RESULTS.json');status=data['status'];verify_manifest(d,receipt)
 assert status['FORMAL_TRAJECTORIES_COMPLETED']==expected
 assert status['FINAL_EXECUTION_STATUS']=='FAIL_CLOSED_STOP' and status['SCIENTIFIC_CLASSIFICATION']=='INCOMPLETE'
 assert not status['FORMAL_SCIENCE_COMPLETE']
 for doc in data['documents']:
  for row in doc['conditions'].values():
   if row['status']!='PASS':assert row['aggregate'] is None
 return receipt,data

results=[]
def case(name,fn):
 try:results.append({'name':name,'status':'PASS','detail':fn()})
 except BaseException as e:results.append({'name':name,'status':'FAIL','type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})

def complete():
 d,docs=make_case('complete30')
 (d/'payloads').mkdir();(d/'checkpoints').mkdir();(d/'audits').mkdir()
 (d/'payloads/fake_logits.npy.partial.4242').write_bytes(b'CPU_SYNTHETIC_PARTIAL_FULLV_PAYLOAD')
 (d/'checkpoints/fake_state.pt.partial.4242').write_bytes(b'CPU_SYNTHETIC_PARTIAL_STATE_PAYLOAD')
 (d/'audits/small_CPU_fixture.pt').write_bytes(b'CPU_SYNTHETIC_NOT_A_CHECKPOINT')
 receipt=A.seal(d);data=read(d/'SCIENCE_RESULTS.json');m=verify_manifest(d,receipt)
 assert receipt['FINAL_EXECUTION_STATUS']=='COMPLETE_AND_STOP' and receipt['SCIENTIFIC_CLASSIFICATION']=='PRIMARY_DIRECTIONALLY_REPLICATED'
 assert data['status']['FORMAL_TRAJECTORIES_COMPLETED']==30 and data['status']['COMPLETE_DOCUMENTS']==6
 assert data['analysis']['metrics']['Delta_L6']['positive']==6 and data['analysis']['confirmatory_sign_test_p'] is None
 assert data['analysis']['project_independence']=='UNKNOWN'
 assert all(x['metrics']['actual_best_student_by_point_estimate']=='L6_DUAL_M' for x in data['documents'])
 for doc in data['documents']:
  mech=doc['local_mechanism_observations'];assert len(mech['conditions'])==4
  for obs in mech['conditions'].values():assert 'own-history' in obs['scope'] and obs['recorded_layer_head_probe_values']==4
  for obs in mech['own_history_codec_pairs'].values():assert obs['directional_concordance'] is True and 'not causal' in obs['interpretation']
 for filename in ['payloads/fake_logits.npy.partial.4242','checkpoints/fake_state.pt.partial.4242']:
  assert filename in m['large_payloads_server_only'] and filename not in m['files']
  assert m['large_payloads_server_only'][filename]['evidence_role']=='PARTIAL_UNQUALIFIED_PAYLOAD'
 assert m['files']['audits/small_CPU_fixture.pt']['evidence_role']=='CPU_TEST_FIXTURE_NOT_SCIENTIFIC_CHECKPOINT'
 assert not any(x in m['files'] for x in ['controller.log','CONTROLLER.lock','controller.lock','runner.lock'])
 before=(sha(d/'SHA256_MANIFEST.json'),sha(d/'SHA256SUMS'))
 with (d/'controller.log').open('a') as f:f.write('SYNTHETIC controller after seal\n')
 assert A.seal(d)==receipt and before==(sha(d/'SHA256_MANIFEST.json'),sha(d/'SHA256SUMS'))
 verify_manifest(d,receipt)
 return {'formal_trajectories':30,'complete_documents':6,'primary_positive':6,'manifest_and_self_SHA_actual':'PASS','CPU_fixture_small_not_checkpoint':True,'partial_payloads_server_only':True,'postseal_controller_append_no_change':True,'own_history_mechanism_exploratory_not_causal':True}
case('complete30_legal_receipts_science_mechanism_manifest_partial_and_log_scope',complete)

def missing():
 d,_=make_case('missing1');r,j=paths(d);r.unlink()
 receipt,data=validate_incomplete(d,29)
 p=data['analysis']['metrics']['Delta_L6'];assert p['paired_documents']==5 and p['missing']==1 and p['positive']==5
 bad=data['documents'][0];assert bad['conditions']['L6_DUAL_M']['status']=='NOT_RUN' and bad['metrics']['Delta_L6'] is None
 return {'completed':29,'primary_paired_n':5,'missing':1,'missing_K':None}
case('missing1_formal_result_not_zero_primary_paired5',missing)

def failed_exit():
 d,_=make_case('rawPASS_exitFAIL');r,j=paths(d)
 update(j/'EXIT.json',lambda x:x.update(status='FAIL',qualified_completion=False))
 put(d/'FIRST_FAILURE.json',{'status':'FIRST_SYNTHETIC_FAILURE','reason':'CPU_FIXTURE_EXIT_FAIL','no_retry':True});first=sha(d/'FIRST_FAILURE.json')
 receipt,data=validate_incomplete(d,29)
 assert read(r)['status']=='PASS' and sha(d/'FIRST_FAILURE.json')==first
 assert data['status']['FIRST_FAILURE']['reason']=='CPU_FIXTURE_EXIT_FAIL'
 return {'raw_artifact_PASS':True,'EXIT_FAIL_not_qualified':True,'completed':29,'first_failure_unchanged':True}
case('raw_PASS_EXIT_FAIL_and_qualifiedFalse_never_counts_complete',failed_exit)

def unqualified():
 d,_=make_case('rawPASS_unqualified');r,j=paths(d)
 update(j/'EXIT.json',lambda x:x.update(qualified_completion=False))
 receipt,data=validate_incomplete(d,29)
 assert read(j/'EXIT.json')['status']=='PASS' and read(r)['status']=='PASS'
 return {'raw_PASS':True,'EXIT_status_PASS_but_qualifiedFalse':True,'completed':29}
case('raw_PASS_EXIT_PASS_qualifiedFalse_never_counts_complete',unqualified)

def wrong_proofs():
 d,_=make_case('wrong_ENV_terminal_SHA')
 _,j0=paths(d,0,'H_R128');_,j1=paths(d,1,'L6_DUAL_M')
 update(j0/'ENV.json',lambda x:x.update(tampered=True));update(j1/'TERMINAL_RESOURCE.PASS.json',lambda x:x.update(tampered=True))
 receipt,data=validate_incomplete(d,28)
 assert len(data['status']['CPU_ANALYSIS_ERRORS'])==2
 assert data['documents'][0]['conditions']['H_R128']['status']!='PASS'
 assert data['documents'][1]['conditions']['L6_DUAL_M']['status']!='PASS'
 return {'completed':28,'ENV_and_terminal_tamper_rejected':True,'CPU_analysis_errors':2}
case('wrong_ENV_or_terminal_SHA_rejects_completion',wrong_proofs)

def all_receipt_bindings():
 d,_=make_case('wrong_all_remaining_proofs')
 alterations=[(0,'L6_DUAL_M','result_sha256'),(1,'L6_DUAL_M','source_manifest_sha256'),(2,'L6_DUAL_M','asset_binding_sha256'),(3,'L6_DUAL_M','profile_sha256'),(4,'L6_DUAL_M','result_path')]
 for index,cond,key in alterations:
  r,j=paths(d,index,cond);update(j/'EXIT.json',lambda x,k=key:x.update({k:'CPU_SYNTHETIC_WRONG_BINDING'}))
 receipt,data=validate_incomplete(d,25)
 assert len(data['status']['CPU_ANALYSIS_ERRORS'])==5 and data['analysis']['metrics']['Delta_L6']['paired_documents']==1
 return {'completed':25,'five_other_receipt_binding_tampers_rejected':True,'primary_paired_n':1}
case('result_source_asset_profile_path_receipt_bindings_all_required',all_receipt_bindings)

def bad_diagnostic():
 d,_=make_case('invalid_diagnostic_before_count');r,j=paths(d)
 update(r,lambda x:x['local_diagnostics'][0].update(native_local_QDQ_relative_mse_per_head=[[-1.0]]))
 update(j/'EXIT.json',lambda x:x.update(result_sha256=sha(r)))
 receipt,data=validate_incomplete(d,29)
 assert read(r)['status']=='PASS' and len(data['status']['CPU_ANALYSIS_ERRORS'])==1
 assert data['documents'][0]['conditions']['L6_DUAL_M']['aggregate'] is None
 return {'completed':29,'invalid_local_probe_rejected_before_count_increment':True,'raw_and_receipt_binding_otherwise_PASS':True}
case('diagnostic_validation_precedes_completed_increment',bad_diagnostic)

def active_worker():
 d,_=make_case('active_worker_failclosed');update(d/'GPU_RESOURCE_LEDGER.json',lambda x:x.update(active_workers=[{'fixture':'not_real_PID'}]))
 try:A.seal(d)
 except AssertionError as e:
  assert str(e)=='SEAL_REQUIRES_NO_ACTIVE_OWN_GPU_WORKERS' and not (d/'FINAL_SEAL_RECEIPT.json').exists()
  return {'active_worker_seal_rejected':True}
 raise AssertionError('ACTIVE_WORKER_ALLOWED_SEAL')
case('seal_rejects_nonempty_active_workers_ledger',active_worker)

def budget():
 d,_=make_case('overbudget_failclosed');update(d/'GPU_RESOURCE_LEDGER.json',lambda x:x.update(charged_GPU_hours=16.0001))
 try:A.seal(d)
 except AssertionError as e:
  assert str(e)=='ABSOLUTE_GPU_BUDGET_EXCEEDED' and not (d/'FINAL_SEAL_RECEIPT.json').exists()
  return {'over16GPUh_synthetic_ledger_rejected':True}
 raise AssertionError('OVERBUDGET_ALLOWED_SEAL')
case('seal_rejects_synthetic_charged_ledger_over16',budget)

def repeat_original45():
 original=HERE/'test_analysis_cpu.py';text=original.read_text();marker="output=D/'audits/CPU_ANALYSIS_TESTS.json'"
 assert text.count(marker)==1
 prefix=text.split(marker)[0]
 scope={'__file__':str(original),'__name__':'CPU_NO_WRITE_REPLAY_ORIGINAL45'}
 exec(compile(prefix,str(original),'exec'),scope)
 report=scope['report']
 assert report['status']=='PASS_CPU_ANALYSIS_TESTS' and report['actual_CPU_cases']==45 and report['failed']==0 and not report['GPU_initialized']
 return {'actual_replayed_CPU_cases':45,'failed':0,'original_report_not_overwritten':True,'mode':'exact original test source prefix before report output; no production monkeypatch','original_test_source_sha256':sha(original),'executed_prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'final_production_sources':report['production_sources_before']}
old_report=REAL_D/'audits/CPU_ANALYSIS_TESTS.json';old_report_sha=sha(old_report)
case('replay_original45_pure_function_fixtures_on_final_analysis_source',repeat_original45)
assert sha(old_report)==old_report_sha,'ORIGINAL_CPU_ANALYSIS_REPORT_CHANGED'
SOURCE_AFTER={str(p):sha(p) for p in SOURCE_PATHS}
all_pass=all(x['status']=='PASS' for x in results) and SOURCE_BEFORE==SOURCE_AFTER and not torch.cuda.is_initialized()
report={'TASK_ID':A.TASK,'status':'PASS_CPU_FINAL_SEAL_TESTS' if all_pass else 'FAIL_CPU_FINAL_SEAL_TESTS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_integration_cases':len(results),'passed':sum(x['status']=='PASS' for x in results),'failed':sum(x['status']=='FAIL' for x in results),'results':results,'replayed_pure_function_cases':45,'actual_GPU_forwards':0,'CUDA_initialized':torch.cuda.is_initialized(),'GPU_hours_real':0.0,'all_GPU_hours_values_inside_fixture_ledgers_are_synthetic':True,'actual_scientific_results_accessed':False,'real_formal_or_qualification_directories_modified':False,'production_sources_modified':False,'source_before':SOURCE_BEFORE,'source_after':SOURCE_AFTER,'source_stable':SOURCE_BEFORE==SOURCE_AFTER,'original_CPU_ANALYSIS_TESTS_sha256_preserved':old_report_sha,'test_source':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'fixture_temp_root':str(ROOT),'fixture_role':'CPU-generated dummy JSON/receipt/payload bytes only; not real GPU completion, qualification, science results or model checkpoints','scope_limits':['Validates seal parser, completion counting, analysis, proof file SHA binding, manifest/SHA self binding and artifact classification on explicit synthetic fixtures.','Does not establish real worker/GPU numerical qualification, actual CUDA/NVML measurement, FP payload correctness, or model-forward science results.','Qualification semantic truth is not inferred from synthetic PASS fixtures.'],'fixture_tree_removed_after_test':True}
shutil.rmtree(ROOT)
output=REAL_D/'audits/CPU_FINAL_SEAL_TESTS.json'
with output.open('x') as f:json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':report['status'],'integration_cases':len(results),'passed':report['passed'],'failed':report['failed'],'failures':[x['name'] for x in results if x['status']=='FAIL'],'replayed_pure45':45,'report_sha256':sha(output),'test_source_sha256':sha(__file__),'source_stable':report['source_stable'],'CUDA_initialized':report['CUDA_initialized']}))
if not all_pass:raise SystemExit(1)
