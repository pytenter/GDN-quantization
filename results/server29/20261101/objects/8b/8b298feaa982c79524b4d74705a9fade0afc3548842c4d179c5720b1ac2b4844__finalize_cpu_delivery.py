"""Finalize CPU preparation. No Torch/CUDA imports, GPU commands, remote writes.
Two steps: --freeze-source, then --seal after CPU checks/staging checks.
"""
import argparse,ast,csv,hashlib,json,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def static_files():
 roots=[R/'scripts',R/'evidence',R/'runtime_presets',R/'candidate_configs']
 paths={p for b in roots for p in b.rglob('*') if p.is_file()}
 paths.update(p for p in (R/'qualification').glob('*.py'))
 paths.update(R/n for n in ['USER_REQUEST.txt','EXECUTION_PLAN.md','EXECUTION_PLAN_FREEZE.json','analysis/STRICT_BASELINE_CONTRACT.json','qualification/CUBLAS_PREREGISTRATION.json','qualification/AUTHORIZATION_TEMPLATE.json','qualification/FILL_INDEPENDENT_GATE.json'])
 return sorted(paths)
def freeze_source():
 c=load(R/'analysis/STRICT_BASELINE_CONTRACT.json');external={}
 for row in c['required_files']+c['model']['weight_shards']+list(c['source_bindings'].values()):
  if 'bytes' not in row or 'sha256' not in row:raise AssertionError('missing boundidentity')
  b={k:row[k] for k in ['path','bytes','sha256']}
  if b['path'] in external:assert external[b['path']]==b,'conflicting external identity'
  external[b['path']]=b
 files=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in static_files()]
 m={'task_id':TASK,'status':'FROZEN_CPU_PREPARATION_GPU_NOT_AUTHORIZED','ALLOW_GPU_DIAG':'NO','execution_sources':files,'external_dependencies':sorted(external.values(),key=lambda x:x['path']),'external_weight_hash_scope':'REUSED_FROZEN_EXPECTATION_MUST_FRESH_VERIFY_BEFORE_GPU','no_current_GPU_qualification':True}
 p=R/'SOURCE_MANIFEST.json'
 if p.exists():assert load(p)==m,'source changed after freeze; no automatic refreeze'
 else:save(p,m)
 freeze={'task_id':TASK,'phase':'CPU_SOURCE_AND_PLAN_FROZEN_BEFORE_ANY_GPU','ALLOW_GPU_DIAG':'NO','source_manifest_sha256':sha(p),'contract_sha256':sha(R/'analysis/STRICT_BASELINE_CONTRACT.json'),'preregistration_sha256':sha(R/'qualification/CUBLAS_PREREGISTRATION.json'),'GPU_dispatches':0}
 p=R/'SOURCE_FREEZE.json'
 if p.exists():assert load(p)==freeze
 else:save(p,freeze)
 print(json.dumps({'status':'SOURCE_FROZEN_GPU_NOT_AUTHORIZED','execution_sources':len(files),'external_dependencies':len(external),'source_manifest_sha256':sha(R/'SOURCE_MANIFEST.json')}))
def seal():
 checks=[]
 def check(n,ok):
  if not ok:raise AssertionError(n)
  checks.append({'check':n,'status':'PASS'})
 frozen=load(R/'EXECUTION_PLAN_FREEZE.json');check('initial_plan_and_request_unchanged',sha(R/'EXECUTION_PLAN.md')==frozen['plan_sha256'] and sha(R/'USER_REQUEST.txt')==frozen['request_sha256'])
 m=load(R/'SOURCE_MANIFEST.json');sf=load(R/'SOURCE_FREEZE.json')
 check('SOURCE_freeze_matches',sha(R/'SOURCE_MANIFEST.json')==sf['source_manifest_sha256'] and sha(R/'analysis/STRICT_BASELINE_CONTRACT.json')==sf['contract_sha256'] and sha(R/'qualification/CUBLAS_PREREGISTRATION.json')==sf['preregistration_sha256'])
 for row in m['execution_sources']:
  p=R/row['path'];check('source_SHA:'+row['path'],p.stat().st_size==row['bytes'] and sha(p)==row['sha256'])
 ledger=load(R/'RESOURCE_LEDGER.json');check('GPU_zero_not_unknown_measurement',ledger['ALLOW_GPU_DIAG']=='NO' and ledger['NEW_GPUh']==0 and ledger['GPU_dispatches']==0 and ledger['events']==[])
 decision=load(R/'FAST_RUNTIME_DECISION.json');check('no_false_fast_class',decision['status']=='CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED' and not decision['FAST_STRICT_EQUIVALENT_VALIDATED'] and not decision['FAST_NEW_PROTOCOL_REPRODUCIBLE'] and not decision['approved_Fast_codecs'])
 check('no_unqualified_Fast_launcher',not(R/'runtime_presets/fast_validated_runtime.sh').exists())
 for n in ['STRICT_RUNTIME_CONTRACT.md','HISTORICAL_EVIDENCE_REUSE.md','CUBLAS_CONFIG_QUALIFICATION.md','UNINITIALIZED_MEMORY_SAFETY_AUDIT.md','NUMERICAL_PARITY_RESULTS.json','PERFORMANCE_COMPARISON.csv','FAST_RUNTIME_DECISION.json','OPTIMIZATION_DIFF.md','RESOURCE_LEDGER.json','FINAL_SCIENTIFIC_REPORT.md']:
  check('deliverable_nonempty:'+n,(R/n).is_file() and (R/n).stat().st_size>100)
 for n in ['qualification/CUBLAS_CPU_TEST_RESULTS.json','qualification/CONTROLLER_CPU_TEST_RESULTS.json','qualification/RESOURCE_EVENT_CPU_QUALIFICATION.json']:
  x=load(R/n);check('CPU_only_PASS:'+n,x['status'].startswith('PASS') or x['status']=='CPU_SYNTHETIC_PASS')
  check('CPU_checks_all_PASS:'+n,all(v.get('status')=='PASS' for v in x['checks']))
 review=load(R/'analysis/CONTROLLER_CPU_REVIEW.json');check('independent_controller_review_PASS',review['status'].startswith('PASS'))
 for p in R.rglob('*.py'):ast.parse(p.read_text())
 check('allPython_sources_AST_parse',True)
 table=list(csv.DictReader((R/'PERFORMANCE_COMPARISON.csv').open()));check('no_new_fake_measurements',all(x['FAST_value']=='' and x['FAST_status']=='NOT_RUN_GPU_NOT_AUTHORIZED' and x['new_GPU_dispatches']=='0' for x in table))
 for row in table:
  p=Path(row['STRICT_source']);p=p if p.is_absolute() else R/p
  check('CSV_source_identity:'+row['metric']+':'+row['codec'],p.is_file() and sha(p)==row['STRICT_source_sha256'])
 # Actual CPU CLI entry checks do not reach any GPU engine or probe.
 p=subprocess.run([sys.executable,'-B',str(R/'scripts/run_cublas_controller.py'),'--root',str(R)],capture_output=True,text=True)
 check('controller_prepare_CLI',p.returncode==0 and json.loads(p.stdout)['GPU_dispatches']==0)
 p=subprocess.run(['sh',str(R/'runtime_presets/strict_runtime.sh'),'--check'],capture_output=True,text=True)
 check('Strict_entry_CPU_identity_CLI',p.returncode==0 and json.loads(p.stdout)['GPU_dispatches']==0)
 # Stage a separate CPU temporary bundle and verify every frozen file there.
 sys.path.insert(0,str(R/'scripts'));from prepare_run_bundle import prepare
 with tempfile.TemporaryDirectory(prefix='qwen_fast_staging_test_') as td:
  dest=Path(td)/'run_copy';result=prepare(R,dest)
  check('independent_CPU_run_bundle_staging',result['GPU_dispatches']==0)
  for row in m['execution_sources']:assert sha(dest/row['path'])==row['sha256']
  check('allstaged_source_bytes_match',True)
  try:prepare(R,R/'forbidden_cpu_seal_child')
  except RuntimeError:denied=True
  else:denied=False
  check('cannot_stage_inside_CPU_seal',denied)
 check('no_engine_imported_in_finalizer','torch' not in sys.modules and not any(n.startswith('transformers') for n in sys.modules))
 report={'task_id':TASK,'status':'PASS_CPU_ONLY_DELIVERY_VERIFICATION','decision':'CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED','GPU_dispatches':0,'NEW_GPUh':0,'checks_count':len(checks),'checks':checks,'source_manifest_sha256':sha(R/'SOURCE_MANIFEST.json'),'not_GPU_runtime_qualification':True}
 save(R/'DELIVERY_VERIFICATION.json',report)
 excluded={R/'SHA256_MANIFEST.json',R/'SHA256SUMS.txt',R/'FINAL_SEAL_RECEIPT.json'}
 files=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and p not in excluded]
 save(R/'SHA256_MANIFEST.json',{'task_id':TASK,'status':decision['status'],'files_count':len(files),'total_bytes':sum(v['bytes'] for v in files),'exclusions':['root SHA256_MANIFEST.json self-reference','root SHA256SUMS.txt derived index','root FINAL_SEAL_RECEIPT.json external seal anchor'],'files':files})
 (R/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+v['path']+'\n' for v in files)+sha(R/'SHA256_MANIFEST.json')+'  SHA256_MANIFEST.json\n')
 assert all((R/v['path']).stat().st_size==v['bytes'] and sha(R/v['path'])==v['sha256'] for v in files)
 receipt={'task_id':TASK,'status':'SEALED_CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED','sealed_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(R/'SHA256_MANIFEST.json'),'SHA256SUMS_sha256':sha(R/'SHA256SUMS.txt'),'source_manifest_sha256':sha(R/'SOURCE_MANIFEST.json'),'final_report_sha256':sha(R/'FINAL_SCIENTIFIC_REPORT.md'),'files_verified':len(files),'bytes_verified':sum(v['bytes'] for v in files),'GPU_dispatches':0,'NEW_GPUh':0,'no_Fast_formal_launcher':True,'seal_receipt_is_external_manifest_anchor':True}
 save(R/'FINAL_SEAL_RECEIPT.json',receipt)
 print(json.dumps(receipt,ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze-source',action='store_true');g.add_argument('--seal',action='store_true');a=p.parse_args();freeze_source() if a.freeze_source else seal()
if __name__=='__main__':main()
