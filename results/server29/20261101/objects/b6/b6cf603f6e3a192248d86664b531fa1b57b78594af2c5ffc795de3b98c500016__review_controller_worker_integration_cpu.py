#!/usr/bin/env python3
"""Independent stdlib-only cross-interface checks; no engine or GPU use."""
import ast
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
R=Path('/Users/pytenter/Documents/临时项目1/QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1')
O=R/'analysis/final_review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readj(p):return json.loads(p.read_text())
cp=R/'scripts/restoration_controller.py';wp=R/'worker/restoration_worker.py';pp=R/'qualification/GPU_PLAN.json'
cs=cp.read_text();ws=wp.read_text();ct=ast.parse(cs);wt=ast.parse(ws)
checks=[]
def check(name,ok,**evidence):
 assert ok,name
 checks.append(dict(name=name,status='PASS',**evidence))
check('controller_exact_reviewed_final_SHA',sha(cp)=='3c4db67f17a337124d6498f9e24804dd22b12a0df1f1016922b4152fdbf5855e')
check('worker_exact_reviewed_final_SHA',sha(wp)=='e3f63c2535bf00e563f391f570df63c8da5a083720f2c4c5abe74af8aa36730c')
# Extract pure constants and selected stdlib functions only. No module invocation.
constants={'TASK','SCOPE','UUID','UNITS','LIMITS','ENV_KEYS'}
fnames={'require','canonical','stable','sha','read','relative','validate_plan','binding_rows','validate_result','authorize','worker_environment'}
nodes=[]
for n in ct.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets):nodes.append(n)
 elif isinstance(n,ast.FunctionDef) and n.name in fnames:nodes.append(n)
ns={'Path':Path,'json':json,'math':math,'os':os,'hashlib':hashlib}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<controller-integration-stdlib-extract>','exec'),ns)
plan=readj(pp);validated=ns['validate_plan'](plan)
check('actual_plan_matches_controller_fixed_schema_and_roster',validated==plan,units=plan['units'])
check('actual_ROOT_and_authorization_are_NO',readj(R/'ROOT.json')['ALLOW_GPU_RESTORE']=='NO' and readj(R/'HUMAN_GPU_RESTORE_AUTHORIZATION.json')['ALLOW_GPU_RESTORE']=='NO')
for cli,message in [('NO','explicit CLI YES missing'),('YES','ROOT ALLOW_GPU_RESTORE is not YES')]:
 try:ns['authorize'](R,cli,R/'HUMAN_GPU_RESTORE_AUTHORIZATION.json')
 except RuntimeError as e:denied=message in str(e)
 else:denied=False
 check('actual_NO_admission_denies_before_GPU_or_external_hash_'+cli,denied)
report=readj(R/'analysis/environment/CONTROLLER_CPU_TEST_RESULTS.json')
check('existing_69_synthetic_checks_bound_to_current_controller',report['controller_sha256']==sha(cp) and len(report['checks'])==report['checks_passed']==69 and all(x['status']=='PASS' for x in report['checks']) and report['GPU_dispatches']==report['GPU_queries']==0 and report['real_Torch_imports'] is False)
check('worker_32_tests_bound_to_same_current_worker',readj(R/'analysis/protocol/WORKER_PREPARATION_CPU_MOCKS.json')['reviewed_worker_sha256']==sha(wp) and readj(R/'analysis/protocol/WORKER_PREPARATION_CPU_MOCKS.json')['CPU_checks_passed']==32)
# Current allowed-library interface, not the initial worker draft.
check('worker_reads_actual_extra_binding_schema',"extra=r/'analysis/environment/CPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json'" in ws and "read(extra)['native_library_rows']" in ws and 'ADDITIONAL_CRITICAL_NATIVE_BINDINGS.json' not in ws)
primary=readj(R/'analysis/environment/PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json');extra=readj(R/'analysis/environment/CPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json')
rows=primary['native_library_inventory']+extra['native_library_rows'];allowed={}
for row in rows:
 key=row.get('resolved_path',row['path']);old=allowed.get(key)
 check('native_binding_duplicate_consistent_'+str(len(checks)),old is None or (old['bytes'],old['sha256'])==(row['bytes'],row['sha256'])) if old is not None else None
 allowed[key]=row
check('all_current_CPU_mapped_selected_paths_covered_by_worker_union',set(extra['actual_CPU_mapped_selected_paths'])<=set(allowed),allowed_resolved_count=len(allowed),CPU_mapped_selected_count=len(extra['actual_CPU_mapped_selected_paths']))
check('controller_binding_rows_supports_native_and_historical_sources',len(ns['binding_rows'](extra))==len(extra['native_library_rows']) and len(ns['binding_rows'](readj(R/'analysis/HISTORICAL_EXTERNAL_SOURCE_BINDINGS.json')))==194)
# Check frozen local inputs and ensure all plan external binding metadata files are frozen.
freeze=readj(R/'SOURCE_FREEZE.json');seen={}
for row in freeze['files']:
 q=ns['relative'](R,row['path']);assert row['path'] not in seen
 seen[row['path']]=row
 assert q.is_file() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],row['path']
check('actual_source_freeze_local_files_fresh_SHA_bytes',True,frozen_files=len(seen))
check('controller_worker_plan_and_binding_files_source_frozen',{'scripts/restoration_controller.py','worker/restoration_worker.py','qualification/GPU_PLAN.json','USER_REQUEST.txt',*plan['binding_files']}<=set(seen))
check('runtime_authorization_not_circularly_source_frozen','SOURCE_FREEZE.json' not in seen and 'HUMAN_GPU_RESTORE_AUTHORIZATION.json' not in seen and 'ROOT.json' not in seen)
# Compare194-source prospective binding to preserved remote CPU fresh verification, no SSH.
h=readj(R/'analysis/HISTORICAL_EXTERNAL_SOURCE_BINDINGS.json')['external_source_files'];f=readj(R/'analysis/CURRENT_ALL_HISTORICAL_EXTERNAL_SOURCE_VERIFICATION.json');by={x['path']:x for x in f['files']}
check('historical_external194_fresh_CPU_receipt_matches_expected',len(h)==len(by)==194 and all(x['path'] in by and by[x['path']]['passed'] and by[x['path']]['stable'] and by[x['path']]['actual_bytes']==x['bytes'] and by[x['path']]['actual_sha256']==x['sha256'] for x in h) and f['GPU_dispatches']==f['GPU_queries']==0 and not f['imported_torch'])
# Original benchmark root is an isolated rawcopy of the exact7 execution files.
hm=readj(R/'candidate/benchmark_replay/SOURCE_SHA256_MANIFEST.json')
check('all_original_benchmark7_sources_fit_rebound_root',len(hm['execution_sources'])==7 and all((R/'candidate/benchmark_replay'/x['path']).stat().st_size==x['bytes'] and sha(R/'candidate/benchmark_replay'/x['path'])==x['sha256'] for x in hm['execution_sources']))
check('isolated_old_loader_rawSHA_matches_original_assert',sha(R/'candidate/source_tree/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py')=='9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd')
check('benchmark_shim_rebinds_root_and_path_consistently',"m.OLD=oldpath" in ws and "str(r/'candidate/benchmark_replay')" in ws and "p=r/'candidate/benchmark_replay/execution'/a.unit/'RESULT.json'" in ws)
main=next(n for n in wt.body if isinstance(n,ast.FunctionDef) and n.name=='main')
permitline=next(n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='require_worker_permit')
torchline=next(n.lineno for n in ast.walk(main) if isinstance(n,ast.Import) and any(x.name=='torch' for x in n.names))
check('worker_permit_validation_precedes_Torch_import',permitline<torchline)
check('controller_dispatch_CLI_names_match_worker_parser',all(x in cs and x in ws for x in ['--root','--mode','--unit','--permit']))
check('worker_RESULT_keys_match_controller_validator',"j.update(unit_id=a.unit,mode=a.mode,process_identity=process_identity,permit_sha256=digest(a.permit))" in ws)
identity={'pid':12345,'starttick':67890,'cmdline':['python','worker']}
with tempfile.TemporaryDirectory(prefix='restoration_controller_integration_') as t:
 phase=Path(t)
 for unit in [plan['units'][0],plan['units'][2]]:
  obj={'status':'PASS','unit_id':unit['id'],'mode':unit['mode'],'process_identity':identity,'permit_sha256':'digest'}
  (phase/'RESULT.json').write_text(json.dumps(obj))
  check('actual_result_interface_ACCEPTS_'+unit['id'],ns['validate_result'](phase,unit,identity,'digest')==obj)
  obj['process_identity']=dict(identity,starttick=67891);(phase/'RESULT.json').write_text(json.dumps(obj))
  try:ns['validate_result'](phase,unit,identity,'digest')
  except RuntimeError:rejected=True
  else:rejected=False
  check('result_interface_rejects_starttick_mismatch_'+unit['id'],rejected)
check('both_numeric_Pass_required_before_benchmark_dispatch',"['NUMERIC_FRESH0','NUMERIC_FRESH1']" in cs and "Both numeric fresh processes must PASS before performance" in cs and "result=validate_result(phase,unit,worker_id,permit_sha);results.append(result)" in cs)
check('direct_parent_exact_PID_starttick_cmdline_and_permitSHA_required',all(x in cs for x in ["reader(pid)==identity","dispatch['worker_identity']==own","own['cmdline']==permit['worker_command']","dispatch['permit_sha256']==sha(pp)","Prior failure prohibits worker"]))
check('first_exit_and_release_unique_ledger_and_unknown_physical_GPUh',all(x in cs for x in ["unit['id']+':exit'","unit['id']+':release'","last_process_alive_monotonic=last_alive","actual_physical_GPU_hours='UNRESOLVED_NOT_ZERO'","REFUSED_SIGNAL_PROCESS_IDENTITY_NOT_VERIFIED"]))
check('same_phase_library_signature_policy_present',all(x in ws for x in ["a.unit!='NUMERIC_FRESH0' and not a.unit.startswith('NEW_')","(phase+'_LIBRARY_IDENTITY.json')","selection_matches=signature==reference_signature","and selection_matches else 'FAIL_CLOSED_STOP'"]))
check('no_pre_post_maps_equality_requirement','PREFLIGHT' not in ws[ws.index('reference_path='):ws.index("receipt=dict(phase=")] and 'POSTFLIGHT' not in ws[ws.index('reference_path='):ws.index("receipt=dict(phase=")])
check('no_GPU_or_engine_work_in_this_review',True)
res={'status':'PASS_CPU_CONTROLLER_WORKER_INTEGRATION_REVIEW_NOT_GPU_QUALIFICATION','task_id':'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1','controller_sha256':sha(cp),'worker_sha256':sha(wp),'plan_sha256':sha(pp),'checks_passed':len(checks),'checks':checks,'GPU_dispatches':0,'GPU_queries':0,'Torch_imports':0,'did_not_repeat_controller69_tests':True,'scope_limits':['No actual parent/child Linux process launch, NVML, model loading, CUDA, numerical qualifier orbenchmark execution.','Controller69 tests are previous owner-provided synthetic evidence checked bySHA andscope, not rerun.','Local source-freeze bytes are freshlychecked; remote194source freshcheck is savedCPUreceipt reused, no additionalSSH.','Actual current numerical32prefix/twofresh/256timing, GPU memory, deadlines andexit/release remainNOT_RUN.','The controller requires explicitCLIYES+ROOTYES+independenthumanYES withplan/source/request hashes; currentlyrealROOT andhuman recordNO.','SOURCE_FREEZE isfutureexecution identity metadata, notGPUauthorization orscientificPASS.'],'audit_tool_history':[{'event':'STALE_FINDING_WITHDRAWN','fact':'Auditor initially inferred allowed-library filename/key mismatch from old initialworker fragment. Currentactual e3 source and its exact savedreview snapshot both already readCPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json.native_library_rows. Realread+SHA confirmed no currentdefect; rootcorrected theaudit; no sourcechange orGPUtrial occurred.'}],'evidence':[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in [cp,wp,pp,R/'SOURCE_FREEZE.json',R/'analysis/environment/CONTROLLER_CPU_TEST_RESULTS.json',R/'analysis/environment/CONTROLLER_IMPLEMENTATION_REVIEW.json',R/'analysis/CURRENT_ALL_HISTORICAL_EXTERNAL_SOURCE_VERIFICATION.json',R/'analysis/protocol/WORKER_PREPARATION_REVIEW.json']]}
(O/'CONTROLLER_WORKER_INTEGRATION_REVIEW.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':res['status'],'checks':len(checks),'allowed_libraries':len(allowed),'controller_SHA':res['controller_sha256'],'worker_SHA':res['worker_sha256'],'review_SHA':sha(O/'CONTROLLER_WORKER_INTEGRATION_REVIEW.json')}))
