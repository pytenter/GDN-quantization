"""Atomic CPU preparation; all prior executions are read-only input assets."""
from pathlib import Path
import ast,hashlib,json,os,shutil,subprocess,sys,time
R=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1');D=R/'execution_05';O=R/'execution_04';E=R/'engineering_repair_05';P=E/'production_candidate';S=P/'staging_execution05';W=Path('/data/zypan/worktrees/ling-kda-prospective-execution05')
HEAD='902d7974c89465df882403e96cc8dcc8dc1c0e85'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=S/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n')
def cp(src,n):
 p=S/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as out:out.write(Path(src).read_bytes())
assert not D.exists(),'EXISTING_EXECUTION05_MUST_NOT_OVERWRITE'
assert not S.exists(),'EXISTING_STAGING_REVIEW_REQUIRED'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==HEAD
assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
sys.path.insert(0,str(W));os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from frozen_runtime_profile import validate_files,CONFIG_SHA256
validate_files()
chain=read(P/'CPU_launch_chain_release_02/CPU_LAUNCH_CHAIN_REGRESSION.json');profile=read(P/'CPU_profile_tests/CPU_FROZEN_PROFILE_TESTS.json')
assert chain['status']=='PASS_CPU_TRUE_LAUNCH_CHAIN' and chain['case_count']==10
assert profile['status']=='PASS_CPU_FROZEN_PROFILE_AND_ENVELOPE_CHECKS' and profile['check_count']==11
for test,key in [(chain,'source_sha256'),(profile,'source_SHA256')]:
 for n,h in test[key].items():assert sha(W/n)==h,n
assert read(P/'CPU_recovery_final/NEW_RECOVERY_TEST_REPORT.json')['test_count']==13
for n in ['GPU_ADMISSION_TEST_REPORT.json','CONTROLLER_RESUME_TEST_REPORT.json']:assert read(P/'CPU_baseline_fixture_corrected'/n)['status']=='PASS'
assert shutil.disk_usage(R).free>=64*1024**3
S.mkdir()
for n in ['FROZEN_SPEC.json','FROZEN_4DOC_SELECTION.json','SELECTION_FREEZE_RECEIPT.json','CALIBRATION_CONFIG.json','CODE_FREEZE.json','ANALYSIS_PLAN.json','SECONDARY_SPEC.json','FROZEN_RUNTIME_ENVIRONMENT_AUDIT.json','INPUT_ELIGIBILITY_REPRODUCTION.json','SHORT_HORIZON_SELECTION.csv','SHORT_HORIZON_SELECTION_STATUS.json','MODEL_WEIGHT_SHA256.txt']:
 cp(O/n,n)
assert sha(S/'FROZEN_SPEC.json')=='8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1'
assert sha(S/'FROZEN_4DOC_SELECTION.json')=='a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b'
for folder in ['preregistration','screening']:
 for src in sorted((O/folder).rglob('*')):
  if src.is_file():cp(src,str(src.relative_to(O)))
for folder in ['runs','formal','qualification_units','runtime_canary','admission','admission_tickets']:(S/folder).mkdir()
for src,n in [
 (E/'CONFIRMED_ROOT_CAUSE.json','CONFIRMED_ROOT_CAUSE.json'),
 (E/'CHECKPOINT_REUSE_DECISION.json','CHECKPOINT_REUSE_DECISION.json'),
 (E/'ENGINEERING_DIAGNOSTIC_RESOURCE_REPORT.json','ENGINEERING_DIAGNOSTIC_RESOURCE_REPORT.json'),
 (E/'DIAGNOSTIC_EVIDENCE_MANIFEST.json','ENGINEERING_DIAGNOSTIC_EVIDENCE_MANIFEST.json'),
 (E/'DIAGNOSTIC_SHA256SUMS.txt','ENGINEERING_DIAGNOSTIC_SHA256SUMS.txt'),
 (P/'CPU_launch_chain_release_02/CPU_LAUNCH_CHAIN_REGRESSION.json','CPU_LAUNCH_CHAIN_REGRESSION.json'),
 (P/'CPU_launch_chain_release_02/CPU_TRUE_LAUNCH_CHAIN_EVIDENCE.tar.gz','CPU_TRUE_LAUNCH_CHAIN_EVIDENCE.tar.gz'),
 (P/'CPU_profile_tests/CPU_FROZEN_PROFILE_TESTS.json','CPU_FROZEN_PROFILE_TESTS.json'),
 (P/'CPU_DELIVERY_COMPLETION_TEST_RELEASE.json','CPU_DELIVERY_COMPLETION_TEST.json')]:cp(src,n)
for folder in ['CPU_baseline_final','CPU_baseline_fixture_corrected','CPU_recovery_final']:
 for src in (P/folder).rglob('*'):
  if src.is_file():cp(src,'CPU_reverification/'+folder+'/'+str(src.relative_to(P/folder)))
cp(Path(__file__),'PREPARATION_CPU_SOURCE.py')
reconcile=read(O/'PREREGISTRATION_RECONCILIATION.json')
reconcile['historical_STOP_preserved'][str(O/'STOP.json')]='00d9626ad165a992b1cf2dc0c406bc3bb8f5822377090add6667973c9c600d3c'
for n,h in reconcile['historical_STOP_preserved'].items():assert sha(n)==h,n
reconcile['authorization_reconciliation']='User authorized autonomous engineering repair after execution04 STOP; restore original frozen profile; independently qualify all12 unchanged keys on GPU2/3 before formal.'
reconcile['current_execution']=str(D);reconcile['source_recovery_audit']=str(E/'CONFIRMED_ROOT_CAUSE.json');reconcile['new_long_GPU_scientific_units']=0
write('PREREGISTRATION_RECONCILIATION.json',reconcile)
routing=read(O/'EXECUTION_ROUTING.json');routing.update(execution=str(D),source_worktree=str(W),reference_experiment=str(D),prior_stopped_executions=[str(R/f'execution_{i:02d}') for i in range(1,5)],current_device_preflight_required='GPU_PREFLIGHT_WINDOW_01_TO_03_BOUNDED_THEN_EVERY_WORKER_60_SECONDS',routing_rebound_epoch=time.time(),user_authorization='你自己来进行修复吧；GPU2/3；原12科学key；全部原数值资格后原四文档正式32K',historical_prefixes_reused=0,qualification_new_parents=12,qualification_new_fresh64=12,full_FLA_profile='recovered original six-config profile, every process before import, actual post-forward attestation')
for k in ['preparation_old_window_sha256','prior_preparation_routing_sha256']:routing.pop(k,None)
write('EXECUTION_ROUTING.json',routing)
write('REUSED_PARENT_BINDINGS.json',dict(status='NO_PREFIX_REUSE_UNATTESTED_FULL_PROFILE',units={},decision_sha256=sha(E/'CHECKPOINT_REUSE_DECISION.json'),science_key_count=12,new_parent_workers=12,new_fresh64_workers=12))
write('FP_PAIRING_CONTRACT_GATE.json',dict(status='PASS',mode='EXECUTION05_OWN_MATCHED_FP_SAME_DOCUMENT_UUID_RUNTIME',reference_experiment=str(D),spec_sha256=sha(S/'FROZEN_SPEC.json'),actual_FP_payloads='PENDING_FORMAL'))
audit=read(P/'NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json');audit.update(source_commit=HEAD,source_SHA256={n:sha(W/n) for n in audit['allowed_modified_engineering_files']},fixed_FLA_config_SHA256=CONFIG_SHA256)
write('NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json',audit)
phase0=read(P/'PHASE0_HISTORICAL_ASSET_AUDIT.json');phase0['source_commit']=HEAD;phase0['original_execution02_checkpoint_rows']=phase0.pop('checkpoints');phase0['historical_checkpoints']=read(E/'CHECKPOINT_REUSE_DECISION.json')['actual_byte_SHA_verified']
write('PHASE0_HISTORICAL_ASSET_AUDIT.json',phase0)
files=subprocess.check_output(['git','ls-files'],cwd=W,text=True).splitlines();files=[n for n in files if n.endswith('.py') or n.startswith('frozen_fla_configs/') or n=='frozen_runtime_provenance.json'];hashes={n:sha(W/n) for n in files}
for n in files:
 target=S/n;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(W/n)
formal=read(O/'FORMAL_CODE_FREEZE.json');formal.update(source_commit=HEAD,files=hashes)
formal['dependencies'].update(read(W/'frozen_runtime_provenance.json')['installed_runtime_source_SHA256'])
for n,h in formal['dependencies'].items():assert sha(n)==h,n
write('FORMAL_CODE_FREEZE.json',formal)
freeze=dict(source_commit=HEAD,base_commit='d3cd88d4643a460e027cae6ba6989628c21f60bf',files=hashes,config_sha256=sha(S/'CALIBRATION_CONFIG.json'),routing_sha256=sha(S/'EXECUTION_ROUTING.json'),scientific_spec_sha256=sha(S/'FROZEN_SPEC.json'),runtime_environment_audit_sha256=sha(S/'FROZEN_RUNTIME_ENVIRONMENT_AUDIT.json'),true_CPU_chain_regression_sha256=sha(S/'CPU_LAUNCH_CHAIN_REGRESSION.json'),CPU_profile_test_sha256=sha(S/'CPU_FROZEN_PROFILE_TESTS.json'),epoch=time.time())
write('SOURCE_CODE_FREEZE.json',freeze)
cp_bytes=subprocess.check_output(['git','diff','d3cd88d4643a460e027cae6ba6989628c21f60bf',HEAD,'--'],cwd=W)
with (S/'SOURCE_DIFF.patch').open('xb') as out:out.write(cp_bytes)
write('ENGINEERING_READINESS.json',dict(status='CPU_CANDIDATE_FROZEN_CANARY_PENDING',source_commit=HEAD,ENGINEERING_COMPLETE=False,NUMERICAL_QUALIFICATION_COMPLETE=False,FORMAL_SCIENCE_COMPLETE=False,CPU_real_launch_cases=10,CPU_profile_envelope_checks=11,CPU_recovery_and_admission_cases=37,CPU_delivery_identity_checks=4,all_CPU_cases_PASS=True,canary_GPU_workers_planned=2,canary_scientific_units=0,new_qualification_GPU_workers_planned=24,old_checkpoints_reused=0,old_checkpoints_SHA_preserved=10,formal_planned_FP=4,formal_planned_students=8,math_workers_exact_bytes_unchanged=True))
write('EXECUTION05_REPAIR_AND_RUN_PLAN.json',dict(status='FROZEN_BEFORE_NEW_GPU',source_commit=HEAD,original_selection_and_64_scores_unchanged=True,preregistration_SHA='54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722',selection_SHA=sha(S/'FROZEN_4DOC_SELECTION.json'),kernel_profile_authority=str(W/'frozen_runtime_provenance.json'),diagnosis_manifest=str(E/'DIAGNOSTIC_EVIDENCE_MANIFEST.json'),diagnosis_manifest_sha256=sha(E/'DIAGNOSTIC_EVIDENCE_MANIFEST.json'),qualification_workers=24,new_scientific_keys=0,canary_workers=2,formal_workers=12,formal_global_order=routing['formal_global_order'],qualification_parallel_lanes=True,formal_serial=True,first_failure_no_retry=True,no_science_formula_or_tolerance_changes=True,old_scientific_executions_read_only=True))
# Capture only read-only telemetry. A snapshot is never formal admission.
os.environ['LING_PROSPECTIVE_RESULTS']=str(S)
from gpu_telemetry import sample
from gpu_admission import graphics_allowed
snapshots=[]
for uuid,index in sorted(routing['authorized_devices'].items(),key=lambda x:x[1]):
 sample_row=sample(uuid);assert len(sample_row['target'])==1 and int(sample_row['target'][0]['index'])==index
 assert graphics_allowed(sample_row,routing['baseline_graphics'][uuid]),'CURRENT_GRAPHICS_BASELINE_NOT_VERIFIED'
 snapshots.append(sample_row)
write('INITIAL_AUTHORIZED_GPU_SNAPSHOT.json',dict(epoch=time.time(),snapshots=snapshots,formal_admission=False))
write('INITIAL_CPU_STORAGE_STATUS.json',dict(epoch=time.time(),disk_free_bytes=shutil.disk_usage(R).free,physical_minimum_bytes=64*1024**3,MemAvailable=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')],GPU_not_started=True))
(S/'EXECUTION_PLAN.md').write_text('execution_05: restore the original frozen FLA launch profile; 37+10+11+4 CPU checks PASS. Two minimal runtime canaries, then 12 own parent +12 fresh64 target-device qualifications, then unchanged global serial 12 formal trajectories only after all strict gates PASS. No historical prefix profile attestation means no old parent reuse; old CPs remain read-only. Exact original formulas, four docs, 64 screening scores, rotations, thresholds and first-failure/no-retry rules remain frozen.\n')
assert not D.exists();os.rename(S,D)
print(json.dumps(dict(status='ATOMIC_CPU_PREPARATION_COMPLETE_GPU_NOT_STARTED',execution=str(D),source_commit=HEAD,source_files=len(hashes),qualification_workers=24,formal_workers=12)))
