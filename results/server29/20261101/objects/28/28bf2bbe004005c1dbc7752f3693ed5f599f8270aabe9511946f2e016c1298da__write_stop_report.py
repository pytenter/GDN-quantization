"""Post-STOP read-only evidence audit and append-only failure report generation."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
SOURCE=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v3')
TARGET=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
ROOT=SOURCE if sys.argv[1]=='source' else TARGET
C=ROOT/'provenance_capsule_v3'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
def preservation(name):
 m=read(C/name);bad=[]
 for i,r in enumerate(m['files']):
  p=Path(r['path'])
  if not p.is_file() or sha(p)!=r['sha256']:bad.append(str(p))
  if i%200==0:print('PRESERVATION_AUDIT',i,len(m['files']),flush=True)
 return dict(gate='PASS' if not bad else 'FAIL',checked_count=len(m['files']),mismatches=bad,inventory_sha256=sha(C/name))
def main():
 if ROOT==SOURCE:
  audit=preservation('historical_preservation_before.json')
  old=Path('/data01/user2/.codex/worktrees/GDN-quantization/ling-kda-int8-rc-granularity')
  env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
  status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=old,env=env)
  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old,env=env,text=True).strip()
  audit.update(legacy_HEAD=head,legacy_status_sha256=hashlib.sha256(status).hexdigest(),legacy_status_byte_identity=status==(C/'legacy_git_status_porcelain_v1.txt').read_bytes(),legacy_read_only_operations_only=True)
  assert audit['gate']=='PASS' and audit['legacy_status_byte_identity']
  emit(C/'runs/qualification/source_post_stop_preservation.json',audit)
  print('SOURCE_HISTORY_PRESERVED',json.dumps(audit),flush=True);return
 stop_path=C/'STOP.json';stop=read(stop_path);stop_hash=sha(stop_path)
 assert stop['mode']=='verify' and stop['FORMAL_MEASUREMENTS']==0
 preservation_audit=preservation('target_historical_preservation_before_qualification.json')
 snapshot=read(C/'historical_source_snapshot_manifest.json');bad=[]
 for r in snapshot['files']:
  if sha(C/r['snapshot_relative_path'])!=r['original_sha256'] or r['original_sha256']!=r['snapshot_sha256']:bad.append(r['original_absolute_path'])
 assert not bad
 bridge=next(r for r in snapshot['files'] if r['original_absolute_path'].endswith('/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/scripts/bridge.py'))
 expected=next(l for l in (C/'sealed_runner.py').read_text().splitlines() if 'assert sha(b.__file__)' in l).split("==")[1].strip().strip("'")
 processes=subprocess.check_output(['ps','-eo','pid,ppid,user,args'],text=True)
 task=[l for l in processes.splitlines() if 'sealed_runner.py' in l and 'python' in l]
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
 uuid='GPU-de53f243-b271-521c-d875-d96611ed8574';compute=[l for l in gpu.splitlines() if uuid in l]
 gpu_identity=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid,name,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader'],text=True).strip()
 assert not task and not compute
 launch=read(C/'candidate_launch_identity.json');candidate_bad=[r['path'] for r in launch['files'] if sha(C/r['path'])!=r['sha256']]
 assert not candidate_bad and sha(stop_path)==stop_hash
 audit=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),target_historical_preservation=preservation_audit,source_historical_preservation=read(C/'runs/qualification/source_post_stop_preservation.json'),snapshot_byte_identity=dict(gate='PASS',count=len(snapshot['files']),mismatches=bad),candidate_input_byte_identity=dict(gate='PASS',count=len(launch['files'])),blocker=dict(classification='SOURCE_CAPSULE_EXPECTED_HASH_ASSERTION_FAILURE',location='sealed_runner.py:174',actual_sha256=bridge['original_sha256'],hardcoded_expected_literal=expected,literal_length=len(expected),actual_length=len(bridge['original_sha256']),historical_source_changed=False,cause='Engineering implementation copied the SHA256 literal incorrectly: one extra c (65 rather than 64 hexadecimal characters).',retry_performed=False),STOP_sha256=stop_hash,SUPERVISOR_CLEANUP_LEDGER_REMAINING=1,ACTUAL_TASK_PROCESS_COUNT=len(task),GPU6_COMPUTE_PROCESS_COUNT=len(compute),gpu_identity=gpu_identity,available_bytes=shutil.disk_usage(C).free,git_subprocess_count_during_failed_qualification=stop['git_subprocess_count'],FORMAL_MEASUREMENTS=0)
 emit(C/'runs/qualification/STOP_AUDIT.json',audit)
 gates={
 'V3_CLEAN_WORKTREE_GATE':'PASS (both candidates clean before launch)',
 'V3_SOURCE_SNAPSHOT_BYTE_IDENTITY_GATE':'PASS (125 exact source copies; separate adapter expected-hash assertion FAILED)',
 'V3_PROJECT_LOCAL_DEPENDENCY_CLOSURE_GATE':'NOT_COMPLETED (qualification stopped during entrypoint binding)',
 'V3_NO_DIRTY_WORKTREE_EXECUTION_GATE':'NOT_QUALIFIED (no original dirty source executed before STOP)',
 'V3_RUNTIME_ENVIRONMENT_IDENTITY_GATE':'PREPARATION_PASS; SEALED_RUNTIME_NOT_QUALIFIED',
 'V3_SCIENTIFIC_ASSET_IDENTITY_GATE':'PREPARATION_PASS (406 assets); SEALED_REVALIDATION_NOT_RUN',
 'V3_NO_GIT_DURING_MEASUREMENT_GATE':'NOT_QUALIFIED (failed engineering attempt Git count=0; formal phase not started)',
 'V3_HISTORICAL_FUNCTIONAL_PARITY_SUITE_GATE':'NOT_RUN',
 'V3_FRESH_PROCESS_DETERMINISM_GATE':'NOT_RUN (0/3)',
 'V3_HARDWARE_AND_STORAGE_GATE':'PREPARATION_PASS; NUMERICAL_HARDWARE_PAIRING_NOT_QUALIFIED'}
 emit(C/'runs/qualification/GATE_STATUS_AT_STOP.json',gates)
 report=C/'FINAL_V3_QUALIFICATION_REPORT.md'
 with report.open('x') as f:
  f.write('# PROVENANCE_EXECUTION_CAPSULE_V3 — Final qualification report\n\n')
  f.write('Experiment: `LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1`\n\nTask: `ENGINEERING_ONLY_PROVENANCE_RECOVERY`\n\n')
  f.write('V3 qualification **did not pass**. The first sealed verify attempt stopped before model loading, CUDA measurement, or scientific computation. No retry or source correction was performed after STOP.\n\n')
  f.write('## Blocking evidence\n\n')
  f.write('The engineering adapter at `'+str(C/'sealed_runner.py')+':174` contains an incorrectly transcribed expected hash. The original historical bridge, its source-server snapshot, and its target-server snapshot agree with the frozen source manifest. This is an adapter implementation error, not a demonstrated mutation of historical scientific source or a numerical/scientific failure.\n\n')
  f.write('```text\nACTUAL_AND_MANIFEST_SHA256 = '+bridge['original_sha256']+'\nADAPTER_EXPECTED_LITERAL = '+expected+'\nEXPECTED_LITERAL_LENGTH = '+str(len(expected))+'\nFIRST_ATTEMPT = verify\nFINAL_ACTION = FAIL_AND_STOP\n```\n\n')
  f.write('Immutable first STOP: `'+str(stop_path)+'`\n\nSTOP SHA256: `'+stop_hash+'`\n\n')
  f.write('## Gates\n\n| Gate | Evidence-backed state |\n| --- | --- |\n')
  for k,v in gates.items():f.write('| '+k+' | '+v+' |\n')
  f.write('\nPreparation PASS is not an overall sealed-runtime qualification PASS. The source-snapshot identity gate passed; the independent adapter expected-hash assertion failed. All remaining numerical qualification and post-commit gates are unexecuted.\n\n')
  f.write('## Frozen candidate, not qualified\n\n```text\nUNQUALIFIED_SCAFFOLDING_COMMIT = a2b434c0efac572d7cd83f393c7c8b331012e151\nUNQUALIFIED_INPUTS_COMMIT = '+launch['commit']+'\nCAPSULE_ID = NOT_CREATED\nV3_QUALIFICATION_COMMIT = NOT_CREATED\nV3_PREMEASUREMENT_COMMIT = NOT_CREATED\nPOST_COMMIT_GATES_ALL_PASS = NO\nGPU6_FORMAL_AUTHORIZED = NO\n```\n\n')
  f.write('Execution roots:\n\n- `'+str(SOURCE)+'`\n- `'+str(TARGET)+'` (exact sparse candidate checkout; not a full-history/full-repository checkout)\n\n')
  f.write('## Historical integrity and process state\n\n```text\nV1_STATUS = FAIL_PRESERVED\nV2_STATUS = FAIL_PRESERVED\nV2_RUNTIME_INGRESS_ARCHITECTURE = RETIRED_AS_OVERBROAD_FOR_FORMAL_EXECUTION\nHISTORICAL_SOURCE_PRESERVATION = '+audit['source_historical_preservation']['gate']+'\nHISTORICAL_TARGET_PRESERVATION = '+preservation_audit['gate']+'\nSUPERVISOR_CLEANUP_LEDGER_REMAINING = 1\nACTUAL_TASK_PROCESS_COUNT = '+str(len(task))+'\nGPU6_COMPUTE_PROCESS_COUNT = '+str(len(compute))+'\nGIT_SUBPROCESS_COUNT_DURING_FAILED_QUALIFICATION = 0\nFP_REFERENCES_MIXED_ACROSS_HARDWARE = NO\n```\n\n')
  f.write('Original V1 STOP and all three V2 A-attempt failure receipts are retained in place. '+str(audit['source_historical_preservation']['checked_count'])+' source-side and '+str(preservation_audit['checked_count'])+' target-side historical files were rehashed after STOP. Legacy HEAD/status and source snapshot bytes remain unchanged.\n\n')
  f.write('V2 retirement means runtime-wide regular-file ingress closure was too broad for practical scientific execution; it does not rewrite or invalidate V2 receipts. V3 installs no general ordinary-file/OS metadata classifier and makes no `NO_EXTERNAL_FILES_READ` claim. The claim that all scientific/executable dependencies are qualified cannot yet be made.\n\n')
  f.write('## Runtime and preparation evidence\n\n52 historically frozen package versions matched exactly; Python 3.11.16, Torch 2.7.1+cu128, Transformers 4.57.6, Triton 3.3.1, FLA-core 0.5.2, NumPy 2.4.6; SciPy is not installed (not an expected historical package). 10,179 environment source/extension identities and 156 explicitly identified native library identities were recorded. Third-party source was not copied into the project snapshot.\n\n')
  f.write('GPU6: `'+gpu_identity+'`\n\nFree storage after STOP: '+str(audit['available_bytes'])+' bytes. Historical storage estimate: 2 GiB non-trace overhead plus unchanged 20 GiB reserve; all four frozen same-GPU6 FP drives are reused, not regenerated.\n\n')
  f.write('Decisive receipts:\n\n- `candidate_launch_identity.json`\n- `historical_source_snapshot_manifest.json`\n- `runtime_environment_manifest.json`\n- `scientific_asset_manifest.json`\n- `shared_library_manifest.json`\n- `storage_capacity_identity.json`\n- `runs/qualification/source_post_stop_preservation.json`\n- `runs/qualification/STOP_AUDIT.json`\n- `runs/qualification/GATE_STATUS_AT_STOP.json`\n\n')
  f.write('## Scientific state and stop boundary\n\n```text\nFORMAL_MEASUREMENTS = 0\nDISCOVERY_PRIMARY_COMPLETE = 0/64\nDISCOVERY_SECONDARY_COMPLETE = 0/64\nFRESH_PROCESS_SMOKE_COMPLETE = 0/3\nL9_TRAINING_EXECUTED = NO\nNEW_ROTATION_TRAINING_EXECUTED = NO\nAIME_EXECUTED = NO\nFINAL_ACTION = FAIL_AND_STOP\n```\n\n')
  f.write('No conclusion about the observability hypothesis is authorized. A further explicit engineering correction authorization is required; the first STOP must remain immutable. No automatic retry, Discovery, training, or AIME is permitted from this failed attempt.\n')
 assert sha(stop_path)==stop_hash
 print('FINAL_V3_FAIL_AND_STOP_REPORT_CREATED',str(report),sha(report),flush=True)
if __name__=='__main__':main()
