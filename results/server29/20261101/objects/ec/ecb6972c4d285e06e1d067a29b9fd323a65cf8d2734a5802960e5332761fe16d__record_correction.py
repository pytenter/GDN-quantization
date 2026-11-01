"""Append-only hash correction scope and pre-launch resource audit; no CUDA."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,time
ROOT=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
C=ROOT/'provenance_capsule_v3';A=C/'attempts/hash_correction_attempt02'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(name,x):
 with (A/name).open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
def main():
 A.mkdir(parents=True,exist_ok=True)
 before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert before=='4ce17f7e13a82f65416b0a891284a290f1064ce7'
 assert sha(C/'STOP.json')=='d2330b426afbea3e443bc44fc0381b9d4e045fcfa3a18739bd6d5570a85e2fe8'
 assert sha(C/'FINAL_V3_QUALIFICATION_REPORT.md')=='2c37492aef9958b5b6d64c28e3783420d92b1ff3b03fb9f66df78bb0667a5b9b'
 assert sha(C/'sealed_runner.py')=='abbdcc3e60be859ab579fdb1ffd273e9479cf07b3c0c7f872b5000f601f73d95'
 old=(C/'sealed_runner.py').read_text();new=(C/'sealed_runner_attempt02.py').read_text()
 expected=old.replace("C=Path(__file__).resolve().parent\n","C=Path(__file__).resolve().parent\nA=C/'attempts/hash_correction_attempt02'\n")
 expected=expected.replace("C/'candidate_launch_identity.json'","A/'candidate_launch_identity.json'")
 expected=expected.replace('e4baf3f7e3ecc19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd','e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd')
 expected=expected.replace("assert not (C/'STOP.json').exists(),'PRESERVED_V3_STOP'","assert sha(C/'STOP.json')=='d2330b426afbea3e443bc44fc0381b9d4e045fcfa3a18739bd6d5570a85e2fe8','FIRST_STOP_IDENTITY_FAILURE'\n        assert not (A/'STOP.json').exists(),'PRESERVED_ATTEMPT02_STOP'")
 expected=expected.replace("out=C/'runs/qualification';out.mkdir","out=C/'runs/hash_correction_attempt02/qualification';out.mkdir")
 expected=expected.replace("emit(C/'STOP.json',dict(","emit(A/'STOP.json',dict(")
 expected=expected.replace("p.name=='sealed_runner.py'","p.name=='sealed_runner_attempt02.py'")
 assert new==expected,'CORRECTION_SCOPE_CHANGED'
 ast.parse(new);compile(new,str(C/'sealed_runner_attempt02.py'),'exec')
 rows=read(C/'historical_source_snapshot_manifest.json')['files']
 assert all(sha(C/r['snapshot_relative_path'])==r['original_sha256']==r['snapshot_sha256'] for r in rows)
 bridge=next(r for r in rows if r['original_absolute_path'].endswith('/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/scripts/bridge.py'))
 literal=next(l for l in new.splitlines() if 'assert sha(b.__file__)' in l).split('==')[1].strip().strip("'")
 assert len(literal)==64 and literal==bridge['original_sha256']
 protected=[dict(path=r['path'],sha256=r['sha256']) for r in read(C/'candidate_launch_identity.json')['files']]
 assert all(sha(C/r['path'])==r['sha256'] for r in protected),'PREVIOUS_CANDIDATE_CHANGED'
 emit('HASH_CORRECTION_AMENDMENT.json',dict(task='PROVENANCE_EXECUTION_CAPSULE_V3',classification='ENGINEERING_ONLY_EXPECTED_HASH_LITERAL_CORRECTION',user_authorization='批准修正; approval follows explicit request for correcting only the extra-c SHA256 literal while preserving first STOP and creating an append-only attempt',pre_task_HEAD=before,old_runner=dict(path=str(C/'sealed_runner.py'),sha256=sha(C/'sealed_runner.py')),new_runner=dict(path=str(C/'sealed_runner_attempt02.py'),sha256=sha(C/'sealed_runner_attempt02.py')),corrected_expected_sha256=literal,first_STOP_sha256=sha(C/'STOP.json'),first_report_sha256=sha(C/'FINAL_V3_QUALIFICATION_REPORT.md'),scope='Hash literal corrected in separate entrypoint; only launch/run/STOP receipt namespace plumbing changed. All historical scientific and numerical function bytes unchanged.',SCIENTIFIC_PROTOCOL_CHANGED=False,RETRY_STARTED=False,GPU6_FORMAL_AUTHORIZED='NO'))
 emit('STATIC_CORRECTION_TESTS.json',dict(gate='PASS',syntax='PASS',exact_allowed_change_scope='PASS',expected_hash_64_hex_and_manifest_equal='PASS',snapshot_byte_identity_count=len(rows),previous_candidate_byte_identity_count=len(protected),first_STOP_byte_identity='PASS',first_report_byte_identity='PASS',MODEL_LOADED=False,CUDA_INITIALIZED=False,NUMERICAL_QUALIFICATION_EXECUTED=False))
 uuid='GPU-de53f243-b271-521c-d875-d96611ed8574'
 gpu=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid,name,memory.used,memory.free,driver_version','--format=csv,noheader'],text=True).strip()
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
 active=[l for l in procs.splitlines() if uuid in l]
 owners=[]
 for l in active:
  pid=int(l.split(',')[1]);owners.append(dict(gpu_process=l,process_owner=subprocess.check_output(['ps','-p',str(pid),'-o','user='],text=True).strip(),pid=pid))
 audit=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_identity=gpu,active_compute_processes=owners,HARDWARE_ASSIGNMENT_GATE='FAIL' if active else 'PASS',qualification_launch_attempted=False,other_user_processes_modified=False,FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO',SUPERVISOR_CLEANUP_LEDGER_REMAINING=1,ACTUAL_TASK_PROCESS_COUNT=0,GPU6_COMPUTE_PROCESS_COUNT=len(active))
 emit('PRELAUNCH_RESOURCE_AUDIT.json',audit)
 if active:emit('STOP.json',dict(reason='GPU6_CONCURRENT_OCCUPANT',stage='prelaunch_resource_check',resource=audit,engineering_hash_correction='PASS',fresh_process_runs=0,FORMAL_MEASUREMENTS=0,DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64',FINAL_ACTION='FAIL_AND_STOP'))
 with (A/'CORRECTION_REPORT.md').open('x') as f:
  f.write('# V3 expected-hash correction — attempt 02\n\n')
  f.write('The approved hash-only correction is complete in a separate `sealed_runner_attempt02.py` entrypoint. The original runner, first STOP, first report and all original candidate source/input manifests remain byte-identical. No change was made to scientific functions, fixtures, documents, rotations, metrics, thresholds or tolerances.\n\n')
  f.write('Static checks: syntax, exact allowed diff, corrected 64-character hash matching original source manifest, all '+str(len(rows))+' source snapshots and '+str(len(protected))+' previous candidate files: PASS. This is not numerical/runtime qualification.\n\n')
  f.write('GPU6 snapshot: `'+gpu+'`\n\n')
  for r in owners:f.write('Concurrent occupant: account `'+r['process_owner']+'`, PID `'+str(r['pid'])+'`, `'+r['gpu_process']+'`.\n\n')
  f.write('```ini\nHASH_CORRECTION = PASS\nSTATIC_CORRECTION_SCOPE_GATE = PASS\nFIRST_V3_STOP_PRESERVED = YES\nNUMERICAL_QUALIFICATION_LAUNCH = NOT_STARTED\nFRESH_PROCESS_COMPLETE = 0/3\nGPU6_FORMAL_AUTHORIZED = NO\nV3_QUALIFICATION_COMMIT = NOT_CREATED\nV3_PREMEASUREMENT_COMMIT = NOT_CREATED\nFORMAL_MEASUREMENTS = 0\nDISCOVERY_PRIMARY_COMPLETE = 0/64\nDISCOVERY_SECONDARY_COMPLETE = 0/64\nL9_TRAINING_EXECUTED = NO\nNEW_ROTATION_TRAINING_EXECUTED = NO\nAIME_EXECUTED = NO\nSUPERVISOR_CLEANUP_LEDGER_REMAINING = 1\nACTUAL_TASK_PROCESS_COUNT = 0\nGPU6_COMPUTE_PROCESS_COUNT = '+str(len(active))+'\nHARDWARE_ASSIGNMENT_GATE = '+audit['HARDWARE_ASSIGNMENT_GATE']+'\nFINAL_ACTION = '+('FAIL_AND_STOP' if active else 'READY_FOR_QUALIFICATION')+'\n```\n\n')
  f.write('No unrelated GPU process was signalled or terminated, no GPU reassignment was performed, and no model was loaded. Numerical parity, three fresh processes and post-commit gates remain unexecuted. The original first-attempt report remains valid for that attempt; this report is append-only.\n')
 print('HASH_CORRECTION_STATIC_PASS; RESOURCE_GATE='+audit['HARDWARE_ASSIGNMENT_GATE'],flush=True)
if __name__=='__main__':main()
