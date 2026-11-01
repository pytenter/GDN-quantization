"""Preparation-only append-only GPU3 assignment; no model or CUDA computation."""
from pathlib import Path
import ast,hashlib,json,os,shutil,subprocess,time
ROOT=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');C=ROOT/'provenance_capsule_v3';A=C/'attempts/gpu3_assignment_attempt03'
UUID='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(name,x):
 with (A/name).open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
def cmd(a):return subprocess.check_output(a,text=True).strip()
def main():
 A.mkdir(parents=True,exist_ok=True)
 head=cmd(['git','-C',str(ROOT),'rev-parse','HEAD']);assert head=='2a761b8c569b3cd579d041b13c81ddebf55dc859'
 original=read(C/'candidate_launch_identity.json')
 assert all(sha(C/r['path'])==r['sha256'] for r in original['files'])
 stop=read(C/'attempts/hash_correction_attempt02/HASH_CORRECTION_AMENDMENT.json')
 assert sha(C/'STOP.json')==stop['first_STOP_sha256'] and sha(C/'FINAL_V3_QUALIFICATION_REPORT.md')==stop['first_report_sha256']
 assert sha(C/'sealed_runner.py')==stop['old_runner']['sha256'] and sha(C/'sealed_runner_attempt02.py')==stop['new_runner']['sha256']
 gpu=cmd(['nvidia-smi','-i','3','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader,nounits']).split(',')
 proc=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'])
 assert gpu[0].strip()=='3' and gpu[1].strip()==UUID and not any(UUID in l for l in proc.splitlines()),'GPU3_ASSIGNMENT_NOT_IDLE'
 frozen=read(C/'frozen_protocol/preregistration.json');drives=[r for r in read(C/'scientific_asset_manifest.json')['files'] if r.get('scientific_role')=='same_GPU6_frozen_FP_drive_manifest']
 assert frozen['hardware']['same_gpu_FP_pairing_required'] and frozen['hardware']['FP_REFERENCES_MIXED_ACROSS_HARDWARE']=='NO'
 env=read(C/'measurement_launcher_environment.json');env['set']['CUDA_VISIBLE_DEVICES']='3';env.pop('GPU6_FORMAL_AUTHORIZED',None);env['GPU3_FORMAL_AUTHORIZED']='NO'
 emit('measurement_launcher_environment.json',env)
 emit('HARDWARE_ASSIGNMENT_AMENDMENT.json',dict(task='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',architecture='PROVENANCE_EXECUTION_CAPSULE_V3',amendment='GPU3_ASSIGNMENT_AMENDMENT_V1',user_authorization='那你就使用GPU3',pre_task_HEAD=head,old_physical_gpu=6,old_uuid=frozen['hardware']['uuid'],proposed_physical_gpu=3,proposed_uuid=UUID,host=cmd(['hostname']),model=gpu[2].strip(),driver=gpu[6].strip(),formal_authorization='CONDITIONAL_ONLY',GPU3_FORMAL_AUTHORIZED='NO',GPU6_FORMAL_AUTHORIZED='NO',FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',cross_hardware_FP_bitwise='FAIL_PRESERVED',scientific_protocol_unchanged=True,prior_STOPs_preserved=True,required_gates=['RUNTIME_IDENTITY_GATE','HARDWARE_ASSIGNMENT_GATE','FP_REFERENCE_PAIRING_GATE','FRESH_PROCESS_DETERMINISM_GATE'],old_FP_drives_for_GPU3_formal_measurement='FORBIDDEN',old_FP_drives_retained_as='READ_ONLY_HISTORICAL_INPUTS',same_GPU3_FP_drives='MUST_BE_CREATED_AND_FROZEN_IF_ADVANCING; not yet available',engineering_fixture_comparison='Historical GPU6 smoke is a parity comparator only, never a GPU3 FP drive or formal paired reference',qualification_only_before_all_blocking_gates=True))
 required_new_peak=4*13325495104+2*1024**3;reserve=20*1024**3;free=shutil.disk_usage(C).free
 assert free>=required_new_peak+reserve,'STORAGE_CAPACITY_FAILURE'
 emit('HARDWARE_STORAGE_PREPARATION.json',dict(gate='PASS',GPU3_FORMAL_AUTHORIZED='NO',uuid=UUID,physical_gpu=3,available_VRAM_bytes=int(gpu[5])*1024**2,available_bytes=free,remaining_same_GPU_FP_drives=4,required_new_peak_bytes=required_new_peak,safety_margin_bytes=reserve,formula='Original coherence rule 4 * 13325495104 plus 2 GiB overhead; unchanged bridge 20 GiB reserve',SUPERVISOR_CLEANUP_LEDGER_REMAINING=1,ACTUAL_TASK_PROCESS_COUNT=0,GPU3_COMPUTE_PROCESS_COUNT=0))
 source=read(C/'historical_source_snapshot_manifest.json')['files'];assert all(sha(C/r['snapshot_relative_path'])==r['original_sha256'] for r in source)
 new=(C/'sealed_runner_attempt03_gpu3.py').read_text();ast.parse(new);compile(new,str(C/'sealed_runner_attempt03_gpu3.py'),'exec')
 emit('STATIC_PREPARATION_TESTS.json',dict(gate='PASS',snapshot_count=len(source),previous_original_candidate_files=len(original['files']),original_runner_and_both_prior_receipts_byte_identity='PASS',new_runner_sha256=sha(C/'sealed_runner_attempt03_gpu3.py'),MODEL_LOADED=False,CUDA_INITIALIZED=False,FORMAL_MEASUREMENTS=0))
 print('GPU3_ASSIGNMENT_PREPARED_NOT_FORMALLY_AUTHORIZED',UUID,flush=True)
if __name__=='__main__':main()
