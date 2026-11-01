"""CPU-only final closure, including failures. Never launches GPU/AIME/scoring."""
import time
import persistent_v2_common as A
def main():
 A.verify_manifest();matrix=A.read(A.ROOT/'configs/v2_worker_matrix.json')['runs'];workers={};observed=[]
 for j in matrix:
  p=A.ROOT/'analysis'/f"{j['key']}_worker.json"
  if not p.exists():A.save(p,dict(status='NOT_RUN',ready='NO',job=j,PROCESS_EXIT_RECLAIM_GATE='NOT_RUN',reason='prospective fail-closed stop; no retries'))
  w=A.read(p);workers[j['key']]=w
  rp=A.ROOT/'analysis'/f"{j['key']}_worker_result.json"
  if rp.exists():observed.append(A.read(rp))
 determinism={}
 for arm in ('C6R','C7'):
  paths=[A.ROOT/'analysis'/f'gpu0_{arm.lower()}{suffix}_worker_result.json' for suffix in ('','_repeat_b')]
  status='NOT_RUN';comparison=None
  if all(p.exists() for p in paths):
   one,two=map(A.read,paths)
   def canonical(r):return dict(semantic=r['initial_semantic'],cycles_semantic=[c['semantic'] for c in r['cycles']],generated=[c['record']['generated_token_ids'] for c in r['cycles']],rotation=r['rotation_SHA'],runtime=r['runtime'],config=r['config_SHA'])
   comparison=dict(A_fingerprint=A.digest(canonical(one)),B_fingerprint=A.digest(canonical(two)));status='PASS' if canonical(one)==canonical(two) else 'FAIL'
  determinism[arm]=status
  A.save('analysis/gpu0_'+arm.lower()+'_fresh_process_determinism.json',dict(status=status,comparison=comparison,exact_policy=True,no_cross_hardware_tolerance=True))
 primary={k:w for k,w in workers.items() if w['job']['repeat']=='A'}
 allpass=all(w['ready']=='YES' for w in primary.values()) and all(s=='PASS' for s in determinism.values()) and len(observed)==10 and all(r['status']=='PASS' for r in observed)
 mem={r['key']:dict(status=r['RESIDENT_WORKER_MEMORY_STABILITY_GATE'],cycles=[c['memory'] for c in r['cycles']]) for r in observed}
 A.save('analysis/resident_worker_memory_stability.json',dict(status='PASS' if allpass else 'FAIL',measured_workers=len(observed),expected_workers=10,workers=mem,endpoint_criterion='allocated and reserved exact byte plateau, no tolerance; warmup excluded'))
 exitgates={k:w['PROCESS_EXIT_RECLAIM_GATE'] for k,w in workers.items()};tested=[s for s in exitgates.values() if s!='NOT_RUN']
 exitstatus='NOT_RUN' if not tested else 'PASS' if all(s=='PASS' for s in tested) else 'FAIL'
 A.save('analysis/process_exit_reclaim.json',dict(status=exitstatus,coverage=len(tested),required=10,workers=exitgates,NVML_resolution='BYTES',timeout_seconds=10,interval_seconds=1,old_V1_full_unload='FAIL_UNCHANGED'))
 geometry=[dict(key=r['key'],rotation_tensor_bytes=r['rotation_tensor_bytes'],rotation_file_bytes=r['rotation_file_bytes'],additional_resident_allocated_bytes=r['additional_resident_allocated_bytes'],cache=r['initial_semantic']['geometry']) for r in observed]
 geometrypass=bool(geometry) and all(row['rotation_tensor_bytes']==24*128*128*4 and all(v['shape']==[1,32,128,128] for v in row['cache'].values()) for row in geometry)
 A.save('analysis/rotation_cache_geometry_audit.json',dict(status='PASS' if geometrypass else 'FAIL',rows=geometry,scope='fixed recurrent state dynamic shapes plus unchanged canonical cache architecture; no 81920 memory claim',frozen_bank_is_not_cache=True))
 # Hash all protected target scientific artifacts, without interpreting outputs.
 inv=A.read(A.ROOT/'provenance/source_artifact_revalidation.json');changed=[]
 for i,r in enumerate(inv['files']):
  if A.sha(r['path'])!=r['sha256']:changed.append(r['path'])
  if i%400==0:print('FINAL_PROTECTED_IDENTITY',i+1,flush=True)
 A.save('provenance/protected_artifact_final_revalidation.json',dict(status='PASS' if not changed else 'FAIL',files=len(inv['files']),changed=changed,old_V1_read_only=True))
 allpass=allpass and geometrypass and not changed
 A.save('analysis/future_canonical_activation.json',dict(status='FROZEN' if allpass else 'NOT_FROZEN',assignment_SHA=A.sha(A.ROOT/'configs/v2_future_canonical_assignment.json'),protocol_SHA=A.sha(A.ROOT/'configs/v2_future_canonical_protocol.json'),prospective_plans_unchanged=True,execution_authorized=False,NEW_AIME_GENERATION=0))
 gates=dict(EXPERIMENT=A.TASK,PARENT_V1='QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1',V1_IN_PROCESS_FULL_UNLOAD_GATE='FAIL',V1_FINAL_ALLOCATED_BYTES=8519680,V1_ROOT_CAUSE='UNRESOLVED',V1_ROOT_CAUSE_DETAIL='UNRESOLVED_AFTER_OBSERVER_HYPOTHESIS_REJECTED',V2_PROTOCOL_FREEZE_GATE='PASS',PRODUCTION_RUNNER_SCIENTIFIC_PATH_IDENTITY_GATE='PASS',SOURCE_ARTIFACT_REVALIDATION_GATE='PASS' if not changed else 'FAIL',C6R_ROTATION_SHA=A.SHAS['C6R'],C7_ROTATION_SHA=A.SHAS['C7'])
 for gpu in range(4):
  for arm in ('C6R','C7'):
   key=f'gpu{gpu}_{arm.lower()}';gates[f'GPU{gpu}_{arm}_RESIDENT_MEMORY_GATE']=mem.get(key,{}).get('status','NOT_RUN');gates[f'GPU{gpu}_{arm}_WORKER_READY']=primary[key]['ready']
 gates.update(C6R_FRESH_PROCESS_DETERMINISM_GATE=determinism['C6R'],C7_FRESH_PROCESS_DETERMINISM_GATE=determinism['C7'],PROCESS_EXIT_RECLAIM_GATE=exitstatus,FOUR_GPU_PERSISTENT_WORKER_GATE='PASS' if allpass else 'FAIL',HISTORICAL_81920_RUNTIME_COMPATIBILITY_GATE=A.read(A.ROOT/'provenance/historical_81920_runtime_compatibility.json')['status'],ROTATION_DOES_NOT_CHANGE_CACHE_GEOMETRY_GATE='PASS' if geometrypass else 'FAIL',CANONICAL60_WORKER_ASSIGNMENT='FROZEN' if allpass else 'NOT_FROZEN',CANONICAL60_CONDITION_ORDER='PREEXISTING_FROZEN_ORDER_C6R_THEN_C7',V2_PERSISTENT_WORKER_READINESS='PASS' if allpass else 'FAIL',C6R_3090_DEPLOYMENT_READY='YES' if allpass else 'NO',C7_3090_DEPLOYMENT_READY='YES' if allpass else 'NO',READY_FOR_C6R_C7_CANONICAL60='YES' if allpass else 'NO',CROSS_GPU_NUMERICAL_EQUALITY_GATE='NOT_REQUIRED',A800_3090_NUMERICAL_PARITY_POLICY='NO_PREEXISTING_CROSS_HARDWARE_TOLERANCE',NEW_TRAINING_STARTED='NO',TRAINING_RERUN='NO',C6R_AIME_STARTED='NO',C7_AIME_STARTED='NO',NEW_AIME_GENERATION=0,SCORING_STARTED='NO',PUSH_PERFORMED='NO')
 A.save('analysis/four_gpu_worker_readiness.json',dict(gates=gates,primary_workers=primary,all_workers=workers,observed_results=len(observed),controller_failure=A.read(A.ROOT/'analysis/v2_controller.error.json') if (A.ROOT/'analysis/v2_controller.error.json').exists() else None))
 lines=['# '+A.TASK,'','```text']+[str(k)+' = '+str(v) for k,v in gates.items()]+['```','','V2 DOES NOT OVERRIDE V1. V1 and V2 test different lifecycle requirements. V1 residual owner remains unresolved; no claim of a cuBLAS or other root cause.','','## Measured resident endpoints','','| Worker | Cycle | Allocated bytes | Reserved bytes | Peak allocated | Peak reserved |','|---|---:|---:|---:|---:|---:|']
 for r in observed:
  for i,c in enumerate(r['cycles'],1):
   m=c['memory'];lines.append(f"| {r['key']} | {i} | {m['allocated_end']} | {m['reserved_end']} | {m['peak_allocated']} | {m['peak_reserved']} |")
 lines += ['','## Interpretation and scope','','Production requests directly call the frozen manual-HF generate_one function, with a prospective non-AIME TRAIN:first-row prompt and max_new_tokens=128. One model/one frozen condition per process; one warmup and three identical measured requests. Semantic smoke repeats in each cycle using the same frozen32 input. Public output_hidden_states capture is semantic-only, with no extra diagnostic hooks. Original required scientific rotation patch remains unchanged. No extra gc/empty_cache, no full model deletion test.','','Generated IDs and compact semantic fingerprints are exact within each worker; GPU0 A/B comparison requires exact equality. Cross-GPU hashes are descriptive only, no invented tolerance. The original source/weights/tokenizer/runtime and all protected V1 evidence were rehashed.','','128-token smoke tests request lifetime, NOT 81920-token memory feasibility. Historical long-generation capability is inherited only via exact source/runtime/cache compatibility; bank changes do not alter cache geometry. Rotation tensor payload is 1,572,864 bytes per condition; actual allocated increment is recorded separately.','','All workers were sequential fresh subprocesses to enforce first-failure stop without interrupting any currently running work. Every measured process has its exact NVML byte baseline, PID/descendant audit, exit code and <=10-second reclaim history in analysis/*_worker.json. Incomplete rows are NOT_RUN, not evidence of success. No semantic/model/answer retries.','','## Frozen protocol and reproducibility','','Premeasurement commit: '+A.read(A.ROOT/'provenance/v2_premeasurement_commit.json')['commit'], '', '```sh',f'cd {A.REPO}',f'{A.PY} {A.ROOT}/scripts/persistent_v2_controller.py','# Single-use controller refuses existing evidence; do not rerun.', '```','','Future Canonical60 plan is same problem/seed→same physical GPU, rank%4, earlier frozen C6R_THEN_C7 order; 60/arm, C5R=0. Readiness activation is recorded independently from immutable premeasurement plans. No AIME/scoring run was started.','','STOP: await explicit user authorization for QWEN_GDN_C6R_C7_CANONICAL60_V1.' if allpass else 'STOP: V2 hard failure or incomplete matrix. No rule changes, retries, V1 patches or AIME permitted.']
 if (A.ROOT/'analysis/v2_controller.error.json').exists():lines += ['', 'Controller failure: '+A.read(A.ROOT/'analysis/v2_controller.error.json')['error']]
 report=A.ROOT/'reports'/f'{A.TASK}.md';report.parent.mkdir(parents=True,exist_ok=True)
 with report.open('x') as f:f.write('\n'.join(lines)+'\n')
 paths=[p for p in A.ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp') and p.name not in ('v2_final_artifact_sha256.txt','offline.log','offline.exitcode','v2_closure_commit.json')]
 manifest=A.ROOT/'manifests/v2_final_artifact_sha256.txt'
 with manifest.open('x') as f:
  for p in sorted(paths):f.write(A.sha(p)+'  '+str(p.relative_to(A.ROOT))+'\n')
 for l in manifest.read_text().splitlines():d,p=l.split('  ',1);A.require(A.sha(A.ROOT/p)==d,'final hash drift')
 print('V2_CLOSURE',gates['V2_PERSISTENT_WORKER_READINESS'],'STOP_NO_AIME',flush=True)
if __name__=='__main__':main()
