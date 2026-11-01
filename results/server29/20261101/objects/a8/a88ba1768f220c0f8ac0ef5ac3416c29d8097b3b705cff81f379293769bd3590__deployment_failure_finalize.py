#!/usr/bin/env python3
"""CPU-only evidence closure after a failed deployment smoke. No test retry."""
import json, subprocess, time
import deployment_closure as A
ROOT=A.ROOT

def main():
    inv=A.read(ROOT/'provenance/a800_source_deployment_inventory.json')
    source=inv['A800_SOURCE_COMMIT']
    A.require(A.cmd(['git','rev-parse','HEAD'])==source,'target HEAD drift')
    error=A.read(ROOT/'analysis/gpu0_repeat1_smoke.error.json')
    A.require(error['status']=='FAIL' and error['error']=='unexpected memory retention','unexpected failure identity')
    A.require((ROOT/'logs/gpu0_repeat1.exitcode').read_text().strip()=='1','exitcode mismatch')
    A.require(not list(ROOT.glob('analysis/gpu*_repeat*_smoke.json')),'unexpected successful units')
    A.require(len(list(ROOT.glob('analysis/gpu*_repeat*_smoke.error.json')))==1,'unexpected additional failed units')
    sf=A.read(ROOT/'provenance/source_final_revalidation.json')
    A.require(sf['status']=='PASS' and sf['count']==51 and sf['inventory_sha256']==A.sha(ROOT/'provenance/a800_source_deployment_inventory.json'),'source final audit mismatch')
    rows=A.read(ROOT/'provenance/protected_target_inventory.json')['files']
    for i,r in enumerate(rows):
        A.require(A.identity(r['path'],r['sha256'])==r,'protected file size/mtime/SHA changed: '+r['path'])
        if i%300==0: print(json.dumps({'event':'final_readonly_revalidation','files':i+1}),flush=True)
    transfers=[A.identity(ROOT/d,r['sha256']) for d,r in inv['transfer_members'].items()]
    A.save('provenance/protected_artifact_final_revalidation.json',dict(status='PASS',target_files=len(rows),target_size_mtime_SHA_unchanged=True,transfer_members=len(transfers),source_files=sf['count'],source_final_audit_sha256=A.sha(ROOT/'provenance/source_final_revalidation.json'),historical_outputs_and_scores_unchanged=True,rotations_unchanged=True))
    runtime=A.read(ROOT/'provenance/deployment_runtime_identity.json')
    gpu=A.cmd(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'])
    processes=subprocess.run(['pgrep','-af','[d]eployment_closure_gpu.py|[d]eployment_smoke_controller.py'],capture_output=True,text=True)
    A.require(processes.returncode==1 and not processes.stdout.strip(),'deployment process remains active')
    A.save('analysis/failure_static_reference_audit.json',dict(status='HYPOTHESIS_NOT_GPU_VERIFIED',failure=error,reference_chain=['smoke local original = patched torch_recurrent_gated_delta_rule','QwenKeyHadamardPatch._wrap.wrapped captures self','self.model retains loaded model; C7 patch also retains bank','local original and observed survive del patch/model until smoke returns'],observer_source=A.identity(ROOT/'scripts/deployment_closure_gpu.py'),formal_source=A.identity(A.FORMAL),evidence_limit='No in-process memory byte counts or per-condition fingerprints were persisted before the final cleanup assertion. Reached assertion implies preceding checks did not raise, but is not a saved successful smoke unit.',model_runtime_fault_proven=False,observer_reference_cause_verified_by_new_GPU_run=False,patch_applied_after_failure=False,retry=False))
    A.save('analysis/deployment_memory_audit.json',dict(status='FAIL',executed_units=['GPU0 repeat1'],failure_point='post_model_cleanup',criterion='allocated < 1048576 bytes',criterion_type='engineering observer cleanup check, not a cross-hardware numerical tolerance',in_process_memory_points=None,in_process_memory_points_missing_reason='smoke saves unit results only after the failing assertion',failure_record='analysis/gpu0_repeat1_smoke.error.json',post_process_nvidia_smi=gpu,post_process_snapshot_not_substitute_for_in_process_cleanup=True,no_active_deployment_process=True,no_81920_token_memory_proof=True,other_GPU_runs='NOT_RUN',retry=False))
    for arm in A.ARMS:
        m=A.read(ROOT/'configs'/f'{arm.lower()}_rotation_manifest.json')
        A.save(f'analysis/{arm.lower()}_rotation_load_audit.json',dict(status='NOT_VERIFIED',source_export_and_transfer='PASS',rotation_sha256=m['file_sha256'],attempted='GPU0 repeat1',evidence_limit='Per-condition load, roundtrip and orthogonality results were not serialized because the smoke failed at final cleanup.',no_numeric_modification=True,no_reexport=True))
        A.save(f'analysis/{arm.lower()}_deployment_smoke.json',dict(status='FAIL',unit='GPU0 repeat1',error='unexpected memory retention',condition_result_not_persisted=True,teacher_forced=True,no_free_generation=True,deployment_smoke_only=True,fresh_repeat2='NOT_RUN',other_GPUs='NOT_RUN',retry=False))
        A.save(f'configs/frozen_{arm.lower()}_deployment_config.json',dict(status='NOT_FROZEN',arm=arm,reason='Deployment hard gate failed; no future endpoint config may be frozen.',rotation_sha256=m['file_sha256'],generation_authorized=False))
    A.save('analysis/qwen_3090_recurrent_writeback_audit.json',dict(status='NOT_VERIFIED',static_formal_path_unmodified=True,formal_source=A.identity(A.FORMAL),dynamic_unit_aborted=True,dynamic_results_not_persisted=True,no_claim_of_success=True))
    A.save('analysis/four_gpu_consistency_audit.json',dict(status='NOT_RUN',partial_attempt='GPU0 repeat1 failed',GPUs_1_2_3='NOT_RUN',fresh_process_determinism='NOT_RUN',reason='Fail-closed controller stopped before next batch'))
    for name in ('canonical60_worker_assignment.json','future_canonical60_protocol.json'):
        A.save('configs/'+name,dict(status='NOT_FROZEN',reason='All deployment gates must pass before prospective assignment/config freeze',jobs=[],new_generation=0,scoring_started=False))
    gates={
        'EXPERIMENT':A.TASK,'PARENT_EXPERIMENT':'QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1',
        'C5R_REPRODUCTION':'INCONCLUSIVE','C5R_AIME':'SKIP',
        'C6R_SOURCE_EXPORT_GATE':'PASS','C7_SOURCE_EXPORT_GATE':'PASS',
        'C6R_TRANSFER_SHA_GATE':'PASS','C7_TRANSFER_SHA_GATE':'PASS',
        'QWEN_3090_MODEL_IDENTITY_GATE':'PASS',
        'C6R_ROTATION_LOAD_GATE':'FAIL','C7_ROTATION_LOAD_GATE':'FAIL',
        'C6R_ORTHOGONALITY_GATE':'FAIL','C7_ORTHOGONALITY_GATE':'FAIL',
        'ROTATION_MAPPING_GATE':'FAIL','QDQ_SEMANTICS_GATE':'FAIL','REAL_RECURRENT_WRITEBACK_GATE':'FAIL',
        'A800_3090_NUMERICAL_PARITY_POLICY':'NO_PREEXISTING_CROSS_HARDWARE_TOLERANCE',
        'C6R_DEPLOYMENT_SMOKE_GATE':'FAIL','C7_DEPLOYMENT_SMOKE_GATE':'FAIL',
        'C6R_3090_DETERMINISM_GATE':'NOT_RUN','C7_3090_DETERMINISM_GATE':'NOT_RUN',
        'FOUR_GPU_CONSISTENCY_GATE':'NOT_RUN','DEPLOYMENT_MEMORY_GATE':'FAIL',
        'PROTECTED_ARTIFACT_REVALIDATION_GATE':'PASS',
        'C6R_3090_DEPLOYMENT_READY':'NO','C7_3090_DEPLOYMENT_READY':'NO','READY_FOR_C6R_C7_CANONICAL60':'NO',
        'C6R_AIME_STARTED':'NO','C7_AIME_STARTED':'NO','NEW_AIME_GENERATION':0,'SCORING_STARTED':'NO','PUSH_PERFORMED':'NO',
        'TRAINING_RERUN':'NO','C5R_DEPLOYED':'NO','SMOKE_RETRY_STARTED':'NO','FUTURE_CONFIG_FREEZE':'NOT_PERFORMED','FINAL_STATUS':'STOPPED_FAIL_CLOSED'}
    A.save('analysis/deployment_gates.json',dict(gates=gates,gate_interpretation='FAIL for unclosed dynamic load/orthogonality/mapping/QDQ/writeback means no verified closure evidence, NOT a demonstrated numerical discrepancy. Source export orthogonality and CPU static semantics remain PASS; missing final results are never promoted to PASS.'))
    report=['# '+A.TASK,'','```text']+[f'{k} = {v}' for k,v in gates.items()]+['```','',
        '## Outcome','',
        'Stopped after GPU0 repeat1 failed at final model cleanup with `RuntimeError: unexpected memory retention`. No smoke retry, second repeat, GPU1–3 smoke, future config freeze, AIME generation or scoring was started.',
        '', 'Dynamic load/orthogonality/mapping/QDQ/writeback FAIL entries mean closure is unverified because the unit did not persist results. They do not establish numerical failure of the frozen rotations. The original CPU-only QDQ audit is static PASS only.',
        '', '## Failure evidence and limits','',
        'The exception is at the final allocated-memory cleanup assertion (<1 MiB). Earlier condition loops and base-weight rehash occur before it, but results and memory points are written only after it. Exact in-process allocated/reserved/peak bytes and output fingerprints are unavailable; no values are fabricated.',
        '', 'Static inspection finds a retained reference chain: observer local `original` -> formal `_wrap.wrapped` closure -> patch `self` -> `self.model` (and rotation bank). The local `observed` also retains `original`. This is a plausible observer-lifetime defect, not a GPU-verified root cause. No post-failure patch or rerun was applied. Frozen scientific code was not modified.',
        '', 'Files: `analysis/gpu0_repeat1_smoke.error.json`, `logs/gpu0_repeat1.log`, `logs/gpu0_repeat1.exitcode`, `logs/smoke_controller.error.json`, `analysis/failure_static_reference_audit.json`.',
        '', '## Passed identity and preservation gates','',
        f'A800 selected exports and exact transfer SHA PASS. Base model/config/tokenizer bytes match the A800 trusted freeze and historical canonical runtime. Source final revalidation: {sf["count"]} files PASS. Target protected final revalidation: {len(rows)} files PASS (size, mtime, SHA); transferred members also unchanged.',
        '', 'C6R rotation SHA256: `'+A.read(ROOT/'configs/c6r_rotation_manifest.json')['file_sha256']+'`.',
        'C7 rotation SHA256: `'+A.read(ROOT/'configs/c7_rotation_manifest.json')['file_sha256']+'`.',
        '', '## Hardware and provenance','',
        f'Hostname: `{runtime["hostname"]}`. Source A800 closure commit and target pre-task HEAD: `{source}`. Target branch: `'+A.cmd(['git','symbolic-ref','--short','HEAD'])+'`.',
        '', '```json',json.dumps(runtime['runtime'],indent=2),'```','', '```text',*runtime['GPUs'],'```','',
        'Architecture: manual HF, independent single-GPU workers; no TP/FSDP/ZeRO/SGLang. Cross-hardware policy: no preexisting A800–3090 numerical tolerance; none invented. Fixed historical non-AIME smoke: seed1, TRAIN:first-row 32-token prefix, Hadamard/C6R/C7, no free generation, engineering-only.',
        '', '## Launch incidents and code review','',
        'An initial tmux redirection failed because the logs directory did not exist, before Python/GPU work. This wrapper-I/O-only incident was recorded in logs/tmux_launch_incident_001.json. Directory creation was corrected before the sole GPU attempt; it was not a model/sample retry.',
        '', 'Original prepare source preserved. A prospective observer amendment before GPU execution removed an unused thread override and deleted a local parameter reference; its exact diff and source hashes are in analysis/pre_gpu_runtime_and_cleanup_source_amendment.json. The later wrapped-operator reference issue was not modified.',
        '', '## Stop and next authority','',
        'READY_FOR_C6R_C7_CANONICAL60 = NO. All four GPUs released their deployment process. Proposed next action, not executed: separately authorize a minimal observer-reference-lifetime repair with audit and identical fixed smoke; do not change model, quantizer, rotation, numerical tolerances or protocol. AIME requires a later explicit authorization even after readiness.',
        '', '## Commit and integrity','',
        'JSON parse and SHA manifest are verified before one local diagnostic deployment-closure commit. No push. The exact commit SHA and reviewed staged scope are saved after commit in logs/deployment_commit.json (excluded from the pre-commit manifest to avoid a self-referential commit). Mutable finalizer console log is excluded; all earlier failure logs are included.']
    p=ROOT/'reports'/f'{A.TASK}.md';p.parent.mkdir(exist_ok=True)
    with p.open('x') as f:f.write('\n'.join(report)+'\n')
    n=0
    for p in ROOT.rglob('*.json'):json.loads(p.read_text());n+=1
    A.save('analysis/failure_closure_schema_validation.json',dict(status='PASS',json_files_parsed=n,CSV='NOT_APPLICABLE',no_missing_metric_fabrication=True))
    manifest=ROOT/'manifests/deployment_artifact_sha256.txt';manifest.parent.mkdir(exist_ok=True)
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=manifest and '__pycache__' not in p.parts and p.name not in ('failure_finalize.log','deployment_commit.json')]
    with manifest.open('x') as f:
        for p in files:f.write(A.sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    for line in manifest.read_text().splitlines():d,p=line.split('  ',1);A.require(A.sha(ROOT/p)==d,'manifest mismatch')
    prefix=str(ROOT.relative_to(A.REPO))+'/'
    A.require(not A.cmd(['git','diff','--name-only']) and not A.cmd(['git','diff','--cached','--name-only']),'unexpected tracked changes')
    A.require(all(x.startswith('?? '+prefix) for x in A.cmd(['git','status','--porcelain']).splitlines()),'unexpected untracked changes')
    A.cmd(['git','add','-f','--',str(ROOT.relative_to(A.REPO))])
    # Exclude the changing console log, not the preserved worker/error logs.
    A.cmd(['git','reset','--',str((ROOT/'logs/failure_finalize.log').relative_to(A.REPO))])
    A.cmd(['git','diff','--cached','--check'])
    scope=A.cmd(['git','diff','--cached','--name-only']).splitlines()
    A.require(scope and all(p.startswith(prefix) for p in scope),'staged scope violation')
    A.require(not A.cmd(['git','diff','--cached','--diff-filter=MDR','--name-only']),'existing artifact staged modification')
    stat=A.cmd(['git','diff','--cached','--stat'])
    A.cmd(['git','-c','user.name=Qwen Scientific Automation','-c','user.email=qwen-scientific-automation@localhost','commit','-m','Preserve C6R C7 3090 deployment memory-gate failure; stop without AIME'])
    for line in manifest.read_text().splitlines():d,p=line.split('  ',1);A.require(A.sha(ROOT/p)==d,'postcommit manifest mismatch')
    A.save('logs/deployment_commit.json',dict(status='PASS_EVIDENCE_CLOSURE_ONLY',source_A800_commit=source,target_pre_task_HEAD=source,branch=A.cmd(['git','symbolic-ref','--short','HEAD']),deployment_closure_commit=A.cmd(['git','rev-parse','HEAD']),staged_diff_review=dict(additions_only=True,scope_only_task=True,check='PASS',scope=scope,stat=stat),json_parse='PASS',SHA_manifest='PASS',manifest_entries=len(files),manifest_sha256=A.sha(manifest),protected_artifacts='PASS',scientific_deployment_ready=False,push=False,stop=True))
    print('EVIDENCE_CLOSURE_COMPLETE_STOP '+A.cmd(['git','rev-parse','HEAD']),flush=True)

if __name__=='__main__':main()
