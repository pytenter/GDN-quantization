"""Final evidence manifest and local Commit B; no push/model execution."""
import argparse, os, subprocess, time
import canonical60_common as A

def wait_own_workers():
    import psutil
    while True:
        active=[]
        for p in (A.ROOT/'logs').glob('*_launch.json'):
            pid=A.read(p)['pid']
            try:
                proc=psutil.Process(pid);line=' '.join(proc.cmdline())
                if 'canonical60_worker.py' in line and str(A.ROOT) in line and proc.status()!=psutil.STATUS_ZOMBIE:active.append(pid)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        if not active:return
        # Current requests finish naturally; never signal a process or GPU.
        time.sleep(2)

def failure_report():
    counts={a:len(list((A.ROOT/'raw'/a.lower()).glob('*.json'))) for a in A.SHAS}
    fields=dict(EXPERIMENT=A.TASK,PARENT_READINESS='QWEN_GDN_C6R_C7_4X3090_PERSISTENT_WORKER_READINESS_V2',CANONICAL60_PREMEASUREMENT_FREEZE_GATE='PASS',CANONICAL60_RUNTIME_IDENTITY_GATE='PASS',C6R_ROTATION_SHA=A.SHAS['C6R'],C7_ROTATION_SHA=A.SHAS['C7'],CONDITION_ORDER='C6R_THEN_C7',WORKER_ASSIGNMENT='RANK_MOD_4_SAME_GPU_ACROSS_CONDITIONS',C6R_RAW_COMPLETE=f"{counts['C6R']} / 60",C7_RAW_COMPLETE=f"{counts['C7']} / 60",TOTAL_NEW_AIME_GENERATIONS=f"{sum(counts.values())} completed / 120",C6R_RAW_OUTPUT_FREEZE_GATE='PASS' if (A.ROOT/'manifests/c6r_raw_output_manifest.json').exists() else 'NOT_RUN',C7_RAW_OUTPUT_FREEZE_GATE='PASS' if (A.ROOT/'manifests/c7_raw_output_manifest.json').exists() else 'NOT_RUN',CANONICAL60_RAW_OUTPUT_CLOSURE_GATE='PASS' if (A.ROOT/'analysis/canonical60_raw_output_closure.json').exists() else 'NOT_RUN',FROZEN_V4_IDENTITY_GATE='PASS' if (A.ROOT/'provenance/frozen_v4_identity.json').exists() else 'NOT_RUN',SCORING_COMPLETE='NO',C6R_CORRECT='NOT_REPORTED_INCOMPLETE_CLOSURE',C7_CORRECT='NOT_REPORTED_INCOMPLETE_CLOSURE',C7_RESCUE_C6R='NOT_RUN',C7_REGRESSION_C6R='NOT_RUN',BOTH_CORRECT='NOT_RUN',BOTH_WRONG='NOT_RUN',C7_MINUS_C6R_ACCURACY='NOT_RUN',PRIMARY_EXACT_P='NOT_RUN',PRIMARY_95CI_CLUSTER_BOOTSTRAP='NOT_RUN',PRIMARY_ENDPOINT_SIGNAL='NOT_RUN',PROTECTED_ARTIFACT_REVALIDATION_GATE='NOT_RUN',NEW_TRAINING_STARTED='NO',TRAINING_RERUN='NO',C5R_AIME_GENERATIONS=0,PUSH_PERFORMED='NO')
    try:
        protected=A.protected_revalidate();protected.update(PROTECTED_ARTIFACT_REVALIDATION_GATE='PASS',new_raw_outputs='PARTIAL_NOT_SCORED')
        if not (A.ROOT/'provenance/protected_artifact_revalidation.json').exists():A.save('provenance/protected_artifact_revalidation.json',protected,True)
        fields['PROTECTED_ARTIFACT_REVALIDATION_GATE']='PASS'
    except Exception as e:fields['PROTECTED_ARTIFACT_REVALIDATION_GATE']='FAIL';fields['REVALIDATION_ERROR']=str(e)
    errors=[]
    for p in sorted((A.ROOT/'logs').glob('*.error.json')):errors.append(dict(path=str(p),sha256=A.sha(p),error=A.read(p).get('error')))
    A.save('analysis/fail_closed_status.json',dict(status='FAIL_CLOSED',counts=counts,started_samples=len(list((A.ROOT/'logs/started').rglob('*.json'))),errors=errors,automatic_retry=False,new_training=False,STOP=True),True)
    report=A.ROOT/'reports'/f'{A.TASK}.md'
    if not report.exists():
        lines=['# '+A.TASK,'','```text']+[f'{k} = {v}' for k,v in fields.items()]+['```','','FAIL_CLOSED. Evidence retained; no automatic retry, GPU reassignment, length/config changes or follow-on experiment. Any already-started peer request was allowed to finish naturally; next-sample launches stopped. Inspect logs/*.error.json and analysis/fail_closed_status.json. Await explicit user authorization. No incomplete-score correctness is reported.']
        A.write_text(report,'\n'.join(lines)+'\n',True)

def finish(success):
    wait_own_workers()
    if success:A.require(A.read(A.ROOT/'analysis/endpoint_completion.json')['status']=='PASS','endpoint not complete')
    else:failure_report()
    excluded={'final_artifact_sha256.txt','offline.log','offline.exitcode','closure_commit.json'}
    paths=sorted(p for p in A.ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded and '.tmp.' not in p.name)
    A.write_text('manifests/final_artifact_sha256.txt',''.join(A.sha(p)+'  '+str(p.relative_to(A.ROOT))+'\n' for p in paths),True)
    for line in (A.ROOT/'manifests/final_artifact_sha256.txt').read_text().splitlines():
        h,p=line.split('  ',1);A.require(A.sha(A.ROOT/p)==h,'final artifact changed '+p)
    # Commit scientific summaries, metadata and SHA manifests, not raw blobs/logs.
    rel=str(A.ROOT.relative_to(A.REPO))
    for folder in ('configs','manifests','scores','analysis','reports','provenance','scripts'):
        subprocess.run(['git','add','-f',rel+'/'+folder],cwd=A.REPO,check=True)
    subprocess.run(['git','-c','user.name=Qwen Scientific Automation','-c','user.email=qwen-scientific-automation@localhost','commit','-m',('Freeze C6R/C7 Canonical60 endpoint closure' if success else 'Preserve Canonical60 fail-closed evidence')],cwd=A.REPO,check=True)
    A.require(not A.git('status','--porcelain'),'post-closure checkout not clean')
    A.save('logs/closure_commit.json',dict(commit=A.git('rev-parse','HEAD'),parent=A.git('rev-parse','HEAD^'),clean=True,status='COMPLETE_STOP' if success else 'FAIL_CLOSED_STOP',artifact_manifest_sha256=A.sha(A.ROOT/'manifests/final_artifact_sha256.txt'),PUSH_PERFORMED='NO',utc=A.utc()),True)
    print('LOCAL_COMMIT_B_STOP',A.git('rev-parse','HEAD'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--success',action='store_true');p.add_argument('--failure',action='store_true');a=p.parse_args()
    A.require(a.success!=a.failure,'one finalization mode required');finish(a.success)
