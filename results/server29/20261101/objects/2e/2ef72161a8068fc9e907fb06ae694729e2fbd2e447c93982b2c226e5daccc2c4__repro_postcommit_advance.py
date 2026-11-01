"""Minimal post-commit scientific checks, then original Discovery and STOP."""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,time
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');C=R/'provenance_capsule_v3';Q=C/'scientific_reproducibility_v2_loader_alias'
RUNS=Path('/data/zypan/experiment_runs/ling_observability_scientific_repro_v2');PY='/data/zypan/envs/ling-kda/bin/python'
D=R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def emit(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*a):return subprocess.check_output(['git',*a],cwd=R,text=True).strip()
def child(mode,name):
    assert not (Q/'STOP.json').exists()
    with (RUNS/(name+'.log')).open('x') as log:
        p=subprocess.Popen([PY,'-B',str(Q/'repro_launch.py'),mode,name],cwd=R,stdout=log,stderr=subprocess.STDOUT)
        emit(RUNS/(name+'_LAUNCH.json'),dict(mode=mode,pid=p.pid,time_unix=time.time(),premeasurement_commit=read(Q/'PREMEASUREMENT_COMMIT.json')['commit'],
             launcher_sha256=sha(Q/'repro_launch.py'),runner_sha256=sha(Q/'repro_runner.py')))
        print('POSTCOMMIT_STAGE_LAUNCH',mode,name,p.pid,flush=True)
        rc=p.wait()
    assert rc==0,('SCIENTIFIC_CHILD_FAIL',mode,name,rc)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('commit');a=parser.parse_args()
    assert git('rev-parse','HEAD')==a.commit and not git('status','--porcelain=v1')
    report=read(Q/'SCIENTIFIC_QUALIFICATION_REPORT.json')
    assert report['gate']=='PASS' and all(v=='PASS' for v in report['gates'].values())
    emit(Q/'PREMEASUREMENT_COMMIT.json',dict(commit=a.commit,source_tree=git('rev-parse','HEAD^{tree}'),
         original_scientific_preregistration_sha256=sha(D/'preregistration.json'),qualification_report_sha256=sha(Q/'SCIENTIFIC_QUALIFICATION_REPORT.json'),
         formal_authorized=False,postcommit_checks='PENDING'))
    child('postcommit','postcommit_01')
    post=read(RUNS/'postcommit_01/QUALIFICATION_COMPLETE.json');assert post['gate']=='PASS'
    assert read(RUNS/'postcommit_01/NUMERIC_FINGERPRINT.json')==read(RUNS/'fresh_process_01/NUMERIC_FINGERPRINT.json'),'POSTCOMMIT_NUMERICAL_PARITY_FAILURE'
    for row in read(Q/'HISTORICAL_FAILURES_AND_SOURCE_IDENTITY.json')['files']:assert sha(C/row['path'])==row['sha256']
    assert git('rev-parse','HEAD')==a.commit and not git('diff','--name-only')
    emit(Q/'POSTCOMMIT_SCIENTIFIC_IDENTITY_PARITY.json',dict(gate='PASS',commit=a.commit,
         postcommit_receipt_sha256=sha(RUNS/'postcommit_01/QUALIFICATION_COMPLETE.json'),
         numeric_fingerprint_sha256=sha(RUNS/'postcommit_01/NUMERIC_FINGERPRINT.json'),
         all_nine_gates=report['gates'],same_experiment_local_cache=True,historical_failures_byte_identical=True))
    emit(Q/'GPU6_FORMAL_AUTHORIZATION.json',dict(GPU6_FORMAL_AUTHORIZED='YES',physical_gpu=6,
         gpu_uuid='GPU-de53f243-b271-521c-d875-d96611ed8574',premeasurement_commit=a.commit,
         scientific_qualification_sha256=sha(Q/'SCIENTIFIC_QUALIFICATION_REPORT.json'),
         postcommit_identity_parity_sha256=sha(Q/'POSTCOMMIT_SCIENTIFIC_IDENTITY_PARITY.json'),
         FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',formal_scope='Discovery primary64+secondary64 only',time_unix=time.time()))
    print('GPU6_FORMAL_AUTHORIZED_YES_IMMEDIATELY_START_DISCOVERY',a.commit,flush=True)
    child('discovery','discovery_formal_01')
    raw=read(D/'discovery/raw_freeze.json');assert raw['complete']==128
    for row in raw['files']:assert sha(row['path'])==row['sha256']
    child('analyze','final_analysis_01')
    final=read(D/'FINAL_REPORT.json');assert final['FINAL_ACTION']=='STOP_AFTER_DISCOVERY'
    # Required descriptive reporting supplements; all numbers come from frozen raw artifacts.
    endpoints=[]
    for p in sorted((D/'discovery/summaries').glob('*__endpoint32K.json')):
        prefix=p.name.removesuffix('__endpoint32K.json');doc,condition=prefix.split('__',1)
        endpoints.append(dict(document=doc,condition=condition,**read(p)))
    assert len(endpoints)==64
    with (D/'analysis/endpoint_32K_snapshot.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(endpoints[0]));w.writeheader();w.writerows(endpoints)
    emit(D/'analysis/por_vs_prop_report.json',dict(primary=final['primary'],secondary=final['secondary'],
         interpretation='Exploratory within-document rotation ranking only; no pooled primary p-value.'))
    emit(D/'analysis/downstream_target_report.json',dict(target='TARGET_FUTURE_KL_AUC',direction='ref||pulse',
         reduction='full-vocabulary FP32 KL per token; streaming FP64 sum',teacher_forcing=True,
         anchors=[8192,2048],endpoint=32768,primary_complete='64/64',secondary_complete='64/64',raw_freeze_sha256=sha(D/'discovery/raw_freeze.json')))
    for row in read(Q/'HISTORICAL_FAILURES_AND_SOURCE_IDENTITY.json')['files']:assert sha(C/row['path'])==row['sha256']
    emit(D/'provenance/SCIENTIFIC_REPRO_COMPLETION.json',dict(EXPERIMENT=final['EXPERIMENT'],
         FINAL_ACTION='STOP_AFTER_DISCOVERY',DISCOVERY_PRIMARY_COMPLETE='64/64',DISCOVERY_SECONDARY_COMPLETE='64/64',
         L9_TRAINING_EXECUTED='NO',NEW_ROTATION_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO',FREE_GENERATION_EXECUTED='NO',
         FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',historical_failures_preserved=True,
         cleanup_ledger_bookkeeping_remaining=1,actual_task_processes='discovery and analysis children exited',
         actual_gpu_state='must be independently inspected by monitor',premeasurement_commit=a.commit))
    emit(D/'FINAL_SHA_INVENTORY.json',dict(files=[dict(path=str(p.relative_to(D)),sha256=sha(p),bytes=p.stat().st_size)
         for p in sorted(D.rglob('*')) if p.is_file() and p.name!='FINAL_SHA_INVENTORY.json']))
    print('FINAL_ACTION_STOP_AFTER_DISCOVERY',final['OBSERVABILITY_DISCOVERY_LABEL'],flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (RUNS/'POSTCOMMIT_ADVANCEMENT_STOP.json').exists():emit(RUNS/'POSTCOMMIT_ADVANCEMENT_STOP.json',dict(reason=repr(e),FINAL_ACTION='FAIL_AND_STOP'))
        raise
