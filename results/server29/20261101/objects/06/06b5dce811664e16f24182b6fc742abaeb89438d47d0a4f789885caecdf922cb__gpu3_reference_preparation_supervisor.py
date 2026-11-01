"""Wait for actual GPU3 three-process PASS, freeze new inputs, prepare FP only."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
W=Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1')
A=W/'hardware_expansion_gpu3_v1'
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
BASE='0e7cec9d7ed83888b93ad18accb08970a3852e1c'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*args,cwd=W):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()

def main():
    print('WAITING_FOR_ACTUAL_GPU3_THREE_FRESH_PROCESS_PASS',flush=True)
    while not (A/'FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json').exists():
        assert not (A/'QUALIFICATION_STOP.json').exists(),'PRESERVED_GPU3_QUALIFICATION_STOP'
        assert Path('/proc/2736759').exists(),'QUALIFICATION_SUPERVISOR_EXITED_WITHOUT_COMPLETION'
        time.sleep(10)
    gate=read(A/'FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json')
    assert gate['gate']=='PASS' and gate['repeat_count']==3
    assert len({sha(row['path']) for row in gate['files']})==1
    assert all(sha(row['path'])==row['sha256'] for row in gate['files'])
    while Path('/proc/2736759').exists():
        assert not (A/'QUALIFICATION_STOP.json').exists()
        time.sleep(1)
    assert git('rev-parse','HEAD')==BASE
    assert not git('status','--porcelain=v1','--untracked-files=no')
    assert git('rev-parse','HEAD',cwd=R)=='e428b5deb0579b926b2ad098a8699b3f4f2233f0'
    for run in ['fresh_process_01','fresh_process_02','fresh_process_03']:
        receipt=read(A/'qualification'/run/'QUALIFICATION_COMPLETE.json')
        assert receipt['gate']=='PASS' and receipt['hardware']['physical_gpu']==3
        for key in ['SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY_GATE','RUNTIME_ENVIRONMENT_IDENTITY_GATE',
            'BASIS_SEMANTICS_GATE','HISTORICAL_K3_PARITY_GATE','HISTORICAL_JPROP_PARITY_GATE',
            'FUTURE_KL_HARNESS_PARITY_GATE','FP_REFERENCE_PAIRING_GATE','HARDWARE_ASSIGNMENT_GATE','STORAGE_CAPACITY_GATE']:
            assert receipt[key]=='PASS',('MISSING_ACTUAL_QUALIFICATION_GATE',run,key)
    # Existing scripts/receipts unchanged. Append independent preparation inputs.
    names=['gpu3_same_gpu_reference_capture.py','gpu3_reference_preparation_supervisor.py']
    for name in names:
        source=Path('/data/zypan')/name;dest=A/'reference_preparation_inputs'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(source.read_bytes())
        assert sha(dest)==sha(source)
    files=[]
    for p in sorted((A/'qualification').rglob('*')):
        if p.is_file():files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    emit(A/'GPU3_QUALIFICATION_EVIDENCE_SHA_INVENTORY.json',dict(gate='PASS',files=files,
        actual_three_process_complete=True,FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',GPU3_FORMAL_AUTHORIZED='NO'))
    paths=[str(p.relative_to(W)) for p in sorted(A.rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.log']]
    subprocess.run(['git','add','--sparse','--',*paths],cwd=W,check=True)
    assert all(s.startswith('A\thardware_expansion_gpu3_v1/') for s in git('diff','--cached','--name-status').splitlines())
    tree=git('write-tree','--missing-ok')
    diff=git('diff-tree','--no-commit-id','--name-status','-r',BASE,tree).splitlines()
    assert diff and all(s.startswith('A\thardware_expansion_gpu3_v1/') for s in diff)
    env=os.environ.copy();env.update(GIT_AUTHOR_NAME='Codex',GIT_AUTHOR_EMAIL='codex@local',GIT_COMMITTER_NAME='Codex',GIT_COMMITTER_EMAIL='codex@local')
    commit=subprocess.check_output(['git','commit-tree',tree,'-p',BASE,'-m','ling: freeze actual GPU3 qualification and independent FP reference preparation inputs'],cwd=W,env=env,text=True).strip()
    subprocess.run(['git','update-ref','refs/heads/exp/ling-observability-gpu3-expansion-v1',commit,BASE],cwd=W,check=True)
    assert not git('status','--porcelain=v1','--untracked-files=no')
    inputs=[dict(path=str(A/'reference_preparation_inputs'/name),sha256=sha(A/'reference_preparation_inputs'/name)) for name in names]
    emit(A/'REFERENCE_PREPARATION_INPUT_COMMIT.json',dict(commit=commit,parent=BASE,inputs=inputs,
        gate='PASS',not_scientific_premeasurement_commit=True,GPU3_FORMAL_AUTHORIZED='NO',FORMAL_MEASUREMENTS=0))
    print('GPU3_QUALIFICATION_FREEZE_REFERENCE_PREPARATION_INPUT_COMMIT',commit,flush=True)
    launch=read(A/'GPU3_LAUNCH_IDENTITY.json')['environment'];env=os.environ.copy()
    for key,value in launch.items():
        if value is None:env.pop(key,None)
        else:env[key]=value
    with (A/'gpu3_reference_capture.log').open('xb') as f:
        proc=subprocess.run(['/data/zypan/envs/ling-kda/bin/python','-B',str(A/'reference_preparation_inputs/gpu3_same_gpu_reference_capture.py')],
            cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert proc.returncode==0,('GPU3_REFERENCE_CAPTURE_FAIL',proc.returncode)
    print('GPU3_FP_REFERENCE_PREPARATION_COMPLETE_FORMAL_AUTHORIZATION_STILL_PENDING',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        p=A/'REFERENCE_PREPARATION_SUPERVISOR_STOP.json'
        if not p.exists():emit(p,dict(error=repr(e),traceback=traceback.format_exc(),GPU3_FORMAL_AUTHORIZED='NO',FORMAL_MEASUREMENTS=0,FINAL_ACTION='FAIL_AND_STOP'))
        raise
