"""Sequential fresh-process qualification; no Discovery or automatic gate synthesis."""
from pathlib import Path
import hashlib,json,os,subprocess,tarfile,time
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
RUNS=Path('/data/zypan/experiment_runs/ling_historical_runtime_rebase_v1')
AUTH=Path(str(R)+'-authority');PY='/data/zypan/envs/ling-kda/bin/python'
def read(p):return json.loads(Path(p).read_text())
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert read(Q/'HISTORICAL_RUNNER_SELF_REPLAY_GATE.json')['gate']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=R,text=True).strip()
    RUNS.mkdir();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
    env=os.environ.copy();profile=read(Q/'HISTORICAL_RUNNER_LAUNCH_IDENTITY.json')['environment']
    for k,v in profile.items():
        if v is None:env.pop(k,None)
        else:env[k]=str(v)
    emit(RUNS/'INPUT_COMMIT.json',dict(commit=commit,scientific_premeasurement=False,FORMAL_MEASUREMENTS=0))
    inputs=Q/'phase_b_inputs'
    for i in range(1,4):
        run_id=f'fresh_process_{i:02d}'
        assert not subprocess.check_output(['nvidia-smi','-i','6','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        args=[PY,'-B',str(inputs/'historical_rebase_qualification.py'),run_id,'--hooks','on']
        with (RUNS/(run_id+'.log')).open('xb') as f:
            p=subprocess.Popen(args,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
            emit(RUNS/(run_id+'_LAUNCH.json'),dict(pid=p.pid,args=args,time_unix=time.time(),input_commit=commit))
            code=p.wait()
        assert code==0 and read(RUNS/run_id/'QUALIFICATION_COMPLETE.json')['gate']=='PASS',('QUALIFICATION_FAILED',run_id,code)
        print('FRESH_PROCESS_COMPLETE',i,flush=True)
    values=[read(RUNS/f'fresh_process_{i:02d}/NUMERIC_FINGERPRINT.json') for i in range(1,4)]
    assert values[0]==values[1]==values[2],'FRESH_PROCESS_NUMERICAL_NONDETERMINISM'
    emit(RUNS/'FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json',dict(gate='PASS',fresh_processes=3,
         cache=profile['TRITON_CACHE_DIR'],fingerprints=[dict(path=str(RUNS/f'fresh_process_{i:02d}/NUMERIC_FINGERPRINT.json'),sha256=sha(RUNS/f'fresh_process_{i:02d}/NUMERIC_FINGERPRINT.json')) for i in range(1,4)],
         exact_numerical_equality=True,native_binary_byte_identity_required=False,FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
    preserve=Q/'phase_b_qualification_evidence';preserve.mkdir()
    for p in sorted(RUNS.rglob('*')):
        if p.is_file():
            d=preserve/p.relative_to(RUNS);d.parent.mkdir(parents=True,exist_ok=True)
            with d.open('xb') as f:f.write(p.read_bytes())
            assert sha(d)==sha(p)
    with tarfile.open(AUTH/'scientific-repro-historical-phase-b.tar.gz','w:gz') as t:
        for p in sorted(Q.rglob('*')):
            if p.is_file():t.add(p,arcname=str(p.relative_to(R)))
    print('PHASE_B_QUALIFICATION_PASS_READY_FOR_PREMEASUREMENT',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        if not (RUNS/'SUPERVISOR_STOP.json').exists():emit(RUNS/'SUPERVISOR_STOP.json',dict(error=repr(e),traceback=traceback.format_exc(),FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
        raise
