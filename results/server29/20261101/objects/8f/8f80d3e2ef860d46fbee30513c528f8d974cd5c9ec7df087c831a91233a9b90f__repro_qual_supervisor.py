"""Sequential real qualification processes, numerical comparison only."""
from pathlib import Path
import hashlib,json,subprocess,time
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');Q=R/'provenance_capsule_v3/scientific_reproducibility_v3_virtual_namespace'
RUNS=Path('/data/zypan/experiment_runs/ling_observability_scientific_repro_v3');PY='/data/zypan/envs/ling-kda/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    receipts=[]
    for n in range(1,4):
        assert not (Q/'STOP.json').exists()
        name='fresh_process_%02d'%n
        assert not (RUNS/name).exists()
        with (RUNS/(name+'.log')).open('x') as log:
            child=subprocess.Popen([PY,'-B',str(Q/'repro_launch.py'),'qualify',name],cwd=R,stdout=log,stderr=subprocess.STDOUT)
            emit(RUNS/(name+'_LAUNCH.json'),dict(pid=child.pid,time_unix=time.time(),input_commit=read(RUNS/'INPUT_COMMIT.json')['commit'],
                 launcher_sha256=sha(Q/'repro_launch.py'),runner_sha256=sha(Q/'repro_runner.py'),formal=False))
            print('FRESH_PROCESS_LAUNCHED',name,child.pid,flush=True)
            rc=child.wait()
        assert rc==0,('QUALIFICATION_CHILD_FAIL',name,rc)
        receipt=read(RUNS/name/'QUALIFICATION_COMPLETE.json');assert receipt['gate']=='PASS'
        receipts.append(receipt)
        print('FRESH_PROCESS_COMPLETE',name,flush=True)
    values=[read(RUNS/('fresh_process_%02d'%n)/'NUMERIC_FINGERPRINT.json') for n in range(1,4)]
    assert values[0]==values[1]==values[2],'FRESH_PROCESS_NUMERICAL_NONDETERMINISM'
    policy=read(Q/'repro_policy.json');gates={}
    for name in policy['blocking_gates']:
        if name=='FRESH_PROCESS_NUMERIC_DETERMINISM_GATE':gates[name]='PASS'
        else:
            assert all(r[name]=='PASS' for r in receipts);gates[name]='PASS'
    emit(RUNS/'ALL_SCIENTIFIC_QUALIFICATION_GATES.json',dict(gate='PASS',gates=gates,independent_processes=3,
         numerical_fields_bitwise_identical=True,native_binary_identity_compared=False,generated_source_identity_compared=False,
         same_experiment_local_cache=True,cache=receipts[0]['triton_cache'],input_commit=read(RUNS/'INPUT_COMMIT.json')['commit'],
         evidence=[dict(path=str(RUNS/('fresh_process_%02d'%n)/'QUALIFICATION_COMPLETE.json'),sha256=sha(RUNS/('fresh_process_%02d'%n)/'QUALIFICATION_COMPLETE.json'),
                        numeric_fingerprint_sha256=sha(RUNS/('fresh_process_%02d'%n)/'NUMERIC_FINGERPRINT.json')) for n in range(1,4)],
         GPU6_FORMAL_AUTHORIZED='NO',FORMAL_MEASUREMENTS=0))
    print('ALL_NINE_SCIENTIFIC_QUALIFICATION_GATES_PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (RUNS/'QUALIFICATION_SUPERVISOR_STOP.json').exists():emit(RUNS/'QUALIFICATION_SUPERVISOR_STOP.json',dict(reason=repr(e),FINAL_ACTION='STOP_CHECK_ACTUAL_CHILD_RECEIPT',formal=False))
        raise
