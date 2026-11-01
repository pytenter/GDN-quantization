"""Sequential GPU3 qualification only. No formal launch or GPU6 interruption."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
W=Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1')
A=W/'hardware_expansion_gpu3_v1'
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def main():
    cache=Path('/data/zypan/experiment_caches/ling_observability_gpu3_expansion_v1')
    assert not cache.exists();cache.mkdir()
    for name in ['triton','torchinductor','tmp']:(cache/name).mkdir()
    identity=read(Q/'HISTORICAL_RUNNER_LAUNCH_IDENTITY.json');profile=identity['environment'].copy()
    profile.update(CUDA_VISIBLE_DEVICES='3',TRITON_CACHE_DIR=str(cache/'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(cache/'torchinductor'),TMPDIR=str(cache/'tmp'))
    env=os.environ.copy()
    for k,v in profile.items():
        if v is None:env.pop(k,None)
        else:env[k]=v
    emit(A/'GPU3_LAUNCH_IDENTITY.json',dict(environment=profile,historical_environment_sha256=sha(Q/'HISTORICAL_RUNNER_LAUNCH_IDENTITY.json'),
        exclusive_initially_empty_cache=str(cache),GPU6_cache_not_used=True))
    for run_id,hooks in [('reference_off','off'),('fresh_process_01','on'),('fresh_process_02','on'),('fresh_process_03','on')]:
        assert not (A/'QUALIFICATION_STOP.json').exists()
        log=A/(run_id+'.log')
        with log.open('xb') as f:
            proc=subprocess.run([identity['historical_python'],'-B',str(A/'gpu3_expansion_qualification.py'),run_id,'--hooks',hooks],
                cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
        assert proc.returncode==0,('QUALIFICATION_PROCESS_FAIL',run_id,proc.returncode)
        print('GPU3_STAGE_COMPLETE',run_id,flush=True)
    paths=[A/'qualification'/r/'NUMERIC_FINGERPRINT.json' for r in ['fresh_process_01','fresh_process_02','fresh_process_03']]
    fingerprints=[sha(p) for p in paths]
    assert len(set(fingerprints))==1,'FRESH_PROCESS_NUMERIC_DETERMINISM_FAIL'
    emit(A/'FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json',dict(gate='PASS',repeat_count=3,comparison='exact bytes; no tolerance change',
        files=[dict(path=str(p),sha256=sha(p)) for p in paths],GPU3_FORMAL_AUTHORIZED='NO',formal_measurements=0))
    print('GPU3_QUALIFICATION_COMPLETE_FORMAL_LAUNCH_PENDING_TRACE_FREEZE_AND_ASSIGNMENT_HANDOFF',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (A/'QUALIFICATION_STOP.json').exists():emit(A/'QUALIFICATION_STOP.json',dict(error=repr(e),GPU3_FORMAL_AUTHORIZED='NO',FINAL_ACTION='FAIL_AND_STOP'))
        raise
