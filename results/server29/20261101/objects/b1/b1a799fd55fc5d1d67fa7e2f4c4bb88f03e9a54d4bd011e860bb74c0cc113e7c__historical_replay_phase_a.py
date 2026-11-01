"""Whole historical runner replay. No model hooks, guards, or source patches."""
from pathlib import Path
import hashlib, importlib.util, json, os, shutil, subprocess, tarfile, time

R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
H=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v1')
OLD=H/'checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
AUTH=Path(str(R)+'-authority')
BASE=Path('/data/zypan/experiment_runs/ling_historical_runtime_replay_v1')
CACHE=Path('/data/zypan/experiment_caches/ling_historical_runtime_replay_v1')
PY='/data/zypan/envs/ling-kda/bin/python'
COMMIT='66966044c0e25eb31fc15ffb1724f48de734a926'
REFERENCE_SHA='9d902526babf110261b86c56533986f5eaf70b394ea1078b4208560f4024829d'
SCOPE='experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def run(*a,cwd=None):return subprocess.check_output(a,cwd=cwd,text=True).strip()
def cp(p,d):
    d.parent.mkdir(parents=True,exist_ok=True)
    with d.open('xb') as f:f.write(p.read_bytes())
    assert sha(p)==sha(d)
def archive():
    with tarfile.open(AUTH/'scientific-repro-historical-phase-a.tar.gz','w:gz') as t:
        for p in sorted(Q.rglob('*')):
            if p.is_file():t.add(p,arcname=str(p.relative_to(R)))

def main():
    assert run('git','rev-parse','HEAD',cwd=R)=='1adb70d7f9df3f5434811dc65311440f6aba3322'
    assert not run('git','status','--porcelain=v1',cwd=R)
    assert sha(OLD/'smoke/repeat0/result.json')==REFERENCE_SHA
    assert not run('nvidia-smi','-i','6','--query-compute-apps=pid','--format=csv,noheader')
    assert 'GPU-de53f243-b271-521c-d875-d96611ed8574' in run('nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader')
    assert shutil.disk_usage(R).free>24*1024**3
    Q.mkdir();BASE.mkdir();CACHE.mkdir()
    for name in ['triton','torchinductor','tmp']: (CACHE/name).mkdir()
    emit(Q/'CACHE_CREATED_EMPTY.json',dict(path=str(CACHE),initial_files=[],old_attempt12_cache_used=False,
         operational_exception='Exclusive fresh cache placement only; historical numerical variables and fixed FLA configs unchanged. No source-to-binary closure.'))
    cp(Path(__file__),Q/'historical_replay_phase_a.py')
    cp(OLD/'scripts/controller.py',Q/'original_historical_controller.py')
    cp(OLD/'runtime_manifest.json',Q/'original_historical_runtime_manifest.json')
    cp(OLD/'smoke/repeat0/result.json',Q/'original_historical_reference.json')
    package_rows=read(OLD/'smoke/repeat0/result.json')['dependencies']['execution_module_files']
    package_checks=[]
    for row in package_rows:
        p=Path(row['path'])
        if str(p).startswith('/data/zypan/envs/ling-kda/'):
            assert p.is_file() and sha(p)==row['sha256'],('HISTORICAL_PACKAGE_DRIFT',str(p))
            package_checks.append(row)
    emit(Q/'HISTORICAL_EXECUTED_PACKAGE_IDENTITY.json',dict(gate='PASS',files=package_checks))
    bare=BASE/'authority.git';checkout=BASE/'checkout'
    subprocess.run(['git','init','--bare',str(bare)],check=True)
    with (H/'premeasurement.pack').open('rb') as f:
        subprocess.run(['git','--git-dir='+str(bare),'unpack-objects'],stdin=f,check=True)
    (bare/'shallow').write_text(COMMIT+'\n')
    subprocess.run(['git','--git-dir='+str(bare),'update-ref','refs/heads/historical-reference-replay',COMMIT],check=True)
    subprocess.run(['git','--git-dir='+str(bare),'worktree','add','--no-checkout',str(checkout),'historical-reference-replay'],check=True)
    subprocess.run(['git','config','core.sparseCheckout','true'],cwd=checkout,check=True)
    sparse=Path(run('git','rev-parse','--git-path','info/sparse-checkout',cwd=checkout))
    if not sparse.is_absolute():sparse=checkout/sparse
    sparse.parent.mkdir(parents=True,exist_ok=True);sparse.write_text('/'+SCOPE+'/\n')
    subprocess.run(['git','read-tree','-mu','HEAD'],cwd=checkout,check=True)
    d=checkout/SCOPE
    if not (d/'premeasurement_commit.json').exists():cp(OLD/'premeasurement_commit.json',d/'premeasurement_commit.json')
    assert not (d/'STOP.json').exists() and not (d/'smoke/repeat0').exists()
    tracked=run('git','ls-files','--',SCOPE,cwd=checkout).splitlines()
    for p in tracked:assert sha(checkout/p)==sha(H/'checkout'/p),('HISTORICAL_SOURCE_DIFF',p)
    assert not run('git','status','--porcelain=v1','--untracked-files=no',cwd=checkout)
    # Reuse the original controller's environment derivation; numerical execution
    # is the original byte-identical bridge CLI, not a current wrapper/importer.
    spec=importlib.util.spec_from_file_location('historical_replay_controller',d/'scripts/controller.py')
    ctrl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctrl)
    bridge=ctrl.module();_,transport,*_=bridge.modules()
    profile=read(d/'runtime_manifest.json')['expected']['launch_profile']
    env=os.environ.copy()
    for k,v in profile.items():
        if v is None:env.pop(k,None)
        else:env[k]=str(transport.mapped(v)) if isinstance(v,str) and v.startswith('/data01/user2/') else str(v)
    env.update(CUDA_VISIBLE_DEVICES='6',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONNOUSERSITE='1',
         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.update(TRITON_CACHE_DIR=str(CACHE/'triton'),TORCHINDUCTOR_CACHE_DIR=str(CACHE/'torchinductor'),TMPDIR=str(CACHE/'tmp'))
    for k in ['TRITON_DUMP_DIR','TRITON_OVERRIDE_DIR']:env.pop(k,None)
    emit(Q/'HISTORICAL_RUNNER_LAUNCH_IDENTITY.json',dict(task='LING_HISTORICAL_DETERMINISTIC_RUNTIME_REPLAY_AND_REBASE_V1',
         historical_source_commit=COMMIT,original_launcher=str(OLD/'scripts/controller.py'),historical_python=PY,
         reference=dict(path=str(OLD/'smoke/repeat0/result.json'),sha256=REFERENCE_SHA),
         execution_checkout=str(checkout),byte_identical_historical_source=True,
         original_command=[PY,str(OLD/'scripts/bridge.py'),'smoke','--repeat','0'],
         replay_command=[PY,str(d/'scripts/bridge.py'),'smoke','--repeat','0'],
         environment={k:env.get(k) for k in list(profile)+['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','TORCHINDUCTOR_CACHE_DIR','TMPDIR']},
         new_model_instrumentation=False,current_V3_guard=False,native_binary_guard=False,source_edits=False,kernel_patches=False,
         operational_changes=['new isolated output checkout','fresh experiment-only cache to avoid Attempt12 cache'],formal_measurements=0))
    for args in [['verify'],['smoke','--repeat','0']]:
        label='__'.join(args);dest=Q/(label+'.log')
        with dest.open('xb') as f:
            p=subprocess.Popen([PY,str(d/'scripts/bridge.py'),*args],cwd=checkout,env=env,stdout=f,stderr=subprocess.STDOUT)
            emit(Q/(label+'_PROCESS.json'),dict(pid=p.pid,args=args,time_unix=time.time()))
            code=p.wait()
        assert code==0,('HISTORICAL_RUNNER_EXECUTION_FAILED',args,code)
    got=read(d/'smoke/repeat0/result.json');old=read(OLD/'smoke/repeat0/result.json')
    comparisons={k:got[k]==old[k] for k in ['FP_step_hashes','target','scores','endpoint']}
    cp(d/'smoke/repeat0/result.json',Q/'historical_self_replay_result.json')
    first=next((i+1 for i,(a,b) in enumerate(zip(got['FP_step_hashes'],old['FP_step_hashes'])) if a!=b),None)
    passed=all(comparisons.values())
    emit(Q/'HISTORICAL_RUNNER_SELF_REPLAY_GATE.json',dict(gate='PASS' if passed else 'FAIL',comparisons=comparisons,
         historical_future_KL=old['target']['TARGET_FUTURE_KL_AUC'],current_future_KL=got['target']['TARGET_FUTURE_KL_AUC'],
         first_FP_hash_divergence_token=first,exactness='Original historical scores/target/endpoint/FP_step_hashes equality rule',
         next_phase='B' if passed else 'C',FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
    archive();print('PHASE_A_SELF_REPLAY', 'PASS' if passed else 'FAIL','NEXT_PHASE','B' if passed else 'C',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        if Q.exists() and not (Q/'PHASE_A_ENGINEERING_STOP.json').exists():
            emit(Q/'PHASE_A_ENGINEERING_STOP.json',dict(error=repr(e),traceback=traceback.format_exc(),FORMAL_MEASUREMENTS=0))
            archive()
        raise
