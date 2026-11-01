"""Original scientific implementation, whole historical runtime; operational guards only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys,traceback
sys.dont_write_bytecode=True
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
H=Path('/data/zypan/experiment_runs/ling_historical_runtime_replay_v1/checkout')
HD=H/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
FD=R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    mode=sys.argv[1];assert mode in ['postcommit','discovery','analyze']
    receipt=read(Q/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json');commit=receipt['commit']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==commit
    assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=R,text=True).strip()
    assert read(Q/'HISTORICAL_RUNNER_SELF_REPLAY_GATE.json')['gate']=='PASS'
    assert read(Q/'phase_b_qualification_evidence/FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json')['gate']=='PASS'
    assert sha(HD/'scripts/bridge.py')=='e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd'
    for row in read(FD/'WHOLE_RUNTIME_PREMEASUREMENT_IDENTITY.json')['files']:
        assert sha(row['path'])==row['sha256'],('PREMEASUREMENT_IDENTITY_FAIL',row['path'])
    spec=importlib.util.spec_from_file_location('whole_known_good_historical_bridge',HD/'scripts/bridge.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    out=FD if mode!='postcommit' else FD/'postcommit'
    out.mkdir(parents=True,exist_ok=True);b.D=out;b.ROOT=R
    if mode=='postcommit':
        for name in ['source_identity.json','document_manifest.json','rotation_manifest.json','runtime_manifest.json','preregistration.json']:
            with (out/name).open('xb') as f:f.write((FD/name).read_bytes())
    verified=False;stats=[]
    def guard():
        nonlocal verified,stats
        assert not (FD/'STOP.json').exists(),'PRESERVED_FORMAL_STOP'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==commit
        assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=R,text=True).strip()
        if not verified:
            for row in read(R/'provenance_capsule_v3/scientific_asset_manifest.json')['files']:
                p=Path(row['path']);assert sha(p)==row['sha256'],('SCIENTIFIC_ASSET_MISMATCH',str(p))
                s=p.stat();stats.append((p,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
            for row in read(Q/'HISTORICAL_EXECUTED_PACKAGE_IDENTITY.json')['files']:
                p=Path(row['path']);assert sha(p)==row['sha256'],('HISTORICAL_RUNTIME_DRIFT',str(p))
                s=p.stat();stats.append((p,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
            verified=True
        else:
            for p,dev,ino,size,mtime,ctime in stats:
                s=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(dev,ino,size,mtime,ctime),('FROZEN_IDENTITY_CHANGED',str(p))
        if mode!='postcommit':assert read(Q/'GPU6_FORMAL_AUTHORIZATION.json')['GPU6_FORMAL_AUTHORIZED']=='YES'
        return dict(commit=commit,formal_authorized=mode!='postcommit')
    b.guard=guard
    if mode=='postcommit':
        b.verify();b.smoke(0)
        result=read(out/'smoke/repeat0/result.json');baseline=read(Q/'original_historical_reference.json')
        for k in ['FP_step_hashes','scores','target','endpoint']:assert result[k]==baseline[k],('POSTCOMMIT_PARITY_FAIL',k)
        for k in ['host','physical_gpu','gpu_uuid','driver','CUDA','python','torch','environment']:
            assert result['hardware'][k]==baseline['hardware'][k],('HARDWARE_RUNTIME_IDENTITY_FAIL',k)
        emit(out/'MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json',dict(gate='PASS',commit=commit,exact_historical_parity=True))
        print('POSTCOMMIT_IDENTITY_PARITY_PASS',flush=True)
    elif mode=='discovery':b.discovery()
    else:b.analyze()
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (FD/'STOP.json').exists():emit(FD/'STOP.json',dict(error=repr(e),traceback=traceback.format_exc(),FINAL_ACTION='FAIL_AND_STOP',mode=sys.argv[1]))
        raise
