"""Identity preparation for libraries actually loaded now or in frozen smoke."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent
OLD=Path('/data/zypan/worktrees/ling-observability-provenance-amendment-v1/checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def emit(name,x):
    with (C/name).open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
def main():
    assert not (C/'STOP.json').exists(),'NO_GATE_RETRY'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    import torch, transformers, triton, numpy
    torch.cuda.init()
    libs={}
    origins={}
    for repeat in [0,1]:
        p=OLD/'smoke'/f'repeat{repeat}'/'result.json'
        x=json.loads(p.read_text())
        for r in x['dependencies']['execution_module_files']:
            if r['path'].endswith('.so'):
                f=Path(r['path'])
                assert f.is_file() and sha(f)==r['sha256'],('HISTORICAL_NATIVE_RUNTIME_IDENTITY_FAILURE',r)
                libs[str(f)]=f;origins[str(f)]='actually_imported_in_frozen_historical_smoke'
    libs[sys.executable]=Path(sys.executable);origins[sys.executable]='python_executable'
    ldd=[]
    for p in list(libs.values()):
        run=subprocess.run(['ldd',str(p)],capture_output=True,text=True)
        assert run.returncode==0,('LDD_FAILED',str(p),run.stderr)
        ldd.append(dict(path=str(p),stdout=run.stdout))
        for line in run.stdout.splitlines():
            tokens=line.replace('=>',' ').split()
            for word in tokens:
                if word.startswith('/') and Path(word).is_file():
                    libs[word]=Path(word);origins[word]='linked_dependency_of_actually_loaded_numerical_extension'
    for line in Path('/proc/self/maps').read_text().splitlines():
        bits=line.split(None,5)
        if len(bits)==6 and bits[5].startswith('/') and '.so' in bits[5]:
            f=Path(bits[5]);assert f.is_file()
            libs[str(f)]=f;origins[str(f)]='actually_loaded_during_Python_Torch_CUDA_identity_preparation'
    rows=[]
    for name,p in sorted(libs.items()):
        rows.append(dict(path=name,realpath=str(p.resolve()),sha256=sha(p),role=origins[name]))
    emit('shared_library_manifest.json',dict(files=rows,ldd=ldd,policy='Actual historical imported extensions, their resolved linked dependencies, and currently mapped numerical runtime; not system-wide library hashing.',GPU6_FORMAL_AUTHORIZED='NO'))
    exes=[]
    for name in ['hostname','nvidia-smi','gcc','cc','g++','ld','as','objcopy','strip','ptxas','ldconfig','uname']:
        path=shutil.which(name)
        if path:exes.append(dict(logical_name=name,path=path,realpath=str(Path(path).resolve()),sha256=sha(path)))
    exes.append(dict(logical_name='python',path=sys.executable,realpath=str(Path(sys.executable).resolve()),sha256=sha(sys.executable)))
    for p in Path('/data/zypan/envs/ling-kda/lib/python3.11/site-packages/triton/backends').rglob('*'):
        if p.is_file() and p.name in ['ptxas','cuobjdump','nvdisasm']:
            exes.append(dict(logical_name='triton_'+p.name,path=str(p),realpath=str(p.resolve()),sha256=sha(p)))
    emit('executable_runtime_manifest.json',dict(files=exes,Git_allowed_in_sealed_measurement=False))
    print('EXPLICIT_NATIVE_RUNTIME_PREPARATION_COMPLETE',len(rows),'library identities',flush=True)
if __name__=='__main__':main()
