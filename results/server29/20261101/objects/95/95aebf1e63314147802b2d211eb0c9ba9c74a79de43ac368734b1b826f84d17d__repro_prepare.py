"""Prepare append-only scientific reproducibility qualification; no model execution."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, sys
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3'); C=R/'provenance_capsule_v3'
Q=C/'scientific_reproducibility_v1'; INPUT=Path('/data/zypan/scientific_repro_v1_inputs')
CACHE=Path('/data/zypan/experiment_caches/ling_observability_scientific_repro_v1')
RUNS=Path('/data/zypan/experiment_runs/ling_observability_scientific_repro_v1')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
    assert head=='19e7b791207a5796985df1ce6afcf72b4a7111d4'
    assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=R,text=True).strip()
    Q.mkdir(); CACHE.mkdir(parents=True,exist_ok=False); RUNS.mkdir(parents=True,exist_ok=False)
    assert not list(CACHE.iterdir())
    emit(Q/'CACHE_CREATED_EMPTY.json',dict(path=str(CACHE),entries=[],old_unknown_cache_used=False,
         timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    for name in ['triton','torchinductor','tmp','triton_dump','triton_override']:(CACHE/name).mkdir()
    old_files=subprocess.check_output(['git','ls-files','provenance_capsule_v3'],cwd=R,text=True).splitlines()
    emit(Q/'HISTORICAL_FAILURES_AND_SOURCE_IDENTITY.json',dict(parent_commit=head,
         files=[dict(path=str((R/n).relative_to(C)),sha256=sha(R/n)) for n in old_files],
         historical_FAIL_status='PERMANENT_FAIL_PRESERVED; no retroactive PASS'))
    sources=[]
    for p in sorted(INPUT.iterdir()):
        if p.is_file():
            data=p.read_bytes()
            with (Q/p.name).open('xb') as f:f.write(data)
            sources.append(dict(path=p.name,sha256=sha(Q/p.name)))
    emit(Q/'INPUT_SOURCE_IDENTITY.json',dict(files=sources))
    profile=read(C/'measurement_launcher_environment.json')
    profile['set'].update(TRITON_CACHE_DIR=str(CACHE/'triton'),TORCHINDUCTOR_CACHE_DIR=str(CACHE/'torchinductor'),
                          TMPDIR=str(CACHE/'tmp'),TRITON_DUMP_DIR=str(CACHE/'triton_dump'),TRITON_OVERRIDE_DIR=str(CACHE/'triton_override'))
    emit(Q/'launcher_profile.json',profile)
    assets={}
    for n in ['scientific_asset_manifest.json','target_historical_input_identity.json','target_historical_preservation_before_qualification.json']:
        for row in read(C/n)['files']:
            assert row['path'] not in assets or assets[row['path']]['sha256']==row['sha256']
            assets[row['path']]=dict(row)
    checked=[]
    for i,(path,row) in enumerate(sorted(assets.items())):
        p=Path(path);assert sha(p)==row['sha256'],('SCIENTIFIC_ASSET_MISMATCH',path)
        stat=p.stat();checked.append(dict(path=path,sha256=row['sha256'],size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                                          ctime_ns=stat.st_ctime_ns,dev=stat.st_dev,ino=stat.st_ino))
        if i%100==0:print('FROZEN_ASSET_SHA_VERIFIED',i,len(assets),flush=True)
    runtime=[]
    for row in read(C/'runtime_environment_manifest.json')['environment_files']:
        assert sha(row['path'])==row['sha256'],('RUNTIME_IDENTITY_MISMATCH',row['path'])
        runtime.append(row)
    emit(Q/'ASSET_IDENTITY_FREEZE.json',dict(files=checked,source_manifest_sha256=sha(C/'scientific_asset_manifest.json'),
         verification='full SHA256 before qualification; each child rechecks stat identity plus full frozen scientific SHA once per process'))
    emit(Q/'RUNTIME_IDENTITY_FREEZE.json',dict(manifest_sha256=sha(C/'runtime_environment_manifest.json'),
         environment_files_count=len(runtime),actual=read(C/'runtime_environment_manifest.json')['actual']))
    required=read(C/'storage_capacity_identity.json')['required_total_bytes'];free=shutil.disk_usage(R).free
    assert free>required,('STORAGE_CAPACITY_FAILURE',free,required)
    emit(Q/'PREPARATION_COMPLETE.json',dict(gate='PASS',parent_commit=head,source_count=len(old_files),
         scientific_assets_count=len(checked),runtime_files_count=len(runtime),free_bytes=free,required_bytes=required,
         FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO',DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64'))
    print('SCIENTIFIC_REPRO_PREPARATION_PASS',flush=True)
if __name__=='__main__':main()
