"""CPU-only final archive, after all measurement/supervisor processes exit."""
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from v4_common import CODE,OUT,RUNS,LATCH,digest,read,emit,checked,alive

def main():
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')==''
    final=checked(OUT/'FINAL_STATUS.json')
    assert not final['active_measurement_workers']
    assert not any(alive(p.name.removesuffix('.PROCESS_IDENTITY.json')) for p in RUNS.glob('*.PROCESS_IDENTITY.json'))
    for session in ('ling_p1l7_v4_stage1','ling_p1l7_v4_portability','ling_p1l7_v4_transition','ling_p1l7_v4_persistent'):
        assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0,('SUPERVISOR_STILL_ACTIVE',session)
    if final['completed_student_32K']:
        tables=checked(OUT/'FORMAL_NUMERIC_TABLES.json')
        assert (OUT/'PERSISTENT_FUTURE_KL_RESULTS.csv').exists() and (OUT/'PAIRED_P1_L7_COMPARISON.csv').exists()
        validation=checked(OUT/'CSV_DELIVERY_VALIDATION.json')
        assert validation['status']=='PASS' and validation['source_sha256']==digest(OUT/'FORMAL_NUMERIC_TABLES.json')
    else:
        assert not (OUT/'PERSISTENT_FUTURE_KL_RESULTS.csv').exists() and not (OUT/'PAIRED_P1_L7_COMPARISON.csv').exists()
    freeze_path=OUT/'V4_PERSISTENT_CODE_FREEZE.json'
    if not freeze_path.exists(): freeze_path=OUT/'V4_PORTABILITY_CODE_FREEZE.json'
    freeze=checked(freeze_path)
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==freeze['code_commit']
    for p,h in freeze['code_sha256'].items(): assert digest(p)==h,('FROZEN_CODE_CHANGED',p)
    ids=checked(OUT/'V4_REUSED_SOURCE_IDENTITY.json')
    for p,h in ids['historical_STOP_sha256'].items(): assert digest(p)==h,('HISTORICAL_STOP_CHANGED',p)
    # Weight bytes stay in place; their frozen fresh metadata and earlier full
    # hash audit are delivered. Copy all other reused source/input/rotation assets
    # once, with original-path identity, without overwriting the source or output.
    provenance=OUT/'reused_readonly_provenance'; provenance.mkdir(exist_ok=False)
    copies={}
    for source,row in ids['assets'].items():
        p=Path(source); stat=p.stat()
        assert stat.st_size==row['bytes'] and stat.st_mtime_ns==row['mtime_ns']
        if p.suffix=='.safetensors': continue
        assert digest(p)==row['sha256']
        target=provenance/(row['sha256'][:16]+'_'+p.name)
        if not target.exists():
            with p.open('rb') as src,target.open('xb') as dst: shutil.copyfileobj(src,dst)
        assert digest(target)==row['sha256']
        copies[source]=dict(relative_path=str(target.relative_to(OUT)),sha256=row['sha256'],bytes=row['bytes'])
    for p,h in ids['historical_STOP_sha256'].items():
        source=Path(p); target=provenance/(h[:16]+'_'+source.name)
        if not target.exists():
            with source.open('rb') as src,target.open('xb') as dst: shutil.copyfileobj(src,dst)
        copies[p]=dict(relative_path=str(target.relative_to(OUT)),sha256=h,bytes=target.stat().st_size)
    emit(OUT/'DELIVERED_REUSED_ASSET_PROVENANCE.json',dict(files=copies,weight_binaries_not_duplicated=True,
        frozen_weight_hash_and_fresh_metadata_manifest='V4_REUSED_SOURCE_IDENTITY.json',historical_bytes_never_modified=True))
    archive=OUT/'MEASUREMENT_CODE_ARCHIVE.tar.gz'
    with archive.open('xb') as f:
        subprocess.run(['git','-C',str(CODE),'archive','--format=tar.gz','--prefix=measurement_code/',freeze['code_commit']],stdout=f,check=True)
    helpers=OUT/'append_only_CPU_delivery_code'; helpers.mkdir(exist_ok=False)
    for n in ('v4_monitor.py','v4_seal_delivery.py'):
        p=CODE/n
        if p.exists():
            with p.open('rb') as src,(helpers/n).open('xb') as dst: shutil.copyfileobj(src,dst)
    for p in OUT.rglob('*.json'):
        if p.name.endswith('.sha256.json'): continue
        side=p.with_suffix('.sha256.json')
        if side.exists(): assert digest(p)==read(side)['sha256'],('SIDECAR_MISMATCH',p)
    files={str(p.relative_to(OUT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file()}
    emit(OUT/'RESULT_AND_DELIVERY_HASHES.json',dict(status='SEALED',files=files,code_commit=freeze['code_commit'],
        archive_sha256=digest(archive),epoch=time.time(),total_pre_manifest_bytes=sum(x['bytes'] for x in files.values()),
        self_and_SHA256SUMS_excluded_to_avoid_circular_hash=True,formal_students=final['completed_student_32K'],
        scientific_verdict=final['P1_VS_L7_SCIENTIFIC_VERDICT']))
    files={str(p.relative_to(OUT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file()}
    with (OUT/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for name,row in sorted(files.items()): f.write(row['sha256']+'  '+name+'\n')
    files['SHA256SUMS.txt']=dict(sha256=digest(OUT/'SHA256SUMS.txt'),bytes=(OUT/'SHA256SUMS.txt').stat().st_size)
    path=OUT.parent/(OUT.name+'__delivery_'+str(time.time_ns())+'.zip')
    with path.open('xb') as f:
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for name,row in sorted(files.items()):
                p=OUT/name; assert digest(p)==row['sha256'],('FILE_CHANGED_DURING_SEAL',p)
                z.write(p,name)
    print(json.dumps(dict(status='ZIP_CREATED_NOT_YET_LOCAL_VERIFIED',zip_path=str(path),zip_sha256=digest(path),zip_bytes=path.stat().st_size,
        archive_entry_count=len(files),SHA256SUMS_sha256=files['SHA256SUMS.txt']['sha256'],all_files=files),sort_keys=True))

if __name__=='__main__': main()
