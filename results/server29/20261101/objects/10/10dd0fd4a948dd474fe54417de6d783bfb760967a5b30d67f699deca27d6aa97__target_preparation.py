"""V3 preparation: explicit identities, not regular-file ingress tracing."""
from pathlib import Path
import hashlib, importlib.metadata as md, json, locale, os, shutil, subprocess, sys, time
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent
TASK='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
OLD=Path('/data/zypan/worktrees/ling-observability-provenance-amendment-v1/checkout/experiments')/TASK
PROP=Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1')
CO=Path('/data/zypan/worktrees/ling-error-coherence-diagnosis-v1/experiments/LING_ERROR_COHERENCE_DIAGNOSIS_V1')
SRC_PROP='/data01/user2/worktrees/ling-long-horizon-propagation-confirm-v1'
SRC_ENV='/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM/runtime/env'
ENV=Path('/data/zypan/envs/ling-kda')
UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(name,x):
    p=C/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def mapped(p):
    p=str(p)
    if p.startswith(SRC_PROP+'/'):return PROP/Path(p).relative_to(SRC_PROP)
    if p.startswith(SRC_ENV+'/'):return ENV/Path(p).relative_to(SRC_ENV)
    if p.startswith('/data01/user2/models/'):return Path('/data/zypan/models')/Path(p).relative_to('/data01/user2/models')
    if p.startswith('/data01/user2/'):return PROP/'inherited'/p.lstrip('/')
    return Path(p)
def command(a):return subprocess.check_output(a,text=True).strip()
def main():
    status=command(['git','-C',str(C.parent),'status','--porcelain=v1'])
    assert not status,('CLEAN_WORKTREE_GATE',status)
    emit('target_initial_clean_worktree.json',dict(gate='PASS',HEAD=command(['git','-C',str(C.parent),'rev-parse','HEAD']),status_porcelain='',root=str(C.parent)))
    rows=[];bad=[]
    original=read(OLD/'source_identity.json')
    for i,r in enumerate(original['inputs']):
        p=mapped(r['path']);actual=sha(p) if p.is_file() else None
        role='runtime_package' if '/site-packages/' in str(p) else 'project_source' if p.suffix=='.py' and '/models/' not in str(p) else 'scientific_input'
        item=dict(original_absolute_path=r['path'],path=str(p),size=p.stat().st_size if p.exists() else None,sha256=actual,expected_sha256=r['sha256'],role=role,identity='PASS' if actual==r['sha256'] else 'FAIL')
        rows.append(item)
        if actual!=r['sha256']:bad.append(item)
        if i%100==0:print('EXPLICIT_HISTORICAL_IDENTITY',i,len(original['inputs']),flush=True)
    for r in read(C/'historical_source_snapshot_manifest.json')['files']:
        p=C/r['snapshot_relative_path'];assert sha(p)==r['original_sha256']==r['snapshot_sha256']
    emit('target_historical_input_identity.json',dict(gate='PASS' if not bad else 'FAIL',files=rows,mismatches=bad))
    if bad:
        emit('STOP.json',dict(reason='SCIENTIFIC_ASSET_HASH_MISMATCH',mismatches=bad,FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO',FINAL_ACTION='FAIL_AND_STOP'));return
    assets=[x for x in rows if x['role']=='scientific_input']
    for name in ['preregistration.json','preregistration.md','document_manifest.json','rotation_manifest.json','basis_semantics_audit.md','pulse_definition.md','runtime_manifest.json','source_identity.json']:
        src=OLD/name;dest=C/'frozen_protocol'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest);assert sha(src)==sha(dest)
        assets.append(dict(path=str(dest),size=dest.stat().st_size,sha256=sha(dest),scientific_role='byte_identical_existing_preregistration_or_authority'))
    docs=read(OLD/'document_manifest.json')['documents']
    for doc in docs:
        p=CO/'discovery/traces'/doc['document_id']/'trace_freeze.json';trace=read(p)
        assert trace['gate']=='PASS' and trace['tokens']==32768
        assets.append(dict(path=str(p),size=p.stat().st_size,sha256=sha(p),scientific_role='same_GPU6_frozen_FP_drive_manifest'))
        for j,entry in enumerate(trace['chunks']):
            f=entry['artifact'];assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256'],('FROZEN_DRIVE_IDENTITY_FAILURE',f)
            assets.append(dict(path=f['path'],size=f['bytes'],sha256=f['sha256'],scientific_role='same_GPU6_frozen_teacher_forced_FP_drive_chunk'))
        print('FROZEN_DRIVE_COMPLETE',doc['document_id'],len(trace['chunks']),flush=True)
    emit('scientific_asset_manifest.json',dict(gate='PASS',model_revision='e3a47d5b986e7141b6efd62597d598ebb392060d',FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',files=assets))
    import torch, transformers, triton, fla, numpy
    expected_path=mapped('/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V4_DETERMINISTIC/runtime/v4_deterministic_runtime_manifest.json')
    assert sha(expected_path)=='0fd12303bdb54c6e54ee72963f0fc78f6e1791c5a807869fca9cf7b5d723dbb4'
    expected=read(expected_path)['package_environment']['current']
    actual=dict(python=sys.version,packages={k:md.version(k) for k in expected['packages']})
    equal=actual==expected
    packages={}
    for dist in ['torch','transformers','triton','flash-linear-attention','fla-core','numpy','scipy']:
        try:packages[dist]=dict(version=md.version(dist),distribution_path=str(md.distribution(dist).locate_file('')))
        except md.PackageNotFoundError:packages[dist]=dict(version='NOT_INSTALLED')
    # Freeze environment file identities, but copy no third-party source.
    envrows=[]
    for p in sorted((ENV/'lib/python3.11').rglob('*')):
        if p.is_file() and (p.suffix in {'.py','.so'} or p.name.endswith('.dist-info/RECORD')) and '__pycache__' not in p.parts:
            envrows.append(dict(path=str(p),realpath=str(p.resolve()),size=p.stat().st_size,sha256=sha(p)))
    generated=[]
    for name,m in list(sys.modules.items()):
        p=getattr(m,'__file__',None)
        if p and Path(p).is_file() and str(p).startswith('/tmp/'):
            generated.append(dict(module=name,path=p,sha256=sha(p),kind='environment_generated_module_after_torch_import'))
    emit('runtime_environment_manifest.json',dict(gate='PASS' if equal else 'FAIL',python=dict(path=sys.executable,realpath=str(Path(sys.executable).resolve()),sha256=sha(sys.executable),version=sys.version),packages=packages,torch_file=torch.__file__,torch_cuda=torch.version.cuda,transformers_file=transformers.__file__,FLA_path=fla.__file__,numpy_file=numpy.__file__,environment_files=envrows,generated_environment_modules=generated,expected=expected,actual=actual,exact_historical_comparison=equal,pip_freeze=command([sys.executable,'-m','pip','freeze']),locale=dict(environment={k:os.environ.get(k) for k in ['LANG','LC_ALL','LC_CTYPE','LC_NUMERIC','LC_COLLATE']},active=locale.setlocale(locale.LC_ALL)),ambient_OS_regular_file_closure_claim=False))
    if not equal:
        emit('STOP.json',dict(reason='RUNTIME_PACKAGE_IDENTITY_FAILURE',expected=expected,actual=actual,FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO',FINAL_ACTION='FAIL_AND_STOP'));return
    gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader,nounits'])
    proc=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'])
    g=next(l for l in gpu.splitlines() if l.split(',')[0].strip()=='6').split(',')
    clean=not any(UUID in l for l in proc.splitlines())
    assert g[1].strip()==UUID and '3090' in g[2] and clean,('HARDWARE_IDENTITY_FAILURE',gpu,proc)
    free=shutil.disk_usage(C).free
    emit('hardware_storage_preparation.json',dict(hardware=dict(host=command(['hostname']),physical_gpu=6,uuid=UUID,model=g[2].strip(),driver=g[5].strip(),available_VRAM_bytes=(int(g[3])-int(g[4]))*1024**2),available_bytes=free,historical_free_space_gate_bytes=20*1024**3,original_explicit_required_peak_bytes='NOT_DEFINED_IN_ORIGINAL_PREREGISTRATION',estimated_new_disk_peak_bytes='NOT_YET_BOUNDED',gate='PENDING_STORAGE_PEAK_RECONCILIATION',GPU6_FORMAL_AUTHORIZED='NO',SUPERVISOR_CLEANUP_LEDGER_REMAINING=1,GPU6_COMPUTE_PROCESS_COUNT=0,actual_GPU_process_table=proc))
    print('V3_TARGET_EXPLICIT_ASSET_ENVIRONMENT_PREPARATION_COMPLETE',len(assets),len(envrows),flush=True)
if __name__=='__main__':main()
