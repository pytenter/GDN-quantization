"""Read-only final source/tensor identity verification after human stop.

The only output is a new independent-review JSON. Never loads a model, tensor,
Torch or CUDA, sends a signal, edits frozen files, or reseals the parent run.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(root,rows):
    failures=[];total=0
    for row in rows:
        p=Path(row['path']);p=p if p.is_absolute() else root/p
        if not p.is_file():
            failures.append({'path':str(p),'reason':'missing'});continue
        size=p.stat().st_size;digest=sha(p);total+=size
        if size!=row.get('bytes',size) or digest!=row['sha256']:
            failures.append({'path':str(p),'actual_bytes':size,'actual_sha256':digest,'expected':row})
    return {'status':'PASS' if not failures else 'FAIL','files':len(rows),'total_bytes':total,'mismatches':failures}


def run(root):
    started=time.time();root=root.resolve();parent=root.parent
    source=read(root/'SOURCE_MANIFEST.json')
    internal=verify(root,source['execution_sources']);external=verify(root,source['external_dependencies'])
    tensors=[]
    for name,expected,size in [
        ('qwen35_9b_stepquant6_plan.pt','a0613ebdf249f90f6ef128b283733ca4368baca8fbeef6e09b1f4db25cd0fa79',1209542),
        ('qwen35_9b_statistics.pt','92522eb402cc84ff0d8f3757bfe9ce57d6a056c2ff958ba82cf8fbcac9d0b35c',3222054270)]:
        p=root/'calibration'/name;before=p.stat();digest=sha(p);after=p.stat()
        stable=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
        tensors.append({'path':str(p),'status':'PASS' if digest==expected and before.st_size==size and stable else 'FAIL',
            'bytes':before.st_size,'sha256':digest,'expected_sha256':expected,'stable_stat_during_hash':stable,
            'hash_scope':'Read-only remote streaming; no deserialize/model/CUDA/download'})
    parent_manifest=parent/'SHA256_MANIFEST.json';parent_sha=sha(parent_manifest)
    lineage_sha=sha(root/'lineage/SHA256_MANIFEST.json')
    seal=read(parent_manifest)
    parent_entries=verify(parent,seal['files'])
    parent_reference={'manifest_path':str(parent_manifest),'manifest_sha256':parent_sha,
        'lineage_manifest_sha256':lineage_sha,'expected_manifest_sha256':'dc0e70df32b2ff64ab6230409657589fdaacad99500e526c8583241d61424e34',
        'expected_file_count':247,'fresh_entries':parent_entries,'SHA256SUMS_sha256':sha(parent/'SHA256SUMS.txt'),
        'expected_SHA256SUMS_sha256':'85bbcda8ce943e2a8f93e59fb9a3503b5e105287e137beeca7461b850a9e0409',
        'action':'Read-only reference/fresh validation; parent was not resealed or modified'}
    parent_reference['status']='PASS' if (parent_sha==lineage_sha==parent_reference['expected_manifest_sha256'] and len(seal['files'])==247 and
        parent_entries['status']=='PASS' and parent_reference['SHA256SUMS_sha256']==parent_reference['expected_SHA256SUMS_sha256']) else 'FAIL'
    auth=read(root/'AUTHORIZATION_FORMAL.json');model=root/'MODEL_TOKENIZER_FRESH_IDENTITY.json';identity=read(model)
    model_proof={'status':'PASS' if sha(model)==auth['model_tokenizer_identity_sha256'] and identity['status']=='PASS' and
        len(identity['files'])==17 and all(v['PASS'] and v['sha256']==v['expected_sha256'] for v in identity['files']) else 'FAIL',
        'path':str(model),'sha256':sha(model),'files':17,'scope':'Reference prior fresh complete-content SHA proof. Model weights not re-read by this check.'}
    ledger=read(root/'GPU_RESOURCE_LEDGER.json');stop=read(root/'USER_REQUESTED_STOP_AND_THROUGHPUT_SCOPE_20261017.json')
    formal=[d for d in ledger['dispatches'] if d['stage']=='formal']
    nvml=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    active={int(s.split(',')[0]) for s in nvml.splitlines() if s.strip()}
    pids=[stop['expected_pid']]+[d['pid'] for d in formal]
    processes=[{'pid':p,'proc_exists':Path('/proc/'+str(p)).exists(),'in_NVML':p in active} for p in pids]
    release={'status':'PASS' if (len(formal)==4 and ledger['all_GPUs_released'] is True and ledger['own_NVML_pids_remaining']==[] and
        all(d['process_exited'] and d['NVML_released'] for d in formal) and all(not p['proc_exists'] and not p['in_NVML'] for p in processes)) else 'FAIL',
        'formal_worker_count':len(formal),'tracked_processes':processes,'formal_dispatch_records':formal,
        'ledger_sha256':sha(root/'GPU_RESOURCE_LEDGER.json'),'ledger_status':ledger['status'],
        'all_old_GPUs_released_per_ledger':ledger['all_GPUs_released'],
        'current_NVML_compute_apps':nvml,'scope':'Only the stopped old coordinator/workers must be gone. Other newly authorized jobs, if any, are not this run and are never terminated by this review.'}
    checks=[internal['status'],external['status'],parent_reference['status'],model_proof['status'],release['status']]+[v['status'] for v in tensors]
    result={'task_id':auth['task_id'],'RUN_ID':auth.get('RUN_ID'),'status':'PASS' if all(s=='PASS' for s in checks) else 'FAIL',
        'scope':'Final source/Plan/statistics/parentseal identity and stopped-process release verification. PASS is identity verification only; old scientific comparison remains user-stopped incomplete.',
        'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_source_sha256':sha(__file__),
        'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'formal_authorization_source_manifest_sha256':auth['source_manifest_sha256'],
        'execution_source_verification':internal,'external_source_verification':external,'actual_large_assets':tensors,
        'parent247_original_stop_reference':parent_reference,'model17_reference':model_proof,'old_run_release':release,
        'human_stop_reason_path':'USER_REQUESTED_STOP_AND_THROUGHPUT_SCOPE_20261017.json',
        'human_stop_reason_sha256':sha(root/'USER_REQUESTED_STOP_AND_THROUGHPUT_SCOPE_20261017.json'),
        'scientific_status':'USER_REQUESTED_STOP_INCOMPLETE_COMPARISON_NOT_SCORED',
        'no_frozen_source_or_parent_modification':True,'GPU_execution':False,'model_forward_count':0,
        'large_tensor_download':False,'elapsed_seconds':time.time()-started}
    if result['source_manifest_sha256']!=auth['source_manifest_sha256']:result['status']='FAIL'
    target=root/'independent_review/USER_STOP_FINAL_IDENTITY_VERIFICATION.json'
    with target.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,
        'execution_files':internal['files'],'external_files':external['files'],'parent_files':parent_entries['files']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    run(p.parse_args().root)
