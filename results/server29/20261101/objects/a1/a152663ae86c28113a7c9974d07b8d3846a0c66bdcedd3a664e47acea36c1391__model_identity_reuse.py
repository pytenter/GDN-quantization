"""Read-only exact model/checkpoint identity against the retained seal."""
import os
from pathlib import Path
import recovery_audit as A

N=A.N
manifest=Path('/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/provenance/runtime_identity.json')
N.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only checkpoint identity')
data=N.read(manifest);N.check(data['QWEN_RUNTIME_IDENTITY_GATE']=='PASS','retained model provenance')
rows=[]
for row in data['model_identity']['files']:
    path=Path(row['path']);before=path.stat();actual=N.sha(path);after=path.stat()
    N.check(actual==row['sha256'],'exact retained model/checkpoint hash '+str(path))
    N.check((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'model unchanged during identity read')
    rows.append({'path':str(path),'sha256':actual,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'status':'PASS'})
A.save('MODEL_CHECKPOINT_IDENTITY_REUSE.json',{'status':'PASS','source_manifest':str(manifest),'source_manifest_sha256':N.sha(manifest),
    'model_files':rows,'GPU_forward_count':0,'model_checkpoint_writes':0,'no_new_downloads':True})
print('MODEL_CHECKPOINT_IDENTITY_REUSE_PASS',len(rows),flush=True)
