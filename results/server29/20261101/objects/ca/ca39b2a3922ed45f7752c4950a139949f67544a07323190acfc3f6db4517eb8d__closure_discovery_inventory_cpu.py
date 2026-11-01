#!/usr/bin/env python3
"""No-import, CPU disk inventory of all strict-classifier native candidates."""
import hashlib,json,os,platform,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
SITE=Path('/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages')
ROOTS=[SITE/x for x in ['torch','torchaudio','torchvision','torch.libs','torchaudio.libs','torchvision.libs','nvidia']]
CONTRACT=Path('/data/zypan/analyses/qwen_gdn_fast_runtime_qualification_v1/gpu_runs/CUBLAS_4GPU_20261019_V1/gpu0/analysis/STRICT_BASELINE_CONTRACT.json')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def meta(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def role(p):
 b=Path(p).name
 return 'torch' if b.startswith(('libtorch','libc10')) else 'cuBLAS' if b.startswith('libcublas') else 'cuDNN' if b.startswith('libcudnn') else 'CUDA_runtime' if b.startswith('libcudart') else 'driver' if b.startswith('libcuda.so') else None
aliases={};errors=[]
for root in ROOTS:
 if not root.exists():continue
 for base,dirs,files in os.walk(root,followlinks=False):
  for n in files:
   p=Path(base)/n
   if role(p) is None or '.so' not in n:continue
   if not p.is_file():errors.append({'path':str(p),'error':'NOT_REGULAR_OR_BROKEN_SYMLINK'});continue
   q=p.resolve()
   with q.open('rb')as f:magic=f.read(4)
   if magic!=b'\x7fELF':errors.append({'path':str(p),'error':'SELECTED_SO_NOT_ELF'});continue
   aliases.setdefault(str(q),[]).append(str(p))
rows=[]
for i,(p,alias)in enumerate(sorted(aliases.items())):
 q=Path(p);before=meta(q);h=sha(q);after=meta(q)
 if before!=after:raise RuntimeError('Native candidate changed while SHA read:'+p)
 rows.append(dict(path=p,resolved_path=p,bytes=after['bytes'],sha256=h,role=role(q),status='STABLE_CURRENT_CPU_HASH',metadata_before=before,metadata_after=after,symlink=False,ELF=True,discovery_basis='CONSERVATIVE_PACKAGE_FAMILY_DISK_INVENTORY_NOT_PROOF_OF_LOADING',inventory_alias_paths=sorted(set(alias))))
 if i%5==0:print(json.dumps({'phase':'CPU_candidate_SHA_progress','completed':i+1,'total':len(aliases)}),file=sys.stderr,flush=True)
old=json.loads(CONTRACT.read_bytes());allowed={str(Path(r.get('resolved_path',r['path'])).resolve()):r for r in old['prospective_library_bindings'] if 'path'in r and 'sha256'in r}
missing=[r for r in rows if r['role']!='driver'and r['resolved_path']not in allowed]
mismatches=[dict(current=r,old=allowed[r['resolved_path']])for r in rows if r['resolved_path']in allowed and(r['sha256']!=allowed[r['resolved_path']]['sha256']or r['bytes']!=allowed[r['resolved_path']]['bytes'])]
print(json.dumps(dict(schema='CLOSURE_DISCOVERY_CONSERVATIVE_NATIVE_INVENTORY_V1',status='PASS_CPU_NATIVE_FAMILY_DISK_INVENTORY'if not(errors or mismatches)else'FAIL_CPU_NATIVE_FAMILY_DISK_INVENTORY',unix=time.time(),hostname=platform.node(),inventory_roots=[dict(path=str(p),exists=p.exists())for p in ROOTS],candidate_inventory_rows=rows,missing_prospective_bindings=missing,existing_binding_mismatches=mismatches,errors=errors,no_import_torch_or_transformers=True,no_GPU_calls=True,current_disk_identity_only_not_actual_loading_proof=True),sort_keys=True))
