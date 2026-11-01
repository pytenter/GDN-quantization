#!/usr/bin/env python3
"""Current CPU-only import closure probe; never model-construction or GPU work."""
import ast, hashlib, importlib, importlib.util, json, os, platform, sys, time, traceback, types
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
CONTRACT=Path('/data/zypan/analyses/qwen_gdn_fast_runtime_qualification_v1/gpu_runs/CUBLAS_4GPU_20261019_V1/gpu0/analysis/STRICT_BASELINE_CONTRACT.json')
START=time.time()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def metadata(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def stable_row(p):
 p=Path(p);resolved=p.resolve();before=metadata(resolved);h=sha(resolved);after=metadata(resolved)
 if before!=after:raise RuntimeError('File changed while CPU SHA read:'+str(p))
 with resolved.open('rb') as f:magic=f.read(4)
 return dict(path=str(p),resolved_path=str(resolved),bytes=after['bytes'],sha256=h,status='STABLE_CURRENT_CPU_HASH',metadata_before=before,metadata_after=after,symlink=p.is_symlink(),ELF=magic==b'\x7fELF')
def role(p):
 base=Path(p).name
 return 'torch' if base.startswith(('libtorch','libc10')) else 'cuBLAS' if base.startswith('libcublas') else 'cuDNN' if base.startswith('libcudnn') else 'CUDA_runtime' if base.startswith('libcudart') else 'driver' if base.startswith('libcuda.so') else None
phases=[]
def snap(name):
 text=Path('/proc/self/maps').read_text();paths=[];deleted=[]
 for line in text.splitlines():
  f=line.split(None,5)
  if len(f)==6 and f[5].startswith('/'):
   if f[5].endswith(' (deleted)'):deleted.append(f[5])
   else:paths.append(str(Path(f[5]).resolve()))
 item=dict(phase=name,unix=time.time(),full_maps_text=text,full_maps_sha256=hashlib.sha256(text.encode()).hexdigest(),mapped_paths=sorted(set(paths)),deleted_paths=sorted(set(deleted)))
 phases.append(item);return item
contract=json.loads(CONTRACT.read_bytes());source_proofs={'contract':stable_row(CONTRACT)}
for key in ['core','codec','worker','baseline_worker','architecture']:
 if key in contract['source_bindings']:
  r=contract['source_bindings'][key];actual=stable_row(r['path'])
  if actual['sha256']!=r['sha256'] or actual['bytes']!=r['bytes']:raise RuntimeError('Bound source changed:'+key)
  source_proofs[key]=actual
snap('stdlib_before_engine_import')
errors=[];blocked=[]
try:
 import torch
 snap('torch_import_only_before_guards')
 def deny(name):
  def blocked_cuda(*args,**kwargs):
   blocked.append(name);raise RuntimeError('CPU_ONLY_DISCOVERY_BLOCKED_CUDA:'+name)
  return blocked_cuda
 # These guards prevent CUDA initialization/allocations and all explicit CUDA
 # availability queries from our downstream CPU import exploration. They are
 # disclosed evidence conditions, never a prospective scientific runtime.
 torch.cuda._lazy_init=deny('torch.cuda._lazy_init')
 torch._C._cuda_init=deny('torch._C._cuda_init')
 torch.cuda.is_available=lambda:False
 torch.cuda.device_count=lambda:0
 if hasattr(torch._C,'_cuda_getDeviceCount'):torch._C._cuda_getDeviceCount=lambda:0
 for name in ['_cuda_setDevice','_cuda_getDevice','_cuda_getDeviceProperties']:
  if hasattr(torch._C,name):setattr(torch._C,name,deny('torch._C.'+name))
 bindings=contract['source_bindings']
 def original_module(key,modname):
  spec=importlib.util.spec_from_file_location(modname,bindings[key]['path']);mod=importlib.util.module_from_spec(spec);sys.modules[modname]=mod;spec.loader.exec_module(mod);return mod
 K=original_module('core','_repair_closure_original_core')
 C=original_module('codec','_repair_closure_original_codec')
 snap('original_core_and_codec_imports_no_configure')
 for p in ('/data/zypan/experiments/qwen35_gdn_quant','/data/zypan/transformers-qwen35'):
  if p not in sys.path:sys.path.insert(0,p)
 from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
 snap('original_transformers_AutoModel_AutoTokenizer_imports')
 # Resolve the exact AutoModel class selected from local config, never invoke
 # from_pretrained/model constructor nor read weight shards.
 model_path=contract['model']['path']
 config=AutoConfig.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
 from transformers.models.auto.auto_factory import _get_model_class
 model_class=_get_model_class(config,AutoModelForCausalLM._model_mapping)
 qmod=importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
 architecture=bindings['architecture']['path'] if 'architecture'in bindings else '/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py'
 if Path(qmod.__file__).resolve()!=Path(architecture).resolve():raise RuntimeError('Architecture module path differs')
 snap('exact_Qwen3_5_AutoModel_class_resolved_without_constructor')
 tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
 snap('original_tokenizer_loaded_CPU_only')
 versions=dict(python=platform.python_version(),torch=torch.__version__,torch_build_CUDA=torch.version.cuda,transformers=importlib.import_module('transformers').__version__,tokenizer_class=type(tokenizer).__name__,model_class_module=model_class.__module__,model_class_name=model_class.__name__,model_constructed=False,weight_shards_opened_by_probe=False)
except BaseException as exc:
 errors.append(dict(error=repr(exc),traceback=traceback.format_exc()));versions=dict(python=platform.python_version())
final=snap('final_CPU_import_maps')
# Hash every actual mapped physical file, including non-selected DSOs. No
# ldd/load/probe of additional unobserved libraries is performed.
all_paths=sorted(set(p for phase in phases for p in phase['mapped_paths']))
all_rows=[]
for i,p in enumerate(all_paths):
 r=stable_row(p);r['role']=role(p);r['discovery_phases']=[x['phase'] for x in phases if p in x['mapped_paths']];all_rows.append(r)
 if i%20==0:print(json.dumps({'phase':'CPU_SHA_progress','files_completed':i+1,'files_total':len(all_paths)}),file=sys.stderr,flush=True)
selected=[r for r in all_rows if r['role'] is not None]
allowed={str(Path(r.get('resolved_path',r['path'])).resolve()):r for r in contract['prospective_library_bindings'] if 'path'in r and 'sha256'in r}
missing=[r for r in selected if r['role']!='driver' and r['resolved_path'] not in allowed]
mismatches=[]
for r in selected:
 old=allowed.get(r['resolved_path'])
 if old and (old['bytes']!=r['bytes'] or old['sha256']!=r['sha256']):mismatches.append(dict(actual=r,old=old))
modules=[]
for name,mod in sorted(sys.modules.copy().items()):
 p=getattr(mod,'__file__',None)
 if p:modules.append({'module':name,'file':str(p)})
result=dict(schema='CLOSURE_DISCOVERY_CPU_IMPORT_V1',status='PASS_CPU_IMPORT_CLOSURE_DISCOVERY' if not(errors or mismatches or blocked or any(x['deleted_paths']for x in phases)) else 'FAIL_CPU_IMPORT_CLOSURE_DISCOVERY',started_unix=START,finished_unix=time.time(),machine=dict(hostname=platform.node(),uname=list(platform.uname()),executable=sys.executable,affinity=sorted(os.sched_getaffinity(0)),uid=os.getuid(),cwd=os.getcwd(),sys_path=sys.path,loadavg=list(os.getloadavg())),versions=versions,environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH']},source_proofs=source_proofs,phases=phases,full_maps_text=final['full_maps_text'],full_maps_sha256=final['full_maps_sha256'],all_mapped_files=all_rows,observed_library_rows=selected,missing_prospective_bindings=missing,existing_binding_mismatches=mismatches,deleted_mappings=[p for x in phases for p in x['deleted_paths']],imported_modules=modules,errors=errors,blocked_CUDA_calls=blocked,no_model_constructor=True,no_model_forward=True,no_GPU_workload=True,no_explicit_CUDA_or_NVML_or_nvidia_smi_queries=True,no_old_root_writes=True,current_evidence_not_retrospective_runtime_PASS=True,limitations=['CPU import discovery hides GPUs and returns False/0 for downstream CUDA availability checks; initialization entrypoints throw. This is not an exact GPU runtime observation.','Model class and original tokenizer were loaded without constructing a model or reading weight shards; GPU-lazy or from_pretrained-only native dependencies may be absent.','Fresh actual loaded-map SHA/bytes gate must run after real model loading before the first scientific forward, and again after each unit.','Old STOP/failed library identity remains permanent; current disk SHA does not retroactively qualify old runtime.'])
print(json.dumps(result,ensure_ascii=False,sort_keys=True))
