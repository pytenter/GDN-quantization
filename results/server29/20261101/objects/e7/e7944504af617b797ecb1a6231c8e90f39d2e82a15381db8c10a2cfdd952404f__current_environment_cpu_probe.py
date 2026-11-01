#!/usr/bin/env python3
"""Read-only stdlib-only current source/env/model/native identity, never GPU."""
import hashlib,importlib.metadata,json,os,platform,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
request=json.load(sys.stdin)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def stat(p):
 s=p.stat();return {'bytes':s.st_size,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def identity(path,expected=None):
 p=Path(path)
 if not p.is_file():return {'path':str(p),'status':'MISSING_CURRENT_PATH'}
 q=p.resolve();before=stat(q);digest=sha(q);after=stat(q);stable=before==after and p.resolve()==q
 row={'path':str(p),'resolved_path':str(q),'bytes':after['bytes'],'sha256':digest,'symlink':p.is_symlink(),'metadata_before':before,'metadata_after':after,'status':'PASS_STABLE_CURRENT_CPU_SHA'if stable else'FAIL_CHANGED_DURING_HASH'}
 if expected is not None:row.update(expected_bytes=expected['bytes'],expected_sha256=expected['sha256'],historical_or_prior_binding_match=stable and after['bytes']==expected['bytes']and digest==expected['sha256'])
 return row
started=time.time();result={'schema':'CURRENT_ENVIRONMENT_CPU_DISK_BINDINGS_V1','status':'PASS_CPU_READ_ONLY_IDENTITY_COLLECTION','started_unix':started,'machine':{'hostname':platform.node(),'uname':list(platform.uname()),'python_version':platform.python_version(),'sys_version':sys.version,'executable':sys.executable,'libc':platform.libc_ver(),'affinity':sorted(os.sched_getaffinity(0))},'authorization':{'ALLOW_GPU_RESTORE':'NO','prior_four_GPU_authorization_applies':False},'no_torch_import':True,'no_model_import_or_forward':True,'no_GPU_query_or_initialization':True,'no_environment_installation_changes':True,'no_model_or_tensor_download':True}
keys=['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUDA_LAUNCH_BLOCKING']
result['ambient_login_environment']={k:{'present':k in os.environ,'value':os.environ.get(k),'scope':'CURRENT_SSH_CHILD_AMBIENT_NOT_HISTORICAL_PROOF'}for k in keys}
result['python_executable']=identity(sys.executable)
packages=[]
for name in ['torch','transformers','nvidia-cublas-cu12','nvidia-cuda-runtime-cu12','nvidia-cudnn-cu12','torchaudio','torchvision','safetensors','tokenizers','accelerate']:
 try:
  dist=importlib.metadata.distribution(name);row={'name':name,'version':dist.version,'metadata_directory':str(dist._path),'files':[]}
  for filename in ['METADATA','WHEEL','INSTALLER','RECORD','direct_url.json']:
   p=Path(dist._path)/filename;x=identity(p)
   if p.is_file()and filename in ['INSTALLER','direct_url.json','WHEEL']:x['text']=p.read_text()
   row['files'].append(x)
  packages.append(row)
 except importlib.metadata.PackageNotFoundError:packages.append({'name':name,'status':'NOT_INSTALLED_DISTRIBUTION_METADATA'})
result['installed_distribution_metadata']=packages
result['native_libraries']=[]
for i,expected in enumerate(request['native_libraries']):
 row=identity(expected['path'],expected);row['role']=expected.get('role');row['identity_scope']='CURRENT_DISK_NOT_HISTORICAL_LOADED_LIBRARY_PROOF';result['native_libraries'].append(row)
 if i%5==0:print(json.dumps({'phase':'current_native_CPU_SHA','completed':i+1,'total':len(request['native_libraries'])}),file=sys.stderr,flush=True)
result['model_small_files']=[]
for row in request['model_small_files']:result['model_small_files'].append(identity(row['path'],row if 'sha256'in row else None))
result['model_weight_shards']=[]
for i,row in enumerate(request['weight_shards']):
 print(json.dumps({'phase':'current_model_shard_CPU_SHA_started','shard':i+1,'bytes':row['bytes']}),file=sys.stderr,flush=True);result['model_weight_shards'].append(identity(row['path'],row));print(json.dumps({'phase':'current_model_shard_CPU_SHA_complete','shard':i+1}),file=sys.stderr,flush=True)
conda=Path(sys.prefix)/'conda-meta';files=set()
for pattern in ['python-*.json','pytorch-*.json','cuda-runtime-*.json','cuda-cudart-*.json','libcublas-*.json','libcudnn-*.json','cudnn-*.json']:files.update(conda.glob(pattern))
result['conda_package_records']=[identity(p)for p in sorted(files)]
result['torch_python_files']=[identity(Path(sys.prefix)/'lib/python3.10/site-packages/torch'/name)for name in ['__init__.py','version.py']]
result['finished_unix']=time.time();result['torch_imported']=('torch'in sys.modules);assert not result['torch_imported']
fail=[x for group in ['native_libraries','model_weight_shards','model_small_files']for x in result[group]if x['status'].startswith('FAIL')or x.get('historical_or_prior_binding_match')is False]
if fail:result['status']='FAIL_CURRENT_EXPECTED_IDENTITY_COLLECTION';result['failures']=fail
print(json.dumps(result,sort_keys=True))
