"""Read-only remote package bytes -> local metadata only; no GPU imports."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[1]
transport_path=ROOT.parent/'QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/operations/remote.py'
spec=importlib.util.spec_from_file_location('readonly_metadata_transport',transport_path)
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
CODE=r'''
from pathlib import Path
import os,json,sys,hashlib,ast,importlib.metadata,shutil,subprocess
b=Path('/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/triton')
rows=[];env_names=set();total=0
skip={'__pycache__','.git','.cache','cache','caches'}
for dirname,dirs,files in os.walk(b,followlinks=False):
 dirs[:]=sorted(d for d in dirs if d not in skip)
 for name in sorted(files):
  p=Path(dirname)/name
  if p.suffix in {'.pyc','.pyo'}:continue
  if not p.is_file():raise RuntimeError('Nonregular dependency:'+str(p))
  pre=p.stat();h=hashlib.sha256()
  with p.open('rb') as f:
   while True:
    chunk=f.read(8*1024*1024)
    if not chunk:break
    h.update(chunk)
  post=p.stat()
  keys=('st_dev','st_ino','st_size','st_mtime_ns')
  if any(getattr(pre,k)!=getattr(post,k) for k in keys):raise RuntimeError('Dependency changed during SHA:'+str(p))
  total+=pre.st_size
  if total>1024**3 or len(rows)>=4000:raise RuntimeError('Bounded Triton dependency inventory exceeded')
  kind='python_source' if p.suffix=='.py' else 'shared_object' if '.so' in p.name else 'libdevice_bitcode' if p.suffix=='.bc' else 'executable_binary' if pre.st_mode&0o111 else 'compiler_header_or_package_resource'
  rows.append(dict(path=str(p),relative_path=str(p.relative_to(b)),bytes=pre.st_size,sha256=h.hexdigest(),inode=pre.st_ino,device=pre.st_dev,mtime_ns=pre.st_mtime_ns,mode=oct(pre.st_mode),kind=kind,symlink=p.is_symlink(),resolved_path=str(p.resolve())))
  if p.suffix=='.py':
   tree=ast.parse(p.read_text())
   for node in ast.walk(tree):
    if isinstance(node,ast.Call) and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
     fn=node.func
     if isinstance(fn,ast.Attribute) and fn.attr in ('getenv','get') and isinstance(fn.value,(ast.Name,ast.Attribute)):
      if isinstance(fn.value,ast.Name) and fn.value.id=='os' or isinstance(fn.value,ast.Attribute) and fn.value.attr=='environ':env_names.add(node.args[0].value)
env_names.update({'TRITON_PTXAS_PATH','TRITON_NVDISASM_PATH','TRITON_CUOBJDUMP_PATH'})
dist=importlib.metadata.distribution('triton');meta=[]
for f in dist.files:
 if '.dist-info/' in str(f) and Path(str(f)).name in {'METADATA','WHEEL','RECORD','entry_points.txt','top_level.txt','INSTALLER','REQUESTED'}:
  p=Path(dist.locate_file(f));d=p.read_bytes();s=p.stat();meta.append(dict(path=str(p),bytes=len(d),sha256=hashlib.sha256(d).hexdigest(),inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns))
selected={k:os.environ.get(k) for k in sorted(env_names)}
host_compiler=[]
cc=os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')
if not cc:raise RuntimeError('Actual Triton host C compiler absent')
if not Path(cc).is_absolute():cc=shutil.which(cc)
compiler_paths=[('driver',cc)]
for name in ('cc1','as','collect2','ld'):
 candidate=subprocess.check_output([cc,'-print-prog-name='+name],text=True).strip()
 path=candidate if Path(candidate).is_absolute() else shutil.which(candidate)
 if path and Path(path).is_file():compiler_paths.append((name,path))
for role,path in compiler_paths:
 p=Path(path);pre=p.stat();h=hashlib.sha256(p.read_bytes()).hexdigest();post=p.stat()
 if (pre.st_ino,pre.st_size,pre.st_mtime_ns)!=(post.st_ino,post.st_size,post.st_mtime_ns):raise RuntimeError('Host compiler changed')
 host_compiler.append(dict(role=role,path=str(p),resolved_path=str(p.resolve()),bytes=pre.st_size,sha256=h,inode=pre.st_ino,device=pre.st_dev,mtime_ns=pre.st_mtime_ns))
compiler=[x for x in rows if x['relative_path'].endswith(('compiler.py','libdevice.py','libdevice.10.bc','/ptxas')) or '/include/' in x['relative_path']]
print(json.dumps(dict(status='PASS_FRESH_CPU_READONLY_TRITON_PACKAGE_SHA',package_root=str(b),distribution_version=dist.version,distribution_metadata=meta,count=len(rows),total_bytes=total,files=rows,compiler_required_paths=[x['path'] for x in compiler],observed_package_environment=selected,host_c_compiler=host_compiler,all_environment_names_found_in_package=sorted(env_names),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),no_torch_import='torch' not in sys.modules,no_triton_import='triton' not in sys.modules,gpu_calls=0,model_calls=0,scope='Fresh all regular Triton package files including Python sources/shared objects/ptxas/libdevice bitcode/compiler headers, plus selected host compiler driver/cc1/as/collect2/ld; excludes __pycache/cache directories and pyc only; metadata-only local receipt, no package files downloaded; not a full distro/toolchain source closure')))
'''
if __name__=='__main__':
    out=transport.run(CODE,role='55',python='/data/ydai/miniconda3/envs/bitdecode/bin/python',timeout=55)
    p=ROOT/'analysis/TRITON_RUNTIME_FRESH_SHA.json'
    p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('status','distribution_version','count','total_bytes',
        'observed_package_environment','no_torch_import','no_triton_import')},ensure_ascii=False))
    print('manifest_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
