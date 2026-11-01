"""CPU stdlib-only immutable official-source acquisition and previous-evidence copy."""
from pathlib import Path
import ast,hashlib,json,urllib.request,time
R=Path(__file__).resolve().parents[1];OLD=R.parent/'QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1';manifest=json.loads((OLD/'SHA256_MANIFEST.json').read_text());indexed={x['path']:x for x in manifest['files']};rows=[]
wanted=['analysis/trace_kernel_delta.json','analysis/SOURCE_CALLPATH_INVENTORY.json','evidence/c12b7c1a89f7ac9066258b6a1b0608bf369436eef7a06fc1f1fa02c9bc73211c/modeling_qwen3_5.py','evidence/b0db6b7bdeddb7bb1f91f3bca500760275e65ee7484dcb766d6d45de5222b82c/hub_kernels.py','evidence/96591ecf7d0191cb52bd056ddd8ebb755b1aa13d8a4f9eb02702b54c3651588f/worker.py','evidence/28f10b0f2a7e75dc606b38f63d526a91e4b220d6af98a28e536dfef4f10ebe8f/core.py','evidence/f3de9f4d2d526a48b5056ae4806bfc5826e26521de73e12f98243f1170eeddd8/Context.h','evidence/bfcd5c3038213a8334547251b3cbe2a19cd0ec3ad570992202fde0063a50c463/TensorFactories.h','evidence/3b905b71215f9c459031fb01c10d8f800bc77442563c66a195041a11e1f482e9/deterministic.py']
for n in wanted:
 p=OLD/n;raw=p.read_bytes();m=indexed[n];assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256'];dest=R/'evidence/fill_previous'/n;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert dest.read_bytes()==raw
 else:dest.write_bytes(raw)
 rows.append(dict(source=str(p),role='PREVIOUS_MANIFEST_VERIFIED_ASSET',local=str(dest.relative_to(R)),sha256=m['sha256'],bytes=len(raw)))
p=R/'evidence/2b6d0bddf7ef7d8e5a1e93b2200c357008c30d1e135bd980773d7e9073cd5698/version.py';tree=ast.parse(p.read_text());version=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='git_version' for t in n.targets))
# Official immutable PyTorch tree for the exact installed version.git_version.
paths=['aten/src/ATen/native/cuda/TensorFactories.cu','aten/src/ATen/native/TensorFactories.cpp','aten/src/ATen/native/cuda/Copy.cu','aten/src/ATen/native/TensorConversions.cpp','aten/src/ATen/native/TensorShape.cpp','aten/src/ATen/native/cuda/UnaryOpsKernel.cu','aten/src/ATen/native/cuda/RangeFactories.cu','aten/src/ATen/native/cuda/Resize.cu','aten/src/ATen/native/cuda/DepthwiseConv2d.cu','aten/src/ATen/native/transformers/cuda/attention.cu','aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp','aten/src/ATen/native/transformers/cuda/flash_attn/flash_fwd_launch_template.h','aten/src/ATen/native/transformers/cuda/flash_attn/src/flash_fwd_kernel.h','aten/src/ATen/native/transformers/cuda/flash_attn/src/softmax.h','aten/src/ATen/native/cuda/Loops.cuh','aten/src/ATen/native/cuda/MemoryAccess.cuh','aten/src/ATen/native/cuda/FillKernel.cu','aten/src/ATen/native/cuda/Reduce.cuh']
for n in paths:
 url='https://raw.githubusercontent.com/pytorch/pytorch/'+version+'/'+n;dest=R/'evidence/fill_official_pytorch'/n;dest.parent.mkdir(parents=True,exist_ok=True)
 try:
  raw=urllib.request.urlopen(url,timeout=30).read();assert len(raw)<8000000
  if dest.exists():assert dest.read_bytes()==raw
  else:dest.write_bytes(raw)
  rows.append(dict(source=url,role='OFFICIAL_SOURCE_EXACT_VERSION_GIT_NOT_BINARY_PROOF',commit=version,local=str(dest.relative_to(R)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),status='READ_PASS'))
 except Exception as ex:rows.append(dict(source=url,status='UNAVAILABLE',error=str(ex)))
(R/'analysis/fill_source_catalog.json').write_text(json.dumps(dict(installed_torch_git_version=version,source_identity_note='Exact public source commit from installed version.py; no proof the compiled binary is unpatched or that every runtime branch follows this source.',files=rows),indent=2)+'\n');print('collected',len(rows),'unavailable',sum(x.get('status')=='UNAVAILABLE' for x in rows))
