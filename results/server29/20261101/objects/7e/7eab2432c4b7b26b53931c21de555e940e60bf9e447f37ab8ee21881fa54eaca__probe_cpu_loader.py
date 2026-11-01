import os,sys,ctypes,json,hashlib
from pathlib import Path
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert torch.cuda.is_initialized() is False
# No CUDA API, allocation or kernel function is called.
lib=ctypes.CDLL(sys.argv[1]);getattr(lib,'canonical_v2_launch')
assert torch.cuda.is_initialized() is False
maps=Path('/proc/self/maps').read_text();libraries=sorted({line.split()[-1] for line in maps.splitlines() if '/' in line and any(key in line for key in ['libcudart.so','libcuda.so','canonical_v2_sm86_sm89_shared_cudart.so'])})
refs=[]
for name in libraries:
 p=Path(name)
 if p.is_file():refs.append(dict(path=name,actual_path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(dict(status='CPU_DLOPEN_PASS_NOT_GPU_QUALIFICATION',CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CUDA_initialized_before=False,CUDA_initialized_after=torch.cuda.is_initialized(),CUDA_dispatches=0,torch_version=torch.__version__,libraries=refs)))
