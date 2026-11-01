import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json
base=Path('/data/ydai/miniconda3/envs/bitdecode')
patterns=[('torch',base/'lib/python3.10/site-packages/torch/lib','libtorch_cuda.so'),('torch',base/'lib/python3.10/site-packages/torch/lib','libtorch_cpu.so'),('torch',base/'lib/python3.10/site-packages/torch/lib','libc10_cuda.so'),('cuBLAS',base/'lib/python3.10/site-packages/nvidia/cublas/lib','libcublas*.so*'),('cuDNN',base/'lib/python3.10/site-packages/nvidia/cudnn/lib','libcudnn*.so*'),('CUDA_runtime',base/'lib/python3.10/site-packages/nvidia/cuda_runtime/lib','libcudart*.so*'),('conda_candidate',base/'lib','libcublas*.so*'),('conda_candidate',base/'lib','libcudnn*.so*')]
out=[]
for role,directory,pattern in patterns:
 matches=sorted(directory.glob(pattern))
 if not matches:out.append(dict(role=role,directory=str(directory),pattern=pattern,status='MISSING'))
 for p in matches:
  if p.is_file():out.append(dict(role=role,path=str(p),resolved_path=str(p.resolve()),bytes=p.stat().st_size,symlink=p.is_symlink(),status='PRESENT'))
print(json.dumps(out))