import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json
p=Path('/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch')
rows=[]
for n in ['version.py','include/ATen/native/cuda/Copy.cuh','include/ATen/native/cuda/Loops.cuh','include/ATen/native/cuda/KernelUtils.cuh','include/ATen/native/cuda/Reduce.cuh','include/ATen/native/cuda/MemoryAccess.cuh','include/ATen/native/TensorIterator.h']:
 q=p/n;rows.append(dict(path=str(q),exists=q.is_file(),bytes=q.stat().st_size if q.is_file() else None))
print(json.dumps(rows))