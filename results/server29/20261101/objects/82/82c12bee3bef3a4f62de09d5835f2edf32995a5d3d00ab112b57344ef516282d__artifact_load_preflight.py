"""Fail closed on host ABI/runtime loading before any model/GPU worker launch."""
import argparse,os,json,hashlib,ctypes,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--expected-cudart-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_LOADER_REQUIRES_EMPTY_CUDA_VISIBILITY'
 lib=Path(a.library);assert hashlib.sha256(lib.read_bytes()).hexdigest()==a.expected_sha256,'ARTIFACT_SHA_MISMATCH'
 import torch
 assert torch.__version__=='2.9.1+cu128' and not torch.cuda.is_initialized(),'PINNED_CPU_RUNTIME_REQUIRED'
 loaded=ctypes.CDLL(str(lib));getattr(loaded,'canonical_v2_launch')
 assert not torch.cuda.is_initialized(),'CPU_LOAD_MUST_NOT_INITIALIZE_TORCH_CUDA'
 mapped=sorted({row.split()[-1] for row in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so.' in row and '/' in row})
 assert len(mapped)==1,'EXACT_SINGLE_CUDA_RUNTIME_REQUIRED'
 runtime=Path(mapped[0]);runtime_sha=hashlib.sha256(runtime.read_bytes()).hexdigest();assert runtime_sha==a.expected_cudart_sha256,'FROZEN_CUDA_RUNTIME_IDENTITY_MISMATCH'
 result=dict(status='CPU_ARTIFACT_ABI_AND_RUNTIME_PASS_NOT_GPU_QUALIFICATION',library=dict(path=str(lib.resolve()),sha256=a.expected_sha256),runtime=dict(path=str(runtime.resolve()),sha256=runtime_sha),torch_version=torch.__version__,CUDA_VISIBLE_DEVICES='',Torch_CUDA_initialized=False,GPU_dispatches=0)
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x') as f:json.dump(result,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
 print(json.dumps(result))
if __name__=='__main__':main()
