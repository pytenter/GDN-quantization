"""Actual CUDA field checks and negative controls, after task permit admission."""
import sys,importlib.util,json
from pathlib import Path
import execution_control as C

def run_micro(root,target):
 import torch,numpy as np
 from group_codecs import encode,decode,tensor_bytes
 from canonical_h128 import hadamard
 torch.set_num_threads(24);torch.set_num_interop_threads(24)
 def load(name,path):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
 old=load('sealed_original_group32_micro',root/'history/group32/inherited/group32/src/group_codecs.py')
 oldh=load('sealed_original_h128_micro',root/'history/group32/inherited/group32/src/canonical_h128.py')
 h=hadamard(128,device='cuda:0');C.require(tensor_bytes(h)==tensor_bytes(oldh.hadamard(128,device='cuda:0')) and hadamard(128,device='cuda:0')is h,'Cached H128 actual GPU bytes differ')
 checks=[]
 for a in C.read(root/'GPU_MICRO_INPUT_MANIFEST.json')['files']:
  p=root/a['path'];C.require(C.sha(p)==a['sha256'],'Frozen micro fixture changed')
  with np.load(p,allow_pickle=False)as f:
   x=torch.from_numpy(f['input'].copy()).to('cuda:0')
   for method in ('B1','B2'):
    e=encode(x,method,group_size=64)
    for field in ['codes','scales','zero_points','negative_zero_flags','decoded']:
     actual=decode(e)if field=='decoded'else getattr(e,field)
     if actual is None:C.require(method+'_'+field not in f,'Unexpected missing field');continue
     ref=f[method+'_'+field];C.require(tuple(actual.shape)==ref.shape and actual.detach().cpu().numpy().dtype==ref.dtype and tensor_bytes(actual)==ref.tobytes(),'CUDA64 independent field mismatch:'+a['path']+':'+method+':'+field)
    original=old.encode(x,method);param=encode(x,method,group_size=32)
    for field in ['codes','scales','zero_points','negative_zero_flags','decoded']:
     v=old.decode(original)if field=='decoded'else getattr(original,field);w=decode(param)if field=='decoded'else getattr(param,field)
     C.require((v is None and w is None)or(v is not None and w is not None and v.dtype==w.dtype and v.shape==w.shape and tensor_bytes(v)==tensor_bytes(w)),'Negative A CUDA32 original mismatch:'+field)
    C.require(e.scales.shape[-2:]==(2,128),'Group64 actual Key layout differs')
    checks.append(dict(fixture=a['path'],method=method,independent64_fields='PASS',original32_fields='PASS'))
  C.progress(target,'GPU_MICRO_FIXTURE_PASS',fixture=a['path'],count=len(checks))
 torch.cuda.synchronize()
 return dict(status='PASS_GROUP64_GPU_FIELDS_AND_GROUP32_NEGATIVE',checks=checks,count=len(checks),H128_cached_original_bytes=True,CUDA_visible_device_count=torch.cuda.device_count(),GPU_uuid=C.UUIDS[int(C.read(target/'DISPATCH.json')['gpu'])],scope='Codec fields, no AIME, no 9B numerical trajectory claim')
