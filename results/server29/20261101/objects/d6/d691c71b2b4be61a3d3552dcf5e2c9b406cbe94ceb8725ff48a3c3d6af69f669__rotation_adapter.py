"""H or UH persistent coordinates; the PBR path never changes Query/Key."""
import sys,os
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'inherited/group32/src'))
from canonical_h128 import QwenKeyHadamardPatch,normalize_qk,hadamard,get_state
from persistent_adapter import PersistentQDQ
import rotation_codec as codec
import execution_control as C
class RotationPatch(QwenKeyHadamardPatch):
 def __init__(self,model,condition):
  super().__init__(model);self.condition=condition;self.matrices={}
  if condition in ('C1','S3'):
   import numpy as np
   manifest=C.read(R/'U_EIGENBASIS_MATRICES_AND_SHA.json')
   C.require(manifest['status']=='PASS_SEALED_SHARED_U_AND_A','Shared U not qualified')
   for row in manifest['layers']:
    p=R/row['path'];C.require(C.sha(p)==row['sha256'],'Frozen U/A bytes differ')
    with np.load(p) as f:a=f['A_fp32'].copy()
    C.require(a.shape==(32,128,128),'Per-recurrent-head A matrix shape differs')
    self.matrices[row['layer']]=torch.tensor(a,dtype=torch.float32,device='cuda:0')
 def _transform(self,value,use_norm):
  if self.condition in ('P0','P1'):return super()._transform(value,use_norm)
  value=normalize_qk(value) if use_norm else value
  a=self.matrices[int(self.current_layer)]
  return torch.matmul(value.float().transpose(1,2),a.unsqueeze(0)).transpose(1,2)
 def __exit__(self,*args):
  super().__exit__(*args);self.matrices.clear()
class RotationQDQ(PersistentQDQ):
 def __init__(self,model,condition,layers,observe=True,callback=None):
  if condition not in ('P0','P1','C1','S3'):raise ValueError('Unknown rotation condition')
  super().__init__(model,'B1',layers,observe,callback);self.method=condition;self.condition=condition;self.h=RotationPatch(model,condition)
 def __enter__(self):
  import transformers.models.qwen3_5.modeling_qwen3_5 as Q
  self.h.__enter__()
  for name in ('torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule'):
   original=getattr(Q,name);self.saved.append((Q,name,original))
   def factory(fn):
    def call(query,key,value,*args,**kwargs):
     if query.shape[1]!=1:raise RuntimeError('Only single-token actual prompt/decode QDQ is qualified')
     layer=int(self.h.current_layer);state=kwargs.get('initial_state',args[2] if len(args)>2 else None)
     if self.current_token>0 and self.observe:
      if state is None or layer not in self.previous:raise RuntimeError('Missing previous encoded writeback')
      try:codec.require_consumed(state,self.previous[layer])
      except BaseException:
       if self.callback:self.callback('NEXT_CONSUMPTION_FAILURE',self.current_token,layer,dict(actual=state,expected=codec.decode(self.previous[layer])))
       raise
      self.consumption_calls+=1;self.consumption_layers.add(layer)
      if self.callback:self.callback('KERNEL_INITIAL',self.current_token,layer,dict(actual=state))
     return fn(query,key,value,*args,**kwargs)
    return call
   setattr(Q,name,factory(original))
  return self
 def after_token(self,cache):
  if self.current_token!=self.qdq_calls:raise RuntimeError('QDQ sequence mismatch')
  for layer in self.layers:
   state=get_state(cache,layer)
   if state.dtype!=torch.float32:raise RuntimeError('Persistent carrier must be original FP32')
   if self.callback:self.callback('PRE_QDQ',self.current_token,layer,dict(state=state))
   e=codec.encode(state,self.method);d=codec.decode(e);state.copy_(d)
   if self.observe:
    expected=e.cpu();codec.require_consumed(state,expected);self.previous[layer]=expected
   if self.callback:self.callback('POST_QDQ',self.current_token,layer,dict(codes=e.codes,scales=e.scales,negative_zero_flags=e.negative_zero_flags,decoded=d,writeback=state))
  self.qdq_calls+=1;self.current_token+=1
 def audit(self):
  return dict(super().audit(),condition=self.condition,coordinate='H' if self.condition in ('P0','P1') else 'A=U@H',P_only_inside_QDQ=self.condition in ('P1','S3'),codec_source_sha256=C.sha(R/'inherited/group32/src/group_codecs.py'),shared_U_manifest_sha256=C.sha(R/'U_EIGENBASIS_MATRICES_AND_SHA.json'),group_axis='Key',group_size=32,static_rotation_bytes=sum(x.numel()*x.element_size() for x in self.h.matrices.values()),unquantized_shadow=False)
