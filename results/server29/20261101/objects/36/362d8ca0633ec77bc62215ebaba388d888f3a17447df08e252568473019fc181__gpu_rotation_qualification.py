"""GPU arithmetic/parity and native FP rotation controls, no AIME inputs."""
import sys,gc,hashlib,contextlib,math
from pathlib import Path
import numpy as np
import torch
import execution_control as C
import group_codecs as B
import rotation_codec as K
from canonical_h128 import normalize_qk,get_state,QwenKeyHadamardPatch
from rotation_adapter import RotationPatch
def numerical_diagnostics(engine):
 """Paired codec perturbations of real heldout FP states/readout queries, no AIME."""
 R=engine.root;T=engine.torch;ids=C.read(R/'calibration/doc_05_tokens.json')[:128];A={}
 for row in C.read(R/'CANONICAL_UH_MATRICES_AND_SHA.json')['layers']:
  with np.load(R/row['path']) as f:A[row['layer']]=T.tensor(f['A_fp32_unique'][f['recurrent_to_unique']].copy(),device='cuda:0')
 H=__import__('canonical_h128').hadamard(device='cuda:0');current=[None];queries={};handles=[];saved=[];stats={c:[] for c in ('P0','P1','C1','S3')}
 layers=engine.model.model.layers if hasattr(engine.model.model,'layers') else engine.model.model.language_model.layers
 for i in engine.layers:handles.append(layers[i].linear_attn.register_forward_pre_hook(lambda m,a,i=i:current.__setitem__(0,i)))
 for name in ('torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule'):
  fn=getattr(engine.native,name);saved.append((name,fn))
  def factory(fn):
   def call(q,k,v,*args,**kw):
    C.require(kw.get('use_qk_l2norm_in_kernel') is True,'Heldout native Query path differs');queries[current[0]]=normalize_qk(q).float()/math.sqrt(128);return fn(q,k,v,*args,**kw)
   return call
  setattr(engine.native,name,factory(fn))
 cache=None
 try:
  with T.inference_mode():
   for t,token in enumerate(ids):
    out,cache=engine.forward(token,cache,None,False,False)
    if t not in (0,15,31,63,127):continue
    for i in engine.layers:
     native=get_state(cache,i);qn=queries[i]
     for c in stats:
      m=H if c in ('P0','P1') else A[i];s=T.matmul(m.transpose(-1,-2),native);q=T.matmul(qn.transpose(1,2),m.unsqueeze(0)).transpose(1,2);e=K.encode(s,c);d=K.decode(e);delta=d-s
      before=(s*q[0,0][None,:,:,None]).sum(-2);after=(d*q[0,0][None,:,:,None]).sum(-2)
      x=s.index_select(-2,torch.tensor(K.P,device=s.device)) if c in ('P1','S3') else s;y=x.reshape(1,32,4,32,128);amax=y.abs().amax(-2);levels=y/e.scales.unsqueeze(-2);rounds=T.round(levels)
      stats[c].append(dict(token=t,layer=i,state_MSE=float((delta.double()**2).mean()),state_relative_frobenius=float(T.linalg.vector_norm(delta)/T.linalg.vector_norm(s).clamp_min(1e-30)),actual_query_readout_MSE=float(((after-before).double()**2).mean()),readout_relative_frobenius=float(T.linalg.vector_norm(after-before)/T.linalg.vector_norm(before).clamp_min(1e-30)),group_amax_mean=float(amax.mean()),group_amax_max=float(amax.max()),group_amax_quantiles=T.quantile(amax.flatten(),T.tensor([0.,.5,.9,.99,1.],device=s.device)).cpu().tolist(),scale_min=float(e.scales.min()),scale_max=float(e.scales.max()),scale_quantiles=T.quantile(e.scales.flatten(),T.tensor([0.,.5,.9,.99,1.],device=s.device)).cpu().tolist(),clipped_count=int(((rounds < -127)|(rounds>127)).sum()),rounded_changed_count=int((rounds!=levels).sum()),codes_at_limit=int((e.codes.abs()==127).sum()),outlier_groups_above_half_global_amax=int((amax>.5*amax.max(dim=-2,keepdim=True).values).sum()),total_groups=32*4*128))
    C.progress(engine.target,'HELDOUT_NUMERICAL_DIAGNOSTICS_COMMITTED',tokens=t+1)
 finally:
  for name,fn in saved:setattr(engine.native,name,fn)
  for h in handles:h.remove()
  del cache,out;engine.last_cache=None;A.clear();gc.collect();T.cuda.empty_cache()
 return dict(status='PASS_NONAIME_REAL_STATE_QUERY_PAIRED_CODEC_DIAGNOSTICS',tokens=128,positions=[0,15,31,63,127],cases=stats,scope='Same heldout actual unquantized native-FP state and actual native readout Query transformed to H or UH. Immediate paired codec perturbation readout error, not a free-generation or future-Query oracle, not an AIME admission metric.',PBR_changes_only_Key_group_membership=True,accuracy_inference=False)
def codec_math(target):
 h_cached=__import__('canonical_h128').hadamard(device='cuda:0');C.require(B.tensor_bytes(h_cached)==B.tensor_bytes(__import__('canonical_h128').hadamard(device='cuda:0')),'Cached H must retain exact original GPU generator bytes')
 rng=np.random.default_rng(20261022);fixtures=[np.zeros((1,2,128,128),np.float32),np.full((1,2,128,128),-0.,np.float32),rng.normal(size=(1,2,128,128)).astype(np.float32),rng.uniform(-1e-15,1e-15,(1,2,128,128)).astype(np.float32)];checks=[];p=torch.tensor(K.P,device='cuda:0')
 for i,x in enumerate(fixtures):
  gpu=torch.from_numpy(x).cuda()
  for cond in ('P1','C1','S3'):
   e=K.encode(gpu,cond);d=K.decode(e);perm=cond in ('P1','S3');xp=x[...,list(K.P),:] if perm else x
   cpu=B.encode(torch.from_numpy(xp.copy()),'B1',cpu_semantics='CUDA_LEGACY_EMULATED');expected=B.decode(cpu)
   if perm:expected=expected.index_select(-2,torch.tensor(K.P))
   for n,a,b in [('codes',e.codes,cpu.codes),('scale',e.scales,cpu.scales),('negativezero',e.negative_zero_flags,cpu.negative_zero_flags),('decoded',d,expected)]:
    if B.tensor_bytes(a)!=B.tensor_bytes(b):
     np.savez(Path(target)/'FIRST_CODEC_DIFFERENCE.npz',input=x,actual=a.detach().cpu().numpy(),expected=b.numpy());raise RuntimeError('GPU frozen-field bytes differ:'+str((i,cond,n)))
   K.require_consumed(d,e.cpu());checks.append(dict(fixture=i,condition=cond,status='PASS_EXACT_FIELDS',scale_min=float(e.scales.min()),scale_max=float(e.scales.max())))
  direct=B.decode(B.encode(gpu,'B0'));perm=B.decode(B.encode(gpu.index_select(-2,p),'B0')).index_select(-2,p);C.require(B.tensor_bytes(direct)==B.tensor_bytes(perm),'GPU Group128 signed-zero permutation invariance failed')
 return dict(status='PASS_UNCHANGED_GPU_SYM32_PBR_EXACT_FIELDS',checks=checks,base_codec_sha256=C.sha(C.R/'inherited/group32/src/group_codecs.py'),PBR_key_groups=4,group_size=32,negative_zero=True,CPU_reference='Already-qualified canonical CUDA reciprocal semantics; new PBR independently CPU tested',no_new_codec_or_scale=True)
def fp_equivalence(engine):
 R=engine.root;T=engine.torch;layers=engine.layers
 ids=C.read(R/'calibration/doc_05_tokens.json')[:128]
 # Native baseline tensors are temporary FP-control references, never production continuation state.
 baseline={};noise_H={};noise_A={};snap_A={};fp64_state_errors=[];fp64_read_errors=[]
 def metric(a,b):
  a=a.double();b=b.double();return dict(relative=float(T.linalg.vector_norm(a-b)/T.linalg.vector_norm(b).clamp_min(1e-30)),maxabs=float((a-b).abs().max()),reference_max=float(b.abs().max()))
 for cond in ('NATIVE','P0','C1','S3'):
  cache=None;step=[0];current=[None];handles=[];saved=[];captured={}
  model_layers=engine.model.model.layers if hasattr(engine.model.model,'layers') else engine.model.model.language_model.layers
  for i in layers:handles.append(model_layers[i].linear_attn.register_forward_pre_hook(lambda m,a,i=i:current.__setitem__(0,i)))
  for i,l in enumerate(model_layers):
   handles.append(l.register_forward_hook(lambda m,a,o,i=i:captured.__setitem__('layerout/'+str(i),(o if isinstance(o,T.Tensor) else o[0]).detach().cpu())))
  patch=contextlib.nullcontext() if cond=='NATIVE' else QwenKeyHadamardPatch(engine.model) if cond=='P0' else RotationPatch(engine.model,cond)
  with T.inference_mode(),patch:
   for name in ('torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule'):
    fn=getattr(engine.native,name);saved.append((name,fn))
    def factory(fn):
     def wrapped(q,k,v,*args,**kw):
      i=current[0];qn=normalize_qk(q) if kw.get('use_qk_l2norm_in_kernel') else q;kn=normalize_qk(k) if kw.get('use_qk_l2norm_in_kernel') else k
      # Outer wrapper sees native q/k for transformed patch; effective q/k obtained using the identical patch transformation.
      if cond!='NATIVE':qn=patch._transform(q,bool(kw.get('use_qk_l2norm_in_kernel')));kn=patch._transform(k,bool(kw.get('use_qk_l2norm_in_kernel')))
      qa=qn.float()[0,0].double().cpu()/math.sqrt(128);ka=kn.float()[0,0].double().cpu();vv=v.float()[0,0].double().cpu();initial=kw.get('initial_state',args[2] if len(args)>2 else None)
      ss=T.zeros((32,128,128),dtype=T.float64) if initial is None else initial.detach().cpu()[0].double();alpha=kw['g'].float()[0,0].exp().double().cpu();beta=kw['beta'][0,0].double().cpu()
      ss=ss*alpha[:,None,None];delta=(vv-(ss*ka[:,:,None]).sum(-2))*beta[:,None];reference=ss+ka[:,:,None]*delta[:,None,:];read=(reference*qa[:,:,None]).sum(-2)
      core,state=fn(q,k,v,*args,**kw);actual=state.detach().cpu()[0];em=metric(actual,reference);perhead_state=(T.linalg.vector_norm(actual.double()-reference,dim=(-2,-1))/T.linalg.vector_norm(reference,dim=(-2,-1)).clamp_min(1e-30));em['max_per_head_relative']=float(perhead_state.max());fp64_state_errors.append(em['max_per_head_relative']);
      if em['max_per_head_relative']>2e-4:
       engine.diagnostic('FIRST_FP64_STATE_DIFFERENCE.json',dict(actual=actual,expected_FP64=reference,query=qa,key=ka,alpha=alpha,beta=beta));C.atomic_new(engine.target/'FP_ROTATION_FIRST_FAILURE.json',dict(condition=cond,token=step[0],layer=i,scope='independent_FP64_state_update',metrics=em))
       raise RuntimeError('Actual GDN state vs independent FP64 formula exceeds frozen tolerance')
      # Native output has BF16 rounding; apply that actual output dtype before comparing.
      expected_core=read.to(core.dtype);rm=metric(core.detach().cpu()[0,0],expected_core);perhead_read=(T.linalg.vector_norm(core.detach().cpu()[0,0].double()-expected_core.double(),dim=-1)/T.linalg.vector_norm(expected_core.double(),dim=-1).clamp_min(1e-30));rm['max_per_head_relative']=float(perhead_read.max());fp64_read_errors.append(rm['max_per_head_relative'])
      bound=2e-4 if core.dtype==T.float32 else 2*T.finfo(core.dtype).eps
      
      if rm['max_per_head_relative']>bound:
       engine.diagnostic('FIRST_FP64_READOUT_DIFFERENCE.json',dict(actual=core.detach().cpu(),expected_FP64=read,expected_rounded=expected_core));C.atomic_new(engine.target/'FP_ROTATION_FIRST_FAILURE.json',dict(condition=cond,token=step[0],layer=i,scope='independent_FP64_readout',metrics=rm,bound=bound))
       raise RuntimeError('Actual core readout vs independently rounded FP64 reference failed')
      captured['q/'+str(i)]=qa;captured['k/'+str(i)]=ka;captured['core/'+str(i)]=core.detach().cpu();captured['g/'+str(i)]=kw['g'].detach().cpu();captured['beta/'+str(i)]=kw['beta'].detach().cpu()
      return core,state
     return wrapped
    setattr(engine.native,name,factory(fn))
   try:
    for t,token in enumerate(ids):
     step[0]=t;captured.clear();out,cache=engine.forward(token,cache,None,False,False)
     fields={'logits':out.logits.detach().cpu(),**{key:value for key,value in captured.items() if key.startswith(('layerout/','g/','beta/'))}}
     for i in layers:
      state=get_state(cache,i).detach().cpu();q=captured['q/'+str(i)];k=captured['k/'+str(i)]
      if cond=='P0':matrix=__import__('canonical_h128').hadamard(dtype=T.float64)
      elif cond in ('C1','S3'):matrix=patch.matrices[i].detach().cpu().double()
      else:matrix=None
      fields['state/'+str(i)]=state if matrix is None else T.matmul(matrix,state.double())
      fields['q/'+str(i)]=q if matrix is None else T.matmul(q[:,None,:],matrix.transpose(-1,-2)).squeeze(-2)
      fields['k/'+str(i)]=k if matrix is None else T.matmul(k[:,None,:],matrix.transpose(-1,-2)).squeeze(-2)
      fields['readout/'+str(i)]=captured['core/'+str(i)]
     for key,value in fields.items():
      ident=(t,key)
      if cond=='NATIVE':baseline[ident]=value.clone()
      elif cond=='P0':noise_H[ident]=metric(value,baseline[ident])
      elif cond=='C1':
       m=metric(value,baseline[ident]);noise_A[ident]=m;h=noise_H[ident]
       if not (m['relative']<=max(4*h['relative'],.02) and m['maxabs']<=max(4*h['maxabs'],.02*m['reference_max'],1e-12)):
        engine.diagnostic('FIRST_FULL_MODEL_FP_DIFFERENCE.json',dict(actual=value,expected_native=baseline[ident]));C.atomic_new(engine.target/'FP_ROTATION_FIRST_FAILURE.json',dict(condition=cond,token=t,field=key,scope='native_H_A_functional',actual_metrics=m,original_H_noise=h));raise RuntimeError('FP full-model/State/Query/Key/readout rotation equivalence failed:'+str(ident))
       snap_A[ident]=hashlib.sha256(B.tensor_bytes(value)).hexdigest()
      else:
       if snap_A[ident]!=hashlib.sha256(B.tensor_bytes(value)).hexdigest():
        engine.diagnostic('FIRST_C1_S3_FP_DIFFERENCE.json',dict(actual_S3=value));C.atomic_new(engine.target/'FP_ROTATION_FIRST_FAILURE.json',dict(condition=cond,token=t,field=key,scope='C1_S3_FP_same_function_same_matrix',expected_C1_sha256=snap_A[ident],actual_S3_sha256=hashlib.sha256(B.tensor_bytes(value)).hexdigest()));raise RuntimeError('C1/S3 no-QDQ FP recursion values/bytes differ')
     C.progress(engine.target,'FP_ROTATION_EQUIVALENCE_TOKEN',condition=cond,tokens=t+1)
   finally:
    for name,fn in saved:setattr(engine.native,name,fn)
  for h in handles:h.remove()
  del cache,out;engine.last_cache=None;gc.collect();T.cuda.empty_cache()
 return dict(status='PASS_ACTUAL_9B_NATIVE_H_A_FP_AND_C1_S3_IDENTICAL',tokens_per_condition=128,conditions=['NATIVE','H','C1','S3'],all24_layer_state_query_key_readout=True,full_model_logits=True,FP64_state_checks=len(fp64_state_errors),worst_actual_state_vs_FP64=max(fp64_state_errors),worst_actual_readout_vs_rounded_FP64=max(fp64_read_errors),C1_S3_actual_FP_values_and_bytes_identical=True,noise_summary={name:dict(max_relative=max(m['relative'] for m in d.values()),max_abs=max(m['maxabs'] for m in d.values())) for name,d in [('H_native',noise_H),('A_native',noise_A)]},continuous_token_error_trajectories={name:[dict(token=t,groups={prefix:dict(max_relative=max(v['relative'] for (tt,key),v in metrics.items() if tt==t and key.startswith(prefix)),max_abs=max(v['maxabs'] for (tt,key),v in metrics.items() if tt==t and key.startswith(prefix))) for prefix in ['state/','q/','k/','readout/','g/','beta/','layerout/','logits']}) for t in range(128)] for name,metrics in [('H_vs_native',noise_H),('UH_vs_native',noise_A)]},frozen_tolerances='max(4*measured_original_H_noise,.02relative); state local FP64 2e-4; actual BF16 core includes <=2ULP bound',reference_lifetime='Temporary CPU FP-control values discarded on process exit, never quantized continuation')
