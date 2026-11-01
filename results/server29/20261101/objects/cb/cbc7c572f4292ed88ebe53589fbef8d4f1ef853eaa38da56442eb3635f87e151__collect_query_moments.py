"""Once-only frozen non-AIME native-FP Query measurement; never AIME scoring."""
import os,sys,json,time,traceback,hashlib,math,gc
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'))
import execution_control as C
TGT=R/'calibration/execution';TGT.mkdir(exist_ok=True)
permit=C.read(R/'CALIBRATION_PERMIT.json')
C.require(permit['task_id']==C.TASK and permit['source_sha256']==C.verify_sources(R),'Calibration source/permit mismatch')
C.require(C.sha(R/'QUERY_CALIBRATION_DATASET_FREEZE.json')==permit['dataset_sha256'],'Calibration dataset changed')
C.require(os.environ.get('CUDA_VISIBLE_DEVICES')==C.UUIDS[1],'Calibration must use authorized physicalGPU1 only')
C.atomic_new(TGT/'STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic()))
try:
 C.device(1);C.model_guard(R);C.storage_guard(R)
 C.atomic_new(TGT/'DRIVER_IDENTITY.json',C.driver_identity(R,1))
 from calibration_runtime_engine import Engine
 engine=Engine(R,1,TGT);torch=engine.torch
 import numpy as np
 from canonical_h128 import normalize_qk
 libs=C.library_check(R,1);C.require(libs['status']=='PASS','Calibration actual mapped library identity failed');C.atomic_new(TGT/'PREFLIGHT_LIBRARY.json',libs)
 sums={i:np.zeros((32,128,128),dtype=np.float64) for i in engine.layers};counts={i:0 for i in engine.layers};fingerprints=[];handles=[];saved=[];current=[None];per_token=set()
 layers=engine.model.model.layers if hasattr(engine.model.model,'layers') else engine.model.model.language_model.layers
 for i in engine.layers:
  m=layers[i].linear_attn
  C.require(m.num_k_heads==16 and m.num_v_heads==32 and m.head_k_dim==128 and m.head_v_dim==128,'Actual independent/shared head mapping differs')
  handles.append(m.register_forward_pre_hook(lambda m,a,i=i:current.__setitem__(0,i)))
 def wrapper(fn):
  def call(query,key,value,*args,**kwargs):
   i=current[0];C.require(query.shape==(1,1,32,128) and i not in per_token,'Query helper shape/coverage differs');per_token.add(i)
   C.require(kwargs.get('use_qk_l2norm_in_kernel') is True,'Actual Query normalization switch differs')
   # Same native L2 operations and cast/scale order as actual native helper readout.
   q=normalize_qk(query).float()*(1/math.sqrt(128))
   C.require(bool(torch.isfinite(q).all()),'Nonfinite actual Query')
   v=q[0,0].detach().cpu().numpy().astype(np.float64)
   C.require(np.array_equal(v[::2],v[1::2]),'Shared-query adjacent head identity differs')
   sums[i]+=v[:,:,None]*v[:,None,:];counts[i]+=1
   if counts[i] in (1,1024,2048,3072,4096):fingerprints.append(dict(layer=i,sample=counts[i],sha256=hashlib.sha256(v.tobytes()).hexdigest()))
   return fn(query,key,value,*args,**kwargs)
  return call
 for name in ('torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule'):
  original=getattr(engine.native,name);saved.append((name,original));setattr(engine.native,name,wrapper(original))
 dataset=C.read(R/'QUERY_CALIBRATION_DATASET_FREEZE.json');start=time.monotonic()
 try:
  with torch.inference_mode():
   for doc in dataset['documents']:
    if doc['role']!='calibration':continue
    p=R/doc['selected_tokens_path'];C.require(C.sha(p)==doc['selected_tokens_sha256'],'Calibration input bytes changed');ids=C.read(p);cache=None
    for t,token in enumerate(ids):
     per_token.clear();out,cache=engine.forward(token,cache,None,False,False)
     C.require(per_token==set(engine.layers),'Actual24GDN Query coverage missing')
     if (t+1)%32==0:C.progress(TGT,'QUERY_SECOND_MOMENT_COMMITTED',document_id=doc['document_id'],tokens=t+1,counts={str(i):counts[i] for i in engine.layers})
    del cache,out;engine.last_cache=None;gc.collect();torch.cuda.empty_cache()
 finally:
  for name,fn in saved:setattr(engine.native,name,fn)
  for h in handles:h.remove()
 C.require(set(counts.values())=={4096},'Incomplete4096 Query samples per head')
 arrays={str(i):sums[i]/counts[i] for i in engine.layers}
 C.require(all(np.isfinite(a).all() and np.array_equal(a,a.swapaxes(-1,-2)) and np.array_equal(a[::2],a[1::2]) for a in arrays.values()),'Moment finite/symmetry/head sharing failed')
 path=R/'calibration/QUERY_SECOND_MOMENTS_FP64.npz'
 with path.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
 libs=C.library_check(R,1);C.require(libs['status']=='PASS','Calibration postflight library identity failed');C.atomic_new(TGT/'POSTFLIGHT_LIBRARY.json',libs)
 C.atomic_new(R/'QUERY_SECOND_MOMENT_AUDIT.json',dict(status='PASS_ACTUAL_NATIVE_READOUT_QUERY_FP64_SECOND_MOMENT',centered=False,samples_per_head=4096,layers=24,recurrent_heads=32,independent_query_heads=16,query_scale=1/math.sqrt(128),actual_query_head_sharing='adjacent pairs exact equal, no mixing across layer/head',source='Four frozen nonAIME RepoBench contexts, tokens0..1023 each, fresh document caches',data_sha256=C.sha(R/'QUERY_CALIBRATION_DATASET_FREEZE.json'),moments=dict(path=str(path.relative_to(R)),bytes=path.stat().st_size,sha256=C.sha(path)),fingerprints=fingerprints,temporal_samples_iid=False,model_environment=engine.environment,seconds=time.monotonic()-start,identity=C.process_identity(os.getpid()),source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),quantized=False,new_AIME_forwards=0))
 C.progress(TGT,'CALIBRATION_COMPLETE')
except BaseException as exc:
 C.atomic_new(TGT/'FIRST_FAILURE.json',dict(status='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),identity=C.process_identity(os.getpid())));raise
