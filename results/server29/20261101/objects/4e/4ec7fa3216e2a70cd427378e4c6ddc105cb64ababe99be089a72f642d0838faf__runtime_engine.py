"""Shared single-token model/QDQ/sampler loop for qualification and formal."""
import sys,os,time,json,gc,contextlib,hashlib
from pathlib import Path
import execution_control as C

class Engine:
 def __init__(self,root,gpu,target):
  import torch
  self.root=root;self.gpu=gpu;self.target=target;self.torch=torch
  sys.path.insert(0,str(root/'inherited/group32/src'))
  from canonical_h128 import load_model_and_tokenizer,setup_seed,GDN_LAYERS,get_state
  from four_codec_adapter import NativeLayerTracker
  from rotation_adapter import RotationQDQ as FourCodecQDQ
  from protocol_binding import sampler
  from group_codecs import tensor_bytes
  self.QDQ=FourCodecQDQ;self.Native=NativeLayerTracker;self.layers=GDN_LAYERS;self.get_state=get_state;self.bytes=tensor_bytes
  C.require(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1','Runtime torch/CUDA differs')
  torch.set_num_threads(24);torch.set_num_interop_threads(24)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
  setup_seed(1);C.progress(target,'LOAD_MODEL');load_start=time.monotonic();self.model,self.tokenizer=load_model_and_tokenizer();model_load_seconds=time.monotonic()-load_start;model_resident_allocated=torch.cuda.memory_allocated();self.f,self.sampler_ast=sampler(root,torch)
  C.require(all(p.dtype==torch.bfloat16 and p.device.type=='cuda' and p.device.index==0 for p in self.model.parameters()),'BF16 singleGPU model/offload differs')
  layers=self.model.model.layers if hasattr(self.model.model,'layers') else self.model.model.language_model.layers
  C.require(len(layers)==32 and {i for i,l in enumerate(layers) if getattr(l,'linear_attn',None) is not None}==set(self.layers),'Full actual32/24 layer mapping differs')
  self.vocab=self.model.get_output_embeddings().weight.shape[0];C.require(self.vocab==248320,'Actual vocabulary changed')
  self.native=__import__('transformers.models.qwen3_5.modeling_qwen3_5',fromlist=['x']);C.require(C.sha(Path(self.native.__file__))=='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a','Actual architecture changed')
  textconfig=self.model.config.text_config if hasattr(self.model.config,'text_config') else self.model.config
  C.require(textconfig.num_attention_heads==16 and textconfig.num_key_value_heads==4 and textconfig.head_dim==256 and textconfig.max_position_embeddings>=82293,'Attention/context shape differs')
  C.require(textconfig._attn_implementation=='sdpa','Actual attention backend differs from bound SDPA')
  C.require(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,6),'Visible GPU count/capability differs')
  self.environment=dict(model_load_seconds=model_load_seconds,model_resident_allocated_bytes=model_resident_allocated,torch=torch.__version__,cuda=torch.version.cuda,capability=list(torch.cuda.get_device_capability(0)),device_name=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],deterministic=torch.are_deterministic_algorithms_enabled(),cuda_matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,threads=[torch.get_num_threads(),torch.get_num_interop_threads()],architecture_sha256=C.sha(Path(self.native.__file__)),attention='sdpa',source_freeze_sha256=C.sha(root/'SOURCE_FREEZE.json'),protocol_sha256=C.sha(root/'AIME26_SEED1_PROTOCOL.json'),model_identity_sha256=C.sha(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json'))
  C.progress(target,'MODEL_LOADED')
 def walk(self,x,name='cache',seen=None):
  seen=set() if seen is None else seen
  if isinstance(x,self.torch.Tensor):yield name,x;return
  if id(x) in seen:return
  seen.add(id(x))
  if isinstance(x,dict):
   for k in sorted(x,key=str):yield from self.walk(x[k],name+'/'+str(k),seen)
  elif isinstance(x,(tuple,list)):
   for i,v in enumerate(x):yield from self.walk(v,name+'/'+str(i),seen)
  elif type(x).__module__.startswith('transformers') and hasattr(x,'__dict__'):
   for k in sorted(vars(x)):yield from self.walk(vars(x)[k],name+'/'+k,seen)
 def snapshot(self,out,cache,qdq):
  fields=[('logits',out.logits)]+list(self.walk(cache))
  if qdq is not None:
   for layer,e in sorted(qdq.previous.items()):
    for key in ['codes','scales','zero_points','negative_zero_flags']:
     x=getattr(e,key)
     if x is not None:fields.append(('encoded/'+str(layer)+'/'+key,x))
  chunks=[];index=[];offset=0;dedup={}
  for name,x in fields:
   b=self.bytes(x);h=hashlib.sha256(b).hexdigest();key=(h,len(b));prior=dedup.get(key)
   if prior is not None and prior[1]==b:where=prior[0]
   else:where=offset;chunks.append(b);dedup[key]=(where,b);offset+=len(b)
   index.append(dict(name=name,dtype=str(x.dtype),shape=list(x.shape),offset=where,bytes=len(b),sha256=h))
  return b''.join(chunks),index
 def diagnostic(self,name,values):
  p=self.target/name
  if p.exists():return
  arrays=[];rows=[];offset=0
  for k,x in values.items():
   if not isinstance(x,self.torch.Tensor):continue
   b=self.bytes(x);arrays.append(b);rows.append(dict(name=k,offset=offset,bytes=len(b),dtype=str(x.dtype),shape=list(x.shape)));offset+=len(b)
  C.require(offset<=1024**3,'First-difference tensor bound exceeded')
  with p.with_suffix('.bin').open('xb') as f:
   for b in arrays:f.write(b)
   f.flush();os.fsync(f.fileno())
  C.atomic_new(p,dict(tensors=rows,bytes=offset,sha256=C.sha(p.with_suffix('.bin'))))
 def callback(self,phase,t,layer,values):
  if phase=='NEXT_CONSUMPTION_FAILURE':self.diagnostic('NEXT_CONSUMPTION_FIRST_FAILURE.json',values)
  if phase=='POST_QDQ':self.writebacks[str(layer)]+=1
 def reset_audit(self):self.writebacks={str(i):0 for i in self.layers}
 def forward(self,token,cache,q,observe_before,observe_after):
  if q is not None:q.observe=observe_before
  expected=1
  if cache is not None:
   existing=[l.keys.shape[-2] for l in cache.layers if isinstance(getattr(l,'keys',None),self.torch.Tensor)]
   C.require(existing and len(set(existing))==1,'Attention cache prior lengths differ');expected=existing[0]+1
  out=self.model(input_ids=self.torch.tensor([[int(token)]],device='cuda:0'),past_key_values=cache,use_cache=True);cache=out.past_key_values
  actual=[l.keys.shape[-2] for l in cache.layers if isinstance(getattr(l,'keys',None),self.torch.Tensor)]
  C.require(actual and all(v==expected for v in actual),'Attention cache length did not advance exactly1')
  self.last_cache=cache
  C.require(bool(self.torch.isfinite(out.logits).all().item()),'Nonfinite logits')
  if q is not None:q.observe=observe_after;q.after_token(cache)
  return out,cache
 def state_fingerprint(self,cache):return {str(i):hashlib.sha256(self.bytes(self.get_state(cache,i))).hexdigest() for i in self.layers}
 def teacher(self,unit,ids,reference=None):
  T=self.torch;N=unit['tokens'];condition=unit.get('condition');self.reset_audit();cache=None;rows=[];fingerprints=[];timing={};start=time.monotonic()
  islong=unit['kind']=='nonAIME_persistent_long';audit_targets=set(range(1,128))|{512,1024,2048,3072,3967} if islong else set(range(1,N))
  q=self.QDQ(self.model,condition,self.layers,observe=True,callback=self.callback) if condition else None
  patch=q if q is not None else self.Native(self.model) if unit.get('repeat')==1 else contextlib.nullcontext()
  sampler_gen=T.Generator(device='cuda:0').manual_seed(1);mask=T.zeros((1,self.vocab),dtype=T.bool,device='cuda:0');warpers=self.f.build_warpers()
  with T.inference_mode(),patch:
   for t in range(N):
    if islong and t in (128,N-128):T.cuda.synchronize();timing['start_'+str(t)]=time.perf_counter()
    out,cache=self.forward(ids[t],cache,q,t in audit_targets,(t+1) in audit_targets)
    if condition is not None:
     current,_=self.f.sample_next(warpers,out.logits[:,-1,:],mask,sampler_gen);mask.scatter_(1,current,True)
    if reference is not None:
     raw,index=self.snapshot(out,cache,q);reference.step(t,raw,index)
    if islong and t in (255,N-1):
     T.cuda.synchronize();begin=128 if t==255 else N-128;elapsed=time.perf_counter()-timing['start_'+str(begin)];timing['short' if t==255 else 'late']=dict(warmup=128 if t==255 else begin,measured=128,seconds=elapsed,TPS=128/elapsed,ms_per_token=elapsed*1000/128,observer=False,sampling_invoked=True,fixed_replay_input=True)
    if islong and (t in (0,31,127,255,511,1023,2047,3071,4095)):fingerprints.append(dict(token=t,states=self.state_fingerprint(cache)))
    if (t+1)%32==0:C.progress(self.target,'TEACHER_TOKEN_COMMITTED',token=t+1,total=N)
   if q is not None:
    a=q.audit();C.require(a['qdq_layer_updates']==N*len(self.layers) and a['next_consumption_checks']==len([x for x in audit_targets if x<N])*len(self.layers),'Teacher QDQ/next-consumption count differs')
    C.require(all(v==N for v in self.writebacks.values()),'Incomplete actual layer writeback callback coverage')
   else:a=dict(unquantized_math_control=True,tokens=N,Native_tracker_only=unit.get('repeat')==1)
  if reference is not None:reference.finish(N)
  result=dict(status='PASS',tokens=N,condition=condition,audit=a,writebacks_by_layer=self.writebacks,actual_next_consumption_positions=sorted(x for x in audit_targets if x<N) if q else [],timing=timing,sparse_fingerprints=fingerprints,seconds=time.monotonic()-start,persistent_carrier='native_FP32_post_QDQ' if q else 'Native_FP_math_control')
  del q,cache,out;self.last_cache=None;gc.collect();T.cuda.empty_cache();return result
 def generate(self,condition,input_ids,cap,item,reference=None):
  T=self.torch;self.f.setup_seed(1);self.reset_audit();generator=T.Generator(device='cuda:0').manual_seed(1);initial_generator=hashlib.sha256(self.bytes(generator.get_state())).hexdigest();warpers=self.f.build_warpers();stops=self.f.stop_ids(self.tokenizer)
  survivor_counts=[];tokens=[];self.active_tokens=tokens;cache=None;mask=T.zeros((1,self.vocab),dtype=T.bool,device='cuda:0');p=len(input_ids);start=time.monotonic();fingerprints=[];audit_targets=set(range(1,128));trace=reference is not None;total=0;chunk=[];prefill_started=time.monotonic()
  q=self.QDQ(self.model,condition,self.layers,observe=True,callback=self.callback)
  def step(token):
   nonlocal cache,total
   before=total>0 and (trace or total in audit_targets or total%2048==0)
   after=trace or total+1 in audit_targets or (total+1)%2048==0
   out,cache=self.forward(token,cache,q,before,after)
   if trace:
    raw,index=self.snapshot(out,cache,q);reference.step(total,raw,index)
   elif total in (0,p-1) or (total+1)%2048==0:fingerprints.append(dict(input_step=total,states=self.state_fingerprint(cache)))
   total+=1
   if total%128==0:C.progress(self.target,'PROMPT_OR_DECODE_FORWARD_COMMITTED',input_forwards=total,generated_tokens=len(tokens),problem_id=item.get('problem_id'),condition=condition)
   return out
  with T.inference_mode(),q:
   for token in input_ids:out=step(token)
   prefill_seconds=time.monotonic()-prefill_started;decode_start=time.monotonic()
   while len(tokens)<cap:
    current,survivors=self.f.sample_next(warpers,out.logits[:,-1,:],mask,generator);mask.scatter_(1,current,True);token=int(current.item());tokens.append(token);chunk.append(token)
    C.require(1<=survivors<=self.vocab,'Original sampler survivors invalid');survivor_counts.append(survivors)
    if len(tokens)%128==0 or token in stops or len(tokens)==cap:
     C.event(self.target/'TOKEN_CHUNKS.jsonl','GENERATED_TOKEN_CHUNK',count=len(tokens),token_ids=chunk);chunk=[]
     C.progress(self.target,'GENERATED_TOKENS_COMMITTED',generated_tokens=len(tokens),input_forwards=total,problem_id=item.get('problem_id'),condition=condition)
    if token in stops or len(tokens)==cap:break
    out=step(token)
   n=len(tokens);audit=q.audit();C.require(total==p+n-1 and audit['qdq_layer_updates']==len(self.layers)*total and all(v==total for v in self.writebacks.values()),'Prompt/decode QDQ timing or actual layer coverage differs')
   C.require(audit['next_consumption_checks']==len(self.layers)*(total-1) if trace else audit['next_consumption_checks']>0,'Next-consumption audit missing')
  expected_checks=sum(t>0 and (trace or t in audit_targets or t%2048==0) for t in range(total))*len(self.layers)
  C.require(audit['next_consumption_checks']==expected_checks,'Sparse/full consumption roster incomplete')
  audit.update(full_actual_values_and_bytes_each_next=trace,sparse_formal_next_checks=not trace)
  if reference is not None:reference.finish(total)
  elapsed=time.monotonic()-start
  result=dict(status='COMPLETE',task_id=C.TASK,condition=condition,problem_id=item.get('problem_id'),seed=1,gpu=self.gpu,gpu_uuid=C.UUIDS[self.gpu],input_ids_sha256=self.f.ids_sha256(input_ids),input_tokens=p,generated_token_ids=tokens,generated_token_ids_sha256=C.objsha(tokens),generated_token_count=n,response=self.tokenizer.decode(tokens,skip_special_tokens=True),finish_reason='eos' if tokens[-1] in stops else 'length',stop_token_ids=sorted(stops),nonfinite=False,runtime_error=None,actual_model_forward_calls=total,quantizer_audit=audit,writebacks_by_layer=self.writebacks,prompt_layer_QDQ=len(self.layers)*p,decode_layer_QDQ=len(self.layers)*(n-1),sampler_calls=n,sampling_audit=dict(top_k_parameter=20,min_survivors=min(survivor_counts),max_survivors=max(survivor_counts),above20_due_to_original_native_tie_semantics=sum(x>20 for x in survivor_counts),original_warper_unchanged=True),initial_generator_sha256=initial_generator,final_generator_sha256=hashlib.sha256(self.bytes(generator.get_state())).hexdigest(),prompt_seconds=prefill_seconds,decode_and_sampling_seconds=time.monotonic()-decode_start,wall_seconds=elapsed,TPS=n/elapsed,sparse_fingerprints=fingerprints,peak_allocated_bytes=T.cuda.max_memory_allocated(),peak_reserved_bytes=T.cuda.max_memory_reserved(),sampler_source_sha256=C.sha(self.root/'protocol/sampler.py'),scorer_source_sha256=C.sha(self.root/'protocol/aime26_scorer_v4.py'),dataset_freeze_sha256=C.sha(self.root/'AIME26_DATASET_FREEZE.json'),environment=self.environment)
  del q,cache,out;self.last_cache=None;gc.collect();T.cuda.empty_cache();return result
 def pressure(self,condition):
  T=self.torch;n=82293;results=[]
  for condition in (condition,):
   self.reset_audit();cache=None
   with T.inference_mode(),self.QDQ(self.model,condition,self.layers,observe=False,callback=self.callback) as q:
    for token in C.read(self.root/'protocol/non_aime_fixture.json')['tokens'][:16]:out,cache=self.forward(token,cache,q,False,False)
    held=[];extra=[];expanded=[];T.cuda.reset_peak_memory_stats()
    try:
     for _ in range(8):held.extend([T.zeros((1,4,n,256),dtype=T.bfloat16,device='cuda:0') for _ in range(2)])
     extra=[T.zeros((1,4,n+1,256),dtype=T.bfloat16,device='cuda:0') for _ in range(2)];expanded=[T.zeros((1,16,n+1,256),dtype=T.bfloat16,device='cuda:0') for _ in range(2)];query=T.zeros((1,16,1,256),dtype=T.bfloat16,device='cuda:0');beg=time.monotonic();attended=T.nn.functional.scaled_dot_product_attention(query,*expanded,dropout_p=0.,is_causal=False);C.require(bool(T.isfinite(attended).all()),'Nonfinite maximum-context SDPA');T.cuda.synchronize();free,total=T.cuda.mem_get_info();C.require(free>=1024**3,'Maximum-context free VRAM below1GiB')
     results.append(dict(condition=condition,total_context=n,actual_short_QDQ_tokens=16,pressure_scope='SYNTHETIC_KV_NATIVE_SDPA_WITH_LIVE_MODEL_NOT81K_GDN_TRAJECTORY',KV_bytes=sum(t.numel()*t.element_size() for t in held),overlap_and_expanded_bytes=sum(t.numel()*t.element_size() for t in extra+expanded),free_bytes=free,peak_allocated=T.cuda.max_memory_allocated(),peak_reserved=T.cuda.max_memory_reserved(),SDPA_wall_seconds=time.monotonic()-beg))
    finally:
     held.clear();extra.clear();expanded.clear();del cache,out;self.last_cache=None;gc.collect();T.cuda.empty_cache()
    C.progress(self.target,'CONTEXT_PRESSURE_CASE_PASS',condition=condition)
  return dict(status='PASS',cases=results,numerical_81K_proof=False)

class ByteReference:
 def __init__(self,root,target,unit):
  self.root=root;self.target=target;self.unit=unit;self.rows=[];self.total_bytes=0;self.refname=unit['reference_unit'];self.path=C.reference_root()/self.refname;self.repeat=unit.get('repeat',0)
  if self.repeat==0:self.path.mkdir(parents=True,exist_ok=False)
  else:self.expected=C.read(root/'execution'/self.refname/'TRACE.json')['tokens']
 def step(self,t,raw,index):
  row=dict(token=t,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),tensors=index)
  p=self.path/('%05d.bin'%t)
  if self.repeat==0:
   s=os.statvfs(self.path);C.require(s.f_bavail*s.f_frsize>=len(raw)+20*1024**3,'Temporary RAM reference margin below20GiB');self.total_bytes+=len(raw);C.require(self.total_bytes<=80*1024**3,'Temporary reference80GiB limit')
   with p.open('xb') as f:f.write(raw)
  else:
   C.require(t<len(self.expected),'Fresh trajectory longer than reference');old=self.expected[t]
   with p.open('rb') as f:expected=f.read()
   if old!=row or expected!=raw:
    for name,b in [('EXPECTED_FIRST_DIFFERENCE.bin',expected),('ACTUAL_FIRST_DIFFERENCE.bin',raw)]:
     C.require(len(b)<=1024**3,'First-difference bound')
     with (self.target/name).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    C.atomic_new(self.target/'FULL_FIRST_DIFFERENCE.json',dict(token=t,expected=old,actual=row));raise RuntimeError('Fresh actual tensor values/bytes differ')
  self.rows.append(row)
 def finish(self,N):
  C.require(len(self.rows)==N,'Incomplete full-tensor trace')
  if self.repeat:C.require(len(self.expected)==N,'Fresh trajectory terminated earlier than reference')
  C.atomic_new(self.target/'TRACE.json',dict(status='PASS_FULL_ACTUAL_TENSOR_BYTES' if self.repeat else 'REFERENCE_COLLECTED',tokens=self.rows,temporary_reference_path=str(self.path),reference_lifetime='Delete only after paired PASS, durable trace/source receipts; reference is engineering comparison data, never continuation state'))
 def cleanup_success(self):
  if self.repeat:
   import shutil
   shutil.rmtree(self.path)
   C.atomic_new(self.target/'TEMP_REFERENCE_RELEASED.json',dict(path=str(self.path),only_new_engineering_reference=True,paired_full_byte_PASS=True))
