#!/usr/bin/env python3
"""One frozen, non-AIME fixed-token diagnostic unit. No source/math modification."""
import argparse, collections, contextlib, gc, hashlib, importlib.util, json, math, os
from pathlib import Path
import statistics, subprocess, sys, time, traceback
sys.dont_write_bytecode=True
TASK='QWEN_GDN_HISTORICAL_THROUGHPUT_RECONCILIATION_V1'
UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
OLD=Path('/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26/run_qwen_aime26_formal.py')
CURRENT=Path('/data/zypan/analyses/qwen_gdn_dual_m_equivalent_speedup_v1')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def save(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def percentile(x,q):return sorted(x)[max(0,math.ceil(len(x)*q)-1)]
def settings(torch,model,qmod):
 import transformers
 return dict(python=sys.version.split()[0],torch=torch.__version__,cuda_build=torch.version.cuda,transformers=transformers.__version__,
  intraop_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
  deterministic=torch.are_deterministic_algorithms_enabled(),matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
  cudnn_TF32=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,
  attention_backend=getattr(model.config.get_text_config(),'_attn_implementation',None),
  architecture_source= qmod.__file__,architecture_sha256=sha(qmod.__file__),
  parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
  environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER']})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--condition',required=True)
 ap.add_argument('--repeat',type=int,required=True);ap.add_argument('--profile',action='store_true');a=ap.parse_args();root=a.root
 unit=a.condition+('_PROFILE' if a.profile else '')+'_R'+str(a.repeat);target=root/'execution'/unit
 require(not (target/'STARTED.json').exists(),'Unit already started; no retry')
 save(target/'STARTED.json',dict(task_id=TASK,pid=os.getpid(),unix=time.time(),condition=a.condition,repeat=a.repeat,profile=a.profile))
 try:
  manifest=read(root/'SOURCE_SHA256_MANIFEST.json')
  for row in manifest['execution_sources']:
   p=root/row['path'];require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'New frozen source changed')
  for row in manifest['external_dependencies']:
   p=Path(row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'External frozen dependency changed: '+str(p))
  require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU mapping changed')
  is_old=a.condition.startswith('OLD_');is_h=a.condition.endswith('H_C128')
  import torch
  require(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1','Runtime version changed')
  if is_old:
   sys.path.insert(0,str(OLD.parent));f=load_module('matched_original_aime',OLD)
   require(sha(OLD)=='9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd','Old actual f552 runner identity')
   f.setup_seed(20261015);model,tokenizer=f.load_model_and_tokenizer()
   import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
   K=None;ref=None
  else:
   M=load_module('matched_current_measurement',CURRENT/'source/baseline_worker.py')
   K,C,ref,model,qmod,environment=M.reference(root,read(CURRENT/'REFERENCE_BINDING.json'));f=K.formal(True)
   if a.condition=='DIAG_NEW_DETERMINISM_OFF':torch.use_deterministic_algorithms(False)
   elif a.condition=='DIAG_NEW_INTRAOP24':torch.set_num_threads(24)
  require(all(p.device.type=='cuda' and p.device.index==0 and p.dtype==torch.bfloat16 for p in model.parameters()),'Device/dtype/offload changed')
  live=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
  rows=[x.split(',') for x in live.splitlines() if x.strip()]
  require(any(int(x[0])==os.getpid() and x[1].strip()==UUID for x in rows),'Actual PID/UUID mismatch')
  require(sha(qmod.__file__)=='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a','Architecture source changed')
  save(target/'ENVIRONMENT.json',settings(torch,model,qmod))
  obj=read(root/'fixtures/doc1_tokens.json');ids=obj if isinstance(obj,list) else obj.get('input_ids',obj.get('tokens'))
  require(isinstance(ids,list) and len(ids)==32768,'Document input fixture')
  gpu_ids=torch.tensor([ids[:256]],dtype=torch.long,device='cuda:0')
  layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
  require(len(layers)==32,'Not complete32layerQwen model')
  cache=None;iteration=[];position_tensors=[];cache_rows=[];layer_counts=collections.Counter();core_counts=collections.Counter()
  state_rows={};handles=[];cursor={'t':-1};model_calls=0;prof=None
  for i,layer in enumerate(layers):
   handles.append(layer.register_forward_pre_hook(lambda m,args,i=i:layer_counts.update([(cursor['t'],i)])))
  rotary=[m for m in model.modules() if type(m).__name__=='Qwen3_5TextRotaryEmbedding'];require(len(rotary)==1,'Native text rotary identity')
  handles.append(rotary[0].register_forward_pre_hook(lambda m,args:position_tensors.append(args[1])))
  obs=ref['Observe'](model,qmod,f,False) if not is_old else None
  hctx=f.QwenKeyHadamardPatch(model) if is_h else contextlib.nullcontext()
  torch.manual_seed(20261015);torch.cuda.manual_seed_all(20261015);torch.cuda.reset_peak_memory_stats()
  started=time.monotonic();source_core={};original_helpers={};wrapper_restore={};qdq_count=0;profiler_rows=None
  with hctx as h,torch.inference_mode():
   # Only Python counters and dtype/shape metadata. No added tensor math, scalar reads,
   # state copies, or global per-token synchronization in these common audit wrappers.
   for name,kind in [('torch_chunk_gated_delta_rule','chunk'),('torch_recurrent_gated_delta_rule','recurrent')]:
    fn=getattr(qmod,name);wrapper_restore[name]=fn
    def wrap(fn=fn,kind=kind):
     def invoke(q,k,v,*args,**kwargs):
      core_counts[(cursor['t'],kind)]+=1
      y,state=fn(q,k,v,*args,**kwargs)
      require(state.dtype==torch.float32 and tuple(state.shape)==(1,32,128,128),'Actual GDN state dtype/shape')
      state_rows[kind]=dict(shape=list(state.shape),dtype=str(state.dtype),q_shape=list(q.shape),q_dtype=str(q.dtype))
      return y,state
     return invoke
    setattr(qmod,name,wrap())
   for t in range(256):
    if t==128:
     torch.cuda.synchronize()
     if a.profile:
      prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
      prof.__enter__()
     window_start=time.monotonic()
    cursor['t']=t
    if obs is not None:obs.position=t;obs.finite=[];obs.details={}
    tick=time.monotonic();before=0 if cache is None else cache.get_seq_length()
    scope=torch.profiler.record_function if prof is not None else lambda name:contextlib.nullcontext()
    with scope('MATCH/model_forward'):
     out=model(input_ids=gpu_ids[:,t:t+1],past_key_values=cache,use_cache=True);cache=out.past_key_values;model_calls+=1
    with scope('MATCH/finite_and_QDQ'):
     if is_old:require(bool(torch.isfinite(out.logits).all().item()),'Original old logits finite gate')
     else:K.check(torch.stack(obs.finite).all() & torch.isfinite(out.logits).all(),'Original new state/logits finite gate')
     if is_h and (not is_old or t>0):
      audit={'calls':0,'layers':set(),'scale_shapes':{},'code_min':0,'code_max':0};f.quantize_c128(cache,audit)
      require(audit['calls']==1 and audit['layers']==set(f.GDN_LAYERS),'Original C128 audit');qdq_count+=1
     if obs is not None:
      obs.previous={l:(f.get_state(cache,l).data_ptr(),None) for l in f.GDN_LAYERS} if is_h else {}
      for l in f.GDN_LAYERS:K.check(obs.calls[(t,l)]==1,'Original new per-layer GDN call gate')
    after=cache.get_seq_length();kv=[int(cache.layers[i].keys.shape[-2]) for i in range(32) if model.config.get_text_config().layer_types[i]=='full_attention']
    require(before==t and after==t+1 and kv==[t+1]*8,'Cache did not grow at actual input position')
    cache_rows.append(dict(token_index=t,before=int(before),after=int(after),all8attention_KV_lengths=kv))
    del out
    if t>=128:iteration.append(time.monotonic()-tick)
    if (t+1)%128==0:print(json.dumps(dict(event='PROGRESS',unit=unit,consumed=t+1,elapsed=time.monotonic()-started)),flush=True)
   torch.cuda.synchronize();wall=time.monotonic()-window_start
   if prof is not None:prof.__exit__(None,None,None)
   for name,fn in wrapper_restore.items():setattr(qmod,name,fn)
  require(model_calls==256 and len(layer_counts)==256*32 and all(v==1 for v in layer_counts.values()),'Missing full model/layer calls')
  require(core_counts[(0,'chunk')]==24 and sum(v for (t,k),v in core_counts.items() if k=='chunk')==24,'Chunk repeated during recurrent decode')
  require(sum(v for (t,k),v in core_counts.items() if k=='recurrent')==255*24,'Recurrent core missing')
  positions=[x.detach().cpu().tolist() for x in position_tensors]
  require(len(positions)==256 and all(x==[[[t]],[[t]],[[t]]] for t,x in enumerate(positions)),'Actual consumed rotary positions differ')
  if a.profile:
   profiler_rows=[]
   for e in prof.key_averages():
    if e.cpu_time_total>0:profiler_rows.append(dict(name=e.key,count=e.count,cpu_total_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,device_attributed_us=getattr(e,'device_time_total',0)))
   trace=root/'profiles'/(unit+'.trace.json');trace.parent.mkdir(exist_ok=True);require(not trace.exists(),'Trace already exists');prof.export_chrome_trace(str(trace))
   trace_desc=dict(path=str(trace.relative_to(root)),bytes=trace.stat().st_size,sha256=sha(trace))
  else:trace_desc=None
  result=dict(task_id=TASK,status='PASS',condition=a.condition,repeat=a.repeat,profile=a.profile,scope='MATCHED_NON_AIME_PERFORMANCE_NOT_NUMERICAL_EQUIVALENCE',
   model_forward_calls=model_calls,decoder_layer_calls=sum(layer_counts.values()),GDN_chunk_calls=24,GDN_recurrent_calls=255*24,measured_GDN_recurrent_calls=128*24,
   measured_tokens=128,warmup_tokens=128,prefill_tokens=1,cache_retained_at_measurement_start=128,cache_final_length=256,
   actual_positions=positions,actual_KV_growth=cache_rows,state_metadata=state_rows,QDQ_calls=qdq_count,
   source_manifest_sha256=sha(root/'SOURCE_SHA256_MANIFEST.json'),plan_sha256=sha(root/'GPU_DIAGNOSTIC_PLAN.json'),fixture_sha256=sha(root/'fixtures/doc1_tokens.json'),
   decode_wall_seconds=wall,tokens_per_second=128/wall,wall_ms_per_token=1000*wall/128,
   host_iteration_p50_seconds=statistics.median(iteration),host_iteration_p95_seconds=percentile(iteration,.95),host_iteration_seconds=iteration,
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU_UUID=UUID,
   hadamard_calls=getattr(h,'calls',0),hadamard_phase_counts=getattr(h,'phase_counts',None),
   CPU_averages=profiler_rows,chrome_trace=trace_desc,sampling='NOT_RUN_FIXED_INPUT_REPLAY',new_science='NOT_RUN',
   audit_overhead='Common Python-only metadata/counters and retained small position tensor references; no added tensor arithmetic or per-token GPU sync. Position bytes read after timing. Original NEW Observe and finite/QDQ gates retained; OLD original logits finite and post-decode QDQ retained.')
  save(target/'RESULT.json',result)
  for hdl in handles:hdl.remove()
  if obs is not None:obs.close()
  del cache,model,gpu_ids,position_tensors;gc.collect();torch.cuda.empty_cache()
  print(json.dumps(dict(event='PASS',unit=unit,TPS=128/wall,wall=wall)),flush=True)
 except BaseException:
  failure=dict(task_id=TASK,status='FAIL_CLOSED_STOP',unit=unit,pid=os.getpid(),traceback=traceback.format_exc(),retry=False)
  if not (root/'FIRST_FAILURE.json').exists():save(root/'FIRST_FAILURE.json',failure)
  raise

if __name__=='__main__':main()
