#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict 32K Reference binding plus unchanged Canonical AIME sampler/auditors."""
import ast,collections,contextlib,gc,hashlib,importlib.util,inspect,json,math,os,random,sys,time,traceback,copy
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
TASK='QWEN_GDN_HADAMARD_DUAL_M_AIME26_FORMAL_60_V1';METHOD='HADAMARD_H128_DUAL_M_INT8'
REF=Path('/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1')
AUDIT=Path('/data/zypan/analyses/qwen_gdn_aime26_stepquant6_vs_h_dualm_2seed_v1/repair_20261017_v1/source/shared_runtime_worker.py')
ARCH=Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
LAYERS=(0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30)
def check(ok,why):
 if not bool(ok):raise RuntimeError(why)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def canonical_sha(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic_new(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with t.open('x') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 try:os.link(t,p)
 finally:t.unlink()
def atomic_status(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def load_module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def nodes(p,names):
 arr=[];found=set()
 for n in ast.parse(Path(p).read_bytes()).body:
  ass={t.id for t in n.targets if isinstance(t,ast.Name)} if isinstance(n,ast.Assign) else set()
  if getattr(n,'name',None) in names or ass&names:arr.append(n);found|={getattr(n,'name',None)}|ass
 check(names<=found,'Missing original AST '+str(names-found));return arr
def extract(p,names,ns):exec(compile(ast.Module(body=nodes(p,names),type_ignores=[]),str(p),'exec'),ns)
def verify(root):
 m=read(root/'SOURCE_SHA256_MANIFEST.json')
 for x in m['execution_sources']+m['external_dependencies']:
  p=Path(x['path']);p=p if p.is_absolute() else root/p
  check(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Source identity changed '+str(p))
 model=read(root/'MODEL_AND_CODEC_IDENTITY.json')
 for x in model['files']:
  st=Path(x['path']).stat();check(st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns'] and st.st_ino==x['inode'],'Model/tokenizer postSHA stat changed')
 return m
class HOnlyQDQ:
 """I/O counters only; calls the unchanged original dual() for all arithmetic."""
 def __init__(self,c,model,method,artifact,trace):
  check(method=='H_DUALM','Only authorized arm');self.c=c;self.trace=trace;self.calls=0;self.writebacks=0;self.scale_counts=collections.Counter();self.peak_payload=0;self.payload=None;self.trace_rows=[];self.all_layers_payload_bytes=0;self.max_single_layer_payload_bytes=0
  check({int(m.layer_idx) for m in model.modules() if type(m).__name__=='Qwen3_5GatedDeltaNet'}==set(LAYERS),'24layer mapping changed')
 def step_qdq(self,cache,phase,step):
  self.c.current.update(method=METHOD,phase=phase,step=step)
  detail,stat,timing=self.c.dual(cache,self.c.f,trace=self.trace,metrics=self.trace or step%1024==0,parity=self.trace and step==0)
  for key in ('row_floor_hits','row_ceiling_hits','column_floor_hits','column_ceiling_hits'):self.scale_counts[key]+=stat[key]
  
  if step==0:self.c.prefill_after_QDQ_memory=dict(allocated=self.c.torch.cuda.memory_allocated(),reserved=self.c.torch.cuda.memory_reserved())
  self.calls+=1;self.writebacks+=24;payload=24*(32*128*128+2*32*128*2);self.peak_payload=max(self.peak_payload,payload);self.all_layers_payload_bytes=max(self.all_layers_payload_bytes,payload)
  if self.trace:self.trace_rows.append(dict(step=step,phase=phase,layers=detail,scale_safety=stat))
  return stat,timing
class Components:
 def __init__(self,root,gpu):
  import torch
  self.root=root;self.torch=torch;self.gpu=gpu
  self.K=load_module('aime60_strict_reference_core',REF/'src/core.py');self.C=load_module('aime60_strict_reference_codec',REF/'src/codec.py');K=self.K;K.ROOT=root
  K.save=lambda rel,obj:atomic_new(root/'numeric_failures'/rel,obj)
  # The only configure adapter is the physical visible-GPU admission comparison.
  # All logical-device0, numerical flags, seeds and original load/chunk math stay unchanged.
  fn=copy.deepcopy(nodes(REF/'src/core.py',{'configure'})[0]);fn.name='configure_for_selected_physical_gpu';changed=0
  for n in ast.walk(fn):
   if isinstance(n,ast.Compare) and isinstance(n.left,ast.Call) and isinstance(n.left.func,ast.Attribute) and n.left.func.attr=='get' and n.left.args and isinstance(n.left.args[0],ast.Constant) and n.left.args[0].value=='CUDA_VISIBLE_DEVICES':
    check(len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value=='0','Unexpected device admission AST');n.comparators[0]=ast.Constant(str(gpu));changed+=1
  check(changed==1,'Only1 physical mapping adaptation allowed');ns=dict(K.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(REF/'src/core.py'),'exec'),ns);ns[fn.name]()
  self.model,self.qmod,self.environment=K.load();self.f=K.formal(True);self.tsha=K.tsha;self.current={}
  # Original dual() and original sampler functions compile unchanged from pinned bytes.
  d=dict(K=K,C=self.C,torch=torch,CURRENT=self.current);extract(REF/'src/worker.py',{'dual'},d);self.dual=d['dual']
  common=dict(hashlib=hashlib,json=json);extract(root/'protocol/common.py',{'raw_messages','ids_sha256'},common)
  sam=dict(torch=torch,math=math,os=os,sys=sys,random=random,Path=Path,**common)
  names={'setup_seed','render_input','build_warpers','sample_next','stop_ids','build_top_k_audit','top_k_audit_valid','TEMPERATURE','TOP_P','TOP_K','MIN_P','PRESENCE_PENALTY','REPETITION_PENALTY','MAX_NEW_TOKENS','TOP_K_AUDIT_VERSION','TOP_K_IMPLEMENTATION','LEGACY_TOP_K_AUDIT_COMMIT'}
  extract(root/'protocol/sampler.py',names,sam)
  for n in names|{'ids_sha256'}:setattr(self.f,n,sam[n])
  ag=dict(globals(),torch=torch);extract(AUDIT,{'tensor_version','Consumption','native_kernel_identity'},ag);self.Consumption=ag['Consumption']
  self.recurrent_binding=ag['native_kernel_identity'](self.qmod.torch_recurrent_gated_delta_rule)
  gen=copy.deepcopy(nodes(AUDIT,{'generate'})[0]);removed=0
  for n in ast.walk(gen):
   for field in ('body','orelse','finalbody'):
    arr=getattr(n,field,None)
    if isinstance(arr,list):
     keep=[]
     for z in arr:
      if isinstance(z,ast.Expr) and isinstance(z.value,ast.Call) and ast.unparse(z.value.func)=='torch.cuda.synchronize':removed+=1
      else:keep.append(z)
     setattr(n,field,keep)
  check(removed==2,'Only original timing-only global sync2 sites removed; reference internal synchronization remains')
  for n in ast.walk(gen):
   if isinstance(n,ast.Constant) and n.value=='DEFERRED_FROZEN_V4_UNTIL_ALL120_LEGAL':n.value='DEFERRED_FROZEN_V4_UNTIL_ALL60_LEGAL'
  ag.update(PostForwardQDQ=HOnlyQDQ,TASK=TASK);exec(compile(ast.fix_missing_locations(ast.Module(body=[gen],type_ignores=[])),str(AUDIT),'exec'),ag);self.generate=ag['generate']
  check(torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'Strict flags changed')
  check(all(p.dtype==torch.bfloat16 and p.device.type=='cuda' and p.device.index==0 for p in self.model.parameters()),'Device/dtype/offload changed')
  self.environment.update(deterministic=True,TF32=False,compensated_chunk=True,intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads(),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),physical_GPU=gpu,logical_GPU=0,recurrent_binding=self.recurrent_binding,initial_model_memory=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()),generation_adapter='Unchanged original sample_next/render/warpers/EOS; unchanged mature generate flow except2timing-onlyglobalSync removed and scoring-status text changed; H-onlyI/Ocounters call exactdual.')
def validate_record(row,job,root):
 check(row['status']=='COMPLETE' and row['runtime_error'] is None and not row['nonfinite'],'Incomplete/error result')
 check(row['canonical_job_id']==job['canonical_job_id'] and row['seed']==job['seed'] and row['problem_id']==job['problem_id'],'Job mapping changed')
 check(row['method']==METHOD and row['input_ids_hash']==job['input_ids_hash'] and row['input_tokens']==job['input_tokens'],'Method/prompt changed')
 ts=row['generated_token_ids'];n=len(ts);check(n==row['generated_token_count'] and 0<n<=81920 and all(type(t)==int and 0<=t<248320 for t in ts),'Token payload changed')
 check(row['generated_token_ids_sha256']==canonical_sha(ts),'Generated token hash changed')
 eos=ts[-1] in row['stop_token_ids'];check(row['finish_reason']==('eos' if eos else 'length') and (eos or n==81920),'Illegal termination')
 check(not any(t in row['stop_token_ids'] for t in ts[:-1]),'Tokens after EOS')
 check(row['actual_model_forward_calls']==n and row['hadamard_audit']['calls']==24*n and row['hadamard_audit']['phase_counts']=={'chunk_prefill':24,'recurrent_decode':24*(n-1)},'Full forward/H mapping changed')
 check(row['gpu_uuid']==job['gpu_uuid'] and row['gpu']==job['gpu'],'GPU binding changed')
 check(row['generation_config']==dict(enable_thinking=True,do_sample=True,temperature=1.,top_p=.95,top_k=20,min_p=0.,presence_penalty=1.5,repetition_penalty=1.,presence_penalty_scope='generated_output_tokens_only',max_new_tokens=81920,sampling_seed=job['seed']),'Sampling config changed')
 q=row['quantizer_audit'];check(q['calls']==n and q['prefill_calls']==1 and q['decode_calls']==n-1 and q['layer_writebacks']==24*n and q['next_consumption_checks']==24*(n-1),'QDQ timing/count mismatch')
 check(row['source_manifest_sha256']==sha(root/'SOURCE_SHA256_MANIFEST.json') and row['protocol_sha256']==sha(root/'FROZEN_PROTOCOL.json'),'Frozen source/protocol changed')
 payload=dict(row);saved=payload.pop('output_payload_sha256');check(saved==canonical_sha(payload),'Payload SHA changed')
 return row
