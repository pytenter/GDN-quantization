#!/usr/bin/env python3
"""Frozen future single-unit GPU worker. Importing/preparing it is stdlib-only.

Default denial occurs before torch/model/runtime import. Source math is loaded
from pinned original AST; only the module-level cuBLAS env literal is changed.
"""
import argparse,ast,collections,contextlib,gc,hashlib,importlib.util,json,math,os,random,statistics,subprocess,sys,time,traceback,types
from pathlib import Path
sys.dont_write_bytecode=True
from cublas_compare import SnapshotStore,atomic_new,canonical,read,require,sha_bytes
from cublas_harness import WORKSPACES,build_stage_schedule,fresh_environment
TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'
def process_identity():
 fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split();return dict(pid=os.getpid(),starttick=int(fields[19]))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def bound(row,root=None):
 p=Path(row['path']);p=p if p.is_absolute() else Path(root)/p
 require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Source/bytes changed:'+str(p));return p

def transform_core(source,workspace):
 require(workspace in WORKSPACES.values(),'Unregistered workspace');tree=ast.parse(source);changed=[]
 for node in tree.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1:
   t=node.targets[0]
   if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Attribute) and isinstance(t.value.value,ast.Name) and t.value.value.id=='os' and t.value.attr=='environ' and isinstance(t.slice,ast.Constant) and t.slice.value=='CUBLAS_WORKSPACE_CONFIG':
    require(isinstance(node.value,ast.Constant) and node.value.value==':4096:8','Unexpected original cuBLAS assignment');changed.append(node);node.value=ast.copy_location(ast.Constant(workspace),node.value)
 require(len(changed)==1,'Expected exactly1 core startup cuBLAS literal');ast.fix_missing_locations(tree)
 # Reverse transformation must recover the exact semantic original AST.
 reverse=ast.parse(ast.unparse(tree))
 for node in reverse.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and ast.dump(node.targets[0],include_attributes=False)==ast.dump(changed[0].targets[0],include_attributes=False):node.value=ast.Constant(':4096:8')
 require(ast.dump(reverse,include_attributes=False)==ast.dump(ast.parse(source),include_attributes=False),'Core AST changed outside environment literal')
 return tree

def profile_replay_ast(source):
 tree=ast.parse(source);replaynode=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='replay');original=ast.dump(replaynode,include_attributes=False);changed=[]
 for node in ast.walk(replaynode):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='profile':
   kw={x.arg:x for x in node.keywords};require(all(k in kw for k in ['record_shapes','profile_memory','with_stack']),'Original profile control missing')
   for key,before,after in [('record_shapes',False,True),('profile_memory',True,False),('with_stack',False,True)]:
    require(isinstance(kw[key].value,ast.Constant) and kw[key].value.value is before,'Unexpected original profiler control');kw[key].value=ast.Constant(after)
   changed.append(node)
 require(len(changed)==1,'Expected one instrumentation-only profile call');replaynode.name='stack_profile_replay';ast.fix_missing_locations(replaynode)
 reverse=ast.parse(ast.unparse(replaynode)).body[0];reverse.name='replay'
 for node in ast.walk(reverse):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='profile':
   for kw in node.keywords:
    if kw.arg in ['record_shapes','profile_memory','with_stack']:kw.value=ast.Constant({'record_shapes':False,'profile_memory':True,'with_stack':False}[kw.arg])
 require(ast.dump(reverse,include_attributes=False)==original,'Profile AST changed outside instrumentation keywords/name')
 return ast.Module(body=[replaynode],type_ignores=[])

def extract(source,names,namespace,filename):
 tree=ast.parse(source);nodes=[x for x in tree.body if getattr(x,'name',None) in names];require({x.name for x in nodes}==set(names),'Original AST function/class missing')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),namespace)
 return namespace

def controller_admission(root,authorization,auth,unit,plan,reader=None,parent_pid=None):
 require((root/'RUN_COPY_ORIGIN.json').is_file(),'GPU must use isolated authorized run copy')
 require(Path(plan['future_execution_root']).resolve()==root,'Not authorized future execution root')
 meta=read(root/'execution/CONTROLLER_STARTED.json');dispatch_path=root/'execution'/unit['unit_id']/'DISPATCH.json';dispatch=read(dispatch_path)
 require(os.environ.get('QWEN_CONTROLLER_DISPATCH_SHA256')==sha(dispatch_path),'Missing exact immutable controller dispatch binding')
 pid=os.getppid() if parent_pid is None else parent_pid
 if reader is None:
  def reader(pid):
   fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split();argv=Path('/proc/'+str(pid)+'/cmdline').read_bytes().rstrip(b'\0').split(b'\0');return int(fields[19]),[x.decode() for x in argv]
 tick,argv=reader(pid);script=str((root/'scripts/run_cublas_controller.py').resolve())
 require(pid==meta['controller_pid']==dispatch['controller_pid'] and tick==meta['controller_starttick']==dispatch['controller_starttick'],'Worker not direct child of admitted exact controller')
 require(argv==meta['controller_cmdline'] and any(str(Path(x).resolve())==script for x in argv if not x.startswith('-')),'Controller cmdline/source mismatch')
 require('--root' in argv and Path(argv[argv.index('--root')+1]).resolve()==root,'Controller root mismatch')
 digest=sha(authorization);require(meta['authorization_sha256']==dispatch['authorization_sha256']==digest and meta['source_manifest_sha256']==dispatch['source_manifest_sha256']==auth['source_manifest_sha256'],'Controller authorization/source binding mismatch')
 require(dispatch['unit']==unit and isinstance(dispatch['budget_snapshot'],dict),'Wrong unit or missing protected budget dispatch')
 budget=dispatch['budget_snapshot'];require(budget['unit_allowance_seconds']==unit['allowance_seconds'] and budget['protective_seconds']==1650 and budget['hard_seconds']==1800,'Budget contract changed')
 require(math.isfinite(budget['completed_observed_seconds_upper']) and 0<=budget['completed_observed_seconds_upper'] and budget['completed_observed_seconds_upper']+unit['allowance_seconds']<=1650,'Dispatch exceeds protected cumulative budget')
 return dispatch

def ambient_controls(contract):
 require(sys.version.split()[0]==contract['environment']['python'],'Python version changed before engine import')
 selected={key:os.environ.get(key) for key in ['NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUDA_LAUNCH_BLOCKING']}
 for key,value in selected.items():require(value!='1','Ambient override violates strict math/timing:'+key)
 return selected

def admission(root,args):
 # Import-only/--help/defaultNO cannot reach any CUDA-capable import.
 require(os.environ.get('ALLOW_GPU_DIAG')=='YES','GPU_NOT_AUTHORIZED: explicit ALLOW_GPU_DIAG=YES required')
 require(not(root/'FIRST_FAILURE.json').exists(),'Prior failure prohibits dispatch/retry')
 a=read(args.authorization);require(a.get('task_id')==TASK and a.get('status')=='PASS' and a.get('ALLOW_GPU_DIAG')=='YES','Wrong/missing explicit GPU authorization')
 require(Path(a['root']).resolve()==root,'Authorization root mismatch')
 require(a['device']['physical_index']==0 and a['device']['uuid']=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422','Device authorization changed')
 require(sha(root/'SOURCE_MANIFEST.json')==a['source_manifest_sha256'],'Execution source manifest changed')
 manifest=read(root/'SOURCE_MANIFEST.json');rows=manifest['execution_sources']+manifest['external_dependencies'];require(rows,'Empty source closure')
 for row in rows:bound(row,root)
 require(any((root/row['path']).resolve()==Path(__file__).resolve() for row in manifest['execution_sources']),'Worker absent execution source manifest')
 cp=root/'analysis/STRICT_BASELINE_CONTRACT.json';require(sha(cp)==a['contract_sha256'],'Strict contract changed');contract=read(cp);ambient_controls(contract)
 pp=root/'qualification/CUBLAS_PREREGISTRATION.json';require(sha(pp)==a['preregistration_sha256'],'Preregistration changed')
 units=[x for x in build_stage_schedule() if x['unit_id']==args.unit_id];require(len(units)==1,'Unregistered unit');unit=units[0]
 require(unit in a['units'],'Unit not authorized');require(all(getattr(args,k)==unit[k] for k in ['mode','condition','codec']),'CLI/unit mismatch')
 controller_admission(root,args.authorization,a,unit,read(pp))
 for k,v in fresh_environment(args.condition,{}).items():require(os.environ.get(k)==v,'Fresh startup environment changed:'+k)
 require('torch' not in sys.modules and not any(x.startswith('transformers') for x in sys.modules),'GPU-capable engine imported before admission')
 for row in contract['required_files']:bound(row)
 for row in contract['model']['weight_shards']:bound(row)
 for row in contract['source_bindings'].values():bound(row)
 rb=contract['reference_binding'];require(sha(rb['path'])==rb['sha256'],'Original reference binding changed')
 for row in contract['closure_manifests']:require(sha(row['path'])==row['sha256'],'Original dependency manifest changed')
 fi=contract['fixture'];require(fi['non_AIME'] is True and sha(fi['tokens_path'])==fi['tokens_file_sha256'],'Input fixture changed');ids=read(fi['tokens_path'])
 require(len(ids)==fi['length'] and len(ids)>=256 and all(type(x) is int and 0<=x<248320 for x in ids),'Invalid fixed inputs');require(sha_bytes(json.dumps(ids,separators=(',',':')).encode())==fi['tokens_canonical_sha256'],'Canonical fixture identity changed')
 phase=root/'execution'/unit['unit_id'];require(not(phase/'STARTED.json').exists(),'Duplicate/failed/completed unit cannot restart')
 if unit['mode'] in ['benchmark','profile'] or unit['codec']=='H_C128':
  q=read(root/'qualification/FP_NUMERICAL_VERDICT.json');require(q.get('status')=='PASS','Strict numerical qualification must precede performance')
  require(all(q['identities'][key]==a[key] for key in ['source_manifest_sha256','contract_sha256','preregistration_sha256']),'Stale numerical verdict')
 if unit['mode']=='profile' or unit['codec']=='H_C128':require(read(root/'qualification/PERFORMANCE_VERDICT.json').get('status')=='PASS_BENEFIT','Only credible qualified benefit admits profile')
 atomic_new(phase/'STARTED.json',dict(task_id=TASK,unit=unit,process_identity=process_identity(),pid=os.getpid(),unix=time.time(),authorization_sha256=sha(args.authorization),source_manifest_sha256=a['source_manifest_sha256']))
 return a,contract,ids,unit,phase

def runtime(root,contract,condition):
 bindings=contract['source_bindings'];src=Path(bindings['core']['path']).read_text();tree=transform_core(src,WORKSPACES[condition]);K=types.ModuleType('_fast_runtime_core');K.__file__=bindings['core']['path'];K._core_AST_sha256=sha_bytes(ast.dump(tree,include_attributes=False).encode());sys.modules[K.__name__]=K
 # This is the first torch-capable import, after full source/authorization gate.
 exec(compile(tree,bindings['core']['path'],'exec'),K.__dict__);require(os.environ['CUBLAS_WORKSPACE_CONFIG']==WORKSPACES[condition],'Core reset requested workspace')
 K.ROOT=root;K.save=lambda rel,obj:atomic_new(root/'reference_failures'/rel,obj)
 spec=importlib.util.spec_from_file_location('_fast_runtime_codec',bindings['codec']['path']);C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
 ns={'K':K,'C':C,'torch':K.torch,'collections':collections,'CURRENT':{}}
 extract(Path(bindings['worker']['path']).read_text(),{'Observe','dual','describe'},ns,bindings['worker']['path'])
 baseline=dict(ns,Path=Path,contextlib=contextlib,random=random,time=time,gc=gc,math=math,statistics=statistics,sys=sys,os=os,sha=sha,check=require,WARMUP=128,TASK=TASK,json=json)
 extract(Path(bindings['baseline_worker']['path']).read_text(),{'replay','summary','profile_rows'},baseline,bindings['baseline_worker']['path'])
 exec(compile(profile_replay_ast(Path(bindings['baseline_worker']['path']).read_text()),bindings['baseline_worker']['path'],'exec'),baseline)
 K.configure();torch=K.torch
 importlib.import_module('torch.utils.deterministic')
 require(torch.utils.deterministic.fill_uninitialized_memory is True,'Required deterministic fill flag changed')
 require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(),'Determinism contract changed')
 require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'TF32 contract changed')
 require(torch.backends.cudnn.deterministic is False,'cuDNN deterministic flag changed')
 require(sys.version.split()[0]==contract['environment']['python'],'Python version changed')
 require(torch.get_num_threads()==torch.get_num_interop_threads()==1,'Thread contract changed')
 actual=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True,timeout=5);row=[x.split(', ') for x in actual.strip().splitlines() if x.split(', ')[0]=='0']
 require(len(row)==1 and row[0][1]=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422' and '3090' in row[0][2],'Actual physical GPU differs')
 model,qmod,env=K.load();require(sys.modules['transformers'].__version__==contract['environment']['transformers'],'Transformers version changed');require(os.environ['CUBLAS_WORKSPACE_CONFIG']==WORKSPACES[condition],'Runtime changed workspace');require(env['parameter_dtypes']==['torch.bfloat16'],'Model dtype changed')
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=5);own=[x.split(',') for x in active.strip().splitlines() if x.split(',')[0].strip()==str(os.getpid())];require(len(own)==1 and own[0][1].strip()=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422','Own PID GPU binding changed')
 env.update(CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],deterministic_fill=True,deterministic_algorithms=True,matmul_TF32=False,cudnn_TF32=False,intraop=1,interop=1)
 return K,C,ns,baseline,model,qmod,env

def loaded_library_identity(contract,maps_text=None):
 # Actual selected paths, not assumption from pip/conda disk inventory.
 allowed={str(Path(x.get('resolved_path',x['path'])).resolve()):x for x in contract['prospective_library_bindings'] if 'path' in x and 'sha256' in x};selected=[]
 mapped=set()
 for line in (Path('/proc/self/maps').read_text() if maps_text is None else maps_text).splitlines():
  fields=line.split(None,5)
  if len(fields)==6 and fields[5].startswith('/') and '(deleted)' not in fields[5]:mapped.add(str(Path(fields[5]).resolve()))
 for name in sorted(mapped):
  base=Path(name).name
  role='torch' if base.startswith(('libtorch','libc10')) else 'cuBLAS' if base.startswith('libcublas') else 'cuDNN' if base.startswith('libcudnn') else 'CUDA_runtime' if base.startswith('libcudart') else 'driver' if base.startswith('libcuda.so') else None
  if role is None:continue
  p=Path(name);row=allowed.get(name)
  if role!='driver':require(row is not None,'Unbound actual loaded library:'+name)
  actual=dict(path=name,bytes=p.stat().st_size,sha256=sha(p),role=role)
  if row is not None:require(actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'],'Loaded library bytes/SHA changed:'+name)
  selected.append(actual)
 require(any(x['role']=='cuBLAS' for x in selected) and any(x['role']=='torch' for x in selected),'Actual cuBLAS/torch map unavailable')
 return selected

def tensor_bytes(torch,t):return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
class DeepRecorder:
 def __init__(self,K,qmod,obs,phase):self.K=K;self.qmod=qmod;self.obs=obs;self.items=[];self.old={};self.calls=collections.Counter();self.phase=phase;self.reference=None
 def add(self,name,value):
  torch=self.K.torch
  if isinstance(value,torch.Tensor):
   cpu=value.detach().cpu().contiguous();finite=bool(torch.isfinite(cpu).all().item()) if cpu.is_floating_point() else True
   raw=cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
   if not finite:
    label='NONFINITE_'+str(self.obs.position)+'_'+hashlib.sha256(name.encode()).hexdigest()[:16];out=self.phase/'numeric';out.mkdir(parents=True,exist_ok=True);rp=out/(label+'.bin')
    with rp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    atomic_new(out/(label+'.json'),dict(status='NONFINITE_STOP',token=self.obs.position,name=name,dtype=str(value.dtype),shape=list(value.shape),stride=list(value.stride()),raw_file=rp.name,raw_sha256=sha_bytes(raw),reference_receipt=str(self.reference/f'token_{self.obs.position:03d}.json') if self.reference else None))
    raise RuntimeError('Nonfinite tensor token='+str(self.obs.position)+' name='+name+';raw preserved')
   self.items.append((name,str(value.dtype),tuple(value.shape),tuple(value.stride()),raw))
  elif isinstance(value,(list,tuple)):
   for i,v in enumerate(value):self.add(name+'/'+str(i),v)
  elif isinstance(value,dict):
   for k,v in sorted(value.items()):self.add(name+'/'+str(k),v)
 def deep(self,name,value,seen=None):
  if seen is None:seen=set()
  if isinstance(value,self.K.torch.Tensor):self.add(name,value);return
  if isinstance(value,(str,int,float,bool,type(None),self.K.torch.dtype,self.K.torch.device,type)):return
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,dict):
   for k,v in sorted(value.items(),key=lambda x:str(x[0])):self.deep(name+'/'+str(k),v,seen)
  elif isinstance(value,(list,tuple)):
   for k,v in enumerate(value):self.deep(name+'/'+str(k),v,seen)
  elif hasattr(value,'__dict__'):
   for k,v in sorted(vars(value).items()):self.deep(name+'/'+str(k),v,seen)
 def wrap(self,fn):
  def call(q,k,v,*args,**kw):
   layer=self.obs.active;self.calls[layer]+=1;prefix='GDN/'+str(layer)
   self.add(prefix+'/q',q);self.add(prefix+'/k',k);self.add(prefix+'/v',v);self.add(prefix+'/args',args);self.add(prefix+'/kwargs',kw)
   y,s=fn(q,k,v,*args,**kw);self.add(prefix+'/readout',y);self.add(prefix+'/updated_state',s);return y,s
  return call
 def install(self):
  for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']:self.old[n]=getattr(self.qmod,n);setattr(self.qmod,n,self.wrap(self.old[n]))
 def close(self):
  for n,v in self.old.items():setattr(self.qmod,n,v)

def qualify(K,ns,model,qmod,ids,unit,phase,root):
 torch=K.torch;f=K.formal(True);obs=ns['Observe'](model,qmod,f,False);rec=DeepRecorder(K,qmod,obs,phase);rec.install();cache=None
 reference=root/'execution'/unit['reference_unit']/'numeric' if unit['reference_unit'] else None
 rec.reference=reference
 if reference:require(read(reference/'NUMERICAL_INDEX.json')['status']=='PASS','Incomplete strict reference')
 store=SnapshotStore(phase/'numeric',reference);gpuids=torch.tensor([ids[:256]],dtype=torch.long,device='cuda:0');condition='FP_NATIVE' if unit['codec']=='FP' else 'H_C128';hctx=f.QwenKeyHadamardPatch(model) if condition!='FP_NATIVE' else contextlib.nullcontext();start=time.monotonic()
 try:
  with hctx as h,torch.inference_mode():
   for t in range(256):
    obs.position=t;obs.details={};obs.finite=[];rec.items=[];rec.calls.clear();rec.add('input_ids',gpuids[:,t:t+1]);ns['CURRENT'].update(tag=unit['unit_id'],condition=condition,token_index=t,consumed=t+1,role='STRICT_BYTE_QUALIFICATION')
    out=model(input_ids=gpuids[:,t:t+1],past_key_values=cache,use_cache=True);cache=out.past_key_values;K.check(torch.stack(obs.finite).all()&torch.isfinite(out.logits).all(),'Nonfinite state/logits');rec.add('logits',out.logits)
    if condition=='H_C128':
     audit=dict(calls=0,layers=set(),scale_shapes={},code_min=0,code_max=0);lines=Path(f.quantize_c128.__code__.co_filename).read_text().splitlines();line=next(i+1 for i,s in enumerate(lines) if 'state.copy_(dequant.to(state.dtype))' in s)
     def trace(frame,event,arg):
      if event=='line' and frame.f_code is f.quantize_c128.__code__ and frame.f_lineno==line:
       x=frame.f_locals;p='C128/'+str(x['layer_idx']);rec.add(p+'/codes_int8',x['codes'].to(torch.int8));rec.add(p+'/scales',x['scale']);rec.add(p+'/dequant',x['dequant']);rec.add(p+'/pre_state',x['state'])
      return trace
     require(sys.gettrace() is None,'Nested Python tracer not allowed');sys.settrace(trace)
     try:f.quantize_c128(cache,audit)
     finally:sys.settrace(None)
     K.check(audit['calls']==1 and audit['layers']==set(f.GDN_LAYERS),'OriginalC128 writeback audit failed')
    require(set(rec.calls)==set(f.GDN_LAYERS) and all(v==1 for v in rec.calls.values()),'Exactly1 GDN call pertoken/layer required')
    rec.deep('CACHE',cache)
    require(int(cache.get_seq_length())==t+1,'Cache cursor mismatch')
    # Original deep describe includes all cache attributes, tensor hashes, cursor.
    metadata=dict(token=t,input_id=ids[t],next_input_id=ids[t+1] if t+1<256 else None,cache_cursor=int(cache.get_seq_length()),deep_cache=ns['describe'](cache),codec=unit['codec'],GDN_layers=list(f.GDN_LAYERS))
    row=store.record(t,rec.items,metadata)
    if not row['byte_equal']:
     atomic_new(phase/'EARLIEST_NUMERICAL_FAILURE.json',dict(status='FAIL',token=t,mismatched_names=row['mismatched_names'],actual_receipt=str(phase/'numeric'/f'token_{t:03d}.json'),reference_receipt=str(reference/f'token_{t:03d}.json')));raise RuntimeError('First strict byte mismatch at token '+str(t))
    obs.previous={l:(f.get_state(cache,l).data_ptr(),K.tsha(f.get_state(cache,l))) for l in f.GDN_LAYERS}
    for l in f.GDN_LAYERS:K.check(obs.calls[(t,l)]==1,'Original safety GDN count mismatch')
    del out
    if (t+1)%32==0:print(json.dumps(dict(event='QUALIFICATION_PROGRESS',unit_id=unit['unit_id'],tokens=t+1,wall=time.monotonic()-start)),flush=True)
   if condition!='FP_NATIVE':require(h.calls==6144 and h.phase_counts=={'chunk_prefill':24,'recurrent_decode':6120},'H boundary/count changed')
   index=store.finish()
  return dict(numerical_index=dict(path='numeric/NUMERICAL_INDEX.json',sha256=sha(phase/'numeric/NUMERICAL_INDEX.json'),bytes=(phase/'numeric/NUMERICAL_INDEX.json').stat().st_size),consumed_tokens=256,native_core_calls=6144,next_token_consumption_checks=6120,all_tokens_finite=True,strict_byte_pass=True,numerical_wall_seconds=time.monotonic()-start,intermediate_scope='All helper q/k/v,tensor args/kwargs inclg/beta/initial,returnedreadout/updatedstate; not all internal temporary ATen values',per_token_full_cache=True)
 finally:rec.close();obs.close();cache=None;gc.collect();torch.cuda.empty_cache()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--unit-id',required=True);p.add_argument('--mode',choices=['qualify','benchmark','profile'],required=True);p.add_argument('--condition',choices=['B0','B1'],required=True);p.add_argument('--codec',choices=['FP','H_C128'],required=True);args=p.parse_args();root=args.root.resolve();phase=root/'execution'/args.unit_id;admitted=False;job_started=time.monotonic();ns=None
 try:
  auth,contract,ids,unit,phase=admission(root,args);admitted=True;K,C,ns,base,model,qmod,env=runtime(root,contract,args.condition);atomic_new(phase/'ENVIRONMENT.json',env)
  if args.mode=='qualify':result=qualify(K,ns,model,qmod,ids,unit,phase,root)
  elif args.mode=='benchmark':
   gpuids=K.torch.tensor([ids[:256]],dtype=K.torch.long,device='cuda:0');raw,_=base['replay'](K,C,ns,model,qmod,gpuids,'FP_NATIVE',128,unit['repeat'],False,phase,'benchmark','NUMERIC_QUALIFIED_NO_PROFILER')
   result=dict(benchmark_reference_result=raw,tokens_per_second=raw['decode_tokens_per_second'],decode_wall_seconds=raw['decode_window_wall_seconds'],measured_tokens=128,warmup_tokens=128,peak_allocated_bytes=raw['peak_allocated_bytes'],peak_reserved_bytes=raw['peak_reserved_bytes'])
  else:
   gpuids=K.torch.tensor([ids[:160]],dtype=K.torch.long,device='cuda:0');raw,_=base['stack_profile_replay'](K,C,ns,model,qmod,gpuids,'FP_NATIVE',32,0,True,phase,'profile','QUALIFIED_SHORT_STACK_DIAGNOSTIC_NOT_TPS')
   result=dict(profile_reference_result=raw,profile_tokens=32,warmup_tokens=128,record_shapes=True,with_stack=True,profile_is_not_unprofiled_performance=True,chrome_trace=raw['chrome_trace'])
  result['core_environment_control']=dict(original_source_sha256=contract['source_bindings']['core']['sha256'],executed_semantic_AST_sha256=K._core_AST_sha256,workspace=WORKSPACES[args.condition],only_core_AST_difference='Single module-level CUBLAS_WORKSPACE_CONFIG string literal; reverse AST exact')
  libs=loaded_library_identity(contract);result.update(runtime_library_identity=libs,runtime_library_digest=sha_bytes(canonical(libs)))
  overrides=ambient_controls(contract);result.update(runtime_override_environment=overrides,runtime_override_env_digest=sha_bytes(canonical(overrides)))
  result.update(seed=20261015,process_identity=process_identity(),whole_process_wall_seconds=time.monotonic()-job_started)
  result.update(status='PASS',task_id=TASK,unit_id=args.unit_id,mode=args.mode,condition=args.condition,workspace=WORKSPACES[args.condition],codec=args.codec,repeat=unit['repeat'],GPU_UUID=auth['device']['uuid'],source_manifest_sha256=auth['source_manifest_sha256'],contract_sha256=auth['contract_sha256'],preregistration_sha256=auth['preregistration_sha256'],fixture_canonical_sha256=contract['fixture']['tokens_canonical_sha256'],authorization_sha256=sha(args.authorization),qualification_not_long_context_guarantee=True)
  atomic_new(phase/'RESULT.json',result)
 except BaseException as exc:
  failure=dict(status='FAIL_CLOSED_STOP' if admitted else 'DENIED_BEFORE_ENGINE_IMPORT',task_id=TASK,unit_id=args.unit_id,error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc(),GPU_admitted=admitted,unix=time.time())
  if admitted:
   failure['execution_context']=dict(ns.get('CURRENT',{})) if ns else {}
   if not(phase/'RESULT.json').exists():atomic_new(phase/'RESULT.json',failure)
   if not(root/'FIRST_FAILURE.json').exists():atomic_new(root/'FIRST_FAILURE.json',failure)
  print(json.dumps(failure),file=sys.stderr);raise SystemExit(2)
if __name__=='__main__':main()
