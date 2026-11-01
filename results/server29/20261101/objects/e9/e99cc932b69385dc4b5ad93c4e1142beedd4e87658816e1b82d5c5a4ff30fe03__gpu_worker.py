"""Prepared micro/canary/timing worker. No runtime imports until fresh task permit."""
import argparse,os,sys,json,hashlib,time,gc
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--permit',type=Path,required=True);ap.add_argument('--unit',required=True);a=ap.parse_args()
R=a.root.resolve();sys.path.insert(0,str(R/'src'))
import gpu_control as C
d=C.worker_admit(R,a.permit,a.unit)  # default NO exits before any GPU import/API
target=R/'execution'/a.unit
C.atomic_new(target/'STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic()))
plan=C.read(R/'GPU_PLAN.json');unit=next(x for x in plan['units'] if x['id']==a.unit)
C.atomic_new(target/'DRIVER_IDENTITY.json',C.require_driver_ordinal_identity(list(C.bindings(R).values())))
import torch,numpy as np
from group_codecs import encode,decode,tensor_bytes
from persistent_adapter import PersistentQDQ
from canonical_h128 import setup_seed,load_model_and_tokenizer,GDN_LAYERS,hadamard
from inherited_archive_transport import SSHArchiveTransport
archive=SSHArchiveTransport(dict(host='zypan@202.38.247.29',control_path='/tmp/qwen_replication_archive29.sock',remote_python='/usr/bin/python3',global_headroom_bytes=100*1024**3,io_timeout_seconds=120))
archive_root=plan['archive_root']
def save(name,v):return C.atomic_new(target/name,v)
def fields_raw(values):
    raws=[];rows=[];offset=0;seen={}
    for name,x in values:
        b=tensor_bytes(x);key=(hashlib.sha256(b).hexdigest(),len(b));previous=seen.get(key)
        if previous is not None and previous[1]==b:
            rows.append(dict(name=name,dtype=str(x.dtype),shape=list(x.shape),bytes=len(b),offset=previous[0]));continue
        seen[key]=(offset,b);rows.append(dict(name=name,dtype=str(x.dtype),shape=list(x.shape),bytes=len(b),offset=offset));raws.append(b);offset+=len(b)
    return b''.join(raws),rows
def publish(name,raw):
    path=archive_root+'/'+a.unit+'/'+name
    return archive.put_immutable(path,[raw],len(raw),hashlib.sha256(raw).hexdigest())
def progress(t):save('PROGRESS_%04d.json'%t,dict(token=t,monotonic=time.monotonic()))
def preflight():
    v=C.library_check(R);save('PREFLIGHT_LIBRARY.json',v);C.require(v['status']=='PASS','Library identity failed')
def postflight():
    v=C.library_check(R);save('POSTFLIGHT_LIBRARY.json',v);C.require(v['status']=='PASS','Library identity failed')
result={}
try:
 if unit['kind']=='micro':
    torch.empty(1,device='cuda');preflight();results={};h=hadamard(128,device='cuda')
    C.require(float((h.T@h-torch.eye(128,device='cuda')).abs().max())<2e-6,'GPU H orthogonality')
    rows=C.read(R/'fixtures/GPU_MICRO_INPUT_MANIFEST.json')['files']
    for method in ('B0','B1','B2'):
      records=[]
      try:
       for row in rows:
        p=R/row['file'];C.require(C.sha(p)==row['sha256'],'Micro input identity')
        fixture=np.load(p);x=torch.from_numpy(fixture['input']).cuda();previous=None
        for rep in range(2):
         debug={}
         e,debug=encode(x,method,debug=True)
         for key in ('codes','scales','zero_points','negative_zero_flags','decoded'):
          actual=decode(e) if key=='decoded' else getattr(e,key)
          if actual is None:continue
          ref=fixture[method+'_'+key]
          C.require(actual.detach().cpu().numpy().dtype==ref.dtype and tuple(actual.shape)==ref.shape and tensor_bytes(actual)==ref.tobytes(),'Micro strict field mismatch:'+row['file']+':'+method+':'+key)
         now=tensor_bytes(decode(e));C.require(previous is None or previous==now,'Same GPU codec repeat failed');previous=now
        records.append(dict(fixture=row['file'],pass_fields=True,repeat=2))
       results[method]=dict(status='GPU_CODEC_QUALIFIED',fixtures=records)
      except BaseException as exc:
       values=[('input',x)]+list(debug.items()) if 'debug' in locals() else [('input',x)]
       raw,idx=fields_raw(values);receipt=publish('FIRST_FAILURE_'+method+'.bin',raw)
       results[method]=dict(status='NUMERICAL_QUALIFICATION_FAIL',error=str(exc),full_first_failure=receipt,tensors=idx)
       save('FIRST_FAILURE_'+method+'.json',results[method])
    result=dict(status='PASS_MICRO_TERMINAL',conditions=results)
 else:
    C.require(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1','Runtime changed')
    torch.set_num_threads(24);torch.set_num_interop_threads(24)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
    setup_seed(20261019);model,tokenizer=load_model_and_tokenizer()
    C.require(len(model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers)==32,'Not complete model')
    C.require(all(x.device.type=='cuda' and x.device.index==0 and x.dtype==torch.bfloat16 for x in model.parameters()),'Model dtype/device/offload changed')
    preflight();obj=C.read(R/'inherited/historical_port_payload/candidate/benchmark_replay/fixtures/doc1_tokens.json')
    ids=obj if isinstance(obj,list) else obj.get('input_ids',obj.get('tokens'));C.require(len(ids)==32768,'Frozen input changed')
    observe=unit['kind']=='canary';N=unit.get('tokens',256);pending=[];trace=[];cache=None
    def capture(phase,t,layer,values):
        for name,x in values.items():pending.append((phase+'/'+str(layer)+'/'+name,x.detach().cpu().clone()))
    with torch.inference_mode(),PersistentQDQ(model,unit['method'],GDN_LAYERS,observe=observe,callback=capture if observe else None) as q:
      torch.cuda.reset_peak_memory_stats();start=None
      for t in range(N):
        if unit['kind']=='benchmark' and t==128:torch.cuda.synchronize();start=time.perf_counter()
        out=model(input_ids=torch.tensor([[ids[t]]],device='cuda'),past_key_values=cache,use_cache=True);cache=out.past_key_values
        q.after_token(cache)
        C.require(bool(torch.isfinite(out.logits).all().item()),'Nonfinite logits')
        if observe:
          pending.append(('logits',out.logits.detach().cpu()))
          # Full cache tensors, including convolution and all Attention K/V.
          def walk(x,name):
            if isinstance(x,torch.Tensor):pending.append((name,x.detach().cpu().clone()))
            elif isinstance(x,dict):
              for k in sorted(x,key=str):walk(x[k],name+'/'+str(k))
            elif isinstance(x,(list,tuple)):
              for i,v in enumerate(x):walk(v,name+'/'+str(i))
            elif type(x).__module__.startswith('transformers') and hasattr(x,'__dict__'):
              for k in sorted(vars(x)):walk(vars(x)[k],name+'/'+k)
          walk(cache,'cache')
          raw,index=fields_raw(pending);pending.clear()
          if unit['repeat']==0:
            receipt=publish('token_%04d.bin'%t,raw)
          else:
            ref=C.read(R/'execution'/unit['reference']/('TOKEN_%04d.json'%t))
            C.require(ref['tensors']==index,'Fresh tensor roster/layout differs')
            with archive.open_verified(ref['raw']['path'],ref['raw']['bytes'],ref['raw']['sha256']) as stream:
                expected=stream.read(len(raw));same=expected==raw
            if not same:
                receipt=publish('FIRST_FAILURE_token_%04d.bin'%t,raw)
                save('FIRST_FAILURE.json',dict(kind='FRESH_ACTUAL_BYTES',token=t,reference=ref,actual=receipt,tensors=index));raise RuntimeError('Fresh exact bytes failed')
            receipt=ref['raw']
          save('TOKEN_%04d.json'%t,dict(token=t,method=unit['method'],tensors=index,raw=receipt,fresh_actual_bytes_compared=unit['repeat']==1));progress(t)
        elif t==255:
          torch.cuda.synchronize();elapsed=time.perf_counter()-start
      audit=q.audit()
      if observe:C.require(audit['next_consumption_checks']==(N-1)*24,'Incomplete actual next-consumption roster')
      result=dict(status='PASS',method=unit['method'],audit=audit,tokens=N,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
      if not observe:result.update(warmup=128,measured=128,seconds=elapsed,TPS=128/elapsed,ms_per_token=elapsed*1000/128,observer=False,profile=False)
    del model,cache,out;gc.collect();torch.cuda.empty_cache()
 postflight();save('RESULT.json',dict(result,unit=a.unit,identity=C.process_identity(os.getpid()),permit_sha256=C.sha(a.permit)))
except BaseException as exc:
 if not (target/'FIRST_FAILURE.json').exists():
  failure=dict(status='FAIL_CLOSED_STOP',error=repr(exc))
  if 'pending' in globals() and pending:
   raw,idx=fields_raw(pending)
   try:failure.update(raw=publish('FIRST_FAILURE_PENDING.bin',raw),tensors=idx)
   except BaseException as archive_error:
    # Preserve new failed raw locally; never delete before independent archive.
    p=target/'FIRST_FAILURE_PENDING.bin'
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    failure.update(local_raw=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),tensors=idx,archive_error=repr(archive_error))
  save('FIRST_FAILURE.json',failure)
 raise
