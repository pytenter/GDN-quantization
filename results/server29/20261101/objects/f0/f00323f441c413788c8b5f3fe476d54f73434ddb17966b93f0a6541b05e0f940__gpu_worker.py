"""Bounded gate worker. Any exception terminates this GPU execution attempt."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
import argparse,ast,collections,gc,hashlib,inspect,json,math,pathlib,random,subprocess,sys,time,traceback
from types import SimpleNamespace
ROOT=pathlib.Path(__file__).resolve().parents[1];PARENT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import torch
import gpu_codec as C
CHECKPOINTS=(1,8,32,128);SCORES=(8,32,128)
GPU_UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
CURRENT_CONTEXT={}
class GateFailure(RuntimeError):
    def __init__(self,record):self.record=record;super().__init__(json.dumps(record,ensure_ascii=False))
def check(ok,message,**context):
    if isinstance(ok,torch.Tensor):ok=bool(ok.item())
    if not ok:raise GateFailure(dict(reason=message,**context))
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(rel,obj):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def tsha(x):return hashlib.sha256(x.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tensors(x,prefix='',seen=None):
    if seen is None:seen=set()
    if id(x) in seen:return
    seen.add(id(x))
    if isinstance(x,torch.Tensor):yield prefix,x
    elif isinstance(x,dict):
        for k in sorted(x,key=str):yield from tensors(x[k],prefix+'/'+str(k),seen)
    elif isinstance(x,(list,tuple)):
        for i,v in enumerate(x):yield from tensors(v,prefix+'/'+str(i),seen)
    elif hasattr(x,'__dict__'):
        for k,v in sorted(vars(x).items()):
            if not k.startswith('_'):yield from tensors(v,prefix+'/'+k,seen)
def cache_digest(cache):
    rows=[dict(path=p,shape=list(x.shape),dtype=str(x.dtype),sha256=tsha(x)) for p,x in tensors(cache)]
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),rows
def extract(path,names,ns):
    s=pathlib.Path(path).read_text();nodes=[]
    for n in ast.parse(s).body:
        if getattr(n,'name',None) in names:nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id in names for a in n.targets):nodes.append(n)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def originals():
    r=PARENT/'runtime_sources'
    fns=extract(r/'formal_c128_runner.py',{'EPS','GDN_LAYERS','get_state','quantize_c128','hadamard','normalize_qk','QwenKeyHadamardPatch','load_model_and_tokenizer'},dict(torch=torch,math=math,os=os,sys=sys,Path=pathlib.Path,MODEL_PATH=pathlib.Path('/data/zypan/modelscope_models/Qwen3.5-9B'),TRANSFORMERS_SRC=pathlib.Path('/data/zypan/transformers-qwen35'),STATE_QUANT_SRC=pathlib.Path('/data/zypan/experiments/qwen35_gdn_quant')))
    F=SimpleNamespace(**fns)
    nroot=pathlib.Path('/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3')
    arch=pathlib.Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
    ns=extract(r/'numerical_recovery.py',{'tsha','compensated_scan','backend','chunk_binding','compare'},dict(torch=torch,ast=ast,hashlib=hashlib,Path=pathlib.Path,ROOT=nroot,ARCH=arch,ARCH_SHA=sha(r/'current_modeling_qwen3_5.py'),check=check,sha=sha,read=read))
    return F,SimpleNamespace(**ns),arch

def configure():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','frozen GPU mapping mismatch')
    check(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1','runtime version mismatch')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=False)
    random.seed(73619);torch.manual_seed(73619);torch.cuda.manual_seed_all(73619)
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
    props=torch.cuda.get_device_properties(0)
    check('3090' in props.name and props.total_memory>=24*1000**3,'GPU hardware mismatch',name=props.name)
    nv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    check(nv.splitlines()[0].split(',')[1].strip()==GPU_UUID,'GPU UUID mapping mismatch')
    return dict(torch=torch.__version__,cuda=torch.version.cuda,name=props.name,total_memory=props.total_memory,visible_mapping='physical0 -> cuda0',GPU_UUID=GPU_UUID,TF32=False,strict=True,seed=73619,nvidia_smi=nv)

class Harness:
    def __init__(self,model,F,N,arch):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        self.model=model;self.F=F;self.N=N;self.qmod=qmod;self.active=None;self.mode='H';self.observer=True;self.position=0
        self.pre={};self.readouts={};self.calls=collections.Counter();self.details={};self.previous={}
        self.final_hidden=None
        self.handles=[];self.patch=F.QwenKeyHadamardPatch(None)
        check(pathlib.Path(qmod.__file__).resolve()==arch.resolve(),'live model architecture path mismatch')
        self.original_chunk=qmod.torch_chunk_gated_delta_rule;self.original_recurrent=qmod.torch_recurrent_gated_delta_rule
        recurrent=inspect.unwrap(self.original_recurrent)
        check(pathlib.Path(recurrent.__code__.co_filename).resolve()==arch.resolve(),'live recurrent helper is not frozen torch source',filename=recurrent.__code__.co_filename)
        self.chunk=N.chunk_binding(qmod,'compensated')
        layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
        for l in F.GDN_LAYERS:
            block=getattr(layers[l],'linear_attn',None) or getattr(layers[l],'self_attn',None)
            check(block is not None,'GDN layer not found',layer=l)
            self.handles.append(block.register_forward_pre_hook(lambda m,a,l=l:setattr(self,'active',l)))
        self.handles.append(model.lm_head.register_forward_pre_hook(self.hidden_hook))
        qmod.torch_chunk_gated_delta_rule=self.wrap(self.chunk,'chunk')
        qmod.torch_recurrent_gated_delta_rule=self.wrap(self.original_recurrent,'recurrent')
        self.dispatch=dict(architecture_path=str(arch),architecture_SHA=sha(arch),recurrent_unwrapped_source=recurrent.__code__.co_filename,recurrent_qualname=recurrent.__qualname__,chunk='unchanged qualified V3 compensated chunk_binding',GDN_module_classes={str(l):type(getattr(layers[l],'linear_attn',None) or getattr(layers[l],'self_attn',None)).__name__ for l in F.GDN_LAYERS})
    def hidden_hook(self,module,args):
        if self.observer:self.final_hidden=args[0].detach().cpu().clone()
    def wrap(self,fn,kind):
        def invoke(query,key,value,*args,**kw):
            l=self.active;check(l in self.F.GDN_LAYERS,'unmapped GDN helper')
            self.calls[(self.position,l)]+=1
            initial=kw.get('initial_state');initial_hash=None
            if initial is not None:
                initial_hash=tsha(initial)
                if self.previous:check(initial_hash==self.previous[l],'next-token QDQ consumption mismatch',token=self.position,layer=l)
            if self.mode=='H':
                norm=kw.pop('use_qk_l2norm_in_kernel',False)
                query=self.patch._transform(query,norm);key=self.patch._transform(key,norm)
                kw['use_qk_l2norm_in_kernel']=False
            if self.observer:
                qhash=tsha(query);khash=tsha(key)
            result,state=fn(query,key,value,*args,**kw)
            if self.mode=='H':result=result.to(value.dtype)
            check(state.dtype==torch.float32 and list(state.shape)==[1,32,128,128],'state dtype/shape mismatch',layer=l)
            if self.observer:
                self.details[str(l)]=dict(q=qhash,k=khash,pre_state=tsha(state),readout=tsha(result),initial_state=initial_hash,kind=kind)
                self.readouts[l]=result.detach().cpu().clone()
            return result,state
        return invoke
    def forward(self,token,cache,position,mode='H',observer=True):
        self.mode=mode;self.observer=observer;self.position=position;self.readouts={};self.details={};self.previous={}
        out=self.model(input_ids=torch.tensor([[int(token)]],device='cuda:0'),past_key_values=cache,use_cache=True)
        return out,out.past_key_values
    def close(self):
        self.qmod.torch_chunk_gated_delta_rule=self.original_chunk;self.qmod.torch_recurrent_gated_delta_rule=self.original_recurrent
        for h in self.handles:h.remove()

def load():
    F,N,arch=originals();start=time.monotonic();model,tokenizer=F.load_model_and_tokenizer()
    for p in model.parameters():p.requires_grad_(False)
    model.eval();H=Harness(model,F,N,arch)
    check(all(p.device.type=='cuda' and p.device.index==0 for p in model.parameters()),'model device map/offload changed')
    model_dtypes=collections.Counter(str(p.dtype) for p in model.parameters())
    return model,H,F,N,dict(load_wall_seconds=time.monotonic()-start,parameter_dtype_counts=dict(model_dtypes),tokenizer_class=tokenizer.__class__.__name__,dispatch=H.dispatch)

def document(i):
    a=read(PARENT/'analysis/FROZEN_CANARY_DOCUMENTS.json')['selected_documents'][i-1]
    ids=read(PARENT/a['token_ids_path'])
    check(len(ids)==128 and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==a['token_ids_canonical_json_sha256'],'frozen token identity mismatch',doc=i)
    return a,ids

def compare(N,ref,cand,kind,context):
    result=N.compare(ref,cand,kind)
    if not result['PASS']:
        record=dict(context=context,comparison=result,reference_sha=tsha(ref),candidate_sha=tsha(cand))
        p=ROOT/'analysis/FIRST_FAILURE_TENSORS.pt'
        torch.save(dict(reference=ref.detach().cpu(),candidate=cand.detach().cpu(),context=context),p)
        raise GateFailure(dict(reason='frozen numerical bound failed',**record))
    return result

def a0(phase):
    model,H,F,N,loadinfo=load();h=F.hadamard(128,device='cuda:0');hsha=tsha(h)
    check(hsha==read(PARENT/'analysis/HADAMARD_IDENTITY.json')['sha256_raw_c_order_bytes'],'Hadamard identity mismatch')
    docs={};comparison_rows=[];started=time.monotonic()
    original=read(ROOT/'analysis/a0_1.json') if phase=='a0_2' else None
    with torch.inference_mode():
        for doc in (1,2):
            a,ids=document(doc);caches={'N':None,'H':None};tokens=[]
            for position,token in enumerate(ids,1):
                CURRENT_CONTEXT.update(phase=phase,doc=doc,token=position)
                row={'token':position,'input_id':token};readouts={};outputs={};hidden={}
                for mode in ('N','H'):
                    out,cache=H.forward(token,caches[mode],position,mode,True);caches[mode]=cache
                    check(torch.isfinite(out.logits).all(),'nonfinite FP logits',doc=doc,token=position,mode=mode)
                    dg,cache_rows=cache_digest(cache)
                    row[mode]=dict(logits=tsha(out.logits),cache=dg,helper=H.details,final_hidden=tsha(H.final_hidden))
                    readouts[mode]=H.readouts;outputs[mode]=out.logits.detach();hidden[mode]=H.final_hidden
                records=[]
                records.append(dict(object='logits',**compare(N,outputs['N'],outputs['H'],'model_bf16',dict(doc=doc,token=position,object='full_vocab_logits'))))
                records.append(dict(object='final_hidden',**compare(N,hidden['N'],hidden['H'],'model_bf16',dict(doc=doc,token=position,object='normalized_final_hidden'))))
                for layer in F.GDN_LAYERS:
                    n=F.get_state(caches['N'],layer);rotated=F.get_state(caches['H'],layer)
                    physical=h@rotated
                    records.append(dict(object=f'state/{layer}',**compare(N,n,physical,'model_fp32',dict(doc=doc,token=position,layer=layer,object='H_state_to_native_comparison_only'))))
                    records.append(dict(object=f'readout/{layer}',**compare(N,readouts['N'][layer],readouts['H'][layer],'model_bf16',dict(doc=doc,token=position,layer=layer,object='current_readout'))))
                if original:
                    check(row==original['documents'][str(doc)]['tokens'][position-1],'FP fresh-process identity mismatch',doc=doc,token=position)
                if phase=='a0_1' and position in CHECKPOINTS:
                    snap={str(l):F.get_state(caches['H'],l).detach().cpu().clone() for l in F.GDN_LAYERS}
                    path=ROOT/f'raw_snapshots/doc{doc}_token{position}_FP_H.pt';torch.save(snap,path)
                    checkpoint=dict(logits=outputs['H'].cpu().clone(),readouts=readouts['H'],doc_id=a['document_id'],token=position,FP_H_state_path=str(path.relative_to(ROOT)),coordinate='actual fullmodel FP_H trajectory; no QDQ; Key H exactly once')
                    torch.save(checkpoint,ROOT/f'raw_snapshots/doc{doc}_token{position}_outputs.pt')
                    del snap,checkpoint
                tokens.append(row);comparison_rows.append(dict(doc=doc,token=position,objects=records))
                if position in CHECKPOINTS:
                    save(f'analysis/progress_{phase}_doc{doc}_token{position}.json',dict(status='PASS_TO_POSITION',position=position,elapsed_s=time.monotonic()-started,doc=doc,phase=phase))
                    print(json.dumps(dict(phase=phase,doc=doc,token=position,elapsed_s=time.monotonic()-started)),flush=True)
                del outputs,readouts,out,records
            docs[str(doc)]=dict(doc_id=a['document_id'],tokens=tokens)
            del caches;gc.collect();torch.cuda.empty_cache()
    result=dict(status='PASS',phase=phase,documents=docs,load=loadinfo,Hadamard_sha256=hsha,comparison_semantics='unchanged V3 N.compare; state H inverse only for comparison, never used as H trajectory',fresh_process_exact=phase=='a0_2',wall_seconds=time.monotonic()-started)
    save(f'analysis/{phase}.json',result);save(f'analysis/{phase}_numerical_comparisons.json',comparison_rows)
    H.close();del model,H;gc.collect();torch.cuda.empty_cache()
    return dict(status='PASS',documents=2,tokens_per_trajectory=128)

def small_gpu_parity(x):
    e=C.encode(x,'DUAL_M_INT8')
    plan=C.references.STEP['QuantizationPlan']('gdn',torch.full(x.shape[1:3],8,dtype=torch.long,device=x.device),torch.ones(x.shape[1:3],device=x.device))
    z,r,c,two,y=C.references.STEP['fit_state'](x,plan)
    check(two is None and torch.equal(e.codes.to(torch.int16),z) and torch.equal(e.rows.squeeze(-1),r) and torch.equal(e.columns.squeeze(-2),c) and torch.equal(e.decode(),y),'GPU official STEP exact parity mismatch')
    a=C.encode(x,C.NAMES[0]);ref=C.references.CANON['qwen_c128_qdq'](x)
    check(torch.equal(a.codes.float(),ref.codes) and torch.equal(a.columns,ref.scale) and torch.equal(a.decode(),ref.dequant),'GPU canonical source parity mismatch')

def quantile(x):
    q=torch.quantile(x.double().flatten(),torch.tensor([0.,.01,.25,.5,.75,.99,1.],dtype=torch.float64))
    return dict(zip(('min','p01','p25','median','p75','p99','max'),map(float,q)))

def a1():
    check(read(ROOT/'analysis/a0_2.json')['status']=='PASS','A0 prerequisite')
    F,N,arch=originals();rows=[];safety=[];dist=[]
    with torch.inference_mode():
        for doc in (1,2):
            for position in CHECKPOINTS:
                states=torch.load(ROOT/f'raw_snapshots/doc{doc}_token{position}_FP_H.pt',map_location='cpu',weights_only=True)
                for layer in F.GDN_LAYERS:
                    cpu=states[str(layer)];x=cpu.to('cuda:0');small_gpu_parity(x)
                    for head in range(32):
                        s=cpu[0,head].double();rm=s.abs().mean(-1);cm=s.abs().mean(-2);rmedian=rm.median().item();cmedian=cm.median().item();rms=s.square().mean().sqrt().item()
                        dist.append(dict(doc=doc,token=position,layer=layer,head=head,doc_id=document(doc)[0]['document_id'],shape=[128,128],dtype='torch.float32',state_sha256=tsha(cpu[0,head]),coordinate='actual_H_FP',max_abs=s.abs().max().item(),RMS=rms,norm=s.square().sum().sqrt().item(),nonfinite=int((~torch.isfinite(s)).sum()),row_mean_abs=quantile(rm),column_mean_abs=quantile(cm),row_max_over_mean=float(rm.max()/rm.mean()) if rm.mean()>0 else 0.,row_p99_over_median=float(torch.quantile(rm,.99)/rmedian) if rmedian>0 else (0. if rm.max()==0 else None),column_max_over_mean=float(cm.max()/cm.mean()) if cm.mean()>0 else 0.,outlier_max_over_RMS=float(s.abs().max()/rms) if rms else 0.))
                    for codec in C.NAMES:
                        CURRENT_CONTEXT.update(phase='a1',doc=doc,token=position,layer=layer,codec=codec)
                        e,raw=C.trace_encode(x,codec);y=e.decode();check(torch.isfinite(y).all(),'nonfinite snapshot decode',doc=doc,token=position,layer=layer,codec=codec)
                        out=y.cpu();codes=e.codes.cpu();cols=e.columns.float().cpu();rs=e.rows.float().cpu() if e.rows is not None else None
                        step=(rs*cols) if rs is not None else cols.expand_as(cpu)
                        levels=(cpu/rs)/cols if rs is not None else cpu/cols
                        rawcpu=[a.cpu() for a in raw]
                        affected=torch.zeros_like(cpu,dtype=torch.bool)
                        rowhits=0;colhits=0
                        for rawindex,a in enumerate(rawcpu):
                            bad=(a<2**-24)|(a>65504.)
                            if codec==C.NAMES[3] and rawindex==0:rowhits+=int(bad.sum())
                            else:colhits+=int(bad.sum())
                            affected |= bad.expand_as(cpu)
                        for head in range(32):
                            original=cpu[0,head].double();err=out[0,head].double()-original;energy=original.square().sum();mask=affected[0,head]&(original!=0);ae=original[mask].square().sum();ratio=float(ae/energy) if energy else 0.;am=float(err[mask].square().sum()/ae) if ae else 0.
                            rh=sum(int(((a<2**-24)|(a>65504.))[0,head].sum()) for j,a in enumerate(rawcpu) if codec==C.NAMES[3] and j==0)
                            ch=sum(int(((a<2**-24)|(a>65504.))[0,head].sum()) for j,a in enumerate(rawcpu) if not(codec==C.NAMES[3] and j==0))
                            record=dict(doc=doc,token=position,layer=layer,head=head,codec=codec,affected_energy_fraction=ratio,affected_relative_mse=am,row_raw_bound_hits=rh,column_raw_bound_hits=ch,range_mask='union of all preclamp row/column scale calls in unchanged fit; nonzero S only',PASS=not(ratio>=1e-4 and am>1e-2))
                            safety.append(record)
                            if not record['PASS']:
                                torch.save(dict(input=cpu[0,head].clone(),record=record),ROOT/'analysis/SCALE_FIRST_FAILURE_INPUT.pt')
                                save('analysis/scale_safety.json',dict(status='FAIL',first_failure=record,observations=safety))
                                raise GateFailure(dict(reason='frozen FP16 scale safety STOP',**record))
                            rows.append(dict(doc=doc,token=position,layer=layer,head=head,codec=codec,relative_mse=float(err.square().sum()/energy) if energy else 0.,absolute_mse=err.square().mean().item(),max_abs_error=err.abs().max().item(),p99_abs_error=torch.quantile(err.abs().flatten(),.99).item(),zero_code_fraction=(codes[0,head]==0).double().mean().item(),saturation_fraction=(codes[0,head].abs()==127).double().mean().item(),actual_rounded_clamp_fraction=(levels[0,head].round().abs()>127).double().mean().item(),unrounded_over127_fraction=(levels[0,head].abs()>127).double().mean().item(),effective_step=quantile(step[0,head])))
                        del e,raw,y,out,rawcpu
                    del x
                save(f'analysis/progress_a1_doc{doc}_token{position}.json',dict(status='PASS_TO_POSITION',doc=doc,token=position,rows=len(rows)))
                print(json.dumps(dict(phase='a1',doc=doc,token=position,rows=len(rows))),flush=True)
                del states
    save('analysis/real_state_distribution.json',dist);save('analysis/real_state_codec_comparison.json',rows);save('analysis/scale_safety.json',dict(status='PASS',observations=safety));return dict(status='PASS',state_groups=len(dist),codec_rows=len(rows))

def quantize(cache,F,codec,reference=False):
    details={};times=[];before={l:tsha(F.get_state(cache,l)) for l in F.GDN_LAYERS}
    if reference:
        body=(PARENT/'runtime_sources/formal_c128_runner.py').read_text().splitlines()
        lineno=next(i+1 for i,s in enumerate(body) if 'audit["layers"].add(layer_idx)' in s)
        def localtrace(frame,event,arg):
            if event=='line' and frame.f_lineno==lineno:
                d=frame.f_locals;l=d['layer_idx'];details[str(l)]=dict(codes=tsha(d['codes'].to(torch.int8)),scale=tsha(d['scale']),dequant=tsha(d['dequant']))
            return localtrace
        def globaltrace(frame,event,arg):return localtrace if frame.f_code is F.quantize_c128.__code__ else None
        audit={'calls':0,'layers':set(),'scale_shapes':{},'code_min':0,'code_max':0};old=sys.gettrace();sys.settrace(globaltrace)
        try:F.quantize_c128(cache,audit)
        finally:sys.settrace(old)
        check(audit['calls']==1 and audit['layers']==set(F.GDN_LAYERS),'formal writeback counts mismatch')
    else:
        for l in F.GDN_LAYERS:
            state=F.get_state(cache,l);events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
            events[0].record();e,raw=C.trace_encode(state,codec);events[1].record();dq=C.decode_for_writeback(e,state);events[2].record()
            badmask=torch.zeros_like(state,dtype=torch.bool)
            for a in raw:badmask|=((a<2**-24)|(a>65504.)).expand_as(state)
            bound_hits=int(badmask.sum())
            if bound_hits:
                mask=badmask&(state!=0);s=state.double();error=(dq.double()-s);energy=s.square().sum((-1,-2));ae=(s.square()*mask).sum((-1,-2));ae_ratio=torch.where(energy>0,ae/energy,0.);affected=torch.where(ae>0,(error.square()*mask).sum((-1,-2))/ae,0.)
                check(~((ae_ratio>=1e-4)&(affected>1e-2)).any(),'student frozen scale safety STOP',layer=l,codec=codec,context=dict(CURRENT_CONTEXT),affected_energy_fraction=ae_ratio.cpu().tolist(),affected_relative_MSE=affected.cpu().tolist())
                del s,error,energy,ae,ae_ratio,affected
            effective_levels=(state/e.rows.float())/e.columns.float() if e.rows is not None else state/e.columns.float()
            rounded_clamp=float((effective_levels.round().abs()>127).float().mean());unrounded_over=float((effective_levels.abs()>127).float().mean())
            events[3].record();state.copy_(dq.to(state.dtype));events[4].record();events[4].synchronize()
            check(torch.isfinite(state).all(),'nonfinite student writeback',layer=l,codec=codec)
            details[str(l)]=dict(codes=tsha(e.codes),scale=tsha(e.columns),dequant=tsha(dq),code_min=int(e.codes.min()),code_max=int(e.codes.max()),row_min=None if e.rows is None else float(e.rows.min()),row_max=None if e.rows is None else float(e.rows.max()),column_min=float(e.columns.min()),column_max=float(e.columns.max()),encoded_tensor_bytes=e.nbytes(),raw_bound_affected_values=bound_hits,zero_code_fraction=float((e.codes==0).float().mean()),saturation_fraction=float((e.codes.abs()==127).float().mean()),actual_rounded_clamp_fraction=rounded_clamp,unrounded_over127_fraction=unrounded_over)
            times.append(dict(layer=l,encode_ms=events[0].elapsed_time(events[1]),decode_ms=events[1].elapsed_time(events[2]),writeback_ms=events[3].elapsed_time(events[4])))
            del e,dq,events,raw,badmask,effective_levels
    after={l:tsha(F.get_state(cache,l)) for l in F.GDN_LAYERS}
    return details,times,before,after

def checkpoint_metrics(doc,pos,cache,readouts,F,logits):
    target=torch.load(ROOT/f'raw_snapshots/doc{doc}_token{pos}_FP_H.pt',map_location='cpu',weights_only=True)
    refout=torch.load(ROOT/f'raw_snapshots/doc{doc}_token{pos}_outputs.pt',map_location='cpu',weights_only=True)
    values=[]
    for l in F.GDN_LAYERS:
        student=F.get_state(cache,l).detach().cpu();original=target[str(l)];d=(student.double()-original.double());den=original.double().square().sum()
        ro=readouts[l].double();rf=refout['readouts'][l].double();rn=rf.square().sum()
        for head in range(32):
            s=original[0,head].double();error=student[0,head].double()-s;en=s.square().sum();r=rf[...,head,:];re=ro[...,head,:]-r;enr=r.square().sum()
            values.append(dict(layer=l,head=head,state_relative_mse=float(error.square().sum()/en) if en else 0.,current_readout_relative_mse=float(re.square().sum()/enr) if enr else 0.,state_dtype=str(student.dtype),state_shape=list(student.shape)))
    return values

def trajectory(H,F,doc,codec,observer,reference=False,tag=''):
    a,ids=document(doc);cache=None;rows=[];checkpoint_rows={};timings=[];previous={};start=time.monotonic()
    with torch.inference_mode():
        for position,token in enumerate(ids,1):
            CURRENT_CONTEXT.update(doc=doc,token=position,codec=codec,observer=observer,reference=reference)
            H.previous=previous;H.mode='H';H.observer=observer;H.position=position;H.readouts={};H.details={}
            t0=time.monotonic();out=H.model(input_ids=torch.tensor([[int(token)]],device='cuda:0'),past_key_values=cache,use_cache=True);cache=out.past_key_values;torch.cuda.synchronize();forward_s=time.monotonic()-t0
            check(torch.isfinite(out.logits).all(),'nonfinite student logits',doc=doc,token=position,codec=codec)
            helper=dict(H.details)
            if codec is not None:qdq,events,before,previous=quantize(cache,F,codec,reference)
            else:qdq={};events=[];before={};previous={}
            dg,cache_info=cache_digest(cache)
            record=dict(token=position,input_id=token,logits=tsha(out.logits),cache=dg,QDQ=qdq,helper=helper,pre_QDQ={str(l):v for l,v in before.items()},post_QDQ={str(l):v for l,v in previous.items()})
            rows.append(record);timings.append(dict(token=position,full_model_forward_ms=1000*forward_s,encode_decode_writeback=events,full_token_with_audit_ms=1000*(time.monotonic()-t0)))
            if position in SCORES:
                checkpoint_rows[str(position)]=dict(logits=out.logits.detach().cpu().float(),state_metrics=checkpoint_metrics(doc,position,cache,H.readouts,F,out.logits) if observer else None,readouts=H.readouts if observer else None)
            del out
    check(all(H.calls[(p,l)]>=1 for p in range(1,129) for l in F.GDN_LAYERS),'missing helper invocation')
    p=ROOT/f'analysis/{tag}_doc{doc}_checkpoints.pt';torch.save(checkpoint_rows,p)
    result=dict(doc_id=a['document_id'],codec=codec,observer=observer,reference=reference,tokens=rows,checkpoint_file=str(p.relative_to(ROOT)),timings=timings,trajectory_wall_s=time.monotonic()-start,writebacks_per_token=24 if codec else 0)
    del cache;gc.collect();torch.cuda.empty_cache();return result

def baseline(phase):
    model,H,F,N,loadinfo=load();results={};fp=read(ROOT/'analysis/a0_1.json')
    with torch.inference_mode():
        for doc in (1,2):
            if phase=='b0_1':
                off=trajectory(H,F,doc,None,False,False,'FP_H_OFF')
                for x,y in zip(off['tokens'],fp['documents'][str(doc)]['tokens']):check(x['logits']==y['H']['logits'] and x['cache']==y['H']['cache'],'FP_H observer OFF/ON changed outputs',doc=doc,token=x['token'])
                ref=trajectory(H,F,doc,C.NAMES[0],False,True,'A_REFERENCE_OFF')
                on=trajectory(H,F,doc,C.NAMES[0],True,True,'A_REFERENCE_ON')
                adapter=trajectory(H,F,doc,C.NAMES[0],True,False,'A_ADAPTER_ON')
                for x,y,z in zip(ref['tokens'],on['tokens'],adapter['tokens']):
                    for key in ('token','input_id','logits','cache','QDQ','pre_QDQ','post_QDQ'):
                        if key=='QDQ':
                            zq={k:{a:v[a] for a in ('codes','scale','dequant')} for k,v in z[key].items()};check(x[key]==y[key]==zq,'canonical runtime codes/scales/dequant mismatch',doc=doc,token=x['token'])
                        else:check(x[key]==y[key]==z[key],'canonical runtime/observer parity mismatch',doc=doc,token=x['token'],object=key)
                    check(y['helper']==z['helper'],'QK/update/readout/cache-consumption parity mismatch',doc=doc,token=x['token'])
                results[str(doc)]=dict(reference_off=ref,reference_on=on,adapter_on=adapter,FP_H_OFF_ON='BYTE_EXACT')
            else:
                ref=trajectory(H,F,doc,C.NAMES[0],False,True,'A_REFERENCE_FRESH')
                first=read(ROOT/'analysis/b0_1.json')['documents'][str(doc)]['reference_off']
                check(ref['tokens']==first['tokens'],'A runtime fresh-process identity mismatch',doc=doc)
                results[str(doc)]=dict(reference_off=ref,fresh_process_exact=True)
            print(json.dumps(dict(phase=phase,doc=doc,status='PASS')),flush=True)
    save(f'analysis/{phase}.json',dict(status='PASS',phase=phase,documents=results,load=loadinfo,HISTORICAL_END_TO_END_REPRODUCTION='NOT_RUN',independent_reference='unchanged formal GPU quantize_c128 function'))
    H.close();del model,H;gc.collect();torch.cuda.empty_cache();return dict(status='PASS')

def students():
    check(read(ROOT/'analysis/b0_2.json')['status']=='PASS','B0 prerequisite')
    model,H,F,N,loadinfo=load();results={}
    for codec in (C.NAMES[3],C.NAMES[1],C.NAMES[2]):
        results[codec]={}
        for doc in (1,2):
            result=trajectory(H,F,doc,codec,True,False,codec)
            results[codec][str(doc)]=result
            print(json.dumps(dict(phase='students',codec=codec,doc=doc,status='PASS')),flush=True)
            save(f'analysis/{codec}_doc{doc}.json',result)
    save('analysis/students.json',dict(status='PASS',results=results,load=loadinfo,order=['A_REUSED_B0',C.NAMES[3],C.NAMES[1],C.NAMES[2]],physical_INT8_runtime='NOT_QUALIFIED',cache='same original FP32 QDQ lifecycle for all four codecs'))
    H.close();del model,H;gc.collect();torch.cuda.empty_cache();return dict(status='PASS')

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['a0_1','a0_2','a1','b0_1','b0_2','students']);args=p.parse_args();start=time.time()
    check(not (ROOT/'FIRST_FAILURE.json').exists(),'attempt already failed; no GPU retry permitted')
    try:
        env=configure();save(f'analysis/ENVIRONMENT_{args.phase}.json',env)
        if args.phase.startswith('a0'):result=a0(args.phase)
        elif args.phase=='a1':result=a1()
        elif args.phase.startswith('b0'):result=baseline(args.phase)
        else:result=students()
        save(f'analysis/RECEIPT_{args.phase}.json',dict(status='PASS',result=result,wall_seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),explicit_codec_scalar_sync_counter=C.SYNC_COUNT))
        print(json.dumps(dict(phase=args.phase,status='PASS',wall_seconds=time.time()-start)),flush=True)
    except BaseException as exc:
        failure=dict(status='FAIL_AND_STOP',phase=args.phase,error=repr(exc),first_failure=getattr(exc,'record',None),current_context=dict(CURRENT_CONTEXT),traceback=traceback.format_exc(),wall_seconds=time.time()-start,no_GPU_retry=True)
        if not (ROOT/'FIRST_FAILURE.json').exists():save('FIRST_FAILURE.json',failure)
        print(json.dumps(failure),flush=True);raise

if __name__=='__main__':main()
