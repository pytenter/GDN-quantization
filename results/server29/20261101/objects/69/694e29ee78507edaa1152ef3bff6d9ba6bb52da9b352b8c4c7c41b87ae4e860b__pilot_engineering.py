"""Additive, fail-closed full-model qualification; historical sources are read-only."""
import ast, copy, gc, hashlib, importlib.util, json, math, os, subprocess, sys, time, traceback
from pathlib import Path

TASK = 'QWEN_GDN_SCALE_FEEDBACK_FULLMODEL_FUNCTIONAL_CAUSAL_PILOT_V1'
REPO = Path('/data/zypan/worktrees/qwen-gdn-scale-feedback-fullmodel-functional-causal-pilot-v1')
ROOT = REPO/'experiments'/TASK
V3 = Path('/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3')
sys.path.insert(0, str(V3/'scripts'))
import numerical_recovery as N
import scale_science as S
import matched_history as H

def check(ok, message):
    if not ok: raise RuntimeError(message)

def save(name, obj):
    path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def sha(path): return N.sha(path)

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def structure(x, seen=None):
    import torch
    if seen is None: seen={}
    if torch.is_tensor(x): return {'tensor_shape':list(x.shape),'dtype':str(x.dtype)}
    if x is None or isinstance(x,(bool,int,float,str)): return x
    if id(x) in seen: return {'alias':seen[id(x)]}
    seen[id(x)]=len(seen)
    if isinstance(x,dict):return {str(k):structure(v,seen) for k,v in sorted(x.items(),key=lambda kv:str(kv[0]))}
    if isinstance(x,(list,tuple)): return {'type':type(x).__name__,'values':[structure(v,seen) for v in x]}
    if hasattr(x,'__dict__'):return {'type':type(x).__module__+'.'+type(x).__name__,'attributes':structure(vars(x),seen)}
    return repr(x)

def runtime_state(model):
    return {'rope_deltas':getattr(model.model,'rope_deltas',None),
            'rotary_buffers':{k:v for k,v in model.named_buffers() if 'rotary_emb' in k}}

def fingerprint(cache, model, logits=None, codes=None):
    result={'cache_tensors':S.snapshot_hash(cache),'cache_metadata':structure(cache),
            'runtime_tensors':S.snapshot_hash(runtime_state(model)),'runtime_metadata':structure(runtime_state(model))}
    if logits is not None:result['logits']=N.tsha(logits)
    if codes is not None: result['codes']=codes
    return result

def snapshot(cache, model, consumed):
    return H.device_tree({'cache':cache,'runtime':runtime_state(model),'consumed':consumed,
                          'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},'cpu')

def restore(payload, model):
    q=H.device_tree(payload,'cuda:0')
    model.model.rope_deltas=q['runtime']['rope_deltas']
    buffers=dict(model.named_buffers())
    for name,value in q['runtime']['rotary_buffers'].items():buffers[name].copy_(value)
    torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state(payload['cuda_rng'])
    return q['cache']

class Harness(S.Harness):
    def __init__(self,*args):
        self.capture=False;self.pre={};self.outputs={}
        super().__init__(*args)

    def wrap(self, fn):
        original=super().wrap(fn)
        def invoke(*args,**kw):
            result,state=original(*args,**kw)
            if self.capture:
                self.pre[self.active]=state.detach().clone()
                self.outputs[self.active]=result.detach().clone()
            return result,state
        return invoke

    def forward(self, token, cache, position, mode='PARENT', clean=None, capture=False, qual=False):
        self.condition='C6R';self.position=position;self.observe=False;self.target=None;self.qual=qual
        self.pre={};self.outputs={};self.capture=capture;self.updated={};self.codes={}
        out=self.model(input_ids=self.t.tensor([[token]],device='cuda:0'),past_key_values=cache,use_cache=True)
        cache=out.past_key_values
        check(int(cache.get_seq_length())==position+1,'actual consumed token/cache position pairing')
        if capture:
            check(set(self.pre)==set(S.LAYERS),'all24 current helper outputs captured')
            for l in S.LAYERS:check(self.t.equal(self.pre[l],self.F.get_state(cache,l)),'current clean pre-QDQ/cache coordinate equality')
        if mode=='CLEAN_FP_REFERENCE':
            check(not qual,'FP reference must not execute QDQ/writeback qualification')
        elif mode=='PARENT':
            # Execute original formal QDQ function unchanged.
            audit={'calls':0,'layers':set(),'scale_shapes':{},'code_min':0,'code_max':0}
            self.F.quantize_c128(cache,audit)
            check(audit['calls']==1 and audit['layers']==set(S.LAYERS),'every-token original Parent QDQ')
        else:
            check(mode in ['NO_SCALE_INFLATION','NO_SCALE_DEFLATION'],'frozen intervention name')
            check(clean is not None and clean.dtype==self.t.float32 and list(clean.shape)==[24,1,32,1,128], 'indexed FP32 clean scales')
            for i,l in enumerate(S.LAYERS):
                state=self.F.get_state(cache,l)
                adaptive,_,_=self.quant.original(state)
                c=clean[i]
                chosen=self.t.minimum(adaptive,c) if mode=='NO_SCALE_INFLATION' else self.t.maximum(adaptive,c)
                ns={'torch':self.t,'source':state.detach().float(),'scale':chosen}
                exec(self.quant.grid_code,ns)
                check(bool(self.t.isfinite(ns['dequant']).all()),'finite effective QDQ state')
                state.copy_(ns['dequant'].to(state.dtype))
        if mode!='CLEAN_FP_REFERENCE' and qual:
            for l in S.LAYERS:check(self.t.equal(self.F.get_state(cache,l),self.updated[l]),'original canonical QDQ exact writeback')
            self.previous['C6R']={l:self.F.get_state(cache,l).detach().clone() for l in S.LAYERS}
        check(bool(self.t.isfinite(out.logits).all()),'finite full-model logits')
        return out,cache

def kl32(fp, student):
    a=fp.float().reshape(-1);b=student.float().reshape(-1)
    check(a.numel()==248320 and a.shape==b.shape,'full vocabulary248320 pairing')
    la=torch.log_softmax(a,dim=-1);lb=torch.log_softmax(b,dim=-1)
    value=(la.exp()*(la-lb)).sum(dtype=torch.float32)
    check(bool(torch.isfinite(value)),'finite full-vocabulary FP32 KL')
    return float(value)

def kl64(fp, student):
    a=fp.detach().double().cpu().reshape(-1);b=student.detach().double().cpu().reshape(-1)
    la=a-torch.logsumexp(a,0);lb=b-torch.logsumexp(b,0)
    return float((la.exp()*(la-lb)).sum())

def setup():
    global torch
    torch,F,core,policy,model,tokenizer,started=S.load(0)
    check(sha(V3/'NEW_NUMERICAL_CONTRACT.json')=='0031725d882b758ab59d292124b8604638f21a93a7e26a0bd60a74cf3716089a','sealed V3 numerical contract identity')
    check(sha(N.ARCH)==N.ARCH_SHA and sha(N.FORMAL)==N.FORMAL_SHA,'unchanged numerical/quantizer source')
    return F,core,policy,model,tokenizer,started

def small(index):
    F,core,policy,model,tokenizer,started=setup()
    d=policy['documents'][0];ids=N.read(d['token_path'])[:36]
    check(tokenizer(Path(d['text_path']).read_text(),add_special_tokens=False).input_ids[:36]==ids,'old engineering text/token identity')
    h=Harness(model,F,core,torch);teacher=[];teacher_trace=[]
    with torch.inference_mode():
        for route in ['oracle','compensated']:
            h.qmod.torch_chunk_gated_delta_rule=h.wrap(N.chunk_binding(h.qmod,route))
            h.previous={};cache=None;trace=[]
            for p,token in enumerate(ids):
                out,cache=h.forward(token,cache,p,mode='CLEAN_FP_REFERENCE',capture=True)
                scale=torch.stack([h.quant.original(h.pre[l])[0].cpu() for l in S.LAYERS])
                for i,l in enumerate(S.LAYERS):check(torch.equal(scale[i],h.quant.original(F.get_state(cache,l))[0].cpu()),'scale token/layer/group cache index parity')
                trace.append(digest({'trajectory':fingerprint(cache,model,out.logits),
                    'pre':{str(l):N.tsha(h.pre[l]) for l in S.LAYERS},'out':{str(l):N.tsha(h.outputs[l]) for l in S.LAYERS},'scale':N.tsha(scale)}))
                if route=='oracle':teacher.append({'logits':out.logits.detach().float().cpu(),'scale':scale})
            if route=='oracle':teacher_trace=trace
            else:check(trace==teacher_trace,'same-C6R full-model mathematical scan oracle/candidate byte parity')
            del cache,out;gc.collect();torch.cuda.empty_cache()
        baseline=None;parent_logits=[];fork_receipt=None
        # Original V3 Parent, then new Parent with observer dataflow OFF/ON.
        for variant in ['original','new_off','new_on']:
            h.previous={};cache=None;trace=[];anchor=None
            for p,token in enumerate(ids):
                if variant=='original':out,cache=h.step('C6R',token,cache,p,qual=True)
                else:
                    out,cache=h.forward(token,cache,p,capture=variant=='new_on',qual=True)
                    if variant=='new_on':
                        for i,l in enumerate(S.LAYERS):
                            adaptive,_,_=h.quant.original(h.pre[l]);clean=teacher[p]['scale'][i].to('cuda:0')
                            _=torch.minimum(adaptive,clean);_=torch.maximum(adaptive,clean)
                            check(bool(torch.isfinite(adaptive).all()),'finite observer scale')
                trace.append(digest(fingerprint(cache,model,out.logits,h.codes)))
                if variant=='original':parent_logits.append(out.logits.detach().float().cpu())
                if variant=='new_on' and p==31:
                    before=fingerprint(cache,model)
                    anchor=snapshot(cache,model,32)
                    check(before['cache_tensors']==S.snapshot_hash(anchor['cache']),'lossless CPU snapshot tensors')
                    cpu_ptr={t.data_ptr() for _,t in S.tree_tensors(anchor)}
                    gpu_ptr={t.data_ptr() for _,t in S.tree_tensors(cache)}
                    check(not cpu_ptr.intersection(gpu_ptr),'snapshot independent tensor storage')
            if variant=='original':baseline=trace
            else:check(trace==baseline,'Parent original/observer OFF/ON complete trajectory byte parity')
            if variant=='new_on':
                check(anchor is not None,'complete32 engineering snapshot')
                unchanged=digest(fingerprint(anchor['cache'],model))
                del cache;gc.collect();torch.cuda.empty_cache()
                cache=restore(anchor,model)
                h.previous['C6R']={l:F.get_state(cache,l).detach().clone() for l in S.LAYERS}
                restored=[]
                for p in range(32,36):
                    out,cache=h.forward(ids[p],cache,p,qual=True)
                    restored.append(digest(fingerprint(cache,model,out.logits,h.codes)))
                check(restored==baseline[32:],'uninterrupted/restored Parent exact next-token parity')
                # Immutable CPU source must not be changed by a resumed branch.
                check(S.snapshot_hash(anchor['cache'])==before['cache_tensors'],'anchor no shared mutable cache')
                fork_receipt={'status':'PASS','anchor_consumed':32,'resumed_tokens':4,'baseline':baseline[32:],'snapshot_identity':digest(before)}
            del cache,out;gc.collect();torch.cuda.empty_cache()
        # Exact original AST vs clamp policy, equal-scales, both signs and rounding ties.
        state=torch.linspace(-4,4,128,device='cuda:0').reshape(1,1,128,1).expand(1,32,128,128).clone()
        adaptive,codes,dequant=h.quant.original(state)
        for factor in [.5,1.,2.]:
            clean=adaptive*factor
            for arm in ['NO_SCALE_INFLATION','NO_SCALE_DEFLATION']:
                effective=torch.minimum(adaptive,clean) if arm=='NO_SCALE_INFLATION' else torch.maximum(adaptive,clean)
                ns={'torch':torch,'source':state.float(),'scale':effective};exec(h.quant.grid_code,ns)
                expected=torch.round(state.float()/effective).clamp(-127,127)*effective
                check(torch.equal(ns['dequant'],expected),'min/max original code-round-clamp-writeback semantics')
                if factor==1:check(torch.equal(ns['dequant'],dequant),'equal clean/student exact original quantizer')
        values=[];errors=[]
        for a,b in zip(teacher,parent_logits):
            v=kl32(a['logits'].to('cuda:0'),b.to('cuda:0'));ref=kl64(a['logits'],b)
            values.append(v);errors.append(abs(v-ref))
        for shift in [0.,1000.,-1000.]:
            a=teacher[-1]['logits']+shift;b=parent_logits[-1]+shift
            errors.append(abs(kl32(a.to('cuda:0'),b.to('cuda:0'))-kl64(a,b)))
        check(kl32(teacher[0]['logits'].to('cuda:0'),teacher[0]['logits'].to('cuda:0'))==0,'identical-distribution KL zero')
        aggregate=math.fsum(values)/len(values)
        independent=float(torch.tensor(values,dtype=torch.float64).sum()/len(values))
        reducer_error=abs(aggregate-independent)
        resolution=2*max(errors)+2*reducer_error
        receipt={'status':'PASS','process_index':index,'tokens':36,'all24_layers':True,
            'oracle_compensated_trace':teacher_trace,'parent_observer_trace':baseline,
            'coordinate_and_index':'current token; same frozen C6R helper/cache; FP32 [24,1,32,1,128]',
            'checkpoint_short':fork_receipt,'KL_max_abs_FP32_vs_FP64_error':max(errors),
            'KL_time_reducer_error':reducer_error,'prospective_delta_resolution':resolution,
            'seconds':time.monotonic()-started,'GPUh':(time.monotonic()-started)/3600,
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'runtime':F.runtime_version(),'GPU_UUID':S.UUIDS[0],'code_sha256':sha(__file__)}
        save(f'qualification/small_{index}.json',receipt)
    h.close()
    print('SMALL_QUALIFICATION_PASS',index,flush=True)

if __name__=='__main__':
    try:small(int(sys.argv[1]))
    except Exception as exc:
        if not (ROOT/'FIRST_FAILURE_GATE.json').exists():save('FIRST_FAILURE_GATE.json',{
            'status':'FAIL_AND_STOP','stage':'STAGE0_SMALL_FULLMODEL_QUALIFICATION','utc':N.now(),
            'exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'formal_outcomes_produced':False,'no_retry':True})
        raise
