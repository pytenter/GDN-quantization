#!/usr/bin/env python3
"""Read-only fixed-token mechanism capture for Native/H/C5/C6.

No sampling, scoring, optimizer, checkpoint search, or parameter mutation occurs.
Controlled mode preserves FP histories in each basis and only observes QDQ.
Recurrent mode applies canonical C128 QDQ writeback after every token update.
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json, math, os, sys, time, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

TASK="QWEN_GDN_HADAMARD_VS_LEARNED_ROTATION_MECHANISM_CLOSURE_V1"
REPO=Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
ROOT=REPO/"experiments/qwen_gdn"/TASK
MODEL=Path("/data/zypan/modelscope_models/Qwen3.5-9B")
TRANSFORMERS_SRC=Path("/data/zypan/transformers-qwen35/src")
STATE_QUANT_SRC=Path("/data/zypan/experiments/qwen35_gdn_quant")
TEACHER_ROOT=Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
DEP=REPO/"experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment"
GDN_LAYERS=tuple(i for i in range(32) if i%4!=3)
EPS=1e-12
LOG_LO,LOG_HI,NBINS=-50.0,30.0,256
PRIMARY=("range","absmax","scale","papr_linear","papr_power","qdq_relative_mse","qdq_relative_l2")
ALL_METRICS=("minimum","maximum","range","absmax","mean","rms","std","scale","papr_linear","papr_power",
             "qdq_mse","qdq_relative_mse","qdq_l2","qdq_relative_l2","qdq_max_abs_error","snr_db",
             "qcode_min","qcode_max","saturation_count","saturation_fraction","unique_qcode_count")
EXPECTED={"C5":"0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
          "C6":"67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c"}

def dump(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n")
def file_sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def load_jsonl(p):
    with p.open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def hadamard(n=128,device=None):
    h=torch.ones((1,1),dtype=torch.float32,device=device)
    while h.shape[0]<n:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return h/math.sqrt(n)
def qdq_metrics(x):
    # x: [C,H,K,V]; canonical groups are K columns => one group per C,H,V.
    xf=x.detach().float()
    minimum=xf.amin(dim=-2); maximum=xf.amax(dim=-2); rng=maximum-minimum
    absmax=xf.abs().amax(dim=-2); mean=xf.mean(dim=-2); power=xf.square().mean(dim=-2)
    rms=power.sqrt(); std=((xf-mean.unsqueeze(-2)).square().mean(dim=-2)).sqrt()
    scale=absmax.clamp_min(EPS)/127.0
    codes=torch.round(xf/scale.unsqueeze(-2)).clamp(-127,127)
    deq=codes*scale.unsqueeze(-2); err=deq-xf; mse=err.square().mean(dim=-2)
    relmse=mse/power.clamp_min(EPS); l2=err.square().sum(dim=-2).sqrt()
    rell2=l2/xf.square().sum(dim=-2).sqrt().clamp_min(EPS)
    maxerr=err.abs().amax(dim=-2); snr=10*torch.log10(power.clamp_min(EPS)/mse.clamp_min(EPS))
    sorted_codes=torch.sort(codes,dim=-2).values
    unique=1+(sorted_codes[:,:,1:,:]!=sorted_codes[:,:,:-1,:]).sum(dim=-2)
    sat=(codes.abs()==127).sum(dim=-2)
    return {"minimum":minimum,"maximum":maximum,"range":rng,"absmax":absmax,"mean":mean,
            "rms":rms,"std":std,"scale":scale,"papr_linear":absmax/rms.clamp_min(EPS),
            "papr_power":absmax.square()/power.clamp_min(EPS),"qdq_mse":mse,
            "qdq_relative_mse":relmse,"qdq_l2":l2,"qdq_relative_l2":rell2,
            "qdq_max_abs_error":maxerr,"snr_db":snr,"qcode_min":codes.amin(dim=-2),
            "qcode_max":codes.amax(dim=-2),"saturation_count":sat.float(),
            "saturation_fraction":sat.float()/128.0,"unique_qcode_count":unique.float()},deq

class GroupAccumulator:
    def __init__(self,conditions):
        self.conditions=tuple(conditions); c=len(conditions); h=32
        self.data={}
        for layer in GDN_LAYERS:
            self.data[layer]={}
            for m in ALL_METRICS:
                self.data[layer][m]={"n":np.zeros((c,h),np.int64),"sum":np.zeros((c,h)),"sumsq":np.zeros((c,h)),
                                     "min":np.full((c,h),np.inf),"max":np.full((c,h),-np.inf)}
                if m in PRIMARY:self.data[layer][m]["hist"]=np.zeros((c,h,NBINS),np.int64)
    def update(self,layer,metrics):
        # tensors [C,T,H,V]
        C,T,H,V=next(iter(metrics.values())).shape
        for name,x in metrics.items():
            q=x.detach().float()
            rec=self.data[layer][name]
            rec["n"] += T*V
            rec["sum"] += q.sum(dim=(1,3)).double().cpu().numpy()
            rec["sumsq"] += q.square().sum(dim=(1,3)).double().cpu().numpy()
            rec["min"] = np.minimum(rec["min"],q.amin(dim=(1,3)).cpu().numpy())
            rec["max"] = np.maximum(rec["max"],q.amax(dim=(1,3)).cpu().numpy())
            if name in PRIMARY:
                z=torch.log2(q.clamp_min(2.0**LOG_LO))
                idx=((z-LOG_LO)/(LOG_HI-LOG_LO)*NBINS).floor().clamp(0,NBINS-1).long()
                off=torch.arange(C*H,device=q.device).reshape(C,1,H,1)*NBINS
                hist=torch.bincount((idx+off).reshape(-1),minlength=C*H*NBINS).reshape(C,H,NBINS)
                rec["hist"] += hist.cpu().numpy()
    @staticmethod
    def hq(hist,q):
        n=int(hist.sum())
        if not n:return None
        k=max(0,min(n-1,int(math.ceil(q*n))-1)); i=int(np.searchsorted(np.cumsum(hist),k+1))
        return float(2**(LOG_LO+(i+.5)*(LOG_HI-LOG_LO)/NBINS))
    @staticmethod
    def hcvar(hist,q):
        n=int(hist.sum()); target=max(1,int(math.ceil((1-q)*n))); centers=2**(LOG_LO+(np.arange(NBINS)+.5)*(LOG_HI-LOG_LO)/NBINS)
        remaining=target; total=0.0
        for i in range(NBINS-1,-1,-1):
            take=min(remaining,int(hist[i])); total+=take*centers[i]; remaining-=take
            if remaining<=0:break
        return float(total/target)
    def rows(self,sample_id):
        out=[]
        for layer in GDN_LAYERS:
            for ci,cond in enumerate(self.conditions):
                for head in range(32):
                    row={"sample_id":sample_id,"condition":cond,"layer":layer,"head":head,
                         "quantile_method":"fixed log2 histogram midpoint; 256 bins over [-50,30]"}
                    for m,rec in self.data[layer].items():
                        n=int(rec["n"][ci,head]); mean=float(rec["sum"][ci,head]/max(n,1)); var=max(0.0,float(rec["sumsq"][ci,head]/max(n,1)-mean*mean))
                        row[f"{m}_n"]=n; row[f"{m}_mean"]=mean; row[f"{m}_std"]=math.sqrt(var)
                        row[f"{m}_min"]=float(rec["min"][ci,head]); row[f"{m}_max"]=float(rec["max"][ci,head])
                        if m in PRIMARY:
                            hist=rec["hist"][ci,head]
                            for qn,q in (("median",.5),("p90",.9),("p95",.95),("p99",.99),("p999",.999)):
                                row[f"{m}_{qn}"]=self.hq(hist,q)
                            row[f"{m}_cvar95"]=self.hcvar(hist,.95); row[f"{m}_cvar99"]=self.hcvar(hist,.99)
                    out.append(row)
        return out

class Patch:
    def __init__(self,model,conditions,mode,bank):
        self.model=model; self.conditions=tuple(conditions); self.mode=mode; self.bank=bank
        self.device=next(model.parameters()).device; self.H=hadamard(device=self.device)
        self.current_layer=None; self.capture=False; self.saved=[]; self.handles=[]; self.acc=None
        self.chunk={}; self.post={}; self.layers_seen=set(); self.calls=0
    def R(self,cond,layer):
        if cond in ("FP","Native"):return None
        if cond=="Hadamard":return self.H
        return self.H.matmul(self.bank[cond][str(layer)].to(self.device))
    def transform(self,x,use_norm):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        if use_norm:x=qmod.l2norm(x,dim=-1,eps=1e-6)
        x=x.clone()
        for i,c in enumerate(self.conditions):
            r=self.R(c,int(self.current_layer))
            if r is not None:x[i]=x[i].float().matmul(r)
        return x
    def recover(self,s,ci,layer):
        r=self.R(self.conditions[ci],layer)
        return s[ci].float() if r is None else r.matmul(s[ci].float())
    def begin(self,acc):self.capture=True; self.acc=acc; self.chunk={}; self.post={}
    def _fast_recurrent(self,query,key,value,g,beta,initial_state,output_final_state,use_norm,kwargs):
        query=self.transform(query,use_norm); key=self.transform(key,use_norm)
        return self.orig_rec(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel=False,**kwargs)
    def _run(self,query,key,value,g,beta,initial_state,output_final_state,use_norm):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        initial_dtype=query.dtype
        if use_norm: query=qmod.l2norm(query,dim=-1,eps=1e-6); key=qmod.l2norm(key,dim=-1,eps=1e-6)
        query=self.transform(query,False); key=self.transform(key,False)
        query,key,value,beta,g=[x.transpose(1,2).contiguous().float() for x in (query,key,value,beta,g)]
        B,H,T,K=key.shape; V=value.shape[-1]; query=query*(K**-.5)
        state=torch.zeros(B,H,K,V,device=value.device,dtype=torch.float32) if initial_state is None else initial_state.float().clone()
        core=torch.empty(B,H,T,V,device=value.device,dtype=torch.float32)
        group_store={m:[] for m in ALL_METRICS}; state_err=[]; readout=[]; query_prev=[]
        layer=int(self.current_layer)
        for t in range(T):
            qt=query[:,:,t]; kt=key[:,:,t]; vt=value[:,:,t]
            fp_prev=self.recover(state,0,layer) if self.conditions[0]=="FP" else None
            if fp_prev is not None:
                qp=[]
                for ci in range(1,B):
                    prev=self.recover(state,ci,layer)
                    # Actual pre-update readout path difference using condition-specific q/state.
                    qp.append(((state[ci]*qt[ci].unsqueeze(-1)).sum(-2)-(state[0]*qt[0].unsqueeze(-1)).sum(-2)).square().sum().sqrt())
                query_prev.append(torch.stack(qp))
            gt=g[:,:,t].exp().unsqueeze(-1).unsqueeze(-1); bt=beta[:,:,t].unsqueeze(-1)
            state=state*gt; memory=(state*kt.unsqueeze(-1)).sum(-2); delta=(vt-memory)*bt
            state=state+kt.unsqueeze(-1)*delta.unsqueeze(-2); core[:,:,t]=(state*qt.unsqueeze(-1)).sum(-2)
            qm,deq=qdq_metrics(state)
            # Controlled mode observes canonical QDQ but never writes it back.
            for m in ALL_METRICS:group_store[m].append(qm[m])
            if self.mode=="recurrent":
                for ci,c in enumerate(self.conditions):
                    if c!="FP":state[ci]=deq[ci]
            if self.conditions[0]=="FP":
                fp=self.recover(state,0,layer); se=[]; rd=[]
                for ci in range(1,B):
                    e=self.recover(state,ci,layer)-fp; se.append(e.square().sum().sqrt())
                    rd.append((core[ci,:,t]-core[0,:,t]).square().sum().sqrt())
                state_err.append(torch.stack(se)); readout.append(torch.stack(rd))
        metrics={m:torch.stack(v,dim=1) for m,v in group_store.items()} # [C,T,H,V]
        self.acc.update(layer,metrics)
        self.chunk[layer]={
            "scale_median":metrics["scale"].flatten(2).median(-1).values,
            "scale_p95":torch.quantile(metrics["scale"].flatten(2),.95,dim=-1),
            "scale_p99":torch.quantile(metrics["scale"].flatten(2),.99,dim=-1),
            "scale_cvar99":self._cvar(metrics["scale"].flatten(2),.99),
            "papr_median":metrics["papr_linear"].flatten(2).median(-1).values,
            "papr_p99":torch.quantile(metrics["papr_linear"].flatten(2),.99,dim=-1),
            "qdq_rel":metrics["qdq_relative_l2"].mean(dim=(2,3)),
            "state_error":torch.stack(state_err,dim=1) if state_err else None,
            "readout_error":torch.stack(readout,dim=1) if readout else None,
            "query_prev_error":torch.stack(query_prev,dim=1) if query_prev else None,
        }
        self.layers_seen.add(layer); self.calls+=1
        return core.transpose(1,2).contiguous().to(initial_dtype),(state if output_final_state else None)
    @staticmethod
    def _cvar(x,q):
        k=max(1,int(math.ceil((1-q)*x.shape[-1]))); return torch.topk(x,k,dim=-1).values.mean(-1)
    def recurrent(self,query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel=False,**kwargs):
        if not self.capture:return self._fast_recurrent(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel,kwargs)
        return self._run(query,key,value,g,beta,initial_state,output_final_state,bool(use_qk_l2norm_in_kernel))
    def chunked(self,query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
        if not self.capture:
            query=self.transform(query,use_qk_l2norm_in_kernel); key=self.transform(key,use_qk_l2norm_in_kernel)
            return self.orig_chunk(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,
                                   output_final_state=output_final_state,use_qk_l2norm_in_kernel=False,**kwargs)
        return self._run(query,key,value,g,beta,initial_state,output_final_state,bool(use_qk_l2norm_in_kernel))
    def layer_post(self,layer,out):
        if not self.capture:return
        x=out[0] if isinstance(out,(tuple,list)) else out
        if not torch.is_tensor(x) or x.ndim<3:return
        vals=[]
        for ci in range(1,len(self.conditions)):vals.append((x[ci].float()-x[0].float()).square().sum(-1).sqrt())
        self.post[layer]=torch.stack(vals,0)
    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        self.orig_rec=qmod.torch_recurrent_gated_delta_rule; self.orig_chunk=qmod.torch_chunk_gated_delta_rule
        self.saved=[(qmod,"torch_recurrent_gated_delta_rule",self.orig_rec),(qmod,"torch_chunk_gated_delta_rule",self.orig_chunk)]
        qmod.torch_recurrent_gated_delta_rule=self.recurrent; qmod.torch_chunk_gated_delta_rule=self.chunked
        layers=self.model.model.layers if hasattr(self.model.model,"layers") else self.model.model.language_model.layers
        for i,layer in enumerate(layers):
            mod=getattr(layer,"linear_attn",None)
            if mod is not None:
                self.handles.append(mod.register_forward_pre_hook(lambda _m,_a,i=i:setattr(self,"current_layer",i)))
            if i in GDN_LAYERS:
                self.handles.append(layer.register_forward_hook(lambda _m,_a,o,i=i:self.layer_post(i,o)))
        return self
    def __exit__(self,*_):
        for m,n,o in self.saved:setattr(m,n,o)
        for h in self.handles:h.remove()

def load_bank():
    out={}
    for c in ("C5","C6"):
        p=DEP/c/"frozen_rotation_fp32.pt"
        if file_sha(p)!=EXPECTED[c]:raise RuntimeError(f"{c} hash mismatch")
        z=torch.load(p,map_location="cpu",weights_only=False)
        if tuple(z["layer_ids"])!=GDN_LAYERS:raise RuntimeError(f"{c} layer mapping")
        out[c]=z["matrices"]
    return out
def load_rows():
    rows=[]
    for p in sorted(TEACHER_ROOT.glob("worker*/fp_state.jsonl")):rows+=load_jsonl(p)
    rows.sort(key=lambda r:(int(r["canonical_index"]),int(r["seed"])))
    return rows
def load_model():
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1"); os.environ.setdefault("HF_HUB_OFFLINE","1")
    for p in (str(STATE_QUANT_SRC),str(TRANSFORMERS_SRC)):
        if p not in sys.path:sys.path.insert(0,p)
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
    tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,device_map={"":0},trust_remote_code=True,local_files_only=True)
    model.eval(); return model,tok,DynamicCache

TOKEN_FIELDS=["sample_id","condition","timestep","token_id","scale_median","scale_p95","scale_p99","scale_cvar99","papr_median","papr_p99","qdq_relative_l2","state_error","query_prev_error","readout_error","post_block_error","kl_fp_to_condition","delta_nll","margin_degradation"]
def decision(torch,logits,targets):
    fp=logits[0].float(); fpl=torch.log_softmax(fp,dim=-1); top=torch.topk(fp,2,dim=-1); result=[]
    fp_nll=-fpl.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    for ci in range(1,logits.shape[0]):
        q=logits[ci].float(); ql=torch.log_softmax(q,dim=-1); qp=ql.exp()
        kl=(fpl.exp()*(fpl-ql)).sum(-1)
        qnll=-ql.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
        fpm=top.values[:,0]-top.values[:,1]
        qm=q.gather(-1,top.indices[:,0,None]).squeeze(-1)-q.gather(-1,top.indices[:,1,None]).squeeze(-1)
        result.append({"kl":kl,"delta_nll":qnll-fp_nll,"margin_degradation":fpm-qm})
    return result
def aggregate_chunk(patch,conditions,logits,targets):
    layers=sorted(patch.chunk); T=logits.shape[1]; out=[]
    dec=decision(torch,logits,targets) if conditions[0]=="FP" else None
    for ci,c in enumerate(conditions):
        d={"condition":c}
        for key in ("scale_median","scale_p95","scale_p99","scale_cvar99","papr_median","papr_p99","qdq_rel"):
            a=torch.stack([patch.chunk[l][key][ci] for l in layers],0)
            d[key if key!="qdq_rel" else "qdq_relative_l2"]=a.median(0).values
        if ci==0 or conditions[0]!="FP":
            for k in ("state_error","query_prev_error","readout_error","post_block_error","kl_fp_to_condition","delta_nll","margin_degradation"):
                d[k]=torch.zeros(T,device=logits.device)
        else:
            j=ci-1
            for key in ("state_error","query_prev_error","readout_error"):
                a=torch.stack([patch.chunk[l][key][j] for l in layers],0); d[key]=a.square().sum(0).sqrt()
            a=torch.stack([patch.post[l][j] for l in layers if l in patch.post],0); d["post_block_error"]=a.square().sum(0).sqrt()
            d["kl_fp_to_condition"]=dec[j]["kl"]; d["delta_nll"]=dec[j]["delta_nll"]; d["margin_degradation"]=dec[j]["margin_degradation"]
        out.append(d)
    return out

def capture_one(model,tok,Cache,patch,row,args,token_writer,group_writer):
    device=next(model.parameters()).device; conditions=patch.conditions; B=len(conditions)
    prompt=tok.apply_chat_template(row["raw_messages"],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    enc=tok(prompt,return_tensors="pt",add_special_tokens=False); ids=enc["input_ids"].to(device).repeat(B,1)
    att=enc.get("attention_mask"); att=att.to(device).repeat(B,1) if att is not None else None
    tokens=list(map(int,row["generated_token_ids"])); tokens=tokens[:args.max_tokens or None]
    acc=GroupAccumulator(conditions); patch.capture=False
    cache=Cache(config=model.config,offloading=True,offload_only_non_sliding=True)
    with torch.inference_mode(): pre=model(input_ids=ids,attention_mask=att,past_key_values=cache,use_cache=True,logits_to_keep=1)
    past=pre.past_key_values; del pre,ids,att
    started=time.time(); sample_id=f"{row['problem_id']}_seed{int(row['seed'])}"
    for start in range(0,len(tokens),args.chunk_size):
        chunk=tokens[start:start+args.chunk_size]
        x=torch.tensor(chunk,dtype=torch.long,device=device)[None].repeat(B,1)
        # Each input token predicts the next fixed teacher token. Last target is itself only for bookkeeping and is excluded from causal claims.
        nxt=tokens[start+1:start+len(chunk)+1]
        if len(nxt)<len(chunk):nxt=nxt+[chunk[-1]]
        targets=torch.tensor(nxt,dtype=torch.long,device=device)
        patch.begin(acc)
        with torch.inference_mode(): out=model(input_ids=x,past_key_values=past,use_cache=True,logits_to_keep=0)
        past=out.past_key_values
        missing=sorted(set(GDN_LAYERS)-set(patch.chunk))
        if missing:raise RuntimeError(f"missing layers {missing}")
        tm=aggregate_chunk(patch,conditions,out.logits.float(),targets)
        for ci,d in enumerate(tm):
            for j,tokid in enumerate(chunk):
                rec={"sample_id":sample_id,"condition":d["condition"],"timestep":start+j,"token_id":tokid}
                for k in TOKEN_FIELDS[4:]:rec[k]=float(d[k][j].detach().cpu())
                token_writer.writerow(rec)
        del out,x,targets
        if start//args.chunk_size%64==0:
            print(json.dumps({"event":"progress","mode":args.mode,"set":args.condition_set,"sample":sample_id,"tokens":start+len(chunk),"total":len(tokens),"tokens_per_s":(start+len(chunk))/max(time.time()-started,1e-6),"gpu_mb":torch.cuda.max_memory_allocated()/2**20}),flush=True)
    for rec in acc.rows(sample_id):group_writer.write(json.dumps(rec,sort_keys=True)+"\n")
    del past,acc; torch.cuda.empty_cache()
    return len(tokens)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["controlled","recurrent"],required=True)
    ap.add_argument("--condition-set",choices=["all","baseline","learned"],required=True); ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--shard-count",type=int,default=1)
    ap.add_argument("--chunk-size",type=int,default=8); ap.add_argument("--max-units",type=int,default=0); ap.add_argument("--max-tokens",type=int,default=0); ap.add_argument("--output-dir",required=True); args=ap.parse_args()
    if args.mode=="controlled" and args.condition_set!="all":raise SystemExit("controlled requires all")
    if args.mode=="recurrent" and args.condition_set=="all":raise SystemExit("recurrent must split baseline/learned")
    conditions=("Native","Hadamard","C5","C6") if args.condition_set=="all" else (("FP","Native","Hadamard") if args.condition_set=="baseline" else ("FP","C5","C6"))
    out=Path(args.output_dir)/f"{args.mode}_{args.condition_set}_shard{args.shard_index:02d}"; out.mkdir(parents=True,exist_ok=True)
    status=out/"status.json"; tokenp=out/"token_metrics.csv.gz"; groupp=out/"group_rows.jsonl"
    if any(p.exists() for p in (status,tokenp,groupp)):raise FileExistsError(f"nonempty unit {out}")
    rows=[r for i,r in enumerate(load_rows()) if i%args.shard_count==args.shard_index]
    if args.max_units:rows=rows[:args.max_units]
    dump(status,{"status":"STARTING","mode":args.mode,"condition_set":args.condition_set,"conditions":conditions,"units":len(rows)})
    model,tok,Cache=load_model(); bank=load_bank(); done=0; started=time.time()
    try:
        with gzip.open(tokenp,"wt",newline="",encoding="utf-8") as tf,groupp.open("w",encoding="utf-8") as gf,Patch(model,conditions,args.mode,bank) as patch:
            tw=csv.DictWriter(tf,fieldnames=TOKEN_FIELDS);tw.writeheader()
            for row in rows:
                print(json.dumps({"event":"start","mode":args.mode,"set":args.condition_set,"problem":row["problem_id"],"seed":row["seed"]}),flush=True)
                capture_one(model,tok,Cache,patch,row,args,tw,gf);tf.flush();gf.flush();done+=1
                dump(status,{"status":"RUNNING","completed":done,"total":len(rows),"elapsed":time.time()-started,"last":f"{row['problem_id']}_seed{row['seed']}","layers_seen":sorted(patch.layers_seen)})
        dump(status,{"status":"COMPLETE","completed":done,"total":len(rows),"elapsed":time.time()-started,"peak_gpu_mb":torch.cuda.max_memory_allocated()/2**20,"layers_seen":sorted(patch.layers_seen)})
    except Exception as e:
        dump(status,{"status":"FAILED","completed":done,"total":len(rows),"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc(),"elapsed":time.time()-started});raise
if __name__=="__main__":main()
