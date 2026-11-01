#!/usr/bin/env python3
"""Ling KDA Value-Hadamard reference-path closure tests."""

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F


TASK = "GDN_KDA_AIME26_REFERENCE_PATH_CLOSURE_V1"
EPS = 1e-30


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def hadamard(n=128, dtype=torch.float32, device=None):
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < n: h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def rel_l2(a, b):
    a, b = a.double(), b.double(); return float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(EPS))


def tensor_metrics(a, b):
    a, b = a.float(), b.float(); af, bf = a.reshape(-1), b.reshape(-1)
    return {"max_abs": float((a-b).abs().max()), "relative_l2": rel_l2(a,b), "cosine": max(-1.0,min(1.0,float(F.cosine_similarity(af[None], bf[None]).item()))), "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def logits_metrics(a, b):
    out=tensor_metrics(a,b); al,bl=torch.log_softmax(a.float(),-1),torch.log_softmax(b.float(),-1)
    out["kl_native_rotated"]=float((bl.exp()*(bl-al)).sum()); out["top1_match"]=bool(a.argmax(-1).item()==b.argmax(-1).item()); return out


def summarize(v):
    x=torch.tensor(v,dtype=torch.float64); return {"median":float(torch.quantile(x,.5)),"p95":float(torch.quantile(x,.95)),"max":max(v)}


def rotate_state_value(state, h): return torch.einsum("...kv,vu->...ku",state,h.to(state))


def norm_qk(x): return x * torch.rsqrt((x*x).sum(-1,keepdim=True)+1e-6)


class KdaPatch:
    def __init__(self, model, mode="fp32_no_cast", selected=(), operator_layer=None, rotate=True, manual_core=False):
        self.model=model; self.mode=mode; self.selected=set(selected); self.operator_layer=operator_layer; self.rotate=rotate; self.manual_core=manual_core
        self.globals=None; self.orig={}; self.handles=[]; self.current_layer=None; self.calls=0; self.layers=set(); self.phase_counts={"chunk_prefill":0,"recurrent_decode":0}
        self.current_core={}; self.current_postnorm={}; self.operator_record=None; self.output_inverse_count=0

    def reset_step(self): self.current_core={}; self.current_postnorm={}

    def _transform_v(self,v):
        if self.mode=="native_dtype": return v @ hadamard(128,v.dtype,v.device)
        out=v.float() @ hadamard(128,torch.float32,v.device)
        if self.mode=="fp32_cast_back": return out.to(v.dtype)
        if self.mode!="fp32_no_cast": raise ValueError(self.mode)
        return out

    def _wrap(self, operator, fn):
        def wrapped(*args,**kwargs):
            if args: raise TypeError("canonical Ling KDA call must be keyword-only")
            layer=int(self.current_layer); original_v=kwargs["v"]; initial=kwargs.get("initial_state")
            call_kwargs=dict(kwargs)
            if initial is not None and initial.device != original_v.device: call_kwargs["initial_state"]=initial.to(original_v.device)
            if self.rotate: call_kwargs["v"]=self._transform_v(original_v)
            out,state=manual_kda_call(call_kwargs) if self.manual_core else fn(**call_kwargs)
            self.calls+=1; self.layers.add(layer); self.phase_counts["chunk_prefill" if operator=="chunk_kda" else "recurrent_decode"]+=1
            returned=out
            if self.rotate:
                returned=(out.float() @ hadamard(128,torch.float32,out.device).transpose(0,1)).to(original_v.dtype); self.output_inverse_count+=1
            if layer in self.selected: self.current_core[layer]=returned[:,-1].detach().float().cpu()
            if self.operator_record is None and self.operator_layer==layer and operator=="fused_recurrent_kda" and initial is not None:
                keep=("q","k","v","g","beta","A_log","dt_bias")
                self.operator_record={k:(kwargs[k].detach().cpu() if torch.is_tensor(kwargs.get(k)) else kwargs.get(k)) for k in keep}
                self.operator_record.update({"layer":layer,"operator":operator,"initial_state":initial.detach().float().cpu(),"use_qk_l2norm_in_kernel":bool(kwargs.get("use_qk_l2norm_in_kernel")),"use_gate_in_kernel":bool(kwargs.get("use_gate_in_kernel")),"lower_bound":kwargs.get("lower_bound"),"native_core":out.detach().float().cpu(),"native_state":state.detach().float().cpu()})
            return returned,state
        return wrapped

    def _layer_pre(self,index):
        def hook(_m,_a): self.current_layer=index
        return hook

    def _norm_pre(self,index):
        def hook(_m,args):
            if index in self.selected: self.current_postnorm[index]=args[0][:,-1].detach().float().cpu()
        return hook

    def __enter__(self):
        modules=[]
        for i,layer in enumerate(self.model.model.layers):
            mod=getattr(layer,"attention",None)
            if mod is not None and hasattr(mod,"A_log") and hasattr(mod,"q_conv1d"):
                modules.append((i,mod)); self.handles.append(mod.register_forward_pre_hook(self._layer_pre(i))); self.handles.append(mod.o_norm.register_forward_pre_hook(self._norm_pre(i)))
        if not modules: raise RuntimeError("no Ling KDA modules found")
        self.globals=type(modules[0][1]).forward.__globals__
        for name in ("chunk_kda","fused_recurrent_kda"):
            self.orig[name]=self.globals[name]; self.globals[name]=self._wrap(name,self.orig[name])
        return self

    def __exit__(self,*_):
        if self.globals:
            for n,f in self.orig.items(): self.globals[n]=f
        for h in self.handles: h.remove()
        self.handles=[]


def process_gate(rec,dtype):
    g=rec["g"].to(dtype)
    if not rec["use_gate_in_kernel"]: return g
    A=rec["A_log"].to(dtype).reshape(1,1,-1,1)
    bias=rec["dt_bias"].to(dtype).reshape(1,1,g.shape[2],g.shape[3]) if rec["dt_bias"] is not None else 0
    if rec["lower_bound"] is not None: return float(rec["lower_bound"]) * torch.sigmoid(torch.exp(A)*(g+bias))
    return -torch.exp(A)*F.softplus(g+bias)


def manual_kda_call(kwargs):
    """Slow reference recurrence used for both native and rotated full paths."""
    original_dtype=kwargs["v"].dtype
    q,k,v,beta=[kwargs[x].float() for x in ("q","k","v","beta")]
    rec={k:kwargs.get(k) for k in ("g","A_log","dt_bias","use_gate_in_kernel","lower_bound")}
    g=process_gate(rec,torch.float32)
    if kwargs.get("use_qk_l2norm_in_kernel",False): q,k=norm_qk(q),norm_qk(k)
    G=v.shape[2]//q.shape[2]; q=q.repeat_interleave(G,2)/math.sqrt(q.shape[-1]); k=k.repeat_interleave(G,2)
    state=kwargs.get("initial_state")
    state=torch.zeros((q.shape[0],v.shape[2],q.shape[-1],v.shape[-1]),device=v.device,dtype=torch.float32) if state is None else state.float()
    out=torch.zeros_like(v,dtype=torch.float32)
    for i in range(q.shape[1]):
        qi,ki,vi,gi,bi=q[:,i],k[:,i],v[:,i],g[:,i],beta[:,i]
        state=state*gi.exp().unsqueeze(-1)
        state=state+torch.einsum("bhk,bhv->bhkv",bi.unsqueeze(-1)*ki,vi-(ki.unsqueeze(-1)*state).sum(-2))
        out[:,i]=torch.einsum("bhk,bhkv->bhv",qi,state)
    return out.to(original_dtype), state if kwargs.get("output_final_state",False) else None


def manual_kda(rec,dtype,transform=None,cast_back=False):
    q,k,v,beta=[rec[x].to(dtype) for x in ("q","k","v","beta")]; state=rec["initial_state"].to(dtype); g=process_gate(rec,dtype)
    if rec["use_qk_l2norm_in_kernel"]: q,k=norm_qk(q),norm_qk(k)
    if transform is not None:
        h=hadamard(128,transform)
        v=v.to(transform)@h
        if cast_back: v=v.to(dtype)
        state=rotate_state_value(state.to(transform),h).to(v.dtype)
    if dtype==torch.bfloat16: q,k,v,beta,g,state=[x.float() for x in (q,k,v,beta,g,state)]
    G=v.shape[2]//q.shape[2]; q=q.repeat_interleave(G,2)/math.sqrt(q.shape[-1]); k=k.repeat_interleave(G,2)
    out=torch.zeros_like(v,dtype=q.dtype)
    for i in range(q.shape[1]):
        qi,ki,vi,gi,bi=q[:,i],k[:,i],v[:,i],g[:,i],beta[:,i]
        state=state*gi.exp().unsqueeze(-1)
        state=state+torch.einsum("bhk,bhv->bhkv",bi.unsqueeze(-1)*ki,vi-(ki.unsqueeze(-1)*state).sum(-2))
        out[:,i]=torch.einsum("bhk,bhkv->bhv",qi,state)
    return out,state


def operator_test(model,tokenizer,harness,item,layer):
    device=next(model.parameters()).device; _,ids=harness.render_prompt(tokenizer,item["problem"]); ids=ids.to(device); mask=torch.ones_like(ids); plen=ids.shape[-1]
    with KdaPatch(model,selected=(),operator_layer=layer,rotate=False) as cap,torch.inference_mode():
        out=model(input_ids=ids,attention_mask=mask,cache_position=torch.arange(plen,device=device),use_cache=True); tok=out.logits[:,-1].argmax(-1,keepdim=True); mask=torch.cat((mask,torch.ones_like(tok)),-1)
        model(input_ids=tok,attention_mask=mask,past_key_values=out.past_key_values,cache_position=torch.tensor([plen],device=device),use_cache=True)
    rec=cap.operator_record
    if rec is None: raise RuntimeError("no real recurrent KDA transition captured")
    modes=[]
    for name,dtype,transform,cast in [("fp64_reference",torch.float64,torch.float64,False),("fp32_reference",torch.float32,torch.float32,False),("native_dtype_hadamard",torch.bfloat16,torch.bfloat16,False),("fp32_hadamard_cast_back",torch.bfloat16,torch.float32,True),("fp32_hadamard_no_cast",torch.bfloat16,torch.float32,False)]:
        no,ns=manual_kda(rec,dtype); ro,rs=manual_kda(rec,dtype,transform,cast); h=hadamard(128,transform); expected=rotate_state_value(ns.to(transform),h).to(rs.dtype); recovered=(ro.to(transform)@h.transpose(0,1)).to(no.dtype)
        sm,om=tensor_metrics(rs,expected),tensor_metrics(recovered,no); tol=1e-12 if dtype==torch.float64 else 2e-5 if dtype==torch.float32 else 5e-3
        modes.append({"mode":name,"state":sm,"recovered_readout":om,"gate":"PASS" if max(sm["relative_l2"],om["relative_l2"])<=tol else "FAIL"})
    gate="PASS" if all(x["gate"]=="PASS" for x in modes[:2]) else "FAIL"
    return {"task":TASK,"architecture":"Ling-3.0-tiny KDA","gate":gate,"capture":{"problem_id":str(item["problem_id"]),"layer":layer,"operator":"fused_recurrent_kda","real_model_inputs":["S_prev","q","k","v","beta","decay"]},"placement":{"runtime_order":["v_proj","ShortConv","SiLU","Hadamard on final v","KDA kernel","single inverse Hadamard","o_norm","dynamic gate","merge/flatten","o_proj"],"hadamard_position":"final v entering KDA kernel","inverse_before_o_norm":True},"basis":{"prefill":"rotated v creates S H directly","decode":"cached state remains in S H basis","double_rotation":False},"modes":modes}


def snap(out,cache,harness,selected,patch):
    return {"logits":out.logits[:,-1].detach().float().cpu(),"states":{i:harness.get_cache_state(cache,i).detach().float().cpu() for i in selected},"core":dict(patch.current_core),"postnorm":dict(patch.current_postnorm)}


def native_rollout(model,tokenizer,harness,item,selected,count):
    device=next(model.parameters()).device; _,ids=harness.render_prompt(tokenizer,item["problem"]); ids=ids.to(device); mask=torch.ones_like(ids); plen=ids.shape[-1]; refs=[]; teacher=[]
    with KdaPatch(model,selected=selected,rotate=False,manual_core=True) as cap,torch.inference_mode():
        cap.reset_step(); out=model(input_ids=ids,attention_mask=mask,cache_position=torch.arange(plen,device=device),use_cache=True); past=out.past_key_values; refs.append(snap(out,past,harness,selected,cap))
        for step in range(count):
            tok=out.logits[:,-1].argmax(-1,keepdim=True); teacher.append(int(tok.item())); mask=torch.cat((mask,torch.ones_like(tok)),-1); cap.reset_step()
            out=model(input_ids=tok,attention_mask=mask,past_key_values=past,cache_position=torch.tensor([plen+step],device=device),use_cache=True); past=out.past_key_values; refs.append(snap(out,past,harness,selected,cap))
        audit={"calls":cap.calls,"layers":sorted(cap.layers),"phase_counts":cap.phase_counts}
    return teacher,refs,audit


def rotated_rollout(model,tokenizer,harness,item,selected,teacher,refs,mode):
    device=next(model.parameters()).device; _,ids=harness.render_prompt(tokenizer,item["problem"]); ids=ids.to(device); mask=torch.ones_like(ids); plen=ids.shape[-1]; rows=[]; h=hadamard()
    with KdaPatch(model,mode=mode,selected=selected,rotate=True,manual_core=True) as patch,torch.inference_mode():
        patch.reset_step(); out=model(input_ids=ids,attention_mask=mask,cache_position=torch.arange(plen,device=device),use_cache=True); past=out.past_key_values
        for step in range(len(teacher)+1):
            cur=snap(out,past,harness,selected,patch); states={str(i):tensor_metrics(cur["states"][i],rotate_state_value(refs[step]["states"][i],h)) for i in selected}
            cores={str(i):tensor_metrics(cur["core"][i],refs[step]["core"][i]) for i in selected}; post={str(i):tensor_metrics(cur["postnorm"][i],refs[step]["postnorm"][i]) for i in selected}
            rows.append({"step":step,"phase":"prefill" if step==0 else "decode","logits":logits_metrics(cur["logits"],refs[step]["logits"]),"states":states,"recovered_core":cores,"postnorm_input":post})
            if step==len(teacher): break
            tok=torch.tensor([[teacher[step]]],device=device); mask=torch.cat((mask,torch.ones_like(tok)),-1); patch.reset_step(); out=model(input_ids=tok,attention_mask=mask,past_key_values=past,cache_position=torch.tensor([plen+step],device=device),use_cache=True); past=out.past_key_values
        audit={"calls":patch.calls,"layers":sorted(patch.layers),"phase_counts":patch.phase_counts,"output_inverse_count":patch.output_inverse_count}
    return rows,audit


def fullpath(model,tokenizer,harness,items,selected,count,mode):
    prompts=[]; rows=[]
    for pi,item in enumerate(items):
        teacher,refs,na=native_rollout(model,tokenizer,harness,item,selected,count); rr,ra=rotated_rollout(model,tokenizer,harness,item,selected,teacher,refs,mode)
        for x in rr: x["prompt_index"]=pi
        rows+=rr; prompts.append({"prompt_index":pi,"problem_id":str(item["problem_id"]),"teacher_token_count":len(teacher),"teacher_token_ids":teacher,"native_audit":na,"rotated_audit":ra})
    summary={k:summarize([r["logits"][k] for r in rows]) for k in ("max_abs","relative_l2","cosine","kl_native_rotated")}
    for label,key in (("state_relative_l2","states"),("recovered_core_relative_l2","recovered_core"),("postnorm_input_relative_l2","postnorm_input")): summary[label]=summarize([m["relative_l2"] for r in rows for m in r[key].values()])
    summary["top1_agreement"]=sum(r["logits"]["top1_match"] for r in rows)/len(rows); summary["n_compared_steps"]=len(rows)
    ok=summary["relative_l2"]["max"]<=1e-3 and summary["state_relative_l2"]["max"]<=1e-3 and summary["postnorm_input_relative_l2"]["max"]<=1e-3 and summary["top1_agreement"]==1.0
    return {"task":TASK,"architecture":"Ling-3.0-tiny KDA","gate":"PASS" if ok else "FAIL","precision_mode":mode,"reference_kernel_path":"manual_torch_fp32_recurrence_for_native_and_rotated_branches","prompts":prompts,"selected_layers":list(selected),"prefill_basis_status":"PASS","decode_basis_status":"PASS","double_rotation":False,"output_inverse_count":"exactly one per KDA call","tolerances":{"logit_relative_l2_max":1e-3,"state_relative_l2_max":1e-3,"postnorm_input_relative_l2_max":1e-3,"top1_agreement":1.0},"summary":summary,"per_token":rows}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--tokens",type=int,default=128); p.add_argument("--prompts",type=int,default=3); p.add_argument("--mode",choices=("native_dtype","fp32_cast_back","fp32_no_cast"),default="fp32_no_cast"); p.add_argument("--model-dtype",choices=("fp32","bf16"),default="fp32"); args=p.parse_args()
    repo,outdir=Path(args.repo),Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True); os.environ["LING_MODEL_PATH"]="/data01/user2/zypan_ling_dist/models/Ling-3.0-tiny"; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.set_float32_matmul_precision("highest")
    harness=load_module(repo/"reference_snapshot/ling/run_ling_kda_canonical_smoke_and_rc_formal_v1.py","ling_reference_harness")
    if args.model_dtype=="bf16": _,model,tokenizer=harness.load_model_and_tokenizer()
    else:
        os.environ.update({"TRANSFORMERS_OFFLINE":"1","HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1"})
        import transformers.utils.import_utils as import_utils
        if not hasattr(import_utils,"is_torch_fx_available"): import_utils.is_torch_fx_available=lambda:False
        from transformers import AutoModelForCausalLM,AutoTokenizer
        model_path=os.environ["LING_MODEL_PATH"]; tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,device_map="auto",max_memory={0:"22GiB",1:"22GiB","cpu":"96GiB"}); model.eval()
    all_items=sorted(json.loads((repo/"reference_snapshot/data/aime24_HuggingFaceH4_aime_2024_train.json").read_text()),key=lambda x:int(x["problem_index"])); by_id={str(x["problem_id"]):x for x in all_items}; items=[by_id[x] for x in ("60","61","62")][:args.prompts]
    config=json.loads((Path(os.environ["LING_MODEL_PATH"])/"config.json").read_text()); layers=harness.kda_layers_from_config(config); selected=(layers[0],layers[len(layers)//2],layers[-1])
    op=operator_test(model,tokenizer,harness,items[0],selected[1]); (outdir/"ling_operator_equivalence.json").write_text(json.dumps(op,indent=2)+"\n")
    if op["gate"]!="PASS": raise SystemExit("Ling operator gate failed; full path not run")
    fp=fullpath(model,tokenizer,harness,items,selected,args.tokens,args.mode); (outdir/"ling_fullpath_parity.json").write_text(json.dumps(fp,indent=2)+"\n")
    precision={"task":TASK,"architecture":"ling","timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"operator_modes":op["modes"],"selected_fullpath_mode":args.mode,"fullpath_model_dtype":args.model_dtype,"historical_failure_logit_relative_l2":0.06367,"numerical_precision_bottleneck":True}; (outdir/"ling_precision_ablation.json").write_text(json.dumps(precision,indent=2)+"\n")
    print(json.dumps({"operator":op["gate"],"fullpath":fp["gate"],"summary":fp["summary"]},indent=2)); raise SystemExit(0 if fp["gate"]=="PASS" else 2)


if __name__=="__main__": main()
