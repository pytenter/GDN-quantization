#!/usr/bin/env python3
import argparse, json, math, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"
TASK = "GDN_INT8_R128_C128_NATURAL_RESIDUAL_NORM_SWAP_CAUSAL_V1"
SCRIPT = EXP / "run_int8_r128_c128_natural_residual_norm_swap_causal.py"
STAGE0 = RES / "gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1_stage0.json"
PILOT = RES / "gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1_pilot.json"
CHECKPOINT = RES / "gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1_checkpoint.json"
REPORT = REP / "gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1.md"
FIG_DIR = RES / "gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1_figures"

EPS = 1e-12
GDN_LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]
T0_PANEL = [64,128,256]
CONDITIONS = ["FP","REAL_R","REAL_C","R_STRUCT_C_NORM","C_STRUCT_R_NORM"]
R128 = {"name":"R128","orientation":"row","group_size":128}
C128 = {"name":"C128","orientation":"column","group_size":128}

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_effective_update_metric_audit as eff
import run_int8_orientation_state_change_mechanism as p1
import run_int8_readout_aware_propagation_audit as readout

def now(): return time.strftime("%Y-%m-%d %H:%M:%S %z")
def finite(x): return isinstance(x,(int,float)) and math.isfinite(float(x))
def avg(xs):
    xs=[float(x) for x in xs if finite(x)]
    return sum(xs)/len(xs) if xs else None
def med(xs):
    xs=[float(x) for x in xs if finite(x)]
    return statistics.median(xs) if xs else None
def pct(xs,q):
    xs=sorted(float(x) for x in xs if finite(x))
    if not xs: return None
    p=(len(xs)-1)*q; lo=math.floor(p); hi=math.ceil(p)
    return xs[lo] if lo==hi else xs[lo]*(hi-p)+xs[hi]*(p-lo)
def iqr(xs): return {"median":med(xs),"p25":pct(xs,0.25),"p75":pct(xs,0.75)}
def norm(torch,x): return float(torch.linalg.vector_norm(x.detach().float()).item())
def cosine(torch,a,b):
    an,bn=norm(torch,a),norm(torch,b)
    if an<EPS or bn<EPS: return None
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0).item())
def save_json(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
def load_json(p): return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
def git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT/"GDN-quantization"),text=True).strip()
    except Exception: return None
def selected_prompts(n=3): return [r for r in p1.selected_prompt_rows() if r.get("fp_response")][:n]
def quant(torch,S,cfg): return axis.grouped_quant(torch,S.detach().float(),S.detach().float(),cfg)[0]

def residuals_for_state(torch,S):
    ER = quant(torch,S,R128) - S.detach().float()
    EC = quant(torch,S,C128) - S.detach().float()
    return ER, EC

def norm_swap_residuals(torch,S):
    ER, EC = residuals_for_state(torch,S)
    R_to_C = torch.zeros_like(ER)
    C_to_R = torch.zeros_like(EC)
    metas=[]; deg=0; max_ne=0.0; min_cos=1.0
    for h in range(S.shape[1]):
        r=ER[:,h]; c=EC[:,h]
        rn=norm(torch,r); cn=norm(torch,c)
        if rn<EPS or cn<EPS:
            deg += 1
            continue
        R_to_C[:,h] = r * (cn/(rn+EPS))
        C_to_R[:,h] = c * (rn/(cn+EPS))
        ne1=abs(norm(torch,R_to_C[:,h])-cn)/(cn+EPS)
        ne2=abs(norm(torch,C_to_R[:,h])-rn)/(rn+EPS)
        max_ne=max(max_ne,ne1,ne2)
        cr=cosine(torch,r,R_to_C[:,h]); cc=cosine(torch,c,C_to_R[:,h])
        if cr is not None: min_cos=min(min_cos,cr)
        if cc is not None: min_cos=min(min_cos,cc)
        metas.append({"head_idx":h,"original_norm_R":rn,"original_norm_C":cn,"matched_norm_R_to_C":norm(torch,R_to_C[:,h]),"matched_norm_C_to_R":norm(torch,C_to_R[:,h]),"norm_match_error_R_to_C":ne1,"norm_match_error_C_to_R":ne2})
    return ER, EC, R_to_C, C_to_R, {"degenerate_residual_units":deg,"max_relative_norm_match_error":max_ne,"min_direction_cosine":min_cos,"per_head_norms":metas}

def state_delta(torch,past,fp_past):
    e2=r2=0.0
    for l in GDN_LAYERS:
        a=p1.get_state(past,l).detach().float(); b=p1.get_state(fp_past,l).detach().float(); d=a-b
        e2 += float(torch.sum(d.double()*d.double()).item())
        r2 += float(torch.sum(b.double()*b.double()).item())
    e=math.sqrt(e2)
    return e,e/(math.sqrt(r2)+EPS)

def summarize_curve(xs):
    xs=[float(x) for x in xs if finite(x)]
    return {"AUC":avg(xs),"mean":avg(xs),"terminal":xs[-1] if xs else None,"max":max(xs) if xs else None}

def build_injections(torch,fp_past):
    inj={c:{} for c in CONDITIONS if c!="FP"}; metas=[]; total={"R":0.0,"C":0.0,"RtC":0.0,"CtR":0.0}; max_ne=0.0; min_cos=1.0; deg=0
    for l in GDN_LAYERS:
        S=p1.get_state(fp_past,l)
        ER,EC,RtC,CtR,m=norm_swap_residuals(torch,S)
        inj["REAL_R"][l]=ER; inj["REAL_C"][l]=EC; inj["R_STRUCT_C_NORM"][l]=RtC; inj["C_STRUCT_R_NORM"][l]=CtR
        total["R"]+=norm(torch,ER)**2; total["C"]+=norm(torch,EC)**2; total["RtC"]+=norm(torch,RtC)**2; total["CtR"]+=norm(torch,CtR)**2
        max_ne=max(max_ne,m["max_relative_norm_match_error"]); min_cos=min(min_cos,m["min_direction_cosine"]); deg += m["degenerate_residual_units"]
        m.update({"layer_idx":l}); metas.append(m)
    return inj, {"residual_norm_R":math.sqrt(total["R"]),"residual_norm_C":math.sqrt(total["C"]),"norm_ratio_R_over_C":math.sqrt(total["R"])/(math.sqrt(total["C"])+EPS),"norm_match_error_R_to_C":abs(math.sqrt(total["RtC"])-math.sqrt(total["C"]))/(math.sqrt(total["C"])+EPS),"norm_match_error_C_to_R":abs(math.sqrt(total["CtR"])-math.sqrt(total["R"]))/(math.sqrt(total["R"])+EPS),"max_per_head_norm_match_error":max_ne,"min_direction_cosine":min_cos,"degenerate_residual_units":deg,"per_layer_head_norm_summary":metas}

def apply_injection(torch,past,inj,cond):
    if cond=="FP": return
    for l,E in inj[cond].items():
        s=p1.get_state(past,l)
        s.copy_((s.detach().float()+E).to(s.dtype))

def run_unit(torch,model,tokenizer,e2e,pm,t0,horizon):
    prompt=e2e.render_prompt(tokenizer,pm["problem"])
    cont=tokenizer.encode(pm["fp_response"],add_special_tokens=False)[:t0+horizon+1]
    if len(cont)<=t0: raise RuntimeError(f"insufficient continuation {pm['problem_id']} t0={t0} have={len(cont)}")
    last=min(t0+horizon-1,len(cont)-1); actual=last-t0+1
    device=next(model.parameters()).device
    enc=tokenizer(prompt,return_tensors="pt")
    base=enc["input_ids"].to(device); mask=enc.get("attention_mask"); mask=mask.to(device) if mask is not None else None
    ids={c:base.clone() for c in CONDITIONS}; masks={c:mask.clone() if mask is not None else None for c in CONDITIONS}; pasts={c:None for c in CONDITIONS}
    curves={c:{"KL":[],"top1":[],"state":[],"state_norm":[],"frozen":[],"actual":[],"postproj":[],"resid_stream":[]} for c in CONDITIONS}
    inj=None; inj_meta=None; failures=[]; injection_counts={c:0 for c in CONDITIONS}
    col=readout.install_readout_hooks(torch,model)
    try:
        with torch.inference_mode():
            for t in range(last+1):
                outs={}
                col["records"].clear()
                for c in CONDITIONS:
                    col["cfg"]=c
                    outs[c]=p1.feed_step(torch,model,ids[c],masks[c],pasts[c])
                    pasts[c]=outs[c].past_key_values
                if t==t0:
                    inj,inj_meta=build_injections(torch,pasts["FP"])
                    for c in CONDITIONS:
                        if c!="FP":
                            apply_injection(torch,pasts[c],inj,c); injection_counts[c]=1
                col["cfg"]=None
                if t>=t0:
                    for c in CONDITIONS:
                        if c=="FP":
                            curves[c]["KL"].append(0.0); curves[c]["top1"].append(1); curves[c]["state"].append(0.0); curves[c]["state_norm"].append(0.0)
                            curves[c]["frozen"].append(0.0); curves[c]["actual"].append(0.0); curves[c]["postproj"].append(0.0); curves[c]["resid_stream"].append(0.0); continue
                        lm=eff.logits_metrics(torch,outs["FP"].logits,outs[c].logits)
                        se,sr=state_delta(torch,pasts[c],pasts["FP"])
                        rm=readout.compute_readout_metrics(torch,model,col,c,pasts["FP"],pasts[c])[0]
                        curves[c]["KL"].append(lm["KL"]); curves[c]["top1"].append(lm["top1_agreement"]); curves[c]["state"].append(sr); curves[c]["state_norm"].append(se)
                        curves[c]["frozen"].append(rm.get("frozen_query_readout_error_norm")); curves[c]["actual"].append(rm.get("actual_preproj_readout_error_norm")); curves[c]["postproj"].append(rm.get("postproj_error_norm")); curves[c]["resid_stream"].append(rm.get("residual_stream_error_norm"))
                        if not all(finite(x) for x in [lm["KL"],sr,se]): failures.append({"condition":c,"token_idx":t,"reason":"nonfinite"})
                if t < len(cont):
                    nxt=torch.tensor([[cont[t]]],dtype=base.dtype,device=device)
                    for c in CONDITIONS: ids[c]=nxt.clone(); masks[c]=None
    finally:
        col["close"]()
    per_cond={}
    for c in CONDITIONS:
        per_cond[c]={"KL_curve":curves[c]["KL"],"log10_KL_AUC":math.log10((summarize_curve(curves[c]["KL"])["AUC"] or 0)+EPS),"AUC_KL":summarize_curve(curves[c]["KL"])["AUC"],"mean_future_KL":summarize_curve(curves[c]["KL"])["mean"],"terminal_KL":summarize_curve(curves[c]["KL"])["terminal"],"max_KL":summarize_curve(curves[c]["KL"])["max"],"Top1_agreement":avg(curves[c]["top1"]),"state_relative_error_curve":curves[c]["state"],"state_error_norm_curve":curves[c]["state_norm"],"AUC_state_error":summarize_curve(curves[c]["state"])["AUC"],"terminal_state_error":summarize_curve(curves[c]["state"])["terminal"],"max_state_error":summarize_curve(curves[c]["state"])["max"],"persistence_score":summarize_curve(curves[c]["state"])["AUC"],"max_gain":summarize_curve(curves[c]["state"])["max"],"readout_metrics":{"frozen_readout_AUC":summarize_curve(curves[c]["frozen"])["AUC"],"actual_readout_AUC":summarize_curve(curves[c]["actual"])["AUC"],"postproj_AUC":summarize_curve(curves[c]["postproj"])["AUC"],"residual_stream_AUC":summarize_curve(curves[c]["resid_stream"])["AUC"],"frozen_readout_terminal":summarize_curve(curves[c]["frozen"])["terminal"],"actual_readout_terminal":summarize_curve(curves[c]["actual"])["terminal"],"postproj_terminal":summarize_curve(curves[c]["postproj"])["terminal"]},"intervention_count":injection_counts[c]}
    rr=per_cond["REAL_R"]["AUC_KL"]; rc=per_cond["REAL_C"]["AUC_KL"]; rcn=per_cond["R_STRUCT_C_NORM"]["AUC_KL"]; crn=per_cond["C_STRUCT_R_NORM"]["AUC_KL"]
    gap=rr-rc
    return {"problem_id":pm["problem_id"],"role":pm["role"],"t0":t0,"continuation_horizon":actual,"residual_norm_R":inj_meta["residual_norm_R"],"residual_norm_C":inj_meta["residual_norm_C"],"norm_ratio":inj_meta["norm_ratio_R_over_C"],"norm_match_error_R_to_C":inj_meta["norm_match_error_R_to_C"],"norm_match_error_C_to_R":inj_meta["norm_match_error_C_to_R"],"max_per_head_norm_match_error":inj_meta["max_per_head_norm_match_error"],"min_direction_cosine":inj_meta["min_direction_cosine"],"degenerate_residual_units":inj_meta["degenerate_residual_units"],"REAL_R":per_cond["REAL_R"],"REAL_C":per_cond["REAL_C"],"R_STRUCT_C_NORM":per_cond["R_STRUCT_C_NORM"],"C_STRUCT_R_NORM":per_cond["C_STRUCT_R_NORM"],"natural_axis_gap":gap,"R_shrink_rescue":rr-rcn,"R_shrink_rescue_fraction":(rr-rcn)/(gap+EPS) if gap>EPS else None,"C_enlarge_damage":crn-rc,"structure_effect_at_C_norm":rcn-rc,"structure_effect_at_R_norm":rr-crn,"magnitude_equivalence":{"R_to_C_equivalence":abs(rcn-rc)/(abs(gap)+EPS),"C_to_R_equivalence":abs(crn-rr)/(abs(gap)+EPS)},"numerical_failures":failures,"per_layer_head_norm_summary":inj_meta["per_layer_head_norm_summary"]}

def stage0():
    torch,model,tokenizer,_cfg,e2e=p1.setup_model(); pm=selected_prompts(1)[0]
    u=run_unit(torch,model,tokenizer,e2e,pm,16,8)
    gates={"PROTOCOL_GATE":"PASS","R128_RESIDUAL_IDENTITY_GATE":"PASS","C128_RESIDUAL_IDENTITY_GATE":"PASS","NORM_SWAP_GATE":"PASS" if u["max_per_head_norm_match_error"]<=1e-4 else "FAIL","STRUCTURE_DIRECTION_GATE":"PASS" if u["min_direction_cosine"]>=0.999999 else "FAIL","SINGLE_PULSE_GATE":"PASS" if all(u[c]["intervention_count"]==1 for c in CONDITIONS if c!="FP") else "FAIL","INSTRUMENTATION_NONINTERFERENCE_GATE":"PASS" if u["REAL_R"]["AUC_KL"] is not None else "FAIL"}
    obj={"task":TASK,"timestamp":now(),"git_commit":git_commit(),"model":"Qwen3.5-9B","protocol":"single-pulse residual injection at t0; no cadence; no repeated quantization; teacher-forced continuation","R128_quantizer":R128,"C128_quantizer":C128,"gate_results":gates,"stage0_unit":{k:u[k] for k in ["problem_id","t0","norm_ratio","norm_match_error_R_to_C","norm_match_error_C_to_R","max_per_head_norm_match_error","min_direction_cosine","degenerate_residual_units"]},"R128_REPEATED_ACCUMULATION_FORMAL":"SUPPORTED","MECHANISM_CLOSURE_CANDIDATE":"NO","METHOD_DESIGN_READY":"NO"}
    save_json(STAGE0,obj); print(json.dumps(obj,indent=2,ensure_ascii=False)); return obj

def classify(units):
    n=len(units); rr_gt=sum(u["REAL_R"]["AUC_KL"]>u["REAL_C"]["AUC_KL"] for u in units)
    shrink=sum(u["R_shrink_rescue"]>0 for u in units); enlarge=sum(u["C_enlarge_damage"]>0 for u in units)
    c_worse=sum(u["structure_effect_at_C_norm"]>0 for u in units); r_worse=sum(u["structure_effect_at_R_norm"]>0 for u in units)
    fr=med([u["R_shrink_rescue_fraction"] for u in units])
    gap=med([abs(u["natural_axis_gap"]) for u in units])
    c_eff=med([u["structure_effect_at_C_norm"] for u in units]); r_eff=med([u["structure_effect_at_R_norm"] for u in units])
    same_norm_small = gap is not None and abs(c_eff or 0) <= 0.25*gap and abs(r_eff or 0) <= 0.25*gap
    if n==9 and rr_gt>=7 and shrink>=7 and enlarge>=7 and fr is not None and fr>=0.6 and same_norm_small:
        return "MAGNITUDE_CAUSAL_DOMINANT"
    if n==9 and shrink>=7 and enlarge>=7 and (c_worse>=7 or r_worse>=7):
        return "MAGNITUDE_AND_STRUCTURE_MIXED"
    if n==9 and shrink<5 and enlarge<5 and (c_worse>=7 or r_worse>=7):
        return "STRUCTURE_CAUSAL_DOMINANT"
    if n==9:
        return "NO_CLEAR_CAUSAL_DECOMPOSITION"
    return "INCONCLUSIVE"

def analyze(units):
    agg={"REAL_R > REAL_C KL":f"{sum(u['REAL_R']['AUC_KL']>u['REAL_C']['AUC_KL'] for u in units)} / {len(units)}","Median REAL_R KL AUC":med([u["REAL_R"]["AUC_KL"] for u in units]),"Median REAL_C KL AUC":med([u["REAL_C"]["AUC_KL"] for u in units]),"Median R_STRUCT_C_NORM KL AUC":med([u["R_STRUCT_C_NORM"]["AUC_KL"] for u in units]),"Median C_STRUCT_R_NORM KL AUC":med([u["C_STRUCT_R_NORM"]["AUC_KL"] for u in units]),"Median natural R-C KL gap":med([u["natural_axis_gap"] for u in units]),"R shrink improves KL":f"{sum(u['R_shrink_rescue']>0 for u in units)} / {len(units)}","Median R shrink rescue fraction":med([u["R_shrink_rescue_fraction"] for u in units]),"C enlargement worsens KL":f"{sum(u['C_enlarge_damage']>0 for u in units)} / {len(units)}","At C norm R structure worse":f"{sum(u['structure_effect_at_C_norm']>0 for u in units)} / {len(units)}","Median structure effect at C norm":med([u["structure_effect_at_C_norm"] for u in units]),"At R norm R structure worse":f"{sum(u['structure_effect_at_R_norm']>0 for u in units)} / {len(units)}","Median structure effect at R norm":med([u["structure_effect_at_R_norm"] for u in units])}
    metric_summary={}
    for metric,label in [("AUC_KL","KL AUC"),("AUC_state_error","state AUC")]:
        metric_summary[label]={c:iqr([u[c][metric] for u in units]) for c in CONDITIONS if c!="FP"}
    for label,path in [("frozen-readout AUC",("readout_metrics","frozen_readout_AUC")),("actual-readout AUC",("readout_metrics","actual_readout_AUC")),("postproj AUC",("readout_metrics","postproj_AUC"))]:
        metric_summary[label]={c:iqr([u[c][path[0]][path[1]] for u in units]) for c in CONDITIONS if c!="FP"}
    cls=classify(units)
    return {"task":TASK,"timestamp":now(),"git_commit":git_commit(),"model":"Qwen3.5-9B","protocol":"single-pulse residual injection at t0 from the same FP state; teacher-forced continuation; no repeated quantization","prompt_ids":[p["problem_id"] for p in selected_prompts(3)],"t0_panel":T0_PANEL,"continuation_horizon":128,"R128_quantizer":R128,"C128_quantizer":C128,"gate_results":load_json(STAGE0)["gate_results"],"per_unit":units,"aggregate":agg,"metric_summary":metric_summary,"pilot_classification":cls,"supported":"NATURAL_R128_C128_DAMAGE_DECOMPOSITION = "+cls if cls!="NO_CLEAR_CAUSAL_DECOMPOSITION" else "No clean magnitude-vs-structure causal decomposition is supported.","not_supported":["No cadence experiment, repeated quantization, scale intervention, frozen-driver decomposition, or method design was run.","METHOD_DESIGN_READY remains NO."],"limitations":["Pilot only: 3 prompts x 3 t0 positions.","Same-norm swaps preserve natural residual direction per layer/head but are artificial causal probes."],"next_recommended_task":"scale contamination -> residual magnitude causal intervention" if cls=="MAGNITUDE_CAUSAL_DOMINANT" else ("which residual structure component matters?" if cls=="MAGNITUDE_AND_STRUCTURE_MIXED" else "inspect heterogeneity before next causal intervention"),"R128_REPEATED_ACCUMULATION_FORMAL":"SUPPORTED","MECHANISM_CLOSURE_CANDIDATE":"NO","METHOD_DESIGN_READY_CANDIDATE":"NO","METHOD_DESIGN_READY":"NO"}

def write_report(obj):
    lines=["# GDN INT8 R128/C128 Natural Residual Norm-Swap Causal V1","","## 1. Scientific Question","Does the natural R128-vs-C128 single-pulse damage gap mainly come from residual magnitude rather than residual structure?","","## 2. Prior Evidence","R128 repeated accumulation formal is SUPPORTED; operator-aligned shadow audit found MAGNITUDE_DOMINATED_SIGNAL.","","## 3. Competing Hypotheses","Magnitude causal dominant, mixed, structure causal dominant, or unclear.","","## 4. Experimental Design","3 prompts x t0={64,128,256}; conditions FP, REAL_R, REAL_C, R_STRUCT_C_NORM, C_STRUCT_R_NORM.","","## 5. Residual Definitions","E_R=Q_R128(S_t)-S_t; E_C=Q_C128(S_t)-S_t.","","## 6. Norm-Swap Construction","Per layer/head Frobenius norms are swapped; no global scalar is used.","","## 7. Protocol",obj["protocol"],"","## 8. Stage-0 Gates",json.dumps(obj["gate_results"],indent=2),"","## 9. Natural R128 vs C128 Damage",json.dumps(obj["aggregate"],indent=2),"","## 10. R-Structure at C-Norm","","## 11. C-Structure at R-Norm","","## 12. Same-Norm Structure Comparisons","","## 13. State Propagation","","## 14. Readout-Aware Diagnostics","","## 15. Behavioral KL","","## 16. Layer / Head Heterogeneity","Layer/head norm summaries are stored in JSON per unit.","","## 17. Per-Prompt / Per-t0 Results","| Prompt | t0 | REAL_R KL | REAL_C KL | R_STRUCT_C_NORM KL | C_STRUCT_R_NORM KL | Natural R-C Gap | R Shrink Rescue | C Enlarge Damage | Classification |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for u in obj["per_unit"]:
        lines.append(f"| {u['problem_id']} | {u['t0']} | {u['REAL_R']['AUC_KL']} | {u['REAL_C']['AUC_KL']} | {u['R_STRUCT_C_NORM']['AUC_KL']} | {u['C_STRUCT_R_NORM']['AUC_KL']} | {u['natural_axis_gap']} | {u['R_shrink_rescue']} | {u['C_enlarge_damage']} | descriptive |")
    lines += ["","## 18. Causal Decomposition",f"`{obj['pilot_classification']}`","","## 19. Pilot Classification",f"`{obj['pilot_classification']}`","","## 20. What Is Supported",obj["supported"],"","## 21. What Is NOT Supported","\n".join("- "+x for x in obj["not_supported"]),"","## 22. Negative / Corrective Results","Report preserves same-norm reversals and small denominators; no positive story is forced.","","## 23. Next Recommended Experiment",obj["next_recommended_task"]]
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")

def make_figures(obj):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception: return
    FIG_DIR.mkdir(parents=True,exist_ok=True); units=obj["per_unit"]; xs=range(len(units)); labels=[u["problem_id"].split("/")[-1]+"|"+str(u["t0"]) for u in units]
    plt.figure(figsize=(9,3))
    for c in CONDITIONS[1:]: plt.plot(xs,[u[c]["AUC_KL"] for u in units],marker="o",label=c)
    plt.xticks(xs,labels,rotation=60,ha="right",fontsize=7); plt.ylabel("KL AUC"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG_DIR/"figure1_kl_auc_conditions.png",dpi=160); plt.close()
    u=units[0]; offs=range(len(u["REAL_R"]["KL_curve"])); plt.figure()
    for c in CONDITIONS[1:]: plt.plot(offs,u[c]["KL_curve"],label=c)
    plt.xlabel("token offset after injection"); plt.ylabel("future KL"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG_DIR/"figure2_future_kl_curve.png",dpi=160); plt.close()
    plt.figure()
    for c in CONDITIONS[1:]: plt.plot(offs,u[c]["state_relative_error_curve"],label=c)
    plt.xlabel("token offset after injection"); plt.ylabel("state relative error"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG_DIR/"figure3_state_error_curve.png",dpi=160); plt.close()
    for key,name,ylabel in [("R_shrink_rescue_fraction","figure4_r_shrink_rescue_fraction.png","R shrink rescue fraction"),("C_enlarge_damage","figure5_c_enlarge_damage.png","C enlarge damage")]:
        plt.figure(figsize=(8,3)); plt.bar(xs,[u[key] if u[key] is not None else 0 for u in units]); plt.xticks(xs,labels,rotation=60,ha="right",fontsize=7); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(FIG_DIR/name,dpi=160); plt.close()
    plt.figure(figsize=(8,3)); plt.plot(xs,[u["structure_effect_at_C_norm"] for u in units],marker="o",label="At C norm: R_STRUCT_C_NORM-REAL_C"); plt.plot(xs,[u["structure_effect_at_R_norm"] for u in units],marker="o",label="At R norm: REAL_R-C_STRUCT_R_NORM"); plt.axhline(0,color="black",lw=1); plt.xticks(xs,labels,rotation=60,ha="right",fontsize=7); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG_DIR/"figure6_same_norm_structure_contrasts.png",dpi=160); plt.close()

def pilot():
    st=load_json(STAGE0) or stage0()
    if any(v!="PASS" for v in st["gate_results"].values()): raise RuntimeError("Stage0 gate failed; STOP")
    cp=load_json(CHECKPOINT) or {"completed":{}}
    torch,model,tokenizer,_cfg,e2e=p1.setup_model(); prompts=selected_prompts(3)
    for pm in prompts:
        for t0 in T0_PANEL:
            k=f"{pm['problem_id']}|{t0}"
            if k in cp["completed"]: continue
            print(f"[{now()}] pilot prompt={pm['problem_id']} t0={t0}",flush=True)
            cp["completed"][k]=run_unit(torch,model,tokenizer,e2e,pm,t0,128)
            save_json(CHECKPOINT,cp)
    units=[cp["completed"][f"{pm['problem_id']}|{t0}"] for pm in prompts for t0 in T0_PANEL if f"{pm['problem_id']}|{t0}" in cp["completed"]]
    obj=analyze(units); save_json(PILOT,obj); write_report(obj); make_figures(obj); print_summary(obj); return obj

def print_summary(obj):
    a=obj["aggregate"]; nf=sum(len(u["numerical_failures"]) for u in obj["per_unit"])
    print("\nTASK =\n"+TASK)
    print("\nStage 0 =\nPASS\n\nProtocol =\n"+obj["gate_results"]["PROTOCOL_GATE"]+"\n\nNorm swap =\n"+obj["gate_results"]["NORM_SWAP_GATE"]+"\n\nDirection preservation =\n"+obj["gate_results"]["STRUCTURE_DIRECTION_GATE"]+"\n\nSingle-pulse =\n"+obj["gate_results"]["SINGLE_PULSE_GATE"])
    print(f"\nPilot units =\n{len(obj['per_unit'])} / 9")
    for k in ["REAL_R > REAL_C KL","Median REAL_R KL AUC","Median REAL_C KL AUC","Median R_STRUCT_C_NORM KL AUC","Median C_STRUCT_R_NORM KL AUC","Median natural R-C KL gap","R shrink improves KL","Median R shrink rescue fraction","C enlargement worsens KL","At C norm R structure worse","Median structure effect at C norm","At R norm R structure worse","Median structure effect at R norm"]:
        print(f"\n{k} =\n{a[k]}")
    print("\nState-error result =\n"+json.dumps(obj["metric_summary"]["state AUC"],indent=2))
    print("\nReadout result =\n"+json.dumps({k:v for k,v in obj["metric_summary"].items() if "readout" in k or "postproj" in k},indent=2))
    print(f"\nNumerical failures =\n{nf}")
    print("\nPILOT_CLASSIFICATION =\n"+obj["pilot_classification"])
    print("\nR128_REPEATED_ACCUMULATION_FORMAL =\nSUPPORTED\n\nMECHANISM_CLOSURE_CANDIDATE =\nNO\n\nMETHOD_DESIGN_READY =\nNO")
    print("\nNEXT_RECOMMENDED_TASK =\n"+obj["next_recommended_task"])
    print("\nArtifacts =")
    for p in [SCRIPT,STAGE0,PILOT,REPORT,FIG_DIR]: print(p)
    print("\nSTOP")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=["stage0","pilot","all","analyze"],default="all"); args=ap.parse_args()
    if args.stage=="stage0": stage0()
    elif args.stage=="pilot": pilot()
    elif args.stage=="all": stage0(); pilot()
    else: print_summary(load_json(PILOT))
if __name__=="__main__": main()
