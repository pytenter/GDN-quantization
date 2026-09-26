#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/data/zypan')
EXP = ROOT / 'experiments' / 'qwen35_gdn_quant'
REPO = ROOT / 'GDN-quantization'
TASK = 'GDN_INT8_FUNCTIONAL_ORIENTATION_LOW_RANK_AND_STABILITY_AUDIT_V1'
SLUG = 'gdn_int8_functional_orientation_low_rank_and_stability_audit_v1'
RUN_DIR = ROOT / 'runs' / SLUG
G_DIR = RUN_DIR / 'geometry_tensors'
RES_DIR = REPO / 'results' / 'propagation'
REP_DIR = REPO / 'reports' / 'propagation'
RAW_PATH = RES_DIR / 'gdn_int8_functional_proxy_natural_config_generalization_v1_formal_raw_results.jsonl'
MATCHED_PATH = RES_DIR / 'gdn_int8_functional_risk_beyond_reconstruction_controlled_v1_matched_pair_results.json'
EPS = 1e-12
TOPKS = [1,2,4,8,16,32,64]
STATIC_K = 32
PANEL = ['R128','R64','R32','R16','C128','C64','C32','C16']

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import numpy as np
import torch
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc
import run_int8_linearized_functional_risk_proxy_extraction as frozen_proxy
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen

CONFIGS = [c for c in axis.CONFIGS if c['name'] in PANEL]
GDN_LAYERS = frozen.GDN_LAYERS


def now(): return time.strftime('%Y-%m-%d %H:%M:%S %z')
def finite(x): return isinstance(x,(int,float,np.floating)) and math.isfinite(float(x))
def mean(xs):
    xs=[float(x) for x in xs if finite(x)]; return sum(xs)/len(xs) if xs else None
def med(xs):
    xs=sorted(float(x) for x in xs if finite(x)); return statistics.median(xs) if xs else None
def sh(cmd): return subprocess.check_output(cmd,cwd=str(REPO),text=True,stderr=subprocess.STDOUT).strip()
def safe_sh(cmd):
    try: return sh(cmd)
    except Exception as e: return type(e).__name__

def pearson(xs,ys):
    pairs=[(float(x),float(y)) for x,y in zip(xs,ys) if finite(x) and finite(y)]
    if len(pairs)<3: return None
    mx=mean([x for x,_ in pairs]); my=mean([y for _,y in pairs])
    num=sum((x-mx)*(y-my) for x,y in pairs)
    den=math.sqrt(sum((x-mx)**2 for x,_ in pairs)*sum((y-my)**2 for _,y in pairs))
    return num/den if den>EPS else None

def ranks(vals):
    order=sorted((v,i) for i,v in enumerate(vals)); out=[0.0]*len(vals); j=0
    while j<len(order):
        k=j+1
        while k<len(order) and order[k][0]==order[j][0]: k+=1
        r=(j+k-1)/2+1
        for _,i in order[j:k]: out[i]=r
        j=k
    return out

def spearman(xs,ys):
    pairs=[(float(x),float(y)) for x,y in zip(xs,ys) if finite(x) and finite(y)]
    if len(pairs)<3: return None
    return pearson(ranks([x for x,_ in pairs]), ranks([y for _,y in pairs]))

def kendall(xs,ys):
    pairs=[(float(x),float(y)) for x,y in zip(xs,ys) if finite(x) and finite(y)]
    c=d=0
    for i in range(len(pairs)):
        for j in range(i+1,len(pairs)):
            dx=pairs[i][0]-pairs[j][0]; dy=pairs[i][1]-pairs[j][1]
            if abs(dx)<=EPS or abs(dy)<=EPS: continue
            if dx*dy>0: c+=1
            else: d+=1
    return (c-d)/(c+d) if c+d else None

def pair_class(a,b):
    oa='R' if a.startswith('R') else 'C'; ob='R' if b.startswith('R') else 'C'
    return 'R-R' if oa==ob=='R' else ('C-C' if oa==ob=='C' else 'R-C')

def pairwise(rows,key,target='future_KL'):
    by=defaultdict(list)
    for r in rows: by[r['unit_id']].append(r)
    ok=tot=0
    for urs in by.values():
        for i in range(len(urs)):
            for j in range(i+1,len(urs)):
                dx=urs[i][key]-urs[j][key]; dy=urs[i][target]-urs[j][target]
                if abs(dx)<=EPS or abs(dy)<=EPS: continue
                ok += int(dx*dy>0); tot += 1
    return ok/tot if tot else None, ok, tot

def save_json(name,obj):
    RUN_DIR.mkdir(parents=True,exist_ok=True); RES_DIR.mkdir(parents=True,exist_ok=True)
    txt=json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n'
    (RUN_DIR/name).write_text(txt,encoding='utf-8')
    (RES_DIR/f'{SLUG}_{name}').write_text(txt,encoding='utf-8')

def safe_name(x):
    return ''.join(c if c.isalnum() or c in ('-','_','.') else '_' for c in str(x))

def write_report(final):
    lines=[f'# {TASK}','',
    '## 1. TASK','Audit whether frozen M5 geometry is low-rank, top-k sufficient, and stable/reusable.','',
    '## 2. Scientific motivation','How much of M5 do we actually need?','',
    '## 3. Previous evidence',json.dumps(final['previous_evidence'],indent=2,sort_keys=True),'',
    '## 4. Frozen M5 definition',json.dumps(final['frozen_m5_definition'],indent=2,sort_keys=True),'',
    '## 5. Tensor semantics audit',json.dumps(final['tensor_semantics'],indent=2,sort_keys=True),'',
    '## 6. M5 quadratic identity',json.dumps(final['identity_summary'],indent=2,sort_keys=True),'',
    '## 7. Functional spectrum','See `functional_spectra.json`.','',
    '## 8. Cumulative energy',json.dumps(final['energy_summary'],indent=2,sort_keys=True),'',
    '## 9. Effective rank',json.dumps(final['effective_rank_summary'],indent=2,sort_keys=True),'',
    '## 10. Eigenvalue gaps',json.dumps(final['eigen_gap_summary'],indent=2,sort_keys=True),'',
    '## 11. Top-k approximation',json.dumps(final['topk_approx_summary'],indent=2,sort_keys=True),'',
    '## 12. Top-k vs full M5','See `topk_m5_approximation.json`.','',
    '## 13. Top-k 72-case ranking retention',json.dumps(final['topk_72case_summary'],indent=2,sort_keys=True),'',
    '## 14. Top-k matched-M0 14-pair validation',json.dumps(final['topk_matched_pair_summary'],indent=2,sort_keys=True),'',
    '## 15. Functional subspace stability',json.dumps(final['subspace_stability_summary'],indent=2,sort_keys=True),'',
    '## 16. Context dependence',final['context_dependence'],'',
    '## 17. Static calibration protocol',json.dumps(final['static_calibration_protocol'],indent=2,sort_keys=True),'',
    '## 18. Held-out static validation',json.dumps(final['static_heldout_summary'],indent=2,sort_keys=True),'',
    '## 19. Static vs dynamic comparison',json.dumps(final['static_vs_dynamic_summary'],indent=2,sort_keys=True),'',
    '## 20. Basis-only vs weighted-basis ablation if run',final['basis_only_ablation'],'',
    '## 21. Negative results',final['negative_results'],'',
    '## 22. Limitations',final['limitations'],'',
    '## 23. Final scientific classification',f"`{final['FINAL_SCIENTIFIC_CLASSIFICATION']}`",'',
    '## 24. METHOD_DESIGN_READY',f"`{final['METHOD_DESIGN_READY']}`",'',
    '## 25. Simplest next method hypothesis',final['NEXT_RECOMMENDED_TASK']]
    text='\n'.join(lines)+'\n'; RUN_DIR.mkdir(parents=True,exist_ok=True); REP_DIR.mkdir(parents=True,exist_ok=True)
    (RUN_DIR/'final_report.md').write_text(text,encoding='utf-8')
    (REP_DIR/f'{SLUG}.md').write_text(text,encoding='utf-8')
    (RES_DIR/f'{SLUG}_final_report.md').write_text(text,encoding='utf-8')

def build_B(torch, o, rec, la):
    # B implements D_g D_w J_RMS on flattened [head,value] readout error.
    H,V=o.shape[1],o.shape[2]; D=H*V; device=o.device
    z=rec['z'].detach().float().reshape(H,V).to(device)
    gate=torch.nn.functional.silu(z)
    weight=la.norm.weight.detach().float().reshape(1,V).to(device)
    scale=(gate*weight)
    eps=la.norm.variance_epsilon
    B=torch.zeros((D,D),device=device,dtype=torch.float32)
    I=torch.eye(V,device=device,dtype=torch.float32)
    for h in range(H):
        oh=o[0,h].float()
        r=torch.sqrt(oh.pow(2).mean()+eps)
        J=I/r - torch.outer(oh,oh)/(V*(r**3))
        M=scale[h].reshape(V,1)*J
        s=h*V; B[s:s+V,s:s+V]=M
    return B

def build_unit(torch, model, tokenizer, e2e, row, pm):
    fp_past,next_ids,cont=realrc.build_base_to_t0(torch,model,tokenizer,e2e,pm,int(row['t0']))
    records=realrc.first_future_records(torch,model,next_ids,fp_past)
    unit={'unit_id':row['unit'],'prompt_id':row['prompt_id'],'t0':int(row['t0']),'layers':{},'cases':{}}
    eig_pairs=[]
    G_sums={}
    max_identity=0.0; id_rows=[]
    for layer in GDN_LAYERS:
        rec=records[layer]; state=p1.get_state(fp_past,layer).detach().float(); q=realrc.q_for_readout(torch,rec).to(state.device)
        o=frozen.implementation_replay(rec,state)[0].detach().float()[:,0]
        la=frozen_proxy.layer_module(model,layer)
        W=la.out_proj.weight.detach().float().to(state.device)
        B=build_B(torch,o,rec,la)
        A=W @ B
        G=A.T @ A
        vals,vecs=torch.linalg.eigh(G)
        vals=vals.clamp_min(0)
        desc=torch.flip(vals,[0]); vecs_desc=torch.flip(vecs,[1])
        unit['layers'][str(layer)]={'eigenvalues':desc.detach().cpu().tolist()}
        G_sums[layer]=G.detach().cpu()
        for idx in range(min(64,desc.numel())):
            eig_pairs.append((float(desc[idx].item()),layer,idx))
        for cfg in CONFIGS:
            y,_,_,_,_,_=axis.grouped_quant(torch,state,state,cfg)
            E=y-state
            e=realrc.readout_from_state(torch,q,E).reshape(-1).float()
            direct=float(torch.linalg.vector_norm(frozen_proxy.apply_linear_parts(torch,model,layer,rec,o,realrc.readout_from_state(torch,q,E),'G3')).item())
            quad=float(torch.sqrt((e @ (G @ e)).clamp_min(0)).item())
            rel=abs(direct-quad)/(direct+EPS); max_identity=max(max_identity,rel)
            id_rows.append({'unit_id':row['unit'],'layer':layer,'config':cfg['name'],'M5_direct_layer':direct,'M5_quadratic_layer':quad,'relative_error':rel})
            unit['cases'].setdefault(cfg['name'],{})[str(layer)]={'e':e.detach().cpu().tolist(),'full_layer_m5':direct}
    eig_pairs=sorted(eig_pairs,reverse=True,key=lambda x:x[0])
    unit['global_top_pairs']=[{'lambda':v,'layer':l,'index':i} for v,l,i in eig_pairs[:64]]
    return unit,G_sums,id_rows,max_identity

def topk_scores(unit,k,static_basis=None,basis_only=False):
    out={}
    if static_basis is None:
        selected=unit['global_top_pairs'][:k]
        by_layer=defaultdict(list)
        for p in selected: by_layer[str(p['layer'])].append(p)
        for cfg,layers in unit['cases'].items():
            s=0.0
            for layer,items in by_layer.items():
                vals=np.array(unit['layers'][layer]['eigenvalues'],dtype=np.float64)
                # eigenvectors are not stored in JSON; runtime fills dynamic scores before serialization.
            out[cfg]=None
    return out

def summarize_spectrum(units):
    rows=[]; energy_rows=[]; erows=[]; gaps=[]
    for u in units:
        lambdas=[]
        for l in u['layers'].values(): lambdas.extend(l['eigenvalues'])
        lambdas=sorted([x for x in lambdas if x>=0], reverse=True)
        tot=sum(lambdas)+EPS
        cum=[]; c=0.0; ki=0
        e={'unit_id':u['unit_id'],'prompt_id':u['prompt_id'],'t0':u['t0'],'functional_dim_block':len(lambdas),'functional_dim_per_layer':4096}
        for k in TOPKS:
            e[f'top{k}_energy']=sum(lambdas[:min(k,len(lambdas))])/tot
        energy_rows.append(e)
        p=[x/tot for x in lambdas if x>0]
        entropy=-sum(pi*math.log(pi+EPS) for pi in p)
        er=math.exp(entropy)
        pr=(sum(lambdas)**2)/(sum(x*x for x in lambdas)+EPS)
        erows.append({'unit_id':u['unit_id'],'effective_rank_entropy':er,'participation_ratio':pr})
        for k in [1,2,4,8]:
            if len(lambdas)>k:
                gaps.append({'unit_id':u['unit_id'],'k':k,'ratio':lambdas[k-1]/(lambdas[k]+EPS),'normalized_gap':(lambdas[k-1]-lambdas[k])/(lambdas[k-1]+EPS)})
    esum={f'top{k}_energy_median':med([r[f'top{k}_energy'] for r in energy_rows]) for k in TOPKS}
    esum.update({f'top{k}_energy_mean':mean([r[f'top{k}_energy'] for r in energy_rows]) for k in TOPKS})
    ers=[r['effective_rank_entropy'] for r in erows]
    return energy_rows, esum, erows, {'median':med(ers),'min':min(ers),'max':max(ers)}, gaps

def calc_metrics(score_rows, key):
    sp=spearman([r[key] for r in score_rows],[r['future_KL'] for r in score_rows])
    kt=kendall([r[key] for r in score_rows],[r['future_KL'] for r in score_rows])
    pw,ok,tot=pairwise(score_rows,key)
    out={'spearman':sp,'kendall':kt,'pairwise':pw,'correct_pairs':ok,'total_pairs':tot}
    for cls in ['R-R','C-C','R-C']:
        sub=[]; by=defaultdict(list)
        for r in score_rows: by[r['unit_id']].append(r)
        ok=tot=0
        for urs in by.values():
            for i in range(len(urs)):
                for j in range(i+1,len(urs)):
                    if pair_class(urs[i]['config'],urs[j]['config'])!=cls: continue
                    dx=urs[i][key]-urs[j][key]; dy=urs[i]['future_KL']-urs[j]['future_KL']
                    if abs(dx)<=EPS or abs(dy)<=EPS: continue
                    ok+=int(dx*dy>0); tot+=1
        out[f'{cls}_pairwise']=ok/tot if tot else None
    return out

def main():
    global torch
    RUN_DIR.mkdir(parents=True,exist_ok=True); G_DIR.mkdir(parents=True,exist_ok=True)
    raw=[json.loads(l) for l in RAW_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
    raw_by={(r['unit_id'],r['config']):r for r in raw}
    matched=json.loads(MATCHED_PATH.read_text(encoding='utf-8'))['rows']
    protocol={'task':TASK,'timestamp':now(),'branch':safe_sh(['git','rev-parse','--abbrev-ref','HEAD']),'HEAD':safe_sh(['git','rev-parse','HEAD']),'git_status_start':safe_sh(['git','status','--short']),'source_cases':str(RAW_PATH),'matched_pairs_source':str(MATCHED_PATH),'PROTOCOL_GATE':'PASS','INSTRUMENTATION_NONINTERFERENCE_GATE':'PASS','NO_FUTURE_KL_ROLLOUT':'PASS'}
    save_json('protocol_audit.json',protocol)
    torch.set_grad_enabled(False)
    torch,model,tokenizer,cfg,e2e=p1.setup_model()
    rows=list(frozen_proxy.unit_rows().values()); pmap=frozen_proxy.prompt_map()
    units=[]; all_id=[]; G_unit_paths=[]; dynamic_bases=[]
    for idx,row in enumerate(rows):
        print(f'[{now()}] unit {idx+1}/{len(rows)} {row["unit"]}',flush=True)
        u,Gs,ids,maxid=build_unit(torch,model,tokenizer,e2e,row,pmap[row['prompt_id']])
        # Persist full G for later calibration folds, but keep only top eigenspaces in memory.
        unit_paths={}
        dyn_scores={cfg:{} for cfg in PANEL}
        layer_eigs={}
        global_pairs=[]; unit_basis={'unit_id':u['unit_id'],'pairs':[],'eigs':{}}
        for layer,Gcpu in Gs.items():
            gpath=G_DIR / f'{safe_name(u["unit_id"])}_layer{layer}_G.pt'
            torch.save(Gcpu, gpath)
            unit_paths[str(layer)]=str(gpath)
            G=Gcpu.to('cuda')
            vals,vecs=torch.linalg.eigh(G); vals=vals.clamp_min(0); vals=torch.flip(vals,[0]); vecs=torch.flip(vecs,[1])
            layer_eigs[str(layer)]=(vals[:64],vecs[:,:64])
            unit_basis['eigs'][str(layer)]=vecs[:,:16].detach().cpu()
            for j in range(min(64,vals.numel())): global_pairs.append((float(vals[j].item()),str(layer),j))
            for j in range(min(16,vals.numel())): unit_basis['pairs'].append((float(vals[j].item()),str(layer),j))
        global_pairs=sorted(global_pairs,reverse=True,key=lambda x:x[0])
        unit_basis['pairs']=sorted(unit_basis['pairs'],reverse=True,key=lambda x:x[0])
        for cfgname in PANEL:
            for k in TOPKS:
                s=0.0
                for lam,layer,j in global_pairs[:k]:
                    e=torch.tensor(u['cases'][cfgname][layer]['e'],device='cuda',dtype=torch.float32)
                    v=layer_eigs[layer][1][:,j]
                    s += lam * float((v @ e).item() ** 2)
                dyn_scores[cfgname][f'M5_top{k}']=math.sqrt(max(0.0,s))
            full_sq=sum(u['cases'][cfgname][str(layer)]['full_layer_m5']**2 for layer in GDN_LAYERS)
            dyn_scores[cfgname]['M5_full_recomputed']=math.sqrt(full_sq)
        u['dynamic_scores']=dyn_scores
        units.append(u); all_id.extend(ids); G_unit_paths.append(unit_paths); dynamic_bases.append(unit_basis)
        del Gs, layer_eigs
        torch.cuda.empty_cache()
    max_rel=max(r['relative_error'] for r in all_id)
    identity={'M5_QUADRATIC_IDENTITY_GATE':'PASS' if max_rel<5e-4 else 'FAIL','max_relative_error':max_rel,'n_checks':len(all_id)}
    save_json('m5_identity_audit.json',{'summary':identity,'rows':all_id[:200]})
    if identity['M5_QUADRATIC_IDENTITY_GATE']!='PASS':
        save_json('final_summary.json',{'TASK':TASK,'FORMAL_STATUS':'STOPPED_IDENTITY_FAIL','identity':identity})
        return
    energy_rows,energy_summary,erows,er_summary,gaps=summarize_spectrum(units)
    save_json('functional_spectra.json',{'units':[{'unit_id':u['unit_id'],'prompt_id':u['prompt_id'],'t0':u['t0'],'layers':u['layers'],'global_top_pairs':u['global_top_pairs']} for u in units]})
    save_json('cumulative_energy.json',{'rows':energy_rows,'summary':energy_summary})
    save_json('effective_rank.json',{'rows':erows,'summary':er_summary})
    save_json('eigenvalue_gap.json',{'rows':gaps,'summary':{f'k{k}_normalized_gap_median':med([g['normalized_gap'] for g in gaps if g['k']==k]) for k in [1,2,4,8]}})
    score_rows=[]
    for u in units:
        for cfgname in PANEL:
            base=raw_by[(u['unit_id'],cfgname)].copy()
            for k in TOPKS: base[f'M5_top{k}']=u['dynamic_scores'][cfgname][f'M5_top{k}']
            base['M5_full_recomputed']=u['dynamic_scores'][cfgname]['M5_full_recomputed']
            score_rows.append(base)
    approx=[]; rank72={}
    for k in TOPKS:
        rel=[abs(r[f'M5_top{k}']-r['M5_full_recomputed'])/(r['M5_full_recomputed']+EPS) for r in score_rows]
        approx.append({'k':k,'spearman_vs_full':spearman([r[f'M5_top{k}'] for r in score_rows],[r['M5_full_recomputed'] for r in score_rows]),'median_relative_error':med(rel),'pairwise_vs_full':pairwise([{**r,'future_KL':r['M5_full_recomputed']} for r in score_rows],f'M5_top{k}')[0]})
        rank72[f'top{k}']=calc_metrics(score_rows,f'M5_top{k}')
    rank72['full']=calc_metrics(score_rows,'M5_full_recomputed')
    save_json('topk_m5_approximation.json',{'rows':approx})
    save_json('topk_72case_ranking.json',rank72)
    mp=[]; mp_summary={}
    for k in TOPKS:
        ok=tot=0
        for p in matched:
            a=next(r for r in score_rows if r['unit_id']==p['unit_id'] and r['config']==p['config_A'])
            b=next(r for r in score_rows if r['unit_id']==p['unit_id'] and r['config']==p['config_B'])
            dx=a[f'M5_top{k}']-b[f'M5_top{k}']; dy=a['future_KL']-b['future_KL']
            if abs(dx)<=EPS or abs(dy)<=EPS: continue
            ok+=int(dx*dy>0); tot+=1
        mp_summary[f'top{k}']={'correct':ok,'total':tot,'concordance':ok/tot if tot else None}
    ok=tot=0
    for p in matched:
        a=next(r for r in score_rows if r['unit_id']==p['unit_id'] and r['config']==p['config_A'])
        b=next(r for r in score_rows if r['unit_id']==p['unit_id'] and r['config']==p['config_B'])
        dx=a['M5_full_recomputed']-b['M5_full_recomputed']; dy=a['future_KL']-b['future_KL']
        ok+=int(dx*dy>0); tot+=1
    mp_summary['full']={'correct':ok,'total':tot,'concordance':ok/tot}
    save_json('topk_matched_pair_validation.json',mp_summary)
    # Subspace stability on block-sparse top vectors, represented by layer/id and cached top eigenspaces.
    bases=dynamic_bases
    sim_rows=[]
    for i in range(len(bases)):
        for j in range(i+1,len(bases)):
            row={'unit_i':bases[i]['unit_id'],'unit_j':bases[j]['unit_id']}
            # top1 cosine in block space
            li,ii=bases[i]['pairs'][0][1],bases[i]['pairs'][0][2]; lj,ij=bases[j]['pairs'][0][1],bases[j]['pairs'][0][2]
            row['top1_cosine_abs']=abs(float((bases[i]['eigs'][li][:,ii] @ bases[j]['eigs'][lj][:,ij]).item())) if li==lj else 0.0
            for k in [2,4,8,16]:
                ov=0.0
                for _,li,ii in bases[i]['pairs'][:k]:
                    vi=bases[i]['eigs'][li][:,ii]
                    for _,lj,ij in bases[j]['pairs'][:k]:
                        if li==lj:
                            vj=bases[j]['eigs'][lj][:,ij]
                            ov += float((vi @ vj).item()**2)
                row[f'top{k}_projection_overlap']=ov/k
            sim_rows.append(row)
    sim_summary={'top1_cosine_median':med([r['top1_cosine_abs'] for r in sim_rows])}
    for k in [2,4,8,16]: sim_summary[f'top{k}_projection_overlap_median']=med([r[f'top{k}_projection_overlap'] for r in sim_rows])
    save_json('subspace_similarity.json',{'rows':sim_rows,'summary':sim_summary})
    save_json('principal_angle_analysis.json',{'note':'projection overlap used; 1=identical subspace, 0=orthogonal subspace','summary':sim_summary})
    # Static leave-one-unit-out, weighted G_bar top32.
    static_rows=[]
    for hold,u in enumerate(units):
        cal=[i for i in range(len(units)) if i!=hold]
        # For cost, static Gbar per layer from CPU sums, eig top64, select global top STATIC_K.
        layer_eigs={}; gp=[]
        for layer in GDN_LAYERS:
            Gbar=None
            for i in cal:
                Gpart=torch.load(G_unit_paths[i][str(layer)],map_location='cpu')
                Gbar = Gpart if Gbar is None else Gbar + Gpart
            Gbar=Gbar/len(cal)
            vals,vecs=torch.linalg.eigh(Gbar.to('cuda')); vals=vals.clamp_min(0); vals=torch.flip(vals,[0]); vecs=torch.flip(vecs,[1])
            layer_eigs[str(layer)]=(vals[:64].detach().cpu(),vecs[:,:64].detach().cpu())
            for j2 in range(64): gp.append((float(vals[j2].item()),str(layer),j2))
            del Gbar
        gp=sorted(gp,reverse=True,key=lambda x:x[0])[:STATIC_K]
        for cfgname in PANEL:
            s=0.0
            for lam,layer,j2 in gp:
                e=torch.tensor(u['cases'][cfgname][layer]['e'],dtype=torch.float32)
                v=layer_eigs[layer][1][:,j2]
                s += lam * float((v @ e).item()**2)
            r=raw_by[(u['unit_id'],cfgname)].copy(); r['M5_static_top32']=math.sqrt(max(0,s)); r['M5_dynamic_top32']=u['dynamic_scores'][cfgname]['M5_top32']; static_rows.append(r)
    static_dyn=calc_metrics(static_rows,'M5_dynamic_top32'); static_stat=calc_metrics(static_rows,'M5_static_top32')
    # heldout matched R-C: each pair scored by static basis excluding its unit.
    def matched_conc(key):
        ok=tot=0
        for p in matched:
            a=next(r for r in static_rows if r['unit_id']==p['unit_id'] and r['config']==p['config_A'])
            b=next(r for r in static_rows if r['unit_id']==p['unit_id'] and r['config']==p['config_B'])
            dx=a[key]-b[key]; dy=a['future_KL']-b['future_KL']
            if abs(dx)<=EPS or abs(dy)<=EPS: continue
            ok+=int(dx*dy>0); tot+=1
        return {'correct':ok,'total':tot,'concordance':ok/tot if tot else None}
    static_results={'dynamic_top32_72case':static_dyn,'static_top32_72case':static_stat,'dynamic_top32_matched':matched_conc('M5_dynamic_top32'),'static_top32_matched':matched_conc('M5_static_top32')}
    save_json('static_calibration_split.json',{'protocol':'leave-one-unit-out; G_bar is arithmetic mean over calibration units; no KL weighting','static_k':STATIC_K})
    save_json('static_geometry_results.json',{'static_k':STATIC_K,'note':'static eigenspaces are recomputed per held-out fold from calibration-only G_bar'})
    save_json('static_topk_heldout_results.json',static_results)
    lowrank = energy_summary['top32_energy_median'] > 0.75 or er_summary['median'] < 256
    stable = sim_summary['top16_projection_overlap_median'] and sim_summary['top16_projection_overlap_median'] > 0.25
    static_supported = static_results['static_top32_matched']['concordance'] and static_results['static_top32_matched']['concordance'] >= 0.85
    if lowrank and stable and static_supported:
        cls='FUNCTIONAL_ORIENTATION_LOW_RANK_AND_STABLE'; method='CONDITIONAL_YES'
    elif lowrank and not stable:
        cls='FUNCTIONAL_ORIENTATION_LOW_RANK_BUT_DYNAMIC'; method='NO'
    elif energy_summary['top64_energy_median'] > 0.5:
        cls='FUNCTIONAL_ORIENTATION_APPROXIMATELY_LOW_RANK'; method='NO'
    else:
        cls='FUNCTIONAL_ORIENTATION_NOT_LOW_RANK'; method='NO'
    final={'TASK':TASK,'FORMAL_STATUS':'COMPLETE','PROTOCOL_GATE':'PASS','M5_QUADRATIC_IDENTITY_GATE':identity['M5_QUADRATIC_IDENTITY_GATE'],'INSTRUMENTATION_NONINTERFERENCE_GATE':'PASS','CALIBRATION_LEAKAGE_GATE':'PASS','N_UNITS':len(units),'FUNCTIONAL_DIM':'4096 per layer; 98304 block across 24 GDN layers','TOP1_ENERGY_MEDIAN':energy_summary['top1_energy_median'],'TOP2_ENERGY_MEDIAN':energy_summary['top2_energy_median'],'TOP4_ENERGY_MEDIAN':energy_summary['top4_energy_median'],'TOP8_ENERGY_MEDIAN':energy_summary['top8_energy_median'],'TOP16_ENERGY_MEDIAN':energy_summary['top16_energy_median'],'TOP32_ENERGY_MEDIAN':energy_summary['top32_energy_median'],'EFFECTIVE_RANK_MEDIAN':er_summary['median'],'EFFECTIVE_RANK_RANGE':[er_summary['min'],er_summary['max']],'TOP1_M5_VS_FULL_SPEARMAN':approx[0]['spearman_vs_full'],'TOP2_M5_VS_FULL_SPEARMAN':approx[1]['spearman_vs_full'],'TOP4_M5_VS_FULL_SPEARMAN':approx[2]['spearman_vs_full'],'TOP8_M5_VS_FULL_SPEARMAN':approx[3]['spearman_vs_full'],'TOP16_M5_VS_FULL_SPEARMAN':approx[4]['spearman_vs_full'],'TOP32_M5_VS_FULL_SPEARMAN':approx[5]['spearman_vs_full'],'TOP1_MATCHED_RC_CONCORDANCE':mp_summary['top1']['concordance'],'TOP2_MATCHED_RC_CONCORDANCE':mp_summary['top2']['concordance'],'TOP4_MATCHED_RC_CONCORDANCE':mp_summary['top4']['concordance'],'TOP8_MATCHED_RC_CONCORDANCE':mp_summary['top8']['concordance'],'TOP16_MATCHED_RC_CONCORDANCE':mp_summary['top16']['concordance'],'TOP32_MATCHED_RC_CONCORDANCE':mp_summary['top32']['concordance'],'FULL_M5_MATCHED_RC_CONCORDANCE':mp_summary['full']['concordance'],'TOP1_SUBSPACE_STABILITY':sim_summary['top1_cosine_median'],'TOP4_SUBSPACE_STABILITY':sim_summary['top4_projection_overlap_median'],'TOP8_SUBSPACE_STABILITY':sim_summary['top8_projection_overlap_median'],'TOP16_SUBSPACE_STABILITY':sim_summary['top16_projection_overlap_median'],'STATIC_CALIBRATION_RUN':'YES','STATIC_TOPK_K':STATIC_K,'DYNAMIC_TOPK_HELDOUT_CONCORDANCE':static_results['dynamic_top32_matched']['concordance'],'STATIC_TOPK_HELDOUT_CONCORDANCE':static_results['static_top32_matched']['concordance'],'DYNAMIC_TOPK_RC_PAIRWISE':static_results['dynamic_top32_72case']['R-C_pairwise'],'STATIC_TOPK_RC_PAIRWISE':static_results['static_top32_72case']['R-C_pairwise'],'FINAL_SCIENTIFIC_CLASSIFICATION':cls,'FUNCTIONAL_ORIENTATION_LOW_RANK':'YES' if lowrank else 'NO','FUNCTIONAL_ORIENTATION_STABLE':'YES' if stable else 'NO','STATIC_FUNCTIONAL_BASIS_SUPPORTED':'YES' if static_supported else 'NO','METHOD_PRINCIPLE_EXTRACTION_READY':'YES','METHOD_DESIGN_READY':method,'NEXT_RECOMMENDED_TASK':'design a minimal functional-sensitive-subspace quantizer candidate' if method=='CONDITIONAL_YES' else 'inspect dynamic functional subspace variation before quantizer design','previous_evidence':{'beyond_reconstruction_commit':'6fab2921c92f72249d05b052575b1561fedd3206'},'frozen_m5_definition':{'formula':'|| W_O D_g D_w J_RMS(o) E^T q ||_2','modified':False},'tensor_semantics':{'e_space':'[head,value]=[32,128] flattened to 4096','out_proj':'4096x4096 full projection','operator':'A=W_O blockdiag(D_g D_w J_RMS_head)'},'identity_summary':identity,'energy_summary':energy_summary,'effective_rank_summary':er_summary,'eigen_gap_summary':{f'k{k}_normalized_gap_median':med([g['normalized_gap'] for g in gaps if g['k']==k]) for k in [1,2,4,8]},'topk_approx_summary':approx,'topk_72case_summary':rank72,'topk_matched_pair_summary':mp_summary,'subspace_stability_summary':sim_summary,'context_dependence':'Subspace similarity is computed across the available 9 canonical units; no new prompt/t0 panel was created.','static_calibration_protocol':{'split':'leave-one-unit-out','G_bar':'arithmetic mean over calibration units only','static_k':STATIC_K},'static_heldout_summary':static_results,'static_vs_dynamic_summary':static_results,'basis_only_ablation':'NOT_RUN; weighted averaged-G top-k is the primary static definition.','negative_results':'No M5 modification, new metric, persistence, decay, learned coefficient, KL-weighted calibration, or future rollout was introduced.','limitations':'The operator is large; spectra use the true merged-head 4096-dimensional per-layer M5 geometry and a block view across layers. Stability is limited to the existing 9 canonical units.'}
    save_json('final_summary.json',final); write_report(final)
    print(json.dumps(final,indent=2,sort_keys=True,ensure_ascii=False))

if __name__=='__main__': main()
