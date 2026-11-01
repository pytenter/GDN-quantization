#!/usr/bin/env python3
# Frozen CPU-only analysis of 80 immutable Observability Bridge canonical units.
from pathlib import Path
import csv, hashlib, html, itertools, json, math, os
DEFAULT_OUT=Path('/data/zypan/experiment_runs/LING_OBSERVABILITY_MECHANISM_TRIAGE_V1')
OUT=Path(os.environ.get('LING_TRIAGE_OUT', str(DEFAULT_OUT)))
PLAN_SRC=Path(os.environ.get('LING_TRIAGE_PLAN', str(DEFAULT_OUT/'ANALYSIS_PLAN.json')))
OUT.mkdir(parents=True,exist_ok=True)
if not (OUT/'ANALYSIS_PLAN.json').exists():
 data=PLAN_SRC.read_bytes()
 with (OUT/'ANALYSIS_PLAN.json').open('xb') as f:
  f.write(data);f.flush();os.fsync(f.fileno())
ROOTS=[
 ('historical_handoff10',Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/discovery/unit_results')),
 ('attempt04_gpu6_new35',Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1/hardware_expansion_gpu3_v1/discovery_shards/gpu6/discovery/unit_results')),
 ('attempt04_gpu3_new35',Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1/hardware_expansion_gpu3_v1/discovery_shards/gpu3/discovery/unit_results'))]
DOC_ORDER=['pg19-test-37702','pg19-test-28988','pg19-test-5396','pg19-test-26493'];CORE={'pg19-test-37702','pg19-test-5396'};AUX=set(DOC_ORDER)-CORE;ANCHOR_ORDER=[8192,2048]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def dumpx(name,obj):
 p=OUT/name
 with p.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 return p
def writex(name,text):
 p=OUT/name
 with p.open('x') as f:f.write(text if text.endswith('\n') else text+'\n');f.flush();os.fsync(f.fileno())
 return p
def rankdata(xs):
 n=len(xs);order=sorted(range(n),key=lambda i:xs[i]);r=[0.0]*n;i=0
 while i<n:
  j=i+1
  while j<n and xs[order[j]]==xs[order[i]]:j+=1
  v=(i+1+j)/2
  for k in range(i,j):r[order[k]]=v
  i=j
 return r
def pearson(x,y):
 if len(x)<2:return None
 mx=sum(x)/len(x);my=sum(y)/len(y);xx=sum((a-mx)**2 for a in x);yy=sum((b-my)**2 for b in y)
 if xx==0 or yy==0:return None
 return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy)
def spear(x,y):return pearson(rankdata(x),rankdata(y))
def solve(A,b):
 n=len(b);M=[list(map(float,A[i]))+[float(b[i])] for i in range(n)]
 for c in range(n):
  pivot=max(range(c,n),key=lambda r:abs(M[r][c]))
  if abs(M[pivot][c])<1e-12:return None
  M[c],M[pivot]=M[pivot],M[c];q=M[c][c];M[c]=[v/q for v in M[c]]
  for r in range(n):
   if r==c:continue
   q=M[r][c];M[r]=[M[r][j]-q*M[c][j] for j in range(n+1)]
 return [M[i][-1] for i in range(n)]
def fit(X,y):
 Z=[[1.0]+list(row) for row in X];p=len(Z[0]);A=[[sum(row[i]*row[j] for row in Z) for j in range(p)] for i in range(p)];b=[sum(row[i]*yy for row,yy in zip(Z,y)) for i in range(p)];return solve(A,b)
def pred(beta,row):return beta[0]+sum(b*x for b,x in zip(beta[1:],row))
def residualize(y,x):
 b=fit([[v] for v in x],y);return [yy-pred(b,[xx]) for yy,xx in zip(y,x)]
def r2(X,y):
 b=fit(X,y)
 if b is None:return None
 ss=sum((v-sum(y)/len(y))**2 for v in y);return 1-sum((v-pred(b,row))**2 for row,v in zip(X,y))/ss if ss else None
def rank_against(train,v):return 1+sum(x<v for x in train)+0.5*sum(x==v for x in train)
def loocv(rows,transform):
 ps1=[];ps2=[];ys=[]
 for i,t in enumerate(rows):
  tr=[r for j,r in enumerate(rows) if j!=i]
  if transform=='rank':
   jtr=[r['JPROP'] for r in tr];ptr=[r['POR'] for r in tr];ytr=[r['FutureKL'] for r in tr]
   X1=[[v] for v in rankdata(jtr)];X2=[list(v) for v in zip(rankdata(jtr),rankdata(ptr))];Y=rankdata(ytr);tx1=[rank_against(jtr,t['JPROP'])];tx2=[tx1[0],rank_against(ptr,t['POR'])];ty=rank_against(ytr,t['FutureKL'])
  else:
   jtr=[math.log10(r['JPROP']) for r in tr];ptr=[math.log10(r['POR']) for r in tr];ytr=[math.log10(r['FutureKL']) for r in tr]
   X1=[[v] for v in jtr];X2=[list(v) for v in zip(jtr,ptr)];Y=ytr;tx1=[math.log10(t['JPROP'])];tx2=[tx1[0],math.log10(t['POR'])];ty=math.log10(t['FutureKL'])
  b1=fit(X1,Y);b2=fit(X2,Y);ps1.append(pred(b1,tx1));ps2.append(pred(b2,tx2));ys.append(ty)
 def met(p):
  e=[a-b for a,b in zip(p,ys)];return {'rmse':math.sqrt(sum(v*v for v in e)/len(e)),'mae':sum(abs(v) for v in e)/len(e)}
 a,b=met(ps1),met(ps2);return {'magnitude_only':a,'magnitude_plus_POR':b,'delta_rmse_magnitude_minus_plus':a['rmse']-b['rmse'],'relative_rmse_improvement':(a['rmse']-b['rmse'])/a['rmse'] if a['rmse'] else None}
def analyze_panel(rows):
 j=[r['JPROP'] for r in rows];p=[r['POR'] for r in rows];y=[r['FutureKL'] for r in rows];rj,rp,ry=rankdata(j),rankdata(p),rankdata(y);partial=pearson(residualize(ry,rj),residualize(rp,rj));full_delta=spear(p,y)-spear(j,y);loo=[]
 for i,r in enumerate(rows):
  z=rows[:i]+rows[i+1:];loo.append({'omitted_condition':r['condition'],'delta_rho':spear([q['POR'] for q in z],[q['FutureKL'] for q in z])-spear([q['JPROP'] for q in z],[q['FutureKL'] for q in z])})
 matches=[]
 for a,b in itertools.combinations(rows,2):
  gap=abs(a['JPROP']-b['JPROP'])/max(abs(a['JPROP']),abs(b['JPROP']));ratio=max(a['POR'],b['POR'])/min(a['POR'],b['POR']) if min(a['POR'],b['POR'])>0 else math.inf
  if gap<=0.10 and ratio>=1.50:
   hi,lo=(a,b) if a['POR']>b['POR'] else (b,a);matches.append({'condition_low_POR':lo['condition'],'condition_high_POR':hi['condition'],'relative_JPROP_gap':gap,'POR_ratio_high_over_low':ratio,'FutureKL_low_POR':lo['FutureKL'],'FutureKL_high_POR':hi['FutureKL'],'FutureKL_difference_high_minus_low':hi['FutureKL']-lo['FutureKL'],'direction_supports_observability':hi['FutureKL']>lo['FutureKL']})
 n=len(rows);qr=[]
 for r,a,b,c in zip(rows,rj,rp,ry):
  if a>0.75*n and b<=0.25*n:qr.append({'condition':r['condition'],'JPROP_rank':a,'POR_rank':b,'FutureKL_rank':c,'JPROP':r['JPROP'],'POR':r['POR'],'FutureKL':r['FutureKL']})
 return {'n_conditions':len(rows),'rho_JPROP_FutureKL':spear(j,y),'rho_POR_FutureKL':spear(p,y),'DeltaRho':full_delta,'rho_JPROP_POR_collinearity':spear(j,p),'partial_rank_POR_FutureKL_given_JPROP':partial,'rank_model_in_sample_R2':{'magnitude_only':r2([[v] for v in rj],ry),'magnitude_plus_POR':r2([list(v) for v in zip(rj,rp)],ry)},'loocv_rank':loocv(rows,'rank'),'loocv_log10':loocv(rows,'log'),'leave_one_condition_out_DeltaRho':{'min':min(v['delta_rho'] for v in loo),'max':max(v['delta_rho'] for v in loo),'rows':loo},'magnitude_matched_pairs':matches,'magnitude_matched_summary':{'count':len(matches),'support_count':sum(x['direction_supports_observability'] for x in matches),'counterexample_count':sum(not x['direction_supports_observability'] for x in matches)},'high_magnitude_low_observability_cases':qr,'condition_rows':[{'condition':r['condition'],'JPROP':r['JPROP'],'POR':r['POR'],'FutureKL':r['FutureKL'],'JPROP_rank':a,'POR_rank':b,'FutureKL_rank':c} for r,a,b,c in zip(rows,rj,rp,ry)]}
units=[];manifest=[]
for source,root in ROOTS:
 for p in sorted(root.glob('*.json')):
  if p.name.endswith('.sha256.json'):continue
  s=p.with_name(p.stem+'.sha256.json');assert s.is_file(),('MISSING_SIDECAR',p);sj=load(s);digest=sha(p);size=p.stat().st_size;assert sj['sha256']==digest and sj['bytes']==size and Path(sj['path']).resolve()==p.resolve(),('SIDECAR_FAIL',p)
  u=load(p);key=(u['document'],u['condition'],int(u['anchor']),int(u['endpoint']));assert u['gate']=='PASS' and key[3]==32768;sc=u['scores'];tg=u['target'];required=['RAW_ACC_ERROR','LOCAL_K3_ACCUMULATED','HISTORICAL_J_PROP','J_INJ','J_ACC','PULSE_PROP_ENERGY_AUC','POR','POR_GAIN','future_tokens'];assert all(k in sc for k in required) and 'TARGET_FUTURE_KL_AUC' in tg
  values=[sc[k] for k in required if isinstance(sc[k],(int,float))]+[tg['TARGET_FUTURE_KL_AUC']];assert all(math.isfinite(float(v)) for v in values),('NONFINITE',p);assert sc['PULSE_PROP_ENERGY_AUC']>0 and sc['POR']>0 and tg['TARGET_FUTURE_KL_AUC']>0
  row={'document':u['document'],'condition':u['condition'],'anchor':int(u['anchor']),'endpoint':int(u['endpoint']),'anchor_role':'Primary' if int(u['anchor'])==8192 else 'Secondary','source_group':source,'core_or_auxiliary':'core_complete_document' if u['document'] in CORE else 'auxiliary_incomplete_document','physical_gpu':u['hardware']['physical_gpu'],'gpu_uuid':u['hardware']['gpu_uuid'],'premeasurement_commit':u['premeasurement_commit'],'token_sha256':u['token_sha256'],'rotation_sha256':u['rotation_sha256'],'same_gpu_fp_reference_sha256':u.get('same_gpu_fp_reference_sha256'),'RAW_ACC_ERROR':sc['RAW_ACC_ERROR'],'LOCAL_K3_ACCUMULATED':sc['LOCAL_K3_ACCUMULATED'],'HISTORICAL_J_PROP':sc['HISTORICAL_J_PROP'],'J_INJ':sc['J_INJ'],'J_ACC':sc['J_ACC'],'JPROP':sc['PULSE_PROP_ENERGY_AUC'],'POR':sc['POR'],'POR_GAIN':sc['POR_GAIN'],'FutureKL':tg['TARGET_FUTURE_KL_AUC'],'future_tokens':sc['future_tokens'],'teacher_tokens':tg['teacher_tokens'],'initial_pulse_gate':tg['INITIAL_PULSE_IDENTITY_GATE'],'fp_pairing_gate':tg['FP_REFERENCE_PAIRING_GATE'],'no_further_qdq_gate':tg['NO_FURTHER_QDQ_GATE'],'top_K3_event_count':len(sc.get('top_K3_events',[])),'canonical_path':str(p),'canonical_sha256':digest,'canonical_bytes':size,'sidecar_path':str(s),'sidecar_sha256':sha(s)}
  units.append((key,row));manifest.append({'key':list(key),'source_group':source,'canonical':{'path':str(p),'bytes':size,'sha256':digest},'sidecar':{'path':str(s),'bytes':s.stat().st_size,'sha256':sha(s)},'premeasurement_commit':u['premeasurement_commit'],'gpu_uuid':u['hardware']['gpu_uuid'],'token_sha256':u['token_sha256'],'rotation_sha256':u['rotation_sha256'],'same_gpu_fp_reference_sha256':u.get('same_gpu_fp_reference_sha256')})
keys=[k for k,_ in units];assert len(units)==80 and len(set(keys))==80;rows=[r for _,r in units];keyset=set(keys)
for r in rows:r['pair_complete']=(r['document'],r['condition'],2048 if r['anchor']==8192 else 8192,32768) in keyset
assert sum(r['anchor']==8192 for r in rows)==41 and sum(r['anchor']==2048 for r in rows)==39 and sum(r['pair_complete'] for r in rows)==78
for d in CORE:assert sum(r['document']==d and r['anchor']==8192 for r in rows)==16 and sum(r['document']==d and r['anchor']==2048 for r in rows)==16
fields=['document','condition','anchor_role','anchor','endpoint','core_or_auxiliary','pair_complete','source_group','physical_gpu','gpu_uuid','premeasurement_commit','RAW_ACC_ERROR','LOCAL_K3_ACCUMULATED','HISTORICAL_J_PROP','J_INJ','J_ACC','JPROP','POR','POR_GAIN','FutureKL','future_tokens','teacher_tokens','top_K3_event_count','initial_pulse_gate','fp_pairing_gate','no_further_qdq_gate','token_sha256','rotation_sha256','same_gpu_fp_reference_sha256','canonical_path','canonical_sha256','canonical_bytes','sidecar_path','sidecar_sha256']
with (OUT/'OBSERVABILITY_ANALYSIS_DATASET.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda r:(DOC_ORDER.index(r['document']),ANCHOR_ORDER.index(r['anchor']),r['condition'])))
dumpx('CANONICAL_SOURCE_MANIFEST.json',{'gate':'PASS','expected':80,'verified':80,'duplicate_keys':0,'unknown_keys':0,'Primary':41,'Secondary':39,'complete_pairs':39,'unpaired_primary':2,'files':sorted(manifest,key=lambda x:tuple(x['key']))})
results={'method':load(OUT/'ANALYSIS_PLAN.json'),'panels':{},'anchor_stability':{},'auxiliary':{},'global_counts':{'units':80,'Primary':41,'Secondary':39,'complete_pairs':39,'core_units':64,'auxiliary_units':16}};matched=[]
for d in [x for x in DOC_ORDER if x in CORE]:
 results['panels'][d]={}
 for a in ANCHOR_ORDER:
  rr=sorted([r for r in rows if r['document']==d and r['anchor']==a],key=lambda r:r['condition']);q=analyze_panel(rr);results['panels'][d][str(a)]=q
  for m in q['magnitude_matched_pairs']:matched.append({'document':d,'anchor':a,**m})
 p1=sorted([r for r in rows if r['document']==d and r['anchor']==8192],key=lambda r:r['condition']);p2=sorted([r for r in rows if r['document']==d and r['anchor']==2048],key=lambda r:r['condition']);assert [r['condition'] for r in p1]==[r['condition'] for r in p2]
 results['anchor_stability'][d]={'rho_condition_JPROP_primary_secondary':spear([r['JPROP'] for r in p1],[r['JPROP'] for r in p2]),'rho_condition_POR_primary_secondary':spear([r['POR'] for r in p1],[r['POR'] for r in p2]),'rho_condition_FutureKL_primary_secondary':spear([r['FutureKL'] for r in p1],[r['FutureKL'] for r in p2])}
for d in [x for x in DOC_ORDER if x in AUX]:
 results['auxiliary'][d]={}
 for a in ANCHOR_ORDER:
  rr=sorted([r for r in rows if r['document']==d and r['anchor']==a],key=lambda r:r['condition'])
  if not rr:continue
  results['auxiliary'][d][str(a)]={'n':len(rr),'conditions':[r['condition'] for r in rr],'rho_JPROP_FutureKL':spear([r['JPROP'] for r in rr],[r['FutureKL'] for r in rr]) if len(rr)>=3 else None,'rho_POR_FutureKL':spear([r['POR'] for r in rr],[r['FutureKL'] for r in rr]) if len(rr)>=3 else None,'DeltaRho':(spear([r['POR'] for r in rr],[r['FutureKL'] for r in rr])-spear([r['JPROP'] for r in rr],[r['FutureKL'] for r in rr])) if len(rr)>=3 else None}
rot=load(Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/rotation_manifest.json'))['bank']['execution_order'];results['rotation_execution_order']=rot
for d,v in results['auxiliary'].items():
 for a,q in v.items():q['observed_execution_indices']=[rot.index(c) for c in q['conditions']];q['is_execution_prefix']=set(q['conditions'])==set(rot[:len(q['conditions'])])
dumpx('ANALYSIS_RESULTS.json',results)
with (OUT/'MAGNITUDE_MATCHED_PAIRS.csv').open('x',newline='') as f:
 fields2=['document','anchor','condition_low_POR','condition_high_POR','relative_JPROP_gap','POR_ratio_high_over_low','FutureKL_low_POR','FutureKL_high_POR','FutureKL_difference_high_minus_low','direction_supports_observability'];w=csv.DictWriter(f,fieldnames=fields2);w.writeheader();w.writerows(matched)
W,H=1200,850;pw=520;ph=330;svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif}.t{font-size:18px;font-weight:bold}.l{font-size:11px}.a{font-size:13px}</style>']
for idx,(d,a) in enumerate([(d,a) for d in [x for x in DOC_ORDER if x in CORE] for a in ANCHOR_ORDER]):
 x0=70+(idx%2)*580;y0=60+(idx//2)*390;rr=sorted([r for r in rows if r['document']==d and r['anchor']==a],key=lambda r:r['condition']);rj=rankdata([r['JPROP'] for r in rr]);rp=rankdata([r['POR'] for r in rr]);ry=rankdata([r['FutureKL'] for r in rr]);svg += [f'<text class="t" x="{x0}" y="{y0-20}">{html.escape(d)} — {"Primary" if a==8192 else "Secondary"}</text>',f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="black"/>',f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="black"/>',f'<text class="a" x="{x0+180}" y="{y0+ph+35}">JPROP rank</text>',f'<text class="a" transform="translate({x0-42},{y0+220}) rotate(-90)">Future-KL rank</text>']
 for r,xr,cr,yr in zip(rr,rj,rp,ry):
  x=x0+(xr-1)/15*pw;y=y0+ph-(yr-1)/15*ph;red=int(255*(cr-1)/15);blue=255-red;svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="rgb({red},70,{blue})" stroke="black"/><text class="l" x="{x+7:.1f}" y="{y-5:.1f}">{html.escape(r["condition"])}</text>')
svg += ['<text class="a" x="70" y="830">Color: blue=low POR rank, red=high POR rank. Each panel contains 16 related rotation conditions; panels are not independent-sample tests.</text>','</svg>'];writex('EXISTING_80_SCATTER_PANELS.svg','\n'.join(svg))
audit='''# Available metrics audit

## Canonical inventory

- Verified canonical + sidecar bytes: **80/80**
- Unique `(document, condition, anchor, endpoint)` keys: **80**; duplicates: **0**
- Primary: **41/64**; Secondary: **39/64**; complete pairs: **39**; unpaired Primary: **2**
- Core complete documents: `pg19-test-37702`, `pg19-test-5396` (64 units, 32 pairs)
- Auxiliary incomplete documents: `pg19-test-28988`, `pg19-test-26493` (16 units, 7 pairs + 2 unpaired Primary)
- All scalar analysis fields are present, finite, and strictly positive where log sensitivity is used. No imputation was performed.

## Actual metric definitions

- `RAW_ACC_ERROR`: squared Frobenius norm of the anchor error `E = quantized_state - FP_state`.
- `LOCAL_K3_ACCUMULATED`: squared norm of historical K3 query/readout applied to anchor error.
- `HISTORICAL_J_PROP`: unchanged historical replay-step statistic, averaged over its historical support; retained separately and **not** used as the primary propagation-magnitude variable.
- `J_INJ`, `J_ACC`: unchanged historical replay-step summaries.
- `JPROP` in this analysis: canonical `PULSE_PROP_ENERGY_AUC`, the sum over future tokens of squared Frobenius norm between the FP path and the pulse-injected path.
- `POR`: sum over the same future tokens of squared K3 readout difference between those two paths.
- `POR_GAIN`: `POR / RAW_ACC_ERROR`, with the original zero-denominator rule.
- `FutureKL`: canonical `TARGET_FUTURE_KL_AUC`, the accumulated full-logit KL between the FP reference path and the pulse-injected path.

`POR` is calculated in `bridge.py::replay` from state-path differences and future K3 query/readout records. `FutureKL` is calculated separately in `bridge.py::pulse_target` from full logits. POR does **not** read the Future-KL scalar or logit target, so there is no direct label leakage. It does use the realized future token/query/readout sequence, making it a retrospective future-aware diagnostic rather than a presently deployable training objective.

## Alignment

For each unit, JPROP, POR and FutureKL use the same document, condition, anchor, endpoint=32768 and future window. Primary and Secondary use identical metric definitions but different pulse anchors and hence different future-token counts. All units preserve same-GPU FP pairing, initial pulse identity and no-further-QDQ gates.

## Rotation comparability and missingness

Each complete document has the full frozen 16-condition rotation panel at both anchors. Auxiliary documents are analyzed only on observed conditions. Their missingness follows scheduler execution order after the reboot rather than random sampling, so auxiliary correlations are selection-biased descriptive checks only.
''';writex('AVAILABLE_METRICS_AUDIT.md',audit);print('ANALYSIS_COMPLETE',len(rows),len(matched))
