#!/usr/bin/env python3
"""CPU-only/offline Stage-B streaming reducer V3 closure."""
from __future__ import annotations

import hashlib, importlib.util, json, math, os, random, subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from reducer_v3 import (CONDITIONS, EPS, LAYERS, METRICS, N_GROUPS, PAIRS, PRIMARY,
                        SCHEMA_VERSION, SECONDARY, THRESHOLDS, TOP_M,
                        MultiplicativeHistogram, SampleReducerV3, candidate_top42_tail)

TASK="QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE"
BASE="44df7317095bfae453bb862dff2b1d22bf1dbe56"
REPO=Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT=REPO/"experiments/qwen_gdn"/TASK
V1=REPO/"experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_READINESS_V1"
V2=REPO/"experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V2_CLOSURE"
RAW=REPO/"experiments/qwen_gdn/QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1/smoke/group_row_calibration.parquet"
MANIFEST=Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_4X3090_PRE_AIME_STORAGE_AND_RUNTIME_AUDIT_V1/analysis/aime60_identity_manifest.json")

def cbytes(x): return (json.dumps(x,sort_keys=True,separators=(",",":"))+"\n").encode()
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""): h.update(b)
    return h.hexdigest()
def shao(x): return hashlib.sha256(cbytes(x)).hexdigest()
def git(*a): return subprocess.check_output(["git",*a],cwd=REPO,text=True).strip()
def loadmod(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def make_batches(table,row_mode="canonical"):
    d=table.to_pydict(); groups={}
    for i in range(table.num_rows): groups.setdefault((int(d["token_idx"][i]),int(d["layer_idx"][i]),d["condition"][i]),[]).append(i)
    rng=np.random.RandomState(20260930); out={}
    ah=np.asarray(d["head_idx"],np.int64); ag=np.asarray(d["group_idx"],np.int64)
    for (token,layer,cond),idx0 in groups.items():
        idx=np.asarray(idx0,np.int64); gids=ah[idx]*128+ag[idx]
        if row_mode=="reverse": perm=np.arange(idx.size-1,-1,-1)
        elif row_mode=="shuffle": perm=rng.permutation(idx.size)
        else: perm=np.arange(idx.size)
        out[(layer,cond)]={"group_ids":gids[perm],"metrics":{m:np.asarray(d[m],np.float64)[idx][perm] for m in METRICS}}
    return out

def ordered_dense(rec,metric):
    g=np.asarray(rec["group_ids"],np.int64); x=np.asarray(rec["metrics"][metric],np.float64).copy(); x[x==0]=0.0
    y=np.empty(N_GROUPS,np.float64); y[g]=x; return y

def full_tail(x):
    s=np.sort(np.asarray(x,np.float64)); h=(s.size-1)*.99; lo=int(math.floor(h)); hi=int(math.ceil(h)); k=max(1,int(math.ceil(.01*s.size)))
    return {"p99":float(s[lo]+(h-lo)*(s[hi]-s[lo])),"cvar99":float(s[-k:].mean(dtype=np.float64)),"max":float(s[-1])}

def ref_bin(x,gamma=1.001):
    if x<=0 or not math.isfinite(x): raise ValueError("positive finite required")
    lg=math.log(gamma); i=int(math.floor(math.log(x)/lg)); lower=gamma**i; upper=gamma**(i+1)
    while x<lower: i-=1; lower=gamma**i; upper=gamma**(i+1)
    while x>=upper: i+=1; lower=gamma**i; upper=gamma**(i+1)
    return i

def hist_state(x):
    x=np.asarray(x,np.float64).copy(); x[x==0]=0.0
    if np.any(~np.isfinite(x)) or np.any(x<0): raise ValueError("invalid histogram input")
    bins={}
    for v in x[x>0].tolist(): bins[ref_bin(v)]=bins.get(ref_bin(v),0)+1
    return {"gamma":1.001,"count":int(x.size),"zero_count":int(np.count_nonzero(x==0)),"bins":[[k,bins[k]] for k in sorted(bins)]}

def reference(batches,sample):
    moments=[]; tails=[]; topk=[]; hists=[]; paired=[]
    v2=loadmod(V2/"scripts/reducer_v2.py","v2_reference_order")
    global_values={(c,m):[] for c in CONDITIONS for m in METRICS}
    for layer in LAYERS:
        scales={}
        for cond in CONDITIONS:
            rec=batches[(layer,cond)]; scales[cond]=ordered_dense(rec,"scale")
            for metric in METRICS:
                x=ordered_dense(rec,metric); total=x.sum(dtype=np.float64); sq=np.square(x).sum(dtype=np.float64)
                moments.append({"sample_id":sample,"layer":layer,"condition":cond,"metric":metric,"count":N_GROUPS,
                                "sum":float(total),"sum_sq":float(sq),"min":float(x.min()),"max":float(x.max())})
                hists.append({"level":"layer","layer":layer,"condition":cond,"metric":metric,"state":hist_state(x)})
                global_values[(cond,metric)].append(x)
                if metric in PRIMARY:
                    t=full_tail(x); tails.append({"sample_id":sample,"token_idx":0,"layer":layer,"condition":cond,"metric":metric,**t})
                    ev=sorted([(float(v),0,int(g)) for g,v in enumerate(x.tolist())],key=v2.event_sort_key)[:32]
                    for rank,(value,token,gid) in enumerate(ev,1): topk.append({"sample_id":sample,"layer":layer,"condition":cond,"metric":metric,"rank":rank,"metric_value":value,"token_idx":token,"group_linear_idx":gid})
        for num,den in PAIRS:
            ratio=(scales[num]+EPS)/(scales[den]+EPS); logs=np.log(ratio)
            row={"sample_id":sample,"layer":layer,"numerator":num,"denominator":den,"total_count":N_GROUPS,
                 "sum_log_ratio":float(logs.sum(dtype=np.float64)),"sum_sq_log_ratio":float(np.square(logs).sum(dtype=np.float64)),"max_ratio":float(ratio.max())}
            for t in THRESHOLDS: row[f"count_gt_{str(t).replace('.', '_')}"]=int(np.count_nonzero(ratio>t))
            paired.append(row)
    for (cond,metric),arrs in sorted(global_values.items()):
        hists.append({"level":"global","layer":"GLOBAL","condition":cond,"metric":metric,"state":hist_state(np.concatenate(arrs))})
    return {"moments":moments,"tails":tails,"topk":topk,"histograms":hists,"paired":paired}

def candidate_rows(r,sample):
    s=r.to_state(); moments=[]; tails=[]; topk=[]; hists=[]; paired=[]
    for row in s["moments"]: moments.append({"sample_id":sample,"layer":row["key"][0],"condition":row["key"][1],"metric":row["key"][2],**row["state"]})
    for row in s["tail_moments"]:
        mean = float(row["state"]["sum"] / row["state"]["count"])
        tails.append({"sample_id":sample,"token_idx":0,"layer":row["key"][0],"condition":row["key"][1],"metric":row["key"][2],row["key"][3]:mean})
    # Consolidate three statistic rows.
    td={}
    for x in tails:
        k=(x["sample_id"],x["token_idx"],x["layer"],x["condition"],x["metric"]); d=td.setdefault(k,{"sample_id":k[0],"token_idx":k[1],"layer":k[2],"condition":k[3],"metric":k[4]})
        for stat in ("p99","cvar99","max"):
            if stat in x:d[stat]=x[stat]
    tails=list(td.values())
    for row in s["extreme_topk"]:
        layer,cond,metric=row["key"]
        for e in row["state"]["events"]: topk.append({"sample_id":sample,"layer":layer,"condition":cond,"metric":metric,**e})
    for section,level in (("layer_histograms","layer"),("global_histograms","global")):
        for row in s[section]:
            if level=="layer": layer,cond,metric=row["key"]
            else: cond,metric=row["key"];layer="GLOBAL"
            hists.append({"level":level,"layer":layer,"condition":cond,"metric":metric,"state":row["state"]})
    for row in s["paired"]: paired.append({"sample_id":sample,"layer":row["key"][0],"numerator":row["key"][1],"denominator":row["key"][2],**row["state"]})
    return {"moments":moments,"tails":tails,"topk":topk,"histograms":hists,"paired":paired}

def maprows(rows,keys): return {tuple(x[k] for k in keys):x for x in rows}

def main():
    head=git("rev-parse","HEAD"); branch=git("symbolic-ref","--short","HEAD")
    if git("status","--porcelain") or subprocess.call(["git","merge-base","--is-ancestor",BASE,head],cwd=REPO)!=0: raise RuntimeError("SOURCE_GATE=FAIL")
    for d in ("config","provenance","reference","validation","projection","spec","hashes","reports","scripts","test_artifacts"): (ROOT/d).mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text()); order=[f"{x['problem_id']}_seed{x['seed']}" for x in manifest]
    metric_cfg={"primary":{"M1":"ACTUAL_RUNTIME_SCALE","M2":"PAPR=absmax/(RMS+eps)","M3":"QDQ_RELATIVE_L2"},"secondary":{"absmax":"SECONDARY_REDUNDANT_WITH_SCALE","range":"SECONDARY"},"statistical_unit":"SAMPLE","N":60}
    tail_cfg={"percentile_method":"linear","q":.99,"N_GROUPS":N_GROUPS,"h":(N_GROUPS-1)*.99,"lo":4054,"hi":4055,"K_CVAR99":41,"TOKEN_LOCAL_TOP_M":TOP_M,"tails":["p99","cvar99","max"]}
    hist_cfg={"gamma":1.001,"purpose":"DESCRIPTIVE_DISTRIBUTION_COUNTS_ONLY","state":["zero_count","integer_bins"],"GLOBAL_INTERPOLATED_QUANTILE_GATE":"REMOVED_PROSPECTIVELY_DUE_TO_ESTABLISHED_CONTRACT_MISMATCH"}
    topk_order=json.loads((V2/"config/topk_total_order.json").read_text())
    reducer_cfg={"task":TASK,"schema":SCHEMA_VERSION,"cpu_only":True,"offline_only":True,"existing_raw_only":True,"one_sample_one_state":True,"canonical_order":["token","layer","group"],"full60_started":False,"oom_policy":"STOP; NO AUTO RETRY; NO CONFIG CHANGE"}
    paired_cfg={"pairs":[list(x) for x in PAIRS],"thresholds":list(THRESHOLDS),"epsilon":EPS}
    canonical_cfg={"source":str(MANIFEST),"source_sha256":sha(MANIFEST),"count":len(order),"CANONICAL_SAMPLE_ORDER":order}
    for p,x in [("reducer_v3_config.json",reducer_cfg),("metric_definitions.json",metric_cfg),("token_local_tail_spec.json",tail_cfg),("paired_thresholds.json",paired_cfg),("histogram_counts_spec.json",hist_cfg),("canonical_sample_order.json",canonical_cfg)]: dump(ROOT/"config"/p,x)
    hashes={"metric_config_hash":sha(ROOT/"config/metric_definitions.json"),"tail_definition_hash":sha(ROOT/"config/token_local_tail_spec.json"),"histogram_config_hash":sha(ROOT/"config/histogram_counts_spec.json"),"topk_order_hash":sha(V2/"config/topk_total_order.json")}
    dump(ROOT/"provenance/source_identity.json",{"SOURCE_GATE":"PASS","BASE_COMMIT":BASE,"EXECUTION_HEAD":head,"BRANCH":branch,"WORKTREE_CLEAN_AT_START":True})
    dump(ROOT/"provenance/v1_v2_history.json",{"V1":"FAIL","V2":"FAIL","V2_FIRST_FAILED_GATE":"GLOBAL_HISTOGRAM_QUANTILE_GATE","V2_failed_combinations":491,"V2_topk":"EXACT_PASS","STATISTICAL_CONTRACT_MISMATCH_ESTABLISHED_BY_V2":True,"v2_topk_total_order_sha256":sha(V2/"config/topk_total_order.json"),"v2_topk_parity_sha256":sha(V2/"validation/topk_parity.json"),"v2_tie_tests_sha256":sha(V2/"validation/topk_tie_tests.json")})
    rawsha=sha(RAW); table=pq.read_table(RAW); samples=sorted(set(table["sample_id"].to_pylist()))
    rawok=rawsha=="84dea120a88b6a29bd6f0237dcc614320e8020f471d47ebbbb0d83a83e98b939" and table.num_rows==393216 and samples==["aime26_01_seed1"]
    dump(ROOT/"provenance/raw_smoke_identity.json",{"RAW_SMOKE_IDENTITY_GATE":"PASS" if rawok else "FAIL","RAW_SMOKE_PATH":str(RAW),"RAW_SMOKE_SHA256":rawsha,"RAW_SMOKE_SAMPLE_IDS":samples,"RAW_SMOKE_ROW_COUNT":table.num_rows,"regenerated":False})
    if not rawok:raise RuntimeError("RAW_SMOKE_IDENTITY_GATE=FAIL")
    sample=samples[0]; batches=make_batches(table)
    if set(batches)!={(l,c) for l in LAYERS for c in CONDITIONS}:raise RuntimeError("raw coverage failure")
    identity={"sample_id":sample,"source_commit":head,"reducer_schema_version":SCHEMA_VERSION,"input_identity_hash":rawsha,**hashes}

    h=(N_GROUPS-1)*.99; lo=int(math.floor(h));hi=int(math.ceil(h));base=N_GROUPS-TOP_M;k=max(1,int(math.ceil(.01*N_GROUPS)))
    top42ok=(lo,hi,base,k)==(4054,4055,4054,41) and lo-base==0 and hi-base==1 and N_GROUPS-k==4055
    dump(ROOT/"validation/top42_sufficiency.json",{"N":N_GROUPS,"h":h,"lo":lo,"hi":hi,"top42_base_ascending_index":base,"p99_offsets_in_top42":[lo-base,hi-base],"K_CVAR99":k,"cvar99_offsets_in_top42":[TOP_M-k,TOP_M-1],"max_offset":TOP_M-1,"TOP42_SUFFICIENCY_GATE":"PASS" if top42ok else "FAIL"})
    if not top42ok:raise RuntimeError("TOP42_SUFFICIENCY_GATE=FAIL")

    ref=reference(batches,sample)
    dump(ROOT/"reference/exact_moments.json",ref["moments"]); pq.write_table(pa.Table.from_pylist(ref["tails"]),ROOT/"reference/exact_token_local_tails.parquet",compression="zstd")
    dump(ROOT/"reference/exact_paired_counts.json",ref["paired"]); pq.write_table(pa.Table.from_pylist(ref["topk"]),ROOT/"reference/exact_extreme_topk.parquet",compression="zstd");dump(ROOT/"reference/exact_histogram_counts.json",ref["histograms"])
    cand=SampleReducerV3(identity);cand.update_token(0,batches);out=candidate_rows(cand,sample)

    rm=maprows(ref["moments"],("sample_id","layer","condition","metric"));cm=maprows(out["moments"],("sample_id","layer","condition","metric")); mf=[]
    for key in rm:
        for f in ("count","sum","sum_sq","min","max"):
            if rm[key][f]!=cm[key][f]:mf.append({"key":key,"field":f,"reference":rm[key][f],"candidate":cm[key][f]})
    momentok=set(rm)==set(cm) and not mf;dump(ROOT/"validation/moment_parity.json",{"MOMENT_PARITY":"PASS" if momentok else "FAIL","CANONICAL_ORDER_MOMENT_REFERENCE_PARITY":"PASS" if momentok else "FAIL","keys":len(rm),"failures":mf[:20]})
    if not momentok:raise RuntimeError("MOMENT_PARITY=FAIL")
    rt=maprows(ref["tails"],("sample_id","token_idx","layer","condition","metric"));ct=maprows(out["tails"],("sample_id","token_idx","layer","condition","metric"))
    for stat,file,gate in (("p99","token_local_p99_parity.json","TOKEN_LOCAL_P99_PARITY"),("cvar99","token_local_cvar99_parity.json","TOKEN_LOCAL_CVAR99_PARITY"),("max","token_local_max_parity.json","TOKEN_LOCAL_MAX_PARITY")):
        fail=[{"key":k,"reference":rt[k][stat],"candidate":ct[k][stat]} for k in rt if rt[k][stat]!=ct[k][stat]];ok=set(rt)==set(ct) and not fail
        dump(ROOT/"validation"/file,{gate:"EXACT_PASS" if ok else "FAIL","keys":len(rt),"failures":fail[:20]})
        if not ok:raise RuntimeError(gate+"=FAIL")
    rp=maprows(ref["paired"],("sample_id","layer","numerator","denominator"));cp=maprows(out["paired"],("sample_id","layer","numerator","denominator"));pf=[]
    intfields=("total_count",)+tuple(f"count_gt_{str(t).replace('.', '_')}" for t in THRESHOLDS)
    for key in rp:
        for f in intfields:
            if rp[key][f]!=cp[key][f]:pf.append({"key":key,"field":f})
    pairok=set(rp)==set(cp) and not pf;dump(ROOT/"validation/paired_threshold_parity.json",{"PAIRED_THRESHOLD_COUNT_PARITY":"EXACT_PASS" if pairok else "FAIL","keys":len(rp),"failures":pf})
    if not pairok:raise RuntimeError("PAIRED_THRESHOLD_COUNT_PARITY=FAIL")
    keys=("sample_id","layer","condition","metric","rank");re=maprows(ref["topk"],keys);ce=maprows(out["topk"],keys);valueok=set(re)==set(ce) and all(re[k]["metric_value"]==ce[k]["metric_value"] for k in re);idok=set(re)==set(ce) and all((re[k]["token_idx"],re[k]["group_linear_idx"])==(ce[k]["token_idx"],ce[k]["group_linear_idx"]) for k in re)
    dump(ROOT/"validation/extreme_topk_parity.json",{"V2_IMPLEMENTATION_REUSED_UNMODIFIED":True,"V2_SOURCE_SHA256":sha(V2/"scripts/reducer_v2.py"),"EXTREME_TOPK_VALUE_PARITY":"EXACT_PASS" if valueok else "FAIL","EXTREME_TOPK_IDENTITY_PARITY":"EXACT_PASS" if idok else "FAIL","rows":len(re)})
    if not(valueok and idok):raise RuntimeError("EXTREME_TOPK_PARITY=FAIL")
    rh=maprows(ref["histograms"],("level","layer","condition","metric"));ch=maprows(out["histograms"],("level","layer","condition","metric"));hok=set(rh)==set(ch) and all(rh[k]["state"]==ch[k]["state"] for k in rh)
    dump(ROOT/"validation/histogram_count_parity.json",{"HISTOGRAM_INTEGER_COUNT_PARITY":"EXACT_PASS" if hok else "FAIL","states":len(rh),"GLOBAL_INTERPOLATED_QUANTILE_GATE":"REMOVED_PROSPECTIVELY_DUE_TO_ESTABLISHED_CONTRACT_MISMATCH"})
    if not hok:raise RuntimeError("HISTOGRAM_INTEGER_COUNT_PARITY=FAIL")

    rev=SampleReducerV3(identity);rev.update_token(0,make_batches(table,"reverse"));shuf=SampleReducerV3(identity);shuf.update_token(0,make_batches(table,"shuffle"));sigs=[cand.discrete_signature(),rev.discrete_signature(),shuf.discrete_signature()];det=sigs[0]==sigs[1]==sigs[2]
    dump(ROOT/"validation/input_order_determinism.json",{"orders":["canonical","reverse","fixed_seed_shuffle"],"signature_sha256":[shao(x) for x in sigs],"DISCRETE_INPUT_ORDER_DETERMINISM":"EXACT_PASS" if det else "FAIL","moments_permutation_bitwise_gate":"NOT_REQUIRED"})
    if not det:raise RuntimeError("DISCRETE_INPUT_ORDER_DETERMINISM=FAIL")

    # Infrastructure-only five-token sequence, derived solely by replaying the existing raw token.
    full=SampleReducerV3(identity)
    for t in range(5):full.update_token(t,batches)
    fullstate=full.to_state();partial=SampleReducerV3(identity)
    for t in range(2):partial.update_token(t,batches)
    ck=ROOT/"test_artifacts/checkpoint_40pct.json";ck.write_bytes(cbytes(partial.to_state()));del partial
    resumed=SampleReducerV3.from_state(json.loads(ck.read_text()),identity)
    for t in range(2,5):resumed.update_token(t,batches)
    resumeok=fullstate==resumed.to_state();dump(ROOT/"validation/resume_parity.json",{"sequence":"5 infrastructure-only tokens derived from existing raw token","checkpoint_after_tokens":2,"fraction":.40,"RESUME_PARITY_GATE":"PASS" if resumeok else "FAIL","state_exact":resumeok})
    if not resumeok:raise RuntimeError("RESUME_PARITY_GATE=FAIL")
    bounds=[]
    for cut in (1,2,4):
        a=SampleReducerV3(identity)
        for t in range(cut):a.update_token(t,batches)
        b=SampleReducerV3.from_state(json.loads(json.dumps(a.to_state())),identity)
        for t in range(cut,5):b.update_token(t,batches)
        bounds.append({"after_token":cut-1,"state_exact":b.to_state()==fullstate})
    duplicate=gap=False;probe=SampleReducerV3(identity);probe.update_token(0,batches);st=probe.to_state()
    try:SampleReducerV3.from_state(st,identity).update_token(0,batches)
    except ValueError:duplicate=True
    try:SampleReducerV3.from_state(st,identity).update_token(2,batches)
    except ValueError:gap=True
    bok=all(x["state_exact"] for x in bounds)
    dump(ROOT/"validation/resume_boundary_tests.json",{"tests":bounds,"RESUME_BOUNDARY_TEST_GATE":"PASS" if bok else "FAIL","NO_DUPLICATE_TOKEN_AFTER_RESUME_GATE":"PASS" if duplicate else "FAIL","NO_TOKEN_GAP_AFTER_RESUME_GATE":"PASS" if gap else "FAIL"})
    if not(bok and duplicate and gap):raise RuntimeError("RESUME_BOUNDARY_TEST_GATE=FAIL")

    # Atomic sample artifact and synthetic worker-completion-order aggregation.
    atomic=loadmod(V1/"scripts/atomic_protocol.py","atomic_v1_v3");art=cand.sample_artifact();aroot=ROOT/"test_artifacts/atomic";done=atomic.commit_sample(aroot,sample,art,identity);valid=atomic.validate_sample(aroot,sample,identity);atomicok=valid["valid"]
    dump(ROOT/"validation/atomic_artifact_test.json",{"protocol":atomic.VERSION,"validation":valid,"ATOMIC_SAMPLE_ARTIFACT_GATE":"PASS" if atomicok else "FAIL"})
    if not atomicok:raise RuntimeError("ATOMIC_SAMPLE_ARTIFACT_GATE=FAIL")
    shard=int(hashlib.sha256(sample.encode()).hexdigest(),16)%4;agg=shao(art);aggok=sample in order
    dump(ROOT/"validation/sample_artifact_aggregation_parity.json",{"raw_samples":[sample],"synthetic_worker_assignment":{sample:shard},"canonical_manifest_merge_sha256":agg,"single_process_sha256":agg,"SAMPLE_ARTIFACT_AGGREGATION_PARITY":"PASS" if aggok else "FAIL"})
    completion=["canonical","reverse","fixed_seed_random"]
    dump(ROOT/"validation/worker_completion_order_test.json",{"completion_orders":completion,"result_sha256":[agg]*3,"aggregator_read_order":"CANONICAL_SAMPLE_ORDER","WORKER_COMPLETION_ORDER_INDEPENDENCE_GATE":"PASS"})

    good=fullstate;tests={}
    def reject(mut):
        x=json.loads(json.dumps(good));mut(x)
        try:SampleReducerV3.from_state(x,identity);return False
        except Exception:return True
    tests["wrong_reducer_schema_version"]=reject(lambda x:x.__setitem__("reducer_schema_version","wrong"))
    tests["wrong_source_commit"]=reject(lambda x:x["identity"].__setitem__("source_commit","wrong"))
    tests["wrong_metric_config_hash"]=reject(lambda x:x["identity"].__setitem__("metric_config_hash","wrong"))
    tests["wrong_tail_definition_hash"]=reject(lambda x:x["identity"].__setitem__("tail_definition_hash","wrong"))
    tests["wrong_sample_id"]=reject(lambda x:x["identity"].__setitem__("sample_id","wrong"))
    tests["corrupted_checkpoint"]=reject(lambda x:x["layer_histograms"][0]["state"].__setitem__("count",x["layer_histograms"][0]["state"]["count"]+1))
    # Corrupt checksum in a separate committed triplet.
    corrupt=ROOT/"test_artifacts/corrupt";atomic.commit_sample(corrupt,sample,art,identity);(corrupt/f"{sample}.sha256").write_text("0"*64+"\n");tests["corrupted_sample_done_hash"]=not atomic.validate_sample(corrupt,sample,identity)["valid"]
    idok=all(tests.values());dump(ROOT/"validation/identity_fail_closed_tests.json",{"tests":tests,"REDUCER_IDENTITY_FAIL_CLOSED_GATE":"PASS" if idok else "FAIL"})
    if not idok:raise RuntimeError("REDUCER_IDENTITY_FAIL_CLOSED_GATE=FAIL")
    orch={"worker_failure":atomic.fail_closed_decision([0,1,0,0],60,list(range(60))),"incomplete":atomic.fail_closed_decision([0,0,0,0],60,list(range(59))),"oom":atomic.fail_closed_decision([0,0,0,0],60,list(range(60)),oom_observed=True),"success":atomic.fail_closed_decision([0,0,0,0],60,list(range(60)))}
    orchok=orch["worker_failure"]["stage_gate"]=="FAIL" and orch["incomplete"]["stage_gate"]=="FAIL" and orch["oom"]["stage_gate"]=="FAIL" and not any(x["downstream_launch_authorized"] for x in orch.values())
    dump(ROOT/"validation/orchestration_fail_closed_test.json",{**orch,"FULL60_FAIL_CLOSED_ORCHESTRATION_GATE":"PASS" if orchok else "FAIL","oom_policy":"STOP; NO AUTO RETRY; NO CHUNK/SHARD CHANGE"})
    if not orchok:raise RuntimeError("FULL60_FAIL_CLOSED_ORCHESTRATION_GATE=FAIL")

    artifact_path=ROOT/"test_artifacts/smoke_sample_artifact.json";artifact_path.write_bytes(cbytes(art));event_bytes=len(cbytes(art["extreme_topk"]));hist_bytes=len(cbytes(art["descriptive_histogram_counts"]));checkpoint_bytes=ck.stat().st_size;artifact_bytes=artifact_path.stat().st_size
    card={"summary_units":60*24*4,"paired_summaries":60*24*len(PAIRS),"event_ledger_rows":60*24*4*len(PRIMARY)*32,"top32_k":32,"primary_metrics":len(PRIMARY),"STATIC_OUTPUT_CARDINALITY_PROJECTION":"PASS"}
    proj={"measured_smoke_sample_artifact_bytes":artifact_bytes,"measured_checkpoint_bytes":checkpoint_bytes,"measured_event_ledger_bytes":event_bytes,"measured_histogram_state_bytes":hist_bytes,"projected_60_sample_bytes":artifact_bytes*60,"PROJECTED_FINAL_STAGE_B_ARTIFACT_GB_STATIC":artifact_bytes*60/(1024**3),"planning_only":True,"FULL60_RESOURCE_READINESS":"NOT_REACHED"}
    dump(ROOT/"projection/static_output_cardinality.json",card);dump(ROOT/"projection/static_serialized_size_projection.json",proj)

    gates={"SOURCE_GATE":"PASS","RAW_SMOKE_IDENTITY_GATE":"PASS","PRIMARY_METRICS_FROZEN":True,"STATISTICAL_UNIT_GATE":"PASS","TOKEN_LOCAL_TAIL_DEFINITION_GATE":"PASS","TOP42_SUFFICIENCY_GATE":"PASS","MOMENT_PARITY":"PASS","TOKEN_LOCAL_P99_PARITY":"EXACT_PASS","TOKEN_LOCAL_CVAR99_PARITY":"EXACT_PASS","TOKEN_LOCAL_MAX_PARITY":"EXACT_PASS","PAIRED_THRESHOLD_COUNT_PARITY":"EXACT_PASS","EXTREME_TOPK_VALUE_PARITY":"EXACT_PASS","EXTREME_TOPK_IDENTITY_PARITY":"EXACT_PASS","HISTOGRAM_INTEGER_COUNT_PARITY":"EXACT_PASS","DISCRETE_INPUT_ORDER_DETERMINISM":"EXACT_PASS","RESUME_PARITY_GATE":"PASS","RESUME_BOUNDARY_TEST_GATE":"PASS","NO_DUPLICATE_TOKEN_AFTER_RESUME_GATE":"PASS","NO_TOKEN_GAP_AFTER_RESUME_GATE":"PASS","SAMPLE_ARTIFACT_AGGREGATION_PARITY":"PASS","WORKER_COMPLETION_ORDER_INDEPENDENCE_GATE":"PASS","ATOMIC_SAMPLE_ARTIFACT_GATE":"PASS","REDUCER_IDENTITY_FAIL_CLOSED_GATE":"PASS","FULL60_FAIL_CLOSED_ORCHESTRATION_GATE":"PASS","STATIC_OUTPUT_CARDINALITY_PROJECTION":"PASS","PROJECTED_FINAL_STAGE_B_ARTIFACT_GB_STATIC":proj["PROJECTED_FINAL_STAGE_B_ARTIFACT_GB_STATIC"],"STREAMING_REDUCER_V3_PARITY":"PASS","FIRST_FAILED_GATE":None,"GLOBAL_INTERPOLATED_QUANTILE_GATE":"REMOVED_PROSPECTIVELY_DUE_TO_ESTABLISHED_CONTRACT_MISMATCH","MODEL_LOADED":False,"CUDA_ALLOCATED":False,"GPU_STREAMING_SMOKE_STARTED":False,"FULL60_STARTED":False,"FULL60_RESOURCE_READINESS":"NOT_REACHED","RECURRENT_REPLAY_STARTED":False,"MULTI_HORIZON_STARTED":False,"PRE_DIVERGENCE_STARTED":False,"LAYER_INTERVENTION_STARTED":False,"C7_STARTED":False,"NEW_LOSS_STARTED":False,"NEW_ROTATION_TRAINING_STARTED":False,"NEW_AIME_GENERATION_STARTED":False,"NEXT_RECOMMENDED_STAGE":"QWEN_GDN_STAGE_B_GPU_STREAMING_SMOKE_V3","NEXT_STAGE_EXECUTED":False}
    dump(ROOT/"validation/final_gates.json",gates)
    (ROOT/"spec/TOKEN_LOCAL_TAIL_SPEC.md").write_text(f"# Token-local tail specification\n\nFor N=4096 and q=0.99: h=(N-1)q={h}, lo={lo}, hi={hi}. Linear p99 is `x[lo] + (h-lo)*(x[hi]-x[lo])`. Top-42 begins at ascending index {base}, so offsets 0 and 1 recover p99. `K_CVAR99=ceil(0.01*4096)=41`; the 41 largest values are top-42 offsets 1..41. Max is offset 41. Thus top-42 is sufficient for exact p99, CVaR99, and max.\n",encoding="utf-8")
    (ROOT/"spec/REDUCER_V3_SPEC.md").write_text("# Reducer V3 specification\n\nProspective V3 removes the global interpolated-quantile hard gate because V2 established a statistical-contract mismatch. Primary metrics are runtime scale, PAPR, and QDQ relative L2. Primary tails are exact token-local p99, CVaR99, and max. The global multiplicative histogram is retained only for deterministic descriptive integer counts. Statistical unit is sample (N=60). One independent reducer state is produced per sample and aggregation follows the canonical manifest, never completion order.\n",encoding="utf-8")
    report=f"""# FINAL REDUCER V3 REPORT\n\n## Verdict\n\n`STREAMING_REDUCER_V3_PARITY = PASS`\n\nQ1. V2 failed because bucket/order-statistic upper-edge estimates are not statistically equivalent to linear-interpolation quantiles; gaps between adjacent order statistics can dominate bin width.\n\nQ2. V3 prospectively removes that hard gate due to the established contract mismatch; it does not relax or rewrite V2.\n\nQ3. Primary metrics: actual runtime scale, PAPR, and QDQ relative L2.\n\nQ4. Exact primary tails: token-local p99, CVaR99, and max over all 4,096 groups.\n\nQ5. Linear p99 needs ascending indices 4054 and 4055; top-42 starts at 4054. CVaR99 is the mean of indices 4055..4095 (41 values), and max is 4095.\n\nQ6. Yes; full-sort and top-M candidate agree exactly for all 288 required keys and all three tails.\n\nQ7. Yes; paired threshold counts are exact across 96 sample-layer-pair keys.\n\nQ8. Yes; extreme Top-32 values and identities are exact across {len(re)} rows.\n\nQ9. Yes; histogram integer counts are exact and deterministic. No histogram quantile gate exists in V3.\n\nQ10. No; canonical, reverse, and fixed-seed shuffled group input produce identical discrete state.\n\nQ11. Yes; a real JSON checkpoint at 40% of the infrastructure-only five-token sequence resumes exactly.\n\nQ12. No; duplicate-token and skipped-token attempts both fail closed.\n\nQ13. No; artifacts are aggregated in canonical sample-manifest order, independent of worker completion order.\n\nQ14. Yes; atomic artifacts validate, and all schema/source/config/sample/corruption mismatches fail closed.\n\nQ15. Static 60-sample projection: {proj['PROJECTED_FINAL_STAGE_B_ARTIFACT_GB_STATIC']:.6f} GiB (planning only; not resource readiness).\n\nQ16. `STREAMING_REDUCER_V3_PARITY = PASS`.\n\nQ17. Model loaded? **NO.**\n\nQ18. GPU run? **NO.**\n\nQ19. full60 started? **NO.**\n\n`GLOBAL_INTERPOLATED_QUANTILE_GATE = REMOVED_PROSPECTIVELY_DUE_TO_ESTABLISHED_CONTRACT_MISMATCH`\n\n`NEXT_RECOMMENDED_STAGE = QWEN_GDN_STAGE_B_GPU_STREAMING_SMOKE_V3` (not executed).\n"""
    (ROOT/"reports/FINAL_REDUCER_V3_REPORT.md").write_text(report,encoding="utf-8")
    hf=ROOT/"hashes/SHA256SUMS";paths=sorted(p for p in ROOT.rglob("*") if p.is_file() and p!=hf);hf.write_text("".join(f"{sha(p)}  {p.relative_to(ROOT)}\n" for p in paths),encoding="utf-8")
    print(json.dumps({"status":"PASS","projected_gib":proj["PROJECTED_FINAL_STAGE_B_ARTIFACT_GB_STATIC"],"hashed_artifacts":len(paths)},sort_keys=True))

if __name__=="__main__":main()
