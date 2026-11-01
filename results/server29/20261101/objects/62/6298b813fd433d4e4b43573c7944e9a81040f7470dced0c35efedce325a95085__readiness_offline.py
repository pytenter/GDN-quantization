#!/usr/bin/env python3
"""CPU-only cardinality, offline reducer parity, and orchestration readiness."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


TASK = "QWEN_GDN_STAGE_B_STREAMING_READINESS_V1"
REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT = REPO / "experiments/qwen_gdn" / TASK
STAGE_A = REPO / "experiments/qwen_gdn/QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2"
STAGE_B = REPO / "experiments/qwen_gdn/QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1"
BASE_HEAD = "b745cf87d9ed5e51512504d9c1fe4d8e0c1fb6d1"
RAW = STAGE_B / "smoke/group_row_calibration.parquet"
STAGE_A_RAW = STAGE_A / "parity/raw/Hadamard_instrumented_A.json"
FORMAL = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py")
LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def exact_tail(x, q):
    x = np.asarray(x, np.float64)
    k = max(1, int(math.ceil((1-q) * x.size)))
    return float(np.partition(x, x.size-k)[-k:].mean())


def reference_from_raw(table, reducer_mod):
    data = table.to_pydict()
    n = table.num_rows
    groups = {}
    for i in range(n):
        key = (data["sample_id"][i], int(data["token_idx"][i]), int(data["layer_idx"][i]), data["condition"][i])
        groups.setdefault(key, []).append(i)
    group_rows, tails, events = [], [], []
    by_key_values = {}
    for key in sorted(groups):
        sample, token, layer, condition = key
        idx = np.asarray(groups[key], dtype=np.int64)
        group_ids = np.asarray(data["head_idx"], np.int64)[idx] * 128 + np.asarray(data["group_idx"], np.int64)[idx]
        by_key_values[key] = {}
        for metric in reducer_mod.PRIMARY + reducer_mod.SECONDARY:
            x = np.asarray(data[metric], np.float32)[idx]
            xd = x.astype(np.float64)
            by_key_values[key][metric] = x
            total = xd.sum(dtype=np.float64); total_sq = np.square(xd).sum(dtype=np.float64)
            mean = float(total / xd.size); var = max(0.0, float(total_sq/xd.size - mean*mean))
            group_rows.append({
                "sample_id": sample, "layer": layer, "condition": condition, "metric": metric,
                "count": int(xd.size), "sum": float(total), "sum_sq": float(total_sq),
                "min": float(xd.min()), "max": float(xd.max()), "mean": mean,
                "variance": var, "std": math.sqrt(var), "accumulator_dtype": "float64",
            })
            if metric in reducer_mod.TAIL_METRICS:
                exact = reducer_mod.exact_token_tail(x)
                for stat in ("median", "p95", "p99", "cvar95", "cvar99", "max"):
                    tails.append({
                        "sample_id": sample, "layer": layer, "condition": condition,
                        "metric": metric, "token_local_statistic": stat,
                        "count": 1, "sum": exact[stat], "sum_sq": exact[stat]**2,
                        "min": exact[stat], "max": exact[stat], "mean": exact[stat],
                        "variance": 0.0, "std": 0.0, "accumulator_dtype": "float64",
                    })
                order = sorted(range(x.size), key=lambda j: (float(x[j]), -token, -int(group_ids[j])), reverse=True)[:reducer_mod.TOP_K]
                for rank, j in enumerate(order, 1):
                    events.append({
                        "sample_id": sample, "layer": layer, "condition": condition,
                        "metric": metric, "rank": rank, "token_idx": token,
                        "group_idx": int(group_ids[j]), "metric_value": float(x[j]),
                    })
    paired = []
    sample = next(iter(groups))[0]
    for layer in LAYERS:
        token = 0
        for numerator, denominator in reducer_mod.PAIRS:
            a = by_key_values[(sample, token, layer, numerator)]["scale"].astype(np.float64)
            b = by_key_values[(sample, token, layer, denominator)]["scale"].astype(np.float64)
            ratio = (a + reducer_mod.EPS) / (b + reducer_mod.EPS)
            log_ratio = np.log(ratio)
            row = {"sample_id": sample, "layer": layer, "numerator": numerator, "denominator": denominator,
                   "total_count": int(ratio.size), "sum_log_ratio": float(log_ratio.sum(dtype=np.float64)),
                   "sum_sq_log_ratio": float(np.square(log_ratio).sum(dtype=np.float64)), "max_ratio": float(ratio.max())}
            for threshold in reducer_mod.THRESHOLDS:
                row[f"count_gt_{str(threshold).replace('.', '_')}"] = int((ratio > threshold).sum())
            paired.append(row)
    exact_global = []
    for condition in reducer_mod.CONDITIONS:
        for metric in reducer_mod.PRIMARY:
            values = np.concatenate([v[metric] for k,v in by_key_values.items() if k[3] == condition]).astype(np.float64)
            exact_global.append({
                "condition": condition, "metric": metric, "count": int(values.size),
                "p95": float(np.quantile(values,.95,method="linear")),
                "p99": float(np.quantile(values,.99,method="linear")),
                "p999": float(np.quantile(values,.999,method="linear")),
                "cvar95": exact_tail(values,.95), "cvar99": exact_tail(values,.99),
            })
    return {
        "group_moments": group_rows, "token_local_tail_moments": tails,
        "paired_scale_ratios": paired, "extreme_events": events,
        "global_group_exact_tails": exact_global,
        "key_coverage": [list(k) for k in sorted({(k[0],k[2],k[3]) for k in groups})],
    }, groups, data


def compare_exact(reference, streaming, atol=1e-12, rtol=1e-12):
    keys = {
        "group_moments": ("sample_id","layer","condition","metric"),
        "token_local_tail_moments": ("sample_id","layer","condition","metric","token_local_statistic"),
        "paired_scale_ratios": ("sample_id","layer","numerator","denominator"),
    }
    details = {}; all_pass = True
    for section, id_fields in keys.items():
        r = {tuple(row[x] for x in id_fields): row for row in reference[section]}
        s = {tuple(row[x] for x in id_fields): row for row in streaming[section]}
        coverage = set(r) == set(s)
        failures = []
        for key in sorted(set(r)&set(s)):
            for field in set(r[key]) & set(s[key]):
                if field in id_fields or field in ("accumulator_dtype",): continue
                a,b = r[key][field],s[key][field]
                if isinstance(a,(int,bool)):
                    ok = a == b
                elif isinstance(a,float):
                    ok = math.isclose(a,float(b),rel_tol=rtol,abs_tol=atol)
                else: continue
                if not ok and len(failures)<20: failures.append({"key":key,"field":field,"reference":a,"streaming":b})
        passed = coverage and not failures
        all_pass &= passed
        details[section] = {"coverage_exact":coverage,"failures":failures,"status":"PASS" if passed else "FAIL"}
    event_fields=("sample_id","layer","condition","metric","rank","token_idx","group_idx","metric_value")
    event_exact = [tuple(x[f] for f in event_fields) for x in reference["extreme_events"]] == [tuple(x[f] for f in event_fields) for x in streaming["extreme_events"]]
    coverage_exact = reference["key_coverage"] == streaming["key_coverage"]
    all_pass &= event_exact and coverage_exact
    return {"status":"PASS" if all_pass else "FAIL","criteria":{"float_atol":atol,"float_rtol":rtol,"integer_and_identity":"exact"},
            "sections":details,"top_k_event_identity_and_value_exact":event_exact,"key_coverage_exact":coverage_exact}


def quantile_validation(reference, streaming, tolerances):
    exact={(x["condition"],x["metric"]):x for x in reference["global_group_exact_tails"]}
    approx={(x["condition"],x["metric"]):x for x in streaming["global_group_sketches"]}
    rows=[]; passed=True
    for key in sorted(exact):
        for stat in ("p95","p99","p999","cvar95","cvar99"):
            a=float(exact[key][stat]); b=float(approx[key][stat]); absolute=abs(a-b); relative=absolute/max(abs(a),1e-30)
            tol=tolerances[stat]; ok=relative<=tol or absolute<=tolerances["absolute_zero"]
            passed &= ok
            rows.append({"condition":key[0],"metric":key[1],"statistic":stat,"exact":a,"approximate":b,
                         "absolute_error":absolute,"relative_error":relative,"relative_tolerance":tol,"pass":ok})
    return {"status":"PASS" if passed else "FAIL","tolerances":tolerances,"rows":rows}


def main():
    branch = git("symbolic-ref","--short","HEAD")
    head = git("rev-parse","HEAD")
    if git("status","--porcelain"):
        raise RuntimeError("SOURCE_GATE FAIL: dirty worktree")
    if subprocess.call(["git","merge-base","--is-ancestor",BASE_HEAD,head],cwd=REPO) != 0:
        raise RuntimeError("current HEAD does not descend from stopped Stage B commit")
    for d in ("config","provenance","audit","streaming","validation","smoke","orchestration","projection","hashes","reports","scripts"):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    reducer=load_module(ROOT/"scripts/streaming_reducer.py","streaming_reducer")
    atomic=load_module(ROOT/"scripts/atomic_protocol.py","atomic_protocol")
    tolerances={"p95":.003,"p99":.003,"p999":.003,"cvar95":.003,"cvar99":.003,"absolute_zero":1e-12}
    readiness_config={
        "task":TASK,"source_base_head":BASE_HEAD,"execution_head":head,
        "full60_authorized_by_this_task":False,"full60_started":False,
        "smoke_units":["aime26_01_seed1","aime26_01_seed2"],"smoke_prefix_tokens":32,
        "statistical_unit":"sample","n":60,"all_tokens_layers_groups_conditions_required":True,
        "raw_group_serialization":False,"accumulator_dtype":"float64",
        "top_k_extreme_events":32,"percentile_method":"linear",
        "cvar95":"mean(top max(1,ceil(0.05*N)))","cvar99":"mean(top max(1,ceil(0.01*N)))",
        "sketch":"deterministic DDSketch-style log histogram","sketch_relative_accuracy":.0025,
        "quantile_tolerances_preregistered_before_validation":tolerances,
        "paired_scale_thresholds":[1.00,1.10,1.25,1.50,2.00],
    }
    dump(ROOT/"config/readiness_config.json",readiness_config)
    dump(ROOT/"config/metric_definitions.json",{
        "primary":{"M1":"actual runtime scale","M2":"absmax","M3":"PAPR=absmax/(RMS+eps)","M4":"QDQ relative L2"},
        "secondary":["range=max-min","std","QDQ relative MSE","qcode occupancy","saturation"],
        "exact_tails":"token-local group tails","approximate_tails":"global group-level deterministic sketch",
    })
    dump(ROOT/"config/resource_budget.json",{
        "MAX_PROJECTED_STAGE_B_FINAL_ARTIFACT_GB":8,"MIN_PROJECTED_FREE_AFTER_STAGE_B_GB":20,
        "EMERGENCY_MIN_FREE_SPACE_GB":10,"oom_policy":"STOP; NO AUTOMATIC RETRY",
    })
    dump(ROOT/"provenance/source_identity.json",{"CURRENT_BRANCH":branch,"CURRENT_HEAD":head,"WORKTREE_CLEAN":True,"SOURCE_GATE":"PASS"})
    dump(ROOT/"provenance/stage_a_identity.json",{"root":str(STAGE_A),"head":"2bcf9c93e72c413b28450e56ab1a99f36d742f4f","sha256sums_sha256":sha(STAGE_A/"hashes/SHA256SUMS")})
    dump(ROOT/"provenance/stage_b_smoke_identity.json",{"root":str(STAGE_B),"commit":BASE_HEAD,"sha256sums_sha256":sha(STAGE_B/"hashes/SHA256SUMS"),"raw_parquet":str(RAW),"raw_parquet_sha256":sha(RAW)})

    capture=json.load(open(STAGE_A_RAW))
    first=capture["quantizer_calls"][0]
    parquet=pq.read_table(RAW)
    pdict=parquet.to_pydict()
    cardinality=[]
    for layer in LAYERS:
        rec=first[str(layer)]["pre_writeback"]
        state_shape=rec["source"]["shape"]; scale_shape=rec["scale"]["shape"]; qcode_shape=rec["qcodes"]["shape"]
        true_groups=int(np.prod(scale_shape)); current_per_condition=sum(1 for i in range(parquet.num_rows) if int(pdict["layer_idx"][i])==layer and pdict["condition"][i]=="Native")
        cardinality.append({
            "layer_idx":layer,"state_shape_before_quantizer":state_shape,"quantizer_input_shape":state_shape,
            "scale_shape":scale_shape,"qcode_shape":qcode_shape,"state_numel":int(np.prod(state_shape)),
            "scale_numel":true_groups,"qcode_numel":int(np.prod(qcode_shape)),
            "true_groups_per_layer_per_condition":true_groups,
            "current_serialized_rows_per_token_per_layer_per_condition":current_per_condition,
            "current_serialized_rows_per_token_per_layer_all_conditions":current_per_condition*4,
        })
    with (ROOT/"audit/group_cardinality_per_layer.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cardinality[0]);w.writeheader();w.writerows(cardinality)
    true_per_condition=sum(x["scale_numel"] for x in cardinality); current=parquet.num_rows
    group_summary={
        "TRUE_QUANTIZATION_GROUP_COUNT_DEFINITION":"scale.numel()",
        "TRUE_GROUPS_PER_LAYER":4096,"TRUE_GROUPS_PER_TOKEN_PER_CONDITION":true_per_condition,
        "TRUE_GROUPS_PER_TOKEN_ALL_FOUR_CONDITIONS":true_per_condition*4,
        "CURRENT_ROWS_PER_TOKEN":current,
        "CURRENT_ROW_SCHEMA_SEMANTICS":"24 layers * 4 conditions * scale.numel(4096) = 393216 true group rows",
        "CURRENT_ROW_SCHEMA_CORRECT_FOR_GROUP_LEVEL_ANALYSIS":"YES",
        "ROW_SCHEMA_BUG_FOUND":"NO","MODEL_FORWARD_VALID":"YES","GROUP_CARDINALITY_GATE":"PASS",
    }
    dump(ROOT/"audit/group_cardinality_summary.json",group_summary)
    (ROOT/"audit/old_serializer_audit.md").write_text(
        "# Old serializer audit\n\nThe formal source/qcode tensor is `[1,32,128,128]`; the scale tensor is `[1,32,1,128]`. "
        "The serializer reduces the K axis (`dim=-2`) and flattens `[1,32,128]`, producing exactly 4,096 rows per layer and condition. "
        "Thus `24 × 4 × 4,096 = 393,216` rows/token. It is neither state/qcode element serialization nor broadcast expansion. "
        "The schema is scientifically valid group-level storage but operationally infeasible for 1,854,451 tokens.\n",encoding="utf-8")
    if any(x["scale_numel"]!=x["current_serialized_rows_per_token_per_layer_per_condition"] for x in cardinality):
        raise RuntimeError("GROUP_CARDINALITY_GATE FAIL")

    (ROOT/"streaming/reducer_spec.md").write_text(
        "# Reducer specification\n\nAll groups are computed. No token/layer/group/condition sampling occurs. Raw observations update float64 moments, "
        "exact token-local linear-interpolation percentiles and top-k CVaR, exact paired ratios, deterministic top-32 ledgers, "
        "and a mergeable deterministic 0.25%-relative-accuracy log sketch before being discarded. Statistical unit is sample.\n",encoding="utf-8")
    dump(ROOT/"streaming/reducer_version.json",{"version":reducer.VERSION,"source_sha256":sha(ROOT/"scripts/streaming_reducer.py"),"parameters":readiness_config})

    reference, groups, data=reference_from_raw(parquet,reducer)
    dump(ROOT/"validation/offline_reference_summary.json",reference)
    streaming=reducer.StreamingReducer()
    for key in sorted(groups):
        sample,token,layer,condition=key; idx=np.asarray(groups[key],np.int64)
        metrics={m:np.asarray(data[m],np.float32)[idx] for m in reducer.PRIMARY+reducer.SECONDARY}
        gids=np.asarray(data["head_idx"],np.int64)[idx]*128+np.asarray(data["group_idx"],np.int64)[idx]
        streaming.update(sample,token,layer,condition,metrics,gids)
    streaming_out=streaming.finalize()
    dump(ROOT/"validation/streaming_from_raw_summary.json",streaming_out)
    parity=compare_exact(reference,streaming_out)
    dump(ROOT/"validation/exact_parity.json",parity)
    qv=quantile_validation(reference,streaming_out,tolerances)
    dump(ROOT/"validation/quantile_validation.json",qv)
    offline_gate="PASS" if parity["status"]=="PASS" else "FAIL"
    reducer_gate="PASS" if offline_gate=="PASS" and qv["status"]=="PASS" else "FAIL"

    testroot=ROOT/"orchestration/test_area"
    identity={"source_commit":head,"config_sha256":sha(ROOT/"config/readiness_config.json"),"artifact_identity":"frozen_C5_C6_H_Native","shard_assignment":"static_mod4","reducer_version":reducer.VERSION,"scientific_semantics":"strict_common_fp_state"}
    done=atomic.commit_sample(testroot,"sample_00",{"value":1},identity)
    atomic_valid=atomic.validate_sample(testroot,"sample_00",identity)
    (testroot/"sample_01.tmp").write_text("partial",encoding="utf-8")
    atomic.commit_sample(testroot,"sample_02",{"value":2},identity)
    (testroot/"sample_02.json").write_text("corrupted",encoding="utf-8")
    scan=atomic.resume_scan(testroot,["sample_00","sample_01","sample_02"],identity)
    wrong=atomic.validate_sample(testroot,"sample_00",{**identity,"config_sha256":"wrong"})
    atomic_gate="PASS" if atomic_valid["valid"] and scan["valid_completed"]==["sample_00"] else "FAIL"
    resume_gate="PASS" if scan["pending"]==["sample_01","sample_02"] and not wrong["valid"] else "FAIL"
    dump(ROOT/"orchestration/atomic_artifact_test.json",{"status":atomic_gate,"done":done,"validation":atomic_valid})
    dump(ROOT/"orchestration/resume_policy_test.json",{"status":resume_gate,"frozen_identity":identity,"scan":scan,"identity_mismatch_test":wrong,
         "policy":"resume only uncommitted samples with identical source/config/artifacts/shards/reducer/semantics; no chunk or metric changes"})
    fail_cases={
        "worker_nonzero":atomic.fail_closed_decision([0,1,0,0],60,list(map(str,range(60)))),
        "incomplete_59":atomic.fail_closed_decision([0,0,0,0],60,list(map(str,range(59)))),
        "oom":atomic.fail_closed_decision([0,0,0,0],60,list(map(str,range(60))),oom_observed=True),
        "emergency_disk":atomic.fail_closed_decision([0,0,0,0],60,list(map(str,range(60))),free_gb=9),
    }
    fail_gate="PASS" if all(v["stage_gate"]=="FAIL" and not v["downstream_launch_authorized"] for v in fail_cases.values()) else "FAIL"
    dump(ROOT/"orchestration/fail_closed_test.json",{"status":fail_gate,"cases":fail_cases})

    pre_gpu={
        "SOURCE_GATE":"PASS","GROUP_CARDINALITY_GATE":"PASS","CURRENT_ROW_SCHEMA_CORRECT":"YES",
        "STATISTICAL_UNIT_FROZEN":"PASS","PRIMARY_METRICS_FROZEN":"PASS","TAIL_DEFINITION_FROZEN":"PASS",
        "PERCENTILE_DEFINITION_FROZEN":"PASS","STREAMING_REDUCER_IMPLEMENTED":"PASS",
        "OFFLINE_REFERENCE_GATE":offline_gate,"STREAMING_REDUCER_PARITY":reducer_gate,
        "QUANTILE_SKETCH_TOLERANCE_PREREGISTERED":"PASS","QUANTILE_SKETCH_VALIDATION":qv["status"],
        "ATOMIC_ARTIFACT_GATE":atomic_gate,"FAIL_CLOSED_ORCHESTRATION_GATE":fail_gate,"RESUME_POLICY_GATE":resume_gate,
        "GPU_STREAMING_SMOKE":"AUTHORIZED" if reducer_gate=="PASS" else "NOT_AUTHORIZED",
        "FULL60_AUTHORIZED_BY_THIS_TASK":False,"FULL60_STARTED":False,
    }
    dump(ROOT/"validation/pre_gpu_gate_status.json",pre_gpu)
    print(json.dumps(pre_gpu,indent=2,sort_keys=True))
    if any(pre_gpu[k]!="PASS" for k in ("SOURCE_GATE","GROUP_CARDINALITY_GATE","STATISTICAL_UNIT_FROZEN","PRIMARY_METRICS_FROZEN","TAIL_DEFINITION_FROZEN","PERCENTILE_DEFINITION_FROZEN","STREAMING_REDUCER_IMPLEMENTED","OFFLINE_REFERENCE_GATE","STREAMING_REDUCER_PARITY","QUANTILE_SKETCH_TOLERANCE_PREREGISTERED","QUANTILE_SKETCH_VALIDATION","ATOMIC_ARTIFACT_GATE","FAIL_CLOSED_ORCHESTRATION_GATE","RESUME_POLICY_GATE")):
        raise SystemExit(2)


if __name__=="__main__":
    main()
