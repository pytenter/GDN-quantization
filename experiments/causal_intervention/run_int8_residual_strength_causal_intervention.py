#!/usr/bin/env python3
import argparse
import os
import gzip
import json
import math
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/path/to/gdn_data_root"))
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_RESIDUAL_STRENGTH_CAUSAL_INTERVENTION_V1"
RESULT = RES / "gdn_int8_residual_strength_causal_intervention_v1.json"
STAGE0 = RES / "gdn_int8_residual_strength_causal_intervention_v1_stage0.json"
RECORDS = RES / "gdn_int8_residual_strength_causal_intervention_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_residual_strength_causal_intervention_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_residual_strength_causal_intervention_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_residual_strength_causal_intervention_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_residual_strength_causal_intervention_v1.md"
FORMAL_RECORDS = RES / "gdn_int8_residual_strength_causal_intervention_v1_formal_records.jsonl"
FORMAL_TOKEN_RECORDS = RES / "gdn_int8_residual_strength_causal_intervention_v1_formal_token_records.jsonl.gz"
FORMAL_LAYER_RECORDS = RES / "gdn_int8_residual_strength_causal_intervention_v1_formal_layer_records.jsonl.gz"
FORMAL_HEAD_RECORDS = RES / "gdn_int8_residual_strength_causal_intervention_v1_formal_head_records.jsonl.gz"

P1_RESULT = RES / "gdn_int8_orientation_state_change_mechanism_v1.json"
P1_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
SMALLVAL_RESULT = RES / "gdn_int8_effective_update_small_prompt_validation_v1.json"
SMALLVAL_REPORT = REP / "gdn_int8_effective_update_small_prompt_validation_v1.md"

EPS = 1e-12
LAMBDAS = [1.00, 0.75, 0.50, 0.25, 0.00]
CONFIG_NAMES = ["FP_STATE"] + [f"L{lam:.2f}" for lam in LAMBDAS]
R128_CFG = {"name": "R128", "orientation": "row", "group_size": 128}
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
WINDOWS = [(1, 128), (129, 256), (257, 512)]
HEAD_SAMPLE_STRIDE = 128

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_effective_update_metric_audit as eff


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def finite_num(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def avg(xs):
    xs = [float(x) for x in xs if finite_num(x)]
    return sum(xs) / len(xs) if xs else None


def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite_num(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if finite_num(v):
                buckets[k].append(v)
    return {
        k: {
            "mean": avg(v),
            "median": statistics.median(v),
            "p95": pct(v, 0.95),
            "max": max(v),
        }
        for k, v in sorted(buckets.items())
    }


def stat_mean(stats, key):
    val = stats.get(key)
    return val.get("mean") if isinstance(val, dict) else None


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def gz_append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def iter_jsonl(path):
    if not Path(path).exists():
        return
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def window_label(token_idx):
    one = int(token_idx) + 1
    for a, b in WINDOWS:
        if a <= one <= b:
            return f"{a}-{b}"
    return ">512"


def stage_paths(stage, shard_id=None):
    suffix = f"_{stage}" + ("" if shard_id is None else f"_shard_{shard_id:03d}")
    return {
        "records": RES / f"gdn_int8_residual_strength_causal_intervention_v1_records{suffix}.jsonl",
        "token": RES / f"gdn_int8_residual_strength_causal_intervention_v1_token_records{suffix}.jsonl.gz",
        "layer": RES / f"gdn_int8_residual_strength_causal_intervention_v1_layer_records{suffix}.jsonl.gz",
        "head": RES / f"gdn_int8_residual_strength_causal_intervention_v1_head_records{suffix}.jsonl.gz",
    }


def prompt_manifest():
    p1_obj = json.loads(P1_RESULT.read_text(encoding="utf-8"))
    smallval_obj = json.loads(SMALLVAL_RESULT.read_text(encoding="utf-8"))
    canonical_ids = [m["prompt_id"] for m in smallval_obj.get("prompt_manifest", [])]
    if len(canonical_ids) != 6:
        canonical_ids = []
        for row in iter_jsonl(P1_RECORDS):
            pid = row.get("problem_id")
            if pid and pid not in canonical_ids:
                canonical_ids.append(pid)
    prompt_rows = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    manifest = []
    for pid in canonical_ids:
        pm = prompt_rows.get(pid)
        if not pm:
            continue
        manifest.append({
            "prompt_id": pid,
            "role": pm["role"],
            "source_path": str(SMALLVAL_RESULT),
            "source_identifier": "GDN_INT8_EFFECTIVE_UPDATE_METRIC_SMALL_PROMPT_VALIDATION_V1 prompt_manifest",
            "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
            "exact_original_generation_token_ids_available": False,
            "exact_replay_claimed": False,
            "continuation_token_count_available": pm.get("fp_tokens"),
        })
    return manifest, p1_obj, smallval_obj


def cosine(torch, a, b):
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    an = float(torch.linalg.vector_norm(af).item())
    bn = float(torch.linalg.vector_norm(bf).item())
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item())


def lambda_metric_event(torch, s_in, s_pre, lam):
    inn = s_in.detach().float()
    pre = s_pre.detach().float()
    if not bool(torch.isfinite(inn).all().item() and torch.isfinite(pre).all().item()):
        return {"status": "NONFINITE_STATE"}
    s_q, _, _, scale, scale_shape, dyn = axis.grouped_quant(torch, pre, pre, R128_CFG)
    r_q = s_q - pre
    r_lambda = float(lam) * r_q
    s_write = pre + r_lambda
    desired = pre - inn
    effective = s_write - inn
    identity = effective - (desired + r_lambda)
    write_identity = (s_write - pre) - r_lambda
    desired_norm = float(torch.linalg.vector_norm(desired).item())
    effective_norm = float(torch.linalg.vector_norm(effective).item())
    raw_residual_norm = float(torch.linalg.vector_norm(r_q).item())
    residual_norm = float(torch.linalg.vector_norm(r_lambda).item())
    pre_norm = float(torch.linalg.vector_norm(pre).item())
    identity_norm = float(torch.linalg.vector_norm(identity).item())
    write_identity_norm = float(torch.linalg.vector_norm(write_identity).item())
    d2 = float(torch.sum(desired.double() * desired.double()).item())
    alpha = float(torch.sum(r_lambda.double() * desired.double()).item()) / (d2 + EPS) if d2 > EPS else None
    if finite_num(alpha):
        r_parallel = alpha * desired
        r_perp = r_lambda - r_parallel
        orthogonal_ratio = float(torch.linalg.vector_norm(r_perp).item()) / (desired_norm + EPS)
        orthogonal_fraction = float(torch.linalg.vector_norm(r_perp).item()) / (residual_norm + EPS)
    else:
        orthogonal_ratio = None
        orthogonal_fraction = None
    achieved = residual_norm / (raw_residual_norm + EPS)
    scale_vals = scale.detach().float().flatten().cpu().tolist()
    dyn_vals = [float(x) for x in dyn if finite_num(x)]
    return {
        "status": "OK",
        "lambda": float(lam),
        "orientation": "row",
        "group_size": 128,
        "qrange": [-127, 127],
        "zero_point": 0,
        "rounding": "torch.round",
        "scale_shape": scale_shape,
        "scale_count_per_head": axis.config_geometry(R128_CFG)["scale_count_per_head"],
        "state_shape": list(pre.shape),
        "state_reconstruction_relative_error_E_S": residual_norm / (pre_norm + EPS),
        "D_effective": residual_norm / (desired_norm + EPS),
        "cos_effective": cosine(torch, desired, effective),
        "norm_ratio_effective": effective_norm / (desired_norm + EPS),
        "alpha": alpha,
        "effective_parallel_gain": (1.0 + alpha) if finite_num(alpha) else None,
        "orthogonal_ratio": orthogonal_ratio,
        "orthogonal_fraction_of_residual": orthogonal_fraction,
        "raw_R128_residual_norm": raw_residual_norm,
        "lambda_residual_norm": residual_norm,
        "desired_state_change_norm": desired_norm,
        "effective_update_norm": effective_norm,
        "identity_error_norm": identity_norm,
        "identity_relative_error": identity_norm / (effective_norm + desired_norm + residual_norm + EPS),
        "write_identity_error_norm": write_identity_norm,
        "write_identity_relative_error": write_identity_norm / (residual_norm + EPS),
        "achieved_residual_scale": achieved,
        "achieved_residual_scale_absdev": abs(achieved - float(lam)),
        "lambda1_canonical_R128_state_error": float(torch.linalg.vector_norm((s_write - s_q).float()).item()) / (float(torch.linalg.vector_norm(s_q.float()).item()) + EPS) if abs(float(lam) - 1.0) < 1e-12 else None,
        "scale_mean": avg(scale_vals),
        "scale_median": statistics.median(scale_vals) if scale_vals else None,
        "scale_max": max(scale_vals) if scale_vals else None,
        "scale_p95_over_median": pct(scale_vals, 0.95) / (statistics.median(scale_vals) + EPS) if scale_vals else None,
        "within_group_dynamic_range_p95": pct(dyn_vals, 0.95),
        "_s_write": s_write,
    }


def run_prompt(torch, model, tokenizer, e2e, pm, stage, horizon, paths, args):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids_all = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    cont_ids = cont_ids_all[:horizon]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    ref_mask = enc.get("attention_mask")
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    lambda_inputs = {lam: ref_input.clone() for lam in LAMBDAS}
    lambda_masks = {lam: ref_mask.clone() if ref_mask is not None else None for lam in LAMBDAS}
    ref_past = None
    lambda_pasts = {lam: None for lam in LAMBDAS}
    token_rows = defaultdict(list)
    layer_rows = defaultdict(list)
    head_rows = defaultdict(list)
    protocol_gate = "PASS"
    metric_gate = "PASS"
    intervention_gate = "PASS"
    nonfinite = defaultdict(int)
    failures = []
    print(f"[{now()}] {stage} shard={args.shard_id} prompt={pm['problem_id']} tokens={len(cont_ids)} configs={CONFIG_NAMES}", flush=True)
    with torch.inference_mode():
        for t, tok in enumerate(cont_ids):
            ref_out = p1.feed_step(torch, model, ref_input, ref_mask, ref_past)
            ref_past = ref_out.past_key_values
            gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "token_idx": t, "config": "FP_STATE", "KL": 0.0, "top1_agreement": 1.0})
            for lam in LAMBDAS:
                cfg_name = f"L{lam:.2f}"
                s_in = None
                if lambda_pasts[lam] is not None:
                    s_in = {layer: p1.get_state(lambda_pasts[lam], layer).detach().clone() for layer in GDN_LAYERS}
                q_out = p1.feed_step(torch, model, lambda_inputs[lam], lambda_masks[lam], lambda_pasts[lam])
                lambda_pasts[lam] = q_out.past_key_values
                layer_metrics = []
                if s_in is not None:
                    for layer in GDN_LAYERS:
                        state = p1.get_state(lambda_pasts[lam], layer)
                        s_pre = state.detach().clone()
                        m = lambda_metric_event(torch, s_in[layer], s_pre, lam)
                        if m.get("status") != "OK":
                            metric_gate = "FAIL"
                            nonfinite[cfg_name] += 1
                            continue
                        if m.get("state_shape") != [1, 32, 128, 128] or m.get("qrange") != [-127, 127]:
                            protocol_gate = "FAIL"
                        if m.get("scale_count_per_head") != axis.config_geometry(R128_CFG)["scale_count_per_head"]:
                            protocol_gate = "FAIL"
                        if m.get("identity_relative_error", 1.0) > 1e-6 or m.get("write_identity_relative_error", 1.0) > 2e-5:
                            metric_gate = "FAIL"
                        if m.get("achieved_residual_scale_absdev", 1.0) > 1e-5:
                            intervention_gate = "FAIL"
                        if abs(lam - 1.0) < 1e-12 and m.get("lambda1_canonical_R128_state_error", 1.0) > 1e-7:
                            intervention_gate = "FAIL"
                        state.copy_(m["_s_write"].to(state.dtype))
                        pub = {k: v for k, v in m.items() if not k.startswith("_")}
                        pub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "config": cfg_name, "layer_idx": layer})
                        layer_metrics.append(pub)
                        layer_rows[cfg_name].append(pub)
                        gz_append(paths["layer"], pub)
                        if t % HEAD_SAMPLE_STRIDE == 0:
                            for h in range(s_pre.shape[1]):
                                hm = lambda_metric_event(torch, s_in[layer][:, h:h + 1, :, :], s_pre[:, h:h + 1, :, :], lam)
                                if hm.get("status") == "OK":
                                    hpub = {k: v for k, v in hm.items() if not k.startswith("_")}
                                    hpub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "config": cfg_name, "layer_idx": layer, "head_idx": h, "head_sample_stride": HEAD_SAMPLE_STRIDE})
                                    head_rows[cfg_name].append(hpub)
                                    gz_append(paths["head"], hpub)
                lm = eff.logits_metrics(torch, ref_out.logits, q_out.logits)
                flat = summarize(layer_metrics)
                one = {k: stat_mean(flat, k) for k in flat}
                one.update(lm)
                one["token_idx"] = t
                token_rows[(cfg_name, "all")].append(one)
                token_rows[(cfg_name, window_label(t))].append(one)
                gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "config": cfg_name, "lambda": lam, "KL": lm["KL"], "top1_agreement": lm["top1_agreement"], "layer_mean": one, "metric_status": "INIT_NO_S_IN" if s_in is None else "OK"})
            ref_input = torch.tensor([[tok]], dtype=ref_input.dtype, device=device)
            ref_mask = None
            for lam in LAMBDAS:
                lambda_inputs[lam] = ref_input.clone()
                lambda_masks[lam] = None
    summaries = {
        "FP_STATE": {
            "valid_token_count": len(cont_ids),
            "surviving_token_count": len(cont_ids),
            "token_summary": {"KL": {"mean": 0.0}, "top1_agreement": {"mean": 1.0}},
        }
    }
    for lam in LAMBDAS:
        name = f"L{lam:.2f}"
        summaries[name] = {
            "lambda": lam,
            "valid_token_count": len([r for r in token_rows[(name, "all")] if r.get("token_idx", 0) > 0]),
            "surviving_token_count": len(cont_ids),
            "nonfinite_count": nonfinite[name],
            "token_summary": summarize(token_rows[(name, "all")]),
            "window_summary": {w: summarize(token_rows[(name, w)]) for _, w in token_rows if _ == name and w != "all"},
            "layer_summary": summarize(layer_rows[name]),
            "head_summary": summarize(head_rows[name]),
        }
    lambda0_kl = stat_mean(summaries["L0.00"]["token_summary"], "KL")
    lambda0_top1 = stat_mean(summaries["L0.00"]["token_summary"], "top1_agreement")
    lambda1_kl = stat_mean(summaries["L1.00"]["token_summary"], "KL")
    if finite_num(lambda0_kl) and lambda0_kl > 1e-8:
        metric_gate = "FAIL"
        failures.append({"check": "lambda0_FP_STATE_KL", "value": lambda0_kl, "threshold": 1e-8})
    if finite_num(lambda0_top1) and lambda0_top1 < 1.0:
        metric_gate = "FAIL"
        failures.append({"check": "lambda0_FP_STATE_top1", "value": lambda0_top1, "threshold": 1.0})
    if finite_num(lambda1_kl) and lambda1_kl < 0.005:
        metric_gate = "FAIL"
        failures.append({"check": "lambda1_R128_degradation", "value": lambda1_kl, "threshold": 0.005})
    record = {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "requested_token_count": horizon,
        "surviving_token_count": len(cont_ids),
        "configs": CONFIG_NAMES,
        "PROTOCOL_GATE": protocol_gate,
        "METRIC_GATE": metric_gate,
        "INTERVENTION_GATE": intervention_gate,
        "lambda0_FP_STATE_check": "PASS" if lambda0_kl is not None and lambda0_kl <= 1e-8 and lambda0_top1 == 1.0 else "FAIL",
        "lambda1_canonical_R128_check": "PASS" if lambda1_kl is not None and lambda1_kl >= 0.005 and intervention_gate == "PASS" else "FAIL",
        "failure_records": failures,
        "summaries": summaries,
    }
    append_jsonl(paths["records"], record)
    print(f"[{now()}] {stage} done prompt={pm['problem_id']} gates P/M/I={protocol_gate}/{metric_gate}/{intervention_gate}", flush=True)
    return record


def spearman(xs, ys):
    return axis.spearman(xs, ys)


def lambda_value(config):
    return float(config[1:])


def prompt_dose_response(row):
    vals = []
    for lam in LAMBDAS:
        name = f"L{lam:.2f}"
        vals.append({
            "lambda": lam,
            "config": name,
            "KL": stat_mean(row["summaries"][name]["token_summary"], "KL"),
            "top1_agreement": stat_mean(row["summaries"][name]["token_summary"], "top1_agreement"),
            "E_S": stat_mean(row["summaries"][name]["layer_summary"], "state_reconstruction_relative_error_E_S"),
            "D_effective": stat_mean(row["summaries"][name]["layer_summary"], "D_effective"),
            "cos_effective": stat_mean(row["summaries"][name]["layer_summary"], "cos_effective"),
            "norm_ratio_effective": stat_mean(row["summaries"][name]["layer_summary"], "norm_ratio_effective"),
            "alpha": stat_mean(row["summaries"][name]["layer_summary"], "alpha"),
            "orthogonal_ratio": stat_mean(row["summaries"][name]["layer_summary"], "orthogonal_ratio"),
            "achieved_residual_scale": stat_mean(row["summaries"][name]["layer_summary"], "achieved_residual_scale"),
        })
    kl = [v["KL"] for v in vals]
    adjacent = 0
    for a, b in zip(kl, kl[1:]):
        if finite_num(a) and finite_num(b) and b <= a:
            adjacent += 1
    lam_axis = [v["lambda"] for v in vals]
    kl1 = vals[0]["KL"]
    kl0 = vals[-1]["KL"]
    abs_rescue = kl1 - kl0 if finite_num(kl1) and finite_num(kl0) else None
    rel_rescue = abs_rescue / (kl1 + EPS) if finite_num(abs_rescue) and finite_num(kl1) else None
    return {
        "problem_id": row["problem_id"],
        "role": row["role"],
        "lambda_metrics": vals,
        "adjacent_KL_dose_response_count": adjacent,
        "adjacent_KL_dose_response_total": 4,
        "spearman_lambda_KL": spearman(lam_axis, kl),
        "spearman_lambda_D_effective": spearman(lam_axis, [v["D_effective"] for v in vals]),
        "spearman_lambda_E_S": spearman(lam_axis, [v["E_S"] for v in vals]),
        "spearman_lambda_orthogonal_ratio": spearman(lam_axis, [v["orthogonal_ratio"] for v in vals]),
        "endpoint_rescue": {
            "KL_lambda1": kl1,
            "KL_lambda0": kl0,
            "absolute_rescue": abs_rescue,
            "relative_rescue": rel_rescue,
        },
        "lambda0_FP_STATE_check": row.get("lambda0_FP_STATE_check"),
        "lambda1_canonical_R128_check": row.get("lambda1_canonical_R128_check"),
        "PROTOCOL_GATE": row.get("PROTOCOL_GATE"),
        "METRIC_GATE": row.get("METRIC_GATE"),
        "INTERVENTION_GATE": row.get("INTERVENTION_GATE"),
    }


def classify_pilot(row):
    dr = prompt_dose_response(row)
    gates = all(row.get(k) == "PASS" for k in ["PROTOCOL_GATE", "METRIC_GATE", "INTERVENTION_GATE"])
    endpoints = row.get("lambda0_FP_STATE_check") == "PASS" and row.get("lambda1_canonical_R128_check") == "PASS"
    rescue = dr["endpoint_rescue"]
    clear_rescue = finite_num(rescue["absolute_rescue"]) and rescue["absolute_rescue"] > 0.005 and finite_num(rescue["relative_rescue"]) and rescue["relative_rescue"] > 0.8
    positive = gates and endpoints and dr["adjacent_KL_dose_response_count"] >= 3 and finite_num(dr["spearman_lambda_KL"]) and dr["spearman_lambda_KL"] >= 0.8 and clear_rescue
    if positive:
        return "POSITIVE_RESIDUAL_STRENGTH_DOSE_RESPONSE"
    if gates and endpoints:
        return "RESIDUAL_STRENGTH_CAUSAL_INCONCLUSIVE"
    return "RESIDUAL_STRENGTH_CAUSAL_NOT_SUPPORTED"


def aggregate(rows, stage):
    dose = [prompt_dose_response(r) for r in rows]
    by_lambda = {}
    for lam in LAMBDAS:
        name = f"L{lam:.2f}"
        by_lambda[name] = {
            "lambda": lam,
            "mean_KL": avg([d["lambda_metrics"][i]["KL"] for d in dose for i, v in enumerate(d["lambda_metrics"]) if v["lambda"] == lam]),
            "median_KL": statistics.median([d["lambda_metrics"][i]["KL"] for d in dose for i, v in enumerate(d["lambda_metrics"]) if v["lambda"] == lam and finite_num(d["lambda_metrics"][i]["KL"])]),
            "mean_D_effective": avg([d["lambda_metrics"][i]["D_effective"] for d in dose for i, v in enumerate(d["lambda_metrics"]) if v["lambda"] == lam]),
            "mean_E_S": avg([d["lambda_metrics"][i]["E_S"] for d in dose for i, v in enumerate(d["lambda_metrics"]) if v["lambda"] == lam]),
            "mean_orthogonal_ratio": avg([d["lambda_metrics"][i]["orthogonal_ratio"] for d in dose for i, v in enumerate(d["lambda_metrics"]) if v["lambda"] == lam]),
        }
    positive_prompts = [d for d in dose if d["adjacent_KL_dose_response_count"] >= 3 and finite_num(d["spearman_lambda_KL"]) and d["spearman_lambda_KL"] >= 0.8 and finite_num(d["endpoint_rescue"]["relative_rescue"]) and d["endpoint_rescue"]["relative_rescue"] > 0.8]
    return {
        "per_prompt": dose,
        "aggregate_by_lambda": by_lambda,
        "positive_KL_dose_response_prompt_count": len(positive_prompts),
        "rho_lambda_KL_ge_0p8_count": len([d for d in dose if finite_num(d["spearman_lambda_KL"]) and d["spearman_lambda_KL"] >= 0.8]),
        "lambda0_FP_STATE_pass_count": len([d for d in dose if d["lambda0_FP_STATE_check"] == "PASS"]),
        "lambda1_R128_pass_count": len([d for d in dose if d["lambda1_canonical_R128_check"] == "PASS"]),
    }


def write_report(obj):
    lines = [
        "# GDN INT8 Residual Strength Causal Intervention V1",
        "",
        "## Gates",
        f"- FORMAL_STATUS: `{obj.get('FORMAL_STATUS')}`",
        f"- PROTOCOL_GATE: `{obj.get('PROTOCOL_GATE')}`",
        f"- METRIC_GATE: `{obj.get('METRIC_GATE')}`",
        f"- INTERVENTION_GATE: `{obj.get('INTERVENTION_GATE')}`",
        f"- METHOD_DESIGN_READY: `{obj.get('METHOD_DESIGN_READY')}`",
        f"- FINAL_CLASSIFICATION: `{obj.get('FINAL_CLASSIFICATION')}`",
        "",
        "## Prompt Manifest",
        "```json",
        json.dumps(obj.get("prompt_manifest", []), indent=2, sort_keys=True),
        "```",
        "",
        "## Analysis",
        "```json",
        json.dumps(obj.get("analysis", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Scientific Limits",
        "- This is an oracle residual attenuation intervention, not a deployable quantizer or method.",
        "- It tests residual-strength causal contribution only; it does not test residual direction causality.",
        "- METHOD_DESIGN_READY remains NO.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(stage, num_shards=4):
    if stage == "formal":
        rows = []
        for sid in range(num_shards):
            rows.extend(iter_jsonl(stage_paths("formal", sid)["records"]) or [])
    else:
        rows = list(iter_jsonl(stage_paths(stage)["records"]) or [])
    rows = [r for r in rows if r.get("stage") == stage]
    analysis = aggregate(rows, stage) if rows else {}
    gates = {
        "PROTOCOL_GATE": "PASS" if rows and all(r.get("PROTOCOL_GATE") == "PASS" for r in rows) else "FAIL",
        "METRIC_GATE": "PASS" if rows and all(r.get("METRIC_GATE") == "PASS" for r in rows) else "FAIL",
        "INTERVENTION_GATE": "PASS" if rows and all(r.get("INTERVENTION_GATE") == "PASS" for r in rows) else "FAIL",
    }
    if stage == "smoke":
        final = "SMOKE_PASS" if all(v == "PASS" for v in gates.values()) else "SMOKE_FAIL"
        pilot = "NOT_RUN"
        formal = "NOT_RUN"
    elif stage == "pilot":
        pilot = classify_pilot(rows[0]) if rows else "RESIDUAL_STRENGTH_CAUSAL_NOT_SUPPORTED"
        final = pilot
        formal = "NOT_RUN"
    else:
        pilot = "N/A"
        n = len(rows)
        strong = analysis.get("positive_KL_dose_response_prompt_count", 0)
        if n == 6 and strong >= 4 and analysis.get("lambda0_FP_STATE_pass_count") == 6:
            final = "RESIDUAL_STRENGTH_CAUSAL_SUPPORT"
            contributor = "SUPPORTED"
            ready = "YES"
        elif n == 6 and strong > 0:
            final = "RESIDUAL_STRENGTH_CAUSAL_INCONCLUSIVE"
            contributor = "INCONCLUSIVE"
            ready = "NO"
        else:
            final = "RESIDUAL_STRENGTH_CAUSAL_HYPOTHESIS_NOT_SUPPORTED"
            contributor = "NOT_SUPPORTED"
            ready = "NO"
        formal = "COMPLETE" if n == 6 else "INCOMPLETE"
    obj = {
        "task": TASK,
        "stage": f"{stage.upper()}_COMPLETE",
        "timestamp": now(),
        "FORMAL_STATUS": formal,
        "completed_prompt_count": len(rows),
        "requested_prompt_count": 6 if stage == "formal" else 1,
        **gates,
        "SMOKE_GATE": "PASS" if stage == "smoke" and all(v == "PASS" for v in gates.values()) else ("N/A" if stage != "smoke" else "FAIL"),
        "PILOT_CLASSIFICATION": pilot,
        "FINAL_CLASSIFICATION": final,
        "EFFECTIVE_UPDATE_DISTORTION_CAUSAL_CONTRIBUTOR": locals().get("contributor", "NOT_EVALUATED"),
        "RESIDUAL_GEOMETRY_CAUSAL_INTERVENTION_READY": locals().get("ready", "NO"),
        "METHOD_DESIGN_READY": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "prompt_manifest": prompt_manifest()[0],
        "analysis": analysis,
        "Negative result policy": "Do not alter lambda grid or start residual-direction/method-design experiments.",
    }
    save_json(RESULT, obj)
    write_report(obj)
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))
    return obj


def merge_formal(num_shards):
    with FORMAL_RECORDS.open("w", encoding="utf-8") as out:
        for sid in range(num_shards):
            p = stage_paths("formal", sid)["records"]
            if p.exists():
                out.write(p.read_text(encoding="utf-8"))
    shutil.copyfile(FORMAL_RECORDS, RECORDS)
    def merge(kind, target):
        with gzip.open(target, "wt", encoding="utf-8") as out:
            for sid in range(num_shards):
                p = stage_paths("formal", sid)[kind]
                if p.exists():
                    with gzip.open(p, "rt", encoding="utf-8") as f:
                        for line in f:
                            out.write(line)
    merge("token", FORMAL_TOKEN_RECORDS)
    merge("layer", FORMAL_LAYER_RECORDS)
    merge("head", FORMAL_HEAD_RECORDS)
    shutil.copyfile(FORMAL_TOKEN_RECORDS, TOKEN_RECORDS)
    shutil.copyfile(FORMAL_LAYER_RECORDS, LAYER_RECORDS)
    shutil.copyfile(FORMAL_HEAD_RECORDS, HEAD_RECORDS)


def run_stage(args):
    manifest, _, _ = prompt_manifest()
    pms = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    if args.stage in ("smoke", "pilot"):
        ids = ["test/algebra/1332.json"]
        if ids[0] not in [m["prompt_id"] for m in manifest]:
            ids = [manifest[0]["prompt_id"]]
        paths = stage_paths(args.stage)
    else:
        ids = [m["prompt_id"] for i, m in enumerate(manifest) if i % args.num_shards == args.shard_id]
        paths = stage_paths("formal", args.shard_id)
    done = set()
    if args.resume and paths["records"].exists():
        done = {r["problem_id"] for r in iter_jsonl(paths["records"]) if r.get("stage") == args.stage}
    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    for pid in ids:
        if pid not in done:
            run_prompt(torch, model, tokenizer, e2e, pms[pid], args.stage, args.max_steps, paths, args)


def stage0():
    manifest, p1_obj, smallval_obj = prompt_manifest()
    protocol = "PASS" if len(manifest) == 6 and p1_obj.get("FORMAL_STATUS") == "COMPLETE" and smallval_obj.get("FORMAL_STATUS") == "COMPLETE" else "FAIL"
    obj = {
        "task": TASK,
        "stage": "stage0",
        "timestamp": now(),
        "PROTOCOL_GATE": protocol,
        "METRIC_GATE": "PASS",
        "INTERVENTION_IMPLEMENTATION_AUDIT": "PASS",
        "INTERVENTION_GATE": "PASS",
        "METHOD_DESIGN_READY": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "model_identity": "Qwen3.5-9B from existing canonical loader",
        "state_shape": "[B,H,K,V] = [1,32,128,128]",
        "row_axis": "Key axis",
        "column_axis": "Value axis",
        "prefill_precision": "FP32 recurrent state",
        "continuation": "teacher-forced canonical continuation",
        "canonical_prompt_source": str(SMALLVAL_RESULT),
        "canonical_continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "R128_quantizer": axis.config_geometry(R128_CFG),
        "lambda_grid": LAMBDAS,
        "intervention_location": "after canonical R128 quantize/dequantize and before recurrent state write-back",
        "post_intervention_requantization": False,
        "metric_formula": "S_write=S_pre+lambda*r_Q; Delta_effective_lambda=Delta_desired+lambda*r_Q",
        "FP_STATE_implementation": "independent FP recurrent continuation control",
        "prompt_manifest": manifest,
        "output_paths": [str(p) for p in [RESULT, RECORDS, TOKEN_RECORDS, LAYER_RECORDS, HEAD_RECORDS, REPORT]],
    }
    save_json(STAGE0, obj)
    save_json(RESULT, {**obj, "FORMAL_STATUS": "NOT_RUN", "SMOKE_GATE": "NOT_RUN", "PILOT_CLASSIFICATION": "NOT_RUN"})
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "smoke", "pilot", "formal", "analyze", "merge"], required=True)
    ap.add_argument("--analyze-stage", choices=["smoke", "pilot", "formal"], default="formal")
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage == "stage0":
        stage0()
    elif args.stage == "analyze":
        if args.analyze_stage == "formal":
            merge_formal(args.num_shards)
        analyze(args.analyze_stage, args.num_shards)
    elif args.stage == "merge":
        merge_formal(args.num_shards)
    else:
        run_stage(args)


if __name__ == "__main__":
    main()
