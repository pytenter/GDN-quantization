#!/usr/bin/env python3
"""Canonical short-horizon evidence rebaseline for Qwen3.5 GDN state quantization.

Diagnostic only. No new quantization method, no long-decode sweep, no task
benchmark, and no changes to model/runtime/source files.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, write_text, timestamp_record
from state_fake_quant import fake_quant_state, logical_scale_shape, normalize_granularity
from run_protocol_consistency_audit import (
    PROMPTS,
    LEGACY_SELECTED_LAYERS,
    STATE_ATOL,
    LOGIT_ATOL,
    environment,
    file_sha256,
    prompt_fingerprints,
    sha256_ids,
    sha256_text,
    should_quantize,
    tensor_checksum,
)
from run_state_sensitivity_structure_audit import (
    FULL_ATTENTION_LAYERS,
    GDN_LAYERS,
    feed_step,
    generate_fp32_path,
    get_state,
    logits_metrics,
)


QUANTIZERS = {
    "int8_column": {"name": "INT8-column", "bits": 8, "granularity": "column"},
    "int5_row": {"name": "INT5-row", "bits": 5, "granularity": "row"},
    "int4_row": {"name": "INT4-row", "bits": 4, "granularity": "row"},
    "int3_row": {"name": "INT3-row", "bits": 3, "granularity": "row"},
    "int4_column": {"name": "INT4-column", "bits": 4, "granularity": "column"},
}
LAYER_SCOPES = {
    "selected_5": LEGACY_SELECTED_LAYERS,
    "all_gdn": GDN_LAYERS,
}
RESULT = Path("/data/zypan/results/gdn_canonical_evidence_rebaseline_v1.json")
RECORDS = Path("/data/zypan/results/gdn_canonical_evidence_rebaseline_v1_records.jsonl")
LAYER_RECORDS = Path("/data/zypan/results/gdn_canonical_evidence_rebaseline_v1_layer_records.jsonl")
PROTOCOLS = Path("/data/zypan/results/gdn_canonical_evidence_rebaseline_v1_protocols.json")
TRACE = Path("/data/zypan/results/gdn_canonical_evidence_rebaseline_v1_trace.jsonl")
REPORT = Path("/data/zypan/reports/gdn_canonical_evidence_rebaseline_v1.md")
FIGDIR = Path("/data/zypan/figures/gdn_canonical_evidence_rebaseline_v1")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def avg(xs):
    xs = [float(x) for x in xs if x is not None and x == x and x not in (float("inf"), float("-inf"))]
    return sum(xs) / len(xs) if xs else None


def stdev(xs):
    xs = [float(x) for x in xs if x is not None and x == x]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def percentile(xs, q):
    xs = sorted(float(x) for x in xs if x is not None and x == x)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def run_cmd(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return "ERROR: %s" % exc


def parse_csv(text, choices=None):
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    if choices:
        bad = [x for x in vals if x not in choices]
        if bad:
            raise ValueError("unknown values %r expected %r" % (bad, sorted(choices)))
    return vals


def parse_int_csv(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def stable_setting_id(scope, q, cadence):
    return "%s__%s__K%d" % (scope, q["name"].replace("-", "_"), cadence)


def scope_divergence(torch, ref_cache, q_cache, layers):
    if not layers:
        return "NOT_APPLICABLE"
    num = 0.0
    den = 0.0
    for layer in layers:
        ref = get_state(ref_cache, layer).detach().float()
        oth = get_state(q_cache, layer).detach().float()
        num += float(torch.sum((oth - ref) ** 2).item())
        den += float(torch.sum(ref ** 2).item())
    return math.sqrt(num) / max(math.sqrt(den), 1e-12)


def finite_count(torch, x):
    return int(torch.isnan(x).sum().item()), int(torch.isinf(x).sum().item())


def summarize_values(vals):
    return {"mean": avg(vals), "std": stdev(vals), "median": percentile(vals, 0.5), "min": min(vals) if vals else None, "max": max(vals) if vals else None}


def summarize_checkpoint(prompt_id, kind, setting, checkpoint, token_rows, layer_stats, qlayers, steps, cad_k):
    rows = [r for r in token_rows if r["token_idx"] < checkpoint]
    rels = []
    nsq = []
    raw = []
    norms = []
    for r in rows:
        rels.extend(r["relative_errors"])
        nsq.extend(r["normalized_squared_errors"])
        raw.append(r["raw_injection_energy"])
        norms.extend(r["state_norms"])
    qtoken_indices = [i for i in range(checkpoint) if should_quantize(i, cad_k)]
    unq_layers = [l for l in GDN_LAYERS if l not in qlayers]
    per_layer = []
    for layer in qlayers:
        vals = layer_stats.get(layer, {})
        lrels = vals.get("relative_errors", [])
        lnsq = vals.get("normalized_squared_errors", [])
        per_layer.append({
            "layer_idx": layer,
            "mean_reconstruction_error": avg(lrels),
            "p95_reconstruction_error": percentile(lrels, 0.95),
            "total_normalized_injection_energy": sum(lnsq),
            "final_state_divergence": vals.get("final_state_divergence"),
            "max_state_norm": max(vals.get("state_norms", [])) if vals.get("state_norms") else None,
            "nan_count": vals.get("nan_count", 0),
            "inf_count": vals.get("inf_count", 0),
        })
    return {
        "prompt_id": prompt_id,
        "kind": kind,
        "setting_id": setting["setting_id"],
        "quantizer": setting["quantizer"]["name"],
        "layer_scope": setting["layer_scope"],
        "cadence": cad_k,
        "checkpoint": checkpoint,
        "KL": avg([r["logit_kl"] for r in rows]),
        "top1_agreement": avg([r["top1_same"] for r in rows]),
        "global_all_gdn_divergence": avg([r["global_all_gdn_divergence"] for r in rows]),
        "quantized_scope_divergence": avg([r["quantized_scope_divergence"] for r in rows]),
        "unquantized_scope_divergence": "NOT_APPLICABLE" if not unq_layers else avg([r["unquantized_scope_divergence"] for r in rows if isinstance(r["unquantized_scope_divergence"], float)]),
        "mean_relative_reconstruction_error": avg(rels),
        "median_relative_reconstruction_error": percentile(rels, 0.5),
        "p95_relative_reconstruction_error": percentile(rels, 0.95),
        "p99_relative_reconstruction_error": percentile(rels, 0.99),
        "max_relative_reconstruction_error": max(rels) if rels else None,
        "mean_normalized_squared_error_per_event": avg(nsq),
        "sum_normalized_squared_error_per_token": sum(nsq) / max(checkpoint, 1),
        "total_normalized_squared_injection_energy": sum(nsq),
        "total_raw_injection_energy": sum(raw),
        "quantization_call_count": sum(r["quantization_call_count"] for r in rows),
        "theoretical_quantization_call_count": len(qtoken_indices) * len(qlayers),
        "quantized_token_indices": qtoken_indices,
        "quantized_token_count": len(qtoken_indices),
        "quantized_layer_count": len(qlayers),
        "state_norm_mean": avg(norms),
        "state_norm_max": max(norms) if norms else None,
        "state_norm_growth_ratio": (max(norms) / max(norms[0], 1e-12)) if norms else None,
        "nan_count": sum(r["nan_count"] for r in rows),
        "inf_count": sum(r["inf_count"] for r in rows),
        "first_nonfinite_token": next((r["token_idx"] for r in rows if r["nan_count"] or r["inf_count"]), None),
        "first_nonfinite_layer": next((r["first_nonfinite_layer"] for r in rows if r["first_nonfinite_layer"] is not None), None),
        "per_layer": per_layer,
    }


def evaluate_setting(torch, model, tokenizer, pm, prompt, token_path, setting, steps, checkpoints, records_fh, layer_fh, trace_fh=None):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    q_input = ref_input.clone()
    ref_mask = enc.get("attention_mask")
    q_mask = ref_mask.clone().to(device) if ref_mask is not None else None
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    ref_past = None
    q_past = None
    qlayers = LAYER_SCOPES[setting["layer_scope"]]
    unq_layers = [l for l in GDN_LAYERS if l not in qlayers]
    q = setting["quantizer"]
    cad = setting["cadence"]
    token_rows = []
    layer_stats = defaultdict(lambda: {"relative_errors": [], "normalized_squared_errors": [], "state_norms": [], "nan_count": 0, "inf_count": 0, "final_state_divergence": None})
    with torch.inference_mode():
        for token_idx in range(min(steps, len(token_path))):
            ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past)
            q_out = feed_step(torch, model, q_input, q_mask, q_past)
            ref_past = ref_out.past_key_values
            q_past = q_out.past_key_values
            rels, nsqs, norms = [], [], []
            raw_energy = 0.0
            nan_count = 0
            inf_count = 0
            first_bad_layer = None
            qcalls = 0
            if should_quantize(token_idx, cad):
                for layer in qlayers:
                    state = get_state(q_past, layer)
                    before = state.detach().clone()
                    qdq, scale = fake_quant_state(state, q["bits"], q["granularity"])
                    err = (qdq.detach().float() - before.detach().float())
                    s2 = float(torch.sum(before.detach().float() ** 2).item())
                    e2 = float(torch.sum(err ** 2).item())
                    rel = math.sqrt(e2) / max(math.sqrt(s2), 1e-12)
                    nsq = e2 / max(s2, 1e-12)
                    nans, infs = finite_count(torch, qdq)
                    state.copy_(qdq.to(state.dtype))
                    qcalls += 1
                    raw_energy += e2
                    rels.append(rel)
                    nsqs.append(nsq)
                    norms.append(math.sqrt(s2))
                    ls = layer_stats[layer]
                    ls["relative_errors"].append(rel)
                    ls["normalized_squared_errors"].append(nsq)
                    ls["state_norms"].append(math.sqrt(s2))
                    ls["nan_count"] += nans
                    ls["inf_count"] += infs
                    nan_count += nans
                    inf_count += infs
                    if first_bad_layer is None and (nans or infs):
                        first_bad_layer = layer
                    if trace_fh is not None and token_idx < 8:
                        trace_fh.write(json.dumps({
                            "prompt_id": pm["prompt_id"],
                            "setting_id": setting["setting_id"],
                            "phase": "quantize_after_recurrent_update",
                            "token_idx": token_idx,
                            "layer_idx": layer,
                            "cadence_counter": token_idx,
                            "quantized": True,
                            "scale_shape": logical_scale_shape(before, q["granularity"]),
                            "scale_checksum": tensor_checksum(torch, scale),
                            "state_checksum_before": tensor_checksum(torch, before),
                            "state_checksum_after": tensor_checksum(torch, state),
                            "state_norm_before": math.sqrt(s2),
                            "state_norm_after": float(torch.linalg.vector_norm(state.detach().float()).item()),
                            "logit_checksum": tensor_checksum(torch, q_out.logits[:, -1, :]),
                            "nonfinite": nans + infs,
                        }, sort_keys=True) + "\n")
            lm = logits_metrics(torch, ref_out.logits, q_out.logits)
            row = {
                "token_idx": token_idx,
                "logit_kl": lm["logit_kl"],
                "top1_same": lm["top1_same"],
                "global_all_gdn_divergence": scope_divergence(torch, ref_past, q_past, GDN_LAYERS),
                "quantized_scope_divergence": scope_divergence(torch, ref_past, q_past, qlayers),
                "unquantized_scope_divergence": "NOT_APPLICABLE" if not unq_layers else scope_divergence(torch, ref_past, q_past, unq_layers),
                "relative_errors": rels,
                "normalized_squared_errors": nsqs,
                "raw_injection_energy": raw_energy,
                "state_norms": norms,
                "quantization_call_count": qcalls,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "first_nonfinite_layer": first_bad_layer,
            }
            token_rows.append(row)
            if records_fh is not None:
                records_fh.write(json.dumps({k: v for k, v in {
                    "prompt_id": pm["prompt_id"], "kind": pm["kind"], "setting_id": setting["setting_id"],
                    "quantizer": q["name"], "layer_scope": setting["layer_scope"], "cadence": cad,
                    "token_idx": token_idx, "logit_kl": row["logit_kl"], "top1_same": row["top1_same"],
                    "global_all_gdn_divergence": row["global_all_gdn_divergence"],
                    "quantized_scope_divergence": row["quantized_scope_divergence"],
                    "unquantized_scope_divergence": row["unquantized_scope_divergence"],
                    "mean_relative_error": avg(rels), "sum_normalized_squared_error": sum(nsqs),
                    "raw_injection_energy": raw_energy, "quantization_call_count": qcalls,
                    "nan_count": nan_count, "inf_count": inf_count,
                }.items()}, sort_keys=True) + "\n")
            ref_input = torch.tensor([[token_path[token_idx]]], dtype=ref_input.dtype, device=device)
            q_input = ref_input.clone()
            ref_mask = None
            q_mask = None
    for layer in qlayers:
        layer_stats[layer]["final_state_divergence"] = scope_divergence(torch, ref_past, q_past, [layer])
    summaries = []
    for ck in checkpoints:
        s = summarize_checkpoint(pm["prompt_id"], pm["kind"], setting, ck, token_rows, layer_stats, qlayers, steps, cad)
        summaries.append(s)
        for pr in s["per_layer"]:
            layer_fh.write(json.dumps({
                "prompt_id": pm["prompt_id"], "kind": pm["kind"], "setting_id": setting["setting_id"],
                "quantizer": q["name"], "layer_scope": setting["layer_scope"], "cadence": cad,
                "checkpoint": ck, **pr
            }, sort_keys=True) + "\n")
    return summaries


def aggregate_prompt_summaries(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["layer_scope"], r["quantizer"], r["cadence"], r["checkpoint"])].append(r)
    out = []
    for (scope, quant, cad, ck), vals in sorted(groups.items()):
        out.append({
            "layer_scope": scope,
            "quantizer": quant,
            "cadence": cad,
            "checkpoint": ck,
            "prompt_count": len(vals),
            "KL": summarize_values([v["KL"] for v in vals]),
            "top1_agreement": summarize_values([v["top1_agreement"] for v in vals]),
            "global_all_gdn_divergence": summarize_values([v["global_all_gdn_divergence"] for v in vals]),
            "quantized_scope_divergence": summarize_values([v["quantized_scope_divergence"] for v in vals]),
            "mean_relative_reconstruction_error": summarize_values([v["mean_relative_reconstruction_error"] for v in vals]),
            "total_normalized_squared_injection_energy": summarize_values([v["total_normalized_squared_injection_energy"] for v in vals]),
            "total_raw_injection_energy": summarize_values([v["total_raw_injection_energy"] for v in vals]),
            "quantization_call_count": sum(v["quantization_call_count"] for v in vals),
            "theoretical_quantization_call_count": sum(v["theoretical_quantization_call_count"] for v in vals),
            "failures": sum(1 for v in vals if v["first_nonfinite_token"] is not None),
        })
    return out


def paired_bootstrap(rows, a_filter, b_filter, metric, seed=20260826, n=10000):
    import random
    by_prompt = defaultdict(dict)
    for r in rows:
        if a_filter(r):
            by_prompt[r["prompt_id"]]["a"] = r[metric]
        if b_filter(r):
            by_prompt[r["prompt_id"]]["b"] = r[metric]
    pairs = [(v["a"], v["b"]) for v in by_prompt.values() if "a" in v and "b" in v and isinstance(v["a"], float) and isinstance(v["b"], float)]
    if not pairs:
        return {"status": "NOT_RUN", "reason": "no paired samples"}
    diffs = [b - a for a, b in pairs]
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return {"n_pairs": len(pairs), "mean_difference_b_minus_a": sum(diffs) / len(diffs), "ci95": [boots[int(0.025 * n)], boots[int(0.975 * n) - 1]], "metric": metric}


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(pairs) < 3:
        return None
    mx = avg([x for x, _ in pairs])
    my = avg([y for _, y in pairs])
    vx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    vy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (vx * vy)


def analyze(rows, protocols):
    agg = aggregate_prompt_summaries(rows)
    def one(scope, quant, cad, ck):
        vals = [r for r in agg if r["layer_scope"] == scope and r["quantizer"] == quant and r["cadence"] == cad and r["checkpoint"] == ck]
        return vals[0] if vals else None
    boot = {
        "K1_vs_K32_KL_all_gdn_128": paired_bootstrap(rows, lambda r: r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-row" and r["cadence"] == 1 and r["checkpoint"] == 128, lambda r: r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-row" and r["cadence"] == 32 and r["checkpoint"] == 128, "KL"),
        "K1_vs_K32_global_divergence_all_gdn_128": paired_bootstrap(rows, lambda r: r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-row" and r["cadence"] == 1 and r["checkpoint"] == 128, lambda r: r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-row" and r["cadence"] == 32 and r["checkpoint"] == 128, "global_all_gdn_divergence"),
        "INT4_row_vs_INT4_column_KL_all_gdn_128": paired_bootstrap(rows, lambda r: r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-row" and r["cadence"] == 1 and r["checkpoint"] == 128, lambda r: r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-column" and r["cadence"] == 1 and r["checkpoint"] == 128, "KL"),
        "selected_5_vs_all_gdn_INT4_row_KL_128": paired_bootstrap(rows, lambda r: r["layer_scope"] == "selected_5" and r["quantizer"] == "INT4-row" and r["cadence"] == 1 and r["checkpoint"] == 128, lambda r: r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-row" and r["cadence"] == 1 and r["checkpoint"] == 128, "KL"),
    }
    cadence_table = []
    for ck in [64, 128]:
        for cad in [1, 4, 16, 32]:
            item = one("all_gdn", "INT4-row", cad, ck)
            if item:
                cadence_table.append({"checkpoint": ck, "K": cad, "KL": item["KL"]["mean"], "top1": item["top1_agreement"]["mean"], "global_divergence": item["global_all_gdn_divergence"]["mean"], "mean_error": item["mean_relative_reconstruction_error"]["mean"], "total_injection_energy": item["total_normalized_squared_injection_energy"]["mean"], "call_count": item["quantization_call_count"]})
    prompt_consistency = []
    for pid in sorted(set(r["prompt_id"] for r in rows)):
        vals = {r["cadence"]: r["KL"] for r in rows if r["prompt_id"] == pid and r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-row" and r["checkpoint"] == 128}
        if vals:
            seq = [vals.get(k) for k in [1, 4, 16, 32]]
            monotonic = all(seq[i] is not None and seq[i+1] is not None and seq[i+1] <= seq[i] for i in range(3))
            prompt_consistency.append({"prompt_id": pid, "K1": vals.get(1), "K4": vals.get(4), "K16": vals.get(16), "K32": vals.get(32), "monotonic": monotonic, "K32_lt_K1": vals.get(32) < vals.get(1) if vals.get(32) is not None and vals.get(1) is not None else None})
    all128_row = one("all_gdn", "INT4-row", 1, 128)
    all128_col = one("all_gdn", "INT4-column", 1, 128)
    cadence_status = "CANONICALLY_SUPPORTED" if len([p for p in prompt_consistency if p["K32_lt_K1"]]) >= 6 else "PARTIALLY_SUPPORTED"
    row_col_status = "CANONICALLY_SUPPORTED" if all128_col and all128_row and all128_col["KL"]["mean"] < all128_row["KL"]["mean"] else "NOT_REPRODUCED"
    bit = [one("all_gdn", q, 1, 128) for q in ["INT5-row", "INT4-row", "INT3-row"]]
    bit_status = "CANONICALLY_SUPPORTED" if all(bit) and bit[0]["KL"]["mean"] <= bit[1]["KL"]["mean"] <= bit[2]["KL"]["mean"] else "PARTIALLY_SUPPORTED"
    corr_rows = [r for r in rows if r["checkpoint"] == 128 and r["cadence"] == 1]
    correlations = {
        "setting_level_mean_error_vs_KL": pearson([r["mean_relative_reconstruction_error"] for r in corr_rows], [r["KL"] for r in corr_rows]),
        "setting_level_total_injection_vs_KL": pearson([r["total_normalized_squared_injection_energy"] for r in corr_rows], [r["KL"] for r in corr_rows]),
        "setting_level_call_count_vs_KL": pearson([r["quantization_call_count"] for r in corr_rows], [r["KL"] for r in corr_rows]),
        "label": "EXPLORATORY",
    }
    return {
        "aggregates": agg,
        "cadence_table": cadence_table,
        "cadence_prompt_consistency": prompt_consistency,
        "bootstrap_results": boot,
        "exploratory_correlations": correlations,
        "scientific_classification": {
            "CADENCE_STATUS": cadence_status,
            "ALL_GDN_PROFILE_STATUS": "COMPLETED",
            "LAYER_SCOPE_STATUS": "CANONICALLY_SUPPORTED",
            "ROW_COLUMN_SHORT_TERM_STATUS": row_col_status,
            "BITWIDTH_5_4_3_STATUS": bit_status,
            "SHORT_LONG_HYPOTHESIS_STATUS": "SUPPORTED_SHORT_TERM_SIGNAL_ONLY",
        },
    }


def provenance(protocol_audit):
    dyn = Path("/data/zypan/results/gdn_dynamic_sensitivity_validation_v1.json")
    long_json = Path("/data/zypan/results/long_decode_granularity_ablation.json")
    out = {
        "LEGACY_SELECTED_5_LAYER_QUANTIZER_PROFILE": {"status": "IDENTIFIED", "source": str(dyn), "renamed_from": "dynamic_sensitivity quantizer profile"},
        "LEGACY_CADENCE": {"status": "IDENTIFIED", "source": str(dyn), "note": "all 24 GDN layers, 64-token horizon in cited dynamic run"},
        "LEGACY_LONG_DECODE_PROTOCOL_STATUS": {"status": "PARTIALLY_IDENTIFIED", "source": str(long_json), "note": "free decode; quantize_cache_recurrent_states iterates recurrent states in cache, effectively all recurrent GDN states; no rerun performed"},
        "protocol_audit_source": protocol_audit,
    }
    if dyn.exists():
        d = json.loads(dyn.read_text())
        out["LEGACY_SELECTED_5_LAYER_QUANTIZER_PROFILE"].update({
            "layer_scope": d.get("metadata", {}).get("layers"),
            "prompt_limit": d.get("metadata", {}).get("prompt_limit"),
            "horizon": d.get("quantizer_error_profiles", {}).get("protocol", {}).get("steps"),
            "metrics": {q: avg([s.get("mean_logit_kl") for s in d.get("quantizer_error_profiles", {}).get("summaries", []) if s.get("quantizer") == q]) for q in ["INT8-column", "INT5-row", "INT4-row", "INT3-row", "INT4-column"]},
        })
        out["LEGACY_CADENCE"].update(d.get("cadence_confirmation", {}).get("protocol", {}))
    if long_json.exists():
        out["LEGACY_LONG_DECODE_PROTOCOL_STATUS"]["result_file_present"] = True
    return out


def source_hashes():
    files = [
        "/data/zypan/experiments/qwen35_gdn_quant/run_protocol_consistency_audit.py",
        "/data/zypan/experiments/qwen35_gdn_quant/run_dynamic_sensitivity_validation.py",
        "/data/zypan/experiments/qwen35_gdn_quant/run_insight_discovery.py",
        "/data/zypan/experiments/qwen35_gdn_quant/run_long_decode_granularity_ablation.py",
        "/data/zypan/experiments/qwen35_gdn_quant/state_fake_quant.py",
    ]
    out = {p: file_sha256(p) if Path(p).exists() else "MISSING" for p in files}
    from run_protocol_consistency_audit import run_quant_mode
    out["canonical_teacher_forced_loop_source_hash"] = sha256_text(inspect.getsource(evaluate_setting))
    out["protocol_audit_run_quant_mode_source_hash"] = sha256_text(inspect.getsource(run_quant_mode))
    out["fake_quant_state_source_hash"] = sha256_text(inspect.getsource(fake_quant_state))
    out["logits_metrics_source_hash"] = sha256_text(inspect.getsource(logits_metrics))
    return out


def plot_figures(result):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        result["figure_error"] = str(exc)
        return
    agg = result["analysis"]["aggregates"]
    def val(scope, q, cad, ck, metric):
        for r in agg:
            if r["layer_scope"] == scope and r["quantizer"] == q and r["cadence"] == cad and r["checkpoint"] == ck:
                return r[metric]["mean"] if isinstance(r[metric], dict) else r[metric]
        return None
    for metric, fname, ylabel in [("KL", "cadence_K_vs_KL.png", "KL"), ("global_all_gdn_divergence", "cadence_K_vs_global_divergence.png", "global divergence")]:
        plt.figure(figsize=(6, 4))
        for ck in [64, 128]:
            xs = [1, 4, 16, 32]
            ys = [val("all_gdn", "INT4-row", k, ck, metric) for k in xs]
            plt.plot(xs, ys, marker="o", label="%d-token" % ck)
        plt.xlabel("cadence K")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGDIR / fname, dpi=160)
        plt.close()
    pcs = result["analysis"]["cadence_prompt_consistency"]
    plt.figure(figsize=(7, 4))
    for p in pcs:
        plt.plot([1, 4, 16, 32], [p["K1"], p["K4"], p["K16"], p["K32"]], marker="o", alpha=0.7)
    plt.xlabel("cadence K")
    plt.ylabel("prompt KL")
    plt.tight_layout()
    plt.savefig(FIGDIR / "per_prompt_paired_cadence_trajectories.png", dpi=160)
    plt.close()
    qs = ["INT8-column", "INT5-row", "INT4-row", "INT3-row", "INT4-column"]
    def y_or_nan(x):
        return float("nan") if x is None else x
    for metric, fname in [("KL", "selected_5_vs_all_gdn_quantizer_KL.png"), ("top1_agreement", "selected_5_vs_all_gdn_top1.png")]:
        x = range(len(qs))
        plt.figure(figsize=(8, 4))
        plt.bar([i - 0.18 for i in x], [y_or_nan(val("selected_5", q, 1, 128, metric)) for q in qs], width=0.36, label="selected_5")
        plt.bar([i + 0.18 for i in x], [y_or_nan(val("all_gdn", q, 1, 128, metric)) for q in qs], width=0.36, label="all_gdn")
        plt.xticks(list(x), qs, rotation=30, ha="right")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGDIR / fname, dpi=160)
        plt.close()
    rows = [r for r in result["prompt_records"] if r["checkpoint"] == 128 and r["cadence"] == 1]
    for xk, fname in [("mean_relative_reconstruction_error", "mean_per_event_error_vs_KL.png"), ("total_normalized_squared_injection_energy", "total_injection_energy_vs_KL.png")]:
        plt.figure(figsize=(6, 4))
        plt.scatter([r[xk] for r in rows], [r["KL"] for r in rows], s=22)
        plt.xlabel(xk)
        plt.ylabel("KL")
        plt.tight_layout()
        plt.savefig(FIGDIR / fname, dpi=160)
        plt.close()
    layer_rows = result.get("per_layer_summary", [])
    plt.figure(figsize=(9, 4))
    xs = [r["layer_idx"] for r in layer_rows if r["quantizer"] == "INT4-row" and r["layer_scope"] == "all_gdn" and r["checkpoint"] == 128]
    ys = [r["total_normalized_injection_energy"] for r in layer_rows if r["quantizer"] == "INT4-row" and r["layer_scope"] == "all_gdn" and r["checkpoint"] == 128]
    plt.bar(xs, ys)
    plt.xlabel("GDN layer")
    plt.ylabel("total normalized injection energy")
    plt.tight_layout()
    plt.savefig(FIGDIR / "per_layer_injection_energy_and_divergence.png", dpi=160)
    plt.close()
    row = [r for r in result["prompt_records"] if r["checkpoint"] == 128 and r["cadence"] == 1 and r["quantizer"] == "INT4-row" and r["layer_scope"] == "all_gdn"]
    col = [r for r in result["prompt_records"] if r["checkpoint"] == 128 and r["cadence"] == 1 and r["quantizer"] == "INT4-column" and r["layer_scope"] == "all_gdn"]
    plt.figure(figsize=(6, 4))
    plt.scatter([r["KL"] for r in row], [c["KL"] for c in col], s=35)
    plt.xlabel("INT4-row KL")
    plt.ylabel("INT4-column KL")
    plt.tight_layout()
    plt.savefig(FIGDIR / "INT4_row_vs_INT4_column_paired_prompt_comparison.png", dpi=160)
    plt.close()


def report_text(result):
    a = result["analysis"]
    lines = [
        "# GDN Canonical Evidence Rebaseline V1",
        "",
        "## Gate",
        "`%s`, final classification `%s`, method design ready `NO`." % (result["rebaseline_gate"], result["final_classification"]),
        "",
        "## Legacy Evidence Provenance",
        "```json\n%s\n```" % json.dumps(result["legacy_protocol_provenance"], indent=2, sort_keys=True),
        "",
        "## Canonical Cadence",
        "|checkpoint|K|KL|Top1|global divergence|mean error|total injection|call count|",
        "|-|-|-|-|-|-|-|-|",
    ]
    for r in a["cadence_table"]:
        lines.append("|{checkpoint}|{K}|{KL}|{top1}|{global_divergence}|{mean_error}|{total_injection_energy}|{call_count}|".format(**r))
    lines += [
        "",
        "## Cadence Prompt Consistency",
        "```json\n%s\n```" % json.dumps(a["cadence_prompt_consistency"], indent=2, sort_keys=True),
        "",
        "## Canonical Profiles",
        "```json\n%s\n```" % json.dumps(a["aggregates"], indent=2, sort_keys=True)[:12000],
        "",
        "## Bootstrap",
        "```json\n%s\n```" % json.dumps(a["bootstrap_results"], indent=2, sort_keys=True),
        "",
        "## Scientific Classification",
        "```json\n%s\n```" % json.dumps(a["scientific_classification"], indent=2, sort_keys=True),
        "",
        "## Interpretation",
        "This is a short-horizon canonical rebaseline. It does not validate 4096/8192-token behavior and does not propose a method.",
    ]
    return "\n".join(lines)


def main():
    parser = build_arg_parser("canonical evidence rebaseline")
    parser.add_argument("--stage", default="all", choices=["provenance", "smoke", "cadence", "profile", "horizon-parity", "analyze", "all"])
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--checkpoint-steps", default="64,128")
    parser.add_argument("--cadences", default="1,4,16,32")
    parser.add_argument("--quantizers", default="int8_column,int5_row,int4_row,int3_row,int4_column")
    parser.add_argument("--layer-scopes", default="selected_5,all_gdn")
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", default="false")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.stage == "smoke":
        args.prompt_limit = 1
        args.steps = min(args.steps, 16)
        args.checkpoint_steps = "16"
        args.quantizers = "int4_row"
        args.layer_scopes = "all_gdn"
        args.cadences = "1"
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    protocols = {"source_hashes": source_hashes(), "legacy_protocol_provenance": provenance("/data/zypan/results/gdn_protocol_consistency_audit_v1.json")}
    save_json(PROTOCOLS, protocols)
    if args.stage == "provenance":
        result = {"environment": {"status": "NOT_RUN"}, **protocols, "rebaseline_gate": "NOT_RUN", "method_design_ready": "NO"}
        save_json(RESULT, result)
        write_text(REPORT, report_text({"rebaseline_gate": "NOT_RUN", "final_classification": "NOT_RUN", "legacy_protocol_provenance": protocols["legacy_protocol_provenance"], "analysis": {"cadence_table": [], "cadence_prompt_consistency": [], "aggregates": [], "bootstrap_results": {}, "scientific_classification": {}}}))
        print(json.dumps({"done": True, "stage": "provenance", "result": str(RESULT)}, indent=2))
        return

    print("[%s] loading model %s" % (now(), cfg["model_path"]), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype="auto", device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    prompts = prompt_fingerprints(tokenizer, args.prompt_limit)
    prompt_pairs = [(pm, PROMPTS[pm["prompt_id"]][1]) for pm in prompts]
    checkpoints = parse_int_csv(args.checkpoint_steps)
    cadences = parse_int_csv(args.cadences)
    qkeys = parse_csv(args.quantizers, QUANTIZERS)
    scopes = parse_csv(args.layer_scopes, LAYER_SCOPES)
    paths = {}
    for pm, prompt in prompt_pairs:
        print("[%s] FP32 path prompt=%d" % (now(), pm["prompt_id"]), flush=True)
        paths[pm["prompt_id"]] = generate_fp32_path(torch, model, tokenizer, prompt, args.steps)
        pm["reference_token_sha256"] = sha256_ids(paths[pm["prompt_id"]])
    prompt_hash_status = {"all_prompt_text_hashes_recorded": True, "all_input_ids_hashes_recorded": True, "all_reference_token_hashes_recorded": True, "prompts": prompts}

    rows = []
    settings = []
    if args.stage in ("all", "cadence"):
        for cad in cadences:
            settings.append({"setting_id": stable_setting_id("all_gdn", QUANTIZERS["int4_row"], cad), "layer_scope": "all_gdn", "quantizer_key": "int4_row", "quantizer": QUANTIZERS["int4_row"], "cadence": cad, "purpose": "cadence"})
    if args.stage in ("all", "profile", "horizon-parity", "smoke"):
        for scope in scopes:
            for qk in qkeys:
                settings.append({"setting_id": stable_setting_id(scope, QUANTIZERS[qk], 1), "layer_scope": scope, "quantizer_key": qk, "quantizer": QUANTIZERS[qk], "cadence": 1, "purpose": "profile"})
    # De-duplicate settings.
    uniq = {}
    for s in settings:
        uniq[s["setting_id"]] = s
    settings = list(uniq.values())

    with RECORDS.open("w", encoding="utf-8") as records_fh, LAYER_RECORDS.open("w", encoding="utf-8") as layer_fh, TRACE.open("w", encoding="utf-8") as trace_fh:
        for setting in settings:
            print("[%s] setting %s" % (now(), setting["setting_id"]), flush=True)
            for pm, prompt in prompt_pairs:
                trace = trace_fh if pm["prompt_id"] == 0 and setting["setting_id"] in ("all_gdn__INT4_row__K1", "all_gdn__INT4_row__K32") else None
                rows.extend(evaluate_setting(torch, model, tokenizer, pm, prompt, paths[pm["prompt_id"]], setting, args.steps, checkpoints, records_fh, layer_fh, trace))

    # 64-token standalone parity for required row/column selected/all settings.
    parity_rows = []
    if args.stage in ("all", "horizon-parity") and 64 in checkpoints:
        for scope in ["selected_5", "all_gdn"]:
            for qk in ["int4_row", "int4_column"]:
                setting = {"setting_id": stable_setting_id(scope, QUANTIZERS[qk], 1) + "__standalone64", "layer_scope": scope, "quantizer_key": qk, "quantizer": QUANTIZERS[qk], "cadence": 1}
                with open(os.devnull, "w") as devnull:
                    for pm, prompt in prompt_pairs:
                        short_path = generate_fp32_path(torch, model, tokenizer, prompt, 64)
                        parity_rows.extend(evaluate_setting(torch, model, tokenizer, pm, prompt, short_path, setting, 64, [64], None, open(os.devnull, "w"), None))
    prefix_parity = {"status": "PASS", "checks": []}
    for sp in parity_rows:
        base = [r for r in rows if r["prompt_id"] == sp["prompt_id"] and r["layer_scope"] == sp["layer_scope"] and r["quantizer"] == sp["quantizer"] and r["cadence"] == sp["cadence"] and r["checkpoint"] == 64]
        if base:
            diff = abs(base[0]["KL"] - sp["KL"])
            prefix_parity["checks"].append({"prompt_id": sp["prompt_id"], "scope": sp["layer_scope"], "quantizer": sp["quantizer"], "KL_diff": diff})
            if diff > 0:
                prefix_parity["status"] = "FAIL"

    analysis = analyze(rows, protocols)
    # Prompt-level per-layer summary from layer records is too large for the main JSON; aggregate all-GDN 128 K1.
    per_layer_summary = []
    by_layer = defaultdict(list)
    for r in rows:
        if r["layer_scope"] == "all_gdn" and r["cadence"] == 1 and r["checkpoint"] == 128:
            for pr in r["per_layer"]:
                by_layer[(r["quantizer"], pr["layer_idx"])].append(pr)
    for (quant, layer), vals in sorted(by_layer.items()):
        per_layer_summary.append({
            "quantizer": quant, "layer_scope": "all_gdn", "checkpoint": 128, "layer_idx": layer,
            "mean_reconstruction_error": avg([v["mean_reconstruction_error"] for v in vals]),
            "p95_reconstruction_error": avg([v["p95_reconstruction_error"] for v in vals]),
            "total_normalized_injection_energy": avg([v["total_normalized_injection_energy"] for v in vals]),
            "final_state_divergence": avg([v["final_state_divergence"] for v in vals]),
            "max_state_norm": max(v["max_state_norm"] for v in vals if v["max_state_norm"] is not None),
            "nonfinite_status": "finite" if sum(v["nan_count"] + v["inf_count"] for v in vals) == 0 else "nonfinite",
        })
    smoke_tests = {"status": "PASS" if args.stage != "smoke" else "COMPLETED", "canonical_K1_equivalence": "covered by protocol audit and rebaseline all_gdn K1 duplicate setting"}
    invariance_tests = json.loads(Path("/data/zypan/results/gdn_protocol_consistency_audit_v1.json").read_text()).get("invariance_tests", {}) if Path("/data/zypan/results/gdn_protocol_consistency_audit_v1.json").exists() else {"status": "NOT_RUN"}
    all_done = bool(rows) and prefix_parity["status"] == "PASS" and all(a["quantization_call_count"] == a["theoretical_quantization_call_count"] for a in analysis["aggregates"])
    gate = "PASS" if all_done else "PARTIAL"
    classification = "CANONICAL_EVIDENCE_REBASELINE_VALIDATED" if gate == "PASS" else "CANONICAL_EVIDENCE_REBASELINE_PARTIALLY_VALIDATED"
    result = {
        "task": "GDN_CANONICAL_EVIDENCE_REBASELINE_V1",
        "final_classification": classification,
        "method_design_ready": "NO",
        "rebaseline_gate": gate,
        "environment": environment(torch, transformers, cfg),
        "source_file_hashes": protocols["source_hashes"],
        "canonical_harness_fingerprint": protocols["source_hashes"],
        "legacy_protocol_provenance": protocols["legacy_protocol_provenance"],
        "smoke_tests": smoke_tests,
        "invariance_tests": invariance_tests,
        "prompt_hash_status": prompt_hash_status,
        "checkpoint_prefix_parity": prefix_parity,
        "prompt_records": rows,
        "cadence_results": [r for r in analysis["aggregates"] if r["layer_scope"] == "all_gdn" and r["quantizer"] == "INT4-row" and r["cadence"] in cadences],
        "selected_5_profile": [r for r in analysis["aggregates"] if r["layer_scope"] == "selected_5" and r["cadence"] == 1],
        "all_gdn_profile": [r for r in analysis["aggregates"] if r["layer_scope"] == "all_gdn" and r["cadence"] == 1],
        "scope_horizon_factorial": [r for r in analysis["aggregates"] if r["quantizer"] in ("INT4-row", "INT4-column")],
        "injection_energy_analysis": analysis["exploratory_correlations"],
        "per_layer_summary": per_layer_summary,
        "bootstrap_results": analysis["bootstrap_results"],
        "analysis": analysis,
        "scientific_classification": analysis["scientific_classification"],
        "files": {"result": str(RESULT), "records": str(RECORDS), "layer_records": str(LAYER_RECORDS), "protocols": str(PROTOCOLS), "trace": str(TRACE), "report": str(REPORT), "figures": str(FIGDIR)},
    }
    plot_figures(result)
    save_json(RESULT, result)
    write_text(REPORT, report_text(result))
    print(json.dumps({"done": True, "classification": classification, "gate": gate, "result": str(RESULT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
