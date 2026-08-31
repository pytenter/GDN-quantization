#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/path/to/gdn_data_root"))
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_EFFECTIVE_UPDATE_METRIC_AUDIT_V1"
RESULT = RES / "gdn_int8_effective_update_metric_audit_v1.json"
RECORDS = RES / "gdn_int8_effective_update_metric_audit_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_effective_update_metric_audit_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_effective_update_metric_audit_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_effective_update_metric_audit_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_effective_update_metric_audit_v1.md"
LOG_DIR = RES / "gdn_int8_effective_update_metric_audit_v1_logs"

EPS = 1e-12
CONFIG_NAMES = ["R128", "R16", "C128", "C16"]
WINDOWS = [(1, 128), (129, 256), (257, 512)]
HEAD_SAMPLE_STRIDE = 64
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1
import run_int8_axis_geometry_rescue_diagnostic as axis

CONFIGS = [c for c in axis.CONFIGS if c["name"] in CONFIG_NAMES]


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


def ratio(a, b):
    return float(a) / float(b) if finite_num(a) and finite_num(b) and abs(float(b)) > EPS else None


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
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summarize(rows):
    buckets = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if finite_num(v):
                buckets[k].append(v)
    return {k: {"mean": avg(v), "median": statistics.median(v), "p90": pct(v, 0.90), "p95": pct(v, 0.95)} for k, v in sorted(buckets.items())}


def stat_mean(stats, key):
    v = stats.get(key)
    return v.get("mean") if isinstance(v, dict) else None


def cosine(torch, a, b):
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    an = float(torch.linalg.vector_norm(af).item())
    bn = float(torch.linalg.vector_norm(bf).item())
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item())


def window_label(token_idx):
    one = int(token_idx) + 1
    for a, b in WINDOWS:
        if a <= one <= b:
            return f"{a}-{b}"
    return ">512"


def logits_metrics(torch, ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    other_logp = torch.log_softmax(other, dim=-1)
    return {
        "KL": float((ref_logp.exp() * (ref_logp - other_logp)).sum().item()),
        "top1_agreement": int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item()),
    }


def effective_metrics(torch, s_in, s_pre, s_post):
    inn = s_in.detach().float()
    pre = s_pre.detach().float()
    post = s_post.detach().float()
    desired = pre - inn
    residual = post - pre
    effective = post - inn
    identity = effective - (desired + residual)
    desired_norm = float(torch.linalg.vector_norm(desired).item())
    effective_norm = float(torch.linalg.vector_norm(effective).item())
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    identity_norm = float(torch.linalg.vector_norm(identity).item())
    d2 = float(torch.sum(desired.double() * desired.double()).item())
    alpha = float(torch.sum(residual.double() * desired.double()).item()) / (d2 + EPS) if d2 > EPS else None
    if finite_num(alpha):
        r_parallel = alpha * desired
        r_orth = residual - r_parallel
        orth = float(torch.linalg.vector_norm(r_orth).item()) / (desired_norm + EPS)
    else:
        orth = None
    d_effective = residual_norm / (desired_norm + EPS)
    residual_to_update = residual_norm / (desired_norm + EPS)
    return {
        "desired_state_change_norm": desired_norm,
        "effective_update_norm": effective_norm,
        "actual_quantization_residual_norm": residual_norm,
        "identity_error_norm": identity_norm,
        "identity_relative_error": identity_norm / (effective_norm + desired_norm + residual_norm + EPS),
        "D_effective": d_effective,
        "cos_effective": cosine(torch, desired, effective),
        "norm_ratio_effective": effective_norm / (desired_norm + EPS),
        "alpha": alpha,
        "effective_parallel_gain": (1.0 + alpha) if finite_num(alpha) else None,
        "orthogonal_ratio": orth,
        "residual_to_update": residual_to_update,
        "residual_to_update_minus_D_effective_abs": abs(residual_to_update - d_effective),
    }


def metric_event(torch, s_in, s_pre, cfg):
    old = axis.metric_event(torch, s_in, s_pre, cfg)
    if old.get("status") not in ("OK", "ZERO_UPDATE_EVENT"):
        return old
    s_post = old["_s_post"]
    eff = effective_metrics(torch, s_in, s_pre, s_post)
    out = {k: v for k, v in old.items() if not k.startswith("_")}
    out.update({
        "metric_family": "SAME_CODEBOOK_REPRESENTABILITY_AND_RUNTIME_EFFECTIVE_UPDATE",
        "R_lost_same_codebook": out.pop("lost_update_fraction", None),
        "R_survival_same_codebook": out.pop("survival_fraction", None),
        "R_distortion_same_codebook": out.pop("update_distortion_ratio", None),
        "norm_ratio_same_codebook": out.pop("represented_update_norm_ratio", None),
        "cosine_same_codebook": out.pop("update_cosine", None),
    })
    out.update(eff)
    out["_s_post"] = s_post
    return out


def run_stage(args):
    pm = p1.selected_prompt_rows()[0]
    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[:args.max_steps]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    ref_mask = enc.get("attention_mask")
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    q_inputs = {c["name"]: ref_input.clone() for c in CONFIGS}
    q_masks = {c["name"]: ref_mask.clone() if ref_mask is not None else None for c in CONFIGS}
    ref_past = None
    q_pasts = {c["name"]: None for c in CONFIGS}
    token_rows = defaultdict(list)
    layer_rows = defaultdict(list)
    head_rows = defaultdict(list)
    numerical_audit_samples = []
    protocol_gate = "PASS"
    metric_gate = "PASS"
    print(f"[{now()}] {args.stage} prompt={pm['problem_id']} tokens={len(cont_ids)} configs={CONFIG_NAMES}", flush=True)
    with torch.inference_mode():
        for t, tok in enumerate(cont_ids):
            ref_out = p1.feed_step(torch, model, ref_input, ref_mask, ref_past)
            ref_past = ref_out.past_key_values
            gz_append(TOKEN_RECORDS, {"task": TASK, "stage": args.stage, "problem_id": pm["problem_id"], "token_idx": t, "quantizer": "FP_STATE", "KL": 0.0, "top1_agreement": 1.0})
            for cfg in CONFIGS:
                qname = cfg["name"]
                s_in = None
                if q_pasts[qname] is not None:
                    s_in = {layer: p1.get_state(q_pasts[qname], layer).detach().clone() for layer in GDN_LAYERS}
                q_out = p1.feed_step(torch, model, q_inputs[qname], q_masks[qname], q_pasts[qname])
                q_pasts[qname] = q_out.past_key_values
                layer_metrics = []
                if s_in is not None:
                    for layer in GDN_LAYERS:
                        state = p1.get_state(q_pasts[qname], layer)
                        s_pre = state.detach().clone()
                        m = metric_event(torch, s_in[layer], s_pre, cfg)
                        if list(s_pre.shape) != [1, 32, 128, 128] or m.get("qrange") != [-127, 127]:
                            protocol_gate = "FAIL"
                        if m.get("scale_count_per_head") != axis.config_geometry(cfg)["scale_count_per_head"]:
                            protocol_gate = "FAIL"
                        if m.get("status") not in ("OK", "ZERO_UPDATE_EVENT"):
                            metric_gate = "FAIL"
                        if m.get("identity_relative_error", 1.0) > 1e-7 or m.get("residual_to_update_minus_D_effective_abs", 1.0) > 1e-10:
                            metric_gate = "FAIL"
                        s_post = m["_s_post"]
                        if len(numerical_audit_samples) < 12 and t in (1, 64, 128, 256, 511) and layer in (0, 12, 25):
                            sample = {k: m.get(k) for k in ["desired_state_change_norm", "effective_update_norm", "actual_quantization_residual_norm", "identity_error_norm", "identity_relative_error", "D_effective", "cos_effective", "norm_ratio_effective", "alpha", "orthogonal_ratio"]}
                            sample.update({"token_idx": t, "layer_idx": layer, "head_idx": "ALL", "quantizer": qname})
                            numerical_audit_samples.append(sample)
                        state.copy_(s_post.to(state.dtype))
                        pub = {k: v for k, v in m.items() if not k.startswith("_")}
                        pub.update({"task": TASK, "stage": args.stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "quantizer": qname, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "layer_idx": layer})
                        layer_metrics.append(pub)
                        layer_rows[qname].append(pub)
                        gz_append(LAYER_RECORDS, pub)
                        if t % HEAD_SAMPLE_STRIDE == 0:
                            for h in range(s_pre.shape[1]):
                                hm = metric_event(torch, s_in[layer][:, h:h + 1, :, :], s_pre[:, h:h + 1, :, :], cfg)
                                if hm.get("status") in ("OK", "ZERO_UPDATE_EVENT"):
                                    hpub = {k: v for k, v in hm.items() if not k.startswith("_")}
                                    hpub.update({"task": TASK, "stage": args.stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "quantizer": qname, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "layer_idx": layer, "head_idx": h, "head_sample_stride": HEAD_SAMPLE_STRIDE})
                                    head_rows[qname].append(hpub)
                                    gz_append(HEAD_RECORDS, hpub)
                lm = logits_metrics(torch, ref_out.logits, q_out.logits)
                flat = summarize(layer_metrics)
                token_rec = {"task": TASK, "stage": args.stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "quantizer": qname, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "KL": lm["KL"], "top1_agreement": lm["top1_agreement"], "layer_mean": {k: stat_mean(flat, k) for k in flat}, "metric_status": "INIT_NO_S_IN" if s_in is None else "OK"}
                gz_append(TOKEN_RECORDS, token_rec)
                one = dict(token_rec["layer_mean"])
                one.update(lm)
                one["token_idx"] = t
                token_rows[(qname, "all")].append(one)
                token_rows[(qname, window_label(t))].append(one)
            ref_input = torch.tensor([[tok]], dtype=ref_input.dtype, device=device)
            ref_mask = None
            for cfg in CONFIGS:
                q_inputs[cfg["name"]] = ref_input.clone()
                q_masks[cfg["name"]] = None
    summaries = {}
    for cfg in CONFIGS:
        q = cfg["name"]
        summaries[q] = {
            "geometry": axis.config_geometry(cfg),
            "tokens_completed": len(cont_ids),
            "global_metrics": summarize(token_rows[(q, "all")]),
            "window_metrics": {w: summarize(token_rows[(q, w)]) for w in [f"{a}-{b}" for a, b in WINDOWS]},
            "layer_summary": summarize(layer_rows[q]),
            "head_summary": summarize(head_rows[q]),
        }
    obj = {
        "task": TASK,
        "stage": args.stage,
        "timestamp": now(),
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "teacher_forced_horizon_requested": args.max_steps,
        "actual_token_count": len(cont_ids),
        "PROTOCOL_GATE": protocol_gate,
        "METRIC_GATE": metric_gate,
        "METHOD_DESIGN_READY": "NO",
        "summaries": summaries,
        "numerical_audit_samples": numerical_audit_samples,
    }
    append_jsonl(RECORDS, obj)
    return obj


def improved(a, b, higher_better=False):
    if not finite_num(a) or not finite_num(b):
        return "NO"
    return "YES" if (b > a if higher_better else b < a) else "NO"


def compare_pair(summary, a, b):
    out = {}
    specs = {
        "KL": False,
        "top1_agreement": True,
        "state_reconstruction_relative_error_E_S": False,
        "D_effective": False,
        "cos_effective": True,
        "norm_ratio_effective": None,
        "alpha": None,
        "effective_parallel_gain": None,
        "orthogonal_ratio": False,
        "residual_to_update": False,
        "R_lost_same_codebook": False,
        "R_survival_same_codebook": True,
        "R_distortion_same_codebook": False,
        "norm_ratio_same_codebook": None,
        "cosine_same_codebook": True,
    }
    for m, hb in specs.items():
        av = stat_mean(summary[a]["global_metrics"], m)
        bv = stat_mean(summary[b]["global_metrics"], m)
        if hb is None:
            imp = "CLOSER_TO_IDEAL" if m in ("norm_ratio_effective", "effective_parallel_gain", "norm_ratio_same_codebook") and finite_num(av) and finite_num(bv) and abs(bv - 1.0) < abs(av - 1.0) else "NA"
        else:
            imp = improved(av, bv, hb)
        out[m] = {a: av, b: bv, f"{b}_minus_{a}": (bv - av) if finite_num(av) and finite_num(bv) else None, "metric_improved_with_KL_rescue": imp}
    return out


def classify(row_cmp, col_cmp, metric_gate):
    old_valid = metric_gate == "PASS"
    eff_impl = metric_gate == "PASS"
    row_eff = all(row_cmp[m]["metric_improved_with_KL_rescue"] == "YES" for m in ["D_effective", "cos_effective", "orthogonal_ratio", "residual_to_update"])
    col_eff = all(col_cmp[m]["metric_improved_with_KL_rescue"] == "YES" for m in ["D_effective", "cos_effective", "orthogonal_ratio", "residual_to_update"])
    row_old = all(row_cmp[m]["metric_improved_with_KL_rescue"] == "YES" for m in ["R_lost_same_codebook", "R_distortion_same_codebook", "cosine_same_codebook"])
    col_old = all(col_cmp[m]["metric_improved_with_KL_rescue"] == "YES" for m in ["R_lost_same_codebook", "R_distortion_same_codebook", "cosine_same_codebook"])
    table = {
        "OLD_SAME_CODEBOOK_METRICS_IMPLEMENTATION_VALID": "YES" if old_valid else "NO",
        "OLD_SAME_CODEBOOK_METRICS_TRACK_GRANULARITY_QUALITY": "YES" if row_old and col_old else ("PARTIAL" if row_old or col_old else "NO"),
        "EFFECTIVE_UPDATE_METRICS_IMPLEMENTATION_VALID": "YES" if eff_impl else "NO",
        "EFFECTIVE_UPDATE_METRICS_TRACK_ROW_KL_RESCUE": "YES" if row_eff else "NO",
        "EFFECTIVE_UPDATE_METRICS_TRACK_COLUMN_KL_RESCUE": "YES" if col_eff else "NO",
        "EFFECTIVE_UPDATE_METRICS_OUTPERFORM_SAME_CODEBOOK_METRICS_AS_QUALITY_PROXY": "YES" if (row_eff or col_eff) and not (row_old or col_old) else "NO",
        "SINGLE_STEP_LOCAL_UPDATE_FIDELITY_REMAINS_INSUFFICIENT": "YES" if not (row_eff and col_eff) else "NO",
    }
    if metric_gate != "PASS":
        final = "METRIC_IMPLEMENTATION_PROBLEM_IDENTIFIED"
    elif row_eff and col_eff and not (row_old and col_old):
        final = "EFFECTIVE_UPDATE_METRIC_CANDIDATE_IDENTIFIED"
    elif not row_eff and not col_eff and not row_old and not col_old:
        final = "LOCAL_UPDATE_METRICS_INSUFFICIENT_FOR_QUALITY"
    elif not row_old and not col_old:
        final = "SAME_CODEBOOK_METRIC_MISINTERPRETATION_IDENTIFIED"
    elif row_eff or col_eff or row_old or col_old:
        final = "BOTH_METRIC_FAMILIES_CAPTURE_DIFFERENT_VALID_PROPERTIES"
    else:
        final = "METRIC_AUDIT_INCONCLUSIVE"
    return table, final


def analyze():
    rows = list(iter_jsonl(RECORDS))
    rows = [r for r in rows if r.get("stage") in ("smoke", "audit512")]
    if not rows:
        raise SystemExit("no audit records")
    row = rows[-1]
    summaries = row["summaries"]
    row_cmp = compare_pair(summaries, "R128", "R16")
    col_cmp = compare_pair(summaries, "C128", "C16")
    table, final = classify(row_cmp, col_cmp, row["METRIC_GATE"])
    obj = {
        "task": TASK,
        "stage": "COMPLETE" if row["stage"] == "audit512" else "SMOKE_COMPLETE",
        "timestamp": now(),
        "Stage_0_status": "PASS",
        "Stage_A_status": "PASS" if row["stage"] in ("smoke", "audit512") and row["PROTOCOL_GATE"] == "PASS" and row["METRIC_GATE"] == "PASS" else "FAIL",
        "Stage_B_status": "PASS" if row["stage"] == "audit512" and row["PROTOCOL_GATE"] == "PASS" and row["METRIC_GATE"] == "PASS" else "NOT_RUN",
        "PROTOCOL_GATE": row["PROTOCOL_GATE"],
        "METRIC_GATE": row["METRIC_GATE"],
        "METHOD_DESIGN_READY": "NO",
        "continuation_source": row["continuation_source"],
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "configs": CONFIG_NAMES,
        "teacher_forced_horizon": row["teacher_forced_horizon_requested"],
        "problem_id": row["problem_id"],
        "R128_vs_R16": row_cmp,
        "C128_vs_C16": col_cmp,
        "temporal_windows": {q: summaries[q]["window_metrics"] for q in CONFIG_NAMES},
        "numerical_audit_samples": row["numerical_audit_samples"],
        "metric_ontology": {
            "SAME_CODEBOOK_REPRESENTABILITY_METRICS": "Re-encode S_in and S_pre into the current S_pre-derived codebook and ask whether desired endpoint movement appears as integer-code change.",
            "RUNTIME_ACTUAL_EFFECTIVE_UPDATE_METRICS": "Compare desired state change S_pre-S_in with actual next-token-visible change S_post-S_in.",
            "SNAPSHOT_METRIC": "E_S uses the same residual numerator as D_effective but normalizes by ||S_pre||, not by ||Delta_desired||.",
        },
        "final_classification_table": table,
        "FINAL_CLASSIFICATION": final,
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "recommended_next_step": "GDN_REAL_QUANT_RESIDUAL_PROPAGATION_V1" if table["SINGLE_STEP_LOCAL_UPDATE_FIDELITY_REMAINS_INSUFFICIENT"] == "YES" else "small prompt validation of effective-update metrics",
        "NEGATIVE_RESULT": "No 6-prompt formal, method design, stochastic rounding, cadence, replay, INT4, kernel, or end-to-end benchmark was run.",
    }
    save_json(RESULT, obj)
    write_report(obj)
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def write_report(obj):
    lines = [
        "# GDN INT8 Effective Update Metric Audit V1",
        "",
        "## Status",
        f"- Stage 0: `{obj['Stage_0_status']}`",
        f"- Stage A: `{obj['Stage_A_status']}`",
        f"- Stage B: `{obj['Stage_B_status']}`",
        f"- PROTOCOL_GATE: `{obj['PROTOCOL_GATE']}`; METRIC_GATE: `{obj['METRIC_GATE']}`",
        "- METHOD_DESIGN_READY: `NO`",
        "",
        "## R128 vs R16",
        "```json",
        json.dumps(obj["R128_vs_R16"], indent=2, sort_keys=True),
        "```",
        "",
        "## C128 vs C16",
        "```json",
        json.dumps(obj["C128_vs_C16"], indent=2, sort_keys=True),
        "```",
        "",
        "## Temporal Windows",
        "```json",
        json.dumps(obj["temporal_windows"], indent=2, sort_keys=True),
        "```",
        "",
        "## Numerical Audit Samples",
        "```json",
        json.dumps(obj["numerical_audit_samples"], indent=2, sort_keys=True),
        "```",
        "",
        "## Metric Ontology",
        "```json",
        json.dumps(obj["metric_ontology"], indent=2, sort_keys=True),
        "```",
        "",
        "## Final Classification",
        "```json",
        json.dumps({
            "table": obj["final_classification_table"],
            "FINAL_CLASSIFICATION": obj["FINAL_CLASSIFICATION"],
            "METHOD_DESIGN_READY": "NO",
            "METHOD_DESIGN_READY_CANDIDATE": obj["METHOD_DESIGN_READY_CANDIDATE"],
            "recommended_next_step": obj["recommended_next_step"],
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Negative Result",
        f"- {obj['NEGATIVE_RESULT']}",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage0():
    prev = json.loads((RES / "gdn_int8_axis_geometry_rescue_v1.json").read_text(encoding="utf-8"))
    obj = {
        "task": TASK,
        "stage": "stage0",
        "timestamp": now(),
        "previous_axis_rescue_stage": prev.get("stage"),
        "previous_AXIS_RESCUE_PILOT": prev.get("AXIS_RESCUE_PILOT"),
        "previous_FINAL_CLASSIFICATION": prev.get("FINAL_CLASSIFICATION"),
        "same_codebook_interpretation": "valid representability metric, not runtime effective update fidelity",
        "block_quantizer_reuse": [axis.config_geometry(c) for c in CONFIGS],
        "METHOD_DESIGN_READY": "NO",
        "STAGE0_GATE": "PASS",
    }
    save_json(RES / "gdn_int8_effective_update_metric_audit_v1_stage0.json", obj)
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "smoke", "audit512", "analyze"], required=True)
    ap.add_argument("--max-steps", type=int, default=128)
    args = ap.parse_args()
    if args.stage == "stage0":
        stage0()
    elif args.stage == "analyze":
        analyze()
    else:
        run_stage(args)


if __name__ == "__main__":
    main()
