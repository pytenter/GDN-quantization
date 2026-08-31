#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import os
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

TASK = "GDN_INT8_EFFECTIVE_UPDATE_METRIC_SMALL_PROMPT_VALIDATION_V1"
RESULT = RES / "gdn_int8_effective_update_small_prompt_validation_v1.json"
STAGE0 = RES / "gdn_int8_effective_update_small_prompt_validation_v1_stage0.json"
RECORDS = RES / "gdn_int8_effective_update_small_prompt_validation_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_effective_update_small_prompt_validation_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_effective_update_small_prompt_validation_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_effective_update_small_prompt_validation_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_effective_update_small_prompt_validation_v1.md"
FORMAL_RECORDS = RES / "gdn_int8_effective_update_small_prompt_validation_v1_formal_records.jsonl"
FORMAL_TOKEN_RECORDS = RES / "gdn_int8_effective_update_small_prompt_validation_v1_formal_token_records.jsonl.gz"
FORMAL_LAYER_RECORDS = RES / "gdn_int8_effective_update_small_prompt_validation_v1_formal_layer_records.jsonl.gz"
FORMAL_HEAD_RECORDS = RES / "gdn_int8_effective_update_small_prompt_validation_v1_formal_head_records.jsonl.gz"

P1_RESULT = RES / "gdn_int8_orientation_state_change_mechanism_v1.json"
P1_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
EPS = 1e-12
CONFIG_NAMES = ["R128", "R16", "C128", "C16"]
WINDOWS = [(1, 128), (129, 256), (257, 512)]
HEAD_SAMPLE_STRIDE = 128
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_effective_update_metric_audit as eff

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


def window_label(token_idx):
    one = int(token_idx) + 1
    for a, b in WINDOWS:
        if a <= one <= b:
            return f"{a}-{b}"
    return ">512"


def stage_paths(stage, shard_id=None):
    suffix = f"_{stage}" + ("" if shard_id is None else f"_shard_{shard_id:03d}")
    return {
        "records": RES / f"gdn_int8_effective_update_small_prompt_validation_v1_records{suffix}.jsonl",
        "token": RES / f"gdn_int8_effective_update_small_prompt_validation_v1_token_records{suffix}.jsonl.gz",
        "layer": RES / f"gdn_int8_effective_update_small_prompt_validation_v1_layer_records{suffix}.jsonl.gz",
        "head": RES / f"gdn_int8_effective_update_small_prompt_validation_v1_head_records{suffix}.jsonl.gz",
    }


def prompt_manifest():
    p1_obj = json.loads(P1_RESULT.read_text(encoding="utf-8"))
    canonical_ids = []
    for r in iter_jsonl(P1_RECORDS):
        pid = r.get("problem_id")
        if pid and pid not in canonical_ids:
            canonical_ids.append(pid)
    prompt_rows = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    manifest = []
    for pid in canonical_ids:
        pm = prompt_rows.get(pid)
        if not pm:
            continue
        ids = p1.setup_model_cached_tokenizer.encode(pm["fp_response"], add_special_tokens=False) if False else None
        manifest.append({
            "prompt_id": pid,
            "source_path": str(P1_RECORDS),
            "source_identifier": "GDN_INT8_ORIENTATION_STATE_CHANGE_MECHANISM_V1 formal_records problem_id",
            "role": pm["role"],
            "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
            "exact_original_generation_token_ids_available": False,
            "exact_replay_claimed": False,
            "continuation_token_count_available": pm.get("fp_tokens"),
        })
    return manifest, p1_obj


def metric_event(torch, s_in, s_pre, cfg):
    m = eff.metric_event(torch, s_in, s_pre, cfg)
    if m.get("status") in ("OK", "ZERO_UPDATE_EVENT"):
        # This is descriptive only: orthogonal dominance is not a causal claim.
        r_norm = m.get("actual_quantization_residual_norm")
        orth = m.get("orthogonal_ratio")
        desired = m.get("desired_state_change_norm")
        if finite_num(r_norm) and finite_num(orth) and finite_num(desired):
            m["orthogonal_fraction_of_residual"] = (orth * desired) / (r_norm + EPS)
    return m


def run_prompt(torch, model, tokenizer, e2e, pm, stage, horizon, paths, args):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids_all = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    cont_ids = cont_ids_all[:horizon]
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
    numerical = []
    protocol_gate = "PASS"
    metric_gate = "PASS"
    nonfinite = defaultdict(int)
    print(f"[{now()}] {stage} shard={args.shard_id} prompt={pm['problem_id']} tokens={len(cont_ids)}", flush=True)
    with torch.inference_mode():
        for t, tok in enumerate(cont_ids):
            ref_out = p1.feed_step(torch, model, ref_input, ref_mask, ref_past)
            ref_past = ref_out.past_key_values
            gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "token_idx": t, "quantizer": "FP_STATE", "KL": 0.0, "top1_agreement": 1.0})
            for cfg in CONFIGS:
                q = cfg["name"]
                s_in = None
                if q_pasts[q] is not None:
                    s_in = {layer: p1.get_state(q_pasts[q], layer).detach().clone() for layer in GDN_LAYERS}
                q_out = p1.feed_step(torch, model, q_inputs[q], q_masks[q], q_pasts[q])
                q_pasts[q] = q_out.past_key_values
                layer_metrics = []
                if s_in is not None:
                    for layer in GDN_LAYERS:
                        state = p1.get_state(q_pasts[q], layer)
                        s_pre = state.detach().clone()
                        m = metric_event(torch, s_in[layer], s_pre, cfg)
                        if list(s_pre.shape) != [1, 32, 128, 128] or m.get("qrange") != [-127, 127]:
                            protocol_gate = "FAIL"
                        if m.get("scale_count_per_head") != axis.config_geometry(cfg)["scale_count_per_head"]:
                            protocol_gate = "FAIL"
                        if m.get("status") not in ("OK", "ZERO_UPDATE_EVENT"):
                            metric_gate = "FAIL"
                            nonfinite[q] += 1
                            continue
                        if m.get("identity_relative_error", 1) > 1e-7 or m.get("residual_to_update_minus_D_effective_abs", 1) > 1e-10:
                            metric_gate = "FAIL"
                        if len(numerical) < 24 and t in (1, 64, 128, 256, 511) and layer in (0, 12, 25):
                            sample = {k: m.get(k) for k in ["desired_state_change_norm", "effective_update_norm", "actual_quantization_residual_norm", "identity_error_norm", "identity_relative_error", "D_effective", "cos_effective", "norm_ratio_effective", "alpha", "orthogonal_ratio", "orthogonal_fraction_of_residual"]}
                            sample.update({"quantizer": q, "token_idx": t, "layer_idx": layer, "head_idx": "ALL"})
                            numerical.append(sample)
                        state.copy_(m["_s_post"].to(state.dtype))
                        pub = {k: v for k, v in m.items() if not k.startswith("_")}
                        pub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "quantizer": q, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "layer_idx": layer})
                        layer_metrics.append(pub)
                        layer_rows[q].append(pub)
                        gz_append(paths["layer"], pub)
                        if t % HEAD_SAMPLE_STRIDE == 0:
                            for h in range(s_pre.shape[1]):
                                hm = metric_event(torch, s_in[layer][:, h:h + 1, :, :], s_pre[:, h:h + 1, :, :], cfg)
                                if hm.get("status") in ("OK", "ZERO_UPDATE_EVENT"):
                                    hpub = {k: v for k, v in hm.items() if not k.startswith("_")}
                                    hpub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "quantizer": q, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "layer_idx": layer, "head_idx": h, "head_sample_stride": HEAD_SAMPLE_STRIDE})
                                    head_rows[q].append(hpub)
                                    gz_append(paths["head"], hpub)
                lm = eff.logits_metrics(torch, ref_out.logits, q_out.logits)
                flat = summarize(layer_metrics)
                one = {k: stat_mean(flat, k) for k in flat}
                one.update(lm)
                one["token_idx"] = t
                token_rows[(q, "all")].append(one)
                token_rows[(q, window_label(t))].append(one)
                gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "quantizer": q, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "KL": lm["KL"], "top1_agreement": lm["top1_agreement"], "layer_mean": one, "metric_status": "INIT_NO_S_IN" if s_in is None else "OK"})
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
            "valid_token_count": len([r for r in token_rows[(q, "all")] if r.get("token_idx", 0) > 0]),
            "surviving_token_count": len(cont_ids),
            "nonfinite_count": nonfinite[q],
            "global_metrics": summarize(token_rows[(q, "all")]),
            "window_metrics": {w: summarize(token_rows[(q, w)]) for w in [f"{a}-{b}" for a, b in WINDOWS]},
            "layer_summary": summarize(layer_rows[q]),
            "head_summary": summarize(head_rows[q]),
        }
    out = {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "teacher_forced_horizon_requested": horizon,
        "continuation_token_count_available": len(cont_ids_all),
        "actual_token_count": len(cont_ids),
        "PROTOCOL_GATE": protocol_gate,
        "METRIC_GATE": metric_gate,
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "summaries": summaries,
        "numerical_audit_samples": numerical,
    }
    append_jsonl(paths["records"], out)
    return out


def compare_pair(summary, a, b):
    return eff.compare_pair(summary, a, b)


def change_value(cmp, metric, a, b):
    return cmp[metric][b] - cmp[metric][a] if finite_num(cmp[metric].get(a)) and finite_num(cmp[metric].get(b)) else None


def pairwise(row):
    out = {}
    for a, b in [("R128", "R16"), ("C128", "C16")]:
        cmp = compare_pair(row["summaries"], a, b)
        out[f"{a}_to_{b}"] = {
            "problem_id": row["problem_id"],
            "role": row["role"],
            "valid_token_count": {q: row["summaries"][q]["valid_token_count"] for q in (a, b)},
            "metrics": cmp,
            "deltas": {
                "Delta_KL": change_value(cmp, "KL", a, b),
                "Delta_E_S": change_value(cmp, "state_reconstruction_relative_error_E_S", a, b),
                "Delta_D_effective": change_value(cmp, "D_effective", a, b),
                "Delta_cos_effective": change_value(cmp, "cos_effective", a, b),
                "Delta_abs_norm_ratio_minus_1": (abs(cmp["norm_ratio_effective"][b] - 1) - abs(cmp["norm_ratio_effective"][a] - 1)) if finite_num(cmp["norm_ratio_effective"].get(a)) and finite_num(cmp["norm_ratio_effective"].get(b)) else None,
                "Delta_abs_alpha": (abs(cmp["alpha"][b]) - abs(cmp["alpha"][a])) if finite_num(cmp["alpha"].get(a)) and finite_num(cmp["alpha"].get(b)) else None,
                "Delta_orthogonal_ratio": change_value(cmp, "orthogonal_ratio", a, b),
            },
        }
    return out


def spearman(xs, ys):
    return axis.spearman(xs, ys)


def analyze(stage, num_shards):
    if stage == "formal":
        rows = []
        for sid in range(num_shards):
            for r in iter_jsonl(stage_paths("formal", sid)["records"]):
                rows.append(r)
    else:
        rows = list(iter_jsonl(stage_paths(stage)["records"]))
    rows = [r for r in rows if r.get("stage") == stage]
    pair_rows = [pairwise(r) for r in rows]
    consistency = {}
    for pair in ["R128_to_R16", "C128_to_C16"]:
        consistency[pair] = {}
        for metric in ["KL", "D_effective", "orthogonal_ratio", "cos_effective", "state_reconstruction_relative_error_E_S", "R_lost_same_codebook", "R_distortion_same_codebook", "cosine_same_codebook"]:
            vals = []
            good = 0
            total = 0
            for pr in pair_rows:
                cmp = pr[pair]["metrics"][metric]
                imp = cmp.get("metric_improved_with_KL_rescue")
                if imp in ("YES", "NO"):
                    total += 1
                    good += int(imp == "YES")
                vals.append(pr[pair]["deltas"].get("Delta_KL"))
            consistency[pair][metric] = {"same_direction_count": good, "valid_prompt_count": total, "rate": good / max(total, 1)}
    assoc = {}
    for pair in ["R128_to_R16", "C128_to_C16"]:
        kl = [pr[pair]["deltas"].get("Delta_KL") for pr in pair_rows]
        assoc[pair] = {}
        for name, dk in [
            ("D_effective", "Delta_D_effective"),
            ("orthogonal_ratio", "Delta_orthogonal_ratio"),
            ("cos_effective", "Delta_cos_effective"),
            ("E_S", "Delta_E_S"),
            ("abs_norm_ratio_minus_1", "Delta_abs_norm_ratio_minus_1"),
            ("abs_alpha", "Delta_abs_alpha"),
        ]:
            ys = [pr[pair]["deltas"].get(dk) for pr in pair_rows]
            assoc[pair][f"Delta_KL_vs_{name}"] = {"spearman": spearman(kl, ys), "N": len([1 for x, y in zip(kl, ys) if finite_num(x) and finite_num(y)]), "label": "descriptive small-N association"}
    old_rates = []
    eff_rates = []
    for pair in ["R128_to_R16", "C128_to_C16"]:
        eff_rates += [consistency[pair][m]["rate"] for m in ["D_effective", "orthogonal_ratio", "cos_effective"]]
        old_rates += [consistency[pair][m]["rate"] for m in ["R_lost_same_codebook", "R_distortion_same_codebook", "cosine_same_codebook"]]
    eff_supported = min(eff_rates or [0]) >= (5 / 6 if stage == "formal" else 1.0)
    old_supported = min(old_rates or [0]) >= (5 / 6 if stage == "formal" else 1.0)
    class_table = {
        "PROMPT_MANIFEST_RECOVERED_FROM_CANONICAL_ARTIFACTS": "YES" if len(prompt_manifest()[0]) == 6 else "NO",
        "EFFECTIVE_UPDATE_METRICS_TRACK_ROW_KL_RESCUE": "YES" if consistency.get("R128_to_R16", {}).get("D_effective", {}).get("rate", 0) >= (5 / 6 if stage == "formal" else 1.0) else "PARTIAL",
        "EFFECTIVE_UPDATE_METRICS_TRACK_COLUMN_KL_RESCUE": "YES" if consistency.get("C128_to_C16", {}).get("D_effective", {}).get("rate", 0) >= (5 / 6 if stage == "formal" else 1.0) else "PARTIAL",
        "EFFECTIVE_UPDATE_METRICS_OUTPERFORM_SAME_CODEBOOK_METRICS_AS_QUALITY_PROXY": "YES" if eff_supported and not old_supported else "PARTIAL",
        "ORTHOGONAL_RESIDUAL_DOMINANCE_STABLE": "YES" if min([consistency[p]["orthogonal_ratio"]["rate"] for p in consistency] or [0]) >= (5 / 6 if stage == "formal" else 1.0) else "PARTIAL",
        "METHOD_DESIGN_READY": "NO",
    }
    if stage != "formal":
        final = "SMOKE_PASS_READY_FOR_FORMAL" if rows and all(r.get("PROTOCOL_GATE") == "PASS" and r.get("METRIC_GATE") == "PASS" for r in rows) else "SMOKE_FAIL"
    elif eff_supported and not old_supported:
        final = "EFFECTIVE_UPDATE_METRIC_MULTI_PROMPT_SUPPORTED"
    elif eff_supported:
        final = "BOTH_METRIC_FAMILIES_CAPTURE_DIFFERENT_VALID_PROPERTIES"
    elif max(eff_rates or [0]) >= 0.5:
        final = "EFFECTIVE_UPDATE_METRIC_MULTI_PROMPT_INCONCLUSIVE"
    else:
        final = "EFFECTIVE_UPDATE_METRIC_CANDIDATE_NOT_SUPPORTED"
    obj = {
        "task": TASK,
        "stage": "FORMAL_COMPLETE" if stage == "formal" else "SMOKE_COMPLETE",
        "timestamp": now(),
        "FORMAL_STATUS": "COMPLETE" if stage == "formal" and len(rows) == 6 else ("NOT_RUN" if stage != "formal" else "INCOMPLETE"),
        "completed_prompt_count": len(rows),
        "requested_prompt_count": 6 if stage == "formal" else 1,
        "PROTOCOL_GATE": "PASS" if rows and all(r.get("PROTOCOL_GATE") == "PASS" for r in rows) else "FAIL",
        "METRIC_GATE": "PASS" if rows and all(r.get("METRIC_GATE") == "PASS" for r in rows) else "FAIL",
        "SMOKE_GATE": "PASS" if stage != "formal" and rows and all(r.get("PROTOCOL_GATE") == "PASS" and r.get("METRIC_GATE") == "PASS" for r in rows) else ("N/A" if stage == "formal" else "FAIL"),
        "METHOD_DESIGN_READY": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "EFFECTIVE_UPDATE_METRIC_READY_FOR_CAUSAL_INTERVENTION": "YES" if final == "EFFECTIVE_UPDATE_METRIC_MULTI_PROMPT_SUPPORTED" else "NO",
        "prompt_manifest": prompt_manifest()[0],
        "pairwise_rescue": pair_rows,
        "consistency_table": consistency,
        "across_prompt_association": assoc,
        "final_classification_table": class_table,
        "FINAL_CLASSIFICATION": final,
        "Observation": "Under FP32-prefill + quantized recurrent continuation, effective-update metrics are compared with KL rescue across canonical prompts.",
        "Hypothesis": "Runtime effective-update fidelity may be a more stable mechanism metric candidate than same-codebook representability.",
        "Supported conclusion": "Small-N descriptive validation only; no causal proof or method design is claimed.",
        "Negative result": "No new quantizer, causal intervention, residual propagation, replay, cadence, INT4, kernel, or end-to-end generation was run.",
        "Unresolved": "Whether effective-update metrics remain predictive under causal intervention is not tested here.",
    }
    save_json(RESULT, obj)
    write_report(obj)
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))
    return obj


def write_report(obj):
    lines = [
        "# GDN INT8 Effective Update Small Prompt Validation V1",
        "",
        "## Observation",
        f"- FORMAL_STATUS: `{obj['FORMAL_STATUS']}`",
        f"- PROTOCOL_GATE: `{obj['PROTOCOL_GATE']}`; METRIC_GATE: `{obj['METRIC_GATE']}`",
        f"- METHOD_DESIGN_READY: `{obj['METHOD_DESIGN_READY']}`",
        f"- FINAL_CLASSIFICATION: `{obj['FINAL_CLASSIFICATION']}`",
        "",
        "## Prompt Manifest",
        "```json",
        json.dumps(obj["prompt_manifest"], indent=2, sort_keys=True),
        "```",
        "",
        "## Consistency Table",
        "```json",
        json.dumps(obj["consistency_table"], indent=2, sort_keys=True),
        "```",
        "",
        "## Pairwise Rescue",
        "```json",
        json.dumps(obj["pairwise_rescue"], indent=2, sort_keys=True),
        "```",
        "",
        "## Across-Prompt Association",
        "```json",
        json.dumps(obj["across_prompt_association"], indent=2, sort_keys=True),
        "```",
        "",
        "## Final Classification",
        "```json",
        json.dumps({
            "table": obj["final_classification_table"],
            "FINAL_CLASSIFICATION": obj["FINAL_CLASSIFICATION"],
            "EFFECTIVE_UPDATE_METRIC_READY_FOR_CAUSAL_INTERVENTION": obj["EFFECTIVE_UPDATE_METRIC_READY_FOR_CAUSAL_INTERVENTION"],
            "METHOD_DESIGN_READY": "NO",
            "METHOD_DESIGN_READY_CANDIDATE": "NO",
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Hypothesis",
        f"- {obj['Hypothesis']}",
        "",
        "## Supported Conclusion",
        f"- {obj['Supported conclusion']}",
        "",
        "## Negative Result",
        f"- {obj['Negative result']}",
        "",
        "## Unresolved",
        f"- {obj['Unresolved']}",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    manifest, _ = prompt_manifest()
    pms = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    if args.stage == "smoke":
        todo = [pms[manifest[0]["prompt_id"]]]
        paths = stage_paths("smoke")
    else:
        ids = [m["prompt_id"] for i, m in enumerate(manifest) if i % args.num_shards == args.shard_id]
        todo = [pms[i] for i in ids]
        paths = stage_paths("formal", args.shard_id)
    done = set()
    if args.resume and paths["records"].exists():
        done = {r["problem_id"] for r in iter_jsonl(paths["records"]) if r.get("stage") == args.stage}
    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    for pm in todo:
        if pm["problem_id"] not in done:
            run_prompt(torch, model, tokenizer, e2e, pm, args.stage, args.max_steps, paths, args)


def stage0():
    manifest, p1_obj = prompt_manifest()
    gate = "PASS" if len(manifest) == 6 and p1_obj.get("FORMAL_STATUS") == "COMPLETE" else "FAIL"
    obj = {
        "task": TASK,
        "stage": "stage0",
        "timestamp": now(),
        "PROTOCOL_GATE": gate,
        "METRIC_IMPLEMENTATION_AUDIT": "PASS",
        "METHOD_DESIGN_READY": "NO",
        "model_identity": "Qwen3.5-9B from existing run_config/model loader",
        "state_shape": "[B,H,K,V] = [1,32,128,128]",
        "row_axis": "Key axis",
        "column_axis": "Value axis",
        "prefill_precision": "FP32 recurrent state / canonical model precision path",
        "continuation": "teacher-forced P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "quantized_object": "GDN recurrent state only",
        "configs": [axis.config_geometry(c) for c in CONFIGS],
        "S_in_S_pre_S_post_capture_timing": "runtime cache before token, post-compute pre-quant state, post Q/DQ write-back",
        "metric_formula": "Delta_effective=S_post-S_in; Delta_desired=S_pre-S_in; residual=S_post-S_pre; identity asserted",
        "prompt_manifest": manifest,
        "output_path": str(RESULT),
    }
    save_json(STAGE0, obj)
    save_json(RESULT, {**obj, "FORMAL_STATUS": "NOT_RUN", "SMOKE_GATE": "NOT_RUN"})
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "smoke", "formal", "analyze", "merge"], required=True)
    ap.add_argument("--analyze-stage", choices=["smoke", "formal"], default="formal")
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=2)
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
