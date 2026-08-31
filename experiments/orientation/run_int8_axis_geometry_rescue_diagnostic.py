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

TASK = "GDN_INT8_AXIS_GEOMETRY_RESCUE_CAUSAL_DIAGNOSTIC_V1"
RESULT = RES / "gdn_int8_axis_geometry_rescue_v1.json"
RECORDS = RES / "gdn_int8_axis_geometry_rescue_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_axis_geometry_rescue_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_axis_geometry_rescue_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_axis_geometry_rescue_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_axis_geometry_rescue_v1.md"
LOG_DIR = RES / "gdn_int8_axis_geometry_rescue_v1_logs"

EPS = 1e-12
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
GROUP_SIZES = [128, 64, 32, 16]
CONFIGS = [{"name": f"R{g}", "orientation": "row", "group_size": g} for g in GROUP_SIZES] + [
    {"name": f"C{g}", "orientation": "column", "group_size": g} for g in GROUP_SIZES
]
WINDOWS_1024 = [(1, 128), (129, 256), (257, 512), (513, 1024)]
HEAD_SAMPLE_STRIDE = 64
METRIC_KEYS = [
    "state_reconstruction_relative_error_E_S",
    "state_change_error_E_Delta",
    "lost_update_fraction",
    "survival_fraction",
    "update_distortion_ratio",
    "represented_update_norm_ratio",
    "update_cosine",
    "residual_update_cosine",
    "residual_update_projection_coeff",
    "state_norm",
    "delta_norm",
    "scale_mean",
    "scale_median",
    "scale_max",
    "scale_p95_over_median",
    "within_group_dynamic_range_p95",
    "saturation_fraction",
]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


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
    op = gzip.open if str(path).endswith(".gz") else open
    if not Path(path).exists():
        return
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def cosine(torch, a, b):
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    an = float(torch.linalg.vector_norm(af).item())
    bn = float(torch.linalg.vector_norm(bf).item())
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item())


def config_geometry(cfg):
    g = cfg["group_size"]
    return {
        "config": cfg["name"],
        "orientation": cfg["orientation"],
        "group_size": g,
        "bit_width": 8,
        "qrange": [-127, 127],
        "zero_point": 0,
        "rounding": "torch.round",
        "scale_count_per_head": 128 * (128 // g),
        "scale_count_per_layer": 32 * 128 * (128 // g),
        "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    }


def grouped_quant(torch, x, scale_from, cfg):
    qmax = 127
    g = int(cfg["group_size"])
    b, h, k, v = x.shape
    if k != 128 or v != 128 or 128 % g != 0:
        raise ValueError(f"unexpected state shape/group: {list(x.shape)} G={g}")
    if cfg["orientation"] == "row":
        sx = scale_from.reshape(b, h, k, v // g, g)
        xx = x.reshape(b, h, k, v // g, g)
        scale = sx.detach().abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / qmax
        q = torch.round(xx / scale).clamp(-qmax, qmax)
        y = (q * scale).reshape_as(x)
        q_full = q.reshape_as(x)
        scale_full = scale.repeat_interleave(g, dim=-1).reshape_as(x)
        scale_shape = list(scale.shape)
        grouped_abs = sx.detach().abs().reshape(-1, g)
    elif cfg["orientation"] == "column":
        sx = scale_from.reshape(b, h, k // g, g, v)
        xx = x.reshape(b, h, k // g, g, v)
        scale = sx.detach().abs().amax(dim=3, keepdim=True).clamp_min(EPS) / qmax
        q = torch.round(xx / scale).clamp(-qmax, qmax)
        y = (q * scale).reshape_as(x)
        q_full = q.reshape_as(x)
        scale_full = scale.repeat_interleave(g, dim=3).reshape_as(x)
        scale_shape = list(scale.shape)
        grouped_abs = sx.detach().abs().permute(0, 1, 2, 4, 3).reshape(-1, g)
    else:
        raise ValueError(cfg["orientation"])
    gmax = grouped_abs.max(dim=1).values
    gmed = grouped_abs.median(dim=1).values.clamp_min(EPS)
    dyn = (gmax / gmed).detach().cpu().tolist()
    return y, q_full, scale_full, scale, scale_shape, dyn


def metric_event(torch, s_in, s_pre, cfg):
    pre = s_pre.detach().float()
    inn = s_in.detach().float()
    if not bool(torch.isfinite(pre).all().item() and torch.isfinite(inn).all().item()):
        return {"status": "NONFINITE_STATE"}
    s_post, q_pre, scale_full, scale, scale_shape, dyn = grouped_quant(torch, pre, pre, cfg)
    _, q_in, _, _, _, _ = grouped_quant(torch, inn, pre, cfg)
    if q_pre.shape != pre.shape or q_in.shape != pre.shape:
        return {"status": "BAD_Q_SHAPE", "scale_shape": scale_shape}
    delta = pre - inn
    delta_hat = (q_pre - q_in) * scale_full
    residual = s_post - pre
    d2 = float(torch.sum(delta.double() * delta.double()).item())
    lost = float(torch.sum(delta.double()[q_pre == q_in] ** 2).item()) if d2 > EPS else 0.0
    err = delta - delta_hat
    dh2 = float(torch.sum(delta_hat.double() * delta_hat.double()).item())
    scale_vals = scale.detach().float().flatten().cpu().tolist()
    return {
        "status": "OK" if d2 > EPS else "ZERO_UPDATE_EVENT",
        "state_reconstruction_relative_error_E_S": float(torch.linalg.vector_norm(residual).item()) / max(float(torch.linalg.vector_norm(pre).item()), EPS),
        "state_change_error_E_Delta": float(torch.linalg.vector_norm(err).item()) / max(float(torch.linalg.vector_norm(delta).item()), EPS) if d2 > EPS else None,
        "lost_update_fraction": lost / (d2 + EPS) if d2 > EPS else None,
        "survival_fraction": float((q_pre != q_in).float().mean().item()),
        "update_distortion_ratio": float(torch.sum(err.double() * err.double()).item()) / (d2 + EPS) if d2 > EPS else None,
        "represented_update_norm_ratio": math.sqrt(dh2) / max(math.sqrt(d2), EPS) if d2 > EPS else None,
        "update_cosine": cosine(torch, delta, delta_hat),
        "residual_update_cosine": cosine(torch, residual, delta),
        "residual_update_projection_coeff": float(torch.sum(residual.double() * delta.double()).item()) / (d2 + EPS) if d2 > EPS else None,
        "state_norm": float(torch.linalg.vector_norm(pre).item()),
        "delta_norm": math.sqrt(d2),
        "scale_mean": avg(scale_vals),
        "scale_median": statistics.median(scale_vals) if scale_vals else None,
        "scale_max": max(scale_vals) if scale_vals else None,
        "scale_p95": pct(scale_vals, 0.95),
        "scale_p95_over_median": ratio(pct(scale_vals, 0.95), statistics.median(scale_vals) if scale_vals else None),
        "within_group_dynamic_range_mean": avg(dyn),
        "within_group_dynamic_range_median": statistics.median(dyn) if dyn else None,
        "within_group_dynamic_range_p95": pct(dyn, 0.95),
        "saturation_fraction": float((q_pre.abs() >= 127).float().mean().item()),
        "zero_code_fraction": float((q_pre == 0).float().mean().item()),
        "qrange": [-127, 127],
        "scale_shape": scale_shape,
        "scale_count_per_head": 128 * (128 // cfg["group_size"]),
        "_s_post": s_post,
    }


def public(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def aggregate(rows):
    keys = sorted({k for r in rows for k, v in r.items() if finite_num(v)})
    return {k: avg([r.get(k) for r in rows]) for k in keys}


def logits_metrics(torch, ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    other_logp = torch.log_softmax(other, dim=-1)
    return {
        "KL": float((ref_logp.exp() * (ref_logp - other_logp)).sum().item()),
        "top1_agreement": int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item()),
    }


def stage_paths(stage, shard_id=None):
    suffix = f"_{stage}" + ("" if shard_id is None else f"_shard_{shard_id:03d}")
    return {
        "records": RES / f"gdn_int8_axis_geometry_rescue_v1_records{suffix}.jsonl",
        "token": RES / f"gdn_int8_axis_geometry_rescue_v1_token_records{suffix}.jsonl.gz",
        "layer": RES / f"gdn_int8_axis_geometry_rescue_v1_layer_records{suffix}.jsonl.gz",
        "head": RES / f"gdn_int8_axis_geometry_rescue_v1_head_records{suffix}.jsonl.gz",
    }


def window_label(token_idx, horizon):
    one = int(token_idx) + 1
    windows = WINDOWS_1024 if horizon > 512 else [(1, 128), (129, 256), (257, 512)]
    for a, b in windows:
        if a <= one <= b:
            return f"{a}-{b}"
    return f">{windows[-1][1]}"


def run_prompt(torch, model, tokenizer, e2e, pm, stage, horizon, paths, shard_id=0):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[:horizon]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    ref_mask = enc.get("attention_mask")
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    q_inputs = {c["name"]: ref_input.clone() for c in CONFIGS}
    q_masks = {c["name"]: ref_mask.clone() if ref_mask is not None else None for c in CONFIGS}
    ref_past = None
    q_pasts = {c["name"]: None for c in CONFIGS}
    token_agg = defaultdict(list)
    layer_agg = defaultdict(list)
    head_agg = defaultdict(list)
    protocol_gate = "PASS"
    metric_gate = "PASS"
    with torch.inference_mode():
        for t, tok in enumerate(cont_ids):
            ref_out = p1.feed_step(torch, model, ref_input, ref_mask, ref_past)
            ref_past = ref_out.past_key_values
            gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "token_idx": t, "quantizer": "FP_STATE", "KL": 0.0, "top1_agreement": 1.0})
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
                        if m.get("scale_count_per_head") != config_geometry(cfg)["scale_count_per_head"]:
                            protocol_gate = "FAIL"
                        if m.get("status") not in ("OK", "ZERO_UPDATE_EVENT"):
                            metric_gate = "FAIL"
                        state.copy_(m["_s_post"].to(state.dtype))
                        pub = public(m)
                        pub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t, horizon), "quantizer": qname, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "layer_idx": layer})
                        layer_metrics.append(pub)
                        layer_agg[(qname, layer)].append(pub)
                        gz_append(paths["layer"], pub)
                        if t % HEAD_SAMPLE_STRIDE == 0:
                            for h in range(s_pre.shape[1]):
                                hm = metric_event(torch, s_in[layer][:, h:h+1, :, :], s_pre[:, h:h+1, :, :], cfg)
                                hpub = public(hm)
                                hpub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t, horizon), "quantizer": qname, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "layer_idx": layer, "head_idx": h, "head_sample_stride": HEAD_SAMPLE_STRIDE})
                                head_agg[(qname, layer, h)].append(hpub)
                                gz_append(paths["head"], hpub)
                lm = logits_metrics(torch, ref_out.logits, q_out.logits)
                mean_layer = aggregate(layer_metrics)
                rec = {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t, horizon), "quantizer": qname, "orientation": cfg["orientation"], "group_size": cfg["group_size"], "KL": lm["KL"], "top1_agreement": lm["top1_agreement"], "layer_mean": mean_layer, "metric_status": "INIT_NO_S_IN" if s_in is None else "OK"}
                gz_append(paths["token"], rec)
                flat = dict(mean_layer)
                flat.update(lm)
                token_agg[(qname, "all")].append(flat)
                token_agg[(qname, window_label(t, horizon))].append(flat)
            ref_input = torch.tensor([[tok]], dtype=ref_input.dtype, device=device)
            ref_mask = None
            for cfg in CONFIGS:
                q_inputs[cfg["name"]] = ref_input.clone()
                q_masks[cfg["name"]] = None
    summaries = {}
    for cfg in CONFIGS:
        qname = cfg["name"]
        summaries[qname] = {
            "geometry": config_geometry(cfg),
            "tokens_completed": len(cont_ids),
            "global_metrics": summarize_rows(token_agg[(qname, "all")]),
            "window_metrics": {k[1]: summarize_rows(v) for k, v in token_agg.items() if k[0] == qname and k[1] != "all"},
        }
    out = {
        "task": TASK,
        "stage": stage,
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "teacher_forced_horizon_requested": horizon,
        "actual_token_count": len(cont_ids),
        "survived_full_horizon": len(cont_ids) >= horizon,
        "PROTOCOL_GATE": protocol_gate,
        "METRIC_GATE": metric_gate,
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timestamp": now(),
        "summaries": summaries,
        "layer_summary": {f"{k[0]}|{k[1]}": summarize_rows(v) for k, v in layer_agg.items()},
        "head_summary": {f"{k[0]}|{k[1]}|{k[2]}": summarize_rows(v) for k, v in head_agg.items()},
    }
    append_jsonl(paths["records"], out)
    append_jsonl(RECORDS, out)
    return out


def summarize_rows(rows):
    buckets = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if finite_num(v):
                buckets[k].append(v)
    return {k: {"mean": avg(v), "median": statistics.median(v), "p90": pct(v, 0.90), "p95": pct(v, 0.95)} for k, v in buckets.items()}


def stat_mean(stats, key):
    v = stats.get(key)
    return v.get("mean") if isinstance(v, dict) else None


def load_stage_rows(stage):
    rows = []
    for p in sorted(RES.glob(f"gdn_int8_axis_geometry_rescue_v1_records_{stage}*.jsonl")):
        for r in iter_jsonl(p):
            if r.get("stage") == stage:
                rows.append(r)
    by_key = {(r["problem_id"], r.get("gpu_id", ""), r.get("stage")): r for r in rows}
    return list(by_key.values())


def global_by_config(rows):
    out = {}
    for cfg in CONFIGS:
        q = cfg["name"]
        vals = []
        for r in rows:
            gm = r["summaries"][q]["global_metrics"]
            flat = {k: stat_mean(gm, k) for k in METRIC_KEYS + ["KL", "top1_agreement"]}
            vals.append(flat)
        out[q] = {"geometry": config_geometry(cfg), "metrics": summarize_rows(vals)}
    return out


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_num(x) and finite_num(y)]
    if len(pairs) < 3:
        return None
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = rank
            i = j + 1
        return out
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = avg(rx), avg(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (denx * deny) if denx > EPS and deny > EPS else None


def direction_count(rows, orientation, metric, smaller_better=True):
    names = [("R" if orientation == "row" else "C") + str(g) for g in GROUP_SIZES]
    good = 0
    total = 0
    per_prompt = {}
    for r in rows:
        vals = [stat_mean(r["summaries"][n]["global_metrics"], metric) for n in names]
        if all(finite_num(v) for v in vals):
            total += 1
            direction_ok = vals[-1] < vals[0] if smaller_better else vals[-1] > vals[0]
            trend = spearman(GROUP_SIZES, vals)
            if direction_ok:
                good += 1
            per_prompt[r["problem_id"]] = {"values": dict(zip(names, vals)), "spearman_group_size_metric": trend, "rescued_R128_to_16": direction_ok}
    return {"improved_prompt_count": good, "total_prompt_count": total, "rate": good / max(total, 1), "per_prompt": per_prompt}


def dose_response(rows, orientation):
    specs = {
        "within_group_dynamic_range_p95": True,
        "lost_update_fraction": True,
        "state_change_error_E_Delta": True,
        "update_distortion_ratio": True,
        "KL": True,
        "survival_fraction": False,
        "update_cosine": False,
        "top1_agreement": False,
        "state_reconstruction_relative_error_E_S": True,
    }
    return {m: direction_count(rows, orientation, m, smaller_better=sb) for m, sb in specs.items()}


def matched_rc(global_cfg):
    out = {}
    for g in GROUP_SIZES:
        r = global_cfg[f"R{g}"]["metrics"]
        c = global_cfg[f"C{g}"]["metrics"]
        item = {}
        for m in ["within_group_dynamic_range_p95", "lost_update_fraction", "state_change_error_E_Delta", "update_distortion_ratio", "KL", "survival_fraction", "update_cosine", "top1_agreement"]:
            rv = stat_mean(r, m)
            cv = stat_mean(c, m)
            item[m] = {"R": rv, "C": cv, "R_minus_C": rv - cv if finite_num(rv) and finite_num(cv) else None, "R_over_C": ratio(rv, cv)}
        out[f"G{g}"] = item
    return out


def temporal(rows):
    out = {}
    for cfg in CONFIGS:
        q = cfg["name"]
        out[q] = defaultdict(list)
        for r in rows:
            for w, wm in r["summaries"][q].get("window_metrics", {}).items():
                flat = {k: stat_mean(wm, k) for k in METRIC_KEYS + ["KL", "top1_agreement"]}
                out[q][w].append(flat)
        out[q] = {w: summarize_rows(v) for w, v in out[q].items()}
    return out


def association_from_layers(rows):
    pairs = defaultdict(list)
    for r in rows:
        for key, sm in r.get("layer_summary", {}).items():
            dyn = stat_mean(sm, "within_group_dynamic_range_p95")
            if finite_num(dyn) and dyn > 0:
                for m in ["lost_update_fraction", "state_change_error_E_Delta", "update_distortion_ratio"]:
                    v = stat_mean(sm, m)
                    if finite_num(v):
                        pairs[m].append((math.log(dyn), v))
    return {m: {"spearman_log_dynamic_range_vs_metric": spearman([x for x, _ in vals], [y for _, y in vals]), "N": len(vals), "label": "DOSE-RESPONSE ASSOCIATION"} for m, vals in pairs.items()}


def strongest_rescue(rows):
    by_loc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for key, sm in r.get("head_summary", {}).items():
            q, layer, head = key.split("|")
            for m in ["within_group_dynamic_range_p95", "lost_update_fraction", "state_change_error_E_Delta", "update_distortion_ratio", "update_cosine"]:
                v = stat_mean(sm, m)
                if finite_num(v):
                    by_loc[(layer, head, m)][q].append(v)
    out = defaultdict(list)
    for (layer, head, m), vals in by_loc.items():
        base = avg(vals.get("R128", []))
        r16 = avg(vals.get("R16", []))
        if finite_num(base) and finite_num(r16):
            rescue = base - r16 if m != "update_cosine" else r16 - base
            out[m].append({"layer": int(layer), "head": int(head), "R128": base, "R16": r16, "rescue_magnitude": rescue})
    return {m: sorted(v, key=lambda x: x["rescue_magnitude"], reverse=True)[:15] for m, v in out.items()}


def classify(obj):
    row = obj["row_dose_response"]
    matched = obj["matched_group_row_column"]
    def rate(m):
        return row[m]["rate"]
    dyn = rate("within_group_dynamic_range_p95")
    state = min(rate("lost_update_fraction"), rate("state_change_error_E_Delta"), rate("update_distortion_ratio"), rate("survival_fraction"), rate("update_cosine"))
    logit = min(rate("KL"), rate("top1_agreement"))
    dose = min(dyn, state, logit)
    es = rate("state_reconstruction_relative_error_E_S")
    state_exceeds_snapshot = state >= es and state >= 5 / 6
    gaps = []
    for g in GROUP_SIZES:
        gap = matched[f"G{g}"]["KL"]["R_minus_C"]
        if finite_num(gap):
            gaps.append(abs(gap))
    gap_shrinks = len(gaps) == 4 and gaps[-1] < gaps[0] * 0.75
    table = {
        "ROW_GROUP_REDUCTION_LOWERS_DYNAMIC_RANGE": "YES" if dyn >= 5 / 6 else ("PARTIAL" if dyn >= 4 / 6 else "NO"),
        "ROW_GROUP_REDUCTION_RESCUES_STATE_CHANGE_PRESERVATION": "YES" if state >= 5 / 6 else ("PARTIAL" if state >= 4 / 6 else "NO"),
        "ROW_GROUP_REDUCTION_RESCUES_LOGIT_FIDELITY": "YES" if logit >= 5 / 6 else ("PARTIAL" if logit >= 4 / 6 else "NO"),
        "ROW_RESCUE_SHOWS_DOSE_RESPONSE": "YES" if dose >= 5 / 6 else ("PARTIAL" if dose >= 4 / 6 else "NO"),
        "RESCUE_IS_CROSS_PROMPT_CONSISTENT": "YES" if dose >= 5 / 6 else ("PARTIAL" if dose >= 4 / 6 else "NO"),
        "STATE_CHANGE_RESCUE_EXCEEDS_SNAPSHOT_RESCUE": "YES" if state_exceeds_snapshot else ("PARTIAL" if state >= es else "NO"),
        "MATCHED_GROUP_ROW_COLUMN_GAP_SHRINKS": "YES" if gap_shrinks else ("PARTIAL" if len(gaps) == 4 and gaps[-1] < gaps[0] else "NO"),
        "AXIS_GEOMETRY_CAUSAL_SUPPORT": "YES" if min(dyn, state, logit) >= 5 / 6 else ("PARTIAL" if min(dyn, state, logit) >= 4 / 6 else "NO"),
    }
    if table["AXIS_GEOMETRY_CAUSAL_SUPPORT"] == "YES" and table["MATCHED_GROUP_ROW_COLUMN_GAP_SHRINKS"] in ("YES", "PARTIAL"):
        final = "AXIS_GEOMETRY_CAUSAL_SUPPORT_STRONG"
    elif table["AXIS_GEOMETRY_CAUSAL_SUPPORT"] in ("YES", "PARTIAL"):
        final = "AXIS_GEOMETRY_CAUSAL_SUPPORT_PARTIAL"
    elif table["MATCHED_GROUP_ROW_COLUMN_GAP_SHRINKS"] == "NO" and table["ROW_GROUP_REDUCTION_LOWERS_DYNAMIC_RANGE"] in ("YES", "PARTIAL"):
        final = "ORIENTATION_EFFECT_REMAINS_AFTER_GEOMETRY_RESCUE"
    else:
        final = "AXIS_GEOMETRY_CAUSAL_HYPOTHESIS_NOT_SUPPORTED"
    candidate = "YES" if table["AXIS_GEOMETRY_CAUSAL_SUPPORT"] == "YES" and table["ROW_GROUP_REDUCTION_RESCUES_STATE_CHANGE_PRESERVATION"] == "YES" and table["ROW_GROUP_REDUCTION_RESCUES_LOGIT_FIDELITY"] == "YES" else "NO"
    return table, final, candidate


def analyze(stage, num_shards=1):
    rows = load_stage_rows(stage)
    global_cfg = global_by_config(rows)
    obj = {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "FORMAL_STATUS": "COMPLETE" if stage == "formal" and len(rows) == 6 else ("NOT_RUN" if stage != "formal" else "INCOMPLETE"),
        "STAGE_STATUS": "COMPLETE",
        "METHOD_DESIGN_READY": "NO",
        "requested_prompt_count": 6 if stage == "formal" else 1,
        "completed_prompt_count": len(rows),
        "requested_teacher_forced_horizon": 1024 if stage == "formal" else (512 if stage == "pilot" else 128),
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "quantization_geometry": [config_geometry(c) for c in CONFIGS],
        "PROTOCOL_GATE": "PASS" if rows and all(r.get("PROTOCOL_GATE") == "PASS" for r in rows) else "FAIL",
        "METRIC_GATE": "PASS" if rows and all(r.get("METRIC_GATE") == "PASS" for r in rows) else "FAIL",
        "global_metrics": global_cfg,
        "row_dose_response": dose_response(rows, "row"),
        "column_dose_response": dose_response(rows, "column"),
        "matched_group_row_column": matched_rc(global_cfg),
        "temporal_windows": temporal(rows),
        "dose_response_association": association_from_layers(rows),
        "layer_head_strongest_rescue_locations": strongest_rescue(rows),
        "scale_count_confound_note": "R128->R16 changes both group size and scale count; matched R/C at the same group size controls scale count and isolates orientation/membership geometry more closely.",
    }
    table, final, candidate = classify(obj)
    obj["final_classification_table"] = table
    obj["AXIS_GEOMETRY_CAUSAL_SUPPORT"] = table["AXIS_GEOMETRY_CAUSAL_SUPPORT"]
    obj["FINAL_CLASSIFICATION"] = final
    obj["METHOD_DESIGN_READY_CANDIDATE"] = candidate
    obj["SUPPORTED_CONCLUSION"] = "Group-geometry intervention provides causal support only if dose-response and matched R/C controls pass; it is not a proposed method."
    obj["NEGATIVE_RESULT"] = "No formal method design, kernel, mixed precision, stochastic rounding, clipping, INT4, cadence, replay, or serving algorithm is validated."
    if stage == "pilot":
        obj["AXIS_RESCUE_PILOT"] = "POSITIVE" if table["AXIS_GEOMETRY_CAUSAL_SUPPORT"] in ("YES", "PARTIAL") else "NEGATIVE_OR_INCONCLUSIVE"
    save_json(RESULT, obj)
    write_report(obj)
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))
    return obj


def write_report(obj):
    lines = [
        "# GDN INT8 Axis Geometry Rescue Causal Diagnostic V1",
        "",
        "## Observation",
        f"- stage: `{obj['stage']}`",
        f"- FORMAL_STATUS: `{obj['FORMAL_STATUS']}`",
        f"- PROTOCOL_GATE: `{obj['PROTOCOL_GATE']}`; METRIC_GATE: `{obj['METRIC_GATE']}`",
        "- METHOD_DESIGN_READY: `NO`",
        f"- METHOD_DESIGN_READY_CANDIDATE: `{obj['METHOD_DESIGN_READY_CANDIDATE']}`",
        f"- AXIS_GEOMETRY_CAUSAL_SUPPORT: `{obj['AXIS_GEOMETRY_CAUSAL_SUPPORT']}`",
        f"- FINAL_CLASSIFICATION: `{obj['FINAL_CLASSIFICATION']}`",
        "",
        "## Table 1: Quantization Geometry",
        "```json",
        json.dumps(obj["quantization_geometry"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 2: Global Metrics",
        "```json",
        json.dumps(obj["global_metrics"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 3: Row Dose Response",
        "```json",
        json.dumps(obj["row_dose_response"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 4: Column Dose Response",
        "```json",
        json.dumps(obj["column_dose_response"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 5: Matched Group Size R/C",
        "```json",
        json.dumps(obj["matched_group_row_column"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 6: 6-Prompt Consistency",
        "```json",
        json.dumps(obj["row_dose_response"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 7: Temporal Windows",
        "```json",
        json.dumps(obj["temporal_windows"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 8: Snapshot Rescue vs State-Change Rescue",
        "```json",
        json.dumps({
            "snapshot": obj["row_dose_response"].get("state_reconstruction_relative_error_E_S"),
            "state_change": {k: obj["row_dose_response"].get(k) for k in ["state_change_error_E_Delta", "lost_update_fraction", "update_distortion_ratio"]},
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Table 9: Layer/Head Strongest Rescue Locations",
        "```json",
        json.dumps(obj["layer_head_strongest_rescue_locations"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 10: Final Causal Classification",
        "```json",
        json.dumps({
            "table": obj["final_classification_table"],
            "AXIS_GEOMETRY_CAUSAL_SUPPORT": obj["AXIS_GEOMETRY_CAUSAL_SUPPORT"],
            "FINAL_CLASSIFICATION": obj["FINAL_CLASSIFICATION"],
            "METHOD_DESIGN_READY_CANDIDATE": obj["METHOD_DESIGN_READY_CANDIDATE"],
            "METHOD_DESIGN_READY": "NO",
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Limitations",
        f"- {obj['scale_count_confound_note']}",
        "- These quantizers are CAUSAL / MECHANISM DIAGNOSTIC CONTROLS, not a proposed method.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage(args):
    prompts = p1.selected_prompt_rows()
    if args.stage in ("smoke", "pilot"):
        prompts = [prompts[0]]
    else:
        prompts = [p for i, p in enumerate(prompts) if i % args.num_shards == args.shard_id]
    paths = stage_paths(args.stage, args.shard_id if args.num_shards > 1 and args.stage == "formal" else None)
    done = set()
    if args.resume and paths["records"].exists():
        done = {r["problem_id"] for r in iter_jsonl(paths["records"]) if r.get("stage") == args.stage}
    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    for pm in prompts:
        if pm["problem_id"] in done:
            continue
        print(f"[{now()}] {args.stage} shard={args.shard_id} prompt={pm['problem_id']}", flush=True)
        run_prompt(torch, model, tokenizer, e2e, pm, args.stage, args.max_steps, paths, args.shard_id)


def audit():
    p1_result = RES / "gdn_int8_orientation_state_change_mechanism_v1.json"
    p1_obj = json.loads(p1_result.read_text(encoding="utf-8"))
    obj = {
        "task": TASK,
        "stage": "stage0_audit",
        "timestamp": now(),
        "P1_FORMAL_STATUS": p1_obj.get("FORMAL_STATUS"),
        "P1_completed_prompt_count": p1_obj.get("completed_prompt_count"),
        "P1_PROTOCOL_GATE": p1_obj.get("PROTOCOL_GATE"),
        "P1_METRIC_GATE": p1_obj.get("METRIC_GATE"),
        "METHOD_DESIGN_READY": "NO",
        "block_quantizer_plan": [config_geometry(c) for c in CONFIGS],
        "same_scale_rule": "scale generated from S_pre group and reused for q_pre and q_in",
        "allowed_intervention": "regular contiguous block group geometry only",
        "forbidden": ["stochastic rounding", "percentile clipping", "outlier extraction", "mixed precision", "INT4", "kernel", "serving", "method design"],
        "STAGE0_GATE": "PASS" if p1_obj.get("FORMAL_STATUS") == "COMPLETE" and p1_obj.get("PROTOCOL_GATE") == "PASS" and p1_obj.get("METRIC_GATE") == "PASS" else "FAIL",
    }
    save_json(RES / "gdn_int8_axis_geometry_rescue_v1_stage0_audit.json", obj)
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "smoke", "pilot", "formal", "analyze"], required=True)
    ap.add_argument("--analyze-stage", choices=["smoke", "pilot", "formal"], default="formal")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage == "audit":
        audit()
    elif args.stage == "analyze":
        analyze(args.analyze_stage, args.num_shards)
    else:
        run_stage(args)


if __name__ == "__main__":
    main()
