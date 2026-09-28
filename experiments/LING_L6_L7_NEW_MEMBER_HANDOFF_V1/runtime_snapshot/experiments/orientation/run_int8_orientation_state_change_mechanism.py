#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/path/to/gdn_data_root"))
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"
SHARDS = RES / "gdn_end2end_bit_axis_screening_v1_shards"
SUBSET = RES / "gdn_end2end_bit_axis_screening_v1_math_subset.json"
P0 = RES / "gdn_int8_row_e2e_completion_v1.json"
TRAJ_AUDIT = RES / "gdn_int8_orientation_trajectory_source_audit_v1.json"
TRAJ_AUDIT_MD = REP / "gdn_int8_orientation_trajectory_source_audit_v1.md"
RESULT = RES / "gdn_int8_orientation_state_change_mechanism_v1.json"
RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_orientation_state_change_mechanism_v1.md"
FORMAL_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
FORMAL_TOKEN_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_formal_token_records.jsonl.gz"
FORMAL_LAYER_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_formal_layer_records.jsonl.gz"
FORMAL_HEAD_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_formal_head_records.jsonl.gz"

TASK = "GDN_INT8_ORIENTATION_STATE_CHANGE_MECHANISM_V1"
EPS = 1e-12
SELECTED_PATHOLOGICAL = [
    "test/algebra/1332.json",
    "test/counting_and_probability/119.json",
    "test/geometry/477.json",
    "test/geometry/702.json",
]
SELECTED_CONTROLS = [
    "test/algebra/1214.json",
    "test/intermediate_algebra/207.json",
]
GDN_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
    16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30,
]
QUANTIZERS = {
    "int8_row": {"name": "INT8-row", "bits": 8, "granularity": "row"},
    "int8_column": {"name": "INT8-column", "bits": 8, "granularity": "column"},
}
WINDOWS = [(1, 128), (129, 256), (257, 512), (513, 1024), (1025, 1536), (1537, 2048)]
FORMAL_HORIZON = 2048
HEAD_SAMPLE_STRIDE = 32
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


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sha256_text(x):
    return hashlib.sha256((x or "").encode("utf-8")).hexdigest()


def run_cmd(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return "ERROR: %s" % exc


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


def avg(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else None


def pct(xs, q):
    xs = sorted(float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x)))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def finite_num(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_subset():
    obj = json.loads(SUBSET.read_text(encoding="utf-8"))
    return {r["problem_id"]: r for r in obj["records"]}


def read_all_e2e_rows():
    rows = []
    for pat in ("shard_*.jsonl", "shard_p0_int8_row_queue_*.jsonl"):
        for p in sorted(SHARDS.glob(pat)):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        r["_source_file"] = str(p)
                        rows.append(r)
    bykey = {}
    for r in rows:
        bykey[(r.get("quantizer"), r.get("benchmark"), r.get("problem_id"))] = r
    return bykey


def selected_prompt_rows():
    subset = load_subset()
    e2e = read_all_e2e_rows()
    out = []
    for pid in SELECTED_PATHOLOGICAL + SELECTED_CONTROLS:
        role = "INT8-row truncated pathological candidate" if pid in SELECTED_PATHOLOGICAL else "INT8-row non-truncated termination control"
        fp = e2e.get(("FP_STATE", "MATH-500", pid), {})
        row = e2e.get(("INT8-row", "MATH-500", pid), {})
        col = e2e.get(("INT8-column", "MATH-500", pid), {})
        item = subset[pid]
        out.append({
            "role": role,
            "problem_id": pid,
            "problem_index": item["problem_index"],
            "subject": item.get("subject"),
            "level": item.get("level"),
            "problem": item["problem"],
            "fp_response": fp.get("response", ""),
            "fp_correct": fp.get("correct"),
            "fp_truncated": fp.get("truncated"),
            "fp_tokens": fp.get("output_token_count"),
            "int8_row_correct": row.get("correct"),
            "int8_row_truncated": row.get("truncated"),
            "int8_row_tokens": row.get("output_token_count"),
            "int8_column_correct": col.get("correct"),
            "int8_column_truncated": col.get("truncated"),
            "int8_column_tokens": col.get("output_token_count"),
        })
    return out


def setup_model():
    import run_end2end_bit_axis_screening as e2e
    e2e.ensure_imports()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.loads((EXP / "run_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return torch, model, tokenizer, cfg, e2e


def get_state(cache, layer):
    states = getattr(cache.layers[layer], "recurrent_states")
    return states[0] if not isinstance(states, dict) else states[0]


def feed_step(torch, model, input_ids, mask, past):
    if past is None:
        return model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    return model(input_ids=input_ids, past_key_values=past, use_cache=True)


def logits_metrics(torch, ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    other_logp = torch.log_softmax(other, dim=-1)
    kl = float((ref_logp.exp() * (ref_logp - other_logp)).sum().item())
    top1 = int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item())
    return {"KL": kl, "top1_agreement": top1}


def qparams(torch, state, q):
    bits = q["bits"]
    gran = q["granularity"]
    qmax = (2 ** (bits - 1)) - 1
    if gran == "row":
        scale = state.detach().abs().amax(dim=(-1,), keepdim=True).clamp_min(EPS) / qmax
    elif gran == "column":
        scale = state.detach().abs().amax(dim=(-2,), keepdim=True).clamp_min(EPS) / qmax
    else:
        raise ValueError(gran)
    return scale, qmax


def cosine(torch, a, b):
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    an = float(torch.linalg.vector_norm(af).item())
    bn = float(torch.linalg.vector_norm(bf).item())
    if an < EPS and bn < EPS:
        return None
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item())


def metric_event(torch, s_in, s_pre, q):
    pre = s_pre.detach().float()
    inn = s_in.detach().float()
    finite = bool(torch.isfinite(pre).all().item() and torch.isfinite(inn).all().item())
    if not finite:
        return {"status": "NONFINITE_STATE"}
    scale, qmax = qparams(torch, pre, q)
    q_pre = torch.round(pre / scale).clamp(-qmax, qmax)
    q_in = torch.round(inn / scale).clamp(-qmax, qmax)
    s_post = q_pre * scale
    delta = pre - inn
    delta_hat = (q_pre - q_in) * scale
    residual = s_post - pre
    d2 = float(torch.sum(delta.double() * delta.double()).item())
    lost = float(torch.sum(delta.double()[q_pre == q_in] ** 2).item()) if d2 > EPS else 0.0
    r_lost = lost / (d2 + EPS) if d2 > EPS else None
    if r_lost is not None and (r_lost < -1e-6 or r_lost > 1.0 + 1e-6):
        raise AssertionError("bounded lost-update fraction out of range: %r" % r_lost)
    survival = float((q_pre != q_in).float().mean().item())
    if survival < -1e-6 or survival > 1.0 + 1e-6:
        raise AssertionError("survival out of range: %r" % survival)
    err = delta - delta_hat
    dh2 = float(torch.sum(delta_hat.double() * delta_hat.double()).item())
    dist = float(torch.sum(err.double() * err.double()).item()) / (d2 + EPS) if d2 > EPS else None
    norm_ratio = math.sqrt(dh2) / max(math.sqrt(d2), EPS) if d2 > EPS else None
    es_num = float(torch.linalg.vector_norm(residual).item())
    es_den = max(float(torch.linalg.vector_norm(pre).item()), EPS)
    ed_num = float(torch.linalg.vector_norm(err).item())
    ed_den = max(float(torch.linalg.vector_norm(delta).item()), EPS)
    proj = float(torch.sum(residual.double() * delta.double()).item()) / (d2 + EPS) if d2 > EPS else None
    scale_f = scale.detach().float().flatten()
    scale_vals = scale_f.detach().cpu().tolist()
    # Within-group dynamic range: max(abs(values in quant group)) / median(abs(values in group)).
    if q["granularity"] == "row":
        grouped = pre.abs().reshape(-1, pre.shape[-1])
    else:
        grouped = pre.abs().transpose(-2, -1).reshape(-1, pre.shape[-2])
    gmax = grouped.max(dim=1).values
    gmed = grouped.median(dim=1).values.clamp_min(EPS)
    dyn = (gmax / gmed).detach().cpu().tolist()
    return {
        "status": "OK" if d2 > EPS else "ZERO_UPDATE_EVENT",
        "state_reconstruction_relative_error_E_S": es_num / es_den,
        "state_change_error_E_Delta": ed_num / ed_den if d2 > EPS else None,
        "lost_update_fraction": r_lost,
        "survival_fraction": survival,
        "update_distortion_ratio": dist,
        "represented_update_norm_ratio": norm_ratio,
        "update_cosine": cosine(torch, delta, delta_hat),
        "residual_update_cosine": cosine(torch, residual, delta),
        "residual_update_projection_coeff": proj,
        "state_norm": float(torch.linalg.vector_norm(pre).item()),
        "delta_norm": math.sqrt(d2),
        "scale_mean": avg(scale_vals),
        "scale_median": statistics.median(scale_vals) if scale_vals else None,
        "scale_max": max(scale_vals) if scale_vals else None,
        "scale_p95": pct(scale_vals, 0.95),
        "scale_p95_over_median": (pct(scale_vals, 0.95) / max(statistics.median(scale_vals), EPS)) if scale_vals else None,
        "within_group_dynamic_range_mean": avg(dyn),
        "within_group_dynamic_range_median": statistics.median(dyn) if dyn else None,
        "within_group_dynamic_range_p95": pct(dyn, 0.95),
        "saturation_fraction": float((q_pre.abs() >= qmax).float().mean().item()),
        "zero_code_fraction": float((q_pre == 0).float().mean().item()),
        "qrange": [-qmax, qmax],
        "scale_shape": list(scale.shape),
        "_s_post": s_post,
    }


def public(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def aggregate(rows):
    keys = sorted({k for r in rows for k, v in r.items() if finite_num(v)})
    return {k: avg([r.get(k) for r in rows]) for k in keys}


def formal_paths(shard_id=None):
    suffix = "" if shard_id is None else "_shard_%03d" % shard_id
    return {
        "records": RES / ("gdn_int8_orientation_state_change_mechanism_v1_formal_records%s.jsonl" % suffix),
        "token": RES / ("gdn_int8_orientation_state_change_mechanism_v1_formal_token_records%s.jsonl.gz" % suffix),
        "layer": RES / ("gdn_int8_orientation_state_change_mechanism_v1_formal_layer_records%s.jsonl.gz" % suffix),
        "head": RES / ("gdn_int8_orientation_state_change_mechanism_v1_formal_head_records%s.jsonl.gz" % suffix),
    }


class OnlineAgg:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.values = []

    def add(self, x):
        if finite_num(x):
            x = float(x)
            self.n += 1
            self.sum += x
            self.values.append(x)

    def out(self):
        return {
            "N": self.n,
            "mean": self.sum / self.n if self.n else None,
            "median": statistics.median(self.values) if self.values else None,
            "p90": pct(self.values, 0.90),
            "p95": pct(self.values, 0.95),
        }


def add_metrics(bucket, rec):
    for k in METRIC_KEYS:
        if k not in bucket:
            bucket[k] = OnlineAgg()
        bucket[k].add(rec.get(k))


def bucket_out(bucket):
    return {k: v.out() for k, v in sorted(bucket.items())}


def window_label(token_idx):
    one = int(token_idx) + 1
    for a, b in WINDOWS:
        if a <= one <= b:
            return "%d-%d" % (a, b)
    return ">%d" % WINDOWS[-1][1]


def merge_gz(parts, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as out:
        for p in sorted(parts):
            if not p.exists():
                continue
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run_formal_prompt(torch, model, tokenizer, cfg, e2e, pm, paths, args):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[:args.max_steps]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    ref_mask = enc.get("attention_mask")
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    ref_past = None
    q_inputs = {k: ref_input.clone() for k in QUANTIZERS}
    q_masks = {k: ref_mask.clone() if ref_mask is not None else None for k in QUANTIZERS}
    q_pasts = {k: None for k in QUANTIZERS}
    started = now()
    protocol_gate = "PASS"
    metric_gate = "PASS"
    token_agg = defaultdict(dict)
    layer_agg = defaultdict(dict)
    head_agg = defaultdict(dict)
    token_counts = defaultdict(int)
    exceptions = []
    with torch.inference_mode():
        for t, tok in enumerate(cont_ids):
            ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past)
            ref_past = ref_out.past_key_values
            fp_record = {
                "task": TASK, "stage": "formal", "problem_id": pm["problem_id"], "role": pm["role"],
                "token_idx": t, "window": window_label(t), "quantizer": "FP_STATE",
                "KL": 0.0, "top1_agreement": 1.0, "quantization_metrics": "N/A_REFERENCE",
            }
            gz_append(paths["token"], fp_record)
            for qk, q in QUANTIZERS.items():
                s_in = None
                if q_pasts[qk] is not None:
                    s_in = {layer: get_state(q_pasts[qk], layer).detach().clone() for layer in GDN_LAYERS}
                q_out = feed_step(torch, model, q_inputs[qk], q_masks[qk], q_pasts[qk])
                q_pasts[qk] = q_out.past_key_values
                layer_metrics = []
                if s_in is not None:
                    for layer in GDN_LAYERS:
                        state = get_state(q_pasts[qk], layer)
                        s_pre = state.detach().clone()
                        if list(s_pre.shape) != [1, 32, 128, 128]:
                            protocol_gate = "FAIL"
                        m = metric_event(torch, s_in[layer], s_pre, q)
                        if m.get("status") not in ("OK", "ZERO_UPDATE_EVENT"):
                            metric_gate = "FAIL"
                        expected_shape = [1, 32, 128, 1] if q["granularity"] == "row" else [1, 32, 1, 128]
                        if m.get("scale_shape") != expected_shape or m.get("qrange") != [-127, 127]:
                            protocol_gate = "FAIL"
                        state.copy_(m["_s_post"].to(state.dtype))
                        pub = public(m)
                        pub.update({
                            "task": TASK, "stage": "formal", "problem_id": pm["problem_id"], "role": pm["role"],
                            "token_idx": t, "window": window_label(t), "quantizer": q["name"], "layer_idx": layer,
                        })
                        layer_metrics.append(pub)
                        gz_append(paths["layer"], pub)
                        add_metrics(layer_agg[(q["name"], layer)], pub)
                        if t % HEAD_SAMPLE_STRIDE == 0:
                            for h in range(s_pre.shape[1]):
                                hm = metric_event(torch, s_in[layer][:, h:h+1, :, :], s_pre[:, h:h+1, :, :], q)
                                hpub = public(hm)
                                hpub.update({
                                    "task": TASK, "stage": "formal", "problem_id": pm["problem_id"], "role": pm["role"],
                                    "token_idx": t, "window": window_label(t), "quantizer": q["name"],
                                    "layer_idx": layer, "head_idx": h, "head_sample_stride": HEAD_SAMPLE_STRIDE,
                                })
                                gz_append(paths["head"], hpub)
                                add_metrics(head_agg[(q["name"], layer, h)], hpub)
                lm = logits_metrics(torch, ref_out.logits, q_out.logits)
                token_rec = {
                    "task": TASK, "stage": "formal", "problem_id": pm["problem_id"], "role": pm["role"],
                    "token_idx": t, "window": window_label(t), "quantizer": q["name"],
                    "KL": lm["KL"], "top1_agreement": lm["top1_agreement"],
                    "layer_mean": aggregate(layer_metrics), "metric_status": "INIT_NO_S_IN" if s_in is None else "OK",
                }
                gz_append(paths["token"], token_rec)
                add_metrics(token_agg[(q["name"], "all")], token_rec["layer_mean"])
                add_metrics(token_agg[(q["name"], window_label(t))], token_rec["layer_mean"])
                if "KL" not in token_agg[(q["name"], "all")]:
                    token_agg[(q["name"], "all")]["KL"] = OnlineAgg()
                    token_agg[(q["name"], "all")]["top1_agreement"] = OnlineAgg()
                if "KL" not in token_agg[(q["name"], window_label(t))]:
                    token_agg[(q["name"], window_label(t))]["KL"] = OnlineAgg()
                    token_agg[(q["name"], window_label(t))]["top1_agreement"] = OnlineAgg()
                token_agg[(q["name"], "all")]["KL"].add(lm["KL"])
                token_agg[(q["name"], "all")]["top1_agreement"].add(lm["top1_agreement"])
                token_agg[(q["name"], window_label(t))]["KL"].add(lm["KL"])
                token_agg[(q["name"], window_label(t))]["top1_agreement"].add(lm["top1_agreement"])
                token_counts[q["name"]] += 1
            ref_input = torch.tensor([[tok]], dtype=ref_input.dtype, device=device)
            ref_mask = None
            for qk in QUANTIZERS:
                q_inputs[qk] = ref_input.clone()
                q_masks[qk] = None
    ended = now()
    summaries = {}
    for qname in ("FP_STATE", "INT8-row", "INT8-column"):
        if qname == "FP_STATE":
            summaries[qname] = {"tokens_completed": len(cont_ids), "quantization_metrics": "N/A_REFERENCE", "mean_KL": 0.0, "top1_agreement": 1.0}
            continue
        summaries[qname] = {
            "tokens_completed": token_counts[qname],
            "global_metrics": bucket_out(token_agg[(qname, "all")]),
            "window_metrics": {w: bucket_out(token_agg[(qname, w)]) for w in [("%d-%d" % x) for x in WINDOWS]},
        }
    summary = {
        "task": TASK,
        "stage": "formal",
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "prompt_sha256": sha256_text(prompt),
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "teacher_forced_horizon_requested": args.max_steps,
        "actual_token_count": len(cont_ids),
        "survived_full_horizon": len(cont_ids) >= args.max_steps,
        "start_timestamp": started,
        "end_timestamp": ended,
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "PROTOCOL_GATE": protocol_gate,
        "METRIC_GATE": metric_gate,
        "finite_status": "PASS" if not exceptions else "EXCEPTION",
        "exception": exceptions,
        "summaries": summaries,
        "layer_summary": {"%s|%s" % k: bucket_out(v) for k, v in layer_agg.items()},
        "head_summary": {"%s|%s|%s" % k: bucket_out(v) for k, v in head_agg.items()},
    }
    append_jsonl(paths["records"], summary)
    return summary


def formal_stage(args):
    prompts = selected_prompt_rows()
    if args.num_shards > 1:
        prompts = [p for i, p in enumerate(prompts) if i % args.num_shards == args.shard_id]
    paths = formal_paths(args.shard_id if args.num_shards > 1 else None)
    done = set()
    if args.resume and paths["records"].exists():
        for r in iter_jsonl(paths["records"]):
            if r.get("stage") == "formal":
                done.add(r.get("problem_id"))
    torch, model, tokenizer, cfg, e2e = setup_model()
    out = []
    for pm in prompts:
        if pm["problem_id"] in done:
            continue
        print("[%s] formal shard=%d prompt=%s" % (now(), args.shard_id, pm["problem_id"]), flush=True)
        out.append(run_formal_prompt(torch, model, tokenizer, cfg, e2e, pm, paths, args))
    return out


def stat_mean(stats, key):
    obj = stats.get(key)
    if isinstance(obj, dict):
        return obj.get("mean")
    return None


def read_formal_summaries():
    rows = []
    for p in sorted(RES.glob("gdn_int8_orientation_state_change_mechanism_v1_formal_records*.jsonl")):
        for r in iter_jsonl(p):
            if r.get("stage") == "formal":
                rows.append(r)
    by_pid = {}
    for r in rows:
        by_pid[r["problem_id"]] = r
    return [by_pid[k] for k in sorted(by_pid)]


def ratio(a, b):
    return float(a) / float(b) if finite_num(a) and finite_num(b) and abs(float(b)) > EPS else None


def prompt_comparison_row(r):
    row = r["summaries"]["INT8-row"]["global_metrics"]
    col = r["summaries"]["INT8-column"]["global_metrics"]
    out = {"problem_id": r["problem_id"], "role": r["role"], "actual_token_count": r["actual_token_count"]}
    for k in ["state_reconstruction_relative_error_E_S", "state_change_error_E_Delta", "lost_update_fraction",
              "survival_fraction", "update_distortion_ratio", "represented_update_norm_ratio", "update_cosine",
              "residual_update_cosine", "scale_p95_over_median", "within_group_dynamic_range_p95", "KL", "top1_agreement"]:
        rv = stat_mean(row, k)
        cv = stat_mean(col, k)
        out[k + "_row"] = rv
        out[k + "_column"] = cv
        out[k + "_row_minus_column"] = (rv - cv) if finite_num(rv) and finite_num(cv) else None
        out[k + "_row_over_column"] = ratio(rv, cv)
    return out


def global_from_prompts(rows):
    out = {}
    prompt_rows = [prompt_comparison_row(r) for r in rows]
    for q in ("INT8-row", "INT8-column"):
        buckets = defaultdict(list)
        for r in rows:
            gm = r["summaries"][q]["global_metrics"]
            for k in METRIC_KEYS + ["KL", "top1_agreement"]:
                v = stat_mean(gm, k)
                if finite_num(v):
                    buckets[k].append(v)
        out[q] = {k: {"mean": avg(v), "median": statistics.median(v), "p90": pct(v, 0.90), "p95": pct(v, 0.95)} for k, v in buckets.items()}
    consistency = {}
    for k in ["state_reconstruction_relative_error_E_S", "state_change_error_E_Delta", "lost_update_fraction",
              "survival_fraction", "update_distortion_ratio", "represented_update_norm_ratio", "update_cosine",
              "scale_p95_over_median", "within_group_dynamic_range_p95", "KL"]:
        vals = [p.get(k + "_row_minus_column") for p in prompt_rows]
        vals = [v for v in vals if finite_num(v)]
        if k in ("survival_fraction", "update_cosine", "represented_update_norm_ratio", "top1_agreement"):
            good = sum(1 for v in vals if v < 0)
            direction = "row < column"
        else:
            good = sum(1 for v in vals if v > 0)
            direction = "row > column"
        consistency[k] = {"direction": direction, "consistent_prompts": good, "total_prompts": len(vals), "rate": good / max(len(vals), 1)}
    return out, prompt_rows, consistency


def temporal_summary(rows):
    out = {}
    for q in ("INT8-row", "INT8-column"):
        out[q] = {}
        for w in ["%d-%d" % x for x in WINDOWS]:
            buckets = defaultdict(list)
            for r in rows:
                wm = r["summaries"][q]["window_metrics"].get(w, {})
                for k in METRIC_KEYS + ["KL", "top1_agreement"]:
                    v = stat_mean(wm, k)
                    if finite_num(v):
                        buckets[k].append(v)
            out[q][w] = {k: avg(v) for k, v in buckets.items()}
    gaps = {}
    for w in ["%d-%d" % x for x in WINDOWS]:
        gaps[w] = {}
        for k in ["state_reconstruction_relative_error_E_S", "state_change_error_E_Delta", "lost_update_fraction",
                  "survival_fraction", "update_distortion_ratio", "update_cosine", "KL", "within_group_dynamic_range_p95"]:
            rv = out["INT8-row"][w].get(k)
            cv = out["INT8-column"][w].get(k)
            gaps[w][k + "_row_minus_column"] = (rv - cv) if finite_num(rv) and finite_num(cv) else None
            gaps[w][k + "_row_over_column"] = ratio(rv, cv)
    return {"by_quantizer": out, "row_minus_column": gaps}


def phenotype_summary(rows):
    groups = {
        "INT8-row truncated pathological candidates": [r for r in rows if "truncated pathological" in r["role"]],
        "INT8-row non-truncated termination controls": [r for r in rows if "non-truncated termination control" in r["role"]],
    }
    out = {}
    for name, vals in groups.items():
        buckets = defaultdict(list)
        for r in vals:
            gm = r["summaries"]["INT8-row"]["global_metrics"]
            for k in METRIC_KEYS + ["KL", "top1_agreement"]:
                v = stat_mean(gm, k)
                if finite_num(v):
                    buckets[k].append(v)
        out[name] = {"prompt_count": len(vals), "metrics": {k: avg(v) for k, v in buckets.items()}}
    return out


def ranking_from_summary(rows, kind, metric, higher_worse=True, limit=15):
    vals = defaultdict(lambda: {"row": [], "col": [], "prompts": set()})
    field = "layer_summary" if kind == "layer" else "head_summary"
    for r in rows:
        for key, summary in r.get(field, {}).items():
            parts = key.split("|")
            q = parts[0]
            loc = tuple(parts[1:])
            v = stat_mean(summary, metric)
            if finite_num(v):
                if q == "INT8-row":
                    vals[loc]["row"].append(v)
                    vals[loc]["prompts"].add(r["problem_id"])
                elif q == "INT8-column":
                    vals[loc]["col"].append(v)
    out = []
    for loc, d in vals.items():
        rv = avg(d["row"])
        cv = avg(d["col"])
        if finite_num(rv) and finite_num(cv):
            gap = rv - cv
            score = gap if higher_worse else -gap
            out.append({"location": loc, "row_mean": rv, "column_mean": cv, "row_minus_column": gap,
                        "score": score, "prompt_count": len(d["prompts"])})
    return sorted(out, key=lambda x: x["score"], reverse=True)[:limit]


def classify(rows, global_summary, consistency, temporal, phenotype):
    def yn(cond, partial=False):
        return "YES" if cond else ("PARTIAL" if partial else "NO")
    geom = consistency.get("within_group_dynamic_range_p95", {}).get("rate", 0)
    state = min(consistency.get("lost_update_fraction", {}).get("rate", 0),
                consistency.get("state_change_error_E_Delta", {}).get("rate", 0),
                consistency.get("update_distortion_ratio", {}).get("rate", 0))
    es_ratio = ratio(global_summary["INT8-row"].get("state_reconstruction_relative_error_E_S", {}).get("mean"),
                     global_summary["INT8-column"].get("state_reconstruction_relative_error_E_S", {}).get("mean"))
    ed_ratio = ratio(global_summary["INT8-row"].get("state_change_error_E_Delta", {}).get("mean"),
                     global_summary["INT8-column"].get("state_change_error_E_Delta", {}).get("mean"))
    lost_ratio = ratio(global_summary["INT8-row"].get("lost_update_fraction", {}).get("mean"),
                       global_summary["INT8-column"].get("lost_update_fraction", {}).get("mean"))
    progressive = False
    gaps = temporal["row_minus_column"]
    early = gaps.get("1-128", {}).get("KL_row_minus_column")
    late = gaps.get("1537-2048", {}).get("KL_row_minus_column")
    if finite_num(early) and finite_num(late) and late > early * 1.25:
        progressive = True
    t = phenotype["INT8-row truncated pathological candidates"]["metrics"]
    c = phenotype["INT8-row non-truncated termination controls"]["metrics"]
    phenotype_signal = 0
    for k in ["lost_update_fraction", "state_change_error_E_Delta", "update_distortion_ratio", "KL"]:
        if finite_num(t.get(k)) and finite_num(c.get(k)) and t[k] > c[k]:
            phenotype_signal += 1
    table = {
        "INT8_AXIS_SCALE_GEOMETRY_DIFFERENCE_REPLICATED": yn(geom >= 5/6, geom >= 4/6),
        "INT8_STATE_CHANGE_PRESERVATION_DIFFERENCE_REPLICATED": yn(state >= 5/6, state >= 4/6),
        "STATE_CHANGE_SEPARATION_EXCEEDS_SNAPSHOT_SEPARATION": yn(finite_num(ed_ratio) and finite_num(es_ratio) and ed_ratio > es_ratio * 1.25, finite_num(lost_ratio) and lost_ratio > 1.5),
        "INT8_PROGRESSIVE_RECURRENT_ACCUMULATION_SIGNAL": yn(progressive, False),
        "INT8_ROW_TRUNCATION_PHENOTYPE_ASSOCIATED_WITH_STATE_CHANGE_METRICS": yn(phenotype_signal >= 3, phenotype_signal >= 2),
        "INT8_ORIENTATION_X_RECURRENCE_INTERACTION_SUPPORTED": "PARTIAL",
    }
    if table["INT8_AXIS_SCALE_GEOMETRY_DIFFERENCE_REPLICATED"] in ("YES", "PARTIAL") and table["INT8_STATE_CHANGE_PRESERVATION_DIFFERENCE_REPLICATED"] in ("YES", "PARTIAL"):
        final = "INT8_ORIENTATION_X_RECURRENCE_INTERACTION"
    elif table["INT8_STATE_CHANGE_PRESERVATION_DIFFERENCE_REPLICATED"] == "YES":
        final = "INT8_STATE_CHANGE_PRESERVATION_SIGNAL_IDENTIFIED"
    elif table["INT8_AXIS_SCALE_GEOMETRY_DIFFERENCE_REPLICATED"] == "YES":
        final = "INT8_LOCAL_AXIS_GEOMETRY_DOMINANT"
    elif progressive:
        final = "INT8_PROGRESSIVE_RECURRENT_ACCUMULATION_SIGNAL"
    else:
        final = "INT8_MECHANISM_INCONCLUSIVE"
    return table, final


def analyze_formal(args):
    parts = [formal_paths(i) for i in range(args.num_shards)]
    merge_records = [p["records"] for p in parts]
    FORMAL_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with FORMAL_RECORDS.open("w", encoding="utf-8") as out:
        for p in sorted(merge_records):
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out.write(line + "\n")
    merge_gz([p["token"] for p in parts], FORMAL_TOKEN_RECORDS)
    merge_gz([p["layer"] for p in parts], FORMAL_LAYER_RECORDS)
    merge_gz([p["head"] for p in parts], FORMAL_HEAD_RECORDS)
    rows = read_formal_summaries()
    global_summary, prompt_rows, consistency = global_from_prompts(rows)
    temporal = temporal_summary(rows)
    phenotype = phenotype_summary(rows)
    class_table, final_classification = classify(rows, global_summary, consistency, temporal, phenotype)
    layer_rankings = {
        "R_lost_gap": ranking_from_summary(rows, "layer", "lost_update_fraction"),
        "R_distortion_gap": ranking_from_summary(rows, "layer", "update_distortion_ratio"),
        "E_Delta_gap": ranking_from_summary(rows, "layer", "state_change_error_E_Delta"),
        "update_cosine_gap": ranking_from_summary(rows, "layer", "update_cosine", higher_worse=False),
        "scale_geometry_gap": ranking_from_summary(rows, "layer", "within_group_dynamic_range_p95"),
    }
    head_rankings = {
        "R_lost_gap": ranking_from_summary(rows, "head", "lost_update_fraction"),
        "R_distortion_gap": ranking_from_summary(rows, "head", "update_distortion_ratio"),
        "E_Delta_gap": ranking_from_summary(rows, "head", "state_change_error_E_Delta"),
        "update_cosine_gap": ranking_from_summary(rows, "head", "update_cosine", higher_worse=False),
        "scale_geometry_gap": ranking_from_summary(rows, "head", "within_group_dynamic_range_p95"),
    }
    smoke_prov = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
    completed = len(rows)
    finite = sum(1 for r in rows if r.get("finite_status") == "PASS")
    survived = sum(1 for r in rows if r.get("survived_full_horizon"))
    obj = {
        "task": TASK,
        "stage": "FORMAL_COMPLETE" if completed == 6 else "FORMAL_INCOMPLETE",
        "timestamp": now(),
        "FORMAL_STATUS": "COMPLETE" if completed == 6 else "INCOMPLETE",
        "METHOD_DESIGN_READY": "NO",
        "requested_prompt_count": 6,
        "completed_prompt_count": completed,
        "surviving_prompt_count": survived,
        "finite_prompt_count": finite,
        "requested_config_count_per_prompt": 3,
        "requested_teacher_forced_horizon": FORMAL_HORIZON,
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "PROTOCOL_GATE": "PASS" if all(r.get("PROTOCOL_GATE") == "PASS" for r in rows) and completed == 6 else "FAIL",
        "METRIC_GATE": "PASS" if all(r.get("METRIC_GATE") == "PASS" for r in rows) and completed == 6 else "FAIL",
        "smoke_provenance": smoke_prov,
        "global_row_vs_column": global_summary,
        "prompt_wise_paired_values": prompt_rows,
        "cross_prompt_consistency": consistency,
        "temporal_windows": temporal,
        "truncated_vs_nontruncated_termination_controls": phenotype,
        "layer_level_strongest_gaps": layer_rankings,
        "head_level_strongest_gaps": head_rankings,
        "final_classification_table": class_table,
        "FINAL_CLASSIFICATION": final_classification,
        "SUPPORTED_CONCLUSION": "Formal evidence supports mechanistic association only; no causal intervention was run.",
        "NEGATIVE_RESULT": "This experiment does not validate method design, replay readiness, or termination causality.",
        "PROPOSED_FOLLOW_UP": "GDN_INT8_AXIS_RESCUE_CAUSAL_DIAGNOSTIC_V1" if final_classification == "INT8_ORIENTATION_X_RECURRENCE_INTERACTION" else "Review mechanism table before selecting cadence or single-shot follow-up.",
    }
    save_json(RESULT, obj)
    write_formal_report(obj)
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def fmt(x):
    return "NA" if x is None else ("%.6g" % x if isinstance(x, float) else str(x))


def write_formal_report(obj):
    lines = [
        "# GDN INT8 Orientation State-Change Mechanism V1",
        "",
        "## OBSERVATION",
        "- FORMAL_STATUS: `%s`" % obj["FORMAL_STATUS"],
        "- PROTOCOL_GATE: `%s`; METRIC_GATE: `%s`" % (obj["PROTOCOL_GATE"], obj["METRIC_GATE"]),
        "- continuation_source: `P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED`; exact original generation replay is not claimed.",
        "- METHOD_DESIGN_READY: `NO`",
        "",
        "## Table 1: Global INT8-row vs INT8-column",
        "```json",
        json.dumps(obj["global_row_vs_column"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 2: 6 Prompts Paired Values",
        "```json",
        json.dumps(obj["prompt_wise_paired_values"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 3: Temporal Windows",
        "```json",
        json.dumps(obj["temporal_windows"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 4: Truncated vs Non-Truncated Termination Controls",
        "```json",
        json.dumps(obj["truncated_vs_nontruncated_termination_controls"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 5: Layer-Level Strongest Row-Column Gaps",
        "```json",
        json.dumps(obj["layer_level_strongest_gaps"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 6: Head-Level Strongest Row-Column Gaps",
        "```json",
        json.dumps(obj["head_level_strongest_gaps"], indent=2, sort_keys=True),
        "```",
        "",
        "## Table 7: Snapshot vs State-Change Separation",
        "```json",
        json.dumps({
            "E_S_row_over_column": ratio(obj["global_row_vs_column"]["INT8-row"].get("state_reconstruction_relative_error_E_S", {}).get("mean"), obj["global_row_vs_column"]["INT8-column"].get("state_reconstruction_relative_error_E_S", {}).get("mean")),
            "E_Delta_row_over_column": ratio(obj["global_row_vs_column"]["INT8-row"].get("state_change_error_E_Delta", {}).get("mean"), obj["global_row_vs_column"]["INT8-column"].get("state_change_error_E_Delta", {}).get("mean")),
            "R_lost_row_over_column": ratio(obj["global_row_vs_column"]["INT8-row"].get("lost_update_fraction", {}).get("mean"), obj["global_row_vs_column"]["INT8-column"].get("lost_update_fraction", {}).get("mean")),
            "R_distortion_row_over_column": ratio(obj["global_row_vs_column"]["INT8-row"].get("update_distortion_ratio", {}).get("mean"), obj["global_row_vs_column"]["INT8-column"].get("update_distortion_ratio", {}).get("mean")),
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Final Classification",
        "```json",
        json.dumps({"table": obj["final_classification_table"], "FINAL_CLASSIFICATION": obj["FINAL_CLASSIFICATION"]}, indent=2, sort_keys=True),
        "```",
        "",
        "## HYPOTHESIS",
        "- Axis scale geometry may be associated with state-change preservation differences under recurrent feedback.",
        "",
        "## SUPPORTED CONCLUSION",
        "- %s" % obj["SUPPORTED_CONCLUSION"],
        "",
        "## NEGATIVE RESULT",
        "- %s" % obj["NEGATIVE_RESULT"],
        "",
        "## PROPOSED FOLLOW-UP",
        "- %s" % obj["PROPOSED_FOLLOW_UP"],
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def trajectory_source_audit():
    torch, model, tokenizer, cfg, e2e = setup_model()
    records = []
    for pm in selected_prompt_rows():
        response = pm["fp_response"]
        ids_plain = tokenizer.encode(response, add_special_tokens=False)
        ids_special = tokenizer.encode(response, add_special_tokens=True)
        decoded = tokenizer.decode(ids_plain, skip_special_tokens=False)
        decoded_skip = tokenizer.decode(ids_plain, skip_special_tokens=True)
        records.append({
            "problem_id": pm["problem_id"],
            "role": pm["role"],
            "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
            "exact_original_generation_token_ids_available": False,
            "retokenized_token_count": len(ids_plain),
            "retokenized_with_special_token_count": len(ids_special),
            "special_token_delta": len(ids_special) - len(ids_plain),
            "decode_encode_text_exact_match": decoded == response,
            "decode_encode_skip_special_text_exact_match": decoded_skip == response,
            "normalization_difference": {
                "original_sha256": sha256_text(response),
                "decoded_sha256": sha256_text(decoded),
                "decoded_skip_special_sha256": sha256_text(decoded_skip),
                "original_len": len(response),
                "decoded_len": len(decoded),
                "decoded_skip_special_len": len(decoded_skip),
                "leading_changed": response[:80] != decoded[:80],
                "trailing_changed": response[-80:] != decoded[-80:],
            },
            "fp_tokens_reported_by_p0": pm["fp_tokens"],
            "fp_correct": pm["fp_correct"],
            "fp_truncated": pm["fp_truncated"],
            "int8_row_correct": pm["int8_row_correct"],
            "int8_row_truncated": pm["int8_row_truncated"],
            "int8_column_correct": pm["int8_column_correct"],
            "int8_column_truncated": pm["int8_column_truncated"],
        })
    obj = {
        "task": TASK,
        "stage": "trajectory_source_audit",
        "timestamp": now(),
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "low_cost_fp_regeneration_decision": "NOT_RUN",
        "low_cost_fp_regeneration_reason": "P0 trajectories can be up to 32768 generated tokens; regenerating exact P0 FP_STATE for 6 prompts is not cheap. Formal mechanism smoke uses retokenized P0 FP response with explicit non-exact metadata.",
        "selected_prompts": [{k: v for k, v in pm.items() if k != "problem" and k != "fp_response"} for pm in selected_prompt_rows()],
        "records": records,
        "tokenizer": {
            "model_path": cfg["model_path"],
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        },
    }
    save_json(TRAJ_AUDIT, obj)
    lines = [
        "# GDN INT8 Orientation Trajectory Source Audit V1",
        "",
        "## OBSERVATION",
        "- continuation_source: `P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED`",
        "- exact_original_generation_token_ids_available: `false`",
        "- exact replay claimed: `false`",
        "",
        "## RECORDS",
        "```json",
        json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## NEGATIVE RESULT",
        "- Retokenized decoded P0 text is not labeled as exact replay of original generation token IDs.",
    ]
    TRAJ_AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    TRAJ_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def smoke():
    torch, model, tokenizer, cfg, e2e = setup_model()
    pm = selected_prompt_rows()[0]
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[:128]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    q_inputs = {k: ref_input.clone() for k in QUANTIZERS}
    ref_mask = enc.get("attention_mask")
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    q_masks = {k: ref_mask.clone() if ref_mask is not None else None for k in QUANTIZERS}
    ref_past = None
    q_pasts = {k: None for k in QUANTIZERS}
    q_prev = {k: None for k in QUANTIZERS}
    summaries = {}
    metric_gate = "PASS"
    protocol_gate = "PASS"
    rows_by_q = defaultdict(list)
    layer_rows = []
    head_rows = []
    with torch.inference_mode():
        for t, tok in enumerate(cont_ids):
            ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past)
            ref_past = ref_out.past_key_values
            for qk, q in QUANTIZERS.items():
                s_in = None
                if q_pasts[qk] is not None:
                    s_in = {layer: get_state(q_pasts[qk], layer).detach().clone() for layer in GDN_LAYERS}
                q_out = feed_step(torch, model, q_inputs[qk], q_masks[qk], q_pasts[qk])
                q_pasts[qk] = q_out.past_key_values
                layer_metrics = []
                if s_in is not None:
                    for layer in GDN_LAYERS:
                        state = get_state(q_pasts[qk], layer)
                        s_pre = state.detach().clone()
                        if list(s_pre.shape) != [1, 32, 128, 128]:
                            protocol_gate = "FAIL"
                        m = metric_event(torch, s_in[layer], s_pre, q)
                        if m.get("status") not in ("OK", "ZERO_UPDATE_EVENT"):
                            metric_gate = "FAIL"
                        expected_shape = [1, 32, 128, 1] if q["granularity"] == "row" else [1, 32, 1, 128]
                        if m.get("scale_shape") != expected_shape:
                            protocol_gate = "FAIL"
                        state.copy_(m["_s_post"].to(state.dtype))
                        pub = public(m)
                        pub.update({"token_idx": t, "layer_idx": layer, "quantizer": q["name"], "problem_id": pm["problem_id"]})
                        layer_metrics.append(pub)
                        gz_append(LAYER_RECORDS, pub)
                        if layer == 25:
                            for h in range(s_pre.shape[1]):
                                hm = metric_event(torch, s_in[layer][:, h:h+1, :, :], s_pre[:, h:h+1, :, :], q)
                                hpub = public(hm)
                                hpub.update({"token_idx": t, "layer_idx": layer, "head_idx": h, "quantizer": q["name"], "problem_id": pm["problem_id"]})
                                head_rows.append(hpub)
                                gz_append(HEAD_RECORDS, hpub)
                lm = logits_metrics(torch, ref_out.logits, q_out.logits)
                row = {
                    "task": TASK,
                    "stage": "smoke",
                    "problem_id": pm["problem_id"],
                    "role": pm["role"],
                    "token_idx": t,
                    "quantizer": q["name"],
                    "KL": lm["KL"],
                    "top1_agreement": lm["top1_agreement"],
                    "layer_mean": aggregate(layer_metrics),
                    "metric_status": "INIT_NO_S_IN" if s_in is None else "OK",
                }
                rows_by_q[q["name"]].append(row)
                gz_append(TOKEN_RECORDS, row)
            ref_input = torch.tensor([[tok]], dtype=ref_input.dtype, device=device)
            ref_mask = None
            for qk in QUANTIZERS:
                q_inputs[qk] = ref_input.clone()
                q_masks[qk] = None
    for qname, rows in rows_by_q.items():
        lm = [r["layer_mean"] for r in rows if r["layer_mean"]]
        summaries[qname] = {
            "tokens": len(rows),
            "mean_KL": avg([r["KL"] for r in rows]),
            "top1_agreement": avg([r["top1_agreement"] for r in rows]),
            "metrics": aggregate(lm),
        }
    row = summaries.get("INT8-row", {}).get("metrics", {})
    col = summaries.get("INT8-column", {}).get("metrics", {})
    separation = {
        k: (row.get(k) - col.get(k)) if finite_num(row.get(k)) and finite_num(col.get(k)) else None
        for k in [
            "state_reconstruction_relative_error_E_S",
            "lost_update_fraction",
            "survival_fraction",
            "update_distortion_ratio",
            "represented_update_norm_ratio",
            "update_cosine",
            "residual_update_cosine",
            "KL",
            "scale_p95_over_median",
            "within_group_dynamic_range_p95",
        ]
    }
    obj = {
        "task": TASK,
        "stage": "smoke",
        "timestamp": now(),
        "PROTOCOL_GATE": protocol_gate,
        "METRIC_GATE": metric_gate,
        "formal_run_started": False,
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "problem_id": pm["problem_id"],
        "tokens_requested": 128,
        "tokens_available": len(tokenizer.encode(pm["fp_response"], add_special_tokens=False)),
        "tokens_used": len(cont_ids),
        "summaries": summaries,
        "row_minus_column": separation,
        "metadata": {
            "model_path": cfg["model_path"],
            "transformers_path": "os.environ.get(GDN_TRANSFORMERS_SRC, /path/to/transformers-qwen35)",
            "revision": run_cmd(["git", "rev-parse", "HEAD"], cwd=str(EXP)),
            "GPU_ID": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "dtype": "bfloat16 model load",
            "prefill_precision": "FP recurrent state prefill; no quantization before continuation",
            "continuation_precision": "teacher-forced continuation; INT8 row/column fake-quantized after each recurrent update",
            "qrange": {"INT8": [-127, 127]},
            "rounding_mode": "torch.round",
            "zero_point": 0,
            "layer_scope": "all 24 GDN layers",
            "prompt_selection_rule": "4 INT8-row truncated pathological candidates plus 2 INT8-row non-truncated termination controls from P0 paired records",
            "teacher_forced_horizon": 128,
            "metric_implementation_version": "same-scale q_in/q_pre; E_S separated from E_Delta; residual-update cosine/projection included",
        },
    }
    save_json(RESULT, obj)
    append_jsonl(RECORDS, obj)
    lines = [
        "# GDN INT8 Orientation State-Change Mechanism V1 Smoke",
        "",
        "## OBSERVATION",
        "- PROTOCOL_GATE: `%s`" % protocol_gate,
        "- METRIC_GATE: `%s`" % metric_gate,
        "- continuation_source: `P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED`; exact replay is not claimed.",
        "",
        "## RESULTS",
        "```json",
        json.dumps({"summaries": summaries, "row_minus_column": separation}, indent=2, sort_keys=True),
        "```",
        "",
        "## NEGATIVE RESULT",
        "- No formal 6x3x2048 run has been started by this smoke stage.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["trajectory_audit", "smoke", "formal", "analyze"], required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=FORMAL_HORIZON)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage == "trajectory_audit":
        trajectory_source_audit()
    elif args.stage == "smoke":
        smoke()
    elif args.stage == "formal":
        formal_stage(args)
    elif args.stage == "analyze":
        analyze_formal(args)


if __name__ == "__main__":
    main()
