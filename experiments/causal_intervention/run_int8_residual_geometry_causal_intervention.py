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
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/path/to/gdn_data_root"))
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_RESIDUAL_GEOMETRY_CAUSAL_INTERVENTION_V1"
RESULT = RES / "gdn_int8_residual_geometry_causal_intervention_v1.json"
STAGE0 = RES / "gdn_int8_residual_geometry_causal_intervention_v1_stage0.json"
RECORDS = RES / "gdn_int8_residual_geometry_causal_intervention_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_residual_geometry_causal_intervention_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_residual_geometry_causal_intervention_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_residual_geometry_causal_intervention_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_residual_geometry_causal_intervention_v1.md"
FORMAL_RECORDS = RES / "gdn_int8_residual_geometry_causal_intervention_v1_formal_records.jsonl"
FORMAL_TOKEN_RECORDS = RES / "gdn_int8_residual_geometry_causal_intervention_v1_formal_token_records.jsonl.gz"
FORMAL_LAYER_RECORDS = RES / "gdn_int8_residual_geometry_causal_intervention_v1_formal_layer_records.jsonl.gz"
FORMAL_HEAD_RECORDS = RES / "gdn_int8_residual_geometry_causal_intervention_v1_formal_head_records.jsonl.gz"

P1_RESULT = RES / "gdn_int8_orientation_state_change_mechanism_v1.json"
P1_RECORDS = RES / "gdn_int8_orientation_state_change_mechanism_v1_formal_records.jsonl"
SMALLVAL_RESULT = RES / "gdn_int8_effective_update_small_prompt_validation_v1.json"
STRENGTH_RESULT = RES / "gdn_int8_residual_strength_causal_intervention_v1.json"

EPS = 1e-12
EPS_DIRECTION = 1e-12
RANDOM_SEED = 20260831
R128_CFG = {"name": "R128", "orientation": "row", "group_size": 128}
CONFIGS = ["FP_STATE", "REAL_R128", "PARALLEL_PLUS", "PARALLEL_MINUS", "ORTHOGONAL_REALDIR", "ORTHOGONAL_RANDOM"]
GEOMETRY_CONFIGS = ["PARALLEL_PLUS", "PARALLEL_MINUS", "ORTHOGONAL_REALDIR", "ORTHOGONAL_RANDOM"]
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


def median(xs):
    xs = [float(x) for x in xs if finite_num(x)]
    return statistics.median(xs) if xs else None


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if finite_num(v):
                buckets[k].append(v)
    return {k: {"mean": avg(v), "median": median(v), "p95": pct(v, 0.95), "max": max(v)} for k, v in sorted(buckets.items())}


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
        "records": RES / f"gdn_int8_residual_geometry_causal_intervention_v1_records{suffix}.jsonl",
        "token": RES / f"gdn_int8_residual_geometry_causal_intervention_v1_token_records{suffix}.jsonl.gz",
        "layer": RES / f"gdn_int8_residual_geometry_causal_intervention_v1_layer_records{suffix}.jsonl.gz",
        "head": RES / f"gdn_int8_residual_geometry_causal_intervention_v1_head_records{suffix}.jsonl.gz",
    }


def prompt_manifest():
    p1_obj = json.loads(P1_RESULT.read_text(encoding="utf-8"))
    small_obj = json.loads(SMALLVAL_RESULT.read_text(encoding="utf-8"))
    strength_obj = json.loads(STRENGTH_RESULT.read_text(encoding="utf-8"))
    ids = [m["prompt_id"] for m in small_obj.get("prompt_manifest", [])]
    if len(ids) != 6:
        ids = []
        for row in iter_jsonl(P1_RECORDS):
            pid = row.get("problem_id")
            if pid and pid not in ids:
                ids.append(pid)
    prompt_rows = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    manifest = []
    for pid in ids:
        pm = prompt_rows.get(pid)
        if pm:
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
    return manifest, p1_obj, small_obj, strength_obj


def cosine(torch, a, b):
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    an = float(torch.linalg.vector_norm(af).item())
    bn = float(torch.linalg.vector_norm(bf).item())
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item())


def random_like(torch, shape, device, dtype, token_idx, layer_idx, head_idx):
    gen = torch.Generator(device=device)
    gen.manual_seed(RANDOM_SEED + int(token_idx) * 1000003 + int(layer_idx) * 1009 + int(head_idx))
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def project_off(torch, vec, base):
    denom = torch.sum(base.double() * base.double())
    coeff = torch.sum(vec.double() * base.double()) / (denom + EPS)
    return vec - coeff.to(vec.dtype) * base, float(coeff.item())


def construct_residual(torch, cfg, delta_h, shadow_h, ref_norm, token_idx, layer_idx, head_idx):
    delta = delta_h.detach().float()
    shadow = shadow_h.detach().float() if shadow_h is not None else None
    dnorm = float(torch.linalg.vector_norm(delta).item())
    if dnorm <= EPS_DIRECTION:
        return None, {"direction_status": "DEGENERATE_DELTA", "degenerate_delta": 1, "realdir_fallback": 0}
    target = float(ref_norm)
    u_delta = delta / (dnorm + EPS)
    fallback = 0
    if cfg == "PARALLEL_PLUS":
        vec = target * u_delta
        target_cos = 1.0
    elif cfg == "PARALLEL_MINUS":
        vec = -target * u_delta
        target_cos = -1.0
    elif cfg in ("ORTHOGONAL_REALDIR", "ORTHOGONAL_RANDOM"):
        if cfg == "ORTHOGONAL_REALDIR":
            raw, _ = project_off(torch, shadow, delta)
            if float(torch.linalg.vector_norm(raw).item()) <= EPS_DIRECTION:
                fallback = 1
                raw = random_like(torch, delta.shape, delta.device, delta.dtype, token_idx, layer_idx, head_idx)
                raw, _ = project_off(torch, raw, delta)
        else:
            raw = random_like(torch, delta.shape, delta.device, delta.dtype, token_idx, layer_idx, head_idx)
            raw, _ = project_off(torch, raw, delta)
        rnorm = float(torch.linalg.vector_norm(raw).item())
        if rnorm <= EPS_DIRECTION:
            return None, {"direction_status": "DEGENERATE_ORTHOGONAL", "degenerate_delta": 0, "realdir_fallback": fallback}
        vec = target * raw / (rnorm + EPS)
        target_cos = 0.0
    else:
        raise ValueError(cfg)
    return vec, {"direction_status": "OK", "target_cosine": target_cos, "degenerate_delta": 0, "realdir_fallback": fallback}


def residual_metrics(torch, s_in_h, s_pre_h, residual_h, ref_norm, target_cosine, direction_status, fallback):
    inn = s_in_h.detach().float()
    pre = s_pre_h.detach().float()
    res = residual_h.detach().float()
    desired = pre - inn
    write = pre + res
    effective = write - inn
    desired_norm = float(torch.linalg.vector_norm(desired).item())
    res_norm = float(torch.linalg.vector_norm(res).item())
    pre_norm = float(torch.linalg.vector_norm(pre).item())
    eff_norm = float(torch.linalg.vector_norm(effective).item())
    cos_ru = cosine(torch, res, desired)
    d2 = float(torch.sum(desired.double() * desired.double()).item())
    alpha = float(torch.sum(res.double() * desired.double()).item()) / (d2 + EPS) if d2 > EPS else None
    if finite_num(alpha):
        r_parallel = alpha * desired
        r_perp = res - r_parallel
        orth_ratio = float(torch.linalg.vector_norm(r_perp).item()) / (desired_norm + EPS)
        par_component = float(torch.linalg.vector_norm(r_parallel).item()) / (res_norm + EPS)
        orth_component = float(torch.linalg.vector_norm(r_perp).item()) / (res_norm + EPS)
    else:
        orth_ratio = None
        par_component = None
        orth_component = None
    ratio = res_norm / (float(ref_norm) + EPS)
    return {
        "status": "OK",
        "direction_status": direction_status,
        "reference_residual_norm": float(ref_norm),
        "injected_residual_norm": res_norm,
        "norm_match_ratio": ratio,
        "norm_match_absdev": abs(ratio - 1.0) if float(ref_norm) > EPS else abs(res_norm),
        "target_residual_update_cosine": target_cosine,
        "residual_update_cosine": cos_ru,
        "angle_cosine_abs_error": abs(cos_ru - target_cosine) if finite_num(cos_ru) and finite_num(target_cosine) else None,
        "actual_angle_degrees": math.degrees(math.acos(max(-1.0, min(1.0, cos_ru)))) if finite_num(cos_ru) else None,
        "state_reconstruction_relative_error_E_S": res_norm / (pre_norm + EPS),
        "D_effective": res_norm / (desired_norm + EPS),
        "cos_effective": cosine(torch, desired, effective),
        "norm_ratio_effective": eff_norm / (desired_norm + EPS),
        "alpha": alpha,
        "orthogonal_ratio": orth_ratio,
        "parallel_component_ratio": par_component,
        "orthogonal_component_ratio": orth_component,
        "desired_state_change_norm": desired_norm,
        "effective_update_norm": eff_norm,
        "valid_direction_count": 1,
        "degenerate_delta_count": 0,
        "realdir_fallback_count": fallback,
    }


def layer_head_intervention(torch, cfg, s_in, s_pre, ref_norms, token_idx, layer_idx):
    pre = s_pre.detach().float()
    inn = s_in.detach().float()
    shadow_q, _, _, _, _, _ = axis.grouped_quant(torch, pre, pre, R128_CFG)
    shadow_res = shadow_q - pre
    residual = torch.zeros_like(pre)
    head_rows = []
    degenerate = 0
    fallback_count = 0
    for h in range(pre.shape[1]):
        delta_h = pre[:, h:h + 1, :, :] - inn[:, h:h + 1, :, :]
        shadow_h = shadow_res[:, h:h + 1, :, :]
        vec, info = construct_residual(torch, cfg, delta_h, shadow_h, ref_norms[h], token_idx, layer_idx, h)
        if vec is None:
            degenerate += 1
            continue
        residual[:, h:h + 1, :, :] = vec.to(residual.dtype)
        hm = residual_metrics(torch, inn[:, h:h + 1, :, :], pre[:, h:h + 1, :, :], vec, ref_norms[h], info["target_cosine"], info["direction_status"], info["realdir_fallback"])
        hm["head_idx"] = h
        head_rows.append(hm)
        fallback_count += info["realdir_fallback"]
    s_write = pre + residual
    lm = summarize(head_rows)
    layer = {k: stat_mean(lm, k) for k in lm}
    layer.update({
        "status": "OK" if head_rows else "NO_VALID_DIRECTIONS",
        "valid_direction_count": len(head_rows),
        "degenerate_delta_count": degenerate,
        "realdir_fallback_count": fallback_count,
        "_s_write": s_write,
        "_head_rows": head_rows,
    })
    return layer


def real_r128_layer(torch, s_in, s_pre):
    pre = s_pre.detach().float()
    inn = s_in.detach().float()
    s_q, _, _, _, _, _ = axis.grouped_quant(torch, pre, pre, R128_CFG)
    r = s_q - pre
    ref_norms = [float(torch.linalg.vector_norm(r[:, h:h + 1, :, :].float()).item()) for h in range(r.shape[1])]
    m = eff.effective_metrics(torch, inn, pre, s_q)
    m.update({
        "status": "OK",
        "reference_residual_norm": float(torch.linalg.vector_norm(r.float()).item()),
        "injected_residual_norm": float(torch.linalg.vector_norm(r.float()).item()),
        "norm_match_ratio": 1.0,
        "norm_match_absdev": 0.0,
        "residual_update_cosine": cosine(torch, r, pre - inn),
        "actual_angle_degrees": None,
        "parallel_component_ratio": None,
        "orthogonal_component_ratio": None,
        "valid_direction_count": r.shape[1],
        "degenerate_delta_count": 0,
        "realdir_fallback_count": 0,
        "_s_write": s_q,
        "_ref_norms": ref_norms,
    })
    return m


def run_prompt(torch, model, tokenizer, e2e, pm, stage, horizon, paths, args):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[:horizon]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    ref_mask = enc.get("attention_mask")
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    inputs = {c: ref_input.clone() for c in CONFIGS if c != "FP_STATE"}
    masks = {c: ref_mask.clone() if ref_mask is not None else None for c in CONFIGS if c != "FP_STATE"}
    ref_past = None
    pasts = {c: None for c in CONFIGS if c != "FP_STATE"}
    token_rows = defaultdict(list)
    layer_rows = defaultdict(list)
    head_rows = defaultdict(list)
    gates = {"PROTOCOL_GATE": "PASS", "METRIC_GATE": "PASS", "NORM_MATCH_GATE": "PASS", "ANGLE_GATE": "PASS"}
    print(f"[{now()}] {stage} shard={args.shard_id} prompt={pm['problem_id']} tokens={len(cont_ids)} configs={CONFIGS}", flush=True)
    with torch.inference_mode():
        for t, tok in enumerate(cont_ids):
            ref_out = p1.feed_step(torch, model, ref_input, ref_mask, ref_past)
            ref_past = ref_out.past_key_values
            gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "token_idx": t, "config": "FP_STATE", "KL": 0.0, "top1_agreement": 1.0})
            s_in_real = None
            if pasts["REAL_R128"] is not None:
                s_in_real = {layer: p1.get_state(pasts["REAL_R128"], layer).detach().clone() for layer in GDN_LAYERS}
            real_out = p1.feed_step(torch, model, inputs["REAL_R128"], masks["REAL_R128"], pasts["REAL_R128"])
            pasts["REAL_R128"] = real_out.past_key_values
            ref_norms_by_layer = {}
            real_layer_metrics = []
            if s_in_real is not None:
                for layer in GDN_LAYERS:
                    state = p1.get_state(pasts["REAL_R128"], layer)
                    pre = state.detach().clone()
                    m = real_r128_layer(torch, s_in_real[layer], pre)
                    state.copy_(m["_s_write"].to(state.dtype))
                    ref_norms_by_layer[layer] = m["_ref_norms"]
                    pub = {k: v for k, v in m.items() if not k.startswith("_")}
                    pub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "config": "REAL_R128", "layer_idx": layer})
                    real_layer_metrics.append(pub)
                    layer_rows["REAL_R128"].append(pub)
                    gz_append(paths["layer"], pub)
            lm = eff.logits_metrics(torch, ref_out.logits, real_out.logits)
            flat = summarize(real_layer_metrics)
            one = {k: stat_mean(flat, k) for k in flat}
            one.update(lm)
            one["token_idx"] = t
            token_rows["REAL_R128"].append(one)
            gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "config": "REAL_R128", "KL": lm["KL"], "top1_agreement": lm["top1_agreement"], "layer_mean": one, "metric_status": "INIT_NO_S_IN" if s_in_real is None else "OK"})
            for cfg in GEOMETRY_CONFIGS:
                s_in_cfg = None
                if pasts[cfg] is not None:
                    s_in_cfg = {layer: p1.get_state(pasts[cfg], layer).detach().clone() for layer in GDN_LAYERS}
                out = p1.feed_step(torch, model, inputs[cfg], masks[cfg], pasts[cfg])
                pasts[cfg] = out.past_key_values
                cfg_layer_metrics = []
                if s_in_cfg is not None:
                    for layer in GDN_LAYERS:
                        state = p1.get_state(pasts[cfg], layer)
                        pre = state.detach().clone()
                        m = layer_head_intervention(torch, cfg, s_in_cfg[layer], pre, ref_norms_by_layer[layer], t, layer)
                        if m["status"] != "OK":
                            gates["METRIC_GATE"] = "FAIL"
                            continue
                        if m.get("norm_match_absdev", 1.0) > 2e-5:
                            gates["NORM_MATCH_GATE"] = "FAIL"
                        target = 1.0 if cfg == "PARALLEL_PLUS" else (-1.0 if cfg == "PARALLEL_MINUS" else 0.0)
                        if m.get("angle_cosine_abs_error", 1.0) > (2e-5 if abs(target) == 1.0 else 2e-4):
                            gates["ANGLE_GATE"] = "FAIL"
                        state.copy_(m["_s_write"].to(state.dtype))
                        pub = {k: v for k, v in m.items() if not k.startswith("_")}
                        pub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "config": cfg, "layer_idx": layer})
                        cfg_layer_metrics.append(pub)
                        layer_rows[cfg].append(pub)
                        gz_append(paths["layer"], pub)
                        if t % HEAD_SAMPLE_STRIDE == 0:
                            for hr in m["_head_rows"]:
                                hpub = dict(hr)
                                hpub.update({"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "config": cfg, "layer_idx": layer, "head_sample_stride": HEAD_SAMPLE_STRIDE})
                                head_rows[cfg].append(hpub)
                                gz_append(paths["head"], hpub)
                lm = eff.logits_metrics(torch, ref_out.logits, out.logits)
                flat = summarize(cfg_layer_metrics)
                one = {k: stat_mean(flat, k) for k in flat}
                one.update(lm)
                one["token_idx"] = t
                token_rows[cfg].append(one)
                gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "token_idx": t, "window": window_label(t), "config": cfg, "KL": lm["KL"], "top1_agreement": lm["top1_agreement"], "layer_mean": one, "metric_status": "INIT_NO_S_IN" if s_in_cfg is None else "OK"})
            ref_input = torch.tensor([[tok]], dtype=ref_input.dtype, device=device)
            ref_mask = None
            for cfg in inputs:
                inputs[cfg] = ref_input.clone()
                masks[cfg] = None
    summaries = {"FP_STATE": {"valid_token_count": len(cont_ids), "surviving_token_count": len(cont_ids), "token_summary": {"KL": {"mean": 0.0}, "top1_agreement": {"mean": 1.0}}}}
    for cfg in CONFIGS:
        if cfg == "FP_STATE":
            continue
        summaries[cfg] = {
            "valid_token_count": len([r for r in token_rows[cfg] if r.get("token_idx", 0) > 0]),
            "surviving_token_count": len(cont_ids),
            "token_summary": summarize(token_rows[cfg]),
            "layer_summary": summarize(layer_rows[cfg]),
            "head_summary": summarize(head_rows[cfg]),
        }
    real_kl = stat_mean(summaries["REAL_R128"]["token_summary"], "KL")
    if not finite_num(real_kl) or real_kl < 0.005:
        gates["METRIC_GATE"] = "FAIL"
    record = {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "configs": CONFIGS,
        "requested_token_count": horizon,
        "surviving_token_count": len(cont_ids),
        "REAL_R128_canonical_check": "PASS" if gates["METRIC_GATE"] == "PASS" else "FAIL",
        **gates,
        "summaries": summaries,
    }
    append_jsonl(paths["records"], record)
    print(f"[{now()}] {stage} done prompt={pm['problem_id']} gates P/M/N/A={gates['PROTOCOL_GATE']}/{gates['METRIC_GATE']}/{gates['NORM_MATCH_GATE']}/{gates['ANGLE_GATE']}", flush=True)
    return record


def config_metrics(row):
    out = {}
    for cfg in CONFIGS:
        s = row["summaries"][cfg]
        out[cfg] = {
            "KL": stat_mean(s["token_summary"], "KL"),
            "top1_agreement": stat_mean(s["token_summary"], "top1_agreement"),
            "E_S": stat_mean(s.get("layer_summary", {}), "state_reconstruction_relative_error_E_S"),
            "D_effective": stat_mean(s.get("layer_summary", {}), "D_effective"),
            "cos_effective": stat_mean(s.get("layer_summary", {}), "cos_effective"),
            "norm_ratio_effective": stat_mean(s.get("layer_summary", {}), "norm_ratio_effective"),
            "alpha": stat_mean(s.get("layer_summary", {}), "alpha"),
            "orthogonal_ratio": stat_mean(s.get("layer_summary", {}), "orthogonal_ratio"),
            "mean_injected_residual_norm": stat_mean(s.get("layer_summary", {}), "injected_residual_norm"),
            "mean_reference_residual_norm": stat_mean(s.get("layer_summary", {}), "reference_residual_norm"),
            "mean_norm_match_ratio": stat_mean(s.get("layer_summary", {}), "norm_match_ratio"),
            "mean_residual_update_cosine": stat_mean(s.get("layer_summary", {}), "residual_update_cosine"),
            "mean_angle_error": stat_mean(s.get("layer_summary", {}), "angle_cosine_abs_error"),
            "valid_direction_count": stat_mean(s.get("layer_summary", {}), "valid_direction_count"),
            "degenerate_delta_count": stat_mean(s.get("layer_summary", {}), "degenerate_delta_count"),
            "realdir_fallback_count": stat_mean(s.get("layer_summary", {}), "realdir_fallback_count"),
        }
    return out


def prompt_analysis(row):
    metrics = config_metrics(row)
    geom_kls = [metrics[c]["KL"] for c in GEOMETRY_CONFIGS if finite_num(metrics[c]["KL"])]
    real_kl = metrics["REAL_R128"]["KL"]
    spread = max(geom_kls) - min(geom_kls) if geom_kls else None
    threshold = max(0.001, 0.10 * real_kl) if finite_num(real_kl) else None
    pairwise = {}
    for a in CONFIGS:
        for b in CONFIGS:
            if a < b and finite_num(metrics[a]["KL"]) and finite_num(metrics[b]["KL"]):
                pairwise[f"{a}_minus_{b}"] = metrics[a]["KL"] - metrics[b]["KL"]
    return {
        "problem_id": row["problem_id"],
        "role": row["role"],
        "metrics": metrics,
        "geometry_KL_spread": spread,
        "geometry_KL_spread_ratio": spread / (real_kl + EPS) if finite_num(spread) and finite_num(real_kl) else None,
        "geometry_KL_spread_threshold": threshold,
        "clear_same_norm_geometry_separation": bool(finite_num(spread) and finite_num(threshold) and spread >= threshold),
        "pairwise_KL_differences": pairwise,
        "REAL_R128_canonical_check": row.get("REAL_R128_canonical_check"),
        "PROTOCOL_GATE": row.get("PROTOCOL_GATE"),
        "METRIC_GATE": row.get("METRIC_GATE"),
        "NORM_MATCH_GATE": row.get("NORM_MATCH_GATE"),
        "ANGLE_GATE": row.get("ANGLE_GATE"),
    }


def analyze_rows(rows):
    per = [prompt_analysis(r) for r in rows]
    agg = {}
    for cfg in CONFIGS:
        agg[cfg] = {
            "mean_KL": avg([p["metrics"][cfg]["KL"] for p in per]),
            "median_KL": median([p["metrics"][cfg]["KL"] for p in per]),
            "mean_residual_norm": avg([p["metrics"][cfg]["mean_injected_residual_norm"] for p in per]),
            "mean_norm_match_ratio": avg([p["metrics"][cfg]["mean_norm_match_ratio"] for p in per]),
            "mean_target_angle_error": avg([p["metrics"][cfg]["mean_angle_error"] for p in per]),
            "mean_residual_update_cosine": avg([p["metrics"][cfg]["mean_residual_update_cosine"] for p in per]),
        }
    return {
        "per_prompt": per,
        "aggregate_by_config": agg,
        "clear_same_norm_geometry_separation_count": sum(1 for p in per if p["clear_same_norm_geometry_separation"]),
        "orthogonal_realdir_gt_parallel_plus_KL_count": sum(1 for p in per if p["metrics"]["ORTHOGONAL_REALDIR"]["KL"] > p["metrics"]["PARALLEL_PLUS"]["KL"]),
        "orthogonal_random_gt_parallel_plus_KL_count": sum(1 for p in per if p["metrics"]["ORTHOGONAL_RANDOM"]["KL"] > p["metrics"]["PARALLEL_PLUS"]["KL"]),
        "parallel_minus_gt_parallel_plus_KL_count": sum(1 for p in per if p["metrics"]["PARALLEL_MINUS"]["KL"] > p["metrics"]["PARALLEL_PLUS"]["KL"]),
    }


def classify_pilot(row):
    p = prompt_analysis(row)
    gates = all(row.get(k) == "PASS" for k in ["PROTOCOL_GATE", "METRIC_GATE", "NORM_MATCH_GATE", "ANGLE_GATE"])
    return "POSITIVE_RESIDUAL_GEOMETRY_SIGNAL" if gates and p["clear_same_norm_geometry_separation"] else ("RESIDUAL_GEOMETRY_CAUSAL_INCONCLUSIVE" if gates else "RESIDUAL_GEOMETRY_CAUSAL_NOT_SUPPORTED")


def write_report(obj):
    lines = [
        "# GDN INT8 Residual Geometry Causal Intervention V1",
        "",
        "## Gates",
        f"- FORMAL_STATUS: `{obj.get('FORMAL_STATUS')}`",
        f"- PROTOCOL_GATE: `{obj.get('PROTOCOL_GATE')}`; METRIC_GATE: `{obj.get('METRIC_GATE')}`",
        f"- NORM_MATCH_GATE: `{obj.get('NORM_MATCH_GATE')}`; ANGLE_GATE: `{obj.get('ANGLE_GATE')}`",
        f"- FINAL_CLASSIFICATION: `{obj.get('FINAL_CLASSIFICATION')}`",
        f"- METHOD_DESIGN_READY: `{obj.get('METHOD_DESIGN_READY')}`",
        "",
        "## Analysis",
        "```json",
        json.dumps(obj.get("analysis", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Limits",
        "- Oracle same-magnitude geometry intervention only; not a deployable quantizer.",
        "- No residual propagation and no method design are run.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(stage, num_shards):
    if stage == "formal":
        rows = []
        for sid in range(num_shards):
            rows.extend(iter_jsonl(stage_paths("formal", sid)["records"]) or [])
    else:
        rows = list(iter_jsonl(stage_paths(stage)["records"]) or [])
    rows = [r for r in rows if r.get("stage") == stage]
    analysis = analyze_rows(rows) if rows else {}
    gates = {k: ("PASS" if rows and all(r.get(k) == "PASS" for r in rows) else "FAIL") for k in ["PROTOCOL_GATE", "METRIC_GATE", "NORM_MATCH_GATE", "ANGLE_GATE"]}
    if stage == "smoke":
        final = "SMOKE_PASS" if all(v == "PASS" for v in gates.values()) else "SMOKE_FAIL"
        pilot = "NOT_RUN"
        formal = "NOT_RUN"
    elif stage == "pilot":
        pilot = classify_pilot(rows[0]) if rows else "RESIDUAL_GEOMETRY_CAUSAL_NOT_SUPPORTED"
        final = pilot
        formal = "NOT_RUN"
    else:
        pilot = "N/A"
        n = len(rows)
        clear = analysis.get("clear_same_norm_geometry_separation_count", 0)
        if n == 6 and all(v == "PASS" for v in gates.values()) and clear >= 4:
            final = "RESIDUAL_GEOMETRY_CAUSAL_SUPPORT"
            contributor = "SUPPORTED"
            prop_ready = "YES"
        elif n == 6 and clear > 0:
            final = "RESIDUAL_GEOMETRY_CAUSAL_INCONCLUSIVE"
            contributor = "INCONCLUSIVE"
            prop_ready = "NO"
        else:
            final = "RESIDUAL_GEOMETRY_CAUSAL_HYPOTHESIS_NOT_SUPPORTED"
            contributor = "NOT_SUPPORTED"
            prop_ready = "NO"
        formal = "COMPLETE" if n == 6 else "INCOMPLETE"
    ortho = "INCONCLUSIVE"
    if stage == "formal" and analysis:
        if analysis.get("orthogonal_realdir_gt_parallel_plus_KL_count", 0) >= 4 and analysis.get("orthogonal_random_gt_parallel_plus_KL_count", 0) >= 4:
            ortho = "SUPPORTED"
        elif analysis.get("orthogonal_realdir_gt_parallel_plus_KL_count", 0) == 0 and analysis.get("orthogonal_random_gt_parallel_plus_KL_count", 0) == 0:
            ortho = "NOT_SUPPORTED"
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
        "RESIDUAL_GEOMETRY_CAUSAL_CONTRIBUTOR": locals().get("contributor", "NOT_EVALUATED"),
        "ORTHOGONAL_RESIDUAL_CAUSAL_HYPOTHESIS": ortho,
        "RESIDUAL_PROPAGATION_EXPERIMENT_READY": locals().get("prop_ready", "NO"),
        "METHOD_DESIGN_READY": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "prompt_manifest": prompt_manifest()[0],
        "analysis": analysis,
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
    manifest, _, _, _ = prompt_manifest()
    pms = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    if args.stage in ("smoke", "pilot"):
        ids = ["test/algebra/1332.json"] if "test/algebra/1332.json" in [m["prompt_id"] for m in manifest] else [manifest[0]["prompt_id"]]
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
    manifest, p1_obj, small_obj, strength_obj = prompt_manifest()
    protocol = "PASS" if len(manifest) == 6 and p1_obj.get("FORMAL_STATUS") == "COMPLETE" and small_obj.get("FORMAL_STATUS") == "COMPLETE" and strength_obj.get("FINAL_CLASSIFICATION") == "RESIDUAL_STRENGTH_CAUSAL_SUPPORT" and strength_obj.get("RESIDUAL_GEOMETRY_CAUSAL_INTERVENTION_READY") == "YES" else "FAIL"
    obj = {
        "task": TASK,
        "stage": "stage0",
        "timestamp": now(),
        "PROTOCOL_GATE": protocol,
        "METRIC_GATE": "PASS",
        "GEOMETRY_IMPLEMENTATION_GATE": "PASS",
        "NORM_MATCH_GATE": "NOT_RUN",
        "ANGLE_GATE": "NOT_RUN",
        "METHOD_DESIGN_READY": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "model_identity": "Qwen3.5-9B from existing canonical loader",
        "prompt_manifest": manifest,
        "teacher_forcing": True,
        "prefill_precision": "FP32 recurrent state",
        "canonical_R128": axis.config_geometry(R128_CFG),
        "reference_magnitude_schedule": "REAL_R128 per-token/per-layer/per-head residual norm",
        "intervention_granularity": "token x layer x head",
        "configs": CONFIGS,
        "random_seed": RANDOM_SEED,
        "degenerate_handling": "record degenerate; realdir orthogonal fallback to deterministic random only when r_perp degenerates",
        "write_back": "direct S_pre + r_intervention; no post-intervention requantization",
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
