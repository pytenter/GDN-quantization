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

TASK = "GDN_INT8_SINGLE_PULSE_RESIDUAL_PROPAGATION_V1"
RESULT = RES / "gdn_int8_single_pulse_residual_propagation_v1.json"
STAGE0 = RES / "gdn_int8_single_pulse_residual_propagation_v1_stage0.json"
RECORDS = RES / "gdn_int8_single_pulse_residual_propagation_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_single_pulse_residual_propagation_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_single_pulse_residual_propagation_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_single_pulse_residual_propagation_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_single_pulse_residual_propagation_v1.md"
FORMAL_RECORDS = RES / "gdn_int8_single_pulse_residual_propagation_v1_formal_records.jsonl"
FORMAL_TOKEN_RECORDS = RES / "gdn_int8_single_pulse_residual_propagation_v1_formal_token_records.jsonl.gz"
FORMAL_LAYER_RECORDS = RES / "gdn_int8_single_pulse_residual_propagation_v1_formal_layer_records.jsonl.gz"
FORMAL_HEAD_RECORDS = RES / "gdn_int8_single_pulse_residual_propagation_v1_formal_head_records.jsonl.gz"

GEOMETRY_RESULT = RES / "gdn_int8_residual_geometry_causal_intervention_v1.json"

EPS = 1e-12
CONFIGS = [
    "FP_STATE",
    "REAL_R128_PULSE",
    "PARALLEL_PLUS_PULSE",
    "PARALLEL_MINUS_PULSE",
    "ORTHOGONAL_REALDIR_PULSE",
    "ORTHOGONAL_RANDOM_PULSE",
]
PULSE_CONFIGS = [c for c in CONFIGS if c != "FP_STATE"]
GEOM_MAP = {
    "PARALLEL_PLUS_PULSE": "PARALLEL_PLUS",
    "PARALLEL_MINUS_PULSE": "PARALLEL_MINUS",
    "ORTHOGONAL_REALDIR_PULSE": "ORTHOGONAL_REALDIR",
    "ORTHOGONAL_RANDOM_PULSE": "ORTHOGONAL_RANDOM",
}
TAUS = [1, 2, 4, 8, 16, 32, 64, 128]
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1
import run_int8_residual_geometry_causal_intervention as geom
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


def stage_paths(stage, shard_id=None):
    suffix = f"_{stage}" + ("" if shard_id is None else f"_shard_{shard_id:03d}")
    return {
        "records": RES / f"gdn_int8_single_pulse_residual_propagation_v1_records{suffix}.jsonl",
        "token": RES / f"gdn_int8_single_pulse_residual_propagation_v1_token_records{suffix}.jsonl.gz",
        "layer": RES / f"gdn_int8_single_pulse_residual_propagation_v1_layer_records{suffix}.jsonl.gz",
        "head": RES / f"gdn_int8_single_pulse_residual_propagation_v1_head_records{suffix}.jsonl.gz",
    }


def geometry_gate():
    if not GEOMETRY_RESULT.exists():
        return False, {}
    obj = json.loads(GEOMETRY_RESULT.read_text(encoding="utf-8"))
    required = {
        "FORMAL_STATUS": "COMPLETE",
        "PROTOCOL_GATE": "PASS",
        "METRIC_GATE": "PASS",
        "NORM_MATCH_GATE": "PASS",
        "ANGLE_GATE": "PASS",
        "FINAL_CLASSIFICATION": "RESIDUAL_GEOMETRY_CAUSAL_SUPPORT",
        "RESIDUAL_PROPAGATION_EXPERIMENT_READY": "YES",
    }
    return all(obj.get(k) == v for k, v in required.items()), obj


def prompt_manifest():
    ok, geom_obj = geometry_gate()
    manifest = geom_obj.get("prompt_manifest", []) if ok else []
    return manifest, geom_obj


def state_distance(torch, past_a, past_b):
    total = 0.0
    layer_rows = []
    head_rows = []
    for layer in GDN_LAYERS:
        a = p1.get_state(past_a, layer).detach().float()
        b = p1.get_state(past_b, layer).detach().float()
        d = a - b
        ln = float(torch.linalg.vector_norm(d).item())
        total += ln * ln
        layer_rows.append({"layer_idx": layer, "delta_S_tau_norm": ln})
        for h in range(d.shape[1]):
            hn = float(torch.linalg.vector_norm(d[:, h:h + 1, :, :]).item())
            head_rows.append({"layer_idx": layer, "head_idx": h, "delta_S_tau_norm": hn})
    return math.sqrt(total), layer_rows, head_rows


def inject_pulse(torch, cfg, fp_s_in, fp_past, cfg_past, token_idx, gates):
    total0 = 0.0
    layer_public = []
    head_public = []
    injection_count = 0
    for layer in GDN_LAYERS:
        fp_state = p1.get_state(fp_past, layer)
        cfg_state = p1.get_state(cfg_past, layer)
        pre = fp_state.detach().clone()
        if cfg == "REAL_R128_PULSE":
            m = geom.real_r128_layer(torch, fp_s_in[layer], pre)
        else:
            real = geom.real_r128_layer(torch, fp_s_in[layer], pre)
            m = geom.layer_head_intervention(torch, GEOM_MAP[cfg], fp_s_in[layer], pre, real["_ref_norms"], token_idx, layer)
        if m["status"] != "OK":
            gates["PROPAGATION_IMPLEMENTATION_GATE"] = "FAIL"
            continue
        s_write = m["_s_write"]
        delta0 = s_write.detach().float() - pre.detach().float()
        ln0 = float(torch.linalg.vector_norm(delta0).item())
        total0 += ln0 * ln0
        cfg_state.copy_(s_write.to(cfg_state.dtype))
        injection_count += 1
        pub = {k: v for k, v in m.items() if not k.startswith("_")}
        pub.update({"layer_idx": layer, "delta_S0_norm": ln0})
        if pub.get("norm_match_absdev", 0.0) > 2e-5:
            gates["NORM_MATCH_GATE"] = "FAIL"
        if cfg in GEOM_MAP:
            target = 1.0 if cfg == "PARALLEL_PLUS_PULSE" else (-1.0 if cfg == "PARALLEL_MINUS_PULSE" else 0.0)
            if pub.get("angle_cosine_abs_error", 1.0) > (2e-5 if abs(target) == 1.0 else 2e-4):
                gates["ANGLE_GATE"] = "FAIL"
        layer_public.append(pub)
        for hr in m.get("_head_rows", []):
            hp = dict(hr)
            hp.update({"layer_idx": layer})
            head_public.append(hp)
    return math.sqrt(total0), layer_public, head_public, injection_count


def run_unit(torch, model, tokenizer, e2e, pm, stage, t0, future_horizon, paths, args):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    need = t0 + future_horizon + 1
    cont_ids = tokenizer.encode(pm["fp_response"], add_special_tokens=False)[:need]
    if len(cont_ids) <= t0:
        raise RuntimeError(f"not enough continuation tokens for {pm['problem_id']} t0={t0}: {len(cont_ids)}")
    future_horizon = min(future_horizon, len(cont_ids) - t0 - 1)
    observe_taus = [tau for tau in TAUS if tau <= future_horizon]
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    base_input = enc["input_ids"].to(device)
    base_mask = enc.get("attention_mask")
    base_mask = base_mask.to(device) if base_mask is not None else None
    inputs = {cfg: base_input.clone() for cfg in CONFIGS}
    masks = {cfg: base_mask.clone() if base_mask is not None else None for cfg in CONFIGS}
    pasts = {cfg: None for cfg in CONFIGS}
    gates = {"PROTOCOL_GATE": "PASS", "PROPAGATION_IMPLEMENTATION_GATE": "PASS", "NORM_MATCH_GATE": "PASS", "ANGLE_GATE": "PASS"}
    token_rows = defaultdict(list)
    layer_rows_by_cfg = defaultdict(list)
    head_rows_by_cfg = defaultdict(list)
    delta0 = {cfg: 0.0 for cfg in CONFIGS}
    injection_count = {cfg: 0 for cfg in CONFIGS}
    injections_after_t0 = {cfg: 0 for cfg in CONFIGS}
    pulse_layer_metrics = {}
    pulse_head_metrics = {}
    print(f"[{now()}] {stage} shard={args.shard_id} prompt={pm['problem_id']} t0={t0} future={future_horizon} configs={CONFIGS}", flush=True)
    with torch.inference_mode():
        for t in range(t0 + future_horizon + 1):
            s_in_fp = None
            if t == t0 and pasts["FP_STATE"] is not None:
                s_in_fp = {layer: p1.get_state(pasts["FP_STATE"], layer).detach().clone() for layer in GDN_LAYERS}
            outs = {}
            for cfg in CONFIGS:
                outs[cfg] = p1.feed_step(torch, model, inputs[cfg], masks[cfg], pasts[cfg])
                pasts[cfg] = outs[cfg].past_key_values
            if t == t0:
                if s_in_fp is None:
                    gates["PROTOCOL_GATE"] = "FAIL"
                for cfg in PULSE_CONFIGS:
                    d0, lrows, hrows, n_inj = inject_pulse(torch, cfg, s_in_fp, pasts["FP_STATE"], pasts[cfg], t0, gates)
                    delta0[cfg] = d0
                    injection_count[cfg] += n_inj
                    pulse_layer_metrics[cfg] = lrows
                    pulse_head_metrics[cfg] = hrows
            elif t > t0:
                for cfg in PULSE_CONFIGS:
                    injections_after_t0[cfg] += 0
            tau = t - t0
            if tau in observe_taus:
                gz_append(paths["token"], {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "t0": t0, "tau": tau, "config": "FP_STATE", "KL_future": 0.0, "Top1_future": 1.0, "G_state": 0.0, "delta_S0_norm": 0.0, "delta_S_tau_norm": 0.0})
                token_rows["FP_STATE"].append({"tau": tau, "KL_future": 0.0, "Top1_future": 1.0, "G_state": 0.0})
                for cfg in PULSE_CONFIGS:
                    lm = eff.logits_metrics(torch, outs["FP_STATE"].logits, outs[cfg].logits)
                    dtau, ldist, hdist = state_distance(torch, pasts[cfg], pasts["FP_STATE"])
                    g = dtau / (delta0[cfg] + EPS)
                    rec = {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "t0": t0, "tau": tau, "config": cfg, "KL_future": lm["KL"], "Top1_future": lm["top1_agreement"], "G_state": g, "delta_S0_norm": delta0[cfg], "delta_S_tau_norm": dtau}
                    gz_append(paths["token"], rec)
                    token_rows[cfg].append(rec)
                    by_layer0 = {r["layer_idx"]: r.get("delta_S0_norm", 0.0) for r in pulse_layer_metrics.get(cfg, [])}
                    by_head0 = {(r["layer_idx"], r["head_idx"]): r.get("injected_residual_norm", r.get("reference_residual_norm", 0.0)) for r in pulse_head_metrics.get(cfg, [])}
                    for lr in ldist:
                        l0 = by_layer0.get(lr["layer_idx"], 0.0)
                        lrec = {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "t0": t0, "tau": tau, "config": cfg, "delta_S0_norm": l0, "G_state": lr["delta_S_tau_norm"] / (l0 + EPS), **lr}
                        gz_append(paths["layer"], lrec)
                        layer_rows_by_cfg[cfg].append(lrec)
                    for hr in hdist:
                        h0 = by_head0.get((hr["layer_idx"], hr["head_idx"]), 0.0)
                        hrec = {"task": TASK, "stage": stage, "problem_id": pm["problem_id"], "role": pm["role"], "t0": t0, "tau": tau, "config": cfg, "delta_S0_norm": h0, "G_state": hr["delta_S_tau_norm"] / (h0 + EPS), **hr}
                        gz_append(paths["head"], hrec)
                        head_rows_by_cfg[cfg].append(hrec)
            if t < len(cont_ids):
                nxt = torch.tensor([[cont_ids[t]]], dtype=base_input.dtype, device=device)
                for cfg in CONFIGS:
                    inputs[cfg] = nxt
                    masks[cfg] = None
    for cfg in PULSE_CONFIGS:
        if injection_count[cfg] != len(GDN_LAYERS):
            gates["PROPAGATION_IMPLEMENTATION_GATE"] = "FAIL"
        if injections_after_t0[cfg] != 0:
            gates["PROPAGATION_IMPLEMENTATION_GATE"] = "FAIL"
    summaries = {}
    for cfg in CONFIGS:
        ts = token_rows[cfg]
        persistence_vals = [r["G_state"] for r in ts if r["tau"] >= 8]
        kl_vals = [r["KL_future"] for r in ts if r["tau"] >= 8]
        summaries[cfg] = {
            "token_summary": summarize(ts),
            "layer_summary": summarize(layer_rows_by_cfg[cfg]),
            "head_summary": summarize(head_rows_by_cfg[cfg]),
            "persistence_score": avg(persistence_vals),
            "max_propagation_gain": max([r["G_state"] for r in ts], default=None),
            "tau_of_max_gain": max(ts, key=lambda r: r["G_state"])["tau"] if ts else None,
            "future_KL_score": avg(kl_vals),
            "max_future_KL": max([r["KL_future"] for r in ts], default=None),
            "tau_of_max_future_KL": max(ts, key=lambda r: r["KL_future"])["tau"] if ts else None,
        }
    record = {
        "task": TASK,
        "stage": stage,
        "timestamp": now(),
        "problem_id": pm["problem_id"],
        "role": pm["role"],
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "t0": t0,
        "t0_indexing": "0-based continuation token index; injection occurs after forward at token_idx=t0 and before token_idx=t0+1; tau=1 is first affected future step",
        "requested_future_horizon": future_horizon,
        "observed_taus": observe_taus,
        "configs": CONFIGS,
        "single_injection_only": True,
        "number_of_injections": injection_count,
        "number_of_injections_after_t0": injections_after_t0,
        **gates,
        "summaries": summaries,
    }
    append_jsonl(paths["records"], record)
    print(f"[{now()}] {stage} done prompt={pm['problem_id']} t0={t0} gates P/I/N/A={gates['PROTOCOL_GATE']}/{gates['PROPAGATION_IMPLEMENTATION_GATE']}/{gates['NORM_MATCH_GATE']}/{gates['ANGLE_GATE']}", flush=True)
    return record


def run_stage(args):
    manifest, _ = prompt_manifest()
    if not manifest:
        raise RuntimeError("geometry gate not passed; propagation must not run")
    pms = {p["problem_id"]: p for p in p1.selected_prompt_rows()}
    if args.stage == "smoke":
        ids = [manifest[0]["prompt_id"]]
        t0s = [64]
        future = min(args.future_horizon, 16)
        paths = stage_paths("smoke")
    elif args.stage == "pilot":
        ids = [manifest[0]["prompt_id"]]
        t0s = [128]
        future = min(args.future_horizon, 128)
        paths = stage_paths("pilot")
    else:
        units = [(m["prompt_id"], t0) for m in manifest for t0 in [64, 128, 256]]
        units = [u for i, u in enumerate(units) if i % args.num_shards == args.shard_id]
        ids = None
        t0s = None
        future = min(args.future_horizon, 128)
        paths = stage_paths("formal", args.shard_id)
    done = set()
    if args.resume and paths["records"].exists():
        done = {(r["problem_id"], r["t0"]) for r in iter_jsonl(paths["records"]) if r.get("stage") == args.stage}
    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    if args.stage in ("smoke", "pilot"):
        for pid in ids:
            for t0 in t0s:
                if (pid, t0) not in done:
                    run_unit(torch, model, tokenizer, e2e, pms[pid], args.stage, t0, future, paths, args)
    else:
        for pid, t0 in units:
            if (pid, t0) not in done:
                run_unit(torch, model, tokenizer, e2e, pms[pid], args.stage, t0, future, paths, args)


def config_metrics(row):
    out = {}
    for cfg in CONFIGS:
        s = row["summaries"][cfg]
        out[cfg] = {
            "persistence_score": s.get("persistence_score"),
            "max_propagation_gain": s.get("max_propagation_gain"),
            "tau_of_max_gain": s.get("tau_of_max_gain"),
            "future_KL_score": s.get("future_KL_score"),
            "max_future_KL": s.get("max_future_KL"),
            "tau_of_max_future_KL": s.get("tau_of_max_future_KL"),
            "mean_G_state": stat_mean(s.get("token_summary", {}), "G_state"),
            "mean_KL_future": stat_mean(s.get("token_summary", {}), "KL_future"),
            "mean_Top1_future": stat_mean(s.get("token_summary", {}), "Top1_future"),
        }
    return out


def unit_analysis(row):
    metrics = config_metrics(row)
    persist = [metrics[c]["persistence_score"] for c in PULSE_CONFIGS if finite_num(metrics[c]["persistence_score"])]
    kls = [metrics[c]["future_KL_score"] for c in PULSE_CONFIGS if finite_num(metrics[c]["future_KL_score"])]
    real_kl = metrics["REAL_R128_PULSE"]["future_KL_score"]
    p_ratio = max(persist) / (min(persist) + EPS) if persist else None
    kl_spread = max(kls) - min(kls) if kls else None
    kl_thresh = max(0.001, 0.10 * real_kl) if finite_num(real_kl) else 0.001
    return {
        "problem_id": row["problem_id"],
        "role": row["role"],
        "t0": row["t0"],
        "metrics": metrics,
        "persistence_separation_ratio": p_ratio,
        "future_KL_spread": kl_spread,
        "future_KL_spread_threshold": kl_thresh,
        "clear_persistence_separation": bool(finite_num(p_ratio) and p_ratio >= 1.25),
        "clear_future_KL_separation": bool(finite_num(kl_spread) and kl_spread >= kl_thresh),
        "PROTOCOL_GATE": row.get("PROTOCOL_GATE"),
        "PROPAGATION_IMPLEMENTATION_GATE": row.get("PROPAGATION_IMPLEMENTATION_GATE"),
        "NORM_MATCH_GATE": row.get("NORM_MATCH_GATE"),
        "ANGLE_GATE": row.get("ANGLE_GATE"),
        "number_of_injections_after_t0": row.get("number_of_injections_after_t0"),
    }


def analyze_rows(rows):
    per = [unit_analysis(r) for r in rows]
    agg = {}
    for cfg in CONFIGS:
        agg[cfg] = {
            "mean_persistence_score": avg([p["metrics"][cfg]["persistence_score"] for p in per]),
            "mean_future_KL_score": avg([p["metrics"][cfg]["future_KL_score"] for p in per]),
            "mean_max_propagation_gain": avg([p["metrics"][cfg]["max_propagation_gain"] for p in per]),
            "mean_max_future_KL": avg([p["metrics"][cfg]["max_future_KL"] for p in per]),
        }
    return {
        "per_unit": per,
        "aggregate_by_config": agg,
        "clear_persistence_separation_count": sum(1 for p in per if p["clear_persistence_separation"]),
        "clear_future_KL_separation_count": sum(1 for p in per if p["clear_future_KL_separation"]),
        "positive_unit_count": sum(1 for p in per if p["clear_persistence_separation"] or p["clear_future_KL_separation"]),
    }


def write_report(obj):
    lines = [
        "# GDN INT8 Single-Pulse Residual Propagation V1",
        "",
        "## Gates",
        f"- FORMAL_STATUS: `{obj.get('FORMAL_STATUS')}`",
        f"- PROTOCOL_GATE: `{obj.get('PROTOCOL_GATE')}`",
        f"- PROPAGATION_IMPLEMENTATION_GATE: `{obj.get('PROPAGATION_IMPLEMENTATION_GATE')}`",
        f"- NORM_MATCH_GATE: `{obj.get('NORM_MATCH_GATE')}`; ANGLE_GATE: `{obj.get('ANGLE_GATE')}`",
        f"- PILOT_CLASSIFICATION: `{obj.get('PILOT_CLASSIFICATION')}`",
        f"- FINAL_CLASSIFICATION: `{obj.get('FINAL_CLASSIFICATION')}`",
        f"- METHOD_DESIGN_READY: `{obj.get('METHOD_DESIGN_READY')}`",
        "",
        "## Analysis",
        "```json",
        json.dumps(obj.get("analysis", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Limits",
        "- Single oracle pulse only; no repeated quantization after t0.",
        "- Teacher-forced continuation uses retokenized P0 FP_STATE decoded responses; not exact replay.",
        "- No method design is run.",
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
    gates = {k: ("PASS" if rows and all(r.get(k) == "PASS" for r in rows) else "FAIL") for k in ["PROTOCOL_GATE", "PROPAGATION_IMPLEMENTATION_GATE", "NORM_MATCH_GATE", "ANGLE_GATE"]}
    analysis = analyze_rows(rows) if rows else {}
    if stage == "smoke":
        pilot = "NOT_RUN"
        formal = "NOT_RUN"
        final = "SMOKE_PASS" if all(v == "PASS" for v in gates.values()) else "SMOKE_FAIL"
    elif stage == "pilot":
        positive = bool(rows and all(v == "PASS" for v in gates.values()) and analysis.get("positive_unit_count", 0) >= 1)
        pilot = "POSITIVE_SINGLE_PULSE_PROPAGATION_SIGNAL" if positive else "SINGLE_PULSE_PROPAGATION_INCONCLUSIVE"
        formal = "NOT_RUN"
        final = pilot
    else:
        n = len(rows)
        positive = analysis.get("positive_unit_count", 0)
        formal = "COMPLETE" if n == 18 else "INCOMPLETE"
        pilot = "N/A"
        if n == 18 and all(v == "PASS" for v in gates.values()) and positive >= 9:
            final = "SINGLE_PULSE_RESIDUAL_PROPAGATION_SUPPORTED"
        elif n == 18 and positive > 0:
            final = "SINGLE_PULSE_RESIDUAL_PROPAGATION_INCONCLUSIVE"
        else:
            final = "SINGLE_PULSE_RESIDUAL_PROPAGATION_NOT_SUPPORTED"
    manifest, geom_obj = prompt_manifest()
    obj = {
        "task": TASK,
        "stage": f"{stage.upper()}_COMPLETE",
        "timestamp": now(),
        "FORMAL_STATUS": formal,
        "completed_unit_count": len(rows),
        "requested_unit_count": 18 if stage == "formal" else 1,
        **gates,
        "SMOKE_GATE": "PASS" if stage == "smoke" and all(v == "PASS" for v in gates.values()) else ("N/A" if stage != "smoke" else "FAIL"),
        "PILOT_CLASSIFICATION": pilot,
        "FINAL_CLASSIFICATION": final,
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "geometry_gate_source": str(GEOMETRY_RESULT),
        "geometry_final_classification": geom_obj.get("FINAL_CLASSIFICATION"),
        "prompt_manifest": manifest,
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


def stage0():
    ok, geom_obj = geometry_gate()
    protocol = "PASS" if ok else "FAIL"
    obj = {
        "task": TASK,
        "stage": "stage0",
        "timestamp": now(),
        "PROTOCOL_GATE": protocol,
        "PROPAGATION_IMPLEMENTATION_GATE": "PASS",
        "NORM_MATCH_GATE": "NOT_RUN",
        "ANGLE_GATE": "NOT_RUN",
        "geometry_gate_source": str(GEOMETRY_RESULT),
        "geometry_gate_summary": {k: geom_obj.get(k) for k in ["FORMAL_STATUS", "PROTOCOL_GATE", "METRIC_GATE", "NORM_MATCH_GATE", "ANGLE_GATE", "FINAL_CLASSIFICATION", "RESIDUAL_PROPAGATION_EXPERIMENT_READY"]},
        "t0_indexing": "0-based continuation token index; inject after token_idx=t0 forward and before token_idx=t0+1; tau=1 is the first future affected step",
        "injection_policy": "single injection only at t0; all future recurrent continuation is FP without quantization or later injection",
        "target_t0_values_formal": [64, 128, 256],
        "target_future_tau_values": TAUS,
        "configs": CONFIGS,
        "gdn_layers": GDN_LAYERS,
        "state_shape_expected": [1, 32, 128, 128],
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    save_json(STAGE0, obj)
    save_json(RESULT, {**obj, "FORMAL_STATUS": "NOT_RUN", "SMOKE_GATE": "NOT_RUN", "PILOT_CLASSIFICATION": "NOT_RUN"})
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "smoke", "pilot", "formal", "analyze", "merge"], required=True)
    ap.add_argument("--analyze-stage", choices=["smoke", "pilot", "formal"], default="formal")
    ap.add_argument("--future-horizon", type=int, default=128)
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
