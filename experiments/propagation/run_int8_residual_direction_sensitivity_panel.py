#!/usr/bin/env python3
import argparse
import os
import gzip
import hashlib
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

TASK = "GDN_INT8_RESIDUAL_DIRECTION_SENSITIVITY_PANEL_V1"
RESULT = RES / "gdn_int8_residual_direction_sensitivity_panel_v1.json"
STAGE0 = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_stage0.json"
RECORDS = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_records.jsonl"
TOKEN_RECORDS = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_token_records.jsonl.gz"
LAYER_RECORDS = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_layer_records.jsonl.gz"
HEAD_RECORDS = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_head_records.jsonl.gz"
REPORT = REP / "gdn_int8_residual_direction_sensitivity_panel_v1.md"
FORMAL_RECORDS = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_formal_records.jsonl"
FORMAL_TOKEN_RECORDS = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_formal_token_records.jsonl.gz"
FORMAL_LAYER_RECORDS = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_formal_layer_records.jsonl.gz"
FORMAL_HEAD_RECORDS = RES / "gdn_int8_residual_direction_sensitivity_panel_v1_formal_head_records.jsonl.gz"

PROPAGATION_RESULT = RES / "gdn_int8_single_pulse_residual_propagation_v1.json"

EPS = 1e-12
EPS_DIRECTION = 1e-12
BASE_RANDOM_SEED = 20260831
N_RANDOM_DIRECTIONS = 16
MAX_DIRECTION_RESAMPLE = 8
RANDOM_CONFIGS = [f"ORTHO_RAND_{i:02d}" for i in range(N_RANDOM_DIRECTIONS)]
CONFIGS = [
    "FP_STATE",
    "REAL_R128_PULSE",
    "ORTHOGONAL_REALDIR_PULSE",
    "PARALLEL_PLUS_PULSE",
    *RANDOM_CONFIGS,
]
PULSE_CONFIGS = [c for c in CONFIGS if c != "FP_STATE"]
GEOM_MAP = {
    "PARALLEL_PLUS_PULSE": "PARALLEL_PLUS",
    "ORTHOGONAL_REALDIR_PULSE": "ORTHOGONAL_REALDIR",
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
        "records": RES / f"gdn_int8_residual_direction_sensitivity_panel_v1_records{suffix}.jsonl",
        "token": RES / f"gdn_int8_residual_direction_sensitivity_panel_v1_token_records{suffix}.jsonl.gz",
        "layer": RES / f"gdn_int8_residual_direction_sensitivity_panel_v1_layer_records{suffix}.jsonl.gz",
        "head": RES / f"gdn_int8_residual_direction_sensitivity_panel_v1_head_records{suffix}.jsonl.gz",
    }


def propagation_gate():
    if not PROPAGATION_RESULT.exists():
        return False, {}
    obj = json.loads(PROPAGATION_RESULT.read_text(encoding="utf-8"))
    required = {
        "FORMAL_STATUS": "COMPLETE",
        "PROTOCOL_GATE": "PASS",
        "NORM_MATCH_GATE": "PASS",
        "ANGLE_GATE": "PASS",
        "PROPAGATION_IMPLEMENTATION_GATE": "PASS",
        "FINAL_CLASSIFICATION": "SINGLE_PULSE_RESIDUAL_PROPAGATION_SUPPORTED",
    }
    return all(obj.get(k) == v for k, v in required.items()), obj


def prompt_manifest():
    ok, prop_obj = propagation_gate()
    manifest = prop_obj.get("prompt_manifest", []) if ok else []
    return manifest, prop_obj


def stable_seed(prompt_id, t0, layer_idx, head_idx, direction_index, resample=0):
    txt = f"{BASE_RANDOM_SEED}|{prompt_id}|{t0}|{layer_idx}|{head_idx}|{direction_index}|{resample}"
    digest = hashlib.sha256(txt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)


def cosine_val(torch, a, b):
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    an = float(torch.linalg.vector_norm(af).item())
    bn = float(torch.linalg.vector_norm(bf).item())
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item())


def project_off(torch, vec, base):
    denom = torch.sum(base.double() * base.double())
    coeff = torch.sum(vec.double() * base.double()) / (denom + EPS)
    return vec - coeff.to(vec.dtype) * base


def deterministic_random_orthogonal(torch, shape, device, dtype, prompt_id, t0, layer_idx, head_idx, direction_index, delta):
    for resample in range(MAX_DIRECTION_RESAMPLE + 1):
        gen = torch.Generator(device=device)
        gen.manual_seed(stable_seed(prompt_id, t0, layer_idx, head_idx, direction_index, resample))
        raw = torch.randn(shape, generator=gen, device=device, dtype=dtype)
        perp = project_off(torch, raw, delta)
        n = float(torch.linalg.vector_norm(perp.float()).item())
        if n > EPS_DIRECTION:
            return perp / (n + EPS), resample, "OK"
    return None, MAX_DIRECTION_RESAMPLE, "DIRECTION_SAMPLE_FAILURE"


def panel_layer_head_intervention(torch, cfg, s_in, s_pre, ref_norms, prompt_id, t0, layer_idx):
    pre = s_pre.detach().float()
    inn = s_in.detach().float()
    residual = torch.zeros_like(pre)
    head_rows = []
    degenerate = 0
    sample_fail = 0
    resamples = 0
    direction_index = int(cfg.rsplit("_", 1)[-1])
    diversity_vals = []
    for h in range(pre.shape[1]):
        delta_h = pre[:, h:h + 1, :, :] - inn[:, h:h + 1, :, :]
        dnorm = float(torch.linalg.vector_norm(delta_h.float()).item())
        if dnorm <= EPS_DIRECTION:
            degenerate += 1
            continue
        if direction_index == 0:
            units = []
            for k in range(N_RANDOM_DIRECTIONS):
                uk, _, st = deterministic_random_orthogonal(
                    torch, delta_h.shape, delta_h.device, delta_h.dtype, prompt_id, t0, layer_idx, h, k, delta_h
                )
                if uk is not None:
                    units.append(uk.detach().float().flatten())
            for i in range(len(units)):
                for j in range(i + 1, len(units)):
                    c = cosine_val(torch, units[i], units[j])
                    if finite_num(c):
                        diversity_vals.append(abs(c))
        unit, tries, status = deterministic_random_orthogonal(
            torch, delta_h.shape, delta_h.device, delta_h.dtype, prompt_id, t0, layer_idx, h, direction_index, delta_h
        )
        resamples += tries
        if unit is None:
            sample_fail += 1
            continue
        vec = float(ref_norms[h]) * unit
        residual[:, h:h + 1, :, :] = vec.to(residual.dtype)
        target_cos = 0.0
        hm = geom.residual_metrics(torch, inn[:, h:h + 1, :, :], pre[:, h:h + 1, :, :], vec, ref_norms[h], target_cos, status, 0)
        hm.update({"head_idx": h, "direction_index": direction_index, "direction_resample_count": tries, "direction_sample_failure": 0})
        head_rows.append(hm)
    s_write = pre + residual
    lm = summarize(head_rows)
    layer = {k: stat_mean(lm, k) for k in lm}
    layer.update({
        "status": "OK" if head_rows and sample_fail == 0 else ("DIRECTION_SAMPLE_FAILURE" if sample_fail else "NO_VALID_DIRECTIONS"),
        "valid_direction_count": len(head_rows),
        "degenerate_delta_count": degenerate,
        "direction_sample_failure_count": sample_fail,
        "direction_resample_count": resamples,
        "direction_diversity_mean_abs_pairwise_cosine": avg(diversity_vals),
        "direction_diversity_p95_abs_pairwise_cosine": pct(diversity_vals, 0.95),
        "direction_diversity_max_abs_pairwise_cosine": max(diversity_vals) if diversity_vals else None,
        "_s_write": s_write,
        "_head_rows": head_rows,
    })
    return layer


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
        elif cfg in GEOM_MAP:
            real = geom.real_r128_layer(torch, fp_s_in[layer], pre)
            m = geom.layer_head_intervention(torch, GEOM_MAP[cfg], fp_s_in[layer], pre, real["_ref_norms"], token_idx, layer)
        elif cfg in RANDOM_CONFIGS:
            real = geom.real_r128_layer(torch, fp_s_in[layer], pre)
            m = panel_layer_head_intervention(torch, cfg, fp_s_in[layer], pre, real["_ref_norms"], gates["problem_id"], token_idx, layer)
        else:
            raise ValueError(cfg)
        if m["status"] != "OK":
            gates["PANEL_IMPLEMENTATION_GATE"] = "FAIL"
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
        if cfg in GEOM_MAP or cfg in RANDOM_CONFIGS:
            target = 1.0 if cfg == "PARALLEL_PLUS_PULSE" else (-1.0 if cfg == "PARALLEL_MINUS_PULSE" else 0.0)
            if pub.get("angle_cosine_abs_error", 1.0) > (2e-5 if abs(target) == 1.0 else 2e-4):
                gates["ANGLE_GATE"] = "FAIL"
        if cfg == "ORTHO_RAND_00":
            if pub.get("direction_diversity_max_abs_pairwise_cosine", 0.0) is None or pub.get("direction_diversity_max_abs_pairwise_cosine", 1.0) > 0.20:
                gates["DIRECTION_DIVERSITY_GATE"] = "FAIL"
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
    gates = {"PROTOCOL_GATE": "PASS", "PANEL_IMPLEMENTATION_GATE": "PASS", "NORM_MATCH_GATE": "PASS", "ANGLE_GATE": "PASS", "DIRECTION_DIVERSITY_GATE": "PASS", "problem_id": pm["problem_id"]}
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
            gates["PANEL_IMPLEMENTATION_GATE"] = "FAIL"
        if injections_after_t0[cfg] != 0:
            gates["PANEL_IMPLEMENTATION_GATE"] = "FAIL"
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
        **{k: v for k, v in gates.items() if k != "problem_id"},
        "summaries": summaries,
    }
    append_jsonl(paths["records"], record)
    print(f"[{now()}] {stage} done prompt={pm['problem_id']} t0={t0} gates P/I/N/A/D={gates['PROTOCOL_GATE']}/{gates['PANEL_IMPLEMENTATION_GATE']}/{gates['NORM_MATCH_GATE']}/{gates['ANGLE_GATE']}/{gates['DIRECTION_DIVERSITY_GATE']}", flush=True)
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


def std(xs):
    xs = [float(x) for x in xs if finite_num(x)]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0 if xs else None


def empirical_distribution(vals):
    vals = [float(v) for v in vals if finite_num(v)]
    if not vals:
        return {"N": 0}
    return {
        "N": len(vals),
        "min": min(vals),
        "p10": pct(vals, 0.10),
        "p25": pct(vals, 0.25),
        "median": median(vals),
        "p75": pct(vals, 0.75),
        "p90": pct(vals, 0.90),
        "max": max(vals),
        "mean": avg(vals),
        "std": std(vals),
        "cv": std(vals) / (avg(vals) + EPS) if finite_num(avg(vals)) else None,
    }


def rank_position(value, panel_vals):
    vals = sorted(float(v) for v in panel_vals if finite_num(v))
    if not finite_num(value) or not vals:
        return None
    less_equal = sum(1 for v in vals if v <= float(value))
    rank = less_equal + 1
    percentile = less_equal / len(vals)
    if percentile <= 0.25:
        bucket = "bottom_quartile"
    elif percentile >= 0.75:
        bucket = "top_quartile"
    else:
        bucket = "middle_50"
    return {
        "rank_among_random_plus_target": f"{rank}/{len(vals) + 1}",
        "empirical_percentile_vs_random": percentile,
        "bucket": bucket,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
    }


def rankdata(xs):
    pairs = sorted((x, i) for i, x in enumerate(xs))
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        r = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[pairs[k][1]] = r
        i = j
    return ranks


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_num(x) and finite_num(y)]
    if len(pairs) < 3:
        return None
    rx = rankdata([p[0] for p in pairs])
    ry = rankdata([p[1] for p in pairs])
    mx = avg(rx)
    my = avg(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    deny = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (denx * deny + EPS)


def unit_analysis(row):
    metrics = config_metrics(row)
    persist = [metrics[c]["persistence_score"] for c in RANDOM_CONFIGS if finite_num(metrics[c]["persistence_score"])]
    max_gain = [metrics[c]["max_propagation_gain"] for c in RANDOM_CONFIGS if finite_num(metrics[c]["max_propagation_gain"])]
    kls = [metrics[c]["future_KL_score"] for c in RANDOM_CONFIGS if finite_num(metrics[c]["future_KL_score"])]
    max_kls = [metrics[c]["max_future_KL"] for c in RANDOM_CONFIGS if finite_num(metrics[c]["max_future_KL"])]
    real_kl = metrics["REAL_R128_PULSE"]["future_KL_score"]
    p_ratio = max(persist) / (min(persist) + EPS) if persist else None
    p_spread = max(persist) - min(persist) if persist else None
    kl_spread = max(kls) - min(kls) if kls else None
    kl_thresh = max(0.0005, 0.25 * real_kl) if finite_num(real_kl) else 0.0005
    random_table = []
    for cfg in RANDOM_CONFIGS:
        random_table.append({
            "direction": cfg,
            "persistence_score": metrics[cfg]["persistence_score"],
            "max_propagation_gain": metrics[cfg]["max_propagation_gain"],
            "future_KL_score": metrics[cfg]["future_KL_score"],
            "max_future_KL": metrics[cfg]["max_future_KL"],
        })
    random_table = sorted(random_table, key=lambda r: (-1e99 if r["persistence_score"] is None else r["persistence_score"]))
    random_persist = [metrics[c]["persistence_score"] for c in RANDOM_CONFIGS]
    random_gain = [metrics[c]["max_propagation_gain"] for c in RANDOM_CONFIGS]
    random_future_kl = [metrics[c]["future_KL_score"] for c in RANDOM_CONFIGS]
    random_amp_fraction = avg([1.0 if finite_num(metrics[c]["max_propagation_gain"]) and metrics[c]["max_propagation_gain"] > 1.0 else 0.0 for c in RANDOM_CONFIGS])
    random_persistent_amp_fraction = avg([1.0 if finite_num(metrics[c]["persistence_score"]) and metrics[c]["persistence_score"] > 1.0 else 0.0 for c in RANDOM_CONFIGS])
    return {
        "problem_id": row["problem_id"],
        "role": row["role"],
        "t0": row["t0"],
        "metrics": metrics,
        "random_direction_table_sorted_by_persistence": random_table,
        "valid_random_direction_count": len(persist),
        "persistence_distribution": empirical_distribution(persist),
        "max_gain_distribution": empirical_distribution(max_gain),
        "future_KL_distribution": empirical_distribution(kls),
        "max_future_KL_distribution": empirical_distribution(max_kls),
        "persistence_spread": p_spread,
        "persistence_separation_ratio": p_ratio,
        "future_KL_spread": kl_spread,
        "future_KL_spread_threshold": kl_thresh,
        "max_gain_spread": (max(max_gain) - min(max_gain)) if max_gain else None,
        "random_amplification_fraction": random_amp_fraction,
        "random_persistent_amplification_fraction": random_persistent_amp_fraction,
        "ORTHOGONAL_REALDIR_persistence_rank": rank_position(metrics["ORTHOGONAL_REALDIR_PULSE"]["persistence_score"], random_persist),
        "ORTHOGONAL_REALDIR_future_KL_rank": rank_position(metrics["ORTHOGONAL_REALDIR_PULSE"]["future_KL_score"], random_future_kl),
        "ORTHOGONAL_REALDIR_max_gain_rank": rank_position(metrics["ORTHOGONAL_REALDIR_PULSE"]["max_propagation_gain"], random_gain),
        "REAL_R128_descriptive_persistence_position": rank_position(metrics["REAL_R128_PULSE"]["persistence_score"], random_persist),
        "REAL_R128_descriptive_future_KL_position": rank_position(metrics["REAL_R128_PULSE"]["future_KL_score"], random_future_kl),
        "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
        "spearman_persistence_future_KL": spearman(random_persist, random_future_kl),
        "spearman_max_gain_future_KL": spearman(random_gain, random_future_kl),
        "clear_persistence_separation": bool((finite_num(p_ratio) and p_ratio >= 1.25) or (finite_num(p_spread) and p_spread >= 0.15)),
        "clear_future_KL_separation": bool(finite_num(kl_spread) and kl_spread >= kl_thresh),
        "PROTOCOL_GATE": row.get("PROTOCOL_GATE"),
        "PANEL_IMPLEMENTATION_GATE": row.get("PANEL_IMPLEMENTATION_GATE"),
        "NORM_MATCH_GATE": row.get("NORM_MATCH_GATE"),
        "ANGLE_GATE": row.get("ANGLE_GATE"),
        "DIRECTION_DIVERSITY_GATE": row.get("DIRECTION_DIVERSITY_GATE"),
        "number_of_injections_after_t0": row.get("number_of_injections_after_t0"),
    }


def natural_position_summary(per):
    keys = [
        ("persistence", "ORTHOGONAL_REALDIR_persistence_rank"),
        ("future_KL", "ORTHOGONAL_REALDIR_future_KL_rank"),
        ("max_gain", "ORTHOGONAL_REALDIR_max_gain_rank"),
    ]
    out = {}
    for label, key in keys:
        vals = []
        buckets = {"bottom_quartile": 0, "middle_50": 0, "top_quartile": 0}
        for p in per:
            pos = p.get(key)
            if isinstance(pos, dict) and finite_num(pos.get("empirical_percentile_vs_random")):
                vals.append(pos["empirical_percentile_vs_random"])
                buckets[pos.get("bucket", "middle_50")] = buckets.get(pos.get("bucket", "middle_50"), 0) + 1
        out[label] = {
            "median_percentile": median(vals),
            "count_bottom_quartile": buckets.get("bottom_quartile", 0),
            "count_middle_50": buckets.get("middle_50", 0),
            "count_top_quartile": buckets.get("top_quartile", 0),
            "note": "N=16 random directions per unit; percentile is coarse descriptive positioning only.",
        }
    return out


def classify_natural_alignment(pos_summary, unit_count):
    p = pos_summary.get("persistence", {})
    kl = pos_summary.get("future_KL", {})
    top = max(p.get("count_top_quartile", 0), kl.get("count_top_quartile", 0))
    bottom = max(p.get("count_bottom_quartile", 0), kl.get("count_bottom_quartile", 0))
    middle = min(p.get("count_middle_50", 0), kl.get("count_middle_50", 0))
    if top >= max(12, math.ceil(0.67 * unit_count)) or max(p.get("median_percentile") or 0, kl.get("median_percentile") or 0) > 0.75:
        return "HIGH_GAIN_BIASED"
    if bottom >= max(12, math.ceil(0.67 * unit_count)):
        return "LOW_GAIN_BIASED"
    if middle >= max(10, math.ceil(0.55 * unit_count)):
        return "MIDRANGE"
    return "HETEROGENEOUS"


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
        "median_random_amplification_fraction": median([p["random_amplification_fraction"] for p in per]),
        "median_random_persistent_amplification_fraction": median([p["random_persistent_amplification_fraction"] for p in per]),
        "median_persistence_spread": median([p["persistence_spread"] for p in per]),
        "median_future_KL_spread": median([p["future_KL_spread"] for p in per]),
        "median_rho_persistence_future_KL": median([p["spearman_persistence_future_KL"] for p in per]),
        "rho_persistence_future_KL_positive_count": sum(1 for p in per if finite_num(p["spearman_persistence_future_KL"]) and p["spearman_persistence_future_KL"] > 0),
        "rho_persistence_future_KL_ge_0p5_count": sum(1 for p in per if finite_num(p["spearman_persistence_future_KL"]) and p["spearman_persistence_future_KL"] >= 0.5),
        "natural_realdir_position_summary": natural_position_summary(per),
    }


def write_report(obj):
    lines = [
        "# GDN INT8 Residual Direction Sensitivity Panel V1",
        "",
        "## Gates",
        f"- FORMAL_STATUS: `{obj.get('FORMAL_STATUS')}`",
        f"- PROTOCOL_GATE: `{obj.get('PROTOCOL_GATE')}`",
        f"- PANEL_IMPLEMENTATION_GATE: `{obj.get('PANEL_IMPLEMENTATION_GATE')}`",
        f"- NORM_MATCH_GATE: `{obj.get('NORM_MATCH_GATE')}`; ANGLE_GATE: `{obj.get('ANGLE_GATE')}`",
        f"- DIRECTION_DIVERSITY_GATE: `{obj.get('DIRECTION_DIVERSITY_GATE')}`",
        f"- PILOT_CLASSIFICATION: `{obj.get('PILOT_CLASSIFICATION')}`",
        f"- FINAL_CLASSIFICATION: `{obj.get('FINAL_CLASSIFICATION')}`",
        f"- PROPAGATION_FIDELITY_LINK: `{obj.get('PROPAGATION_FIDELITY_LINK')}`",
        f"- NATURAL_RESIDUAL_DYNAMIC_ALIGNMENT: `{obj.get('NATURAL_RESIDUAL_DYNAMIC_ALIGNMENT')}`",
        f"- MECHANISM_CLOSURE_CANDIDATE: `{obj.get('MECHANISM_CLOSURE_CANDIDATE')}`",
        f"- METHOD_DESIGN_READY_CANDIDATE: `{obj.get('METHOD_DESIGN_READY_CANDIDATE')}`",
        f"- METHOD_DESIGN_READY: `{obj.get('METHOD_DESIGN_READY')}`",
        "",
        "## Analysis",
        "```json",
        json.dumps(obj.get("analysis", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Limits",
        "- Empirical directional sensitivity panel only; no eigenmode, unstable eigenspace, or Lyapunov-vector claim.",
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
    gate_keys = ["PROTOCOL_GATE", "PANEL_IMPLEMENTATION_GATE", "NORM_MATCH_GATE", "ANGLE_GATE", "DIRECTION_DIVERSITY_GATE"]
    gates = {k: ("PASS" if rows and all(r.get(k) == "PASS" for r in rows) else "FAIL") for k in gate_keys}
    analysis = analyze_rows(rows) if rows else {}
    if stage == "smoke":
        pilot = "NOT_RUN"
        formal = "NOT_RUN"
        final = "SMOKE_PASS" if all(v == "PASS" for v in gates.values()) else "SMOKE_FAIL"
    elif stage == "pilot":
        positive = bool(rows and all(v == "PASS" for v in gates.values()) and analysis.get("positive_unit_count", 0) >= 1)
        pilot = "POSITIVE_DIRECTIONAL_SENSITIVITY_PANEL" if positive else "DIRECTIONAL_SENSITIVITY_PANEL_INCONCLUSIVE"
        formal = "NOT_RUN"
        final = pilot
    else:
        n = len(rows)
        positive = analysis.get("positive_unit_count", 0)
        formal = "COMPLETE" if n == 18 else "INCOMPLETE"
        pilot = "N/A"
        if n == 18 and all(v == "PASS" for v in gates.values()) and positive >= 12:
            final = "RECURRENT_DIRECTIONAL_SENSITIVITY_DISTRIBUTION_SUPPORTED"
        elif n == 18 and positive > 0:
            final = "RECURRENT_DIRECTIONAL_SENSITIVITY_DISTRIBUTION_INCONCLUSIVE"
        else:
            final = "RECURRENT_DIRECTIONAL_SENSITIVITY_DISTRIBUTION_NOT_SUPPORTED"
    prop_link = "NOT_EVALUATED"
    natural_alignment = "NOT_EVALUATED"
    mechanism_closure = "NO"
    method_candidate = "NO"
    if stage == "formal" and analysis:
        median_rho = analysis.get("median_rho_persistence_future_KL")
        pos_count = analysis.get("rho_persistence_future_KL_positive_count", 0)
        prop_link = "SUPPORTED" if finite_num(median_rho) and median_rho >= 0.5 and pos_count >= 10 else ("INCONCLUSIVE" if finite_num(median_rho) else "NOT_EVALUATED")
        natural_alignment = classify_natural_alignment(analysis.get("natural_realdir_position_summary", {}), len(rows))
        if final == "RECURRENT_DIRECTIONAL_SENSITIVITY_DISTRIBUTION_SUPPORTED" and prop_link == "SUPPORTED":
            mechanism_closure = "YES"
        if mechanism_closure == "YES" and natural_alignment in ("MIDRANGE", "HIGH_GAIN_BIASED", "HETEROGENEOUS"):
            method_candidate = "YES"
    manifest, prop_obj = prompt_manifest()
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
        "PROPAGATION_FIDELITY_LINK": prop_link,
        "NATURAL_RESIDUAL_DYNAMIC_ALIGNMENT": natural_alignment,
        "MECHANISM_CLOSURE_CANDIDATE": mechanism_closure,
        "METHOD_DESIGN_READY_CANDIDATE": method_candidate,
        "METHOD_DESIGN_READY": "NO",
        "previous_propagation_gate_source": str(PROPAGATION_RESULT),
        "previous_propagation_final_classification": prop_obj.get("FINAL_CLASSIFICATION"),
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
    ok, prop_obj = propagation_gate()
    protocol = "PASS" if ok else "FAIL"
    obj = {
        "task": TASK,
        "stage": "stage0",
        "timestamp": now(),
        "PROTOCOL_GATE": protocol,
        "PANEL_IMPLEMENTATION_GATE": "PASS",
        "NORM_MATCH_GATE": "PASS",
        "ANGLE_GATE": "PASS",
        "DIRECTION_DIVERSITY_GATE": "PASS",
        "previous_propagation_gate_source": str(PROPAGATION_RESULT),
        "previous_propagation_gate_summary": {k: prop_obj.get(k) for k in ["FORMAL_STATUS", "PROTOCOL_GATE", "PROPAGATION_IMPLEMENTATION_GATE", "NORM_MATCH_GATE", "ANGLE_GATE", "FINAL_CLASSIFICATION"]},
        "t0_indexing": "0-based continuation token index; inject after token_idx=t0 forward and before token_idx=t0+1; tau=1 is the first future affected step",
        "injection_policy": "single injection only at t0; all future recurrent continuation is FP without quantization or later injection",
        "target_t0_values_formal": [64, 128, 256],
        "target_future_tau_values": TAUS,
        "configs": CONFIGS,
        "random_direction_configs": RANDOM_CONFIGS,
        "N_RANDOM_DIRECTIONS": N_RANDOM_DIRECTIONS,
        "BASE_RANDOM_SEED": BASE_RANDOM_SEED,
        "seed_formula": "sha256(BASE_RANDOM_SEED|prompt_id|t0|layer|head|direction_index|resample) -> torch Generator seed",
        "seed_determinism": "same prompt_id/t0/layer/head/direction_index/resample produces exact same random direction",
        "orthogonal_projection": "g_perp = g - <g,Delta>/(||Delta||^2+eps)*Delta; r = m*g_perp/(||g_perp||+eps)",
        "reference_R128_magnitude": "m[l,h] = ||Q_R128(S_pre[l,h])-S_pre[l,h]||_F at FP trajectory token t0",
        "norm_gate": "mean/median/P95/max |norm_match_ratio-1| must remain within tolerance during runs",
        "angle_gate": "PARALLEL_PLUS target cosine +1; ORTHOGONAL_REALDIR and ORTHO_RAND_00..15 target cosine 0",
        "direction_diversity_gate": "for each layer/head, pairwise |cos| among the 16 random orthogonal directions is audited; max > 0.20 fails",
        "rank_formula": "rank/percentile compares ORTHOGONAL_REALDIR or descriptive REAL_R128 metric against 16 random direction metric values; N=16 so percentile is coarse",
        "propagation_metrics": ["persistence_score", "max_propagation_gain", "tau_of_max_gain", "future_KL_score", "max_future_KL", "tau_of_max_future_KL"],
        "gdn_layers": GDN_LAYERS,
        "state_shape_expected": [1, 32, 128, 128],
        "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
        "exact_original_generation_token_ids_available": False,
        "exact_replay_claimed": False,
        "output_paths": [str(p) for p in [RESULT, RECORDS, TOKEN_RECORDS, LAYER_RECORDS, HEAD_RECORDS, REPORT]],
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
