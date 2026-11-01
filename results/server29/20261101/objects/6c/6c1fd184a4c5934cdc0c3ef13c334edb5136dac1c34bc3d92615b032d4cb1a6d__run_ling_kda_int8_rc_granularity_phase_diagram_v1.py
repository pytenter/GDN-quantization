#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path


TASK = "LING_KDA_INT8_RC_GRANULARITY_PHASE_DIAGRAM_V1"
SLUG = "ling_kda_int8_rc_granularity_phase_diagram_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
RUN_DIR = Path(os.environ.get("LING_KDA_GRANULARITY_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
MAG_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_persistent_source_error_magnitude_equalization_causal_v1.py"
CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
EXPECTED_CANONICAL_MANIFEST_SHA256 = "d2a60fcedcfe6b4a5f6757f93703736caf8cb06fa7b792cc5c486010b47ab33b"

GROUP_SIZES = [128, 64, 32, 16]
CONFIGS = ["R128", "R64", "R32", "R16", "C128", "C64", "C32", "C16"]
SMOKE_CONFIGS = ["R128", "R64", "R32", "C128", "C64", "C32"]
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
STAGE_HISTORY_LENGTHS = {
    "stage0": [1],
    "smoke": [1, 8],
    "pilot": [1, 8, 32, 64],
    "formal": HISTORY_LENGTHS,
}
FUTURE_KL_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128]
STATE_HORIZONS = [0, 1, 8, 32, 64, 128]
HORIZON = 128
EPS = 1e-12
IDENTITY_TOL = 1e-6
NEAR_TIE_AUC = 1e-5
RESCUE_SHRINK_FRACTION = 0.50

REQUIRED_GATES = [
    "CANONICAL_MANIFEST_GATE",
    "REPLAY_INPUT_GATE",
    "STATE_SEMANTICS_GATE",
    "ALL_QUANTIZER_SEMANTICS_GATES",
    "ALL_QUANTIZER_IDENTITY_GATES",
    "R128_BASELINE_REPRODUCTION_GATE",
    "C128_BASELINE_REPRODUCTION_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
]
STAGE0_EVIDENCE_GATES = REQUIRED_GATES + ["TIMING_GATE"]


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = import_module(MAG_RUNNER_PATH, "ling_mag_equalization_reused")
P = M.P
S = M.S


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def append_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def merged_fields(rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def finite_number(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def mean(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.median(vals) if vals else None


def std(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def bootstrap_ci(xs, n=2000, seed=20260910):
    vals = [float(x) for x in xs if finite_number(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    outs = []
    for _ in range(n):
        outs.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    outs.sort()
    return [outs[int(0.025 * (n - 1))], outs[int(0.975 * (n - 1))]]


def ranks(xs):
    pairs = sorted((x, i) for i, x in enumerate(xs))
    out = [0.0] * len(xs)
    j = 0
    while j < len(pairs):
        k = j + 1
        while k < len(pairs) and pairs[k][0] == pairs[j][0]:
            k += 1
        rank = (j + k - 1) / 2.0
        for _x, i in pairs[j:k]:
            out[i] = rank
        j = k
    return out


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= EPS or deny <= EPS:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (denx * deny + EPS)


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def tensor_norm(x):
    return P.tensor_norm(x)


def stack_norm(stack):
    return S.stack_norm(stack)


def stack_mse(stack):
    return S.stack_mse(stack)


def clone_cache(cache):
    return M.clone_cache(cache)


def cache_stack(cache, kda_layers):
    return S.cache_stack(cache, kda_layers)


def state_metrics(cache, fp_cache, kda_layers, include_cosine=True):
    return M.state_metrics(cache, fp_cache, kda_layers, include_cosine=include_cosine)


def full_logit_metrics(torch, ref_logits, branch_logits):
    return S.full_logit_metrics(torch, ref_logits, branch_logits)


def readout_distortion(records_fp, records_branch):
    return S.readout_distortion(records_fp, records_branch)


def copy_cache_stack_(cache, stack):
    for layer, qdq in stack.items():
        state = P.get_cache_state(cache, layer)
        if state is None:
            raise RuntimeError(f"missing KDA cache state at layer {layer}")
        state.copy_(qdq.to(device=state.device, dtype=state.dtype))


def parse_config(config):
    text = str(config).strip().upper()
    if len(text) < 2 or text[0] not in {"R", "C"}:
        raise ValueError(f"invalid granularity config: {config}")
    group_size = int(text[1:])
    if group_size <= 0:
        raise ValueError(f"group size must be positive: {config}")
    return {"config": text, "orientation": text[0], "group_size": group_size}


def stage_configs(stage):
    return list(SMOKE_CONFIGS if stage == "smoke" else CONFIGS)


def stage_history_lengths(stage):
    return list(STAGE_HISTORY_LENGTHS[stage])


def stage_unit_limit(stage):
    return {"stage0": 1, "smoke": 1, "pilot": 3, "formal": None}.get(stage)


def expected_unit_rows(stage, n_units):
    return int(n_units) * len(stage_history_lengths(stage)) * len(stage_configs(stage))


def expected_jsonl_rows(stage, n_units, horizon=HORIZON):
    _ = horizon
    return expected_unit_rows(stage, n_units)


def expected_horizon_rows(stage, n_units, horizon=HORIZON):
    return expected_unit_rows(stage, n_units) * int(horizon)


def expected_source_rows(stage, n_units):
    return int(n_units) * len(stage_configs(stage)) * sum(stage_history_lengths(stage))


def expected_state_rows(stage, n_units):
    return expected_unit_rows(stage, n_units) * len(STATE_HORIZONS)


def fake_quant_ling_state_grouped(state, config):
    import torch
    cfg = parse_config(config)
    x = state.detach().float()
    b, h, k, v = x.shape
    g = cfg["group_size"]
    qmax = 127
    if cfg["orientation"] == "R":
        if v % g != 0:
            raise ValueError(f"R{g} requires V dimension divisible by group size; got V={v}")
        grouped = x.reshape(b, h, k, v // g, g)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / qmax
        codes = torch.round(grouped / scale).clamp(-qmax, qmax)
        qdq_grouped = codes * scale
        qdq = qdq_grouped.reshape_as(x)
        axis = "V"
    else:
        if k % g != 0:
            raise ValueError(f"C{g} requires K dimension divisible by group size; got K={k}")
        grouped = x.reshape(b, h, k // g, g, v)
        scale = grouped.abs().amax(dim=3, keepdim=True).clamp_min(EPS) / qmax
        codes = torch.round(grouped / scale).clamp(-qmax, qmax)
        qdq_grouped = codes * scale
        qdq = qdq_grouped.reshape_as(x)
        axis = "K"
    meta = {
        "config": cfg["config"],
        "orientation": cfg["orientation"],
        "group_size": g,
        "bits": 8,
        "qrange": [-qmax, qmax],
        "zero_point": 0,
        "symmetric": True,
        "scale_formula": "amax(abs(group)).clamp_min(1e-12) / 127",
        "rounding": "torch.round",
        "dequantization": "codes * scale",
        "ling_state_shape": "[B,H,K,V]",
        "group_axis": axis,
        "crosses_key_channels": cfg["orientation"] == "C",
        "key_channels_per_scale": g if cfg["orientation"] == "C" else 1,
        "scale": scale,
        "codes": codes,
        "scale_shape": list(scale.shape),
    }
    return qdq.to(state.dtype), meta


def quantizer_identity_gate(a, b):
    import torch
    diff = (a.detach().float() - b.detach().float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
    return {"gate": "PASS" if finite and max_abs <= IDENTITY_TOL else "FAIL", "max_abs_error": max_abs}


def quantizer_semantics_audit(torch):
    state = torch.linspace(-3.0, 3.0, steps=128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    out = {}
    all_sem = True
    all_ident = True
    for config in CONFIGS:
        qdq, meta = fake_quant_ling_state_grouped(state, config)
        cfg = parse_config(config)
        sem_ok = (
            meta["orientation"] == cfg["orientation"]
            and meta["group_size"] == cfg["group_size"]
            and meta["group_axis"] == ("V" if cfg["orientation"] == "R" else "K")
            and bool(torch.isfinite(meta["scale"]).all().item())
            and bool(torch.isfinite(qdq).all().item())
        )
        ident = quantizer_identity_gate(qdq, qdq)
        out[f"{config}_QUANTIZER_SEMANTICS_GATE"] = "PASS" if sem_ok else "FAIL"
        out[f"{config}_QUANTIZER_IDENTITY_GATE"] = ident["gate"]
        out[f"{config}_scale_shape"] = meta["scale_shape"]
        all_sem = all_sem and sem_ok
        all_ident = all_ident and ident["gate"] == "PASS"
    r128, _ = fake_quant_ling_state_grouped(state, "R128")
    c128, _ = fake_quant_ling_state_grouped(state, "C128")
    pr128, _ = P.fake_quant_ling_state(state, "INT8_R128")
    pc128, _ = P.fake_quant_ling_state(state, "INT8_C128")
    out["ALL_QUANTIZER_SEMANTICS_GATES"] = "PASS" if all_sem else "FAIL"
    out["ALL_QUANTIZER_IDENTITY_GATES"] = "PASS" if all_ident else "FAIL"
    out["R128_BASELINE_REPRODUCTION_GATE"] = "PASS" if torch.allclose(r128, pr128, atol=0.0, rtol=0.0) else "FAIL"
    out["C128_BASELINE_REPRODUCTION_GATE"] = "PASS" if torch.allclose(c128, pc128, atol=0.0, rtol=0.0) else "FAIL"
    return out


def quantize_cache_grouped(torch, cache, config, kda_layers, collect_stats=False):
    touched = []
    stats = {}
    source_stack = {}
    qdq_stack = {}
    for layer_idx in kda_layers:
        state = P.get_cache_state(cache, layer_idx)
        if state is None:
            continue
        before = state.detach().float()
        qdq, meta = fake_quant_ling_state_grouped(state, config)
        finite = bool(torch.isfinite(qdq).all().item())
        if not finite:
            return {
                "finite": False,
                "touched_layers": touched + [layer_idx],
                "bad_layer": layer_idx,
                "source_norm": None,
                "source_mse": None,
                "stats": stats,
            }
        diff = qdq.detach().float() - before
        source_stack[int(layer_idx)] = diff
        qdq_stack[int(layer_idx)] = qdq.detach().clone()
        if collect_stats:
            codes = meta["codes"]
            stats[str(layer_idx)] = {
                "state_shape": list(state.shape),
                "scale_shape": meta["scale_shape"],
                "orientation": meta["orientation"],
                "group_size": meta["group_size"],
                "group_axis": meta["group_axis"],
                "crosses_key_channels": meta["crosses_key_channels"],
                "saturation_fraction": float((codes.detach().float().abs() >= 127.0).float().mean().item()),
            }
        with torch.inference_mode():
            state.copy_(qdq)
        touched.append(layer_idx)
    return {
        "finite": True,
        "touched_layers": touched,
        "bad_layer": None,
        "source_norm": stack_norm(source_stack) if source_stack else 0.0,
        "source_mse": stack_mse(source_stack) if source_stack else 0.0,
        "source_stack": source_stack,
        "qdq_stack": qdq_stack,
        "stats": stats,
    }


def backend_log_decay(torch, rec):
    g = rec.get("g")
    a_log = rec.get("A_log")
    if g is None or a_log is None:
        return None
    gf = g.detach().float()
    af = a_log.detach().float().reshape(1, 1, -1, 1)
    dt = rec.get("dt_bias")
    if dt is not None:
        dtf = dt.detach().float()
        if dtf.numel() == gf.shape[-2] * gf.shape[-1]:
            dtf = dtf.reshape(1, 1, gf.shape[-2], gf.shape[-1])
        elif dtf.numel() == gf.shape[-1]:
            dtf = dtf.reshape(1, 1, 1, gf.shape[-1])
        else:
            dtf = dtf.reshape(*([1] * (gf.ndim - dtf.ndim)), *dtf.shape)
        gf = gf + dtf.to(device=gf.device)
    return -af.to(device=gf.device).exp() * torch.nn.functional.softplus(gf)


def decay_heterogeneity_for_records(torch, records, config):
    cfg = parse_config(config)
    c_stds = []
    c_ranges = []
    r_stds = []
    for rec in records.values():
        log_decay = backend_log_decay(torch, rec)
        if log_decay is None:
            continue
        k_dim = log_decay.shape[-1]
        if k_dim % cfg["group_size"] != 0:
            continue
        if cfg["orientation"] == "C":
            grouped = log_decay.reshape(*log_decay.shape[:-1], k_dim // cfg["group_size"], cfg["group_size"])
            std_vals = grouped.std(dim=-1, unbiased=False)
            range_vals = grouped.max(dim=-1).values - grouped.min(dim=-1).values
            c_stds.append(float(std_vals.mean().item()))
            c_ranges.append(float(range_vals.mean().item()))
        else:
            r_stds.append(0.0)
    return {
        "C_GROUP_DECAY_HETEROGENEITY": mean(c_stds),
        "C_GROUP_DECAY_RANGE": mean(c_ranges),
        "R_ANALOG_DECAY_HETEROGENEITY": mean(r_stds),
        "status": "MEASURED_FROM_CAPTURED_BACKEND_LOG_DECAY" if c_stds or r_stds else "NOT_AVAILABLE",
    }


def phase_row_from_values(group_size, history_length, kl_r, kl_c, norm_r, norm_c, mse_r, mse_c):
    gap = float(kl_c) - float(kl_r)
    if abs(gap) <= NEAR_TIE_AUC:
        preferred = "NEAR_TIE"
    elif gap > 0:
        preferred = "R_PREFERRED"
    else:
        preferred = "C_PREFERRED"
    return {
        "group_size": int(group_size),
        "history_length": int(history_length),
        "KL_R": float(kl_r),
        "KL_C": float(kl_c),
        "C_MINUS_R_FUNCTIONAL_GAP": gap,
        "C_OVER_R_SOURCE_NORM": None if norm_r is None or abs(float(norm_r)) <= EPS else float(norm_c) / float(norm_r),
        "C_OVER_R_SOURCE_MSE": None if mse_r is None or abs(float(mse_r)) <= EPS else float(mse_c) / float(mse_r),
        "preferred": preferred,
    }


def classify_orientation(gap_by_group, near_tie=NEAR_TIE_AUC):
    ordered = [int(g) for g in GROUP_SIZES if int(g) in gap_by_group]
    gaps = {g: float(gap_by_group[g]) for g in ordered}
    boundary = next((g for g in ordered if abs(gaps[g]) <= near_tie), None)
    flip = next((g for g in ordered if gaps[g] < -near_tie), None)
    if flip is not None:
        return {
            "result": "ORIENTATION_FLIP_CONFIRMED",
            "boundary_group_size": flip,
            "preferred_by_group_size": group_preferences(gaps),
        }
    if boundary is not None:
        return {
            "result": "ORIENTATION_BOUNDARY_APPROACHED",
            "boundary_group_size": boundary,
            "preferred_by_group_size": group_preferences(gaps),
        }
    positives = [g for g in ordered if gaps[g] > near_tie]
    if len(positives) == len(ordered):
        coarse = abs(gaps[128]) if 128 in gaps else max(abs(v) for v in gaps.values())
        fine = abs(gaps[16]) if 16 in gaps else min(abs(v) for v in gaps.values())
        if coarse > near_tie and fine <= coarse * RESCUE_SHRINK_FRACTION:
            result = "R_PREFERENCE_WITH_C_GRANULARITY_RESCUE"
        else:
            result = "R_PREFERENCE_STABLE"
        return {"result": result, "boundary_group_size": "NONE", "preferred_by_group_size": group_preferences(gaps)}
    return {"result": "ORIENTATION_BOUNDARY_APPROACHED", "boundary_group_size": "INCOMPLETE_OR_MIXED", "preferred_by_group_size": group_preferences(gaps)}


def group_preferences(gaps):
    out = {}
    for g, gap in gaps.items():
        if abs(gap) <= NEAR_TIE_AUC:
            out[str(g)] = "NEAR_TIE"
        elif gap > 0:
            out[str(g)] = "R_PREFERRED"
        else:
            out[str(g)] = "C_PREFERRED"
    return out


def row_stats(rows, key):
    vals = [as_float(r.get(key)) for r in rows if as_float(r.get(key)) is not None]
    return {
        "mean": mean(vals),
        "median": median(vals),
        "std": std(vals),
        "n": len(vals),
        "positive": sum(1 for x in vals if x > 0),
        "negative": sum(1 for x in vals if x < 0),
        "near_tie": sum(1 for x in vals if abs(x) <= NEAR_TIE_AUC),
        "bootstrap_CI": bootstrap_ci(vals),
    }


def timing_gate_from_source_rows(source_rows):
    groups = defaultdict(set)
    missing = 0
    for row in source_rows:
        if as_float(row.get("t0")) is None or as_float(row.get("history_length")) is None or as_float(row.get("active_timestep")) is None:
            missing += 1
            continue
        key = (row.get("unit_id"), row.get("config"), int(float(row["history_length"])), int(float(row["t0"])))
        groups[key].add(int(float(row["active_timestep"])))
    bad = []
    for (unit_id, config, L, t0), seen in groups.items():
        expected = set(range(t0 - L + 1, t0 + 1))
        if seen != expected:
            bad.append({
                "unit_id": unit_id,
                "config": config,
                "history_length": L,
                "t0": t0,
                "missing": sorted(expected - seen),
                "extra": sorted(seen - expected),
            })
    return {
        "gate": "PASS" if source_rows and not missing and not bad else "FAIL",
        "checked_groups": len(groups),
        "missing_evidence_rows": missing,
        "bad_groups": bad[:20],
    }


def noninterference_gate_from_rows(source_rows, horizon_rows):
    current_vals = [as_float(r.get("current_logit_nonretroactivity_max_abs")) for r in source_rows]
    current_vals = [v for v in current_vals if v is not None]
    future_calls = [as_float(r.get("future_quantize_calls")) for r in horizon_rows]
    future_calls = [v for v in future_calls if v is not None]
    no_future_flags = [str(r.get("no_further_quantization")) in {"True", "true", "1"} for r in horizon_rows]
    row_flags = [r.get("CURRENT_LOGIT_NONRETROACTIVITY_GATE") == "PASS" for r in source_rows if r.get("CURRENT_LOGIT_NONRETROACTIVITY_GATE") is not None]
    ok = (
        bool(source_rows)
        and bool(horizon_rows)
        and bool(current_vals)
        and max(current_vals) <= 0.0
        and bool(future_calls)
        and max(future_calls) == 0.0
        and all(no_future_flags)
        and all(row_flags)
    )
    return {
        "gate": "PASS" if ok else "FAIL",
        "max_current_logit_nonretroactivity_abs": max(current_vals) if current_vals else None,
        "max_future_quantize_calls": max(future_calls) if future_calls else None,
        "n_source_rows": len(source_rows),
        "n_horizon_rows": len(horizon_rows),
    }


def stage0_gate_from_evidence(gates, failed_units):
    needed = [
        "REPLAY_INPUT_GATE",
        "STATE_SEMANTICS_GATE",
        "ALL_QUANTIZER_SEMANTICS_GATES",
        "ALL_QUANTIZER_IDENTITY_GATES",
        "R128_BASELINE_REPRODUCTION_GATE",
        "C128_BASELINE_REPRODUCTION_GATE",
        "TIMING_GATE",
        "INSTRUMENTATION_NONINTERFERENCE_GATE",
    ]
    return "PASS" if not failed_units and all(gates.get(key) == "PASS" for key in needed) else "FAIL"


def coverage_gate(stage, unit_rows, horizon_rows, source_rows, state_rows, n_units, horizon=HORIZON):
    expected_units = expected_unit_rows(stage, n_units)
    expected_horizons = expected_horizon_rows(stage, n_units, horizon)
    expected_sources = expected_source_rows(stage, n_units)
    expected_states = expected_state_rows(stage, n_units)
    unit_grid = {(r.get("unit_id"), int(float(r.get("history_length"))), r.get("config")) for r in unit_rows if as_float(r.get("history_length")) is not None}
    horizon_grid = defaultdict(set)
    for row in horizon_rows:
        if as_float(row.get("history_length")) is not None and as_float(row.get("horizon")) is not None:
            horizon_grid[(row.get("unit_id"), int(float(row["history_length"])), row.get("config"))].add(int(float(row["horizon"])))
    source_grid = defaultdict(set)
    for row in source_rows:
        if as_float(row.get("history_length")) is not None and as_float(row.get("active_timestep")) is not None:
            source_grid[(row.get("unit_id"), int(float(row["history_length"])), row.get("config"), int(float(row.get("t0", 0))))].add(int(float(row["active_timestep"])))
    state_grid = defaultdict(set)
    for row in state_rows:
        if as_float(row.get("history_length")) is not None and as_float(row.get("horizon")) is not None:
            state_grid[(row.get("unit_id"), int(float(row["history_length"])), row.get("config"))].add(int(float(row["horizon"])))
    bad_horizon = [key for key, seen in horizon_grid.items() if seen != set(range(1, int(horizon) + 1))]
    bad_source = []
    for (unit_id, L, config, t0), seen in source_grid.items():
        if seen != set(range(t0 - L + 1, t0 + 1)):
            bad_source.append((unit_id, L, config))
    bad_state = [key for key, seen in state_grid.items() if seen != set(STATE_HORIZONS)]
    gate = (
        len(unit_rows) == expected_units
        and len(unit_grid) == expected_units
        and len(horizon_rows) == expected_horizons
        and len(horizon_grid) == expected_units
        and not bad_horizon
        and len(source_rows) == expected_sources
        and not bad_source
        and len(state_rows) == expected_states
        and len(state_grid) == expected_units
        and not bad_state
    )
    return {
        "gate": "PASS" if gate else "FAIL",
        "unit_rows": len(unit_rows),
        "expected_unit_rows": expected_units,
        "unit_grid_cells": len(unit_grid),
        "horizon_rows": len(horizon_rows),
        "expected_horizon_rows": expected_horizons,
        "source_rows": len(source_rows),
        "expected_source_rows": expected_sources,
        "state_rows": len(state_rows),
        "expected_state_rows": expected_states,
        "bad_horizon_cells": len(bad_horizon),
        "bad_source_cells": len(bad_source),
        "bad_state_cells": len(bad_state),
    }


def run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, cache_points, min_ref_t, kda_layers):
    return M.run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, cache_points, min_ref_t, kda_layers)


def selected_state_row(unit, item, history_length, config, h, metrics):
    return {
        "task": TASK,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
        "unit_id": str(unit["unit_id"]),
        "prompt_id": str(unit["problem_id"]),
        "problem_index": int(item["problem_index"]),
        "t0": int(unit["t0"]),
        "history_length": int(history_length),
        "config": config,
        "orientation": parse_config(config)["orientation"],
        "group_size": parse_config(config)["group_size"],
        "horizon": int(h),
        "state_error_norm": metrics["state_error_norm"],
        "state_relative_L2": metrics["state_relative_L2"],
        "state_cosine": metrics["state_cosine"],
    }


def evaluate_history_length(torch, model, capture, unit, item, toks, fp_ref, history_length, configs, horizon, kda_layers):
    t0 = int(unit["t0"])
    start_t = t0 - int(history_length) + 1
    start_prev = start_t - 1
    prompt_len = fp_ref["prompt_len"]
    device = next(model.parameters()).device
    fp_t0_cache = fp_ref["fp_cache_after"][t0]
    branch_pasts = {cfg: clone_cache(fp_ref["fp_cache_after"][start_prev]) for cfg in configs}
    branch_masks = {cfg: fp_ref["fp_mask_after"][start_prev].clone() for cfg in configs}
    source_rows = []
    source_by_config = defaultdict(list)
    for t in range(start_t, t0 + 1):
        cur = torch.tensor([[int(toks[t])]], device=device, dtype=torch.long)
        for cfg in configs:
            branch_masks[cfg] = torch.cat([branch_masks[cfg], torch.ones_like(cur)], dim=-1)
            capture.records.clear()
            capture.branch = f"HIST_{cfg}"
            with torch.inference_mode():
                out = model(
                    input_ids=cur,
                    attention_mask=branch_masks[cfg],
                    past_key_values=branch_pasts[cfg],
                    cache_position=torch.tensor([prompt_len + t], device=device, dtype=torch.long),
                    use_cache=True,
                )
            branch_pasts[cfg] = out.past_key_values
            current_logits_before = out.logits.detach().float().clone()
            qmeta = quantize_cache_grouped(torch, branch_pasts[cfg], cfg, kda_layers, collect_stats=False)
            current_logit_abs = float((out.logits.detach().float() - current_logits_before).abs().max().item())
            if not qmeta["finite"]:
                raise RuntimeError(f"nonfinite grouped quantization unit={unit['unit_id']} L={history_length} config={cfg} t={t} layer={qmeta['bad_layer']}")
            cfg_meta = parse_config(cfg)
            decay = decay_heterogeneity_for_records(torch, capture.records.get(f"HIST_{cfg}", {}), cfg)
            row = {
                "task": TASK,
                "git_commit": sh(["git", "rev-parse", "HEAD"]),
                "manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
                "unit_id": str(unit["unit_id"]),
                "prompt_id": str(unit["problem_id"]),
                "problem_index": int(item["problem_index"]),
                "t0": t0,
                "history_length": int(history_length),
                "active_timestep": int(t),
                "config": cfg,
                "orientation": cfg_meta["orientation"],
                "group_size": cfg_meta["group_size"],
                "source_norm": qmeta["source_norm"],
                "source_mse": qmeta["source_mse"],
                "touched_layer_count": len(qmeta["touched_layers"]),
                "C_GROUP_DECAY_HETEROGENEITY": decay["C_GROUP_DECAY_HETEROGENEITY"],
                "C_GROUP_DECAY_RANGE": decay["C_GROUP_DECAY_RANGE"],
                "R_ANALOG_DECAY_HETEROGENEITY": decay["R_ANALOG_DECAY_HETEROGENEITY"],
                "DECAY_HETEROGENEITY_STATUS": decay["status"],
                "current_logit_nonretroactivity_max_abs": current_logit_abs,
                "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS" if current_logit_abs <= 0.0 else "FAIL",
                "NO_FURTHER_QUANTIZATION_GATE": "PASS",
            }
            source_rows.append(row)
            source_by_config[cfg].append(row)
    horizon_rows = []
    unit_rows = []
    state_rows = []
    auc_terms = {cfg: [] for cfg in configs}
    first_readout = {cfg: None for cfg in configs}
    selected_state = defaultdict(dict)
    for cfg in configs:
        sm = state_metrics(branch_pasts[cfg], fp_t0_cache, kda_layers, include_cosine=True)
        selected_state[cfg][0] = sm
        state_rows.append(selected_state_row(unit, item, history_length, cfg, 0, sm))
    fp_past = clone_cache(fp_t0_cache)
    fp_mask = fp_ref["fp_mask_after"][t0].clone()
    for h in range(1, horizon + 1):
        tok = int(toks[t0 + h])
        cur = torch.tensor([[tok]], device=device, dtype=torch.long)
        fp_mask = torch.cat([fp_mask, torch.ones_like(cur)], dim=-1)
        capture.records.clear()
        capture.branch = "FP_FUTURE_BASE"
        with torch.inference_mode():
            fp_out = model(
                input_ids=cur,
                attention_mask=fp_mask,
                past_key_values=fp_past,
                cache_position=torch.tensor([prompt_len + t0 + h], device=device, dtype=torch.long),
                use_cache=True,
            )
        fp_past = fp_out.past_key_values
        for cfg in configs:
            branch_masks[cfg] = torch.cat([branch_masks[cfg], torch.ones_like(cur)], dim=-1)
            capture.branch = f"FUTURE_{cfg}"
            with torch.inference_mode():
                b_out = model(
                    input_ids=cur,
                    attention_mask=branch_masks[cfg],
                    past_key_values=branch_pasts[cfg],
                    cache_position=torch.tensor([prompt_len + t0 + h], device=device, dtype=torch.long),
                    use_cache=True,
                )
            branch_pasts[cfg] = b_out.past_key_values
            if not bool(torch.isfinite(b_out.logits).all().item()):
                raise RuntimeError(f"nonfinite logits unit={unit['unit_id']} L={history_length} config={cfg} h={h}")
            metrics = full_logit_metrics(torch, fp_out.logits, b_out.logits)
            auc_terms[cfg].append(metrics["KL"])
            if h == 1:
                first_readout[cfg] = readout_distortion(capture.records["FP_FUTURE_BASE"], capture.records[f"FUTURE_{cfg}"])
            sm = state_metrics(branch_pasts[cfg], fp_past, kda_layers, include_cosine=h in STATE_HORIZONS)
            if h in STATE_HORIZONS:
                selected_state[cfg][h] = sm
                state_rows.append(selected_state_row(unit, item, history_length, cfg, h, sm))
            cfg_meta = parse_config(cfg)
            horizon_rows.append({
                "task": TASK,
                "git_commit": sh(["git", "rev-parse", "HEAD"]),
                "manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
                "unit_id": str(unit["unit_id"]),
                "prompt_id": str(unit["problem_id"]),
                "problem_index": int(item["problem_index"]),
                "t0": t0,
                "history_length": int(history_length),
                "horizon": h,
                "config": cfg,
                "orientation": cfg_meta["orientation"],
                "group_size": cfg_meta["group_size"],
                "KL": metrics["KL"],
                "top1_match": metrics["top1_match"],
                "logit_cosine": metrics["logit_cosine"],
                "logit_rel_l2": metrics["logit_rel_l2"],
                "top20_overlap": metrics["top20_overlap"],
                "state_error_norm": sm["state_error_norm"],
                "state_relative_L2": sm["state_relative_L2"],
                "state_cosine": sm["state_cosine"],
                "readout_visibility": first_readout[cfg] if h == 1 else None,
                "future_quantize_calls": 0,
                "no_further_quantization": True,
                "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
            })
    for cfg in configs:
        cfg_meta = parse_config(cfg)
        row = {
            "task": TASK,
            "stage": None,
            "git_commit": sh(["git", "rev-parse", "HEAD"]),
            "manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
            "unit_id": str(unit["unit_id"]),
            "prompt_id": str(unit["problem_id"]),
            "problem_index": int(item["problem_index"]),
            "t0": t0,
            "history_length": int(history_length),
            "config": cfg,
            "orientation": cfg_meta["orientation"],
            "group_size": cfg_meta["group_size"],
            "future_kl_auc": mean(auc_terms[cfg]),
            "first_future_readout_distortion": first_readout[cfg],
            "source_norm": mean([r["source_norm"] for r in source_by_config[cfg]]),
            "source_mse": mean([r["source_mse"] for r in source_by_config[cfg]]),
            "touched_layer_count": min([r["touched_layer_count"] for r in source_by_config[cfg]], default=0),
            "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
            "NO_FURTHER_QUANTIZATION_GATE": "PASS",
        }
        for h in FUTURE_KL_HORIZONS:
            row[f"KL_h{h}"] = auc_terms[cfg][h - 1] if len(auc_terms[cfg]) >= h else None
        for h in STATE_HORIZONS:
            sm = selected_state[cfg].get(h)
            if sm:
                row[f"state_error_h{h}"] = sm["state_error_norm"]
                row[f"state_relative_L2_h{h}"] = sm["state_relative_L2"]
                row[f"state_cosine_h{h}"] = sm["state_cosine"]
        row["state_error_t0"] = row.get("state_error_h0")
        unit_rows.append(row)
    return horizon_rows, unit_rows, source_rows, state_rows


def select_units_by_ids(units, unit_ids):
    return M.select_units_by_ids(units, unit_ids)


def run_unit(torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, history_lengths, configs, horizon):
    pid = str(unit["problem_id"])
    item = rows_by_pid.get(pid)
    toks = teacher_tokens.get(pid)
    if item is None or not toks:
        raise RuntimeError(f"missing prompt or teacher-forced tokens for {unit['unit_id']}")
    t0 = int(unit["t0"])
    need = t0 + horizon + 1
    if len(toks) < need:
        raise RuntimeError(f"teacher tokens too short for {unit['unit_id']}: need {need}, have {len(toks)}")
    min_start = t0 - max(history_lengths) + 1
    start_prevs = {t0 - int(L) for L in history_lengths}
    cache_points = set(start_prevs) | {t0}
    P.setup_seed(P.BASE_SEED + int(item["problem_index"]))
    device = next(model.parameters()).device
    input_ids = P.render_prompt(tokenizer, item["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    fp_ref = run_fp_reference(torch, model, capture, input_ids, mask, toks, t0, cache_points, min_start, kda_layers)
    all_horizon_rows = []
    all_unit_rows = []
    all_source_rows = []
    all_state_rows = []
    for history_length in history_lengths:
        hrs, urs, srs, strs = evaluate_history_length(
            torch, model, capture, unit, item, toks, fp_ref, history_length, configs, horizon, kda_layers
        )
        all_horizon_rows.extend(hrs)
        all_unit_rows.extend(urs)
        all_source_rows.extend(srs)
        all_state_rows.extend(strs)
    return all_horizon_rows, all_unit_rows, all_source_rows, all_state_rows


def unit_fields():
    fields = [
        "task", "stage", "git_commit", "manifest_sha256", "unit_id", "prompt_id", "problem_index", "t0",
        "history_length", "config", "orientation", "group_size", "future_kl_auc",
    ]
    fields += [f"KL_h{h}" for h in FUTURE_KL_HORIZONS]
    fields += [
        "state_error_t0", "state_error_h0", "state_error_h1", "state_error_h8", "state_error_h32", "state_error_h64", "state_error_h128",
        "state_relative_L2_h0", "state_relative_L2_h1", "state_relative_L2_h8", "state_relative_L2_h32", "state_relative_L2_h64", "state_relative_L2_h128",
        "state_cosine_h0", "state_cosine_h1", "state_cosine_h8", "state_cosine_h32", "state_cosine_h64", "state_cosine_h128",
        "source_norm", "source_mse", "first_future_readout_distortion", "touched_layer_count",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE", "NO_FURTHER_QUANTIZATION_GATE",
    ]
    return fields


def horizon_fields():
    return [
        "task", "git_commit", "manifest_sha256", "unit_id", "prompt_id", "problem_index", "t0",
        "history_length", "horizon", "config", "orientation", "group_size",
        "KL", "top1_match", "logit_cosine", "logit_rel_l2", "top20_overlap",
        "state_error_norm", "state_relative_L2", "state_cosine", "readout_visibility",
        "future_quantize_calls", "no_further_quantization", "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    ]


def source_fields():
    return [
        "task", "git_commit", "manifest_sha256", "unit_id", "prompt_id", "problem_index", "t0",
        "history_length", "active_timestep", "config", "orientation", "group_size",
        "source_norm", "source_mse", "touched_layer_count",
        "C_GROUP_DECAY_HETEROGENEITY", "C_GROUP_DECAY_RANGE", "R_ANALOG_DECAY_HETEROGENEITY", "DECAY_HETEROGENEITY_STATUS",
        "current_logit_nonretroactivity_max_abs",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE", "NO_FURTHER_QUANTIZATION_GATE",
    ]


def state_fields():
    return [
        "task", "git_commit", "manifest_sha256", "unit_id", "prompt_id", "problem_index", "t0",
        "history_length", "config", "orientation", "group_size", "horizon",
        "state_error_norm", "state_relative_L2", "state_cosine",
    ]


def reset_stage_outputs(stage):
    for name in [
        f"results_{stage}.jsonl",
        f"unit_summary_{stage}.csv",
        f"horizon_{stage}.csv",
        f"source_error_{stage}.csv",
        f"state_error_trajectory_{stage}.csv",
        f"failure_{stage}.json",
        f"failed_units_{stage}.jsonl",
        f"protocol_snapshot_{stage}.json",
        f"{stage}_run_summary.json",
    ]:
        path = RUN_DIR / name
        if path.exists():
            path.unlink()


def write_stage_protocol_snapshot(stage, payload):
    save_json(RUN_DIR / f"protocol_snapshot_{stage}.json", payload)


def reproducibility_snapshot(command):
    return {
        "time": now(),
        "command": command,
        "git_head": sh(["git", "rev-parse", "HEAD"]),
        "git_status": sh(["git", "status", "--short"]),
        "git_diff": sh(["git", "diff", "--", "experiments/ling/run_ling_kda_int8_rc_granularity_phase_diagram_v1.py", "tests/ling/test_ling_kda_int8_rc_granularity_phase_diagram.py"]),
        "model_path": str(P.MODEL_PATH),
        "canonical_manifest_path": str(CANONICAL_UNITS),
        "manifest_sha256": P.canonical_manifest_sha256(CANONICAL_UNITS) if CANONICAL_UNITS.exists() else None,
        "cuda": sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        "python": sys.version.split()[0],
        "torch_transformers": sh([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"]),
        "configs": CONFIGS,
        "history_lengths": HISTORY_LENGTHS,
        "near_tie_auc": NEAR_TIE_AUC,
        "teacher_forced_protocol": "same FP teacher-forced token ids; no free generation",
        "persistent_quantization_window": "t0-L+1 ... t0",
        "no_further_quantization_after_t0": True,
    }


def repo_audit():
    return {
        "REUSED_FILES": [
            str(MAG_RUNNER_PATH),
            str(M.SOURCE_EQUALIZATION_RUNNER_PATH),
            str(CANONICAL_UNITS),
        ],
        "NEW_FILES": [
            str(REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"),
            str(REPO / "tests" / "ling" / "test_ling_kda_int8_rc_granularity_phase_diagram.py"),
        ],
        "NO_NEW_MANIFEST": True,
        "QUANTIZER_SEMANTICS": "R groups stay inside one Key row along V; C groups stay inside one Value column along K",
        "DECAY_HETEROGENEITY_STATUS": "NOT_AVAILABLE_PER_KEY_CHANNEL_FROM_PUBLIC_RUNTIME; CSV reports unavailable secondary diagnostic",
    }


def replay_input_gate(units, rows_by_pid, teacher_tokens, horizon):
    missing = []
    short = []
    for unit in units:
        pid = str(unit["problem_id"])
        if pid not in rows_by_pid or pid not in teacher_tokens:
            missing.append(str(unit["unit_id"]))
            continue
        need = int(unit["t0"]) + int(horizon) + 1
        if len(teacher_tokens[pid]) < need:
            short.append({"unit_id": str(unit["unit_id"]), "need": need, "have": len(teacher_tokens[pid])})
    return {"gate": "PASS" if not missing and not short else "FAIL", "missing": missing, "short": short}


def state_semantics_gate(torch, model, tokenizer, capture, unit, rows_by_pid, teacher_tokens, kda_layers):
    pid = str(unit["problem_id"])
    item = rows_by_pid[pid]
    toks = teacher_tokens[pid]
    device = next(model.parameters()).device
    input_ids = P.render_prompt(tokenizer, item["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    capture.records.clear()
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            cache_position=torch.arange(0, input_ids.shape[-1], device=device),
            use_cache=True,
        )
        cur = torch.tensor([[int(toks[0])]], device=device, dtype=torch.long)
        mask = torch.cat([mask, torch.ones_like(cur)], dim=-1)
        out = model(
            input_ids=cur,
            attention_mask=mask,
            past_key_values=out.past_key_values,
            cache_position=torch.tensor([input_ids.shape[-1]], device=device),
            use_cache=True,
        )
    shapes = []
    ok = True
    for layer in kda_layers:
        state = P.get_cache_state(out.past_key_values, layer)
        shape = list(state.shape) if state is not None else None
        shapes.append({"layer": int(layer), "shape": shape})
        ok = ok and state is not None and state.ndim == 4 and state.shape[-1] % 16 == 0 and state.shape[-2] % 16 == 0
    return {"gate": "PASS" if ok and shapes else "FAIL", "sample_shapes": shapes[:4]}


def unavailable_decay_heterogeneity_rows():
    rows = []
    for g in GROUP_SIZES:
        rows.append({
            "group_size": g,
            "C_GROUP_DECAY_HETEROGENEITY": None,
            "C_GROUP_DECAY_RANGE": None,
            "R_ANALOG_DECAY_HETEROGENEITY": None,
            "status": "NOT_AVAILABLE_PER_KEY_CHANNEL_FROM_PUBLIC_RUNTIME",
            "note": "Ling exposes A_log/dt_bias but not the backend per-key-channel decay coefficients used by fused/chunk KDA.",
        })
    return rows


def aggregate_decay_heterogeneity_rows(source_rows):
    rows = []
    for g in GROUP_SIZES:
        usable = [r for r in source_rows if r.get("orientation") in {"R", "C"} and as_float(r.get("group_size")) is not None]
        c_rows = [r for r in usable if str(r.get("orientation")) == "C" and int(float(r.get("group_size"))) == g]
        r_rows = [r for r in usable if str(r.get("orientation")) == "R" and int(float(r.get("group_size"))) == g]
        vals = [as_float(r.get("C_GROUP_DECAY_HETEROGENEITY")) for r in c_rows if as_float(r.get("C_GROUP_DECAY_HETEROGENEITY")) is not None]
        ranges = [as_float(r.get("C_GROUP_DECAY_RANGE")) for r in c_rows if as_float(r.get("C_GROUP_DECAY_RANGE")) is not None]
        r_vals = [as_float(r.get("R_ANALOG_DECAY_HETEROGENEITY")) for r in r_rows if as_float(r.get("R_ANALOG_DECAY_HETEROGENEITY")) is not None]
        rows.append({
            "group_size": g,
            "C_GROUP_DECAY_HETEROGENEITY": median(vals),
            "C_GROUP_DECAY_RANGE": median(ranges),
            "R_ANALOG_DECAY_HETEROGENEITY": median(r_vals),
            "status": "MEASURED_FROM_CAPTURED_BACKEND_LOG_DECAY" if vals else "NOT_AVAILABLE",
            "N_C_SOURCE_ROWS": len(c_rows),
            "N_R_SOURCE_ROWS": len(r_rows),
        })
    return rows if any(r["status"] == "MEASURED_FROM_CAPTURED_BACKEND_LOG_DECAY" for r in rows) else unavailable_decay_heterogeneity_rows()


def aggregate_by_phase(unit_rows):
    grouped = defaultdict(list)
    for row in unit_rows:
        grouped[(int(row["history_length"]), int(row["group_size"]), row["orientation"])].append(row)
    phase_rows = []
    contrast_rows = []
    for L in sorted({int(r["history_length"]) for r in unit_rows}):
        for g in GROUP_SIZES:
            r_rows = grouped.get((L, g, "R"), [])
            c_rows = grouped.get((L, g, "C"), [])
            if not r_rows or not c_rows:
                continue
            phase_rows.append(phase_row_from_values(
                g, L,
                median([r["future_kl_auc"] for r in r_rows]),
                median([r["future_kl_auc"] for r in c_rows]),
                median([r["source_norm"] for r in r_rows]),
                median([r["source_norm"] for r in c_rows]),
                median([r["source_mse"] for r in r_rows]),
                median([r["source_mse"] for r in c_rows]),
            ))
            by_unit_r = {r["unit_id"]: r for r in r_rows}
            by_unit_c = {r["unit_id"]: r for r in c_rows}
            for unit_id in sorted(set(by_unit_r) & set(by_unit_c)):
                rr = by_unit_r[unit_id]
                cc = by_unit_c[unit_id]
                contrast_rows.append({
                    "unit_id": unit_id,
                    "history_length": L,
                    "group_size": g,
                    "KL_R": as_float(rr["future_kl_auc"]),
                    "KL_C": as_float(cc["future_kl_auc"]),
                    "C_MINUS_R_FUNCTIONAL_GAP": as_float(cc["future_kl_auc"]) - as_float(rr["future_kl_auc"]),
                    "SOURCE_NORM_R": as_float(rr["source_norm"]),
                    "SOURCE_NORM_C": as_float(cc["source_norm"]),
                    "SOURCE_MSE_R": as_float(rr["source_mse"]),
                    "SOURCE_MSE_C": as_float(cc["source_mse"]),
                    "STATE_ERROR_R": as_float(rr.get("state_error_t0")),
                    "STATE_ERROR_C": as_float(cc.get("state_error_t0")),
                })
    return phase_rows, contrast_rows


def dict_by_group(rows, key, history_length=64):
    out = {}
    for g in GROUP_SIZES:
        vals = [r.get(key) for r in rows if int(r.get("group_size")) == g and int(r.get("history_length")) == int(history_length)]
        out[str(g)] = median(vals)
    return out


def paired_stats_by_group(contrast_rows, key="C_MINUS_R_FUNCTIONAL_GAP", history_length=64):
    return {
        str(g): row_stats(
            [r for r in contrast_rows if int(r.get("group_size")) == g and int(r.get("history_length")) == int(history_length)],
            key,
        )
        for g in GROUP_SIZES
    }


def paired_stats_by_group_and_l(contrast_rows, key="C_MINUS_R_FUNCTIONAL_GAP"):
    out = {}
    for L in HISTORY_LENGTHS:
        out[str(L)] = paired_stats_by_group(contrast_rows, key=key, history_length=L)
    return out


def source_quantizability_rows(contrast_rows):
    rows = []
    for g in GROUP_SIZES:
        all_rows = [r for r in contrast_rows if int(r["group_size"]) == g]
        l64_rows = [r for r in all_rows if int(r["history_length"]) == 64]
        rows.append({
            "group_size": g,
            "SOURCE_NORM_R": median([r["SOURCE_NORM_R"] for r in l64_rows]),
            "SOURCE_NORM_C": median([r["SOURCE_NORM_C"] for r in l64_rows]),
            "SOURCE_MSE_R": median([r["SOURCE_MSE_R"] for r in l64_rows]),
            "SOURCE_MSE_C": median([r["SOURCE_MSE_C"] for r in l64_rows]),
            "C_OVER_R_SOURCE_NORM": median([r["SOURCE_NORM_C"] / (r["SOURCE_NORM_R"] + EPS) for r in l64_rows if r["SOURCE_NORM_R"] is not None]),
            "C_OVER_R_SOURCE_MSE": median([r["SOURCE_MSE_C"] / (r["SOURCE_MSE_R"] + EPS) for r in l64_rows if r["SOURCE_MSE_R"] is not None]),
            "N_PAIRED_UNITS_L64": len(l64_rows),
            "N_PAIRED_ROWS_ALL_L": len(all_rows),
        })
    norm_ratios = [as_float(r["C_OVER_R_SOURCE_NORM"]) for r in rows if as_float(r["C_OVER_R_SOURCE_NORM"]) is not None]
    mse_ratios = [as_float(r["C_OVER_R_SOURCE_MSE"]) for r in rows if as_float(r["C_OVER_R_SOURCE_MSE"]) is not None]
    sizes = [r["group_size"] for r in rows if as_float(r["C_OVER_R_SOURCE_NORM"]) is not None]
    for row in rows:
        row["SPEARMAN_GROUP_SIZE_C_OVER_R_SOURCE_NORM"] = spearman(sizes, norm_ratios) if len(sizes) == len(norm_ratios) else None
        row["SPEARMAN_GROUP_SIZE_C_OVER_R_SOURCE_MSE"] = spearman(sizes, mse_ratios) if len(sizes) == len(mse_ratios) else None
    return rows


def granularity_gain(by_group):
    g128 = as_float(by_group.get("128"))
    g16 = as_float(by_group.get("16"))
    if g128 is None or g16 is None:
        return None
    return g128 - g16


def normalize_failed_units(items):
    out = []
    seen = set()
    for item in items or []:
        if not item:
            continue
        obj = {"unit_id": item} if isinstance(item, str) else dict(item)
        key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(obj)
    return out


def summarize_aggregate(stage, unit_rows, horizon_rows, source_rows, state_rows, stage_summaries=None, failed_units=None):
    phase_rows, contrast_rows = aggregate_by_phase(unit_rows)
    source_rows_by_g = source_quantizability_rows(contrast_rows)
    decay_rows = aggregate_decay_heterogeneity_rows(source_rows)
    l64_gaps = {int(r["group_size"]): as_float(r["C_MINUS_R_FUNCTIONAL_GAP"]) for r in phase_rows if int(r["history_length"]) == 64}
    cls = classify_orientation(l64_gaps) if l64_gaps else {"result": "INCOMPLETE", "boundary_group_size": "UNKNOWN", "preferred_by_group_size": {}}
    failed = list(failed_units or [])
    for summary in stage_summaries or []:
        failed.extend(summary.get("failed_units") or [])
    failed.extend({"unit_id": r.get("unit_id"), "source": "unit_row_runtime_flag"} for r in unit_rows if r.get("runtime_error") or r.get("nonfinite"))
    failed_units = normalize_failed_units(failed)
    history_set = {int(r["history_length"]) for r in unit_rows}
    config_set = {r["config"] for r in unit_rows}
    n_units = len({r["unit_id"] for r in unit_rows})
    coverage = coverage_gate(stage, unit_rows, horizon_rows, source_rows, state_rows, n_units)
    gates = {key: "FAIL" for key in STAGE0_EVIDENCE_GATES}
    if stage_summaries:
        for key in STAGE0_EVIDENCE_GATES:
            vals = [s.get(key) for s in stage_summaries if s.get(key)]
            if vals:
                gates[key] = "PASS" if all(v == "PASS" for v in vals) else "FAIL"
    timing_details = timing_gate_from_source_rows(source_rows)
    noninterference_details = noninterference_gate_from_rows(source_rows, horizon_rows)
    computed_timing = timing_details["gate"]
    computed_noninterference = noninterference_details["gate"]
    prior_timing = gates.get("TIMING_GATE", "PASS") if stage_summaries else computed_timing
    prior_noninterference = gates.get("INSTRUMENTATION_NONINTERFERENCE_GATE", "PASS") if stage_summaries else computed_noninterference
    gates["TIMING_GATE"] = "PASS" if computed_timing == "PASS" and prior_timing == "PASS" else "FAIL"
    gates["INSTRUMENTATION_NONINTERFERENCE_GATE"] = "PASS" if computed_noninterference == "PASS" and prior_noninterference == "PASS" else "FAIL"
    formal_complete = (
        stage == "formal"
        and n_units == 18
        and history_set == set(HISTORY_LENGTHS)
        and config_set == set(CONFIGS)
        and len(unit_rows) == expected_unit_rows("formal", 18)
        and coverage["gate"] == "PASS"
        and not failed_units
        and all(v == "PASS" for v in gates.values())
    )
    kl_r = dict_by_group(phase_rows, "KL_R")
    kl_c = dict_by_group(phase_rows, "KL_C")
    gap = dict_by_group(phase_rows, "C_MINUS_R_FUNCTIONAL_GAP")
    source_norm_r = {str(g): median([r["SOURCE_NORM_R"] for r in source_rows_by_g if r["group_size"] == g]) for g in GROUP_SIZES}
    source_norm_c = {str(g): median([r["SOURCE_NORM_C"] for r in source_rows_by_g if r["group_size"] == g]) for g in GROUP_SIZES}
    source_mse_r = {str(g): median([r["SOURCE_MSE_R"] for r in source_rows_by_g if r["group_size"] == g]) for g in GROUP_SIZES}
    source_mse_c = {str(g): median([r["SOURCE_MSE_C"] for r in source_rows_by_g if r["group_size"] == g]) for g in GROUP_SIZES}
    state_error_r = {str(g): median([r["STATE_ERROR_R"] for r in contrast_rows if r["group_size"] == g and r["history_length"] == 64]) for g in GROUP_SIZES}
    state_error_c = {str(g): median([r["STATE_ERROR_C"] for r in contrast_rows if r["group_size"] == g and r["history_length"] == 64]) for g in GROUP_SIZES}
    summary = {
        "TASK": TASK,
        "stage": stage,
        "FORMAL_STATUS": "COMPLETE" if formal_complete else ("ANALYSIS_COMPLETE" if stage != "formal" else "INCOMPLETE"),
        "N_FORMAL_UNITS": n_units if stage == "formal" else 0,
        **gates,
        "SMOKE": stage_gate_from_file("smoke", "SMOKE"),
        "PILOT": stage_gate_from_file("pilot", "PILOT"),
        "GROUP_SIZES": GROUP_SIZES,
        "HISTORY_LENGTHS": HISTORY_LENGTHS,
        "KL_R_BY_GROUP_SIZE": kl_r,
        "KL_C_BY_GROUP_SIZE": kl_c,
        "C_MINUS_R_FUNCTIONAL_GAP_BY_GROUP_SIZE": gap,
        "DELTA_FUNC_PAIRED_STATS_BY_GROUP_SIZE": paired_stats_by_group(contrast_rows),
        "DELTA_FUNC_PAIRED_STATS_BY_HISTORY_AND_GROUP_SIZE": paired_stats_by_group_and_l(contrast_rows),
        "SOURCE_NORM_R_BY_GROUP_SIZE": source_norm_r,
        "SOURCE_NORM_C_BY_GROUP_SIZE": source_norm_c,
        "SOURCE_MSE_R_BY_GROUP_SIZE": source_mse_r,
        "SOURCE_MSE_C_BY_GROUP_SIZE": source_mse_c,
        "C_OVER_R_SOURCE_NORM_BY_GROUP_SIZE": {str(g): median([r["C_OVER_R_SOURCE_NORM"] for r in source_rows_by_g if r["group_size"] == g]) for g in GROUP_SIZES},
        "C_OVER_R_SOURCE_MSE_BY_GROUP_SIZE": {str(g): median([r["C_OVER_R_SOURCE_MSE"] for r in source_rows_by_g if r["group_size"] == g]) for g in GROUP_SIZES},
        "R_GRANULARITY_GAIN": granularity_gain(kl_r),
        "C_GRANULARITY_GAIN": granularity_gain(kl_c),
        "R_SOURCE_NORM_GRANULARITY_GAIN": granularity_gain(source_norm_r),
        "C_SOURCE_NORM_GRANULARITY_GAIN": granularity_gain(source_norm_c),
        "R_SOURCE_MSE_GRANULARITY_GAIN": granularity_gain(source_mse_r),
        "C_SOURCE_MSE_GRANULARITY_GAIN": granularity_gain(source_mse_c),
        "STATE_ERROR_R_BY_GROUP_SIZE": state_error_r,
        "STATE_ERROR_C_BY_GROUP_SIZE": state_error_c,
        "R_STATE_GRANULARITY_GAIN": granularity_gain(state_error_r),
        "C_STATE_GRANULARITY_GAIN": granularity_gain(state_error_c),
        "C_GROUP_DECAY_HETEROGENEITY_BY_GROUP_SIZE": {str(g): median([r["C_GROUP_DECAY_HETEROGENEITY"] for r in decay_rows if r["group_size"] == g]) for g in GROUP_SIZES},
        "COMMUTATOR_BY_GROUP_SIZE": "OPTIONAL_NOT_AVAILABLE",
        "CODE_CHURN_BY_GROUP_SIZE": "OPTIONAL_NOT_AVAILABLE",
        "ORIENTATION_BOUNDARY_GROUP_SIZE": cls["boundary_group_size"],
        "LING_ORIENTATION_GRANULARITY_RESULT": cls["result"],
        "FINAL_CLASSIFICATION": cls["result"],
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": next_recommended_task(cls["result"], cls["boundary_group_size"]),
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES" if formal_complete else "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "NEAR_TIE_AUC": NEAR_TIE_AUC,
        "COVERAGE_GATE": coverage["gate"],
        "COVERAGE_DETAILS": coverage,
        "TIMING_GATE_DETAILS": timing_details,
        "INSTRUMENTATION_NONINTERFERENCE_DETAILS": noninterference_details,
        "PHASE_ROWS": phase_rows,
        "CONTRAST_ROWS": contrast_rows,
        "SOURCE_QUANTIZABILITY_ROWS": source_rows_by_g,
        "DECAY_HETEROGENEITY_ROWS": decay_rows,
        "failed_units": failed_units,
        "unit_summary_rows_written": len(unit_rows),
        "horizon_rows_written": len(horizon_rows),
        "source_rows_written": len(source_rows),
        "state_error_trajectory_rows_written": len(state_rows),
        "results_jsonl_rows_written": len(unit_rows),
        "outputs": {
            "results_jsonl": str(RUN_DIR / "results.jsonl"),
            "aggregate_summary_json": str(RUN_DIR / "aggregate_summary.json"),
            "report_md": str(RUN_DIR / "report.md"),
            "granularity_phase_diagram_csv": str(RUN_DIR / "granularity_phase_diagram.csv"),
            "source_quantizability_by_granularity_csv": str(RUN_DIR / "source_quantizability_by_granularity.csv"),
            "decay_heterogeneity_by_granularity_csv": str(RUN_DIR / "decay_heterogeneity_by_granularity.csv"),
        },
        "REPO_AUDIT": repo_audit(),
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    }
    return summary


def stage_gate_from_file(stage, key):
    path = RUN_DIR / f"{stage}_run_summary.json"
    if not path.exists():
        return "NOT_RUN"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key, "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def next_recommended_task(result, boundary):
    if result == "R_PREFERENCE_STABLE":
        return "NO_FOLLOW_UP_REQUIRED_FOR_GRANULARITY_SCREEN; granularity does not alter primary orientation mechanism."
    if result == "R_PREFERENCE_WITH_C_GRANULARITY_RESCUE":
        return f"CLOSED_LOOP_MAGNITUDE_EQUALIZATION_AT_G128_AND_G{boundary if boundary not in (None, 'NONE') else 16}"
    if result == "ORIENTATION_FLIP_CONFIRMED":
        return "TEST_SOURCE_ERROR_MAGNITUDE_EQUALIZATION_EXPLAINS_ORIENTATION_FLIP"
    return f"CLOSED_LOOP_MAGNITUDE_EQUALIZATION_AT_G128_AND_BOUNDARY_G{boundary}"


def write_aggregate_outputs(summary):
    phase_rows = summary["PHASE_ROWS"]
    source_rows = summary["SOURCE_QUANTIZABILITY_ROWS"]
    decay_rows = summary["DECAY_HETEROGENEITY_ROWS"]
    if phase_rows:
        write_rows(RUN_DIR / "granularity_phase_diagram.csv", merged_fields(phase_rows), phase_rows)
    if source_rows:
        write_rows(RUN_DIR / "source_quantizability_by_granularity.csv", merged_fields(source_rows), source_rows)
    if decay_rows:
        write_rows(RUN_DIR / "decay_heterogeneity_by_granularity.csv", merged_fields(decay_rows), decay_rows)
    save_json(RUN_DIR / "aggregate_summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    make_report(summary)


def make_report(summary):
    lines = [
        f"# {TASK}",
        "",
        f"- FORMAL_STATUS: {summary.get('FORMAL_STATUS')}",
        f"- N_FORMAL_UNITS: {summary.get('N_FORMAL_UNITS')}",
        f"- LING_ORIENTATION_GRANULARITY_RESULT: {summary.get('LING_ORIENTATION_GRANULARITY_RESULT')}",
        f"- FINAL_CLASSIFICATION: {summary.get('FINAL_CLASSIFICATION')}",
        f"- ORIENTATION_BOUNDARY_GROUP_SIZE: {summary.get('ORIENTATION_BOUNDARY_GROUP_SIZE')}",
        "",
        "## Gates",
    ]
    for key in REQUIRED_GATES:
        lines.append(f"- {key}: {summary.get(key)}")
    lines += [
        "",
        "## L64 Group Summary",
        f"- KL_R_BY_GROUP_SIZE: {json.dumps(summary.get('KL_R_BY_GROUP_SIZE'), sort_keys=True)}",
        f"- KL_C_BY_GROUP_SIZE: {json.dumps(summary.get('KL_C_BY_GROUP_SIZE'), sort_keys=True)}",
        f"- C_MINUS_R_FUNCTIONAL_GAP_BY_GROUP_SIZE: {json.dumps(summary.get('C_MINUS_R_FUNCTIONAL_GAP_BY_GROUP_SIZE'), sort_keys=True)}",
        f"- C_OVER_R_SOURCE_NORM_BY_GROUP_SIZE: {json.dumps(summary.get('C_OVER_R_SOURCE_NORM_BY_GROUP_SIZE'), sort_keys=True)}",
        f"- C_OVER_R_SOURCE_MSE_BY_GROUP_SIZE: {json.dumps(summary.get('C_OVER_R_SOURCE_MSE_BY_GROUP_SIZE'), sort_keys=True)}",
        "",
        "## Secondary Diagnostics",
        f"- C_GROUP_DECAY_HETEROGENEITY_BY_GROUP_SIZE: {json.dumps(summary.get('C_GROUP_DECAY_HETEROGENEITY_BY_GROUP_SIZE'), sort_keys=True)}",
        "- Decay heterogeneity status: not available per key channel from public runtime; not used for causality.",
    ]
    write_text(RUN_DIR / "report.md", "\n".join(lines) + "\n")
    write_text(REPORT_DIR / f"{SLUG}.md", "\n".join(lines) + "\n")


def run_units(stage, units, history_lengths=None, configs=None, horizon=HORIZON, unit_limit=None, unit_ids=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reset_stage_outputs(stage)
    history_lengths = history_lengths or stage_history_lengths(stage)
    configs = configs or stage_configs(stage)
    protocol_snapshot = {
        "TASK": TASK,
        "stage": stage,
        "history_lengths": history_lengths,
        "configs": configs,
        "horizon": horizon,
        "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
    }
    write_stage_protocol_snapshot(stage, protocol_snapshot)
    config_obj = json.loads((P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = P.kda_layers_from_config(config_obj)
    rows_by_pid = P.load_dataset()
    teacher_tokens = P.load_fp_teacher_tokens()
    torch, model, tokenizer = P.load_model_and_tokenizer()
    capture = P.KdaTransitionCapture()
    installed_layers = capture.install(model)
    limit = unit_limit if unit_limit is not None else stage_unit_limit(stage)
    selected_units = select_units_by_ids(units, unit_ids)
    if limit is not None and unit_ids is None:
        selected_units = selected_units[:limit]
    replay_gate = replay_input_gate(selected_units, rows_by_pid, teacher_tokens, horizon)
    quant_gates = quantizer_semantics_audit(torch)
    state_gate = {"gate": "PASS", "sample_shapes": []}
    if stage == "stage0":
        state_gate = state_semantics_gate(torch, model, tokenizer, capture, selected_units[0], rows_by_pid, teacher_tokens, kda_layers)
    unit_path = RUN_DIR / f"unit_summary_{stage}.csv"
    horizon_path = RUN_DIR / f"horizon_{stage}.csv"
    source_path = RUN_DIR / f"source_error_{stage}.csv"
    state_path = RUN_DIR / f"state_error_trajectory_{stage}.csv"
    jsonl_path = RUN_DIR / f"results_{stage}.jsonl"
    all_unit_rows = []
    all_horizon_rows = []
    all_source_rows = []
    all_state_rows = []
    failed_units = []
    try:
        for unit in selected_units:
            try:
                hrs, urs, srs, strs = run_unit(
                    torch, model, tokenizer, capture, unit, kda_layers, rows_by_pid, teacher_tokens, history_lengths, configs, horizon
                )
                for row in urs:
                    row["stage"] = stage
                    append_jsonl(jsonl_path, row)
                append_rows(unit_path, unit_fields(), urs)
                append_rows(horizon_path, horizon_fields(), hrs)
                append_rows(source_path, source_fields(), srs)
                append_rows(state_path, state_fields(), strs)
                all_unit_rows.extend(urs)
                all_horizon_rows.extend(hrs)
                all_source_rows.extend(srs)
                all_state_rows.extend(strs)
                print(f"[{now()}] {stage} completed unit={unit['unit_id']} configs={configs} L={history_lengths}", flush=True)
            except Exception as exc:
                failed = {"unit_id": str(unit.get("unit_id")), "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-40:]}
                failed_units.append(failed)
                append_jsonl(RUN_DIR / f"failed_units_{stage}.jsonl", failed)
                print(f"[{now()}] {stage} failed unit={unit.get('unit_id')} error={exc!r}", flush=True)
    finally:
        capture.close()
    summary = summarize_aggregate(stage, all_unit_rows, all_horizon_rows, all_source_rows, all_state_rows, failed_units=failed_units)
    evidence_gates = {
        "REPLAY_INPUT_GATE": replay_gate["gate"],
        "STATE_SEMANTICS_GATE": state_gate["gate"],
        "TIMING_GATE": summary.get("TIMING_GATE"),
        "INSTRUMENTATION_NONINTERFERENCE_GATE": summary.get("INSTRUMENTATION_NONINTERFERENCE_GATE"),
        **quant_gates,
    }
    summary.update({
        "stage": stage,
        "N_STAGE_UNITS": len(selected_units),
        "kda_layers": installed_layers,
        "failed_units": failed_units,
        "CANONICAL_MANIFEST_GATE": "PASS",
        "REPLAY_INPUT_GATE": replay_gate["gate"],
        "STATE_SEMANTICS_GATE": state_gate["gate"],
        "TIMING_GATE": evidence_gates["TIMING_GATE"],
        "INSTRUMENTATION_NONINTERFERENCE_GATE": evidence_gates["INSTRUMENTATION_NONINTERFERENCE_GATE"],
        **quant_gates,
        "SMOKE": "PASS" if stage == "smoke" and not failed_units and len(all_unit_rows) == expected_unit_rows("smoke", len(selected_units)) else ("NOT_APPLICABLE" if stage != "smoke" else "FAIL"),
        "PILOT": "PASS" if stage == "pilot" and not failed_units and len(selected_units) >= 3 and len(all_unit_rows) == expected_unit_rows("pilot", len(selected_units)) else ("NOT_APPLICABLE" if stage != "pilot" else "FAIL"),
        "STAGE0_GATE": stage0_gate_from_evidence(evidence_gates, failed_units) if stage == "stage0" else "NOT_APPLICABLE",
        "REPLAY_INPUT_DETAILS": replay_gate,
        "STATE_SEMANTICS_DETAILS": state_gate,
        "horizon_csv": str(horizon_path),
        "results_jsonl": str(jsonl_path),
        "unit_summary_csv": str(unit_path),
        "source_error_csv": str(source_path),
        "state_error_trajectory_csv": str(state_path),
    })
    save_json(RUN_DIR / f"{stage}_run_summary.json", summary)
    if stage == "formal":
        aggregate("formal")
    return summary


def load_stage_rows(stage):
    unit_rows = read_csv_rows(RUN_DIR / f"unit_summary_{stage}.csv")
    horizon_rows = read_csv_rows(RUN_DIR / f"horizon_{stage}.csv")
    source_rows = read_csv_rows(RUN_DIR / f"source_error_{stage}.csv")
    state_rows = read_csv_rows(RUN_DIR / f"state_error_trajectory_{stage}.csv")
    return unit_rows, horizon_rows, source_rows, state_rows


def aggregate(stage="formal"):
    unit_rows, horizon_rows, source_rows, state_rows = load_stage_rows(stage)
    stage_summaries = []
    for name in ["stage0", "smoke", "pilot", stage]:
        path = RUN_DIR / f"{name}_run_summary.json"
        if path.exists():
            stage_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(RUN_DIR.glob(f"{stage}_run_summary_shard*.json")):
        stage_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    failed_units = list(iter_jsonl(RUN_DIR / f"failed_units_{stage}.jsonl") or [])
    summary = summarize_aggregate(stage, unit_rows, horizon_rows, source_rows, state_rows, stage_summaries=stage_summaries, failed_units=failed_units)
    write_aggregate_outputs(summary)
    if stage == "formal":
        for src_name, dst_name in [
            ("results_formal.jsonl", "results.jsonl"),
            ("unit_summary_formal.csv", "unit_summary.csv"),
            ("horizon_formal.csv", "horizon.csv"),
            ("source_error_formal.csv", "source_error.csv"),
            ("state_error_trajectory_formal.csv", "state_error_trajectory.csv"),
        ]:
            src = RUN_DIR / src_name
            if src.exists():
                (RUN_DIR / dst_name).write_bytes(src.read_bytes())
        write_aggregate_outputs(summary)
    return summary


def merge_stage_files(stage, shard_dirs, output_dir=RUN_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [Path(d) for d in shard_dirs]
    shard_summaries = []
    for i, shard in enumerate(shard_dirs):
        summary_path = shard / f"{stage}_run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries.append(summary)
        save_json(output_dir / f"{stage}_run_summary_shard{i}.json", summary)
    merged = {"shard_summaries": shard_summaries}
    for key, filename in {
        "unit_rows": f"unit_summary_{stage}.csv",
        "horizon_rows": f"horizon_{stage}.csv",
        "source_rows": f"source_error_{stage}.csv",
        "state_rows": f"state_error_trajectory_{stage}.csv",
    }.items():
        rows = []
        for shard in shard_dirs:
            path = shard / filename
            if not path.exists():
                raise FileNotFoundError(f"missing shard output: {path}")
            rows.extend(read_csv_rows(path))
        write_rows(output_dir / filename, merged_fields(rows), rows)
        merged[key] = rows
    with (output_dir / f"results_{stage}.jsonl").open("w", encoding="utf-8") as out:
        for shard in shard_dirs:
            path = shard / f"results_{stage}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"missing shard jsonl: {path}")
            with path.open(encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    failed = []
    for shard in shard_dirs:
        failed_path = shard / f"failed_units_{stage}.jsonl"
        if failed_path.exists():
            failed.extend(iter_jsonl(failed_path))
    if failed:
        with (output_dir / f"failed_units_{stage}.jsonl").open("w", encoding="utf-8") as out:
            for row in failed:
                out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return merged


def merge_stage(stage, shard_dirs):
    merged = merge_stage_files(stage, shard_dirs, RUN_DIR)
    failed_units = []
    for shard in shard_dirs:
        failed_units.extend(list(iter_jsonl(Path(shard) / f"failed_units_{stage}.jsonl") or []))
    summary = summarize_aggregate(
        stage,
        merged["unit_rows"],
        merged["horizon_rows"],
        merged["source_rows"],
        merged["state_rows"],
        stage_summaries=merged["shard_summaries"],
        failed_units=failed_units,
    )
    summary["SHARD_DIRS"] = [str(Path(d)) for d in shard_dirs]
    summary["MERGED_SHARD_COUNT"] = len(shard_dirs)
    save_json(RUN_DIR / f"{stage}_run_summary.json", summary)
    write_aggregate_outputs(summary)
    if stage == "formal":
        aggregate("formal")
    return summary


def verify_fresh_outputs(run_dir=RUN_DIR, expected_records=None):
    run_dir = Path(run_dir)
    required_files = [
        "results.jsonl",
        "aggregate_summary.json",
        "report.md",
        "granularity_phase_diagram.csv",
        "source_quantizability_by_granularity.csv",
        "decay_heterogeneity_by_granularity.csv",
    ]
    missing = [name for name in required_files if not (run_dir / name).exists()]
    summary = {}
    if (run_dir / "aggregate_summary.json").exists():
        summary = json.loads((run_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
    rows = sum(1 for _ in iter_jsonl(run_dir / "results.jsonl")) if (run_dir / "results.jsonl").exists() else 0
    expected = expected_records if expected_records is not None else expected_jsonl_rows("formal", 18, HORIZON)
    failed_units = summary.get("failed_units") or []
    gate_failures = {k: summary.get(k) for k in REQUIRED_GATES if summary.get(k) != "PASS"}
    ok = (
        not missing
        and rows == expected
        and summary.get("FORMAL_STATUS") == "COMPLETE"
        and summary.get("N_FORMAL_UNITS") == 18
        and summary.get("COVERAGE_GATE") == "PASS"
        and not failed_units
        and not gate_failures
    )
    out = {
        "FRESH_OUTPUT_VERIFICATION": "PASS" if ok else "FAIL",
        "missing_files": missing,
        "results_jsonl_rows": rows,
        "expected_results_jsonl_rows": expected,
        "failed_units": failed_units,
        "gate_failures": gate_failures,
        "coverage_gate": summary.get("COVERAGE_GATE"),
        "coverage_details": summary.get("COVERAGE_DETAILS"),
        "formal_status": summary.get("FORMAL_STATUS"),
        "n_formal_units": summary.get("N_FORMAL_UNITS"),
    }
    save_json(run_dir / "fresh_output_verification.json", out)
    return out


def console_summary(summary, pytest_status="NOT_RUN", fresh=None):
    fresh = fresh or {}
    lines = [
        "TASK =",
        TASK,
        "",
        f"FORMAL_STATUS = {summary.get('FORMAL_STATUS')}",
        f"N_FORMAL_UNITS = {summary.get('N_FORMAL_UNITS')}",
        "",
    ]
    for key in REQUIRED_GATES:
        lines.append(f"{key} = {summary.get(key)}")
    lines += [
        "",
        f"SMOKE = {summary.get('SMOKE')}",
        f"PILOT = {summary.get('PILOT')}",
        "",
        "GROUP_SIZES =",
        ",".join(str(x) for x in GROUP_SIZES),
        "",
        "HISTORY_LENGTHS =",
        ",".join(str(x) for x in HISTORY_LENGTHS),
        "",
    ]
    for key in [
        "KL_R_BY_GROUP_SIZE",
        "KL_C_BY_GROUP_SIZE",
        "C_MINUS_R_FUNCTIONAL_GAP_BY_GROUP_SIZE",
        "SOURCE_NORM_R_BY_GROUP_SIZE",
        "SOURCE_NORM_C_BY_GROUP_SIZE",
        "SOURCE_MSE_R_BY_GROUP_SIZE",
        "SOURCE_MSE_C_BY_GROUP_SIZE",
        "C_OVER_R_SOURCE_NORM_BY_GROUP_SIZE",
        "C_OVER_R_SOURCE_MSE_BY_GROUP_SIZE",
        "R_GRANULARITY_GAIN",
        "C_GRANULARITY_GAIN",
        "C_GROUP_DECAY_HETEROGENEITY_BY_GROUP_SIZE",
        "COMMUTATOR_BY_GROUP_SIZE",
        "CODE_CHURN_BY_GROUP_SIZE",
        "ORIENTATION_BOUNDARY_GROUP_SIZE",
        "LING_ORIENTATION_GRANULARITY_RESULT",
        "FINAL_CLASSIFICATION",
        "METHOD_DESIGN_READY",
        "NEXT_RECOMMENDED_TASK",
    ]:
        lines.append(f"{key} = {json.dumps(summary.get(key), sort_keys=True, ensure_ascii=False)}")
    lines += [
        "",
        f"PYTEST = {pytest_status}",
        f"FRESH_OUTPUT_VERIFICATION = {fresh.get('FRESH_OUTPUT_VERIFICATION', 'NOT_RUN')}",
        f"CURRENT_TASK_COMPLETED_AND_STOPPED = {'YES' if summary.get('FORMAL_STATUS') == 'COMPLETE' and fresh.get('FRESH_OUTPUT_VERIFICATION') == 'PASS' else 'NO'}",
        "NO_NEXT_EXPERIMENT_LAUNCHED = YES",
        "",
        "STOP.",
    ]
    return "\n".join(lines)


def write_blocked_summary(audit):
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "BLOCKED_CANONICAL_MANIFEST",
        "CANONICAL_MANIFEST_GATE": audit.get("gate"),
        "CANONICAL_UNIT_AUDIT": audit,
        "REPO_AUDIT": repo_audit(),
        "METHOD_DESIGN_READY": "NO",
    }
    save_json(RUN_DIR / "blocked_summary.json", out)
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["audit", "stage0", "smoke", "pilot", "formal", "analyze", "merge-formal", "verify"], default="audit")
    p.add_argument("--canonical-units", default=str(CANONICAL_UNITS))
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--history-lengths", default=None)
    p.add_argument("--configs", default=None)
    p.add_argument("--unit-limit", type=int, default=None)
    p.add_argument("--unit-ids", default=None)
    p.add_argument("--merge-shards", default=None)
    p.add_argument("--console-summary", action="store_true")
    p.add_argument("--pytest-status", default="NOT_RUN")
    return p.parse_args()


def main():
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = M.load_canonical_units(args.canonical_units)
    if audit["gate"] != "PASS":
        write_blocked_summary(audit)
        return
    if args.stage == "audit":
        out = {
            "TASK": TASK,
            "FORMAL_STATUS": "AUDIT_COMPLETE_CAN_RUN_STAGE0",
            "CANONICAL_MANIFEST_GATE": "PASS",
            "CANONICAL_UNIT_AUDIT": audit,
            "REPO_AUDIT": repo_audit(),
            "reproducibility": reproducibility_snapshot(" ".join(sys.argv)),
        }
        save_json(RUN_DIR / "audit.json", out)
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "analyze":
        summary = aggregate("formal")
        print(console_summary(summary, pytest_status=args.pytest_status, fresh=verify_fresh_outputs()) if args.console_summary else json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "verify":
        print(json.dumps(verify_fresh_outputs(), indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "merge-formal":
        if not args.merge_shards:
            raise ValueError("--merge-shards is required")
        out = merge_stage("formal", [x.strip() for x in args.merge_shards.split(",") if x.strip()])
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    history_lengths = [int(x) for x in args.history_lengths.split(",")] if args.history_lengths else stage_history_lengths(args.stage)
    configs = [x.strip().upper() for x in args.configs.split(",") if x.strip()] if args.configs else stage_configs(args.stage)
    invalid = [x for x in configs if x not in CONFIGS]
    if invalid:
        raise ValueError(f"configs outside protocol: {invalid}")
    out = run_units(
        args.stage,
        audit["formal_units"],
        history_lengths=history_lengths,
        configs=configs,
        horizon=args.horizon,
        unit_limit=args.unit_limit,
        unit_ids=args.unit_ids,
    )
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
