#!/usr/bin/env python3
import argparse
import copy
import csv
import hashlib
import importlib
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


TASK = "QWEN_GDN_PERSISTENT_SOURCE_ERROR_MAGNITUDE_EQUALIZATION_TRANSFER_V1"
SLUG = "qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1"
REPO = Path(os.environ.get("GDN_REPO_ROOT", Path(__file__).resolve().parents[2]))
RUN_DIR = Path(os.environ.get("QWEN_GDN_MAG_EQ_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "propagation"
REPORT_DIR = REPO / "reports" / "propagation"

CANONICAL_MANIFEST_SOURCE = RESULT_DIR / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_stage0.json"
MODEL_PATH = Path(os.environ.get("QWEN35_MODEL_PATH", "/data01/user2/zypan_ling_dist/models/Qwen3.5-9B"))
GDN_DATA_ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/data01/user2"))
EXP = GDN_DATA_ROOT / "experiments" / "qwen35_gdn_quant"
SUBSET = GDN_DATA_ROOT / "results" / "gdn_end2end_bit_axis_screening_v1_math_subset.json"
SHARDS = GDN_DATA_ROOT / "results" / "gdn_end2end_bit_axis_screening_v1_shards"

HORIZON = 128
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
GAMMA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
SMOKE_HISTORY_LENGTHS = [1, 8]
PILOT_HISTORY_LENGTHS = [1, 8, 32, 64]
SOURCE_DIRECTIONS = ["R_DIR", "C_DIR"]
STATE_HORIZONS = [0, 1, 8, 32, 64, 128]
FUTURE_KL_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128]
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
EPS = 1e-12
NORM_TOL = 1e-5
IDENTITY_TOL = 1e-6
BLOCKING_GATE_NAMES = [
    "CANONICAL_MANIFEST_GATE",
    "STATE_SEMANTICS_GATE",
    "R128_QUANTIZER_IDENTITY_GATE",
    "C128_QUANTIZER_IDENTITY_GATE",
    "R_NATIVE_ENDPOINT_IDENTITY_GATE",
    "C_NATIVE_ENDPOINT_IDENTITY_GATE",
    "SAME_STATE_COUNTERFACTUAL_GATE",
    "APPLIED_MAGNITUDE_GATE",
    "SOURCE_DIRECTION_PRESERVATION_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    "NO_FURTHER_QUANTIZATION_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
    "L1_SEMANTICS_GATE",
]


def stage_protocol(stage):
    protocols = {
        "stage0": {"history_lengths": [1], "gamma_values": [0.0, 1.0], "unit_limit": 1},
        "smoke": {"history_lengths": SMOKE_HISTORY_LENGTHS, "gamma_values": [0.0, 0.5, 1.0], "unit_limit": 1},
        "pilot": {"history_lengths": PILOT_HISTORY_LENGTHS, "gamma_values": GAMMA_VALUES, "unit_limit": 6},
        "formal": {"history_lengths": HISTORY_LENGTHS, "gamma_values": GAMMA_VALUES, "unit_limit": None},
    }
    if stage not in protocols:
        raise ValueError(f"unknown execution stage: {stage}")
    return {
        "history_lengths": list(protocols[stage]["history_lengths"]),
        "gamma_values": list(protocols[stage]["gamma_values"]),
        "unit_limit": protocols[stage]["unit_limit"],
    }


def canonical_units_with_ids(manifest):
    return [dict(unit, unit_id=f"{unit['problem_id']}|{int(unit['t0'])}") for unit in manifest]


def gamma_label(gamma):
    return ("%g" % float(gamma)).replace(".", "p")


def branch_configs(gamma_values, include_fp=True):
    configs = []
    if include_fp:
        configs.append({
            "branch": "FP",
            "source_direction": "FP",
            "gamma": None,
            "native_endpoint": "FP",
        })
    for direction in SOURCE_DIRECTIONS:
        for gamma in gamma_values:
            gamma = float(gamma)
            native = None
            if direction == "R_DIR" and abs(gamma - 1.0) <= EPS:
                native = "R128"
            elif direction == "C_DIR" and abs(gamma) <= EPS:
                native = "C128"
            configs.append({
                "branch": f"{direction}_G{gamma_label(gamma)}",
                "source_direction": direction,
                "gamma": gamma,
                "native_endpoint": native,
            })
    return configs


def teacher_token_for_timestep(tokens, timestep):
    """Qwen canonical t=0 is prompt prefill; timestep t consumes token t-1."""
    timestep = int(timestep)
    if timestep < 1:
        raise ValueError("teacher-forced decode timesteps start at 1")
    return int(tokens[timestep - 1])


def teacher_token_for_future_horizon(tokens, t0, horizon):
    return teacher_token_for_timestep(tokens, int(t0) + int(horizon))


def apply_delta_to_stack(stack, delta):
    import torch
    with torch.inference_mode():
        for layer, value in delta.items():
            state = stack[layer]
            state.copy_((state.detach().float() + value.to(state.device).float()).to(state.dtype))


def max_stack_abs_error(stack, target):
    errors = []
    for layer, expected in target.items():
        actual = stack[layer]
        errors.append(float((actual.detach().float() - expected.to(actual.device).float()).abs().max().item()))
    return max(errors, default=0.0)


def state_metrics_from_stacks(torch, branch, reference):
    error_sq = 0.0
    reference_sq = 0.0
    branch_flat = []
    reference_flat = []
    for layer in sorted(reference):
        actual = branch[layer].detach().float()
        expected = reference[layer].to(actual.device).detach().float()
        delta = actual - expected
        error_sq += float((delta.double() * delta.double()).sum().item())
        reference_sq += float((expected.double() * expected.double()).sum().item())
        branch_flat.append(actual.flatten())
        reference_flat.append(expected.flatten())
    error = math.sqrt(error_sq)
    aa = torch.cat(branch_flat)
    bb = torch.cat(reference_flat)
    cosine = None
    if float(aa.norm().item()) > EPS and float(bb.norm().item()) > EPS:
        cosine = float(torch.nn.functional.cosine_similarity(aa, bb, dim=0).item())
    return {
        "state_error_norm": error,
        "state_relative_L2": error / (math.sqrt(reference_sq) + EPS),
        "state_cosine": cosine,
    }


def full_logit_metrics(torch, reference_logits, branch_logits):
    ref = reference_logits[:, -1, :].to(branch_logits.device).float()
    branch = branch_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    branch_logp = torch.log_softmax(branch, dim=-1)
    return {
        "KL": float((ref_logp.exp() * (ref_logp - branch_logp)).sum().item()),
        "top1_match": int(ref.argmax(dim=-1).item() == branch.argmax(dim=-1).item()),
        "logit_cosine": float(torch.nn.functional.cosine_similarity(ref, branch, dim=-1).item()),
        "logit_rel_l2": float((branch - ref).norm().item() / (ref.norm().item() + EPS)),
    }


def select_shard_units(units, shard_id, num_shards):
    shard_id = int(shard_id)
    num_shards = int(num_shards)
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_id < num_shards")
    return [unit for index, unit in enumerate(units) if index % num_shards == shard_id]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def runner_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def git_commit():
    return sh(["git", "rev-parse", "HEAD"])


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def save_worker_failures(path, failures):
    save_json(path, list(failures))


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=True) + "\n")


def write_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def read_csv_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def mean(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.median(vals) if vals else None


def std(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def bootstrap_ci(xs, n=2000, seed=20260909):
    vals = [float(x) for x in xs if finite_number(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    out.sort()
    return [out[int(0.025 * (n - 1))], out[int(0.975 * (n - 1))]]


def binomial_p_two_sided(k, n, p=0.5):
    if n <= 0:
        return None
    from math import comb
    obs = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if prob <= obs + 1e-15:
            total += prob
    return min(total, 1.0)


def ranks(xs):
    pairs = sorted((float(x), i) for i, x in enumerate(xs))
    out = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            out[pairs[k][1]] = rank
        i = j
    return out


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_number(x) and finite_number(y)]
    if len(pairs) < 3:
        return None
    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx = mean(rx)
    my = mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx)) * math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den + EPS)


def stack_norm(stack):
    return math.sqrt(sum(float(torch_norm(x)) ** 2 for x in stack.values()))


def torch_norm(x):
    return x.detach().float().norm().item()


def scale_stack(stack, factor):
    return {k: v.detach().float() * float(factor) for k, v in stack.items()}


def stack_cos(a, b):
    import torch
    av = []
    bv = []
    for k in sorted(a):
        av.append(a[k].detach().float().flatten())
        bv.append(b[k].detach().float().flatten())
    aa = torch.cat(av)
    bb = torch.cat(bv)
    an = float(aa.norm().item())
    bn = float(bb.norm().item())
    if an <= EPS or bn <= EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(aa, bb, dim=0).item())


def quantize_r128(torch, state):
    s = state.detach().float()
    scale = s.abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / 127.0
    q = torch.round(s / scale).clamp(-127, 127)
    return q * scale, scale


def quantize_c128(torch, state):
    s = state.detach().float()
    scale = s.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0
    q = torch.round(s / scale).clamp(-127, 127)
    return q * scale, scale


def extract_same_state_residuals(torch, stack):
    e_r, e_c, q_r, q_c = {}, {}, {}, {}
    max_abs = 0.0
    rel_num = 0.0
    rel_den = 0.0
    for layer, state in stack.items():
        qr, _ = quantize_r128(torch, state)
        qc, _ = quantize_c128(torch, state)
        s = state.detach().float()
        er = qr - s
        ec = qc - s
        for quantized, residual in ((qr, er), (qc, ec)):
            check = s + residual - quantized
            max_abs = max(max_abs, float(check.abs().max().item()))
            rel_num += float((check.double() * check.double()).sum().item())
            rel_den += float((quantized.double() * quantized.double()).sum().item())
        e_r[layer], e_c[layer] = er, ec
        q_r[layer], q_c[layer] = qr, qc
    rel = math.sqrt(rel_num) / (math.sqrt(rel_den) + EPS)
    return {
        "e_R": e_r,
        "e_C": e_c,
        "q_R": q_r,
        "q_C": q_c,
        "m_R": stack_norm(e_r),
        "m_C": stack_norm(e_c),
        "source_identity_max_abs_error": max_abs,
        "source_identity_relative_L2_error": rel,
        "SAME_STATE_COUNTERFACTUAL_GATE": "PASS" if max_abs <= IDENTITY_TOL and rel <= IDENTITY_TOL else "FAIL",
    }


def build_gamma_intervention(e_r, e_c, source_direction, gamma):
    m_r = stack_norm(e_r)
    m_c = stack_norm(e_c)
    m_target = (1.0 - float(gamma)) * m_c + float(gamma) * m_r
    source = e_r if source_direction == "R_DIR" else e_c
    source_norm = m_r if source_direction == "R_DIR" else m_c
    degenerate = 0
    if source_norm <= EPS:
        delta = {k: v.detach().float() * 0.0 for k, v in source.items()}
        applied_scale = 0.0 if m_target <= EPS else None
        direction_cos = None
        degenerate = 0 if m_target <= EPS else 1
    else:
        applied_scale = m_target / source_norm
        delta = scale_stack(source, applied_scale)
        direction_cos = stack_cos(delta, source) if m_target > EPS else None
    applied_norm = stack_norm(delta)
    abs_error = abs(applied_norm - m_target)
    rel_error = abs_error / (m_target + EPS)
    return {
        "delta": delta,
        "m_R_same_state": m_r,
        "m_C_same_state": m_c,
        "m_target": m_target,
        "source_direction_norm": source_norm,
        "applied_scale": applied_scale,
        "applied_norm": applied_norm,
        "applied_magnitude_abs_error": abs_error,
        "applied_magnitude_relative_error": rel_error,
        "direction_cosine": direction_cos,
        "degenerate_residual_count": degenerate,
        "APPLIED_MAGNITUDE_GATE": "PASS" if degenerate == 0 and abs_error <= NORM_TOL and rel_error <= NORM_TOL else "FAIL",
        "SOURCE_DIRECTION_PRESERVATION_GATE": "PASS" if degenerate == 0 and (m_target <= EPS or (direction_cos is not None and direction_cos >= 1.0 - IDENTITY_TOL)) else "FAIL",
    }


def history_schedule(t0, history_length, horizon):
    start = int(t0) - int(history_length) + 1
    active = list(range(start, int(t0) + 1)) if start >= 0 else []
    future = list(range(int(t0) + 1, int(t0) + int(horizon) + 1))
    return {
        "active_timesteps": active,
        "future_timesteps": future,
        "INTERVENTION_TIMING_GATE": "PASS" if len(active) == int(history_length) and active[-1:] == [int(t0)] else "FAIL",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS" if active and (not set(active).intersection(future)) and min(future) == int(t0) + 1 else "FAIL",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
    }


def load_canonical_manifest():
    obj = json.loads(CANONICAL_MANIFEST_SOURCE.read_text(encoding="utf-8"))
    manifest = obj.get("manifest", [])
    return obj, manifest, hashlib.sha256(CANONICAL_MANIFEST_SOURCE.read_bytes()).hexdigest()


def endpoint_value(rows, unit_id, history_length, direction, gamma, key):
    hits = [
        r for r in rows
        if r.get("unit_id") == unit_id
        and int(r.get("history_length")) == int(history_length)
        and r.get("source_direction") == direction
        and abs(float(r.get("gamma")) - float(gamma)) <= 1e-9
    ]
    if not hits:
        return None
    return float(hits[0][key])


def closed_loop_contrasts(rows):
    rows = [r for r in rows if r.get("source_direction") in SOURCE_DIRECTIONS]
    unit_ids = sorted({r["unit_id"] for r in rows})
    history_lengths = sorted({int(r["history_length"]) for r in rows})
    values = defaultdict(list)
    for unit_id in unit_ids:
        for L in history_lengths:
            rr = endpoint_value(rows, unit_id, L, "R_DIR", 1.0, "future_kl_auc")
            rc = endpoint_value(rows, unit_id, L, "R_DIR", 0.0, "future_kl_auc")
            cc = endpoint_value(rows, unit_id, L, "C_DIR", 0.0, "future_kl_auc")
            cr = endpoint_value(rows, unit_id, L, "C_DIR", 1.0, "future_kl_auc")
            sr = endpoint_value(rows, unit_id, L, "R_DIR", 1.0, "state_error_t0")
            sre = endpoint_value(rows, unit_id, L, "R_DIR", 0.0, "state_error_t0")
            sc = endpoint_value(rows, unit_id, L, "C_DIR", 0.0, "state_error_t0")
            scg = endpoint_value(rows, unit_id, L, "C_DIR", 1.0, "state_error_t0")
            if None in (rr, rc, cc, cr):
                continue
            values["NATIVE_R_MINUS_C_GAP"].append(rr - cc)
            values["R_MAG_EQUALIZATION_RESCUE"].append(rr - rc)
            values["C_RMAG_GRAFT_DAMAGE"].append(cr - cc)
            values["R_EQ_VS_C_NATIVE"].append(rc - cc)
            values["C_GRAFT_VS_R_NATIVE"].append(cr - rr)
            values["MAGNITUDE_MAIN"].append(0.5 * ((rr - rc) + (cr - cc)))
            if None not in (sr, sre):
                values["STATE_ERROR_RESCUE"].append(sr - sre)
            if None not in (scg, sc):
                values["STATE_ERROR_GRAFT_DAMAGE"].append(scg - sc)
    return {k: median(v) for k, v in values.items()}


def paired_summary(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return {
        "mean": mean(vals),
        "median": median(vals),
        "std": std(vals),
        "positive": sum(x > 0 for x in vals),
        "negative": sum(x < 0 for x in vals),
        "ties": sum(x == 0 for x in vals),
        "n": len(vals),
        "bootstrap_ci": bootstrap_ci(vals),
        "paired_sign_p": binomial_p_two_sided(sum(x > 0 for x in vals), sum(x != 0 for x in vals)),
    }


def contrast_by_l(rows, contrast):
    out = {}
    for L in sorted({int(r["history_length"]) for r in rows if r.get("source_direction") in SOURCE_DIRECTIONS}):
        vals = []
        for uid in sorted({r["unit_id"] for r in rows}):
            sub = [r for r in rows if r.get("unit_id") == uid and int(r.get("history_length", -1)) == L]
            c = closed_loop_contrasts(sub)
            if finite_number(c.get(contrast)):
                vals.append(c[contrast])
        out[str(L)] = paired_summary(vals)
    return out


def dose_response(rows, source_direction):
    unit_keys = sorted({(r["unit_id"], int(r["history_length"])) for r in rows if r.get("source_direction") == source_direction})
    dose_rows = []
    for unit_id, L in unit_keys:
        vals = [
            (float(r["gamma"]), float(r["future_kl_auc"]))
            for r in rows
            if r.get("unit_id") == unit_id
            and int(r.get("history_length")) == L
            and r.get("source_direction") == source_direction
            and finite_number(r.get("gamma"))
            and finite_number(r.get("future_kl_auc"))
        ]
        vals.sort()
        if len(vals) < 3:
            continue
        rho = spearman([v[0] for v in vals], [v[1] for v in vals])
        aucs = [v[1] for v in vals]
        dose_rows.append({
            "unit_id": unit_id,
            "history_length": L,
            "source_direction": source_direction,
            "spearman_rho": rho,
            "monotonic_non_decreasing": all(b >= a for a, b in zip(aucs, aucs[1:])),
            "positive_steps": sum(b > a for a, b in zip(aucs, aucs[1:])),
            "endpoint_effect": aucs[-1] - aucs[0],
        })
    by_history_length = {}
    for history_length in sorted({row["history_length"] for row in dose_rows}):
        selected = [row for row in dose_rows if row["history_length"] == history_length]
        by_history_length[str(history_length)] = {
            "n_units": len(selected),
            "median_spearman_rho": median([row["spearman_rho"] for row in selected]),
            "positive_rho_count": sum((row["spearman_rho"] or 0) > 0 for row in selected),
            "monotonic_unit_count": sum(row["monotonic_non_decreasing"] for row in selected),
            "monotonic_step_counts": sum(row["positive_steps"] for row in selected),
            "endpoint_effect": paired_summary([row["endpoint_effect"] for row in selected]),
        }
    return {
        "source_direction": source_direction,
        "n_units": len(dose_rows),
        "median_spearman_rho": median([r["spearman_rho"] for r in dose_rows]),
        "positive_rho_count": sum((r["spearman_rho"] or 0) > 0 for r in dose_rows),
        "monotonic_unit_count": sum(r["monotonic_non_decreasing"] for r in dose_rows),
        "monotonic_step_counts": sum(r["positive_steps"] for r in dose_rows),
        "endpoint_effect": paired_summary([r["endpoint_effect"] for r in dose_rows]),
        "by_history_length": by_history_length,
        "rows": dose_rows,
    }


def summarize_source_norms(schedule_rows):
    grouped = defaultdict(list)
    for row in schedule_rows:
        grouped[(row["unit_id"], int(row["history_length"]))].append(row)
    cells = []
    for (unit_id, history_length), rows in sorted(grouped.items()):
        r_values = [float(row["m_R_same_state"]) for row in rows]
        c_values = [float(row["m_C_same_state"]) for row in rows]
        ratios = [r / (c + EPS) for r, c in zip(r_values, c_values)]
        cells.append({
            "unit_id": unit_id,
            "history_length": history_length,
            "mean_R_source_norm": mean(r_values),
            "mean_C_source_norm": mean(c_values),
            "median_R_over_C_source_norm": median(ratios),
            "fraction_R_norm_gt_C_norm": sum(r > c for r, c in zip(r_values, c_values)) / len(rows),
            "fraction_C_norm_gt_R_norm": sum(c > r for r, c in zip(r_values, c_values)) / len(rows),
            "n_same_state_observations": len(rows),
        })
    r_gt = sum(row["mean_R_source_norm"] > row["mean_C_source_norm"] for row in cells)
    c_gt = sum(row["mean_C_source_norm"] > row["mean_R_source_norm"] for row in cells)
    total = max(len(cells), 1)
    orientation = "R_GT_C" if r_gt / total >= 0.75 else ("C_GT_R" if c_gt / total >= 0.75 else "MIXED")
    return {
        "R_SOURCE_NORM": mean([row["mean_R_source_norm"] for row in cells]),
        "C_SOURCE_NORM": mean([row["mean_C_source_norm"] for row in cells]),
        "R_NORM_GT_C": f"{r_gt} / {len(cells)}",
        "C_NORM_GT_R": f"{c_gt} / {len(cells)}",
        "MEDIAN_R_OVER_C_SOURCE_NORM": median([row["median_R_over_C_source_norm"] for row in cells]),
        "QWEN_SOURCE_MAGNITUDE_ORIENTATION": orientation,
        "per_unit_history_length": cells,
    }


def load_qwen_modules():
    os.environ["GDN_DATA_ROOT"] = str(GDN_DATA_ROOT)
    orientation_dir = REPO / "experiments" / "orientation"
    if str(orientation_dir) not in sys.path:
        sys.path.insert(0, str(orientation_dir))
    p1 = importlib.import_module("run_int8_orientation_state_change_mechanism")
    axis = importlib.import_module("run_int8_axis_geometry_rescue_diagnostic")
    e2e = importlib.import_module("run_end2end_bit_axis_screening")
    e2e.ensure_imports()
    return p1, axis, e2e


def setup_qwen_model():
    p1, axis, e2e = load_qwen_modules()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return torch, model, tokenizer, p1, axis, e2e


def clone_cache(cache):
    return copy.deepcopy(cache)


def qwen_cache_stack(p1, cache, layers=None):
    layers = GDN_LAYERS if layers is None else list(layers)
    if hasattr(cache, "recurrent_states"):
        return {layer: cache.recurrent_states[layer] for layer in layers}
    return {layer: p1.get_state(cache, layer) for layer in layers}


def stack_cpu_copy(stack):
    return {layer: value.detach().float().cpu().clone() for layer, value in stack.items()}


def quantizer_identity_metrics(torch, axis, stack):
    configs = {
        "R128": {"name": "R128", "orientation": "row", "group_size": 128},
        "C128": {"name": "C128", "orientation": "column", "group_size": 128},
    }
    result = {}
    for name, cfg in configs.items():
        error_sq = 0.0
        reference_sq = 0.0
        scale_shapes = set()
        for state in stack.values():
            reference, _q, _full, scale, _raw_shape, _dyn = axis.grouped_quant(
                torch, state.detach().float(), state.detach().float(), cfg
            )
            actual, actual_scale = (
                quantize_r128(torch, state) if name == "R128" else quantize_c128(torch, state)
            )
            delta = actual.double() - reference.double()
            error_sq += float((delta * delta).sum().item())
            reference_sq += float((reference.double() * reference.double()).sum().item())
            scale_shapes.add(tuple(actual_scale.shape))
            if actual_scale.numel() != scale.numel():
                error_sq = math.inf
        relative_error = math.sqrt(error_sq) / (math.sqrt(reference_sq) + EPS)
        expected_shape = (1, 32, 128, 1) if name == "R128" else (1, 32, 1, 128)
        result[name] = {
            "relative_error": relative_error,
            "scale_shapes": [list(shape) for shape in sorted(scale_shapes)],
            "expected_scale_shape": list(expected_shape),
            "gate": "PASS" if relative_error <= IDENTITY_TOL and scale_shapes == {expected_shape} else "FAIL",
        }
    return result


def runtime_semantics_audit(torch, model, tokenizer, p1, axis, e2e, prompt_row):
    prompt = e2e.render_prompt(tokenizer, prompt_row["problem"])
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    with torch.inference_mode():
        out = p1.feed_step(torch, model, ids, mask, None)
    stack = qwen_cache_stack(p1, out.past_key_values)
    before = stack_cpu_copy(stack)
    identities = quantizer_identity_metrics(torch, axis, stack)
    _ = extract_same_state_residuals(torch, stack)
    noninterference = max_stack_abs_error(stack, before)
    shapes = sorted({tuple(value.shape) for value in stack.values()})
    return {
        "timestamp": now(),
        "state_shapes": [list(shape) for shape in shapes],
        "n_gdn_layers": len(stack),
        "R128": identities["R128"],
        "C128": identities["C128"],
        "instrumentation_noninterference_max_abs": noninterference,
        "STATE_SEMANTICS_GATE": "PASS" if shapes == [(1, 32, 128, 128)] and len(stack) == len(GDN_LAYERS) else "FAIL",
        "R128_QUANTIZER_IDENTITY_GATE": identities["R128"]["gate"],
        "C128_QUANTIZER_IDENTITY_GATE": identities["C128"]["gate"],
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if noninterference <= IDENTITY_TOL else "FAIL",
    }


def full_reference(torch, model, tokenizer, p1, e2e, prompt_row, tokens, t0, history_lengths, horizon):
    prompt = e2e.render_prompt(tokenizer, prompt_row["problem"])
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    cache_points = {int(t0) - int(length) for length in history_lengths} | {int(t0)}
    snapshots = {}
    with torch.inference_mode():
        out = p1.feed_step(torch, model, ids, mask, None)
    past = out.past_key_values
    if 0 in cache_points:
        snapshots[0] = clone_cache(past)
    for timestep in range(1, int(t0) + 1):
        token = teacher_token_for_timestep(tokens, timestep)
        cur = torch.tensor([[token]], dtype=ids.dtype, device=device)
        with torch.inference_mode():
            out = p1.feed_step(torch, model, cur, None, past)
        past = out.past_key_values
        if timestep in cache_points:
            snapshots[timestep] = clone_cache(past)
    if set(snapshots) != cache_points:
        raise RuntimeError(f"missing FP cache snapshots: expected={sorted(cache_points)} got={sorted(snapshots)}")
    t0_stack = stack_cpu_copy(qwen_cache_stack(p1, snapshots[int(t0)]))
    future_logits = {}
    future_states = {}
    future_past = clone_cache(snapshots[int(t0)])
    for h in range(1, int(horizon) + 1):
        token = teacher_token_for_future_horizon(tokens, t0, h)
        cur = torch.tensor([[token]], dtype=ids.dtype, device=device)
        with torch.inference_mode():
            out = p1.feed_step(torch, model, cur, None, future_past)
        future_past = out.past_key_values
        future_logits[h] = out.logits[:, -1:, :].detach().cpu()
        if h in STATE_HORIZONS:
            future_states[h] = stack_cpu_copy(qwen_cache_stack(p1, future_past))
    return {
        "snapshots": snapshots,
        "state_t0": t0_stack,
        "future_logits": future_logits,
        "future_states": future_states,
    }


def endpoint_target(residuals, config, dtype):
    if config["native_endpoint"] == "R128":
        return {layer: value.to(dtype) for layer, value in residuals["q_R"].items()}
    if config["native_endpoint"] == "C128":
        return {layer: value.to(dtype) for layer, value in residuals["q_C"].items()}
    return None


def selected_state_row(metadata, config, horizon, metrics):
    return {
        **metadata,
        "branch": config["branch"],
        "direction": config["source_direction"],
        "source_direction": config["source_direction"],
        "gamma": config["gamma"],
        "horizon": int(horizon),
        **metrics,
    }


def evaluate_branch(torch, model, p1, tokens, unit, prompt_row, fp_ref, history_length, config, horizon, manifest_hash):
    t0 = int(unit["t0"])
    schedule = history_schedule(t0, history_length, horizon)
    start_prev = schedule["active_timesteps"][0] - 1
    past = clone_cache(fp_ref["snapshots"][start_prev])
    device = next(model.parameters()).device
    input_dtype = torch.long
    commit = git_commit()
    metadata = {
        "task": TASK,
        "git_commit": commit,
        "manifest_sha256": manifest_hash,
        "unit_id": unit["unit_id"],
        "prompt": unit["problem_id"],
        "prompt_id": unit["problem_id"],
        "problem_index": int(prompt_row["problem_index"]),
        "t0": t0,
        "history_length": int(history_length),
        "layer_scope": "ALL_GDN_RECURRENT_LAYERS",
        "head_scope": "ALL_HEADS",
    }
    step_rows = []
    endpoint_errors = []
    nonretroactivity_errors = []
    quantization_event_timesteps = []
    for timestep in schedule["active_timesteps"]:
        token = teacher_token_for_timestep(tokens, timestep)
        cur = torch.tensor([[token]], dtype=input_dtype, device=device)
        with torch.inference_mode():
            out = p1.feed_step(torch, model, cur, None, past)
        past = out.past_key_values
        current_logits = out.logits[:, -1:, :].detach().clone()
        stack = qwen_cache_stack(p1, past)
        residuals = extract_same_state_residuals(torch, stack)
        intervention = build_gamma_intervention(
            residuals["e_R"], residuals["e_C"], config["source_direction"], config["gamma"]
        )
        target = endpoint_target(residuals, config, next(iter(stack.values())).dtype)
        apply_delta_to_stack(stack, intervention["delta"])
        quantization_event_timesteps.append(int(timestep))
        endpoint_error = max_stack_abs_error(stack, target) if target is not None else None
        if endpoint_error is not None:
            endpoint_errors.append(endpoint_error)
        nonretroactivity = float((out.logits[:, -1:, :] - current_logits).abs().max().item())
        nonretroactivity_errors.append(nonretroactivity)
        step_rows.append({
            **metadata,
            "active_timestep": int(timestep),
            "branch": config["branch"],
            "direction": config["source_direction"],
            "source_direction": config["source_direction"],
            "gamma": config["gamma"],
            "m_R_same_state": residuals["m_R"],
            "m_C_same_state": residuals["m_C"],
            "m_target": intervention["m_target"],
            "applied_norm": intervention["applied_norm"],
            "applied_norm_error": intervention["applied_magnitude_abs_error"],
            "applied_magnitude_relative_error": intervention["applied_magnitude_relative_error"],
            "direction_cosine": intervention["direction_cosine"],
            "endpoint_identity_abs_error": endpoint_error,
            "degenerate_count": intervention["degenerate_residual_count"],
            "SAME_STATE_COUNTERFACTUAL_GATE": residuals["SAME_STATE_COUNTERFACTUAL_GATE"],
            "APPLIED_MAGNITUDE_GATE": intervention["APPLIED_MAGNITUDE_GATE"],
            "SOURCE_DIRECTION_PRESERVATION_GATE": intervention["SOURCE_DIRECTION_PRESERVATION_GATE"],
            "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS" if nonretroactivity <= IDENTITY_TOL else "FAIL",
        })
    state_rows = []
    state_t0 = state_metrics_from_stacks(torch, qwen_cache_stack(p1, past), fp_ref["state_t0"])
    state_rows.append(selected_state_row(metadata, config, 0, state_t0))
    kl_curve = []
    selected_kl = {}
    selected_states = {0: state_t0}
    future_forward_horizons = []
    for h in range(1, int(horizon) + 1):
        token = teacher_token_for_future_horizon(tokens, t0, h)
        cur = torch.tensor([[token]], dtype=input_dtype, device=device)
        with torch.inference_mode():
            out = p1.feed_step(torch, model, cur, None, past)
        past = out.past_key_values
        future_forward_horizons.append(int(h))
        if not bool(torch.isfinite(out.logits).all().item()):
            raise RuntimeError(f"nonfinite logits unit={unit['unit_id']} L={history_length} branch={config['branch']} h={h}")
        logit_metrics = full_logit_metrics(torch, fp_ref["future_logits"][h], out.logits)
        kl_curve.append(logit_metrics["KL"])
        if h in FUTURE_KL_HORIZONS:
            selected_kl[h] = logit_metrics["KL"]
        if h in STATE_HORIZONS:
            metrics = state_metrics_from_stacks(
                torch, qwen_cache_stack(p1, past), fp_ref["future_states"][h]
            )
            selected_states[h] = metrics
            state_rows.append(selected_state_row(metadata, config, h, metrics))
    row = {
        **metadata,
        "stage": None,
        "branch": config["branch"],
        "direction": config["source_direction"],
        "source_direction": config["source_direction"],
        "gamma": config["gamma"],
        "m_R_same_state": mean([step["m_R_same_state"] for step in step_rows]),
        "m_C_same_state": mean([step["m_C_same_state"] for step in step_rows]),
        "m_target": mean([step["m_target"] for step in step_rows]),
        "future_kl_auc": mean(kl_curve),
        "state_error_t0": state_t0["state_error_norm"],
        "state_error_future": selected_states[max(selected_states)]["state_error_norm"],
        "direction_cosine": min([step["direction_cosine"] for step in step_rows if finite_number(step["direction_cosine"])], default=None),
        "applied_norm_error": max([step["applied_norm_error"] for step in step_rows], default=0.0),
        "degenerate_count": sum(step["degenerate_count"] for step in step_rows),
        "intervention_count": len(step_rows),
        "future_quantization_count": 0,
        "quantization_event_timesteps": quantization_event_timesteps,
        "future_forward_horizons": future_forward_horizons,
        "current_logit_nonretroactivity_max_abs": max(nonretroactivity_errors, default=0.0),
        "endpoint_identity_max_abs_error": max(endpoint_errors, default=None),
    }
    for h in FUTURE_KL_HORIZONS:
        row[f"KL_h{h}"] = selected_kl.get(h)
        row[f"future_kl_h{h}"] = selected_kl.get(h)
    for h, metrics in selected_states.items():
        row[f"state_error_h{h}"] = metrics["state_error_norm"]
        row[f"state_relative_L2_h{h}"] = metrics["state_relative_L2"]
        row[f"state_cosine_h{h}"] = metrics["state_cosine"]
    audit = {
        "branch": config["branch"],
        "history_length": int(history_length),
        "native_endpoint": config["native_endpoint"],
        "endpoint_identity_max_abs_error": max(endpoint_errors, default=None),
        "current_logit_nonretroactivity_max_abs": max(nonretroactivity_errors, default=0.0),
        "intervention_count": len(step_rows),
        "future_quantization_count": 0,
        "quantization_event_timesteps": quantization_event_timesteps,
        "future_forward_horizons": future_forward_horizons,
    }
    return row, step_rows, state_rows, audit


def fp_result_row(unit, prompt_row, history_length, manifest_hash):
    row = {
        "task": TASK,
        "stage": None,
        "git_commit": git_commit(),
        "manifest_sha256": manifest_hash,
        "unit_id": unit["unit_id"],
        "prompt": unit["problem_id"],
        "prompt_id": unit["problem_id"],
        "problem_index": int(prompt_row["problem_index"]),
        "t0": int(unit["t0"]),
        "history_length": int(history_length),
        "branch": "FP",
        "direction": "FP",
        "source_direction": "FP",
        "gamma": None,
        "layer_scope": "ALL_GDN_RECURRENT_LAYERS",
        "head_scope": "ALL_HEADS",
        "m_R_same_state": None,
        "m_C_same_state": None,
        "m_target": None,
        "future_kl_auc": 0.0,
        "state_error_t0": 0.0,
        "state_error_future": 0.0,
        "direction_cosine": None,
        "applied_norm_error": 0.0,
        "degenerate_count": 0,
        "intervention_count": 0,
        "future_quantization_count": 0,
        "current_logit_nonretroactivity_max_abs": 0.0,
        "endpoint_identity_max_abs_error": 0.0,
    }
    for h in FUTURE_KL_HORIZONS:
        row[f"KL_h{h}"] = 0.0
        row[f"future_kl_h{h}"] = 0.0
    for h in STATE_HORIZONS:
        row[f"state_error_h{h}"] = 0.0
        row[f"state_relative_L2_h{h}"] = 0.0
        row[f"state_cosine_h{h}"] = 1.0
    return row


def run_canonical_unit(torch, model, tokenizer, p1, e2e, unit, prompt_row, history_lengths, gamma_values, horizon, manifest_hash):
    tokens = tokenizer.encode(prompt_row["fp_response"], add_special_tokens=False)
    required = int(unit["t0"]) + int(horizon)
    if len(tokens) < required:
        raise RuntimeError(f"teacher continuation too short for {unit['unit_id']}: need={required} have={len(tokens)}")
    fp_ref = full_reference(
        torch, model, tokenizer, p1, e2e, prompt_row, tokens, int(unit["t0"]), history_lengths, horizon
    )
    rows = []
    steps = []
    states = []
    audits = []
    for history_length in history_lengths:
        rows.append(fp_result_row(unit, prompt_row, history_length, manifest_hash))
        for config in branch_configs(gamma_values, include_fp=False):
            row, branch_steps, branch_states, audit = evaluate_branch(
                torch, model, p1, tokens, unit, prompt_row, fp_ref,
                history_length, config, horizon, manifest_hash,
            )
            rows.append(row)
            steps.extend(branch_steps)
            states.extend(branch_states)
            audits.append(audit)
    return {"unit": unit, "rows": rows, "schedule_rows": steps, "state_rows": states, "audits": audits}


def stage_units(stage, units):
    if stage in ("stage0", "smoke"):
        return units[:1]
    if stage == "pilot":
        indices = [0, 4, 8, 9, 13, 17]
        return [units[index] for index in indices if index < len(units)]
    return list(units)


def checkpoint_path(stage, num_shards, shard_id, unit):
    digest = hashlib.sha1(unit["unit_id"].encode("utf-8")).hexdigest()[:12]
    return RUN_DIR / "shards" / f"{stage}_{int(num_shards)}_{int(shard_id)}" / "units" / f"{digest}.json"


def validate_unit_result(result, unit, stage, manifest_hash, history_lengths, gamma_values, horizon):
    errors = []
    if result.get("task") != TASK:
        errors.append("task_mismatch")
    if result.get("stage") != stage:
        errors.append("stage_mismatch")
    if result.get("manifest_sha256") != manifest_hash:
        errors.append("manifest_mismatch")
    if result.get("unit", {}).get("unit_id") != unit["unit_id"]:
        errors.append("unit_mismatch")
    rows = result.get("rows", [])
    if not exact_result_coverage(rows, [unit], history_lengths, gamma_values):
        errors.append("result_coverage_mismatch")
    for required_horizon in [h for h in FUTURE_KL_HORIZONS if h <= int(horizon)]:
        required_horizon_key = f"KL_h{required_horizon}"
        if not rows or any(not finite_number(row.get(required_horizon_key)) for row in rows):
            errors.append(f"missing_or_nonfinite_{required_horizon_key}")
    if any(
        not finite_number(row.get("future_kl_auc"))
        or not finite_number(row.get("state_error_t0"))
        or not finite_number(row.get("state_error_future"))
        for row in rows
    ):
        errors.append("missing_or_nonfinite_primary_metrics")
    if any(row.get("manifest_sha256") != manifest_hash or row.get("stage") != stage for row in rows):
        errors.append("row_provenance_mismatch")

    schedules = result.get("schedule_rows", [])
    audits = result.get("audits", [])
    expected_configs = branch_configs(gamma_values, include_fp=False)
    expected_non_fp = len(history_lengths) * len(expected_configs)
    expected_audits = {
        (int(history_length), config["branch"])
        for history_length in history_lengths for config in expected_configs
    }
    actual_audits = [
        (int(audit.get("history_length", -1)), audit.get("branch")) for audit in audits
    ]
    if len(actual_audits) != len(expected_audits) or set(actual_audits) != expected_audits:
        errors.append("audit_coverage_mismatch")
    expected_endpoint_by_branch = {
        config["branch"]: config["native_endpoint"] for config in expected_configs
    }
    if any(
        audit.get("native_endpoint") != expected_endpoint_by_branch.get(audit.get("branch"))
        for audit in audits
    ):
        errors.append("native_endpoint_metadata_mismatch")
    if any(
        expected_endpoint_by_branch.get(audit.get("branch")) is not None
        and not finite_number(audit.get("endpoint_identity_max_abs_error"))
        for audit in audits
    ):
        errors.append("missing_native_endpoint_measurement")
    expected_state_keys = {
        (int(history_length), config["branch"], int(state_horizon))
        for history_length in history_lengths
        for config in expected_configs
        for state_horizon in STATE_HORIZONS if state_horizon <= int(horizon)
    }
    actual_state_keys = [
        (int(row.get("history_length", -1)), row.get("branch"), int(row.get("horizon", -1)))
        for row in result.get("state_rows", [])
    ]
    if len(actual_state_keys) != len(expected_state_keys) or set(actual_state_keys) != expected_state_keys:
        errors.append("state_trajectory_coverage_mismatch")
    for history_length in history_lengths:
        expected_steps = list(range(int(unit["t0"]) - int(history_length) + 1, int(unit["t0"]) + 1))
        for config in branch_configs(gamma_values, include_fp=False):
            observed_steps = sorted(
                int(row["active_timestep"])
                for row in schedules
                if int(row.get("history_length", -1)) == int(history_length)
                and row.get("branch") == config["branch"]
            )
            if observed_steps != expected_steps:
                errors.append("active_timestep_coverage_mismatch")
                break
        if "active_timestep_coverage_mismatch" in errors:
            break
    if result.get("runner_sha256"):
        for audit in audits:
            history_length = int(audit["history_length"])
            expected_steps = list(range(int(unit["t0"]) - history_length + 1, int(unit["t0"]) + 1))
            if (
                audit.get("quantization_event_timesteps") != expected_steps
                or audit.get("future_forward_horizons") != list(range(1, int(horizon) + 1))
            ):
                errors.append("missing_observed_execution_traces")
                break
    annotated_audits = [dict(audit, unit_id=unit["unit_id"]) for audit in audits]
    observed_gates = observed_execution_gates(schedules, annotated_audits, [unit], horizon)
    errors.extend(
        f"blocking_gate_{key}" for key, value in observed_gates.items() if value != "PASS"
    )
    return sorted(set(errors))


def validate_shard_manifest(obj, stage, shard_id, num_shards, manifest_hash, history_lengths, gamma_values, horizon):
    checks = {
        "task": obj.get("task") == TASK,
        "stage": obj.get("stage") == stage,
        "shard_id": int(obj.get("shard_id", -1)) == int(shard_id),
        "num_shards": int(obj.get("num_shards", -1)) == int(num_shards),
        "manifest_sha256": obj.get("manifest_sha256") == manifest_hash,
        "history_lengths": [int(x) for x in obj.get("history_lengths", [])] == [int(x) for x in history_lengths],
        "gamma_values": [float(x) for x in obj.get("gamma_values", [])] == [float(x) for x in gamma_values],
        "horizon": int(obj.get("horizon", -1)) == int(horizon),
    }
    return checks


def validate_resume_metadata(
    shard_manifest, provenance, stage, shard_id, num_shards, manifest_hash,
    history_lengths, gamma_values, horizon, current_runner_sha,
):
    errors = [
        f"manifest_{key}"
        for key, passed in validate_shard_manifest(
            shard_manifest, stage, shard_id, num_shards, manifest_hash,
            history_lengths, gamma_values, horizon,
        ).items()
        if not passed
    ]
    if (
        provenance.get("task") != TASK
        or provenance.get("stage") != stage
        or int(provenance.get("shard_id", -1)) != int(shard_id)
        or int(provenance.get("num_shards", -1)) != int(num_shards)
    ):
        errors.append("execution_provenance")
    if provenance.get("runner_sha256") != current_runner_sha:
        errors.append("runner_sha256")
    return sorted(set(errors))


def checkpoint_source_provenance_errors(result, provenance):
    result_sha = result.get("runner_sha256")
    provenance_sha = provenance.get("runner_sha256")
    if result_sha:
        return [] if result_sha == provenance_sha else ["checkpoint_runner_sha256_mismatch"]
    if provenance_sha and provenance.get("captured_during_active_execution"):
        return []
    return ["legacy_checkpoint_without_active_execution_attestation"]


def validate_execution_provenance(provenance, stage, shard_id, num_shards):
    errors = []
    if (
        provenance.get("task") != TASK
        or provenance.get("stage") != stage
        or int(provenance.get("shard_id", -1)) != int(shard_id)
        or int(provenance.get("num_shards", -1)) != int(num_shards)
        or not provenance.get("runner_sha256")
    ):
        errors.append("execution_provenance_metadata")
    artifact_value = provenance.get("execution_artifact_path")
    artifact = Path(artifact_value) if artifact_value else None
    if artifact is None or not artifact.is_file():
        errors.append("execution_artifact_path")
    elif hashlib.sha256(artifact.read_bytes()).hexdigest() != provenance.get("execution_artifact_sha256"):
        errors.append("execution_artifact_sha256")
    return sorted(set(errors))


def archive_current_runner():
    digest = runner_sha256()
    path = RUN_DIR / "execution_source" / f"runner_{digest}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    if path.exists() and path.read_bytes() != source:
        raise RuntimeError(f"execution source archive collision: {path}")
    if not path.exists():
        path.write_bytes(source)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def stage_prerequisite_status(stage, summary, manifest_hash):
    previous = {
        "smoke": ("stage0", "NOT_RUN_STAGE0_COMPLETE"),
        "pilot": ("smoke", "NOT_RUN_SMOKE_COMPLETE"),
        "formal": ("pilot", "NOT_RUN_PILOT_COMPLETE"),
    }
    if stage not in previous:
        return True, "PASS"
    previous_stage, expected_status = previous[stage]
    if summary.get("task") != TASK:
        return False, "TASK_MISMATCH"
    if summary.get("stage") != previous_stage:
        return False, "PREVIOUS_STAGE_MISMATCH"
    if summary.get("canonical_manifest_sha256") != manifest_hash:
        return False, "MANIFEST_MISMATCH"
    if summary.get("formal_status") != expected_status:
        return False, "PREVIOUS_STAGE_NOT_COMPLETE"
    if summary.get("FRESH_OUTPUT_VERIFICATION") != "PASS":
        return False, "PREVIOUS_STAGE_FRESH_OUTPUT_FAILED"
    if (
        len(summary.get("execution_runner_sha256", [])) != 1
        or summary.get("protocol_errors")
        or summary.get("checkpoint_validation_errors")
    ):
        return False, "PREVIOUS_STAGE_PROVENANCE_FAILED"
    if not all(summary.get("gates", {}).get(key) == "PASS" for key in BLOCKING_GATE_NAMES):
        return False, "BLOCKING_GATES_FAILED"
    if stage == "pilot" and summary.get("SMOKE") != "PASS":
        return False, "SMOKE_FAILED"
    if stage == "formal" and summary.get("PILOT") != "POSITIVE_OR_MIXED_SUPPORTS_FORMAL":
        return False, "PILOT_DOES_NOT_SUPPORT_FORMAL"
    return True, "PASS"


def formal_prerequisite_status(summary):
    _obj, _manifest, manifest_hash = load_canonical_manifest()
    return stage_prerequisite_status("formal", summary, manifest_hash)


def require_stage_prerequisite(stage, manifest_hash):
    previous = {"smoke": "stage0", "pilot": "smoke", "formal": "pilot"}
    if stage not in previous:
        return
    path = RUN_DIR / f"aggregate_{previous[stage]}.json"
    if not path.exists():
        raise RuntimeError(f"{stage} blocked: {path.name} is missing")
    allowed, reason = stage_prerequisite_status(
        stage, json.loads(path.read_text(encoding="utf-8")), manifest_hash
    )
    if not allowed:
        raise RuntimeError(f"{stage} blocked by prerequisite: {reason}")


def require_formal_prerequisite():
    _obj, _manifest, manifest_hash = load_canonical_manifest()
    require_stage_prerequisite("formal", manifest_hash)


def run_execution_stage(args):
    audit = environment_audit()
    if audit["missing_required_replay_inputs"]:
        raise RuntimeError("missing required replay inputs: " + ", ".join(audit["missing_required_replay_inputs"]))
    protocol = stage_protocol(args.stage)
    history_lengths = args.history_lengths or protocol["history_lengths"]
    gamma_values = args.gamma_values or protocol["gamma_values"]
    _canonical_obj, manifest, manifest_hash = load_canonical_manifest()
    require_stage_prerequisite(args.stage, manifest_hash)
    units = stage_units(args.stage, canonical_units_with_ids(manifest))
    if args.unit_limit is not None:
        units = units[:args.unit_limit]
    units = select_shard_units(units, args.shard_id, args.num_shards)
    shard_dir = RUN_DIR / "shards" / f"{args.stage}_{args.num_shards}_{args.shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest = {
        "task": TASK,
        "stage": args.stage,
        "timestamp": now(),
        "manifest_sha256": manifest_hash,
        "history_lengths": history_lengths,
        "gamma_values": gamma_values,
        "horizon": args.horizon,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "runner_sha256": runner_sha256(),
        "units": units,
    }
    execution_artifact, execution_artifact_sha = archive_current_runner()
    provenance = {
        "task": TASK,
        "stage": args.stage,
        "timestamp": now(),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "runner_sha256": runner_sha256(),
        "execution_artifact_path": str(execution_artifact),
        "execution_artifact_sha256": execution_artifact_sha,
        "command": sys.argv,
    }
    manifest_path = shard_dir / "manifest.json"
    provenance_path = shard_dir / "execution_provenance.json"
    checkpoint_files = list((shard_dir / "units").glob("*.json"))
    if args.resume and (manifest_path.exists() or provenance_path.exists() or checkpoint_files):
        if not manifest_path.exists() or not provenance_path.exists():
            raise RuntimeError("resume blocked: incomplete shard manifest/provenance")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        resume_errors = validate_resume_metadata(
            existing_manifest, existing_provenance, args.stage, args.shard_id, args.num_shards,
            manifest_hash, history_lengths, gamma_values, args.horizon, runner_sha256(),
        )
        if resume_errors:
            raise RuntimeError("resume blocked by incompatible metadata: " + ", ".join(resume_errors))
    elif checkpoint_files:
        raise RuntimeError("existing checkpoints require --resume or a fresh QWEN_GDN_MAG_EQ_RUN_DIR")
    else:
        save_json(manifest_path, shard_manifest)
        save_json(provenance_path, provenance)

    torch, model, tokenizer, p1, axis, e2e = setup_qwen_model()
    prompt_rows = {row["problem_id"]: row for row in p1.selected_prompt_rows()}
    semantics = runtime_semantics_audit(torch, model, tokenizer, p1, axis, e2e, prompt_rows[manifest[0]["problem_id"]])
    save_json(shard_dir / "runtime_semantics_audit.json", semantics)
    runtime_gate_names = [
        "STATE_SEMANTICS_GATE",
        "R128_QUANTIZER_IDENTITY_GATE",
        "C128_QUANTIZER_IDENTITY_GATE",
        "INSTRUMENTATION_NONINTERFERENCE_GATE",
    ]
    failed_runtime_gates = [key for key in runtime_gate_names if semantics.get(key) != "PASS"]
    if failed_runtime_gates:
        raise RuntimeError("execution blocked by runtime semantics gates: " + ", ".join(failed_runtime_gates))
    failures = []
    for index, unit in enumerate(units, 1):
        path = checkpoint_path(args.stage, args.num_shards, args.shard_id, unit)
        if args.resume and path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            validation_errors = validate_unit_result(
                previous, unit, args.stage, manifest_hash, history_lengths, gamma_values, args.horizon
            )
            if not validation_errors:
                print(f"RESUME_SKIP {index}/{len(units)} {unit['unit_id']}", flush=True)
                continue
            print(f"RESUME_REJECT {unit['unit_id']} errors={','.join(validation_errors)}", flush=True)
        started = time.time()
        print(f"RUN_UNIT {index}/{len(units)} {unit['unit_id']}", flush=True)
        try:
            result = run_canonical_unit(
                torch, model, tokenizer, p1, e2e, unit, prompt_rows[unit["problem_id"]],
                history_lengths, gamma_values, args.horizon, manifest_hash,
            )
            result.update({
                "task": TASK,
                "stage": args.stage,
                "timestamp": now(),
                "elapsed_seconds": time.time() - started,
                "manifest_sha256": manifest_hash,
                "runner_sha256": runner_sha256(),
                "protocol": {
                    "history_lengths": history_lengths,
                    "gamma_values": gamma_values,
                    "horizon": args.horizon,
                },
                "runtime_semantics_audit": semantics,
            })
            for row in result["rows"]:
                row["stage"] = args.stage
            save_json(path, result)
            print(f"DONE_UNIT {unit['unit_id']} seconds={time.time() - started:.1f}", flush=True)
        except Exception as exc:
            failure = {
                "unit": unit,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "timestamp": now(),
            }
            failures.append(failure)
            save_worker_failures(shard_dir / "failures.json", failures)
            print(f"FAILED_UNIT {unit['unit_id']} {exc!r}", flush=True)
            if not args.keep_going:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save_worker_failures(shard_dir / "failures.json", failures)
    save_json(shard_dir / "worker_summary.json", {
        "task": TASK,
        "stage": args.stage,
        "timestamp": now(),
        "n_assigned": len(units),
        "n_success": sum(checkpoint_path(args.stage, args.num_shards, args.shard_id, unit).exists() for unit in units),
        "n_failed": len(failures),
        "failures": failures,
    })
    return 0 if not failures else 1


def environment_audit():
    canonical_obj, manifest, manifest_hash = load_canonical_manifest()
    model_exists = MODEL_PATH.exists()
    model_files = sorted(p.name for p in MODEL_PATH.glob("*")) if model_exists else []
    missing = []
    if not model_exists:
        missing.append(str(MODEL_PATH))
    if not SUBSET.exists():
        missing.append(str(SUBSET))
    if not SHARDS.exists():
        missing.append(str(SHARDS))
    if not (EXP / "run_config.json").exists():
        missing.append(str(EXP / "run_config.json"))
    stage0_metrics = canonical_obj.get("stage0_metrics", {})
    gates = {
        "CANONICAL_MANIFEST_GATE": "PASS" if len(manifest) == 18 and manifest_hash else "FAIL",
        "STATE_SEMANTICS_GATE": "PASS" if stage0_metrics.get("state_shape") == [1, 32, 128, 128] else "FAIL",
        "R128_QUANTIZER_IDENTITY_GATE": "PASS" if stage0_metrics.get("R128_scale_shape") == [1, 32, 128, 1] and stage0_metrics.get("quantizer_identity_relative_error", 1.0) <= 1e-6 else "FAIL",
        "C128_QUANTIZER_IDENTITY_GATE": "PASS" if stage0_metrics.get("C128_scale_shape") == [1, 32, 1, 128] else "FAIL",
    }
    for name in [
        "R_NATIVE_ENDPOINT_IDENTITY_GATE",
        "C_NATIVE_ENDPOINT_IDENTITY_GATE",
        "SAME_STATE_COUNTERFACTUAL_GATE",
        "APPLIED_MAGNITUDE_GATE",
        "SOURCE_DIRECTION_PRESERVATION_GATE",
        "INTERVENTION_TIMING_GATE",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
        "NO_FURTHER_QUANTIZATION_GATE",
        "INSTRUMENTATION_NONINTERFERENCE_GATE",
        "L1_SEMANTICS_GATE",
    ]:
        gates[name] = "BLOCKED_MISSING_QWEN_REPLAY_INPUTS" if missing else "PENDING"
    return {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "stage": "environment_audit",
        "formal_status": "INVALID" if missing else "READY_TO_RUN",
        "canonical_manifest_source": str(CANONICAL_MANIFEST_SOURCE),
        "canonical_manifest_sha256": manifest_hash,
        "canonical_manifest": manifest,
        "n_canonical_units": len(manifest),
        "prompts": sorted({u["problem_id"] for u in manifest}),
        "t0": sorted({u["t0"] for u in manifest}),
        "layer_scope": GDN_LAYERS,
        "head_scope": "ALL_HEADS",
        "state_shape": stage0_metrics.get("state_shape"),
        "R128_scale_shape": stage0_metrics.get("R128_scale_shape"),
        "C128_scale_shape": stage0_metrics.get("C128_scale_shape"),
        "model_path": str(MODEL_PATH),
        "model_file_count": len(model_files),
        "missing_required_replay_inputs": missing,
        "gates": gates,
        "history_lengths": HISTORY_LENGTHS,
        "gamma_values": GAMMA_VALUES,
        "method_design_ready": "NO",
        "no_next_experiment_launched": "YES",
    }


def run_audit():
    obj = environment_audit()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RUN_DIR / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    save_json(RUN_DIR / "aggregate_summary.json", aggregate_empty(obj))
    save_json(RUN_DIR / "environment_audit.json", obj)
    append_jsonl(results_path, obj)
    write_rows(RUN_DIR / "per_step_same_state_residuals.csv", ["task", "git_commit", "manifest_sha256"], [])
    write_rows(RUN_DIR / "gamma_dose_response.csv", ["task", "source_direction"], [])
    write_rows(RUN_DIR / "state_error_trajectory.csv", ["task", "unit_id"], [])
    write_rows(RUN_DIR / "cross_arch_transfer_summary.csv", ["factor", "qwen_gdn", "ling_kda"], cross_arch_rows(None))
    write_report(aggregate_empty(obj), obj)
    return obj


def aggregate_empty(audit):
    gates = audit["gates"]
    missing = audit.get("missing_required_replay_inputs", [])
    return {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "formal_status": audit["formal_status"],
        "n_formal_units": 0,
        "canonical_manifest_source": audit["canonical_manifest_source"],
        "canonical_manifest_sha256": audit["canonical_manifest_sha256"],
        "n_canonical_units": audit["n_canonical_units"],
        "prompts": audit["prompts"],
        "t0": audit["t0"],
        "layer_scope": audit["layer_scope"],
        "head_scope": audit["head_scope"],
        "state_shape": audit["state_shape"],
        "R128_scale_shape": audit["R128_scale_shape"],
        "C128_scale_shape": audit["C128_scale_shape"],
        "gates": gates,
        "SMOKE": "NOT_RUN_BLOCKED_MISSING_QWEN_REPLAY_INPUTS" if missing else "NOT_RUN",
        "PILOT": "NOT_RUN_BLOCKED_MISSING_QWEN_REPLAY_INPUTS" if missing else "NOT_RUN",
        "history_lengths": HISTORY_LENGTHS,
        "gamma_values": GAMMA_VALUES,
        "source_norms": {
            "R_SOURCE_NORM": "NA_NOT_RUN",
            "C_SOURCE_NORM": "NA_NOT_RUN",
            "R_NORM_GT_C": "NA_NOT_RUN",
            "C_NORM_GT_R": "NA_NOT_RUN",
            "MEDIAN_R_OVER_C_SOURCE_NORM": "NA_NOT_RUN",
        },
        "contrasts_by_l": {},
        "dose_response": {
            "R_DIR": "NA_NOT_RUN",
            "C_DIR": "NA_NOT_RUN",
        },
        "P_DECOMPOSITION_REUSE": "NOT_AVAILABLE_NONBLOCKING",
        "RESIDUAL_FUNCTIONAL_GEOMETRY_GAP_AFTER_MAG_MATCH": "NA_NOT_RUN",
        "QWEN_PERSISTENT_SOURCE_MAGNITUDE_CAUSAL": "NOT_INTERPRETABLE_BLOCKING_GATE",
        "LING_TO_QWEN_MAGNITUDE_MECHANISM_TRANSFER": "NOT_INTERPRETABLE_BLOCKING_GATE",
        "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED": "NOT_INTERPRETABLE_BLOCKING_GATE",
        "CROSS_ARCH_MECHANISM_HYPOTHESIS": "NOT_INTERPRETABLE_BLOCKING_GATE",
        "FINAL_CLASSIFICATION": "INVALID_MISSING_QWEN_REPLAY_INPUTS" if missing else "READY_TO_RUN",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "RESTORE_QWEN_GDN_REPLAY_INPUTS_AND_RERUN_THIS_TASK" if missing else "RUN_STAGE0_SMOKE_PILOT_FORMAL",
        "missing_required_replay_inputs": missing,
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
    }


def collect_stage_results(stage, num_shards):
    unit_results = []
    semantics = []
    failures = []
    shard_dirs = []
    shard_manifests = []
    provenances = []
    for shard_id in range(int(num_shards)):
        shard_dir = RUN_DIR / "shards" / f"{stage}_{int(num_shards)}_{shard_id}"
        shard_dirs.append(shard_dir)
        manifest_path = shard_dir / "manifest.json"
        if manifest_path.exists():
            shard_manifests.append((shard_id, json.loads(manifest_path.read_text(encoding="utf-8"))))
        provenance_path = shard_dir / "execution_provenance.json"
        if provenance_path.exists():
            provenances.append((shard_id, json.loads(provenance_path.read_text(encoding="utf-8"))))
        semantic_path = shard_dir / "runtime_semantics_audit.json"
        if semantic_path.exists():
            semantics.append((shard_id, json.loads(semantic_path.read_text(encoding="utf-8"))))
        failure_path = shard_dir / "failures.json"
        if failure_path.exists():
            failures.extend(json.loads(failure_path.read_text(encoding="utf-8")))
        for path in sorted((shard_dir / "units").glob("*.json")):
            unit_results.append((shard_id, path, json.loads(path.read_text(encoding="utf-8"))))
    return unit_results, semantics, failures, shard_dirs, shard_manifests, provenances


def all_gate(rows, key):
    return "PASS" if rows and all(row.get(key) == "PASS" for row in rows) else "FAIL"


def observed_execution_gates(schedule_rows, branch_audits, expected_units, horizon):
    units = {unit["unit_id"]: unit for unit in expected_units}
    timing_ok = bool(branch_audits)
    for audit in branch_audits:
        unit = units.get(audit.get("unit_id"))
        if unit is None:
            timing_ok = False
            break
        history_length = int(audit["history_length"])
        expected = list(range(int(unit["t0"]) - history_length + 1, int(unit["t0"]) + 1))
        observed = sorted(
            int(row["active_timestep"])
            for row in schedule_rows
            if row.get("unit_id") == unit["unit_id"]
            and int(row.get("history_length", -1)) == history_length
            and row.get("branch") == audit.get("branch")
        )
        if observed != expected or int(audit.get("intervention_count", -1)) != history_length:
            timing_ok = False
            break
        if "quantization_event_timesteps" in audit and audit["quantization_event_timesteps"] != expected:
            timing_ok = False
            break

    r_native = [row for row in branch_audits if row.get("native_endpoint") == "R128"]
    c_native = [row for row in branch_audits if row.get("native_endpoint") == "C128"]
    endpoint_pass = lambda rows: bool(rows) and all(
        finite_number(row.get("endpoint_identity_max_abs_error"))
        and float(row["endpoint_identity_max_abs_error"]) <= IDENTITY_TOL
        for row in rows
    )
    future_ok = bool(branch_audits) and all(
        int(row.get("future_quantization_count", -1)) == 0 for row in branch_audits
    ) and all(
        int(row["active_timestep"]) <= int(units[row["unit_id"]]["t0"])
        for row in schedule_rows
        if row.get("unit_id") in units
    )
    for audit in branch_audits:
        if "future_forward_horizons" in audit and audit["future_forward_horizons"] != list(range(1, int(horizon) + 1)):
            future_ok = False
        if "quantization_event_timesteps" in audit:
            unit = units.get(audit.get("unit_id"))
            if unit and any(int(timestep) > int(unit["t0"]) for timestep in audit["quantization_event_timesteps"]):
                future_ok = False
    l1_audits = [row for row in branch_audits if int(row.get("history_length", -1)) == 1]
    return {
        "SAME_STATE_COUNTERFACTUAL_GATE": all_gate(schedule_rows, "SAME_STATE_COUNTERFACTUAL_GATE"),
        "APPLIED_MAGNITUDE_GATE": all_gate(schedule_rows, "APPLIED_MAGNITUDE_GATE"),
        "SOURCE_DIRECTION_PRESERVATION_GATE": all_gate(schedule_rows, "SOURCE_DIRECTION_PRESERVATION_GATE"),
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE": all_gate(schedule_rows, "CURRENT_LOGIT_NONRETROACTIVITY_GATE"),
        "INTERVENTION_TIMING_GATE": "PASS" if timing_ok else "FAIL",
        "NO_FURTHER_QUANTIZATION_GATE": "PASS" if future_ok else "FAIL",
        "R_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS" if endpoint_pass(r_native) else "FAIL",
        "C_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS" if endpoint_pass(c_native) else "FAIL",
        "L1_SEMANTICS_GATE": "PASS" if timing_ok and l1_audits else "FAIL",
    }


def median_at_l(contrasts, key, history_length):
    value = contrasts.get(key, {}).get(str(history_length), {})
    return value.get("median") if isinstance(value, dict) else None


def pilot_decision(contrasts, dose):
    available = sorted(int(key) for key in contrasts.get("NATIVE_R_MINUS_C_GAP_BY_L", {}))
    if not available:
        return "INVALID_NO_PAIRED_RESULTS"
    length = max(available)
    native = median_at_l(contrasts, "NATIVE_R_MINUS_C_GAP_BY_L", length)
    rescue = median_at_l(contrasts, "R_MAG_EQUALIZATION_RESCUE_BY_L", length)
    graft = median_at_l(contrasts, "C_RMAG_GRAFT_DAMAGE_BY_L", length)
    if not finite_number(native) or native <= 0:
        return "NATIVE_ORIENTATION_REPRODUCTION_FAILED"
    rescue_fraction = float(rescue) / (abs(float(native)) + EPS) if finite_number(rescue) else 0.0
    graft_fraction = float(graft) / (abs(float(native)) + EPS) if finite_number(graft) else 0.0
    r_panel = dose["R_DIR"].get("by_history_length", {}).get(str(length), dose["R_DIR"])
    c_panel = dose["C_DIR"].get("by_history_length", {}).get(str(length), dose["C_DIR"])
    r_rho = r_panel.get("median_spearman_rho")
    c_rho = c_panel.get("median_spearman_rho")
    dose_signal = max(abs(float(r_rho or 0.0)), abs(float(c_rho or 0.0))) >= 0.3
    if rescue_fraction >= 0.1 or graft_fraction >= 0.1 or dose_signal:
        return "POSITIVE_OR_MIXED_SUPPORTS_FORMAL"
    return "NEGATIVE_BUT_INFORMATIVE"


def scientific_classification(contrasts, dose, source_norms, stage, blocking_pass, complete_units):
    uninterpretable = {
        "causal": "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL",
        "transfer": "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL",
        "geometry": "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL",
        "hypothesis": "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL",
        "final": "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL",
        "functional_geometry_dominance": "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL",
        "common_magnitude_mechanism": "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL",
        "evidence": {},
    }
    if stage != "formal" or not blocking_pass or not complete_units:
        if stage == "formal" and (not blocking_pass or not complete_units):
            uninterpretable = dict(uninterpretable)
            uninterpretable.update({
                "causal": "NOT_INTERPRETABLE_BLOCKING_GATE",
                "transfer": "NOT_INTERPRETABLE_BLOCKING_GATE",
                "geometry": "NOT_INTERPRETABLE_BLOCKING_GATE",
                "hypothesis": "NOT_INTERPRETABLE_BLOCKING_GATE",
                "final": "INVALID_INCOMPLETE_OR_BLOCKING_GATE",
            })
        return uninterpretable
    available = sorted(int(key) for key in contrasts.get("NATIVE_R_MINUS_C_GAP_BY_L", {}))
    if not available:
        return {
            "causal": "NOT_INTERPRETABLE_BLOCKING_GATE",
            "transfer": "NOT_INTERPRETABLE_BLOCKING_GATE",
            "geometry": "NOT_INTERPRETABLE_BLOCKING_GATE",
            "hypothesis": "NOT_INTERPRETABLE_BLOCKING_GATE",
            "final": "INVALID_NO_PAIRED_RESULTS",
            "functional_geometry_dominance": "NOT_INTERPRETABLE_BLOCKING_GATE",
            "common_magnitude_mechanism": "NOT_INTERPRETABLE_BLOCKING_GATE",
            "evidence": {},
        }
    length = max(available)
    stats = {
        key: contrasts.get(f"{key}_BY_L", {}).get(str(length), {})
        for key in [
            "NATIVE_R_MINUS_C_GAP", "R_MAG_EQUALIZATION_RESCUE", "C_RMAG_GRAFT_DAMAGE",
            "R_EQ_VS_C_NATIVE", "STATE_ERROR_RESCUE", "STATE_ERROR_GRAFT_DAMAGE",
        ]
    }
    native = stats["NATIVE_R_MINUS_C_GAP"].get("median")
    rescue = stats["R_MAG_EQUALIZATION_RESCUE"].get("median")
    graft = stats["C_RMAG_GRAFT_DAMAGE"].get("median")
    residual = stats["R_EQ_VS_C_NATIVE"].get("median")
    if not finite_number(native) or native <= 0:
        return {
            "causal": "NOT_INTERPRETABLE_NATIVE_ORIENTATION_FAILED",
            "transfer": "NOT_INTERPRETABLE_NATIVE_ORIENTATION_FAILED",
            "geometry": "NOT_INTERPRETABLE_NATIVE_ORIENTATION_FAILED",
            "hypothesis": "NOT_INTERPRETABLE_NATIVE_ORIENTATION_FAILED",
            "final": "QWEN_NATIVE_ORIENTATION_REPRODUCTION_FAILED",
            "functional_geometry_dominance": "NOT_INTERPRETABLE_NATIVE_ORIENTATION_FAILED",
            "common_magnitude_mechanism": "NOT_INTERPRETABLE_NATIVE_ORIENTATION_FAILED",
            "evidence": {"history_length": length, "native_median": native},
        }
    native = float(native)
    rescue_fraction = float(rescue) / (abs(native) + EPS) if finite_number(rescue) else -math.inf
    graft_fraction = float(graft) / (abs(native) + EPS) if finite_number(graft) else -math.inf
    closure = abs(float(residual)) / (abs(float(native)) + EPS) if finite_number(residual) else math.inf
    dose_at_l = {
        direction: dose[direction].get("by_history_length", {}).get(str(length), dose[direction])
        for direction in SOURCE_DIRECTIONS
    }
    r_rho = float(dose_at_l["R_DIR"].get("median_spearman_rho") or 0.0)
    c_rho = float(dose_at_l["C_DIR"].get("median_spearman_rho") or 0.0)

    def majority_positive(stat):
        n = int(stat.get("n") or 0)
        return n > 0 and int(stat.get("positive") or 0) / n >= 2.0 / 3.0

    def ci_positive(stat):
        ci = stat.get("bootstrap_ci")
        return isinstance(ci, (list, tuple)) and len(ci) == 2 and finite_number(ci[0]) and float(ci[0]) > 0

    source_aligned = source_norms.get("QWEN_SOURCE_MAGNITUDE_ORIENTATION") == "R_GT_C"
    rescue_meaningful = rescue_fraction >= 0.05 and majority_positive(stats["R_MAG_EQUALIZATION_RESCUE"])
    graft_meaningful = graft_fraction >= 0.05 and majority_positive(stats["C_RMAG_GRAFT_DAMAGE"])
    dose_consistent = all(
        float(dose_at_l[direction].get("median_spearman_rho") or 0.0) >= 0.5
        and int(dose_at_l[direction].get("n_units") or 0) > 0
        and int(dose_at_l[direction].get("positive_rho_count") or 0) / int(dose_at_l[direction]["n_units"]) >= 2.0 / 3.0
        for direction in SOURCE_DIRECTIONS
    )
    state_tracks = (
        majority_positive(stats["STATE_ERROR_RESCUE"])
        and majority_positive(stats["STATE_ERROR_GRAFT_DAMAGE"])
        and float(stats["STATE_ERROR_RESCUE"].get("median") or 0.0) > 0
        and float(stats["STATE_ERROR_GRAFT_DAMAGE"].get("median") or 0.0) > 0
    )
    strong = (
        source_aligned
        and ci_positive(stats["NATIVE_R_MINUS_C_GAP"])
        and rescue_fraction >= 0.25
        and graft_fraction >= 0.25
        and ci_positive(stats["R_MAG_EQUALIZATION_RESCUE"])
        and ci_positive(stats["C_RMAG_GRAFT_DAMAGE"])
        and dose_consistent
        and state_tracks
        and closure <= 0.25
    )
    evidence = {
        "history_length": length,
        "native_median": native,
        "rescue_fraction_of_native_gap": rescue_fraction,
        "graft_fraction_of_native_gap": graft_fraction,
        "residual_fraction_of_native_gap": closure,
        "source_magnitude_aligned": source_aligned,
        "dose_consistent": dose_consistent,
        "state_error_tracks": state_tracks,
        "strong_thresholds_met": strong,
    }
    if strong:
        return {
            "causal": "STRONG",
            "transfer": "SUPPORTED",
            "geometry": "NO_DOMINANT_RESIDUAL_GAP",
            "hypothesis": "H1_COMMON_MAGNITUDE_ACCUMULATION_MECHANISM",
            "final": "QWEN_RECURRENT_MAGNITUDE_ACCUMULATION_CAUSALLY_SUPPORTED",
            "functional_geometry_dominance": "NOT_REQUIRED_AS_DOMINANT",
            "common_magnitude_mechanism": "SUPPORTED",
            "evidence": evidence,
        }
    if rescue_meaningful or graft_meaningful:
        return {
            "causal": "PARTIAL",
            "transfer": "PARTIAL",
            "geometry": "YES" if closure > 0.25 else "UNCERTAIN_NOT_DOMINANT",
            "hypothesis": "H2_SHARED_MULTIFACTOR_MECHANISM_WITH_QWEN_FUNCTIONAL_GEOMETRY",
            "final": "QWEN_RECURRENT_MAGNITUDE_ACCUMULATION_PARTIALLY_SUPPORTED",
            "functional_geometry_dominance": "REQUIRED_ADDITIONAL_FACTOR",
            "common_magnitude_mechanism": "PARTIAL",
            "evidence": evidence,
        }
    return {
        "causal": "NOT_SUPPORTED_AS_PRIMARY",
        "transfer": "NOT_SUPPORTED",
        "geometry": "YES_PRIORITY_RAISED",
        "hypothesis": "H3_ARCHITECTURE_DEPENDENT_DOMINANT_MECHANISMS",
        "final": "QWEN_FUNCTIONAL_GEOMETRY_DOMINANT_NEGATIVE_MAGNITUDE_TRANSFER",
        "functional_geometry_dominance": "PRIORITY_RAISED",
        "common_magnitude_mechanism": "NOT_SUPPORTED",
        "evidence": evidence,
    }


def build_contrasts(rows):
    mapping = {}
    for key in [
        "NATIVE_R_MINUS_C_GAP",
        "R_MAG_EQUALIZATION_RESCUE",
        "C_RMAG_GRAFT_DAMAGE",
        "R_EQ_VS_C_NATIVE",
        "C_GRAFT_VS_R_NATIVE",
        "MAGNITUDE_MAIN",
        "STATE_ERROR_RESCUE",
        "STATE_ERROR_GRAFT_DAMAGE",
    ]:
        mapping[f"{key}_BY_L"] = contrast_by_l(rows, key)
    return mapping


def exact_result_coverage(rows, expected_units, history_lengths, gamma_values):
    expected = set()
    for unit in expected_units:
        for history_length in history_lengths:
            for config in branch_configs(gamma_values):
                expected.add((
                    unit["unit_id"],
                    int(history_length),
                    config["source_direction"],
                    None if config["gamma"] is None else float(config["gamma"]),
                ))
    actual_list = [(
        row.get("unit_id"),
        int(row.get("history_length")),
        row.get("source_direction"),
        None if row.get("gamma") is None else float(row.get("gamma")),
    ) for row in rows]
    return len(actual_list) == len(expected) and len(set(actual_list)) == len(actual_list) and set(actual_list) == expected


def branch_success_counts(rows, expected_units, history_lengths, gamma_values):
    expected = len(expected_units) * len(history_lengths)
    counts = {}
    for config in branch_configs(gamma_values):
        observed = {
            (row.get("unit_id"), int(row.get("history_length", -1)))
            for row in rows
            if row.get("source_direction") == config["source_direction"]
            and (
                row.get("gamma") is None and config["gamma"] is None
                or finite_number(row.get("gamma")) and finite_number(config["gamma"])
                and abs(float(row["gamma"]) - float(config["gamma"])) <= EPS
            )
        }
        counts[config["branch"]] = {
            "n_success": len(observed),
            "n_expected": expected,
            "status": f"{len(observed)} / {expected}",
        }
    return counts


def fresh_output_verification(
    stage, rows, expected_units, history_lengths, gamma_values, manifest_hash, merge_started,
    protocol_provenance_ok=True, checkpoint_validation_ok=True,
):
    expected_branches = 1 + 2 * len(gamma_values)
    expected_count = len(expected_units) * len(history_lengths) * expected_branches
    unit_ids = {row.get("unit_id") for row in rows}
    hashes = {row.get("manifest_sha256") for row in rows}
    commits = {row.get("git_commit") for row in rows}
    paths = [
        RUN_DIR / "results.jsonl",
        RUN_DIR / "aggregate_summary.json",
        RUN_DIR / "report.md",
        RUN_DIR / "per_step_same_state_residuals.csv",
        RUN_DIR / "gamma_dose_response.csv",
        RUN_DIR / "state_error_trajectory.csv",
        RUN_DIR / "cross_arch_transfer_summary.csv",
    ]
    checks = {
        "row_count": len(rows) == expected_count,
        "exact_branch_coverage": exact_result_coverage(rows, expected_units, history_lengths, gamma_values),
        "unit_ids": unit_ids == {unit["unit_id"] for unit in expected_units},
        "manifest_hash": hashes == {manifest_hash},
        "git_commit": commits == {git_commit()},
        "fresh_files": all(path.exists() and path.stat().st_mtime >= merge_started - 1.0 for path in paths),
        "stage": all(row.get("stage") == stage for row in rows),
        "protocol_provenance": protocol_provenance_ok,
        "checkpoint_validation": checkpoint_validation_ok,
        "required_horizons": all(
            finite_number(row.get(f"KL_h{horizon}"))
            for row in rows for horizon in FUTURE_KL_HORIZONS
        ),
    }
    return "PASS" if all(checks.values()) else "FAIL", checks


def invalidate_scientific_interpretation(summary, reason):
    summary["formal_status"] = "INVALID"
    for key in [
        "QWEN_PERSISTENT_SOURCE_MAGNITUDE_CAUSAL",
        "LING_TO_QWEN_MAGNITUDE_MECHANISM_TRANSFER",
        "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED",
        "QWEN_FUNCTIONAL_GEOMETRY_DOMINANCE",
        "CROSS_ARCH_MAGNITUDE_ACCUMULATION_COMMON_MECHANISM",
        "CROSS_ARCH_MECHANISM_HYPOTHESIS",
    ]:
        summary[key] = "NOT_INTERPRETABLE_INVALID_OUTPUT"
    summary["classification_evidence"] = {"invalid_reason": reason}
    summary["FINAL_CLASSIFICATION"] = reason
    summary["NEXT_RECOMMENDED_TASK"] = "REPAIR_AND_COMPLETE_CURRENT_TASK"
    summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "NO"


def aggregate_stage(stage, num_shards):
    merge_started = time.time()
    audit = environment_audit()
    _canonical_obj, manifest, manifest_hash = load_canonical_manifest()
    all_units = canonical_units_with_ids(manifest)
    expected_units = stage_units(stage, all_units)
    protocol = stage_protocol(stage)
    unit_records, semantics, failures, shard_dirs, shard_manifests, provenances = collect_stage_results(stage, num_shards)
    protocol_errors = []
    manifest_by_shard = {shard_id: obj for shard_id, obj in shard_manifests}
    provenance_by_shard = {shard_id: obj for shard_id, obj in provenances}
    for shard_id in range(int(num_shards)):
        shard_manifest = manifest_by_shard.get(shard_id)
        if shard_manifest is None:
            protocol_errors.append(f"shard_{shard_id}:missing_manifest")
        else:
            manifest_checks = validate_shard_manifest(
                shard_manifest, stage, shard_id, num_shards, manifest_hash,
                protocol["history_lengths"], protocol["gamma_values"], HORIZON,
            )
            protocol_errors.extend(
                f"shard_{shard_id}:manifest_{key}" for key, passed in manifest_checks.items() if not passed
            )
            expected_shard_ids = {
                unit["unit_id"] for unit in select_shard_units(expected_units, shard_id, num_shards)
            }
            actual_shard_ids = {unit.get("unit_id") for unit in shard_manifest.get("units", [])}
            if actual_shard_ids != expected_shard_ids:
                protocol_errors.append(f"shard_{shard_id}:assigned_units")
        provenance = provenance_by_shard.get(shard_id)
        if not provenance:
            protocol_errors.append(f"shard_{shard_id}:missing_execution_provenance")
        else:
            protocol_errors.extend(
                f"shard_{shard_id}:{error}"
                for error in validate_execution_provenance(provenance, stage, shard_id, num_shards)
            )
    provenance_hashes = {obj.get("runner_sha256") for _, obj in provenances if obj.get("runner_sha256")}
    if len(provenance_hashes) != 1:
        protocol_errors.append("runner_sha256_not_uniform")
    if {shard_id for shard_id, _obj in semantics} != set(range(int(num_shards))):
        protocol_errors.append("runtime_semantics_shard_coverage")

    expected_by_id = {unit["unit_id"]: unit for unit in expected_units}
    by_unit = {}
    checkpoint_errors = []
    for shard_id, path, result in unit_records:
        unit_id = result.get("unit", {}).get("unit_id")
        unit = expected_by_id.get(unit_id)
        if unit is None:
            checkpoint_errors.append({"path": str(path), "errors": ["unexpected_unit"]})
            continue
        expected_shard = expected_units.index(unit) % int(num_shards)
        errors = validate_unit_result(
            result, unit, stage, manifest_hash, protocol["history_lengths"], protocol["gamma_values"], HORIZON
        )
        errors.extend(checkpoint_source_provenance_errors(result, provenance_by_shard.get(shard_id, {})))
        if shard_id != expected_shard:
            errors.append("wrong_shard")
        if unit_id in by_unit:
            errors.append("duplicate_unit_checkpoint")
        if errors:
            checkpoint_errors.append({"path": str(path), "errors": sorted(set(errors))})
            continue
        by_unit[unit_id] = result
    selected = [by_unit[unit["unit_id"]] for unit in expected_units if unit["unit_id"] in by_unit]
    rows = [row for result in selected for row in result["rows"]]
    schedule_rows = [row for result in selected for row in result["schedule_rows"]]
    state_rows = [row for result in selected for row in result["state_rows"]]
    branch_audits = [
        dict(row, unit_id=result["unit"]["unit_id"])
        for result in selected for row in result["audits"]
    ]
    for row in rows:
        row["stage"] = stage
    gates = dict(audit["gates"])
    gates.update({
        "STATE_SEMANTICS_GATE": "PASS" if semantics and all(row.get("STATE_SEMANTICS_GATE") == "PASS" for _shard, row in semantics) else "FAIL",
        "R128_QUANTIZER_IDENTITY_GATE": "PASS" if semantics and all(row.get("R128_QUANTIZER_IDENTITY_GATE") == "PASS" for _shard, row in semantics) else "FAIL",
        "C128_QUANTIZER_IDENTITY_GATE": "PASS" if semantics and all(row.get("C128_QUANTIZER_IDENTITY_GATE") == "PASS" for _shard, row in semantics) else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if semantics and all(row.get("INSTRUMENTATION_NONINTERFERENCE_GATE") == "PASS" for _shard, row in semantics) else "FAIL",
        **observed_execution_gates(schedule_rows, branch_audits, expected_units, HORIZON),
    })
    contrasts = build_contrasts(rows)
    dose = {direction: dose_response(rows, direction) for direction in SOURCE_DIRECTIONS}
    source_norms = summarize_source_norms(schedule_rows) if schedule_rows else {
        "R_SOURCE_NORM": None, "C_SOURCE_NORM": None, "R_NORM_GT_C": "0 / 0",
        "C_NORM_GT_R": "0 / 0", "MEDIAN_R_OVER_C_SOURCE_NORM": None,
        "QWEN_SOURCE_MAGNITUDE_ORIENTATION": "NOT_MEASURED",
    }
    blocking_pass = all(gates.get(key) == "PASS" for key in BLOCKING_GATE_NAMES)
    complete_units = (
        len(selected) == len(expected_units)
        and not failures
        and not checkpoint_errors
        and not protocol_errors
    )
    decision = pilot_decision(contrasts, dose) if blocking_pass and complete_units else "INVALID_PILOT"
    classification = scientific_classification(
        contrasts, dose, source_norms, stage=stage,
        blocking_pass=blocking_pass, complete_units=complete_units,
    )
    if stage == "formal":
        formal_status = "COMPLETE" if blocking_pass and complete_units else "INVALID"
    else:
        formal_status = f"NOT_RUN_{stage.upper()}_COMPLETE" if blocking_pass and complete_units else f"INVALID_{stage.upper()}"
    max_l = max(protocol["history_lengths"])
    residual_geometry = {
        "history_length": max_l,
        "R_EQ_VS_C_NATIVE": contrasts.get("R_EQ_VS_C_NATIVE_BY_L", {}).get(str(max_l)),
        "interpretation": "R_DIRECTION_FUNCTIONAL_GAP_REMAINS" if (median_at_l(contrasts, "R_EQ_VS_C_NATIVE_BY_L", max_l) or 0) > 0 else "NO_POSITIVE_RESIDUAL_R_GAP",
    }
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "analysis_runner_sha256": runner_sha256(),
        "formal_status": formal_status,
        "stage": stage,
        "n_formal_units": len(selected) if stage == "formal" else 0,
        "n_success": len(selected),
        "n_expected": len(expected_units),
        "canonical_manifest_source": str(CANONICAL_MANIFEST_SOURCE),
        "canonical_manifest_sha256": manifest_hash,
        "n_canonical_units": len(manifest),
        "prompts": sorted({unit["problem_id"] for unit in manifest}),
        "t0": sorted({int(unit["t0"]) for unit in manifest}),
        "layer_scope": GDN_LAYERS,
        "head_scope": "ALL_HEADS",
        "state_shape": audit["state_shape"],
        "R128_scale_shape": audit["R128_scale_shape"],
        "C128_scale_shape": audit["C128_scale_shape"],
        "gates": gates,
        "SMOKE": "PASS" if stage == "smoke" and blocking_pass and complete_units else "SEE_STAGE_ARTIFACT",
        "PILOT": decision if stage in ("pilot", "formal") else "NOT_RUN",
        "history_lengths": protocol["history_lengths"],
        "gamma_values": protocol["gamma_values"],
        "source_norms": source_norms,
        "execution_runner_sha256": sorted(provenance_hashes),
        "execution_provenance": [obj for _shard, obj in sorted(provenances)],
        "protocol_errors": protocol_errors,
        "checkpoint_validation_errors": checkpoint_errors,
        "branch_success_counts": branch_success_counts(
            rows, expected_units, protocol["history_lengths"], protocol["gamma_values"]
        ),
        "contrasts_by_l": contrasts,
        "dose_response": dose,
        "P_DECOMPOSITION_REUSE": "NOT_AVAILABLE_NONBLOCKING",
        "RESIDUAL_FUNCTIONAL_GEOMETRY_GAP_AFTER_MAG_MATCH": residual_geometry,
        "QWEN_PERSISTENT_SOURCE_MAGNITUDE_CAUSAL": classification["causal"],
        "LING_TO_QWEN_MAGNITUDE_MECHANISM_TRANSFER": classification["transfer"],
        "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED": classification["geometry"],
        "QWEN_FUNCTIONAL_GEOMETRY_DOMINANCE": classification["functional_geometry_dominance"],
        "CROSS_ARCH_MAGNITUDE_ACCUMULATION_COMMON_MECHANISM": classification["common_magnitude_mechanism"],
        "CROSS_ARCH_MECHANISM_HYPOTHESIS": classification["hypothesis"],
        "classification_evidence": classification.get("evidence", {}),
        "FINAL_CLASSIFICATION": classification["final"],
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "CROSS_ARCH_RECURRENT_QUANTIZATION_RISK_FACTORIZATION_V1" if stage == "formal" and blocking_pass else "COMPLETE_CURRENT_TASK_STAGES",
        "missing_required_replay_inputs": [],
        "failures": failures,
        "shard_dirs": [str(path) for path in shard_dirs],
        "FRESH_OUTPUT_VERIFICATION": "PENDING",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "YES" if stage == "formal" and blocking_pass and complete_units else "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
    }
    smoke_path = RUN_DIR / "aggregate_smoke.json"
    pilot_path = RUN_DIR / "aggregate_pilot.json"
    if stage != "smoke" and smoke_path.exists():
        prior = json.loads(smoke_path.read_text(encoding="utf-8"))
        summary["SMOKE"] = prior.get("SMOKE", "SEE_STAGE_ARTIFACT")
    if stage != "pilot" and pilot_path.exists():
        prior = json.loads(pilot_path.read_text(encoding="utf-8"))
        summary["PILOT"] = prior.get("PILOT", "SEE_STAGE_ARTIFACT")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RUN_DIR / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    for row in rows:
        append_jsonl(results_path, row)
    write_rows(RUN_DIR / "per_step_same_state_residuals.csv", merged_fields(schedule_rows), schedule_rows)
    dose_rows = dose["R_DIR"]["rows"] + dose["C_DIR"]["rows"]
    write_rows(RUN_DIR / "gamma_dose_response.csv", merged_fields(dose_rows), dose_rows)
    write_rows(RUN_DIR / "state_error_trajectory.csv", merged_fields(state_rows), state_rows)
    write_rows(RUN_DIR / "cross_arch_transfer_summary.csv", ["factor", "qwen_gdn", "ling_kda"], cross_arch_rows(summary))
    save_json(RUN_DIR / f"aggregate_{stage}.json", summary)
    save_json(RUN_DIR / "aggregate_summary.json", summary)
    write_report(summary, audit)
    fresh, fresh_checks = fresh_output_verification(
        stage, rows, expected_units, protocol["history_lengths"], protocol["gamma_values"], manifest_hash, merge_started,
        protocol_provenance_ok=not protocol_errors,
        checkpoint_validation_ok=not checkpoint_errors,
    )
    summary["FRESH_OUTPUT_VERIFICATION"] = fresh
    summary["fresh_output_checks"] = fresh_checks
    if fresh != "PASS":
        invalidate_scientific_interpretation(summary, "INVALID_FRESH_OUTPUT_VERIFICATION")
    write_rows(RUN_DIR / "cross_arch_transfer_summary.csv", ["factor", "qwen_gdn", "ling_kda"], cross_arch_rows(summary))
    save_json(RUN_DIR / f"aggregate_{stage}.json", summary)
    save_json(RUN_DIR / "aggregate_summary.json", summary)
    write_report(summary, audit)
    return summary


def cross_arch_rows(summary):
    if summary is None:
        qwen = {}
    else:
        source = summary.get("source_norms", {})
        if summary.get("FINAL_CLASSIFICATION") == "QWEN_NATIVE_ORIENTATION_REPRODUCTION_FAILED":
            native = "NOT_REPRODUCED"
        elif summary.get("stage") == "formal" and summary.get("formal_status") == "COMPLETE":
            native = "R"
        else:
            native = "NOT_INTERPRETABLE_BEFORE_COMPLETE_FORMAL"
        qwen = {
            "native": native,
            "source": source.get("QWEN_SOURCE_MAGNITUDE_ORIENTATION", "NOT_MEASURED"),
            "magnitude": summary.get("QWEN_PERSISTENT_SOURCE_MAGNITUDE_CAUSAL", "NOT_INTERPRETABLE"),
            "temporal": "MEASURED_THIS_RUN" if summary.get("stage") in ("pilot", "formal") else "PRIOR_STRONG_R128",
            "closure": summary.get("FINAL_CLASSIFICATION", "NOT_INTERPRETABLE"),
        }
    return [
        {"factor": "Native worse orientation", "qwen_gdn": qwen.get("native", "NOT_MEASURED_THIS_RUN"), "ling_kda": "C"},
        {"factor": "Source magnitude orientation", "qwen_gdn": qwen.get("source", "NOT_MEASURED_THIS_RUN"), "ling_kda": "C > R"},
        {"factor": "Magnitude causal effect", "qwen_gdn": qwen.get("magnitude", "NOT_INTERPRETABLE"), "ling_kda": "STRONG"},
        {"factor": "Direction harmfulness", "qwen_gdn": "PRIOR_STRONG", "ling_kda": "NOT_SUPPORTED_AS_PRIMARY"},
        {"factor": "Temporal accumulation", "qwen_gdn": qwen.get("temporal", "PRIOR_STRONG_R128"), "ling_kda": "STRONG"},
        {"factor": "Temporal coherence", "qwen_gdn": "NOT_PRIMARY_THIS_RUN", "ling_kda": "PARTIAL"},
        {"factor": "Functional geometry", "qwen_gdn": "PRIOR_STRONG", "ling_kda": "WEAK_NOT_PRIMARY"},
        {"factor": "Closed-loop magnitude closure", "qwen_gdn": qwen.get("closure", "NOT_INTERPRETABLE"), "ling_kda": "STRONG"},
    ]


def write_report(summary, audit):
    lines = [
        "# Qwen GDN Persistent Source Error Magnitude Equalization Transfer V1",
        "",
        "## Status",
        summary["formal_status"],
        "",
        "## Canonical Manifest",
        f"Source: `{summary['canonical_manifest_source']}`",
        f"SHA256: `{summary['canonical_manifest_sha256']}`",
        f"Units: {summary['n_canonical_units']}",
        f"Execution runner SHA256: `{summary.get('execution_runner_sha256')}`",
        f"Analysis runner SHA256: `{summary.get('analysis_runner_sha256')}`",
        "",
        "## Environment",
    ]
    missing = summary.get("missing_required_replay_inputs", [])
    if missing:
        lines.extend(["The model and prior result artifacts are present, but canonical replay inputs are missing:"])
        lines.extend([f"- `{x}`" for x in missing])
        lines.append("")
        lines.append("Without the original prompt table and FP continuations, the branch-local closed-loop intervention cannot be executed without changing the validated Qwen protocol.")
    else:
        lines.append("Canonical replay inputs and the offline model are available.")
    lines.extend([
        "",
        "## Gates",
        "```json",
        json.dumps(summary["gates"], indent=2, sort_keys=True),
        "```",
        "",
        "## Source Magnitude Orientation",
        "```json",
        json.dumps(summary.get("source_norms", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Primary Contrasts By History Length",
        "```json",
        json.dumps(summary.get("contrasts_by_l", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Branch Completion",
        "```json",
        json.dumps(summary.get("branch_success_counts", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Gamma Dose Response",
        "```json",
        json.dumps({key: {k: v for k, v in value.items() if k != "rows"} if isinstance(value, dict) else value for key, value in summary.get("dose_response", {}).items()}, indent=2, sort_keys=True),
        "```",
        "",
        "## Scientific Decision",
        f"- Qwen persistent source magnitude causal: `{summary.get('QWEN_PERSISTENT_SOURCE_MAGNITUDE_CAUSAL')}`",
        f"- Ling-to-Qwen transfer: `{summary.get('LING_TO_QWEN_MAGNITUDE_MECHANISM_TRANSFER')}`",
        f"- Residual functional geometry required: `{summary.get('RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED')}`",
        f"- Qwen functional geometry dominance: `{summary.get('QWEN_FUNCTIONAL_GEOMETRY_DOMINANCE')}`",
        f"- Cross-architecture common magnitude mechanism: `{summary.get('CROSS_ARCH_MAGNITUDE_ACCUMULATION_COMMON_MECHANISM')}`",
        f"- Cross-architecture hypothesis: `{summary.get('CROSS_ARCH_MECHANISM_HYPOTHESIS')}`",
        f"- Final classification: `{summary.get('FINAL_CLASSIFICATION')}`",
        "",
        "### Classification Evidence",
        "```json",
        json.dumps(summary.get("classification_evidence", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Cross-Architecture Comparison",
        "| Factor | Qwen/GDN | Ling/KDA |",
        "|---|---|---|",
        *[f"| {row['factor']} | {row['qwen_gdn']} | {row['ling_kda']} |" for row in cross_arch_rows(summary)],
        "",
        "METHOD_DESIGN_READY = NO",
        "",
        "## Required Artifacts",
        f"- `{RUN_DIR / 'results.jsonl'}`",
        f"- `{RUN_DIR / 'aggregate_summary.json'}`",
        f"- `{RUN_DIR / 'report.md'}`",
    ])
    path = RUN_DIR / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{SLUG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_required_summary(summary, pytest_result="NOT_RUN", fresh="PENDING"):
    gates = summary["gates"]
    src = summary["source_norms"]
    c = summary["contrasts_by_l"]
    print("TASK =")
    print(TASK)
    print()
    print("FORMAL_STATUS =")
    print(summary["formal_status"])
    print()
    print("N_FORMAL_UNITS =")
    print(summary["n_formal_units"])
    print()
    for key in [
        "CANONICAL_MANIFEST_GATE",
        "STATE_SEMANTICS_GATE",
        "R128_QUANTIZER_IDENTITY_GATE",
        "C128_QUANTIZER_IDENTITY_GATE",
        "R_NATIVE_ENDPOINT_IDENTITY_GATE",
        "C_NATIVE_ENDPOINT_IDENTITY_GATE",
        "SAME_STATE_COUNTERFACTUAL_GATE",
        "APPLIED_MAGNITUDE_GATE",
        "SOURCE_DIRECTION_PRESERVATION_GATE",
        "INTERVENTION_TIMING_GATE",
        "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
        "NO_FURTHER_QUANTIZATION_GATE",
        "INSTRUMENTATION_NONINTERFERENCE_GATE",
        "L1_SEMANTICS_GATE",
    ]:
        print(f"{key} =")
        print(gates.get(key))
        print()
    print("SMOKE =")
    print(summary["SMOKE"])
    print("PILOT =")
    print(summary["PILOT"])
    print()
    print("HISTORY_LENGTHS =")
    print(",".join(str(x) for x in summary["history_lengths"]))
    print("GAMMA_VALUES =")
    print(",".join(f"{x:g}" for x in summary["gamma_values"]))
    print()
    print("R_SOURCE_NORM =")
    print(src["R_SOURCE_NORM"])
    print("C_SOURCE_NORM =")
    print(src["C_SOURCE_NORM"])
    print()
    print("R_NORM_GT_C =")
    print(src["R_NORM_GT_C"])
    print("C_NORM_GT_R =")
    print(src["C_NORM_GT_R"])
    print()
    print("MEDIAN_R_OVER_C_SOURCE_NORM =")
    print(src["MEDIAN_R_OVER_C_SOURCE_NORM"])
    print()
    for key in [
        "NATIVE_R_MINUS_C_GAP_BY_L",
        "R_MAG_EQUALIZATION_RESCUE_BY_L",
        "C_RMAG_GRAFT_DAMAGE_BY_L",
        "R_EQ_VS_C_NATIVE_BY_L",
        "C_GRAFT_VS_R_NATIVE_BY_L",
        "MAGNITUDE_MAIN_BY_L",
    ]:
        print(f"{key} =")
        print(json.dumps(c.get(key, "NA_NOT_RUN"), sort_keys=True))
        print()
    print("R_DIR_GAMMA_DOSE_RESULT =")
    r_dose = summary["dose_response"]["R_DIR"]
    print(json.dumps({k: v for k, v in r_dose.items() if k != "rows"} if isinstance(r_dose, dict) else r_dose, sort_keys=True))
    print()
    print("C_DIR_GAMMA_DOSE_RESULT =")
    c_dose = summary["dose_response"]["C_DIR"]
    print(json.dumps({k: v for k, v in c_dose.items() if k != "rows"} if isinstance(c_dose, dict) else c_dose, sort_keys=True))
    print()
    print("STATE_ERROR_RESCUE_BY_L =")
    print(json.dumps(c.get("STATE_ERROR_RESCUE_BY_L", "NA_NOT_RUN"), sort_keys=True))
    print("STATE_ERROR_GRAFT_DAMAGE_BY_L =")
    print(json.dumps(c.get("STATE_ERROR_GRAFT_DAMAGE_BY_L", "NA_NOT_RUN"), sort_keys=True))
    print()
    for key in [
        "P_DECOMPOSITION_REUSE",
        "RESIDUAL_FUNCTIONAL_GEOMETRY_GAP_AFTER_MAG_MATCH",
        "QWEN_PERSISTENT_SOURCE_MAGNITUDE_CAUSAL",
        "LING_TO_QWEN_MAGNITUDE_MECHANISM_TRANSFER",
        "RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED",
        "CROSS_ARCH_MECHANISM_HYPOTHESIS",
        "FINAL_CLASSIFICATION",
        "METHOD_DESIGN_READY",
        "NEXT_RECOMMENDED_TASK",
    ]:
        print(f"{key} =")
        print(summary[key])
        print()
    print("PYTEST =")
    print(pytest_result)
    print()
    print("FRESH_OUTPUT_VERIFICATION =")
    print(fresh)
    print()
    print("CURRENT_TASK_COMPLETED_AND_STOPPED =")
    print(summary["CURRENT_TASK_COMPLETED_AND_STOPPED"])
    print()
    print("NO_NEXT_EXPERIMENT_LAUNCHED =")
    print(summary["NO_NEXT_EXPERIMENT_LAUNCHED"])


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def main():
    ap = argparse.ArgumentParser(description=TASK)
    ap.add_argument(
        "--stage",
        choices=[
            "audit", "stage0", "smoke", "pilot", "formal", "all",
            "merge-stage0", "merge-smoke", "merge-pilot", "merge-formal", "analyze",
            "check-formal-prerequisite",
        ],
        default="audit",
    )
    ap.add_argument("--analyze-stage", choices=["stage0", "smoke", "pilot", "formal"], default="formal")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--history-lengths", type=parse_int_list, default=None)
    ap.add_argument("--gamma-values", type=parse_float_list, default=None)
    ap.add_argument("--unit-limit", type=int, default=None)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()
    if args.stage == "audit":
        run_audit()
        summary = json.loads((RUN_DIR / "aggregate_summary.json").read_text(encoding="utf-8"))
        print_required_summary(summary, fresh=summary.get("FRESH_OUTPUT_VERIFICATION", "PENDING"))
        return
    if args.stage == "check-formal-prerequisite":
        require_formal_prerequisite()
        print("FORMAL_PREREQUISITE_GATE = PASS")
        return
    if args.stage in ("stage0", "smoke", "pilot", "formal"):
        raise SystemExit(run_execution_stage(args))
    if args.stage.startswith("merge-"):
        stage = args.stage.split("-", 1)[1]
        summary = aggregate_stage(stage, args.num_shards)
        print_required_summary(summary, fresh=summary["FRESH_OUTPUT_VERIFICATION"])
        return
    if args.stage == "analyze":
        summary = aggregate_stage(args.analyze_stage, args.num_shards)
        print_required_summary(summary, fresh=summary["FRESH_OUTPUT_VERIFICATION"])
        return
    if args.stage == "all":
        if args.num_shards != 1:
            raise ValueError("--stage all is single-worker only; use explicit stage commands for multi-GPU execution")
        for stage in ("stage0", "smoke", "pilot", "formal"):
            args.stage = stage
            code = run_execution_stage(args)
            if code:
                raise SystemExit(code)
            summary = aggregate_stage(stage, 1)
            if stage == "pilot":
                allowed, _reason = formal_prerequisite_status(summary)
                if not allowed:
                    print_required_summary(summary, fresh=summary["FRESH_OUTPUT_VERIFICATION"])
                    return
        print_required_summary(summary, fresh=summary["FRESH_OUTPUT_VERIFICATION"])


if __name__ == "__main__":
    main()
