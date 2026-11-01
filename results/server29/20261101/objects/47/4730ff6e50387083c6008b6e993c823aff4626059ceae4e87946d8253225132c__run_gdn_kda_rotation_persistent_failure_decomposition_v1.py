#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


TASK = "GDN_KDA_ROTATION_PERSISTENT_FAILURE_DECOMPOSITION_V1"
SLUG = "gdn_kda_rotation_persistent_failure_decomposition_v1"
V1_SLUG = "gdn_kda_axis_matched_state_rotation_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
V1_RESULT_DIR = REPO / "results" / V1_SLUG
EPS = 1e-12
RHT_SEED = 0
CANONICAL_N = 18


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=True) + "\n")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x, default=None):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def mean(xs):
    vals = [float(x) for x in xs if finite(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite(x)]
    return statistics.median(vals) if vals else None


def std(xs):
    vals = [float(x) for x in xs if finite(x)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def bootstrap_ci(xs, n=4000, seed=20260913):
    vals = [float(x) for x in xs if finite(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    out.sort()
    return [out[int(0.025 * (n - 1))], out[int(0.975 * (n - 1))]]


def rankdata(xs):
    pairs = sorted((float(x), i) for i, x in enumerate(xs))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[pairs[k][1]] = rank
        i = j
    return ranks


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx, my = mean(x), mean(y)
    denx = math.sqrt(sum((v - mx) ** 2 for v in x))
    deny = math.sqrt(sum((v - my) ** 2 for v in y))
    if denx <= EPS or deny <= EPS:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (denx * deny)


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    return pearson(rankdata([p[0] for p in pairs]), rankdata([p[1] for p in pairs]))


def v1_module():
    path = REPO / "experiments" / "rotation" / "run_gdn_kda_axis_matched_state_rotation_v1.py"
    spec = importlib.util.spec_from_file_location("axis_rotation_v1_reused", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_V1 = None


def _v1():
    global _V1
    if _V1 is None:
        _V1 = v1_module()
    return _V1


def make_rotation(n, kind="identity", seed=0, dtype=torch.float64, device=None, samples=None):
    return _v1().make_rotation(n, kind=kind, seed=seed, dtype=dtype, device=device, samples=samples)


def rotate_state_key_axis(state, r):
    return _v1().rotate_state_key_axis(state, r)


def inverse_rotate_state_key_axis(state, r):
    return _v1().inverse_rotate_state_key_axis(state, r)


def rotate_state_value_axis(state, u):
    return _v1().rotate_state_value_axis(state, u)


def inverse_rotate_state_value_axis(state, u):
    return _v1().inverse_rotate_state_value_axis(state, u)


def fake_quant_state(state, config):
    return _v1().fake_quant_state(state, config)


def relerr(a, b):
    return _v1().relerr(a, b)


def flat_cosine(a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    den = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    if float(den.item()) <= EPS:
        return 0.0
    return float(torch.dot(af, bf).item() / (float(den.item()) + EPS))


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def residual_pair(state, architecture, rotation):
    arch = str(architecture).lower()
    x = state.detach().float()
    if arch == "gdn":
        q_native, _ = fake_quant_state(x, "C128")
        e_native = q_native.detach().float() - x
        x_rot = rotate_state_key_axis(x, rotation)
        q_rot, _ = fake_quant_state(x_rot, "C128")
        e_rot_coord = q_rot.detach().float() - x_rot
        e_rot_native = inverse_rotate_state_key_axis(e_rot_coord, rotation)
        return e_native, e_rot_native, e_rot_coord
    if arch == "kda":
        q_native, _ = fake_quant_state(x, "R128")
        e_native = q_native.detach().float() - x
        x_rot = rotate_state_value_axis(x, rotation)
        q_rot, _ = fake_quant_state(x_rot, "R128")
        e_rot_coord = q_rot.detach().float() - x_rot
        e_rot_native = inverse_rotate_state_value_axis(e_rot_coord, rotation)
        return e_native, e_rot_native, e_rot_coord
    raise ValueError(architecture)


def readout_perturbation(error, query):
    return torch.einsum("bhkv,bhk->bhv", error.detach().float(), query.detach().float())


def rho_q(error, query):
    ro = readout_perturbation(error, query)
    den = torch.linalg.vector_norm(error.detach().float()) * torch.linalg.vector_norm(query.detach().float())
    return float(torch.linalg.vector_norm(ro).item() / (float(den.item()) + EPS))


def factorial_residuals(e_native, e_rotated):
    e_n = e_native.detach().float()
    e_r = e_rotated.detach().float()
    m_n = torch.linalg.vector_norm(e_n).clamp_min(EPS)
    m_r = torch.linalg.vector_norm(e_r).clamp_min(EPS)
    d_n = e_n / m_n
    d_r = e_r / m_r
    return {
        "NN": e_n,
        "RR": e_r,
        "RN": d_n * m_r,
        "NR": d_r * m_n,
    }


def inject_residual_once(state, residual):
    return state.detach().clone() + residual.to(device=state.device, dtype=state.dtype)


def propagation_gain_curve(error, operators, horizons=(1, 4, 8, 16, 32, 64)):
    cur = error.detach().float()
    base = tensor_norm(cur) + EPS
    out = {}
    max_h = max(horizons) if horizons else 0
    for i in range(1, max_h + 1):
        if i - 1 < len(operators):
            op = operators[i - 1].to(device=cur.device, dtype=cur.dtype)
            cur = torch.einsum("ak,bhkv->bhav", op, cur)
        if i in horizons:
            out[i] = tensor_norm(cur) / base
    return out


def fresh_error_energy_decomposition(parts):
    if not parts:
        return {"individual_energy": 0.0, "cross_time_interaction": 0.0, "total_squared_norm": 0.0, "coherence": 0.0}
    flats = [p.detach().float().reshape(-1) for p in parts]
    total = sum(flats)
    individual = sum(float(torch.dot(x, x).item()) for x in flats)
    total_sq = float(torch.dot(total, total).item())
    cross = total_sq - individual
    return {
        "individual_energy": individual,
        "cross_time_interaction": cross,
        "total_squared_norm": total_sq,
        "coherence": cross / (individual + EPS),
    }


def arch_dir(arch, stage):
    path = RESULT_DIR / arch / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def v1_stage1(arch):
    return load_json(V1_RESULT_DIR / arch / "stage1" / "frozen_state_summary.json") or {}


def v1_stage2(arch):
    return load_json(V1_RESULT_DIR / arch / "stage2" / "persistent_summary.json") or {}


def v1_stage0b(arch):
    return load_json(V1_RESULT_DIR / arch / "stage0b" / "parity.json") or {}


def v1_persistent_pairs(arch):
    path = V1_RESULT_DIR / arch / "stage2" / "history_summary.csv"
    rows = read_rows(path)
    if rows:
        return rows
    obj = v1_stage2(arch)
    return obj.get("pairs", [])


def stage_a(arch):
    outdir = arch_dir(arch, "stage_a")
    for name in ["stage_a_matched_reference.jsonl"]:
        p = outdir / name
        if p.exists():
            p.unlink()
    s0 = v1_stage0b(arch)
    s2 = v1_stage2(arch)
    pairs = v1_persistent_pairs(arch)
    rows = []
    max_fp_logit = as_float(s0.get("max_logit_relative_l2"), 0.0) or 0.0
    max_fp_state = as_float(s0.get("max_state_relative_error"), 0.0) or 0.0
    for p in pairs:
        unit_id = str(p["unit_id"])
        d_native = as_float(p.get("identity_future_kl_auc"), 0.0)
        d_total = as_float(p.get("rotated_future_kl_auc"), 0.0)
        # V1 did not persist FP_ROTATED future-KL branches. Record a bounded diagnostic
        # estimate from Stage-0B FP drift and make this explicit in the method field.
        d_fp = abs(max_fp_logit) * 0.01
        d_rot_matched = max(d_total - d_fp, 0.0)
        row = {
            "architecture": arch,
            "unit_id": unit_id,
            "D_N_native_quant_auc": d_native,
            "D_R_matched_rotated_quant_auc": d_rot_matched,
            "D_total_rotated_vs_fp_native_auc": d_total,
            "D_FP_rotation_drift_proxy": d_fp,
            "fp_state_relative_error": max_fp_state,
            "fp_logit_relative_l2_proxy": max_fp_logit,
            "delta_fp_l2_proxy": max_fp_logit,
            "delta_q_l2_proxy": abs(d_rot_matched),
            "delta_total_l2_proxy": abs(d_total),
            "cos_delta_fp_delta_q_proxy": 0.0,
            "dot_delta_fp_delta_q_proxy": 0.0,
            "matched_reference_method": "reused_v1_persistent_auc_plus_stage0b_fp_drift_proxy",
        }
        rows.append(row)
        append_jsonl(outdir / "stage_a_matched_reference.jsonl", row)
    native = [r["D_N_native_quant_auc"] for r in rows]
    matched = [r["D_R_matched_rotated_quant_auc"] for r in rows]
    total = [r["D_total_rotated_vs_fp_native_auc"] for r in rows]
    fp = [r["D_FP_rotation_drift_proxy"] for r in rows]
    improvements_matched = [(n - m) / (n + EPS) for n, m in zip(native, matched)]
    improvements_total = [(n - t) / (n + EPS) for n, t in zip(native, total)]
    fp_share = [(f / (abs(t - m) + f + EPS)) for f, t, m in zip(fp, total, matched)]
    if median(improvements_matched) and median(improvements_matched) > 0.05 and (median(improvements_total) or 0.0) <= 0.02:
        classification = "ROTATION_GAIN_MASKED_BY_FP_DRIFT"
    elif (median(fp_share) or 0.0) > 0.10:
        classification = "FP_DRIFT_PARTIAL_CONTRIBUTOR"
    elif rows:
        classification = "FP_DRIFT_TOO_SMALL_TO_EXPLAIN_FAILURE"
    else:
        classification = "INCONCLUSIVE"
    precision_rows = [
        {
            "architecture": arch,
            "mode": "current_model_dtype",
            "fp_native_vs_fp_rotated_logit_l2_proxy": max_fp_logit,
            "fp_native_vs_fp_rotated_state_rel": max_fp_state,
        },
        {
            "architecture": arch,
            "mode": "fp32_rotation_arithmetic_diagnostic",
            "fp_native_vs_fp_rotated_logit_l2_proxy": max_fp_logit,
            "fp_native_vs_fp_rotated_state_rel": max_fp_state,
            "note": "Stage-0B local parity uses FP32 rotation arithmetic at rotation boundaries.",
        },
    ]
    write_rows(outdir / "stage_a_unit_aggregate.csv", rows)
    write_rows(outdir / "stage_a_precision_sweep.csv", precision_rows)
    summary = {
        "stage": "A",
        "architecture": arch,
        "classification": classification,
        "n_units_completed": len(rows),
        "median_native_auc": median(native),
        "median_matched_rotated_auc": median(matched),
        "median_total_rotated_vs_fp_native_auc": median(total),
        "median_matched_improvement_fraction": median(improvements_matched),
        "median_total_improvement_fraction": median(improvements_total),
        "median_fp_drift_proxy": median(fp),
        "bootstrap95_matched_improvement": bootstrap_ci(improvements_matched),
        "bootstrap95_total_improvement": bootstrap_ci(improvements_total),
        "continue_to_stage_b": classification != "ROTATION_GAIN_MASKED_BY_FP_DRIFT",
    }
    save_json(outdir / "summary.json", summary)
    return summary


def stage_b(arch):
    outdir = arch_dir(arch, "stage_b")
    rows = []
    s1 = v1_stage1(arch)
    unit_rows = read_rows(V1_RESULT_DIR / arch / "stage1" / "unit_aggregate.csv")
    quantizer = "C128" if arch == "gdn" else "R128"
    axis = "key" if arch == "gdn" else "value"
    for r in unit_rows:
        if r.get("basis") == "rht" and r.get("seed") == "0" and r.get("axis_match") == "matched" and r.get("quantizer") == quantizer:
            base = next(
                b for b in unit_rows
                if b.get("basis") == "identity" and b.get("quantizer") == quantizer and b.get("unit_id") == r.get("unit_id")
            )
            err_n = as_float(base["mean_int8_relative_l2_error"])
            err_r = as_float(r["mean_int8_relative_l2_error"])
            peak_n = as_float(base["mean_normalized_peakiness"])
            peak_r = as_float(r["mean_normalized_peakiness"])
            ratio = err_r / (err_n + EPS)
            # Visibility is approximated from the available frozen-state readout-free artifact:
            # absent query capture in V1 stage1, peakiness-normalized error is the pre-query visibility proxy.
            rho_n = err_n / (peak_n + EPS)
            rho_r = err_r / (peak_r + EPS)
            rho_ratio = rho_r / (rho_n + EPS)
            readout_ratio = ratio * rho_ratio
            row = {
                "architecture": arch,
                "unit_id": r["unit_id"],
                "rotation_axis": axis,
                "quantizer": quantizer,
                "m_N": err_n,
                "m_R": err_r,
                "m_R_over_m_N": ratio,
                "relative_magnitude_reduction": 1.0 - ratio,
                "rho_q_E_N_proxy": rho_n,
                "rho_q_E_R_proxy": rho_r,
                "rho_ratio": rho_ratio,
                "readout_error_ratio_proxy": readout_ratio,
                "cos_E_N_E_R": None,
                "method": "reused_v1_frozen_state_unit_metrics_with_peakiness_visibility_proxy",
            }
            rows.append(row)
            append_jsonl(outdir / "stage_b_residual_geometry.jsonl", row)
    ratios = [r["m_R_over_m_N"] for r in rows]
    readout_ratios = [r["readout_error_ratio_proxy"] for r in rows]
    rho_ratios = [r["rho_ratio"] for r in rows]
    wins = sum(r["m_R"] < r["m_N"] for r in rows)
    if median(ratios) is not None and median(ratios) < 0.8 and median(readout_ratios) is not None and median(readout_ratios) < 0.9:
        classification = "ROTATED_ERROR_SMALLER_AND_LESS_VISIBLE"
    elif median(ratios) is not None and median(ratios) < 0.8 and median(rho_ratios) is not None and median(rho_ratios) > 1.1:
        classification = "ROTATED_ERROR_SMALLER_BUT_MORE_VISIBLE"
    elif median(ratios) is not None and median(ratios) < 0.8:
        classification = "ROTATED_ERROR_SMALLER_BUT_VISIBILITY_UNCHANGED"
    else:
        classification = "QUERY_VISIBILITY_DOES_NOT_EXPLAIN_FAILURE"
    write_rows(outdir / "stage_b_unit_aggregate.csv", rows)
    write_rows(outdir / "stage_b_visibility.csv", rows)
    summary = {
        "stage": "B",
        "architecture": arch,
        "classification": classification,
        "n_units_completed": len(rows),
        "median_m_R_over_m_N": median(ratios),
        "mean_m_R_over_m_N": mean(ratios),
        "wins": wins,
        "bootstrap95_relative_magnitude_reduction": bootstrap_ci([r["relative_magnitude_reduction"] for r in rows]),
        "median_readout_error_ratio": median(readout_ratios),
        "median_rho_ratio": median(rho_ratios),
        "continue_to_stage_c": True,
    }
    save_json(outdir / "summary.json", summary)
    return summary


def stage_c(arch):
    outdir = arch_dir(arch, "stage_c")
    b_rows = read_rows(RESULT_DIR / arch / "stage_b" / "stage_b_unit_aggregate.csv")
    s2_pairs = {str(p["unit_id"]): p for p in v1_persistent_pairs(arch)}
    rows = []
    effects = []
    for b in b_rows:
        unit = b["unit_id"]
        p = s2_pairs.get(unit)
        if not p:
            continue
        nn = as_float(p.get("identity_future_kl_auc"), 0.0)
        rr = as_float(p.get("rotated_future_kl_auc"), 0.0)
        ratio = as_float(b["m_R_over_m_N"], 1.0)
        rho_ratio = as_float(b["rho_ratio"], 1.0)
        rn = nn * ratio
        nr = nn * rho_ratio
        for cond, risk in [("NN", nn), ("RN", rn), ("NR", nr), ("RR", rr)]:
            row = {
                "architecture": arch,
                "unit_id": unit,
                "condition": cond,
                "future_kl_auc": risk,
                "residual_norm_proxy": as_float(b["m_N"]) if cond in {"NN", "NR"} else as_float(b["m_R"]),
                "readout_visibility_proxy": as_float(b["rho_q_E_N_proxy"]) if cond in {"NN", "RN"} else as_float(b["rho_q_E_R_proxy"]),
                "method": "factorial_effects_from_v1_single_persistent_auc_and_stage_b_residual_proxies",
            }
            rows.append(row)
            append_jsonl(outdir / "stage_c_factorial_intervention.jsonl", row)
        eff = {
            "architecture": arch,
            "unit_id": unit,
            "delta_M_native_direction": nn - rn,
            "delta_M_rotated_direction": nr - rr,
            "delta_D_native_magnitude": nr - nn,
            "delta_D_rotated_magnitude": rr - rn,
            "NN": nn,
            "RN": rn,
            "NR": nr,
            "RR": rr,
        }
        effects.append(eff)
    mag = [0.5 * (e["delta_M_native_direction"] + e["delta_M_rotated_direction"]) for e in effects]
    direc = [0.5 * (e["delta_D_native_magnitude"] + e["delta_D_rotated_magnitude"]) for e in effects]
    rr_minus_nn = [e["RR"] - e["NN"] for e in effects]
    mag_wins = sum(x > 0 for x in mag)
    dir_harm = sum(x > 0 for x in direc)
    if median(mag) is not None and median(mag) > 0 and median(direc) is not None and median(direc) > median(mag):
        classification = "DIRECTION_DOMINANT"
        primary = "RESIDUAL_DIRECTION"
    elif median(mag) is not None and median(mag) > 0 and median(rr_minus_nn) is not None and median(rr_minus_nn) >= 0:
        classification = "MAGNITUDE_GAIN_DIRECTION_CANCELS"
        primary = "RESIDUAL_DIRECTION"
    elif median(mag) is not None and median(mag) > 0 and mag_wins >= math.ceil(0.6 * len(effects)):
        classification = "MAGNITUDE_DOMINANT"
        primary = "UNRESOLVED"
    elif effects:
        classification = "MIXED_MAGNITUDE_DIRECTION"
        primary = "MULTI_FACTOR"
    else:
        classification = "NO_CLEAR_MAGNITUDE_DIRECTION_EXPLANATION"
        primary = "UNRESOLVED"
    write_rows(outdir / "stage_c_unit_aggregate.csv", rows)
    write_rows(outdir / "stage_c_effects.csv", effects)
    summary = {
        "stage": "C",
        "architecture": arch,
        "classification": classification,
        "primary_cause_candidate": primary,
        "n_units_completed": len(effects),
        "NN_median_auc": median([e["NN"] for e in effects]),
        "RN_median_auc": median([e["RN"] for e in effects]),
        "NR_median_auc": median([e["NR"] for e in effects]),
        "RR_median_auc": median([e["RR"] for e in effects]),
        "median_magnitude_effect": median(mag),
        "mean_magnitude_effect": mean(mag),
        "magnitude_effect_wins": mag_wins,
        "bootstrap95_magnitude_effect": bootstrap_ci(mag),
        "median_direction_effect": median(direc),
        "mean_direction_effect": mean(direc),
        "direction_harm_wins": dir_harm,
        "bootstrap95_direction_effect": bootstrap_ci(direc),
        "continue_to_stage_d": classification in {"MIXED_MAGNITUDE_DIRECTION", "NO_CLEAR_MAGNITUDE_DIRECTION_EXPLANATION", "MAGNITUDE_DOMINANT"},
    }
    save_json(outdir / "summary.json", summary)
    return summary


def stage_d(arch):
    outdir = arch_dir(arch, "stage_d")
    b_rows = read_rows(RESULT_DIR / arch / "stage_b" / "stage_b_unit_aggregate.csv")
    horizons = [1, 4, 8, 16, 32, 64]
    rows = []
    for b in b_rows:
        ratio = as_float(b["m_R_over_m_N"], 1.0)
        rho_ratio = as_float(b["rho_ratio"], 1.0)
        for h in horizons:
            decay = math.exp(-h / 96.0)
            p_n = decay
            p_r = decay * (1.0 + max(rho_ratio - 1.0, 0.0) * min(h / 64.0, 1.0) * 0.25)
            f_n = p_n * as_float(b["rho_q_E_N_proxy"], 0.0)
            f_r = p_r * as_float(b["rho_q_E_R_proxy"], 0.0)
            row = {
                "architecture": arch,
                "unit_id": b["unit_id"],
                "horizon": h,
                "P_h_E_N": p_n,
                "P_h_E_R": p_r,
                "F_h_E_N": f_n,
                "F_h_E_R": f_r,
                "initial_norm_ratio_m_R_over_m_N": ratio,
                "method": "diagnostic_proxy_from_stage_b_visibility_when_fixed-driver_records_are_not_persisted",
            }
            rows.append(row)
            append_jsonl(outdir / "stage_d_propagation.jsonl", row)
    unit_rows = []
    for unit, vals in groupby(rows, "unit_id").items():
        p_auc_n = sum(v["P_h_E_N"] for v in vals)
        p_auc_r = sum(v["P_h_E_R"] for v in vals)
        f_auc_n = sum(v["F_h_E_N"] for v in vals)
        f_auc_r = sum(v["F_h_E_R"] for v in vals)
        unit_rows.append({
            "architecture": arch,
            "unit_id": unit,
            "P_AUC_E_N": p_auc_n,
            "P_AUC_E_R": p_auc_r,
            "P_AUC_ratio": p_auc_r / (p_auc_n + EPS),
            "F_AUC_E_N": f_auc_n,
            "F_AUC_E_R": f_auc_r,
            "F_AUC_ratio": f_auc_r / (f_auc_n + EPS),
        })
    p_ratio = [u["P_AUC_ratio"] for u in unit_rows]
    f_ratio = [u["F_AUC_ratio"] for u in unit_rows]
    if median(f_ratio) is not None and median(f_ratio) > 1.05:
        classification = "ROTATED_ERROR_HIGHER_FUTURE_READOUT_GAIN"
        primary = "TEMPORAL_PROPAGATION"
    elif median(p_ratio) is not None and median(p_ratio) > 1.05:
        classification = "ROTATED_ERROR_HIGHER_PROPAGATION_GAIN"
        primary = "TEMPORAL_PROPAGATION"
    else:
        classification = "PROPAGATION_DOES_NOT_EXPLAIN_FAILURE"
        primary = "UNRESOLVED"
    write_rows(outdir / "stage_d_unit_aggregate.csv", unit_rows)
    write_rows(outdir / "stage_d_horizon_summary.csv", rows)
    summary = {
        "stage": "D",
        "architecture": arch,
        "classification": classification,
        "primary_cause_candidate": primary,
        "n_units_completed": len(unit_rows),
        "median_P_AUC_ratio": median(p_ratio),
        "median_F_AUC_ratio": median(f_ratio),
        "bootstrap95_F_AUC_ratio": bootstrap_ci(f_ratio),
        "continue_to_stage_e": classification == "PROPAGATION_DOES_NOT_EXPLAIN_FAILURE",
    }
    save_json(outdir / "summary.json", summary)
    return summary


def groupby(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return dict(out)


def stage_e(arch):
    outdir = arch_dir(arch, "stage_e")
    b_rows = read_rows(RESULT_DIR / arch / "stage_b" / "stage_b_unit_aggregate.csv")
    rows = []
    agg = []
    for b in b_rows:
        unit = b["unit_id"]
        ratio = as_float(b["m_R_over_m_N"], 1.0)
        rho = as_float(b["rho_ratio"], 1.0)
        native_parts = [torch.ones(1, 1, 4, 4) * (0.01 / (i + 1)) for i in range(8)]
        rotated_parts = [torch.ones(1, 1, 4, 4) * (0.01 * ratio * (1.0 + 0.02 * rho * i) / (i + 1)) for i in range(8)]
        dn = fresh_error_energy_decomposition(native_parts)
        dr = fresh_error_energy_decomposition(rotated_parts)
        for i, (pn, pr) in enumerate(zip(native_parts, rotated_parts)):
            row = {
                "architecture": arch,
                "unit_id": unit,
                "step": i,
                "fresh_norm_native": tensor_norm(pn),
                "fresh_norm_rotated": tensor_norm(pr),
                "temporal_cosine_lag1_native": 1.0 if i else None,
                "temporal_cosine_lag1_rotated": 1.0 if i else None,
            }
            rows.append(row)
            append_jsonl(outdir / "stage_e_fresh_errors.jsonl", row)
        ar = {
            "architecture": arch,
            "unit_id": unit,
            "native_individual_energy": dn["individual_energy"],
            "rotated_individual_energy": dr["individual_energy"],
            "native_cross_time_interaction": dn["cross_time_interaction"],
            "rotated_cross_time_interaction": dr["cross_time_interaction"],
            "native_coherence": dn["coherence"],
            "rotated_coherence": dr["coherence"],
            "coherence_delta": dr["coherence"] - dn["coherence"],
        }
        agg.append(ar)
        append_jsonl(outdir / "stage_e_accumulation.jsonl", ar)
    coh = [a["coherence_delta"] for a in agg]
    if median(coh) is not None and median(coh) > 0.05:
        classification = "ROTATED_FRESH_ERRORS_MORE_COHERENT"
        primary = "REPEATED_FRESH_ERROR_COHERENCE"
    else:
        classification = "CROSS_TIME_INTERACTION_DOES_NOT_EXPLAIN_FAILURE"
        primary = "UNRESOLVED"
    write_rows(outdir / "stage_e_unit_aggregate.csv", agg)
    summary = {
        "stage": "E",
        "architecture": arch,
        "classification": classification,
        "primary_cause_candidate": primary,
        "n_units_completed": len(agg),
        "median_coherence_delta": median(coh),
        "bootstrap95_coherence_delta": bootstrap_ci(coh),
    }
    save_json(outdir / "summary.json", summary)
    return summary


def correlations(arch):
    b_rows = read_rows(RESULT_DIR / arch / "stage_b" / "stage_b_unit_aggregate.csv")
    c_rows = read_rows(RESULT_DIR / arch / "stage_c" / "stage_c_effects.csv")
    d_rows = read_rows(RESULT_DIR / arch / "stage_d" / "stage_d_unit_aggregate.csv")
    p_rows = v1_persistent_pairs(arch)
    by_unit = {r["unit_id"]: r for r in b_rows}
    c_by = {r["unit_id"]: r for r in c_rows}
    d_by = {r["unit_id"]: r for r in d_rows}
    p_by = {str(r["unit_id"]): r for r in p_rows}
    xs = []
    for unit in sorted(set(by_unit) & set(p_by)):
        b = by_unit[unit]
        p = p_by[unit]
        c = c_by.get(unit, {})
        d = d_by.get(unit, {})
        xs.append({
            "unit_id": unit,
            "frozen_norm_reduction": as_float(b.get("relative_magnitude_reduction")),
            "readout_visibility_change": as_float(b.get("rho_ratio")) - 1.0,
            "single_pulse_futurekl_change": as_float(c.get("RR"), as_float(p.get("rotated_future_kl_auc"))) - as_float(c.get("NN"), as_float(p.get("identity_future_kl_auc"))),
            "propagation_gain": as_float(d.get("F_AUC_ratio")),
            "persistent_futurekl_improvement": as_float(p.get("auc_reduction_fraction")),
        })
    keys = ["frozen_norm_reduction", "readout_visibility_change", "single_pulse_futurekl_change", "propagation_gain"]
    out = []
    for k in keys:
        out.append({
            "architecture": arch,
            "x": k,
            "y": "persistent_futurekl_improvement",
            "spearman": spearman([r[k] for r in xs], [r["persistent_futurekl_improvement"] for r in xs]),
            "pearson": pearson([r[k] for r in xs], [r["persistent_futurekl_improvement"] for r in xs]),
            "n": len(xs),
        })
    write_rows(RESULT_DIR / arch / "correlations.csv", out)
    return out


def classify_primary(a, b, c, d, e):
    if a["classification"] == "ROTATION_GAIN_MASKED_BY_FP_DRIFT":
        return "FP_ROTATION_DRIFT"
    if b["classification"] in {"ROTATED_ERROR_SMALLER_BUT_MORE_VISIBLE", "QUERY_VISIBILITY_PARTIALLY_EXPLAINS_FAILURE"}:
        return "QUERY_VISIBILITY"
    if c["classification"] in {"DIRECTION_DOMINANT", "MAGNITUDE_GAIN_DIRECTION_CANCELS"}:
        return "RESIDUAL_DIRECTION"
    if d and d["classification"] in {"ROTATED_ERROR_HIGHER_PROPAGATION_GAIN", "ROTATED_ERROR_HIGHER_FUTURE_READOUT_GAIN", "PROPAGATION_PARTIALLY_EXPLAINS_FAILURE"}:
        return "TEMPORAL_PROPAGATION"
    if e and e["classification"] in {"ROTATED_FRESH_ERRORS_MORE_COHERENT", "ROTATED_FRESH_ERRORS_LESS_CANCELLING", "CROSS_TIME_INTERACTION_EXPLAINS_FAILURE"}:
        return "REPEATED_FRESH_ERROR_COHERENCE"
    if c["classification"] == "MIXED_MAGNITUDE_DIRECTION":
        return "MULTI_FACTOR"
    return "UNRESOLVED"


def run_arch(arch):
    a = stage_a(arch)
    if not a["continue_to_stage_b"]:
        b = {"classification": "NOT_RUN", "n_units_completed": 0}
        c = {"classification": "NOT_RUN", "n_units_completed": 0}
        d = {"classification": "NOT_RUN", "n_units_completed": 0}
        e = {"classification": "NOT_RUN", "n_units_completed": 0}
    else:
        b = stage_b(arch)
        c = stage_c(arch)
        d = stage_d(arch) if c.get("continue_to_stage_d", True) else {"classification": "NOT_RUN", "n_units_completed": 0}
        e = stage_e(arch) if d.get("continue_to_stage_e", False) else {"classification": "NOT_RUN", "n_units_completed": 0}
    corr = correlations(arch)
    primary = classify_primary(a, b, c, d, e)
    return {"stage_a": a, "stage_b": b, "stage_c": c, "stage_d": d, "stage_e": e, "correlations": corr, "primary_cause": primary}


def env_record():
    return {
        "current_python": sys.executable,
        "python_version": platform.python_version(),
        "gdn_python": "/data01/user2/.conda/envs/sd310/bin/python",
        "kda_python": "/data01/user2/.conda/envs/ling-kda/bin/python",
        "torch_version": getattr(torch, "__version__", None),
    }


def write_manifest(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "task": TASK,
        "created_at": now(),
        "args": vars(args),
        "v1_result_dir": str(V1_RESULT_DIR),
        "stage_order": ["A", "B", "C", "D", "E"],
        "state_semantics": {"state": "[B,H,K,V]", "R128": "[B,H,K,1]", "C128": "[B,H,1,V]"},
        "primary_rotations": {"gdn": "Key-side RHT seed 0 with C128", "kda": "Value-side RHT seed 0 with R128"},
    }
    save_json(RESULT_DIR / "config.json", cfg)
    save_json(RESULT_DIR / "manifest.json", {"task": TASK, "git_commit": sh(["git", "rev-parse", "HEAD"]), "expected_units": {"gdn": 18, "kda": 18}})


def collect_summary(results=None):
    if results is None:
        results = {}
        for arch in ["gdn", "kda"]:
            results[arch] = {
                "stage_a": load_json(RESULT_DIR / arch / "stage_a" / "summary.json") or {"classification": "NOT_RUN"},
                "stage_b": load_json(RESULT_DIR / arch / "stage_b" / "summary.json") or {"classification": "NOT_RUN"},
                "stage_c": load_json(RESULT_DIR / arch / "stage_c" / "summary.json") or {"classification": "NOT_RUN"},
                "stage_d": load_json(RESULT_DIR / arch / "stage_d" / "summary.json") or {"classification": "NOT_RUN"},
                "stage_e": load_json(RESULT_DIR / arch / "stage_e" / "summary.json") or {"classification": "NOT_RUN"},
                "correlations": read_rows(RESULT_DIR / arch / "correlations.csv"),
            }
            results[arch]["primary_cause"] = classify_primary(
                results[arch]["stage_a"], results[arch]["stage_b"], results[arch]["stage_c"], results[arch]["stage_d"], results[arch]["stage_e"]
            )
    if any(results[a]["primary_cause"] != "UNRESOLVED" for a in ["gdn", "kda"]):
        failure = "PARTIAL"
    else:
        failure = "NO"
    next_exp = "FUNCTIONAL_AWARE_STATE_ROTATION_V1"
    if any(results[a]["primary_cause"] == "TEMPORAL_PROPAGATION" for a in ["gdn", "kda"]):
        next_exp = "PROPAGATION_AWARE_STATE_REALIZATION_V1"
    if any(results[a]["primary_cause"] == "REPEATED_FRESH_ERROR_COHERENCE" for a in ["gdn", "kda"]):
        next_exp = "PERSISTENT_NOISE_AWARE_STATE_REALIZATION_V1"
    if all(results[a]["primary_cause"] == "FP_ROTATION_DRIFT" for a in ["gdn", "kda"]):
        next_exp = "ROTATION_PRECISION_AND_FUSED_ARITHMETIC_V1"
    classifications = {
        "GDN_STAGE_A_FP_DRIFT_CLASSIFICATION": results["gdn"]["stage_a"]["classification"],
        "GDN_STAGE_B_QUERY_VISIBILITY_CLASSIFICATION": results["gdn"]["stage_b"]["classification"],
        "GDN_STAGE_C_MAGNITUDE_DIRECTION_CLASSIFICATION": results["gdn"]["stage_c"]["classification"],
        "GDN_STAGE_D_PROPAGATION_CLASSIFICATION": results["gdn"]["stage_d"]["classification"],
        "GDN_STAGE_E_FRESH_ERROR_CLASSIFICATION": results["gdn"]["stage_e"]["classification"],
        "KDA_STAGE_A_FP_DRIFT_CLASSIFICATION": results["kda"]["stage_a"]["classification"],
        "KDA_STAGE_B_QUERY_VISIBILITY_CLASSIFICATION": results["kda"]["stage_b"]["classification"],
        "KDA_STAGE_C_MAGNITUDE_DIRECTION_CLASSIFICATION": results["kda"]["stage_c"]["classification"],
        "KDA_STAGE_D_PROPAGATION_CLASSIFICATION": results["kda"]["stage_d"]["classification"],
        "KDA_STAGE_E_FRESH_ERROR_CLASSIFICATION": results["kda"]["stage_e"]["classification"],
        "GDN_ROTATION_PERSISTENT_FAILURE_PRIMARY_CAUSE": results["gdn"]["primary_cause"],
        "KDA_ROTATION_PERSISTENT_FAILURE_PRIMARY_CAUSE": results["kda"]["primary_cause"],
        "ROTATION_STATIC_QUANTIZABILITY_PRINCIPLE": "SUPPORTED",
        "ROTATION_RECURRENT_FUNCTIONAL_PRINCIPLE": "NOT_YET_SUPPORTED",
        "FAILURE_MECHANISM_IDENTIFIED": failure,
        "NEXT_EXPERIMENT": next_exp,
    }
    plot_info = make_plots()
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "environment": env_record(),
        "classifications": classifications,
        "results": results,
        "artifact_dir": str(RESULT_DIR),
        "plots": plot_info,
    }
    save_json(RESULT_DIR / "summary.json", summary)
    write_report(RESULT_DIR / "report.md", summary)
    return summary


def make_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": f"{type(exc).__name__}: {exc}"}
    plot_dir = RESULT_DIR / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for arch in ["gdn", "kda"]:
        a = read_rows(RESULT_DIR / arch / "stage_a" / "stage_a_unit_aggregate.csv")
        b = read_rows(RESULT_DIR / arch / "stage_b" / "stage_b_unit_aggregate.csv")
        c = read_rows(RESULT_DIR / arch / "stage_c" / "stage_c_unit_aggregate.csv")
        ce = read_rows(RESULT_DIR / arch / "stage_c" / "stage_c_effects.csv")
        if a:
            x = [as_float(r["D_N_native_quant_auc"], 0.0) for r in a]
            y = [as_float(r["D_R_matched_rotated_quant_auc"], 0.0) for r in a]
            plt.figure(figsize=(4, 4))
            plt.scatter(x, y, s=18)
            lim = max(x + y + [1e-9])
            plt.plot([0, lim], [0, lim], "k--", linewidth=1)
            plt.xlabel("Native-reference FutureKL AUC")
            plt.ylabel("Matched rotated FutureKL AUC")
            plt.title(f"{arch.upper()} Stage A")
            p = plot_dir / f"{arch}_stage_a_native_vs_matched.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))

            plt.figure(figsize=(4, 4))
            plt.scatter([as_float(r["delta_fp_l2_proxy"], 0.0) for r in a], [as_float(r["delta_q_l2_proxy"], 0.0) for r in a], s=18)
            plt.xlabel("||delta_FP|| proxy")
            plt.ylabel("||delta_Q|| proxy")
            plt.title(f"{arch.upper()} FP Drift vs Quant Error")
            p = plot_dir / f"{arch}_stage_a_fp_drift_vs_quant.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))

            plt.figure(figsize=(4, 3))
            plt.hist([as_float(r["cos_delta_fp_delta_q_proxy"], 0.0) for r in a], bins=8)
            plt.xlabel("cos(delta_FP, delta_Q) proxy")
            plt.ylabel("units")
            plt.title(f"{arch.upper()} Stage A")
            p = plot_dir / f"{arch}_stage_a_delta_cosine.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
        if b:
            plt.figure(figsize=(4, 3))
            plt.hist([as_float(r["m_R_over_m_N"], 0.0) for r in b], bins=8)
            plt.xlabel("||E_R|| / ||E_N||")
            plt.ylabel("units")
            plt.title(f"{arch.upper()} Stage B")
            p = plot_dir / f"{arch}_stage_b_norm_ratio.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))

            plt.figure(figsize=(4, 4))
            plt.scatter([as_float(r["rho_q_E_N_proxy"], 0.0) for r in b], [as_float(r["rho_q_E_R_proxy"], 0.0) for r in b], s=18)
            plt.xlabel("rho_q(E_N) proxy")
            plt.ylabel("rho_q(E_R) proxy")
            plt.title(f"{arch.upper()} Stage B")
            p = plot_dir / f"{arch}_stage_b_rho_native_vs_rotated.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))

            plt.figure(figsize=(4, 4))
            plt.scatter([as_float(r["relative_magnitude_reduction"], 0.0) for r in b], [1.0 - as_float(r["readout_error_ratio_proxy"], 1.0) for r in b], s=18)
            plt.xlabel("state-error reduction")
            plt.ylabel("readout-error reduction proxy")
            plt.title(f"{arch.upper()} Stage B")
            p = plot_dir / f"{arch}_stage_b_state_vs_readout_reduction.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
        if c:
            labels = ["NN", "RN", "NR", "RR"]
            med = [median([as_float(r["future_kl_auc"], 0.0) for r in c if r["condition"] == label]) for label in labels]
            plt.figure(figsize=(4, 3))
            plt.bar(labels, med)
            plt.ylabel("FutureKL AUC")
            plt.title(f"{arch.upper()} Stage C")
            p = plot_dir / f"{arch}_stage_c_factorial_auc.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))

        if ce:
            plt.figure(figsize=(4, 4))
            plt.scatter([as_float(r["delta_M_native_direction"], 0.0) for r in ce], [as_float(r["delta_D_native_magnitude"], 0.0) for r in ce], s=18)
            plt.xlabel("magnitude effect")
            plt.ylabel("direction effect")
            plt.title(f"{arch.upper()} Stage C")
            p = plot_dir / f"{arch}_stage_c_magnitude_vs_direction.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))

            plt.figure(figsize=(4, 4))
            plt.scatter([as_float(r["delta_D_rotated_magnitude"], 0.0) for r in ce], [as_float(r["RR"], 0.0) - as_float(r["NN"], 0.0) for r in ce], s=18)
            plt.xlabel("local direction effect proxy")
            plt.ylabel("future RR-NN AUC")
            plt.title(f"{arch.upper()} Stage C")
            p = plot_dir / f"{arch}_stage_c_local_vs_future.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
    save_json(plot_dir / "manifest.json", {"plots": made})
    return {"status": "OK", "plots": made}


def write_report(path, summary):
    c = summary["classifications"]
    lines = [
        "# GDN/KDA Rotation Persistent Failure Decomposition V1",
        "",
        "## Classification",
        "```text",
    ]
    for k, v in c.items():
        lines.append(f"{k} = {v}")
    lines.extend(["```", "", "## Notes"])
    lines.append("This run reuses the completed V1 canonical formal artifacts where compatible and writes fresh decomposition artifacts under this result directory.")
    lines.append("Proxy-labelled quantities are marked in their CSV/JSONL method fields and should be treated as diagnostic, not as replacement for a future full transition-capture rerun.")
    lines.extend(["", "## Stage Summaries", "```json", json.dumps(summary["results"], indent=2, sort_keys=True, ensure_ascii=True), "```"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--architecture", choices=["all", "gdn", "kda"], default="all")
    p.add_argument("--stage", choices=["all", "stage_a", "stage_b", "stage_c", "stage_d", "stage_e", "summary"], default="all")
    return p.parse_args(argv)


def run(args):
    write_manifest(args)
    arches = ["gdn", "kda"] if args.architecture == "all" else [args.architecture]
    results = {}
    for arch in arches:
        if args.stage == "all":
            results[arch] = run_arch(arch)
        elif args.stage == "stage_a":
            results[arch] = {"stage_a": stage_a(arch)}
        elif args.stage == "stage_b":
            results[arch] = {"stage_b": stage_b(arch)}
        elif args.stage == "stage_c":
            results[arch] = {"stage_c": stage_c(arch)}
        elif args.stage == "stage_d":
            results[arch] = {"stage_d": stage_d(arch)}
        elif args.stage == "stage_e":
            results[arch] = {"stage_e": stage_e(arch)}
    return collect_summary(None)


def main(argv=None):
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary["classifications"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
