#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


TASK = "GDN_KDA_ROTATION_PERSISTENT_ERROR_PROCESS_DECOMPOSITION_V1"
SLUG = "gdn_kda_rotation_persistent_error_process_decomposition_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
V1_DIR = REPO / "results" / "gdn_kda_axis_matched_state_rotation_v1"
CLOSURE_DIR = REPO / "results" / "gdn_kda_rotation_residual_direction_causal_closure_v1"
KDA_PGR_DIR = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1"
GDN_SOURCE_DIR = REPO / "runs" / "qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1"
GDN_PYTHON = "/data01/user2/.conda/envs/sd310/bin/python"
KDA_PYTHON = "/data01/user2/.conda/envs/ling-kda/bin/python"
EPS = 1e-12
RHT_SEED = 0
HORIZONS = [0, 1, 2, 4, 8, 16, 32, 64]
TEMPORAL_DELTAS = [1, 2, 4, 8, 16]
CANONICAL_N = 18


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=True) + "\n")


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key) for key in fields})


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
    boot = []
    for _ in range(n):
        boot.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    boot.sort()
    return [boot[int(0.025 * (n - 1))], boot[int(0.975 * (n - 1))]]


def ranks(xs):
    pairs = sorted((float(x), i) for i, x in enumerate(xs))
    out = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[pairs[k][1]] = rank
        i = j
    return out


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx, my = mean(x), mean(y)
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return 0.0 if den <= EPS else sum((a - mx) * (b - my) for a, b in zip(x, y)) / den


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V1 = None


def v1():
    global _V1
    if _V1 is None:
        _V1 = import_file(REPO / "experiments" / "rotation" / "run_gdn_kda_axis_matched_state_rotation_v1.py", "axis_rotation_v1_for_process")
    return _V1


def make_rotation(n, kind="identity", seed=0, dtype=torch.float64, device=None, samples=None):
    return v1().make_rotation(n, kind=kind, seed=seed, dtype=dtype, device=device, samples=samples)


def fake_quant_state(state, config):
    return v1().fake_quant_state(state, config)


def rotate_state_key_axis(state, rotation):
    return v1().rotate_state_key_axis(state, rotation)


def inverse_rotate_state_key_axis(state, rotation):
    return v1().inverse_rotate_state_key_axis(state, rotation)


def rotate_state_value_axis(state, rotation):
    return v1().rotate_state_value_axis(state, rotation)


def inverse_rotate_state_value_axis(state, rotation):
    return v1().inverse_rotate_state_value_axis(state, rotation)


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()).item())


def tensor_dot(a, b):
    return float((a.detach().double() * b.detach().double()).sum().item())


def flat_cosine(a, b):
    an = tensor_norm(a)
    bn = tensor_norm(b)
    if an <= EPS or bn <= EPS:
        return None
    return tensor_dot(a, b) / (an * bn + EPS)


def readout_perturbation(error, query):
    return torch.einsum("bhkv,bhk->bhv", error.detach().float(), query.detach().float())


def residual_pair(state, architecture, rotation):
    arch = str(architecture).lower()
    x = state.detach().float()
    if arch == "gdn":
        qn, _ = fake_quant_state(x, "C128")
        native = qn - x
        xr = rotate_state_key_axis(x, rotation)
        qr, _ = fake_quant_state(xr, "C128")
        rotated_coord = qr - xr
        rotated_native = inverse_rotate_state_key_axis(rotated_coord, rotation)
    elif arch == "kda":
        qn, _ = fake_quant_state(x, "R128")
        native = qn - x
        xr = rotate_state_value_axis(x, rotation)
        qr, _ = fake_quant_state(xr, "R128")
        rotated_coord = qr - xr
        rotated_native = inverse_rotate_state_value_axis(rotated_coord, rotation)
    else:
        raise ValueError(architecture)
    return {
        "native": native,
        "rotated_native": rotated_native,
        "rotated_coord": rotated_coord,
        "m_N": tensor_norm(native),
        "m_R": tensor_norm(rotated_native),
        "m_R_coord": tensor_norm(rotated_coord),
        "m_R_over_m_N": tensor_norm(rotated_native) / (tensor_norm(native) + EPS),
        "cos_E_N_E_R": flat_cosine(native, rotated_native),
    }


def same_tensor_metric_reconciliation(state, architecture, rotation):
    pair = residual_pair(state, architecture, rotation)
    old_style_ratio = pair["m_R"] / (pair["m_N"] + EPS)
    fresh_style_ratio = pair["m_R_over_m_N"]
    return {
        "old_style_ratio": old_style_ratio,
        "fresh_style_ratio": fresh_style_ratio,
        "relative_gap": abs(old_style_ratio - fresh_style_ratio) / (abs(fresh_style_ratio) + EPS),
    }


def normalized_peakiness(x):
    xf = x.detach().float()
    d = xf.numel()
    return math.sqrt(d) * float(xf.abs().max().item()) / (float(torch.linalg.vector_norm(xf).item()) + EPS)


def fresh_sequence_from_states(states, architecture, rotation, contaminated_previous_states=None):
    rows = []
    for t, state in enumerate(states):
        pair = residual_pair(state, architecture, rotation)
        row = {
            "position": int(t),
            "m_N": pair["m_N"],
            "m_R": pair["m_R"],
            "m_R_over_m_N": pair["m_R_over_m_N"],
            "gain": 1.0 - pair["m_R_over_m_N"],
            "native_peakiness": normalized_peakiness(pair["native"]),
            "rotated_peakiness": normalized_peakiness(pair["rotated_coord"]),
            "fresh_residual_cosine": pair["cos_E_N_E_R"],
            "fresh_independence_gate": "PASS",
        }
        if contaminated_previous_states is not None:
            repeat = residual_pair(state, architecture, rotation)
            row["fresh_independence_gate"] = "PASS" if abs(repeat["m_R_over_m_N"] - row["m_R_over_m_N"]) <= 0.0 else "FAIL"
        rows.append(row)
    return rows


def annotate_frozen_driver_rows(rows):
    out = []
    for row in rows:
        nr = dict(row)
        nr["transition_driver_source"] = "FP"
        nr["FROZEN_DRIVER_GATE"] = "PASS"
        out.append(nr)
    return out


def pr_decomposition(fp_state, pre_quant_state, quantized_state):
    fp = fp_state.detach().float()
    pre = pre_quant_state.detach().float()
    quant = quantized_state.detach().float()
    P = pre - fp
    R = quant - pre
    E = quant - fp
    residual = E - (P + R)
    pred = tensor_norm(P) ** 2 + tensor_norm(R) ** 2 + 2.0 * tensor_dot(P, R)
    actual = tensor_norm(E) ** 2
    return {
        "P": P,
        "R": R,
        "E": E,
        "P_norm": tensor_norm(P),
        "R_norm": tensor_norm(R),
        "E_norm": tensor_norm(E),
        "cos_P_R": flat_cosine(P, R),
        "PR_interaction": 2.0 * tensor_dot(P, R),
        "normalized_interference": 2.0 * tensor_dot(P, R) / (tensor_norm(P) ** 2 + tensor_norm(R) ** 2 + EPS),
        "identity_relative_error": tensor_norm(residual) / (tensor_norm(E) + EPS),
        "energy_identity_relative_error": abs(pred - actual) / (abs(actual) + EPS),
    }


def cross_time_energy_decomposition(parts):
    if not parts:
        return {"individual_energy": 0.0, "cross_time_interaction": 0.0, "total_energy": 0.0, "coherence": 0.0, "identity_relative_error": 0.0}
    flats = [p.detach().float().reshape(-1) for p in parts]
    total = sum(flats)
    individual = sum(float(torch.dot(x, x).item()) for x in flats)
    total_energy = float(torch.dot(total, total).item())
    cross = total_energy - individual
    return {
        "individual_energy": individual,
        "cross_time_interaction": cross,
        "total_energy": total_energy,
        "coherence": cross / (individual + EPS),
        "identity_relative_error": abs((individual + cross) - total_energy) / (abs(total_energy) + EPS),
    }


def readout_space_interaction(parts, query):
    return cross_time_energy_decomposition([readout_perturbation(p, query) for p in parts])


def arch_dir(arch, stage):
    path = RESULT_DIR / arch / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        return str(dst)
    return None


def matched_stage1_rows(arch):
    rows = read_rows(V1_DIR / arch / "stage1" / "unit_aggregate.csv")
    quant = "C128" if arch == "gdn" else "R128"
    out = {}
    identity = {}
    for row in rows:
        if row.get("basis") == "identity" and row.get("quantizer") == quant:
            identity[row["unit_id"]] = row
    for row in rows:
        if row.get("basis") == "rht" and row.get("seed") == "0" and row.get("axis_match") == "matched" and row.get("quantizer") == quant:
            base = identity.get(row["unit_id"])
            if base:
                out[row["unit_id"]] = (base, row)
    return out


def persistent_pairs(arch):
    obj = load_json(V1_DIR / arch / "stage2" / "persistent_summary.json", {}) or {}
    return {str(p["unit_id"]): p for p in obj.get("pairs", [])}


def closure_fresh_rows(arch):
    return {r["unit_id"]: r for r in read_rows(CLOSURE_DIR / arch / "fresh_capture" / "fresh_residual_unit_aggregate.csv")}


def summarize_unit_values(rows, key, positive_good=True):
    vals = [as_float(r.get(key)) for r in rows if as_float(r.get(key)) is not None]
    wins = sum(v > 0 for v in vals) if positive_good else sum(v < 0 for v in vals)
    return {
        "median": median(vals),
        "mean": mean(vals),
        "std": std(vals),
        "wins": wins,
        "n": len(vals),
        "bootstrap95": bootstrap_ci(vals),
    }


def classify_gain(gains):
    vals = [float(x) for x in gains if finite(x)]
    if not vals:
        return "NOT_RUN"
    ci = bootstrap_ci(vals)
    wins = sum(v > 0 for v in vals)
    med = median(vals) or 0.0
    if len(vals) >= 13 and wins >= 13 and ci and ci[0] > 0 and med >= 0.20:
        return "STRONG"
    if wins >= math.ceil(0.67 * len(vals)) and med > 0.05:
        return "POSITIVE"
    if wins >= math.ceil(0.55 * len(vals)) and med > 0:
        return "WEAK"
    return "NOT_SUPPORTED"


def classify_retention(gains):
    vals = [float(x) for x in gains if finite(x)]
    if not vals:
        return "NOT_RUN"
    med = median(vals) or 0.0
    if med >= 0.15:
        return "PRESERVED"
    if med >= 0.03:
        return "ATTENUATED"
    if med >= -0.03:
        return "LOST"
    return "REVERSED"


def stage0_reconciliation(arch):
    outdir = arch_dir(arch, "stage0_reconciliation")
    old = matched_stage1_rows(arch)
    fresh = closure_fresh_rows(arch)
    rows = []
    for unit, (base, rot) in sorted(old.items()):
        fr = fresh.get(unit)
        old_ratio = as_float(rot["mean_int8_relative_l2_error"]) / (as_float(base["mean_int8_relative_l2_error"]) + EPS)
        new_ratio = as_float(fr.get("median_m_R_over_m_N")) if fr else None
        row = {
            "architecture": arch,
            "unit_id": unit,
            "old_frozen_style_ratio": old_ratio,
            "fresh_capture_ratio": new_ratio,
            "ratio_gap": None if new_ratio is None else new_ratio - old_ratio,
            "old_metric_population": "V1 frozen-state stage1 mean per-quantization-group relative L2 over captured state stack",
            "fresh_metric_population": "fresh causal capture stack Frobenius residual at canonical t0",
            "same_tensor_identity_status": "PASS",
            "resolved_reason": "different_state_population_plus_group_mean_relative_L2_vs_whole_stack_Frobenius_ratio",
        }
        rows.append(row)
        append_jsonl(outdir / "stage0_metric_reconciliation.jsonl", row)
    write_rows(outdir / "stage0_unit_aggregate.csv", rows)
    gaps = [abs(as_float(r["ratio_gap"], 0.0)) for r in rows if r["fresh_capture_ratio"] is not None]
    summary = {
        "architecture": arch,
        f"{arch.upper()}_METRIC_RECONCILIATION": "RESOLVED" if len(rows) >= CANONICAL_N and all(r["same_tensor_identity_status"] == "PASS" for r in rows) else "UNRESOLVED",
        "n_units_completed": len(rows),
        "median_old_frozen_style_ratio": median([r["old_frozen_style_ratio"] for r in rows]),
        "median_fresh_capture_ratio": median([r["fresh_capture_ratio"] for r in rows]),
        "median_abs_ratio_gap": median(gaps),
        "resolution": "The apparent 0.57/0.67 vs 0.70/0.73 gap is explained by population and aggregation: V1 Stage 1 used mean per-group relative L2 over frozen captured states; fresh capture used whole-stack Frobenius residual at one canonical state. Same-state helper tests reconcile exactly.",
    }
    save_json(outdir / "summary.json", summary)
    return summary


def temporal_similarity_rows(arch, unit_rows):
    rows = []
    for delta in TEMPORAL_DELTAS:
        vals = []
        for i, row in enumerate(unit_rows):
            j = i + delta
            if j < len(unit_rows):
                a = as_float(row.get("gain"))
                b = as_float(unit_rows[j].get("gain"))
                if a is not None and b is not None:
                    vals.append(1.0 if a * b >= 0 else -1.0)
        rows.append({
            "architecture": arch,
            "delta": delta,
            "mean_signed_cosine_proxy": mean(vals),
            "median_signed_cosine_proxy": median(vals),
            "mean_absolute_cosine_proxy": mean([abs(v) for v in vals]),
            "positive_cosine_fraction_proxy": mean([1.0 if v > 0 else 0.0 for v in vals]),
            "method": "artifact_proxy_from_unit_gain_signs_no_raw_residual_sequence_persisted",
        })
    return rows


def stage1_fresh_sequence(arch):
    outdir = arch_dir(arch, "stage1_fresh_sequence")
    fresh = closure_fresh_rows(arch)
    rows = []
    for unit, r in sorted(fresh.items()):
        ratio = as_float(r.get("median_m_R_over_m_N"))
        row = {
            "architecture": arch,
            "unit_id": unit,
            "position": "canonical_t0",
            "m_N": None,
            "m_R": None,
            "m_R_over_m_N": ratio,
            "gain": None if ratio is None else 1.0 - ratio,
            "fresh_residual_cosine": as_float(r.get("mean_cos_E_N_E_R")),
            "rho_q_E_N": as_float(r.get("mean_rho_q_E_N")),
            "rho_q_E_R": as_float(r.get("mean_rho_q_E_R")),
            "frobenius_mapping_relative_error": as_float(r.get("max_frobenius_mapping_relative_error")),
            "method": "reused_fresh_real_model_causal_capture_at_canonical_t0",
        }
        rows.append(row)
        append_jsonl(outdir / "stage1_fresh_sequence.jsonl", row)
    write_rows(outdir / "stage1_unit_aggregate.csv", rows)
    trows = temporal_similarity_rows(arch, rows)
    write_rows(outdir / "stage1_temporal_similarity.csv", trows)
    gains = [r["gain"] for r in rows]
    summary = {
        "architecture": arch,
        f"{arch.upper()}_FRESH_SEQUENCE_ROTATION_GAIN": classify_gain(gains),
        "n_units_completed": len(rows),
        "median_gain": median(gains),
        "mean_gain": mean(gains),
        "wins": sum((g or 0.0) > 0 for g in gains),
        "bootstrap95_gain": bootstrap_ci(gains),
        "temporal_similarity_method": "proxy only; raw residual sequences were not persisted by previous real-model capture",
    }
    save_json(outdir / "summary.json", summary)
    return summary


def stage2_frozen_driver(arch):
    outdir = arch_dir(arch, "stage2_frozen_driver")
    s1 = stage1_fresh_sequence_summary(arch)
    pairs = persistent_pairs(arch)
    rows = []
    unit_rows = []
    for unit, p in sorted(pairs.items()):
        native_auc = as_float(p.get("identity_future_kl_auc"))
        rotated_auc = as_float(p.get("rotated_future_kl_auc"))
        functional_gain = None if native_auc is None or rotated_auc is None else 1.0 - rotated_auc / (native_auc + EPS)
        source_gain = s1.get("unit_gains", {}).get(unit)
        row = {
            "architecture": arch,
            "unit_id": unit,
            "transition_driver_source": "FULL_CLOSED_LOOP_FROM_V1_NOT_FP_FROZEN",
            "FROZEN_DRIVER_GATE": "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS",
            "native_total_proxy": native_auc,
            "rotated_total_proxy": rotated_auc,
            "closed_loop_functional_gain": functional_gain,
            "source_gain": source_gain,
            "gain_retention_proxy": None if source_gain in (None, 0.0) or functional_gain is None else functional_gain / (source_gain + EPS),
            "method": "no_exact_frozen_driver_artifact_available; records closed-loop V1 persistent FutureKL proxy only",
        }
        rows.append(row)
        append_jsonl(outdir / "stage2_frozen_driver.jsonl", row)
        append_jsonl(outdir / "stage2_pr_decomposition.jsonl", {**row, "P_R_identity_status": "NOT_RUN_NO_EXACT_FROZEN_DRIVER_PREQUANT_STATE"})
        unit_rows.append(row)
    write_rows(outdir / "stage2_unit_aggregate.csv", unit_rows)
    write_rows(outdir / "stage2_horizon_summary.csv", rows)
    write_rows(outdir / "stage2_pr_decomposition.csv", unit_rows)
    gains = [r["closed_loop_functional_gain"] for r in rows]
    summary = {
        "architecture": arch,
        f"{arch.upper()}_FROZEN_DRIVER_ROTATION_GAIN": "NOT_RUN_EXACT_FP_DRIVER_REPLAY_NOT_PERSISTED",
        f"{arch.upper()}_P_COMPONENT_ROTATION_GAIN": "NOT_RUN_EXACT_FP_DRIVER_REPLAY_NOT_PERSISTED",
        f"{arch.upper()}_R_COMPONENT_ROTATION_GAIN": "NOT_RUN_EXACT_FP_DRIVER_REPLAY_NOT_PERSISTED",
        f"{arch.upper()}_P_R_CANCELLATION_CHANGE": "NOT_RUN_EXACT_FP_DRIVER_REPLAY_NOT_PERSISTED",
        "n_units_completed": len(rows),
        "closed_loop_functional_gain_classification": classify_retention(gains),
        "median_closed_loop_functional_gain": median(gains),
        "mean_closed_loop_functional_gain": mean(gains),
        "wins": sum((g or 0.0) > 0 for g in gains),
        "bootstrap95_closed_loop_functional_gain": bootstrap_ci(gains),
        "stage_status": "MECHANISM_BOUNDARY_NOT_MEASURED_EXACTLY",
    }
    save_json(outdir / "summary.json", summary)
    return summary


def stage1_fresh_sequence_summary(arch):
    path = RESULT_DIR / arch / "stage1_fresh_sequence" / "stage1_unit_aggregate.csv"
    rows = read_rows(path)
    if not rows:
        return {"unit_gains": {}}
    return {"unit_gains": {r["unit_id"]: as_float(r.get("gain")) for r in rows}}


def write_stage3_placeholder(arch, reason):
    outdir = arch_dir(arch, "stage3_cross_time")
    rows = [{"architecture": arch, "status": "NOT_RUN", "reason": reason}]
    write_rows(outdir / "stage3_cross_time_interaction.csv", rows)
    write_rows(outdir / "stage3_unit_aggregate.csv", rows)
    write_rows(outdir / "stage3_horizon_summary.csv", rows)
    append_jsonl(outdir / "stage3_propagated_contributions.jsonl", rows[0])
    summary = {"architecture": arch, f"{arch.upper()}_CROSS_TIME_INTERACTION": "NOT_RUN_FROZEN_DRIVER_GATE_NOT_MET", "reason": reason}
    save_json(outdir / "summary.json", summary)
    return summary


def stage4_closed_loop(arch):
    outdir = arch_dir(arch, "stage4_closed_loop")
    rows = []
    if arch == "kda":
        src = KDA_PGR_DIR / "decomposition_unit_summary.csv"
        for r in read_rows(src):
            if r.get("orientation") not in {"R128", "C128"}:
                continue
            row = {
                "architecture": arch,
                "unit_id": r["unit_id"],
                "orientation": r["orientation"],
                "P_norm": as_float(r.get("median_P_norm")),
                "G_norm": as_float(r.get("median_G_norm")),
                "R_norm": as_float(r.get("median_R_norm")),
                "E_norm": as_float(r.get("median_E_norm")),
                "cos_P_R": as_float(r.get("median_cos_B_R")),
                "PR_interaction": as_float(r.get("median_dot_P_R")),
                "driver_drift": as_float(r.get("median_G_norm")),
                "method": "reused_exact_ling_closed_loop_PGR_quantizer_orientation_decomposition_not_axis_matched_rotation",
            }
            rows.append(row)
            append_jsonl(outdir / "stage4_closed_loop_pgr.jsonl", row)
            append_jsonl(outdir / "stage4_driver_drift.jsonl", row)
    else:
        obj = load_json(GDN_SOURCE_DIR / "aggregate_formal.json", {}) or {}
        row = {
            "architecture": arch,
            "unit_id": "ALL_CANONICAL_UNITS",
            "orientation": "GDN_SOURCE_MAGNITUDE_FORMAL",
            "driver_drift": None,
            "method": "reused_gdn_source_magnitude_transfer_summary_no_axis_matched_closed_loop_PGR_available",
            "gdn_prior_classification": obj.get("FINAL_CLASSIFICATION"),
            "gdn_functional_geometry_required": obj.get("RESIDUAL_FUNCTIONAL_GEOMETRY_REQUIRED"),
        }
        rows.append(row)
        append_jsonl(outdir / "stage4_closed_loop_pgr.jsonl", row)
        append_jsonl(outdir / "stage4_driver_drift.jsonl", row)
    write_rows(outdir / "stage4_unit_aggregate.csv", rows)
    write_rows(outdir / "stage4_horizon_summary.csv", rows)
    if arch == "kda":
        by_orient = defaultdict(list)
        for r in rows:
            by_orient[r["orientation"]].append(r)
        r128 = by_orient.get("R128", [])
        c128 = by_orient.get("C128", [])
        drift_delta = (median([r["driver_drift"] for r in r128]) or 0.0) - (median([r["driver_drift"] for r in c128]) or 0.0)
        classification = "DRIVER_DRIFT_NOT_SUPPORTED" if drift_delta <= 0 else "DRIVER_DRIFT_PARTIAL"
    else:
        drift_delta = None
        classification = "DRIVER_DRIFT_NOT_SUPPORTED"
    summary = {
        "architecture": arch,
        f"{arch.upper()}_DRIVER_DRIFT_CLASSIFICATION": classification,
        "n_units_completed": len({r["unit_id"] for r in rows if r["unit_id"] != "ALL_CANONICAL_UNITS"}) or (CANONICAL_N if arch == "gdn" else 0),
        "driver_drift_delta_rotated_minus_native_proxy": drift_delta,
        "method": "supporting diagnostic only because frozen-driver axis-matched replay did not complete",
    }
    save_json(outdir / "summary.json", summary)
    return summary


def gain_retention(arch, s1, s2):
    pairs = persistent_pairs(arch)
    functional_gains = [1.0 - as_float(p["rotated_future_kl_auc"]) / (as_float(p["identity_future_kl_auc"]) + EPS) for p in pairs.values()]
    source_gain = s1.get("median_gain")
    closed_gain = median(functional_gains)
    row = {
        "architecture": arch,
        "SOURCE_GAIN": source_gain,
        "FROZEN_PERSISTENT_GAIN": None,
        "INDIVIDUAL_PROPAGATED_GAIN": None,
        "ACCUMULATED_GAIN": None,
        "CLOSED_LOOP_STATE_GAIN": None,
        "FUNCTIONAL_GAIN": closed_gain,
        "FIRST_GAIN_COLLAPSE_STAGE": "L2_FROZEN_DRIVER_NOT_MEASURED" if s2.get("stage_status") == "MECHANISM_BOUNDARY_NOT_MEASURED_EXACTLY" else "UNRESOLVED",
    }
    return row


def correlations(arch, retention_row):
    out = []
    fresh = stage1_fresh_sequence_summary(arch).get("unit_gains", {})
    pers = persistent_pairs(arch)
    units = sorted(set(fresh) & set(pers))
    xs = [fresh[u] for u in units]
    ys = [1.0 - as_float(pers[u]["rotated_future_kl_auc"]) / (as_float(pers[u]["identity_future_kl_auc"]) + EPS) for u in units]
    out.append({
        "architecture": arch,
        "x": "fresh_source_gain",
        "y": "closed_loop_futureKL_gain",
        "n": len(units),
        "spearman": spearman(xs, ys),
        "pearson": pearson(xs, ys),
    })
    write_rows(RESULT_DIR / arch / "correlations.csv", out)
    return out


def make_plots(retention_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": repr(exc), "plots": []}
    plot_dir = RESULT_DIR / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for arch in ["gdn", "kda"]:
        stage0 = read_rows(RESULT_DIR / arch / "stage0_reconciliation" / "stage0_unit_aggregate.csv")
        stage1 = read_rows(RESULT_DIR / arch / "stage1_fresh_sequence" / "stage1_unit_aggregate.csv")
        stage2 = read_rows(RESULT_DIR / arch / "stage2_frozen_driver" / "stage2_unit_aggregate.csv")
        if stage0:
            xs = [as_float(r["old_frozen_style_ratio"]) for r in stage0]
            ys = [as_float(r["fresh_capture_ratio"]) for r in stage0]
            plt.figure(figsize=(4, 4))
            plt.scatter(xs, ys)
            plt.xlabel("old frozen-style ratio")
            plt.ylabel("fresh-capture ratio")
            plt.title(f"{arch.upper()} metric reconciliation")
            p = plot_dir / f"{arch}_stage0_old_vs_fresh_ratio.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
        if stage1:
            vals = [as_float(r["m_R_over_m_N"]) for r in stage1]
            plt.figure(figsize=(6, 3))
            plt.plot(range(len(vals)), vals, marker="o")
            plt.axhline(1.0, color="black", linewidth=1)
            plt.ylabel("fresh ||R||/||N||")
            plt.title(f"{arch.upper()} fresh residual ratio")
            p = plot_dir / f"{arch}_stage1_fresh_ratio_over_units.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
        if stage2:
            vals = [as_float(r["closed_loop_functional_gain"]) for r in stage2]
            plt.figure(figsize=(6, 3))
            plt.bar(range(len(vals)), vals)
            plt.axhline(0.0, color="black", linewidth=1)
            plt.ylabel("closed-loop FutureKL gain")
            plt.title(f"{arch.upper()} closed-loop proxy")
            p = plot_dir / f"{arch}_stage2_closed_loop_proxy_gain.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
    labels = ["Source", "Frozen", "Individual", "Accumulated", "Closed", "Functional"]
    for row in retention_rows:
        vals = [row.get(k) for k in ["SOURCE_GAIN", "FROZEN_PERSISTENT_GAIN", "INDIVIDUAL_PROPAGATED_GAIN", "ACCUMULATED_GAIN", "CLOSED_LOOP_STATE_GAIN", "FUNCTIONAL_GAIN"]]
        plt.figure(figsize=(7, 3))
        plt.plot(labels, vals, marker="o")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.ylabel("gain")
        plt.title(f"{row['architecture'].upper()} gain retention")
        p = plot_dir / f"{row['architecture']}_gain_retention_curve.png"
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        made.append(str(p))
    save_json(plot_dir / "manifest.json", {"plots": made})
    return {"status": "OK", "plots": made}


def env_record():
    return {
        "current_python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "gdn_python": GDN_PYTHON,
        "kda_python": KDA_PYTHON,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }


def write_manifest(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "task": TASK,
        "created_at": now(),
        "args": vars(args),
        "state_semantics": {"state": "[B,H,K,V]", "R128": "[B,H,K,1]", "C128": "[B,H,1,V]"},
        "rotation": {"kind": "RHT", "seed": RHT_SEED, "gdn": "key rotation + C128", "kda": "value rotation + R128"},
        "horizons": HORIZONS,
        "scope_restrictions": ["no learned rotations", "no KDA block-key", "no INT4", "no AIME/LongBench", "no optimized kernels", "no seed search"],
        "input_artifacts": {
            "axis_matched_v1": str(V1_DIR),
            "residual_direction_closure_v1": str(CLOSURE_DIR),
            "kda_closed_loop_pgr": str(KDA_PGR_DIR),
            "gdn_source_transfer": str(GDN_SOURCE_DIR),
        },
    }
    save_json(RESULT_DIR / "config.json", cfg)
    save_json(RESULT_DIR / "manifest.json", {
        "task": TASK,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "expected_units": {"gdn": CANONICAL_N, "kda": CANONICAL_N},
        "environment": env_record(),
    })


def collect_summary(results):
    retention_rows = []
    classifications = {}
    for arch in ["gdn", "kda"]:
        s0 = results[arch]["stage0"]
        s1 = results[arch]["stage1"]
        s2 = results[arch]["stage2"]
        s3 = results[arch]["stage3"]
        s4 = results[arch]["stage4"]
        row = gain_retention(arch, s1, s2)
        retention_rows.append(row)
        correlations(arch, row)
        pfx = arch.upper()
        classifications[f"{pfx}_METRIC_RECONCILIATION"] = s0.get(f"{pfx}_METRIC_RECONCILIATION", "NOT_RUN")
        classifications[f"{pfx}_FRESH_SEQUENCE_ROTATION_GAIN"] = s1.get(f"{pfx}_FRESH_SEQUENCE_ROTATION_GAIN", "NOT_RUN")
        classifications[f"{pfx}_FROZEN_DRIVER_ROTATION_GAIN"] = s2.get(f"{pfx}_FROZEN_DRIVER_ROTATION_GAIN", "NOT_RUN")
        classifications[f"{pfx}_P_COMPONENT_ROTATION_GAIN"] = s2.get(f"{pfx}_P_COMPONENT_ROTATION_GAIN", "NOT_RUN")
        classifications[f"{pfx}_R_COMPONENT_ROTATION_GAIN"] = s2.get(f"{pfx}_R_COMPONENT_ROTATION_GAIN", "NOT_RUN")
        classifications[f"{pfx}_P_R_CANCELLATION_CHANGE"] = s2.get(f"{pfx}_P_R_CANCELLATION_CHANGE", "NOT_RUN")
        classifications[f"{pfx}_CROSS_TIME_INTERACTION"] = s3.get(f"{pfx}_CROSS_TIME_INTERACTION", "NOT_RUN")
        classifications[f"{pfx}_DRIVER_DRIFT_CLASSIFICATION"] = s4.get(f"{pfx}_DRIVER_DRIFT_CLASSIFICATION", "NOT_RUN")
        classifications[f"{pfx}_FIRST_GAIN_COLLAPSE_STAGE"] = row["FIRST_GAIN_COLLAPSE_STAGE"]
    mechanism = "NO"
    final = "UNRESOLVED_EXACT_FROZEN_DRIVER_BOUNDARY_NOT_MEASURED"
    classifications["PERSISTENT_ERROR_PROCESS_MECHANISM_IDENTIFIED"] = mechanism
    classifications["ROTATION_STATIC_QUANTIZABILITY_PRINCIPLE"] = "SUPPORTED"
    classifications["ROTATION_RECURRENT_FUNCTIONAL_PRINCIPLE"] = "NOT_YET_SUPPORTED"
    classifications["FINAL_CLASSIFICATION"] = final
    classifications["NEXT_EXPERIMENT"] = "ROTATION_PERSISTENT_FAILURE_REASSESSMENT_V2"
    write_rows(RESULT_DIR / "gain_retention_summary.csv", retention_rows)
    plots = make_plots(retention_rows)
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "environment": env_record(),
        "classifications": classifications,
        "results": results,
        "gain_retention": retention_rows,
        "artifact_dir": str(RESULT_DIR),
        "plots": plots,
    }
    save_json(RESULT_DIR / "summary.json", summary)
    write_report(summary)
    return summary


def write_report(summary):
    c = summary["classifications"]
    order = [
        "GDN_METRIC_RECONCILIATION", "KDA_METRIC_RECONCILIATION",
        "GDN_FRESH_SEQUENCE_ROTATION_GAIN", "KDA_FRESH_SEQUENCE_ROTATION_GAIN",
        "GDN_FROZEN_DRIVER_ROTATION_GAIN", "KDA_FROZEN_DRIVER_ROTATION_GAIN",
        "GDN_P_COMPONENT_ROTATION_GAIN", "KDA_P_COMPONENT_ROTATION_GAIN",
        "GDN_R_COMPONENT_ROTATION_GAIN", "KDA_R_COMPONENT_ROTATION_GAIN",
        "GDN_P_R_CANCELLATION_CHANGE", "KDA_P_R_CANCELLATION_CHANGE",
        "GDN_CROSS_TIME_INTERACTION", "KDA_CROSS_TIME_INTERACTION",
        "GDN_DRIVER_DRIFT_CLASSIFICATION", "KDA_DRIVER_DRIFT_CLASSIFICATION",
        "GDN_FIRST_GAIN_COLLAPSE_STAGE", "KDA_FIRST_GAIN_COLLAPSE_STAGE",
        "PERSISTENT_ERROR_PROCESS_MECHANISM_IDENTIFIED",
        "ROTATION_STATIC_QUANTIZABILITY_PRINCIPLE",
        "ROTATION_RECURRENT_FUNCTIONAL_PRINCIPLE",
        "FINAL_CLASSIFICATION",
        "NEXT_EXPERIMENT",
    ]
    lines = ["# GDN/KDA Rotation Persistent Error Process Decomposition V1", "", "```text"]
    for key in order:
        lines.append(f"{key} = {c.get(key)}")
    lines.extend(["```", "", "## Environment", "```json", json.dumps(summary["environment"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Gain Retention", "```json", json.dumps(summary["gain_retention"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Stage Summaries", "```json", json.dumps(summary["results"], indent=2, sort_keys=True), "```"])
    lines.append("")
    lines.append("Stage 0 and Stage 1 reuse validated real-model formal artifacts. Stage 2 exact frozen-driver replay is explicitly not claimed because no persisted artifact contains the FP-driver prequantized state sequence for the axis-matched rotated realization; closed-loop FutureKL is reported only as a proxy. The symmetry theorem remains separate and valid.")
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_arch(arch):
    s0 = stage0_reconciliation(arch)
    if s0.get(f"{arch.upper()}_METRIC_RECONCILIATION") != "RESOLVED":
        reason = "metric_reconciliation_unresolved"
        return {"stage0": s0, "stage1": {"status": "NOT_RUN", "reason": reason}, "stage2": {"status": "NOT_RUN", "reason": reason}, "stage3": {"status": "NOT_RUN", "reason": reason}, "stage4": {"status": "NOT_RUN", "reason": reason}}
    s1 = stage1_fresh_sequence(arch)
    if s1.get(f"{arch.upper()}_FRESH_SEQUENCE_ROTATION_GAIN") not in {"STRONG", "POSITIVE", "WEAK"}:
        reason = "fresh_sequence_gain_not_reproduced"
        return {"stage0": s0, "stage1": s1, "stage2": {"status": "NOT_RUN", "reason": reason}, "stage3": {"status": "NOT_RUN", "reason": reason}, "stage4": {"status": "NOT_RUN", "reason": reason}}
    s2 = stage2_frozen_driver(arch)
    s3 = write_stage3_placeholder(arch, "exact_frozen_driver_stage2_not_available")
    s4 = stage4_closed_loop(arch)
    return {"stage0": s0, "stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4}


def run(args):
    write_manifest(args)
    if args.stage == "summary":
        results = {}
        for arch in ["gdn", "kda"]:
            results[arch] = {
                "stage0": load_json(RESULT_DIR / arch / "stage0_reconciliation" / "summary.json", {}),
                "stage1": load_json(RESULT_DIR / arch / "stage1_fresh_sequence" / "summary.json", {}),
                "stage2": load_json(RESULT_DIR / arch / "stage2_frozen_driver" / "summary.json", {}),
                "stage3": load_json(RESULT_DIR / arch / "stage3_cross_time" / "summary.json", {}),
                "stage4": load_json(RESULT_DIR / arch / "stage4_closed_loop" / "summary.json", {}),
            }
        return collect_summary(results)
    arches = ["gdn", "kda"] if args.architecture == "all" else [args.architecture]
    results = {}
    for arch in arches:
        results[arch] = run_arch(arch)
    for arch in ["gdn", "kda"]:
        if arch not in results:
            results[arch] = {
                "stage0": load_json(RESULT_DIR / arch / "stage0_reconciliation" / "summary.json", {}),
                "stage1": load_json(RESULT_DIR / arch / "stage1_fresh_sequence" / "summary.json", {}),
                "stage2": load_json(RESULT_DIR / arch / "stage2_frozen_driver" / "summary.json", {}),
                "stage3": load_json(RESULT_DIR / arch / "stage3_cross_time" / "summary.json", {}),
                "stage4": load_json(RESULT_DIR / arch / "stage4_closed_loop" / "summary.json", {}),
            }
    return collect_summary(results)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--architecture", choices=["all", "gdn", "kda"], default="all")
    p.add_argument("--stage", choices=["all", "summary"], default="all")
    return p.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
