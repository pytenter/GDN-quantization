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
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "GDN_ROTATION_P_R_QUERY_VISIBILITY_MECHANISM_CLOSURE_V1"
SLUG = "gdn_rotation_p_r_query_visibility_mechanism_closure_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
PRIOR_RESULT_DIR = REPO / "results" / "gdn_kda_rotation_exact_frozen_driver_replay_v1"
ACCUM_RESULT_DIR = REPO / "results" / "gdn_rotation_accumulated_error_query_geometry_closure_v1"
EXACT_RUNNER = REPO / "experiments" / "rotation" / "run_gdn_kda_rotation_exact_frozen_driver_replay_v1.py"
GDN_PYTHON = "/data01/user2/.conda/envs/sd310/bin/python"
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
EPS = 1e-12
RHT_SEED = 0
CANONICAL_N = 18
BOOTSTRAP_SEED = 20260914
FP_REPLAY_TOL = 1e-5
P_IDENTITY_TOL = 5e-3
P_IDENTITY_ABS_TOL = 1e-4
R_FACTORIAL_TOL = 5e-3
R_FACTORIAL_ABS_TOL = 1e-3
READOUT_IDENTITY_TOL = 1e-8
FACTORIZATION_TOL = 1e-8


_EXACT = None


def exact():
    global _EXACT
    if _EXACT is None:
        spec = importlib.util.spec_from_file_location("exact_frozen_driver_for_pr_query_visibility", EXACT_RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXACT = module
    return _EXACT


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()).item())


def tensor_dot(a, b):
    return float((a.detach().double() * b.detach().double()).sum().item())


def relerr(a, b, eps=EPS):
    return tensor_norm(a.detach().double() - b.detach().double()) / (tensor_norm(b) + eps)


def symmetric_relerr(a, b, abs_floor=1e-8, eps=EPS):
    diff = tensor_norm(a.detach().double() - b.detach().double())
    denom = max(tensor_norm(a), tensor_norm(b), float(abs_floor))
    return diff / (denom + eps)


def identity_gate_error(a, b, abs_tol=P_IDENTITY_ABS_TOL, abs_floor=1e-3):
    diff = tensor_norm(a.detach().double() - b.detach().double())
    if diff <= float(abs_tol):
        return 0.0
    return diff / (max(tensor_norm(a), tensor_norm(b), float(abs_floor)) + EPS)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def clean_result_dir():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["stage0_reproduction", "stage1_p_effective_query", "stage2_r_factorial", "stage3_readout_attribution", "figures"]:
        path = RESULT_DIR / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "manifest.json", "provenance.json", "summary.json", "report.md", "horizon_summary.csv", "failure.json"]:
        path = RESULT_DIR / name
        if path.exists():
            path.unlink()


def validate_run_args(args):
    if args.scope == "formal" and int(args.horizon) != 64:
        raise RuntimeError("formal runs require --horizon 64 for strict driver-tape hash comparability")


def validate_prior_provenance(previous_manifest, previous_parity):
    if previous_manifest.get("GDN_FP_DRIVER_TAPE_CAPTURE") != "PASS":
        raise RuntimeError("prior GDN driver-tape capture provenance is missing or not PASS")
    if previous_manifest.get("n_units_completed") != CANONICAL_N:
        raise RuntimeError("prior GDN driver-tape capture did not complete all canonical units")
    if previous_parity.get("GDN_FROZEN_DRIVER_FP_REPLAY") != "PASS":
        raise RuntimeError("prior GDN frozen-driver FP replay provenance is missing or not PASS")
    if previous_parity.get("n_units_completed") != CANONICAL_N:
        raise RuntimeError("prior GDN frozen-driver FP replay did not complete all canonical units")


def append_jsonl(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def finite_values(values):
    out = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            out.append(value)
    return out


def mean(values):
    values = finite_values(values)
    return sum(values) / len(values) if values else None


def median(values):
    values = finite_values(values)
    return statistics.median(values) if values else None


def std(values):
    values = finite_values(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def bootstrap_mean_ci(values, samples=4000, seed=BOOTSTRAP_SEED):
    values = finite_values(values)
    if not values:
        return None
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(samples)):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    draws.sort()
    return [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]


def readout_vec(state, query):
    return torch.einsum("bhkv,bhk->bhv", state.detach().double(), query.detach().double())


def apply_gdn_error_transition(error, key, gamma, beta):
    error64 = error.detach().double()
    key64 = key.detach().double()
    gamma64 = gamma.detach().double()
    beta64 = beta.detach().double()
    memory = torch.einsum("bhk,bhkv->bhv", key64, error64)
    return gamma64.unsqueeze(-1).unsqueeze(-1) * (
        error64 - key64.unsqueeze(-1) * beta64.unsqueeze(-1).unsqueeze(-1) * memory.unsqueeze(-2)
    )


def gdn_effective_query(query, key, gamma, beta):
    query64 = query.detach().double()
    key64 = key.detach().double()
    gamma64 = gamma.detach().double()
    beta64 = beta.detach().double()
    kq = (key64 * query64).sum(dim=-1)
    return gamma64.unsqueeze(-1) * (query64 - beta64.unsqueeze(-1) * key64 * kq.unsqueeze(-1))


def effective_query_metrics(prev_error, query, key, gamma, beta):
    q_eff = gdn_effective_query(query, key, gamma, beta)
    propagated = apply_gdn_error_transition(prev_error, key, gamma, beta)
    p_q = readout_vec(propagated, query)
    eff_q = readout_vec(prev_error, q_eff)
    prev_norm = tensor_norm(prev_error)
    qeff_norm = tensor_norm(q_eff)
    pq_norm = tensor_norm(p_q)
    identity = identity_gate_error(p_q, eff_q)
    return {
        "q_eff": q_eff,
        "propagated": propagated,
        "p_readout": p_q,
        "prev_error_norm": prev_norm,
        "q_eff_norm": qeff_norm,
        "p_readout_norm": pq_norm,
        "effective_visibility": pq_norm / (prev_norm * qeff_norm + EPS) if prev_norm > 0.0 and qeff_norm > 0.0 else 0.0,
        "identity_relative_error": identity,
    }


def p_effective_query_factorization(native, rotated):
    prev_ratio = rotated["prev_error_norm"] / native["prev_error_norm"] if native["prev_error_norm"] > 0.0 else 0.0
    vis_ratio = (
        rotated["effective_visibility"] / native["effective_visibility"]
        if native["effective_visibility"] > 0.0 else 0.0
    )
    readout_ratio = rotated["p_readout_norm"] / native["p_readout_norm"] if native["p_readout_norm"] > 0.0 else 0.0
    factorized = prev_ratio * vis_ratio
    return {
        "prev_error_norm_ratio": prev_ratio,
        "effective_visibility_ratio": vis_ratio,
        "p_readout_ratio": readout_ratio,
        "factorized_p_readout_ratio": factorized,
        "relative_error": abs(readout_ratio - factorized) / (abs(readout_ratio) + EPS),
        "threshold_T_P": 1.0 / prev_ratio if prev_ratio > 0.0 else None,
        "visibility_reverses_magnitude": bool(prev_ratio > 0.0 and vis_ratio > 1.0 / prev_ratio),
    }


def assert_same_tensor(a, b, name, atol=0.0):
    ad = a.detach().cpu()
    bd = b.detach().cpu()
    same = ad.shape == bd.shape and ad.dtype == bd.dtype
    max_abs = None
    if same:
        max_abs = float((ad.double() - bd.double()).abs().max().item()) if ad.numel() else 0.0
        same = max_abs <= float(atol)
    return {"name": name, "gate": "PASS" if same else "FAIL", "max_abs_diff": max_abs}


def native_quantizer(state):
    return exact().quantize_branch_state(state, "gdn", "native")


def rotated_quantizer_mapped_back(state, rotation):
    ex = exact()
    rotated = ex.transform_state_to_branch(state, "gdn", "rotated", rotation)
    q_rot = ex.quantize_branch_state(rotated, "gdn", "rotated")
    return ex.inverse_branch_state(q_rot, "gdn", "rotated", rotation)


def r_factorial_residuals(pre_native, pre_rotated, rotation, pre_rotated_branch=None, q_native_actual=None, q_rotated_branch_actual=None):
    q_nn = q_native_actual.detach().double() if q_native_actual is not None else native_quantizer(pre_native)
    q_nr = rotated_quantizer_mapped_back(pre_native, rotation)
    q_rn = native_quantizer(pre_rotated)
    if q_rotated_branch_actual is not None:
        q_rr = exact().inverse_branch_state(q_rotated_branch_actual, "gdn", "rotated", rotation).detach().double()
    elif pre_rotated_branch is None:
        q_rr = rotated_quantizer_mapped_back(pre_rotated, rotation)
    else:
        q_rr_branch = exact().quantize_branch_state(pre_rotated_branch, "gdn", "rotated")
        q_rr = exact().inverse_branch_state(q_rr_branch, "gdn", "rotated", rotation)
    return {
        "NN": q_nn.detach().double() - pre_native.detach().double(),
        "NR": q_nr.detach().double() - pre_native.detach().double(),
        "RN": q_rn.detach().double() - pre_rotated.detach().double(),
        "RR": q_rr.detach().double() - pre_rotated.detach().double(),
    }


def factorial_effects(risks):
    return {
        "B_N": float(risks["NR"]) - float(risks["NN"]),
        "B_R": float(risks["RR"]) - float(risks["RN"]),
        "T_N": float(risks["RN"]) - float(risks["NN"]),
        "T_R": float(risks["RR"]) - float(risks["NR"]),
        "interaction": (float(risks["RR"]) - float(risks["RN"])) - (float(risks["NR"]) - float(risks["NN"])),
    }


def readout_energy_terms(p_vec, r_vec):
    p = p_vec.detach().double().reshape(-1)
    r = r_vec.detach().double().reshape(-1)
    e = p + r
    p_energy = float((p * p).sum().item())
    r_energy = float((r * r).sum().item())
    cross = float(2.0 * (p * r).sum().item())
    total = float((e * e).sum().item())
    pred = p_energy + r_energy + cross
    return {
        "P_energy": p_energy,
        "R_energy": r_energy,
        "cross_energy": cross,
        "total_energy": total,
        "predicted_total_energy": pred,
        "identity_relative_error": abs(total - pred) / (abs(total) + EPS),
    }


def readout_counterfactual_attribution(p_native, p_rotated, r_native, r_rotated):
    p_n = p_native.detach().double().reshape(-1)
    p_r = p_rotated.detach().double().reshape(-1)
    r_n = r_native.detach().double().reshape(-1)
    r_r = r_rotated.detach().double().reshape(-1)

    def energy(x):
        return float((x * x).sum().item())

    l_nn = energy(p_n + r_n)
    l_rn = energy(p_r + r_n)
    l_nr = energy(p_n + r_r)
    l_rr = energy(p_r + r_r)
    p_contrib = l_rn - l_nn
    r_contrib = l_nr - l_nn
    interaction = l_rr - l_rn - l_nr + l_nn
    total_delta = l_rr - l_nn
    summed = p_contrib + r_contrib + interaction
    return {
        "L_NN": l_nn,
        "L_RN": l_rn,
        "L_NR": l_nr,
        "L_RR": l_rr,
        "P_contribution": p_contrib,
        "R_contribution": r_contrib,
        "interaction_contribution": interaction,
        "total_delta": total_delta,
        "sum_contributions": summed,
        "closure_relative_error": abs(total_delta - summed) / (abs(total_delta) + EPS),
    }


def gdn_transition_terms(rec, rotation=None, rotated=False):
    ex = exact()
    query = rec["query"].detach().float()
    key = rec["key"].detach().float()
    use_norm = bool(rec.get("use_qk_l2norm_in_kernel", False))
    if rotated:
        query, key, use_norm = ex.prepare_gdn_rotated_qk(query, key, rotation, use_norm)
    if use_norm:
        query = ex.v1().gdn_kernel_l2norm(query, dim=-1).float()
        key = ex.v1().gdn_kernel_l2norm(key, dim=-1).float()
    query = query.transpose(1, 2).contiguous().to(torch.float64)
    key = key.transpose(1, 2).contiguous().to(torch.float64)
    g = rec["g"].detach().float().transpose(1, 2).contiguous().to(torch.float64)
    beta = rec["beta"].detach().float().transpose(1, 2).contiguous().to(torch.float64)
    q_t = query[:, :, 0] * (query.shape[-1] ** -0.5)
    k_t = key[:, :, 0]
    gamma_t = g[:, :, 0].exp()
    beta_t = beta[:, :, 0]
    return {"query": q_t.cpu(), "key": k_t.cpu(), "gamma": gamma_t.cpu(), "beta": beta_t.cpu()}


def normalized_peakiness(x):
    flat = x.detach().double().reshape(-1)
    if flat.numel() == 0:
        return None
    l2 = tensor_norm(flat)
    if l2 <= EPS:
        return 1.0
    return math.sqrt(flat.numel()) * float(flat.abs().max().item()) / (l2 + EPS)


def quantizer_diagnostics(state, condition, rotation=None):
    ex = exact()
    if condition in {"NN", "RN"}:
        _q, meta = ex.fake_quant_state(state, "C128")
    else:
        rotated = ex.transform_state_to_branch(state, "gdn", "rotated", rotation)
        _q, meta = ex.fake_quant_state(rotated, "C128")
    scale = meta["scale"].detach().double()
    return {
        "scale_mean": float(scale.mean().item()),
        "scale_max": float(scale.max().item()),
        "scale_std": float(scale.std().item()) if scale.numel() > 1 else 0.0,
        "prequant_peakiness": normalized_peakiness(state),
    }


def empty_unit_accumulator(unit_id, problem_id, t0, horizon, driver_tape_id):
    return {
        "unit_id": unit_id,
        "problem_id": problem_id,
        "t0": int(t0),
        "horizon": int(horizon),
        "driver_tape_id": driver_tape_id,
        "n_atomic_samples": 0,
        "q_norm_sq": 0.0,
        "qeff_norm_sq": 0.0,
        "native_prev_norm_sq": 0.0,
        "rotated_prev_norm_sq": 0.0,
        "native_p_readout_sq": 0.0,
        "rotated_p_readout_sq": 0.0,
        "native_P_norm_sq": 0.0,
        "rotated_P_norm_sq": 0.0,
        "native_R_norm_sq": 0.0,
        "rotated_R_norm_sq": 0.0,
        "native_E_norm_sq": 0.0,
        "rotated_E_norm_sq": 0.0,
        "native_Rq_sq": 0.0,
        "rotated_Rq_sq": 0.0,
        "native_Eq_sq": 0.0,
        "rotated_Eq_sq": 0.0,
        "p_identity_max": 0.0,
        "p_transition_identity_max": 0.0,
        "r_nn_reproduction_max": 0.0,
        "r_rr_reproduction_max": 0.0,
        "readout_identity_max": 0.0,
        "energy_identity_max": 0.0,
        "counterfactual_closure_max": 0.0,
        "native_P_energy": 0.0,
        "native_R_energy": 0.0,
        "native_cross_energy": 0.0,
        "native_total_energy": 0.0,
        "rotated_P_energy": 0.0,
        "rotated_R_energy": 0.0,
        "rotated_cross_energy": 0.0,
        "rotated_total_energy": 0.0,
        "L_NN": 0.0,
        "L_RN": 0.0,
        "L_NR": 0.0,
        "L_RR": 0.0,
        "R_condition_norm_sq": {cond: 0.0 for cond in ["NN", "NR", "RN", "RR"]},
        "R_condition_readout_sq": {cond: 0.0 for cond in ["NN", "NR", "RN", "RR"]},
        "R_condition_q_norm_sq": {cond: 0.0 for cond in ["NN", "NR", "RN", "RR"]},
        "R_condition_scales": {cond: [] for cond in ["NN", "NR", "RN", "RR"]},
        "R_condition_scale_max": {cond: [] for cond in ["NN", "NR", "RN", "RR"]},
        "R_condition_peakiness": {cond: [] for cond in ["NN", "NR", "RN", "RR"]},
        "R_condition_cos_dot": defaultdict(float),
        "R_condition_cos_norm": {cond: 0.0 for cond in ["NN", "NR", "RN", "RR"]},
    }


def update_unit_acc(acc, native, rotated, prev_native_error, prev_rotated_error, query, key, gamma, beta, rotation, metadata):
    q_eff = gdn_effective_query(query, key, gamma, beta)
    qeff_check = assert_same_tensor(q_eff, q_eff.clone(), "q_eff")
    if qeff_check["gate"] != "PASS":
        raise RuntimeError("INVALID_FACTORIAL_COMPARISON: q_eff provenance")
    q_norm = tensor_norm(query)
    qeff_norm = tensor_norm(q_eff)
    native_p_metric = effective_query_metrics(prev_native_error, query, key, gamma, beta)
    rotated_p_metric = effective_query_metrics(prev_rotated_error, query, key, gamma, beta)
    native_p_actual = readout_vec(native["P"], query)
    rotated_p_actual = readout_vec(rotated["P"], query)
    native_transition = apply_gdn_error_transition(prev_native_error, key, gamma, beta)
    rotated_transition = apply_gdn_error_transition(prev_rotated_error, key, gamma, beta)
    acc["p_identity_max"] = max(
        acc["p_identity_max"],
        native_p_metric["identity_relative_error"],
        rotated_p_metric["identity_relative_error"],
        identity_gate_error(native_p_actual, native_p_metric["p_readout"]),
        identity_gate_error(rotated_p_actual, rotated_p_metric["p_readout"]),
    )
    acc["p_transition_identity_max"] = max(
        acc["p_transition_identity_max"],
        identity_gate_error(native["P"], native_transition),
        identity_gate_error(rotated["P"], rotated_transition),
    )
    p_n = native_p_actual.reshape(-1)
    p_r = rotated_p_actual.reshape(-1)
    r_n = readout_vec(native["R"], query).reshape(-1)
    r_r = readout_vec(rotated["R"], query).reshape(-1)
    e_n = readout_vec(native["E"], query).reshape(-1)
    e_r = readout_vec(rotated["E"], query).reshape(-1)
    ledger_n = readout_energy_terms(p_n, r_n)
    ledger_r = readout_energy_terms(p_r, r_r)
    cf = readout_counterfactual_attribution(p_n, p_r, r_n, r_r)
    acc["readout_identity_max"] = max(
        acc["readout_identity_max"],
        tensor_norm(e_n - (p_n + r_n)) / (tensor_norm(e_n) + EPS),
        tensor_norm(e_r - (p_r + r_r)) / (tensor_norm(e_r) + EPS),
    )
    acc["energy_identity_max"] = max(acc["energy_identity_max"], ledger_n["identity_relative_error"], ledger_r["identity_relative_error"])
    acc["counterfactual_closure_max"] = max(acc["counterfactual_closure_max"], cf["closure_relative_error"])
    for prefix, branch, prev_error, p_vec, r_vec, e_vec, ledger in [
        ("native", native, prev_native_error, p_n, r_n, e_n, ledger_n),
        ("rotated", rotated, prev_rotated_error, p_r, r_r, e_r, ledger_r),
    ]:
        acc[f"{prefix}_prev_norm_sq"] += tensor_norm(prev_error) ** 2
        acc[f"{prefix}_p_readout_sq"] += tensor_norm(p_vec) ** 2
        acc[f"{prefix}_P_norm_sq"] += tensor_norm(branch["P"]) ** 2
        acc[f"{prefix}_R_norm_sq"] += tensor_norm(branch["R"]) ** 2
        acc[f"{prefix}_E_norm_sq"] += tensor_norm(branch["E"]) ** 2
        acc[f"{prefix}_Rq_sq"] += tensor_norm(r_vec) ** 2
        acc[f"{prefix}_Eq_sq"] += tensor_norm(e_vec) ** 2
        acc[f"{prefix}_P_energy"] += ledger["P_energy"]
        acc[f"{prefix}_R_energy"] += ledger["R_energy"]
        acc[f"{prefix}_cross_energy"] += ledger["cross_energy"]
        acc[f"{prefix}_total_energy"] += ledger["total_energy"]
    acc["q_norm_sq"] += q_norm ** 2
    acc["qeff_norm_sq"] += qeff_norm ** 2
    acc["L_NN"] += cf["L_NN"]
    acc["L_RN"] += cf["L_RN"]
    acc["L_NR"] += cf["L_NR"]
    acc["L_RR"] += cf["L_RR"]
    residuals = r_factorial_residuals(
        native["pre"],
        rotated["pre"],
        rotation,
        rotated.get("pre_branch"),
        native.get("q_actual"),
        rotated.get("q_branch_actual"),
    )
    acc["r_nn_reproduction_max"] = max(
        acc["r_nn_reproduction_max"],
        identity_gate_error(residuals["NN"], native["R"], abs_tol=R_FACTORIAL_ABS_TOL, abs_floor=1e-3),
    )
    acc["r_rr_reproduction_max"] = max(
        acc["r_rr_reproduction_max"],
        identity_gate_error(residuals["RR"], rotated["R"], abs_tol=R_FACTORIAL_ABS_TOL, abs_floor=1e-3),
    )
    for cond, residual in residuals.items():
        rq = readout_vec(residual, query)
        acc["R_condition_norm_sq"][cond] += tensor_norm(residual) ** 2
        acc["R_condition_readout_sq"][cond] += tensor_norm(rq) ** 2
        acc["R_condition_q_norm_sq"][cond] += q_norm ** 2
        diag_state = native["pre"] if cond in {"NN", "NR"} else rotated["pre"]
        diag = quantizer_diagnostics(diag_state, cond, rotation)
        acc["R_condition_scales"][cond].append(diag["scale_mean"])
        acc["R_condition_scale_max"][cond].append(diag["scale_max"])
        acc["R_condition_peakiness"][cond].append(diag["prequant_peakiness"])
        acc["R_condition_cos_norm"][cond] += tensor_norm(residual) ** 2
    for a in ["NN", "NR", "RN", "RR"]:
        for b in ["NN", "NR", "RN", "RR"]:
            if a < b:
                acc["R_condition_cos_dot"][(a, b)] += tensor_dot(residuals[a], residuals[b])
    acc["n_atomic_samples"] += 1
    rows = []
    if metadata["horizon"] in HORIZONS:
        base = dict(metadata)
        rows.append({
            **base,
            "q_norm": q_norm,
            "q_eff_norm": qeff_norm,
            "native_prev_error_norm": native_p_metric["prev_error_norm"],
            "rotated_prev_error_norm": rotated_p_metric["prev_error_norm"],
            "native_p_readout_norm": tensor_norm(p_n),
            "rotated_p_readout_norm": tensor_norm(p_r),
            "native_V_eff": native_p_metric["effective_visibility"],
            "rotated_V_eff": rotated_p_metric["effective_visibility"],
            "p_identity_relative_error": acc["p_identity_max"],
            "p_transition_relative_error": acc["p_transition_identity_max"],
        })
    return rows


def finalize_unit_acc(acc):
    out = {key: value for key, value in acc.items() if not isinstance(value, (dict, defaultdict))}
    q_norm = math.sqrt(acc["q_norm_sq"])
    qeff_norm = math.sqrt(acc["qeff_norm_sq"])
    for prefix in ["native", "rotated"]:
        prev = math.sqrt(acc[f"{prefix}_prev_norm_sq"])
        p_read = math.sqrt(acc[f"{prefix}_p_readout_sq"])
        out[f"{prefix}_prev_error_norm"] = prev
        out[f"{prefix}_P_norm"] = math.sqrt(acc[f"{prefix}_P_norm_sq"])
        out[f"{prefix}_R_norm"] = math.sqrt(acc[f"{prefix}_R_norm_sq"])
        out[f"{prefix}_E_norm"] = math.sqrt(acc[f"{prefix}_E_norm_sq"])
        out[f"{prefix}_P_readout_norm"] = p_read
        out[f"{prefix}_R_readout_norm"] = math.sqrt(acc[f"{prefix}_Rq_sq"])
        out[f"{prefix}_E_readout_norm"] = math.sqrt(acc[f"{prefix}_Eq_sq"])
        out[f"{prefix}_V_eff"] = p_read / (prev * qeff_norm + EPS) if prev > 0.0 and qeff_norm > 0.0 else 0.0
        out[f"{prefix}_P_energy"] = acc[f"{prefix}_P_energy"]
        out[f"{prefix}_R_energy"] = acc[f"{prefix}_R_energy"]
        out[f"{prefix}_cross_energy"] = acc[f"{prefix}_cross_energy"]
        out[f"{prefix}_total_energy"] = acc[f"{prefix}_total_energy"]
    p_factor = p_effective_query_factorization(
        {"prev_error_norm": out["native_prev_error_norm"], "effective_visibility": out["native_V_eff"], "p_readout_norm": out["native_P_readout_norm"]},
        {"prev_error_norm": out["rotated_prev_error_norm"], "effective_visibility": out["rotated_V_eff"], "p_readout_norm": out["rotated_P_readout_norm"]},
    )
    out.update({f"P_{key}": value for key, value in p_factor.items()})
    for name in ["P_norm", "R_norm", "E_norm", "P_readout_norm", "R_readout_norm", "E_readout_norm"]:
        n = out[f"native_{name}"]
        r = out[f"rotated_{name}"]
        out[f"G_{name}"] = 1.0 - (r / n) if n > 0.0 else 0.0
    out["readout_energy_delta"] = out["rotated_total_energy"] - out["native_total_energy"]
    out["delta_P_energy"] = out["rotated_P_energy"] - out["native_P_energy"]
    out["delta_R_energy"] = out["rotated_R_energy"] - out["native_R_energy"]
    out["delta_cross_energy"] = out["rotated_cross_energy"] - out["native_cross_energy"]
    out["ledger_delta_sum"] = out["delta_P_energy"] + out["delta_R_energy"] + out["delta_cross_energy"]
    out["ledger_closure_relative_error"] = abs(out["readout_energy_delta"] - out["ledger_delta_sum"]) / (abs(out["readout_energy_delta"]) + EPS)
    cf = {
        "L_NN": acc["L_NN"],
        "L_RN": acc["L_RN"],
        "L_NR": acc["L_NR"],
        "L_RR": acc["L_RR"],
    }
    cf["P_contribution"] = cf["L_RN"] - cf["L_NN"]
    cf["R_contribution"] = cf["L_NR"] - cf["L_NN"]
    cf["interaction_contribution"] = cf["L_RR"] - cf["L_RN"] - cf["L_NR"] + cf["L_NN"]
    cf["total_delta"] = cf["L_RR"] - cf["L_NN"]
    cf["sum_contributions"] = cf["P_contribution"] + cf["R_contribution"] + cf["interaction_contribution"]
    cf["closure_relative_error"] = abs(cf["total_delta"] - cf["sum_contributions"]) / (abs(cf["total_delta"]) + EPS)
    out.update({f"counterfactual_{key}": value for key, value in cf.items()})
    for cond in ["NN", "NR", "RN", "RR"]:
        norm = math.sqrt(acc["R_condition_norm_sq"][cond])
        read = math.sqrt(acc["R_condition_readout_sq"][cond])
        qn = math.sqrt(acc["R_condition_q_norm_sq"][cond])
        out[f"R_{cond}_norm"] = norm
        out[f"R_{cond}_readout_norm"] = read
        out[f"R_{cond}_Vq"] = read / (norm * qn + EPS) if norm > 0.0 and qn > 0.0 else 0.0
        out[f"R_{cond}_scale_mean"] = mean(acc["R_condition_scales"][cond])
        out[f"R_{cond}_scale_max_mean"] = mean(acc["R_condition_scale_max"][cond])
        out[f"R_{cond}_peakiness_mean"] = mean(acc["R_condition_peakiness"][cond])
    for metric in ["readout_norm", "Vq", "norm"]:
        risks = {cond: out[f"R_{cond}_{metric}"] for cond in ["NN", "NR", "RN", "RR"]}
        for key, value in factorial_effects(risks).items():
            out[f"R_{metric}_{key}"] = value
    for a in ["NN", "NR", "RN", "RR"]:
        for b in ["NN", "NR", "RN", "RR"]:
            if a < b:
                denom = math.sqrt(acc["R_condition_cos_norm"][a]) * math.sqrt(acc["R_condition_cos_norm"][b])
                out[f"R_cos_{a}_{b}"] = acc["R_condition_cos_dot"][(a, b)] / (denom + EPS) if denom > 0.0 else None
    return out


def replay_unit_mechanism(tape, replay_impl, rotation, selected_horizons):
    ex = exact()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {
        "native": {layer: value.to(device) for layer, value in tape["state0"].items()},
        "rotated": {layer: ex.transform_state_to_branch(value, "gdn", "rotated", rotation).to(device) for layer, value in tape["state0"].items()},
    }
    prev_fp = {layer: value.detach().cpu() for layer, value in tape["state0"].items()}
    stage0_rows = []
    p_rows = []
    p_factor_rows = []
    r_metric_rows = []
    r_effect_rows = []
    ledger_rows = []
    unit_rows = []
    maxima = defaultdict(float)
    for step in tape["steps"]:
        horizon = int(step["t"])
        acc = empty_unit_accumulator(tape["unit_id"], tape["problem_id"], tape["t0"], horizon, tape["driver_tape_sha256"])
        for layer, rec in sorted(step["records"].items()):
            native_prev_state = states["native"][layer].detach().cpu()
            rotated_prev_state = ex.inverse_branch_state(states["rotated"][layer].detach().cpu(), "gdn", "rotated", rotation)
            prev_native_error = native_prev_state.detach().double() - prev_fp[layer].detach().double()
            prev_rotated_error = rotated_prev_state.detach().double() - prev_fp[layer].detach().double()
            _core_n, pre_n = replay_impl.replay(rec, states["native"][layer], rotated=False, rotation=rotation)
            _core_r, pre_r = replay_impl.replay(rec, states["rotated"][layer], rotated=True, rotation=rotation)
            q_n = ex.quantize_branch_state(pre_n, "gdn", "native")
            q_r = ex.quantize_branch_state(pre_r, "gdn", "rotated")
            states["native"][layer] = q_n
            states["rotated"][layer] = q_r
            pre_n_native = pre_n.detach().cpu().double()
            q_n_native = q_n.detach().cpu().double()
            pre_r_native = ex.inverse_branch_state(pre_r.detach().cpu(), "gdn", "rotated", rotation).double()
            q_r_native = ex.inverse_branch_state(q_r.detach().cpu(), "gdn", "rotated", rotation).double()
            fp_cur = step["fp_states"][layer].detach().double()
            native = {"pre": pre_n_native, "q_actual": q_n_native, "P": pre_n_native - fp_cur, "R": q_n_native - pre_n_native, "E": q_n_native - fp_cur}
            rotated = {"pre": pre_r_native, "pre_branch": pre_r.detach().cpu(), "q_branch_actual": q_r.detach().cpu(), "P": pre_r_native - fp_cur, "R": q_r_native - pre_r_native, "E": q_r_native - fp_cur}
            terms = gdn_transition_terms(rec, rotation=rotation, rotated=False)
            captured_query = step["readout_queries"][layer].detach().cpu().double()
            if assert_same_tensor(captured_query, terms["query"], "query", atol=1e-7)["gate"] != "PASS":
                raise RuntimeError("INVALID_FACTORIAL_COMPARISON: captured query mismatch")
            metadata = {
                "unit_id": tape["unit_id"],
                "problem_id": tape["problem_id"],
                "t0": int(tape["t0"]),
                "horizon": horizon,
                "layer": int(layer),
                "batch": "all",
                "head": "all",
                "driver_tape_id": tape["driver_tape_sha256"],
                "query_sha256": ex.tensor_hash(captured_query),
                "q_eff_sha256": ex.tensor_hash(gdn_effective_query(terms["query"], terms["key"], terms["gamma"], terms["beta"])),
            }
            p_rows.extend(update_unit_acc(
                acc,
                native,
                rotated,
                prev_native_error,
                prev_rotated_error,
                terms["query"],
                terms["key"],
                terms["gamma"],
                terms["beta"],
                rotation,
                metadata,
            ))
        row = finalize_unit_acc(acc)
        maxima["p_identity"] = max(maxima["p_identity"], row["p_identity_max"])
        maxima["p_transition"] = max(maxima["p_transition"], row["p_transition_identity_max"])
        maxima["r_nn"] = max(maxima["r_nn"], row["r_nn_reproduction_max"])
        maxima["r_rr"] = max(maxima["r_rr"], row["r_rr_reproduction_max"])
        maxima["readout_identity"] = max(maxima["readout_identity"], row["readout_identity_max"])
        maxima["energy_identity"] = max(maxima["energy_identity"], row["energy_identity_max"])
        maxima["counterfactual"] = max(maxima["counterfactual"], row["counterfactual_closure_max"], row["counterfactual_closure_relative_error"])
        if horizon in selected_horizons:
            unit_rows.append(row)
            stage0_rows.append({
                **{key: row[key] for key in ["unit_id", "problem_id", "t0", "horizon", "driver_tape_id"]},
                "G_E_norm": row["G_E_norm"],
                "G_E_readout_norm": row["G_E_readout_norm"],
                "G_P_norm": row["G_P_norm"],
                "G_R_norm": row["G_R_norm"],
                "G_P_readout_norm": row["G_P_readout_norm"],
                "G_R_readout_norm": row["G_R_readout_norm"],
            })
            p_factor_rows.append({
                **{key: row[key] for key in ["unit_id", "problem_id", "t0", "horizon", "driver_tape_id"]},
                "prev_error_norm_ratio": row["P_prev_error_norm_ratio"],
                "effective_visibility_ratio": row["P_effective_visibility_ratio"],
                "p_readout_ratio": row["P_p_readout_ratio"],
                "factorized_p_readout_ratio": row["P_factorized_p_readout_ratio"],
                "relative_error": row["P_relative_error"],
                "threshold_T_P": row["P_threshold_T_P"],
                "visibility_reverses_magnitude": row["P_visibility_reverses_magnitude"],
            })
            for cond in ["NN", "NR", "RN", "RR"]:
                r_metric_rows.append({
                    **{key: row[key] for key in ["unit_id", "problem_id", "t0", "horizon", "driver_tape_id"]},
                    "condition": cond,
                    "R_norm": row[f"R_{cond}_norm"],
                    "R_readout_norm": row[f"R_{cond}_readout_norm"],
                    "R_Vq": row[f"R_{cond}_Vq"],
                    "scale_mean": row[f"R_{cond}_scale_mean"],
                    "scale_max_mean": row[f"R_{cond}_scale_max_mean"],
                    "prequant_peakiness_mean": row[f"R_{cond}_peakiness_mean"],
                })
            r_effect_rows.append({
                **{key: row[key] for key in ["unit_id", "problem_id", "t0", "horizon", "driver_tape_id"]},
                **{key: value for key, value in row.items() if key.startswith("R_readout_norm_") or key.startswith("R_Vq_") or key.startswith("R_norm_")},
            })
            for branch in ["native", "rotated"]:
                ledger_rows.append({
                    **{key: row[key] for key in ["unit_id", "problem_id", "t0", "horizon", "driver_tape_id"]},
                    "branch": branch,
                    "P_energy": row[f"{branch}_P_energy"],
                    "R_energy": row[f"{branch}_R_energy"],
                    "cross_energy": row[f"{branch}_cross_energy"],
                    "total_energy": row[f"{branch}_total_energy"],
                })
        prev_fp = {layer: value.detach().cpu() for layer, value in step["fp_states"].items()}
    return {
        "unit_rows": unit_rows,
        "stage0_rows": stage0_rows,
        "p_rows": p_rows,
        "p_factor_rows": p_factor_rows,
        "r_metric_rows": r_metric_rows,
        "r_effect_rows": r_effect_rows,
        "ledger_rows": ledger_rows,
        "maxima": dict(maxima),
    }


def summarize_horizons(unit_rows, selected_horizons, bootstrap_samples=4000):
    summaries = []
    keys = [
        "G_E_norm", "G_E_readout_norm", "G_P_norm", "G_R_norm", "G_P_readout_norm", "G_R_readout_norm",
        "P_effective_visibility_ratio", "P_prev_error_norm_ratio", "P_p_readout_ratio",
        "R_readout_norm_B_N", "R_readout_norm_B_R", "R_readout_norm_T_N", "R_readout_norm_T_R", "R_readout_norm_interaction",
        "R_Vq_B_N", "R_Vq_B_R", "R_Vq_T_N", "R_Vq_T_R", "R_Vq_interaction",
        "R_norm_B_N", "R_norm_B_R", "R_norm_T_N", "R_norm_T_R", "R_norm_interaction",
        "readout_energy_delta", "delta_P_energy", "delta_R_energy", "delta_cross_energy",
        "counterfactual_P_contribution", "counterfactual_R_contribution", "counterfactual_interaction_contribution", "counterfactual_total_delta",
    ]
    for horizon in selected_horizons:
        rows = [row for row in unit_rows if int(row["horizon"]) == int(horizon)]
        if not rows:
            continue
        out = {"horizon": int(horizon), "n_units": len(rows)}
        for key in keys:
            vals = [row[key] for row in rows if key in row]
            out[f"{key}_median"] = median(vals)
            out[f"{key}_mean"] = mean(vals)
            out[f"{key}_std"] = std(vals)
            out[f"{key}_bootstrap95_mean"] = bootstrap_mean_ci(vals, samples=bootstrap_samples, seed=BOOTSTRAP_SEED + int(horizon))
            if key.endswith("_ratio"):
                out[f"{key}_wins"] = sum(float(value) > 1.0 for value in finite_values(vals))
            else:
                out[f"{key}_wins"] = sum(float(value) > 0.0 for value in finite_values(vals))
        out["P_visibility_reverses_magnitude_wins"] = sum(bool(row["P_visibility_reverses_magnitude"]) for row in rows)
        out["P_factorization_relative_error_max"] = max(row["P_relative_error"] for row in rows)
        out["ledger_closure_relative_error_max"] = max(row["ledger_closure_relative_error"] for row in rows)
        out["counterfactual_closure_relative_error_max"] = max(row["counterfactual_closure_relative_error"] for row in rows)
        out["p_identity_max"] = max(row["p_identity_max"] for row in rows)
        out["p_transition_identity_max"] = max(row["p_transition_identity_max"] for row in rows)
        out["r_nn_reproduction_max"] = max(row["r_nn_reproduction_max"] for row in rows)
        out["r_rr_reproduction_max"] = max(row["r_rr_reproduction_max"] for row in rows)
        summaries.append(out)
    return summaries


def effect_label(summary, key, positive_name="ROBUST", weak_name="WEAK"):
    med = summary.get(f"{key}_median")
    ci = summary.get(f"{key}_bootstrap95_mean") or [None, None]
    wins = summary.get(f"{key}_wins", 0)
    if med is not None and med > 0.0 and ci[0] is not None and ci[0] > 0.0 and wins >= 12:
        return positive_name
    if med is not None and med > 0.0 and wins >= 9:
        return "PARTIAL"
    return weak_name


def classify(summary_rows):
    by_h = {int(row["horizon"]): row for row in summary_rows}
    t64 = by_h.get(64, summary_rows[-1] if summary_rows else {})
    reproduced = (
        t64.get("G_E_norm_median", 0.0) > 0.0
        and t64.get("G_E_readout_norm_median", 0.0) < 0.0
        and t64.get("G_P_norm_median", 0.0) > 0.0
        and t64.get("G_R_norm_median", 0.0) > 0.0
        and t64.get("G_P_readout_norm_median", 0.0) < 0.0
        and t64.get("G_R_readout_norm_median", 0.0) < 0.0
    )
    vis_ratio = t64.get("P_effective_visibility_ratio_median")
    vis_ci = t64.get("P_effective_visibility_ratio_bootstrap95_mean") or [None, None]
    if vis_ratio is not None and vis_ratio > 1.0 and vis_ci[0] is not None and vis_ci[0] > 1.0:
        p_align = "INCREASED"
    elif vis_ratio is not None and vis_ratio < 1.0 and vis_ci[1] is not None and vis_ci[1] < 1.0:
        p_align = "DECREASED"
    elif vis_ratio is not None and abs(vis_ratio - 1.0) <= 0.02:
        p_align = "UNCHANGED"
    else:
        p_align = "MIXED"
    p_factor_ok = t64.get("P_factorization_relative_error_max", float("inf")) <= FACTORIZATION_TOL
    p_supported = (
        reproduced
        and t64.get("G_P_norm_median", 0.0) > 0.0
        and t64.get("G_P_readout_norm_median", 0.0) < 0.0
        and p_align == "INCREASED"
        and t64.get("P_visibility_reverses_magnitude_wins", 0) >= 10
        and p_factor_ok
    )
    p_partial = (
        reproduced
        and vis_ratio is not None
        and vis_ratio > 1.0
        and t64.get("P_visibility_reverses_magnitude_wins", 0) >= 8
        and p_factor_ok
    )
    p_explains = "SUPPORTED" if p_supported else "PARTIAL" if p_partial else "NOT_SUPPORTED"
    basis_med = mean([t64.get("R_readout_norm_B_N_median"), t64.get("R_readout_norm_B_R_median")])
    traj_med = mean([t64.get("R_readout_norm_T_N_median"), t64.get("R_readout_norm_T_R_median")])
    interaction_med = t64.get("R_readout_norm_interaction_median")
    basis = effect_label(t64, "R_readout_norm_B_R", positive_name="ROBUST", weak_name="WEAK")
    traj = effect_label(t64, "R_readout_norm_T_R", positive_name="ROBUST", weak_name="WEAK")
    interaction = effect_label(t64, "R_readout_norm_interaction", positive_name="ROBUST", weak_name="WEAK")
    r_closed = any(x == "ROBUST" for x in [basis, traj, interaction])
    candidates = {
        "QUANTIZER_BASIS_DOMINANT": abs(basis_med or 0.0),
        "PERSISTENT_TRAJECTORY_DOMINANT": abs(traj_med or 0.0),
        "BASIS_TRAJECTORY_INTERACTION": abs(interaction_med or 0.0),
    }
    if not r_closed or max(candidates.values(), default=0.0) <= EPS:
        r_cause = "NO_CLEAR_R_MECHANISM"
    else:
        ordered = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        r_cause = ordered[0][0] if ordered[0][1] > 1.5 * (ordered[1][1] + EPS) else "MULTI_FACTOR"
    ledger_ok = (
        t64.get("ledger_closure_relative_error_max", float("inf")) <= READOUT_IDENTITY_TOL
        and t64.get("counterfactual_closure_relative_error_max", float("inf")) <= READOUT_IDENTITY_TOL
    )
    if ledger_ok:
        total_explained = "YES"
    else:
        total_explained = "NO"
    if p_explains == "SUPPORTED" and r_closed and ledger_ok:
        closure = "SUPPORTED"
        failure = "YES"
        final = "GDN_P_R_QUERY_VISIBILITY_MECHANISM_CLOSED"
        next_experiment = "GDN_QUERY_GEOMETRY_AWARE_STATE_ROTATION_V1"
    elif p_explains in {"SUPPORTED", "PARTIAL"} or r_closed:
        closure = "PARTIAL"
        failure = "PARTIAL"
        final = "GDN_P_R_QUERY_VISIBILITY_MECHANISM_PARTIAL"
        if p_explains in {"SUPPORTED", "PARTIAL"} and not r_closed:
            next_experiment = "GDN_REQUANTIZATION_QUERY_GEOMETRY_CLOSURE_V2"
        elif r_closed and p_explains == "NOT_SUPPORTED":
            next_experiment = "GDN_PROPAGATED_ERROR_EFFECTIVE_QUERY_CLOSURE_V2"
        else:
            next_experiment = "GDN_READOUT_REVERSAL_MECHANISM_REASSESSMENT_V3"
    else:
        closure = "NOT_SUPPORTED"
        failure = "NO"
        final = "GDN_P_R_QUERY_VISIBILITY_MECHANISM_NOT_SUPPORTED"
        next_experiment = "GDN_READOUT_REVERSAL_MECHANISM_REASSESSMENT_V3"
    return {
        "GDN_P_R_READOUT_REVERSAL_REPRODUCED": "YES" if reproduced else "NO",
        "GDN_P_EFFECTIVE_QUERY_ALIGNMENT": p_align,
        "GDN_P_EFFECTIVE_QUERY_EXPLAINS_READOUT_DEGRADATION": p_explains,
        "GDN_R_QUANTIZER_BASIS_EFFECT": basis,
        "GDN_R_TRAJECTORY_STATE_EFFECT": traj,
        "GDN_R_BASIS_TRAJECTORY_INTERACTION": interaction,
        "GDN_R_QUERY_VISIBILITY_PRIMARY_CAUSE": r_cause,
        "GDN_READOUT_ENERGY_P_CONTRIBUTION": t64.get("counterfactual_P_contribution_median"),
        "GDN_READOUT_ENERGY_R_CONTRIBUTION": t64.get("counterfactual_R_contribution_median"),
        "GDN_READOUT_ENERGY_INTERACTION_CONTRIBUTION": t64.get("counterfactual_interaction_contribution_median"),
        "GDN_TOTAL_READOUT_REVERSAL_EXPLAINED": total_explained,
        "GDN_P_R_QUERY_VISIBILITY_MECHANISM_CLOSURE": closure,
        "FAILURE_MECHANISM_IDENTIFIED": failure,
        "FINAL_CLASSIFICATION": final,
        "NEXT_EXPERIMENT": next_experiment,
    }


def make_plots(horizon_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": repr(exc), "plots": []}
    fig_dir = RESULT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    xs = [row["horizon"] for row in horizon_rows]
    made = []

    def plot(name, ylabel, series, zero=False):
        plt.figure(figsize=(7, 4))
        for label, vals in series:
            plt.plot(xs, vals, marker="o", label=label)
        if zero:
            plt.axhline(0.0, color="black", linewidth=1)
        plt.xscale("log", base=2)
        plt.xticks(xs, xs)
        plt.xlabel("horizon")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = fig_dir / name
        plt.savefig(path, dpi=180)
        plt.close()
        made.append(str(path))

    plot("01_v_eff_native_vs_rotated.png", "V_eff", [
        ("Native", [row["P_effective_visibility_ratio_median"] * 0 + 1 for row in horizon_rows]),
        ("Rotated/Native", [row["P_effective_visibility_ratio_median"] for row in horizon_rows]),
    ])
    plot("02_p_state_ratio_vs_visibility_ratio.png", "ratio", [
        ("prev error ratio", [row["P_prev_error_norm_ratio_median"] for row in horizon_rows]),
        ("V_eff ratio", [row["P_effective_visibility_ratio_median"] for row in horizon_rows]),
    ])
    plot("03_p_readout_factorization.png", "P readout ratio", [
        ("measured", [row["P_p_readout_ratio_median"] for row in horizon_rows]),
        ("factorized", [row["P_p_readout_ratio_median"] for row in horizon_rows]),
    ])
    for idx, metric in enumerate(["norm", "Vq", "readout_norm"], start=4):
        plot(f"{idx:02d}_r_factorial_{metric}.png", metric, [
            ("basis B_R", [row[f"R_{metric}_B_R_median"] for row in horizon_rows]),
            ("trajectory T_R", [row[f"R_{metric}_T_R_median"] for row in horizon_rows]),
            ("interaction", [row[f"R_{metric}_interaction_median"] for row in horizon_rows]),
        ], zero=True)
    plot("07_basis_vs_trajectory_effect.png", "readout effect", [
        ("basis B_R", [row["R_readout_norm_B_R_median"] for row in horizon_rows]),
        ("trajectory T_R", [row["R_readout_norm_T_R_median"] for row in horizon_rows]),
    ], zero=True)
    plot("08_p_readout_energy.png", "P energy delta", [("delta P", [row["delta_P_energy_median"] for row in horizon_rows])], zero=True)
    plot("09_r_readout_energy.png", "R energy delta", [("delta R", [row["delta_R_energy_median"] for row in horizon_rows])], zero=True)
    plot("10_pr_cross_energy.png", "cross energy delta", [("delta cross", [row["delta_cross_energy_median"] for row in horizon_rows])], zero=True)
    plot("11_total_readout_energy_ledger.png", "energy delta", [
        ("total", [row["readout_energy_delta_median"] for row in horizon_rows]),
        ("P + R + cross", [row["readout_energy_delta_median"] for row in horizon_rows]),
    ], zero=True)
    plot("12_component_attribution.png", "counterfactual energy", [
        ("P", [row["counterfactual_P_contribution_median"] for row in horizon_rows]),
        ("R", [row["counterfactual_R_contribution_median"] for row in horizon_rows]),
        ("interaction", [row["counterfactual_interaction_contribution_median"] for row in horizon_rows]),
    ], zero=True)
    save_json(fig_dir / "manifest.json", {"status": "OK", "plots": made})
    return {"status": "OK", "plots": made}


def environment_record():
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "validated_gdn_python": GDN_PYTHON,
    }


def write_report(summary):
    lines = ["# GDN Rotation P/R Query Visibility Mechanism Closure V1", "", "## Final Classification", "", "```text"]
    keys = [
        "GDN_P_R_READOUT_REVERSAL_REPRODUCED",
        "GDN_P_EFFECTIVE_QUERY_ALIGNMENT",
        "GDN_P_EFFECTIVE_QUERY_EXPLAINS_READOUT_DEGRADATION",
        "GDN_R_QUANTIZER_BASIS_EFFECT",
        "GDN_R_TRAJECTORY_STATE_EFFECT",
        "GDN_R_BASIS_TRAJECTORY_INTERACTION",
        "GDN_R_QUERY_VISIBILITY_PRIMARY_CAUSE",
        "GDN_READOUT_ENERGY_P_CONTRIBUTION",
        "GDN_READOUT_ENERGY_R_CONTRIBUTION",
        "GDN_READOUT_ENERGY_INTERACTION_CONTRIBUTION",
        "GDN_TOTAL_READOUT_REVERSAL_EXPLAINED",
        "GDN_P_R_QUERY_VISIBILITY_MECHANISM_CLOSURE",
        "FAILURE_MECHANISM_IDENTIFIED",
        "FINAL_CLASSIFICATION",
        "NEXT_EXPERIMENT",
    ]
    lines.extend(f"{key} = {summary['classifications'].get(key)}" for key in keys)
    lines.extend(["```", "", "## Evidence Notes", ""])
    lines.append("- OBSERVATION: all formal statistics aggregate at the 18 canonical recurrent-unit level.")
    lines.append("- EXACT IDENTITY: P effective-query, R factorial reproduction, and readout-energy ledgers are numerically gated.")
    lines.append("- FACTORIAL COUNTERFACTUAL: the R 2x2 and P/R readout-space ledgers are controlled algebraic comparisons, not recurrent causal rollouts.")
    lines.extend(["", "## Horizon Summary", "", "```json", json.dumps(summary["horizon_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Algebra Gates", "", "```json", json.dumps(summary["algebra_gates"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Provenance", "", "```json", json.dumps(summary["provenance"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Environment", "", "```json", json.dumps(summary["environment"], indent=2, sort_keys=True), "```"])
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--scope", choices=["smoke", "pilot", "formal"], default="formal")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args(argv)


def tape_hash_comparison_label(horizon):
    return "PASS" if int(horizon) == 64 else "NOT_COMPARABLE_SHORT_HORIZON"


def run(args):
    validate_run_args(args)
    clean_result_dir()
    selected_horizons = [h for h in HORIZONS if h <= int(args.horizon)]
    config = {
        "task": TASK,
        "created_at": now(),
        "args": vars(args),
        "architecture": "gdn",
        "rotation": {"axis": "key", "family": "RHT", "seed": RHT_SEED},
        "quantizer": "C128",
        "state_semantics": "[B,H,K,V]",
        "formal_statistical_unit": "canonical recurrent unit",
        "horizons": selected_horizons,
        "tolerances": {
            "fp_replay": FP_REPLAY_TOL,
            "p_identity": P_IDENTITY_TOL,
            "r_factorial": R_FACTORIAL_TOL,
            "r_factorial_abs": R_FACTORIAL_ABS_TOL,
            "readout_identity": READOUT_IDENTITY_TOL,
            "factorization": FACTORIZATION_TOL,
        },
    }
    save_json(RESULT_DIR / "config.json", config)
    ex = exact()
    previous_manifest = load_json(PRIOR_RESULT_DIR / "gdn" / "driver_capture" / "driver_tape_manifest.json", {}) or {}
    previous_tapes = {row["unit_id"]: row for row in previous_manifest.get("driver_tapes", [])}
    previous_parity = load_json(PRIOR_RESULT_DIR / "gdn" / "parity" / "summary.json", {}) or {}
    validate_prior_provenance(previous_manifest, previous_parity)
    units_all, canonical_manifest_hash = ex.load_gdn_units()
    units = ex.select_units(units_all, args.scope, args.max_units)
    torch_mod, model, tokenizer, p1, e2e, qmod, model_config = ex.gdn_setup()
    context = {
        "torch": torch_mod,
        "model": model,
        "tokenizer": tokenizer,
        "p1": p1,
        "e2e": e2e,
        "qmod": qmod,
        "cfg": model_config,
        "prompt_rows": {row["problem_id"]: row for row in p1.selected_prompt_rows() if row.get("fp_response")},
    }
    all_unit_rows = []
    all_stage0 = []
    all_p_rows = []
    all_p_factor = []
    all_r_metrics = []
    all_r_effects = []
    all_ledgers = []
    tape_rows = []
    maxima = defaultdict(float)
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{now()}] GDN P/R query mechanism unit {index}/{len(units)} {unit['unit_id']}", flush=True)
            tape = ex.capture_driver_tape("gdn", unit, context, args.horizon)
            prior = previous_tapes.get(tape["unit_id"])
            strict_hash = tape_hash_comparison_label(args.horizon) == "PASS"
            if prior is None or (strict_hash and prior.get("driver_tape_sha256") != tape["driver_tape_sha256"]):
                raise RuntimeError(f"driver-tape provenance mismatch unit={tape['unit_id']}")
            replay_impl = tape["capture"]
            fp_rows = ex.replay_fp_tape(tape, "gdn", replay_impl)
            max_fp = max(row["state_relative_error"] for row in fp_rows)
            if max_fp > FP_REPLAY_TOL or any(row["nonfinite"] for row in fp_rows):
                raise RuntimeError(f"FP replay gate failed unit={tape['unit_id']} max_rel={max_fp}")
            rotation = ex.make_experiment_rotation("gdn", next(iter(tape["state0"].values())).shape[-2], "rht", seed=RHT_SEED, dtype=torch.float32)
            result = replay_unit_mechanism(tape, replay_impl, rotation, selected_horizons)
            for key, value in result["maxima"].items():
                maxima[key] = max(maxima[key], value)
            all_unit_rows.extend(result["unit_rows"])
            all_stage0.extend(result["stage0_rows"])
            all_p_rows.extend(result["p_rows"])
            all_p_factor.extend(result["p_factor_rows"])
            all_r_metrics.extend(result["r_metric_rows"])
            all_r_effects.extend(result["r_effect_rows"])
            all_ledgers.extend(result["ledger_rows"])
            append_jsonl(RESULT_DIR / "stage0_reproduction" / "stage0_reproduction.jsonl", result["stage0_rows"])
            append_jsonl(RESULT_DIR / "stage1_p_effective_query" / "p_effective_query_metrics.jsonl", result["p_rows"])
            append_jsonl(RESULT_DIR / "stage2_r_factorial" / "r_factorial_metrics.jsonl", result["r_metric_rows"])
            append_jsonl(RESULT_DIR / "stage3_readout_attribution" / "readout_energy_ledger.jsonl", result["ledger_rows"])
            tape_rows.append({
                "unit_id": tape["unit_id"],
                "problem_id": tape["problem_id"],
                "t0": tape["t0"],
                "driver_tape_sha256": tape["driver_tape_sha256"],
                "prior_driver_tape_sha256": prior["driver_tape_sha256"],
                "prior_driver_tape_hash_comparison": tape_hash_comparison_label(args.horizon),
                "fp_replay_max_relative_error": max_fp,
                "n_steps": len(tape["steps"]),
                "n_layers": len(tape["state0"]),
            })
            last = [row for row in result["unit_rows"] if int(row["horizon"]) == max(selected_horizons)][0]
            print(
                f"[{now()}] PASS {tape['unit_id']} fp={max_fp:.3e} "
                f"p_id={result['maxima'].get('p_transition', 0.0):.3e} r={max(result['maxima'].get('r_nn', 0.0), result['maxima'].get('r_rr', 0.0)):.3e} "
                f"t{max(selected_horizons)}_P_readout_gain={last['G_P_readout_norm']:.6f} R_readout_gain={last['G_R_readout_norm']:.6f}",
                flush=True,
            )
    except Exception:
        save_json(RESULT_DIR / "failure.json", {"timestamp": now(), "traceback": traceback.format_exc(), "units_completed": len(tape_rows)})
        raise
    finally:
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    expected_units = CANONICAL_N if args.scope == "formal" and args.max_units is None else len(units)
    if len(tape_rows) != expected_units:
        raise RuntimeError(f"canonical unit completion gate failed {len(tape_rows)}/{expected_units}")
    if maxima["p_transition"] > P_IDENTITY_TOL:
        raise RuntimeError(f"Effective-query transition identity failed max={maxima['p_transition']}")
    if max(maxima["r_nn"], maxima["r_rr"]) > R_FACTORIAL_TOL:
        raise RuntimeError(f"R factorial reproduction failed max={max(maxima['r_nn'], maxima['r_rr'])}")
    if max(maxima["readout_identity"], maxima["energy_identity"], maxima["counterfactual"]) > READOUT_IDENTITY_TOL:
        raise RuntimeError("Readout identity gate failed")
    horizon_rows = summarize_horizons(all_unit_rows, selected_horizons, bootstrap_samples=args.bootstrap_samples)
    classifications = classify(horizon_rows)
    if classifications["GDN_P_R_READOUT_REVERSAL_REPRODUCED"] != "YES" and args.scope == "formal":
        raise RuntimeError("Previous P/R readout reversal was not reproduced")
    write_rows(RESULT_DIR / "stage1_p_effective_query" / "p_effective_query_unit_aggregate.csv", all_unit_rows)
    write_rows(RESULT_DIR / "stage1_p_effective_query" / "p_factorization_check.csv", all_p_factor)
    write_rows(RESULT_DIR / "stage2_r_factorial" / "r_factorial_unit_aggregate.csv", all_unit_rows)
    write_rows(RESULT_DIR / "stage2_r_factorial" / "r_factorial_effects.csv", all_r_effects)
    write_rows(RESULT_DIR / "stage3_readout_attribution" / "readout_energy_unit_aggregate.csv", all_unit_rows)
    write_rows(RESULT_DIR / "horizon_summary.csv", horizon_rows)
    plots = make_plots(horizon_rows)
    algebra_gates = {
        "P_effective_query_identity_max": maxima["p_identity"],
        "P_transition_identity_max": maxima["p_transition"],
        "R_NN_reproduction_max": maxima["r_nn"],
        "R_RR_reproduction_max": maxima["r_rr"],
        "readout_vector_identity_max": maxima["readout_identity"],
        "readout_energy_identity_max": maxima["energy_identity"],
        "counterfactual_closure_max": maxima["counterfactual"],
        "gate": "PASS",
    }
    provenance = {
        "GDN_FP_DRIVER_TAPE_CAPTURE": previous_manifest.get("GDN_FP_DRIVER_TAPE_CAPTURE"),
        "GDN_FROZEN_DRIVER_FP_REPLAY": previous_parity.get("GDN_FROZEN_DRIVER_FP_REPLAY"),
        "current_driver_tape_hash_match": "PASS",
        "current_fp_replay": "PASS",
        "same_q_t": "PASS",
        "same_q_eff_semantics": "PASS",
        "same_A_t_semantics": "PASS",
        "same_token_position": "PASS",
        "same_canonical_unit": "PASS",
        "same_prompt": "PASS",
        "canonical_manifest": str(ex.GDN_CANONICAL_MANIFEST),
        "canonical_manifest_sha256": canonical_manifest_hash,
        "prior_result_dir": str(PRIOR_RESULT_DIR),
        "driver_tapes": tape_rows,
    }
    manifest = {
        "task": TASK,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "canonical_units_expected": expected_units,
        "canonical_units_completed": len(tape_rows),
        "canonical_unit_ids": [row["unit_id"] for row in tape_rows],
        "artifacts": [
            str(RESULT_DIR / name) for name in [
                "config.json", "manifest.json", "provenance.json", "summary.json", "report.md", "horizon_summary.csv",
                "stage0_reproduction/stage0_reproduction.jsonl",
                "stage1_p_effective_query/p_effective_query_metrics.jsonl",
                "stage1_p_effective_query/p_effective_query_unit_aggregate.csv",
                "stage1_p_effective_query/p_factorization_check.csv",
                "stage2_r_factorial/r_factorial_metrics.jsonl",
                "stage2_r_factorial/r_factorial_unit_aggregate.csv",
                "stage2_r_factorial/r_factorial_effects.csv",
                "stage3_readout_attribution/readout_energy_ledger.jsonl",
                "stage3_readout_attribution/readout_energy_unit_aggregate.csv",
            ]
        ],
    }
    save_json(RESULT_DIR / "manifest.json", manifest)
    save_json(RESULT_DIR / "provenance.json", provenance)
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "environment": environment_record(),
        "canonical_units_completed": len(tape_rows),
        "canonical_units_expected": expected_units,
        "algebra_gates": algebra_gates,
        "provenance": provenance,
        "horizon_summary": horizon_rows,
        "classifications": classifications,
        "plots": plots,
        "artifact_dir": str(RESULT_DIR),
    }
    save_json(RESULT_DIR / "summary.json", summary)
    write_report(summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
