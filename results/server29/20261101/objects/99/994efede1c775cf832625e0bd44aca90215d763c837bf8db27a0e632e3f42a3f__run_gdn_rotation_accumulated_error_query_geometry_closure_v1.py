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
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "GDN_ROTATION_ACCUMULATED_ERROR_QUERY_GEOMETRY_CLOSURE_V1"
SLUG = "gdn_rotation_accumulated_error_query_geometry_closure_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
PRIOR_RESULT_DIR = REPO / "results" / "gdn_kda_rotation_exact_frozen_driver_replay_v1"
EXACT_RUNNER = REPO / "experiments" / "rotation" / "run_gdn_kda_rotation_exact_frozen_driver_replay_v1.py"
GDN_PYTHON = "/data01/user2/.conda/envs/sd310/bin/python"
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
EPS = 1e-12
GEOMETRY_IDENTITY_TOL = 1e-10
PR_IDENTITY_TOL = 1e-6
FACTORIZATION_TOL = 1e-8
FP_REPLAY_TOL = 1e-5
RHT_SEED = 0
CANONICAL_N = 18
BOOTSTRAP_SEED = 20260914


_EXACT = None


def exact():
    global _EXACT
    if _EXACT is None:
        spec = importlib.util.spec_from_file_location("exact_frozen_driver_for_query_geometry", EXACT_RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXACT = module
    return _EXACT


def tensor_norm(value):
    return float(torch.linalg.vector_norm(value.detach().double()).item())


def relative_error(actual, expected, eps=EPS):
    return abs(float(actual) - float(expected)) / (abs(float(expected)) + eps)


def query_projection(error, query, eps=EPS):
    """Project one [B,H,K,V] error onto one query direction per [B,H]."""
    error64 = error.detach().double()
    query64 = query.detach().double()
    if error64.ndim != 4 or query64.ndim != 3 or error64.shape[:-1] != query64.shape:
        raise ValueError(f"expected error [B,H,K,V] and query [B,H,K], got {tuple(error64.shape)} and {tuple(query64.shape)}")
    if error64.shape[0] * error64.shape[1] != 1:
        raise ValueError("query_projection requires one atomic batch/head sample; use atomic_query_geometry_rows")
    query_norm = tensor_norm(query64)
    qhat = query64 / query_norm if query_norm > 0.0 else torch.zeros_like(query64)
    visible_readout = torch.einsum("bhk,bhkv->bhv", qhat, error64)
    parallel = qhat.unsqueeze(-1) * visible_readout.unsqueeze(-2)
    perpendicular = error64 - parallel
    readout = torch.einsum("bhkv,bhk->bhv", error64, query64)
    error_norm = tensor_norm(error64)
    parallel_norm = tensor_norm(parallel)
    perpendicular_norm = tensor_norm(perpendicular)
    readout_norm = tensor_norm(readout)
    reconstruction_error = tensor_norm(error64 - parallel - perpendicular)
    energy_lhs = error_norm ** 2
    energy_rhs = parallel_norm ** 2 + perpendicular_norm ** 2
    readout_rhs = query_norm * parallel_norm
    visibility = parallel_norm / error_norm if error_norm > 0.0 else 0.0
    rho = readout_norm / (error_norm * query_norm) if error_norm > 0.0 and query_norm > 0.0 else 0.0
    query_weighted_visibility = readout_norm / error_norm if error_norm > 0.0 else 0.0
    return {
        "parallel": parallel,
        "perpendicular": perpendicular,
        "readout": readout,
        "error_norm": error_norm,
        "parallel_norm": parallel_norm,
        "perpendicular_norm": perpendicular_norm,
        "readout_norm": readout_norm,
        "query_norm": query_norm,
        "visibility": visibility,
        "rho": rho,
        "query_weighted_visibility": query_weighted_visibility,
        "reconstruction_relative_error": reconstruction_error / (error_norm + eps),
        "orthogonal_energy_relative_error": abs(energy_lhs - energy_rhs) / (abs(energy_lhs) + eps),
        "readout_identity_relative_error": abs(readout_norm - readout_rhs) / (abs(readout_norm) + eps),
    }


def atomic_query_geometry_rows(error, query):
    if error.ndim != 4 or query.ndim != 3 or error.shape[:-1] != query.shape:
        raise ValueError("incompatible error/query shapes")
    rows = []
    for batch in range(error.shape[0]):
        for head in range(error.shape[1]):
            out = query_projection(error[batch : batch + 1, head : head + 1], query[batch : batch + 1, head : head + 1])
            rows.append({
                "batch": batch,
                "head": head,
                "error_norm": out["error_norm"],
                "parallel_norm": out["parallel_norm"],
                "perpendicular_norm": out["perpendicular_norm"],
                "visibility": out["visibility"],
                "rho": out["rho"],
                "readout_norm": out["readout_norm"],
                "query_norm": out["query_norm"],
                "reconstruction_relative_error": out["reconstruction_relative_error"],
                "orthogonal_energy_relative_error": out["orthogonal_energy_relative_error"],
                "readout_identity_relative_error": out["readout_identity_relative_error"],
                "visibility_rho_abs_diff": abs(out["visibility"] - out["rho"]),
            })
    return rows


def assert_same_query_provenance(native_query, rotated_query, captured_query):
    same = (
        native_query.shape == rotated_query.shape == captured_query.shape
        and native_query.dtype == rotated_query.dtype == captured_query.dtype
        and torch.equal(native_query.detach().cpu(), rotated_query.detach().cpu())
        and torch.equal(native_query.detach().cpu(), captured_query.detach().cpu())
    )
    max_diff = max(
        float((native_query.detach().double().cpu() - rotated_query.detach().double().cpu()).abs().max().item()),
        float((native_query.detach().double().cpu() - captured_query.detach().double().cpu()).abs().max().item()),
    )
    return {"gate": "PASS" if same else "FAIL", "max_abs_diff": max_diff}


def readout_ratio_factorization(native_metrics, rotated_metrics, eps=EPS):
    state_ratio = rotated_metrics["error_norm"] / native_metrics["error_norm"] if native_metrics["error_norm"] > 0.0 else 0.0
    visibility_ratio = rotated_metrics["visibility"] / native_metrics["visibility"] if native_metrics["visibility"] > 0.0 else 0.0
    readout_ratio = rotated_metrics["readout_norm"] / native_metrics["readout_norm"] if native_metrics["readout_norm"] > 0.0 else 0.0
    factorized = state_ratio * visibility_ratio
    return {
        "state_ratio": state_ratio,
        "visibility_ratio": visibility_ratio,
        "readout_ratio": readout_ratio,
        "factorized_readout_ratio": factorized,
        "relative_error": abs(readout_ratio - factorized) / (abs(readout_ratio) + eps),
    }


def pr_readout_decomposition(propagated, fresh, error, query, eps=EPS):
    p = torch.einsum("bhkv,bhk->bhv", propagated.detach().double(), query.detach().double())
    r = torch.einsum("bhkv,bhk->bhv", fresh.detach().double(), query.detach().double())
    e = torch.einsum("bhkv,bhk->bhv", error.detach().double(), query.detach().double())
    p_norm = tensor_norm(p)
    r_norm = tensor_norm(r)
    e_norm = tensor_norm(e)
    dot = float((p * r).sum().item())
    pred_energy = p_norm ** 2 + r_norm ** 2 + 2.0 * dot
    return {
        "p": p,
        "r": r,
        "e": e,
        "Pq_norm": p_norm,
        "Rq_norm": r_norm,
        "Eq_norm": e_norm,
        "dot_Pq_Rq": dot,
        "cos_Pq_Rq": dot / (p_norm * r_norm + eps) if p_norm > 0.0 and r_norm > 0.0 else None,
        "normalized_interference": 2.0 * dot / (p_norm ** 2 + r_norm ** 2 + eps),
        "vector_identity_relative_error": tensor_norm(e - p - r) / (e_norm + eps),
        "energy_identity_relative_error": abs(e_norm ** 2 - pred_energy) / (abs(e_norm ** 2) + eps),
    }


def pr_state_decomposition_fp64(fp_state, pre_quant_state, quantized_state):
    fp = fp_state.detach().double()
    pre = pre_quant_state.detach().double()
    quantized = quantized_state.detach().double()
    propagated = pre - fp
    fresh = quantized - pre
    return {"P": propagated, "R": fresh, "E": propagated + fresh}


def scalar_geometry_metrics(projection, pr_metrics):
    return {
        "error_norm": projection["error_norm"],
        "parallel_norm": projection["parallel_norm"],
        "perpendicular_norm": projection["perpendicular_norm"],
        "readout_norm": projection["readout_norm"],
        "query_norm": projection["query_norm"],
        "visibility": projection["visibility"],
        "rho": projection["rho"],
        "reconstruction_relative_error": projection["reconstruction_relative_error"],
        "orthogonal_energy_relative_error": projection["orthogonal_energy_relative_error"],
        "readout_identity_relative_error": projection["readout_identity_relative_error"],
        "visibility_rho_abs_diff": abs(projection["visibility"] - projection["rho"]),
        "Pq_norm": pr_metrics["Pq_norm"],
        "Rq_norm": pr_metrics["Rq_norm"],
        "Eq_norm": pr_metrics["Eq_norm"],
        "dot_Pq_Rq": pr_metrics["dot_Pq_Rq"],
        "cos_Pq_Rq": pr_metrics["cos_Pq_Rq"],
        "normalized_interference": pr_metrics["normalized_interference"],
        "pr_vector_identity_relative_error": pr_metrics["vector_identity_relative_error"],
        "pr_energy_identity_relative_error": pr_metrics["energy_identity_relative_error"],
    }


def aggregate_atomic_metrics(rows):
    if not rows:
        raise ValueError("cannot aggregate empty atomic metrics")

    def root_sum_sq(name):
        return math.sqrt(sum(float(row[name]) ** 2 for row in rows))

    error_norm = root_sum_sq("error_norm")
    parallel_norm = root_sum_sq("parallel_norm")
    perpendicular_norm = root_sum_sq("perpendicular_norm")
    readout_norm = root_sum_sq("readout_norm")
    p_norm = root_sum_sq("Pq_norm")
    r_norm = root_sum_sq("Rq_norm")
    e_norm = root_sum_sq("Eq_norm")
    dot = sum(float(row["dot_Pq_Rq"]) for row in rows)
    qnorms = [float(row["query_norm"]) for row in rows]
    query_norm = statistics.median(qnorms)
    visibility = parallel_norm / error_norm if error_norm > 0.0 else 0.0
    rho = readout_norm / (error_norm * query_norm) if error_norm > 0.0 and query_norm > 0.0 else 0.0
    query_weighted_visibility = readout_norm / error_norm if error_norm > 0.0 else 0.0
    return {
        "n_atomic_samples": len(rows),
        "error_norm": error_norm,
        "parallel_norm": parallel_norm,
        "perpendicular_norm": perpendicular_norm,
        "readout_norm": readout_norm,
        "query_norm": query_norm,
        "query_norm_min": min(qnorms),
        "query_norm_max": max(qnorms),
        "query_norm_relative_span": (max(qnorms) - min(qnorms)) / (abs(query_norm) + EPS),
        "visibility": visibility,
        "rho": rho,
        "query_weighted_visibility": query_weighted_visibility,
        "visibility_rho_abs_diff": abs(visibility - rho),
        "Pq_norm": p_norm,
        "Rq_norm": r_norm,
        "Eq_norm": e_norm,
        "dot_Pq_Rq": dot,
        "cos_Pq_Rq": dot / (p_norm * r_norm) if p_norm > 0.0 and r_norm > 0.0 else None,
        "normalized_interference": 2.0 * dot / (p_norm ** 2 + r_norm ** 2 + EPS),
        "pr_energy_identity_relative_error": abs(e_norm ** 2 - (p_norm ** 2 + r_norm ** 2 + 2.0 * dot)) / (e_norm ** 2 + EPS),
        "max_projection_reconstruction_relative_error": max(float(row["reconstruction_relative_error"]) for row in rows),
        "max_projection_energy_relative_error": max(float(row["orthogonal_energy_relative_error"]) for row in rows),
        "max_readout_identity_relative_error": max(float(row["readout_identity_relative_error"]) for row in rows),
        "max_visibility_rho_abs_diff": max(float(row["visibility_rho_abs_diff"]) for row in rows),
        "max_pr_vector_identity_relative_error": max(float(row["pr_vector_identity_relative_error"]) for row in rows),
        "max_pr_energy_identity_relative_error": max(float(row["pr_energy_identity_relative_error"]) for row in rows),
    }


def pair_branch_metrics(native, rotated):
    state_ratio = rotated["error_norm"] / native["error_norm"] if native["error_norm"] > 0.0 else 0.0
    parallel_ratio = rotated["parallel_norm"] / native["parallel_norm"] if native["parallel_norm"] > 0.0 else 0.0
    visibility_ratio = rotated["visibility"] / native["visibility"] if native["visibility"] > 0.0 else 0.0
    query_weighted_visibility_ratio = (
        rotated["query_weighted_visibility"] / native["query_weighted_visibility"]
        if native["query_weighted_visibility"] > 0.0 else 0.0
    )
    readout_ratio = rotated["readout_norm"] / native["readout_norm"] if native["readout_norm"] > 0.0 else 0.0
    factorized = state_ratio * query_weighted_visibility_ratio
    geometric_factorized = state_ratio * visibility_ratio
    pq_ratio = rotated["Pq_norm"] / native["Pq_norm"] if native["Pq_norm"] > 0.0 else 0.0
    rq_ratio = rotated["Rq_norm"] / native["Rq_norm"] if native["Rq_norm"] > 0.0 else 0.0
    return {
        "state_ratio": state_ratio,
        "parallel_ratio": parallel_ratio,
        "visibility_ratio": visibility_ratio,
        "query_weighted_visibility_ratio": query_weighted_visibility_ratio,
        "readout_ratio": readout_ratio,
        "factorized_readout_ratio": factorized,
        "factorization_relative_error": abs(readout_ratio - factorized) / (abs(readout_ratio) + EPS),
        "geometric_factorized_readout_ratio": geometric_factorized,
        "geometric_factorization_relative_error": abs(readout_ratio - geometric_factorized) / (abs(readout_ratio) + EPS),
        "reversal_visibility_threshold": 1.0 / state_ratio if state_ratio > 0.0 else None,
        "reversal_threshold_holds": bool(state_ratio > 0.0 and query_weighted_visibility_ratio > 1.0 / state_ratio),
        "G_state": 1.0 - state_ratio,
        "G_parallel": 1.0 - parallel_ratio,
        "G_read": 1.0 - readout_ratio,
        "G_Pq": 1.0 - pq_ratio,
        "G_Rq": 1.0 - rq_ratio,
        "native_Iq": native["normalized_interference"],
        "rotated_Iq": rotated["normalized_interference"],
        "delta_Iq": rotated["normalized_interference"] - native["normalized_interference"],
        "horizon_classification": classify_horizon(state_ratio, query_weighted_visibility_ratio, readout_ratio),
    }


def classify_horizon(state_ratio, visibility_ratio, readout_ratio, tol=1e-10):
    if readout_ratio > 1.0 + tol and visibility_ratio > 1.0:
        return "VISIBILITY_REVERSES_READOUT_GAIN"
    if abs(readout_ratio - 1.0) <= tol and visibility_ratio >= 1.0:
        return "VISIBILITY_CANCELS_MAGNITUDE_GAIN"
    if visibility_ratio > 1.0:
        return "VISIBILITY_INCREASE_PARTIAL"
    return "MAGNITUDE_DOMINANT"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def shell_output(command):
    try:
        return subprocess.check_output(command, cwd=str(REPO), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def append_csv(path, rows, fieldnames):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def finite_values(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean(values):
    values = finite_values(values)
    return sum(values) / len(values) if values else None


def median(values):
    values = finite_values(values)
    return statistics.median(values) if values else None


def std(values):
    values = finite_values(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def bootstrap_mean_ci(values, samples=10000, seed=BOOTSTRAP_SEED):
    values = finite_values(values)
    if not values:
        return None
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(samples)):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    draws.sort()
    return [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]


def scalar_atom_row(error, propagated, fresh, query):
    projection = query_projection(error, query)
    pr_metrics = pr_readout_decomposition(propagated, fresh, error, query)
    return scalar_geometry_metrics(projection, pr_metrics)


def validate_atomic_identities(rows, geometry_tol=GEOMETRY_IDENTITY_TOL, pr_tol=PR_IDENTITY_TOL):
    checks = {
        "projection_reconstruction": max((row["reconstruction_relative_error"] for row in rows), default=0.0),
        "projection_energy": max((row["orthogonal_energy_relative_error"] for row in rows), default=0.0),
        "readout_norm": max((row["readout_identity_relative_error"] for row in rows), default=0.0),
        "visibility_rho": max((row["visibility_rho_abs_diff"] for row in rows), default=0.0),
        "pr_vector": max((row["pr_vector_identity_relative_error"] for row in rows), default=0.0),
        "pr_energy": max((row["pr_energy_identity_relative_error"] for row in rows), default=0.0),
    }
    geometry_max = max(checks[key] for key in ["projection_reconstruction", "projection_energy", "readout_norm", "visibility_rho"])
    pr_max = max(checks[key] for key in ["pr_vector", "pr_energy"])
    checks["gate"] = "PASS" if geometry_max <= float(geometry_tol) and pr_max <= float(pr_tol) else "FAIL"
    return checks


def replay_quantized_geometry(tape, replay_impl, branch, rotation):
    ex = exact()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {
        layer: ex.transform_state_to_branch(value, "gdn", branch, rotation).to(device)
        for layer, value in tape["state0"].items()
    }
    all_atoms = []
    all_pr = []
    horizon_summaries = []
    query_provenance = []
    for step in tape["steps"]:
        horizon_atoms = []
        for layer, rec in sorted(step["records"].items()):
            _core, pre = replay_impl.replay(
                rec,
                states[layer],
                rotated=(branch == "rotated"),
                rotation=rotation,
            )
            quantized = ex.quantize_branch_state(pre, "gdn", branch, quantizer_off=False)
            states[layer] = quantized
            native_pre = ex.inverse_branch_state(pre.detach().cpu(), "gdn", branch, rotation)
            native_quantized = ex.inverse_branch_state(quantized.detach().cpu(), "gdn", branch, rotation)
            decomposition = pr_state_decomposition_fp64(step["fp_states"][layer], native_pre, native_quantized)
            error = decomposition["E"]
            propagated = decomposition["P"]
            fresh = decomposition["R"]
            captured_query = step["readout_queries"][layer].detach().cpu()
            for batch in range(error.shape[0]):
                for head in range(error.shape[1]):
                    es = error[batch : batch + 1, head : head + 1]
                    ps = propagated[batch : batch + 1, head : head + 1]
                    rs = fresh[batch : batch + 1, head : head + 1]
                    qs = captured_query[batch : batch + 1, head : head + 1]
                    row = scalar_atom_row(es, ps, rs, qs)
                    metadata = {
                        "unit_id": tape["unit_id"],
                        "problem_id": tape["problem_id"],
                        "t0": tape["t0"],
                        "horizon": int(step["t"]),
                        "layer": int(layer),
                        "batch": int(batch),
                        "head": int(head),
                        "branch": branch,
                        "driver_tape_id": tape["driver_tape_sha256"],
                        "query_sha256": ex.tensor_hash(qs),
                    }
                    full = dict(metadata, **row)
                    horizon_atoms.append(full)
                    if int(step["t"]) in HORIZONS:
                        all_atoms.append(full)
                        all_pr.append({
                            **metadata,
                            "Pq_norm": row["Pq_norm"],
                            "Rq_norm": row["Rq_norm"],
                            "Eq_norm": row["Eq_norm"],
                            "dot_Pq_Rq": row["dot_Pq_Rq"],
                            "cos_Pq_Rq": row["cos_Pq_Rq"],
                            "normalized_interference": row["normalized_interference"],
                            "vector_identity_relative_error": row["pr_vector_identity_relative_error"],
                            "energy_identity_relative_error": row["pr_energy_identity_relative_error"],
                        })
                    query_provenance.append((int(step["t"]), int(layer), int(batch), int(head), ex.tensor_hash(qs)))
        gate = validate_atomic_identities(horizon_atoms)
        if gate["gate"] != "PASS":
            raise RuntimeError(f"geometry identity gate failed unit={tape['unit_id']} horizon={step['t']} checks={gate}")
        aggregate = aggregate_atomic_metrics(horizon_atoms)
        aggregate.update({
            "unit_id": tape["unit_id"],
            "problem_id": tape["problem_id"],
            "t0": tape["t0"],
            "horizon": int(step["t"]),
            "branch": branch,
            "driver_tape_id": tape["driver_tape_sha256"],
        })
        horizon_summaries.append(aggregate)
    return {
        "atoms": all_atoms,
        "pr_rows": all_pr,
        "horizons": horizon_summaries,
        "query_provenance": query_provenance,
    }


FACTORIZATION_FIELDS = [
    "unit_id", "horizon", "layer", "batch", "head", "driver_tape_id", "query_sha256",
    "state_ratio", "visibility_ratio", "readout_ratio", "factorized_readout_ratio", "relative_error",
]


def atomic_factorization_rows(native_atoms, rotated_atoms):
    if len(native_atoms) != len(rotated_atoms):
        raise RuntimeError("INVALID_QUERY_GEOMETRY_COMPARISON: unequal atomic row counts")
    rows = []
    for native, rotated in zip(native_atoms, rotated_atoms):
        keys = ["unit_id", "horizon", "layer", "batch", "head", "driver_tape_id", "query_sha256"]
        if any(native[key] != rotated[key] for key in keys):
            raise RuntimeError(f"INVALID_QUERY_GEOMETRY_COMPARISON: atomic provenance mismatch {native.get('unit_id')}")
        factor = readout_ratio_factorization(native, rotated)
        rows.append({key: native[key] for key in keys} | factor)
    return rows


def pair_unit_horizons(native_horizons, rotated_horizons):
    native_by_h = {int(row["horizon"]): row for row in native_horizons}
    rotated_by_h = {int(row["horizon"]): row for row in rotated_horizons}
    rows = []
    for horizon in sorted(set(native_by_h) & set(rotated_by_h)):
        native = native_by_h[horizon]
        rotated = rotated_by_h[horizon]
        if native["driver_tape_id"] != rotated["driver_tape_id"]:
            raise RuntimeError("INVALID_QUERY_GEOMETRY_COMPARISON: driver tape mismatch")
        pair = pair_branch_metrics(native, rotated)
        rows.append({
            "unit_id": native["unit_id"],
            "problem_id": native["problem_id"],
            "t0": native["t0"],
            "horizon": horizon,
            "driver_tape_id": native["driver_tape_id"],
            **{f"native_{key}": value for key, value in native.items() if key not in {"unit_id", "problem_id", "t0", "horizon", "branch", "driver_tape_id"}},
            **{f"rotated_{key}": value for key, value in rotated.items() if key not in {"unit_id", "problem_id", "t0", "horizon", "branch", "driver_tape_id"}},
            **pair,
        })
    return rows


def aggregate_horizons(unit_rows, horizons, bootstrap_samples=10000):
    output = []
    for horizon in horizons:
        rows = [row for row in unit_rows if int(row["horizon"]) == int(horizon)]
        if not rows:
            continue
        summary = {"horizon": int(horizon), "n_units": len(rows)}
        for key in [
            "G_state", "G_parallel", "G_read", "visibility_ratio", "query_weighted_visibility_ratio", "delta_Iq", "G_Pq", "G_Rq",
            "state_ratio", "parallel_ratio", "readout_ratio", "factorized_readout_ratio",
        ]:
            values = [row[key] for row in rows]
            summary[f"{key}_median"] = median(values)
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = std(values)
            summary[f"{key}_bootstrap95_mean"] = bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                seed=BOOTSTRAP_SEED + int(horizon),
            )
            if key in {"visibility_ratio", "query_weighted_visibility_ratio"}:
                summary[f"{key}_wins"] = sum(float(value) > 1.0 for value in values)
            elif key in {"state_ratio", "parallel_ratio", "readout_ratio", "factorized_readout_ratio"}:
                summary[f"{key}_wins"] = sum(float(value) < 1.0 for value in values)
            else:
                summary[f"{key}_wins"] = sum(float(value) > 0.0 for value in values)
        for branch in ["native", "rotated"]:
            for key in [
                "error_norm", "parallel_norm", "perpendicular_norm", "visibility", "rho", "query_weighted_visibility", "readout_norm",
                "query_norm", "Pq_norm", "Rq_norm", "normalized_interference",
            ]:
                summary[f"{branch}_{key}_median"] = median([row[f"{branch}_{key}"] for row in rows])
                summary[f"{branch}_{key}_mean"] = mean([row[f"{branch}_{key}"] for row in rows])
        summary["reversal_threshold_holds"] = sum(bool(row["reversal_threshold_holds"]) for row in rows)
        summary["state_smaller_visibility_higher_readout_worse"] = sum(
            row["state_ratio"] < 1.0 and row["visibility_ratio"] > 1.0 and row["readout_ratio"] > 1.0
            for row in rows
        )
        summary["factorization_relative_error_max"] = max(row["factorization_relative_error"] for row in rows)
        output.append(summary)
    return output


def classify_results(unit_rows, horizon_rows, previous_summary):
    by_h = {int(row["horizon"]): row for row in horizon_rows}
    t64 = by_h.get(64, {})
    unit64 = [row for row in unit_rows if int(row["horizon"]) == 64]
    previous_source = previous_summary.get("median_fresh_source_gain")
    state_gain = t64.get("G_state_median")
    readout_gain = t64.get("G_read_median")
    previous_state = previous_summary.get("t64_median_G_E")
    previous_readout = previous_summary.get("t64_median_G_read")
    reproduced = bool(
        state_gain is not None
        and readout_gain is not None
        and state_gain > 0.05
        and readout_gain < 0.0
        and (previous_state is None or abs(state_gain - previous_state) <= 0.05)
        and (previous_readout is None or abs(readout_gain - previous_readout) <= 0.05)
    )
    if state_gain is None or state_gain <= 0.0:
        magnitude = "LOST"
    elif previous_source is not None and state_gain < 0.5 * float(previous_source):
        magnitude = "ATTENUATED"
    else:
        magnitude = "PRESERVED"
    visibility_ratio = t64.get("visibility_ratio_median")
    visibility_ci = t64.get("visibility_ratio_bootstrap95_mean") or [None, None]
    if visibility_ratio is None:
        visibility_shift = "MIXED"
    elif visibility_ratio > 1.0 and visibility_ci[0] is not None and visibility_ci[0] > 1.0:
        visibility_shift = "INCREASED"
    elif visibility_ratio < 1.0 and visibility_ci[1] is not None and visibility_ci[1] < 1.0:
        visibility_shift = "DECREASED"
    elif abs(visibility_ratio - 1.0) <= 0.02:
        visibility_shift = "UNCHANGED"
    else:
        visibility_shift = "MIXED"
    closure_wins = sum(
        row["state_ratio"] < 1.0
        and row["query_weighted_visibility_ratio"] > 1.0 / row["state_ratio"]
        and row["readout_ratio"] > 1.0
        for row in unit64
        if row["state_ratio"] > 0.0
    )
    max_factor_error = max((row["factorization_relative_error"] for row in unit_rows), default=float("inf"))
    if reproduced and visibility_shift == "INCREASED" and closure_wins >= 13 and max_factor_error <= FACTORIZATION_TOL:
        visibility_explains = "SUPPORTED"
        closure = "SUPPORTED"
    elif reproduced and visibility_ratio is not None and visibility_ratio > 1.0 and closure_wins >= 9:
        visibility_explains = "PARTIAL"
        closure = "PARTIAL"
    else:
        visibility_explains = "NOT_SUPPORTED"
        closure = "NOT_SUPPORTED"
    delta_i = t64.get("delta_Iq_median")
    delta_i_ci = t64.get("delta_Iq_bootstrap95_mean") or [None, None]
    if delta_i is not None and delta_i > 0.05 and delta_i_ci[0] is not None and delta_i_ci[0] > 0.0:
        interaction = "MORE_CONSTRUCTIVE"
    elif delta_i is not None and delta_i < -0.05 and delta_i_ci[1] is not None and delta_i_ci[1] < 0.0:
        interaction = "MORE_CANCELLING"
    else:
        interaction = "NO_CLEAR_CHANGE"
    gpq = t64.get("G_Pq_median")
    grq = t64.get("G_Rq_median")
    if visibility_explains == "SUPPORTED" and interaction == "NO_CLEAR_CHANGE":
        primary = "TOTAL_QUERY_ALIGNMENT"
    elif gpq is not None and gpq < 0.0 and (grq is None or grq >= 0.0) and interaction == "NO_CLEAR_CHANGE":
        primary = "P_PROPAGATED"
    elif grq is not None and grq < 0.0 and (gpq is None or gpq >= 0.0) and interaction == "NO_CLEAR_CHANGE":
        primary = "R_FRESH"
    elif interaction == "MORE_CONSTRUCTIVE" and (gpq is None or gpq >= 0.0) and (grq is None or grq >= 0.0):
        primary = "P_R_INTERACTION"
    elif any(value is not None and value < 0.0 for value in [gpq, grq]) or interaction != "NO_CLEAR_CHANGE":
        primary = "MULTI_FACTOR"
    else:
        primary = "UNRESOLVED"
    if closure == "SUPPORTED":
        failure = "YES"
        final = "ACCUMULATED_QUERY_VISIBILITY_EXPLAINS_GDN_READOUT_REVERSAL"
        next_experiment = "GDN_QUERY_GEOMETRY_AWARE_STATE_ROTATION_V1"
    elif primary == "P_PROPAGATED":
        failure = "PARTIAL"
        final = "PROPAGATED_READOUT_ERROR_GEOMETRY_CANDIDATE"
        next_experiment = "GDN_PROPAGATED_ERROR_GEOMETRY_CAUSAL_V1"
    elif primary == "R_FRESH":
        failure = "PARTIAL"
        final = "REQUANTIZATION_READOUT_ERROR_GEOMETRY_CANDIDATE"
        next_experiment = "GDN_REQUANTIZATION_READOUT_GEOMETRY_V1"
    elif primary == "P_R_INTERACTION":
        failure = "PARTIAL"
        final = "READOUT_SPACE_P_R_INTERACTION_CANDIDATE"
        next_experiment = "GDN_READOUT_INTERACTION_AWARE_ROTATION_V1"
    else:
        failure = "PARTIAL" if reproduced else "NO"
        final = "GDN_READOUT_REVERSAL_MECHANISM_PARTIAL" if reproduced else "GDN_READOUT_REVERSAL_NOT_REPRODUCED"
        next_experiment = "GDN_READOUT_REVERSAL_REASSESSMENT_V2"
    return {
        "GDN_FROZEN_DRIVER_PROVENANCE": "PASS",
        "GDN_STATE_READOUT_REVERSAL_REPRODUCED": "YES" if reproduced else "NO",
        "GDN_ACCUMULATED_STATE_MAGNITUDE_GAIN": magnitude,
        "GDN_QUERY_VISIBILITY_SHIFT": visibility_shift,
        "GDN_QUERY_VISIBILITY_EXPLAINS_READOUT_REVERSAL": visibility_explains,
        "GDN_READOUT_SPACE_P_COMPONENT_GAIN": gpq,
        "GDN_READOUT_SPACE_R_COMPONENT_GAIN": grq,
        "GDN_READOUT_SPACE_P_R_INTERACTION": interaction,
        "GDN_READOUT_REVERSAL_PRIMARY_COMPONENT": primary,
        "GDN_ACCUMULATED_ERROR_QUERY_GEOMETRY_CLOSURE": closure,
        "FAILURE_MECHANISM_IDENTIFIED": failure,
        "FINAL_CLASSIFICATION": final,
        "NEXT_EXPERIMENT": next_experiment,
        "t64_closure_wins": closure_wins,
        "max_unit_horizon_factorization_relative_error": max_factor_error,
    }


def make_plots(unit_rows, horizon_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": repr(exc), "plots": []}
    figure_dir = RESULT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    xs = [int(row["horizon"]) for row in horizon_rows]
    made = []

    def line_plot(filename, ylabel, series, zero=False):
        plt.figure(figsize=(7, 4))
        for label, values in series:
            plt.plot(xs, values, marker="o", label=label)
        if zero:
            plt.axhline(0.0, color="black", linewidth=1)
        plt.xscale("log", base=2)
        plt.xticks(xs, xs)
        plt.xlabel("horizon")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = figure_dir / filename
        plt.savefig(path, dpi=180)
        plt.close()
        made.append(str(path))

    line_plot("01_total_state_error.png", "total state error norm", [
        ("Native", [row["native_error_norm_median"] for row in horizon_rows]),
        ("Rotated", [row["rotated_error_norm_median"] for row in horizon_rows]),
    ])
    line_plot("02_query_visible_error.png", "query-visible error norm", [
        ("Native", [row["native_parallel_norm_median"] for row in horizon_rows]),
        ("Rotated", [row["rotated_parallel_norm_median"] for row in horizon_rows]),
    ])
    line_plot("03_visibility_fraction.png", "Vq", [
        ("Native", [row["native_visibility_median"] for row in horizon_rows]),
        ("Rotated", [row["rotated_visibility_median"] for row in horizon_rows]),
    ])
    line_plot("04_gain_panel.png", "gain", [
        ("state", [row["G_state_median"] for row in horizon_rows]),
        ("query-visible", [row["G_parallel_median"] for row in horizon_rows]),
        ("readout", [row["G_read_median"] for row in horizon_rows]),
    ], zero=True)
    line_plot("05_visibility_ratio_threshold.png", "ratio", [
        ("geometric Vq ratio", [row["visibility_ratio_median"] for row in horizon_rows]),
        ("query-weighted ratio", [row["query_weighted_visibility_ratio_median"] for row in horizon_rows]),
        ("reversal threshold", [1.0 / row["state_ratio_median"] for row in horizon_rows]),
    ])
    line_plot("06_readout_factorization.png", "readout ratio", [
        ("measured", [row["readout_ratio_median"] for row in horizon_rows]),
        ("state x visibility", [row["factorized_readout_ratio_median"] for row in horizon_rows]),
    ])
    line_plot("07_propagated_readout.png", "||P^T q||", [
        ("Native", [row["native_Pq_norm_median"] for row in horizon_rows]),
        ("Rotated", [row["rotated_Pq_norm_median"] for row in horizon_rows]),
    ])
    line_plot("08_fresh_readout.png", "||R^T q||", [
        ("Native", [row["native_Rq_norm_median"] for row in horizon_rows]),
        ("Rotated", [row["rotated_Rq_norm_median"] for row in horizon_rows]),
    ])
    line_plot("09_pr_readout_interference.png", "Iq", [
        ("Native", [row["native_normalized_interference_median"] for row in horizon_rows]),
        ("Rotated", [row["rotated_normalized_interference_median"] for row in horizon_rows]),
    ], zero=True)
    t64 = sorted([row for row in unit_rows if int(row["horizon"]) == 64], key=lambda row: row["unit_id"])
    plt.figure(figsize=(10, 4))
    positions = list(range(len(t64)))
    plt.bar(positions, [row["visibility_ratio"] - 1.0 for row in t64], color=["#d95f02" if row["visibility_ratio"] > 1.0 else "#1b9e77" for row in t64])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(positions, [str(index + 1) for index in positions])
    plt.xlabel("canonical unit")
    plt.ylabel("visibility ratio - 1")
    plt.tight_layout()
    unit_plot = figure_dir / "10_t64_visibility_by_unit.png"
    plt.savefig(unit_plot, dpi=180)
    plt.close()
    made.append(str(unit_plot))
    save_json(figure_dir / "manifest.json", {"status": "OK", "plots": made})
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
    classifications = summary["classifications"]
    keys = [
        "GDN_FROZEN_DRIVER_PROVENANCE",
        "GDN_STATE_READOUT_REVERSAL_REPRODUCED",
        "GDN_ACCUMULATED_STATE_MAGNITUDE_GAIN",
        "GDN_QUERY_VISIBILITY_SHIFT",
        "GDN_QUERY_VISIBILITY_EXPLAINS_READOUT_REVERSAL",
        "GDN_READOUT_SPACE_P_COMPONENT_GAIN",
        "GDN_READOUT_SPACE_R_COMPONENT_GAIN",
        "GDN_READOUT_SPACE_P_R_INTERACTION",
        "GDN_READOUT_REVERSAL_PRIMARY_COMPONENT",
        "GDN_ACCUMULATED_ERROR_QUERY_GEOMETRY_CLOSURE",
        "FAILURE_MECHANISM_IDENTIFIED",
        "FINAL_CLASSIFICATION",
        "NEXT_EXPERIMENT",
    ]
    lines = ["# GDN Rotation Accumulated-Error Query Geometry Closure V1", "", "## Final Classification", "", "```text"]
    lines.extend(f"{key} = {classifications.get(key)}" for key in keys)
    lines.extend(["```", "", "## Evidence Discipline", ""])
    lines.append("- OBSERVATION: horizon curves and canonical-unit paired summaries are measured from fresh exact replay.")
    lines.append("- EXACT ALGEBRAIC IDENTITY: projection, readout, ratio-factorization, and P/R identities are gated numerically.")
    lines.append("- CONTROLLED FROZEN-DRIVER RESULT: Native and Rotated branches consume the same recaptured FP driver tensors and query hashes.")
    lines.append("- SUPPORTED MECHANISM: used only when the prespecified unit-level closure and bootstrap criteria pass.")
    lines.append("- HYPOTHESIS: any proposed next experiment is not implemented or claimed as validated here.")
    lines.extend(["", "## Visibility Accounting", ""])
    lines.append(
        "- Geometric Vq = ||E_parallel|| / ||E|| is exact for each atomic layer/head query sample and is reported separately from pooled query-weighted visibility."
    )
    lines.append(
        "- Canonical units pool heterogeneous layer/head query norms, so unit-level readout-ratio factorization uses query-weighted visibility ||E^T q|| / ||E||."
    )
    lines.extend(["", "## Reproduction", "", "```json", json.dumps(summary["reproduction"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Horizon Results", "", "```json", json.dumps(summary["horizon_summary"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Identity Gates", "", "```json", json.dumps(summary["identity_gates"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Provenance", "", "```json", json.dumps(summary["provenance"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Environment", "", "```json", json.dumps(summary["environment"], indent=2, sort_keys=True), "```"])
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--scope", choices=["smoke", "pilot", "formal"], default="formal")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args(argv)


def run(args):
    global RESULT_DIR
    if args.horizon < 1 or args.horizon > 64:
        raise ValueError("horizon must be in [1,64]")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_paths = [
        RESULT_DIR / "accumulated_error_geometry.jsonl",
        RESULT_DIR / "query_projection_metrics.jsonl",
        RESULT_DIR / "pr_readout_decomposition.jsonl",
        RESULT_DIR / "factorization_check.csv",
        RESULT_DIR / "failure.json",
    ]
    for path in artifact_paths:
        if path.exists():
            path.unlink()
    ex = exact()
    previous_manifest = load_json(PRIOR_RESULT_DIR / "gdn" / "driver_capture" / "driver_tape_manifest.json", {}) or {}
    previous_tapes = {row["unit_id"]: row for row in previous_manifest.get("driver_tapes", [])}
    previous_summary = load_json(PRIOR_RESULT_DIR / "gdn" / "native_replay" / "summary.json", {}) or {}
    previous_parity = load_json(PRIOR_RESULT_DIR / "gdn" / "parity" / "summary.json", {}) or {}
    units_all, canonical_manifest_hash = ex.load_gdn_units()
    units = ex.select_units(units_all, args.scope, args.max_units)
    selected_horizons = [horizon for horizon in HORIZONS if horizon <= args.horizon]
    config = {
        "task": TASK,
        "created_at": now(),
        "args": vars(args),
        "architecture": "gdn",
        "state_semantics": "[B,H,K,V]",
        "rotation": {"axis": "key", "family": "RHT", "seed": RHT_SEED},
        "quantizer": "C128",
        "horizons": selected_horizons,
        "all_intermediate_steps_identity_checked": True,
        "detailed_rows_logged_at": selected_horizons,
        "formal_statistical_unit": "canonical recurrent unit",
        "geometry_identity_tolerance": GEOMETRY_IDENTITY_TOL,
        "pr_identity_tolerance": PR_IDENTITY_TOL,
        "factorization_tolerance": FACTORIZATION_TOL,
        "fp_replay_tolerance": FP_REPLAY_TOL,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "scope_restrictions": ["GDN only", "no new rotations", "no KDA", "no INT4", "no AIME/LongBench", "no kernels"],
    }
    save_json(RESULT_DIR / "config.json", config)
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
    unit_rows = []
    tape_rows = []
    parity_rows = []
    all_factor_max = 0.0
    identity_maxima = defaultdict(float)
    factor_fields = FACTORIZATION_FIELDS
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{now()}] GDN query geometry unit {index}/{len(units)} {unit['unit_id']}", flush=True)
            tape = ex.capture_driver_tape("gdn", unit, context, args.horizon)
            prior = previous_tapes.get(tape["unit_id"])
            if prior is None or prior.get("driver_tape_sha256") != tape["driver_tape_sha256"]:
                raise RuntimeError(
                    f"driver-tape provenance mismatch unit={tape['unit_id']} prior={None if prior is None else prior.get('driver_tape_sha256')} current={tape['driver_tape_sha256']}"
                )
            replay_impl = tape["capture"]
            fp_rows = ex.replay_fp_tape(tape, "gdn", replay_impl)
            max_fp = max(row["state_relative_error"] for row in fp_rows)
            if max_fp > FP_REPLAY_TOL or any(row["nonfinite"] for row in fp_rows):
                raise RuntimeError(f"FP replay gate failed unit={tape['unit_id']} max_rel={max_fp}")
            parity_rows.extend(fp_rows)
            rotation = ex.make_experiment_rotation("gdn", next(iter(tape["state0"].values())).shape[-2], "rht", seed=RHT_SEED, dtype=torch.float32)
            native = replay_quantized_geometry(tape, replay_impl, "native", rotation)
            rotated = replay_quantized_geometry(tape, replay_impl, "rotated", rotation)
            if native["query_provenance"] != rotated["query_provenance"]:
                raise RuntimeError("INVALID_QUERY_GEOMETRY_COMPARISON")
            factor_rows = atomic_factorization_rows(native["atoms"], rotated["atoms"])
            factor_max = max((row["relative_error"] for row in factor_rows), default=0.0)
            all_factor_max = max(all_factor_max, factor_max)
            if factor_max > FACTORIZATION_TOL:
                raise RuntimeError(f"atomic readout factorization failed unit={tape['unit_id']} max={factor_max}")
            pairs = pair_unit_horizons(native["horizons"], rotated["horizons"])
            if max((row["factorization_relative_error"] for row in pairs), default=0.0) > FACTORIZATION_TOL:
                raise RuntimeError(f"unit-horizon readout factorization failed unit={tape['unit_id']}")
            unit_rows.extend(row for row in pairs if int(row["horizon"]) in selected_horizons)
            tape_rows.append({
                "unit_id": tape["unit_id"],
                "problem_id": tape["problem_id"],
                "t0": tape["t0"],
                "driver_tape_sha256": tape["driver_tape_sha256"],
                "prior_driver_tape_sha256": prior["driver_tape_sha256"],
                "n_steps": len(tape["steps"]),
                "n_layers": len(tape["state0"]),
                "fp_replay_max_relative_error": max_fp,
                "query_provenance": "PASS",
            })
            for row in native["atoms"] + rotated["atoms"]:
                identity_maxima["projection_reconstruction"] = max(identity_maxima["projection_reconstruction"], row["reconstruction_relative_error"])
                identity_maxima["projection_energy"] = max(identity_maxima["projection_energy"], row["orthogonal_energy_relative_error"])
                identity_maxima["readout_norm"] = max(identity_maxima["readout_norm"], row["readout_identity_relative_error"])
                identity_maxima["visibility_rho"] = max(identity_maxima["visibility_rho"], row["visibility_rho_abs_diff"])
                identity_maxima["pr_vector"] = max(identity_maxima["pr_vector"], row["pr_vector_identity_relative_error"])
                identity_maxima["pr_energy"] = max(identity_maxima["pr_energy"], row["pr_energy_identity_relative_error"])
            append_jsonl_rows(RESULT_DIR / "accumulated_error_geometry.jsonl", native["atoms"] + rotated["atoms"])
            projection_keys = {
                "unit_id", "problem_id", "t0", "horizon", "layer", "batch", "head", "branch", "driver_tape_id", "query_sha256",
                "error_norm", "parallel_norm", "perpendicular_norm", "visibility", "rho", "readout_norm", "query_norm",
                "reconstruction_relative_error", "orthogonal_energy_relative_error", "readout_identity_relative_error", "visibility_rho_abs_diff",
            }
            append_jsonl_rows(
                RESULT_DIR / "query_projection_metrics.jsonl",
                [{key: value for key, value in row.items() if key in projection_keys} for row in native["atoms"] + rotated["atoms"]],
            )
            append_jsonl_rows(RESULT_DIR / "pr_readout_decomposition.jsonl", native["pr_rows"] + rotated["pr_rows"])
            append_csv(RESULT_DIR / "factorization_check.csv", factor_rows, factor_fields)
            print(
                f"[{now()}] PASS {tape['unit_id']} fp={max_fp:.3e} factor={factor_max:.3e} "
                f"t{args.horizon}_state_gain={pairs[-1]['G_state']:.6f} t{args.horizon}_readout_gain={pairs[-1]['G_read']:.6f}",
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
    horizon_rows = aggregate_horizons(unit_rows, selected_horizons, bootstrap_samples=args.bootstrap_samples)
    if args.scope == "formal" and 64 in selected_horizons:
        classifications = classify_results(unit_rows, horizon_rows, previous_summary)
        if classifications["GDN_STATE_READOUT_REVERSAL_REPRODUCED"] != "YES":
            raise RuntimeError("previous state/readout reversal was not reproduced")
    else:
        classifications = {
            "GDN_FROZEN_DRIVER_PROVENANCE": "PASS",
            "GDN_STATE_READOUT_REVERSAL_REPRODUCED": "NOT_EVALUATED_NONFORMAL",
            "GDN_ACCUMULATED_STATE_MAGNITUDE_GAIN": "NOT_EVALUATED_NONFORMAL",
            "GDN_QUERY_VISIBILITY_SHIFT": "NOT_EVALUATED_NONFORMAL",
            "GDN_QUERY_VISIBILITY_EXPLAINS_READOUT_REVERSAL": "NOT_EVALUATED_NONFORMAL",
            "GDN_READOUT_SPACE_P_COMPONENT_GAIN": None,
            "GDN_READOUT_SPACE_R_COMPONENT_GAIN": None,
            "GDN_READOUT_SPACE_P_R_INTERACTION": "NOT_EVALUATED_NONFORMAL",
            "GDN_READOUT_REVERSAL_PRIMARY_COMPONENT": "UNRESOLVED",
            "GDN_ACCUMULATED_ERROR_QUERY_GEOMETRY_CLOSURE": "NOT_EVALUATED_NONFORMAL",
            "FAILURE_MECHANISM_IDENTIFIED": "NO",
            "FINAL_CLASSIFICATION": "NONFORMAL_VALIDATION_ONLY",
            "NEXT_EXPERIMENT": "GDN_READOUT_REVERSAL_REASSESSMENT_V2",
        }
    by_h = {int(row["horizon"]): row for row in horizon_rows}
    reproduction = {
        "prior_t64_state_gain": previous_summary.get("t64_median_G_E"),
        "current_t64_state_gain": by_h.get(64, {}).get("G_state_median"),
        "prior_t64_readout_gain": previous_summary.get("t64_median_G_read"),
        "current_t64_readout_gain": by_h.get(64, {}).get("G_read_median"),
    }
    identity_gates = {
        **dict(identity_maxima),
        "atomic_ratio_factorization": all_factor_max,
        "unit_horizon_ratio_factorization": max((row["factorization_relative_error"] for row in unit_rows), default=0.0),
        "geometry_tolerance": GEOMETRY_IDENTITY_TOL,
        "pr_tolerance": PR_IDENTITY_TOL,
        "factorization_tolerance": FACTORIZATION_TOL,
        "gate": "PASS" if (
            max((identity_maxima[key] for key in ["projection_reconstruction", "projection_energy", "readout_norm", "visibility_rho"]), default=0.0) <= GEOMETRY_IDENTITY_TOL
            and max((identity_maxima[key] for key in ["pr_vector", "pr_energy"]), default=0.0) <= PR_IDENTITY_TOL
            and all_factor_max <= FACTORIZATION_TOL
        ) else "FAIL",
    }
    provenance = {
        "GDN_FP_DRIVER_TAPE_CAPTURE": previous_manifest.get("GDN_FP_DRIVER_TAPE_CAPTURE"),
        "GDN_FROZEN_DRIVER_FP_REPLAY": previous_parity.get("GDN_FROZEN_DRIVER_FP_REPLAY"),
        "current_driver_tape_hash_match": "PASS",
        "current_fp_replay": "PASS",
        "branch_query_source": "captured_fp_driver_tape_readout_queries",
        "driver_query_hashes_native_rotated": "PASS",
        "canonical_manifest": str(ex.GDN_CANONICAL_MANIFEST),
        "canonical_manifest_sha256": canonical_manifest_hash,
        "prior_result_dir": str(PRIOR_RESULT_DIR),
        "driver_tapes": tape_rows,
    }
    write_csv(RESULT_DIR / "query_projection_unit_aggregate.csv", unit_rows)
    write_csv(RESULT_DIR / "pr_readout_unit_aggregate.csv", unit_rows)
    write_csv(RESULT_DIR / "horizon_summary.csv", horizon_rows)
    save_json(RESULT_DIR / "manifest.json", {
        "task": TASK,
        "git_commit": shell_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": shell_output(["git", "status", "--short"]),
        "canonical_units_expected": expected_units,
        "canonical_units_completed": len(tape_rows),
        "canonical_unit_ids": [row["unit_id"] for row in tape_rows],
        "artifacts": [str(path) for path in [
            RESULT_DIR / "config.json", RESULT_DIR / "manifest.json", RESULT_DIR / "provenance.json",
            RESULT_DIR / "accumulated_error_geometry.jsonl", RESULT_DIR / "query_projection_metrics.jsonl",
            RESULT_DIR / "query_projection_unit_aggregate.csv", RESULT_DIR / "pr_readout_decomposition.jsonl",
            RESULT_DIR / "pr_readout_unit_aggregate.csv", RESULT_DIR / "horizon_summary.csv",
            RESULT_DIR / "factorization_check.csv", RESULT_DIR / "summary.json", RESULT_DIR / "report.md",
        ]],
    })
    save_json(RESULT_DIR / "provenance.json", provenance)
    plots = make_plots(unit_rows, horizon_rows)
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": shell_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": shell_output(["git", "status", "--short"]),
        "environment": environment_record(),
        "canonical_units_completed": len(tape_rows),
        "canonical_units_expected": expected_units,
        "reproduction": reproduction,
        "identity_gates": identity_gates,
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
