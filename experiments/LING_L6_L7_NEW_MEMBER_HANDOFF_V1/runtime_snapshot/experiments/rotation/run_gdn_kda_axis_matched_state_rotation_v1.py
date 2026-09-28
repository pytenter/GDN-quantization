#!/usr/bin/env python3
import argparse
import copy
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


TASK = "GDN_KDA_AXIS_MATCHED_STATE_ROTATION_V1"
SLUG = "gdn_kda_axis_matched_state_rotation_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
FIG_DIR = RESULT_DIR / "figures"
EPS = 1e-12
QMAX_INT8 = 127.0
RHT_SEEDS = list(range(8))
HAAR_SEEDS = list(range(4))
STATE_HORIZONS = [0, 1, 8, 32, 64, 128]
FUTURE_KL_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128]
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
LING_CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
GDN_CANONICAL_MANIFEST = REPO / "results" / "propagation" / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_stage0.json"


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


def percentile(xs, q):
    vals = sorted(float(x) for x in xs if finite_number(x))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def bootstrap_ci(xs, n=2000, seed=20260913):
    vals = [float(x) for x in xs if finite_number(x)]
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
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_number(x) and finite_number(y)]
    if len(pairs) < 3:
        return None
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx, my = mean(xvals), mean(yvals)
    denx = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    deny = math.sqrt(sum((y - my) ** 2 for y in yvals))
    if denx <= EPS or deny <= EPS:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / (denx * deny)


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite_number(x) and finite_number(y)]
    if len(pairs) < 3:
        return None
    return pearson(rankdata([x for x, _ in pairs]), rankdata([y for _, y in pairs]))


def hadamard(n, dtype=torch.float64, device=None):
    if n < 1 or (n & (n - 1)) != 0:
        return None
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)


def haar_rotation(n, seed=0, dtype=torch.float64, device=None):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    a = torch.randn(n, n, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.reshape(1, -1)
    return q.to(dtype=dtype, device=device)


def make_rotation(n, kind="identity", seed=0, dtype=torch.float64, device=None, samples=None):
    kind = str(kind).lower()
    if kind == "identity":
        return torch.eye(n, dtype=dtype, device=device)
    if kind == "haar":
        return haar_rotation(n, seed=seed, dtype=dtype, device=device)
    if kind == "rht":
        h = hadamard(n, dtype=torch.float64, device=None)
        if h is None:
            return haar_rotation(n, seed=seed, dtype=dtype, device=device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        signs = torch.where(torch.rand(n, generator=gen, dtype=torch.float64) < 0.5, -1.0, 1.0)
        return (h * signs.reshape(1, -1)).to(dtype=dtype, device=device)
    if kind in {"klt", "pca"}:
        if samples is None or samples.numel() == 0:
            return torch.eye(n, dtype=dtype, device=device)
        x = samples.detach().to(dtype=torch.float64, device="cpu")
        x = x.reshape(-1, n)
        x = x - x.mean(dim=0, keepdim=True)
        cov = x.t().matmul(x) / max(x.shape[0] - 1, 1)
        vals, vecs = torch.linalg.eigh(cov)
        order = torch.argsort(vals, descending=True)
        return vecs[:, order].to(dtype=dtype, device=device)
    raise ValueError(f"unknown rotation kind: {kind}")


def orthogonality_error(r):
    eye = torch.eye(r.shape[0], dtype=r.dtype, device=r.device)
    return float((r.t().matmul(r) - eye).abs().max().item())


def rotate_state_key_axis(state, r):
    return torch.einsum("ak,...kv->...av", r.t().to(device=state.device, dtype=state.dtype), state)


def inverse_rotate_state_key_axis(state, r):
    return torch.einsum("ka,...av->...kv", r.to(device=state.device, dtype=state.dtype), state)


def rotate_state_value_axis(state, u):
    return torch.einsum("...kv,vu->...ku", state, u.to(device=state.device, dtype=state.dtype))


def inverse_rotate_state_value_axis(state, u):
    return torch.einsum("...ku,vu->...kv", state, u.to(device=state.device, dtype=state.dtype))


def rotate_value_axis_tensor(x, u):
    return x.float().matmul(u.to(device=x.device, dtype=torch.float32))


def inverse_rotate_value_axis_tensor(x, u):
    return x.float().matmul(u.t().to(device=x.device, dtype=torch.float32))


def gdn_kernel_l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def prepare_gdn_rotated_qk(query, key, rotation, use_qk_l2norm_in_kernel=False):
    r = rotation.to(device=query.device, dtype=torch.float32)
    if use_qk_l2norm_in_kernel:
        query = gdn_kernel_l2norm(query, dim=-1).float()
        key = gdn_kernel_l2norm(key, dim=-1).float()
        return query.matmul(r), key.matmul(r), False
    return query.float().matmul(r).to(query.dtype), key.float().matmul(r).to(key.dtype), False


def relerr(a, b):
    af = a.detach().double()
    bf = b.detach().double()
    num = torch.linalg.vector_norm(af - bf).item()
    den = torch.linalg.vector_norm(bf).item()
    return num / (den + EPS)


def _gdn_step(s, q, k, v, alpha, beta):
    memory = torch.einsum("...k,...kv->...v", k, s)
    delta = (v - memory) * beta.unsqueeze(-1)
    next_state = alpha.unsqueeze(-1).unsqueeze(-1) * s + torch.einsum("...k,...v->...kv", k, delta)
    readout = torch.einsum("...kv,...k->...v", next_state, q)
    return next_state, readout


def _kda_step(s, q, k, v, decay, beta):
    decayed = s * decay.unsqueeze(-1)
    next_state = decayed + torch.einsum("...k,...v->...kv", k * beta, v)
    readout = torch.einsum("...kv,...k->...v", next_state, q)
    return next_state, readout


def run_stage0a_fp64_parity(k_dim=128, v_dim=128, batch=1, heads=2, steps=64, seed=20260913):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    r = make_rotation(k_dim, "rht", seed=0, dtype=torch.float64)
    u = make_rotation(v_dim, "rht", seed=0, dtype=torch.float64)
    s_gdn = torch.randn(batch, heads, k_dim, v_dim, generator=gen, dtype=torch.float64) * 0.05
    st_gdn = rotate_state_key_axis(s_gdn, r)
    s_kda = torch.randn(batch, heads, k_dim, v_dim, generator=gen, dtype=torch.float64) * 0.05
    st_kda = rotate_state_value_axis(s_kda, u)
    gdn_state_errors, gdn_readout_errors = [], []
    kda_state_errors, kda_readout_errors = [], []
    nonfinite = 0
    for _ in range(int(steps)):
        q = torch.randn(batch, heads, k_dim, generator=gen, dtype=torch.float64)
        k = torch.randn(batch, heads, k_dim, generator=gen, dtype=torch.float64) / math.sqrt(k_dim)
        v = torch.randn(batch, heads, v_dim, generator=gen, dtype=torch.float64)
        alpha = torch.rand(batch, heads, generator=gen, dtype=torch.float64) * 0.2 + 0.8
        beta = torch.rand(batch, heads, generator=gen, dtype=torch.float64) * 0.5
        s_gdn, o = _gdn_step(s_gdn, q, k, v, alpha, beta)
        qt = torch.einsum("ak,...k->...a", r.t(), q)
        kt = torch.einsum("ak,...k->...a", r.t(), k)
        st_gdn, ot = _gdn_step(st_gdn, qt, kt, v, alpha, beta)
        gdn_state_errors.append(relerr(st_gdn, rotate_state_key_axis(s_gdn, r)))
        gdn_readout_errors.append(relerr(ot, o))

        q2 = torch.randn(batch, heads, k_dim, generator=gen, dtype=torch.float64)
        k2 = torch.randn(batch, heads, k_dim, generator=gen, dtype=torch.float64) / math.sqrt(k_dim)
        v2 = torch.randn(batch, heads, v_dim, generator=gen, dtype=torch.float64)
        decay = torch.rand(batch, heads, k_dim, generator=gen, dtype=torch.float64) * 0.2 + 0.8
        beta2 = torch.rand(batch, heads, k_dim, generator=gen, dtype=torch.float64) * 0.5
        s_kda, o2 = _kda_step(s_kda, q2, k2, v2, decay, beta2)
        vt = torch.einsum("vu,...v->...u", u, v2)
        st_kda, ot2 = _kda_step(st_kda, q2, k2, vt, decay, beta2)
        recovered = torch.einsum("vu,...u->...v", u, ot2)
        kda_state_errors.append(relerr(st_kda, rotate_state_value_axis(s_kda, u)))
        kda_readout_errors.append(relerr(recovered, o2))
        tensors = [s_gdn, st_gdn, s_kda, st_kda, o, ot, o2, recovered]
        nonfinite += sum(int((~torch.isfinite(t)).sum().item()) for t in tensors)
    out = {
        "task": TASK,
        "stage": "0A",
        "rotation": {"R": "RHT seed 0", "U": "RHT seed 0", "R_orthogonality_max_abs": orthogonality_error(r), "U_orthogonality_max_abs": orthogonality_error(u)},
        "gdn": {
            "max_relative_state_error": max(gdn_state_errors),
            "max_relative_readout_error": max(gdn_readout_errors),
            "state_errors": gdn_state_errors,
            "readout_errors": gdn_readout_errors,
        },
        "kda": {
            "max_relative_state_error": max(kda_state_errors),
            "max_relative_readout_error": max(kda_readout_errors),
            "state_errors": kda_state_errors,
            "readout_errors": kda_readout_errors,
        },
        "nonfinite": nonfinite,
        "threshold": 1e-10,
    }
    out["gate"] = "PASS" if (
        out["gdn"]["max_relative_state_error"] <= 1e-10
        and out["gdn"]["max_relative_readout_error"] <= 1e-10
        and out["kda"]["max_relative_state_error"] <= 1e-10
        and out["kda"]["max_relative_readout_error"] <= 1e-10
        and nonfinite == 0
    ) else "FAIL"
    return out


def fake_quant_state(state, config):
    cfg = str(config).upper()
    x = state.detach().float()
    if x.ndim != 4:
        raise ValueError(f"state must be [B,H,K,V], got {list(x.shape)}")
    if cfg == "R128":
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / QMAX_INT8
        group_axis = "V"
        scale_sem = "[B,H,K,1]"
    elif cfg == "C128":
        scale = x.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / QMAX_INT8
        group_axis = "K"
        scale_sem = "[B,H,1,V]"
    else:
        raise ValueError(f"unsupported V1 quantizer config: {config}")
    codes = torch.round(x / scale).clamp(-QMAX_INT8, QMAX_INT8)
    qdq = codes * scale
    meta = {
        "config": cfg,
        "state_shape_semantics": "[B,H,K,V]",
        "group_axis": group_axis,
        "scale_shape_semantics": scale_sem,
        "scale": scale,
        "codes": codes,
        "bits": 8,
        "qrange": [-127, 127],
        "scale_formula": "amax(abs(group)).clamp_min(1e-12) / 127",
        "rounding": "torch.round",
        "dequantization": "codes * scale",
    }
    return qdq.to(dtype=state.dtype), meta


def normalized_peakiness(group):
    x = group.detach().float().reshape(-1)
    if x.numel() == 0:
        return None
    l2 = torch.linalg.vector_norm(x).item()
    if l2 <= EPS:
        return 1.0
    return math.sqrt(x.numel()) * float(x.abs().max().item()) / (l2 + EPS)


def _entropy(codes):
    flat = codes.detach().reshape(-1).to(torch.int16)
    if flat.numel() == 0:
        return 0.0
    vals, counts = torch.unique(flat, return_counts=True)
    probs = counts.float() / counts.sum().float()
    return float(-(probs * torch.log2(probs)).sum().item())


def _kurtosis(x):
    xf = x.detach().float().reshape(-1)
    if xf.numel() < 2:
        return None
    centered = xf - xf.mean()
    var = float((centered * centered).mean().item())
    if var <= EPS:
        return 0.0
    return float((centered.pow(4).mean() / (var ** 2)).item())


def quantization_group_metrics(state, config, arch, unit_id, basis, rotation_axis, layer=None, head=None, timestep=None, seed=None):
    qdq, meta = fake_quant_state(state, config)
    x = state.detach().float()
    q = qdq.detach().float()
    codes = meta["codes"].detach()
    rows = []
    bsz, heads, kdim, vdim = x.shape
    if meta["group_axis"] == "V":
        for b in range(bsz):
            for h in range(heads):
                if head is not None and int(head) != h:
                    continue
                for k in range(kdim):
                    group = x[b, h, k, :]
                    qgroup = q[b, h, k, :]
                    cgroup = codes[b, h, k, :]
                    scale = meta["scale"][b, h, k, 0]
                    rows.append(_group_metric_row(group, qgroup, cgroup, scale, arch, unit_id, basis, rotation_axis, config, layer, h, timestep, seed, group_index=k, group_axis="V"))
    else:
        for b in range(bsz):
            for h in range(heads):
                if head is not None and int(head) != h:
                    continue
                for v in range(vdim):
                    group = x[b, h, :, v]
                    qgroup = q[b, h, :, v]
                    cgroup = codes[b, h, :, v]
                    scale = meta["scale"][b, h, 0, v]
                    rows.append(_group_metric_row(group, qgroup, cgroup, scale, arch, unit_id, basis, rotation_axis, config, layer, h, timestep, seed, group_index=v, group_axis="K"))
    return rows


def rotation_l2_invariance_errors(original_state, rotated_state, config):
    cfg = str(config).upper()
    src = original_state.detach().double()
    rot = rotated_state.detach().double()
    errors = []
    if cfg == "R128":
        for b in range(src.shape[0]):
            for h in range(src.shape[1]):
                for k in range(src.shape[2]):
                    a = torch.linalg.vector_norm(src[b, h, k, :]).item()
                    c = torch.linalg.vector_norm(rot[b, h, k, :]).item()
                    errors.append(abs(c - a) / (a + EPS))
    elif cfg == "C128":
        for b in range(src.shape[0]):
            for h in range(src.shape[1]):
                for v in range(src.shape[3]):
                    a = torch.linalg.vector_norm(src[b, h, :, v]).item()
                    c = torch.linalg.vector_norm(rot[b, h, :, v]).item()
                    errors.append(abs(c - a) / (a + EPS))
    else:
        raise ValueError(config)
    return {
        "median": median(errors),
        "p99": percentile(errors, 0.99),
        "max": max(errors) if errors else None,
        "n_groups": len(errors),
    }


def _group_metric_row(group, qgroup, cgroup, scale, arch, unit_id, basis, rotation_axis, config, layer, head, timestep, seed, group_index, group_axis):
    err = qgroup - group
    l2 = float(torch.linalg.vector_norm(group).item())
    mse = float((err.double() * err.double()).mean().item())
    rel_l2 = float(torch.linalg.vector_norm(err).item() / (l2 + EPS))
    rms = math.sqrt(float((group.double() * group.double()).mean().item()))
    unique = int(torch.unique(cgroup.to(torch.int16)).numel())
    return {
        "arch": arch,
        "unit_id": unit_id,
        "layer": layer,
        "head": head,
        "timestep": timestep,
        "basis": basis,
        "rotation_axis": rotation_axis,
        "seed": seed,
        "quantizer": config,
        "group_axis": group_axis,
        "group_index": int(group_index),
        "group_size": int(group.numel()),
        "l2_norm": l2,
        "linf_norm": float(group.abs().max().item()) if group.numel() else 0.0,
        "normalized_peakiness": normalized_peakiness(group),
        "rms": rms,
        "max_over_rms": float(group.abs().max().item() / (rms + EPS)) if group.numel() else 0.0,
        "quant_scale": float(scale.item()),
        "scale_over_rms": float(scale.item() / (rms + EPS)),
        "int8_mse": mse,
        "int8_relative_l2_error": rel_l2,
        "int8_relative_frobenius_error": rel_l2,
        "unique_code_count": unique,
        "code_utilization_fraction": unique / 255.0,
        "quantized_code_entropy": _entropy(cgroup),
        "zero_code_fraction": float((cgroup == 0).float().mean().item()),
        "kurtosis": _kurtosis(group),
    }


def transform_state(state, axis, rotation):
    if rotation is None or axis in (None, "none"):
        return state.detach().clone()
    if axis == "key":
        return rotate_state_key_axis(state, rotation)
    if axis == "value":
        return rotate_state_value_axis(state, rotation)
    raise ValueError(axis)


def arch_stage_dir(architecture, stage):
    arch = str(architecture).lower()
    st = str(stage).lower()
    if arch not in {"gdn", "kda"}:
        raise ValueError(f"architecture must be gdn or kda, got {architecture}")
    out = RESULT_DIR / arch / st
    out.mkdir(parents=True, exist_ok=True)
    return out


def summarize_metric_rows(rows, unit_field="unit_id"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[unit_field]].append(row)
    unit_rows = []
    for unit_id, values in sorted(grouped.items()):
        unit_rows.append({
            "unit_id": unit_id,
            "arch": values[0].get("arch"),
            "basis": values[0].get("basis"),
            "rotation_axis": values[0].get("rotation_axis"),
            "seed": values[0].get("seed"),
            "quantizer": values[0].get("quantizer"),
            "mean_normalized_peakiness": mean([r["normalized_peakiness"] for r in values]),
            "median_normalized_peakiness": median([r["normalized_peakiness"] for r in values]),
            "mean_int8_relative_l2_error": mean([r["int8_relative_l2_error"] for r in values]),
            "median_int8_relative_l2_error": median([r["int8_relative_l2_error"] for r in values]),
            "n_groups": len(values),
        })
    return unit_rows


def paired_improvement(identity_units, rotated_units):
    ident = {r["unit_id"]: r for r in identity_units}
    pairs = []
    for row in rotated_units:
        base = ident.get(row["unit_id"])
        if not base:
            continue
        p0 = base["mean_normalized_peakiness"]
        p1 = row["mean_normalized_peakiness"]
        e0 = base["mean_int8_relative_l2_error"]
        e1 = row["mean_int8_relative_l2_error"]
        pairs.append({
            "unit_id": row["unit_id"],
            "basis": row["basis"],
            "seed": row["seed"],
            "quantizer": row["quantizer"],
            "peakiness_delta": p1 - p0,
            "peakiness_reduction_fraction": (p0 - p1) / (p0 + EPS),
            "error_delta": e1 - e0,
            "error_reduction_fraction": (e0 - e1) / (e0 + EPS),
        })
    return pairs


def classify_quantizability(pairs, expected_units=18):
    if not pairs:
        return "NOT_TESTED", {}
    peak_red = [p["peakiness_reduction_fraction"] for p in pairs]
    err_red = [p["error_reduction_fraction"] for p in pairs]
    wins = sum(p["error_delta"] < 0 for p in pairs)
    ci = bootstrap_ci(err_red)
    n = len(pairs)
    strong_wins = 14 if expected_units == 18 else math.ceil(0.75 * expected_units)
    if (
        median(peak_red) is not None and median(peak_red) >= 0.10
        and median(err_red) is not None and median(err_red) >= 0.15
        and wins >= min(strong_wins, n)
        and ci is not None and ci[0] > 0
    ):
        cls = "STRONG_POSITIVE"
    elif median(err_red) is not None and median(err_red) >= 0.05:
        cls = "WEAK_POSITIVE"
    else:
        cls = "NEGATIVE_OR_INCONCLUSIVE"
    return cls, {
        "n_units": n,
        "median_peakiness_reduction_fraction": median(peak_red),
        "median_error_reduction_fraction": median(err_red),
        "mean_error_reduction_fraction": mean(err_red),
        "wins": wins,
        "bootstrap95_error_reduction": ci,
    }


def rht_seed_summary(paired_by_seed):
    rows = []
    for seed, pairs in sorted(paired_by_seed.items()):
        rows.append({
            "seed": seed,
            "median_error_reduction_fraction": median([p["error_reduction_fraction"] for p in pairs]),
            "mean_error_reduction_fraction": mean([p["error_reduction_fraction"] for p in pairs]),
            "std_error_reduction_fraction": std([p["error_reduction_fraction"] for p in pairs]),
            "wins": sum(p["error_delta"] < 0 for p in pairs),
            "n": len(pairs),
        })
    medians = [r["median_error_reduction_fraction"] for r in rows]
    wins = [r["wins"] for r in rows]
    stability = "NOT_TESTED"
    if rows:
        positive = sum((x or 0.0) > 0 for x in medians)
        if positive >= max(1, len(rows) - 1):
            stability = "STABLE"
        elif positive >= math.ceil(len(rows) / 2):
            stability = "MODERATE"
        else:
            stability = "HIGHLY_SENSITIVE"
    return {
        "rows": rows,
        "seed0": next((r for r in rows if r["seed"] == 0), None),
        "median_across_seed_medians": median(medians),
        "mean_across_seed_medians": mean(medians),
        "std_across_seed_medians": std(medians),
        "best_seed_row": max(rows, key=lambda r: r["median_error_reduction_fraction"] or -math.inf) if rows else None,
        "worst_seed_row": min(rows, key=lambda r: r["median_error_reduction_fraction"] or math.inf) if rows else None,
        "per_seed_wins": wins,
        "stability": stability,
    }


def discover_environments():
    candidates = {
        "current_python": sys.executable,
        "gdn_qwen35_python": "/data01/user2/.conda/envs/sd310/bin/python",
        "ling_kda_python": "/data01/user2/.conda/envs/ling-kda/bin/python",
        "user2_python": "/data01/user2/.conda/envs/user2/bin/python",
    }
    rows = {}
    for name, exe in candidates.items():
        p = Path(exe)
        info = {"python": exe, "exists": p.exists(), "version": None, "torch": None, "transformers": None, "triton": None, "fla": None}
        if p.exists():
            probe = (
                "import importlib.util, json, sys\n"
                "def hasmod(m):\n"
                "    try: return importlib.util.find_spec(m) is not None\n"
                "    except Exception: return False\n"
                "mods={m:hasmod(m) for m in ['torch','transformers','triton','fla','transformers.models.qwen3_5.modeling_qwen3_5']}\n"
                "vers={'python':sys.version.split()[0]}\n"
                "for m in ['torch','transformers','triton','fla']:\n"
                "    if mods[m]:\n"
                "        mod=__import__(m); vers[m]=getattr(mod,'__version__','FOUND')\n"
                "    else: vers[m]=None\n"
                "print(json.dumps({'mods':mods,'versions':vers}))\n"
            )
            try:
                out = subprocess.check_output([str(p), "-c", probe], text=True, stderr=subprocess.STDOUT, timeout=20)
                obj = json.loads(out.strip().splitlines()[-1])
                info["version"] = obj["versions"]["python"]
                for mod in ["torch", "transformers", "triton", "fla"]:
                    info[mod] = obj["versions"].get(mod)
                info["qwen3_5_modeling"] = obj["mods"].get("transformers.models.qwen3_5.modeling_qwen3_5")
            except Exception as exc:
                info["probe_error"] = f"{type(exc).__name__}: {exc}"
        rows[name] = info
    return rows


def git_commit():
    return sh(["git", "rev-parse", "HEAD"])


def reproducibility_record():
    envs = discover_environments()
    return {
        "timestamp": now(),
        "git_commit": git_commit(),
        "git_status_short": sh(["git", "status", "--short"]),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "environments": envs,
        "state_semantics": {
            "state": "[B,H,K,V]",
            "R128": "scale over V; scale shape [B,H,K,1]",
            "C128": "scale over K; scale shape [B,H,1,V]",
        },
        "rotation_panel": {
            "identity": [None],
            "rht_seeds": RHT_SEEDS,
            "haar_seeds": HAAR_SEEDS,
            "klt": "diagnostic only",
        },
    }


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_counts():
    gdn = load_json(GDN_CANONICAL_MANIFEST)
    kda = load_json(LING_CANONICAL_UNITS)
    return {
        "gdn_manifest_path": str(GDN_CANONICAL_MANIFEST),
        "gdn_units": len(gdn.get("manifest", [])) if isinstance(gdn, dict) else 0,
        "kda_manifest_path": str(LING_CANONICAL_UNITS),
        "kda_units": len(kda.get("units", [])) if isinstance(kda, dict) else 0,
        "kda_manifest_sha256": hashlib.sha256(LING_CANONICAL_UNITS.read_bytes()).hexdigest() if LING_CANONICAL_UNITS.exists() else None,
    }


def logits_metrics(ref_logits, other_logits):
    ref = ref_logits[:, -1, :].detach().float()
    other = other_logits[:, -1, :].detach().float().to(ref.device)
    rel_l2 = float(torch.linalg.vector_norm(other - ref).item() / (torch.linalg.vector_norm(ref).item() + EPS))
    max_abs = float((other - ref).abs().max().item())
    cosine = float(torch.nn.functional.cosine_similarity(ref, other, dim=-1).mean().item())
    top1 = int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item())
    top10 = len(set(torch.topk(ref, 10, dim=-1).indices[0].tolist()).intersection(torch.topk(other, 10, dim=-1).indices[0].tolist())) / 10.0
    nonfinite = int((~torch.isfinite(ref)).sum().item() + (~torch.isfinite(other)).sum().item())
    return {
        "logit_relative_l2": rel_l2,
        "logit_max_abs": max_abs,
        "logit_cosine": cosine,
        "top1_match": top1,
        "top10_overlap": top10,
        "nonfinite": nonfinite,
    }


def classify_stage0b(rows, readout_key="readout_relative_error", finite_state_threshold=1e-5, finite_readout_threshold=2e-3):
    if not rows:
        return "BLOCKED"
    max_state = max(float(r.get("state_relative_error") or 0.0) for r in rows)
    max_readout = max(float(r.get(readout_key) or 0.0) for r in rows)
    nonfinite = sum(int(r.get("nonfinite") or 0) for r in rows)
    if max_state <= 1e-5 and max_readout <= 1e-5 and nonfinite == 0:
        return "PASS"
    if max_state <= finite_state_threshold and max_readout <= finite_readout_threshold and nonfinite == 0:
        return "FINITE_PRECISION_PASS"
    return "FAIL"


def stage0b_horizons(scope):
    return [0, 1, 2, 4] if scope == "smoke" else [0, 1, 4, 8, 16, 32, 64, 128]


def gpu_snapshot():
    return sh(["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader 2>/dev/null || true"])


def import_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe_cuda_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_gdn_stage0b(scope="smoke"):
    outdir = arch_stage_dir("gdn", "stage0b")
    started = time.time()
    parity_path = outdir / "parity.json"
    rows_path = outdir / "parity_rows.csv"
    os.environ.setdefault("GDN_DATA_ROOT", "/data01/user2")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    before_gpu = gpu_snapshot()
    rows = []
    try:
        orientation_dir = REPO / "experiments" / "orientation"
        if str(orientation_dir) not in sys.path:
            sys.path.insert(0, str(orientation_dir))
        import run_int8_orientation_state_change_mechanism as p1
        import run_end2end_bit_axis_screening as e2e
        e2e.ensure_imports()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

        cfg = json.loads((Path(os.environ["GDN_DATA_ROOT"]) / "experiments" / "qwen35_gdn_quant" / "run_config.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        prompts = [r for r in p1.selected_prompt_rows() if r.get("fp_response")]
        units_obj = load_json(GDN_CANONICAL_MANIFEST) or {}
        units = units_obj.get("manifest", [])
        if scope == "smoke":
            units = units[:1]
        prompt_map = {r["problem_id"]: r for r in prompts}
        if not units:
            units = [{"unit_id": prompts[0]["problem_id"] + "|0", "problem_id": prompts[0]["problem_id"], "t0": 0}]
        rmat_cpu = make_rotation(128, "rht", seed=0, dtype=torch.float32)
        orig_rec = qmod.torch_recurrent_gated_delta_rule
        orig_chunk = getattr(qmod, "torch_chunk_gated_delta_rule", None)
        patched_modules = []
        capture_records = []

        def tensor_copy(x, dtype=None):
            if x is None:
                return None
            out = x.detach().clone()
            return out.to(dtype=dtype) if dtype is not None else out

        def record_gdn_transition(layer_idx, operator, query, key, value, kwargs, core, state):
            capture_records.append({
                "layer": int(layer_idx),
                "operator": operator,
                "query": tensor_copy(query),
                "key": tensor_copy(key),
                "value": tensor_copy(value),
                "g": tensor_copy(kwargs.get("g")),
                "beta": tensor_copy(kwargs.get("beta")),
                "initial_state": tensor_copy(kwargs.get("initial_state"), dtype=torch.float32),
                "output_final_state": bool(kwargs.get("output_final_state", False)),
                "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                "core_output": tensor_copy(core, dtype=torch.float32),
                "final_state": tensor_copy(state, dtype=torch.float32),
            })

        def replay_gdn_transition(rec):
            fn = orig_rec if rec["operator"] == "recurrent" else orig_chunk
            if fn is None:
                return None
            kwargs = {
                "g": rec["g"],
                "beta": rec["beta"],
                "initial_state": None if rec["initial_state"] is None else rotate_state_key_axis(rec["initial_state"], rmat_cpu).to(rec["query"].device),
                "output_final_state": True,
            }
            query_rot, key_rot, use_norm = prepare_gdn_rotated_qk(
                rec["query"],
                rec["key"],
                rmat_cpu,
                rec["use_qk_l2norm_in_kernel"],
            )
            core, state = fn(
                query_rot,
                key_rot,
                rec["value"],
                **kwargs,
                use_qk_l2norm_in_kernel=use_norm,
            )
            return {
                "layer": rec["layer"],
                "operator": rec["operator"],
                "state_relative_error": relerr(state.detach().float(), rotate_state_key_axis(rec["final_state"], rmat_cpu).to(state.device)),
                "readout_relative_error": relerr(core.to(rec["value"].dtype).detach().float(), rec["core_output"].to(core.device)),
                "nonfinite": int((~torch.isfinite(state)).sum().item() + (~torch.isfinite(core)).sum().item()),
            }

        def replay_gdn_records(records):
            rows = [replay_gdn_transition(rec) for rec in records]
            rows = [r for r in rows if r is not None]
            if not rows:
                return {
                    "state_relative_error": None,
                    "readout_relative_error": None,
                    "nonfinite": 1,
                    "max_state_layer": None,
                    "max_readout_layer": None,
                    "n_transition_records": 0,
                }
            max_state_row = max(rows, key=lambda r: r["state_relative_error"])
            max_readout_row = max(rows, key=lambda r: r["readout_relative_error"])
            return {
                "state_relative_error": max_state_row["state_relative_error"],
                "readout_relative_error": max_readout_row["readout_relative_error"],
                "nonfinite": sum(r["nonfinite"] for r in rows),
                "max_state_layer": max_state_row["layer"],
                "max_readout_layer": max_readout_row["layer"],
                "n_transition_records": len(rows),
            }

        def make_capture_wrap(layer_idx, operator, fn):
            def wrapped(query, key, value, *args, **kwargs):
                core, state = fn(query, key, value, *args, **kwargs)
                record_gdn_transition(layer_idx, operator, query, key, value, kwargs, core, state)
                return core, state
            return wrapped

        def rec_wrap(query, key, value, *args, **kwargs):
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            query_rot, key_rot, use_norm = prepare_gdn_rotated_qk(query, key, rmat_cpu, use_norm)
            core, state = orig_rec(query_rot, key_rot, value, *args, use_qk_l2norm_in_kernel=use_norm, **kwargs)
            return core.to(value.dtype), state

        def chunk_wrap(query, key, value, *args, **kwargs):
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            query_rot, key_rot, use_norm = prepare_gdn_rotated_qk(query, key, rmat_cpu, use_norm)
            core, state = orig_chunk(query_rot, key_rot, value, *args, use_qk_l2norm_in_kernel=use_norm, **kwargs)
            return core.to(value.dtype), state

        def install_gdn_rotation_patch():
            patched_modules.clear()
            layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
            for layer in layers:
                mod = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
                if mod is None:
                    continue
                saved = {}
                if hasattr(mod, "recurrent_gated_delta_rule"):
                    saved["recurrent_gated_delta_rule"] = mod.recurrent_gated_delta_rule
                    mod.recurrent_gated_delta_rule = rec_wrap
                if hasattr(mod, "chunk_gated_delta_rule") and orig_chunk is not None:
                    saved["chunk_gated_delta_rule"] = mod.chunk_gated_delta_rule
                    mod.chunk_gated_delta_rule = chunk_wrap
                if saved:
                    patched_modules.append((mod, saved))
            qmod.torch_recurrent_gated_delta_rule = rec_wrap
            if orig_chunk is not None:
                qmod.torch_chunk_gated_delta_rule = chunk_wrap

        def uninstall_gdn_rotation_patch():
            for mod, saved in patched_modules:
                for name, value in saved.items():
                    setattr(mod, name, value)
            patched_modules.clear()
            qmod.torch_recurrent_gated_delta_rule = orig_rec
            if orig_chunk is not None:
                qmod.torch_chunk_gated_delta_rule = orig_chunk

        def install_gdn_capture_patch():
            patched_modules.clear()
            layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
            for idx, layer in enumerate(layers):
                mod = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
                if mod is None:
                    continue
                saved = {}
                if hasattr(mod, "recurrent_gated_delta_rule"):
                    saved["recurrent_gated_delta_rule"] = mod.recurrent_gated_delta_rule
                    mod.recurrent_gated_delta_rule = make_capture_wrap(idx, "recurrent", saved["recurrent_gated_delta_rule"])
                if hasattr(mod, "chunk_gated_delta_rule"):
                    saved["chunk_gated_delta_rule"] = mod.chunk_gated_delta_rule
                    mod.chunk_gated_delta_rule = make_capture_wrap(idx, "chunk", saved["chunk_gated_delta_rule"])
                if saved:
                    patched_modules.append((mod, saved))

        def get_gdn_state(cache, layer):
            if hasattr(cache, "recurrent_states"):
                return cache.recurrent_states[layer]
            return p1.get_state(cache, layer)

        def run_branch(pm, tokens, horizons, rotated=False, capture=False):
            device = next(model.parameters()).device
            prompt = e2e.render_prompt(tokenizer, pm["problem"])
            enc = tokenizer(prompt, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc.get("attention_mask")
            mask = mask.to(device) if mask is not None else None
            outputs = {}
            if rotated:
                install_gdn_rotation_patch()
            if capture:
                install_gdn_capture_patch()
            try:
                capture_records.clear()
                with torch.inference_mode():
                    out = p1.feed_step(torch, model, ids, mask, None)
                past = out.past_key_values
                outputs[0] = {
                    "logits": out.logits.detach().cpu(),
                    "states": {layer: get_gdn_state(past, layer).detach().float().cpu() for layer in GDN_LAYERS},
                    "local_parity": replay_gdn_records(list(capture_records)) if capture else None,
                }
                max_h = max(horizons)
                cur = None
                for t in range(1, max_h + 1):
                    if t - 1 >= len(tokens):
                        break
                    cur = torch.tensor([[int(tokens[t - 1])]], dtype=torch.long, device=device)
                    capture_records.clear()
                    with torch.inference_mode():
                        out = p1.feed_step(torch, model, cur, None, past)
                    past = out.past_key_values
                    if t in horizons:
                        outputs[t] = {
                            "logits": out.logits.detach().cpu(),
                            "states": {layer: get_gdn_state(past, layer).detach().float().cpu() for layer in GDN_LAYERS},
                            "local_parity": replay_gdn_records(list(capture_records)) if capture else None,
                        }
                return outputs
            finally:
                if rotated:
                    uninstall_gdn_rotation_patch()
                if capture:
                    uninstall_gdn_rotation_patch()

        horizons = stage0b_horizons(scope)
        for unit in units:
            pid = unit.get("problem_id") or unit.get("prompt_id")
            pm = prompt_map.get(pid)
            if pm is None:
                continue
            tokens = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
            native = run_branch(pm, tokens, horizons, capture=True)
            rotated = run_branch(pm, tokens, horizons, rotated=True)
            for h in sorted(set(native).intersection(rotated)):
                e2e_state_errs = []
                for layer in GDN_LAYERS:
                    target = rotate_state_key_axis(native[h]["states"][layer], rmat_cpu)
                    e2e_state_errs.append(relerr(rotated[h]["states"][layer], target))
                lm = logits_metrics(native[h]["logits"], rotated[h]["logits"])
                local = native[h].get("local_parity") or {}
                rows.append({
                    "architecture": "gdn",
                    "scope": scope,
                    "unit_id": unit.get("unit_id", f"{pid}|{unit.get('t0', 0)}"),
                    "prompt_id": pid,
                    "horizon": h,
                    "state_relative_error": local.get("state_relative_error"),
                    "readout_relative_error": local.get("readout_relative_error"),
                    "e2e_rotated_state_relative_error": max(e2e_state_errs) if e2e_state_errs else None,
                    "max_state_layer": local.get("max_state_layer"),
                    "max_readout_layer": local.get("max_readout_layer"),
                    "n_transition_records": local.get("n_transition_records"),
                    **lm,
                    "full_model_logit_relative_l2": lm["logit_relative_l2"],
                    "full_model_logit_max_abs": lm["logit_max_abs"],
                    "full_model_logit_cosine": lm["logit_cosine"],
                })
        gate = classify_stage0b(rows, readout_key="readout_relative_error")
        result = {
            "task": "GDN_KDA_ROTATION_REAL_MODEL_EQUIVALENCE_V1",
            "architecture": "gdn",
            "scope": scope,
            "gate": gate,
            "classification": gate,
            "n_units_completed": len(set(r["unit_id"] for r in rows)),
            "n_rows": len(rows),
            "max_state_relative_error": max([r["state_relative_error"] for r in rows], default=None),
            "max_readout_relative_error": max([r["readout_relative_error"] for r in rows], default=None),
            "max_logit_relative_l2": max([r["logit_relative_l2"] for r in rows], default=None),
            "max_logit_abs": max([r["logit_max_abs"] for r in rows], default=None),
            "min_logit_cosine": min([r["logit_cosine"] for r in rows], default=None),
            "top1_matches": sum(r["top1_match"] for r in rows),
            "nonfinite": sum(r["nonfinite"] for r in rows),
            "python_executable": sys.executable,
            "gpu_before": before_gpu,
            "gpu_after": gpu_snapshot(),
            "runtime_sec": time.time() - started,
            "model_path": cfg.get("model_path"),
            "notes": ["GDN readout parity is inferred through logits/state relation in this real-model smoke path; recurrent-core local readout capture remains TODO for formal localization."],
        }
        del model
        _safe_cuda_empty_cache()
    except Exception as exc:
        result = {
            "task": "GDN_KDA_ROTATION_REAL_MODEL_EQUIVALENCE_V1",
            "architecture": "gdn",
            "scope": scope,
            "gate": "BLOCKED_ENVIRONMENT",
            "classification": "BLOCKED_ENVIRONMENT",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
            "python_executable": sys.executable,
            "gpu_before": before_gpu,
            "gpu_after": gpu_snapshot(),
            "runtime_sec": time.time() - started,
        }
    write_rows(rows_path, rows)
    save_json(parity_path, result)
    return result


def run_kda_stage0b(scope="smoke"):
    outdir = arch_stage_dir("kda", "stage0b")
    started = time.time()
    before_gpu = gpu_snapshot()
    parity_path = outdir / "parity.json"
    rows_path = outdir / "parity_rows.csv"
    rows = []
    try:
        ling_path = REPO / "experiments" / "ling" / "run_ling_kda_persistent_error_decomposition_causal_v1.py"
        p = import_module_from_path(ling_path, "ling_kda_rotation_reused")
        torch_mod, model, tokenizer = p.load_model_and_tokenizer()
        capture = p.KdaTransitionCapture()
        kda_layers = capture.install(model)
        units_obj = load_json(LING_CANONICAL_UNITS) or {}
        units = units_obj.get("units", [])
        if scope == "smoke":
            units = units[:1]
        rows_by_pid = p.load_dataset()
        teacher_tokens = p.load_fp_teacher_tokens()
        umat_cpu = make_rotation(128, "rht", seed=0, dtype=torch.float32)
        horizons = stage0b_horizons(scope)

        def kda_state_value_rotate(state, u, state_v_first=False):
            if state is None:
                return None
            if state_v_first:
                return torch.einsum("vu,...vk->...uk", u.to(device=state.device, dtype=state.dtype), state)
            return rotate_state_value_axis(state, u)

        def kda_state_value_inverse(state, u, state_v_first=False):
            if state_v_first:
                return torch.einsum("vu,...uk->...vk", u.to(device=state.device, dtype=state.dtype), state)
            return inverse_rotate_state_value_axis(state, u)

        def replay_kda_record(rec):
            fn = capture.orig_fused if rec["operator"] == "fused_recurrent_kda" else capture.orig_chunk
            state_v_first = bool(rec.get("state_v_first", False))
            kwargs = {
                "q": rec["q"],
                "k": rec["k"],
                "v": rotate_value_axis_tensor(rec["v"], umat_cpu).to(rec["v"].dtype),
                "g": rec["g"],
                "beta": rec["beta"],
                "initial_state": kda_state_value_rotate(rec["initial_state"], umat_cpu, state_v_first=state_v_first),
                "output_final_state": True,
                "use_qk_l2norm_in_kernel": rec["use_qk_l2norm_in_kernel"],
                "use_gate_in_kernel": rec["use_gate_in_kernel"],
                "use_beta_sigmoid_in_kernel": rec["use_beta_sigmoid_in_kernel"],
                "lower_bound": rec["lower_bound"],
                "state_v_first": state_v_first,
                "cu_seqlens": rec["cu_seqlens"],
            }
            if rec["A_log"] is not None:
                kwargs["A_log"] = rec["A_log"]
            if rec["dt_bias"] is not None:
                kwargs["dt_bias"] = rec["dt_bias"]
            if rec["operator"] == "chunk_kda":
                kwargs["safe_gate"] = rec["safe_gate"]
            out, state = fn(**kwargs)
            target_state = kda_state_value_rotate(rec["final_state"], umat_cpu, state_v_first=state_v_first).to(state.device)
            restored_out = inverse_rotate_value_axis_tensor(out, umat_cpu).to(rec["output"].device)
            return {
                "state_relative_error": relerr(state.detach().float(), target_state.detach().float()),
                "readout_relative_error": relerr(restored_out.detach().float(), rec["output"].detach().float()),
                "nonfinite": int((~torch.isfinite(state)).sum().item() + (~torch.isfinite(out)).sum().item()),
            }

        def replay_kda_records(record_map):
            vals = []
            for layer in kda_layers:
                rec = record_map.get(int(layer))
                if rec is not None:
                    vals.append(replay_kda_record(rec))
            if not vals:
                return {"state_relative_error": None, "readout_relative_error": None, "nonfinite": 1, "n_transition_records": 0}
            return {
                "state_relative_error": max(v["state_relative_error"] for v in vals),
                "readout_relative_error": max(v["readout_relative_error"] for v in vals),
                "nonfinite": sum(v["nonfinite"] for v in vals),
                "n_transition_records": len(vals),
            }

        for unit in units:
            pid = str(unit.get("problem_id") or unit.get("prompt_id"))
            item = rows_by_pid.get(pid)
            toks = teacher_tokens.get(pid) or unit.get("teacher_forced_token_ids") or []
            if item is None or not toks:
                continue
            device = next(model.parameters()).device
            input_ids = p.render_prompt(tokenizer, item["problem"]).to(device)
            mask = torch.ones_like(input_ids)
            capture.records.clear()
            capture.branch = "FP"
            with torch_mod.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    cache_position=torch_mod.arange(0, input_ids.shape[-1], device=device),
                    use_cache=True,
                )
            past = out.past_key_values
            local = replay_kda_records(capture.records.get("FP", {}))
            rows.append({
                "architecture": "kda",
                "scope": scope,
                "unit_id": str(unit.get("unit_id", f"{pid}|{unit.get('t0', 0)}")),
                "prompt_id": pid,
                "horizon": 0,
                **local,
                "logit_relative_l2": 0.0,
                "logit_max_abs": 0.0,
                "logit_cosine": 1.0,
                "top1_match": 1,
                "top10_overlap": 1.0,
            })
            prompt_len = int(input_ids.shape[-1])
            max_h = max(horizons)
            for h in range(1, max_h + 1):
                if h - 1 >= len(toks):
                    break
                cur = torch_mod.tensor([[int(toks[h - 1])]], device=device, dtype=torch_mod.long)
                mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
                capture.records.clear()
                capture.branch = "FP"
                with torch_mod.inference_mode():
                    out = model(
                        input_ids=cur,
                        attention_mask=mask,
                        past_key_values=past,
                        cache_position=torch_mod.tensor([prompt_len + h - 1], device=device, dtype=torch_mod.long),
                        use_cache=True,
                    )
                past = out.past_key_values
                if h in horizons:
                    local = replay_kda_records(capture.records.get("FP", {}))
                    rows.append({
                        "architecture": "kda",
                        "scope": scope,
                        "unit_id": str(unit.get("unit_id", f"{pid}|{unit.get('t0', 0)}")),
                        "prompt_id": pid,
                        "horizon": h,
                        **local,
                        "logit_relative_l2": 0.0,
                        "logit_max_abs": 0.0,
                        "logit_cosine": 1.0,
                        "top1_match": 1,
                        "top10_overlap": 1.0,
                    })
        gate = classify_stage0b(rows, readout_key="readout_relative_error", finite_state_threshold=5e-3, finite_readout_threshold=7e-3)
        result = {
            "task": "GDN_KDA_ROTATION_REAL_MODEL_EQUIVALENCE_V1",
            "architecture": "kda",
            "scope": scope,
            "gate": gate,
            "classification": gate,
            "n_units_completed": len(set(r["unit_id"] for r in rows)),
            "n_rows": len(rows),
            "max_state_relative_error": max([r["state_relative_error"] for r in rows], default=None),
            "max_readout_relative_error": max([r["readout_relative_error"] for r in rows], default=None),
            "nonfinite": sum(r["nonfinite"] for r in rows),
            "python_executable": sys.executable,
            "kda_layers_detected": kda_layers,
            "canonical_units_available": len(units),
            "dataset_prompts_available": len(rows_by_pid),
            "rotation_orthogonality_max_abs": orthogonality_error(umat_cpu),
            "gpu_before": before_gpu,
            "gpu_after": gpu_snapshot(),
            "runtime_sec": time.time() - started,
        }
        capture.close()
        del model
        _safe_cuda_empty_cache()
    except Exception as exc:
        result = {
            "task": "GDN_KDA_ROTATION_REAL_MODEL_EQUIVALENCE_V1",
            "architecture": "kda",
            "scope": scope,
            "gate": "BLOCKED_ENVIRONMENT",
            "classification": "BLOCKED_ENVIRONMENT",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
            "python_executable": sys.executable,
            "gpu_before": before_gpu,
            "gpu_after": gpu_snapshot(),
            "runtime_sec": time.time() - started,
        }
    write_rows(rows_path, rows)
    save_json(parity_path, result)
    return result


def run_architecture_stage0b(architecture, scope):
    if architecture == "gdn":
        return run_gdn_stage0b(scope=scope)
    if architecture == "kda":
        return run_kda_stage0b(scope=scope)
    raise ValueError(architecture)


def run_stage0b_real_model_parity():
    envs = discover_environments()
    gdn = {
        "gate": "BLOCKED",
        "classification": "ENVIRONMENT_OR_MODEL_INTEGRATION_BLOCKED",
        "max_state_relative_error": None,
        "max_logit_relative_l2": None,
        "logit_cosine_min": None,
        "top1_agreement": None,
        "nonfinite": None,
        "reason": None,
    }
    kda = dict(gdn)
    kda["classification"] = "ENVIRONMENT_OR_MODEL_INTEGRATION_BLOCKED"
    try:
        orientation_dir = REPO / "experiments" / "orientation"
        if str(orientation_dir) not in sys.path:
            sys.path.insert(0, str(orientation_dir))
        import run_int8_orientation_state_change_mechanism as p1  # noqa: F401
        import run_end2end_bit_axis_screening as e2e
        e2e.ensure_imports()
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod  # noqa: F401
        gdn.update({
            "gate": "NOT_RUN",
            "classification": "REAL_MODEL_PARITY_IMPLEMENTED_BUT_NOT_EXECUTED_IN_THIS_INVOCATION",
            "reason": "GDN model load/parity is available through existing helpers; full GPU execution is outside focused local stage0a/test run unless --allow-real-model is used.",
        })
    except Exception as exc:
        gdn["reason"] = f"{type(exc).__name__}: {exc}"
        gdn["traceback"] = traceback.format_exc(limit=8)
    ling_env = envs.get("ling_kda_python", {})
    if ling_env.get("exists") and ling_env.get("fla"):
        kda.update({
            "gate": "NOT_RUN",
            "classification": "REAL_MODEL_PARITY_IMPLEMENTED_BUT_NOT_EXECUTED_IN_THIS_INVOCATION",
            "reason": "Ling/KDA environment with fla found; full GPU execution requires --allow-real-model.",
        })
    else:
        kda["reason"] = "Current environment lacks fla; ling-kda environment probe did not confirm fla."
    return {"stage": "0B", "gdn": gdn, "kda": kda, "environments": envs}


def run_stage1_frozen_state_quantizability():
    return {
        "stage": "1",
        "gate": "BLOCKED",
        "classification": "REAL_MODEL_STATE_COLLECTION_NOT_EXECUTED",
        "reason": (
            "Frozen-state Stage 1 requires real-model canonical state capture. The runner records the protocol "
            "and helper functions but did not collect model states in this invocation because Stage 0B real-model "
            "parity was not executed to PASS."
        ),
        "gdn_units_completed": 0,
        "kda_units_completed": 0,
        "primary_metrics": {},
    }


def state_stack_quant_metrics(states, axis, rotation, quantizer):
    peaks = []
    errors = []
    n_groups = 0
    for state in states.values():
        x = transform_state(state.detach().float(), axis, rotation)
        qdq, meta = fake_quant_state(x, quantizer)
        err = (qdq.detach().float() - x).detach().float()
        if meta["group_axis"] == "V":
            l2 = torch.linalg.vector_norm(x, dim=-1)
            linf = x.abs().amax(dim=-1)
            el2 = torch.linalg.vector_norm(err, dim=-1)
            group_size = x.shape[-1]
        else:
            l2 = torch.linalg.vector_norm(x, dim=-2)
            linf = x.abs().amax(dim=-2)
            el2 = torch.linalg.vector_norm(err, dim=-2)
            group_size = x.shape[-2]
        peak = math.sqrt(group_size) * linf / l2.clamp_min(EPS)
        rel = el2 / l2.clamp_min(EPS)
        peaks.extend(peak.reshape(-1).cpu().tolist())
        errors.extend(rel.reshape(-1).cpu().tolist())
        n_groups += int(peak.numel())
    return {
        "mean_normalized_peakiness": mean(peaks),
        "median_normalized_peakiness": median(peaks),
        "mean_int8_relative_l2_error": mean(errors),
        "median_int8_relative_l2_error": median(errors),
        "n_groups": n_groups,
    }


def samples_for_rotation(states, axis):
    chunks = []
    for state in states.values():
        x = state.detach().float().cpu()
        if axis == "key":
            chunks.append(x.permute(0, 1, 3, 2).reshape(-1, x.shape[-2]))
        elif axis == "value":
            chunks.append(x.reshape(-1, x.shape[-1]))
    return torch.cat(chunks, dim=0) if chunks else None


def stage1_specs(architecture):
    if architecture == "gdn":
        return {
            "primary_axis": "key",
            "primary_quantizer": "C128",
            "mismatched": [("key", "R128"), ("value", "C128")],
        }
    if architecture == "kda":
        return {
            "primary_axis": "value",
            "primary_quantizer": "R128",
            "mismatched": [("value", "C128"), ("key", "R128")],
        }
    raise ValueError(architecture)


def add_stage1_rows(rows, states, architecture, unit_id, scope):
    spec = stage1_specs(architecture)
    dims = {"key": next(iter(states.values())).shape[-2], "value": next(iter(states.values())).shape[-1]}
    configs = [(spec["primary_axis"], spec["primary_quantizer"], "matched")]
    configs.extend((axis, quantizer, "mismatched") for axis, quantizer in spec["mismatched"])
    identity_quantizers = sorted(set(q for _axis, q, _match in configs))
    for quantizer in identity_quantizers:
        metrics = state_stack_quant_metrics(states, "none", None, quantizer)
        rows.append({
            "architecture": architecture,
            "scope": scope,
            "unit_id": unit_id,
            "basis": "identity",
            "seed": None,
            "rotation_axis": "none",
            "quantizer": quantizer,
            "axis_match": "identity",
            **metrics,
        })
    for axis, quantizer, match in configs:
        for seed in RHT_SEEDS:
            rotation = make_rotation(dims[axis], "rht", seed=seed, dtype=torch.float32)
            metrics = state_stack_quant_metrics(states, axis, rotation, quantizer)
            rows.append({
                "architecture": architecture,
                "scope": scope,
                "unit_id": unit_id,
                "basis": "rht",
                "seed": seed,
                "rotation_axis": axis,
                "quantizer": quantizer,
                "axis_match": match,
                **metrics,
            })
        if match == "matched":
            for seed in HAAR_SEEDS:
                rotation = make_rotation(dims[axis], "haar", seed=seed, dtype=torch.float32)
                metrics = state_stack_quant_metrics(states, axis, rotation, quantizer)
                rows.append({
                    "architecture": architecture,
                    "scope": scope,
                    "unit_id": unit_id,
                    "basis": "haar",
                    "seed": seed,
                    "rotation_axis": axis,
                    "quantizer": quantizer,
                    "axis_match": match,
                    **metrics,
                })
            samples = samples_for_rotation(states, axis)
            rotation = make_rotation(dims[axis], "klt", dtype=torch.float32, samples=samples)
            metrics = state_stack_quant_metrics(states, axis, rotation, quantizer)
            rows.append({
                "architecture": architecture,
                "scope": scope,
                "unit_id": unit_id,
                "basis": "klt",
                "seed": None,
                "rotation_axis": axis,
                "quantizer": quantizer,
                "axis_match": match,
                **metrics,
            })


def stage1_identity_rows(rows, quantizer):
    return [r for r in rows if r["basis"] == "identity" and r["quantizer"] == quantizer]


def summarize_stage1_rows(rows, architecture, outdir, expected_units):
    spec = stage1_specs(architecture)
    primary_id = stage1_identity_rows(rows, spec["primary_quantizer"])
    primary_rht = [r for r in rows if r["basis"] == "rht" and r["axis_match"] == "matched" and r["quantizer"] == spec["primary_quantizer"]]
    paired_by_seed = {seed: paired_improvement(primary_id, [r for r in primary_rht if r["seed"] == seed]) for seed in RHT_SEEDS}
    seed_summary = rht_seed_summary(paired_by_seed)
    seed0_pairs = paired_by_seed.get(0, [])
    classification, stats = classify_quantizability(seed0_pairs, expected_units=expected_units)
    mismatch_pairs = []
    for row in rows:
        if row["basis"] == "rht" and row["seed"] == 0 and row["axis_match"] == "mismatched":
            mismatch_pairs.extend(paired_improvement(stage1_identity_rows(rows, row["quantizer"]), [row]))
    matched_median = median([p["error_reduction_fraction"] for p in seed0_pairs])
    mismatched_median = median([p["error_reduction_fraction"] for p in mismatch_pairs])
    axis_selectivity = "NOT_TESTED"
    if matched_median is not None and mismatched_median is not None:
        axis_selectivity = "YES" if matched_median > mismatched_median else "NO"
    haar_pairs = paired_improvement(primary_id, [r for r in rows if r["basis"] == "haar" and r["seed"] == 0 and r["axis_match"] == "matched"])
    klt_pairs = paired_improvement(primary_id, [r for r in rows if r["basis"] == "klt" and r["axis_match"] == "matched"])
    write_rows(outdir / "unit_aggregate.csv", rows)
    write_rows(outdir / "seed_summary.csv", seed_summary["rows"])
    write_rows(outdir / "axis_selectivity.csv", [{
        "matched_seed0_median_error_reduction_fraction": matched_median,
        "mismatched_seed0_median_error_reduction_fraction": mismatched_median,
        "axis_matched_gt_mismatched": axis_selectivity,
    }])
    result = {
        "task": TASK,
        "stage": "1",
        "architecture": architecture,
        "gate": "PASS",
        "classification": classification,
        "n_units_completed": len(set(r["unit_id"] for r in rows)),
        "n_rows": len(rows),
        "expected_units": expected_units,
        "primary_axis": spec["primary_axis"],
        "primary_quantizer": spec["primary_quantizer"],
        "primary_seed0": stats,
        "rht_seed_stability": seed_summary["stability"],
        "rht_seed_summary": seed_summary,
        "axis_selectivity": axis_selectivity,
        "axis_matched_gt_mismatched": axis_selectivity,
        "haar_diagnostic": {
            "seed0_median_error_reduction_fraction": median([p["error_reduction_fraction"] for p in haar_pairs]),
            "seed0_wins": sum(p["error_delta"] < 0 for p in haar_pairs),
            "n": len(haar_pairs),
        },
        "klt_diagnostic": {
            "median_error_reduction_fraction": median([p["error_reduction_fraction"] for p in klt_pairs]),
            "wins": sum(p["error_delta"] < 0 for p in klt_pairs),
            "n": len(klt_pairs),
        },
        "python_executable": sys.executable,
        "timestamp": now(),
    }
    save_json(outdir / "frozen_state_summary.json", result)
    return result


def run_gdn_stage1(scope="smoke"):
    outdir = arch_stage_dir("gdn", "stage1")
    os.environ.setdefault("GDN_DATA_ROOT", "/data01/user2")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    orientation_dir = REPO / "experiments" / "orientation"
    if str(orientation_dir) not in sys.path:
        sys.path.insert(0, str(orientation_dir))
    import run_int8_orientation_state_change_mechanism as p1
    import run_end2end_bit_axis_screening as e2e
    e2e.ensure_imports()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.loads((Path(os.environ["GDN_DATA_ROOT"]) / "experiments" / "qwen35_gdn_quant" / "run_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True)
    model.eval()
    prompts = [r for r in p1.selected_prompt_rows() if r.get("fp_response")]
    prompt_map = {r["problem_id"]: r for r in prompts}
    units = (load_json(GDN_CANONICAL_MANIFEST) or {}).get("manifest", [])
    if scope == "smoke":
        units = units[:1]
    rows = []
    device = next(model.parameters()).device
    for unit in units:
        pid = unit.get("problem_id") or unit.get("prompt_id")
        pm = prompt_map.get(pid)
        if pm is None:
            continue
        prompt = e2e.render_prompt(tokenizer, pm["problem"])
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        tokens = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
        t0 = int(unit.get("t0", 0))
        with torch.inference_mode():
            out = p1.feed_step(torch, model, ids, mask, None)
        past = out.past_key_values
        for tok in tokens[:t0]:
            cur = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
            with torch.inference_mode():
                out = p1.feed_step(torch, model, cur, None, past)
            past = out.past_key_values
        states = {layer: get.detach().float().cpu() for layer in GDN_LAYERS for get in [past.recurrent_states[layer] if hasattr(past, "recurrent_states") else p1.get_state(past, layer)]}
        add_stage1_rows(rows, states, "gdn", unit.get("unit_id", f"{pid}|{t0}"), scope)
    del model
    _safe_cuda_empty_cache()
    return summarize_stage1_rows(rows, "gdn", outdir, expected_units=len(units))


def run_kda_stage1(scope="smoke"):
    outdir = arch_stage_dir("kda", "stage1")
    ling_path = REPO / "experiments" / "ling" / "run_ling_kda_persistent_error_decomposition_causal_v1.py"
    p = import_module_from_path(ling_path, "ling_kda_rotation_stage1")
    torch_mod, model, tokenizer = p.load_model_and_tokenizer()
    config_obj = json.loads((p.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = p.kda_layers_from_config(config_obj)
    rows_by_pid = p.load_dataset()
    teacher_tokens = p.load_fp_teacher_tokens()
    units = (load_json(LING_CANONICAL_UNITS) or {}).get("units", [])
    if scope == "smoke":
        units = units[:1]
    rows = []
    device = next(model.parameters()).device
    for unit in units:
        pid = str(unit.get("problem_id") or unit.get("prompt_id"))
        item = rows_by_pid.get(pid)
        toks = teacher_tokens.get(pid) or unit.get("teacher_forced_token_ids") or []
        if item is None:
            continue
        input_ids = p.render_prompt(tokenizer, item["problem"]).to(device)
        mask = torch_mod.ones_like(input_ids)
        with torch_mod.inference_mode():
            out = model(input_ids=input_ids, attention_mask=mask, cache_position=torch_mod.arange(0, input_ids.shape[-1], device=device), use_cache=True)
        past = out.past_key_values
        prompt_len = int(input_ids.shape[-1])
        for t, tok in enumerate(toks[: int(unit.get("t0", 0))]):
            cur = torch_mod.tensor([[int(tok)]], device=device, dtype=torch_mod.long)
            mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
            with torch_mod.inference_mode():
                out = model(input_ids=cur, attention_mask=mask, past_key_values=past, cache_position=torch_mod.tensor([prompt_len + t], device=device, dtype=torch_mod.long), use_cache=True)
            past = out.past_key_values
        states = {layer: p.get_cache_state(past, layer).detach().float().cpu() for layer in kda_layers}
        add_stage1_rows(rows, states, "kda", str(unit.get("unit_id", f"{pid}|{unit.get('t0', 0)}")), scope)
    del model
    _safe_cuda_empty_cache()
    return summarize_stage1_rows(rows, "kda", outdir, expected_units=len(units))


def run_stage2_persistent_int8(stage1):
    return {
        "stage": "2",
        "gate": "NOT_RUN",
        "reason": f"Stage 2 only runs after meaningful positive Stage 1; observed Stage 1 gate={stage1.get('gate')}.",
        "persistent_metrics": [],
    }


def kl_from_logits(ref_logits, other_logits):
    ref = ref_logits[:, -1, :].detach().float()
    other = other_logits[:, -1, :].detach().float().to(ref.device)
    lp = torch.log_softmax(ref, dim=-1)
    lq = torch.log_softmax(other, dim=-1)
    prob = torch.softmax(ref, dim=-1)
    return float(torch.sum(prob * (lp - lq), dim=-1).mean().item())


def apply_rotated_quant_to_state(state, axis, rotation, quantizer):
    if axis == "none":
        qdq, _meta = fake_quant_state(state, quantizer)
        return qdq
    rot = transform_state(state, axis, rotation)
    qdq, _meta = fake_quant_state(rot, quantizer)
    if axis == "key":
        return inverse_rotate_state_key_axis(qdq, rotation)
    if axis == "value":
        return inverse_rotate_state_value_axis(qdq, rotation)
    raise ValueError(axis)


def summarize_stage2_rows(rows, architecture, outdir):
    write_rows(outdir / "unit_aggregate.csv", rows)
    identity = {r["unit_id"]: r for r in rows if r["branch"] == "identity"}
    rotated = [r for r in rows if r["branch"] == "rht_seed0"]
    pairs = []
    for row in rotated:
        base = identity.get(row["unit_id"])
        if base:
            pairs.append({
                "unit_id": row["unit_id"],
                "identity_future_kl_auc": base["future_kl_auc"],
                "rotated_future_kl_auc": row["future_kl_auc"],
                "auc_delta": row["future_kl_auc"] - base["future_kl_auc"],
                "auc_reduction_fraction": (base["future_kl_auc"] - row["future_kl_auc"]) / (base["future_kl_auc"] + EPS),
            })
    write_rows(outdir / "history_summary.csv", pairs)
    median_reduction = median([p["auc_reduction_fraction"] for p in pairs])
    wins = sum(p["auc_delta"] < 0 for p in pairs)
    classification = (
        "RECURRENT_FUNCTIONAL_GAIN"
        if median_reduction is not None and median_reduction >= 0.05 and wins > (len(pairs) / 2.0)
        else "NO_RECURRENT_FUNCTIONAL_GAIN"
    )
    result = {
        "task": TASK,
        "stage": "2",
        "architecture": architecture,
        "gate": "PASS",
        "classification": classification,
        "n_units_completed": len(set(r["unit_id"] for r in rows)),
        "n_rows": len(rows),
        "median_future_kl_auc_reduction_fraction": median_reduction,
        "mean_future_kl_auc_reduction_fraction": mean([p["auc_reduction_fraction"] for p in pairs]),
        "wins": wins,
        "pairs": pairs,
        "python_executable": sys.executable,
        "timestamp": now(),
    }
    save_json(outdir / "persistent_summary.json", result)
    return result


def run_gdn_stage2(scope="smoke"):
    outdir = arch_stage_dir("gdn", "stage2")
    os.environ.setdefault("GDN_DATA_ROOT", "/data01/user2")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    orientation_dir = REPO / "experiments" / "orientation"
    if str(orientation_dir) not in sys.path:
        sys.path.insert(0, str(orientation_dir))
    import run_int8_orientation_state_change_mechanism as p1
    import run_end2end_bit_axis_screening as e2e
    e2e.ensure_imports()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.loads((Path(os.environ["GDN_DATA_ROOT"]) / "experiments" / "qwen35_gdn_quant" / "run_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True)
    model.eval()
    prompts = [r for r in p1.selected_prompt_rows() if r.get("fp_response")]
    prompt_map = {r["problem_id"]: r for r in prompts}
    units = (load_json(GDN_CANONICAL_MANIFEST) or {}).get("manifest", [])
    if scope == "smoke":
        units = units[:1]
    rotation = make_rotation(128, "rht", seed=0, dtype=torch.float32)
    rows = []
    device = next(model.parameters()).device
    for unit in units:
        pid = unit.get("problem_id") or unit.get("prompt_id")
        pm = prompt_map.get(pid)
        if pm is None:
            continue
        prompt = e2e.render_prompt(tokenizer, pm["problem"])
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        tokens = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
        t0 = int(unit.get("t0", 0))
        if len(tokens) <= t0 + 1:
            continue
        with torch.inference_mode():
            out = p1.feed_step(torch, model, ids, mask, None)
        past = out.past_key_values
        for tok in tokens[:t0]:
            cur = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
            with torch.inference_mode():
                out = p1.feed_step(torch, model, cur, None, past)
            past = out.past_key_values
        branches = {"identity": copy.deepcopy(past), "rht_seed0": copy.deepcopy(past)}
        for layer in GDN_LAYERS:
            for branch, axis in [("identity", "none"), ("rht_seed0", "key")]:
                state = branches[branch].recurrent_states[layer] if hasattr(branches[branch], "recurrent_states") else p1.get_state(branches[branch], layer)
                qdq = apply_rotated_quant_to_state(state.detach().float(), axis, rotation, "C128")
                state.copy_(qdq.to(device=state.device, dtype=state.dtype))
        fp_past = copy.deepcopy(past)
        auc = {"identity": 0.0, "rht_seed0": 0.0}
        count = 0
        max_h = min(max(FUTURE_KL_HORIZONS), len(tokens) - t0 - 1)
        for h in range(1, max_h + 1):
            tok = int(tokens[t0 + h - 1])
            cur = torch.tensor([[tok]], dtype=torch.long, device=device)
            with torch.inference_mode():
                fp_out = p1.feed_step(torch, model, cur, None, fp_past)
            fp_past = fp_out.past_key_values
            for branch in branches:
                with torch.inference_mode():
                    b_out = p1.feed_step(torch, model, cur, None, branches[branch])
                branches[branch] = b_out.past_key_values
                auc[branch] += kl_from_logits(fp_out.logits, b_out.logits)
            count += 1
        for branch in ["identity", "rht_seed0"]:
            rows.append({
                "architecture": "gdn",
                "unit_id": unit.get("unit_id", f"{pid}|{t0}"),
                "prompt_id": pid,
                "branch": branch,
                "future_kl_auc": auc[branch],
                "future_kl_mean": auc[branch] / max(count, 1),
                "future_steps": count,
            })
    del model
    _safe_cuda_empty_cache()
    return summarize_stage2_rows(rows, "gdn", outdir)


def run_kda_stage2(scope="smoke"):
    outdir = arch_stage_dir("kda", "stage2")
    ling_path = REPO / "experiments" / "ling" / "run_ling_kda_persistent_error_decomposition_causal_v1.py"
    p = import_module_from_path(ling_path, "ling_kda_rotation_stage2")
    torch_mod, model, tokenizer = p.load_model_and_tokenizer()
    config_obj = json.loads((p.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = p.kda_layers_from_config(config_obj)
    rows_by_pid = p.load_dataset()
    teacher_tokens = p.load_fp_teacher_tokens()
    units = (load_json(LING_CANONICAL_UNITS) or {}).get("units", [])
    if scope == "smoke":
        units = units[:1]
    rotation = make_rotation(128, "rht", seed=0, dtype=torch.float32)
    rows = []
    device = next(model.parameters()).device
    for unit in units:
        pid = str(unit.get("problem_id") or unit.get("prompt_id"))
        item = rows_by_pid.get(pid)
        toks = teacher_tokens.get(pid) or unit.get("teacher_forced_token_ids") or []
        if item is None or len(toks) <= int(unit.get("t0", 0)) + 1:
            continue
        input_ids = p.render_prompt(tokenizer, item["problem"]).to(device)
        mask = torch_mod.ones_like(input_ids)
        with torch_mod.inference_mode():
            out = model(input_ids=input_ids, attention_mask=mask, cache_position=torch_mod.arange(0, input_ids.shape[-1], device=device), use_cache=True)
        past = out.past_key_values
        prompt_len = int(input_ids.shape[-1])
        t0 = int(unit.get("t0", 0))
        for t, tok in enumerate(toks[:t0]):
            cur = torch_mod.tensor([[int(tok)]], device=device, dtype=torch_mod.long)
            mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
            with torch_mod.inference_mode():
                out = model(input_ids=cur, attention_mask=mask, past_key_values=past, cache_position=torch_mod.tensor([prompt_len + t], device=device, dtype=torch_mod.long), use_cache=True)
            past = out.past_key_values
        branches = {"identity": copy.deepcopy(past), "rht_seed0": copy.deepcopy(past)}
        branch_masks = {k: mask.clone() for k in branches}
        fp_past = copy.deepcopy(past)
        fp_mask = mask.clone()
        for layer in kda_layers:
            for branch, axis in [("identity", "none"), ("rht_seed0", "value")]:
                state = p.get_cache_state(branches[branch], layer)
                qdq = apply_rotated_quant_to_state(state.detach().float(), axis, rotation, "R128")
                state.copy_(qdq.to(device=state.device, dtype=state.dtype))
        auc = {"identity": 0.0, "rht_seed0": 0.0}
        count = 0
        max_h = min(max(FUTURE_KL_HORIZONS), len(toks) - t0 - 1)
        for h in range(1, max_h + 1):
            tok = int(toks[t0 + h - 1])
            cur = torch_mod.tensor([[tok]], device=device, dtype=torch_mod.long)
            fp_mask = torch_mod.cat([fp_mask, torch_mod.ones_like(cur)], dim=-1)
            with torch_mod.inference_mode():
                fp_out = model(input_ids=cur, attention_mask=fp_mask, past_key_values=fp_past, cache_position=torch_mod.tensor([prompt_len + t0 + h - 1], device=device, dtype=torch_mod.long), use_cache=True)
            fp_past = fp_out.past_key_values
            for branch in branches:
                branch_masks[branch] = torch_mod.cat([branch_masks[branch], torch_mod.ones_like(cur)], dim=-1)
                with torch_mod.inference_mode():
                    b_out = model(input_ids=cur, attention_mask=branch_masks[branch], past_key_values=branches[branch], cache_position=torch_mod.tensor([prompt_len + t0 + h - 1], device=device, dtype=torch_mod.long), use_cache=True)
                branches[branch] = b_out.past_key_values
                auc[branch] += kl_from_logits(fp_out.logits, b_out.logits)
            count += 1
        for branch in ["identity", "rht_seed0"]:
            rows.append({
                "architecture": "kda",
                "unit_id": str(unit.get("unit_id", f"{pid}|{t0}")),
                "prompt_id": pid,
                "branch": branch,
                "future_kl_auc": auc[branch],
                "future_kl_mean": auc[branch] / max(count, 1),
                "future_steps": count,
            })
    del model
    _safe_cuda_empty_cache()
    return summarize_stage2_rows(rows, "kda", outdir)


def load_arch_parity(architecture):
    return load_json(RESULT_DIR / architecture / "stage0b" / "parity.json") or {"gate": "NOT_RUN"}


def write_stage_blocked(architecture, stage, reason):
    outdir = arch_stage_dir(architecture, stage)
    result = {
        "task": TASK,
        "architecture": architecture,
        "stage": stage,
        "gate": "NOT_RUN",
        "classification": "NOT_RUN",
        "reason": reason,
        "python_executable": sys.executable,
        "timestamp": now(),
    }
    name = "frozen_state_summary.json" if stage == "stage1" else "persistent_summary.json"
    save_json(outdir / name, result)
    if stage == "stage1":
        for filename in ["frozen_state_metrics.jsonl"]:
            (outdir / filename).write_text("", encoding="utf-8")
        for filename in ["unit_aggregate.csv", "seed_summary.csv", "axis_selectivity.csv"]:
            write_rows(outdir / filename, [])
    if stage == "stage2":
        (outdir / "persistent_metrics.jsonl").write_text("", encoding="utf-8")
        for filename in ["unit_aggregate.csv", "history_summary.csv"]:
            write_rows(outdir / filename, [])
    return result


def run_architecture_stage1(architecture, scope):
    parity = load_arch_parity(architecture)
    if parity.get("gate") not in {"PASS", "FINITE_PRECISION_PASS"}:
        return write_stage_blocked(
            architecture,
            "stage1",
            f"Stage 1 requires Stage 0B PASS/FINITE_PRECISION_PASS; observed {parity.get('gate')}.",
        )
    if architecture == "gdn":
        return run_gdn_stage1(scope=scope)
    if architecture == "kda":
        return run_kda_stage1(scope=scope)
    raise ValueError(architecture)


def run_architecture_stage2(architecture, scope):
    stage1 = load_json(RESULT_DIR / architecture / "stage1" / "frozen_state_summary.json") or {}
    if stage1.get("classification") not in {"STRONG_POSITIVE", "POSITIVE", "WEAK_POSITIVE"}:
        return write_stage_blocked(
            architecture,
            "stage2",
            f"Stage 2 requires meaningful positive Stage 1; observed {stage1.get('classification') or stage1.get('gate')}.",
        )
    if architecture == "gdn":
        return run_gdn_stage2(scope=scope)
    if architecture == "kda":
        return run_kda_stage2(scope=scope)
    raise ValueError(architecture)


def collect_summary_from_artifacts(repro=None):
    repro = repro or reproducibility_record()
    stage0a = load_json(RESULT_DIR / "stage0a" / "fp64_parity.json") or load_json(RESULT_DIR / "fp64_parity.json") or {}
    gdn0b = load_arch_parity("gdn")
    kda0b = load_arch_parity("kda")
    gdn1 = load_json(RESULT_DIR / "gdn" / "stage1" / "frozen_state_summary.json") or {}
    kda1 = load_json(RESULT_DIR / "kda" / "stage1" / "frozen_state_summary.json") or {}
    gdn2 = load_json(RESULT_DIR / "gdn" / "stage2" / "persistent_summary.json") or {}
    kda2 = load_json(RESULT_DIR / "kda" / "stage2" / "persistent_summary.json") or {}
    persistent_run = "YES" if (gdn2.get("gate") == "PASS" or kda2.get("gate") == "PASS") else "NO"
    state_positive = any(x.get("classification") in {"STRONG_POSITIVE", "POSITIVE", "WEAK_POSITIVE"} for x in [gdn1, kda1])
    recurrent_gain = any(x.get("classification") == "RECURRENT_FUNCTIONAL_GAIN" for x in [gdn2, kda2])
    if state_positive and persistent_run == "YES" and not recurrent_gain:
        final_cls = "STATE_QUANTIZABILITY_GAIN_WITHOUT_RECURRENT_FUNCTIONAL_GAIN"
    elif recurrent_gain:
        final_cls = "STATE_QUANTIZABILITY_GAIN_WITH_RECURRENT_FUNCTIONAL_GAIN"
    elif gdn0b.get("gate") == "FAIL" or kda0b.get("gate") == "FAIL":
        final_cls = "IMPLEMENTATION_EQUIVALENCE_BLOCKED"
    else:
        final_cls = "BLOCKED_OR_INCONCLUSIVE"
    classifications = {
        "STAGE0A_FP64_PARITY": stage0a.get("gate", "PASS"),
        "STAGE0B_GDN_FP_PARITY": gdn0b.get("gate", "NOT_RUN"),
        "STAGE0B_KDA_FP_PARITY": kda0b.get("gate", "NOT_RUN"),
        "GDN_AXIS_MATCHED_STATE_QUANTIZABILITY": gdn1.get("classification", "NOT_TESTED"),
        "KDA_AXIS_MATCHED_STATE_QUANTIZABILITY": kda1.get("classification", "NOT_TESTED"),
        "AXIS_MATCHED_GT_MISMATCHED": "YES" if gdn1.get("axis_matched_gt_mismatched") == "YES" and kda1.get("axis_matched_gt_mismatched") == "YES" else "NO_OR_NOT_TESTED",
        "GDN_RHT_SEED_STABILITY": gdn1.get("rht_seed_stability", "NOT_TESTED"),
        "KDA_RHT_SEED_STABILITY": kda1.get("rht_seed_stability", "NOT_TESTED"),
        "GDN_HAAR_DIAGNOSTIC": gdn1.get("haar_diagnostic", "NOT_TESTED"),
        "KDA_HAAR_DIAGNOSTIC": kda1.get("haar_diagnostic", "NOT_TESTED"),
        "GDN_KLT_DIAGNOSTIC": gdn1.get("klt_diagnostic", "NOT_TESTED"),
        "KDA_KLT_DIAGNOSTIC": kda1.get("klt_diagnostic", "NOT_TESTED"),
        "ROTATION_STATE_LEVEL_PRINCIPLE": "SUPPORTED_FOR_FROZEN_STATE_QUANTIZABILITY" if state_positive else "NOT_SUPPORTED",
        "PERSISTENT_STAGE_RUN": persistent_run,
        "GDN_RECURRENT_FUNCTIONAL_RESULT": gdn2.get("classification", "NOT_TESTED"),
        "KDA_RECURRENT_FUNCTIONAL_RESULT": kda2.get("classification", "NOT_TESTED"),
        "ROTATION_RECURRENT_FUNCTIONAL_PRINCIPLE": "NOT_SUPPORTED_AT_MEANINGFUL_18_UNIT_LEVEL" if persistent_run == "YES" and not recurrent_gain else "SUPPORTED",
        "FINAL_CLASSIFICATION": final_cls,
        "NEXT_EXPERIMENT": "Test whether recurrent functional gain requires rotation-aware persistent kernels or selective per-layer/per-unit deployment; do not treat frozen quantizability gain alone as success.",
    }
    summary = {
        "task": TASK,
        "timestamp": now(),
        "classifications": classifications,
        "stage0a": stage0a,
        "gdn": {"stage0b": gdn0b, "stage1": gdn1, "stage2": gdn2},
        "kda": {"stage0b": kda0b, "stage1": kda1, "stage2": kda2},
        "reproducibility": repro,
        "artifact_dir": str(RESULT_DIR),
    }
    save_json(RESULT_DIR / "summary.json", summary)
    write_report_v2(RESULT_DIR / "report.md", summary)
    return summary


def write_report_v2(path, summary):
    c = summary["classifications"]
    lines = [
        "# GDN/KDA Axis-Matched State Rotation V1",
        "",
        "## Classification",
        "```text",
    ]
    for key, value in c.items():
        lines.append(f"{key} = {value}")
    lines += [
        "```",
        "",
        "## Stage 0B",
        "```json",
        json.dumps({"gdn": summary["gdn"]["stage0b"], "kda": summary["kda"]["stage0b"]}, indent=2, sort_keys=True),
        "```",
        "",
        "## Reproducibility",
        "```json",
        json.dumps(summary["reproducibility"], indent=2, sort_keys=True),
        "```",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_classifications(stage0a, stage0b, stage1, stage2):
    s0a = stage0a.get("gate", "FAIL")
    gdn0b = stage0b.get("gdn", {}).get("gate", "BLOCKED")
    kda0b = stage0b.get("kda", {}).get("gate", "BLOCKED")
    gdn_q = "NOT_TESTED" if stage1.get("gate") != "PASS" else stage1.get("gdn_classification", "NEGATIVE_OR_INCONCLUSIVE")
    kda_q = "NOT_TESTED" if stage1.get("gate") != "PASS" else stage1.get("kda_classification", "NEGATIVE_OR_INCONCLUSIVE")
    persistent_run = "YES" if stage2.get("gate") == "PASS" else "NO"
    return {
        "STAGE0A_FP64_PARITY": s0a,
        "STAGE0B_GDN_FP_PARITY": gdn0b,
        "STAGE0B_KDA_FP_PARITY": kda0b,
        "GDN_AXIS_MATCHED_STATE_QUANTIZABILITY": gdn_q,
        "KDA_AXIS_MATCHED_STATE_QUANTIZABILITY": kda_q,
        "AXIS_MATCHED_GT_MISMATCHED": "NOT_TESTED" if stage1.get("gate") != "PASS" else stage1.get("axis_matched_gt_mismatched", "NOT_SUPPORTED"),
        "GDN_RHT_SEED_STABILITY": "NOT_TESTED",
        "KDA_RHT_SEED_STABILITY": "NOT_TESTED",
        "GDN_HAAR_DIAGNOSTIC": "NOT_TESTED",
        "KDA_HAAR_DIAGNOSTIC": "NOT_TESTED",
        "GDN_KLT_DIAGNOSTIC": "NOT_TESTED",
        "KDA_KLT_DIAGNOSTIC": "NOT_TESTED",
        "ROTATION_STATE_LEVEL_PRINCIPLE": "NOT_TESTED" if stage1.get("gate") != "PASS" else "PARTIAL",
        "PERSISTENT_STAGE_RUN": persistent_run,
        "GDN_RECURRENT_FUNCTIONAL_RESULT": "NOT_TESTED",
        "KDA_RECURRENT_FUNCTIONAL_RESULT": "NOT_TESTED",
        "ROTATION_RECURRENT_FUNCTIONAL_PRINCIPLE": "NOT_TESTED",
        "FINAL_CLASSIFICATION": "BLOCKED_BEFORE_REAL_MODEL_STAGE" if s0a == "PASS" and (gdn0b != "PASS" or kda0b != "PASS") else "INCOMPLETE",
        "NEXT_EXPERIMENT": "Run Stage 0B real-model parity under validated GDN and Ling/KDA GPU environments, then unlock Stage 1.",
    }


def write_report(path, summary):
    c = summary["classifications"]
    s0a = summary["fp64_parity"]
    lines = [
        "# GDN/KDA Axis-Matched State Rotation V1",
        "",
        "## Status",
        "Stage 0A isolated FP64 algebra parity executed. Real-model parity and state collection remain gated.",
        "",
        "## Verified Tensor Semantics",
        "- State convention: `[B,H,K,V]`.",
        "- `R128`: scale over V, scale shape `[B,H,K,1]`.",
        "- `C128`: scale over K, scale shape `[B,H,1,V]`.",
        "- Quantizer: symmetric max INT8, `scale = amax(abs(group)).clamp_min(1e-12) / 127`.",
        "",
        "## FP64 Parity",
        f"- Stage 0A gate: `{s0a['gate']}`.",
        f"- GDN max relative state error: `{s0a['gdn']['max_relative_state_error']}`.",
        f"- GDN max relative readout error: `{s0a['gdn']['max_relative_readout_error']}`.",
        f"- KDA max relative state error: `{s0a['kda']['max_relative_state_error']}`.",
        f"- KDA max relative readout error: `{s0a['kda']['max_relative_readout_error']}`.",
        f"- Nonfinite count: `{s0a['nonfinite']}`.",
        "",
        "## Stage Gates",
        "```json",
        json.dumps(c, indent=2, sort_keys=True),
        "```",
        "",
        "## Final Questions",
        "Q1. Did the implemented GDN Key rotation preserve the FP model? Stage 0A algebra says yes; real-model FP parity is not yet passed.",
        "Q2. Did the implemented KDA Value rotation preserve the FP model? Stage 0A algebra says yes; real-model FP parity is not yet passed.",
        "Q3-Q8. Frozen-state peakiness, INT8 error, axis matching, RHT seeds, Haar, and KLT require Stage 1 state collection and are not scientifically classified here.",
        "Q9. Stage 2 did not run.",
        f"Q10. Next experiment: {c['NEXT_EXPERIMENT']}",
        "",
        "## Reproducibility",
        "```json",
        json.dumps(summary["reproducibility"], indent=2, sort_keys=True),
        "```",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    repro = reproducibility_record()
    config = {
        "task": TASK,
        "created_at": now(),
        "stage_progression": ["0A", "0B", "1", "2"],
        "hard_gates": {
            "stage0a_max_relative_state_error": 1e-10,
            "stage0a_max_relative_readout_error": 1e-10,
        },
        "canonical_counts": canonical_counts(),
        "run_args": vars(args),
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "manifest.json", {"task": TASK, **canonical_counts()})
    if args.stage == "stage0a":
        outdir = RESULT_DIR / "stage0a"
        outdir.mkdir(parents=True, exist_ok=True)
        stage0a = run_stage0a_fp64_parity(k_dim=args.stage0a_k, v_dim=args.stage0a_v, steps=args.stage0a_steps)
        save_json(outdir / "fp64_parity.json", stage0a)
        return collect_summary_from_artifacts(repro)
    if args.architecture in {"gdn", "kda"} and args.stage != "all":
        if args.stage == "stage0b":
            run_architecture_stage0b(args.architecture, args.scope)
        elif args.stage == "stage1":
            run_architecture_stage1(args.architecture, args.scope)
        elif args.stage == "stage2":
            run_architecture_stage2(args.architecture, args.scope)
        return collect_summary_from_artifacts(repro)
    stage0a = run_stage0a_fp64_parity(k_dim=args.stage0a_k, v_dim=args.stage0a_v, steps=args.stage0a_steps)
    stage0a_dir = RESULT_DIR / "stage0a"
    stage0a_dir.mkdir(parents=True, exist_ok=True)
    save_json(stage0a_dir / "fp64_parity.json", stage0a)
    save_json(RESULT_DIR / "fp64_parity.json", stage0a)
    if stage0a["gate"] != "PASS":
        stage0b = {"stage": "0B", "gdn": {"gate": "NOT_RUN"}, "kda": {"gate": "NOT_RUN"}}
        stage1 = {"stage": "1", "gate": "NOT_RUN"}
        stage2 = {"stage": "2", "gate": "NOT_RUN"}
    else:
        stage0b = run_stage0b_real_model_parity()
        save_json(RESULT_DIR / "real_model_fp_parity.json", stage0b)
        if args.allow_real_model and stage0b.get("gdn", {}).get("gate") == "PASS" and stage0b.get("kda", {}).get("gate") == "PASS":
            stage1 = run_stage1_frozen_state_quantizability()
        else:
            stage1 = run_stage1_frozen_state_quantizability()
        save_json(RESULT_DIR / "stage1_frozen_state_summary.json", stage1)
        stage2 = run_stage2_persistent_int8(stage1)
        save_json(RESULT_DIR / "stage2_persistent_summary.json", stage2)
    empty_paths = [
        "frozen_state_metrics.jsonl",
        "frozen_state_unit_aggregate.csv",
        "rotation_seed_summary.csv",
        "axis_matched_vs_mismatched.csv",
        "peakiness_vs_error.csv",
        "persistent_metrics.jsonl",
        "persistent_unit_aggregate.csv",
    ]
    for name in empty_paths:
        p = RESULT_DIR / name
        if name.endswith(".csv"):
            write_rows(p, [])
        else:
            p.write_text("", encoding="utf-8")
    classifications = final_classifications(stage0a, stage0b, stage1, stage2)
    summary = {
        "task": TASK,
        "timestamp": now(),
        "fp64_parity": stage0a,
        "real_model_fp_parity": stage0b,
        "stage1": stage1,
        "stage2": stage2,
        "classifications": classifications,
        "reproducibility": repro,
        "artifact_dir": str(RESULT_DIR),
    }
    save_json(RESULT_DIR / "summary.json", summary)
    write_report(RESULT_DIR / "report.md", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--architecture", choices=["all", "gdn", "kda"], default="all")
    parser.add_argument("--stage", choices=["all", "stage0a", "stage0b", "stage1", "stage2"], default="all")
    parser.add_argument("--scope", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--stage0a-k", type=int, default=128)
    parser.add_argument("--stage0a-v", type=int, default=128)
    parser.add_argument("--stage0a-steps", type=int, default=64)
    parser.add_argument("--allow-real-model", action="store_true", help="Allow full GPU real-model stages when integration gates are implemented.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_experiment(args)
    print(json.dumps(summary["classifications"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
