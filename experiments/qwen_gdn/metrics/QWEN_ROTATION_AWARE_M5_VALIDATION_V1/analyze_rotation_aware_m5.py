#!/usr/bin/env python3
"""Aggregate preregistered rotation-aware M5 component replay."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_ROTATION_AWARE_M5_VALIDATION_V1"
PRIOR = ROOT / "experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
LABELS = ROOT / "results/rotation/qwen_metric_alignment_audit_v1/canonical_trajectory_metrics/sample_metrics.jsonl"
MODEL = Path("/data/zypan/modelscope_models/Qwen3.5-9B")
BAD_PAIR = ("aime26_28", 2)
COND_NAME = {"INT8_C128": "Identity", "INT8_C128_KEY_HADAMARD": "Hadamard"}
METRICS = {
    "raw_M5": "m5_canonical",
    "M5_canonical": "m5_canonical",
    "FP_signal_normalized_M5": "m5_over_fp_signal",
    "M5_over_M0": "m5_over_m0",
}
COMPONENTS = [
    "m0", "m0_relative", "readout_error_norm", "rms_denominator_min", "rms_denominator_mean",
    "rms_jvp_norm", "gate_rms", "weighted_jvp_norm", "m5_canonical", "fp_signal_norm",
    "m5_over_m0", "m5_over_fp_signal", "readout_over_m0", "jvp_over_readout", "m5_over_jvp",
    "head_readout_concentration", "head_jvp_concentration", "head_projected_concentration",
]
RATIO_COMPONENTS = [
    "m0", "m0_relative", "readout_error_norm", "rms_denominator_min", "rms_denominator_mean",
    "rms_jvp_norm", "gate_rms", "weighted_jvp_norm", "m5_canonical", "fp_signal_norm",
    "m5_over_m0", "m5_over_fp_signal", "readout_over_m0", "jvp_over_readout", "m5_over_jvp",
]
TOP_COLUMNS = [
    "problem_id", "seed", "canonical_index", "condition", "timestep", "token_id", "layer",
    "m0", "m5_canonical", "fp_signal_norm", "m5_over_m0", "m5_over_fp_signal",
    "readout_error_norm", "rms_denominator_min", "rms_jvp_norm", "gate_rms", "weighted_jvp_norm",
    "head_readout_argmax", "head_readout_concentration", "head_jvp_argmax", "head_jvp_concentration",
    "head_projected_argmax", "head_projected_concentration",
]
SEED = 20260924
BOOT = 5000
PERM = 10000
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if math.isnan(float(x)):
            return None
        if math.isinf(float(x)):
            return "Infinity" if float(x) > 0 else "-Infinity"
        return float(x)
    return x


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def summary(a):
    raw = np.asarray(a)
    a = finite(raw)
    if not len(a):
        return {"n": int(len(raw)), "finite": 0, "nonfinite": int(len(raw)), "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(len(raw)), "finite": int(len(a)), "nonfinite": int(len(raw) - len(a)),
        "mean": float(a.mean()), "p50": float(np.quantile(a, .5)), "p90": float(np.quantile(a, .9)),
        "p95": float(np.quantile(a, .95)), "p99": float(np.quantile(a, .99)), "max": float(a.max()),
    }


def trajectory_summary(a):
    a = finite(a)
    p95 = float(np.quantile(a, .95))
    return {"mean": float(a.mean()), "p95": p95, "cvar95": float(a[a >= p95].mean()), "max": float(a.max()), "n": int(len(a))}


def rankdata(a):
    return pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)


def auroc(y, score):
    y, score = np.asarray(y, dtype=np.int8), np.asarray(score, dtype=np.float64)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    n1, n0 = int(y.sum()), int((1-y).sum())
    if not n1 or not n0:
        return None
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if not len(x) or not len(y):
        return None
    gt = sum(int((v > y).sum()) for v in x)
    lt = sum(int((v < y).sum()) for v in x)
    return float((gt-lt)/(len(x)*len(y)))


def bootstrap_auc(y, score):
    rng = np.random.default_rng(SEED)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(BOOT):
        idx = rng.integers(0, len(y), len(y))
        value = auroc(y[idx], score[idx])
        if value is not None:
            vals.append(value)
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def bootstrap_median(a):
    a = np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    vals = [np.median(a[rng.integers(0, len(a), len(a))]) for _ in range(BOOT)]
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h, c+h]


def sign_p(negative, positive):
    n = negative+positive
    if not n:
        return None
    k = min(negative, positive)
    return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/(2**n))


def permutation_one_sided(rescue, stable):
    rescue, stable = np.asarray(rescue), np.asarray(stable)
    observed = float(np.median(rescue)-np.median(stable))
    pool = np.concatenate([rescue, stable])
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(PERM):
        z = rng.permutation(pool)
        value = float(np.median(z[:len(rescue)])-np.median(z[len(rescue):]))
        hits += value <= observed
    return {"observed_rescue_minus_stable_failure_median": observed, "p_one_sided_more_negative": (hits+1)/(PERM+1), "permutations": PERM}


def load_labels():
    out = {}
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out.setdefault((row["problem_id"], int(row["seed"])), {})[row["condition"]] = bool(row["v4_correct"])
    return out


def rescue_group(d):
    i, h = d["int8_c128"], d["int8_c128_key_h"]
    if i and h: return "stable_correct"
    if not i and h: return "hadamard_rescue"
    if not i and not h: return "stable_failure"
    return "identity_only_correct"


def main():
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    prereg = OUT / "preregistration.json"
    if not prereg.exists():
        raise RuntimeError("missing preregistration")
    if (OUT / "results.json").exists():
        raise RuntimeError("refusing to overwrite formal results")
    statuses = [load_json(OUT / f"component_replay/shard_{i:02d}/status.json") for i in range(4)]
    if any(s.get("status") != "COMPLETE" or s.get("completed") != 15 for s in statuses):
        raise RuntimeError(f"component replay incomplete: {statuses}")
    parity = load_json(OUT / "source_audit/replay_parity.json")
    if parity["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("parity gate not passed")

    labels = load_labels()
    component_arrays = {c: defaultdict(list) for c in COND_NAME.values()}
    ratio_arrays = defaultdict(list)
    ratio_layer_arrays = defaultdict(lambda: defaultdict(list))
    layer_trajectory_rows = []
    sample_ratio_rows = []
    trajectory_rows = []
    top_raw = {c: pd.DataFrame() for c in COND_NAME.values()}
    top_norm = {c: pd.DataFrame() for c in COND_NAME.values()}
    head_counts = defaultdict(Counter)
    formal_audit_arrays = defaultdict(list)
    token_pairing_pass = True
    total_rows = 0

    files = sorted((OUT / "component_replay").glob("shard_*/*.parquet"), key=lambda p: int(p.name.split("_", 1)[0]))
    if len(files) != 60:
        raise RuntimeError(f"expected 60 component parquet files, got {len(files)}")
    for ordinal, path in enumerate(files, 1):
        df = pd.read_parquet(path)
        total_rows += len(df)
        problem, seed = str(df.iloc[0]["problem_id"]), int(df.iloc[0]["seed"])
        source_index = int(path.name.split("_", 1)[0])
        expected_rows = int(df["timestep"].nunique()) * 24 * 2
        if len(df) != expected_rows:
            raise RuntimeError(f"row count mismatch {path}: {len(df)} != {expected_rows}")
        for layer, ldf in df.groupby("layer"):
            a = ldf[ldf["condition"] == "INT8_C128"].sort_values("timestep")
            b = ldf[ldf["condition"] == "INT8_C128_KEY_HADAMARD"].sort_values("timestep")
            token_pairing_pass &= np.array_equal(a[["timestep", "token_id"]].to_numpy(), b[["timestep", "token_id"]].to_numpy())
        if (problem, seed) == BAD_PAIR:
            continue
        by_cond = {}
        for condition, candidate in COND_NAME.items():
            g = df[df["condition"] == condition].sort_values(["timestep", "layer"])
            by_cond[candidate] = g
            base = {
                "problem_id": problem, "seed": seed, "source_index": source_index, "candidate": candidate,
                "condition": condition, "fp_correct": labels[(problem, seed)]["fp_state"],
                "correct": labels[(problem, seed)]["int8_c128" if candidate == "Identity" else "int8_c128_key_h"],
                "rescue_group": rescue_group(labels[(problem, seed)]), "panel": "clean59",
            }
            for metric_name, column in METRICS.items():
                values = g[column].to_numpy(dtype=np.float64)
                stats = trajectory_summary(values)
                for stat, value in stats.items():
                    base[f"{metric_name}_{stat}"] = value
            trajectory_rows.append(base)
            for component in COMPONENTS:
                component_arrays[candidate][component].append(g[component].to_numpy(copy=True))
            for head_kind in ["readout", "jvp", "projected"]:
                counts = g[f"head_{head_kind}_argmax"].value_counts()
                for head, count in counts.items():
                    head_counts[(candidate, head_kind)][int(head)] += int(count)
            raw_keep = g.nlargest(50, "m5_canonical")[TOP_COLUMNS]
            norm_keep = g.nlargest(50, "m5_over_fp_signal")[TOP_COLUMNS]
            top_raw[candidate] = pd.concat([top_raw[candidate], raw_keep], ignore_index=True).nlargest(100, "m5_canonical")
            top_norm[candidate] = pd.concat([top_norm[candidate], norm_keep], ignore_index=True).nlargest(100, "m5_over_fp_signal")
            for layer, lg in g.groupby("layer"):
                row = {"problem_id": problem, "seed": seed, "source_index": source_index, "candidate": candidate, "layer": int(layer)}
                for component in RATIO_COMPONENTS:
                    s = summary(lg[component].to_numpy())
                    row[f"{component}_mean"] = s["mean"]
                    row[f"{component}_p95"] = s["p95"]
                    row[f"{component}_max"] = s["max"]
                layer_trajectory_rows.append(row)
        i, h = by_cond["Identity"], by_cond["Hadamard"]
        if not np.array_equal(i[["timestep", "layer", "token_id"]].to_numpy(), h[["timestep", "layer", "token_id"]].to_numpy()):
            raise RuntimeError(f"condition alignment failed {problem} {seed}")
        sample_row = {"problem_id": problem, "seed": seed, "source_index": source_index}
        for component in RATIO_COMPONENTS:
            iv = i[component].to_numpy(dtype=np.float64)
            hv = h[component].to_numpy(dtype=np.float64)
            lr = np.log10((np.maximum(hv, 0)+EPS)/(np.maximum(iv, 0)+EPS))
            ratio_arrays[component].append(lr.astype(np.float32))
            s = summary(lr)
            for stat in ["p50", "p90", "p95", "p99", "max"]:
                sample_row[f"log10_H_over_I_{component}_{stat}"] = s[stat]
            for layer in sorted(i["layer"].unique()):
                mask = i["layer"].to_numpy() == layer
                ratio_layer_arrays[component][int(layer)].append(lr[mask].astype(np.float32))
        sample_ratio_rows.append(sample_row)
        if source_index in (0, 20, 40, 59):
            for candidate, g in by_cond.items():
                audit = g[g["timestep"] < 512]
                for metric_name, column in METRICS.items():
                    formal_audit_arrays[(candidate, metric_name)].append(audit[column].to_numpy(copy=True))
        del df, by_cond
        if ordinal % 5 == 0:
            print(json.dumps({"event": "aggregate_progress", "files": ordinal, "total": len(files), "rows": total_rows}), flush=True)

    if not token_pairing_pass:
        raise RuntimeError("token pairing gate failed")
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_df.to_csv(OUT / "metric_analysis/trajectory_metrics.csv", index=False)
    pd.DataFrame(layer_trajectory_rows).to_csv(OUT / "coordinate_audit/layer_trajectory_metrics.csv", index=False)
    pd.DataFrame(sample_ratio_rows).to_csv(OUT / "coordinate_audit/sample_scale_ratios.csv", index=False)

    fp_files = sorted((OUT / "coordinate_audit/fp_equivalent_replay").glob("*.parquet"))
    fp_df = pd.concat([pd.read_parquet(p) for p in fp_files], ignore_index=True)
    fp_summary = {}
    coordinate_gate = {}
    for candidate, condition in [("Identity", "INT8_C128"), ("Hadamard", "INT8_C128_KEY_HADAMARD")]:
        g = fp_df[fp_df["condition"] == condition]
        fp_summary[candidate] = {}
        coordinate_gate[candidate] = {}
        for metric_name, column in METRICS.items():
            residual = g[column].to_numpy(dtype=np.float64)
            quantized = np.concatenate(formal_audit_arrays[(candidate, metric_name)]).astype(np.float64)
            rs, qs = summary(residual), summary(quantized)
            ratio = rs["p95"] / max(qs["p95"], EPS)
            fp_summary[candidate][metric_name] = {"fp_equivalent_residual": rs, "int8_same_points": qs, "p95_residual_over_int8": ratio}
            coordinate_gate[candidate][metric_name] = ratio <= .01
    denom = fp_df["fp_signal_norm"].to_numpy(dtype=np.float64)
    denom_audit = {
        "summary": summary(denom), "fraction_lt_1e8": float((denom < 1e-8).mean()),
        "fraction_lt_1e6": float((denom < 1e-6).mean()), "fraction_lt_1e4": float((denom < 1e-4).mean()),
        "pathology": bool((denom < 1e-8).mean() > .01),
    }
    dump_json(OUT / "coordinate_audit/fp_equivalent_coordinate_audit.json", {"sample_indices": [0,20,40,59], "tokens_each": 512, "metrics": fp_summary, "coordinate_gate": coordinate_gate, "fp_signal_denominator": denom_audit})

    component_rows = []
    component_summaries = {}
    scale_rows = []
    common_scale = {}
    for candidate in ["Identity", "Hadamard"]:
        component_summaries[candidate] = {}
        combined = {}
        for component in COMPONENTS:
            values = np.concatenate(component_arrays[candidate][component]).astype(np.float64)
            combined[component] = values
            s = summary(values)
            component_summaries[candidate][component] = s
            component_rows.append({"record_type": "component_distribution", "candidate": candidate, "component": component, **s})
        log_m0 = np.log10(np.maximum(combined["m0"], EPS))
        log_m5 = np.log10(np.maximum(combined["m5_canonical"], EPS))
        common_scale[candidate] = {"pearson_log10_m0_vs_log10_m5": float(np.corrcoef(log_m0, log_m5)[0,1])}
        del combined, log_m0, log_m5
    for component in RATIO_COMPONENTS:
        values = np.concatenate(ratio_arrays[component]).astype(np.float64)
        s = summary(values)
        scale_rows.append({"scope": "all_clean59_token_layer", "component": component, **s})
        component_rows.append({"record_type": "log10_H_over_I", "candidate": "Hadamard_vs_Identity", "component": component, **s})
    m0_ratio = np.concatenate(ratio_arrays["m0"]).astype(np.float64)
    m5_ratio = np.concatenate(ratio_arrays["m5_canonical"]).astype(np.float64)
    common_scale["paired_scale_shift"] = {"pearson_log10_H_over_I_m0_vs_m5": float(np.corrcoef(m0_ratio, m5_ratio)[0,1])}
    pd.DataFrame(component_rows).to_csv(OUT / "coordinate_audit/component_decomposition.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(OUT / "coordinate_audit/scale_shift_summary.csv", index=False)

    layer_scale_rows = []
    for layer in range(32):
        if layer % 4 == 3:
            continue
        row = {"layer": layer}
        for component in RATIO_COMPONENTS:
            values = np.concatenate(ratio_layer_arrays[component][layer]).astype(np.float64)
            s = summary(values)
            for stat in ["p50", "p90", "p95", "p99", "max"]:
                row[f"log10_H_over_I_{component}_{stat}"] = s[stat]
        layer_scale_rows.append(row)
    pd.DataFrame(layer_scale_rows).to_csv(OUT / "coordinate_audit/layer_scale.csv", index=False)

    head_rows = []
    for (candidate, kind), counts in sorted(head_counts.items()):
        total = sum(counts.values())
        for head, count in sorted(counts.items()):
            head_rows.append({"candidate": candidate, "primitive": kind, "head": head, "argmax_count": count, "argmax_fraction": count/total})
    pd.DataFrame(head_rows).to_csv(OUT / "coordinate_audit/head_localization.csv", index=False)

    tail_rows = []
    tail_summary = {}
    for candidate in ["Identity", "Hadamard"]:
        tail_summary[candidate] = {}
        for metric_name, column in [("raw_M5", "m5_canonical"), ("FP_signal_normalized_M5", "m5_over_fp_signal")]:
            values = np.concatenate(component_arrays[candidate][column]).astype(np.float64)
            values = values[np.isfinite(values) & (values >= 0)]
            desc = np.sort(values)[::-1]
            total = desc.sum()
            shares = {}
            for frac in [.05,.01,.001,.0001]:
                n = max(1, math.ceil(len(desc)*frac))
                share = float(desc[:n].sum()/total) if total else 0.0
                shares[str(frac)] = share
                tail_rows.append({"record_type": "tail_mass", "candidate": candidate, "metric": metric_name, "fraction": frac, "observations": n, "mass_share": share})
            status = "EXTREME" if shares["0.001"] >= .25 or shares["0.01"] >= .5 else ("MODERATE" if shares["0.01"] >= .2 or shares["0.05"] >= .5 else "DIFFUSE")
            tail_summary[candidate][metric_name] = {"status": status, "mass_shares": shares}
        for metric_name, frame, sort_col in [("raw_M5", top_raw[candidate], "m5_canonical"), ("FP_signal_normalized_M5", top_norm[candidate], "m5_over_fp_signal")]:
            for rank, (_, r) in enumerate(frame.nlargest(50, sort_col).iterrows(), 1):
                row = {"record_type": "top_offender", "candidate": candidate, "metric": metric_name, "rank": rank}
                row.update({k: r[k] for k in TOP_COLUMNS})
                tail_rows.append(row)
    pd.DataFrame(tail_rows).to_csv(OUT / "coordinate_audit/tail_localization.csv", index=False)

    within = {}
    for candidate in ["Identity", "Hadamard"]:
        cdf = trajectory_df[(trajectory_df["candidate"] == candidate) & (trajectory_df["fp_correct"] == True)].copy()
        y = (~cdf["correct"].astype(bool)).astype(int).to_numpy()
        within[candidate] = {"panel": "FP-correct", "n": len(cdf), "failures": int(y.sum()), "metrics": {}}
        for metric_name in METRICS:
            score = cdf[f"{metric_name}_p95"].to_numpy(dtype=np.float64)
            correct, failed = score[y == 0], score[y == 1]
            within[candidate]["metrics"][metric_name] = {
                "auroc": auroc(y, score), "auroc_bootstrap_95ci": bootstrap_auc(y, score),
                "spearman_risk_vs_failure": spearman(score, y), "median_correct": float(np.median(correct)),
                "median_failure": float(np.median(failed)), "cliffs_delta_failure_vs_correct": cliffs_delta(failed, correct),
            }
        dump_json(OUT / f"within_rotation_analysis/{candidate.lower()}.json", within[candidate])

    idf = trajectory_df[trajectory_df["candidate"] == "Identity"].set_index(["problem_id","seed"])
    hdf = trajectory_df[trajectory_df["candidate"] == "Hadamard"].set_index(["problem_id","seed"])
    pair_rows = []
    group_rows = []
    for key in idf.index.intersection(hdf.index):
        group = hdf.loc[key,"rescue_group"]
        group_rows.append({"problem_id": key[0], "seed": key[1], "group": group, "identity_correct": bool(idf.loc[key,"correct"]), "hadamard_correct": bool(hdf.loc[key,"correct"])})
        row = dict(group_rows[-1])
        for metric_name in METRICS:
            iv, hv = float(idf.loc[key,f"{metric_name}_p95"]), float(hdf.loc[key,f"{metric_name}_p95"])
            row[f"identity_{metric_name}_p95"] = iv; row[f"hadamard_{metric_name}_p95"] = hv; row[f"delta_{metric_name}_p95"] = hv-iv
        pair_rows.append(row)
    group_df, pair_df = pd.DataFrame(group_rows), pd.DataFrame(pair_rows)
    group_df.to_csv(OUT / "rescue_analysis/rescue_groups.csv", index=False)
    pair_df.to_csv(OUT / "rescue_analysis/paired_metric_deltas.csv", index=False)
    rescue = {"group_counts": group_df["group"].value_counts().to_dict(), "metrics": {}}
    for metric_name in METRICS:
        field = f"delta_{metric_name}_p95"
        groups = {}
        for group, g in pair_df.groupby("group"):
            values = g[field].to_numpy(dtype=np.float64)
            neg, pos, ties = int((values<0).sum()), int((values>0).sum()), int((values==0).sum())
            groups[group] = {"n": len(values), "improved": neg, "worsened": pos, "ties": ties, "p_improved": neg/len(values), "p_improved_wilson_95ci": wilson(neg,len(values)), "median_delta": float(np.median(values)), "median_delta_bootstrap_95ci": bootstrap_median(values), "two_sided_sign_p": sign_p(neg,pos)}
        rv = pair_df[pair_df["group"] == "hadamard_rescue"][field].to_numpy(dtype=np.float64)
        sv = pair_df[pair_df["group"] == "stable_failure"][field].to_numpy(dtype=np.float64)
        comp = permutation_one_sided(rv,sv); comp["cliffs_delta_rescue_vs_stable_failure"] = cliffs_delta(rv,sv)
        rescue["metrics"][metric_name] = {"groups": groups, "rescue_vs_stable_failure": comp}
    dump_json(OUT / "rescue_analysis/rescue_statistics.json", rescue)

    candidate_aggregates = {}
    gates = {}
    classifications = {}
    for metric_name in METRICS:
        iv = float(np.median(idf[f"{metric_name}_p95"].to_numpy(dtype=np.float64)))
        hv = float(np.median(hdf[f"{metric_name}_p95"].to_numpy(dtype=np.float64)))
        pair_direction = hv < iv
        coordinate_ok = coordinate_gate["Hadamard"][metric_name] and (metric_name != "FP_signal_normalized_M5" or not denom_audit["pathology"])
        within_ok = all(within[c]["metrics"][metric_name]["auroc"] > .5 and within[c]["metrics"][metric_name]["median_failure"] > within[c]["metrics"][metric_name]["median_correct"] for c in ["Identity","Hadamard"])
        rg = rescue["metrics"][metric_name]
        rescue_group_stats = rg["groups"]["hadamard_rescue"]
        comparison = rg["rescue_vs_stable_failure"]
        rescue_ok = rescue_group_stats["p_improved"] > .5 and comparison["observed_rescue_minus_stable_failure_median"] < 0 and comparison["cliffs_delta_rescue_vs_stable_failure"] < 0 and comparison["p_one_sided_more_negative"] < .10
        gates[metric_name] = {"coordinate_sanity": coordinate_ok, "within_rotation": within_ok, "pairwise_direction": pair_direction, "rescue_association": rescue_ok, "PAIRWISE_DIRECTION_CHECK": "CORRECT" if pair_direction else "INCORRECT"}
        candidate_aggregates[metric_name] = {"Identity": iv, "Hadamard": hv}
        if metric_name == "raw_M5":
            classification = "DIAGNOSTIC_ONLY" if not pair_direction else ("ROTATION_AWARE_PROMISING" if all([coordinate_ok,within_ok,pair_direction,rescue_ok]) else "INSUFFICIENT_EVIDENCE")
        elif not coordinate_ok:
            classification = "FAILED_COORDINATE_SANITY"
        elif not pair_direction:
            classification = "FAILED_CROSS_ROTATION"
        elif all([within_ok,rescue_ok]):
            classification = "ROTATION_AWARE_PROMISING"
        else:
            classification = "INSUFFICIENT_EVIDENCE"
        classifications[metric_name] = classification
    classifications["Delta_M5"] = "DIAGNOSTIC_ONLY"
    classifications["relative_M5_reference"] = "NOT_APPLICABLE"
    gates["Delta_M5"] = {"reason": "Exact FP-vs-self reference is zero, so Delta_M5 is algebraically identical to M5_canonical and is not independent."}
    gates["relative_M5_reference"] = {"reason": "The mathematically valid M5 reference is zero; dividing by finite-precision residual would create denominator pathology."}
    promising = [m for m,c in classifications.items() if c == "ROTATION_AWARE_PROMISING"]
    overall_gate = "PASS" if promising else "FAIL"

    stats = {"within_rotation": within, "rescue": rescue, "candidate_aggregates": candidate_aggregates, "gates": gates, "classifications": classifications, "common_scale": common_scale, "tail": tail_summary, "coordinate_audit": {"fp_equivalent": fp_summary, "denominator": denom_audit}}
    dump_json(OUT / "statistical_analysis.json", stats)
    result_rows = []
    for metric_name in ["raw_M5","M5_canonical","FP_signal_normalized_M5","M5_over_M0","Delta_M5","relative_M5_reference"]:
        row = {"metric": metric_name, "classification": classifications[metric_name]}
        row.update(candidate_aggregates.get(metric_name, {"Identity":None,"Hadamard":None}))
        row.update(gates.get(metric_name, {}))
        if metric_name in METRICS:
            row["identity_auroc"] = within["Identity"]["metrics"][metric_name]["auroc"]
            row["hadamard_auroc"] = within["Hadamard"]["metrics"][metric_name]["auroc"]
            row["rescue_p_improved"] = rescue["metrics"][metric_name]["groups"]["hadamard_rescue"]["p_improved"]
            row["rescue_vs_failure_permutation_p"] = rescue["metrics"][metric_name]["rescue_vs_stable_failure"]["p_one_sided_more_negative"]
        result_rows.append(row)
    pd.DataFrame(result_rows).to_csv(OUT / "results.csv", index=False)

    results = {
        "task": "QWEN_ROTATION_AWARE_M5_VALIDATION_V1", "ARTIFACT_AUDIT": "PASS",
        "REPLAY_PARITY_GATE": "PASS", "M5_EXISTING_IMPLEMENTATION_ALREADY_CANONICAL": True,
        "ROTATION_AWARE_M5_GATE": overall_gate, "promising_metrics": promising,
        "classifications": classifications, "gates": gates, "candidate_aggregates": candidate_aggregates,
        "common_scale": common_scale, "tail_summary": tail_summary,
        "PERSISTENT_PIPELINE_EXECUTED": False, "NEW_AIME_GENERATION": False,
        "NEW_ROTATION_TRAINING": False, "MODEL_MODIFIED": False, "QUANTIZER_MODIFIED": False, "SCORER_MODIFIED": False,
    }
    dump_json(OUT / "results.json", results)

    audit = {
        "ARTIFACT_AUDIT": "PASS", "paired_trajectories": 60, "primary_clean_pairs": 59,
        "token_pairing_exact": True, "formal_component_files": len(files), "formal_component_rows": total_rows,
        "model_config_sha256": sha256(MODEL / "config.json"),
        "checkpoint_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        "prior_capture_script_sha256": sha256(PRIOR / "run_capture.py"),
        "component_replay_script_sha256": sha256(OUT / "replay_rotation_aware_m5.py"),
        "preregistration_sha256": sha256(prereg), "frozen_scorer_labels_sha256": sha256(LABELS),
        "semantics": "Same prior batch-3 recurrent/QDQ implementation; observation-only norm hook; no labels loaded by replay workers.",
    }
    dump_json(OUT / "artifact_manifest.json", audit)

    # Quantitative component attribution: largest absolute median log scale shifts.
    scale_df = pd.DataFrame(scale_rows).sort_values("p50", key=lambda s: s.abs(), ascending=False)
    leading = scale_df.head(5)[["component","p50","p95","max"]].to_dict("records")
    best_rescue = min(METRICS, key=lambda m: rescue["metrics"][m]["rescue_vs_stable_failure"]["cliffs_delta_rescue_vs_stable_failure"])
    best_within = max(METRICS, key=lambda m: np.mean([within[c]["metrics"][m]["auroc"] for c in ["Identity","Hadamard"]]))
    report = f"""# Qwen Rotation-Aware M5 Validation V1\n\n## Executive conclusion\n\n`ROTATION_AWARE_M5_GATE = {overall_gate}`. Promising variants: `{', '.join(promising) if promising else 'NONE'}`. No persistent-KL replay was executed.\n\n## Coordinate and transformation audit\n\nThe prior “raw” M5 is already `M5_canonical`: Hadamard recurrent state is pulled back with the normalized symmetric orthogonal H128 before the error is formed, and the functional map uses the shared native FP query, RMS point, gate, norm weight and output projection. In exact arithmetic, `S_native=H S_rotated` and `q_rotated=q_native H` preserve the GDN readout. Consequently `M5_canonical` is not a newly repaired metric; it is an explicit name for the existing implementation.\n\nThe replay parity gate reproduced prior M0/M5 p95 to ~1e-15 and logit-derived scalars exactly. In the preregistered four-sample, 512-token FP-equivalence audit, Identity residual is zero. Hadamard finite-precision residuals and their ratios to the same-point INT8 risks are recorded in `fp_equivalent_coordinate_audit.json`. The shared FP-signal denominator has fraction <1e-8 = {denom_audit['fraction_lt_1e8']:.3%}, <1e-6 = {denom_audit['fraction_lt_1e6']:.3%}, and <1e-4 = {denom_audit['fraction_lt_1e4']:.3%}; denominator pathology = `{denom_audit['pathology']}`.\n\n## Primitive scale source\n\nThe largest median absolute Hadamard/Identity log10 shifts are:\n\n{json.dumps(leading, indent=2)}\n\nM0/M5 shared-scale evidence: log10(M0) versus log10(M5) Pearson is {common_scale['Identity']['pearson_log10_m0_vs_log10_m5']:.3f} in Identity and {common_scale['Hadamard']['pearson_log10_m0_vs_log10_m5']:.3f} in Hadamard; paired log10(H/I) M0 versus M5 Pearson is {common_scale['paired_scale_shift']['pearson_log10_H_over_I_m0_vs_m5']:.3f}. Layer, sample, token and head-localization tables are included without selecting favorable subsets.\n\n## Candidate metrics and four gates\n\n| Metric | Identity median trajectory p95 | Hadamard median trajectory p95 | Coordinate | Within both | Direction | Rescue | Classification |\n|---|---:|---:|---|---|---|---|---|\n"""
    for metric_name in METRICS:
        g = gates[metric_name]; c = candidate_aggregates[metric_name]
        report += f"| {metric_name} | {c['Identity']:.6g} | {c['Hadamard']:.6g} | {'PASS' if g['coordinate_sanity'] else 'FAIL'} | {'PASS' if g['within_rotation'] else 'FAIL'} | {g['PAIRWISE_DIRECTION_CHECK']} | {'PASS' if g['rescue_association'] else 'FAIL'} | {classifications[metric_name]} |\n"
    report += f"""\n`Delta_M5` is `DIAGNOSTIC_ONLY` because the exact reference is zero and it collapses to M5_canonical. `relative_M5_reference` is `NOT_APPLICABLE`: dividing by zero or by a tiny finite-precision coordinate residual would manufacture a denominator pathology.\n\n## Within-rotation diagnosis\n\nBest descriptive within-rotation metric among the preregistered variants: `{best_within}`. Both rotations are reported independently with AUROC, 5000-sample bootstrap CI, Spearman and Cliff's delta in `within_rotation_analysis/`. Raw M5 is compared directly rather than replaced by a favorable subset.\n\n## Rescue association\n\nGroups were reconstructed from true IDs: `{json.dumps(rescue['group_counts'], sort_keys=True)}`. Best descriptive rescue association: `{best_rescue}`. `paired_metric_deltas.csv` contains every clean pair; `rescue_statistics.json` reports P(delta<0), Wilson interval, sign test, one-sided 10,000-permutation comparison against stable failure, and Cliff's delta.\n\n## Decision\n\n`ROTATION_AWARE_M5_GATE = {overall_gate}`. {'At least one variant clears all preregistered gates; stop before persistent replay and await the next rotation-loss decision.' if overall_gate == 'PASS' else 'No mathematically justified M5 variant clears all four gates. The preserved persistent-functional pipeline is now eligible as a fallback, but this task stops here as instructed and does not launch it.'}\n\nNo model, quantizer, rotation, kernel semantics or scorer was modified; no new token was generated.\n"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")

    ended_at = datetime.now(timezone.utc).isoformat()
    required = [
        "preregistration.json", "artifact_manifest.json", "source_audit/replay_parity.json",
        "coordinate_audit/fp_equivalent_coordinate_audit.json", "coordinate_audit/component_decomposition.csv",
        "coordinate_audit/scale_shift_summary.csv", "coordinate_audit/layer_scale.csv",
        "coordinate_audit/sample_scale_ratios.csv", "coordinate_audit/head_localization.csv",
        "coordinate_audit/tail_localization.csv", "metric_analysis/trajectory_metrics.csv",
        "within_rotation_analysis/identity.json", "within_rotation_analysis/hadamard.json",
        "rescue_analysis/rescue_groups.csv", "rescue_analysis/paired_metric_deltas.csv",
        "rescue_analysis/rescue_statistics.json", "statistical_analysis.json", "results.json", "results.csv", "final_report.md",
    ]
    output_hashes = {rel: {"bytes": (OUT/rel).stat().st_size, "sha256": sha256(OUT/rel)} for rel in required}
    dump_json(OUT / "run_manifest.json", {
        "task": "QWEN_ROTATION_AWARE_M5_VALIDATION_V1", "status": "COMPLETE", "started_at": started_at, "completed_at": ended_at,
        "elapsed_seconds": time.time()-started, "host": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__, "analysis_script_sha256": sha256(Path(__file__)),
        "outputs": output_hashes, "persistent_pipeline_executed": False, "new_generation": False, "training": False,
    })

    print("QWEN_ROTATION_AWARE_M5_VALIDATION_V1")
    print("ARTIFACT_AUDIT = PASS")
    print("REPLAY_PARITY_GATE = PASS")
    print("M5_EXISTING_IMPLEMENTATION_ALREADY_CANONICAL = YES")
    print("DENOMINATOR_PATHOLOGY =", "YES" if denom_audit["pathology"] else "NO")
    print("BEST_WITHIN_ROTATION_DIAGNOSTIC =", best_within)
    print("BEST_RESCUE_ASSOCIATION_METRIC =", best_rescue)
    print("ROTATION_AWARE_M5_GATE =", overall_gate)
    print("PERSISTENT_PIPELINE_EXECUTED = NO")
    print("NEW_AIME_GENERATION = NO")
    print("NEW_ROTATION_TRAINING = NO")
    print("MODEL_MODIFIED = NO")
    print("QUANTIZER_MODIFIED = NO")
    print("SCORER_MODIFIED = NO")


if __name__ == "__main__":
    main()
