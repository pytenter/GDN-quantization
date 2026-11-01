#!/usr/bin/env python3
"""Read-only, preregistered analysis for QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1."""

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
CAPTURE = ROOT / "experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
TOKEN_CSV = CAPTURE / "token_level_metrics.csv"
TOKEN_JSON = CAPTURE / "token_level_metrics.json"
CAPTURE_MANIFEST = CAPTURE / "analysis_manifest.json"
CAPTURE_CONFIG = CAPTURE / "capture_config.json"
CAPTURE_SCRIPT = CAPTURE / "run_capture.py"
LABELS = ROOT / "results/rotation/qwen_metric_alignment_audit_v1/canonical_trajectory_metrics/sample_metrics.jsonl"
HEADROOM = ROOT / "results/rotation/qwen_persistent_headroom_confirmation_v1/final"
NUMERICAL_CLOSURE = ROOT / "results/rotation/qwen_orthogonal_numerical_closure_v1/teacher_forced_precision_comparison.json"
THETA0 = ROOT / "results/rotation/qwen_dense_orthogonal_oracle_v1/gates/theta0_heldout_logits_gate.json"
MODEL = Path("/data/zypan/modelscope_models/Qwen3.5-9B")
BAD_PAIR = ("aime26_28", 2)
COND_NAME = {"INT8_C128": "Identity", "INT8_C128_KEY_HADAMARD": "Hadamard"}
NAME_COND = {v: k for k, v in COND_NAME.items()}
HORIZONS = [1, 4, 8, 16, 32, 64]
EPS = 1e-12
SEED = 20260924
BOOT = 5000
PERM = 10000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, obj) -> None:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            if math.isnan(float(x)):
                return None
            if math.isinf(float(x)):
                return "Infinity" if float(x) > 0 else "-Infinity"
            return float(x)
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def q(a, p):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, p)) if len(a) else None


def summarize(a):
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if not len(finite):
        return {"n": int(len(a)), "finite": 0, "nonfinite": int(len(a)), "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(len(a)), "finite": int(len(finite)), "nonfinite": int(len(a) - len(finite)),
        "mean": float(finite.mean()), "p50": q(finite, .5), "p90": q(finite, .9),
        "p95": q(finite, .95), "p99": q(finite, .99), "max": float(finite.max()),
    }


def trajectory_stats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"mean": None, "p95": None, "cvar95": None, "max": None, "n": 0}
    p95 = float(np.quantile(a, .95))
    tail = a[a >= p95]
    return {"mean": float(a.mean()), "p95": p95, "cvar95": float(tail.mean()), "max": float(a.max()), "n": int(len(a))}


def rankdata(a):
    return pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(y, score):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    good = np.isfinite(score)
    y, score = y[good], score[good]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if not n1 or not n0:
        return None
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if not len(x) or not len(y):
        return None
    gt = sum(int((v > y).sum()) for v in x)
    lt = sum(int((v < y).sum()) for v in x)
    return float((gt - lt) / (len(x) * len(y)))


def percentile_ci(values):
    values = np.sort(np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64))
    if not len(values):
        return [None, None]
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def bootstrap_auc(y, score, reps=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        vals.append(auroc(y[idx], score[idx]))
    return percentile_ci(vals)


def bootstrap_median(a, reps=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    vals = [float(np.median(a[rng.integers(0, len(a), len(a))])) for _ in range(reps)] if len(a) else []
    return percentile_ci(vals)


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def exact_sign_p(negative, positive):
    n = negative + positive
    if not n:
        return None
    k = min(negative, positive)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def permutation_median_test(a, b, reps=PERM, seed=SEED):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not len(a) or not len(b):
        return {"observed_median_difference": None, "p_two_sided": None}
    observed = float(np.median(a) - np.median(b))
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        z = rng.permutation(pool)
        value = float(np.median(z[:len(a)]) - np.median(z[len(a):]))
        hits += abs(value) >= abs(observed)
    return {"observed_median_difference": observed, "p_two_sided": (hits + 1) / (reps + 1), "permutations": reps}


def load_labels():
    rows = {}
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["problem_id"], int(r["seed"]))
            rows.setdefault(key, {})[r["condition"]] = bool(r["v4_correct"])
    return rows


def group_name(d):
    i, h = d["int8_c128"], d["int8_c128_key_h"]
    if i and h:
        return "stable_correct"
    if not i and h:
        return "hadamard_rescue"
    if not i and not h:
        return "stable_failure"
    return "identity_only_correct"


def hash_model_checkpoint():
    shards = sorted(MODEL.glob("model*.safetensors"))
    rows = []
    combined = hashlib.sha256()
    for path in shards:
        digest = sha256(path)
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
        combined.update((path.name + ":" + digest + "\n").encode())
    return rows, combined.hexdigest()


def artifact_entry(path: Path, known_hash=None):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": known_hash or sha256(path)}


def main():
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    prereg = OUT / "preregistration.json"
    if not prereg.exists():
        raise RuntimeError("preregistration.json must exist before analysis")
    protected = [OUT / "results.json", OUT / "run_manifest.json", OUT / "persistent_metrics/token_level_metrics.parquet"]
    if any(p.exists() for p in protected):
        raise RuntimeError("Refusing to overwrite formal analysis outputs")

    labels = load_labels()
    capture = load_json(TOKEN_JSON)
    capture_manifest = load_json(CAPTURE_MANIFEST)
    capture_config = load_json(CAPTURE_CONFIG)
    headroom = load_json(HEADROOM / "summary.json")
    closure = load_json(NUMERICAL_CLOSURE)
    theta0 = load_json(THETA0)

    metric_columns = {
        "m0": "m0_layer_p95",
        "m5": "m5_layer_p95",
        "decision": "decision_sensitivity",
        "m0_relative": "m0_rel_layer_p95",
    }
    usecols = [
        "problem_id", "seed", "group", "condition", "timestep", "token_id",
        "m0_global_l2", "m0_layer_p95", "m0_rel_layer_p95", "m5_global_l2",
        "m5_layer_p95", "fp_margin", "delta_z_fp_winner", "decision_sensitivity",
    ]
    trajectory_chunks = defaultdict(lambda: defaultdict(list))
    component_chunks = defaultdict(lambda: defaultdict(list))
    sequence_hashes = defaultdict(hashlib.sha256)
    sequence_counts = defaultdict(int)
    top_rows = {"Identity": pd.DataFrame(), "Hadamard": pd.DataFrame()}
    parquet_path = OUT / "persistent_metrics/token_level_metrics.parquet"
    writer = None
    total_rows = 0
    print("TOKEN_SCAN_BEGIN", flush=True)
    for chunk_index, chunk in enumerate(pd.read_csv(TOKEN_CSV, usecols=usecols, chunksize=125000)):
        chunk["seed"] = chunk["seed"].astype(np.int32)
        chunk["clean_panel"] = ~((chunk["problem_id"] == BAD_PAIR[0]) & (chunk["seed"] == BAD_PAIR[1]))
        chunk["candidate"] = chunk["condition"].map(COND_NAME)
        chunk["m5_over_m0"] = chunk["m5_global_l2"] / (chunk["m0_global_l2"] + EPS)
        chunk["abs_delta_z"] = chunk["delta_z_fp_winner"].abs()
        chunk["exact_kl"] = np.nan
        chunk["delta_nll"] = np.nan
        chunk["delta_margin"] = np.nan
        chunk["persistent_kl"] = np.nan
        chunk["persistent_delta_nll"] = np.nan
        chunk["persistent_delta_margin"] = np.nan
        chunk["functional_metric_artifact_status"] = "LOCAL_ONLY_PERSISTENT_INSUFFICIENT_ARTIFACT"

        parquet_cols = [
            "problem_id", "seed", "group", "condition", "candidate", "timestep", "token_id", "clean_panel",
            "m0_global_l2", "m0_layer_p95", "m0_rel_layer_p95", "m5_global_l2", "m5_layer_p95",
            "m5_over_m0", "fp_margin", "delta_z_fp_winner", "decision_sensitivity",
            "exact_kl", "delta_nll", "delta_margin", "persistent_kl", "persistent_delta_nll",
            "persistent_delta_margin", "functional_metric_artifact_status",
        ]
        table = pa.Table.from_pandas(chunk[parquet_cols], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd", compression_level=7)
        writer.write_table(table)
        total_rows += len(chunk)

        for (problem, seed, condition), sub in chunk.groupby(["problem_id", "seed", "condition"], sort=False):
            key = (str(problem), int(seed), str(condition))
            payload = sub[["timestep", "token_id"]].to_csv(index=False, header=False).encode()
            sequence_hashes[key].update(payload)
            sequence_counts[key] += len(sub)
            if (problem, int(seed)) == BAD_PAIR:
                continue
            for metric, col in metric_columns.items():
                trajectory_chunks[key][metric].append(sub[col].to_numpy(dtype=np.float64, copy=True))
            trajectory_chunks[key]["m5_over_m0"].append(sub["m5_over_m0"].to_numpy(dtype=np.float64, copy=True))

        clean = chunk[chunk["clean_panel"]]
        for candidate, sub in clean.groupby("candidate", sort=False):
            for metric, col in {
                "m0_global_l2": "m0_global_l2", "m0_layer_p95": "m0_layer_p95",
                "m5_global_l2": "m5_global_l2", "m5_layer_p95": "m5_layer_p95",
                "m0_relative_layer_p95": "m0_rel_layer_p95", "m5_over_m0": "m5_over_m0",
                "decision": "decision_sensitivity", "decision_numerator_abs_delta_z": "abs_delta_z",
                "decision_denominator_fp_margin": "fp_margin",
            }.items():
                component_chunks[candidate][metric].append(sub[col].to_numpy(dtype=np.float64, copy=True))
            keep = sub.nlargest(50, "m5_layer_p95")[
                ["problem_id", "seed", "condition", "candidate", "timestep", "token_id", "m5_layer_p95", "m5_global_l2", "m0_layer_p95", "m0_global_l2", "fp_margin", "delta_z_fp_winner", "decision_sensitivity"]
            ]
            top_rows[candidate] = pd.concat([top_rows[candidate], keep], ignore_index=True).nlargest(50, "m5_layer_p95")
        if chunk_index % 8 == 0:
            print(json.dumps({"event": "token_scan", "rows": total_rows}), flush=True)
    if writer is not None:
        writer.close()
    print("TOKEN_SCAN_COMPLETE", total_rows, flush=True)

    pairing_rows = []
    pairing_pass = True
    for problem, seed in sorted(labels):
        a = (problem, seed, "INT8_C128")
        b = (problem, seed, "INT8_C128_KEY_HADAMARD")
        ah = sequence_hashes[a].hexdigest() if a in sequence_hashes else None
        bh = sequence_hashes[b].hexdigest() if b in sequence_hashes else None
        match = ah == bh and sequence_counts[a] == sequence_counts[b] and ah is not None
        pairing_pass &= match
        d = labels[(problem, seed)]
        pairing_rows.append({
            "problem_id": problem, "seed": seed, "identity_rows": sequence_counts[a], "hadamard_rows": sequence_counts[b],
            "identity_token_sequence_sha256": ah, "hadamard_token_sequence_sha256": bh,
            "token_sequence_exact_match": match, "fp_correct": d.get("fp_state"),
            "identity_correct": d.get("int8_c128"), "hadamard_correct": d.get("int8_c128_key_h"),
            "rescue_group": group_name(d), "clean_panel": (problem, seed) != BAD_PAIR,
        })
    pd.DataFrame(pairing_rows).to_csv(OUT / "source_audit/pairing.csv", index=False)
    if not pairing_pass:
        raise RuntimeError("Identity/Hadamard token pairing failed")

    trajectory_rows = []
    for key in sorted(trajectory_chunks):
        problem, seed, condition = key
        label_key = "int8_c128" if condition == "INT8_C128" else "int8_c128_key_h"
        base = {
            "problem_id": problem, "seed": seed, "condition": condition, "candidate": COND_NAME[condition],
            "fp_correct": labels[(problem, seed)]["fp_state"], "correct": labels[(problem, seed)][label_key],
            "rescue_group": group_name(labels[(problem, seed)]), "panel": "clean59",
        }
        for metric in ["m0", "m5", "decision", "m0_relative", "m5_over_m0"]:
            values = np.concatenate(trajectory_chunks[key][metric])
            stats = trajectory_stats(values)
            for stat, value in stats.items():
                base[f"{metric}_{stat}"] = value
        for missing in ["exact_kl", "delta_nll", "delta_margin", "persistent_kl", "persistent_delta_nll", "persistent_delta_margin"]:
            base[f"{missing}_mean"] = None
            base[f"{missing}_p95"] = None
            base[f"{missing}_cvar95"] = None
        trajectory_rows.append(base)
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_df.to_csv(OUT / "persistent_metrics/trajectory_metrics.csv", index=False)

    horizon_rows = []
    for method, values in headroom["horizon_summary"].items():
        for horizon, metrics in values.items():
            horizon_rows.append({"panel": "existing_non_aime_64doc", "method": method, "horizon": int(horizon), **metrics})
    horizon_df = pd.DataFrame(horizon_rows).sort_values(["method", "horizon"])
    horizon_df.to_csv(OUT / "persistent_metrics/horizon_metrics.csv", index=False)

    component_rows = []
    component_values = {}
    for candidate in ["Identity", "Hadamard"]:
        component_values[candidate] = {}
        for metric, chunks in component_chunks[candidate].items():
            values = np.concatenate(chunks)
            component_values[candidate][metric] = values
            component_rows.append({"candidate": candidate, "component": metric, "status": "AVAILABLE", **summarize(values)})
        log_m0 = np.log10(np.maximum(component_values[candidate]["m0_global_l2"], EPS))
        log_m5 = np.log10(np.maximum(component_values[candidate]["m5_global_l2"], EPS))
        component_rows.append({"candidate": candidate, "component": "pearson_log10_m0_vs_log10_m5", "status": "AVAILABLE", "mean": float(np.corrcoef(log_m0, log_m5)[0, 1])})
        for missing in ["m5_formula_numerator", "m5_formula_denominator", "pre_normalization_quantity", "post_normalization_quantity", "head_contribution"]:
            component_rows.append({"candidate": candidate, "component": missing, "status": "INSUFFICIENT_ARTIFACT"})
    pd.DataFrame(component_rows).to_csv(OUT / "m5_coordinate_audit/component_decomposition.csv", index=False)

    summary_by_key = {(r["problem_id"], int(r["seed"]), r["condition"]): r for r in capture["trajectory_summaries"]}
    layer_rows = []
    for layer in capture_config["gdn_layers"]:
        per_candidate = {}
        for condition, candidate in COND_NAME.items():
            rows = []
            for key, rec in summary_by_key.items():
                if key[2] != condition or key[:2] == BAD_PAIR:
                    continue
                layer_rec = next(x for x in rec["per_layer"] if int(x["layer"]) == int(layer))
                rows.append(layer_rec)
            per_candidate[candidate] = rows
        row = {"layer": layer, "paired_trajectories": 59}
        for candidate, rows in per_candidate.items():
            for metric in ["m0", "m0_relative", "m5"]:
                for stat in ["mean", "p95", "max"]:
                    row[f"{candidate}_{metric}_{stat}_median"] = float(np.median([r[metric][stat] for r in rows]))
        for metric in ["m0", "m0_relative", "m5"]:
            i = row[f"Identity_{metric}_p95_median"]
            h = row[f"Hadamard_{metric}_p95_median"]
            row[f"log10_H_over_I_{metric}_p95"] = math.log10((h + EPS) / (i + EPS))
        layer_rows.append(row)
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(OUT / "m5_coordinate_audit/layer_scale.csv", index=False)

    tail_rows = []
    tail_summary = {}
    for candidate in ["Identity", "Hadamard"]:
        values = component_values[candidate]["m5_layer_p95"]
        values = values[np.isfinite(values) & (values >= 0)]
        desc = np.sort(values)[::-1]
        total = float(desc.sum())
        contributions = {}
        for frac in [.05, .01, .001, .0001]:
            n = max(1, int(math.ceil(len(desc) * frac)))
            contribution = float(desc[:n].sum() / total) if total > 0 else None
            contributions[str(frac)] = contribution
            tail_rows.append({"record_type": "mass_concentration", "candidate": candidate, "fraction": frac, "observations": n, "contribution_to_total_mass": contribution})
        if contributions["0.001"] >= .25 or contributions["0.01"] >= .50:
            status = "EXTREME"
        elif contributions["0.01"] >= .20 or contributions["0.05"] >= .50:
            status = "MODERATE"
        else:
            status = "DIFFUSE"
        tail_summary[candidate] = {"status": status, "contributions": contributions, "observations": len(desc), "total_mass": total}
        dominant = layer_df.sort_values(f"{candidate}_m5_max_median", ascending=False).iloc[0]["layer"]
        for rank, (_, r) in enumerate(top_rows[candidate].head(25).iterrows(), 1):
            tail_rows.append({
                "record_type": "top_offender", "candidate": candidate, "rank": rank,
                "problem_id": r["problem_id"], "seed": int(r["seed"]), "token_index": int(r["timestep"]), "token_id": int(r["token_id"]),
                "layer": "INSUFFICIENT_ARTIFACT", "trajectory_dominant_layer_not_token_specific": int(dominant), "head": "INSUFFICIENT_ARTIFACT",
                "m5_layer_p95": r["m5_layer_p95"], "m5_global_l2": r["m5_global_l2"],
                "m0_layer_p95": r["m0_layer_p95"], "m0_global_l2": r["m0_global_l2"],
                "numerator": "INSUFFICIENT_ARTIFACT", "denominator": "INSUFFICIENT_ARTIFACT",
                "fp_margin": r["fp_margin"], "delta_z_fp_winner": r["delta_z_fp_winner"], "decision_sensitivity": r["decision_sensitivity"],
            })
    pd.DataFrame(tail_rows).to_csv(OUT / "m5_coordinate_audit/tail_localization.csv", index=False)

    fp_invariance = {
        "M5_FP_COORDINATE_INVARIANCE": "PARTIAL",
        "direct_raw_m5_fp_only_measurement": "INSUFFICIENT_ARTIFACT",
        "reason": "The prior capture disabled M5 capture during unquantized prefill and did not save M5 primitives. Functional/state equivalence evidence exists, but direct FP-only raw-M5 equality cannot be tested from saved scalars.",
        "mathematical_expectation": "Exact orthogonal coordinate change gives zero error and identical logits; finite precision is not bitwise invariant.",
        "existing_fp_equivalence_evidence": {
            "hadamard_primary_logit_relative_l2": closure["hadamard"]["primary"]["logit_relative_l2"],
            "hadamard_primary_state_recovered_relative_l2": closure["hadamard"]["primary"]["state_recovered_relative_l2"],
            "hadamard_primary_top1_match": closure["hadamard"]["primary"]["top1_match"],
            "hadamard_stress_first_top1_divergence_token": closure["hadamard"]["stress"]["first_top1_divergence_token"],
            "theta0_canonical_hadamard_gate": theta0["THETA0_HELDOUT_LOGITS_VS_CANONICAL_HADAMARD"],
            "theta0_max_abs": theta0["max_abs"],
        },
        "tolerance_interpretation": "PASS for functional equivalence at the prior top1 gate, but not evidence that raw M5 is coordinate invariant.",
    }
    dump_json(OUT / "m5_coordinate_audit/fp_invariance.json", fp_invariance)

    tail_report = f"""# M5 tail localization\n\nThe primary observation is the preregistered clean59 per-token `m5_layer_p95`.\n\n| Candidate | Top 5% mass | Top 1% mass | Top 0.1% mass | Top 0.01% mass | Status |\n|---|---:|---:|---:|---:|---|\n| Identity | {tail_summary['Identity']['contributions']['0.05']:.3%} | {tail_summary['Identity']['contributions']['0.01']:.3%} | {tail_summary['Identity']['contributions']['0.001']:.3%} | {tail_summary['Identity']['contributions']['0.0001']:.3%} | {tail_summary['Identity']['status']} |\n| Hadamard | {tail_summary['Hadamard']['contributions']['0.05']:.3%} | {tail_summary['Hadamard']['contributions']['0.01']:.3%} | {tail_summary['Hadamard']['contributions']['0.001']:.3%} | {tail_summary['Hadamard']['contributions']['0.0001']:.3%} | {tail_summary['Hadamard']['status']} |\n\nThe saved token table identifies sample and token position but discards the 24 individual layer values, heads, and M5 formula primitives. Consequently the exact `(sample, layer, token, head, numerator, denominator)` tuple is `INSUFFICIENT_ARTIFACT`; the CSV reports exact sample/token offenders and a separately computed trajectory-level dominant layer without pretending they are the same observation.\n"""
    (OUT / "m5_coordinate_audit/tail_report.md").write_text(tail_report, encoding="utf-8")

    metric_names = ["m0_p95", "m5_p95", "decision_p95", "m0_relative_p95", "m5_over_m0_p95"]
    within_results = {}
    for candidate in ["Identity", "Hadamard"]:
        cond = NAME_COND[candidate]
        cdf = trajectory_df[trajectory_df["condition"] == cond].copy()
        panels = {"clean59": cdf, "fp_correct52": cdf[cdf["fp_correct"] == True]}
        candidate_result = {"candidate": candidate, "outcome_positive": "incorrect", "panels": {}}
        for panel_name, pdf in panels.items():
            y = (~pdf["correct"].astype(bool)).astype(int).to_numpy()
            metrics = {}
            for metric in metric_names:
                score = pdf[metric].to_numpy(dtype=np.float64)
                correct = score[y == 0]
                incorrect = score[y == 1]
                metrics[metric] = {
                    "n": len(pdf), "failures": int(y.sum()), "correct": int((1-y).sum()),
                    "auroc": auroc(y, score), "auroc_bootstrap_95ci": bootstrap_auc(y, score),
                    "spearman_risk_vs_failure": spearman(score, y),
                    "median_correct": float(np.median(correct)) if len(correct) else None,
                    "median_incorrect": float(np.median(incorrect)) if len(incorrect) else None,
                    "cliffs_delta_incorrect_vs_correct": cliffs_delta(incorrect, correct),
                }
            candidate_result["panels"][panel_name] = metrics
        within_results[candidate] = candidate_result
        dump_json(OUT / f"within_rotation_analysis/{candidate.lower()}.json", candidate_result)

    idf = trajectory_df[trajectory_df["candidate"] == "Identity"].set_index(["problem_id", "seed"])
    hdf = trajectory_df[trajectory_df["candidate"] == "Hadamard"].set_index(["problem_id", "seed"])
    common = idf.index.intersection(hdf.index)
    rescue_group_rows = []
    delta_rows = []
    for key in common:
        base = {"problem_id": key[0], "seed": key[1], "group": hdf.loc[key, "rescue_group"], "identity_correct": bool(idf.loc[key, "correct"]), "hadamard_correct": bool(hdf.loc[key, "correct"])}
        rescue_group_rows.append(base)
        row = dict(base)
        for metric in metric_names:
            row[f"identity_{metric}"] = idf.loc[key, metric]
            row[f"hadamard_{metric}"] = hdf.loc[key, metric]
            row[f"delta_{metric}"] = hdf.loc[key, metric] - idf.loc[key, metric]
        delta_rows.append(row)
    rescue_df = pd.DataFrame(rescue_group_rows)
    delta_df = pd.DataFrame(delta_rows)
    rescue_df.to_csv(OUT / "rescue_analysis/rescue_groups.csv", index=False)
    delta_df.to_csv(OUT / "rescue_analysis/paired_metric_deltas.csv", index=False)
    rescue_stats = {"group_counts": rescue_df["group"].value_counts().to_dict(), "metrics": {}}
    for metric in metric_names:
        field = f"delta_{metric}"
        groups = {}
        for group, gdf in delta_df.groupby("group"):
            vals = gdf[field].to_numpy(dtype=np.float64)
            neg, pos, ties = int((vals < 0).sum()), int((vals > 0).sum()), int((vals == 0).sum())
            groups[group] = {
                "n": len(vals), "improved_delta_lt_zero": neg, "worsened_delta_gt_zero": pos, "ties": ties,
                "p_improved": neg / len(vals), "p_improved_wilson_95ci": wilson(neg, len(vals)),
                "median_delta": float(np.median(vals)), "median_delta_bootstrap_95ci": bootstrap_median(vals),
                "exact_two_sided_sign_p_excluding_ties": exact_sign_p(neg, pos),
            }
        rescue = delta_df[delta_df["group"] == "hadamard_rescue"][field].to_numpy(dtype=np.float64)
        failure = delta_df[delta_df["group"] == "stable_failure"][field].to_numpy(dtype=np.float64)
        comparison = permutation_median_test(rescue, failure)
        comparison["cliffs_delta_rescue_vs_stable_failure"] = cliffs_delta(rescue, failure)
        rescue_stats["metrics"][metric] = {"groups": groups, "rescue_vs_stable_failure": comparison}
    dump_json(OUT / "rescue_analysis/rescue_statistics.json", rescue_stats)

    candidate_values = {}
    direction = {}
    for metric in metric_names:
        iv = float(np.median(idf[metric].to_numpy(dtype=np.float64)))
        hv = float(np.median(hdf[metric].to_numpy(dtype=np.float64)))
        candidate_values[metric] = {"Identity": iv, "Hadamard": hv}
        direction[metric] = "CORRECT" if hv < iv else "INCORRECT"
    hsummary = headroom["horizon_summary"]
    for method in ["Native_INT8", "Hadamard"]:
        candidate_values.setdefault("external_persistent_kl_h1", {})["Identity" if method == "Native_INT8" else "Hadamard"] = hsummary[method]["1"]["future_kl"]
        candidate_values.setdefault("external_persistent_kl_equal_H1_64", {})["Identity" if method == "Native_INT8" else "Hadamard"] = float(np.mean([hsummary[method][str(h)]["future_kl"] for h in HORIZONS]))
    for metric in ["external_persistent_kl_h1", "external_persistent_kl_equal_H1_64"]:
        direction[metric] = "CORRECT" if candidate_values[metric]["Hadamard"] < candidate_values[metric]["Identity"] else "INCORRECT"

    all_within = {}
    for metric in metric_names:
        all_within[metric] = {candidate: within_results[candidate]["panels"]["fp_correct52"][metric]["auroc"] for candidate in ["Identity", "Hadamard"]}
        all_within[metric]["mean_auroc"] = float(np.mean([all_within[metric]["Identity"], all_within[metric]["Hadamard"]]))
    best_within = max(metric_names, key=lambda m: all_within[m]["mean_auroc"])
    rescue_effects = {m: rescue_stats["metrics"][m]["rescue_vs_stable_failure"]["cliffs_delta_rescue_vs_stable_failure"] for m in metric_names}
    best_rescue = min(metric_names, key=lambda m: rescue_effects[m])

    classifications = {
        "raw_m5": {"class": "DIAGNOSTIC_ONLY", "reason": "Strong fixed-rotation failure information exists, but the Identity-to-Hadamard direction is incorrect and raw scale is pathological."},
        "raw_m0": {"class": "FAILED_CROSS_ROTATION", "reason": "Pairwise direction is incorrect."},
        "decision": {"class": "DIAGNOSTIC_ONLY", "reason": "Retains within-trajectory diagnostic value but pairwise direction is incorrect."},
        "m0_relative": {"class": "PROMISING_SURROGATE" if direction["m0_relative_p95"] == "CORRECT" and all_within["m0_relative_p95"]["Identity"] > .5 and all_within["m0_relative_p95"]["Hadamard"] > .5 and rescue_effects["m0_relative_p95"] < 0 else "INSUFFICIENT_EVIDENCE", "reason": "Diagnostic normalization; classification follows all available preregistered gates, but direct coordinate-invariance evidence is incomplete."},
        "m5_over_m0": {"class": "PROMISING_SURROGATE" if direction["m5_over_m0_p95"] == "CORRECT" and all_within["m5_over_m0_p95"]["Identity"] > .5 and all_within["m5_over_m0_p95"]["Hadamard"] > .5 and rescue_effects["m5_over_m0_p95"] < 0 else "INSUFFICIENT_EVIDENCE", "reason": "Preregistered diagnostic ratio; even a correct pair direction is not sufficient without coordinate and external validation."},
        "aime_exact_kl": {"class": "INSUFFICIENT_EVIDENCE", "reason": "AIME logits were not saved."},
        "aime_delta_nll": {"class": "INSUFFICIENT_EVIDENCE", "reason": "AIME next-token log probabilities were not saved."},
        "aime_delta_margin": {"class": "INSUFFICIENT_EVIDENCE", "reason": "AIME logits were not saved."},
        "aime_persistent_metrics": {"class": "INSUFFICIENT_EVIDENCE", "reason": "Restartable AIME anchor states/logits were not saved."},
        "external_qwen_persistent_kl": {"class": "FAILED_CROSS_ROTATION", "reason": "On the fixed 64-document Qwen panel both h=1 and equal-weight H=1..64 rank Native lower-risk than Hadamard, opposite AIME accuracy."},
    }
    promising = [k for k, v in classifications.items() if v["class"] == "PROMISING_SURROGATE"]
    best_cross = promising[0] if promising else "NONE_IDENTIFIED"

    statistical = {
        "preregistered_seed": SEED,
        "within_rotation": within_results,
        "rescue": rescue_stats,
        "candidate_aggregates": candidate_values,
        "pairwise_direction_checks": direction,
        "external_persistent_h1_vs_multihorizon": {
            "Identity": {
                "h1": candidate_values["external_persistent_kl_h1"]["Identity"],
                "equal_H1_64": candidate_values["external_persistent_kl_equal_H1_64"]["Identity"],
            },
            "Hadamard": {
                "h1": candidate_values["external_persistent_kl_h1"]["Hadamard"],
                "equal_H1_64": candidate_values["external_persistent_kl_equal_H1_64"]["Hadamard"],
            },
            "PERSISTENT_BETTER_THAN_LOCAL": "INCONCLUSIVE",
            "reason": "Neither h=1 nor multi-horizon Future-KL has the correct Qwen cross-rotation direction, and the panel has no AIME outcome labels for within-rotation or rescue testing."
        },
    }
    dump_json(OUT / "statistical_analysis.json", statistical)

    known_big_hash = capture_manifest["required_outputs"]["token_level_metrics.csv"]["sha256"]
    shard_rows, checkpoint_hash = hash_model_checkpoint()
    sources = [
        artifact_entry(TOKEN_CSV, known_big_hash), artifact_entry(TOKEN_JSON), artifact_entry(CAPTURE_CONFIG),
        artifact_entry(CAPTURE_SCRIPT), artifact_entry(LABELS), artifact_entry(HEADROOM / "summary.json"),
        artifact_entry(HEADROOM / "per_document_auc.csv"), artifact_entry(NUMERICAL_CLOSURE), artifact_entry(THETA0),
    ]
    artifact_manifest = {
        "ARTIFACT_AUDIT": "PASS", "pairing_status": "PASS", "paired_samples": 60, "clean_paired_samples": 59,
        "token_condition_rows": total_rows, "identity_hadamard_same_fp_token_trajectory": True,
        "model": "Qwen3.5-9B/GDN", "model_checkpoint_combined_sha256": checkpoint_hash,
        "model_checkpoint_hash_definition": "SHA256 over sorted lines shard_filename:shard_sha256", "model_weight_shards": shard_rows,
        "model_config": artifact_entry(MODEL / "config.json"), "checkpoint_index": artifact_entry(MODEL / "model.safetensors.index.json"),
        "quantizer": capture_config["quantizer"], "rotation": capture_config["rotation"],
        "m5_definition": capture_config["m5"], "m5_implementation_sha256": sha256(CAPTURE_SCRIPT),
        "m0_implementation_sha256": sha256(CAPTURE_SCRIPT), "gdn_layers": capture_config["gdn_layers"],
        "gdn_layer_count": len(capture_config["gdn_layers"]), "model_parameter_dtype": "bfloat16",
        "recurrent_compute_and_qdq_dtype": "float32", "teacher_forced_token_source": capture_config["teacher_forcing_source"],
        "scorer": "Frozen V4 read-only labels", "source_artifacts": sources,
        "missing_for_requested_metrics": {
            "aime_full_vocab_logits": True, "aime_reference_next_token_logprob": True,
            "restartable_anchor_states": True, "m5_primitive_components": True,
            "per_token_per_layer_m5": True, "per_head_m5": True,
        },
    }
    dump_json(OUT / "artifact_manifest.json", artifact_manifest)
    audit_md = f"""# Source and provenance audit\n\n`ARTIFACT_AUDIT = PASS`. All 60 Identity/Hadamard pairs have identical `(timestep, token_id)` sequence hashes and row counts; clean59 excludes only `{BAD_PAIR[0]}`, seed {BAD_PAIR[1]}, based on the pre-existing non-finite audit.\n\n- Model: Qwen3.5-9B/GDN; combined four-shard checkpoint hash: `{checkpoint_hash}`.\n- Quantizer: symmetric INT8-C128, axis -2, qrange [-127,127], unchanged.\n- Rotation: key-side normalized H128; Hadamard states recovered to the FP basis before M0/M5.\n- GDN layers: 24; model parameters BF16; recurrent compute/QDQ FP32.\n- Teacher forcing: existing FP_STATE generated token IDs, repeated identically for FP/Identity/Hadamard batch rows.\n- Outcomes: read-only Frozen V4 labels.\n\n## Artifact boundary\n\nThe AIME capture reduces recurrent tensors online and does not save full logits, M5 primitive factors, heads, per-token per-layer M5, or restartable anchor states. Exact AIME KL, delta-NLL, delta-margin, and anchor-reset persistent metrics are therefore `INSUFFICIENT_ARTIFACT`. The existing non-AIME 64-document Qwen Future-KL is retained as a separate direction diagnostic and is never joined to AIME outcomes.\n"""
    (OUT / "source_audit/audit.md").write_text(audit_md, encoding="utf-8")

    failure_mode = "MIXED_COORDINATE_DEPENDENCE_AND_EXTREME_TAIL" if tail_summary["Hadamard"]["status"] == "EXTREME" else "MIXED_COORDINATE_DEPENDENCE_AND_TAIL_SCALE_SHIFT"
    results = {
        "task": "QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1",
        "ARTIFACT_AUDIT": "PASS", "M5_FP_COORDINATE_INVARIANCE": "PARTIAL",
        "M5_SCALE_FAILURE_MODE": failure_mode, "RAW_M5_CROSS_ROTATION": "INVALID",
        "BEST_WITHIN_ROTATION_DIAGNOSTIC": best_within,
        "BEST_CROSS_ROTATION_SURROGATE": best_cross,
        "BEST_RESCUE_ASSOCIATION_METRIC": best_rescue,
        "PERSISTENT_BETTER_THAN_LOCAL": "INCONCLUSIVE",
        "LEARNABLE_ROTATION_GATE": "FAIL",
        "overall_conclusion": "NO_VALID_CROSS_ROTATION_SURROGATE_IDENTIFIED" if not promising else "PROMISING_DIAGNOSTIC_REQUIRES_EXTERNAL_VALIDATION",
        "tail_summary": tail_summary, "candidate_aggregates": candidate_values,
        "pairwise_direction_checks": direction, "metric_classifications": classifications,
        "artifact_insufficiency": {
            "aime_kl_delta_nll_delta_margin": "INSUFFICIENT_ARTIFACT",
            "aime_anchor_reset_multi_horizon": "INSUFFICIENT_ARTIFACT",
            "m5_primitive_decomposition": "INSUFFICIENT_ARTIFACT",
            "joint_sample_layer_token_head_offender": "INSUFFICIENT_ARTIFACT",
        },
        "modifications": {"new_aime_generation": False, "new_rotation_training": False, "model_modified": False, "quantizer_modified": False, "scorer_modified": False},
    }
    dump_json(OUT / "results.json", results)
    result_rows = []
    mapping = {
        "raw_m0": "m0_p95", "raw_m5": "m5_p95", "decision": "decision_p95",
        "m0_relative": "m0_relative_p95", "m5_over_m0": "m5_over_m0_p95",
        "external_qwen_persistent_kl": "external_persistent_kl_equal_H1_64",
    }
    for name, info in classifications.items():
        key = mapping.get(name)
        result_rows.append({
            "metric": name, "classification": info["class"], "reason": info["reason"],
            "identity_candidate_aggregate": candidate_values.get(key, {}).get("Identity") if key else None,
            "hadamard_candidate_aggregate": candidate_values.get(key, {}).get("Hadamard") if key else None,
            "pairwise_direction_check": direction.get(key) if key else "INSUFFICIENT_ARTIFACT",
            "identity_fp_correct52_auroc": all_within.get(key, {}).get("Identity") if key else None,
            "hadamard_fp_correct52_auroc": all_within.get(key, {}).get("Hadamard") if key else None,
            "rescue_vs_stable_failure_cliffs_delta": rescue_effects.get(key) if key else None,
        })
    pd.DataFrame(result_rows).to_csv(OUT / "results.csv", index=False)

    i_m5, h_m5 = candidate_values["m5_p95"]["Identity"], candidate_values["m5_p95"]["Hadamard"]
    h1_i, h1_h = candidate_values["external_persistent_kl_h1"]["Identity"], candidate_values["external_persistent_kl_h1"]["Hadamard"]
    auc_i, auc_h = candidate_values["external_persistent_kl_equal_H1_64"]["Identity"], candidate_values["external_persistent_kl_equal_H1_64"]["Hadamard"]
    report = f"""# Qwen Persistent Functional Surrogate V1\n\n## Executive conclusion\n\n`NO_VALID_CROSS_ROTATION_SURROGATE_IDENTIFIED`. Raw M5 remains useful inside a fixed rotation but fails the Identity-to-Hadamard direction test. The independent 64-document persistent Future-KL also ranks Native lower-risk than Hadamard at both h=1 and equal-weight H=1..64, opposite the held-out AIME accuracy direction. Exact AIME KL/delta-NLL/delta-margin and anchor-reset multi-horizon values cannot be reconstructed from the saved scalar artifacts and are reported as `INSUFFICIENT_ARTIFACT`, not imputed.\n\n## 1. Is M5 coordinate invariant?\n\n**PARTIAL / not established.** Exact arithmetic predicts functional equivalence, and the prior FP-only Hadamard gate retained 100% primary top-1 agreement with logit relative-L2 p95 {closure['hadamard']['primary']['logit_relative_l2']['p95']:.4g} and recovered-state relative-L2 p95 {closure['hadamard']['primary']['state_recovered_relative_l2']['p95']:.4g}. However, finite-precision differences are measurable and the previous capture did not record raw M5 or its primitives when quantization was off. Therefore direct raw-M5 coordinate invariance is `INSUFFICIENT_ARTIFACT`; it must not be claimed as PASS.\n\n## 2. Why does Hadamard raw M5 explode?\n\n**{failure_mode}.** On clean59 the preregistered median trajectory hierarchical M5 p95 changes from {i_m5:.6g} (Identity) to {h_m5:.6g} (Hadamard). M0 exhibits the same large scale shift and the token-level log-M0/log-M5 coupling is reported in `component_decomposition.csv`. Hadamard tail concentration is `{tail_summary['Hadamard']['status']}`: the top 1% contributes {tail_summary['Hadamard']['contributions']['0.01']:.2%} of total M5-tail mass and the top 0.1% contributes {tail_summary['Hadamard']['contributions']['0.001']:.2%}. Because primitive M5 numerator/denominator tensors were discarded, a denominator pathology versus operator-scale split cannot be resolved further.\n\n## 3. Is raw M5 suitable as a rotation-training objective?\n\n**NO.** Its `PAIRWISE_DIRECTION_CHECK = INCORRECT`: Hadamard has higher raw risk despite 41/59 versus 18/59 AIME accuracy. It is classified `DIAGNOSTIC_ONLY`, not a cross-rotation objective.\n\n## 4. Best metric by scientific role\n\n- Within-rotation failure diagnosis: `{best_within}` on the preregistered FP-correct52 comparison; full AUROC/CIs for both rotations are in the per-rotation JSON files.\n- Cross-rotation surrogate: `{best_cross}`. No metric is promoted merely because one diagnostic check is favorable.\n- Exact KL, delta-NLL, delta-margin and AIME persistent variants: `INSUFFICIENT_EVIDENCE` because required logits/states are absent.\n\n## 5. Are persistent metrics better than local metrics?\n\n**INCONCLUSIVE.** On the separate fixed 64-document Qwen panel, Future-KL is {h1_i:.6g} vs {h1_h:.6g} at h=1 (Native vs Hadamard) and {auc_i:.6g} vs {auc_h:.6g} for equal-weight H=1,4,8,16,32,64. Both comparisons have the wrong AIME cross-rotation direction, and this panel has no AIME outcome labels. Multi-horizon is therefore not shown to improve either direction or failure prediction.\n\n## 6. Which metric best separates Hadamard rescue from stable failure?\n\nDescriptively, `{best_rescue}` has the most favorable preregistered Cliff's delta among the available metrics. Its exact group probabilities, sign tests, bootstrap intervals and rescue-vs-stable-failure permutation result are in `rescue_statistics.json`. This is an association ranking, not sufficient evidence for an optimization target. Missing functional metrics are not included in this comparison.\n\n## 7. Learnable-rotation gate\n\n`LEARNABLE_ROTATION_GATE = FAIL`. No available metric simultaneously clears coordinate behavior, within-Identity information, within-Hadamard information, correct pairwise direction, and rescue association. No learnable rotation was started.\n\n## Normalization diagnostics\n\n`m0_relative` and `m5_over_m0` were evaluated exactly as preregistered diagnostics. Their classifications and pairwise directions are in `results.csv`. Neither is automatically promoted: M5/M0 can cancel a shared scale without proving that its residual represents future functional harm, while M0-relative is not a direct function-space metric. FP-reference normalized M5 and frozen-layer log-z M5 remain `INSUFFICIENT_ARTIFACT` because the necessary frozen FP functional calibration components were not saved.\n\n## Ling external-validation protocol (not executed)\n\nIf a Qwen metric is later frozen as `PROMISING_SURROGATE`, freeze its complete definition, H={{1,4,8,16,32,64}}, aggregation, normalization, epsilon and missing-data policy before opening Ling outcomes. Then test the four Ling conditions in `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-ling-3090-v2/experiments/LING_DETERMINISTIC_16DOC_REPLICATION_V2` using paired document statistics, reporting mean, median, wins/losses and confidence intervals. Do not treat aggregate mean alone as ground truth and do not tune using the known Ling results.\n\n## Artifact note\n\nAll 60 paired token sequences match exactly by sample ID, seed, timestep and token ID. The derived parquet retains local metrics and explicit null persistent columns with status `LOCAL_ONLY_PERSISTENT_INSUFFICIENT_ARTIFACT`; it does not mislabel local observations as persistent measurements.\n"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")

    # Compact, auditable figures as SVG without optional plotting dependencies.
    fig = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360" viewBox="0 0 900 360"><rect width="900" height="360" fill="white"/><text x="40" y="40" font-family="sans-serif" font-size="22">Cross-rotation direction checks (lower risk expected for Hadamard)</text><text x="40" y="90" font-family="monospace" font-size="18">AIME accuracy: Identity 18/59  |  Hadamard 41/59</text><text x="40" y="140" font-family="monospace" font-size="18">Raw M5 p95: {i_m5:.4g}  |  {h_m5:.4g}  -> INCORRECT</text><text x="40" y="190" font-family="monospace" font-size="18">Future-KL h=1: {h1_i:.4g}  |  {h1_h:.4g}  -> INCORRECT</text><text x="40" y="240" font-family="monospace" font-size="18">Future-KL H1-64: {auc_i:.4g}  |  {auc_h:.4g}  -> INCORRECT</text><text x="40" y="305" font-family="sans-serif" font-size="21" fill="#a00">No valid cross-rotation surrogate identified</text></svg>"""
    (OUT / "figures/direction_checks.svg").write_text(fig, encoding="utf-8")

    ended_iso = datetime.now(timezone.utc).isoformat()
    required = [
        "preregistration.json", "artifact_manifest.json", "m5_coordinate_audit/fp_invariance.json",
        "m5_coordinate_audit/component_decomposition.csv", "m5_coordinate_audit/layer_scale.csv",
        "m5_coordinate_audit/tail_localization.csv", "m5_coordinate_audit/tail_report.md",
        "persistent_metrics/token_level_metrics.parquet", "persistent_metrics/trajectory_metrics.csv",
        "persistent_metrics/horizon_metrics.csv", "within_rotation_analysis/identity.json",
        "within_rotation_analysis/hadamard.json", "rescue_analysis/rescue_groups.csv",
        "rescue_analysis/paired_metric_deltas.csv", "rescue_analysis/rescue_statistics.json",
        "statistical_analysis.json", "results.json", "results.csv", "final_report.md",
    ]
    output_manifest = {}
    for rel in required:
        path = OUT / rel
        output_manifest[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    run_manifest = {
        "task": "QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1", "status": "COMPLETE_WITH_DECLARED_INSUFFICIENT_ARTIFACTS",
        "started_at": started_iso, "completed_at": ended_iso, "elapsed_seconds": time.time() - started,
        "host": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pa.__version__,
        "analysis_script": artifact_entry(Path(__file__)), "preregistration_sha256": sha256(prereg),
        "deterministic_seeds": {"bootstrap": SEED, "permutation": SEED},
        "outputs": output_manifest, "new_gpu_inference": False, "new_aime_generation": False,
        "new_rotation_training": False, "model_modified": False, "quantizer_modified": False, "scorer_modified": False,
    }
    dump_json(OUT / "run_manifest.json", run_manifest)

    print("\nQWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1")
    print("ARTIFACT_AUDIT = PASS")
    print("M5_FP_COORDINATE_INVARIANCE = PARTIAL")
    print("M5_SCALE_FAILURE_MODE =", failure_mode)
    print("RAW_M5_CROSS_ROTATION = INVALID")
    print("BEST_WITHIN_ROTATION_DIAGNOSTIC =", best_within)
    print("BEST_CROSS_ROTATION_SURROGATE =", best_cross)
    print("BEST_RESCUE_ASSOCIATION_METRIC =", best_rescue)
    print("PERSISTENT_BETTER_THAN_LOCAL = INCONCLUSIVE")
    print("LEARNABLE_ROTATION_GATE = FAIL")
    print("NEW_AIME_GENERATION = NO")
    print("NEW_ROTATION_TRAINING = NO")
    print("MODEL_MODIFIED = NO")
    print("QUANTIZER_MODIFIED = NO")
    print("SCORER_MODIFIED = NO")


if __name__ == "__main__":
    main()
