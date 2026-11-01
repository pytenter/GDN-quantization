#!/usr/bin/env python3
"""Offline Qwen recurrent-state quantization metric benchmark.

This script intentionally does not load the model or generate tokens.  It consumes
only existing AIME26 generations and existing independent mechanism artifacts.
Missing AIME-aligned tensor metrics are represented as null, never imputed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK = "QWEN_ROTATION_METRIC_BENCHMARK_V1"
CONDITIONS = ("fp_state", "int8_c128", "int8_c128_key_h")
DISPLAY = {
    "fp_state": "FP_STATE",
    "int8_c128": "INT8_C128",
    "int8_c128_key_h": "INT8_C128_KEY_HADAMARD",
}
MECHANISM_FIELDS = [
    "state_error_mean",
    "state_error_p95",
    "state_error_max",
    "future_kl_mean",
    "future_kl_p95",
    "future_kl_max",
    "m5_mean",
    "m5_median",
    "m5_p95",
    "m5_max",
    "persistence_mean",
    "persistence_p95",
    "persistence_max",
    "decision_sensitivity_mean",
    "decision_sensitivity_p95",
    "decision_sensitivity_max",
    "decision_sensitivity_count_gt1",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, obj) -> None:
    path.write_text(json.dumps(sanitize(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize(x):
    if isinstance(x, dict):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def rankdata(values):
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def auc_score(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def quantile(values, q):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals), q))


def summary(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    a = np.asarray(vals)
    return {
        "n": len(vals),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_of(fp: bool, native: bool, had: bool) -> str:
    if fp and native:
        return "A_BOTH_CORRECT"
    if fp and (not native) and had:
        return "B_HADAMARD_RESCUED"
    if fp and (not native) and (not had):
        return "C_HADAMARD_FAILED"
    return "D_FP_WRONG"


def metric_eval(units, label_key, subset, metric_key, sign=1.0):
    rows = [u for u in units if subset(u) and u.get(metric_key) is not None]
    y = [int(u[label_key]) for u in rows]
    x = [float(u[metric_key]) * sign for u in rows]
    return {
        "n": len(rows),
        "positive": sum(y),
        "negative": len(y) - sum(y),
        "score_direction": "higher means more positive risk",
        "spearman": spearman(x, y),
        "pearson": pearson(x, y),
        "auc": auc_score(y, x),
    }


def main():
    script = Path(__file__).resolve()
    repo = script.parents[2]
    out = script.parent
    if out.name != TASK:
        raise RuntimeError(f"script must live in experiments/{TASK}, got {out}")

    alignment = repo / "results/rotation/qwen_metric_alignment_audit_v1/canonical_trajectory_metrics"
    sample_path = alignment / "sample_metrics.jsonl"
    manifest_path = alignment / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = load_jsonl(sample_path)
    if len(samples) != 180:
        raise RuntimeError(f"expected 180 sample records, found {len(samples)}")

    sample_map = {(r["problem_id"], int(r["seed"]), r["condition"]): r for r in samples}
    raw_map = {}
    source_inventory = []
    for src in manifest["sources"]:
        p = Path(src["path"])
        cond = src["condition"]
        rows = load_jsonl(p)
        source_inventory.append(
            {
                "condition": cond,
                "path": str(p),
                "rows": len(rows),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
        )
        for r in rows:
            raw_map[(r["problem_id"], int(r["seed"]), cond)] = r
    if len(raw_map) != 180:
        raise RuntimeError(f"expected 180 raw records, found {len(raw_map)}")

    keys = sorted({(k[0], k[1]) for k in sample_map})
    if len(keys) != 60:
        raise RuntimeError(f"expected 60 paired units, found {len(keys)}")

    units = []
    trajectory_rows = []
    for problem_id, seed in keys:
        s = {c: sample_map[(problem_id, seed, c)] for c in CONDITIONS}
        r = {c: raw_map[(problem_id, seed, c)] for c in CONDITIONS}
        fp_ok = bool(s["fp_state"]["v4_correct"])
        native_ok = bool(s["int8_c128"]["v4_correct"])
        had_ok = bool(s["int8_c128_key_h"]["v4_correct"])
        group = group_of(fp_ok, native_ok, had_ok)
        fp_ids = r["fp_state"]["generated_token_ids"]
        native_ids = r["int8_c128"]["generated_token_ids"]
        had_ids = r["int8_c128_key_h"]["generated_token_ids"]
        native_cp = common_prefix(fp_ids, native_ids)
        had_cp = common_prefix(fp_ids, had_ids)
        unit = {
            "problem_id": problem_id,
            "seed": seed,
            "group": group,
            "fp_correct": fp_ok,
            "int8_correct": native_ok,
            "hadamard_correct": had_ok,
            "int8_failure_given_fp_correct": bool(fp_ok and not native_ok),
            "hadamard_rescue_given_int8_failure": bool(fp_ok and not native_ok and had_ok),
            "native_common_prefix_tokens": native_cp,
            "hadamard_common_prefix_tokens": had_cp,
            "native_common_prefix_fraction_of_fp": native_cp / max(1, len(fp_ids)),
            "hadamard_common_prefix_fraction_of_fp": had_cp / max(1, len(fp_ids)),
            "fp_generated_tokens": int(s["fp_state"]["num_generated_tokens"]),
            "fp_repeated_16gram_fraction": float(s["fp_state"]["repeated_16gram_fraction"]),
            "fp_self_correction_count": int(s["fp_state"]["self_correction_count"]),
            "fp_candidate_changes": int(s["fp_state"]["num_candidate_changes"]),
            "fp_unique_4gram_ratio": float(s["fp_state"]["unique_4gram_ratio"]),
            "fp_hit_limit": bool(s["fp_state"]["hit_limit"]),
        }
        units.append(unit)
        for cond in CONDITIONS:
            sm = s[cond]
            rr = r[cond]
            ids = rr["generated_token_ids"]
            cp = len(ids) if cond == "fp_state" else common_prefix(fp_ids, ids)
            qa = rr.get("quantizer_audit") or {}
            ha = rr.get("hadamard_audit") or {}
            row = {
                "problem_id": problem_id,
                "seed": seed,
                "condition": DISPLAY[cond],
                "group": group,
                "v4_correct": bool(sm["v4_correct"]),
                "generated_tokens": int(sm["num_generated_tokens"]),
                "eos_seen": bool(sm["eos_seen"]),
                "hit_limit": bool(sm["hit_limit"]),
                "finish_type": sm["finish_type"],
                "repeated_16gram_fraction": float(sm["repeated_16gram_fraction"]),
                "self_correction_count": int(sm["self_correction_count"]),
                "candidate_changes": int(sm["num_candidate_changes"]),
                "unique_4gram_ratio": float(sm["unique_4gram_ratio"]),
                "fp_common_prefix_tokens": cp,
                "fp_common_prefix_fraction": cp / max(1, len(fp_ids)),
                "quantizer_calls": int(qa.get("calls", 0)),
                "quantizer_layers": len(qa.get("layers", [])),
                "qrange_min": (qa.get("qrange_observed") or [None, None])[0],
                "qrange_max": (qa.get("qrange_observed") or [None, None])[1],
                "hadamard_calls": int(ha.get("calls", 0)),
                "mechanism_metrics_status": "NOT_CAPTURED_FOR_AIME26",
            }
            for field in MECHANISM_FIELDS:
                row[field] = None
            trajectory_rows.append(row)

    counts = Counter(u["group"] for u in units)
    accuracy = {
        "FP_STATE": sum(u["fp_correct"] for u in units),
        "INT8_C128": sum(u["int8_correct"] for u in units),
        "INT8_C128_KEY_HADAMARD": sum(u["hadamard_correct"] for u in units),
    }
    assert accuracy == {"FP_STATE": 52, "INT8_C128": 18, "INT8_C128_KEY_HADAMARD": 41}
    assert counts == Counter(
        {"A_BOTH_CORRECT": 18, "B_HADAMARD_RESCUED": 23, "C_HADAMARD_FAILED": 11, "D_FP_WRONG": 8}
    )

    # Independent mechanism evidence; these units are not joined to AIME labels.
    dense_summary_path = repo / "results/rotation/qwen_dense_orthogonal_oracle_v1/summary.json"
    dense_local_path = repo / "results/rotation/qwen_dense_orthogonal_oracle_v1/evaluation/heldout_local_metrics.json"
    dense_persistent_path = repo / "results/rotation/qwen_persistent_headroom_confirmation_v1/final/summary.json"
    m5_summary_path = Path("/data/zypan/GDN-quantization/results/propagation/gdn_int8_linearized_functional_risk_proxy_extraction_v1_final_summary.json")
    dense_summary = json.loads(dense_summary_path.read_text())
    dense_local = json.loads(dense_local_path.read_text())
    dense_persistent = json.loads(dense_persistent_path.read_text())
    m5_summary = json.loads(m5_summary_path.read_text())

    independent = {
        "joinable_to_aime_trajectory_labels": False,
        "m5_natural_panel": {
            "units": m5_summary["valid_units"],
            "m0_state_error_vs_future_kl_spearman": m5_summary["m0_future_kl_spearman"],
            "m5_vs_future_kl_spearman": m5_summary["m5_future_kl_spearman"],
            "m4_vs_future_kl_spearman": m5_summary["m4_future_kl_spearman"],
            "classification": m5_summary["final_scientific_classification"],
        },
        "dense_oracle": {
            "aime26_used": bool(dense_summary["safety"]["AIME26_used"]),
            "heldout_local_metrics": dense_local,
            "persistent_64doc_auc_mean": {
                k: v["mean"] for k, v in dense_persistent["auc_distribution"].items()
            },
            "persistent_best_learned": dense_persistent["best_learned_by_mean_auc"],
            "persistent_verdict": dense_persistent["best_verdict"],
        },
    }

    # Group summaries: downstream descriptors are explicitly kept separate from candidate metrics.
    group_comparison = {
        "task": TASK,
        "paired_units": len(units),
        "group_counts": dict(sorted(counts.items())),
        "accuracy_out_of_60": accuracy,
        "mechanism_metric_availability": {m: 0 for m in MECHANISM_FIELDS},
        "downstream_descriptive_only": {},
        "independent_nonjoinable_evidence": independent,
    }
    for group in sorted(counts):
        gu = [u for u in units if u["group"] == group]
        group_comparison["downstream_descriptive_only"][group] = {
            "n": len(gu),
            "fp_generated_tokens": summary([u["fp_generated_tokens"] for u in gu]),
            "fp_repeated_16gram_fraction": summary([u["fp_repeated_16gram_fraction"] for u in gu]),
            "fp_self_correction_count": summary([u["fp_self_correction_count"] for u in gu]),
            "native_divergence_step": summary([u["native_common_prefix_tokens"] for u in gu]),
            "hadamard_divergence_step": summary([u["hadamard_common_prefix_tokens"] for u in gu]),
        }

    unavailable_reason = (
        "AIME26 generations contain full text and token IDs but no per-step logits, recurrent states, "
        "state quantization tensors, or M5 captures; no metric values were imputed or copied from unmatched prompts."
    )
    corr = {
        "task": TASK,
        "primary_outcomes": {
            "int8_failure": {"population": "FP-correct paired units", "n": 52, "positive": 34},
            "hadamard_rescue": {"population": "FP-correct and INT8-failed paired units", "n": 34, "positive": 23},
        },
        "requested_mechanism_metrics": {},
        "exploratory_fp_behavioral_proxies": {},
        "independent_nonjoinable_evidence": independent,
    }
    for metric in MECHANISM_FIELDS:
        corr["requested_mechanism_metrics"][metric] = {
            "availability": "NOT_IDENTIFIABLE_FROM_EXISTING_AIME_ARTIFACTS",
            "n": 0,
            "spearman": None,
            "pearson": None,
            "auc_int8_failure": None,
            "auc_hadamard_rescue": None,
            "reason": unavailable_reason,
        }

    behavioral = {
        "fp_generated_tokens": 1.0,
        "fp_repeated_16gram_fraction": 1.0,
        "fp_self_correction_count": 1.0,
        "fp_candidate_changes": 1.0,
        "fp_unique_4gram_ratio": -1.0,
        "fp_hit_limit": 1.0,
    }
    for metric, sign in behavioral.items():
        corr["exploratory_fp_behavioral_proxies"][metric] = {
            "eligible_rotation_objective": False,
            "why_not_eligible": "requires an already-generated FP reasoning trajectory and is not a recurrent-state mechanism metric",
            "int8_failure": metric_eval(
                units,
                "int8_failure_given_fp_correct",
                lambda u: u["fp_correct"],
                metric,
                sign,
            ),
            "hadamard_rescue": metric_eval(
                units,
                "hadamard_rescue_given_int8_failure",
                lambda u: u["fp_correct"] and not u["int8_correct"],
                metric,
                sign,
            ),
        }

    # Write machine-readable trajectory tables.
    trajectory_json = {
        "task": TASK,
        "schema": "long form: one row per (problem_id, seed, condition)",
        "paired_units": len(units),
        "rows": trajectory_rows,
    }
    json_dump(out / "trajectory_metrics.json", trajectory_json)
    csv_fields = list(trajectory_rows[0].keys())
    with (out / "trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(trajectory_rows)
    json_dump(out / "group_comparison.json", group_comparison)
    json_dump(out / "correlation_analysis.json", corr)

    # Plots use only honest, explicitly labelled evidence layers.
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    colors = ["#4C78A8", "#59A14F", "#E15759", "#B07AA1"]
    group_order = ["A_BOTH_CORRECT", "B_HADAMARD_RESCUED", "C_HADAMARD_FAILED", "D_FP_WRONG"]
    short = ["A both\ncorrect", "B rescued", "C not\nrescued", "D FP wrong"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(short, [counts[g] for g in group_order], color=colors)
    ax.set_ylabel("Paired trajectories")
    ax.set_title("AIME26 paired outcome groups (n=60)")
    for i, g in enumerate(group_order):
        ax.text(i, counts[g] + 0.4, str(counts[g]), ha="center")
    fig.tight_layout()
    fig.savefig(plots / "group_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(accuracy)
    vals = [accuracy[n] for n in names]
    ax.bar(["FP_STATE", "INT8_C128", "INT8+C128\nKey-H"], vals, color=["#4C78A8", "#E15759", "#59A14F"])
    ax.set_ylim(0, 60)
    ax.set_ylabel("Correct / 60")
    ax.set_title("Observed AIME26 accuracy")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(plots / "aime_accuracy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, key, title in [
        (axes[0], "native_common_prefix_tokens", "FP vs INT8_C128"),
        (axes[1], "hadamard_common_prefix_tokens", "FP vs INT8_C128 + Key-H"),
    ]:
        data = [[u[key] for u in units if u["group"] == g] for g in group_order]
        ax.boxplot(data, tick_labels=["A", "B", "C", "D"], showfliers=False)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(title)
        ax.set_xlabel("Outcome group")
    axes[0].set_ylabel("Common prefix tokens (symlog)")
    fig.suptitle("Post-hoc trajectory divergence (not a pre-generation metric)")
    fig.tight_layout()
    fig.savefig(plots / "trajectory_divergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    metrics = ["M0 state error", "M5 functional"]
    vals = [
        independent["m5_natural_panel"]["m0_state_error_vs_future_kl_spearman"],
        independent["m5_natural_panel"]["m5_vs_future_kl_spearman"],
    ]
    ax.bar(metrics, vals, color=["#9C755F", "#F28E2B"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Spearman vs Future-KL")
    ax.set_title("Independent 9-unit mechanism panel (not AIME-joinable)")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.04 if v >= 0 else -0.08), f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(plots / "independent_m5_evidence.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    conds = ["Native_INT8", "Hadamard", "Dense_State", "Dense_Functional"]
    fkl = independent["dense_oracle"]["persistent_64doc_auc_mean"]
    x = np.arange(len(conds))
    ax1.bar(x, [fkl[c] for c in conds], color="#76B7B2", alpha=0.85)
    ax1.set_xticks(x, conds, rotation=15)
    ax1.set_ylabel("Mean persistent Future-KL AUC (lower is better)")
    ax1.set_title("Independent 64-document panel; Dense conditions lack AIME labels")
    ax1.text(0, fkl["Native_INT8"] + 0.00012, "AIME: 18/60", ha="center", fontsize=9)
    ax1.text(1, fkl["Hadamard"] + 0.00012, "AIME: 41/60", ha="center", fontsize=9)
    ax1.text(2, fkl["Dense_State"] + 0.00012, "AIME: not run", ha="center", fontsize=9)
    ax1.text(3, fkl["Dense_Functional"] + 0.00012, "AIME: not run", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "future_kl_downstream_mismatch.png", dpi=180)
    plt.close(fig)

    # Human-readable audit and definitions.
    audit = f"""# Artifact audit — {TASK}

## Scope and server

- Repository used: `{repo}`
- Analysis is offline only. No model was loaded, no tensors regenerated, and no AIME generation was run.
- Canonical AIME input: `{manifest['input_root']}`
- Model: Qwen3.5-9B / GDN; conditions: FP_STATE, INT8_C128, INT8_C128 + Key-Hadamard.

## Search result

The requested name searches located the canonical AIME26 V2 shards, Key-Hadamard artifacts, Dense Oracle checkpoints/evaluations, M5 propagation studies, and state/logit-oriented diagnostic artifacts. The files actually used are listed below; unmatched prompt panels are kept as independent evidence and are never joined to AIME labels.

### AIME26 paired generations

- 12 JSONL shards, 15 rows each, 180 condition-trajectories total.
- 60 paired `(problem_id, seed)` units, each with all three conditions.
- Full generated text and token IDs: available.
- V4 correctness labels: available.
- Per-step logits: **not captured**.
- Per-step recurrent states / quantization errors: **not captured**.
- Per-layer/per-timestep M5: **not captured**.
- Quantizer invocation metadata: available, but value/percentile/clipping/kurtosis statistics are not present.

### Source integrity

"""
    for src in source_inventory:
        audit += f"- `{src['path']}` — rows={src['rows']}, bytes={src['bytes']}, sha256=`{src['sha256']}`\n"
    audit += f"""

### Dense Oracle

- Checkpoints exist for Dense-State and Dense-Functional.
- Existing held-out local metrics and 64-document persistent Future-KL summaries are included as independent evidence.
- Dense Oracle summary explicitly records `AIME26_used=false`; therefore Dense-State and Dense-Functional cannot be assigned AIME groups or downstream AUCs.
- No retraining was performed.

### M5

- Existing natural-panel summary: `{m5_summary_path}`.
- It contains 9 valid units and validates frozen M0–M5 against Future-KL, not against AIME reasoning outcomes.
- Because prompt/unit identities do not match AIME26 trajectories, these values are not copied into `trajectory_metrics.*`.

## Audit conclusion

The current artifacts are sufficient for exact AIME grouping and token-level trajectory-divergence diagnostics, and for an independent mechanism-evidence summary. They are **not sufficient** for the requested trajectory-level comparison of state error, Future-KL, M5, persistence, and decision-boundary sensitivity. Missing values remain null by design.
"""
    (out / "artifact_audit.md").write_text(audit, encoding="utf-8")

    definitions = r"""# Metric definitions

All metric directions and aggregations below are frozen before any future capture is joined to AIME labels.

## Representation fidelity

For clean recurrent state `S_t` and quantized-condition state `S_hat_t`:

`state_error_t = ||S_t - S_hat_t||_2 / (||S_t||_2 + eps)`.

Aggregate over layer × timestep with mean, median, p95, and max. Also retain max absolute value, p50/p95/p99/p99.9, clipping ratio, channel max/RMS imbalance, and excess kurtosis. Existing AIME artifacts do not contain the required tensors or summaries.

## Functional sensitivity (M5)

Frozen existing definition:

`M5_t = || W_O D_g D_w J_RMS(o) E_t^T q_t ||_2`.

Retain per-layer and per-timestep values; aggregate mean, median, p95, and max. No normalization may be refit using AIME labels. Existing independent M5 panels are reported separately; AIME-aligned M5 is absent.

## Persistence / recurrent amplification

For fixed `alpha` and horizon set `K` declared in capture metadata:

`P_t = sum_{k in K} alpha^k ||E_{t+k}||_2 / (||E_t||_2 + eps)`.

Primary aggregation is p95; mean and max are secondary. The clean and perturbed paths must share the same teacher-forced token suffix so persistence is not confounded by sampled-token divergence. Existing AIME artifacts do not contain state-error trajectories.

## Decision-boundary sensitivity

Let `m_t = z_(1),t - z_(2),t` be the FP top-1/top-2 margin. Let `delta_z_t` be the change in the FP-winning token's logit under quantization (and additionally store the worst top-2 contrast shift). The frozen primary statistic is:

`D_t = |delta_z_t| / (m_t + eps)`.

Retain mean, p95, max, and `count(D_t > 1)`. Logits must be taken on the same teacher-forced prefix. Existing AIME artifacts do not contain logits.

## Outcomes and populations

- INT8 failure: `FP correct AND INT8_C128 wrong`; primary population is the 52 FP-correct trajectories.
- Hadamard rescue: `FP correct AND INT8_C128 wrong AND INT8_C128+Key-H correct`; primary population is the 34 FP-correct INT8 failures.
- Group A/B/C/D assignments follow these exact Boolean rules and use labels only as outcomes, never inside a metric.

## Statistics

For each captured metric, report raw-direction Spearman, Pearson when nondegenerate, and ROC AUC with higher metric value predeclared as higher failure risk. No orientation flipping, threshold selection, normalization fitting, or metric redesign is allowed after labels are inspected. Tail statistics are predeclared because rare recurrent excursions are the scientific hypothesis.

## Non-mechanism diagnostics

Common-prefix length, generated-token count, repetition, and self-correction are post-hoc trajectory descriptors. FP-only behavioral descriptors are evaluated exploratorily but are ineligible as rotation optimization objectives because they require an expensive generated reasoning trajectory and do not directly measure recurrent quantization quality.
"""
    (out / "metric_definition.md").write_text(definitions, encoding="utf-8")

    proxy_rows = corr["exploratory_fp_behavioral_proxies"]
    best_failure = max(
        ((k, v["int8_failure"]["auc"]) for k, v in proxy_rows.items() if v["int8_failure"]["auc"] is not None),
        key=lambda x: x[1],
    )
    best_rescue = max(
        ((k, v["hadamard_rescue"]["auc"]) for k, v in proxy_rows.items() if v["hadamard_rescue"]["auc"] is not None),
        key=lambda x: x[1],
    )
    fkl = independent["dense_oracle"]["persistent_64doc_auc_mean"]
    report = f"""# Final report — {TASK}

## Executive result

The benchmark framework is complete and the existing 60 paired AIME trajectories are audited and grouped exactly: A={counts['A_BOTH_CORRECT']}, B={counts['B_HADAMARD_RESCUED']}, C={counts['C_HADAMARD_FAILED']}, D={counts['D_FP_WRONG']}. Accuracy reproduces FP_STATE={accuracy['FP_STATE']}/60, INT8_C128={accuracy['INT8_C128']}/60, and INT8_C128+Key-Hadamard={accuracy['INT8_C128_KEY_HADAMARD']}/60.

The decisive scientific result is an identifiability limit: the AIME generations did not save per-step logits or recurrent states, so state error, Future-KL, M5, persistence, and decision sensitivity cannot be computed per AIME trajectory from existing artifacts. The requested correlations and AUCs for those metrics are therefore null rather than fabricated. Existing M5 and Dense Oracle evidence uses different prompts and is reported only as independent supporting evidence.

## Answers to the six questions

### 1. Which existing metric best predicts INT8 reasoning failure?

No existing **AIME-aligned mechanism metric** can be ranked. All requested mechanism metrics have n=0 on the 60 AIME units. Among exploratory FP-only behavioral descriptors, the largest predeclared-direction failure AUC is `{best_failure[0]}` at {best_failure[1]:.3f}; this is not an acceptable rotation objective and does not answer the mechanism question.

### 2. Does M5 outperform state reconstruction error?

On the independent 9-unit natural mechanism panel, yes for predicting Future-KL: M5 Spearman={m5_summary['m5_future_kl_spearman']:.3f} versus raw state error M0 Spearman={m5_summary['m0_future_kl_spearman']:.3f}. This is strong mechanism evidence, but it is **not evidence of superior AIME failure prediction**, because those units cannot be joined to AIME labels.

### 3. Does persistence provide additional information beyond M5?

Unknown. No existing artifact has persistence and M5 on the same AIME trajectories, so incremental value beyond M5 cannot be estimated without leakage or cross-dataset substitution.

### 4. Does decision-boundary sensitivity explain Hadamard rescue?

Unknown. The 34 FP-correct INT8 failures contain 23 rescues and 11 non-rescues, which is an adequate labelled comparison, but the logits needed for `D_t` were not saved. The strongest exploratory FP-only behavioral rescue AUC is `{best_rescue[0]}` at {best_rescue[1]:.3f}; it is not decision sensitivity.

### 5. Which metric should be a future rotation optimization objective?

None is ready to be selected from the current downstream evidence. The most defensible **candidate** is a frozen tail-risk objective led by M5 p95/max, with persistence p95 and decision-sensitivity exceedance count as preregistered secondary constraints. M5 has the strongest independent functional evidence, but it must first be validated against held-out downstream labels. Do not train a new rotation from the present result.

### 6. What additional experiment is needed before training a new rotation?

Run one narrow, paired **teacher-forced replay capture** over the already-generated FP token paths—no sampling and no AIME regeneration. For FP_STATE, INT8_C128, and INT8_C128+Key-Hadamard, save sparse per-layer/per-timestep state error and quantization statistics, frozen M5, fixed-horizon persistence, FP top-1/top-2 margins, and quantization-induced logit shifts. Use the definitions in `metric_definition.md` unchanged, publish missingness, and evaluate on a predeclared held-out split or a fresh reasoning set before choosing an objective. Dense-State and Dense-Functional may be included in this replay from existing checkpoints, without retraining.

## Traditional versus recurrent-aware evidence

The condition-level mismatch is already clear but not sufficient for trajectory prediction. In the independent 64-document panel, mean persistent Future-KL AUC is Native_INT8={fkl['Native_INT8']:.6f}, Hadamard={fkl['Hadamard']:.6f}, Dense_State={fkl['Dense_State']:.6f}, and Dense_Functional={fkl['Dense_Functional']:.6f} (lower is better), while AIME accuracy is 18/60 for Native and 41/60 for Hadamard. Thus generic Future-KL ranks Native better despite much worse reasoning. Held-out local state error is also nearly identical for Hadamard and Dense-State, while Dense conditions lack AIME outcomes. These facts reject simple condition-level substitution and motivate trajectory-aligned recurrent metrics.

## Dense Oracle status

Dense-State and Dense-Functional artifacts exist and were audited. They are included only in the independent evidence block because their summary records `AIME26_used=false`. No Dense retraining or rotation modification was performed.

## Validity guardrails

- No new rotation, model change, quantizer change, scorer change, AIME generation, or expensive tensor regeneration was performed.
- Gold correctness is used only to define outcomes/groups, never to construct a metric.
- Cross-dataset mechanism values are not broadcast onto AIME trajectories.
- Null requested statistics mean “not identifiable from saved artifacts,” not zero effect.

The practical answer to the final principle is therefore: a candidate rotation can only be screened cheaply after a lightweight teacher-forced instrumentation pass captures M5 tail risk, persistence, and decision-margin risk on the same trajectories. The current saved AIME outputs alone cannot support that decision.
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")

    run_manifest = {
        "task": TASK,
        "status": "COMPLETE_WITH_IDENTIFIABILITY_LIMIT",
        "repo": str(repo),
        "script": str(script),
        "inputs": {
            "aime_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "aime_sample_metrics": {"path": str(sample_path), "sha256": sha256(sample_path)},
            "dense_summary": {"path": str(dense_summary_path), "sha256": sha256(dense_summary_path)},
            "dense_local_metrics": {"path": str(dense_local_path), "sha256": sha256(dense_local_path)},
            "dense_persistent": {"path": str(dense_persistent_path), "sha256": sha256(dense_persistent_path)},
            "m5_summary": {"path": str(m5_summary_path), "sha256": sha256(m5_summary_path)},
        },
        "validation": {
            "paired_units": len(units),
            "condition_rows": len(trajectory_rows),
            "accuracy": accuracy,
            "group_counts": dict(counts),
            "requested_mechanism_non_null_values": 0,
        },
    }
    json_dump(out / "analysis_manifest.json", run_manifest)
    print(json.dumps(sanitize(run_manifest["validation"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
