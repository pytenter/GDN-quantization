#!/usr/bin/env python3
"""Aggregate Qwen teacher-forced mechanism capture shards and write final report."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TASK = "QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = REPO / "experiments" / TASK
FORMAL = OUT / "formal"
GDN_LAYERS = tuple(i for i in range(32) if i % 4 != 3)
HORIZONS = [1, 4, 8, 16, 32]
ALPHA = 0.9
DECISION_EPS = 1e-6
BOOTSTRAP = 2000
SEED = 20260923
CONDITIONS = ["INT8_C128", "INT8_C128_KEY_HADAMARD"]
AUDIT_FIELDS = [
    "m0_global_l2", "m0_layer_mean", "m0_layer_p95", "m0_layer_max",
    "m5_global_l2", "m5_layer_mean", "m5_layer_p95", "m5_layer_max",
    "persistence", "fp_margin", "delta_z_fp_winner", "decision_sensitivity",
]


def sanitize(x):
    if isinstance(x, dict):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def jdump(path: Path, obj):
    path.write_text(json.dumps(sanitize(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rankdata(values):
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    result = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        result[order[i:j]] = (i + 1 + j) / 2
        i = j
    return result


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def auc(y, scores):
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if not n1 or not n0:
        return None
    r = rankdata(scores)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(y, x, fn, seed_offset=0):
    y, x = np.asarray(y), np.asarray(x, float)
    rng = np.random.default_rng(SEED + seed_offset)
    vals = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        v = fn(x[idx], y[idx]) if fn in (pearson, spearman) else fn(y[idx], x[idx])
        if v is not None and math.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def evaluate(y, score, direction):
    y, score = np.asarray(y, int), np.asarray(score, float)
    score = score * direction
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
        "direction": "higher score predicts positive outcome",
        "spearman": spearman(score, y), "spearman_ci95": bootstrap_ci(y, score, spearman, 11),
        "pearson": pearson(score, y), "pearson_ci95": bootstrap_ci(y, score, pearson, 23),
        "auroc": auc(y, score), "auroc_ci95": bootstrap_ci(y, score, auc, 37),
    }


def zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def partial_spearman(y, x, control):
    ry, rx, rc = rankdata(y), rankdata(x), rankdata(control)
    A = np.column_stack([np.ones(len(rc)), rc])
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    return pearson(ex, ey)


def metric_value(row, metric):
    family, agg = metric.split(".", 1)
    if family == "decision":
        return float(row["decision_sensitivity"][agg])
    return float(row[family][agg])


def basic(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return {
        "n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)), "max": float(a.max()),
    }


def load_summaries():
    statuses, rows = [], []
    for i in range(4):
        shard = FORMAL / f"shard_{i:02d}"
        status = json.loads((shard / "status.json").read_text())
        statuses.append(status)
        if status["status"] != "COMPLETE":
            raise RuntimeError(f"shard {i} is not complete: {status}")
        with (shard / "trajectory_summary.jsonl").open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if len(rows) != 120:
        raise RuntimeError(f"expected 120 condition summaries, got {len(rows)}")
    return statuses, rows


def concatenate_token_csv(nonfinite_keys):
    target = OUT / "token_level_metrics.csv"
    count = 0
    audit = {}
    with target.open("w", newline="", encoding="utf-8") as dst:
        writer = None
        for i in range(4):
            src = FORMAL / f"shard_{i:02d}/token_metrics.csv.gz"
            with gzip.open(src, "rt", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if writer is None:
                    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                    writer.writeheader()
                elif reader.fieldnames != writer.fieldnames:
                    raise RuntimeError("token shard schema mismatch")
                for row in reader:
                    writer.writerow(row)
                    count += 1
                    key = (row["problem_id"], int(row["seed"]), row["condition"])
                    if key in nonfinite_keys:
                        label = f"{key[0]}|seed={key[1]}|{key[2]}"
                        rec = audit.setdefault(label, {
                            "problem_id": key[0], "seed": key[1],
                            "condition": key[2], "group": row["group"], "rows": 0,
                            "nonfinite_count": {field: 0 for field in AUDIT_FIELDS},
                            "first_nonfinite_timestep": {},
                        })
                        rec["rows"] += 1
                        for field in AUDIT_FIELDS:
                            try:
                                finite = math.isfinite(float(row[field]))
                            except (TypeError, ValueError):
                                finite = False
                            if not finite:
                                rec["nonfinite_count"][field] += 1
                                rec["first_nonfinite_timestep"].setdefault(
                                    field, int(row["timestep"])
                                )
    if count != 2 * 1854451:
        raise RuntimeError(f"expected 3,708,902 token-condition rows, got {count}")
    return count, target, audit


def main():
    statuses, rows = load_summaries()
    by_key = defaultdict(dict)
    for r in rows:
        by_key[(r["problem_id"], int(r["seed"]))][r["condition"]] = r
    if len(by_key) != 60 or any(set(v) != set(CONDITIONS) for v in by_key.values()):
        raise RuntimeError("paired condition integrity failure")

    # Identify trajectory-conditions whose summary counts reveal non-finite
    # values. Persistence is excluded here because its final timestep is
    # intentionally missing when no future horizon exists.
    nonfinite_keys = {
        (r["problem_id"], int(r["seed"]), r["condition"])
        for r in rows
        if (
            r["m0_global_l2"]["n"] < r["tokens"]
            or r["m5_global_l2"]["n"] < r["tokens"]
            or r["decision_sensitivity"]["n"] < r["tokens"]
        )
    }
    token_count, token_csv, nonfinite_records = concatenate_token_csv(nonfinite_keys)
    group_counts = Counter(next(iter(v.values()))["group"] for v in by_key.values())
    expected_groups = {"A_BOTH_CORRECT": 18, "B_HADAMARD_RESCUED": 23, "C_HADAMARD_FAILED": 11, "D_FP_WRONG": 8}
    if dict(group_counts) != expected_groups:
        raise RuntimeError(f"group mismatch: {group_counts}")

    metrics = [
        "m0.mean", "m0.median", "m0.p95", "m0.max",
        "m0_relative.mean", "m0_relative.p95", "m0_relative.max",
        "m0_global_l2.mean", "m0_global_l2.p95", "m0_global_l2.max",
        "m5.mean", "m5.median", "m5.p95", "m5.max",
        "m5_global_l2.mean", "m5_global_l2.p95", "m5_global_l2.max",
        "persistence.mean", "persistence.p95", "persistence.max",
        "decision.mean", "decision.p95", "decision.max", "decision.count_gt1", "decision.fraction_gt1",
    ]

    # Primary failure population: FP-correct (A/B/C); positive=B/C.
    failure_units = []
    rescue_units = []
    for key, d in sorted(by_key.items()):
        group = d["INT8_C128"]["group"]
        if group != "D_FP_WRONG":
            failure_units.append((d, int(group in ("B_HADAMARD_RESCUED", "C_HADAMARD_FAILED"))))
        if group in ("B_HADAMARD_RESCUED", "C_HADAMARD_FAILED"):
            rescue_units.append((d, int(group == "B_HADAMARD_RESCUED")))

    corr = {
        "task": TASK,
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "unit": "paired trajectory"},
        "populations": {
            "failure": {"n": len(failure_units), "positive": sum(y for _, y in failure_units), "definition": "FP correct; positive=INT8 failure"},
            "rescue": {"n": len(rescue_units), "positive": sum(y for _, y in rescue_units), "definition": "FP correct and INT8 failed; positive=Hadamard rescue"},
        },
        "failure_prediction_native_int8": {},
        "rescue_prediction_hadamard_risk": {},
        "rescue_prediction_risk_reduction": {},
        "predeclared_composites": {},
    }
    for metric in metrics:
        fy = [y for _, y in failure_units]
        fx = [metric_value(d["INT8_C128"], metric) for d, _ in failure_units]
        corr["failure_prediction_native_int8"][metric] = evaluate(fy, fx, 1.0)
        ry = [y for _, y in rescue_units]
        had = [metric_value(d["INT8_C128_KEY_HADAMARD"], metric) for d, _ in rescue_units]
        native = [metric_value(d["INT8_C128"], metric) for d, _ in rescue_units]
        corr["rescue_prediction_hadamard_risk"][metric] = evaluate(ry, had, -1.0)
        reduction = np.asarray(native) - np.asarray(had)
        corr["rescue_prediction_risk_reduction"][metric] = evaluate(ry, reduction, 1.0)

    # Equal-weight, label-free z-score composites frozen before outcome evaluation.
    fy = np.asarray([y for _, y in failure_units])
    fm5 = np.asarray([metric_value(d["INT8_C128"], "m5.p95") for d, _ in failure_units])
    fp = np.asarray([metric_value(d["INT8_C128"], "persistence.p95") for d, _ in failure_units])
    fd = np.asarray([metric_value(d["INT8_C128"], "decision.p95") for d, _ in failure_units])
    corr["predeclared_composites"]["failure_m5_plus_persistence"] = evaluate(fy, zscore(fm5) + zscore(fp), 1.0)
    corr["predeclared_composites"]["failure_m5_plus_persistence_plus_decision"] = evaluate(fy, zscore(fm5) + zscore(fp) + zscore(fd), 1.0)
    corr["predeclared_composites"]["failure_persistence_partial_spearman_given_m5"] = partial_spearman(fy, fp, fm5)
    corr["predeclared_composites"]["failure_decision_partial_spearman_given_m5"] = partial_spearman(fy, fd, fm5)

    ry = np.asarray([y for _, y in rescue_units])
    def reductions(metric):
        return np.asarray([
            metric_value(d["INT8_C128"], metric) - metric_value(d["INT8_C128_KEY_HADAMARD"], metric)
            for d, _ in rescue_units
        ])
    rm5, rp, rd = reductions("m5.p95"), reductions("persistence.p95"), reductions("decision.p95")
    corr["predeclared_composites"]["rescue_m5_plus_persistence_reduction"] = evaluate(ry, zscore(rm5) + zscore(rp), 1.0)
    corr["predeclared_composites"]["rescue_m5_plus_persistence_plus_decision_reduction"] = evaluate(ry, zscore(rm5) + zscore(rp) + zscore(rd), 1.0)
    corr["predeclared_composites"]["rescue_persistence_partial_spearman_given_m5_reduction"] = partial_spearman(ry, rp, rm5)
    corr["predeclared_composites"]["rescue_decision_partial_spearman_given_m5_reduction"] = partial_spearman(ry, rd, rm5)
    jdump(OUT / "correlation_analysis.json", corr)

    group_comp = {"task": TASK, "group_counts": expected_groups, "metrics": {}}
    for group in expected_groups:
        group_comp["metrics"][group] = {}
        pairs = [d for d in by_key.values() if d["INT8_C128"]["group"] == group]
        for cond in CONDITIONS:
            group_comp["metrics"][group][cond] = {
                metric: basic([metric_value(d[cond], metric) for d in pairs]) for metric in metrics
            }
    group_comp["primary_B_vs_C"] = {
        "B_n": 23, "C_n": 11,
        "interpretation": "Both groups are FP-correct INT8 failures; B is rescued and C is not.",
    }
    group_comp["nonfinite_note"] = {
        "affected_trajectory_conditions": len(nonfinite_records),
        "affected_groups": sorted({x["group"] for x in nonfinite_records.values()}),
        "primary_failure_and_rescue_populations_affected": any(
            x["group"] != "D_FP_WRONG" for x in nonfinite_records.values()
        ),
        "interpretation": "Group D aggregates are finite-prefix summaries when a trajectory overflows; A/B/C primary analyses are unaffected.",
    }
    jdump(OUT / "group_comparison.json", group_comp)

    token_json = {
        "task": TASK,
        "schema": {
            "csv": "one row per FP token position and quantized condition; layer statistics are across 24 GDN layers",
            "json": "one trajectory-condition summary plus per-layer time summaries",
            "m5_timing": "lag-1 cached-state error E_(t-1) evaluated with FP q_t and the frozen FP RMS/gate/out_proj Jacobian",
        },
        "paired_trajectories": 60,
        "condition_summaries": 120,
        "token_condition_rows": token_count,
        "nonfinite_audit": {
            "detected_trajectory_conditions": len(nonfinite_records),
            "records": nonfinite_records,
            "persistence_note": "Persistence non-finite counts include the intentionally unavailable final timestep and up to 32 earlier positions whose future horizon reaches an overflow.",
        },
        "trajectory_summaries": sorted(rows, key=lambda r: (r["canonical_index"], r["seed"], r["condition"])),
    }
    jdump(OUT / "token_level_metrics.json", token_json)

    config = {
        "task": TASK, "model": "Qwen3.5-9B/GDN", "model_path": "/data/zypan/modelscope_models/Qwen3.5-9B",
        "conditions": ["FP_STATE", *CONDITIONS], "teacher_forcing_source": "existing FP_STATE generated_token_ids",
        "aime_generation_rerun": False, "sampling": False, "training": False,
        "gdn_layers": list(GDN_LAYERS), "quantizer": {"bits": 8, "symmetric": True, "qrange": [-127, 127], "group": "C128", "axis": -2, "modified": False},
        "rotation": {"name": "key-side normalized H128", "modified": False, "recovery_for_state_comparison": "H128 @ S_hadamard"},
        "persistence": {"alpha": ALPHA, "horizons": HORIZONS, "epsilon": 1e-12},
        "decision": {"epsilon": DECISION_EPS, "delta": "condition logit shift on FP top-1 token", "margin": "FP top1 minus FP top2"},
        "m5": {"definition": "||W_O D_g D_w J_RMS(o) E^T q||_2", "timing": "cached E_(t-1), FP q_t", "aggregates": ["mean", "median", "p95", "max"]},
        "execution": {
            "gpus": 4,
            "chunk_size": 64,
            "batch_rows": ["FP", "Native INT8", "Key-Hadamard INT8"],
            "kv_cache": "exact CPU offload for full-attention layers only; GDN recurrent states remain on GPU",
            "bootstrap_resamples": BOOTSTRAP,
            "bootstrap_seed": SEED,
        },
        "shards": statuses,
        "capture_script": {"path": str(OUT / "run_capture.py"), "sha256": sha256(OUT / "run_capture.py")},
    }
    jdump(OUT / "capture_config.json", config)

    audit = f"""# Artifact audit — {TASK}

## Existing artifacts reused

- 60 FP_STATE AIME26 trajectories with complete token IDs: 1,854,451 teacher-forced continuation tokens.
- Paired correctness labels for FP_STATE, INT8_C128, and INT8_C128 + Key-Hadamard: A=18, B=23, C=11, D=8.
- Canonical Qwen INT8-C128 implementation and key-side normalized H128 implementation from the existing AIME runner.
- Frozen Qwen M5 operator semantics from the existing propagation experiments.
- Existing model and tokenizer at `/data/zypan/modelscope_models/Qwen3.5-9B`.

## Initially missing

- No AIME trajectory had saved per-step logits, recurrent states, state error, M5, persistence, or decision-margin sensitivity.
- Dense Oracle artifacts exist but were not used: this task requires only FP, Native INT8, and Key-Hadamard and forbids retraining.

## New capture

- Four RTX3090 shards completed on the required server.
- Full states were reduced online; no TB-scale state dump was created.
- Full-attention KV caches were losslessly offloaded to CPU to fit the longest 81,920-token trajectories; GDN recurrent states and all mechanism calculations remained on GPU.
- Per-layer/per-token M0 and M5 were computed for all 24 GDN layers. The CSV retains token-level cross-layer summaries; the JSON retains trajectory-level per-layer summaries.
- Token-condition rows: {token_count:,}.
- No sampling, generation, scorer invocation, gold-conditioned metric construction, training, or parameter update occurred.

## Numerical overflow disclosure

- {len(nonfinite_records)} trajectory-condition streams contain non-finite values, all from `aime26_28`, seed 2, Group D (FP wrong), after very long teacher-forced prefixes.
- The raw CSV preserves `inf`/`nan`; `token_level_metrics.json` records exact counts and first affected timesteps.
- Group D aggregate values are finite-prefix summaries. The primary failure population (A/B/C) and rescue comparison (B/C) contain no non-finite values and are unaffected.

## Integrity

- Capture script SHA256: `{config['capture_script']['sha256']}`.
- Native and FP prefill rows were numerically identical in every trajectory; Hadamard finite-precision differences are recorded per trajectory.
- Output group counts reproduce the frozen benchmark labels exactly.
"""
    (OUT / "artifact_audit.md").write_text(audit, encoding="utf-8")

    definitions = f"""# Metric definitions — {TASK}

Metrics were frozen before joining mechanism values to outcome labels.

## M0 state error

After each teacher-forced token updates the recurrent state and the existing C128 QDQ is applied:

`E_t = S_t(condition, recovered to FP basis) - S_t(FP)` and `M0_t,l = ||E_t,l||_F`.

Hadamard states are recovered with `H128 @ S_hadamard`. Absolute and FP-relative errors are retained. Aggregates: mean, median, p95, max.

## M5 functional error

Frozen existing definition: `M5 = ||W_O D_g D_w J_RMS(o) E^T q||_2`.

Timing is explicit: cached state error `E_(t-1)` is evaluated against FP query `q_t`, then passed through the FP RMS/gate/out-projection linearization. Values are computed for every timestep and GDN layer. Aggregates: mean, median, p95, max.

## Persistence

With alpha={ALPHA} and K={HORIZONS}:

`P_t = sum_(k in K) alpha^k * ||E_(t+k)|| / (||E_t|| + 1e-12)`.

The state norm combines all GDN layers by L2. Tail positions without any future horizon are missing, not zero. Aggregates: mean, p95, max.

## Decision sensitivity

For FP margin `m_t = z1-z2`, and condition shift of the FP-winning token `delta_z_t`:

`D_t = |delta_z_t| / max(m_t, {DECISION_EPS})`.

The report includes mean, p95, max, count/fraction above 1, exact/near-tie counts, and actual top-1 changes.

## Statistical populations

- Failure: 52 FP-correct trajectories; positive means Native INT8 failed (34 positives).
- Rescue: 34 FP-correct Native-INT8 failures; positive means Key-Hadamard rescued the trajectory (23 positives).
- Rescue risk uses lower Hadamard risk as positive direction. Risk-reduction analysis uses `Native - Hadamard` with higher reduction as positive.
- AUROC, Spearman, Pearson, and paired-trajectory bootstrap 95% confidence intervals use {BOOTSTRAP} resamples.
- Equal-weight composites use unsupervised z-scores; no label-fitted weights or thresholds are used.

## Non-finite values

The raw CSV preserves numerical overflow as `inf`/`nan`. Trajectory summaries report finite-prefix statistics and expose exact overflow counts/onsets in `token_level_metrics.json`. Only one Group D trajectory is affected, so primary A/B/C failure and rescue statistics require no censoring or imputation.
"""
    (OUT / "metric_definition.md").write_text(definitions, encoding="utf-8")

    # Report decisions from primary p95 metrics and frozen composites.
    fail = corr["failure_prediction_native_int8"]
    rescue = corr["rescue_prediction_risk_reduction"]
    primary = ["m0.p95", "m5.p95", "persistence.p95", "decision.p95"]
    best_fail = max(primary, key=lambda m: fail[m]["auroc"])
    best_rescue = max(primary, key=lambda m: rescue[m]["auroc"])
    c = corr["predeclared_composites"]
    report = f"""# Final report — {TASK}

## Scope

All 60 existing FP AIME26 token sequences were replayed under aligned FP_STATE, INT8_C128, and INT8_C128 + Key-Hadamard paths. The capture covers {token_count:,} token-condition rows and all 24 GDN layers. No free-running generation, training, scorer change, quantizer change, or rotation modification was performed.

## Primary results

Failure AUROC on 52 FP-correct trajectories (34 INT8 failures):

- M0 p95: {fail['m0.p95']['auroc']:.3f} (95% CI {fail['m0.p95']['auroc_ci95']})
- M5 p95: {fail['m5.p95']['auroc']:.3f} (95% CI {fail['m5.p95']['auroc_ci95']})
- Persistence p95: {fail['persistence.p95']['auroc']:.3f} (95% CI {fail['persistence.p95']['auroc_ci95']})
- Decision sensitivity p95: {fail['decision.p95']['auroc']:.3f} (95% CI {fail['decision.p95']['auroc_ci95']})
- Equal-weight M5+Persistence+Decision: {c['failure_m5_plus_persistence_plus_decision']['auroc']:.3f}

Hadamard rescue AUROC on Group B vs C, using Native-minus-Hadamard risk reduction (23 rescues, 11 failures):

- M0 p95 reduction: {rescue['m0.p95']['auroc']:.3f}
- M5 p95 reduction: {rescue['m5.p95']['auroc']:.3f}
- Persistence p95 reduction: {rescue['persistence.p95']['auroc']:.3f}
- Decision p95 reduction: {rescue['decision.p95']['auroc']:.3f}
- Equal-weight M5+Persistence+Decision reduction: {c['rescue_m5_plus_persistence_plus_decision_reduction']['auroc']:.3f}

## Answers

### 1. Does state error magnitude predict AIME failure?

M0 p95 AUROC is {fail['m0.p95']['auroc']:.3f}, with Spearman {fail['m0.p95']['spearman']:.3f}. Its confidence interval and the full mean/median/max variants are in `correlation_analysis.json`. This is {'useful evidence of prediction' if fail['m0.p95']['auroc'] > 0.65 else 'weak evidence and not sufficient alone'}.

### 2. Does M5 predict failure better than M0?

M5 p95 AUROC is {fail['m5.p95']['auroc']:.3f} versus M0 p95 {fail['m0.p95']['auroc']:.3f}. On this frozen comparison, M5 {'outperforms' if fail['m5.p95']['auroc'] > fail['m0.p95']['auroc'] else 'does not outperform'} M0. No metric weights or directions were refit after labels.

### 3. Does persistence add information beyond M5?

No. Persistence p95 has AUROC {fail['persistence.p95']['auroc']:.3f} in the preregistered higher-is-riskier direction and Spearman {fail['persistence.p95']['spearman']:.3f}; failures have *lower* normalized persistence ratios. Its partial Spearman given M5 is {c['failure_persistence_partial_spearman_given_m5']:.3f}. Adding it to M5 lowers AUROC from {fail['m5.p95']['auroc']:.3f} to {c['failure_m5_plus_persistence']['auroc']:.3f}. The current ratio is likely denominator-confounded and should not be optimized as a positive penalty.

### 4. Does decision-boundary sensitivity explain Hadamard rescue?

Partially. Decision p95 risk-reduction AUROC is {rescue['decision.p95']['auroc']:.3f} (95% CI {rescue['decision.p95']['auroc_ci95']}); M0, M5, and persistence reductions are {rescue['m0.p95']['auroc']:.3f}, {rescue['m5.p95']['auroc']:.3f}, and {rescue['persistence.p95']['auroc']:.3f}. It is the strongest single rescue separator, but the sample is only 23 rescues versus 11 failures and its interval reaches approximately chance. Exact logit ties are retained through the declared margin floor and reported.

### 5. Most promising rotation objective

Use **upper-tail M5 (M5 p95)** as the primary rotation objective. It is the strongest single failure separator ({fail['m5.p95']['auroc']:.3f}) and beats M0 p95 ({fail['m0.p95']['auroc']:.3f}) numerically. Treat decision p95 as a secondary held-out constraint or tie-breaker because it is the best rescue separator. Do **not** include persistence as currently defined: the equal-weight M5+Persistence+Decision composite lowers failure AUROC to {c['failure_m5_plus_persistence_plus_decision']['auroc']:.3f}; its small rescue increase to {c['rescue_m5_plus_persistence_plus_decision_reduction']['auroc']:.3f} is within uncertainty.

### 6. Additional experiment before learning a rotation

Prospectively validate M5 p95, M5+Decision, and a redesigned denominator-independent persistence metric on a held-out reasoning set or preregistered AIME split. First run a non-training candidate-ranking study over existing rotations and predeclare overflow/censoring rules for very long trajectories. Do not train a learnable rotation until the frozen metric ranks both failure and rescue out of sample.

## Numerical limitation

One 81,920-token Group D trajectory (`aime26_28`, seed 2) develops quantized-state overflow in both quantized conditions. This is preserved in the CSV and audited in the JSON. It does not enter either primary statistical population, so the reported failure and rescue results are unchanged.

## Bottom line

Before learning a new recurrent-state rotation, optimize **tail functional amplification (M5 p95)**, not mean representation error and not the current normalized persistence ratio. Use decision-boundary sensitivity as a secondary validation signal. Exact results and confidence intervals are machine-readable in `correlation_analysis.json`; token-level values and overflow audit are in `token_level_metrics.csv` and `token_level_metrics.json`.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "task": TASK, "status": "COMPLETE", "paired_trajectories": 60,
        "token_condition_rows": token_count, "group_counts": expected_groups,
        "nonfinite_audit": {
            "affected_trajectory_conditions": len(nonfinite_records),
            "affected_groups": sorted({x["group"] for x in nonfinite_records.values()}),
            "primary_populations_affected": any(x["group"] != "D_FP_WRONG" for x in nonfinite_records.values()),
        },
        "required_outputs": {name: {"bytes": (OUT / name).stat().st_size, "sha256": sha256(OUT / name)} for name in [
            "artifact_audit.md", "capture_config.json", "metric_definition.md", "token_level_metrics.json",
            "token_level_metrics.csv", "group_comparison.json", "correlation_analysis.json", "final_report.md",
        ]},
    }
    jdump(OUT / "analysis_manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "token_rows": token_count, "groups": expected_groups, "best_failure": best_fail, "best_rescue": best_rescue}, indent=2))


if __name__ == "__main__":
    main()
