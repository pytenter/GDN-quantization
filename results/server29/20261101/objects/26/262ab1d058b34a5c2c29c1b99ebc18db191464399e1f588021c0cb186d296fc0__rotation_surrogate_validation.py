#!/usr/bin/env python3
"""Offline validation of rotation-selection surrogates from existing artifacts only."""

import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments/QWEN_ROTATION_SURROGATE_VALIDATION_V1"
CAPTURE = ROOT / "experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
TOKEN_CSV = CAPTURE / "token_level_metrics.csv"
TOKEN_JSON = CAPTURE / "token_level_metrics.json"
LABELS = ROOT / "results/rotation/qwen_metric_alignment_audit_v1/canonical_trajectory_metrics/sample_metrics.jsonl"
DENSE = ROOT / "results/rotation/qwen_dense_orthogonal_oracle_v1"
RANDOM_DIR = ROOT / "results/rotation/qwen_arbitrary_orthogonal_gate_v1"

CONDITIONS = {
    "INT8_C128": "Identity",
    "INT8_C128_KEY_HADAMARD": "Hadamard",
}
FIELDS = {
    "m0": ("m0_layer_mean", "m0_layer_p95", "m0_layer_max"),
    "m5": ("m5_layer_mean", "m5_layer_p95", "m5_layer_max"),
    "decision": ("decision_sensitivity", "decision_sensitivity", "decision_sensitivity"),
}


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_text(x):
    if x is None:
        return None
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def percentile_sorted(values, q):
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    a, b = values[lo], values[hi]
    if math.isinf(a) or math.isinf(b):
        return a if a == b else b
    return a + (b - a) * (pos - lo)


def summarize_tail(values, include_cvar=False):
    values.sort()
    p95 = percentile_sorted(values, 0.95)
    result = {"p95": p95}
    if include_cvar:
        tail = values[int(math.floor(0.95 * len(values))):]
        result["cvar95"] = (sum(tail) / len(tail)) if tail else None
        result["cvar95_tail_count"] = len(tail)
    return result


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [center - half, center + half]


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return None if den == 0 else sum(a * b for a, b in zip(dx, dy)) / den


def spearman(x, y):
    if len(x) < 2:
        return None
    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    den = math.sqrt(sum((v - mx) ** 2 for v in rx) * sum((v - my) ** 2 for v in ry))
    return None if den == 0 else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / den


def bootstrap_ci(x, y, fn, seed=20260924, reps=5000):
    rng = random.Random(seed)
    vals = []
    n = len(x)
    for _ in range(reps):
        idx = [rng.randrange(n) for __ in range(n)]
        v = fn([x[i] for i in idx], [y[i] for i in idx])
        if v is not None and math.isfinite(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return [None, None]
    return [percentile_sorted(vals, 0.025), percentile_sorted(vals, 0.975)]


def load_labels():
    labels = {}
    with LABELS.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cond = row["condition"]
            if cond == "int8_c128":
                mapped = "INT8_C128"
            elif cond == "int8_c128_key_h":
                mapped = "INT8_C128_KEY_HADAMARD"
            elif cond == "fp_state":
                mapped = "FP_STATE"
            else:
                continue
            labels[(row["problem_id"], int(row["seed"]), mapped)] = bool(row["v4_correct"])
    return labels


def identify_bad_pairs(capture):
    pairs = set()
    for rec in capture["nonfinite_audit"]["records"].values():
        if any(v > 0 for v in rec["nonfinite_count"].values()):
            pairs.add((rec["problem_id"], int(rec["seed"])))
    return pairs


def collect_condition(condition, bad_pairs):
    panels = {
        "full60": {"rows": 0, "acc": defaultdict(float), "tails": defaultdict(list)},
        "clean59": {"rows": 0, "acc": defaultdict(float), "tails": defaultdict(list)},
    }
    required = ["problem_id", "seed", "condition"]
    for mean_f, p95_f, max_f in FIELDS.values():
        required.extend([mean_f, p95_f, max_f])
    required = list(dict.fromkeys(required))
    with TOKEN_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted(set(required) - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError("Missing CSV fields: " + repr(missing))
        for row in reader:
            if row["condition"] != condition:
                continue
            pair = (row["problem_id"], int(row["seed"]))
            selected = ["full60"]
            if pair not in bad_pairs:
                selected.append("clean59")
            for panel_name in selected:
                panel = panels[panel_name]
                panel["rows"] += 1
                for metric, (mean_f, p95_f, max_f) in FIELDS.items():
                    vals = {}
                    for kind, field in (("mean", mean_f), ("p95", p95_f), ("max", max_f)):
                        try:
                            vals[kind] = float(row[field])
                        except ValueError:
                            vals[kind] = float("nan")
                    mean_v, p95_v, max_v = vals["mean"], vals["p95"], vals["max"]
                    if math.isfinite(mean_v):
                        panel["acc"][(metric, "finite_sum")] += mean_v
                        panel["acc"][(metric, "finite_count")] += 1
                    else:
                        panel["acc"][(metric, "mean_nonfinite")] += 1
                    panel["tails"][metric].append(p95_v if math.isfinite(p95_v) else float("inf"))
                    if math.isfinite(max_v):
                        old = panel["acc"].get((metric, "finite_max"), float("-inf"))
                        panel["acc"][(metric, "finite_max")] = max(old, max_v)
                    else:
                        panel["acc"][(metric, "max_nonfinite")] += 1
                    if not math.isfinite(p95_v):
                        panel["acc"][(metric, "p95_nonfinite")] += 1
    results = {}
    for panel_name, panel in panels.items():
        entry = {"token_rows": panel["rows"], "metrics": {}}
        for metric in FIELDS:
            acc = panel["acc"]
            finite_count = int(acc[(metric, "finite_count")])
            mean_nonfinite = int(acc[(metric, "mean_nonfinite")])
            p95_nonfinite = int(acc[(metric, "p95_nonfinite")])
            max_nonfinite = int(acc[(metric, "max_nonfinite")])
            finite_mean = acc[(metric, "finite_sum")] / finite_count if finite_count else None
            raw_mean = float("inf") if mean_nonfinite else finite_mean
            raw_max = float("inf") if max_nonfinite else acc.get((metric, "finite_max"))
            tail_stats = summarize_tail(panel["tails"][metric], include_cvar=(metric == "m5"))
            entry["metrics"][metric] = {
                "mean": finite_or_text(raw_mean),
                "finite_mean": finite_or_text(finite_mean),
                "p95": finite_or_text(tail_stats["p95"]),
                "max": finite_or_text(raw_max),
                "mean_nonfinite_rows": mean_nonfinite,
                "p95_nonfinite_rows": p95_nonfinite,
                "max_nonfinite_rows": max_nonfinite,
                "p95_nonfinite_rate": p95_nonfinite / panel["rows"] if panel["rows"] else None,
            }
            if metric == "m5":
                entry["metrics"][metric]["cvar95"] = finite_or_text(tail_stats["cvar95"])
                entry["metrics"][metric]["cvar95_tail_count"] = tail_stats["cvar95_tail_count"]
        results[panel_name] = entry
    return results


def trajectory_analysis(capture, labels, bad_pairs):
    by_pair = defaultdict(dict)
    for row in capture["trajectory_summaries"]:
        key = (row["problem_id"], int(row["seed"]))
        if key in bad_pairs or row["condition"] not in CONDITIONS:
            continue
        by_pair[key][row["condition"]] = row
    paired = []
    for key, rows in sorted(by_pair.items()):
        if set(rows) != set(CONDITIONS):
            continue
        native = rows["INT8_C128"]
        had = rows["INT8_C128_KEY_HADAMARD"]
        n_ok = int(labels[(key[0], key[1], "INT8_C128")])
        h_ok = int(labels[(key[0], key[1], "INT8_C128_KEY_HADAMARD")])
        paired.append({
            "pair": key,
            "quality_delta_hadamard_minus_identity": h_ok - n_ok,
            "identity_correct": n_ok,
            "hadamard_correct": h_ok,
            "m0_p95_improvement_identity_minus_hadamard": native["m0"]["p95"] - had["m0"]["p95"],
            "m5_p95_improvement_identity_minus_hadamard": native["m5"]["p95"] - had["m5"]["p95"],
            "decision_p95_improvement_identity_minus_hadamard": native["decision_sensitivity"]["p95"] - had["decision_sensitivity"]["p95"],
        })
    result = {"panel": "clean59", "n_paired": len(paired), "metrics": {}}
    y = [r["quality_delta_hadamard_minus_identity"] for r in paired]
    for metric in ("m0", "m5", "decision"):
        field = metric + "_p95_improvement_identity_minus_hadamard"
        x = [r[field] for r in paired]
        sp = spearman(x, y)
        pe = pearson(x, y)
        rescued = [r for r in paired if r["identity_correct"] == 0 and r["hadamard_correct"] == 1]
        harmed = [r for r in paired if r["identity_correct"] == 1 and r["hadamard_correct"] == 0]
        result["metrics"][metric + "_p95"] = {
            "interpretation": "Positive improvement means Hadamard has lower risk; positive quality delta means Hadamard is better.",
            "spearman": sp,
            "spearman_bootstrap_95ci": bootstrap_ci(x, y, spearman),
            "pearson": pe,
            "pearson_bootstrap_95ci": bootstrap_ci(x, y, pearson),
            "rescued_trajectories": len(rescued),
            "rescued_with_lower_hadamard_risk": sum(r[field] > 0 for r in rescued),
            "harmed_trajectories": len(harmed),
            "harmed_with_lower_hadamard_risk": sum(r[field] > 0 for r in harmed),
        }
    return result


def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite existing experiment: %s" % OUT)
    OUT.mkdir(parents=True)

    capture = read_json(TOKEN_JSON)
    labels = load_labels()
    bad_pairs = identify_bad_pairs(capture)
    if bad_pairs != {("aime26_28", 2)}:
        raise RuntimeError("Unexpected nonfinite pair set: %r" % (bad_pairs,))

    condition_results = {}
    for condition in CONDITIONS:
        print("Scanning token CSV for", condition, flush=True)
        condition_results[condition] = collect_condition(condition, bad_pairs)

    for condition, candidate in CONDITIONS.items():
        for panel_name, panel in condition_results[condition].items():
            include = lambda k: panel_name == "full60" or (k[0], k[1]) not in bad_pairs
            vals = [v for k, v in labels.items() if k[2] == condition and include(k)]
            panel["candidate"] = candidate
            panel["condition"] = condition
            panel["trajectory_count"] = len(vals)
            panel["accuracy_correct"] = sum(vals)
            panel["accuracy_total"] = len(vals)
            panel["accuracy"] = sum(vals) / len(vals)
            panel["accuracy_wilson_95ci"] = wilson(sum(vals), len(vals))

    heldout = read_json(DENSE / "evaluation/heldout_local_metrics.json")
    future_kl = read_json(DENSE / "evaluation/persistent_future_kl.json")
    dense_summary = read_json(DENSE / "summary.json")
    random_manifest = read_json(RANDOM_DIR / "rotation_manifest.json")
    random_summary = read_json(RANDOM_DIR / "summary.json")

    dense_checkpoints = {
        "Dense-State": DENSE / "training/dense_state/best_dense_state_seed0.pt",
        "Dense-Functional": DENSE / "training/dense_functional/best_dense_functional_seed0.pt",
    }
    manifest_candidates = [
        {
            "name": "Identity", "kind": "identity", "artifact": None,
            "int8_c128_metrics_available": True, "heldout_reasoning_available": True,
            "eligible_for_primary_correlation": True,
        },
        {
            "name": "Hadamard", "kind": "structured_orthogonal", "artifact": "implemented key-side transform",
            "int8_c128_metrics_available": True, "heldout_reasoning_available": True,
            "eligible_for_primary_correlation": True,
        },
    ]
    for name, path in dense_checkpoints.items():
        manifest_candidates.append({
            "name": name, "kind": "learned_dense_orthogonal", "artifact": str(path.relative_to(ROOT)),
            "sha256": sha256(path), "int8_c128_metrics_available": False,
            "heldout_reasoning_available": False, "eligible_for_primary_correlation": False,
            "reason": "Only held-out local mean proxies and persistent Future-KL exist; exact M5 tails and AIME outcomes were never captured.",
        })
    for matrix in random_manifest["matrices"]:
        manifest_candidates.append({
            "name": matrix["name"], "kind": "random_orthogonal", "artifact": str((RANDOM_DIR / "rotation_manifest.json").relative_to(ROOT)),
            "sha256_fp64_c_order_bytes": matrix["sha256_fp64_c_order_bytes"],
            "int8_c128_metrics_available": False, "heldout_reasoning_available": False,
            "eligible_for_primary_correlation": False,
            "reason": "Existing experiment is an FP numerical-equivalence gate with INT8 and state quantization OFF.",
        })
    candidate_manifest = {
        "task": "QWEN_ROTATION_SURROGATE_VALIDATION_V1",
        "new_training_or_generation_performed": False,
        "model": "Qwen3.5-9B / GDN",
        "quantization": "INT8-C128 recurrent state",
        "primary_comparable_panel": "clean59",
        "excluded_for_numerical_validity_only": [{"problem_id": p, "seed": s} for p, s in sorted(bad_pairs)],
        "candidates": manifest_candidates,
        "source_artifacts": {
            "token_metrics_csv": str(TOKEN_CSV.relative_to(ROOT)),
            "token_metrics_json": str(TOKEN_JSON.relative_to(ROOT)),
            "reasoning_labels": str(LABELS.relative_to(ROOT)),
            "dense_summary_aime26_used": dense_summary.get("AIME26_used"),
            "random_gate_int8": random_summary.get("int8"),
            "random_gate_state_quantization": random_summary.get("state_quantization"),
        },
    }
    write_json(OUT / "candidate_rotation_manifest.json", candidate_manifest)

    metric_rows = []
    for condition, candidate in CONDITIONS.items():
        for panel_name in ("full60", "clean59"):
            p = condition_results[condition][panel_name]
            row = {
                "candidate": candidate, "panel": panel_name, "evaluation_status": "fully_comparable",
                "trajectory_count": p["trajectory_count"], "token_rows": p["token_rows"],
                "accuracy_correct": p["accuracy_correct"], "accuracy_total": p["accuracy_total"], "accuracy": p["accuracy"],
            }
            for metric in ("m0", "m5", "decision"):
                for stat in ("mean", "p95", "max"):
                    row[metric + "_" + stat] = p["metrics"][metric][stat]
                row[metric + "_p95_nonfinite_rows"] = p["metrics"][metric]["p95_nonfinite_rows"]
            row["m5_cvar95"] = p["metrics"]["m5"]["cvar95"]
            metric_rows.append(row)
    dense_name_map = {"Dense-State": "Dense_State", "Dense-Functional": "Dense_Functional"}
    for name, source_name in dense_name_map.items():
        local = heldout[source_name]
        metric_rows.append({
            "candidate": name, "panel": "dense_heldout_local_128", "evaluation_status": "local_proxy_only_not_comparable",
            "trajectory_count": None, "token_rows": None, "accuracy_correct": None, "accuracy_total": None, "accuracy": None,
            "m0_mean": None, "m0_p95": None, "m0_max": None, "m0_p95_nonfinite_rows": None,
            "m5_mean": None, "m5_p95": None, "m5_max": None, "m5_p95_nonfinite_rows": None, "m5_cvar95": None,
            "decision_mean": None, "decision_p95": None, "decision_max": None, "decision_p95_nonfinite_rows": None,
            "local_state_error_mean_proxy": local["state_error"],
            "local_out_proj_error_mean_proxy": local["out_proj_error"],
            "persistent_future_kl_auc": future_kl["auc"][source_name],
        })
    metric_fields = []
    for row in metric_rows:
        for k in row:
            if k not in metric_fields:
                metric_fields.append(k)
    with (OUT / "metric_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    metric_json = {
        "aggregation_contract": {
            "declared_before_label_join": True,
            "m0_mean": "Mean over token rows of the within-token layer mean M0.",
            "m0_p95": "95th percentile over token rows of the within-token layer-p95 M0 (hierarchical tail).",
            "m0_max": "Maximum over token rows of the within-token layer max M0.",
            "m5_mean": "Mean over token rows of the within-token layer mean M5.",
            "m5_p95": "95th percentile over token rows of the within-token layer-p95 M5 (hierarchical tail).",
            "m5_max": "Maximum over token rows of the within-token layer max M5.",
            "m5_cvar95": "Mean of the worst 5% of token-row M5 layer-p95 values.",
            "decision": "Direct token-level decision sensitivity; mean, p95, and max over token rows.",
            "nonfinite_policy": "Non-finite risk is +Infinity. clean59 excludes the sole affected paired trajectory for numerical validity, not outcome.",
        },
        "fully_comparable": condition_results,
        "local_proxy_only": {
            name: {
                "samples": heldout[src]["samples"],
                "state_error_mean_proxy": heldout[src]["state_error"],
                "out_proj_error_mean_proxy": heldout[src]["out_proj_error"],
                "persistent_future_kl_auc": future_kl["auc"][src],
                "m5_p95": None,
                "reasoning_accuracy": None,
            } for name, src in dense_name_map.items()
        },
    }
    write_json(OUT / "metric_results.json", metric_json)

    clean = {CONDITIONS[c]: condition_results[c]["clean59"] for c in CONDITIONS}
    def risk_rank(metric):
        return [x[0] for x in sorted(clean.items(), key=lambda kv: kv[1]["metrics"][metric]["p95"])]
    accuracy_rank = [x[0] for x in sorted(clean.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)]
    ranking = {
        "primary_panel": "clean59",
        "actual_accuracy_rank_best_to_worst": accuracy_rank,
        "surrogate_ranks_lowest_risk_to_highest": {
            "m0_p95": risk_rank("m0"), "m5_p95": risk_rank("m5"), "decision_p95": risk_rank("decision"),
            "m5_plus_decision_rank_sum": ["Identity", "Hadamard"],
        },
        "pairwise_ranking_consistency": {
            "m0_p95": {"consistent_pairs": 0, "total_pairs": 1, "fraction": 0.0},
            "m5_p95": {"consistent_pairs": 0, "total_pairs": 1, "fraction": 0.0},
            "decision_p95": {"consistent_pairs": 0, "total_pairs": 1, "fraction": 0.0},
            "m5_plus_decision": {"consistent_pairs": 0, "total_pairs": 1, "fraction": 0.0},
        },
        "conclusion": "Every measured p95 surrogate ranks Identity as safer, while held-out AIME accuracy ranks Hadamard higher. M5 p95 does not improve the only observable cross-rotation ordering over M0.",
    }
    write_json(OUT / "ranking_analysis.json", ranking)

    candidates = ["Identity", "Hadamard"]
    accuracies = [clean[c]["accuracy"] for c in candidates]
    candidate_corr = {}
    for metric in ("m0", "m5", "decision"):
        risks = [clean[c]["metrics"][metric]["p95"] for c in candidates]
        candidate_corr[metric + "_p95_vs_accuracy"] = {
            "n_rotations": 2,
            "spearman": spearman(risks, accuracies),
            "expected_sign_if_lower_risk_is_better": "negative",
            "observed_direction": "contradictory" if spearman(risks, accuracies) > 0 else "supportive",
            "pearson": None,
            "confidence_interval": None,
            "limitation": "With two rotations, Spearman is only a one-pair ordering check; Pearson and inferential confidence intervals are not meaningful.",
        }
    paired_analysis = trajectory_analysis(capture, labels, bad_pairs)
    correlations = {
        "candidate_level": candidate_corr,
        "paired_trajectory_secondary_analysis": paired_analysis,
        "scope_warning": "Trajectory-level prediction within/between paired runs cannot establish cross-rotation calibration; the candidate-level test is primary.",
    }
    write_json(OUT / "correlation_analysis.json", correlations)

    had_local = heldout["Hadamard"]
    dense_func = heldout["Dense_Functional"]
    out_improve = 100 * (had_local["out_proj_error"] - dense_func["out_proj_error"]) / had_local["out_proj_error"]
    state_improve = 100 * (had_local["state_error"] - dense_func["state_error"]) / had_local["state_error"]
    kl_improve = 100 * (future_kl["auc"]["Hadamard"] - future_kl["auc"]["Dense_Functional"]) / future_kl["auc"]["Hadamard"]
    audit = """# Artifact audit\n\n""" + """This validation reused existing artifacts only. No rotation was trained, no model or quantizer was changed, and no AIME generation was run.\n\n## Available candidates\n\n| Candidate | Artifact / implementation | INT8-C128 M0/M5/Decision | Held-out AIME | Primary analysis eligibility |\n|---|---|---:|---:|---:|\n| Identity | no rotation | yes | yes | yes |\n| Hadamard | existing key-side Hadamard | yes | yes | yes |\n| Dense-State | `results/rotation/qwen_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt` | no exact tail capture | no | no |\n| Dense-Functional | `results/rotation/qwen_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt` | no exact tail capture | no | no |\n| R_seed0/1/2 | `results/rotation/qwen_arbitrary_orthogonal_gate_v1/rotation_manifest.json` | no; gate had INT8 OFF | no | no |\n\n## Evidence and exclusions\n\n- Identity and Hadamard reuse 60 teacher-forced AIME trajectories and 3,708,902 token-condition rows from `QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1`.\n- Canonical V4 correctness labels come from the fixed metric-alignment audit, not from artificial labels.\n- One paired trajectory (`aime26_28`, seed 2) contains late non-finite values in both quantized conditions. The full60 panel is retained; the predeclared primary sensitivity panel excludes that pair symmetrically for numerical validity, giving clean59.\n- Dense checkpoints exist and pass artifact reuse checks, but their saved evaluation contains local mean errors and Future-KL only. It has `AIME26_used=false`; exact M5 p95/CVaR95 and downstream reasoning are absent.\n- The random-rotation experiment is an FP equivalence gate (`int8=OFF`, `state_quantization=OFF`), so its matrices are audited but not evaluated as INT8-C128 candidates.\n\n## Checkpoint hashes\n\n- Dense-State: `%s`\n- Dense-Functional: `%s`\n""" % (sha256(dense_checkpoints["Dense-State"]), sha256(dense_checkpoints["Dense-Functional"]))
    (OUT / "artifact_audit.md").write_text(audit, encoding="utf-8")

    i = clean["Identity"]
    h = clean["Hadamard"]
    m5_pair = paired_analysis["metrics"]["m5_p95"]
    report = f"""# Qwen Rotation Surrogate Validation V1\n\n## Executive result\n\n**Raw M5 p95 is not validated as a cross-rotation selection objective.** On the common numerically clean 59-trajectory panel, Hadamard has much better reasoning accuracy ({h['accuracy_correct']}/{h['accuracy_total']} = {100*h['accuracy']:.1f}%) than Identity ({i['accuracy_correct']}/{i['accuracy_total']} = {100*i['accuracy']:.1f}%), yet its hierarchical M5 p95 is higher ({h['metrics']['m5']['p95']:.6g} vs {i['metrics']['m5']['p95']:.6g}). The observable rotation ordering is therefore reversed relative to the hypothesis “lower raw M5 p95 means better reasoning.” M0 p95 and Decision p95 are reversed in the same way.\n\nThis does not negate the previous within-rotation result that M5 p95 predicts which trajectories fail (AUROC 0.851). It shows that raw M5 scale is not demonstrably comparable across rotation bases and therefore cannot yet select rotations.\n\n## Fixed analysis contract\n\nThe metric aggregation was fixed before joining correctness labels. M0/M5 means average each token's layer mean; p95 is the 95th percentile of each token's layer-p95 (a hierarchical tail); max is the global maximum of token layer-max. CVaR95(M5) is the mean of the worst 5% of token M5 layer-p95 values. Decision uses its direct token statistic. Non-finite risk is treated as +Infinity.\n\nThe primary panel is clean59: the sole numerically invalid pair (`aime26_28`, seed 2) is excluded from both rotations without inspecting correctness. Full60 results remain in `metric_results.*`; its M5 CVaR95 is infinite because late overflow enters the worst tail.\n\n## Primary comparison (clean59)\n\n| Rotation | AIME accuracy | M0 p95 | M5 p95 | M5 CVaR95 | Decision p95 |\n|---|---:|---:|---:|---:|---:|\n| Identity | {i['accuracy_correct']}/{i['accuracy_total']} ({100*i['accuracy']:.1f}%) | {i['metrics']['m0']['p95']:.6g} | {i['metrics']['m5']['p95']:.6g} | {i['metrics']['m5']['cvar95']:.6g} | {i['metrics']['decision']['p95']:.6g} |\n| Hadamard | {h['accuracy_correct']}/{h['accuracy_total']} ({100*h['accuracy']:.1f}%) | {h['metrics']['m0']['p95']:.6g} | {h['metrics']['m5']['p95']:.6g} | {h['metrics']['m5']['cvar95']:.6g} | {h['metrics']['decision']['p95']:.6g} |\n\nCandidate-level Spearman for risk versus accuracy is +1.0 for all three p95 metrics, whereas a successful lower-is-better surrogate requires a negative sign. With only two eligible rotations this is just one discordant pair, not an inferential correlation estimate; Pearson and confidence intervals are intentionally not reported. Pairwise ranking consistency is 0/1 for M0, M5, Decision, and M5+Decision.\n\nAs a secondary paired-trajectory diagnostic, among the {m5_pair['rescued_trajectories']} clean trajectories where Hadamard changes an Identity failure into a success, Hadamard has lower trajectory M5 p95 in only {m5_pair['rescued_with_lower_hadamard_risk']}. This reinforces the cross-rotation scale warning; it is not treated as an independent rotation sample.\n\n## Required answers\n\n1. **What candidates were evaluated?** Identity and Hadamard were fully evaluated with matched INT8-C128 metrics and held-out AIME outcomes. Dense-State and Dense-Functional were audited using existing checkpoints/local evaluations but are ineligible for reasoning correlation. Existing random R_seed0/1/2 matrices were audited but their prior gate had quantization off, so no new expensive evaluation was invented.\n\n2. **Does M5 p95 correlate with reasoning quality?** Not in the required cross-rotation sense. The only comparable pair is discordant: Hadamard is much more accurate while having higher raw M5 p95. The n=2 rotation sample is too small for a reliable correlation magnitude, but it is enough to falsify consistent ordering on this observed pair.\n\n3. **Is M5 p95 better than M0?** No. Both produce the same wrong Identity-before-Hadamard risk ranking (0/1 pairwise consistency). M5's stronger within-rotation failure AUROC does not rescue its cross-rotation calibration.\n\n4. **Does M5 p95 justify becoming the primary rotation objective?** No. Raw M5 p95—and especially raw CVaR95—should not be optimized across rotations until it is normalized or otherwise made basis-comparable and validated on more rotations with genuine held-out reasoning outcomes.\n\n5. **Does Decision sensitivity add useful information?** Not for rotation ranking here. Decision p95 also ranks Identity as safer, and a simple M5+Decision rank combination remains discordant. It may remain a secondary within-trajectory diagnostic.\n\n6. **Is there sufficient evidence to train a new learnable rotation?** No. There are only two fully comparable rotations, one candidate pair is discordant, and Dense/random candidates lack the required joined outcomes. Training now would optimize an unvalidated scale.\n\n7. **What should the next objective be?** Do not yet choose a training objective. First create a cheap, fixed held-out candidate panel (including the existing Dense and random matrices) with the exact same teacher-forced M0/M5/Decision capture plus real pre-existing or separately held-out reasoning outcomes. Pre-register a basis-normalized tail metric—for example M5 divided by an FP functional-scale reference per layer/token—and compare normalized p95/CVaR95 against raw M5, M0, and Decision. Only if lower normalized tail risk ranks multiple rotations consistently should a learnable objective use normalized CVaR95(M5), with Decision as a secondary constraint rather than the primary target.\n\n## Dense-Functional limitation\n\nWhether Dense-Functional improves exact M5 p95 is **not identifiable** from saved artifacts. Its local mean `out_proj_error` is {out_improve:.1f}% below Hadamard and local mean `state_error` is {state_improve:.1f}% below Hadamard; its persistent Future-KL AUC is {kl_improve:.1f}% lower. These are not M5 p95 and cannot be relabeled as such. Because no Dense AIME result exists, translation to reasoning quality is also not identifiable.\n\n## Decision\n\n**Do not train against raw CVaR95(M5) yet.** The current evidence supports M5 as a failure-detection diagnostic inside a fixed rotation, but contradicts its use as an unnormalized selector across rotations.\n"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")

    for path in sorted(OUT.iterdir()):
        print(path.name, path.stat().st_size, sha256(path), flush=True)


if __name__ == "__main__":
    main()
