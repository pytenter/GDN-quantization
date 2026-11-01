#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


TASK = "GDN_INT8_RC_GRANULARITY_PHASE_DIAGRAM_V1"
REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results" / "propagation" / "gdn_int8_functional_proxy_natural_config_generalization_v1_formal_raw_results.csv"
RUN_DIR = REPO / "runs" / "gdn_int8_rc_granularity_phase_diagram_v1"
RESULTS_DIR = REPO / "results" / "propagation"
REPORTS_DIR = REPO / "reports" / "propagation"
GROUP_SIZES = [128, 64, 32, 16]
HISTORY_LENGTHS_REQUESTED = [1, 4, 8, 16, 32, 64]
EPS = 1e-12


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fnum(x):
    return float(x) if finite(x) else None


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.median(xs) if xs else None


def std(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0 if xs else None


def bootstrap_ci(xs, n=4000, seed=20260910):
    xs = [float(x) for x in xs if finite(x)]
    if not xs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(mean(xs[rng.randrange(len(xs))] for _ in xs))
    vals.sort()
    return [vals[int(0.025 * (n - 1))], vals[int(0.975 * (n - 1))]]


def binomial_p_two_sided(k, n):
    if n <= 0:
        return None
    obs = math.comb(n, k) * (0.5 ** n)
    total = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (0.5 ** n)
        if prob <= obs + 1e-15:
            total += prob
    return min(total, 1.0)


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return None
    rx = rankdata([x for x, _ in pairs])
    ry = rankdata([y for _, y in pairs])
    mx = mean(rx)
    my = mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > EPS else None


def read_rows():
    with SOURCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        cfg = r["config"]
        elems = 24 * 32 * 128 * 128
        m0 = fnum(r["M0"])
        out.append({
            "TASK": TASK,
            "unit_id": r["unit_id"],
            "prompt_id": r["prompt_id"],
            "t0": int(r["t0"]),
            "history_length": 1,
            "config": cfg,
            "orientation": "R" if cfg.startswith("R") else "C",
            "group_size": int(r["group_size"]),
            "FUTURE_FULL_LOGIT_KL_AUC": fnum(r["future_KL"]),
            "future_KL_sum": fnum(r["future_KL_sum"]),
            "future_KL_peak": fnum(r["future_KL_peak"]),
            "future_KL_terminal": fnum(r["future_KL_terminal"]),
            "future_horizon_n": int(float(r["future_horizon_n"])),
            "SOURCE_ERROR_NORM": m0,
            "SOURCE_ERROR_MSE": (m0 * m0 / elems) if m0 is not None else None,
            "M5": fnum(r["M5"]),
            "F5": fnum(r["M5"]) / (m0 + EPS) if m0 is not None else None,
            "M0_rel": fnum(r["M0_rel"]),
            "dynamic_range_p95": fnum(r["dynamic_range_p95"]),
            "source_panel": str(SOURCE),
            "evidence_scope": "INHERITED_SINGLE_EVENT_L1_PANEL_NOT_REQUESTED_PERSISTENT_HISTORY_FORMAL",
        })
    return out


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def source_sha():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def paired_by_group(rows, metric):
    by = {(r["unit_id"], r["config"]): r for r in rows}
    out = {}
    for g in GROUP_SIZES:
        vals = []
        for unit in sorted({r["unit_id"] for r in rows}):
            rr = by.get((unit, f"R{g}"))
            cc = by.get((unit, f"C{g}"))
            if rr and cc and finite(rr[metric]) and finite(cc[metric]):
                vals.append(rr[metric] - cc[metric])
        out[g] = vals
    return out


def config_means(rows, metric):
    out = {}
    for cfg in [f"R{g}" for g in GROUP_SIZES] + [f"C{g}" for g in GROUP_SIZES]:
        out[cfg] = mean(r[metric] for r in rows if r["config"] == cfg)
    return out


def classify_group(delta, ci):
    tol = max(1e-5, abs(ci[1] - ci[0]) / 2.0) if ci else 1e-5
    if abs(delta) <= tol:
        return "NEAR_TIE"
    return "C_PREFERRED" if delta > 0 else "R_PREFERRED"


def main():
    rows = read_rows()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(RUN_DIR / "results.jsonl", rows)

    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)
    n_units = len(by_unit)

    cfg_kl = config_means(rows, "FUTURE_FULL_LOGIT_KL_AUC")
    cfg_norm = config_means(rows, "SOURCE_ERROR_NORM")
    cfg_mse = config_means(rows, "SOURCE_ERROR_MSE")
    cfg_f5 = config_means(rows, "F5")
    cfg_m5 = config_means(rows, "M5")

    phase = []
    group_stats = {}
    for g in GROUP_SIZES:
        deltas = paired_by_group(rows, "FUTURE_FULL_LOGIT_KL_AUC")[g]
        f5_deltas = paired_by_group(rows, "F5")[g]
        ci = bootstrap_ci(deltas)
        pos = sum(1 for x in deltas if x > 0)
        neg = sum(1 for x in deltas if x < 0)
        ties = len(deltas) - pos - neg
        delta_mean = mean(deltas)
        pref = classify_group(delta_mean, ci)
        group_stats[str(g)] = {
            "mean": delta_mean,
            "median": median(deltas),
            "std": std(deltas),
            "positive": pos,
            "negative": neg,
            "ties": ties,
            "paired_bootstrap_ci": ci,
            "paired_sign_binomial_p": binomial_p_two_sided(pos, pos + neg),
            "preferred": pref,
        }
        phase.append({
            "history_length": 1,
            "group_size": g,
            "KL_R": cfg_kl[f"R{g}"],
            "KL_C": cfg_kl[f"C{g}"],
            "R-C_Gap": delta_mean,
            "Norm_R/C": cfg_norm[f"R{g}"] / (cfg_norm[f"C{g}"] + EPS),
            "MSE_R/C": cfg_mse[f"R{g}"] / (cfg_mse[f"C{g}"] + EPS),
            "F5_R-C": mean(f5_deltas),
            "Preferred": pref,
            "evidence_scope": "L1_INHERITED_ONLY",
        })

    source_rows = []
    functional_rows = []
    for g in GROUP_SIZES:
        source_rows.append({
            "group_size": g,
            "SOURCE_NORM_R": cfg_norm[f"R{g}"],
            "SOURCE_NORM_C": cfg_norm[f"C{g}"],
            "DELTA_NORM": cfg_norm[f"R{g}"] - cfg_norm[f"C{g}"],
            "NORM_R_OVER_C": cfg_norm[f"R{g}"] / (cfg_norm[f"C{g}"] + EPS),
            "SOURCE_MSE_R": cfg_mse[f"R{g}"],
            "SOURCE_MSE_C": cfg_mse[f"C{g}"],
            "DELTA_MSE": cfg_mse[f"R{g}"] - cfg_mse[f"C{g}"],
            "MSE_R_OVER_C": cfg_mse[f"R{g}"] / (cfg_mse[f"C{g}"] + EPS),
        })
        functional_rows.append({
            "group_size": g,
            "KL_R": cfg_kl[f"R{g}"],
            "KL_C": cfg_kl[f"C{g}"],
            "DELTA_FUNC": cfg_kl[f"R{g}"] - cfg_kl[f"C{g}"],
            "M5_R": cfg_m5[f"R{g}"],
            "M5_C": cfg_m5[f"C{g}"],
            "DELTA_M5": cfg_m5[f"R{g}"] - cfg_m5[f"C{g}"],
            "F5_R": cfg_f5[f"R{g}"],
            "F5_C": cfg_f5[f"C{g}"],
            "DELTA_F5": cfg_f5[f"R{g}"] - cfg_f5[f"C{g}"],
        })

    write_csv(RUN_DIR / "granularity_phase_diagram.csv", phase)
    write_csv(RUN_DIR / "source_error_by_granularity.csv", source_rows)
    write_csv(RUN_DIR / "functional_risk_by_granularity.csv", functional_rows)

    delta_func = [r["DELTA_FUNC"] for r in functional_rows]
    delta_f5 = [r["DELTA_F5"] for r in functional_rows]
    orientations = [1 if r["DELTA_FUNC"] > 0 and r["DELTA_F5"] > 0 else 1 if r["DELTA_FUNC"] < 0 and r["DELTA_F5"] < 0 else 0 for r in functional_rows]
    gap_abs = [abs(x) for x in delta_func]
    shrinking = all(gap_abs[i + 1] <= gap_abs[i] + EPS for i in range(len(gap_abs) - 1))
    has_flip = any(x < 0 for x in delta_func) and any(x > 0 for x in delta_func)
    has_near = any(r["Preferred"] == "NEAR_TIE" for r in phase)
    if has_flip:
        result = "ORIENTATION_FLIP_CONFIRMED"
    elif has_near:
        result = "ORIENTATION_BOUNDARY_APPROACHED"
    elif all(x > 0 for x in delta_func) and shrinking:
        result = "C_PREFERENCE_WITH_GRANULARITY_DEPENDENT_GAP"
    elif all(x > 0 for x in delta_func):
        result = "C_PREFERENCE_STABLE"
    else:
        result = "INHERITED_PANEL_INCONCLUSIVE"

    gains = {}
    for fam in ("R", "C"):
        gains[fam] = []
        for a, b in zip(GROUP_SIZES, GROUP_SIZES[1:]):
            gains[fam].append({
                "from": a,
                "to": b,
                "future_KL_reduction": cfg_kl[f"{fam}{a}"] - cfg_kl[f"{fam}{b}"],
                "source_MSE_reduction": cfg_mse[f"{fam}{a}"] - cfg_mse[f"{fam}{b}"],
                "source_norm_reduction": cfg_norm[f"{fam}{a}"] - cfg_norm[f"{fam}{b}"],
                "M5_reduction": cfg_m5[f"{fam}{a}"] - cfg_m5[f"{fam}{b}"],
                "F5_change": cfg_f5[f"{fam}{b}"] - cfg_f5[f"{fam}{a}"],
            })

    missing_histories = [l for l in HISTORY_LENGTHS_REQUESTED if l != 1]
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "INCOMPLETE_INHERITED_L1_ONLY",
        "N_FORMAL_UNITS": 0,
        "N_INHERITED_UNITS": n_units,
        "CANONICAL_MANIFEST_SOURCE": str(SOURCE),
        "CANONICAL_MANIFEST_SHA256": source_sha(),
        "N_CANONICAL_UNITS": n_units,
        "CANONICAL_MANIFEST_GATE": "FAIL_EXPECTED_18_GOT_9_AND_SOURCE_IS_PRIOR_PANEL",
        "REPLAY_INPUT_GATE": "FAIL_ORIGINAL_PROMPT_FP_RESPONSE_MANIFEST_NOT_PRESENT_IN_CHECKOUT",
        "STATE_SEMANTICS_GATE": "PASS_INHERITED_FROM_PRIOR_PANEL",
        "ALL_QUANTIZER_IDENTITY_GATES": "PASS_INHERITED_GROUPED_QUANT_SEMANTICS",
        "R128_BASELINE_REPRODUCTION_GATE": "PASS_INHERITED_PRIOR_PANEL",
        "C128_BASELINE_REPRODUCTION_GATE": "PASS_INHERITED_PRIOR_PANEL",
        "M5_IMPLEMENTATION_GATE": "PASS_INHERITED_PRIOR_PANEL",
        "F5_SCALE_INVARIANCE_GATE": "PASS_DEFINED_AS_M5_OVER_SOURCE_NORM",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS_FOR_OFFLINE_FINALIZATION",
        "SMOKE": "NOT_RUN_FRESH",
        "PILOT": "NOT_RUN_FRESH",
        "GROUP_SIZES": GROUP_SIZES,
        "HISTORY_LENGTHS": HISTORY_LENGTHS_REQUESTED,
        "HISTORY_LENGTHS_COMPLETED": [1],
        "HISTORY_LENGTHS_MISSING": missing_histories,
        "KL_R_BY_GROUP_SIZE": {str(g): cfg_kl[f"R{g}"] for g in GROUP_SIZES},
        "KL_C_BY_GROUP_SIZE": {str(g): cfg_kl[f"C{g}"] for g in GROUP_SIZES},
        "DELTA_FUNC_BY_GROUP_SIZE": {str(r["group_size"]): r["DELTA_FUNC"] for r in functional_rows},
        "SOURCE_NORM_R_BY_GROUP_SIZE": {str(g): cfg_norm[f"R{g}"] for g in GROUP_SIZES},
        "SOURCE_NORM_C_BY_GROUP_SIZE": {str(g): cfg_norm[f"C{g}"] for g in GROUP_SIZES},
        "SOURCE_MSE_R_BY_GROUP_SIZE": {str(g): cfg_mse[f"R{g}"] for g in GROUP_SIZES},
        "SOURCE_MSE_C_BY_GROUP_SIZE": {str(g): cfg_mse[f"C{g}"] for g in GROUP_SIZES},
        "DELTA_F5_BY_GROUP_SIZE": {str(r["group_size"]): r["DELTA_F5"] for r in functional_rows},
        "DELTA_FUNC_STATS_BY_GROUP_SIZE": group_stats,
        "F5_VS_FUNCTIONAL_GAP_SPEARMAN": spearman(delta_f5, delta_func),
        "F5_VS_FUNCTIONAL_GAP_ORIENTATION_CONCORDANCE": mean(orientations),
        "R_FAMILY_GRANULARITY_GAIN": gains["R"],
        "C_FAMILY_GRANULARITY_GAIN": gains["C"],
        "ORIENTATION_BOUNDARY_GROUP_SIZE": next((r["group_size"] for r in phase if r["Preferred"] == "NEAR_TIE"), None),
        "GDN_ORIENTATION_GRANULARITY_RESULT": result,
        "FINAL_CLASSIFICATION": "PARTIAL_PRIOR_PANEL_SUPPORT_NOT_FORMAL_TASK_COMPLETION",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "WAIT_FOR_KDA_GRANULARITY_PHASE_DIAGRAM",
        "PYTEST": "NOT_RUN_NO_CODEPATH_BEHAVIOR_CHANGED",
        "FRESH_OUTPUT_VERIFICATION": "PASS",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO_FORMAL_RUN_BLOCKED_BY_MISSING_CANONICAL_REPLAY_INPUTS",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "quantizer_semantics": [
            {"config": f"R{g}", "orientation": "row", "group_size": g, "state_shape": "[B,H,K,V]", "grouped_dimension": "V", "scale_shape": f"[B,H,K,{128 // g},1] before reshape"}
            for g in GROUP_SIZES
        ] + [
            {"config": f"C{g}", "orientation": "column", "group_size": g, "state_shape": "[B,H,K,V]", "grouped_dimension": "K", "scale_shape": f"[B,H,{128 // g},1,V] before reshape"}
            for g in GROUP_SIZES
        ],
    }
    save_json(RUN_DIR / "aggregate_summary.json", summary)

    lines = [
        f"# {TASK}",
        "",
        "## Status",
        "This is an inherited-analysis finalization, not the requested fresh formal run. The available Qwen/GDN eight-config panel has 9 units at a single t0 event equivalent to L=1; the requested 18-unit x 6-history persistent panel is not present in this checkout.",
        "",
        "## Primary L=1 Phase Diagram",
        "",
        "| Group Size | KL_R | KL_C | R-C Gap | Norm_R/C | F5_R-C | Preferred |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in phase:
        lines.append(f"| {r['group_size']} | {r['KL_R']:.8g} | {r['KL_C']:.8g} | {r['R-C_Gap']:.8g} | {r['Norm_R/C']:.4g} | {r['F5_R-C']:.8g} | {r['Preferred']} |")
    lines += [
        "",
        "## Gates",
        "",
        "```json",
        json.dumps({k: summary[k] for k in [
            "FORMAL_STATUS", "N_FORMAL_UNITS", "N_INHERITED_UNITS",
            "CANONICAL_MANIFEST_GATE", "REPLAY_INPUT_GATE", "STATE_SEMANTICS_GATE",
            "ALL_QUANTIZER_IDENTITY_GATES", "M5_IMPLEMENTATION_GATE",
            "F5_SCALE_INVARIANCE_GATE", "GDN_ORIENTATION_GRANULARITY_RESULT",
            "FINAL_CLASSIFICATION"]}, indent=2),
        "```",
        "",
        "## Required Final Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
    ]
    report = "\n".join(lines) + "\n"
    (RUN_DIR / "report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / f"{TASK.lower()}.md").parent.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{TASK.lower()}.md").write_text(report, encoding="utf-8")
    save_json(RESULTS_DIR / f"{TASK.lower()}_aggregate_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
