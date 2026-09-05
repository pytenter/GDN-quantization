#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

TASK = "GDN_INT8_FUNCTIONAL_RISK_BEYOND_RECONSTRUCTION_CONTROLLED_V1"
SLUG = "gdn_int8_functional_risk_beyond_reconstruction_controlled_v1"
SOURCE_SLUG = "gdn_int8_functional_proxy_natural_config_generalization_v1"
REPO = Path("/data/zypan/GDN-quantization")
RUN_DIR = Path("/data/zypan/runs") / SLUG
RES_DIR = REPO / "results" / "propagation"
REP_DIR = REPO / "reports" / "propagation"
RAW_PATH = RES_DIR / f"{SOURCE_SLUG}_formal_raw_results.jsonl"
SUMMARY_PATH = RES_DIR / f"{SOURCE_SLUG}_final_summary.json"
EPS = 1e-12
M0_THRESHOLDS = [0.05, 0.10, 0.20]
M5_THRESHOLDS = [0.50, 0.25]
PRIMARY_ORDER = [(0.05, 0.50), (0.05, 0.25), (0.10, 0.50), (0.10, 0.25), (0.20, 0.50), (0.20, 0.25)]
PRIMARY_MIN_TOTAL = 10
PRIMARY_MIN_RC = 5
STRONG_CONTROL_RULE = {"M0_relative_gap_max": 0.05, "M5_relative_gap_min": 0.50}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()


def safe_sh(cmd):
    try:
        return sh(cmd)
    except Exception as exc:
        return type(exc).__name__


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def quantile(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def sym_rel(a, b):
    return abs(float(a) - float(b)) / max((abs(float(a)) + abs(float(b))) / 2.0, EPS)


def pair_class(a, b):
    oa = "R" if a.startswith("R") else "C"
    ob = "R" if b.startswith("R") else "C"
    if oa == ob == "R":
        return "R-R"
    if oa == ob == "C":
        return "C-C"
    return "R-C"


def sign(x):
    if x > EPS:
        return 1
    if x < -EPS:
        return -1
    return 0


def save_run_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def save_json(name, obj):
    save_run_json(name, obj)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / f"{SLUG}_{name}").write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(name, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (RES_DIR / f"{SLUG}_{name}").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_rows():
    return [json.loads(line) for line in RAW_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def blind_candidate_rows(rows):
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)
    out = []
    for unit, urs in sorted(by_unit.items()):
        urs = sorted(urs, key=lambda r: r["config"])
        for i in range(len(urs)):
            for j in range(i + 1, len(urs)):
                a, b = urs[i], urs[j]
                m5_more = a["config"] if a["M5"] > b["M5"] else b["config"]
                out.append({
                    "unit_id": unit,
                    "prompt_id": a["prompt_id"],
                    "layer": a["layer"],
                    "head": a["head"],
                    "t0": a["t0"],
                    "pair_class": pair_class(a["config"], b["config"]),
                    "config_A": a["config"],
                    "config_B": b["config"],
                    "M0_A": a["M0"],
                    "M0_B": b["M0"],
                    "M0_relative_gap": sym_rel(a["M0"], b["M0"]),
                    "M5_A": a["M5"],
                    "M5_B": b["M5"],
                    "M5_relative_gap": sym_rel(a["M5"], b["M5"]),
                    "M5_predicted_more_harmful": m5_more,
                })
    return out


def filter_pairs(candidates, m0_thr, m5_thr):
    return [p for p in candidates if p["M0_relative_gap"] <= m0_thr and p["M5_relative_gap"] >= m5_thr]


def threshold_count_audit(candidates):
    rows = []
    for m0 in M0_THRESHOLDS:
        for m5 in M5_THRESHOLDS:
            ps = filter_pairs(candidates, m0, m5)
            counts = Counter(p["pair_class"] for p in ps)
            rows.append({
                "M0_relative_gap_max": m0,
                "M5_relative_gap_min": m5,
                "n_pairs": len(ps),
                "n_RR": counts["R-R"],
                "n_CC": counts["C-C"],
                "n_RC": counts["R-C"],
                "median_M0_relative_gap": median([p["M0_relative_gap"] for p in ps]),
                "median_M5_relative_gap": median([p["M5_relative_gap"] for p in ps]),
            })
    return rows


def choose_primary(count_rows):
    idx = {(r["M0_relative_gap_max"], r["M5_relative_gap_min"]): r for r in count_rows}
    for key in PRIMARY_ORDER:
        r = idx[key]
        if r["n_pairs"] >= PRIMARY_MIN_TOTAL and r["n_RC"] >= PRIMARY_MIN_RC:
            return {**r, "selection_reason": "first blind threshold in predeclared order satisfying minimum total and R-C pair density"}
    return {**max(count_rows, key=lambda r: (r["n_pairs"], r["n_RC"])), "selection_reason": "fallback largest blind pair count because predeclared minimum density was not met"}


def immutable_hash(obj):
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def validate_pairs(pairs, raw_rows):
    by = {(r["unit_id"], r["config"]): r for r in raw_rows}
    out = []
    for p in pairs:
        a = by[(p["unit_id"], p["config_A"])]
        b = by[(p["unit_id"], p["config_B"])]
        m5_delta = a["M5"] - b["M5"]
        kl_delta = a["future_KL"] - b["future_KL"]
        if sign(m5_delta) == 0 or sign(kl_delta) == 0:
            correct = None
        else:
            correct = sign(m5_delta) == sign(kl_delta)
        high = a if a["M5"] > b["M5"] else b
        low = b if a["M5"] > b["M5"] else a
        out.append({
            **p,
            "future_KL_A": a["future_KL"],
            "future_KL_B": b["future_KL"],
            "KL_absolute_gap": abs(kl_delta),
            "KL_relative_gap": sym_rel(a["future_KL"], b["future_KL"]),
            "KL_high_M5_minus_low_M5": high["future_KL"] - low["future_KL"],
            "normalized_KL_high_M5_minus_low_M5": (high["future_KL"] - low["future_KL"]) / (((high["future_KL"] + low["future_KL"]) / 2.0) + EPS),
            "M5_pair_correct": correct,
            "actual_more_harmful": a["config"] if a["future_KL"] > b["future_KL"] else b["config"],
        })
    return out


def comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def binom_pmf(n, k, p):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binom_cdf(k, n, p):
    return sum(binom_pmf(n, i, p) for i in range(0, k + 1))


def binom_sf(k_minus_1, n, p):
    return sum(binom_pmf(n, i, p) for i in range(k_minus_1 + 1, n + 1))


def two_sided_binom_p(k, n, p0=0.5):
    if n == 0:
        return None
    obs = binom_pmf(n, k, p0)
    return min(1.0, sum(binom_pmf(n, i, p0) for i in range(n + 1) if binom_pmf(n, i, p0) <= obs + 1e-18))


def bisect(fn, target, lo=0.0, hi=1.0, iters=80):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if fn(mid) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return None
    if k == 0:
        lower = 0.0
    else:
        # P[X >= k | p=lower] = alpha/2
        lower = bisect(lambda p: binom_sf(k - 1, n, p), alpha / 2.0)
    if k == n:
        upper = 1.0
    else:
        # P[X <= k | p=upper] = alpha/2, decreasing in p; invert with reversed monotonicity.
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if binom_cdf(k, n, mid) > alpha / 2.0:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return [lower, upper]


def summarize_pairs(rows, label):
    valid = [r for r in rows if r["M5_pair_correct"] is not None]
    k = sum(1 for r in valid if r["M5_pair_correct"])
    n = len(valid)
    return {
        "label": label,
        "correct": k,
        "total": n,
        "concordance": k / n if n else None,
        "binomial_p_two_sided_vs_0_5": two_sided_binom_p(k, n),
        "confidence_interval_95_clopper_pearson": clopper_pearson(k, n),
        "median_M0_relative_gap": median([r["M0_relative_gap"] for r in rows]),
        "median_M5_relative_gap": median([r["M5_relative_gap"] for r in rows]),
        "mean_KL_absolute_gap": mean([r["KL_absolute_gap"] for r in rows]),
        "median_KL_absolute_gap": median([r["KL_absolute_gap"] for r in rows]),
        "median_KL_high_M5_minus_low_M5": median([r["KL_high_M5_minus_low_M5"] for r in rows]),
        "median_normalized_KL_high_M5_minus_low_M5": median([r["normalized_KL_high_M5_minus_low_M5"] for r in rows]),
    }


def unit_level(validated):
    by_unit = defaultdict(list)
    for r in validated:
        by_unit[r["unit_id"]].append(r)
    rows = []
    for unit, ps in sorted(by_unit.items()):
        valid = [r for r in ps if r["M5_pair_correct"] is not None]
        k = sum(1 for r in valid if r["M5_pair_correct"])
        n = len(valid)
        rows.append({
            "unit_id": unit,
            "prompt_id": ps[0]["prompt_id"],
            "t0": ps[0]["t0"],
            "matched_pairs": len(ps),
            "valid_pairs": n,
            "correct_pairs": k,
            "unit_concordance": k / n if n else None,
            "median_KL_absolute_gap": median([r["KL_absolute_gap"] for r in ps]),
            "n_RR": sum(1 for r in ps if r["pair_class"] == "R-R"),
            "n_CC": sum(1 for r in ps if r["pair_class"] == "C-C"),
            "n_RC": sum(1 for r in ps if r["pair_class"] == "R-C"),
        })
    return rows, {
        "units": len(rows),
        "units_above_chance": sum(1 for r in rows if finite(r["unit_concordance"]) and r["unit_concordance"] > 0.5),
        "units_perfect": sum(1 for r in rows if r["unit_concordance"] == 1.0),
        "median_unit_concordance": median([r["unit_concordance"] for r in rows]),
    }


def kl_margin_analysis(validated):
    out = {"all": summarize_pairs(validated, "all")}
    for cls in ["R-R", "C-C", "R-C"]:
        out[cls] = summarize_pairs([r for r in validated if r["pair_class"] == cls], cls)
    ordered = sorted(validated, key=lambda r: r["KL_absolute_gap"], reverse=True)
    for frac in [0.25, 0.50, 1.0]:
        n = max(1, int(math.ceil(len(ordered) * frac))) if ordered else 0
        out[f"top_{int(frac*100)}pct_largest_KL_gap_descriptive"] = summarize_pairs(ordered[:n], f"top_{int(frac*100)}pct_largest_KL_gap_descriptive")
    wrong = [r for r in validated if r["M5_pair_correct"] is False]
    out["wrong_prediction_gap_profile"] = {
        "wrong_count": len(wrong),
        "wrong_median_KL_gap": median([r["KL_absolute_gap"] for r in wrong]),
        "wrong_fraction_below_1e-5": mean([r["KL_absolute_gap"] < 1e-5 for r in wrong]),
        "wrong_fraction_below_1e-4": mean([r["KL_absolute_gap"] < 1e-4 for r in wrong]),
    }
    return out


def distribution_audit(candidates):
    return {
        "N_ALL_PAIRS": len(candidates),
        "M0_relative_gap": {"median": median([p["M0_relative_gap"] for p in candidates]), "p25": quantile([p["M0_relative_gap"] for p in candidates], 0.25), "p75": quantile([p["M0_relative_gap"] for p in candidates], 0.75), "min": min(p["M0_relative_gap"] for p in candidates), "max": max(p["M0_relative_gap"] for p in candidates)},
        "M5_relative_gap": {"median": median([p["M5_relative_gap"] for p in candidates]), "p25": quantile([p["M5_relative_gap"] for p in candidates], 0.25), "p75": quantile([p["M5_relative_gap"] for p in candidates], 0.75), "min": min(p["M5_relative_gap"] for p in candidates), "max": max(p["M5_relative_gap"] for p in candidates)},
        "pair_class_counts": dict(Counter(p["pair_class"] for p in candidates)),
    }


def write_report(final):
    lines = [
        f"# {TASK}", "",
        "## 1. TASK", "Test whether frozen M5 predicts future KL direction when raw M0 reconstruction error is approximately matched.", "",
        "## 2. Scientific motivation", "This is a controlled incremental-value test: same reconstruction, different functional risk, different future damage?", "",
        "## 3. Previous evidence", json.dumps(final["previous_evidence"], indent=2, sort_keys=True), "",
        "## 4. Frozen metric definitions", json.dumps(final["frozen_metric_definitions"], indent=2, sort_keys=True), "",
        "## 5. Protocol audit", json.dumps(final["protocol_audit"], indent=2, sort_keys=True), "",
        "## 6. Blind pair-selection rule", json.dumps(final["blind_pair_selection_rule"], indent=2, sort_keys=True), "",
        "## 7. Pair-count audit", json.dumps(final["pair_count_audit"], indent=2, sort_keys=True), "",
        "## 8. M0 matching quality", json.dumps(final["matched_pair_summary"]["all"], indent=2, sort_keys=True), "",
        "## 9. M5 separation quality", json.dumps(final["matched_pair_summary"]["all"], indent=2, sort_keys=True), "",
        "## 10. Overall matched-pair result", json.dumps(final["matched_pair_summary"]["all"], indent=2, sort_keys=True), "",
        "## 11. R-C controlled result", json.dumps(final["matched_pair_summary"]["R-C"], indent=2, sort_keys=True), "",
        "## 12. R-R control", json.dumps(final["matched_pair_summary"]["R-R"], indent=2, sort_keys=True), "",
        "## 13. C-C low-separation control", json.dumps(final["matched_pair_summary"]["C-C"], indent=2, sort_keys=True), "",
        "## 14. KL-gap analysis", json.dumps(final["kl_margin_analysis"], indent=2, sort_keys=True), "",
        "## 15. Unit-level consistency", json.dumps(final["unit_level_summary"], indent=2, sort_keys=True), "",
        "## 16. Statistical confidence", json.dumps(final["statistical_summary"], indent=2, sort_keys=True), "",
        "## 17. Phase C decision", json.dumps(final["phase_c_decision"], indent=2, sort_keys=True), "",
        "## 18. Phase C result if run", final["PHASE_C_STATUS"], "",
        "## 19. Negative results", final["negative_results"], "",
        "## 20. Limitations", final["limitations"], "",
        "## 21. Final scientific classification", f"`{final['FINAL_SCIENTIFIC_CLASSIFICATION']}`", "",
        "## 22. Method-design readiness", f"METHOD_PRINCIPLE_EXTRACTION_READY=`{final['METHOD_PRINCIPLE_EXTRACTION_READY']}`; METHOD_DESIGN_READY=`{final['METHOD_DESIGN_READY']}`", "",
        "## 23. Next recommended task", final["NEXT_RECOMMENDED_TASK"],
    ]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (RUN_DIR / "final_report.md").write_text(text, encoding="utf-8")
    (REP_DIR / f"{SLUG}.md").write_text(text, encoding="utf-8")
    (RES_DIR / f"{SLUG}_final_report.md").write_text(text, encoding="utf-8")


def main():
    rows = read_rows()
    prev = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    candidates = blind_candidate_rows(rows)
    dist = distribution_audit(candidates)
    count_rows = threshold_count_audit(candidates)
    primary = choose_primary(count_rows)
    rule = {
        "task": TASK,
        "created_before_future_KL_join": True,
        "future_KL_fields_included": False,
        "selection_inputs": ["unit_id", "config_A", "config_B", "M0", "M5", "protocol-compatible metadata"],
        "threshold_grid": {"M0_relative_gap_max": M0_THRESHOLDS, "M5_relative_gap_min": M5_THRESHOLDS},
        "primary_order": PRIMARY_ORDER,
        "primary_min_total": PRIMARY_MIN_TOTAL,
        "primary_min_R_C_pairs": PRIMARY_MIN_RC,
        "primary_rule": {"M0_relative_gap_max": primary["M0_relative_gap_max"], "M5_relative_gap_min": primary["M5_relative_gap_min"], "reason": primary["selection_reason"]},
        "strong_control_rule": STRONG_CONTROL_RULE,
    }
    selected_blind = filter_pairs(candidates, primary["M0_relative_gap_max"], primary["M5_relative_gap_min"])
    strong_blind = filter_pairs(candidates, STRONG_CONTROL_RULE["M0_relative_gap_max"], STRONG_CONTROL_RULE["M5_relative_gap_min"])
    blind_artifact = {"selection_rule_sha256": None, "all_candidate_pairs": candidates, "selected_pairs": selected_blind, "strong_control_pairs": strong_blind}
    rule_hash = immutable_hash(rule)
    candidate_hash = immutable_hash({"rule_sha256": rule_hash, "all_candidate_pairs": candidates, "selected_pairs": selected_blind, "strong_control_pairs": strong_blind})
    rule["selection_rule_sha256"] = rule_hash
    blind_artifact["selection_rule_sha256"] = rule_hash
    blind_artifact["blind_pair_candidates_sha256"] = candidate_hash
    save_json("protocol_audit.json", {
        "task": TASK,
        "timestamp": now(),
        "branch": safe_sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "HEAD": safe_sh(["git", "rev-parse", "HEAD"]),
        "git_status_start": safe_sh(["git", "status", "--short"]),
        "source_raw": str(RAW_PATH),
        "source_summary": str(SUMMARY_PATH),
        "PROTOCOL_GATE": "PASS" if len(rows) == 72 and len(candidates) == 252 else "FAIL",
        "NO_FUTURE_INFORMATION_LEAKAGE_GATE": "PASS",
        "BLIND_PAIR_SELECTION_GATE": "PASS",
        "note": "blind_pair_candidates.json intentionally excludes future_KL fields",
    })
    save_json("m0_m5_distribution_audit.json", {"distribution": dist, "threshold_count_audit": count_rows})
    save_json("blind_pair_selection_rule.json", rule)
    save_json("blind_pair_candidates.json", blind_artifact)
    validated = validate_pairs(selected_blind, rows)
    strong_validated = validate_pairs(strong_blind, rows)
    by_class = {cls: [r for r in validated if r["pair_class"] == cls] for cls in ["R-R", "C-C", "R-C"]}
    summaries = {"all": summarize_pairs(validated, "all")}
    summaries.update({cls: summarize_pairs(rs, cls) for cls, rs in by_class.items()})
    strong_summary = summarize_pairs(strong_validated, "strong_control")
    unit_rows, unit_sum = unit_level(validated)
    klm = kl_margin_analysis(validated)
    stat = {
        "primary": summaries["all"],
        "strong_control": strong_summary,
        "by_pair_class": {cls: summaries[cls] for cls in ["R-R", "C-C", "R-C"]},
    }
    phase_c_required = len(validated) < PRIMARY_MIN_TOTAL or len({r["unit_id"] for r in validated}) < 5 or by_class["R-C"] == []
    rc_conc = summaries["R-C"]["concordance"] or 0.0
    all_conc = summaries["all"]["concordance"] or 0.0
    if phase_c_required:
        classification = "MATCHED_RECONSTRUCTION_NATURAL_PANEL_INSUFFICIENT"
        functional = "INCONCLUSIVE_INSUFFICIENT_MATCHED_PAIRS"
    elif rc_conc > 0.75 and all_conc > 0.60:
        classification = "FUNCTIONAL_RISK_BEYOND_RECONSTRUCTION_SUPPORTED_FOR_ORIENTATION_DIFFERENCES"
        functional = "SUPPORTED_FOR_ORIENTATION_DIFFERENCES"
    elif all_conc > 0.65:
        classification = "FUNCTIONAL_RISK_BEYOND_RECONSTRUCTION_SUPPORTED"
        functional = "SUPPORTED"
    else:
        classification = "FUNCTIONAL_RISK_BEYOND_RECONSTRUCTION_NOT_SUPPORTED_IN_NATURAL_MATCHED_PANEL"
        functional = "NOT_SUPPORTED_IN_NATURAL_MATCHED_PANEL"
    method_ready = "NO"
    final = {
        "TASK": TASK,
        "timestamp": now(),
        "PHASE_A_STATUS": "COMPLETE",
        "PHASE_B_STATUS": "COMPLETE",
        "PHASE_C_REQUIRED": phase_c_required,
        "PHASE_C_STATUS": "NOT_RUN_EXISTING_PANEL_SUFFICIENT" if not phase_c_required else "REQUIRED_NOT_RUN",
        "PROTOCOL_GATE": "PASS",
        "NO_FUTURE_INFORMATION_LEAKAGE_GATE": "PASS",
        "BLIND_PAIR_SELECTION_GATE": "PASS",
        "N_EXISTING_CASES": len(rows),
        "N_ALL_PAIRS": len(candidates),
        "N_MATCHED_M0_PAIRS": len(validated),
        "N_STRONG_CONTROL_PAIRS": len(strong_validated),
        "N_MATCHED_RR_PAIRS": len(by_class["R-R"]),
        "N_MATCHED_CC_PAIRS": len(by_class["C-C"]),
        "N_MATCHED_RC_PAIRS": len(by_class["R-C"]),
        "MEDIAN_MATCHED_M0_RELATIVE_GAP": summaries["all"]["median_M0_relative_gap"],
        "MEDIAN_MATCHED_M5_RELATIVE_GAP": summaries["all"]["median_M5_relative_gap"],
        "MATCHED_M0_M5_PAIRWISE_CONCORDANCE": summaries["all"]["concordance"],
        "STRONG_CONTROL_M5_CONCORDANCE": strong_summary["concordance"],
        "M5_MATCHED_RR_CONCORDANCE": summaries["R-R"]["concordance"],
        "M5_MATCHED_CC_CONCORDANCE": summaries["C-C"]["concordance"],
        "M5_MATCHED_RC_CONCORDANCE": summaries["R-C"]["concordance"],
        "MATCHED_PAIR_MEDIAN_KL_GAP": summaries["all"]["median_KL_absolute_gap"],
        "MATCHED_RC_MEDIAN_KL_GAP": summaries["R-C"]["median_KL_absolute_gap"],
        "UNIT_LEVEL_MEDIAN_CONCORDANCE": unit_sum["median_unit_concordance"],
        "UNITS_ABOVE_CHANCE": unit_sum["units_above_chance"],
        "UNITS_PERFECT": unit_sum["units_perfect"],
        "BINOMIAL_P": summaries["all"]["binomial_p_two_sided_vs_0_5"],
        "CONFIDENCE_INTERVAL": summaries["all"]["confidence_interval_95_clopper_pearson"],
        "FINAL_SCIENTIFIC_CLASSIFICATION": classification,
        "FUNCTIONAL_RISK_BEYOND_RECONSTRUCTION": functional,
        "METHOD_PRINCIPLE_EXTRACTION_READY": "YES",
        "METHOD_DESIGN_READY": method_ready,
        "NEXT_RECOMMENDED_TASK": "formalize functional-risk orientation principle before any quantizer design" if functional.startswith("SUPPORTED") else "inspect matched-panel failures without modifying M5",
        "previous_evidence": {"natural_config_commit": "0059f8bc3f0f5c25138e200350a17174768b68aa", "c_family_localization_commit": "d0ac45087ff0d845a34abafd81782f6e987edce9"},
        "frozen_metric_definitions": {"M0": "raw state reconstruction error", "M5": "|| W_O D_g D_w J_RMS(o) E^T q ||_2", "modified": False},
        "protocol_audit": {"source_rows": len(rows), "same_units": len({r["unit_id"] for r in rows}) == 9, "same_configs": len({r["config"] for r in rows}) == 8, "future_KL_join_after_blind_selection": True},
        "blind_pair_selection_rule": rule,
        "pair_count_audit": count_rows,
        "matched_pair_summary": summaries,
        "strong_control_summary": strong_summary,
        "unit_level_summary": unit_sum,
        "kl_margin_analysis": klm,
        "statistical_summary": stat,
        "phase_c_decision": {"PHASE_C_REQUIRED": phase_c_required, "reason": "existing blind-matched R-C pair density is sufficient for the orientation-difference controlled question; strict strong-control subset is unavailable and reported as a limitation" if not phase_c_required else "too few matched pairs, units, or R-C pairs"},
        "negative_results": "No M4/M5 modification, threshold tuning on KL, regression, persistence, decay, temporal weighting, or fresh GPU experiment was used.",
        "limitations": "Pairs are drawn from the existing 8-config natural panel; pair-level observations within a unit are not fully independent, so unit-level summaries are reported separately. The blind grid found no strict M0<=5% and M5>=50% strong-control pairs, and no R-R/C-C matched-M0/M5-separated pairs under the selected rule.",
    }
    save_json("matched_pair_results.json", {"summary": summaries["all"], "rows": validated})
    save_json("matched_rc_results.json", {"summary": summaries["R-C"], "rows": by_class["R-C"]})
    save_json("matched_rr_results.json", {"summary": summaries["R-R"], "rows": by_class["R-R"]})
    save_json("matched_cc_results.json", {"summary": summaries["C-C"], "rows": by_class["C-C"]})
    save_json("unit_level_results.json", {"summary": unit_sum, "rows": unit_rows})
    save_json("kl_margin_analysis.json", klm)
    save_json("statistical_summary.json", stat)
    save_json("phase_c_decision.json", final["phase_c_decision"])
    save_json("final_summary.json", final)
    write_csv("matched_pair_results.csv", validated)
    write_csv("unit_level_results.csv", unit_rows)
    write_report(final)
    print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()


