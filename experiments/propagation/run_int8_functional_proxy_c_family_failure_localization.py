#!/usr/bin/env python3
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TASK = "GDN_INT8_FUNCTIONAL_PROXY_C_FAMILY_FAILURE_LOCALIZATION_V1"
SLUG = "gdn_int8_functional_proxy_c_family_failure_localization_v1"
REPO = Path("/data/zypan/GDN-quantization")
RUN_DIR = Path("/data/zypan/runs") / SLUG
RES_DIR = REPO / "results" / "propagation"
REP_DIR = REPO / "reports" / "propagation"
SRC_SLUG = "gdn_int8_functional_proxy_natural_config_generalization_v1"
RAW = RES_DIR / f"{SRC_SLUG}_formal_raw_results.jsonl"
SUMMARY = RES_DIR / f"{SRC_SLUG}_final_summary.json"
METRICS = ["M0", "M1", "M2", "M3", "M4", "M5"]
R_CONFIGS = ["R128", "R64", "R32", "R16"]
C_CONFIGS = ["C128", "C64", "C32", "C16"]
ALL_CONFIGS = R_CONFIGS + C_CONFIGS
EPS = 1e-12


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


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs))
    return num / den if den > EPS else None


def ranks(vals):
    order = sorted((v, i) for i, v in enumerate(vals))
    out = [0.0] * len(vals)
    j = 0
    while j < len(order):
        k = j + 1
        while k < len(order) and order[k][0] == order[j][0]:
            k += 1
        r = (j + k - 1) / 2.0 + 1.0
        for _, idx in order[j:k]:
            out[idx] = r
        j = k
    return out


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def kendall_tau(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return None
    conc = disc = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            if abs(dx) <= EPS or abs(dy) <= EPS:
                continue
            if dx * dy > 0:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else None


def pair_class(a, b):
    if a.startswith("R") and b.startswith("R"):
        return "R-R"
    if a.startswith("C") and b.startswith("C"):
        return "C-C"
    return "R-C"


def pairwise_concordance(rows, metric, configs=None):
    if configs is not None:
        rows = [r for r in rows if r["config"] in configs]
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)
    ok = total = 0
    for unit, urs in by_unit.items():
        for i in range(len(urs)):
            for j in range(i + 1, len(urs)):
                a, b = urs[i], urs[j]
                dp = a[metric] - b[metric]
                dk = a["future_KL"] - b["future_KL"]
                if abs(dp) <= EPS or abs(dk) <= EPS:
                    continue
                ok += int(dp * dk > 0)
                total += 1
    return {"correct_pairs": ok, "total_pairs": total, "pairwise_concordance": ok / total if total else None}


def ranking(rows, key):
    return [r["config"] for r in sorted(rows, key=lambda r: r[key])]


def family_name(configs):
    return "R" if configs == R_CONFIGS else "C"


def separability(rows, configs):
    fam = family_name(configs)
    out = []
    by_unit = defaultdict(list)
    for r in rows:
        if r["config"] in configs:
            by_unit[r["unit_id"]].append(r)
    all_pair_gaps = []
    for unit, urs in sorted(by_unit.items()):
        urs = sorted(urs, key=lambda r: r["future_KL"])
        vals = [r["future_KL"] for r in urs]
        gaps = []
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                gaps.append(abs(vals[j] - vals[i]))
        all_pair_gaps.extend(gaps)
        best_gap = vals[1] - vals[0] if len(vals) > 1 else None
        row = {
            "family": fam,
            "unit_id": unit,
            "prompt_id": urs[0]["prompt_id"],
            "t0": urs[0]["t0"],
            "KL_min": min(vals),
            "KL_max": max(vals),
            "KL_range": max(vals) - min(vals),
            "KL_range_relative_to_min": (max(vals) - min(vals)) / (min(vals) + EPS),
            "best_config": urs[0]["config"],
            "second_best_config": urs[1]["config"],
            "best_second_gap": best_gap,
            "best_second_gap_relative_to_best": best_gap / (vals[0] + EPS) if best_gap is not None else None,
            "median_pairwise_KL_gap": median(gaps),
            "max_pairwise_KL_gap": max(gaps) if gaps else None,
        }
        out.append(row)
    gaps = [r["best_second_gap"] for r in out]
    ranges = [r["KL_range"] for r in out]
    return out, {
        "family": fam,
        "n_units": len(out),
        "KL_range_mean": mean(ranges),
        "KL_range_median": median(ranges),
        "best_second_gap_mean": mean(gaps),
        "best_second_gap_median": median(gaps),
        "median_pairwise_gap": median(all_pair_gaps),
        "max_pairwise_gap": max(all_pair_gaps) if all_pair_gaps else None,
        "pair_gap_fraction_lt_1e-6": sum(g < 1e-6 for g in all_pair_gaps) / len(all_pair_gaps),
        "pair_gap_fraction_lt_1e-5": sum(g < 1e-5 for g in all_pair_gaps) / len(all_pair_gaps),
        "pair_gap_fraction_lt_1e-4": sum(g < 1e-4 for g in all_pair_gaps) / len(all_pair_gaps),
    }


def winner_distribution(rows, configs):
    by_unit = defaultdict(list)
    for r in rows:
        if r["config"] in configs:
            by_unit[r["unit_id"]].append(r)
    true = Counter()
    m4 = Counter()
    m5 = Counter()
    m0 = Counter()
    for urs in by_unit.values():
        true[min(urs, key=lambda r: r["future_KL"])["config"]] += 1
        m0[min(urs, key=lambda r: r["M0"])["config"]] += 1
        m4[min(urs, key=lambda r: r["M4"])["config"]] += 1
        m5[min(urs, key=lambda r: r["M5"])["config"]] += 1
    return {"true_KL": dict(true), "M0_argmin": dict(m0), "M4_argmin": dict(m4), "M5_argmin": dict(m5)}


def monotonicity(rows, configs):
    fam = family_name(configs)
    by_unit = defaultdict(dict)
    for r in rows:
        if r["config"] in configs:
            by_unit[r["unit_id"]][r["config"]] = r
    out = []
    dec = inc = adj_dec = adj_inc = adj_total = 0
    for unit, d in sorted(by_unit.items()):
        vals = [d[c]["future_KL"] for c in configs]
        nonincreasing = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        nondecreasing = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        dec += int(nonincreasing)
        inc += int(nondecreasing)
        for i in range(len(vals) - 1):
            adj_dec += int(vals[i] >= vals[i + 1])
            adj_inc += int(vals[i] <= vals[i + 1])
            adj_total += 1
        out.append({"family": fam, "unit_id": unit, "configs_order": configs, "KL_values": vals, "nonincreasing_128_to_16": nonincreasing, "nondecreasing_128_to_16": nondecreasing})
    return out, {"family": fam, "strict_order_units_128_ge_64_ge_32_ge_16": dec, "strict_order_units_128_le_64_le_32_le_16": inc, "unit_count": len(out), "adjacent_decrease_fraction": adj_dec / adj_total if adj_total else None, "adjacent_increase_fraction": adj_inc / adj_total if adj_total else None}


def metric_ladder(rows):
    out = []
    for fam, configs in [("R", R_CONFIGS), ("C", C_CONFIGS)]:
        fr = [r for r in rows if r["config"] in configs]
        for m in METRICS:
            pw = pairwise_concordance(fr, m)
            out.append({
                "family": fam,
                "metric": m,
                "spearman": spearman([r[m] for r in fr], [r["future_KL"] for r in fr]),
                "kendall_tau": kendall_tau([r[m] for r in fr], [r["future_KL"] for r in fr]),
                **pw,
            })
    return out


def m4_m5_increment(rows):
    buckets = {k: {"M4_correct": 0, "M4_incorrect": 0, "M5_correct": 0, "M5_incorrect": 0, "wrong_to_right": 0, "right_to_wrong": 0, "same_correct": 0, "same_wrong": 0, "total": 0} for k in ["R-R", "C-C", "R-C", "overall"]}
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)
    for urs in by_unit.values():
        for i in range(len(urs)):
            for j in range(i + 1, len(urs)):
                a, b = urs[i], urs[j]
                dk = a["future_KL"] - b["future_KL"]
                if abs(dk) <= EPS:
                    continue
                cls = pair_class(a["config"], b["config"])
                for bucket in [cls, "overall"]:
                    d = buckets[bucket]
                    m4_ok = (a["M4"] - b["M4"]) * dk > 0 if abs(a["M4"] - b["M4"]) > EPS else None
                    m5_ok = (a["M5"] - b["M5"]) * dk > 0 if abs(a["M5"] - b["M5"]) > EPS else None
                    if m4_ok is None or m5_ok is None:
                        continue
                    d["total"] += 1
                    d["M4_correct"] += int(m4_ok)
                    d["M4_incorrect"] += int(not m4_ok)
                    d["M5_correct"] += int(m5_ok)
                    d["M5_incorrect"] += int(not m5_ok)
                    d["wrong_to_right"] += int((not m4_ok) and m5_ok)
                    d["right_to_wrong"] += int(m4_ok and (not m5_ok))
                    d["same_correct"] += int(m4_ok and m5_ok)
                    d["same_wrong"] += int((not m4_ok) and (not m5_ok))
    for d in buckets.values():
        d["M4_pairwise"] = d["M4_correct"] / d["total"] if d["total"] else None
        d["M5_pairwise"] = d["M5_correct"] / d["total"] if d["total"] else None
    return buckets


def candidate_failure(rows):
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)
    out = []
    true_family = Counter()
    for unit, urs in sorted(by_unit.items()):
        urs = list(urs)
        true_rank = ranking(urs, "future_KL")
        true_winner = true_rank[0]
        true_family["R" if true_winner.startswith("R") else "C"] += 1
        best_kl = min(r["future_KL"] for r in urs)
        second_kl = sorted(r["future_KL"] for r in urs)[1]
        row = {
            "unit_id": unit,
            "prompt_id": urs[0]["prompt_id"],
            "t0": urs[0]["t0"],
            "true_KL_ranking": true_rank,
            "M0_ranking": ranking(urs, "M0"),
            "M4_ranking": ranking(urs, "M4"),
            "M5_ranking": ranking(urs, "M5"),
            "true_winner": true_winner,
            "M0_winner": ranking(urs, "M0")[0],
            "M4_winner": ranking(urs, "M4")[0],
            "M5_winner": ranking(urs, "M5")[0],
            "best_KL": best_kl,
            "best_second_gap": second_kl - best_kl,
        }
        for m in ["M0", "M4", "M5"]:
            pred = min(urs, key=lambda r: r[m])
            row[f"{m}_selected_KL"] = pred["future_KL"]
            row[f"{m}_absolute_regret"] = pred["future_KL"] - best_kl
            row[f"{m}_relative_regret"] = (pred["future_KL"] - best_kl) / (best_kl + EPS)
            row[f"{m}_hit"] = pred["config"] == true_winner
        out.append(row)
    return out, dict(true_family)


def unit_failure_map(rows, sep_r, sep_c):
    by_unit = defaultdict(list)
    sr = {r["unit_id"]: r for r in sep_r}
    sc = {r["unit_id"]: r for r in sep_c}
    out = []
    for r in rows:
        by_unit[r["unit_id"]].append(r)
    for unit, urs in sorted(by_unit.items()):
        r_rows = [r for r in urs if r["config"].startswith("R")]
        c_rows = [r for r in urs if r["config"].startswith("C")]
        m5c = pairwise_concordance(c_rows, "M5")
        m5r = pairwise_concordance(r_rows, "M5")
        all_best = min(urs, key=lambda r: r["future_KL"])
        m5_best = min(urs, key=lambda r: r["M5"])
        out.append({
            "unit_id": unit,
            "prompt_id": urs[0]["prompt_id"],
            "t0": urs[0]["t0"],
            "R_family_KL_range": sr[unit]["KL_range"],
            "C_family_KL_range": sc[unit]["KL_range"],
            "R_family_best_second_gap": sr[unit]["best_second_gap"],
            "C_family_best_second_gap": sc[unit]["best_second_gap"],
            "M5_within_R_pairwise": m5r["pairwise_concordance"],
            "M5_within_C_pairwise": m5c["pairwise_concordance"],
            "true_C_ranking": ranking(c_rows, "future_KL"),
            "M5_C_ranking": ranking(c_rows, "M5"),
            "selection_winner": m5_best["config"],
            "true_winner": all_best["config"],
            "selection_regret": m5_best["future_KL"] - all_best["future_KL"],
        })
    return out


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (RES_DIR / f"{SLUG}_{name}").write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(final):
    lines = [
        f"# {TASK}", "",
        "## 1. Task", "Localize why frozen M4/M5 rank R-family configs well but fail within C-family.", "",
        "## 2. Data Gate", json.dumps(final["data_gate"], indent=2, sort_keys=True), "",
        "## 3. Family Separability", json.dumps(final["family_separability_summary"], indent=2, sort_keys=True), "",
        "## 4. Winner Distribution", json.dumps(final["winner_distribution"], indent=2, sort_keys=True), "",
        "## 5. Granularity Monotonicity", json.dumps(final["granularity_monotonicity_summary"], indent=2, sort_keys=True), "",
        "## 6. Metric Ladder By Family", json.dumps(final["metric_ladder_key"], indent=2, sort_keys=True), "",
        "## 7. M4 vs M5 Increment", json.dumps(final["m4_m5_increment_summary"], indent=2, sort_keys=True), "",
        "## 8. Candidate Selection Failure", "See `candidate_failure_map.json` and `regret_margin_analysis.json`.", "",
        "## 9. Unit Failure Map", "See `unit_failure_map.json`.", "",
        "## 10. Phase A Decision", json.dumps(final["phase_a_decision"], indent=2, sort_keys=True), "",
        "## 11. Phase B", json.dumps({"PHASE_B_REQUIRED": final["PHASE_B_REQUIRED"], "PHASE_B_STATUS": final["PHASE_B_STATUS"]}, indent=2, sort_keys=True), "",
        "## 12. Final Scientific Classification", f"`{final['FINAL_SCIENTIFIC_CLASSIFICATION']}`", "",
        "## 13. Method Design", f"METHOD_PRINCIPLE_EXTRACTION_READY=`{final['METHOD_PRINCIPLE_EXTRACTION_READY']}`; METHOD_DESIGN_READY=`{final['METHOD_DESIGN_READY']}`", "",
        "## 14. Interpretation", final["scientific_interpretation"], "",
        "## 15. Next Recommended Task", final["NEXT_RECOMMENDED_TASK"],
    ]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REP_DIR / f"{SLUG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RES_DIR / f"{SLUG}_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    prev = json.loads(SUMMARY.read_text(encoding="utf-8"))
    data_gate = {
        "source_raw": str(RAW),
        "source_summary": str(SUMMARY),
        "source_commit": prev.get("git_environment", {}).get("HEAD") or prev.get("HEAD"),
        "n_cases": len(rows),
        "n_units": len({r["unit_id"] for r in rows}),
        "n_configs": len({r["config"] for r in rows}),
        "required_fields_present": all(all(k in r for k in ["unit_id", "prompt_id", "layer", "head", "t0", "config", "M0", "M1", "M2", "M3", "M4", "M5", "future_KL"]) for r in rows),
        "EXISTING_DATA_GATE": "PASS" if len(rows) == 72 else "FAIL",
    }
    sep_r, sep_r_sum = separability(rows, R_CONFIGS)
    sep_c, sep_c_sum = separability(rows, C_CONFIGS)
    wd_r = winner_distribution(rows, R_CONFIGS)
    wd_c = winner_distribution(rows, C_CONFIGS)
    mono_r, mono_r_sum = monotonicity(rows, R_CONFIGS)
    mono_c, mono_c_sum = monotonicity(rows, C_CONFIGS)
    ladder = metric_ladder(rows)
    inc = m4_m5_increment(rows)
    cf, true_family = candidate_failure(rows)
    uf = unit_failure_map(rows, sep_r, sep_c)
    m5_misses = [r for r in cf if not r["M5_hit"]]
    regret_margin = {
        "per_unit": cf,
        "M5_miss_count": len(m5_misses),
        "M5_hit_rate": mean([r["M5_hit"] for r in cf]),
        "M5_mean_regret": mean([r["M5_absolute_regret"] for r in cf]),
        "M5_median_regret": median([r["M5_absolute_regret"] for r in cf]),
        "M5_max_regret": max([r["M5_absolute_regret"] for r in cf]),
        "M5_miss_regret_le_best_second_gap_fraction": mean([r["M5_absolute_regret"] <= r["best_second_gap"] + EPS for r in m5_misses]),
        "M5_miss_regret_le_C_KL_range_fraction": mean([next(u for u in uf if u["unit_id"] == r["unit_id"])["C_family_KL_range"] + EPS >= r["M5_absolute_regret"] for r in m5_misses]),
    }
    low_sep = sep_c_sum["KL_range_median"] < sep_r_sum["KL_range_median"] * 0.5 and sep_c_sum["best_second_gap_median"] < sep_r_sum["best_second_gap_median"] * 0.5
    meaningful_sep = not low_sep
    phase_b_required = bool(meaningful_sep and inc["C-C"]["M5_pairwise"] < 0.6)
    phase_a_case = "C_FAMILY_STRUCTURAL_PROXY_FAILURE" if phase_b_required else "C_FAMILY_LOW_RISK_LOW_SEPARATION"
    classification = "C_FAMILY_FAILURE_MIXED" if phase_b_required else "C_FAMILY_LOW_RISK_LOW_SEPARATION"
    ladder_key = {fam: {r["metric"]: r for r in ladder if r["family"] == fam} for fam in ["R", "C"]}
    final = {
        "TASK": TASK,
        "timestamp": now(),
        "git": {"branch": safe_sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]), "HEAD": safe_sh(["git", "rev-parse", "HEAD"]), "status": safe_sh(["git", "status", "--short"])},
        "PHASE_A_STATUS": "COMPLETE",
        "EXISTING_DATA_GATE": data_gate["EXISTING_DATA_GATE"],
        "data_gate": data_gate,
        "family_separability_summary": {"R": sep_r_sum, "C": sep_c_sum, "C_over_R_KL_range_median": sep_c_sum["KL_range_median"] / (sep_r_sum["KL_range_median"] + EPS), "C_over_R_best_second_gap_median": sep_c_sum["best_second_gap_median"] / (sep_r_sum["best_second_gap_median"] + EPS)},
        "winner_distribution": {"R_family": wd_r, "C_family": wd_c, "overall_true_winner_family": true_family},
        "granularity_monotonicity_summary": {"R": mono_r_sum, "C": mono_c_sum},
        "metric_ladder_key": ladder_key,
        "m4_m5_increment_summary": inc,
        "candidate_failure_summary": {"true_winner_family": true_family, "M5_hit_rate": regret_margin["M5_hit_rate"], "M5_mean_regret": regret_margin["M5_mean_regret"]},
        "regret_margin_summary": {k: v for k, v in regret_margin.items() if k != "per_unit"},
        "phase_a_decision": {"case": phase_a_case, "C_FAMILY_LOW_SEPARATION": low_sep, "C_FAMILY_MEANINGFUL_SEPARATION": meaningful_sep, "reason": "C separability is compared directly with R and M5 C-C pairwise remains below 0.6."},
        "C_FAMILY_SEPARABILITY": "LOW" if low_sep else "MEANINGFUL_OR_MIXED",
        "R_FAMILY_SEPARABILITY": "HIGHER_THAN_C" if sep_r_sum["KL_range_median"] > sep_c_sum["KL_range_median"] else "NOT_HIGHER_THAN_C",
        "C_LOW_SEPARATION_HYPOTHESIS": "SUPPORTED" if low_sep else "NOT_SUPPORTED_AS_SOLE_EXPLANATION",
        "C_STRUCTURAL_PROXY_FAILURE_HYPOTHESIS": "SUPPORTED" if phase_b_required else "PARTIAL_OR_NOT_REQUIRED",
        "PHASE_B_REQUIRED": phase_b_required,
        "PHASE_B_STATUS": "NOT_RUN_PHASE_A_SUFFICIENT" if not phase_b_required else "REQUIRED_NOT_RUN_IN_THIS_PHASE_A_SCRIPT",
        "FINAL_SCIENTIFIC_CLASSIFICATION": classification,
        "METHOD_PRINCIPLE_EXTRACTION_READY": "YES",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "minimal C-only multi-horizon localization" if phase_b_required else "report C-family low-separation failure localization; do not design quantizer yet",
        "scientific_interpretation": "Phase A uses only the previous 72 formal cases. It does not modify M4/M5, refit normalization, introduce persistence, or define a new proxy. The key question is whether C-family failures are near-ties or a real missing damage source.",
    }
    save_json("phase_a_raw_audit.json", {"data_gate": data_gate, "rows_preview": rows[:3]})
    save_json("family_separability.json", {"R_per_unit": sep_r, "C_per_unit": sep_c, "summary": final["family_separability_summary"]})
    save_json("winner_distribution.json", final["winner_distribution"])
    save_json("granularity_monotonicity.json", {"R_per_unit": mono_r, "C_per_unit": mono_c, "summary": final["granularity_monotonicity_summary"]})
    save_json("metric_ladder_by_family.json", {"rows": ladder, "key": ladder_key})
    save_json("m4_m5_increment_audit.json", inc)
    save_json("candidate_failure_map.json", {"rows": cf})
    save_json("regret_margin_analysis.json", regret_margin)
    save_json("unit_failure_map.json", {"rows": uf})
    save_json("phase_a_decision.json", final["phase_a_decision"])
    save_json("final_summary.json", final)
    write_report(final)
    print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
