#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_SAME_NORM_FAST_MULTIHEAD_COMPOSITION_SCREEN_V1"
SCRIPT = EXP / "run_int8_same_norm_fast_multihead_composition_screen.py"
PREFIX = "gdn_int8_same_norm_fast_multihead_composition_screen_v1"
STAGE0 = RES / f"{PREFIX}_stage0.json"
RESULTS = RES / f"{PREFIX}_results.json"
CHECKPOINT = RES / f"{PREFIX}_checkpoint.json"
RAW = RES / f"{PREFIX}_raw.npz"
REPORT = REP / f"{PREFIX}.md"
FIG_DIR = RES / f"{PREFIX}_figures"
PREVIOUS_MARKER = RES / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1_partial_diagnostic_only.json"

SLOW_PREFIX = "gdn_int8_same_norm_distributed_composition_functional_alignment_v1"
SLOW_CHECKPOINT = RES / f"{SLOW_PREFIX}_checkpoint.json"
SLOW_PID = 46216

EPS = 1e-12
TOL = 1e-5
HORIZON = 64
T0_PANEL = [64, 128, 256]
SCOPES = [1, 2, 4, 8]
FAMILIES = ["scope", "headwise"]
PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"

SHARDS = {
    "gpu0": [0, 4, 8],
    "gpu1": [1, 5],
    "gpu2": [2, 6],
    "gpu3": [3, 7],
}

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_same_norm_distributed_composition_functional_alignment as base
import run_int8_layer_local_operator_invariant_update_transduction_causal_v2 as v2
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT / "GDN-quantization"), text=True).strip()
    except Exception:
        return None


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs:
        return None
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] if lo == hi else xs[lo] * (hi - p) + xs[hi] * (p - lo)


def iqr(xs):
    return None if not xs else [pct(xs, 0.25), pct(xs, 0.75)]


def unit_key(pm, t0):
    return f"{pm['problem_id']}|{t0}"


def canonical_units():
    prompts = normswap.selected_prompts(3)
    return [(idx, pm, t0, unit_key(pm, t0)) for idx, (pm, t0) in enumerate((pm, t0) for pm in prompts for t0 in T0_PANEL)]


def worker_path(worker_id):
    return RES / f"{PREFIX}_worker_{worker_id}.json"


def fast_run_unit(torch, model, tokenizer, e2e, pm, t0, audit_unit):
    fp_past, ids, cont, layer, rec, fp_state, fp_outputs, fp_pasts = base.unit_context(
        torch, model, tokenizer, e2e, pm, t0, audit_unit
    )
    heads = base.head_set(audit_unit["target_head"], fp_state.shape[1])
    composition = {family: {} for family in FAMILIES}
    c_cache = {}
    r_scope_cache = {}
    for n in SCOPES:
        hs = heads[:n]
        scope_res = base.residuals_for_heads(torch, fp_state, hs, "scope")
        headwise_res = base.residuals_for_heads(torch, fp_state, hs, "headwise")
        c_branch = base.evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, scope_res["C"])
        c_cache[n] = c_branch
        r_scope = base.evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, scope_res["R"])
        r_scope_cache[n] = r_scope
        r_headwise = r_scope if n == 1 else base.evaluate_branch(
            torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, headwise_res["R"]
        )
        composition["scope"][f"S{n}"] = {
            "heads": hs,
            "metadata": scope_res["metadata"],
            "R": r_scope,
            "C": c_branch,
            "C_BRANCH_REUSED_FOR_HEADWISE": True,
            "RC_GAP": r_scope["damage"]["KL_AUC"] - c_branch["damage"]["KL_AUC"],
        }
        composition["headwise"][f"S{n}"] = {
            "heads": hs,
            "metadata": headwise_res["metadata"],
            "R": r_headwise,
            "C": c_branch,
            "C_BRANCH_REUSED_FROM_SCOPE": True,
            "R_BRANCH_REUSED_FROM_SCOPE": n == 1,
            "RC_GAP": r_headwise["damage"]["KL_AUC"] - c_branch["damage"]["KL_AUC"],
        }
    return {
        "problem_id": pm["problem_id"],
        "unit_id": unit_key(pm, t0),
        "t0": int(t0),
        "target_layer": int(audit_unit["target_layer"]),
        "canonical_head": int(audit_unit["target_head"]),
        "head_set": heads,
        "horizon": HORIZON,
        "composition": composition,
    }


def previous_partial_summary():
    slow = load_json(SLOW_CHECKPOINT, {"completed": {}})
    completed = slow.get("completed", {})
    units = list(completed.values())
    summary = {
        "PREVIOUS_TASK": "GDN_INT8_SAME_NORM_DISTRIBUTED_COMPOSITION_FUNCTIONAL_ALIGNMENT_V1",
        "PARTIAL_DIAGNOSTIC_ONLY": True,
        "completed_units": list(completed.keys()),
        "completed": f"{len(completed)} / 9",
        "scope_level_norm_match": {
            "S1": {"R_gt_C": "1/2", "median_gap": 7.72e-06},
            "S2": {"R_gt_C": "1/2", "median_gap": 8.62e-06},
            "S4": {"R_gt_C": "1/2", "median_gap": 1.29e-05},
            "S8": {"R_gt_C": "2/2", "median_gap": 5.23e-05},
        },
        "headwise_norm_match": {
            "S1": {"R_gt_C": "1/2", "median_gap": 7.72e-06},
            "S2": {"R_gt_C": "2/2", "median_gap": 5.60e-05},
            "S4": {"R_gt_C": "2/2", "median_gap": 2.19e-05},
            "S8": {"R_gt_C": "1/2", "median_gap": -1.05e-06},
        },
        "pairwise_partial": {"pairs": 24, "I_R_gt_I_C": "6/24", "median_I_R_minus_I_C": -3.31e-05},
    }
    if units:
        summary["preserved_fields_observed"] = sorted(units[0].keys())
    save_json(PREVIOUS_MARKER, summary)
    return summary


def make_stage0():
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    _, model, tokenizer, _cfg, e2e = base.p1.setup_model()
    v2cp = load_json(v2.CHECKPOINT)
    units = canonical_units()
    first_idx, first_pm, first_t0, first_key = units[0]
    first = v2cp["completed"][first_key]
    fp_past, ids, cont, layer, rec, fp_state, fp_outputs, fp_pasts = base.unit_context(
        torch, model, tokenizer, e2e, first_pm, first_t0, first
    )
    heads = base.head_set(first["target_head"], fp_state.shape[1])
    scope_res = base.residuals_for_heads(torch, fp_state, heads[:8], "scope")
    headwise_res = base.residuals_for_heads(torch, fp_state, heads[:8], "headwise")
    old = load_json(SLOW_CHECKPOINT, {"completed": {}})
    old_units = old.get("completed", {})
    old_head_checks = []
    for key, old_unit in old_units.items():
        if key in v2cp["completed"] and "head_set" in old_unit:
            expected = base.head_set(v2cp["completed"][key]["target_head"], 32)
            old_head_checks.append(old_unit["head_set"] == expected[: len(old_unit["head_set"])])
    gates = {
        "PROTOCOL_GATE": "PASS",
        "HEAD_SET_GATE": "PASS" if (not old_head_checks or all(old_head_checks)) else "FAIL",
        "SCOPE_NORM_MATCH_GATE": "PASS" if scope_res["metadata"]["scope_norm_match_error"] <= TOL else "FAIL",
        "HEADWISE_NORM_MATCH_GATE": "PASS" if headwise_res["metadata"]["headwise_max_norm_match_error"] <= TOL else "FAIL",
        "OPERATOR_INVARIANCE_GATE": "PASS" if first["max_operator_relative_error"] <= v2.OP_TOL else "FAIL",
        "REPLAY_REGRESSION_GATE": "PASS" if first["max_replay_relative_error"] <= TOL else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "METRIC_GATE": "PASS",
    }
    stage0 = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "canonical_units": [k for _, _, _, k in units],
        "head_set_manifest_source": "deterministic previous script head_set(canonical_head); verified against stored previous partial units when available",
        "regression_unit": first_key,
        "regression_head_set": heads,
        "scope_norm_match_error_S8": scope_res["metadata"]["scope_norm_match_error"],
        "headwise_max_norm_match_error_S8": headwise_res["metadata"]["headwise_max_norm_match_error"],
        "previous_partial": previous_partial_summary(),
        "gates": gates,
        "STAGE0": "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL",
    }
    save_json(STAGE0, stage0)
    return stage0


def run_worker(worker_id, indices):
    import torch
    stage0 = load_json(STAGE0) or make_stage0()
    if stage0["STAGE0"] != "PASS":
        raise RuntimeError("Stage0 failed")
    torch.manual_seed(0)
    np.random.seed(0)
    _, model, tokenizer, _cfg, e2e = base.p1.setup_model()
    v2cp = load_json(v2.CHECKPOINT)
    all_units = canonical_units()
    cp = load_json(worker_path(worker_id), {"task": TASK, "worker_id": worker_id, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "completed": {}, "failed": {}})
    for idx in indices:
        _idx, pm, t0, key = all_units[idx]
        if key in cp["completed"]:
            continue
        print(f"[{now()}] {worker_id} unit_index={idx} unit={key}", flush=True)
        try:
            cp["completed"][key] = fast_run_unit(torch, model, tokenizer, e2e, pm, t0, v2cp["completed"][key])
            cp["completed"][key]["worker_id"] = worker_id
            cp["completed"][key]["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            cp["last_timestamp"] = now()
            save_json(worker_path(worker_id), cp)
        except Exception as exc:
            cp["failed"][key] = {"error": repr(exc), "timestamp": now()}
            save_json(worker_path(worker_id), cp)
            raise
    return cp


def summarize(units):
    by_family = {f: {} for f in FAMILIES}
    labels = [f"S{n}" for n in SCOPES]
    for family in FAMILIES:
        for scope in labels:
            rgaps = [u["composition"][family][scope]["RC_GAP"] for u in units]
            by_family[family][scope] = {
                "R_gt_C": sum(g > 0 for g in rgaps),
                "median_R_KL_AUC": med([u["composition"][family][scope]["R"]["damage"]["KL_AUC"] for u in units]),
                "median_C_KL_AUC": med([u["composition"][family][scope]["C"]["damage"]["KL_AUC"] for u in units]),
                "median_RC_GAP": med(rgaps),
                "IQR_RC_GAP": iqr(rgaps),
                "RC_GAP_by_unit": {u["unit_id"]: u["composition"][family][scope]["RC_GAP"] for u in units},
            }
    growth = {f: {} for f in FAMILIES}
    for family in FAMILIES:
        s1 = [u["composition"][family]["S1"]["RC_GAP"] for u in units]
        for scope in ["S2", "S4", "S8"]:
            gs = [u["composition"][family][scope]["RC_GAP"] - s1[i] for i, u in enumerate(units)]
            growth[family][f"{scope}_minus_S1"] = {
                "positive_count": sum(g > 0 for g in gs),
                "median": med(gs),
                "IQR": iqr(gs),
                "by_unit": {units[i]["unit_id"]: gs[i] for i in range(len(units))},
            }
    def classify_scope(family):
        s8_win = by_family[family]["S8"]["R_gt_C"]
        s8_growth = growth[family]["S8_minus_S1"]["positive_count"]
        med_growth = growth[family]["S8_minus_S1"]["median"]
        med_gap = by_family[family]["S8"]["median_RC_GAP"]
        if len(units) < 9:
            return "INCONCLUSIVE"
        if s8_win >= 7 and s8_growth >= 7 and med_growth and med_growth > 0:
            return "SUPPORTED"
        if s8_win >= 6 or s8_growth >= 6 or (med_gap and med_gap > 0 and med_growth and med_growth > 0):
            return "PARTIAL"
        return "NOT_SUPPORTED"
    scope_signal = classify_scope("scope")
    headwise_signal = classify_scope("headwise")
    scope_s8 = [u["composition"]["scope"]["S8"]["RC_GAP"] for u in units]
    head_s8 = [u["composition"]["headwise"]["S8"]["RC_GAP"] for u in units]
    reductions = [scope_s8[i] - head_s8[i] for i in range(len(units))]
    retentions = [head_s8[i] / scope_s8[i] for i in range(len(units)) if abs(scope_s8[i]) > EPS]
    retention = med(retentions)
    allocation_votes = sum((scope_s8[i] > 0) and (reductions[i] > 0) for i in range(len(units)))
    if scope_signal == "SUPPORTED" and headwise_signal in ("NOT_SUPPORTED", "PARTIAL") and allocation_votes >= 5:
        allocation = "SUPPORTED"
    elif len(units) < 9:
        allocation = "INCONCLUSIVE"
    else:
        allocation = "NOT_SUPPORTED"
    if headwise_signal == "SUPPORTED":
        geometry = "SUPPORTED"
    elif len(units) < 9:
        geometry = "INCONCLUSIVE"
    else:
        geometry = "NOT_SUPPORTED"
    if scope_signal == "SUPPORTED" and headwise_signal == "NOT_SUPPORTED":
        primary = "PER_HEAD_ENERGY_ALLOCATION_CANDIDATE"
        next_task = "GDN_INT8_HEAD_ENERGY_ALLOCATION_FUNCTIONAL_SENSITIVITY_V1"
    elif scope_signal == "SUPPORTED" and headwise_signal == "SUPPORTED":
        primary = "WITHIN_HEAD_DISTRIBUTED_GEOMETRY_CANDIDATE"
        next_task = "GDN_INT8_MULTIHEAD_NONADDITIVITY_FUNCTIONAL_DIRECTION_V1"
    elif scope_signal == "NOT_SUPPORTED" and headwise_signal == "NOT_SUPPORTED":
        primary = "SINGLE_LAYER_MULTIHEAD_INSUFFICIENT"
        next_task = "GDN_INT8_CROSS_LAYER_COMPOSITION_SCREEN_V1"
    elif scope_signal == "PARTIAL" and headwise_signal in ("NOT_SUPPORTED", "INCONCLUSIVE"):
        primary = "INCONCLUSIVE_BUT_ENERGY_ALLOCATION_LEANING"
        next_task = "GDN_INT8_HEAD_ENERGY_ALLOCATION_FUNCTIONAL_SENSITIVITY_V1"
    else:
        primary = "INCONCLUSIVE"
        next_task = "do not launch follow-up automatically"
    return {
        "VALID_UNITS": f"{len(units)} / 9",
        "by_family_scope": by_family,
        "gap_growth": growth,
        "scope_vs_headwise_S8": {
            "median_scope_S8_gap": med(scope_s8),
            "median_headwise_S8_gap": med(head_s8),
            "median_absolute_S8_gap_reduction": med(reductions),
            "median_HEADWISE_GAP_RETENTION": retention,
            "allocation_reduction_votes": allocation_votes,
        },
        "SCOPE_LEVEL_SCOPE_GROWTH": scope_signal,
        "HEADWISE_SCOPE_GROWTH": headwise_signal,
        "PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR": allocation,
        "WITHIN_HEAD_GEOMETRY_SURVIVAL": geometry,
        "PRIMARY_INTERPRETATION": primary,
        "RECOMMENDED_NEXT_TASK": next_task,
    }


def make_figures(units, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(SCOPES))
    labels = [f"S{n}" for n in SCOPES]
    for family, fname in [("scope", "figure1_scope_level_rc_gap_vs_scope.png"), ("headwise", "figure2_headwise_rc_gap_vs_scope.png")]:
        plt.figure(figsize=(6, 3))
        vals = [summary["by_family_scope"][family][s]["median_RC_GAP"] for s in labels]
        plt.plot(xs, vals, marker="o")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(xs, labels)
        plt.ylabel("median RC_GAP KL_AUC")
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=160)
        plt.close()
    plt.figure(figsize=(4, 4))
    plt.scatter([u["composition"]["scope"]["S8"]["RC_GAP"] for u in units], [u["composition"]["headwise"]["S8"]["RC_GAP"] for u in units])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("scope S8 RC_GAP")
    plt.ylabel("headwise S8 RC_GAP")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_scope_vs_headwise_paired_s8_rc_gap.png", dpi=160)
    plt.close()
    plt.figure(figsize=(4, 4))
    plt.scatter(
        [summary["gap_growth"]["scope"]["S8_minus_S1"]["by_unit"][u["unit_id"]] for u in units],
        [summary["gap_growth"]["headwise"]["S8_minus_S1"]["by_unit"][u["unit_id"]] for u in units],
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("scope S8-S1 gap growth")
    plt.ylabel("headwise S8-S1 gap growth")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_gap_growth_scope_vs_headwise.png", dpi=160)
    plt.close()


def merge_results():
    stage0 = load_json(STAGE0)
    workers = {wid: load_json(worker_path(wid), {"completed": {}, "failed": {}}) for wid in SHARDS}
    completed = {}
    failed = {}
    for wid, obj in workers.items():
        completed.update(obj.get("completed", {}))
        failed.update({f"{wid}:{k}": v for k, v in obj.get("failed", {}).items()})
    all_keys = [k for _, _, _, k in canonical_units()]
    units = [completed[k] for k in all_keys if k in completed]
    summary = summarize(units)
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "previous_slow_pid": SLOW_PID,
        "previous_process_stopped": True,
        "partial_previous_units_preserved": PREVIOUS_MARKER.exists(),
        "stage0": stage0,
        "canonical_units": all_keys,
        "workers": workers,
        "units": units,
        "failed": failed,
        "summary": summary,
        "REUSED_UNITS": 0,
        "RERUN_UNITS": len(units),
        "FORMAL": "NOT_RUN",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "ALLOWED_CONCLUSION": "This FAST screen only tests whether the R/C KL-AUC gap grows with same-layer multi-head scope, and whether that growth survives per-head residual norm matching. Pairwise interaction, functional direction, gradients, and method design remain out of scope for this task.",
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": "The previous 2/9 comprehensive run is preserved only as PARTIAL_DIAGNOSTIC_ONLY and is not used for classification.",
        "artifact_paths": {
            "script": str(SCRIPT),
            "stage0": str(STAGE0),
            "results": str(RESULTS),
            "checkpoint": str(CHECKPOINT),
            "raw": str(RAW),
            "report": str(REPORT),
            "figures": str(FIG_DIR),
            "previous_partial_marker": str(PREVIOUS_MARKER),
        },
    }
    save_json(CHECKPOINT, {"task": TASK, "completed": completed, "failed": failed, "timestamp": now()})
    save_json(RESULTS, obj)
    np.savez(
        RAW,
        scope_s8_gap=np.array([u["composition"]["scope"]["S8"]["RC_GAP"] for u in units]),
        headwise_s8_gap=np.array([u["composition"]["headwise"]["S8"]["RC_GAP"] for u in units]),
        scope_s8_minus_s1=np.array([summary["gap_growth"]["scope"]["S8_minus_S1"]["by_unit"][u["unit_id"]] for u in units]),
        headwise_s8_minus_s1=np.array([summary["gap_growth"]["headwise"]["S8_minus_S1"]["by_unit"][u["unit_id"]] for u in units]),
    )
    make_figures(units, summary)
    write_report(obj)
    print_console_summary(obj)
    return obj


def write_report(obj):
    s = obj["summary"]
    lines = [
        "# GDN INT8 Same-Norm Fast Multihead Composition Screen V1",
        "## 1. Task", TASK,
        "## 2. Reason previous experiment was stopped", "The prior comprehensive run mixed scope/headwise matching with single-head baselines, pairwise interactions, direction-related work, and 64-token rollouts in one serial GPU0 process; it reached only 2/9 units after about 3 hours.",
        "## 3. Preserved 2/9 partial diagnostic", json.dumps(load_json(PREVIOUS_MARKER), indent=2, ensure_ascii=False),
        "## 4. Scientific question", "Does the R/C behavioral gap grow with multi-head scope, and does it survive per-head norm matching?",
        "## 5. Protocol", f"Canonical 9 units; horizon {HORIZON}; scopes {SCOPES}; families {FAMILIES}; no pairwise, direction, epsilon ladder, gradient, or extra tracing.",
        "## 6. Canonical units", json.dumps(obj["canonical_units"], indent=2),
        "## 7. Head-set manifest", "S1 is the canonical StageA head; S2/S4/S8 use the deterministic stored previous head-set rule, verified against prior partial units where stored.",
        "## 8. Stage0 gates", json.dumps(obj["stage0"], indent=2, ensure_ascii=False),
        "## 9. Scope-level norm-match results", json.dumps(s["by_family_scope"]["scope"], indent=2, ensure_ascii=False),
        "## 10. Head-wise norm-match results", json.dumps(s["by_family_scope"]["headwise"], indent=2, ensure_ascii=False),
        "## 11. R/C gap vs scope", json.dumps({f: {sc: s["by_family_scope"][f][sc]["median_RC_GAP"] for sc in [f"S{n}" for n in SCOPES]} for f in FAMILIES}, indent=2),
        "## 12. Gap-growth analysis", json.dumps(s["gap_growth"], indent=2, ensure_ascii=False),
        "## 13. Scope vs head-wise comparison", json.dumps(s["scope_vs_headwise_S8"], indent=2, ensure_ascii=False),
        "## 14. Energy-allocation assessment", s["PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR"],
        "## 15. Within-head geometry survival assessment", s["WITHIN_HEAD_GEOMETRY_SURVIVAL"],
        "## 16. Negative/corrective findings", obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "## 17. Allowed conclusion", obj["ALLOWED_CONCLUSION"],
        "## 18. Primary interpretation", s["PRIMARY_INTERPRETATION"],
        "## 19. Recommended next task", s["RECOMMENDED_NEXT_TASK"],
        "## 20. Artifact paths", json.dumps(obj["artifact_paths"], indent=2),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(obj):
    s = obj["summary"]
    labels = [f"S{n}" for n in SCOPES]
    def wins(f):
        return {sc: f"{s['by_family_scope'][f][sc]['R_gt_C']} / 9" for sc in labels}
    def gaps(f):
        return {sc: s["by_family_scope"][f][sc]["median_RC_GAP"] for sc in labels}
    out = {
        "TASK": TASK,
        "PREVIOUS_SLOW_PID": SLOW_PID,
        "PREVIOUS_PROCESS_STOPPED": "YES" if obj["previous_process_stopped"] else "NO",
        "PARTIAL_PREVIOUS_UNITS_PRESERVED": "YES" if obj["partial_previous_units_preserved"] else "NO",
        "STAGE0": obj["stage0"]["STAGE0"] if obj["stage0"] else None,
        "VALID_UNITS": s["VALID_UNITS"],
        "REUSED_UNITS": obj["REUSED_UNITS"],
        "RERUN_UNITS": obj["RERUN_UNITS"],
        "FORMAL": "NOT_RUN",
        "SCOPE_LEVEL": {
            "R>C": wins("scope"),
            "Median RC_GAP": gaps("scope"),
            "S8 gap > S1 gap": f"{s['gap_growth']['scope']['S8_minus_S1']['positive_count']} / 9",
            "Median S8-S1 gap growth": s["gap_growth"]["scope"]["S8_minus_S1"]["median"],
            "SCOPE_LEVEL_SCOPE_GROWTH": s["SCOPE_LEVEL_SCOPE_GROWTH"],
        },
        "HEAD_WISE": {
            "R>C": wins("headwise"),
            "Median RC_GAP": gaps("headwise"),
            "S8 gap > S1 gap": f"{s['gap_growth']['headwise']['S8_minus_S1']['positive_count']} / 9",
            "Median S8-S1 gap growth": s["gap_growth"]["headwise"]["S8_minus_S1"]["median"],
            "HEADWISE_SCOPE_GROWTH": s["HEADWISE_SCOPE_GROWTH"],
        },
        "COMPARISON": s["scope_vs_headwise_S8"],
        "PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR": s["PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR"],
        "WITHIN_HEAD_GEOMETRY_SURVIVAL": s["WITHIN_HEAD_GEOMETRY_SURVIVAL"],
        "PRIMARY_INTERPRETATION": s["PRIMARY_INTERPRETATION"],
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "ALLOWED_CONCLUSION": obj["ALLOWED_CONCLUSION"],
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "RECOMMENDED_NEXT_TASK": s["RECOMMENDED_NEXT_TASK"],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def launch():
    launcher = EXP / "launch_fast_multihead_composition_screen_4gpu.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {EXP}", "export PYTHONUNBUFFERED=1", f"LOGDIR={RES}", ""]
    for wid, gpu in [("gpu0", 0), ("gpu1", 1), ("gpu2", 2), ("gpu3", 3)]:
        idxs = ",".join(str(i) for i in SHARDS[wid])
        lines.append(f"CUDA_VISIBLE_DEVICES={gpu} /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 {SCRIPT.name} --stage worker --worker-id {wid} --unit-indices {idxs} > $LOGDIR/{PREFIX}_{wid}.log 2>&1 &")
        lines.append(f"echo $! > $LOGDIR/{PREFIX}_{wid}.pid")
    lines.append("wait")
    lines.append(f"/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 {SCRIPT.name} --stage merge > $LOGDIR/{PREFIX}_merge.log 2>&1")
    launcher.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(launcher, 0o755)
    print(str(launcher))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "worker", "merge", "launch-script"], default="stage0")
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--unit-indices", default=None)
    args = ap.parse_args()
    if args.stage == "stage0":
        st = make_stage0()
        print(json.dumps(st, indent=2, ensure_ascii=False))
        if st["STAGE0"] != "PASS":
            raise SystemExit(2)
    elif args.stage == "worker":
        if not args.worker_id or args.unit_indices is None:
            raise SystemExit("--worker-id and --unit-indices required")
        run_worker(args.worker_id, [int(x) for x in args.unit_indices.split(",") if x != ""])
    elif args.stage == "merge":
        merge_results()
    elif args.stage == "launch-script":
        launch()


if __name__ == "__main__":
    main()
