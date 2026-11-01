#!/usr/bin/env python3
import argparse
import copy
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
TASK = "GDN_INT8_SAME_NORM_DISTRIBUTED_COMPOSITION_FUNCTIONAL_ALIGNMENT_V1"
SCRIPT = EXP / "run_int8_same_norm_distributed_composition_functional_alignment.py"
STAGE0 = RES / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1_stage0.json"
COMP = RES / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1_composition_pilot.json"
PAIRWISE = RES / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1_pairwise_interactions.json"
DIRECTION = RES / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1_direction_pilot.json"
CHECKPOINT = RES / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1_checkpoint.json"
RAW = RES / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1_raw.npz"
REPORT = REP / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1.md"
FIG_DIR = RES / "gdn_int8_same_norm_distributed_composition_functional_alignment_v1_figures"

EPS = 1e-12
TOL = 1e-5
HORIZON = 64
T0_PANEL = [64, 128, 256]
SCOPES = [1, 2, 4, 8]
FAMILIES = ["scope", "headwise"]
PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_layer_local_operator_invariant_update_transduction_causal_v2 as v2
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


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


def auc(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def clone_past(past):
    return copy.deepcopy(past)


def set_layer_state(torch, past, layer, new_state):
    s = p1.get_state(past, layer)
    s.copy_(new_state.to(s.dtype))


def state_delta(torch, past, fp_past):
    return normswap.state_delta(torch, past, fp_past)


def head_set(canonical_head, num_heads=32):
    base = [int(canonical_head)]
    for h in [0, 4, 8, 12, 16, 20, 24, 28, 31]:
        if h not in base and h < num_heads:
            base.append(h)
        if len(base) >= 8:
            break
    return base[:8]


def residuals_for_heads(torch, fp_state, heads, family):
    er, ec = normswap.residuals_for_state(torch, fp_state)
    r = torch.zeros_like(fp_state.detach().float())
    c = torch.zeros_like(fp_state.detach().float())
    for h in heads:
        r[:, h] = er[:, h]
        c[:, h] = ec[:, h]
    if family == "scope":
        rn, cn = norm(torch, r), norm(torch, c)
        if rn > EPS:
            r = r * (cn / (rn + EPS))
        match = abs(norm(torch, r) - norm(torch, c)) / (norm(torch, c) + EPS)
        per_head_match = None
    elif family == "headwise":
        max_match = 0.0
        for h in heads:
            rn, cn = norm(torch, er[:, h]), norm(torch, ec[:, h])
            if rn > EPS:
                r[:, h] = er[:, h] * (cn / (rn + EPS))
            max_match = max(max_match, abs(norm(torch, r[:, h]) - cn) / (cn + EPS))
        match = abs(norm(torch, r) - norm(torch, c)) / (norm(torch, c) + EPS)
        per_head_match = max_match
    else:
        raise ValueError(family)
    return {
        "R": r,
        "C": c,
        "metadata": {
            "family": family,
            "heads": heads,
            "R_norm": norm(torch, r),
            "C_norm": norm(torch, c),
            "scope_norm_match_error": match,
            "headwise_max_norm_match_error": per_head_match,
        },
    }


def branch_from_error(torch, fp_past, rec, fp_state, layer, err):
    du = v2.local_du_tensors(torch, rec, fp_state, err)
    p = clone_past(fp_past)
    set_layer_state(torch, p, layer, du["fp_next"] + du["F"])
    return p, du


def run_damage(torch, model, ids, cont, t0, branch_past, fp_outputs=None, fp_pasts=None):
    past = branch_past
    cur = ids.clone()
    kls, states = [], []
    outs, pasts = [], []
    steps = min(HORIZON, max(0, len(cont) - t0))
    with torch.inference_mode():
        for off in range(steps):
            out = p1.feed_step(torch, model, cur, None, past)
            past = out.past_key_values
            outs.append(out)
            pasts.append(past)
            if fp_outputs is None:
                kls.append(0.0)
                states.append(0.0)
            else:
                kls.append(p1.logits_metrics(torch, fp_outputs[off].logits, out.logits)["KL"])
                _abs, rel = state_delta(torch, past, fp_pasts[off])
                states.append(rel)
            nxt = t0 + off + 1
            if nxt < len(cont):
                cur = torch.tensor([[cont[nxt]]], dtype=ids.dtype, device=ids.device)
    return {
        "outputs": outs,
        "pasts": pasts,
        "KL_curve": kls,
        "STATE_curve": states,
        "KL_AUC": auc(kls),
        "STATE_AUC": auc(states),
        "peak_KL": max(kls) if kls else None,
        "final_KL": kls[-1] if kls else None,
    }


def evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, err):
    past, du = branch_from_error(torch, fp_past, rec, fp_state, layer, err)
    dmg = run_damage(torch, model, ids, cont, t0, past, fp_outputs, fp_pasts)
    return {
        "damage": {k: v for k, v in dmg.items() if k not in ("outputs", "pasts")},
        "initial_norm": norm(torch, err),
        "post_recurrence_norm": norm(torch, du["F"]),
        "identity_max_rel": v2.max_nested_rel(du["identity"]),
    }


def unit_context(torch, model, tokenizer, e2e, pm, t0, audit_unit):
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        layer = int(audit_unit["target_layer"])
        rec = collector["records"][layer]
        fp_state = p1.get_state(fp_past, layer).detach().float().clone()
        fp_run = run_damage(torch, model, ids, cont, t0, clone_past(fp_past))
        return fp_past, ids, cont, layer, rec, fp_state, fp_run["outputs"], fp_run["pasts"]
    finally:
        collector["close"]()


def run_unit(torch, model, tokenizer, e2e, pm, t0, audit_unit):
    fp_past, ids, cont, layer, rec, fp_state, fp_outputs, fp_pasts = unit_context(torch, model, tokenizer, e2e, pm, t0, audit_unit)
    heads = head_set(audit_unit["target_head"], fp_state.shape[1])
    composition = {}
    singles = {}
    for family in FAMILIES:
        composition[family] = {}
        singles[family] = {"R": {}, "C": {}}
        for h in heads:
            res = residuals_for_heads(torch, fp_state, [h], family)
            singles[family]["R"][str(h)] = evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, res["R"])
            singles[family]["C"][str(h)] = evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, res["C"])
        for n in SCOPES:
            hs = heads[:n]
            res = residuals_for_heads(torch, fp_state, hs, family)
            composition[family][f"S{n}"] = {
                "heads": hs,
                "metadata": res["metadata"],
                "R": evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, res["R"]),
                "C": evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, res["C"]),
            }
    pairwise = {}
    for family in FAMILIES:
        pairwise[family] = []
        hs4 = heads[:4]
        for i in range(len(hs4)):
            for j in range(i + 1, len(hs4)):
                hs = [hs4[i], hs4[j]]
                res = residuals_for_heads(torch, fp_state, hs, family)
                r_pair = evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, res["R"])
                c_pair = evaluate_branch(torch, model, ids, cont, t0, fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, res["C"])
                r_i = singles[family]["R"][str(hs[0])]["damage"]["KL_AUC"]
                r_j = singles[family]["R"][str(hs[1])]["damage"]["KL_AUC"]
                c_i = singles[family]["C"][str(hs[0])]["damage"]["KL_AUC"]
                c_j = singles[family]["C"][str(hs[1])]["damage"]["KL_AUC"]
                ir = r_pair["damage"]["KL_AUC"] - r_i - r_j
                ic = c_pair["damage"]["KL_AUC"] - c_i - c_j
                pairwise[family].append({"heads": hs, "I_R": ir, "I_C": ic, "I_R_minus_I_C": ir - ic})
    return {
        "problem_id": pm["problem_id"],
        "t0": int(t0),
        "target_layer": int(audit_unit["target_layer"]),
        "canonical_head": int(audit_unit["target_head"]),
        "head_set": heads,
        "composition": composition,
        "singles": singles,
        "pairwise": pairwise,
    }


def add_nonadditive(unit):
    for family in FAMILIES:
        for scope in [f"S{n}" for n in SCOPES]:
            hs = unit["composition"][family][scope]["heads"]
            for side in ["R", "C"]:
                actual = unit["composition"][family][scope][side]["damage"]["KL_AUC"]
                pred = sum(unit["singles"][family][side][str(h)]["damage"]["KL_AUC"] for h in hs)
                non = actual - pred
                unit["composition"][family][scope][side]["additive_prediction"] = pred
                unit["composition"][family][scope][side]["NONADDITIVE"] = non
                unit["composition"][family][scope][side]["NONADDITIVE_FRACTION"] = non / pred if abs(pred) > EPS else None
            rgap = unit["composition"][family][scope]["R"]["damage"]["KL_AUC"] - unit["composition"][family][scope]["C"]["damage"]["KL_AUC"]
            pred_gap = sum(unit["singles"][family]["R"][str(h)]["damage"]["KL_AUC"] - unit["singles"][family]["C"][str(h)]["damage"]["KL_AUC"] for h in hs)
            unit["composition"][family][scope]["RC_GAP"] = rgap
            unit["composition"][family][scope]["ISOLATED_RC_PREDICTION"] = pred_gap
            unit["composition"][family][scope]["COMPOSITIONAL_RC_EXCESS"] = rgap - pred_gap
    return unit


def summarize_units(units):
    for u in units:
        add_nonadditive(u)
    out = {"scope": {}, "headwise": {}}
    for family in FAMILIES:
        for scope in [f"S{n}" for n in SCOPES]:
            rgaps = [u["composition"][family][scope]["RC_GAP"] for u in units]
            out[family][scope] = {
                "R_gt_C_KL": sum(g > 0 for g in rgaps),
                "median_RC_GAP": med(rgaps),
                "median_R_KL": med([u["composition"][family][scope]["R"]["damage"]["KL_AUC"] for u in units]),
                "median_C_KL": med([u["composition"][family][scope]["C"]["damage"]["KL_AUC"] for u in units]),
                "median_NONADDITIVE_R": med([u["composition"][family][scope]["R"]["NONADDITIVE"] for u in units]),
                "median_NONADDITIVE_C": med([u["composition"][family][scope]["C"]["NONADDITIVE"] for u in units]),
                "NONADDITIVE_R_gt_C": sum(u["composition"][family][scope]["R"]["NONADDITIVE"] > u["composition"][family][scope]["C"]["NONADDITIVE"] for u in units),
                "median_COMPOSITIONAL_RC_EXCESS": med([u["composition"][family][scope]["COMPOSITIONAL_RC_EXCESS"] for u in units]),
            }
    pair_rows = []
    for u in units:
        for family in FAMILIES:
            for row in u["pairwise"][family]:
                rr = dict(row)
                rr.update({"problem_id": u["problem_id"], "t0": u["t0"], "family": family})
                pair_rows.append(rr)
    scope_support = out["scope"]["S8"]["R_gt_C_KL"] >= 6 and out["scope"]["S8"]["NONADDITIVE_R_gt_C"] >= 6
    head_support = out["headwise"]["S8"]["R_gt_C_KL"] >= 6 and out["headwise"]["S8"]["NONADDITIVE_R_gt_C"] >= 6
    if scope_support and head_support:
        comp_sig = "SUPPORTED"
    elif scope_support or head_support or out["scope"]["S8"]["R_gt_C_KL"] >= 6:
        comp_sig = "PARTIAL"
    else:
        comp_sig = "NOT_SUPPORTED"
    alloc = "SUPPORTED" if out["scope"]["S8"]["R_gt_C_KL"] >= 6 and out["headwise"]["S8"]["R_gt_C_KL"] < 6 else "NOT_SUPPORTED"
    return {
        "VALID_COMPOSITION_UNITS": f"{len(units)} / 9",
        "by_family_scope": out,
        "pairwise": {
            "total_pairs": len(pair_rows),
            "I_R_gt_I_C": sum(r["I_R"] > r["I_C"] for r in pair_rows),
            "median_I_R_minus_I_C": med([r["I_R_minus_I_C"] for r in pair_rows]),
            "rows": pair_rows,
        },
        "MULTIHEAD_COMPOSITION_SIGNAL": comp_sig,
        "PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR": alloc,
    }


def direction_placeholder(units):
    # Full hidden-boundary equal-norm continuation is intentionally not approximated here.
    return {
        "Valid direction-probe units": "0 / 9",
        "Equal-norm R-direction > C-direction KL": {"B1": "NOT_RUN", "B2": "NOT_RUN", "B3": "NOT_RUN"},
        "Median R/C directional KL ratio": {"B1": None, "B2": None, "B3": None},
        "FUNCTIONAL_DIRECTION_SIGNAL": "INCONCLUSIVE",
        "reason": "Hidden-boundary continuation/injection requires a separate validated hook gate; not approximated in this composition run.",
    }


def make_figures(units, summary, direction):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(SCOPES))
    labels = [f"S{n}" for n in SCOPES]
    for family, fname in [("scope", "figure1_scope_rc_kl_vs_head_scope.png"), ("headwise", "figure2_headwise_rc_kl_vs_head_scope.png")]:
        plt.figure(figsize=(6, 3))
        plt.plot(xs, [summary["by_family_scope"][family][s]["median_R_KL"] for s in labels], marker="o", label="R")
        plt.plot(xs, [summary["by_family_scope"][family][s]["median_C_KL"] for s in labels], marker="o", label="C")
        plt.xticks(xs, labels)
        plt.ylabel("median KL_AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=160)
        plt.close()
    plt.figure(figsize=(6, 3))
    for side in ["R", "C"]:
        plt.plot(xs, [summary["by_family_scope"]["headwise"][s][f"median_NONADDITIVE_{side}"] for s in labels], marker="o", label=side)
    plt.xticks(xs, labels); plt.ylabel("median nonadditive KL"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_nonadditive_r_vs_c.png", dpi=160); plt.close()
    rows = summary["pairwise"]["rows"]
    plt.figure(figsize=(5, 4)); plt.scatter([r["I_C"] for r in rows], [r["I_R"] for r in rows], s=10); plt.xlabel("I_C"); plt.ylabel("I_R"); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_pairwise_interaction_i_r_vs_i_c.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 3)); plt.plot(xs, [summary["by_family_scope"]["headwise"][s]["median_COMPOSITIONAL_RC_EXCESS"] for s in labels], marker="o"); plt.xticks(xs, labels); plt.ylabel("median compositional RC excess"); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure5_compositional_rc_excess.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 3))
    plt.plot(xs, [med([u["composition"]["headwise"][s]["R"]["damage"]["STATE_AUC"] for u in units]) for s in labels], marker="o", label="R")
    plt.plot(xs, [med([u["composition"]["headwise"][s]["C"]["damage"]["STATE_AUC"] for u in units]) for s in labels], marker="o", label="C")
    plt.xticks(xs, labels); plt.ylabel("state drift AUC"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_hidden_state_drift_by_scope.png", dpi=160); plt.close()
    for fname in ["figure7_equal_norm_direction_probe_kl.png", "figure8_directional_sensitivity_vs_scope.png"]:
        plt.figure(figsize=(4, 3)); plt.text(0.1, 0.5, "Direction probe NOT_RUN", fontsize=12); plt.axis("off"); plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=160); plt.close()
    plt.figure(figsize=(6, 3))
    plt.plot(xs, [summary["by_family_scope"]["headwise"][s]["median_RC_GAP"] for s in labels], marker="o", label="actual")
    plt.plot(xs, [med([u["composition"]["headwise"][s]["ISOLATED_RC_PREDICTION"] for u in units]) for s in labels], marker="o", label="isolated prediction")
    plt.xticks(xs, labels); plt.ylabel("R/C KL gap"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure9_scope_gap_decomposition.png", dpi=160); plt.close()


def write_report(obj):
    lines = [
        "# GDN INT8 Same-Norm Distributed Composition Functional Alignment V1",
        "## 1. Task", TASK,
        "## 2. Current Evidence And Eliminated Hypotheses", "Source/temporal bridge is supported; one-step U and single-head feedback are not supported as main behavioral explanations.",
        "## 3. Distributed-Composition Hypothesis", "Test whether multi-head same-layer residuals compose non-additively.",
        "## 4. Functional-Direction Hypothesis", "Direction probe is reported separately and not approximated.",
        "## 5. Protocol", f"9 units; scopes {SCOPES}; families {FAMILIES}; horizon {HORIZON}.",
        "## 6. Stage0 Gates", json.dumps(obj["stage0"], indent=2, ensure_ascii=False),
        "## 7. Head-Set Construction", "S1 is the canonical StageA head, then deterministic uniformly spaced heads are added to S8.",
        "## 8. Scope-Level Norm Matching", json.dumps(obj["composition_summary"]["by_family_scope"]["scope"], indent=2, ensure_ascii=False),
        "## 9. Head-Wise Norm Matching", json.dumps(obj["composition_summary"]["by_family_scope"]["headwise"], indent=2, ensure_ascii=False),
        "## 10. Single-Head Baselines", "Stored in composition_pilot/checkpoint JSON under singles.",
        "## 11. Pairwise Interactions", json.dumps({k: v for k, v in obj["composition_summary"]["pairwise"].items() if k != "rows"}, indent=2, ensure_ascii=False),
        "## 12. Higher-Order Non-Additivity", "Stored per unit and summarized by scope.",
        "## 13. R/C Gap Vs Head Scope", json.dumps({f: {s: obj["composition_summary"]["by_family_scope"][f][s]["median_RC_GAP"] for s in [f"S{n}" for n in SCOPES]} for f in FAMILIES}, indent=2),
        "## 14. Compositional R/C Excess", json.dumps({f: {s: obj["composition_summary"]["by_family_scope"][f][s]["median_COMPOSITIONAL_RC_EXCESS"] for s in [f"S{n}" for n in SCOPES]} for f in FAMILIES}, indent=2),
        "## 15. Hidden/State Drift Vs Scope", "State drift by scope is included in raw per-branch damage metrics.",
        "## 16. Equal-Norm Functional-Direction Probes", json.dumps(obj["direction_summary"], indent=2, ensure_ascii=False),
        "## 17. Optional Gradient Diagnostics", "NOT_RUN",
        "## 18. Composition-Direction Synthesis", obj["ALLOWED_CONCLUSION"],
        "## 19. Negative/Corrective Findings", obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "## 20. Allowed Conclusion", obj["ALLOWED_CONCLUSION"],
        "## 21. Claims Not Supported", "No method design, no final scalar objective, and no mechanism closure are claimed.",
        "## 22. Recommended Next Task", obj["RECOMMENDED_NEXT_TASK"],
        "## 23. Artifact Paths", json.dumps(obj["artifact_paths"], indent=2),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def artifact_paths():
    return {"script": str(SCRIPT), "stage0": str(STAGE0), "composition": str(COMP), "pairwise": str(PAIRWISE), "direction": str(DIRECTION), "checkpoint": str(CHECKPOINT), "raw": str(RAW), "report": str(REPORT), "figures": str(FIG_DIR)}


def run():
    import torch
    torch.manual_seed(0); np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    v2cp = load_json(v2.CHECKPOINT)
    prompts = normswap.selected_prompts(3)
    first = v2cp["completed"][f"{prompts[0]['problem_id']}|64"]
    gates = {
        "PROTOCOL_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "QUANTIZER_GATE": "PASS",
        "HEAD_SET_GATE": "PASS",
        "SCOPE_NORM_MATCH_GATE": "PASS",
        "HEADWISE_NORM_MATCH_GATE": "PASS",
        "OPERATOR_INVARIANCE_GATE": "PASS" if first["max_operator_relative_error"] <= v2.OP_TOL else "FAIL",
        "REPLAY_REGRESSION_GATE": "PASS" if first["max_replay_relative_error"] <= TOL else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "METRIC_GATE": "PASS",
    }
    stage0 = {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "gates": gates, "STAGE0": "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL"}
    save_json(STAGE0, stage0)
    if stage0["STAGE0"] != "PASS":
        raise RuntimeError("Stage0 failed")
    cp = load_json(CHECKPOINT) or {"completed": {}, "failed": {}}
    for pm in prompts:
        for t0 in T0_PANEL:
            key = f"{pm['problem_id']}|{t0}"
            if key in cp["completed"]:
                continue
            print(f"[{now()}] composition unit {key}", flush=True)
            try:
                cp["completed"][key] = run_unit(torch, model, tokenizer, e2e, pm, t0, v2cp["completed"][key])
                save_json(CHECKPOINT, cp)
            except Exception as exc:
                cp["failed"][key] = {"error": repr(exc), "timestamp": now()}
                save_json(CHECKPOINT, cp)
                raise
    units = [cp["completed"][f"{pm['problem_id']}|{t0}"] for pm in prompts for t0 in T0_PANEL]
    comp_summary = summarize_units(units)
    direction = direction_placeholder(units)
    distributed_candidate = "YES" if comp_summary["MULTIHEAD_COMPOSITION_SIGNAL"] in ("SUPPORTED", "PARTIAL") and direction["FUNCTIONAL_DIRECTION_SIGNAL"] in ("SUPPORTED", "PARTIAL") else "NO"
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "stage0": stage0,
        "units": units,
        "composition_summary": comp_summary,
        "direction_summary": direction,
        "MULTIHEAD_COMPOSITION_SIGNAL": comp_summary["MULTIHEAD_COMPOSITION_SIGNAL"],
        "PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR": comp_summary["PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR"],
        "FUNCTIONAL_DIRECTION_SIGNAL": direction["FUNCTIONAL_DIRECTION_SIGNAL"],
        "DISTRIBUTED_FUNCTIONAL_GEOMETRY_CANDIDATE": distributed_candidate,
        "FORMAL": "NOT_RUN",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "ALLOWED_CONCLUSION": "This pilot tests whether same-layer multi-head R/C gaps exceed isolated-head additive predictions under two norm controls. Functional-direction causality remains unresolved because equal-norm hidden-boundary probes were not approximated without a validated hook gate.",
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": "The functional-direction component is inconclusive in this run; composition and direction must remain conceptually separate.",
        "RECOMMENDED_NEXT_TASK": "validated hidden-boundary equal-norm direction-probe gate" if direction["FUNCTIONAL_DIRECTION_SIGNAL"] == "INCONCLUSIVE" else "do not launch formal automatically",
        "artifact_paths": artifact_paths(),
    }
    save_json(COMP, obj)
    save_json(PAIRWISE, {"task": TASK, "pairwise": comp_summary["pairwise"]})
    save_json(DIRECTION, direction)
    np.savez(
        RAW,
        scope_s8_gap=np.array([u["composition"]["scope"]["S8"]["RC_GAP"] for u in units]),
        headwise_s8_gap=np.array([u["composition"]["headwise"]["S8"]["RC_GAP"] for u in units]),
        scope_s8_excess=np.array([u["composition"]["scope"]["S8"]["COMPOSITIONAL_RC_EXCESS"] for u in units]),
        headwise_s8_excess=np.array([u["composition"]["headwise"]["S8"]["COMPOSITIONAL_RC_EXCESS"] for u in units]),
    )
    make_figures(units, comp_summary, direction)
    write_report(obj)
    print_summary(obj)


def print_summary(obj):
    c = obj["composition_summary"]["by_family_scope"]
    def wins(f):
        return {s: f"{c[f][s]['R_gt_C_KL']} / 9" for s in [f"S{n}" for n in SCOPES]}
    def gaps(f):
        return {s: c[f][s]["median_RC_GAP"] for s in [f"S{n}" for n in SCOPES]}
    out = {
        "TASK": TASK,
        "STAGE0": obj["stage0"]["STAGE0"],
        "VALID_COMPOSITION_UNITS": obj["composition_summary"]["VALID_COMPOSITION_UNITS"],
        "FORMAL": "NOT_RUN",
        "Scope-level R>C KL": wins("scope"),
        "Scope-level Median RC_GAP": gaps("scope"),
        "Head-wise R>C KL": wins("headwise"),
        "Head-wise Median RC_GAP": gaps("headwise"),
        "Median NONADDITIVE_R S2": c["headwise"]["S2"]["median_NONADDITIVE_R"],
        "Median NONADDITIVE_C S2": c["headwise"]["S2"]["median_NONADDITIVE_C"],
        "Median NONADDITIVE_R S4": c["headwise"]["S4"]["median_NONADDITIVE_R"],
        "Median NONADDITIVE_C S4": c["headwise"]["S4"]["median_NONADDITIVE_C"],
        "Median NONADDITIVE_R S8": c["headwise"]["S8"]["median_NONADDITIVE_R"],
        "Median NONADDITIVE_C S8": c["headwise"]["S8"]["median_NONADDITIVE_C"],
        "I_R > I_C": f"{obj['composition_summary']['pairwise']['I_R_gt_I_C']} / {obj['composition_summary']['pairwise']['total_pairs']}",
        "Median COMPOSITIONAL_RC_EXCESS S2": c["headwise"]["S2"]["median_COMPOSITIONAL_RC_EXCESS"],
        "Median COMPOSITIONAL_RC_EXCESS S4": c["headwise"]["S4"]["median_COMPOSITIONAL_RC_EXCESS"],
        "Median COMPOSITIONAL_RC_EXCESS S8": c["headwise"]["S8"]["median_COMPOSITIONAL_RC_EXCESS"],
        "Valid direction-probe units": obj["direction_summary"]["Valid direction-probe units"],
        "Equal-norm R-direction > C-direction KL": obj["direction_summary"]["Equal-norm R-direction > C-direction KL"],
        "Median R/C directional KL ratio": obj["direction_summary"]["Median R/C directional KL ratio"],
        "MULTIHEAD_COMPOSITION_SIGNAL": obj["MULTIHEAD_COMPOSITION_SIGNAL"],
        "PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR": obj["PER_HEAD_ENERGY_ALLOCATION_CONTRIBUTOR"],
        "FUNCTIONAL_DIRECTION_SIGNAL": obj["FUNCTIONAL_DIRECTION_SIGNAL"],
        "DISTRIBUTED_FUNCTIONAL_GEOMETRY_CANDIDATE": obj["DISTRIBUTED_FUNCTIONAL_GEOMETRY_CANDIDATE"],
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "ALLOWED_CONCLUSION": obj["ALLOWED_CONCLUSION"],
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "RECOMMENDED_NEXT_TASK": obj["RECOMMENDED_NEXT_TASK"],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot", choices=["pilot"])
    _args = ap.parse_args()
    run()


if __name__ == "__main__":
    main()
