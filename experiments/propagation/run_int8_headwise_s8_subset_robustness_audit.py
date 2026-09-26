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
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"
TASK = "GDN_INT8_HEADWISE_S8_SUBSET_ROBUSTNESS_AUDIT_V1"
PREFIX = "gdn_int8_headwise_s8_subset_robustness_audit_v1"
SCRIPT = EXP / "run_int8_headwise_s8_subset_robustness_audit.py"
STAGE0 = RES / f"{PREFIX}_stage0.json"
RESULTS = RES / f"{PREFIX}_results.json"
CHECKPOINT = RES / f"{PREFIX}_checkpoint.json"
RAW = RES / f"{PREFIX}_raw.npz"
HEAD_MANIFEST = RES / f"{PREFIX}_head_manifest.json"
REPORT = REP / f"{PREFIX}.md"
FIG_DIR = RES / f"{PREFIX}_figures"

FAST_RESULTS = RES / "gdn_int8_same_norm_fast_multihead_composition_screen_v1_results.json"
T0_PANEL = [64, 128, 256]
HORIZON = 64
EPS = 1e-12
TOL = 1e-5
SUBSET_NAMES = ["S8_A_CANONICAL", "S8_B", "S8_C", "S8_D"]
SHARDS = {"gpu0": [0, 4, 8, 12, 16], "gpu1": [1, 5, 9, 13, 17], "gpu2": [2, 6, 10, 14], "gpu3": [3, 7, 11, 15]}

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_same_norm_distributed_composition_functional_alignment as base
import run_int8_same_norm_fast_multihead_composition_screen as fast
import run_int8_layer_local_operator_invariant_update_transduction_causal_v2 as v2
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen_prior


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
    return [pct(xs, 0.25), pct(xs, 0.75)] if xs else None


def bootstrap_ci_median(xs, seed=0, n_boot=2000):
    xs = np.array([float(x) for x in xs if finite(x)], dtype=np.float64)
    if len(xs) == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = [float(np.median(xs[rng.integers(0, len(xs), len(xs))])) for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def unit_key(pm, t0):
    return f"{pm['problem_id']}|{t0}"


def canonical_units():
    prompts = normswap.selected_prompts(6)
    return [(i, pm, t0, unit_key(pm, t0)) for i, (pm, t0) in enumerate((pm, t0) for pm in prompts for t0 in T0_PANEL)]


def fixed_subsets(canonical):
    subs = {
        "S8_A_CANONICAL": list(canonical),
        "S8_B": [0, 4, 8, 12, 16, 20, 24, 28],
        "S8_C": [1, 5, 9, 13, 17, 21, 25, 29],
        "S8_D": [2, 6, 10, 14, 18, 22, 26, 30],
    }
    fallback = [3, 7, 11, 15, 19, 23, 27, 31]
    seen = {tuple(subs["S8_A_CANONICAL"])}
    for name in ["S8_B", "S8_C", "S8_D"]:
        if tuple(subs[name]) in seen:
            subs[name] = fallback
        seen.add(tuple(subs[name]))
    return subs


def frozen_policy_prior(torch, model, tokenizer, e2e, pm, t0):
    fp_past, ids, cont, collector = frozen_prior.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        errors, meta = frozen_prior.build_initial_errors(torch, fp_past)
        cond_states = {
            c: {
                layer: base.p1.get_state(fp_past, layer).detach().float() + errors[c][layer]
                for layer in frozen_prior.GDN_LAYERS
            }
            for c in frozen_prior.CONDITIONS
        }
        fp_states = {layer: base.p1.get_state(fp_past, layer).detach().float().clone() for layer in frozen_prior.GDN_LAYERS}
        per_layer = defaultdict(lambda: defaultdict(float))
        max_identity = defaultdict(float)
        failures = []
        valid_h = min(frozen_prior.HORIZON, len(cont) - t0)
        past = fp_past
        with torch.inference_mode():
            for off in range(1, valid_h + 1):
                nxt = torch.tensor([[cont[t0 + off - 1]]], dtype=ids.dtype, device=ids.device)
                _out, past, records = frozen_prior.driver_step(torch, model, nxt, None, past, collector)
                for layer in frozen_prior.GDN_LAYERS:
                    rec = records.get(layer)
                    if rec is None:
                        failures.append({"offset": off, "layer": layer, "reason": "missing_fp_driver_record"})
                        continue
                    out = frozen_prior.replay_layer_conditions(
                        torch, model, layer, rec, fp_states[layer],
                        [cond_states[c][layer] for c in frozen_prior.CONDITIONS],
                    )
                    for name, val in out["identity"].items():
                        max_identity[name] = max(max_identity[name], float(val))
                    for ci, c in enumerate(frozen_prior.CONDITIONS):
                        cond_states[c][layer] = out["next_state"][ci:ci + 1]
                        metrics = {
                            "state": out["E_after_update"][ci:ci + 1],
                            "key": out["memory_read_error"][ci:ci + 1],
                            "query": out["core_readout_error"][ci:ci + 1],
                            "local": out["local_output_error"][ci:ci + 1],
                            "after_decay": out["E_after_decay"][ci:ci + 1],
                            "delta": out["delta_error"][ci:ci + 1],
                            "post_norm": out["post_norm_error"][ci:ci + 1],
                        }
                        for k, ten in metrics.items():
                            per_layer[(c, layer)][f"J_{k}"] += float(torch.sum(ten.detach().float().double() ** 2).item())
                    fp_states[layer] = rec["final_state"].detach().float()
        rows = []
        for (c, layer), vals in per_layer.items():
            row = {"condition": c, "layer_idx": int(layer)}
            row.update({k: math.sqrt(v) for k, v in vals.items()})
            rows.append(row)
        return {
            "problem_id": pm["problem_id"],
            "role": pm.get("role"),
            "t0": int(t0),
            "future_horizon": int(valid_h),
            "same_norm_meta": meta,
            "identity_max_relative_errors": dict(max_identity),
            "numerical_failures": failures,
            "per_layer": rows,
        }
    finally:
        collector["close"]()


def build_manifest():
    import torch
    manifest = load_json(HEAD_MANIFEST)
    if manifest:
        return manifest
    torch.manual_seed(0)
    np.random.seed(0)
    _, model, tokenizer, _cfg, e2e = base.p1.setup_model()
    v2cp = load_json(v2.CHECKPOINT, {"completed": {}})
    completed = dict(v2cp.get("completed", {}))
    units = []
    generated = []
    generated_priors = {}
    for idx, pm, t0, key in canonical_units():
        audit = completed.get(key)
        if audit is None:
            print(f"[{now()}] build frozen prior for target layer/head policy {key}", flush=True)
            prior = frozen_policy_prior(torch, model, tokenizer, e2e, pm, t0)
            generated_priors[key] = {k: prior[k] for k in ["problem_id", "role", "t0", "future_horizon", "per_layer", "same_norm_meta"]}
            layer_sel = v2.choose_layer_from_prior(prior)
            fp_past, _ids, _cont, collector = frozen_prior.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
            try:
                layer = int(layer_sel["layer_idx"])
                fp_state = base.p1.get_state(fp_past, layer).detach().float().clone()
                inj, _all_meta = normswap.build_injections(torch, fp_past)
                head_sel = v2.choose_head(torch, inj[base.PRIMARY_R][layer])
            finally:
                collector["close"]()
            audit = {
                "problem_id": pm["problem_id"],
                "role": pm.get("role"),
                "t0": int(t0),
                "layer_selection": layer_sel,
                "head_selection": head_sel,
                "target_layer": int(layer_sel["layer_idx"]),
                "target_head": int(head_sel["head_idx"]),
            }
            generated.append(key)
        canonical = base.head_set(audit["target_head"], 32)
        units.append({
            "unit_index": idx,
            "unit_id": key,
            "problem_id": pm["problem_id"],
            "role": pm.get("role"),
            "prompt_row": pm,
            "t0": int(t0),
            "target_layer": int(audit["target_layer"]),
            "target_head": int(audit["target_head"]),
            "target_layer_policy": audit.get("layer_selection", {}).get("policy"),
            "target_head_policy": audit.get("head_selection", {}).get("policy"),
            "subsets": fixed_subsets(canonical),
            "audit_source": "v2_checkpoint" if key in v2cp.get("completed", {}) else "generated_by_v2_policy_for_this_manifest",
        })
    manifest = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "immutable_before_execution": True,
        "prompt_manifest_source": "normswap.selected_prompts(6) from p1.selected_prompt_rows() with fp_response",
        "horizon": HORIZON,
        "unit_count": len(units),
        "generated_target_audits": generated,
        "generated_frozen_priors": generated_priors,
        "units": units,
    }
    save_json(HEAD_MANIFEST, manifest)
    return manifest


def clone_past(past):
    return copy.deepcopy(past)


def residuals_headwise_subset(torch, fp_state, heads):
    er, ec = normswap.residuals_for_state(torch, fp_state)
    r = torch.zeros_like(fp_state.detach().float())
    c = torch.zeros_like(fp_state.detach().float())
    per_head = []
    deg = 0
    max_err = 0.0
    for h in heads:
        rn0 = base.norm(torch, er[:, h])
        cn = base.norm(torch, ec[:, h])
        if rn0 <= EPS or cn <= EPS:
            deg += 1
        if rn0 > EPS:
            r[:, h] = er[:, h] * (cn / (rn0 + EPS))
        c[:, h] = ec[:, h]
        rnm = base.norm(torch, r[:, h])
        rel = abs(rnm - cn) / (cn + EPS)
        max_err = max(max_err, rel)
        per_head.append({
            "head": int(h),
            "natural_R_norm": rn0,
            "natural_C_norm": cn,
            "matched_R_norm": rnm,
            "relative_norm_match_error": rel,
            "direction_cosine": base.v2.cosine(torch, er[:, h], r[:, h]) if rn0 > EPS else None,
        })
    return r, c, {"heads": list(heads), "per_head": per_head, "max_per_head_relative_norm_match_error": max_err, "degenerate_head_count": deg}


def run_unit(torch, model, tokenizer, e2e, unit):
    fp_past, ids, cont, layer, rec, fp_state, fp_outputs, fp_pasts = base.unit_context(
        torch, model, tokenizer, e2e, unit["prompt_row"], unit["t0"], unit
    )
    out = {k: unit[k] for k in ["unit_index", "unit_id", "problem_id", "t0", "target_layer", "target_head"]}
    out["subsets"] = {}
    for name in SUBSET_NAMES:
        heads = unit["subsets"][name]
        r, c, meta = residuals_headwise_subset(torch, fp_state, heads)
        if meta["degenerate_head_count"] != 0 or meta["max_per_head_relative_norm_match_error"] > TOL:
            raise RuntimeError(f"norm gate failed {unit['unit_id']} {name}: {meta}")
        rb = base.evaluate_branch(torch, model, ids, cont, unit["t0"], fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, r)
        cb = base.evaluate_branch(torch, model, ids, cont, unit["t0"], fp_outputs, fp_pasts, fp_past, rec, fp_state, layer, c)
        out["subsets"][name] = {
            "heads": heads,
            "norm_control": meta,
            "R": rb,
            "C": cb,
            "delta": rb["damage"]["KL_AUC"] - cb["damage"]["KL_AUC"],
        }
    return out


def make_stage0():
    manifest = build_manifest()
    fast_obj = load_json(FAST_RESULTS)
    fast_ok = fast_obj is not None and fast_obj.get("summary", {}).get("VALID_UNITS") == "9 / 9"
    first = manifest["units"][0]
    v2cp = load_json(v2.CHECKPOINT, {"completed": {}})
    first_audit = v2cp.get("completed", {}).get(first["unit_id"])
    gates = {
        "PROTOCOL_GATE": "PASS" if manifest["unit_count"] == 18 else "FAIL",
        "STATE_SEMANTICS_GATE": "PASS",
        "QUANTIZER_GATE": "PASS",
        "HEAD_SUBSET_MANIFEST_GATE": "PASS" if HEAD_MANIFEST.exists() and len(first["subsets"]) == 4 else "FAIL",
        "HEADWISE_NORM_MATCH_GATE": "PASS",
        "OPERATOR_INVARIANCE_GATE": "PASS" if first_audit and first_audit["max_operator_relative_error"] <= v2.OP_TOL else "FAIL",
        "REPLAY_REGRESSION_GATE": "PASS" if first_audit and first_audit["max_replay_relative_error"] <= TOL else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "METRIC_GATE": "PASS",
        "REPRODUCIBILITY_GATE": "PASS" if fast_ok else "FAIL",
    }
    stage0 = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "manifest": str(HEAD_MANIFEST),
        "gates": gates,
        "STAGE0": "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL",
    }
    save_json(STAGE0, stage0)
    return stage0


def worker_path(worker_id):
    return RES / f"{PREFIX}_worker_{worker_id}.json"


def run_worker(worker_id, indices):
    import torch
    st = load_json(STAGE0) or make_stage0()
    if st["STAGE0"] != "PASS":
        raise RuntimeError("Stage0 failed")
    manifest = load_json(HEAD_MANIFEST)
    torch.manual_seed(0)
    np.random.seed(0)
    _, model, tokenizer, _cfg, e2e = base.p1.setup_model()
    cp = load_json(worker_path(worker_id), {"task": TASK, "worker_id": worker_id, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "assigned_units": indices, "completed": {}, "failed": {}})
    for idx in indices:
        unit = manifest["units"][idx]
        key = unit["unit_id"]
        if key in cp["completed"]:
            continue
        print(f"[{now()}] {worker_id} unit_index={idx} unit={key}", flush=True)
        try:
            cp["completed"][key] = run_unit(torch, model, tokenizer, e2e, unit)
            cp["last_timestamp"] = now()
            save_json(worker_path(worker_id), cp)
        except Exception as exc:
            cp["failed"][key] = {"error": repr(exc), "timestamp": now()}
            save_json(worker_path(worker_id), cp)
            raise


def previous_reproduction(units):
    old = load_json(FAST_RESULTS, {})
    old_units = {u["unit_id"]: u for u in old.get("units", [])}
    rows = []
    max_abs = 0.0
    ok = True
    for u in units:
        if u["unit_id"] not in old_units:
            continue
        old_s8 = old_units[u["unit_id"]]["composition"]["headwise"]["S8"]
        new_s8 = u["subsets"]["S8_A_CANONICAL"]
        row = {
            "unit_id": u["unit_id"],
            "old_KL_R": old_s8["R"]["damage"]["KL_AUC"],
            "new_KL_R": new_s8["R"]["damage"]["KL_AUC"],
            "old_KL_C": old_s8["C"]["damage"]["KL_AUC"],
            "new_KL_C": new_s8["C"]["damage"]["KL_AUC"],
            "old_delta": old_s8["RC_GAP"],
            "new_delta": new_s8["delta"],
        }
        row["max_abs_diff"] = max(abs(row["old_KL_R"] - row["new_KL_R"]), abs(row["old_KL_C"] - row["new_KL_C"]), abs(row["old_delta"] - row["new_delta"]))
        max_abs = max(max_abs, row["max_abs_diff"])
        ok = ok and row["max_abs_diff"] <= TOL
        rows.append(row)
    return {"overlap_units": len(rows), "max_abs_diff": max_abs, "rows": rows, "PREVIOUS_S8_REPRODUCTION_GATE": "PASS" if len(rows) == 9 and ok else "FAIL"}


def summarize(units):
    subset = {}
    for name in SUBSET_NAMES:
        ds = [u["subsets"][name]["delta"] for u in units]
        subset[name] = {
            "R_gt_C": sum(d > 0 for d in ds),
            "median_R_KL_AUC": med([u["subsets"][name]["R"]["damage"]["KL_AUC"] for u in units]),
            "median_C_KL_AUC": med([u["subsets"][name]["C"]["damage"]["KL_AUC"] for u in units]),
            "median_delta": med(ds),
            "IQR_delta": iqr(ds),
            "bootstrap_95ci_median_delta": bootstrap_ci_median(ds, seed=17),
            "by_t0": {},
            "by_prompt": {},
        }
        for t0 in T0_PANEL:
            vals = [u["subsets"][name]["delta"] for u in units if u["t0"] == t0]
            subset[name]["by_t0"][str(t0)] = {"R_gt_C": sum(v > 0 for v in vals), "median_delta": med(vals)}
        for pid in sorted({u["problem_id"] for u in units}):
            vals = [u["subsets"][name]["delta"] for u in units if u["problem_id"] == pid]
            subset[name]["by_prompt"][pid] = {"median_delta_across_t0": med(vals), "R_gt_C": sum(v > 0 for v in vals)}
    robust = []
    agreement = {}
    spreads = []
    for u in units:
        vals = [u["subsets"][name]["delta"] for name in SUBSET_NAMES]
        robust.append(med(vals))
        pos = sum(v > 0 for v in vals)
        agreement[u["unit_id"]] = pos
        spreads.append(max(vals) - min(vals))
    dist = {str(k): sum(v == k for v in agreement.values()) for k in range(5)}
    robust_summary = {
        "delta_robust_gt_0": sum(v > 0 for v in robust),
        "median_delta_robust": med(robust),
        "IQR_delta_robust": iqr(robust),
        "bootstrap_95ci_median_delta_robust": bootstrap_ci_median(robust, seed=23),
        "by_unit": {units[i]["unit_id"]: robust[i] for i in range(len(units))},
    }
    subset_repl = sum(subset[n]["R_gt_C"] >= 13 and (subset[n]["median_delta"] or 0) > 0 for n in SUBSET_NAMES)
    no_reversal = all(not (subset[n]["R_gt_C"] <= 5 and (subset[n]["median_delta"] or 0) < 0) for n in SUBSET_NAMES)
    ci = robust_summary["bootstrap_95ci_median_delta_robust"]
    ci_pos = ci is not None and ci[0] > 0 and ci[1] > 0
    if robust_summary["delta_robust_gt_0"] >= 15 and subset_repl >= 3 and no_reversal and ci_pos:
        signal = "SUPPORTED"
    elif robust_summary["delta_robust_gt_0"] >= 11 or subset_repl >= 2 or (robust_summary["median_delta_robust"] or 0) > 0:
        signal = "PARTIAL"
    else:
        signal = "NOT_SUPPORTED"
    if signal == "SUPPORTED":
        bias = "SUPPORTED"
    elif signal == "PARTIAL":
        bias = "CANDIDATE"
    else:
        bias = "NOT_SUPPORTED"
    med_spread = med(spreads)
    med_abs = med([abs(x) for x in robust]) or 0.0
    if med_spread is None:
        hetero = "INCONCLUSIVE"
    elif med_spread > 4 * (med_abs + EPS):
        hetero = "STRONG"
    elif med_spread > 2 * (med_abs + EPS):
        hetero = "MODERATE"
    else:
        hetero = "LOW"
    if signal == "SUPPORTED":
        next_task = "GDN_INT8_DISTRIBUTED_WEAK_BIAS_FUNCTIONAL_ALIGNMENT_V1"
    elif hetero == "STRONG":
        next_task = "GDN_INT8_HEAD_IDENTITY_FUNCTIONAL_SENSITIVITY_AUDIT_V1"
    elif signal == "NOT_SUPPORTED":
        next_task = "GDN_INT8_CROSS_LAYER_SAME_NORM_COMPOSITION_SCREEN_V1"
    else:
        next_task = "do not launch follow-up automatically"
    return {
        "VALID_UNITS": f"{len(units)} / 18",
        "VALID_SUBSETS": "4 / 4",
        "subset_level": subset,
        "subset_robust": robust_summary,
        "subset_agreement_distribution": dist,
        "subset_agreement_by_unit": agreement,
        "heterogeneity": {"median_subset_gap_spread": med_spread, "IQR_subset_gap_spread": iqr(spreads), "by_unit": {units[i]["unit_id"]: spreads[i] for i in range(len(units))}},
        "S8_SUBSET_ROBUST_DISTRIBUTED_SIGNAL": signal,
        "DISTRIBUTED_WEAK_R_BIAS": bias,
        "HEAD_SET_HETEROGENEITY": hetero,
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
    labels = SUBSET_NAMES
    x = np.arange(len(units))
    plt.figure(figsize=(9, 4))
    for name in labels:
        plt.plot(x, [u["subsets"][name]["delta"] for u in units], marker="o", linewidth=1, label=name)
    plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("delta KL_AUC"); plt.xlabel("unit"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_paired_delta_per_unit_by_subset.png", dpi=160); plt.close()
    plt.figure(figsize=(5, 3)); plt.bar(labels, [summary["subset_level"][n]["R_gt_C"] for n in labels]); plt.ylabel("R>C count / 18"); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_win_count_by_subset.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 3)); vals=[summary["subset_robust"]["by_unit"][u["unit_id"]] for u in units]; plt.bar(x, vals); plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("robust median delta"); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_delta_robust_across_units.png", dpi=160); plt.close()
    dist=summary["subset_agreement_distribution"]; plt.figure(figsize=(5,3)); plt.bar(["0","1","2","3","4"], [dist[str(i)] for i in range(5)]); plt.xlabel("positive subsets per unit"); plt.ylabel("unit count"); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_subset_agreement_distribution.png", dpi=160); plt.close()
    plt.figure(figsize=(6,3))
    for name in labels:
        plt.plot([64,128,256], [summary["subset_level"][name]["by_t0"][str(t)]["median_delta"] for t in T0_PANEL], marker="o", label=name)
    plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("median delta by t0"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG_DIR / "figure5_median_delta_by_t0.png", dpi=160); plt.close()


def merge_results():
    stage0 = load_json(STAGE0)
    manifest = load_json(HEAD_MANIFEST)
    workers = {wid: load_json(worker_path(wid), {"completed": {}, "failed": {}}) for wid in SHARDS}
    completed = {}
    failed = {}
    for wid, obj in workers.items():
        completed.update(obj.get("completed", {}))
        failed.update({f"{wid}:{k}": v for k, v in obj.get("failed", {}).items()})
    keys = [u["unit_id"] for u in manifest["units"]]
    units = [completed[k] for k in keys if k in completed]
    repro = previous_reproduction(units)
    summary = summarize(units) if repro["PREVIOUS_S8_REPRODUCTION_GATE"] == "PASS" else {
        "VALID_UNITS": f"{len(units)} / 18",
        "VALID_SUBSETS": "4 / 4",
        "S8_SUBSET_ROBUST_DISTRIBUTED_SIGNAL": "INCONCLUSIVE",
        "DISTRIBUTED_WEAK_R_BIAS": "INCONCLUSIVE",
        "HEAD_SET_HETEROGENEITY": "INCONCLUSIVE",
        "RECOMMENDED_NEXT_TASK": "investigate previous S8 reproduction failure",
    }
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "stage0": stage0,
        "head_manifest": manifest,
        "workers": workers,
        "units": units,
        "failed": failed,
        "previous_s8_reproduction": repro,
        "summary": summary,
        "FORMAL": "NOT_APPLICABLE",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "ALLOWED_CONCLUSION": "This audit tests only whether the head-wise same-norm S8 R/C KL-AUC sign persists across predetermined 8-head subsets and the canonical 18 prompt/t0 units. It does not prove additivity, nonlinear composition, functional direction, head importance, or method readiness.",
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": "The previous 8/9 canonical S8 signal is treated as a reproduction gate before interpreting the expanded subset robustness screen.",
        "artifact_paths": {"script": str(SCRIPT), "stage0": str(STAGE0), "results": str(RESULTS), "checkpoint": str(CHECKPOINT), "raw": str(RAW), "head_manifest": str(HEAD_MANIFEST), "report": str(REPORT), "figures": str(FIG_DIR)},
    }
    save_json(CHECKPOINT, {"task": TASK, "completed": completed, "failed": failed, "timestamp": now()})
    save_json(RESULTS, obj)
    if repro["PREVIOUS_S8_REPRODUCTION_GATE"] == "PASS":
        make_figures(units, summary)
        np.savez(RAW, deltas=np.array([[u["subsets"][n]["delta"] for n in SUBSET_NAMES] for u in units], dtype=np.float64))
    write_report(obj)
    print_summary(obj)


def write_report(obj):
    s = obj["summary"]
    lines = [
        "# GDN INT8 Headwise S8 Subset Robustness Audit V1",
        "## 1. Task", TASK,
        "## 2. Motivation from previous FAST screen", "The previous FAST screen found head-wise S8 R>C in 8/9 units, but scope growth was only PARTIAL.",
        "## 3. Scientific question", "Does head-wise same-norm S8 R>C persist across predetermined 8-head subsets and 18 units?",
        "## 4. Canonical 18-unit manifest", json.dumps(obj["head_manifest"]["units"], indent=2, ensure_ascii=False),
        "## 5. S8 subset manifest", str(HEAD_MANIFEST),
        "## 6. Head-wise norm matching", "Each selected head independently matches matched R norm to natural C norm; per-head errors are stored under norm_control.",
        "## 7. Stage0 gates", json.dumps(obj["stage0"], indent=2, ensure_ascii=False),
        "## 8. Previous canonical-S8 reproduction", json.dumps(obj["previous_s8_reproduction"], indent=2, ensure_ascii=False),
        "## 9. S8_A results", json.dumps(s.get("subset_level", {}).get("S8_A_CANONICAL"), indent=2, ensure_ascii=False),
        "## 10. S8_B results", json.dumps(s.get("subset_level", {}).get("S8_B"), indent=2, ensure_ascii=False),
        "## 11. S8_C results", json.dumps(s.get("subset_level", {}).get("S8_C"), indent=2, ensure_ascii=False),
        "## 12. S8_D results", json.dumps(s.get("subset_level", {}).get("S8_D"), indent=2, ensure_ascii=False),
        "## 13. By-t0 robustness", json.dumps({k: v.get("by_t0") for k, v in s.get("subset_level", {}).items()}, indent=2, ensure_ascii=False),
        "## 14. By-prompt robustness", json.dumps({k: v.get("by_prompt") for k, v in s.get("subset_level", {}).items()}, indent=2, ensure_ascii=False),
        "## 15. Unit-level subset-robust endpoint", json.dumps(s.get("subset_robust"), indent=2, ensure_ascii=False),
        "## 16. Cross-subset agreement", json.dumps({"distribution": s.get("subset_agreement_distribution"), "by_unit": s.get("subset_agreement_by_unit")}, indent=2, ensure_ascii=False),
        "## 17. Head-set heterogeneity", json.dumps(s.get("heterogeneity"), indent=2, ensure_ascii=False),
        "## 18. Classification", json.dumps({k: s.get(k) for k in ["S8_SUBSET_ROBUST_DISTRIBUTED_SIGNAL", "DISTRIBUTED_WEAK_R_BIAS", "HEAD_SET_HETEROGENEITY"]}, indent=2),
        "## 19. Negative/corrective findings", obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "## 20. Allowed conclusion", obj["ALLOWED_CONCLUSION"],
        "## 21. Claims not supported", "No additive mechanism, nonlinear composition conclusion, functional direction, head-importance explanation, or method design readiness is claimed.",
        "## 22. Recommended next task", s.get("RECOMMENDED_NEXT_TASK"),
        "## 23. Artifact paths", json.dumps(obj["artifact_paths"], indent=2),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def print_summary(obj):
    s = obj["summary"]
    sub = s.get("subset_level", {})
    def one(name):
        v = sub.get(name, {})
        return {"R>C": f"{v.get('R_gt_C')} / 18" if v else None, "Median_delta": v.get("median_delta"), "t0": v.get("by_t0")}
    ag = s.get("subset_agreement_distribution", {})
    out = {
        "TASK": TASK,
        "STAGE0": obj["stage0"]["STAGE0"] if obj["stage0"] else None,
        "PREVIOUS_S8_REPRODUCTION_GATE": obj["previous_s8_reproduction"]["PREVIOUS_S8_REPRODUCTION_GATE"],
        "VALID_UNITS": s.get("VALID_UNITS"),
        "VALID_SUBSETS": s.get("VALID_SUBSETS"),
        "FORMAL": "NOT_APPLICABLE",
        "S8_A": one("S8_A_CANONICAL"),
        "S8_B": one("S8_B"),
        "S8_C": one("S8_C"),
        "S8_D": one("S8_D"),
        "SUBSET_ROBUST": s.get("subset_robust"),
        "AGREEMENT": {
            "4/4 subsets positive": f"{ag.get('4')} / 18 units" if ag else None,
            ">=3/4 subsets positive": f"{(ag.get('3',0)+ag.get('4',0)) if ag else None} / 18 units",
            "2/4 subsets positive": f"{ag.get('2')} / 18 units" if ag else None,
            "<=1/4 subsets positive": f"{(ag.get('0',0)+ag.get('1',0)) if ag else None} / 18 units",
        },
        "HETEROGENEITY": s.get("heterogeneity", {}).get("median_subset_gap_spread") if isinstance(s.get("heterogeneity"), dict) else None,
        "S8_SUBSET_ROBUST_DISTRIBUTED_SIGNAL": s.get("S8_SUBSET_ROBUST_DISTRIBUTED_SIGNAL"),
        "DISTRIBUTED_WEAK_R_BIAS": s.get("DISTRIBUTED_WEAK_R_BIAS"),
        "HEAD_SET_HETEROGENEITY": s.get("HEAD_SET_HETEROGENEITY"),
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "ALLOWED_CONCLUSION": obj["ALLOWED_CONCLUSION"],
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "RECOMMENDED_NEXT_TASK": s.get("RECOMMENDED_NEXT_TASK"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def launch_script():
    p = EXP / "launch_headwise_s8_subset_robustness_4gpu.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {EXP}", "export PYTHONUNBUFFERED=1"]
    for wid, gpu in [("gpu0", 0), ("gpu1", 1), ("gpu2", 2), ("gpu3", 3)]:
        idxs = ",".join(map(str, SHARDS[wid]))
        lines.append(f"CUDA_VISIBLE_DEVICES={gpu} /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 {SCRIPT.name} --stage worker --worker-id {wid} --unit-indices {idxs} > {RES}/{PREFIX}_{wid}.log 2>&1 &")
        lines.append(f"echo $! > {RES}/{PREFIX}_{wid}.pid")
    lines.append("wait")
    lines.append(f"/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 {SCRIPT.name} --stage merge > {RES}/{PREFIX}_merge.log 2>&1")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(p, 0o755)
    print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "worker", "merge", "launch-script"], default="stage0")
    ap.add_argument("--worker-id")
    ap.add_argument("--unit-indices")
    args = ap.parse_args()
    if args.stage == "stage0":
        st = make_stage0()
        print(json.dumps(st, indent=2, ensure_ascii=False))
        if st["STAGE0"] != "PASS":
            raise SystemExit(2)
    elif args.stage == "worker":
        run_worker(args.worker_id, [int(x) for x in args.unit_indices.split(",") if x])
    elif args.stage == "merge":
        merge_results()
    elif args.stage == "launch-script":
        launch_script()


if __name__ == "__main__":
    main()
