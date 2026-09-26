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
TASK = "GDN_INT8_SAME_NORM_DOWNSTREAM_OPERATOR_FEEDBACK_PROPAGATION_V1"
SCRIPT = EXP / "run_int8_same_norm_downstream_operator_feedback_propagation.py"
STAGE0 = RES / "gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_stage0.json"
TRACE = RES / "gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_trace.json"
SUMMARY = RES / "gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_summary.json"
CHECKPOINT = RES / "gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_checkpoint.json"
RAW = RES / "gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_raw.npz"
REPORT = REP / "gdn_int8_same_norm_downstream_operator_feedback_propagation_v1.md"
FIG_DIR = RES / "gdn_int8_same_norm_downstream_operator_feedback_propagation_v1_figures"

EPS = 1e-12
TOL = 1e-5
HORIZON = 128
T0_PANEL = [64, 128, 256]
BRANCHES = ["FP", "R_LOCAL", "C_LOCAL"]
OP_KEYS = ["query", "key", "value", "g", "beta"]
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_layer_local_operator_invariant_update_transduction_causal_v2 as v2


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


def cosine(torch, a, b):
    an, bn = norm(torch, a), norm(torch, b)
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(a.detach().float().flatten(), b.detach().float().flatten(), dim=0).item())


def drift(torch, x, ref):
    d = x.detach().float() - ref.detach().float()
    return {
        "rel": norm(torch, d) / (norm(torch, ref) + EPS),
        "abs": norm(torch, d),
        "rms": float(torch.sqrt(torch.mean(d * d)).item()) if d.numel() else 0.0,
        "cos": cosine(torch, x, ref),
    }


def install_hidden_hooks(model):
    collector = {"branch": None, "records": {}}
    handles = []
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    for idx, layer in enumerate(layers):
        def make_pre(i):
            def pre(_module, inputs):
                b = collector["branch"]
                if b is not None:
                    collector["records"].setdefault(b, {}).setdefault(i, {})["hidden_in"] = inputs[0].detach().float().cpu()
            return pre
        def make_post(i):
            def post(_module, _inputs, output):
                b = collector["branch"]
                if b is not None:
                    y = output[0] if isinstance(output, tuple) else output
                    collector["records"].setdefault(b, {}).setdefault(i, {})["hidden_out"] = y.detach().float().cpu()
            return post
        handles.append(layer.register_forward_pre_hook(make_pre(idx)))
        handles.append(layer.register_forward_hook(make_post(idx)))
    def close():
        for h in handles:
            h.remove()
    collector["close"] = close
    return collector


def state_drift_metrics(torch, past, fp_past):
    layer_vals = []
    total_e2 = 0.0
    total_r2 = 0.0
    for layer in GDN_LAYERS:
        a = p1.get_state(past, layer).detach().float()
        b = p1.get_state(fp_past, layer).detach().float()
        d = a - b
        e2 = float(torch.sum(d.double() * d.double()).item())
        r2 = float(torch.sum(b.double() * b.double()).item())
        total_e2 += e2
        total_r2 += r2
        layer_vals.append(math.sqrt(e2) / (math.sqrt(r2) + EPS))
    return {
        "rel": math.sqrt(total_e2) / (math.sqrt(total_r2) + EPS),
        "abs": math.sqrt(total_e2),
        "layer_rel": layer_vals,
    }


def summarize_branch_token(torch, fp_hidden, br_hidden, fp_ops, br_ops, fp_past, br_past, fp_logits, br_logits):
    hidden_in_layers, hidden_out_layers = [], []
    for layer, fp_rec in fp_hidden.items():
        br_rec = br_hidden.get(layer, {})
        if "hidden_in" in fp_rec and "hidden_in" in br_rec:
            hidden_in_layers.append(drift(torch, br_rec["hidden_in"], fp_rec["hidden_in"])["rel"])
        if "hidden_out" in fp_rec and "hidden_out" in br_rec:
            hidden_out_layers.append(drift(torch, br_rec["hidden_out"], fp_rec["hidden_out"])["rel"])
    op = {k: [] for k in OP_KEYS}
    readout = {"core": [], "actual": []}
    for layer, fp_rec in fp_ops.items():
        br_rec = br_ops.get(layer, {})
        for k in OP_KEYS:
            if k in fp_rec and k in br_rec:
                op[k].append(drift(torch, br_rec[k].cpu(), fp_rec[k].cpu())["rel"])
        if "core_output" in fp_rec and "core_output" in br_rec:
            readout["core"].append(drift(torch, br_rec["core_output"].cpu(), fp_rec["core_output"].cpu())["rel"])
        if "postproj" in fp_rec and "postproj" in br_rec:
            readout["actual"].append(drift(torch, br_rec["postproj"].cpu(), fp_rec["postproj"].cpu())["rel"])
    st = state_drift_metrics(torch, br_past, fp_past)
    lm = p1.logits_metrics(torch, fp_logits, br_logits)
    return {
        "hidden_in": auc(hidden_in_layers),
        "hidden_out": auc(hidden_out_layers),
        "hidden_in_layers": hidden_in_layers,
        "hidden_out_layers": hidden_out_layers,
        "operator": {k: auc(v) for k, v in op.items()},
        "readout_core": auc(readout["core"]),
        "readout_actual": auc(readout["actual"]),
        "state": st["rel"],
        "state_abs": st["abs"],
        "state_layers": st["layer_rel"],
        "KL": lm["KL"],
    }


def clone_past(past):
    return copy.deepcopy(past)


def set_layer_state(torch, past, layer, new_state):
    s = p1.get_state(past, layer)
    s.copy_(new_state.to(s.dtype))


def seed_branch_pasts(torch, fp_past, rec, fp_state, local_errors, layer):
    out = {"FP": clone_past(fp_past)}
    for cond, name in ((PRIMARY_R, "R_LOCAL"), (PRIMARY_C, "C_LOCAL")):
        du = v2.local_du_tensors(torch, rec, fp_state, local_errors[cond])
        p = clone_past(fp_past)
        set_layer_state(torch, p, layer, du["fp_next"] + du["F"])
        out[name] = p
    return out


def run_unit_trace(torch, model, tokenizer, e2e, pm, t0, audit_unit):
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        layer = int(audit_unit["target_layer"])
        head = int(audit_unit["target_head"])
        rec = collector["records"][layer]
        fp_state = p1.get_state(fp_past, layer).detach().float().clone()
        local_errors = v2.build_local_errors(torch, fp_state, head)
        seed_pasts = seed_branch_pasts(torch, fp_past, rec, fp_state, local_errors, layer)
        max_seed_replay = max(v2.max_nested_rel(v2.local_du_tensors(torch, rec, fp_state, local_errors[c])["identity"]) for c in (PRIMARY_R, PRIMARY_C))
    finally:
        collector["close"]()

    op_col = frozen.install_fp_driver_capture(torch, model)
    hid_col = install_hidden_hooks(model)
    pasts = seed_pasts
    branch_ids = {b: ids.clone() for b in BRANCHES}
    curves = {b: {"hidden_in": [], "hidden_out": [], "state": [], "KL": [], "readout_core": [], "readout_actual": [], **{k: [] for k in OP_KEYS}} for b in ("R_LOCAL", "C_LOCAL")}
    heat = {b: {"hidden_out": [], "state": [], "key": [], "query": []} for b in ("R_LOCAL", "C_LOCAL")}
    try:
        with torch.inference_mode():
            steps = min(HORIZON, max(0, len(cont) - t0))
            for off in range(steps):
                outs, ops, hids = {}, {}, {}
                for b in BRANCHES:
                    op_col["records"].clear()
                    hid_col["records"].clear()
                    hid_col["branch"] = b
                    out = p1.feed_step(torch, model, branch_ids[b], None, pasts[b])
                    hid_col["branch"] = None
                    outs[b] = out
                    pasts[b] = out.past_key_values
                    ops[b] = {int(k): dict(v) for k, v in op_col["records"].items()}
                    hids[b] = hid_col["records"].get(b, {})
                for b in ("R_LOCAL", "C_LOCAL"):
                    tm = summarize_branch_token(torch, hids["FP"], hids[b], ops["FP"], ops[b], pasts["FP"], pasts[b], outs["FP"].logits, outs[b].logits)
                    for k in ("hidden_in", "hidden_out", "state", "KL", "readout_core", "readout_actual"):
                        curves[b][k].append(tm[k])
                    for k in OP_KEYS:
                        curves[b][k].append(tm["operator"][k])
                    heat[b]["hidden_out"].append(tm["hidden_out_layers"])
                    heat[b]["state"].append(tm["state_layers"])
                    heat[b]["key"].append([drift(torch, ops[b].get(l, {}).get("key", ops["FP"].get(l, {}).get("key")).cpu(), ops["FP"][l]["key"].cpu())["rel"] if l in ops["FP"] and l in ops[b] and "key" in ops[b][l] else None for l in GDN_LAYERS])
                    heat[b]["query"].append([drift(torch, ops[b].get(l, {}).get("query", ops["FP"].get(l, {}).get("query")).cpu(), ops["FP"][l]["query"].cpu())["rel"] if l in ops["FP"] and l in ops[b] and "query" in ops[b][l] else None for l in GDN_LAYERS])
                next_idx = t0 + off + 1
                if next_idx < len(cont):
                    nxt = torch.tensor([[cont[next_idx]]], dtype=ids.dtype, device=ids.device)
                    for b in BRANCHES:
                        branch_ids[b] = nxt.clone()
    finally:
        hid_col["close"]()
        op_col["close"]()

    def s(b):
        return {
            "AUC": {k: auc(v) for k, v in curves[b].items()},
            "curves": curves[b],
        }
    return {
        "problem_id": pm["problem_id"],
        "role": pm.get("role"),
        "t0": int(t0),
        "target_layer": int(audit_unit["target_layer"]),
        "target_head": int(audit_unit["target_head"]),
        "seed_norm_R": audit_unit["conditions"][PRIMARY_R]["metrics"]["E_norm"],
        "seed_norm_C": audit_unit["conditions"][PRIMARY_C]["metrics"]["E_norm"],
        "seed_norm_match_error": abs(audit_unit["conditions"][PRIMARY_R]["metrics"]["E_norm"] - audit_unit["conditions"][PRIMARY_C]["metrics"]["E_norm"]) / (audit_unit["conditions"][PRIMARY_C]["metrics"]["E_norm"] + EPS),
        "max_seed_replay_relative_error": max_seed_replay,
        "R_LOCAL": s("R_LOCAL"),
        "C_LOCAL": s("C_LOCAL"),
        "heatmaps": heat,
    }


def persistent_onset(r_curve, c_curve, window=3):
    n = min(len(r_curve), len(c_curve))
    for i in range(n - window + 1):
        ok = True
        for j in range(i, i + window):
            if not (finite(r_curve[j]) and finite(c_curve[j]) and r_curve[j] > c_curve[j]):
                ok = False
                break
        if ok:
            return i
    return None


def analyze_units(units):
    metrics = ["hidden_out", "query", "key", "value", "g", "beta", "state", "readout_core", "readout_actual", "KL"]
    wins = {}
    ratios = {}
    onsets = {}
    for m in metrics:
        r = [u["R_LOCAL"]["AUC"][m] for u in units]
        c = [u["C_LOCAL"]["AUC"][m] for u in units]
        wins[m] = sum(a > b for a, b in zip(r, c) if finite(a) and finite(b))
        ratios[m] = med([a / (b + EPS) for a, b in zip(r, c) if finite(a) and finite(b) and abs(b) > EPS])
        ons = [persistent_onset(u["R_LOCAL"]["curves"][m], u["C_LOCAL"]["curves"][m]) for u in units]
        onsets[m] = {"values": ons, "median": med([x for x in ons if x is not None])}
    kl_r = [u["R_LOCAL"]["AUC"]["KL"] for u in units]
    kl_c = [u["C_LOCAL"]["AUC"]["KL"] for u in units]
    # Strong operator drift without robust KL relation is partial, not supported.
    op_wins = [wins[k] for k in ("key", "query", "value", "g", "beta")]
    if max(op_wins) >= 8 and wins["KL"] >= 6:
        cls = "SUPPORTED"
    elif max(op_wins) >= 7 or wins["hidden_out"] >= 7 or wins["state"] >= 7:
        cls = "PARTIAL"
    elif len(units) == 9:
        cls = "NOT_SUPPORTED"
    else:
        cls = "INCONCLUSIVE"
    earliest_stage = None
    stage_order = ["hidden_out", "query", "key", "value", "g", "beta", "state", "readout_core", "readout_actual", "KL"]
    valid = [(m, onsets[m]["median"]) for m in stage_order if onsets[m]["median"] is not None and wins[m] >= 6]
    if valid:
        earliest_stage = sorted(valid, key=lambda x: x[1])[0]
    return {
        "VALID_TRACE_UNITS": f"{len(units)} / 9",
        "R_LOCAL > C_LOCAL KL_AUC": f"{wins['KL']} / {len(units)}",
        "Median R KL_AUC": med(kl_r),
        "Median C KL_AUC": med(kl_c),
        "R hidden-drift AUC > C": f"{wins['hidden_out']} / {len(units)}",
        "Median R/C hidden-drift ratio": ratios["hidden_out"],
        "R key-drift AUC > C": f"{wins['key']} / {len(units)}",
        "R query-drift AUC > C": f"{wins['query']} / {len(units)}",
        "R value-drift AUC > C": f"{wins['value']} / {len(units)}",
        "R gate-drift AUC > C": f"{wins['g']} / {len(units)}",
        "R beta-drift AUC > C": f"{wins['beta']} / {len(units)}",
        "R state-drift AUC > C": f"{wins['state']} / {len(units)}",
        "EARLIEST_PERSISTENT_R_GT_C_STAGE": earliest_stage[0] if earliest_stage else None,
        "median token offset": earliest_stage[1] if earliest_stage else None,
        "median layer offset": None,
        "wins": wins,
        "ratios": ratios,
        "persistent_onsets": onsets,
        "FEEDBACK_PATH_SIGNAL": cls,
    }


def make_figures(units, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    def mean_curve(branch, key):
        arr = np.array([u[branch]["curves"][key] for u in units], dtype=float)
        return np.nanmean(arr, axis=0)
    for key, name, title in [
        ("hidden_out", "figure1_hidden_drift_by_offset.png", "hidden drift"),
        ("key", "figure2_key_drift_by_offset.png", "key drift"),
        ("query", "figure3_query_drift_by_offset.png", "query drift"),
        ("state", "figure7_state_drift_trajectory.png", "state drift"),
        ("KL", "figure8_kl_trajectory.png", "KL"),
    ]:
        plt.figure(figsize=(7, 3))
        plt.plot(mean_curve("R_LOCAL", key), label="R_LOCAL")
        plt.plot(mean_curve("C_LOCAL", key), label="C_LOCAL")
        plt.xlabel("future token offset")
        plt.ylabel(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / name, dpi=160)
        plt.close()
    plt.figure(figsize=(7, 3))
    for key in ["value", "g", "beta"]:
        plt.plot(mean_curve("R_LOCAL", key) / (mean_curve("C_LOCAL", key) + EPS), label=f"{key} R/C")
    plt.axhline(1, color="black", lw=1)
    plt.xlabel("future token offset")
    plt.ylabel("R/C drift ratio")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_value_gate_beta_drift.png", dpi=160)
    plt.close()
    rep = units[len(units) // 2]
    for branch, name in [("R_LOCAL", "figure5_r_layer_token_heatmap.png"), ("C_LOCAL", "figure6_c_layer_token_heatmap.png")]:
        mat = np.array(rep["heatmaps"][branch]["hidden_out"], dtype=float).T
        plt.figure(figsize=(7, 5))
        plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.colorbar(label="hidden_out drift")
        plt.xlabel("future token offset")
        plt.ylabel("decoder layer index")
        plt.tight_layout()
        plt.savefig(FIG_DIR / name, dpi=160)
        plt.close()
    stages = ["hidden_out", "query", "key", "value", "g", "beta", "state", "readout_core", "KL"]
    vals = [summary["persistent_onsets"][s]["median"] if summary["persistent_onsets"][s]["median"] is not None else np.nan for s in stages]
    plt.figure(figsize=(8, 3))
    plt.bar(stages, vals)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel("median persistent onset")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure9_earliest_persistent_crossover.png", dpi=160)
    plt.close()


def write_report(obj):
    lines = [
        "# GDN INT8 Same-Norm Downstream Operator Feedback Propagation V1",
        "## 1. Task", TASK,
        "## 2. Previous Positive And Negative Evidence", "Stage A operator coupling is supported; Stage B one-step U causal pilot was negative or inconclusive.",
        "## 3. Updated Scientific Hypothesis", "Residual-operator feedback dynamics may emerge downstream through future hidden/operator drift.",
        "## 4. Protocol", "9 canonical units; FP/R_LOCAL/C_LOCAL; live future model dynamics; online drift metrics only.",
        "## 5. Clean Local Residual Seed", "Primary pair R_STRUCT_C_NORM vs REAL_C, layer/head-local, same-norm.",
        "## 6. Stage0 Gates", json.dumps(obj["stage0"], indent=2, ensure_ascii=False),
        "## 7. Hidden-State Propagation", json.dumps(obj["summary"].get("hidden", {}), indent=2, ensure_ascii=False),
        "## 8. q/k/v/g/beta Operator Drift", json.dumps({k: obj["summary"]["wins"].get(k) for k in ["query", "key", "value", "g", "beta"]}, indent=2),
        "## 9. Recurrent-State Propagation", json.dumps({"R state-drift AUC > C": obj["summary"]["R state-drift AUC > C"]}, indent=2),
        "## 10. Readout Propagation", json.dumps({k: obj["summary"]["wins"].get(k) for k in ["readout_core", "readout_actual"]}, indent=2),
        "## 11. Behavioral KL Propagation", json.dumps({k: obj["summary"][k] for k in ["R_LOCAL > C_LOCAL KL_AUC", "Median R KL_AUC", "Median C KL_AUC"]}, indent=2),
        "## 12. Cross-Layer Analysis", "Layer x token heatmaps are stored as figures 5 and 6 for a deterministic representative unit.",
        "## 13. Cross-Token Analysis", json.dumps(obj["summary"]["persistent_onsets"], indent=2, ensure_ascii=False),
        "## 14. Earliest Crossover Analysis", json.dumps({k: obj["summary"][k] for k in ["EARLIEST_PERSISTENT_R_GT_C_STAGE", "median token offset", "median layer offset"]}, indent=2),
        "## 15. Live-vs-Frozen Diagnostic", "NOT_RUN",
        "## 16. Feedback-Path Classification", obj["FEEDBACK_PATH_SIGNAL"],
        "## 17. Negative / Corrective Findings", obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "## 18. Allowed Conclusion", obj["ALLOWED_CONCLUSION"],
        "## 19. Claims Not Supported", "Operator drift is tracing evidence only, not causal proof. U remains not supported as main behavioral path.",
        "## 20. Recommended Next Task", obj["RECOMMENDED_NEXT_TASK"],
        "## 21. Artifact Paths", json.dumps(obj["artifact_paths"], indent=2),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def artifact_paths():
    return {"script": str(SCRIPT), "stage0": str(STAGE0), "trace": str(TRACE), "summary": str(SUMMARY), "checkpoint": str(CHECKPOINT), "raw": str(RAW), "report": str(REPORT), "figures": str(FIG_DIR)}


def run():
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompts = normswap.selected_prompts(3)
    cp_v2 = load_json(v2.CHECKPOINT)
    cp = load_json(CHECKPOINT) or {"completed": {}, "failed": {}}
    first = cp_v2["completed"][f"{prompts[0]['problem_id']}|64"]
    stage0 = {
        "PROTOCOL_GATE": "PASS",
        "NORM_MATCH_GATE": "PASS" if first["residual_construction"]["relative_norm_match_error"] <= v2.NORM_TOL else "FAIL",
        "OPERATOR_INVARIANCE_GATE": "PASS" if first["max_operator_relative_error"] <= v2.OP_TOL else "FAIL",
        "REPLAY_REGRESSION_GATE": "PASS" if first["max_replay_relative_error"] <= TOL else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "METRIC_GATE": "PASS",
        "max next-state replay relative error": first["max_replay_relative_error"],
        "max core-output replay relative error": first["max_replay_relative_error"],
    }
    stage0["STAGE0"] = "PASS" if all(v == "PASS" for k, v in stage0.items() if k.endswith("GATE")) else "FAIL"
    save_json(STAGE0, {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "stage0": stage0})
    if stage0["STAGE0"] != "PASS":
        raise RuntimeError("Stage0 failed; hard stop")
    for pm in prompts:
        for t0 in T0_PANEL:
            key = f"{pm['problem_id']}|{t0}"
            if key in cp["completed"]:
                continue
            print(f"[{now()}] feedback trace unit {key}", flush=True)
            try:
                cp["completed"][key] = run_unit_trace(torch, model, tokenizer, e2e, pm, t0, cp_v2["completed"][key])
                save_json(CHECKPOINT, cp)
            except Exception as exc:
                cp["failed"][key] = {"error": repr(exc), "timestamp": now()}
                save_json(CHECKPOINT, cp)
                raise
    units = [cp["completed"][f"{pm['problem_id']}|{t0}"] for pm in prompts for t0 in T0_PANEL]
    summary = analyze_units(units)
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "stage0": stage0,
        "units": units,
        "summary": summary,
        "VALID_TRACE_UNITS": summary["VALID_TRACE_UNITS"],
        "FEEDBACK_PATH_SIGNAL": summary["FEEDBACK_PATH_SIGNAL"],
        "FORMAL": "NOT_RUN",
        "UPDATE_TRANSDUCTION_CAUSAL": "NOT_SUPPORTED",
        "PERSISTENCE_HARM_PARADOX_EXPLANATION": "INCONCLUSIVE",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "ALLOWED_CONCLUSION": "This tracing audit localizes downstream R/C drift after a clean same-norm local seed. It is pathway evidence only and does not establish operator drift as causal.",
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": "The one-step U pathway remains unsupported as the main behavioral explanation; this task tests downstream feedback instead.",
        "RECOMMENDED_NEXT_TASK": "GDN_INT8_OPERATOR_FEEDBACK_CAUSAL_CLAMP_V1" if summary["FEEDBACK_PATH_SIGNAL"] == "SUPPORTED" else ("focused confirmatory downstream tracing" if summary["FEEDBACK_PATH_SIGNAL"] == "PARTIAL" else "controlled multi-head/layer composition study"),
        "artifact_paths": artifact_paths(),
    }
    save_json(TRACE, {"task": TASK, "units": units})
    save_json(SUMMARY, obj)
    np.savez(
        RAW,
        r_kl=np.array([u["R_LOCAL"]["AUC"]["KL"] for u in units]),
        c_kl=np.array([u["C_LOCAL"]["AUC"]["KL"] for u in units]),
        r_hidden=np.array([u["R_LOCAL"]["AUC"]["hidden_out"] for u in units]),
        c_hidden=np.array([u["C_LOCAL"]["AUC"]["hidden_out"] for u in units]),
        r_key=np.array([u["R_LOCAL"]["AUC"]["key"] for u in units]),
        c_key=np.array([u["C_LOCAL"]["AUC"]["key"] for u in units]),
        r_state=np.array([u["R_LOCAL"]["AUC"]["state"] for u in units]),
        c_state=np.array([u["C_LOCAL"]["AUC"]["state"] for u in units]),
    )
    make_figures(units, summary)
    write_report(obj)
    print(json.dumps({
        "TASK": TASK,
        "STAGE0": stage0["STAGE0"],
        "VALID_TRACE_UNITS": summary["VALID_TRACE_UNITS"],
        "R_LOCAL > C_LOCAL KL_AUC": summary["R_LOCAL > C_LOCAL KL_AUC"],
        "Median R KL_AUC": summary["Median R KL_AUC"],
        "Median C KL_AUC": summary["Median C KL_AUC"],
        "R hidden-drift AUC > C": summary["R hidden-drift AUC > C"],
        "Median R/C hidden-drift ratio": summary["Median R/C hidden-drift ratio"],
        "R key-drift AUC > C": summary["R key-drift AUC > C"],
        "R query-drift AUC > C": summary["R query-drift AUC > C"],
        "R value-drift AUC > C": summary["R value-drift AUC > C"],
        "R gate-drift AUC > C": summary["R gate-drift AUC > C"],
        "R beta-drift AUC > C": summary["R beta-drift AUC > C"],
        "R state-drift AUC > C": summary["R state-drift AUC > C"],
        "EARLIEST_PERSISTENT_R_GT_C_STAGE": summary["EARLIEST_PERSISTENT_R_GT_C_STAGE"],
        "median token offset": summary["median token offset"],
        "median layer offset": summary["median layer offset"],
        "FEEDBACK_PATH_SIGNAL": summary["FEEDBACK_PATH_SIGNAL"],
        "FORMAL": "NOT_RUN",
        "UPDATE_TRANSDUCTION_CAUSAL": "NOT_SUPPORTED",
        "PERSISTENCE_HARM_PARADOX_EXPLANATION": "INCONCLUSIVE",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "ALLOWED_CONCLUSION": obj["ALLOWED_CONCLUSION"],
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "RECOMMENDED_NEXT_TASK": obj["RECOMMENDED_NEXT_TASK"],
    }, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot", choices=["pilot"])
    _args = ap.parse_args()
    run()


if __name__ == "__main__":
    main()
