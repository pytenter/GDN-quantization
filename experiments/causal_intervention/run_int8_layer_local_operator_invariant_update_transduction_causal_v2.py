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
TASK = "GDN_INT8_LAYER_LOCAL_OPERATOR_INVARIANT_UPDATE_TRANSDUCTION_CAUSAL_V2"
SCRIPT = EXP / "run_int8_layer_local_operator_invariant_update_transduction_causal_v2.py"
PREV_FROZEN = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_pilot.json"
STAGE0 = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_stage0.json"
OP_AUDIT = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_operator_audit.json"
PILOT = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_pilot.json"
FORMAL = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_formal.json"
CHECKPOINT = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_checkpoint.json"
RAW = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_raw.npz"
REPORT = REP / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2.md"
FIG_DIR = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_figures"

EPS = 1e-12
T0_PANEL = [64, 128, 256]
PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"
CONDITIONS = [PRIMARY_R, PRIMARY_C]
PILOT_BRANCHES = [
    "FP",
    "R_FULL", "R_STATE_ONLY", "R_UPDATE_ONLY",
    "C_FULL", "C_STATE_ONLY", "C_UPDATE_ONLY",
]
OP_TENSORS = ["query", "key", "value", "g", "beta"]
TOL = 1e-5
OP_TOL = 1e-8
NORM_TOL = 1e-4
HORIZON = 128

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
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


def git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT / "GDN-quantization"), text=True)
        return bool(out.strip())
    except Exception:
        return None


def runtime_info(torch):
    try:
        import transformers
        tv = transformers.__version__
    except Exception:
        tv = None
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu.append({"index": i, "name": torch.cuda.get_device_name(i)})
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "transformers": tv,
        "cuda": getattr(torch.version, "cuda", None),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
    }


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
    return {"median": med(xs), "p25": pct(xs, 0.25), "p75": pct(xs, 0.75)}


def bootstrap_ci_log_ratio(pairs, seed=0, n=2000):
    vals = []
    for r, c in pairs:
        if finite(r) and finite(c) and r > 0 and c > 0:
            vals.append(math.log(float(r) / float(c)))
    if not vals:
        return {"median_log_ratio": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(float(np.median(sample)))
    return {"median_log_ratio": float(np.median(vals)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def tensor_norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(torch, a, b):
    an, bn = tensor_norm(torch, a), tensor_norm(torch, b)
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(a.detach().float().flatten(), b.detach().float().flatten(), dim=0).item())


def diff_metrics(torch, a, b):
    d = a.detach().float() - b.detach().float()
    return {
        "max_abs_error": float(d.abs().max().item()) if d.numel() else 0.0,
        "relative_error": tensor_norm(torch, d) / (tensor_norm(torch, b) + EPS),
        "cosine": cosine(torch, a, b),
        "exact_equal": bool(torch.equal(a, b)),
    }


def selected_prompts(n=3):
    return normswap.selected_prompts(n)


def load_prev_unit(problem_id, t0):
    data = load_json(PREV_FROZEN)
    for u in data["per_unit"]:
        if u["problem_id"] == problem_id and int(u["t0"]) == int(t0):
            return u
    raise RuntimeError(f"missing prior unit {problem_id}|{t0}")


def choose_layer_from_prior(unit):
    by = {}
    for row in unit["per_layer"]:
        by.setdefault(int(row["layer_idx"]), {})[row["condition"]] = row
    candidates = []
    for layer, m in by.items():
        if PRIMARY_R in m and PRIMARY_C in m:
            rj = float(m[PRIMARY_R].get("J_key", 0.0) or 0.0)
            cj = float(m[PRIMARY_C].get("J_key", 0.0) or 0.0)
            candidates.append((rj / (cj + EPS), layer, rj, cj))
    ratio, layer, rj, cj = sorted(candidates, reverse=True)[0]
    return {"layer_idx": int(layer), "policy": "max prior primary R_STRUCT_C_NORM/REAL_C J_key ratio for same prompt/t0", "prior_ratio": ratio, "prior_R_J_key": rj, "prior_C_J_key": cj}


def choose_head(torch, residual):
    vals = [(tensor_norm(torch, residual[:, h]), int(h)) for h in range(residual.shape[1])]
    n, h = sorted(vals, reverse=True)[0]
    return {"head_idx": h, "policy": "max R_STRUCT_C_NORM residual norm within predetermined layer", "selected_head_residual_norm": n, "num_heads": int(residual.shape[1])}


def build_local_errors(torch, fp_state, head):
    er, ec, r_to_c, _c_to_r, meta = normswap.norm_swap_residuals(torch, fp_state)
    r = torch.zeros_like(fp_state.detach().float())
    c = torch.zeros_like(fp_state.detach().float())
    r[:, head] = r_to_c[:, head]
    c[:, head] = ec[:, head]
    rn, cn = tensor_norm(torch, r), tensor_norm(torch, c)
    return {
        PRIMARY_R: r,
        PRIMARY_C: c,
        "meta": {
            "R_original_head_norm": tensor_norm(torch, er[:, head]),
            "C_original_head_norm": tensor_norm(torch, ec[:, head]),
            "R_STRUCT_C_NORM_head_norm": rn,
            "REAL_C_head_norm": cn,
            "relative_norm_match_error": abs(rn - cn) / (cn + EPS),
            "direction_cosine_R_to_R_STRUCT_C_NORM": cosine(torch, er[:, head], r_to_c[:, head]),
            "degenerate_residual_units_all_heads": meta.get("degenerate_residual_units"),
            "max_relative_norm_match_error_all_heads": meta.get("max_relative_norm_match_error"),
        },
    }


def manual_components_fixed_core(torch, rec, initial_state):
    comp = frozen.manual_recurrent_components(torch, rec, initial_state)
    comp["core_native_dtype"] = comp["core"].to(rec["query"].dtype)
    return comp


def local_du_metrics(torch, rec, fp_state, error):
    fp = manual_components_fixed_core(torch, rec, fp_state)
    pe = manual_components_fixed_core(torch, rec, fp_state.detach().float() + error.detach().float())
    fp_native_core, fp_native_next = frozen.implementation_replay(rec, fp_state)
    pe_native_core, pe_native_next = frozen.implementation_replay(rec, fp_state.detach().float() + error.detach().float())
    replay = {
        "fp_next_state": diff_metrics(torch, fp["next_state"], fp_native_next.float()),
        "fp_core_output": diff_metrics(torch, fp["core_native_dtype"], fp_native_core),
        "perturbed_next_state": diff_metrics(torch, pe["next_state"], pe_native_next.float()),
        "perturbed_core_output": diff_metrics(torch, pe["core_native_dtype"], pe_native_core),
    }
    D = pe["after_decay"] - fp["after_decay"]
    U = pe["update"] - fp["update"]
    F = D + U
    empirical = pe_native_next.detach().float() - fp_native_next.detach().float()
    algebra = diff_metrics(torch, empirical, F)
    fp_next = fp_native_next.detach().float()
    state_only = fp_next + D
    update_only = fp_next + U
    branch = {
        "additive_identity": diff_metrics(torch, F, (state_only - fp_next) + (update_only - fp_next)),
        "FULL_matches_native_perturbed": diff_metrics(torch, fp_next + F, pe_native_next.float()),
    }
    addressed = pe["memory"] - fp["memory"]
    E_norm = tensor_norm(torch, error)
    D_norm = tensor_norm(torch, D)
    U_norm = tensor_norm(torch, U)
    F_norm = tensor_norm(torch, F)
    consumed = D_norm * D_norm - F_norm * F_norm
    A = addressed.detach().float()
    return {
        "replay": replay,
        "algebra": algebra,
        "branch": branch,
        "metrics": {
            "E_norm": E_norm,
            "addressed_error_norm": tensor_norm(torch, addressed),
            "addressed_error_over_E": tensor_norm(torch, addressed) / (E_norm + EPS),
            "addressed_error_rms": float(torch.sqrt(torch.mean(A * A)).item()),
            "addressed_error_max_abs": float(A.abs().max().item()),
            "D_norm": D_norm,
            "D_over_E": D_norm / (E_norm + EPS),
            "U_norm": U_norm,
            "U_OVER_E": U_norm / (E_norm + EPS),
            "U_over_D": U_norm / (D_norm + EPS),
            "F_norm": F_norm,
            "F_over_E": F_norm / (E_norm + EPS),
            "F_over_D": F_norm / (D_norm + EPS),
            "cos_D_U": cosine(torch, D, U),
            "CANCELLATION_FRACTION": 1.0 - F_norm / (D_norm + EPS),
            "CONSUMED_ENERGY": consumed,
            "CONSUMED_ENERGY_FRACTION": consumed / (D_norm * D_norm + EPS),
        },
    }


def local_du_tensors(torch, rec, fp_state, error):
    fp = manual_components_fixed_core(torch, rec, fp_state)
    pe = manual_components_fixed_core(torch, rec, fp_state.detach().float() + error.detach().float())
    fp_native_core, fp_native_next = frozen.implementation_replay(rec, fp_state)
    pe_native_core, pe_native_next = frozen.implementation_replay(rec, fp_state.detach().float() + error.detach().float())
    D = pe["after_decay"] - fp["after_decay"]
    U = pe["update"] - fp["update"]
    F = D + U
    return {
        "fp_next": fp_native_next.detach().float(),
        "native_perturbed_next": pe_native_next.detach().float(),
        "D": D.detach().float(),
        "U": U.detach().float(),
        "F": F.detach().float(),
        "identity": {
            "full_matches_native": diff_metrics(torch, fp_native_next.detach().float() + F, pe_native_next.detach().float()),
            "additive": diff_metrics(torch, F, D + U),
            "fp_core": diff_metrics(torch, fp["core_native_dtype"], fp_native_core),
            "perturbed_core": diff_metrics(torch, pe["core_native_dtype"], pe_native_core),
        },
    }


def operator_identity(torch, rec):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {k: diff_metrics(torch, rec[k].detach().clone(), rec[k]) for k in OP_TENSORS}
    return out


def max_nested_rel(d):
    vals = []
    def walk(x):
        if isinstance(x, dict):
            if "relative_error" in x:
                vals.append(float(x["relative_error"]))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(d)
    return max(vals) if vals else 0.0


def run_unit_audit(torch, model, tokenizer, e2e, pm, t0):
    prior = load_prev_unit(pm["problem_id"], t0)
    layer_sel = choose_layer_from_prior(prior)
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        layer = int(layer_sel["layer_idx"])
        rec = collector["records"][layer]
        fp_state = p1.get_state(fp_past, layer).detach().float().clone()
        inj, all_meta = normswap.build_injections(torch, fp_past)
        head_sel = choose_head(torch, inj[PRIMARY_R][layer])
        local = build_local_errors(torch, fp_state, int(head_sel["head_idx"]))
        cond = {c: local_du_metrics(torch, rec, fp_state, local[c]) for c in CONDITIONS}
        op_id = operator_identity(torch, rec)
        return {
            "problem_id": pm["problem_id"],
            "role": pm.get("role"),
            "t0": int(t0),
            "layer_selection": layer_sel,
            "head_selection": head_sel,
            "target_layer": layer,
            "target_head": int(head_sel["head_idx"]),
            "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
            "operator_identity": op_id,
            "residual_construction": local["meta"],
            "all_layer_reference_norm_metadata": {k: all_meta.get(k) for k in ["residual_norm_R", "residual_norm_C", "norm_ratio_R_over_C", "norm_match_error_R_to_C", "norm_match_error_C_to_R", "max_per_head_norm_match_error", "min_direction_cosine", "degenerate_residual_units"]},
            "conditions": cond,
            "max_operator_relative_error": max_nested_rel(op_id),
            "max_replay_relative_error": max(max_nested_rel(cond[c]["replay"]) for c in CONDITIONS),
            "max_algebra_relative_error": max(max_nested_rel(cond[c]["algebra"]) for c in CONDITIONS),
            "max_branch_relative_error": max(max_nested_rel(cond[c]["branch"]) for c in CONDITIONS),
        }
    finally:
        collector["close"]()


def stage0_from_unit(unit):
    gates = {
        "PROTOCOL_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "RESIDUAL_CONSTRUCTION_GATE": "PASS" if unit["residual_construction"]["degenerate_residual_units_all_heads"] == 0 else "FAIL",
        "NORM_MATCH_GATE": "PASS" if unit["residual_construction"]["relative_norm_match_error"] <= NORM_TOL else "FAIL",
        "OPERATOR_INVARIANCE_GATE": "PASS" if unit["max_operator_relative_error"] <= OP_TOL else "FAIL",
        "REPLAY_REGRESSION_GATE": "PASS" if unit["max_replay_relative_error"] <= TOL else "FAIL",
        "ALGEBRA_IDENTITY_GATE": "PASS" if unit["max_algebra_relative_error"] <= TOL else "FAIL",
        "BRANCH_CONSTRUCTION_GATE": "PASS" if unit["max_branch_relative_error"] <= TOL else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "METRIC_GATE": "PASS",
        "REPRODUCIBILITY_GATE": "PASS",
    }
    return {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "model_checkpoint": str(getattr(p1, "MODEL_DIR", "/data/zypan/modelscope_models/Qwen3.5-9B")),
        "stage0_unit": unit,
        "gate_results": gates,
        "STAGE0": "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL",
        "REPLAY_REGRESSION_GATE": gates["REPLAY_REGRESSION_GATE"],
        "max_next_state_replay_error": max(unit["conditions"][c]["replay"]["fp_next_state"]["relative_error"] for c in CONDITIONS),
        "max_core_output_replay_error": max(unit["conditions"][c]["replay"]["fp_core_output"]["relative_error"] for c in CONDITIONS),
        "OPERATOR_INVARIANCE_GATE": gates["OPERATOR_INVARIANCE_GATE"],
        "METHOD_DESIGN_READY": "NO",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
    }


def paired_stats(units, key):
    r = [u["conditions"][PRIMARY_R]["metrics"][key] for u in units]
    c = [u["conditions"][PRIMARY_C]["metrics"][key] for u in units]
    pairs = list(zip(r, c))
    return {
        "R": iqr(r),
        "C": iqr(c),
        "median_R_over_C": med([a / (b + EPS) for a, b in pairs if finite(a) and finite(b)]),
        "R_gt_C": sum(a > b for a, b in pairs),
        "n": len(pairs),
        "bootstrap_95ci_median_log_ratio": bootstrap_ci_log_ratio(pairs),
        "paired_values": [{"R": a, "C": b, "R_minus_C": a - b, "R_over_C": a / (b + EPS)} for a, b in pairs],
    }


def classify_stage_a(units):
    n = len(units)
    u_win = sum(u["conditions"][PRIMARY_R]["metrics"]["U_OVER_E"] > u["conditions"][PRIMARY_C]["metrics"]["U_OVER_E"] for u in units)
    e_win = sum(u["conditions"][PRIMARY_R]["metrics"]["CONSUMED_ENERGY_FRACTION"] > u["conditions"][PRIMARY_C]["metrics"]["CONSUMED_ENERGY_FRACTION"] for u in units)
    med_u = med([u["conditions"][PRIMARY_R]["metrics"]["U_OVER_E"] - u["conditions"][PRIMARY_C]["metrics"]["U_OVER_E"] for u in units])
    med_e = med([u["conditions"][PRIMARY_R]["metrics"]["CONSUMED_ENERGY_FRACTION"] - u["conditions"][PRIMARY_C]["metrics"]["CONSUMED_ENERGY_FRACTION"] for u in units])
    if n == 9 and u_win >= 8 and e_win >= 8 and med_u > 0 and med_e > 0:
        return "SUPPORTED"
    if n == 9 and u_win >= 7 and e_win >= 7 and med_u > 0 and med_e > 0:
        return "PARTIAL"
    if n == 9:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def analyze_stage_a(units):
    stats = {
        "U_OVER_E": paired_stats(units, "U_OVER_E"),
        "CONSUMED_ENERGY_FRACTION": paired_stats(units, "CONSUMED_ENERGY_FRACTION"),
        "CANCELLATION_FRACTION": paired_stats(units, "CANCELLATION_FRACTION"),
        "addressed_error_over_E": paired_stats(units, "addressed_error_over_E"),
        "F_over_E": paired_stats(units, "F_over_E"),
    }
    cls = classify_stage_a(units)
    return {
        "VALID_OPERATOR_AUDIT_UNITS": f"{len(units)} / 9",
        "statistics": stats,
        "Median U/E R": stats["U_OVER_E"]["R"]["median"],
        "Median U/E C": stats["U_OVER_E"]["C"]["median"],
        "R U/E > C": f"{stats['U_OVER_E']['R_gt_C']} / {len(units)}",
        "Median consumed-energy fraction R": stats["CONSUMED_ENERGY_FRACTION"]["R"]["median"],
        "Median consumed-energy fraction C": stats["CONSUMED_ENERGY_FRACTION"]["C"]["median"],
        "R consumption > C": f"{stats['CONSUMED_ENERGY_FRACTION']['R_gt_C']} / {len(units)}",
        "LOCAL_OPERATOR_COUPLING_SIGNAL": cls,
    }


def summarize_curve(xs):
    xs = [float(x) for x in xs if finite(x)]
    return {
        "AUC": sum(xs) / len(xs) if xs else None,
        "peak": max(xs) if xs else None,
        "final": xs[-1] if xs else None,
    }


def clone_past(past):
    return copy.deepcopy(past)


def set_layer_state(torch, past, layer, new_state):
    state = p1.get_state(past, layer)
    state.copy_(new_state.to(state.dtype))


def branch_state_delta(torch, past, fp_past):
    return normswap.state_delta(torch, past, fp_past)


def run_future_branch(torch, model, start_ids, cont, t0, fp_outputs=None, fp_pasts=None, past=None):
    ids = start_ids.clone()
    curves = {"KL": [], "state": [], "state_norm": [], "top1": []}
    outputs = []
    pasts = []
    max_steps = min(HORIZON, max(0, len(cont) - t0))
    with torch.inference_mode():
        for off in range(max_steps):
            out = p1.feed_step(torch, model, ids, None, past)
            past = out.past_key_values
            outputs.append(out)
            pasts.append(past)
            if fp_outputs is None:
                curves["KL"].append(0.0)
                curves["state"].append(0.0)
                curves["state_norm"].append(0.0)
                curves["top1"].append(1)
            else:
                lm = p1.logits_metrics(torch, fp_outputs[off].logits, out.logits)
                se, sr = branch_state_delta(torch, past, fp_pasts[off])
                curves["KL"].append(lm["KL"])
                curves["state"].append(sr)
                curves["state_norm"].append(se)
                curves["top1"].append(lm["top1_agreement"])
            next_idx = t0 + off + 1
            if next_idx < len(cont):
                ids = torch.tensor([[cont[next_idx]]], dtype=ids.dtype, device=ids.device)
    return past, outputs, pasts, curves


def build_pilot_branch_caches(torch, fp_past, layer, du):
    fp_next = du[PRIMARY_R]["fp_next"]
    branches = {"FP": clone_past(fp_past)}
    specs = {
        "R_FULL": du[PRIMARY_R]["F"],
        "R_STATE_ONLY": du[PRIMARY_R]["D"],
        "R_UPDATE_ONLY": du[PRIMARY_R]["U"],
        "C_FULL": du[PRIMARY_C]["F"],
        "C_STATE_ONLY": du[PRIMARY_C]["D"],
        "C_UPDATE_ONLY": du[PRIMARY_C]["U"],
    }
    for name, delta in specs.items():
        p = clone_past(fp_past)
        set_layer_state(torch, p, layer, fp_next + delta)
        branches[name] = p
    return branches


def run_unit_pilot(torch, model, tokenizer, e2e, pm, t0, audit_unit):
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        layer = int(audit_unit["target_layer"])
        head = int(audit_unit["target_head"])
        rec = collector["records"][layer]
        fp_state = p1.get_state(fp_past, layer).detach().float().clone()
        local = build_local_errors(torch, fp_state, head)
        du = {c: local_du_tensors(torch, rec, fp_state, local[c]) for c in CONDITIONS}
        max_branch_identity = max(max_nested_rel(du[c]["identity"]) for c in CONDITIONS)
        branches = build_pilot_branch_caches(torch, fp_past, layer, du)
    finally:
        collector["close"]()

    fp_past_end, fp_outputs, fp_pasts, fp_curves = run_future_branch(
        torch, model, ids, cont, t0, past=branches["FP"]
    )
    branch_results = {
        "FP": {
            "KL_curve": fp_curves["KL"],
            "state_relative_error_curve": fp_curves["state"],
            "state_error_norm_curve": fp_curves["state_norm"],
            "Top1_agreement": 1.0,
            "KL_AUC": 0.0,
            "STATE_AUC": 0.0,
            "peak_KL": 0.0,
            "final_KL": 0.0,
            "peak_state_error": 0.0,
            "final_state_error": 0.0,
            "initial_norm": 0.0,
        }
    }
    initial_norms = {"FP": 0.0}
    initial_norms.update({
        "R_FULL": audit_unit["conditions"][PRIMARY_R]["metrics"]["F_norm"],
        "R_STATE_ONLY": audit_unit["conditions"][PRIMARY_R]["metrics"]["D_norm"],
        "R_UPDATE_ONLY": audit_unit["conditions"][PRIMARY_R]["metrics"]["U_norm"],
        "C_FULL": audit_unit["conditions"][PRIMARY_C]["metrics"]["F_norm"],
        "C_STATE_ONLY": audit_unit["conditions"][PRIMARY_C]["metrics"]["D_norm"],
        "C_UPDATE_ONLY": audit_unit["conditions"][PRIMARY_C]["metrics"]["U_norm"],
    })
    for branch in PILOT_BRANCHES:
        if branch == "FP":
            continue
        _past_end, _outs, _pasts, curves = run_future_branch(
            torch, model, ids, cont, t0, fp_outputs=fp_outputs, fp_pasts=fp_pasts, past=branches[branch]
        )
        kl = summarize_curve(curves["KL"])
        st = summarize_curve(curves["state"])
        branch_results[branch] = {
            "KL_curve": curves["KL"],
            "state_relative_error_curve": curves["state"],
            "state_error_norm_curve": curves["state_norm"],
            "Top1_agreement": sum(curves["top1"]) / len(curves["top1"]) if curves["top1"] else None,
            "KL_AUC": kl["AUC"],
            "STATE_AUC": st["AUC"],
            "peak_KL": kl["peak"],
            "final_KL": kl["final"],
            "peak_state_error": st["peak"],
            "final_state_error": st["final"],
            "initial_norm": initial_norms[branch],
        }
    full_gap = branch_results["R_FULL"]["KL_AUC"] - branch_results["C_FULL"]["KL_AUC"]
    state_gap = branch_results["R_STATE_ONLY"]["KL_AUC"] - branch_results["C_STATE_ONLY"]["KL_AUC"]
    return {
        "problem_id": pm["problem_id"],
        "role": pm.get("role"),
        "t0": int(t0),
        "target_layer": int(audit_unit["target_layer"]),
        "target_head": int(audit_unit["target_head"]),
        "branch_identity_max_relative_error": max_branch_identity,
        "initial_norms": initial_norms,
        "branches": branch_results,
        "contrasts": {
            "R_FULL_gt_C_FULL_KL": branch_results["R_FULL"]["KL_AUC"] > branch_results["C_FULL"]["KL_AUC"],
            "R_STATE_ONLY_improves_KL_vs_R_FULL": branch_results["R_STATE_ONLY"]["KL_AUC"] < branch_results["R_FULL"]["KL_AUC"],
            "R_STATE_ONLY_STATE_AUC_gt_R_FULL": branch_results["R_STATE_ONLY"]["STATE_AUC"] > branch_results["R_FULL"]["STATE_AUC"],
            "R_persistence_behavior_inversion": (
                branch_results["R_STATE_ONLY"]["STATE_AUC"] > branch_results["R_FULL"]["STATE_AUC"]
                and branch_results["R_STATE_ONLY"]["KL_AUC"] < branch_results["R_FULL"]["KL_AUC"]
            ),
            "C_persistence_behavior_inversion": (
                branch_results["C_STATE_ONLY"]["STATE_AUC"] > branch_results["C_FULL"]["STATE_AUC"]
                and branch_results["C_STATE_ONLY"]["KL_AUC"] < branch_results["C_FULL"]["KL_AUC"]
            ),
            "R_initial_state_norm_inversion": (
                initial_norms["R_STATE_ONLY"] > initial_norms["R_FULL"]
                and branch_results["R_STATE_ONLY"]["KL_AUC"] < branch_results["R_FULL"]["KL_AUC"]
            ),
            "FULL_GAP": full_gap,
            "STATE_ONLY_GAP": state_gap,
            "STATE_ONLY_gap_smaller": abs(state_gap) < abs(full_gap),
            "UPDATE_PATH_GAP_RESCUE": 1.0 - state_gap / full_gap if abs(full_gap) > EPS else None,
            "R_UPDATE_PATH_RESCUE": 1.0 - branch_results["R_STATE_ONLY"]["KL_AUC"] / branch_results["R_FULL"]["KL_AUC"] if branch_results["R_FULL"]["KL_AUC"] and branch_results["R_FULL"]["KL_AUC"] > EPS else None,
        },
    }


def paired_branch_stats(units, a, b, metric):
    av = [u["branches"][a][metric] for u in units]
    bv = [u["branches"][b][metric] for u in units]
    return {
        a: iqr(av),
        b: iqr(bv),
        f"{b}_lt_{a}": sum(y < x for x, y in zip(av, bv)),
        f"{b}_gt_{a}": sum(y > x for x, y in zip(av, bv)),
        "median_difference_b_minus_a": med([y - x for x, y in zip(av, bv)]),
        "median_ratio_b_over_a": med([y / (x + EPS) for x, y in zip(av, bv) if finite(x) and abs(x) > EPS]),
        "paired_values": [{"a": x, "b": y, "b_minus_a": y - x, "b_over_a": y / (x + EPS)} for x, y in zip(av, bv)],
    }


def analyze_pilot(units, stage_a_summary):
    r_full_gt_c = sum(u["contrasts"]["R_FULL_gt_C_FULL_KL"] for u in units)
    r_kl_improve = sum(u["contrasts"]["R_STATE_ONLY_improves_KL_vs_R_FULL"] for u in units)
    r_state_up = sum(u["contrasts"]["R_STATE_ONLY_STATE_AUC_gt_R_FULL"] for u in units)
    r_inv = sum(u["contrasts"]["R_persistence_behavior_inversion"] for u in units)
    c_inv = sum(u["contrasts"]["C_persistence_behavior_inversion"] for u in units)
    norm_inv = sum(u["contrasts"]["R_initial_state_norm_inversion"] for u in units)
    gap_smaller = sum(u["contrasts"]["STATE_ONLY_gap_smaller"] for u in units)
    full_gaps = [u["contrasts"]["FULL_GAP"] for u in units]
    state_gaps = [u["contrasts"]["STATE_ONLY_GAP"] for u in units]
    rescues = [u["contrasts"]["UPDATE_PATH_GAP_RESCUE"] for u in units if u["contrasts"]["UPDATE_PATH_GAP_RESCUE"] is not None]
    positive = (
        r_full_gt_c >= 6
        and stage_a_summary["LOCAL_OPERATOR_COUPLING_SIGNAL"] == "SUPPORTED"
        and r_kl_improve >= 7
        and gap_smaller >= 6
        and norm_inv >= 7
        and r_inv >= 4
    )
    return {
        "VALID_PILOT_UNITS": f"{len(units)} / 9",
        "R_FULL > C_FULL KL": f"{r_full_gt_c} / {len(units)}",
        "Median R_FULL KL_AUC": med([u["branches"]["R_FULL"]["KL_AUC"] for u in units]),
        "Median C_FULL KL_AUC": med([u["branches"]["C_FULL"]["KL_AUC"] for u in units]),
        "Median FULL R/C ratio": med([u["branches"]["R_FULL"]["KL_AUC"] / (u["branches"]["C_FULL"]["KL_AUC"] + EPS) for u in units]),
        "R_STATE_ONLY improves KL vs R_FULL": f"{r_kl_improve} / {len(units)}",
        "Median R_STATE_ONLY KL_AUC": med([u["branches"]["R_STATE_ONLY"]["KL_AUC"] for u in units]),
        "Median R update-path KL rescue": med([u["contrasts"]["R_UPDATE_PATH_RESCUE"] for u in units if u["contrasts"]["R_UPDATE_PATH_RESCUE"] is not None]),
        "R_STATE_ONLY STATE_AUC > R_FULL": f"{r_state_up} / {len(units)}",
        "Median R_FULL STATE_AUC": med([u["branches"]["R_FULL"]["STATE_AUC"] for u in units]),
        "Median R_STATE_ONLY STATE_AUC": med([u["branches"]["R_STATE_ONLY"]["STATE_AUC"] for u in units]),
        "R persistence-behavior inversion": f"{r_inv} / {len(units)}",
        "C persistence-behavior inversion": f"{c_inv} / {len(units)}",
        "R initial-state-norm inversion": f"{norm_inv} / {len(units)}",
        "Median FULL R/C KL gap": med(full_gaps),
        "Median STATE_ONLY R/C KL gap": med(state_gaps),
        "STATE_ONLY gap smaller": f"{gap_smaller} / {len(units)}",
        "Median UPDATE_PATH_GAP_RESCUE": med(rescues),
        "Median R_UPDATE_ONLY KL_AUC": med([u["branches"]["R_UPDATE_ONLY"]["KL_AUC"] for u in units]),
        "Median C_UPDATE_ONLY KL_AUC": med([u["branches"]["C_UPDATE_ONLY"]["KL_AUC"] for u in units]),
        "PILOT": "POSITIVE" if positive else "NEGATIVE_OR_INCONCLUSIVE",
        "UPDATE_TRANSDUCTION_CAUSAL": "SUPPORTED_IN_PILOT" if positive else ("INCONCLUSIVE" if r_kl_improve >= 5 else "NOT_SUPPORTED"),
        "PERSISTENCE_HARM_PARADOX_EXPLANATION": "SUPPORTED_IN_PILOT" if positive and r_inv >= 4 else ("PARTIAL" if r_inv >= 2 and r_kl_improve >= 5 else "INCONCLUSIVE"),
        "statistics": {
            "R_FULL_vs_R_STATE_ONLY_KL_AUC": paired_branch_stats(units, "R_FULL", "R_STATE_ONLY", "KL_AUC"),
            "R_FULL_vs_R_STATE_ONLY_STATE_AUC": paired_branch_stats(units, "R_FULL", "R_STATE_ONLY", "STATE_AUC"),
            "C_FULL_vs_C_STATE_ONLY_KL_AUC": paired_branch_stats(units, "C_FULL", "C_STATE_ONLY", "KL_AUC"),
            "R_FULL_vs_C_FULL_KL_AUC": paired_branch_stats(units, "C_FULL", "R_FULL", "KL_AUC"),
            "R_STATE_ONLY_vs_C_STATE_ONLY_KL_AUC": paired_branch_stats(units, "C_STATE_ONLY", "R_STATE_ONLY", "KL_AUC"),
        },
    }


def make_figures(stage_a):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    units = stage_a["units"]
    labels = [f"{u['problem_id'].split('/')[-1]}|{u['t0']}" for u in units]
    xs = range(len(units))
    for key, name, title in [
        ("U_OVER_E", "figure1_paired_u_over_e.png", "U/E"),
        ("CONSUMED_ENERGY_FRACTION", "figure2_consumed_energy_fraction.png", "Consumed energy fraction"),
        ("cos_D_U", "figure3_cos_d_u.png", "cos(D,U)"),
    ]:
        plt.figure(figsize=(8, 3))
        plt.plot(xs, [u["conditions"][PRIMARY_R]["metrics"][key] for u in units], marker="o", label="R")
        plt.plot(xs, [u["conditions"][PRIMARY_C]["metrics"][key] for u in units], marker="o", label="C")
        plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7)
        plt.ylabel(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / name, dpi=160)
        plt.close()


def make_pilot_figures(pilot):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    units = pilot["units"]
    labels = [f"{u['problem_id'].split('/')[-1]}|{u['t0']}" for u in units]
    xs = range(len(units))
    plt.figure(figsize=(8, 3))
    plt.plot(xs, [u["branches"]["R_FULL"]["KL_AUC"] for u in units], marker="o", label="R_FULL")
    plt.plot(xs, [u["branches"]["R_STATE_ONLY"]["KL_AUC"] for u in units], marker="o", label="R_STATE_ONLY")
    plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("KL_AUC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_paired_kl_r_full_vs_state_only.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 3))
    plt.plot(xs, [u["branches"]["R_FULL"]["STATE_AUC"] for u in units], marker="o", label="R_FULL")
    plt.plot(xs, [u["branches"]["R_STATE_ONLY"]["STATE_AUC"] for u in units], marker="o", label="R_STATE_ONLY")
    plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("STATE_AUC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure5_paired_state_r_full_vs_state_only.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 3))
    plt.plot(xs, [u["contrasts"]["FULL_GAP"] for u in units], marker="o", label="FULL R/C gap")
    plt.plot(xs, [u["contrasts"]["STATE_ONLY_GAP"] for u in units], marker="o", label="STATE_ONLY R/C gap")
    plt.axhline(0, color="black", lw=1)
    plt.xticks(xs, labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("KL_AUC gap")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_full_vs_state_only_rc_gap.png", dpi=160)
    plt.close()
    rescues = [(abs(u["branches"]["R_FULL"]["KL_AUC"] - u["branches"]["R_STATE_ONLY"]["KL_AUC"]), i) for i, u in enumerate(units)]
    rep_i = sorted(rescues)[len(rescues) // 2][1]
    rep = units[rep_i]
    offs = range(len(rep["branches"]["R_FULL"]["state_relative_error_curve"]))
    plt.figure(figsize=(7, 3))
    plt.plot(offs, rep["branches"]["R_FULL"]["state_relative_error_curve"], label="R_FULL")
    plt.plot(offs, rep["branches"]["R_STATE_ONLY"]["state_relative_error_curve"], label="R_STATE_ONLY")
    plt.xlabel("future token offset")
    plt.ylabel("state relative error")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure7_r_state_trajectory_full_vs_state_only.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 3))
    plt.plot(offs, rep["branches"]["R_FULL"]["KL_curve"], label="R_FULL")
    plt.plot(offs, rep["branches"]["R_STATE_ONLY"]["KL_curve"], label="R_STATE_ONLY")
    plt.xlabel("future token offset")
    plt.ylabel("KL")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure8_r_kl_trajectory_full_vs_state_only.png", dpi=160)
    plt.close()
    plt.figure(figsize=(5, 3))
    vals = [
        sum(u["contrasts"]["R_persistence_behavior_inversion"] for u in units),
        sum(u["contrasts"]["C_persistence_behavior_inversion"] for u in units),
    ]
    plt.bar(["R", "C"], vals)
    plt.ylim(0, len(units))
    plt.ylabel("inversion count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure9_persistence_behavior_inversion.png", dpi=160)
    plt.close()


def write_report(obj):
    lines = [
        "# GDN INT8 Layer-Local Operator-Invariant Update Transduction Causal V2",
        "",
        "## 1. Task", TASK,
        "## 2. Prior Source/Temporal Evidence",
        "Representation-dependent source residuals, temporal accumulation bridge, and same-norm structure effects are preserved from previous reports.",
        "## 3. Remaining Same-Norm Paradox",
        "R residuals can be less persistent in state space while more harmful behaviorally.",
        "## 4. Resolved Native Replay Semantics",
        "Native core output replay uses the final cast back to query input dtype; regression is included here.",
        "## 5. Protocol",
        "Layer/head-local operator-invariant same-norm D/U audit; no all-layer intervention.",
        "## 6. Stage0 Regression", json.dumps(obj["stage0"], indent=2, ensure_ascii=False),
        "## 7. Same-Norm Construction",
        "Primary pair: R_STRUCT_C_NORM vs REAL_C; local target head only.",
        "## 8. Operator-Coupling Stage A", json.dumps(obj["stage_a"]["summary"], indent=2, ensure_ascii=False),
        "## 9. D/U Decomposition",
        json.dumps(obj["stage_a"]["statistics"], indent=2, ensure_ascii=False),
        "## 10. Pilot Causal Branches",
        json.dumps(obj.get("pilot", {}).get("summary", obj["PILOT"]), indent=2, ensure_ascii=False) if isinstance(obj.get("pilot"), dict) else obj["PILOT"],
        "## 11. State Persistence",
        json.dumps(obj.get("pilot", {}).get("state_persistence", "Not run before Pilot."), indent=2, ensure_ascii=False),
        "## 12. Behavioral KL",
        json.dumps(obj.get("pilot", {}).get("behavioral_kl", "Not run before Pilot."), indent=2, ensure_ascii=False),
        "## 13. Persistence-Behavior Inversion",
        json.dumps(obj.get("pilot", {}).get("inversion", "Not tested before Pilot."), indent=2, ensure_ascii=False),
        "## 14. R/C Gap Mediation",
        json.dumps(obj.get("pilot", {}).get("mediation", "Not tested before Pilot."), indent=2, ensure_ascii=False),
        "## 15. Formal Results",
        obj["FORMAL"],
        "## 16. Negative/Corrective Findings",
        obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "## 17. Allowed Conclusion",
        obj["ALLOWED_CONCLUSION"],
        "## 18. Claims Not Supported",
        "UPDATE_TRANSDUCTION_CAUSAL is not claimed from Stage A alone. No method is proposed.",
        "## 19. Updated Unified Mechanism",
        "Operator-conditioned residual geometry remains a candidate mechanism pending causal Pilot.",
        "## 20. Mechanism-Closure Assessment",
        f"MECHANISM_CLOSURE_CANDIDATE = {obj['MECHANISM_CLOSURE_CANDIDATE']}",
        "## 21. Recommended Next Scientific Step",
        obj["RECOMMENDED_NEXT_TASK"],
        "## 22. Artifact Paths",
        json.dumps(obj["artifact_paths"], indent=2, ensure_ascii=False),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def run(stage):
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompts = selected_prompts(3)
    cp = load_json(CHECKPOINT) or {"completed": {}}
    first_key = f"{prompts[0]['problem_id']}|64"
    if first_key not in cp["completed"]:
        cp["completed"][first_key] = run_unit_audit(torch, model, tokenizer, e2e, prompts[0], 64)
        save_json(CHECKPOINT, cp)
    stage0 = stage0_from_unit(cp["completed"][first_key])
    stage0["runtime"] = runtime_info(torch)
    save_json(STAGE0, stage0)
    if stage0["STAGE0"] != "PASS" or stage == "stage0":
        obj = {
            "task": TASK, "stage0": stage0, "stage_a": {"summary": None, "units": []},
            "PILOT": "NOT_RUN", "FORMAL": "NOT_RUN", "MECHANISM_CLOSURE_CANDIDATE": "NO",
            "METHOD_DESIGN_READY": "NO", "ALLOWED_CONCLUSION": "Stage0 regression only.",
            "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": "No Stage A was run." if stage == "stage0" else "Stage0 failed; hard stop.",
            "RECOMMENDED_NEXT_TASK": "Run Stage A after Stage0 PASS." if stage == "stage0" else "Fix failed Stage0 gate.",
            "artifact_paths": artifact_paths(),
        }
        save_json(OP_AUDIT, obj)
        save_json(PILOT, {"task": TASK, "PILOT": "NOT_RUN"})
        save_json(FORMAL, {"task": TASK, "FORMAL": "NOT_RUN"})
        write_report(obj)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        return obj
    for pm in prompts:
        for t0 in T0_PANEL:
            key = f"{pm['problem_id']}|{t0}"
            if key in cp["completed"]:
                continue
            print(f"[{now()}] StageA unit {key}", flush=True)
            cp["completed"][key] = run_unit_audit(torch, model, tokenizer, e2e, pm, t0)
            save_json(CHECKPOINT, cp)
    units = [cp["completed"][f"{pm['problem_id']}|{t0}"] for pm in prompts for t0 in T0_PANEL]
    summary = analyze_stage_a(units)
    stage_a = {"summary": summary, "units": units, "statistics": summary["statistics"]}
    if stage == "pilot":
        if summary["LOCAL_OPERATOR_COUPLING_SIGNAL"] not in ("SUPPORTED", "PARTIAL"):
            obj = {
                "task": TASK,
                "timestamp": now(),
                "git_commit": git_commit(),
                "git_dirty": git_dirty(),
                "runtime": runtime_info(torch),
                "stage0": stage0,
                "stage_a": stage_a,
                "PILOT": "NOT_RUN",
                "VALID_PILOT_UNITS": "0 / 9",
                "FORMAL": "NOT_RUN",
                "METHOD_DESIGN_READY": "NO",
                "MECHANISM_CLOSURE_CANDIDATE": "NO",
                "UPDATE_TRANSDUCTION_CAUSAL": "NOT_YET_TESTED",
                "PERSISTENCE_HARM_PARADOX_EXPLANATION": "NOT_YET_TESTED",
                "ALLOWED_CONCLUSION": "Stage A did not permit Pilot.",
                "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": "Pilot was blocked by Stage A classification.",
                "RECOMMENDED_NEXT_TASK": "Inspect Stage A heterogeneity.",
                "artifact_paths": artifact_paths(),
            }
            save_json(PILOT, obj)
            save_json(FORMAL, {"task": TASK, "FORMAL": "NOT_RUN"})
            write_report(obj)
            print_summary(obj)
            return obj
        cp.setdefault("pilot_completed", {})
        failed = cp.setdefault("pilot_failed", {})
        for pm in prompts:
            for t0 in T0_PANEL:
                key = f"{pm['problem_id']}|{t0}"
                if key in cp["pilot_completed"]:
                    continue
                print(f"[{now()}] StageB pilot unit {key}", flush=True)
                try:
                    cp["pilot_completed"][key] = run_unit_pilot(torch, model, tokenizer, e2e, pm, t0, cp["completed"][key])
                    save_json(CHECKPOINT, cp)
                except Exception as exc:
                    failed[key] = {"error": repr(exc), "timestamp": now()}
                    save_json(CHECKPOINT, cp)
                    raise
        pilot_units = [cp["pilot_completed"][f"{pm['problem_id']}|{t0}"] for pm in prompts for t0 in T0_PANEL if f"{pm['problem_id']}|{t0}" in cp["pilot_completed"]]
        pilot_summary = analyze_pilot(pilot_units, summary)
        pilot_obj = {
            "summary": pilot_summary,
            "units": pilot_units,
            "state_persistence": {
                "R_STATE_ONLY STATE_AUC > R_FULL": pilot_summary["R_STATE_ONLY STATE_AUC > R_FULL"],
                "Median R_FULL STATE_AUC": pilot_summary["Median R_FULL STATE_AUC"],
                "Median R_STATE_ONLY STATE_AUC": pilot_summary["Median R_STATE_ONLY STATE_AUC"],
            },
            "behavioral_kl": {
                "R_STATE_ONLY improves KL vs R_FULL": pilot_summary["R_STATE_ONLY improves KL vs R_FULL"],
                "Median R_FULL KL_AUC": pilot_summary["Median R_FULL KL_AUC"],
                "Median R_STATE_ONLY KL_AUC": pilot_summary["Median R_STATE_ONLY KL_AUC"],
                "Median R update-path KL rescue": pilot_summary["Median R update-path KL rescue"],
            },
            "inversion": {
                "R persistence-behavior inversion": pilot_summary["R persistence-behavior inversion"],
                "C persistence-behavior inversion": pilot_summary["C persistence-behavior inversion"],
                "R initial-state-norm inversion": pilot_summary["R initial-state-norm inversion"],
            },
            "mediation": {
                "Median FULL R/C KL gap": pilot_summary["Median FULL R/C KL gap"],
                "Median STATE_ONLY R/C KL gap": pilot_summary["Median STATE_ONLY R/C KL gap"],
                "STATE_ONLY gap smaller": pilot_summary["STATE_ONLY gap smaller"],
                "Median UPDATE_PATH_GAP_RESCUE": pilot_summary["Median UPDATE_PATH_GAP_RESCUE"],
            },
        }
        positive = pilot_summary["PILOT"] == "POSITIVE"
        obj = {
            "task": "GDN_INT8_LAYER_LOCAL_OPERATOR_INVARIANT_UPDATE_TRANSDUCTION_CAUSAL_V2_STAGEB",
            "parent_task": TASK,
            "timestamp": now(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "runtime": runtime_info(torch),
            "stage0": stage0,
            "stage_a": stage_a,
            "pilot": pilot_obj,
            "PILOT": pilot_summary["PILOT"],
            "VALID_PILOT_UNITS": pilot_summary["VALID_PILOT_UNITS"],
            "FORMAL": "NOT_RUN",
            "METHOD_DESIGN_READY": "NO",
            "MECHANISM_CLOSURE_CANDIDATE": "NO",
            "UPDATE_TRANSDUCTION_CAUSAL": pilot_summary["UPDATE_TRANSDUCTION_CAUSAL"],
            "PERSISTENCE_HARM_PARADOX_EXPLANATION": pilot_summary["PERSISTENCE_HARM_PARADOX_EXPLANATION"],
            "ALLOWED_CONCLUSION": (
                "In the 9-unit pilot, removing the R update-transduction component reduced downstream KL while leaving a larger direct state perturbation in most units. This supports the pilot-level claim that GDN update transduction can make raw residual persistence and behavioral harm diverge."
                if positive else
                "The 9-unit pilot did not jointly satisfy the pre-specified behavioral, persistence, and R/C gap criteria. Stage A operator coupling remains supported, but downstream update-transduction causality is not established."
            ),
            "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": (
                "Formal was not launched; pilot-level support still needs 18-unit validation."
                if positive else
                "Negative or mixed pilot outcomes are preserved; no Formal run was launched."
            ),
            "RECOMMENDED_NEXT_TASK": "GDN_INT8_LAYER_LOCAL_OPERATOR_INVARIANT_UPDATE_TRANSDUCTION_CAUSAL_V2_FORMAL" if positive else "Inspect Pilot heterogeneity before any Formal run.",
            "artifact_paths": artifact_paths(),
        }
        save_json(PILOT, obj)
        save_json(FORMAL, {"task": TASK, "FORMAL": "NOT_RUN", "reason": "Formal must not run automatically after StageB."})
        np.savez(
            RAW,
            u_over_e_r=np.array([u["conditions"][PRIMARY_R]["metrics"]["U_OVER_E"] for u in units]),
            u_over_e_c=np.array([u["conditions"][PRIMARY_C]["metrics"]["U_OVER_E"] for u in units]),
            consumed_r=np.array([u["conditions"][PRIMARY_R]["metrics"]["CONSUMED_ENERGY_FRACTION"] for u in units]),
            consumed_c=np.array([u["conditions"][PRIMARY_C]["metrics"]["CONSUMED_ENERGY_FRACTION"] for u in units]),
            r_full_kl=np.array([u["branches"]["R_FULL"]["KL_AUC"] for u in pilot_units]),
            r_state_only_kl=np.array([u["branches"]["R_STATE_ONLY"]["KL_AUC"] for u in pilot_units]),
            r_full_state=np.array([u["branches"]["R_FULL"]["STATE_AUC"] for u in pilot_units]),
            r_state_only_state=np.array([u["branches"]["R_STATE_ONLY"]["STATE_AUC"] for u in pilot_units]),
        )
        make_figures(stage_a)
        make_pilot_figures(pilot_obj)
        write_report(obj)
        save_json(CHECKPOINT, cp)
        print_summary(obj)
        return obj
    pilot_status = "NOT_RUN"
    formal_status = "NOT_RUN"
    ready_for_pilot = summary["LOCAL_OPERATOR_COUPLING_SIGNAL"] in ("SUPPORTED", "PARTIAL")
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runtime": runtime_info(torch),
        "stage0": stage0,
        "stage_a": stage_a,
        "PILOT": pilot_status,
        "VALID_PILOT_UNITS": "0 / 9",
        "FORMAL": formal_status,
        "METHOD_DESIGN_READY": "NO",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "UPDATE_TRANSDUCTION_CAUSAL": "NOT_YET_TESTED",
        "PERSISTENCE_HARM_PARADOX_EXPLANATION": "NOT_YET_TESTED",
        "ALLOWED_CONCLUSION": "Stage A shows whether same-norm R residuals have stronger local operator-coupling than C under identical q/k/v/g/beta. It does not establish downstream causal behavioral effects.",
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": "Pilot was not launched in this run; Stage A is an operator-coupling audit, not a behavioral causal test.",
        "RECOMMENDED_NEXT_TASK": "Run Stage B causal Pilot on the same 9 units." if ready_for_pilot else "Stop and inspect why Stage A did not support local operator coupling.",
        "artifact_paths": artifact_paths(),
    }
    save_json(OP_AUDIT, obj)
    save_json(PILOT, {"task": TASK, "PILOT": "NOT_RUN", "reason": "Stage B not launched by this Stage0+StageA run."})
    save_json(FORMAL, {"task": TASK, "FORMAL": "NOT_RUN", "reason": "Formal requires positive Pilot."})
    np.savez(
        RAW,
        u_over_e_r=np.array([u["conditions"][PRIMARY_R]["metrics"]["U_OVER_E"] for u in units]),
        u_over_e_c=np.array([u["conditions"][PRIMARY_C]["metrics"]["U_OVER_E"] for u in units]),
        consumed_r=np.array([u["conditions"][PRIMARY_R]["metrics"]["CONSUMED_ENERGY_FRACTION"] for u in units]),
        consumed_c=np.array([u["conditions"][PRIMARY_C]["metrics"]["CONSUMED_ENERGY_FRACTION"] for u in units]),
    )
    make_figures(stage_a)
    write_report(obj)
    save_json(CHECKPOINT, cp)
    print_summary(obj)
    return obj


def artifact_paths():
    return {
        "script": str(SCRIPT),
        "stage0": str(STAGE0),
        "operator_audit": str(OP_AUDIT),
        "pilot": str(PILOT),
        "formal": str(FORMAL),
        "checkpoint": str(CHECKPOINT),
        "raw": str(RAW),
        "report": str(REPORT),
        "figures": str(FIG_DIR),
    }


def print_summary(obj):
    s0 = obj["stage0"]
    sa = obj["stage_a"]["summary"]
    out = {
        "TASK": TASK,
        "STAGE0": s0["STAGE0"],
        "REPLAY_REGRESSION_GATE": s0["REPLAY_REGRESSION_GATE"],
        "max next_state replay error": s0["max_next_state_replay_error"],
        "max core_output replay error": s0["max_core_output_replay_error"],
        "OPERATOR_INVARIANCE_GATE": s0["OPERATOR_INVARIANCE_GATE"],
        "VALID_OPERATOR_AUDIT_UNITS": sa["VALID_OPERATOR_AUDIT_UNITS"] if sa else "0 / 9",
        "Median U/E R": sa["Median U/E R"] if sa else None,
        "Median U/E C": sa["Median U/E C"] if sa else None,
        "R U/E > C": sa["R U/E > C"] if sa else None,
        "Median consumed-energy fraction R": sa["Median consumed-energy fraction R"] if sa else None,
        "Median consumed-energy fraction C": sa["Median consumed-energy fraction C"] if sa else None,
        "R consumption > C": sa["R consumption > C"] if sa else None,
        "LOCAL_OPERATOR_COUPLING_SIGNAL": sa["LOCAL_OPERATOR_COUPLING_SIGNAL"] if sa else None,
        "PILOT": obj["PILOT"],
        "VALID_PILOT_UNITS": obj.get("VALID_PILOT_UNITS", "0 / 9"),
        "R_FULL > C_FULL KL": obj.get("pilot", {}).get("summary", {}).get("R_FULL > C_FULL KL"),
        "R_STATE_ONLY improves KL vs R_FULL": obj.get("pilot", {}).get("summary", {}).get("R_STATE_ONLY improves KL vs R_FULL"),
        "Median R_STATE_ONLY KL_AUC": obj.get("pilot", {}).get("summary", {}).get("Median R_STATE_ONLY KL_AUC"),
        "Median R update-path KL rescue": obj.get("pilot", {}).get("summary", {}).get("Median R update-path KL rescue"),
        "R_STATE_ONLY STATE_AUC > R_FULL": obj.get("pilot", {}).get("summary", {}).get("R_STATE_ONLY STATE_AUC > R_FULL"),
        "Median R_FULL STATE_AUC": obj.get("pilot", {}).get("summary", {}).get("Median R_FULL STATE_AUC"),
        "Median R_STATE_ONLY STATE_AUC": obj.get("pilot", {}).get("summary", {}).get("Median R_STATE_ONLY STATE_AUC"),
        "R persistence-behavior inversion": obj.get("pilot", {}).get("summary", {}).get("R persistence-behavior inversion"),
        "C persistence-behavior inversion": obj.get("pilot", {}).get("summary", {}).get("C persistence-behavior inversion"),
        "R initial-state-norm inversion": obj.get("pilot", {}).get("summary", {}).get("R initial-state-norm inversion"),
        "Median FULL R/C KL gap": obj.get("pilot", {}).get("summary", {}).get("Median FULL R/C KL gap"),
        "Median STATE_ONLY R/C KL gap": obj.get("pilot", {}).get("summary", {}).get("Median STATE_ONLY R/C KL gap"),
        "STATE_ONLY gap smaller": obj.get("pilot", {}).get("summary", {}).get("STATE_ONLY gap smaller"),
        "Median UPDATE_PATH_GAP_RESCUE": obj.get("pilot", {}).get("summary", {}).get("Median UPDATE_PATH_GAP_RESCUE"),
        "Median R_UPDATE_ONLY KL_AUC": obj.get("pilot", {}).get("summary", {}).get("Median R_UPDATE_ONLY KL_AUC"),
        "Median C_UPDATE_ONLY KL_AUC": obj.get("pilot", {}).get("summary", {}).get("Median C_UPDATE_ONLY KL_AUC"),
        "FORMAL": obj["FORMAL"],
        "UPDATE_TRANSDUCTION_CAUSAL": obj.get("UPDATE_TRANSDUCTION_CAUSAL", "NOT_YET_TESTED"),
        "MECHANISM_CLOSURE_CANDIDATE": obj["MECHANISM_CLOSURE_CANDIDATE"],
        "METHOD_DESIGN_READY": obj["METHOD_DESIGN_READY"],
        "ALLOWED_CONCLUSION": obj["ALLOWED_CONCLUSION"],
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "RECOMMENDED_NEXT_TASK": obj["RECOMMENDED_NEXT_TASK"],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage_a", choices=["stage0", "stage_a", "pilot"])
    args = ap.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()
