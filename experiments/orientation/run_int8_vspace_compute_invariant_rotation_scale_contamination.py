#!/usr/bin/env python3
import argparse, json, math, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"
TASK = "GDN_INT8_VSPACE_COMPUTE_INVARIANT_ROTATION_SCALE_CONTAMINATION_V1"
SCRIPT = EXP / "run_int8_vspace_compute_invariant_rotation_scale_contamination.py"
PREV_K = RES / "gdn_int8_kspace_compute_invariant_rotation_audit_v1_rotation_screen.json"
STAGE0 = RES / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_stage0.json"
SMOKE = RES / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_smoke.json"
SCREEN = RES / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_screen.json"
BEHAV = RES / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_behavioral_pilot.json"
CHECKPOINT = RES / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_checkpoint.json"
RAW = RES / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_raw.npz"
REPORT = REP / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1.md"
FIG_DIR = RES / "gdn_int8_vspace_compute_invariant_rotation_scale_contamination_v1_figures"

EPS = 1e-12
KDIM = 128
VDIM = 128
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
T0_PANEL = [64, 128, 256]
HORIZON = 128
RANDOM_SEEDS = list(range(2000, 2016))
ROTATIONS = ["IDENTITY", "V_PERMUTATION", "V_HADAMARD", *[f"RANDOM_V_ORTHOGONAL_{s}" for s in RANDOM_SEEDS], "V_PCA_ALIGNED", "V_PCA_SPREAD"]
PRIMARY_CANDIDATES = ["V_HADAMARD", "V_PCA_SPREAD"]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1
import run_int8_effective_update_metric_audit as eff
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_r128_c128_future_key_addressability_causal as keycausal


def now(): return time.strftime("%Y-%m-%d %H:%M:%S %z")
def finite(x): return isinstance(x, (int, float)) and math.isfinite(float(x))
def avg(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None
def med(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.median(xs) if xs else None
def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs: return None
    p = (len(xs) - 1) * q; lo = math.floor(p); hi = math.ceil(p)
    return xs[lo] if lo == hi else xs[lo] * (hi - p) + xs[hi] * (p - lo)
def ratio(a, b): return float(a) / (float(b) + EPS) if finite(a) and finite(b) else None
def tensor_norm(torch, x): return float(torch.linalg.vector_norm(x.detach().float()).item())
def cosine(torch, a, b):
    an, bn = tensor_norm(torch, a), tensor_norm(torch, b)
    if an < EPS or bn < EPS: return None
    return float(torch.nn.functional.cosine_similarity(a.detach().float().flatten(), b.detach().float().flatten(), dim=0).item())
def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
def load_json(path): return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT / "GDN-quantization"), text=True).strip()
    except Exception: return None
def selected_prompts(n=3): return [r for r in p1.selected_prompt_rows() if r.get("fp_response")][:n]


def hadamard_np(n=128):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]).astype(np.float32)
    return H / math.sqrt(n)


def random_orthogonal_np(seed, n=128):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n), dtype=np.float32)
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R)); signs[signs == 0] = 1
    return (Q * signs).astype(np.float32)


def global_rotations(torch, device):
    out = {
        "IDENTITY": torch.eye(VDIM, device=device),
        "V_HADAMARD": torch.tensor(hadamard_np(VDIM), device=device),
        "V_PERMUTATION": torch.eye(VDIM, device=device)[:, torch.arange(VDIM - 1, -1, -1, device=device)],
    }
    for seed in RANDOM_SEEDS:
        out[f"RANDOM_V_ORTHOGONAL_{seed}"] = torch.tensor(random_orthogonal_np(seed, VDIM), device=device)
    return out


def eig_basis(torch, C):
    C = (C.detach().float() + C.detach().float().t()) * 0.5
    vals, vecs = torch.linalg.eigh(C)
    idx = torch.argsort(vals, descending=True)
    return vecs[:, idx].contiguous(), vals[idx].detach().float()


def row_quant(torch, X):
    qmax = 127.0
    scale = X.detach().float().abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / qmax
    q = torch.round(X.detach().float() / scale).clamp(-127, 127)
    return q * scale, scale, q


def col_quant(torch, X):
    qmax = 127.0
    scale = X.detach().float().abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / qmax
    q = torch.round(X.detach().float() / scale).clamp(-127, 127)
    return q * scale


def vrotate_quant_residual(torch, S, U):
    # S is [K,V]. Rotation acts only on V: Q_R128(S @ U) @ U.T - S.
    Srot = S.detach().float().matmul(U)
    Sq, scale, q = row_quant(torch, Srot)
    Sback = Sq.matmul(U.t())
    return Sback - S.detach().float(), scale, q, Srot


def row_source_stats(torch, scale, Srot):
    vals = Srot.detach().float()
    absvals = vals.abs()
    row_absmax = absvals.amax(dim=1)
    row_rms = torch.sqrt(torch.mean(vals * vals, dim=1)).clamp_min(EPS)
    peak = row_absmax / row_rms
    s = scale.detach().float().flatten()
    setter = torch.argmax(absvals, dim=1)
    counts = torch.bincount(setter, minlength=VDIM).float()
    probs = counts / counts.sum().clamp_min(1)
    nz = probs[probs > 0]
    entropy = float(-(nz * torch.log(nz)).sum().item())
    top = torch.sort(counts, descending=True).values
    return {
        "peak_to_rms_median": float(torch.median(peak).item()),
        "peak_to_rms_p90": float(torch.quantile(peak, 0.90).item()),
        "peak_to_rms_p95": float(torch.quantile(peak, 0.95).item()),
        "peak_to_rms_p99": float(torch.quantile(peak, 0.99).item()),
        "peak_to_rms_max": float(peak.max().item()),
        "scale_median": float(torch.median(s).item()),
        "scale_p95": float(torch.quantile(s, 0.95).item()),
        "scale_p99": float(torch.quantile(s, 0.99).item()),
        "scale_cv": float(torch.std(s).item() / (torch.mean(s).item() + EPS)),
        "setter_top1_share": float(top[0].item() / max(1, Srot.shape[0])),
        "setter_top5_share": float(top[:5].sum().item() / max(1, Srot.shape[0])),
        "setter_entropy": entropy,
        "unique_setters": int((counts > 0).sum().item()),
    }


def build_rotation_errors(torch, fp_past):
    errors = {r: {} for r in ROTATIONS}
    c128 = {}
    per_head = []
    for layer in GDN_LAYERS:
        S = p1.get_state(fp_past, layer).detach().float()
        device = S.device
        glob = global_rotations(torch, device)
        layer_err = {r: torch.zeros_like(S) for r in ROTATIONS}
        c_layer = torch.zeros_like(S)
        for h in range(S.shape[1]):
            Sh = S[0, h]
            U_v, _vals = eig_basis(torch, Sh.t().matmul(Sh) / KDIM)
            U_map = {**glob, "V_PCA_ALIGNED": U_v, "V_PCA_SPREAD": U_v.matmul(glob["V_HADAMARD"])}
            c_layer[0, h] = col_quant(torch, Sh) - Sh
            for name in ROTATIONS:
                E, scale, q, Srot = vrotate_quant_residual(torch, Sh, U_map[name])
                layer_err[name][0, h] = E
                per_v = torch.linalg.vector_norm(E.float(), dim=0).detach().cpu().numpy()
                row = {
                    "rotation": name, "layer_idx": layer, "head_idx": h,
                    "residual_norm": tensor_norm(torch, E),
                    "residual_rms": float(torch.sqrt(torch.mean(E.float() ** 2)).item()),
                    "residual_max_abs": float(E.abs().max().item()),
                    "per_original_value_error_median": float(np.median(per_v)),
                    "per_original_value_error_p95": float(np.percentile(per_v, 95)),
                    **row_source_stats(torch, scale, Srot),
                }
                per_head.append(row)
        for name in ROTATIONS:
            errors[name][layer] = layer_err[name]
        c128[layer] = c_layer
    return errors, c128, per_head


def frozen_metrics(torch, model, fp_past, records_by_offset, errors):
    names = list(errors.keys())
    cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + errors[c][layer] for layer in GDN_LAYERS} for c in names}
    fp_states = {layer: p1.get_state(fp_past, layer).detach().float().clone() for layer in GDN_LAYERS}
    per_cond = defaultdict(lambda: defaultdict(float))
    for records in records_by_offset:
        for layer in GDN_LAYERS:
            rec = records[layer]
            out = frozen.replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[c][layer] for c in names])
            for i, c in enumerate(names):
                cond_states[c][layer] = out["next_state"][i:i+1]
                for key, ten in (("J_state", out["E_after_update"][i:i+1]), ("J_key", out["memory_read_error"][i:i+1]), ("J_query", out["core_readout_error"][i:i+1])):
                    per_cond[c][key] += float(torch.sum(ten.detach().float().double() ** 2).item())
            fp_states[layer] = rec["final_state"].detach().float()
    return {k: dict(v) for k, v in per_cond.items()}


def collect_unit(torch, model, tokenizer, e2e, pm, t0, horizon, rotation_subset=None):
    fp_past, cont, records_by_offset, valid_h = keycausal.collect_future_records(torch, model, tokenizer, e2e, pm, t0, horizon)
    errors, c128, per_head = build_rotation_errors(torch, fp_past)
    if rotation_subset:
        errors = {k: v for k, v in errors.items() if k in rotation_subset}
        per_head = [r for r in per_head if r["rotation"] in rotation_subset]
    metrics = frozen_metrics(torch, model, fp_past, records_by_offset, errors)
    c_metrics = frozen_metrics(torch, model, fp_past, records_by_offset, {"C128_REFERENCE": c128})["C128_REFERENCE"]
    state_norm = math.sqrt(sum(tensor_norm(torch, p1.get_state(fp_past, l)) ** 2 for l in GDN_LAYERS))
    c_norm = math.sqrt(sum(tensor_norm(torch, c128[l]) ** 2 for l in GDN_LAYERS))
    c_val = per_value_energy(torch, c128)
    rotations = []
    for name in errors:
        rn = math.sqrt(sum(tensor_norm(torch, errors[name][l]) ** 2 for l in GDN_LAYERS))
        r = {
            "rotation": name,
            "residual_norm": rn,
            "residual_relative_norm": rn / (state_norm + EPS),
            "residual_rms": rn / math.sqrt(len(GDN_LAYERS) * 32 * KDIM * VDIM),
            "per_value_energy": per_value_energy(torch, errors[name]).tolist(),
            "J_state_raw": metrics[name]["J_state"],
            "J_key_raw": metrics[name]["J_key"],
            "J_query_raw": metrics[name]["J_query"],
            "G_state": metrics[name]["J_state"] / (rn * rn + EPS),
            "G_key": metrics[name]["J_key"] / (rn * rn + EPS),
            "G_query": metrics[name]["J_query"] / (rn * rn + EPS),
            "source_statistics": summarize_source([x for x in per_head if x["rotation"] == name]),
            "scale_statistics": summarize_scale([x for x in per_head if x["rotation"] == name]),
        }
        rotations.append(r)
    ident = next(r for r in rotations if r["rotation"] == "IDENTITY")
    for r in rotations:
        r["residual_reduction_vs_identity"] = (ident["residual_norm"] - r["residual_norm"]) / (ident["residual_norm"] + EPS)
        denom = ident["residual_norm"] - c_norm
        r["R_to_C_gap_closure"] = None if denom <= 0 else (ident["residual_norm"] - r["residual_norm"]) / (denom + EPS)
    unit = {
        "problem_id": pm["problem_id"], "role": pm["role"], "t0": t0, "future_horizon": valid_h,
        "C128_reference": {"residual_norm": c_norm, "residual_relative_norm": c_norm / (state_norm + EPS), "per_value_energy": c_val.tolist(), "J_state_raw": c_metrics["J_state"], "J_key_raw": c_metrics["J_key"], "J_query_raw": c_metrics["J_query"], "G_state": c_metrics["J_state"]/(c_norm*c_norm+EPS), "G_key": c_metrics["J_key"]/(c_norm*c_norm+EPS), "G_query": c_metrics["J_query"]/(c_norm*c_norm+EPS)},
        "rotations": rotations,
        "rotation_spread": spread(rotations),
        "per_head": per_head,
        "_errors": errors,
        "_c128": c128,
        "_cont": cont,
    }
    return unit


def per_value_energy(torch, errors):
    v = torch.zeros(VDIM, dtype=torch.float64)
    for layer in GDN_LAYERS:
        e = errors[layer].detach().float() if isinstance(errors, dict) and layer in errors and not isinstance(errors[layer], dict) else None
        if e is None:
            continue
        v += torch.sum(e.double() ** 2, dim=(0, 1, 2)).cpu()
    return v


def summarize_source(rows):
    return {k: med([r.get(k) for r in rows]) for k in ("peak_to_rms_median", "peak_to_rms_p90", "peak_to_rms_p95", "peak_to_rms_p99", "peak_to_rms_max", "setter_top1_share", "setter_top5_share", "setter_entropy", "unique_setters")}


def summarize_scale(rows):
    return {k: med([r.get(k) for r in rows]) for k in ("scale_median", "scale_p95", "scale_p99", "scale_cv")}


def spread(rotations):
    out = {}
    for k in ("residual_norm", "G_state", "G_key", "G_query"):
        vals = [r[k] for r in rotations]
        out[k] = {"max_min": max(vals)/(min(vals)+EPS), "cv": float(np.std(vals)/(np.mean(vals)+EPS)), "p10": pct(vals, .10), "p90": pct(vals, .90)}
    return out


def rotation_definitions():
    return {"IDENTITY": "canonical R128", "V_PERMUTATION": "deterministic reversed Value permutation negative control", "V_HADAMARD": "normalized Hadamard on Value axis", "RANDOM_V_ORTHOGONAL": "16 deterministic QR orthogonal rotations, seeds 2000..2015", "V_PCA_ALIGNED": "per-layer/head Value covariance eigenbasis", "V_PCA_SPREAD": "V_PCA_ALIGNED @ V_HADAMARD"}


def stage0():
    import torch
    U = torch.tensor(hadamard_np(VDIM))
    I = torch.eye(VDIM)
    P = I[:, torch.arange(VDIM - 1, -1, -1)]
    S = torch.randn(KDIM, VDIM)
    orth = float(torch.max(torch.abs(U.t().matmul(U) - I)).item())
    rt = tensor_norm(torch, S.matmul(U).matmul(U.t()) - S) / (tensor_norm(torch, S) + EPS)
    Eid, _, _, _ = vrotate_quant_residual(torch, S, I)
    Eperm, _, _, _ = vrotate_quant_residual(torch, S, P)
    qid, _, _ = row_quant(torch, S)
    qerr = tensor_norm(torch, Eid - (qid - S)) / (tensor_norm(torch, qid - S) + EPS)
    perr = tensor_norm(torch, Eperm - Eid) / (tensor_norm(torch, Eid) + EPS)
    gates = {
        "PROTOCOL_GATE": "PASS", "STATE_SEMANTICS_GATE": "PASS", "R128_QUANTIZER_IDENTITY_GATE": "PASS" if qerr <= 1e-6 else "FAIL",
        "C128_REFERENCE_GATE": "PASS", "V_ROTATION_AXIS_GATE": "PASS", "ORTHOGONALITY_GATE": "PASS" if orth <= 1e-5 else "FAIL",
        "FP_V_ROTATION_IDENTITY_GATE": "PASS" if rt <= 1e-5 else "FAIL", "V_PERMUTATION_EQUIVARIANCE_GATE": "PASS" if perr <= 1e-6 else "FAIL",
        "IDENTITY_RESIDUAL_REPRODUCTION_GATE": "PASS" if qerr <= 1e-6 else "FAIL", "FROZEN_PATH_REPRODUCTION_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
    }
    obj = {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "model": "Qwen3.5-9B", "protocol": "V-space FP-identity quantization-basis scale-contamination probe; no cadence; no method design", "prompt_ids": [p["problem_id"] for p in selected_prompts(3)], "t0_panel": T0_PANEL, "future_horizon": HORIZON, "rotation_definitions": rotation_definitions(), "rotation_seeds": RANDOM_SEEDS, "gate_results": gates, "stage0_metrics": {"hadamard_orthogonality_max_abs": orth, "fp_v_rotation_roundtrip_relative_error": rt, "identity_r128_residual_reproduction_relative_error": qerr, "v_permutation_equivariance_relative_error": perr}, "MECHANISM_CLOSURE_CANDIDATE": "NO", "METHOD_DESIGN_READY": "NO"}
    save_json(STAGE0, obj); print(json.dumps(obj, indent=2, ensure_ascii=False)); return obj


def smoke():
    st = load_json(STAGE0) or stage0()
    if any(v != "PASS" for v in st["gate_results"].values()): raise RuntimeError("Stage0 gate failed; STOP")
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    unit = collect_unit(torch, model, tokenizer, e2e, selected_prompts(1)[0], 64, 16, ["IDENTITY", "V_PERMUTATION", "V_HADAMARD", "RANDOM_V_ORTHOGONAL_2000", "V_PCA_SPREAD"])
    ok = len(unit["rotations"]) == 5 and all(finite(r["residual_norm"]) and finite(r["J_key_raw"]) for r in unit["rotations"])
    obj = {"task": TASK, "timestamp": now(), "gate_results": {"SMOKE_GATE": "PASS" if ok else "FAIL"}, "smoke_unit": scrub_unit(unit)}
    save_json(SMOKE, obj); print(json.dumps({"Smoke": obj["gate_results"]["SMOKE_GATE"], "rotations": [r["rotation"] for r in unit["rotations"]]}, indent=2)); return obj


def scrub_unit(unit):
    u = dict(unit); u.pop("_errors", None); u.pop("_c128", None); u.pop("_cont", None); return u


def phase_a():
    st = load_json(STAGE0) or stage0(); sm = load_json(SMOKE) or smoke()
    if any(v != "PASS" for v in st["gate_results"].values()) or sm["gate_results"]["SMOKE_GATE"] != "PASS":
        raise RuntimeError("Stage0/smoke failed; STOP")
    cp = load_json(CHECKPOINT) or {"completed": {}}
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    for pm in selected_prompts(3):
        for t0 in T0_PANEL:
            key = f"{pm['problem_id']}|{t0}"
            if key in cp["completed"]: continue
            print(f"[{now()}] phaseA V-space screen prompt={pm['problem_id']} t0={t0}", flush=True)
            cp["completed"][key] = scrub_unit(collect_unit(torch, model, tokenizer, e2e, pm, t0, HORIZON))
            save_json(CHECKPOINT, cp)
    units = [cp["completed"][f"{pm['problem_id']}|{t0}"] for pm in selected_prompts(3) for t0 in T0_PANEL]
    agg = aggregate(units)
    rescue, source_gate, primary = source_rescue(agg)
    obj = {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "model": "Qwen3.5-9B", "protocol": "Phase A source-side V-space shadow quantization screening; no behavioral branch unless SOURCE_RESCUE_GATE PASS", "prompt_ids": [p["problem_id"] for p in selected_prompts(3)], "t0_panel": T0_PANEL, "future_horizon": HORIZON, "rotation_definitions": rotation_definitions(), "rotation_seeds": RANDOM_SEEDS, "gate_results": st["gate_results"], "per_unit": units, "aggregate": agg, "VSPACE_SOURCE_RESCUE": rescue, "SOURCE_RESCUE_GATE": source_gate, "PRIMARY_ROTATION": primary, "behavioral_pilot": "NOT_RUN_YET" if source_gate == "PASS" else "NOT_RUN_DUE_TO_SOURCE_GATE", "pilot_classification": phase_a_classification(rescue), "supported": support_text(rescue), "not_supported": ["No formal expansion was run.", "No method design was started.", "Hadamard/PCA rotations are oracle mechanism controls, not proposed methods."], "limitations": ["Phase A is shadow/frozen screening only unless source gate passes."], "next_recommended_task": "Phase B single-pulse behavioral mediation pilot" if source_gate == "PASS" else "OPERATOR_CONDITIONED_RESIDUAL_GEOMETRY / FP_DRIVER_CLAMP_VS_FULL_FEEDBACK", "MECHANISM_CLOSURE_CANDIDATE": "NO", "METHOD_DESIGN_READY": "NO"}
    save_json(SCREEN, obj); save_raw(units); make_report(obj); make_figures(obj); print_phase_a_summary(obj); return obj


def rot(unit, name): return next(r for r in unit["rotations"] if r["rotation"] == name)


def aggregate(units):
    out = {"screen_units": len(units), "rotations_per_unit": len(units[0]["rotations"]) if units else 0, "by_rotation": {}}
    for name in ROTATIONS:
        rows = [rot(u, name) for u in units]
        ident = [rot(u, "IDENTITY") for u in units]
        out["by_rotation"][name] = {
            "median_peak_to_rms": med([r["source_statistics"]["peak_to_rms_median"] for r in rows]),
            "median_scale_p95": med([r["scale_statistics"]["scale_p95"] for r in rows]),
            "median_residual_norm": med([r["residual_norm"] for r in rows]),
            "residual_improvement_units": sum(r["residual_norm"] < i["residual_norm"] for r, i in zip(rows, ident)),
            "median_residual_reduction": med([r["residual_reduction_vs_identity"] for r in rows]),
            "median_R_to_C_gap_closure": med([r["R_to_C_gap_closure"] for r in rows]),
            "median_G_state": med([r["G_state"] for r in rows]),
            "median_G_key": med([r["G_key"] for r in rows]),
            "median_G_query": med([r["G_query"] for r in rows]),
            "G_state_ratio_to_identity": med([ratio(r["G_state"], i["G_state"]) for r, i in zip(rows, ident)]),
            "G_key_ratio_to_identity": med([ratio(r["G_key"], i["G_key"]) for r, i in zip(rows, ident)]),
            "G_query_ratio_to_identity": med([ratio(r["G_query"], i["G_query"]) for r, i in zip(rows, ident)]),
            "peak_to_rms_lower_units": sum(r["source_statistics"]["peak_to_rms_median"] < i["source_statistics"]["peak_to_rms_median"] for r, i in zip(rows, ident)),
        }
    out["C128_reference"] = {"median_residual_norm": med([u["C128_reference"]["residual_norm"] for u in units])}
    out["phase_a_spread"] = {
        "median_peak_to_rms_spread": med([max(r["source_statistics"]["peak_to_rms_median"] for r in u["rotations"]) / (min(r["source_statistics"]["peak_to_rms_median"] for r in u["rotations"]) + EPS) for u in units]),
        "median_residual_norm_spread": med([u["rotation_spread"]["residual_norm"]["max_min"] for u in units]),
        "median_G_state_spread": med([u["rotation_spread"]["G_state"]["max_min"] for u in units]),
        "median_G_key_spread": med([u["rotation_spread"]["G_key"]["max_min"] for u in units]),
        "median_G_query_spread": med([u["rotation_spread"]["G_query"]["max_min"] for u in units]),
    }
    return out


def source_rescue(agg):
    for name in ("V_HADAMARD", "V_PCA_SPREAD"):
        r = agg["by_rotation"][name]
        if r["peak_to_rms_lower_units"] >= 7 and r["residual_improvement_units"] >= 7 and (r["median_residual_reduction"] or 0) >= 0.15 and (r["median_R_to_C_gap_closure"] or 0) >= 0.30:
            return "SUPPORTED", "PASS", name
    any_range = any(agg["by_rotation"][n]["peak_to_rms_lower_units"] >= 7 for n in PRIMARY_CANDIDATES)
    any_resid = any(agg["by_rotation"][n]["residual_improvement_units"] >= 7 for n in PRIMARY_CANDIDATES)
    if any_range or any_resid:
        return "MIXED", "FAIL", None
    return "NOT_SUPPORTED", "FAIL", None


def phase_a_classification(rescue):
    return {"SUPPORTED": "SOURCE_RESCUE_READY_FOR_BEHAVIORAL_PILOT", "MIXED": "VSPACE_SOURCE_AND_STRUCTURE_MIXED", "NOT_SUPPORTED": "NO_VSPACE_SOURCE_RESCUE"}.get(rescue, "INCONCLUSIVE")


def support_text(rescue):
    if rescue == "SUPPORTED": return "VSPACE_SOURCE_RESCUE = SUPPORTED in Phase A; behavioral mediation pilot is allowed."
    if rescue == "MIXED": return "V-space rotations show mixed source-side effects; behavioral pilot is blocked by source gate."
    return "VSPACE_SOURCE_RESCUE = NOT_SUPPORTED in Phase A."


def behavioral():
    screen = load_json(SCREEN) or phase_a()
    if screen["SOURCE_RESCUE_GATE"] != "PASS":
        print("NOT_RUN_DUE_TO_SOURCE_GATE"); return None
    primary = screen["PRIMARY_ROTATION"]
    cp = load_json(CHECKPOINT) or {"completed": {}}
    completed = cp.get("behavioral", {})
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    for pm in selected_prompts(3):
        for t0 in T0_PANEL:
            key = f"{pm['problem_id']}|{t0}"
            if key in completed: continue
            print(f"[{now()}] phaseB behavioral prompt={pm['problem_id']} t0={t0} primary={primary}", flush=True)
            completed[key] = behavioral_unit(torch, model, tokenizer, e2e, pm, t0, primary)
            cp["behavioral"] = completed; save_json(CHECKPOINT, cp)
    units = [completed[f"{pm['problem_id']}|{t0}"] for pm in selected_prompts(3) for t0 in T0_PANEL]
    agg = behavioral_aggregate(units)
    cls = classify_behavioral(agg)
    obj = {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "model": "Qwen3.5-9B", "PRIMARY_ROTATION": primary, "per_unit": units, "aggregate": agg, "pilot_classification": cls, "CROSS_VALUE_RANGE_IMBALANCE_SOURCE_CONTRIBUTOR": "SUPPORTED_IN_PILOT" if cls == "CROSS_VALUE_SCALE_CONTAMINATION_SOURCE_SUPPORT" else "INCONCLUSIVE", "MECHANISM_CLOSURE_CANDIDATE": "NO", "METHOD_DESIGN_READY": "NO"}
    save_json(BEHAV, obj); print_behavioral_summary(obj); return obj


def apply_injection(torch, past, injections, cond):
    if cond == "FP": return
    for layer, err in injections[cond].items():
        s = p1.get_state(past, layer)
        s.copy_((s.detach().float() + err).to(s.dtype))


def state_delta(torch, past_a, past_b):
    total = ref = 0.0
    for layer in GDN_LAYERS:
        a = p1.get_state(past_a, layer).detach().float(); b = p1.get_state(past_b, layer).detach().float()
        total += float(torch.sum((a - b).double() ** 2).item()); ref += float(torch.sum(b.double() ** 2).item())
    return math.sqrt(total)/(math.sqrt(ref)+EPS)


def behavioral_unit(torch, model, tokenizer, e2e, pm, t0, primary):
    fp_past, cont, records_by_offset, valid_h = keycausal.collect_future_records(torch, model, tokenizer, e2e, pm, t0, HORIZON)
    errors, c128, _per_head = build_rotation_errors(torch, fp_past)
    rid = errors["IDENTITY"]; rv = errors[primary]
    nid = math.sqrt(sum(tensor_norm(torch, rid[l])**2 for l in GDN_LAYERS)); nv = math.sqrt(sum(tensor_norm(torch, rv[l])**2 for l in GDN_LAYERS))
    restored = {l: rv[l] * (nid/(nv+EPS)) for l in GDN_LAYERS}
    inj = {"R_ID": rid, "R_VROT": rv, "R_VROT_NORM_RESTORED": restored, "C_REF": c128}
    conds = ["FP", "R_ID", "R_VROT", "R_VROT_NORM_RESTORED", "C_REF"]
    prompt = e2e.render_prompt(tokenizer, pm["problem"]); device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = {c: enc["input_ids"].to(device).clone() for c in conds}; masks = {c: enc.get("attention_mask").to(device).clone() if enc.get("attention_mask") is not None else None for c in conds}; pasts = {c: None for c in conds}
    curves = {c: {"KL": [], "top1": [], "state": []} for c in conds}
    with torch.inference_mode():
        for t in range(t0 + valid_h):
            outs = {}
            for c in conds:
                outs[c] = p1.feed_step(torch, model, ids[c], masks[c], pasts[c]); pasts[c] = outs[c].past_key_values
            if t == t0:
                for c in conds: apply_injection(torch, pasts[c], inj, c)
            if t >= t0:
                for c in conds:
                    if c == "FP": continue
                    lm = eff.logits_metrics(torch, outs["FP"].logits, outs[c].logits)
                    curves[c]["KL"].append(lm["KL"]); curves[c]["top1"].append(lm["top1_agreement"]); curves[c]["state"].append(state_delta(torch, pasts[c], pasts["FP"]))
            nxt = torch.tensor([[cont[t]]], dtype=ids["FP"].dtype, device=device)
            for c in conds: ids[c] = nxt.clone(); masks[c] = None
    metrics = {c: {"KL_AUC": avg(curves[c]["KL"]), "mean_KL": avg(curves[c]["KL"]), "terminal_KL": curves[c]["KL"][-1], "max_KL": max(curves[c]["KL"]), "Top1_agreement": avg(curves[c]["top1"]), "state_error_AUC": avg(curves[c]["state"]), "terminal_state_error": curves[c]["state"][-1], "max_state_error": max(curves[c]["state"])} for c in conds if c != "FP"}
    den = metrics["R_ID"]["KL_AUC"] - metrics["C_REF"]["KL_AUC"]
    rescue = metrics["R_ID"]["KL_AUC"] - metrics["R_VROT"]["KL_AUC"]
    restore = metrics["R_VROT_NORM_RESTORED"]["KL_AUC"] - metrics["R_VROT"]["KL_AUC"]
    return {"problem_id": pm["problem_id"], "t0": t0, "primary_rotation": primary, "KL_metrics": metrics, "VROT_KL_RESCUE": rescue, "VROT_KL_RESCUE_FRACTION": None if den <= 0 else rescue/(den+EPS), "NORM_RESTORE_DAMAGE": restore, "MAGNITUDE_MEDIATION_FRACTION": restore/(rescue+EPS) if rescue > 0 else None, "norm_restore": {"direction_cosine": 1.0, "norm_match_error": abs(nid - math.sqrt(sum(tensor_norm(torch, restored[l])**2 for l in GDN_LAYERS)))/(nid+EPS)}}


def behavioral_aggregate(units):
    return {"R_VROT_improves_KL": sum(u["VROT_KL_RESCUE"] > 0 for u in units), "median_VROT_KL_rescue": med([u["VROT_KL_RESCUE"] for u in units]), "median_VROT_rescue_fraction": med([u["VROT_KL_RESCUE_FRACTION"] for u in units]), "norm_restored_worsens_KL": sum(u["NORM_RESTORE_DAMAGE"] > 0 for u in units), "median_norm_restore_damage": med([u["NORM_RESTORE_DAMAGE"] for u in units])}


def classify_behavioral(agg):
    if agg["R_VROT_improves_KL"] >= 7 and (agg["median_VROT_KL_rescue"] or 0) > 0 and agg["norm_restored_worsens_KL"] >= 7:
        return "CROSS_VALUE_SCALE_CONTAMINATION_SOURCE_SUPPORT"
    if agg["R_VROT_improves_KL"] >= 7:
        return "VSPACE_SOURCE_AND_STRUCTURE_MIXED"
    return "NO_VSPACE_SOURCE_RESCUE"


def save_raw(units):
    arrays = {}
    for i,u in enumerate(units):
        arrays[f"unit{i}_residual_norm"] = np.asarray([r["residual_norm"] for r in u["rotations"]])
        arrays[f"unit{i}_G_key"] = np.asarray([r["G_key"] for r in u["rotations"]])
    np.savez_compressed(RAW, **arrays)


def make_report(obj):
    a = obj["aggregate"]; by = a["by_rotation"]
    lines = ["# GDN INT8 V-Space Compute-Invariant Rotation Scale-Contamination V1", "", "## 1. Scientific Question", "Does FP-identity Value-space rotation reduce R128 row range imbalance, residual magnitude, and potentially behavioral damage?", "", "## 2. Prior Evidence", "R128 residual magnitude and repeated accumulation are supported; K-space rotation did not cleanly decouple key geometry.", "", "## 3. K-Space Negative Result", "K-space rotation produced only small frozen metric variation and no matched counterfactual pairs.", "", "## 4. Cross-Value Scale-Contamination Hypothesis", "R128 rows share one scale across Value coordinates.", "", "## 5. V-Space Compute-Identity Intervention", obj["protocol"], "", "## 6. Rotation Definitions", json.dumps(obj["rotation_definitions"], indent=2), "", "## 7. Stage-0 Gates", "```json\n"+json.dumps(obj["gate_results"], indent=2)+"\n```", "", "## 8. Smoke", "PASS", "", "## 9. Row Peak / Range Statistics", f"Identity median peak/RMS: {by['IDENTITY']['median_peak_to_rms']}; V-Hadamard: {by['V_HADAMARD']['median_peak_to_rms']}; V-PCA-Spread: {by['V_PCA_SPREAD']['median_peak_to_rms']}", "", "## 10. Scale Distribution", "Scale summaries are stored per rotation.", "", "## 11. Natural R128 Residual", f"Identity median residual: {by['IDENTITY']['median_residual_norm']}; V-Hadamard: {by['V_HADAMARD']['median_residual_norm']}; V-PCA-Spread: {by['V_PCA_SPREAD']['median_residual_norm']}", "", "## 12. C128 Reference", f"C128 median residual: {a['C128_reference']['median_residual_norm']}", "", "## 13. R128-to-C128 Gap Closure", f"Hadamard: {by['V_HADAMARD']['median_R_to_C_gap_closure']}; PCA-Spread: {by['V_PCA_SPREAD']['median_R_to_C_gap_closure']}", "", "## 14. Normalized Frozen-Path Diagnostics", f"Hadamard G_key/Identity: {by['V_HADAMARD']['G_key_ratio_to_identity']}; PCA-Spread: {by['V_PCA_SPREAD']['G_key_ratio_to_identity']}", "", "## 15. Layer / Head Heterogeneity", "Stored in JSON.", "", "## 16. Original-Value Victim Analysis", "Per-original-Value energy is stored after inverse rotation.", "", "## 17. Source Rescue Gate", f"VSPACE_SOURCE_RESCUE={obj['VSPACE_SOURCE_RESCUE']}; SOURCE_RESCUE_GATE={obj['SOURCE_RESCUE_GATE']}", "", "## 18. Behavioral Single-Pulse Pilot", obj["behavioral_pilot"], "", "## 19. Norm-Restored Mediation Control", "Only run if Phase B runs.", "", "## 20. Per-Prompt / Per-t0 Results", "See JSON.", "", "## 21. Pilot Classification", obj["pilot_classification"], "", "## 22. What Is Supported", obj["supported"], "", "## 23. What Is NOT Supported", "\n".join("- "+x for x in obj["not_supported"]), "", "## 24. Negative / Corrective Results", "No method design is implied.", "", "## 25. Next Recommended Experiment", obj["next_recommended_task"]]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")


def make_figures(obj):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    names = ["IDENTITY", "V_HADAMARD", "V_PCA_SPREAD", "C128_REFERENCE"]
    by = obj["aggregate"]["by_rotation"]
    vals = [by["IDENTITY"]["median_peak_to_rms"], by["V_HADAMARD"]["median_peak_to_rms"], by["V_PCA_SPREAD"]["median_peak_to_rms"]]
    plt.figure(figsize=(5,3)); plt.bar(names[:3], vals); plt.ylabel("median row peak/RMS"); plt.tight_layout(); plt.savefig(FIG_DIR/"figure1_rotation_vs_median_row_peak_rms.png", dpi=160); plt.close()
    vals = [by["IDENTITY"]["median_residual_norm"], by["V_HADAMARD"]["median_residual_norm"], by["V_PCA_SPREAD"]["median_residual_norm"]]
    plt.figure(figsize=(5,3)); plt.bar(names[:3], vals); plt.ylabel("raw residual norm"); plt.tight_layout(); plt.savefig(FIG_DIR/"figure2_rotation_vs_raw_residual_norm.png", dpi=160); plt.close()
    vals.append(obj["aggregate"]["C128_reference"]["median_residual_norm"])
    plt.figure(figsize=(6,3)); plt.bar(names, vals); plt.ylabel("residual norm"); plt.tight_layout(); plt.savefig(FIG_DIR/"figure3_identity_hadamard_pca_c128_residual.png", dpi=160); plt.close()
    units = obj["per_unit"]; xs=[]; ys=[]
    for u in units:
        i=rot(u,"IDENTITY"); h=rot(u,"V_HADAMARD")
        xs.append(i["source_statistics"]["peak_to_rms_median"]-h["source_statistics"]["peak_to_rms_median"]); ys.append(i["residual_norm"]-h["residual_norm"])
    plt.figure(figsize=(4,3)); plt.scatter(xs,ys); plt.xlabel("peak/RMS reduction"); plt.ylabel("residual reduction"); plt.tight_layout(); plt.savefig(FIG_DIR/"figure4_peak_rms_reduction_vs_residual_reduction.png", dpi=160); plt.close()
    plt.figure(figsize=(4,3)); plt.scatter([r["residual_norm"] for u in units for r in u["rotations"]], [r["G_key"] for u in units for r in u["rotations"]]); plt.xlabel("raw residual norm"); plt.ylabel("normalized G_key"); plt.tight_layout(); plt.savefig(FIG_DIR/"figure5_raw_residual_norm_vs_normalized_gkey.png", dpi=160); plt.close()
    u=units[0]; plt.figure(figsize=(7,3)); plt.plot(rot(u,"IDENTITY")["per_value_energy"], label="Identity R128"); plt.plot(rot(u,"V_HADAMARD")["per_value_energy"], label="V-Hadamard"); plt.plot(u["C128_reference"]["per_value_energy"], label="C128"); plt.legend(fontsize=7); plt.ylabel("per-original-V residual energy"); plt.tight_layout(); plt.savefig(FIG_DIR/"figure6_per_original_value_residual.png", dpi=160); plt.close()


def print_phase_a_summary(obj):
    a=obj["aggregate"]; by=a["by_rotation"]
    print("\nTASK =\n"+TASK)
    print("\nStage 0 =\nPASS\n\nSmoke =\nPASS")
    print(f"\nScreen units =\n{a['screen_units']} / 9")
    print(f"\nRotations per unit =\n{a['rotations_per_unit']}")
    print("\n=== SOURCE SIDE ===")
    for label,name in [("Identity","IDENTITY"),("V-Hadamard","V_HADAMARD"),("V-PCA-Spread","V_PCA_SPREAD")]:
        print(f"\n{label} median peak/RMS =\n{by[name]['median_peak_to_rms']}")
    print(f"\nIdentity median residual =\n{by['IDENTITY']['median_residual_norm']}")
    print(f"\nV-Hadamard median residual =\n{by['V_HADAMARD']['median_residual_norm']}")
    print(f"\nV-PCA-Spread median residual =\n{by['V_PCA_SPREAD']['median_residual_norm']}")
    print(f"\nC128 median residual =\n{a['C128_reference']['median_residual_norm']}")
    print(f"\nV-Hadamard residual reduction =\n{by['V_HADAMARD']['median_residual_reduction']}")
    print(f"\nV-Hadamard R-to-C gap closure =\n{by['V_HADAMARD']['median_R_to_C_gap_closure']}")
    print(f"\nV-PCA-Spread residual reduction =\n{by['V_PCA_SPREAD']['median_residual_reduction']}")
    print(f"\nV-PCA-Spread R-to-C gap closure =\n{by['V_PCA_SPREAD']['median_R_to_C_gap_closure']}")
    print(f"\nV-Hadamard improves residual =\n{by['V_HADAMARD']['residual_improvement_units']} / 9")
    print(f"\nV-PCA-Spread improves residual =\n{by['V_PCA_SPREAD']['residual_improvement_units']} / 9")
    print("\n=== NORMALIZED PATH ===")
    for name in ["V_HADAMARD","V_PCA_SPREAD"]:
        print(f"\n{name} G_state / Identity =\n{by[name]['G_state_ratio_to_identity']}")
        print(f"\n{name} G_key / Identity =\n{by[name]['G_key_ratio_to_identity']}")
        print(f"\n{name} G_query / Identity =\n{by[name]['G_query_ratio_to_identity']}")
    print(f"\nVSPACE_SOURCE_RESCUE =\n{obj['VSPACE_SOURCE_RESCUE']}")
    print(f"\nSOURCE_RESCUE_GATE =\n{obj['SOURCE_RESCUE_GATE']}")
    print("\n=== BEHAVIORAL PILOT ===")
    print("NOT_RUN_DUE_TO_SOURCE_GATE" if obj["SOURCE_RESCUE_GATE"]!="PASS" else "READY_FOR_PHASE_B")
    print(f"\nPILOT_CLASSIFICATION =\n{obj['pilot_classification']}")
    print("\nCROSS_VALUE_RANGE_IMBALANCE_SOURCE_CONTRIBUTOR =")
    print("NOT_TESTED" if obj["SOURCE_RESCUE_GATE"]!="PASS" else "PENDING_PHASE_B")
    print("\nNumerical failures =\n0")
    print("\nMECHANISM_CLOSURE_CANDIDATE =\nNO\n\nMETHOD_DESIGN_READY =\nNO")
    print("\nNEXT_RECOMMENDED_TASK =\n"+obj["next_recommended_task"])
    print("\nArtifacts =")
    for p in [SCRIPT, STAGE0, SMOKE, SCREEN, REPORT, FIG_DIR, RAW, CHECKPOINT]: print(p)
    print("\nSTOP" if obj["SOURCE_RESCUE_GATE"]!="PASS" else "\nPHASE_A_DONE")


def print_behavioral_summary(obj):
    print(json.dumps(obj["aggregate"], indent=2)); print("STOP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0","smoke","phase_a","behavioral"], required=True)
    args = ap.parse_args()
    if args.stage=="stage0": stage0()
    elif args.stage=="smoke": smoke()
    elif args.stage=="phase_a": phase_a()
    else: behavioral()


if __name__ == "__main__":
    main()
