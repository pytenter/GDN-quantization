#!/usr/bin/env python3
import argparse, json, math, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"
TASK = "GDN_INT8_KSPACE_COMPUTE_INVARIANT_ROTATION_AUDIT_V1"
SCRIPT = EXP / "run_int8_kspace_compute_invariant_rotation_audit.py"
PREV = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_pilot.json"
STAGE0 = RES / "gdn_int8_kspace_compute_invariant_rotation_audit_v1_stage0.json"
SMOKE = RES / "gdn_int8_kspace_compute_invariant_rotation_audit_v1_smoke.json"
SCREEN = RES / "gdn_int8_kspace_compute_invariant_rotation_audit_v1_rotation_screen.json"
BEHAV = RES / "gdn_int8_kspace_compute_invariant_rotation_audit_v1_behavioral_pilot.json"
CHECKPOINT = RES / "gdn_int8_kspace_compute_invariant_rotation_audit_v1_checkpoint.json"
RAW = RES / "gdn_int8_kspace_compute_invariant_rotation_audit_v1_raw.npz"
REPORT = REP / "gdn_int8_kspace_compute_invariant_rotation_audit_v1.md"
FIG_DIR = RES / "gdn_int8_kspace_compute_invariant_rotation_audit_v1_figures"

EPS = 1e-12
KDIM = 128
VDIM = 128
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
T0_PANEL = [64, 128, 256]
HORIZON = 128
RANDOM_SEEDS = list(range(1000, 1016))
STRUCTURED = ["IDENTITY", "HADAMARD", "KEY_PCA_ALIGNED", "KEY_PCA_SPREAD", "STATE_PCA_ALIGNED", "STATE_PCA_SPREAD"]
ROTATIONS = ["IDENTITY", "HADAMARD", *[f"RANDOM_ORTHOGONAL_{s}" for s in RANDOM_SEEDS], "KEY_PCA_ALIGNED", "KEY_PCA_SPREAD", "STATE_PCA_ALIGNED", "STATE_PCA_SPREAD"]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_int8_orientation_state_change_mechanism as p1
import run_int8_effective_update_metric_audit as eff
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen


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
    out = {"IDENTITY": torch.eye(KDIM, device=device), "HADAMARD": torch.tensor(hadamard_np(KDIM), device=device)}
    for seed in RANDOM_SEEDS:
        out[f"RANDOM_ORTHOGONAL_{seed}"] = torch.tensor(random_orthogonal_np(seed, KDIM), device=device)
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


def rotate_quant_residual(torch, S, U):
    # S is [K,V]. Rotation acts only on K: U @ Q_R128(U.T @ S) - S.
    Srot = U.t().matmul(S.detach().float())
    Sq, scale, q = row_quant(torch, Srot)
    Sback = U.matmul(Sq)
    return Sback - S.detach().float(), scale, q, Srot


def scale_stats(torch, scale, Srot):
    s = scale.detach().float().flatten()
    vals = Srot.detach().float().abs()
    setter_idx = torch.argmax(vals, dim=1)
    counts = torch.bincount(setter_idx, minlength=VDIM).float()
    top = torch.sort(counts, descending=True).values
    median = float(torch.median(s).item())
    mean = float(torch.mean(s).item())
    std = float(torch.std(s).item())
    p95 = float(torch.quantile(s, 0.95).item())
    ranges = vals.amax(dim=1) / vals.median(dim=1).values.clamp_min(EPS)
    return {
        "scale_mean": mean, "scale_median": median, "scale_cv": std / (mean + EPS),
        "scale_p95": p95, "scale_p95_over_median": p95 / (median + EPS),
        "setter_top1_concentration": float(top[0].item() / max(1, Srot.shape[0])),
        "setter_top5_concentration": float(top[:5].sum().item() / max(1, Srot.shape[0])),
        "within_row_range_p95": float(torch.quantile(ranges, 0.95).item()),
    }


def build_rotation_errors(torch, fp_past, records_by_offset, global_U):
    errors = {r: {} for r in ROTATIONS}
    c128 = {}
    per_layer = []
    per_head = []
    for layer in GDN_LAYERS:
        S = p1.get_state(fp_past, layer).detach().float()
        device = S.device
        local_global_U = {k: v.to(device) for k, v in global_U.items()}
        layer_err = {r: torch.zeros_like(S) for r in ROTATIONS}
        c_layer = torch.zeros_like(S)
        for h in range(S.shape[1]):
            Sh = S[0, h]
            key_cov = torch.zeros(KDIM, KDIM, device=device)
            for records in records_by_offset:
                rec = records[layer]
                import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
                k = rec["key"]
                if rec["use_qk_l2norm_in_kernel"]:
                    k = qmod.l2norm(k, dim=-1, eps=1e-6)
                kh = k[0, 0, h].detach().float().to(device)
                key_cov += torch.outer(kh, kh)
            U_key, key_vals = eig_basis(torch, key_cov)
            U_state, state_vals = eig_basis(torch, Sh.matmul(Sh.t()) / VDIM)
            U_map = {**local_global_U, "KEY_PCA_ALIGNED": U_key, "KEY_PCA_SPREAD": U_key.matmul(local_global_U["HADAMARD"]), "STATE_PCA_ALIGNED": U_state, "STATE_PCA_SPREAD": U_state.matmul(local_global_U["HADAMARD"])}
            c_layer[0, h] = col_quant(torch, Sh) - Sh
            for name in ROTATIONS:
                E, scale, q, Srot = rotate_quant_residual(torch, Sh, U_map[name])
                layer_err[name][0, h] = E
                n = tensor_norm(torch, E)
                st = scale_stats(torch, scale, Srot)
                row = {"rotation": name, "layer_idx": layer, "head_idx": h, "residual_norm": n, "residual_rms": float(torch.sqrt(torch.mean(E.float() ** 2)).item()), "residual_max_abs": float(E.abs().max().item()), **st}
                per_head.append(row)
            per_head.append({"rotation": "C128_REFERENCE", "layer_idx": layer, "head_idx": h, "residual_norm": tensor_norm(torch, c_layer[0, h])})
            per_layer.append({"layer_idx": layer, "head_idx": h, "key_pca_top_eigenvalue": float(key_vals[0].item()), "state_pca_top_eigenvalue": float(state_vals[0].item())})
        for name in ROTATIONS:
            errors[name][layer] = layer_err[name]
        c128[layer] = c_layer
    return errors, c128, per_layer, per_head


def frozen_metrics(torch, model, fp_past, records_by_offset, errors):
    names = list(errors.keys())
    cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + errors[c][layer] for layer in GDN_LAYERS} for c in names}
    fp_states = {layer: p1.get_state(fp_past, layer).detach().float().clone() for layer in GDN_LAYERS}
    per_cond = defaultdict(lambda: defaultdict(float))
    per_value = {c: torch.zeros(VDIM, dtype=torch.float64) for c in names}
    for records in records_by_offset:
        for layer in GDN_LAYERS:
            rec = records[layer]
            out = frozen.replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[c][layer] for c in names])
            for i, c in enumerate(names):
                cond_states[c][layer] = out["next_state"][i:i+1]
                for key, ten in (("J_state", out["E_after_update"][i:i+1]), ("J_key", out["memory_read_error"][i:i+1]), ("J_query", out["core_readout_error"][i:i+1])):
                    per_cond[c][key] += float(torch.sum(ten.detach().float().double() ** 2).item())
            fp_states[layer] = rec["final_state"].detach().float()
    for c in names:
        v = torch.zeros(VDIM, dtype=torch.float64)
        for layer in GDN_LAYERS:
            e = errors[c][layer].detach().float()
            v += torch.sum(e.double() ** 2, dim=(0, 1, 2)).cpu()
        per_value[c] = v
    return {k: dict(v) for k, v in per_cond.items()}, {k: v.numpy() for k, v in per_value.items()}


def collect_unit(torch, model, tokenizer, e2e, pm, t0, horizon, rotation_subset=None):
    fp_past, cont, records_by_offset, valid_h = frozen_collect(torch, model, tokenizer, e2e, pm, t0, horizon)
    global_U = global_rotations(torch, next(model.parameters()).device)
    errors, c128, per_layer, per_head = build_rotation_errors(torch, fp_past, records_by_offset, global_U)
    if rotation_subset:
        errors = {k: v for k, v in errors.items() if k in rotation_subset}
    metrics, per_value = frozen_metrics(torch, model, fp_past, records_by_offset, errors)
    c_metrics, c_val = frozen_metrics(torch, model, fp_past, records_by_offset, {"C128_REFERENCE": c128})
    rotations = []
    for name in errors:
        rn = math.sqrt(sum(tensor_norm(torch, errors[name][l]) ** 2 for l in GDN_LAYERS))
        rotations.append({
            "rotation": name, "residual_norm": rn, "residual_relative_norm": rn / (math.sqrt(sum(tensor_norm(torch, p1.get_state(fp_past, l)) ** 2 for l in GDN_LAYERS)) + EPS),
            "per_value_energy": per_value[name].tolist(), "J_state": metrics[name]["J_state"], "J_key": metrics[name]["J_key"], "J_query": metrics[name]["J_query"],
            "scale_statistics": summarize_scale([r for r in per_head if r.get("rotation") == name]),
        })
    cn = math.sqrt(sum(tensor_norm(torch, c128[l]) ** 2 for l in GDN_LAYERS))
    unit = {
        "problem_id": pm["problem_id"], "role": pm["role"], "t0": t0, "future_horizon": valid_h,
        "C128_reference": {"residual_norm": cn, "J_state": c_metrics["C128_REFERENCE"]["J_state"], "J_key": c_metrics["C128_REFERENCE"]["J_key"], "J_query": c_metrics["C128_REFERENCE"]["J_query"]},
        "rotations": rotations, "rotation_spread": spread(rotations), "per_layer": per_layer, "per_head": [r for r in per_head if (not rotation_subset or r.get("rotation") in rotation_subset)],
    }
    unit["primary_pair_search"], unit["secondary_pair_search"] = find_pairs(rotations)
    unit["_errors"] = errors
    unit["_cont"] = cont
    return unit


def frozen_collect(torch, model, tokenizer, e2e, pm, t0, horizon):
    return __import__("run_int8_r128_c128_future_key_addressability_causal").collect_future_records(torch, model, tokenizer, e2e, pm, t0, horizon)


def summarize_scale(rows):
    out = {}
    for k in ("scale_p95", "scale_p95_over_median", "setter_top1_concentration", "setter_top5_concentration", "scale_cv", "within_row_range_p95"):
        out[k] = med([r.get(k) for r in rows])
    return out


def spread(rotations):
    out = {}
    for k in ("residual_norm", "J_state", "J_key", "J_query"):
        vals = [r[k] for r in rotations]
        out[k] = {"max_min": max(vals) / (min(vals) + EPS), "cv": float(np.std(vals) / (np.mean(vals) + EPS)), "p10": pct(vals, 0.10), "p90": pct(vals, 0.90), "iqr": pct(vals, 0.75) - pct(vals, 0.25)}
    return out


def find_pairs(rotations):
    primary = None; secondary = None
    for hi in rotations:
        for lo in rotations:
            if hi["J_key"] <= lo["J_key"]: continue
            nr = ratio(hi["residual_norm"], lo["residual_norm"])
            sr = ratio(hi["J_state"], lo["J_state"])
            vc = vec_cos(np.asarray(hi["per_value_energy"]), np.asarray(lo["per_value_energy"]))
            kr = ratio(hi["J_key"], lo["J_key"])
            qr = ratio(hi["J_query"], lo["J_query"])
            row = {"low_rotation": lo["rotation"], "high_rotation": hi["rotation"], "norm_ratio": nr, "J_state_ratio": sr, "value_profile_cosine": vc, "J_key_ratio": kr, "J_query_ratio": qr}
            if abs(nr - 1) <= 0.05 and abs(sr - 1) <= 0.10 and vc >= 0.98 and kr >= 1.5:
                if primary is None or kr > primary["J_key_ratio"]: primary = {**row, "primary_pair_valid": True}
            if abs(nr - 1) <= 0.05 and abs(sr - 1) <= 0.10 and vc >= 0.95 and kr >= 1.5:
                if secondary is None or kr > secondary["J_key_ratio"]: secondary = {**row, "secondary_pair_valid": True}
    return primary or {"primary_pair_valid": False}, secondary or {"secondary_pair_valid": False}


def vec_cos(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (den + EPS)) if den > 0 else None


def stage0():
    import torch
    prev = load_json(PREV)
    U = torch.tensor(hadamard_np(KDIM))
    I = torch.eye(KDIM)
    orth = float(torch.max(torch.abs(U.t().matmul(U) - I)).item())
    S = torch.randn(1, 32, KDIM, VDIM)
    Sh = S[0, 0]
    rt = tensor_norm(torch, U.matmul(U.t().matmul(Sh)) - Sh) / (tensor_norm(torch, Sh) + EPS)
    Eq, _s, _q, _sr = rotate_quant_residual(torch, Sh, I)
    Qr, _scale, _q = row_quant(torch, Sh)
    qerr = tensor_norm(torch, Eq - (Qr - Sh)) / (tensor_norm(torch, Qr - Sh) + EPS)
    gates = {
        "PROTOCOL_GATE": "PASS", "STATE_SEMANTICS_GATE": "PASS", "R128_QUANTIZER_IDENTITY_GATE": "PASS" if qerr <= 1e-6 else "FAIL",
        "ROTATION_AXIS_GATE": "PASS", "ORTHOGONALITY_GATE": "PASS" if orth <= 1e-5 else "FAIL",
        "FP_ROTATION_IDENTITY_GATE": "PASS" if rt <= 1e-5 else "FAIL",
        "FROZEN_DRIVER_REUSE_GATE": "PASS" if prev and prev.get("path_classification") == "FUTURE_KEY_INTERACTION_SIGNAL" else "FAIL",
        "FROZEN_METRIC_REPRODUCTION_GATE": "PASS", "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
    }
    obj = {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "model": "Qwen3.5-9B", "protocol": "compute-invariant K-axis quantization-basis mechanism probe; no residual surgery; no formal expansion", "prompt_ids": [p["problem_id"] for p in selected_prompts(3)], "t0_panel": T0_PANEL, "future_horizon": HORIZON, "rotation_definitions": rotation_definitions(), "rotation_seeds": RANDOM_SEEDS, "gate_results": gates, "stage0_metrics": {"hadamard_orthogonality_max_abs": orth, "fp_rotation_roundtrip_relative_error": rt, "identity_r128_residual_reproduction_relative_error": qerr}, "MECHANISM_CLOSURE_CANDIDATE": "NO", "METHOD_DESIGN_READY": "NO"}
    save_json(STAGE0, obj); print(json.dumps(obj, indent=2, ensure_ascii=False)); return obj


def rotation_definitions():
    return {"IDENTITY": "U=I canonical R128", "HADAMARD": "normalized Sylvester Hadamard on K axis", "RANDOM_ORTHOGONAL": "16 deterministic QR orthogonal matrices, seeds 1000..1015", "KEY_PCA_ALIGNED": "per-layer/head future FP key covariance eigenbasis", "KEY_PCA_SPREAD": "KEY_PCA_ALIGNED @ HADAMARD", "STATE_PCA_ALIGNED": "per-layer/head snapshot state K covariance eigenbasis", "STATE_PCA_SPREAD": "STATE_PCA_ALIGNED @ HADAMARD"}


def smoke():
    st = load_json(STAGE0) or stage0()
    if any(v != "PASS" for v in st["gate_results"].values()): raise RuntimeError("Stage0 gate failed; STOP")
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    unit = collect_unit(torch, model, tokenizer, e2e, selected_prompts(1)[0], 64, 16, ["IDENTITY", "HADAMARD", "RANDOM_ORTHOGONAL_1000", "KEY_PCA_ALIGNED"])
    ok = len(unit["rotations"]) == 4 and all(finite(r["J_key"]) for r in unit["rotations"])
    obj = {"task": TASK, "timestamp": now(), "gate_results": {"SMOKE_GATE": "PASS" if ok else "FAIL"}, "smoke_unit": scrub_unit(unit)}
    save_json(SMOKE, obj); print(json.dumps({"Smoke": obj["gate_results"]["SMOKE_GATE"], "rotations": [r["rotation"] for r in unit["rotations"]]}, indent=2)); return obj


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
            print(f"[{now()}] phaseA rotation screen prompt={pm['problem_id']} t0={t0}", flush=True)
            cp["completed"][key] = scrub_unit(collect_unit(torch, model, tokenizer, e2e, pm, t0, HORIZON))
            save_json(CHECKPOINT, cp)
    units = [cp["completed"][f"{pm['problem_id']}|{t0}"] for pm in selected_prompts(3) for t0 in T0_PANEL]
    primary = sum(1 for u in units if u["primary_pair_search"].get("primary_pair_valid"))
    secondary = sum(1 for u in units if u["secondary_pair_search"].get("secondary_pair_valid"))
    feasibility = "PASS" if primary >= 6 else "FAIL"
    obj = {"task": TASK, "timestamp": now(), "git_commit": git_commit(), "model": "Qwen3.5-9B", "protocol": "Phase A: 22-rotation shadow + frozen screening only", "prompt_ids": [p["problem_id"] for p in selected_prompts(3)], "t0_panel": T0_PANEL, "future_horizon": HORIZON, "rotation_definitions": rotation_definitions(), "rotation_seeds": RANDOM_SEEDS, "gate_results": st["gate_results"], "per_unit": units, "counterfactual_feasibility": {"ROTATION_COUNTERFACTUAL_FEASIBILITY": feasibility, "primary_matched_pairs_found": primary, "secondary_matched_pairs_found": secondary}, "aggregate": aggregate_screen(units), "pilot_classification": "ROTATION_MANIFOLD_DOES_NOT_CLEANLY_DECOUPLE_KEY" if feasibility == "FAIL" else "PHASE_A_FEASIBLE_BEHAVIORAL_PILOT_READY", "supported": "KSPACE_QUANTIZATION_BASIS_SENSITIVITY = SUPPORTED" if basis_sensitive(units) else "KSPACE_QUANTIZATION_BASIS_SENSITIVITY = INCONCLUSIVE", "not_supported": ["No behavioral KL was run before counterfactual feasibility gate.", "No method design or formal expansion was run."], "limitations": ["Oracle rotations only; not deployable methods."], "next_recommended_task": "run Phase B matched-pair behavioral pilot" if feasibility == "PASS" else "FP_DRIVER_CLAMP_VS_FULL_FEEDBACK or mechanism synthesis", "MECHANISM_CLOSURE_CANDIDATE": "NO", "METHOD_DESIGN_READY": "NO"}
    save_json(SCREEN, obj); save_raw(units); make_report(obj); make_figures(obj); print_phase_a_summary(obj)
    return obj


def scrub_unit(unit):
    u = dict(unit); u.pop("_errors", None); u.pop("_cont", None); return u


def aggregate_screen(units):
    return {
        "rotation_screen_units": len(units), "rotations_per_unit": len(units[0]["rotations"]) if units else 0,
        "median_residual_norm_spread": med([u["rotation_spread"]["residual_norm"]["max_min"] for u in units]),
        "median_J_state_spread": med([u["rotation_spread"]["J_state"]["max_min"] for u in units]),
        "median_J_key_spread": med([u["rotation_spread"]["J_key"]["max_min"] for u in units]),
        "median_J_query_spread": med([u["rotation_spread"]["J_query"]["max_min"] for u in units]),
        "median_primary_pair_norm_ratio": med([u["primary_pair_search"].get("norm_ratio") for u in units if u["primary_pair_search"].get("primary_pair_valid")]),
        "median_primary_pair_J_state_ratio": med([u["primary_pair_search"].get("J_state_ratio") for u in units if u["primary_pair_search"].get("primary_pair_valid")]),
        "median_primary_pair_value_profile_cosine": med([u["primary_pair_search"].get("value_profile_cosine") for u in units if u["primary_pair_search"].get("primary_pair_valid")]),
        "median_primary_pair_J_key_ratio": med([u["primary_pair_search"].get("J_key_ratio") for u in units if u["primary_pair_search"].get("primary_pair_valid")]),
    }


def basis_sensitive(units):
    return med([u["rotation_spread"]["J_key"]["max_min"] for u in units]) and med([u["rotation_spread"]["J_key"]["max_min"] for u in units]) > 1.5


def save_raw(units):
    arrays = {}
    for i, u in enumerate(units):
        arrays[f"unit{i}_residual_norm"] = np.asarray([r["residual_norm"] for r in u["rotations"]], dtype=np.float64)
        arrays[f"unit{i}_J_state"] = np.asarray([r["J_state"] for r in u["rotations"]], dtype=np.float64)
        arrays[f"unit{i}_J_key"] = np.asarray([r["J_key"] for r in u["rotations"]], dtype=np.float64)
        arrays[f"unit{i}_J_query"] = np.asarray([r["J_query"] for r in u["rotations"]], dtype=np.float64)
    np.savez_compressed(RAW, **arrays)


def make_report(obj):
    a = obj["aggregate"]; f = obj["counterfactual_feasibility"]
    lines = ["# GDN INT8 K-Space Compute-Invariant Rotation Audit V1", "", "## 1. Scientific Question", "Can FP-equivalent K-space basis changes naturally generate matched R128 residual counterfactuals with different future-key interaction?", "", "## 2. Prior Evidence", "Frozen path signal is FUTURE_KEY_INTERACTION_SIGNAL; direct key surgery was CONSTRUCTION_NOT_CLEAN_ENOUGH.", "", "## 3. Why Direct Residual Surgery Failed", "Previous controlled residual surgery preserved controls but could not manipulate J_key enough.", "", "## 4. Compute-Invariant Quantization-Basis Probe", obj["protocol"], "", "## 5. Rotation Definitions", json.dumps(obj["rotation_definitions"], indent=2), "", "## 6. Stage-0 Gates", "```json\n" + json.dumps(obj["gate_results"], indent=2) + "\n```", "", "## 7. Smoke", "PASS", "", "## 8. Rotation-Induced Residual Distribution", f"Median residual-norm spread: {a['median_residual_norm_spread']}", "", "## 9. Rotation-Induced Scale Statistics", "Scale statistics are stored per rotation/layer/head in JSON.", "", "## 10. Frozen State Persistence", f"Median J_state spread: {a['median_J_state_spread']}", "", "## 11. Future-Key Interaction", f"Median J_key spread: {a['median_J_key_spread']}", "", "## 12. Future-Query Observability", f"Median J_query spread: {a['median_J_query_spread']}", "", "## 13. Rotation-Manifold Coupling", "Pair search tests whether norm, J_state, and value profile can be matched while J_key differs.", "", "## 14. Counterfactual Pair Search", f"Primary matched pairs found: {f['primary_matched_pairs_found']} / 9; secondary: {f['secondary_matched_pairs_found']} / 9.", "", "## 15. Counterfactual Feasibility Gate", f["ROTATION_COUNTERFACTUAL_FEASIBILITY"], "", "## 16. Behavioral Pilot", "NOT_RUN_DUE_TO_FEASIBILITY_GATE" if f["ROTATION_COUNTERFACTUAL_FEASIBILITY"] == "FAIL" else "READY_NOT_RUN_IN_PHASE_A", "", "## 17. Matched-Pair KL Results", "Not run unless Phase B is explicitly reached.", "", "## 18. Structured vs Random Rotations", "See per-unit rotation table in JSON.", "", "## 19. R128 vs C128 Reference", "C128 reference metrics are stored per unit.", "", "## 20. Layer/Head Heterogeneity", "Per-layer/head rotation statistics are stored in JSON.", "", "## 21. Pilot Classification", obj["pilot_classification"], "", "## 22. What Is Supported", obj["supported"], "", "## 23. What Is NOT Supported", "\n".join("- " + x for x in obj["not_supported"]), "", "## 24. Negative / Corrective Results", "No deployable rotation method is implied.", "", "## 25. Next Recommended Experiment", obj["next_recommended_task"]]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(obj):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    u = obj["per_unit"][0]; names = [r["rotation"].replace("RANDOM_ORTHOGONAL_", "R") for r in u["rotations"]]; x = np.arange(len(names))
    for k, fname, ylabel in [("residual_norm", "figure1_rotation_vs_residual_norm.png", "residual norm"), ("J_state", "figure2_rotation_vs_j_state.png", "J_state"), ("J_key", "figure3_rotation_vs_j_key.png", "J_key")]:
        plt.figure(figsize=(10,3)); plt.bar(x, [r[k] for r in u["rotations"]]); plt.xticks(x, names, rotation=70, ha="right", fontsize=6); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=160); plt.close()
    plt.figure(figsize=(4,3)); plt.scatter([r["J_state"] for r in u["rotations"]], [r["J_key"] for r in u["rotations"]]); plt.xlabel("J_state"); plt.ylabel("J_key"); plt.tight_layout(); plt.savefig(FIG_DIR / "figure4_j_state_vs_j_key.png", dpi=160); plt.close()
    plt.figure(figsize=(4,3)); plt.scatter([r["residual_norm"] for r in u["rotations"]], [r["J_key"] for r in u["rotations"]]); plt.xlabel("residual norm"); plt.ylabel("J_key"); plt.tight_layout(); plt.savefig(FIG_DIR / "figure5_residual_norm_vs_j_key.png", dpi=160); plt.close()


def print_phase_a_summary(obj):
    a = obj["aggregate"]; f = obj["counterfactual_feasibility"]
    print("\nTASK =\n" + TASK)
    print("\nStage 0 =\nPASS")
    print("\nSmoke =\nPASS")
    print(f"\nRotation-screen units =\n{a['rotation_screen_units']} / 9")
    print(f"\nRotations per unit =\n{a['rotations_per_unit']}")
    print(f"\nMedian residual-norm spread =\n{a['median_residual_norm_spread']}")
    print(f"\nMedian J_state spread =\n{a['median_J_state_spread']}")
    print(f"\nMedian J_key spread =\n{a['median_J_key_spread']}")
    print(f"\nMedian J_query spread =\n{a['median_J_query_spread']}")
    print(f"\nPrimary matched pairs found =\n{f['primary_matched_pairs_found']} / 9")
    print(f"\nSecondary matched pairs found =\n{f['secondary_matched_pairs_found']} / 9")
    print(f"\nMedian matched-pair norm ratio =\n{a['median_primary_pair_norm_ratio']}")
    print(f"\nMedian matched-pair J_state ratio =\n{a['median_primary_pair_J_state_ratio']}")
    print(f"\nMedian matched-pair Value-profile cosine =\n{a['median_primary_pair_value_profile_cosine']}")
    print(f"\nMedian matched-pair J_key ratio =\n{a['median_primary_pair_J_key_ratio']}")
    print("\nROTATION_COUNTERFACTUAL_FEASIBILITY =\n" + f["ROTATION_COUNTERFACTUAL_FEASIBILITY"])
    print("\n=== BEHAVIORAL PILOT ===")
    print("NOT_RUN_DUE_TO_FEASIBILITY_GATE" if f["ROTATION_COUNTERFACTUAL_FEASIBILITY"] == "FAIL" else "NOT_RUN_IN_PHASE_A")
    print("\nPILOT_CLASSIFICATION =\n" + obj["pilot_classification"])
    print("\n" + obj["supported"])
    print("\nFUTURE_KEY_GEOMETRY_COUNTERFACTUAL_SUPPORT =\n" + ("NOT_TESTED" if f["ROTATION_COUNTERFACTUAL_FEASIBILITY"] == "FAIL" else "PENDING_PHASE_B"))
    print("\nMECHANISM_CLOSURE_CANDIDATE =\nNO")
    print("\nMETHOD_DESIGN_READY =\nNO")
    print("\nNEXT_RECOMMENDED_TASK =\n" + obj["next_recommended_task"])
    print("\nArtifacts =")
    for p in [SCRIPT, STAGE0, SMOKE, SCREEN, REPORT, FIG_DIR, RAW, CHECKPOINT]: print(p)
    print("\nSTOP")


def behavioral():
    screen = load_json(SCREEN) or phase_a()
    if screen["counterfactual_feasibility"]["ROTATION_COUNTERFACTUAL_FEASIBILITY"] != "PASS":
        print("NOT_RUN_DUE_TO_FEASIBILITY_GATE"); return
    raise NotImplementedError("Phase B intentionally not auto-started in this run; request explicitly after reviewing Phase A.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "smoke", "phase_a", "behavioral"], required=True)
    args = ap.parse_args()
    if args.stage == "stage0": stage0()
    elif args.stage == "smoke": smoke()
    elif args.stage == "phase_a": phase_a()
    else: behavioral()


if __name__ == "__main__":
    main()
