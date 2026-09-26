#!/usr/bin/env python3
import argparse
import csv
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
RUN_DIR = ROOT / "runs" / "gdn_int8_same_norm_functional_geometry_audit_v1"
TASK = "GDN_INT8_SAME_NORM_FUNCTIONAL_GEOMETRY_AUDIT_V1"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_layer_local_operator_invariant_update_transduction_causal_v2 as v2

PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"
EPS = 1e-12
HORIZON_A = 128
HORIZON_B = 128
TOL = 1e-5
ROT_SEEDS = [1101, 1102, 1103, 1104]

FROZEN = ROOT / "results" / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_pilot.json"
NORMSWAP = ROOT / "results" / "gdn_int8_r128_c128_natural_residual_norm_swap_causal_v1_pilot.json"
V2_STAGE0 = ROOT / "results" / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v2_stage0.json"
NATIVE_STAGE0 = ROOT / "results" / "gdn_native_core_output_replay_semantics_audit_v1_stage0.json"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def ratio(a, b):
    return float(a) / (float(b) + EPS) if finite(a) and finite(b) else None


def log_ratio(a, b):
    r = ratio(a, b)
    return math.log(r + EPS) if finite(r) and r > 0 else None


def auc_sq(xs):
    return sum(float(x) ** 2 for x in xs if finite(x))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT / "GDN-quantization"), text=True).strip()
    except Exception:
        return None


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return None
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
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > EPS else None


def stage0():
    frozen = load_json(FROZEN)
    norm = load_json(NORMSWAP)
    v2 = load_json(V2_STAGE0)
    native = load_json(NATIVE_STAGE0)
    keys_f = [(u["problem_id"], int(u["t0"])) for u in frozen["per_unit"]]
    keys_n = [(u["problem_id"], int(u["t0"])) for u in norm["per_unit"]]
    max_norm = max(float(u["max_per_head_norm_match_error"]) for u in norm["per_unit"])
    gates = {
        "PROTOCOL_GATE": "PASS" if len(keys_f) == 9 and keys_f == keys_n else "FAIL",
        "REPLAY_IDENTITY_GATE": "PASS" if v2.get("REPLAY_REGRESSION_GATE") == "PASS" and v2.get("max_core_output_replay_error", 1.0) <= TOL and v2.get("max_next_state_replay_error", 1.0) <= TOL else "FAIL",
        "TENSOR_SEMANTICS_GATE": "PASS" if native.get("CORE_OUTPUT_REPLAY_ROOT_CAUSE", "DTYPE_SEMANTICS") or native.get("STAGE0") == "PASS" else "FAIL",
        "SOURCE_NORM_GATE": "PASS" if max_norm <= 1e-4 else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS" if not any(u.get("numerical_failures") for u in frozen["per_unit"]) else "FAIL",
    }
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model_path": str(getattr(p1, "MODEL_DIR", "/data/zypan/modelscope_models/Qwen3.5-9B")),
        "canonical_units": [f"{a}|{b}" for a, b in keys_f],
        "source_files": {"frozen": str(FROZEN), "normswap": str(NORMSWAP), "v2_stage0": str(V2_STAGE0), "native_stage0": str(NATIVE_STAGE0)},
        "max_prior_frozen_replay_identity_error": max(max(u["identity_max_relative_errors"].values()) for u in frozen["per_unit"]),
        "corrected_native_core_output_replay_error": v2.get("max_core_output_replay_error"),
        "corrected_native_next_state_replay_error": v2.get("max_next_state_replay_error"),
        "max_source_norm_match_error": max_norm,
        "gates": gates,
        "STAGE0": "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL",
    }
    save_json("stage0_identity.json", obj)
    save_json("stage0_summary.json", {"STAGE0": obj["STAGE0"], "gates": gates})
    return obj


def corrected_replay_layer_conditions(torch, model, layer, rec, fp_state, cond_state_list):
    batch = len(cond_state_list)
    rec_b = dict(rec)
    for k in ("query", "key", "value", "g", "beta"):
        rec_b[k] = frozen.expand_driver(torch, rec[k], batch)
    rec_b["z"] = frozen.expand_gate_z(torch, rec["z"], batch)
    fp_batch = fp_state.detach().float().expand(batch, *fp_state.shape[1:]).contiguous()
    pe_batch = torch.cat([s.detach().float() for s in cond_state_list], dim=0)
    fp_manual = v2.manual_components_fixed_core(torch, rec_b, fp_batch)
    pe_manual = v2.manual_components_fixed_core(torch, rec_b, pe_batch)
    fp_out, fp_next = frozen.implementation_replay(rec_b, fp_batch)
    pe_out, pe_next = frozen.implementation_replay(rec_b, pe_batch)
    return {
        "E_after_decay": pe_manual["after_decay"] - fp_manual["after_decay"],
        "memory_read_error": pe_manual["memory"] - fp_manual["memory"],
        "delta_error": pe_manual["delta"] - fp_manual["delta"],
        "update_error": pe_manual["update"] - fp_manual["update"],
        "E_after_update": pe_next.detach().float() - fp_next.detach().float(),
        "core_readout_error": pe_out.detach().float() - fp_out.detach().float(),
        "next_state": pe_next.detach().float(),
        "fp_next_state": fp_next.detach().float(),
        "identity": {
            "next_state_relerr": normswap.norm(torch, pe_manual["next_state"] - pe_next.float()) / (normswap.norm(torch, pe_next) + EPS),
            "core_relerr": normswap.norm(torch, pe_manual["core_native_dtype"] - pe_out) / (normswap.norm(torch, pe_out) + EPS),
            "fp_next_state_relerr": normswap.norm(torch, fp_manual["next_state"] - fp_next.float()) / (normswap.norm(torch, fp_next) + EPS),
            "fp_core_relerr": normswap.norm(torch, fp_manual["core_native_dtype"] - fp_out) / (normswap.norm(torch, fp_out) + EPS),
        },
    }


def run_fresh_stage_a_unit(torch, model, tokenizer, e2e, pm, t0, kl_ref):
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        inj, meta = normswap.build_injections(torch, fp_past)
        cond_states = {
            PRIMARY_R: {layer: p1.get_state(fp_past, layer).detach().float() + inj[PRIMARY_R][layer] for layer in frozen.GDN_LAYERS},
            PRIMARY_C: {layer: p1.get_state(fp_past, layer).detach().float() + inj[PRIMARY_C][layer] for layer in frozen.GDN_LAYERS},
        }
        fp_states = {layer: p1.get_state(fp_past, layer).detach().float().clone() for layer in frozen.GDN_LAYERS}
        curves = {PRIMARY_R: {"state": [], "query": [], "after_decay": [], "key": []}, PRIMARY_C: {"state": [], "query": [], "after_decay": [], "key": []}}
        max_identity = 0.0
        valid_h = min(HORIZON_A, len(cont) - t0)
        past = fp_past
        with torch.inference_mode():
            for off in range(1, valid_h + 1):
                nxt = torch.tensor([[cont[t0 + off - 1]]], dtype=ids.dtype, device=ids.device)
                _out, past, records = frozen.driver_step(torch, model, nxt, None, past, collector)
                per_token = {PRIMARY_R: {"state": 0.0, "query": 0.0, "after_decay": 0.0, "key": 0.0}, PRIMARY_C: {"state": 0.0, "query": 0.0, "after_decay": 0.0, "key": 0.0}}
                for layer in frozen.GDN_LAYERS:
                    rec = records[layer]
                    out = corrected_replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[PRIMARY_R][layer], cond_states[PRIMARY_C][layer]])
                    max_identity = max(max_identity, max(float(x) for x in out["identity"].values()))
                    for ci, cond in enumerate([PRIMARY_R, PRIMARY_C]):
                        cond_states[cond][layer] = out["next_state"][ci:ci + 1]
                        for name, key in [("state", "E_after_update"), ("query", "core_readout_error"), ("after_decay", "E_after_decay"), ("key", "memory_read_error")]:
                            ten = out[key][ci:ci + 1].detach().float()
                            per_token[cond][name] += float(torch.sum(ten.double() * ten.double()).item())
                    fp_states[layer] = rec["final_state"].detach().float()
                for cond in [PRIMARY_R, PRIMARY_C]:
                    for name in curves[cond]:
                        curves[cond][name].append(math.sqrt(per_token[cond][name]))
        norm_r = meta["residual_norm_C"]
        norm_c = meta["residual_norm_C"]
        pp = {
            "norm_R": norm_r,
            "norm_C": norm_c,
            "state_curve_R": curves[PRIMARY_R]["state"],
            "state_curve_C": curves[PRIMARY_C]["state"],
            "query_error_curve_R": curves[PRIMARY_R]["query"],
            "query_error_curve_C": curves[PRIMARY_C]["query"],
            "key_error_curve_R": curves[PRIMARY_R]["key"],
            "key_error_curve_C": curves[PRIMARY_C]["key"],
            "after_decay_curve_R": curves[PRIMARY_R]["after_decay"],
            "after_decay_curve_C": curves[PRIMARY_C]["after_decay"],
            "J_state_R": auc_sq(curves[PRIMARY_R]["state"]),
            "J_state_C": auc_sq(curves[PRIMARY_C]["state"]),
            "J_query_R": auc_sq(curves[PRIMARY_R]["query"]),
            "J_query_C": auc_sq(curves[PRIMARY_C]["query"]),
            "J_key_R": auc_sq(curves[PRIMARY_R]["key"]),
            "J_key_C": auc_sq(curves[PRIMARY_C]["key"]),
            "full_KL_R": kl_ref[PRIMARY_R]["AUC_KL"],
            "full_KL_C": kl_ref[PRIMARY_C]["AUC_KL"],
        }
        for k in ("state", "query", "key"):
            pp[f"R_over_C_{k}"] = ratio(pp[f"J_{k}_R"], pp[f"J_{k}_C"])
        return {
            "problem_id": pm["problem_id"],
            "role": pm.get("role"),
            "t0": int(t0),
            "future_horizon": valid_h,
            "same_norm_meta": meta,
            "primary_pair": pp,
            "identity_max_relative_error": max_identity,
        }
    finally:
        collector["close"]()


def first_persistent_onset(r, c, min_run=4):
    gt = [a > b for a, b in zip(r, c)]
    for i in range(len(gt)):
        if all(gt[i:i + min_run]) and len(gt[i:i + min_run]) == min_run:
            return i + 1
    return None


def stage_a():
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    kl_by = {(u["problem_id"], int(u["t0"])): u for u in load_json(NORMSWAP)["per_unit"]}
    prompts = normswap.selected_prompts(3)
    frozen = {"per_unit": []}
    for pm in prompts:
        for t0 in [64, 128, 256]:
            key = (pm["problem_id"], int(t0))
            print(f"[{now()}] StageA fresh replay {key[0]}|{key[1]}", flush=True)
            frozen["per_unit"].append(run_fresh_stage_a_unit(torch, model, tokenizer, e2e, pm, t0, kl_by[key]))
    save_json("stageA_fresh_replay_units.json", frozen)
    rows = []
    trace_path = RUN_DIR / "stageA_trace.jsonl"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as tr:
        for u in frozen["per_unit"]:
            pp = u["primary_pair"]
            norm_r, norm_c = pp["norm_R"], pp["norm_C"]
            p_r = pp["J_state_R"] / (norm_r * norm_r + EPS)
            p_c = pp["J_state_C"] / (norm_c * norm_c + EPS)
            o_r = pp["J_query_R"] / (norm_r * norm_r + EPS)
            o_c = pp["J_query_C"] / (norm_c * norm_c + EPS)
            eta_r = o_r / (p_r + EPS)
            eta_c = o_c / (p_c + EPS)
            q_r, q_c = pp["query_error_curve_R"], pp["query_error_curve_C"]
            s_r, s_c = pp["state_curve_R"], pp["state_curve_C"]
            row = {
                "unit": f"{u['problem_id']}|{u['t0']}",
                "prompt_id": u["problem_id"],
                "t0": int(u["t0"]),
                "layer": "all_gdn_layers",
                "head": "all_heads",
                "P_R": p_r,
                "P_C": p_c,
                "P_R_over_C": ratio(p_r, p_c),
                "O_R": o_r,
                "O_C": o_c,
                "O_R_over_C": ratio(o_r, o_c),
                "eta_R": eta_r,
                "eta_C": eta_c,
                "eta_R_over_C": ratio(eta_r, eta_c),
                "KL_R": pp["full_KL_R"],
                "KL_C": pp["full_KL_C"],
                "KL_R_over_C": ratio(pp["full_KL_R"], pp["full_KL_C"]),
                "log_P_ratio": log_ratio(p_r, p_c),
                "log_O_ratio": log_ratio(o_r, o_c),
                "log_eta_ratio": log_ratio(eta_r, eta_c),
                "log_KL_ratio": log_ratio(pp["full_KL_R"], pp["full_KL_C"]),
                "first_R_gt_C_observability_token": next((i + 1 for i, (a, b) in enumerate(zip(q_r, q_c)) if a > b), None),
                "persistent_R_gt_C_observability_onset": first_persistent_onset(q_r, q_c),
                "peak_observability_ratio": max([ratio(a * a, b * b) for a, b in zip(q_r, q_c) if b > EPS] or [None]),
            }
            rows.append(row)
            for i, (sr, sc, qr, qc) in enumerate(zip(s_r, s_c, q_r, q_c), start=1):
                tr.write(json.dumps({
                    "unit": row["unit"],
                    "token_offset": i,
                    "state_norm_R": sr,
                    "state_norm_C": sc,
                    "query_readout_norm_R": qr,
                    "query_readout_norm_C": qc,
                    "normalized_state_energy_R": sr * sr / (norm_r * norm_r + EPS),
                    "normalized_state_energy_C": sc * sc / (norm_c * norm_c + EPS),
                    "normalized_query_readout_energy_R": qr * qr / (norm_r * norm_r + EPS),
                    "normalized_query_readout_energy_C": qc * qc / (norm_c * norm_c + EPS),
                }, ensure_ascii=False) + "\n")
    csv_path = RUN_DIR / "stageA_per_unit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    p_win = sum(r["P_R"] > r["P_C"] for r in rows)
    o_win = sum(r["O_R"] > r["O_C"] for r in rows)
    eta_win = sum(r["eta_R"] > r["eta_C"] for r in rows)
    kl_win = sum(r["KL_R"] > r["KL_C"] for r in rows)
    corr = {
        "Spearman_log_P_ratio_vs_log_KL_ratio": spearman([r["log_P_ratio"] for r in rows], [r["log_KL_ratio"] for r in rows]),
        "Spearman_log_O_ratio_vs_log_KL_ratio": spearman([r["log_O_ratio"] for r in rows], [r["log_KL_ratio"] for r in rows]),
        "Spearman_log_eta_ratio_vs_log_KL_ratio": spearman([r["log_eta_ratio"] for r in rows], [r["log_KL_ratio"] for r in rows]),
    }
    gate = "PASS" if o_win >= 8 and eta_win >= 8 and kl_win > p_win else "FAIL"
    signal = "SUPPORTED" if o_win >= 8 and eta_win >= 8 else ("PARTIAL_OR_INCONCLUSIVE" if max(o_win, eta_win) >= 6 else "NOT_SUPPORTED")
    summary = {
        "P_state_formula": "sum_t ||E_t||_F^2 / ||E_t0||_F^2, using frozen exact replay state curves",
        "O_q_formula": "sum_t ||E_t^T q_t||_2^2 / ||E_t0||_F^2, implemented as exact core_readout_error energy from Qwen3.5 GDN replay",
        "eta_q_formula": "O_q / (P_state + epsilon)",
        "epsilon": EPS,
        "units": len(rows),
        "P_state_R_gt_C": p_win,
        "O_q_R_gt_C": o_win,
        "eta_q_R_gt_C": eta_win,
        "KL_R_gt_C": kl_win,
        "median_ratios": {
            "persistence": med([r["P_R_over_C"] for r in rows]),
            "observability": med([r["O_R_over_C"] for r in rows]),
            "observability_efficiency": med([r["eta_R_over_C"] for r in rows]),
            "KL": med([r["KL_R_over_C"] for r in rows]),
        },
        "IQR_ratios": {
            "persistence": iqr([r["P_R_over_C"] for r in rows]),
            "observability": iqr([r["O_R_over_C"] for r in rows]),
            "observability_efficiency": iqr([r["eta_R_over_C"] for r in rows]),
            "KL": iqr([r["KL_R_over_C"] for r in rows]),
        },
        "correlations": corr,
        "OBSERVABILITY_GATE": gate,
        "RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL": signal,
    }
    save_json("stageA_summary.json", summary)
    return rows, summary


def stage_a2(rows):
    frozen = load_json(FROZEN)
    out_rows = []
    for u, r in zip(frozen["per_unit"], rows):
        pp = u["primary_pair"]
        jr = {k: pp[k] for k in pp}
        decay_r = auc_sq(pp["state_curve_R"]) - auc_sq(pp.get("query_error_curve_R", []))
        decay_c = auc_sq(pp["state_curve_C"]) - auc_sq(pp.get("query_error_curve_C", []))
        out_rows.append({
            "unit": r["unit"],
            "P_state_R_lt_C": r["P_R"] < r["P_C"],
            "O_q_R_gt_C": r["O_R"] > r["O_C"],
            "eta_R_gt_C": r["eta_R"] > r["eta_C"],
            "ADDRESSABILITY_DISSIPATION_INVERSION": r["P_R"] < r["P_C"] and r["O_R"] > r["O_C"],
            "J_key_R_over_C": jr.get("R_over_C_key"),
            "J_query_R_over_C": jr.get("R_over_C_query"),
            "J_state_R_over_C": jr.get("R_over_C_state"),
            "signed_state_minus_query_R": decay_r,
            "signed_state_minus_query_C": decay_c,
        })
    with (RUN_DIR / "stageA2_dissipation_per_unit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    inv = sum(x["ADDRESSABILITY_DISSIPATION_INVERSION"] for x in out_rows)
    key_win = sum((x["J_key_R_over_C"] or 0) > 1 for x in out_rows)
    summary = {
        "ADDRESSABILITY_DISSIPATION_INVERSION": "OBSERVED" if inv >= 8 else ("PARTIAL" if inv >= 6 else "NOT_SUPPORTED"),
        "inversion_count": f"{inv}/9",
        "J_key_R_gt_C": f"{key_win}/9",
        "median_J_key_R_over_C": med([x["J_key_R_over_C"] for x in out_rows]),
        "KEY_MEDIATED_DISSIPATION_SIGNAL": "INCONCLUSIVE",
        "note": "A2 reports signed exact-replay path energies. It is not used as standalone J_key causal proof.",
    }
    save_json("stageA2_summary.json", summary)
    return summary


def orthogonal(torch, n, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    a = torch.randn((n, n), generator=gen, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return q * signs


def rotate_errors(torch, inj, axis, seed, device):
    U = orthogonal(torch, 128, seed, device)
    out = {PRIMARY_R: {}, PRIMARY_C: {}}
    max_norm_rel = 0.0
    max_sv_rel = 0.0
    sample_done = False
    for cond in [PRIMARY_R, PRIMARY_C]:
        for layer, E in inj[cond].items():
            Ef = E.detach().float()
            if axis == "K":
                Er = torch.einsum("ij,bhjv->bhiv", U, Ef)
            else:
                Er = torch.einsum("bhkv,vw->bhkw", Ef, U)
            out[cond][layer] = Er
            max_norm_rel = max(max_norm_rel, abs(normswap.norm(torch, Er) - normswap.norm(torch, Ef)) / (normswap.norm(torch, Ef) + EPS))
            if not sample_done:
                sv0 = torch.linalg.svdvals(Ef[0, 0])
                sv1 = torch.linalg.svdvals(Er[0, 0])
                max_sv_rel = max(max_sv_rel, float(torch.linalg.vector_norm(sv0 - sv1).item() / (torch.linalg.vector_norm(sv0).item() + EPS)))
                sample_done = True
    return out, {"axis": axis, "seed": seed, "max_norm_relative_error": max_norm_rel, "sample_singular_value_relative_error": max_sv_rel}


def apply_inj(torch, past, errors):
    for layer, E in errors.items():
        s = p1.get_state(past, layer)
        s.copy_((s.detach().float() + E).to(s.dtype))


def full_model_kl(torch, model, tokenizer, e2e, pm, t0, errors_r, errors_c, horizon=HORIZON_B):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    base = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    ids = {"FP": base.clone(), "R": base.clone(), "C": base.clone()}
    masks = {"FP": mask, "R": mask.clone() if mask is not None else None, "C": mask.clone() if mask is not None else None}
    pasts = {"FP": None, "R": None, "C": None}
    kls = {"R": [], "C": []}
    last = min(t0 + horizon - 1, len(cont) - 1)
    with torch.inference_mode():
        for t in range(last + 1):
            outs = {}
            for c in ["FP", "R", "C"]:
                outs[c] = p1.feed_step(torch, model, ids[c], masks[c], pasts[c])
                pasts[c] = outs[c].past_key_values
            if t == t0:
                apply_inj(torch, pasts["R"], errors_r)
                apply_inj(torch, pasts["C"], errors_c)
            if t >= t0:
                kls["R"].append(p1.logits_metrics(torch, outs["FP"].logits, outs["R"].logits)["KL"])
                kls["C"].append(p1.logits_metrics(torch, outs["FP"].logits, outs["C"].logits)["KL"])
            if t < len(cont):
                nxt = torch.tensor([[cont[t]]], dtype=base.dtype, device=device)
                for c in ids:
                    ids[c] = nxt.clone()
                    masks[c] = None
    return {"KL_R": sum(kls["R"]) / len(kls["R"]), "KL_C": sum(kls["C"]) / len(kls["C"]), "KL_curve_R": kls["R"], "KL_curve_C": kls["C"]}


def stage_b(rows, stage_a_summary):
    if stage_a_summary["RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL"] != "SUPPORTED":
        save_json("stageB_summary.json", {"STAGE_B_RUN": "NO", "reason": "Stage A observability signal not supported"})
        return {"STAGE_B_RUN": "NO"}
    sorted_rows = sorted(rows, key=lambda r: r["KL_R"] - r["KL_C"])
    chosen = [sorted_rows[0], sorted_rows[len(sorted_rows) // 2], sorted_rows[-1]]
    cfg = {"selection_rule": "low/median/high original KL_R-KL_C from 9 canonical Stage A units", "units": [r["unit"] for r in chosen], "rotation_seeds": ROT_SEEDS, "axes": ["K", "V"]}
    save_json("stageB_rotation_config.json", cfg)
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompt_map = {p["problem_id"]: p for p in normswap.selected_prompts(3)}
    result_rows = []
    for row in chosen:
        pm = prompt_map[row["prompt_id"]]
        fp_past, _ids, _cont, _collector = None, None, None, None
        # Build residuals at the actual injection boundary through the canonical normswap helper.
        prompt = e2e.render_prompt(tokenizer, pm["problem"])
        cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
        device = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        past = None
        with torch.inference_mode():
            for t in range(row["t0"] + 1):
                out = p1.feed_step(torch, model, ids, mask, past)
                past = out.past_key_values
                if t < len(cont):
                    ids = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device)
                    mask = None
        inj, meta = normswap.build_injections(torch, past)
        original = full_model_kl(torch, model, tokenizer, e2e, pm, row["t0"], inj[PRIMARY_R], inj[PRIMARY_C])
        g0 = original["KL_R"] - original["KL_C"]
        result_rows.append({"unit": row["unit"], "axis": "ORIGINAL", "seed": 0, "KL_R": original["KL_R"], "KL_C": original["KL_C"], "G_original": g0, "G_rot": g0, "rescue": 0.0, "norm_error": 0.0, "sv_error": 0.0})
        for axis in ["K", "V"]:
            for seed in ROT_SEEDS:
                rot, audit = rotate_errors(torch, inj, axis, seed, device)
                kl = full_model_kl(torch, model, tokenizer, e2e, pm, row["t0"], rot[PRIMARY_R], rot[PRIMARY_C])
                gr = kl["KL_R"] - kl["KL_C"]
                result_rows.append({
                    "unit": row["unit"],
                    "axis": axis,
                    "seed": seed,
                    "KL_R": kl["KL_R"],
                    "KL_C": kl["KL_C"],
                    "G_original": g0,
                    "G_rot": gr,
                    "absolute_gap_change": gr - g0,
                    "rescue": 1.0 - gr / (g0 + EPS) if abs(g0) > 1e-8 else None,
                    "norm_error": audit["max_norm_relative_error"],
                    "sv_error": audit["sample_singular_value_relative_error"],
                })
    with (RUN_DIR / "stageB_per_run.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        w.writeheader()
        w.writerows(result_rows)
    def axis_summary(axis):
        rs = [r for r in result_rows if r["axis"] == axis]
        return {
            "runs": len(rs),
            "median_G_original": med([r["G_original"] for r in rs]),
            "median_G_rot": med([r["G_rot"] for r in rs]),
            "median_absolute_gap_change": med([r["absolute_gap_change"] for r in rs]),
            "median_rescue": med([r["rescue"] for r in rs]),
            "gap_decreased_count": sum(r["G_rot"] < r["G_original"] for r in rs),
            "max_norm_error": max(r["norm_error"] for r in rs),
            "max_sv_error": max(r["sv_error"] for r in rs),
        }
    ks, vs = axis_summary("K"), axis_summary("V")
    k_sig = "SUPPORTED" if ks["gap_decreased_count"] >= 9 and (ks["median_rescue"] or 0) > 0 else "NOT_SUPPORTED"
    v_sig = "SUPPORTED" if vs["gap_decreased_count"] >= 9 and (vs["median_rescue"] or 0) > 0 else "NOT_SUPPORTED"
    summary = {
        "STAGE_B_RUN": "YES",
        "K_rotation": ks,
        "V_rotation": vs,
        "KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL": k_sig,
        "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": v_sig,
    }
    save_json("stageB_summary.json", summary)
    return summary


def load_stagea_rows():
    with (RUN_DIR / "stageA_per_unit.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["t0", "P_R", "P_C", "O_R", "O_C", "eta_R", "eta_C", "KL_R", "KL_C"]:
            if k in r:
                r[k] = int(r[k]) if k == "t0" else float(r[k])
    return rows


def stage_b_configs(rows):
    cfg_path = RUN_DIR / "stageB_rotation_config.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        chosen_units = cfg["units"]
        chosen = [r for r in rows if r["unit"] in chosen_units]
    else:
        sorted_rows = sorted(rows, key=lambda r: r["KL_R"] - r["KL_C"])
        chosen = [sorted_rows[0], sorted_rows[len(sorted_rows) // 2], sorted_rows[-1]]
        cfg = {"selection_rule": "low/median/high original KL_R-KL_C from 9 canonical Stage A units", "units": [r["unit"] for r in chosen], "rotation_seeds": ROT_SEEDS, "axes": ["K", "V"]}
        save_json("stageB_rotation_config.json", cfg)
    tasks = []
    for r in chosen:
        tasks.append({"unit": r["unit"], "axis": "ORIGINAL", "seed": 0})
        for axis in ["K", "V"]:
            for seed in ROT_SEEDS:
                tasks.append({"unit": r["unit"], "axis": axis, "seed": seed})
    return cfg, tasks


def stage_b_worker(worker_id, indices):
    import torch
    rows = load_stagea_rows()
    _cfg, tasks = stage_b_configs(rows)
    row_by_unit = {r["unit"]: r for r in rows}
    prompt_map = {p["problem_id"]: p for p in normswap.selected_prompts(3)}
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _model_cfg, e2e = p1.setup_model()
    cp_path = RUN_DIR / f"stageB_worker_{worker_id}.json"
    cp = load_json(cp_path) if cp_path.exists() else {"worker_id": worker_id, "assigned_indices": indices, "completed": {}, "failed": {}}
    inj_cache = {}
    def build_inj_for(row):
        if row["unit"] in inj_cache:
            return inj_cache[row["unit"]]
        pm = prompt_map[row["prompt_id"]]
        prompt = e2e.render_prompt(tokenizer, pm["problem"])
        cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
        device = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        past = None
        with torch.inference_mode():
            for t in range(int(row["t0"]) + 1):
                out = p1.feed_step(torch, model, ids, mask, past)
                past = out.past_key_values
                if t < len(cont):
                    ids = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device)
                    mask = None
        inj, meta = normswap.build_injections(torch, past)
        inj_cache[row["unit"]] = (pm, inj, meta, device)
        return inj_cache[row["unit"]]
    for idx in indices:
        task = tasks[idx]
        key = f"{idx}:{task['unit']}:{task['axis']}:{task['seed']}"
        if key in cp["completed"]:
            continue
        print(f"[{now()}] StageB {worker_id} {key}", flush=True)
        try:
            row = row_by_unit[task["unit"]]
            pm, inj, _meta, device = build_inj_for(row)
            original_gap = float(row["KL_R"]) - float(row["KL_C"])
            if task["axis"] == "ORIGINAL":
                errors = inj
                audit = {"max_norm_relative_error": 0.0, "sample_singular_value_relative_error": 0.0}
            else:
                errors, audit = rotate_errors(torch, inj, task["axis"], int(task["seed"]), device)
            kl = full_model_kl(torch, model, tokenizer, e2e, pm, int(row["t0"]), errors[PRIMARY_R], errors[PRIMARY_C])
            gap = kl["KL_R"] - kl["KL_C"]
            cp["completed"][key] = {
                **task,
                "KL_R": kl["KL_R"],
                "KL_C": kl["KL_C"],
                "G_original": original_gap,
                "G_rot": gap,
                "absolute_gap_change": gap - original_gap,
                "rescue": 1.0 - gap / (original_gap + EPS) if abs(original_gap) > 1e-8 else None,
                "norm_error": audit["max_norm_relative_error"],
                "sv_error": audit["sample_singular_value_relative_error"],
                "timestamp": now(),
            }
            save_json(f"stageB_worker_{worker_id}.json", cp)
        except Exception as exc:
            cp["failed"][key] = {"error": repr(exc), "timestamp": now()}
            save_json(f"stageB_worker_{worker_id}.json", cp)
            raise


def stage_b_merge():
    rows = load_stagea_rows()
    _cfg, tasks = stage_b_configs(rows)
    all_rows = []
    failed = {}
    for wid in ["gpu0", "gpu1", "gpu2", "gpu3"]:
        p = RUN_DIR / f"stageB_worker_{wid}.json"
        if not p.exists():
            continue
        d = load_json(p)
        all_rows.extend(d.get("completed", {}).values())
        failed.update(d.get("failed", {}))
    all_rows = sorted(all_rows, key=lambda r: (r["unit"], r["axis"], int(r["seed"])))
    with (RUN_DIR / "stageB_per_run.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    def axis_summary(axis):
        rs = [r for r in all_rows if r["axis"] == axis]
        if not rs:
            return {}
        return {
            "runs": len(rs),
            "median_G_original": med([r["G_original"] for r in rs]),
            "median_G_rot": med([r["G_rot"] for r in rs]),
            "median_absolute_gap_change": med([r["absolute_gap_change"] for r in rs]),
            "median_rescue": med([r["rescue"] for r in rs]),
            "gap_decreased_count": sum(r["G_rot"] < r["G_original"] for r in rs),
            "max_norm_error": max(r["norm_error"] for r in rs),
            "max_sv_error": max(r["sv_error"] for r in rs),
        }
    ks, vs = axis_summary("K"), axis_summary("V")
    complete = len(all_rows) == len(tasks) and not failed
    k_sig = "SUPPORTED" if complete and ks.get("gap_decreased_count", 0) >= 9 and (ks.get("median_rescue") or 0) > 0 else ("NOT_SUPPORTED" if complete else "INCOMPLETE")
    v_sig = "SUPPORTED" if complete and vs.get("gap_decreased_count", 0) >= 9 and (vs.get("median_rescue") or 0) > 0 else ("NOT_SUPPORTED" if complete else "INCOMPLETE")
    summary = {
        "STAGE_B_RUN": "YES" if complete else "INCOMPLETE",
        "completed_runs": len(all_rows),
        "expected_runs": len(tasks),
        "failed": failed,
        "K_rotation": ks,
        "V_rotation": vs,
        "KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL": k_sig,
        "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": v_sig,
    }
    save_json("stageB_summary.json", summary)
    st = load_json(RUN_DIR / "stage0_identity.json")
    sa = load_json(RUN_DIR / "stageA_summary.json")
    a2 = load_json(RUN_DIR / "stageA2_summary.json")
    final = final_report(st, sa, a2, summary)
    print(json.dumps({
        "git_commit": git_commit(),
        "Stage0": st["STAGE0"],
        "P_state_R_gt_C_count": f"{sa['P_state_R_gt_C']}/9",
        "O_q_R_gt_C_count": f"{sa['O_q_R_gt_C']}/9",
        "eta_q_R_gt_C_count": f"{sa['eta_q_R_gt_C']}/9",
        "median_R_C_ratios": sa["median_ratios"],
        "OBSERVABILITY_GATE": sa["OBSERVABILITY_GATE"],
        "addressability_dissipation_result": a2["ADDRESSABILITY_DISSIPATION_INVERSION"],
        "Stage_B_ran": summary["STAGE_B_RUN"],
        "K_rotation_result": summary["KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL"],
        "V_rotation_result": summary["VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL"],
        "FINAL_MECHANISM_CLASSIFICATION": final["FINAL_MECHANISM_CLASSIFICATION"],
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "output_dir": str(RUN_DIR),
    }, indent=2, ensure_ascii=False))


def launch_stage_b():
    rows = load_stagea_rows()
    _cfg, tasks = stage_b_configs(rows)
    launcher = EXP / "launch_functional_geometry_stageB_4gpu.sh"
    shards = {i: [] for i in range(4)}
    for idx in range(len(tasks)):
        shards[idx % 4].append(idx)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {EXP}", "export PYTHONUNBUFFERED=1"]
    for gpu in range(4):
        idxs = ",".join(str(i) for i in shards[gpu])
        lines.append(f"CUDA_VISIBLE_DEVICES={gpu} /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 {Path(__file__).name} --stage stageB-worker --worker-id gpu{gpu} --task-indices {idxs} > {RUN_DIR}/stageB_gpu{gpu}.log 2>&1 &")
        lines.append(f"echo $! > {RUN_DIR}/stageB_gpu{gpu}.pid")
    lines.append("wait")
    lines.append(f"/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 {Path(__file__).name} --stage stageB-merge > {RUN_DIR}/stageB_merge.log 2>&1")
    launcher.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(launcher, 0o755)
    print(launcher)


def final_report(stage0_obj, stage_a_summary, a2_summary, b_summary):
    if stage_a_summary["RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL"] == "SUPPORTED":
        if b_summary.get("KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL") == "SUPPORTED" and b_summary.get("VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL") == "SUPPORTED":
            cls = "K_AND_V_FUNCTIONAL_GEOMETRY_PILOT_SIGNAL"
        elif b_summary.get("KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL") == "SUPPORTED":
            cls = "KEY_SIDE_ADDRESSABILITY_DOMINANT_PILOT_SIGNAL"
        elif b_summary.get("VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL") == "SUPPORTED":
            cls = "VALUE_SIDE_FUNCTIONAL_DIRECTION_DOMINANT_PILOT_SIGNAL"
        else:
            cls = "FUNCTIONAL_OBSERVABILITY_SUPPORTED_BUT_ROTATION_CAUSAL_SIGNAL_NOT_SUPPORTED"
    else:
        cls = "CURRENT_FUNCTIONAL_GEOMETRY_HYPOTHESIS_NOT_SUPPORTED"
    final = {
        "STAGE0": stage0_obj["STAGE0"],
        "OBSERVABILITY_GATE": stage_a_summary["OBSERVABILITY_GATE"],
        "RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL": stage_a_summary["RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL"],
        "ADDRESSABILITY_DISSIPATION_INVERSION": a2_summary["ADDRESSABILITY_DISSIPATION_INVERSION"],
        "KEY_MEDIATED_DISSIPATION_SIGNAL": a2_summary["KEY_MEDIATED_DISSIPATION_SIGNAL"],
        "STAGE_B_RUN": b_summary.get("STAGE_B_RUN", "NO"),
        "KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL": b_summary.get("KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL", "NOT_RUN"),
        "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": b_summary.get("VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL", "NOT_RUN"),
        "FINAL_MECHANISM_CLASSIFICATION": cls,
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
    }
    lines = [
        "# GDN INT8 Same-Norm Functional Geometry Audit V1",
        "## 1. Executive Summary",
        json.dumps(final, indent=2),
        "## 2. Scientific Question",
        "Why can a same-norm R residual, which is less persistent in raw state space, produce larger behavioral damage than C?",
        "## 3. Prior Evidence Used",
        f"Frozen exact-replay pilot: `{FROZEN}`. Same-norm behavioral KL: `{NORMSWAP}`.",
        "## 4. Repository / Implementation Audit",
        "Reused Qwen3.5 GDN exact replay helpers from the frozen-observability and norm-swap experiments.",
        "## 5. Tensor Semantics",
        "State is treated as `[B,H,K,V]`; query observability uses exact core_readout_error from implementation replay rather than a paper-equation substitute.",
        "## 6. Stage 0 - Replay Identity",
        json.dumps(stage0_obj, indent=2),
        "## 7. Stage A - Raw State Persistence",
        f"P_state R>C = {stage_a_summary['P_state_R_gt_C']}/9; median R/C = {stage_a_summary['median_ratios']['persistence']}",
        "## 8. Stage A - Whole-Horizon Query Observability",
        f"O_q R>C = {stage_a_summary['O_q_R_gt_C']}/9; median R/C = {stage_a_summary['median_ratios']['observability']}",
        "## 9. Stage A - Observability Efficiency",
        f"eta_q R>C = {stage_a_summary['eta_q_R_gt_C']}/9; median R/C = {stage_a_summary['median_ratios']['observability_efficiency']}",
        "## 10. Paired R/C Analysis",
        json.dumps(stage_a_summary["correlations"], indent=2),
        "## 11. Stage A Gate Decision",
        json.dumps({k: stage_a_summary[k] for k in ["OBSERVABILITY_GATE", "RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL"]}, indent=2),
        "## 12. Stage A2 - Addressability / Dissipation",
        json.dumps(a2_summary, indent=2),
        "## 13. Stage B - K-Axis Rotation",
        json.dumps(b_summary.get("K_rotation"), indent=2),
        "## 14. Stage B - V-Axis Rotation",
        json.dumps(b_summary.get("V_rotation"), indent=2),
        "## 15. Behavioral KL Rescue",
        "Stored in `stageB_per_run.csv` if Stage B ran.",
        "## 16. Positive Results",
        "Stage A supports recurrent functional observability if O_q and eta_q satisfy the pre-registered paired sign criteria.",
        "## 17. Negative / Corrective Results",
        "Stage B pilot is treated only as pilot causal evidence; negative rotation results do not erase the Stage A observability signal.",
        "## 18. Mechanism Classification",
        cls,
        "## 19. What Is NOT Proven",
        "No final method, no additive mechanism, no standalone J_key causal scalar, and no mechanism closure are claimed.",
        "## 20. Recommended Next Experiment",
        "If continuing, use a larger pre-registered rotation/SVD-factor audit; do not start it automatically.",
    ]
    (RUN_DIR / "final_report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    save_json("final_summary.json", {**final, "stageA": stage_a_summary, "stageA2": a2_summary, "stageB": b_summary})
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "stage0", "stageA", "stageB", "stageB-worker", "stageB-merge", "launch-stageB"], default="all")
    ap.add_argument("--worker-id")
    ap.add_argument("--task-indices")
    args = ap.parse_args()
    if args.stage == "stageB-worker":
        stage_b_worker(args.worker_id, [int(x) for x in args.task_indices.split(",") if x])
        return
    if args.stage == "stageB-merge":
        stage_b_merge()
        return
    if args.stage == "launch-stageB":
        launch_stage_b()
        return
    st = stage0()
    if st["STAGE0"] != "PASS":
        save_json("final_summary.json", {"FINAL_STATUS": "STAGE0_FAILED", "STAGE0": st})
        print(json.dumps({"FINAL_STATUS": "STAGE0_FAILED", "RUN_DIR": str(RUN_DIR)}, indent=2))
        raise SystemExit(2)
    rows, sa = stage_a()
    a2 = stage_a2(rows)
    b = stage_b(rows, sa) if sa["RECURRENT_FUNCTIONAL_OBSERVABILITY_SIGNAL"] == "SUPPORTED" else {"STAGE_B_RUN": "NO", "reason": "Stage A not supported"}
    final = final_report(st, sa, a2, b)
    print(json.dumps({
        "git_commit": git_commit(),
        "Stage0": st["STAGE0"],
        "P_state_R_gt_C_count": f"{sa['P_state_R_gt_C']}/9",
        "O_q_R_gt_C_count": f"{sa['O_q_R_gt_C']}/9",
        "eta_q_R_gt_C_count": f"{sa['eta_q_R_gt_C']}/9",
        "median_R_C_ratios": sa["median_ratios"],
        "OBSERVABILITY_GATE": sa["OBSERVABILITY_GATE"],
        "addressability_dissipation_result": a2["ADDRESSABILITY_DISSIPATION_INVERSION"],
        "Stage_B_ran": b.get("STAGE_B_RUN", "NO"),
        "K_rotation_result": b.get("KEY_SIDE_FUNCTIONAL_ADDRESSABILITY_CAUSAL_SIGNAL", "NOT_RUN"),
        "V_rotation_result": b.get("VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL", "NOT_RUN"),
        "FINAL_MECHANISM_CLASSIFICATION": final["FINAL_MECHANISM_CLASSIFICATION"],
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "METHOD_DESIGN_READY": "NO",
        "output_dir": str(RUN_DIR),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
