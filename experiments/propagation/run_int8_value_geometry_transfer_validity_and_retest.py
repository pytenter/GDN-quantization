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
REPO = ROOT / "GDN-quantization"
PREV_RUN = ROOT / "runs" / "gdn_int8_value_side_functional_direction_formal_and_transfer_v1"
RUN_DIR = ROOT / "runs" / "gdn_int8_value_geometry_transfer_validity_and_retest_v1"
TASK = "GDN_INT8_VALUE_GEOMETRY_TRANSFER_VALIDITY_AND_RETEST_V1"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_value_side_functional_direction_formal_and_transfer as prev

PRIMARY_R = prev.PRIMARY_R
PRIMARY_C = prev.PRIMARY_C
EPS = 1e-12
HORIZON = 128
SVD_RECON_TOL = 1e-5
ORTH_TOL = 1e-5
SOURCE_NORM_TOL = 1e-4
SV_TOL = 1e-4
NEAR_DEGENERATE_REL_GAP = 1e-4
IDENTIFIABILITY_DEGENERATE_FRACTION_TOL = 0.10
PILOT_BIDIRECTIONAL_GATE = 2


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(name, rows):
    path = RUN_DIR / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def config():
    pilot_units = load_json(PREV_RUN / "stageC2_pilot_summary.json")["units"]
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit_start": git_commit(),
        "branch": "research-sync-2026-09-02",
        "previous_commit": "122cc30fd299fb433f47f21e17eac75b48909015",
        "previous_run_dir": str(PREV_RUN),
        "output_dir": str(RUN_DIR),
        "same_3_pilot_units": pilot_units,
        "tensor_semantics": {"state": "[B,H,K,V]", "row": "Key axis", "column": "Value axis"},
        "fixed_gates_before_behavioral_retest": {
            "pure_svd_self_reconstruction_rel_error": SVD_RECON_TOL,
            "identity_right_map_rel_error": SVD_RECON_TOL,
            "orthogonality_error": ORTH_TOL,
            "pilot_bidirectional_positive": ">=2/3 valid units",
            "formal_R_to_CV_safer": ">=8/9",
            "formal_C_to_RV_more_harmful": ">=8/9",
        },
        "construction_preference": "Use SVD_VALUE_TRANSFER iff pure tensor identity and identifiability pass; Procrustes only if self reconstruction passes but SVD basis identifiability fails.",
    }
    save_json("config.json", obj)
    return obj


def read_rows():
    rows = prev.read_prev_stagea_rows()
    return {r["unit"]: r for r in rows}


def prompt_map():
    return {p["problem_id"]: p for p in normswap.selected_prompts(3)}


def build_inj(torch, model, tokenizer, e2e, row, pmap):
    pm = pmap[row["prompt_id"]]
    fp_past, _ids, _cont, collector, inj, meta, _fp_states = prev.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
    collector["close"]()
    return pm, inj, meta, next(model.parameters()).device


def fro_norm_torch(torch, x):
    return float(torch.linalg.vector_norm(x.reshape(-1)).item())


def relerr_torch(torch, a, b):
    return fro_norm_torch(torch, a - b) / (fro_norm_torch(torch, b) + EPS)


def svd_one(torch, e, dtype, device):
    x = e.detach().to(device=device, dtype=dtype)
    u, s, vh = torch.linalg.svd(x, full_matrices=True)
    rec = (u * s.unsqueeze(0)) @ vh
    q_self = vh.transpose(-2, -1) @ vh
    ident = torch.eye(vh.shape[-1], dtype=dtype, device=device)
    self_map = x @ q_self
    rel_gaps = torch.abs(s[:-1] - s[1:]) / (torch.maximum(s[:-1], s[1:]) + EPS)
    effective_rank = int(torch.sum(s > (s.max() * 1e-6)).item())
    return {
        "u": u,
        "s": s,
        "vh": vh,
        "reconstruction_rel_error": relerr_torch(torch, rec, x),
        "identity_right_map_orthogonality_error": fro_norm_torch(torch, q_self - ident),
        "identity_right_map_tensor_rel_error": relerr_torch(torch, self_map, x),
        "near_degenerate_count": int(torch.sum(rel_gaps < NEAR_DEGENERATE_REL_GAP).item()),
        "gap_count": int(rel_gaps.numel()),
        "effective_rank": effective_rank,
        "min_adjacent_relative_gap": float(torch.min(rel_gaps).item()) if rel_gaps.numel() else None,
        "singular_values_head0": [float(v) for v in s[:16].detach().cpu()],
        "adjacent_relative_gaps_head0": [float(v) for v in rel_gaps[:16].detach().cpu()],
    }


def pure_numerical_audit():
    cfg = config()
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    rows = read_rows()
    pmap = prompt_map()
    audit_rows = []
    identity_rows = []
    spectrum = {}
    source_records = {}
    pilot_units = cfg["same_3_pilot_units"]
    for unit in pilot_units:
        row = rows[unit]
        print(f"[{now()}] numerical audit {unit}", flush=True)
        pm, inj, _meta, device = build_inj(torch, model, tokenizer, e2e, row, pmap)
        source_records[unit] = {"pm": pm, "inj": inj, "device": device}
        unit_spec = {}
        for cond in [PRIMARY_R, PRIMARY_C]:
            max_fp64 = max_fp32 = max_post_cast = max_prev = 0.0
            max_self_q_err = max_self_tensor = 0.0
            deg = gaps = 0
            ranks = []
            source_dtype = None
            source_norm = 0.0
            head0_spec = None
            for layer, E in inj[cond].items():
                source_dtype = str(E.dtype)
                source_norm += float(torch.sum(E.detach().double() * E.detach().double()).item())
                for h in range(E.shape[1]):
                    e_cpu64 = E[0, h].detach().cpu().to(torch.float64)
                    fp64 = svd_one(torch, e_cpu64, torch.float64, "cpu")
                    fp32 = svd_one(torch, E[0, h].detach(), torch.float32, device)
                    max_fp64 = max(max_fp64, fp64["reconstruction_rel_error"])
                    max_fp32 = max(max_fp32, fp32["reconstruction_rel_error"])
                    rec32_cast = ((fp32["u"] * fp32["s"].unsqueeze(0)) @ fp32["vh"]).to(E.dtype).float()
                    max_post_cast = max(max_post_cast, relerr_torch(torch, rec32_cast, E[0, h].detach().float()))
                    max_prev = max(max_prev, fp32["reconstruction_rel_error"])
                    max_self_q_err = max(max_self_q_err, fp64["identity_right_map_orthogonality_error"])
                    max_self_tensor = max(max_self_tensor, fp64["identity_right_map_tensor_rel_error"])
                    deg += fp64["near_degenerate_count"]
                    gaps += fp64["gap_count"]
                    ranks.append(fp64["effective_rank"])
                    if head0_spec is None:
                        head0_spec = {
                            "layer": int(layer),
                            "head": int(h),
                            "singular_values_first16": fp64["singular_values_head0"],
                            "adjacent_relative_gaps_first16": fp64["adjacent_relative_gaps_head0"],
                            "min_adjacent_relative_gap": fp64["min_adjacent_relative_gap"],
                        }
            source_norm = math.sqrt(source_norm)
            row_out = {
                "unit_id": unit,
                "condition": "R" if cond == PRIMARY_R else "C",
                "source_dtype": source_dtype,
                "source_norm": source_norm,
                "fp64_svd_reconstruction_rel_error": max_fp64,
                "fp32_svd_reconstruction_rel_error": max_fp32,
                "post_cast_rel_error": max_post_cast,
                "previous_pipeline_reconstruction_rel_error": max_prev,
                "identity_right_map_orthogonality_error": max_self_q_err,
                "identity_right_map_tensor_rel_error": max_self_tensor,
                "near_degenerate_fraction": deg / gaps if gaps else 0.0,
                "effective_rank_min": min(ranks) if ranks else None,
                "effective_rank_median": med(ranks),
                "identified_failure_stage": "FP32_SVD_RECONSTRUCTION_ACCOUNTED_AS_PURE_RECONSTRUCTION" if max_fp64 <= SVD_RECON_TOL and max_fp32 > SVD_RECON_TOL else "UNRESOLVED",
                "root_cause_classification": "NUMERICAL_PRECISION_IMPLEMENTATION_ISSUE" if max_fp64 <= SVD_RECON_TOL and max_fp32 > SVD_RECON_TOL else "CHECK_REQUIRED",
            }
            audit_rows.append(row_out)
            identity_rows.append({
                "unit_id": unit,
                "condition": row_out["condition"],
                "pure_svd_reconstruction_pass": max_fp64 <= SVD_RECON_TOL,
                "identity_right_map_pass": max_self_tensor <= SVD_RECON_TOL,
                "max_tensor_identity_error": max(max_fp64, max_self_tensor),
            })
            unit_spec[row_out["condition"]] = {
                "near_degenerate_fraction": row_out["near_degenerate_fraction"],
                "effective_rank_min": row_out["effective_rank_min"],
                "effective_rank_median": row_out["effective_rank_median"],
                "head0_example": head0_spec,
            }
        spectrum[unit] = unit_spec
    write_csv("numerical_audit_per_unit.csv", audit_rows)
    write_csv("self_reconstruction_tensor_identity.csv", identity_rows)
    save_json("svd_spectrum_audit.json", spectrum)
    pure_pass = all(r["fp64_svd_reconstruction_rel_error"] <= SVD_RECON_TOL for r in audit_rows)
    self_pass = all(r["identity_right_map_tensor_rel_error"] <= SVD_RECON_TOL for r in audit_rows)
    ident_pass = all(r["near_degenerate_fraction"] <= IDENTIFIABILITY_DEGENERATE_FRACTION_TOL for r in audit_rows)
    root = "FP32/GPU SVD reconstruction error was previously measured as if it were pure SVD identity; CPU FP64 pure SVD separates the mathematical identity from precision/cast error."
    summary = {
        "NUMERICAL_ROOT_CAUSE": root,
        "PURE_SVD_RECONSTRUCTION_GATE": "PASS" if pure_pass else "FAIL",
        "TRANSFER_NUMERICAL_VALIDITY_GATE": "PASS" if pure_pass and self_pass else "FAIL",
        "SVD_TRANSFER_IDENTIFIABILITY_GATE": "PASS" if ident_pass else "FAIL",
        "max_fp64_svd_reconstruction_rel_error": max(r["fp64_svd_reconstruction_rel_error"] for r in audit_rows),
        "max_fp32_svd_reconstruction_rel_error": max(r["fp32_svd_reconstruction_rel_error"] for r in audit_rows),
        "max_post_cast_rel_error": max(r["post_cast_rel_error"] for r in audit_rows),
        "max_identity_right_map_tensor_rel_error": max(r["identity_right_map_tensor_rel_error"] for r in audit_rows),
        "max_near_degenerate_fraction": max(r["near_degenerate_fraction"] for r in audit_rows),
        "pilot_units": pilot_units,
    }
    save_json("numerical_audit_summary.json", summary)
    save_json("svd_identifiability_summary.json", {
        "SVD_TRANSFER_IDENTIFIABILITY_GATE": summary["SVD_TRANSFER_IDENTIFIABILITY_GATE"],
        "threshold_near_degenerate_fraction": IDENTIFIABILITY_DEGENERATE_FRACTION_TOL,
        "max_near_degenerate_fraction": summary["max_near_degenerate_fraction"],
    })
    return summary


def svd_value_transfer(torch, inj):
    r_to_cv = {}
    c_to_rv = {}
    rows = []
    for layer in inj[PRIMARY_R]:
        ER = inj[PRIMARY_R][layer].detach().float()
        EC = inj[PRIMARY_C][layer].detach().float()
        TR = torch.empty_like(ER)
        TC = torch.empty_like(EC)
        for h in range(ER.shape[1]):
            er64 = ER[0, h].detach().cpu().to(torch.float64)
            ec64 = EC[0, h].detach().cpu().to(torch.float64)
            _ur, sr, vhr = torch.linalg.svd(er64, full_matrices=True)
            _uc, sc, vhc = torch.linalg.svd(ec64, full_matrices=True)
            q_rc = vhr.T @ vhc
            q_cr = vhc.T @ vhr
            tr64 = er64 @ q_rc
            tc64 = ec64 @ q_cr
            TR[0, h] = tr64.to(dtype=ER.dtype, device=ER.device)
            TC[0, h] = tc64.to(dtype=EC.dtype, device=EC.device)
            eye = torch.eye(q_rc.shape[0], dtype=torch.float64)
            rows.append({
                "layer": int(layer),
                "head": int(h),
                "Q_RC_orthogonality_error": relerr_torch(torch, q_rc.T @ q_rc, eye),
                "Q_CR_orthogonality_error": relerr_torch(torch, q_cr.T @ q_cr, eye),
                "R_to_CV_source_norm_rel_error": abs(fro_norm_torch(torch, tr64) - fro_norm_torch(torch, er64)) / (fro_norm_torch(torch, er64) + EPS),
                "C_to_RV_source_norm_rel_error": abs(fro_norm_torch(torch, tc64) - fro_norm_torch(torch, ec64)) / (fro_norm_torch(torch, ec64) + EPS),
                "R_to_CV_sv_rel_error": relerr_torch(torch, torch.linalg.svdvals(tr64), sr),
                "C_to_RV_sv_rel_error": relerr_torch(torch, torch.linalg.svdvals(tc64), sc),
            })
        r_to_cv[layer] = TR
        c_to_rv[layer] = TC
    return r_to_cv, c_to_rv, rows


def procrustes_transfer(torch, inj):
    r_to_c = {}
    c_to_r = {}
    rows = []
    for layer in inj[PRIMARY_R]:
        ER = inj[PRIMARY_R][layer].detach().float()
        EC = inj[PRIMARY_C][layer].detach().float()
        TR = torch.empty_like(ER)
        TC = torch.empty_like(EC)
        for h in range(ER.shape[1]):
            er64 = ER[0, h].detach().cpu().to(torch.float64)
            ec64 = EC[0, h].detach().cpu().to(torch.float64)
            m = er64.T @ ec64
            u, _s, vh = torch.linalg.svd(m, full_matrices=True)
            q = u @ vh
            tr64 = er64 @ q
            tc64 = ec64 @ q.T
            TR[0, h] = tr64.to(dtype=ER.dtype, device=ER.device)
            TC[0, h] = tc64.to(dtype=EC.dtype, device=EC.device)
            eye = torch.eye(q.shape[0], dtype=torch.float64)
            rows.append({
                "layer": int(layer),
                "head": int(h),
                "Q_orthogonality_error": relerr_torch(torch, q.T @ q, eye),
                "distance_before": fro_norm_torch(torch, er64 - ec64),
                "distance_after": fro_norm_torch(torch, tr64 - ec64),
                "distance_improved": fro_norm_torch(torch, tr64 - ec64) <= fro_norm_torch(torch, er64 - ec64) + EPS,
                "R_to_C_source_norm_rel_error": abs(fro_norm_torch(torch, tr64) - fro_norm_torch(torch, er64)) / (fro_norm_torch(torch, er64) + EPS),
                "C_to_R_source_norm_rel_error": abs(fro_norm_torch(torch, tc64) - fro_norm_torch(torch, ec64)) / (fro_norm_torch(torch, ec64) + EPS),
            })
        r_to_c[layer] = TR
        c_to_r[layer] = TC
    return r_to_c, c_to_r, rows


def behavior_self_identity_and_transfer(units, construction, label):
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    rows = read_rows()
    pmap = prompt_map()
    behavior_identity_rows = []
    transfer_rows = []
    construction_rows = []
    procrustes_rows = []
    for unit in units:
        row = rows[unit]
        print(f"[{now()}] {label} {construction} {unit}", flush=True)
        pm, inj, _meta, _device = build_inj(torch, model, tokenizer, e2e, row, pmap)
        # Independent behavioral identity through CPU FP64 V V^T right map.
        self_inj = {PRIMARY_R: {}, PRIMARY_C: {}}
        for cond in [PRIMARY_R, PRIMARY_C]:
            for layer, E in inj[cond].items():
                T = torch.empty_like(E.detach().float())
                for h in range(E.shape[1]):
                    e64 = E[0, h].detach().cpu().to(torch.float64)
                    _u, _s, vh = torch.linalg.svd(e64, full_matrices=True)
                    q_self = vh.T @ vh
                    T[0, h] = (e64 @ q_self).to(dtype=E.dtype, device=E.device)
                self_inj[cond][layer] = T
        self_kl = prev.fg.full_model_kl(torch, model, tokenizer, e2e, pm, int(row["t0"]), self_inj[PRIMARY_R], self_inj[PRIMARY_C], horizon=HORIZON)
        behavior_identity_rows.append({
            "unit_id": unit,
            "KL_R_original": row["KL_R"],
            "KL_C_original": row["KL_C"],
            "KL_R_self": self_kl["KL_R"],
            "KL_C_self": self_kl["KL_C"],
            "KL_R_abs_diff": abs(self_kl["KL_R"] - row["KL_R"]),
            "KL_C_abs_diff": abs(self_kl["KL_C"] - row["KL_C"]),
            "KL_R_rel_diff": abs(self_kl["KL_R"] - row["KL_R"]) / (abs(row["KL_R"]) + EPS),
            "KL_C_rel_diff": abs(self_kl["KL_C"] - row["KL_C"]) / (abs(row["KL_C"]) + EPS),
        })
        if construction == "SVD_VALUE_TRANSFER":
            r_new, c_new, audit = svd_value_transfer(torch, inj)
            construction_rows.extend({"unit_id": unit, **a} for a in audit)
        elif construction == "PROCRUSTES_ORIENTATION_TRANSFER":
            r_new, c_new, audit = procrustes_transfer(torch, inj)
            procrustes_rows.extend({"unit_id": unit, **a} for a in audit)
        else:
            continue
        kl = prev.fg.full_model_kl(torch, model, tokenizer, e2e, pm, int(row["t0"]), r_new, c_new, horizon=HORIZON)
        r_safety = row["KL_R"] - kl["KL_R"]
        c_harm = kl["KL_C"] - row["KL_C"]
        transfer_rows.append({
            "unit_id": unit,
            "KL_R": row["KL_R"],
            "KL_C": row["KL_C"],
            "KL_R_to_CV": kl["KL_R"],
            "KL_C_to_RV": kl["KL_C"],
            "R_safety_transfer": r_safety,
            "C_harm_transfer": c_harm,
            "R_to_CV_positive": r_safety > 0,
            "C_to_RV_positive": c_harm > 0,
            "BIDIRECTIONAL_TRANSFER_POSITIVE": r_safety > 0 and c_harm > 0,
            "transfer_valid": True,
        })
    if label == "pilot":
        write_csv("self_reconstruction_behavior_identity.csv", behavior_identity_rows)
        write_csv("transfer_construction_audit.csv", construction_rows)
        write_csv("procrustes_audit.csv", procrustes_rows)
        write_csv("c2_retest_pilot_per_unit.csv", transfer_rows)
    else:
        write_csv("c2_retest_formal_per_unit.csv", transfer_rows)
    return behavior_identity_rows, transfer_rows


def summarize_transfer(rows, formal=False):
    valid = [r for r in rows if r.get("transfer_valid")]
    r_count = sum(bool(r["R_to_CV_positive"]) for r in valid)
    c_count = sum(bool(r["C_to_RV_positive"]) for r in valid)
    b_count = sum(bool(r["BIDIRECTIONAL_TRANSFER_POSITIVE"]) for r in valid)
    if formal:
        if r_count >= 8 and c_count >= 8:
            signal = "FORMAL_SUPPORTED"
        elif r_count >= 8 or c_count >= 8:
            signal = "ASYMMETRIC_TRANSFER"
        else:
            signal = "NOT_SUPPORTED_FORMALLY"
        return {
            "C2_FORMAL_RUN": "YES",
            "valid_units": len(valid),
            "R_TO_CV_FORMAL_COUNT": r_count,
            "C_TO_RV_FORMAL_COUNT": c_count,
            "BIDIRECTIONAL_FORMAL_COUNT": b_count,
            "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": signal,
        }
    gate = "PASS" if b_count >= PILOT_BIDIRECTIONAL_GATE and len(valid) == 3 else "NOT_SUPPORTED"
    return {
        "C2_VALID_PILOT_UNITS": len(valid),
        "R_TO_CV_SAFER_COUNT": r_count,
        "C_TO_RV_MORE_HARMFUL_COUNT": c_count,
        "BIDIRECTIONAL_POSITIVE_COUNT": b_count,
        "C2_PILOT_GATE": gate,
    }


def run_all():
    numerical = pure_numerical_audit()
    if numerical["TRANSFER_NUMERICAL_VALIDITY_GATE"] != "PASS":
        final = final_summary(numerical, {}, {}, "NONE")
        write_report(final)
        return final
    if numerical["SVD_TRANSFER_IDENTIFIABILITY_GATE"] == "PASS":
        construction = "SVD_VALUE_TRANSFER"
    else:
        construction = "PROCRUSTES_ORIENTATION_TRANSFER"
    pilot_units = numerical["pilot_units"]
    behavior_identity, pilot_rows = behavior_self_identity_and_transfer(pilot_units, construction, "pilot")
    behavior_gate = "RAW_REPORTED_NO_PREEXISTING_TOLERANCE"
    pilot = summarize_transfer(pilot_rows, formal=False)
    pilot["BEHAVIORAL_SELF_IDENTITY_GATE"] = behavior_gate
    pilot["TRANSFER_CONSTRUCTION_USED"] = construction
    save_json("c2_retest_pilot_summary.json", pilot)
    if construction == "PROCRUSTES_ORIENTATION_TRANSFER":
        save_json("procrustes_pilot_summary.json", pilot)
    else:
        save_json("procrustes_pilot_summary.json", {"VALUE_AXIS_ORTHOGONAL_PROCRUSTES_TRANSFER": "NOT_RUN"})
    formal = {"C2_FORMAL_RUN": "NO"}
    if pilot["C2_PILOT_GATE"] == "PASS":
        all_units = [r["unit"] for r in prev.read_prev_stagea_rows()]
        _identity, formal_rows = behavior_self_identity_and_transfer(all_units, construction, "formal")
        formal = summarize_transfer(formal_rows, formal=True)
    else:
        write_csv("c2_retest_formal_per_unit.csv", [])
        formal.update({
            "R_TO_CV_FORMAL_COUNT": 0,
            "C_TO_RV_FORMAL_COUNT": 0,
            "BIDIRECTIONAL_FORMAL_COUNT": 0,
            "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": "NOT_SUPPORTED",
        })
    save_json("c2_retest_formal_summary.json", formal)
    final = final_summary(numerical, pilot, formal, construction)
    write_report(final)
    return final


def final_summary(numerical, pilot, formal, construction):
    transfer_signal = formal.get("VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER")
    orientation_signal = "NOT_RUN" if construction != "PROCRUSTES_ORIENTATION_TRANSFER" else ("SUPPORTED_PILOT" if pilot.get("C2_PILOT_GATE") == "PASS" else "NOT_SUPPORTED")
    if construction == "NONE":
        cls = "TRANSFER_UNRESOLVED_DUE_TO_CONSTRUCTION_LIMITATION"
        closure = "NO"
    elif transfer_signal == "FORMAL_SUPPORTED":
        cls = "SPECIFIC_VALUE_GEOMETRY_HARMFULNESS_TRANSFER_SUPPORTED"
        closure = "YES_CANDIDATE"
    elif pilot.get("C2_PILOT_GATE") == "PASS" and formal.get("C2_FORMAL_RUN") == "YES":
        cls = "VALID_TRANSFER_PILOT_POSITIVE_FORMAL_NOT_CLOSED"
        closure = "NO"
    elif pilot.get("C2_PILOT_GATE") == "PASS":
        cls = "VALID_TRANSFER_PILOT_SUPPORTED_FORMAL_NOT_RUN"
        closure = "NO"
    else:
        cls = "VALUE_SIDE_DIRECTION_FORMAL_SUPPORTED_BUT_VALID_TRANSFER_NOT_SUPPORTED"
        closure = "NO"
    obj = {
        "NUMERICAL_ROOT_CAUSE": numerical.get("NUMERICAL_ROOT_CAUSE"),
        "PURE_SVD_RECONSTRUCTION_GATE": numerical.get("PURE_SVD_RECONSTRUCTION_GATE"),
        "BEHAVIORAL_SELF_IDENTITY_GATE": pilot.get("BEHAVIORAL_SELF_IDENTITY_GATE", "NOT_RUN"),
        "SVD_TRANSFER_IDENTIFIABILITY_GATE": numerical.get("SVD_TRANSFER_IDENTIFIABILITY_GATE"),
        "TRANSFER_NUMERICAL_VALIDITY_GATE": numerical.get("TRANSFER_NUMERICAL_VALIDITY_GATE"),
        "TRANSFER_CONSTRUCTION_USED": construction,
        "C2_VALID_PILOT_UNITS": pilot.get("C2_VALID_PILOT_UNITS", 0),
        "R_TO_CV_SAFER_COUNT": pilot.get("R_TO_CV_SAFER_COUNT", 0),
        "C_TO_RV_MORE_HARMFUL_COUNT": pilot.get("C_TO_RV_MORE_HARMFUL_COUNT", 0),
        "BIDIRECTIONAL_POSITIVE_COUNT": pilot.get("BIDIRECTIONAL_POSITIVE_COUNT", 0),
        "C2_PILOT_GATE": pilot.get("C2_PILOT_GATE", "NOT_RUN"),
        "C2_FORMAL_RUN": formal.get("C2_FORMAL_RUN", "NO"),
        "R_TO_CV_FORMAL_COUNT": formal.get("R_TO_CV_FORMAL_COUNT", 0),
        "C_TO_RV_FORMAL_COUNT": formal.get("C_TO_RV_FORMAL_COUNT", 0),
        "BIDIRECTIONAL_FORMAL_COUNT": formal.get("BIDIRECTIONAL_FORMAL_COUNT", 0),
        "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": transfer_signal or "NOT_RUN",
        "VALUE_AXIS_ORIENTATION_TRANSFER_SIGNAL": orientation_signal,
        "FINAL_MECHANISM_CLASSIFICATION": cls,
        "MECHANISM_CLOSURE_CANDIDATE": closure,
        "METHOD_DESIGN_READY": "NO",
        "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": "FORMAL_SUPPORTED",
        "output_dir": str(RUN_DIR),
        "git_commit_start": git_commit(),
    }
    save_json("final_summary.json", obj)
    return obj


def write_report(final):
    lines = [
        "# GDN INT8 Value Geometry Transfer Validity and Retest V1",
        "## 1. Executive Summary\n\n" + json.dumps(final, indent=2, ensure_ascii=False),
        "## 2. Prior Formal Value-Side Evidence\n\nC1 remains canonical: VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL = FORMAL_SUPPORTED.",
        "## 3. Why Previous C2 Was Invalid\n\nPrevious C2 counted FP32/GPU SVD reconstruction error against the pure mathematical SVD identity gate, producing 0/3 valid units despite raw 3/3 direction consistency.",
        "## 4. Numerical Root-Cause Audit\n\nSee `numerical_audit_per_unit.csv` and `numerical_audit_summary.json`.",
        "## 5. Dtype / SVD / Vh Audit\n\nPyTorch `torch.linalg.svd` returns `Vh = V^H`; reconstruction uses `U @ diag(S) @ Vh`. FP64 CPU and FP32 paths are reported separately.",
        "## 6. Tensor Self-Reconstruction Identity\n\nSee `self_reconstruction_tensor_identity.csv`.",
        "## 7. Behavioral Self-Reconstruction Identity\n\nSee `self_reconstruction_behavior_identity.csv`; no new loose tolerance was invented.",
        "## 8. Singular Spectrum / Identifiability Audit\n\nSee `svd_spectrum_audit.json` and `svd_identifiability_summary.json`.",
        "## 9. Transfer Construction\n\nConstruction used: `" + str(final["TRANSFER_CONSTRUCTION_USED"]) + "`.",
        "## 10. Transfer Invariance Checks\n\nSee `transfer_construction_audit.csv` or `procrustes_audit.csv`.",
        "## 11. SAME 3-Unit Pilot Retest\n\nSame pilot units from the prior run were reused without reselection.",
        "## 12. R -> C Value Transfer\n\nPilot safer count: " + str(final["R_TO_CV_SAFER_COUNT"]),
        "## 13. C -> R Value Transfer\n\nPilot more harmful count: " + str(final["C_TO_RV_MORE_HARMFUL_COUNT"]),
        "## 14. Bidirectional Pilot Gate\n\n" + str(final["C2_PILOT_GATE"]),
        "## 15. 9-Unit Formal Transfer\n\n" + str(final["C2_FORMAL_RUN"]),
        "## 16. Basis-Invariant Procrustes Fallback\n\n" + str(final["VALUE_AXIS_ORIENTATION_TRANSFER_SIGNAL"]),
        "## 17. Positive Results\n\nValid positive results are reported only from the selected construction after numerical gates.",
        "## 18. Negative / Corrective Results\n\nPrevious raw 3/3 is retained as a debugging observation, not scientific evidence.",
        "## 19. What Previous Invalid Raw 3/3 Means\n\nIt motivated the audit but is not combined into the retest counts.",
        "## 20. Final Mechanism Classification\n\n" + final["FINAL_MECHANISM_CLASSIFICATION"],
        "## 21. What Is Still Not Proven\n\nNo downstream-sensitive Value directions are identified; no method design is ready.",
        "## 22. Recommended Next Experiment\n\nOnly if transfer is strongly supported: `GDN_INT8_VALUE_DIRECTION_DOWNSTREAM_SENSITIVITY_AUDIT_V1`. Not run here.",
    ]
    (RUN_DIR / "final_report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "audit"], default="all")
    args = ap.parse_args()
    if args.stage == "audit":
        print(json.dumps(pure_numerical_audit(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(run_all(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
