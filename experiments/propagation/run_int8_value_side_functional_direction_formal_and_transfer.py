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
PREV_RUN = ROOT / "runs" / "gdn_int8_same_norm_functional_geometry_audit_v1"
RUN_DIR = ROOT / "runs" / "gdn_int8_value_side_functional_direction_formal_and_transfer_v1"
TASK = "GDN_INT8_VALUE_SIDE_FUNCTIONAL_DIRECTION_FORMAL_AND_TRANSFER_V1"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_same_norm_functional_geometry_audit as fg

PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"
EPS = 1e-12
HORIZON = 128
ROT_SEEDS = [1101, 1102, 1103, 1104]
SOURCE_NORM_TOL = 1e-4
SV_TOL = 1e-4
PERSISTENCE_RATIO_BOUNDS = [0.80, 1.25]
OBSERVABILITY_RATIO_BOUNDS = [0.50, 2.00]
SVD_RECON_TOL = 1e-5
SVD_NEAR_DEGENERATE_REL_GAP = 1e-4


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


def ratio(a, b):
    return float(a) / (float(b) + EPS) if finite(a) and finite(b) else None


def auc_sq(xs):
    return sum(float(x) ** 2 for x in xs if finite(x))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_prev_stagea_rows():
    with (PREV_RUN / "stageA_per_unit.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["t0", "P_R", "P_C", "O_R", "O_C", "eta_R", "eta_C", "KL_R", "KL_C"]:
            r[k] = int(r[k]) if k == "t0" else float(r[k])
        r["unit_id"] = r["unit"]
        r["original_gap"] = r["KL_R"] - r["KL_C"]
    return rows


def setup_config():
    cfg = {
        "task": TASK,
        "timestamp": now(),
        "git_commit_start": git_commit(),
        "previous_task": "GDN_INT8_SAME_NORM_FUNCTIONAL_GEOMETRY_AUDIT_V1",
        "previous_commit": "25e97ad3fa827c13c8579f851c31c94cf5156013",
        "previous_run_dir": str(PREV_RUN),
        "output_dir": str(RUN_DIR),
        "tensor_semantics": {"state": "[B,H,K,V]", "row": "Key axis", "column": "Value axis"},
        "rotation": {
            "axis": "V",
            "seeds": ROT_SEEDS,
            "construction": "seeded Gaussian -> QR -> deterministic diagonal-sign convention",
            "same_Q_for_R_and_C": True,
        },
        "horizon": HORIZON,
        "invariance_thresholds": {
            "source_norm_rel_error": SOURCE_NORM_TOL,
            "singular_value_rel_error": SV_TOL,
            "persistence_rot_over_original_bounds": PERSISTENCE_RATIO_BOUNDS,
            "observability_rot_over_original_bounds": OBSERVABILITY_RATIO_BOUNDS,
        },
        "c1_gate": {
            "formal_supported": "UNIT_V_RESCUE_POSITIVE >= 8/9, median unit rescue > 0, no systematic invariance failure",
            "partial": "6/9 or 7/9 positive units",
            "not_supported": "<=5/9 positive units",
        },
        "c2_gate": {
            "pilot_pass": "bidirectional positive >= 2/3 and no serious numerical confound",
            "formal_supported": "R_to_CV safer >= 8/9 and C_to_RV more harmful >= 8/9",
        },
    }
    save_json("config.json", cfg)
    return cfg


def stage0():
    cfg = setup_config()
    prev_summary = load_json(PREV_RUN / "final_summary.json")
    prev_stage0 = load_json(PREV_RUN / "stage0_summary.json")
    rows = read_prev_stagea_rows()
    norm_errors = []
    for u in load_json(PREV_RUN / "stageA_fresh_replay_units.json")["per_unit"]:
        meta = u["same_norm_meta"]
        norm_errors.append(max(float(meta.get("norm_match_error_R_to_C", 1.0)), float(meta.get("norm_match_error_C_to_R", 1.0))))
    import torch
    q = fg.orthogonal(torch, 128, ROT_SEEDS[0], "cpu")
    orth_err = float(torch.linalg.matrix_norm(q.T @ q - torch.eye(128)).item())
    gates = {
        "PREVIOUS_STAGE0_PASS": prev_stage0.get("STAGE0") == "PASS",
        "CANONICAL_9_UNITS_LOADED": len(rows) == 9,
        "ORIGINAL_R_GT_C_9_OF_9": sum(r["KL_R"] > r["KL_C"] for r in rows) == 9,
        "ORIGINAL_R_C_NORM_MATCH": max(norm_errors) <= SOURCE_NORM_TOL,
        "PREVIOUS_OBSERVABILITY_GATE_PASS": prev_summary.get("OBSERVABILITY_GATE") == "PASS",
        "PREVIOUS_V_PILOT_SUPPORTED": prev_summary.get("VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL") == "SUPPORTED",
        "V_ROTATION_ORTHOGONALITY_SMOKE": orth_err <= 1e-5,
        "INJECTION_HOOK_REUSED_FROM_PREVIOUS_EXPERIMENT": True,
    }
    summary = {
        "STAGE0": "PASS" if all(gates.values()) else "FAIL",
        "timestamp": now(),
        "config": cfg,
        "gates": gates,
        "orthogonality_error_seed0": orth_err,
        "max_original_norm_rel_error": max(norm_errors),
        "canonical_units": [r["unit"] for r in rows],
    }
    save_json("stage0_summary.json", summary)
    return summary


def build_base_injections(torch, model, tokenizer, e2e, pm, t0):
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, int(t0))
    try:
        inj, meta = normswap.build_injections(torch, fp_past)
        fp_states = {layer: p1.get_state(fp_past, layer).detach().float().clone() for layer in frozen.GDN_LAYERS}
        return fp_past, ids, cont, collector, inj, meta, fp_states
    except Exception:
        collector["close"]()
        raise


def propagate_metrics(torch, model, tokenizer, e2e, pm, t0, inj):
    fp_past, ids, cont, collector, _base_inj, _meta, fp_states = build_base_injections(torch, model, tokenizer, e2e, pm, t0)
    try:
        cond_states = {
            PRIMARY_R: {layer: p1.get_state(fp_past, layer).detach().float() + inj[PRIMARY_R][layer] for layer in frozen.GDN_LAYERS},
            PRIMARY_C: {layer: p1.get_state(fp_past, layer).detach().float() + inj[PRIMARY_C][layer] for layer in frozen.GDN_LAYERS},
        }
        source_norms = {
            PRIMARY_R: math.sqrt(sum(float(torch.sum(E.detach().double() * E.detach().double()).item()) for E in inj[PRIMARY_R].values())),
            PRIMARY_C: math.sqrt(sum(float(torch.sum(E.detach().double() * E.detach().double()).item()) for E in inj[PRIMARY_C].values())),
        }
        curves = {PRIMARY_R: {"state": [], "query": []}, PRIMARY_C: {"state": [], "query": []}}
        max_identity = 0.0
        valid_h = min(HORIZON, len(cont) - int(t0))
        past = fp_past
        with torch.inference_mode():
            for off in range(1, valid_h + 1):
                nxt = torch.tensor([[cont[int(t0) + off - 1]]], dtype=ids.dtype, device=ids.device)
                _out, past, records = frozen.driver_step(torch, model, nxt, None, past, collector)
                per_token = {PRIMARY_R: {"state": 0.0, "query": 0.0}, PRIMARY_C: {"state": 0.0, "query": 0.0}}
                for layer in frozen.GDN_LAYERS:
                    rec = records[layer]
                    out = fg.corrected_replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[PRIMARY_R][layer], cond_states[PRIMARY_C][layer]])
                    max_identity = max(max_identity, max(float(x) for x in out["identity"].values()))
                    for ci, cond in enumerate([PRIMARY_R, PRIMARY_C]):
                        cond_states[cond][layer] = out["next_state"][ci:ci + 1]
                        for name, key in [("state", "E_after_update"), ("query", "core_readout_error")]:
                            ten = out[key][ci:ci + 1].detach().float()
                            per_token[cond][name] += float(torch.sum(ten.double() * ten.double()).item())
                    fp_states[layer] = rec["final_state"].detach().float()
                for cond in [PRIMARY_R, PRIMARY_C]:
                    curves[cond]["state"].append(math.sqrt(per_token[cond]["state"]))
                    curves[cond]["query"].append(math.sqrt(per_token[cond]["query"]))
        out = {}
        for cond in [PRIMARY_R, PRIMARY_C]:
            norm2 = source_norms[cond] ** 2 + EPS
            out[cond] = {
                "source_norm": source_norms[cond],
                "P_state": auc_sq(curves[cond]["state"]) / norm2,
                "O_q": auc_sq(curves[cond]["query"]) / norm2,
                "identity_max_relative_error": max_identity,
            }
        return out
    finally:
        collector["close"]()


def c1_tasks():
    tasks = []
    for r in read_prev_stagea_rows():
        for seed in ROT_SEEDS:
            tasks.append({"unit": r["unit"], "rotation_seed": seed})
    return tasks


def c1_worker(worker_id, indices):
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    rows = read_prev_stagea_rows()
    row_by_unit = {r["unit"]: r for r in rows}
    prompt_map = {p["problem_id"]: p for p in normswap.selected_prompts(3)}
    tasks = c1_tasks()
    cp_path = RUN_DIR / f"stageC1_worker_{worker_id}.json"
    cp = load_json(cp_path) if cp_path.exists() else {"worker_id": worker_id, "assigned_indices": indices, "completed": {}, "failed": {}}
    base_cache = {}
    for idx in indices:
        task = tasks[idx]
        key = f"{idx}:{task['unit']}:{task['rotation_seed']}"
        if key in cp["completed"]:
            continue
        print(f"[{now()}] C1 {worker_id} {key}", flush=True)
        try:
            row = row_by_unit[task["unit"]]
            pm = prompt_map[row["prompt_id"]]
            if task["unit"] not in base_cache:
                _fp, _ids, _cont, collector, inj, meta, _fp_states = build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
                collector["close"]()
                base_cache[task["unit"]] = (inj, meta, next(model.parameters()).device)
            inj, _meta, device = base_cache[task["unit"]]
            rot, audit = fg.rotate_errors(torch, inj, "V", int(task["rotation_seed"]), device)
            m = propagate_metrics(torch, model, tokenizer, e2e, pm, row["t0"], rot)
            kl = fg.full_model_kl(torch, model, tokenizer, e2e, pm, int(row["t0"]), rot[PRIMARY_R], rot[PRIMARY_C], horizon=HORIZON)
            g0 = row["original_gap"]
            gr = kl["KL_R"] - kl["KL_C"]
            rp_r = ratio(m[PRIMARY_R]["P_state"], row["P_R"])
            rp_c = ratio(m[PRIMARY_C]["P_state"], row["P_C"])
            ro_r = ratio(m[PRIMARY_R]["O_q"], row["O_R"])
            ro_c = ratio(m[PRIMARY_C]["O_q"], row["O_C"])
            src_ok = audit["max_norm_relative_error"] <= SOURCE_NORM_TOL
            sv_ok = audit["sample_singular_value_relative_error"] <= SV_TOL
            p_ok = PERSISTENCE_RATIO_BOUNDS[0] <= rp_r <= PERSISTENCE_RATIO_BOUNDS[1] and PERSISTENCE_RATIO_BOUNDS[0] <= rp_c <= PERSISTENCE_RATIO_BOUNDS[1]
            o_ok = OBSERVABILITY_RATIO_BOUNDS[0] <= ro_r <= OBSERVABILITY_RATIO_BOUNDS[1] and OBSERVABILITY_RATIO_BOUNDS[0] <= ro_c <= OBSERVABILITY_RATIO_BOUNDS[1]
            if not (src_ok and sv_ok):
                status = "INVALID_ROTATION_RUN"
            elif not (p_ok and o_ok):
                status = "VALUE_ROTATION_CONFOUNDED"
            else:
                status = "PASS"
            cp["completed"][key] = {
                "unit_id": row["unit"],
                "prompt_id": row["prompt_id"],
                "layer": "all_gdn_layers",
                "head": "all_heads",
                "t0": row["t0"],
                "rotation_seed": int(task["rotation_seed"]),
                "R_source_norm": m[PRIMARY_R]["source_norm"],
                "C_source_norm": m[PRIMARY_C]["source_norm"],
                "R_source_norm_rel_error": audit["max_norm_relative_error"],
                "C_source_norm_rel_error": audit["max_norm_relative_error"],
                "R_sv_rel_error": audit["sample_singular_value_relative_error"],
                "C_sv_rel_error": audit["sample_singular_value_relative_error"],
                "R_P_state_original": row["P_R"],
                "R_P_state_rot": m[PRIMARY_R]["P_state"],
                "C_P_state_original": row["P_C"],
                "C_P_state_rot": m[PRIMARY_C]["P_state"],
                "R_Oq_original": row["O_R"],
                "R_Oq_rot": m[PRIMARY_R]["O_q"],
                "C_Oq_original": row["O_C"],
                "C_Oq_rot": m[PRIMARY_C]["O_q"],
                "KL_R_original": row["KL_R"],
                "KL_C_original": row["KL_C"],
                "KL_R_rot": kl["KL_R"],
                "KL_C_rot": kl["KL_C"],
                "G_original": g0,
                "G_rot": gr,
                "gap_change": gr - g0,
                "rescue": 1.0 - gr / (g0 + EPS) if abs(g0) > 1e-8 else None,
                "sign_inverted": g0 > 0 and gr < 0,
                "SOURCE_NORM_INVARIANT": src_ok,
                "SINGULAR_VALUES_INVARIANT": sv_ok,
                "PERSISTENCE_APPROX_INVARIANT": p_ok,
                "QUERY_OBSERVABILITY_APPROX_INVARIANT": o_ok,
                "run_valid": src_ok and sv_ok,
                "invariance_status": status,
                "timestamp": now(),
            }
            save_json(f"stageC1_worker_{worker_id}.json", cp)
        except Exception as exc:
            cp["failed"][key] = {"error": repr(exc), "timestamp": now()}
            save_json(f"stageC1_worker_{worker_id}.json", cp)
            raise


def launch_c1():
    tasks = c1_tasks()
    shards = {i: [] for i in range(4)}
    for idx in range(len(tasks)):
        shards[idx % 4].append(idx)
    launcher = EXP / "launch_value_side_c1_4gpu.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {EXP}", "export PYTHONUNBUFFERED=1"]
    for gpu in range(4):
        idxs = ",".join(str(i) for i in shards[gpu])
        lines.append(f"CUDA_VISIBLE_DEVICES={gpu} /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 {Path(__file__).name} --stage c1-worker --worker-id gpu{gpu} --task-indices {idxs} > {RUN_DIR}/stageC1_gpu{gpu}.log 2>&1 &")
        lines.append(f"echo $! > {RUN_DIR}/stageC1_gpu{gpu}.pid")
    lines.append("wait")
    lines.append(f"/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 {Path(__file__).name} --stage c1-merge > {RUN_DIR}/stageC1_merge.log 2>&1")
    launcher.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(launcher, 0o755)
    print(launcher)


def c1_merge():
    all_rows = []
    failed = {}
    for wid in ["gpu0", "gpu1", "gpu2", "gpu3"]:
        p = RUN_DIR / f"stageC1_worker_{wid}.json"
        if not p.exists():
            continue
        d = load_json(p)
        all_rows.extend(d.get("completed", {}).values())
        failed.update(d.get("failed", {}))
    all_rows = sorted(all_rows, key=lambda r: (r["unit_id"], int(r["rotation_seed"])))
    write_csv(RUN_DIR / "stageC1_per_run.csv", all_rows)
    unit_rows = []
    for unit in sorted(set(r["unit_id"] for r in all_rows)):
        rs = [r for r in all_rows if r["unit_id"] == unit]
        valid = [r for r in rs if r["run_valid"]]
        original_gap = valid[0]["G_original"] if valid else rs[0]["G_original"]
        median_rot_gap = med([r["G_rot"] for r in valid])
        unit_rows.append({
            "unit_id": unit,
            "original_gap": original_gap,
            "median_rotated_gap": median_rot_gap,
            "median_rescue": med([r["rescue"] for r in valid]),
            "gap_decreased_seeds": sum(r["G_rot"] < r["G_original"] for r in valid),
            "sign_inverted_seeds": sum(r["sign_inverted"] for r in valid),
            "median_P_state_change_R": med([ratio(r["R_P_state_rot"], r["R_P_state_original"]) for r in valid]),
            "median_P_state_change_C": med([ratio(r["C_P_state_rot"], r["C_P_state_original"]) for r in valid]),
            "median_Oq_change_R": med([ratio(r["R_Oq_rot"], r["R_Oq_original"]) for r in valid]),
            "median_Oq_change_C": med([ratio(r["C_Oq_rot"], r["C_Oq_original"]) for r in valid]),
            "UNIT_V_RESCUE_POSITIVE": median_rot_gap is not None and median_rot_gap < original_gap,
            "UNIT_VALID": len(valid) == 4,
            "confounded_runs": sum(r["invariance_status"] == "VALUE_ROTATION_CONFOUNDED" for r in rs),
            "invalid_runs": sum(r["invariance_status"] == "INVALID_ROTATION_RUN" for r in rs),
        })
    write_csv(RUN_DIR / "stageC1_per_unit.csv", unit_rows)
    valid_runs = [r for r in all_rows if r["run_valid"]]
    valid_units = [u for u in unit_rows if u["UNIT_VALID"]]
    pos_units = sum(u["UNIT_V_RESCUE_POSITIVE"] for u in valid_units)
    confounded = sum(r["invariance_status"] == "VALUE_ROTATION_CONFOUNDED" for r in all_rows)
    invalid = sum(r["invariance_status"] == "INVALID_ROTATION_RUN" for r in all_rows)
    systematic_fail = invalid > 0 or confounded > len(all_rows) // 3
    if pos_units >= 8 and (med([u["median_rescue"] for u in valid_units]) or 0) > 0 and not systematic_fail:
        signal = "FORMAL_SUPPORTED"
    elif pos_units >= 6:
        signal = "PARTIAL_OR_INCONCLUSIVE"
    else:
        signal = "NOT_SUPPORTED_FORMALLY"
    summary = {
        "STAGE_C1_RUN": "YES",
        "expected_runs": len(c1_tasks()),
        "completed_runs": len(all_rows),
        "failed": failed,
        "C1_VALID_RUNS": len(valid_runs),
        "C1_VALID_UNITS": len(valid_units),
        "V_ROTATION_GAP_DECREASE_RUNS": sum(r["G_rot"] < r["G_original"] for r in valid_runs),
        "V_ROTATION_SIGN_INVERSIONS": sum(r["sign_inverted"] for r in valid_runs),
        "V_ROTATION_POSITIVE_UNITS": pos_units,
        "C1_MEDIAN_RESCUE": med([u["median_rescue"] for u in valid_units]),
        "confounded_runs": confounded,
        "invalid_runs": invalid,
        "systematic_invariance_failure": systematic_fail,
        "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": signal,
    }
    save_json("stageC1_summary.json", summary)
    return summary


def base_inj_for_unit(torch, model, tokenizer, e2e, row, prompt_map):
    pm = prompt_map[row["prompt_id"]]
    fp_past, _ids, _cont, collector, inj, meta, _fp_states = build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
    collector["close"]()
    return pm, inj, meta, next(model.parameters()).device


def svd_transfer(torch, inj, device):
    out_r = {}
    out_c = {}
    audit = {
        "max_reconstruction_error_R": 0.0,
        "max_reconstruction_error_C": 0.0,
        "max_norm_rel_error_R_to_CV": 0.0,
        "max_norm_rel_error_C_to_RV": 0.0,
        "max_sv_rel_error_R_to_CV": 0.0,
        "max_sv_rel_error_C_to_RV": 0.0,
        "near_degenerate_fraction_R": 0.0,
        "near_degenerate_fraction_C": 0.0,
        "svd_matrices": 0,
    }
    deg_r = deg_c = gaps_r = gaps_c = 0
    for layer in inj[PRIMARY_R]:
        ER = inj[PRIMARY_R][layer].detach().float()
        EC = inj[PRIMARY_C][layer].detach().float()
        TR = torch.empty_like(ER)
        TC = torch.empty_like(EC)
        for h in range(ER.shape[1]):
            er = ER[0, h]
            ec = EC[0, h]
            ur, sr, vhr = torch.linalg.svd(er, full_matrices=True)
            uc, sc, vhc = torch.linalg.svd(ec, full_matrices=True)
            rec_r = (ur * sr.unsqueeze(0)) @ vhr
            rec_c = (uc * sc.unsqueeze(0)) @ vhc
            tr = (ur * sr.unsqueeze(0)) @ vhc
            tc = (uc * sc.unsqueeze(0)) @ vhr
            TR[0, h] = tr
            TC[0, h] = tc
            audit["max_reconstruction_error_R"] = max(audit["max_reconstruction_error_R"], float(torch.linalg.vector_norm(rec_r - er).item() / (torch.linalg.vector_norm(er).item() + EPS)))
            audit["max_reconstruction_error_C"] = max(audit["max_reconstruction_error_C"], float(torch.linalg.vector_norm(rec_c - ec).item() / (torch.linalg.vector_norm(ec).item() + EPS)))
            audit["max_norm_rel_error_R_to_CV"] = max(audit["max_norm_rel_error_R_to_CV"], abs(float(torch.linalg.vector_norm(tr).item() - torch.linalg.vector_norm(er).item())) / (float(torch.linalg.vector_norm(er).item()) + EPS))
            audit["max_norm_rel_error_C_to_RV"] = max(audit["max_norm_rel_error_C_to_RV"], abs(float(torch.linalg.vector_norm(tc).item() - torch.linalg.vector_norm(ec).item())) / (float(torch.linalg.vector_norm(ec).item()) + EPS))
            audit["max_sv_rel_error_R_to_CV"] = max(audit["max_sv_rel_error_R_to_CV"], float(torch.linalg.vector_norm(torch.linalg.svdvals(tr) - sr).item() / (torch.linalg.vector_norm(sr).item() + EPS)))
            audit["max_sv_rel_error_C_to_RV"] = max(audit["max_sv_rel_error_C_to_RV"], float(torch.linalg.vector_norm(torch.linalg.svdvals(tc) - sc).item() / (torch.linalg.vector_norm(sc).item() + EPS)))
            for svals, which in [(sr, "R"), (sc, "C")]:
                rel_gaps = torch.abs(svals[:-1] - svals[1:]) / (torch.maximum(svals[:-1], svals[1:]) + EPS)
                n = int(rel_gaps.numel())
                if which == "R":
                    gaps_r += n
                    deg_r += int(torch.sum(rel_gaps < SVD_NEAR_DEGENERATE_REL_GAP).item())
                else:
                    gaps_c += n
                    deg_c += int(torch.sum(rel_gaps < SVD_NEAR_DEGENERATE_REL_GAP).item())
            audit["svd_matrices"] += 2
        out_r[layer] = TR.to(device)
        out_c[layer] = TC.to(device)
    audit["near_degenerate_fraction_R"] = deg_r / gaps_r if gaps_r else 0.0
    audit["near_degenerate_fraction_C"] = deg_c / gaps_c if gaps_c else 0.0
    audit["degenerate_subspace_flag"] = audit["near_degenerate_fraction_R"] > 0.10 or audit["near_degenerate_fraction_C"] > 0.10
    return out_r, out_c, audit


def c2_run(units, label):
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    rows = read_prev_stagea_rows()
    selected = [r for r in rows if r["unit"] in units]
    prompt_map = {p["problem_id"]: p for p in normswap.selected_prompts(3)}
    out_rows = []
    audits = {}
    for row in selected:
        print(f"[{now()}] C2 {label} {row['unit']}", flush=True)
        pm, inj, _meta, device = base_inj_for_unit(torch, model, tokenizer, e2e, row, prompt_map)
        r_to_cv, c_to_rv, audit = svd_transfer(torch, inj, device)
        self_kl = fg.full_model_kl(torch, model, tokenizer, e2e, pm, int(row["t0"]), inj[PRIMARY_R], inj[PRIMARY_C], horizon=HORIZON)
        transfer_kl = fg.full_model_kl(torch, model, tokenizer, e2e, pm, int(row["t0"]), r_to_cv, c_to_rv, horizon=HORIZON)
        self_gap_ok = abs((self_kl["KL_R"] - row["KL_R"]) / (row["KL_R"] + EPS)) <= 0.05 and abs((self_kl["KL_C"] - row["KL_C"]) / (row["KL_C"] + EPS)) <= 0.05
        recon_ok = audit["max_reconstruction_error_R"] <= SVD_RECON_TOL and audit["max_reconstruction_error_C"] <= SVD_RECON_TOL
        r_safety = row["KL_R"] - transfer_kl["KL_R"]
        c_harm = transfer_kl["KL_C"] - row["KL_C"]
        transfer_valid = recon_ok and self_gap_ok
        row_out = {
            "unit_id": row["unit"],
            "KL_R": row["KL_R"],
            "KL_C": row["KL_C"],
            "KL_R_self_reconstruction": self_kl["KL_R"],
            "KL_C_self_reconstruction": self_kl["KL_C"],
            "KL_R_to_CV": transfer_kl["KL_R"],
            "KL_C_to_RV": transfer_kl["KL_C"],
            "R_safety_transfer": r_safety,
            "C_harm_transfer": c_harm,
            "R_to_CV_positive": r_safety > 0,
            "C_to_RV_positive": c_harm > 0,
            "BIDIRECTIONAL_TRANSFER_POSITIVE": r_safety > 0 and c_harm > 0,
            "SVD_reconstruction_error_R": audit["max_reconstruction_error_R"],
            "SVD_reconstruction_error_C": audit["max_reconstruction_error_C"],
            "degenerate_subspace_flag": audit["degenerate_subspace_flag"],
            "transfer_valid": transfer_valid,
        }
        out_rows.append(row_out)
        audits[row["unit"]] = audit
    save_json(f"stageC2_{label}_svd_audit.json", audits)
    return out_rows


def c2_pipeline(c1_summary):
    if c1_summary["VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL"] != "FORMAL_SUPPORTED":
        summary = {
            "STAGE_C2_RUN": "NO",
            "C2_PILOT_STATUS": "NOT_RUN_C1_NOT_FORMAL_SUPPORTED",
            "C2_FORMAL_RUN": "NO",
            "R_TO_CV_SAFETY_TRANSFER": "NOT_RUN",
            "C_TO_RV_HARM_TRANSFER": "NOT_RUN",
            "BIDIRECTIONAL_TRANSFER_POSITIVE_UNITS": 0,
            "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": "NOT_RUN",
        }
        save_json("stageC2_pilot_summary.json", summary)
        return summary
    rows = read_prev_stagea_rows()
    sorted_rows = sorted(rows, key=lambda r: r["original_gap"])
    pilot = [sorted_rows[0]["unit"], sorted_rows[len(sorted_rows) // 2]["unit"], sorted_rows[-1]["unit"]]
    save_json("stageC2_pilot_config.json", {"selection_rule": "low/median/high original KL_R-KL_C from canonical 9 units before transfer results", "units": pilot})
    pilot_rows = c2_run(pilot, "pilot")
    write_csv(RUN_DIR / "stageC2_pilot_per_run.csv", pilot_rows)
    valid = [r for r in pilot_rows if r["transfer_valid"]]
    pilot_bidir = sum(r["BIDIRECTIONAL_TRANSFER_POSITIVE"] for r in valid)
    pilot_summary = {
        "STAGE_C2_RUN": "YES",
        "C2_PILOT_STATUS": "PASS" if pilot_bidir >= 2 and len(valid) == 3 else "NOT_SUPPORTED_OR_INCONCLUSIVE",
        "valid_units": len(valid),
        "R_to_CV_safer_count": sum(r["R_to_CV_positive"] for r in valid),
        "C_to_RV_more_harmful_count": sum(r["C_to_RV_positive"] for r in valid),
        "bidirectional_positive_count": pilot_bidir,
        "units": pilot,
    }
    save_json("stageC2_pilot_summary.json", pilot_summary)
    if pilot_summary["C2_PILOT_STATUS"] != "PASS":
        pilot_summary.update({
            "C2_FORMAL_RUN": "NO",
            "R_TO_CV_SAFETY_TRANSFER": "PILOT_ONLY",
            "C_TO_RV_HARM_TRANSFER": "PILOT_ONLY",
            "BIDIRECTIONAL_TRANSFER_POSITIVE_UNITS": pilot_bidir,
            "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": "NOT_SUPPORTED_OR_INCONCLUSIVE",
        })
        save_json("stageC2_formal_summary.json", pilot_summary)
        return pilot_summary
    formal_units = [r["unit"] for r in rows]
    formal_rows = c2_run(formal_units, "formal")
    write_csv(RUN_DIR / "stageC2_formal_per_unit.csv", formal_rows)
    valid_f = [r for r in formal_rows if r["transfer_valid"]]
    r_count = sum(r["R_to_CV_positive"] for r in valid_f)
    c_count = sum(r["C_to_RV_positive"] for r in valid_f)
    b_count = sum(r["BIDIRECTIONAL_TRANSFER_POSITIVE"] for r in valid_f)
    if r_count >= 8 and c_count >= 8:
        transfer_signal = "FORMAL_SUPPORTED"
    elif r_count >= 8 or c_count >= 8:
        transfer_signal = "ASYMMETRIC_TRANSFER"
    else:
        transfer_signal = "NOT_SUPPORTED_OR_INCONCLUSIVE"
    formal_summary = {
        **pilot_summary,
        "C2_FORMAL_RUN": "YES",
        "valid_formal_units": len(valid_f),
        "R_TO_CV_SAFETY_TRANSFER": f"{r_count}/{len(valid_f)}",
        "C_TO_RV_HARM_TRANSFER": f"{c_count}/{len(valid_f)}",
        "BIDIRECTIONAL_TRANSFER_POSITIVE_UNITS": b_count,
        "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": transfer_signal,
    }
    save_json("stageC2_formal_summary.json", formal_summary)
    return formal_summary


def final_report(stage0_obj, c1, c2):
    transfer = c2.get("VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER", "NOT_RUN")
    c1_signal = c1.get("VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL", "NOT_RUN")
    if c1_signal == "FORMAL_SUPPORTED" and transfer == "FORMAL_SUPPORTED":
        cls = "VALUE_SIDE_GEOMETRY_BIDIRECTIONAL_TRANSFER_SUPPORTED"
        closure = "POSSIBLE"
    elif c1_signal == "FORMAL_SUPPORTED" and transfer == "ASYMMETRIC_TRANSFER":
        cls = "VALUE_SIDE_DIRECTION_FORMAL_SUPPORTED_WITH_ASYMMETRIC_TRANSFER"
        closure = "NO"
    elif c1_signal == "FORMAL_SUPPORTED":
        cls = "VALUE_SIDE_DIRECTION_FORMAL_SUPPORTED_TRANSFER_NOT_CLOSED"
        closure = "NO"
    else:
        cls = "VALUE_SIDE_DIRECTION_NOT_FORMALLY_SUPPORTED"
        closure = "NO"
    final = {
        "STAGE0": stage0_obj["STAGE0"],
        "C1_VALID_UNITS": c1.get("C1_VALID_UNITS", 0),
        "C1_VALID_RUNS": c1.get("C1_VALID_RUNS", 0),
        "V_ROTATION_GAP_DECREASE_RUNS": c1.get("V_ROTATION_GAP_DECREASE_RUNS", 0),
        "V_ROTATION_SIGN_INVERSIONS": c1.get("V_ROTATION_SIGN_INVERSIONS", 0),
        "V_ROTATION_POSITIVE_UNITS": c1.get("V_ROTATION_POSITIVE_UNITS", 0),
        "C1_MEDIAN_RESCUE": c1.get("C1_MEDIAN_RESCUE"),
        "VALUE_SIDE_FUNCTIONAL_DIRECTION_CAUSAL_SIGNAL": c1_signal,
        "STAGE_C2_RUN": c2.get("STAGE_C2_RUN", "NO"),
        "C2_PILOT_STATUS": c2.get("C2_PILOT_STATUS", "NOT_RUN"),
        "C2_FORMAL_RUN": c2.get("C2_FORMAL_RUN", "NO"),
        "R_TO_CV_SAFETY_TRANSFER": c2.get("R_TO_CV_SAFETY_TRANSFER", "NOT_RUN"),
        "C_TO_RV_HARM_TRANSFER": c2.get("C_TO_RV_HARM_TRANSFER", "NOT_RUN"),
        "BIDIRECTIONAL_TRANSFER_POSITIVE_UNITS": c2.get("BIDIRECTIONAL_TRANSFER_POSITIVE_UNITS", 0),
        "VALUE_GEOMETRY_BIDIRECTIONAL_CAUSAL_TRANSFER": transfer,
        "FINAL_MECHANISM_CLASSIFICATION": cls,
        "MECHANISM_CLOSURE_CANDIDATE": closure,
        "METHOD_DESIGN_READY": "NO",
        "git_commit_start": git_commit(),
        "output_dir": str(RUN_DIR),
    }
    sections = [
        "# GDN INT8 Value-Side Functional Direction Formal and Transfer V1",
        "## 1. Executive Summary\n\n" + json.dumps(final, indent=2),
        "## 2. Prior Evidence\n\nPrior evidence is inherited from `GDN_INT8_SAME_NORM_FUNCTIONAL_GEOMETRY_AUDIT_V1`; Stage0 PASS, observability gate PASS, and V-axis pilot SUPPORTED.",
        "## 3. Scientific Question\n\nDoes canonical same-norm R harmfulness travel through Value-side residual geometry, and can that geometry transfer harmfulness bidirectionally between R and C?",
        "## 4. Repository Audit\n\nReused previous canonical 9 units, residual construction, full-model KL metric, exact injection hook, QR rotation construction, and replay identity infrastructure.",
        "## 5. Tensor / Axis Semantics\n\nState is `[B,H,K,V]`; row is Key axis and column is Value axis. C1 applies right-side orthogonal Value rotations `E Q_V` with the same `Q_V` for R and C.",
        "## 6. Stage 0\n\n" + json.dumps(stage0_obj, indent=2),
        "## 7. Stage C1 Protocol\n\nAll 9 canonical units were evaluated with four fixed V-rotation seeds: " + ", ".join(map(str, ROT_SEEDS)) + ".",
        "## 8. V-Rotation Invariance Audit\n\n" + json.dumps({k: c1.get(k) for k in ["invalid_runs", "confounded_runs", "systematic_invariance_failure"]}, indent=2),
        "## 9. Stage C1 Formal Results\n\n" + json.dumps(c1, indent=2),
        "## 10. Unit-Level Rescue Analysis\n\nSee `stageC1_per_unit.csv`.",
        "## 11. Sign-Inversion Analysis\n\nSign inversions are secondary evidence and are not used as the formal gate.",
        "## 12. Stage C1 Gate\n\n" + c1_signal,
        "## 13. Stage C2 SVD / Subspace Audit\n\nSVD reconstruction and near-degenerate spectrum diagnostics are saved in `stageC2_*_svd_audit.json` when C2 runs.",
        "## 14. Stage C2 Pilot\n\n" + json.dumps(c2, indent=2),
        "## 15. Stage C2 Formal Results\n\n" + ("See `stageC2_formal_per_unit.csv`." if c2.get("C2_FORMAL_RUN") == "YES" else "Not run."),
        "## 16. R->C Value Geometry Transfer\n\n" + str(c2.get("R_TO_CV_SAFETY_TRANSFER", "NOT_RUN")),
        "## 17. C->R Value Geometry Transfer\n\n" + str(c2.get("C_TO_RV_HARM_TRANSFER", "NOT_RUN")),
        "## 18. Bidirectional Transfer Analysis\n\n" + str(c2.get("BIDIRECTIONAL_TRANSFER_POSITIVE_UNITS", 0)),
        "## 19. Positive Results\n\nC1 positive units and C2 transfer counts are reported only at unit level.",
        "## 20. Negative / Corrective Results\n\nAny C1 confounding or asymmetric C2 transfer is preserved as a scientific result.",
        "## 21. Mechanism Interpretation\n\n" + cls,
        "## 22. What Is NOT Proven\n\nNo downstream sensitive Value directions are identified here. No method design, bit allocation, quantizer, or deployment path is claimed.",
        "## 23. Mechanism Readiness\n\nMECHANISM_CLOSURE_CANDIDATE = " + closure + "; METHOD_DESIGN_READY = NO.",
        "## 24. Recommended Next Experiment\n\nOnly if C2 is strongly positive, consider `GDN_INT8_VALUE_DIRECTION_DOWNSTREAM_SENSITIVITY_AUDIT_V1`; it is not run in this task.",
    ]
    (RUN_DIR / "final_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    save_json("final_summary.json", {**final, "stage0": stage0_obj, "stageC1": c1, "stageC2": c2})
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "launch-c1", "c1-worker", "c1-merge", "c2", "final"], default="stage0")
    ap.add_argument("--worker-id")
    ap.add_argument("--task-indices")
    args = ap.parse_args()
    if args.stage == "stage0":
        st = stage0()
        print(json.dumps(st, indent=2, ensure_ascii=False))
        if st["STAGE0"] != "PASS":
            raise SystemExit(2)
    elif args.stage == "launch-c1":
        launch_c1()
    elif args.stage == "c1-worker":
        c1_worker(args.worker_id, [int(x) for x in args.task_indices.split(",") if x])
    elif args.stage == "c1-merge":
        print(json.dumps(c1_merge(), indent=2, ensure_ascii=False))
    elif args.stage == "c2":
        c1 = load_json(RUN_DIR / "stageC1_summary.json")
        print(json.dumps(c2_pipeline(c1), indent=2, ensure_ascii=False))
    elif args.stage == "final":
        st = load_json(RUN_DIR / "stage0_summary.json")
        c1 = load_json(RUN_DIR / "stageC1_summary.json") if (RUN_DIR / "stageC1_summary.json").exists() else {}
        c2 = load_json(RUN_DIR / "stageC2_formal_summary.json") if (RUN_DIR / "stageC2_formal_summary.json").exists() else load_json(RUN_DIR / "stageC2_pilot_summary.json")
        print(json.dumps(final_report(st, c1, c2), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
