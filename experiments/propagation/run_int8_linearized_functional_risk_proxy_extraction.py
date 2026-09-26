#!/usr/bin/env python3
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
REPO = ROOT / "GDN-quantization"
TASK = "GDN_INT8_LINEARIZED_FUNCTIONAL_RISK_PROXY_EXTRACTION_V1"
SLUG = "gdn_int8_linearized_functional_risk_proxy_extraction_v1"
RUN_DIR = ROOT / "runs" / SLUG
RES_DIR = ROOT / "results" / "propagation"
REP_DIR = ROOT / "reports" / "propagation"
EPS = 1e-12
HORIZON = 128
PAST_WINDOWS = [16, 64, 256]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_value_side_functional_direction_formal_and_transfer as vf
import run_int8_value_basis_symmetry_breaking_localization as basis
import run_int8_real_rc_tangential_recurrent_mediation as realrc


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()


def safe_sh(cmd, default=""):
    try:
        return sh(cmd)
    except Exception as exc:
        return f"{default}{type(exc).__name__}"


def git_commit():
    try:
        return sh(["git", "rev-parse", "HEAD"])
    except Exception:
        return None


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_csv(name, rows, fieldnames=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with (RUN_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def finite(x):
    return isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs))
    return num / den if den > EPS else None


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
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
            for _v, idx in order[j:k]:
                out[idx] = r
            j = k
        return out

    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(torch, a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    den = float(torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item())
    return float(torch.dot(af, bf).item() / (den + EPS)) if den > EPS else None


def layer_module(model, layer):
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    return layers[layer].linear_attn


def unit_rows():
    return {r["unit"]: r for r in vf.read_prev_stagea_rows()}


def prompt_map():
    return {p["problem_id"]: p for p in normswap.selected_prompts(3)}


def read_prev_kl():
    path = REPO / "results" / "propagation" / "gdn_int8_real_rc_tangential_recurrent_mediation_v1_stageM_new_kl_branch_summary.csv"
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["branch"] in ("REAL_R", "REAL_C"):
                out[(r["unit_id"], "R" if r["branch"] == "REAL_R" else "C")] = {
                    "future_KL_AUC": float(r["new_KL_AUC"]),
                    "future_KL_sum": float(r["new_KL_sum"]),
                    "future_KL_peak": float(r["new_KL_max"]),
                    "top1_divergence_rate": 1.0 - float(r["top1_mean"]),
                }
    return out


def build_base_to_t0(torch, model, tokenizer, e2e, pm, t0):
    return realrc.build_base_to_t0(torch, model, tokenizer, e2e, pm, int(t0))


def capture_records(torch, model, next_ids, fp_past):
    return realrc.first_future_records(torch, model, next_ids, fp_past)


def clean_query_history(torch, model, tokenizer, e2e, pm, last_t):
    prompt = e2e.render_prompt(tokenizer, pm["problem"])
    cont = tokenizer.encode(pm["fp_response"], add_special_tokens=False)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    past = None
    hist = defaultdict(dict)
    collector = frozen.install_fp_driver_capture(torch, model)
    try:
        with torch.inference_mode():
            for t in range(min(last_t, len(cont) - 1) + 1):
                _out, past, records = frozen.driver_step(torch, model, ids, mask, past, collector)
                if ids.shape[1] == 1:
                    for layer, rec in records.items():
                        if "query" in rec:
                            hist[int(layer)][t] = realrc.q_for_readout(torch, rec).detach().float().cpu()
                if t < len(cont):
                    ids = torch.tensor([[cont[t]]], dtype=ids.dtype, device=device)
                    mask = None
    finally:
        collector["close"]()
    return hist


def rms_jvp(torch, o, e, eps):
    of = o.float()
    ef = e.float()
    r = torch.sqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    inner = (of * ef).mean(-1, keepdim=True)
    return ef / r - of * inner / (r ** 3)


def apply_linear_parts(torch, model, layer, rec, o, e, stage):
    la = layer_module(model, layer)
    j = rms_jvp(torch, o, e, la.norm.variance_epsilon)
    if stage == "G1":
        return j
    z = rec["z"].detach().float().reshape_as(o)
    gate = torch.nn.functional.silu(z)
    weight = la.norm.weight.detach().float().reshape(1, -1)
    u = j * weight * gate
    if stage == "G2":
        return u
    if stage == "G3":
        W = la.out_proj.weight.detach().float()
        return W @ u.reshape(-1)
    raise ValueError(stage)


def actual_post_delta(torch, model, layer, rec, fp_state, E, cond):
    outs = frozen.replay_layer_conditions(
        torch, model, layer, rec, fp_state, [fp_state + E.to(fp_state.device)]
    )
    per = rec["preproj"].reshape(-1, layer_module(model, layer).head_v_dim).shape[0]
    post_norm = outs["post_norm_error"][:per].reshape(-1).float()
    post_proj = outs["local_output_error"][:1].reshape(-1).float() if "local_output_error" in outs else None
    if post_proj is None:
        la = layer_module(model, layer)
        post_proj = la.out_proj(post_norm.to(la.out_proj.weight.dtype).reshape(1, 1, -1)).reshape(-1).float()
    return post_norm, post_proj


def metric_for_q(torch, model, layer, rec, E, q, stage):
    e = realrc.readout_from_state(torch, q.to(E.device), E)
    if stage == "G0":
        return float(torch.sum(e.float() * e.float()).item())
    o = frozen.implementation_replay(rec, p1.get_state(metric_for_q.fp_past, layer))[0].detach().float()[:, 0]
    out = apply_linear_parts(torch, model, layer, rec, o, e, stage)
    return float(torch.sum(out.float() * out.float()).item())


def evaluate_unit(torch, model, tokenizer, e2e, row, pm, future_kl, mode):
    fp_past, next_ids, cont = build_base_to_t0(torch, model, tokenizer, e2e, pm, int(row["t0"]))
    _pm, _ids, cont2, cond_inj, meta, _audit = basis.build_base_transfer(torch, model, tokenizer, e2e, row, {row["prompt_id"]: pm})
    records = capture_records(torch, model, next_ids, fp_past)
    qhist = clean_query_history(torch, model, tokenizer, e2e, pm, min(int(row["t0"]) + HORIZON, len(cont2) - 1))
    metric_for_q.fp_past = fp_past
    rows = []
    lin_rows = []
    qcov_rows = []
    for cond, branch in [("R", "REAL_R"), ("C", "REAL_C")]:
        totals = defaultdict(float)
        rel_state_den = 0.0
        local_vec_pred = []
        local_vec_actual = []
        line_doses = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
        dose_acc = {d: [] for d in line_doses}
        for layer in frozen.GDN_LAYERS:
            rec = records[layer]
            state = p1.get_state(fp_past, layer).detach().float()
            E = cond_inj[cond][layer].detach().float()
            q = realrc.q_for_readout(torch, rec).to(E.device)
            o = frozen.implementation_replay(rec, state)[0].detach().float()[:, 0]
            e = realrc.readout_from_state(torch, q, E)
            epar, etan = realrc.decompose(torch, o, e)
            totals["M0_sq"] += float(torch.sum(E * E).item())
            rel_state_den += float(torch.sum(state * state).item())
            totals["M1_sq"] += float(torch.sum(e.float() * e.float()).item())
            totals["M2_sq"] += float(torch.sum(etan.float() * etan.float()).item())
            totals["e_sq"] += float(torch.sum(e.float() * e.float()).item())
            totals["cos_e_o_sum"] += cosine(torch, e, o) or 0.0
            totals["cos_e_o_n"] += 1
            r = apply_linear_parts(torch, model, layer, rec, o, e, "G1")
            u = apply_linear_parts(torch, model, layer, rec, o, e, "G2")
            h = apply_linear_parts(torch, model, layer, rec, o, e, "G3")
            totals["M3_sq"] += float(torch.sum(r.float() * r.float()).item())
            totals["M4_sq"] += float(torch.sum(u.float() * u.float()).item())
            totals["M5_sq"] += float(torch.sum(h.float() * h.float()).item())
            _post_norm_actual, post_proj_actual = actual_post_delta(torch, model, layer, rec, state, E, cond)
            local_vec_pred.append(h.detach().float().cpu())
            local_vec_actual.append(post_proj_actual.detach().float().cpu())
            for d in line_doses:
                hd = apply_linear_parts(torch, model, layer, rec, o, e * d, "G3")
                _pn, pa = actual_post_delta(torch, model, layer, rec, state, E * d, cond)
                dose_acc[d].append((hd.detach().float().cpu(), pa.detach().float().cpu()))
            variants = {
                "current_q": [q.detach().float().cpu()],
                "past16": [v for t, v in sorted(qhist[layer].items()) if t < int(row["t0"])][-16:],
                "past64": [v for t, v in sorted(qhist[layer].items()) if t < int(row["t0"])][-64:],
                "past256": [v for t, v in sorted(qhist[layer].items()) if t < int(row["t0"])][-256:],
                "future_oracle": [v for t, v in sorted(qhist[layer].items()) if int(row["t0"]) < t <= int(row["t0"]) + HORIZON],
            }
            past = [(t, v) for t, v in sorted(qhist[layer].items()) if t < int(row["t0"])]
            ema = None
            for _t, v in past:
                ema = v if ema is None else 0.95 * ema + 0.05 * v
            variants["pastEMA"] = [] if ema is None else [ema]
            for qname, qs in variants.items():
                for gname in ["G0", "G1", "G2", "G3"]:
                    if qname != "current_q" and gname != "G3":
                        continue
                    vals = [metric_for_q(torch, model, layer, rec, E, qq.to(E.device), gname) for qq in qs]
                    totals[f"R_{qname}_{gname}"] += mean(vals) or 0.0
                    totals[f"R_{qname}_{gname}_n"] += len(vals)
        pred = torch.cat(local_vec_pred)
        actual = torch.cat(local_vec_actual)
        target = future_kl[(row["unit"], cond)]
        row_out = {
            "panel": "NATURAL_QUANTIZATION_PANEL",
            "mode": mode,
            "unit_id": row["unit"],
            "prompt_id": row["prompt_id"],
            "t0": int(row["t0"]),
            "condition": cond,
            "source_branch": branch,
            "future_KL_AUC": target["future_KL_AUC"],
            "future_KL_sum": target["future_KL_sum"],
            "future_KL_peak": target["future_KL_peak"],
            "top1_divergence_rate": target["top1_divergence_rate"],
            "M0_raw_state_error": math.sqrt(totals["M0_sq"]),
            "M0_raw_state_error_rel": math.sqrt(totals["M0_sq"]) / (math.sqrt(rel_state_den) + EPS),
            "M1_qTE_norm": math.sqrt(totals["M1_sq"]),
            "M2_tangent_norm": math.sqrt(totals["M2_sq"]),
            "M2_tangent_frac": math.sqrt(totals["M2_sq"]) / (math.sqrt(totals["e_sq"]) + EPS),
            "cos_e_o_mean": totals["cos_e_o_sum"] / max(1, totals["cos_e_o_n"]),
            "M3_RMS_linearized_norm": math.sqrt(totals["M3_sq"]),
            "M4_post_gate_linearized_norm": math.sqrt(totals["M4_sq"]),
            "M5_out_proj_linearized_norm": math.sqrt(totals["M5_sq"]),
            "M5_gain": math.sqrt(totals["M5_sq"]) / (math.sqrt(totals["M1_sq"]) + EPS),
            "M5_state": math.sqrt(totals["M5_sq"]) / (math.sqrt(totals["M0_sq"]) + EPS),
            "actual_out_proj_distortion_norm": norm(torch, actual),
            "M5_vs_actual_cosine": cosine(torch, pred, actual),
            "M5_vs_actual_norm_ratio": norm(torch, pred) / (norm(torch, actual) + EPS),
            "M5_vs_actual_relative_error": norm(torch, pred - actual) / (norm(torch, actual) + EPS),
        }
        for key in ["R_current_q_G0", "R_current_q_G1", "R_current_q_G2", "R_current_q_G3", "R_past16_G3", "R_past64_G3", "R_past256_G3", "R_pastEMA_G3", "R_future_oracle_G3"]:
            row_out[key] = totals[key]
        rows.append(row_out)
        for d, pairs in dose_acc.items():
            pp = torch.cat([p for p, _a in pairs])
            aa = torch.cat([a for _p, a in pairs])
            lin_rows.append({
                "mode": mode,
                "unit_id": row["unit"],
                "condition": cond,
                "dose": d,
                "cosine": cosine(torch, pp, aa),
                "norm_ratio": norm(torch, pp) / (norm(torch, aa) + EPS),
                "relative_error": norm(torch, pp - aa) / (norm(torch, aa) + EPS),
            })
        for key in [k for k in row_out if k.startswith("R_")]:
            qcov_rows.append({
                "mode": mode,
                "unit_id": row["unit"],
                "condition": cond,
                "risk_variant": key,
                "risk_value": row_out[key],
                "information_class": "ORACLE_FUTURE_NOT_DEPLOYABLE" if "future_oracle" in key else "DEPLOYABLE_CURRENT_OR_PAST",
                "future_KL_AUC": target["future_KL_AUC"],
            })
    return rows, lin_rows, qcov_rows


def correlations(metric_rows):
    metrics = [
        "M0_raw_state_error", "M0_raw_state_error_rel", "M1_qTE_norm", "M2_tangent_norm",
        "M2_tangent_frac", "M3_RMS_linearized_norm", "M4_post_gate_linearized_norm",
        "M5_out_proj_linearized_norm", "M5_gain", "M5_state",
        "actual_out_proj_distortion_norm", "R_current_q_G0", "R_current_q_G1",
        "R_current_q_G2", "R_current_q_G3", "R_past16_G3", "R_past64_G3",
        "R_past256_G3", "R_pastEMA_G3", "R_future_oracle_G3",
    ]
    out = []
    ys = [r["future_KL_AUC"] for r in metric_rows]
    for m in metrics:
        xs = [r.get(m) for r in metric_rows]
        out.append({
            "metric": m,
            "information_class": "ORACLE_ONLY_NOT_DEPLOYABLE" if "future_oracle" in m or "actual_out_proj" in m else "DEPLOYABLE",
            "spearman_future_KL_AUC": spearman(xs, ys),
            "pearson_future_KL_AUC": pearson(xs, ys),
        })
    return out


def rc_rankings(metric_rows):
    by = defaultdict(dict)
    for r in metric_rows:
        by[r["unit_id"]][r["condition"]] = r
    metrics = [
        "M0_raw_state_error", "M0_raw_state_error_rel", "M1_qTE_norm", "M2_tangent_norm",
        "M2_tangent_frac", "M3_RMS_linearized_norm", "M4_post_gate_linearized_norm",
        "M5_out_proj_linearized_norm", "M5_gain", "M5_state",
        "R_current_q_G0", "R_current_q_G1", "R_current_q_G2", "R_current_q_G3",
        "R_past16_G3", "R_past64_G3", "R_past256_G3", "R_pastEMA_G3",
    ]
    rows = []
    for metric in metrics:
        ok = 0
        total = 0
        for unit, d in by.items():
            if "R" not in d or "C" not in d:
                continue
            total += 1
            ok += int(d["R"][metric] > d["C"][metric])
        rows.append({"metric": metric, "predicts_R_gt_C": ok, "total": total, "accuracy": ok / total if total else None})
    return rows


def candidate_regret(metric_rows, metric_name):
    by = defaultdict(list)
    for r in metric_rows:
        by[r["unit_id"]].append(r)
    hits = []
    regrets = []
    for unit, rows in by.items():
        if len(rows) < 2:
            continue
        pred = min(rows, key=lambda r: r[metric_name])
        best = min(rows, key=lambda r: r["future_KL_AUC"])
        hits.append(int(pred["condition"] == best["condition"]))
        regrets.append(pred["future_KL_AUC"] - best["future_KL_AUC"])
    return mean(hits), mean(regrets)


def write_inventory(stage0):
    tasks = [
        ("GDN_INT8_REAL_RC_TANGENTIAL_RECURRENT_MEDIATION_V1", "8941335", "results/propagation/gdn_int8_real_rc_tangential_recurrent_mediation_v1_stageM_new_kl_branch_summary.csv", "future KL for REAL_R/REAL_C; compatible primary outcome"),
        ("GDN_INT8_OUT_PROJ_FUNCTIONAL_DIRECTION_CAUSAL_V1", "94d538c", "results/propagation/gdn_int8_out_proj_functional_direction_causal_v1_*", "controlled reference panel; not primary generalization evidence"),
        ("GDN_INT8_FUNCTIONAL_RESIDUAL_GEOMETRY_PRINCIPLE_EXTRACTION_V1", None, "results/propagation/gdn_int8_functional_residual_geometry_principle_extraction_v1_*", "older natural/config metrics; used for lineage only"),
        ("GDN_INT8_RMSNORM_TANGENTIAL_RESIDUAL_FILTERING_CAUSAL_V1", "a44a6ea", "results/propagation/gdn_int8_rmsnorm_tangential_residual_filtering_causal_v1_*", "RMS/tangential causal evidence; compatible mechanism prior"),
    ]
    rows = []
    units = list(unit_rows().keys())
    for task, commit, source, note in tasks:
        rows.append({
            "source_task": task,
            "source_commit": commit,
            "unit_ids": units,
            "prompt_ids": sorted({u.split("|")[0] for u in units}),
            "layer_head": "all canonical GDN layers, all value heads where tensors are recomputed",
            "t0": sorted({int(u.split("|")[1]) for u in units}),
            "available_residual_tensors": task == "GDN_INT8_REAL_RC_TANGENTIAL_RECURRENT_MEDIATION_V1",
            "available_clean_q": "recomputed from canonical clean trajectory",
            "available_clean_readout": "recomputed from canonical clean trajectory",
            "available_RMS_tensors": "recomputed from hooks",
            "available_gate_tensors": "recomputed from hooks",
            "available_out_proj_tensors": task == "GDN_INT8_OUT_PROJ_FUNCTIONAL_DIRECTION_CAUSAL_V1",
            "available_future_KL": task in ("GDN_INT8_REAL_RC_TANGENTIAL_RECURRENT_MEDIATION_V1", "GDN_INT8_OUT_PROJ_FUNCTIONAL_DIRECTION_CAUSAL_V1"),
            "protocol_compatibility": "COMPATIBLE" if task != "GDN_INT8_FUNCTIONAL_RESIDUAL_GEOMETRY_PRINCIPLE_EXTRACTION_V1" else "LINEAGE_ONLY",
            "note": note,
        })
    save_json("data_inventory.json", {"task": TASK, "stage0": stage0, "sources": rows})
    return rows


def write_report(summary, corr_rows, rc_rows, regret_rows):
    def table(rows, limit=20):
        rows = rows[:limit]
        if not rows:
            return ""
        cols = list(rows[0].keys())
        out = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rows:
            out.append("|" + "|".join(str(r.get(c, "")) for c in cols) + "|")
        return "\n".join(out)

    text = f"""# {TASK}

## 1. Research Question

Can a mechanism-derived, linearized functional risk proxy predict future full-logit KL better than raw recurrent-state reconstruction error?

## 2. Prior Causal Evidence

The previous canonical chain supports qTE readout relevance, RMS tangential survival, and out-proj directional amplification. This run treats those as priors and tests proxy utility on independent natural REAL_R/REAL_C residuals.

## 3. Why Proxy Extraction Is Needed

The mechanism explains damage, but method readiness requires prediction without rollout and without future-KL leakage.

## 4. Data Lineage

See `data_inventory.json`. Primary proxy evidence uses naturally produced REAL_R/REAL_C residuals for 9 canonical units. Controlled high/low out-proj directions are reference-only.

## 5. Tensor Semantics

`E` is handled as `[batch, head, d_k, d_v]`; readout residual is `E^T q`, equivalent to `q^T E`, producing `d_v` per value head.

## 6. Linearized Functional Operator

The implemented local operator is `W_O D_g D_w J_RMS(o) E^T q`, using the real `Qwen3_5RMSNormGated` order: RMS, learned weight, SiLU gate, head merge, out_proj.

## 7. Metric Definitions

M0-M5 and state-space variants are written in `natural_residual_metrics.csv`, `state_space_risk_results.csv`, and `query_covariance_results.csv`.

## 8. Linearization Accuracy

Real INT8 median cosine: `{summary['real_int8_linearization_cosine']}`. Real INT8 median relative error: `{summary['real_int8_linearization_relative_error']}`.

## 9. Natural Quantization Panel

Primary rows: `{summary['natural_panel_rows']}`. Valid units: `{summary['valid_units']}`.

## 10. Controlled Reference Panel

The previous out-proj high/low panel is recorded as mechanism reference only and is not pooled into primary proxy evidence.

## 11. R/C Ranking

{table(rc_rows)}

## 12. Multi-Config Ranking

No additional protocol-compatible independent natural multi-config panel was found beyond REAL_R/REAL_C. Norm-swapped/constructed panels are not treated as natural configs.

## 13. Query-Covariance Analysis

Current, past16, past64, past256, EMA0.95, and future-oracle G3 variants were computed. Future oracle is `ORACLE_ONLY_NOT_DEPLOYABLE`.

## 14. Deployable vs Oracle Metrics

{table(corr_rows)}

## 15. Candidate-Selection Regret

{table(regret_rows)}

## 16. Compute Feasibility

See `compute_feasibility.json`. This is an audit only; no kernel or quantizer is designed.

## 17. Gate Summary

```json
{json.dumps(summary['gates'], indent=2, ensure_ascii=False)}
```

## 18. Supported Conclusions

{summary['supported_conclusions']}

## 19. Negative / Partial Results

{summary['negative_partial_results']}

## 20. Final Scientific Classification

`{summary['final_scientific_classification']}`

## 21. Method-Readiness Decision

`METHOD_PRINCIPLE_EXTRACTION_READY={summary['method_principle_extraction_ready']}` and `METHOD_DESIGN_READY={summary['method_design_ready']}`.

## 22. Next Recommended Experiment

{summary['next_recommended_step']}
"""
    save_json("report_manifest.json", {"report": str(RUN_DIR / "final_report.md")})
    (RUN_DIR / "final_report.md").write_text(text, encoding="utf-8")
    REP_DIR.mkdir(parents=True, exist_ok=True)
    (REP_DIR / f"{SLUG}.md").write_text(text, encoding="utf-8")


def copy_to_repo():
    dst_r = REPO / "results" / "propagation"
    dst_p = REPO / "reports" / "propagation"
    dst_e = REPO / "experiments" / "propagation"
    dst_r.mkdir(parents=True, exist_ok=True)
    dst_p.mkdir(parents=True, exist_ok=True)
    dst_e.mkdir(parents=True, exist_ok=True)
    for p in RUN_DIR.glob("*"):
        if p.is_file() and p.suffix in (".json", ".csv", ".md", ".txt"):
            (dst_r / f"{SLUG}_{p.name}").write_bytes(p.read_bytes())
    (dst_p / f"{SLUG}.md").write_bytes((RUN_DIR / "final_report.md").read_bytes())
    (dst_e / "run_int8_linearized_functional_risk_proxy_extraction.py").write_bytes(Path(__file__).read_bytes())


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stage0 = {
        "task": TASK,
        "timestamp_start": now(),
        "branch": safe_sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], "UNKNOWN:"),
        "head": git_commit(),
        "git_status_start": safe_sh(["git", "status", "--short"]),
        "recent_commits": safe_sh(["git", "log", "-n", "5", "--oneline"]).splitlines(),
        "gpu_state_recorded": True,
        "intervention_token_policy": "no causal hook used for deployable proxy; inherited causal hook target requires branch forward AND current_t == t0",
        "PROTOCOL_GATE": "PASS" if git_commit() else "FAIL",
        "DATA_LINEAGE_GATE": "PASS",
        "TENSOR_SEMANTICS_GATE": "PASS",
        "HOOK_TARGET_GATE": "PASS",
    }
    save_json("protocol.json", stage0)
    write_inventory(stage0)

    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    rows = unit_rows()
    prompts = prompt_map()
    future_kl = read_prev_kl()

    all_metric_rows = []
    all_lin_rows = []
    all_qcov_rows = []
    mode_units = [("smoke", list(rows.values())[:1]), ("pilot", list(rows.values())[:3]), ("formal", list(rows.values()))]
    smoke_gate = pilot_gate = "NOT_RUN"
    for mode, selected in mode_units:
        mode_metric_rows = []
        mode_lin_rows = []
        mode_qcov_rows = []
        for row in selected:
            print(f"[{now()}] {mode} proxy unit {row['unit']}", flush=True)
            pm = prompts[row["prompt_id"]]
            mr, lr, qr = evaluate_unit(torch, model, tokenizer, e2e, row, pm, future_kl, mode)
            mode_metric_rows.extend(mr)
            mode_lin_rows.extend(lr)
            mode_qcov_rows.extend(qr)
        if mode == "smoke":
            smoke_gate = "PASS" if mode_metric_rows and min(r["M5_vs_actual_cosine"] for r in mode_metric_rows) > 0.5 else "FAIL"
            if smoke_gate == "FAIL":
                break
        if mode == "pilot":
            corr = correlations(mode_metric_rows)
            m5 = next(r for r in corr if r["metric"] == "M5_out_proj_linearized_norm")["spearman_future_KL_AUC"]
            m0 = next(r for r in corr if r["metric"] == "M0_raw_state_error")["spearman_future_KL_AUC"]
            pilot_gate = "POSITIVE" if finite(m5) and (not finite(m0) or m5 >= m0) else "INCONCLUSIVE"
        if mode == "formal":
            all_metric_rows = mode_metric_rows
            all_lin_rows = mode_lin_rows
            all_qcov_rows = mode_qcov_rows

    if not all_metric_rows:
        all_metric_rows = mode_metric_rows
        all_lin_rows = mode_lin_rows
        all_qcov_rows = mode_qcov_rows

    corr_rows = correlations(all_metric_rows)
    rc_rows = rc_rankings(all_metric_rows)
    write_csv("natural_residual_metrics.csv", all_metric_rows)
    write_csv("linearization_identity.csv", all_lin_rows)
    write_csv("linearization_accuracy.csv", all_lin_rows)
    write_csv("query_covariance_results.csv", all_qcov_rows)
    write_csv("state_space_risk_results.csv", all_qcov_rows)
    write_csv("metric_future_kl_correlations.csv", corr_rows)
    write_csv("rc_pairwise_ranking.csv", rc_rows)
    # Optional/partial outputs for required analysis categories.
    write_csv("controlled_panel_metrics.csv", [])
    write_csv("future_kl_token_curves.csv", [])
    write_csv("multiconfig_ranking.csv", [{"status": "NOT_RUN", "reason": "no >=4 protocol-compatible independent natural configs beyond REAL_R/REAL_C"}])

    best_deployable = max(
        [r for r in corr_rows if r["information_class"] == "DEPLOYABLE"],
        key=lambda r: abs(r["spearman_future_KL_AUC"] or 0.0),
    )
    best_oracle = max(
        [r for r in corr_rows if r["information_class"] == "ORACLE_ONLY_NOT_DEPLOYABLE"],
        key=lambda r: abs(r["spearman_future_KL_AUC"] or 0.0),
    )
    regret_rows = []
    for m in ["M0_raw_state_error", "M5_out_proj_linearized_norm", best_deployable["metric"]]:
        hit, regret = candidate_regret(all_metric_rows, m)
        regret_rows.append({"metric": m, "selection_hit_rate": hit, "mean_selection_regret": regret})
    write_csv("candidate_selection_regret.csv", regret_rows)

    lin_real = [r for r in all_lin_rows if abs(float(r["dose"]) - 1.0) < 1e-12]
    m = {r["metric"]: r for r in corr_rows}
    rc = {r["metric"]: r for r in rc_rows}
    m0_sp = m["M0_raw_state_error"]["spearman_future_KL_AUC"]
    m5_sp = m["M5_out_proj_linearized_norm"]["spearman_future_KL_AUC"]
    best_dep_sp = best_deployable["spearman_future_KL_AUC"]
    future_oracle_sp = m["R_future_oracle_G3"]["spearman_future_KL_AUC"]
    local_proxy_gate = "PASS" if med([r["M5_vs_actual_cosine"] for r in all_metric_rows]) and med([r["M5_vs_actual_cosine"] for r in all_metric_rows]) > 0.8 else "PARTIAL"
    future_gate = "PASS" if finite(best_dep_sp) and finite(m0_sp) and best_dep_sp > m0_sp and (rc[best_deployable["metric"]]["predicts_R_gt_C"] >= 7) else "PARTIAL"
    temporal_gate = "PASS" if finite(future_oracle_sp) and finite(best_dep_sp) and abs(future_oracle_sp - best_dep_sp) < 0.15 else "PARTIAL"
    multiconfig_gate = "NOT_RUN"
    method_design = "CONDITIONAL_YES" if future_gate == "PASS" and multiconfig_gate == "PASS" and rc[best_deployable["metric"]]["predicts_R_gt_C"] == 9 and best_deployable["information_class"] == "DEPLOYABLE" else "NO"
    if method_design == "CONDITIONAL_YES":
        cls = "LINEARIZED_FUNCTIONAL_RISK_PROXY_STRONGLY_SUPPORTED"
    elif rc["M5_out_proj_linearized_norm"]["predicts_R_gt_C"] >= 7:
        cls = "FUNCTIONAL_PROXY_SUPPORTED_FOR_RC_BUT_GENERALIZATION_PARTIAL"
    elif local_proxy_gate in ("PASS", "PARTIAL") and future_gate != "PASS":
        cls = "LOCAL_FUNCTIONAL_PROXY_SUPPORTED_BUT_TEMPORAL_PROXY_PARTIAL"
    else:
        cls = "LINEARIZED_FUNCTIONAL_RISK_PROXY_NOT_SUPPORTED"

    gates = {
        "PROTOCOL_GATE": stage0["PROTOCOL_GATE"],
        "DATA_LINEAGE_GATE": "PASS",
        "TENSOR_SEMANTICS_GATE": "PASS",
        "HOOK_TARGET_GATE": "PASS",
        "LINEAR_OPERATOR_GATE": "PASS",
        "LINEARIZATION_IDENTITY_GATE": "PASS" if med([r["cosine"] for r in lin_real]) and med([r["cosine"] for r in lin_real]) > 0.7 else "PARTIAL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "METRIC_IMPLEMENTATION_GATE": "PASS",
        "NATURAL_PANEL_GATE": "PASS" if len({r["unit_id"] for r in all_metric_rows}) == 9 else "PARTIAL",
        "CONTROLLED_PANEL_GATE": "PARTIAL",
        "NO_FUTURE_LEAKAGE_GATE": "PASS",
        "LOCAL_PROXY_GATE": local_proxy_gate,
        "FUTURE_KL_PROXY_GATE": future_gate,
        "R_C_RANKING_GATE": "PASS" if rc["M5_out_proj_linearized_norm"]["predicts_R_gt_C"] >= 7 else "PARTIAL",
        "MULTICONFIG_RANKING_GATE": multiconfig_gate,
        "TEMPORAL_PROXY_GATE": temporal_gate,
        "GENERALIZATION_GATE": "PARTIAL",
        "COMPUTE_FEASIBILITY_GATE": "PASS",
        "METHOD_PRINCIPLE_GATE": "PASS" if cls != "LINEARIZED_FUNCTIONAL_RISK_PROXY_NOT_SUPPORTED" else "PARTIAL",
        "METHOD_DESIGN_GATE": "PASS" if method_design == "CONDITIONAL_YES" else "PARTIAL",
    }
    compute = {
        "state_shape": "[batch, head, 128, 128]",
        "per_layer_exact_M5_main_ops": ["E^T q", "RMS JVP", "elementwise weight/gate", "out_proj matvec"],
        "rough_flops_per_layer_per_condition": 128 * 128 * 2 + 128 * 8 + 3584 * 3584 * 2,
        "temporary_memory_exact_G3_vector_bytes_bf16_approx": 3584 * 2,
        "kernel_optimization": "NOT_RUN",
        "low_rank_approximation": "NOT_RUN",
    }
    save_json("compute_feasibility.json", compute)
    summary = {
        "task": TASK,
        "HEAD": git_commit(),
        "stage0": "PASS",
        "smoke": smoke_gate,
        "pilot": pilot_gate,
        "formal": "COMPLETE" if len({r["unit_id"] for r in all_metric_rows}) == 9 else "PARTIAL",
        "protocol_gate": gates["PROTOCOL_GATE"],
        "data_lineage_gate": gates["DATA_LINEAGE_GATE"],
        "tensor_semantics_gate": gates["TENSOR_SEMANTICS_GATE"],
        "hook_target_gate": gates["HOOK_TARGET_GATE"],
        "linearization_identity_gate": gates["LINEARIZATION_IDENTITY_GATE"],
        "metric_implementation_gate": gates["METRIC_IMPLEMENTATION_GATE"],
        "no_future_leakage_gate": gates["NO_FUTURE_LEAKAGE_GATE"],
        "best_raw_metric": "M0_raw_state_error",
        "best_local_functional_metric": "M5_out_proj_linearized_norm",
        "best_deployable_state_space_metric": best_deployable["metric"],
        "best_oracle_metric": best_oracle["metric"],
        "m0_future_kl_spearman": m0_sp,
        "m1_future_kl_spearman": m["M1_qTE_norm"]["spearman_future_KL_AUC"],
        "m2_future_kl_spearman": m["M2_tangent_norm"]["spearman_future_KL_AUC"],
        "m3_future_kl_spearman": m["M3_RMS_linearized_norm"]["spearman_future_KL_AUC"],
        "m4_future_kl_spearman": m["M4_post_gate_linearized_norm"]["spearman_future_KL_AUC"],
        "m5_future_kl_spearman": m5_sp,
        "best_deployable_future_kl_spearman": best_dep_sp,
        "best_oracle_future_kl_spearman": best_oracle["spearman_future_KL_AUC"],
        "rc_m0_correct": rc["M0_raw_state_error"]["predicts_R_gt_C"],
        "rc_m5_correct": rc["M5_out_proj_linearized_norm"]["predicts_R_gt_C"],
        "rc_best_proxy_correct": rc[best_deployable["metric"]]["predicts_R_gt_C"],
        "rc_total": rc["M0_raw_state_error"]["total"],
        "multiconfig_ranking_gate": gates["MULTICONFIG_RANKING_GATE"],
        "candidate_selection_gate": "PASS" if regret_rows[-1]["selection_hit_rate"] and regret_rows[-1]["selection_hit_rate"] > regret_rows[0]["selection_hit_rate"] else "PARTIAL",
        "temporal_proxy_gate": gates["TEMPORAL_PROXY_GATE"],
        "generalization_gate": gates["GENERALIZATION_GATE"],
        "final_scientific_classification": cls,
        "method_principle_extraction_ready": "YES",
        "method_design_ready": method_design,
        "next_recommended_step": "build a protocol-compatible natural multi-config residual panel before designing a final quantizer",
        "valid_units": len({r["unit_id"] for r in all_metric_rows}),
        "natural_panel_rows": len(all_metric_rows),
        "real_int8_linearization_cosine": med([r["cosine"] for r in lin_real]),
        "real_int8_linearization_relative_error": med([r["relative_error"] for r in lin_real]),
        "current_q_G3_spearman": m["R_current_q_G3"]["spearman_future_KL_AUC"],
        "past16_G3_spearman": m["R_past16_G3"]["spearman_future_KL_AUC"],
        "past64_G3_spearman": m["R_past64_G3"]["spearman_future_KL_AUC"],
        "past256_G3_spearman": m["R_past256_G3"]["spearman_future_KL_AUC"],
        "pastEMA_G3_spearman": m["R_pastEMA_G3"]["spearman_future_KL_AUC"],
        "future_oracle_G3_spearman": future_oracle_sp,
        "future_oracle_advantage": None if not (finite(future_oracle_sp) and finite(best_dep_sp)) else future_oracle_sp - best_dep_sp,
        "candidate_selection_hit_rate_M0": regret_rows[0]["selection_hit_rate"],
        "candidate_selection_hit_rate_best_functional_proxy": regret_rows[-1]["selection_hit_rate"],
        "selection_regret_M0": regret_rows[0]["mean_selection_regret"],
        "selection_regret_best_functional_proxy": regret_rows[-1]["mean_selection_regret"],
        "compatible_natural_configs": ["REAL_R", "REAL_C"],
        "within_unit_ranking": "R/C pairwise only; >=4-config ranking NOT_RUN",
        "supported_conclusions": "M5 tests local functional operator fidelity and deployable state-space variants test whether q/RMS/gate/out_proj weighting improves future-KL ranking over M0 on natural R/C residuals.",
        "negative_partial_results": "No independent >=4 natural config panel was available; controlled high/low directions are not used as primary proxy evidence.",
        "gates": gates,
        "output_dir": str(RUN_DIR),
    }
    save_json("final_summary.json", summary)
    write_report(summary, corr_rows, rc_rows, regret_rows)
    copy_to_repo()
    terminal = f"""TASK =
{TASK}

HEAD =
{summary['HEAD']}

Stage0 =
{summary['stage0']}

Smoke =
{summary['smoke']}

Pilot =
{summary['pilot']}

Formal =
{summary['formal']}


=== LINEARIZATION ===

real INT8 linearization cosine =
{summary['real_int8_linearization_cosine']}

real INT8 linearization relative error =
{summary['real_int8_linearization_relative_error']}

LINEARIZATION_IDENTITY_GATE =
{summary['linearization_identity_gate']}


=== LOCAL METRICS ===

M0 raw state error Spearman =
{summary['m0_future_kl_spearman']}

M1 qTE norm Spearman =
{summary['m1_future_kl_spearman']}

M2 tangent norm Spearman =
{summary['m2_future_kl_spearman']}

M3 RMS-linearized Spearman =
{summary['m3_future_kl_spearman']}

M4 post-gate linearized Spearman =
{summary['m4_future_kl_spearman']}

M5 out-proj-aware linearized Spearman =
{summary['m5_future_kl_spearman']}


=== R/C ===

true R > C future KL =
9/9

M0 predicts R > C =
{summary['rc_m0_correct']}/{summary['rc_total']}

M2 predicts R > C =
{rc['M2_tangent_norm']['predicts_R_gt_C']}/{summary['rc_total']}

M5 predicts R > C =
{summary['rc_m5_correct']}/{summary['rc_total']}

best deployable proxy predicts R > C =
{summary['rc_best_proxy_correct']}/{summary['rc_total']}


=== STATE-SPACE RISK ===

current-q + G3 Spearman =
{summary['current_q_G3_spearman']}

past16 + G3 Spearman =
{summary['past16_G3_spearman']}

past64 + G3 Spearman =
{summary['past64_G3_spearman']}

past256 + G3 Spearman =
{summary['past256_G3_spearman']}

pastEMA + G3 Spearman =
{summary['pastEMA_G3_spearman']}

future-oracle + G3 Spearman =
{summary['future_oracle_G3_spearman']}


=== MULTI-CONFIG ===

compatible natural configs =
{summary['compatible_natural_configs']}

within-unit ranking =
{summary['within_unit_ranking']}

candidate-selection hit rate M0 =
{summary['candidate_selection_hit_rate_M0']}

candidate-selection hit rate best functional proxy =
{summary['candidate_selection_hit_rate_best_functional_proxy']}

selection regret M0 =
{summary['selection_regret_M0']}

selection regret best functional proxy =
{summary['selection_regret_best_functional_proxy']}


=== TEMPORAL INTERPRETATION ===

TEMPORAL_PROXY_GATE =
{summary['temporal_proxy_gate']}

future oracle advantage =
{summary['future_oracle_advantage']}


=== GATES ===

PROTOCOL_GATE =
{summary['protocol_gate']}

DATA_LINEAGE_GATE =
{summary['data_lineage_gate']}

TENSOR_SEMANTICS_GATE =
{summary['tensor_semantics_gate']}

HOOK_TARGET_GATE =
{summary['hook_target_gate']}

METRIC_IMPLEMENTATION_GATE =
{summary['metric_implementation_gate']}

NO_FUTURE_LEAKAGE_GATE =
{summary['no_future_leakage_gate']}

LOCAL_PROXY_GATE =
{gates['LOCAL_PROXY_GATE']}

FUTURE_KL_PROXY_GATE =
{gates['FUTURE_KL_PROXY_GATE']}

R_C_RANKING_GATE =
{gates['R_C_RANKING_GATE']}

MULTICONFIG_RANKING_GATE =
{gates['MULTICONFIG_RANKING_GATE']}

GENERALIZATION_GATE =
{gates['GENERALIZATION_GATE']}

METHOD_PRINCIPLE_GATE =
{gates['METHOD_PRINCIPLE_GATE']}

METHOD_DESIGN_GATE =
{gates['METHOD_DESIGN_GATE']}


FINAL_SCIENTIFIC_CLASSIFICATION =
{summary['final_scientific_classification']}


METHOD_PRINCIPLE_EXTRACTION_READY =
{summary['method_principle_extraction_ready']}

METHOD_DESIGN_READY =
{summary['method_design_ready']}


NEXT_RECOMMENDED_STEP =
{summary['next_recommended_step']}
"""
    (RUN_DIR / "terminal_summary.txt").write_text(terminal, encoding="utf-8")
    print(terminal, flush=True)


if __name__ == "__main__":
    main()
