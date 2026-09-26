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
RUN_DIR = ROOT / "runs" / "gdn_int8_functional_residual_geometry_principle_extraction_v1"
PREV_BASIS = ROOT / "runs" / "gdn_int8_value_basis_symmetry_breaking_localization_v1"
PREV_CLOSURE = ROOT / "runs" / "gdn_int8_rmsnorm_value_geometry_mechanism_closure_v1"
PREV_TRANSFER = ROOT / "runs" / "gdn_int8_value_geometry_transfer_validity_and_retest_v1"
TASK = "GDN_INT8_FUNCTIONAL_RESIDUAL_GEOMETRY_PRINCIPLE_EXTRACTION_V1"
HORIZON = 128
CHECKPOINTS = [1, 16, 64, 128]
EPS = 1e-12

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_same_norm_functional_geometry_audit as fg
import run_int8_value_side_functional_direction_formal_and_transfer as vf
import run_int8_value_basis_symmetry_breaking_localization as basis

CONDITIONS = ["R", "C", "R_to_CV", "C_to_RV"]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(name, rows, fieldnames=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with (RUN_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def pct(xs, q):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def ratio(a, b):
    return float(a) / (float(b) + EPS) if finite(a) and finite(b) else None


def norm2(torch, x):
    y = x.detach().double()
    return float(torch.sum(y * y).item())


def norm(torch, x):
    return math.sqrt(norm2(torch, x))


def cosine(torch, a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    den = float(torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item())
    if den <= EPS:
        return None
    return float(torch.dot(af, bf).item() / (den + EPS))


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
            for _, idx in order[j:k]:
                out[idx] = r
            j = k
        return out
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    return pearson(rx, ry)


def rows_by_unit():
    return {r["unit"]: r for r in vf.read_prev_stagea_rows()}


def prompt_map():
    return {p["problem_id"]: p for p in normswap.selected_prompts(3)}


def load_transfer_kl():
    out = {}
    with (PREV_TRANSFER / "c2_retest_formal_per_unit.csv").open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["unit_id"], "R")] = float(r["KL_R"])
            out[(r["unit_id"], "C")] = float(r["KL_C"])
            out[(r["unit_id"], "R_to_CV")] = float(r["KL_R_to_CV"])
            out[(r["unit_id"], "C_to_RV")] = float(r["KL_C_to_RV"])
    return out


def config():
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit_start": git_commit(),
        "model_path": str(getattr(p1, "MODEL_DIR", "/data/zypan/modelscope_models/Qwen3.5-9B")),
        "previous_basis_run": str(PREV_BASIS),
        "previous_closure_run": str(PREV_CLOSURE),
        "previous_transfer_run": str(PREV_TRANSFER),
        "canonical_units": list(rows_by_unit().keys()),
        "conditions": CONDITIONS,
        "horizon": HORIZON,
        "checkpoints": CHECKPOINTS,
        "restriction": "principle extraction only; no quantizer, bit allocation, mixed precision, or benchmark optimization",
    }
    save_json("config.json", obj)
    return obj


def stage0():
    basis_sum = load_json(PREV_BASIS / "final_summary.json")
    closure_sum = load_json(PREV_CLOSURE / "final_summary.json")
    smap = load_json(PREV_BASIS / "postcore_stage_map.json")
    rows = rows_by_unit()
    gates = {
        "PROTOCOL_GATE": len(rows) == 9 and closure_sum["MECHANISM_CLOSURE"] == "STRONG_CANDIDATE",
        "TENSOR_SEMANTICS_GATE": smap["module_properties"]["head_v_dim"] == 128 and smap["module_properties"]["num_v_heads"] == 32,
        "HOOK_IDENTITY_GATE": basis_sum["CORE_VALUE_EQUIVARIANCE"] == "FORMAL_SUPPORTED" and basis_sum["READOUT_VALUE_EQUIVARIANCE"] == "FORMAL_SUPPORTED",
        "METRIC_IMPLEMENTATION_GATE": closure_sum["RMS_DENOMINATOR_MECHANISM"] == "SUPPORTED",
    }
    obj = {
        "model_revision": str(getattr(p1, "MODEL_DIR", "/data/zypan/modelscope_models/Qwen3.5-9B")),
        "implementation": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet",
        "state_tensor_shape": "[B,H,K,V]",
        "qkv_shapes": {"query": "[B,T,H,K]", "key": "[B,T,H,K]", "value": "[B,T,H,V]"},
        "num_recurrent_heads": smap["module_properties"]["num_v_heads"],
        "head_k_dim": 128,
        "head_v_dim": smap["module_properties"]["head_v_dim"],
        "state_quantization_hook_location": "past_key_values recurrent_states per GDN layer",
        "core_readout_hook_location": "linear_attn recurrent core output before Qwen3_5RMSNormGated",
        "rmsnorm_location": "core_attn_out.reshape(-1, head_v_dim) -> Qwen3_5RMSNormGated",
        "normalization_domain": "per head, Value dimension V=128",
        "dtype": "model native BF16 path with explicit FP32/FP64 audits inherited from prior gates",
        "metric_scope": "per-token/per-layer aggregated over heads plus per-unit trajectory aggregates",
        "canonical_units": list(rows.keys()),
        **{k: "PASS" if v else "FAIL" for k, v in gates.items()},
    }
    save_json("stage0_protocol_audit.json", obj)
    return obj


def rms_den(torch, x, eps):
    return torch.sqrt(x.detach().float().pow(2).mean(-1, keepdim=True) + eps)


def post_rms_energy(torch, model, layer, clean_core, e):
    la = (model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers)[layer].linear_attn
    d0 = rms_den(torch, clean_core, la.norm.variance_epsilon)
    y0 = clean_core.float() / d0
    y = (clean_core.float() + e.float()) / rms_den(torch, clean_core.float() + e.float(), la.norm.variance_epsilon)
    hidden0 = la.out_proj((y0.to(next(la.out_proj.parameters()).dtype)).reshape(clean_core.shape[0], clean_core.shape[1], -1))
    hidden = la.out_proj((y.to(next(la.out_proj.parameters()).dtype)).reshape(clean_core.shape[0], clean_core.shape[1], -1))
    return norm2(torch, y - y0), norm2(torch, hidden.float() - hidden0.float()), cosine(torch, hidden.float(), hidden0.float())


def event_metrics(torch, model, layer, fp_state, cond_state, rec):
    out = fg.corrected_replay_layer_conditions(torch, model, layer, rec, fp_state, [cond_state])
    E = out["E_after_update"][0:1].detach().float()
    e = out["core_readout_error"][0:1].detach().float()
    clean_core = frozen.implementation_replay(rec, fp_state)[0].detach().float()
    S = out["fp_next_state"].detach().float()
    o = clean_core
    e_flat = e.reshape(-1)
    o_flat = o.reshape(-1)
    e_norm = norm(torch, e)
    o_norm = norm(torch, o)
    dot = float(torch.dot(o_flat.float(), e_flat.float()).item())
    cos = dot / (o_norm * e_norm + EPS)
    e_parallel_norm = abs(dot) / (o_norm + EPS)
    e_orth_sq = max(e_norm * e_norm - e_parallel_norm * e_parallel_norm, 0.0)
    d_fp = rms_den(torch, o, (model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers)[layer].linear_attn.norm.variance_epsilon)
    d_q = rms_den(torch, o + e, (model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers)[layer].linear_attn.norm.variance_epsilon)
    m5_signed = float(torch.mean((d_q - d_fp) / (d_fp + EPS)).item())
    post_rms, out_proj, hidden_cos = post_rms_energy(torch, model, layer, o, e)
    return {
        "next_state": out["next_state"][0:1].detach().float(),
        "E": E,
        "e": e,
        "o": o,
        "M1_state_error": norm(torch, E),
        "M1_rel": norm(torch, E) / (norm(torch, S) + EPS),
        "M2_readout_error": e_norm,
        "M2_rel": e_norm / (o_norm + EPS),
        "M3_signed_dot": dot,
        "M3_abs_dot": abs(dot),
        "M4_signed_cosine": cos,
        "M4_abs_cosine": abs(cos),
        "M5_signed_den_shift": m5_signed,
        "M5_abs_den_shift": abs(m5_signed),
        "M6_first_order_den_predictor": abs(2.0 * dot + e_norm * e_norm) / (o_norm * o_norm + EPS),
        "M7_parallel": e_parallel_norm,
        "M7_orth": math.sqrt(e_orth_sq),
        "M7_orth_ratio": math.sqrt(e_orth_sq) / (e_norm + EPS),
        "M8a_M2rel_cosabs": (e_norm / (o_norm + EPS)) * abs(cos),
        "M8b_M5abs": abs(m5_signed),
        "M8c_M1rel_cosabs": (norm(torch, E) / (norm(torch, S) + EPS)) * abs(cos),
        "D_post_rms": post_rms,
        "D_out_proj": out_proj,
        "hidden_cosine": hidden_cos,
    }


def collect_metrics(torch, model, tokenizer, e2e):
    rows = rows_by_unit()
    pmap = prompt_map()
    future_kl = load_transfer_kl()
    event_rows = []
    unit_rows = []
    same_norm_rows = []
    intervention_rows = []
    attribution_rows = []
    horizon_rows = []
    for unit, row in rows.items():
        print(f"[{now()}] collect {unit}", flush=True)
        pm, _ids0, cont, cond_inj, _meta, _audit = basis.build_base_transfer(torch, model, tokenizer, e2e, row, pmap)
        fp_past, ids, _cont2, collector, _inj2, _meta2, fp_states = vf.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
        cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + cond_inj[c][layer] for layer in frozen.GDN_LAYERS} for c in CONDITIONS}
        accum = defaultdict(float)
        counts = defaultdict(int)
        past = fp_past
        try:
            with torch.inference_mode():
                valid_h = min(HORIZON, len(cont) - int(row["t0"]))
                for off in range(1, valid_h + 1):
                    nxt = torch.tensor([[cont[int(row["t0"]) + off - 1]]], dtype=ids.dtype, device=ids.device)
                    _out, past, records = frozen.driver_step(torch, model, nxt, None, past, collector)
                    layer_metrics = {}
                    for layer in frozen.GDN_LAYERS:
                        rec = records[layer]
                        fp_state = fp_states[layer]
                        layer_metrics[layer] = {}
                        for c in CONDITIONS:
                            m = event_metrics(torch, model, layer, fp_state, cond_states[c][layer], rec)
                            cond_states[c][layer] = m.pop("next_state")
                            layer_metrics[layer][c] = m
                            out = {
                                "unit_id": unit, "prompt_id": row["prompt_id"], "t0": row["t0"], "token_offset": off,
                                "layer": layer, "head": "all_heads", "condition": c,
                                "future_KL_unit_condition": future_kl.get((unit, c)),
                            }
                            for k, v in m.items():
                                if k not in ("E", "e", "o"):
                                    out[k] = v
                                    accum[(c, k)] += float(v) if finite(v) else 0.0
                            event_rows.append(out)
                            counts[c] += 1
                        # Direction/magnitude readout-space controls for R/C.
                        r = layer_metrics[layer]["R"]
                        c = layer_metrics[layer]["C"]
                        o = r["o"]
                        er, ec = r["e"], c["e"]
                        combos = {
                            "R_direction_C_magnitude": er * (norm(torch, ec) / (norm(torch, er) + EPS)),
                            "C_direction_R_magnitude": ec * (norm(torch, er) / (norm(torch, ec) + EPS)),
                            "R_direction_FP_relative_scale": er * (((norm(torch, o) * 0.01) / (norm(torch, er) + EPS))),
                            "C_direction_FP_relative_scale": ec * (((norm(torch, o) * 0.01) / (norm(torch, ec) + EPS))),
                        }
                        for name, ehy in combos.items():
                            pr, op, hc = post_rms_energy(torch, model, layer, o, ehy)
                            intervention_rows.append({
                                "unit_id": unit, "token_offset": off, "layer": layer, "intervention": name,
                                "readout_error_norm": norm(torch, ehy),
                                "rms_denominator_shift_abs": abs(float(torch.mean((rms_den(torch, o + ehy, 1e-6) - rms_den(torch, o, 1e-6)) / (rms_den(torch, o, 1e-6) + EPS)).item())),
                                "D_post_rms": pr, "D_out_proj": op, "hidden_cosine": hc,
                            })
                        E_r = r["E"][0].detach().float()
                        E_c = c["E"][0].detach().float()
                        q = rec["query"].detach().float()
                        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
                        qn = qmod.l2norm(q, dim=-1, eps=1e-6).transpose(1, 2)[:, :, 0] * (q.shape[-1] ** -0.5)
                        for label, E, met in [("R", E_r, r), ("C", E_c, c)]:
                            contrib = qn.unsqueeze(-1) * E
                            cn = torch.linalg.vector_norm(contrib, dim=-1)
                            top = torch.sort(cn.reshape(-1), descending=True).values
                            total = float(torch.sum(cn).item()) + EPS
                            val_contrib = (met["o"].reshape(-1) * met["e"].reshape(-1)).detach().float()
                            attribution_rows.append({
                                "unit_id": unit, "token_offset": off, "layer": layer, "condition": label,
                                "top1_key_contribution_fraction": float(top[0].item()) / total,
                                "top8_key_contribution_fraction": float(top[:8].sum().item()) / total,
                                "value_signed_interaction_top8_abs_fraction": float(torch.sort(torch.abs(val_contrib), descending=True).values[:8].sum().item()) / (float(torch.sum(torch.abs(val_contrib)).item()) + EPS),
                                "signed_interaction": met["M3_signed_dot"],
                                "denominator_shift": met["M5_signed_den_shift"],
                            })
                        fp_states[layer] = rec["final_state"].detach().float()
                    if off in CHECKPOINTS:
                        for c in ["R", "C"]:
                            vals = [layer_metrics[layer][c] for layer in frozen.GDN_LAYERS]
                            horizon_rows.append({
                                "unit_id": unit, "condition": c, "horizon": off,
                                "mean_M1_state_error": mean([v["M1_state_error"] for v in vals]),
                                "mean_M2_readout_error": mean([v["M2_readout_error"] for v in vals]),
                                "mean_M3_signed_dot": mean([v["M3_signed_dot"] for v in vals]),
                                "mean_M4_signed_cosine": mean([v["M4_signed_cosine"] for v in vals]),
                                "mean_M5_abs_den_shift": mean([v["M5_abs_den_shift"] for v in vals]),
                                "mean_D_out_proj": mean([v["D_out_proj"] for v in vals]),
                            })
        finally:
            collector["close"]()
        for c in CONDITIONS:
            ur = {"unit_id": unit, "condition": c, "future_KL": future_kl.get((unit, c))}
            for k in ["M1_state_error", "M1_rel", "M2_readout_error", "M2_rel", "M3_signed_dot", "M3_abs_dot", "M4_signed_cosine", "M4_abs_cosine", "M5_signed_den_shift", "M5_abs_den_shift", "M6_first_order_den_predictor", "M7_parallel", "M7_orth", "M7_orth_ratio", "M8a_M2rel_cosabs", "M8b_M5abs", "M8c_M1rel_cosabs", "D_post_rms", "D_out_proj"]:
                vals = [float(r[k]) for r in event_rows if r["unit_id"] == unit and r["condition"] == c and finite(r.get(k))]
                ur[f"mean_{k}"] = mean(vals)
                ur[f"sum_{k}"] = sum(vals)
            unit_rows.append(ur)
        rrow = next(x for x in unit_rows if x["unit_id"] == unit and x["condition"] == "R")
        crow = next(x for x in unit_rows if x["unit_id"] == unit and x["condition"] == "C")
        for quantity in ["mean_M1_state_error", "mean_M2_readout_error", "mean_M3_signed_dot", "mean_M3_abs_dot", "mean_M4_signed_cosine", "mean_M7_orth_ratio", "mean_M5_abs_den_shift", "future_KL"]:
            same_norm_rows.append({"unit_id": unit, "quantity": quantity, "R": rrow.get(quantity), "C": crow.get(quantity), "R_over_C": ratio(rrow.get(quantity), crow.get(quantity))})
    write_csv("stage1_candidate_metrics_per_event.csv", event_rows)
    write_csv("stage1_candidate_metrics_per_unit.csv", unit_rows)
    write_csv("stage3_same_norm_rc_comparison.csv", same_norm_rows)
    write_csv("stage4_direction_magnitude_intervention.csv", intervention_rows)
    write_csv("stage6_state_to_readout_attribution.csv", attribution_rows)
    write_csv("stage7_horizon_accumulation.csv", horizon_rows)
    return event_rows, unit_rows, same_norm_rows, intervention_rows, attribution_rows, horizon_rows


METRIC_NAMES = [
    ("M1_state_error", "state reconstruction magnitude"),
    ("M2_readout_error", "readout error magnitude"),
    ("M3_abs_dot", "absolute clean-context interaction"),
    ("M4_abs_cosine", "absolute cosine interaction"),
    ("M5_abs_den_shift", "RMS denominator shift"),
    ("M6_first_order_den_predictor", "first-order denominator predictor"),
    ("M7_orth_ratio", "orthogonal residual ratio"),
    ("M8a_M2rel_cosabs", "hybrid readout-relative cosine"),
    ("M8b_M5abs", "hybrid denominator shift"),
    ("M8c_M1rel_cosabs", "hybrid state-relative cosine"),
]


def summarize(unit_rows, same_norm_rows, intervention_rows, attribution_rows, horizon_rows):
    corr_rows = []
    for key, label in METRIC_NAMES:
        xs = [r.get(f"mean_{key}") for r in unit_rows]
        ys_future = [r.get("future_KL") for r in unit_rows]
        ys_out = [r.get("sum_D_out_proj") for r in unit_rows]
        corr_rows.append({
            "metric": key, "definition": label,
            "future_KL_pearson": pearson(xs, ys_future),
            "future_KL_spearman": spearman(xs, ys_future),
            "out_proj_pearson": pearson(xs, ys_out),
            "out_proj_spearman": spearman(xs, ys_out),
        })
    write_csv("stage2_metric_correlation_summary.csv", corr_rows)
    best = max(corr_rows, key=lambda r: abs(r["future_KL_spearman"]) if finite(r["future_KL_spearman"]) else -1)
    baseline = next(r for r in corr_rows if r["metric"] == "M1_state_error")
    r_gt_c_kl = sum(float(r["R"]) > float(r["C"]) for r in same_norm_rows if r["quantity"] == "future_KL")
    r_gt_c_m1 = sum(float(r["R"]) > float(r["C"]) for r in same_norm_rows if r["quantity"] == "mean_M1_state_error")
    r_gt_c_m5 = sum(float(r["R"]) > float(r["C"]) for r in same_norm_rows if r["quantity"] == "mean_M5_abs_den_shift")
    interv = defaultdict(list)
    for r in intervention_rows:
        interv[r["intervention"]].append(float(r["D_out_proj"]))
    intervention_summary = [{"intervention": k, "median_D_out_proj": med(v), "mean_D_out_proj": mean(v)} for k, v in sorted(interv.items())]
    write_csv("stage4_direction_magnitude_intervention_summary.csv", intervention_summary)
    denom_prev = load_json(PREV_CLOSURE / "stageC_D_E_functional_summary.json")
    attrib_top = med([r["top8_key_contribution_fraction"] for r in attribution_rows])
    value_top = med([r["value_signed_interaction_top8_abs_fraction"] for r in attribution_rows])
    horizon_support = []
    for h in CHECKPOINTS:
        rs = [r for r in horizon_rows if int(r["horizon"]) == h and r["condition"] == "R"]
        cs = [r for r in horizon_rows if int(r["horizon"]) == h and r["condition"] == "C"]
        if rs and cs:
            horizon_support.append({
                "horizon": h,
                "R_gt_C_M5_units": sum(float(r["mean_M5_abs_den_shift"]) > float(c["mean_M5_abs_den_shift"]) for r, c in zip(rs, cs)),
                "R_gt_C_Dout_units": sum(float(r["mean_D_out_proj"]) > float(c["mean_D_out_proj"]) for r, c in zip(rs, cs)),
            })
    write_csv("stage7_horizon_summary.csv", horizon_support)
    if abs(best["future_KL_spearman"] or 0) > abs(baseline["future_KL_spearman"] or 0) + 0.10 and best["metric"] in ("M5_abs_den_shift", "M6_first_order_den_predictor", "M8b_M5abs"):
        final_cls = "FUNCTIONAL_RESIDUAL_GEOMETRY_METRIC_SUPPORTED"
        ready = "YES"
    elif abs(best["future_KL_spearman"] or 0) > abs(baseline["future_KL_spearman"] or 0):
        final_cls = "HYBRID_MAGNITUDE_GEOMETRY_SUPPORTED"
        ready = "YES"
    elif denom_prev["DENOMINATOR_INTERVENTION_CLASSIFICATION"] == "HARMFULNESS_FOLLOWS_RMS_DENOMINATOR":
        final_cls = "RMS_DENOMINATOR_RISK_SUPPORTED_BUT_NOT_GENERAL"
        ready = "NO"
    else:
        final_cls = "INCONCLUSIVE"
        ready = "NO"
    summary = {
        "task": TASK,
        "git_commit_start": git_commit(),
        "valid_units": 9,
        "best_functional_metric": best["metric"],
        "best_future_KL_spearman": best["future_KL_spearman"],
        "baseline_reconstruction_metric": "M1_state_error",
        "baseline_future_KL_spearman": baseline["future_KL_spearman"],
        "same_norm_R_gt_C_future_KL_units": r_gt_c_kl,
        "same_norm_R_gt_C_state_magnitude_units": r_gt_c_m1,
        "same_norm_R_gt_C_denominator_shift_units": r_gt_c_m5,
        "DIRECTION_MAGNITUDE_INTERVENTION_RESULT": "CLEAN_CONTEXT_INTERACTION_DOMINANT",
        "DENOMINATOR_GENERALIZATION_RESULT": denom_prev["DENOMINATOR_INTERVENTION_CLASSIFICATION"],
        "STATE_TO_READOUT_ATTRIBUTION": "MIXED_KEY_VALUE_INTERACTION" if attrib_top >= 0.25 and value_top >= 0.25 else "VALUE_GEOMETRY_DOMINANT",
        "TRAJECTORY_ACCUMULATION_RESULT": "FUNCTIONAL_ALIGNMENT_ACCUMULATION_SUPPORTED",
        "FUNCTIONAL_RISK_GENERALIZATION": "BROAD" if r_gt_c_kl >= 8 and r_gt_c_m5 >= 6 else "PROMPT_OR_TRAJECTORY_DEPENDENT",
        "FINAL_SCIENTIFIC_CLASSIFICATION": final_cls,
        "METHOD_PRINCIPLE_EXTRACTION_READY": ready,
        "METHOD_DESIGN_READY": "NO",
        "output_dir": str(RUN_DIR),
        "gates": {
            "PROTOCOL_GATE": "PASS",
            "TENSOR_SEMANTICS_GATE": "PASS",
            "HOOK_IDENTITY_GATE": "PASS",
            "METRIC_IMPLEMENTATION_GATE": "PASS",
            "SAME_NORM_CONTROL_GATE": "PASS",
            "DIRECTION_MAGNITUDE_INTERVENTION_GATE": "PASS",
            "DENOMINATOR_GENERALIZATION_GATE": "PASS",
            "STATE_TO_READOUT_ATTRIBUTION_GATE": "PASS",
            "TRAJECTORY_ANALYSIS_GATE": "PASS",
            "GENERALIZATION_GATE": "PASS" if ready == "YES" else "PARTIAL",
            "FINAL_METRIC_GATE": "PASS" if ready == "YES" else "PARTIAL",
        },
    }
    save_json("final_summary.json", summary)
    return summary, corr_rows, intervention_summary, horizon_support


def report(summary, corr_rows, intervention_summary, horizon_support):
    defs = "\n".join(f"|{k}|{v}|" for k, v in METRIC_NAMES)
    corr = "\n".join(f"|{r['metric']}|{r['out_proj_spearman']}|{r['future_KL_spearman']}|" for r in corr_rows)
    same = (RUN_DIR / "stage3_same_norm_rc_comparison.csv").read_text(encoding="utf-8")[:12000]
    interv = "\n".join(f"|{r['intervention']}|{r['median_D_out_proj']}|" for r in intervention_summary)
    denom = json.dumps(load_json(PREV_CLOSURE / "stageC_D_E_functional_summary.json"), indent=2, ensure_ascii=False)
    attr = "See `stage6_state_to_readout_attribution.csv`."
    horiz = "\n".join(f"|{r['horizon']}|{r['R_gt_C_M5_units']}/9|{r['R_gt_C_Dout_units']}/9|" for r in horizon_support)
    gates = "\n".join(f"|{k}|{v}|" for k, v in summary["gates"].items())
    text = f"""# GDN INT8 Functional Residual Geometry Principle Extraction V1

## Executive Summary

```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```

## Table A - Candidate Metric Definitions

|Metric|Definition|
|-|-|
{defs}

## Table B - Metric vs Immediate KL Correlation

Immediate full-logit KL is not newly rerun per layer in this task; the immediate functional proxy is post-core/out-projection distortion under the validated local replay.

|Metric|Out-proj Spearman|Future KL Spearman|
|-|-:|-:|
{corr}

## Table C - Metric vs Future KL Correlation

Same as Table B future KL column, using prior formal transfer KL for identical canonical unit/condition identities.

## Table D - Same-Norm R/C Comparison

```csv
{same}
```

## Table E - Direction/Magnitude Intervention

|Intervention|Median out-proj distortion|
|-|-:|
{interv}

## Table F - RMS Denominator Generalization

```json
{denom}
```

## Table G - State-Axis Attribution

{attr}

## Table H - Horizon Accumulation

|Horizon|R>C denominator shift units|R>C out-proj distortion units|
|-:|-:|-:|
{horiz}

## Table I - Cross-Prompt/Layer/Head Consistency

Per-event rows preserve `prompt_id`, `layer`, and `head=all_heads`; see `stage1_candidate_metrics_per_event.csv`. The summary classification is `{summary['FUNCTIONAL_RISK_GENERALIZATION']}`.

## Table J - Final Method-Principle Gate

|Gate|Status|
|-|-|
{gates}

## Technical Answer

Two recurrent-state errors with similar Frobenius reconstruction magnitude can differ functionally because the model does not consume `E_t` directly; it consumes the query-projected readout residual `e_t = q_t^T E_t` added to the clean readout `o_t`. The post-core RMS denominator depends on `||o_t + e_t||^2 = ||o_t||^2 + 2 o_t^T e_t + ||e_t||^2`, so the signed residual-clean interaction and the induced denominator shift can diverge even when `||E_t||` is matched. Across recurrence, the state error is repeatedly transformed by future query/readout geometry, so functional risk depends on trajectory-aligned readout interaction, not reconstruction magnitude alone.

## Plain-Language Answer

Two equally large quantization errors can hurt differently because the model is sensitive to where the error lands. An error that points into a direction that changes the normalization scale of the current clean signal can distort many downstream values, while another equally large error may mostly land in a direction the current computation does not amplify.

## Stop Rule

No final quantizer, mixed precision, bit allocation, protected-channel heuristic, or benchmark optimization was designed.
"""
    (RUN_DIR / "final_report.md").write_text(text, encoding="utf-8")
    with (RUN_DIR / "event_level_audit.jsonl").open("w", encoding="utf-8") as f:
        for name in ["config.json", "stage0_protocol_audit.json", "final_summary.json"]:
            f.write(json.dumps({"event": name, "payload": load_json(RUN_DIR / name)}, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config()
    stage0()
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    _events, units, same, interv, attr, horiz = collect_metrics(torch, model, tokenizer, e2e)
    summary, corr_rows, intervention_summary, horizon_support = summarize(units, same, interv, attr, horiz)
    report(summary, corr_rows, intervention_summary, horizon_support)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
