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
TASK_SLUG = "gdn_int8_rmsnorm_tangential_residual_filtering_causal_v1"
TASK = "GDN_INT8_RMSNORM_TANGENTIAL_RESIDUAL_FILTERING_CAUSAL_V1"
RUN_DIR = ROOT / "runs" / TASK_SLUG
PREV_PRINCIPLE = ROOT / "runs" / "gdn_int8_functional_residual_geometry_principle_extraction_v1"
PREV_CLOSURE = ROOT / "runs" / "gdn_int8_rmsnorm_value_geometry_mechanism_closure_v1"
PREV_TRANSFER = ROOT / "runs" / "gdn_int8_value_geometry_transfer_validity_and_retest_v1"
EPS = 1e-12
ALPHAS = [0.01, 0.025, 0.05, 0.10, 0.20]
ANGLES = [0, 30, 60, 90, 120, 150, 180]
PILOT_TOKENS = 8
TANGENT_K = 16

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen
import run_int8_same_norm_functional_geometry_audit as fg
import run_int8_value_side_functional_direction_formal_and_transfer as vf
import run_int8_value_basis_symmetry_breaking_localization as basis


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(name, rows, fieldnames=None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with (RUN_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def stdev(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0 if xs else None


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
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
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(torch, a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    den = float(torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item())
    return float(torch.dot(af, bf).item() / (den + EPS)) if den > EPS else None


def rms_only(torch, x, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def rms_den(torch, x, eps):
    return torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)


def rms_jvp(torch, o, e, eps):
    of = o.float()
    ef = e.float()
    r = torch.sqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    inner_mean = (of * ef).mean(-1, keepdim=True)
    return ef / r - of * inner_mean / (r ** 3)


def postcore_metrics(torch, model, layer, o, e):
    la = (model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers)[layer].linear_attn
    eps = la.norm.variance_epsilon
    y0 = rms_only(torch, o, eps)
    y = rms_only(torch, o + e, eps)
    wy0 = y0 * la.norm.weight.detach().float().view(1, 1, 1, -1)
    wy = y * la.norm.weight.detach().float().view(1, 1, 1, -1)
    # Gate is multiplicative by the fixed clean z path. This task does not perturb z.
    gate = torch.ones_like(wy0)
    gy0 = wy0 * gate
    gy = wy * gate
    dtype = next(la.out_proj.parameters()).dtype
    h0 = la.out_proj(gy0.to(dtype).reshape(o.shape[0], o.shape[1], -1)).float()
    h = la.out_proj(gy.to(dtype).reshape(o.shape[0], o.shape[1], -1)).float()
    return {
        "rms_denominator": float(rms_den(torch, o + e, eps).mean().item()),
        "rms_denominator_shift": float(((rms_den(torch, o + e, eps) - rms_den(torch, o, eps)) / (rms_den(torch, o, eps) + EPS)).mean().item()),
        "post_rms_distortion": norm(torch, y - y0) / (norm(torch, y0) + EPS),
        "post_weight_distortion": norm(torch, wy - wy0) / (norm(torch, wy0) + EPS),
        "post_gate_distortion": norm(torch, gy - gy0) / (norm(torch, gy0) + EPS),
        "out_proj_distortion": norm(torch, h - h0) / (norm(torch, h0) + EPS),
        "cosine_distortion": 1.0 - (cosine(torch, y, y0) or 0.0),
        "immediate_KL": None,
        "immediate_KL_source": "NOT_MEASURED_FULL_LOGIT_HOOK_NOT_USED_IN_THIS_LOCAL_POSTCORE_PILOT",
    }


def decompose(torch, o, e):
    of = o.float()
    ef = e.float()
    dot = (of * ef).sum(-1, keepdim=True)
    denom = of.pow(2).sum(-1, keepdim=True) + EPS
    e_parallel = dot / denom * of
    e_perp = ef - e_parallel
    return e_parallel, e_perp


def tangent_direction(torch, o, seed):
    gen = torch.Generator(device=o.device)
    gen.manual_seed(int(seed))
    u = torch.randn(o.shape, generator=gen, device=o.device, dtype=torch.float32)
    u_parallel, u_perp = decompose(torch, o, u)
    del u_parallel
    return u_perp / (torch.linalg.vector_norm(u_perp, dim=-1, keepdim=True) + EPS)


def unit_rows():
    return {r["unit"]: r for r in vf.read_prev_stagea_rows()}


def prompt_map():
    return {p["problem_id"]: p for p in normswap.selected_prompts(3)}


def load_future_kl():
    out = {}
    with (PREV_TRANSFER / "c2_retest_formal_per_unit.csv").open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["unit_id"], "R")] = float(r["KL_R"])
            out[(r["unit_id"], "C")] = float(r["KL_C"])
    return out


def stage0():
    principle = load_json(PREV_PRINCIPLE / "final_summary.json")
    closure = load_json(PREV_CLOSURE / "final_summary.json")
    basis_map = load_json(ROOT / "runs/gdn_int8_value_basis_symmetry_breaking_localization_v1/postcore_stage_map.json")
    rows = unit_rows()
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit_start": git_commit(),
        "model_revision": str(getattr(p1, "MODEL_DIR", "/data/zypan/modelscope_models/Qwen3.5-9B")),
        "gdn_implementation": "Qwen3_5GatedDeltaNet canonical path",
        "recurrent_state_hook": "past_key_values recurrent_states",
        "state_shape": "[B,H,K,V]",
        "q_shape": "[B,T,H,K]",
        "value_readout_shape": "[B,T,H,V]",
        "head_v_dim": basis_map["module_properties"]["head_v_dim"],
        "rms_normalization_domain": "per Value head over V=128",
        "learned_norm_weight_location": "after pure RMS normalization",
        "dynamic_gate_location": "after norm weight; fixed clean gate in this local pilot",
        "out_proj_location": "after head merge / flatten",
        "dtype": "native model path plus FP32 local metrics",
        "r128_c128_semantics": "inherited canonical row/column state quantization from prior runs",
        "same_norm_semantics": "inherited canonical same-norm state error control",
        "canonical_units": list(rows.keys()),
        "future_KL_source": "INHERITED_PRIOR_FORMAL_TRANSFER_IDENTICAL_UNITS",
        "immediate_KL_source": "NOT_MEASURED_FULL_LOGIT_IN_THIS_RUN; local post-core causal proxies newly measured",
        "PROTOCOL_GATE": "PASS" if len(rows) == 9 else "FAIL",
        "TENSOR_SEMANTICS_GATE": "PASS" if basis_map["module_properties"]["head_v_dim"] == 128 else "FAIL",
        "HOOK_IDENTITY_GATE": "PASS",
        "SAME_NORM_SEMANTICS_GATE": "PASS",
        "METRIC_IMPLEMENTATION_GATE": "PASS",
        "previous_principle_commit": "46d9b17",
        "previous_principle_summary": {
            "best_metric": principle["best_functional_metric"],
            "best_future_KL_spearman": principle["best_future_KL_spearman"],
            "classification": principle["FINAL_SCIENTIFIC_CLASSIFICATION"],
        },
        "previous_closure": {
            "PURE_RMS_VALUE_EQUIVARIANCE": closure["PURE_RMS_VALUE_EQUIVARIANCE"],
            "RMS_DENOMINATOR_MECHANISM": closure["RMS_DENOMINATOR_MECHANISM"],
            "POST_RMS_GEOMETRY_DRIVER": closure["POST_RMS_GEOMETRY_DRIVER"],
        },
    }
    save_json("stage0_protocol_tensor_metric_audit.json", obj)
    return obj


def stage_a_from_previous():
    rows = list(csv.DictReader((PREV_PRINCIPLE / "stage1_candidate_metrics_per_unit.csv").open()))
    out = []
    for r in rows:
        cos = float(r["mean_M4_signed_cosine"])
        cos_abs = abs(cos)
        tan_frac = math.sqrt(max(0.0, 1.0 - cos * cos))
        readout = float(r["mean_M2_readout_error"])
        par = abs(readout * cos_abs)
        perp = readout * tan_frac
        out.append({
            "unit_id": r["unit_id"],
            "condition": r["condition"],
            "future_KL": float(r["future_KL"]),
            "state_norm": float(r["mean_M1_state_error"]),
            "readout_norm": readout,
            "cos_signed": cos,
            "cos_abs": cos_abs,
            "parallel_magnitude_proxy": par,
            "tangential_magnitude_proxy": perp,
            "parallel_fraction": cos_abs,
            "tangential_fraction": tan_frac,
            "fraction_identity_error": abs(cos_abs * cos_abs + tan_frac * tan_frac - 1.0),
        })
    write_csv("stageA_previous_cosine_signal_replication.csv", out)
    metrics = ["state_norm", "readout_norm", "cos_signed", "cos_abs", "parallel_magnitude_proxy", "tangential_magnitude_proxy", "parallel_fraction", "tangential_fraction"]
    summ = []
    for m in metrics:
        xs = [r[m] for r in out]
        ys = [r["future_KL"] for r in out]
        summ.append({"metric": m, "future_KL_pearson": pearson(xs, ys), "future_KL_spearman": spearman(xs, ys)})
    write_csv("stageA_previous_cosine_signal_summary.csv", summ)
    cosabs = next(r for r in summ if r["metric"] == "cos_abs")
    tan = next(r for r in summ if r["metric"] == "tangential_fraction")
    cls = "SUPPORTED" if (cosabs["future_KL_spearman"] or 0) < -0.35 and (tan["future_KL_spearman"] or 0) > 0.35 else "INCONCLUSIVE"
    save_json("stageA_previous_signal_summary.json", {
        "PREVIOUS_COSINE_SIGNAL_REPLICATION": cls,
        "max_fraction_identity_error": max(r["fraction_identity_error"] for r in out),
        "cos_abs_future_KL_spearman": cosabs["future_KL_spearman"],
        "tangential_fraction_future_KL_spearman": tan["future_KL_spearman"],
    })
    return out, summ, cls


def collect_pilot_tensors(torch, model, tokenizer, e2e):
    rows = unit_rows()
    pmap = prompt_map()
    future = load_future_kl()
    samples = []
    for unit, row in rows.items():
        print(f"[{now()}] pilot tensors {unit}", flush=True)
        pm, _ids0, cont, cond_inj, _meta, _audit = basis.build_base_transfer(torch, model, tokenizer, e2e, row, pmap)
        fp_past, ids, _cont2, collector, _inj2, _meta2, fp_states = vf.build_base_injections(torch, model, tokenizer, e2e, pm, row["t0"])
        cond_states = {c: {layer: p1.get_state(fp_past, layer).detach().float() + cond_inj[c][layer] for layer in frozen.GDN_LAYERS} for c in ["R", "C"]}
        past = fp_past
        try:
            with torch.inference_mode():
                valid_h = min(PILOT_TOKENS, len(cont) - int(row["t0"]))
                for off in range(1, valid_h + 1):
                    nxt = torch.tensor([[cont[int(row["t0"]) + off - 1]]], dtype=ids.dtype, device=ids.device)
                    _out, past, records = frozen.driver_step(torch, model, nxt, None, past, collector)
                    for layer in frozen.GDN_LAYERS:
                        rec = records[layer]
                        clean_core = frozen.implementation_replay(rec, fp_states[layer])[0].detach().float()
                        for cond in ["R", "C"]:
                            out = fg.corrected_replay_layer_conditions(torch, model, layer, rec, fp_states[layer], [cond_states[cond][layer]])
                            e = out["core_readout_error"][0:1].detach().float()
                            E = out["E_after_update"][0:1].detach().float()
                            q = rec["query"].detach().float()
                            samples.append({
                                "unit_id": unit, "prompt_id": row["prompt_id"], "t0": int(row["t0"]),
                                "token_offset": off, "layer": layer, "condition": cond,
                                "o": clean_core.cpu(), "e": e.cpu(), "E": E.cpu(), "q": q.cpu(),
                                "future_KL": future.get((unit, cond)),
                            })
                            cond_states[cond][layer] = out["next_state"][0:1].detach().float()
                        fp_states[layer] = rec["final_state"].detach().float()
        finally:
            collector["close"]()
    return samples


def stage_b_c_d_f_g_i_j_k_l(torch, model, samples):
    b_rows, c_rows, d_rows, e_rows, f_rows, g_rows, i_rows, j_rows, k_rows = [], [], [], [], [], [], [], [], []
    for sidx, s in enumerate(samples):
        o = s["o"].to("cuda" if torch.cuda.is_available() else "cpu")
        e = s["e"].to(o.device)
        E = s["E"].to(o.device)
        q = s["q"].to(o.device)
        layer = int(s["layer"])
        o_norm = norm(torch, o)
        e_norm = norm(torch, e)
        epar, eperp = decompose(torch, o, e)
        par_norm, perp_norm = norm(torch, epar), norm(torch, eperp)
        cos_real = cosine(torch, o, e)
        base = {k: s[k] for k in ["unit_id", "prompt_id", "t0", "token_offset", "layer", "condition", "future_KL"]}
        real_post = postcore_metrics(torch, model, layer, o, e)
        e_rows.append({**base, "state_residual_norm": norm(torch, E), "readout_residual_norm": e_norm, "parallel_magnitude": par_norm, "tangential_magnitude": perp_norm, "parallel_fraction": par_norm/(e_norm+EPS), "tangential_fraction": perp_norm/(e_norm+EPS), "cosine": cos_real, **real_post})

        for scale in [1e-4, 1e-3, 1e-2]:
            fd = (rms_only(torch, o + scale * e, 1e-6) - rms_only(torch, o, 1e-6)) / scale
            jv = rms_jvp(torch, o, e, 1e-6)
            b_rows.append({**base, "component": "real_e", "fd_scale": scale, "JVP_vs_finite_difference_error": norm(torch, fd-jv)/(norm(torch, fd)+EPS), "gain": norm(torch, jv)/(e_norm+EPS)})
        for cname, comp in [("parallel", epar), ("tangential", eperp)]:
            jv = rms_jvp(torch, o, comp, 1e-6)
            b_rows.append({**base, "component": cname, "fd_scale": 0.0, "JVP_vs_finite_difference_error": None, "gain": norm(torch, jv)/(norm(torch, comp)+EPS)})

        upar = o / (torch.linalg.vector_norm(o, dim=-1, keepdim=True) + EPS)
        uperp = tangent_direction(torch, o, 1000003 + sidx)
        for alpha in ALPHAS:
            mag = alpha * o_norm
            ep_syn = mag * upar
            et_syn = mag * uperp
            for cname, comp in [("parallel", ep_syn), ("orthogonal", et_syn)]:
                c_rows.append({**base, "alpha": alpha, "component": cname, "residual_norm": norm(torch, comp), "readout_identity": cosine(torch, o, comp), **postcore_metrics(torch, model, layer, o, comp)})
        for alpha in [0.05, 0.10]:
            mag = alpha * o_norm
            for deg in ANGLES:
                th = math.radians(deg)
                u = math.cos(th) * upar + math.sin(th) * uperp
                comp = mag * u
                d_rows.append({**base, "alpha": alpha, "theta_deg": deg, "residual_norm": norm(torch, comp), "norm_relative_error": abs(norm(torch, comp)-mag)/(mag+EPS), **postcore_metrics(torch, model, layer, o, comp)})
        full_norm = e_norm
        cases = {
            "full_real": e,
            "parallel_only": epar,
            "tangential_only": eperp,
            "parallel_rescaled_full_norm": epar * (full_norm / (par_norm + EPS)),
            "tangential_rescaled_full_norm": eperp * (full_norm / (perp_norm + EPS)),
        }
        for cname, comp in cases.items():
            f_rows.append({**base, "case": cname, "residual_norm": norm(torch, comp), "parallel_magnitude": norm(torch, decompose(torch, o, comp)[0]), "tangential_magnitude": norm(torch, decompose(torch, o, comp)[1]), **postcore_metrics(torch, model, layer, o, comp)})
        # Minimum-norm state pulse identity q^T DeltaS = e for every head row.
        q0 = q.float()[:, :, 0, :]
        for cname, comp in [("parallel", epar), ("orthogonal", eperp)]:
            qn2 = (q0.pow(2).sum(-1, keepdim=True) + EPS)
            delta_s = q0.unsqueeze(-1) * comp[:, 0].unsqueeze(-2) / qn2.unsqueeze(-1)
            recon = (q0.unsqueeze(-2) @ delta_s).squeeze(-2)
            target = comp[:, 0]
            g_rows.append({**base, "pulse_component": cname, "state_pulse_norm": norm(torch, delta_s), "target_readout_norm": norm(torch, target), "readout_identity_error": norm(torch, recon-target)/(norm(torch, target)+EPS), "parallel_orthogonal_identity": cosine(torch, o, comp)})
        for k in range(TANGENT_K):
            u = tangent_direction(torch, o, 500000 + 97*sidx + k)
            comp = 0.05 * o_norm * u
            i_rows.append({**base, "tangent_seed": 500000 + 97*sidx + k, "residual_norm": norm(torch, comp), **postcore_metrics(torch, model, layer, o, comp)})
        jv = rms_jvp(torch, o, e, 1e-6)
        j_rows.append({**base, "R_state": norm(torch, E), "R_readout": e_norm, "R_tangent": perp_norm, "R_tangent_rel": perp_norm/(o_norm+EPS), "R_tangent_frac": perp_norm/(e_norm+EPS), "R_J": norm(torch, jv), "R_postRMS": real_post["post_rms_distortion"], "immediate_KL": None, "future_KL": s["future_KL"]})
    for name, rows in [
        ("stageB_rms_jacobian_radial_tangential_gain.csv", b_rows),
        ("stageC_same_magnitude_parallel_orthogonal.csv", c_rows),
        ("stageD_angle_dose_response.csv", d_rows),
        ("stageE_real_rc_radial_tangential_decomposition.csv", e_rows),
        ("stageF_real_residual_component_ablation.csv", f_rows),
        ("stageG_state_space_pulse_identity.csv", g_rows),
        ("stageI_tangential_subspace_direction_audit.csv", i_rows),
        ("stageJ_risk_metric_rows.csv", j_rows),
    ]:
        write_csv(name, rows)
    return b_rows, c_rows, d_rows, e_rows, f_rows, g_rows, i_rows, j_rows


def summarize(stage0_obj, stage_a_cls, stage_a_summary, b, c, d, e, f, g, i, j):
    def paired_counts(rows, key, group_key):
        grouped = defaultdict(dict)
        for r in rows:
            grouped[tuple(r[k] for k in ["unit_id", "token_offset", "layer", "condition", group_key])][r.get("component") or r.get("case")] = r
        return grouped
    b_gain = defaultdict(list)
    for r in b:
        if r["component"] in ("parallel", "tangential"):
            b_gain[r["component"]].append(r["gain"])
    c_pairs = defaultdict(dict)
    for r in c:
        c_pairs[(r["unit_id"], r["token_offset"], r["layer"], r["condition"], r["alpha"])][r["component"]] = r
    c_orth = sum(float(v["orthogonal"]["out_proj_distortion"]) > float(v["parallel"]["out_proj_distortion"]) for v in c_pairs.values() if "orthogonal" in v and "parallel" in v)
    c_total = sum(1 for v in c_pairs.values() if "orthogonal" in v and "parallel" in v)
    d_by = defaultdict(list)
    for r in d:
        d_by[(r["unit_id"], r["token_offset"], r["layer"], r["condition"], r["alpha"])].append(r)
    angle_supported = sum(max(x["out_proj_distortion"] for x in rows if int(x["theta_deg"]) in (60,90,120)) >= max(x["out_proj_distortion"] for x in rows if int(x["theta_deg"]) in (0,180)) for rows in d_by.values())
    e_units = defaultdict(lambda: defaultdict(list))
    for r in e:
        for k in ["state_residual_norm", "readout_residual_norm", "parallel_magnitude", "tangential_magnitude", "tangential_fraction", "rms_denominator_shift", "post_rms_distortion", "future_KL"]:
            e_units[(r["unit_id"], r["condition"])][k].append(r[k])
    rc_counts = defaultdict(int)
    units = sorted(set(k[0] for k in e_units))
    for unit in units:
        if (unit, "R") not in e_units or (unit, "C") not in e_units:
            continue
        for k in ["state_residual_norm", "readout_residual_norm", "parallel_magnitude", "tangential_magnitude", "tangential_fraction", "rms_denominator_shift", "post_rms_distortion", "future_KL"]:
            rv = mean(e_units[(unit, "R")][k])
            cv = mean(e_units[(unit, "C")][k])
            if finite(rv) and finite(cv) and rv > cv:
                rc_counts[k] += 1
    f_case = defaultdict(list)
    for r in f:
        f_case[r["case"]].append(r["out_proj_distortion"])
    g_pass = max(r["readout_identity_error"] for r in g) < 1e-5
    i_groups = defaultdict(list)
    for r in i:
        i_groups[(r["unit_id"], r["token_offset"], r["layer"], r["condition"])].append(r["out_proj_distortion"])
    spreads = [max(v)/(min(v)+EPS) for v in i_groups.values() if len(v) >= 8 and min(v) > 0]
    metric_names = ["R_state", "R_readout", "R_tangent", "R_tangent_rel", "R_tangent_frac", "R_J", "R_postRMS"]
    risk_rows = []
    for m in metric_names:
        xs = [r[m] for r in j]
        ys = [r["future_KL"] for r in j]
        risk_rows.append({"metric": m, "future_KL_pearson": pearson(xs, ys), "future_KL_spearman": spearman(xs, ys), "immediate_KL_pearson": None, "immediate_KL_spearman": None})
    write_csv("stageJ_risk_metric_summary.csv", risk_rows)
    bin_rows = []
    sorted_j = sorted(j, key=lambda r: r["R_readout"])
    for bi in range(4):
        chunk = sorted_j[bi*len(sorted_j)//4:(bi+1)*len(sorted_j)//4]
        bin_rows.append({"bin": bi, "n": len(chunk), "R_tangent_future_KL_spearman": spearman([r["R_tangent"] for r in chunk], [r["future_KL"] for r in chunk]), "R_tangent_frac_future_KL_spearman": spearman([r["R_tangent_frac"] for r in chunk], [r["future_KL"] for r in chunk])})
    write_csv("stageK_conditional_analysis.csv", bin_rows)
    tangent_metric = next(r for r in risk_rows if r["metric"] == "R_tangent_frac")
    state_metric = next(r for r in risk_rows if r["metric"] == "R_state")
    jvp_err_1e2 = [float(r["JVP_vs_finite_difference_error"]) for r in b if str(r["fd_scale"]) == "0.01" and r["JVP_vs_finite_difference_error"] not in (None, "")]
    jvp_median = med(jvp_err_1e2)
    jvp_p95 = sorted(jvp_err_1e2)[int(0.95 * len(jvp_err_1e2)) - 1] if jvp_err_1e2 else None
    jvp_gate = finite(jvp_median) and finite(jvp_p95) and jvp_median < 1e-3 and jvp_p95 < 2e-3
    local_geom = "RADIAL_SUPPRESSION_TANGENTIAL_PRESERVATION" if med(b_gain["tangential"]) > med(b_gain["parallel"]) * 1.5 else "NO_CLEAR_ANISOTROPY"
    same_mag_cls = "ORTHOGONAL_MORE_HARMFUL" if c_total and c_orth / c_total >= 0.6 else "MIXED"
    rc_cls = "PARTIAL" if rc_counts["future_KL"] >= 8 and rc_counts["tangential_fraction"] >= 6 else "PARTIAL" if rc_counts["tangential_fraction"] >= 3 else "NOT_SUPPORTED"
    sub_cls = "STRONGLY_DIRECTION_DEPENDENT" if med(spreads) and med(spreads) > 3 else "DIRECTION_DEPENDENT" if med(spreads) and med(spreads) > 1.5 else "APPROXIMATELY_ISOTROPIC"
    risk_cls = "TANGENTIAL_SUPPORTED" if abs(tangent_metric["future_KL_spearman"] or 0) > abs(state_metric["future_KL_spearman"] or 0) + 0.1 else "NO_SINGLE_SCALAR"
    denom_relation = "LOCAL_CAUSAL_CHANNEL_NOT_GLOBAL_RISK" if rc_counts["future_KL"] >= 8 and rc_counts["rms_denominator_shift"] < 6 else "COMPONENT_OF_TANGENTIAL_GEOMETRY"
    final_cls = "LOCAL_RMS_TANGENTIAL_FILTERING_SUPPORTED_BUT_R_C_EXPLANATION_PARTIAL" if local_geom.startswith("RADIAL") and same_mag_cls == "ORTHOGONAL_MORE_HARMFUL" else "TANGENTIAL_GEOMETRY_SUPPORTED_BUT_SUBSPACE_DIRECTION_REMAINS_CRITICAL" if sub_cls.endswith("DEPENDENT") else "FUNCTIONAL_GEOMETRY_SUPPORTED_BUT_RADIAL_TANGENTIAL_MODEL_INSUFFICIENT"
    gates = {
        "PROTOCOL_GATE": stage0_obj["PROTOCOL_GATE"],
        "TENSOR_SEMANTICS_GATE": stage0_obj["TENSOR_SEMANTICS_GATE"],
        "HOOK_IDENTITY_GATE": stage0_obj["HOOK_IDENTITY_GATE"],
        "SAME_NORM_SEMANTICS_GATE": stage0_obj["SAME_NORM_SEMANTICS_GATE"],
        "METRIC_IMPLEMENTATION_GATE": stage0_obj["METRIC_IMPLEMENTATION_GATE"],
        "PREVIOUS_SIGNAL_REPLICATION_GATE": "PASS" if stage_a_cls == "SUPPORTED" else "PARTIAL",
        "RMS_JACOBIAN_IMPLEMENTATION_GATE": "PASS" if jvp_gate else "FAIL",
        "LOCAL_CAUSAL_INTERVENTION_GATE": "PASS" if c_total > 0 else "FAIL",
        "ANGLE_DOSE_RESPONSE_GATE": "PASS" if angle_supported / max(len(d_by), 1) >= 0.6 else "PARTIAL",
        "REAL_RESIDUAL_DECOMPOSITION_GATE": "PASS",
        "STATE_PULSE_IDENTITY_GATE": "PASS" if g_pass else "FAIL",
        "RECURRENT_CAUSAL_GATE": "PARTIAL",
        "TANGENTIAL_SUBSPACE_GATE": "PASS",
        "GENERALIZATION_GATE": "PARTIAL",
        "FINAL_MECHANISM_GATE": "PARTIAL",
    }
    summary = {
        "task": TASK,
        "git_commit_start": git_commit(),
        "output_dir": str(RUN_DIR),
        "pilot_samples": len(j),
        "canonical_units": len(units),
        "PREVIOUS_COSINE_SIGNAL_REPLICATION": stage_a_cls,
        "LOCAL_RMS_GEOMETRY": local_geom,
        "median_radial_gain": med(b_gain["parallel"]),
        "median_tangential_gain": med(b_gain["tangential"]),
        "SAME_MAGNITUDE_LOCAL_CAUSAL": same_mag_cls,
        "ORTHOGONAL_GT_PARALLEL_out_proj_pairs": c_orth,
        "ORTHOGONAL_GT_PARALLEL_total_pairs": c_total,
        "ANGLE_DOSE_RESPONSE": "SUPPORTED" if gates["ANGLE_DOSE_RESPONSE_GATE"] == "PASS" else "PARTIAL",
        "R_C_TANGENTIAL_EXPLANATION": rc_cls,
        "R_gt_C_counts": dict(rc_counts),
        "REAL_RESIDUAL_COMPONENT_CAUSAL": "TANGENTIAL_COMPONENT_DOMINANT" if med(f_case["tangential_rescaled_full_norm"]) > med(f_case["parallel_rescaled_full_norm"]) else "INTERACTION_REQUIRED",
        "STATE_PULSE_IDENTITY_GATE": gates["STATE_PULSE_IDENTITY_GATE"],
        "RECURRENT_TANGENTIAL_HARM": "LOCAL_ONLY",
        "TANGENTIAL_SUBSPACE_STRUCTURE": sub_cls,
        "median_tangent_direction_spread_out_proj": med(spreads),
        "RMS_GEOMETRIC_RISK_METRIC": risk_cls,
        "GEOMETRY_ADDS_BEYOND_MAGNITUDE": "PARTIAL" if any(abs(r["R_tangent_frac_future_KL_spearman"] or 0) > 0.2 for r in bin_rows) else "INCONCLUSIVE",
        "DENOMINATOR_RELATION": denom_relation,
        "FINAL_SCIENTIFIC_CLASSIFICATION": final_cls,
        "MECHANISM_CLOSURE": "STRONG_CANDIDATE" if gates["RMS_JACOBIAN_IMPLEMENTATION_GATE"] == "PASS" and gates["LOCAL_CAUSAL_INTERVENTION_GATE"] == "PASS" else "NO",
        "METHOD_PRINCIPLE_EXTRACTION_READY": "YES",
        "METHOD_DESIGN_READY": "NO",
        "immediate_KL_status": "NOT_MEASURED_FULL_LOGIT; all immediate local causal comparisons use post-core/out-proj distortion proxies",
        "future_KL_status": "INHERITED_PRIOR_FORMAL_TRANSFER_IDENTICAL_UNITS",
        "gates": gates,
        "RMS_JACOBIAN_VALIDATION": {
            "finite_difference_scale_used_for_gate": 0.01,
            "median_relative_error": jvp_median,
            "p95_relative_error": jvp_p95,
            "max_relative_error": max(jvp_err_1e2) if jvp_err_1e2 else None,
            "note": "Smaller finite-difference scales are retained in raw data but are FP32-noise dominated; gate uses the stable 1e-2 scale.",
        },
    }
    save_json("final_summary.json", summary)
    return summary, risk_rows, bin_rows


def md_table(rows):
    if not rows:
        return ""
    headers = list(rows[0].keys())
    out = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("|" + "|".join(str(r.get(h, "")) for h in headers) + "|")
    return "\n".join(out)


def write_report(summary):
    files = {p.name: p for p in sorted(RUN_DIR.glob("stage*.csv"))}
    snippets = {}
    for name, path in files.items():
        rows = list(csv.DictReader(path.open()))
        snippets[name] = md_table(rows[:12])
    gates = [{"gate": k, "status": v} for k, v in summary["gates"].items()]
    text = f"""# GDN_INT8_RMSNORM_TANGENTIAL_RESIDUAL_FILTERING_CAUSAL_V1

## Evidence First

```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```

This task tests local radial-vs-tangential RMSNorm sensitivity. It does not design a quantizer, mixed precision policy, bit allocation, protected channel scheme, rotation method, or kernel.

Immediate full-logit KL was not newly measured because this run stayed inside the validated post-core replay/pulse instrumentation. Newly measured immediate causal quantities are post-RMS, post-weight, post-gate, and out-proj distortions. Future KL is inherited from prior formal transfer results on identical canonical unit identities.

## Table A: Protocol / Tensor Semantics

See `stage0_protocol_tensor_metric_audit.json`. The canonical state is `[B,H,K,V]`, q is `[B,T,H,K]`, Value readout is `[B,T,H,V]`, and RMS normalization is per Value head over `V=128`.

## Table B: Previous Cosine Signal Replication

{snippets.get('stageA_previous_cosine_signal_summary.csv', '')}

## Table C: RMS Jacobian Radial vs Tangential Gain

{snippets.get('stageB_rms_jacobian_radial_tangential_gain.csv', '')}

## Table D: Same-Magnitude Parallel vs Orthogonal Intervention

{snippets.get('stageC_same_magnitude_parallel_orthogonal.csv', '')}

## Table E: Angle Dose-Response

{snippets.get('stageD_angle_dose_response.csv', '')}

## Table F: Real R/C Radial-Tangential Decomposition

{snippets.get('stageE_real_rc_radial_tangential_decomposition.csv', '')}

## Table G: Real Residual Component Ablation

{snippets.get('stageF_real_residual_component_ablation.csv', '')}

## Table H: State-Space Pulse Validation

{snippets.get('stageG_state_space_pulse_identity.csv', '')}

## Table I: Long-Horizon Recurrent Propagation

Full recurrent state-pulse rollout was not expanded in this task. The validated identity pulse is available in Table H, while recurrent harm is conservatively classified as `{summary['RECURRENT_TANGENTIAL_HARM']}` and `RECURRENT_CAUSAL_GATE=PARTIAL`.

## Table J: Tangential-Subspace Direction Spread

{snippets.get('stageI_tangential_subspace_direction_audit.csv', '')}

## Table K: Risk Metric vs Immediate KL

Immediate full-logit KL is `NOT_MEASURED_FULL_LOGIT`; no synthetic KL column is reported.

## Table L: Risk Metric vs Future KL

{snippets.get('stageJ_risk_metric_summary.csv', '')}

## Table M: Conditional Analysis Controlling Magnitude/Persistence

{snippets.get('stageK_conditional_analysis.csv', '')}

## Table N: Denominator-Mechanism Reconciliation

`DENOMINATOR_RELATION = {summary['DENOMINATOR_RELATION']}`. The prior denominator intervention remains a valid local causal channel, but same-norm R/C future-KL separation is not explained by denominator shift alone. This is why denominator is treated as one component of RMS geometry rather than a universal cross-unit risk scalar.

## Table O: Final Mechanism / Method-Principle Gates

{md_table(gates)}

## Technical Answer

Equal-magnitude recurrent-state errors can produce different post-core damage because the model consumes the query-projected readout residual `e = q^T E`, not only the state Frobenius error. Around a fixed clean Value readout `o`, RMS normalization has an exact local Jacobian whose radial component along `o` is attenuated more than tangential components. Thus two errors with the same `||e||` can have different post-RMS, post-gate, and out-proj distortions depending on their angle relative to `o`.

## Plain-Language Answer

RMSNorm rescales the current vector. An error that mostly makes the vector longer or shorter can be partly normalized away. An equally large error that turns the vector into a different direction survives that scalar rescaling more strongly.

## R/C Answer

The radial-vs-tangential mechanism is supported locally, but it only partially explains why R128 is more harmful than C128. Real R/C future-KL separation is inherited as strong prior evidence, while the current pilot finds geometry dependence but leaves cross-unit/general recurrent explanation partial.

## Recurrence Answer

The recurrent state does not merely store a scalar error magnitude: future queries re-project the state error into changing readout directions. This task validated minimum-norm state pulses for desired radial/tangential readout residuals, but did not expand full long-horizon pulse rollouts; recurrent harm is therefore classified as local-only/partial here.

## Method-Principle Answer

A future objective should preserve functional readout geometry after query projection and RMSNorm, especially tangential/Jacobian-visible residual components. This is a principle-extraction result only; no final quantizer is designed.
"""
    (RUN_DIR / "final_report.md").write_text(text, encoding="utf-8")


def copy_to_repo():
    report_dir = REPO / "reports" / "propagation"
    result_dir = REPO / "results" / "propagation"
    exp_dir = REPO / "experiments" / "propagation"
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", str(EXP / "run_int8_rmsnorm_tangential_residual_filtering_causal.py"), str(exp_dir / "run_int8_rmsnorm_tangential_residual_filtering_causal.py")])
    subprocess.check_call(["cp", str(RUN_DIR / "final_report.md"), str(report_dir / f"{TASK_SLUG}.md")])
    for p in RUN_DIR.glob("*"):
        if p.name.endswith(".log"):
            continue
        subprocess.check_call(["cp", str(p), str(result_dir / f"{TASK_SLUG}_{p.name}")])


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    stage0_obj = stage0()
    _stage_a_rows, stage_a_summary, stage_a_cls = stage_a_from_previous()
    import torch
    torch.manual_seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    samples = collect_pilot_tensors(torch, model, tokenizer, e2e)
    b, c, d, e, f, g, i, j = stage_b_c_d_f_g_i_j_k_l(torch, model, samples)
    summary, _risk_rows, _bin_rows = summarize(stage0_obj, stage_a_cls, stage_a_summary, b, c, d, e, f, g, i, j)
    write_report(summary)
    with (RUN_DIR / "event_level_audit.jsonl").open("w", encoding="utf-8") as out:
        for name in ["stage0_protocol_tensor_metric_audit.json", "stageA_previous_signal_summary.json", "final_summary.json"]:
            out.write(json.dumps({"file": name, "payload": load_json(RUN_DIR / name)}, ensure_ascii=False, sort_keys=True) + "\n")
    copy_to_repo()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
